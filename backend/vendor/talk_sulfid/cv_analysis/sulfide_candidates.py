"""Fast CV segmentation of bright sulfide-like inclusions with optional MobileSAM."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("sulfide_candidates.yaml")
DEFAULT_OVERLAY_COLOR_RGB = (255, 180, 40)
DEFAULT_OVERLAY_ALPHA = 0.65


def segment_sulfides(
    image: np.ndarray,
    config: dict[str, Any] | str | Path | None = None,
    sam_refiner: "MobileSamRefiner | None" = None,
) -> np.ndarray:
    """Return a binary uint8 mask of sulfide-like inclusions with values 0/255."""

    cfg = load_sulfide_config(config)
    cv_mask = _segment_sulfides_cv(image, cfg)
    if sam_refiner is None:
        return cv_mask

    image_rgb = _to_uint8_rgb(image, cfg["preprocessing"].get("input_color", "rgb"))
    sam_cfg = cfg.get("sam", {})
    return sam_refiner.refine(
        image_rgb=image_rgb,
        cv_mask=cv_mask,
        min_area=int(sam_cfg.get("min_area", 100)),
        max_components=int(sam_cfg.get("max_components", 20)),
        box_padding=int(sam_cfg.get("box_padding", 10)),
        box_padding_ratio=float(sam_cfg.get("box_padding_ratio", 0.0)),
        max_positive_points=int(sam_cfg.get("max_positive_points", 1)),
        min_coverage=float(sam_cfg.get("min_coverage", 0.70)),
        min_area_ratio=float(sam_cfg.get("min_area_ratio", 0.60)),
        max_area_ratio=float(sam_cfg.get("max_area_ratio", 2.50)),
        prefer_larger_mask=bool(sam_cfg.get("prefer_larger_mask", False)),
    )


def load_sulfide_config(
    config: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    if config is None:
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    if isinstance(config, (str, Path)):
        with Path(config).open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("config must be a mapping, path, or None")
    return config


def make_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any] | str | Path | None = None,
) -> np.ndarray:
    """Overlay a semi-transparent sulfide mask on an image and return RGB uint8."""

    cfg = load_sulfide_config(config)
    image_rgb = _to_uint8_rgb(image, cfg["preprocessing"].get("input_color", "rgb"))
    overlay_cfg = cfg.get("overlay", {})
    color = np.asarray(
        overlay_cfg.get("color_rgb", DEFAULT_OVERLAY_COLOR_RGB), dtype=np.float32
    )
    alpha = float(overlay_cfg.get("alpha", DEFAULT_OVERLAY_ALPHA))
    mask_bool = _validate_mask(mask, image_rgb.shape[:2])

    output = image_rgb.astype(np.float32, copy=True)
    output[mask_bool] = (1.0 - alpha) * output[mask_bool] + alpha * color
    return np.clip(output, 0, 255).astype(np.uint8)


class MobileSamRefiner:
    """Optional MobileSAM boundary refiner for existing CV components."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
    ) -> None:
        try:
            import torch
            from mobile_sam import SamPredictor, sam_model_registry
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError(f"MobileSAM dependencies are not available: {exc}") from exc

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"MobileSAM checkpoint not found: {checkpoint_path}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for MobileSAM")

        model = sam_model_registry["vit_t"](checkpoint=str(checkpoint_path))
        model.to(device)
        model.eval()

        self.device = device
        self.torch = torch
        self.predictor = SamPredictor(model)
        self.last_timings: dict[str, float] = {}
        self.last_stats: dict[str, int] = {}

    def refine(
        self,
        image_rgb: np.ndarray,
        cv_mask: np.ndarray,
        min_area: int = 100,
        max_components: int = 20,
        box_padding: int = 10,
        box_padding_ratio: float = 0.0,
        max_positive_points: int = 1,
        min_coverage: float = 0.70,
        min_area_ratio: float = 0.60,
        max_area_ratio: float = 2.50,
        prefer_larger_mask: bool = False,
    ) -> np.ndarray:
        image_rgb = _to_uint8_rgb(image_rgb, "rgb")
        cv_mask_bool = _binary_mask(cv_mask, image_rgb.shape[:2])
        result = cv_mask_bool.copy()

        total_start = time.perf_counter()
        set_image_start = time.perf_counter()
        with self.torch.inference_mode():
            self.predictor.set_image(image_rgb)
        set_image_ms = _elapsed_ms(set_image_start)

        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            cv_mask_bool.astype(np.uint8),
            connectivity=8,
        )
        component_ids = _largest_component_ids(
            stats, labels_count, min_area, max_components
        )

        refined_count = 0
        fallback_count = 0
        prompt_start = time.perf_counter()
        height, width = cv_mask_bool.shape
        with self.torch.inference_mode():
            for label_id in component_ids:
                component = labels == label_id
                component_area = int(stats[label_id, cv2.CC_STAT_AREA])
                points = _positive_points_from_component(
                    component,
                    stats[label_id],
                    max_positive_points,
                )
                if points.size == 0:
                    fallback_count += 1
                    continue
                accepted_mask = self._predict_component_mask(
                    points=points,
                    box=_padded_component_box(
                        stats[label_id],
                        width,
                        height,
                        box_padding,
                        box_padding_ratio,
                    ),
                    component=component,
                    component_area=component_area,
                    min_coverage=min_coverage,
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio,
                    prefer_larger_mask=prefer_larger_mask,
                )
                if accepted_mask is None:
                    fallback_count += 1
                    continue
                result[component] = False
                result[accepted_mask] = True
                refined_count += 1
        prompts_ms = _elapsed_ms(prompt_start)

        self.last_timings = {
            "set_image_ms": set_image_ms,
            "prompts_ms": prompts_ms,
            "total_refine_ms": _elapsed_ms(total_start),
        }
        self.last_stats = {
            "components_considered": len(component_ids),
            "components_refined": refined_count,
            "components_fallback": fallback_count,
        }
        return (result.astype(np.uint8) * 255).astype(np.uint8, copy=False)

    def _predict_component_mask(
        self,
        points: np.ndarray,
        box: tuple[int, int, int, int],
        component: np.ndarray,
        component_area: int,
        min_coverage: float,
        min_area_ratio: float,
        max_area_ratio: float,
        prefer_larger_mask: bool,
    ) -> np.ndarray | None:
        masks, scores, _ = self.predictor.predict(
            point_coords=points.astype(np.float32, copy=False),
            point_labels=np.ones(points.shape[0], dtype=np.int32),
            box=np.array(box, dtype=np.float32),
            multimask_output=True,
        )
        masks_bool = np.asarray(masks).astype(bool)
        scores_array = np.asarray(scores, dtype=np.float32)
        if masks_bool.ndim == 2:
            masks_bool = masks_bool[None, ...]

        best_index = None
        best_score = -float("inf")
        best_values: dict[str, float] = {}
        for index, sam_mask in enumerate(masks_bool):
            sam_area = int(np.count_nonzero(sam_mask))
            if sam_area <= 0:
                continue
            intersection = int(np.count_nonzero(sam_mask & component))
            coverage = intersection / max(component_area, 1)
            area_ratio = sam_area / max(component_area, 1)
            sam_score = float(scores_array[index]) if index < scores_array.size else 0.0
            if prefer_larger_mask:
                custom_score = (
                    0.50 * coverage
                    + 0.30 * sam_score
                    + 0.20 * min(area_ratio / 3.0, 1.0)
                )
            else:
                custom_score = 0.7 * coverage + 0.3 * sam_score
            if custom_score > best_score:
                best_score = custom_score
                best_index = index
                best_values = {
                    "coverage": coverage,
                    "area_ratio": area_ratio,
                }

        if best_index is None:
            return None

        best_mask = masks_bool[best_index]
        if best_values["coverage"] < min_coverage:
            return None
        if not (min_area_ratio <= best_values["area_ratio"] <= max_area_ratio):
            return None
        point_x = points[:, 0].astype(np.int64)
        point_y = points[:, 1].astype(np.int64)
        if not np.all(best_mask[point_y, point_x]):
            return None
        return best_mask


def _segment_sulfides_cv(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    image_rgb = _to_uint8_rgb(image, cfg["preprocessing"].get("input_color", "rgb"))
    lightness = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[..., 0]

    blur_kernel = int(cfg["preprocessing"].get("blur_kernel", 3))
    if blur_kernel > 1:
        kernel_size = _odd_kernel_size(blur_kernel)
        lightness = cv2.GaussianBlur(lightness, (kernel_size, kernel_size), 0)

    otsu_threshold, _ = cv2.threshold(
        lightness,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    threshold = np.clip(
        float(otsu_threshold) + float(cfg["thresholds"].get("otsu_offset", -5)),
        0,
        255,
    )

    mask = lightness >= threshold
    mask = _clean_mask(mask, cfg["morphology"])
    return (mask.astype(np.uint8) * 255).astype(np.uint8, copy=False)


def _to_uint8_rgb(image: np.ndarray, input_color: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")

    if array.dtype == np.uint8:
        image_uint8 = array.copy()
    elif array.dtype == np.uint16:
        image_uint8 = np.rint(array.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    elif np.issubdtype(array.dtype, np.floating):
        finite = np.isfinite(array)
        safe = np.where(finite, array, 0).astype(np.float32, copy=False)
        min_value = float(np.min(safe)) if safe.size else 0.0
        max_value = float(np.max(safe)) if safe.size else 0.0
        if min_value < 0:
            raise ValueError("floating image values must be >= 0")
        if max_value <= 1.0:
            safe = safe * 255.0
        elif max_value > 255.0:
            raise ValueError("floating image values must be in [0, 1] or [0, 255]")
        image_uint8 = np.rint(safe).clip(0, 255).astype(np.uint8)
    else:
        raise TypeError("image dtype must be uint8, uint16, float32, or float64")

    if input_color == "rgb":
        return image_uint8
    if input_color == "bgr":
        return image_uint8[..., ::-1].copy()
    raise ValueError("preprocessing.input_color must be 'rgb' or 'bgr'")


def _clean_mask(mask: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    open_kernel = int(cfg.get("open_kernel", 0))
    if open_kernel > 0:
        result = _morphology(result, cv2.MORPH_OPEN, open_kernel)

    close_kernel = int(cfg.get("close_kernel", 0))
    if close_kernel > 0:
        result = _morphology(result, cv2.MORPH_CLOSE, close_kernel)

    result = _fill_small_holes(
        result,
        int(cfg.get("fill_holes_area", 0)),
        int(cfg.get("connectivity", 8)),
    )
    result = _remove_small_components(
        result,
        int(cfg.get("min_component_area", 1)),
        int(cfg.get("connectivity", 8)),
    )
    return result


def _morphology(mask: np.ndarray, operation: int, kernel_size: int) -> np.ndarray:
    kernel_size = _odd_kernel_size(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel).astype(bool)


def _remove_small_components(mask: np.ndarray, min_area: int, connectivity: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(bool, copy=False)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=connectivity,
    )
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    if count:
        keep[0] = False
    return keep[labels]


def _fill_small_holes(mask: np.ndarray, max_area: int, connectivity: int) -> np.ndarray:
    if max_area <= 0:
        return mask.astype(bool, copy=False)
    inverse = ~mask.astype(bool, copy=False)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse.astype(np.uint8),
        connectivity=connectivity,
    )
    touches_border = np.zeros(count, dtype=bool)
    touches_border[np.unique(labels[0, :])] = True
    touches_border[np.unique(labels[-1, :])] = True
    touches_border[np.unique(labels[:, 0])] = True
    touches_border[np.unique(labels[:, -1])] = True
    fill = (stats[:, cv2.CC_STAT_AREA] <= max_area) & ~touches_border
    if count:
        fill[0] = False
    result = mask.astype(bool, copy=True)
    result[fill[labels]] = True
    return result


def _binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask_array = np.asarray(mask)
    if mask_array.shape != shape:
        raise ValueError("cv_mask shape must match image height and width")
    return mask_array.astype(np.uint8) > 0


def _largest_component_ids(
    stats: np.ndarray,
    labels_count: int,
    min_area: int,
    max_components: int,
) -> list[int]:
    component_ids = [
        label_id
        for label_id in range(1, labels_count)
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= min_area
    ]
    component_ids.sort(
        key=lambda label_id: int(stats[label_id, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    return component_ids[: max(max_components, 0)]


def _padded_component_box(
    stat: np.ndarray,
    width: int,
    height: int,
    base_padding: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x = int(stat[cv2.CC_STAT_LEFT])
    y = int(stat[cv2.CC_STAT_TOP])
    w = int(stat[cv2.CC_STAT_WIDTH])
    h = int(stat[cv2.CC_STAT_HEIGHT])
    component_size = max(w, h)
    padding = max(
        int(base_padding),
        int(component_size * max(float(padding_ratio), 0.0)),
    )
    return (
        max(0, x - padding),
        max(0, y - padding),
        min(width - 1, x + w - 1 + padding),
        min(height - 1, y + h - 1 + padding),
    )


def _positive_points_from_component(
    component: np.ndarray,
    stat: np.ndarray,
    max_points: int,
) -> np.ndarray:
    if max_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    x = int(stat[cv2.CC_STAT_LEFT])
    y = int(stat[cv2.CC_STAT_TOP])
    w = int(stat[cv2.CC_STAT_WIDTH])
    h = int(stat[cv2.CC_STAT_HEIGHT])
    roi = component[y : y + h, x : x + w].astype(np.uint8)
    distance = cv2.distanceTransform(roi, cv2.DIST_L2, 5)

    points: list[list[int]] = []
    for _ in range(max_points):
        _, max_value, _, max_location = cv2.minMaxLoc(distance)
        if max_value <= 0:
            break
        point_x, point_y = int(max_location[0]), int(max_location[1])
        points.append([x + point_x, y + point_y])
        radius = max(5, int(max_value * 1.5))
        cv2.circle(distance, (point_x, point_y), radius, 0, -1)

    if not points:
        ys, xs = np.where(roi > 0)
        if ys.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        points.append(
            [
                x + int(round(float(xs.mean()))),
                y + int(round(float(ys.mean()))),
            ]
        )
    return np.asarray(points, dtype=np.float32)


def _validate_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask_array = np.asarray(mask)
    if mask_array.shape != shape:
        raise ValueError("mask shape must match image height and width")
    return mask_array.astype(bool)


def _odd_kernel_size(value: int) -> int:
    if value <= 0:
        raise ValueError("kernel/window size must be > 0")
    return value if value % 2 == 1 else value + 1


def _elapsed_ms(start: float) -> float:
    return 1000.0 * (time.perf_counter() - start)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "MobileSamRefiner",
    "load_sulfide_config",
    "make_overlay",
    "segment_sulfides",
]
