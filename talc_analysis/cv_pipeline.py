"""Classical-CV refinement of a coarse neural-network talc-zone mask."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class CVResult:
    mask: np.ndarray
    confidence: np.ndarray
    component_metrics: list[dict[str, Any]]


def _steps(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pipeline = config.get("pipeline")
    if not isinstance(pipeline, list):
        raise ValueError("cv.pipeline must be a list")
    result: dict[str, dict[str, Any]] = {}
    for item in pipeline:
        if not isinstance(item, dict):
            raise ValueError("Every CV pipeline item must be a mapping")
        if not item.get("enabled", True):
            continue
        name = item.get("type")
        params = item.get("params", {})
        if not isinstance(name, str) or not isinstance(params, dict):
            raise ValueError("CV pipeline items require string type and mapping params")
        result[name] = params
    required = {
        "prepare_image",
        "local_darkness",
        "multiscale_blackhat",
        "normalize_features",
        "confidence_fusion",
        "hysteresis",
        "morphology_cleanup",
        "component_filter",
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(f"CV pipeline is missing required stages: {sorted(missing)}")
    return result


def _robust_normalize(
    feature: np.ndarray,
    valid_mask: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    output = np.zeros(feature.shape, dtype=np.float32)
    finite_valid = valid_mask & np.isfinite(feature)
    values = feature[finite_valid]
    if values.size == 0:
        return output
    maximum = int(params["max_percentile_samples"])
    if values.size > maximum:
        indices = np.linspace(0, values.size - 1, maximum, dtype=np.int64)
        percentile_values = values[indices]
    else:
        percentile_values = values
    low = float(np.percentile(percentile_values, float(params["lower_percentile"])))
    high = float(np.percentile(percentile_values, float(params["upper_percentile"])))
    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or high - low < float(params["min_dynamic_range"])
    ):
        return output
    eps = float(params["eps"])
    output[finite_valid] = np.clip(
        (feature[finite_valid] - low) / (high - low + eps),
        0,
        1,
    )
    return output


def _remove_small(mask: np.ndarray, min_area: int, connectivity: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(bool, copy=False)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=connectivity
    )
    if count <= 1:
        return np.zeros(mask.shape, dtype=bool)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def _fill_holes(mask: np.ndarray, max_area: int, connectivity: int) -> np.ndarray:
    if max_area <= 0:
        return mask.astype(bool, copy=False)
    inverse = ~mask.astype(bool, copy=False)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse.astype(np.uint8), connectivity=connectivity
    )
    touches = np.zeros(count, dtype=bool)
    for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        touches[np.unique(edge)] = True
    fill = (stats[:, cv2.CC_STAT_AREA] <= max_area) & ~touches
    fill[0] = False
    result = mask.astype(bool, copy=True)
    result[fill[labels]] = True
    return result


def _elongation(xs: np.ndarray, ys: np.ndarray, eps: float) -> float:
    if xs.size < 2:
        return 1.0
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / np.float32(max(points.shape[0] - 1, 1))
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(
        np.sqrt(max(float(eigenvalues[-1]), 0.0) / (max(float(eigenvalues[0]), 0.0) + eps))
    )


def _solidity(component: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    points = np.concatenate(contours, axis=0)
    if points.shape[0] < 3:
        return 1.0
    hull = cv2.convexHull(points)
    hull_mask = np.zeros(component.shape, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    hull_area = int(hull_mask.sum())
    return 1.0 if hull_area == 0 else float(min(component.sum() / hull_area, 1.0))


class TalcCVPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.params = _steps(config)

    def run(self, image_rgb: np.ndarray, coarse_mask: np.ndarray) -> CVResult:
        image = np.asarray(image_rgb)
        mask = np.asarray(coarse_mask)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image_rgb must have shape HxWx3")
        if mask.shape != image.shape[:2]:
            raise ValueError("coarse_mask must match image dimensions")
        coarse = mask.astype(bool)

        prepare = self.params["prepare_image"]
        image_float = image.astype(np.float32) / 255.0
        lab = cv2.cvtColor(image_float, cv2.COLOR_RGB2LAB)
        luminance = (lab[..., 0] / 100.0).astype(np.float32)
        valid = (
            np.all(np.isfinite(image_float), axis=2)
            & (luminance > float(prepare["black_clip"]))
            & (luminance < float(prepare["white_clip"]))
            & coarse
        )
        margin = int(prepare.get("border_margin", 0))
        if margin:
            valid[:margin] = False
            valid[-margin:] = False
            valid[:, :margin] = False
            valid[:, -margin:] = False

        darkness_params = self.params["local_darkness"]
        eps = float(darkness_params["eps"])
        background = cv2.GaussianBlur(
            luminance,
            (0, 0),
            sigmaX=float(darkness_params["background_sigma"]),
            borderType=cv2.BORDER_REFLECT,
        )
        local_log_darkness = np.maximum(
            np.log(background + eps) - np.log(luminance + eps), 0
        ).astype(np.float32)
        window = int(darkness_params["local_stats_window"])
        local_mean = cv2.boxFilter(
            luminance, -1, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT
        )
        local_square_mean = cv2.boxFilter(
            luminance * luminance,
            -1,
            (window, window),
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        local_std = np.sqrt(np.maximum(local_square_mean - local_mean**2, 0) + eps)
        local_zscore = np.clip(
            np.maximum((local_mean - luminance) / local_std, 0),
            0,
            float(darkness_params["zscore_clip"]),
        ).astype(np.float32)

        blackhat_params = self.params["multiscale_blackhat"]
        blackhat_max = np.zeros(luminance.shape, dtype=np.float32)
        persistence_count = np.zeros(luminance.shape, dtype=np.uint16)
        kernel_sizes = [int(value) for value in blackhat_params["kernel_sizes"]]
        for size in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            closed = cv2.morphologyEx(luminance, cv2.MORPH_CLOSE, kernel)
            response = np.maximum(closed - luminance, 0).astype(np.float32)
            normalized = _robust_normalize(response, valid, blackhat_params)
            blackhat_max = np.maximum(blackhat_max, normalized)
            persistence_count += (
                normalized >= float(blackhat_params["persistence_threshold"])
            ).astype(np.uint16)
        persistence = persistence_count.astype(np.float32) / len(kernel_sizes)
        persistence[~valid] = 0

        normalize_params = self.params["normalize_features"]
        raw_features = {
            "local_log_darkness": local_log_darkness,
            "local_zscore": local_zscore,
            "blackhat_max": blackhat_max,
        }
        features = {
            f"{name}_norm": _robust_normalize(raw_features[name], valid, normalize_params)
            for name in normalize_params["features"]
        }
        features["blackhat_persistence"] = persistence

        fusion = self.params["confidence_fusion"]
        weights = {key: float(value) for key, value in fusion["weights"].items()}
        weight_sum = sum(weights.values())
        weighted_mean = np.zeros(luminance.shape, dtype=np.float32)
        evidence_count = np.zeros(luminance.shape, dtype=np.uint8)
        for name, weight in weights.items():
            weighted_mean += (weight / weight_sum) * features[name]
            evidence_count += (
                features[name] >= float(fusion["vote_thresholds"][name])
            ).astype(np.uint8)
        primary = np.maximum.reduce(
            [
                features["local_log_darkness_norm"],
                features["local_zscore_norm"],
                features["blackhat_max_norm"],
            ]
        )
        max_weight = float(fusion["max_weight"])
        mean_weight = float(fusion["mean_weight"])
        confidence = (
            max_weight * primary + mean_weight * weighted_mean
        ) / (max_weight + mean_weight)
        confidence = np.clip(confidence, 0, 1).astype(np.float32)
        confidence[~valid] = 0
        primary[~valid] = 0
        evidence_count[~valid] = 0

        hysteresis = self.params["hysteresis"]
        seeds = (
            (confidence >= float(hysteresis["seed_threshold"]))
            & (evidence_count >= int(hysteresis["seed_min_evidence"]))
        )
        strong_threshold = hysteresis.get("strong_response_threshold")
        if strong_threshold is not None:
            seeds |= primary >= float(strong_threshold)
        seeds &= valid
        connectivity = int(hysteresis["connectivity"])
        seeds = _remove_small(seeds, int(hysteresis["min_seed_area"]), connectivity)
        grow = (
            (confidence >= float(hysteresis["grow_threshold"]))
            & (evidence_count >= int(hysteresis["grow_min_evidence"]))
            & valid
        )
        if np.any(seeds):
            count, grow_labels = cv2.connectedComponents(
                grow.astype(np.uint8), connectivity=connectivity
            )
            accepted = np.unique(grow_labels[seeds])
            accepted = accepted[accepted != 0]
            lookup = np.zeros(count, dtype=bool)
            lookup[accepted] = True
            candidates = lookup[grow_labels]
        else:
            candidates = np.zeros(valid.shape, dtype=bool)

        cleanup = self.params["morphology_cleanup"]
        cleaned = candidates
        cleanup_connectivity = int(cleanup["connectivity"])
        for operation in cleanup["operations"]:
            if not operation.get("enabled", True):
                continue
            operation_type = operation["type"]
            if operation_type == "remove_small_objects":
                cleaned = _remove_small(
                    cleaned, int(operation["min_area"]), cleanup_connectivity
                )
            elif operation_type == "fill_small_holes":
                cleaned = _fill_holes(
                    cleaned, int(operation["max_area"]), cleanup_connectivity
                )
            elif operation_type in {"opening", "closing"}:
                size = int(operation["kernel_size"])
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                code = cv2.MORPH_OPEN if operation_type == "opening" else cv2.MORPH_CLOSE
                cleaned = cv2.morphologyEx(
                    cleaned.astype(np.uint8), code, kernel
                ).astype(bool)
            else:
                raise ValueError(f"Unknown morphology operation {operation_type!r}")
        cleaned &= valid

        final_mask, component_metrics = self._filter_components(
            cleaned, valid, luminance, confidence, persistence
        )
        final_mask &= coarse
        return CVResult(
            mask=final_mask.astype(np.uint8),
            confidence=confidence,
            component_metrics=component_metrics,
        )

    def _filter_components(
        self,
        mask: np.ndarray,
        valid: np.ndarray,
        luminance: np.ndarray,
        confidence: np.ndarray,
        persistence: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        params = self.params["component_filter"]
        connectivity = int(params["connectivity"])
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=connectivity
        )
        accepted_labels: list[int] = []
        metrics_list: list[dict[str, Any]] = []
        eps = float(params["eps"])
        log_luminance = np.log(luminance + eps)
        radius = int(params["ring_radius"])
        ring_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        for label_id in range(1, count):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            local_labels = labels[y : y + height, x : x + width]
            local_component = local_labels == label_id
            ys_local, xs_local = np.where(local_component)
            ys = ys_local + y
            xs = xs_local + x
            area = int(xs.size)

            roi_x0 = max(0, x - radius)
            roi_y0 = max(0, y - radius)
            roi_x1 = min(mask.shape[1], x + width + radius)
            roi_y1 = min(mask.shape[0], y + height + radius)
            roi_labels = labels[roi_y0:roi_y1, roi_x0:roi_x1]
            roi_component = roi_labels == label_id
            dilated = cv2.dilate(
                roi_component.astype(np.uint8), ring_kernel
            ).astype(bool)
            ring = (
                dilated
                & ~roi_component
                & valid[roi_y0:roi_y1, roi_x0:roi_x1]
                & ~mask[roi_y0:roi_y1, roi_x0:roi_x1]
            )
            ring_pixels = int(ring.sum())
            ring_applicable = ring_pixels >= int(params["min_ring_pixels"])
            roi_log_luminance = log_luminance[roi_y0:roi_y1, roi_x0:roi_x1]
            ring_contrast = (
                float(
                    np.median(roi_log_luminance[ring])
                    - np.median(roi_log_luminance[roi_component])
                )
                if ring_applicable
                else None
            )
            local_confidence = confidence[y : y + height, x : x + width]
            local_persistence = persistence[y : y + height, x : x + width]
            metrics: dict[str, Any] = {
                "label": label_id,
                "area": area,
                "bbox": [x, y, width, height],
                "centroid": [float(xs.mean()), float(ys.mean())],
                "elongation": _elongation(xs, ys, eps),
                "solidity": _solidity(local_component),
                "ring_contrast": ring_contrast,
                "ring_pixels": ring_pixels,
                "ring_filter_applicable": ring_applicable,
                "median_confidence": float(
                    np.median(local_confidence[local_component])
                ),
                "median_persistence": float(
                    np.median(local_persistence[local_component])
                ),
                "touches_border": bool(
                    x == 0
                    or y == 0
                    or x + width == mask.shape[1]
                    or y + height == mask.shape[0]
                ),
            }
            shape_passes = (
                area < int(params["shape_filter_min_area"])
                or (
                    metrics["elongation"] <= float(params["max_elongation"])
                    and metrics["solidity"] >= float(params["min_solidity"])
                )
            )
            if ring_applicable:
                ring_passes = ring_contrast is not None and ring_contrast >= float(
                    params["min_ring_contrast"]
                )
            else:
                ring_passes = params["insufficient_ring_policy"] == "skip_filter"
            accepted = bool(
                int(params["min_area"]) <= area <= int(params["max_area"])
                and shape_passes
                and ring_passes
                and metrics["median_confidence"]
                >= float(params["min_median_confidence"])
                and metrics["median_persistence"] >= float(params["min_persistence"])
                and not (
                    bool(params["reject_border_touching"])
                    and metrics["touches_border"]
                )
            )
            metrics["accepted"] = accepted
            metrics_list.append(metrics)
            if accepted:
                accepted_labels.append(label_id)
        lookup = np.zeros(count, dtype=bool)
        lookup[np.asarray(accepted_labels, dtype=np.int32)] = True
        return lookup[labels], metrics_list
