"""Generate visual photometric robustness checks for TalcCVPipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv_analysis.post_segformer import TalcCVPipeline  # noqa: E402


@dataclass(frozen=True)
class TransformCase:
    name: str
    steps: list[dict[str, Any]]


def main() -> None:
    args = _parse_args()
    config_path = _resolve_repo_path(args.config)
    config = _load_yaml(config_path)

    output_dir = _resolve_repo_path(args.output_dir or config["output_dir"])
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _unique_run_dir(output_dir / _safe_slug(run_name))
    run_dir.mkdir(parents=True, exist_ok=False)

    pipeline_config_path = _resolve_repo_path(config["pipeline_config"])
    pipeline = TalcCVPipeline.from_yaml(pipeline_config_path)
    samples = _load_samples(config)
    cases = _build_cases(config["tests"])

    if args.case_filter:
        filters = [value.lower() for value in args.case_filter]
        cases = [
            case
            for case in cases
            if any(value in case.name.lower() for value in filters)
        ]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    shutil.copy2(config_path, run_dir / "photometric_tests.yaml")
    shutil.copy2(pipeline_config_path, run_dir / "post_segformer.pipeline.yaml")

    visual_config = config.get("visualization", {})
    seed = int(config.get("random_seed", 0))
    records: list[dict[str, Any]] = []

    for sample in samples:
        image_rgb = _read_rgb(sample["image_path"])
        segformer_mask = _load_yolo_segmentation_mask(
            sample["label_path"],
            image_rgb.shape[:2],
        )

        for case in cases:
            case_rng = _case_rng(seed, sample["image_path"].name, case.name)
            transformed = _apply_steps(
                _to_float01(image_rgb),
                case.steps,
                case_rng,
            )
            transformed_uint8 = _to_uint8(transformed)
            result_mask = pipeline(transformed_uint8, segformer_mask)

            case_dir = run_dir / sample["image_path"].stem / case.name
            case_dir.mkdir(parents=True, exist_ok=True)

            segformer_overlay = _overlay_mask(
                transformed_uint8,
                segformer_mask,
                visual_config.get("segformer_color_rgb", [0, 170, 255]),
                float(visual_config.get("alpha", 0.55)),
            )
            result_overlay = _overlay_mask(
                transformed_uint8,
                result_mask,
                visual_config.get("result_color_rgb", [255, 0, 0]),
                float(visual_config.get("alpha", 0.55)),
            )
            triptych = _triptych(
                [
                    transformed_uint8,
                    segformer_overlay,
                    result_overlay,
                ],
                [
                    "transformed",
                    "segformer area",
                    "filtered result",
                ],
                bool(visual_config.get("draw_triptych_labels", True)),
            )

            _save_rgb(case_dir / "01_original.png", image_rgb)
            _save_rgb(case_dir / "02_transformed.png", transformed_uint8)
            _save_rgb(case_dir / "03_segformer_overlay.png", segformer_overlay)
            _save_rgb(case_dir / "04_result_filtering.png", result_overlay)
            _save_rgb(case_dir / "05_triptych.png", triptych)

            records.append(
                {
                    "sample": sample["image_path"].name,
                    "case": case.name,
                    "steps": case.steps,
                    "segformer_pixels": int(np.count_nonzero(segformer_mask)),
                    "result_pixels": int(np.count_nonzero(result_mask)),
                    "output_dir": str(case_dir.relative_to(run_dir)),
                }
            )

    manifest = {
        "config": str(config_path.relative_to(REPO_ROOT)),
        "pipeline_config": str(pipeline_config_path.relative_to(REPO_ROOT)),
        "sample_count": len(samples),
        "case_count": len(cases),
        "records": records,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(records)} test cases to {run_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run photometric robustness visual checks.",
    )
    parser.add_argument(
        "--config",
        default="test/photometric_tests.yaml",
        help="Path to photometric test YAML.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory from config.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional output run folder name.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Limit number of image/label pairs for smoke runs.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Limit number of transform cases for smoke runs.",
    )
    parser.add_argument(
        "--case-filter",
        action="append",
        help="Run only cases whose name contains this substring.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def _resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return REPO_ROOT / value


def _unique_run_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}-{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique output directory near {path}")


def _load_samples(config: dict[str, Any]) -> list[dict[str, Path]]:
    input_config = config["inputs"]
    image_dir = _resolve_repo_path(input_config["image_dir"])
    label_dir = _resolve_repo_path(input_config["label_dir"])
    image_paths = sorted(image_dir.glob(input_config.get("image_glob", "*.JPG")))

    samples = []
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            samples.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                }
            )

    limit = input_config.get("sample_limit")
    if limit is not None:
        samples = samples[: int(limit)]

    if not samples:
        raise FileNotFoundError(
            f"No image/label pairs found in {image_dir} and {label_dir}"
        )

    mask_source = config.get("segformer_mask", {}).get("source", "yolo_labels")
    if mask_source != "yolo_labels":
        raise ValueError(
            "Only segformer_mask.source: yolo_labels is currently supported"
        )

    return samples


def _read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _load_yolo_segmentation_mask(
    label_path: Path,
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.strip().split()
            if not values:
                continue
            coords = np.asarray(values[1:], dtype=np.float32)
            if coords.size < 6 or coords.size % 2 != 0:
                raise ValueError(f"Bad polygon at {label_path}:{line_number}")

            points = coords.reshape(-1, 2)
            points[:, 0] *= width - 1
            points[:, 1] *= height - 1
            polygon = np.rint(points).astype(np.int32)
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [polygon], 1)
    return mask


def _build_cases(test_config: dict[str, Any]) -> list[TransformCase]:
    cases: list[TransformCase] = []

    for value in test_config.get("exposure_gain", []):
        cases.append(_case("exposure_gain", value, {"type": "exposure_gain", "gain": value}))

    for value in test_config.get("brightness_shift", []):
        cases.append(_case("brightness_shift", value, {"type": "brightness_shift", "shift": value}))

    for value in test_config.get("contrast", []):
        cases.append(_case("contrast", value, {"type": "contrast", "factor": value}))

    for value in test_config.get("gamma", []):
        cases.append(_case("gamma", value, {"type": "gamma", "gamma": value}))

    for item in test_config.get("white_balance", []):
        name = item.get("name") or f"gains_{_slug_value(item['gains'])}"
        cases.append(TransformCase(f"white_balance_{_safe_slug(name)}", [{"type": "white_balance", "gains": item["gains"]}]))

    for value in test_config.get("saturation", []):
        cases.append(_case("saturation", value, {"type": "saturation", "factor": value}))

    for value in test_config.get("hue_shift_degrees", []):
        cases.append(_case("hue_shift", value, {"type": "hue_shift", "degrees": value}))

    gradient_config = test_config.get("illumination_gradient")
    if gradient_config:
        start_gain, end_gain = gradient_config["gains"]
        for direction in gradient_config.get("directions", ["left_to_right"]):
            cases.append(
                TransformCase(
                    f"illumination_gradient_{_safe_slug(direction)}_{_slug_value(start_gain)}_{_slug_value(end_gain)}",
                    [
                        {
                            "type": "illumination_gradient",
                            "start_gain": start_gain,
                            "end_gain": end_gain,
                            "direction": direction,
                        }
                    ],
                )
            )

    for value in test_config.get("vignette_edge_gain", []):
        cases.append(_case("vignette", value, {"type": "vignette", "edge_gain": value}))

    for item in test_config.get("local_shadow", []):
        cases.append(
            TransformCase(
                f"local_shadow_{_safe_slug(item['name'])}",
                [dict(item, type="local_shadow")],
            )
        )

    for item in test_config.get("local_highlight", []):
        cases.append(
            TransformCase(
                f"local_highlight_{_safe_slug(item['name'])}",
                [dict(item, type="local_highlight")],
            )
        )

    for value in test_config.get("gaussian_noise_sigma", []):
        cases.append(_case("gaussian_noise", value, {"type": "gaussian_noise", "sigma": value}))

    for value in test_config.get("poisson_peak", []):
        cases.append(_case("poisson_noise", value, {"type": "poisson_noise", "peak": value}))

    for value in test_config.get("gaussian_blur_sigma", []):
        cases.append(_case("gaussian_blur", value, {"type": "gaussian_blur", "sigma": value}))

    for value in test_config.get("defocus_blur_kernel", []):
        cases.append(_case("defocus_blur", value, {"type": "defocus_blur", "kernel_size": value}))

    for value in test_config.get("jpeg_quality", []):
        cases.append(_case("jpeg_quality", value, {"type": "jpeg_compression", "quality": value}))

    for value in test_config.get("quantization_bits", []):
        cases.append(_case("quantization", value, {"type": "quantization", "bits": value}))

    for value in test_config.get("sharpening_amount", []):
        cases.append(_case("sharpening", value, {"type": "sharpening", "amount": value}))

    for item in test_config.get("combinations", []):
        cases.append(
            TransformCase(
                f"combo_{_safe_slug(item['name'])}",
                item["steps"],
            )
        )

    return cases


def _case(prefix: str, value: Any, step: dict[str, Any]) -> TransformCase:
    return TransformCase(f"{prefix}_{_slug_value(value)}", [step])


def _apply_steps(
    image: np.ndarray,
    steps: list[dict[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    result = image.astype(np.float32, copy=True)
    for step in steps:
        result = _apply_step(result, step, rng)
    return np.clip(result, 0, 1).astype(np.float32, copy=False)


def _apply_step(
    image: np.ndarray,
    step: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    transform_type = step["type"]
    if transform_type == "exposure_gain":
        return image * np.float32(step["gain"])
    if transform_type == "brightness_shift":
        return image + np.float32(step["shift"])
    if transform_type == "contrast":
        mean = image.mean(axis=(0, 1), keepdims=True)
        return mean + np.float32(step["factor"]) * (image - mean)
    if transform_type == "gamma":
        return np.power(np.clip(image, 0, 1), np.float32(step["gamma"]))
    if transform_type == "white_balance":
        gains = np.asarray(step["gains"], dtype=np.float32).reshape(1, 1, 3)
        return image * gains
    if transform_type == "saturation":
        hsv = cv2.cvtColor(np.clip(image, 0, 1), cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * np.float32(step["factor"]), 0, 1)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if transform_type == "hue_shift":
        hsv = cv2.cvtColor(np.clip(image, 0, 1), cv2.COLOR_RGB2HSV)
        hsv[..., 0] = np.mod(hsv[..., 0] + np.float32(step["degrees"]), 360.0)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if transform_type == "illumination_gradient":
        return image * _illumination_gradient(image.shape[:2], step)[..., None]
    if transform_type == "vignette":
        return image * _vignette(image.shape[:2], float(step["edge_gain"]))[..., None]
    if transform_type == "local_shadow":
        mask = _local_ellipse_mask(image.shape[:2], step)
        return image * (1.0 - (1.0 - np.float32(step["multiplier"])) * mask[..., None])
    if transform_type == "local_highlight":
        mask = _local_ellipse_mask(image.shape[:2], step)
        return image * (1.0 + (np.float32(step["multiplier"]) - 1.0) * mask[..., None])
    if transform_type == "gaussian_noise":
        noise = rng.normal(0.0, float(step["sigma"]), size=image.shape).astype(np.float32)
        return image + noise
    if transform_type == "poisson_noise":
        peak = np.float32(step["peak"])
        return rng.poisson(np.clip(image, 0, 1) * peak).astype(np.float32) / peak
    if transform_type == "gaussian_blur":
        sigma = float(step["sigma"])
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    if transform_type == "defocus_blur":
        kernel = _disk_kernel(int(step["kernel_size"]))
        return cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)
    if transform_type == "jpeg_compression":
        return _jpeg_roundtrip(image, int(step["quality"]))
    if transform_type == "quantization":
        levels = np.float32((1 << int(step["bits"])) - 1)
        return np.round(np.clip(image, 0, 1) * levels) / levels
    if transform_type == "sharpening":
        sigma = float(step.get("sigma", 1.0))
        amount = np.float32(step["amount"])
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
        return image + amount * (image - blurred)

    raise ValueError(f"Unknown transform type: {transform_type!r}")


def _illumination_gradient(
    shape: tuple[int, int],
    step: dict[str, Any],
) -> np.ndarray:
    height, width = shape
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    direction = step.get("direction", "left_to_right")
    if direction == "left_to_right":
        t = np.broadcast_to(x, (height, width))
    elif direction == "right_to_left":
        t = np.broadcast_to(1.0 - x, (height, width))
    elif direction == "top_to_bottom":
        t = np.broadcast_to(y, (height, width))
    elif direction == "bottom_to_top":
        t = np.broadcast_to(1.0 - y, (height, width))
    elif direction == "diagonal":
        t = (x + y) / 2.0
    elif direction == "anti_diagonal":
        t = ((1.0 - x) + y) / 2.0
    else:
        raise ValueError(f"Unknown illumination gradient direction: {direction!r}")

    start = np.float32(step["start_gain"])
    end = np.float32(step["end_gain"])
    return start + (end - start) * t.astype(np.float32)


def _vignette(shape: tuple[int, int], edge_gain: float) -> np.ndarray:
    height, width = shape
    y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    radius = np.sqrt(x * x + y * y)
    radius /= np.float32(radius.max())
    return 1.0 - (1.0 - np.float32(edge_gain)) * np.clip(radius, 0, 1) ** 2


def _local_ellipse_mask(shape: tuple[int, int], step: dict[str, Any]) -> np.ndarray:
    height, width = shape
    center_x = int(round(float(step["center"][0]) * (width - 1)))
    center_y = int(round(float(step["center"][1]) * (height - 1)))
    axis_x = max(1, int(round(float(step["axes"][0]) * width)))
    axis_y = max(1, int(round(float(step["axes"][1]) * height)))
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(
        mask,
        (center_x, center_y),
        (axis_x, axis_y),
        float(step.get("angle_degrees", 0.0)),
        0,
        360,
        1.0,
        -1,
    )
    sigma = float(step.get("blur_sigma", 0.0))
    if sigma > 0:
        mask = cv2.GaussianBlur(
            mask,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
    max_value = float(mask.max())
    if max_value > 0:
        mask /= np.float32(max_value)
    return np.clip(mask, 0, 1)


def _disk_kernel(kernel_size: int) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("defocus_blur kernel_size must be a positive odd integer")
    radius = kernel_size // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (xx * xx + yy * yy <= radius * radius).astype(np.float32)
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0:
        raise ValueError("empty defocus kernel")
    return kernel / np.float32(kernel_sum)


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    image_bgr = cv2.cvtColor(_to_uint8(image), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return _to_float01(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


def _overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: list[int],
    alpha: float,
) -> np.ndarray:
    base = _to_uint8(image_rgb)
    output = base.astype(np.float32)
    mask_bool = mask.astype(bool)
    color = np.asarray(color_rgb, dtype=np.float32)
    output[mask_bool] = (1.0 - alpha) * output[mask_bool] + alpha * color
    return np.clip(output, 0, 255).astype(np.uint8)


def _triptych(
    images: list[np.ndarray],
    labels: list[str],
    draw_labels: bool,
) -> np.ndarray:
    prepared = [_to_uint8(image) for image in images]
    if draw_labels:
        prepared = [
            _draw_label(image.copy(), label)
            for image, label in zip(prepared, labels)
        ]
    return np.concatenate(prepared, axis=1)


def _draw_label(image: np.ndarray, label: str) -> np.ndarray:
    cv2.rectangle(image, (14, 14), (360, 58), (0, 0, 0), thickness=-1)
    cv2.putText(
        image,
        label,
        (28, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return image


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(_to_uint8(image_rgb), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Could not write {path}")


def _to_float01(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        return np.clip(array.astype(np.float32), 0, 1)
    return array.astype(np.float32) / np.float32(255.0)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    return np.clip(np.rint(array.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)


def _case_rng(seed: int, sample_name: str, case_name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{sample_name}:{case_name}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(value)


def _safe_slug(value: Any) -> str:
    text = str(value).strip().lower()
    replacements = {
        " ": "_",
        ".": "p",
        "-": "m",
        "+": "p",
        "/": "_",
        "\\": "_",
        ":": "_",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return "".join(char for char in text if char.isalnum() or char == "_")


def _slug_value(value: Any) -> str:
    if isinstance(value, float):
        return _safe_slug(f"{value:.3g}")
    if isinstance(value, list):
        return "_".join(_slug_value(item) for item in value)
    return _safe_slug(value)


if __name__ == "__main__":
    main()
