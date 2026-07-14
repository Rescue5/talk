from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TALK_COMBINED = PROJECT_ROOT / "backend" / "vendor" / "talk_combined"
sys.path.insert(0, str(TALK_COMBINED))

from talc_analysis.config import RuntimeConfig  # noqa: E402
from talc_analysis.cv_pipeline import TalcCVPipeline  # noqa: E402
from talc_analysis.inference import SegmentationMode, TileSegmenter  # noqa: E402
from talc_analysis.models import load_checkpoint  # noqa: E402


GROUPS = {
    "talc": Path("Оталькованные руды"),
    "ordinary": Path("ordinary"),
    "difficult": Path("difficult"),
}


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB"))


def image_files(root: Path, limit_per_group: int | None) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for group, relative in GROUPS.items():
        group_files = sorted(
            path
            for path in (root / relative).iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        )
        if limit_per_group is not None:
            group_files = group_files[:limit_per_group]
        files.extend((group, path) for path in group_files)
    return files


def cache_path(cache_dir: Path, image_path: Path) -> Path:
    digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.npz"


def build_segmenter(checkpoint: Path, config: Path, threshold: float) -> tuple[TileSegmenter, RuntimeConfig]:
    runtime = RuntimeConfig.from_yaml(config)
    model, checkpoint_config, device, _ = load_checkpoint(checkpoint)
    checkpoint_config = replace(checkpoint_config, threshold=threshold)
    segmenter = TileSegmenter(
        model=model,
        checkpoint=checkpoint_config,
        device=device,
        overlap=runtime.overlap,
        batch_size=runtime.batch_size,
    )
    return segmenter, runtime


def ensure_cache(
    files: list[tuple[str, Path]],
    cache_dir: Path,
    checkpoint: Path,
    config: Path,
    threshold: float,
    mode: str,
) -> RuntimeConfig:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [(group, path) for group, path in files if not cache_path(cache_dir, path).is_file()]
    segmenter: TileSegmenter | None = None
    runtime = RuntimeConfig.from_yaml(config)
    if missing:
        print(f"cache: {len(missing)} missing coarse masks, loading talc model", flush=True)
        segmenter, runtime = build_segmenter(checkpoint, config, threshold)
        print(f"cache: model device={segmenter.device}", flush=True)
    for index, (group, path) in enumerate(missing, start=1):
        started = perf_counter()
        image = read_rgb(path)
        result = segmenter.predict(image, SegmentationMode(mode))  # type: ignore[union-attr]
        np.savez(
            cache_path(cache_dir, path),
            mask=result.mask.astype(np.uint8),
            confidence=result.confidence.astype(np.float32),
            group=np.asarray(group),
            path=np.asarray(str(path)),
        )
        print(
            f"cache {index:03d}/{len(missing):03d} {group:9s} {path.name} "
            f"mask={float(result.mask.mean() * 100):5.2f}% {perf_counter() - started:5.2f}s",
            flush=True,
        )
    return runtime


def set_step_param(cv_config: dict[str, Any], step_type: str, key: str, value: Any) -> None:
    for step in cv_config.get("pipeline", []):
        if step.get("type") == step_type:
            step.setdefault("params", {})[key] = value
            return
    raise KeyError(step_type)


def set_operation_param(
    cv_config: dict[str, Any], operation_type: str, key: str, value: Any
) -> None:
    for step in cv_config.get("pipeline", []):
        if step.get("type") != "morphology_cleanup":
            continue
        for operation in step.get("params", {}).get("operations", []):
            if operation.get("type") == operation_type:
                operation[key] = value
                return
    raise KeyError(operation_type)


def variant_config(base_cv_config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_cv_config)
    for dotted_key, value in variant.items():
        if dotted_key == "name":
            continue
        if dotted_key.startswith("op."):
            _, operation, key = dotted_key.split(".", 2)
            set_operation_param(config, operation, key, value)
        else:
            step, key = dotted_key.split(".", 1)
            set_step_param(config, step, key, value)
    return config


def built_in_variants() -> list[dict[str, Any]]:
    return [
        {"name": "current"},
        {
            "name": "current_large_components",
            "component_filter.max_area": 2_000_000,
        },
        {
            "name": "large_components_seed045_grow010",
            "component_filter.max_area": 2_000_000,
            "hysteresis.seed_threshold": 0.45,
            "hysteresis.grow_threshold": 0.10,
            "hysteresis.strong_response_threshold": 0.52,
        },
        {
            "name": "large_components_seed040_grow008",
            "component_filter.max_area": 2_000_000,
            "hysteresis.seed_threshold": 0.40,
            "hysteresis.grow_threshold": 0.08,
            "hysteresis.strong_response_threshold": 0.50,
        },
        {
            "name": "large_components_cleanup",
            "component_filter.max_area": 2_000_000,
            "hysteresis.seed_threshold": 0.42,
            "hysteresis.grow_threshold": 0.10,
            "hysteresis.strong_response_threshold": 0.52,
            "op.remove_small_objects.min_area": 8,
            "op.fill_small_holes.max_area": 24,
        },
        {
            "name": "permissive_components_seed040",
            "component_filter.max_area": 2_000_000,
            "component_filter.shape_filter_min_area": 2_000_000,
            "component_filter.min_ring_pixels": 2_000_000,
            "component_filter.min_median_confidence": 0.0,
            "component_filter.min_persistence": 0.0,
            "hysteresis.seed_threshold": 0.40,
            "hysteresis.grow_threshold": 0.08,
            "hysteresis.strong_response_threshold": 0.50,
        },
        {
            "name": "permissive_components_seed030",
            "component_filter.max_area": 2_000_000,
            "component_filter.shape_filter_min_area": 2_000_000,
            "component_filter.min_ring_pixels": 2_000_000,
            "component_filter.min_median_confidence": 0.0,
            "component_filter.min_persistence": 0.0,
            "hysteresis.seed_threshold": 0.30,
            "hysteresis.grow_threshold": 0.03,
            "hysteresis.strong_response_threshold": 0.42,
        },
        {
            "name": "permissive_near_coarse",
            "component_filter.max_area": 2_000_000,
            "component_filter.shape_filter_min_area": 2_000_000,
            "component_filter.min_ring_pixels": 2_000_000,
            "component_filter.min_median_confidence": 0.0,
            "component_filter.min_persistence": 0.0,
            "hysteresis.seed_threshold": 0.20,
            "hysteresis.grow_threshold": 0.01,
            "hysteresis.strong_response_threshold": 0.30,
            "hysteresis.grow_min_evidence": 0,
            "op.remove_small_objects.min_area": 1,
            "op.fill_small_holes.max_area": 32,
        },
        {
            "name": "compromise_grow002",
            "component_filter.max_area": 2_000_000,
            "component_filter.shape_filter_min_area": 2_000_000,
            "component_filter.min_ring_pixels": 2_000_000,
            "component_filter.min_median_confidence": 0.0,
            "component_filter.min_persistence": 0.0,
            "hysteresis.seed_threshold": 0.25,
            "hysteresis.grow_threshold": 0.02,
            "hysteresis.strong_response_threshold": 0.34,
            "hysteresis.grow_min_evidence": 0,
            "op.remove_small_objects.min_area": 2,
            "op.fill_small_holes.max_area": 16,
        },
        {
            "name": "compromise_grow003",
            "component_filter.max_area": 2_000_000,
            "component_filter.shape_filter_min_area": 2_000_000,
            "component_filter.min_ring_pixels": 2_000_000,
            "component_filter.min_median_confidence": 0.0,
            "component_filter.min_persistence": 0.0,
            "hysteresis.seed_threshold": 0.25,
            "hysteresis.grow_threshold": 0.03,
            "hysteresis.strong_response_threshold": 0.36,
            "hysteresis.grow_min_evidence": 0,
            "op.remove_small_objects.min_area": 2,
            "op.fill_small_holes.max_area": 12,
        },
        {
            "name": "compromise_grow005",
            "component_filter.max_area": 2_000_000,
            "component_filter.shape_filter_min_area": 2_000_000,
            "component_filter.min_ring_pixels": 2_000_000,
            "component_filter.min_median_confidence": 0.0,
            "component_filter.min_persistence": 0.0,
            "hysteresis.seed_threshold": 0.30,
            "hysteresis.grow_threshold": 0.05,
            "hysteresis.strong_response_threshold": 0.40,
            "hysteresis.grow_min_evidence": 0,
            "op.remove_small_objects.min_area": 3,
            "op.fill_small_holes.max_area": 8,
        },
        {
            "name": "compromise_grow008",
            "component_filter.max_area": 2_000_000,
            "component_filter.shape_filter_min_area": 2_000_000,
            "component_filter.min_ring_pixels": 2_000_000,
            "component_filter.min_median_confidence": 0.0,
            "component_filter.min_persistence": 0.0,
            "hysteresis.seed_threshold": 0.35,
            "hysteresis.grow_threshold": 0.08,
            "hysteresis.strong_response_threshold": 0.45,
            "hysteresis.grow_min_evidence": 0,
            "op.remove_small_objects.min_area": 4,
            "op.fill_small_holes.max_area": 6,
        },
        {
            "name": "more_recall_seed045_grow010",
            "hysteresis.seed_threshold": 0.45,
            "hysteresis.grow_threshold": 0.10,
            "hysteresis.strong_response_threshold": 0.52,
        },
        {
            "name": "more_recall_seed040_grow008",
            "hysteresis.seed_threshold": 0.40,
            "hysteresis.grow_threshold": 0.08,
            "hysteresis.strong_response_threshold": 0.50,
        },
        {
            "name": "more_recall_seed035_grow006",
            "hysteresis.seed_threshold": 0.35,
            "hysteresis.grow_threshold": 0.06,
            "hysteresis.strong_response_threshold": 0.48,
        },
        {
            "name": "aggressive_seed030_grow003",
            "hysteresis.seed_threshold": 0.30,
            "hysteresis.grow_threshold": 0.03,
            "hysteresis.strong_response_threshold": 0.42,
            "hysteresis.grow_min_evidence": 1,
            "op.remove_small_objects.min_area": 4,
            "op.fill_small_holes.max_area": 16,
        },
        {
            "name": "aggressive_seed025_grow002",
            "hysteresis.seed_threshold": 0.25,
            "hysteresis.grow_threshold": 0.02,
            "hysteresis.strong_response_threshold": 0.36,
            "hysteresis.grow_min_evidence": 1,
            "op.remove_small_objects.min_area": 4,
            "op.fill_small_holes.max_area": 24,
        },
        {
            "name": "near_coarse_seed020_grow001",
            "hysteresis.seed_threshold": 0.20,
            "hysteresis.grow_threshold": 0.01,
            "hysteresis.strong_response_threshold": 0.30,
            "hysteresis.grow_min_evidence": 0,
            "op.remove_small_objects.min_area": 1,
            "op.fill_small_holes.max_area": 32,
            "component_filter.min_median_confidence": 0.0,
            "component_filter.min_persistence": 0.0,
        },
        {
            "name": "less_noise_seed050_grow012_shape",
            "hysteresis.seed_threshold": 0.50,
            "hysteresis.grow_threshold": 0.12,
            "hysteresis.strong_response_threshold": 0.56,
            "component_filter.shape_filter_min_area": 50,
            "component_filter.min_solidity": 0.10,
            "component_filter.max_elongation": 10.0,
        },
        {
            "name": "balanced_seed042_grow010_cleanup",
            "hysteresis.seed_threshold": 0.42,
            "hysteresis.grow_threshold": 0.10,
            "hysteresis.strong_response_threshold": 0.52,
            "op.remove_small_objects.min_area": 8,
            "op.fill_small_holes.max_area": 12,
            "component_filter.shape_filter_min_area": 40,
            "component_filter.min_solidity": 0.08,
        },
    ]


def evaluate(
    files: list[tuple[str, Path]],
    cache_dir: Path,
    cv_config: dict[str, Any],
    threshold_percent: float,
) -> dict[str, Any]:
    pipeline = TalcCVPipeline(cv_config)
    rows: list[dict[str, Any]] = []
    started = perf_counter()
    for group, path in files:
        cached = np.load(cache_path(cache_dir, path))
        image = read_rgb(path)
        mask = cached["mask"].astype(np.uint8)
        refined = pipeline.run(image, mask)
        percent = float(np.count_nonzero(refined.mask) / refined.mask.size * 100.0)
        coarse_pixels = int(np.count_nonzero(mask))
        refined_pixels = int(np.count_nonzero(refined.mask))
        rows.append(
            {
                "group": group,
                "path": str(path),
                "file": path.name,
                "percent": percent,
                "positive": percent > threshold_percent,
                "coarse_percent": float(coarse_pixels / mask.size * 100.0),
                "retained_ratio": (
                    float(refined_pixels / coarse_pixels) if coarse_pixels else 0.0
                ),
            }
        )
    summary: dict[str, Any] = {"seconds": perf_counter() - started, "groups": {}}
    for group in GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        positives = sum(row["positive"] for row in group_rows)
        percents = np.asarray([row["percent"] for row in group_rows], dtype=np.float32)
        summary["groups"][group] = {
            "total": len(group_rows),
            "positive": int(positives),
            "negative": int(len(group_rows) - positives),
            "positive_rate": float(positives / len(group_rows)) if group_rows else 0.0,
            "median_percent": float(np.median(percents)) if percents.size else 0.0,
            "mean_percent": float(np.mean(percents)) if percents.size else 0.0,
            "min_percent": float(np.min(percents)) if percents.size else 0.0,
            "max_percent": float(np.max(percents)) if percents.size else 0.0,
            "median_retained_ratio": float(
                np.median([row.get("retained_ratio", 0.0) for row in group_rows])
            )
            if group_rows
            else 0.0,
            "mean_retained_ratio": float(
                np.mean([row.get("retained_ratio", 0.0) for row in group_rows])
            )
            if group_rows
            else 0.0,
        }
    summary["rows"] = rows
    return summary


def score(summary: dict[str, Any]) -> float:
    groups = summary["groups"]
    talc_good = groups["talc"]["positive_rate"]
    ordinary_good = 1.0 - groups["ordinary"]["positive_rate"]
    difficult_good = 1.0 - groups["difficult"]["positive_rate"]
    return (2.0 * talc_good + ordinary_good + difficult_good) / 4.0


def print_summary(name: str, summary: dict[str, Any]) -> None:
    print(f"\n{name} score={score(summary):.3f} seconds={summary['seconds']:.1f}")
    for group, stats in summary["groups"].items():
        print(
            f"  {group:9s}: >10% {stats['positive']:3d}/{stats['total']:3d} "
            f"rate={stats['positive_rate']:.2f} "
            f"median={stats['median_percent']:.2f} "
            f"mean={stats['mean_percent']:.2f} "
            f"range=[{stats['min_percent']:.2f}, {stats['max_percent']:.2f}] "
            f"retain_med={stats.get('median_retained_ratio', 0.0):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "dataset" / "dataset" / "set1")
    parser.add_argument("--checkpoint", type=Path, default=Path(r"C:/Users/0000/Downloads/weights/weights/talc.pt"))
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "backend" / "vendor" / "talk_combined" / "talc_analysis" / "default_config.yaml")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "runs" / "talc_cv_tuning" / "cache")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "talc_cv_tuning" / "summary.json")
    parser.add_argument("--limit-per-group", type=int, default=None)
    parser.add_argument("--segmentation-threshold", type=float, default=0.5)
    parser.add_argument("--talc-threshold-percent", type=float, default=10.0)
    parser.add_argument("--mode", choices=["overlap", "no_overlap"], default="overlap")
    parser.add_argument("--cv-threads", type=int, default=8)
    parser.add_argument("--only", action="append", default=None)
    args = parser.parse_args()

    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, args.cv_threads))

    files = image_files(args.root, args.limit_per_group)
    print(f"files={len(files)} root={args.root}")
    runtime = ensure_cache(
        files,
        args.cache_dir,
        args.checkpoint,
        args.config,
        args.segmentation_threshold,
        args.mode,
    )

    coarse_summary = {
        "seconds": 0.0,
        "groups": {},
        "rows": [],
    }
    for group, path in files:
        cached = np.load(cache_path(args.cache_dir, path))
        mask = cached["mask"].astype(np.uint8)
        percent = float(np.count_nonzero(mask) / mask.size * 100.0)
        coarse_summary["rows"].append(
            {
                "group": group,
                "path": str(path),
                "file": path.name,
                "percent": percent,
                "positive": percent > args.talc_threshold_percent,
                "coarse_percent": percent,
                "retained_ratio": 1.0 if percent > 0 else 0.0,
            }
        )
    for group in GROUPS:
        group_rows = [row for row in coarse_summary["rows"] if row["group"] == group]
        positives = sum(row["positive"] for row in group_rows)
        percents = np.asarray([row["percent"] for row in group_rows], dtype=np.float32)
        coarse_summary["groups"][group] = {
            "total": len(group_rows),
            "positive": int(positives),
            "negative": int(len(group_rows) - positives),
            "positive_rate": float(positives / len(group_rows)) if group_rows else 0.0,
            "median_percent": float(np.median(percents)) if percents.size else 0.0,
            "mean_percent": float(np.mean(percents)) if percents.size else 0.0,
            "min_percent": float(np.min(percents)) if percents.size else 0.0,
            "max_percent": float(np.max(percents)) if percents.size else 0.0,
            "median_retained_ratio": 1.0,
            "mean_retained_ratio": 1.0,
        }
    print_summary("coarse_upper_bound", coarse_summary)

    selected = set(args.only or [])
    variants = [
        variant
        for variant in built_in_variants()
        if not selected or str(variant["name"]) in selected
    ]
    all_summaries: dict[str, Any] = {"coarse_upper_bound": coarse_summary}
    for variant in variants:
        name = str(variant["name"])
        config = variant_config(runtime.cv_config, variant)
        summary = evaluate(files, args.cache_dir, config, args.talc_threshold_percent)
        all_summaries[name] = summary
        print_summary(name, summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(all_summaries, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
