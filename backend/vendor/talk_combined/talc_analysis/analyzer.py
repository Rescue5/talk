from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import RuntimeConfig
from .cv_pipeline import TalcCVPipeline
from .inference import SegmentationMode, SegmentationResult, TileSegmenter
from .models import load_checkpoint
from .results import AnalysisResult


class Segmenter(Protocol):
    def predict(
        self, image_rgb: np.ndarray, mode: SegmentationMode | str
    ) -> SegmentationResult: ...


def canonical_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")
    if array.size == 0:
        raise ValueError("image_rgb must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("image_rgb contains non-finite values")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if array.dtype == np.uint16:
        return np.ascontiguousarray(np.rint(array / 257.0).astype(np.uint8))
    if np.issubdtype(array.dtype, np.floating):
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < 0:
            raise ValueError("Floating image values must be non-negative")
        if maximum <= 1:
            array = array * 255.0
        elif maximum > 255:
            raise ValueError("Floating image values must be in [0, 1] or [0, 255]")
        return np.ascontiguousarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))
    raise TypeError("image_rgb dtype must be uint8, uint16, float32, or float64")


def _confidence_summary(
    confidence: np.ndarray, selected_mask: np.ndarray
) -> dict[str, Any]:
    def summarize(values: np.ndarray) -> dict[str, float | None]:
        if values.size == 0:
            return {"mean": None, "median": None, "min": None, "max": None}
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return {
        "all_pixels": summarize(confidence),
        "inside_mask": summarize(confidence[selected_mask.astype(bool)]),
    }


def _contours(mask: np.ndarray) -> list[dict[str, Any]]:
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []
    return [
        {
            "id": index,
            "points": contour.reshape(-1, 2).astype(int).tolist(),
            "hierarchy": {
                "next": int(hierarchy[0, index, 0]),
                "previous": int(hierarchy[0, index, 1]),
                "first_child": int(hierarchy[0, index, 2]),
                "parent": int(hierarchy[0, index, 3]),
            },
        }
        for index, contour in enumerate(contours)
    ]


def _mask_statistics(mask: np.ndarray) -> dict[str, Any]:
    binary = mask.astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = [
        {
            "label": label,
            "area": int(stats[label, cv2.CC_STAT_AREA]),
            "bbox": [
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            ],
            "centroid": [
                float(centroids[label, 0]),
                float(centroids[label, 1]),
            ],
        }
        for label in range(1, count)
    ]
    pixels = int(binary.sum())
    total = int(binary.size)
    return {
        "pixel_count": pixels,
        "fraction": pixels / total,
        "percent": pixels / total * 100.0,
        "component_count": count - 1,
        "components": components,
        "contours": _contours(binary),
    }


def classify_talc(
    refined_mask: np.ndarray, threshold_percent: float
) -> dict[str, Any]:
    percent = float(np.count_nonzero(refined_mask) / refined_mask.size * 100.0)
    positive = percent > threshold_percent
    return {
        "code": "talc_bearing" if positive else "non_talc_bearing",
        "label_ru": "оталькованная руда" if positive else "НЕ оталькованная руда",
        "talc_percent": percent,
        "threshold_percent": threshold_percent,
        "rule": ">",
        "margin_percent": percent - threshold_percent,
    }


class TalcAnalyzer:
    def __init__(
        self,
        segmenter: Segmenter,
        cv_pipeline: TalcCVPipeline,
        runtime_config: RuntimeConfig,
        model_metadata: dict[str, Any],
        model_load_seconds: float = 0.0,
    ) -> None:
        self.segmenter = segmenter
        self.cv_pipeline = cv_pipeline
        self.runtime_config = runtime_config
        self.model_metadata = model_metadata
        self.model_load_seconds = model_load_seconds

    @classmethod
    def from_files(
        cls,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
    ) -> "TalcAnalyzer":
        runtime = RuntimeConfig.from_yaml(config_path)
        model, checkpoint, device, load_seconds = load_checkpoint(checkpoint_path)
        segmenter = TileSegmenter(
            model=model,
            checkpoint=checkpoint,
            device=device,
            overlap=runtime.overlap,
            batch_size=runtime.batch_size,
        )
        checkpoint_file = Path(checkpoint_path).expanduser().resolve()
        metadata = {
            "checkpoint_path": str(checkpoint_file),
            "checkpoint_size_bytes": checkpoint_file.stat().st_size,
            "architecture": checkpoint.model_name,
            "encoder": checkpoint.encoder_name,
            "class_names": list(checkpoint.class_names),
            "image_size": checkpoint.image_size,
            "tile_size": checkpoint.tile_size,
            "threshold": checkpoint.threshold,
            "device": str(device),
        }
        return cls(
            segmenter,
            TalcCVPipeline(runtime.cv_config),
            runtime,
            metadata,
            load_seconds,
        )

    def analyze(
        self,
        image_rgb: np.ndarray,
        mode: SegmentationMode | str,
        *,
        source_path: str | Path | None = None,
        image_load_seconds: float = 0.0,
    ) -> AnalysisResult:
        total_started = perf_counter()
        image = canonical_rgb(image_rgb)
        selected_mode = SegmentationMode(mode)

        started = perf_counter()
        segmentation = self.segmenter.predict(image, selected_mode)
        segmentation_seconds = perf_counter() - started

        started = perf_counter()
        refined = self.cv_pipeline.run(image, segmentation.mask)
        cv_seconds = perf_counter() - started
        if np.any(refined.mask.astype(bool) & ~segmentation.mask.astype(bool)):
            raise RuntimeError("CV refinement escaped the coarse segmentation mask")

        started = perf_counter()
        coarse_statistics = _mask_statistics(segmentation.mask)
        refined_statistics = _mask_statistics(refined.mask)
        classification = classify_talc(
            refined.mask, self.runtime_config.classification_threshold_percent
        )
        statistics_seconds = perf_counter() - started
        processing_total = perf_counter() - total_started + image_load_seconds

        source = Path(source_path).expanduser().resolve() if source_path else None
        statistics: dict[str, Any] = {
            "schema_version": "1.0",
            "source": {
                "path": str(source) if source else None,
                "file_name": source.name if source else None,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "channels": 3,
            },
            "model": self.model_metadata,
            "processing": {
                "mode": selected_mode.value,
                "overlap": (
                    self.runtime_config.overlap
                    if selected_mode is SegmentationMode.OVERLAP
                    else 0.0
                ),
                "tile_count": segmentation.tile_count,
            },
            "areas": {
                "segmentation": coarse_statistics,
                "refined_talc": refined_statistics,
            },
            "confidence": {
                "segmentation": _confidence_summary(
                    segmentation.confidence, segmentation.mask
                ),
                "cv": _confidence_summary(refined.confidence, refined.mask),
            },
            "cv_component_metrics": refined.component_metrics,
            "classification": classification,
            "timings_seconds": {
                "image_load": image_load_seconds,
                "segmentation": segmentation_seconds,
                "cv_refinement": cv_seconds,
                "statistics": statistics_seconds,
                "processing_total": processing_total,
            },
        }
        return AnalysisResult(
            image_rgb=image,
            segmentation_mask=segmentation.mask,
            refined_talc_mask=refined.mask,
            segmentation_confidence=segmentation.confidence,
            cv_confidence=refined.confidence,
            positive_votes=segmentation.positive_votes,
            vote_count=segmentation.vote_count,
            statistics=statistics,
        )

    def analyze_path(
        self,
        path: str | Path,
        mode: SegmentationMode | str,
    ) -> AnalysisResult:
        started = perf_counter()
        image_path = Path(path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        load_seconds = perf_counter() - started
        return self.analyze(
            image,
            mode,
            source_path=image_path,
            image_load_seconds=load_seconds,
        )

