from __future__ import annotations

import importlib
import json
import os
import random
import sys
import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps

from .config import ServiceConfig
from .schemas import JobSettings

StageCallback = Callable[[str, float, str | None], None]
PNG_SAVE_OPTIONS = {"compress_level": 1}


def save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path, **PNG_SAVE_OPTIONS)


def assess_image_quality(image_rgb: np.ndarray) -> dict[str, Any]:
    """Return inexpensive exposure diagnostics used to warn about unstable input."""

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")
    sample = image[::4, ::4].astype(np.float32) / 255.0
    luminance = (
        0.2126 * sample[..., 0]
        + 0.7152 * sample[..., 1]
        + 0.0722 * sample[..., 2]
    )
    median = float(np.median(luminance))
    mean = float(np.mean(luminance))
    dark_fraction = float(np.mean(luminance < 0.18))
    warnings: list[dict[str, Any]] = []
    if dark_fraction >= 0.60 or median < 0.13:
        warnings.append(
            {
                "code": "underexposed_image",
                "severity": "warning",
                "message": (
                    "Изображение слишком тёмное: сегментация и классификация "
                    "могут быть нестабильными. По возможности увеличьте экспозицию."
                ),
            }
        )
    return {
        "mean_luminance": mean,
        "median_luminance": median,
        "dark_pixel_fraction": dark_fraction,
        "warnings": warnings,
    }


def save_semantic_overlays(
    output_dir: Path,
    segmentation_mask: np.ndarray,
    refined_talc_mask: np.ndarray,
    sulfide_cv_mask: np.ndarray | None = None,
    sulfide_sam_mask: np.ndarray | None = None,
) -> None:
    import cv2

    height, width = segmentation_mask.shape
    coarse_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    contours, _ = cv2.findContours(
        segmentation_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(coarse_rgba, contours, -1, (255, 210, 0, 255), 2)
    save_png(output_dir / "coarse_overlay.png", coarse_rgba)

    talc_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    talc_rgba[refined_talc_mask.astype(bool)] = np.asarray(
        [255, 40, 40, 180], dtype=np.uint8
    )
    save_png(output_dir / "talc_overlay.png", talc_rgba)

    if sulfide_cv_mask is not None:
        _save_rgba_mask_layer(
            output_dir / "sulfide_cv_overlay.png",
            sulfide_cv_mask,
            np.asarray([255, 178, 38, 175], dtype=np.uint8),
        )
    if sulfide_sam_mask is not None:
        _save_rgba_mask_layer(
            output_dir / "sulfide_sam_overlay.png",
            sulfide_sam_mask,
            np.asarray([36, 144, 255, 180], dtype=np.uint8),
        )


def _save_rgba_mask_layer(
    path: Path, mask: np.ndarray, color_rgba: np.ndarray
) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask.astype(bool)] = color_rgba
    save_png(path, rgba)


def _as_binary_uint8(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != shape:
        raise ValueError("mask shape must match image height and width")
    return (array.astype(np.uint8) > 0).astype(np.uint8) * 255


def _exclude_talc(mask: np.ndarray, refined_talc_mask: np.ndarray) -> np.ndarray:
    shape = refined_talc_mask.shape
    mask_bool = _as_binary_uint8(mask, shape).astype(bool)
    talc_bool = refined_talc_mask.astype(bool)
    return ((mask_bool & ~talc_bool).astype(np.uint8) * 255).astype(np.uint8)


def _sulfide_mask_summary(mask: np.ndarray) -> dict[str, Any]:
    import cv2

    binary = mask.astype(np.uint8) > 0
    pixel_count = int(np.count_nonzero(binary))
    total = int(binary.size)
    component_count = 0
    if pixel_count:
        count, _, _, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8),
            connectivity=8,
        )
        component_count = int(count - 1)
    return {
        "pixel_count": pixel_count,
        "fraction": pixel_count / total if total else 0.0,
        "percent": pixel_count / total * 100.0 if total else 0.0,
        "component_count": component_count,
    }


def _save_sulfide_artifacts(
    output_dir: Path,
    sulfide_cv_mask: np.ndarray,
    sulfide_sam_mask: np.ndarray | None,
) -> dict[str, str]:
    save_png(output_dir / "sulfide_cv_mask.png", sulfide_cv_mask)
    artifacts = {
        "sulfide_cv_mask": "sulfide_cv_mask.png",
        "sulfide_cv_overlay": "sulfide_cv_overlay.png",
    }
    if sulfide_sam_mask is not None:
        save_png(output_dir / "sulfide_sam_mask.png", sulfide_sam_mask)
        artifacts["sulfide_sam_mask"] = "sulfide_sam_mask.png"
        artifacts["sulfide_sam_overlay"] = "sulfide_sam_overlay.png"
    return artifacts


def run_sulfide_segmentation(
    image_rgb: np.ndarray,
    refined_talc_mask: np.ndarray,
    config: dict[str, Any],
    segment_sulfides_fn: Callable[..., np.ndarray],
    *,
    sam_refiner: Any | None = None,
    sam_error: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build CV/SAM sulfide masks and enforce talc mask precedence."""

    started = perf_counter()
    raw_cv_mask = segment_sulfides_fn(image_rgb, config)
    cv_seconds = perf_counter() - started
    sulfide_cv_mask = _exclude_talc(raw_cv_mask, refined_talc_mask)

    sulfide_sam_mask = None
    sam_seconds = 0.0
    effective_sam_error = sam_error
    sam_stats: dict[str, Any] = {}
    sam_timings: dict[str, float] = {}
    if sam_refiner is not None and np.any(sulfide_cv_mask):
        sam_started = perf_counter()
        try:
            sam_cfg = config.get("sam", {})
            raw_sam_mask = sam_refiner.refine(
                image_rgb=image_rgb,
                cv_mask=sulfide_cv_mask,
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
            sulfide_sam_mask = _exclude_talc(raw_sam_mask, refined_talc_mask)
            effective_sam_error = None
            sam_stats = dict(getattr(sam_refiner, "last_stats", {}) or {})
            sam_timings = {
                key: float(value) / 1000.0
                for key, value in (
                    getattr(sam_refiner, "last_timings", {}) or {}
                ).items()
            }
        except Exception as exc:
            effective_sam_error = {
                "code": "sam_refinement_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }
        sam_seconds = perf_counter() - sam_started

    cv_summary = _sulfide_mask_summary(sulfide_cv_mask)
    sam_summary = (
        _sulfide_mask_summary(sulfide_sam_mask)
        if sulfide_sam_mask is not None
        else None
    )
    summary = {
        "cv": cv_summary,
        "sam": sam_summary,
        "selected": "sam" if sam_summary is not None else "cv",
        "sam_error": effective_sam_error,
        "sam_stats": sam_stats,
        "timings_seconds": {
            "cv": cv_seconds,
            "sam": sam_seconds,
            **{f"sam_{key}": value for key, value in sam_timings.items()},
            "total": cv_seconds + sam_seconds,
        },
    }
    return {
        "cv_mask": sulfide_cv_mask,
        "sam_mask": sulfide_sam_mask,
        "summary": summary,
    }


def cv_config_with_threshold(
    base_config: dict[str, Any], seed_threshold: float
) -> tuple[dict[str, Any], dict[str, float | None]]:
    config = deepcopy(base_config)
    for step in config.get("pipeline", []):
        if step.get("type") != "hysteresis" or not step.get("enabled", True):
            continue
        params = step.setdefault("params", {})
        params["seed_threshold"] = seed_threshold
        grow = float(params["grow_threshold"])
        if grow >= seed_threshold:
            grow = seed_threshold * 0.5
            params["grow_threshold"] = grow
        strong_raw = params.get("strong_response_threshold")
        strong = None if strong_raw is None else float(strong_raw)
        return config, {
            "seed_threshold": seed_threshold,
            "grow_threshold": grow,
            "strong_response_threshold": strong,
        }
    raise ValueError("CV configuration has no enabled hysteresis stage")


def combine_classification(
    talc: dict[str, Any],
    sulfide: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if talc["code"] == "talc_bearing":
        return {
            "code": "talc_bearing",
            "label_ru": "оталькованная руда",
            "confidence": None,
            "source": "talc_pipeline",
        }
    if sulfide is None:
        return None
    return {
        "code": sulfide["code"],
        "label_ru": sulfide["label_ru"],
        "confidence": sulfide["confidence"],
        "source": "sulfide_model",
    }


class ModelUnavailable(RuntimeError):
    def __init__(self, model: str, reason: str) -> None:
        super().__init__(f"{model} model is unavailable: {reason}")
        self.model = model
        self.reason = reason

    def payload(self) -> dict[str, str]:
        return {
            "code": "model_unavailable",
            "model": self.model,
            "message": str(self),
            "reason": self.reason,
        }


def _require_file(path: Path | None, model: str) -> Path:
    if path is None:
        raise ModelUnavailable(model, "checkpoint_not_configured")
    if not path.is_file():
        raise ModelUnavailable(model, "checkpoint_not_found")
    return path


def _add_source_path(path: Path, package: str) -> None:
    if not path.is_dir():
        raise ModelUnavailable(package, f"source_not_found:{path}")
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
    importlib.invalidate_caches()


def _configure_cv_runtime() -> None:
    try:
        import cv2

        cv2.setUseOptimized(True)
        default_threads = max(1, min(os.cpu_count() or 1, 8))
        threads = int(os.getenv("CV_NUM_THREADS", str(default_threads)))
        cv2.setNumThreads(max(1, threads))
    except Exception:
        pass


def _configure_torch_inference(torch_module: Any, device: Any) -> None:
    if not str(device).startswith("cuda"):
        return
    torch_module.backends.cudnn.benchmark = True
    try:
        torch_module.set_float32_matmul_precision("high")
    except Exception:
        pass


def _warmup_torch_model(
    torch_module: Any,
    model: Any,
    device: Any,
    input_shape: tuple[int, ...],
) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        with torch_module.inference_mode():
            model(torch_module.zeros(input_shape, device=device))
            torch_module.cuda.synchronize(device)
    except Exception:
        pass


class SulfideClassifier:
    def __init__(self, config: ServiceConfig) -> None:
        _add_source_path(config.sulfide_source_path, "sulfide")
        checkpoint_path = _require_file(config.sulfide_checkpoint_path, "sulfide")
        config_path = _require_file(config.sulfide_config_path, "sulfide")

        import torch
        from ore_classifier.config import load_config
        from ore_classifier.model import build_model

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            model_config = payload.get("config") or load_config(config_path)
            state_dict = payload["model_state_dict"]
        else:
            model_config = load_config(config_path)
            state_dict = payload
        model_config["model"]["pretrained"] = False
        model = build_model(model_config)
        model.load_state_dict(state_dict)

        requested = config.model_device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise ModelUnavailable("sulfide", "cuda_requested_but_unavailable")
        self.device = torch.device(requested)
        _configure_torch_inference(torch, self.device)
        self.torch = torch
        self.model = model.to(self.device).eval()
        self.config = model_config
        data_config = model_config.get("data", {})
        image_size = int(data_config.get("image_size", 384))
        view_count = int(data_config.get("num_local_crops", 8)) + 1
        _warmup_torch_model(
            torch,
            self.model,
            self.device,
            (1, view_count, 3, image_size, image_size),
        )

    def predict(self, image_rgb: np.ndarray, threshold: float) -> dict[str, Any]:
        from ore_classifier.dataset import make_views

        started = perf_counter()
        views = make_views(
            image_rgb, self.config, train=False, rng=random.Random(0)
        ).unsqueeze(0)
        preprocess_seconds = perf_counter() - started

        started = perf_counter()
        with self.torch.inference_mode():
            logits = self.model(views.to(self.device))
            probability_difficult = float(
                self.torch.sigmoid(logits)[0].detach().cpu()
            )
        inference_seconds = perf_counter() - started
        probability_ordinary = 1.0 - probability_difficult
        code = "difficult" if probability_difficult >= threshold else "ordinary"
        return {
            "code": code,
            "label_ru": (
                "трудно-обогатимая руда" if code == "difficult" else "рядовая руда"
            ),
            "confidence": (
                probability_difficult
                if code == "difficult"
                else probability_ordinary
            ),
            "probability_ordinary": probability_ordinary,
            "probability_difficult": probability_difficult,
            "threshold": threshold,
            "timings_seconds": {
                "preprocess": preprocess_seconds,
                "inference": inference_seconds,
                "total": preprocess_seconds + inference_seconds,
            },
        }


class InferenceProcessor:
    """Lazy, process-wide model owner used by the single-worker job queue."""

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        _configure_cv_runtime()
        self._talc_analyzer: Any | None = None
        self._sulfide_classifier: SulfideClassifier | None = None
        self._sulfide_segmentation_config: dict[str, Any] | None = None
        self._sulfide_sam_refiner: Any | None = None
        self._sulfide_sam_checked = False
        self._sulfide_sam_error: dict[str, str] | None = None
        self._model_lock = threading.Lock()

    def _talc(self) -> Any:
        with self._model_lock:
            if self._talc_analyzer is None:
                _add_source_path(self.config.talc_source_path, "talc")
                checkpoint = _require_file(self.config.talc_checkpoint_path, "talc")
                from talc_analysis import TalcAnalyzer

                self._talc_analyzer = TalcAnalyzer.from_files(
                    checkpoint, self.config.talc_config_path
                )
                import torch

                _configure_torch_inference(
                    torch, self._talc_analyzer.segmenter.device
                )
                _warmup_torch_model(
                    torch,
                    self._talc_analyzer.segmenter.model,
                    self._talc_analyzer.segmenter.device,
                    (
                        1,
                        3,
                        self._talc_analyzer.segmenter.checkpoint.image_size,
                        self._talc_analyzer.segmenter.checkpoint.image_size,
                    ),
                )
            return self._talc_analyzer

    def _sulfide(self) -> SulfideClassifier:
        with self._model_lock:
            if self._sulfide_classifier is None:
                self._sulfide_classifier = SulfideClassifier(self.config)
            return self._sulfide_classifier

    def _sulfide_segmentation_tools(
        self,
    ) -> tuple[dict[str, Any], Callable[..., np.ndarray]]:
        _add_source_path(self.config.sulfide_source_path, "sulfide_segmentation")
        from cv_analysis.sulfide_candidates import load_sulfide_config, segment_sulfides

        with self._model_lock:
            if self._sulfide_segmentation_config is None:
                config_path = _require_file(
                    self.config.sulfide_segmentation_config_path,
                    "sulfide_segmentation_config",
                )
                loaded = load_sulfide_config(config_path)
                loaded = deepcopy(loaded)
                sam_cfg = loaded.setdefault("sam", {})
                if self.config.sulfide_sam_checkpoint_path is not None:
                    sam_cfg["checkpoint"] = str(
                        self.config.sulfide_sam_checkpoint_path
                    )
                sam_cfg["device"] = self.config.sulfide_sam_device
                self._sulfide_segmentation_config = loaded
            return deepcopy(self._sulfide_segmentation_config), segment_sulfides

    def _sulfide_sam(self) -> tuple[Any | None, dict[str, str] | None]:
        with self._model_lock:
            if self._sulfide_sam_checked:
                return self._sulfide_sam_refiner, self._sulfide_sam_error
            self._sulfide_sam_checked = True

            checkpoint = self.config.sulfide_sam_checkpoint_path
            if checkpoint is None:
                self._sulfide_sam_error = {
                    "code": "sam_checkpoint_not_configured",
                    "message": "MobileSAM checkpoint is not configured; CV sulfide mask is used.",
                }
                return None, self._sulfide_sam_error
            if not checkpoint.is_file():
                self._sulfide_sam_error = {
                    "code": "sam_checkpoint_not_found",
                    "message": f"MobileSAM checkpoint not found: {checkpoint}",
                }
                return None, self._sulfide_sam_error

            try:
                _add_source_path(
                    self.config.sulfide_source_path, "sulfide_segmentation"
                )
                from cv_analysis.sulfide_candidates import MobileSamRefiner

                device = self.config.sulfide_sam_device
                if device == "auto":
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    import torch

                _configure_torch_inference(torch, device)
                self._sulfide_sam_refiner = MobileSamRefiner(
                    checkpoint=checkpoint,
                    device=device,
                )
                if str(device).startswith("cuda"):
                    try:
                        with torch.inference_mode():
                            self._sulfide_sam_refiner.predictor.set_image(
                                np.zeros((256, 256, 3), dtype=np.uint8)
                            )
                            torch.cuda.synchronize(device)
                    except Exception:
                        pass
                self._sulfide_sam_error = None
            except Exception as exc:
                self._sulfide_sam_error = {
                    "code": "sam_unavailable",
                    "message": f"{type(exc).__name__}: {exc}",
                }
                self._sulfide_sam_refiner = None
            return self._sulfide_sam_refiner, self._sulfide_sam_error

    def preload(self) -> None:
        if self.config.demo_mode:
            return
        try:
            self._talc()
        except ModelUnavailable:
            pass
        try:
            self._sulfide_segmentation_tools()
        except ModelUnavailable:
            pass
        try:
            self._sulfide_sam()
        except ModelUnavailable:
            pass
        try:
            self._sulfide()
        except ModelUnavailable:
            pass

    def process(
        self,
        image_path: Path,
        output_dir: Path,
        settings: JobSettings,
        progress: StageCallback,
    ) -> dict[str, Any]:
        if self.config.demo_mode:
            return self._process_demo(image_path, output_dir, settings, progress)
        return self._process_models(image_path, output_dir, settings, progress)

    def reprocess(
        self,
        image_path: Path,
        output_dir: Path,
        settings: JobSettings,
        progress: StageCallback,
        *,
        recompute_from: str,
    ) -> dict[str, Any]:
        if recompute_from == "classification":
            return self._reclassify_cached(output_dir, settings, progress)
        if self.config.demo_mode:
            return self._process_demo(image_path, output_dir, settings, progress)
        return self._process_models(
            image_path,
            output_dir,
            settings,
            progress,
            reuse_segmentation=recompute_from
            in {"cv_refinement", "segmentation_threshold"},
            rethreshold_segmentation=recompute_from == "segmentation_threshold",
            reuse_sulfide=recompute_from
            in {"cv_refinement", "segmentation_threshold"},
        )

    def _process_models(
        self,
        image_path: Path,
        output_dir: Path,
        settings: JobSettings,
        progress: StageCallback,
        *,
        reuse_segmentation: bool = False,
        rethreshold_segmentation: bool = False,
        reuse_sulfide: bool = False,
    ) -> dict[str, Any]:
        _add_source_path(self.config.talc_source_path, "talc")
        analyzer = None if reuse_segmentation else self._talc()
        from talc_analysis.analyzer import (
            _confidence_summary,
            _mask_statistics,
            canonical_rgb,
            classify_talc,
        )
        from talc_analysis.config import RuntimeConfig
        from talc_analysis.cv_pipeline import TalcCVPipeline
        from talc_analysis.inference import SegmentationMode, TileSegmenter
        from talc_analysis.results import AnalysisResult, json_safe

        runtime_config = (
            RuntimeConfig.from_yaml(self.config.talc_config_path)
            if analyzer is None
            else analyzer.runtime_config
        )

        total_started = perf_counter()
        started = perf_counter()
        with Image.open(image_path) as source:
            image = canonical_rgb(
                np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
            )
        load_seconds = perf_counter() - started
        quality = assess_image_quality(image)

        cached_manifest: dict[str, Any] = {}
        if reuse_segmentation:
            progress(
                (
                    "segmentation_threshold"
                    if rethreshold_segmentation
                    else "cv_refinement"
                ),
                0.05,
                (
                    "Re-thresholding cached segmentation probability"
                    if rethreshold_segmentation
                    else "Using cached coarse segmentation"
                ),
            )
            with (output_dir / "result.json").open("r", encoding="utf-8") as stream:
                cached_manifest = json.load(stream)
            segmentation_mask = (
                np.asarray(Image.open(output_dir / "segmentation_mask.png")) > 0
            ).astype(np.uint8)
            with np.load(output_dir / "confidence_maps.npz") as maps:
                segmentation_confidence = maps[
                    "segmentation_confidence"
                ].copy()
                if rethreshold_segmentation:
                    segmentation_mask = (
                        segmentation_confidence
                        >= settings.segmentation_threshold
                    ).astype(np.uint8)
                    vote_count = maps["vote_count"].copy()
                    positive_votes = (
                        segmentation_mask.astype(vote_count.dtype) * vote_count
                    )
                else:
                    vote_count = maps["vote_count"].copy()
                    positive_votes = maps["positive_votes"].copy()
                segmentation = SimpleNamespace(
                    mask=segmentation_mask,
                    confidence=segmentation_confidence,
                    positive_votes=positive_votes,
                    vote_count=vote_count,
                    tile_count=int(
                        cached_manifest.get("processing", {}).get("tile_count", 0)
                    ),
                )
            segmentation_seconds = 0.0
        else:
            progress("talc_segmentation", 0.05, "Segmenting talc regions")
            started = perf_counter()
            job_checkpoint = replace(
                # analyzer is guaranteed for fresh neural inference.
                analyzer.segmenter.checkpoint,
                threshold=settings.segmentation_threshold,
            )
            job_segmenter = TileSegmenter(
                model=analyzer.segmenter.model,
                checkpoint=job_checkpoint,
                device=analyzer.segmenter.device,
                overlap=analyzer.segmenter.overlap,
                batch_size=analyzer.segmenter.batch_size,
            )
            segmentation = job_segmenter.predict(
                image, SegmentationMode(settings.mode)
            )
            segmentation_seconds = perf_counter() - started

        progress("cv_refinement", 0.45, "Refining talc regions")
        started = perf_counter()
        cv_config, applied_cv = cv_config_with_threshold(
            runtime_config.cv_config, settings.cv_threshold
        )
        refined = TalcCVPipeline(cv_config).run(image, segmentation.mask)
        cv_seconds = perf_counter() - started
        if np.any(refined.mask.astype(bool) & ~segmentation.mask.astype(bool)):
            raise RuntimeError("CV refinement escaped the coarse segmentation mask")

        progress("sulfide_segmentation", 0.58, "Segmenting sulfide inclusions")
        sulfide_segmentation_config, segment_sulfides_fn = (
            self._sulfide_segmentation_tools()
        )
        sam_refiner = None
        sam_error = None
        if bool(sulfide_segmentation_config.get("sam", {}).get("enabled", False)):
            sam_refiner, sam_error = self._sulfide_sam()
        started = perf_counter()
        sulfide_segmentation_output = run_sulfide_segmentation(
            image,
            refined.mask,
            sulfide_segmentation_config,
            segment_sulfides_fn,
            sam_refiner=sam_refiner,
            sam_error=sam_error,
        )
        sulfide_segmentation_seconds = perf_counter() - started
        sulfide_cv_mask = sulfide_segmentation_output["cv_mask"]
        sulfide_sam_mask = sulfide_segmentation_output["sam_mask"]
        sulfide_segmentation = sulfide_segmentation_output["summary"]

        started = perf_counter()
        coarse_statistics = _mask_statistics(segmentation.mask)
        refined_statistics = _mask_statistics(refined.mask)
        classification = classify_talc(
            refined.mask, settings.talc_threshold_percent
        )
        statistics_seconds = perf_counter() - started
        statistics: dict[str, Any] = {
            "schema_version": "1.1",
            "source": {
                "path": str(image_path),
                "file_name": image_path.name,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "channels": 3,
            },
            "model": (
                cached_manifest.get("model", {})
                if reuse_segmentation
                else analyzer.model_metadata
            ),
            "processing": {
                "mode": settings.mode,
                "overlap": (
                    runtime_config.overlap
                    if settings.mode == "overlap"
                    else 0.0
                ),
                "tile_count": segmentation.tile_count,
                "thresholds": {
                    "segmentation_probability": settings.segmentation_threshold,
                    "cv_seed": applied_cv["seed_threshold"],
                    "cv_grow": applied_cv["grow_threshold"],
                    "cv_strong_response": applied_cv[
                        "strong_response_threshold"
                    ],
                    "talc_percent": settings.talc_threshold_percent,
                    "sulfide_probability": settings.sulfide_threshold,
                },
            },
            "areas": {
                "segmentation": coarse_statistics,
                "refined_talc": refined_statistics,
                "sulfide_cv": sulfide_segmentation["cv"],
                **(
                    {"sulfide_sam": sulfide_segmentation["sam"]}
                    if sulfide_segmentation["sam"] is not None
                    else {}
                ),
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
                "image_load": load_seconds,
                "segmentation": segmentation_seconds,
                "cv_refinement": cv_seconds,
                "sulfide_segmentation": sulfide_segmentation_seconds,
                "statistics": statistics_seconds,
            },
            "sulfide_segmentation": sulfide_segmentation,
            "quality": quality,
        }
        talc_public = {
            **classification,
            "coarse_percent": coarse_statistics["percent"],
            "refined_percent": refined_statistics["percent"],
        }
        result = AnalysisResult(
            image_rgb=image,
            segmentation_mask=segmentation.mask,
            refined_talc_mask=refined.mask,
            segmentation_confidence=segmentation.confidence,
            cv_confidence=refined.confidence,
            positive_votes=segmentation.positive_votes,
            vote_count=segmentation.vote_count,
            statistics=statistics,
        )

        sulfide: dict[str, Any] | None = None
        final: dict[str, Any] | None
        unavailable: dict[str, str] | None = None
        sulfide_error: dict[str, str] | None = None
        progress(
            "sulfide_classification",
            0.72,
            (
                "Using cached sulfide probabilities"
                if reuse_sulfide
                else "Caching ordinary/difficult probabilities"
            ),
        )
        try:
            if reuse_sulfide:
                cached_sulfide = cached_manifest.get("sulfide")
                if not cached_sulfide:
                    raise ModelUnavailable("sulfide", "cached_prediction_not_found")
                sulfide = self._sulfide_from_probabilities(
                    cached_sulfide, settings.sulfide_threshold
                )
            else:
                sulfide = self._sulfide().predict(image, settings.sulfide_threshold)
            statistics["timings_seconds"]["sulfide"] = sulfide.get(
                "timings_seconds", {}
            ).get("total", 0.0)
        except ModelUnavailable as error:
            sulfide_error = error.payload()
            if classification["code"] != "talc_bearing":
                unavailable = sulfide_error

        final = combine_classification(talc_public, sulfide)
        if classification["code"] == "talc_bearing":
            progress(
                "sulfide_classification",
                0.72,
                (
                    "Sulfide probabilities cached"
                    if sulfide
                    else "Talc result is valid; sulfide cache unavailable"
                ),
            )

        progress("export", 0.9, "Saving masks, overlay, confidence and statistics")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_png(output_dir / "original.png", image)
        result.save(output_dir, overwrite=True, write_manifest=False)
        save_semantic_overlays(
            output_dir,
            segmentation.mask,
            refined.mask,
            sulfide_cv_mask,
            sulfide_sam_mask,
        )
        sulfide_artifacts = _save_sulfide_artifacts(
            output_dir,
            sulfide_cv_mask,
            sulfide_sam_mask,
        )
        final_manifest = dict(result.statistics)
        final_manifest["demo"] = False
        final_manifest["sulfide"] = sulfide
        final_manifest["sulfide_error"] = sulfide_error
        final_manifest["sulfide_segmentation"] = sulfide_segmentation
        final_manifest["ore_classification"] = final
        final_manifest.setdefault("artifacts", {})["original"] = "original.png"
        final_manifest["artifacts"]["coarse_overlay"] = "coarse_overlay.png"
        final_manifest["artifacts"]["talc_overlay"] = "talc_overlay.png"
        final_manifest["artifacts"].update(sulfide_artifacts)
        if unavailable:
            final_manifest["error"] = unavailable
        final_manifest["timings_seconds"]["pipeline_total"] = (
            perf_counter() - total_started
        )
        with (output_dir / "result.json").open("w", encoding="utf-8") as stream:
            json.dump(
                json_safe(final_manifest),
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
        progress("export", 1.0, "Image processing completed")

        return {
            "status": "model_unavailable" if unavailable else "completed",
            "demo": False,
            "classification": final,
            "talc": talc_public,
            "sulfide": sulfide,
            "sulfide_segmentation": sulfide_segmentation,
            "sulfide_error": sulfide_error,
            "warnings": quality["warnings"],
            "timings": final_manifest["timings_seconds"],
            "error": unavailable,
            "artifacts": {
                "original": "original.png",
                "segmentation_mask": "segmentation_mask.png",
                "refined_talc_mask": "refined_talc_mask.png",
                "coarse_overlay": "coarse_overlay.png",
                "talc_overlay": "talc_overlay.png",
                "overlay": "overlay.png",
                "confidence_maps": "confidence_maps.npz",
                "result": "result.json",
                **sulfide_artifacts,
            },
        }

    @staticmethod
    def _sulfide_from_probabilities(
        cached: dict[str, Any], threshold: float
    ) -> dict[str, Any]:
        difficult = float(cached["probability_difficult"])
        ordinary = float(cached.get("probability_ordinary", 1.0 - difficult))
        code = "difficult" if difficult >= threshold else "ordinary"
        return {
            **cached,
            "code": code,
            "label_ru": (
                "трудно-обогатимая руда" if code == "difficult" else "рядовая руда"
            ),
            "confidence": difficult if code == "difficult" else ordinary,
            "probability_ordinary": ordinary,
            "probability_difficult": difficult,
            "threshold": threshold,
        }

    def _reclassify_cached(
        self,
        output_dir: Path,
        settings: JobSettings,
        progress: StageCallback,
    ) -> dict[str, Any]:
        started = perf_counter()
        progress(
            "sulfide_classification", 0.2, "Reclassifying cached probabilities"
        )
        manifest_path = output_dir / "result.json"
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        talc = dict(manifest["classification"])
        percent = float(talc["talc_percent"])
        talc.update(
            {
                "code": (
                    "talc_bearing"
                    if percent > settings.talc_threshold_percent
                    else "non_talc_bearing"
                ),
                "label_ru": (
                    "оталькованная руда"
                    if percent > settings.talc_threshold_percent
                    else "НЕ оталькованная руда"
                ),
                "threshold_percent": settings.talc_threshold_percent,
                "rule": ">",
                "margin_percent": percent - settings.talc_threshold_percent,
            }
        )
        sulfide_cached = manifest.get("sulfide")
        sulfide = (
            self._sulfide_from_probabilities(
                sulfide_cached, settings.sulfide_threshold
            )
            if sulfide_cached
            else None
        )
        final = combine_classification(talc, sulfide)
        unavailable = None
        if final is None:
            unavailable = {
                "code": "model_unavailable",
                "model": "sulfide",
                "reason": "cached_prediction_not_found",
                "message": "Cached sulfide prediction is unavailable.",
            }
        thresholds = manifest.setdefault("processing", {}).setdefault(
            "thresholds", {}
        )
        thresholds["talc_percent"] = settings.talc_threshold_percent
        thresholds["sulfide_probability"] = settings.sulfide_threshold
        manifest["classification"] = talc
        manifest["sulfide"] = sulfide
        manifest["ore_classification"] = final
        manifest["error"] = unavailable
        manifest.setdefault("timings_seconds", {})["reclassification"] = (
            perf_counter() - started
        )
        progress("export", 0.8, "Saving updated classification")
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(
                manifest,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
        progress("export", 1.0, "Cached reclassification completed")
        talc_public = {
            **talc,
            "coarse_percent": manifest.get("areas", {})
            .get("segmentation", {})
            .get("percent"),
            "refined_percent": manifest.get("areas", {})
            .get("refined_talc", {})
            .get("percent", percent),
        }
        artifacts = {
            key: value
            for key, value in manifest.get("artifacts", {}).items()
            if isinstance(value, str)
        }
        return {
            "status": "model_unavailable" if unavailable else "completed",
            "demo": bool(manifest.get("demo", False)),
            "classification": final,
            "talc": talc_public,
            "sulfide": sulfide,
            "sulfide_segmentation": manifest.get("sulfide_segmentation"),
            "sulfide_error": manifest.get("sulfide_error"),
            "warnings": manifest.get("quality", {}).get("warnings", []),
            "timings": manifest["timings_seconds"],
            "error": unavailable,
            "artifacts": artifacts,
        }

    def _process_demo(
        self,
        image_path: Path,
        output_dir: Path,
        settings: JobSettings,
        progress: StageCallback,
    ) -> dict[str, Any]:
        """Explicitly marked visual demo. It is never enabled implicitly."""

        started = perf_counter()
        with Image.open(image_path) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        quality = assess_image_quality(image)
        progress("talc_segmentation", 0.1, "DEMO: creating illustrative mask")
        gray = image.astype(np.float32).mean(axis=2)
        segmentation = gray < float(np.percentile(gray, 45))
        progress("cv_refinement", 0.45, "DEMO: refining illustrative mask")
        refined = segmentation & (image[:, :, 1] < image[:, :, 0])
        progress(
            "sulfide_segmentation",
            0.58,
            "DEMO: creating illustrative sulfide masks",
        )
        sulfide_started = perf_counter()
        import cv2

        sulfide_cv_bool = (gray > float(np.percentile(gray, 74))) & ~refined
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sulfide_sam_bool = cv2.morphologyEx(
            sulfide_cv_bool.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
        ).astype(bool) & ~refined
        sulfide_cv_mask = (sulfide_cv_bool.astype(np.uint8) * 255).astype(np.uint8)
        sulfide_sam_mask = (sulfide_sam_bool.astype(np.uint8) * 255).astype(np.uint8)
        sulfide_cv_summary = _sulfide_mask_summary(sulfide_cv_mask)
        sulfide_sam_summary = _sulfide_mask_summary(sulfide_sam_mask)
        sulfide_total_seconds = perf_counter() - sulfide_started
        sulfide_segmentation = {
            "cv": sulfide_cv_summary,
            "sam": sulfide_sam_summary,
            "selected": "sam",
            "sam_error": None,
            "sam_stats": {
                "components_considered": sulfide_cv_summary["component_count"],
                "components_refined": sulfide_sam_summary["component_count"],
                "components_fallback": 0,
            },
            "timings_seconds": {
                "cv": 0.0,
                "sam": sulfide_total_seconds,
                "total": sulfide_total_seconds,
            },
        }
        talc_percent = float(refined.mean() * 100.0)
        talc = {
            "code": (
                "talc_bearing"
                if talc_percent > settings.talc_threshold_percent
                else "non_talc_bearing"
            ),
            "talc_percent": talc_percent,
            "threshold_percent": settings.talc_threshold_percent,
            "rule": ">",
        }
        progress("sulfide_classification", 0.72, "DEMO: illustrative classification")
        difficult_probability = float(np.clip(1.0 - gray.mean() / 255.0, 0, 1))
        ordinary_probability = 1.0 - difficult_probability
        sulfide = self._sulfide_from_probabilities(
            {
                "probability_ordinary": ordinary_probability,
                "probability_difficult": difficult_probability,
                "confidence": max(ordinary_probability, difficult_probability),
                "timings_seconds": {"total": 0.0},
            },
            settings.sulfide_threshold,
        )
        final = combine_classification(talc, sulfide)
        if final is not None:
            final["source"] = "explicit_demo"

        progress("export", 0.9, "DEMO: saving illustrative artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_png(output_dir / "original.png", image)
        save_png(
            output_dir / "segmentation_mask.png",
            segmentation.astype(np.uint8) * 255,
        )
        save_png(
            output_dir / "refined_talc_mask.png",
            refined.astype(np.uint8) * 255,
        )
        overlay = image.copy()
        overlay[refined] = np.array([255, 40, 40], dtype=np.uint8)
        save_png(output_dir / "overlay.png", overlay)
        save_semantic_overlays(
            output_dir,
            segmentation,
            refined,
            sulfide_cv_mask,
            sulfide_sam_mask,
        )
        sulfide_artifacts = _save_sulfide_artifacts(
            output_dir,
            sulfide_cv_mask,
            sulfide_sam_mask,
        )
        np.savez(
            output_dir / "confidence_maps.npz",
            segmentation_confidence=segmentation.astype(np.float32),
            cv_confidence=refined.astype(np.float32),
            positive_votes=segmentation.astype(np.uint16),
            vote_count=np.ones(segmentation.shape, dtype=np.uint16),
        )
        timings = {
            "sulfide_segmentation": sulfide_total_seconds,
            "pipeline_total": perf_counter() - started,
        }
        manifest = {
            "schema_version": "1.1",
            "demo": True,
            "warning": "Illustrative DEMO_MODE output; not produced by trained models.",
            "processing": {
                "mode": settings.mode,
                "thresholds": {
                    "segmentation_probability": settings.segmentation_threshold,
                    "cv_seed": settings.cv_threshold,
                    "talc_percent": settings.talc_threshold_percent,
                    "sulfide_probability": settings.sulfide_threshold,
                },
            },
            "areas": {
                "sulfide_cv": sulfide_segmentation["cv"],
                "sulfide_sam": sulfide_segmentation["sam"],
            },
            "classification": talc,
            "sulfide": sulfide,
            "sulfide_segmentation": sulfide_segmentation,
            "quality": quality,
            "ore_classification": final,
            "timings_seconds": timings,
            "artifacts": {
                "original": "original.png",
                "segmentation_mask": "segmentation_mask.png",
                "refined_talc_mask": "refined_talc_mask.png",
                "coarse_overlay": "coarse_overlay.png",
                "talc_overlay": "talc_overlay.png",
                "overlay": "overlay.png",
                "confidence_maps": "confidence_maps.npz",
                "result": "result.json",
                **sulfide_artifacts,
            },
        }
        with (output_dir / "result.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
        progress("export", 1.0, "DEMO image processing completed")
        return {
            "status": "completed",
            "demo": True,
            "classification": final,
            "talc": talc,
            "sulfide": sulfide,
            "sulfide_segmentation": sulfide_segmentation,
            "warnings": quality["warnings"],
            "timings": timings,
            "error": None,
            "artifacts": {
                "original": "original.png",
                "segmentation_mask": "segmentation_mask.png",
                "refined_talc_mask": "refined_talc_mask.png",
                "coarse_overlay": "coarse_overlay.png",
                "talc_overlay": "talc_overlay.png",
                "overlay": "overlay.png",
                "confidence_maps": "confidence_maps.npz",
                "result": "result.json",
                **sulfide_artifacts,
            },
        }
