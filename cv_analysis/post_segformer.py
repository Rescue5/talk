"""High-precision CV post-processing for dark talc-like candidates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


@dataclass
class ProcessingState:
    image_rgb: np.ndarray
    data: dict[str, np.ndarray]
    metadata: dict[str, Any]


class CVProcessor(ABC):
    @abstractmethod
    def process(self, state: ProcessingState) -> ProcessingState:
        """Process a state and return it."""

    def __call__(self, state: ProcessingState) -> ProcessingState:
        return self.process(state)


class PrepareImage(CVProcessor):
    def __init__(
        self,
        input_color: str,
        black_clip: float,
        white_clip: float,
        border_margin: int,
    ) -> None:
        if input_color not in {"rgb", "bgr"}:
            raise ValueError("input_color must be 'rgb' or 'bgr'")
        if not 0.0 <= black_clip < white_clip <= 1.0:
            raise ValueError("black_clip and white_clip must satisfy 0 <= black < white <= 1")
        if border_margin < 0:
            raise ValueError("border_margin must be non-negative")

        self.input_color = input_color
        self.black_clip = float(black_clip)
        self.white_clip = float(white_clip)
        self.border_margin = int(border_margin)

    def process(self, state: ProcessingState) -> ProcessingState:
        image_rgb, finite_mask = _to_float_rgb(state.image_rgb, self.input_color)

        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        luminance = (lab[..., 0] / 100.0).astype(np.float32, copy=False)

        valid_mask = (
            finite_mask
            & (luminance > self.black_clip)
            & (luminance < self.white_clip)
        )
        segformer_mask = _to_bool_mask(
            _require_array(state, "segformer_mask"),
            luminance.shape,
        )
        valid_mask &= segformer_mask
        state.data["segformer_mask"] = segformer_mask

        if self.border_margin > 0:
            margin = self.border_margin
            valid_mask[:margin, :] = False
            valid_mask[-margin:, :] = False
            valid_mask[:, :margin] = False
            valid_mask[:, -margin:] = False

        state.image_rgb = image_rgb
        state.data["luminance"] = luminance
        state.data["valid_mask"] = valid_mask
        return state


class ComputeLocalDarkness(CVProcessor):
    def __init__(
        self,
        background_sigma: float,
        local_stats_window: int,
        zscore_clip: float,
        eps: float,
    ) -> None:
        if background_sigma <= 0:
            raise ValueError("background_sigma must be positive")
        if local_stats_window <= 0 or local_stats_window % 2 == 0:
            raise ValueError("local_stats_window must be a positive odd integer")
        if zscore_clip <= 0:
            raise ValueError("zscore_clip must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.background_sigma = float(background_sigma)
        self.local_stats_window = int(local_stats_window)
        self.zscore_clip = float(zscore_clip)
        self.eps = float(eps)

    def process(self, state: ProcessingState) -> ProcessingState:
        luminance = _require_array(state, "luminance")

        background = cv2.GaussianBlur(
            luminance,
            (0, 0),
            sigmaX=self.background_sigma,
            borderType=cv2.BORDER_REFLECT,
        ).astype(np.float32, copy=False)

        darkness_difference = np.maximum(background - luminance, 0).astype(np.float32)
        local_log_darkness = np.maximum(
            np.log(background + self.eps) - np.log(luminance + self.eps),
            0,
        ).astype(np.float32)

        ksize = (self.local_stats_window, self.local_stats_window)
        local_mean = cv2.boxFilter(
            luminance,
            ddepth=-1,
            ksize=ksize,
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        local_mean_square = cv2.boxFilter(
            luminance * luminance,
            ddepth=-1,
            ksize=ksize,
            normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        local_variance = np.maximum(local_mean_square - local_mean * local_mean, 0)
        local_std = np.sqrt(local_variance + self.eps)
        local_zscore = np.maximum((local_mean - luminance) / local_std, 0)
        local_zscore = np.clip(local_zscore, 0, self.zscore_clip).astype(np.float32)

        state.data["background"] = background
        state.data["darkness_difference"] = darkness_difference
        state.data["local_log_darkness"] = local_log_darkness
        state.data["local_zscore"] = local_zscore
        return state


class MultiScaleBlackHat(CVProcessor):
    def __init__(
        self,
        kernel_sizes: Iterable[int],
        persistence_threshold: float,
        lower_percentile: float,
        upper_percentile: float,
        min_dynamic_range: float,
        max_percentile_samples: int,
        eps: float,
    ) -> None:
        self.kernel_sizes = [int(size) for size in kernel_sizes]
        if not self.kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")
        for size in self.kernel_sizes:
            if size <= 0 or size % 2 == 0:
                raise ValueError("all black-hat kernel_sizes must be positive odd integers")
        if not 0.0 <= persistence_threshold <= 1.0:
            raise ValueError("persistence_threshold must be in [0, 1]")

        self.persistence_threshold = float(persistence_threshold)
        self.lower_percentile = float(lower_percentile)
        self.upper_percentile = float(upper_percentile)
        self.min_dynamic_range = float(min_dynamic_range)
        self.max_percentile_samples = int(max_percentile_samples)
        self.eps = float(eps)
        _validate_percentile_params(
            self.lower_percentile,
            self.upper_percentile,
            self.min_dynamic_range,
            self.max_percentile_samples,
            self.eps,
        )

    def process(self, state: ProcessingState) -> ProcessingState:
        luminance = _require_array(state, "luminance")
        valid_mask = _require_bool_array(state, "valid_mask")

        blackhat_max = np.zeros(luminance.shape, dtype=np.float32)
        persistence_count = np.zeros(luminance.shape, dtype=np.uint16)

        for size in self.kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            closed = cv2.morphologyEx(luminance, cv2.MORPH_CLOSE, kernel)
            response = np.maximum(closed - luminance, 0).astype(np.float32)
            normalized = _robust_normalize(
                response,
                valid_mask,
                lower_percentile=self.lower_percentile,
                upper_percentile=self.upper_percentile,
                min_dynamic_range=self.min_dynamic_range,
                max_percentile_samples=self.max_percentile_samples,
                eps=self.eps,
            )
            blackhat_max = np.maximum(blackhat_max, normalized)
            persistence_count += (normalized >= self.persistence_threshold).astype(np.uint16)

        blackhat_persistence = (
            persistence_count.astype(np.float32) / np.float32(len(self.kernel_sizes))
        )
        blackhat_persistence[~valid_mask] = 0

        state.data["blackhat_max"] = blackhat_max
        state.data["blackhat_persistence"] = blackhat_persistence
        return state


class NormalizeFeatures(CVProcessor):
    def __init__(
        self,
        features: Iterable[str],
        lower_percentile: float,
        upper_percentile: float,
        min_dynamic_range: float,
        max_percentile_samples: int,
        eps: float,
    ) -> None:
        self.features = list(features)
        if not self.features:
            raise ValueError("features must not be empty")
        self.lower_percentile = float(lower_percentile)
        self.upper_percentile = float(upper_percentile)
        self.min_dynamic_range = float(min_dynamic_range)
        self.max_percentile_samples = int(max_percentile_samples)
        self.eps = float(eps)
        _validate_percentile_params(
            self.lower_percentile,
            self.upper_percentile,
            self.min_dynamic_range,
            self.max_percentile_samples,
            self.eps,
        )

    def process(self, state: ProcessingState) -> ProcessingState:
        valid_mask = _require_bool_array(state, "valid_mask")
        for feature_name in self.features:
            feature = _require_array(state, feature_name)
            state.data[f"{feature_name}_norm"] = _robust_normalize(
                feature,
                valid_mask,
                lower_percentile=self.lower_percentile,
                upper_percentile=self.upper_percentile,
                min_dynamic_range=self.min_dynamic_range,
                max_percentile_samples=self.max_percentile_samples,
                eps=self.eps,
            )
        return state


class ConfidenceFusion(CVProcessor):
    FEATURE_KEYS = (
        "local_log_darkness_norm",
        "local_zscore_norm",
        "blackhat_max_norm",
        "blackhat_persistence",
    )

    PRIMARY_FEATURE_KEYS = (
        "local_log_darkness_norm",
        "local_zscore_norm",
        "blackhat_max_norm",
    )

    def __init__(
        self,
        weights: dict[str, float],
        vote_thresholds: dict[str, float],
        max_weight: float = 0.65,
        mean_weight: float = 0.35,
    ) -> None:
        self.weights = _validated_feature_mapping(
            weights,
            self.FEATURE_KEYS,
            "weights",
        )
        self.vote_thresholds = _validated_feature_mapping(
            vote_thresholds,
            self.FEATURE_KEYS,
            "vote_thresholds",
        )

        weight_sum = sum(self.weights.values())
        if weight_sum <= 0:
            raise ValueError("sum of feature weights must be positive")

        self.weights = {
            key: value / weight_sum
            for key, value in self.weights.items()
        }

        if max_weight < 0 or mean_weight < 0:
            raise ValueError("max_weight and mean_weight must be non-negative")

        fusion_weight_sum = max_weight + mean_weight
        if fusion_weight_sum <= 0:
            raise ValueError("sum of fusion weights must be positive")

        self.max_weight = max_weight / fusion_weight_sum
        self.mean_weight = mean_weight / fusion_weight_sum

    def process(self, state: ProcessingState) -> ProcessingState:
        valid_mask = _require_bool_array(state, "valid_mask")
        shape = valid_mask.shape

        weighted_mean = np.zeros(shape, dtype=np.float32)
        evidence_count = np.zeros(shape, dtype=np.uint8)

        features: dict[str, np.ndarray] = {}

        for feature_name in self.FEATURE_KEYS:
            feature = _require_array(state, feature_name).astype(
                np.float32,
                copy=False,
            )

            if feature.shape != shape:
                raise ValueError(
                    f"Feature '{feature_name}' has shape {feature.shape}, "
                    f"expected {shape}"
                )

            features[feature_name] = feature

            weighted_mean += (
                np.float32(self.weights[feature_name]) * feature
            )

            evidence_count += (
                feature >= self.vote_thresholds[feature_name]
            ).astype(np.uint8)

        strongest_response = np.maximum.reduce(
            [
                features[feature_name]
                for feature_name in self.PRIMARY_FEATURE_KEYS
            ]
        )

        confidence = (
            np.float32(self.max_weight) * strongest_response
            + np.float32(self.mean_weight) * weighted_mean
        )

        confidence = np.clip(confidence, 0.0, 1.0)

        confidence[~valid_mask] = 0.0
        strongest_response[~valid_mask] = 0.0
        evidence_count[~valid_mask] = 0

        state.data["weighted_mean_confidence"] = weighted_mean
        state.data["strongest_response"] = strongest_response
        state.data["confidence"] = confidence
        state.data["evidence_count"] = evidence_count

        return state


class HysteresisSegmentation(CVProcessor):
    def __init__(
        self,
        seed_threshold: float,
        grow_threshold: float,
        seed_min_evidence: int,
        grow_min_evidence: int,
        min_seed_area: int,
        connectivity: int,
        strong_response_threshold: float | None = None,
    ) -> None:
        if not grow_threshold < seed_threshold:
            raise ValueError(
                "grow_threshold must be lower than seed_threshold"
            )
        if grow_min_evidence > seed_min_evidence:
            raise ValueError(
                "grow_min_evidence must be <= seed_min_evidence"
            )
        if min_seed_area < 1:
            raise ValueError("min_seed_area must be positive")
        if (
            strong_response_threshold is not None
            and not 0.0 <= strong_response_threshold <= 1.0
        ):
            raise ValueError(
                "strong_response_threshold must be in [0, 1]"
            )

        _validate_connectivity(connectivity)

        self.seed_threshold = float(seed_threshold)
        self.grow_threshold = float(grow_threshold)
        self.seed_min_evidence = int(seed_min_evidence)
        self.grow_min_evidence = int(grow_min_evidence)
        self.min_seed_area = int(min_seed_area)
        self.connectivity = int(connectivity)
        self.strong_response_threshold = strong_response_threshold

    def process(
        self,
        state: ProcessingState,
    ) -> ProcessingState:
        confidence = _require_array(state, "confidence")
        evidence_count = _require_array(state, "evidence_count")
        valid_mask = _require_bool_array(state, "valid_mask")

        seed_mask = (
            (confidence >= self.seed_threshold)
            & (evidence_count >= self.seed_min_evidence)
        )

        if self.strong_response_threshold is not None:
            strongest_response = _require_array(
                state,
                "strongest_response",
            )
            seed_mask |= (
                strongest_response
                >= self.strong_response_threshold
            )

        seed_mask &= valid_mask

        seed_mask = _remove_small_components(
            seed_mask,
            self.min_seed_area,
            self.connectivity,
        )

        grow_mask = (
            (confidence >= self.grow_threshold)
            & (evidence_count >= self.grow_min_evidence)
            & valid_mask
        )

        if not np.any(seed_mask):
            candidate_mask = np.zeros(
                valid_mask.shape,
                dtype=bool,
            )
        else:
            num_labels, labels = cv2.connectedComponents(
                grow_mask.astype(np.uint8),
                connectivity=self.connectivity,
            )

            accepted_labels = np.unique(labels[seed_mask])
            accepted_labels = accepted_labels[
                accepted_labels != 0
            ]

            lookup = np.zeros(num_labels, dtype=bool)
            lookup[accepted_labels] = True
            candidate_mask = lookup[labels]

        state.data["seed_mask"] = seed_mask
        state.data["grow_mask"] = grow_mask
        state.data["candidate_mask"] = candidate_mask
        return state


class MorphologyCleanup(CVProcessor):
    def __init__(self, operations: list[dict[str, Any]], connectivity: int) -> None:
        if not isinstance(operations, list):
            raise ValueError("operations must be a list")
        _validate_connectivity(connectivity)
        self.operations = operations
        self.connectivity = int(connectivity)

    def process(self, state: ProcessingState) -> ProcessingState:
        mask = _require_bool_array(state, "candidate_mask").copy()
        valid_mask = _require_bool_array(state, "valid_mask")

        for operation in self.operations:
            if not operation.get("enabled", True):
                continue
            operation_type = operation.get("type")
            if operation_type == "remove_small_objects":
                mask = _remove_small_components(
                    mask,
                    min_area=int(operation["min_area"]),
                    connectivity=self.connectivity,
                )
            elif operation_type == "fill_small_holes":
                mask = _fill_small_holes(
                    mask,
                    max_area=int(operation["max_area"]),
                    connectivity=self.connectivity,
                )
            elif operation_type == "opening":
                mask = _morphology(mask, cv2.MORPH_OPEN, int(operation["kernel_size"]))
            elif operation_type == "closing":
                mask = _morphology(mask, cv2.MORPH_CLOSE, int(operation["kernel_size"]))
            else:
                raise ValueError(f"unknown morphology operation: {operation_type!r}")

        state.data["cleaned_candidate_mask"] = mask & valid_mask
        return state


class ComponentFilter(CVProcessor):
    def __init__(
        self,
        min_area: int,
        max_area: int,
        max_elongation: float,
        min_solidity: float,
        ring_radius: int,
        min_ring_pixels: int,
        insufficient_ring_policy: str,
        min_ring_contrast: float,
        min_median_confidence: float,
        min_persistence: float,
        reject_border_touching: bool,
        connectivity: int,
        eps: float,
        shape_filter_min_area: int,
    ) -> None:
        if min_area < 1:
            raise ValueError("min_area must be positive")
        if max_area < min_area:
            raise ValueError("max_area must be >= min_area")
        if max_elongation <= 0:
            raise ValueError("max_elongation must be positive")
        if not 0 <= min_solidity <= 1:
            raise ValueError("min_solidity must be in [0, 1]")
        if ring_radius < 1:
            raise ValueError("ring_radius must be positive")
        if min_ring_pixels < 0:
            raise ValueError("min_ring_pixels must be non-negative")
        if insufficient_ring_policy not in {"skip_filter", "reject"}:
            raise ValueError("insufficient_ring_policy must be 'skip_filter' or 'reject'")
        _validate_connectivity(connectivity)
        if eps <= 0:
            raise ValueError("eps must be positive")
        if shape_filter_min_area < 1:
            raise ValueError(
                "shape_filter_min_area must be positive"
            )

        self.shape_filter_min_area = int(shape_filter_min_area)
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.max_elongation = float(max_elongation)
        self.min_solidity = float(min_solidity)
        self.ring_radius = int(ring_radius)
        self.min_ring_pixels = int(min_ring_pixels)
        self.insufficient_ring_policy = insufficient_ring_policy
        self.min_ring_contrast = float(min_ring_contrast)
        self.min_median_confidence = float(min_median_confidence)
        self.min_persistence = float(min_persistence)
        self.reject_border_touching = bool(reject_border_touching)
        self.connectivity = int(connectivity)
        self.eps = float(eps)

    def process(self, state: ProcessingState) -> ProcessingState:
        mask = _require_bool_array(state, "cleaned_candidate_mask")
        luminance = _require_array(state, "luminance")
        valid_mask = _require_bool_array(state, "valid_mask")
        confidence = _require_array(state, "confidence")
        persistence = _require_array(state, "blackhat_persistence")

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=self.connectivity,
        )
        log_luminance = np.log(luminance + self.eps).astype(np.float32)

        component_metrics: list[dict[str, Any]] = []
        accepted_labels: list[int] = []

        for label_id in range(1, num_labels):
            x_min = int(stats[label_id, cv2.CC_STAT_LEFT])
            y_min = int(stats[label_id, cv2.CC_STAT_TOP])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            if width <= 0 or height <= 0:
                continue
            x_max = x_min + width
            y_max = y_min + height
            local_labels = labels[y_min:y_max, x_min:x_max]
            ys_local, xs_local = np.where(local_labels == label_id)
            if ys_local.size == 0:
                continue
            ys = ys_local + y_min
            xs = xs_local + x_min

            metrics = self._measure_component(
                label_id=label_id,
                ys=ys,
                xs=xs,
                labels=labels,
                candidate_mask=mask,
                valid_mask=valid_mask,
                log_luminance=log_luminance,
                confidence=confidence,
                persistence=persistence,
            )
            accepted = self._passes_component(metrics)
            metrics["accepted"] = accepted
            component_metrics.append(metrics)
            if accepted:
                accepted_labels.append(label_id)

        lookup = np.zeros(num_labels, dtype=bool)
        lookup[np.asarray(accepted_labels, dtype=np.int32)] = True
        final_mask = lookup[labels]

        state.data["final_mask"] = final_mask
        state.metadata["component_metrics"] = component_metrics
        return state

    def _measure_component(
        self,
        label_id: int,
        ys: np.ndarray,
        xs: np.ndarray,
        labels: np.ndarray,
        candidate_mask: np.ndarray,
        valid_mask: np.ndarray,
        log_luminance: np.ndarray,
        confidence: np.ndarray,
        persistence: np.ndarray,
    ) -> dict[str, Any]:
        area = int(ys.size)
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        centroid = (float(xs.mean()), float(ys.mean()))

        component_values = labels[ys, xs] == label_id
        if not np.all(component_values):
            raise RuntimeError("internal label indexing error")

        ring_contrast, ring_pixels, ring_filter_applicable = self._ring_contrast(
            label_id=label_id,
            bbox=(x_min, y_min, x_max, y_max),
            labels=labels,
            candidate_mask=candidate_mask,
            valid_mask=valid_mask,
            log_luminance=log_luminance,
        )

        touches_border = (
            x_min == 0
            or y_min == 0
            or x_max == labels.shape[1]
            or y_max == labels.shape[0]
        )

        return {
            "label": label_id,
            "area": area,
            "bbox": bbox,
            "centroid": centroid,
            "elongation": _component_elongation(xs, ys, self.eps),
            "solidity": _component_solidity(labels[y_min:y_max, x_min:x_max] == label_id),
            "ring_contrast": ring_contrast,
            "ring_pixels": ring_pixels,
            "ring_filter_applicable": ring_filter_applicable,
            "median_confidence": float(np.median(confidence[ys, xs])),
            "median_persistence": float(np.median(persistence[ys, xs])),
            "touches_border": touches_border,
        }

    def _ring_contrast(
        self,
        label_id: int,
        bbox: tuple[int, int, int, int],
        labels: np.ndarray,
        candidate_mask: np.ndarray,
        valid_mask: np.ndarray,
        log_luminance: np.ndarray,
    ) -> tuple[float, int, bool]:
        x_min, y_min, x_max, y_max = bbox
        height, width = labels.shape
        roi_x0 = max(0, x_min - self.ring_radius)
        roi_y0 = max(0, y_min - self.ring_radius)
        roi_x1 = min(width, x_max + self.ring_radius)
        roi_y1 = min(height, y_max + self.ring_radius)

        label_roi = labels[roi_y0:roi_y1, roi_x0:roi_x1]
        component_roi = label_roi == label_id
        kernel_size = 2 * self.ring_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(component_roi.astype(np.uint8), kernel).astype(bool)

        ring_roi = (
            dilated
            & ~component_roi
            & valid_mask[roi_y0:roi_y1, roi_x0:roi_x1]
            & ~candidate_mask[roi_y0:roi_y1, roi_x0:roi_x1]
        )

        ring_pixels = int(np.count_nonzero(ring_roi))
        if ring_pixels < self.min_ring_pixels:
            return np.nan, ring_pixels, False

        log_roi = log_luminance[roi_y0:roi_y1, roi_x0:roi_x1]
        component_value = float(np.median(log_roi[component_roi]))
        ring_value = float(np.median(log_roi[ring_roi]))
        return ring_value - component_value, ring_pixels, True

    def _passes_component(self, metrics: dict[str, Any]) -> bool:
        return (
            self._passes_area(metrics)
            and self._passes_elongation(metrics)
            and self._passes_solidity(metrics)
            and self._passes_ring_contrast(metrics)
            and self._passes_medians(metrics)
            and self._passes_border(metrics)
        )

    def _passes_area(self, metrics: dict[str, Any]) -> bool:
        return self.min_area <= metrics["area"] <= self.max_area

    def _passes_elongation(
        self,
        metrics: dict[str, Any],
    ) -> bool:
        if metrics["area"] < self.shape_filter_min_area:
            return True

        return (
            metrics["elongation"]
            <= self.max_elongation
        )

    def _passes_solidity(
        self,
        metrics: dict[str, Any],
    ) -> bool:
        if metrics["area"] < self.shape_filter_min_area:
            return True

        return metrics["solidity"] >= self.min_solidity

    def _passes_ring_contrast(self, metrics: dict[str, Any]) -> bool:
        if not metrics["ring_filter_applicable"]:
            return self.insufficient_ring_policy == "skip_filter"
        return metrics["ring_contrast"] >= self.min_ring_contrast

    def _passes_medians(self, metrics: dict[str, Any]) -> bool:
        return (
            metrics["median_confidence"] >= self.min_median_confidence
            and metrics["median_persistence"] >= self.min_persistence
        )

    def _passes_border(self, metrics: dict[str, Any]) -> bool:
        return not (self.reject_border_touching and metrics["touches_border"])


PROCESSOR_TYPES = {
    "prepare_image": PrepareImage,
    "local_darkness": ComputeLocalDarkness,
    "multiscale_blackhat": MultiScaleBlackHat,
    "normalize_features": NormalizeFeatures,
    "confidence_fusion": ConfidenceFusion,
    "hysteresis": HysteresisSegmentation,
    "morphology_cleanup": MorphologyCleanup,
    "component_filter": ComponentFilter,
}


class TalcCVPipeline:
    def __init__(self, processors: list[CVProcessor]) -> None:
        self.processors = processors

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TalcCVPipeline":
        with Path(path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TalcCVPipeline":
        if not isinstance(config, dict):
            raise ValueError("YAML config must contain a mapping")
        pipeline_config = config.get("pipeline")
        if not isinstance(pipeline_config, list):
            raise ValueError("YAML config must contain a 'pipeline' list")

        processors: list[CVProcessor] = []
        for index, item in enumerate(pipeline_config):
            if not isinstance(item, dict):
                raise ValueError(f"pipeline item {index} must be a mapping")
            if not item.get("enabled", True):
                continue
            processor_type = item.get("type")
            processor_cls = PROCESSOR_TYPES.get(processor_type)
            if processor_cls is None:
                raise ValueError(f"unknown processor type: {processor_type!r}")
            params = item.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(f"params for processor {processor_type!r} must be a mapping")
            processors.append(processor_cls(**params))

        return cls(processors)

    def __call__(
        self,
        image_rgb: np.ndarray,
        segformer_mask: np.ndarray,
    ) -> np.ndarray:
        data = {"segformer_mask": np.asarray(segformer_mask)}
        state = ProcessingState(image_rgb=image_rgb, data=data, metadata={})
        for processor in self.processors:
            state = processor(state)
        if "final_mask" not in state.data:
            raise KeyError("pipeline did not produce state.data['final_mask']")
        return state.data["final_mask"].astype(np.uint8, copy=False)


def _to_float_rgb(image: np.ndarray, input_color: str) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")

    finite_mask = np.all(np.isfinite(array), axis=2)
    safe = np.where(np.isfinite(array), array, 0)

    if safe.dtype == np.uint8:
        image_float = safe.astype(np.float32) / np.float32(255.0)
    elif safe.dtype == np.uint16:
        image_float = safe.astype(np.float32) / np.float32(65535.0)
    elif np.issubdtype(safe.dtype, np.floating):
        image_float = safe.astype(np.float32, copy=True)
        max_value = float(np.nanmax(image_float)) if image_float.size else 0.0
        min_value = float(np.nanmin(image_float)) if image_float.size else 0.0
        if min_value < 0:
            raise ValueError("floating image values must be non-negative")
        if max_value <= 1.0:
            pass
        elif max_value <= 255.0:
            image_float /= np.float32(255.0)
        else:
            raise ValueError("floating image values must be in [0, 1] or [0, 255]")
    else:
        raise TypeError("image dtype must be uint8, uint16, float32, or float64")

    image_float = np.clip(image_float, 0, 1).astype(np.float32, copy=False)
    if input_color == "bgr":
        image_float = image_float[..., ::-1].copy()
    return image_float, finite_mask


def _to_bool_mask(mask: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.shape != expected_shape:
        raise ValueError(
            "segformer_mask must have shape HxW matching image_rgb; "
            f"got {array.shape}, expected {expected_shape}"
        )

    if array.dtype == bool:
        return array.copy()
    if np.issubdtype(array.dtype, np.floating):
        return (np.isfinite(array) & (array > 0.5)).astype(bool, copy=False)
    if np.issubdtype(array.dtype, np.integer):
        return array != 0
    raise TypeError("segformer_mask dtype must be bool, integer, or floating")


def _robust_normalize(
    feature: np.ndarray,
    valid_mask: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
    min_dynamic_range: float,
    max_percentile_samples: int,
    eps: float,
) -> np.ndarray:
    output = np.zeros(feature.shape, dtype=np.float32)
    finite_valid = valid_mask & np.isfinite(feature)
    values = feature[finite_valid]
    if values.size == 0:
        return output

    if values.size > max_percentile_samples:
        indices = np.linspace(
            0,
            values.size - 1,
            num=max_percentile_samples,
            dtype=np.int64,
        )
        values_for_percentiles = values[indices]
    else:
        values_for_percentiles = values

    low = np.percentile(values_for_percentiles, lower_percentile)
    high = np.percentile(values_for_percentiles, upper_percentile)
    if not np.isfinite(low) or not np.isfinite(high) or high - low < min_dynamic_range:
        return output

    output[finite_valid] = np.clip(
        (feature[finite_valid] - low) / (high - low + eps),
        0,
        1,
    )
    return output


def _validate_percentile_params(
    lower_percentile: float,
    upper_percentile: float,
    min_dynamic_range: float,
    max_percentile_samples: int,
    eps: float,
) -> None:
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    if min_dynamic_range <= 0:
        raise ValueError("min_dynamic_range must be positive")
    if max_percentile_samples < 1:
        raise ValueError("max_percentile_samples must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")


def _remove_small_components(mask: np.ndarray, min_area: int, connectivity: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(bool, copy=False)
    _validate_connectivity(connectivity)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=connectivity,
    )
    if num_labels <= 1:
        return np.zeros(mask.shape, dtype=bool)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


def _fill_small_holes(mask: np.ndarray, max_area: int, connectivity: int) -> np.ndarray:
    if max_area <= 0:
        return mask.astype(bool, copy=False)
    _validate_connectivity(connectivity)
    inverse = ~mask.astype(bool, copy=False)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverse.astype(np.uint8),
        connectivity=connectivity,
    )
    if num_labels <= 1:
        return mask.astype(bool, copy=False)

    touches_border = np.zeros(num_labels, dtype=bool)
    touches_border[np.unique(labels[0, :])] = True
    touches_border[np.unique(labels[-1, :])] = True
    touches_border[np.unique(labels[:, 0])] = True
    touches_border[np.unique(labels[:, -1])] = True

    areas = stats[:, cv2.CC_STAT_AREA]
    fill = (areas <= max_area) & ~touches_border
    fill[0] = False
    result = mask.astype(bool, copy=True)
    result[fill[labels]] = True
    return result


def _morphology(mask: np.ndarray, operation: int, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("morphology kernel_size must be a positive odd integer")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel).astype(bool)


def _component_elongation(xs: np.ndarray, ys: np.ndarray, eps: float) -> float:
    if xs.size < 2:
        return 1.0
    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    centered = coords - coords.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / np.float32(max(coords.shape[0] - 1, 1))
    eigenvalues = np.linalg.eigvalsh(covariance)
    lambda_min = float(max(eigenvalues[0], 0.0))
    lambda_max = float(max(eigenvalues[-1], 0.0))
    return float(np.sqrt(lambda_max / (lambda_min + eps)))


def _component_solidity(component_mask: np.ndarray) -> float:
    component_uint8 = component_mask.astype(np.uint8)
    contours, _ = cv2.findContours(component_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    points = np.concatenate(contours, axis=0)
    if points.shape[0] < 3:
        return 1.0
    hull = cv2.convexHull(points)
    hull_mask = np.zeros(component_mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    hull_area = int(np.count_nonzero(hull_mask))
    if hull_area == 0:
        return 1.0
    area = int(np.count_nonzero(component_mask))
    return float(min(area / hull_area, 1.0))


def _validated_feature_mapping(
    values: dict[str, float],
    expected_keys: tuple[str, ...],
    name: str,
) -> dict[str, float]:
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a mapping")
    missing = set(expected_keys) - set(values)
    extra = set(values) - set(expected_keys)
    if missing:
        raise ValueError(f"{name} missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"{name} has unknown keys: {sorted(extra)}")
    return {key: float(values[key]) for key in expected_keys}


def _validate_connectivity(connectivity: int) -> None:
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")


def _require_array(state: ProcessingState, key: str) -> np.ndarray:
    if key not in state.data:
        raise KeyError(f"state.data[{key!r}] is required")
    return state.data[key]


def _require_bool_array(state: ProcessingState, key: str) -> np.ndarray:
    array = _require_array(state, key)
    if array.dtype == bool:
        return array
    return array.astype(bool)


__all__ = [
    "CVProcessor",
    "ComponentFilter",
    "ComputeLocalDarkness",
    "ConfidenceFusion",
    "HysteresisSegmentation",
    "MorphologyCleanup",
    "MultiScaleBlackHat",
    "NormalizeFeatures",
    "PrepareImage",
    "ProcessingState",
    "PROCESSOR_TYPES",
    "TalcCVPipeline",
]
