from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import torch

from .config import CheckpointConfig
from .models import preprocessing_parameters


class SegmentationMode(str, Enum):
    OVERLAP = "overlap"
    NO_OVERLAP = "no_overlap"


@dataclass(frozen=True)
class TileWindow:
    x: int
    y: int
    width: int
    height: int


@dataclass
class SegmentationResult:
    mask: np.ndarray
    confidence: np.ndarray
    positive_votes: np.ndarray
    vote_count: np.ndarray
    tile_count: int


def _overlap_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    last_start = length - tile_size
    target_stride = max(1.0, tile_size * (1.0 - overlap))
    intervals = max(1, math.ceil(last_start / target_stride))
    return [round(index * last_start / intervals) for index in range(intervals + 1)]


def make_tile_windows(
    width: int,
    height: int,
    tile_size: int,
    mode: SegmentationMode | str,
    overlap: float,
) -> list[TileWindow]:
    selected_mode = SegmentationMode(mode)
    if width <= 0 or height <= 0 or tile_size <= 0:
        raise ValueError("Image dimensions and tile_size must be positive")
    if selected_mode is SegmentationMode.NO_OVERLAP:
        x_starts = list(range(0, width, tile_size))
        y_starts = list(range(0, height, tile_size))
    else:
        if not 0 < overlap < 1:
            raise ValueError("overlap must be between 0 and 1")
        x_starts = _overlap_starts(width, tile_size, overlap)
        y_starts = _overlap_starts(height, tile_size, overlap)
    return [
        TileWindow(
            x=x,
            y=y,
            width=min(tile_size, width - x),
            height=min(tile_size, height - y),
        )
        for y in y_starts
        for x in x_starts
    ]


def merge_tile_predictions(
    width: int,
    height: int,
    predictions: list[tuple[TileWindow, np.ndarray]],
    threshold: float,
    blend_margin: int = 0,
) -> SegmentationResult:
    probability_sum = np.zeros((height, width), dtype=np.float32)
    probability_weighted_sum = np.zeros((height, width), dtype=np.float32)
    positive_weight = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    positive_votes = np.zeros((height, width), dtype=np.uint32)
    vote_count = np.zeros((height, width), dtype=np.uint32)
    for window, probability in predictions:
        if probability.shape != (window.height, window.width):
            raise ValueError("Tile probability shape does not match its window")
        ys = slice(window.y, window.y + window.height)
        xs = slice(window.x, window.x + window.width)
        probability_float = probability.astype(np.float32, copy=False)
        positive = probability_float >= threshold
        probability_sum[ys, xs] += probability_float
        positive_votes[ys, xs] += positive
        vote_count[ys, xs] += 1
        weights = _tile_reliability(
            window,
            image_width=width,
            image_height=height,
            blend_margin=blend_margin,
        )
        probability_weighted_sum[ys, xs] += probability_float * weights
        positive_weight[ys, xs] += positive * weights
        weight_sum[ys, xs] += weights
    if np.any(vote_count == 0):
        raise RuntimeError("Tile windows did not cover the entire image")
    if np.any(weight_sum <= 0):
        raise RuntimeError("Tile blending weights did not cover the entire image")
    confidence = (
        probability_weighted_sum / weight_sum
        if blend_margin > 0
        else probability_sum / vote_count
    )
    if blend_margin > 0:
        doubled_weight = positive_weight * 2.0
        mask = (doubled_weight > weight_sum) | (
            np.isclose(doubled_weight, weight_sum, rtol=0.0, atol=1.0e-6)
            & (confidence >= threshold)
        )
    else:
        doubled = positive_votes * 2
        mask = (doubled > vote_count) | (
            (doubled == vote_count) & (confidence >= threshold)
        )
    max_votes = int(vote_count.max())
    count_dtype = np.uint16 if max_votes <= np.iinfo(np.uint16).max else np.uint32
    return SegmentationResult(
        mask=mask.astype(np.uint8),
        confidence=confidence.astype(np.float32),
        positive_votes=positive_votes.astype(count_dtype),
        vote_count=vote_count.astype(count_dtype),
        tile_count=len(predictions),
    )


def _tile_reliability(
    window: TileWindow,
    *,
    image_width: int,
    image_height: int,
    blend_margin: int,
) -> np.ndarray:
    """Reduce neural edge artefacts while keeping image-border predictions intact."""

    weights_y = np.ones(window.height, dtype=np.float32)
    weights_x = np.ones(window.width, dtype=np.float32)
    margin_x = min(max(0, blend_margin), max(0, window.width // 2))
    margin_y = min(max(0, blend_margin), max(0, window.height // 2))

    def ramp(length: int) -> np.ndarray:
        # Smoothstep avoids a visible derivative break where the reliable centre starts.
        x = np.linspace(0.0, 1.0, length, endpoint=False, dtype=np.float32)
        return np.maximum(x * x * (3.0 - 2.0 * x), np.float32(1.0e-3))

    if margin_x and window.x > 0:
        weights_x[:margin_x] *= ramp(margin_x)
    if margin_x and window.x + window.width < image_width:
        weights_x[-margin_x:] *= ramp(margin_x)[::-1]
    if margin_y and window.y > 0:
        weights_y[:margin_y] *= ramp(margin_y)
    if margin_y and window.y + window.height < image_height:
        weights_y[-margin_y:] *= ramp(margin_y)[::-1]
    return weights_y[:, None] * weights_x[None, :]


class TileSegmenter:
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint: CheckpointConfig,
        device: torch.device,
        overlap: float,
        batch_size: int | None = None,
    ) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.overlap = overlap
        self.batch_size = batch_size or checkpoint.batch_size
        mean, std = preprocessing_parameters(checkpoint)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def _prepare(
        self,
        tile: np.ndarray,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        tile_size = self.checkpoint.tile_size
        height, width = tile.shape[:2]
        left = (tile_size - width) // 2
        top = (tile_size - height) // 2
        right = tile_size - width - left
        bottom = tile_size - height - top
        padded = cv2.copyMakeBorder(
            tile,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        resized = cv2.resize(
            padded,
            (self.checkpoint.image_size, self.checkpoint.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self.mean) / self.std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
        return tensor, (left, top)

    @torch.inference_mode()
    def predict(
        self,
        image_rgb: np.ndarray,
        mode: SegmentationMode | str,
    ) -> SegmentationResult:
        height, width = image_rgb.shape[:2]
        windows = make_tile_windows(
            width,
            height,
            self.checkpoint.tile_size,
            mode,
            self.overlap,
        )
        predictions: list[tuple[TileWindow, np.ndarray]] = []
        for start in range(0, len(windows), self.batch_size):
            batch_windows = windows[start : start + self.batch_size]
            tensors: list[torch.Tensor] = []
            paddings: list[tuple[int, int]] = []
            for window in batch_windows:
                tile = image_rgb[
                    window.y : window.y + window.height,
                    window.x : window.x + window.width,
                ]
                tensor, padding = self._prepare(tile)
                tensors.append(tensor)
                paddings.append(padding)
            logits = self.model(torch.stack(tensors).to(self.device))
            probabilities = logits.sigmoid().float().cpu().numpy()[:, 0]
            for index, (window, (left, top)) in enumerate(
                zip(batch_windows, paddings, strict=True)
            ):
                full_probability = cv2.resize(
                    probabilities[index],
                    (self.checkpoint.tile_size, self.checkpoint.tile_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                probability = full_probability[
                    top : top + window.height,
                    left : left + window.width,
                ]
                predictions.append((window, probability))
        return merge_tile_predictions(
            width,
            height,
            predictions,
            self.checkpoint.threshold,
            blend_margin=(
                max(1, round(self.checkpoint.tile_size * self.overlap * 0.5))
                if SegmentationMode(mode) is SegmentationMode.OVERLAP
                else 0
            ),
        )
