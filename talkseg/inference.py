from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from .config import ExperimentConfig
from .data import IMAGE_EXTENSIONS, _normalization, make_tile_windows
from .models import build_model
from .training import choose_device


def _pad_tile(
    tile: np.ndarray, tile_size: int
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = tile.shape[:2]
    left = (tile_size - width) // 2
    top = (tile_size - height) // 2
    right = tile_size - width - left
    bottom = tile_size - height - top
    padded = cv2.copyMakeBorder(
        tile, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )
    return padded, (left, top)


def _blend_weight(height: int, width: int) -> np.ndarray:
    """Center-weight overlapping predictions while keeping image edges non-zero."""

    y_weight = np.sin(np.linspace(0, np.pi, height, dtype=np.float32))
    x_weight = np.sin(np.linspace(0, np.pi, width, dtype=np.float32))
    return np.maximum(np.outer(y_weight, x_weight), 0.05)


def _prepare_tensor(
    tile: np.ndarray,
    config: ExperimentConfig,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[torch.Tensor, tuple[int, int]]:
    padded, padding = _pad_tile(tile, config.data.tile_size)
    resized = cv2.resize(
        padded,
        (config.data.image_size, config.data.image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - mean) / std
    return torch.from_numpy(normalized.transpose(2, 0, 1)).float(), padding


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB"))


@torch.inference_mode()
def predict_one(
    model: torch.nn.Module,
    image: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> np.ndarray:
    height, width = image.shape[:2]
    windows = make_tile_windows(
        sample_index=0,
        width=width,
        height=height,
        tile_size=config.data.tile_size,
        overlap=config.data.tile_overlap,
    )
    mean, std = _normalization(config.model)
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    output_channels = int(model.segmentation_head[0].out_channels)
    probability_sum = np.zeros(
        (output_channels, height, width), dtype=np.float32
    )
    weight_sum = np.zeros((height, width), dtype=np.float32)
    batch_size = max(1, config.train.batch_size)

    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        tensors: list[torch.Tensor] = []
        paddings: list[tuple[int, int]] = []
        for window in batch_windows:
            tile = image[
                window.y : window.y + window.height,
                window.x : window.x + window.width,
            ]
            tensor, padding = _prepare_tensor(
                tile, config, mean_array, std_array
            )
            tensors.append(tensor)
            paddings.append(padding)

        logits = model(torch.stack(tensors).to(device))
        probabilities = (
            logits.sigmoid() if config.data.task == "binary" else logits.softmax(dim=1)
        ).float().cpu().numpy()

        for index, (window, (left, top)) in enumerate(
            zip(batch_windows, paddings, strict=True)
        ):
            tile_probabilities = np.stack(
                [
                    cv2.resize(
                        channel,
                        (config.data.tile_size, config.data.tile_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for channel in probabilities[index]
                ]
            )
            tile_probabilities = tile_probabilities[
                :,
                top : top + window.height,
                left : left + window.width,
            ]
            weight = _blend_weight(window.height, window.width)
            y_slice = slice(window.y, window.y + window.height)
            x_slice = slice(window.x, window.x + window.width)
            probability_sum[:, y_slice, x_slice] += tile_probabilities * weight
            weight_sum[y_slice, x_slice] += weight

    averaged = probability_sum / np.maximum(weight_sum[None], 1e-7)
    if config.data.task == "binary":
        return (averaged[0] >= config.train.threshold).astype(np.uint8)
    return averaged.argmax(axis=0).astype(np.uint8)


def load_checkpoint(path: Path):
    device = choose_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    class_names = checkpoint["class_names"]
    output_channels = 1 if config.data.task == "binary" else len(class_names) + 1
    model = build_model(config.model, output_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, device


def _input_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run segmentation inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, config, device = load_checkpoint(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = _input_paths(args.input)
    if not paths:
        raise ValueError(f"No images found at {args.input}")
    for path in paths:
        image = _load_image(path)
        mask = predict_one(model, image, config, device)
        output = args.output / f"{path.stem}_mask.png"
        multiplier = 255 if config.data.task == "binary" else 1
        if not cv2.imwrite(str(output), mask * multiplier):
            raise OSError(f"Could not write {output}")
        print(output)
