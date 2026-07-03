from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .config import DataConfig, ModelConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class TileWindow:
    sample_index: int
    x: int
    y: int
    width: int
    height: int


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    """Return starts that cover an axis completely, including its far edge."""

    if length <= 0 or tile_size <= 0:
        raise ValueError("length and tile_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if length <= tile_size:
        return [0]
    last_start = length - tile_size
    target_stride = max(1.0, tile_size * (1 - overlap))
    ratio = last_start / target_stride
    minimum_intervals = max(1, math.ceil(last_start / tile_size))
    candidates = {
        max(minimum_intervals, math.floor(ratio)),
        max(minimum_intervals, math.ceil(ratio)),
    }
    intervals = min(
        candidates,
        key=lambda count: abs(last_start / count - target_stride),
    )
    return [
        round(index * last_start / intervals)
        for index in range(intervals + 1)
    ]


def make_tile_windows(
    sample_index: int,
    width: int,
    height: int,
    tile_size: int,
    overlap: float,
) -> list[TileWindow]:
    return [
        TileWindow(
            sample_index=sample_index,
            x=x,
            y=y,
            width=min(tile_size, width - x),
            height=min(tile_size, height - y),
        )
        for y in tile_starts(height, tile_size, overlap)
        for x in tile_starts(width, tile_size, overlap)
    ]


def read_class_names(data_dir: Path, classes_file: str) -> list[str]:
    path = data_dir / classes_file
    if not path.exists():
        raise FileNotFoundError(f"Classes file not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names:
        raise ValueError(f"No classes found in {path}")
    return names


def discover_samples(data_dir: Path) -> tuple[list[Sample], int]:
    image_dir = data_dir / "images"
    label_dir = data_dir / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{image_dir}' and '{label_dir}' directories"
        )

    images: dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in images:
                raise ValueError(f"Duplicate image stem: {path.stem}")
            images[path.stem] = path

    samples: list[Sample] = []
    orphan_labels: list[Path] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = images.get(label_path.stem)
        if image_path is None:
            orphan_labels.append(label_path)
            continue
        samples.append(Sample(label_path.stem, image_path, label_path))

    if orphan_labels:
        paths = ", ".join(str(path) for path in orphan_labels[:3])
        raise ValueError(f"Labels without matching images: {paths}")
    if not samples:
        raise ValueError(f"No labeled image/label pairs found in {data_dir}")
    return samples, len(images)


def _stable_score(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def load_or_update_split(
    samples: list[Sample], split_file: Path, val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Keep old assignments stable and allocate newly labeled images as they appear."""

    current_ids = {sample.sample_id for sample in samples}
    train_ids: list[str] = []
    val_ids: list[str] = []
    if split_file.exists():
        payload = json.loads(split_file.read_text(encoding="utf-8"))
        train_ids = [item for item in payload.get("train", []) if item in current_ids]
        val_ids = [item for item in payload.get("val", []) if item in current_ids]

    assigned = set(train_ids) | set(val_ids)
    new_ids = sorted(current_ids - assigned, key=lambda item: _stable_score(item, seed))

    if not train_ids and not val_ids and len(current_ids) > 1:
        desired_val = max(1, round(len(current_ids) * val_fraction))
        desired_val = min(desired_val, len(current_ids) - 1)
        val_ids.extend(new_ids[:desired_val])
        train_ids.extend(new_ids[desired_val:])
    else:
        desired_val = max(1, round(len(current_ids) * val_fraction))
        slots = max(0, desired_val - len(val_ids))
        val_ids.extend(new_ids[:slots])
        train_ids.extend(new_ids[slots:])

    if len(current_ids) == 1:
        only_id = next(iter(current_ids))
        train_ids = [only_id]
        val_ids = [only_id]

    split_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "val_fraction": val_fraction,
        "train": sorted(train_ids),
        "val": sorted(val_ids),
    }
    split_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload["train"], payload["val"]


def yolo_polygons_to_mask(
    label_path: Path,
    height: int,
    width: int,
    task: str = "binary",
    foreground_classes: int = 1,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        values = raw_line.split()
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"Invalid polygon at {label_path}:{line_number}")
        try:
            class_id = int(values[0])
            coordinates = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
        except ValueError as error:
            raise ValueError(
                f"Invalid numeric value at {label_path}:{line_number}"
            ) from error
        if class_id < 0 or class_id >= foreground_classes:
            raise ValueError(
                f"Class id {class_id} is outside [0, {foreground_classes - 1}] "
                f"at {label_path}:{line_number}"
            )
        coordinates[:, 0] = np.clip(coordinates[:, 0], 0, 1) * (width - 1)
        coordinates[:, 1] = np.clip(coordinates[:, 1], 0, 1) * (height - 1)
        polygon = np.rint(coordinates).astype(np.int32)
        value = 1 if task == "binary" else class_id + 1
        cv2.fillPoly(mask, [polygon], color=value)
    return mask


def _normalization(model: ModelConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if model.encoder_weights is None:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    import segmentation_models_pytorch as smp

    params: dict[str, Any] = smp.encoders.get_preprocessing_params(
        model.encoder_name, pretrained=model.encoder_weights
    )
    return tuple(params["mean"]), tuple(params["std"])


def build_train_transform(data: DataConfig, model: ModelConfig) -> A.Compose:
    mean, std = _normalization(model)
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=data.tile_size,
                min_width=data.tile_size,
                border_mode=cv2.BORDER_REFLECT_101,
                fill_mask=0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    A.CLAHE(p=1.0),
                ],
                p=0.5,
            ),
            A.Resize(data.image_size, data.image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def build_val_transform(data: DataConfig, model: ModelConfig) -> A.Compose:
    mean, std = _normalization(model)
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=data.tile_size,
                min_width=data.tile_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),
            A.Resize(data.image_size, data.image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


class YoloTiledSegmentationDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        samples: list[Sample],
        transform: A.Compose,
        task: str,
        foreground_classes: int,
        tile_size: int,
        tile_overlap: float,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.task = task
        self.foreground_classes = foreground_classes
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tiles: list[TileWindow] = []
        for sample_index, sample in enumerate(samples):
            with Image.open(sample.image_path) as source:
                width, height = ImageOps.exif_transpose(source).size
            self.tiles.extend(
                make_tile_windows(
                    sample_index,
                    width,
                    height,
                    tile_size,
                    tile_overlap,
                )
            )

    def __len__(self) -> int:
        return len(self.tiles)

    @lru_cache(maxsize=8)
    def _load_sample(self, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.samples[sample_index]
        with Image.open(sample.image_path) as source:
            image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        height, width = image.shape[:2]
        mask = yolo_polygons_to_mask(
            sample.label_path,
            height,
            width,
            task=self.task,
            foreground_classes=self.foreground_classes,
        )
        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        tile = self.tiles[index]
        sample = self.samples[tile.sample_index]
        image, mask = self._load_sample(tile.sample_index)
        y_slice = slice(tile.y, tile.y + tile.height)
        x_slice = slice(tile.x, tile.x + tile.width)
        tile_image = image[y_slice, x_slice].copy()
        tile_mask = mask[y_slice, x_slice].copy()
        transformed = self.transform(image=tile_image, mask=tile_mask)
        return {
            "image": transformed["image"].float(),
            "mask": transformed["mask"].long(),
            "image_id": sample.sample_id,
            "tile": torch.tensor(
                [tile.x, tile.y, tile.width, tile.height], dtype=torch.int32
            ),
        }


class RepeatDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(self, dataset: Dataset, repeat_factor: int) -> None:
        self.dataset = dataset
        self.repeat_factor = repeat_factor

    def __len__(self) -> int:
        return len(self.dataset) * self.repeat_factor

    def __getitem__(self, index: int):
        return self.dataset[index % len(self.dataset)]


def select_samples(samples: list[Sample], ids: list[str]) -> list[Sample]:
    by_id = {sample.sample_id: sample for sample in samples}
    return [by_id[sample_id] for sample_id in ids]
