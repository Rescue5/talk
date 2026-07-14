"""PyTorch datasets and view sampling for ore image classification."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .inspection import apply_duplicate_group_ids, build_dataset_index
from .utils import (
    CLASS_TO_IDX,
    deterministic_crop_boxes,
    dhash_image,
    random_crop_box,
    read_image_rgb,
    read_rows_csv,
    write_rows_csv,
)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _is_readable_row(row: dict[str, Any]) -> bool:
    return str(row.get("readable", "True")).lower() in {"true", "1"}


def _ensure_perceptual_hashes(rows: list[dict[str, Any]], index_csv: Path | None) -> None:
    changed = False
    for row in rows:
        if row.get("target") not in CLASS_TO_IDX or not _is_readable_row(row) or row.get("dhash"):
            continue
        row["dhash"] = dhash_image(row["file_path"]) or ""
        changed = True
    if changed and index_csv is not None:
        write_rows_csv(rows, index_csv)


def load_or_build_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    index_csv = config["data"].get("index_csv")
    default_index_csv = Path(config["data"].get("output_dir") or "runs/ore_classifier") / "inspection" / "dataset_index.csv"
    group_perceptual_duplicates = bool(config["data"].get("group_perceptual_duplicates", True))
    loaded_index_csv: Path | None = None
    if index_csv and Path(index_csv).exists():
        loaded_index_csv = Path(index_csv)
        rows = read_rows_csv(loaded_index_csv)
    elif default_index_csv.exists():
        loaded_index_csv = default_index_csv
        rows = read_rows_csv(default_index_csv)
    else:
        rows, _summary = build_dataset_index(config, compute_dhash=group_perceptual_duplicates)
        write_rows_csv(rows, default_index_csv)
        loaded_index_csv = default_index_csv
    if group_perceptual_duplicates:
        _ensure_perceptual_hashes(rows, loaded_index_csv)
    apply_duplicate_group_ids(
        rows,
        group_perceptual_duplicates=group_perceptual_duplicates,
    )
    samples = [
        row
        for row in rows
        if row.get("target") in CLASS_TO_IDX and _is_readable_row(row)
    ]
    include_sources = config["data"].get("include_sources")
    if include_sources:
        allowed_sources = {str(source) for source in include_sources}
        samples = [sample for sample in samples if sample["dataset_source"] in allowed_sources]
    return samples


def limit_samples_per_class(samples: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if not limit:
        return samples
    rng = random.Random(seed)
    result: list[dict[str, Any]] = []
    for target in sorted(CLASS_TO_IDX):
        class_samples = [sample for sample in samples if sample["target"] == target]
        rng.shuffle(class_samples)
        result.extend(class_samples[:limit])
    return sorted(result, key=lambda row: row["rel_path"])


def _apply_train_aug(image: np.ndarray, config: dict[str, Any], rng: random.Random) -> np.ndarray:
    aug = config.get("augmentations", {})
    out = image.copy()
    if rng.random() < float(aug.get("horizontal_flip_p", 0.0)):
        out = np.ascontiguousarray(out[:, ::-1])
    if rng.random() < float(aug.get("vertical_flip_p", 0.0)):
        out = np.ascontiguousarray(out[::-1])
    if rng.random() < float(aug.get("rotate90_p", 0.0)):
        out = np.ascontiguousarray(np.rot90(out, rng.randint(0, 3)))
    if rng.random() < float(aug.get("brightness_contrast_p", 0.0)):
        brightness = rng.uniform(-float(aug.get("brightness_limit", 0.0)), float(aug.get("brightness_limit", 0.0))) * 255.0
        contrast = 1.0 + rng.uniform(-float(aug.get("contrast_limit", 0.0)), float(aug.get("contrast_limit", 0.0)))
        out = np.clip(out.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
    if rng.random() < float(aug.get("gamma_p", 0.0)):
        low, high = aug.get("gamma_range", [0.9, 1.1])
        gamma = rng.uniform(float(low), float(high))
        table = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
        out = cv2.LUT(out, table)
    if rng.random() < float(aug.get("gaussian_blur_p", 0.0)):
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if rng.random() < float(aug.get("noise_p", 0.0)):
        noise = rng.normalvariate(0.0, float(aug.get("noise_std", 5.0)))
        noise_map = np.random.normal(0.0, abs(noise) + 1.0, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise_map, 0, 255).astype(np.uint8)
    if rng.random() < float(aug.get("jpeg_p", 0.0)):
        low, high = aug.get("jpeg_quality", [65, 95])
        quality = int(rng.randint(int(low), int(high)))
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            out = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if rng.random() < float(aug.get("random_grayscale_p", 0.0)):
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        out = np.repeat(gray[:, :, None], 3, axis=2)
    return out


def _to_tensor(image: np.ndarray, color_mode: str) -> torch.Tensor:
    if color_mode == "grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.repeat(gray[:, :, None], 3, axis=2)
    arr = image.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr.astype(np.float32))


def make_views(image: np.ndarray, config: dict[str, Any], train: bool, rng: random.Random) -> torch.Tensor:
    data = config["data"]
    image_size = int(data.get("image_size", 384))
    num_local_crops = int(data.get("num_local_crops", 8))
    color_mode = data.get("color_mode", "rgb")
    height, width = image.shape[:2]

    views: list[torch.Tensor] = []
    global_view = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if train:
        global_view = _apply_train_aug(global_view, config, rng)
    views.append(_to_tensor(global_view, color_mode))

    if train:
        boxes = [random_crop_box(width, height, data.get("train_crop_scale", [0.35, 0.75]), rng) for _ in range(num_local_crops)]
    else:
        boxes = deterministic_crop_boxes(width, height, num_local_crops, float(data.get("val_crop_scale", 0.55)))

    for x1, y1, x2, y2 in boxes:
        crop = image[y1:y2, x1:x2]
        crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
        if train:
            crop = _apply_train_aug(crop, config, rng)
        views.append(_to_tensor(crop, color_mode))
    return torch.stack(views, dim=0)


class OreImageDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], config: dict[str, Any], train: bool, seed: int = 42):
        self.samples = samples
        self.config = config
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = read_image_rgb(sample["file_path"])
        rng = random.Random(self.seed + index if not self.train else random.randint(0, 2**31 - 1))
        views = make_views(image, self.config, self.train, rng)
        label = torch.tensor(float(CLASS_TO_IDX[sample["target"]]), dtype=torch.float32)
        return views, label, torch.tensor(index, dtype=torch.long)
