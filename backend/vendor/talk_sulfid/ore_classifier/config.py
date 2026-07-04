"""Configuration helpers for the ore classifier pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "root": "C:/Users/0000/Desktop/projects/talk/data/dataset/dataset",
        "image_size": 384,
        "num_local_crops": 8,
        "split_strategy": "train_val",
        "val_fraction": 0.3,
        "num_folds": 5,
        "exclude_talc": True,
        "include_sources": ["set1", "set2"],
        "group_perceptual_duplicates": True,
        "manual_group_csv": None,
        "index_csv": None,
        "output_dir": "runs/ore_classifier",
        "val_crop_scale": 0.55,
        "train_crop_scale": [0.35, 0.75],
        "color_mode": "rgb",
        "class_mapping": {
            "ordinary": "ordinary",
            "difficult": "difficult",
            "Рядовые руды": "ordinary",
            "рядовые": "ordinary",
            "Труднообогатимые руды": "difficult",
            "тонкие": "difficult",
            "Оталькованные руды": "exclude_talc",
            "оталькованные": "exclude_talc",
            "Панорамы": "exclude_other",
        },
    },
    "model": {
        "backbone": "convnext_tiny",
        "pretrained": True,
        "allow_pretrained_fallback": True,
        "input_channels": 3,
        "pooling": "mean",
        "finetune_mode": "last_stage",
        "dropout": 0.2,
        "hidden_dim": 256,
    },
    "training": {
        "epochs": 30,
        "batch_size": 4,
        "num_workers": 0,
        "backbone_lr": 1.0e-5,
        "head_lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "mixed_precision": True,
        "early_stopping_patience": 7,
        "gradient_clip_norm": 1.0,
        "seed": 42,
        "metric_for_best": "macro_f1",
        "scheduler": "cosine",
        "warmup_epochs": 1,
        "class_weights": True,
        "weighted_sampler": False,
        "max_folds": None,
        "max_samples_per_class": None,
        "experiment_name": None,
        "timestamp_subdir": True,
    },
    "augmentations": {
        "horizontal_flip_p": 0.5,
        "vertical_flip_p": 0.5,
        "rotate90_p": 0.5,
        "brightness_contrast_p": 0.55,
        "brightness_limit": 0.18,
        "contrast_limit": 0.18,
        "gamma_p": 0.35,
        "gamma_range": [0.85, 1.20],
        "gaussian_blur_p": 0.15,
        "noise_p": 0.20,
        "noise_std": 5.0,
        "jpeg_p": 0.15,
        "jpeg_quality": [55, 95],
        "random_grayscale_p": 0.08,
    },
    "reports": {
        "save_examples": True,
        "num_examples": 12,
        "save_plots": True,
        "tensorboard": True,
        "progress_bar": True,
        "save_pt_weights": True,
    },
    "inference": {
        "device": "auto",
        "benchmark_runs": 1,
        "warmup_runs": 0,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating ``base``."""

    result = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = deep_update(config, loaded)
    if overrides:
        config = deep_update(config, overrides)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
