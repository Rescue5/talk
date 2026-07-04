from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.yaml")


@dataclass(frozen=True)
class RuntimeConfig:
    overlap: float
    batch_size: int | None
    classification_threshold_percent: float
    cv_config: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "RuntimeConfig":
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RuntimeConfig":
        if not isinstance(raw, dict):
            raise ValueError("Configuration root must be a mapping")
        segmentation = raw.get("segmentation", {})
        classification = raw.get("classification", {})
        cv_config = raw.get("cv")
        if not isinstance(segmentation, dict) or not isinstance(classification, dict):
            raise ValueError("segmentation and classification must be mappings")
        if not isinstance(cv_config, dict):
            raise ValueError("cv must be a mapping")

        overlap = float(segmentation.get("overlap", 0.33))
        if not 0 < overlap < 1:
            raise ValueError("segmentation.overlap must be between 0 and 1")
        batch_size_raw = segmentation.get("batch_size")
        batch_size = None if batch_size_raw is None else int(batch_size_raw)
        if batch_size is not None and batch_size < 1:
            raise ValueError("segmentation.batch_size must be positive")
        threshold = float(classification.get("threshold_percent", 10.0))
        if not 0 <= threshold <= 100:
            raise ValueError("classification.threshold_percent must be in [0, 100]")
        return cls(overlap, batch_size, threshold, cv_config)


@dataclass(frozen=True)
class CheckpointConfig:
    model_name: str
    encoder_name: str
    encoder_weights: str | None
    encoder_depth: int
    decoder_channels: int
    image_size: int
    tile_size: int
    checkpoint_overlap: float
    batch_size: int
    threshold: float
    task: str
    class_names: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CheckpointConfig":
        try:
            raw = payload["config"]
            model = raw["model"]
            data = raw["data"]
            train = raw["train"]
            class_names = tuple(str(item) for item in payload["class_names"])
        except (KeyError, TypeError) as error:
            raise ValueError("Unsupported checkpoint structure") from error
        task = str(data.get("task", "binary"))
        if task != "binary" or len(class_names) != 1:
            raise ValueError(
                "Only a binary checkpoint with exactly one foreground class is supported"
            )
        threshold = float(train.get("threshold", 0.5))
        if not 0 <= threshold <= 1:
            raise ValueError("Checkpoint threshold must be in [0, 1]")
        return cls(
            model_name=str(model["name"]).lower(),
            encoder_name=str(model["encoder_name"]),
            encoder_weights=model.get("encoder_weights"),
            encoder_depth=int(model.get("encoder_depth", 5)),
            decoder_channels=int(model.get("decoder_channels", 256)),
            image_size=int(data["image_size"]),
            tile_size=int(data["tile_size"]),
            checkpoint_overlap=float(data.get("tile_overlap", 0.33)),
            batch_size=int(train.get("batch_size", 1)),
            threshold=threshold,
            task=task,
            class_names=class_names,
        )

