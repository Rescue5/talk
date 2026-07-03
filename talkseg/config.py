from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "deeplabv3"
    encoder_name: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    encoder_depth: int = 5
    decoder_channels: int = 256


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    classes_file: str = "classes.txt"
    image_size: int = 512
    tile_size: int = 512
    tile_overlap: float = 0.33
    val_fraction: float = 0.25
    split_file: Path = Path("artifacts/splits.json")
    repeat_factor: int = 1
    task: str = "binary"


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    freeze_encoder_epochs: int = 5
    seed: int = 42
    amp: bool = True
    output_dir: Path = Path("runs")
    threshold: float = 0.5
    bce_weight: float = 0.5
    dice_weight: float = 0.5


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        model = ModelConfig(**raw.get("model", {}))
        data_raw = dict(raw.get("data", {}))
        train_raw = dict(raw.get("train", {}))
        if "data_dir" in data_raw:
            data_raw["data_dir"] = Path(data_raw["data_dir"]).expanduser()
        if "split_file" in data_raw:
            data_raw["split_file"] = Path(data_raw["split_file"]).expanduser()
        if "output_dir" in train_raw:
            train_raw["output_dir"] = Path(train_raw["output_dir"]).expanduser()
        data = DataConfig(**data_raw)
        train = TrainConfig(**train_raw)
        config = cls(model=model, data=data, train=train)
        config.validate()
        return config

    def validate(self) -> None:
        if self.data.task not in {"binary", "multiclass"}:
            raise ValueError("data.task must be 'binary' or 'multiclass'")
        if not 0 < self.data.val_fraction < 1:
            raise ValueError("data.val_fraction must be between 0 and 1")
        if self.data.image_size <= 0 or self.data.tile_size <= 0:
            raise ValueError("image_size and tile_size must be positive")
        if not 0 <= self.data.tile_overlap < 1:
            raise ValueError("data.tile_overlap must be in [0, 1)")
        if self.data.repeat_factor < 1:
            raise ValueError("repeat_factor must be >= 1")
        if self.train.epochs < 1 or self.train.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return _paths_to_strings(asdict(self))


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_paths_to_strings(item) for item in value]
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return ExperimentConfig.from_dict(raw)
