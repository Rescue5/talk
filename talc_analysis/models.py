from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn

from .config import CheckpointConfig


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: CheckpointConfig) -> nn.Module:
    import segmentation_models_pytorch as smp

    common: dict[str, Any] = {
        "encoder_name": config.encoder_name,
        "encoder_depth": config.encoder_depth,
        # All encoder weights are restored from model_state. Passing "imagenet"
        # here would trigger an unnecessary network download during deployment.
        "encoder_weights": None,
        "in_channels": 3,
        "classes": 1,
        "activation": None,
    }
    if config.model_name == "deeplabv3":
        return smp.DeepLabV3(
            **common,
            decoder_channels=config.decoder_channels,
        )
    if config.model_name == "segformer":
        model = smp.Segformer(
            **common,
            decoder_segmentation_channels=config.decoder_channels,
        )
        groups = min(32, config.decoder_channels)
        while config.decoder_channels % groups:
            groups -= 1
        model.decoder.fuse_stage[1] = nn.GroupNorm(groups, config.decoder_channels)
        return model
    raise ValueError(
        f"Unsupported model architecture {config.model_name!r}; "
        "expected 'deeplabv3' or 'segformer'"
    )


def load_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, CheckpointConfig, torch.device, float]:
    started = perf_counter()
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    selected_device = device or choose_device()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint root must be a mapping")
    config = CheckpointConfig.from_payload(payload)
    model = build_model(config)
    try:
        model.load_state_dict(payload["model_state"])
    except KeyError as error:
        raise ValueError("Checkpoint does not contain model_state") from error
    model.to(selected_device)
    model.eval()
    return model, config, selected_device, perf_counter() - started


def preprocessing_parameters(
    config: CheckpointConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if config.encoder_weights is None:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    import segmentation_models_pytorch as smp

    params = smp.encoders.get_preprocessing_params(
        config.encoder_name,
        pretrained=config.encoder_weights,
    )
    return tuple(params["mean"]), tuple(params["std"])
