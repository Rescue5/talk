from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .config import ModelConfig

ModelBuilder = Callable[[ModelConfig, int], nn.Module]
_MODEL_BUILDERS: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Register a model builder so new architectures stay isolated in this module."""

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        _MODEL_BUILDERS[name.lower()] = builder
        return builder

    return decorator


@register_model("deeplabv3")
def _build_deeplabv3(config: ModelConfig, output_channels: int) -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.DeepLabV3(
        encoder_name=config.encoder_name,
        encoder_depth=config.encoder_depth,
        encoder_weights=config.encoder_weights,
        decoder_channels=config.decoder_channels,
        in_channels=3,
        classes=output_channels,
        activation=None,
    )


@register_model("segformer")
def _build_segformer(config: ModelConfig, output_channels: int) -> nn.Module:
    import segmentation_models_pytorch as smp

    model = smp.Segformer(
        encoder_name=config.encoder_name,
        encoder_depth=config.encoder_depth,
        encoder_weights=config.encoder_weights,
        decoder_segmentation_channels=config.decoder_channels,
        in_channels=3,
        classes=output_channels,
        activation=None,
    )
    # SMP uses BatchNorm in the otherwise MLP-based decoder. Its backward pass
    # fails for this decoder layout on Apple MPS, and batch statistics are also
    # unstable for the small batches used by this project. Encoder weights are
    # unaffected because only the randomly initialized decoder norm is replaced.
    decoder_channels = config.decoder_channels
    groups = min(32, decoder_channels)
    while decoder_channels % groups:
        groups -= 1
    model.decoder.fuse_stage[1] = nn.GroupNorm(groups, decoder_channels)
    return model


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_BUILDERS))


def build_model(config: ModelConfig, output_channels: int) -> nn.Module:
    name = config.name.lower()
    try:
        builder = _MODEL_BUILDERS[name]
    except KeyError as error:
        supported = ", ".join(available_models())
        raise ValueError(f"Unknown model '{config.name}'. Available: {supported}") from error
    return builder(config, output_channels)


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    for parameter in encoder.parameters():
        parameter.requires_grad = trainable
