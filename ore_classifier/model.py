"""Multi-view classifier model with a shared timm encoder."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class GeMPool(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1.0e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)


class MultiCropClassifier(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("timm is required for ore classifier models") from exc

        model_config = config["model"]
        pretrained = bool(model_config.get("pretrained", True))
        backbone = model_config.get("backbone", "convnext_tiny")
        try:
            self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
            self.pretrained_loaded = pretrained
        except Exception as exc:
            if not model_config.get("allow_pretrained_fallback", True) or not pretrained:
                raise
            print(f"WARNING: could not load pretrained weights for {backbone}: {exc}")
            print("WARNING: continuing with randomly initialized backbone.")
            self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="")
            self.pretrained_loaded = False

        feature_dim = int(getattr(self.encoder, "num_features"))
        pooling = model_config.get("pooling", "mean")
        self.spatial_pool = GeMPool() if pooling == "gem" else None
        hidden_dim = int(model_config.get("hidden_dim", 256))
        dropout = float(model_config.get("dropout", 0.2))
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.apply_finetune_mode(model_config.get("finetune_mode", "last_stage"))

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            if self.spatial_pool is not None:
                return self.spatial_pool(features)
            return features.mean(dim=(-2, -1))
        return features

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        batch, num_views, channels, height, width = views.shape
        flat_views = views.reshape(batch * num_views, channels, height, width)
        features = self._pool_features(self.encoder(flat_views))
        features = features.reshape(batch, num_views, -1).mean(dim=1)
        return self.head(features).squeeze(1)

    def apply_finetune_mode(self, mode: str) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        mode = mode or "last_stage"
        if mode == "head_only":
            pass
        elif mode == "full_finetune":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
        else:
            trainable_prefixes = []
            if mode == "last_stage":
                trainable_prefixes = ["stages.3", "layer4"]
            elif mode == "last_two_stages":
                trainable_prefixes = ["stages.2", "stages.3", "layer3", "layer4"]
            else:
                raise ValueError(f"Unknown finetune_mode: {mode}")
            for name, parameter in self.encoder.named_parameters():
                if any(name.startswith(prefix) or f".{prefix}" in name for prefix in trainable_prefixes):
                    parameter.requires_grad = True
        for parameter in self.head.parameters():
            parameter.requires_grad = True
        if self.spatial_pool is not None:
            for parameter in self.spatial_pool.parameters():
                parameter.requires_grad = True


def build_optimizer(model: MultiCropClassifier, config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    backbone_params = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    head_params = list(model.head.parameters())
    if model.spatial_pool is not None:
        head_params += list(model.spatial_pool.parameters())
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(training["backbone_lr"])})
    groups.append({"params": head_params, "lr": float(training["head_lr"])})
    return torch.optim.AdamW(groups, weight_decay=float(training.get("weight_decay", 1.0e-4)))


def build_model(config: dict[str, Any]) -> MultiCropClassifier:
    return MultiCropClassifier(config)
