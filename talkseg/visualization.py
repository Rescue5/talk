from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "talkseg-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import _normalization


@torch.inference_mode()
def save_validation_grid(
    model: nn.Module,
    loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
    output_path: Path,
    rows: int = 3,
) -> None:
    """Save a 3-column grid: image, model prediction, ground truth."""

    was_training = model.training
    model.eval()
    images_to_show: list[torch.Tensor] = []
    predictions_to_show: list[torch.Tensor] = []
    targets_to_show: list[torch.Tensor] = []
    image_ids: list[str] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True).contiguous()
        logits = model(images).contiguous()
        if config.data.task == "binary":
            predictions = (
                logits.sigmoid()[:, 0] >= config.train.threshold
            ).long()
        else:
            predictions = logits.argmax(dim=1)

        required = rows - len(images_to_show)
        images_to_show.extend(batch["image"][:required].cpu())
        predictions_to_show.extend(predictions[:required].cpu())
        targets_to_show.extend(batch["mask"][:required].cpu())
        image_ids.extend(list(batch["image_id"][:required]))
        if len(images_to_show) >= rows:
            break

    model.train(was_training)
    if not images_to_show:
        raise ValueError("Validation loader is empty; cannot create visualization")

    mean, std = _normalization(config.model)
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    figure, axes = plt.subplots(rows, 3, figsize=(12, 12), squeeze=False)
    column_titles = ("Исходное изображение", "Предсказание", "Ground truth")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=13)

    for row in range(rows):
        if row >= len(images_to_show):
            for axis in axes[row]:
                axis.axis("off")
            continue

        image = images_to_show[row].permute(1, 2, 0).numpy()
        image = np.clip(image * std_array + mean_array, 0.0, 1.0)
        prediction = predictions_to_show[row].numpy()
        target = targets_to_show[row].numpy()
        color_map = "gray" if config.data.task == "binary" else "tab20"

        axes[row, 0].imshow(image)
        axes[row, 1].imshow(prediction, cmap=color_map, interpolation="nearest")
        axes[row, 2].imshow(target, cmap=color_map, interpolation="nearest")
        axes[row, 0].set_ylabel(image_ids[row], fontsize=9)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(f"Validation predictions — epoch {epoch}", fontsize=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_history(history_path: Path, output_path: Path) -> None:
    with history_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Training history is empty: {history_path}")

    epochs = [int(row["epoch"]) for row in rows]

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("Loss", ("train_loss", "val_loss")),
        ("IoU", ("train_iou", "val_iou")),
        ("Dice", ("train_dice", "val_dice")),
        (
            "Precision / Recall",
            (
                "train_precision",
                "val_precision",
                "train_recall",
                "val_recall",
            ),
        ),
    ]
    for axis, (title, series_names) in zip(axes.flat, panels, strict=True):
        for series_name in series_names:
            axis.plot(
                epochs,
                values(series_name),
                label=series_name.replace("_", " "),
                linewidth=2,
                marker="o",
                markersize=3,
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle("Training history", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
