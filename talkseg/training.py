from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .data import (
    RepeatDataset,
    YoloTiledSegmentationDataset,
    build_train_transform,
    build_val_transform,
    discover_samples,
    load_or_update_split,
    read_class_names,
    select_samples,
)
from .models import build_model, set_encoder_trainable


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CompositeSegmentationLoss(nn.Module):
    def __init__(self, task: str, bce_weight: float, dice_weight: float) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.task = task
        self.primary_weight = bce_weight
        self.dice_weight = dice_weight
        if task == "binary":
            self.primary = nn.BCEWithLogitsLoss()
            self.dice = smp.losses.DiceLoss(
                mode=smp.losses.BINARY_MODE, from_logits=True
            )
        else:
            self.primary = nn.CrossEntropyLoss()
            self.dice = smp.losses.DiceLoss(
                mode=smp.losses.MULTICLASS_MODE, from_logits=True
            )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.task == "binary":
            target_for_loss = target.unsqueeze(1).float()
        else:
            target_for_loss = target
        return self.primary_weight * self.primary(
            logits, target_for_loss
        ) + self.dice_weight * self.dice(logits, target_for_loss)


class SegmentationStats:
    def __init__(self, task: str, threshold: float) -> None:
        self.task = task
        self.threshold = threshold
        self.intersection = 0
        self.predicted = 0
        self.target = 0
        self.union = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if self.task == "binary":
            prediction = (logits.sigmoid()[:, 0] >= self.threshold).long()
            target_foreground = target > 0
            prediction_foreground = prediction > 0
        else:
            prediction = logits.argmax(dim=1)
            target_foreground = target > 0
            prediction_foreground = prediction > 0
        intersection = prediction_foreground & target_foreground
        union = prediction_foreground | target_foreground
        self.intersection += int(intersection.sum().item())
        self.predicted += int(prediction_foreground.sum().item())
        self.target += int(target_foreground.sum().item())
        self.union += int(union.sum().item())

    def compute(self) -> dict[str, float]:
        epsilon = 1e-7
        return {
            "iou": (self.intersection + epsilon) / (self.union + epsilon),
            "dice": (2 * self.intersection + epsilon)
            / (self.predicted + self.target + epsilon),
            "precision": (self.intersection + epsilon)
            / (self.predicted + epsilon),
            "recall": (self.intersection + epsilon) / (self.target + epsilon),
        }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
    threshold: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    encoder_frozen: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training and encoder_frozen and hasattr(model, "encoder"):
        model.encoder.eval()

    stats = SegmentationStats(task, threshold)
    total_loss = 0.0
    total_items = 0
    progress = tqdm(loader, leave=False, desc="train" if training else "val")
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type, enabled=use_amp, dtype=torch.float16
            ):
                logits = model(images)
                loss = criterion(logits.float(), targets)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        stats.update(logits.detach(), targets)
        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        progress.set_postfix(loss=f"{total_loss / total_items:.4f}")

    return {"loss": total_loss / max(total_items, 1), **stats.compute()}


def _make_loaders(config: ExperimentConfig):
    samples, image_count = discover_samples(config.data.data_dir)
    class_names = read_class_names(config.data.data_dir, config.data.classes_file)
    train_ids, val_ids = load_or_update_split(
        samples,
        config.data.split_file,
        config.data.val_fraction,
        config.train.seed,
    )
    train_samples = select_samples(samples, train_ids)
    val_samples = select_samples(samples, val_ids)
    train_base = YoloTiledSegmentationDataset(
        train_samples,
        build_train_transform(config.data, config.model),
        config.data.task,
        len(class_names),
        config.data.tile_size,
        config.data.tile_overlap,
    )
    val_dataset = YoloTiledSegmentationDataset(
        val_samples,
        build_val_transform(config.data, config.model),
        config.data.task,
        len(class_names),
        config.data.tile_size,
        config.data.tile_overlap,
    )
    train_dataset = RepeatDataset(train_base, config.data.repeat_factor)
    common = {
        "num_workers": config.train.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        drop_last=len(train_dataset) >= config.train.batch_size,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        **common,
    )
    return (
        train_loader,
        val_loader,
        class_names,
        len(samples),
        image_count,
        len(train_samples),
        len(val_samples),
        len(train_base),
        len(val_dataset),
    )


def _append_history(path: Path, row: dict[str, float | int]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: ExperimentConfig,
    class_names: list[str],
    epoch: int,
    best_iou: float,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config.to_dict(),
        "class_names": class_names,
        "epoch": epoch,
        "best_iou": best_iou,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(
    config: ExperimentConfig, run_dir: Path, resume: Path | None = None
) -> Path:
    seed_everything(config.train.seed)
    device = choose_device()
    (
        train_loader,
        val_loader,
        class_names,
        labeled_count,
        image_count,
        train_image_count,
        val_image_count,
        train_tile_count,
        val_tile_count,
    ) = _make_loaders(config)
    output_channels = 1 if config.data.task == "binary" else len(class_names) + 1
    model = build_model(config.model, output_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.epochs
    )
    criterion = CompositeSegmentationLoss(
        config.data.task, config.train.bce_weight, config.train.dice_weight
    )
    use_amp = config.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    best_iou = -1.0

    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_iou = float(checkpoint.get("best_iou", -1.0))

    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device} model={config.model.name}/{config.model.encoder_name} "
        f"labeled={labeled_count}/{image_count} "
        f"train_images={train_image_count} train_tiles={train_tile_count} "
        f"val_images={val_image_count} val_tiles={val_tile_count}"
    )

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    for epoch in range(start_epoch, config.train.epochs + 1):
        encoder_frozen = epoch <= config.train.freeze_encoder_epochs
        set_encoder_trainable(model, not encoder_frozen)
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            config.data.task,
            config.train.threshold,
            optimizer,
            scaler,
            use_amp,
            encoder_frozen,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            config.data.task,
            config.train.threshold,
            None,
            scaler,
            use_amp,
            False,
        )
        scheduler.step()
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["lr"] = optimizer.param_groups[0]["lr"]
        _append_history(run_dir / "history.csv", row)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_dice={val_metrics['dice']:.4f}"
        )
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                config,
                class_names,
                epoch,
                best_iou,
            )
        _save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            config,
            class_names,
            epoch,
            best_iou,
        )
    return best_path
