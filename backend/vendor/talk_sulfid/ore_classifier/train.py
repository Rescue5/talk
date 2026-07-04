"""Training, validation, grouped CV, and reporting for ore classification."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .config import load_config, save_config
from .dataset import OreImageDataset, limit_samples_per_class, load_or_build_samples
from .model import build_model, build_optimizer
from .utils import CLASS_TO_IDX, IDX_TO_CLASS, set_global_seed, write_json, write_rows_csv


def compute_metrics(y_true: np.ndarray, prob_difficult: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (prob_difficult >= threshold).astype(np.int64)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    metrics: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "confusion_matrix": cm.tolist(),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob_difficult))
        metrics["pr_auc"] = float(average_precision_score(y_true, prob_difficult))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def _warn_mixed_groups(groups: np.ndarray, y: np.ndarray) -> None:
    group_targets: dict[str, set[int]] = defaultdict(set)
    for group, label in zip(groups, y):
        group_targets[str(group)].add(int(label))
    mixed_groups = {group: labels for group, labels in group_targets.items() if len(labels) > 1}
    if mixed_groups:
        print(f"WARNING: {len(mixed_groups)} group(s) contain both labels; exact duplicates may be inconsistent.")


def _make_train_val_split(samples: list[dict[str, Any]], config: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.array([CLASS_TO_IDX[sample["target"]] for sample in samples])
    groups = np.array([sample["group_id"] for sample in samples])
    _warn_mixed_groups(groups, y)

    group_to_indices: dict[str, list[int]] = defaultdict(list)
    group_to_labels: dict[str, list[int]] = defaultdict(list)
    for index, (group, label) in enumerate(zip(groups, y)):
        group_to_indices[str(group)].append(index)
        group_to_labels[str(group)].append(int(label))

    unique_groups = sorted(group_to_indices)
    group_labels = []
    for group in unique_groups:
        group_labels.append(Counter(group_to_labels[group]).most_common(1)[0][0])

    val_fraction = float(config["data"].get("val_fraction", 0.3))
    seed = int(config["training"].get("seed", 42))
    stratify = group_labels if min(Counter(group_labels).values()) >= 2 else None
    if stratify is None:
        print("WARNING: train/val split is not stratified because at least one class has fewer than 2 groups.")
    train_groups, val_groups = train_test_split(
        unique_groups,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    train_group_set = set(train_groups)
    val_group_set = set(val_groups)
    train_indices = [index for group in sorted(train_group_set) for index in group_to_indices[group]]
    val_indices = [index for group in sorted(val_group_set) for index in group_to_indices[group]]
    return [(np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64))]


def _make_cross_val_splits(samples: list[dict[str, Any]], config: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.array([CLASS_TO_IDX[sample["target"]] for sample in samples])
    groups = np.array([sample["group_id"] for sample in samples])
    _warn_mixed_groups(groups, y)

    class_group_counts = []
    for label in sorted(set(y.tolist())):
        class_group_counts.append(len({group for group, item_y in zip(groups, y) if item_y == label}))
    requested = int(config["data"].get("num_folds", 5))
    n_splits = min(requested, len(set(groups.tolist())), *class_group_counts)
    if n_splits < 2:
        raise RuntimeError("Not enough groups per class for grouped validation split.")
    if n_splits < requested:
        print(f"WARNING: using {n_splits} folds instead of requested {requested}; not enough unique groups.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(config["training"].get("seed", 42)))
    return list(splitter.split(np.zeros(len(samples)), y, groups))


def make_splits(samples: list[dict[str, Any]], config: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    split_strategy = config["data"].get("split_strategy", "train_val")
    if split_strategy == "train_val":
        return _make_train_val_split(samples, config)
    if split_strategy == "cross_val":
        return _make_cross_val_splits(samples, config)
    raise ValueError(f"Unknown split_strategy: {split_strategy}")


def _make_loader(samples: list[dict[str, Any]], config: dict[str, Any], train: bool) -> DataLoader:
    seed = int(config["training"].get("seed", 42))
    dataset = OreImageDataset(samples, config, train=train, seed=seed)
    sampler = None
    shuffle = train
    if train and config["training"].get("weighted_sampler", False):
        labels = [CLASS_TO_IDX[sample["target"]] for sample in samples]
        counts = Counter(labels)
        weights = [1.0 / counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 4)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def _loss_function(train_samples: list[dict[str, Any]], config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    if not config["training"].get("class_weights", True):
        return torch.nn.BCEWithLogitsLoss()
    labels = [CLASS_TO_IDX[sample["target"]] for sample in train_samples]
    neg = max(1, labels.count(0))
    pos = max(1, labels.count(1))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config["training"].get("scheduler") != "cosine":
        return None
    epochs = max(1, int(config["training"].get("epochs", 1)))
    warmup = max(0, int(config["training"].get("warmup_epochs", 0)))

    def lr_lambda(epoch: int) -> float:
        if warmup and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _sanitize_experiment_name(name: str) -> str:
    value = str(name).strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip(" ._")
    return value or "experiment"


def _make_run_dir(config: dict[str, Any]) -> Path:
    output_dir = Path(config["data"].get("output_dir") or "runs/ore_classifier")
    timestamp = datetime.now().strftime("train_%Y%m%d_%H%M%S")
    experiment_name = config["training"].get("experiment_name")
    append_timestamp = bool(config["training"].get("timestamp_subdir", True))
    if experiment_name:
        experiment_dir = output_dir / _sanitize_experiment_name(str(experiment_name))
        return experiment_dir / timestamp if append_timestamp else experiment_dir
    return output_dir / timestamp


def _make_summary_writer(config: dict[str, Any], log_dir: Path):
    if not config.get("reports", {}).get("tensorboard", True):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"WARNING: TensorBoard logging disabled: {exc}")
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(log_dir))


def _log_epoch_to_tensorboard(writer, split_name: str, epoch: int, row: dict[str, Any], optimizer: torch.optim.Optimizer) -> None:
    if writer is None:
        return
    scalars = {
        "loss/train": row.get("train_loss"),
        "loss/val": row.get("val_loss"),
        "metrics/macro_f1": row.get("macro_f1"),
        "metrics/balanced_accuracy": row.get("balanced_accuracy"),
        "metrics/accuracy": row.get("accuracy"),
        "metrics/precision": row.get("precision"),
        "metrics/recall": row.get("recall"),
        "metrics/specificity": row.get("specificity"),
        "metrics/roc_auc": row.get("roc_auc"),
        "metrics/pr_auc": row.get("pr_auc"),
    }
    for name, value in scalars.items():
        if value is not None:
            writer.add_scalar(f"{split_name}/{name}", float(value), epoch)
    for index, group in enumerate(optimizer.param_groups):
        writer.add_scalar(f"{split_name}/lr/group_{index}", float(group["lr"]), epoch)


def _log_final_to_tensorboard(writer, report: dict[str, Any]) -> None:
    if writer is None:
        return
    for key, value in report.get("aggregate_oof_metrics", {}).items():
        if isinstance(value, (float, int)):
            writer.add_scalar(f"aggregate_oof/{key}", float(value), 0)
    for source, metrics in report.get("source_metrics", {}).items():
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                writer.add_scalar(f"source/{source}/{key}", float(value), 0)


def _train_one_epoch(model, loader, criterion, optimizer, scaler, device, config, split_name: str, epoch: int) -> float:
    model.train()
    losses = []
    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    progress = tqdm(
        loader,
        desc=f"{split_name} epoch {epoch} train",
        leave=False,
        disable=not config.get("reports", {}).get("progress_bar", True),
    )
    for views, labels, _indices in progress:
        views = views.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(views)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        clip_norm = config["training"].get("gradient_clip_norm")
        if clip_norm:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def _evaluate(model, loader, criterion, device, config, split_name: str = "val", epoch: int | None = None) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    y_true, probs, indices = [], [], []
    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    desc = f"{split_name} val" if epoch is None else f"{split_name} epoch {epoch} val"
    progress = tqdm(
        loader,
        desc=desc,
        leave=False,
        disable=not config.get("reports", {}).get("progress_bar", True),
    )
    for views, labels, batch_indices in progress:
        views = views.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(views)
            loss = criterion(logits, labels)
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")
        y_true.extend(labels.detach().cpu().numpy().astype(np.int64).tolist())
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        indices.extend(batch_indices.numpy().tolist())
    return (
        float(np.mean(losses)) if losses else 0.0,
        np.asarray(y_true, dtype=np.int64),
        np.asarray(probs, dtype=np.float32),
        np.asarray(indices, dtype=np.int64),
    )


def _save_prediction_rows(
    samples: list[dict[str, Any]],
    y_true: np.ndarray,
    probs: np.ndarray,
    indices: np.ndarray,
    fold: int,
    path: Path,
    split_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, prob, dataset_index in zip(y_true, probs, indices):
        sample = samples[int(dataset_index)]
        predicted = int(prob >= 0.5)
        rows.append(
            {
                "file_path": sample["file_path"],
                "rel_path": sample["rel_path"],
                "dataset_source": sample["dataset_source"],
                "group_id": sample["group_id"],
                "ground_truth": IDX_TO_CLASS[int(label)],
                "predicted_class": IDX_TO_CLASS[predicted],
                "probability_ordinary": float(1.0 - prob),
                "probability_difficult": float(prob),
                "fold": fold,
                "split": split_name,
            }
        )
    write_rows_csv(rows, path)
    return rows


def _save_sample_list(samples: list[dict[str, Any]], path: Path, split_name: str) -> None:
    rows = []
    for sample in samples:
        rows.append(
            {
                "split": split_name,
                "file_path": sample["file_path"],
                "rel_path": sample["rel_path"],
                "dataset_source": sample["dataset_source"],
                "group_id": sample["group_id"],
                "target": sample["target"],
            }
        )
    write_rows_csv(rows, path)


def _nonempty_set(samples: list[dict[str, Any]], key: str) -> set[str]:
    return {str(sample.get(key)) for sample in samples if sample.get(key)}


def _counter_dict(samples: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(sample.get(key, "")) for sample in samples))


def _audit_split(train_samples: list[dict[str, Any]], val_samples: list[dict[str, Any]]) -> dict[str, Any]:
    overlap_keys = ["file_path", "rel_path", "group_id", "sha256", "dhash"]
    overlaps = {}
    for key in overlap_keys:
        train_values = _nonempty_set(train_samples, key)
        val_values = _nonempty_set(val_samples, key)
        overlaps[key] = sorted(train_values & val_values)

    audit = {
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_target_counts": _counter_dict(train_samples, "target"),
        "val_target_counts": _counter_dict(val_samples, "target"),
        "train_source_counts": _counter_dict(train_samples, "dataset_source"),
        "val_source_counts": _counter_dict(val_samples, "dataset_source"),
        "train_group_count": len(_nonempty_set(train_samples, "group_id")),
        "val_group_count": len(_nonempty_set(val_samples, "group_id")),
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "overlap_examples": {key: value[:25] for key, value in overlaps.items() if value},
        "leak_free": all(len(value) == 0 for value in overlaps.values()),
    }
    return audit


def _assert_no_split_leakage(audit: dict[str, Any]) -> None:
    if audit["leak_free"]:
        return
    raise RuntimeError(f"Train/val leakage detected: {audit['overlap_counts']}")


def _save_plots(y_true: np.ndarray, probs: np.ndarray, output_dir: Path, prefix: str) -> None:
    if len(np.unique(y_true)) < 2:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_roc.png", dpi=160)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_pr.png", dpi=160)
    plt.close()


def _copy_examples(rows: list[dict[str, Any]], output_dir: Path, num_examples: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for row in rows:
        prob = float(row["probability_difficult"])
        label = 1 if row["ground_truth"] == "difficult" else 0
        predicted = 1 if row["predicted_class"] == "difficult" else 0
        confidence = max(prob, 1.0 - prob)
        uncertainty = abs(prob - 0.5)
        scored.append((row, label == predicted, confidence, uncertainty))
    buckets = {
        "correct_confident": sorted([item for item in scored if item[1]], key=lambda item: -item[2])[:num_examples],
        "errors": sorted([item for item in scored if not item[1]], key=lambda item: -item[2])[:num_examples],
        "uncertain": sorted(scored, key=lambda item: item[3])[:num_examples],
    }
    for bucket, items in buckets.items():
        bucket_dir = output_dir / bucket
        bucket_dir.mkdir(exist_ok=True)
        for rank, (row, _ok, _conf, _unc) in enumerate(items):
            source = Path(row["file_path"])
            suffix = source.suffix.lower()
            target = bucket_dir / f"{rank:02d}_{row['ground_truth']}_as_{row['predicted_class']}{suffix}"
            try:
                shutil.copyfile(source, target)
            except Exception:
                pass


def train(config_path: str | Path = "configs/classifier.yaml", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_config(config_path, overrides)
    set_global_seed(int(config["training"].get("seed", 42)))
    run_dir = _make_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    samples = load_or_build_samples(config)
    samples = limit_samples_per_class(samples, config["training"].get("max_samples_per_class"), int(config["training"].get("seed", 42)))
    splits = make_splits(samples, config)
    max_folds = config["training"].get("max_folds")
    if max_folds and config["data"].get("split_strategy") == "cross_val":
        splits = splits[: int(max_folds)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_strategy = config["data"].get("split_strategy", "train_val")
    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Samples: {len(samples)}; split_strategy: {split_strategy}; splits: {len(splits)}")

    tensorboard_writer = _make_summary_writer(config, run_dir / "tensorboard")
    fold_metrics = []
    all_oof_rows: list[dict[str, Any]] = []
    split_counts = []
    for fold, (train_indices, val_indices) in enumerate(splits):
        train_samples = [samples[index] for index in train_indices]
        val_samples = [samples[index] for index in val_indices]
        split_name = "train_val" if split_strategy == "train_val" else f"fold_{fold}"
        split_dir = run_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_counts.append({"split": split_name, "train_count": len(train_samples), "val_count": len(val_samples)})
        write_json(
            {
                "train_groups": sorted({sample["group_id"] for sample in train_samples}),
                "val_groups": sorted({sample["group_id"] for sample in val_samples}),
                "train_count": len(train_samples),
                "val_count": len(val_samples),
            },
            split_dir / "split_groups.json",
        )
        _save_sample_list(train_samples, split_dir / "train_samples.csv", "train")
        _save_sample_list(val_samples, split_dir / "val_samples.csv", "val")
        split_audit = _audit_split(train_samples, val_samples)
        write_json(split_audit, split_dir / "split_audit.json")
        _assert_no_split_leakage(split_audit)
        if split_strategy == "train_val":
            _save_sample_list(train_samples, run_dir / "train_samples.csv", "train")
            _save_sample_list(val_samples, run_dir / "val_samples.csv", "val")
            write_json(split_audit, run_dir / "split_audit.json")

        model = build_model(config).to(device)
        optimizer = build_optimizer(model, config)
        scheduler = _scheduler(optimizer, config)
        criterion = _loss_function(train_samples, config, device)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(config["training"].get("mixed_precision", True)) and device.type == "cuda")
        train_loader = _make_loader(train_samples, config, train=True)
        val_loader = _make_loader(val_samples, config, train=False)

        best_metric = -float("inf")
        best_epoch = -1
        best_checkpoint = split_dir / "best.ckpt"
        best_weights = split_dir / "best.pt"
        history: list[dict[str, Any]] = []
        patience = int(config["training"].get("early_stopping_patience", 7))
        metric_name = config["training"].get("metric_for_best", "macro_f1")
        for epoch in range(int(config["training"].get("epochs", 30))):
            train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config, split_name, epoch)
            val_loss, y_true, probs, _indices = _evaluate(model, val_loader, criterion, device, config, split_name, epoch)
            metrics = compute_metrics(y_true, probs)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
            history.append(row)
            _log_epoch_to_tensorboard(tensorboard_writer, split_name, epoch, row, optimizer)
            current = metrics.get(metric_name)
            current_value = -float("inf") if current is None else float(current)
            print(
                f"split={split_name} epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} bal_acc={metrics['balanced_accuracy']:.4f}"
            )
            if current_value > best_metric:
                best_metric = current_value
                best_epoch = epoch
                state_dict = model.state_dict()
                torch.save(
                    {
                        "model_state_dict": state_dict,
                        "config": config,
                        "class_to_idx": CLASS_TO_IDX,
                        "fold": fold,
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                    },
                    best_checkpoint,
                )
                if config.get("reports", {}).get("save_pt_weights", True):
                    torch.save(state_dict, best_weights)
            if scheduler is not None:
                scheduler.step()
            if epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

        write_json(history, split_dir / "history.json")
        model.load_state_dict(torch.load(best_checkpoint, map_location=device)["model_state_dict"])
        val_loss, y_true, probs, indices = _evaluate(model, val_loader, criterion, device, config, split_name)
        metrics = compute_metrics(y_true, probs)
        metrics.update(
            {
                "fold": fold,
                "split": split_name,
                "best_epoch": best_epoch,
                "val_loss": val_loss,
                "best_checkpoint": str(best_checkpoint),
                "best_weights": str(best_weights) if best_weights.exists() else "",
            }
        )
        fold_metrics.append(metrics)
        rows = _save_prediction_rows(val_samples, y_true, probs, indices, fold, split_dir / "predictions.csv", split_name)
        all_oof_rows.extend(rows)
        if config.get("reports", {}).get("save_plots", True):
            _save_plots(y_true, probs, split_dir, "val")

    write_rows_csv(all_oof_rows, run_dir / "predictions.csv")
    y_true = np.array([CLASS_TO_IDX[row["ground_truth"]] for row in all_oof_rows])
    probs = np.array([float(row["probability_difficult"]) for row in all_oof_rows], dtype=np.float32)
    aggregate_metrics = compute_metrics(y_true, probs) if len(all_oof_rows) else {}
    source_metrics = {}
    for source in sorted({row["dataset_source"] for row in all_oof_rows}):
        source_rows = [row for row in all_oof_rows if row["dataset_source"] == source]
        source_y = np.array([CLASS_TO_IDX[row["ground_truth"]] for row in source_rows])
        source_probs = np.array([float(row["probability_difficult"]) for row in source_rows], dtype=np.float32)
        source_metrics[source] = compute_metrics(source_y, source_probs) if len(source_rows) else {}

    metric_keys = ["macro_f1", "balanced_accuracy", "accuracy", "precision", "recall", "specificity", "roc_auc", "pr_auc"]
    mean_std = {}
    for key in metric_keys:
        values = [metric[key] for metric in fold_metrics if metric.get(key) is not None]
        if values:
            mean_std[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    report = {
        "run_dir": str(run_dir),
        "experiment_name": config["training"].get("experiment_name") or "",
        "split_strategy": split_strategy,
        "val_fraction": config["data"].get("val_fraction", ""),
        "split_counts": split_counts,
        "tensorboard_dir": str(run_dir / "tensorboard") if tensorboard_writer is not None else "",
        "fold_metrics": fold_metrics,
        "mean_std": mean_std,
        "aggregate_oof_metrics": aggregate_metrics,
        "source_metrics": source_metrics,
    }
    _log_final_to_tensorboard(tensorboard_writer, report)
    if tensorboard_writer is not None:
        tensorboard_writer.flush()
        tensorboard_writer.close()
    write_json(report, run_dir / "metrics_summary.json")
    if config.get("reports", {}).get("save_plots", True) and len(all_oof_rows):
        _save_plots(y_true, probs, run_dir, "oof")
    if config.get("reports", {}).get("save_examples", True):
        _copy_examples(all_oof_rows, run_dir / "examples", int(config.get("reports", {}).get("num_examples", 12)))
    print(f"Finished. Metrics: {aggregate_metrics}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ore binary classifier.")
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--split-strategy", choices=["train_val", "cross_val"], default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--folds", type=int, default=None, help="Maximum number of folds to run in cross_val mode.")
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-local-crops", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--no-timestamp-subdir", action="store_true")
    args = parser.parse_args()

    overrides: dict[str, Any] = {"training": {}, "data": {}, "model": {}}
    if args.epochs is not None:
        overrides["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["training"]["batch_size"] = args.batch_size
    if args.split_strategy is not None:
        overrides["data"]["split_strategy"] = args.split_strategy
    if args.val_fraction is not None:
        overrides["data"]["val_fraction"] = args.val_fraction
    if args.folds is not None:
        overrides["training"]["max_folds"] = args.folds
    if args.max_samples_per_class is not None:
        overrides["training"]["max_samples_per_class"] = args.max_samples_per_class
    if args.image_size is not None:
        overrides["data"]["image_size"] = args.image_size
    if args.num_local_crops is not None:
        overrides["data"]["num_local_crops"] = args.num_local_crops
    if args.no_pretrained:
        overrides["model"]["pretrained"] = False
    if args.output_dir:
        overrides["data"]["output_dir"] = args.output_dir
    if args.experiment_name:
        overrides["training"]["experiment_name"] = args.experiment_name
    if args.no_timestamp_subdir:
        overrides["training"]["timestamp_subdir"] = False
    train(args.config, overrides)


if __name__ == "__main__":
    main()
