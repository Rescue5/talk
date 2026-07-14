"""Inference CLI for trained ore classifier checkpoints."""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from .config import load_config
from .dataset import make_views
from .model import build_model
from .utils import IDX_TO_CLASS, IMAGE_EXTENSIONS, read_image_rgb, write_rows_csv


def _iter_inputs(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
    return [path]


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _predict_once(model, config: dict[str, Any], image_path: Path, device: torch.device) -> dict[str, Any]:
    total_start = time.perf_counter()

    read_start = time.perf_counter()
    image = read_image_rgb(image_path)
    read_end = time.perf_counter()

    preprocess_start = time.perf_counter()
    views = make_views(image, config, train=False, rng=random.Random(0))
    preprocess_end = time.perf_counter()

    transfer_start = time.perf_counter()
    batch = views.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    _sync_if_cuda(device)
    transfer_end = time.perf_counter()

    forward_start = time.perf_counter()
    logits = model(batch)
    _sync_if_cuda(device)
    forward_end = time.perf_counter()

    postprocess_start = time.perf_counter()
    prob_difficult = float(torch.sigmoid(logits)[0].detach().cpu())
    prob_ordinary = 1.0 - prob_difficult
    predicted_index = int(prob_difficult >= 0.5)
    confidence = max(prob_ordinary, prob_difficult)
    postprocess_end = time.perf_counter()

    total_end = time.perf_counter()
    return {
        "probability_ordinary": prob_ordinary,
        "probability_difficult": prob_difficult,
        "predicted_class": IDX_TO_CLASS[predicted_index],
        "confidence": confidence,
        "image_read_ms": (read_end - read_start) * 1000.0,
        "preprocess_views_ms": (preprocess_end - preprocess_start) * 1000.0,
        "transfer_to_device_ms": (transfer_end - transfer_start) * 1000.0,
        "model_forward_ms": (forward_end - forward_start) * 1000.0,
        "postprocess_ms": (postprocess_end - postprocess_start) * 1000.0,
        "total_inference_ms": (total_end - total_start) * 1000.0,
        "num_views": int(views.shape[0]),
        "image_height": int(image.shape[0]),
        "image_width": int(image.shape[1]),
    }


def _predict_with_benchmark(
    model,
    config: dict[str, Any],
    image_path: Path,
    device: torch.device,
    benchmark_runs: int,
    warmup_runs: int,
) -> dict[str, Any]:
    for _ in range(max(0, warmup_runs)):
        _predict_once(model, config, image_path, device)

    runs = [_predict_once(model, config, image_path, device) for _ in range(max(1, benchmark_runs))]
    last = runs[-1]
    row: dict[str, Any] = {
        "file_path": str(image_path),
        "probability_ordinary": last["probability_ordinary"],
        "probability_difficult": last["probability_difficult"],
        "predicted_class": last["predicted_class"],
        "confidence": last["confidence"],
        "benchmark_runs": len(runs),
        "warmup_runs": max(0, warmup_runs),
        "num_views": last["num_views"],
        "image_height": last["image_height"],
        "image_width": last["image_width"],
    }
    timing_keys = [
        "image_read_ms",
        "preprocess_views_ms",
        "transfer_to_device_ms",
        "model_forward_ms",
        "postprocess_ms",
        "total_inference_ms",
    ]
    for key in timing_keys:
        values = [float(run[key]) for run in runs]
        row[key] = _mean(values)
        row[f"{key}_std"] = _std(values)
    return row


@torch.no_grad()
def predict(
    config_path: str | Path,
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_csv: str | Path | None = None,
    benchmark_runs: int | None = None,
    warmup_runs: int | None = None,
    device_name: str | None = None,
) -> list[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint.get("config") or load_config(config_path)
        state_dict = checkpoint["model_state_dict"]
    else:
        config = load_config(config_path)
        state_dict = checkpoint
    inference_config = config.get("inference", {})
    if benchmark_runs is None:
        benchmark_runs = int(inference_config.get("benchmark_runs", 1))
    if warmup_runs is None:
        warmup_runs = int(inference_config.get("warmup_runs", 0))
    if device_name is None:
        device_name = str(inference_config.get("device", "auto"))
    config["model"]["pretrained"] = False
    model = build_model(config)
    model.load_state_dict(state_dict)
    device = _select_device(device_name)
    model.to(device).eval()

    rows: list[dict[str, Any]] = []
    for image_path in _iter_inputs(input_path):
        rows.append(
            _predict_with_benchmark(
                model=model,
                config=config,
                image_path=image_path,
                device=device,
                benchmark_runs=benchmark_runs,
                warmup_runs=warmup_runs,
            )
        )
    if output_csv:
        write_rows_csv(rows, output_csv)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ordinary/difficult ore class for image(s).")
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--benchmark-runs", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    args = parser.parse_args()
    rows = predict(
        args.config,
        args.checkpoint,
        args.input,
        args.output_csv,
        benchmark_runs=args.benchmark_runs,
        warmup_runs=args.warmup_runs,
        device_name=args.device,
    )
    for row in rows:
        print(
            f"{row['file_path']}: ordinary={row['probability_ordinary']:.4f} "
            f"difficult={row['probability_difficult']:.4f} "
            f"predicted={row['predicted_class']} confidence={row['confidence']:.4f} "
            f"total={row['total_inference_ms']:.2f}ms"
        )
        print(
            "  timing avg/std ms: "
            f"read={row['image_read_ms']:.2f}/{row['image_read_ms_std']:.2f}, "
            f"views={row['preprocess_views_ms']:.2f}/{row['preprocess_views_ms_std']:.2f}, "
            f"to_device={row['transfer_to_device_ms']:.2f}/{row['transfer_to_device_ms_std']:.2f}, "
            f"forward={row['model_forward_ms']:.2f}/{row['model_forward_ms_std']:.2f}, "
            f"post={row['postprocess_ms']:.2f}/{row['postprocess_ms_std']:.2f}, "
            f"runs={row['benchmark_runs']}, warmup={row['warmup_runs']}, views={row['num_views']}"
        )
    if rows:
        print(
            "Average total inference time: "
            f"{statistics.fmean(float(row['total_inference_ms']) for row in rows):.2f} ms/image "
            f"over {len(rows)} image(s)"
        )


if __name__ == "__main__":
    main()
