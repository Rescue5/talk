from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Sequence

from .analyzer import TalcAnalyzer
from .inference import SegmentationMode
from .results import write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def input_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    paths = sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found in {path}")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment talc zones, refine them with CV, and export statistics"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in SegmentationMode],
        default=SegmentationMode.OVERLAP.value,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    paths = input_paths(args.input.expanduser().resolve())
    stems: set[str] = set()
    duplicates: set[str] = set()
    for path in paths:
        if path.stem in stems:
            duplicates.add(path.stem)
        stems.add(path.stem)
    if duplicates:
        raise ValueError(f"Duplicate image stems: {sorted(duplicates)}")

    args.output.mkdir(parents=True, exist_ok=True)
    run_path = args.output / "run.json"
    if run_path.exists() and not args.overwrite:
        raise FileExistsError(f"{run_path} already exists; use --overwrite")

    analyzer = TalcAnalyzer.from_files(args.checkpoint, args.config)
    items: list[dict[str, object]] = []
    failures = 0
    for path in paths:
        item_started = perf_counter()
        try:
            result = analyzer.analyze_path(path, args.mode)
            destination = result.save(args.output / path.stem, overwrite=args.overwrite)
            items.append(
                {
                    "source": str(path),
                    "status": "ok",
                    "result": str(destination / "result.json"),
                    "classification": result.statistics["classification"]["code"],
                    "talc_percent": result.statistics["classification"]["talc_percent"],
                    "elapsed_seconds": perf_counter() - item_started,
                }
            )
            print(destination)
        except Exception as error:
            failures += 1
            items.append(
                {
                    "source": str(path),
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": perf_counter() - item_started,
                }
            )
            print(f"ERROR {path}: {error}", file=sys.stderr)

    manifest = {
        "schema_version": "1.0",
        "started_at": started_at,
        "input": str(args.input.expanduser().resolve()),
        "mode": args.mode,
        "model_load_seconds": analyzer.model_load_seconds,
        "processed": len(paths) - failures,
        "failed": failures,
        "elapsed_seconds": perf_counter() - run_started,
        "items": items,
    }
    write_json(run_path, manifest)
    return 1 if failures else 0


def main() -> None:
    raise SystemExit(run())
