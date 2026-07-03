from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from talkseg.config import load_config
from talkseg.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = args.run_dir or (
        config.train.output_dir
        / f"{datetime.now():%Y%m%d-%H%M%S}_{config.model.name}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    best_path = train(config, run_dir, args.resume)
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
