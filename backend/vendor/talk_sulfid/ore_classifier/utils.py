"""Shared utilities for dataset indexing, image IO, and crop sampling."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import warnings
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_TO_IDX = {"ordinary": 0, "difficult": 1}
IDX_TO_CLASS = {value: key for key, value in CLASS_TO_IDX.items()}


def is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def read_image_rgb(path: str | Path, max_side: int | None = None) -> np.ndarray:
    """Read an RGB image from paths that may contain non-ASCII characters."""

    path = Path(path)
    raw = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        with Image.open(path) as pil_image:
            rgb = np.asarray(pil_image.convert("RGB"))
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if max_side and max(rgb.shape[:2]) > max_side:
        scale = float(max_side) / float(max(rgb.shape[:2]))
        width = max(1, int(round(rgb.shape[1] * scale)))
        height = max(1, int(round(rgb.shape[0] * scale)))
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return rgb


def read_image_size(path: str | Path) -> tuple[int | None, int | None, str | None]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                return int(image.width), int(image.height), None
    except Exception as exc:  # pragma: no cover - data dependent
        return None, None, f"{type(exc).__name__}: {exc}"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash_image(path: str | Path) -> str | None:
    try:
        image = read_image_rgb(path, max_side=256)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        bits = small[:, 1:] > small[:, :-1]
        value = 0
        for bit in bits.reshape(-1):
            value = (value << 1) | int(bit)
        return f"{value:016x}"
    except Exception:
        return None


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def write_json(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_stem(stem: str) -> str:
    value = stem.lower().replace("х", "x")
    value = re.sub(r"\(\d+\)", " ", value)
    value = re.sub(r"\b(?:5|10|20|40)\s*x\b", " ", value)
    value = re.sub(r"\b(?:аншлиф|шлиф|ом|om)\b", " ", value)
    value = re.sub(r"[_]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" -_.")
    return value


def extract_group_id(path: str | Path) -> str:
    """Best-effort sample/group ID extraction from microscopy filenames."""

    stem = normalize_stem(Path(path).stem)
    if not stem:
        return Path(path).stem.lower()

    dscn = re.search(r"\bdscn\s*[-_ ]?(\d+)\b", stem)
    if dscn:
        return f"dscn{dscn.group(1)}"

    first_number = re.match(r"^(\d{6,}(?:-\d+)?)\b", stem)
    if first_number:
        return first_number.group(1)

    numeric = re.match(r"^[- ]*(\d+)[- ]*$", stem)
    if numeric:
        return numeric.group(1)

    compact = re.sub(r"[^0-9a-zа-яё-]+", "-", stem, flags=re.IGNORECASE)
    compact = compact.strip("-")
    return compact or Path(path).stem.lower()


def load_manual_group_map(path: str | Path | None, root: str | Path | None = None) -> dict[str, str]:
    if not path:
        return {}
    rows = read_rows_csv(path)
    mapping: dict[str, str] = {}
    for row in rows:
        group_id = row.get("group_id")
        file_key = row.get("file_path") or row.get("rel_path") or row.get("path")
        if not group_id or not file_key:
            continue
        mapping[file_key] = group_id
        if root:
            try:
                rel = str(Path(file_key).resolve().relative_to(Path(root).resolve()))
                mapping[rel] = group_id
            except Exception:
                pass
    return mapping


def infer_magnification(path: str | Path) -> str:
    text = str(path).lower().replace("х", "x")
    match = re.search(r"\b(5|10|20|40)\s*x\b", text)
    return f"{match.group(1)}x" if match else "unknown"


def deterministic_crop_boxes(width: int, height: int, num_crops: int, scale: float) -> list[tuple[int, int, int, int]]:
    if num_crops <= 0:
        return []
    side = max(1, int(round(min(width, height) * scale)))
    side = min(side, width, height)
    x_mid = max(0, (width - side) // 2)
    y_mid = max(0, (height - side) // 2)
    x_max = max(0, width - side)
    y_max = max(0, height - side)
    anchors = [
        (0, 0),
        (x_max, 0),
        (0, y_max),
        (x_max, y_max),
        (x_mid, y_mid),
        (x_mid, 0),
        (x_mid, y_max),
        (0, y_mid),
        (x_max, y_mid),
    ]
    boxes = [(x, y, x + side, y + side) for x, y in anchors]
    if num_crops > len(boxes):
        grid = int(math.ceil(math.sqrt(num_crops)))
        for gy in range(grid):
            for gx in range(grid):
                if len(boxes) >= num_crops:
                    break
                x = int(round((width - side) * gx / max(1, grid - 1)))
                y = int(round((height - side) * gy / max(1, grid - 1)))
                box = (x, y, x + side, y + side)
                if box not in boxes:
                    boxes.append(box)
    return boxes[:num_crops]


def random_crop_box(width: int, height: int, scale_range: Iterable[float], rng: random.Random) -> tuple[int, int, int, int]:
    low, high = list(scale_range)
    scale = rng.uniform(float(low), float(high))
    side = max(1, int(round(min(width, height) * scale)))
    side = min(side, width, height)
    x = rng.randint(0, max(0, width - side))
    y = rng.randint(0, max(0, height - side))
    return x, y, x + side, y + side


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
