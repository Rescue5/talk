from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            json_safe(payload),
            stream,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        stream.write("\n")


PNG_COMPRESSION = 1


def write_png(path: Path, image: np.ndarray) -> None:
    params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"Could not write {path}")


@dataclass
class AnalysisResult:
    image_rgb: np.ndarray
    segmentation_mask: np.ndarray
    refined_talc_mask: np.ndarray
    segmentation_confidence: np.ndarray
    cv_confidence: np.ndarray
    positive_votes: np.ndarray
    vote_count: np.ndarray
    statistics: dict[str, Any]

    def save(
        self,
        output_dir: str | Path,
        overwrite: bool = False,
        *,
        write_manifest: bool = True,
    ) -> Path:
        started = perf_counter()
        destination = Path(output_dir)
        artifact_names = {
            "segmentation_mask": "segmentation_mask.png",
            "refined_talc_mask": "refined_talc_mask.png",
            "overlay": "overlay.png",
            "confidence_maps": "confidence_maps.npz",
            "result": "result.json",
        }
        existing = [
            destination / name
            for name in artifact_names.values()
            if (destination / name).exists()
        ]
        if existing and not overwrite:
            raise FileExistsError(
                f"Result already exists at {destination}; use overwrite=True"
            )
        destination.mkdir(parents=True, exist_ok=True)

        coarse_path = destination / artifact_names["segmentation_mask"]
        refined_path = destination / artifact_names["refined_talc_mask"]
        write_png(coarse_path, self.segmentation_mask.astype(np.uint8) * 255)
        write_png(refined_path, self.refined_talc_mask.astype(np.uint8) * 255)

        overlay = self.image_rgb.copy()
        refined = self.refined_talc_mask.astype(bool)
        if np.any(refined):
            red = np.asarray([255, 40, 40], dtype=np.float32)
            overlay[refined] = (
                0.45 * overlay[refined].astype(np.float32) + 0.55 * red
            ).astype(np.uint8)
        contours, _ = cv2.findContours(
            self.segmentation_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (255, 210, 0), 2)
        overlay_path = destination / artifact_names["overlay"]
        write_png(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        np.savez(
            destination / artifact_names["confidence_maps"],
            segmentation_confidence=self.segmentation_confidence.astype(np.float32),
            cv_confidence=self.cv_confidence.astype(np.float32),
            positive_votes=self.positive_votes,
            vote_count=self.vote_count,
        )
        serialization_seconds = perf_counter() - started
        manifest = copy.deepcopy(self.statistics)
        manifest["artifacts"] = artifact_names
        timings = manifest.setdefault("timings_seconds", {})
        timings["serialization"] = serialization_seconds
        timings["total"] = float(timings.get("processing_total", 0.0)) + serialization_seconds
        if write_manifest:
            write_json(destination / artifact_names["result"], manifest)
        self.statistics = manifest
        return destination

