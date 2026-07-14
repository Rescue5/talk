from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.processor import apply_manual_talc_edits
from app.schemas import JobSettings


def _prepare_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True)
    shape = (20, 20)
    Image.fromarray(np.full((*shape, 3), 100, dtype=np.uint8)).save(
        output_dir / "original.png"
    )
    Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(
        output_dir / "segmentation_mask.png"
    )
    Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(
        output_dir / "refined_talc_mask.png"
    )
    Image.fromarray(np.full(shape, 255, dtype=np.uint8)).save(
        output_dir / "sulfide_cv_mask.png"
    )
    (output_dir / "result.json").write_text(
        json.dumps(
            {
                "demo": False,
                "areas": {"segmentation": {"percent": 0.0}},
                "classification": {"talc_percent": 0.0},
                "sulfide": {
                    "code": "ordinary",
                    "label_ru": "рядовая руда",
                    "confidence": 0.8,
                },
                "sulfide_segmentation": {"selected": "cv"},
                "timings_seconds": {"pipeline_total": 1.0},
            }
        ),
        encoding="utf-8",
    )


def test_manual_polygons_add_remove_and_reset_from_cv_base(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    _prepare_artifacts(output_dir)
    settings = JobSettings(talc_threshold_percent=10)
    polygon = {
        "operation": "add",
        "points": [
            {"x": 0.0, "y": 0.0},
            {"x": 0.6, "y": 0.0},
            {"x": 0.6, "y": 0.6},
            {"x": 0.0, "y": 0.6},
        ],
    }

    added = apply_manual_talc_edits(output_dir, settings, [polygon])
    assert added["classification"]["code"] == "talc_bearing"
    assert added["talc"]["talc_percent"] > 10
    talc = np.asarray(Image.open(output_dir / "refined_talc_mask.png")) > 0
    sulfide = np.asarray(Image.open(output_dir / "sulfide_cv_mask.png")) > 0
    assert not np.any(talc & sulfide)

    reset = apply_manual_talc_edits(output_dir, settings, [])
    assert reset["talc"]["talc_percent"] == 0.0
    assert reset["classification"]["code"] == "ordinary"
    assert not np.any(
        np.asarray(Image.open(output_dir / "refined_talc_mask.png")) > 0
    )
