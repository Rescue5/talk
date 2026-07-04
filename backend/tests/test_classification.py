import numpy as np
from PIL import Image

from app.processor import (
    InferenceProcessor,
    combine_classification,
    cv_config_with_threshold,
    save_semantic_overlays,
)
from app.schemas import JobSettings


def test_talc_classification_wins_without_sulfide() -> None:
    result = combine_classification({"code": "talc_bearing"}, None)
    assert result == {
        "code": "talc_bearing",
        "label_ru": "оталькованная руда",
        "confidence": None,
        "source": "talc_pipeline",
    }


def test_non_talc_uses_sulfide_classification() -> None:
    result = combine_classification(
        {"code": "non_talc_bearing"},
        {
            "code": "difficult",
            "label_ru": "трудно-обогатимая руда",
            "confidence": 0.77,
        },
    )
    assert result["code"] == "difficult"
    assert result["confidence"] == 0.77
    assert result["source"] == "sulfide_model"


def test_non_talc_without_sulfide_has_no_invented_result() -> None:
    assert (
        combine_classification({"code": "non_talc_bearing"}, None)
        is None
    )


def test_sulfide_confidence_matches_threshold_selected_class() -> None:
    result = InferenceProcessor._sulfide_from_probabilities(
        {
            "probability_ordinary": 0.4,
            "probability_difficult": 0.6,
        },
        threshold=0.7,
    )
    assert result["code"] == "ordinary"
    assert result["confidence"] == 0.4


def test_threshold_defaults_match_api_contract() -> None:
    settings = JobSettings()
    assert settings.segmentation_threshold == 0.5
    assert settings.cv_threshold == 0.55


def test_cv_threshold_is_job_local_and_keeps_grow_below_seed() -> None:
    original = {
        "pipeline": [
            {
                "type": "hysteresis",
                "enabled": True,
                "params": {
                    "seed_threshold": 0.45,
                    "grow_threshold": 0.15,
                    "strong_response_threshold": 0.58,
                },
            }
        ]
    }
    adjusted, applied = cv_config_with_threshold(original, 0.1)
    params = adjusted["pipeline"][0]["params"]
    assert params["seed_threshold"] == 0.1
    assert params["grow_threshold"] == 0.05
    assert params["strong_response_threshold"] == 0.58
    assert applied["grow_threshold"] < applied["seed_threshold"]
    assert original["pipeline"][0]["params"]["seed_threshold"] == 0.45
    assert original["pipeline"][0]["params"]["grow_threshold"] == 0.15


def test_semantic_overlays_are_independent_transparent_rgba(tmp_path) -> None:
    coarse = np.zeros((20, 20), dtype=np.uint8)
    coarse[3:17, 3:17] = 1
    talc = np.zeros((20, 20), dtype=np.uint8)
    talc[8:12, 8:12] = 1
    save_semantic_overlays(tmp_path, coarse, talc)

    coarse_rgba = np.asarray(Image.open(tmp_path / "coarse_overlay.png"))
    talc_rgba = np.asarray(Image.open(tmp_path / "talc_overlay.png"))
    assert coarse_rgba.shape == (20, 20, 4)
    assert talc_rgba.shape == (20, 20, 4)
    assert coarse_rgba[0, 0, 3] == 0
    assert coarse_rgba[3, 5].tolist() == [255, 210, 0, 255]
    assert coarse_rgba[10, 10, 3] == 0
    assert talc_rgba[0, 0, 3] == 0
    assert talc_rgba[9, 9].tolist() == [255, 40, 40, 180]
