import numpy as np
from PIL import Image

from app.processor import (
    InferenceProcessor,
    assess_image_quality,
    combine_classification,
    cv_config_with_threshold,
    run_sulfide_segmentation,
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
    assert settings.cv_threshold == 0.35


def test_underexposed_image_gets_stability_warning() -> None:
    image = np.full((64, 64, 3), 25, dtype=np.uint8)
    quality = assess_image_quality(image)
    assert quality["dark_pixel_fraction"] == 1.0
    assert quality["warnings"][0]["code"] == "underexposed_image"


def test_normally_exposed_image_has_no_warning() -> None:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert assess_image_quality(image)["warnings"] == []


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
    sulfide_cv = np.zeros((20, 20), dtype=np.uint8)
    sulfide_cv[2:4, 2:4] = 255
    sulfide_sam = np.zeros((20, 20), dtype=np.uint8)
    sulfide_sam[14:17, 14:17] = 255
    save_semantic_overlays(tmp_path, coarse, talc, sulfide_cv, sulfide_sam)

    coarse_rgba = np.asarray(Image.open(tmp_path / "coarse_overlay.png"))
    talc_rgba = np.asarray(Image.open(tmp_path / "talc_overlay.png"))
    sulfide_cv_rgba = np.asarray(Image.open(tmp_path / "sulfide_cv_overlay.png"))
    sulfide_sam_rgba = np.asarray(Image.open(tmp_path / "sulfide_sam_overlay.png"))
    assert coarse_rgba.shape == (20, 20, 4)
    assert talc_rgba.shape == (20, 20, 4)
    assert coarse_rgba[0, 0, 3] == 0
    assert coarse_rgba[3, 5].tolist() == [255, 210, 0, 255]
    assert coarse_rgba[10, 10, 3] == 0
    assert talc_rgba[0, 0, 3] == 0
    assert talc_rgba[9, 9].tolist() == [255, 40, 40, 180]
    assert sulfide_cv_rgba[2, 2].tolist() == [255, 178, 38, 175]
    assert sulfide_sam_rgba[14, 14].tolist() == [36, 144, 255, 180]


def test_sulfide_segmentation_excludes_talc_for_cv_and_sam() -> None:
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    talc = np.zeros((12, 12), dtype=np.uint8)
    talc[2:5, 2:5] = 1

    def fake_segmenter(image_rgb, config):
        mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        mask[1:8, 1:8] = 255
        return mask

    class FakeRefiner:
        last_stats = {
            "components_considered": 1,
            "components_refined": 1,
            "components_fallback": 0,
        }
        last_timings = {
            "set_image_ms": 1.0,
            "prompts_ms": 2.0,
            "total_refine_ms": 3.0,
        }

        def refine(self, image_rgb, cv_mask, **kwargs):
            return np.full(cv_mask.shape, 255, dtype=np.uint8)

    result = run_sulfide_segmentation(
        image,
        talc,
        {"sam": {"enabled": True}},
        fake_segmenter,
        sam_refiner=FakeRefiner(),
    )

    talc_bool = talc.astype(bool)
    assert not np.any(result["cv_mask"].astype(bool) & talc_bool)
    assert not np.any(result["sam_mask"].astype(bool) & talc_bool)
    assert result["summary"]["selected"] == "sam"
    assert result["summary"]["sam"]["pixel_count"] == 12 * 12 - int(talc_bool.sum())


def test_sulfide_segmentation_uses_cv_when_sam_is_missing() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    talc = np.zeros((8, 8), dtype=np.uint8)

    def fake_segmenter(image_rgb, config):
        mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        mask[1:4, 1:4] = 255
        return mask

    result = run_sulfide_segmentation(
        image,
        talc,
        {"sam": {"enabled": True}},
        fake_segmenter,
        sam_error={
            "code": "sam_checkpoint_not_configured",
            "message": "MobileSAM checkpoint is not configured.",
        },
    )

    assert result["sam_mask"] is None
    assert result["summary"]["selected"] == "cv"
    assert result["summary"]["sam_error"]["code"] == "sam_checkpoint_not_configured"
    assert result["summary"]["cv"]["pixel_count"] == 9
