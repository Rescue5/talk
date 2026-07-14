from __future__ import annotations

import json
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.config import ServiceConfig
from app.main import create_app
from app.processor import ModelUnavailable


def config(tmp_path: Path, *, demo: bool = False) -> ServiceConfig:
    backend = Path(__file__).resolve().parents[1]
    return ServiceConfig(
        jobs_data_dir=tmp_path / "jobs",
        talc_checkpoint_path=None,
        talc_config_path=None,
        sulfide_checkpoint_path=None,
        sulfide_config_path=backend
        / "vendor"
        / "talk_sulfid"
        / "configs"
        / "classifier.yaml",
        sulfide_segmentation_config_path=backend
        / "vendor"
        / "talk_sulfid"
        / "cv_analysis"
        / "sulfide_candidates.yaml",
        sulfide_sam_checkpoint_path=None,
        sulfide_sam_device="cpu",
        talc_source_path=backend / "vendor" / "talk_combined",
        sulfide_source_path=backend / "vendor" / "talk_sulfid",
        demo_mode=demo,
        model_device="cpu",
        max_upload_bytes=1024 * 1024,
        allowed_origins=("http://localhost:5173",),
    )


class FakeProcessor:
    def process(self, image_path, output_dir, settings, progress):
        output_dir.mkdir(parents=True, exist_ok=True)
        progress("talc_segmentation", 0.1, None)
        progress("cv_refinement", 0.5, None)
        progress("sulfide_classification", 0.75, None)
        progress("export", 0.9, None)
        (output_dir / "result.json").write_text(
            json.dumps({"ok": True}), encoding="utf-8"
        )
        (output_dir / "original.png").write_bytes(image_path.read_bytes())
        progress("export", 1.0, None)
        return {
            "status": "completed",
            "demo": False,
            "classification": {
                "code": "ordinary",
                "label_ru": "рядовая руда",
                "confidence": 0.8,
            },
            "talc": {"talc_percent": 2.0},
            "sulfide": {"probability_ordinary": 0.8},
            "timings": {"pipeline_total": 0.01},
            "error": None,
            "artifacts": {
                "original": "original.png",
                "result": "result.json",
            },
        }


class PartialProcessor(FakeProcessor):
    def __init__(self) -> None:
        self.calls = 0

    def process(self, image_path, output_dir, settings, progress):
        self.calls += 1
        if self.calls == 2:
            raise ModelUnavailable("sulfide", "checkpoint_not_found")
        return super().process(image_path, output_dir, settings, progress)


class BlockingProcessor(FakeProcessor):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def process(self, image_path, output_dir, settings, progress):
        self.started.set()
        assert self.release.wait(timeout=2)
        return super().process(image_path, output_dir, settings, progress)


class ReprocessProcessor(FakeProcessor):
    def __init__(self) -> None:
        self.recompute_from: list[str] = []

    def reprocess(
        self, image_path, output_dir, settings, progress, *, recompute_from
    ):
        self.recompute_from.append(recompute_from)
        return super().process(image_path, output_dir, settings, progress)


class ManualEditProcessor(FakeProcessor):
    def process(self, image_path, output_dir, settings, progress):
        item = super().process(image_path, output_dir, settings, progress)
        shape = np.asarray(Image.open(output_dir / "original.png")).shape[:2]
        empty = np.zeros(shape, dtype=np.uint8)
        Image.fromarray(empty).save(output_dir / "segmentation_mask.png")
        Image.fromarray(empty).save(output_dir / "refined_talc_mask.png")
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
                    "timings_seconds": {"pipeline_total": 0.01},
                }
            ),
            encoding="utf-8",
        )
        item["artifacts"].update(
            {
                "segmentation_mask": "segmentation_mask.png",
                "refined_talc_mask": "refined_talc_mask.png",
                "sulfide_cv_mask": "sulfide_cv_mask.png",
                "talc_overlay": "talc_overlay.png",
                "coarse_overlay": "coarse_overlay.png",
                "sulfide_cv_overlay": "sulfide_cv_overlay.png",
                "overlay": "overlay.png",
            }
        )
        return item


def image_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "source.png"
    Image.fromarray(np.full((12, 16, 3), 128, dtype=np.uint8)).save(path)
    return path.read_bytes()


def wait_for_terminal(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(100):
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] not in {"queued", "running"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_job_lifecycle_and_artifact(tmp_path: Path) -> None:
    app = create_app(config(tmp_path), processor=FakeProcessor())
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files=[
                (
                    "files",
                    ("../../unsafe name.png", image_bytes(tmp_path), "image/png"),
                )
            ],
            data={
                "settings": json.dumps(
                    {
                        "mode": "no_overlap",
                        "segmentation_threshold": 0.42,
                        "cv_threshold": 0.61,
                        "talc_threshold_percent": 10,
                        "sulfide_threshold": 0.4,
                    }
                )
            },
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert response.json()["settings"]["segmentation_threshold"] == 0.42
        assert response.json()["settings"]["cv_threshold"] == 0.61
        terminal = wait_for_terminal(client, job_id)
        assert terminal["status"] == "completed"
        assert terminal["progress"]["percent"] == 100

        results = client.get(f"/api/jobs/{job_id}/results").json()
        assert results["items"][0]["filename"] == "unsafe_name.png"
        assert results["items"][0]["classification"]["code"] == "ordinary"
        artifact_url = results["items"][0]["artifacts"]["result"]
        artifact = client.get(artifact_url)
        assert artifact.status_code == 200
        assert artifact.json() == {"ok": True}
        original = client.get(results["items"][0]["artifacts"]["original"])
        assert original.status_code == 200
        assert original.headers["content-type"] == "image/png"


def test_missing_checkpoint_is_model_unavailable(tmp_path: Path) -> None:
    app = create_app(config(tmp_path))
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "degraded"
        response = client.post(
            "/api/jobs",
            files={"files": ("image.png", image_bytes(tmp_path), "image/png")},
            data={"settings": "{}"},
        )
        job = wait_for_terminal(client, response.json()["id"])
        assert job["status"] == "model_unavailable"
        results = client.get(f"/api/jobs/{job['id']}/results").json()
        assert results["items"][0]["classification"] is None
        assert results["items"][0]["error"]["code"] == "model_unavailable"


def test_completed_results_remain_available_when_another_image_lacks_model(
    tmp_path: Path,
) -> None:
    app = create_app(config(tmp_path), processor=PartialProcessor())
    payload = image_bytes(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files=[
                ("files", ("first.png", payload, "image/png")),
                ("files", ("second.png", payload, "image/png")),
            ],
            data={"settings": "{}"},
        )
        job = wait_for_terminal(client, response.json()["id"])
        assert job["status"] == "partial_failed"
        results = client.get(f"/api/jobs/{job['id']}/results").json()
        assert [item["status"] for item in results["items"]] == [
            "completed",
            "model_unavailable",
        ]


def test_invalid_settings_and_extension(tmp_path: Path) -> None:
    app = create_app(config(tmp_path), processor=FakeProcessor())
    with TestClient(app) as client:
        invalid = client.post(
            "/api/jobs",
            files={"files": ("image.png", image_bytes(tmp_path), "image/png")},
            data={"settings": '{"mode":"unknown"}'},
        )
        assert invalid.status_code == 422
        unsupported = client.post(
            "/api/jobs",
            files={"files": ("payload.exe", b"x", "application/octet-stream")},
            data={"settings": "{}"},
        )
        assert unsupported.status_code == 415


def test_cors_allows_image_settings_patch(tmp_path: Path) -> None:
    app = create_app(config(tmp_path), processor=FakeProcessor())
    with TestClient(app) as client:
        response = client.options(
            "/api/jobs/example/images/image/settings",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "PATCH",
            },
        )
        assert response.status_code == 200
        assert "PATCH" in response.headers["access-control-allow-methods"]
        assert "PUT" in response.headers["access-control-allow-methods"]
        assert "DELETE" in response.headers["access-control-allow-methods"]


def test_manual_talc_polygons_are_persisted_and_resettable(tmp_path: Path) -> None:
    app = create_app(config(tmp_path), processor=ManualEditProcessor())
    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            files={"files": ("image.png", image_bytes(tmp_path), "image/png")},
            data={"settings": "{}"},
        ).json()
        wait_for_terminal(client, created["id"])
        image_id = client.get(
            f"/api/jobs/{created['id']}/results"
        ).json()["items"][0]["image_id"]
        polygon = {
            "operation": "add",
            "points": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.7, "y": 0.0},
                {"x": 0.7, "y": 0.7},
                {"x": 0.0, "y": 0.7},
            ],
        }

        edited = client.put(
            f"/api/jobs/{created['id']}/images/{image_id}/talc-edits",
            json={"polygons": [polygon]},
        )
        assert edited.status_code == 200
        assert edited.json()["classification"]["code"] == "talc_bearing"
        assert edited.json()["manual_talc_edits"] == [polygon]
        persisted = client.get(
            f"/api/jobs/{created['id']}/results"
        ).json()["items"][0]
        assert persisted["manual_talc_edits"] == [polygon]

        reset = client.delete(
            f"/api/jobs/{created['id']}/images/{image_id}/talc-edits"
        )
        assert reset.status_code == 200
        assert reset.json()["talc"]["talc_percent"] == 0.0
        assert reset.json()["manual_talc_edits"] == []


def test_cache_settings_and_clear_history(tmp_path: Path) -> None:
    app = create_app(config(tmp_path), processor=FakeProcessor())
    with TestClient(app) as client:
        payload = image_bytes(tmp_path)
        created = client.post(
            "/api/jobs",
            files={"files": ("image.png", payload, "image/png")},
            data={"settings": "{}"},
        ).json()
        wait_for_terminal(client, created["id"])

        info = client.get("/api/cache")
        assert info.status_code == 200
        assert info.json()["stored_images"] == 1
        assert info.json()["size_bytes"] > 0

        updated = client.patch("/api/cache", json={"max_images": 20})
        assert updated.status_code == 200
        assert updated.json()["max_images"] == 20

        cleared = client.delete("/api/cache")
        assert cleared.status_code == 200
        assert cleared.json()["stored_images"] == 0
        assert cleared.json()["max_images"] == 20


def test_explicit_demo_is_marked(tmp_path: Path) -> None:
    app = create_app(config(tmp_path, demo=True))
    with TestClient(app) as client:
        assert client.get("/api/health").json()["demo_mode"] is True
        response = client.post(
            "/api/jobs",
            files={"files": ("image.png", image_bytes(tmp_path), "image/png")},
            data={"settings": "{}"},
        )
        job = wait_for_terminal(client, response.json()["id"])
        assert job["status"] == "completed"
        results = client.get(f"/api/jobs/{job['id']}/results").json()
        assert results["demo"] is True
        assert results["items"][0]["demo"] is True
        assert results["items"][0]["classification"]["source"] == "explicit_demo"
        assert client.get(results["items"][0]["artifacts"]["original"]).status_code == 200
        for key in (
            "coarse_overlay",
            "talc_overlay",
            "sulfide_cv_overlay",
            "sulfide_sam_overlay",
            "sulfide_cv_mask",
            "sulfide_sam_mask",
        ):
            layer = client.get(results["items"][0]["artifacts"][key])
            assert layer.status_code == 200
            assert layer.headers["content-type"] == "image/png"
        item = results["items"][0]
        assert item["sulfide_segmentation"]["selected"] == "sam"
        talc_mask = np.asarray(
            Image.open(BytesIO(client.get(item["artifacts"]["refined_talc_mask"]).content))
        ) > 0
        cv_mask = np.asarray(
            Image.open(BytesIO(client.get(item["artifacts"]["sulfide_cv_mask"]).content))
        ) > 0
        sam_mask = np.asarray(
            Image.open(BytesIO(client.get(item["artifacts"]["sulfide_sam_mask"]).content))
        ) > 0
        assert not np.any(cv_mask & talc_mask)
        assert not np.any(sam_mask & talc_mask)
        assert results["items"][0]["sulfide"] is not None
        image_id = results["items"][0]["image_id"]
        patched_settings = {
            **job["settings"],
            "sulfide_threshold": 0.4,
        }
        patched = client.patch(
            f"/api/jobs/{job['id']}/images/{image_id}/settings",
            json=patched_settings,
        )
        assert patched.status_code == 202
        wait_for_terminal(client, job["id"])
        updated = client.get(f"/api/jobs/{job['id']}/results").json()
        assert updated["items"][0]["classification"]["code"] == "difficult"


def test_pending_images_are_visible_before_processing_finishes(tmp_path: Path) -> None:
    processor = BlockingProcessor()
    app = create_app(config(tmp_path), processor=processor)
    with TestClient(app) as client:
        response = client.post(
            "/api/jobs",
            files={"files": ("image.png", image_bytes(tmp_path), "image/png")},
            data={"settings": "{}"},
        )
        job_id = response.json()["id"]
        assert processor.started.wait(timeout=1)
        results = client.get(f"/api/jobs/{job_id}/results").json()
        assert results["items"][0]["status"] == "running"
        assert results["items"][0]["progress"]["stage"] == "segmentation"
        processor.release.set()
        assert wait_for_terminal(client, job_id)["status"] == "completed"


def test_append_images_preserves_existing_results(tmp_path: Path) -> None:
    app = create_app(config(tmp_path), processor=FakeProcessor())
    payload = image_bytes(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            files={"files": ("first.png", payload, "image/png")},
            data={"settings": "{}"},
        ).json()
        wait_for_terminal(client, created["id"])
        appended = client.post(
            f"/api/jobs/{created['id']}/images",
            files={"files": ("second.png", payload, "image/png")},
        )
        assert appended.status_code == 202
        terminal = wait_for_terminal(client, created["id"])
        assert terminal["status"] == "completed"
        results = client.get(f"/api/jobs/{created['id']}/results").json()
        assert {item["filename"] for item in results["items"]} == {
            "first.png",
            "second.png",
        }
        assert len(client.get("/api/history").json()["items"]) == 2


def test_settings_patch_selects_minimal_recompute_stage(tmp_path: Path) -> None:
    processor = ReprocessProcessor()
    app = create_app(config(tmp_path), processor=processor)
    with TestClient(app) as client:
        created = client.post(
            "/api/jobs",
            files={"files": ("image.png", image_bytes(tmp_path), "image/png")},
            data={"settings": "{}"},
        ).json()
        wait_for_terminal(client, created["id"])
        patch = client.patch(
            f"/api/jobs/{created['id']}/images/{created['images'][0]['image_id']}/settings",
            json={
                **created["settings"],
                "talc_threshold_percent": 12.0,
            },
        )
        assert patch.status_code == 202
        wait_for_terminal(client, created["id"])
        assert processor.recompute_from[-1] == "classification"
        results = client.get(f"/api/jobs/{created['id']}/results").json()
        assert results["items"][0]["settings"]["talc_threshold_percent"] == 12.0

        patch = client.patch(
            f"/api/jobs/{created['id']}/settings",
            json={"cv_threshold": 0.62},
        )
        assert patch.status_code == 202
        wait_for_terminal(client, created["id"])
        assert processor.recompute_from[-1] == "cv_refinement"

        patch = client.patch(
            f"/api/jobs/{created['id']}/settings",
            json={"segmentation_threshold": 0.63},
        )
        assert patch.status_code == 202
        wait_for_terminal(client, created["id"])
        assert processor.recompute_from[-1] == "segmentation_threshold"

        mode_change = client.patch(
            f"/api/jobs/{created['id']}/settings",
            json={"mode": "no_overlap"},
        )
        assert mode_change.status_code == 422
