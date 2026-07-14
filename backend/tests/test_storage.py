from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas import JobSettings
from app.storage import JobNotFound, JobStore


def input_record(index: int) -> dict:
    return {
        "image_id": f"image-{index}",
        "filename": f"{index}.png",
        "stored_name": f"{index}.png",
        "size_bytes": 1,
    }


def complete(store: JobStore, job_id: str, index: int) -> None:
    def update(job):
        image = job["images"][0]
        image["status"] = "completed"
        image["updated_at"] = f"2026-07-04T00:00:{index:02d}+00:00"
        job["status"] = "completed"
        job["items"] = [
            {
                "image_id": image["image_id"],
                "filename": image["filename"],
                "status": "completed",
                "classification": {"code": "ordinary"},
                "talc": {"talc_percent": 1.0},
                "sulfide": {"probability_ordinary": 0.8},
                "artifacts": {"original": "/api/example.png"},
                "error": None,
            }
        ]

    store.update(job_id, update)


def test_history_and_cleanup_are_global_per_image(tmp_path: Path) -> None:
    store = JobStore(tmp_path, max_history=2)
    settings = JobSettings().model_dump()
    for index in range(3):
        job_id = f"job-{index}"
        store.create(job_id, settings, [input_record(index)], demo=False)
        complete(store, job_id, index)
    store.prune()

    history = store.history()
    assert [item["image_id"] for item in history] == ["image-2", "image-1"]
    with pytest.raises(JobNotFound):
        store.get("job-0")


def test_v1_manifest_is_migrated_to_per_image_state(tmp_path: Path) -> None:
    job_dir = tmp_path / "legacy"
    job_dir.mkdir(parents=True)
    legacy = {
        "id": "legacy",
        "status": "completed",
        "demo": False,
        "settings": JobSettings().model_dump(),
        "progress": {
            "percent": 100,
            "stage": "completed",
            "completed_images": 1,
            "total_images": 1,
        },
        "inputs": [input_record(1)],
        "items": [
            {
                "image_id": "image-1",
                "filename": "1.png",
                "status": "completed",
                "classification": {"code": "ordinary"},
                "artifacts": {},
            }
        ],
        "created_at": "2026-07-04T00:00:00+00:00",
        "updated_at": "2026-07-04T00:00:01+00:00",
    }
    (job_dir / "job.json").write_text(json.dumps(legacy), encoding="utf-8")

    migrated = JobStore(tmp_path).get("legacy")
    assert migrated["images"][0]["status"] == "completed"
    assert migrated["images"][0]["settings"]["mode"] == "overlap"
    persisted = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert persisted["images"][0]["image_id"] == "image-1"


def test_cache_limit_persists_and_clear_removes_terminal_images(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path, max_history=50)
    settings = JobSettings().model_dump()
    store.create("job", settings, [input_record(1)], demo=False)
    complete(store, "job", 1)

    store.set_max_history(25)
    assert JobStore(tmp_path, max_history=50).cache_info()["max_images"] == 25

    store.clear_history()
    info = store.cache_info()
    assert info["stored_images"] == 0
    assert info["max_images"] == 25
    with pytest.raises(JobNotFound):
        store.get("job")
