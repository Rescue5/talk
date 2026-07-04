from __future__ import annotations

import os
import shutil
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

from .processor import ModelUnavailable, StageCallback
from .schemas import JobSettings
from .storage import JobStore, utc_now


class Processor(Protocol):
    def process(
        self,
        image_path: Path,
        output_dir: Path,
        settings: JobSettings,
        progress: StageCallback,
    ) -> dict[str, Any]: ...


class JobManager:
    def __init__(self, store: JobStore, processor: Processor) -> None:
        self.store = store
        self.processor = processor
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ore-inference"
        )

    def submit(
        self,
        job_id: str,
        image_ids: list[str] | None = None,
        *,
        recompute_from: str = "segmentation",
    ) -> None:
        self.executor.submit(self._run, job_id, image_ids, recompute_from)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)

    def _run(
        self,
        job_id: str,
        image_ids: list[str] | None,
        recompute_from: str,
    ) -> None:
        snapshot = self.store.get(job_id)
        selected = [
            image
            for image in snapshot["images"]
            if (image_ids is None or image["image_id"] in image_ids)
            and image["status"] in {"queued", "reprocessing"}
        ]
        if not selected:
            return
        selected_ids = {image["image_id"] for image in selected}

        def start(state: dict[str, Any]) -> None:
            state["status"] = "running"
            state["error"] = None
            for image in state["images"]:
                if image["image_id"] in selected_ids:
                    image["status"] = "running"
                    image["progress"] = {
                        "percent": 0.0,
                        "stage": recompute_from,
                        "message": "Processing started",
                    }
                    image["updated_at"] = utc_now()
            self._refresh_job_progress(state, "Processing started")

        self.store.update(job_id, start)

        for image_snapshot in selected:
            image_id = image_snapshot["image_id"]
            image_path = (
                self.store.job_dir(job_id)
                / "uploads"
                / image_snapshot["stored_name"]
            )
            output_dir = self.store.job_dir(job_id) / "artifacts" / image_id
            settings = JobSettings.model_validate(image_snapshot["settings"])

            def progress(stage: str, local: float, message: str | None) -> None:
                local_percent = round(max(0.0, min(1.0, local)) * 100.0, 2)

                def update(state: dict[str, Any]) -> None:
                    image = self._image(state, image_id)
                    image["progress"] = {
                        "percent": local_percent,
                        "stage": stage,
                        "message": message,
                    }
                    image["updated_at"] = utc_now()
                    self._refresh_job_progress(state, message)

                self.store.update(job_id, update)

            try:
                reprocess = getattr(self.processor, "reprocess", None)
                if recompute_from != "segmentation" and callable(reprocess):
                    staging_dir = output_dir.parent / (
                        f".{image_id}.{uuid.uuid4().hex}.staging"
                    )
                    try:
                        shutil.copytree(output_dir, staging_dir)
                        item = reprocess(
                            image_path,
                            staging_dir,
                            settings,
                            progress,
                            recompute_from=recompute_from,
                        )
                        for staged in staging_dir.rglob("*"):
                            if not staged.is_file():
                                continue
                            relative = staged.relative_to(staging_dir)
                            destination = output_dir / relative
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(staged, destination)
                    finally:
                        shutil.rmtree(staging_dir, ignore_errors=True)
                else:
                    item = self.processor.process(
                        image_path, output_dir, settings, progress
                    )
                item.update(
                    {
                        "image_id": image_id,
                        "filename": image_snapshot["filename"],
                    }
                )
                if item.get("artifacts"):
                    item["artifacts"] = {
                        key: (
                            value
                            if value.startswith("/api/")
                            else f"/api/jobs/{job_id}/artifacts/{image_id}/{value}"
                        )
                        for key, value in item["artifacts"].items()
                    }
            except ModelUnavailable as error:
                item = self._error_item(
                    image_id,
                    image_snapshot["filename"],
                    "model_unavailable",
                    error.payload(),
                    bool(snapshot["demo"]),
                )
            except Exception as error:  # per-image isolation boundary
                item = self._error_item(
                    image_id,
                    image_snapshot["filename"],
                    "failed",
                    {
                        "code": "processing_failed",
                        "message": f"{type(error).__name__}: {error}",
                    },
                    bool(snapshot["demo"]),
                )
                traceback.print_exc()

            terminal_progress = {
                "percent": 100.0,
                "stage": item["status"],
                "message": "Image processing finished",
            }
            item["settings"] = settings.model_dump()
            item["progress"] = terminal_progress

            def save_result(state: dict[str, Any]) -> None:
                state["items"] = [
                    current
                    for current in state["items"]
                    if current.get("image_id") != image_id
                ]
                state["items"].append(item)
                image = self._image(state, image_id)
                image["status"] = item["status"]
                image["settings"] = settings.model_dump()
                image["progress"] = terminal_progress
                image["updated_at"] = utc_now()
                self._refresh_job_progress(state, "Image processing finished")

            self.store.update(job_id, save_result)

        self.store.update(job_id, self._finish)
        self.store.prune()

    @staticmethod
    def _image(job: dict[str, Any], image_id: str) -> dict[str, Any]:
        return next(
            image for image in job["images"] if image["image_id"] == image_id
        )

    @staticmethod
    def _refresh_job_progress(
        state: dict[str, Any], message: str | None
    ) -> None:
        images = state["images"]
        total = len(images)
        completed = sum(
            image["status"] in {"completed", "failed", "model_unavailable"}
            for image in images
        )
        mean_percent = (
            sum(float(image["progress"].get("percent", 0.0)) for image in images)
            / total
            if total
            else 100.0
        )
        running = next(
            (image for image in images if image["status"] == "running"), None
        )
        state["progress"] = {
            "percent": round(mean_percent, 2),
            "stage": (
                running["progress"]["stage"]
                if running
                else ("completed" if completed == total else "queued")
            ),
            "completed_images": completed,
            "total_images": total,
            "message": message,
        }

    @classmethod
    def _finish(cls, state: dict[str, Any]) -> None:
        statuses = [image["status"] for image in state["images"]]
        if any(status in {"queued", "reprocessing", "running"} for status in statuses):
            state["status"] = "queued"
            cls._refresh_job_progress(state, "Waiting for queued image processing")
            return
        has_completed = "completed" in statuses
        has_error = any(
            status in {"failed", "model_unavailable"} for status in statuses
        )
        if has_completed and has_error:
            state["status"] = "partial_failed"
            state["error"] = None
        elif "model_unavailable" in statuses:
            state["status"] = "model_unavailable"
            failed_item = next(
                item
                for item in state["items"]
                if item["status"] == "model_unavailable"
            )
            state["error"] = failed_item["error"]
        elif statuses and all(status == "failed" for status in statuses):
            state["status"] = "failed"
            state["error"] = {
                "code": "all_images_failed",
                "message": "All uploaded images failed processing.",
            }
        else:
            state["status"] = "completed"
            state["error"] = None
        cls._refresh_job_progress(state, "Processing finished")
        state["progress"]["percent"] = 100.0
        state["progress"]["stage"] = state["status"]

    @staticmethod
    def _error_item(
        image_id: str,
        filename: str,
        status: str,
        error: dict[str, Any],
        demo: bool,
    ) -> dict[str, Any]:
        return {
            "image_id": image_id,
            "filename": filename,
            "status": status,
            "demo": demo,
            "classification": None,
            "talc": None,
            "sulfide": None,
            "sulfide_segmentation": None,
            "artifacts": {},
            "error": error,
        }
