from __future__ import annotations

import json
import os
import shutil
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobNotFound(KeyError):
    pass


class JobStore:
    """Thread-safe filesystem storage. Each mutation is atomically persisted."""

    def __init__(self, root: str | Path, max_history: int = 50) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_history = max(1, min(int(max_history), 50))
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        for manifest in self.root.glob("*/job.json"):
            try:
                with manifest.open("r", encoding="utf-8") as stream:
                    job = json.load(stream)
                self._normalize(job)
                if job.get("status") in {"queued", "running"}:
                    job["status"] = "failed"
                    job["error"] = {
                        "code": "service_restarted",
                        "message": "Processing was interrupted by a service restart.",
                    }
                    job["updated_at"] = utc_now()
                    for image in job["images"]:
                        if image["status"] in {"queued", "running", "reprocessing"}:
                            image["status"] = "failed"
                            image["progress"] = {
                                "percent": image["progress"].get("percent", 0.0),
                                "stage": "failed",
                                "message": "Interrupted by service restart",
                            }
                            image["updated_at"] = job["updated_at"]
                    self._persist(job)
                self._jobs[str(job["id"])] = job
                self._persist(job)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        self.prune()

    def _normalize(self, job: dict[str, Any]) -> None:
        """Migrate v1 job manifests without breaking their public contract."""

        job.setdefault("settings", {})
        job.setdefault("inputs", [])
        job.setdefault("items", [])
        job.setdefault("error", None)
        job.setdefault("demo", False)
        created_at = job.setdefault("created_at", utc_now())
        job.setdefault("updated_at", created_at)
        results = {
            item.get("image_id"): item
            for item in job["items"]
            if isinstance(item, dict) and item.get("image_id")
        }
        if "images" not in job:
            images = []
            for input_item in job["inputs"]:
                result = results.get(input_item.get("image_id"))
                status = result.get("status", "queued") if result else "queued"
                terminal = status in {"completed", "failed", "model_unavailable"}
                images.append(
                    {
                        **input_item,
                        "status": status,
                        "settings": deepcopy(job["settings"]),
                        "progress": {
                            "percent": 100.0 if terminal else 10.0,
                            "stage": status if terminal else "upload",
                            "message": None,
                        },
                        "created_at": created_at,
                        "updated_at": job["updated_at"],
                    }
                )
            job["images"] = images
        for image in job["images"]:
            image.setdefault("settings", deepcopy(job["settings"]))
            image.setdefault("status", "queued")
            image.setdefault(
                "progress",
                {"percent": 10.0, "stage": "upload", "message": None},
            )
            image.setdefault("created_at", created_at)
            image.setdefault("updated_at", job["updated_at"])

    def create(
        self,
        job_id: str,
        settings: dict[str, Any],
        inputs: list[dict[str, Any]],
        *,
        demo: bool,
    ) -> dict[str, Any]:
        now = utc_now()
        images = [
            {
                **input_item,
                "status": "queued",
                "settings": deepcopy(settings),
                "progress": {
                    "percent": 10.0,
                    "stage": "upload",
                    "message": "File uploaded",
                },
                "created_at": now,
                "updated_at": now,
            }
            for input_item in inputs
        ]
        job = {
            "id": job_id,
            "status": "queued",
            "demo": demo,
            "settings": settings,
            "progress": {
                "percent": 10.0,
                "stage": "upload",
                "completed_images": 0,
                "total_images": len(inputs),
                "message": "Files uploaded",
            },
            "inputs": inputs,
            "images": images,
            "items": [],
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)
            self.prune()
        return deepcopy(job)

    def add_images(
        self,
        job_id: str,
        inputs: list[dict[str, Any]],
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        now = utc_now()

        def add(job: dict[str, Any]) -> None:
            job["inputs"].extend(inputs)
            for input_item in inputs:
                job["images"].append(
                    {
                        **input_item,
                        "status": "queued",
                        "settings": deepcopy(settings),
                        "progress": {
                            "percent": 10.0,
                            "stage": "upload",
                            "message": "File uploaded",
                        },
                        "created_at": now,
                        "updated_at": now,
                    }
                )
            if job["status"] != "running":
                job["status"] = "queued"
            job["error"] = None
            job["progress"] = {
                "percent": 10.0,
                "stage": "upload",
                "completed_images": sum(
                    image["status"] == "completed" for image in job["images"]
                ),
                "total_images": len(job["images"]),
                "message": f"Added {len(inputs)} image(s)",
            }

        return self.update(job_id, add)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda item: item.get("created_at", ""),
                reverse=True,
            )
            return deepcopy(jobs[: max(1, min(int(limit), self.max_history))])

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            records: list[dict[str, Any]] = []
            for job in self._jobs.values():
                results = {
                    item.get("image_id"): item
                    for item in job["items"]
                    if isinstance(item, dict)
                }
                for image in job["images"]:
                    if image["status"] not in {
                        "completed",
                        "failed",
                        "model_unavailable",
                    }:
                        continue
                    result = results.get(image["image_id"], {})
                    records.append(
                        {
                            # `id` keeps older job-history clients functional;
                            # `job_id` + `image_id` is the canonical v2 identity.
                            "id": job["id"],
                            "job_id": job["id"],
                            "image_id": image["image_id"],
                            "filename": image["filename"],
                            "status": image["status"],
                            "demo": job["demo"],
                            "classification": result.get("classification"),
                            "talc": result.get("talc"),
                            "sulfide": result.get("sulfide"),
                            "artifacts": result.get("artifacts", {}),
                            "settings": image["settings"],
                            "created_at": image["created_at"],
                            "updated_at": image["updated_at"],
                            "error": result.get("error"),
                            "progress": {
                                "percent": 100.0,
                                "stage": image["status"],
                                "completed_images": 1,
                                "total_images": 1,
                                "message": None,
                            },
                            "images": [deepcopy(image)],
                        }
                    )
            records.sort(key=lambda item: item["updated_at"], reverse=True)
            return deepcopy(records[: max(1, min(int(limit), self.max_history))])

    def prune(self) -> None:
        """Keep the newest terminal image records globally, preserving active images."""

        with self._lock:
            terminal: list[tuple[str, dict[str, Any]]] = sorted(
                (
                    (str(job["id"]), image)
                    for job in self._jobs.values()
                    for image in job["images"]
                    if image["status"]
                    in {"completed", "failed", "model_unavailable"}
                ),
                key=lambda pair: pair[1].get("updated_at", ""),
                reverse=True,
            )
            affected: set[str] = set()
            for job_id, image in terminal[self.max_history :]:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                image_id = image["image_id"]
                job["images"] = [
                    current
                    for current in job["images"]
                    if current["image_id"] != image_id
                ]
                job["inputs"] = [
                    current
                    for current in job["inputs"]
                    if current["image_id"] != image_id
                ]
                job["items"] = [
                    current
                    for current in job["items"]
                    if current.get("image_id") != image_id
                ]
                upload = self.root / job_id / "uploads" / image.get("stored_name", "")
                if upload.is_file():
                    upload.unlink()
                shutil.rmtree(
                    self.root / job_id / "artifacts" / image_id,
                    ignore_errors=True,
                )
                affected.add(job_id)
            for job_id in affected:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                if not job["images"]:
                    self._jobs.pop(job_id, None)
                    shutil.rmtree(self.root / job_id, ignore_errors=True)
                else:
                    job["progress"]["total_images"] = len(job["images"])
                    job["progress"]["completed_images"] = sum(
                        image["status"]
                        in {"completed", "failed", "model_unavailable"}
                        for image in job["images"]
                    )
                    self._persist(job)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                return deepcopy(self._jobs[job_id])
            except KeyError as error:
                raise JobNotFound(job_id) from error

    def update(
        self, job_id: str, updater: Callable[[dict[str, Any]], None]
    ) -> dict[str, Any]:
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as error:
                raise JobNotFound(job_id) from error
            updater(job)
            job["updated_at"] = utc_now()
            self._persist(job)
            return deepcopy(job)

    def job_dir(self, job_id: str) -> Path:
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _persist(self, job: dict[str, Any]) -> None:
        destination = self.root / str(job["id"]) / "job.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(job, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
