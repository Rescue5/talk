from __future__ import annotations

import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    FastAPI,
    File,
    Form,
    Body,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import ValidationError

from .config import ServiceConfig
from .jobs import JobManager, Processor
from .processor import InferenceProcessor
from .schemas import (
    ImageHistory,
    JobHistory,
    JobPublic,
    JobResults,
    JobSettings,
    JobSettingsPatch,
)
from .storage import JobNotFound, JobStore

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _safe_display_name(raw: str | None) -> str:
    basename = Path(raw or "image").name
    stem = re.sub(r"[^\w.-]+", "_", Path(basename).stem, flags=re.UNICODE).strip(
        "._"
    )
    suffix = Path(basename).suffix.lower()
    return f"{(stem or 'image')[:120]}{suffix}"


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        key: job[key]
        for key in (
            "id",
            "status",
            "demo",
            "settings",
            "progress",
            "created_at",
            "updated_at",
            "error",
            "images",
        )
    }


def _result_items(job: dict[str, Any]) -> list[dict[str, Any]]:
    persisted = {
        item.get("image_id"): item
        for item in job["items"]
        if isinstance(item, dict)
    }
    items = []
    for image in job["images"]:
        result = dict(persisted.get(image["image_id"], {}))
        result.update(
            {
                "image_id": image["image_id"],
                "filename": image["filename"],
                "status": image["status"],
                "demo": job["demo"],
                "settings": image["settings"],
                "progress": image["progress"],
            }
        )
        result.setdefault("classification", None)
        result.setdefault("talc", None)
        result.setdefault("sulfide", None)
        result.setdefault("sulfide_segmentation", None)
        result.setdefault("timings", None)
        result.setdefault("artifacts", {})
        result.setdefault("error", None)
        items.append(result)
    return items


async def _save_uploads(
    files: list[UploadFile],
    upload_dir: Path,
    max_upload_bytes: int,
) -> list[dict[str, Any]]:
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required")
    upload_dir.mkdir(parents=True, exist_ok=True)
    inputs: list[dict[str, Any]] = []
    written: list[Path] = []
    try:
        for upload in files:
            filename = _safe_display_name(upload.filename)
            suffix = Path(filename).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                raise HTTPException(
                    status_code=415,
                    detail={"code": "unsupported_file_type", "filename": filename},
                )
            image_id = uuid.uuid4().hex[:16]
            stored_name = f"{image_id}{suffix}"
            destination = upload_dir / stored_name
            written.append(destination)
            size = 0
            with destination.open("wb") as stream:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail={
                                "code": "file_too_large",
                                "filename": filename,
                                "max_bytes": max_upload_bytes,
                            },
                        )
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "empty_file", "filename": filename},
                )
            inputs.append(
                {
                    "image_id": image_id,
                    "filename": filename,
                    "stored_name": stored_name,
                    "size_bytes": size,
                }
            )
            await upload.close()
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return inputs


def create_app(
    config: ServiceConfig | None = None,
    *,
    processor: Processor | None = None,
) -> FastAPI:
    service_config = config or ServiceConfig.from_env()
    store = JobStore(service_config.jobs_data_dir, max_history=50)
    manager = JobManager(store, processor or InferenceProcessor(service_config))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.startup()
        yield
        manager.shutdown()

    application = FastAPI(
        title="PyTorchi: Ore analyzer API",
        version="0.2.0",
        lifespan=lifespan,
    )
    application.state.config = service_config
    application.state.store = store
    application.state.manager = manager
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(service_config.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        current: ServiceConfig = request.app.state.config
        models = current.model_status()
        available = all(
            value["status"] == "configured"
            for value in models.values()
            if value.get("required", True)
        )
        return {
            "status": "ok" if available or current.demo_mode else "degraded",
            "service": "pytorchi-ore-analyzer",
            "demo_mode": current.demo_mode,
            "models": models,
        }

    @application.post(
        "/api/jobs",
        response_model=JobPublic,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_job(
        request: Request,
        files: Annotated[list[UploadFile], File(...)],
        settings: Annotated[str, Form()] = "{}",
    ) -> dict[str, Any]:
        try:
            parsed_settings = JobSettings.model_validate_json(settings)
        except (ValidationError, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_settings", "message": str(error)},
            ) from error
        current: ServiceConfig = request.app.state.config
        current_store: JobStore = request.app.state.store
        job_id = uuid.uuid4().hex
        upload_dir = current_store.job_dir(job_id) / "uploads"
        try:
            inputs = await _save_uploads(
                files, upload_dir, current.max_upload_bytes
            )
        except Exception:
            shutil.rmtree(current_store.job_dir(job_id), ignore_errors=True)
            raise

        job = current_store.create(
            job_id,
            parsed_settings.model_dump(),
            inputs,
            demo=current.demo_mode,
        )
        request.app.state.manager.submit(
            job_id, [item["image_id"] for item in inputs]
        )
        return _public_job(job)

    @application.get("/api/jobs", response_model=JobHistory)
    def list_jobs(request: Request, limit: int = 50) -> dict[str, Any]:
        return {
            "items": [
                _public_job(job) for job in request.app.state.store.list(limit)
            ]
        }

    @application.get("/api/history", response_model=ImageHistory)
    def image_history(request: Request, limit: int = 50) -> dict[str, Any]:
        return {"items": request.app.state.store.history(limit)}

    @application.post(
        "/api/jobs/{job_id}/images",
        response_model=JobPublic,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def append_images(
        job_id: str,
        request: Request,
        files: Annotated[list[UploadFile], File(...)],
        settings: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        current_store: JobStore = request.app.state.store
        try:
            job = current_store.get(job_id)
        except JobNotFound as error:
            raise HTTPException(status_code=404, detail="Job not found") from error
        try:
            parsed = (
                JobSettings.model_validate_json(settings)
                if settings is not None
                else JobSettings.model_validate(job["settings"])
            )
        except (ValidationError, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_settings", "message": str(error)},
            ) from error
        inputs = await _save_uploads(
            files,
            current_store.job_dir(job_id) / "uploads",
            request.app.state.config.max_upload_bytes,
        )
        updated = current_store.add_images(
            job_id, inputs, parsed.model_dump()
        )
        request.app.state.manager.submit(
            job_id, [item["image_id"] for item in inputs]
        )
        return _public_job(updated)

    def apply_settings_patch(
        job_id: str,
        request: Request,
        payload: dict[str, Any],
        forced_image_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        current_store: JobStore = request.app.state.store
        try:
            job = current_store.get(job_id)
        except JobNotFound as error:
            raise HTTPException(status_code=404, detail="Job not found") from error
        payload = dict(payload)
        image_ids = forced_image_ids or payload.pop("image_ids", None)
        if image_ids is not None and (
            not isinstance(image_ids, list)
            or not all(isinstance(value, str) for value in image_ids)
        ):
            raise HTTPException(
                status_code=422, detail="image_ids must be an array of strings"
            )
        raw_patch = payload.pop("settings", payload)
        if payload and raw_patch is not payload:
            raise HTTPException(status_code=422, detail="Unexpected patch fields")
        try:
            patch = JobSettingsPatch.model_validate(raw_patch)
        except ValidationError as error:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_settings", "message": str(error)},
            ) from error
        changes = patch.model_dump(exclude_none=True)
        if not changes:
            return _public_job(job)
        target_ids = (
            set(image_ids)
            if image_ids is not None
            else {image["image_id"] for image in job["images"]}
        )
        known = {image["image_id"] for image in job["images"]}
        if not target_ids or not target_ids <= known:
            raise HTTPException(status_code=404, detail="Image not found")
        if any(
            image["image_id"] in target_ids
            and image["status"] in {"queued", "running", "reprocessing"}
            for image in job["images"]
        ):
            raise HTTPException(
                status_code=409,
                detail="Cannot change settings while selected images are processing",
            )

        changed_keys = {
            key
            for key, value in changes.items()
            if any(
                image["image_id"] in target_ids
                and image["settings"].get(key) != value
                for image in job["images"]
            )
        }
        if not changed_keys:
            return _public_job(job)
        if "mode" in changed_keys:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Segmentation mode is fixed after upload; create a new job "
                    "to change overlap mode"
                ),
            )
        changes = {key: value for key, value in changes.items() if key in changed_keys}

        if "segmentation_threshold" in changed_keys:
            recompute_from = "segmentation_threshold"
        elif "cv_threshold" in changed_keys:
            recompute_from = "cv_refinement"
        else:
            recompute_from = "classification"

        def apply(state: dict[str, Any]) -> None:
            state["settings"] = JobSettings.model_validate(
                {**state["settings"], **changes}
            ).model_dump()
            state["status"] = "queued"
            state["error"] = None
            for image in state["images"]:
                if image["image_id"] not in target_ids:
                    continue
                image["settings"] = JobSettings.model_validate(
                    {**image["settings"], **changes}
                ).model_dump()
                image["status"] = "reprocessing"
                image["progress"] = {
                    "percent": 0.0,
                    "stage": recompute_from,
                    "message": "Settings changed; reprocessing queued",
                }

        updated = current_store.update(job_id, apply)
        request.app.state.manager.submit(
            job_id, list(target_ids), recompute_from=recompute_from
        )
        return _public_job(updated)

    @application.patch(
        "/api/jobs/{job_id}/settings",
        response_model=JobPublic,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def patch_settings(
        job_id: str,
        request: Request,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        return apply_settings_patch(job_id, request, payload)

    @application.patch(
        "/api/jobs/{job_id}/images/{image_id}/settings",
        response_model=JobPublic,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def patch_image_settings(
        job_id: str,
        image_id: str,
        request: Request,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        return apply_settings_patch(
            job_id, request, payload, forced_image_ids=[image_id]
        )

    @application.get("/api/jobs/{job_id}", response_model=JobPublic)
    def get_job(job_id: str, request: Request) -> dict[str, Any]:
        try:
            job = request.app.state.store.get(job_id)
        except JobNotFound as error:
            raise HTTPException(status_code=404, detail="Job not found") from error
        return _public_job(job)

    @application.get("/api/jobs/{job_id}/results", response_model=JobResults)
    def get_results(job_id: str, request: Request) -> dict[str, Any]:
        try:
            job = request.app.state.store.get(job_id)
        except JobNotFound as error:
            raise HTTPException(status_code=404, detail="Job not found") from error
        return {
            "job_id": job_id,
            "status": job["status"],
            "demo": job["demo"],
            "items": _result_items(job),
        }

    @application.get(
        "/api/jobs/{job_id}/artifacts/{image_id}/{artifact_name}",
        response_class=FileResponse,
    )
    def get_artifact(
        job_id: str, image_id: str, artifact_name: str, request: Request
    ) -> FileResponse:
        try:
            job = request.app.state.store.get(job_id)
        except JobNotFound as error:
            raise HTTPException(status_code=404, detail="Job not found") from error
        item = next(
            (candidate for candidate in job["items"] if candidate["image_id"] == image_id),
            None,
        )
        if item is None or artifact_name not in {
            Path(url).name for url in item.get("artifacts", {}).values()
        }:
            raise HTTPException(status_code=404, detail="Artifact not found")
        root = (
            request.app.state.store.job_dir(job_id) / "artifacts" / image_id
        ).resolve()
        artifact = (root / Path(artifact_name).name).resolve()
        if artifact.parent != root or not artifact.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(artifact)

    return application


app = create_app()
