from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal[
    "queued",
    "running",
    "completed",
    "partial_failed",
    "failed",
    "model_unavailable",
]


class JobSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["overlap", "no_overlap"] = "overlap"
    segmentation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    cv_threshold: float = Field(default=0.55, gt=0.0, le=1.0)
    talc_threshold_percent: float = Field(default=10.0, ge=0.0, le=100.0)
    sulfide_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class JobSettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["overlap", "no_overlap"] | None = None
    segmentation_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    cv_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    talc_threshold_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    sulfide_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class Progress(BaseModel):
    percent: float = Field(ge=0.0, le=100.0)
    stage: str
    completed_images: int = Field(ge=0)
    total_images: int = Field(ge=0)
    message: str | None = None


class ImageProgress(BaseModel):
    percent: float = Field(ge=0.0, le=100.0)
    stage: str
    message: str | None = None


class ImageState(BaseModel):
    image_id: str
    filename: str
    status: str
    settings: JobSettings
    progress: ImageProgress
    created_at: str
    updated_at: str


class JobPublic(BaseModel):
    id: str
    status: JobStatus
    demo: bool
    settings: JobSettings
    progress: Progress
    created_at: str
    updated_at: str
    error: dict | None = None
    images: list[ImageState] = Field(default_factory=list)


class ResultItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    image_id: str
    filename: str
    status: Literal[
        "queued",
        "running",
        "reprocessing",
        "completed",
        "failed",
        "model_unavailable",
    ]
    demo: bool
    classification: dict | None
    talc: dict | None
    sulfide: dict | None
    sulfide_segmentation: dict | None = None
    timings: dict[str, float] | None = None
    artifacts: dict[str, str]
    error: dict | None = None
    settings: JobSettings | None = None
    progress: ImageProgress | None = None


class JobResults(BaseModel):
    job_id: str
    status: JobStatus
    demo: bool
    items: list[ResultItem]


class JobHistory(BaseModel):
    items: list[JobPublic]


class HistoryItem(BaseModel):
    id: str
    job_id: str
    image_id: str
    filename: str
    status: Literal["completed", "failed", "model_unavailable"]
    demo: bool
    classification: dict | None
    talc: dict | None
    sulfide: dict | None
    sulfide_segmentation: dict | None = None
    artifacts: dict[str, str]
    settings: JobSettings
    created_at: str
    updated_at: str
    error: dict | None = None
    progress: Progress
    images: list[ImageState]


class ImageHistory(BaseModel):
    items: list[HistoryItem]
