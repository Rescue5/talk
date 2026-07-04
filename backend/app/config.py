from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path_env(name: str, default: Path | None = None) -> Path | None:
    raw = os.getenv(name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default.resolve() if default is not None else None


@dataclass(frozen=True)
class ServiceConfig:
    jobs_data_dir: Path
    talc_checkpoint_path: Path | None
    talc_config_path: Path | None
    sulfide_checkpoint_path: Path | None
    sulfide_config_path: Path | None
    sulfide_segmentation_config_path: Path
    sulfide_sam_checkpoint_path: Path | None
    sulfide_sam_device: str
    talc_source_path: Path
    sulfide_source_path: Path
    demo_mode: bool
    model_device: str
    max_upload_bytes: int
    allowed_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        backend_root = Path(__file__).resolve().parents[1]
        vendor_root = backend_root / "vendor"
        model_device = os.getenv("MODEL_DEVICE", "auto").strip().lower()
        origins = tuple(
            item.strip()
            for item in os.getenv(
                "CORS_ORIGINS",
                "http://localhost,http://localhost:3000,http://localhost:5173,"
                "http://127.0.0.1,http://127.0.0.1:3000,http://127.0.0.1:5173",
            ).split(",")
            if item.strip()
        )
        return cls(
            jobs_data_dir=_path_env(
                "JOBS_DATA_DIR", Path(__file__).resolve().parents[1] / "data"
            )
            or Path("data").resolve(),
            talc_checkpoint_path=_path_env("TALC_CHECKPOINT_PATH"),
            talc_config_path=_path_env("TALC_CONFIG_PATH"),
            sulfide_checkpoint_path=_path_env("SULFIDE_CHECKPOINT_PATH"),
            sulfide_config_path=_path_env(
                "SULFIDE_CONFIG_PATH",
                vendor_root / "talk_sulfid" / "configs" / "classifier.yaml",
            ),
            sulfide_segmentation_config_path=_path_env(
                "SULFIDE_SEGMENTATION_CONFIG_PATH",
                vendor_root
                / "talk_sulfid"
                / "cv_analysis"
                / "sulfide_candidates.yaml",
            )
            or (
                vendor_root
                / "talk_sulfid"
                / "cv_analysis"
                / "sulfide_candidates.yaml"
            ),
            sulfide_sam_checkpoint_path=_path_env("SULFIDE_SAM_CHECKPOINT_PATH"),
            sulfide_sam_device=os.getenv(
                "SULFIDE_SAM_DEVICE", model_device
            ).strip().lower(),
            talc_source_path=_path_env(
                "TALC_SOURCE_PATH", vendor_root / "talk_combined"
            )
            or vendor_root / "talk_combined",
            sulfide_source_path=_path_env(
                "SULFIDE_SOURCE_PATH", vendor_root / "talk_sulfid"
            )
            or vendor_root / "talk_sulfid",
            demo_mode=_as_bool(os.getenv("DEMO_MODE"), False),
            model_device=model_device,
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024))),
            allowed_origins=origins,
        )

    def model_status(self) -> dict[str, dict[str, object]]:
        def status(path: Path | None, *, required: bool) -> dict[str, object]:
            if path is None:
                return {
                    "status": "unavailable" if required else "optional_unavailable",
                    "reason": "checkpoint_not_configured",
                    "required": required,
                }
            if not path.is_file():
                return {
                    "status": "unavailable" if required else "optional_unavailable",
                    "reason": "checkpoint_not_found",
                    "required": required,
                }
            return {"status": "configured", "reason": None, "required": required}

        return {
            "talc": status(self.talc_checkpoint_path, required=True),
            "sulfide": status(self.sulfide_checkpoint_path, required=True),
            "sulfide_sam": status(
                self.sulfide_sam_checkpoint_path, required=False
            ),
        }
