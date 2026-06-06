"""Runtime settings for the lightweight AIMS4PT web app."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int = _int_from_env("MAX_UPLOAD_MB", 10)
    session_ttl_seconds: int = _int_from_env("SESSION_TTL_SECONDS", 7200)
    max_concurrent_calculations: int = _int_from_env(
        "MAX_CONCURRENT_CALCULATIONS", 3
    )
    web_workers_recommended: int = _int_from_env("WEB_WORKERS_RECOMMENDED", 1)
    app_env: str = os.getenv("APP_ENV", "local")
    enable_r_models: bool = _bool_from_env("AIMS4PT_WEB_ENABLE_R_MODELS", True)
    enable_tensorflow_models: bool = _bool_from_env(
        "AIMS4PT_WEB_ENABLE_TENSORFLOW_MODELS", True
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
