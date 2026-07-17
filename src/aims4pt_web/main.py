"""FastAPI app entry point for the lightweight AIMS4PT_cpx calculator."""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from aims4pt_web.routes import calculate, download, home
from aims4pt_web.config import settings
from aims4pt_web.routes import upload
from aims4pt_web.services.calculation_runner import ensure_model_registry_loaded
from aims4pt_web.services.memory import log_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

try:
    app_version = distribution_version("AIMS4PT")
except PackageNotFoundError:
    app_version = "0.0.0"

app = FastAPI(
    title="AIMS4PT_cpx Web Calculator",
    description="Lightweight local and production web interface for AIMS4PT_cpx.",
    version=app_version,
)

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(home.router)
app.include_router(upload.router)
app.include_router(calculate.router)
app.include_router(download.router)


@app.on_event("startup")
async def startup_event() -> None:
    """Warm the model registry and emit lightweight memory logs."""
    logging.getLogger("aims4pt_web.main").info(
        "app_starting env=%s recommended_workers=%s max_upload_mb=%s ttl_seconds=%s max_concurrent_calculations=%s enable_r_models=%s enable_tensorflow_models=%s",
        settings.app_env,
        settings.web_workers_recommended,
        settings.max_upload_mb,
        settings.session_ttl_seconds,
        settings.max_concurrent_calculations,
        settings.enable_r_models,
        settings.enable_tensorflow_models,
    )
    log_memory("startup-before-model-loading")
    ensure_model_registry_loaded()
    log_memory("startup-after-model-loading")
