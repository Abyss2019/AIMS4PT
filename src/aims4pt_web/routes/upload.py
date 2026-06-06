"""Upload and validation routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import HTMLResponse

from aims4pt_web.config import settings
from aims4pt_web.routes.shared import base_context, templates
from aims4pt_web.services.calculation_runner import initialize_model_pools_for_session
from aims4pt_web.services.input_validation import validate_workbook_bytes
from aims4pt_web.services.memory import log_memory
from aims4pt_web.services.result_views import build_session_context
from aims4pt_web.services.session_store import create_session, get_session

logger = logging.getLogger("web.upload")
router = APIRouter()


def _render_upload_error(request: Request, message: str, status_code: int = 400):
    return templates.TemplateResponse(
        request,
        "index.html",
        base_context(request, error_message=message),
        status_code=status_code,
    )


@router.post("/upload", response_class=HTMLResponse)
async def upload_input(request: Request, file: UploadFile = File(...)):
    """Read an uploaded XLSX file into memory and validate it."""
    filename = file.filename or ""
    if not filename.lower().endswith(".xlsx"):
        await file.close()
        return _render_upload_error(
            request,
            "Only .xlsx input files are supported in this version.",
            status_code=400,
        )

    file_bytes = await file.read(settings.max_upload_bytes + 1)
    await file.close()
    if len(file_bytes) > settings.max_upload_bytes:
        return _render_upload_error(
            request,
            f"Uploaded file is larger than {settings.max_upload_mb} MB.",
            status_code=413,
        )

    validation = validate_workbook_bytes(file_bytes)
    session = create_session(validation)
    logger.info(
        "input_validated session_id=%s rows=%s errors=%s warnings=%s has_liquid=%s",
        session.session_id,
        validation.sample_count,
        len(validation.errors),
        len(validation.warnings),
        validation.has_liquid,
    )
    log_memory("after-input-validation")
    if validation.is_valid:
        initialize_model_pools_for_session(session)

    context = build_session_context(session)
    return templates.TemplateResponse(
        request, "validation_result.html", base_context(request, **context)
    )


@router.get("/session/{session_id}", response_class=HTMLResponse)
async def show_session(request: Request, session_id: str):
    """Render a cached validation/calculation session."""
    session = get_session(session_id)
    if session is None:
        return templates.TemplateResponse(
            request,
            "error.html",
            base_context(
                request,
                message="This session has expired. Please upload the file again.",
            ),
            status_code=404,
        )
    context = build_session_context(session)
    return templates.TemplateResponse(
        request, "validation_result.html", base_context(request, **context)
    )
