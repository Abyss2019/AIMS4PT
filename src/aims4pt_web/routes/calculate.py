"""Calculation routes for independent AIMS4PT actions."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from aims4pt_web.routes.shared import base_context, templates
from aims4pt_web.services.calculation_runner import run_calculation_async
from aims4pt_web.services.result_views import build_session_context
from aims4pt_web.services.session_store import get_session, store_result

logger = logging.getLogger("web.calculate")
router = APIRouter()


@router.post("/calculate/{session_id}/{calculation_key}", response_class=HTMLResponse)
async def calculate(
    request: Request,
    session_id: str,
    calculation_key: str,
    melt_tas_fields: str | None = Form(default=None),
):
    """Run one calculation and redisplay the session page."""
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

    try:
        result = await run_calculation_async(session, calculation_key, melt_tas_fields)
        session = store_result(session_id, result)
    except Exception as exc:
        logger.exception(
            "calculation_failed session_id=%s calculation=%s error_type=%s",
            session_id,
            calculation_key,
            type(exc).__name__,
        )
        context = build_session_context(session, error_message=str(exc))
        return templates.TemplateResponse(
            request,
            "validation_result.html",
            base_context(request, **context),
            status_code=400,
        )

    context = build_session_context(session)
    return templates.TemplateResponse(
        request, "validation_result.html", base_context(request, **context)
    )
