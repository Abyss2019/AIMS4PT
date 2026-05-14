"""Per-calculation report download routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from web.routes.shared import base_context, templates
from web.services.calculation_runner import get_calculation_spec
from web.services.memory import log_memory
from web.services.report_builder import write_payload_to_bytes
from web.services.session_store import get_session

logger = logging.getLogger("web.download")
router = APIRouter()


@router.get("/download/{session_id}/{calculation_key}")
async def download_report(request: Request, session_id: str, calculation_key: str):
    """Generate and stream one report from the current session cache."""
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

    result = session.results.get(calculation_key)
    if result is None:
        return templates.TemplateResponse(
            request,
            "error.html",
            base_context(
                request,
                message="This report is not available because the corresponding calculation has not been run.",
                back_url=f"/session/{session_id}",
            ),
            status_code=400,
        )

    log_memory("before-report-generation")
    output = write_payload_to_bytes(result.payload)
    log_memory("after-report-generation")

    spec = get_calculation_spec(calculation_key)
    filename = f"AIMS4PT_{spec.key}_report.xlsx"
    logger.info(
        "report_generated session_id=%s calculation=%s", session_id, calculation_key
    )
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
