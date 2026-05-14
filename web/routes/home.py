"""Home and input-template routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pathlib import Path

from fastapi.responses import FileResponse, HTMLResponse

from web.routes.shared import base_context, templates

router = APIRouter()
TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "static" / "input_template.xlsx"


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the lightweight upload page."""
    return templates.TemplateResponse(request, "index.html", base_context(request))


@router.get("/template")
async def download_template():
    """Download the static XLSX input template."""
    return FileResponse(
        TEMPLATE_PATH,
        filename="AIMS4PT_cpx_input_template.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
