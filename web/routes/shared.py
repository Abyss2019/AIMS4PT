"""Shared template helpers for FastAPI routes."""

from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

from web.config import settings
from web.services.calculation_runner import CALCULATION_SPECS

templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parents[1] / "templates")
)


def base_context(request, **kwargs) -> dict:
    """Build a base Jinja context used by every page."""
    context = {
        "request": request,
        "settings": settings,
        "calculation_specs": list(CALCULATION_SPECS.values()),
    }
    context.update(kwargs)
    return context

