"""In-memory TTL session cache for uploaded AIMS4PT inputs and results."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

import pandas as pd
from cachetools import TTLCache

from aims4pt_web.config import settings
from aims4pt_web.services.input_validation import ValidationResult


@dataclass
class CalculationResult:
    key: str
    label: str
    summary: str
    model_summary_df: pd.DataFrame
    model_votes_df: pd.DataFrame
    report_bytes: bytes
    warnings: list[str] = field(default_factory=list)
    completed_at: float = field(default_factory=time.time)


@dataclass
class SessionData:
    session_id: str
    parsed_df: pd.DataFrame | None
    cleaned_df: pd.DataFrame | None
    validation: ValidationResult
    has_liquid: bool
    created_at: float = field(default_factory=time.time)
    model_pools: dict[str, list] = field(default_factory=dict)
    environment_skipped_models: dict[str, list[str]] = field(default_factory=dict)
    model_pool_errors: dict[str, str] = field(default_factory=dict)
    results: dict[str, CalculationResult] = field(default_factory=dict)

    @property
    def warning_count(self) -> int:
        return len(self.validation.warnings)

    @property
    def reports_ready(self) -> int:
        return len(self.results)

    @property
    def completed_labels(self) -> str:
        if not self.results:
            return "none"
        return ", ".join(result.label for result in self.results.values())


_cache: TTLCache[str, SessionData] = TTLCache(
    maxsize=128, ttl=settings.session_ttl_seconds
)
_lock = threading.RLock()


def create_session(validation: ValidationResult) -> SessionData:
    """Create a new cache session from a validation result."""
    session_id = uuid.uuid4().hex
    session = SessionData(
        session_id=session_id,
        parsed_df=validation.parsed_df,
        cleaned_df=validation.cleaned_df,
        validation=validation,
        has_liquid=validation.has_liquid,
    )
    with _lock:
        _cache[session_id] = session
    return session


def get_session(session_id: str) -> SessionData | None:
    """Return a session if it still exists in the TTL cache."""
    with _lock:
        return _cache.get(session_id)


def store_result(session_id: str, result: CalculationResult) -> SessionData:
    """Store a completed calculation result in an existing session."""
    with _lock:
        session = _cache[session_id]
        session.results[result.key] = result
        _cache[session_id] = session
        return session


def expire_sessions() -> None:
    """Force cache expiry during request handling or tests."""
    with _lock:
        _cache.expire()
