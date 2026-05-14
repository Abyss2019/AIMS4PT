"""Helpers for converting calculation results to lightweight template data."""

from __future__ import annotations

import pandas as pd

from web.services.session_store import CalculationResult, SessionData


def _format_cell(value: object, column: str) -> str:
    """Format result table values for compact web display."""
    if value == "" or pd.isna(value):
        return ""

    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric_value):
        if column.startswith("T_"):
            return f"{numeric_value:.0f}"
        if column.startswith("P_"):
            return f"{numeric_value:.1f}"
    return str(value)


def dataframe_table(df: pd.DataFrame, max_rows: int = 100) -> dict:
    """Convert a DataFrame to a simple table structure for Jinja templates."""
    display_df = df.head(max_rows).copy()
    display_df = display_df.where(pd.notna(display_df), "")
    columns = [str(column) for column in display_df.columns]
    return {
        "columns": columns,
        "rows": [
            [_format_cell(value, column) for column, value in zip(columns, row)]
            for row in display_df.astype(object).itertuples(index=False, name=None)
        ],
        "truncated": len(df) > max_rows,
        "shown": len(display_df),
        "total": len(df),
    }


def result_view(result: CalculationResult) -> dict:
    """Build the template view for one completed calculation."""
    payload = result.payload
    return {
        "key": result.key,
        "label": result.label,
        "summary": result.summary,
        "warnings": result.warnings,
        "model_summary": dataframe_table(payload.model_summary_df),
        "model_votes": dataframe_table(payload.model_votes_df),
    }


def build_session_context(session: SessionData, error_message: str | None = None) -> dict:
    """Build a complete template context from one cached session."""
    return {
        "session": session,
        "validation": session.validation,
        "results": [result_view(result) for result in session.results.values()],
        "error_message": error_message,
    }
