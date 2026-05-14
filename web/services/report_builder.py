"""Web-facing wrappers around AIMS4PT Excel report payloads."""

from __future__ import annotations

from io import BytesIO

from aims4pt.reporting.excel import ReportPayload, write_report_excel


def write_payload_to_bytes(payload: ReportPayload) -> BytesIO:
    """Generate an Excel report in memory from an existing report payload."""
    output = BytesIO()
    write_report_excel(payload, output)
    output.seek(0)
    return output

