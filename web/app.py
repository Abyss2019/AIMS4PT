"""Streamlit interface for the AIMS4PT calculator."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from calculator_service import (
    MAX_INPUT_ROWS,
    REPO_ROOT,
    VALID_MELT_TAS_FIELDS,
    CalculatorRunResult,
    InputBundle,
    load_input_data,
    prepare_model_context,
    run_calculator,
    sanitize_project_name,
)


st.set_page_config(
    page_title="AIMS4PT Calculator",
    page_icon="PT",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_style() -> None:
    """Add small visual refinements without changing Streamlit behavior."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .app-title {
            font-size: 2rem;
            font-weight: 760;
            letter-spacing: 0;
            margin-bottom: 0.2rem;
        }
        .app-subtitle {
            color: #4b5563;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _read_default_input() -> bytes | None:
    """Return bundled input.xlsx bytes if it exists."""
    default_path = REPO_ROOT / "input.xlsx"
    if default_path.exists():
        return default_path.read_bytes()
    return None


@st.cache_data(show_spinner="Parsing Excel input")
def _cached_load_input(input_bytes: bytes) -> InputBundle:
    """Parse uploaded Excel bytes and cache the resulting DataFrames."""
    return load_input_data(BytesIO(input_bytes), max_rows=MAX_INPUT_ROWS)


@st.cache_resource(show_spinner="Loading AIMS4PT model pools")
def _cached_model_context(has_liquid_data: bool):
    """Cache model module imports and initialized model pools across reruns."""
    return prepare_model_context(has_liquid_data)


def _show_input_preview(bundle: InputBundle) -> None:
    """Render parsed input tables and detection metadata."""
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", len(bundle.original_data))
    metric_cols[1].metric("Cpx columns", len(bundle.x_cpx.columns))
    metric_cols[2].metric("Liquid columns", len(bundle.x_liq.columns))
    metric_cols[3].metric("Liquid data", "Yes" if bundle.has_liquid_data else "No")

    if bundle.water_was_filled:
        st.info("H2O_liq is empty or missing, so it will be filled with 0 wt%.")
    st.caption(bundle.liquid_detection_message)

    preview_tabs = st.tabs(["Original input", "Clinopyroxene", "Liquid"])
    with preview_tabs[0]:
        st.dataframe(bundle.original_data.head(25), use_container_width=True)
    with preview_tabs[1]:
        st.dataframe(bundle.x_cpx.head(25), use_container_width=True)
    with preview_tabs[2]:
        st.dataframe(bundle.x_liq.head(25), use_container_width=True)


def _show_run_summary(result: CalculatorRunResult) -> None:
    """Render run metadata and per-report status."""
    st.subheader("Run summary")
    st.caption(f"Output folder: `{result.output_dir}`")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Project", result.project_name)
    metric_cols[1].metric("Rows", len(result.input_bundle.original_data))
    metric_cols[2].metric("Completed reports", sum(r.completed for r in result.reports))
    metric_cols[3].metric("Import warnings", len(result.import_failures))

    summary_rows = []
    for report in result.reports:
        summary_rows.append(
            {
                "Workflow": report.label,
                "Status": report.status,
                "Samples": report.sample_count,
                "Models": report.model_count,
                "Report": str(report.report_path) if report.report_path else "",
                "Reason": report.skipped_reason or report.error_message or "",
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if result.import_failures:
        with st.expander("Model import warnings"):
            for module, message in result.import_failures.items():
                st.warning(f"{module}: {message}")

    st.subheader("Selected model counts")
    completed_reports = [report for report in result.reports if report.completed]
    if not completed_reports:
        st.info("No completed reports are available.")
        return

    for report in completed_reports:
        with st.expander(report.label, expanded=False):
            count_df = pd.DataFrame(
                report.selected_model_counts.items(),
                columns=["Selected model", "Count"],
            )
            st.dataframe(count_df, use_container_width=True, hide_index=True)


def _show_report_downloads(result: CalculatorRunResult) -> None:
    """Render report download buttons."""
    st.subheader("Excel reports")
    completed_reports = [report for report in result.reports if report.completed and report.report_path]
    if not completed_reports:
        st.info("No downloadable reports are available.")
        return

    for report in completed_reports:
        path = Path(report.report_path)
        with st.container(border=True):
            st.markdown(f"**{report.label}**")
            st.caption(str(path))
            st.download_button(
                label="Download Excel report",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download-{path.name}",
            )


def main() -> None:
    """Run the Streamlit app."""
    _inject_style()

    st.markdown('<div class="app-title">AIMS4PT Calculator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Clinopyroxene thermobarometer selection and Excel report generation.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader("Upload input.xlsx", type=["xlsx"])
        use_default = st.checkbox("Use bundled input.xlsx", value=uploaded_file is None)
        st.caption(f"Maximum input size: {MAX_INPUT_ROWS} rows.")

        st.header("Settings")
        project_name = st.text_input("Project name", value="cpx_liq_example_project")
        safe_project_name = sanitize_project_name(project_name)
        if safe_project_name != project_name.strip():
            st.caption(f"Folder name will be sanitized as `{safe_project_name}`.")

        calculate_cpx_liq = st.toggle("Run cpx-liquid workflows", value=True)
        tas_fields = st.multiselect(
            "Melt TAS fields",
            options=list(VALID_MELT_TAS_FIELDS),
            default=[field for field in ["Basalt", "Andesite"] if field in VALID_MELT_TAS_FIELDS],
            max_selections=3,
        )
        run_button = st.button("Run calculator", type="primary", use_container_width=True)

    input_bytes = None
    input_name = None
    if uploaded_file is not None:
        input_bytes = uploaded_file.getvalue()
        input_name = uploaded_file.name
    elif use_default:
        input_bytes = _read_default_input()
        input_name = "input.xlsx"

    if input_bytes is None:
        st.info("Upload an Excel file or enable the bundled input.xlsx option.")
        return

    try:
        input_bundle = _cached_load_input(input_bytes)
    except Exception as exc:
        st.error(f"Failed to read {input_name or 'input.xlsx'}: {exc}")
        return

    tabs = st.tabs(["Input preview", "Run summary", "Reports"])
    with tabs[0]:
        _show_input_preview(input_bundle)

    if run_button:
        if not 1 <= len(tas_fields) <= 3:
            st.error("Choose 1 to 3 melt TAS fields before running.")
            return

        progress = st.progress(0, text="Preparing run")
        steps = []

        def on_progress(message: str) -> None:
            steps.append(message)
            progress.progress(min(len(steps) * 15, 90), text=message)

        with st.status("Running AIMS4PT calculator", expanded=True) as status:
            try:
                model_context = _cached_model_context(input_bundle.has_liquid_data)
                result = run_calculator(
                    project_name=project_name,
                    calculate_cpx_liq=calculate_cpx_liq,
                    tas_fields=tas_fields,
                    input_bundle=input_bundle,
                    model_context=model_context,
                    progress_callback=on_progress,
                )
            except Exception as exc:
                progress.empty()
                status.update(label="Run failed", state="error")
                st.exception(exc)
                return

            progress.progress(100, text="Finished")
            status.update(label="Run finished", state="complete")
            st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if result is None:
        with tabs[1]:
            st.info("Run the calculator to see workflow summaries.")
        with tabs[2]:
            st.info("Run the calculator to generate downloadable Excel reports.")
        return

    with tabs[1]:
        _show_run_summary(result)
    with tabs[2]:
        _show_report_downloads(result)


if __name__ == "__main__":
    main()
