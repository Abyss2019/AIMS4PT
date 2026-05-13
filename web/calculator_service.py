"""Service layer for the AIMS4PT Streamlit calculator."""

from __future__ import annotations

import importlib
import pickle as pkl
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aims4pt.model_tools.CpxTBSelect import (  # noqa: E402
    VALID_MELT_TAS_FIELDS,
    workflow_thermobarometry,
)
from aims4pt.model_tools.model_registry import (  # noqa: E402
    ALL_MODELS_MODULES,
    get_models_initial_pools,
)
from aims4pt.reporting.excel import report_excel  # noqa: E402


MAX_INPUT_ROWS = 500


@dataclass
class InputBundle:
    """Parsed calculator input tables and detection metadata."""

    original_data: pd.DataFrame
    x_cpx: pd.DataFrame
    x_liq: pd.DataFrame
    has_liquid_data: bool
    water_was_filled: bool
    liquid_detection_message: str


@dataclass
class ModelContext:
    """Cached model imports and initialized model pools for one input mode."""

    pressure_pool: list
    temperature_pool: list
    import_failures: dict[str, str]
    has_liquid_data: bool


@dataclass
class ReportResult:
    """Summary of one completed or skipped report task."""

    label: str
    report_path: Path | None
    cache_path: Path | None
    sample_count: int
    model_count: int
    selected_model_counts: dict[str, int] = field(default_factory=dict)
    skipped_reason: str | None = None
    error_message: str | None = None

    @property
    def completed(self) -> bool:
        return (
            self.report_path is not None
            and self.skipped_reason is None
            and self.error_message is None
        )

    @property
    def status(self) -> str:
        if self.completed:
            return "Completed"
        if self.error_message:
            return "Failed"
        return "Skipped"


@dataclass
class CalculatorRunResult:
    """Complete output bundle for one web calculator run."""

    project_name: str
    output_dir: Path
    timestamp: str
    input_bundle: InputBundle
    reports: list[ReportResult]
    import_failures: dict[str, str]


def sanitize_project_name(project_name: str) -> str:
    """Return a filesystem-safe project name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", project_name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "aims4pt_project"


def load_input_data(
    input_file: str | Path | BinaryIO,
    max_rows: int | None = MAX_INPUT_ROWS,
) -> InputBundle:
    """Load the Excel input and split clinopyroxene/liquid composition columns."""
    original_data = pd.read_excel(input_file, skiprows=1)
    if max_rows is not None and len(original_data) > max_rows:
        raise ValueError(
            f"Input contains {len(original_data)} rows, but the web calculator "
            f"currently allows at most {max_rows} rows."
        )

    x_cpx = original_data.loc[:, original_data.columns.str.endswith("_cpx")].copy()
    x_liq = original_data.loc[:, original_data.columns.str.endswith("_liq")].copy()

    water_was_filled = False
    if "H2O_liq" not in x_liq.columns:
        x_liq["H2O_liq"] = 0.0
        water_was_filled = True
    elif x_liq["H2O_liq"].isnull().all():
        x_liq["H2O_liq"] = 0.0
        water_was_filled = True

    if "SiO2_liq" in x_liq.columns:
        has_liquid_data = not x_liq["SiO2_liq"].isnull().all()
    else:
        has_liquid_data = False

    if has_liquid_data:
        liquid_detection_message = (
            "Melt composition data detected. Both clinopyroxene-only and "
            "clinopyroxene-liquid workflows can be calculated."
        )
    else:
        liquid_detection_message = (
            "No melt composition data detected. Only clinopyroxene-only workflows "
            "will be calculated."
        )

    return InputBundle(
        original_data=original_data,
        x_cpx=x_cpx,
        x_liq=x_liq,
        has_liquid_data=has_liquid_data,
        water_was_filled=water_was_filled,
        liquid_detection_message=liquid_detection_message,
    )


def import_model_modules() -> dict[str, str]:
    """Import all registered model modules and return failures without stopping."""
    failures: dict[str, str] = {}
    for module in ALL_MODELS_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - environment/model dependent
            failures[module] = str(exc)
    return failures


def build_model_pools(has_liquid_data: bool):
    """Build pressure and temperature model pools following the notebook behavior."""
    method_type = "both" if has_liquid_data else "cpx_only"
    pressure_pool = get_models_initial_pools("P", method_type, False)
    temperature_pool = get_models_initial_pools("T", method_type, False)
    return pressure_pool, temperature_pool


def prepare_model_context(has_liquid_data: bool) -> ModelContext:
    """Import model modules and initialize model pools once for a web session."""
    import_failures = import_model_modules()
    pressure_pool, temperature_pool = build_model_pools(has_liquid_data)
    return ModelContext(
        pressure_pool=pressure_pool,
        temperature_pool=temperature_pool,
        import_failures=import_failures,
        has_liquid_data=has_liquid_data,
    )


def run_calculator(
    project_name: str,
    calculate_cpx_liq: bool,
    tas_fields: list[str],
    input_file: str | Path | BinaryIO | None = None,
    input_bundle: InputBundle | None = None,
    model_context: ModelContext | None = None,
    output_root: str | Path | None = None,
    progress_callback=None,
) -> CalculatorRunResult:
    """Run all applicable AIMS4PT calculator branches and export Excel reports."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_project_name = sanitize_project_name(project_name)
    output_base = Path(output_root) if output_root is not None else REPO_ROOT / "report_output"
    output_dir = output_base / safe_project_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Loading input data")
    if input_bundle is None:
        if input_file is None:
            raise ValueError("Either input_file or input_bundle must be provided.")
        input_bundle = load_input_data(input_file)

    if not 1 <= len(tas_fields) <= 3:
        raise ValueError("Choose 1 to 3 melt TAS fields.")

    invalid_tas = [field for field in tas_fields if field not in VALID_MELT_TAS_FIELDS]
    if invalid_tas:
        raise ValueError(f"Unknown melt TAS fields: {invalid_tas}")

    if progress_callback:
        progress_callback("Initializing thermobarometer model pools")
    if model_context is None:
        model_context = prepare_model_context(input_bundle.has_liquid_data)
    elif model_context.has_liquid_data != input_bundle.has_liquid_data:
        raise ValueError("Cached model context does not match liquid-data detection.")

    import_failures = model_context.import_failures
    pressure_pool = model_context.pressure_pool
    temperature_pool = model_context.temperature_pool

    reports: list[ReportResult] = []

    tasks = [
        ("Clinopyroxene-only pressure", "cpx_only", "P", [m for m in pressure_pool if m.cpx_only]),
        ("Clinopyroxene-only temperature", "cpx_only", "T", [m for m in temperature_pool if m.cpx_only]),
    ]

    if input_bundle.has_liquid_data and calculate_cpx_liq:
        tasks.extend(
            [
                ("Clinopyroxene-liquid pressure", "cpx_liq", "P", [m for m in pressure_pool if not m.cpx_only]),
                ("Clinopyroxene-liquid temperature", "cpx_liq", "T", [m for m in temperature_pool if not m.cpx_only]),
            ]
        )
    else:
        reason = (
            "No liquid composition data were detected."
            if not input_bundle.has_liquid_data
            else "Clinopyroxene-liquid calculation is disabled."
        )
        reports.extend(
            [
                ReportResult("Clinopyroxene-liquid pressure", None, None, len(input_bundle.original_data), 0, skipped_reason=reason),
                ReportResult("Clinopyroxene-liquid temperature", None, None, len(input_bundle.original_data), 0, skipped_reason=reason),
            ]
        )

    for label, method_name, target_name, model_pool in tasks:
        if progress_callback:
            progress_callback(f"Running {label}")
        try:
            report = _run_report_task(
                label=label,
                method_name=method_name,
                target_name=target_name,
                model_pool=model_pool,
                input_bundle=input_bundle,
                tas_fields=tas_fields,
                output_dir=output_dir,
                timestamp=timestamp,
            )
        except Exception as exc:  # pragma: no cover - depends on local model/runtime setup
            report = ReportResult(
                label=label,
                report_path=None,
                cache_path=None,
                sample_count=len(input_bundle.original_data),
                model_count=len(model_pool),
                error_message=str(exc),
            )
        reports.append(
            report
        )

    return CalculatorRunResult(
        project_name=safe_project_name,
        output_dir=output_dir,
        timestamp=timestamp,
        input_bundle=input_bundle,
        reports=reports,
        import_failures=import_failures,
    )


def _run_report_task(
    label: str,
    method_name: str,
    target_name: str,
    model_pool: list,
    input_bundle: InputBundle,
    tas_fields: list[str],
    output_dir: Path,
    timestamp: str,
) -> ReportResult:
    """Run one notebook-equivalent workflow branch and export its report."""
    if not model_pool:
        return ReportResult(
            label=label,
            report_path=None,
            cache_path=None,
            sample_count=len(input_bundle.original_data),
            model_count=0,
            skipped_reason="No models were available for this workflow.",
        )

    cache_path = output_dir / f"computed_cache_{method_name}_{target_name}.pkl"

    if cache_path.exists():
        with cache_path.open("rb") as file_obj:
            workflow = pkl.load(file_obj)
    else:
        workflow = workflow_thermobarometry(model_pool)
        if method_name == "cpx_only" and not input_bundle.has_liquid_data:
            workflow.predict(
                input_bundle.x_cpx,
                input_bundle.x_liq,
                input_melt_TAS=tas_fields,
                melt_TAS_source="input_melt_TAS",
            )
        else:
            workflow.predict(input_bundle.x_cpx, input_bundle.x_liq)

        with cache_path.open("wb") as file_obj:
            pkl.dump(workflow, file_obj)

    report_path = output_dir / f"{timestamp}_report_{method_name}_{target_name}.xlsx"
    report_excel(
        workflow_obj=workflow,
        model_list=model_pool,
        original_data=input_bundle.original_data,
        out_path=str(report_path),
    )

    selected_counts = workflow.get_best_model_series().value_counts(dropna=False).to_dict()
    selected_counts = {str(key): int(value) for key, value in selected_counts.items()}

    return ReportResult(
        label=label,
        report_path=report_path,
        cache_path=cache_path,
        sample_count=len(input_bundle.original_data),
        model_count=len(model_pool),
        selected_model_counts=selected_counts,
    )
