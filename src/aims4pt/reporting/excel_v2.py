"""Version-two Excel reporting with a complete model-summary worksheet."""

from __future__ import annotations

import io
import math
import re
import struct
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
import pandas as pd

from aims4pt.reporting.excel import (
    MODEL_PLOT_COLORS,
    ReportPayload,
    _figure_to_png,
    _format_model_summary_values,
    _model_summary_row,
    _model_votes,
    _numeric_column,
    _phase_mg_number,
    _results_summary,
    _total_alkalis,
)
from aims4pt.reporting.model_abbreviations import (
    model_abbreviation,
    model_abbreviation_lookup,
)

if TYPE_CHECKING:
    from aims4pt.model_tools.CpxTBSelect import workflow_thermobarometry
    from aims4pt.model_tools.ModelManager import ModelManager


REFERENCE_PACKAGE = "aims4pt.reporting.data"
REFERENCE_WORKBOOK = "independent_data_final.xlsx"
REFERENCE_KDE_CACHE = "independent_composition_kde.npz"

MODEL_FILL_COLOR = "#7896B8"
MODEL_EDGE_COLOR = "#263746"
OVERALL_COLOR = "#D55E00"
REFERENCE_COLOR = "#9E9E9E"

BASE_EXCEL_IMAGE_SCALE = 0.68
PREDICTION_EXCEL_IMAGE_SCALE = BASE_EXCEL_IMAGE_SCALE
COMPOSITION_EXCEL_IMAGE_SCALE = BASE_EXCEL_IMAGE_SCALE
DEVIATION_EXCEL_IMAGE_SCALE = BASE_EXCEL_IMAGE_SCALE
PREDICTION_DPI_MULTIPLIER = 1.5
COMPOSITION_DPI_MULTIPLIER = 1.5
COMPOSITION_TARGET_ROW_SPAN = 21
DEFAULT_EXCEL_ROW_HEIGHT_PX = 20.0

PREDICTION_FIGURE_HEIGHT = 7.6
PREDICTION_FIGURE_MIN_WIDTH = 12.4
PREDICTION_FIGURE_MAX_WIDTH = 18.0
COMPOSITION_FIGURE_WIDTH = 10.25
COMPOSITION_FIGURE_ONE_ROW_HEIGHT = 5.1
COMPOSITION_FIGURE_TWO_ROW_HEIGHT = 9.1


COMPOSITION_PANEL_SPECS = (
    {
        "key": "cpx_cao_mg_number",
        "label": "(a)",
        "xlabel": r"Clinopyroxene CaO (wt%)",
        "ylabel": "Clinopyroxene Mg#",
        "xlim": (1.5126, 24.7974),
        "ylim": (0.2425254392, 1.0364196825),
        "kde_labels": {"68.3%": (15.7, 0.96, -16), "95.4%": (3.9, 0.65, 0)},
    },
    {
        "key": "cpx_na_al",
        "label": "(b)",
        "xlabel": r"Clinopyroxene Na$_2$O (wt%)",
        "ylabel": r"Clinopyroxene Al$_2$O$_3$ (wt%)",
        "xlim": (0.0, 1.1),
        "ylim": (-0.429045, 10.547795),
        "kde_labels": {"68.3%": (0.58, 6.8, -24), "95.4%": (0.78, 4.8, 28)},
    },
    {
        "key": "liquid_tas",
        "label": "(c)",
        "xlabel": r"Liquid SiO$_2$ (wt%)",
        "ylabel": r"Liquid Na$_2$O + K$_2$O (wt%)",
        "xlim": (39.823264, 79.791136),
        "ylim": (-0.6492, 11.4692),
        "kde_labels": {"68.3%": (73.5, 8.25, 12), "95.4%": (75.0, 10.2, 0)},
    },
    {
        "key": "liquid_sio2_mg_number",
        "label": "(d)",
        "xlabel": r"Liquid SiO$_2$ (wt%)",
        "ylabel": "Liquid Mg#",
        "xlim": (39.823264, 79.791136),
        "ylim": (0.0216188159, 1.0287459769),
        "kde_labels": {"68.3%": (60.5, 0.62, 0), "95.4%": (50.5, 0.96, -18)},
    },
)


def _build_results_sheet(
    workflow_obj: workflow_thermobarometry,
    original_data: pd.DataFrame,
) -> pd.DataFrame:
    results_sheet = pd.concat(
        [
            original_data.reset_index(drop=True),
            workflow_obj.prediction_df.reset_index(drop=True),
        ],
        axis=1,
    )
    results_sheet.columns = pd.MultiIndex.from_product(
        [["Original Data"], original_data.columns]
    ).append(
        pd.MultiIndex.from_product(
            [["Model Predictions"], workflow_obj.prediction_df.columns]
        )
    )
    return results_sheet


def _format_range_endpoint(value: object, target: str) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "not available"
    return f"{numeric:.1f}" if target == "P" else f"{numeric:.0f}"


def _stringify_rock_types(value: object) -> str:
    if value is None:
        return "not available"
    if isinstance(value, dict):
        return "; ".join(f"{key}: {item}" for key, item in value.items())
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return "; ".join(str(item) for item in value)
    return str(value)


def _full_model_summary_row(
    model: ModelManager,
    target: str,
    abbreviation_lookup: dict[object, str] | None = None,
) -> dict[str, object]:
    row = _model_summary_row(model, target)
    model_name = row["Model_name"]
    range_column = f"{target}_range ({'kbar' if target == 'P' else 'C'})"
    if getattr(model, "X_cpx_training", None) is not None:
        row[range_column] = (
            f"{_format_range_endpoint(getattr(model, 'y_min', np.nan), target)}–"
            f"{_format_range_endpoint(getattr(model, 'y_max', np.nan), target)}"
        )
    row["Supported_TAS_rock_types"] = _stringify_rock_types(
        row.get("Supported_TAS_rock_types")
    )
    return {
        "Model_name": model_name,
        "Model_abbreviation": (
            abbreviation_lookup.get(model_name, model_abbreviation(model_name, target))
            if abbreviation_lookup is not None
            else model_abbreviation(model_name, target)
        ),
        "Num_calibration_experiments": row["Num_calibration_experiments"],
        "Composition_range (wt%)": row["Composition_range (wt%)"],
        range_column: row[range_column],
        "Supported_TAS_rock_types": row["Supported_TAS_rock_types"],
    }


def _format_full_model_summary_values(
    df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    formatted = df.copy()
    precision = 1 if target == "P" else 0
    numeric_prefixes = (
        f"{target}_min",
        f"{target}_q1",
        f"{target}_median",
        f"{target}_q3",
        f"{target}_max",
    )
    for column in formatted.columns:
        if str(column).startswith(numeric_prefixes):
            formatted[column] = pd.to_numeric(
                formatted[column], errors="coerce"
            ).round(precision)
        elif column == "Mean_calculated_deviation":
            formatted[column] = pd.to_numeric(
                formatted[column], errors="coerce"
            ).round(precision)
        elif column == "OOD_ratio (%)":
            formatted[column] = pd.to_numeric(
                formatted[column], errors="coerce"
            ).round(1)
    return formatted


def _full_model_votes(
    compact_votes: pd.DataFrame,
    target: str,
    abbreviation_lookup: dict[object, str] | None = None,
) -> pd.DataFrame:
    votes = compact_votes.copy()
    abbreviations = []
    for model_name in votes["Model_name"]:
        if model_name in {"No selected model", "Total"}:
            abbreviations.append("—")
        else:
            abbreviations.append(
                abbreviation_lookup.get(
                    model_name, model_abbreviation(model_name, target)
                )
                if abbreviation_lookup is not None
                else model_abbreviation(model_name, target)
            )
    votes.insert(1, "Model_abbreviation", abbreviations)
    total = float(votes.loc[votes["Model_name"] == "Total", "Votes count"].iloc[0])
    votes["Proportion"] = np.where(
        total > 0,
        pd.to_numeric(votes["Votes count"], errors="coerce") / total,
        0.0,
    )
    votes.loc[votes["Model_name"] == "Total", "Proportion"] = 1.0 if total else 0.0
    return votes


def _report_model_abbreviation_lookups(
    model_names: list[object],
    target: str,
) -> tuple[dict[object, str], dict[object, str]]:
    """Build stable table and plot labels for all models in one report."""
    unique_names = list(dict.fromkeys(model_names))
    table_lookup = model_abbreviation_lookup(unique_names, target)
    plot_lookup = model_abbreviation_lookup(
        unique_names,
        target,
        disambiguate=True,
    )

    base_groups: dict[str, list[object]] = {}
    for name in unique_names:
        base_groups.setdefault(table_lookup[name], []).append(name)
    for abbreviation, group_names in base_groups.items():
        if len(group_names) < 2:
            continue
        numeric_labels = [
            bool(re.fullmatch(rf"{re.escape(abbreviation)}-\d+", plot_lookup[name]))
            for name in group_names
        ]
        if all(numeric_labels):
            for name in group_names:
                table_lookup[name] = plot_lookup[name]
    return table_lookup, plot_lookup


def _panel_xy(
    df: pd.DataFrame,
    key: str,
) -> tuple[pd.Series | None, pd.Series | None]:
    if key == "cpx_cao_mg_number":
        return (
            _numeric_column(df, ("CaO_cpx", "cpx_CaO", "cpx CaO", "CaO")),
            _report_phase_mg_number(df, "cpx"),
        )
    if key == "cpx_na_al":
        return (
            _numeric_column(df, ("Na2O_cpx", "cpx_Na2O", "cpx Na2O", "Na2O")),
            _numeric_column(df, ("Al2O3_cpx", "cpx_Al2O3", "cpx Al2O3", "Al2O3")),
        )
    if key == "liquid_tas":
        return (
            _numeric_column(df, ("SiO2_liq", "liq_SiO2", "liq SiO2")),
            _total_alkalis(df, "liq"),
        )
    if key == "liquid_sio2_mg_number":
        return (
            _numeric_column(df, ("SiO2_liq", "liq_SiO2", "liq SiO2")),
            _report_phase_mg_number(df, "liq"),
        )
    raise KeyError(f"Unknown composition panel key: {key}")


def _report_phase_mg_number(
    df: pd.DataFrame,
    phase: str,
) -> pd.Series | None:
    """Return calculated Mg# or a normalized pre-calculated Mg# column."""
    calculated = _phase_mg_number(df, phase)
    if calculated is not None and calculated.notna().sum() > 0:
        return calculated

    phase_names = (
        ("cpx", "clinopyroxene")
        if phase == "cpx"
        else ("liq", "liquid", "melt", "glass")
    )
    candidates: list[str] = []
    for phase_name in phase_names:
        candidates.extend(
            [
                f"Mg#_{phase_name}",
                f"Mg_number_{phase_name}",
                f"MgNumber_{phase_name}",
                f"Mgno_{phase_name}",
                f"Mg_num_{phase_name}",
                f"{phase_name}_Mg#",
                f"{phase_name}_Mg_number",
                f"{phase_name} Mg#",
            ]
        )
    mg_number = _numeric_column(df, tuple(candidates))
    if mg_number is None or mg_number.notna().sum() == 0:
        return None
    finite = mg_number[np.isfinite(mg_number)]
    if not finite.empty and float(finite.median()) > 1.5:
        mg_number = mg_number / 100.0
    return mg_number


def _finite_xy(
    x_values: pd.Series | None,
    y_values: pd.Series | None,
) -> tuple[np.ndarray, np.ndarray]:
    if x_values is None or y_values is None:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(x_values, errors="coerce"),
            "y": pd.to_numeric(y_values, errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    return frame["x"].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float)


def _kde_level_for_mass(density: np.ndarray, mass: float) -> float:
    values = np.asarray(density, dtype=float).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return np.nan
    values = np.sort(values)[::-1]
    cumulative = np.cumsum(values) / np.sum(values)
    index = min(int(np.searchsorted(cumulative, mass, side="left")), values.size - 1)
    return float(values[index])


def _compute_reference_kde(
    reference_df: pd.DataFrame,
    *,
    grid_size: int = 120,
) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for spec in COMPOSITION_PANEL_SPECS:
        x_values, y_values = _panel_xy(reference_df, spec["key"])
        x, y = _finite_xy(x_values, y_values)
        if x.size < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
            raise ValueError(
                f"Independent dataset cannot build KDE panel {spec['key']!r}."
            )
        xx, yy = np.meshgrid(
            np.linspace(*spec["xlim"], grid_size),
            np.linspace(*spec["ylim"], grid_size),
        )
        density = _evaluate_gaussian_kde_2d(x, y, xx, yy)
        cache[f"{spec['key']}__xx"] = xx
        cache[f"{spec['key']}__yy"] = yy
        cache[f"{spec['key']}__density"] = density
    return cache


def _evaluate_gaussian_kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Evaluate a Scott-bandwidth 2-D Gaussian KDE without BLAS calls."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples = x.size
    if n_samples < 3:
        raise ValueError("At least three finite points are required for a 2-D KDE.")

    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denominator = float(n_samples - 1)
    covariance_xx = float(np.sum(x_centered * x_centered) / denominator)
    covariance_xy = float(np.sum(x_centered * y_centered) / denominator)
    covariance_yy = float(np.sum(y_centered * y_centered) / denominator)
    bandwidth_factor = float(n_samples ** (-1.0 / 6.0))
    bandwidth_squared = bandwidth_factor * bandwidth_factor
    h_xx = covariance_xx * bandwidth_squared
    h_xy = covariance_xy * bandwidth_squared
    h_yy = covariance_yy * bandwidth_squared
    determinant = h_xx * h_yy - h_xy * h_xy
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError("Independent dataset has a singular KDE covariance matrix.")

    inverse_xx = h_yy / determinant
    inverse_xy = -h_xy / determinant
    inverse_yy = h_xx / determinant
    normalization = 1.0 / (
        2.0 * math.pi * float(n_samples) * math.sqrt(determinant)
    )

    grid_x = xx.ravel()
    grid_y = yy.ravel()
    density = np.empty(grid_x.size, dtype=float)
    for start in range(0, grid_x.size, chunk_size):
        stop = min(start + chunk_size, grid_x.size)
        delta_x = grid_x[start:stop, None] - x[None, :]
        delta_y = grid_y[start:stop, None] - y[None, :]
        exponent = -0.5 * (
            inverse_xx * delta_x * delta_x
            + 2.0 * inverse_xy * delta_x * delta_y
            + inverse_yy * delta_y * delta_y
        )
        density[start:stop] = normalization * np.sum(np.exp(exponent), axis=1)
    return density.reshape(xx.shape)


def write_default_reference_kde_cache(
    output_path: str | Path,
    *,
    grid_size: int = 120,
) -> None:
    """Generate the packaged KDE cache from the packaged reference workbook."""
    workbook_resource = files(REFERENCE_PACKAGE).joinpath(REFERENCE_WORKBOOK)
    if not workbook_resource.is_file():
        raise FileNotFoundError(
            f"Packaged independent dataset is missing: {REFERENCE_WORKBOOK}"
        )
    with as_file(workbook_resource) as workbook_path:
        reference_df = pd.read_excel(workbook_path)
    np.savez_compressed(
        Path(output_path),
        **_compute_reference_kde(reference_df, grid_size=grid_size),
    )


def _validate_reference_data(reference_df: pd.DataFrame) -> None:
    missing_panels = []
    for spec in COMPOSITION_PANEL_SPECS:
        x, y = _finite_xy(*_panel_xy(reference_df, spec["key"]))
        if x.size < 3 or y.size < 3:
            missing_panels.append(spec["key"])
    if missing_panels:
        raise ValueError(
            "Independent dataset is missing usable columns for composition panels: "
            + ", ".join(missing_panels)
        )


@lru_cache(maxsize=1)
def _load_default_reference_context() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    workbook_resource = files(REFERENCE_PACKAGE).joinpath(REFERENCE_WORKBOOK)
    cache_resource = files(REFERENCE_PACKAGE).joinpath(REFERENCE_KDE_CACHE)
    if not workbook_resource.is_file():
        raise FileNotFoundError(
            f"Packaged independent dataset is missing: {REFERENCE_WORKBOOK}"
        )
    if not cache_resource.is_file():
        raise FileNotFoundError(
            f"Packaged independent-data KDE cache is missing: {REFERENCE_KDE_CACHE}"
        )
    with as_file(workbook_resource) as workbook_path:
        reference_df = pd.read_excel(workbook_path)
    _validate_reference_data(reference_df)
    with as_file(cache_resource) as cache_path:
        with np.load(cache_path) as archive:
            kde_cache = {key: archive[key].copy() for key in archive.files}
    expected = {
        f"{spec['key']}__{suffix}"
        for spec in COMPOSITION_PANEL_SPECS
        for suffix in ("xx", "yy", "density")
    }
    missing = expected.difference(kde_cache)
    if missing:
        raise ValueError(
            "Packaged independent-data KDE cache is incomplete: "
            + ", ".join(sorted(missing))
        )
    return reference_df, kde_cache


def _resolve_reference_context(
    independent_data: pd.DataFrame | str | Path | None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if independent_data is None:
        reference_df, kde_cache = _load_default_reference_context()
        return reference_df.copy(), {key: value.copy() for key, value in kde_cache.items()}
    if isinstance(independent_data, pd.DataFrame):
        reference_df = independent_data.copy()
    else:
        workbook_path = Path(independent_data)
        if not workbook_path.is_file():
            raise FileNotFoundError(
                f"Independent dataset workbook does not exist: {workbook_path}"
            )
        reference_df = pd.read_excel(workbook_path)
    _validate_reference_data(reference_df)
    return reference_df, _compute_reference_kde(reference_df)


def _add_report_figure_header(
    fig,
    grid_spec,
    title: str,
    legend_handles: list[object],
    *,
    legend_columns: int,
    title_fontsize: float = 17.0,
    legend_fontsize: float = 12.5,
    title_y: float = 0.78,
    legend_anchor_y: float = 0.02,
):
    """Add a compact title-and-legend band shared by report figures."""
    header = fig.add_subplot(grid_spec)
    header.set_axis_off()
    header.set_gid("report-header")
    header.text(
        0.5,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold",
        transform=header.transAxes,
    )
    legend = header.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_anchor_y),
        ncol=max(1, legend_columns),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#C9C9C9",
        fontsize=legend_fontsize,
        handlelength=2.0,
        handletextpad=0.55,
        columnspacing=1.45,
        borderpad=0.30,
        labelspacing=0.25,
    )
    return header, legend


def _draw_reference_kde(
    axis,
    spec: dict[str, object],
    kde_cache: dict[str, np.ndarray],
) -> None:
    key = spec["key"]
    xx = kde_cache[f"{key}__xx"]
    yy = kde_cache[f"{key}__yy"]
    density = kde_cache[f"{key}__density"]
    level_items = []
    for label, mass in (("95.4%", 0.954), ("68.3%", 0.683)):
        level = _kde_level_for_mass(density, mass)
        if np.isfinite(level):
            level_items.append((level, label))
    level_items.sort(key=lambda item: item[0])
    levels = []
    for level, _ in level_items:
        if not levels or not np.isclose(level, levels[-1]):
            levels.append(level)
    if levels:
        axis.contour(
            xx,
            yy,
            density,
            levels=levels,
            colors="0.40",
            linewidths=0.95,
            linestyles="-",
            zorder=1,
        )
    for _, label in level_items:
        position = spec["kde_labels"].get(label)
        if position is None:
            continue
        x_text, y_text, rotation = position
        axis.text(
            x_text,
            y_text,
            label,
            fontsize=11.0,
            color="0.25",
            rotation=rotation,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
            zorder=7,
        )


def _build_model_selection_png(
    workflow_obj: workflow_thermobarometry,
    model_list: list[ModelManager],
    original_data: pd.DataFrame,
    target: str,
    reference_df: pd.DataFrame,
    kde_cache: dict[str, np.ndarray],
    image_dpi: int,
    *,
    label_lookup: dict[object, str] | None = None,
) -> bytes | None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected_models = workflow_obj.get_best_model_series()
    if selected_models is None or len(selected_models) == 0:
        return None
    n_rows = min(len(original_data), len(selected_models))
    if n_rows == 0:
        return None
    source = original_data.reset_index(drop=True).iloc[:n_rows].copy()
    selected_models = selected_models.reset_index(drop=True).iloc[:n_rows].astype(object)
    selected_models = selected_models.where(
        selected_models.notna(), "No selected model"
    )

    liquid_sio2 = _numeric_column(
        source,
        ("SiO2_liq", "liq_SiO2", "liq SiO2", "liquid_SiO2", "melt_SiO2"),
    )
    has_liquid = (
        liquid_sio2 is not None
        and pd.to_numeric(liquid_sio2, errors="coerce").notna().any()
    )
    panel_specs = (
        COMPOSITION_PANEL_SPECS
        if has_liquid
        else COMPOSITION_PANEL_SPECS[:2]
    )
    panels = [
        (spec, *_panel_xy(source, spec["key"]))
        for spec in panel_specs
    ]

    model_order = [
        getattr(model, "model_name", type(model).__name__)
        for model in model_list
        if getattr(model, "model_name", type(model).__name__) in set(selected_models)
    ]
    for model_name in pd.unique(selected_models):
        if model_name not in model_order:
            model_order.append(model_name)
    if label_lookup is None:
        label_lookup = model_abbreviation_lookup(
            model_order,
            target,
            disambiguate=True,
        )
    else:
        label_lookup = {
            model_name: label_lookup.get(
                model_name,
                model_abbreviation(model_name, target),
            )
            for model_name in model_order
        }
    color_lookup = {
        model_name: MODEL_PLOT_COLORS[index % len(MODEL_PLOT_COLORS)]
        for index, model_name in enumerate(model_order)
    }

    ncols = 2 if len(panels) > 1 else 1
    nrows_plot = math.ceil(len(panels) / ncols)
    figure_height = (
        COMPOSITION_FIGURE_ONE_ROW_HEIGHT
        if nrows_plot == 1
        else COMPOSITION_FIGURE_TWO_ROW_HEIGHT
    )
    outside_counts: list[str] = []
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 12,
        }
    ):
        fig = plt.figure(
            figsize=(COMPOSITION_FIGURE_WIDTH, figure_height),
            dpi=image_dpi,
            layout="constrained",
        )
        fig.patch.set_facecolor("white")
        fig.set_constrained_layout_pads(
            w_pad=0.04,
            h_pad=0.03,
            wspace=0.08,
            hspace=0.08,
        )
        outer_grid = fig.add_gridspec(
            2,
            1,
            height_ratios=(0.18, 0.82),
            hspace=0.12,
        )
        panel_grid = outer_grid[1].subgridspec(nrows_plot, ncols)
        axes = np.asarray(
            [
                fig.add_subplot(panel_grid[index // ncols, index % ncols])
                for index in range(nrows_plot * ncols)
            ],
            dtype=object,
        )
        for panel_index, (axis, panel) in enumerate(zip(axes, panels)):
            spec, input_x, input_y = panel
            ref_x, ref_y = _finite_xy(*_panel_xy(reference_df, spec["key"]))
            axis.set_facecolor("white")
            axis.set_axisbelow(True)
            axis.scatter(
                ref_x,
                ref_y,
                s=14,
                marker="o",
                color="0.62",
                alpha=0.85,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
            _draw_reference_kde(axis, spec, kde_cache)

            x_series = (
                pd.Series(np.nan, index=range(n_rows), dtype=float)
                if input_x is None
                else pd.to_numeric(input_x, errors="coerce")
                .reset_index(drop=True)
                .reindex(range(n_rows))
            )
            y_series = (
                pd.Series(np.nan, index=range(n_rows), dtype=float)
                if input_y is None
                else pd.to_numeric(input_y, errors="coerce")
                .reset_index(drop=True)
                .reindex(range(n_rows))
            )
            panel_df = pd.DataFrame(
                {
                    "x": x_series,
                    "y": y_series,
                    "model": selected_models,
                }
            ).replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
            if panel_df.empty:
                axis.text(
                    0.5,
                    0.5,
                    "Input values unavailable",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="0.35",
                    zorder=6,
                )
            for model_name in model_order:
                subset = panel_df.loc[panel_df["model"] == model_name]
                if subset.empty:
                    continue
                axis.scatter(
                    subset["x"],
                    subset["y"],
                    s=24,
                    alpha=0.92,
                    color=color_lookup[model_name],
                    edgecolors="0.15",
                    linewidths=0.45,
                    rasterized=True,
                    zorder=4,
                )

            xlim = spec["xlim"]
            ylim = spec["ylim"]
            outside = (
                (panel_df["x"] < xlim[0])
                | (panel_df["x"] > xlim[1])
                | (panel_df["y"] < ylim[0])
                | (panel_df["y"] > ylim[1])
            )
            if int(outside.sum()) > 0:
                outside_counts.append(f"{spec['label']} n={int(outside.sum())}")

            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.set_xlabel(spec["xlabel"], fontsize=14)
            axis.set_ylabel(spec["ylabel"], fontsize=14)
            axis.grid(True, color="0.92", linewidth=0.4, zorder=0)
            axis.tick_params(labelsize=12)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("0.25")
                spine.set_linewidth(0.8)
            axis.text(
                0.02,
                0.96,
                spec["label"],
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=16,
                fontweight="bold",
            )

        for axis in axes[len(panels):]:
            axis.remove()

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="0.62",
                markeredgecolor="none",
                markersize=7.5,
                label="Independent experimental dataset",
            )
        ]
        legend_handles.extend(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color_lookup[model_name],
                markeredgecolor="0.15",
                markeredgewidth=0.8,
                markersize=7.0,
                label=label_lookup[model_name],
            )
            for model_name in model_order
        )
        _add_report_figure_header(
            fig,
            outer_grid[0],
            "Selected models across compositional space relative to independent experiments",
            legend_handles,
            legend_columns=min(5, len(legend_handles)),
            title_fontsize=17,
            legend_fontsize=12,
            title_y=0.84,
            legend_anchor_y=0.10,
        )
        if outside_counts:
            fig.text(
                0.5,
                0.004,
                "Input points outside fixed comparison limits: " + ", ".join(outside_counts),
                ha="center",
                va="bottom",
                fontsize=10,
                color="0.30",
            )
    return _figure_to_png(fig, dpi=image_dpi)


def _finite_values(values: object) -> np.ndarray:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _draw_distribution(
    axis,
    values: object,
    position: float,
    *,
    width: float,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    linewidth: float,
    linestyle: str = "-",
    open_shape: bool = False,
    edge_alpha: float | None = None,
    zorder: float = 3.0,
    quartile_linewidth: float = 1.45,
    median_linewidth: float = 1.7,
    median_width_fraction: float = 0.20,
) -> bool:
    clean = _finite_values(values)
    if clean.size == 0:
        return False
    can_draw_violin = clean.size >= 2 and float(np.nanstd(clean)) > 0
    if can_draw_violin:
        try:
            violin = axis.violinplot(
                [clean],
                positions=[position],
                widths=width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method=0.25,
            )
            body = violin["bodies"][0]
            if open_shape:
                body.set_facecolor("none")
                body.set_edgecolor(edgecolor)
                body.set_alpha(1.0)
            elif edge_alpha is None:
                body.set_facecolor(facecolor)
                body.set_edgecolor(edgecolor)
                body.set_alpha(alpha)
            else:
                from matplotlib.colors import to_rgba

                body.set_facecolor(to_rgba(facecolor, alpha))
                body.set_edgecolor(to_rgba(edgecolor, edge_alpha))
                body.set_alpha(None)
            body.set_linewidth(linewidth)
            body.set_linestyle(linestyle)
            body.set_zorder(zorder)
        except (ValueError, np.linalg.LinAlgError):
            can_draw_violin = False

    q1, median, q3 = np.nanpercentile(clean, [25, 50, 75])
    if not can_draw_violin:
        axis.scatter(
            [position],
            [median],
            s=25,
            facecolors="none" if open_shape else facecolor,
            edgecolors=edgecolor,
            linewidths=max(linewidth, 1.1),
            zorder=zorder + 0.3,
        )
        axis.hlines(
            median,
            position - width * 0.30,
            position + width * 0.30,
            color=edgecolor,
            linewidth=max(linewidth, 1.1),
            linestyle=linestyle,
            zorder=zorder + 0.4,
        )
        return True

    axis.vlines(
        position,
        q1,
        q3,
        color="#111111",
        linewidth=quartile_linewidth,
        zorder=zorder + 0.4,
    )
    axis.hlines(
        median,
        position - width * median_width_fraction,
        position + width * median_width_fraction,
        color="#111111",
        linewidth=median_linewidth,
        zorder=zorder + 0.5,
    )
    return True


def _selected_predictions_from_best_models(
    prediction_df: pd.DataFrame,
    selected_models: pd.Series,
) -> np.ndarray:
    values = []
    n_rows = min(len(prediction_df), len(selected_models))
    for row_position in range(n_rows):
        model_name = selected_models.iloc[row_position]
        if pd.isna(model_name) or str(model_name) not in prediction_df.columns:
            continue
        value = pd.to_numeric(
            pd.Series(
                [prediction_df.iloc[row_position][str(model_name)]], dtype=object
            ),
            errors="coerce",
        ).iloc[0]
        if pd.notna(value) and np.isfinite(float(value)):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _favored_predictions_for_model(
    prediction_df: pd.DataFrame,
    selected_models: pd.Series,
    model_name: object,
) -> np.ndarray:
    n_rows = min(len(prediction_df), len(selected_models))
    if model_name not in prediction_df.columns or n_rows == 0:
        return np.asarray([], dtype=float)
    selected = selected_models.iloc[:n_rows]
    mask = selected.notna() & selected.astype(str).eq(str(model_name))
    positions = np.flatnonzero(mask.to_numpy())
    return _finite_values(prediction_df.iloc[positions][model_name])


def _prediction_model_order(
    prediction_df: pd.DataFrame,
    model_list: list[ModelManager],
) -> list[object]:
    order = [
        getattr(model, "model_name", type(model).__name__)
        for model in model_list
        if getattr(model, "model_name", type(model).__name__) in prediction_df.columns
    ]
    order.extend(column for column in prediction_df.columns if column not in order)
    return order


def _target_tick_formatter(axis, target: str) -> None:
    from matplotlib.ticker import FormatStrFormatter

    axis.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f" if target == "P" else "%.0f")
    )


def _build_prediction_distribution_png(
    workflow_obj: workflow_thermobarometry,
    model_list: list[ModelManager],
    target: str,
    selected_model_threshold: float,
    image_dpi: int,
    *,
    label_lookup: dict[object, str] | None = None,
) -> bytes | None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import AutoMinorLocator

    prediction_df = workflow_obj.prediction_df.apply(pd.to_numeric, errors="coerce")
    model_order = _prediction_model_order(prediction_df, model_list)
    if not model_order:
        return None
    selected_models = workflow_obj.get_best_model_series()
    if selected_models is None:
        selected_models = pd.Series(index=prediction_df.index, dtype=object)
    vote_proportions = selected_models.value_counts(dropna=False) / max(
        len(selected_models), 1
    )
    overall = _selected_predictions_from_best_models(prediction_df, selected_models)
    if label_lookup is None:
        label_lookup = model_abbreviation_lookup(
            model_order,
            target,
            disambiguate=True,
        )
    else:
        label_lookup = {
            model_name: label_lookup.get(
                model_name,
                model_abbreviation(model_name, target),
            )
            for model_name in model_order
        }
    figure_width = min(
        max(PREDICTION_FIGURE_MIN_WIDTH, 0.95 * len(model_order) + 7.6),
        PREDICTION_FIGURE_MAX_WIDTH,
    )

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    ):
        fig = plt.figure(
            figsize=(figure_width, PREDICTION_FIGURE_HEIGHT),
            dpi=image_dpi,
            layout="constrained",
        )
        fig.patch.set_facecolor("white")
        fig.set_constrained_layout_pads(
            w_pad=0.05,
            h_pad=0.03,
            wspace=0.05,
            hspace=0.04,
        )
        outer_grid = fig.add_gridspec(2, 1, height_ratios=(0.14, 0.86))
        axis = fig.add_subplot(outer_grid[1])
        axis.set_facecolor("white")
        axis.set_axisbelow(True)
        _draw_distribution(
            axis,
            overall,
            1.0,
            width=0.46,
            facecolor=OVERALL_COLOR,
            edgecolor="#7A3100",
            alpha=0.58,
            linewidth=0.9,
            zorder=2.5,
            quartile_linewidth=1.15,
            median_linewidth=1.8,
            median_width_fraction=0.30,
        )

        centers = np.arange(2.45, 2.45 + len(model_order))
        for center, model_name in zip(centers, model_order):
            proportion = float(vote_proportions.get(model_name, 0.0))
            is_selected = proportion > selected_model_threshold
            linestyle = "--" if is_selected else "-"
            all_input_linewidth = 2.0 if is_selected else 0.7
            favored_linewidth = 2.0 if is_selected else 1.3
            _draw_distribution(
                axis,
                prediction_df[model_name],
                center - 0.10,
                width=0.30,
                facecolor=MODEL_FILL_COLOR,
                edgecolor="#111111" if is_selected else MODEL_EDGE_COLOR,
                alpha=0.42,
                linewidth=all_input_linewidth,
                linestyle=linestyle,
                edge_alpha=1.0,
                zorder=3.0,
                quartile_linewidth=1.125,
                median_linewidth=1.5,
                median_width_fraction=0.34,
            )
            favored = _favored_predictions_for_model(
                prediction_df, selected_models, model_name
            )
            _draw_distribution(
                axis,
                favored,
                center + 0.16,
                width=0.16,
                facecolor="white",
                edgecolor="#111111" if is_selected else MODEL_EDGE_COLOR,
                alpha=1.0,
                linewidth=favored_linewidth,
                linestyle=linestyle,
                open_shape=True,
                zorder=3.5,
                quartile_linewidth=1.95,
                median_linewidth=2.6,
                median_width_fraction=0.34,
            )

        axis.set_xticks(np.concatenate([[1.0], centers]))
        axis.set_xticklabels(
            ["Overall", *[label_lookup[name] for name in model_order]],
            rotation=0,
            ha="center",
            fontsize=13,
        )
        axis.set_xlim(0.40, centers[-1] + 0.55)
        unit = "kbar" if target == "P" else "°C"
        axis.set_ylabel(
            f"{'Pressure' if target == 'P' else 'Temperature'} ({unit})",
            fontsize=15,
        )
        _target_tick_formatter(axis, target)
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        axis.grid(
            axis="y", which="major", color="#D9D9D9", linewidth=0.65
        )
        axis.grid(
            axis="y", which="minor", color="#EEEEEE", linewidth=0.40
        )
        axis.tick_params(axis="both", which="major", labelsize=13)
        axis.margins(y=0.04)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#3A3A3A")
            spine.set_linewidth(0.8)
        title = (
            "Pressure distributions: overall, all-input, and favored subsets"
            if target == "P"
            else "Temperature distributions: overall, all-input, and favored subsets"
        )
        legend_handles = [
            Patch(
                facecolor=MODEL_FILL_COLOR,
                edgecolor=MODEL_EDGE_COLOR,
                linewidth=0.7,
                label="All input",
            ),
            Patch(
                facecolor="none",
                edgecolor=MODEL_EDGE_COLOR,
                linewidth=1.3,
                label="Favored subset",
            ),
            Patch(
                facecolor="white",
                edgecolor="#111111",
                linewidth=2.0,
                linestyle="--",
                label=f"Selected model (>{selected_model_threshold:.0%})",
            ),
        ]
        _add_report_figure_header(
            fig,
            outer_grid[0],
            title,
            legend_handles,
            legend_columns=3,
            title_fontsize=17,
            legend_fontsize=12.5,
        )
    return _figure_to_png(fig, dpi=image_dpi)


def _build_deviation_violin_png(
    workflow_obj: workflow_thermobarometry,
    model_list: list[ModelManager],
    target: str,
    image_dpi: int,
) -> bytes | None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    deviation_df = workflow_obj.calculated_deviation_df
    if deviation_df is None or deviation_df.empty:
        return None
    model_order = _prediction_model_order(deviation_df, model_list)
    plottable = [
        model_name
        for model_name in model_order
        if _finite_values(deviation_df[model_name]).size > 0
    ]
    if not plottable:
        return None
    labels = model_abbreviation_lookup(plottable, target, disambiguate=True)
    positions = np.arange(1, len(plottable) + 1, dtype=float)
    figure_width = min(max(5.8, 0.58 * len(plottable) + 2.8), 11.0)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    ):
        fig, axis = plt.subplots(figsize=(figure_width, 4.0), dpi=image_dpi)
        for index, (position, model_name) in enumerate(zip(positions, plottable)):
            _draw_distribution(
                axis,
                deviation_df[model_name],
                position,
                width=0.68,
                facecolor=MODEL_PLOT_COLORS[index % len(MODEL_PLOT_COLORS)],
                edgecolor="#222222",
                alpha=0.72,
                linewidth=0.75,
            )
        all_values = np.concatenate(
            [_finite_values(deviation_df[model_name]) for model_name in plottable]
        )
        if all_values.size and np.nanmin(all_values) >= 0:
            axis.set_ylim(bottom=0)
        unit = "kbar" if target == "P" else "°C"
        axis.set_ylabel(f"Predicted deviation ({unit})", fontsize=10)
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [labels[name] for name in plottable],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        _target_tick_formatter(axis, target)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title("Predicted deviation distributions by model", fontsize=12, pad=10)
        fig.tight_layout(pad=0.8)
    return _figure_to_png(fig, dpi=image_dpi)


def build_report_payload(
    workflow_obj: workflow_thermobarometry,
    model_list: list[ModelManager],
    original_data: pd.DataFrame,
    *,
    independent_data: pd.DataFrame | str | Path | None = None,
    selected_model_threshold: float = 0.05,
    image_dpi: int = 200,
) -> ReportPayload:
    """Build the v2 report payload while retaining compact web tables."""
    if not model_list:
        raise ValueError("Cannot build a report without models.")
    if workflow_obj.prediction_df is None or workflow_obj.prediction_df.empty:
        raise ValueError("Cannot build a report before predictions are available.")
    if not 0 <= selected_model_threshold <= 1:
        raise ValueError("selected_model_threshold must be between 0 and 1.")
    if image_dpi <= 0:
        raise ValueError("image_dpi must be positive.")

    target = str(
        getattr(model_list[0], "T_P", getattr(workflow_obj, "T_P", ""))
    ).upper()
    if target not in {"P", "T"}:
        raise ValueError("Report target must be 'P' or 'T'.")

    results_sheet = _build_results_sheet(workflow_obj, original_data)
    selected_models = workflow_obj.get_best_model_series()
    report_model_names = [
        getattr(model, "model_name", type(model).__name__)
        for model in model_list
    ]
    report_model_names.extend(
        column
        for column in workflow_obj.prediction_df.columns
        if column not in report_model_names
    )
    if selected_models is not None:
        report_model_names.extend(
            model_name
            for model_name in pd.unique(selected_models.dropna())
            if model_name not in report_model_names
        )
    table_abbreviation_lookup, plot_abbreviation_lookup = (
        _report_model_abbreviation_lookups(report_model_names, target)
    )
    full_metadata = pd.DataFrame(
        [
            _full_model_summary_row(
                model,
                target,
                table_abbreviation_lookup,
            )
            for model in model_list
        ]
    )
    results_summary = _results_summary(workflow_obj, target)
    full_summary = pd.concat(
        [
            full_metadata.set_index("Model_name"),
            results_summary.set_index("Model_name"),
        ],
        axis=1,
    ).reset_index()
    full_summary = _format_full_model_summary_values(full_summary, target)
    compact_summary = _format_model_summary_values(full_summary, target).drop(
        columns=["Model_abbreviation"], errors="ignore"
    )

    compact_votes = _model_votes(workflow_obj)
    excel_votes = _full_model_votes(
        compact_votes,
        target,
        table_abbreviation_lookup,
    )
    ranking_details = pd.concat(
        [
            workflow_obj.failure_reason_df,
            selected_models.rename("Selected_model"),
        ],
        axis=1,
    )

    composition_image_dpi = int(math.ceil(image_dpi * COMPOSITION_DPI_MULTIPLIER))
    prediction_image_dpi = int(math.ceil(image_dpi * PREDICTION_DPI_MULTIPLIER))
    deviation_image_dpi = image_dpi
    reference_df, kde_cache = _resolve_reference_context(independent_data)
    model_selection_png = _build_model_selection_png(
        workflow_obj,
        model_list,
        original_data,
        target,
        reference_df,
        kde_cache,
        composition_image_dpi,
        label_lookup=plot_abbreviation_lookup,
    )
    deviation_png = _build_deviation_violin_png(
        workflow_obj, model_list, target, deviation_image_dpi
    )
    prediction_png = _build_prediction_distribution_png(
        workflow_obj,
        model_list,
        target,
        selected_model_threshold,
        prediction_image_dpi,
        label_lookup=plot_abbreviation_lookup,
    )

    return ReportPayload(
        results_sheet=results_sheet,
        model_summary_df=compact_summary,
        model_votes_df=compact_votes,
        ranking_details_df=ranking_details,
        violin_plot_png=prediction_png,
        target=target,
        model_selection_plot_png=model_selection_png,
        deviation_violin_plot_png=deviation_png,
        excel_model_summary_df=full_summary,
        excel_model_votes_df=excel_votes,
        image_dpi=image_dpi,
        model_selection_image_dpi=composition_image_dpi,
        prediction_image_dpi=prediction_image_dpi,
        deviation_image_dpi=deviation_image_dpi,
    )


def _write_value(worksheet, row: int, column: int, value: object, cell_format) -> None:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        worksheet.write_blank(row, column, None, cell_format)
    elif isinstance(value, (np.integer, int)):
        worksheet.write_number(row, column, int(value), cell_format)
    elif isinstance(value, (np.floating, float)):
        worksheet.write_number(row, column, float(value), cell_format)
    else:
        worksheet.write(row, column, str(value), cell_format)


def _png_dimensions(png_bytes: bytes | None) -> tuple[int, int]:
    if not png_bytes or len(png_bytes) < 24 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0
    return struct.unpack(">II", png_bytes[16:24])


def _image_row_span(
    png_bytes: bytes | None,
    *,
    image_dpi: int,
    scale: float,
    row_height_px: float = 20.0,
) -> int:
    _, height = _png_dimensions(png_bytes)
    if height <= 0:
        return 0
    displayed_height_px = height * 96.0 / max(float(image_dpi), 1.0) * scale
    return int(math.ceil(displayed_height_px / row_height_px))


def _image_scale_for_row_span(
    png_bytes: bytes | None,
    *,
    image_dpi: int,
    target_row_span: int,
    max_scale: float,
    row_height_px: float = DEFAULT_EXCEL_ROW_HEIGHT_PX,
) -> float:
    """Return one aspect-preserving scale that fits an image within a row span."""
    _, height = _png_dimensions(png_bytes)
    if height <= 0 or image_dpi <= 0 or target_row_span <= 0:
        return max_scale
    natural_height_px = height * 96.0 / float(image_dpi)
    target_height_px = target_row_span * row_height_px
    return min(max_scale, target_height_px / natural_height_px)


def _summary_cell_format(
    workbook,
    column: str,
    target: str,
    formats: dict[str, object],
):
    if column in {"Composition_range (wt%)", "Supported_TAS_rock_types", "Model_name"}:
        return formats["wrapped"]
    if column == "Num_calibration_experiments":
        return formats["integer"]
    if column == "OOD_ratio (%)":
        return formats["one_decimal"]
    if column == "Mean_calculated_deviation" or str(column).startswith(
        (f"{target}_min", f"{target}_q1", f"{target}_median", f"{target}_q3", f"{target}_max")
    ):
        return formats["one_decimal" if target == "P" else "integer"]
    return formats["text"]


def _estimated_summary_row_height(row: pd.Series) -> float:
    composition = str(row.get("Composition_range (wt%)", ""))
    rock_types = str(row.get("Supported_TAS_rock_types", ""))
    model_name = str(row.get("Model_name", ""))
    lines = max(
        1,
        math.ceil(len(composition) / 31),
        math.ceil(len(rock_types) / 28),
        math.ceil(len(model_name) / 38),
    )
    return min(max(18.0, 14.5 * lines), 96.0)


def _format_model_summary_sheet(
    writer: pd.ExcelWriter,
    summary_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    vote_start_row: int,
    target: str,
) -> None:
    workbook = writer.book
    worksheet = writer.sheets["Model Summary"]
    header = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#1F4E78",
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
            "border": 0,
        }
    )
    formats = {
        "text": workbook.add_format({"align": "left", "valign": "top"}),
        "wrapped": workbook.add_format(
            {"align": "left", "valign": "top", "text_wrap": True}
        ),
        "integer": workbook.add_format({"num_format": "#,##0", "valign": "top"}),
        "one_decimal": workbook.add_format({"num_format": "0.0", "valign": "top"}),
        "percentage": workbook.add_format({"num_format": "0.0%", "valign": "top"}),
        "total_text": workbook.add_format({"bold": True, "valign": "top"}),
        "total_integer": workbook.add_format(
            {"bold": True, "num_format": "#,##0", "valign": "top"}
        ),
        "total_percentage": workbook.add_format(
            {"bold": True, "num_format": "0.0%", "valign": "top"}
        ),
    }

    worksheet.hide_gridlines(2)
    worksheet.set_zoom(85)
    worksheet.set_row(0, 32)
    for column_index, column in enumerate(summary_df.columns):
        worksheet.write(0, column_index, column, header)
        cell_format = _summary_cell_format(workbook, column, target, formats)
        for row_index, value in enumerate(summary_df[column], start=1):
            _write_value(worksheet, row_index, column_index, value, cell_format)
    for row_index, (_, row) in enumerate(summary_df.iterrows(), start=1):
        worksheet.set_row(row_index, _estimated_summary_row_height(row))

    widths = {
        "Model_name": 38,
        "Model_abbreviation": 15,
        "Num_calibration_experiments": 17,
        "Composition_range (wt%)": 26,
        "Supported_TAS_rock_types": 23,
        "Mean_calculated_deviation": 23,
        "OOD_ratio (%)": 14,
    }
    for column_index, column in enumerate(summary_df.columns):
        width = widths.get(column, 15 if "range" in str(column).lower() else 13)
        worksheet.set_column(column_index, column_index, width)

    worksheet.set_row(vote_start_row, 28)
    for column_index, column in enumerate(votes_df.columns):
        worksheet.write(vote_start_row, column_index, column, header)

    first_data_row = vote_start_row + 1
    total_data_row = vote_start_row + len(votes_df)
    first_excel_row = first_data_row + 1
    total_excel_row = total_data_row + 1
    non_total_last_excel_row = max(first_excel_row, total_excel_row - 1)
    for data_index, (_, row) in enumerate(votes_df.iterrows()):
        worksheet_row = first_data_row + data_index
        excel_row = worksheet_row + 1
        is_total = str(row["Model_name"]) == "Total"
        text_format = formats["total_text"] if is_total else formats["text"]
        integer_format = formats["total_integer"] if is_total else formats["integer"]
        percentage_format = (
            formats["total_percentage"] if is_total else formats["percentage"]
        )
        _write_value(worksheet, worksheet_row, 0, row["Model_name"], text_format)
        _write_value(
            worksheet, worksheet_row, 1, row["Model_abbreviation"], text_format
        )
        if is_total:
            votes_formula = f"=SUM(C{first_excel_row}:C{non_total_last_excel_row})"
            worksheet.write_formula(
                worksheet_row,
                2,
                votes_formula,
                integer_format,
                int(row["Votes count"]),
            )
            proportion_formula = (
                f"=SUM(D{first_excel_row}:D{non_total_last_excel_row})"
            )
        else:
            _write_value(
                worksheet, worksheet_row, 2, row["Votes count"], integer_format
            )
            proportion_formula = f"=IFERROR(C{excel_row}/$C${total_excel_row},0)"
        worksheet.write_formula(
            worksheet_row,
            3,
            proportion_formula,
            percentage_format,
            float(row["Proportion"]),
        )


def _format_results_sheet(
    writer: pd.ExcelWriter,
    results_df: pd.DataFrame,
) -> None:
    """Apply restrained report styling to the two-level Results sheet."""
    workbook = writer.book
    worksheet = writer.sheets["Results"]
    last_column = len(results_df.columns)
    last_row = len(results_df) + 3
    group_header = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#1F4E78",
            "align": "center",
            "valign": "vcenter",
        }
    )
    column_header = workbook.add_format(
        {
            "bold": True,
            "font_color": "#1F3349",
            "bg_color": "#DCE6F1",
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
            "bottom": 1,
            "bottom_color": "#AEBECD",
        }
    )
    stripe = workbook.add_format({"bg_color": "#F5F8FA"})

    worksheet.hide_gridlines(2)
    worksheet.set_zoom(90)
    worksheet.set_tab_color("#5B9BD5")
    worksheet.set_default_row(18)
    worksheet.set_row(0, 26)
    worksheet.set_row(1, 34)
    worksheet.set_row(2, 7)
    worksheet.set_column(0, 0, 7)
    for column_index, column in enumerate(results_df.columns, start=1):
        group_name = str(column[0]) if isinstance(column, tuple) else ""
        column_name = str(column[-1]) if isinstance(column, tuple) else str(column)
        if group_name == "Original Data":
            width = min(max(len(column_name) + 2, 11), 18)
        else:
            width = min(max(len(column_name) + 2, 14), 28)
        worksheet.set_column(column_index, column_index, width)

    worksheet.conditional_format(
        0,
        0,
        0,
        last_column,
        {"type": "formula", "criteria": "=TRUE", "format": group_header},
    )
    worksheet.conditional_format(
        1,
        0,
        1,
        last_column,
        {"type": "formula", "criteria": "=TRUE", "format": column_header},
    )
    if len(results_df) > 0:
        worksheet.conditional_format(
            3,
            0,
            last_row,
            last_column,
            {
                "type": "formula",
                "criteria": "=MOD(ROW(),2)=0",
                "format": stripe,
            },
        )
    worksheet.set_landscape()
    worksheet.fit_to_pages(1, 0)
    worksheet.set_margins(left=0.25, right=0.25, top=0.45, bottom=0.45)


def _format_ranking_details_sheet(
    writer: pd.ExcelWriter,
    ranking_df: pd.DataFrame,
) -> None:
    """Apply compact table styling to Ranking Details without changing values."""
    workbook = writer.book
    worksheet = writer.sheets["Ranking Details"]
    header = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#1F4E78",
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        }
    )
    wrapped = workbook.add_format(
        {"align": "left", "valign": "top", "text_wrap": True}
    )
    index_format = workbook.add_format(
        {"align": "right", "valign": "top", "num_format": "#,##0"}
    )
    stripe = workbook.add_format({"bg_color": "#F5F8FA"})

    worksheet.hide_gridlines(2)
    worksheet.set_zoom(90)
    worksheet.set_tab_color("#7F8C99")
    worksheet.set_default_row(20)
    worksheet.set_row(0, 32)
    worksheet.set_column(0, 0, 7, index_format)
    worksheet.write_blank(0, 0, None, header)
    for column_index, column in enumerate(ranking_df.columns, start=1):
        worksheet.write(0, column_index, str(column), header)
        if str(column) == "Selected_model":
            width = 40
        elif "reason" in str(column).lower():
            width = 34
        else:
            width = min(max(len(str(column)) + 3, 18), 34)
        worksheet.set_column(column_index, column_index, width, wrapped)

    if len(ranking_df) > 0:
        worksheet.conditional_format(
            1,
            0,
            len(ranking_df),
            len(ranking_df.columns),
            {
                "type": "formula",
                "criteria": "=MOD(ROW(),2)=0",
                "format": stripe,
            },
        )
        worksheet.autofilter(0, 1, len(ranking_df), len(ranking_df.columns))
    worksheet.set_landscape()
    worksheet.fit_to_pages(1, 0)
    worksheet.set_margins(left=0.25, right=0.25, top=0.45, bottom=0.45)


def write_report_excel(
    payload: ReportPayload,
    output: str | BinaryIO | io.BytesIO,
) -> None:
    """Write the styled v2 Excel report without changing report values."""
    if payload.results_sheet.empty or payload.model_summary_df.empty:
        raise ValueError("Cannot generate an empty report.")
    excel_summary_df = getattr(payload, "excel_model_summary_df", None)
    excel_votes_df = getattr(payload, "excel_model_votes_df", None)
    image_dpi = getattr(payload, "image_dpi", 200)
    composition_image_dpi = (
        getattr(payload, "model_selection_image_dpi", None) or image_dpi
    )
    prediction_image_dpi = getattr(payload, "prediction_image_dpi", None) or image_dpi
    summary_df = (
        excel_summary_df if excel_summary_df is not None else payload.model_summary_df
    )
    votes_df = excel_votes_df if excel_votes_df is not None else payload.model_votes_df
    required_vote_columns = {
        "Model_name",
        "Model_abbreviation",
        "Votes count",
        "Proportion",
    }
    if not required_vote_columns.issubset(votes_df.columns):
        votes_df = _full_model_votes(payload.model_votes_df, payload.target)

    with pd.ExcelWriter(
        output,
        engine="xlsxwriter",
        engine_kwargs={"options": {"in_memory": True}},
    ) as writer:
        payload.results_sheet.to_excel(writer, sheet_name="Results", index=True)
        _format_results_sheet(writer, payload.results_sheet)
        summary_df.to_excel(writer, sheet_name="Model Summary", index=False)
        vote_start_row = len(summary_df) + 3
        votes_df.to_excel(
            writer,
            sheet_name="Model Summary",
            index=False,
            startrow=vote_start_row,
        )
        _format_model_summary_sheet(
            writer,
            summary_df,
            votes_df,
            vote_start_row,
            payload.target,
        )

        worksheet = writer.sheets["Model Summary"]
        vote_block_rows = len(votes_df) + 2
        if payload.violin_plot_png:
            worksheet.insert_image(
                vote_start_row,
                4,
                "prediction_distributions.png",
                {
                    "image_data": io.BytesIO(payload.violin_plot_png),
                    "x_scale": PREDICTION_EXCEL_IMAGE_SCALE,
                    "y_scale": PREDICTION_EXCEL_IMAGE_SCALE,
                },
            )
        composition_start_row = vote_start_row + vote_block_rows
        composition_end_row = composition_start_row

        if payload.model_selection_plot_png:
            composition_scale = _image_scale_for_row_span(
                payload.model_selection_plot_png,
                image_dpi=composition_image_dpi,
                target_row_span=COMPOSITION_TARGET_ROW_SPAN,
                max_scale=COMPOSITION_EXCEL_IMAGE_SCALE,
            )
            worksheet.insert_image(
                composition_start_row,
                0,
                "selected_model_composition.png",
                {
                    "image_data": io.BytesIO(payload.model_selection_plot_png),
                    "x_scale": composition_scale,
                    "y_scale": composition_scale,
                },
            )
            composition_end_row = composition_start_row + _image_row_span(
                payload.model_selection_plot_png,
                image_dpi=composition_image_dpi,
                scale=composition_scale,
            )
        if payload.deviation_violin_plot_png:
            deviation_start_row = composition_end_row
            worksheet.insert_image(
                deviation_start_row,
                0,
                "predicted_deviation_violin.png",
                {
                    "image_data": io.BytesIO(payload.deviation_violin_plot_png),
                    "x_scale": DEVIATION_EXCEL_IMAGE_SCALE,
                    "y_scale": DEVIATION_EXCEL_IMAGE_SCALE,
                },
            )

        payload.ranking_details_df.to_excel(
            writer, sheet_name="Ranking Details", index=True
        )
        _format_ranking_details_sheet(writer, payload.ranking_details_df)


def report_excel(
    workflow_obj: workflow_thermobarometry,
    model_list: list[ModelManager],
    original_data: pd.DataFrame,
    out_path: str | BinaryIO | io.BytesIO,
    *,
    independent_data: pd.DataFrame | str | Path | None = None,
    selected_model_threshold: float = 0.05,
    image_dpi: int = 200,
) -> None:
    """Export a complete v2 AIMS4PT Excel report."""
    payload = build_report_payload(
        workflow_obj,
        model_list,
        original_data,
        independent_data=independent_data,
        selected_model_threshold=selected_model_threshold,
        image_dpi=image_dpi,
    )
    write_report_excel(payload, out_path)


__all__ = ["build_report_payload", "write_report_excel", "report_excel"]
