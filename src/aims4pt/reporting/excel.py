"""Excel reporting helpers for AIMS4PT workflows."""

from __future__ import annotations

import io
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO

import numpy as np
import pandas as pd

from aims4pt.utils import normalize_column_names

if TYPE_CHECKING:
    from aims4pt.model_tools.CpxTBSelect import workflow_thermobarometry
    from aims4pt.model_tools.ModelManager import ModelManager


@dataclass
class ReportPayload:
    """Reusable report data shared by Excel export and lightweight web views."""

    results_sheet: pd.DataFrame
    model_summary_df: pd.DataFrame
    model_votes_df: pd.DataFrame
    ranking_details_df: pd.DataFrame
    violin_plot_png: bytes | None
    target: str
    model_selection_plot_png: bytes | None = None
    deviation_violin_plot_png: bytes | None = None


MODEL_PLOT_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

MGO_MOLAR_MASS = 40.3044
FEO_MOLAR_MASS = 71.844
REPORT_OXIDE_COLUMNS = (
    "SiO2",
    "TiO2",
    "Al2O3",
    "FeO",
    "Fe2O3",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "MnO",
    "Cr2O3",
    "NiO",
    "P2O5",
    "H2O",
    "CO2",
)


def _model_summary_row(model: ModelManager, target: str) -> dict:
    """Build one metadata row for a thermobarometry model."""
    unit = "kbar" if target == "P" else "C"
    x_cpx_training = getattr(model, "X_cpx_training", None)
    x_cpx_all = getattr(model, "X_cpx_all", None)
    rock_types = getattr(model, "rock_types", None)

    comp_range = None
    if x_cpx_training is not None and hasattr(model, "export_composition_range"):
        comp_range = model.export_composition_range("training", False, True, "text")

    if x_cpx_training is not None:
        value_range = (
            f"{getattr(model, 'y_min', 'not available')}-"
            f"{getattr(model, 'y_max', 'not available')}"
        )
    else:
        value_range = "not available"

    return {
        "Model_name": getattr(model, "model_name", type(model).__name__),
        "Num_calibration_experiments": len(x_cpx_all)
        if x_cpx_all is not None
        else "not available",
        "Composition_range (wt%)": comp_range if comp_range else "not available",
        f"{target}_range ({unit})": value_range,
        "Supported_TAS_rock_types": rock_types
        if rock_types is not None
        else "not available",
    }


def _results_summary(workflow_obj: workflow_thermobarometry, target: str) -> pd.DataFrame:
    """Summarize prediction, OOD, and deviation outputs by model."""
    pred = workflow_obj.prediction_df
    deviation_df = workflow_obj.calculated_deviation_df
    ood_mask_df = workflow_obj.ood_mask_df
    unit = "kbar" if target == "P" else "C"

    if deviation_df is None:
        deviation_df = pd.DataFrame(index=pred.index, columns=pred.columns, dtype=float)
    if ood_mask_df is None:
        ood_mask_df = pd.DataFrame(False, index=pred.index, columns=pred.columns)

    results_summary = pd.DataFrame(
        {
            f"{target}_min ({unit})": pred.min(axis=0),
            f"{target}_q1 ({unit})": pred.quantile(0.25, axis=0),
            f"{target}_median ({unit})": pred.median(axis=0),
            f"{target}_q3 ({unit})": pred.quantile(0.75, axis=0),
            f"{target}_max ({unit})": pred.max(axis=0),
            "Mean_calculated_deviation": deviation_df.mean(axis=0),
            "OOD_ratio (%)": ood_mask_df.mean(axis=0) * 100,
        }
    )
    return results_summary.reset_index().rename(columns={"index": "Model_name"})


def _model_votes(workflow_obj: workflow_thermobarometry) -> pd.DataFrame:
    """Count selected best-model labels and append a Total row."""
    votes = workflow_obj.get_best_model_series().value_counts(dropna=False)
    votes_df = votes.rename_axis("Model_name").reset_index(name="Votes count")
    votes_df["Model_name"] = votes_df["Model_name"].where(
        votes_df["Model_name"].notna(), "No selected model"
    )
    total = pd.DataFrame(
        [{"Model_name": "Total", "Votes count": int(votes_df["Votes count"].sum())}]
    )
    return pd.concat([votes_df, total], ignore_index=True)


def _short_model_name(model_name: object, target: str | None = None) -> str:
    """Return a compact plotting label for a thermobarometry model name."""
    if pd.isna(model_name):
        return "No model"

    text = str(model_name)
    lower = text.lower()
    if "no selected model" in lower:
        return "No model"
    if "neave" in lower and "2017" in lower:
        return "NP17"
    if "putirka" in lower and "2008" in lower:
        if target == "P":
            return "Pu08_31"
        if target == "T":
            return "Pu08_33"
        return "Pu08"
    if "petrelli" in lower:
        return "Pet20"
    if "jorgenson" in lower:
        return "Jor22"
    if "chicchi" in lower:
        return "Chi23"
    if "greda" in lower or "lopez" in lower:
        return "AgL24"
    if "higgins" in lower:
        return "Hig21"
    if "wang" in lower:
        return "Wan21"
    match = re.search(r"([A-Za-z]+).*?(\d{4})", text)
    if match:
        return f"{match.group(1)[:3]}{match.group(2)[-2:]}"
    return text[:18]


def _model_phase_suffix(model_name: object) -> str:
    """Return a compact phase suffix for duplicated short model labels."""
    lower = str(model_name).lower()
    if "cpx_only" in lower:
        return "cpx"
    if "cpx_liq" in lower or "liquid" in lower:
        return "liq"
    return ""


def _short_model_label_lookup(
    model_names: list[object],
    target: str | None = None,
) -> dict[object, str]:
    """Build stable short plotting labels and disambiguate duplicates."""
    base_labels = [_short_model_name(model_name, target) for model_name in model_names]
    label_counts = Counter(base_labels)
    labels = {}
    for model_name, base_label in zip(model_names, base_labels):
        label = base_label
        if label_counts[base_label] > 1:
            suffix = _model_phase_suffix(model_name)
            if suffix:
                label = f"{base_label}-{suffix}"
        labels[model_name] = label
    return labels


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> object | None:
    """Find a column using case-insensitive candidate names."""
    lower_to_column = {str(column).strip().lower(): column for column in df.columns}
    compact_to_column = {
        re.sub(r"[^a-z0-9]+", "", str(column).strip().lower()): column
        for column in df.columns
    }
    for candidate in candidates:
        lower_candidate = candidate.strip().lower()
        if lower_candidate in lower_to_column:
            return lower_to_column[lower_candidate]
        compact_candidate = re.sub(r"[^a-z0-9]+", "", lower_candidate)
        if compact_candidate in compact_to_column:
            return compact_to_column[compact_candidate]
    return None


def _compact_name(value: object) -> str:
    """Return a lowercase alphanumeric key for loose column matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _candidate_phase(candidates: tuple[str, ...]) -> str | None:
    """Infer phase from candidate names."""
    compact_candidates = [_compact_name(candidate) for candidate in candidates]
    if any("cpx" in candidate for candidate in compact_candidates):
        return "cpx"
    if any(
        phase_token in candidate
        for candidate in compact_candidates
        for phase_token in ("liq", "liquid", "melt", "glass")
    ):
        return "liq"
    return None


def _candidate_oxide(candidates: tuple[str, ...]) -> str | None:
    """Infer the standard oxide name requested by candidate names."""
    compact_candidates = [_compact_name(candidate) for candidate in candidates]
    for oxide in REPORT_OXIDE_COLUMNS:
        oxide_key = _compact_name(oxide)
        if any(oxide_key in candidate for candidate in compact_candidates):
            return oxide
    if any(
        "feot" in candidate or "feotot" in candidate
        for candidate in compact_candidates
    ):
        return "FeO"
    return None


def _phase_source_columns(df: pd.DataFrame, phase: str) -> list[object]:
    """Select source columns for one phase before oxide normalization."""
    explicit_phase_columns: list[object] = []
    unphased_columns: list[object] = []

    other_phase_tokens = (
        ("liq", "liquid", "melt", "glass") if phase == "cpx" else ("cpx",)
    )
    phase_tokens = ("cpx",) if phase == "cpx" else ("liq", "liquid", "melt", "glass")

    for column in df.columns:
        compact = _compact_name(column)
        if any(token in compact for token in phase_tokens):
            explicit_phase_columns.append(column)
        elif not any(token in compact for token in other_phase_tokens):
            unphased_columns.append(column)

    if explicit_phase_columns:
        return explicit_phase_columns
    if phase == "cpx":
        return unphased_columns
    return []


def _normalized_phase_column(
    df: pd.DataFrame,
    phase: str,
    oxide: str,
) -> pd.Series | None:
    """Return an oxide column after phase-aware column normalization."""
    source_columns = _phase_source_columns(df, phase)
    if not source_columns:
        return None

    try:
        normalized_df = normalize_column_names(
            df[source_columns],
            standard_names_list=list(REPORT_OXIDE_COLUMNS),
            missing_fill=None,
            drop_missing=False,
            report_info=False,
        )
    except Exception:
        return None

    if oxide not in normalized_df.columns:
        return None
    series = pd.to_numeric(normalized_df[oxide], errors="coerce")
    if series.notna().sum() == 0:
        return None
    return series


def _numeric_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series | None:
    """Return one numeric source column, or None if no candidate exists."""
    column = _find_column(df, candidates)
    if column is not None:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.notna().sum() > 0:
            return series

    phase = _candidate_phase(candidates)
    oxide = _candidate_oxide(candidates)
    if phase is None or oxide is None:
        return None
    return _normalized_phase_column(df, phase, oxide)



def _phase_mg_number(df: pd.DataFrame, phase: str) -> pd.Series | None:
    """Calculate molar Mg# for one phase as Mg / (Mg + Fe)."""
    mg = _numeric_column(
        df,
        (
            f"MgO_{phase}",
            f"MgOt_{phase}",
            f"{phase}_MgO",
            f"{phase} MgO",
        ),
    )
    fe = _numeric_column(
        df,
        (
            f"FeO_{phase}",
            f"FeOt_{phase}",
            f"FeOtot_{phase}",
            f"{phase}_FeO",
            f"{phase} FeO",
        ),
    )
    if mg is None or fe is None:
        return None

    mg_moles = mg / MGO_MOLAR_MASS
    fe_moles = fe / FEO_MOLAR_MASS
    denominator = mg_moles + fe_moles
    return mg_moles / denominator.replace(0, np.nan)


def _total_alkalis(df: pd.DataFrame, phase: str) -> pd.Series | None:
    """Calculate Na2O + K2O for one phase."""
    na2o = _numeric_column(df, (f"Na2O_{phase}", f"{phase}_Na2O", f"{phase} Na2O"))
    k2o = _numeric_column(df, (f"K2O_{phase}", f"{phase}_K2O", f"{phase} K2O"))
    if na2o is None or k2o is None:
        return None
    return na2o + k2o


def _figure_to_png(fig, dpi: int = 200) -> bytes:
    """Save a Matplotlib figure to PNG bytes and close it."""
    imgdata = io.BytesIO()
    fig.savefig(imgdata, format="png", bbox_inches="tight", dpi=dpi)
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return imgdata.getvalue()


def _build_model_selection_png(
    workflow_obj: workflow_thermobarometry,
    model_list: list[ModelManager],
    original_data: pd.DataFrame,
    target: str,
) -> bytes | None:
    """Render selected-model composition panels to PNG bytes for Excel insertion."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected_models = workflow_obj.get_best_model_series()
    if selected_models is None or len(selected_models) == 0:
        return None

    plot_source = original_data.reset_index(drop=True).copy()
    selected_models = selected_models.reset_index(drop=True)
    n_rows = min(len(plot_source), len(selected_models))
    if n_rows == 0:
        return None

    plot_source = plot_source.iloc[:n_rows].copy()
    selected_models = selected_models.iloc[:n_rows].astype(object)
    selected_models = selected_models.where(
        selected_models.notna(), "No selected model"
    )

    cpx_cao = _numeric_column(plot_source, ("CaO_cpx", "cpx_CaO", "cpx CaO"))
    cpx_mg_number = _phase_mg_number(plot_source, "cpx")
    cpx_na2o = _numeric_column(plot_source, ("Na2O_cpx", "cpx_Na2O", "cpx Na2O"))
    cpx_al2o3 = _numeric_column(plot_source, ("Al2O3_cpx", "cpx_Al2O3", "cpx Al2O3"))
    liq_sio2 = _numeric_column(plot_source, ("SiO2_liq", "liq_SiO2", "liq SiO2"))
    liq_total_alkalis = _total_alkalis(plot_source, "liq")
    liq_mg_number = _phase_mg_number(plot_source, "liq")

    panel_specs = [
        (
            cpx_cao,
            cpx_mg_number,
            r"Clinopyroxene CaO (wt%)",
            "Clinopyroxene Mg#",
        ),
        (
            cpx_na2o,
            cpx_al2o3,
            r"Clinopyroxene Na$_2$O (wt%)",
            r"Clinopyroxene Al$_2$O$_3$ (wt%)",
        ),
        (
            liq_sio2,
            liq_total_alkalis,
            r"Liquid SiO$_2$ (wt%)",
            r"Liquid Na$_2$O + K$_2$O (wt%)",
        ),
        (
            liq_sio2,
            liq_mg_number,
            r"Liquid SiO$_2$ (wt%)",
            "Liquid Mg#",
        ),
    ]
    panel_specs = [
        panel_spec
        for panel_spec in panel_specs
        if panel_spec[0] is not None
        and panel_spec[1] is not None
        and pd.concat([panel_spec[0], panel_spec[1]], axis=1).dropna().shape[0] > 0
    ]
    if not panel_specs:
        return None

    model_order = [
        getattr(model, "model_name", type(model).__name__)
        for model in model_list
        if getattr(model, "model_name", type(model).__name__) in set(selected_models)
    ]
    for model_name in pd.unique(selected_models):
        if model_name not in model_order:
            model_order.append(model_name)

    label_lookup = _short_model_label_lookup(model_order, target)
    color_lookup = {
        model_name: MODEL_PLOT_COLORS[index % len(MODEL_PLOT_COLORS)]
        for index, model_name in enumerate(model_order)
    }

    n_panels = len(panel_specs)
    ncols = 2 if n_panels > 1 else 1
    nrows_plot = math.ceil(n_panels / ncols)
    figsize = (7.2, 2.65 * nrows_plot + 0.55)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 8,
        }
    ):
        fig, axes = plt.subplots(nrows_plot, ncols, figsize=figsize, dpi=200)
        axes = np.atleast_1d(axes).ravel()

        for panel_index, (axis, panel_spec) in enumerate(zip(axes, panel_specs)):
            x_values, y_values, x_label, y_label = panel_spec
            panel_df = pd.DataFrame(
                {
                    "x": x_values.reset_index(drop=True),
                    "y": y_values.reset_index(drop=True),
                    "model": selected_models.reset_index(drop=True),
                }
            ).dropna(subset=["x", "y"])

            for model_name in model_order:
                model_panel_df = panel_df.loc[panel_df["model"] == model_name]
                if model_panel_df.empty:
                    continue
                axis.scatter(
                    model_panel_df["x"],
                    model_panel_df["y"],
                    s=18,
                    alpha=0.82,
                    color=color_lookup[model_name],
                    edgecolors="#2F2F2F",
                    linewidths=0.35,
                    rasterized=True,
                )

            axis.text(
                0.02,
                0.96,
                f"({chr(97 + panel_index)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
            )
            axis.set_xlabel(x_label, fontsize=10)
            axis.set_ylabel(y_label, fontsize=10)
            axis.grid(True, color="#EAEAEA", linewidth=0.55)
            axis.spines["top"].set_visible(True)
            axis.spines["right"].set_visible(True)
            axis.tick_params(labelsize=9)

        for axis in axes[n_panels:]:
            axis.remove()

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color_lookup[model_name],
                markeredgecolor="#2F2F2F",
                markeredgewidth=0.7,
                markersize=5.2,
                label=label_lookup[model_name],
            )
            for model_name in model_order
        ]
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=min(4, len(legend_handles)),
                frameon=True,
                framealpha=0.88,
                facecolor="white",
                edgecolor="#BDBDBD",
                borderaxespad=0.0,
                borderpad=0.25,
                handletextpad=0.35,
                columnspacing=0.85,
            )
            fig.subplots_adjust(
                top=0.945,
                bottom=0.11,
                left=0.095,
                right=0.98,
                hspace=0.42,
                wspace=0.28,
            )
        else:
            fig.subplots_adjust(
                top=0.98,
                bottom=0.11,
                left=0.095,
                right=0.98,
                hspace=0.42,
                wspace=0.28,
            )

    return _figure_to_png(fig, dpi=200)


def _build_deviation_violin_png(
    workflow_obj: workflow_thermobarometry,
    model_list: list[ModelManager],
    target: str,
) -> bytes | None:
    """Render predicted-deviation violins with median and IQR markers."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    deviation_df = workflow_obj.calculated_deviation_df
    if deviation_df is None or deviation_df.empty:
        return None

    model_name_list = [
        getattr(model, "model_name", type(model).__name__) for model in model_list
    ]
    available_columns = [
        model_name for model_name in model_name_list if model_name in deviation_df
    ]
    available_columns.extend(
        column for column in deviation_df.columns if column not in available_columns
    )
    numeric_deviations = deviation_df[available_columns].apply(
        lambda column: pd.to_numeric(column, errors="coerce")
    )
    plottable_columns = [
        column
        for column in numeric_deviations.columns
        if numeric_deviations[column].dropna().shape[0] >= 2
    ]
    if not plottable_columns:
        return None

    plot_values = [
        numeric_deviations[column].dropna().to_numpy(dtype=float)
        for column in plottable_columns
    ]
    positions = np.arange(1, len(plot_values) + 1)
    label_lookup = _short_model_label_lookup(plottable_columns, target)
    x_labels = [label_lookup[column] for column in plottable_columns]
    unit = "kbar" if target == "P" else r"$^\circ$C"
    figure_width = min(max(5.4, 0.55 * len(plot_values) + 2.6), 10.8)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    ):
        fig, axis = plt.subplots(figsize=(figure_width, 3.85), dpi=200)
        violins = axis.violinplot(
            plot_values,
            positions=positions,
            widths=0.68,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for index, body in enumerate(violins["bodies"]):
            body.set_facecolor(MODEL_PLOT_COLORS[index % len(MODEL_PLOT_COLORS)])
            body.set_edgecolor("#222222")
            body.set_linewidth(0.7)
            body.set_alpha(0.72)

        for position, values in zip(positions, plot_values):
            q1, median, q3 = np.nanquantile(values, [0.25, 0.5, 0.75])
            axis.vlines(
                position,
                q1,
                q3,
                color="#111111",
                linewidth=2.1,
                zorder=4,
            )
            axis.hlines(
                [q1, q3],
                position - 0.13,
                position + 0.13,
                color="#111111",
                linewidth=1.0,
                zorder=5,
            )
            axis.hlines(
                median,
                position - 0.23,
                position + 0.23,
                color="white",
                linewidth=3.0,
                zorder=6,
            )
            axis.hlines(
                median,
                position - 0.23,
                position + 0.23,
                color="#111111",
                linewidth=1.25,
                zorder=7,
            )

        if all(np.nanmin(values) >= 0 for values in plot_values):
            axis.set_ylim(bottom=0)
        axis.set_ylabel(f"Predicted deviation ({unit})", fontsize=10)
        axis.set_xticks(positions)
        axis.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        axis.tick_params(axis="y", labelsize=9)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            handles=[
                Line2D([0], [0], color="#111111", linewidth=2.1, label="q1-q3"),
                Line2D([0], [0], color="#111111", linewidth=1.25, label="median"),
            ],
            frameon=False,
            loc="upper right",
            handlelength=1.3,
            borderaxespad=0.2,
        )
        fig.tight_layout(pad=0.8)

    return _figure_to_png(fig, dpi=200)


def _build_violin_png(
    workflow_obj: workflow_thermobarometry,
    model_list: list[ModelManager],
    target: str,
) -> bytes | None:
    """Render the existing violin plot to PNG bytes for Excel insertion."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from aims4pt.visualization.thermobarometry_plot import violin_plot

    model_uncertainty_dict = {
        getattr(model, "model_name", type(model).__name__): getattr(
            model, "uncertainty", None
        )
        for model in model_list
    }
    model_name_list = [
        getattr(model, "model_name", type(model).__name__) for model in model_list
    ]
    available_columns = [
        model_name
        for model_name in model_name_list
        if model_name in workflow_obj.prediction_df
    ]
    numeric_predictions = workflow_obj.prediction_df[available_columns].apply(
        lambda column: pd.to_numeric(column, errors="coerce")
    )
    if numeric_predictions.empty:
        return None
    plottable_columns = [
        column
        for column in numeric_predictions.columns
        if numeric_predictions[column].dropna().shape[0] >= 2
        and numeric_predictions[column].dropna().nunique() >= 2
    ]
    if not plottable_columns:
        return None
    plottable_uncertainty = {
        model_name: model_uncertainty_dict.get(model_name)
        for model_name in plottable_columns
    }

    try:
        violin_fig, _, _ = violin_plot(
            numeric_predictions[plottable_columns].copy(),
            target,
            plottable_columns,
            model_uncertainty=plottable_uncertainty,
        )
    except Exception:
        return None

    return _figure_to_png(violin_fig, dpi=200)


def _format_model_summary_values(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Drop heavy metadata columns and format numeric summary columns."""
    excluded_columns = {
        "Num_calibration_experiments",
        "Composition_range (wt%)",
        "P_range",
        "T_range (C)",
        f"{target}_range ({'kbar' if target == 'P' else 'C'})",
        "Supported_TAS_rock_types",
    }
    formatted = df.drop(
        columns=[column for column in excluded_columns if column in df.columns]
    ).copy()

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
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").round(
                precision
            )
        elif column == "Mean_calculated_deviation":
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").round(
                0
            )
        elif column == "OOD_ratio (%)":
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").round(
                1
            )
    return formatted


def build_report_payload(
    workflow_obj: workflow_thermobarometry,
    model_list: list[ModelManager],
    original_data: pd.DataFrame,
) -> ReportPayload:
    """Build reusable report tables and figure bytes from a completed workflow."""
    if not model_list:
        raise ValueError("Cannot build a report without models.")
    if workflow_obj.prediction_df is None or workflow_obj.prediction_df.empty:
        raise ValueError("Cannot build a report before predictions are available.")

    target = getattr(model_list[0], "T_P", getattr(workflow_obj, "T_P", ""))

    results_sheet = pd.concat(
        [
            original_data.reset_index(drop=True),
            workflow_obj.prediction_df.reset_index(drop=True),
        ],
        axis=1,
    )
    results_sheet_col = pd.MultiIndex.from_product(
        [["Original Data"], original_data.columns]
    ).append(
        pd.MultiIndex.from_product(
            [["Model Predictions"], workflow_obj.prediction_df.columns]
        )
    )
    results_sheet.columns = results_sheet_col

    model_metadata_df = pd.DataFrame(
        [_model_summary_row(model, target) for model in model_list]
    )
    results_summary_df = _results_summary(workflow_obj, target)
    model_summary_df = pd.concat(
        [
            model_metadata_df.set_index("Model_name"),
            results_summary_df.set_index("Model_name"),
        ],
        axis=1,
    ).reset_index()
    model_summary_df = _format_model_summary_values(model_summary_df, target)

    model_votes_df = _model_votes(workflow_obj)
    failure_reason_df = workflow_obj.failure_reason_df
    selected_models_series = workflow_obj.get_best_model_series()
    ranking_details_df = pd.concat(
        [failure_reason_df, selected_models_series.rename("Selected_model")], axis=1
    )
    violin_plot_png = _build_violin_png(workflow_obj, model_list, target)
    model_selection_plot_png = _build_model_selection_png(
        workflow_obj, model_list, original_data, target
    )
    deviation_violin_plot_png = _build_deviation_violin_png(
        workflow_obj, model_list, target
    )

    return ReportPayload(
        results_sheet=results_sheet,
        model_summary_df=model_summary_df,
        model_votes_df=model_votes_df,
        ranking_details_df=ranking_details_df,
        violin_plot_png=violin_plot_png,
        target=target,
        model_selection_plot_png=model_selection_plot_png,
        deviation_violin_plot_png=deviation_violin_plot_png,
    )


def write_report_excel(
    payload: ReportPayload,
    output: str | BinaryIO | io.BytesIO,
) -> None:
    """Write an Excel report to a filesystem path or in-memory file object."""
    if payload.results_sheet.empty or payload.model_summary_df.empty:
        raise ValueError("Cannot generate an empty report.")

    with pd.ExcelWriter(
        output,
        engine="xlsxwriter",
        engine_kwargs={"options": {"in_memory": True}},
    ) as writer:
        payload.results_sheet.to_excel(writer, sheet_name="Results", index=True)
        payload.model_summary_df.to_excel(
            writer, sheet_name="Model Summary", index=False
        )

        vote_start_row = len(payload.model_summary_df) + 3
        payload.model_votes_df.to_excel(
            writer,
            sheet_name="Model Summary",
            index=False,
            startrow=vote_start_row,
            float_format="%.2f",
        )

        worksheet = writer.sheets["Model Summary"]
        plot_start_row = max(
            len(payload.model_summary_df) + len(payload.model_votes_df) + 6,
            11,
        )
        image_options = {"x_scale": 0.62, "y_scale": 0.62}
        if payload.model_selection_plot_png:
            worksheet.insert_image(
                plot_start_row,
                0,
                "selected_model_composition.png",
                {
                    "image_data": io.BytesIO(payload.model_selection_plot_png),
                    **image_options,
                },
            )
            plot_start_row += 36
        if payload.deviation_violin_plot_png:
            worksheet.insert_image(
                plot_start_row,
                0,
                "predicted_deviation_violin.png",
                {
                    "image_data": io.BytesIO(payload.deviation_violin_plot_png),
                    **image_options,
                },
            )
            plot_start_row += 24
        if payload.violin_plot_png:
            worksheet.insert_image(
                plot_start_row,
                0,
                "violin_plot.png",
                {"image_data": io.BytesIO(payload.violin_plot_png), **image_options},
            )

        payload.ranking_details_df.to_excel(
            writer, sheet_name="Ranking Details", index=True
        )


def report_excel(
    workflow_obj: workflow_thermobarometry,
    model_list: list[ModelManager],
    original_data: pd.DataFrame,
    out_path: str | BinaryIO | io.BytesIO,
) -> None:
    """Export workflow results to a multi-sheet Excel report."""
    payload = build_report_payload(workflow_obj, model_list, original_data)
    write_report_excel(payload, out_path)
