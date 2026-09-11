"""Pressure-distribution and regression diagnostics for AIMS4PT models."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from patsy import dmatrix
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from statsmodels.regression.quantile_regression import QuantReg

from aims4pt.data_tools.compositions import cpx_calculation
from aims4pt.data_tools.equilibrium import kdEquilibrium_test
from paper.scripts.figure_helpers import (
    apply_figure_font_sizes,
    model_abbreviation_labels,
)


CPX_COLUMNS = [
    "SiO2_cpx",
    "TiO2_cpx",
    "Al2O3_cpx",
    "Cr2O3_cpx",
    "FeO_cpx",
    "MnO_cpx",
    "MgO_cpx",
    "NiO_cpx",
    "CaO_cpx",
    "Na2O_cpx",
    "K2O_cpx",
    "P2O5_cpx",
]

LIQ_COLUMNS = [
    "SiO2_liq",
    "TiO2_liq",
    "Al2O3_liq",
    "FeO_liq",
    "MnO_liq",
    "MgO_liq",
    "CaO_liq",
    "Na2O_liq",
    "K2O_liq",
    "P2O5_liq",
]

PRESSURE_COLUMN = "P (kbar)"
TEMPERATURE_COLUMN = "T (C)"

UNIQUE_TRAINING_MODEL_NAMES = [
    "Chicchi et al., 2023 (cpx_only)",
    "Jorgenson et al., 2022 (cpx_only)",
    "Ágreda-López et al., 2024 (cpx_only)",
    "Petrelli et al., 2020 (cpx_only)",
    "Higgins et al., 2021 (cpx_only)",
    "Wang et al., 2021 (cpx_only)",
]
NEAVE_PUTIRKA_MODEL_PREFIX = "Neave & Putirka, 2017 Pu08_eq33_T; eq1_P"
QUANTILES = (0.25, 0.50, 0.75)


def configure_publication_style() -> None:
    """Apply compact, editable-text matplotlib defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def prepare_independent_dataset(
    data_path: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Index, pd.Index, pd.DataFrame]:
    """Reproduce the filtering and stratified split used by notebook 03."""
    raw = pd.read_excel(data_path)
    audit_rows = [{"stage": "raw", "N": len(raw)}]

    cpx_total = raw[CPX_COLUMNS].sum(axis=1)
    cleaned = raw.loc[cpx_total.between(98, 102, inclusive="neither")].reset_index(drop=True)
    audit_rows.append({"stage": "cpx total 98-102 wt.%", "N": len(cleaned)})

    cpx_params = cpx_calculation(cleaned[CPX_COLUMNS])
    m_div_t = cpx_params["(Ca+Fe+Mg)/Si"]
    cleaned = cleaned.loc[m_div_t.between(0.9, 1.1, inclusive="neither")].reset_index(drop=True)
    audit_rows.append({"stage": "cpx stoichiometry 0.9-1.1", "N": len(cleaned)})

    kd_mask, _ = kdEquilibrium_test(
        cleaned[CPX_COLUMNS],
        cleaned[LIQ_COLUMNS],
        kd=0.28,
        error=0.08,
        mode="Fe-Mg",
    )
    cleaned = cleaned.loc[kd_mask].reset_index(drop=True)
    audit_rows.append({"stage": "Fe-Mg Kd equilibrium", "N": len(cleaned)})

    pressure = cleaned[PRESSURE_COLUMN]
    target_bin_size = 50
    num_bins = max(2, min(10, len(pressure) // target_bin_size))
    pressure_strata = pd.qcut(pressure, q=num_bins, labels=False, duplicates="drop")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    (train_position, test_position), = splitter.split(cleaned, pressure_strata)
    train_index = cleaned.iloc[train_position].index
    test_index = cleaned.iloc[test_position].index

    audit_rows.extend(
        [
            {"stage": "stratified training subset", "N": len(train_index)},
            {"stage": "stratified test subset", "N": len(test_index)},
        ]
    )
    return cleaned, train_index, test_index, pd.DataFrame(audit_rows)


def _phase_type(model) -> str:
    return "cpx_only" if bool(model.cpx_only) else "cpx_liq"


def _short_model_labels(models: Sequence, *, include_phase: bool = True) -> dict[str, str]:
    names = [model.model_name for model in models]
    abbreviations = model_abbreviation_labels(names, "P")
    labels = {}
    for model, abbreviation in zip(models, abbreviations):
        if include_phase:
            phase = "cpx" if model.cpx_only else "cpx-liq"
            labels[model.model_name] = f"{abbreviation} ({phase})"
        else:
            labels[model.model_name] = abbreviation
    return labels


def select_unique_training_models(models: Sequence) -> list:
    """Select one representative model per study-level training dataset."""
    model_by_name = {model.model_name: model for model in models}
    selected = []
    missing = []

    for model_name in UNIQUE_TRAINING_MODEL_NAMES:
        model = model_by_name.get(model_name)
        if model is None:
            missing.append(model_name)
        else:
            selected.append(model)

    neave_model = next(
        (model for model in models if model.model_name.startswith(NEAVE_PUTIRKA_MODEL_PREFIX)),
        None,
    )
    if neave_model is None:
        missing.append(NEAVE_PUTIRKA_MODEL_PREFIX)
    else:
        selected.append(neave_model)

    if missing:
        raise KeyError(f"Could not resolve training-dataset models: {missing}")
    return selected


def _get_training_dataframe(model) -> tuple[pd.DataFrame | None, str]:
    """Return the model's training subset used to fit or calibrate it."""
    return getattr(model, "X_cpx_training", None), "X_cpx_training"


def collect_pressure_distributions(
    independent_pressure: pd.Series,
    models: Sequence,
    *,
    p_min: float = 0.0,
    p_max: float = 10.0,
) -> tuple[list[dict], pd.DataFrame]:
    """Collect full-dataset and in-range pressure statistics for every dataset."""
    datasets: list[dict] = [
        {
            "dataset": "Independent experimental dataset",
            "model": None,
            "phase_type": "independent",
            "pressure": pd.to_numeric(independent_pressure, errors="coerce"),
            "available": True,
        }
    ]
    short_labels = _short_model_labels(models, include_phase=False)

    for model in models:
        training, training_source = _get_training_dataframe(model)
        available = isinstance(training, pd.DataFrame) and "P_kbar" in training.columns
        pressure = (
            pd.to_numeric(training["P_kbar"], errors="coerce")
            if available
            else pd.Series(dtype=float)
        )
        datasets.append(
            {
                "dataset": short_labels[model.model_name],
                "model": model.model_name,
                "phase_type": _phase_type(model),
                "pressure": pressure,
                "available": available,
                "training_source": training_source,
            }
        )

    summary_rows = []
    for item in datasets:
        finite = item["pressure"].dropna()
        in_range = finite.loc[finite.between(p_min, p_max, inclusive="both")]
        summary_rows.append(
            {
                "dataset": item["dataset"],
                "model": item["model"],
                "phase_type": item["phase_type"],
                "training_available": item["available"],
                "training_source": item.get("training_source", "independent dataset"),
                "N_total": int(len(finite)),
                "N_0_10_kbar": int(len(in_range)),
                "fraction_0_10_kbar": len(in_range) / len(finite) if len(finite) else np.nan,
                "mean_P_0_10_kbar": in_range.mean() if len(in_range) else np.nan,
                "median_P_0_10_kbar": in_range.median() if len(in_range) else np.nan,
            }
        )
        item["pressure_in_range"] = in_range

    return datasets, pd.DataFrame(summary_rows)


def add_dataset_regression_slopes(
    histogram_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Attach cpx-only and cpx-liquid test slopes to each unique dataset row."""
    required = {"model_short", "phase_type", "slope"}
    missing = required.difference(regression_summary.columns)
    if missing:
        raise KeyError(f"Regression summary is missing required columns: {sorted(missing)}")

    slopes = regression_summary.loc[:, ["model_short", "phase_type", "slope"]].copy()
    slopes["dataset"] = slopes["model_short"].str.split(" (", n=1, regex=False).str[0]
    slope_table = slopes.pivot_table(
        index="dataset",
        columns="phase_type",
        values="slope",
        aggfunc="first",
    ).rename(
        columns={
            "cpx_only": "slope_cpx_only",
            "cpx_liq": "slope_cpx_liq",
        }
    )
    slope_table.columns.name = None
    return histogram_summary.merge(slope_table, left_on="dataset", right_index=True, how="left")


def _quantile_basis(
    pressure: np.ndarray,
    method: str,
    *,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """Build a low-complexity basis over an explicit pressure domain."""
    pressure = np.asarray(pressure, dtype=float)
    if method == "quadratic":
        midpoint = (lower_bound + upper_bound) / 2.0
        half_range = (upper_bound - lower_bound) / 2.0
        scaled = (pressure - midpoint) / half_range
        return np.column_stack([np.ones_like(scaled), scaled, scaled**2])
    if method == "spline":
        formula = (
            "cr(P, knots=(4.0, 7.0), "
            f"lower_bound={lower_bound!r}, upper_bound={upper_bound!r}) - 1"
        )
        return np.asarray(
            dmatrix(
                formula,
                {"P": pressure},
                return_type="dataframe",
            ),
            dtype=float,
        )
    raise ValueError("method must be 'quadratic' or 'spline'.")


def _aligned_test_residuals(
    true_pressure: pd.Series,
    predicted_pressure: pd.Series,
    *,
    p_min: float | None,
    p_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite measured pressures and delta-P values in the target range."""
    measured, predicted = pd.to_numeric(
        true_pressure, errors="coerce"
    ).align(pd.to_numeric(predicted_pressure, errors="coerce"), join="inner")
    valid = np.isfinite(measured) & np.isfinite(predicted)
    if p_min is not None:
        valid &= measured >= p_min
    if p_max is not None:
        valid &= measured <= p_max
    x = measured.loc[valid].to_numpy(dtype=float)
    delta_p = predicted.loc[valid].to_numpy(dtype=float) - x
    return x, delta_p


def _resolve_quantile_domain(
    true_pressure: pd.Series,
    *,
    p_min: float | None,
    p_max: float | None,
) -> tuple[float, float]:
    """Resolve common basis boundaries from the requested analysis population."""
    measured = pd.to_numeric(true_pressure, errors="coerce")
    valid = np.isfinite(measured)
    if p_min is not None:
        valid &= measured >= p_min
    if p_max is not None:
        valid &= measured <= p_max
    selected = measured.loc[valid]
    if selected.empty:
        raise ValueError("No finite true-pressure values remain in the requested range.")

    lower_bound = float(selected.min()) if p_min is None else float(p_min)
    upper_bound = float(selected.max()) if p_max is None else float(p_max)
    if not lower_bound < upper_bound:
        raise ValueError("The quantile-regression pressure domain must have positive width.")
    return lower_bound, upper_bound


def _pinball_loss(observed: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    """Calculate mean quantile pinball loss."""
    error = observed - predicted
    return float(
        np.mean(np.where(error >= 0, quantile * error, (quantile - 1.0) * error))
    )


def calculate_quantile_envelopes(
    true_pressure: pd.Series,
    predictions: pd.DataFrame,
    models: Sequence,
    *,
    method: str = "spline",
    p_min: float | None = 1.0,
    p_max: float | None = 10.0,
    plot_p_min: float | None = None,
    plot_p_max: float | None = None,
    grid_size: int = 181,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit Q1, median, and Q3 delta-P curves for every pressure model."""
    short_labels = _short_model_labels(models)
    lower_bound, upper_bound = _resolve_quantile_domain(
        true_pressure,
        p_min=p_min,
        p_max=p_max,
    )
    grid_lower = lower_bound if plot_p_min is None else float(plot_p_min)
    grid_upper = upper_bound if plot_p_max is None else float(plot_p_max)
    if grid_lower < lower_bound or grid_upper > upper_bound:
        raise ValueError("The plotting grid must stay inside the fitted pressure domain.")
    pressure_grid = np.linspace(grid_lower, grid_upper, grid_size)
    grid_basis = _quantile_basis(
        pressure_grid,
        method,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    curve_frames = []
    summary_rows = []

    for model in models:
        model_name = model.model_name
        x, delta_p = _aligned_test_residuals(
            true_pressure,
            predictions[model_name],
            p_min=p_min,
            p_max=p_max,
        )
        if len(x) < 5 or np.unique(x).size < 3:
            continue

        basis = _quantile_basis(
            x,
            method,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        fitted_curves = []
        iterations = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for quantile in QUANTILES:
                fitted = QuantReg(delta_p, basis).fit(
                    q=quantile,
                    max_iter=20000,
                    p_tol=1e-5,
                )
                fitted_curves.append(
                    np.asarray(fitted.predict(grid_basis), dtype=float)
                )
                iterations.append(int(fitted.iterations))

        raw_curves = np.vstack(fitted_curves)
        crossing = (raw_curves[0] > raw_curves[1]) | (
            raw_curves[1] > raw_curves[2]
        )
        ordered_curves = np.sort(raw_curves, axis=0)
        model_short = short_labels[model_name]
        curve_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "model_short": model_short,
                    "phase_type": _phase_type(model),
                    "method": method,
                    "P_true_kbar": pressure_grid,
                    "deltaP_q25_kbar": ordered_curves[0],
                    "deltaP_q50_kbar": ordered_curves[1],
                    "deltaP_q75_kbar": ordered_curves[2],
                    "raw_quantiles_crossed": crossing,
                }
            )
        )
        summary_rows.append(
            {
                "model": model_name,
                "model_short": model_short,
                "phase_type": _phase_type(model),
                "N": int(len(x)),
                "method": method,
                "fit_P_min_kbar": lower_bound,
                "fit_P_max_kbar": upper_bound,
                "plot_P_min_kbar": grid_lower,
                "plot_P_max_kbar": grid_upper,
                "crossing_fraction_before_rearrangement": float(np.mean(crossing)),
                "monotone_rearrangement_applied": bool(np.any(crossing)),
                "max_iterations": int(np.max(iterations)),
            }
        )

    if not curve_frames:
        return pd.DataFrame(), pd.DataFrame(summary_rows)
    return pd.concat(curve_frames, ignore_index=True), pd.DataFrame(summary_rows)


def compare_quantile_methods(
    true_pressure: pd.Series,
    predictions: pd.DataFrame,
    models: Sequence,
    *,
    methods: Sequence[str] = ("quadratic", "spline"),
    p_min: float | None = 1.0,
    p_max: float | None = 10.0,
    plot_p_min: float | None = None,
    plot_p_max: float | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare quantile bases using fixed-fold out-of-sample pinball loss."""
    short_labels = _short_model_labels(models)
    lower_bound, upper_bound = _resolve_quantile_domain(
        true_pressure,
        p_min=p_min,
        p_max=p_max,
    )
    rows = []

    for model in models:
        model_name = model.model_name
        x, delta_p = _aligned_test_residuals(
            true_pressure,
            predictions[model_name],
            p_min=p_min,
            p_max=p_max,
        )
        if len(x) < n_splits or np.unique(x).size < 3:
            continue

        folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for method in methods:
            basis = _quantile_basis(
                x,
                method,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            losses = []
            max_iterations = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for train_position, test_position in folds.split(basis):
                    for quantile in QUANTILES:
                        fitted = QuantReg(
                            delta_p[train_position], basis[train_position]
                        ).fit(q=quantile, max_iter=20000, p_tol=1e-5)
                        max_iterations = max(max_iterations, int(fitted.iterations))
                        fitted_delta = fitted.predict(basis[test_position])
                        losses.append(
                            _pinball_loss(
                                delta_p[test_position], fitted_delta, quantile
                            )
                        )

            _, fitted_summary = calculate_quantile_envelopes(
                true_pressure,
                predictions[[model_name]],
                [model],
                method=method,
                p_min=p_min,
                p_max=p_max,
                plot_p_min=plot_p_min,
                plot_p_max=plot_p_max,
            )
            crossing_fraction = (
                fitted_summary.iloc[0]["crossing_fraction_before_rearrangement"]
                if not fitted_summary.empty
                else np.nan
            )
            rows.append(
                {
                    "model": model_name,
                    "model_short": short_labels[model_name],
                    "phase_type": _phase_type(model),
                    "N": int(len(x)),
                    "method": method,
                    "fit_P_min_kbar": lower_bound,
                    "fit_P_max_kbar": upper_bound,
                    "cv_pinball_loss": float(np.mean(losses)),
                    "crossing_fraction_before_rearrangement": crossing_fraction,
                    "max_cv_iterations": int(max_iterations),
                }
            )

    return pd.DataFrame(rows)


def plot_pressure_histograms(
    datasets: Sequence[dict],
    *,
    regression_summary: pd.DataFrame | None = None,
    quantile_curves: pd.DataFrame | None = None,
    p_min: float = 0.0,
    p_max: float = 10.0,
    ncols: int = 4,
    phase_type: str | None = None,
    delta_p_ylim: tuple[float, float] = (-8.0, 8.0),
    residual_legend_title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Overlay conditional delta-P summaries on paper-styled pressure histograms."""
    configure_publication_style()
    if phase_type not in {None, "cpx_only", "cpx_liq"}:
        raise ValueError("phase_type must be None, 'cpx_only', or 'cpx_liq'.")
    if regression_summary is not None and quantile_curves is not None:
        raise ValueError("Provide regression_summary or quantile_curves, not both.")

    plotted_datasets = [
        item
        for item in datasets
        if phase_type is None or item["phase_type"] in {"independent", phase_type}
    ]
    nrows = math.ceil(len(plotted_datasets) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12.5, 3.15 * nrows + 0.45),
        dpi=200,
        sharex=True,
        squeeze=False,
    )
    bins = np.arange(p_min, p_max + 1.0, 1.0)
    x_line = np.array([p_min, p_max], dtype=float)

    residual_rows = None
    residual_mode = None
    if regression_summary is not None:
        required = {
            "model_short",
            "phase_type",
            "deltaP_slope",
            "deltaP_intercept_kbar",
            "deltaP_sigma_kbar",
        }
        missing = required.difference(regression_summary.columns)
        if missing:
            raise KeyError(f"Regression summary is missing required columns: {sorted(missing)}")

        residual_rows = regression_summary.copy()
        residual_mode = "ols_sigma"
    elif quantile_curves is not None:
        required = {
            "model_short",
            "phase_type",
            "P_true_kbar",
            "deltaP_q25_kbar",
            "deltaP_q50_kbar",
            "deltaP_q75_kbar",
        }
        missing = required.difference(quantile_curves.columns)
        if missing:
            raise KeyError(f"Quantile curves are missing required columns: {sorted(missing)}")
        residual_rows = quantile_curves.copy()
        residual_mode = "quantile"

    if residual_rows is not None:
        residual_rows["dataset"] = residual_rows["model_short"].str.split(
            " (", n=1, regex=False
        ).str[0]
        training_panels = {
            item["dataset"]
            for item in plotted_datasets
            if item["phase_type"] != "independent"
        }
        is_putirka = residual_rows["dataset"].str.startswith("Pu08_")
        maps_to_training_panel = residual_rows["dataset"].isin(training_panels)
        unmapped = residual_rows.loc[
            ~(maps_to_training_panel | is_putirka), "model_short"
        ].drop_duplicates().tolist()
        if unmapped:
            raise ValueError(f"Regression models cannot be mapped to histogram panels: {unmapped}")
        residual_rows = residual_rows.loc[maps_to_training_panel].copy()
        residual_rows["panel"] = residual_rows["dataset"]

    for index, (ax, item) in enumerate(zip(axes.ravel(), plotted_datasets)):
        color = {
            "independent": "#D55E00",
            "cpx_only": "#4477AA",
            "cpx_liq": "#4477AA",
        }[item["phase_type"]]
        values = item["pressure_in_range"]
        title = (
            "Independent"
            if item["phase_type"] == "independent"
            else item["dataset"]
        )
        ax.set_title(rf"{title} ($n={len(values)}$)", pad=7)
        ax.set_xlim(p_min, p_max)
        ax.set_xticks(np.arange(p_min, p_max + 0.1, 1.0))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.90", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        ax.text(
            -0.12,
            1.06,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
            clip_on=False,
            zorder=5,
        )

        if not item["available"]:
            ax.text(
                0.5,
                0.5,
                "Training dataset\nunavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="0.45",
                fontsize=6.5,
            )
            ax.set_yticks([])
            continue

        ax.hist(
            values,
            bins=bins,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        mean_in_range = pd.to_numeric(values, errors="coerce").dropna().mean()
        if np.isfinite(mean_in_range):
            ax.axvline(
                mean_in_range,
                color="0.30",
                linestyle="-.",
                linewidth=1.2,
            )
            ax.annotate(
                rf"Mean = {mean_in_range:.1f}",
                xy=(mean_in_range, 0.72),
                xycoords=ax.get_xaxis_transform(),
                xytext=(4, 0),
                textcoords="offset points",
                rotation=0,
                rotation_mode="anchor",
                ha="left",
                va="center",
                fontsize=8.5,
                color="0.20",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.6),
                zorder=6,
            )

        panel_rows = (
            residual_rows.loc[residual_rows["panel"] == item["dataset"]]
            if residual_rows is not None
            else pd.DataFrame()
        )
        if panel_rows.empty:
            if index % ncols == 0:
                ax.set_ylabel("Count")
            continue

        residual_ax = ax.twinx()
        residual_ax.set_ylim(*delta_p_ylim)
        residual_ax.set_yticks(
            np.arange(
                math.ceil(delta_p_ylim[0] / 2.0) * 2.0,
                math.floor(delta_p_ylim[1] / 2.0) * 2.0 + 0.1,
                2.0,
            )
        )
        residual_ax.axhline(
            0.0,
            color="0.10",
            linestyle=":",
            linewidth=1.6,
            zorder=4,
        )
        residual_ax.patch.set_visible(False)
        residual_ax.spines["top"].set_visible(False)
        residual_ax.spines["left"].set_visible(False)
        residual_ax.spines["right"].set_linewidth(0.8)

        if index % ncols == ncols - 1:
            residual_ax.set_ylabel(r"$\Delta P=P_{pred}-P_{true}$ (kbar)")
            residual_ax.tick_params(axis="y", labelsize=10, colors="0.20")
        else:
            residual_ax.tick_params(axis="y", right=False, labelright=False)
            residual_ax.spines["right"].set_visible(False)

        if residual_rows is not None:
            phase_counts = {"cpx_only": 0, "cpx_liq": 0}
            if residual_mode == "quantile":
                curve_groups = panel_rows.groupby(
                    ["model_short", "phase_type"], sort=False
                )
                for (_, phase), curve in curve_groups:
                    curve = curve.sort_values("P_true_kbar")
                    phase_index = phase_counts[phase]
                    phase_counts[phase] += 1
                    if phase == "cpx_only":
                        color_line = "#CC3311"
                        line_styles = ["-", "-.", (0, (5, 1.5))]
                        label = "cpx-only"
                    else:
                        color_line = "#8DAA00"
                        line_styles = ["--", (0, (2, 1)), (0, (6, 2, 1, 2))]
                        label = "cpx-liq"
                    if phase_index:
                        label = f"{label} {phase_index + 1}"
                    label = f"{label} median (Q1-Q3)"

                    residual_ax.fill_between(
                        curve["P_true_kbar"],
                        curve["deltaP_q25_kbar"],
                        curve["deltaP_q75_kbar"],
                        color=color_line,
                        alpha=0.16,
                        linewidth=0,
                        zorder=4,
                    )
                    residual_ax.plot(
                        curve["P_true_kbar"],
                        curve["deltaP_q50_kbar"],
                        color=color_line,
                        linestyle=line_styles[phase_index % len(line_styles)],
                        linewidth=1.6,
                        label=label,
                        zorder=5,
                    )
            else:
                for _, row in panel_rows.iterrows():
                    slope = row["deltaP_slope"]
                    intercept = row["deltaP_intercept_kbar"]
                    sigma = row["deltaP_sigma_kbar"]
                    if not all(
                        np.isfinite(value) for value in (slope, intercept, sigma)
                    ):
                        continue

                    phase = row["phase_type"]
                    phase_index = phase_counts[phase]
                    phase_counts[phase] += 1
                    if phase == "cpx_only":
                        color_line = "#CC3311"
                        line_styles = ["-", "-.", (0, (5, 1.5))]
                        label = "cpx-only"
                    else:
                        color_line = "#8DAA00"
                        line_styles = ["--", (0, (2, 1)), (0, (6, 2, 1, 2))]
                        label = "cpx-liq"

                    if phase_index:
                        label = f"{label} {phase_index + 1}"
                    label = f"{label} fit"

                    delta_line = slope * x_line + intercept
                    residual_ax.plot(
                        x_line,
                        delta_line,
                        color=color_line,
                        linestyle=line_styles[phase_index % len(line_styles)],
                        linewidth=1.6,
                        label=label,
                        zorder=5,
                    )

        if index % ncols == 0:
            ax.set_ylabel("Count")

    for ax in axes.ravel()[len(plotted_datasets) :]:
        ax.remove()

    for ax in axes[-1, :]:
        if ax in fig.axes:
            ax.set_xlabel(r"$P_{true}$ (kbar)")

    apply_figure_font_sizes(
        fig,
        labelsize=15,
        ylabelsize=15,
        tick_labelsize=12,
        title_size=14,
        legend_font_size=11,
    )

    if residual_rows is not None:
        shared_handles = []
        shared_labels = []
        phase_types = set(residual_rows["phase_type"])
        if residual_mode == "quantile":
            if "cpx_only" in phase_types:
                shared_handles.append(
                    (
                        Patch(facecolor="#CC3311", edgecolor="none", alpha=0.16),
                        Line2D([], [], color="#CC3311", linestyle="-", linewidth=1.6),
                    )
                )
                shared_labels.append("cpx-only median (Q1-Q3)")
            if "cpx_liq" in phase_types:
                shared_handles.append(
                    (
                        Patch(facecolor="#8DAA00", edgecolor="none", alpha=0.16),
                        Line2D([], [], color="#8DAA00", linestyle="--", linewidth=1.6),
                    )
                )
                shared_labels.append("cpx-liq median (Q1-Q3)")
        else:
            if "cpx_only" in phase_types:
                shared_handles.append(
                    Line2D([], [], color="#CC3311", linestyle="-", linewidth=1.6)
                )
                shared_labels.append("cpx-only fit")
            if "cpx_liq" in phase_types:
                shared_handles.append(
                    Line2D([], [], color="#8DAA00", linestyle="--", linewidth=1.6)
                )
                shared_labels.append("cpx-liq fit")
        shared_handles.append(
            Line2D([], [], color="0.10", linestyle=":", linewidth=1.6)
        )
        shared_labels.append(r"$\Delta P=0$")
        fig.legend(
            shared_handles,
            shared_labels,
            handler_map={tuple: HandlerTuple(ndivide=1)},
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=len(shared_handles),
            fontsize=10.5,
            frameon=False,
            handlelength=2.5,
            columnspacing=1.8,
            title=residual_legend_title,
            title_fontsize=9.5,
        )

    bottom_margin = 0.115 if residual_legend_title else 0.095
    fig.tight_layout(rect=(0.025, bottom_margin, 0.995, 0.995), h_pad=1.5, w_pad=1.2)
    return fig, axes


def load_or_calculate_test_predictions(
    cache_path: str | Path,
    models: Sequence,
    test_cpx: pd.DataFrame,
    test_liq: pd.DataFrame,
) -> pd.DataFrame:
    """Reuse a complete aligned cache; otherwise calculate model predictions."""
    cache_path = Path(cache_path)
    model_names = [model.model_name for model in models]

    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0)
        try:
            cached.index = cached.index.astype(test_cpx.index.dtype)
        except (TypeError, ValueError):
            pass
        if set(model_names).issubset(cached.columns) and set(test_cpx.index).issubset(cached.index):
            return cached.loc[test_cpx.index, model_names].copy()

    predictions = pd.DataFrame(index=test_cpx.index)
    for model in models:
        predictions[model.model_name] = np.asarray(model.predict(test_cpx, test_liq)).reshape(-1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(cache_path, index=True)
    return predictions


def calculate_regression_statistics(
    true_pressure: pd.Series,
    predictions: pd.DataFrame,
    models: Sequence,
    *,
    p_min: float | None = 1.0,
    p_max: float | None = 10.0,
) -> pd.DataFrame:
    """Fit pressure and delta-P trends and estimate residual scatter for each model."""
    true_pressure = pd.to_numeric(true_pressure, errors="coerce")
    short_labels = _short_model_labels(models)
    rows = []

    for model in models:
        model_name = model.model_name
        predicted = pd.to_numeric(predictions[model_name], errors="coerce")
        aligned_true, aligned_pred = true_pressure.align(predicted, join="inner")
        valid = np.isfinite(aligned_true) & np.isfinite(aligned_pred)
        if p_min is not None:
            valid &= aligned_true >= p_min
        if p_max is not None:
            valid &= aligned_true <= p_max
        x = aligned_true.loc[valid].to_numpy(dtype=float)
        y = aligned_pred.loc[valid].to_numpy(dtype=float)

        if len(x) >= 2 and np.unique(x).size >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            correlation = np.corrcoef(x, y)[0, 1]
            r_squared = correlation**2
            if np.isclose(slope, 1.0, atol=1e-8):
                intersection = np.nan
                intersection_note = "coincident" if np.isclose(intercept, 0.0, atol=1e-8) else "parallel"
            else:
                intersection = intercept / (1.0 - slope)
                inside_lower = p_min is None or intersection >= p_min
                inside_upper = p_max is None or intersection <= p_max
                if p_min == 1.0 and p_max == 10.0:
                    inside_note = "within 1-10 kbar"
                    outside_note = "outside 1-10 kbar"
                else:
                    inside_note = "within fitted range"
                    outside_note = "outside fitted range"
                intersection_note = inside_note if inside_lower and inside_upper else outside_note
        else:
            slope = intercept = r_squared = intersection = np.nan
            intersection_note = "insufficient data"

        residual = y - x
        if len(x) >= 2 and np.unique(x).size >= 2:
            delta_slope, delta_intercept = np.polyfit(x, residual, 1)
            delta_fitted = delta_slope * x + delta_intercept
            if len(x) > 2:
                delta_sigma = np.sqrt(
                    np.sum((residual - delta_fitted) ** 2) / (len(x) - 2)
                )
            else:
                delta_sigma = np.nan
        else:
            delta_slope = delta_intercept = delta_sigma = np.nan
        rows.append(
            {
                "model": model_name,
                "model_short": short_labels[model_name],
                "phase_type": _phase_type(model),
                "N": int(len(x)),
                "slope": slope,
                "intercept_kbar": intercept,
                "deltaP_slope": delta_slope,
                "deltaP_intercept_kbar": delta_intercept,
                "deltaP_sigma_kbar": delta_sigma,
                "R_squared": r_squared,
                "intersection_with_1to1_kbar": intersection,
                "intersection_note": intersection_note,
                "mean_y_true_kbar": np.mean(x) if len(x) else np.nan,
                "mean_y_pred_kbar": np.mean(y) if len(y) else np.nan,
                "bias_kbar": np.mean(residual) if len(residual) else np.nan,
                "RMSE_kbar": np.sqrt(np.mean(residual**2)) if len(residual) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_regression_lines(
    regression_summary: pd.DataFrame,
    *,
    p_min: float = 1.0,
    p_max: float = 10.0,
) -> tuple[plt.Figure, np.ndarray]:
    """Compare model OLS lines, in-range intersections, and mean predictions."""
    configure_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True, sharey=True)
    x_line = np.array([p_min, p_max])
    colors = plt.get_cmap("tab10").colors

    for ax, phase_type, title in zip(
        axes,
        ["cpx_only", "cpx_liq"],
        ["Clinopyroxene-only barometers", "Clinopyroxene-liquid barometers"],
    ):
        subset = regression_summary.loc[regression_summary["phase_type"] == phase_type]
        ax.plot(x_line, x_line, color="black", linestyle="--", linewidth=1.1, label="1:1 line")

        for index, (_, row) in enumerate(subset.iterrows()):
            color = colors[index % len(colors)]
            if not np.isfinite(row["slope"]):
                continue
            y_line = row["slope"] * x_line + row["intercept_kbar"]
            ax.plot(x_line, y_line, color=color, linewidth=1.2, label=row["model_short"])
            ax.scatter(
                row["mean_y_true_kbar"],
                row["mean_y_pred_kbar"],
                s=18,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            intersection = row["intersection_with_1to1_kbar"]
            if np.isfinite(intersection) and p_min <= intersection <= p_max:
                ax.scatter(
                    intersection,
                    intersection,
                    marker="x",
                    s=26,
                    color=color,
                    linewidth=1.0,
                    zorder=4,
                )

        ax.set_title(title)
        ax.set_xlim(p_min, p_max)
        ax.set_ylim(p_min, p_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Measured pressure (kbar)")
        ax.grid(color="0.9", linewidth=0.5)
        ax.legend(fontsize=5.7, loc="upper left")

    axes[0].set_ylabel("Predicted pressure (kbar)")
    fig.suptitle(
        "OLS fits on the independent test subset (1-10 kbar)",
        fontsize=9,
        y=1.01,
    )
    fig.text(
        0.5,
        0.01,
        "Diamonds: mean measured/predicted pressure; crosses: in-range intersections with the 1:1 line",
        ha="center",
        va="bottom",
        fontsize=6,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig, axes


def save_figure_bundle(
    fig: plt.Figure,
    output_dir: str | Path,
    stem: str,
    *,
    dpi: int = 600,
) -> list[Path]:
    """Save editable vector files and a high-resolution preview."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.png",
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=dpi, bbox_inches="tight")
    return paths


def export_summary_tables(
    histogram_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Export the exact source tables used by the diagnostic figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    histogram_path = output_dir / "SX_pressure_histogram_summary.csv"
    regression_path = output_dir / "SX_pressure_regression_summary.csv"
    histogram_summary.to_csv(histogram_path, index=False)
    regression_summary.to_csv(regression_path, index=False)
    return histogram_path, regression_path


def export_quantile_tables(
    quantile_curves: pd.DataFrame,
    quantile_summary: pd.DataFrame,
    method_comparison: pd.DataFrame,
    output_dir: str | Path,
    *,
    method: str,
    scope_suffix: str | None = None,
) -> tuple[Path, Path, Path]:
    """Export quantile curves, diagnostics, and basis-comparison source data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{scope_suffix}" if scope_suffix else ""
    curve_path = output_dir / f"SX_pressure_quantile_curves_{method}{suffix}.csv"
    summary_path = output_dir / f"SX_pressure_quantile_summary_{method}{suffix}.csv"
    comparison_path = output_dir / f"SX_pressure_quantile_method_comparison{suffix}.csv"
    quantile_curves.to_csv(curve_path, index=False)
    quantile_summary.to_csv(summary_path, index=False)
    method_comparison.to_csv(comparison_path, index=False)
    return curve_path, summary_path, comparison_path
