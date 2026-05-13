"""Figure helper functions shared by paper notebooks."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull, QhullError

from aims4pt.data_tools.rocks import get_TAS_rock_types
from aims4pt.statistic_tools.density_region_analysis import rock_type_check
from aims4pt.utils import normalize_column_names
from aims4pt.visualization.composition_plot import plot_glass_TAS_diagram
from paper.scripts.cache_helpers import _fast_df_fingerprint, cache_load, cache_save

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None



COL_REFERENCE__nb03_c03 = "0.70"
COL_ID__nb03_c03 = "C2"
COL_OOD__nb03_c03 = "C3"
symbol_reference__nb03_c03 = "s"
symbol_new__nb03_c03 = "o"
S_REFERENCE__nb03_c03 = 18
S_NEW__nb03_c03 = 90
ALPHA_REFERENCE__nb03_c03 = 0.22
ALPHA_NEW__nb03_c03 = 1.0
EDGE_NEW__nb03_c03 = "black"
LW_NEW__nb03_c03 = 0.7
FILL_FACE__nb03_c03 = "0.88"
FILL_ALPHA__nb03_c03 = 0.18
FILL_EDGE__nb03_c03 = "0.65"
FILL_LW__nb03_c03 = 0.8
AXIS_LABEL_SIZE__nb03_c03 = 16
TICK_LABEL_SIZE__nb03_c03 = 14
TITLE_SIZE__nb03_c03 = 15
LEGEND_FONT_SIZE__nb03_c03 = 14
LEGEND_MARKER_SIZE__nb03_c03 = 10
REFERENCE_MODEL_PRIORITY__nb03_c03 = [
    ("Ágreda-López et al., 2024", "Ágreda-López et al. (2024) calibration dataset"),
    ("Jorgenson et al., 2022", "Jorgenson et al. (2022) calibration dataset"),
]


def _match_reference_model__nb03_c03(model_dict, family_name, preferred_suffix="(cpx_only)"):
    preferred = [name for name in model_dict if family_name in name and preferred_suffix in name]
    if preferred:
        return model_dict[preferred[0]]
    matches = [name for name in model_dict if family_name in name]
    if matches:
        return model_dict[matches[0]]
    return None


def resolve_unified_reference_models__nb03_c03(P_model_dict, T_model_dict):
    """Resolve one reference model family shared by pressure and temperature panels."""
    for family_name, label in REFERENCE_MODEL_PRIORITY__nb03_c03:
        P_model = _match_reference_model__nb03_c03(P_model_dict, family_name)
        T_model = _match_reference_model__nb03_c03(T_model_dict, family_name)
        if P_model is not None and T_model is not None:
            return P_model, T_model, label
    raise KeyError(
        "Could not resolve a unified reference model from the current pipeline. "
        f"Available P models: {sorted(P_model_dict.keys())}; "
        f"Available T models: {sorted(T_model_dict.keys())}"
    )


def build_reference_pt_dataset__nb03_c03(P_model, T_model, tp_t_col, tp_p_col):
    """Build a reference P-T table from paired pressure and temperature models."""
    reference_pt = getattr(P_model, "X_cpx_all", None)
    if reference_pt is None or len(reference_pt) == 0:
        raise ValueError("Reference P model has no X_cpx_all calibration dataset.")
    reference_pt = reference_pt.copy()

    reference_t = getattr(T_model, "X_cpx_all", None)
    if reference_t is None or len(reference_t) == 0:
        raise ValueError("Reference T model has no X_cpx_all calibration dataset.")
    reference_t = reference_t.copy()

    if "Sample_ID" in reference_pt.columns and "Sample_ID" in reference_t.columns:
        reference_t_lookup = (
            reference_t[["Sample_ID", tp_t_col]]
            .dropna(subset=[tp_t_col])
            .drop_duplicates(subset="Sample_ID")
        )
        reference_pt = reference_pt.drop(columns=[tp_t_col], errors="ignore").merge(
            reference_t_lookup,
            on="Sample_ID",
            how="left",
        )
    elif tp_t_col not in reference_pt.columns and tp_t_col in reference_t.columns and len(reference_pt) == len(reference_t):
        reference_pt[tp_t_col] = reference_t[tp_t_col].to_numpy()

    missing_cols = [col for col in (tp_p_col, tp_t_col) if col not in reference_pt.columns]
    if missing_cols:
        raise KeyError(f"Reference P-T dataset is missing required columns: {missing_cols}")
    return reference_pt


def get_reference_liq_dataset__nb03_c03(P_model, T_model):
    """Return the liquid calibration dataset for the resolved reference model."""
    reference_liq = getattr(P_model, "X_liq_all", None)
    if reference_liq is None:
        reference_liq = getattr(T_model, "X_liq_all", None)
    if reference_liq is None or len(reference_liq) == 0:
        raise ValueError("Unified reference model has no liquid calibration dataset for the TAS panel.")
    return reference_liq.copy()


def panel_a_TAS__nb03_c03(ax, X_liq_reference, reference_label, reference_model, X_liq_new=None):
    plot_glass_TAS_diagram(
        X_liq_reference,
        axes=ax,
        color=COL_REFERENCE__nb03_c03,
        marker=symbol_reference__nb03_c03,
        label=reference_label,
        fill=True,
        facecolor=FILL_FACE__nb03_c03,
        edgecolor=FILL_EDGE__nb03_c03,
        linewidth=FILL_LW__nb03_c03,
        s=S_REFERENCE__nb03_c03,
        alpha=ALPHA_REFERENCE__nb03_c03,
    )
    if X_liq_new is not None and len(X_liq_new) > 0:
        liq_df = normalize_column_names(X_liq_new).copy()
        tas_types = liq_df.apply(get_TAS_rock_types, axis=1)
        tas_ok = np.array([rock_type_check(rt, reference_model.rock_types, report=False) for rt in tas_types])
        liq_df["Na2O + K2O"] = liq_df["Na2O"] + liq_df["K2O"]
        ax.scatter(
            liq_df["SiO2"].loc[tas_ok],
            liq_df["Na2O + K2O"].loc[tas_ok],
            c=COL_ID__nb03_c03,
            s=S_NEW__nb03_c03,
            alpha=ALPHA_NEW__nb03_c03,
            edgecolor=EDGE_NEW__nb03_c03,
            lw=LW_NEW__nb03_c03,
            label="Compiled / experimental dataset (in-dist)",
            zorder=5,
            marker=symbol_new__nb03_c03,
        )
        ax.scatter(
            liq_df["SiO2"].loc[~tas_ok],
            liq_df["Na2O + K2O"].loc[~tas_ok],
            c=COL_OOD__nb03_c03,
            s=S_NEW__nb03_c03,
            alpha=ALPHA_NEW__nb03_c03,
            edgecolor=EDGE_NEW__nb03_c03,
            lw=LW_NEW__nb03_c03,
            label="Compiled / experimental dataset (OOD)",
            zorder=6,
            marker=symbol_new__nb03_c03,
        )
    ax.set_title("(a)", loc="left", fontsize=TITLE_SIZE__nb03_c03)
    ax.set_xlabel("SiO$_2$ (wt.%)", fontsize=AXIS_LABEL_SIZE__nb03_c03)
    ax.set_ylabel("Na$_2$O + K$_2$O (wt.%)", fontsize=AXIS_LABEL_SIZE__nb03_c03)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE__nb03_c03)
    return ax


def panel_b_PT__nb03_c03(ax, X_reference, reference_label, X_cpx_new, X_liq_new, P_model, T_model, tp_p_col, tp_t_col):
    if tp_p_col not in X_reference.columns or tp_t_col not in X_reference.columns:
        raise KeyError(f"X_reference must contain '{tp_p_col}' and '{tp_t_col}' columns for panel_b_PT.")
    P_ref = X_reference[tp_p_col].to_numpy(float)
    T_ref = X_reference[tp_t_col].to_numpy(float)
    ref_mask = np.isfinite(P_ref) & np.isfinite(T_ref)
    P_ref = P_ref[ref_mask]
    T_ref = T_ref[ref_mask]
    ax.scatter(
        P_ref,
        T_ref,
        c=COL_REFERENCE__nb03_c03,
        s=S_REFERENCE__nb03_c03,
        alpha=ALPHA_REFERENCE__nb03_c03,
        edgecolor="none",
        label=reference_label,
        marker=symbol_reference__nb03_c03,
        zorder=1,
    )
    Pmin, Pmax = np.nanmin(P_ref), np.nanmax(P_ref)
    Tmin, Tmax = np.nanmin(T_ref), np.nanmax(T_ref)
    ax.add_patch(
        Rectangle(
            (Pmin, Tmin),
            Pmax - Pmin,
            Tmax - Tmin,
            facecolor=FILL_FACE__nb03_c03,
            edgecolor=FILL_EDGE__nb03_c03,
            lw=FILL_LW__nb03_c03,
            alpha=FILL_ALPHA__nb03_c03,
            zorder=0,
        )
    )
    P_pred = np.asarray(P_model.predict(X_cpx_new, X_liq_new)).ravel()
    T_pred = np.asarray(T_model.predict(X_cpx_new, X_liq_new)).ravel()
    in_range = (P_pred >= Pmin) & (P_pred <= Pmax) & (T_pred >= Tmin) & (T_pred <= Tmax)
    colors = np.where(in_range, COL_ID__nb03_c03, COL_OOD__nb03_c03)
    ax.scatter(
        P_pred,
        T_pred,
        c=colors,
        s=S_NEW__nb03_c03,
        alpha=ALPHA_NEW__nb03_c03,
        edgecolor=EDGE_NEW__nb03_c03,
        lw=LW_NEW__nb03_c03,
        zorder=5,
        marker=symbol_new__nb03_c03,
    )
    ax.set_xlabel("P (kbar)", fontsize=AXIS_LABEL_SIZE__nb03_c03)
    ax.set_ylabel("T (°C)", fontsize=AXIS_LABEL_SIZE__nb03_c03)
    ax.set_title("(b)", loc="left", fontsize=TITLE_SIZE__nb03_c03)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE__nb03_c03)
    ax.set_xlim(0, 17.5)
    ax.set_ylim(800, 1600)
    return ax


def plot_two_panels_example__nb03_c03(
    P_model,
    T_model,
    reference_liq_df,
    reference_pt_df,
    reference_label,
    sample_cpx_input,
    sample_liq_input,
    tp_p_col,
    tp_t_col,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=200)
    panel_a_TAS__nb03_c03(
        ax=axes[0],
        X_liq_reference=reference_liq_df,
        reference_label=reference_label,
        reference_model=P_model,
        X_liq_new=sample_liq_input,
    )
    panel_b_PT__nb03_c03(
        ax=axes[1],
        X_reference=reference_pt_df,
        reference_label=reference_label,
        X_cpx_new=sample_cpx_input,
        X_liq_new=sample_liq_input,
        P_model=P_model,
        T_model=T_model,
        tp_p_col=tp_p_col,
        tp_t_col=tp_t_col,
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=symbol_reference__nb03_c03,
            color="w",
            label=reference_label,
            markerfacecolor=COL_REFERENCE__nb03_c03,
            markersize=LEGEND_MARKER_SIZE__nb03_c03,
            markeredgecolor="none",
            alpha=0.8,
        ),
        plt.Line2D(
            [0],
            [0],
            marker=symbol_new__nb03_c03,
            color="w",
            label="Compiled / experimental dataset (in-dist)",
            markerfacecolor=COL_ID__nb03_c03,
            markersize=LEGEND_MARKER_SIZE__nb03_c03,
            markeredgecolor=EDGE_NEW__nb03_c03,
            markeredgewidth=LW_NEW__nb03_c03,
        ),
        plt.Line2D(
            [0],
            [0],
            marker=symbol_new__nb03_c03,
            color="w",
            label="Compiled / experimental dataset (OOD)",
            markerfacecolor=COL_OOD__nb03_c03,
            markersize=LEGEND_MARKER_SIZE__nb03_c03,
            markeredgecolor=EDGE_NEW__nb03_c03,
            markeredgewidth=LW_NEW__nb03_c03,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE__nb03_c03,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelspacing=0.4,
    )
    plt.subplots_adjust(top=0.82, wspace=0.28)
    return fig, axes


# Helpers extracted from 00_independent_dataset.ipynb cell 16.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aims4pt.visualization.composition_plot import plot_glass_TAS_diagram
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
required_base__nb00_c16 = ["unseen_experiments_df", "cpx_names", "liq_names", "P_col", "T_col"]
missing_base__nb00_c16 = [name for name in required_base__nb00_c16 if name not in globals()]
required_split__nb00_c16 = ["meta_unseen", "cpx_unseen", "liq_unseen", "training_unseen_id", "testing_unseen_id"]
reference_label__nb00_c16 = "Ágreda-López et al. (2024)"

def _get_agreda_reference_model__nb00_c16(T_P):
    model_dict_name = f"{T_P}_model_dict"
    target_name = "Ágreda-López et al., 2024 (cpx_only)"
    if model_dict_name in globals() and target_name in globals()[model_dict_name]:
        return globals()[model_dict_name][target_name]
    from aims4pt.model_tools.Agreda2024 import Agreda2024
    return Agreda2024(T_P=T_P, cpx_only=True)



# Helpers extracted from 00_independent_dataset.ipynb cell 18.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aims4pt.visualization.composition_plot import plot_glass_TAS_diagram
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
required_base__nb00_c18 = ["unseen_experiments_df", "cpx_names", "liq_names", "P_col", "T_col"]
missing_base__nb00_c18 = [name for name in required_base__nb00_c18 if name not in globals()]
required_split__nb00_c18 = ["meta_unseen", "cpx_unseen", "liq_unseen", "training_unseen_id", "testing_unseen_id"]
reference_label__nb00_c18 = "Ágreda-López et al., 2024"

def _get_agreda_reference_model__nb00_c18(T_P):
    existing_model_name = f"agreda_{T_P}_model"
    if existing_model_name in globals():
        return globals()[existing_model_name]

    model_dict_name = f"{T_P}_model_dict"
    target_name = "Ágreda-López et al., 2024 (cpx_only)"
    if model_dict_name in globals() and target_name in globals()[model_dict_name]:
        return globals()[model_dict_name][target_name]

    if "demo_model" in globals() and "demo_model_name" in globals():
        for name, model in zip(demo_model_name, demo_model):
            is_agreda = "Ágreda-López et al., 2024" in name
            is_matching_target = getattr(model, "T_P", T_P) == T_P and getattr(model, "cpx_only", True)
            if is_agreda and is_matching_target:
                return model

    from aims4pt.model_tools.Agreda2024 import Agreda2024
    return Agreda2024(T_P=T_P, cpx_only=True)

def _valid_xy__nb00_c18(df, x_col, y_col):
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]

def _set_pair_limits__nb00_c18(ax, x_series, y_series, pad_frac=0.05):
    x_all = pd.concat(x_series).dropna()
    y_all = pd.concat(y_series).dropna()
    if x_all.empty or y_all.empty:
        return

    xmin, xmax = x_all.min(), x_all.max()
    ymin, ymax = y_all.min(), y_all.max()
    xpad = (xmax - xmin) * pad_frac if xmax > xmin else 0.5
    ypad = (ymax - ymin) * pad_frac if ymax > ymin else 0.5
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)

def _panel_label__nb00_c18(ax, label):
    ax.text(
        0.01, 0.98, label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=14,
        fontweight="bold",
    )



# Helpers extracted from 00_independent_dataset.ipynb cell 24.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, QhullError
from aims4pt.visualization.composition_plot import plot_glass_TAS_diagram
LARGE_TH__nb00_c24 = 1500
MED_TH__nb00_c24 = 500
TAS_SIO2_COL__nb00_c24 = "SiO2_liq"
TAS_NA2O_COL__nb00_c24 = "Na2O_liq"
TAS_K2O_COL__nb00_c24  = "K2O_liq"
TP_T_COL__nb00_c24 = "T_C"
TP_P_COL__nb00_c24 = "P_kbar"
TAS_XLIM__nb00_c24 = (35, 82)
TAS_YLIM__nb00_c24 = (0, 18.5)
TP_XLIM__nb00_c24  = (650, 1750)
TP_YLIM__nb00_c24  = (-0.5, 42)
MARKERS_CYCLE__nb00_c24 = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
COLOR_CYCLE__nb00_c24 = plt.cm.tab10.colors
COLOR_CYCLE__nb00_c24 = COLOR_CYCLE__nb00_c24[0:8] + COLOR_CYCLE__nb00_c24[9:10]

def safe_numeric__nb00_c24(arr):
    return np.asarray(arr, dtype=float)

def get_n_liq__nb00_c24(model):
    X = getattr(model, 'X_liq_all', None)
    return 0 if X is None else len(X)

def build_global_color_map__nb00_c24(model_names, color_cycle):
    if len(model_names) > len(color_cycle):
        raise ValueError("tab10 has only 10 colors; too many models.")
    return {name: color_cycle[i] for i, name in enumerate(model_names)}

def make_group_style__nb00_c24(names, global_color_map):
    style = {}
    for i, name in enumerate(names):
        style[name] = dict(
            color=global_color_map[name],
            marker=MARKERS_CYCLE__nb00_c24[i % len(MARKERS_CYCLE__nb00_c24)]
        )
    return style

def plot_tas_group__nb00_c24(ax, model_names, name2model, name2style, scatter_style):
    template = None
    for n in model_names:
        X = getattr(name2model[n], 'X_liq_all', None)
        if X is not None:
            template = X.iloc[0:0].copy()
            break

    if template is None:
        return

    plot_glass_TAS_diagram(
        X_liq=template,
        model="LeMaitreCombined",
        axes=ax,
        color="none",
        plot_alkaline_boundary=True,
        linewidth=0.5,
    )

    for n in model_names:
        X = getattr(name2model[n], 'X_liq_all', None)
        if X is None:
            continue

        Si = safe_numeric__nb00_c24(X[TAS_SIO2_COL__nb00_c24])
        Na = safe_numeric__nb00_c24(X[TAS_NA2O_COL__nb00_c24])
        K  = safe_numeric__nb00_c24(X[TAS_K2O_COL__nb00_c24])
        m = np.isfinite(Si) & np.isfinite(Na) & np.isfinite(K)

        st = name2style[n]
        ax.scatter(
            Si[m], Na[m] + K[m],
            c=[st["color"]],
            marker=st["marker"],
            **scatter_style
        )

    ax.set_xlim(*TAS_XLIM__nb00_c24)
    ax.set_ylim(*TAS_YLIM__nb00_c24)

def add_panel_labels__nb00_c24(
    axes,
    labels=None,
    *,
    x=0.02,
    y=0.98,
    fontsize=12,
    fontweight="bold",
    va="top",
    ha="left",
    bbox=True
):
    """
    Add panel labels like (a), (b), ... to a grid of axes.

    Parameters
    ----------
    axes : array-like of matplotlib.axes.Axes
        e.g., the 2D array returned by plt.subplots().
    labels : list[str] or None
        If None, auto-generate (a), (b), ... by row-major order.
    x, y : float
        Position in axes coordinates (0-1). Default is near top-left.
    bbox : bool
        If True, draw a white background box to improve readability.
    """
    ax_list = [ax for row in np.atleast_2d(axes) for ax in row]
    n = len(ax_list)

    if labels is None:
        letters = [chr(ord("a") + i) for i in range(n)]
        labels = [f"({c})" for c in letters]
    if len(labels) != n:
        raise ValueError(f"labels length ({len(labels)}) != number of axes ({n}).")

    bbox_kw = None
    if bbox:
        bbox_kw = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5)

    for ax, lab in zip(ax_list, labels):
        ax.text(
            x, y, lab,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight=fontweight,
            va=va,
            ha=ha,
            bbox=bbox_kw,
            zorder=1000
        )

def plot_tp_scatter_hull__nb00_c24(ax, model_names, name2model, name2style, scatter_style):
    for n in model_names:
        X = getattr(name2model[n], 'X_cpx_all', None)
        if X is None:
            continue

        T = safe_numeric__nb00_c24(X[TP_T_COL__nb00_c24])
        P = safe_numeric__nb00_c24(X[TP_P_COL__nb00_c24])
        m = np.isfinite(T) & np.isfinite(P)
        T, P = T[m], P[m]

        st = name2style[n]
        ax.scatter(T, P, c=[st["color"]], marker=st["marker"], **scatter_style)

        pts = np.unique(np.column_stack([T, P]), axis=0)
        if pts.shape[0] < 3:
            continue

        try:
            hull = ConvexHull(pts)
            hp = pts[hull.vertices]
            hp = np.vstack([hp, hp[0]])
            ax.plot(hp[:, 0], hp[:, 1],
                    color=st["color"], lw=1.2, ls="--", alpha=0.9)
        except QhullError:
            pass

    ax.set_xlim(*TP_XLIM__nb00_c24)
    ax.set_ylim(*TP_YLIM__nb00_c24)

def plot_tp_binned_median_trend__nb00_c24(
    ax, model_names, name2model, name2style,
    dp, min_per_bin,
    *,
    line_lw=1,
    marker_size=7.5,
    marker_edge_w=1.0
):
    for n in model_names:
        X = getattr(name2model[n], 'X_cpx_all', None)
        if X is None:
            continue

        T = safe_numeric__nb00_c24(X[TP_T_COL__nb00_c24])
        P = safe_numeric__nb00_c24(X[TP_P_COL__nb00_c24])
        m = np.isfinite(T) & np.isfinite(P)
        T, P = T[m], P[m]

        if len(T) < 5:
            continue

        st = name2style[n]
        c = st["color"]

        # ====== KEY CHANGE HERE ======
        # Pressure bins start from 0 kbar
        pmax = np.ceil(P.max() / dp) * dp
        edges = np.arange(0.0, pmax + dp, dp)
        centers = edges[:-1] + dp / 2
        # =============================

        P_pos, T_median = [], []
        for lo, hi, pc in zip(edges[:-1], edges[1:], centers):
            idx = (P >= lo) & (P < hi) if hi < edges[-1] else (P >= lo)
            Ti = T[idx]
            if Ti.size >= min_per_bin:
                P_pos.append(pc)
                T_median.append(np.median(Ti))

        if len(T_median) < 2:
            continue

        ax.plot(
            T_median, P_pos,
            color="k",
            lw=line_lw,
            marker='h',
            markersize=marker_size,
            markerfacecolor=c,          # model color fill
            markeredgecolor='black',    # black outline
            markeredgewidth=marker_edge_w,
            zorder=10
        )

def add_legend_fixed_ul__nb00_c24(ax, names, name2style, name2n):
    handles, labels = [], []
    for n in names:
        st = name2style[n]
        handles.append(
            Line2D([0], [0],
                   marker=st["marker"], linestyle='None',
                   markerfacecolor=st["color"],
                   markeredgecolor='none',
                   markersize=7)
        )
        labels.append(f"{n} (N={name2n[n]})")

    ax.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.98),
        bbox_transform=ax.transAxes,
        frameon=False,
        fontsize=9,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelspacing=0.4
    )



# Helpers extracted from 00_independent_dataset.ipynb cell 27.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
required_globals__nb00_c27 = [
    "georoc_df_cleaned", "demo_model", "demo_model_name", "LARGE_TH", "MED_TH",
    "MARKERS_CYCLE", "COLOR_CYCLE", "safe_numeric__nb00_c24", "get_n_liq__nb00_c24",
    "build_global_color_map__nb00_c24", "make_group_style__nb00_c24", "add_panel_labels__nb00_c24",
]
missing__nb00_c27 = [name for name in required_globals__nb00_c27 if name not in globals()]
AXIS_LABEL_SIZE__nb00_c27 = 15
TICK_LABEL_SIZE__nb00_c27 = 14
TITLE_SIZE__nb00_c27 = 16
LEGEND_FONT_SIZE__nb00_c27 = 11
LEGEND_MARKER_SIZE__nb00_c27 = 7
GEOROC_LIMIT_PERCENTILES__nb00_c27 = (1.0, 99.0)
PAIR_LIMITS_OVERRIDE__nb00_c27 = {
    ("CaO_cpx", "Na2O_cpx"): None,
    ("Al2O3_cpx", "MgO_cpx"): None,
}
cpx_pairs__nb00_c27 = [
    ("CaO_cpx", "Na2O_cpx", r"$\mathrm{CaO}$ (wt%)", r"$\mathrm{Na_2O}$ (wt%)"),
    ("Al2O3_cpx", "MgO_cpx", r"$\mathrm{Al_2O_3}$ (wt%)", r"$\mathrm{MgO}$ (wt%)"),
]
GEOROC_STYLE__nb00_c27 = dict(color="0.88", s=10, alpha=1, linewidths=0, marker="o", zorder=0)
GEOROC_LEGEND_STYLE__nb00_c27 = dict(color="0.88", alpha=1, markersize=LEGEND_MARKER_SIZE__nb00_c27 - 2)
MODEL_SCATTER_STYLE__nb00_c27 = dict(s=18, alpha=0.9, linewidths=0, zorder=3)
CPX_COLUMN_ALIASES__nb00_c27 = {
    "FeOt_cpx": ("FeO_cpx",),
    "FeO_cpx": ("FeOt_cpx",),
}

def get_cpx_series__nb00_c27(df, col):
    if col in df.columns:
        return df[col]
    for alias in CPX_COLUMN_ALIASES__nb00_c27.get(col, ()):
        if alias in df.columns:
            return df[alias]
    return pd.Series(np.nan, index=df.index, dtype="float64")

def get_valid_xy__nb00_c27(df, x_col, y_col):
    x = safe_numeric__nb00_c24(get_cpx_series__nb00_c27(df, x_col))
    y = safe_numeric__nb00_c24(get_cpx_series__nb00_c27(df, y_col))
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]

def get_pair_limits__nb00_c27(
    name2model,
    georoc_df_cleaned,
    x_col,
    y_col,
    *,
    pad_frac=0.05,
    georoc_percentiles=GEOROC_LIMIT_PERCENTILES__nb00_c27,
    override_limits=None,
):
    if override_limits is not None:
        return override_limits

    xs, ys = [], []

    gx, gy = get_valid_xy__nb00_c27(georoc_df_cleaned, x_col, y_col)
    if gx.size:
        lo, hi = georoc_percentiles
        gx_lo, gx_hi = np.nanpercentile(gx, [lo, hi])
        gy_lo, gy_hi = np.nanpercentile(gy, [lo, hi])
        gmask = (gx >= gx_lo) & (gx <= gx_hi) & (gy >= gy_lo) & (gy <= gy_hi)
        gx_clip = gx[gmask]
        gy_clip = gy[gmask]
        if gx_clip.size:
            xs.append(gx_clip)
            ys.append(gy_clip)
        else:
            xs.append(gx)
            ys.append(gy)

    for model in name2model.values():
        X = getattr(model, "X_cpx_all", None)
        if X is None:
            continue
        xi, yi = get_valid_xy__nb00_c27(X, x_col, y_col)
        if xi.size:
            xs.append(xi)
            ys.append(yi)

    if not xs:
        return None

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    xmin, xmax = x_all.min(), x_all.max()
    ymin, ymax = y_all.min(), y_all.max()

    xpad = (xmax - xmin) * pad_frac if xmax > xmin else 0.5
    ypad = (ymax - ymin) * pad_frac if ymax > ymin else 0.5
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad

def plot_cpx_pair__nb00_c27(ax, model_names, name2model, name2style, georoc_df_cleaned, x_col, y_col):
    gx, gy = get_valid_xy__nb00_c27(georoc_df_cleaned, x_col, y_col)
    ax.scatter(gx, gy, **GEOROC_STYLE__nb00_c27)

    for name in model_names:
        X = getattr(name2model[name], "X_cpx_all", None)
        if X is None:
            continue

        xi, yi = get_valid_xy__nb00_c27(X, x_col, y_col)
        if xi.size == 0:
            continue

        style = name2style[name]
        ax.scatter(
            xi,
            yi,
            c=[style["color"]],
            marker=style["marker"],
            **MODEL_SCATTER_STYLE__nb00_c27,
        )

    return gx.size

def add_cpx_legend_fixed_ul__nb00_c27(
    ax,
    names,
    name2style,
    name2n,
    georoc_n=None,
    include_georoc=False,
    loc="upper left",
    bbox_to_anchor=(0.08, 0.98),
):
    handles, labels = [], []

    if include_georoc:
        handles.append(
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=GEOROC_LEGEND_STYLE__nb00_c27["color"],
                   markeredgecolor="none", alpha=GEOROC_LEGEND_STYLE__nb00_c27["alpha"], markersize=GEOROC_LEGEND_STYLE__nb00_c27["markersize"])
        )
        labels.append(f"GEOROC (N={georoc_n})")

    for name in names:
        style = name2style[name]
        handles.append(
            Line2D([0], [0], marker=style["marker"], linestyle="None",
                   markerfacecolor=style["color"], markeredgecolor="none",
                   markersize=LEGEND_MARKER_SIZE__nb00_c27)
        )
        labels.append(f"{name} (N={name2n[name]})")

    ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE__nb00_c27,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelspacing=0.4,
    )



# Helpers extracted from 00_independent_dataset.ipynb cell 28.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
required_globals__nb00_c28 = [
    "georoc_df_cleaned", "demo_model", "demo_model_name", "LARGE_TH", "MED_TH",
    "MARKERS_CYCLE", "COLOR_CYCLE", "safe_numeric__nb00_c24", "get_n_liq__nb00_c24",
    "build_global_color_map__nb00_c24", "make_group_style__nb00_c24", "add_panel_labels__nb00_c24",
]
missing__nb00_c28 = [name for name in required_globals__nb00_c28 if name not in globals()]
AXIS_LABEL_SIZE__nb00_c28 = 15
TICK_LABEL_SIZE__nb00_c28 = 14
TITLE_SIZE__nb00_c28 = 16
LEGEND_FONT_SIZE__nb00_c28 = 11
LEGEND_MARKER_SIZE__nb00_c28 = 7
GEOROC_LIMIT_PERCENTILES__nb00_c28 = (1.0, 99.0)
PAIR_LIMITS_OVERRIDE__nb00_c28 = {
    ("SiO2_cpx", "TiO2_cpx"): None,
    ("FeO_cpx", "Cr2O3_cpx"): None,
}
cpx_pairs__nb00_c28 = [
    ("SiO2_cpx", "TiO2_cpx", r"$\mathrm{SiO_2}$ (wt%)", r"$\mathrm{TiO_2}$ (wt%)"),
    ("FeO_cpx", "Cr2O3_cpx", r"$\mathrm{FeO}$ (wt%)", r"$\mathrm{Cr_2O_3}$ (wt%)"),
]
GEOROC_STYLE__nb00_c28 = dict(color="0.88", s=10, alpha=1, linewidths=0, marker="o", zorder=0)
GEOROC_LEGEND_STYLE__nb00_c28 = dict(color="0.88", alpha=1, markersize=LEGEND_MARKER_SIZE__nb00_c28 - 2)
MODEL_SCATTER_STYLE__nb00_c28 = dict(s=18, alpha=0.9, linewidths=0, zorder=3)
CPX_COLUMN_ALIASES__nb00_c28 = {
    "FeOt_cpx": ("FeO_cpx",),
    "FeO_cpx": ("FeOt_cpx",),
}

def get_cpx_series__nb00_c28(df, col):
    if col in df.columns:
        return df[col]
    for alias in CPX_COLUMN_ALIASES__nb00_c28.get(col, ()):
        if alias in df.columns:
            return df[alias]
    return pd.Series(np.nan, index=df.index, dtype="float64")

def get_valid_xy__nb00_c28(df, x_col, y_col):
    x = safe_numeric__nb00_c24(get_cpx_series__nb00_c28(df, x_col))
    y = safe_numeric__nb00_c24(get_cpx_series__nb00_c28(df, y_col))
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]

def get_pair_limits__nb00_c28(
    name2model,
    georoc_df_cleaned,
    x_col,
    y_col,
    *,
    pad_frac=0.05,
    georoc_percentiles=GEOROC_LIMIT_PERCENTILES__nb00_c28,
    override_limits=None,
):
    if override_limits is not None:
        return override_limits

    xs, ys = [], []

    gx, gy = get_valid_xy__nb00_c28(georoc_df_cleaned, x_col, y_col)
    if gx.size:
        lo, hi = georoc_percentiles
        gx_lo, gx_hi = np.nanpercentile(gx, [lo, hi])
        gy_lo, gy_hi = np.nanpercentile(gy, [lo, hi])
        gmask = (gx >= gx_lo) & (gx <= gx_hi) & (gy >= gy_lo) & (gy <= gy_hi)
        gx_clip = gx[gmask]
        gy_clip = gy[gmask]
        if gx_clip.size:
            xs.append(gx_clip)
            ys.append(gy_clip)
        else:
            xs.append(gx)
            ys.append(gy)

    for model in name2model.values():
        X = getattr(model, "X_cpx_all", None)
        if X is None:
            continue
        xi, yi = get_valid_xy__nb00_c28(X, x_col, y_col)
        if xi.size:
            xs.append(xi)
            ys.append(yi)

    if not xs:
        return None

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    xmin, xmax = x_all.min(), x_all.max()
    ymin, ymax = y_all.min(), y_all.max()

    xpad = (xmax - xmin) * pad_frac if xmax > xmin else 0.5
    ypad = (ymax - ymin) * pad_frac if ymax > ymin else 0.5
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad

def plot_cpx_pair__nb00_c28(ax, model_names, name2model, name2style, georoc_df_cleaned, x_col, y_col):
    gx, gy = get_valid_xy__nb00_c28(georoc_df_cleaned, x_col, y_col)
    ax.scatter(gx, gy, **GEOROC_STYLE__nb00_c28)

    for name in model_names:
        X = getattr(name2model[name], "X_cpx_all", None)
        if X is None:
            continue

        xi, yi = get_valid_xy__nb00_c28(X, x_col, y_col)
        if xi.size == 0:
            continue

        style = name2style[name]
        ax.scatter(
            xi,
            yi,
            c=[style["color"]],
            marker=style["marker"],
            **MODEL_SCATTER_STYLE__nb00_c28,
        )

    return gx.size

def add_cpx_legend_fixed_ul__nb00_c28(
    ax,
    names,
    name2style,
    name2n,
    georoc_n=None,
    include_georoc=False,
    loc="upper left",
    bbox_to_anchor=(0.08, 0.98),
):
    handles, labels = [], []

    if include_georoc:
        handles.append(
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=GEOROC_LEGEND_STYLE__nb00_c28["color"],
                   markeredgecolor="none", alpha=GEOROC_LEGEND_STYLE__nb00_c28["alpha"], markersize=GEOROC_LEGEND_STYLE__nb00_c28["markersize"])
        )
        labels.append(f"GEOROC (N={georoc_n})")

    for name in names:
        style = name2style[name]
        handles.append(
            Line2D([0], [0], marker=style["marker"], linestyle="None",
                   markerfacecolor=style["color"], markeredgecolor="none",
                   markersize=LEGEND_MARKER_SIZE__nb00_c28)
        )
        labels.append(f"{name} (N={name2n[name]})")

    ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE__nb00_c28,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelspacing=0.4,
    )



# Helpers extracted from 00_independent_dataset.ipynb cell 29.
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
required_globals__nb00_c29 = ["demo_model", "demo_model_name"]
missing__nb00_c29 = [name for name in required_globals__nb00_c29 if name not in globals()]
hist_cpx_names__nb00_c29 = [
    "SiO2_cpx", "TiO2_cpx", "Al2O3_cpx", "FeOt_cpx", "MnO_cpx",
    "MgO_cpx", "CaO_cpx", "Na2O_cpx", "K2O_cpx", "Cr2O3_cpx",
]
CPX_COLUMN_ALIASES__nb00_c29 = {
    "FeOt_cpx": ("FeO_cpx",),
    "FeO_cpx": ("FeOt_cpx",),
}

def get_existing_cpx_series__nb00_c29(df, col):
    if col in df.columns:
        return df[col]
    for alias in CPX_COLUMN_ALIASES__nb00_c29.get(col, ()):
        if alias in df.columns:
            return df[alias]
    return None



# Helpers extracted from 03_reproduce_AIMS4PT_cpx_figures.ipynb cell 2.
from pathlib import Path

def _get_agreda_reference_model__nb03_c02(T_P):
    model_dict_name = f"{T_P}_model_dict"
    target_name = "Ágreda-López et al., 2024 (cpx_only)"
    if model_dict_name in globals() and target_name in globals()[model_dict_name]:
        return globals()[model_dict_name][target_name]

    from aims4pt.model_tools.Agreda2024 import Agreda2024
    return Agreda2024(T_P=T_P, cpx_only=True)



# Helpers extracted from 03_reproduce_AIMS4PT_cpx_figures.ipynb cell 5.
import math
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from aims4pt.visualization.mode_related_plot import (
    plot_shap_summary_dot_colored,
)

def _clean_model_label__nb03_c05(model_name: str) -> str:
    """Return a short label for plotting (strip anything after '(')."""
    return model_name.split("(")[0].strip()

def _format_panel_title__nb03_c05(model_label: str, plot_kind: str) -> str:
    """
    Create a two-line title.
    First line: plot kind (only shown on first row if desired)
    Second line: model label
    """
    if plot_kind:
        return f"{plot_kind}\n{model_label}"
    return f"{model_label}"

def _add_panel_letter__nb03_c05(ax, letter: str, *, x=-0.08, y=1.04, fontsize=12):
    """Add panel letter like '(a)' at top-left of the axis."""
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
    )

def _safe_concat_cpx_liq__nb03_c05(X_cpx: pd.DataFrame, X_liq: pd.DataFrame | None) -> pd.DataFrame:
    """
    Concatenate cpx + liq features if liq exists; otherwise return cpx only.
    """
    if X_liq is None:
        return X_cpx
    if isinstance(X_liq, pd.DataFrame) and X_liq.shape[1] == 0:
        return X_cpx
    return pd.concat([X_cpx, X_liq], axis=1)

def plot_shap_summary_cpx_only_vs_cpx_liq__nb03_c05(
    model_list,
    *,
    top_n: int = 10,
    fig_width_per_col: float = 4.0,
    row_height: float = 4.0,
):
    """
    Plot SHAP summary plots in a 4-column layout:
    - Left 2 columns: cpx-only models
    - Right 2 columns: cpx–liq models

    Panel letters (a, b, c, ...) are assigned in reading order:
    left-to-right, top-to-bottom across the whole figure.

    Parameters
    ----------
    model_list : list
        List of trained model objects. Each model is expected to have:
        - cpx_only (bool)  [used for grouping]
        - X_cpx_training (pd.DataFrame)
        - X_liq_training (pd.DataFrame or None)  [optional for cpx-only]
        - shap_df
        - model_name
        - T_P  ('P' or 'T')
    top_n : int
        Number of top features in SHAP summary.
    test_size : float
        Subsample proportion for SHAP plotting.
    random_state : int
        Random seed.
    fig_width_per_col : float
        Width scale per column.
    row_height : float
        Height scale per row.

    Returns
    -------
    (fig, axes)
        matplotlib figure and axes array (nrows x 4).
    """
    # ---- split models into two groups ----
    cpx_only_models = [m for m in model_list if getattr(m, "cpx_only", False)]
    cpx_liq_models  = [m for m in model_list if not getattr(m, "cpx_only", False)]

    # ---- fixed 4 columns (2 + 2) ----
    ncols = 4
    nrows_left  = math.ceil(len(cpx_only_models) / 2) if len(cpx_only_models) > 0 else 0
    nrows_right = math.ceil(len(cpx_liq_models) / 2) if len(cpx_liq_models) > 0 else 0
    nrows = max(1, nrows_left, nrows_right)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width_per_col * ncols, row_height * nrows),
        dpi=150,
        squeeze=False,
    )

    shap_cmap = plt.cm.coolwarm

    # ---- add block titles (top) ----
    # Use figure coordinates; y a bit above axes area
    fig.text(0.25, 0.98, "Clinopyroxene-only models", ha="center", va="top", fontsize=16, weight="bold")
    fig.text(0.75, 0.98, "Clinopyroxene–liquid models", ha="center", va="top", fontsize=16, weight="bold")

    # ---- helper to plot one model into a given axis ----
    def _plot_one_model(ax, model, show_kind_title: bool):
        model_label = _clean_model_label__nb03_c05(model.model_name)

        X_cpx_train = model.X_cpx_training
        X_liq_train = getattr(model, "X_liq_training", None)

        shap_df = model.shap_df
        sampled_idx = shap_df.index
        X_cpx_sampled = X_cpx_train.loc[sampled_idx]
        X_liq_sampled = None if X_liq_train is None else X_liq_train.loc[sampled_idx]
        X_sampled = _safe_concat_cpx_liq__nb03_c05(X_cpx_sampled, X_liq_sampled)

        plot_shap_summary_dot_colored(
            shap_df,
            X_sampled,
            top_n=top_n,
            ax=ax,
            cmap=shap_cmap,
            if_show_colorbar=False,
            units="(kbar)" if model.T_P == "P" else "(°C)",
        )

        ax.set_title(
            _format_panel_title__nb03_c05(model_label, ""),
            fontsize=12,
            fontweight="normal",
        )

    # ---- fill panels row-by-row ----
    # left block positions: (row, col 0-1)
    # right block positions: (row, col 2-3)
    # Within each block, index = row*2 + within_block_col
    for r in range(nrows):
        for within in range(2):
            # cpx-only
            idx_left = r * 2 + within
            ax_left = axes[r, within]
            if idx_left < len(cpx_only_models):
                _plot_one_model(ax_left, cpx_only_models[idx_left], show_kind_title=(r == 0))
            else:
                ax_left.axis("off")

            # cpx-liq
            idx_right = r * 2 + within
            ax_right = axes[r, 2 + within]
            if idx_right < len(cpx_liq_models):
                _plot_one_model(ax_right, cpx_liq_models[idx_right], show_kind_title=(r == 0))
            else:
                ax_right.axis("off")

    # ---- panel letters in reading order across all visible panels ----
    # Assign only to axes that are not off (has data).
    letters = list(string.ascii_lowercase)
    letter_i = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if not ax.has_data():
                continue
            if letter_i >= len(letters):
                # if you ever exceed 26 panels, extend to aa, ab... (unlikely here)
                letter = f"a{letters[letter_i - len(letters)]}"
            else:
                letter = letters[letter_i]
            _add_panel_letter__nb03_c05(ax, letter)
            letter_i += 1

    # ---- x-axis labels only on last row (for active axes) ----
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if not ax.has_data():
                continue
            if r != (nrows - 1):
                ax.set_xlabel("")

    # layout: leave a bit of top room for block titles
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.show()
    return fig, axes



# Helpers extracted from 03_reproduce_AIMS4PT_cpx_figures.ipynb cell 8.
import math
import re
import pickle
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

def _add_panel_labels__nb03_c08(
    axes,
    labels=None,
    *,
    x=0.02,
    y=0.96,
    **text_kwargs,
):
    """
    Add panel labels like (a), (b), (c)... to a list/array of axes.

    Notes
    -----
    - Labels are placed in axes coordinates (ax.transAxes).
    - If user passes transform in text_kwargs, it will be ignored to avoid
      duplicate transform error.
    """
    axes = np.asarray(axes).ravel()

    if labels is None:
        labels = [f"({chr(ord('a') + i)})" for i in range(len(axes))]
    if len(labels) != len(axes):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of axes ({len(axes)}).")

    # Default label style
    default_kwargs = dict(
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=50,
    )
    default_kwargs.update(text_kwargs)

    # Avoid passing transform twice
    default_kwargs.pop("transform", None)

    for ax, lab in zip(axes, labels):
        ax.text(
            x,
            y,
            lab,
            transform=ax.transAxes,
            **default_kwargs,
        )

def _draw_side_q1_mean_q3__nb03_c08(
    ax,
    x,
    y,
    *,
    x0=0.0,
    inverted_xaxis=True,
    band_color="0.45",
    band_alpha=0.22,      # darker
    mean_color="0.25",
    mean_lw=1.2,
    mean_marker="o",
    mean_ms=4.2,
    mean_mec="0.15",
    mean_mew=0.7,
    zorder=1,
):
    """
    Draw per-side (relative to vertical line x=x0) Q1-IQR band + mean line.
    "Left/right" are defined by what you SEE on the plot (respects invert_xaxis()).

    If inverted_xaxis=True:
        - display-left side corresponds to x > x0
        - display-right side corresponds to x < x0
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return

    x = x[m]
    y = y[m]

    if inverted_xaxis:
        m_left = x > x0
        m_right = x < x0
    else:
        m_left = x < x0
        m_right = x > x0

    xlim = ax.get_xlim()
    x_min = min(xlim)
    x_max = max(xlim)

    def _span_for_left():
        return (x0, x_max) if inverted_xaxis else (x_min, x0)

    def _span_for_right():
        return (x_min, x0) if inverted_xaxis else (x0, x_max)

    def _draw_one(mask, x_span):
        if mask.sum() < 3:
            return
        yy = y[mask]
        q1 = float(np.quantile(yy, 0.25))
        q3 = float(np.quantile(yy, 0.75))
        mu = float(np.mean(yy))

        xa, xb = x_span
        ax.fill_between(
            [xa, xb], [q1, q1], [q3, q3],
            color=band_color, alpha=band_alpha, linewidth=0,
            zorder=zorder
        )
        ax.hlines(mu, xa, xb, colors=mean_color, linewidth=mean_lw, zorder=zorder + 0.1)

    _draw_one(m_left, _span_for_left())
    _draw_one(m_right, _span_for_right())

def plot_ood_score_vs_deviation__nb03_c08(
    input_cpx,
    input_meta,
    model,
    input_liq=None,
    petrological_ood_marker=False,
    T_P_ood_marker=False,
    ax=None,
    cmap_name="cividis",
    norm_min=0.0,
    norm_max=10.0,
    s_base=55,
    s_overlay=60,
    lw_base=0.85,
    lw_overlay=0.95,
    ind_edge="black",
    ood_edge="red",
    # background stats band
    show_side_stats=True,
    stats_band_color="0.45",
    stats_band_alpha=0.22,
    stats_mean_color="0.25",
):
    model_name = getattr(model, "model_name", "Unnamed model")
    target_col = "P_kbar" if model.T_P == "P" else "T_C"

    if petrological_ood_marker and input_liq is None:
        raise ValueError("petrological_ood_marker=True requires input_liq.")
    if target_col not in input_meta.columns:
        raise ValueError(f"'{target_col}' not in input_meta.columns")

    color_col = "P_kbar" if "P_kbar" in input_meta.columns else target_col

    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=float(norm_min), vmax=float(norm_max), clip=True)

    cpx_key = _fast_df_fingerprint(input_cpx)
    liq_key = _fast_df_fingerprint(input_liq) if input_liq is not None else "none"
    meta_key = _fast_df_fingerprint(input_meta, cols=[target_col, color_col])

    cache_key = hashlib.md5(
        (str(model_name) + "|" + str(model.T_P) + "|" + cpx_key + "|" + liq_key + "|" + meta_key).encode("utf-8")
    ).hexdigest()

    cached = cache_load("ood_scatter", cache_key)

    if cached is None:
        predictions = model.predict(input_cpx, input_liq)

        ood_model = getattr(model, "OOD_detector", None)
        if ood_model is None:
            ood_mask = np.zeros(len(input_cpx), dtype=bool)
            ood_score = np.full(len(input_cpx), np.nan)
        else:
            ood_score = ood_model.score(input_cpx, input_liq)
            ood_mask = ood_model.is_ood(input_cpx, input_liq)

        deviation = (input_meta[target_col] - predictions).abs().to_numpy(float)

        tp_mask = np.zeros(len(input_cpx), bool)
        if T_P_ood_marker:
            y_min = getattr(model, "y_min_95", -np.inf)
            y_max = getattr(model, "y_max_95", np.inf)
            pred_arr = np.asarray(predictions, float)
            tp_mask = (pred_arr < y_min) | (pred_arr > y_max)

        tas_mask = np.zeros(len(input_cpx), bool)
        if petrological_ood_marker:
            from aims4pt.statistic_tools.density_region_analysis import rock_type_check
            tas_types = input_liq.apply(get_TAS_rock_types, axis=1)
            tas_ok = np.array([rock_type_check(rt, model.rock_types, report=False) for rt in tas_types])
            tas_mask = ~tas_ok

        cached = dict(
            ood_score=np.asarray(ood_score, float),
            ood_mask=np.asarray(ood_mask, bool),
            deviation=deviation,
            tp_mask=tp_mask,
            tas_mask=tas_mask,
            color_value=input_meta[color_col].to_numpy(float),
        )
        cache_save("ood_scatter", cache_key, cached)

    ood_score = cached["ood_score"]
    ood_mask = cached["ood_mask"]
    deviation = cached["deviation"]
    tp_mask = cached["tp_mask"] if T_P_ood_marker else np.zeros(len(input_cpx), bool)
    tas_mask = cached["tas_mask"] if petrological_ood_marker else np.zeros(len(input_cpx), bool)
    color_value = cached["color_value"]

    df = pd.DataFrame(
        dict(score=ood_score, deviation=deviation, color=color_value),
        index=input_meta.index
    ).dropna()

    idx = df.index
    m_ood = pd.Series(ood_mask, index=input_meta.index).reindex(idx).fillna(False).to_numpy(bool)
    m_tp = pd.Series(tp_mask, index=input_meta.index).reindex(idx).fillna(False).to_numpy(bool)
    m_tas = pd.Series(tas_mask, index=input_meta.index).reindex(idx).fillna(False).to_numpy(bool)

    excl = m_tp | m_tas
    m_ind = (~m_ood) & (~excl)
    m_oodf = m_ood & (~excl)

    ind_df = df.loc[m_ind]
    ood_df = df.loc[m_oodf]
    tas_df = df.loc[m_tas]
    tp_df = df.loc[m_tp]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    else:
        fig = ax.figure

    # Background stats band (behind points)
    if show_side_stats and len(df) >= 3:
        x_all = df["score"].to_numpy(float)
        if np.isfinite(x_all).any():
            xmin = np.nanmin(x_all)
            xmax = np.nanmax(x_all)
            pad = 0.03 * (xmax - xmin) if (xmax > xmin) else 1.0
            # Provisional xlim (inverted style will be applied later if you want)
            ax.set_xlim(xmax + pad, xmin - pad)

        _draw_side_q1_mean_q3__nb03_c08(
            ax,
            df["score"].to_numpy(float),
            df["deviation"].to_numpy(float),
            x0=0.0,
            inverted_xaxis=True,
            band_color=stats_band_color,
            band_alpha=stats_band_alpha,
            mean_color=stats_mean_color,
            mean_lw=1.2,
            mean_ms=4.2,
            zorder=1,
        )

    # Scatter points
    if len(ind_df) > 0:
        face = cmap(norm(ind_df["color"].to_numpy()))
        ax.scatter(
            ind_df["score"], ind_df["deviation"],
            s=s_base, marker="o",
            facecolors=face, edgecolors=ind_edge, linewidths=lw_base,
            alpha=0.75, zorder=3
        )

    if len(ood_df) > 0:
        face = cmap(norm(ood_df["color"].to_numpy()))
        ax.scatter(
            ood_df["score"], ood_df["deviation"],
            s=s_base, marker="s",
            facecolors=face, edgecolors=ood_edge, linewidths=lw_base,
            alpha=0.75, zorder=4
        )

    if petrological_ood_marker and len(tas_df) > 0:
        face = cmap(norm(tas_df["color"].to_numpy()))
        ax.scatter(
            tas_df["score"], tas_df["deviation"],
            s=s_overlay, marker="D",
            facecolors=face, edgecolors=ood_edge, linewidths=lw_overlay,
            alpha=0.85, zorder=7
        )

    if T_P_ood_marker and len(tp_df) > 0:
        face = cmap(norm(tp_df["color"].to_numpy()))
        ax.scatter(
            tp_df["score"], tp_df["deviation"],
            s=s_overlay, marker="^",
            facecolors=face, edgecolors=ood_edge, linewidths=lw_overlay,
            alpha=0.9, zorder=8
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, zorder=10)
    ax.axhline(float(model.uncertainty), color="0.35", linestyle="--", linewidth=1, zorder=10)

    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlabel("Signed distance to the boundary")
    ax.set_ylabel("Absolute deviation (kbar)")

    return fig, ax

def _fmt_num__nb03_c08(x: float) -> str:
    """Pretty formatting: integers as int; otherwise 1 decimal.
    Special case: values that round to 0.0 at 1 decimal -> "0".
    """
    if x is None:
        return "NA"

    x = float(x)

    if not np.isfinite(x):
        return "NA"

    if abs(x) < 5e-12:
        return "0"

    xr = round(x)
    if abs(x - xr) < 1e-9:
        return str(int(xr))

    if round(x, 1) == 0.0:
        return "0"

    return f"{x:.1f}"

def plot_ood_panels_generic__nb03_c08(
    models,
    test_cpx,
    test_meta,
    *,
    test_liq=None,
    ncols=2,
    petrological_ood_marker=False,
    T_P_ood_marker=False,
    fig_width=14,
    row_height=5.4,
    height_scale=1.08,
    cmap_name="cividis",
    norm_min=0.0,
    clip_to_p99=True,
    cbar_width=0.6,
    cbar_height=0.013,
    cbar_y=0.14,
    legend_anchor_y=0.15,
    tight_bottom=0.18,
    tight_top=0.96,
    # side stats appearance
    show_side_stats=True,
    stats_band_color="0.45",
    stats_band_alpha=0.22,
    stats_mean_color="0.25",
    # NEW: panel labels
    add_panel_labels__nb00_c24=True,
    panel_label_kwargs=None,
):
    models = [m for m in models if getattr(m, "X_cpx_training", None) is not None]
    n = len(models)
    nrows = math.ceil(n / ncols)

    # Colorbar normalization
    if clip_to_p99:
        pvals = pd.to_numeric(test_meta["P_kbar"], errors="coerce").dropna().to_numpy(float)
        p99 = float(np.quantile(pvals, 0.99))
        norm_max = p99
    else:
        norm_max = float(pd.to_numeric(test_meta["P_kbar"], errors="coerce").dropna().max())
    cbar_label = (
        f"True P (kbar), clipped to 99th percentile ({_fmt_num__nb03_c08(norm_max)} kbar)"
        if clip_to_p99 else "True P (kbar)"
    )

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, row_height * nrows * height_scale),
        dpi=200,
        constrained_layout=False
    )
    axes = np.array(axes).ravel()

    # Plot each panel
    for model, ax in zip(models, axes):
        plot_ood_score_vs_deviation__nb03_c08(
            test_cpx, test_meta, model,
            input_liq=test_liq,
            petrological_ood_marker=petrological_ood_marker,
            T_P_ood_marker=T_P_ood_marker,
            ax=ax,
            cmap_name=cmap_name,
            norm_min=norm_min,
            norm_max=norm_max,
            show_side_stats=show_side_stats,
            stats_band_color=stats_band_color,
            stats_band_alpha=stats_band_alpha,
            stats_mean_color=stats_mean_color,
        )

        # Title: use "(Range: a-b kbar)" style
        name = re.sub(r"\s*\(.*?\)\s*$", "", model.model_name)
        y_min = getattr(model, "y_min", None)
        y_max = getattr(model, "y_max", None)
        if y_min is not None and y_max is not None:
            if model.T_P == "P":
                name += f" (Range: {_fmt_num__nb03_c08(float(y_min))}-{_fmt_num__nb03_c08(float(y_max))} kbar)"
            else:
                name += f" (Range: {_fmt_num__nb03_c08(float(y_min))}-{_fmt_num__nb03_c08(float(y_max))} °C)"
        ax.set_title(name, fontsize=13)

    # Remove unused axes
    for ax in axes[n:]:
        fig.delaxes(ax)

    # Only keep last row x-axis labels; only keep first column y-axis labels
    for i, ax in enumerate(axes[:n]):
        row = i // ncols
        col = i % ncols
        if row < nrows - 1:
            ax.set_xlabel("")
        if col > 0:
            ax.set_ylabel("")
    if n % ncols == 1 and n >= 2:
        last_ax = axes[n - 2]
        last_ax.set_xlabel("Signed distance to the boundary")

    # Add panel labels (a)(b)(c)...
    if add_panel_labels__nb00_c24 and n > 0:
        if panel_label_kwargs is None:
            panel_label_kwargs = dict(fontsize=12, fontweight="bold")
        _add_panel_labels__nb03_c08(axes[:n], labels=None, x=0.02, y=0.96, **panel_label_kwargs)

    fig.tight_layout(rect=[0, tight_bottom, 1, tight_top])

    # Legend
    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="white", markeredgecolor="black", label="In-distribution"),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor="white", markeredgecolor="red", label="OOD (features)"),
        Line2D([0], [0], marker="D", linestyle="None",
               markerfacecolor="white", markeredgecolor="red", label="OOD (TAS)"),
        Line2D([0], [0], marker="^", linestyle="None",
               markerfacecolor="white", markeredgecolor="red", label=f"OOD (predicted {models[0].T_P})"),
        Line2D([0], [0], linestyle="--", color="0.35", label="Model uncertainty"),
    ]
    labels = [h.get_label() for h in handles]

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_anchor_y),
        ncol=len(handles),
        frameon=False,
    )

    # Colorbar
    sm = ScalarMappable(norm=Normalize(norm_min, norm_max), cmap=get_cmap(cmap_name))
    cax = fig.add_axes([(1 - cbar_width) / 2, cbar_y, cbar_width, cbar_height])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label)

    return fig, axes



# Helpers extracted from 03_reproduce_AIMS4PT_cpx_figures.ipynb cell 12.
import math
import re
import pickle
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

def _add_panel_labels__nb03_c12(
    axes,
    labels=None,
    *,
    x=0.02,
    y=0.96,
    **text_kwargs,
):
    """
    Add panel labels like (a), (b), (c)... to a list/array of axes.

    Notes
    -----
    - Uses axes coordinates (ax.transAxes).
    - If user passes transform in text_kwargs, it will be ignored to avoid
      duplicate transform error.
    """
    axes = np.asarray(axes).ravel()

    if labels is None:
        labels = [f"({chr(ord('a') + i)})" for i in range(len(axes))]
    if len(labels) != len(axes):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of axes ({len(axes)}).")

    default_kwargs = dict(
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=50,
    )
    default_kwargs.update(text_kwargs)

    # prevent "multiple values for transform"
    default_kwargs.pop("transform", None)

    for ax, lab in zip(axes, labels):
        ax.text(
            x,
            y,
            lab,
            transform=ax.transAxes,
            **default_kwargs,
        )

def _draw_side_q1_mean_q3__nb03_c12(
    ax,
    x,
    y,
    *,
    x0=0.0,
    inverted_xaxis=True,
    band_color="0.45",
    band_alpha=0.22,
    mean_color="0.25",
    mean_lw=1.2,
    zorder=1,
):
    """
    Draw per-side (relative to vertical line x=x0) Q1-IQR band + mean line.
    "Left/right" are defined by what you SEE on the plot (respects invert_xaxis()).

    If inverted_xaxis=True:
        - display-left side corresponds to x > x0
        - display-right side corresponds to x < x0
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return

    x = x[m]
    y = y[m]

    if inverted_xaxis:
        m_left = x > x0
        m_right = x < x0
    else:
        m_left = x < x0
        m_right = x > x0

    xlim = ax.get_xlim()
    x_min = min(xlim)
    x_max = max(xlim)

    def _span_for_left():
        return (x0, x_max) if inverted_xaxis else (x_min, x0)

    def _span_for_right():
        return (x_min, x0) if inverted_xaxis else (x0, x_max)

    def _draw_one(mask, x_span):
        if mask.sum() < 3:
            return
        yy = y[mask]
        q1 = float(np.quantile(yy, 0.25))
        q3 = float(np.quantile(yy, 0.75))
        mu = float(np.mean(yy))

        xa, xb = x_span
        ax.fill_between(
            [xa, xb], [q1, q1], [q3, q3],
            color=band_color, alpha=band_alpha, linewidth=0,
            zorder=zorder
        )
        ax.hlines(mu, xa, xb, colors=mean_color, linewidth=mean_lw, zorder=zorder + 0.1)

    _draw_one(m_left, _span_for_left())
    _draw_one(m_right, _span_for_right())

def plot_ood_score_vs_deviation__nb03_c12(
    input_cpx,
    input_meta,
    model,
    input_liq=None,
    petrological_ood_marker=False,
    T_P_ood_marker=False,
    ax=None,
    cmap_name="cividis",
    norm_min=0.0,
    norm_max=10.0,
    s_base=55,
    s_overlay=60,
    lw_base=0.85,
    lw_overlay=0.95,
    ind_edge="black",
    ood_edge="red",
    stats_band_color="0.45",
    stats_band_alpha=0.22,
    stats_mean_color="0.25",
):
    model_name = getattr(model, "model_name", "Unnamed model")

    target_col = "T_C"
    if target_col not in input_meta.columns:
        raise ValueError(f"'{target_col}' not in input_meta.columns")

    color_col = "T_C"

    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=float(norm_min), vmax=float(norm_max), clip=True)

    cpx_key = _fast_df_fingerprint(input_cpx)
    liq_key = _fast_df_fingerprint(input_liq) if input_liq is not None else "none"
    meta_key = _fast_df_fingerprint(input_meta, cols=[target_col, color_col])
    cache_key = hashlib.md5(
        (str(model_name) + "|T|" + cpx_key + "|" + liq_key + "|" + meta_key).encode("utf-8")
    ).hexdigest()

    cached = cache_load("ood_scatter_T", cache_key)

    if cached is None:
        predictions = model.predict(input_cpx, input_liq)

        ood_model = getattr(model, "OOD_detector", None)
        if ood_model is None:
            ood_mask = np.zeros(len(input_cpx), dtype=bool)
            ood_score = np.full(len(input_cpx), np.nan)
        else:
            ood_score = ood_model.score(input_cpx, input_liq)
            ood_mask = ood_model.is_ood(input_cpx, input_liq)

        deviation = np.abs(input_meta[target_col].to_numpy(float) - np.asarray(predictions, float))

        tp_mask = np.zeros(len(input_cpx), bool)
        if T_P_ood_marker:
            y_min = getattr(model, "y_min_95", -np.inf)
            y_max = getattr(model, "y_max_95", np.inf)
            pred_arr = np.asarray(predictions, float)
            tp_mask = (pred_arr < y_min) | (pred_arr > y_max)

        tas_mask = np.zeros(len(input_cpx), bool)
        if petrological_ood_marker:
            if input_liq is None:
                raise ValueError("petrological_ood_marker=True requires input_liq.")
            from aims4pt.statistic_tools.density_region_analysis import rock_type_check
            tas_types = input_liq.apply(get_TAS_rock_types, axis=1)
            tas_ok = np.array([rock_type_check(rt, model.rock_types, report=False) for rt in tas_types])
            tas_mask = ~tas_ok

        cached = dict(
            ood_score=np.asarray(ood_score, float),
            ood_mask=np.asarray(ood_mask, bool),
            deviation=np.asarray(deviation, float),
            tp_mask=np.asarray(tp_mask, bool),
            tas_mask=np.asarray(tas_mask, bool),
            color_value=input_meta[color_col].to_numpy(float),
        )
        cache_save("ood_scatter_T", cache_key, cached)

    ood_score = cached["ood_score"]
    ood_mask = cached["ood_mask"]
    deviation = cached["deviation"]
    tp_mask = cached["tp_mask"] if T_P_ood_marker else np.zeros(len(input_cpx), bool)
    tas_mask = cached["tas_mask"] if petrological_ood_marker else np.zeros(len(input_cpx), bool)
    color_value = cached["color_value"]

    df = pd.DataFrame(
        dict(score=ood_score, deviation=deviation, color=color_value),
        index=input_meta.index
    ).dropna()

    idx = df.index
    m_ood = pd.Series(ood_mask, index=input_meta.index).reindex(idx).fillna(False).to_numpy(bool)
    m_tp = pd.Series(tp_mask, index=input_meta.index).reindex(idx).fillna(False).to_numpy(bool)
    m_tas = pd.Series(tas_mask, index=input_meta.index).reindex(idx).fillna(False).to_numpy(bool)

    excl = m_tp | m_tas
    m_ind = (~m_ood) & (~excl)
    m_oodf = m_ood & (~excl)

    ind_df = df.loc[m_ind]
    ood_df = df.loc[m_oodf]
    tas_df = df.loc[m_tas]
    tp_df = df.loc[m_tp]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    else:
        fig = ax.figure

    if len(ind_df) > 0:
        face = cmap(norm(ind_df["color"].to_numpy()))
        ax.scatter(
            ind_df["score"], ind_df["deviation"],
            s=s_base, marker="o",
            facecolors=face, edgecolors=ind_edge, linewidths=lw_base,
            alpha=0.75, zorder=3
        )

    if len(ood_df) > 0:
        face = cmap(norm(ood_df["color"].to_numpy()))
        ax.scatter(
            ood_df["score"], ood_df["deviation"],
            s=s_base, marker="s",
            facecolors=face, edgecolors=ood_edge, linewidths=lw_base,
            alpha=0.75, zorder=4
        )

    if petrological_ood_marker and len(tas_df) > 0:
        face = cmap(norm(tas_df["color"].to_numpy()))
        ax.scatter(
            tas_df["score"], tas_df["deviation"],
            s=s_overlay, marker="D",
            facecolors=face, edgecolors=ood_edge, linewidths=lw_overlay,
            alpha=0.85, zorder=7
        )

    if T_P_ood_marker and len(tp_df) > 0:
        face = cmap(norm(tp_df["color"].to_numpy()))
        ax.scatter(
            tp_df["score"], tp_df["deviation"],
            s=s_overlay, marker="^",
            facecolors=face, edgecolors=ood_edge, linewidths=lw_overlay,
            alpha=0.9, zorder=8
        )

    _draw_side_q1_mean_q3__nb03_c12(
        ax,
        df["score"].to_numpy(float),
        df["deviation"].to_numpy(float),
        x0=0.0,
        inverted_xaxis=True,  # consistent with invert_xaxis() below
        band_color=stats_band_color,
        band_alpha=stats_band_alpha,
        mean_color=stats_mean_color,
        mean_lw=1.2,
        zorder=1,
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(float(model.uncertainty), color="0.35", linestyle="--", linewidth=1)

    ax.invert_xaxis()
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlabel("Signed distance to the boundary")
    ax.set_ylabel("Absolute deviation (°C)")

    return fig, ax

def _fmt_num__nb03_c12(x: float) -> str:
    """Pretty formatting: integers as int; otherwise 1 decimal.
    Special case: values that round to 0.0 at 1 decimal -> "0".
    """
    if x is None:
        return "NA"
    x = float(x)
    if not np.isfinite(x):
        return "NA"
    if abs(x) < 5e-12:
        return "0"
    xr = round(x)
    if abs(x - xr) < 1e-9:
        return str(int(xr))
    if round(x, 1) == 0.0:
        return "0"
    return f"{x:.1f}"

def plot_ood_panels_generic__nb03_c12(
    models,
    test_cpx,
    test_meta,
    *,
    test_liq=None,
    ncols=2,
    petrological_ood_marker=False,
    T_P_ood_marker=False,
    fig_width=14,
    row_height=5.4,
    height_scale=1.08,
    cmap_name="cividis",
    norm_min=0.0,
    clip_to_p99=True,
    cbar_width=0.6,
    cbar_height=0.013,
    cbar_y=0.14,
    legend_anchor_y=0.15,
    tight_bottom=0.18,
    tight_top=0.96,
    # NEW: panel label controls
    add_panel_labels__nb00_c24=True,
    panel_label_kwargs=None,
):
    models = [m for m in models if getattr(m, "X_cpx_training", None) is not None]
    n = len(models)
    nrows = math.ceil(n / ncols)

    if "T_C" not in test_meta.columns:
        raise ValueError("'T_C' not in test_meta.columns (did you rename T_col -> 'T_C'?)")

    if clip_to_p99:
        tvals = pd.to_numeric(test_meta["T_C"], errors="coerce").dropna().to_numpy(float)
        t99 = float(np.quantile(tvals, 0.99))
        norm_max = t99
    else:
        norm_max = float(pd.to_numeric(test_meta["T_C"], errors="coerce").dropna().max())

    cbar_label = f"True T (°C), clipped to 99th percentile ({_fmt_num__nb03_c12(norm_max)} °C)"

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_width, row_height * nrows * height_scale),
        dpi=200,
        constrained_layout=False
    )
    axes = np.array(axes).ravel()

    for model, ax in zip(models, axes):
        plot_ood_score_vs_deviation__nb03_c12(
            test_cpx, test_meta, model,
            input_liq=test_liq,
            petrological_ood_marker=petrological_ood_marker,
            T_P_ood_marker=T_P_ood_marker,
            ax=ax,
            cmap_name=cmap_name,
            norm_min=norm_min,
            norm_max=norm_max,
        )

        name = re.sub(r"\s*\(.*?\)\s*$", "", model.model_name)
        y_min = getattr(model, "y_min", None)
        y_max = getattr(model, "y_max", None)
        if y_min is not None and y_max is not None:
            name += f" (Range: {_fmt_num__nb03_c12(float(y_min))}-{_fmt_num__nb03_c12(float(y_max))} °C)"
        ax.set_title(name, fontsize=13)

    for ax in axes[n:]:
        fig.delaxes(ax)

    # Only keep last row x-labels; only keep first column y-labels
    for i, ax in enumerate(axes[:n]):
        row = i // ncols
        col = i % ncols
        if row < nrows - 1:
            ax.set_xlabel("")
        if col > 0:
            ax.set_ylabel("")

    # NEW: panel labels (a)(b)(c)...
    if add_panel_labels__nb00_c24 and n > 0:
        if panel_label_kwargs is None:
            panel_label_kwargs = dict(fontsize=12, fontweight="bold")
        _add_panel_labels__nb03_c12(axes[:n], labels=None, x=0.02, y=0.96, **panel_label_kwargs)

    fig.tight_layout(rect=[0, tight_bottom, 1, tight_top])

    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
               markerfacecolor="white", markeredgecolor="black", label="In-distribution"),
        Line2D([0], [0], marker="s", linestyle="None",
               markerfacecolor="white", markeredgecolor="red", label="OOD (features)"),
        Line2D([0], [0], marker="D", linestyle="None",
               markerfacecolor="white", markeredgecolor="red", label="OOD (TAS)"),
        Line2D([0], [0], marker="^", linestyle="None",
               markerfacecolor="white", markeredgecolor="red", label="OOD (predicted T)"),
        Line2D([0], [0], linestyle="--", color="0.35", label="Model uncertainty"),
    ]
    labels = [h.get_label() for h in handles]

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_anchor_y),
        ncol=len(handles),
        frameon=False,
    )

    sm = ScalarMappable(norm=Normalize(norm_min, norm_max), cmap=get_cmap(cmap_name))
    cax = fig.add_axes([(1 - cbar_width) / 2, cbar_y, cbar_width, cbar_height])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label)

    return fig, axes



# Helpers extracted from 03_reproduce_AIMS4PT_cpx_figures.ipynb cell 16.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import ConvexHull, QhullError
from aims4pt.visualization.composition_plot import plot_glass_TAS_diagram

def _clean_model_label_keep_year__nb03_c16(model_label: str) -> str:
    """
    Clean legend label:
      - remove any suffix like " (cpx_only)" / " (cpx_liq)"
      - special-case Neave & Putirka, 2017 ... -> "Neave & Putirka (2017)"
    """
    base = model_label.split(" (")[0]  # strip trailing "(cpx_only)" etc.

    if base.startswith("Neave & Putirka, 2017"):
        return "Neave & Putirka (2017)"

    return base

def plot_tas_with_deviation_cbar__nb03_c16(
    *,
    model,
    model_label,
    X_liq_test,
    deviation,
    fig=None,
    ax=None,
    tas_model="LeMaitreCombined",
    cmap=plt.cm.viridis,
    q=0.95,
    draw_convex_hull=True,
    hull_ls="k--",
    hull_lw=1.0,
    train_marker="o",
    test_marker="o",
    s=40,
    train_edge_lw=0.5,
    test_edge_lw=0.5,
    legend_loc="upper right",
    legend_ncol=1,
    colorbar_label="Absolute Deviation",
    plot_alkaline_boundary=False,
    nan_color="lightgray",
    add_colorbar=True,
):
    """
    Unified TAS plotting style:
      - training liquids as empty markers (color="none")
      - convex hull (dashed envelope) for training liquids (optional)
      - testing liquids colored by clipped absolute deviation
      - NaN deviations filled with nan_color (NOT transparent)
      - colorbar with quantile clipping
      - legend inside axes (upper right)
      - Testing legend label fixed to "New dataset"
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    # -------------------------
    # Training liquids
    # -------------------------
    X_liq_training = getattr(model, "X_liq_training", None)
    if X_liq_training is None:
        raise ValueError("model has no attribute 'X_liq_training'. Cannot plot training liquids.")

    model_label_text = _clean_model_label_keep_year__nb03_c16(model_label)

    ax = plot_glass_TAS_diagram(
        X_liq=X_liq_training,
        model=tas_model,
        axes=ax,
        color="none",
        marker=train_marker,
        label=f"{model_label_text}; N={len(X_liq_training)}",
        plot_alkaline_boundary=plot_alkaline_boundary,
        linewidth=train_edge_lw,
        s=s,
    )

    # -------------------------
    # Convex hull (training)
    # -------------------------
    if draw_convex_hull:
        try:
            alkali = X_liq_training["Na2O_liq"] + X_liq_training["K2O_liq"]
            sio2 = X_liq_training["SiO2_liq"]
            hull_points = np.column_stack([sio2.to_numpy(dtype=float), alkali.to_numpy(dtype=float)])
            hull_points = hull_points[~np.isnan(hull_points).any(axis=1)]

            if hull_points.shape[0] >= 3:
                hull = ConvexHull(hull_points)
                for simplex in hull.simplices:
                    ax.plot(
                        hull_points[simplex, 0],
                        hull_points[simplex, 1],
                        hull_ls,
                        lw=hull_lw,
                    )
        except (KeyError, QhullError, ValueError):
            pass

    # -------------------------
    # Robust scaling (quantile clipping), ignoring NaNs
    # -------------------------
    dev = np.asarray(deviation, dtype=float)
    dev_finite = dev[np.isfinite(dev)]

    if dev_finite.size == 0:
        vmax = 1.0
    else:
        vmax = np.nanpercentile(dev_finite, q * 100)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

    dev_clip = np.clip(dev, 0.0, vmax)

    # -------------------------
    # Colors (NaN -> nan_color)
    # -------------------------
    colors = np.empty((dev.shape[0], 4), dtype=float)
    nan_rgba = mpl.colors.to_rgba(nan_color)

    is_finite = np.isfinite(dev_clip)
    if np.any(is_finite):
        colors[is_finite] = cmap(dev_clip[is_finite] / vmax)
    if np.any(~is_finite):
        colors[~is_finite] = nan_rgba

    # -------------------------
    # Testing liquids (colored) - legend label fixed
    # -------------------------
    ax = plot_glass_TAS_diagram(
        X_liq=X_liq_test,
        model=tas_model,
        axes=ax,
        marker=test_marker,
        color=colors,
        label="New dataset",
        plot_alkaline_boundary=plot_alkaline_boundary,
        linewidth=test_edge_lw,
        s=s,
    )

    # -------------------------
    # Colorbar
    # -------------------------
    cb = None
    if add_colorbar:
        norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(dev_clip[is_finite] if np.any(is_finite) else np.array([0.0, vmax]))
        cb = fig.colorbar(sm, ax=ax)
        cb.set_label(colorbar_label)

    # -------------------------
    # Legend (inside axes)
    # -------------------------
    ax.legend(loc=legend_loc, fancybox=True, shadow=True, ncol=legend_ncol)

    return fig, ax, cb

def _add_panel_labels__nb03_c16(axes, labels, *, x=0.02, y=0.98, **kwargs):
    """
    Add panel labels like (a), (b) ... to each axes.
    Default placement: top-left inside each panel.
    """
    default_kwargs = dict(
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )
    default_kwargs.update(kwargs)

    axes_flat = np.ravel(axes)
    if len(labels) != len(axes_flat):
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of axes ({len(axes_flat)})."
        )

    for ax, lab in zip(axes_flat, labels):
        ax.text(
            x,
            y,
            lab,
            transform=ax.transAxes,  # ✅ 只在这里指定
            **default_kwargs,
        )



# Helpers extracted from 03_reproduce_AIMS4PT_cpx_figures.ipynb cell 20.
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def _dedupe_handles_labels__nb03_c20(handles, labels):
    """Keep first occurrence order; drop duplicates by label."""
    od = OrderedDict()
    for h, l in zip(handles, labels):
        if l not in od:
            od[l] = h
    return list(od.values()), list(od.keys())

def _add_panel_label__nb03_c20(ax, label, *, x=0.02, y=0.98, fontsize=14):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
        zorder=10,
    )

def plot_pressure_residual_panel__nb03_c20(
    P_real,
    predicted_P_df,
    model_cols,
    workflow_col,
    *,
    model_labels=None,
    P_min=0,
    P_max=30,
    ax=None,
    gray_worse_than_workflow=True,
    skip_tab10_gray=True,
    marker_sets_master=None,
    title="Pressure Residuals Panel",
    # ✅ NEW
    add_workflow_rmse_text=True,
    rmse_text_kwargs=None,
):
    # ----------------------------
    # Filter samples
    # ----------------------------
    interested_ids = P_real[(P_real >= P_min) & (P_real <= P_max)].index
    P_real_ = P_real.loc[interested_ids]
    predicted_ = predicted_P_df.loc[interested_ids]
    assert P_real_.index.equals(predicted_.index)

    if model_labels is None:
        model_labels = [col.split("(")[0] for col in model_cols]
    if len(model_labels) != len(model_cols):
        raise ValueError("model_labels must match model_cols length.")

    wf_pred = predicted_[workflow_col]
    wf_residual = wf_pred - P_real_

    # ✅ NEW: RMSE for workflow in this panel
    # RMSE = sqrt(mean((wf_residual)^2)), computed on the filtered samples
    wf_rmse = float(np.sqrt(np.nanmean((wf_residual.values) ** 2)))
    # avoid nan
    df_ = pd.DataFrame({'P_real': P_real_.values, 'wf_pred': wf_pred.values})
    df_.dropna(inplace=True)
    r2 = r2_score(df_['P_real'], df_['wf_pred'])

    # ----------------------------
    # Styles
    # ----------------------------
    if marker_sets_master is None:
        marker_sets_master = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '8', '<', '>']

    tab10 = list(plt.cm.tab10.colors)
    if skip_tab10_gray:
        color_sets_master = [c for i, c in enumerate(tab10) if i != 7]
    else:
        color_sets_master = tab10

    marker_sets = marker_sets_master.copy()
    color_sets = color_sets_master.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200)

    legend_handles, legend_labels = [], []

    # ----------------------------
    # Plot models
    # ----------------------------
    for col, lab in zip(model_cols, model_labels):
        y_pred = predicted_[col]
        residual = y_pred - P_real_

        marker = marker_sets.pop(0) if marker_sets else 'o'
        color = color_sets.pop(0) if color_sets else 'C0'

        if gray_worse_than_workflow:
            mask_worse = np.abs(residual) > np.abs(wf_residual)
            mask_better = ~mask_worse

            sc = ax.scatter(
                P_real_[mask_better], residual[mask_better],
                marker=marker, s=50,
                facecolors=(0, 0, 0, 0),
                edgecolors=color, linewidths=1.2, zorder=2,
            )
            ax.scatter(
                P_real_[mask_worse], residual[mask_worse],
                marker=marker, s=50,
                facecolors=(0, 0, 0, 0),
                edgecolors='0.7', linewidths=1.0, zorder=1,
            )
        else:
            sc = ax.scatter(
                P_real_, residual,
                marker=marker, s=50,
                facecolors=(0, 0, 0, 0),
                edgecolors=color, linewidths=1.2,
            )

        legend_handles.append(sc)
        legend_labels.append(lab)

    # ----------------------------
    # Plot workflow
    # ----------------------------
    sc_wf = ax.scatter(
        P_real_, wf_residual,
        marker='*', s=150,
        color='red', edgecolor='black', zorder=3,
    )
    legend_handles.append(sc_wf)
    legend_labels.append("AIMS4PT_cpx (this study)")

    # ----------------------------
    # Decorations
    # ----------------------------
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlim(P_min - 1, P_max + 1)
    ax.set_xlabel("True P (kbar)")
    ax.set_ylabel("ΔP (kbar)")
    ax.set_title(title)

    # ✅ NEW: RMSE text in the top-right
    if add_workflow_rmse_text:
        if rmse_text_kwargs is None:
            rmse_text_kwargs = dict(
                fontsize=12,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
            )
        ax.text(
            0.6, 0.90,
            f"RMSE (AIMS4PT_cpx) = {wf_rmse:.2f} kbar",
            transform=ax.transAxes,
            **rmse_text_kwargs,
        )

    return ax, (legend_handles, legend_labels)



# Helpers extracted from 03_reproduce_AIMS4PT_cpx_figures.ipynb cell 23.
from collections import OrderedDict
import matplotlib.pyplot as plt

def _dedupe_handles_labels__nb03_c23(handles, labels):
    """Keep first occurrence order; drop duplicates by label."""
    od = OrderedDict()
    for h, l in zip(handles, labels):
        if l not in od:
            od[l] = h
    return list(od.values()), list(od.keys())

def _add_panel_label__nb03_c23(ax, label, *, x=0.02, y=0.98, fontsize=14):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
        zorder=10,
    )

def plot_temperature_residual_panel__nb03_c23(
    T_real,
    predicted_T_df,
    model_cols,
    workflow_col,
    *,
    model_labels=None,
    T_min=700,
    T_max=1350,
    ax=None,
    gray_worse_than_workflow=True,
    skip_tab10_gray=True,
    marker_sets_master=None,
    title="Temperature Residuals Panel",
    limit_y=False,
    add_workflow_rmse_text=True,
    rmse_text_kwargs=None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    # ----------------------------
    # Filter samples
    # ----------------------------
    interested_ids = T_real[(T_real >= T_min) & (T_real <= T_max)].index
    T_real_ = T_real.loc[interested_ids]
    predicted_ = predicted_T_df.loc[interested_ids]
    assert T_real_.index.equals(predicted_.index)

    if model_labels is None:
        model_labels = [col.split("(")[0] for col in model_cols]
    if len(model_labels) != len(model_cols):
        raise ValueError("model_labels must match model_cols length.")

    wf_pred = predicted_[workflow_col]
    wf_residual = wf_pred - T_real_

        # ✅ NEW: RMSE for workflow in this panel
    # RMSE = sqrt(mean((wf_residual)^2)), computed on the filtered samples
    wf_rmse = float(np.sqrt(np.nanmean((wf_residual.values) ** 2)))
    # avoid nan
    df_ = pd.DataFrame({'T_real': T_real_.values, 'wf_pred': wf_pred.values})
    df_.dropna(inplace=True)
    r2 = r2_score(df_['T_real'], df_['wf_pred'])

    # ----------------------------
    # Styles
    # ----------------------------
    if marker_sets_master is None:
        marker_sets_master = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '8', '<', '>']

    tab10 = list(plt.cm.tab10.colors)
    if skip_tab10_gray:
        color_sets_master = [c for i, c in enumerate(tab10) if i != 7]
    else:
        color_sets_master = tab10

    marker_sets = marker_sets_master.copy()
    color_sets = color_sets_master.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200)

    legend_handles, legend_labels = [], []

    # ----------------------------
    # Plot models
    # ----------------------------
    for col, lab in zip(model_cols, model_labels):
        y_pred = predicted_[col]
        residual = y_pred - T_real_

        marker = marker_sets.pop(0) if marker_sets else 'o'
        color = color_sets.pop(0) if color_sets else 'C0'

        if gray_worse_than_workflow:
            mask_worse = abs(residual) > abs(wf_residual)
            mask_better = ~mask_worse

            sc = ax.scatter(
                T_real_[mask_better], residual[mask_better],
                marker=marker, s=50,
                facecolors=(0, 0, 0, 0),
                edgecolors=color, linewidths=1.2, zorder=2,
            )
            ax.scatter(
                T_real_[mask_worse], residual[mask_worse],
                marker=marker, s=50,
                facecolors=(0, 0, 0, 0),
                edgecolors='0.7', linewidths=1.0, zorder=1,
            )
        else:
            sc = ax.scatter(
                T_real_, residual,
                marker=marker, s=50,
                facecolors=(0, 0, 0, 0),
                edgecolors=color, linewidths=1.2,
            )

        legend_handles.append(sc)
        legend_labels.append(lab)

    # ----------------------------
    # Plot workflow
    # ----------------------------
    sc_wf = ax.scatter(
        T_real_, wf_residual,
        marker='*', s=150,
        color='red', edgecolor='black', zorder=3,
    )
    legend_handles.append(sc_wf)
    legend_labels.append("AIMS4PT_cpx (this study)")

    # ----------------------------
    # Decorations
    # ----------------------------
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlim(T_min, T_max)
    if limit_y:
        ax.set_ylim(top=400,)
    ax.set_xlabel("True T (°C)")
    ax.set_ylabel("ΔT (°C)")
    ax.set_title(title)

        # ✅ NEW: RMSE text in the top-right
    if add_workflow_rmse_text:
        if rmse_text_kwargs is None:
            rmse_text_kwargs = dict(
                fontsize=12,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0),
            )
        ax.text(
            0.6, 0.90,
            f"RMSE (AIMS4PT_cpx) = {wf_rmse:.0f} °C",
            transform=ax.transAxes,
            **rmse_text_kwargs,
        )

    return ax, (legend_handles, legend_labels)



# Helpers extracted from 03_reproduce_AIMS4PT_cpx_figures.ipynb cell 27.
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def _dedupe_handles_labels__nb03_c27(handles, labels):
    """Keep first occurrence order; drop duplicates by label."""
    od = OrderedDict()
    for h, l in zip(handles, labels):
        if l not in od:
            od[l] = h
    return list(od.values()), list(od.keys())

def _add_panel_label__nb03_c27(ax, label, *, x=0.02, y=0.98, fontsize=14):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
        zorder=10,
    )

def _calc_regression_fit__nb03_c27(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return np.nan, np.nan, 0
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    return float(slope), float(intercept), int(len(df))

def plot_pressure_residual_panel__nb03_c27(
    P_real,
    predicted_P_df,
    model_cols,
    workflow_col,
    *,
    model_labels=None,
    P_min=0,
    P_max=30,
    ax=None,
    gray_worse_than_workflow=True,
    skip_tab10_gray=True,
    marker_sets_master=None,
    title="Pressure Residuals Panel",
    # ✅ NEW
    add_workflow_rmse_text=True,
    rmse_text_kwargs=None,
):
    # ----------------------------
    # Filter samples
    # ----------------------------
    interested_ids = P_real[(P_real >= P_min) & (P_real <= P_max)].index
    P_real_ = P_real.loc[interested_ids]
    predicted_ = predicted_P_df.loc[interested_ids]
    assert P_real_.index.equals(predicted_.index)

    if model_labels is None:
        model_labels = [col.split("(")[0] for col in model_cols]
    if len(model_labels) != len(model_cols):
        raise ValueError("model_labels must match model_cols length.")

    wf_pred = predicted_[workflow_col]
    wf_residual = wf_pred - P_real_

    # ✅ NEW: RMSE for workflow in this panel
    # RMSE = sqrt(mean((wf_residual)^2)), computed on the filtered samples
    wf_rmse = float(np.sqrt(np.nanmean((wf_residual.values) ** 2)))
    # avoid nan
    df_ = pd.DataFrame({'P_real': P_real_.values, 'wf_pred': wf_pred.values})
    df_.dropna(inplace=True)
    r2 = r2_score(df_['P_real'], df_['wf_pred'])

    # ----------------------------
    # Styles
    # ----------------------------
    if marker_sets_master is None:
        marker_sets_master = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '8', '<', '>']

    tab10 = list(plt.cm.tab10.colors)
    if skip_tab10_gray:
        color_sets_master = [c for i, c in enumerate(tab10) if i != 7]
    else:
        color_sets_master = tab10

    marker_sets = marker_sets_master.copy()
    color_sets = color_sets_master.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200)

    legend_handles, legend_labels = [], []
    slope_rows = []

    # ----------------------------
    # Plot models
    # ----------------------------
    for col, lab in zip(model_cols, model_labels):
        y_pred = predicted_[col]
        residual = y_pred - P_real_

        marker = marker_sets.pop(0) if marker_sets else 'o'
        color = color_sets.pop(0) if color_sets else 'C0'

        slope, intercept, n_points = _calc_regression_fit__nb03_c27(P_real_, residual)

        if gray_worse_than_workflow:
            mask_worse = np.abs(residual) > np.abs(wf_residual)
            mask_better = ~mask_worse

            sc = ax.scatter(
                P_real_[mask_better], residual[mask_better],
                marker=marker, s=50,
                facecolors=(0, 0, 0, 0),
                edgecolors=color, linewidths=1.2, zorder=2,
            )
            ax.scatter(
                P_real_[mask_worse], residual[mask_worse],
                marker=marker, s=50,
                facecolors=(0, 0, 0, 0),
                edgecolors='0.7', linewidths=1.0, zorder=1,
            )
        else:
            sc = ax.scatter(
                P_real_, residual,
                marker=marker, s=50,
                facecolors=(0, 0, 0, 0),
                edgecolors=color, linewidths=1.2,
            )

        legend_handles.append(sc)
        legend_labels.append(lab)
        if np.isfinite(slope):
            x_line = np.array([P_min, P_max], dtype=float)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=color, linewidth=1.5, alpha=0.9, zorder=2.5)
        slope_rows.append({
            "model_name": lab,
            "slope": slope,
            "intercept": intercept,
            "n_points": n_points,
        })

    # ----------------------------
    # Plot workflow
    # ----------------------------
    sc_wf = ax.scatter(
        P_real_, wf_residual,
        marker='*', s=150,
        color='red', edgecolor='black', zorder=3,
    )
    legend_handles.append(sc_wf)
    legend_labels.append("AIMS4PT_cpx  (this study)")
    wf_slope, wf_intercept, wf_n_points = _calc_regression_fit__nb03_c27(P_real_, wf_residual)
    if np.isfinite(wf_slope):
        x_line = np.array([P_min, P_max], dtype=float)
        y_line = wf_slope * x_line + wf_intercept
        ax.plot(x_line, y_line, color="black", linewidth=2.8, zorder=4)
    slope_rows.append({
        "model_name": "AIMS4PT_cpx  (this study)",
        "slope": wf_slope,
        "intercept": wf_intercept,
        "n_points": wf_n_points,
    })

    # ----------------------------
    # Decorations
    # ----------------------------
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlim(P_min - 1, P_max + 1)
    ax.set_xlabel("True P (kbar)")
    ax.set_ylabel("ΔP (kbar)")
    ax.set_title(title)

    # ✅ NEW: RMSE text in the top-right
    if add_workflow_rmse_text:
        if rmse_text_kwargs is None:
            rmse_text_kwargs = dict(
                fontsize=12,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
            )
        ax.text(
            0.6, 0.90,
            f"RMSE (AIMS4PT_cpx ) = {wf_rmse:.2f} kbar",
            transform=ax.transAxes,
            **rmse_text_kwargs,
        )

    slope_df = pd.DataFrame(slope_rows)
    return ax, (legend_handles, legend_labels), slope_df



# Helpers extracted from 03_reproduce_AIMS4PT_cpx_figures.ipynb cell 29.
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def _dedupe_handles_labels__nb03_c29(handles, labels):
    """Keep first occurrence order; drop duplicates by label."""
    od = OrderedDict()
    for h, l in zip(handles, labels):
        if l not in od:
            od[l] = h
    return list(od.values()), list(od.keys())

def _add_panel_label__nb03_c29(ax, label, *, x=0.02, y=0.98, fontsize=14):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
        zorder=10,
    )

def _calc_regression_fit__nb03_c29(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return np.nan, np.nan, 0
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    return float(slope), float(intercept), int(len(df))

def plot_temperature_residual_panel__nb03_c29(
    T_real,
    predicted_T_df,
    model_cols,
    workflow_col,
    *,
    model_labels=None,
    T_min=700,
    T_max=1350,
    ax=None,
    gray_worse_than_workflow=True,
    skip_tab10_gray=True,
    marker_sets_master=None,
    title="Temperature Residuals Panel",
    limit_y=False,
    add_workflow_rmse_text=True,
    rmse_text_kwargs=None,
):
    # ----------------------------
    # Filter samples
    # ----------------------------
    interested_ids = T_real[(T_real >= T_min) & (T_real <= T_max)].index
    T_real_ = T_real.loc[interested_ids]
    predicted_ = predicted_T_df.loc[interested_ids]
    assert T_real_.index.equals(predicted_.index)

    if model_labels is None:
        model_labels = [col.split("(")[0] for col in model_cols]
    if len(model_labels) != len(model_cols):
        raise ValueError("model_labels must match model_cols length.")

    wf_pred = predicted_[workflow_col]
    wf_residual = wf_pred - T_real_

        # ✅ NEW: RMSE for workflow in this panel
    # RMSE = sqrt(mean((wf_residual)^2)), computed on the filtered samples
    wf_rmse = float(np.sqrt(np.nanmean((wf_residual.values) ** 2)))
    # avoid nan
    df_ = pd.DataFrame({'T_real': T_real_.values, 'wf_pred': wf_pred.values})
    df_.dropna(inplace=True)
    r2 = r2_score(df_['T_real'], df_['wf_pred'])

    # ----------------------------
    # Styles
    # ----------------------------
    if marker_sets_master is None:
        marker_sets_master = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '8', '<', '>']

    tab10 = list(plt.cm.tab10.colors)
    if skip_tab10_gray:
        color_sets_master = [c for i, c in enumerate(tab10) if i != 7]
    else:
        color_sets_master = tab10

    marker_sets = marker_sets_master.copy()
    color_sets = color_sets_master.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200)

    legend_handles, legend_labels = [], []
    slope_rows = []

    # ----------------------------
    # Plot models
    # ----------------------------
    for col, lab in zip(model_cols, model_labels):
        y_pred = predicted_[col]
        residual = y_pred - T_real_

        marker = marker_sets.pop(0) if marker_sets else 'o'
        color = color_sets.pop(0) if color_sets else 'C0'

        slope, intercept, n_points = _calc_regression_fit__nb03_c29(T_real_, residual)

        if gray_worse_than_workflow:
            mask_worse = abs(residual) > abs(wf_residual)
            mask_better = ~mask_worse

            sc = ax.scatter(
                T_real_[mask_better], residual[mask_better],
                marker=marker, s=50,
                facecolors=(0, 0, 0, 0),
                edgecolors=color, linewidths=1.2, zorder=2,
            )
            ax.scatter(
                T_real_[mask_worse], residual[mask_worse],
                marker=marker, s=50,
                facecolors=(0, 0, 0, 0),
                edgecolors='0.7', linewidths=1.0, zorder=1,
            )
        else:
            sc = ax.scatter(
                T_real_, residual,
                marker=marker, s=50,
                facecolors=(0, 0, 0, 0),
                edgecolors=color, linewidths=1.2,
            )

        legend_handles.append(sc)
        legend_labels.append(lab)
        if np.isfinite(slope):
            x_line = np.array([T_min, T_max], dtype=float)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=color, linewidth=1.5, alpha=0.9, zorder=2.5)
        slope_rows.append({
            "model_name": lab,
            "slope": slope,
            "intercept": intercept,
            "n_points": n_points,
        })

    # ----------------------------
    # Plot workflow
    # ----------------------------
    sc_wf = ax.scatter(
        T_real_, wf_residual,
        marker='*', s=150,
        color='red', edgecolor='black', zorder=3,
    )
    legend_handles.append(sc_wf)
    legend_labels.append("AIMS4PT_cpx  (this study)")
    wf_slope, wf_intercept, wf_n_points = _calc_regression_fit__nb03_c29(T_real_, wf_residual)
    if np.isfinite(wf_slope):
        x_line = np.array([T_min, T_max], dtype=float)
        y_line = wf_slope * x_line + wf_intercept
        ax.plot(x_line, y_line, color="black", linewidth=2.8, zorder=4)
    slope_rows.append({
        "model_name": "AIMS4PT_cpx  (this study)",
        "slope": wf_slope,
        "intercept": wf_intercept,
        "n_points": wf_n_points,
    })

    # ----------------------------
    # Decorations
    # ----------------------------
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlim(T_min, T_max)
    if limit_y:
        ax.set_ylim(top=400,)
    ax.set_xlabel("True T (°C)")
    ax.set_ylabel("ΔT (°C)")
    ax.set_title(title)

        # ✅ NEW: RMSE text in the top-right
    if add_workflow_rmse_text:
        if rmse_text_kwargs is None:
            rmse_text_kwargs = dict(
                fontsize=12,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0),
            )
        ax.text(
            0.6, 0.90,
            f"RMSE (AIMS4PT_cpx ) = {wf_rmse:.0f} °C",
            transform=ax.transAxes,
            **rmse_text_kwargs,
        )

    slope_df = pd.DataFrame(slope_rows)
    return ax, (legend_handles, legend_labels), slope_df

