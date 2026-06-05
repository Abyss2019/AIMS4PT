"""Helper functions shared by the Merapi application notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd


MERAPI_2006_COLOR = "#3995d6"
MERAPI_2010_COLOR = "#f15454"
INDEPENDENT_EXPERIMENTAL_COLOR = "0.62"
MERAPI_YEAR_COLORS = {"2006": MERAPI_2006_COLOR, "2010": MERAPI_2010_COLOR}
MOLAR_MASS_MGO = 40.3044
MOLAR_MASS_FEO = 71.844


def _series_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric series, or all-NaN if the column is absent."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _add_mg_number_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add molar Mg# columns for liquid and clinopyroxene compositions."""
    out = df.copy()
    for suffix in ("liq", "cpx"):
        mg = _series_numeric(out, f"MgO_{suffix}") / MOLAR_MASS_MGO
        fe = _series_numeric(out, f"FeO_{suffix}") / MOLAR_MASS_FEO
        if fe.isna().all():
            fe = _series_numeric(out, f"FeOt_{suffix}") / MOLAR_MASS_FEO
        denom = mg + fe
        out[f"MgNumber_{suffix}"] = np.where(denom > 0, mg / denom, np.nan)
    return out


def _add_merapi_plot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived plotting columns used by the Merapi composition figure."""
    out = df.copy()
    if "Na2O_liq" in out.columns and "K2O_liq" in out.columns:
        out["TotalAlkali_liq"] = _series_numeric(out, "Na2O_liq") + _series_numeric(out, "K2O_liq")
    return _add_mg_number_columns(out)


def _merapi_pair_frame(cpx_df: pd.DataFrame, liq_df: pd.DataFrame) -> pd.DataFrame:
    """Return paired Merapi cpx-liquid compositions with paper-style suffixes."""
    cpx = cpx_df.add_suffix("_cpx")
    liq = liq_df.add_suffix("_liq")
    return pd.concat([cpx, liq], axis=1)


def _liquid_endmember_frame(liq_df: pd.DataFrame, year: str) -> pd.DataFrame:
    """Return whole-rock and glass endmember liquids with group labels."""
    liq = liq_df.copy()
    group_col = next((col for col in ["glass/bulk", "glass_bulk", "liquid_group"] if col in liq.columns), None)
    if group_col is None:
        raise KeyError("Merapi liquid endmember table must contain a glass/bulk group column.")
    group = liq[group_col].astype(str).str.lower().str.strip().map(
        {
            "bulk": "whole-rock endmember",
            "whole rock": "whole-rock endmember",
            "whole-rock": "whole-rock endmember",
            "glass": "glass endmember",
        }
    )
    out = liq.add_suffix("_liq")
    out["eruption_year"] = str(year)
    out["liquid_group"] = group.to_numpy()
    return _add_merapi_plot_columns(out)


def _finite_xy_frame(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Return finite x-y rows for plotting and KDE."""
    xy = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return xy


def _finite_xy(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return finite x-y arrays."""
    xy = _finite_xy_frame(df, x_col, y_col)
    return xy[x_col].to_numpy(dtype=float), xy[y_col].to_numpy(dtype=float)


def _auto_xy_limits(frames: list[pd.DataFrame], x_col: str, y_col: str, pad_frac: float = 0.06):
    """Compute loose plotting limits for KDE grids without forcing axis limits."""
    x_parts, y_parts = [], []
    for frame in frames:
        if x_col in frame.columns and y_col in frame.columns:
            xy = _finite_xy_frame(frame, x_col, y_col)
            if not xy.empty:
                x_parts.append(xy[x_col])
                y_parts.append(xy[y_col])
    if not x_parts:
        return None, None
    x_all = pd.concat(x_parts)
    y_all = pd.concat(y_parts)
    x_min, x_max = float(x_all.min()), float(x_all.max())
    y_min, y_max = float(y_all.min()), float(y_all.max())
    x_pad = (x_max - x_min) * pad_frac if x_max > x_min else 0.5
    y_pad = (y_max - y_min) * pad_frac if y_max > y_min else 0.5
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _kde_level_for_mass(z: np.ndarray, mass: float) -> float:
    """Return the density level enclosing an approximate cumulative probability mass."""
    flat = np.asarray(z, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0 or np.all(flat <= 0):
        return np.nan
    order = np.argsort(flat)[::-1]
    sorted_z = flat[order]
    cumulative = np.cumsum(sorted_z) / np.sum(sorted_z)
    idx = np.searchsorted(cumulative, mass, side="left")
    idx = min(idx, len(sorted_z) - 1)
    return float(sorted_z[idx])


def _draw_kde_mass_contours(ax, df: pd.DataFrame, x_col: str, y_col: str, xlim, ylim):
    """Draw independent-data KDE contours enclosing about 50, 80, 95, and 98% mass."""
    from scipy.stats import gaussian_kde

    x, y = _finite_xy(df, x_col, y_col)
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0 or xlim is None or ylim is None:
        return
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 180), np.linspace(ylim[0], ylim[1], 180))
    try:
        zz = gaussian_kde(np.vstack([x, y]))(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    except np.linalg.LinAlgError:
        return

    solid_levels = np.unique(np.sort([_kde_level_for_mass(zz, mass) for mass in (0.95, 0.80, 0.50)]))
    solid_levels = solid_levels[np.isfinite(solid_levels)]
    if len(solid_levels):
        ax.contour(xx, yy, zz, levels=solid_levels, colors="0.48", linewidths=0.95, zorder=1)
    broad_level = _kde_level_for_mass(zz, 0.98)
    if np.isfinite(broad_level):
        ax.contour(xx, yy, zz, levels=[broad_level], colors="0.68", linewidths=0.80, linestyles="--", zorder=1)


def _plot_independent_background(ax, independent_df: pd.DataFrame, x_col: str, y_col: str):
    """Draw independent experimental data as a bottom-layer gray cloud."""
    x_ind, y_ind = _finite_xy(independent_df, x_col, y_col)
    ax.scatter(
        x_ind,
        y_ind,
        s=14,
        marker="o",
        color=INDEPENDENT_EXPERIMENTAL_COLOR,
        alpha=0.85,
        edgecolors="none",
        zorder=2,
    )


def _plot_cpx_points(ax, merapi_2006_df: pd.DataFrame, merapi_2010_df: pd.DataFrame, x_col: str, y_col: str):
    """Plot Merapi cpx points as colored circles by eruption year."""
    for year, data, zorder in [
        ("2010", merapi_2010_df, 4),
        ("2006", merapi_2006_df, 5),
    ]:
        x, y = _finite_xy(data, x_col, y_col)
        ax.scatter(
            x,
            y,
            marker="o",
            s=36,
            facecolors=MERAPI_YEAR_COLORS[year],
            edgecolors="0.15",
            linewidths=0.45,
            alpha=0.92,
            zorder=zorder,
        )


LIQUID_GROUP_STYLES = {
    "whole-rock endmember": {"marker": "s", "size": 62, "filled": True},
    "glass endmember": {"marker": "^", "size": 62, "filled": True},
    "equilibrium liquid": {"marker": "o", "size": 36, "filled": True},
}


def _plot_liquid_groups(ax, liquid_df: pd.DataFrame, x_col: str, y_col: str):
    """Plot Merapi liquids with year encoded by color and liquid group by marker/fill."""
    layer_order = [
        ("2010", "whole-rock endmember", 4),
        ("2010", "glass endmember", 4),
        ("2006", "whole-rock endmember", 5),
        ("2006", "glass endmember", 5),
        ("2010", "equilibrium liquid", 6),
        ("2006", "equilibrium liquid", 7),
    ]
    for year, group, zorder in layer_order:
        style = LIQUID_GROUP_STYLES[group]
        color = MERAPI_YEAR_COLORS[year]
        data = liquid_df[
            liquid_df["eruption_year"].astype(str).eq(year)
            & liquid_df["liquid_group"].astype(str).eq(group)
        ]
        if data.empty:
            continue
        x, y = _finite_xy(data, x_col, y_col)
        if style["filled"]:
            is_equilibrium_liquid = group == "equilibrium liquid"
            ax.scatter(
                x,
                y,
                marker=style["marker"],
                s=style["size"],
                facecolors=color,
                edgecolors="0.15" if is_equilibrium_liquid else "black",
                linewidths=0.45 if is_equilibrium_liquid else 0.95,
                alpha=0.92 if is_equilibrium_liquid else 0.88,
                zorder=zorder,
            )
        else:
            ax.scatter(
                x,
                y,
                marker=style["marker"],
                s=style["size"],
                facecolors="none",
                edgecolors=color,
                linewidths=1.25,
                alpha=0.98,
                zorder=zorder,
            )


def _add_top_merapi_legend(fig):
    """Add the global eruption/dataset legend for fig. 7."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MERAPI_2006_COLOR, markeredgecolor="0.15", markersize=7, label="2006 eruption"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MERAPI_2010_COLOR, markeredgecolor="0.15", markersize=7, label="2010 eruption"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INDEPENDENT_EXPERIMENTAL_COLOR, markeredgecolor="none", markersize=7.5, label="Independent experimental dataset"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=True,
        fontsize=10,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelspacing=0.4,
    )


def _add_liquid_group_legend(ax, loc="upper right", bbox_to_anchor=None):
    """Add a local liquid-group legend to a liquid composition panel."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="0.75", markeredgecolor="black", markeredgewidth=0.95, markersize=6.1, label="whole-rock endmember"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="0.75", markeredgecolor="black", markeredgewidth=0.95, markersize=6.1, label="glass endmember"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45", markeredgecolor="0.15", markeredgewidth=0.45, markersize=5.6, label="equilibrium liquid"),
    ]
    ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        fontsize=7.5,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.2,
        borderpad=0.25,
        borderaxespad=0.25,
    )


def _draw_merapi_composition_panel(
    ax,
    spec: dict,
    independent_pairs: pd.DataFrame,
    merapi_pairs_2006: pd.DataFrame,
    merapi_pairs_2010: pd.DataFrame,
    merapi_liquid_points: pd.DataFrame,
    xlim_override=None,
    ylim_override=None,
):
    """Draw one Merapi composition comparison panel."""
    x_col = spec["x_col"]
    y_col = spec["y_col"]
    xlim_auto, ylim_auto = _auto_xy_limits(
        [independent_pairs, merapi_pairs_2006, merapi_pairs_2010, merapi_liquid_points],
        x_col,
        y_col,
    )
    xlim_for_kde = xlim_override or xlim_auto
    ylim_for_kde = ylim_override or ylim_auto
    _draw_kde_mass_contours(ax, independent_pairs, x_col, y_col, xlim_for_kde, ylim_for_kde)
    _plot_independent_background(ax, independent_pairs, x_col, y_col)
    if spec["kind"] == "liquid":
        _plot_liquid_groups(ax, merapi_liquid_points, x_col, y_col)
        _add_liquid_group_legend(
            ax,
            loc=spec.get("legend_loc", "upper right"),
            bbox_to_anchor=spec.get("legend_bbox"),
        )
    else:
        _plot_cpx_points(ax, merapi_pairs_2006, merapi_pairs_2010, x_col, y_col)

    if xlim_override is not None:
        ax.set_xlim(*xlim_override)
    if ylim_override is not None:
        ax.set_ylim(*ylim_override)
    ax.grid(True, color="0.92", linewidth=0.4)
    ax.tick_params(labelsize=10)
    ax.set_xlabel(spec["xlabel"], fontsize=11)
    ax.set_ylabel(spec["ylabel"], fontsize=11)
    ax.text(0.02, 0.96, spec["label"], transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")


def plot_merapi_composition_comparison(
    unseen_experiments_df: pd.DataFrame,
    paired_cpx_06_pass: pd.DataFrame,
    paired_liq_06_pass: pd.DataFrame,
    paired_cpx_10_pass: pd.DataFrame,
    paired_liq_10_pass: pd.DataFrame,
    liq_2006: pd.DataFrame,
    liq_2010: pd.DataFrame,
    *,
    xlim_overrides: dict[str, tuple[float, float]] | None = None,
    ylim_overrides: dict[str, tuple[float, float]] | None = None,
    figsize=(8.2, 7.2),
    dpi=300,
):
    """Plot Merapi cpx/liquid compositions against the independent experimental dataset."""
    import matplotlib.pyplot as plt

    xlim_overrides = xlim_overrides or {}
    ylim_overrides = ylim_overrides or {}
    independent_pairs = _add_merapi_plot_columns(unseen_experiments_df)
    merapi_pairs_2006 = _add_merapi_plot_columns(_merapi_pair_frame(paired_cpx_06_pass, paired_liq_06_pass))
    merapi_pairs_2010 = _add_merapi_plot_columns(_merapi_pair_frame(paired_cpx_10_pass, paired_liq_10_pass))
    merapi_pairs_2006["eruption_year"] = "2006"
    merapi_pairs_2010["eruption_year"] = "2010"
    merapi_pairs_2006["liquid_group"] = "equilibrium liquid"
    merapi_pairs_2010["liquid_group"] = "equilibrium liquid"
    merapi_liquid_points = pd.concat(
        [
            _liquid_endmember_frame(liq_2006, "2006"),
            _liquid_endmember_frame(liq_2010, "2010"),
            merapi_pairs_2006,
            merapi_pairs_2010,
        ],
        ignore_index=True,
    )

    panel_specs = [
        {
            "key": "cpx_cao_mg_number",
            "kind": "cpx",
            "label": "(a)",
            "x_col": "CaO_cpx",
            "y_col": "MgNumber_cpx",
            "xlabel": r"Clinopyroxene CaO (wt%)",
            "ylabel": r"Clinopyroxene Mg#",
        },
        {
            "key": "cpx_na_al",
            "kind": "cpx",
            "label": "(b)",
            "x_col": "Na2O_cpx",
            "y_col": "Al2O3_cpx",
            "xlabel": r"Clinopyroxene Na$_2$O (wt%)",
            "ylabel": r"Clinopyroxene Al$_2$O$_3$ (wt%)",
            "xlim": (0, 1.1),
        },
        {
            "key": "liquid_tas",
            "kind": "liquid",
            "label": "(c)",
            "x_col": "SiO2_liq",
            "y_col": "TotalAlkali_liq",
            "xlabel": r"Liquid SiO$_2$ (wt%)",
            "ylabel": r"Liquid Na$_2$O + K$_2$O (wt%)",
            "legend_loc": "lower right",
        },
        {
            "key": "liquid_sio2_mg_number",
            "kind": "liquid",
            "label": "(d)",
            "x_col": "SiO2_liq",
            "y_col": "MgNumber_liq",
            "xlabel": r"Liquid SiO$_2$ (wt%)",
            "ylabel": r"Liquid Mg#",
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    for ax, spec in zip(axes.ravel(), panel_specs):
        _draw_merapi_composition_panel(
            ax,
            spec,
            independent_pairs,
            merapi_pairs_2006,
            merapi_pairs_2010,
            merapi_liquid_points,
            xlim_override=xlim_overrides.get(spec["key"], spec.get("xlim")),
            ylim_override=ylim_overrides.get(spec["key"]),
        )
    _add_top_merapi_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig, axes



# Helpers extracted from 04_apply_AIMS4PT_cpx_to_Merapi.ipynb cell 9.
def print_df_for_md_format__nb04_c09(df: pd.DataFrame):
    """
    Print a DataFrame in a markdown table format suitable for GPT.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to print.
    """
    from io import StringIO

    buffer = StringIO()
    df.to_markdown(buf=buffer, index=False)
    markdown_table = buffer.getvalue()
    print(markdown_table)


# Helpers extracted from 04_apply_AIMS4PT_cpx_to_Merapi.ipynb cell 77.
import numpy as np
import pandas as pd

def _safe_positive__nb04_c77(X: np.ndarray, eps: float) -> np.ndarray:
    """
    Replace non-positive values with eps to make log-ratio valid.
    """
    X = np.asarray(X, dtype=float)
    X[X <= 0] = eps
    return X

def clr_transform__nb04_c77(df: pd.DataFrame, comp_cols: list[str], eps: float = 1e-4) -> pd.DataFrame:
    """
    CLR transform for compositional data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    comp_cols : list[str]
        Columns treated as compositional parts (must be positive after replacement).
    eps : float
        Small value to replace zeros/negatives.
    
    Returns
    -------
    pd.DataFrame
        CLR-transformed dataframe (same index, columns named 'clr_{col}').
    """
    X = df[comp_cols].to_numpy(dtype=float)
    X = _safe_positive__nb04_c77(X, eps=eps)

    # geometric mean along parts
    g = np.exp(np.mean(np.log(X), axis=1))  # shape (n,)
    clr = np.log(X / g[:, None])            # shape (n, D)

    out_cols = [f"clr_{c}" for c in comp_cols]
    return pd.DataFrame(clr, index=df.index, columns=out_cols)

def pairwise_euclidean__nb04_c77(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between rows of A (n x d) and B (m x d).
    Returns dist matrix (n x m).
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    A2 = np.sum(A * A, axis=1)[:, None]   # (n,1)
    B2 = np.sum(B * B, axis=1)[None, :]   # (1,m)
    D2 = A2 + B2 - 2.0 * (A @ B.T)
    D2 = np.maximum(D2, 0.0)              # numerical safety
    return np.sqrt(D2)

def topk_neighbors_clr__nb04_c77(
    S_eq: pd.DataFrame,
    S_na: pd.DataFrame,
    comp_cols: list[str],
    k: int = 5,
    eps: float = 1e-4,
    eq_meta_cols: list[str] = None,
    na_meta_cols: list[str] = None,
) -> dict:
    """
    For each natural sample, find top-k nearest experimental samples
    using Euclidean distance in CLR space (Aitchison distance).
    
    Returns
    -------
    dict: {na_id: pd.DataFrame_of_topk_neighbors}
    """
    if eq_meta_cols is None:
        eq_meta_cols = []
    if na_meta_cols is None:
        na_meta_cols = []

    # Ensure required columns exist
    missing_eq = [c for c in comp_cols if c not in S_eq.columns]
    missing_na = [c for c in comp_cols if c not in S_na.columns]
    if missing_eq:
        raise ValueError(f"S_eq missing compositional columns: {missing_eq}")
    if missing_na:
        raise ValueError(f"S_na missing compositional columns: {missing_na}")
    # fill na 
    S_eq.fillna(0, inplace=True)
    S_na.fillna(0, inplace=True)

    # CLR transform
    eq_clr = clr_transform__nb04_c77(S_eq, comp_cols=comp_cols, eps=eps)
    na_clr = clr_transform__nb04_c77(S_na, comp_cols=comp_cols, eps=eps)

    # Distances (na x eq)
    dist = pairwise_euclidean__nb04_c77(na_clr.to_numpy(), eq_clr.to_numpy())
    # dist = pairwise_euclidean__nb04_c77(S_na[comp_cols].to_numpy(), S_eq[comp_cols].to_numpy())

    results = {}
    eq_index = np.array(S_eq.index)

    for i, na_id in enumerate(S_na.index):
        drow = dist[i, :]  # distances to all eq points

        # argsort distances; take top-k
        nn_idx = np.argsort(drow)[:k]
        nn_ids = eq_index[nn_idx]
        nn_dist = drow[nn_idx]

        # Build output table
        out = pd.DataFrame({
            "eq_id": nn_ids,
            "distance": nn_dist,
        })

        # attach eq metadata
        for c in eq_meta_cols:
            if c not in S_eq.columns:
                raise ValueError(f"eq_meta_cols contains missing column in S_eq: {c}")
            out[c] = S_eq.loc[nn_ids, c].to_numpy()

        # attach na metadata (repeat same value)
        for c in na_meta_cols:
            if c not in S_na.columns:
                raise ValueError(f"na_meta_cols contains missing column in S_na: {c}")
            out[f"na_{c}"] = S_na.loc[na_id, c]

        out = out.sort_values("distance", ascending=True).reset_index(drop=True)
        results[na_id] = out

    return results



# Helpers extracted from 04_apply_AIMS4PT_cpx_to_Merapi.ipynb cell 78.
import numpy as np
import pandas as pd

def _safe_positive__nb04_c78(X: np.ndarray, eps: float) -> np.ndarray:
    """
    Replace non-positive values with eps to make log-ratio valid.
    """
    X = np.asarray(X, dtype=float)
    X[X <= 0] = eps
    return X

def clr_transform__nb04_c78(df: pd.DataFrame, comp_cols: list[str], eps: float = 1e-4) -> pd.DataFrame:
    """
    CLR transform for compositional data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    comp_cols : list[str]
        Columns treated as compositional parts (must be positive after replacement).
    eps : float
        Small value to replace zeros/negatives.
    
    Returns
    -------
    pd.DataFrame
        CLR-transformed dataframe (same index, columns named 'clr_{col}').
    """
    X = df[comp_cols].to_numpy(dtype=float)
    X = _safe_positive__nb04_c78(X, eps=eps)

    # geometric mean along parts
    g = np.exp(np.mean(np.log(X), axis=1))  # shape (n,)
    clr = np.log(X / g[:, None])            # shape (n, D)

    out_cols = [f"clr_{c}" for c in comp_cols]
    return pd.DataFrame(clr, index=df.index, columns=out_cols)

def pairwise_euclidean__nb04_c78(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between rows of A (n x d) and B (m x d).
    Returns dist matrix (n x m).
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    A2 = np.sum(A * A, axis=1)[:, None]   # (n,1)
    B2 = np.sum(B * B, axis=1)[None, :]   # (1,m)
    D2 = A2 + B2 - 2.0 * (A @ B.T)
    D2 = np.maximum(D2, 0.0)              # numerical safety
    return np.sqrt(D2)

def topk_neighbors_clr__nb04_c78(
    S_eq: pd.DataFrame,
    S_na: pd.DataFrame,
    comp_cols: list[str],
    k: int = 5,
    eps: float = 1e-4,
    eq_meta_cols: list[str] = None,
    na_meta_cols: list[str] = None,
) -> dict:
    """
    For each natural sample, find top-k nearest experimental samples
    using Euclidean distance in CLR space (Aitchison distance).
    
    Returns
    -------
    dict: {na_id: pd.DataFrame_of_topk_neighbors}
    """
    if eq_meta_cols is None:
        eq_meta_cols = []
    if na_meta_cols is None:
        na_meta_cols = []

    # Ensure required columns exist
    missing_eq = [c for c in comp_cols if c not in S_eq.columns]
    missing_na = [c for c in comp_cols if c not in S_na.columns]
    if missing_eq:
        raise ValueError(f"S_eq missing compositional columns: {missing_eq}")
    if missing_na:
        raise ValueError(f"S_na missing compositional columns: {missing_na}")
    # fill na 
    S_eq.fillna(0, inplace=True)
    S_na.fillna(0, inplace=True)

    # CLR transform
    # eq_clr = clr_transform__nb04_c78(S_eq, comp_cols=comp_cols, eps=eps)
    # na_clr = clr_transform__nb04_c78(S_na, comp_cols=comp_cols, eps=eps)

    # Distances (na x eq)
    # dist = pairwise_euclidean__nb04_c78(na_clr.to_numpy(), eq_clr.to_numpy())
    dist = pairwise_euclidean__nb04_c78(S_na[comp_cols].to_numpy(), S_eq[comp_cols].to_numpy())

    results = {}
    eq_index = np.array(S_eq.index)

    for i, na_id in enumerate(S_na.index):
        drow = dist[i, :]  # distances to all eq points

        # argsort distances; take top-k
        nn_idx = np.argsort(drow)[:k]
        nn_ids = eq_index[nn_idx]
        nn_dist = drow[nn_idx]

        # Build output table
        out = pd.DataFrame({
            "eq_id": nn_ids,
            "distance": nn_dist,
        })

        # attach eq metadata
        for c in eq_meta_cols:
            if c not in S_eq.columns:
                raise ValueError(f"eq_meta_cols contains missing column in S_eq: {c}")
            out[c] = S_eq.loc[nn_ids, c].to_numpy()

        # attach na metadata (repeat same value)
        for c in na_meta_cols:
            if c not in S_na.columns:
                raise ValueError(f"na_meta_cols contains missing column in S_na: {c}")
            out[f"na_{c}"] = S_na.loc[na_id, c]

        out = out.sort_values("distance", ascending=True).reset_index(drop=True)
        results[na_id] = out

    return results

