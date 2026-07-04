"""Helper functions shared by the Merapi application notebook."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

from aims4pt.toolkit_utils import wrap_text


MERAPI_2006_COLOR = "#3995d6"
MERAPI_2010_COLOR = "#f15454"
MERAPI_PRE_2006_COLOR = "#f0c808"
INDEPENDENT_EXPERIMENTAL_COLOR = "0.62"
MERAPI_YEAR_COLORS = {"2006": MERAPI_2006_COLOR, "2010": MERAPI_2010_COLOR}
MERAPI_CIRCLE_SIZES = {"2006": 30, "2010": 38}
MERAPI_LEGEND_CIRCLE_SIZES = {"2006": 6.4, "2010": 7.2}
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


def _blend_color(color: str, target: str, amount: float) -> tuple[float, float, float]:
    """Blend a Matplotlib color toward a target color."""
    source_rgb = np.array(to_rgb(color), dtype=float)
    target_rgb = np.array(to_rgb(target), dtype=float)
    return tuple(source_rgb * (1.0 - amount) + target_rgb * amount)


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
    eruption_col = next((col for col in ["Eruption", "eruption", "eruption_year"] if col in liq.columns), None)
    if eruption_col is None:
        is_pre_2006 = pd.Series(False, index=liq.index)
    else:
        is_pre_2006 = liq[eruption_col].astype(str).str.strip().str.casefold().eq("pre-2006")
    group = liq[group_col].astype(str).str.lower().str.strip().map(
        {
            "bulk": "whole-rock endmember",
            "whole rock": "whole-rock endmember",
            "whole-rock": "whole-rock endmember",
            "glass": "glass endmember",
        }
    )
    group = group.where(~(is_pre_2006 & group.eq("whole-rock endmember")), "whole-rock (pre-2006)")
    out = liq.add_suffix("_liq")
    out["eruption_year"] = np.where(is_pre_2006.to_numpy(), "pre-2006", str(year))
    out["liquid_group"] = group.to_numpy()
    return _add_merapi_plot_columns(out)


def _drop_duplicate_pre_2006_liquid_rows(liquid_df: pd.DataFrame) -> pd.DataFrame:
    """Drop pre-2006 endmember duplicates introduced by year-specific liquid subsets."""
    if {"eruption_year", "liquid_group"}.difference(liquid_df.columns):
        return liquid_df
    pre_2006_mask = (
        liquid_df["eruption_year"].astype(str).str.casefold().eq("pre-2006")
        & liquid_df["liquid_group"].astype(str).eq("whole-rock (pre-2006)")
    )
    duplicate_mask = pre_2006_mask & liquid_df.duplicated(keep="first")
    if not duplicate_mask.any():
        return liquid_df
    return liquid_df.loc[~duplicate_mask].reset_index(drop=True)

def _find_liquid_metadata_col(df: pd.DataFrame, name: str) -> Optional[str]:
    """Return a pairing metadata column, allowing repeated liq__ prefixes."""
    candidates = [name, f"liq__{name}", f"liq__liq__{name}", f"liq__liq__liq__{name}"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    suffix = f"__{name}"
    matches = [col for col in df.columns if str(col).endswith(suffix)]
    return matches[0] if matches else None


def _liquid_group_source_col(liq_df: pd.DataFrame) -> str:
    group_col = next((col for col in ["glass/bulk", "glass_bulk", "liquid_group"] if col in liq_df.columns), None)
    if group_col is None:
        raise KeyError("Merapi liquid endmember table must contain a glass/bulk group column.")
    return group_col


def _liquid_endmember_subsets(liq_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return bulk and glass endmember pools in the same positional order used by pairing."""
    group_col = _liquid_group_source_col(liq_df)
    group = liq_df[group_col].astype(str).str.lower().str.strip()
    bulk = liq_df.loc[group.isin(["bulk", "whole rock", "whole-rock"])].reset_index(drop=True)
    glass = liq_df.loc[group.eq("glass")].reset_index(drop=True)
    return bulk, glass


def _valid_positional_indices(values: pd.Series, upper_bound: int) -> np.ndarray:
    idx = pd.to_numeric(values, errors="coerce").dropna().astype(int).to_numpy()
    idx = idx[(idx >= 0) & (idx < upper_bound)]
    return np.unique(idx)


def _pre_2006_liquid_mask(rows: pd.DataFrame) -> pd.Series:
    """Return rows whose original liquid eruption label is pre-2006."""
    eruption_col = next((col for col in ["Eruption", "eruption", "eruption_year"] if col in rows.columns), None)
    if eruption_col is None:
        return pd.Series(False, index=rows.index)
    return rows[eruption_col].astype(str).str.strip().str.casefold().eq("pre-2006")


def _frame_liquid_endmembers_for_plot(rows: pd.DataFrame, year: str, group: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    eruption_year = pd.Series(str(year), index=rows.index, dtype=object)
    liquid_group = pd.Series(group, index=rows.index, dtype=object)
    if group == "whole-rock endmember":
        pre_2006_mask = _pre_2006_liquid_mask(rows)
        eruption_year = eruption_year.where(~pre_2006_mask, "pre-2006")
        liquid_group = liquid_group.where(~pre_2006_mask, "whole-rock (pre-2006)")
    out = rows.copy().add_suffix("_liq")
    out["eruption_year"] = eruption_year.to_numpy()
    out["liquid_group"] = liquid_group.to_numpy()
    return _add_merapi_plot_columns(out)


def _used_liquid_endmember_frame(liq_df: pd.DataFrame, paired_liq_df: pd.DataFrame, year: str) -> pd.DataFrame:
    """Return only bulk/glass endmembers used by final accepted synthetic liquids."""
    bulk_idx_col = _find_liquid_metadata_col(paired_liq_df, "endmember1_idx")
    glass_idx_col = _find_liquid_metadata_col(paired_liq_df, "endmember2_idx")
    if bulk_idx_col is None or glass_idx_col is None:
        return _liquid_endmember_frame(liq_df, year)

    bulk, glass = _liquid_endmember_subsets(liq_df)
    bulk_idx = _valid_positional_indices(paired_liq_df[bulk_idx_col], len(bulk))
    glass_idx = _valid_positional_indices(paired_liq_df[glass_idx_col], len(glass))
    frames = [
        _frame_liquid_endmembers_for_plot(bulk.iloc[bulk_idx], year, "whole-rock endmember"),
        _frame_liquid_endmembers_for_plot(glass.iloc[glass_idx], year, "glass endmember"),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _liquid_numeric_mix_columns(bulk: pd.DataFrame, glass: pd.DataFrame) -> list[str]:
    cols = []
    for col in bulk.columns.intersection(glass.columns):
        if col in {"Eruption", "eruption", "eruption_year", "glass/bulk", "glass_bulk", "liquid_group"}:
            continue
        bulk_values = pd.to_numeric(bulk[col], errors="coerce")
        glass_values = pd.to_numeric(glass[col], errors="coerce")
        if bulk_values.notna().any() and glass_values.notna().any():
            cols.append(col)
    return cols


def _liquid_mix_path_frame(
    liq_df: pd.DataFrame,
    paired_liq_df: pd.DataFrame,
    year: str,
    *,
    n_points: int = 15,
) -> pd.DataFrame:
    """Return points along each accepted bulk-glass synthetic mixing path."""
    bulk_idx_col = _find_liquid_metadata_col(paired_liq_df, "endmember1_idx")
    glass_idx_col = _find_liquid_metadata_col(paired_liq_df, "endmember2_idx")
    if bulk_idx_col is None or glass_idx_col is None:
        return pd.DataFrame()

    bulk, glass = _liquid_endmember_subsets(liq_df)
    if bulk.empty or glass.empty:
        return pd.DataFrame()
    mix_cols = _liquid_numeric_mix_columns(bulk, glass)
    if not mix_cols:
        return pd.DataFrame()

    pair_df = paired_liq_df[[bulk_idx_col, glass_idx_col]].copy()
    pair_df[bulk_idx_col] = pd.to_numeric(pair_df[bulk_idx_col], errors="coerce")
    pair_df[glass_idx_col] = pd.to_numeric(pair_df[glass_idx_col], errors="coerce")
    pair_df = pair_df.dropna().astype(int).drop_duplicates()
    pair_df = pair_df[
        pair_df[bulk_idx_col].between(0, len(bulk) - 1)
        & pair_df[glass_idx_col].between(0, len(glass) - 1)
    ]
    if pair_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, float]] = []
    f_values = np.linspace(0.0, 1.0, n_points)
    for path_id, (bulk_idx, glass_idx) in enumerate(pair_df[[bulk_idx_col, glass_idx_col]].itertuples(index=False, name=None)):
        bulk_row = pd.to_numeric(bulk.iloc[int(bulk_idx)][mix_cols], errors="coerce")
        glass_row = pd.to_numeric(glass.iloc[int(glass_idx)][mix_cols], errors="coerce")
        for f_value in f_values:
            mixed = f_value * bulk_row + (1.0 - f_value) * glass_row
            row = mixed.to_dict()
            row["mixing_path_id"] = path_id
            row["mix_f"] = f_value
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).add_suffix("_liq")
    out["mixing_path_id"] = out.pop("mixing_path_id_liq").astype(int)
    out["mix_f"] = out.pop("mix_f_liq").astype(float)
    out["eruption_year"] = str(year)
    out["liquid_group"] = "mixing path"
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


def _draw_kde_mass_contours(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlim,
    ylim,
    *,
    label_positions: Mapping[str, tuple[float, ...]] | None = None,
    label_fontsize: float = 9.5,
):
    """Draw solid KDE contours enclosing selected cumulative probability masses."""
    from scipy.stats import gaussian_kde

    x, y = _finite_xy(df, x_col, y_col)
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0 or xlim is None or ylim is None:
        return
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 180), np.linspace(ylim[0], ylim[1], 180))
    try:
        zz = gaussian_kde(np.vstack([x, y]))(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    except np.linalg.LinAlgError:
        return

    sigma_masses = [("95.4%", 0.954), ("68.3%", 0.683)]
    level_items = []
    for label, mass in sigma_masses:
        level = _kde_level_for_mass(zz, mass)
        if np.isfinite(level):
            level_items.append((level, label))
    if not level_items:
        return

    level_items = sorted(level_items, key=lambda item: item[0])
    unique_items = []
    for level, label in level_items:
        if not unique_items or not np.isclose(level, unique_items[-1][0]):
            unique_items.append((level, label))
    levels = [item[0] for item in unique_items]
    fmt = {level: label for level, label in unique_items}
    if label_positions:
        ax.contour(xx, yy, zz, levels=levels, colors="0.40", linewidths=0.95, linestyles="-", zorder=1)
        for _, label in unique_items:
            position = label_positions.get(label)
            if position is None:
                continue
            x_text, y_text = position[:2]
            rotation = position[2] if len(position) > 2 else 0.0
            ax.text(
                x_text,
                y_text,
                label,
                fontsize=label_fontsize,
                color="0.25",
                rotation=rotation,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
                zorder=7,
            )
        return
    contour = ax.contour(xx, yy, zz, levels=levels, colors="0.40", linewidths=0.95, linestyles="-", zorder=1)
    ax.clabel(contour, levels=levels, fmt=fmt, inline=True, fontsize=label_fontsize, colors="0.25")


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
            s=MERAPI_CIRCLE_SIZES[year],
            facecolors=MERAPI_YEAR_COLORS[year],
            edgecolors="0.15",
            linewidths=0.45,
            alpha=0.92,
            zorder=zorder,
        )


LIQUID_GROUP_STYLES = {
    "mixing path": {
        "marker": "o",
        "size": 4.8,
        "filled": True,
        "edgecolor": "none",
        "alpha": 0.52,
        "linewidth": 0.7,
        "color_blend": ("white", 0.34),
    },
    "whole-rock endmember": {"marker": "s", "size": 70, "filled": True, "edgecolor": "black", "alpha": 0.96},
    "whole-rock (pre-2006)": {
        "marker": "s",
        "size": 70,
        "filled": True,
        "facecolor": MERAPI_PRE_2006_COLOR,
        "edgecolor": "black",
        "alpha": 0.96,
    },
    "glass endmember": {"marker": "^", "size": 58, "filled": True, "edgecolor": "black", "alpha": 0.96},
    "equilibrium liquid": {
        "marker": "x",
        "size": 22,
        "filled": False,
        "linewidth": 0.8,
        "alpha": 1.0,
        "color_blend": ("black", 0.32),
    },
}


def _liquid_group_color(year: str, style: Mapping[str, Any]) -> str | tuple[float, float, float]:
    """Return the plotted color for a liquid role and eruption year."""
    color = style.get("facecolor", MERAPI_YEAR_COLORS.get(year, MERAPI_PRE_2006_COLOR))
    if "color_blend" not in style:
        return color
    target, amount = style["color_blend"]
    return _blend_color(color, target, amount)


def _plot_liquid_groups(
    ax,
    liquid_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    mixing_path_display: str = "points",
):
    """Plot Merapi liquids with year encoded by color and liquid role by marker."""
    layer_order = [
        ("2010", "mixing path", 3.0),
        ("2006", "mixing path", 3.1),
        ("2010", "whole-rock endmember", 5.0),
        ("2006", "whole-rock endmember", 5.1),
        ("pre-2006", "whole-rock (pre-2006)", 5.2),
        ("2010", "glass endmember", 5.3),
        ("2006", "glass endmember", 5.4),
        ("2010", "equilibrium liquid", 6.1),
        ("2006", "equilibrium liquid", 6.2),
    ]
    for year, group, zorder in layer_order:
        style = LIQUID_GROUP_STYLES[group]
        color = _liquid_group_color(year, style)
        data = liquid_df[
            liquid_df["eruption_year"].astype(str).eq(year)
            & liquid_df["liquid_group"].astype(str).eq(group)
        ]
        if data.empty:
            continue
        if group == "mixing path" and mixing_path_display == "curves" and "mixing_path_id" in data.columns:
            for _, path_data in data.groupby("mixing_path_id", sort=False):
                path_data = path_data.sort_values("mix_f") if "mix_f" in path_data.columns else path_data
                xy = _finite_xy_frame(path_data, x_col, y_col)
                if len(xy) < 2:
                    continue
                ax.plot(
                    xy[x_col].to_numpy(dtype=float),
                    xy[y_col].to_numpy(dtype=float),
                    color=color,
                    linewidth=style.get("linewidth", 0.7),
                    alpha=style.get("alpha", 0.34),
                    zorder=zorder,
                )
            continue
        x, y = _finite_xy(data, x_col, y_col)
        marker = style["marker"]
        if marker in {"x", "+"}:
            ax.scatter(
                x,
                y,
                marker=marker,
                s=style["size"],
                color=color,
                linewidths=style.get("linewidth", 0.85),
                alpha=style.get("alpha", 0.98),
                zorder=zorder,
            )
            continue
        ax.scatter(
            x,
            y,
            marker=marker,
            s=style["size"],
            facecolors=color,
            edgecolors=style.get("edgecolor", "black"),
            linewidths=style.get("linewidth", 0.55 if group == "mixing path" else 0.95),
            alpha=style.get("alpha", 0.9),
            zorder=zorder,
        )


def _add_top_merapi_legend(fig):
    """Add the global eruption/dataset legend for fig. 7."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MERAPI_2006_COLOR, markeredgecolor="0.15", markersize=MERAPI_LEGEND_CIRCLE_SIZES["2006"], label="2006 eruption"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MERAPI_2010_COLOR, markeredgecolor="0.15", markersize=MERAPI_LEGEND_CIRCLE_SIZES["2010"], label="2010 eruption"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INDEPENDENT_EXPERIMENTAL_COLOR, markeredgecolor="none", markersize=7.5, label="Independent experimental dataset"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=True,
        fontsize=10,
        handletextpad=0.6,
        borderaxespad=0.0,
        labelspacing=0.4,
    )


def _add_liquid_group_legend(ax, loc="upper right", bbox_to_anchor=None, *, mixing_path_display: str = "points"):
    """Add a local liquid-role legend to a liquid composition panel."""
    from matplotlib.lines import Line2D

    def group_color(year: str, group: str) -> str | tuple[float, float, float]:
        return _liquid_group_color(year, LIQUID_GROUP_STYLES[group])

    def year_tuple(marker: str, *, size: float, fill: bool = True, lw: float = 0.9, group: str | None = None):
        handles = []
        for year in ("2006", "2010"):
            color = group_color(year, group) if group is not None else MERAPI_YEAR_COLORS[year]
            if marker in {"x", "+"}:
                handles.append(
                    Line2D([0], [0], marker=marker, color=color, linestyle="None", markeredgewidth=lw, markersize=size)
                )
            else:
                handles.append(
                    Line2D(
                        [0], [0],
                        marker=marker,
                        color="none",
                        markerfacecolor=color if fill else "none",
                        markeredgecolor="black" if fill else color,
                        markeredgewidth=lw,
                        markersize=size,
                    )
                )
        return tuple(handles)

    def mixing_path_tuple():
        return tuple(
            Line2D(
                [0.0, 1.0],
                [0.0, 0.0],
                color=group_color(year, "mixing path"),
                linewidth=1.1,
                alpha=0.7,
            )
            for year in ("2006", "2010")
        )

    handles = [
        year_tuple("s", size=6.8, fill=True, lw=0.95),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=MERAPI_PRE_2006_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.95,
            markersize=6.8,
        ),
        year_tuple("^", size=6.2, fill=True, lw=0.95),
        mixing_path_tuple() if mixing_path_display == "curves" else year_tuple("o", size=2.9, fill=True, lw=0.0, group="mixing path"),
        year_tuple("x", size=4.9, fill=False, lw=0.8, group="equilibrium liquid"),
    ]
    labels = [
        "whole-rock endmember",
        "whole-rock (pre-2006)",
        "glass endmember",
        "bulk-glass mixing path",
        "equilibrium liquid",
    ]
    ax.legend(
        handles=handles,
        labels=labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        fontsize=7.5,
        handlelength=1.6,
        handletextpad=0.45,
        labelspacing=0.25,
        borderpad=0.25,
        borderaxespad=0.25,
        handler_map={tuple: HandlerTuple(ndivide=None)},
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
    mixing_path_display: str = "points",
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
    _draw_kde_mass_contours(
        ax,
        independent_pairs,
        x_col,
        y_col,
        xlim_for_kde,
        ylim_for_kde,
        label_positions=spec.get("kde_label_positions"),
        label_fontsize=spec.get("kde_label_fontsize", 9.5),
    )
    _plot_independent_background(ax, independent_pairs, x_col, y_col)
    if spec["kind"] == "liquid":
        _plot_liquid_groups(ax, merapi_liquid_points, x_col, y_col, mixing_path_display=mixing_path_display)
        _add_liquid_group_legend(
            ax,
            loc=spec.get("legend_loc", "upper right"),
            bbox_to_anchor=spec.get("legend_bbox"),
            mixing_path_display=mixing_path_display,
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
    mixing_path_display: str = "points",
):
    """Plot Merapi cpx/liquid compositions against the independent experimental dataset."""
    import matplotlib.pyplot as plt

    if mixing_path_display not in {"points", "curves"}:
        raise ValueError("mixing_path_display must be either 'points' or 'curves'.")

    xlim_overrides = xlim_overrides or {}
    ylim_overrides = ylim_overrides or {}
    independent_pairs = _add_merapi_plot_columns(unseen_experiments_df)
    merapi_pairs_2006 = _add_merapi_plot_columns(_merapi_pair_frame(paired_cpx_06_pass, paired_liq_06_pass))
    merapi_pairs_2010 = _add_merapi_plot_columns(_merapi_pair_frame(paired_cpx_10_pass, paired_liq_10_pass))
    merapi_pairs_2006["eruption_year"] = "2006"
    merapi_pairs_2010["eruption_year"] = "2010"
    merapi_pairs_2006["liquid_group"] = "equilibrium liquid"
    merapi_pairs_2010["liquid_group"] = "equilibrium liquid"
    liquid_frames = [
        _used_liquid_endmember_frame(liq_2006, paired_liq_06_pass, "2006"),
        _used_liquid_endmember_frame(liq_2010, paired_liq_10_pass, "2010"),
        _liquid_mix_path_frame(liq_2006, paired_liq_06_pass, "2006"),
        _liquid_mix_path_frame(liq_2010, paired_liq_10_pass, "2010"),
        merapi_pairs_2006,
        merapi_pairs_2010,
    ]
    liquid_frames = [frame for frame in liquid_frames if frame is not None and not frame.empty]
    merapi_liquid_points = pd.concat(liquid_frames, ignore_index=True)
    merapi_liquid_points = _drop_duplicate_pre_2006_liquid_rows(merapi_liquid_points)

    panel_specs = [
        {
            "key": "cpx_cao_mg_number",
            "kind": "cpx",
            "label": "(a)",
            "x_col": "CaO_cpx",
            "y_col": "MgNumber_cpx",
            "xlabel": r"Clinopyroxene CaO (wt%)",
            "ylabel": r"Clinopyroxene Mg#",
            "kde_label_positions": {"68.3%": (15.7, 0.96, -16), "95.4%": (3.9, 0.65, 0)},
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
            "kde_label_positions": {"68.3%": (0.58, 6.8, -24), "95.4%": (0.78, 4.8, 28)},
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
            "kde_label_positions": {"68.3%": (73.5, 8.25, 12), "95.4%": (75.0, 10.2, 0)},
        },
        {
            "key": "liquid_sio2_mg_number",
            "kind": "liquid",
            "label": "(d)",
            "x_col": "SiO2_liq",
            "y_col": "MgNumber_liq",
            "xlabel": r"Liquid SiO$_2$ (wt%)",
            "ylabel": r"Liquid Mg#",
            "kde_label_positions": {"68.3%": (60.5, 0.62, 0), "95.4%": (50.5, 0.96, -18)},
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
            mixing_path_display=mixing_path_display,
        )
    _add_top_merapi_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    plt.show()
    return fig, axes


# Ranked Merapi thermobarometry figure helpers.
MANUSCRIPT_AXIS_LABEL_SIZE = 19
MANUSCRIPT_Y_AXIS_LABEL_SIZE = 20
MANUSCRIPT_TICK_LABEL_SIZE = 16
MANUSCRIPT_MODEL_TICK_LABEL_SIZE = 17
MANUSCRIPT_GROUP_LABEL_SIZE = 19
MANUSCRIPT_LEGEND_SIZE = 16
MANUSCRIPT_PANEL_LABEL_SIZE = 20
MANUSCRIPT_ANNOTATION_SIZE = 15
MANUSCRIPT_RESERVOIR_LABEL_SIZE = 17
MANUSCRIPT_THIS_STUDY_AXIS_LABEL_SIZE = 21
MANUSCRIPT_THIS_STUDY_Y_AXIS_LABEL_SIZE = 24
MANUSCRIPT_THIS_STUDY_TICK_LABEL_SIZE = 17
MANUSCRIPT_THIS_STUDY_MODEL_TICK_LABEL_SIZE = 22
MANUSCRIPT_THIS_STUDY_GROUP_LABEL_SIZE = 23
MANUSCRIPT_THIS_STUDY_ANNOTATION_SIZE = 17
MANUSCRIPT_THIS_STUDY_LEGEND_SIZE = 17

@dataclass
class ThermobarometryWorkflowBundle:
    """
    Container for one workflow family across pressure/temperature and eruption year.

    Parameters
    ----------
    pressure_2006, pressure_2010, temperature_2006, temperature_2010
        Workflow objects that expose ``prediction_df`` and
        ``get_best_model_series()``.

    Notes
    -----
    This class is optional. The plotting helpers below also accept dictionaries
    with equivalent keys such as ``pressure_2006`` / ``temperature_2010`` or a
    nested form like ``{"P": {"2006": ...}, "T": {"2010": ...}}``.
    """

    pressure_2006: Any
    pressure_2010: Any
    temperature_2006: Any
    temperature_2010: Any


def _coerce_workflow_bundle(bundle: Any, label: str) -> ThermobarometryWorkflowBundle:
    if isinstance(bundle, ThermobarometryWorkflowBundle):
        return bundle

    if not isinstance(bundle, Mapping):
        raise TypeError(
            f"{label} must be a ThermobarometryWorkflowBundle or mapping, got {type(bundle)!r}."
        )

    flat_keys = {"pressure_2006", "pressure_2010", "temperature_2006", "temperature_2010"}
    if flat_keys.issubset(bundle):
        return ThermobarometryWorkflowBundle(
            pressure_2006=bundle["pressure_2006"],
            pressure_2010=bundle["pressure_2010"],
            temperature_2006=bundle["temperature_2006"],
            temperature_2010=bundle["temperature_2010"],
        )

    short_keys = {"P_2006", "P_2010", "T_2006", "T_2010"}
    if short_keys.issubset(bundle):
        return ThermobarometryWorkflowBundle(
            pressure_2006=bundle["P_2006"],
            pressure_2010=bundle["P_2010"],
            temperature_2006=bundle["T_2006"],
            temperature_2010=bundle["T_2010"],
        )

    if "P" in bundle and "T" in bundle:
        pressure_map = bundle["P"]
        temperature_map = bundle["T"]
        if not isinstance(pressure_map, Mapping) or not isinstance(temperature_map, Mapping):
            raise TypeError(f"{label}['P'] and {label}['T'] must be mappings.")

        def _pick_year(mapping: Mapping[Any, Any], year: str) -> Any:
            if year in mapping:
                return mapping[year]
            year_int = int(year)
            if year_int in mapping:
                return mapping[year_int]
            raise KeyError(year)

        return ThermobarometryWorkflowBundle(
            pressure_2006=_pick_year(pressure_map, "2006"),
            pressure_2010=_pick_year(pressure_map, "2010"),
            temperature_2006=_pick_year(temperature_map, "2006"),
            temperature_2010=_pick_year(temperature_map, "2010"),
        )

    raise KeyError(
        f"{label} must provide either flat keys {sorted(flat_keys)}, short keys {sorted(short_keys)}, "
        "or nested keys {'P': {'2006', '2010'}, 'T': {'2006', '2010'}}."
    )


def _coerce_phase_column_groups(columns: Mapping[str, Sequence[str]], label: str) -> dict[str, list[str]]:
    if not isinstance(columns, Mapping):
        raise TypeError(f"{label} must be a mapping with 'cpx_only' and 'cpx_liq' keys.")

    required = {"cpx_only", "cpx_liq"}
    missing = required.difference(columns)
    if missing:
        raise KeyError(f"{label} is missing required keys: {sorted(missing)}")

    return {
        "cpx_only": list(columns["cpx_only"]),
        "cpx_liq": list(columns["cpx_liq"]),
    }


def _get_prediction_df_from_workflow(workflow: Any, workflow_label: str) -> pd.DataFrame:
    if not hasattr(workflow, "prediction_df"):
        raise AttributeError(f"{workflow_label} does not expose prediction_df.")
    prediction_df = workflow.prediction_df
    if not isinstance(prediction_df, pd.DataFrame):
        raise TypeError(f"{workflow_label}.prediction_df must be a pandas DataFrame.")
    return prediction_df


def _get_best_model_series_from_workflow(workflow: Any, workflow_label: str) -> pd.Series:
    if not hasattr(workflow, "get_best_model_series"):
        raise AttributeError(f"{workflow_label} does not expose get_best_model_series().")
    best_model_series = workflow.get_best_model_series()
    if not isinstance(best_model_series, pd.Series):
        raise TypeError(f"{workflow_label}.get_best_model_series() must return a pandas Series.")
    return best_model_series


def _extract_plot_arrays(df: pd.DataFrame, columns: Sequence[str], *, df_label: str) -> list[np.ndarray]:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{df_label} is missing required columns: {missing}")
    return [df[col].dropna().to_numpy() for col in columns]


def _build_uncertainty_dict(model_pool: Optional[Sequence[Any]]) -> dict[str, float]:
    if model_pool is None:
        return {}
    return {
        getattr(model, "model_name", str(i)): getattr(model, "uncertainty", np.nan)
        for i, model in enumerate(model_pool)
    }


def _remove_boxplot_outliers(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size <= 3:
        return arr

    q1 = np.nanpercentile(arr, 25)
    q3 = np.nanpercentile(arr, 75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return arr

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return arr[(arr >= lower) & (arr <= upper)]


def _filter_data_groups(data_groups: Sequence[Sequence[float]]) -> list[np.ndarray]:
    return [_remove_boxplot_outliers(values) for values in data_groups]


def _coerce_result_array(values: Any, label: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        series = values
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] == 1:
            series = values.iloc[:, 0]
        elif "prediction" in values.columns:
            series = values["prediction"]
        else:
            raise TypeError(f"{label} must be a Series, array-like, or single-column DataFrame.")
    else:
        try:
            series = pd.Series(np.asarray(values, dtype=float).ravel())
        except Exception as exc:
            raise TypeError(f"{label} must be numeric and array-like.") from exc

    numeric = pd.to_numeric(series, errors="coerce")
    return _remove_boxplot_outliers(numeric.dropna().to_numpy())


def _compute_rank_pcts_from_best(best_model_series: pd.Series) -> list[tuple[str, float]]:
    counts = best_model_series.dropna().astype(str).value_counts()
    total = counts.sum()
    if total == 0:
        return []
    pcts = counts / total * 100.0
    return list(zip(pcts.index.tolist(), pcts.values.tolist()))


def _model_to_index(columns: Sequence[str], model_name: str) -> Optional[int]:
    try:
        return list(columns).index(model_name)
    except ValueError:
        return None


def _pick_models_for_single_eruption(
    rank_pcts: Sequence[tuple[str, float]],
    *,
    threshold: float,
    model_selection_mode: str = "cumulative",
    cumulative_threshold: float = 90.0,
) -> list[str]:
    selected_ranks = _selected_rank_set_from_topk(
        rank_pcts,
        threshold=threshold,
        model_selection_mode=model_selection_mode,
        cumulative_threshold=cumulative_threshold,
    )
    return [model_name for rank_i, (model_name, _) in enumerate(rank_pcts) if rank_i in selected_ranks]


def _selected_rank_set_from_topk(
    topk: Sequence[tuple[str, float]],
    *,
    threshold: float,
    model_selection_mode: str = "cumulative",
    cumulative_threshold: float = 90.0,
) -> set[int]:
    if not topk:
        return set()

    mode = str(model_selection_mode).strip().lower()
    if mode == "top1_or_top2":
        if float(topk[0][1]) > threshold or len(topk) == 1:
            return {0}
        return {0, 1}

    if mode in {"above_threshold", "gt_threshold", "all_gt_threshold"}:
        return {
            rank_i
            for rank_i, (_, pct) in enumerate(topk)
            if float(pct) > float(threshold)
        }

    if mode == "cumulative":
        selected = set()
        cumulative_pct = 0.0
        target_pct = float(cumulative_threshold)
        for rank_i, (_, pct) in enumerate(topk):
            selected.add(rank_i)
            cumulative_pct += float(pct)
            if cumulative_pct >= target_pct:
                break
        return selected

    raise ValueError(
        "model_selection_mode must be 'cumulative', 'top1_or_top2', or 'above_threshold'."
    )


def _finite_uncertainty(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except Exception:
        return None
    if not np.isfinite(value) or value < 0:
        return None
    return value


def _clean_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _strip_parens(value: Any) -> str:
    text = _clean_str(value)
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text


def _wrap_label(value: Any, width: int = 12, *, allow_word_break: bool = False) -> str:
    return wrap_text(str(value), max_width=width, allow_word_break=allow_word_break)


def _apply_threshold_xtick_rotation(
    ax: plt.Axes,
    raw_labels: Sequence[Any],
    *,
    rotation_threshold: int = 22,
    long_label_fontsize_scale: float = 0.86,
) -> None:
    for tick_label, raw_label in zip(ax.get_xticklabels(), raw_labels):
        tick_label.set_rotation(0)
        tick_label.set_ha("center")
        tick_label.set_va("top")
        if len(str(raw_label)) > rotation_threshold:
            tick_label.set_fontsize(tick_label.get_fontsize() * long_label_fontsize_scale)
        tick_label.set_linespacing(0.95)


def _pick_kind_mapping(mapping: Optional[Mapping[str, Any]], kind: str) -> Optional[Mapping[str, Any]]:
    if mapping is None:
        return None

    direct_value = mapping.get(kind)
    if direct_value is not None:
        return direct_value

    alias = "pressure" if kind == "P" else "temperature"
    alias_value = mapping.get(alias)
    if alias_value is not None:
        return alias_value

    return None


def _coerce_liquid_kind_results(kind: str, liquid_results: Optional[Mapping[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    kind_mapping = _pick_kind_mapping(liquid_results, kind)
    if kind_mapping is None:
        return {}
    if not isinstance(kind_mapping, Mapping):
        raise TypeError(f"liquid_results[{kind!r}] must be a mapping.")

    coerced: dict[str, dict[str, np.ndarray]] = {}
    for year in ("2006", "2010"):
        year_mapping = kind_mapping.get(year, kind_mapping.get(int(year)))
        if year_mapping is None:
            continue
        if not isinstance(year_mapping, Mapping):
            raise TypeError(f"liquid_results[{kind!r}][{year!r}] must be a mapping.")

        phase_results: dict[str, np.ndarray] = {}
        for phase_name in ("glass", "bulk"):
            if phase_name not in year_mapping or year_mapping[phase_name] is None:
                continue
            phase_results[phase_name] = _coerce_result_array(
                year_mapping[phase_name],
                f"liquid_results[{kind!r}][{year!r}][{phase_name!r}]",
            )

        if phase_results:
            coerced[year] = phase_results

    return coerced


def _liquid_prediction_uncertainty(kind: str, phase_name: str) -> float:
    uncertainty_map = {
        ("P", "glass"): 1.66,
        ("P", "bulk"): 1.29,
        ("T", "glass"): 31.9,
        ("T", "bulk"): 26.2,
    }
    return uncertainty_map[(kind, phase_name)]


def _short_model_name(model_name: str) -> str:
    return _clean_str(model_name).split("(")[0].strip()


def _model_axis_label(model_name: str, kind: Optional[str] = None, *, use_model_abbreviations: bool = False) -> str:
    if not use_model_abbreviations:
        return _short_model_name(model_name)
    if kind is None:
        return _short_model_name(model_name)
    try:
        from paper.scripts.constants_illustration import get_model_abbreviation
    except Exception:
        return _short_model_name(model_name)
    label = get_model_abbreviation(model_name, kind)
    if label == model_name:
        return _short_model_name(model_name)
    if label.startswith("Pu08_"):
        return "Pu08\n_" + label.split("_", 1)[1]
    return label


def _display_phase_type_label(value: Any) -> str:
    label = _clean_str(value)
    key = label.lower().replace("_", "-").replace(" ", "-")
    if key in {"cpx", "clinopyroxene"}:
        return "Clinopyroxene"
    if key in {"cpx-only", "clinopyroxene-only"}:
        return "Clinopyroxene-only"
    if key in {"cpx-liq", "cpx-liquid", "clinopyroxene-liquid"}:
        return "Clinopyroxene-liquid"
    if key in {"amph", "amphibole"}:
        return "Amphibole"
    if key in {"amph-only", "amphibole-only"}:
        return "Amphibole-only"
    if key in {"amph-liq", "amph-liquid", "amphibole-liquid"}:
        return "Amphibole-liquid"
    if key == "melt-inclusion":
        return "Melt\ninclusion"
    return label


def _style_bp_item(
    bp: Mapping[str, Any],
    box_idx: int,
    facecolor: str,
    edgecolor: str = "k",
    lw: float = 1.2,
    alpha: float = 1.0,
    median_color: Optional[str] = None,
    median_lw: Optional[float] = None,
) -> None:
    box = bp["boxes"][box_idx]
    box.set_facecolor(facecolor)
    box.set_alpha(alpha)
    box.set_edgecolor(edgecolor)
    box.set_linewidth(lw)
    box.set_zorder(3)

    median = bp["medians"][box_idx]
    median.set_color(median_color if median_color is not None else edgecolor)
    median.set_linewidth(median_lw if median_lw is not None else lw)
    median.set_zorder(4)

    whisker_low = bp["whiskers"][2 * box_idx]
    whisker_high = bp["whiskers"][2 * box_idx + 1]
    cap_low = bp["caps"][2 * box_idx]
    cap_high = bp["caps"][2 * box_idx + 1]
    for obj in (whisker_low, whisker_high, cap_low, cap_high):
        obj.set_color(edgecolor)
        obj.set_linewidth(lw)
        obj.set_zorder(4)


def _draw_violin(
    ax: plt.Axes,
    data_list: Sequence[Sequence[float]],
    positions: Sequence[float],
    *,
    widths: float = 0.28,
    facecolor: str = "blue",
    edgecolor: str = "k",
    lw: float = 1.2,
    alpha: float = 1.0,
    bw_method: float = 0.25,
    zorder: float = 3,
    show_quartile_interval: bool = True,
) -> Mapping[str, Any]:
    vp = ax.violinplot(
        data_list,
        positions=positions,
        widths=widths,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        bw_method=bw_method,
    )
    for body in vp["bodies"]:
        body.set_facecolor(facecolor)
        body.set_edgecolor(edgecolor)
        body.set_linewidth(lw)
        body.set_alpha(alpha)
        body.set_zorder(zorder)

    if "cmedians" in vp:
        vp["cmedians"].set_color("k")
        vp["cmedians"].set_linewidth(max(lw * 1.35, 1.8))
        vp["cmedians"].set_alpha(1.0)
        vp["cmedians"].set_zorder(zorder + 2.0)
        vp["cmedians"].set_path_effects([
            pe.Stroke(linewidth=max(lw * 2.0, 2.6), foreground="white", alpha=0.85),
            pe.Normal(),
        ])

    if not show_quartile_interval:
        return vp

    if np.isscalar(widths):
        width_values = np.full(len(positions), float(widths), dtype=float)
    else:
        width_values = np.asarray(widths, dtype=float).ravel()
        if width_values.size == 1 and len(positions) != 1:
            width_values = np.full(len(positions), float(width_values[0]), dtype=float)
    for values, position, width_value in zip(data_list, positions, width_values):
        _redraw_violin_quartiles(
            ax,
            float(position),
            values,
            float(width_value),
            color="k",
            lw=max(lw * 0.9, 1.05),
            zorder=zorder + 1.05,
            alpha=min(alpha, 0.95),
            filter_outliers=False,
        )
    return vp


def _add_vertical_uncertainty_band(
    ax: plt.Axes,
    x_center: float,
    values: Sequence[float],
    uncertainty: Any,
    width: float,
    color: str,
    *,
    alpha: float = 0.16,
    zorder: float = 2,
) -> None:
    finite_uncertainty = _finite_uncertainty(uncertainty)
    clean_values = _remove_boxplot_outliers(values)
    if finite_uncertainty is None or clean_values.size == 0:
        return

    median = float(np.nanmedian(clean_values))
    x0 = float(x_center) - width / 2.0
    x1 = float(x_center) + width / 2.0
    y0 = median - finite_uncertainty
    y1 = median + finite_uncertainty
    ax.fill_between([x0, x1], [y0, y0], [y1, y1], color=color, alpha=alpha, zorder=zorder, linewidth=0)
    ax.hlines(median, x0, x1, color=color, lw=1.8, zorder=zorder + 0.2)


def _add_vertical_uncertainty_band_from_median(
    ax: plt.Axes,
    x_center: float,
    median: float,
    uncertainty: Any,
    width: float,
    color: str,
    *,
    alpha: float = 0.16,
    zorder: float = 2,
    median_marker: str = "line",
) -> None:
    finite_uncertainty = _finite_uncertainty(uncertainty)
    if finite_uncertainty is None or median is None or not np.isfinite(median):
        return

    x0 = float(x_center) - width / 2.0
    x1 = float(x_center) + width / 2.0
    y0 = float(median) - finite_uncertainty
    y1 = float(median) + finite_uncertainty
    ax.fill_between([x0, x1], [y0, y0], [y1, y1], color=color, alpha=alpha, zorder=zorder, linewidth=0)
    if median_marker == "dot":
        ax.scatter([float(x_center)], [float(median)], s=22, color=color, zorder=zorder + 0.3)
    else:
        ax.hlines(float(median), x0 + 0.28 * width, x1 - 0.28 * width, color=color, lw=1.4, zorder=zorder + 0.2)


def _add_subgroup_background(
    ax: plt.Axes,
    x_center: float,
    width: float,
    color: str,
    *,
    alpha: float = 0.12,
    zorder: float = 1,
) -> None:
    x0 = float(x_center) - width / 2.0
    x1 = float(x_center) + width / 2.0
    ax.axvspan(x0, x1, color=color, alpha=alpha, zorder=zorder)


def _style_selected_violin(
    vp: Mapping[str, Any],
    body_idx: int,
    *,
    edgecolor: str = "k",
    facecolor: Optional[str] = None,
    lw: float = 3.0,
    linestyle: str = "--",
) -> None:
    body = vp["bodies"][body_idx]
    if facecolor is not None:
        body.set_facecolor(facecolor)
    body.set_edgecolor(edgecolor)
    body.set_linewidth(lw)
    body.set_linestyle(linestyle)
    body.set_alpha(1.0)
    body.set_zorder(4.2)


def _redraw_violin_median(
    ax: plt.Axes,
    x_center: float,
    values: Sequence[float],
    width: float,
    *,
    color: str = "k",
    lw: float = 1.8,
    zorder: float = 4.6,
    linestyle: str = "-",
    alpha: float = 1.0,
    width_frac: float = 0.24,
    filter_outliers: bool = True,
) -> None:
    values_arr = np.asarray(values, dtype=float).ravel()
    values_arr = values_arr[np.isfinite(values_arr)]
    clean_values = _remove_boxplot_outliers(values_arr) if filter_outliers else values_arr
    if clean_values.size == 0:
        return
    median = float(np.nanmedian(clean_values))
    half_width = float(width_frac) * width
    ax.hlines(
        median,
        float(x_center) - half_width,
        float(x_center) + half_width,
        color=color,
        lw=lw,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )


def _model_median_after_tukey(values: Sequence[float]) -> Optional[float]:
    clean_values = _remove_boxplot_outliers(values)
    if clean_values.size == 0:
        return None
    median = float(np.nanmedian(clean_values))
    return median if np.isfinite(median) else None


def _model_medians_for_indices(data_all: Sequence[np.ndarray], indices: Sequence[int]) -> np.ndarray:
    medians = []
    for idx in indices:
        if idx < 0 or idx >= len(data_all):
            continue
        median = _model_median_after_tukey(data_all[idx])
        if median is not None:
            medians.append(median)
    return np.asarray(medians, dtype=float)


def _expand_flat_median_range(y0: float, y1: float, reference_values: Sequence[float]) -> tuple[float, float]:
    if y1 > y0:
        return y0, y1
    reference = np.asarray(reference_values, dtype=float).ravel()
    reference = reference[np.isfinite(reference)]
    if reference.size >= 2:
        reference_span = float(np.nanmax(reference) - np.nanmin(reference))
    else:
        reference_span = 0.0
    min_height = reference_span * 0.04 if reference_span > 0 else max(abs(y0) * 0.002, 1e-6)
    return y0 - min_height / 2.0, y1 + min_height / 2.0


def _draw_median_range_band(
    ax: plt.Axes,
    x_span: tuple[float, float],
    medians: Sequence[float],
    *,
    color: str,
    alpha: float,
    zorder: float,
    edgecolor: Optional[str] = None,
    edge_lw: float = 0.0,
    edge_alpha: float = 1.0,
    edge_zorder: Optional[float] = None,
    reference_values: Optional[Sequence[float]] = None,
) -> None:
    medians_arr = np.asarray(medians, dtype=float).ravel()
    medians_arr = medians_arr[np.isfinite(medians_arr)]
    if medians_arr.size == 0:
        return
    y0 = float(np.nanmin(medians_arr))
    y1 = float(np.nanmax(medians_arr))
    reference = reference_values if reference_values is not None else medians_arr
    y0, y1 = _expand_flat_median_range(y0, y1, reference)
    x0, x1 = map(float, x_span)
    ax.fill_between([x0, x1], [y0, y0], [y1, y1], color=color, alpha=alpha, linewidth=0, zorder=zorder)
    if edgecolor is not None and edge_lw > 0:
        line_zorder = zorder + 0.2 if edge_zorder is None else edge_zorder
        ax.hlines([y0, y1], x0, x1, color=edgecolor, lw=edge_lw, alpha=edge_alpha, zorder=line_zorder)

def _redraw_violin_quartiles(
    ax: plt.Axes,
    x_center: float,
    values: Sequence[float],
    width: float,
    *,
    color: str = "k",
    lw: float = 1.0,
    zorder: float = 4.5,
    linestyle: str = "-",
    alpha: float = 0.95,
    width_frac: float = 0.0,
    filter_outliers: bool = True,
) -> None:
    """Draw the Q1-Q3 interval as a centered vertical line on a violin."""
    values_arr = np.asarray(values, dtype=float).ravel()
    values_arr = values_arr[np.isfinite(values_arr)]
    clean_values = _remove_boxplot_outliers(values_arr) if filter_outliers else values_arr
    if clean_values.size == 0:
        return
    q1, q3 = np.nanpercentile(clean_values, [25, 75])
    if not np.isfinite(q1) or not np.isfinite(q3):
        return
    x = float(x_center) + float(width_frac) * float(width)
    ax.plot(
        [x, x],
        [float(q1), float(q3)],
        color=color,
        lw=lw,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        solid_capstyle="round",
    )


def _annotate_above_data(
    ax: plt.Axes,
    x: float,
    data: Sequence[float],
    text: str,
    color: str,
    *,
    y_pad_frac: float = 0.04,
    fontsize: float = 14,
    va: str = "bottom",
    y_shift: float = 0.0,
) -> None:
    clean_values = np.asarray(data, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size == 0:
        return
    y_top = float(np.nanmax(clean_values))
    y_min, y_max = ax.get_ylim()
    y_pad = (y_max - y_min) * y_pad_frac
    ax.text(
        float(x),
        y_top + y_pad + y_shift,
        text,
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        ha="center",
        va=va,
        zorder=10,
    )


def _boxplot_whisker_bounds(data: Sequence[float], whis: float = 1.5) -> Optional[tuple[float, float]]:
    clean_values = np.asarray(data, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size == 0:
        return None
    if clean_values.size <= 3:
        return float(np.nanmin(clean_values)), float(np.nanmax(clean_values))

    q1 = float(np.nanpercentile(clean_values, 25))
    q3 = float(np.nanpercentile(clean_values, 75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return float(np.nanmin(clean_values)), float(np.nanmax(clean_values))

    lower_fence = q1 - whis * iqr
    upper_fence = q3 + whis * iqr
    lower_whisker = float(np.nanmin(clean_values[clean_values >= lower_fence]))
    upper_whisker = float(np.nanmax(clean_values[clean_values <= upper_fence]))
    return lower_whisker, upper_whisker


def _annotate_above_boxplot_whisker(
    ax: plt.Axes,
    x: float,
    data: Sequence[float],
    text: str,
    color: str,
    *,
    y_pad_frac: float = 0.04,
    fontsize: float = 14,
    va: str = "bottom",
    y_shift: float = 0.0,
    whisker: str = "upper",
) -> None:
    bounds = _boxplot_whisker_bounds(data)
    if bounds is None:
        return
    lower_whisker, upper_whisker = bounds
    y_anchor = lower_whisker if whisker == "lower" else upper_whisker
    y_min, y_max = ax.get_ylim()
    y_pad = (y_max - y_min) * y_pad_frac
    ax.text(
        float(x),
        y_anchor + y_pad + y_shift,
        text,
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        ha="center",
        va=va,
        zorder=10,
    )


def _annotate_grouped_box_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float = 11.5,
    color: str = "0.20",
) -> None:
    ax.text(
        float(x),
        float(y),
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        zorder=6,
        clip_on=True,
    )


def _apply_pressure_depth_axes(
    ax: plt.Axes,
    *,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    pressure_ylim: Optional[tuple[float, float]] = None,
    pressure_ticks: Optional[Sequence[float]] = None,
    depth_tick_step: float = 5,
    depth_max: float = 37,
    label_fontsize: float = MANUSCRIPT_Y_AXIS_LABEL_SIZE,
    tick_labelsize: float = MANUSCRIPT_TICK_LABEL_SIZE,
    depth_label_fontsize: Optional[float] = None,
    depth_tick_labelsize: Optional[float] = None,
) -> plt.Axes:
    depth_label_fontsize = label_fontsize if depth_label_fontsize is None else depth_label_fontsize
    depth_tick_labelsize = tick_labelsize if depth_tick_labelsize is None else depth_tick_labelsize

    ax.set_ylabel("Pressure (kbar)", fontsize=label_fontsize)
    if pressure_ylim is not None:
        ax.set_ylim(*pressure_ylim)
    if pressure_ticks is not None:
        ax.set_yticks(list(pressure_ticks))
    ax.invert_yaxis()

    depth_ticks_km = np.arange(0, depth_max + depth_tick_step, depth_tick_step)
    ax_depth = ax.secondary_yaxis(
        "right",
        functions=(
            lambda pressure_kbar: _pressure_to_depth_km(
                pressure_kbar,
                densities_kg_m3=densities_kg_m3,
                layer_boundaries_km=layer_boundaries_km,
            ),
            lambda depth_km: _depth_to_pressure_kbar(
                depth_km,
                densities_kg_m3=densities_kg_m3,
                layer_boundaries_km=layer_boundaries_km,
            ),
        ),
    )
    ax_depth.yaxis.set_major_locator(FixedLocator(depth_ticks_km))
    ax_depth.set_yticklabels([f"{int(d)}" for d in depth_ticks_km])
    ax_depth.set_ylabel("Depth (km)", rotation=-90, va="bottom", labelpad=10, fontsize=depth_label_fontsize)

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_depth.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", which="minor", length=3, width=0.8)
    ax.tick_params(axis="y", which="major", length=6, width=1.0, labelsize=tick_labelsize)
    ax_depth.tick_params(axis="y", which="minor", length=0)
    ax_depth.tick_params(axis="y", which="major", length=6, width=1.0, labelsize=depth_tick_labelsize)
    return ax_depth


def _validate_multilayer_depth_conversion(
    densities_kg_m3: Optional[Sequence[float]],
    layer_boundaries_km: Optional[Sequence[float]],
) -> None:
    if densities_kg_m3 is None or layer_boundaries_km is None:
        raise ValueError(
            "densities_kg_m3 and layer_boundaries_km must be provided together."
        )


def _pressure_to_depth_km(
    pressure_kbar: Any,
    *,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
) -> Any:
    _validate_multilayer_depth_conversion(densities_kg_m3, layer_boundaries_km)
    from aims4pt.data_tools.crust import (
        DEFAULT_G,
        KBAR_TO_PA,
        M_PER_KM,
        kbar_to_km_multilayer,
    )

    pressure_arr = np.asarray(pressure_kbar, dtype=float)
    depth_km = np.empty_like(pressure_arr, dtype=float)
    positive_mask = pressure_arr >= 0

    if np.any(positive_mask):
        depth_km[positive_mask] = kbar_to_km_multilayer(
            pressure_arr[positive_mask],
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
        )
    if np.any(~positive_mask):
        density_kg_m3 = float(np.asarray(densities_kg_m3, dtype=float)[0])
        depth_km[~positive_mask] = (
            pressure_arr[~positive_mask] * KBAR_TO_PA / (density_kg_m3 * DEFAULT_G) / M_PER_KM
        )

    if np.isscalar(pressure_kbar):
        return float(depth_km)
    return depth_km


def _depth_to_pressure_kbar(
    depth_km: Any,
    *,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
) -> Any:
    _validate_multilayer_depth_conversion(densities_kg_m3, layer_boundaries_km)
    from aims4pt.data_tools.crust import (
        DEFAULT_G,
        KBAR_TO_PA,
        M_PER_KM,
        km_to_kbar_multilayer,
    )

    depth_arr = np.asarray(depth_km, dtype=float)
    pressure_kbar = np.empty_like(depth_arr, dtype=float)
    positive_mask = depth_arr >= 0

    if np.any(positive_mask):
        pressure_kbar[positive_mask] = km_to_kbar_multilayer(
            depth_arr[positive_mask],
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
        )
    if np.any(~positive_mask):
        density_kg_m3 = float(np.asarray(densities_kg_m3, dtype=float)[0])
        pressure_kbar[~positive_mask] = (
            density_kg_m3 * DEFAULT_G * depth_arr[~positive_mask] * M_PER_KM / KBAR_TO_PA
        )

    if np.isscalar(depth_km):
        return float(pressure_kbar)
    return pressure_kbar


def _build_ranked_thermobarometry_state(
    cpx_only_workflows: Any,
    cpx_liq_workflows: Any,
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    liquid_results: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    cpx_only_bundle = _coerce_workflow_bundle(cpx_only_workflows, "cpx_only_workflows")
    cpx_liq_bundle = _coerce_workflow_bundle(cpx_liq_workflows, "cpx_liq_workflows")
    pressure_column_groups = _coerce_phase_column_groups(pressure_columns, "pressure_columns")
    temperature_column_groups = _coerce_phase_column_groups(temperature_columns, "temperature_columns")

    pressure_results = {
        "2006": {
            "cpx_only": _get_prediction_df_from_workflow(cpx_only_bundle.pressure_2006, "cpx_only_workflows.pressure_2006"),
            "cpx_liq": _get_prediction_df_from_workflow(cpx_liq_bundle.pressure_2006, "cpx_liq_workflows.pressure_2006"),
        },
        "2010": {
            "cpx_only": _get_prediction_df_from_workflow(cpx_only_bundle.pressure_2010, "cpx_only_workflows.pressure_2010"),
            "cpx_liq": _get_prediction_df_from_workflow(cpx_liq_bundle.pressure_2010, "cpx_liq_workflows.pressure_2010"),
        },
    }
    temperature_results = {
        "2006": {
            "cpx_only": _get_prediction_df_from_workflow(cpx_only_bundle.temperature_2006, "cpx_only_workflows.temperature_2006"),
            "cpx_liq": _get_prediction_df_from_workflow(cpx_liq_bundle.temperature_2006, "cpx_liq_workflows.temperature_2006"),
        },
        "2010": {
            "cpx_only": _get_prediction_df_from_workflow(cpx_only_bundle.temperature_2010, "cpx_only_workflows.temperature_2010"),
            "cpx_liq": _get_prediction_df_from_workflow(cpx_liq_bundle.temperature_2010, "cpx_liq_workflows.temperature_2010"),
        },
    }

    pressure_best_models = {
        "2006": {
            "cpx_only": _get_best_model_series_from_workflow(cpx_only_bundle.pressure_2006, "cpx_only_workflows.pressure_2006"),
            "cpx_liq": _get_best_model_series_from_workflow(cpx_liq_bundle.pressure_2006, "cpx_liq_workflows.pressure_2006"),
        },
        "2010": {
            "cpx_only": _get_best_model_series_from_workflow(cpx_only_bundle.pressure_2010, "cpx_only_workflows.pressure_2010"),
            "cpx_liq": _get_best_model_series_from_workflow(cpx_liq_bundle.pressure_2010, "cpx_liq_workflows.pressure_2010"),
        },
    }
    temperature_best_models = {
        "2006": {
            "cpx_only": _get_best_model_series_from_workflow(cpx_only_bundle.temperature_2006, "cpx_only_workflows.temperature_2006"),
            "cpx_liq": _get_best_model_series_from_workflow(cpx_liq_bundle.temperature_2006, "cpx_liq_workflows.temperature_2006"),
        },
        "2010": {
            "cpx_only": _get_best_model_series_from_workflow(cpx_only_bundle.temperature_2010, "cpx_only_workflows.temperature_2010"),
            "cpx_liq": _get_best_model_series_from_workflow(cpx_liq_bundle.temperature_2010, "cpx_liq_workflows.temperature_2010"),
        },
    }

    pressure_ranks = {
        "2006": {
            "cpx_only": _compute_rank_pcts_from_best(pressure_best_models["2006"]["cpx_only"]),
            "cpx_liq": _compute_rank_pcts_from_best(pressure_best_models["2006"]["cpx_liq"]),
        },
        "2010": {
            "cpx_only": _compute_rank_pcts_from_best(pressure_best_models["2010"]["cpx_only"]),
            "cpx_liq": _compute_rank_pcts_from_best(pressure_best_models["2010"]["cpx_liq"]),
        },
    }
    temperature_ranks = {
        "2006": {
            "cpx_only": _compute_rank_pcts_from_best(temperature_best_models["2006"]["cpx_only"]),
            "cpx_liq": _compute_rank_pcts_from_best(temperature_best_models["2006"]["cpx_liq"]),
        },
        "2010": {
            "cpx_only": _compute_rank_pcts_from_best(temperature_best_models["2010"]["cpx_only"]),
            "cpx_liq": _compute_rank_pcts_from_best(temperature_best_models["2010"]["cpx_liq"]),
        },
    }

    state = {
        "P": {
            "columns": pressure_column_groups,
            "columns_all": pressure_column_groups["cpx_only"] + pressure_column_groups["cpx_liq"],
            "split_idx": len(pressure_column_groups["cpx_only"]),
            "results": pressure_results,
            "best_models": pressure_best_models,
            "ranks": pressure_ranks,
            "uncertainty": _build_uncertainty_dict(pressure_model_pool),
            "liquid_results": _coerce_liquid_kind_results("P", liquid_results),
        },
        "T": {
            "columns": temperature_column_groups,
            "columns_all": temperature_column_groups["cpx_only"] + temperature_column_groups["cpx_liq"],
            "split_idx": len(temperature_column_groups["cpx_only"]),
            "results": temperature_results,
            "best_models": temperature_best_models,
            "ranks": temperature_ranks,
            "uncertainty": _build_uncertainty_dict(temperature_model_pool),
            "liquid_results": _coerce_liquid_kind_results("T", liquid_results),
        },
    }

    for kind in ("P", "T"):
        kind_state = state[kind]
        kind_state["data"] = {}
        for year in ("2006", "2010"):
            data_groups = []
            for phase_type in ("cpx_only", "cpx_liq"):
                df_label = f"{kind} results {year} {phase_type}"
                df = kind_state["results"][year][phase_type]
                cols = kind_state["columns"][phase_type]
                data_groups.extend(_extract_plot_arrays(df, cols, df_label=df_label))
            kind_state["data"][year] = _filter_data_groups(data_groups)

    return state


def _plot_ranked_this_study_kind_panel(
    ax: plt.Axes,
    kind: str,
    state: Mapping[str, Any],
    *,
    selection_threshold: float,
    model_selection_mode: str,
    cumulative_selection_threshold: float,
    pressure_ylim: Optional[tuple[float, float]] = None,
    pressure_ticks: Optional[Sequence[float]] = None,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    depth_tick_step: float = 5,
    depth_max: float = 37,
    use_model_abbreviations: bool = False,
    axis_label_fontsize: float = MANUSCRIPT_AXIS_LABEL_SIZE,
    y_axis_label_fontsize: float = MANUSCRIPT_Y_AXIS_LABEL_SIZE,
    tick_labelsize: float = MANUSCRIPT_TICK_LABEL_SIZE,
    model_tick_labelsize: float = MANUSCRIPT_MODEL_TICK_LABEL_SIZE,
    group_label_fontsize: float = MANUSCRIPT_GROUP_LABEL_SIZE,
    annotation_fontsize: float = MANUSCRIPT_ANNOTATION_SIZE,
    show_quartile_interval: bool = True,
    show_uncertainty_band: bool = False,
    show_rank_percent_annotations: bool = False,
    show_selected_model_summary: bool = True,
) -> None:
    kind_state = state[kind]
    columns_all = kind_state["columns_all"]
    split_idx = kind_state["split_idx"]
    n_cols = len(columns_all)
    positions = np.arange(1, n_cols + 1)
    offset = 0.14
    positions_06 = positions - offset
    positions_10 = positions + offset

    violin_width = 0.28
    edge_lw_default = 1.2
    edge_lw_selected = 3.0
    uncertainty_band_width = violin_width * 0.72
    color_06 = "blue"
    color_10 = "red"
    unselected_color_06 = _blend_color(color_06, "white", 0.34)
    unselected_color_10 = _blend_color(color_10, "white", 0.34)
    highlight_06 = "#20b8c5"
    highlight_10 = "#f28e2b"

    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=3, width=0.8)
    ax.tick_params(axis="y", which="major", length=6, width=1.0, labelsize=tick_labelsize)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, color="0.85", zorder=0)
    ax.grid(True, which="minor", axis="y", linestyle="-", linewidth=0.5, color="0.92", zorder=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            wrap_text(
                _model_axis_label(col, kind, use_model_abbreviations=use_model_abbreviations),
                max_width=12,
            )
            for col in columns_all
        ],
        rotation=0,
        ha="center",
        fontsize=model_tick_labelsize if use_model_abbreviations else tick_labelsize,
    )
    _apply_threshold_xtick_rotation(
        ax,
        [_model_axis_label(col, kind, use_model_abbreviations=use_model_abbreviations) for col in columns_all],
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="x", which="major", labelsize=model_tick_labelsize if use_model_abbreviations else tick_labelsize)

    ax.axvline(split_idx + 0.5, color="0.6", linewidth=1.0, zorder=1)
    ax.text(split_idx / 2 + 0.5, 1.025, _display_phase_type_label("cpx_only"), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=group_label_fontsize, fontweight="bold")
    ax.text((split_idx + n_cols) / 2 + 0.5, 1.025, _display_phase_type_label("cpx_liq"), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=group_label_fontsize, fontweight="bold")

    vp06 = _draw_violin(
        ax,
        kind_state["data"]["2006"],
        positions_06,
        widths=violin_width,
        facecolor=unselected_color_06,
        edgecolor="k",
        lw=edge_lw_default,
        alpha=1.0,
        bw_method=0.25,
        show_quartile_interval=show_quartile_interval,
    )
    vp10 = _draw_violin(
        ax,
        kind_state["data"]["2010"],
        positions_10,
        widths=violin_width,
        facecolor=unselected_color_10,
        edgecolor="k",
        lw=edge_lw_default,
        alpha=1.0,
        bw_method=0.25,
        show_quartile_interval=show_quartile_interval,
    )

    def _selected_index_set(rank_pcts: Sequence[tuple[str, float]], phase_type: str, idx_shift: int) -> set[int]:
        selected = set()
        selected_ranks = _selected_rank_set_from_topk(
            rank_pcts,
            threshold=selection_threshold,
            model_selection_mode=model_selection_mode,
            cumulative_threshold=cumulative_selection_threshold,
        )
        cols_local = kind_state["columns"][phase_type]
        for rank_i, (model_name, _) in enumerate(rank_pcts):
            if rank_i not in selected_ranks:
                continue
            idx_local = _model_to_index(cols_local, model_name)
            if idx_local is not None:
                selected.add(idx_shift + idx_local)
        return selected

    selected_06 = (
        _selected_index_set(kind_state["ranks"]["2006"]["cpx_only"], "cpx_only", 0)
        | _selected_index_set(kind_state["ranks"]["2006"]["cpx_liq"], "cpx_liq", split_idx)
    )
    selected_10 = (
        _selected_index_set(kind_state["ranks"]["2010"]["cpx_only"], "cpx_only", 0)
        | _selected_index_set(kind_state["ranks"]["2010"]["cpx_liq"], "cpx_liq", split_idx)
    )

    def _draw_all_uncertainty(positions_used: np.ndarray, data_all: Sequence[np.ndarray], band_color: str) -> None:
        for idx, model_name in enumerate(columns_all):
            _add_vertical_uncertainty_band(
                ax,
                positions_used[idx],
                data_all[idx],
                kind_state["uncertainty"].get(model_name),
                uncertainty_band_width,
                band_color,
                alpha=1.0,
                zorder=2.15,
            )

    def _draw_section_median_range_bands(
        data_all: Sequence[np.ndarray],
        selected_indices: set[int],
        phase_type: str,
        idx_shift: int,
        selected_color: str,
    ) -> None:
        n_phase_models = len(kind_state["columns"][phase_type])
        section_indices = list(range(idx_shift, idx_shift + n_phase_models))
        selected_section_indices = [idx for idx in section_indices if idx in selected_indices]
        section_span = (0.5, split_idx + 0.5) if phase_type == "cpx_only" else (split_idx + 0.5, n_cols + 0.5)
        all_medians = _model_medians_for_indices(data_all, section_indices)
        selected_medians = _model_medians_for_indices(data_all, selected_section_indices)
        _draw_median_range_band(
            ax,
            section_span,
            selected_medians,
            color=selected_color,
            alpha=0.15,
            zorder=1.05,
            edgecolor=selected_color,
            edge_lw=1.9,
            edge_alpha=0.95,
            edge_zorder=2.05,
            reference_values=all_medians,
        )

    def _annotate_all_pcts(
        rank_pcts: Sequence[tuple[str, float]],
        phase_type: str,
        idx_shift: int,
        positions_used: np.ndarray,
        data_all: Sequence[np.ndarray],
        selected_indices: set[int],
        selected_color: str,
    ) -> None:
        rank_pct_map = {model_name: pct for model_name, pct in rank_pcts}
        y_pad_frac = 0.06 if kind == "P" else 0.03
        y_shift = -0.12 if kind == "P" else 0.0
        for idx_local, model_name in enumerate(kind_state["columns"][phase_type]):
            pct = rank_pct_map.get(model_name)
            if pct is None:
                continue
            idx = idx_shift + idx_local
            is_selected = idx in selected_indices
            _annotate_above_data(
                ax,
                positions_used[idx],
                data_all[idx],
                f"{int(round(pct))}%",
                selected_color if is_selected else "k",
                y_pad_frac=y_pad_frac,
                fontsize=annotation_fontsize + 1 if is_selected else annotation_fontsize,
                y_shift=y_shift,
            )

    def _highlight_selected(
        vp: Mapping[str, Any],
        rank_pcts: Sequence[tuple[str, float]],
        phase_type: str,
        idx_shift: int,
        positions_used: np.ndarray,
        data_all: Sequence[np.ndarray],
        selected_facecolor: str,
    ) -> None:
        selected_ranks = _selected_rank_set_from_topk(
            rank_pcts,
            threshold=selection_threshold,
            model_selection_mode=model_selection_mode,
            cumulative_threshold=cumulative_selection_threshold,
        )
        for rank_i, (model_name, _) in enumerate(rank_pcts):
            if rank_i not in selected_ranks:
                continue
            idx_local = _model_to_index(kind_state["columns"][phase_type], model_name)
            if idx_local is None:
                continue
            idx = idx_shift + idx_local
            _style_selected_violin(vp, idx, edgecolor="k", facecolor=selected_facecolor, lw=edge_lw_selected, linestyle="--")

    if show_uncertainty_band:
        _draw_all_uncertainty(positions_06, kind_state["data"]["2006"], "lightgray")
        _draw_all_uncertainty(positions_10, kind_state["data"]["2010"], "lightgray")

    if show_selected_model_summary:
        _draw_section_median_range_bands(kind_state["data"]["2006"], selected_06, "cpx_only", 0, highlight_06)
        _draw_section_median_range_bands(kind_state["data"]["2006"], selected_06, "cpx_liq", split_idx, highlight_06)
        _draw_section_median_range_bands(kind_state["data"]["2010"], selected_10, "cpx_only", 0, highlight_10)
        _draw_section_median_range_bands(kind_state["data"]["2010"], selected_10, "cpx_liq", split_idx, highlight_10)

    if show_rank_percent_annotations:
        _annotate_all_pcts(
            kind_state["ranks"]["2006"]["cpx_only"], "cpx_only", 0, positions_06, kind_state["data"]["2006"], selected_06, color_06
        )
        _annotate_all_pcts(
            kind_state["ranks"]["2006"]["cpx_liq"], "cpx_liq", split_idx, positions_06, kind_state["data"]["2006"], selected_06, color_06
        )
        _annotate_all_pcts(
            kind_state["ranks"]["2010"]["cpx_only"], "cpx_only", 0, positions_10, kind_state["data"]["2010"], selected_10, color_10
        )
        _annotate_all_pcts(
            kind_state["ranks"]["2010"]["cpx_liq"], "cpx_liq", split_idx, positions_10, kind_state["data"]["2010"], selected_10, color_10
        )

    _highlight_selected(
        vp06, kind_state["ranks"]["2006"]["cpx_only"], "cpx_only", 0, positions_06, kind_state["data"]["2006"], color_06
    )
    _highlight_selected(
        vp06, kind_state["ranks"]["2006"]["cpx_liq"], "cpx_liq", split_idx, positions_06, kind_state["data"]["2006"], color_06
    )
    _highlight_selected(
        vp10, kind_state["ranks"]["2010"]["cpx_only"], "cpx_only", 0, positions_10, kind_state["data"]["2010"], color_10
    )
    _highlight_selected(
        vp10, kind_state["ranks"]["2010"]["cpx_liq"], "cpx_liq", split_idx, positions_10, kind_state["data"]["2010"], color_10
    )

    ax.set_xlim(0.5, n_cols + 0.5)

    if kind == "P":
        _apply_pressure_depth_axes(
            ax,
            pressure_ylim=pressure_ylim,
            pressure_ticks=pressure_ticks,
            depth_tick_step=depth_tick_step,
            depth_max=depth_max,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
            label_fontsize=y_axis_label_fontsize,
            tick_labelsize=tick_labelsize,
        )
    else:
        ax.set_ylabel("Temperature (°C)", fontsize=y_axis_label_fontsize)
        ax.tick_params(axis="y", labelsize=tick_labelsize)
    ax.xaxis.label.set_size(axis_label_fontsize)


def _load_literature_table(data_or_path: Any, label: str) -> pd.DataFrame:
    if isinstance(data_or_path, pd.DataFrame):
        df = data_or_path.copy()
    else:
        df = pd.read_excel(Path(data_or_path))

    if "plot" in df.columns:
        plot_mask = df["plot"].fillna(False).astype(bool)
        df = df.loc[plot_mask].reset_index(drop=True)
    return df


def _build_method_id(row: pd.Series) -> tuple[str, str, str, str, str]:
    return (
        _clean_str(row.get("category")),
        _clean_str(row.get("type")),
        _strip_parens(row.get("cite")),
        _clean_str(row.get("models")),
        _clean_str(row.get("thermobatometer")),
    )


def _make_lit_xtick_label(row: pd.Series) -> str:
    cite = _strip_parens(row.get("cite"))
    if cite:
        abbreviated = _abbreviate_literature_cite(cite)
        if abbreviated:
            return abbreviated
        return cite
    thermobarometer = _clean_str(row.get("thermobatometer"))
    if thermobarometer:
        return thermobarometer
    models = _clean_str(row.get("models"))
    if models:
        return models
    return _clean_str(row.get("type")) or "unknown"


def _abbreviate_literature_cite(cite: Any) -> str:
    cite_clean = _clean_str(cite)
    cite_lower = cite_clean.lower()
    abbreviation_map = [
        ("erdmann", "2016", "Erd16"),
        ("aisyah", "2018", "Ais18"),
        ("saepuloh", "2013", "Sae13"),
        ("budi-santoso", "2013", "B-S13"),
        ("budi santoso", "2013", "B-S13"),
        ("widiyantoro", "2018", "Wid18"),
        ("li", "2021", "Li21"),
        ("preece", "2014", "Pre14"),
        ("preece", "2016", "Pre16"),
    ]
    for author_token, year_token, label in abbreviation_map:
        if author_token in cite_lower and year_token in cite_lower:
            return label
    return ""


def _extract_numeric_ranges(
    row: pd.Series,
    keys: Sequence[tuple[str, str]],
) -> list[tuple[float, float]]:
    ranges = []
    for key_min, key_max in keys:
        value_min = row.get(key_min)
        value_max = row.get(key_max)
        if pd.isna(value_min) or pd.isna(value_max):
            continue
        try:
            ranges.append((float(value_min), float(value_max)))
        except Exception:
            pass
    return ranges


def _pressure_range_keys() -> list[tuple[str, str]]:
    return [
        ("min kbar", "max kbar"),
        ("min kbar2", "max kbar2"),
        ("min kbar3", "max kbar3"),
        ("min kbar (or equivalent)", "max kbar (or equivalent)"),
        ("min kbar (or equivalent)2", "max kbar (or equivalent)2"),
        ("min kbar (or equivalent)3", "max kbar (or equivalent)3"),
    ]


def _depth_range_keys() -> list[tuple[str, str]]:
    return [
        ("min km", "max km"),
        ("min km2", "max km2"),
        ("min km3", "max km3"),
        ("min depth (km)", "max depth (km)"),
        ("min depth (km)2", "max depth (km)2"),
        ("min depth (km)3", "max depth (km)3"),
    ]


def _first_valid_row_value(row: pd.Series, keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        return value
    return np.nan


def _coerce_literature_pressure_source(literature_pressure_source: str) -> str:
    source = _clean_str(literature_pressure_source).lower()
    if source not in {"pressure", "depth", "auto"}:
        raise ValueError(
            "literature_pressure_source must be 'pressure', 'depth', or 'auto'."
        )
    return source


def _convert_depth_range_to_pressure_range(
    depth_range: tuple[float, float],
    *,
    densities_kg_m3: Optional[Sequence[float]],
    layer_boundaries_km: Optional[Sequence[float]],
) -> tuple[float, float]:
    pressures = np.asarray(
        _depth_to_pressure_kbar(
            [depth_range[0], depth_range[1]],
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
        ),
        dtype=float,
    )
    return float(np.nanmin(pressures)), float(np.nanmax(pressures))


def _convert_depth_uncertainty_to_pressure(
    depth_range: tuple[float, float],
    uncertainty_depth_km: Any,
    *,
    densities_kg_m3: Optional[Sequence[float]],
    layer_boundaries_km: Optional[Sequence[float]],
) -> float:
    if pd.isna(uncertainty_depth_km):
        return np.nan
    try:
        uncertainty_depth_km = float(uncertainty_depth_km)
    except Exception:
        return np.nan
    if uncertainty_depth_km < 0:
        return np.nan
    midpoint_depth_km = 0.5 * (float(depth_range[0]) + float(depth_range[1]))
    lower_depth_km = max(0.0, midpoint_depth_km - uncertainty_depth_km)
    upper_depth_km = midpoint_depth_km + uncertainty_depth_km
    lower_pressure, upper_pressure = _convert_depth_range_to_pressure_range(
        (lower_depth_km, upper_depth_km),
        densities_kg_m3=densities_kg_m3,
        layer_boundaries_km=layer_boundaries_km,
    )
    return 0.5 * abs(upper_pressure - lower_pressure)


def _extract_ranges_pressure(
    row: pd.Series,
    *,
    literature_pressure_source: str = "pressure",
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
) -> tuple[list[tuple[float, float]], list[float]]:
    source = _coerce_literature_pressure_source(literature_pressure_source)

    if source in {"pressure", "auto"}:
        pressure_ranges = _extract_numeric_ranges(row, _pressure_range_keys())
        if pressure_ranges or source == "pressure":
            pressure_uncertainty = _first_valid_row_value(
                row,
                ("uncertainty kbar", "uncertainty"),
            )
            return pressure_ranges, [pressure_uncertainty] * len(pressure_ranges)

    depth_ranges = _extract_numeric_ranges(row, _depth_range_keys())
    depth_uncertainty = _first_valid_row_value(row, ("uncertainty km", "uncertainty"))
    pressure_ranges = [
        _convert_depth_range_to_pressure_range(
            depth_range,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
        )
        for depth_range in depth_ranges
    ]
    pressure_uncertainties = [
        _convert_depth_uncertainty_to_pressure(
            depth_range,
            depth_uncertainty,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
        )
        for depth_range in depth_ranges
    ]
    return pressure_ranges, pressure_uncertainties


def _extract_ranges_temperature(row: pd.Series) -> list[tuple[float, float]]:
    value_min = row.get("min")
    value_max = row.get("max")
    if pd.isna(value_min) or pd.isna(value_max):
        return []
    try:
        return [(float(value_min), float(value_max))]
    except Exception:
        return []


def _eruption_color(eruption: str) -> str:
    eruption_lower = _clean_str(eruption).lower()
    if "2006" in eruption_lower and "2010" in eruption_lower:
        return "purple"
    if "2006&2010" in eruption_lower or "2006 & 2010" in eruption_lower:
        return "purple"
    if "2006" in eruption_lower:
        return "blue"
    if "2010" in eruption_lower:
        return "red"
    return "0.2"


def _eruption_slot_offset(eruption: str, delta: float = 0.16) -> float:
    eruption_lower = _clean_str(eruption).lower()
    if ("2006" in eruption_lower) and ("2010" in eruption_lower):
        return 0.0
    if "2006" in eruption_lower:
        return -delta
    if "2010" in eruption_lower:
        return delta
    return 0.0


def _type_band_label(method: Mapping[str, Any]) -> str:
    """Return the label used in the type band for one method."""
    explicit_label = _clean_str(method.get("type_label"))
    if explicit_label:
        return explicit_label
    return _display_phase_type_label(method.get("type"))


def _add_category_and_type_bands(
    ax: plt.Axes,
    methods: Sequence[Mapping[str, Any]],
    x_positions: np.ndarray,
    *,
    category_fontsize: float = MANUSCRIPT_GROUP_LABEL_SIZE,
    type_fontsize: float = MANUSCRIPT_ANNOTATION_SIZE,
    category_y: float = 1.04,
    type_y: float = 0.985,
    show_type_labels: bool = True,
    category_wrap_width: int = 14,
) -> None:
    x_positions = np.asarray(x_positions, dtype=float)
    if len(methods) == 0 or x_positions.size == 0:
        return

    # Use midpoints between adjacent columns so non-uniform column spacing keeps
    # group separators aligned with the plotted data.
    block_edges = np.empty(x_positions.size + 1, dtype=float)
    block_edges[0] = x_positions[0] - 0.5
    block_edges[-1] = x_positions[-1] + 0.5
    if x_positions.size > 1:
        block_edges[1:-1] = 0.5 * (x_positions[:-1] + x_positions[1:])

    category_ranges = []
    start = 0
    for i in range(1, len(methods) + 1):
        if i == len(methods) or methods[i]["category"] != methods[start]["category"]:
            category_ranges.append((methods[start]["category"], start, i - 1))
            start = i

    for j, (category, i0, i1) in enumerate(category_ranges):
        x0 = block_edges[i0]
        x1 = block_edges[i1 + 1]
        xc = 0.5 * (x0 + x1)
        ax.text(
            xc,
            category_y,
            _wrap_label(category, width=category_wrap_width, allow_word_break=True),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=category_fontsize,
            fontweight="bold",
        )
        if j < len(category_ranges) - 1:
            ax.axvline(x1, color="0.35", lw=2.0, zorder=1.4)

    blocks = []
    start = 0
    for i in range(1, len(methods) + 1):
        current_type_label = _type_band_label(methods[start])
        next_type_label = None if i == len(methods) else _type_band_label(methods[i])
        if i == len(methods) or methods[i]["category"] != methods[start]["category"] or next_type_label != current_type_label:
            blocks.append((methods[start]["category"], current_type_label, start, i - 1))
            start = i

    for _, method_type, i0, i1 in blocks:
        x0 = block_edges[i0]
        x1 = block_edges[i1 + 1]
        ax.axvspan(x0, x1, color="white", alpha=1.0, zorder=0)
        xc = 0.5 * (x0 + x1)
        if show_type_labels:
            ax.text(
                xc,
                type_y,
                _wrap_label(method_type, width=14, allow_word_break=True),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=type_fontsize,
                color="0.20",
            )
        ax.axvline(x0, color="0.75", lw=1.0, zorder=1)
        ax.axvline(x1, color="0.75", lw=1.0, zorder=1)


def _add_pressure_reservoir_bands(
    ax: plt.Axes,
    reservoir_bands: Sequence[Mapping[str, Any]],
    *,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    label_fontsize: float = MANUSCRIPT_RESERVOIR_LABEL_SIZE,
    label_x: Optional[float] = None,
) -> None:
    if label_x is None:
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        x_text = 0.985
        ha = "left"
        clip_on = True
    else:
        transform = ax.transData
        x_text = float(label_x)
        ha = "right"
        clip_on = True

    for band in reservoir_bands:
        min_depth = band.get("min_depth_km", band.get("min km"))
        max_depth = band.get("max_depth_km", band.get("max km"))
        if min_depth is None or max_depth is None:
            continue
        p0, p1 = _convert_depth_range_to_pressure_range(
            (float(min_depth), float(max_depth)),
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
        )
        y0, y1 = sorted((p0, p1))
        label = _clean_str(band.get("label"))
        if label:
            ax.text(
                x_text,
                0.5 * (y0 + y1),
                label,
                transform=transform,
                ha=ha,
                va="center",
                fontsize=label_fontsize,
                fontweight="bold",
                color=band.get("text_color", "0.20"),
                zorder=1.6,
                clip_on=clip_on,
            )


def _group_literature_methods(
    df: pd.DataFrame,
    *,
    kind: str,
    literature_pressure_source: str = "pressure",
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str, str, str, str], list[pd.Series]] = defaultdict(list)
    for _, row in df.iterrows():
        groups[_build_method_id(row)].append(row)

    methods = []
    for rows in groups.values():
        representative = rows[0]
        eruptions = []
        for row in rows:
            if kind == "P":
                ranges, range_uncertainties = _extract_ranges_pressure(
                    row,
                    literature_pressure_source=literature_pressure_source,
                    densities_kg_m3=densities_kg_m3,
                    layer_boundaries_km=layer_boundaries_km,
                )
            else:
                ranges = _extract_ranges_temperature(row)
                range_uncertainties = [row.get("uncertainty", np.nan)] * len(ranges)
            if len(ranges) == 0:
                continue
            eruptions.append(
                {
                    "eruption": _clean_str(row.get("eruption(s)")),
                    "ranges": ranges,
                    "uncertainty": _first_valid_row_value(
                        row,
                        ("uncertainty", "uncertainty kbar", "uncertainty km"),
                    ),
                    "range_uncertainties": range_uncertainties,
                    "notes": _clean_str(row.get("notes")),
                }
            )
        if len(eruptions) == 0:
            continue
        methods.append(
            {
                "category": _clean_str(representative.get("category")),
                "type": _clean_str(representative.get("type")),
                "label": _make_lit_xtick_label(representative),
                "thermobatometer": _clean_str(representative.get("thermobatometer")),
                "eruptions": eruptions,
            }
        )

    category_order = {
        "Volatile saturation": 0,
        "Experiments": 1,
        "Geophysics": 2,
    }
    methods.sort(
        key=lambda method: (
            category_order.get(method["category"], 999),
            method["type"],
            method["label"],
        )
    )
    return methods


def _selected_predictions_from_best_models(df_pred: pd.DataFrame, best_model_series: pd.Series) -> np.ndarray:
    values = []
    common_index = best_model_series.index.intersection(df_pred.index)
    for row_id in common_index:
        model_name = best_model_series.loc[row_id]
        if pd.isna(model_name):
            continue
        model_name = str(model_name)
        if model_name not in df_pred.columns:
            continue
        value = df_pred.at[row_id, model_name]
        if pd.notna(value) and np.isfinite(float(value)):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _selected_predictions_for_model(
    df_pred: pd.DataFrame,
    best_model_series: pd.Series,
    model_name: str,
) -> np.ndarray:
    common_index = best_model_series.index.intersection(df_pred.index)
    if len(common_index) == 0 or model_name not in df_pred.columns:
        return np.asarray([], dtype=float)

    best_models = best_model_series.loc[common_index]
    selected_index = best_models.index[best_models.notna() & best_models.astype(str).eq(str(model_name))]
    values = pd.to_numeric(df_pred.loc[selected_index, model_name], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return values.dropna().to_numpy(dtype=float)


def _build_this_study_columns_for_comparison(
    kind: str,
    state: Mapping[str, Any],
    *,
    selection_threshold: float,
    model_selection_mode: str,
    cumulative_selection_threshold: float,
    use_model_abbreviations: bool = False,
) -> list[dict[str, Any]]:
    kind_state = state[kind]
    columns = []

    for phase_type in ("cpx_only", "cpx_liq"):
        for year in ("2006", "2010"):
            rank_pcts = kind_state["ranks"][year][phase_type]
            df_pred = kind_state["results"][year][phase_type]
            overall_data = _selected_predictions_from_best_models(
                df_pred,
                kind_state["best_models"][year][phase_type],
            )
            overall_data = _remove_boxplot_outliers(overall_data)
            models_to_plot = _pick_models_for_single_eruption(
                rank_pcts,
                threshold=selection_threshold,
                model_selection_mode=model_selection_mode,
                cumulative_threshold=cumulative_selection_threshold,
            )
            rank_pct_map = {model_name: pct for model_name, pct in rank_pcts}

            for rank_i, model_name in enumerate(models_to_plot):
                if model_name not in df_pred.columns:
                    raise KeyError(f"{kind} results {year} {phase_type} are missing column {model_name!r}.")
                model_data = pd.to_numeric(df_pred[model_name], errors="coerce").replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                favored_data = _selected_predictions_for_model(
                    df_pred,
                    kind_state["best_models"][year][phase_type],
                    model_name,
                )
                columns.append(
                    {
                        "category": "This study",
                        "type": phase_type,
                        "eruption": year,
                        "model": model_name,
                        "label": _model_axis_label(
                            model_name,
                            kind,
                            use_model_abbreviations=use_model_abbreviations,
                        ),
                        "data": model_data.dropna().to_numpy(dtype=float),
                        "favored_data": favored_data,
                        "overall_data": overall_data,
                        "rank_i": rank_i,
                        "pct": rank_pct_map.get(model_name),
                        "fill": "blue" if year == "2006" else "red",
                        "uncertainty": kind_state["uncertainty"].get(model_name, np.nan),
                    }
                )

    liquid_results = kind_state.get("liquid_results", {})
    if liquid_results:
        liquid_subcolumns = []
        liquid_colors = {"2006": "blue", "2010": "red"}
        for year in ("2006", "2010"):
            year_results = liquid_results.get(year, {})
            for phase_name in ("glass", "bulk"):
                if phase_name not in year_results:
                    continue
                liquid_subcolumns.append(
                    {
                        "eruption": year,
                        "phase_name": phase_name,
                        "data": year_results[phase_name],
                        "fill": liquid_colors[year],
                        "uncertainty": _liquid_prediction_uncertainty(kind, phase_name),
                    }
                )

        if liquid_subcolumns:
            columns.append(
                {
                    "category": "This study",
                    "type": "liquid",
                    "label": "Weber & Blundy, 2024",
                    "subcolumns": liquid_subcolumns,
                }
            )

    return columns


def _plot_this_study_boxes_on_comparison(
    ax: plt.Axes,
    this_cols: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    annotation_fontsize: float = MANUSCRIPT_ANNOTATION_SIZE,
    annotate_rmse: bool = True,
) -> Optional[Mapping[str, Any]]:
    if len(this_cols) == 0:
        return None

    x_centers = np.arange(1, len(this_cols) + 1)
    last_bp: Optional[Mapping[str, Any]] = None

    for i, col in enumerate(this_cols):
        x_center = x_centers[i]

        if "subcolumns" in col:
            subcolumns = list(col["subcolumns"])
            eruption_positions = {"2006": x_center - 0.10, "2010": x_center + 0.10}
            sub_positions = np.array([eruption_positions.get(subcol["eruption"], x_center) for subcol in subcolumns], dtype=float)
            last_bp = ax.boxplot(
                [subcol["data"] for subcol in subcolumns],
                positions=sub_positions,
                widths=0.20,
                showfliers=False,
                patch_artist=True,
            )

            phase_data: dict[str, list[np.ndarray]] = defaultdict(list)
            phase_positions: dict[str, list[float]] = defaultdict(list)
            for j, subcol in enumerate(subcolumns):
                _style_bp_item(last_bp, j, facecolor=subcol["fill"], edgecolor="k", lw=1.1, alpha=1.0)
                phase_data[subcol["phase_name"]].append(np.asarray(subcol["data"], dtype=float))
                phase_positions[subcol["phase_name"]].append(float(sub_positions[j]))

            y_min, y_max = ax.get_ylim()
            y_pad = max((y_max - y_min) * 0.03, 0.15 if kind == "P" else 3.0)
            for phase_name, phase_label in (("glass", "glass"), ("bulk", "bulk rock")):
                values = phase_data.get(phase_name, [])
                if not values:
                    continue
                phase_concat = np.concatenate(values)
                phase_concat = phase_concat[np.isfinite(phase_concat)]
                if phase_concat.size == 0:
                    continue
                label_x = float(np.mean(phase_positions.get(phase_name, [x_center])))
                if kind == "P":
                    label_y = float(np.nanmin(phase_concat) - y_pad)
                else:
                    label_y = float(np.nanmax(phase_concat) + y_pad)
                _annotate_grouped_box_label(ax, label_x, label_y, phase_label, fontsize=annotation_fontsize, color="0.15")
            continue

        _draw_overall_selected_violin(ax, x_center, col.get("overall_data", []), facecolor=col["fill"])
        last_bp = ax.boxplot([col["data"]], positions=[x_center], widths=0.20, showfliers=False, patch_artist=True)
        _style_bp_item(last_bp, 0, facecolor=col["fill"], edgecolor="k", lw=1.2, alpha=1.0)
        if annotate_rmse:
            _annotate_rmse_below_zero(ax, x_center, col.get("uncertainty"), kind=kind, fontsize=annotation_fontsize)
        pct = col.get("pct")
        if pct is not None:
            y_pad_frac = 0.06 if kind == "P" else 0.03
            _annotate_above_boxplot_whisker(
                ax,
                x_center,
                col["data"],
                f"{int(round(pct))}%",
                col["fill"],
                y_pad_frac=y_pad_frac,
                fontsize=annotation_fontsize,
            )

    return last_bp


def _draw_literature_comparison_violin(
    ax: plt.Axes,
    x_center: float,
    values: Sequence[float],
    *,
    width: float,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    lw: float,
    zorder: float,
    median_color: str,
    median_lw: float,
    median_linestyle: str = "-",
    median_width_frac: float = 0.24,
    median_filter_outliers: bool = True,
    show_quartile_interval: bool = True,
) -> None:
    clean_values = np.asarray(values, dtype=float).ravel()
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size == 0:
        return

    can_draw_violin = clean_values.size >= 3 and np.nanstd(clean_values) > 0
    if can_draw_violin:
        try:
            vp = _draw_violin(
                ax,
                [clean_values],
                [x_center],
                widths=width,
                facecolor=facecolor,
                edgecolor=edgecolor,
                lw=lw,
                alpha=alpha,
                bw_method=0.25,
                zorder=zorder,
                show_quartile_interval=False,
            )
        except (ValueError, np.linalg.LinAlgError):
            vp = {}
        if "cmedians" in vp:
            vp["cmedians"].set_alpha(0.0)

    if show_quartile_interval:
        _redraw_violin_quartiles(
            ax,
            x_center,
            clean_values,
            width,
            color="k",
            lw=max(median_lw * 0.75, 1.05),
            zorder=zorder + 0.42,
            alpha=0.95,
            filter_outliers=median_filter_outliers,
        )

    _redraw_violin_median(
        ax,
        x_center,
        clean_values,
        width,
        color=median_color,
        lw=median_lw,
        zorder=zorder + 0.45,
        linestyle=median_linestyle,
        width_frac=median_width_frac,
        filter_outliers=median_filter_outliers,
    )


def _annotate_favored_pct_above_violin(
    ax: plt.Axes,
    x_center: float,
    values: Sequence[float],
    text: str,
    color: str,
    *,
    kind: str,
    y_pad_frac: float = 0.025,
    fontsize: float = 14,
    x_offset: float = 0.0,
) -> None:
    clean_values = np.asarray(values, dtype=float).ravel()
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size == 0:
        return
    y_min, y_max = ax.get_ylim()
    y_pad = abs(y_max - y_min) * y_pad_frac
    y_anchor = float(np.nanmin(clean_values) - y_pad) if kind == "P" else float(np.nanmax(clean_values) + y_pad)
    ax.text(
        float(x_center) + x_offset,
        y_anchor,
        text,
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=6,
        clip_on=True,
    )


def _plot_liquid_subcolumn_violins_on_literature_comparison(
    ax: plt.Axes,
    x_center: float,
    subcolumns: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    annotation_fontsize: float,
    show_quartile_interval: bool = True,
) -> None:
    if not subcolumns:
        return

    eruption_positions = {"2006": x_center - 0.10, "2010": x_center + 0.10}
    sub_positions = np.array(
        [eruption_positions.get(subcol["eruption"], x_center) for subcol in subcolumns],
        dtype=float,
    )
    phase_data: dict[str, list[np.ndarray]] = defaultdict(list)
    phase_positions: dict[str, list[float]] = defaultdict(list)
    for subcol, sub_position in zip(subcolumns, sub_positions):
        _draw_literature_comparison_violin(
            ax,
            float(sub_position),
            subcol["data"],
            width=0.24,
            facecolor=subcol["fill"],
            edgecolor="none",
            alpha=0.22,
            lw=0.0,
            zorder=2.2,
            median_color="k",
            median_lw=1.2,
            show_quartile_interval=show_quartile_interval,
        )
        phase_data[subcol["phase_name"]].append(np.asarray(subcol["data"], dtype=float))
        phase_positions[subcol["phase_name"]].append(float(sub_position))

    y_min, y_max = ax.get_ylim()
    y_pad = max((y_max - y_min) * 0.03, 0.15 if kind == "P" else 3.0)
    for phase_name, phase_label in (("glass", "glass"), ("bulk", "bulk rock")):
        values = phase_data.get(phase_name, [])
        if not values:
            continue
        phase_concat = np.concatenate(values)
        phase_concat = phase_concat[np.isfinite(phase_concat)]
        if phase_concat.size == 0:
            continue
        label_x = float(np.mean(phase_positions.get(phase_name, [x_center])))
        if kind == "P":
            label_y = float(np.nanmin(phase_concat) - y_pad)
        else:
            label_y = float(np.nanmax(phase_concat) + y_pad)
        _annotate_grouped_box_label(ax, label_x, label_y, phase_label, fontsize=annotation_fontsize, color="0.15")


def _plot_this_study_dual_violins_on_literature_comparison(
    ax: plt.Axes,
    this_cols: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    annotation_fontsize: float = MANUSCRIPT_ANNOTATION_SIZE,
    annotate_rmse: bool = True,
    x_centers: Optional[Sequence[float]] = None,
    show_quartile_interval: bool = True,
    favored_pct_fontsize: Optional[float] = None,
    favored_pct_x_offset: float = 0.0,
) -> None:
    if len(this_cols) == 0:
        return

    if x_centers is None:
        x_centers = np.arange(1, len(this_cols) + 1, dtype=float)
    else:
        x_centers = np.asarray(x_centers, dtype=float)
        if x_centers.size != len(this_cols):
            raise ValueError("x_centers must have the same length as this_cols.")

    for i, col in enumerate(this_cols):
        x_center = float(x_centers[i])
        if "subcolumns" in col:
            _plot_liquid_subcolumn_violins_on_literature_comparison(
                ax,
                x_center,
                list(col["subcolumns"]),
                kind=kind,
                annotation_fontsize=annotation_fontsize,
                show_quartile_interval=show_quartile_interval,
            )
            continue

        eruption_color = col["fill"]
        all_data = col.get("data", [])
        favored_data = col.get("favored_data", [])

        all_x = x_center - 0.10
        favored_x = x_center + 0.16

        _draw_literature_comparison_violin(
            ax,
            all_x,
            all_data,
            width=0.30,
            facecolor=eruption_color,
            edgecolor=eruption_color,
            alpha=0.42,
            lw=0.7,
            zorder=3.0,
            median_color="k",
            median_lw=1.5,
            median_width_frac=0.34,
            median_filter_outliers=False,
            show_quartile_interval=show_quartile_interval,
        )
        _draw_literature_comparison_violin(
            ax,
            favored_x,
            favored_data,
            width=0.16,
            facecolor="none",
            edgecolor=eruption_color,
            alpha=1.0,
            lw=1.3,
            zorder=3.4,
            median_color="k",
            median_lw=1.5,
            median_width_frac=0.34,
            median_filter_outliers=False,
            show_quartile_interval=show_quartile_interval,
        )

        if annotate_rmse:
            _annotate_literature_uncertainty_below_xtick(
                ax,
                x_center,
                col.get("uncertainty"),
                kind=kind,
                fontsize=annotation_fontsize,
            )
        pct = col.get("pct")
        if pct is not None:
            annotation_data = favored_data if np.asarray(favored_data, dtype=float).size > 0 else all_data
            _annotate_favored_pct_above_violin(
                ax,
                favored_x,
                annotation_data,
                f"{int(round(pct))}%",
                eruption_color,
                kind=kind,
                fontsize=annotation_fontsize if favored_pct_fontsize is None else favored_pct_fontsize,
                x_offset=favored_pct_x_offset,
            )


def _build_methods_for_bands(this_cols: Sequence[Mapping[str, Any]], lit_methods: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    this_for_bands = [{"category": col["category"], "type": col["type"], "label": col["label"]} for col in this_cols]
    lit_for_bands = [{"category": method["category"], "type": method["type"], "label": method["label"]} for method in lit_methods]
    return this_for_bands + lit_for_bands


def _draw_overall_selected_violin(
    ax: plt.Axes,
    x_center: float,
    values: Any,
    *,
    facecolor: str,
) -> None:
    clean_values = np.asarray(values, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size < 3 or np.nanstd(clean_values) <= 0:
        return
    vp = _draw_violin(
        ax,
        [clean_values],
        [x_center],
        widths=0.46,
        facecolor=facecolor,
        edgecolor=facecolor,
        lw=0.0,
        alpha=0.18,
        bw_method=0.25,
        zorder=1.6,
        show_quartile_interval=False,
    )
    if "cmedians" in vp:
        vp["cmedians"].set_alpha(0.0)


def _annotate_rmse_below_zero(
    ax: plt.Axes,
    x_center: float,
    uncertainty: Any,
    *,
    kind: str,
    fontsize: float,
) -> None:
    finite_uncertainty = _finite_uncertainty(uncertainty)
    if kind != "P" or finite_uncertainty is None:
        return
    ax.text(
        x_center,
        0.25,
        f"RMSE =\n{finite_uncertainty:.1f} kbar",
        ha="center",
        va="top",
        fontsize=fontsize,
        color="0.15",
        zorder=5,
        clip_on=False,
    )


def _annotate_literature_uncertainty_below_xtick(
    ax: plt.Axes,
    x_center: float,
    uncertainty: Any,
    *,
    kind: str,
    fontsize: float,
) -> None:
    finite_uncertainty = _finite_uncertainty(uncertainty)
    if kind != "P" or finite_uncertainty is None:
        return
    ax.text(
        x_center,
        -0.06,
        f"{finite_uncertainty:.1f} kbar",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=fontsize,
        color="0.45",
        zorder=5,
        clip_on=False,
    )


def _is_melts_modeling_method(method: Mapping[str, Any], kind: str) -> bool:
    return (
        kind == "T"
        and _clean_str(method.get("category")).lower() == "modeling"
        and _clean_str(method.get("type")).upper() == "MELTS"
    )


def _format_melts_note(note: Any) -> str:
    note_clean = _clean_str(note)
    if "4 wt." in note_clean:
        return "3 kbar and\n4 wt.% H2O"
    if "6 wt." in note_clean:
        return "3 kbar and\n6 wt.% H2O"
    return _wrap_label(note_clean, width=14)


def _draw_melts_modeling_column(
    ax: plt.Axes,
    xi: float,
    eruptions: Sequence[Mapping[str, Any]],
    *,
    text_fontsize: float = MANUSCRIPT_ANNOTATION_SIZE,
) -> None:
    melts_points = []
    for eruption_item in eruptions:
        for range_min, range_max in eruption_item["ranges"]:
            melts_points.append(
                {
                    "y": 0.5 * (float(range_min) + float(range_max)),
                    "label": _format_melts_note(eruption_item.get("notes", "")),
                }
            )

    if len(melts_points) == 0:
        return

    melts_points = sorted(melts_points, key=lambda item: item["y"], reverse=True)
    melts_color = "#8a0f9c"
    cap_half_width = 0.13
    reference_arrow_length = None

    for idx, point in enumerate(melts_points):
        y = point["y"]
        ax.text(xi, y, point["label"], fontsize=text_fontsize, color="0.10", ha="center", va="center", zorder=4, clip_on=True)

        arrow_start = y - 7.0
        if idx < len(melts_points) - 1:
            arrow_end = melts_points[idx + 1]["y"] + 7.0
            reference_arrow_length = arrow_start - arrow_end
        else:
            arrow_length = reference_arrow_length if reference_arrow_length is not None else 16.0
            arrow_end = arrow_start - arrow_length

        ax.hlines(arrow_start, xi - cap_half_width, xi + cap_half_width, color=melts_color, lw=2.3, zorder=4)
        ax.annotate(
            "",
            xy=(xi, arrow_end),
            xytext=(xi, arrow_start),
            arrowprops=dict(arrowstyle="-|>", color=melts_color, lw=2.3),
            zorder=4,
        )


def _plot_ranked_literature_panel(
    ax: plt.Axes,
    kind: str,
    state: Mapping[str, Any],
    literature_df: pd.DataFrame,
    *,
    selection_threshold: float,
    model_selection_mode: str,
    cumulative_selection_threshold: float,
    add_literature_legend: bool,
    pressure_ylim: Optional[tuple[float, float]] = None,
    temperature_ylim: Optional[tuple[float, float]] = None,
    pressure_ticks: Optional[Sequence[float]] = None,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    literature_pressure_source: str = "pressure",
    depth_tick_step: float = 5,
    depth_max: float = 37,
    use_model_abbreviations: bool = False,
    pressure_reservoir_bands: Optional[Sequence[Mapping[str, Any]]] = None,
    axis_label_fontsize: float = MANUSCRIPT_AXIS_LABEL_SIZE,
    y_axis_label_fontsize: float = MANUSCRIPT_Y_AXIS_LABEL_SIZE,
    tick_labelsize: float = MANUSCRIPT_TICK_LABEL_SIZE,
    model_tick_labelsize: float = MANUSCRIPT_MODEL_TICK_LABEL_SIZE,
    group_label_fontsize: float = MANUSCRIPT_GROUP_LABEL_SIZE,
    legend_fontsize: float = MANUSCRIPT_LEGEND_SIZE,
    annotation_fontsize: float = MANUSCRIPT_ANNOTATION_SIZE,
    reservoir_label_fontsize: float = MANUSCRIPT_RESERVOIR_LABEL_SIZE,
    split_literature_legend: bool = False,
    this_study_legend_bbox_to_anchor: tuple[float, float] = (0.005, 0.02),
    literature_legend_bbox_to_anchor: tuple[float, float] = (0.58, 0.50),
    reservoir_label_x_offset: float = 0.95,
    reservoir_label_right_margin: float = 0.10,
    this_study_x_spacing: float = 1.0,
    show_quartile_interval: bool = True,
) -> None:
    this_cols = _build_this_study_columns_for_comparison(
        kind,
        state,
        selection_threshold=selection_threshold,
        model_selection_mode=model_selection_mode,
        cumulative_selection_threshold=cumulative_selection_threshold,
        use_model_abbreviations=use_model_abbreviations,
    )
    lit_methods = _group_literature_methods(
        literature_df,
        kind=kind,
        literature_pressure_source=literature_pressure_source,
        densities_kg_m3=densities_kg_m3,
        layer_boundaries_km=layer_boundaries_km,
    )

    methods_all = _build_methods_for_bands(this_cols, lit_methods)
    n_this = len(this_cols)
    if n_this > 0:
        this_x = 1.0 + np.arange(n_this, dtype=float) * float(this_study_x_spacing)
        lit_start = this_x[-1] + 1.0
    else:
        this_x = np.asarray([], dtype=float)
        lit_start = 1.0
    x_lit = lit_start + np.arange(len(lit_methods), dtype=float)
    x_all = np.concatenate([this_x, x_lit])
    reservoir_label_x = None
    if kind == "P" and pressure_reservoir_bands:
        reservoir_label_x = (x_all[-1] if x_all.size else 0.5) + reservoir_label_x_offset
    _add_category_and_type_bands(
        ax,
        methods_all,
        x_all,
        category_fontsize=group_label_fontsize,
        type_fontsize=annotation_fontsize,
    )
    if kind == "P" and pressure_reservoir_bands:
        _add_pressure_reservoir_bands(
            ax,
            pressure_reservoir_bands,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
            label_fontsize=reservoir_label_fontsize,
            label_x=reservoir_label_x,
        )

    if n_this > 0:
        _plot_this_study_dual_violins_on_literature_comparison(
            ax,
            this_cols,
            kind=kind,
            annotation_fontsize=annotation_fontsize,
            x_centers=this_x,
            show_quartile_interval=show_quartile_interval,
        )

    if len(lit_methods) > 0:
        for i, method in enumerate(lit_methods):
            xi = x_lit[i]
            thermobarometer_text = _clean_str(method.get("thermobatometer"))

            if _is_melts_modeling_method(method, kind):
                _draw_melts_modeling_column(ax, xi, method["eruptions"], text_fontsize=annotation_fontsize)
                continue

            for eruption_item in method["eruptions"]:
                color = _eruption_color(eruption_item["eruption"])
                dx = _eruption_slot_offset(eruption_item["eruption"], delta=0.16)
                ranges = eruption_item["ranges"]
                offsets_local = np.linspace(-0.06, 0.06, max(1, len(ranges)))

                for k, (range_min, range_max) in enumerate(ranges):
                    xk = xi + dx + offsets_local[k]
                    if np.isclose(float(range_min), float(range_max), equal_nan=False):
                        ax.plot(
                            xk,
                            range_min,
                            marker="o",
                            markersize=6.5,
                            color=color,
                            markeredgecolor=color,
                            linestyle="None",
                            zorder=3,
                        )
                    else:
                        ax.vlines(xk, range_min, range_max, color=color, lw=4.0, zorder=3)
                        ax.hlines([range_min, range_max], xk - 0.07, xk + 0.07, color=color, lw=2.4, zorder=3)

                    if _clean_str(method["category"]).lower() == "geothermobarometry" and thermobarometer_text:
                        if kind == "P":
                            y_top = min(range_min, range_max)
                            y_text = y_top - 0.25
                        else:
                            y_top = max(range_min, range_max)
                            y_text = y_top + 15.0
                        ax.text(
                            xk,
                            y_text,
                            _wrap_label(thermobarometer_text, width=14),
                            fontsize=annotation_fontsize,
                            color="0.20",
                            ha="center",
                            va="bottom",
                            zorder=4,
                            clip_on=True,
                        )

    x_right = (
        reservoir_label_x + reservoir_label_right_margin
        if reservoir_label_x is not None
        else (x_all[-1] + 0.5 if x_all.size else 0.5)
    )
    ax.set_xlim(0.5, x_right)
    ax.set_xticks(x_all)
    raw_xtick_labels = [method["label"] for method in methods_all]
    xtick_labels = [_wrap_label(label, width=12) for label in raw_xtick_labels]
    ax.set_xticklabels(
        xtick_labels,
        rotation=0,
        ha="center",
        fontsize=model_tick_labelsize if use_model_abbreviations else tick_labelsize,
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="x", which="major", labelsize=model_tick_labelsize if use_model_abbreviations else tick_labelsize)
    _apply_threshold_xtick_rotation(ax, raw_xtick_labels)

    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, color="0.85", zorder=0)
    ax.grid(True, which="minor", axis="y", linestyle="-", linewidth=0.5, color="0.92", zorder=0)

    if kind == "P":
        _apply_pressure_depth_axes(
            ax,
            pressure_ylim=pressure_ylim,
            pressure_ticks=pressure_ticks,
            depth_tick_step=depth_tick_step,
            depth_max=depth_max,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
            label_fontsize=y_axis_label_fontsize,
            tick_labelsize=tick_labelsize,
        )
    else:
        ax.set_ylabel("Temperature (°C)", fontsize=y_axis_label_fontsize)
        if temperature_ylim is not None:
            ax.set_ylim(*temperature_ylim)
        ax.tick_params(axis="y", labelsize=tick_labelsize)
    ax.xaxis.label.set_size(axis_label_fontsize)

    if add_literature_legend:
    
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.legend_handler import HandlerTuple
        year_2006_handle = (
            Patch(
                facecolor="blue",
                alpha=0.55,
            ),
            Line2D(
                [0], [0],
                color="blue",
                lw=2.2,
                alpha=1,
            ),
        )
        year_2010_handle = (
            Patch(
                facecolor="red",
                alpha=0.55,
            ),
            Line2D(
                [0], [0],
                color="red",
                lw=2.2,
                alpha=1,
            ),
        )

        year_both_handle = (
            Line2D([0], [0], color="purple", lw=3.2),
        )

        filled_violin_handle = (
            Patch(facecolor="0.60", edgecolor="0.25", alpha=0.55),
        )

        open_violin_handle = (
            Patch(facecolor="none", edgecolor="0.25", linewidth=1.4),
        )

        blank_handle = (
            Line2D([0], [0], color="none", lw=0.0),
        )
        ax.legend(
            handles=[
                year_2006_handle,
                year_2010_handle,
                year_both_handle,
                filled_violin_handle,
                open_violin_handle,
                blank_handle,
            ],
            labels=[
                "2006",
                "2010",
                "2006 & 2010",
                "all samples",
                "favored subset",
                "",
            ],
            loc="lower left",
            bbox_to_anchor=this_study_legend_bbox_to_anchor,
            frameon=True,
            ncol=2,
            fontsize=legend_fontsize,
            borderpad=0.35,
            columnspacing=1.75,
            handlelength=2.1,
            handletextpad=0.5,
            labelspacing=0.25,
        )


def _build_original_vs_pre_2006_columns_for_comparison(
    kind: str,
    original_state: Mapping[str, Any],
    pre_2006_state: Mapping[str, Any],
    *,
    selection_threshold: float,
    model_selection_mode: str,
    cumulative_selection_threshold: float,
    use_model_abbreviations: bool,
) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    for category_label, state in (("Original", original_state), ("+pre-2006", pre_2006_state)):
        state_columns = _build_this_study_columns_for_comparison(
            kind,
            state,
            selection_threshold=selection_threshold,
            model_selection_mode=model_selection_mode,
            cumulative_selection_threshold=cumulative_selection_threshold,
            use_model_abbreviations=use_model_abbreviations,
        )
        for col in state_columns:
            if "subcolumns" in col:
                continue
            if col.get("type") != "cpx_liq":
                continue
            col_copy = dict(col)
            col_copy["category"] = category_label
            columns.append(col_copy)
    return columns


def _plot_original_vs_pre_2006_kind_panel(
    ax: plt.Axes,
    kind: str,
    original_state: Mapping[str, Any],
    pre_2006_state: Mapping[str, Any],
    *,
    selection_threshold: float,
    model_selection_mode: str,
    cumulative_selection_threshold: float,
    pressure_ylim: Optional[tuple[float, float]],
    temperature_ylim: Optional[tuple[float, float]],
    use_model_abbreviations: bool,
    tick_labelsize: float = MANUSCRIPT_THIS_STUDY_TICK_LABEL_SIZE,
    annotation_fontsize: float = MANUSCRIPT_THIS_STUDY_ANNOTATION_SIZE,
    y_axis_label_fontsize: float = MANUSCRIPT_THIS_STUDY_Y_AXIS_LABEL_SIZE,
    show_quartile_interval: bool = True,
) -> None:
    columns = _build_original_vs_pre_2006_columns_for_comparison(
        kind,
        original_state,
        pre_2006_state,
        selection_threshold=selection_threshold,
        model_selection_mode=model_selection_mode,
        cumulative_selection_threshold=cumulative_selection_threshold,
        use_model_abbreviations=use_model_abbreviations,
    )
    x_centers = np.arange(1, len(columns) + 1, dtype=float)
    all_values = [
        np.asarray(values, dtype=float)
        for col in columns
        for values in (col.get("data", []), col.get("overall_data", []))
        if np.asarray(values, dtype=float).size > 0
    ]

    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=3, width=0.8)
    ax.tick_params(axis="y", which="major", length=6, width=1.0, labelsize=tick_labelsize)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, color="0.85", zorder=0)
    ax.grid(True, which="minor", axis="y", linestyle="-", linewidth=0.5, color="0.92", zorder=0)

    if columns:
        _add_category_and_type_bands(
            ax,
            [{"category": col["category"], "type": col["type"], "label": col["label"]} for col in columns],
            x_centers,
            category_fontsize=MANUSCRIPT_GROUP_LABEL_SIZE,
            type_fontsize=annotation_fontsize,
        )
        _plot_this_study_dual_violins_on_literature_comparison(
            ax,
            columns,
            kind=kind,
            annotation_fontsize=annotation_fontsize,
            annotate_rmse=False,
            show_quartile_interval=show_quartile_interval,
        )

    raw_xtick_labels = [col["label"] for col in columns]
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [_wrap_label(label, width=12) for label in raw_xtick_labels],
        rotation=0,
        ha="center",
        fontsize=tick_labelsize,
    )
    ax.tick_params(axis="x", which="major", labelsize=tick_labelsize)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    _apply_threshold_xtick_rotation(ax, raw_xtick_labels)
    ax.set_xlim(0.5, len(columns) + 0.5)

    if kind == "P":
        ax.set_ylabel("Pressure (kbar)", fontsize=y_axis_label_fontsize)
        if pressure_ylim is not None:
            ax.set_ylim(*pressure_ylim)
        # reverse y-axis to have pressure increase downwards
        ax.invert_yaxis()
    else:
        ax.set_ylabel("Temperature (°C)", fontsize=y_axis_label_fontsize)
        if temperature_ylim is not None:
            ax.set_ylim(*temperature_ylim)

    if kind == "T" and temperature_ylim is None and all_values:
        finite_values = np.concatenate(all_values)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size > 0:
            y_min = float(np.nanmin(finite_values))
            y_max = float(np.nanmax(finite_values))
            y_pad = max((y_max - y_min) * 0.18, 20.0)
            ax.set_ylim(y_min - y_pad * 0.25, y_max + y_pad)


def _build_kd_columns_for_comparison(
    kind: str,
    kd_states: Mapping[str, Mapping[str, Any]],
    *,
    selection_threshold: float,
    model_selection_mode: str,
    cumulative_selection_threshold: float,
    use_model_abbreviations: bool,
) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    for kd_label, state in kd_states.items():
        state_columns = _build_this_study_columns_for_comparison(
            kind,
            state,
            selection_threshold=selection_threshold,
            model_selection_mode=model_selection_mode,
            cumulative_selection_threshold=cumulative_selection_threshold,
            use_model_abbreviations=use_model_abbreviations,
        )
        for col in state_columns:
            if "subcolumns" in col:
                continue
            if col.get("type") != "cpx_liq":
                continue
            col_copy = dict(col)
            col_copy["category"] = kd_label
            columns.append(col_copy)
    return columns


def _plot_kd_comparison_kind_panel(
    ax: plt.Axes,
    kind: str,
    kd_states: Mapping[str, Mapping[str, Any]],
    *,
    selection_threshold: float,
    model_selection_mode: str,
    cumulative_selection_threshold: float,
    pressure_ylim: Optional[tuple[float, float]],
    temperature_ylim: Optional[tuple[float, float]],
    use_model_abbreviations: bool,
    tick_labelsize: float = MANUSCRIPT_THIS_STUDY_TICK_LABEL_SIZE,
    annotation_fontsize: float = MANUSCRIPT_THIS_STUDY_ANNOTATION_SIZE,
    y_axis_label_fontsize: float = MANUSCRIPT_THIS_STUDY_Y_AXIS_LABEL_SIZE,
    show_quartile_interval: bool = True,
) -> None:
    columns = _build_kd_columns_for_comparison(
        kind,
        kd_states,
        selection_threshold=selection_threshold,
        model_selection_mode=model_selection_mode,
        cumulative_selection_threshold=cumulative_selection_threshold,
        use_model_abbreviations=use_model_abbreviations,
    )
    x_centers = np.arange(1, len(columns) + 1, dtype=float)
    all_values = [
        np.asarray(values, dtype=float)
        for col in columns
        for values in (col.get("data", []), col.get("overall_data", []))
        if np.asarray(values, dtype=float).size > 0
    ]

    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=3, width=0.8)
    ax.tick_params(axis="y", which="major", length=6, width=1.0, labelsize=tick_labelsize)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, color="0.85", zorder=0)
    ax.grid(True, which="minor", axis="y", linestyle="-", linewidth=0.5, color="0.92", zorder=0)

    if columns:
        _add_category_and_type_bands(
            ax,
            [{"category": col["category"], "type": col["type"], "label": col["label"]} for col in columns],
            x_centers,
            category_fontsize=MANUSCRIPT_GROUP_LABEL_SIZE,
            type_fontsize=annotation_fontsize,
            show_type_labels=False,
        )
        _plot_this_study_dual_violins_on_literature_comparison(
            ax,
            columns,
            kind=kind,
            annotation_fontsize=annotation_fontsize,
            annotate_rmse=False,
            show_quartile_interval=show_quartile_interval,
            favored_pct_fontsize=max(annotation_fontsize - 2.0, 1.0),
            favored_pct_x_offset=0.04,
        )

    raw_xtick_labels = [col["label"] for col in columns]
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [_wrap_label(label, width=12) for label in raw_xtick_labels],
        rotation=0,
        ha="center",
        fontsize=tick_labelsize,
    )
    ax.tick_params(axis="x", which="major", labelsize=tick_labelsize)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    _apply_threshold_xtick_rotation(ax, raw_xtick_labels)
    ax.set_xlim(0.5, len(columns) + 0.5)

    if kind == "P":
        ax.set_ylabel("Pressure (kbar)", fontsize=y_axis_label_fontsize)
        if pressure_ylim is not None:
            ax.set_ylim(*pressure_ylim)
        # reverse y-axis to have pressure increase downwards
        ax.invert_yaxis()
    else:
        ax.set_ylabel("Temperature (°C)", fontsize=y_axis_label_fontsize)
        if temperature_ylim is not None:
            ax.set_ylim(*temperature_ylim)

    if kind == "T" and temperature_ylim is None and all_values:
        finite_values = np.concatenate(all_values)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size > 0:
            y_min = float(np.nanmin(finite_values))
            y_max = float(np.nanmax(finite_values))
            y_pad = max((y_max - y_min) * 0.18, 20.0)
            ax.set_ylim(y_min - y_pad * 0.25, y_max + y_pad)

def _apply_this_study_axis_font_sizes(
    fig: plt.Figure,
    *,
    labelsize: float = MANUSCRIPT_THIS_STUDY_AXIS_LABEL_SIZE,
    y_labelsize: float = MANUSCRIPT_THIS_STUDY_Y_AXIS_LABEL_SIZE,
    tick_labelsize: float = MANUSCRIPT_THIS_STUDY_TICK_LABEL_SIZE,
    model_tick_labelsize: float = MANUSCRIPT_THIS_STUDY_MODEL_TICK_LABEL_SIZE,
    legend_fontsize: float = MANUSCRIPT_THIS_STUDY_LEGEND_SIZE,
) -> None:
    for ax in fig.axes:
        ax.xaxis.label.set_size(labelsize)
        ax.yaxis.label.set_size(y_labelsize)
        ax.tick_params(axis="both", labelsize=tick_labelsize)
        ax.tick_params(axis="x", labelsize=model_tick_labelsize)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(legend_fontsize)


def _add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.06,
    y: float = 1.045,
    fontsize: float = MANUSCRIPT_PANEL_LABEL_SIZE,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
        zorder=10,
        clip_on=False,
    )


def plot_ranked_thermobarometry_this_study(
    cpx_only_workflows: Any,
    cpx_liq_workflows: Any,
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    selection_threshold: float = 50.0,
    model_selection_mode: str = "cumulative",
    cumulative_selection_threshold: float = 90.0,
    pressure_ylim: Optional[tuple[float, float]] = None,
    pressure_ticks: Optional[Sequence[float]] = None,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    depth_tick_step: float = 5,
    depth_max: float = 37,
    figsize: tuple[float, float] = (20.5, 16),
    constrained_layout: bool = True,
    subplot_hspace: Optional[float] = 0.08,
    add_legend: bool = True,
    panel_labels: tuple[str, str] = ("(a)", "(b)"),
    save_path: Optional[str | Path] = None,
    use_model_abbreviations: bool = False,
    tick_labelsize: float = MANUSCRIPT_THIS_STUDY_TICK_LABEL_SIZE,
    model_tick_labelsize: float = MANUSCRIPT_THIS_STUDY_MODEL_TICK_LABEL_SIZE,
    show_quartile_interval: bool = True,
    show_uncertainty_band: bool = False,
    show_rank_percent_annotations: bool = False,
    show_selected_model_summary: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot the notebook-style two-panel "this study" summary figure.

    Parameters
    ----------
    cpx_only_workflows, cpx_liq_workflows
        Workflow bundles for the `cpx_only` and `cpx_liq` branches. Each input
        may be a :class:`ThermobarometryWorkflowBundle` or a mapping with keys
        like ``pressure_2006`` / ``temperature_2010``.
    pressure_columns, temperature_columns
        Mappings with keys ``"cpx_only"`` and ``"cpx_liq"`` whose values are
        the model-result column names to plot in left-to-right order.
    pressure_model_pool, temperature_model_pool : sequence, optional
        Model objects used to draw uncertainty bands. Their ``model_name`` and
        ``uncertainty`` attributes are read when available.
    selection_threshold : float, default 50.0
        Threshold used by ``model_selection_mode="top1_or_top2"``. If the
        top-ranked model frequency is above this percentage, only that model
        is highlighted. Otherwise the top two are highlighted. Also used by
        ``model_selection_mode="above_threshold"`` to select every model with
        a frequency strictly greater than this percentage.
    model_selection_mode : {"cumulative", "top1_or_top2", "above_threshold"}, default "cumulative"
        Strategy used to select favored models from the ranked frequency list.
    cumulative_selection_threshold : float, default 90.0
        Cumulative percentage target used by ``model_selection_mode="cumulative"``.
    pressure_ylim, pressure_ticks : optional
        Optional fixed pressure-axis limits/ticks for the pressure panel.
    subplot_hspace : float, optional
        Vertical spacing between the pressure and temperature panels.
    save_path : path-like, optional
        If given, save the figure after creation.

    Returns
    -------
    fig, axes
        Matplotlib figure and the two panel axes.
    """
    state = _build_ranked_thermobarometry_state(
        cpx_only_workflows,
        cpx_liq_workflows,
        pressure_columns,
        temperature_columns,
        pressure_model_pool=pressure_model_pool,
        temperature_model_pool=temperature_model_pool,
    )

    fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=constrained_layout)
    if subplot_hspace is not None:
        if constrained_layout:
            fig.set_constrained_layout_pads(hspace=subplot_hspace)
        else:
            fig.subplots_adjust(hspace=subplot_hspace)

    _plot_ranked_this_study_kind_panel(
        axes[0],
        "P",
        state,
        selection_threshold=selection_threshold,
        model_selection_mode=model_selection_mode,
        cumulative_selection_threshold=cumulative_selection_threshold,
        pressure_ylim=pressure_ylim,
        pressure_ticks=pressure_ticks,
        densities_kg_m3=densities_kg_m3,
        layer_boundaries_km=layer_boundaries_km,
        depth_tick_step=depth_tick_step,
        depth_max=depth_max,
        use_model_abbreviations=use_model_abbreviations,
        axis_label_fontsize=MANUSCRIPT_THIS_STUDY_AXIS_LABEL_SIZE,
        y_axis_label_fontsize=MANUSCRIPT_THIS_STUDY_Y_AXIS_LABEL_SIZE,
        tick_labelsize=tick_labelsize,
        model_tick_labelsize=model_tick_labelsize,
        group_label_fontsize=MANUSCRIPT_THIS_STUDY_GROUP_LABEL_SIZE,
        annotation_fontsize=MANUSCRIPT_THIS_STUDY_ANNOTATION_SIZE,
        show_quartile_interval=show_quartile_interval,
        show_uncertainty_band=show_uncertainty_band,
        show_rank_percent_annotations=show_rank_percent_annotations,
        show_selected_model_summary=show_selected_model_summary,
    )
    _plot_ranked_this_study_kind_panel(
        axes[1],
        "T",
        state,
        selection_threshold=selection_threshold,
        model_selection_mode=model_selection_mode,
        cumulative_selection_threshold=cumulative_selection_threshold,
        depth_tick_step=depth_tick_step,
        depth_max=depth_max,
        use_model_abbreviations=use_model_abbreviations,
        axis_label_fontsize=MANUSCRIPT_THIS_STUDY_AXIS_LABEL_SIZE,
        y_axis_label_fontsize=MANUSCRIPT_THIS_STUDY_Y_AXIS_LABEL_SIZE,
        tick_labelsize=tick_labelsize,
        model_tick_labelsize=model_tick_labelsize,
        group_label_fontsize=MANUSCRIPT_THIS_STUDY_GROUP_LABEL_SIZE,
        annotation_fontsize=MANUSCRIPT_THIS_STUDY_ANNOTATION_SIZE,
        show_quartile_interval=show_quartile_interval,
        show_uncertainty_band=show_uncertainty_band,
        show_rank_percent_annotations=show_rank_percent_annotations,
        show_selected_model_summary=show_selected_model_summary,
    )

    if add_legend:
        legend_items = [
            Patch(facecolor="blue", edgecolor="k", alpha=1.0, label="2006 eruption"),
            Patch(facecolor="red", edgecolor="k", alpha=1.0, label="2010 eruption"),
            Patch(facecolor="white", edgecolor="k", linestyle="--", linewidth=3.0, label="Selected model"),
        ]
        axes[0].legend(
            handles=legend_items,
            bbox_to_anchor=(0.02, 0.98),
            ncol=2,
            frameon=True,
            fontsize=MANUSCRIPT_THIS_STUDY_LEGEND_SIZE,
            borderpad=0.35,
            handlelength=1.6,
            handletextpad=0.5,
        )

    if panel_labels:
        _add_panel_label(axes[0], panel_labels[0])
        if len(panel_labels) > 1:
            _add_panel_label(axes[1], panel_labels[1], y=1.045)

    _apply_this_study_axis_font_sizes(fig, tick_labelsize=tick_labelsize, model_tick_labelsize=model_tick_labelsize)

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=300)

    return fig, axes


def plot_ranked_thermobarometry_literature_comparison(
    cpx_only_workflows: Any,
    cpx_liq_workflows: Any,
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    pressure_literature: Any,
    temperature_literature: Any,
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    liquid_results: Optional[Mapping[str, Any]] = None,
    selection_threshold: float = 50.0,
    model_selection_mode: str = "cumulative",
    cumulative_selection_threshold: float = 90.0,
    pressure_ylim: Optional[tuple[float, float]] = None,
    temperature_ylim: Optional[tuple[float, float]] = (900, 1200),
    pressure_ticks: Optional[Sequence[float]] = None,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    literature_pressure_source: str = "pressure",
    depth_tick_step: float = 5,
    depth_max: float = 37,
    figsize: tuple[float, float] = (16.5, 16),
    constrained_layout: bool = True,
    add_literature_legend: bool = True,
    panel_labels: tuple[str, ...] = ("(a)", "(b)"),
    save_path: Optional[str | Path] = None,
    use_model_abbreviations: bool = False,
    pressure_reservoir_bands: Optional[Sequence[Mapping[str, Any]]] = None,
    include_temperature: bool = True,
    split_literature_legend: bool = False,
    literature_legend_bbox_to_anchor: tuple[float, float] = (0.58, 0.50),
    literature_model_tick_labelsize: float = MANUSCRIPT_MODEL_TICK_LABEL_SIZE,
    literature_reservoir_label_x_offset: float = 0.95,
    literature_reservoir_label_right_margin: float = 0.10,
    literature_this_study_x_spacing: float = 1.0,
    show_quartile_interval: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot the notebook-style two-panel "this study vs literature" comparison.

    Parameters
    ----------
    pressure_literature, temperature_literature
        Either DataFrames or Excel paths. If a ``plot`` column is present it is
        used as a boolean include mask, matching the notebook workflow.

    Returns
    -------
    fig, axes
        Matplotlib figure and the two panel axes.
    """
    state = _build_ranked_thermobarometry_state(
        cpx_only_workflows,
        cpx_liq_workflows,
        pressure_columns,
        temperature_columns,
        pressure_model_pool=pressure_model_pool,
        temperature_model_pool=temperature_model_pool,
        liquid_results=liquid_results,
    )
    pressure_literature_df = _load_literature_table(pressure_literature, "pressure_literature")
    temperature_literature_df = (
        _load_literature_table(temperature_literature, "temperature_literature")
        if include_temperature
        else None
    )

    if include_temperature:
        fig, axes_obj = plt.subplots(2, 1, figsize=figsize, constrained_layout=constrained_layout)
        axes = np.asarray(axes_obj)
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=constrained_layout)
        axes = np.asarray([ax])

    _plot_ranked_literature_panel(
        axes[0],
        "P",
        state,
        pressure_literature_df,
        selection_threshold=selection_threshold,
        model_selection_mode=model_selection_mode,
        cumulative_selection_threshold=cumulative_selection_threshold,
        add_literature_legend=add_literature_legend,
        pressure_ylim=pressure_ylim,
        pressure_ticks=pressure_ticks,
        densities_kg_m3=densities_kg_m3,
        layer_boundaries_km=layer_boundaries_km,
        literature_pressure_source=literature_pressure_source,
        depth_tick_step=depth_tick_step,
        depth_max=depth_max,
        use_model_abbreviations=use_model_abbreviations,
        model_tick_labelsize=literature_model_tick_labelsize,
        pressure_reservoir_bands=pressure_reservoir_bands,
        split_literature_legend=split_literature_legend,
        literature_legend_bbox_to_anchor=literature_legend_bbox_to_anchor,
        reservoir_label_x_offset=literature_reservoir_label_x_offset,
        reservoir_label_right_margin=literature_reservoir_label_right_margin,
        this_study_x_spacing=literature_this_study_x_spacing,
        show_quartile_interval=show_quartile_interval,
    )
    if include_temperature and temperature_literature_df is not None:
        _plot_ranked_literature_panel(
            axes[1],
            "T",
            state,
            temperature_literature_df,
            selection_threshold=selection_threshold,
            model_selection_mode=model_selection_mode,
            cumulative_selection_threshold=cumulative_selection_threshold,
            add_literature_legend=False,
            temperature_ylim=temperature_ylim,
            depth_tick_step=depth_tick_step,
            depth_max=depth_max,
            use_model_abbreviations=use_model_abbreviations,
            model_tick_labelsize=literature_model_tick_labelsize,
            this_study_x_spacing=literature_this_study_x_spacing,
            show_quartile_interval=show_quartile_interval,
        )

    if panel_labels:
        for ax_i, label in enumerate(panel_labels[: len(axes)]):
            _add_panel_label(axes[ax_i], label)

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=300)

    return fig, axes


def plot_ranked_thermobarometry_original_vs_pre_2006_comparison(
    original_cpx_only: Any,
    original_cpx_liq: Any,
    pre_2006_cpx_only: Any,
    pre_2006_cpx_liq: Any,
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    selection_threshold: float = 50.0,
    model_selection_mode: str = "cumulative",
    cumulative_selection_threshold: float = 80.0,
    pressure_ylim: Optional[tuple[float, float]] = (-1, 10),
    temperature_ylim: Optional[tuple[float, float]] = None,
    figsize: tuple[float, float] = (14.0, 11.0),
    panel_labels: tuple[str, ...] = ("(a)", "(b)"),
    save_path: Optional[str | Path] = None,
    use_model_abbreviations: bool = True,
    show_quartile_interval: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a two-panel comparison of original and +pre-2006 cpx-liquid results.

    Each panel compares original and +pre-2006 clinopyroxene-liquid selected-model
    prediction distributions for the 2006 and 2010 eruptions.
    """
    original_state = _build_ranked_thermobarometry_state(
        original_cpx_only,
        original_cpx_liq,
        pressure_columns,
        temperature_columns,
        pressure_model_pool=pressure_model_pool,
        temperature_model_pool=temperature_model_pool,
    )
    pre_2006_state = _build_ranked_thermobarometry_state(
        pre_2006_cpx_only,
        pre_2006_cpx_liq,
        pressure_columns,
        temperature_columns,
        pressure_model_pool=pressure_model_pool,
        temperature_model_pool=temperature_model_pool,
    )

    fig, axes_obj = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.14)
    axes = np.asarray(axes_obj)

    _plot_original_vs_pre_2006_kind_panel(
        axes[0],
        "P",
        original_state,
        pre_2006_state,
        selection_threshold=selection_threshold,
        model_selection_mode=model_selection_mode,
        cumulative_selection_threshold=cumulative_selection_threshold,
        pressure_ylim=pressure_ylim,
        temperature_ylim=temperature_ylim,
        use_model_abbreviations=use_model_abbreviations,
        show_quartile_interval=show_quartile_interval,
    )
    _plot_original_vs_pre_2006_kind_panel(
        axes[1],
        "T",
        original_state,
        pre_2006_state,
        selection_threshold=selection_threshold,
        model_selection_mode=model_selection_mode,
        cumulative_selection_threshold=cumulative_selection_threshold,
        pressure_ylim=pressure_ylim,
        temperature_ylim=temperature_ylim,
        use_model_abbreviations=use_model_abbreviations,
        show_quartile_interval=show_quartile_interval,
    )

    year_2006_handle = (
        Patch(facecolor="blue", alpha=0.55),
        Line2D([0], [0], color="blue", lw=2.2, alpha=1.0),
    )
    year_2010_handle = (
        Patch(facecolor="red", alpha=0.55),
        Line2D([0], [0], color="red", lw=2.2, alpha=1.0),
    )
    legend_items = [
        year_2006_handle,
        year_2010_handle,
        Patch(facecolor="0.60", edgecolor="0.25", alpha=0.55),
        Patch(facecolor="none", edgecolor="0.25", linewidth=1.4),
    ]
    axes[0].legend(
        handles=legend_items,
        labels=["2006", "2010", "all samples", "favored subset"],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.25),
        frameon=True,
        ncol=2,
        fontsize=MANUSCRIPT_THIS_STUDY_LEGEND_SIZE,
        borderpad=0.35,
        columnspacing=1.75,
        handlelength=2.1,
        handletextpad=0.5,
        labelspacing=0.25,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )

    if panel_labels:
        for ax_i, label in enumerate(panel_labels[: len(axes)]):
            _add_panel_label(axes[ax_i], label, y = 1.1)

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=300)

    return fig, axes



def plot_ranked_thermobarometry_kd_comparison(
    kd_workflow_bundles: Mapping[str, Mapping[str, Any]],
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    selection_threshold: float = 50.0,
    model_selection_mode: str = "cumulative",
    cumulative_selection_threshold: float = 80.0,
    pressure_ylim: Optional[tuple[float, float]] = (-1, 10),
    temperature_ylim: Optional[tuple[float, float]] = None,
    figsize: tuple[float, float] = (20.0, 15.0),
    panel_labels: tuple[str, ...] = ("(a) Pressure", "(b) Temperature"),
    save_path: Optional[str | Path] = None,
    use_model_abbreviations: bool = True,
    show_quartile_interval: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a two-panel cpx-liquid comparison across Kd workflow groups.

    ``kd_workflow_bundles`` preserves insertion order for the plotted Kd groups.
    Each value must contain ``"cpx_only"`` and ``"cpx_liq"`` workflow bundles.
    """
    kd_states: dict[str, dict[str, Any]] = {}
    for kd_label, bundles in kd_workflow_bundles.items():
        if not isinstance(bundles, Mapping):
            raise TypeError(f"Workflow group {kd_label!r} must be a mapping.")
        if "cpx_only" not in bundles or "cpx_liq" not in bundles:
            raise KeyError(f"Workflow group {kd_label!r} must contain 'cpx_only' and 'cpx_liq'.")
        kd_states[kd_label] = _build_ranked_thermobarometry_state(
            bundles["cpx_only"],
            bundles["cpx_liq"],
            pressure_columns,
            temperature_columns,
            pressure_model_pool=pressure_model_pool,
            temperature_model_pool=temperature_model_pool,
        )

    fig, axes_obj = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.14)
    axes = np.asarray(axes_obj)

    _plot_kd_comparison_kind_panel(
        axes[0],
        "P",
        kd_states,
        selection_threshold=selection_threshold,
        model_selection_mode=model_selection_mode,
        cumulative_selection_threshold=cumulative_selection_threshold,
        pressure_ylim=pressure_ylim,
        temperature_ylim=temperature_ylim,
        use_model_abbreviations=use_model_abbreviations,
        show_quartile_interval=show_quartile_interval,
    )
    _plot_kd_comparison_kind_panel(
        axes[1],
        "T",
        kd_states,
        selection_threshold=selection_threshold,
        model_selection_mode=model_selection_mode,
        cumulative_selection_threshold=cumulative_selection_threshold,
        pressure_ylim=pressure_ylim,
        temperature_ylim=temperature_ylim,
        use_model_abbreviations=use_model_abbreviations,
        show_quartile_interval=show_quartile_interval,
    )

    year_2006_handle = (
        Patch(facecolor="blue", alpha=0.55),
        Line2D([0], [0], color="blue", lw=2.2, alpha=1.0),
    )
    year_2010_handle = (
        Patch(facecolor="red", alpha=0.55),
        Line2D([0], [0], color="red", lw=2.2, alpha=1.0),
    )
    legend_items = [
        year_2006_handle,
        year_2010_handle,
        Patch(facecolor="0.60", edgecolor="0.25", alpha=0.55),
        Patch(facecolor="none", edgecolor="0.25", linewidth=1.4),
    ]
    axes[0].legend(
        handles=legend_items,
        labels=["2006", "2010", "all samples", "favored subset"],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.25),
        frameon=True,
        ncol=2,
        fontsize=MANUSCRIPT_THIS_STUDY_LEGEND_SIZE,
        borderpad=0.35,
        columnspacing=1.75,
        handlelength=2.1,
        handletextpad=0.5,
        labelspacing=0.25,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )

    if panel_labels:
        for ax_i, label in enumerate(panel_labels[: len(axes)]):
            _add_panel_label(axes[ax_i], label, y=1.1)

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=300)

    return fig, axes

def _format_rank_pie_labels(
    rank_pcts: Sequence[tuple[str, float]],
    kind: str,
    *,
    use_model_abbreviations: bool,
    pct_label_cutoff: Optional[float] = None,
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    labels: list[str] = []
    cumulative = 0.0
    for model_name, pct in rank_pcts:
        pct_value = float(pct)
        if not np.isfinite(pct_value) or pct_value <= 0:
            continue
        model_label = _model_axis_label(model_name, kind, use_model_abbreviations=use_model_abbreviations)
        model_label = _wrap_label(model_label, width=9)
        if pct_label_cutoff is None or cumulative < pct_label_cutoff:
            labels.append(f"{model_label}\n{pct_value:.0f}%")
        else:
            labels.append(model_label)
        values.append(pct_value)
        cumulative += pct_value
    return values, labels


def _label_fits_inside_rank_pie(label: str, pct_value: float) -> bool:
    return pct_value >= 5.0


def _single_line_rank_pie_model_label(label: str) -> str:
    lines = [line.strip() for line in str(label).splitlines() if line.strip()]
    if len(lines) >= 2 and lines[-1].endswith("%"):
        lines = lines[:-1]
    return " ".join(lines)


def _spread_rank_pie_outside_labels(items: list[dict[str, Any]], *, min_gap: float = 0.15, y_limit: float = 1.18) -> None:
    if not items:
        return
    if len(items) > 1:
        min_gap = min(min_gap, (2.0 * y_limit) / (len(items) - 1))
    items.sort(key=lambda item: item["y"])
    for idx in range(1, len(items)):
        if items[idx]["y"] - items[idx - 1]["y"] < min_gap:
            items[idx]["y"] = items[idx - 1]["y"] + min_gap
    top_shift = max(0.0, items[-1]["y"] - y_limit)
    bottom_shift = max(0.0, -y_limit - items[0]["y"])
    shift = bottom_shift - top_shift
    if shift:
        for item in items:
            item["y"] += shift
    for idx in range(len(items) - 2, -1, -1):
        if items[idx + 1]["y"] - items[idx]["y"] < min_gap:
            items[idx]["y"] = items[idx + 1]["y"] - min_gap
    for item in items:
        item["y"] = float(np.clip(item["y"], -y_limit, y_limit))


def _draw_rank_pie_with_mixed_labels(
    ax: plt.Axes,
    values: Sequence[float],
    labels: Sequence[str],
    *,
    colors: Sequence[Any],
    startangle: float = 90.0,
    counterclock: bool = False,
    inside_fontsize: float = 13.5,
    outside_fontsize: float = 9.6,
    inside_label_radius: float = 0.78,
    outside_connector_lw: float = 1.15,
) -> None:
    wedges, _ = ax.pie(
        values,
        labels=None,
        startangle=startangle,
        counterclock=counterclock,
        colors=colors[: len(values)],
        radius=1.05,
        wedgeprops={"linewidth": 0.45, "edgecolor": "white"},
    )
    outside_items: list[dict[str, Any]] = []
    total = float(np.nansum(values))
    for wedge, value, label in zip(wedges, values, labels):
        pct_value = 100.0 * float(value) / total if total > 0 else 0.0
        angle = 0.5 * (wedge.theta1 + wedge.theta2)
        angle_rad = np.deg2rad(angle)
        x_unit = float(np.cos(angle_rad))
        y_unit = float(np.sin(angle_rad))
        if _label_fits_inside_rank_pie(label, pct_value):
            face_rgb = np.asarray(wedge.get_facecolor()[:3], dtype=float)
            luminance = float(np.dot(face_rgb, [0.299, 0.587, 0.114]))
            text_color = "white" if luminance < 0.48 else "black"
            text_effects = [pe.Stroke(linewidth=1.4, foreground="black" if text_color == "white" else "white", alpha=0.58), pe.Normal()]
            ax.text(
                inside_label_radius * x_unit,
                inside_label_radius * y_unit,
                label,
                ha="center",
                va="center",
                fontsize=inside_fontsize,
                color=text_color,
                linespacing=0.92,
                zorder=4,
                path_effects=text_effects,
            )
        else:
            outside_items.append(
                {
                    "label": _single_line_rank_pie_model_label(label),
                    "x": x_unit,
                    "y": 1.02 * y_unit,
                    "xy": (0.96 * x_unit, 0.96 * y_unit),
                    "side": 1 if x_unit >= 0 else -1,
                }
            )

    for side in (-1, 1):
        side_items = [item for item in outside_items if item["side"] == side]
        _spread_rank_pie_outside_labels(side_items, min_gap=0.17, y_limit=1.08)
        for item in side_items:
            x_text = 1.14 * side
            ax.annotate(
                item["label"],
                xy=item["xy"],
                xytext=(x_text, item["y"]),
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=outside_fontsize,
                linespacing=0.92,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "0.35",
                    "lw": outside_connector_lw,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "connectionstyle": "arc3,rad=0.0",
                },
                zorder=5,
            )


def _rank_pie_split_save_path(save_path: str | Path, kind: str, phase_type: str) -> Path:
    save_path = Path(save_path)
    return save_path.with_name(f"{save_path.stem}_{kind}_{phase_type}{save_path.suffix}")


def _save_figure_png_pdf_from_path(
    fig: plt.Figure,
    save_path: str | Path,
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
    **savefig_kwargs: Any,
) -> tuple[Path, Path]:
    """Save a figure as one PNG and one PDF using the same file stem."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.suffix.lower() == ".pdf":
        pdf_path = save_path
        png_path = save_path.with_suffix(".png")
    else:
        png_path = save_path.with_suffix(".png")
        pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches=bbox_inches, **savefig_kwargs)
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
    return pdf_path, png_path


def _draw_rank_pie_year_pair(
    state: Mapping[str, Any],
    kind: str,
    phase_type: str,
    *,
    use_model_abbreviations: bool,
    figsize: tuple[float, float],
    dpi: int,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes_obj = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    axes = np.asarray(axes_obj)
    colors = list(plt.cm.tab20.colors)
    for ax, year in zip(axes, ("2006", "2010")):
        rank_pcts = state[kind]["ranks"][year][phase_type]
        values, labels = _format_rank_pie_labels(
            rank_pcts,
            kind,
            use_model_abbreviations=use_model_abbreviations,
        )
        if values:
            _draw_rank_pie_with_mixed_labels(
                ax,
                values,
                labels,
                colors=colors,
                startangle=90,
                counterclock=False,
            )
        ax.set_title(year, fontsize=13.0)
        ax.set_aspect("equal")
    fig.suptitle(f"{kind} {_display_phase_type_label(phase_type)}", fontsize=15.0, fontweight="bold")
    return fig, axes


def plot_ranked_thermobarometry_rank_pies_split(
    cpx_only_workflows: Any,
    cpx_liq_workflows: Any,
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    use_model_abbreviations: bool = True,
    figsize: tuple[float, float] = (6.2, 3.7),
    dpi: int = 200,
    save_path: Optional[str | Path] = None,
) -> dict[str, tuple[plt.Figure, np.ndarray]]:
    """Plot four rank-pie figures, each comparing 2006 and 2010."""
    state = _build_ranked_thermobarometry_state(
        cpx_only_workflows,
        cpx_liq_workflows,
        pressure_columns,
        temperature_columns,
        pressure_model_pool=pressure_model_pool,
        temperature_model_pool=temperature_model_pool,
    )
    figures: dict[str, tuple[plt.Figure, np.ndarray]] = {}
    for kind in ("P", "T"):
        for phase_type in ("cpx_only", "cpx_liq"):
            fig_axes = _draw_rank_pie_year_pair(
                state,
                kind,
                phase_type,
                use_model_abbreviations=use_model_abbreviations,
                figsize=figsize,
                dpi=dpi,
            )
            key = f"rank_pies_{kind}_{phase_type}"
            figures[key] = fig_axes
            if save_path is not None:
                fig_axes[0].savefig(
                    _rank_pie_split_save_path(save_path, kind, phase_type),
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0.03,
                )
    return figures


def plot_ranked_thermobarometry_rank_pies(
    cpx_only_workflows: Any,
    cpx_liq_workflows: Any,
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    use_model_abbreviations: bool = True,
    figsize: tuple[float, float] = (15.0, 8.0),
    dpi: int = 200,
    save_path: Optional[str | Path] = None,
    inside_label_fontsize: float = 24.0,
    outside_label_fontsize: float = 18.0,
    inside_label_radius: float = 0.6,
    outside_connector_lw: float = 1.5,
    year_title_fontsize: float = 30.0,
    year_title_pad: float = 0.0,
    group_title_fontsize: float = 18.0,
    group_title_y: float = 1.14,
    pair_wspace: float = -0.28,
    group_wspace: float = 0.06,
    group_hspace: float = 0.10,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot model-selection percentage pies for P/T, phase type, and eruption year."""
    state = _build_ranked_thermobarometry_state(
        cpx_only_workflows,
        cpx_liq_workflows,
        pressure_columns,
        temperature_columns,
        pressure_model_pool=pressure_model_pool,
        temperature_model_pool=temperature_model_pool,
    )
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, wspace=0.02, hspace=0.04)
    outer = fig.add_gridspec(2, 2, wspace=group_wspace, hspace=group_hspace)
    axes = np.empty((2, 4), dtype=object)
    phase_order = ("cpx_only", "cpx_liq")
    year_order = ("2006", "2010")
    colors = list(plt.cm.tab20.colors)

    for row_idx, kind in enumerate(("P", "T")):
        for phase_idx, phase_type in enumerate(phase_order):
            inner = outer[row_idx, phase_idx].subgridspec(1, 2, wspace=pair_wspace)
            for year_idx, year in enumerate(year_order):
                ax = fig.add_subplot(inner[0, year_idx])
                axes[row_idx, phase_idx * 2 + year_idx] = ax
                rank_pcts = state[kind]["ranks"][year][phase_type]
                values, labels = _format_rank_pie_labels(
                    rank_pcts,
                    kind,
                    use_model_abbreviations=use_model_abbreviations,
                )
                if values:
                    _draw_rank_pie_with_mixed_labels(
                        ax,
                        values,
                        labels,
                        colors=colors,
                        startangle=90,
                        counterclock=False,
                        inside_fontsize=inside_label_fontsize,
                        outside_fontsize=outside_label_fontsize,
                        inside_label_radius=inside_label_radius,
                        outside_connector_lw=outside_connector_lw,
                    )
                ax.set_title(year, fontsize=year_title_fontsize, pad=year_title_pad)
                ax.set_aspect("equal")

            left_ax = axes[row_idx, phase_idx * 2]
            left_ax.text(
                1.0,
                group_title_y,
                f"{kind} {_display_phase_type_label(phase_type)}",
                transform=left_ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=group_title_fontsize,
                fontweight="bold",
                clip_on=False,
            )

    if save_path is not None:
        _save_figure_png_pdf_from_path(fig, save_path, dpi=dpi, bbox_inches="tight")
    return fig, axes

def export_ranked_thermobarometry_model_summary(
    cpx_only_workflows: Any,
    cpx_liq_workflows: Any,
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    out_path: str | Path,
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    selection_threshold: float = 50.0,
    model_selection_mode: str = "cumulative",
    cumulative_selection_threshold: float = 80.0,
) -> dict[str, pd.DataFrame]:
    """Export Fig. 8-9 model summaries using the same Tukey-fence state."""
    state = _build_ranked_thermobarometry_state(
        cpx_only_workflows,
        cpx_liq_workflows,
        pressure_columns,
        temperature_columns,
        pressure_model_pool=pressure_model_pool,
        temperature_model_pool=temperature_model_pool,
    )

    def _summary_value(values: np.ndarray, reducer: str) -> float:
        if values.size == 0:
            return np.nan
        if reducer == "min":
            return float(np.nanmin(values))
        if reducer == "max":
            return float(np.nanmax(values))
        if reducer == "median":
            return float(np.nanmedian(values))
        raise ValueError(f"Unsupported reducer: {reducer}")

    def _depth_value(pressure_kbar: float) -> float:
        if densities_kg_m3 is None or layer_boundaries_km is None or not np.isfinite(pressure_kbar):
            return np.nan
        return float(
            _pressure_to_depth_km(
                pressure_kbar,
                densities_kg_m3=densities_kg_m3,
                layer_boundaries_km=layer_boundaries_km,
            )
        )

    def _raw_model_values(df_pred: pd.DataFrame, model_name: str, label: str) -> np.ndarray:
        if model_name not in df_pred.columns:
            raise KeyError(f"{label} is missing column {model_name!r}.")
        raw_values = pd.to_numeric(df_pred[model_name], errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )
        return raw_values.dropna().to_numpy(dtype=float)

    def _selected_model_names(rank_pcts: Sequence[tuple[str, float]]) -> set[str]:
        selected_rank_ids = _selected_rank_set_from_topk(
            rank_pcts,
            threshold=selection_threshold,
            model_selection_mode=model_selection_mode,
            cumulative_threshold=cumulative_selection_threshold,
        )
        return {
            model_name
            for rank_i, (model_name, _) in enumerate(rank_pcts)
            if rank_i in selected_rank_ids
        }

    def _summary_row(
        kind: str,
        eruption: str,
        phase_type: str,
        model_name: str,
        raw_array: np.ndarray,
        rank_pct_map: Mapping[str, float],
        selected_models: set[str],
        *,
        proportion_samples_favored_model: Optional[float] = None,
        selected_by_AIMS4PT: Optional[bool] = None,
    ) -> dict[str, Any]:
        tukey_array = _remove_boxplot_outliers(raw_array)
        row = {
            "quantity": kind,
            "eruption": eruption,
            "phase_type": phase_type,
            "model": model_name,
            "min": _summary_value(tukey_array, "min"),
            "max": _summary_value(tukey_array, "max"),
            "median": _summary_value(tukey_array, "median"),
            "n_samples_raw": int(raw_array.size),
            "n_samples_after_tukey": int(tukey_array.size),
            "proportion_samples_favored_model": (
                rank_pct_map.get(model_name, 0.0) / 100.0
                if proportion_samples_favored_model is None
                else float(proportion_samples_favored_model)
            ),
            "selected_by_AIMS4PT": (
                model_name in selected_models
                if selected_by_AIMS4PT is None
                else bool(selected_by_AIMS4PT)
            ),
        }
        if kind == "P":
            row.update(
                {
                    "min_depth_km": _depth_value(row["min"]),
                    "max_depth_km": _depth_value(row["max"]),
                    "median_depth_km": _depth_value(row["median"]),
                }
            )
        return row

    def _build_kind_summary(kind: str) -> pd.DataFrame:
        kind_state = state[kind]
        rows: list[dict[str, Any]] = []

        for eruption in ("2006", "2010"):
            for phase_type in ("cpx_only", "cpx_liq"):
                df_pred = kind_state["results"][eruption][phase_type]
                rank_pcts = kind_state["ranks"][eruption][phase_type]
                rank_pct_map = {model_name: float(pct) for model_name, pct in rank_pcts}
                selected_models = _selected_model_names(rank_pcts)

                for model_name in kind_state["columns"][phase_type]:
                    raw_array = _raw_model_values(df_pred, model_name, f"{kind} results {eruption} {phase_type}")
                    rows.append(
                        _summary_row(
                            kind,
                            eruption,
                            phase_type,
                            model_name,
                            raw_array,
                            rank_pct_map,
                            selected_models,
                        )
                    )

                overall_array = _selected_predictions_from_best_models(
                    df_pred,
                    kind_state["best_models"][eruption][phase_type],
                )
                rows.append(
                    _summary_row(
                        kind,
                        eruption,
                        phase_type,
                        "overall",
                        overall_array,
                        {},
                        set(),
                        proportion_samples_favored_model=1.0,
                        selected_by_AIMS4PT=True,
                    )
                )

        return pd.DataFrame(rows)

    summaries = {
        "P_summary": _build_kind_summary("P"),
        "T_summary": _build_kind_summary("T"),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as writer:
        for sheet_name, summary_df in summaries.items():
            summary_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return summaries


def _show_ranked_thermobarometry_figures(figures: Mapping[str, tuple[plt.Figure, np.ndarray]]) -> None:
    backend = plt.get_backend().lower()
    if backend.endswith("agg"):
        try:
            from IPython.display import display
        except ImportError:
            return

        # Agg cannot show interactively, but notebooks can still render figures.
        for fig, _axes in figures.values():
            display(fig)
        return

    plt.show()


def plot_ranked_thermobarometry_summary(
    cpx_only_workflows: Any,
    cpx_liq_workflows: Any,
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    pressure_literature: Any = None,
    temperature_literature: Any = None,
    liquid_results: Optional[Mapping[str, Any]] = None,
    selection_threshold: float = 50.0,
    model_selection_mode: str = "cumulative",
    cumulative_selection_threshold: float = 80.0,
    pressure_ylim: Optional[tuple[float, float]] = None,
    temperature_ylim: Optional[tuple[float, float]] = (900, 1150),
    pressure_ticks: Optional[Sequence[float]] = None,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    literature_pressure_source: str = "pressure",
    depth_tick_step: float = 5,
    depth_max: float = 37,
    this_study_figsize: tuple[float, float] = (20.5, 16),
    literature_figsize: tuple[float, float] = (16.5, 16),
    this_study_save_path: Optional[str | Path] = None,
    literature_save_path: Optional[str | Path] = None,
    this_study_subplot_hspace: Optional[float] = 0.08,
    show: bool = True,
    use_model_abbreviations: bool = False,
    pressure_reservoir_bands: Optional[Sequence[Mapping[str, Any]]] = None,
    literature_include_temperature: bool = True,
    literature_panel_labels: tuple[str, ...] = ("(a)", "(b)"),
    this_study_tick_labelsize: float = MANUSCRIPT_THIS_STUDY_TICK_LABEL_SIZE,
    this_study_model_tick_labelsize: float = MANUSCRIPT_THIS_STUDY_MODEL_TICK_LABEL_SIZE,
    split_literature_legend: bool = False,
    literature_legend_bbox_to_anchor: tuple[float, float] = (0.58, 0.50),
    literature_model_tick_labelsize: Optional[float] = None,
    literature_reservoir_label_x_offset: float = 0.95,
    literature_reservoir_label_right_margin: float = 0.10,
    literature_this_study_x_spacing: float = 1.0,
    show_quartile_interval: bool = False,
    show_uncertainty_band: bool = False,
    show_rank_percent_annotations: bool = False,
    show_selected_model_summary: bool = True,
    rank_pie_save_path: Optional[str | Path] = None,
    rank_pie_figsize: tuple[float, float] = (23.0, 12.0),
    rank_pie_dpi: int = 200,
    rank_pie_inside_label_fontsize: float = 24.0,
    rank_pie_outside_label_fontsize: float = 18.0,
    rank_pie_inside_label_radius: float = 0.6,
    rank_pie_outside_connector_lw: float = 1.5,
    rank_pie_pair_wspace: float = -0.28,
    rank_pie_year_title_fontsize: float = 30.0,
    rank_pie_year_title_pad: float = 0.0,
    rank_pie_group_title_fontsize: float = 18.0,
    rank_pie_group_title_y: float = 1.14,
) -> dict[str, tuple[plt.Figure, np.ndarray]]:
    """
    Convenience wrapper that reproduces both notebook summary figures.

    Parameters
    ----------
    selection_threshold : float, default 50.0
        Threshold used by ``model_selection_mode="top1_or_top2"`` and by
        ``model_selection_mode="above_threshold"``.
    model_selection_mode : {"cumulative", "top1_or_top2", "above_threshold"}, default "cumulative"
        Strategy used by both summary figures to select favored models.
    cumulative_selection_threshold : float, default 80.0
        Cumulative percentage target used by ``model_selection_mode="cumulative"``.
    rank_pie_save_path : path-like, optional
        If provided, save the combined rank-pie figure as both PNG and PDF
        using this path's file stem.
    show : bool, default True
        Whether to display the generated figures before returning them.

    Returns
    -------
    dict
        Always includes ``"this_study"``. Includes ``"rank_pies"`` when
        ``rank_pie_save_path`` is provided. Includes
        ``"literature_comparison"`` only when both literature tables are
        provided.
    """
    literature_model_tick_labelsize = (
        this_study_model_tick_labelsize
        if literature_model_tick_labelsize is None
        else literature_model_tick_labelsize
    )

    figures = {
        "this_study": plot_ranked_thermobarometry_this_study(
            cpx_only_workflows,
            cpx_liq_workflows,
            pressure_columns,
            temperature_columns,
            pressure_model_pool=pressure_model_pool,
            temperature_model_pool=temperature_model_pool,
            selection_threshold=selection_threshold,
            model_selection_mode=model_selection_mode,
            cumulative_selection_threshold=cumulative_selection_threshold,
            pressure_ylim=pressure_ylim,
            pressure_ticks=pressure_ticks,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
            depth_tick_step=depth_tick_step,
            depth_max=depth_max,
            figsize=this_study_figsize,
            subplot_hspace=this_study_subplot_hspace,
            save_path=this_study_save_path,
            use_model_abbreviations=use_model_abbreviations,
            tick_labelsize=this_study_tick_labelsize,
            model_tick_labelsize=this_study_model_tick_labelsize,
            show_quartile_interval=show_quartile_interval,
            show_uncertainty_band=show_uncertainty_band,
            show_rank_percent_annotations=show_rank_percent_annotations,
            show_selected_model_summary=show_selected_model_summary,
        )
    }

    if rank_pie_save_path is not None:
        rank_pie_fig_axes = plot_ranked_thermobarometry_rank_pies(
            cpx_only_workflows,
            cpx_liq_workflows,
            pressure_columns,
            temperature_columns,
            pressure_model_pool=pressure_model_pool,
            temperature_model_pool=temperature_model_pool,
            use_model_abbreviations=use_model_abbreviations,
            figsize=rank_pie_figsize,
            dpi=rank_pie_dpi,
            save_path=None,
            inside_label_fontsize=rank_pie_inside_label_fontsize,
            outside_label_fontsize=rank_pie_outside_label_fontsize,
            inside_label_radius=rank_pie_inside_label_radius,
            outside_connector_lw=rank_pie_outside_connector_lw,
            year_title_fontsize=rank_pie_year_title_fontsize,
            year_title_pad=rank_pie_year_title_pad,
            group_title_fontsize=rank_pie_group_title_fontsize,
            group_title_y=rank_pie_group_title_y,
            pair_wspace=rank_pie_pair_wspace,
        )
        figures["rank_pies"] = rank_pie_fig_axes
        _save_figure_png_pdf_from_path(
            rank_pie_fig_axes[0],
            rank_pie_save_path,
            dpi=rank_pie_dpi,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    if pressure_literature is not None and (temperature_literature is not None or not literature_include_temperature):
        figures["literature_comparison"] = plot_ranked_thermobarometry_literature_comparison(
            cpx_only_workflows,
            cpx_liq_workflows,
            pressure_columns,
            temperature_columns,
            pressure_literature,
            temperature_literature,
            pressure_model_pool=pressure_model_pool,
            temperature_model_pool=temperature_model_pool,
            liquid_results=liquid_results,
            selection_threshold=selection_threshold,
            model_selection_mode=model_selection_mode,
            cumulative_selection_threshold=cumulative_selection_threshold,
            pressure_ylim=pressure_ylim,
            temperature_ylim=temperature_ylim,
            pressure_ticks=pressure_ticks,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
            literature_pressure_source=literature_pressure_source,
            depth_tick_step=depth_tick_step,
            depth_max=depth_max,
            figsize=literature_figsize,
            save_path=literature_save_path,
            use_model_abbreviations=use_model_abbreviations,
            pressure_reservoir_bands=pressure_reservoir_bands,
            include_temperature=literature_include_temperature,
            panel_labels=literature_panel_labels,
            split_literature_legend=split_literature_legend,
            literature_legend_bbox_to_anchor=literature_legend_bbox_to_anchor,
            literature_model_tick_labelsize=literature_model_tick_labelsize,
            literature_reservoir_label_x_offset=literature_reservoir_label_x_offset,
            literature_reservoir_label_right_margin=literature_reservoir_label_right_margin,
            literature_this_study_x_spacing=literature_this_study_x_spacing,
            show_quartile_interval=show_quartile_interval,
        )

    if show:
        _show_ranked_thermobarometry_figures(figures)

    return figures


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
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a璺痓
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
