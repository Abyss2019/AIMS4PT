import sys
from pathlib import Path

# Allow direct execution while keeping project package imports available.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde

from paper.scripts.figure_helpers import (
    add_panel_label_yaxis_aligned,
    save_figure_pdf_png,
)


PAPER_DIR = PROJECT_ROOT / "paper"
CACHE_DIR = PAPER_DIR / ".cache" / "add_pre-2006_028"
IMAGES_DIR = PAPER_DIR / ".images"

PANEL_DATA = {
    "Clinopyroxene-only": CACHE_DIR / "per_point_final_PT_model_results_cpx_only.csv",
    "Clinopyroxene-liquid": CACHE_DIR / "per_point_final_PT_model_results.csv",
}

CPX_COMPOSITION_DATA = {
    2006: CACHE_DIR / "marapi_2006_P_workflow_cpx_only_P_add_pre-2006_028_report.xlsx",
    2010: CACHE_DIR / "marapi_2010_P_workflow_cpx_only_P_add_pre-2006_028_report.xlsx",
}

YEAR_STYLES = {
    2006: {"marker": "o", "edgecolors": "#4A4A4A"},
    2010: {"marker": "^", "edgecolors": "#8B1A1A"},
}

KDE_YEAR_STYLES = {
    2006: {"marker": "o", "edgecolors": "black"},
    2010: {"marker": "^", "edgecolors": "black"},
}

KDE_CONTOUR_COLORS = {
    2006: "#7A7A7A",
    2010: "tab:red",
}

YEAR_ZORDERS = {
    2006: 3,
    2010: 1,
}

AL2O3_COLUMN = "Al2O3_wt_pct"
AL2O3_CMAP = "Blues"
FIGURE_DPI = 200
AXIS_LABEL_SIZE = 13
YEAR_MARKER_SIZES = {
    2006: 55,
    2010: 65,
}
MARKER_EDGE_WIDTH = 1.0
KDE_MARKER_EDGE_WIDTH = 0.6

KDE_CONTOUR_ALPHA = 0.5
KDE_DASHED_CONTOUR_ALPHA = 0.3

KDE_CONTOUR_LINEWIDTHS = {
    2006: 0.8,
    2010: 1.5,
}
KDE_DASHED_LINESTYLES = {
    2006: (0, (5.0, 5.0)),
    2010: (0, (5.0, 2.5)),
}
KDE_GRID_SIZE = 180
KDE_GRID_PADDING = 0.35
KDE_CONTOUR_LEVELS = (
    (0.683, "solid", KDE_CONTOUR_ALPHA),
    # (0.955, "dashed", KDE_DASHED_CONTOUR_ALPHA),
)
COLORBAR_PAD = 0.035

# x_min, x_max = 900, 1150
# y_min, y_max = 0.8, 5.3


def load_cpx_composition():
    composition_frames = []
    for composition_path in CPX_COMPOSITION_DATA.values():
        composition = pd.read_excel(composition_path, sheet_name="Results", header=1)
        report_index_column = composition.columns[0]
        composition_frames.append(
            composition.rename(
                columns={
                    "Eruption": "Year",
                    report_index_column: "Report_row_index",
                    "Al2O3": AL2O3_COLUMN,
                }
            )[["Year", "Report_row_index", AL2O3_COLUMN]].dropna()
        )

    composition = pd.concat(composition_frames, ignore_index=True)
    composition[["Year", "Report_row_index"]] = composition[
        ["Year", "Report_row_index"]
    ].astype(int)
    return composition


def add_kde_contours(ax, data, color, linewidth, dashed_linestyle):
    x = pd.to_numeric(data["Final_T_C"], errors="coerce")
    y = pd.to_numeric(data["Final_P_kbar"], errors="coerce")
    valid = x.notna() & y.notna()
    x = x.loc[valid].to_numpy()
    y = y.loc[valid].to_numpy()

    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return

    values = np.vstack([x, y])
    try:
        kde = gaussian_kde(values)
    except np.linalg.LinAlgError:
        return

    # Extend the grid far enough for outer probability contours to close.
    x_padding = max(np.ptp(x) * KDE_GRID_PADDING, 1.0)
    y_padding = max(np.ptp(y) * KDE_GRID_PADDING, 0.05)
    x_grid = np.linspace(x.min() - x_padding, x.max() + x_padding, KDE_GRID_SIZE)
    y_grid = np.linspace(y.min() - y_padding, y.max() + y_padding, KDE_GRID_SIZE)
    xx, yy = np.meshgrid(x_grid, y_grid)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    sorted_density = np.sort(density.ravel())[::-1]
    cumulative_density = np.cumsum(sorted_density)
    cumulative_density /= cumulative_density[-1]
    for mass_fraction, linestyle, alpha in KDE_CONTOUR_LEVELS:
        if linestyle == "dashed":
            linestyle = dashed_linestyle
        level_index = np.searchsorted(cumulative_density, mass_fraction)
        contour_level = sorted_density[min(level_index, len(sorted_density) - 1)]
        ax.contour(
            xx,
            yy,
            density,
            levels=[contour_level],
            colors=[color],
            alpha=alpha,
            linewidths=linewidth,
            linestyles=[linestyle],
            zorder=0,
        )


def main():
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(9, 4),
        dpi=FIGURE_DPI,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.15},
    )
    composition = load_cpx_composition()
    color_norm = Normalize(
        vmin=composition[AL2O3_COLUMN].min(),
        vmax=composition[AL2O3_COLUMN].max(),
    )
    scatter = None

    for ax, panel_label, (title, data_path) in zip(
        axes,
        ("(a)", "(b)"),
        PANEL_DATA.items(),
    ):
        df = pd.read_csv(data_path).merge(
            composition,
            on=["Year", "Report_row_index"],
            how="left",
            validate="one_to_one",
        )
        if df[AL2O3_COLUMN].isna().any():
            raise ValueError(f"Missing {AL2O3_COLUMN} values after composition merge")

        for year, style in YEAR_STYLES.items():
            year_data = df.loc[df["Year"] == year]
            scatter = ax.scatter(
                year_data["Final_T_C"],
                year_data["Final_P_kbar"],
                c=year_data[AL2O3_COLUMN],
                cmap=AL2O3_CMAP,
                norm=color_norm,
                s=YEAR_MARKER_SIZES[year],
                linewidths=MARKER_EDGE_WIDTH,
                label=f"{year} eruption",
                zorder=YEAR_ZORDERS[year],
                **style,
            )

        ax.set_xlabel("Temperature (°C)", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel("Pressure (kbar)", fontsize=AXIS_LABEL_SIZE)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_title(title)
        add_panel_label_yaxis_aligned(
            ax,
            panel_label,
            x=-0.18,
            y=1.12,
            fontsize=ax.title.get_fontsize(),
            bbox=False,
        )
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)

    axes[0].legend(
        loc="lower right",
        borderpad=0.3,
        labelspacing=0.25,
        handletextpad=0.35,
        borderaxespad=0.35,
    )
    colorbar = fig.colorbar(scatter, ax=axes, shrink=0.8, pad=COLORBAR_PAD)
    colorbar.ax.set_title(
        r"$\mathrm{Al_2O_3}$" "\n" "(wt%)",
        loc="center",
    )
    save_figure_pdf_png(
        fig,
        IMAGES_DIR,
        "fig_S18_P-T_al2o3",
        dpi=FIGURE_DPI,
    )
    plt.show()


def main_with_kde_contours():
    fig, axes = plt.subplots(
        ncols=2,
        figsize=(9, 4),
        dpi=FIGURE_DPI,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.15},
    )
    composition = load_cpx_composition()
    color_norm = Normalize(
        vmin=composition[AL2O3_COLUMN].min(),
        vmax=composition[AL2O3_COLUMN].max(),
    )
    scatter = None

    for ax, panel_label, (title, data_path) in zip(
        axes,
        ("(a)", "(b)"),
        PANEL_DATA.items(),
    ):
        df = pd.read_csv(data_path).merge(
            composition,
            on=["Year", "Report_row_index"],
            how="left",
            validate="one_to_one",
        )
        if df[AL2O3_COLUMN].isna().any():
            raise ValueError(f"Missing {AL2O3_COLUMN} values after composition merge")

        year_data_by_year = {}
        for year, style in KDE_YEAR_STYLES.items():
            year_data = df.loc[df["Year"] == year]
            year_data_by_year[year] = year_data
            scatter = ax.scatter(
                year_data["Final_T_C"],
                year_data["Final_P_kbar"],
                c=year_data[AL2O3_COLUMN],
                cmap=AL2O3_CMAP,
                norm=color_norm,
                s=YEAR_MARKER_SIZES[year],
                linewidths=KDE_MARKER_EDGE_WIDTH,
                label=f"{year} eruption",
                zorder=YEAR_ZORDERS[year],
                **style,
            )

        # Keep the visible limits controlled by the data points, not the KDE grid.
        scatter_xlim = ax.get_xlim()
        scatter_ylim = ax.get_ylim()
        for year, year_data in year_data_by_year.items():
            add_kde_contours(
                ax,
                year_data,
                KDE_CONTOUR_COLORS[year],
                KDE_CONTOUR_LINEWIDTHS[year],
                KDE_DASHED_LINESTYLES[year],
            )
        ax.set_xlim(scatter_xlim)
        ax.set_ylim(scatter_ylim)

        ax.set_xlabel("Temperature (°C)", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel("Pressure (kbar)", fontsize=AXIS_LABEL_SIZE)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_title(title)
        add_panel_label_yaxis_aligned(
            ax,
            panel_label,
            x=-0.18,
            y=1.12,
            fontsize=ax.title.get_fontsize(),
            bbox=False,
        )
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)

    axes[0].legend(
        loc="lower right",
        borderpad=0.3,
        labelspacing=0.25,
        handletextpad=0.35,
        borderaxespad=0.35,
    )
    colorbar = fig.colorbar(scatter, ax=axes, shrink=0.8, pad=COLORBAR_PAD)
    colorbar.ax.set_title(
        r"$\mathrm{Al_2O_3}$" "\n" "(wt%)",
        loc="center",
    )
    save_figure_pdf_png(
        fig,
        IMAGES_DIR,
        "fig_S18_P-T_al2o3_kde",
        dpi=FIGURE_DPI,
    )
    plt.show()


if __name__ == "__main__":
    # main()
    main_with_kde_contours()
