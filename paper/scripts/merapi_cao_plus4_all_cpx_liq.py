"""Recalculate and plot Merapi cpx-liquid responses to CaO_liq +4 wt.%.

The calculation uses the accepted Kd = 0.28 cpx-liquid pairs from the Merapi
workflow, preserves the original liquids as the baseline, and renormalizes the
nine anhydrous oxides to 100 wt.% after adding 4 wt.% CaO.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass


DELTA_CAO_INPUT_WT = 4.0
SEED = 42
OXIDE_COLS = [
    "SiO2",
    "TiO2",
    "Al2O3",
    "FeO",
    "MnO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
]
CPX_COLS = [
    "SiO2",
    "TiO2",
    "Al2O3",
    "FeO",
    "MnO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "Cr2O3",
    "NiO",
]


@dataclass(frozen=True)
class ModelSpec:
    """Describe one plotted model and its lookup alias."""

    kind: str
    key: str
    label: str
    alias: str


# Preserve the cpx-liquid order shown in the reference Merapi P-T figure.
PRESSURE_SPECS = (
    ModelSpec("P", "Pet20", "Pet20", "Petrelli"),
    ModelSpec("P", "Jor22", "Jor22", "Jorgenson"),
    ModelSpec("P", "Chi23", "Chi23", "Chicchi"),
    ModelSpec("P", "AgL24", "AgL24", "Ágreda-López"),
    ModelSpec("P", "Pu08_31", "Pu08\n_31", "Putirka, 2008 eq33_T; eq31_P"),
    ModelSpec("P", "NP17", "NP17", "Neave & Putirka"),
)
TEMPERATURE_SPECS = (
    ModelSpec("T", "AgL24", "AgL24", "Ágreda-López"),
    ModelSpec("T", "Chi23", "Chi23", "Chicchi"),
    ModelSpec("T", "Jor22", "Jor22", "Jorgenson"),
    ModelSpec("T", "Pu08_33", "Pu08\n_33", "Putirka, 2008 eq33_T; eq31_P"),
    ModelSpec("T", "Pet20", "Pet20", "Petrelli"),
)
YEAR_COLORS = {2006: "blue", 2010: "red"}

# Match the eruption-specific cpx-liquid models selected in Figure 7.
SELECTED_MODELS = {
    ("P", 2006): frozenset({"Pet20", "Jor22", "Chi23"}),
    ("P", 2010): frozenset({"Pet20", "Jor22", "Chi23", "AgL24"}),
    ("T", 2006): frozenset({"AgL24", "Chi23"}),
    ("T", 2010): frozenset({"AgL24", "Chi23", "Jor22"}),
}


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by walking upward to pyproject.toml."""
    path = Path.cwd() if start is None else Path(start).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find pyproject.toml above the current directory.")


def parse_boolean_mask(values: pd.Series) -> pd.Series:
    """Parse cached Boolean values robustly."""
    if values.dtype == object:
        return values.astype(str).str.lower().isin(["true", "1", "yes"])
    return values.astype(bool)


def load_accepted_pairs(
    year: int,
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and align accepted cpx-liquid pairs from one cache workbook."""
    cpx_all = pd.read_excel(path, sheet_name="cleaned_minerals")
    liq_all = pd.read_excel(path, sheet_name="cleaned_liquids")
    mask = parse_boolean_mask(
        pd.read_excel(path, sheet_name="pairs_pass_mask")["pairs_pass_mask"]
    )
    source_rows = np.flatnonzero(mask.to_numpy())
    cpx = cpx_all.loc[mask.to_numpy(), CPX_COLS].reset_index(drop=True)
    liq = liq_all.loc[mask.to_numpy(), OXIDE_COLS].reset_index(drop=True)
    meta = pd.DataFrame(
        {
            "eruption": year,
            "source_pair_row": source_rows,
            "pair_id": [f"{year}_{row:03d}" for row in source_rows],
        }
    )
    return cpx, liq, meta


def add_water_relation(liq: pd.DataFrame) -> pd.DataFrame:
    """Apply the water relation used by the reference Merapi workflow."""
    output = liq.copy()
    output["H2O"] = output["SiO2"] * 0.06995 + 0.383
    return output


def build_plus4_liquid(baseline_liq: pd.DataFrame) -> pd.DataFrame:
    """Add 4 wt.% CaO and renormalize the anhydrous oxides to 100 wt.%."""
    adjusted = baseline_liq[OXIDE_COLS].copy()
    adjusted["CaO"] = adjusted["CaO"] + DELTA_CAO_INPUT_WT
    adjusted = adjusted.div(adjusted.sum(axis=1), axis=0) * 100.0
    return add_water_relation(adjusted)


def initialize_model_pools() -> dict[str, list]:
    """Initialize every cpx-liquid pressure and temperature model."""
    from aims4pt.model_tools.model_registry import (
        ALL_MODELS_MODULES,
        get_models_initial_pools,
    )

    for module_name in ALL_MODELS_MODULES:
        importlib.import_module(module_name)
    return {
        "P": get_models_initial_pools("P", "cpx_liq", False),
        "T": get_models_initial_pools("T", "cpx_liq", False),
    }


def find_model(model_pool: list, spec: ModelSpec):
    """Return the unique model matching one figure specification."""
    matches = [model for model in model_pool if spec.alias.casefold() in model.model_name.casefold()]
    if len(matches) != 1:
        names = [model.model_name for model in model_pool]
        raise RuntimeError(
            f"Expected one {spec.kind} model for alias {spec.alias!r}; "
            f"found {len(matches)} in {names}."
        )
    return matches[0]


def predict_common_random_numbers(
    model,
    cpx: pd.DataFrame,
    baseline_liq: pd.DataFrame,
    adjusted_liq: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict baseline and adjusted liquids with a shared random seed."""
    np.random.seed(SEED)
    baseline = np.asarray(model.predict(cpx, baseline_liq), dtype=float).reshape(-1)
    np.random.seed(SEED)
    adjusted = np.asarray(model.predict(cpx, adjusted_liq), dtype=float).reshape(-1)
    if baseline.size != len(cpx) or adjusted.size != len(cpx):
        raise RuntimeError(
            f"Unexpected prediction length for {model.model_name}: "
            f"baseline={baseline.size}, adjusted={adjusted.size}, expected={len(cpx)}."
        )
    return baseline, adjusted


def calculate_all_models(
    cpx: pd.DataFrame,
    baseline_liq: pd.DataFrame,
    adjusted_liq: pd.DataFrame,
    meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate all ordered cpx-liquid models and return long-form results."""
    pools = initialize_model_pools()
    result_blocks = []
    model_rows = []
    for specs in (PRESSURE_SPECS, TEMPERATURE_SPECS):
        for model_order, spec in enumerate(specs):
            model = find_model(pools[spec.kind], spec)
            print(f"Calculating {spec.kind} {spec.key}: {model.model_name}", flush=True)
            baseline, adjusted = predict_common_random_numbers(
                model,
                cpx,
                baseline_liq,
                adjusted_liq,
            )
            block = meta.copy()
            block["output_kind"] = spec.kind
            block["model_order"] = model_order
            block["model_key"] = spec.key
            block["model_label"] = spec.label.replace("\n", "_")
            block["model_name"] = model.model_name
            block["baseline_prediction"] = baseline
            block["plus4_prediction"] = adjusted
            block["delta"] = adjusted - baseline
            result_blocks.append(block)
            model_rows.append(
                {
                    "output_kind": spec.kind,
                    "model_order": model_order,
                    "model_key": spec.key,
                    "model_name": model.model_name,
                }
            )
    results = pd.concat(result_blocks, ignore_index=True)
    model_manifest = pd.DataFrame(model_rows)
    return results, model_manifest


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize each eruption-model delta distribution."""
    return (
        results.groupby(
            ["output_kind", "model_order", "model_key", "model_name", "eruption"],
            sort=False,
        )["delta"]
        .agg(
            n="count",
            mean="mean",
            sd="std",
            minimum="min",
            q1=lambda values: values.quantile(0.25),
            median="median",
            q3=lambda values: values.quantile(0.75),
            maximum="max",
            max_abs=lambda values: values.abs().max(),
        )
        .reset_index()
    )


def configure_matplotlib() -> None:
    """Apply the manuscript styling used by the Merapi P-T summary figure."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 17,
            "axes.titlesize": 23,
            "axes.labelsize": 24,
            "xtick.labelsize": 22,
            "ytick.labelsize": 17,
            "axes.linewidth": 1.0,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "legend.frameon": True,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def blend_color(color: str, target: str, amount: float) -> tuple[float, float, float]:
    """Blend a Matplotlib color toward a target color."""
    source_rgb = np.array(mpl.colors.to_rgb(color), dtype=float)
    target_rgb = np.array(mpl.colors.to_rgb(target), dtype=float)
    return tuple(source_rgb * (1.0 - amount) + target_rgb * amount)


def draw_violin(
    ax: mpl.axes.Axes,
    values: np.ndarray,
    position: float,
    color: str,
    selected: bool = False,
) -> None:
    """Draw a Merapi-style KDE violin with a black median bar."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return

    spread = float(np.ptp(clean))
    if clean.size >= 2 and spread > 1e-12:
        standard_deviation = float(np.std(clean, ddof=1))
        bandwidth = max(
            1.06 * standard_deviation * clean.size ** (-0.2),
            spread / 300.0,
        )
        density_grid = np.linspace(
            float(np.min(clean)) - 0.08 * spread,
            float(np.max(clean)) + 0.08 * spread,
            240,
        )
        standardized = (density_grid[:, None] - clean[None, :]) / bandwidth
        density = np.exp(-0.5 * standardized**2).mean(axis=1)
        density /= bandwidth * np.sqrt(2.0 * np.pi)
        half_width = 0.14 * density / density.max()
        ax.fill_betweenx(
            density_grid,
            position - half_width,
            position + half_width,
            facecolor=color,
            edgecolor="none" if selected else "black",
            linewidth=0.0 if selected else 1.2,
            alpha=1.0,
            zorder=3,
        )
        if selected:
            outline_x = np.concatenate(
                (
                    position - half_width,
                    (position + half_width)[::-1],
                    [position - half_width[0]],
                )
            )
            outline_y = np.concatenate(
                (density_grid, density_grid[::-1], [density_grid[0]])
            )
            ax.plot(
                outline_x,
                outline_y,
                color="black",
                linewidth=2.2,
                linestyle="--",
                zorder=3.2,
            )
    else:
        ax.plot(
            [position - 0.10, position + 0.10],
            [clean[0], clean[0]],
            color="black",
            linewidth=4.0,
            solid_capstyle="round",
            alpha=1.0,
            zorder=3,
        )

    median = float(np.nanmedian(clean))
    median_half_width = 0.24 * 0.28
    median_x = [position - median_half_width, position + median_half_width]
    ax.plot(
        median_x,
        [median, median],
        color="white",
        linewidth=2.8,
        alpha=0.85,
        zorder=4.8,
    )
    ax.plot(
        median_x,
        [median, median],
        color="black",
        linewidth=1.8,
        zorder=5,
    )


def plot_results(results: pd.DataFrame) -> tuple[mpl.figure.Figure, np.ndarray]:
    """Plot pressure and temperature deltas in the reference model order."""
    configure_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 11.0), constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.08)
    years = (2006, 2010)
    offsets = {2006: -0.14, 2010: 0.14}
    year_counts = results.groupby("eruption")["pair_id"].nunique().to_dict()
    year_facecolors = {
        year: blend_color(color, "white", 0.34)
        for year, color in YEAR_COLORS.items()
    }

    for panel_index, (ax, kind, specs, ylabel) in enumerate(
        (
            (axes[0], "P", PRESSURE_SPECS, r"$\Delta P$ (kbar)"),
            (axes[1], "T", TEMPERATURE_SPECS, r"$\Delta T$ (°C)"),
        )
    ):
        positions = np.arange(1, len(specs) + 1, dtype=float)
        for model_position, spec in zip(positions, specs):
            for year in years:
                mask = (
                    results["output_kind"].eq(kind)
                    & results["model_key"].eq(spec.key)
                    & results["eruption"].eq(year)
                )
                draw_violin(
                    ax,
                    results.loc[mask, "delta"].to_numpy(dtype=float),
                    model_position + offsets[year],
                    year_facecolors[year],
                    selected=spec.key in SELECTED_MODELS[(kind, year)],
                )

        ax.plot(
            [0.5, len(specs) + 0.5],
            [0.0, 0.0],
            color="0.35",
            linestyle="--",
            linewidth=1.3,
            zorder=1,
        )
        ax.set_ylabel(ylabel, fontsize=24)
        ax.set_xticks(positions, [spec.label for spec in specs])
        ax.tick_params(axis="x", which="major", labelsize=22)
        ax.tick_params(axis="y", which="major", length=6, width=1.0, labelsize=17)
        ax.minorticks_on()
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        ax.tick_params(axis="y", which="minor", length=3, width=0.8)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(0.5, len(specs) + 0.5)
        ax.text(
            -0.085,
            1.045,
            f"({chr(ord('a') + panel_index)})",
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="bottom",
        )

    axes[0].legend(
        handles=[
            Patch(
                facecolor=YEAR_COLORS[2006],
                edgecolor="black",
                alpha=1.0,
                label=f"2006 eruption (n = {year_counts[2006]})",
            ),
            Patch(
                facecolor=YEAR_COLORS[2010],
                edgecolor="black",
                alpha=1.0,
                label=f"2010 eruption (n = {year_counts[2010]})",
            ),
            Patch(
                facecolor="white",
                edgecolor="black",
                linestyle="--",
                linewidth=2.2,
                label="Selected model",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=15,
        borderpad=0.35,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    return fig, axes


def save_figure_bundle(fig: mpl.figure.Figure, output_stem: Path) -> dict[str, Path]:
    """Save editable vector and high-resolution raster figure formats."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "png": output_stem.with_suffix(".png"),
        "pdf": output_stem.with_suffix(".pdf"),
        "svg": output_stem.with_suffix(".svg"),
        "tiff": output_stem.with_suffix(".tiff"),
    }
    fig.savefig(output_paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_paths["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(output_paths["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(
        output_paths["tiff"],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    return output_paths


def main() -> None:
    """Run the full +4 wt.% calculation, export source data, and draw the figure."""
    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    cache_dir = project_root / "paper" / ".cache" / "add_pre-2006_028"
    output_dir = project_root / "outputs" / "merapi_cao_plus4_all_cpx_liq"
    output_dir.mkdir(parents=True, exist_ok=True)
    pairing_files = {
        2006: cache_dir / "merapi_2006_pairing_kd028_err1e-4_n20000_endmembers.xlsx",
        2010: cache_dir / "merapi_2010_pairing_kd028_err1e-4_n20000_endmembers.xlsx",
    }

    cpx_blocks = []
    liq_blocks = []
    meta_blocks = []
    for year, pairing_path in pairing_files.items():
        cpx_block, liq_block, meta_block = load_accepted_pairs(year, pairing_path)
        cpx_blocks.append(cpx_block)
        liq_blocks.append(liq_block)
        meta_blocks.append(meta_block)
    cpx = pd.concat(cpx_blocks, ignore_index=True)
    baseline_liq_anhydrous = pd.concat(liq_blocks, ignore_index=True)
    meta = pd.concat(meta_blocks, ignore_index=True)
    baseline_liq = add_water_relation(baseline_liq_anhydrous)
    adjusted_liq = build_plus4_liquid(baseline_liq_anhydrous)

    results, model_manifest = calculate_all_models(cpx, baseline_liq, adjusted_liq, meta)
    summary = summarize_results(results)
    composition_audit = meta.copy()
    composition_audit["CaO_original_wt%"] = baseline_liq["CaO"]
    composition_audit["CaO_target_pre_norm_wt%"] = baseline_liq["CaO"] + DELTA_CAO_INPUT_WT
    composition_audit["CaO_plus4_normalized_wt%"] = adjusted_liq["CaO"]
    composition_audit["actual_delta_CaO_normalized_wt%"] = (
        adjusted_liq["CaO"] - baseline_liq["CaO"]
    )
    composition_audit["normalized_oxide_sum_wt%"] = adjusted_liq[OXIDE_COLS].sum(axis=1)

    finite_delta = np.isfinite(results["delta"].to_numpy(dtype=float))
    qa = {
        "accepted_pairs": int(len(meta)),
        "accepted_pairs_2006": int(meta["eruption"].eq(2006).sum()),
        "accepted_pairs_2010": int(meta["eruption"].eq(2010).sum()),
        "pressure_models": len(PRESSURE_SPECS),
        "temperature_models": len(TEMPERATURE_SPECS),
        "result_rows": int(len(results)),
        "finite_delta_rows": int(finite_delta.sum()),
        "max_normalized_oxide_sum_error_wt%": float(
            (composition_audit["normalized_oxide_sum_wt%"] - 100.0).abs().max()
        ),
    }
    assert qa["accepted_pairs"] == 148
    assert qa["accepted_pairs_2006"] == 42
    assert qa["accepted_pairs_2010"] == 106
    assert qa["result_rows"] == 148 * (len(PRESSURE_SPECS) + len(TEMPERATURE_SPECS))
    assert qa["finite_delta_rows"] == qa["result_rows"]
    assert qa["max_normalized_oxide_sum_error_wt%"] < 1e-10

    results.to_csv(output_dir / "merapi_cao_plus4_all_cpx_liq_results.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "merapi_cao_plus4_all_cpx_liq_summary.csv", index=False, encoding="utf-8-sig")
    composition_audit.to_csv(output_dir / "merapi_cao_plus4_composition_audit.csv", index=False, encoding="utf-8-sig")
    model_manifest.to_csv(output_dir / "merapi_cao_plus4_model_manifest.csv", index=False, encoding="utf-8-sig")
    (output_dir / "merapi_cao_plus4_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fig, _ = plot_results(results)
    figure_paths = save_figure_bundle(
        fig,
        output_dir / "merapi_cao_plus4_all_cpx_liq_violins",
    )
    plt.close(fig)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(qa, indent=2), flush=True)
    for file_type, output_path in figure_paths.items():
        print(f"{file_type}: {output_path}", flush=True)


if __name__ == "__main__":
    main()
