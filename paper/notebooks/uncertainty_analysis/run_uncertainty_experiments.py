"""Run cpx-liquid test-subset uncertainty analyses for AIMS4PT.

This script keeps all generated files under this directory by default:
paper/notebooks/uncertainty_analysis/results/uncertainty
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    """Find the repository root from the script location."""
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() and (path / "src").exists():
            return path
    raise RuntimeError("Could not locate project root containing pyproject.toml and src/.")


PROJECT_ROOT = find_project_root(BASE_DIR)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
PAPER_SCRIPTS_DIR = PROJECT_ROOT / "paper" / "scripts"
if str(PAPER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_SCRIPTS_DIR))


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from PIL import Image, ImageColor, ImageDraw, ImageFont

from aims4pt.model_tools.model_registry import get_models_initial_pools, import_all_models
from constants_illustration import get_model_abbreviation


TEST_VALUES = {"testing", "test"}
CPX_OXIDES = [
    "SiO2_cpx",
    "TiO2_cpx",
    "Al2O3_cpx",
    "Fe2O3_cpx",
    "Cr2O3_cpx",
    "FeO_cpx",
    "MnO_cpx",
    "MgO_cpx",
    "NiO_cpx",
    "CoO_cpx",
    "CaO_cpx",
    "Na2O_cpx",
    "K2O_cpx",
    "P2O5_cpx",
]
LIQ_OXIDES = [
    "SiO2_liq",
    "TiO2_liq",
    "Al2O3_liq",
    "Fe2O3_liq",
    "Cr2O3_liq",
    "FeO_liq",
    "MnO_liq",
    "MgO_liq",
    "NiO_liq",
    "CoO_liq",
    "CaO_liq",
    "Na2O_liq",
    "K2O_liq",
    "P2O5_liq",
]
CPX_LEVELS = [(0.005, "cpx_0.5pct"), (0.010, "cpx_1pct")]
LIQ_LEVELS = [(0.010, "liq_1pct"), (0.020, "liq_2pct"), (0.030, "liq_3pct")]
KD_FINE_BIN_EDGES = np.array([0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36], dtype=float)


@dataclass(frozen=True)
class ModelSpec:
    target_type: str
    model: object
    name: str
    model_type: str = "cpx_liq"


def normalize_split(value: object) -> str:
    """Normalize split labels to lower-case strings."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def model_code(model_name: str) -> str:
    """Create compact model labels for filenames."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(model_name)).strip("_")
    return safe[:40] or "model"


def sanitize_filename(text: str) -> str:
    """Return a filesystem-safe token."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    """Create output directories."""
    paths = {
        "root": output_dir,
        "kd": output_dir / "kd",
        "kd_fig": output_dir / "kd" / "figures_test_subset",
        "analytical": output_dir / "analytical",
        "analytical_fig": output_dir / "analytical" / "figures_test_subset",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_test_subset(input_path: Path, sheet: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and validate the independent experimental test subset."""
    df = pd.read_excel(input_path, sheet_name=sheet)
    if "training/testing" not in df.columns:
        raise ValueError("Missing required column: training/testing")

    split_norm = df["training/testing"].map(normalize_split)
    test_mask = split_norm.isin(TEST_VALUES)
    train_mask = split_norm.eq("training")
    test_df = df.loc[test_mask].copy()
    test_df["split"] = "test_subset"

    qc = {
        "total_rows": int(len(df)),
        "training_rows": int(train_mask.sum()),
        "testing_rows": int(test_mask.sum()),
        "unique_split_values": sorted(map(str, df["training/testing"].dropna().unique())),
    }
    print(f"Retained test-subset rows: {len(test_df)}")
    if len(test_df) != 59:
        print(
            "WARNING: test-subset row count is not 59. "
            f"total={qc['total_rows']}; training={qc['training_rows']}; "
            f"testing={qc['testing_rows']}; unique={qc['unique_split_values']}"
        )

    required_truth = ["P (kbar)", "T (C)"]
    missing = [col for col in required_truth if col not in test_df.columns]
    if missing:
        raise ValueError(f"Missing required truth columns: {missing}")

    if "id" not in test_df.columns:
        test_df["id"] = np.arange(1, len(test_df) + 1)
    return test_df.reset_index(drop=True), qc


def make_model_specs(method_type: str = "cpx_liq") -> list[ModelSpec]:
    """Detect thermobarometer models from the registry."""
    imported, failed = import_all_models()
    if failed:
        print(f"WARNING: model import failures: {failed}")
    specs: list[ModelSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for target_type in ("P", "T"):
        pool = get_models_initial_pools(target_type, method_type, False)
        for model in pool:
            desired_cpx_only = method_type == "cpx_only"
            if getattr(model, "cpx_only", None) is desired_cpx_only:
                full_name = getattr(model, "model_name", type(model).__name__)
                abbreviation = get_model_abbreviation(full_name, target_type)
                if abbreviation == "BT19":
                    continue
                key = (method_type, target_type, abbreviation)
                if key in seen:
                    continue
                seen.add(key)
                specs.append(ModelSpec(target_type, model, abbreviation, method_type))
    print(f"Detected {method_type} model-target entries: {len(specs)}")
    print(f"Detected {method_type} pressure models: {sum(s.target_type == 'P' for s in specs)}")
    print(f"Detected {method_type} temperature models: {sum(s.target_type == 'T' for s in specs)}")
    return specs


def split_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned cpx and liquid input tables."""
    cpx_cols = [col for col in df.columns if str(col).endswith("_cpx")]
    liq_cols = [col for col in df.columns if str(col).endswith("_liq")]
    cpx = df[cpx_cols].apply(pd.to_numeric, errors="coerce").copy()
    liq = df[liq_cols].apply(pd.to_numeric, errors="coerce").copy()
    cpx["P_kbar"] = pd.to_numeric(df["P (kbar)"], errors="coerce")
    cpx["T_C"] = pd.to_numeric(df["T (C)"], errors="coerce")
    cpx.index = df.index
    liq.index = df.index
    return cpx, liq


def safe_predict(spec: ModelSpec, cpx: pd.DataFrame, liq: pd.DataFrame) -> tuple[pd.Series, dict[object, str]]:
    """Predict with a model and fall back to per-row calls if needed."""
    if len(cpx) == 0:
        return pd.Series(dtype=float), {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = predict_with_optional_cache(spec, cpx.copy(), liq.copy())
        pred = pd.Series(pred, index=cpx.index, dtype="float64")
        errors = {idx: "" for idx in cpx.index}
        return pred, errors
    except Exception as exc:
        batch_error = f"{type(exc).__name__}: {exc}"
        preds = pd.Series(np.nan, index=cpx.index, dtype="float64")
        errors: dict[object, str] = {}
        for idx in cpx.index:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred_one = predict_with_optional_cache(spec, cpx.loc[[idx]].copy(), liq.loc[[idx]].copy())
                preds.loc[idx] = float(np.asarray(pred_one).reshape(-1)[0])
                errors[idx] = ""
            except Exception as row_exc:
                errors[idx] = f"{batch_error} | row {type(row_exc).__name__}: {row_exc}"
        return preds, errors


def predict_with_optional_cache(spec: ModelSpec, cpx: pd.DataFrame, liq: pd.DataFrame):
    """Use local cached helpers for wrappers whose public predict is slow."""
    if "Chicchi" in spec.name and getattr(spec.model, "cpx_only", None) is False:
        from aims4pt.model_tools.data.Chicchi23.script.functions_cpx_liq import easy_predict_cpx_liq

        input_data = spec.model.format_input(X_cpx=cpx, X_liq=liq)
        pred = easy_predict_cpx_liq(input_data, T_P=spec.target_type, dir=spec.model.models_dir)
        return pd.Series(pred, index=cpx.index, name="P_kbar" if spec.target_type == "P" else "T_C")
    return spec.model.predict(cpx, liq)


def compatible_baseline(path: Path, test_ids: set[object]) -> pd.DataFrame | None:
    """Load a compatible baseline table if possible."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    required = {
        "id",
        "split",
        "model",
        "target_type",
        "P_true",
        "T_true",
        "P_pred",
        "T_pred",
        "delta_P",
        "delta_T",
        "status",
        "error_message",
    }
    if not required.issubset(df.columns):
        return None
    if "test_subset" in set(df["split"].astype(str)):
        df = df[df["split"].astype(str).eq("test_subset")].copy()
    if not set(df["id"]).issubset(test_ids):
        return None
    if not set(df["id"]).intersection(test_ids):
        return None
    df["model"] = [
        get_model_abbreviation(model_name, target_type)
        for model_name, target_type in zip(df["model"], df["target_type"])
    ]
    df = df[df["model"].ne("BT19")].copy()
    return df


def normalize_output_model_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert model names to Table 1 abbreviations and drop BT19."""
    out = df.copy()
    if {"model", "target_type"}.issubset(out.columns):
        out["model"] = [
            get_model_abbreviation(model_name, target_type)
            for model_name, target_type in zip(out["model"], out["target_type"])
        ]
        out = out[out["model"].ne("BT19")].copy()
    return out


def find_existing_baseline(output_dir: Path, test_ids: set[object]) -> tuple[pd.DataFrame | None, Path | None]:
    """Search likely baseline locations."""
    candidates = [
        output_dir / "baseline_cpx_liq_predictions_test_subset.csv",
        output_dir / "baseline_cpx_liq_predictions.csv",
        PROJECT_ROOT / "results" / "uncertainty" / "baseline_cpx_liq_predictions_test_subset.csv",
        PROJECT_ROOT / "results" / "uncertainty" / "baseline_cpx_liq_predictions.csv",
    ]
    for root in [PROJECT_ROOT / "output", PROJECT_ROOT / "report_output", PROJECT_ROOT / "paper"]:
        if root.exists():
            candidates.extend(root.rglob("*baseline*cpx*liq*prediction*.csv"))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        loaded = compatible_baseline(path, test_ids)
        if loaded is not None:
            return loaded, path
    return None, None


def compute_baseline(
    test_df: pd.DataFrame,
    specs: list[ModelSpec],
    output_dir: Path,
    reuse_existing: bool,
) -> tuple[pd.DataFrame, Path | None, int, int]:
    """Create or reuse the baseline prediction table."""
    baseline_path = output_dir / "baseline_cpx_liq_predictions_test_subset.csv"
    test_ids = set(test_df["id"])
    if reuse_existing:
        existing, source = find_existing_baseline(output_dir, test_ids)
        if existing is not None:
            existing.to_csv(baseline_path, index=False)
            print(f"Reused baseline predictions from: {source}")
            return existing, source, len(existing), 0

    cpx, liq = split_inputs(test_df)
    rows: list[dict[str, object]] = []
    for spec in specs:
        pred, errors = safe_predict(spec, cpx, liq)
        for idx, sample in test_df.iterrows():
            p_pred = float(pred.loc[idx]) if spec.target_type == "P" and pd.notna(pred.loc[idx]) else np.nan
            t_pred = float(pred.loc[idx]) if spec.target_type == "T" and pd.notna(pred.loc[idx]) else np.nan
            p_true = float(sample["P (kbar)"])
            t_true = float(sample["T (C)"])
            err = errors.get(idx, "")
            status = "ok" if err == "" and pd.notna(pred.loc[idx]) else "calculation_failed"
            if err == "" and pd.isna(pred.loc[idx]):
                err = "prediction returned NaN"
            rows.append(
                {
                    "id": sample["id"],
                    "split": "test_subset",
                    "model": spec.name,
                    "target_type": spec.target_type,
                    "P_true": p_true,
                    "T_true": t_true,
                    "P_pred": p_pred,
                    "T_pred": t_pred,
                    "delta_P": p_pred - p_true if pd.notna(p_pred) else np.nan,
                    "delta_T": t_pred - t_true if pd.notna(t_pred) else np.nan,
                    "status": status,
                    "error_message": err,
                }
            )
    baseline = pd.DataFrame(rows)
    baseline.to_csv(baseline_path, index=False)
    print(f"Saved baseline predictions: {baseline_path}")
    return baseline, None, 0, len(baseline)


def calculate_kd(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Kd(Fe-Mg) for fixed cpx-liquid pairs."""
    out = df[["id", "split"]].copy()
    required = ["FeO_cpx", "MgO_cpx", "FeO_liq", "MgO_liq"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        out["Kd_FeMg"] = np.nan
        out["kd_status"] = f"missing columns: {missing}"
    else:
        values = df[required].apply(pd.to_numeric, errors="coerce")
        valid = values.notna().all(axis=1) & (values[["MgO_cpx", "FeO_liq", "MgO_liq"]] > 0).all(axis=1)
        kd = pd.Series(np.nan, index=df.index, dtype="float64")
        fe_mg_cpx = (values["FeO_cpx"] / 71.844) / (values["MgO_cpx"] / 40.304)
        fe_mg_liq = (values["FeO_liq"] / 71.844) / (values["MgO_liq"] / 40.304)
        kd.loc[valid] = fe_mg_cpx.loc[valid] / fe_mg_liq.loc[valid]
        out["Kd_FeMg"] = kd
        out["kd_status"] = np.where(valid, "ok", "invalid FeO/MgO input")
    out["Kd_centered"] = out["Kd_FeMg"] - 0.28
    bins = [-np.inf, 0.20, 0.24, 0.28, 0.32, 0.36, np.inf]
    labels = [
        "outside_0.20_0.36",
        "0.20_0.24",
        "0.24_0.28",
        "0.28_0.32",
        "0.32_0.36",
        "outside_0.20_0.36",
    ]
    out["Kd_bin"] = pd.cut(out["Kd_FeMg"], bins=bins, labels=labels, ordered=False)
    out["Kd_bin"] = out["Kd_bin"].astype(object).where(out["Kd_FeMg"].notna(), np.nan)
    out["Kd_outside_0.20_0.36"] = ~out["Kd_FeMg"].between(0.20, 0.36, inclusive="both")
    return out


def regression_metrics(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """Fit y = intercept + slope * x and return diagnostics."""
    mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_valid = x.loc[mask].astype(float)
    y_valid = y.loc[mask].astype(float)
    n = int(len(x_valid))
    if n < 10:
        return {
            "n": n,
            "intercept": np.nan,
            "slope_per_1Kd": np.nan,
            "slope_per_0.01Kd": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "status": "skipped_n_lt_10",
        }
    x_arr = x_valid.to_numpy(dtype=float)
    y_arr = y_valid.to_numpy(dtype=float)
    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    dx = x_arr - x_mean
    dy = y_arr - y_mean
    ss_x = float(np.sum(dx**2))
    ss_y = float(np.sum(dy**2))
    if ss_x <= 0 or ss_y <= 0:
        slope = np.nan
        intercept = np.nan
        pearson_r = np.nan
        y_fit = np.full_like(y_arr, np.nan)
    else:
        slope = float(np.sum(dx * dy) / ss_x)
        intercept = float(y_mean - slope * x_mean)
        pearson_r = float(np.sum(dx * dy) / math.sqrt(ss_x * ss_y))
        y_fit = intercept + slope * x_arr
    residual = y_valid - y_fit
    x_rank = x_valid.rank(method="average").to_numpy(dtype=float)
    y_rank = y_valid.rank(method="average").to_numpy(dtype=float)
    rx = x_rank - float(np.mean(x_rank))
    ry = y_rank - float(np.mean(y_rank))
    r_ss_x = float(np.sum(rx**2))
    r_ss_y = float(np.sum(ry**2))
    spearman_rho = float(np.sum(rx * ry) / math.sqrt(r_ss_x * r_ss_y)) if r_ss_x > 0 and r_ss_y > 0 else np.nan
    return {
        "n": n,
        "intercept": float(intercept),
        "slope_per_1Kd": float(slope),
        "slope_per_0.01Kd": float(slope * 0.01),
        "r2": float(pearson_r**2),
        "pearson_r": float(pearson_r),
        "pearson_p": np.nan,
        "spearman_rho": float(spearman_rho),
        "spearman_p": np.nan,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "status": "ok",
    }


def run_kd_analysis(test_df: pd.DataFrame, baseline: pd.DataFrame, paths: dict[str, Path]) -> tuple[pd.DataFrame, int, int]:
    """Run Kd association analyses and figures."""
    kd_values = calculate_kd(test_df)
    kd_path = paths["kd"] / "kd_values_by_sample_test_subset.csv"
    kd_values.to_csv(kd_path, index=False)

    merged = baseline.merge(kd_values[["id", "Kd_FeMg", "Kd_centered", "Kd_bin"]], on="id", how="left")
    regressions: list[dict[str, object]] = []
    for (model, target_type), group in merged.groupby(["model", "target_type"], dropna=False):
        if target_type == "P":
            pairs = [("delta_P", "delta_P"), ("P_pred", "P_pred")]
        else:
            pairs = [("delta_T", "delta_T"), ("T_pred", "T_pred")]
        valid_status = group["status"].eq("ok")
        for response_name, response_col in pairs:
            metrics = regression_metrics(group.loc[valid_status, "Kd_centered"], group.loc[valid_status, response_col])
            regressions.append({"model": model, "target_type": target_type, "response": response_name, **metrics})
    reg_df = pd.DataFrame(regressions)
    reg_df.to_csv(paths["kd"] / "kd_regression_by_model_test_subset.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for (model, target_type, kd_bin), group in merged.groupby(["model", "target_type", "Kd_bin"], dropna=False):
        if pd.isna(kd_bin):
            continue
        pred_col = "P_pred" if target_type == "P" else "T_pred"
        residual_col = "delta_P" if target_type == "P" else "delta_T"
        ok = group[group["status"].eq("ok") & group[residual_col].notna()]
        residual = ok[residual_col].astype(float)
        summary_rows.append(
            {
                "model": model,
                "target_type": target_type,
                "Kd_bin": kd_bin,
                "n": int(len(ok)),
                "mean_Kd_FeMg": ok["Kd_FeMg"].mean(),
                "median_Kd_FeMg": ok["Kd_FeMg"].median(),
                "mean_prediction": ok[pred_col].mean(),
                "median_prediction": ok[pred_col].median(),
                "mean_residual": residual.mean(),
                "median_residual": residual.median(),
                "rmse": float(np.sqrt(np.mean(residual**2))) if len(residual) else np.nan,
                "mae": float(np.mean(np.abs(residual))) if len(residual) else np.nan,
                "residual_p25": residual.quantile(0.25) if len(residual) else np.nan,
                "residual_p75": residual.quantile(0.75) if len(residual) else np.nan,
                "residual_p05": residual.quantile(0.05) if len(residual) else np.nan,
                "residual_p95": residual.quantile(0.95) if len(residual) else np.nan,
            }
        )
    bin_df = pd.DataFrame(summary_rows)
    bin_df.to_csv(paths["kd"] / "kd_bin_summary_by_model_test_subset.csv", index=False)
    fine_iqr_df, fine_count_df = kd_fine_bin_residual_iqr(merged)
    fine_iqr_df.to_csv(paths["kd"] / "kd_bin_residual_iqr_0p02_by_model_test_subset.csv", index=False)
    fine_count_df.to_csv(paths["kd"] / "kd_bin_counts_0p02_test_subset.csv", index=False)

    log_rows = [
        {"metric": "test_subset_rows", "value": len(test_df)},
        {"metric": "valid_kd_values", "value": int(kd_values["Kd_FeMg"].notna().sum())},
        {
            "metric": "kd_outside_0.20_0.36",
            "value": int((kd_values["Kd_FeMg"].notna() & ~kd_values["Kd_FeMg"].between(0.20, 0.36)).sum()),
        },
    ]
    pd.DataFrame(log_rows).to_csv(paths["kd"] / "kd_analysis_log_test_subset.csv", index=False)

    figure_count = plot_kd_figures(merged, reg_df, bin_df, fine_iqr_df, fine_count_df, paths["kd_fig"])
    valid_kd = int(kd_values["Kd_FeMg"].notna().sum())
    outside_kd = int((kd_values["Kd_FeMg"].notna() & ~kd_values["Kd_FeMg"].between(0.20, 0.36)).sum())
    print(f"Valid Kd values: {valid_kd}")
    print(f"Kd values outside 0.20-0.36: {outside_kd}")
    return kd_values, valid_kd, outside_kd


def save_current_fig(path: Path) -> None:
    """Save and close the active figure."""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def kd_fine_bin_residual_iqr(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize residual distributions in 0.02-wide Kd bins."""
    edges = KD_FINE_BIN_EDGES.astype(float).copy()
    cut_edges = edges.copy()
    cut_edges[-1] += 1e-9
    labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(len(edges) - 1)]
    centers = {label: float((edges[i] + edges[i + 1]) / 2) for i, label in enumerate(labels)}

    sample_level = merged[["id", "Kd_FeMg"]].drop_duplicates().copy()
    sample_level = sample_level[sample_level["Kd_FeMg"].between(edges[0], edges[-1], inclusive="both")]
    sample_level["Kd_bin_0p02"] = pd.cut(
        sample_level["Kd_FeMg"], bins=cut_edges, labels=labels, include_lowest=True, right=False
    )
    count_rows = []
    for label in labels:
        part = sample_level[sample_level["Kd_bin_0p02"].astype(str).eq(label)]
        count_rows.append({"Kd_bin_0p02": label, "Kd_bin_center": centers[label], "sample_n": int(part["id"].nunique())})
    count_df = pd.DataFrame(count_rows)

    plot_rows: list[dict[str, object]] = []
    for target_type in ["P", "T"]:
        residual_col = f"delta_{target_type}"
        if residual_col not in merged.columns:
            continue
        data = merged[
            merged["target_type"].eq(target_type)
            & merged["status"].eq("ok")
            & merged["Kd_FeMg"].between(edges[0], edges[-1], inclusive="both")
            & merged[residual_col].notna()
        ].copy()
        if data.empty:
            continue
        data["Kd_bin_0p02"] = pd.cut(data["Kd_FeMg"], bins=cut_edges, labels=labels, include_lowest=True, right=False)
        for (model, kd_bin), group in data.groupby(["model", "Kd_bin_0p02"], observed=False):
            if pd.isna(kd_bin) or group.empty:
                continue
            residual = group[residual_col].astype(float)
            plot_rows.append(
                {
                    "model": model,
                    "target_type": target_type,
                    "Kd_bin_0p02": str(kd_bin),
                    "Kd_bin_center": centers[str(kd_bin)],
                    "n": int(len(residual)),
                    "sample_n": int(group["id"].nunique()),
                    "median_residual": float(residual.median()),
                    "residual_q1": float(residual.quantile(0.25)),
                    "residual_q3": float(residual.quantile(0.75)),
                    "mean_residual": float(residual.mean()),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                }
            )
    return pd.DataFrame(plot_rows), count_df


def plot_kd_residual_iqr_composite(fine_iqr_df: pd.DataFrame, count_df: pd.DataFrame, fig_dir: Path) -> int:
    """Plot pressure and temperature Kd-bin residual IQR summaries in one PIL-rendered figure."""
    if fine_iqr_df.empty:
        return 0
    tokens = {
        "surface": "#FCFCFD",
        "panel": "#FFFFFF",
        "ink": "#1F2430",
        "muted": "#6F768A",
        "grid": "#E6E8F0",
        "axis": "#D7DBE7",
    }
    palette = {
        "AgL24": "#5477C4",
        "Chi23": "#CC6F47",
        "Jor22": "#71B436",
        "NP17": "#BD569B",
        "Pet20": "#736422",
        "Pu08_31": "#7A828F",
        "Pu08_33": "#7A828F",
    }
    edges = KD_FINE_BIN_EDGES.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2
    count_map = {round(float(row["Kd_bin_center"]), 2): int(row["sample_n"]) for _, row in count_df.iterrows()}
    n_by_center = [count_map.get(round(float(center), 2), 0) for center in centers]
    model_order = ["AgL24", "Chi23", "Jor22", "NP17", "Pet20", "Pu08_31", "Pu08_33"]

    width = 3300
    height = 2100
    left = 270
    plot_w = 2260
    legend_gap = 80
    plot_h = 610
    panel_top = [245, 1115]
    axis_label_gap = 125
    image = Image.new("RGB", (width, height), ImageColor.getrgb(tokens["surface"]))
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": _pil_font(58, bold=True),
        "subtitle": _pil_font(31),
        "panel": _pil_font(42, bold=True),
        "letter": _pil_font(44, bold=True),
        "axis": _pil_font(34),
        "tick": _pil_font(29),
        "count": _pil_font(25),
        "legend": _pil_font(31),
        "legend_title": _pil_font(32, bold=True),
        "note": _pil_font(25),
    }
    ink = ImageColor.getrgb(tokens["ink"])
    muted = ImageColor.getrgb(tokens["muted"])
    grid = ImageColor.getrgb(tokens["grid"])
    axis = ImageColor.getrgb(tokens["axis"])
    panel_bg = ImageColor.getrgb(tokens["panel"])

    def x_to_px(value: float) -> int:
        return int(round(left + (value - 0.20) / (0.36 - 0.20) * plot_w))

    def target_y_ticks(vmin: float, vmax: float) -> list[float]:
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return [-1.0, -0.5, 0.0, 0.5, 1.0]
        raw = (vmax - vmin) / 4
        exponent = math.floor(math.log10(raw)) if raw > 0 else 0
        scale = 10**exponent
        step = min([1, 2, 2.5, 5, 10], key=lambda value: abs(value * scale - raw)) * scale
        start = math.floor(vmin / step) * step
        stop = math.ceil(vmax / step) * step
        ticks = []
        value = start
        while value <= stop + step * 0.5:
            ticks.append(round(value, 6))
            value += step
        return ticks

    def draw_dashed_vertical(x: int, y0: int, y1: int, *, color: tuple[int, int, int], dash: int = 15, gap: int = 11) -> None:
        y = y0
        while y < y1:
            draw.line([(x, y), (x, min(y + dash, y1))], fill=color, width=3)
            y += dash + gap

    def draw_dashed_horizontal(x0: int, x1: int, y: int, *, color: tuple[int, int, int], dash: int = 18, gap: int = 12) -> None:
        x = x0
        while x < x1:
            draw.line([(x, y), (min(x + dash, x1), y)], fill=color, width=3)
            x += dash + gap

    def draw_open_circle(x: int, y: int, radius: int, color: tuple[int, int, int]) -> None:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=panel_bg, outline=color, width=5)

    title = "Kd-bin residual distributions"
    subtitle = (
        "Bins span Kd = 0.20-0.36 at 0.02 width; points show median residual and vertical bars show Q1-Q3. "
        "Grey n labels are unique sample counts per bin."
    )
    draw.text((left, 40), title, font=fonts["title"], fill=ink)
    draw.text((left, 115), subtitle, font=fonts["subtitle"], fill=muted)

    target_specs = [
        ("P", "a", "Pressure residual by Kd bin", "Residual, predicted - true (kbar)"),
        ("T", "b", "Temperature residual by Kd bin", "Residual, predicted - true (deg C)"),
    ]
    for top, (target_type, letter, panel_title, ylabel) in zip(panel_top, target_specs):
        data = fine_iqr_df[fine_iqr_df["target_type"].eq(target_type)].copy()
        bottom = top + plot_h
        right = left + plot_w
        draw.rectangle((left, top, right, bottom), fill=panel_bg)
        finite = data[["residual_q1", "median_residual", "residual_q3"]].to_numpy(dtype=float).ravel() if not data.empty else np.array([])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            y_min = min(float(np.min(finite)), 0.0)
            y_max = max(float(np.max(finite)), 0.0)
        else:
            y_min, y_max = -1.0, 1.0
        pad = max((y_max - y_min) * 0.12, 0.5 if target_type == "P" else 8.0)
        y_min -= pad
        y_max += pad
        y_ticks = target_y_ticks(y_min, y_max)
        y_min = min(y_min, min(y_ticks))
        y_max = max(y_max, max(y_ticks))

        def y_to_px(value: float) -> int:
            return int(round(bottom - (value - y_min) / (y_max - y_min) * plot_h))

        draw.text((left - 105, top - 55), letter, font=fonts["letter"], fill=ink)
        draw.text((left, top - 62), panel_title, font=fonts["panel"], fill=ink)
        ylabel_w, _ = _text_size(draw, ylabel, fonts["axis"])
        _draw_rotated_text(image, ylabel, (left - 215, top + plot_h // 2 - (ylabel_w + 8) // 2), fonts["axis"], ink, angle=90)

        for tick in y_ticks:
            y = y_to_px(float(tick))
            draw.line((left, y, right, y), fill=grid, width=2)
            label = f"{tick:.0f}" if abs(tick) >= 10 or abs(tick - round(tick)) < 1e-6 else f"{tick:.1f}"
            tw, th = _text_size(draw, label, fonts["tick"])
            draw.text((left - tw - 24, y - th // 2), label, font=fonts["tick"], fill=muted)
        for tick in edges:
            x = x_to_px(float(tick))
            draw.line((x, top, x, bottom), fill=grid, width=2)
            label = f"{tick:.2f}"
            tw, th = _text_size(draw, label, fonts["tick"])
            draw.text((x - tw // 2, bottom + 20), label, font=fonts["tick"], fill=muted)
        for center, n in zip(centers, n_by_center):
            x = x_to_px(float(center))
            label = f"n={n}"
            tw, _ = _text_size(draw, label, fonts["count"])
            draw.text((x - tw // 2, bottom + 60), label, font=fonts["count"], fill=muted)
        axis_label = "Kd(Fe-Mg)"
        tw, _ = _text_size(draw, axis_label, fonts["axis"])
        draw.text((left + (plot_w - tw) // 2, bottom + axis_label_gap), axis_label, font=fonts["axis"], fill=ink)

        draw.line((left, bottom, right, bottom), fill=axis, width=4)
        draw.line((left, top, left, bottom), fill=axis, width=4)
        zero_y = y_to_px(0.0)
        draw_dashed_horizontal(left, right, zero_y, color=ink)
        kd_ref_x = x_to_px(0.28)
        draw_dashed_vertical(kd_ref_x, top, bottom, color=muted, dash=14, gap=13)
        ref_label = "Kd = 0.28"
        tw, _ = _text_size(draw, ref_label, fonts["count"])
        draw.text((kd_ref_x - tw // 2, top + 12), ref_label, font=fonts["count"], fill=muted)

        if not data.empty:
            present = [model for model in model_order if model in set(data["model"])]
            models = present + [model for model in data["model"].drop_duplicates() if model not in present]
            offsets = np.linspace(-0.006, 0.006, len(models)) if len(models) > 1 else np.array([0.0])
            for model, offset in zip(models, offsets):
                part = data[data["model"].eq(model)].sort_values("Kd_bin_center")
                if part.empty:
                    continue
                color = ImageColor.getrgb(palette.get(str(model), "#5477C4"))
                x_values = part["Kd_bin_center"].astype(float).to_numpy() + offset
                median = part["median_residual"].astype(float).to_numpy()
                q1 = part["residual_q1"].astype(float).to_numpy()
                q3 = part["residual_q3"].astype(float).to_numpy()
                points = [(x_to_px(float(x_value)), y_to_px(float(value))) for x_value, value in zip(x_values, median)]
                if len(points) > 1:
                    draw.line(points, fill=color, width=4, joint="curve")
                for x_value, med, low, high in zip(x_values, median, q1, q3):
                    x = x_to_px(float(x_value))
                    y_med = y_to_px(float(med))
                    y_low = y_to_px(float(low))
                    y_high = y_to_px(float(high))
                    draw.line((x, y_high, x, y_low), fill=color, width=6)
                    draw.line((x - 12, y_med, x + 12, y_med), fill=color, width=6)
                    draw_open_circle(x, y_med, 11, color)

            legend_x = right + legend_gap
            legend_y = top + 24
            draw.text((legend_x, legend_y), "Model", font=fonts["legend_title"], fill=ink)
            for i, model in enumerate(models):
                y = legend_y + 56 + i * 46
                color = ImageColor.getrgb(palette.get(str(model), "#5477C4"))
                draw.line((legend_x, y + 15, legend_x + 60, y + 15), fill=color, width=5)
                draw_open_circle(legend_x + 30, y + 15, 10, color)
                draw.text((legend_x + 82, y - 4), str(model), font=fonts["legend"], fill=ink)

    out_path = fig_dir / "test_subset_kd_bin_residual_iqr_composite.png"
    image.save(out_path, dpi=(600, 600))
    return 1


def plot_kd_figures(
    merged: pd.DataFrame,
    reg_df: pd.DataFrame,
    bin_df: pd.DataFrame,
    fine_iqr_df: pd.DataFrame,
    fine_count_df: pd.DataFrame,
    fig_dir: Path,
) -> int:
    """Generate Kd figures."""
    for old_png in fig_dir.glob("*.png"):
        old_png.unlink()
    count = 0
    sns.set_theme(style="whitegrid", context="paper")
    count += plot_kd_residual_iqr_composite(fine_iqr_df, fine_count_df, fig_dir)
    return count
    for (model, target_type), group in merged.groupby(["model", "target_type"]):
        residual_col = "delta_P" if target_type == "P" else "delta_T"
        ylabel = "P residual (kbar)" if target_type == "P" else "T residual (deg C)"
        ok = group[group["status"].eq("ok") & group["Kd_FeMg"].notna() & group[residual_col].notna()]
        if len(ok) < 2:
            continue
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        sns.scatterplot(data=ok, x="Kd_FeMg", y=residual_col, ax=ax, s=34, color="#2b6f8a", edgecolor="white")
        if len(ok) >= 10:
            sns.regplot(data=ok, x="Kd_FeMg", y=residual_col, ax=ax, scatter=False, color="#b23a48", ci=95)
        ax.axhline(0, color="0.25", lw=1, ls="--")
        ax.axvline(0.28, color="0.35", lw=1, ls=":")
        metrics = reg_df[
            (reg_df["model"].eq(model)) & (reg_df["target_type"].eq(target_type)) & (reg_df["response"].eq(residual_col))
        ]
        if not metrics.empty:
            row = metrics.iloc[0]
            text = (
                f"n={int(row['n'])}\n"
                f"slope/0.01Kd={row['slope_per_0.01Kd']:.3g}\n"
                f"R2={row['r2']:.2f}\n"
                f"rho={row['spearman_rho']:.2f}"
            )
            ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.set_title(f"{model}")
        ax.set_xlabel("Kd(Fe-Mg)")
        ax.set_ylabel(ylabel)
        filename = f"test_subset_kd_vs_delta{target_type}_{model_code(model)}.png"
        save_current_fig(fig_dir / filename)
        count += 1

    for target_type, label in [("P", "pressure"), ("T", "temperature")]:
        slope_data = reg_df[(reg_df["target_type"].eq(target_type)) & (reg_df["response"].eq(f"delta_{target_type}"))]
        slope_data = slope_data.dropna(subset=["slope_per_0.01Kd"]).sort_values("slope_per_0.01Kd")
        if slope_data.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.8, max(3.0, 0.38 * len(slope_data))))
        sns.barplot(data=slope_data, y="model", x="slope_per_0.01Kd", ax=ax, color="#6c8ebf")
        ax.axvline(0, color="0.2", lw=1)
        ax.set_xlabel("Residual slope per 0.01 Kd")
        ax.set_ylabel("")
        ax.set_title(f"Kd residual slopes: {label}")
        save_current_fig(fig_dir / f"test_subset_kd_slope_summary_{label}.png")
        count += 1

    for target_type, label in [("P", "pressure"), ("T", "temperature")]:
        data = bin_df[bin_df["target_type"].eq(target_type)].dropna(subset=["rmse"])
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        sns.lineplot(data=data, x="Kd_bin", y="rmse", hue="model", marker="o", ax=ax)
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("Kd bin")
        ax.set_ylabel("RMSE (kbar)" if target_type == "P" else "RMSE (deg C)")
        ax.set_title(f"Kd-bin RMSE: {label}")
        ax.legend(fontsize=7, loc="best")
        save_current_fig(fig_dir / f"test_subset_kd_bin_rmse_{label}.png")
        count += 1

    overview_data = merged[merged["status"].eq("ok") & merged["Kd_FeMg"].notna()].copy()
    if not overview_data.empty:
        overview_data["residual"] = np.where(overview_data["target_type"].eq("P"), overview_data["delta_P"], overview_data["delta_T"])
        grid = sns.FacetGrid(overview_data, col="target_type", hue="model", height=3.5, aspect=1.2, sharey=False)
        grid.map_dataframe(sns.scatterplot, x="Kd_FeMg", y="residual", s=18, alpha=0.75)
        for ax in grid.axes.flat:
            ax.axhline(0, color="0.25", lw=1, ls="--")
            ax.axvline(0.28, color="0.35", lw=1, ls=":")
        grid.add_legend(fontsize=7)
        grid.fig.suptitle("Kd residual overview", y=1.04)
        grid.savefig(fig_dir / "test_subset_kd_residual_overview.png", dpi=300, bbox_inches="tight")
        plt.close(grid.fig)
        count += 1
    return count


def oxide_total(df: pd.DataFrame, phase: str) -> pd.Series:
    """Calculate oxide total for one phase, excluding H2O."""
    suffix = f"_{phase}"
    cols = [col for col in df.columns if str(col).endswith(suffix) and not str(col).startswith("H2O")]
    cols = [col for col in cols if any(col.startswith(base) for base in ["SiO2", "TiO2", "Al2O3", "Fe2O3", "Cr2O3", "FeO", "MnO", "MgO", "NiO", "CoO", "CaO", "Na2O", "K2O", "P2O5"])]
    return df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)


def cpx_stoich_ratio(df: pd.DataFrame) -> pd.Series:
    """Calculate simple cpx stoichiometric screening ratio."""
    required = ["SiO2_cpx", "CaO_cpx", "MgO_cpx", "FeO_cpx"]
    if any(col not in df.columns for col in required):
        return pd.Series(np.nan, index=df.index)
    si = pd.to_numeric(df["SiO2_cpx"], errors="coerce") / 60.0843
    ca = pd.to_numeric(df["CaO_cpx"], errors="coerce") / 56.0774
    mg = pd.to_numeric(df["MgO_cpx"], errors="coerce") / 40.3044
    fe = pd.to_numeric(df["FeO_cpx"], errors="coerce") / 71.844
    if "Fe2O3_cpx" in df.columns:
        fe = fe + 2 * pd.to_numeric(df["Fe2O3_cpx"], errors="coerce").fillna(0) / 159.688
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (ca + fe + mg) / si
    ratio[(si <= 0) | ~np.isfinite(ratio)] = np.nan
    return ratio


def model_uses_oxide(spec: ModelSpec, phase: str, oxide: str) -> bool:
    """Infer whether an oxide appears in the model's declared input columns."""
    cols = getattr(spec.model, "standard_columns", []) or []
    suffix = f"_{phase}"
    base = oxide.replace(suffix, "")
    if base == "FeO":
        candidates = {"FeO", "FeOt"}
    else:
        candidates = {base}
    for col in cols:
        col_text = str(col)
        if not col_text.endswith(suffix):
            continue
        if any(col_text.startswith(candidate) for candidate in candidates):
            return True
    return False


def baseline_lookup(baseline: pd.DataFrame) -> dict[tuple[object, str, str], float]:
    """Create a lookup from sample/model/target to baseline prediction."""
    lookup: dict[tuple[object, str, str], float] = {}
    for _, row in baseline.iterrows():
        col = "P_pred" if row["target_type"] == "P" else "T_pred"
        lookup[(row["id"], row["model"], row["target_type"])] = row[col]
    return lookup


def perturbation_groups(test_df: pd.DataFrame) -> list[tuple[str, str, float, str, str]]:
    """Return requested phase/oxide/error/sign perturbation groups."""
    groups: list[tuple[str, str, float, str, str]] = []
    for oxide in [col for col in CPX_OXIDES if col in test_df.columns]:
        for rel_error, label in CPX_LEVELS:
            for sign in ("plus", "minus"):
                groups.append(("cpx", oxide, rel_error, label, sign))
    for oxide in [col for col in LIQ_OXIDES if col in test_df.columns]:
        for rel_error, label in LIQ_LEVELS:
            for sign in ("plus", "minus"):
                groups.append(("liq", oxide, rel_error, label, sign))
    return groups


def run_analytical_oat(
    test_df: pd.DataFrame,
    specs: list[ModelSpec],
    baseline: pd.DataFrame,
    paths: dict[str, Path],
    reuse_existing: bool,
) -> tuple[pd.DataFrame, int]:
    """Run deterministic one-at-a-time analytical perturbations."""
    long_path = paths["analytical"] / "analytical_oat_long_test_subset.csv"
    if reuse_existing and long_path.exists():
        candidate = normalize_output_model_names(pd.read_csv(long_path, low_memory=False))
        expected_models = {spec.name for spec in specs}
        expected_rows = len(test_df) * len(specs) * len(perturbation_groups(test_df))
        actual_models = set(candidate["model"].dropna().astype(str))
        if len(candidate) == expected_rows and actual_models == expected_models:
            long_df = candidate
            long_df.to_csv(long_path, index=False)
            print(f"Reused analytical long-form results: {long_path}")
        else:
            print(
                "Existing analytical long-form results are not compatible after "
                f"model filtering; recomputing. rows={len(candidate)}, "
                f"expected_rows={expected_rows}, models={sorted(actual_models)}"
            )
            long_df = compute_analytical_long(test_df, specs, baseline, long_path)
    else:
        long_df = compute_analytical_long(test_df, specs, baseline, long_path)
        print(f"Saved analytical long-form results: {long_path}")

    write_analytical_summaries(long_df, paths["analytical"])
    fig_count = plot_analytical_figures(long_df, paths["analytical_fig"])
    return long_df, fig_count


def compute_analytical_long(
    test_df: pd.DataFrame,
    specs: list[ModelSpec],
    baseline: pd.DataFrame,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Compute the analytical OAT long table."""
    groups = perturbation_groups(test_df)
    lookup = baseline_lookup(baseline)
    if out_path is not None and out_path.exists():
        out_path.unlink()
    records: list[pd.DataFrame] = []
    for group_i, (phase, oxide, rel_error, label, sign) in enumerate(groups, start=1):
        print(
            f"Analytical perturbation group {group_i}/{len(groups)}: "
            f"{phase} {oxide} {label} {sign}",
            flush=True,
        )
        perturbed = test_df.copy()
        original = pd.to_numeric(perturbed[oxide], errors="coerce").fillna(0.0)
        factor = 1.0 + rel_error if sign == "plus" else 1.0 - rel_error
        new_value = original * factor
        clipped = pd.Series(False, index=perturbed.index)
        if sign == "minus":
            clipped = new_value < 0
            new_value = new_value.clip(lower=0)
        zero_no_effect = original.eq(0)
        perturbed[oxide] = new_value

        cpx_total = oxide_total(perturbed, "cpx")
        liq_total = oxide_total(perturbed, "liq")
        stoich = cpx_stoich_ratio(perturbed)
        base_status = pd.Series("ok", index=perturbed.index, dtype=object)
        qc_message = pd.Series("", index=perturbed.index, dtype=object)
        if phase == "cpx":
            cpx_fail = (~cpx_total.between(98, 102)) | (~stoich.between(0.9, 1.1)) | stoich.isna()
            base_status.loc[cpx_fail] = "qc_failed_cpx"
            qc_message.loc[cpx_fail] = "cpx total or stoichiometry outside accepted range"
        else:
            liq_fail = liq_total > 105
            base_status.loc[liq_fail] = "qc_failed_liq_total_high"
            qc_message.loc[liq_fail] = "liquid total exceeds 105 wt.%"

        cpx_all, liq_all = split_inputs(perturbed)
        for spec in specs:
            not_used_by_model = not model_uses_oxide(spec, phase, oxide)
            rows = pd.DataFrame(
                {
                    "id": perturbed["id"],
                    "split": "test_subset",
                    "model": spec.name,
                    "target_type": spec.target_type,
                    "phase": phase,
                    "oxide": oxide,
                    "rel_error": rel_error,
                    "rel_error_label": label,
                    "sign": sign,
                    "baseline_value": original,
                    "perturbed_value": new_value,
                    "P_baseline": np.nan,
                    "T_baseline": np.nan,
                    "P_perturbed": np.nan,
                    "T_perturbed": np.nan,
                    "deltaP_from_baseline": np.nan,
                    "deltaT_from_baseline": np.nan,
                    "abs_deltaP_from_baseline": np.nan,
                    "abs_deltaT_from_baseline": np.nan,
                    "cpx_total": cpx_total,
                    "cpx_stoich_ratio": stoich,
                    "liq_total": liq_total,
                    "zero_value_no_effect": zero_no_effect,
                    "clipped_to_zero": clipped,
                    "not_used_by_model": not_used_by_model,
                    "status": base_status,
                    "qc_error_message": qc_message,
                    "error_message": "",
                }
            )
            for idx, sample_id in perturbed["id"].items():
                baseline_value = lookup.get((sample_id, spec.name, spec.target_type), np.nan)
                if spec.target_type == "P":
                    rows.loc[idx, "P_baseline"] = baseline_value
                else:
                    rows.loc[idx, "T_baseline"] = baseline_value

            ok_mask = rows["status"].eq("ok")
            if not_used_by_model:
                pred = pd.Series(index=rows.index[ok_mask], dtype="float64")
                errors = {idx: "" for idx in rows.index[ok_mask]}
                for idx in rows.index[ok_mask]:
                    baseline_col = "P_baseline" if spec.target_type == "P" else "T_baseline"
                    pred.loc[idx] = rows.loc[idx, baseline_col]
            else:
                pred, errors = safe_predict(spec, cpx_all.loc[ok_mask], liq_all.loc[ok_mask])

            for idx in rows.index[ok_mask]:
                err = errors.get(idx, "")
                value = pred.loc[idx] if idx in pred.index else np.nan
                if err or pd.isna(value):
                    rows.loc[idx, "status"] = "calculation_failed"
                    rows.loc[idx, "error_message"] = err or "prediction returned NaN or missing baseline"
                    continue
                if spec.target_type == "P":
                    rows.loc[idx, "P_perturbed"] = value
                    delta = value - rows.loc[idx, "P_baseline"]
                    rows.loc[idx, "deltaP_from_baseline"] = delta
                    rows.loc[idx, "abs_deltaP_from_baseline"] = abs(delta)
                else:
                    rows.loc[idx, "T_perturbed"] = value
                    delta = value - rows.loc[idx, "T_baseline"]
                    rows.loc[idx, "deltaT_from_baseline"] = delta
                    rows.loc[idx, "abs_deltaT_from_baseline"] = abs(delta)
            if out_path is not None:
                rows.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            else:
                records.append(rows)
    if out_path is not None:
        return pd.read_csv(out_path)
    return pd.concat(records, ignore_index=True)


def effect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add generic signed and absolute effect columns."""
    out = df.copy()
    out["signed_effect"] = np.where(out["target_type"].eq("P"), out["deltaP_from_baseline"], out["deltaT_from_baseline"])
    out["abs_effect"] = np.where(out["target_type"].eq("P"), out["abs_deltaP_from_baseline"], out["abs_deltaT_from_baseline"])
    return out


def summarize_effects(group: pd.DataFrame) -> pd.Series:
    """Summarize ok perturbation effects for a group."""
    ok = group[group["status"].eq("ok")]
    abs_eff = ok["abs_effect"].dropna().astype(float)
    signed = ok["signed_effect"].dropna().astype(float)
    return pd.Series(
        {
            "n": int(len(abs_eff)),
            "median_abs_effect": abs_eff.median() if len(abs_eff) else np.nan,
            "mean_abs_effect": abs_eff.mean() if len(abs_eff) else np.nan,
            "p90_abs_effect": abs_eff.quantile(0.90) if len(abs_eff) else np.nan,
            "p95_abs_effect": abs_eff.quantile(0.95) if len(abs_eff) else np.nan,
            "max_abs_effect": abs_eff.max() if len(abs_eff) else np.nan,
            "median_signed_effect": signed.median() if len(signed) else np.nan,
            "mean_signed_effect": signed.mean() if len(signed) else np.nan,
            "qc_failed_n": int(group["status"].astype(str).str.startswith("qc_failed").sum()),
            "clipped_to_zero_n": int(group["clipped_to_zero"].sum()),
            "zero_value_no_effect_n": int(group["zero_value_no_effect"].sum()),
        }
    )


def write_analytical_summaries(long_df: pd.DataFrame, out_dir: Path) -> None:
    """Write analytical summary CSV files."""
    df = effect_columns(long_df)
    by_model_oxide = (
        df.groupby(["model", "target_type", "phase", "oxide", "rel_error"], dropna=False)
        .apply(summarize_effects, include_groups=False)
        .reset_index()
    )
    by_model_oxide.to_csv(out_dir / "analytical_oat_summary_by_model_oxide_test_subset.csv", index=False)

    by_phase = (
        df.groupby(["model", "target_type", "phase", "rel_error"], dropna=False)
        .apply(summarize_effects, include_groups=False)
        .reset_index()
    )
    by_phase.to_csv(out_dir / "analytical_oat_summary_by_phase_test_subset.csv", index=False)

    top_rows = []
    largest = df[((df["phase"].eq("cpx")) & (df["rel_error"].eq(0.010))) | ((df["phase"].eq("liq")) & (df["rel_error"].eq(0.030)))]
    feature_summary = (
        largest.groupby(["model", "target_type", "phase", "oxide"], dropna=False)
        .apply(summarize_effects, include_groups=False)
        .reset_index()
    )
    for (model, target_type), group in feature_summary.groupby(["model", "target_type"]):
        ranked = group.sort_values("median_abs_effect", ascending=False, na_position="last").reset_index(drop=True)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            top_rows.append(
                {
                    "rank": rank,
                    "model": model,
                    "target_type": target_type,
                    "phase": row["phase"],
                    "oxide": row["oxide"],
                    "median_abs_effect": row["median_abs_effect"],
                    "p95_abs_effect": row["p95_abs_effect"],
                    "max_abs_effect": row["max_abs_effect"],
                    "valid_n": row["n"],
                    "qc_failed_n": row["qc_failed_n"],
                }
            )
    pd.DataFrame(top_rows).to_csv(out_dir / "analytical_oat_top_sensitive_features_test_subset.csv", index=False)

    total = {
        "total_perturbation_rows": int(len(df)),
        "ok_rows": int(df["status"].eq("ok").sum()),
        "qc_failed_cpx": int(df["status"].eq("qc_failed_cpx").sum()),
        "qc_failed_liq_total_high": int(df["status"].eq("qc_failed_liq_total_high").sum()),
        "cpx_total_min": df["cpx_total"].min(),
        "cpx_total_max": df["cpx_total"].max(),
        "cpx_stoich_ratio_min": df["cpx_stoich_ratio"].min(),
        "cpx_stoich_ratio_max": df["cpx_stoich_ratio"].max(),
        "liq_total_min": df["liq_total"].min(),
        "liq_total_max": df["liq_total"].max(),
    }
    qc_group = (
        df.groupby(["phase", "oxide", "rel_error", "sign", "status"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    for key, value in total.items():
        qc_group[key] = value
    qc_group.to_csv(out_dir / "analytical_oat_qc_summary_test_subset.csv", index=False)


def plot_analytical_figures(long_df: pd.DataFrame, fig_dir: Path) -> int:
    """Generate analytical OAT figures."""
    for old_png in fig_dir.glob("*.png"):
        old_png.unlink()
    df = effect_columns(long_df)
    ok = df[df["status"].eq("ok")].copy()
    count = 0
    sns.set_theme(style="whitegrid", context="paper")

    level_specs = [
        ("cpx", 0.005, "cpx_0p5pct"),
        ("cpx", 0.010, "cpx_1pct"),
        ("liq", 0.010, "liq_1pct"),
        ("liq", 0.020, "liq_2pct"),
        ("liq", 0.030, "liq_3pct"),
    ]
    for target_type, label, unit in [("P", "pressure", "kbar"), ("T", "temperature", "deg C")]:
        for phase, rel_error, token in level_specs:
            data = ok[(ok["target_type"].eq(target_type)) & (ok["phase"].eq(phase)) & (ok["rel_error"].eq(rel_error))]
            if data.empty:
                continue
            pivot = data.groupby(["model", "oxide"])["abs_effect"].median().reset_index()
            matrix = pivot.pivot(index="model", columns="oxide", values="abs_effect")
            if matrix.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(matrix.columns)), max(3, 0.42 * len(matrix.index))))
            sns.heatmap(matrix, cmap="mako", ax=ax, cbar_kws={"label": f"Median abs effect ({unit})"})
            ax.set_title(f"{label.capitalize()} sensitivity: {token}")
            ax.set_xlabel("")
            ax.set_ylabel("")
            save_current_fig(fig_dir / f"test_subset_{label}_heatmap_{token}.png")
            count += 1

    largest = ok[((ok["phase"].eq("cpx")) & (ok["rel_error"].eq(0.010))) | ((ok["phase"].eq("liq")) & (ok["rel_error"].eq(0.030)))]
    tornado = largest.groupby(["model", "target_type", "phase", "oxide"])["abs_effect"].median().reset_index()
    for (model, target_type, phase), group in tornado.groupby(["model", "target_type", "phase"]):
        top = group.sort_values("abs_effect", ascending=False).head(12)
        if top.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.2, max(3.0, 0.32 * len(top))))
        sns.barplot(data=top, y="oxide", x="abs_effect", ax=ax, color="#7a9a64")
        ax.set_xlabel("Median abs effect (kbar)" if target_type == "P" else "Median abs effect (deg C)")
        ax.set_ylabel("")
        ax.set_title(f"{model}: {target_type} {phase}")
        filename = f"test_subset_tornado_{target_type}_{model_code(model)}_{phase}.png"
        save_current_fig(fig_dir / filename)
        count += 1

    phase_summary = ok.groupby(["model", "target_type", "phase", "rel_error"])["abs_effect"].median().reset_index()
    for target_type, label in [("P", "pressure"), ("T", "temperature")]:
        data = phase_summary[phase_summary["target_type"].eq(target_type)]
        if data.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=False)
        for ax, phase in zip(axes, ["cpx", "liq"]):
            pdata = data[data["phase"].eq(phase)]
            if pdata.empty:
                ax.set_visible(False)
                continue
            sns.lineplot(data=pdata, x="rel_error", y="abs_effect", hue="model", marker="o", ax=ax)
            ax.set_title(phase)
            ax.set_xlabel("Relative error")
            ax.set_ylabel("Median abs effect (kbar)" if target_type == "P" else "Median abs effect (deg C)")
            ax.legend(fontsize=6)
        save_current_fig(fig_dir / f"test_subset_phase_summary_{label}.png")
        count += 1

    failures = df[df["status"].astype(str).str.startswith("qc_failed")]
    if failures.empty:
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        ax.text(0.5, 0.5, "No QC-failed perturbations", ha="center", va="center")
        ax.set_axis_off()
    else:
        fail_counts = failures.groupby(["oxide", "rel_error_label", "status"]).size().reset_index(name="n")
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        sns.barplot(data=fail_counts, x="oxide", y="n", hue="status", ax=ax)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("")
        ax.set_ylabel("QC-failed rows")
        ax.set_title("Analytical perturbation QC failures")
    save_current_fig(fig_dir / "test_subset_analytical_qc_failures.png")
    count += 1
    return count


def run_directional_equal_error_analysis(
    test_df: pd.DataFrame,
    paths: dict[str, Path],
    reuse_existing: bool,
) -> tuple[pd.DataFrame, int]:
    """Run equal-error directional OAT for cpx-liq and cpx-only model groups."""
    long_path = paths["analytical"] / "directional_equal_error_oat_long_test_subset.csv"
    specs = make_model_specs("cpx_liq") + make_model_specs("cpx_only")
    expected_rows = 0
    for spec in specs:
        phases = ["cpx", "liq"] if spec.model_type == "cpx_liq" else ["cpx"]
        oxide_count = 0
        if "cpx" in phases:
            oxide_count += len([col for col in CPX_OXIDES if col in test_df.columns])
        if "liq" in phases:
            oxide_count += len([col for col in LIQ_OXIDES if col in test_df.columns])
        expected_rows += len(test_df) * oxide_count * 2

    if reuse_existing and long_path.exists():
        candidate = pd.read_csv(long_path, low_memory=False)
        if len(candidate) == expected_rows and set(candidate["model_type"].dropna()) == {"cpx_liq", "cpx_only"}:
            long_df = candidate
            print(f"Reused directional equal-error OAT results: {long_path}")
        else:
            print(
                "Existing directional equal-error OAT results are not compatible; "
                f"recomputing. rows={len(candidate)}, expected_rows={expected_rows}"
            )
            long_df = compute_directional_equal_error_long(test_df, specs, long_path)
    else:
        long_df = compute_directional_equal_error_long(test_df, specs, long_path)

    summary = summarize_directional_equal_error(long_df)
    summary.to_csv(paths["analytical"] / "directional_equal_error_oat_summary_test_subset.csv", index=False)
    fig_count = plot_directional_equal_error_figures(long_df, paths["analytical_fig"])
    fig_count += plot_directional_equal_abs_composites(long_df, paths["analytical_fig"])
    return long_df, fig_count


def compute_directional_equal_error_long(
    test_df: pd.DataFrame,
    specs: list[ModelSpec],
    out_path: Path,
) -> pd.DataFrame:
    """Compute positive-perturbation directional effects at matched cpx/liquid errors."""
    if out_path.exists():
        out_path.unlink()

    first_write = True
    records: list[pd.DataFrame] = []
    existing_keys: set[tuple[str, str, str, str, str, float]] = set()

    source_path = out_path.parent / "analytical_oat_long_test_subset.csv"
    if source_path.exists():
        source_rows = pd.read_csv(source_path, low_memory=False)
        source_rows = normalize_output_model_names(source_rows)
        source_rows = source_rows[
            source_rows["sign"].eq("plus")
            & source_rows["model"].ne("BT19")
            & (
                (source_rows["phase"].eq("cpx") & source_rows["rel_error"].eq(0.010))
                | (source_rows["phase"].eq("liq") & source_rows["rel_error"].isin([0.010, 0.020]))
            )
        ].copy()
        if not source_rows.empty:
            source_rows["model_type"] = "cpx_liq"
            source_rows["baseline_prediction"] = np.where(
                source_rows["target_type"].eq("P"),
                source_rows["P_baseline"],
                source_rows["T_baseline"],
            )
            source_rows["perturbed_prediction"] = np.where(
                source_rows["target_type"].eq("P"),
                source_rows["P_perturbed"],
                source_rows["T_perturbed"],
            )
            source_rows["signed_effect"] = np.where(
                source_rows["target_type"].eq("P"),
                source_rows["deltaP_from_baseline"],
                source_rows["deltaT_from_baseline"],
            )
            source_rows["abs_effect"] = np.where(
                source_rows["target_type"].eq("P"),
                source_rows["abs_deltaP_from_baseline"],
                source_rows["abs_deltaT_from_baseline"],
            )
            source_rows = source_rows[
                [
                    "id",
                    "split",
                    "model_type",
                    "model",
                    "target_type",
                    "phase",
                    "oxide",
                    "rel_error",
                    "rel_error_label",
                    "sign",
                    "baseline_value",
                    "perturbed_value",
                    "baseline_prediction",
                    "perturbed_prediction",
                    "signed_effect",
                    "abs_effect",
                    "cpx_total",
                    "cpx_stoich_ratio",
                    "liq_total",
                    "not_used_by_model",
                    "status",
                    "qc_error_message",
                    "error_message",
                ]
            ]
            source_rows.to_csv(out_path, mode="a", header=True, index=False)
            first_write = False
            records.append(source_rows)
            existing_keys = {
                (row.model_type, row.target_type, row.model, row.phase, row.oxide, float(row.rel_error))
                for row in source_rows[["model_type", "target_type", "model", "phase", "oxide", "rel_error"]]
                .drop_duplicates()
                .itertuples(index=False)
            }
            print(f"Reused {len(source_rows)} cpx-liq directional rows from {source_path}", flush=True)

    cpx_base, liq_base = split_inputs(test_df)
    baseline_predictions: dict[tuple[str, str, str], tuple[pd.Series, dict[object, str]]] = {}

    def get_baseline_prediction(spec: ModelSpec) -> tuple[pd.Series, dict[object, str]]:
        """Compute and cache baseline predictions only when a model is needed."""
        key = (spec.model_type, spec.target_type, spec.name)
        if key not in baseline_predictions:
            print(
                f"Baseline prediction: {spec.model_type} {spec.target_type} {spec.name}",
                flush=True,
            )
            baseline_predictions[key] = safe_predict(spec, cpx_base, liq_base)
        return baseline_predictions[key]

    levels = [(0.010, "1pct"), (0.020, "2pct")]
    for model_type in ["cpx_liq", "cpx_only"]:
        model_specs = [spec for spec in specs if spec.model_type == model_type]
        phases = ["cpx", "liq"] if model_type == "cpx_liq" else ["cpx"]
        for rel_error, rel_label in levels:
            for phase in phases:
                oxides = CPX_OXIDES if phase == "cpx" else LIQ_OXIDES
                oxides = [oxide for oxide in oxides if oxide in test_df.columns]
                for oxide in oxides:
                    pending_specs = [
                        spec
                        for spec in model_specs
                        if (spec.model_type, spec.target_type, spec.name, phase, oxide, float(rel_error))
                        not in existing_keys
                    ]
                    if not pending_specs:
                        continue
                    print(
                        f"Directional equal-error OAT: {model_type} {rel_label} "
                        f"{phase} {oxide} ({len(pending_specs)} model-target entries)",
                        flush=True,
                    )
                    perturbed = test_df.copy()
                    original = pd.to_numeric(perturbed[oxide], errors="coerce").fillna(0.0)
                    new_value = original * (1.0 + rel_error)
                    perturbed[oxide] = new_value

                    cpx_total = oxide_total(perturbed, "cpx")
                    liq_total = oxide_total(perturbed, "liq")
                    stoich = cpx_stoich_ratio(perturbed)
                    base_status = pd.Series("ok", index=perturbed.index, dtype=object)
                    qc_message = pd.Series("", index=perturbed.index, dtype=object)
                    if phase == "cpx":
                        cpx_fail = (~cpx_total.between(98, 102)) | (~stoich.between(0.9, 1.1)) | stoich.isna()
                        base_status.loc[cpx_fail] = "qc_failed_cpx"
                        qc_message.loc[cpx_fail] = "cpx total or stoichiometry outside accepted range"
                    else:
                        liq_fail = liq_total > 105
                        base_status.loc[liq_fail] = "qc_failed_liq_total_high"
                        qc_message.loc[liq_fail] = "liquid total exceeds 105 wt.%"

                    cpx_perturbed, liq_perturbed = split_inputs(perturbed)
                    for spec in pending_specs:
                        baseline_pred, baseline_errors = get_baseline_prediction(spec)
                        not_used_by_model = not model_uses_oxide(spec, phase, oxide)
                        rows = pd.DataFrame(
                            {
                                "id": perturbed["id"],
                                "split": "test_subset",
                                "model_type": model_type,
                                "model": spec.name,
                                "target_type": spec.target_type,
                                "phase": phase,
                                "oxide": oxide,
                                "rel_error": rel_error,
                                "rel_error_label": rel_label,
                                "sign": "plus",
                                "baseline_value": original,
                                "perturbed_value": new_value,
                                "baseline_prediction": baseline_pred.reindex(perturbed.index),
                                "perturbed_prediction": np.nan,
                                "signed_effect": np.nan,
                                "abs_effect": np.nan,
                                "cpx_total": cpx_total,
                                "cpx_stoich_ratio": stoich,
                                "liq_total": liq_total,
                                "not_used_by_model": not_used_by_model,
                                "status": base_status,
                                "qc_error_message": qc_message,
                                "error_message": "",
                            }
                        )
                        for idx in rows.index:
                            err = baseline_errors.get(idx, "")
                            if err or pd.isna(rows.loc[idx, "baseline_prediction"]):
                                rows.loc[idx, "status"] = "baseline_failed"
                                rows.loc[idx, "error_message"] = err or "baseline prediction returned NaN"

                        ok_mask = rows["status"].eq("ok")
                        if not_used_by_model:
                            pred = rows.loc[ok_mask, "baseline_prediction"].astype(float)
                            errors = {idx: "" for idx in rows.index[ok_mask]}
                        else:
                            pred, errors = safe_predict(spec, cpx_perturbed.loc[ok_mask], liq_perturbed.loc[ok_mask])

                        for idx in rows.index[ok_mask]:
                            err = errors.get(idx, "")
                            value = pred.loc[idx] if idx in pred.index else np.nan
                            if err or pd.isna(value):
                                rows.loc[idx, "status"] = "calculation_failed"
                                rows.loc[idx, "error_message"] = err or "prediction returned NaN"
                                continue
                            rows.loc[idx, "perturbed_prediction"] = value
                            effect = value - rows.loc[idx, "baseline_prediction"]
                            rows.loc[idx, "signed_effect"] = effect
                            rows.loc[idx, "abs_effect"] = abs(effect)

                        rows.to_csv(out_path, mode="a", header=first_write, index=False)
                        first_write = False
                        records.append(rows)

    return pd.concat(records, ignore_index=True)


def repair_directional_equal_error_analysis(
    test_df: pd.DataFrame,
    paths: dict[str, Path],
    model_type: str,
    model_names: list[str],
    target_types: list[str],
) -> tuple[pd.DataFrame, int]:
    """Recompute selected directional OAT model-target entries in-place."""
    if not model_names:
        raise ValueError("--directional-models is required for directional_repair.")
    long_path = paths["analytical"] / "directional_equal_error_oat_long_test_subset.csv"
    if not long_path.exists():
        raise FileNotFoundError(f"Cannot repair missing directional OAT table: {long_path}")

    existing = pd.read_csv(long_path, low_memory=False)
    selected_models = set(model_names)
    selected_targets = {target.upper() for target in target_types}
    specs = [
        spec
        for spec in make_model_specs(model_type)
        if spec.name in selected_models and spec.target_type in selected_targets
    ]
    found = {(spec.name, spec.target_type) for spec in specs}
    requested = {(name, target) for name in selected_models for target in selected_targets}
    missing = sorted(requested - found)
    if missing:
        print(f"WARNING: requested model-target entries not found: {missing}", flush=True)
    if not specs:
        raise ValueError("No matching model-target entries found for directional_repair.")

    rows = compute_directional_equal_error_subset(test_df, specs)
    replace_keys = {(spec.model_type, spec.target_type, spec.name) for spec in specs}
    replace_mask = existing.apply(
        lambda row: (row["model_type"], row["target_type"], row["model"]) in replace_keys,
        axis=1,
    )
    preserved = existing.loc[~replace_mask].copy()
    repaired = pd.concat([preserved, rows], ignore_index=True)
    sort_cols = ["model_type", "target_type", "model", "rel_error", "phase", "oxide", "id"]
    repaired = repaired.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    repaired.to_csv(long_path, index=False)

    summary = summarize_directional_equal_error(repaired)
    summary.to_csv(paths["analytical"] / "directional_equal_error_oat_summary_test_subset.csv", index=False)
    fig_count = plot_directional_equal_error_figures(repaired, paths["analytical_fig"])
    fig_count += plot_directional_equal_abs_composites(repaired, paths["analytical_fig"])
    print(
        f"Repaired directional OAT rows for {len(specs)} model-target entries: "
        f"{', '.join(f'{spec.name}/{spec.target_type}' for spec in specs)}",
        flush=True,
    )
    return repaired, fig_count


def compute_directional_equal_error_subset(test_df: pd.DataFrame, specs: list[ModelSpec]) -> pd.DataFrame:
    """Compute directional OAT rows for a selected set of model-target entries."""
    cpx_base, liq_base = split_inputs(test_df)
    baseline_predictions: dict[tuple[str, str, str], tuple[pd.Series, dict[object, str]]] = {}

    def get_baseline_prediction(spec: ModelSpec) -> tuple[pd.Series, dict[object, str]]:
        """Compute and cache baseline predictions only when a selected model is needed."""
        key = (spec.model_type, spec.target_type, spec.name)
        if key not in baseline_predictions:
            print(
                f"Baseline prediction: {spec.model_type} {spec.target_type} {spec.name}",
                flush=True,
            )
            baseline_predictions[key] = safe_predict(spec, cpx_base, liq_base)
        return baseline_predictions[key]

    records: list[pd.DataFrame] = []
    levels = [(0.010, "1pct"), (0.020, "2pct")]
    for model_type in sorted({spec.model_type for spec in specs}):
        model_specs = [spec for spec in specs if spec.model_type == model_type]
        phases = ["cpx", "liq"] if model_type == "cpx_liq" else ["cpx"]
        for rel_error, rel_label in levels:
            for phase in phases:
                oxides = CPX_OXIDES if phase == "cpx" else LIQ_OXIDES
                oxides = [oxide for oxide in oxides if oxide in test_df.columns]
                for oxide in oxides:
                    print(
                        f"Directional repair OAT: {model_type} {rel_label} "
                        f"{phase} {oxide} ({len(model_specs)} model-target entries)",
                        flush=True,
                    )
                    perturbed = test_df.copy()
                    original = pd.to_numeric(perturbed[oxide], errors="coerce").fillna(0.0)
                    new_value = original * (1.0 + rel_error)
                    perturbed[oxide] = new_value

                    cpx_total = oxide_total(perturbed, "cpx")
                    liq_total = oxide_total(perturbed, "liq")
                    stoich = cpx_stoich_ratio(perturbed)
                    base_status = pd.Series("ok", index=perturbed.index, dtype=object)
                    qc_message = pd.Series("", index=perturbed.index, dtype=object)
                    if phase == "cpx":
                        cpx_fail = (~cpx_total.between(98, 102)) | (~stoich.between(0.9, 1.1)) | stoich.isna()
                        base_status.loc[cpx_fail] = "qc_failed_cpx"
                        qc_message.loc[cpx_fail] = "cpx total or stoichiometry outside accepted range"
                    else:
                        liq_fail = liq_total > 105
                        base_status.loc[liq_fail] = "qc_failed_liq_total_high"
                        qc_message.loc[liq_fail] = "liquid total exceeds 105 wt.%"

                    cpx_perturbed, liq_perturbed = split_inputs(perturbed)
                    for spec in model_specs:
                        baseline_pred, baseline_errors = get_baseline_prediction(spec)
                        not_used_by_model = not model_uses_oxide(spec, phase, oxide)
                        rows = pd.DataFrame(
                            {
                                "id": perturbed["id"],
                                "split": "test_subset",
                                "model_type": model_type,
                                "model": spec.name,
                                "target_type": spec.target_type,
                                "phase": phase,
                                "oxide": oxide,
                                "rel_error": rel_error,
                                "rel_error_label": rel_label,
                                "sign": "plus",
                                "baseline_value": original,
                                "perturbed_value": new_value,
                                "baseline_prediction": baseline_pred.reindex(perturbed.index),
                                "perturbed_prediction": np.nan,
                                "signed_effect": np.nan,
                                "abs_effect": np.nan,
                                "cpx_total": cpx_total,
                                "cpx_stoich_ratio": stoich,
                                "liq_total": liq_total,
                                "not_used_by_model": not_used_by_model,
                                "status": base_status,
                                "qc_error_message": qc_message,
                                "error_message": "",
                            }
                        )
                        for idx in rows.index:
                            err = baseline_errors.get(idx, "")
                            if err or pd.isna(rows.loc[idx, "baseline_prediction"]):
                                rows.loc[idx, "status"] = "baseline_failed"
                                rows.loc[idx, "error_message"] = err or "baseline prediction returned NaN"

                        ok_mask = rows["status"].eq("ok")
                        if not_used_by_model:
                            pred = rows.loc[ok_mask, "baseline_prediction"].astype(float)
                            errors = {idx: "" for idx in rows.index[ok_mask]}
                        else:
                            pred, errors = safe_predict(spec, cpx_perturbed.loc[ok_mask], liq_perturbed.loc[ok_mask])

                        for idx in rows.index[ok_mask]:
                            err = errors.get(idx, "")
                            value = pred.loc[idx] if idx in pred.index else np.nan
                            if err or pd.isna(value):
                                rows.loc[idx, "status"] = "calculation_failed"
                                rows.loc[idx, "error_message"] = err or "prediction returned NaN"
                                continue
                            rows.loc[idx, "perturbed_prediction"] = value
                            effect = value - rows.loc[idx, "baseline_prediction"]
                            rows.loc[idx, "signed_effect"] = effect
                            rows.loc[idx, "abs_effect"] = abs(effect)
                        records.append(rows)

    return pd.concat(records, ignore_index=True)


def summarize_directional_equal_error(long_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize directional equal-error OAT by model and feature."""
    ok = long_df[long_df["status"].eq("ok")].copy()
    summary = (
        ok.groupby(["model_type", "model", "target_type", "phase", "oxide", "rel_error"], dropna=False)
        .agg(
            n=("signed_effect", "size"),
            median_signed_effect=("signed_effect", "median"),
            mean_signed_effect=("signed_effect", "mean"),
            median_abs_effect=("abs_effect", "median"),
            p95_abs_effect=("abs_effect", lambda x: x.quantile(0.95)),
            max_abs_effect=("abs_effect", "max"),
        )
        .reset_index()
    )
    failures = (
        long_df[long_df["status"].ne("ok")]
        .groupby(["model_type", "model", "target_type", "phase", "oxide", "rel_error"], dropna=False)
        .size()
        .reset_index(name="non_ok_n")
    )
    return summary.merge(
        failures,
        on=["model_type", "model", "target_type", "phase", "oxide", "rel_error"],
        how="left",
    ).fillna({"non_ok_n": 0})


def _pil_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a Windows-safe font for PIL figures."""
    names = ["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """Measure text in pixels."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def _blend_rgb(left: tuple[int, int, int], right: tuple[int, int, int], fraction: float) -> tuple[int, int, int]:
    """Blend two RGB colors."""
    fraction = max(0.0, min(1.0, float(fraction)))
    return tuple(int(round(left[i] + (right[i] - left[i]) * fraction)) for i in range(3))


def _signed_effect_rgb(value: float, vmax: float) -> tuple[int, int, int]:
    """Map a signed effect to orange-white-blue."""
    if not np.isfinite(value):
        return (244, 245, 247)
    if vmax <= 0 or not np.isfinite(vmax):
        return (255, 255, 255)
    orange = (204, 111, 71)
    blue = (84, 119, 196)
    white = (255, 255, 255)
    scaled = max(-1.0, min(1.0, float(value) / float(vmax)))
    if scaled < 0:
        return _blend_rgb(white, orange, abs(scaled))
    return _blend_rgb(white, blue, scaled)


def _format_effect_tick(value: float) -> str:
    """Format compact colorbar tick labels."""
    value = float(value)
    if abs(value) >= 100:
        return f"{value:+.0f}"
    if abs(value) >= 10:
        return f"{value:+.1f}"
    return f"{value:+.2f}"


def _draw_directional_panel(
    draw: ImageDraw.ImageDraw,
    matrix: pd.DataFrame | None,
    *,
    x0: int,
    y0: int,
    panel_title: str,
    show_feature_labels: bool,
    vmax: float,
    fonts: dict[str, ImageFont.ImageFont],
    layout: dict[str, int],
) -> tuple[int, int, int, int]:
    """Draw one P or T heatmap panel into a PIL image."""
    label_w = layout["feature_label_w"] if show_feature_labels else 0
    cell_w = layout["cell_w"]
    cell_h = layout["cell_h"]
    header_h = layout["header_h"]
    ink = (31, 36, 48)
    muted = (111, 118, 138)
    axis = (215, 219, 231)

    title_x = x0 + label_w
    draw.text((title_x, y0), panel_title, font=fonts["axis_bold"], fill=ink)
    if matrix is None or matrix.empty:
        draw.text((title_x, y0 + 28), "No models", font=fonts["body"], fill=muted)
        return title_x, y0 + header_h, title_x + 120, y0 + header_h + cell_h

    grid_x = x0 + label_w
    grid_y = y0 + header_h
    nrows, ncols = matrix.shape
    values = matrix.to_numpy(dtype=float)

    for c, model in enumerate(matrix.columns):
        text = str(model)
        tw, th = _text_size(draw, text, fonts["tick"])
        x = grid_x + c * cell_w + max(0, (cell_w - tw) // 2)
        draw.text((x, grid_y - th - 8), text, font=fonts["tick"], fill=ink)

    for r, feature in enumerate(matrix.index):
        y = grid_y + r * cell_h
        if show_feature_labels:
            text = str(feature)
            tw, th = _text_size(draw, text, fonts["tick"])
            draw.text((grid_x - tw - 8, y + max(0, (cell_h - th) // 2)), text, font=fonts["tick"], fill=ink)
        for c in range(ncols):
            x = grid_x + c * cell_w
            fill = _signed_effect_rgb(values[r, c], vmax)
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), fill=fill, outline=(255, 255, 255))

    draw.rectangle(
        (grid_x, grid_y, grid_x + ncols * cell_w, grid_y + nrows * cell_h),
        outline=axis,
        width=1,
    )
    return grid_x, grid_y, grid_x + ncols * cell_w, grid_y + nrows * cell_h


def _draw_directional_colorbar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    height: int,
    vmax: float,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    """Draw a signed-effect colorbar."""
    bar_w = 18
    for i in range(height):
        value = vmax - (2 * vmax * i / max(1, height - 1))
        draw.line((x, y + i, x + bar_w, y + i), fill=_signed_effect_rgb(value, vmax))
    draw.rectangle((x, y, x + bar_w, y + height), outline=(215, 219, 231), width=1)
    draw.text((x - 8, y - 26), "Median signed effect", font=fonts["body"], fill=(31, 36, 48))
    for frac, value in [(0.0, vmax), (0.5, 0.0), (1.0, -vmax)]:
        yy = int(y + frac * height)
        draw.line((x + bar_w + 2, yy, x + bar_w + 7, yy), fill=(70, 76, 85))
        draw.text((x + bar_w + 10, yy - 7), _format_effect_tick(value), font=fonts["small"], fill=(70, 76, 85))


def plot_directional_equal_error_figures(long_df: pd.DataFrame, fig_dir: Path) -> int:
    """Plot model-by-feature directional heatmaps with P and T panels together."""
    for old_png in fig_dir.glob("test_subset_directional_equal_effect_*.png"):
        old_png.unlink()
    for old_png in fig_dir.glob("test_subset_directional_effect_heatmap_*.png"):
        old_png.unlink()

    ok = long_df[long_df["status"].eq("ok")].copy()
    count = 0
    fonts = {
        "title": _pil_font(20, bold=True),
        "subtitle": _pil_font(12),
        "axis_bold": _pil_font(13, bold=True),
        "body": _pil_font(11),
        "tick": _pil_font(10),
        "small": _pil_font(9),
    }
    layout = {"cell_w": 70, "cell_h": 25, "feature_label_w": 112, "header_h": 54}

    for model_type in ["cpx_liq", "cpx_only"]:
        for rel_error, rel_label in [(0.010, "1pct"), (0.020, "2pct")]:
            scope = long_df[long_df["model_type"].eq(model_type) & long_df["rel_error"].eq(rel_error)]
            data = ok[ok["model_type"].eq(model_type) & ok["rel_error"].eq(rel_error)]
            if scope.empty:
                continue

            matrices = []
            feature_order = []
            if model_type == "cpx_liq":
                feature_order.extend([f"cpx:{oxide.replace('_cpx', '')}" for oxide in CPX_OXIDES if oxide in set(scope["oxide"])])
                feature_order.extend([f"liq:{oxide.replace('_liq', '')}" for oxide in LIQ_OXIDES if oxide in set(scope["oxide"])])
            else:
                feature_order.extend([f"cpx:{oxide.replace('_cpx', '')}" for oxide in CPX_OXIDES if oxide in set(scope["oxide"])])

            for target_type in ["P", "T"]:
                target_scope = scope[scope["target_type"].eq(target_type)].copy()
                model_order = list(dict.fromkeys(target_scope["model"].dropna().astype(str)))
                if not model_order:
                    matrices.append(None)
                    continue
                target = data[data["target_type"].eq(target_type)].copy()
                if target.empty:
                    matrix = pd.DataFrame(index=feature_order, columns=model_order, dtype=float)
                    matrices.append(matrix)
                    continue
                target["feature"] = np.where(
                    target["phase"].eq("cpx"),
                    "cpx:" + target["oxide"].str.replace("_cpx", "", regex=False),
                    "liq:" + target["oxide"].str.replace("_liq", "", regex=False),
                )
                signed = target.groupby(["feature", "model"])["signed_effect"].median().reset_index()
                matrix = signed.pivot(index="feature", columns="model", values="signed_effect")
                matrix = matrix.reindex(index=feature_order, columns=model_order)
                matrices.append(matrix)

            finite_values = []
            for matrix in matrices:
                if matrix is not None and not matrix.empty:
                    finite_values.append(matrix.to_numpy(dtype=float).ravel())
            if not finite_values:
                continue
            finite = np.concatenate(finite_values)
            finite = finite[np.isfinite(finite)]
            vmax = np.nanpercentile(np.abs(finite), 95) if len(finite) else 1.0
            if not np.isfinite(vmax) or vmax == 0:
                vmax = np.nanmax(np.abs(finite)) if len(finite) else 1.0
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0

            p_matrix, t_matrix = matrices
            max_rows = max((len(matrix.index) for matrix in matrices if matrix is not None and not matrix.empty), default=1)
            p_cols = 0 if p_matrix is None or p_matrix.empty else len(p_matrix.columns)
            t_cols = 0 if t_matrix is None or t_matrix.empty else len(t_matrix.columns)
            cell_w = layout["cell_w"]
            cell_h = layout["cell_h"]
            label_w = layout["feature_label_w"]
            panel_gap = 46
            margin_x = 28
            title_h = 78
            bottom_h = 42
            colorbar_w = 106
            width = margin_x * 2 + label_w + cell_w * p_cols + panel_gap + cell_w * t_cols + colorbar_w
            height = title_h + layout["header_h"] + cell_h * max_rows + bottom_h

            image = Image.new("RGB", (max(width, 900), max(height, 520)), (252, 252, 253))
            draw = ImageDraw.Draw(image)
            perturbation_label = "cpx/liquid" if model_type == "cpx_liq" else "cpx"
            rel_text = "1%" if rel_label == "1pct" else "2%"
            title = f"Directional OAT effects, {model_type.replace('_', '-')} models"
            subtitle = (
                f"+{rel_text} relative perturbation applied to {perturbation_label}; "
                "cells show median(perturbed prediction - baseline)."
            )
            draw.text((margin_x, 16), title, font=fonts["title"], fill=(31, 36, 48))
            draw.text((margin_x, 45), subtitle, font=fonts["subtitle"], fill=(111, 118, 138))

            y0 = title_h
            p_x0 = margin_x
            _, grid_y, p_right, grid_bottom = _draw_directional_panel(
                draw,
                p_matrix,
                x0=p_x0,
                y0=y0,
                panel_title="P prediction",
                show_feature_labels=True,
                vmax=vmax,
                fonts=fonts,
                layout=layout,
            )
            t_x0 = p_right + panel_gap
            _, _, t_right, _ = _draw_directional_panel(
                draw,
                t_matrix,
                x0=t_x0,
                y0=y0,
                panel_title="T prediction",
                show_feature_labels=False,
                vmax=vmax,
                fonts=fonts,
                layout=layout,
            )
            _draw_directional_colorbar(
                draw,
                x=t_right + 28,
                y=grid_y,
                height=max(40, grid_bottom - grid_y),
                vmax=vmax,
                fonts=fonts,
            )

            filename = f"test_subset_directional_equal_effect_{model_type}_{rel_label}.png"
            image.save(fig_dir / filename, dpi=(300, 300))
            count += 1
    return count


def _abs_effect_rgb(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    """Map an absolute effect to a publication-style sequential blue palette."""
    if not np.isfinite(value):
        return (238, 238, 238)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (255, 255, 255)
    span = float(vmax) - float(vmin)
    if span <= 0:
        return (8, 81, 156)
    stops = [
        (0.00, (255, 255, 255)),
        (0.18, (239, 245, 255)),
        (0.40, (198, 219, 239)),
        (0.64, (107, 174, 214)),
        (0.82, (49, 130, 189)),
        (1.00, (8, 81, 156)),
    ]
    scaled = max(0.0, min(1.0, (float(value) - float(vmin)) / span))
    for (left_pos, left_rgb), (right_pos, right_rgb) in zip(stops[:-1], stops[1:]):
        if scaled <= right_pos:
            span = right_pos - left_pos
            fraction = 0.0 if span == 0 else (scaled - left_pos) / span
            return _blend_rgb(left_rgb, right_rgb, fraction)
    return stops[-1][1]


def _format_abs_tick(value: float) -> str:
    """Format colorbar tick labels for absolute effects."""
    value = float(value)
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _draw_rotated_text(
    image: Image.Image,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    *,
    angle: int = 90,
) -> None:
    """Draw rotated text onto a PIL image."""
    temp = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    tw, th = _text_size(temp_draw, text, font)
    temp = Image.new("RGBA", (tw + 8, th + 8), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((4, 4), text, font=font, fill=fill)
    rotated = temp.rotate(angle, expand=True)
    image.paste(rotated, xy, rotated)


def _phase_oxide_order(phase: str, available: set[str]) -> list[str]:
    """Return alphabetically ordered oxide labels for a phase."""
    oxides = CPX_OXIDES if phase == "cpx" else LIQ_OXIDES
    return sorted([oxide for oxide in oxides if oxide in available])


def _abs_effect_matrix(
    long_df: pd.DataFrame,
    *,
    model_type: str,
    rel_error: float,
    target_type: str,
    phase: str,
) -> pd.DataFrame | None:
    """Build a model-by-oxide matrix of median absolute effects."""
    scope = long_df[
        long_df["model_type"].eq(model_type)
        & long_df["rel_error"].eq(rel_error)
        & long_df["target_type"].eq(target_type)
        & long_df["phase"].eq(phase)
    ].copy()
    if scope.empty:
        return None

    model_order = sorted(scope["model"].dropna().astype(str).unique())
    oxide_order = _phase_oxide_order(phase, set(scope["oxide"].dropna()))
    if not model_order or not oxide_order:
        return None

    ok = scope[scope["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame(index=model_order, columns=oxide_order, dtype=float)

    values = ok.groupby(["model", "oxide"])["abs_effect"].median().reset_index()
    matrix = values.pivot(index="model", columns="oxide", values="abs_effect")
    return matrix.reindex(index=model_order, columns=oxide_order)


def _draw_abs_colorbar(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    height: int,
    vmin: float,
    vmax: float,
    unit: str,
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    """Draw a vertical absolute-effect colorbar."""
    bar_w = 22
    ink = (20, 20, 20)
    muted = (60, 60, 60)
    span = float(vmax) - float(vmin)
    for i in range(height):
        value = float(vmax) - span * i / max(1, height - 1)
        draw.line((x, y + i, x + bar_w, y + i), fill=_abs_effect_rgb(value, vmin, vmax))
    draw.rectangle((x, y, x + bar_w, y + height), outline=(120, 120, 120), width=1)
    for frac, value in [(0.0, vmax), (0.5, (float(vmin) + float(vmax)) / 2), (1.0, vmin)]:
        yy = int(y + frac * height)
        draw.line((x + bar_w + 3, yy, x + bar_w + 12, yy), fill=muted, width=2)
        draw.text((x + bar_w + 16, yy - 9), _format_abs_tick(value), font=fonts["tick"], fill=ink)
    label = f"Median abs effect ({unit})"
    _draw_rotated_text(image, label, (x + bar_w + 54, y + max(0, (height - 260) // 2)), fonts["axis"], ink, angle=90)


def _make_abs_heatmap_panel(
    matrix: pd.DataFrame | None,
    *,
    title: str,
    panel_letter: str,
    unit: str,
    fonts: dict[str, ImageFont.ImageFont],
) -> Image.Image:
    """Render one absolute-effect heatmap panel."""
    cell_w = 52
    cell_h = 44
    left_w = 122
    top_h = 60
    x_label_h = 126
    right_w = 150
    bottom_h = 8
    min_cols = 14
    rows = 1 if matrix is None or matrix.empty else len(matrix.index)
    cols = min_cols if matrix is None or matrix.empty else max(min_cols, len(matrix.columns))
    grid_w = cols * cell_w
    grid_h = rows * cell_h
    width = left_w + grid_w + right_w
    height = top_h + grid_h + x_label_h + bottom_h

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    ink = (20, 20, 20)
    muted = (90, 90, 90)
    axis = (110, 110, 110)

    draw.text((6, 6), panel_letter, font=fonts["letter"], fill=ink)
    tw, _ = _text_size(draw, title, fonts["title"])
    draw.text((left_w + max(0, (grid_w - tw) // 2), 10), title, font=fonts["title"], fill=ink)

    grid_x = left_w
    grid_y = top_h
    if matrix is None or matrix.empty:
        draw.rectangle((grid_x, grid_y, grid_x + grid_w, grid_y + grid_h), fill=(238, 238, 238), outline=axis)
        draw.text((grid_x + 12, grid_y + 8), "No available results", font=fonts["tick"], fill=muted)
        _draw_abs_colorbar(
            image,
            draw,
            x=grid_x + grid_w + 34,
            y=grid_y,
            height=grid_h,
            vmin=0.0,
            vmax=1.0,
            unit=unit,
            fonts=fonts,
        )
        return image

    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmin = float(np.nanmin(finite)) if len(finite) else 0.0
    vmax = float(np.nanmax(finite)) if len(finite) else 1.0
    for r, model in enumerate(matrix.index):
        label = str(model)
        lw, lh = _text_size(draw, label, fonts["model"])
        y = grid_y + r * cell_h + max(0, (cell_h - lh) // 2)
        draw.text((grid_x - lw - 8, y), label, font=fonts["model"], fill=ink)
        for c, oxide in enumerate(matrix.columns):
            value = values[r, c]
            x = grid_x + c * cell_w
            y0 = grid_y + r * cell_h
            draw.rectangle(
                (x, y0, x + cell_w - 1, y0 + cell_h - 1),
                fill=_abs_effect_rgb(value, vmin, vmax),
                outline=(255, 255, 255),
            )

    for c, oxide in enumerate(matrix.columns):
        text = str(oxide)
        _, label_h = _text_size(draw, text, fonts["feature"])
        x = grid_x + c * cell_w + max(0, (cell_w - label_h - 8) // 2)
        y = grid_y + grid_h + 6
        _draw_rotated_text(image, text, (x, y), fonts["feature"], ink, angle=90)

    draw.rectangle((grid_x, grid_y, grid_x + len(matrix.columns) * cell_w, grid_y + grid_h), outline=axis, width=1)
    _draw_abs_colorbar(
        image,
        draw,
        x=grid_x + len(matrix.columns) * cell_w + 34,
        y=grid_y,
        height=grid_h,
        vmin=vmin,
        vmax=vmax,
        unit=unit,
        fonts=fonts,
    )
    return image


def _compose_panel_grid(panels: list[Image.Image], *, columns: int, gap_x: int = 18, gap_y: int = 20) -> Image.Image:
    """Combine rendered panels into a single figure."""
    rows = int(math.ceil(len(panels) / columns))
    col_widths = [
        max((panels[i].width for i in range(col, len(panels), columns)), default=0)
        for col in range(columns)
    ]
    row_heights = [
        max((panels[i].height for i in range(row * columns, min((row + 1) * columns, len(panels)))), default=0)
        for row in range(rows)
    ]
    margin = 6
    width = margin * 2 + sum(col_widths) + gap_x * (columns - 1)
    height = margin * 2 + sum(row_heights) + gap_y * (rows - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = margin
    for row in range(rows):
        x = margin
        for col in range(columns):
            idx = row * columns + col
            if idx >= len(panels):
                break
            canvas.paste(panels[idx], (x, y))
            x += col_widths[col] + gap_x
        y += row_heights[row] + gap_y
    return canvas


def plot_directional_equal_abs_composites(long_df: pd.DataFrame, fig_dir: Path) -> int:
    """Plot composite absolute-effect heatmaps by target and phase."""
    for old_png in fig_dir.glob("test_subset_directional_equal_abs_composite_*.png"):
        old_png.unlink()

    fonts = {
        "letter": _pil_font(30, bold=True),
        "title": _pil_font(30),
        "axis": _pil_font(22),
        "model": _pil_font(22),
        "feature": _pil_font(21),
        "tick": _pil_font(20),
    }
    panel_letters = list("abcd")
    count = 0
    for model_type in ["cpx_liq", "cpx_only"]:
        for rel_error, rel_label in [(0.010, "1pct"), (0.020, "2pct")]:
            rel_text = "1pct" if rel_label == "1pct" else "2pct"
            panel_specs = [
                ("P", "cpx", "Pressure", "kbar"),
                ("P", "liq", "Pressure", "kbar"),
                ("T", "cpx", "Temperature", "deg C"),
                ("T", "liq", "Temperature", "deg C"),
            ]
            if model_type == "cpx_only":
                panel_specs = [
                    ("P", "cpx", "Pressure", "kbar"),
                    ("T", "cpx", "Temperature", "deg C"),
                ]

            panels = []
            for idx, (target_type, phase, target_label, unit) in enumerate(panel_specs):
                matrix = _abs_effect_matrix(
                    long_df,
                    model_type=model_type,
                    rel_error=rel_error,
                    target_type=target_type,
                    phase=phase,
                )
                if matrix is None:
                    continue
                title = f"{target_label} sensitivity: {phase}_{rel_text}"
                panels.append(
                    _make_abs_heatmap_panel(
                        matrix,
                        title=title,
                        panel_letter=panel_letters[idx],
                        unit=unit,
                        fonts=fonts,
                    )
                )

            if not panels:
                continue
            columns = 2 if len(panels) > 1 else 1
            composite = _compose_panel_grid(panels, columns=columns)
            filename = f"test_subset_directional_equal_abs_composite_{model_type}_{rel_label}.png"
            composite.save(fig_dir / filename, dpi=(300, 300))
            count += 1
    return count


def plot_directional_effect_heatmaps(ok: pd.DataFrame, fig_dir: Path) -> int:
    """Plot signed median effects for positive perturbations at the largest error levels."""
    largest_plus = ok[
        ok["sign"].eq("plus")
        & (
            (ok["phase"].eq("cpx") & ok["rel_error"].eq(0.010))
            | (ok["phase"].eq("liq") & ok["rel_error"].eq(0.030))
        )
    ].copy()
    if largest_plus.empty:
        return 0

    order_cpx = [col for col in CPX_OXIDES if col in set(largest_plus["oxide"])]
    order_liq = [col for col in LIQ_OXIDES if col in set(largest_plus["oxide"])]
    oxide_order = order_cpx + order_liq
    count = 0

    for target_type, label, unit in [("P", "pressure", "kbar"), ("T", "temperature", "deg C")]:
        data = largest_plus[largest_plus["target_type"].eq(target_type)]
        if data.empty:
            continue
        signed = data.groupby(["model", "oxide"])["signed_effect"].median().reset_index()
        matrix = signed.pivot(index="model", columns="oxide", values="signed_effect")
        columns = [oxide for oxide in oxide_order if oxide in matrix.columns]
        matrix = matrix.reindex(columns=columns)
        if matrix.empty:
            continue

        finite = matrix.to_numpy(dtype=float)
        vmax = np.nanpercentile(np.abs(finite), 95)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = np.nanmax(np.abs(finite))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0

        fig_width = max(9.5, 0.45 * len(matrix.columns))
        fig_height = max(3.5, 0.48 * len(matrix.index))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="vlag",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            cbar_kws={"label": f"Median signed effect ({unit})"},
            linewidths=0.25,
            linecolor="white",
        )
        if order_cpx and order_liq:
            split_x = len([oxide for oxide in order_cpx if oxide in matrix.columns])
            if 0 < split_x < len(matrix.columns):
                ax.axvline(split_x, color="black", lw=1.2)
                ax.text(split_x / 2, -0.55, "cpx +1%", ha="center", va="bottom", fontsize=9)
                ax.text(
                    split_x + (len(matrix.columns) - split_x) / 2,
                    -0.55,
                    "liq +3%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_title(f"Directional OAT effects on {label} predictions")
        ax.set_xlabel("Positive perturbation feature")
        ax.set_ylabel("Model")
        ax.tick_params(axis="x", rotation=45)
        save_current_fig(fig_dir / f"test_subset_directional_effect_heatmap_{label}.png")
        count += 1

    return count


def write_readme(output_dir: Path, baseline_source: Path | None, qc: dict[str, object]) -> None:
    """Write a concise README describing analysis choices."""
    source_text = str(baseline_source) if baseline_source else "No compatible existing baseline was found; baseline was computed for the test subset only."
    text = f"""# Test-subset uncertainty analysis

This folder contains uncertainty and sensitivity outputs generated from only the independent experimental test subset.

- Input file: `paper/data/independent_data_final.xlsx`, sheet `Sheet1`.
- Split filter: rows where `training/testing` is `testing` or `test` after case-insensitive normalization.
- Retained test-subset rows: {qc.get('testing_rows')}.
- Total input rows: {qc.get('total_rows')}; training rows: {qc.get('training_rows')}.
- Core baseline/Kd/analytical OAT models: cpx-liquid thermobarometers only; cpx-only models were excluded from those core outputs.
- Directional equal-error OAT extension: cpx-liquid and cpx-only model groups are plotted separately at matched 1% and 2% positive perturbations.
- Pairing: fixed experimental cpx-liquid pairs were used; liquids were not re-paired.
- Kd analysis: association between observed Kd(Fe-Mg) and model predictions/residuals, not a re-pairing experiment.
- Analytical perturbations: cpx oxides at 0.5% and 1%; liquid oxides at 1%, 2%, and 3%.
- Perturbation method: deterministic one-at-a-time plus/minus relative perturbations; no Monte Carlo.
- Cpx QC: perturbed cpx compositions must pass 98-102 wt.% total and 0.9-1.1 stoichiometric ratio before model calculation.
- Liquid QC: perturbed liquid compositions must pass the liquid total upper-bound check of <=105 wt.% before model calculation.
- H2O handling: H2O columns were kept fixed and were not perturbed.
- Baseline reuse: {source_text}
"""
    (output_dir / "README_test_subset.md").write_text(text, encoding="utf-8")


def verify_outputs(paths: dict[str, Path]) -> list[Path]:
    """Return all required figure paths that are missing."""
    required = [
        paths["kd_fig"] / "test_subset_kd_bin_residual_iqr_composite.png",
        paths["analytical_fig"] / "test_subset_pressure_heatmap_cpx_0p5pct.png",
        paths["analytical_fig"] / "test_subset_pressure_heatmap_cpx_1pct.png",
        paths["analytical_fig"] / "test_subset_pressure_heatmap_liq_1pct.png",
        paths["analytical_fig"] / "test_subset_pressure_heatmap_liq_2pct.png",
        paths["analytical_fig"] / "test_subset_pressure_heatmap_liq_3pct.png",
        paths["analytical_fig"] / "test_subset_temperature_heatmap_cpx_0p5pct.png",
        paths["analytical_fig"] / "test_subset_temperature_heatmap_cpx_1pct.png",
        paths["analytical_fig"] / "test_subset_temperature_heatmap_liq_1pct.png",
        paths["analytical_fig"] / "test_subset_temperature_heatmap_liq_2pct.png",
        paths["analytical_fig"] / "test_subset_temperature_heatmap_liq_3pct.png",
        paths["analytical_fig"] / "test_subset_phase_summary_pressure.png",
        paths["analytical_fig"] / "test_subset_phase_summary_temperature.png",
        paths["analytical_fig"] / "test_subset_analytical_qc_failures.png",
    ]
    return [path for path in required if not path.exists()]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(PROJECT_ROOT / "paper" / "data" / "independent_data_final.xlsx"))
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "results" / "uncertainty"))
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline", "kd", "analytical_oat", "directional_equal_oat"],
        choices=["baseline", "kd", "analytical_oat", "directional_equal_oat", "directional_repair"],
    )
    parser.add_argument(
        "--directional-model-type",
        choices=["cpx_liq", "cpx_only"],
        default="cpx_only",
        help="Model group to repair when using --experiments directional_repair.",
    )
    parser.add_argument(
        "--directional-models",
        nargs="+",
        default=[],
        help="Model abbreviations to repair, for example: Hig21 Jor22.",
    )
    parser.add_argument(
        "--directional-targets",
        nargs="+",
        choices=["P", "T"],
        default=["P", "T"],
        help="Target types to repair when using --experiments directional_repair.",
    )
    return parser.parse_args()


def main() -> None:
    """Run selected analyses."""
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    paths = ensure_dirs(output_dir)

    test_df, qc = load_test_subset(input_path, args.sheet)
    needs_core_baseline = any(experiment in args.experiments for experiment in ["baseline", "kd", "analytical_oat"])
    if needs_core_baseline:
        specs = make_model_specs()
        baseline, baseline_source, loaded_n, computed_n = compute_baseline(test_df, specs, output_dir, args.reuse_existing)
    else:
        specs = []
        baseline = pd.DataFrame({"status": []})
        baseline_source = None
        loaded_n = 0
        computed_n = 0

    valid_kd = np.nan
    outside_kd = np.nan
    if "kd" in args.experiments:
        _, valid_kd, outside_kd = run_kd_analysis(test_df, baseline, paths)

    analytical_rows = 0
    analytical_ok = 0
    cpx_qc_fail = 0
    liq_qc_fail = 0
    analytical_figures = 0
    if "analytical_oat" in args.experiments:
        long_df, analytical_figures = run_analytical_oat(test_df, specs, baseline, paths, args.reuse_existing)
        analytical_rows = int(len(long_df))
        analytical_ok = int(long_df["status"].eq("ok").sum())
        cpx_qc_fail = int(long_df["status"].eq("qc_failed_cpx").sum())
        liq_qc_fail = int(long_df["status"].eq("qc_failed_liq_total_high").sum())

    directional_rows = 0
    directional_figures = 0
    if "directional_equal_oat" in args.experiments:
        directional_df, directional_figures = run_directional_equal_error_analysis(test_df, paths, args.reuse_existing)
        directional_rows = int(len(directional_df))
    if "directional_repair" in args.experiments:
        directional_df, directional_figures = repair_directional_equal_error_analysis(
            test_df,
            paths,
            args.directional_model_type,
            args.directional_models,
            args.directional_targets,
        )
        directional_rows = int(len(directional_df))

    if needs_core_baseline or not (output_dir / "README_test_subset.md").exists():
        write_readme(output_dir, baseline_source, qc)
    missing_figs = verify_outputs(paths)
    figure_count = len(list(paths["kd_fig"].glob("*.png"))) + len(list(paths["analytical_fig"].glob("*.png")))
    failed_calcs = int((baseline["status"] != "ok").sum())
    if "analytical_oat" in args.experiments:
        failed_calcs += int((long_df["status"] == "calculation_failed").sum())

    print("\nQUALITY CHECKS")
    print(f"1. Test-subset row count: {len(test_df)}")
    print(f"2. Cpx-liquid model-target entries detected: {len(specs)}")
    print(f"3. Baseline predictions loaded: {loaded_n}; newly computed: {computed_n}")
    print(f"4. Valid Kd values: {valid_kd}")
    print(f"5. Kd values outside 0.20-0.36: {outside_kd}")
    print(f"6. Analytical perturbation rows attempted: {analytical_rows}")
    print(f"7. Perturbation rows passing QC/status ok: {analytical_ok}")
    print(f"8. Cpx QC failures: {cpx_qc_fail}")
    print(f"9. Liquid-total QC failures: {liq_qc_fail}")
    print(f"10. Required figures missing: {len(missing_figs)}")
    if missing_figs:
        for path in missing_figs:
            print(f"    MISSING: {path}")
    print(f"11. Failed calculations recorded in output tables: {failed_calcs}")
    print(f"Generated figure count: {figure_count} (analytical figures this run/reuse: {analytical_figures})")
    print(f"Directional equal-error OAT rows: {directional_rows}; figures: {directional_figures}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
