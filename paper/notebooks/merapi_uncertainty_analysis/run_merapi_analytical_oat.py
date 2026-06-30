"""Run analytical OAT sensitivity tests on Merapi kd=0.28 cpx-liquid pairs.

The input pairs are taken from the existing Merapi pairing cache:
paper/.cache/add_pre-2006_028/*pairing_kd028*endmembers.xlsx

All generated files are written below this directory by default.
"""

from __future__ import annotations

import argparse
import sys
import warnings
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
UNCERTAINTY_DIR = PROJECT_ROOT / "paper" / "notebooks" / "uncertainty_analysis"
PAPER_SCRIPTS_DIR = PROJECT_ROOT / "paper" / "scripts"
for path in [SRC_DIR, UNCERTAINTY_DIR, PAPER_SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import matplotlib

matplotlib.use("Agg")

import run_uncertainty_experiments as ua


DATASET_LABEL = "merapi_kd028_pass"
PAIR_FILES = {
    2006: "merapi_2006_pairing_kd028_err1e-4_n20000_endmembers.xlsx",
    2010: "merapi_2010_pairing_kd028_err1e-4_n20000_endmembers.xlsx",
}
CPX_PREFIX = "mine__"
LIQ_PREFIX = "liq__"
OXIDES = [
    "SiO2",
    "TiO2",
    "Al2O3",
    "Fe2O3",
    "Cr2O3",
    "FeO",
    "MnO",
    "MgO",
    "NiO",
    "CoO",
    "CaO",
    "Na2O",
    "K2O",
    "P2O5",
]
TRACE_AS_ZERO = {"Fe2O3", "Cr2O3", "NiO", "CoO", "P2O5"}


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    """Create output directories."""
    paths = {
        "root": output_dir,
        "data": output_dir / "data",
        "analytical": output_dir / "analytical",
        "figures": output_dir / "analytical" / "figures_merapi_kd028_pass",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def model_full_name(spec: ua.ModelSpec) -> str:
    """Return the model's registry name when available."""
    return str(getattr(spec.model, "model_name", type(spec.model).__name__))


def load_pairing_workbook(path: Path, eruption: int) -> pd.DataFrame:
    """Load the pass rows from one Merapi pairing workbook."""
    pairs = pd.read_excel(path, sheet_name="pairs")
    if "if_pass_test" not in pairs.columns:
        raise ValueError(f"{path} does not contain if_pass_test")
    pairs = pairs[pairs["if_pass_test"].fillna(False).astype(bool)].copy()
    pairs["eruption"] = eruption
    return pairs


def build_merapi_pairs(pairing_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a cpx-liquid input table from cached Merapi kd=0.28 pass pairs."""
    source_frames: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}
    for eruption, filename in PAIR_FILES.items():
        path = pairing_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing Merapi pairing workbook: {path}")
        frame = load_pairing_workbook(path, eruption)
        source_frames.append(frame)
        source_counts[str(eruption)] = int(len(frame))

    pairs = pd.concat(source_frames, ignore_index=True)
    out = pd.DataFrame(index=pairs.index)
    out["id"] = [f"merapi_{int(row.eruption)}_{i + 1:04d}" for i, row in pairs.iterrows()]
    out["split"] = DATASET_LABEL
    out["eruption"] = pairs["eruption"]
    out["sample_id"] = pairs.get("mine__Sample_id")
    out["crystal_id"] = pairs.get("mine__Crystal_id")
    out["crystal_number"] = pairs.get("mine__Crystal#")
    out["rim_core"] = pairs.get("mine__Rim/core")
    out["source"] = pairs.get("mine__Source")
    out["pair_mine_id"] = pairs.get("mine__pair_mine_id")
    out["pair_liq_id"] = pairs.get("liq__pair_liq_id")
    out["pair_row_idx_mine"] = pairs.get("mine__pair_row_idx")
    out["pair_row_idx_liq"] = pairs.get("liq__pair_row_idx")
    out["liq_is_synthetic"] = pairs.get("liq__liq__is_synthetic")
    out["liq_mix_f"] = pairs.get("liq__liq__mix_f")
    out["liq_endmember1_idx"] = pairs.get("liq__liq__endmember1_idx")
    out["liq_endmember2_idx"] = pairs.get("liq__liq__endmember2_idx")
    out["kd_value"] = pd.to_numeric(pairs.get("kd_value"), errors="coerce")
    out["kd_error"] = pd.to_numeric(pairs.get("kd_error"), errors="coerce")
    out["if_pass_test"] = pairs["if_pass_test"].astype(bool)

    for oxide in OXIDES:
        cpx_col = f"{CPX_PREFIX}{oxide}"
        liq_col = f"{LIQ_PREFIX}{oxide}"
        if cpx_col in pairs.columns:
            out[f"{oxide}_cpx"] = pd.to_numeric(pairs[cpx_col], errors="coerce")
        if liq_col in pairs.columns:
            out[f"{oxide}_liq"] = pd.to_numeric(pairs[liq_col], errors="coerce")

    for oxide in TRACE_AS_ZERO:
        for suffix in ("cpx", "liq"):
            col = f"{oxide}_{suffix}"
            if col in out.columns:
                out[col] = out[col].fillna(0.0)

    qc = {
        "pairing_dir": str(pairing_dir),
        "source_counts": source_counts,
        "total_pass_pairs": int(len(out)),
        "kd_value_min": float(out["kd_value"].min()),
        "kd_value_max": float(out["kd_value"].max()),
        "kd_error_max": float(out["kd_error"].max()),
        "synthetic_liquid_rows": int(out["liq_is_synthetic"].fillna(False).astype(bool).sum()),
    }
    return out.reset_index(drop=True), qc


def split_inputs(
    df: pd.DataFrame,
    spec: ua.ModelSpec | None = None,
    default_p_kbar: float = 5.0,
    default_t_c: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned cpx and liquid input tables for natural Merapi pairs."""
    cpx_cols = [col for col in df.columns if str(col).endswith("_cpx")]
    liq_cols = [col for col in df.columns if str(col).endswith("_liq")]
    cpx = df[cpx_cols].apply(pd.to_numeric, errors="coerce").copy()
    liq = df[liq_cols].apply(pd.to_numeric, errors="coerce").copy()

    for oxide in OXIDES:
        cpx_col = f"{oxide}_cpx"
        liq_col = f"{oxide}_liq"
        if cpx_col not in cpx.columns:
            cpx[cpx_col] = 0.0
        if liq_col not in liq.columns:
            liq[liq_col] = 0.0
    if "H2O_liq" not in liq.columns:
        liq["H2O_liq"] = 0.0

    if spec is not None:
        for col in getattr(spec.model, "standard_columns", []) or []:
            if str(col).endswith("_cpx") and col not in cpx.columns:
                cpx[col] = 0.0
            elif str(col).endswith("_liq") and col not in liq.columns:
                liq[col] = 0.0

    cpx["P_kbar"] = default_p_kbar
    cpx["T_C"] = default_t_c
    cpx.index = df.index
    liq.index = df.index
    return cpx, liq


def predict_with_optional_cache(spec: ua.ModelSpec, cpx: pd.DataFrame, liq: pd.DataFrame):
    """Use fast local helpers when they match the model's public predictor."""
    full_name = model_full_name(spec)
    if "Chicchi" in full_name and getattr(spec.model, "cpx_only", None) is False:
        from aims4pt.model_tools.data.Chicchi23.script.functions_cpx_liq import easy_predict_cpx_liq

        input_data = spec.model.format_input(X_cpx=cpx, X_liq=liq)
        pred = easy_predict_cpx_liq(input_data, T_P=spec.target_type, dir=spec.model.models_dir)
        return pd.Series(pred, index=cpx.index, name="P_kbar" if spec.target_type == "P" else "T_C")
    return spec.model.predict(cpx, liq)


def safe_predict(
    spec: ua.ModelSpec,
    cpx: pd.DataFrame,
    liq: pd.DataFrame,
) -> tuple[pd.Series, dict[object, str]]:
    """Predict with batch calls and fall back to per-row calls if needed."""
    if len(cpx) == 0:
        return pd.Series(dtype=float), {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = predict_with_optional_cache(spec, cpx.copy(), liq.copy())
        pred = pd.Series(pred, index=cpx.index, dtype="float64")
        return pred, {idx: "" for idx in cpx.index}
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


def compute_baseline(
    merapi_df: pd.DataFrame,
    specs: list[ua.ModelSpec],
    out_path: Path,
    default_p_kbar: float,
    default_t_c: float,
) -> pd.DataFrame:
    """Create baseline model predictions for the Merapi pairs."""
    rows: list[dict[str, object]] = []
    for spec_i, spec in enumerate(specs, start=1):
        print(f"Baseline {spec_i}/{len(specs)}: {spec.target_type} {spec.name}", flush=True)
        cpx, liq = split_inputs(merapi_df, spec, default_p_kbar, default_t_c)
        pred, errors = safe_predict(spec, cpx, liq)
        for idx, sample in merapi_df.iterrows():
            value = pred.loc[idx] if idx in pred.index else np.nan
            err = errors.get(idx, "")
            status = "ok" if err == "" and pd.notna(value) else "calculation_failed"
            if err == "" and pd.isna(value):
                err = "prediction returned NaN"
            rows.append(
                {
                    "id": sample["id"],
                    "split": DATASET_LABEL,
                    "eruption": sample["eruption"],
                    "sample_id": sample["sample_id"],
                    "crystal_id": sample["crystal_id"],
                    "pair_mine_id": sample["pair_mine_id"],
                    "pair_liq_id": sample["pair_liq_id"],
                    "kd_value": sample["kd_value"],
                    "kd_error": sample["kd_error"],
                    "model": spec.name,
                    "model_full_name": model_full_name(spec),
                    "target_type": spec.target_type,
                    "P_pred": float(value) if spec.target_type == "P" and pd.notna(value) else np.nan,
                    "T_pred": float(value) if spec.target_type == "T" and pd.notna(value) else np.nan,
                    "status": status,
                    "error_message": err,
                }
            )
    baseline = pd.DataFrame(rows)
    baseline.to_csv(out_path, index=False)
    return baseline


def baseline_lookup(baseline: pd.DataFrame) -> dict[tuple[object, str, str], float]:
    """Create a lookup from sample/model/target to baseline prediction."""
    lookup: dict[tuple[object, str, str], float] = {}
    ok = baseline[baseline["status"].eq("ok")]
    for _, row in ok.iterrows():
        col = "P_pred" if row["target_type"] == "P" else "T_pred"
        lookup[(row["id"], row["model"], row["target_type"])] = row[col]
    return lookup


def compute_analytical_long(
    merapi_df: pd.DataFrame,
    specs: list[ua.ModelSpec],
    baseline: pd.DataFrame,
    out_path: Path,
    default_p_kbar: float,
    default_t_c: float,
) -> pd.DataFrame:
    """Compute the analytical OAT long table for Merapi pairs."""
    groups = ua.perturbation_groups(merapi_df)
    lookup = baseline_lookup(baseline)
    key_cols = ["phase", "oxide", "rel_error", "rel_error_label", "sign", "model", "target_type"]
    completed_keys: set[tuple[object, ...]] = set()
    if out_path.exists():
        existing = pd.read_csv(out_path, low_memory=False)
        if set(key_cols).issubset(existing.columns):
            counts = existing.groupby(key_cols, dropna=False).size().reset_index(name="n")
            complete = counts[counts["n"].eq(len(merapi_df))][key_cols]
            completed_keys = {tuple(row) for row in complete.itertuples(index=False, name=None)}
            if completed_keys:
                existing_complete = existing.merge(complete, on=key_cols, how="inner")
                existing_complete.to_csv(out_path, index=False)
                dropped = len(existing) - len(existing_complete)
                print(
                    f"Resuming analytical table: kept {len(completed_keys)} complete "
                    f"model/groups and dropped {dropped} partial rows.",
                    flush=True,
                )
            else:
                out_path.unlink()
        else:
            out_path.unlink()

    for group_i, (phase, oxide, rel_error, label, sign) in enumerate(groups, start=1):
        print(
            f"Analytical perturbation group {group_i}/{len(groups)}: "
            f"{phase} {oxide} {label} {sign}",
            flush=True,
        )
        perturbed = merapi_df.copy()
        original = pd.to_numeric(perturbed[oxide], errors="coerce").fillna(0.0)
        factor = 1.0 + rel_error if sign == "plus" else 1.0 - rel_error
        new_value = original * factor
        clipped = pd.Series(False, index=perturbed.index)
        if sign == "minus":
            clipped = new_value < 0
            new_value = new_value.clip(lower=0)
        zero_no_effect = original.eq(0)
        perturbed[oxide] = new_value

        cpx_total = ua.oxide_total(perturbed, "cpx")
        liq_total = ua.oxide_total(perturbed, "liq")
        stoich = ua.cpx_stoich_ratio(perturbed)
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

        for spec in specs:
            group_key = (phase, oxide, rel_error, label, sign, spec.name, spec.target_type)
            if group_key in completed_keys:
                continue
            cpx_all, liq_all = split_inputs(perturbed, spec, default_p_kbar, default_t_c)
            not_used_by_model = not ua.model_uses_oxide(spec, phase, oxide)
            rows = pd.DataFrame(
                {
                    "id": perturbed["id"],
                    "split": DATASET_LABEL,
                    "eruption": perturbed["eruption"],
                    "sample_id": perturbed["sample_id"],
                    "crystal_id": perturbed["crystal_id"],
                    "pair_mine_id": perturbed["pair_mine_id"],
                    "pair_liq_id": perturbed["pair_liq_id"],
                    "kd_value": perturbed["kd_value"],
                    "kd_error": perturbed["kd_error"],
                    "model": spec.name,
                    "model_full_name": model_full_name(spec),
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

            ok_mask = rows["status"].eq("ok") & rows[
                "P_baseline" if spec.target_type == "P" else "T_baseline"
            ].notna()
            if not_used_by_model:
                baseline_col = "P_baseline" if spec.target_type == "P" else "T_baseline"
                pred = rows.loc[ok_mask, baseline_col].astype(float)
                errors = {idx: "" for idx in rows.index[ok_mask]}
            else:
                pred, errors = safe_predict(spec, cpx_all.loc[ok_mask], liq_all.loc[ok_mask])

            missing_baseline = rows["status"].eq("ok") & ~ok_mask
            rows.loc[missing_baseline, "status"] = "calculation_failed"
            rows.loc[missing_baseline, "error_message"] = "missing baseline prediction"

            for idx in rows.index[ok_mask]:
                err = errors.get(idx, "")
                value = pred.loc[idx] if idx in pred.index else np.nan
                if err or pd.isna(value):
                    rows.loc[idx, "status"] = "calculation_failed"
                    rows.loc[idx, "error_message"] = err or "prediction returned NaN"
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

            rows.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)

    return pd.read_csv(out_path, low_memory=False)


def rename_test_subset_outputs(out_dir: Path) -> None:
    """Rename reused uncertainty summary outputs to the Merapi dataset label."""
    for path in list(out_dir.glob("*test_subset*")):
        new_name = path.name.replace("test_subset", DATASET_LABEL)
        path.replace(path.with_name(new_name))
    figures = out_dir / "figures_merapi_kd028_pass"
    if figures.exists():
        for path in list(figures.glob("*test_subset*")):
            new_name = path.name.replace("test_subset", DATASET_LABEL)
            path.replace(path.with_name(new_name))


def dedupe_long_table(
    long_df: pd.DataFrame,
    out_path: Path,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Ensure exactly one row per sample, perturbation, model, and target."""
    key_cols = ["id", "phase", "oxide", "rel_error", "rel_error_label", "sign", "model", "target_type"]
    missing = [col for col in key_cols if col not in long_df.columns]
    if missing:
        raise ValueError(f"Cannot de-duplicate analytical long table; missing columns: {missing}")
    before = len(long_df)
    cleaned = long_df.drop_duplicates(key_cols, keep="last").copy()
    after = len(cleaned)
    if before != after:
        print(f"Removed duplicate analytical rows: {before - after}", flush=True)
        cleaned.to_csv(out_path, index=False)
    if expected_rows is not None and after != expected_rows:
        print(f"WARNING: analytical row count is {after}; expected {expected_rows}", flush=True)
    return cleaned


def write_readme(output_dir: Path, qc: dict[str, object], args: argparse.Namespace) -> None:
    """Write a concise run manifest."""
    text = f"""# Merapi analytical OAT sensitivity

Input: cached Merapi cpx-liquid pairing files from `{qc['pairing_dir']}`.

- Dataset label: `{DATASET_LABEL}`
- Kd target: 0.28
- Pairing tolerance in cache filename: err1e-4
- Pass rows: {qc['total_pass_pairs']}
- 2006 rows: {qc['source_counts'].get('2006', 0)}
- 2010 rows: {qc['source_counts'].get('2010', 0)}
- Synthetic liquid rows: {qc['synthetic_liquid_rows']}
- Kd range: {qc['kd_value_min']:.6f} to {qc['kd_value_max']:.6f}
- Max absolute Kd error: {qc['kd_error_max']:.6g}
- Default P/T supplied to non-iterative model inputs: {args.default_p_kbar} kbar, {args.default_t_c} deg C

Files:

- `data/merapi_kd028_pass_pairs.csv`
- `analytical/baseline_cpx_liq_predictions_merapi_kd028_pass.csv`
- `analytical/analytical_oat_long_merapi_kd028_pass.csv`
- `analytical/analytical_oat_summary_by_model_oxide_merapi_kd028_pass.csv`
- `analytical/analytical_oat_summary_by_phase_merapi_kd028_pass.csv`
- `analytical/analytical_oat_top_sensitive_features_merapi_kd028_pass.csv`
- `analytical/analytical_oat_qc_summary_merapi_kd028_pass.csv`
- `analytical/figures_merapi_kd028_pass/*.png`
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairing-dir",
        type=Path,
        default=PROJECT_ROOT / "paper" / ".cache" / "add_pre-2006_028",
        help="Directory containing Merapi kd=0.28 pairing Excel workbooks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "results" / "merapi_kd028_pass",
        help="Output directory.",
    )
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing baseline and OAT tables.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip PNG figure generation.")
    parser.add_argument("--default-p-kbar", type=float, default=5.0)
    parser.add_argument("--default-t-c", type=float, default=1000.0)
    return parser.parse_args()


def main() -> None:
    """Run the Merapi analytical OAT workflow."""
    args = parse_args()
    paths = ensure_dirs(args.output_dir)
    merapi_df, qc = build_merapi_pairs(args.pairing_dir)
    pair_path = paths["data"] / "merapi_kd028_pass_pairs.csv"
    merapi_df.to_csv(pair_path, index=False)
    print(f"Saved Merapi pass pairs: {pair_path} rows={len(merapi_df)}")

    specs = ua.make_model_specs()
    baseline_path = paths["analytical"] / "baseline_cpx_liq_predictions_merapi_kd028_pass.csv"
    if args.reuse_existing and baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        print(f"Reused baseline predictions: {baseline_path}")
    else:
        baseline = compute_baseline(merapi_df, specs, baseline_path, args.default_p_kbar, args.default_t_c)
        print(f"Saved baseline predictions: {baseline_path}")

    long_path = paths["analytical"] / "analytical_oat_long_merapi_kd028_pass.csv"
    if args.reuse_existing and long_path.exists():
        long_df = pd.read_csv(long_path, low_memory=False)
        print(f"Reused analytical long table: {long_path}")
    else:
        long_df = compute_analytical_long(
            merapi_df,
            specs,
            baseline,
            long_path,
            args.default_p_kbar,
            args.default_t_c,
        )
        print(f"Saved analytical long table: {long_path}")
    expected_rows = len(merapi_df) * len(specs) * len(ua.perturbation_groups(merapi_df))
    long_df = dedupe_long_table(long_df, long_path, expected_rows)

    ua.write_analytical_summaries(long_df, paths["analytical"])
    if not args.skip_figures:
        figure_count = ua.plot_analytical_figures(long_df, paths["figures"])
        print(f"Saved analytical figures: {figure_count}")
    rename_test_subset_outputs(paths["analytical"])
    write_readme(args.output_dir, qc, args)

    status_counts = long_df["status"].value_counts(dropna=False).to_dict()
    print("Baseline status:", baseline["status"].value_counts(dropna=False).to_dict())
    print("Analytical status:", status_counts)


if __name__ == "__main__":
    main()
