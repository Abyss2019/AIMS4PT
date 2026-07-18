"""Check representative AIMS4PT predictions against a fixed reference case."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


# Disable architecture-specific oneDNN graph rewrites before TensorFlow loads.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from aims4pt.model_tools.CpxTBSelect import workflow_thermobarometry
from aims4pt.model_tools.model_registry import (
    get_models_initial_pools,
    import_all_models,
)


DEFAULT_REFERENCE = Path(__file__).with_name(
    "cpx_only_temperature_reference.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    return parser.parse_args()


def main() -> None:
    """Run deterministic analytical, ML, ONNX, and TensorFlow predictions."""
    args = _parse_args()
    reference_path = Path(args.reference).resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = {
        str(name): float(value)
        for name, value in reference["expected"].items()
    }

    imported, failed = import_all_models(skip_failed=True)
    if failed:
        raise RuntimeError(f"Model imports failed: {failed}")
    print(f"regression_model_modules_imported {len(imported)}")

    model_pool = get_models_initial_pools(
        reference["target"],
        reference["method_type"],
        if_hydrous=bool(reference["if_hydrous"]),
    )
    selected_models = [
        model
        for model in model_pool
        if getattr(model, "model_name", "") in expected
    ]
    selected_names = {
        getattr(model, "model_name", "")
        for model in selected_models
    }
    missing_models = sorted(set(expected) - selected_names)
    if missing_models:
        raise RuntimeError(
            "Reference models are unavailable: " + ", ".join(missing_models)
        )

    workflow = workflow_thermobarometry(selected_models)
    workflow.predict(
        pd.DataFrame([reference["input_cpx"]]),
        None,
        input_melt_TAS=[reference["melt_tas"]],
        melt_TAS_source="input_melt_TAS",
        plot=False,
    )
    actual = {
        str(name): float(value)
        for name, value in workflow.prediction_df.iloc[0].items()
    }

    absolute_tolerance = float(reference["absolute_tolerance"])
    relative_tolerance = float(reference["relative_tolerance"])
    failures: list[str] = []
    for model_name, expected_value in expected.items():
        actual_value = actual[model_name]
        difference = abs(actual_value - expected_value)
        passed = math.isclose(
            actual_value,
            expected_value,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )
        print(
            "regression_result "
            f"model={model_name!r} actual={actual_value:.12g} "
            f"expected={expected_value:.12g} delta={difference:.12g} "
            f"passed={passed}"
        )
        if not passed:
            failures.append(
                f"{model_name}: actual={actual_value}, "
                f"expected={expected_value}, delta={difference}"
            )

    if failures:
        raise RuntimeError(
            "Cross-platform numerical regression failed:\n"
            + "\n".join(failures)
        )
    print(
        "cross_platform_regression_ok "
        f"case={reference['case_id']} models={len(expected)}"
    )


if __name__ == "__main__":
    main()
