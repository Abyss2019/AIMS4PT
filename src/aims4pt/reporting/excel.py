"""Excel reporting helpers for AIMS4PT workflows."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO

import pandas as pd

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
            "OOD_ratio": ood_mask_df.mean(axis=0),
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
        model_name for model_name in model_name_list if model_name in workflow_obj.prediction_df
    ]
    numeric_predictions = workflow_obj.prediction_df[available_columns].apply(
        lambda column: pd.to_numeric(column, errors="coerce")
    )
    if numeric_predictions.empty:
        return None
    plottable_columns = [
        column
        for column in numeric_predictions.columns
        if numeric_predictions[column].dropna().shape[0] >= 10
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

    imgdata = io.BytesIO()
    violin_fig.savefig(imgdata, format="png", bbox_inches="tight", dpi=200)
    try:
        import matplotlib.pyplot as plt

        plt.close(violin_fig)
    except Exception:
        pass
    return imgdata.getvalue()


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
    numeric_prefixes = (f"{target}_min", f"{target}_q1", f"{target}_median", f"{target}_q3", f"{target}_max")
    for column in formatted.columns:
        if str(column).startswith(numeric_prefixes):
            formatted[column] = pd.to_numeric(formatted[column], errors="coerce").round(
                precision
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

    return ReportPayload(
        results_sheet=results_sheet,
        model_summary_df=model_summary_df,
        model_votes_df=model_votes_df,
        ranking_details_df=ranking_details_df,
        violin_plot_png=violin_plot_png,
        target=target,
    )


def write_report_excel(payload: ReportPayload, output: str | BinaryIO | io.BytesIO) -> None:
    """Write an Excel report to a filesystem path or in-memory file object."""
    if payload.results_sheet.empty or payload.model_summary_df.empty:
        raise ValueError("Cannot generate an empty report.")

    with pd.ExcelWriter(
        output,
        engine="xlsxwriter",
        engine_kwargs={"options": {"in_memory": True}},
    ) as writer:
        payload.results_sheet.to_excel(writer, sheet_name="Results", index=True)
        payload.model_summary_df.to_excel(writer, sheet_name="Model Summary", index=False)

        vote_start_row = len(payload.model_summary_df) + 3
        payload.model_votes_df.to_excel(
            writer,
            sheet_name="Model Summary",
            index=False,
            startrow=vote_start_row,
            float_format="%.2f",
        )

        worksheet = writer.sheets["Model Summary"]
        if payload.violin_plot_png:
            worksheet.insert_image(
                "E11",
                "violin_plot.png",
                {"image_data": io.BytesIO(payload.violin_plot_png)},
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
