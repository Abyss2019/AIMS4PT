"""Excel reporting helpers for AIMS4PT workflows."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from aims4pt.model_tools.CpxTBSelect import workflow_thermobarometry
    from aims4pt.model_tools.ModelManager import ModelManager

def report_excel(workflow_obj: workflow_thermobarometry, model_list: list[ModelManager], original_data: pd.DataFrame, out_path: str) -> None:
    """Export workflow results to a multi-sheet Excel report.

    Parameters
    ----------
    workflow_obj : workflow_thermobarometry
        Workflow instance that has already run prediction/selection. This function
        reads cached outputs such as ``prediction_df``, ``ood_mask_df``,
        ``calculated_deviation_df``, ``failure_reason_df``, and ``best_model``.
    model_list : list[ModelManager]
        Models included in the report summary and violin-plot uncertainty annotation.
    original_data : pd.DataFrame
        Original input dataset to be included in the report. Should align row-wise
        with ``workflow_obj.prediction_df``.
    out_path : str
        Output ``.xlsx`` file path.

    Returns
    -------
    None

    Report Structure
    ----------------
    ``Results`` sheet:
        Original input columns plus per-model predictions.

    ``Model Summary`` sheet:
        Model metadata, prediction statistics, OOD ratio, model vote counts,
        and an embedded violin plot.

    ``Ranking Details`` sheet:
        Per-sample/per-model failure reason table and selected model label.

    Notes
    -----
    The function expects ``workflow_obj`` to be fully populated (for example via
    ``workflow_obj.predict(...)``) before export.
    """

    # sheet 1: Results 
    results_sheet = pd.concat([original_data.reset_index(drop=True), workflow_obj.prediction_df.reset_index(drop=True)], axis=1)
    results_sheet_col = pd.MultiIndex.from_product([["Original Data"], original_data.columns]).append(
        pd.MultiIndex.from_product([["Model Predictions"], workflow_obj.prediction_df.columns])
    )
    results_sheet.columns = results_sheet_col

    # sheet 2: models summary 
    # a multi-level DataFrame summarizing model details and results summary
    model_summaries = []
    T_P = model_list[0].T_P
    unit = "kbar" if T_P == "P" else "°C"
    for model in model_list:
        comp_range = None
        if model.X_cpx_training is not None:
            comp_range = model.export_composition_range("training", False, True, "text")
        summary = {
            "Model_name": model.model_name,
            "Num_calibration_experiments": len(model.X_cpx_all) if model.X_cpx_all is not None else "not available",
            "Composition_range (wt%)": comp_range if comp_range else "not available",
            f"{T_P}_range ({unit})" : f"{model.y_min}-{model.y_max}" if model.X_cpx_training is not None else "not available",                         
            "Supported_TAS_rock_types": model.rock_types if model.rock_types is not None else "not available",
        }
        model_summaries.append(summary)
    model_summary_df = pd.DataFrame(model_summaries)

    # results summary
    pred = workflow_obj.prediction_df  # shape: (n_samples, n_models)

    results_summary = pd.DataFrame({
        f"{T_P}_min ({'kbar' if T_P=='P' else 'C'})": pred.min(axis=0),
        f"{T_P}_q1 ({'kbar' if T_P=='P' else 'C'})": pred.quantile(0.25, axis=0),
        f"{T_P}_median ({'kbar' if T_P=='P' else 'C'})": pred.median(axis=0),
        f"{T_P}_q3 ({'kbar' if T_P=='P' else 'C'})": pred.quantile(0.75, axis=0),
        f"{T_P}_max ({'kbar' if T_P=='P' else 'C'})": pred.max(axis=0),
        "Mean_calculated_deviation": workflow_obj.calculated_deviation_df.mean(axis=0),
    })

    results_summary = results_summary.reset_index().rename(columns={"index": "Model_name"})

    ood_ratio  = workflow_obj.ood_mask_df.mean(axis=0)


    results_summary["OOD_ratio"] = ood_ratio.values


    model_votes_df = workflow_obj.get_best_model_series().value_counts(dropna=False)
    model_votes_df.columns = ["Votes count"]
    # add a line total number of samples
    model_votes_df.loc["Total"] = model_votes_df.sum()

    from aims4pt.visualization.thermobarometry_plot import violin_plot

    model_uncertainty_dict = {model.model_name: model.uncertainty for model in model_list}
    model_name_list = [model.model_name for model in model_list]
    violin_fig, _, _ = violin_plot(workflow_obj.prediction_df, T_P, model_name_list, model_uncertainty=model_uncertainty_dict)

    # sheet 3: ranking details
    failure_reason_df = workflow_obj.failure_reason_df
    selected_models_series = workflow_obj.get_best_model_series()
    ranking_details_df = pd.concat([failure_reason_df, selected_models_series.rename("Selected_model")], axis=1)


    # create a new Excel writer
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        # write results sheet
        results_sheet.to_excel(writer, sheet_name='Results', index=True)

        model_results_sum_df = pd.concat([model_summary_df.set_index('Model_name'), results_summary.set_index('Model_name')], axis=1).reset_index()
        # write model summary sheet
        model_results_sum_df.to_excel(writer, sheet_name='Model Summary', index=False)

        # write model votes summary
        model_votes_df.to_excel(writer, sheet_name='Model Summary', index=True, startrow=len(model_results_sum_df)+3, float_format="%.2f")

        # write violin plot
        worksheet = writer.sheets['Model Summary']
        # save the figure to a BytesIO object
        import io 
        imgdata = io.BytesIO()
        violin_fig.savefig(imgdata, format='png', bbox_inches='tight', dpi=200)
        imgdata.seek(0)
        # insert the image into the worksheet
        worksheet.insert_image('E11', 'violin_plot.png', {'image_data': imgdata})

        ranking_details_df.to_excel(writer, sheet_name='Ranking Details', index=True)


    return None
