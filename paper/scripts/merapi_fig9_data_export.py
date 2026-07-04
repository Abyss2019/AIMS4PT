"""Export source data for the Merapi Fig. 9 v2 comparison plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from paper.scripts.constants_illustration import get_model_abbreviation
from paper.scripts.merapi_helpers import (
    _convert_depth_range_to_pressure_range,
    _eruption_color,
    _eruption_slot_offset,
    _group_literature_methods,
    _is_melts_modeling_method,
    _model_axis_label,
    _remove_boxplot_outliers,
    _selected_predictions_for_model,
    _selected_predictions_from_best_models,
)


TABLE1_TEST_RMSE = {
    ("P", "cpx_only", "Pu08_32a"): 3.89,
    ("P", "cpx_only", "Pu08_32b"): 3.95,
    ("P", "cpx_only", "Pet20"): 3.77,
    ("P", "cpx_only", "Wan21"): 4.06,
    ("P", "cpx_only", "Hig21"): 4.23,
    ("P", "cpx_only", "Jor22"): 3.68,
    ("P", "cpx_only", "Chi23"): 4.67,
    ("P", "cpx_only", "AgL24"): 4.37,
    ("P", "cpx_liq", "Pu08_31"): 3.57,
    ("P", "cpx_liq", "NP17"): 4.51,
    ("P", "cpx_liq", "Pet20"): 3.45,
    ("P", "cpx_liq", "Jor22"): 2.96,
    ("P", "cpx_liq", "Chi23"): 3.81,
    ("P", "cpx_liq", "AgL24"): 3.00,
    ("T", "cpx_only", "Pu08_32d"): 161.03,
    ("T", "cpx_only", "Pet20"): 82.19,
    ("T", "cpx_only", "Wan21"): 128.45,
    ("T", "cpx_only", "Hig21"): 69.84,
    ("T", "cpx_only", "Jor22"): 94.22,
    ("T", "cpx_only", "Chi23"): 105.68,
    ("T", "cpx_only", "AgL24"): 84.10,
    ("T", "cpx_liq", "Pu08_33"): 81.78,
    ("T", "cpx_liq", "Pet20"): 82.54,
    ("T", "cpx_liq", "Jor22"): 55.11,
    ("T", "cpx_liq", "Chi23"): 59.77,
    ("T", "cpx_liq", "AgL24"): 53.74,
}


AIMS4PT_WORKFLOW_RMSE = {
    ("P", "cpx_only"): 1.532396,
    ("P", "cpx_liq"): 1.684102,
    ("T", "cpx_only"): 58.375700,
    ("T", "cpx_liq"): 40.873331,
}


FIG9_RECOMMENDED_MODEL_ABBREVIATIONS = {
    ("P", "cpx_only", "2006"): "Jor22",
    ("P", "cpx_only", "2010"): "Pet20",
    ("T", "cpx_only", "2006"): "Hig21",
    ("T", "cpx_only", "2010"): "AgL24",
    ("P", "cpx_liq", "2006"): "Pet20",
    ("P", "cpx_liq", "2010"): "Pet20",
    ("T", "cpx_liq", "2006"): "Chi23",
    ("T", "cpx_liq", "2010"): "AgL24",
}


def _quantity_unit(kind: str) -> str:
    return "kbar" if kind == "P" else "degC"


def _finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _table1_test_rmse(kind: str, phase_type: str, model_name: str) -> float:
    abbreviation = get_model_abbreviation(model_name, kind)
    return TABLE1_TEST_RMSE.get((kind, phase_type, abbreviation), np.nan)


def _fig9_recommended_model_name(kind: str, phase_type: str, year: str, df_pred: pd.DataFrame) -> str:
    target_abbrev = FIG9_RECOMMENDED_MODEL_ABBREVIATIONS[(kind, phase_type, year)]
    for model_name in df_pred.columns:
        if get_model_abbreviation(model_name, kind) == target_abbrev:
            return str(model_name)
    raise KeyError(f"Missing Fig. 9 recommended model {target_abbrev} for {kind} {phase_type} {year}.")


def _build_fig9_threshold_columns(
    kind: str,
    state: Mapping[str, Any],
    *,
    use_model_abbreviations: bool = True,
) -> list[dict[str, Any]]:
    kind_state = state[kind]
    columns: list[dict[str, Any]] = []
    for phase_type in ("cpx_only", "cpx_liq"):
        for year in ("2006", "2010"):
            df_pred = kind_state["results"][year][phase_type]
            best_models = kind_state["best_models"][year][phase_type]
            rank_pcts = kind_state["ranks"][year][phase_type]
            fill = "blue" if year == "2006" else "red"
            target_abbrev = FIG9_RECOMMENDED_MODEL_ABBREVIATIONS[(kind, phase_type, year)]
            recommended_model = _fig9_recommended_model_name(kind, phase_type, year, df_pred)
            recommended_pct = next(
                (
                    float(pct)
                    for model_name, pct in rank_pcts
                    if get_model_abbreviation(model_name, kind) == target_abbrev
                ),
                np.nan,
            )

            overall_data = _remove_boxplot_outliers(
                _selected_predictions_from_best_models(df_pred, best_models)
            )
            columns.append(
                {
                    "category": "This study",
                    "type": phase_type,
                    "eruption": year,
                    "model": "overall",
                    "label": "Ov.",
                    "data": overall_data,
                    "fill": fill,
                    "rmse": AIMS4PT_WORKFLOW_RMSE.get((kind, phase_type), np.nan),
                    "rmse_source": "AIMS4PT workflow independent test RMSE",
                    "is_overall": True,
                    "plot_value_filter": "selected best-model values after Tukey filtering",
                }
            )

            model_data = pd.to_numeric(df_pred[recommended_model], errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            )
            favored_data = _selected_predictions_for_model(df_pred, best_models, recommended_model)
            columns.append(
                {
                    "category": "This study",
                    "type": phase_type,
                    "eruption": year,
                    "model": recommended_model,
                    "label": _model_axis_label(
                        recommended_model,
                        kind,
                        use_model_abbreviations=use_model_abbreviations,
                    ),
                    "data": model_data.dropna().to_numpy(dtype=float),
                    "favored_data": favored_data,
                    "pct": recommended_pct,
                    "fill": fill,
                    "rmse": _table1_test_rmse(kind, phase_type, recommended_model),
                    "rmse_source": "Table 1 independent test RMSE",
                    "is_overall": False,
                    "plot_value_filter": "all finite model predictions",
                }
            )
    return columns


def _fig9_published_mineral_thermobarometry_methods(kind: str) -> list[dict[str, Any]]:
    if kind != "P":
        return []
    return [
        {
            "category": "Mineral thermobarometry",
            "type": "cpx-liq",
            "type_label": "cpx-liq",
            "label": "Pre14",
            "thermobatometer": "cpx-liq",
            "eruptions": [
                {
                    "eruption": "2006",
                    "ranges": [(1.0, 5.1)],
                    "uncertainty": np.nan,
                    "range_uncertainties": [np.nan],
                    "notes": "Preece et al. (2014)",
                },
                {
                    "eruption": "2010",
                    "ranges": [(0.8, 5.1)],
                    "uncertainty": np.nan,
                    "range_uncertainties": [np.nan],
                    "notes": "Preece et al. (2014)",
                },
            ],
        },
        {
            "category": "Mineral thermobarometry",
            "type": "amph-only",
            "type_label": "amph-only",
            "label": "Cos13",
            "thermobatometer": "amph-only",
            "eruptions": [
                {
                    "eruption": "2006&2010",
                    "ranges": [(3.0, 5.4), (7.0, 9.0)],
                    "uncertainty": np.nan,
                    "range_uncertainties": [np.nan],
                    "notes": "Costa et al. (2013)",
                },
            ],
        },
        {
            "category": "Mineral thermobarometry",
            "type": "amph-liq",
            "type_label": "amph-liq",
            "label": "Li21",
            "thermobatometer": "amph-liq",
            "eruptions": [
                {
                    "eruption": "2006",
                    "ranges": [(5.6, 5.9)],
                    "uncertainty": np.nan,
                    "range_uncertainties": [np.nan],
                    "notes": "Li et al. (2021)",
                },
                {
                    "eruption": "2010",
                    "ranges": [(5.8, 7.2), (8.4, 8.5)],
                    "uncertainty": np.nan,
                    "range_uncertainties": [np.nan],
                    "notes": "Li et al. (2021)",
                },
            ],
        },
    ]


def _rmse_band(values: Sequence[float], rmse: Any) -> tuple[float, float, float]:
    values_arr = _finite_array(values)
    try:
        rmse_float = float(rmse)
    except Exception:
        return np.nan, np.nan, np.nan
    if values_arr.size == 0 or not np.isfinite(rmse_float) or rmse_float <= 0:
        return np.nan, np.nan, np.nan
    median = float(np.nanmedian(values_arr))
    return median, median - rmse_float, median + rmse_float


def _summary_stats(values: Sequence[float]) -> dict[str, Any]:
    values_arr = _finite_array(values)
    stats = {
        "n_values": int(values_arr.size),
        "min": np.nan,
        "q1": np.nan,
        "median": np.nan,
        "q3": np.nan,
        "max": np.nan,
    }
    if values_arr.size == 0:
        return stats
    q1, median, q3 = np.nanpercentile(values_arr, [25, 50, 75])
    stats.update(
        {
            "min": float(np.nanmin(values_arr)),
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "max": float(np.nanmax(values_arr)),
        }
    )
    return stats


def _this_study_tables_for_kind(
    kind: str,
    state: Mapping[str, Any],
    *,
    this_study_x_spacing: float,
    use_model_abbreviations: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    value_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    axis_rows: list[dict[str, Any]] = []

    this_cols = _build_fig9_threshold_columns(
        kind,
        state,
        use_model_abbreviations=use_model_abbreviations,
    )
    this_x = 1.0 + np.arange(len(this_cols), dtype=float) * float(this_study_x_spacing)

    for plot_index, (col, x_center) in enumerate(zip(this_cols, this_x), start=1):
        values = _finite_array(col.get("data", []))
        rmse_median, rmse_band_min, rmse_band_max = _rmse_band(values, col.get("rmse"))
        model = str(col["model"])
        base = {
            "quantity": kind,
            "unit": _quantity_unit(kind),
            "source_group": "this_study",
            "plot_index": plot_index,
            "x_center": float(x_center),
            "category": col["category"],
            "phase_type": col["type"],
            "eruption": col["eruption"],
            "model": model,
            "model_abbreviation": "overall" if model == "overall" else get_model_abbreviation(model, kind),
            "label": col["label"],
            "is_overall": bool(col.get("is_overall", False)),
            "selection_pct": col.get("pct", np.nan),
            "fill": col.get("fill"),
            "plot_value_filter": col.get("plot_value_filter"),
        }
        summary_rows.append(
            {
                **base,
                **_summary_stats(values),
                "rmse": col.get("rmse", np.nan),
                "rmse_source": col.get("rmse_source"),
                "rmse_band_center": rmse_median,
                "rmse_band_min": rmse_band_min,
                "rmse_band_max": rmse_band_max,
            }
        )
        axis_rows.append(
            {
                "quantity": kind,
                "source_group": "this_study",
                "plot_index": plot_index,
                "x_center": float(x_center),
                "category": col["category"],
                "type": col["type"],
                "label": col["label"],
                "model": model,
                "eruption": col["eruption"],
            }
        )
        for value_index, value in enumerate(values, start=1):
            value_rows.append(
                {
                    **base,
                    "value_index": value_index,
                    "value": float(value),
                }
            )

    return value_rows, summary_rows, axis_rows, this_x


def _literature_tables_for_kind(
    kind: str,
    literature_df: pd.DataFrame,
    this_x: np.ndarray,
    *,
    densities_kg_m3: Optional[Sequence[float]],
    layer_boundaries_km: Optional[Sequence[float]],
    literature_pressure_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    literature_rows: list[dict[str, Any]] = []
    axis_rows: list[dict[str, Any]] = []
    lit_methods = _group_literature_methods(
        literature_df,
        kind=kind,
        literature_pressure_source=literature_pressure_source,
        densities_kg_m3=densities_kg_m3,
        layer_boundaries_km=layer_boundaries_km,
    )
    lit_methods = _fig9_published_mineral_thermobarometry_methods(kind) + lit_methods
    lit_start = (this_x[-1] + 1.0) if this_x.size else 1.0
    x_lit = lit_start + np.arange(len(lit_methods), dtype=float)

    for method_index, (method, x_center) in enumerate(zip(lit_methods, x_lit), start=1):
        is_melts = _is_melts_modeling_method(method, kind)
        axis_rows.append(
            {
                "quantity": kind,
                "source_group": "literature",
                "plot_index": method_index,
                "x_center": float(x_center),
                "category": method.get("category"),
                "type": method.get("type"),
                "label": method.get("label"),
                "model": method.get("thermobatometer"),
                "eruption": np.nan,
            }
        )
        for eruption_index, eruption_item in enumerate(method["eruptions"], start=1):
            ranges = eruption_item["ranges"]
            uncertainties = eruption_item.get("range_uncertainties", [])
            offsets = np.linspace(-0.06, 0.06, max(1, len(ranges)))
            dx = 0.0 if is_melts else _eruption_slot_offset(eruption_item["eruption"], delta=0.16)
            for range_index, (range_min, range_max) in enumerate(ranges, start=1):
                range_offset = 0.0 if is_melts else float(offsets[range_index - 1])
                y_plot = 0.5 * (float(range_min) + float(range_max)) if is_melts else np.nan
                uncertainty = (
                    uncertainties[range_index - 1]
                    if range_index - 1 < len(uncertainties)
                    else eruption_item.get("uncertainty", np.nan)
                )
                literature_rows.append(
                    {
                        "quantity": kind,
                        "unit": _quantity_unit(kind),
                        "source_group": "literature",
                        "method_index": method_index,
                        "eruption_index": eruption_index,
                        "range_index": range_index,
                        "x_center": float(x_center),
                        "x_offset": float(dx + range_offset),
                        "x_plot": float(x_center + dx + range_offset),
                        "y_plot": y_plot,
                        "category": method.get("category"),
                        "type": method.get("type"),
                        "type_label": method.get("type_label", method.get("type")),
                        "label": method.get("label"),
                        "thermobatometer": method.get("thermobatometer"),
                        "eruption": eruption_item.get("eruption"),
                        "range_min": float(range_min),
                        "range_max": float(range_max),
                        "is_point": bool(np.isclose(float(range_min), float(range_max), equal_nan=False)),
                        "is_melts_modeling": bool(is_melts),
                        "uncertainty": uncertainty,
                        "notes": eruption_item.get("notes"),
                        "plot_color": _eruption_color(eruption_item.get("eruption")),
                    }
                )

    return literature_rows, axis_rows


def _reservoir_band_rows(
    pressure_reservoir_bands: Optional[Sequence[Mapping[str, Any]]],
    *,
    densities_kg_m3: Optional[Sequence[float]],
    layer_boundaries_km: Optional[Sequence[float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band_index, band in enumerate(pressure_reservoir_bands or [], start=1):
        min_depth = band.get("min_depth_km", band.get("min km"))
        max_depth = band.get("max_depth_km", band.get("max km"))
        if min_depth is None or max_depth is None:
            continue
        if densities_kg_m3 is None or layer_boundaries_km is None:
            min_pressure = np.nan
            max_pressure = np.nan
        else:
            p0, p1 = _convert_depth_range_to_pressure_range(
                (float(min_depth), float(max_depth)),
                densities_kg_m3=densities_kg_m3,
                layer_boundaries_km=layer_boundaries_km,
            )
            min_pressure, max_pressure = sorted((float(p0), float(p1)))
        rows.append(
            {
                "quantity": "P",
                "unit": "kbar",
                "source_group": "reservoir_band",
                "band_index": band_index,
                "label": band.get("label"),
                "label_clean": str(band.get("label", "")).replace("\n", " "),
                "min_depth_km": float(min_depth),
                "max_depth_km": float(max_depth),
                "min_pressure_kbar": min_pressure,
                "max_pressure_kbar": max_pressure,
                "color": band.get("color"),
                "alpha": band.get("alpha"),
                "hatch": band.get("hatch"),
                "edgecolor": band.get("edgecolor"),
                "linestyle": band.get("linestyle"),
                "text_color": band.get("text_color"),
            }
        )
    return rows


def build_fig9_merapi_different_constraints_v2_data_tables(
    state: Mapping[str, Any],
    pressure_literature: pd.DataFrame,
    temperature_literature: pd.DataFrame,
    *,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    pressure_reservoir_bands: Optional[Sequence[Mapping[str, Any]]] = None,
    literature_pressure_source: str = "auto",
    min_pct: float = 5.0,
    this_study_x_spacing: float = 0.72,
    use_model_abbreviations: bool = True,
) -> dict[str, pd.DataFrame]:
    value_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    literature_rows: list[dict[str, Any]] = []
    axis_rows: list[dict[str, Any]] = []

    for kind, literature_df in (("P", pressure_literature), ("T", temperature_literature)):
        kind_values, kind_summaries, kind_axis, this_x = _this_study_tables_for_kind(
            kind,
            state,
            this_study_x_spacing=this_study_x_spacing,
            use_model_abbreviations=use_model_abbreviations,
        )
        kind_literature, kind_literature_axis = _literature_tables_for_kind(
            kind,
            literature_df,
            this_x,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
            literature_pressure_source=literature_pressure_source,
        )
        value_rows.extend(kind_values)
        summary_rows.extend(kind_summaries)
        literature_rows.extend(kind_literature)
        axis_rows.extend(kind_axis)
        axis_rows.extend(kind_literature_axis)

    metadata_rows = [
        {"key": "figure_name", "value": "fig_9_Merapi_different_constraints_v2"},
        {"key": "pressure_unit", "value": "kbar"},
        {"key": "temperature_unit", "value": "degC"},
        {"key": "this_study_x_spacing", "value": this_study_x_spacing},
        {"key": "min_pct_argument", "value": min_pct},
        {"key": "literature_pressure_source", "value": literature_pressure_source},
        {
            "key": "this_study_values_note",
            "value": "Values are the arrays passed to the Fig. 9 v2 violin plots.",
        },
        {
            "key": "overall_values_note",
            "value": "Overall columns use selected best-model predictions after Tukey-fence filtering.",
        },
        {
            "key": "model_values_note",
            "value": "Representative-model columns use all finite predictions from the configured model.",
        },
    ]

    return {
        "this_study_values": pd.DataFrame(value_rows),
        "this_study_summary": pd.DataFrame(summary_rows),
        "literature_constraints": pd.DataFrame(literature_rows),
        "reservoir_bands": pd.DataFrame(
            _reservoir_band_rows(
                pressure_reservoir_bands,
                densities_kg_m3=densities_kg_m3,
                layer_boundaries_km=layer_boundaries_km,
            )
        ),
        "axis_positions": pd.DataFrame(axis_rows),
        "metadata": pd.DataFrame(metadata_rows),
    }


def export_fig9_merapi_different_constraints_v2_data(
    state: Mapping[str, Any],
    pressure_literature: pd.DataFrame,
    temperature_literature: pd.DataFrame,
    out_path: str | Path,
    *,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    pressure_reservoir_bands: Optional[Sequence[Mapping[str, Any]]] = None,
    literature_pressure_source: str = "auto",
    min_pct: float = 5.0,
    this_study_x_spacing: float = 0.72,
    use_model_abbreviations: bool = True,
) -> dict[str, pd.DataFrame]:
    tables = build_fig9_merapi_different_constraints_v2_data_tables(
        state,
        pressure_literature,
        temperature_literature,
        densities_kg_m3=densities_kg_m3,
        layer_boundaries_km=layer_boundaries_km,
        pressure_reservoir_bands=pressure_reservoir_bands,
        literature_pressure_source=literature_pressure_source,
        min_pct=min_pct,
        this_study_x_spacing=this_study_x_spacing,
        use_model_abbreviations=use_model_abbreviations,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path) as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)
    return tables
