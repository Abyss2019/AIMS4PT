import numpy as np
import pandas as pd
from collections.abc import Iterable
from typing import List

from aims4pt.data_tools.rocks import get_TAS_rock_types, get_volcanic_rock_series
from aims4pt.statistic_tools.density_region_analysis import rock_type_check


VALID_MELT_TAS_FIELDS = (
    "Picrite",
    "Basalt",
    "Basaltic Andesite",
    "Andesite",
    "Dacite",
    "Rhyolite",
    "Foidite",
    "Trachyte",
    "Trachybasalt",
    "Basaltic Trachyandesite",
    "Trachyandesite",
    "Tephrite–Basanite",
    "Phonotephrite",
    "Tephriphonolite",
    "Phonolite",
)


def _validate_input_melt_TAS(input_melt_TAS):
    """Validate optional assumed melt TAS fields."""
    if input_melt_TAS is None:
        return None

    if isinstance(input_melt_TAS, str):
        raise TypeError(
            "input_melt_TAS must be a list-like object of strings, not a single string."
        )

    if isinstance(input_melt_TAS, dict) or not isinstance(input_melt_TAS, Iterable):
        raise TypeError("input_melt_TAS must be a list-like object of strings.")

    fields = list(input_melt_TAS)

    if not all(isinstance(field, str) for field in fields):
        raise TypeError("Every input_melt_TAS field must be a string.")

    deduplicated_fields = list(dict.fromkeys(fields))

    if not 1 <= len(deduplicated_fields) <= 3:
        raise ValueError("input_melt_TAS must contain 1 to 3 TAS fields.")

    invalid_fields = [
        field for field in deduplicated_fields
        if field not in VALID_MELT_TAS_FIELDS
    ]

    if invalid_fields:
        allowed = ", ".join(VALID_MELT_TAS_FIELDS)
        raise ValueError(
            "Unknown input_melt_TAS field(s): "
            f"{invalid_fields}. Allowed options are: {allowed}"
        )

    return deduplicated_fields


def _validate_melt_TAS_source(melt_TAS_source):
    """Validate which melt TAS metadata source should drive petrological checks."""
    allowed_sources = ("input_liq", "input_melt_TAS")
    if melt_TAS_source not in allowed_sources:
        raise ValueError(
            "melt_TAS_source must be one of "
            f"{allowed_sources}, got {melt_TAS_source!r}."
        )
    return melt_TAS_source


class workflow_thermobarometry:
    def __init__(self, model_list, P_T_checker=True, exclude_Putirka2008_models=False):
        """Initialize a thermobarometry model-selection workflow.

        Parameters
        ----------
        model_list : list
            List of model objects to evaluate. Each model is expected to expose
            attributes/methods used in this workflow (for example ``model_name``,
            ``T_P``, ``predict``, optional ``OOD_detector``, optional
            ``deviation_function``, and optional petrological metadata).
        P_T_checker : bool, default True
            Whether to mark predictions outside each model's calibration range
            (``y_min`` to ``y_max``) as invalid.
        exclude_Putirka2008_models : bool, default False
            If ``True``, remove models whose name starts with ``"Putirka, 2008"``
            (case-insensitive) before calculation.

        Notes
        -----
        The workflow assumes all models in ``model_list`` predict the same target
        variable (pressure or temperature). ``self.T_P`` is taken from the first model.
        """
        self.model_list = model_list
        self.T_P = model_list[0].T_P  # assume all models predict the same variable
        self.P_T_checker = P_T_checker
        self.exclude_Putirka2008_models = exclude_Putirka2008_models
        self.prediction_df = None
        self.ood_mask_df = None
        self.calculated_deviation_df = None
        self.petrological_check_df = None
        self.p_t_mask_df = None
        self.has_ood_detector = None

        self.best_model = None
        self.best_prediction = None
        self.failure_reason_df = None
        self.violin_plot_fig = None
        self.violin_plot_ax = None
        self.violin_plot_artists = None

        if self.exclude_Putirka2008_models:
            self.model_list = [
                model for model in self.model_list
                if not model.model_name.lower().startswith("putirka, 2008")
            ]

    def calculation(
        self,
        input_cpx: pd.DataFrame,
        input_liq: pd.DataFrame = None,
        input_melt_TAS: List[str] = None,
        melt_TAS_source: str = "input_liq",
    ) -> None:
        """Run per-model prediction and diagnostics for each sample.

        Parameters
        ----------
        input_cpx : pd.DataFrame
            Clinopyroxene input features. Rows are samples; columns are composition
            or other model-required features.
        input_liq : pd.DataFrame, optional
            Liquid input features aligned to ``input_cpx`` index. If ``None``, a fake
            liquid table containing only ``H2O_liq=0`` is generated for compatibility
            with models that require liquid inputs.
        input_melt_TAS : list of str, optional
            One to three assumed melt TAS fields used for the TAS-field
            melt-composition OOD check. By default these fields are used only when
            ``input_liq`` is not provided. Set ``melt_TAS_source="input_melt_TAS"``
            to use these fields even when ``input_liq`` is provided.

            This parameter does not replace liquid major-element compositions and
            does not create a liquid composition for clinopyroxene-liquid
            thermobarometers. It only supplies TAS-field metadata for the OOD check.
        melt_TAS_source : {"input_liq", "input_melt_TAS"}, default "input_liq"
            Metadata source for TAS petrological checks when both ``input_liq`` and
            ``input_melt_TAS`` are available. ``"input_liq"`` keeps the historical
            behavior and also checks volcanic series from the liquid composition.
            ``"input_melt_TAS"`` uses the supplied TAS fields and skips the
            volcanic-series check.

        Returns
        -------
        None
            Results are stored on the instance:
            - ``self.prediction_df``: model predictions per sample.
            - ``self.ood_mask_df``: feature OOD mask (True means invalid).
            - ``self.calculated_deviation_df``: per-model uncertainty/deviation estimate.
            - ``self.petrological_check_df``: TAS/series validity mask (True means pass).
            - ``self.p_t_mask_df``: calibration-range mask (True means invalid).
            - ``self.has_ood_detector``: whether each model has an OOD detector.

        Notes
        -----
        The TAS-field melt-composition OOD check normally uses measured liquid
        compositions. Users may set ``melt_TAS_source="input_melt_TAS"`` to apply
        this check using independently constrained or assumed melt TAS fields while
        still passing ``input_liq`` for model calculations such as H2O-dependent
        thermometry. If neither ``input_liq`` nor ``input_melt_TAS`` is provided,
        the TAS-field check is skipped.

        Examples
        --------
        >>> wf = workflow_thermobarometry(model_list)
        >>> wf.calculation(
        ...     input_cpx=cpx_df,
        ...     input_liq=None,
        ...     input_melt_TAS=["Basaltic Andesite", "Andesite"],
        ... )
        >>> pred = wf.decision()
        """
        assumed_melt_TAS = _validate_input_melt_TAS(input_melt_TAS)
        melt_TAS_source = _validate_melt_TAS_source(melt_TAS_source)

        # Without measured liquid compositions, only cpx-only models can be evaluated reliably.
        if input_liq is None:
            invalid_models = [
                getattr(model, "model_name", type(model).__name__)
                for model in self.model_list
                if getattr(model, "cpx_only", False) is not True
            ]
            if invalid_models:
                raise ValueError(
                    "input_liq is None, but model_list contains cpx-liq models: "
                    f"{invalid_models}. Provide input_liq or use only models with "
                    "model.cpx_only == True."
                )

        index = input_cpx.index

        # Initialize result DataFrames with aligned index
        prediction_df = pd.DataFrame(index=index)              # predicted value for each model
        ood_mask_df = pd.DataFrame(index=index, dtype=bool)    # True = feature ood (invalid)
        calculated_deviation_df = pd.DataFrame(index=index, dtype=float)
        petrological_check_df = pd.DataFrame(index=index, dtype=bool)  # True = petro pass
        p_t_mask_df = pd.DataFrame(index=index, dtype=bool)    # True = P-T ood (invalid)
        p_t_mask_df[:] = False

        # Track whether each model has an OOD detector
        has_ood_detector = {}

        use_liq_for_petro_check = input_liq is not None and melt_TAS_source == "input_liq"
        use_input_TAS_for_petro_check = (
            assumed_melt_TAS is not None
            and (input_liq is None or melt_TAS_source == "input_melt_TAS")
        )

        # Pre-compute rock types if liquid metadata drives the petrological check.
        if use_liq_for_petro_check:
            TAS_rock_type = input_liq.apply(get_TAS_rock_types, axis=1)
            volcanic_rock_series = input_liq.apply(get_volcanic_rock_series, axis=1)
        else:
            TAS_rock_type = None
            volcanic_rock_series = None

        # If no liq input, create a fake liq input with zeros water
        if input_liq is None:
            input_liq_ = self.get_fake_liq_input(["H2O_liq"], input_cpx)
        else:
            input_liq_ = input_liq

        # Loop over models
        for model in self.model_list:
            model_name = model.model_name

            # 1) model prediction
            y_pred = model.predict(input_cpx, input_liq_)
            y_pred = np.asarray(y_pred).reshape(-1)  # ensure 1D
            prediction_df[model_name] = y_pred

            # 2) feature OOD mask
            ood_detector = getattr(model, "OOD_detector", None)
            has_ood_detector[model_name] = (ood_detector is not None)

            if ood_detector is not None:
                try:
                    ood_mask = ood_detector.is_ood(input_cpx, input_liq_)
                except TypeError:
                    ood_mask = ood_detector.is_ood(input_cpx)
            else:
                ood_mask = np.zeros(len(input_cpx), dtype=bool)

            ood_mask_df[model_name] = np.asarray(ood_mask, dtype=bool)

            # 3) deviation prediction
            deviation_function = getattr(model, "deviation_function", None)
            if deviation_function is not None:
                deviation_values = deviation_function.predict_deviation(input_cpx, input_liq_)
            else:
                deviation_values = np.full(len(input_cpx), np.nan)
            calculated_deviation_df[model_name] = np.asarray(deviation_values, dtype=float)

            # 4) petrological checks (TAS type + volcanic series)
            if use_liq_for_petro_check and hasattr(model, "rock_types") and hasattr(model, "volcanic_rock_series"):
                tas_mask = [
                    rock_type_check(rt, model.rock_types, report=False)
                    for rt in TAS_rock_type
                ]
                series_mask = [
                    rock_type_check(rs, model.volcanic_rock_series, report=False)
                    for rs in volcanic_rock_series
                ]
                petro_mask = [t and s for t, s in zip(tas_mask, series_mask)]
                petrological_check_df[model_name] = np.asarray(petro_mask, dtype=bool)
            elif use_input_TAS_for_petro_check and hasattr(model, "rock_types"):
                tas_pass = any(
                    rock_type_check(rt, model.rock_types, report=False)
                    for rt in assumed_melt_TAS
                )
                petrological_check_df[model_name] = np.full(len(input_cpx), tas_pass, dtype=bool)
            else:
                # Skip this check when liquid/TAS metadata is unavailable.
                petrological_check_df[model_name] = np.ones(len(input_cpx), dtype=bool)

            # 5) P-T check (if enabled)
            if self.P_T_checker:
                y_min = getattr(model, "y_min", None)
                y_max = getattr(model, "y_max", None)
                if y_min is not None and y_max is not None:
                    p_t_mask = (y_pred < y_min) | (y_pred > y_max)
                else:
                    p_t_mask = np.zeros(len(input_cpx), dtype=bool)
                p_t_mask_df[model_name] = np.asarray(p_t_mask, dtype=bool)

        # Save to self
        self.prediction_df = prediction_df
        self.ood_mask_df = ood_mask_df
        self.calculated_deviation_df = calculated_deviation_df
        self.petrological_check_df = petrological_check_df
        self.p_t_mask_df = p_t_mask_df
        self.has_ood_detector = pd.Series(has_ood_detector)

    def decision(
        self,
        prediction_df=None,
        ood_mask_df=None,
        calculated_deviation_df=None,
        petrological_check_df=None,
        p_t_mask_df=None,
    ) -> pd.Series:
        """Select the best model per sample using validity filters and deviation ranking.

        Parameters
        ----------
        prediction_df : pd.DataFrame, optional
            Prediction matrix (samples x models). If ``None``, use
            ``self.prediction_df`` from ``calculation``.
        ood_mask_df : pd.DataFrame, optional
            Feature OOD mask matrix (True means invalid). If ``None``, use
            ``self.ood_mask_df``.
        calculated_deviation_df : pd.DataFrame, optional
            Deviation/uncertainty matrix used for model ranking (lower is better).
            If ``None``, use ``self.calculated_deviation_df``.
        petrological_check_df : pd.DataFrame, optional
            Petrological validity matrix (True means pass). If ``None``, use
            ``self.petrological_check_df``.
        p_t_mask_df : pd.DataFrame, optional
            Calibration-range mask matrix (True means invalid). If ``None``, use
            ``self.p_t_mask_df``.

        Returns
        -------
        pd.Series
            Best prediction value for each sample. Samples with no valid model remain NaN.

        Side Effects
        ------------
        Updates:
        - ``self.best_model``: selected model name per sample.
        - ``self.best_prediction``: returned prediction series.
        - ``self.failure_reason_df``: per-sample x per-model exclusion reason or deviation.

        Notes
        -----
        Invalidity is defined as logical OR of: feature OOD, TAS mismatch (based on
        measured liquid TAS fields or assumed TAS fields supplied through
        ``input_melt_TAS``), volcanic-series mismatch when measured liquid is
        available, and P-T out-of-range.

        Additional rule: if all models that have OOD detectors are feature-OOD for a
        sample, then models without OOD detectors are also excluded as feature-OOD.
        """
        if prediction_df is None:
            prediction_df = self.prediction_df
        if ood_mask_df is None:
            ood_mask_df = self.ood_mask_df
        if calculated_deviation_df is None:
            calculated_deviation_df = self.calculated_deviation_df
        if petrological_check_df is None:
            petrological_check_df = self.petrological_check_df
        if p_t_mask_df is None:
            p_t_mask_df = self.p_t_mask_df

        model_names = prediction_df.columns
        has_detector = self.has_ood_detector.reindex(model_names).fillna(False).to_numpy(dtype=bool)

        # A) Base invalid masks (True = invalid)
        ood_np = ood_mask_df.to_numpy(dtype=bool)                    # feature ood
        tas_ood_np = (~petrological_check_df).to_numpy(dtype=bool)   # TAS ood
        pt_ood_np = p_t_mask_df.to_numpy(dtype=bool)                 # P-T ood

        invalid = (ood_np | tas_ood_np | pt_ood_np).copy()

        # B) if all detector-equipped models are feature ood for a sample,
        #              exclude models without detector too (as feature ood).
        if has_detector.any():
            ood_among_detector = ood_np[:, has_detector]             # (n_samples, n_detector_models)
            all_detector_feature_ood = ood_among_detector.all(axis=1)  # (n_samples,)

            no_detector = ~has_detector
            mark = all_detector_feature_ood[:, None] & no_detector[None, :]
            invalid[mark] = True
            ood_np[mark] = True  # ensure labeled as feature ood in reporting

        # C) Select model by minimal deviation among valid ones
        pred_np = prediction_df.to_numpy()
        dev_raw = calculated_deviation_df.to_numpy()

        dev_np = np.where(np.isfinite(dev_raw), dev_raw, np.inf)
        dev_np = np.where(invalid, np.inf, dev_np)

        n_samples, n_models = pred_np.shape
        chosen_pred = np.full(n_samples, np.nan, dtype=float)
        chosen_model = np.full(n_samples, None, dtype=object)

        rank_idx = np.argsort(dev_np, axis=1)

        for i in range(n_samples):
            if np.isinf(dev_np[i]).all():
                continue

            for j in rank_idx[i]:
                if np.isinf(dev_np[i, j]):
                    continue
                val = pred_np[i, j]
                if not pd.isna(val):
                    chosen_pred[i] = val
                    chosen_model[i] = model_names[j]
                    break

        best_prediction = pd.Series(chosen_pred, index=prediction_df.index, name="Predicted_P_kbar")
        best_model_idx = pd.Series(chosen_model, index=prediction_df.index, name="Best_Model")
        self.best_model = best_model_idx

        # D) Build per-sample × per-model failure_reason_df
        # Priority if multiple invalid: feature ood > TAS ood > P-T ood
        failure_reason_df = pd.DataFrame(index=prediction_df.index, columns=model_names, dtype=object)

        for j, _m in enumerate(model_names):
            for i in range(n_samples):
                if invalid[i, j]:
                    if ood_np[i, j]:
                        failure_reason_df.iat[i, j] = "feature ood"
                    elif tas_ood_np[i, j]:
                        failure_reason_df.iat[i, j] = "TAS ood"
                    elif pt_ood_np[i, j]:
                        failure_reason_df.iat[i, j] = "P-T ood"
                    else:
                        # Fallback (should be rare)
                        failure_reason_df.iat[i, j] = "feature ood"
                else:
                    dv = dev_raw[i, j]
                    if np.isfinite(dv):
                        failure_reason_df.iat[i, j] = float(dv)
                    else:
                        failure_reason_df.iat[i, j] = "Deviation unavailable (NaN/inf)"

        self.failure_reason_df = failure_reason_df
        self.best_prediction = best_prediction
        return best_prediction

    def get_best_model_series(self) -> pd.Series:
        """Return selected best model names for each sample.

        Returns
        -------
        pd.Series
            Model name per sample (object dtype). Unselected samples may be ``None``.

        Raises
        ------
        ValueError
            If predictions have not been generated yet.
        """
        if not hasattr(self, "best_model"):
            raise ValueError("No predictions made yet. Please run the predict method first.")
        return self.best_model

    def get_fake_liq_input(self, columns, input_cpx: pd.DataFrame) -> pd.DataFrame:
        """Create a zero-filled liquid input table aligned to clinopyroxene samples.

        Parameters
        ----------
        columns : list-like
            Column names to create in the fake liquid DataFrame.
        input_cpx : pd.DataFrame
            Clinopyroxene input whose index is reused for row alignment.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``columns`` and the same index as ``input_cpx``,
            filled with zeros.
        """
        fake_liq = pd.DataFrame(index=input_cpx.index, columns=columns)
        fake_liq.fillna(0, inplace=True)
        return fake_liq

    def predict(
        self,
        input_cpx: pd.DataFrame,
        input_liq: pd.DataFrame = None,
        input_melt_TAS: List[str] = None,
        melt_TAS_source: str = "input_liq",
        plot: bool = False,
    ) -> pd.Series:
        """Run the full workflow and return final per-sample predictions.

        Parameters
        ----------
        input_cpx : pd.DataFrame
            Clinopyroxene input table.
        input_liq : pd.DataFrame, optional
            Liquid input table aligned to ``input_cpx``.
        input_melt_TAS : list of str, optional
            One to three assumed melt TAS fields used for the TAS-field OOD check.
            Set ``melt_TAS_source="input_melt_TAS"`` to use these fields even when
            ``input_liq`` is provided. The parameter only supplies TAS-field
            metadata for the OOD check; it does not create liquid oxide
            compositions.
        melt_TAS_source : {"input_liq", "input_melt_TAS"}, default "input_liq"
            Metadata source for TAS petrological checks when both ``input_liq`` and
            ``input_melt_TAS`` are available.
        plot : bool, default False
            Whether to create a violin plot for model predictions after selection.

            

        Returns
        -------
        pd.Series
            Best prediction value per sample after applying all validity checks and
            deviation-based model selection.

        Notes
        -----
        This method calls ``calculation`` followed by ``decision`` and populates
        side outputs on the instance (for example ``best_model`` and
        ``failure_reason_df``).
        """
        self.calculation(
            input_cpx,
            input_liq,
            input_melt_TAS=input_melt_TAS,
            melt_TAS_source=melt_TAS_source,
        )
        prediction = self.decision()
        if plot:
            self._plot_prediction_violin()
        return prediction

    def _plot_prediction_violin(self):
        """Create and store a violin plot for plottable model predictions."""
        from aims4pt.visualization.thermobarometry_plot import violin_plot

        model_uncertainty_dict = {
            getattr(model, "model_name", type(model).__name__): getattr(
                model, "uncertainty", None
            )
            for model in self.model_list
        }
        model_name_list = [
            getattr(model, "model_name", type(model).__name__) for model in self.model_list
        ]
        available_columns = [
            model_name
            for model_name in model_name_list
            if model_name in self.prediction_df
        ]
        if not available_columns:
            self.violin_plot_fig = None
            self.violin_plot_ax = None
            self.violin_plot_artists = None
            return None

        numeric_predictions = self.prediction_df[available_columns].apply(
            lambda column: pd.to_numeric(column, errors="coerce")
        )
        plottable_columns = [
            column
            for column in numeric_predictions.columns
            if numeric_predictions[column].dropna().shape[0] >= 2
            and numeric_predictions[column].dropna().nunique() >= 2
        ]
        if not plottable_columns:
            self.violin_plot_fig = None
            self.violin_plot_ax = None
            self.violin_plot_artists = None
            return None

        plottable_uncertainty = {
            model_name: model_uncertainty_dict.get(model_name)
            for model_name in plottable_columns
        }
        fig, ax, artists = violin_plot(
            numeric_predictions[plottable_columns].copy(),
            self.T_P,
            plottable_columns,
            model_uncertainty=plottable_uncertainty,
        )
        self.violin_plot_fig = fig
        self.violin_plot_ax = ax
        self.violin_plot_artists = artists
        return fig, ax, artists

    # ------------------------------------------------------------------
    # Custom serialization logic
    # ------------------------------------------------------------------
    def __getstate__(self):
        """Prepare object state for pickling/joblib serialization.

        Returns
        -------
        dict
            Serializable state dictionary. ``model_list`` is intentionally replaced
            with ``None`` to avoid serializing potentially non-serializable model
            objects (for example wrapped R models).

        Raises
        ------
        RuntimeError
            If called before ``calculation`` has been executed.

        Notes
        -----
        Serialized objects are intended for result inspection, not for re-running
        predictions with the original model objects.
        """
        if self.prediction_df is None:
            raise RuntimeError(
                "Serialization is only available after calculation."
            )

        state = self.__dict__.copy()
        # model_list may contain R models or other non-serializable
        # resources; we drop it from the serialized state.
        state["model_list"] = None

        return state

    def __setstate__(self, state):
        """Restore object state from pickle/joblib.

        Parameters
        ----------
        state : dict
            State dictionary produced by ``__getstate__``.

        Returns
        -------
        None

        Notes
        -----
        ``model_list`` is forced to ``None`` after restore, so the deserialized
        object is intended for reviewing saved outputs rather than running new
        predictions.
        """
        self.__dict__.update(state)
        self.model_list = None


