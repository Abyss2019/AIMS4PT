"""WandB24 melt thermobarometry wrapper.
Weber, G., & Blundy, J. (2024). A Machine Learning-Based Thermobarometer for Magmatic Liquids. Journal of Petrology, 65(4), egae020. https://doi.org/10.1093/petrology/egae020
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import aims4pt.model_tools.data.WandB24 as wandb24_data
from aims4pt.model_tools.ModelManager import ModelManager
from aims4pt.model_tools.rpy_tools.r_env import (
    configure_r_environment,
    format_rpy2_setup_error,
)
from aims4pt.utils import normalize_column_names

configure_r_environment()

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import default_converter, pandas2ri
    from rpy2.robjects.conversion import localconverter
except Exception as exc:
    raise RuntimeError(format_rpy2_setup_error(exc)) from exc

_R_HELPERS_INITIALIZED = False


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def _format_melt_input(df: pd.DataFrame, melt_names: list[str]) -> pd.DataFrame:
    formatted = normalize_column_names(
        df,
        standard_names_list=melt_names,
        missing_fill=0,
        drop_missing=False,
        report_info=False,
    )
    formatted = _coerce_numeric(formatted).fillna(0.0)
    oxide_cols = [col for col in melt_names if col != "H2O_melt"]
    return formatted[oxide_cols]


def _rename_predictors_for_r(
    df: pd.DataFrame,
    predictor_name_map: dict[str, str],
) -> pd.DataFrame:
    renamed = df.rename(columns=predictor_name_map).copy()
    return renamed.loc[:, list(predictor_name_map.values())]


def _ensure_r_helpers() -> None:
    global _R_HELPERS_INITIALIZED
    if _R_HELPERS_INITIALIZED:
        return

    robjects.r(
        r'''
        options(repos = c(CRAN = "https://cloud.r-project.org"))

        wandb24_required_packages <- c("ranger", "matrixStats")
        wandb24_missing_packages <- wandb24_required_packages[
          !vapply(wandb24_required_packages, requireNamespace, logical(1), quietly = TRUE)
        ]
        if (length(wandb24_missing_packages) > 0) {
          install.packages(wandb24_missing_packages, dependencies = TRUE)
        }

        suppressPackageStartupMessages(library(ranger))
        suppressPackageStartupMessages(library(matrixStats))

        wandb24_train_two_stage <- function(training_data, target_column, scalefac) {
          set.seed(17)

          training <- as.data.frame(training_data)
          colnames(training) <- make.names(colnames(training))
          target_column <- make.names(target_column)

          predictor_cols <- c(
            "ol", "opx", "cpx", "plag", "amph", "ox", "bt", "ksp", "gt", "qz",
            "SiO2.n.", "TiO2.n.", "Al2O3.n.", "FeO.n.", "MgO.n.", "CaO.n.", "Na2O.n.", "K2O.n."
          )
          train_cols <- c(target_column, predictor_cols)
          missing_cols <- setdiff(train_cols, colnames(training))
          if (length(missing_cols) > 0) {
            stop(paste("Training data are missing required columns:", paste(missing_cols, collapse = ", ")))
          }

          y.data_train <- training[, train_cols, drop = FALSE]
          y_train <- y.data_train[[target_column]]

          model1 <- ranger::ranger(
            formula = stats::as.formula(paste(target_column, "~ .")),
            data = y.data_train,
            num.trees = 500,
            mtry = NULL,
            importance = "none",
            min.node.size = NULL,
            max.depth = NULL,
            replace = FALSE,
            sample.fraction = 1,
            num.random.splits = 10,
            splitrule = "extratrees",
            keep.inbag = TRUE
          )

          train_predictions <- predict(model1, data = y.data_train)$predictions
          train_residuals <- y_train - train_predictions
          train_data_with_residuals <- cbind(y.data_train, Residuals = train_residuals)

          model2 <- ranger::ranger(
            formula = Residuals ~ .,
            data = train_data_with_residuals,
            num.trees = 500,
            mtry = NULL,
            importance = "none",
            min.node.size = NULL,
            max.depth = NULL,
            replace = FALSE,
            sample.fraction = 1,
            num.random.splits = 10,
            splitrule = "extratrees",
            keep.inbag = TRUE
          )

          list(
            primary_model = model1,
            residual_model = model2,
            target_column = target_column,
            scalefac = as.numeric(scalefac)
          )
        }

        wandb24_predict_details <- function(model_bundle, newdata) {
          initial_predictions <- predict(model_bundle$primary_model, data = newdata)$predictions
          initial_predictions_votes <- predict(
            model_bundle$primary_model,
            data = newdata,
            type = "response",
            predict.all = TRUE
          )
          initial_predictions_sd <- matrixStats::rowSds(as.matrix(initial_predictions_votes$predictions))

          residual_input <- newdata
          residual_input[[model_bundle$target_column]] <- initial_predictions
          residual_predictions <- predict(model_bundle$residual_model, data = residual_input)$predictions
          predictions <- initial_predictions + model_bundle$scalefac * residual_predictions

          data.frame(
            initial_prediction = as.numeric(initial_predictions),
            initial_prediction_sd = as.numeric(initial_predictions_sd),
            residual_prediction = as.numeric(residual_predictions),
            prediction = as.numeric(predictions)
          )
        }
        '''
    )
    _R_HELPERS_INITIALIZED = True


@dataclass
class _WandB24TwoStageModel:
    model_bundle: Any
    predictor_name_map: dict[str, str]

    def predict_details(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        _ensure_r_helpers()
        feature_data_r = _rename_predictors_for_r(feature_data, self.predictor_name_map)
        predict_details = robjects.globalenv["wandb24_predict_details"]
        with localconverter(default_converter + pandas2ri.converter):
            feature_data_r = pandas2ri.py2rpy(feature_data_r)

        prediction_r = predict_details(self.model_bundle, feature_data_r)

        with localconverter(default_converter + pandas2ri.converter):
            prediction_df = robjects.conversion.rpy2py(prediction_r)

        prediction_df = pd.DataFrame(prediction_df).copy()
        prediction_df.index = feature_data.index
        return prediction_df[
            ["initial_prediction", "initial_prediction_sd", "residual_prediction", "prediction"]
        ]

    def predict(self, feature_data: pd.DataFrame) -> np.ndarray:
        return self.predict_details(feature_data)["prediction"].to_numpy()


class WandB24_melt(ModelManager):
    """
    Weber and Blundy (2024) melt thermobarometer for predicting temperature or pressure.

    This class mirrors the workflow in the bundled `MagmaTAB_Tcalc.R` and
    `MagmaTAB_Pcalc.R` scripts, but exposes it through the same Python-side
    interface used by the other thermobarometers in this project.

    By default, the model is trained every time the class is instantiated from
    `data/WandB24/data_set/REV_MagmaTAB.xlsx`, sheet `workbook`. To train the
    same two-stage model on in-memory dataframes, use
    `WandB24_melt.train_model(training_melt=..., training_phase=..., T_P=...)`.
    Training uses the same two-stage `ranger` workflow as the bundled R scripts:

    1. Fit a primary `ranger` regressor to predict `T_C` or `P_kbar`.
    2. Fit a second `ranger` regressor to predict residuals from the first model.
    3. Combine both predictions as
       `prediction = initial_prediction + scalefac * residual_prediction`
       with `scalefac=2` by default.

    Inputs
    ------
    `predict(X_melt, X_phase=None)` expects:

    - melt compositions in `X_melt`
    - phase-presence indicators either in a separate `X_phase` dataframe or
      embedded directly inside `X_melt`

    Recognized phase columns are:
    `ol`, `opx`, `cpx`, `plag`, `amph`, `ox`, `bt`, `ksp`, `gt`, `qz`

    Melt compositions are accepted as provided after column-name
    standardization. `H2O` can be present in the input, but it is not used as
    a predictive feature.

    Parameters
    ----------
    T_P : str
        `"T"` for temperature prediction or `"P"` for pressure prediction.
    NumbMin : int
        Minimum number of stable phases required for a row to be evaluated.
        Must satisfy `0 <= NumbMin <= 5`, matching the intended usage of the
        published web-app workflow.
    filter : float, default 0
        Quantile threshold applied to the standard deviation of tree votes from
        the first-stage model.
        - `0`: keep all valid rows
        - `0 < filter <= 1`: keep only rows with vote SD below the selected quantile
    comments : str, optional
        Free-form metadata stored on the model instance.

    Training with custom data
    -------------------------
    `train_model()` accepts separate `training_melt` and `training_phase`
    dataframes. `training_melt` contains melt composition columns plus `T_C`
    and `P_kbar`; `training_phase` contains phase-presence columns. If
    `number_phases` is present it is used for the `NumbMin` filter; otherwise
    phase counts are calculated from the formatted phase columns.

    Notes
    -----
    - Rows with phase counts below `NumbMin` return `NaN`.
    - When `filter > 0`, rows failing the uncertainty filter also return `NaN`.
    - Detailed per-row diagnostics are stored in `last_prediction_diagnostics_`
      after each call to `predict()`.
    - Feature importance is not exposed because the reference R scripts train
      `ranger` models with `importance = "none"`.
    """

    phase_names = ["ol", "opx", "cpx", "plag", "amph", "ox", "bt", "ksp", "gt", "qz"]
    melt_names = [
        "SiO2_melt",
        "TiO2_melt",
        "Al2O3_melt",
        "FeO_melt",
        "MgO_melt",
        "CaO_melt",
        "Na2O_melt",
        "K2O_melt",
        "H2O_melt",
    ]
    anhydrous_melt_names = [
        "SiO2_melt",
        "TiO2_melt",
        "Al2O3_melt",
        "FeO_melt",
        "MgO_melt",
        "CaO_melt",
        "Na2O_melt",
        "K2O_melt",
    ]
    training_melt_columns = {
        "SiO2(n)": "SiO2_melt",
        "TiO2(n)": "TiO2_melt",
        "Al2O3(n)": "Al2O3_melt",
        "FeO(n)": "FeO_melt",
        "MgO(n)": "MgO_melt",
        "CaO(n)": "CaO_melt",
        "Na2O(n)": "Na2O_melt",
        "K2O(n)": "K2O_melt",
    }
    r_predictor_name_map = {
        "ol": "ol",
        "opx": "opx",
        "cpx": "cpx",
        "plag": "plag",
        "amph": "amph",
        "ox": "ox",
        "bt": "bt",
        "ksp": "ksp",
        "gt": "gt",
        "qz": "qz",
        "SiO2_melt": "SiO2.n.",
        "TiO2_melt": "TiO2.n.",
        "Al2O3_melt": "Al2O3.n.",
        "FeO_melt": "FeO.n.",
        "MgO_melt": "MgO.n.",
        "CaO_melt": "CaO.n.",
        "Na2O_melt": "Na2O.n.",
        "K2O_melt": "K2O.n.",
    }

    def __init__(
        self,
        T_P: str,
        NumbMin=0,
        filter=0,
        comments: Optional[str] = None,
    ):
        """Initialise, train, and prepare a WandB24 melt thermobarometer."""
        super().__init__(comments=comments)
        self._configure_model(T_P=T_P, NumbMin=NumbMin, filter=filter, scalefac=2.0)
        self._train_model()

    def _configure_model(
        self,
        T_P: str,
        NumbMin=0,
        filter=0,
        scalefac: float = 2.0,
    ) -> None:
        """Validate model options and populate shared instance attributes."""
        if T_P not in {"T", "P"}:
            raise ValueError("T_P must be 'T' or 'P'.")

        self.NumbMin = int(NumbMin)
        if self.NumbMin < 0 or self.NumbMin > 5:
            raise ValueError("NumbMin must be an integer between 0 and 5.")

        self.filter = float(filter)
        if self.filter < 0 or self.filter > 1:
            raise ValueError("filter must be between 0 and 1.")

        self.model_name = "Weber & Blundy, 2024"
        self.T_P = T_P
        self.input_kind = "phase_melt"
        self.require_water = False
        self.if_support_hydrous = True
        self.scalefac = float(scalefac)

        self.prediction_column_name = "T_C" if T_P == "T" else "P_kbar"
        self.standard_columns = self.phase_names + self.anhydrous_melt_names

        self.training_data_path = Path(files(wandb24_data) / "data_set" / "REV_MagmaTAB.xlsx")

        self.last_prediction_diagnostics_: Optional[pd.DataFrame] = None
        self.feature_importance_df: Optional[pd.DataFrame] = None

    @classmethod
    def train_model(
        cls,
        training_melt: pd.DataFrame,
        training_phase: pd.DataFrame,
        T_P: str,
        NumbMin: int = 0,
        filter: float = 0,
        scalefac: float = 2.0,
        comments: Optional[str] = None,
    ) -> "WandB24_melt":
        """
        Train a WandB24 melt thermobarometer from an in-memory dataframe.

        Parameters
        ----------
        training_melt : pd.DataFrame
            Melt training rows containing composition columns accepted by
            `format_input()` plus both target columns, `T_C` and `P_kbar`.
            The dataframe is copied internally and is not modified.
        training_phase : pd.DataFrame
            Phase training rows containing the ten phase-presence columns:
            `ol`, `opx`, `cpx`, `plag`, `amph`, `ox`, `bt`, `ksp`, `gt`, and
            `qz`. A `number_phases` column is optional.
        T_P : {"T", "P"}
            `"T"` trains a temperature model using `T_C`; `"P"` trains a
            pressure model using `P_kbar`.
        NumbMin : int, default 0
            Minimum number of stable phases required for training rows and
            later predictions. If `training_phase` has a `number_phases`
            column, that column is used for training-row filtering; otherwise
            phase counts are calculated from the formatted phase columns.
        filter : float, default 0
            Prediction-time quantile filter for first-stage tree-vote standard
            deviations. Training itself uses all rows that pass `NumbMin` and
            have a finite target value.
        scalefac : float, default 2.0
            Multiplier applied to second-stage residual predictions.
        comments : str, optional
            Free-form metadata stored on the returned model instance.

        Returns
        -------
        WandB24_melt
            A trained model instance ready for `predict()`.

        Raises
        ------
        TypeError
            If `training_melt` or `training_phase` is not a pandas DataFrame.
        ValueError
            If required columns are missing, model options are invalid, or no
            valid training rows remain after filtering.

        Examples
        --------
        >>> model = WandB24_melt.train_model(training_melt, training_phase, T_P="T", NumbMin=2)
        >>> predictions = model.predict(training_melt, training_phase)
        """
        if not isinstance(training_melt, pd.DataFrame):
            raise TypeError("training_melt must be a pandas DataFrame.")
        if not isinstance(training_phase, pd.DataFrame):
            raise TypeError("training_phase must be a pandas DataFrame.")

        instance = cls.__new__(cls)
        ModelManager.__init__(instance, comments=comments)
        instance._configure_model(T_P=T_P, NumbMin=NumbMin, filter=filter, scalefac=scalefac)
        instance.training_data_path = None
        instance._train_model(training_melt=training_melt, training_phase=training_phase)
        return instance

    def __str__(self):
        text = f"""
        ModelManager: {self.__class__.__name__}
        Model name: {self.model_name}
        interested parameter: {self.T_P}; phase-melt
        require_water: {self.require_water}
        uncertainty: {self.uncertainty} {"C" if self.T_P == "T" else "kbar"}
        comment: {self.comments}
        """
        return text

    def _load_training_sheet(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load the bundled workbook and prepare it for model training."""
        training = pd.read_excel(self.training_data_path, sheet_name="workbook")
        phase_cols = self.phase_names + [col for col in ["number_phases"] if col in training.columns]
        training_phase = training[phase_cols].copy()
        training_melt = training.drop(columns=self.phase_names, errors="ignore").copy()
        return self._prepare_training_data(training_melt, training_phase)

    def _validate_training_columns(
        self,
        training_melt: pd.DataFrame,
        training_phase: pd.DataFrame,
    ) -> None:
        """Raise a clear error when custom training data cannot be formatted."""
        target_columns = ["T_C", "P_kbar"]
        missing_target_cols = [col for col in target_columns if col not in training_melt.columns]
        if missing_target_cols:
            raise ValueError(
                "training_melt must include target columns: "
                f"{', '.join(missing_target_cols)}."
            )

        missing_phase_cols = [col for col in self.phase_names if col not in training_phase.columns]
        if missing_phase_cols:
            raise ValueError(
                "training_phase is missing required phase columns: "
                f"{', '.join(missing_phase_cols)}."
            )

        formatted_melt = normalize_column_names(
            training_melt,
            standard_names_list=self.melt_names,
            missing_fill=np.nan,
            drop_missing=False,
            report_info=False,
        )
        missing_melt_cols = [
            col for col in self.anhydrous_melt_names if formatted_melt[col].isna().all()
        ]
        if missing_melt_cols:
            raise ValueError(
                "training_melt is missing required melt composition columns "
                "that can be normalized by format_input(): "
                f"{', '.join(missing_melt_cols)}."
            )

    def _prepare_training_data(
        self,
        training_melt: pd.DataFrame,
        training_phase: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare raw melt and phase training data using the prediction formatter."""
        if not isinstance(training_melt, pd.DataFrame):
            raise TypeError("training_melt must be a pandas DataFrame.")
        if not isinstance(training_phase, pd.DataFrame):
            raise TypeError("training_phase must be a pandas DataFrame.")
        if training_melt.empty:
            raise ValueError("training_melt must contain at least one row.")
        if training_phase.empty:
            raise ValueError("training_phase must contain at least one row.")
        if not training_melt.index.equals(training_phase.index):
            raise ValueError("training_melt and training_phase must have matching indexes.")

        training_melt = training_melt.copy()
        training_phase = training_phase.copy()
        self._validate_training_columns(training_melt, training_phase)

        target = pd.to_numeric(training_melt[self.prediction_column_name], errors="coerce")
        features_all = self.format_input(training_melt, training_phase)

        if "number_phases" in training_phase.columns:
            phase_counts = pd.to_numeric(training_phase["number_phases"], errors="coerce").fillna(0.0)
        elif "number_phases" in training_melt.columns:
            phase_counts = pd.to_numeric(training_melt["number_phases"], errors="coerce").fillna(0.0)
        else:
            phase_counts = features_all[self.phase_names].sum(axis=1)

        valid_mask = (phase_counts >= self.NumbMin) & target.notna()
        features = features_all.loc[valid_mask, self.standard_columns].copy()
        y_train = target.loc[valid_mask].copy()

        if features.empty:
            raise ValueError("No valid training rows remain after applying NumbMin and target filtering.")

        self.X_phase_training = features[self.phase_names].copy()
        if "number_phases" in training_phase.columns:
            self.X_phase_training["number_phases"] = phase_counts.loc[valid_mask].to_numpy()

        self.X_melt_training = features[self.anhydrous_melt_names].copy()
        for target_column in ["T_C", "P_kbar"]:
            self.X_melt_training[target_column] = pd.to_numeric(
                training_melt.loc[valid_mask, target_column],
                errors="coerce",
            )
        self.X_phase_test = None
        self.X_melt_test = None
        self.X_phase_all = self.X_phase_training.copy()
        self.X_melt_all = self.X_melt_training.copy()
        self.y_min = float(y_train.min())
        self.y_max = float(y_train.max())
        self.y_min_95 = float(y_train.quantile(0.025))
        self.y_max_95 = float(y_train.quantile(0.975))

        return features, y_train

    def _format_training_data_for_r(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> pd.DataFrame:
        """Return a training frame with R-compatible predictor names."""
        training_for_r = _rename_predictors_for_r(X_train, self.r_predictor_name_map)
        training_for_r.insert(0, self.prediction_column_name, y_train.to_numpy(dtype=float))
        return training_for_r

    def _train_model(
        self,
        training_melt: Optional[pd.DataFrame] = None,
        training_phase: Optional[pd.DataFrame] = None,
    ) -> None:
        if training_melt is None and training_phase is None:
            X_train, y_train = self._load_training_sheet()
        elif training_melt is not None and training_phase is not None:
            X_train, y_train = self._prepare_training_data(training_melt, training_phase)
        else:
            raise ValueError("training_melt and training_phase must be provided together.")

        _ensure_r_helpers()

        training_for_r = self._format_training_data_for_r(X_train, y_train)
        train_two_stage = robjects.globalenv["wandb24_train_two_stage"]
        with localconverter(default_converter + pandas2ri.converter):
            training_for_r = pandas2ri.py2rpy(training_for_r)

        model_bundle = train_two_stage(
            training_for_r,
            self.prediction_column_name,
            float(self.scalefac),
        )

        self.model = _WandB24TwoStageModel(
            model_bundle=model_bundle,
            predictor_name_map=self.r_predictor_name_map,
        )

        train_predictions = self.model.predict(X_train)
        self.uncertainty = float(np.sqrt(np.mean((train_predictions - y_train.to_numpy()) ** 2)))
        self.feature_importance_df = None

    def _format_phase_input(
        self,
        X_melt: pd.DataFrame,
        X_phase: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if X_phase is None:
            embedded_phase = [col for col in self.phase_names if col in X_melt.columns]
            if not embedded_phase:
                raise ValueError(
                    "Phase presence data are required. Pass X_phase or include phase columns in X_melt."
            )
            phase_input = X_melt
        else:
            if not X_phase.index.equals(X_melt.index):
                raise ValueError(
                    "X_phase index must match X_melt index exactly. "
                    "Align both dataframes before prediction, or reset both indexes "
                    "after verifying that their row order is identical."
                )
            phase_input = X_phase

        phase_df = phase_input.reindex(index=X_melt.index, columns=self.phase_names)
        phase_df = _coerce_numeric(phase_df).fillna(0.0)
        phase_df = (phase_df > 0).astype(float)
        return phase_df

    def format_input(
        self,
        X_melt: pd.DataFrame,
        X_phase: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Standardize phase and melt inputs into the feature matrix expected by the model.

        Parameters
        ----------
        X_melt : pd.DataFrame
            Melt compositions. Columns are normalized with the project's
            `normalize_column_names()` helper, so common oxide naming variants
            such as `SiO2`, `SiO2_melt`, or `SiO2 liquid` are accepted.
            If `X_phase` is omitted, this dataframe may also include phase columns.
        X_phase : pd.DataFrame, optional
            Separate dataframe containing the ten binary or numeric phase
            presence columns. Any value greater than zero is treated as phase
            present; all other values are treated as absent.

        Returns
        -------
        pd.DataFrame
            A dataframe ordered as `self.standard_columns`, ready for prediction.
        """
        X_melt = X_melt.copy()
        phase_df = self._format_phase_input(X_melt, X_phase)
        melt_input = X_melt.drop(columns=[col for col in self.phase_names if col in X_melt.columns], errors="ignore")
        melt_df = _format_melt_input(melt_input, self.melt_names)
        melt_df.index = X_melt.index
        return pd.concat([phase_df, melt_df], axis=1).reindex(columns=self.standard_columns)

    def predict(
        self,
        X_melt: pd.DataFrame,
        X_phase: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Predict melt temperature or pressure for each input row.

        Parameters
        ----------
        X_melt : pd.DataFrame
            Melt compositions. May optionally also contain the phase columns.
        X_phase : pd.DataFrame, optional
            Separate phase-presence dataframe. Use this when phase information
            is not stored inside `X_melt`.

        Returns
        -------
        pd.Series
            Predicted `T_C` or `P_kbar`, indexed to match `X_melt`.
            Rows that fail the `NumbMin` requirement, or the optional vote-SD
            filter, are returned as `NaN`.

        Side Effects
        ------------
        Stores a diagnostics dataframe on `self.last_prediction_diagnostics_`
        with the following columns when available:

        - `phase_count`
        - `passes_numbmin`
        - `passes_filter`
        - `filter_threshold`
        - `initial_prediction`
        - `initial_prediction_sd`
        - `residual_prediction`
        - `prediction`
        """
        formatted = self.format_input(X_melt, X_phase)
        phase_counts = formatted[self.phase_names].sum(axis=1)
        valid_phase_mask = phase_counts >= self.NumbMin

        result = pd.Series(np.nan, index=formatted.index, name=self.prediction_column_name, dtype=float)
        diagnostics = pd.DataFrame(index=formatted.index)
        diagnostics["phase_count"] = phase_counts
        diagnostics["passes_numbmin"] = valid_phase_mask
        diagnostics["passes_filter"] = False
        diagnostics["filter_threshold"] = np.nan

        if valid_phase_mask.any():
            valid_input = formatted.loc[valid_phase_mask, self.standard_columns]
            valid_diagnostics = self.model.predict_details(valid_input)

            if self.filter > 0:
                threshold = float(np.quantile(valid_diagnostics["initial_prediction_sd"], self.filter))
                keep_mask = valid_diagnostics["initial_prediction_sd"] <= threshold
            else:
                threshold = np.inf
                keep_mask = pd.Series(True, index=valid_diagnostics.index)

            result.loc[keep_mask.index[keep_mask]] = valid_diagnostics.loc[keep_mask, "prediction"]

            diagnostics.loc[valid_diagnostics.index, "initial_prediction"] = valid_diagnostics["initial_prediction"]
            diagnostics.loc[valid_diagnostics.index, "initial_prediction_sd"] = valid_diagnostics["initial_prediction_sd"]
            diagnostics.loc[valid_diagnostics.index, "residual_prediction"] = valid_diagnostics["residual_prediction"]
            diagnostics.loc[valid_diagnostics.index, "prediction"] = valid_diagnostics["prediction"]
            diagnostics.loc[valid_diagnostics.index, "passes_filter"] = keep_mask
            diagnostics.loc[valid_diagnostics.index, "filter_threshold"] = threshold

        self.last_prediction_diagnostics_ = diagnostics
        return result

    def get_feature_importance_df(self) -> pd.DataFrame:
        if self.feature_importance_df is None:
            raise ValueError(
                "Feature importance is not available because the reference ranger models use importance = 'none'."
            )
        return self.feature_importance_df.copy()

    def _format_shap_input(
        self,
        test_X_phase: pd.DataFrame,
        test_X_melt: Optional[pd.DataFrame] = None,
        input_name: str = "test_X_phase",
    ) -> pd.DataFrame:
        if test_X_melt is None:
            self._validate_embedded_melt_columns(test_X_phase, input_name=input_name)
            formatted = self.format_input(test_X_phase)
        else:
            formatted = self.format_input(test_X_melt, test_X_phase)
        return formatted.reindex(columns=self.standard_columns).fillna(0.0)

    def _validate_embedded_melt_columns(self, data: pd.DataFrame, input_name: str) -> None:
        """Require melt compositions when SHAP receives a single combined dataframe."""
        formatted_melt = normalize_column_names(
            data,
            standard_names_list=self.melt_names,
            missing_fill=np.nan,
            drop_missing=False,
            report_info=False,
        )
        missing_melt_cols = [
            col for col in self.anhydrous_melt_names if formatted_melt[col].isna().all()
        ]
        if missing_melt_cols:
            raise ValueError(
                f"{input_name} must include melt composition columns when the separate "
                "melt dataframe is not provided. Missing or all-NaN columns after "
                f"normalization: {', '.join(missing_melt_cols)}."
            )

    def shap_calculation(
        self,
        test_X_phase: pd.DataFrame,
        test_X_melt: Optional[pd.DataFrame] = None,
        background_data: Optional[pd.DataFrame] = None,
        bg_melt: Optional[pd.DataFrame] = None,
        sampling_test: Optional[int] = None,
        sampling_bg: Optional[int] = None,
        shap_sort: bool = True,
        package_predict_func: bool = True,
    ):
        """
        Calculate SHAP values for the WandB24 melt thermobarometer.

        Unlike the generic two-phase implementation in ModelManager, this
        model expects phase-presence columns plus melt compositions. If
        `test_X_melt` is provided, `test_X_phase` is treated as the phase
        dataframe; otherwise `test_X_phase` must contain both phase and
        melt columns.
        """
        import shap

        from aims4pt.model_tools.model_utils import random_sample_reduce_data

        shap.initjs()

        shap_features = self._format_shap_input(
            test_X_phase,
            test_X_melt,
            input_name="test_X_phase",
        )
        if background_data is None or background_data.empty:
            background_features = pd.concat(
                [
                    self.X_phase_training.reindex(columns=self.phase_names),
                    self.X_melt_training.reindex(columns=self.anhydrous_melt_names),
                ],
                axis=1,
            ).reindex(columns=self.standard_columns)
        else:
            background_features = self._format_shap_input(
                background_data,
                bg_melt,
                input_name="background_data",
            )

        if sampling_test is not None:
            shap_features = random_sample_reduce_data(shap_features, sampling_test, 42)
        if sampling_bg is not None:
            background_features = random_sample_reduce_data(background_features, sampling_bg, 42)

        shap_features = shap_features.reindex(columns=self.standard_columns).fillna(0.0)
        background_features = background_features.reindex(columns=self.standard_columns).fillna(0.0)

        background_prediction = self.predict(background_features)
        baseline_value = float(np.nanmean(background_prediction.to_numpy()))
        if not np.isfinite(baseline_value):
            baseline_value = 0.0

        def _as_feature_frame(data) -> pd.DataFrame:
            if isinstance(data, pd.DataFrame):
                return data.reindex(columns=self.standard_columns).fillna(0.0)
            return pd.DataFrame(data, columns=self.standard_columns).fillna(0.0)

        def _predict_for_shap(data):
            shap_feature_frame = _as_feature_frame(data)
            prediction = self.predict(shap_feature_frame)
            prediction_values = prediction.to_numpy(dtype=float)
            if package_predict_func:
                prediction_values = np.nan_to_num(prediction_values, nan=baseline_value)
            return prediction_values

        masker = shap.maskers.Independent(background_features)
        explainer = shap.Explainer(_predict_for_shap, masker=masker)
        shap_values = explainer(shap_features)

        shap_df = pd.DataFrame(
            shap_values.values,
            columns=shap_features.columns,
            index=shap_features.index[: len(shap_values.values)],
        )

        feature_importance = (
            shap_df.replace([np.inf, -np.inf], np.nan)
            .abs()
            .mean(skipna=True)
            .sort_values(ascending=False)
        )

        if shap_sort:
            feature_order = feature_importance.index.tolist()
        else:
            feature_order = self.standard_columns

        shap_df = shap_df[feature_order]
        plot_features = shap_features[feature_order]

        if not shap_df.empty:
            shap.summary_plot(
                shap_df.values,
                features=plot_features,
                feature_names=feature_order,
                sort=shap_sort,
            )

        importance_sum = feature_importance.sum()
        if importance_sum and np.isfinite(importance_sum):
            feature_importance_percent = feature_importance / importance_sum
        else:
            feature_importance_percent = feature_importance * 0.0

        feature_importance_df = pd.DataFrame(
            {"Importance": feature_importance_percent, "Feature": feature_importance.index}
        )

        self.shap_df = shap_df
        self.feature_importance_df = feature_importance_df
        self.shap_values = shap_values
        print(feature_importance_df.head())
        return shap_df, feature_importance_df, shap_values


def _run_simple_test() -> None:
    """Run a small smoke test using the first few valid rows from the training workbook."""
    data_path = Path(files(wandb24_data) / "data_set" / "REV_MagmaTAB.xlsx")
    sample = pd.read_excel(data_path, sheet_name="workbook")
    sample = sample.loc[sample["number_phases"].fillna(0) >= 2].head(5).copy()

    if sample.empty:
        raise RuntimeError("No smoke-test rows found in the workbook sheet.")

    phase_cols = WandB24_melt.phase_names
    melt_cols = [
        "SiO2 liquid",
        "TiO2 liquid",
        "Al2O3 liquid",
        "FeO liquid",
        "MgO liquid",
        "CaO liquid",
        "Na2O liquid",
        "K2O liquid",
        "H2O_liquid",
    ]


    X_phase = sample[phase_cols].copy()
    training_phase = sample[phase_cols + ["number_phases"]].copy()
    training_melt = sample[melt_cols + ["T_C", "P_kbar"]].copy()
    X_melt_only = sample[melt_cols].copy()
    X_melt_with_phase = sample[phase_cols + melt_cols].copy()

    t_model = WandB24_melt("T", NumbMin=2, filter=0)
    p_model = WandB24_melt("P", NumbMin=2, filter=0)
    custom_p_model = WandB24_melt.train_model(
        training_melt=training_melt,
        training_phase=training_phase,
        T_P="P",
        NumbMin=2,
        filter=0,
    )

    t_pred = t_model.predict(X_melt_with_phase)
    p_pred = p_model.predict(X_melt_only, X_phase=X_phase)
    custom_p_pred = custom_p_model.predict(X_melt_only, X_phase=X_phase)

    if len(t_pred) != len(sample) or len(p_pred) != len(sample) or len(custom_p_pred) != len(sample):
        raise AssertionError("Prediction length does not match the sample length.")
    if t_pred.isna().all():
        raise AssertionError("Temperature smoke test returned all-NaN predictions.")
    if p_pred.isna().all():
        raise AssertionError("Pressure smoke test returned all-NaN predictions.")
    if custom_p_pred.isna().all():
        raise AssertionError("Custom pressure smoke test returned all-NaN predictions.")

    result = pd.DataFrame(
        {
            "T_C_obs": sample["T_C"].to_numpy(),
            "T_C_pred": t_pred.to_numpy(),
            "P_kbar_obs": sample["P_kbar"].to_numpy(),
            "P_kbar_pred": p_pred.to_numpy(),
            "P_kbar_custom_pred": custom_p_pred.to_numpy(),
        },
        index=sample.index,
    )
    print("WandB24_melt smoke test passed.")
    print(result.round(3).to_string())


if __name__ == "__main__":
    _run_simple_test()
