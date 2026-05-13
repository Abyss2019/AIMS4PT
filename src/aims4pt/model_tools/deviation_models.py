"""Deviation-model utilities for thermobarometric calibration workflows."""

from typing import Dict, Any, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
import pandas as pd

from sklearn.experimental import enable_halving_search_cv  
from sklearn.model_selection import HalvingRandomSearchCV

def mape_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return np.mean(np.abs((y - y_pred) / y)) * 100.0

class KernelDensityRegressor(BaseEstimator, RegressorMixin):
    """
    Simple Nadaraya–Watson kernel regression with a Gaussian kernel.

    Parameters
    ----------
    bandwidth : float
        Bandwidth for the Gaussian kernel. Must be > 0.
    min_weight : float
        Minimum weight sum to avoid division by zero.
    """

    def __init__(self, bandwidth: float = 1.0, min_weight: float = 1e-12):
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        self.bandwidth = bandwidth
        self.min_weight = min_weight

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        self._X = X
        self._y = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_X"):
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        dists = pairwise_distances(X, self._X, metric="euclidean")
        weights = np.exp(-0.5 * (dists / self.bandwidth) ** 2)
        weight_sums = weights.sum(axis=1)
        weight_sums = np.maximum(weight_sums, self.min_weight)
        preds = weights @ self._y
        return preds / weight_sums


# ----------------------------------------------------------------------
# Estimator builders (each model type is fully separated)
# ----------------------------------------------------------------------
def build_knn_regressor() -> Pipeline:
    """
    Build a Pipeline with StandardScaler and KNeighborsRegressor.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", KNeighborsRegressor()),
        ]
    )


def build_rf_regressor(random_state: int = 42) -> RandomForestRegressor:
    """
    Build a base RandomForestRegressor.
    """
    return RandomForestRegressor(random_state=random_state)


def build_extra_trees_regressor(random_state: int = 42) -> ExtraTreesRegressor:
    """
    Build a base ExtraTreesRegressor.
    """
    return ExtraTreesRegressor(random_state=random_state)


# def build_extra_trees_regressor(random_state: int = 42) -> ExtraTreesQuantileRegressor:
#     """
#     Build an ExtraTreesQuantileRegressor for deviation modeling.
#     """
#     return ExtraTreesQuantileRegressor(random_state=random_state)


def build_xgb_regressor(random_state: int = 42) -> "XGBRegressor":
    """
    Build a base XGBRegressor.

    Raises
    ------
    ImportError
        If XGBoost is not installed.
    """
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        tree_method="hist",
    )



def build_svr_regressor() -> Pipeline:
    """
    Build a Pipeline with StandardScaler and SVR (RBF kernel).
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", SVR(kernel="rbf")),
        ]
    )



def build_kr_regressor(bandwidth: float = 1.0) -> Pipeline:
    """
    Build a Pipeline with StandardScaler and KernelDensityRegressor.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", KernelDensityRegressor(bandwidth=bandwidth)),
        ]
    )


# ----------------------------------------------------------------------
# Default parameter spaces (each model type separated)
# ----------------------------------------------------------------------
def knn_param_grid() -> Dict[str, Any]:
    """
    Default hyperparameter search space for Pipeline(kNN) with scaling.

    Values are sampled uniformly when using random search.
    """
    tight_neighbors = list(range(3, 31))
    extended_neighbors = [35, 40, 45, 50, 60, 75, 90, 120]
    neighbor_values = tight_neighbors + extended_neighbors
    return {
        "regressor__n_neighbors": neighbor_values,
        "regressor__weights": ["uniform", "distance"],
        "regressor__p": [1, 2],  # 1 = Manhattan, 2 = Euclidean
    }


def rf_param_grid() -> Dict[str, Any]:
    """
    Default hyperparameter search space for RandomForestRegressor.

    Values are sampled uniformly when using random search.
    """
    n_estimator_values = list(range(200, 1700, 100)) + [1800, 2000]
    max_depth_values = [None] + list(range(4, 21)) + [22, 24, 26, 28, 30, 35, 40, 50, 60]
    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 8, 10]
    min_samples_split_values = [2, 3, 4, 5, 6, 7, 8, 10, 12]
    max_features_values = [None, "sqrt", "log2", 0.4, 0.5, 0.6, 0.75, 0.9]
    return {
        "n_estimators": n_estimator_values,
        "max_depth": max_depth_values,
        "min_samples_leaf": min_samples_leaf_values,
        "min_samples_split": min_samples_split_values,
        "max_features": max_features_values,
    }


def extra_trees_param_grid() -> Dict[str, Any]:
    """
    Default hyperparameter search space for ExtraTreesRegressor.

    Values are sampled uniformly when using random search.
    """
    n_estimator_values = list(range(200, 1700, 100)) + [1800, 2000]
    max_depth_values = [None] + list(range(4, 21)) + [24, 28, 32, 40, 60]
    min_samples_leaf_values = [1, 2, 3, 4, 5, 6, 8, 10]
    min_samples_split_values = [2, 3, 4, 5, 6, 7, 8, 10, 12]
    max_features_values = [None, "sqrt", "log2", 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
    bootstrap_values = [False, True]
    return {
        "n_estimators": n_estimator_values,
        "max_depth": max_depth_values,
        "min_samples_leaf": min_samples_leaf_values,
        "min_samples_split": min_samples_split_values,
        "max_features": max_features_values,
        "bootstrap": bootstrap_values,
    }


def xgb_param_grid() -> Dict[str, Any]:
    """
    Default hyperparameter search space for XGBRegressor.

    Values are sampled uniformly when using random search.

    Raises
    ------
    ImportError
        If XGBoost is not installed.
    """
    n_estimator_values = list(range(200, 1200, 100)) + [1250, 1300, 1400]
    max_depth_values = list(range(3, 9)) + [9, 10]
    learning_rate_values = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.2]
    subsample_values = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
    colsample_values = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
    gamma_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    reg_lambda_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    return {
        "n_estimators": n_estimator_values,
        "max_depth": max_depth_values,
        "learning_rate": learning_rate_values,
        "subsample": subsample_values,
        "colsample_bytree": colsample_values,
        "gamma": gamma_values,
        "reg_lambda": reg_lambda_values,
    }



def svr_param_grid() -> Dict[str, Any]:
    """
    Default hyperparameter search space for Pipeline(SVR) with scaling.

    Values are sampled uniformly when using random search.
    """
    c_values = np.logspace(-1, 3, 20)
    epsilon_values = np.logspace(-3, 0, 15)
    gamma_values = np.logspace(-3, 1, 20)
    return {
        "regressor__C": c_values.tolist(),
        "regressor__epsilon": epsilon_values.tolist(),
        "regressor__gamma": gamma_values.tolist(),
    }



def kr_param_grid() -> Dict[str, Any]:
    """
    Default bandwidth search space for Pipeline(kr) with scaling.

    Values are sampled uniformly when using random search.
    """
    linear_bandwidths = np.linspace(0.02, 0.6, 20)
    mid_bandwidths = np.linspace(0.65, 3.5, 18)
    log_bandwidths = np.logspace(-2.5, 0.7, 20)
    bandwidth_values = np.unique(np.round(np.concatenate([linear_bandwidths, mid_bandwidths, log_bandwidths]), 4))
    return {
        "regressor__bandwidth": bandwidth_values.tolist(),
    }


# ----------------------------------------------------------------------
# Core training function for deviation models
# ----------------------------------------------------------------------
def train_deviation_model(
    X: np.ndarray,
    a: np.ndarray,
    model_type: str = "rf",
    param_distributions: Optional[Dict[str, Any]] = None,
    cv_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 0,
    n_iter: int = 50,
    ) -> Tuple[Any, Dict[str, Any], float, float, float, float, Dict[str, Any]]:
    """
    Random-search CV for deviation-function models.

    This function fits a regression model g(x) ≈ E[a | x],
    where 'a' is typically the absolute error |y_pred - y_true|
    of a given thermobarometer model.

    Parameters
    ----------
        X : array-like of shape (n_samples, n_features)
            Composition features.
        a : array-like of shape (n_samples,)
            Absolute errors for the target model.
        model_type : {"knn", "rf", "et", "xgb", "kr", "svm"}
            Type of regression model used to approximate the deviation function.
        param_distributions : dict, optional
            Hyperparameter sampling space for RandomizedSearchCV.
            If None, use the default space for the given model_type.
        cv_splits : int
            Number of CV splits for KFold.
        random_state : int
            Random seed for KFold, random search, and tree-based models.
        n_jobs : int
            Number of parallel jobs for RandomizedSearchCV and cross-validation.
        verbose : int
            Verbosity level for RandomizedSearchCV.
        n_iter : int
            Number of parameter settings that are sampled.

    Returns
    -------
        best_model : estimator
            Best estimator found by RandomizedSearchCV.
        best_params : dict
            Best hyperparameters.
        best_cv_r2 : float
            Mean cross-validated R² score of the best estimator.
        best_cv_rmse : float
            Mean cross-validated RMSE (lower is better) for the best estimator.
        cv_results : dict
            Full cv_results_ from RandomizedSearchCV (for diagnostics).
    """
    # Ensure numpy array
    X = np.asarray(X)
    a = np.asarray(a).ravel()

    # Select estimator and default param space
    model_type = model_type.lower()
    if model_type == "knn":
        estimator = build_knn_regressor()
        default_space = knn_param_grid()
    elif model_type == "rf":
        estimator = build_rf_regressor(random_state=random_state)
        default_space = rf_param_grid()
    elif model_type in ("et", "extratrees", "extra_trees"):
        estimator = build_extra_trees_regressor(random_state=random_state)
        default_space = extra_trees_param_grid()
    elif model_type in ("xgb", "xgboost"):
        estimator = build_xgb_regressor(random_state=random_state)
        default_space = xgb_param_grid()
    elif model_type in ("svm", "svr"):
        estimator = build_svr_regressor()
        default_space = svr_param_grid()
    elif model_type == "kr":
        estimator = build_kr_regressor()
        default_space = kr_param_grid()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if param_distributions is None:
        param_distributions = default_space

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
    )


    search.fit(X, a)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_cv_r2 = float(search.best_score_)

    rmse_cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rmse_scores = cross_val_score(
        best_model,
        X,
        a,
        cv=rmse_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
    )
    best_cv_rmse = float(-rmse_scores.mean())
    # MAPE
    mape_cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    mape_scores = cross_val_score(
        best_model,
        X,
        a,
        cv=mape_cv,
        scoring=mape_scorer,
        n_jobs=n_jobs,
    )
    best_cv_mape = float(mape_scores.mean())

    cv_results = search.cv_results_

    return best_model, best_params, best_cv_r2, best_cv_rmse,  best_cv_mape ,cv_results


# ----------------------------------------------------------------------
# Simple helper for predicting deviation on new compositions
# ----------------------------------------------------------------------
def deviation_predict(model: Any, x: np.ndarray) -> float:
    """
    Predict deviation (e.g. expected absolute error) for a single composition.

    Parameters
    ----------
    model : estimator
        Trained deviation model (e.g. returned by train_deviation_model).
    x : array-like of shape (n_features,)
        Composition of a single natural sample.

    Returns
    -------
    deviation : float
        Predicted deviation for the given sample.
    """
    x = np.asarray(x).reshape(1, -1)
    return float(model.predict(x))



def compare_algorism(
    X_train,
    deviation_train,
    diagnose: bool = False,
    n_iter: int = 50,
    random_state: int = 42,
    ) -> pd.DataFrame:
    """
    Compare multiple deviation-model algorithms on a common training set.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Feature matrix used for training (shape: n_samples × n_features).
    deviation_train : array-like
        Target deviation values (e.g. absolute errors) aligned with X_train.
    diagnose : bool, default False
        If True, print detailed diagnostic information during training.
    n_iter : int, default 50
        Number of random hyperparameter samples per algorithm.
    random_state : int, default 42
        Random seed shared across algorithms.

    Returns
    -------
    pd.DataFrame
        Table summarizing the best hyperparameters and validation metrics for
        each algorithm. Columns include `algorithm`, `best_params`,
        `best_cv_r2`, and `best_cv_rmse`.
    """
    score_rows = []
    algorithm_list = ["knn", "rf", "et", "xgb", "svm", "kr"]
    verbose_level = 1 if diagnose else 0
    for algorithm in algorithm_list:
        if diagnose:
            print(f"Algorithm: {algorithm}")
        model, best_params, best_cv_r2, best_cv_rmse, best_cv_mape, cv_results = train_deviation_model(
            X_train,
            deviation_train,
            model_type=algorithm,
            param_distributions=None,
            cv_splits=5,
            random_state=random_state,
            n_jobs=-1,
            verbose=verbose_level,
            n_iter=n_iter,
        )
        score_rows.append({
            "model": model,
            "algorithm": algorithm,
            "best_params": best_params,
            "best_cv_r2": best_cv_r2,
            "best_cv_rmse": best_cv_rmse,
            "best_cv_mape": best_cv_mape,
        })
    return pd.DataFrame(score_rows)





from aims4pt.model_tools import ModelManager
from aims4pt.statistic_tools.out_of_distribution import OODDetectorOneClassSVM
from aims4pt.utils import normalize_column_names

class deviation_model_for_a_thermobarometer:
    """Train and apply a deviation regressor for a thermobarometer model."""

    def __init__(
        self,
        thermobarometer_model: ModelManager,
        if_ood_detector: bool = False,
    ):
        """Store the thermobarometer reference and optional OOD detector settings.

        Parameters
        ----------
        thermobarometer_model : ModelManager
            Model wrapper providing feature ordering and prediction interface.
        if_ood_detector : bool, default False
            When True, an out-of-distribution detector is trained alongside
            the deviation model.
        """
        # Full thermobarometer model (may contain R models etc.).
        # This attribute will NOT be serialized.
        self.thermobarometer_model = thermobarometer_model

        # Basic metadata copied from the thermobarometer model.
        self.features = thermobarometer_model.standard_columns
        self.T_P = thermobarometer_model.T_P
        self.cpx_only = thermobarometer_model.cpx_only
        self.require_water = getattr(thermobarometer_model, "require_water", False)

        # Feature name lists for input normalization.
        self.cpx_names = thermobarometer_model.cpx_names
        self.liq_names = getattr(thermobarometer_model, "liq_names", None)

        # Deviation model and optional OOD detector.
        self.ood_detector = None
        self.deviation_model = None

        # Configuration flags and diagnostics.
        self.if_ood_detector = if_ood_detector
        self.r2 = np.nan
        self.rmse = np.nan
        self.mape = np.nan
        self.X_cali: Optional[pd.DataFrame] = None
        self.y_cali: Optional[pd.Series] = None

        self.if_normalize_liq = getattr(thermobarometer_model, "if_normalize_liq", False)

        # Indicates whether the deviation model has been successfully fitted.
        self.fitted = False

    def fit_deviation_model(
        self,
        calibration_cpx: pd.DataFrame,
        calibration_target: pd.Series,
        calibration_liq: Optional[pd.DataFrame] = None,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        """Fit an OOD detector and deviation regressor for either P or T targets.

        Parameters
        ----------
        calibration_cpx : pd.DataFrame
            Calibration records containing model input features.
        calibration_target : pd.Series
            True target values corresponding to the calibration features.
        calibration_liq : pd.DataFrame, optional
            Calibration liquid compositions (if applicable).
        n_iter : int, default 50
            Number of random hyperparameter configurations evaluated.
        random_state : int, default 42
            Seed controlling stochastic hyperparameter search and CV splits.

        Returns
        -------
        None
        """
        if self.thermobarometer_model is None:
            # After deserialization, thermobarometer_model is intentionally None.
            # A new instance should be created if re-fitting is needed.
            raise RuntimeError(
                "thermobarometer_model is not set; cannot fit deviation model."
            )

        if not isinstance(calibration_cpx, pd.DataFrame):
            raise TypeError("calibration_cpx must be a pandas DataFrame.")
        if calibration_cpx.empty:
            raise ValueError("calibration_cpx must not be empty.")

        # Build calibration feature matrix and predictions from the thermobarometer.
        if self.cpx_only and not self.require_water:
            X_cali = normalize_column_names(calibration_cpx, self.features)
            y_pred = self.thermobarometer_model.predict(X_cali)
        else:
            if calibration_liq is None:
                raise ValueError(
                    "calibration_liq must be provided for non-cpx-only models."
                )
            X_cpx = normalize_column_names(calibration_cpx, self.cpx_names)
            if calibration_liq is None:
                print("Warning: Liquid composition is required but not provided. Using zeros.")
                calibration_liq = pd.DataFrame(0, index=calibration_cpx.index, columns=self.liq_names)
            X_liq = normalize_column_names(calibration_liq, self.liq_names)
            if self.if_normalize_liq:
                X_liq_non_water = X_liq.copy().drop(columns=["H2O_liq"], errors='ignore') 
                total = X_liq_non_water.sum(axis=1)
                X_liq.loc[:, X_liq_non_water.columns] = (
                    X_liq_non_water.div(total, axis=0) * 100.0
                )
            y_pred = self.thermobarometer_model.predict(X_cpx, X_liq)
            X_cali = pd.concat([X_cpx, X_liq], axis=1)

        y_true = calibration_target

        # Compute absolute deviation |y_pred - y_true|.
        pred_array = np.asarray(y_pred).ravel()
        y_true_array = np.asarray(y_true).ravel()
        raw_deviation  = np.abs(pred_array - y_true_array)

        # also mask if it is unreasonable large
        threshold = 20 if self.thermobarometer_model.T_P == "P" else 500
        y_cali = np.where(~np.isfinite(raw_deviation), threshold, raw_deviation)
        y_cali = np.clip(y_cali, 0, threshold)

        # check if naN exists
        if np.any(np.isnan(y_cali)):
            print("Debug Info: NaN values found in computed deviations.")
            raise ValueError("NaN values found in computed deviations.")
        
        self.X_cali = X_cali.copy()
        self.y_cali = pd.Series(y_cali, index=X_cali.index, name="deviation")
       
        # Optionally fit an OOD detector on the calibration feature space.
        if self.if_ood_detector:
            self.ood_detector = OODDetectorOneClassSVM().fit(self.X_cali.values)

        # Train the deviation regression model (e.g. ExtraTrees).
        (
            self.deviation_model,
            best_params,
            self.r2,
            self.rmse,
            self.mape,
            self.cv_results,
        ) = train_deviation_model(
            self.X_cali,
            self.y_cali,
            model_type="et",
            param_distributions=None,
            cv_splits=5,
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
            n_iter=n_iter,
        )

        self.fitted = True

    def process_input(
        self,
        x_cpx: pd.DataFrame,
        x_liq: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Normalize and combine input features for deviation prediction.

        Parameters
        ----------
        x_cpx : pd.DataFrame
            Clinopyroxene compositions.
        x_liq : pd.DataFrame, optional
            Liquid compositions when required by the underlying thermobarometer.

        Returns
        -------
        pd.DataFrame
            Normalized feature matrix ready for the deviation model.
        """
        if self.cpx_only and not self.require_water:
            X = normalize_column_names(x_cpx, self.features)
        else:
            if x_liq is None:
                raise ValueError("x_liq must be provided for non-cpx-only models.")
            X_cpx = normalize_column_names(x_cpx, self.cpx_names)
            if x_liq is None:
                print("Warning: Liquid composition is required but not provided. Using zeros.")
                x_liq = pd.DataFrame(0, index=x_cpx.index, columns=self.liq_names)
            X_liq = normalize_column_names(x_liq, self.liq_names)
            if self.if_normalize_liq:
                X_liq_non_water = X_liq.copy().drop(columns=["H2O_liq"], errors='ignore') 
                total = X_liq_non_water.sum(axis=1)
                X_liq.loc[:, X_liq_non_water.columns] = (
                    X_liq_non_water.div(total, axis=0) * 100.0
                )

            X = pd.concat([X_cpx, X_liq], axis=1)
        return X

    def predict_is_odd(
        self,
        x_cpx: pd.DataFrame,
        x_liq: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Return a boolean mask indicating which rows are out-of-distribution.

        Parameters
        ----------
        x_cpx : pd.DataFrame
            Clinopyroxene compositions.
        x_liq : pd.DataFrame, optional
            Liquid compositions when required by the underlying thermobarometer.

        Returns
        -------
        pd.Series
            Boolean flags per row (True for out-of-distribution samples).
        """
        if self.ood_detector is None:
            raise RuntimeError(
                "OOD detector is not available. "
                "Set if_ood_detector=True and refit the model to use this method."
            )

        X_test = self.process_input(x_cpx, x_liq)
        # Apply the detector row-wise to keep index alignment.
        is_ood: pd.Series = X_test.apply(self.ood_detector.is_ood, axis=1)
        return is_ood

    def predict_deviation(
        self,
        x_cpx: pd.DataFrame,
        x_liq: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Predict absolute-error magnitude for each input composition.

        Parameters
        ----------
        x_cpx : pd.DataFrame
            Clinopyroxene compositions.
        x_liq : pd.DataFrame, optional
            Liquid compositions when required by the underlying thermobarometer.

        Returns
        -------
        pd.Series
            Estimated deviation values aligned with the input rows.
        """
        if not self.fitted or self.deviation_model is None:
            raise RuntimeError(
                "Deviation model is not fitted. Call fit_deviation_model() first."
            )

        X_test = self.process_input(x_cpx, x_liq)
        # Use the helper deviation_predict on each row.
        deviation = X_test.apply(
            lambda row: deviation_predict(self.deviation_model, row.values),
            axis=1,
        )
        deviation = pd.Series(deviation.values, index=X_test.index, name="deviation")
        return deviation

    # ------------------------------------------------------------------
    # Custom serialization logic
    # ------------------------------------------------------------------
    def __getstate__(self):
        """Prepare object state for pickling / joblib serialization.

        Only fitted objects are allowed to be serialized. The underlying
        thermobarometer_model is dropped from the serialized state to avoid
        issues with non-serializable components (e.g., R sessions via rpy2).
        """
        if not self.fitted:
            raise RuntimeError(
                "Cannot serialize an unfitted deviation_model_for_a_thermobarometer object."
            )

        state = self.__dict__.copy()
        # thermobarometer_model may contain R models or other non-serializable
        # resources; we drop it from the serialized state.
        state["thermobarometer_model"] = None

        # Optionally, you could drop large training data to reduce file size:
        # state["X_cali"] = None
        # state["y_cali"] = None
        # state["cv_results"] = None

        return state

    def __setstate__(self, state):
        """Restore object state from pickle / joblib.

        After deserialization, thermobarometer_model remains None, meaning this
        instance can be used for prediction but not for re-fitting the deviation
        model.
        """
        self.__dict__.update(state)
        self.thermobarometer_model = None


class empty_deviation_model:
    """A placeholder deviation model that always predicts zero deviation."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "empty_deviation_model":
        """No-op fit method."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict zero deviation for all inputs."""
        X = np.asarray(X)
        return np.zeros(X.shape[0], dtype=float)

__all__ = [
    "KernelDensityRegressor",
    "build_knn_regressor",
    "build_rf_regressor",
    "build_extra_trees_regressor",
    "build_xgb_regressor",
    "build_svr_regressor",
    "build_kr_regressor",
    "knn_param_grid",
    "rf_param_grid",
    "extra_trees_param_grid",
    "xgb_param_grid",
    "svr_param_grid",
    "kr_param_grid",
    "train_deviation_model",
    "deviation_predict",
    "compare_algorism",
    "deviation_model_for_a_thermobarometer",
]
