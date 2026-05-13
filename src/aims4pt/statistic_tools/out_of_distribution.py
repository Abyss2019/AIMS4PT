"""Out-of-distribution detectors tailored for thermobarometric models."""

import numpy as np
import pandas as pd
from itertools import product
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from aims4pt.utils import normalize_column_names
from aims4pt.model_tools import ModelManager


class OODDetectorMahalanobis:
    def __init__(self):
        """Initialise internal state placeholders for the Mahalanobis detector."""
        self.mu = None
        self.cov_inv = None
        self.threshold = None

    def fit(self, X, quantile=0.95):
        """Estimate the Mahalanobis-based decision boundary for inliers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Calibration observations drawn from the in-distribution population.
        quantile : float, default 0.95
            Quantile used to set the distance threshold above which samples are
            flagged as out-of-distribution.

        Returns
        -------
        OODDetectorMahalanobis
            The fitted detector instance.
        """
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)

        lw = LedoitWolf().fit(X)
        cov = lw.covariance_
        self.cov_inv = np.linalg.inv(cov)

        d_train = np.sqrt(
            np.sum((X - self.mu) @ self.cov_inv * (X - self.mu), axis=1))
        self.threshold = np.quantile(d_train, quantile)
        return self

    def score(self, x):
        """Return the Mahalanobis distance of a sample from the fitted center.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Sample for which to compute the distance.

        Returns
        -------
        float
            Mahalanobis distance between ``x`` and the calibrated mean.
        """
        x = np.asarray(x, dtype=float)
        return float(np.sqrt((x - self.mu) @ self.cov_inv @ (x - self.mu).T))

    def is_ood(self, x):
        """Flag samples whose Mahalanobis distance exceeds the calibrated threshold.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Sample to classify.

        Returns
        -------
        bool
            ``True`` if the sample lies outside the learned distribution.
        """
        return self.score(x) > self.threshold


class _FeatureWeighter(BaseEstimator, TransformerMixin):
    """Transformer that applies precomputed per-feature weights."""

    def __init__(self, weights):
        self.weights = weights

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if X.shape[1] != weights.shape[0]:
            raise ValueError(
                "Feature weight vector length must match the number of features."
            )
        if np.any(weights < 0):
            raise ValueError("Feature weights must be non-negative numbers.")
        self.weights_ = weights
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X * self.weights_


class OODDetectorOneClassSVM:
    def __init__(
        self,
        param_grid=None,
        n_iter=20,
        random_state=42,
        weighted: bool = False,
        feature_weights: dict = None,
    ):
        """Configure a One-Class SVM based detector.

        Parameters
        ----------
        param_grid : dict or None, optional
            Parameter search space for the SVM pipeline. Defaults to a small
            grid over ``nu`` and ``gamma`` values.
        n_iter : int, default 20
            Maximum number of sampled hyperparameter combinations.
        random_state : int, default 42
            Seed controlling the stochastic parameter sampling.
        """
        self.param_grid = param_grid or {
            "ocsvm__nu": [0.03, 0.05, 0.07, 0.1],
            "ocsvm__gamma": ["scale", "auto", 0.01, 0.05, 0.1, 0.5, 1.0],
            "ocsvm__kernel": ["rbf"],
        }
        self.n_iter = n_iter
        self.random_state = random_state
        self.pipeline = None
        self.best_params_ = None
        self.best_score_ = None
        self.weighted = weighted
        self.feature_weights = feature_weights
        self.feature_weight_vector_ = None
        self.feature_names_ = None

    def _resolve_feature_weights(self, X, feature_names):
        if not self.weighted:
            return None
        if self.feature_weights is None:
            raise ValueError(
                "Feature weights must be provided when weighted=True."
            )
        n_features = np.shape(X)[1]
        if isinstance(self.feature_weights, dict):
            if feature_names is None:
                raise ValueError(
                    "Feature weights provided as a dict require pandas input with column names."
                )
            missing = [name for name in feature_names if name not in self.feature_weights]
            if missing:
                raise ValueError(
                    "Missing weights for features: {}".format(", ".join(missing))
                )
            weights = np.asarray([self.feature_weights[name] for name in feature_names], dtype=float)
        else:
            weights = np.asarray(self.feature_weights, dtype=float)
            if weights.shape[0] != n_features:
                raise ValueError(
                    "Feature weight vector length must match the number of features."
                )
        if np.any(weights < 0):
            raise ValueError("Feature weights must be non-negative numbers.")
        return weights

    def _param_samples(self):
        """Yield a bounded set of hyperparameter combinations for evaluation.

        Returns
        -------
        list of dict
            Parameter configurations sampled from ``param_grid``.
        """
        rng = np.random.default_rng(self.random_state)
        keys = list(self.param_grid.keys())
        cartesian = list(product(*[self.param_grid[k] for k in keys]))
        if len(cartesian) <= self.n_iter:
            selected = cartesian
        else:
            idx = rng.choice(len(cartesian), size=self.n_iter, replace=False)
            selected = [cartesian[i] for i in idx]
        return [dict(zip(keys, values)) for values in selected]

    def fit(self, X):
        """Train the One-Class SVM pipeline on calibration observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Calibration data representing the in-distribution compositions.

        Returns
        -------
        OODDetectorOneClassSVM
            The fitted detector instance.
        """
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.feature_names_ = feature_names
        weights_vector = self._resolve_feature_weights(X, feature_names)
        X = np.asarray(X, dtype=float)
        steps = [("scaler", StandardScaler())]
        if weights_vector is not None:
            steps.append(("feature_weight", _FeatureWeighter(weights_vector)))
        steps.append(("ocsvm", OneClassSVM()))
        base_pipeline = Pipeline(steps=steps)
        best_score = -np.inf
        best_pipeline = None
        best_params = None
        for params in self._param_samples():
            candidate = clone(base_pipeline)
            candidate.set_params(**params)
            candidate.fit(X)
            decision_scores = candidate.decision_function(X)
            mean_score = float(np.mean(decision_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_pipeline = candidate
                best_params = params
        self.pipeline = best_pipeline
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.feature_weight_vector_ = weights_vector
        return self

    def score(self, x):
        """Return the signed distance of ``x`` to the learned decision boundary. "decision-function score"

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Sample to evaluate.

        Returns
        -------
        float
            Decision-function score (positive for inliers, negative for OOD).
        """
        if self.pipeline is None:
            raise RuntimeError("Detector has not been fitted.")
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return float(self.pipeline.decision_function(x)[0])
    


    def is_ood(self, x):
        """Return ``True`` when the sample is classified as out-of-distribution.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Feature order must match the training data.
            Sample to classify.

        Returns
        -------
        bool
            ``True`` when the model labels the sample as anomalous.
        """
        if self.pipeline is None:
            raise RuntimeError("Detector has not been fitted.")
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return bool(self.pipeline.predict(x)[0] == -1)


class OODDetectorIsolationForest:
    def __init__(
        self,
        param_grid=None,
        n_iter=20,
        random_state=42,
    ):
        """Configure an Isolation Forest based detector.

        Parameters
        ----------
        param_grid : dict or None, optional
            Parameter search space for the Isolation Forest pipeline. Defaults
            to a compact grid over estimator count, sampling fraction, and
            contamination rate.
        n_iter : int, default 20
            Maximum number of sampled hyperparameter combinations.
        random_state : int, default 42
            Seed controlling the stochastic parameter sampling and the forest itself.
        """
        self.param_grid = param_grid or {
            "iforest__n_estimators": [200, 300, 400],
            "iforest__max_samples": [0.6, 0.8, 1.0],
            "iforest__contamination": [0.05, 0.08, 0.1, 0.12],
        }
        self.n_iter = n_iter
        self.random_state = random_state
        self.pipeline = None
        self.best_params_ = None
        self.best_score_ = None

    def _param_samples(self):
        """Yield a bounded set of hyperparameter combinations for evaluation.

        Returns
        -------
        list of dict
            Parameter configurations sampled from ``param_grid``.
        """
        rng = np.random.default_rng(self.random_state)
        keys = list(self.param_grid.keys())
        cartesian = list(product(*[self.param_grid[k] for k in keys]))
        if len(cartesian) <= self.n_iter:
            selected = cartesian
        else:
            idx = rng.choice(len(cartesian), size=self.n_iter, replace=False)
            selected = [cartesian[i] for i in idx]
        return [dict(zip(keys, values)) for values in selected]

    def fit(self, X):
        """Train the Isolation Forest pipeline on calibration observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Calibration data representing the in-distribution compositions.

        Returns
        -------
        OODDetectorIsolationForest
            The fitted detector instance.
        """
        X = np.asarray(X, dtype=float)
        base_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("iforest", IsolationForest(random_state=self.random_state)),
            ]
        )
        best_score = -np.inf
        best_pipeline = None
        best_params = None
        for params in self._param_samples():
            candidate = clone(base_pipeline)
            candidate.set_params(**params)
            candidate.fit(X)
            decision_scores = candidate.decision_function(X)
            mean_score = float(np.mean(decision_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_pipeline = candidate
                best_params = params
        self.pipeline = best_pipeline
        self.best_params_ = best_params
        self.best_score_ = best_score
        return self

    def score(self, x):
        """Return the anomaly score produced by the Isolation Forest.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Sample to evaluate.

        Returns
        -------
        float
            Decision-function score (larger is more in-distribution).
        """
        if self.pipeline is None:
            raise RuntimeError("Detector has not been fitted.")
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return float(self.pipeline.decision_function(x)[0])

    def is_ood(self, x):
        """Return ``True`` when the Isolation Forest marks the sample as OOD.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Sample to classify.

        Returns
        -------
        bool
            ``True`` when the Isolation Forest predicts the sample is anomalous.
        """
        if self.pipeline is None:
            raise RuntimeError("Detector has not been fitted.")
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return bool(self.pipeline.predict(x)[0] == -1)


class OODDetectorOneClassSVM_with_PCA:
    def __init__(
        self,
        param_grid=None,
        n_iter=20,
        random_state=42,
    ):
        """Configure a PCA-preprocessed One-Class SVM detector.

        Parameters
        ----------
        param_grid : dict or None, optional
            Parameter search space for the combined PCA + SVM pipeline. Defaults
            to a grid spanning PCA components and SVM hyperparameters.
        n_iter : int, default 20
            Maximum number of sampled hyperparameter combinations.
        random_state : int, default 42
            Seed controlling stochastic parameter sampling and PCA initialisation.
        """
        self.param_grid = param_grid or {
            "pca__n_components": [0.7, 0.85, 0.95, None],
            "ocsvm__nu": [0.03, 0.05, 0.07, 0.1],
            "ocsvm__gamma": ["scale", "auto", 0.01, 0.05, 0.1, 0.5, 1.0],
            "ocsvm__kernel": ["rbf"],
        }
        self.n_iter = n_iter
        self.random_state = random_state
        self.pipeline = None
        self.best_params_ = None
        self.best_score_ = None

    def _param_samples(self):
        """Yield a bounded set of hyperparameter combinations for evaluation.

        Returns
        -------
        list of dict
            Parameter configurations sampled from ``param_grid``.
        """
        rng = np.random.default_rng(self.random_state)
        keys = list(self.param_grid.keys())
        cartesian = list(product(*[self.param_grid[k] for k in keys]))
        if len(cartesian) <= self.n_iter:
            selected = cartesian
        else:
            idx = rng.choice(len(cartesian), size=self.n_iter, replace=False)
            selected = [cartesian[i] for i in idx]
        return [dict(zip(keys, values)) for values in selected]

    def fit(self, X):
        """Train the PCA + One-Class SVM pipeline on calibration data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Calibration data representing the in-distribution compositions.

        Returns
        -------
        OODDetectorOneClassSVM_with_PCA
            The fitted detector instance.
        """
        X = np.asarray(X, dtype=float)
        base_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(random_state=self.random_state)),
                ("ocsvm", OneClassSVM()),
            ]
        )
        best_score = -np.inf
        best_pipeline = None
        best_params = None
        for params in self._param_samples():
            candidate = clone(base_pipeline)
            candidate.set_params(**params)
            candidate.fit(X)
            decision_scores = candidate.decision_function(X)
            mean_score = float(np.mean(decision_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_pipeline = candidate
                best_params = params
        self.pipeline = best_pipeline
        self.best_params_ = best_params
        self.best_score_ = best_score
        return self

    def score(self, x):
        """Return the decision-function score produced by the PCA + OCSVM pipeline.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Sample to evaluate.

        Returns
        -------
        float
            Decision-function score (positive for inliers, negative for OOD).
        """
        if self.pipeline is None:
            raise RuntimeError("Detector has not been fitted.")
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return float(self.pipeline.decision_function(x)[0])

    def is_ood(self, x):
        """Return ``True`` when the PCA-enhanced detector classifies the sample as OOD.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Sample to classify.

        Returns
        -------
        bool
            ``True`` when the detector labels the sample as anomalous.
        """
        if self.pipeline is None:
            raise RuntimeError("Detector has not been fitted.")
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return bool(self.pipeline.predict(x)[0] == -1)



class ThermobarometerOODFactory:
    """Factory for building OOD detectors tied to a thermobarometer.

    Parameters
    ----------
    model : ModelManager
        Thermobarometer wrapper providing training compositions.
    features : {'key', 'all'}, default 'all'
        Feature subset used when fitting the detector.
    """

    def __init__(self, model: ModelManager, features: str = "all", weighted: bool = False):
        self.model = model
        self.features = features
        self.weighted = weighted
        if self.weighted:
            if not hasattr(model, "feature_importance_df"):
                raise ValueError(
                    "Weighted OOD detection requires 'feature_importance_df' on the model."
                )
            self.feature_weights = (
                model.feature_importance_df.set_index("Feature")["Importance"].to_dict()
            )
        else:
            self.feature_weights = None
        self.cpx_only = model.cpx_only
        self.require_water = getattr(model, "require_water", False)
        self.cpx_names = model.cpx_names
        self.liq_names = getattr(model, "liq_names", None)
        self.if_normalize_liq = getattr(model, "if_normalize_liq", False)
        self.key_features = getattr(model, "key_features", None)

        self.detector = self.fit()

    def fit(self) -> OODDetectorOneClassSVM:
        """Construct and fit a One-Class SVM OOD detector.

        Returns
        -------
        OODDetectorOneClassSVM
            Detector trained on the model's normalized training compositions.

        Raises
        ------
        ValueError
            If training data are missing or an unknown feature subset is selected.
        """
        if self.model is None:
            raise ValueError("Model cannot be None.")
        if self.model.X_cpx_training is None:
            raise ValueError("Model does not provide clinopyroxene training data.")
        training_data = normalize_column_names(
            self.model.X_cpx_training, self.model.cpx_names
        )
        if not self.cpx_only or self.model.require_water:
            liq_training_data = normalize_column_names(
                self.model.X_liq_training, self.model.liq_names
            )
            if self.if_normalize_liq:
                liq_non_water = liq_training_data.drop(columns=["H2O_liq"], errors='ignore') 
                total = liq_non_water.sum(axis=1)
                liq_training_data.loc[:, liq_non_water.columns] = (
                    liq_non_water.div(total, axis=0) * 100.0
                )
                if self.model.require_water:
                    liq_training_data["H2O_liq"] = liq_training_data["H2O_liq"]
            
            training_data = pd.concat([training_data, liq_training_data], axis=1)
        if self.features == "key":
            if not hasattr(self.model, "key_features"):
                raise ValueError("Model does not expose 'key_features'.")
            training_data = training_data[self.model.key_features]
        elif self.features != "all":
            raise ValueError("`features` must be either 'key' or 'all'.")
        detector = OODDetectorOneClassSVM(
            weighted=self.weighted, feature_weights=self.feature_weights
        )
        return detector.fit(training_data)
    
    def is_ood(self, X_cpx, X_liq=None) -> pd.Series:
        """Classify a sample as in-distribution or out-of-distribution.

        Parameters
        ----------
        X_cpx : pd.DataFrame
            Clinopyroxene composition(s) to evaluate.
        X_liq : pd.DataFrame, optional
            Liquid composition(s) to evaluate, if applicable.
        Returns
        -------
        pd.Series
            Boolean series indicating OOD status for each sample.
        """
        X_cpx_norm = normalize_column_names(X_cpx, self.cpx_names)
        if not self.cpx_only or self.require_water:
            if X_liq is None:
                # warning message:
                print("Warning: Liquid composition is required but not provided. Using zeros.")
                X_liq = pd.DataFrame(0, index=X_cpx.index, columns=self.liq_names)
            X_liq_norm = normalize_column_names(X_liq, self.liq_names)
            if self.if_normalize_liq and not self.require_water:
                total = X_liq_norm.sum(axis=1)
                X_liq_norm = X_liq_norm.div(total, axis=0) * 100.0
            elif self.if_normalize_liq:
                X_liq_non_water = X_liq_norm.drop(columns=["H2O_liq"], errors='ignore') 
                total = X_liq_non_water.sum(axis=1)
                X_liq_norm.loc[:, X_liq_non_water.columns] = (
                    X_liq_non_water.div(total, axis=0) * 100.0
                )
                if self.require_water:
                    X_liq_norm["H2O_liq"] = X_liq_norm["H2O_liq"]
            X_input = pd.concat([X_cpx_norm, X_liq_norm], axis=1)

        else:
            X_input = X_cpx_norm
        if self.features == "key":
            X_input = X_input[self.key_features]
        feature_order = getattr(self.detector, "feature_names_", None)
        if feature_order is not None:
            X_input = X_input[feature_order]
        ood_flags = X_input.apply(lambda row: self.detector.is_ood(row.values), axis=1)
        return ood_flags
    
    def score(self, X_cpx, X_liq=None) -> pd.Series:
        """Compute decision-function scores for given samples.

        Parameters
        ----------
        X_cpx : pd.DataFrame
            Clinopyroxene composition(s) to evaluate.
        X_liq : pd.DataFrame, optional
            Liquid composition(s) to evaluate, if applicable.
        Returns
        -------
        pd.Series
            Decision-function scores for each sample.
        """
        X_cpx_norm = normalize_column_names(X_cpx, self.cpx_names)
        if not self.cpx_only or self.require_water:
            if X_liq is None:
                # warning message:
                print("Warning: Liquid composition is required but not provided. Using zeros.")
                X_liq = pd.DataFrame(0, index=X_cpx.index, columns=self.liq_names)
            X_liq_norm = normalize_column_names(X_liq, self.liq_names)
            if self.if_normalize_liq and not self.require_water:
                total = X_liq_norm.sum(axis=1)
                X_liq_norm = X_liq_norm.div(total, axis=0) * 100.0
            elif self.if_normalize_liq and self.require_water:
                X_liq_non_water = X_liq_norm.drop(columns=["H2O_liq"], errors='ignore') 
                total = X_liq_non_water.sum(axis=1)
                X_liq_norm.loc[:, X_liq_non_water.columns] = (
                    X_liq_non_water.div(total, axis=0) * 100.0
                )
                X_liq_norm["H2O_liq"] = X_liq_norm["H2O_liq"]
            X_input = pd.concat([X_cpx_norm, X_liq_norm], axis=1)
        else:
            X_input = X_cpx_norm
        if self.features == "key":
            X_input = X_input[self.key_features]
        feature_order = getattr(self.detector, "feature_names_", None)
        if feature_order is not None:
            X_input = X_input[feature_order]
        scores = X_input.apply(lambda row: self.detector.score(row.values), axis=1)
        return scores
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None  # Exclude the model from serialization
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = None  # Model is not restored upon deserialization
