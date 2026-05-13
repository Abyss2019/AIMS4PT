"""Utilities to load and use R models via :mod:`rpy2`.

This module exposes a lightweight wrapper that mimics a scikit-learn-like
model with ``predict`` and persistence helpers. Environment variables are kept
consistent so that R can locate its runtime when the module is imported.
"""

import locale
from typing import Any, Optional

from aims4pt.model_tools.rpy_tools.r_env import (
    configure_r_environment,
    format_rpy2_setup_error,
)

import numpy as np
import pandas as pd

configure_r_environment()

try:
    import rpy2.robjects as robjects
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter
    from rpy2.robjects import pandas2ri
except Exception as exc:
    raise RuntimeError(format_rpy2_setup_error(exc)) from exc


# -------------------------------------------------------------------
# Locale (best effort; Windows may not have en_US.UTF-8)
# -------------------------------------------------------------------
try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except Exception:
    pass

try:
    robjects.r('Sys.setlocale("LC_ALL", "en_US.UTF-8")')
except Exception:
    pass


# -------------------------------------------------------------------
# R helper: install-if-missing
# -------------------------------------------------------------------
def ensure_r_packages(packages, repos: str = "https://cloud.r-project.org") -> None:
    """
    Ensure required R packages are installed and loadable.

    Parameters
    ----------
    packages : list[str] | tuple[str] | str
        Package name(s).
    repos : str
        CRAN mirror.
    """
    if isinstance(packages, str):
        packages = [packages]

    # Define ensure_package() in the embedded R session once (idempotent)
    robjects.r(f"""
    ensure_package <- function(pkg, repos = "{repos}") {{
      if (!suppressWarnings(require(pkg, character.only = TRUE))) {{
        message(sprintf("Installing package: %s", pkg))
        install.packages(pkg, dependencies = TRUE, repos = repos)
        suppressWarnings(library(pkg, character.only = TRUE))
      }}
    }}
    """)

    for pkg in packages:
        # install + load
        robjects.r(f'ensure_package("{pkg}")')


class r_model:
    """
    Pack an R model into a Python object with a ``predict`` interface.

    The class wraps the underlying R object while keeping a simple, scikit
    learn–like API for prediction and persistence.
    """

    def __init__(self, model: Optional[Any] = None, file_type: str = "rds"):
        self.model = model
        self.file_type = file_type

    def train(self, X: Any, y: Any) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict the response variable.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features to feed into the trained R model.

        Returns
        -------
        pandas.Series
            Predicted responses returned by the R model.
        """
        if self.model is None:
            raise ValueError("No model to predict.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas.DataFrame, got {type(X)}")

        # ---- ensure required R packages ----
        # ranger depends on Matrix on Windows quite often; install both proactively
        ensure_r_packages(["Matrix", "ranger"])

        # Convert pandas -> R within a local conversion context
        with localconverter(default_converter + pandas2ri.converter):
            r_df = robjects.conversion.py2rpy(X)

        predict_function = robjects.r["predict"]
        prediction = predict_function(self.model, r_df)

        # ranger::predict(ranger_model, data=...) often returns a list with $predictions
        # If so, extract $predictions before conversion.
        try:
            if hasattr(prediction, "names") and prediction.names is not None:
                if "predictions" in list(prediction.names):
                    prediction = prediction.rx2("predictions")
        except Exception:
            pass

        # Convert R -> pandas/numpy within the same conversion context
        with localconverter(default_converter + pandas2ri.converter):
            pred_py = robjects.conversion.rpy2py(prediction)

        # Normalize output to pandas.Series
        if isinstance(pred_py, pd.DataFrame):
            if pred_py.shape[1] == 1:
                return pred_py.iloc[:, 0].rename("prediction")
            return pred_py.squeeze(axis=1)

        if isinstance(pred_py, pd.Series):
            return pred_py.rename("prediction")

        if isinstance(pred_py, (np.ndarray, list, tuple)):
            return pd.Series(np.asarray(pred_py), name="prediction")

        return pd.Series(pred_py, name="prediction")

    def save(self, file_path: str) -> None:
        """Save the R model to a file."""
        if self.model is None:
            raise ValueError("No model to save.")

        robjects.globalenv["model"] = self.model
        robjects.globalenv["file_path"] = file_path
        robjects.r("saveRDS(model, file=file_path)")

    @classmethod
    def load(cls, file_path: str, file_type: str = "rds") -> "r_model":
        """
        Load the R model from a file.
        """
        if file_type == "rds":
            model = robjects.r["readRDS"](file_path)
        elif file_type == "rdata":
            robjects.r["load"](file_path)
            model = robjects.globalenv["model"]
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")

        return cls(model, file_type=file_type)

    def __repr__(self):
        return f"r_model(model={self.model}, file_type={self.file_type})"
