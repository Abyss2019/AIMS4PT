
"""R-backed wrapper for the Jorgenson et al. (2022) clinopyroxene thermobarometer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import aims4pt.model_tools.data.Jorgenson22 as jorgenson_data
from aims4pt.model_tools.ModelManager import ModelManager
from aims4pt.model_tools.model_registry import register_model
from aims4pt.model_tools.rpy_tools.r_env import (
    configure_r_environment,
    format_rpy2_setup_error,
)
from aims4pt.toolkit_utils import get_file_path
from aims4pt.utils import normalize_column_names

robjects = None
localconverter = None
default_converter = None
pandas2ri = None
_rpy_conversion = None
_RPY2_READY = False
_ORIGINAL_CCHAR_TO_STR = None


def _safe_cchar_to_str_with_maxlen(cdata, maxlen, encoding):
    try:
        return _ORIGINAL_CCHAR_TO_STR(cdata, maxlen, encoding)
    except UnicodeDecodeError:
        return _rpy_conversion.ffi.string(cdata, maxlen).decode("mbcs", errors="ignore")


def _ensure_rpy2_runtime() -> None:
    global robjects, localconverter, default_converter, pandas2ri
    global _RPY2_READY, _ORIGINAL_CCHAR_TO_STR, _rpy_conversion

    if _RPY2_READY:
        return

    configure_r_environment()
    try:
        import rpy2.robjects as _robjects
        from rpy2.robjects.conversion import localconverter as _localconverter
        from rpy2.robjects import default_converter as _default_converter
        from rpy2.robjects import pandas2ri as _pandas2ri
        from rpy2.rinterface_lib import conversion as _conversion
    except Exception as exc:
        raise RuntimeError(format_rpy2_setup_error(exc)) from exc

    robjects = _robjects
    localconverter = _localconverter
    default_converter = _default_converter
    pandas2ri = _pandas2ri
    _rpy_conversion = _conversion
    _ORIGINAL_CCHAR_TO_STR = _conversion._cchar_to_str_with_maxlen
    _conversion._cchar_to_str_with_maxlen = _safe_cchar_to_str_with_maxlen
    _RPY2_READY = True


_CPX_PICKLE_MAP = {
    "SiO2.cpx": "SiO2_cpx",
    "Al2O3.cpx": "Al2O3_cpx",
    "TiO2.cpx": "TiO2_cpx",
    "CaO.cpx": "CaO_cpx",
    "Na2O.cpx": "Na2O_cpx",
    "FeO.cpx": "FeO_cpx",
    "FeOt_cpx": "FeO_cpx",
    "MgO.cpx": "MgO_cpx",
    "MnO.cpx": "MnO_cpx",
    "Cr2O3.cpx": "Cr2O3_cpx",
    "P": "P_kbar",
    "T": "T_C",
}

_CPX_PICKLE_COLUMNS = [
    "SiO2_cpx",
    "TiO2_cpx",
    "Al2O3_cpx",
    "Cr2O3_cpx",
    "FeO_cpx",
    "MgO_cpx",
    "MnO_cpx",
    "CaO_cpx",
    "Na2O_cpx",
    "P_kbar",
    "T_C",
]

_LIQ_PICKLE_MAP = {
    "SiO2.liq": "SiO2_liq",
    "Al2O3.liq": "Al2O3_liq",
    "TiO2.liq": "TiO2_liq",
    "CaO.liq": "CaO_liq",
    "Na2O.liq": "Na2O_liq",
    "K2O.liq": "K2O_liq",
    "FeO.liq": "FeO_liq",
    "FeOt.liq": "FeO_liq",
    "FeOt_liq": "FeO_liq",
    "MgO.liq": "MgO_liq",
    "MnO.liq": "MnO_liq",
    "Cr2O3.liq": "Cr2O3_liq",
    "P2O5.liq": "P2O5_liq",
    "H2O.liq": "H2O_liq",
}

_LIQ_PICKLE_COLUMNS = [
    "SiO2_liq",
    "Al2O3_liq",
    "TiO2_liq",
    "CaO_liq",
    "Na2O_liq",
    "K2O_liq",
    "FeO_liq",
    "MgO_liq",
    "MnO_liq",
    "Cr2O3_liq",
    "P2O5_liq",
    "H2O_liq",
]


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def _read_training_csv(path: Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "latin1")
    last_error: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(path)


def _build_cpx_training(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_CPX_PICKLE_MAP)
    df = df.reindex(columns=_CPX_PICKLE_COLUMNS)
    df = _coerce_numeric(df).fillna(0.0)
    return df


def _build_liq_training(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=_LIQ_PICKLE_MAP)
    df = df.reindex(columns=_LIQ_PICKLE_COLUMNS)
    df = _coerce_numeric(df).fillna(0.0)
    return df


def _normalize_liquid_anhydrous(df: pd.DataFrame) -> pd.DataFrame:
    """Renormalise liquid oxides to 100 wt.% on an anhydrous basis."""
    if "H2O.liq" not in df.columns:
        return df
    renorm = df.copy()
    oxide_cols = [col for col in df.columns if col != "H2O.liq"]
    totals = renorm[oxide_cols].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(totals > 0, 100.0 / totals, 0.0)
    renorm[oxide_cols] = renorm[oxide_cols].mul(factors, axis=0)
    renorm["H2O.liq"] = renorm["H2O.liq"].fillna(0.0)
    return renorm


@register_model
class Jorgenson22(ModelManager):
    """Predict pressure or temperature using the released R models from Jorgenson et al. (2022).

    The original publication distributes pre-trained ``extraTrees`` models in
    ``RData`` format.  This wrapper mirrors the authors' "Plug and play" script
    by loading those artefacts via :mod:`rpy2`.  No re-training occurs within
    Python.
    """

    _MODEL_FILES = {
        ("T", True): ("model/T_C_noliq.Rdata", "T_C_noliq"),
        ("P", True): ("model/P_C_noliq.Rdata", "P_C_noliq"),
        ("T", False): ("model/T_C_liq.Rdata", "T_C_liq"),
        ("P", False): ("model/P_C_liq.Rdata", "P_C_liq"),
    }

    _CPX_NAMES = [
        "SiO2_cpx",
        "TiO2_cpx",
        "Al2O3_cpx",
        "Cr2O3_cpx",
        "FeO_cpx",
        "MgO_cpx",
        "MnO_cpx",
        "CaO_cpx",
        "Na2O_cpx",
    ]

    _CPX_R_MAP = {
        "SiO2_cpx": "SiO2.cpx",
        "TiO2_cpx": "TiO2.cpx",
        "Al2O3_cpx": "Al2O3.cpx",
        "Cr2O3_cpx": "Cr2O3.cpx",
        "FeO_cpx": "FeO.cpx",
        "MgO_cpx": "MgO.cpx",
        "MnO_cpx": "MnO.cpx",
        "CaO_cpx": "CaO.cpx",
        "Na2O_cpx": "Na2O.cpx",
    }

    _LIQ_NAMES = [
        "SiO2_liq",
        "TiO2_liq",
        "Al2O3_liq",
        "FeO_liq",
        "MgO_liq",
        "MnO_liq",
        "CaO_liq",
        "Na2O_liq",
        "K2O_liq",
    ]

    _LIQ_R_MAP = {
        "SiO2_liq": "SiO2.liq",
        "TiO2_liq": "TiO2.liq",
        "Al2O3_liq": "Al2O3.liq",
        "FeO_liq": "FeO.liq",
        "MgO_liq": "MgO.liq",
        "MnO_liq": "MnO.liq",
        "CaO_liq": "CaO.liq",
        "Na2O_liq": "Na2O.liq",
        "K2O_liq": "K2O.liq",
    }

    def __init__(self, T_P: str, cpx_only: bool = True, comments: Optional[str] = None):
        if T_P not in {"T", "P"}:
            raise ValueError("T_P must be 'T' (temperature) or 'P' (pressure).")

        super().__init__(comments=comments)

        self.model_name = "Jorgenson et al., 2022"
        self.T_P = T_P
        self.cpx_only = cpx_only
        self.require_water = False
        self.if_support_hydrous = True

        self.prediction_column_name = "T_C" if T_P == "T" else "P_kbar"
        # 72.5 (only)/44.9 (cpx-liq)	3.2 (only)/2.7 (cpx-liq)
        if T_P == "T":
            self.uncertainty = 72.5 if cpx_only else 44.9
        else:
            self.uncertainty = 3.2 if cpx_only else 2.7

        model_rel_path, self._r_model_symbol = self._MODEL_FILES[(T_P, cpx_only)]
        self.r_model_path = get_file_path(jorgenson_data, model_rel_path)

        self.cpx_names = [
            "SiO2_cpx",
            "TiO2_cpx",
            "Al2O3_cpx",
            "Cr2O3_cpx",
            "FeO_cpx",
            "MgO_cpx",
            "MnO_cpx",
            "CaO_cpx",
            "Na2O_cpx",
        ]
        self.liq_names = [
            "SiO2_liq",
            "TiO2_liq",
            "Al2O3_liq",
            "FeO_liq",
            "MgO_liq",
            "MnO_liq",
            "CaO_liq",
            "Na2O_liq",
            "K2O_liq",
        ]

        self.standard_columns = (
            self.cpx_names
            if cpx_only
            else self.cpx_names + self.liq_names
        )


        X_cpx_train_pkl_name = "datapkl/X_cpx_train.pkl"
        X_liq_train_pkl_name = "datapkl/X_liq_train.pkl"
        self.X_cpx_train_pkl_path = Path(get_file_path(jorgenson_data, X_cpx_train_pkl_name))
        self.X_liq_train_pkl_path = Path(get_file_path(jorgenson_data, X_liq_train_pkl_name))


        self.initialize_model(
            cpx_training_path=self.X_cpx_train_pkl_path,
            liq_training_path=self.X_liq_train_pkl_path,
        )

        self._r_model = None
        self._r_ready = False

        self.if_normalize_liq = True
        


    # ------------------------------------------------------------------
    # R runtime helpers
    # ------------------------------------------------------------------
    def _prepare_r_runtime(self) -> None:
        _ensure_rpy2_runtime()
        robjects.r(
            r"""
            options(repos = c(CRAN = "https://cloud.r-project.org"))
            options(java.parameters = "-Xmx4g")

            pack1 <- suppressWarnings(require(PerformanceAnalytics))
            if (!pack1) {
              install.packages("PerformanceAnalytics", dependencies = TRUE, type = "win.binary")
              library(PerformanceAnalytics)
            }

            pack2 <- suppressWarnings(require(rJava))
            if (!pack2) {
              install.packages("rJava", dependencies = TRUE, type = "win.binary")
              library(rJava)
            }

            pack3 <- suppressWarnings(require(extraTrees))
            if (!pack3) {
              install.packages("extraTrees", dependencies = TRUE)
              library(extraTrees)
            }

            pack4 <- suppressWarnings(require(readxl))
            if (!pack4) {
              install.packages("readxl", dependencies = TRUE, type = "win.binary")
              library(readxl)
            }

            pack5 <- suppressWarnings(require(EnvStats))
            if (!pack5) {
              install.packages("EnvStats", dependencies = TRUE, type = "win.binary")
              library(EnvStats)
            }

            rm(pack1, pack2, pack3, pack4, pack5)
            """
        )

    def _load_r_model(self) -> None:
        _ensure_rpy2_runtime()
        robjects.r["load"](str(self.r_model_path))
        self._r_model = robjects.globalenv[self._r_model_symbol]

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def format_input(
        self,
        X_cpx: pd.DataFrame,
        X_liq: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        cpx_norm = normalize_column_names(
            X_cpx,
            standard_names_list=self.cpx_names,
            missing_fill=0,
            drop_missing=False,
        )
        cpx_norm = cpx_norm.rename(columns=self._CPX_R_MAP)
        cpx_norm = cpx_norm.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        cpx_norm = cpx_norm.reindex(columns=list(self._CPX_R_MAP.values()))

        if self.cpx_only:
            cpx_norm.index = X_cpx.index
            return cpx_norm

        if X_liq is None:
            raise ValueError("Liquid compositions are required when cpx_only is False.")

        liq_norm = normalize_column_names(
            X_liq,
            standard_names_list=self.liq_names,
            missing_fill=0,
            drop_missing=False,
        )
        liq_norm = liq_norm.rename(columns=self._LIQ_R_MAP)
        liq_norm = liq_norm.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        liq_norm = liq_norm.reindex(columns=list(self._LIQ_R_MAP.values()))
        liq_norm = _normalize_liquid_anhydrous(liq_norm)

        combined = pd.concat([cpx_norm, liq_norm], axis=1)
        combined.index = X_cpx.index
        return combined

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, X_cpx: pd.DataFrame, X_liq: Optional[pd.DataFrame] = None) -> pd.Series:
        # pandas2ri.activate()
        _ensure_rpy2_runtime()

        X_input = self.format_input(X_cpx, X_liq)

        if not self._r_ready:
            self._prepare_r_runtime()
            self._load_r_model()
            self._r_ready = True

        with localconverter(default_converter + pandas2ri.converter):
            input_r = pandas2ri.py2rpy(X_input)
        pred = robjects.r["predict"](
            self._r_model,
            newdata=input_r,
            allValues=True,
        )

        with localconverter(default_converter + pandas2ri.converter):
            pred_obj = robjects.r["as.data.frame"](pred)
            pred_df = (
                pred_obj
                if isinstance(pred_obj, pd.DataFrame)
                else pandas2ri.rpy2py(pred_obj)
            )
        if isinstance(pred_df, pd.DataFrame):
            medians = pred_df.median(axis=1).to_numpy()
        else:
            pred_array = np.asarray(pred_df, dtype=float)
            if pred_array.ndim == 1:
                medians = pred_array
            else:
                medians = np.median(pred_array, axis=1)

        return pd.Series(medians, index=X_input.index, name=self.prediction_column_name)


if __name__ == "__main__":
    CSV_PATH = Path(get_file_path(jorgenson_data, "data_set/input.csv"))
    DATA_ROOT = CSV_PATH.parent.parent
    PKL_DIR = DATA_ROOT / "datapkl"

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Training CSV missing: {CSV_PATH}")

    raw = _read_training_csv(CSV_PATH)

    PKL_DIR.mkdir(parents=True, exist_ok=True)

    cpx_path = PKL_DIR / "X_cpx_train.pkl"
    liq_path = PKL_DIR / "X_liq_train.pkl"

    _build_cpx_training(raw).to_pickle(cpx_path)
    _build_liq_training(raw).to_pickle(liq_path)
    print(_build_cpx_training(raw).head())
    print(_build_liq_training(raw).head())

    print(f"Generated {cpx_path.relative_to(DATA_ROOT)}")
    print(f"Generated {liq_path.relative_to(DATA_ROOT)}")

    cpx_model_map = {v: k for k, v in Jorgenson22._CPX_R_MAP.items()}
    cpx_model_map.update({"FeOt_cpx": "FeO_cpx"})

    cpx_model_df = raw.rename(columns=cpx_model_map)
    cpx_model_df = _coerce_numeric(cpx_model_df)
    cpx_model_df = cpx_model_df.reindex(columns=Jorgenson22._CPX_NAMES).fillna(0.0)

    temp_targets = pd.to_numeric(raw.get("T"), errors="coerce")
    press_targets = pd.to_numeric(raw.get("P"), errors="coerce")

    for mode, targets in (("T", temp_targets), ("P", press_targets)):
        mask = targets.notna()
        if mask.sum() == 0:
            print(f"No targets available for mode {mode}.")
            continue

        model = Jorgenson22(mode, cpx_only=True)
        preds = model.predict(model.X_cpx_all)
        rmse = float(np.sqrt(np.mean((preds.values - targets.loc[mask].to_numpy()) ** 2)))
        unit = "°C" if mode == "T" else "kbar"
        label = "Temperature" if mode == "T" else "Pressure"
        print(f"Training RMSE ({label}): {rmse:.3f} {unit}")


    for mode, targets in (("T", temp_targets), ("P", press_targets)):
        mask = targets.notna()
        if mask.sum() == 0:
            print(f"No targets available for mode {mode}.")
            continue

        model = Jorgenson22(mode, cpx_only=False)
        preds = model.predict(model.X_cpx_all, model.X_liq_all)
        rmse = float(np.sqrt(np.mean((preds.values - targets.loc[mask].to_numpy()) ** 2)))
        unit = "°C" if mode == "T" else "kbar"
        label = "Temperature" if mode == "T" else "Pressure"
        print(f"Training RMSE ({label}): {rmse:.3f} {unit}")
