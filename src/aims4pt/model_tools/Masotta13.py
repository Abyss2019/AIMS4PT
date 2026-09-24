"""Masotta et al. (2013) thermobarometers for alkaline differentiated melts.

Only Talk2012 and Palk2012 are implemented, using the equations on p. 1554
and full-precision coefficients from 410_2013_927_MOESM2_ESM.xlsx,
"Alkaline thermo-barometers", D51:L51 and D53:H53.
DOI: 10.1007/s00410-013-0927-9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import aims4pt.model_tools.data.Masotta13 as Masotta13_data
from aims4pt.data_tools.compositions import calculate_cation_fractions, cpx_calculation
from aims4pt.model_tools.conventional_model import conventional_model, iterative_model
from aims4pt.model_tools.model_registry import register_model
from aims4pt.toolkit_utils import get_file_path
from aims4pt.utils import get_oxides_list


_CPX_OXIDES = [
    "SiO2", "TiO2", "Al2O3", "FeOt", "MnO", "MgO", "CaO", "Na2O", "K2O"
]
_LIQ_OXIDES = _CPX_OXIDES + ["P2O5", "H2O"]
_STANDARD_COLUMNS = [f"{oxide}_cpx" for oxide in _CPX_OXIDES] + [
    f"{oxide}_liq" for oxide in _LIQ_OXIDES
]


def _components(model, X, X_liq):
    """Calculate paired cpx components and anhydrous liquid cation fractions."""
    if X_liq is None:
        raise ValueError("Masotta13 requires paired clinopyroxene and liquid compositions.")
    if not X.index.equals(X_liq.index):
        raise ValueError("Clinopyroxene and liquid inputs must have matching indices.")
    X_cpx, liquid = model.process_input(X, X_liq)
    cpx = cpx_calculation(X_cpx)
    oxides = get_oxides_list(liquid.columns.tolist())
    oxides.remove("H2O")
    liq = calculate_cation_fractions(liquid, oxides)
    # Use total Fe as FeO, as in the author's Excel KD calculation.
    kd = (cpx["FeO_6OBasis"] / cpx["MgO_6OBasis"]) / (liq["FeO"] / liq["MgO"])
    kjd = cpx["Jd"] / (liq["Na2O"] * liq["Al2O3"] * liq["SiO2"] ** 2)
    return cpx, liq, liquid["H2O_liq"], kd, kjd


class Talk2012(conventional_model):
    """Pressure-independent cpx-liquid thermometer; output in degrees Celsius."""

    def __init__(self):
        parameters_dict = {
            "b": 2.90815635794002,
            "w_ln_exchange": -0.400827676578132,
            "w_H2O": 0.0375720784518263,
            "w_Mg_number_DiHd": -1.6383282971929,
            "w_Na_number": 1.01129776262724,
            "w_ln_Ti": -0.21766733252629,
            "w_ln_KJd": 0.466149612620683,
            "w_Kd_FeMg": 1.61626798988239,
            "w_CaSi": 23.3855047471225,
        }
        super().__init__("Masotta et al., 2013 Talk2012", parameters_dict, _STANDARD_COLUMNS)
        self.uncertainty = 18.2  # SEE in degrees Celsius.
        self.cpx_only = False
        self.if_support_hydrous = True
        self.require_water = True
        self.X_cpx_train_pkl_path = get_file_path(Masotta13_data, "datapkl/T/X_cpx_train.pkl")
        self.X_liq_train_pkl_path = get_file_path(Masotta13_data, "datapkl/T/X_liq_train.pkl")

    def predict(self, X, X_liq=None):
        cpx, liq, water, kd, kjd = _components(self, X, X_liq)
        p = self.parameters_dict
        fm = liq["FeO"] + liq["MgO"]
        denominator = (
            p["b"]
            + p["w_ln_exchange"] * np.log(
                cpx["Jd"] * liq["CaO"] * fm
                / (cpx["DiHd"] * liq["Na2O"] * liq["Al2O3"])
            )
            + p["w_H2O"] * water
            + p["w_Mg_number_DiHd"] * (liq["MgO"] / fm) / cpx["DiHd"]
            + p["w_Na_number"] * liq["Na2O"] / (liq["Na2O"] + liq["K2O"])
            + p["w_ln_Ti"] * np.log(liq["TiO2"])
            + p["w_ln_KJd"] * np.log(kjd)
            + p["w_Kd_FeMg"] * kd
            + p["w_CaSi"] * liq["CaO"] * liq["SiO2"]
        )
        return (10000 / denominator - 273.15).rename("T_C")


class Palk2012(conventional_model):
    """Temperature-independent cpx-liquid barometer; output in kbar."""

    def __init__(self):
        parameters_dict = {
            "b": -3.88903951262765,
            "w_KJd": 0.277651046511846,
            "w_H2O": 0.0740292491471828,
            "w_Na_number": 5.00912129248619,
            "w_Kd_FeMg": 6.39451438456963,
        }
        super().__init__("Masotta et al., 2013 Palk2012", parameters_dict, _STANDARD_COLUMNS)
        self.uncertainty = 1.15  # SEE in kbar.
        self.cpx_only = False
        self.if_support_hydrous = True
        self.require_water = True
        self.X_cpx_train_pkl_path = get_file_path(Masotta13_data, "datapkl/P/X_cpx_train.pkl")
        self.X_liq_train_pkl_path = get_file_path(Masotta13_data, "datapkl/P/X_liq_train.pkl")

    def predict(self, X, X_liq=None):
        _, liq, water, kd, kjd = _components(self, X, X_liq)
        p = self.parameters_dict
        # Palk2012 uses the Jd ratio itself, not its logarithm.
        pressure = (
            p["b"]
            + p["w_KJd"] * kjd
            + p["w_H2O"] * water
            + p["w_Na_number"] * liq["Na2O"] / (liq["Na2O"] + liq["K2O"])
            + p["w_Kd_FeMg"] * kd
        )
        return pressure.rename("P_kbar")


@register_model
class Masotta13(iterative_model):
    """Iterate Talk2012 and Palk2012 with output-specific calibration data.

    Inputs are paired oxide wt.% DataFrames with matching indices and the
    standard *_cpx / *_liq columns. H2O_liq is dissolved water in wt.%.
    Missing oxides/water follow conventional_model's zero-fill convention.
    Both equations are explicit and do not require input pressure/temperature.
    The shared iteration therefore reaches a fixed point after updating T/P.
    """

    cpx_only_models = {}
    cpx_liq_models = {
        "T_models": {"hydrous": [], "anhydrous": [], "both": ["Talk2012"]},
        "P_models": {"hydrous": [], "anhydrous": [], "both": ["Palk2012"]},
    }
    model_options = {"Talk2012": Talk2012, "Palk2012": Palk2012}
    model_name_str = "Masotta et al., 2013"

    def __init__(self, T_P="T", comments=None, *, T_model_name="Talk2012",
                 P_model_name="Palk2012", iteration_max=50, stop_criteria=1e-3):
        if T_P not in ("T", "P"):
            raise ValueError("T_P must be 'T' or 'P'.")
        if T_model_name != "Talk2012" or P_model_name != "Palk2012":
            raise ValueError("Masotta13 requires T_model_name='Talk2012' and P_model_name='Palk2012'.")
        if not isinstance(iteration_max, (int, np.integer)) or iteration_max < 1:
            raise ValueError("iteration_max must be a positive integer.")
        if not np.isfinite(stop_criteria) or stop_criteria < 0:
            raise ValueError("stop_criteria must be finite and non-negative.")
        super().__init__(T_P, T_model_name, P_model_name, iteration_max, stop_criteria, comments)

    def initialize_model(self, cpx_training_path=None, cpx_test_path=None,
                         liq_training_path=None, liq_test_path=None):
        """Keep stable feature order and existing output-specific artifact names."""
        model = self.T_model if self.T_P == "T" else self.P_model
        self.standard_columns = list(_STANDARD_COLUMNS)
        self.model_name = model.model_name
        self.cpx_names = model.cpx_names
        self.liq_names = model.liq_names
        self.prediction_column_name = "T_C" if self.T_P == "T" else "P_kbar"
        self.uncertainty = model.uncertainty
        self.X_cpx_train_pkl_path = model.X_cpx_train_pkl_path
        self.X_liq_train_pkl_path = model.X_liq_train_pkl_path
        super().initialize_model(
            cpx_training_path=(self.X_cpx_train_pkl_path if cpx_training_path is None
                               else cpx_training_path),
            cpx_test_path=cpx_test_path,
            liq_training_path=(self.X_liq_train_pkl_path if liq_training_path is None
                               else liq_training_path),
            liq_test_path=liq_test_path,
        )

    def switch_PT(self, T_P):
        """Switch output and reload the matching calibration and diagnostics."""
        super().switch_PT(T_P)
        self.initialize_model()


def main():
    """Print the ten original paired examples against the author's Excel."""
    # Source: 410_2013_927_MOESM2_ESM.xlsx, "Alkaline thermo-barometers".
    # Inputs: D4:L13 (cpx), D18:M27 (liquid), P18:P27 (water).
    samples = [
        "MB-22", "MB-43", "MB-44", "MB-57", "MB-58",
        "MB-111", "MB-112", "MB-113", "MB-114", "MB-117",
    ]
    X_cpx = pd.DataFrame([
        [52.3, 0.86, 0.85, 11.3, 1.14, 11.9, 20.8, 0.92, 0],
        [51.6, 1.35, 1.82, 11.4, 0.99, 10.5, 20.8, 1.54, 0.1],
        [50.2, 2.13, 2.45, 12.3, 1.03, 9.6, 20.5, 1.68, 0],
        [50.6, 0.96, 2.13, 14.9, 1.3, 7.2, 19.9, 1.7, 0.3],
        [50, 1.11, 2.79, 14.5, 1.25, 7.1, 18.6, 2.04, 0.44],
        [50.3, 1.38, 2.25, 12.9, 1.04, 10.1, 20.4, 1.66, 0],
        [50.9, 1.02, 1.31, 12.5, 0.94, 10.7, 21.3, 1.27, 0],
        [50.5, 1, 1.65, 13.1, 1.11, 10.7, 21.3, 0.71, 0],
        [50.9, 1.27, 1.88, 12, 0.95, 11, 21.2, 0.74, 0],
        [49.9, 1.1, 2.34, 11.6, 1.46, 11.3, 21.4, 0.86, 0],
    ], columns=[f"{oxide}_cpx" for oxide in _CPX_OXIDES], index=samples)
    X_liq = pd.DataFrame([
        [60.7, 0.67, 19.4, 3.1, 0.16, 0.28, 0.6, 9.8, 5.34, 0],
        [60.74, 0.66, 19.7, 3.09, 0.19, 0.28, 0.61, 9.4, 5.4, 0],
        [58.9, 0.62, 20.4, 3.18, 0.22, 0.32, 0.78, 10.4, 5.24, 0],
        [61.5, 0.31, 19.4, 2.67, 0.15, 0.18, 0.45, 10.1, 5.21, 0],
        [60.3, 0.47, 20.2, 2.36, 0.25, 0.18, 0.54, 10.5, 5.23, 0],
        [59.9, 0.7, 19, 2.96, 0.21, 0.32, 0.66, 10.6, 5.7, 0],
        [59.5, 0.7, 19.3, 3.02, 0.18, 0.33, 0.65, 10.8, 5.6, 0],
        [59.6, 0.6, 19, 3.11, 0.22, 0.29, 0.68, 10.8, 5.7, 0],
        [60, 0.7, 19.7, 3.05, 0.13, 0.36, 0.63, 9.8, 5.7, 0],
        [59.4, 0.7, 19.4, 3.24, 0.31, 0.37, 0.65, 10.2, 5.8, 0],
    ], columns=[f"{oxide}_liq" for oxide in _CPX_OXIDES + ["P2O5"]], index=samples)
    X_liq["H2O_liq"] = [3.12, 2.79, 2.55, 5.53, 4.79, 4.56, 3.83, 3.31, 2.79, 1.98]

    # Cached paired outputs: AN120:AN129 (T) and AR120:AR129 (P).
    results = pd.DataFrame({
        "T_C": Masotta13("T").predict(X_cpx, X_liq),
        "Excel_T_C": [
            854.0101382912954, 848.3673580797312, 854.0522493443615,
            774.4650184746052, 788.827766291293, 842.5437623923907,
            854.5601407741309, 851.8199174578755, 857.1394354676166,
            870.4680046352297,
        ],
        "P_kbar": Masotta13("P").predict(X_cpx, X_liq),
        "Excel_P_kbar": [
            1.0058776498399888, 1.6650927748021367, 1.7016220181145112,
            3.114767834776881, 3.8298606951613783, 1.771034276946888,
            1.1006826078390843, 1.0074934633565027, 1.3758084289726762,
            0.9668306942588729,
        ],
    }, index=samples)
    results["Delta_T_C"] = results["T_C"] - results["Excel_T_C"]
    results["Delta_P_kbar"] = results["P_kbar"] - results["Excel_P_kbar"]
    print("Full-precision Excel coefficients. Delta = Python - Excel.")
    print(results.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
