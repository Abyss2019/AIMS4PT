"""Illustration constants for manuscript figures."""

from __future__ import annotations


# Model-name to Table 1 abbreviation maps. Pressure and temperature are kept
# separate because iterative models share the same model_name for paired P-T
# equations while Table 1 assigns different abbreviations to the P and T sides.
PRESSURE_MODEL_ABBREVIATIONS = {
    # cpx-only
    "Putirka, 2008 eq32d_T; eq32a_P": "Put08_32a",
    "Putirka, 2008 eq32d_T; eq32b_P": "Put08_32b",
    "Putirka, 2008 eq32d_T_hydrousVersion; eq32b_P": "Put08_32b",
    "Petrelli et al., 2020 (cpx_only)": "Pet20",
    "Wang et al., 2021 (cpx_only)": "Wan21",
    "Higgins et al., 2021 (cpx_only)": "Hig21",
    "Jorgenson et al., 2022 (cpx_only)": "Jor22",
    "Chicchi et al., 2023 (cpx_only)": "Chi23",
    "Ágreda-López et al., 2024 (cpx_only)": "AgL24",
    # cpx-liq
    "Putirka, 2008 eq33_T; eq31_P": "Put08_31",
    "Neave & Putirka, 2017 Pu08_eq33_T; eq1_P": "NP17",
    "Petrelli et al., 2020 (cpx_liq)": "Pet20",
    "Jorgenson et al., 2022 (cpx_liq)": "Jor22",
    "Chicchi et al., 2023 (cpx_liq)": "Chi23",
    "Ágreda-López et al., 2024 (cpx_liq)": "AgL24",
}

TEMPERATURE_MODEL_ABBREVIATIONS = {
    # cpx-only
    "Putirka, 2008 eq32d_T; eq32a_P": "Put08_32d",
    "Putirka, 2008 eq32d_T; eq32b_P": "Put08_32d",
    "Putirka, 2008 eq32d_T_hydrousVersion; eq32b_P": "Put08_32d",
    "Petrelli et al., 2020 (cpx_only)": "Pet20",
    "Wang et al., 2021 (cpx_only)": "Wan21",
    "Higgins et al., 2021 (cpx_only)": "Hig21",
    "Jorgenson et al., 2022 (cpx_only)": "Jor22",
    "Chicchi et al., 2023 (cpx_only)": "Chi23",
    "Ágreda-López et al., 2024 (cpx_only)": "AgL24",
    # cpx-liq
    "Putirka, 2008 eq33_T; eq31_P": "Put08_33",
    "Brugman & Till, 2019 (cpx_liq)": "BT19",
    "Petrelli et al., 2020 (cpx_liq)": "Pet20",
    "Jorgenson et al., 2022 (cpx_liq)": "Jor22",
    "Chicchi et al., 2023 (cpx_liq)": "Chi23",
    "Ágreda-López et al., 2024 (cpx_liq)": "AgL24",
}

MODEL_ABBREVIATIONS_BY_TARGET = {
    "P": PRESSURE_MODEL_ABBREVIATIONS,
    "T": TEMPERATURE_MODEL_ABBREVIATIONS,
}


def get_model_abbreviation(model_name: str, T_P: str) -> str:
    """Return the Table 1 abbreviation for a model name and T_P."""
    T_P_key = T_P.upper()
    if T_P_key not in MODEL_ABBREVIATIONS_BY_TARGET:
        raise ValueError("T_P must be 'P' or 'T'.")
    return MODEL_ABBREVIATIONS_BY_TARGET[T_P_key].get(model_name, model_name)


__all__ = [
    "PRESSURE_MODEL_ABBREVIATIONS",
    "TEMPERATURE_MODEL_ABBREVIATIONS",
    "MODEL_ABBREVIATIONS_BY_TARGET",
    "get_model_abbreviation",
]
