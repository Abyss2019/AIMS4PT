"""Illustration constants for manuscript figures."""

from __future__ import annotations


# Shared manuscript figure font sizes.
AXIS_LABEL_SIZE = 14
Y_AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
MODEL_TICK_LABEL_SIZE = 14
PANEL_TITLE_SIZE = 14
TITLE_SIZE = 16
LEGEND_FONT_SIZE = 13


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
    "\u00c1greda-L\u00f3pez et al., 2024 (cpx_only)": "AgL24",
    # cpx-liq
    "Putirka, 2008 eq33_T; eq31_P": "Put08_31",
    "Putirka, 2008 eq33_T; eq31_P (cpx_liq)": "Put08_31",
    "Neave & Putirka, 2017 Pu08_eq33_T; eq1_P": "NP17",
    "Neave & Putirka, 2017 Pu08_eq33_T; eq1_P (cpx_liq)": "NP17",
    "Neave & Putirka, 2017 Pu08_eq33_T; eq1_P (cpx_liq) (cpx_liq)": "NP17",
    "Petrelli et al., 2020 (cpx_liq)": "Pet20",
    "Jorgenson et al., 2022 (cpx_liq)": "Jor22",
    "Chicchi et al., 2023 (cpx_liq)": "Chi23",
    "\u00c1greda-L\u00f3pez et al., 2024 (cpx_liq)": "AgL24",
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
    "\u00c1greda-L\u00f3pez et al., 2024 (cpx_only)": "AgL24",
    # cpx-liq
    "Putirka, 2008 eq33_T; eq31_P": "Put08_33",
    "Putirka, 2008 eq33_T; eq31_P (cpx_liq)": "Put08_33",
    "Brugman & Till, 2019 (cpx_liq)": "BT19",
    "Petrelli et al., 2020 (cpx_liq)": "Pet20",
    "Jorgenson et al., 2022 (cpx_liq)": "Jor22",
    "Chicchi et al., 2023 (cpx_liq)": "Chi23",
    "\u00c1greda-L\u00f3pez et al., 2024 (cpx_liq)": "AgL24",
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

    abbreviation = MODEL_ABBREVIATIONS_BY_TARGET[T_P_key].get(model_name)
    if abbreviation is not None:
        return abbreviation

    normalized_name = str(model_name).lower()
    if "greda" in normalized_name and "2024" in normalized_name:
        return "AgL24"

    return model_name


__all__ = [
    "AXIS_LABEL_SIZE",
    "Y_AXIS_LABEL_SIZE",
    "TICK_LABEL_SIZE",
    "MODEL_TICK_LABEL_SIZE",
    "PANEL_TITLE_SIZE",
    "TITLE_SIZE",
    "LEGEND_FONT_SIZE",
    "PRESSURE_MODEL_ABBREVIATIONS",
    "TEMPERATURE_MODEL_ABBREVIATIONS",
    "MODEL_ABBREVIATIONS_BY_TARGET",
    "get_model_abbreviation",
]
