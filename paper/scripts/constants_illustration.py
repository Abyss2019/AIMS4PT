"""Illustration constants for manuscript figures."""

from __future__ import annotations


# Shared manuscript figure font sizes.
AXIS_LABEL_SIZE = 17
Y_AXIS_LABEL_SIZE = 17
TICK_LABEL_SIZE = 14
MODEL_TICK_LABEL_SIZE = 16
PANEL_TITLE_SIZE = 17
TITLE_SIZE = 18
LEGEND_FONT_SIZE = 13


# Model-name to Table 1 abbreviation maps. Pressure and temperature are kept
# separate because iterative models share the same model_name for paired P-T
# equations while Table 1 assigns different abbreviations to the P and T sides.
PRESSURE_MODEL_ABBREVIATIONS = {
    # cpx-only
    "Putirka, 2008 eq32d_T; eq32a_P": "Pu08_32a",
    "Putirka, 2008 eq32d_T; eq32b_P": "Pu08_32b",
    "Putirka, 2008 eq32d_T_hydrousVersion; eq32b_P": "Pu08_32b",
    "Petrelli et al., 2020 (cpx_only)": "Pet20",
    "Wang et al., 2021 (cpx_only)": "Wan21",
    "Higgins et al., 2021 (cpx_only)": "Hig21",
    "Jorgenson et al., 2022 (cpx_only)": "Jor22",
    "Chicchi et al., 2023 (cpx_only)": "Chi23",
    "\u00c1greda-L\u00f3pez et al., 2024 (cpx_only)": "AgL24",
    # cpx-liq
    "Putirka, 2008 eq33_T; eq31_P": "Pu08_31",
    "Putirka, 2008 eq33_T; eq31_P (cpx_liq)": "Pu08_31",
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
    "Putirka, 2008 eq32d_T; eq32a_P": "Pu08_32d",
    "Putirka, 2008 eq32d_T; eq32b_P": "Pu08_32d",
    "Putirka, 2008 eq32d_T_hydrousVersion; eq32b_P": "Pu08_32d",
    "Petrelli et al., 2020 (cpx_only)": "Pet20",
    "Wang et al., 2021 (cpx_only)": "Wan21",
    "Higgins et al., 2021 (cpx_only)": "Hig21",
    "Jorgenson et al., 2022 (cpx_only)": "Jor22",
    "Chicchi et al., 2023 (cpx_only)": "Chi23",
    "\u00c1greda-L\u00f3pez et al., 2024 (cpx_only)": "AgL24",
    # cpx-liq
    "Putirka, 2008 eq33_T; eq31_P": "Pu08_33",
    "Putirka, 2008 eq33_T; eq31_P (cpx_liq)": "Pu08_33",
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
    normalized_name = normalized_name.replace(" ", "")
    normalized_name = normalized_name.replace("(cpx_only)", "").replace("(cpx_liq)", "")
    normalized_name = normalized_name.removesuffix("_p").removesuffix("_t")

    if "putirka" in normalized_name and "2008" in normalized_name:
        if T_P_key == "P":
            if "eq32a_p" in normalized_name or "eq32a" in normalized_name:
                return "Pu08_32a"
            if "eq32b_p" in normalized_name or "eq32b" in normalized_name:
                return "Pu08_32b"
            if "eq31_p" in normalized_name or "eq31" in normalized_name:
                return "Pu08_31"
        if T_P_key == "T":
            if "eq32d_t" in normalized_name or "eq32d" in normalized_name:
                return "Pu08_32d"
            if "eq33_t" in normalized_name or "eq33" in normalized_name:
                return "Pu08_33"

    if "neave" in normalized_name and "putirka" in normalized_name:
        return "NP17"

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
