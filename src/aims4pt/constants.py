"""Core constants for oxide naming, stoichiometry, and plotting defaults."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import periodictable as pt
from cycler import cycler

# mapping of standard oxide names to regular expressions
STANDARD_OXIDES = [
    "SiO2",
    "Al2O3",
    "FeO",
    "Fe2O3",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "TiO2",
    "MnO",
    "P2O5",
    "Cr2O3",
    "NiO",
    "H2O",
    "CO2",
    "BaO",
    "SrO",
    "ZrO2",
    "SO3",
    "Cl",
    "F",
    "V2O3",
]

anhydrous_oxides = [
    "SiO2",
    "Al2O3",
    "FeO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "TiO2",
    "MnO",
    "P2O5",
    "Cr2O3",
    "NiO",
]  # anhydrous oxides


def _oxide_pattern(oxide: str) -> str:
    """Build a regex pattern that matches the given oxide column name."""

    base_pattern = rf"(^|[\s_\-\.\(\)]){oxide}([\s_\-\.\(\)]|$)"
    if oxide == "FeO":
        return (
            r"(^|[\s_\-\.\(\)])FeO(t|tot|total)?"
            r"([\s_\-\.\(\)]+tot|[\s_\-\.\(\)]+total)?([\s_\-\.\(\)]|$)"
        )
    return base_pattern


OXIDES_MAPPING: Dict[str, str] = {oxide: _oxide_pattern(oxide) for oxide in STANDARD_OXIDES}

OXIDES_MOLE_MASS = pd.Series(
    {
        "SiO2": pt.formula("SiO2").mass,
        "TiO2": pt.formula("TiO2").mass,
        "Al2O3": pt.formula("Al2O3").mass,
        "FeO": pt.formula("FeO").mass,
        "MnO": pt.formula("MnO").mass,
        "MgO": pt.formula("MgO").mass,
        "CaO": pt.formula("CaO").mass,
        "Na2O": pt.formula("Na2O").mass,
        "K2O": pt.formula("K2O").mass,
        "Cr2O3": pt.formula("Cr2O3").mass,
        "P2O5": pt.formula("P2O5").mass,
        "H2O": pt.formula("H2O").mass,
        "CO2": pt.formula("CO2").mass,
        "BaO": pt.formula("BaO").mass,
        "SrO": pt.formula("SrO").mass,
        "ZrO2": pt.formula("ZrO2").mass,
        "SO3": pt.formula("SO3").mass,
        "Cl": pt.formula("Cl").mass,
        "F": pt.formula("F").mass,
        "Fe2O3": pt.formula("Fe2O3").mass,
        "NiO": pt.formula("NiO").mass,
        "V2O3": pt.formula("V2O3").mass,
    }
)

OXIDES_CATION_NUM = pd.Series(
    {
        "SiO2": 1,
        "TiO2": 1,
        "Al2O3": 2,
        "FeO": 1,
        "MnO": 1,
        "MgO": 1,
        "CaO": 1,
        "Na2O": 2,
        "K2O": 2,
        "Cr2O3": 2,
        "P2O5": 2,
        "H2O": 1,
        "CO2": 1,
        "BaO": 1,
        "SrO": 1,
        "ZrO2": 1,
        "SO3": 1,
        "Cl": 1,
        "F": 1,
        "Fe2O3": 3,
        "NiO": 1,
        "V2O3": 3,
    }
)

OXIDES_O_NUM = pd.Series(
    {
        "SiO2": 2,
        "TiO2": 2,
        "Al2O3": 3,
        "FeO": 1,
        "MnO": 1,
        "MgO": 1,
        "CaO": 1,
        "Na2O": 1,
        "K2O": 1,
        "Cr2O3": 3,
        "P2O5": 5,
        "H2O": 1,
        "CO2": 1,
        "BaO": 1,
        "SrO": 1,
        "ZrO2": 2,
        "SO3": 3,
        "Cl": 1,
        "F": 1,
        "Fe2O3": 3,
        "NiO": 1,
        "V2O3": 3,
    }
)

# physical constants
R = 8.314  # J/(K*mol)

REPORT_INFO = False

# color and marker settings for plotting
ERUPTION_COLOR = {
    "HDT": "#ffcc99",  # light yellow
    "YTT": "#ff9999",  # light red
    "MTT": "#99ccff",  # light blue
    "OTT": "#99ff99",  # light green
}

ERUPTION_MARKER = {
    "HDT": "D",  # diamond
    "YTT": "o",  # circle
    "MTT": "s",  # square
    "OTT": "^",  # triangle
}

SAMPLE_MARKER = {
    "14B": "o",  # circle
    "16B": "^",  # triangle
}

SAMPLE_COLOR = {
    "14B": "C0",  # dark red
    "16B": "C1",  # dark blue
}

Other_color = [
    "#FFD700",  # Gold
    "#FF6347",  # Tomato
    "#4682B4",  # SteelBlue
    "#32CD32",  # LimeGreen
    "#FF4500",  # OrangeRed
    "#8A2BE2",  # BlueViolet
    "#FF1493",  # DeepPink
    "#00FA9A",  # MediumSpringGreen
    "#FF8C00",  # DarkOrange
    "#A52A2A",  # Brown
    "#20B2AA",  # LightSeaGreen
    "#D2691E",  # Chocolate
    "#ADFF2F",  # GreenYellow
    "#DC143C",  # Crimson
    "#008B8B",  # DarkCyan
]
Custom_cycle = cycler("color", Other_color)

__all__ = [
    "STANDARD_OXIDES",
    "anhydrous_oxides",
    "OXIDES_MAPPING",
    "OXIDES_MOLE_MASS",
    "OXIDES_CATION_NUM",
    "OXIDES_O_NUM",
    "R",
    "REPORT_INFO",
    "ERUPTION_COLOR",
    "ERUPTION_MARKER",
    "SAMPLE_MARKER",
    "SAMPLE_COLOR",
    "Other_color",
    "Custom_cycle",
]
