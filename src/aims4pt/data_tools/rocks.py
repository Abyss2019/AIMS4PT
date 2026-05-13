# aims4pt/data_tools/rocks.py
import pandas as pd
import numpy as np
from typing import List, Dict, Union

from aims4pt.utils import normalize_column_names

def get_TAS_rock_types(df, mode = "unique", model="LeMaitreCombined", map_rock=None, return_full_name=True):
    """
    Predict TAS-based rock types from SiO2 and Na2O+K2O data. (already normalized)

    Parameters
    ----------
        df : pd.DataFrame or pd.Series
            DataFrame with columns including 'SiO2', 'Na2O', and 'K2O'.
        mode : str, optional
            Mode of operation, either "unique" to return unique rock types or "all" to
            return all predicted rock types. Default is "unique".
            And "statistics" to return statistics of rock types.
        model : str, optional
            TAS classification model to use. Default is "LeMaitreCombined".
        map_rock : dict, optional
            Mapping from TAS codes to full rock type names.
            If None, will use a default mapping.
        return_full_name : bool, optional
            If True, returns full rock type names. If False, returns TAS codes.

    Returns
    -------
        list [str] or dict
            If mode is "unique" or "all", returns a list of TAS codes or full names.
            If mode is "statistics", returns a dictionary with rock types as keys and their counts as values.
            List of predicted rock types (TAS codes or full names).

    """

    from pyrolite.util.classification import TAS
    if isinstance(df, pd.Series):
        df = df.to_frame().T

    # Default mapping dictionary.
    if map_rock is None:
        map_rock = {
            "Pc": "Picrite", "B": "Basalt", "O1": "Basaltic Andesite",
            "O2": "Andesite", "O3": "Dacite", "R": "Rhyolite",
            "F": "Foidite", "T1T2": "Trachyte", "S1": "Trachybasalt",
            "S2": "Basaltic Trachyandesite", "S3": "Trachyandesite",
            "U1": "Tephrite basanite", "U2": "Phonotephrite",
            "U3": "Tephriphonolite", "U4": "Phonolite", "Ph": "Phonolite"
        }

    # Copy and normalize column names.
    if df is None or df.empty:
        return []
    df = df.copy()

    from aims4pt.utils import normalize_column_names
    df = normalize_column_names(df, ["SiO2", "Na2O", "K2O"])

    # Calculate Na2O + K2O.
    df["Na2O + K2O"] = df["Na2O"] + df["K2O"]

    # Predict with the TAS classifier.
    cm = TAS(which_model=model)
    tas_codes = cm.predict(df[["SiO2", "Na2O + K2O"]])

    if mode == "unique":
        tas_codes = set(tas_codes)  # Remove duplicates.
    elif mode == "all":
        tas_codes = tas_codes.values
    elif mode == "statistics":
        # Count each rock type.
        tas_counts = pd.Series(tas_codes).value_counts()
        return tas_counts.to_dict()

    if return_full_name:
        return [map_rock.get(code, code) for code in tas_codes]
    else:
        return tas_codes.tolist()


import numpy as np
import pandas as pd
from typing import Union, List, Dict

def get_volcanic_rock_series(
    df: Union[pd.DataFrame, pd.Series],
    mode: str = "unique",
    major: bool = False
) -> Union[List[str], Dict[str, int]]:
    """
    Return the classification results for volcanic rock series.
    Series include:
    - Alkaline
    - Calc-alkaline
    - Tholeiitic

    Reference:
    - Irvine, T. N., & Baragar, W. R. A. (1971). A Guide to the Chemical Classification of the Common Volcanic Rocks. Canadian Journal of Earth Sciences, 8(5), 523–548. https://doi.org/10.1139/e71-055

    - Miyashiro, A. (1974). Volcanic rock series in island arcs and active continental margins. American Journal of Science, 274(4), 321–355. https://doi.org/10.2475/ajs.274.4.321


    Parameters
    ----------
        df : pd.DataFrame or pd.Series
            DataFrame containing at least 'SiO2', 'FeO', and 'MgO'.
        mode : str, optional
            Operation mode:
            - "unique": return a list of unique series
            - "all": return a list corresponding to each sample
            - "statistics": return a dict of series counts
            Default is "unique".
        major : bool, optional
            If True and mode is "unique", return only series with >90% occurrence.

    Returns
    -------
        Union[List[str], Dict[str, int]]
            Depending on mode, returns a list of series or a statistics dict.
    """
    # Return empty if input is None or empty
    if df is None or df.empty:
        return []
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    # Make a copy to avoid modifying original
    df = df.copy()

    # 1. Normalize column names and fill missing columns with 0, then convert Fe2O3 to FeO
    df = normalize_column_names(
        df,
        ["SiO2", "FeO", "MgO", "Fe2O3", "Na2O", "K2O"],
        missing_fill=0
    )
    df["FeO"] = df["FeO"] + df["Fe2O3"] * 0.899

    # 2. TAS boundary (Alkaline vs Subalkaline) Ishida & Izu (1987) Irvine & Baragar (1971)
    A = df["Na2O"].fillna(0) + df["K2O"].fillna(0)
    boundary_poly = (
        -3.3539e-4 * A**6
        + 1.2030e-2 * A**5
        - 1.5188e-1 * A**4
        + 8.6096e-1 * A**3
        - 2.1111    * A**2
        + 3.9492    * A
        + 39.0
    )
    is_subalkaline = df["SiO2"] >= boundary_poly

    # 3. FeO*/MgO vs SiO2 boundary (Tholeiitic vs Calc-alkaline) Miyashiro (1974)
    fe_mg_ratio = df["FeO"].div(df["MgO"].replace(0, np.nan)).fillna(0)
    is_calc_alkaline = df["SiO2"] >= 6.4 * fe_mg_ratio + 42.8

    # 4. Determine series for each sample: alkaline, calc-alkaline, tholeiitic, or unknown
    conditions = [
        ~is_subalkaline,
        is_subalkaline & is_calc_alkaline,
        is_subalkaline & ~is_calc_alkaline,
    ]
    choices = ["alkaline", "calc-alkaline", "tholeiitic"]
    series_array = np.select(conditions, choices, default="unknown")

    # 5. Return based on mode
    if mode == "all":
        return series_array.tolist()

    # Compute normalized counts if unique, else absolute counts for statistics
    normalized = (mode == "unique")
    counts = pd.Series(series_array).value_counts(normalize=normalized)

    if mode == "statistics":
        return pd.Series(series_array).value_counts().to_dict()

    # mode == "unique"
    unique_series = counts.index.tolist()
    if major:
        # Keep only series with proportion > 0.5
        unique_series = counts[counts > 0.5].index.tolist()
    return unique_series


if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        "SiO2": [45, 52, 58, 63, 70, 75],
        "Na2O": [2.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        "K2O": [0.5, 1.0, 1.2, 1.5, 2.0, 2.5]
    })
    df = pd.DataFrame(data)
    rock_types = get_TAS_rock_types(df)
    print(rock_types)  # Output: ['Basalt', 'Andesite', 'Dacite']

