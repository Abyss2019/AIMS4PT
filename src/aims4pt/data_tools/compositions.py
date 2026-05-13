# aims4pt/data_tools/compositions.py
'''
A collection of functions for data engineering and transformation.

Functions:

calculate_cation_proportions: Calculate cation proportions from the oxide compositions.

calculate_cation_fractions: Calculate cation fractions from the oxide compositions.

calculate_cation_nO_basis: Calculate the n-oxide based cation numbers from the oxide compositions.

cpx_calculation: Calculate cpx end-member components.


'''



from typing import Dict, Iterable, Tuple, Union, Optional

import numpy as np
import pandas as pd

from aims4pt.constants import OXIDES_CATION_NUM, OXIDES_MOLE_MASS, OXIDES_O_NUM
from aims4pt.utils import get_oxides_list, normalize_column_names


CompositionsType = Union[pd.DataFrame, pd.Series, Dict[str, float]]


def _coerce_compositions(compositions: CompositionsType) -> Tuple[pd.DataFrame, type]:
    """Return a DataFrame representation and remember the original type."""
    if not isinstance(compositions, (pd.DataFrame, pd.Series, dict)):
        raise TypeError("compositions must be a pd.DataFrame, pd.Series, or dict.")

    original_type = type(compositions)
    if isinstance(compositions, pd.Series):
        return pd.DataFrame(compositions).T, original_type
    if isinstance(compositions, dict):
        return pd.DataFrame(compositions), original_type
    return compositions.copy(), original_type


def calculate_oxides_proportions(
    compositions: CompositionsType,
    oxides_list: Iterable[str],
    add_suffix: bool = False,
) -> CompositionsType:
    """Calculate the proportions of oxides from the compositions.
    NO WATER!!!
    """

    compositions_df, original_type = _coerce_compositions(compositions)

    # Normalize column names
    compositions_df = normalize_column_names(compositions_df, oxides_list)
    
    # Calculate oxides proportions
    oxide_proportions = compositions_df[oxides_list] / OXIDES_MOLE_MASS[oxides_list]
    if add_suffix:
        oxide_proportions.columns = [f"{col}_proportion" for col in oxide_proportions.columns]

    if original_type is pd.Series:
        return oxide_proportions.iloc[0]
    if original_type is dict:
        return oxide_proportions.iloc[0].to_dict()
    return oxide_proportions




def calculate_cation_fractions(
    compositions: CompositionsType,
    oxides_list: Iterable[str],
    add_suffix: bool = False,
) -> CompositionsType:
    """Calculate cation fractions from oxide compositions.
    For liquid compositions, the cation fractions are usually calculated on a 100% anhydrous basis
    """

    compositions_df, original_type = _coerce_compositions(compositions)
    compositions_df = normalize_column_names(compositions_df, oxides_list)

    oxides_proportions = calculate_oxides_proportions(
        compositions_df, oxides_list, add_suffix=False
    )

    cation_fractions = oxides_proportions * OXIDES_CATION_NUM[oxides_list]
    cation_sum = cation_fractions.sum(axis=1)
    cation_fractions = cation_fractions.div(cation_sum, axis=0)

    if add_suffix:
        cation_fractions.columns = [f"{col}_cation" for col in cation_fractions.columns]

    if original_type is pd.Series:
        return cation_fractions.iloc[0]
    if original_type is dict:
        return cation_fractions.iloc[0].to_dict()
    return cation_fractions


# Generic function for calculating n-oxygen-based compositions.
def calculate_cation_nO_basis(
    compositions: CompositionsType,
    n_oxygens: Union[int, str],
    oxides_list: Iterable[str],
    add_suffix: Optional[str] = None,
) -> CompositionsType:
    """
    Calculate the n-oxide based cation numbers from the oxide compositions.

    Parameters:
        compositions (pd.DataFrame or dict or pd.Series):
            The oxide compositions.
        n_oxygens (str): 
            The number of oxygen atoms in the formula.
        oxides_list (list): 
            List of oxides to use in the calculation.
        add_suffix (str):
            Whether to add a suffix to the column names. Default is None, which means no suffix.
            A string can be provided to add a custom suffix to the column names.

    Returns:
    pd.DataFrame: The n-oxide based compositions.
    """

    compositions_df, original_type = _coerce_compositions(compositions)

    compositions_df = normalize_column_names(compositions_df, oxides_list)

    # proportion
    oxides_proportion = calculate_oxides_proportions(
        compositions_df, oxides_list, add_suffix=False
    )

    oxygen_totals = oxides_proportion[oxides_list] * OXIDES_O_NUM[oxides_list]
    n_oxide_sum = oxygen_totals.sum(axis=1)

    scale_factor = (pd.to_numeric(pd.Series(n_oxygens, index=n_oxide_sum.index)) / n_oxide_sum).values[:, np.newaxis]

    cation_nO = oxides_proportion * scale_factor * OXIDES_CATION_NUM[oxides_list]

    if add_suffix:
        cation_nO.columns = [f"{col}_{add_suffix}" for col in cation_nO.columns]

    if original_type is pd.Series:
        return cation_nO.iloc[0]
    if original_type is dict:
        return cation_nO.iloc[0].to_dict()
    return cation_nO


def cpx_calculation(X_cpx: CompositionsType) -> CompositionsType:
    """Calculate clinopyroxene end-member components."""

    X_cpx_df, original_type = _coerce_compositions(X_cpx)

    oxide_list = get_oxides_list(X_cpx_df.columns.tolist())

    # Normalize column names
    X_cpx_df = normalize_column_names(X_cpx_df, oxide_list)

    X_cpx_6_oxide = calculate_cation_nO_basis(X_cpx_df, 6, oxide_list)

    # cpx end-member components Putirka, 2008 version if not sp
    X_cpx_Al4 = np.maximum(0.0, 2 - X_cpx_6_oxide["SiO2"])
    X_cpx_Al6 = np.maximum(0.0, X_cpx_6_oxide["Al2O3"] - X_cpx_Al4)
    X_cpx_Fe3_Papike_wang = X_cpx_6_oxide["Na2O"] + X_cpx_Al4 - X_cpx_Al6 - 2 * X_cpx_6_oxide["TiO2"] - X_cpx_6_oxide["Cr2O3"] # Specifically for Wang.
    X_cpx_Fe3_Papike = np.maximum(0.0, X_cpx_6_oxide["Na2O"] + X_cpx_Al4 - X_cpx_Al6 - 2 * X_cpx_6_oxide["TiO2"] - X_cpx_6_oxide["Cr2O3"])  # Papike te al. 1974
    X_cpx_Fe2_Papike = np.maximum(0.0, X_cpx_6_oxide["FeO"] - X_cpx_Fe3_Papike)
    X_cpx_Fe2_Papike_wang = X_cpx_6_oxide["FeO"] - X_cpx_Fe3_Papike_wang

    X_cpx_Jd = np.minimum(X_cpx_Al6, X_cpx_6_oxide["Na2O"])
    X_cpx_Jd = np.maximum(X_cpx_Jd, 0.0)
    X_cpx_CaTs = np.maximum(X_cpx_Al6 - X_cpx_Jd, 0.0)
    X_cpx_CaTi = np.maximum((X_cpx_Al4 - X_cpx_CaTs) / 2, 0.0)
    X_cpx_CrCaTs = X_cpx_6_oxide["Cr2O3"] / 2
    X_cpx_DiHd = np.maximum(X_cpx_6_oxide["CaO"] - X_cpx_CaTs - X_cpx_CaTi - X_cpx_CrCaTs, 0.0) # Putirka (2008)
    X_cpx_DiHd_2003 = X_cpx_6_oxide["CaO"] - X_cpx_CaTs - X_cpx_CaTi - X_cpx_CrCaTs  # Putirka (2003)
    X_cpx_EnFs = np.maximum((X_cpx_6_oxide["FeO"] + X_cpx_6_oxide["MgO"] - X_cpx_DiHd) / 2, 0.0)

    # Nimis (following Putirka's sheet)
    CNM = X_cpx_6_oxide["CaO"] + X_cpx_6_oxide["Na2O"] + X_cpx_6_oxide["MnO"]
    R3 = X_cpx_Al6 + X_cpx_Fe3_Papike + X_cpx_6_oxide["TiO2"] + X_cpx_6_oxide["Cr2O3"]
    kd_FeMg = np.exp(0.238 * R3 + 0.289 * CNM - 2.3315)
    # Fe_Mg_M1 = 1- X_cpx_Al6 - X_cpx_6_oxide["TiO2"]
    Fe2_ = X_cpx_Fe2_Papike
    # Fe_Mg_M2 = Fe2_ + X_cpx_6_oxide["MgO"]- Fe_Mg_M1
    a = 1-kd_FeMg
    b= kd_FeMg*X_cpx_6_oxide["MgO"] - kd_FeMg*(1-CNM) + Fe2_ + (1-CNM)
    c= -Fe2_*(1-CNM) 
    b2_4ac = b**2 - 4*a*c
    # replace negative values with NaN
    b2_4ac[b2_4ac < 0] = np.nan
    x= (-b + np.sqrt(b2_4ac)) / 2 * a # it should be (-b + np.sqrt(b2_4ac)) / (2*a) but we follow Putirka's excel sheet
    Fe_M2 = np.maximum(x, 0.0)  # Fe(M2) end-member
    # Fe_M1= Fe2_- Fe_M2
    Mg_M2= np.maximum(1- Fe_M2 - CNM, 0.0)  # Mg(M2) end-member
    # Mg_M1= X_cpx_6_oxide["MgO"] - Mg_M2


    # cation with suffix
    cation_with_suff_6basis = calculate_cation_nO_basis(
        X_cpx_df, 6, oxide_list, add_suffix="6OBasis"
    )
    return_df = pd.DataFrame({
        "Al4": X_cpx_Al4,
        "Al6": X_cpx_Al6,
        "Fe3_Papike": X_cpx_Fe3_Papike,
        "Fe2_Papike": X_cpx_Fe2_Papike,
        "Jd": X_cpx_Jd,
        "CaTs": X_cpx_CaTs,
        "CaTi": X_cpx_CaTi,
        "CrCaTs": X_cpx_CrCaTs,
        "DiHd": X_cpx_DiHd,
        "DiHd_2003": X_cpx_DiHd_2003,
        "EnFs": X_cpx_EnFs,
        "Fe(M2)": Fe_M2,
        "Mg(M2)": Mg_M2,
        "Fe3_Papike_wang": X_cpx_Fe3_Papike_wang,
        "Fe2_Papike_wang": X_cpx_Fe2_Papike_wang,
        "(Ca+Fe+Mg)/Si": (X_cpx_6_oxide["CaO"] + X_cpx_6_oxide["FeO"] + X_cpx_6_oxide["MgO"]) / X_cpx_6_oxide["SiO2"],
    })
    return_df = pd.concat([cation_with_suff_6basis, return_df], axis=1)

    if original_type is pd.Series:
        return return_df.iloc[0]
    if original_type is dict:
        return return_df.iloc[0].to_dict()
    return return_df

    

# calculate Mg#
def calculate_Mg_number(
    compositions: CompositionsType,
) -> CompositionsType:
    """Calculate Mg# from oxide compositions."""
    oxides_list = get_oxides_list(compositions.columns.tolist())
    compositions_df, original_type = _coerce_compositions(compositions)
    compositions_df = normalize_column_names(compositions_df, oxides_list)

    X_cpx_6_oxide = calculate_cation_nO_basis(compositions_df, 6, oxides_list)

    Mg_number = X_cpx_6_oxide["MgO"] / (X_cpx_6_oxide["MgO"] + X_cpx_6_oxide["FeO"]) * 100

    if original_type is pd.Series:
        return Mg_number.iloc[0]
    if original_type is dict:
        return Mg_number.iloc[0].to_dict()
    return Mg_number


def pyroxene_classification(
    compositions: CompositionsType,
) -> CompositionsType:
    """Classify pyroxenes based on Ca and Na cations per 6 oxygens (Putirka-style)."""
    oxide_list = get_oxides_list(compositions.columns.tolist())
    compositions_df, original_type = _coerce_compositions(compositions)
    compositions_df = normalize_column_names(compositions_df, oxide_list)

    # Cations per 6 O (apfu). NOTE: function may keep oxide-style column names.
    X = calculate_cation_nO_basis(compositions_df, 6, oxide_list)

    # To avoid confusion, map to cation names if needed
    Ca = X["CaO"]
    Mg = X["MgO"]
    Fe = X["FeO"]
    Wo = Ca/(Ca + Mg + Fe)

    conditions = [
        Wo > 0.5,
        (Wo < 0.5) & (Wo > 0.05),
        Wo < 0.05,

    ]
    choices = ["Non-pyroxene", "Clinopyroxene", "Orthopyroxene"]

    classification = pd.Series(
        np.select(conditions, choices, default="Unknown"),
        index=compositions_df.index,
        name="pyroxene_class",
    )

    # Return in the same "shape" as input
    if isinstance(compositions, pd.Series):
        return classification.iloc[0]
    if isinstance(compositions, dict):
        return classification.iloc[0]  # dict input -> single label
    return classification

def cpx_stoichiometry_check(
    compositions: CompositionsType,
    allowed_uncertainty: float = 0.1,
):
    """Check if clinopyroxene compositions are stoichiometric.
    
    (Ca+Fe+Mg)/Si should be in the range 0.9-1.1

    Parameters:
        compositions (pd.DataFrame or dict or pd.Series): The oxide compositions.
        allowed_uncertainty (float): The allowed uncertainty for the (Ca+Fe+Mg)/Si ratio. Default is 0.1 (i.e., 10%).
    Returns:
        tuple: A tuple containing the (Ca+Fe+Mg)/Si ratio and a boolean mask indicating whether each composition is stoichiometric.
    """
    oxide_list = get_oxides_list(compositions.columns.tolist())
    compositions_df, original_type = _coerce_compositions(compositions)
    compositions_df = normalize_column_names(compositions_df, oxide_list)

    # Cations per 6 O (apfu). NOTE: function may keep oxide-style column names.
    X = calculate_cation_nO_basis(compositions_df, 6, oxide_list)
    ratio_CaFeMg_Si = (X["CaO"] + X["FeO"] + X["MgO"]) / X["SiO2"]
    stoichiometric_mask = (ratio_CaFeMg_Si >= (1 - allowed_uncertainty)) & (ratio_CaFeMg_Si <= (1 + allowed_uncertainty))
    stoichiometric_mask.name = "cpx_stoichiometry_check"
    # Return in the same "shape" as input
    if isinstance(compositions, pd.Series):
        return stoichiometric_mask.iloc[0]
    if isinstance(compositions, dict):
        return stoichiometric_mask.iloc[0]
    return ratio_CaFeMg_Si, stoichiometric_mask
        
