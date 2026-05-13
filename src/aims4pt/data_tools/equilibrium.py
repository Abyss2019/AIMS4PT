# aims4pt/data_tools/equilibrium.py
'''
A collection of functions for calculating equilibrium.

Functions:
    kdEquilibrium_test(X_mine, X_liq, kd=0.28, error=0.08):
        Test the equilibrium condition for mineral and liq compositions.

'''
import pandas as pd
from aims4pt.data_tools.compositions import calculate_cation_fractions
from aims4pt.utils import normalize_column_names
from aims4pt.constants import anhydrous_oxides


def kdEquilibrium_test(X_mine, X_liq, kd=0.28, error=0.08, mode='Fe-Mg'):
    """
    Test the equilibrium condition for mineral and liq compositions.

     Kd_Fe-Mg = (FeO_mine / MgO_mine) / (FeO_liq / MgO_liq)

    Parameters:
        X_mine (DataFrame): 
            DataFrame of mineral compositions.
        X_liq (DataFrame): 
            DataFrame of liq compositions.
        kd (float): 
            Equilibrium constant, default is 0.28.
        error (float): 
            Error margin, default is 0.08.

    Returns:
        mask (ndarray):
            Boolean mask indicating which rows satisfy the equilibrium condition.
        kd_condition (ndarray): 
            Array of calculated kd values.
    """
    # Normalize column names if necessary
    X_mine = X_mine.copy()
    X_liq = X_liq.copy()

    X_mine = normalize_column_names(X_mine, anhydrous_oxides, drop_missing=False)
    X_liq = normalize_column_names(X_liq, anhydrous_oxides, drop_missing=False)

    X_mine_cations = calculate_cation_fractions(X_mine, anhydrous_oxides)
    X_liq_cations = calculate_cation_fractions(X_liq, anhydrous_oxides)
    
    # Calculate the equilibrium condition
    try:
        if mode == 'Fe-Mg':
            kd_condition = X_mine_cations["FeO"] / X_mine_cations["MgO"] / (X_liq_cations["FeO"] / X_liq_cations["MgO"])
        elif mode == 'Ab-An':
            X_mine_cations["Ab"] = X_mine_cations["Na2O"] / (X_mine_cations["Na2O"] + X_mine_cations["CaO"]+ X_mine_cations["K2O"])
            X_mine_cations["An"] = X_mine_cations["CaO"] / (X_mine_cations["Na2O"] + X_mine_cations["CaO"]+ X_mine_cations["K2O"])
            kd_condition = (X_mine_cations["Ab"] * X_liq_cations["Al2O3"] * X_liq_cations["CaO"]) / (X_mine_cations["An"] * X_liq_cations["Na2O"] * X_liq_cations["SiO2"])
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    except KeyError as e:
        raise KeyError(f"Missing required columns in input DataFrames: {e}")
    except ZeroDivisionError:
        raise ZeroDivisionError("Division by zero encountered in kd calculation. Check input data for zero values in 'MgO' or 'FeO' columns.")
    
    # return the mask and array of equilibrium condition
    mask = (kd_condition >= kd - error) & (kd_condition <= kd + error)
    return mask.to_numpy(), kd_condition.to_numpy()



import numpy as np
import pandas as pd

def predict_equilibrium_liquid_from_mineral(
    X_mine,
    kd=0.28,
    mode='Mg#'
):
    """
    Predict equilibrium liquid Mg/Fe or Mg# from mineral compositions only,
    using Fe-Mg exchange partition coefficient:
    
        Kd_Fe-Mg = (Fe/Mg)_mine / (Fe/Mg)_liq

    Parameters
    ----------
    X_mine : DataFrame
        DataFrame of mineral compositions.

    kd : float, default=0.28
        Fe-Mg exchange partition coefficient.

    mode : {'Mg#', 'Mg/Fe', 'Fe/Mg'}, default='Mg#'
        Output type:
        - 'Fe/Mg' : predicted equilibrium liquid Fe/Mg
        - 'Mg/Fe' : predicted equilibrium liquid Mg/Fe
        - 'Mg#'   : predicted equilibrium liquid Mg#

    Returns
    -------
    result : ndarray
        Predicted equilibrium liquid values for each mineral row.
    """
    X_mine = X_mine.copy()
    X_mine = normalize_column_names(X_mine, anhydrous_oxides, drop_missing=False)
    X_mine_cations = calculate_cation_fractions(X_mine, anhydrous_oxides)

    required_cols = ["FeO", "MgO"]
    for col in required_cols:
        if col not in X_mine_cations.columns:
            raise KeyError(f"Missing required column in mineral data: {col}")

    Fe_mine = X_mine_cations["FeO"]
    Mg_mine = X_mine_cations["MgO"]

    # avoid division by zero
    if (Mg_mine == 0).any():
        raise ZeroDivisionError("Division by zero encountered: mineral MgO cation fraction contains zero.")
    if (Fe_mine == 0).any() and mode in ['Mg/Fe', 'Mg#']:
        # not always fatal for Fe/Mg, but can produce inf in Mg/Fe
        pass

    # mineral Fe/Mg
    FeMg_mine = Fe_mine / Mg_mine

    # predicted liquid Fe/Mg from Kd
    FeMg_liq = FeMg_mine / kd

    if mode == 'Fe/Mg':
        return FeMg_liq.to_numpy()

    # predicted liquid Mg/Fe
    MgFe_liq = 1 / FeMg_liq

    if mode == 'Mg/Fe':
        return MgFe_liq.to_numpy()

    elif mode == 'Mg#':
        Mg_number_liq = MgFe_liq / (1 + MgFe_liq)
        return Mg_number_liq.to_numpy()

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def calculate_mg_number(X, fe_col="FeO", mg_col="MgO"):
    """
    Calculate mineral Mg# from input mineral compositions.

    Mg# = Mg / (Mg + Fe)

    Parameters
    ----------
    X : DataFrame
        DataFrame of mineral compositions.

    fe_col : str, default="FeO"
        Column name for Fe cation fraction after normalization/calculation.

    mg_col : str, default="MgO"
        Column name for Mg cation fraction after normalization/calculation.

    Returns
    -------
    mg_number : Series
        Mineral Mg# for each row.
    """
    X = X.copy()
    X = normalize_column_names(X, anhydrous_oxides, drop_missing=False)
    X_cations = calculate_cation_fractions(X, anhydrous_oxides)

    try:
        Fe = X_cations[fe_col]
        Mg = X_cations[mg_col]
        mg_number = Mg / (Mg + Fe)
    except KeyError as e:
        raise KeyError(f"Missing required columns in input DataFrame: {e}")
    except ZeroDivisionError:
        raise ZeroDivisionError("Division by zero encountered in Mg# calculation.")

    return mg_number



# for cpx end-member equilibrium test cpx_calculation

def calculate_cpx_end_member(X_cpx, X_liq, P, T):
    """
    Calculate cpx end-member compositions from cpx and liq compositions, pressure, and temperature.

    Parameters:
        X_cpx (DataFrame): 
            DataFrame of cpx compositions.
        X_liq (DataFrame):
            DataFrame of liq compositions.
        P (series like):
            Pressure in kbar.
        T (series like):
            Temperature in °C.
    Returns:
        cpx_end_member_compositions (DataFrame):
            DataFrame of calculated cpx end-member compositions.
    """
    from aims4pt.data_tools.compositions import cpx_calculation
    # Normalize column names if necessary
    X_cpx = X_cpx.copy()
    X_liq = X_liq.copy()

    X_cpx = normalize_column_names(X_cpx, anhydrous_oxides, drop_missing=False)
    X_liq = normalize_column_names(X_liq, anhydrous_oxides, drop_missing=False)

    X_cpx_component_ob = cpx_calculation(X_cpx)
    X_liq_cations = calculate_cation_fractions(X_liq, anhydrous_oxides)

    T_K = T + 273.15  # Convert temperature to Kelvin
    P_bar = P * 1000  # Convert pressure to bar
    P_kbar = P  # Pressure in kbar for Mollo 2013 equations


    # 
    # Putirka (1999) equations are all from the excel spreadsheet.
    dihd_ob = X_cpx_component_ob["DiHd"]
    enfs_ob = X_cpx_component_ob["EnFs"]

    dihd_pu1999 = np.exp(
        -0.482
        - 0.439 * np.log(X_liq_cations["SiO2"])
        + 101.03 * (X_liq_cations["Na2O"] + X_liq_cations["K2O"]) ** 3
        - 51.69 * P_kbar / T_K
        - 3742.5 * (enfs_ob ** 2) / T_K
    )

    enfs_pu1999 = np.exp(
        -6.96
        + 18438 / T_K
        + 8 * np.log(T_K / 1670)
        + 0.66 * np.log(
            (X_liq_cations["FeO"] + X_liq_cations["MgO"]) ** 2
            * X_liq_cations["SiO2"] ** 2
        )
        - 5.1e3 * (dihd_ob ** 2) / T_K
        + 1.81 * np.log(X_liq_cations["SiO2"])
    )

    cats_pu1999 = np.exp(
        2.58
        + 0.12 * P_kbar / T_K
        - 9e-7 * (P_kbar ** 2) / T_K
        + 0.78 * np.log(
            X_liq_cations["CaO"]
            * (X_liq_cations["Al2O3"] ** 2)
            * X_liq_cations["SiO2"]
        )
        - 4.3e3 * (dihd_ob ** 2) / T_K
    )

    caTi_Pu1999 = np.exp(
        5.1
        + 0.52 * np.log(
            X_liq_cations["CaO"]
            * X_liq_cations["TiO2"]
            * (X_liq_cations["Al2O3"] ** 2)
        )
        + 2.04e3 * (dihd_ob ** 2) / T_K
        - 6.2 * X_liq_cations["SiO2"]
        + 42.5 * X_liq_cations["Na2O"] * X_liq_cations["Al2O3"]
        - 45.1 * (X_liq_cations["FeO"] + X_liq_cations["MgO"]) * X_liq_cations["Al2O3"]
    )

    dihd_mo2013 = X_cpx_component_ob["DiHd"].copy()
    enfs_mo2013 = X_cpx_component_ob["EnFs"].copy()

    tol = 0.001
    max_iter = 50
    converged = False

    for i in range(max_iter):
        last_dihd_mo2013 = dihd_mo2013.copy()
        last_enfs_mo2013 = enfs_mo2013.copy()

        enfs_mo2013 = np.exp(
            0.018
            - 9.61 * X_liq_cations["CaO"]
            + 7.46 * X_liq_cations["MgO"] * X_liq_cations["SiO2"]
            - 0.34 * np.log(X_liq_cations["Al2O3"])
            - 3.78 * (X_liq_cations["Na2O"] + X_liq_cations["K2O"])
            - 3737.3 * (dihd_mo2013 ** 2) / T_K
            - 46.8 * P_kbar / T_K
        )

        dihd_mo2013 = np.exp(
            -2.18
            - 3.16 * X_liq_cations["TiO2"]
            - 0.365 * np.log(X_liq_cations["Al2O3"])
            + 0.05 * np.log(X_liq_cations["MgO"])
            - 3858.2 * (enfs_mo2013 ** 2) / T_K
            + 2107.4 / T_K
            - 17.64 * P_kbar / T_K
        )

        difference = (
            np.nanmax(np.abs(dihd_mo2013 - last_dihd_mo2013))
            + np.nanmax(np.abs(enfs_mo2013 - last_enfs_mo2013))
        )

        if difference < tol:
            converged = True
            break

    if not converged:
        print(
            f"Warning: Mollo et al. (2013) DiHd-EnFs iteration did not converge "
            f"after {max_iter} iterations. Final difference = {difference:.4g}"
        )


    cpx_end_member_compositions = pd.DataFrame({
        "DiHd_Putirka1999": dihd_pu1999,
        "EnFs_Putirka1999": enfs_pu1999,
        "DiHd_Mollo2013": dihd_mo2013,
        "EnFs_Mollo2013": enfs_mo2013,
        "CaTs_Putirka1999": cats_pu1999,
        "CaTi_Putirka1999": caTi_Pu1999,  # Placeholder: replace with actual calculation for CaTi
    })

    return cpx_end_member_compositions

# general function for cpx end-member equilibrium test, which can be used for both diopside-hedenbergite and enstatite-ferrosilite equilibrium tests
def cpx_end_member_equilibrium_test(X_cpx, X_liq, P, T, end_member='DiHd', equation = "Putirka1999", n_SEE = 1):
    """
    Test the equilibrium condition for cpx end-member compositions and liq compositions.

    Parameters:
        X_cpx (DataFrame): 
            DataFrame of cpx compositions.
        X_liq (DataFrame):
            DataFrame of liq compositions.
        end_member (str):
            The cpx end-member to test ('DiHd', 'EnFs', 'CaTs', 'CaTi').
        equation (str):
            The equation to use for the equilibrium test ('Putirka1999', 'Mollo2013').
        n_SEE_range (int):
            The number of standard errors to consider for the equilibrium test.
    """
    from aims4pt.data_tools.compositions import cpx_calculation
    tol_LOOKUP = {
        ("DiHd", "Putirka1999", "1SEE"): 0.06,
        ("EnFs", "Putirka1999", "1SEE"): 0.05,
        ("CaTs", "Putirka1999", "1SEE"): 0.03,
        ("CaTs", "Putirka1999", "2SEE"): 0.06,
        ("CaTi", "Putirka1999", "1SEE"): 0.01,
        ("CaTi", "Putirka1999", "2SEE"): 0.02,
        ("DiHd", "Mollo2013", "1SEE"): 0.1, # expand range from Mollo et al., (2018) MacDonald et al. (2023), 
        ("EnFs", "Mollo2013", "1SEE"): 0.05,
    }
    
    
    X_cpx = X_cpx.copy()
    X_liq = X_liq.copy()

    X_cpx = normalize_column_names(X_cpx, anhydrous_oxides, drop_missing=False)
    X_liq = normalize_column_names(X_liq, anhydrous_oxides, drop_missing=False)

    X_cpx_component_ob = cpx_calculation(X_cpx)
    X_cpx_component_cal = calculate_cpx_end_member(X_cpx, X_liq, P, T)

    delta_cpx_component = pd.DataFrame(columns=X_cpx_component_cal.columns)
    mask_df = pd.DataFrame(columns=X_cpx_component_cal.columns)

    for col in X_cpx_component_cal.columns:
        delta_cpx_component[col] = np.abs(X_cpx_component_ob[col.split("_")[0]] - X_cpx_component_cal[col])
        mask_df[col] = (delta_cpx_component[col] <= tol_LOOKUP.get((col.split("_")[0], col.split("_")[1], f"{n_SEE}SEE"), 0.05))  # default tolerance if not found in lookup

    if end_member == 'DiHd':
        if equation == "Putirka1999":
            observed = X_cpx_component_ob["DiHd"]
            calculated = X_cpx_component_cal["DiHd_Putirka1999"]
            tol_error = tol_LOOKUP.get(("DiHd", "Putirka1999"), 0.05)  # Example tolerance, replace with actual value
        elif equation == "Mollo2013":
            observed = X_cpx_component_ob["DiHd"]
            calculated = X_cpx_component_cal["DiHd_Mollo2013"]
            tol_error = tol_LOOKUP.get(("DiHd", "Mollo2013"), 0.04)  # Example tolerance, replace with actual value
        else:
            raise ValueError(f"Unsupported equation: {equation}")
    elif end_member == 'EnFs':
        if equation == "Putirka1999":
            observed = X_cpx_component_ob["EnFs"]
            calculated = X_cpx_component_cal["EnFs_Putirka1999"]
            tol_error = tol_LOOKUP.get(("EnFs", "Putirka1999"), 0.06)  # Example tolerance    , replace with actual value
        elif equation == "Mollo2013":
            observed = X_cpx_component_ob["EnFs"]
            calculated = X_cpx_component_cal["EnFs_Mollo2013"]
            tol_error = tol_LOOKUP.get(("EnFs", "Mollo2013"), 0.05)  # Example tolerance, replace with actual value
        else:
            raise ValueError(f"Unsupported equation: {equation}")
    elif end_member == 'CaTs':
        if equation == "Putirka1999":
            observed = X_cpx_component_ob["CaTs"]
            calculated = X_cpx_component_cal["CaTs_Putirka1999"]
            tol_error = tol_LOOKUP.get(("CaTs", "Putirka1999"), 0.07)  # Example tolerance, replace with actual value
        else:
            raise ValueError(f"Unsupported equation for CaTs: {equation}")
    elif end_member == 'CaTi':
        if equation == "Putirka1999":
            observed = X_cpx_component_ob["CaTi"]
            calculated = X_cpx_component_cal["CaTi_Putirka1999"]
            tol_error = tol_LOOKUP.get(("CaTi", "Putirka1999"), 0.08)  # Example tolerance, replace with actual value
        else:
            raise ValueError(f"Unsupported equation for CaTi: {equation}")
    else:
        if end_member != 'all':
            raise ValueError(f"Unsupported end-member: {end_member}")
    
    mask = (np.abs(observed - calculated) <= tol_error)
    computation_dict = {
        "observed": X_cpx_component_ob,
        "calculated": X_cpx_component_cal,
        "delta": delta_cpx_component,
        "all_masks": mask_df,
    }
    return mask, computation_dict






# test function
def __test_kdEquilibrium_test():
    """
    Test the kdEquilibrium_test function with sample data.
    """
    # Sample mineral and liq compositions
    data_mine = {
        'MgO_mine': [10, 20, 30],
        'FeO_mine': [5, 10, 15]
    }
    data_liq = {
        'MgO_liq': [5, 10, 15],
        'FeO_liq': [2.5, 5, 7.5]
    }
    
    X_mine = pd.DataFrame(data_mine)
    X_liq = pd.DataFrame(data_liq)
    
    mask, kd_condition = kdEquilibrium_test(X_mine, X_liq)
    
    print("Mask:", mask)
    print("kd Condition:", kd_condition)




if __name__ == "__main__":
    # __test_kdEquilibrium_test()
    # This will run the test function when the script is executed directly.
    # You can remove this part if you want to use this module without running tests.

    from  aims4pt.test_utils import get_test_cpx_liq
    X_liq, X_cpx = get_test_cpx_liq()
    P_kbar = pd.Series(
        [1.0, 1.6, 0.9, 1.0, 3.3],

        name="P_kbar"
    )

    T_C = pd.Series(
        [1071.55, 1016.09, 981.61, 1014.17, 935.80],

        name="T_C"
    )



    cpx_end_member_compositions = calculate_cpx_end_member(X_cpx, X_liq, P_kbar, T_C)
    print(cpx_end_member_compositions)