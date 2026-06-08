"""Putirka (2008) clinopyroxene-liquid thermobarometer implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aims4pt.data_tools.compositions import calculate_cation_fractions, cpx_calculation
from aims4pt.model_tools.conventional_model import conventional_model, iterative_model
from aims4pt.model_tools.model_registry import register_model
from aims4pt.utils import get_oxides_list

class eq25a_P(conventional_model):
    '''
    Putirka, K. (2008). Thermometers and Barometers for Volcanic Systems. In Reviews in Mineralogy and Geochemistry (Vol. 69, Issue 1, pp. 61–120). https://doi.org/10.2138/rmg.2008.69.3

    eq31_P
    '''

    def __init__(self):
        '''
        Initialize the model.'''
        parameters_dict = {
            "b": -40.73,
            "w_T": 358,           # T(K)/1e4
            "w_T_ln": 21.7,      # (T(K)/1e4) * ln(...)
            "w_CaO_liq": -106,  # X_CaO_liq
            "w_NaK_liq2": -166, # (X_NaO0.5_liq + X_KO0.5_liq)^2
            "w_SiO2FeOMgO_liq": -50.2, # X_SiO2_liq * (X_FeO_liq + X_MgO_liq)
            "w_ln_DiHd": -3.2,  # ln(X_DiHd_cpx)
            "w_ln_EnFs": -2.2,  # ln(X_EnFs_cpx)
            "w_ln_Al": 0.86,     # ln(X_Al_cpx)
            "w_H2O": 0.4       # H2O_liq
        } # according to Excel sheet provided by Putirka, Clinopyroxene_P-T_2020 v3



        standard_columns = ['SiO2_cpx', 'TiO2_cpx', 'Al2O3_cpx', 'FeOt_cpx',
                            'MnO_cpx', 'MgO_cpx', 'CaO_cpx', 'Na2O_cpx', 'K2O_cpx', 'Cr2O3_cpx', 
                            'SiO2_liq', 'TiO2_liq', 'Al2O3_liq', 'FeOt_liq', 'MnO_liq',
                            'MgO_liq', 'CaO_liq', 'Na2O_liq', 'K2O_liq', 'Cr2O3_liq',
                            'P2O5_liq', 'H2O_liq']

        super().__init__("Putirka et al., 2008 eq.31_P", parameters_dict, standard_columns)
        self.uncertainty = 2.9  # kbar
        self.if_support_hydrous = True  # This model does not support hydrous calculations
        self.require_water = True  # This model does not require water
        self.cpx_only = False  # This model only supports clinopyroxene compositions
        self.X_cpx_train_pkl_path = None # 暂时还没有
        self.X_liq_train_pkl_path = None # 暂时还没有
        

    def predict(self, X, X_liq=None):
        '''

        Parameters:
            X (pd.DataFrame): 
                The input features.

        Returns:
            pd.Series: 
                The predicted response variable.
        '''
        if X_liq is None or X_liq.empty:
            Warning("X_liq is not provided, skip calculations.")
            return pd.Series(np.nan, index=X.index)
        X_cpx, X_liq = self.process_input(X, X_liq)
        T_C = X_cpx["T_C"]
        X_cpx_params:pd.DataFrame = cpx_calculation(X_cpx)


        liq_oxides_list = get_oxides_list(X_liq.columns.tolist())
        # remove H2O from the list if it exists
        if 'H2O' in liq_oxides_list:
            liq_oxides_list.remove('H2O')
        X_liq_cations = calculate_cation_fractions(X_liq,liq_oxides_list)
        

        T_K = T_C + 273.15

        # Extract parameters
        b = self.parameters_dict["b"]
        w_T = self.parameters_dict["w_T"]
        w_T_ln = self.parameters_dict["w_T_ln"]
        w_CaO_liq = self.parameters_dict["w_CaO_liq"]
        w_NaK_liq2 = self.parameters_dict["w_NaK_liq2"]
        w_SiO2FeOMgO_liq = self.parameters_dict["w_SiO2FeOMgO_liq"]
        w_ln_DiHd = self.parameters_dict["w_ln_DiHd"]
        w_ln_EnFs = self.parameters_dict["w_ln_EnFs"]
        w_ln_Al = self.parameters_dict["w_ln_Al"]
        w_H2O = self.parameters_dict["w_H2O"]

        # print(f"X_cpx_params: {X_cpx_params}")

        # Calculate P
        P_kbar = (
            b
            + w_T * (T_K / 1e4)
            + w_T_ln * (T_K / 1e4) * np.log(
                X_cpx_params["Jd"] / (
                    X_liq_cations["Na2O"] * X_liq_cations["Al2O3"] * (X_liq_cations["SiO2"] ** 2)
                )
            )
            + w_CaO_liq * X_liq_cations["CaO"]
            + w_NaK_liq2 * (X_liq_cations["Na2O"] + X_liq_cations["K2O"]) ** 2
            + w_SiO2FeOMgO_liq * X_liq_cations["SiO2"] * (X_liq_cations["FeO"] + X_liq_cations["MgO"])
            + w_ln_DiHd * np.log(X_cpx_params["DiHd"])
            + w_ln_EnFs * np.log(X_cpx_params["EnFs"])
            + w_ln_Al * np.log(X_cpx_params["Al2O3_6OBasis"])
            + w_H2O * X_liq["H2O_liq"]
        )

        
        return P_kbar

    def train(self, X, y, train_method=None):
        pass




class eq24a_T(conventional_model):
    '''
    Putirka, K. (2008). Thermometers and Barometers for Volcanic Systems. In Reviews in Mineralogy and Geochemistry (Vol. 69, Issue 1, pp. 61–120). https://doi.org/10.2138/rmg.2008.69.3

    eq31_P
    '''

    def __init__(self):
        '''
        Initialize the model.'''
        parameters_dict = {
            "b": 7.53,
            "w_ln": -0.14,       # ln(...)
            "w_H2O": 0.07,
            "w_CaOSiO2": -14.9,  # (X_CaO_liq * X_SiO2_liq)
            "w_ln_TiO2": -0.08,  # ln(X_TiO2_liq)
            "w_NaK": -3.62,      # (X_NaO0.5_liq + X_KO0.5_liq)
            "w_Mg#": -1.1,       # (Mg#_liq)
            "w_ln_EnFs": -0.18,  # ln(X_EnFs_cpx)
            "w_P": -0.026        # P (kbar) 这里Putirka的原文是-0.027, 但是给的计算excel里面是-0.026.
            # In Putirka's original paper, the coefficient for pressure (P) is -0.027, but the provided calculation Excel file uses -0.026.
        }

        standard_columns = ['SiO2_cpx', 'TiO2_cpx', 'Al2O3_cpx', 'FeOt_cpx',
                            'MnO_cpx', 'MgO_cpx', 'CaO_cpx', 'Na2O_cpx', 'K2O_cpx', 'Cr2O3_cpx', 
                            'SiO2_liq', 'TiO2_liq', 'Al2O3_liq', 'FeOt_liq', 'MnO_liq',
                            'MgO_liq', 'CaO_liq', 'Na2O_liq', 'K2O_liq', 'Cr2O3_liq',
                            'P2O5_liq', 'H2O_liq']

        super().__init__("Putirka et al., 2008 eq.31_P", parameters_dict, standard_columns)
        self.uncertainty = 45  # C
        self.if_support_hydrous = True  # This model does not support hydrous calculations
        self.require_water = True  # This model does not require water
        self.cpx_only = False  # This model only supports clinopyroxene compositions
        self.X_cpx_train_pkl_path = None # 暂时还没有
        self.X_liq_train_pkl_path = None # 暂时还没有
        

    def predict(self, X, X_liq=None):
        '''


        Parameters:
            X (pd.DataFrame): 
                The input features.

        Returns:
            pd.Series: 
                The predicted response variable.
        '''

        if X_liq is None or X_liq.empty:
            Warning("X_liq is not provided, skip calculations.")
            return pd.Series(np.nan, index=X.index)
        X_cpx, X_liq = self.process_input(X, X_liq)
        P_kbar = X_cpx["P_kbar"]
        X_cpx_params:pd.DataFrame = cpx_calculation(X_cpx)


        liq_oxides_list = get_oxides_list(X_liq.columns.tolist())
        # remove H2O from the list if it exists
        if 'H2O' in liq_oxides_list:
            liq_oxides_list.remove('H2O')
        X_liq_cations = calculate_cation_fractions(X_liq,liq_oxides_list)
        


        # Extract parameters
        b = self.parameters_dict["b"]
        w_ln = self.parameters_dict["w_ln"]
        w_H2O = self.parameters_dict["w_H2O"]
        w_CaOSiO2 = self.parameters_dict["w_CaOSiO2"]
        w_ln_TiO2 = self.parameters_dict["w_ln_TiO2"]
        w_NaK = self.parameters_dict["w_NaK"]
        w_Mg_hash = self.parameters_dict["w_Mg#"]
        w_ln_EnFs = self.parameters_dict["w_ln_EnFs"]
        w_P = self.parameters_dict["w_P"]

        X_liq_cations["Fm"] = X_liq_cations["FeO"] + X_liq_cations["MgO"]
        X_liq_cations["Fm"] = X_liq_cations["Fm"].replace(0, np.nan)  # Avoid division by zero
        X_liq_cations["Mg#"] = X_liq_cations["MgO"] / X_liq_cations["Fm"]

        ln_term = np.log(
            (X_cpx_params["Jd"] * X_liq_cations["CaO"] * X_liq_cations["Fm"])
            / (X_cpx_params["DiHd"] * X_liq_cations["Na2O"] * X_liq_cations["Al2O3"])
        )

        right:pd.Series = (
            b
            + w_ln * ln_term
            + w_H2O * X_liq["H2O_liq"]
            + w_CaOSiO2 * X_liq_cations["CaO"] * X_liq_cations["SiO2"]
            + w_ln_TiO2 * np.log(X_liq_cations["TiO2"])
            + w_NaK * (X_liq_cations["Na2O"] + X_liq_cations["K2O"])
            + w_Mg_hash * X_liq_cations["Mg#"]
            + w_ln_EnFs * np.log(X_cpx_params["EnFs"])
            + w_P * P_kbar
        )
        # print("-----------------")
        # print("w_ln * ln_term",w_ln * ln_term)
        # print("w_H2O * X_liq['H2O_liq']",w_H2O * X_liq["H2O_liq"])
        # print("w_CaOSiO2 * X_liq_cations['CaO'] * X_liq_cations['SiO2']",w_CaOSiO2 * X_liq_cations["CaO"] * X_liq_cations["SiO2"])
        # print("w_ln_TiO2 * np.log(X_liq_cations['TiO2'])",w_ln_TiO2 * np.log(X_liq_cations["TiO2"]))
        # print("w_NaK * (X_liq_cations['Na2O'] + X_liq_cations['K2O'])",w_NaK * (X_liq_cations["Na2O"] + X_liq_cations["K2O"]))
        # print("w_Mg_hash * X_liq_cations['Mg#']",w_Mg_hash * X_liq_cations["Mg#"])
        # print("w_ln_EnFs * np.log(X_cpx_params['EnFs'])",w_ln_EnFs * np.log(X_cpx_params["EnFs"]))
        # print("w_P * P_kbar",w_P * P_kbar)

        T_K = 1e4/ right
        
        T_C = T_K - 273.15

        return T_C

    def train(self, X, y, train_method=None):
        pass





class Putirka_08(iterative_model):
    '''
    Putirka, K. (2008). Thermometers and Barometers for Volcanic Systems. In Reviews in Mineralogy and Geochemistry (Vol. 69, Issue 1, pp. 61–120). https://doi.org/10.2138/rmg.2008.69.3

    Requirements: Wt: SiO2, Wt: TiO2, Wt: Al2O3, Wt: Cr2O3, Wt: FeO, Wt: MnO, Wt: MgO, Wt: CaO, Wt: Na2O, Wt: K2O

    This model requires iterative calculations for pressure (P) and temperature (T). 
    During initialization, the calculation method must be selected.
    To maintain consistency, T and P methods are calculated separately. 
    This is achieved by creating a `Putirka_08_T` object and a `Putirka_08_P` object, ensuring consistent parameters.
    A `switch_PT` method is added to allow switching between T and P outputs.
    '''
    cpx_only_models = {
        "T_models": {
            "hydrous": [],
            "anhydrous": [],
            "both": [],

        },
        "P_models": {
            "hydrous": [],
            "anhydrous": [],
            "both": [] # "eq32b_P" 这个有点问题
        }
    }

    cpx_liq_models = {
        "T_models": {
            "hydrous": [],
            "anhydrous": [],
            "both": ["eq24a_T"],
        },
        "P_models": {
            "hydrous": [],
            "anhydrous": [],
            "both": ["eq25a_P"]
        }
    }

    model_options = {
            'eq24a_T': eq24a_T,
            'eq25a_P': eq25a_P,
        }
    model_name_str = "Putirka, 2008"

    iterative_model = True

    
    def __init__(self, T_P, T_model_name, P_model_name, iteration_max=50, stop_criteria=1e-3, comments=None):
        '''
        Initialize the model.
        e.g., Putirka_08(T_P='T', T_model_name='eq33_T', P_model_name='eq31_P')
        Putirka_08(T_P='P', T_model_name='eq32d_T', P_model_name='eq32a_P')


        Parameters:
            T_P (str):
                The output type. 'T' for temperature, 'P' for pressure.
            T_model_name (str):
                The name of the temperature model.
                if = "Input" then require "T_C" when predicting.
                Options: 'eq32d_T'
            P_model_name (str):
                The name of the pressure model.
                if = "Input" then require "P_kbar" when predicting.
                Options: 'eq32a_P'
            iteration_max (int):
                The maximum number of iterations for the model.
            stop_criteria (float):

                    The stop criteria for the model.
            comments (str):

                    Comments for the model. 
        '''
        super().__init__(T_P, T_model_name, P_model_name, iteration_max, stop_criteria, comments)




if __name__ == "__main__":
    '''
    Liquid (Glass) Composition - in Weight Percent									    	|	Clinopyroxene Compositions - in Weight Percent									
    SiO2	TiO2	Al2O3	FeOt	MnO	    MgO	    CaO	    Na2O	K2O	  Cr2O3	P2O5	H2O |	SiO2	TiO2	Al2O3	FeOt	MnO	    MgO	    CaO 	Na2O	K2O	Cr2O3
    51.1	0.93	17.5	8.91	0.18	6.09	11.5	3.53	0.17	0	0.15	3.8	|	51.5	0.5	    3.7	    5.18	0.09	15.8	22.8	0.24	0	0.66
    51.5	1.19	19.2	8.7	    0.19	4.98	10	    3.72	0.42	0	0.14	6.2	|   50.3	0.73	4.12	5.83	0	    15	    22.7	0.24	0	0.28

    '''
    # test
    X_cpx = pd.DataFrame({
        'SiO2_cpx': [51.5, 50.3],
        'TiO2_cpx': [0.5, 0.73],
        'Al2O3_cpx': [3.7, 4.12],
        'FeOt_cpx': [5.18, 5.83],
        'MnO_cpx': [0.09, 0],
        'MgO_cpx': [15.8, 15],
        'CaO_cpx': [22.8, 22.7],
        'Na2O_cpx': [0.24, 0.24],
        'K2O_cpx': [0, 0],
        'Cr2O3_cpx': [0.66, 0.28]
    })
    X_liq = pd.DataFrame({
        'SiO2_liq': [51.1, 51.5],
        'TiO2_liq': [0.93, 1.19],
        'Al2O3_liq': [17.5, 19.2],
        'FeOt_liq': [8.91, 8.7],
        'MnO_liq': [0.18, 0.19],
        'MgO_liq': [6.09, 4.98],
        'CaO_liq': [11.5, 10],
        'Na2O_liq': [3.53, 3.72],
        'K2O_liq': [0.17, 0.42],
        'Cr2O3_liq': [0, 0],
        'P2O5_liq': [0.15, 0.14],
        'H2O_liq': [3.8, 6.2]
    })
    # Initialize the model
    model = Putirka_08(T_P='T', P_model_name='eq31_P', T_model_name='eq33_T')
    predicted_T = model.predict(X_cpx, X_liq)
    model.switch_PT('P')
    predicted_P = model.predict(X_cpx, X_liq)
    print("Predicted Temperature (C):\n", predicted_T)
    print("Predicted Pressure (kbar):\n", predicted_P)
