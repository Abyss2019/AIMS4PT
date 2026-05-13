"""Neave & Putirka (2017) clinopyroxene-liquid barometer implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

import aims4pt.model_tools.data.NeavePutirka17 as NeavePutirka17_data
from aims4pt.data_tools.compositions import calculate_cation_fractions, cpx_calculation
from aims4pt.model_tools.conventional_model import conventional_model, iterative_model
from aims4pt.model_tools.model_registry import register_model
from aims4pt.utils import get_oxides_list

class eq1_P(conventional_model):
    '''
    Neave, D. A., & Putirka, K. D. (2017). A new clinopyroxene-liquid barometer, and implications for magma storage pressures under Icelandic rift zones. American Mineralogist, 102(4), 777–794. https://doi.org/10.2138/am-2017-5968

    eq1_P
    '''

    def __init__(self):
        '''
        Initialize the model.'''
        parameters_dict = {
            "b": -26.2712,
            "w_T_ln": 39.16138,     # (T(K)/1e4) * ln(...)
            "w_ln_DiHd": -4.21676,  # ln(X_DiHd_cpx)
            "w_AlO_liq": 78.43463,  # X_AlO1.5_liq
            "w_NaK_liq2": 393.8126 # (X_NaO0.5_liq + X_K0.5_liq)^2; this is incorrect in the paper.
        }


        standard_columns = ['SiO2_cpx', 'TiO2_cpx', 'Al2O3_cpx', 'FeOt_cpx',
                            'MnO_cpx', 'MgO_cpx', 'CaO_cpx', 'Na2O_cpx', 'K2O_cpx', 'Cr2O3_cpx', 
                            'SiO2_liq', 'TiO2_liq', 'Al2O3_liq', 'FeOt_liq', 'MnO_liq',
                            'MgO_liq', 'CaO_liq', 'Na2O_liq', 'K2O_liq', 'Cr2O3_liq',
                            'P2O5_liq', 'H2O_liq']

        super().__init__("Putirka et al., 2008 eq.31_P", parameters_dict, standard_columns)
        self.uncertainty = 1.4  # kbar
        self.if_support_hydrous = True  # This model supports hydrous calculations
        self.require_water = False  # This model does not require water
        self.cpx_only = False  # This model only supports clinopyroxene compositions
        self.X_cpx_train_pkl_path = None # see Neave_Putirka_17 class
        self.X_liq_train_pkl_path = None # see Neave_Putirka_17 class

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
        w_T_ln = self.parameters_dict["w_T_ln"]
        w_ln_DiHd = self.parameters_dict["w_ln_DiHd"]
        w_AlO_liq = self.parameters_dict["w_AlO_liq"]
        w_NaK_liq2 = self.parameters_dict["w_NaK_liq2"]

        # print(f"X_cpx_params: {X_cpx_params}")

        # Calculate P
        P_kbar = (
            b
            + w_T_ln * (T_K / 1e4) * np.log(
                X_cpx_params["Jd"] /
                (X_liq_cations["Na2O"]
                * X_liq_cations["Al2O3"]
                * (X_liq_cations["SiO2"] ** 2))
            )
            + w_ln_DiHd * np.log(X_cpx_params["DiHd"])
            + w_AlO_liq * X_liq_cations["Al2O3"]
            + w_NaK_liq2 * (X_liq_cations["Na2O"] + X_liq_cations["K2O"]) ** 2
        )

        
        return P_kbar

    def train(self, X, y, train_method=None):
        pass


from aims4pt.model_tools.Putirka_08 import eq32d_T, eq32d_T_hydrousVersion, eq33_T

@register_model
class Neave_Putirka_17(iterative_model):
    '''
    Neave, D. A., & Putirka, K. D. (2017). A new clinopyroxene-liquid barometer, and implications for magma storage pressures under Icelandic rift zones. American Mineralogist, 102(4), 777–794. https://doi.org/10.2138/am-2017-5968
    (Neave & Putirka, 2017)


    Requirements: Wt: SiO2, Wt: TiO2, Wt: Al2O3, Wt: Cr2O3, Wt: FeO, Wt: MnO, Wt: MgO, Wt: CaO, Wt: Na2O, Wt: K2O

    This model requires iterative calculations for pressure (P) and temperature (T). 
    During initialization, the calculation method must be selected.
    To maintain consistency, T and P methods are calculated separately. 
    This is achieved by creating a `Putirka_08_T` object and a `Putirka_08_P` object, ensuring consistent parameters.
    A `switch_PT` method is added to allow switching between T and P outputs.
    '''
    cpx_only_models = {
        "T_models": {
            "hydrous": ["Pu08_eq32d_T_hydrousVersion"],
            "anhydrous": ["Pu08_eq32d_T"],
            "both": [],

        },
        "P_models": {
            "hydrous": [],
            "anhydrous": [],
            "both": []
        }
    }

    cpx_liq_models = {
        "T_models": {
            "hydrous": [],
            "anhydrous": [],
            "both": ["Pu08_eq33_T"],
        },
        "P_models": {
            "hydrous": [],
            "anhydrous": [],
            "both": ["eq1_P"],
        }
    }

    iterative_model = True

    model_options = {
        'Pu08_eq32d_T': eq32d_T,
        'Pu08_eq32d_T_hydrousVersion': eq32d_T_hydrousVersion,
        'Pu08_eq33_T': eq33_T,
        'eq1_P': eq1_P,
    }
    model_name_str = "Neave & Putirka, 2017"


    def __init__(self,T_P = "P", P_model_name='eq1_P', T_model_name='Pu08_eq33_T', iteration_max=50, stop_criteria=1e-3, comments=None):
        '''
        Initialize the model.

        Consider adding support for input T/P in non-iterative calculations later.

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
        if T_P != "P":
            raise ValueError("Neave_Putirka_17 only supports pressure (P) calculations, not temperature (T) calculations.")
        super().__init__("P", T_model_name, P_model_name, iteration_max, stop_criteria, comments)
        from aims4pt.toolkit_utils import get_file_path
        X_cpx_train_pkl_name = "datapkl/X_cpx_train.pkl"
        X_liq_train_pkl_name = "datapkl/X_liq_train.pkl"
        self.X_cpx_train_pkl_path = get_file_path(
            NeavePutirka17_data, X_cpx_train_pkl_name)
        self.X_liq_train_pkl_path = get_file_path(
            NeavePutirka17_data, X_liq_train_pkl_name)

        self.initialize_model(cpx_training_path=self.X_cpx_train_pkl_path, 
                              liq_training_path=self.X_liq_train_pkl_path, )



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
    model = Neave_Putirka_17( P_model_name='eq1_P', T_model_name='Pu08_eq33_T')
    predicted_P = model.predict(X_cpx, X_liq)
    model.switch_PT('T')
    predicted_T = model.predict(X_cpx, X_liq)
    print("Predicted Temperature (C):\n", predicted_T)
    print("Predicted Pressure (kbar):\n", predicted_P)
