"""Brugman & Till (2019) clinopyroxene-liquid geothermometer implementations."""

from __future__ import annotations
import pandas as pd

import aims4pt.model_tools.data.BrugmanTill19 as BrugmanTill19_data
from aims4pt.data_tools.compositions import calculate_cation_fractions, cpx_calculation
from aims4pt.model_tools.ModelManager import ModelManager
from aims4pt.model_tools.conventional_model import conventional_model
# Import register_model
from aims4pt.model_tools.model_registry import register_model
from aims4pt.utils import get_oxides_list




class eq1_T(conventional_model):
    '''    Brugman, K. K., & Till, C. B. (2019). A low-aluminum clinopyroxene-liquid geothermometer for high-silica magmatic systems. American Mineralogist, 104(7), 996–1004. https://doi.org/10.2138/am-2019-6842

    eq1_T'''

    def __init__(self):
        '''
        Initialize the model.

        '''

        parameters_dict = {
            "b": -1.8946098,
            "w_CaTs": -0.6010197,      # X_Cpx_CaTs
            "w_DiHd2003": -0.1856423,  # X_Cpx_DiHd_2003
            "w_SiO2": 4.71248858,      # X_liq_SiO2
            "w_TiO2": 77.5861878,      # X_liq_TiO2
            "w_FeO": 10.8503727,       # X_liq_FeO
            "w_MgO": 33.6303471,       # X_liq_MgO
            "w_CaO": 15.4532888,       # X_liq_CaO
            "w_K05": 15.6390115        # X_liq_K0.5
        }
        standard_columns = ['SiO2_cpx', 'TiO2_cpx', 'Al2O3_cpx', 'FeOt_cpx',
                            'MnO_cpx', 'MgO_cpx', 'CaO_cpx', 'Na2O_cpx', 'K2O_cpx', 'Cr2O3_cpx', 
                            'SiO2_liq', 'TiO2_liq', 'Al2O3_liq', 'FeOt_liq', 'MnO_liq',
                            'MgO_liq', 'CaO_liq', 'Na2O_liq', 'K2O_liq', 'Cr2O3_liq',]

        super().__init__("Wang et al., 2021 eq.2", parameters_dict, standard_columns)

        self.uncertainty = 20
        self.if_support_hydrous = True  # This model support hydrous calculations
        self.require_water = False  # This model does not require water
        self.cpx_only = False  # This model supports only-clinopyroxene compositions


    def predict(self, X, X_liq=None):
        '''
        T (◦C) = 100 [a · NLT + bTi + cAl + dMn + eMg  +f Ca + gFe2+ + hH2O(wt %) + i  ]  ,

        Parameters:
            X (pd.DataFrame): 
                The input features.

            Returns:
                pd.Series: 
                    The predicted response variable.
        '''

        X_cpx, X_liq = self.process_input(X, X_liq)

        X_cpx_params = cpx_calculation(X_cpx)
        liq_oxides_list = get_oxides_list(X_liq.columns.tolist())
        # remove H2O from the list if it exists
        if 'H2O' in liq_oxides_list:
            liq_oxides_list.remove('H2O')
        X_liq_cations = calculate_cation_fractions(X_liq,liq_oxides_list)
        

        T_C = 300 * (
            self.parameters_dict["b"]
            + self.parameters_dict["w_CaTs"] * X_cpx_params["CaTs"]
            + self.parameters_dict["w_DiHd2003"] * X_cpx_params["DiHd_2003"]
            + self.parameters_dict["w_SiO2"] * X_liq_cations["SiO2"]
            + self.parameters_dict["w_TiO2"] * X_liq_cations["TiO2"]
            + self.parameters_dict["w_FeO"] * X_liq_cations["FeO"]
            + self.parameters_dict["w_MgO"] * X_liq_cations["MgO"]
            + self.parameters_dict["w_CaO"] * X_liq_cations["CaO"]
            + self.parameters_dict["w_K05"] * X_liq_cations["K2O"]
        )
        return T_C


@register_model
class Brugman_Till_19(ModelManager):
    '''
    Brugman, K. K., & Till, C. B. (2019). A low-aluminum clinopyroxene-liquid geothermometer for high-silica magmatic systems. American Mineralogist, 104(7), 996–1004. https://doi.org/10.2138/am-2019-6842
    (Brugman & Till, 2019)

    cpx_only 
    eq1 (P) and 2 (T)

    '''

    def __init__(self, T_P = "T", comments=None):
        '''
        Initialize the model.

        Parameters:
            T_P (str): 
                The name of the model.
            comments (str): 
                Comments about the model or any other information.

        '''
        if T_P != "T":
            raise ValueError("Brugman_Till_19 only supports temperature (T) calculations, not pressure (P) calculations.")
        self.T_P = T_P
        comments = comments
        model = eq1_T()
        standard_columns = model.standard_columns
        super().__init__(model, standard_columns, comments)
        self.model_name = "Brugman & Till, 2019"
        self.cpx_names = model.cpx_names
        self.liq_names = model.liq_names
        self.cpx_only = model.cpx_only
        self.if_support_hydrous = model.if_support_hydrous
        self.require_water = model.require_water

        self.uncertainty = model.uncertainty

        # train, test, path
        from aims4pt.toolkit_utils import get_file_path
        X_cpx_train_pkl_name = r"datapkl/X_cpx_train.pkl"
        X_liq_train_pkl_name = r"datapkl/X_liq_train.pkl"
        self.X_cpx_train_pkl_path = get_file_path(
            BrugmanTill19_data, X_cpx_train_pkl_name)
        self.X_liq_train_pkl_path = get_file_path(
            BrugmanTill19_data, X_liq_train_pkl_name)

        self.initialize_model(cpx_training_path=self.X_cpx_train_pkl_path,
                              liq_training_path=self.X_liq_train_pkl_path)

    def predict(self, X, X_liq=None):
        '''
        Predict the response variable.

        Parameters:
            X (pd.DataFrame): 
                The input features.
            X_liq (pd.DataFrame): 
                The input features for liquid phase.

        Returns:
            pd.Series: 
                The predicted response variable.
        '''
        X = X.copy()
        X.fillna(0, inplace=True)
        
        X_liq.fillna(0, inplace=True)
        return self.model.predict(X, X_liq)

if __name__ == "__main__":
    '''
    '''
    # test
    # CPX composition (wt%).
    X_cpx = pd.DataFrame({
        'SiO2_cpx': [52.37, 48.30, 48.50, 51.44],
        'TiO2_cpx': [ 0.19,  0.16,  0.23,  0.19],
        'Al2O3_cpx': [ 0.63,  0.61,  0.58,  0.68],
        'Cr2O3_cpx': [ 0.00,  0.00,  0.00,  0.00],
        'FeOt_cpx': [12.34, 27.30, 27.40, 16.42],  # FeO(total)
        'MnO_cpx': [ 0.69,  0.86,  0.85,  0.83],
        'MgO_cpx': [12.57,  3.10,  3.44, 10.68],
        'CaO_cpx': [20.92, 19.20, 18.50, 18.74],
        'Na2O_cpx': [ 0.40,  0.30,  0.29,  0.29],
        'K2O_cpx': [ 0.00,  0.00,  0.00,  0.02]
    })

    # Liquid composition (wt%).
    X_liq = pd.DataFrame({
        'SiO2_liq': [77.90, 73.30, 75.30, 73.72],
        'TiO2_liq': [ 0.13,  0.07,  0.09,  0.14],
        'Al2O3_liq': [12.27, 11.62, 11.86, 11.63],
        'Cr2O3_liq': [ 0.00,  0.00,  0.00,  0.00],
        'FeOt_liq': [ 0.54,  1.39,  1.41,  0.36],  # FeO(total)
        'MnO_liq': [ 0.02,  0.06,  0.09,  0.02],
        'MgO_liq': [ 0.08,  0.02,  0.03,  0.02],
        'CaO_liq': [ 0.19,  0.52,  0.55,  0.44],
        'Na2O_liq': [ 3.70,  3.60,  3.60,  3.27],
        'K2O_liq': [ 5.17,  4.90,  4.98,  5.44]
    })
    # Initialize the model
    model = Brugman_Till_19(T_P='T')
    predicted_T = model.predict(X_cpx, X_liq)
    print("test")
    print("Predicted Temperature (C):\n", predicted_T)
