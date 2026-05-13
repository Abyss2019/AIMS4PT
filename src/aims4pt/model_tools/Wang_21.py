"""Wang et al. (2021) clinopyroxene-liquid barometer/thermometer equations."""

from __future__ import annotations

from re import X

import numpy as np
import pandas as pd

import aims4pt.model_tools.data.Wang_21 as Wang21_data
from aims4pt.data_tools.compositions import cpx_calculation
from aims4pt.model_tools.ModelManager import ModelManager
from aims4pt.model_tools.conventional_model import conventional_model
# Import register_model
from aims4pt.model_tools.model_registry import register_model
from aims4pt.utils import normalize_column_names


class eq1_P(conventional_model):
    '''
    P (kbar) = a · NLT · ln AlVI + bSi + cFetot + dMg  + eCa + f Na + g,

    '''

    def __init__(self):
        '''
        Initialize the model.

        '''
        parameters_dict = {
            "ω₀": 1.4844,
            "ω₁": 7.7408,
            "ω₂": 1.1675,
            "ω₃": 1.0604,
            "ω₄": 0.0387,
            "ω₅": -0.0628,
            "a": -7.9509,
            "b": 0.6492,
            "c": -5.9522,
            "d": -11.1942,
            "e": -24.2802,
            "f": 108.663,
            "g": 25.0019

        }
        standard_columns = ['SiO2', 'TiO2', 'Al2O3',
                            'Cr2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
        standard_columns = [col + "_cpx" for col in standard_columns]
        
        super().__init__("Wang et al., 2021 eq.1", parameters_dict, standard_columns)
        self.uncertainty = 1.66
        self.if_support_hydrous = True  # This model support hydrous calculations
        self.require_water = False  # This model does not require water
        self.cpx_only = True  # This model supports only-clinopyroxene compositions



    def predict(self, X, X_liq=None):
        '''
        P (kbar) = a · NLT · ln AlVI + bSi + cFetot + dMg  + eCa + f Na + g,

        Parameters:
            X (pd.DataFrame): 
                The input features.

        Returns:
            pd.Series: 
                The predicted response variable.

        '''

        X_cpx, X_liq = self.process_input(X, X_liq)

        X_cpx_params = cpx_calculation(X_cpx)

        NLT = X_cpx_params["Al6"]*self.parameters_dict["ω₀"] / (X_cpx_params["Al6"]*self.parameters_dict["ω₀"] + X_cpx_params["TiO2_6OBasis"]*self.parameters_dict["ω₁"] +
                                                                X_cpx_params["Cr2O3_6OBasis"]*self.parameters_dict["ω₂"] + X_cpx_params["FeO_6OBasis"]*self.parameters_dict["ω₃"] +
                                                                X_cpx_params["MnO_6OBasis"]*self.parameters_dict["ω₄"] + X_cpx_params["MgO_6OBasis"]*self.parameters_dict["ω₅"])

        # P (kbar) = a · NLT · ln AlVI + bSi + cFetot + dMg  + eCa + f Na + g,
        P_kbar = self.parameters_dict["a"] * NLT * np.log(X_cpx_params["Al6"]) + self.parameters_dict["b"] * X_cpx_params["SiO2_6OBasis"] + \
            self.parameters_dict["c"] * X_cpx_params["FeO_6OBasis"] + self.parameters_dict["d"] * X_cpx_params["MgO_6OBasis"] + \
            self.parameters_dict["e"] * X_cpx_params["CaO_6OBasis"] + \
            self.parameters_dict["f"] * \
            X_cpx_params["Na2O_6OBasis"] + self.parameters_dict["g"]

        return P_kbar

    def train(self, X, y, train_method=None):
        '''
        Train the model.

        Parameters:
            X (pd.DataFrame): 
                The input features.
            y (pd.Series): 
                The response variable.


        '''
        # train by gradient descent
        pass


class eq2_T(conventional_model):
    '''T (◦C) = 100 [a · NLT + bTi + cAl + dMn + eMg  +f Ca + gFe2+ + hH2O(wt %) + i  ]  ,'''

    def __init__(self):
        '''
        Initialize the model.

        '''

        parameters_dict = {
            "ω₀": 1.4844,
            "ω₁": 7.7408,
            "ω₂": 1.1675,
            "ω₃": 1.0604,
            "ω₄": 0.0387,
            "ω₅": -0.0628,
            "a": 3.124395,
            "b": 2.305194,
            "c": -5.68698,
            "d": -32.12,
            "e": -4.76386,
            "f": -7.10883,
            "g": -6.51019,
            "h": -0.23384,
            "i": 23.21929

        }
        standard_columns = ['SiO2', 'TiO2', 'Al2O3',
                            'Cr2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O']
        standard_columns = [
            col + "_cpx" for col in standard_columns] + ['H2O_liq']

        super().__init__("Wang et al., 2021 eq.2", parameters_dict, standard_columns)
        self.uncertainty = 36.6  # C
        self.if_support_hydrous = True  # This model support hydrous calculations
        self.require_water = True  # This model requires water
        self.cpx_only = True  # This model supports only-clinopyroxene compositions


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

        NLT = X_cpx_params["Al6"]*self.parameters_dict["ω₀"] / (X_cpx_params["Al6"]*self.parameters_dict["ω₀"] + X_cpx_params["TiO2_6OBasis"]*self.parameters_dict["ω₁"] +
                                                                X_cpx_params["Cr2O3_6OBasis"]*self.parameters_dict["ω₂"] + X_cpx_params["FeO_6OBasis"]*self.parameters_dict["ω₃"] +
                                                                X_cpx_params["MnO_6OBasis"]*self.parameters_dict["ω₄"] + X_cpx_params["MgO_6OBasis"]*self.parameters_dict["ω₅"])

        # T (◦C) = 100 [a · NLT + bTi + cAl + dMn + eMg  +f Ca + gFe2+ + hH2O(wt %) + i  ]  ,

        T_C = 100 * (self.parameters_dict["a"] * NLT + self.parameters_dict["b"] * X_cpx_params["TiO2_6OBasis"] + self.parameters_dict["c"] * X_cpx_params["Al2O3_6OBasis"] +
                     self.parameters_dict["d"] * X_cpx_params["MnO_6OBasis"] + self.parameters_dict["e"] * X_cpx_params["MgO_6OBasis"] + self.parameters_dict["f"] * X_cpx_params["CaO_6OBasis"] +
                     self.parameters_dict["g"] * X_cpx_params["Fe2_Papike_wang"] + self.parameters_dict["h"] * X_liq["H2O_liq"] + self.parameters_dict["i"])

        return T_C


@register_model
class Wang21(ModelManager):
    '''
    Wang, X., Hou, T., Wang, M., Zhang, C., Zhang, Z., Pan, R., Marxer, F., & Zhang, H. (2021). A new clinopyroxene thermobarometer for mafic to intermediate magmatic systems. European Journal of Mineralogy, 33(5), 621–637. https://doi.org/10.5194/ejm-33-621-2021

    require: Wt: SiO2	Wt: TiO2	Wt: Al2O3	Wt: Cr2O3	Wt: FeO	Wt: MnO	Wt: MgO	Wt: CaO	Wt: Na2O	Wt: K2O	Wt.H2O(melt for T)

    cpx_only 
    eq1 (P) and 2 (T)

    '''

    def __init__(self, T_P, comments=None):
        '''
        Initialize the model.

        Parameters:
            T_P (str): 
                The name of the model.
            comments (str): 
                Comments about the model or any other information.

        '''
        comments = comments
        self.T_P = T_P

        model = eq1_P() if T_P == "P" else eq2_T()
        standard_columns = model.standard_columns
        super().__init__(model, standard_columns, comments)

    
        
        self.model_name = "Wang et al., 2021"
        self.cpx_names = model.cpx_names
        self.liq_names = model.liq_names
        self.prediction_column_name = "P_kbar" if T_P == "P" else "T_C"
        self.cpx_only = model.cpx_only
        self.if_support_hydrous = model.if_support_hydrous
        self.require_water = model.require_water

        self.uncertainty = model.uncertainty

        # train, test, path
        from aims4pt.toolkit_utils import get_file_path
        X_cpx_train_pkl_name = "datapkl/X_cpx_train.pkl"
        X_liq_train_pkl_name = "datapkl/X_liq_train.pkl"
        self.X_cpx_train_pkl_path = get_file_path(
            Wang21_data, X_cpx_train_pkl_name)
        self.X_liq_train_pkl_path = get_file_path(
            Wang21_data, X_liq_train_pkl_name)
        X_cpx_test_pkl_name = "datapkl/X_cpx_test.pkl"
        X_liq_test_pkl_name = "datapkl/X_liq_test.pkl"
        self.X_cpx_test_pkl_path = get_file_path(
            Wang21_data, X_cpx_test_pkl_name)
        self.X_liq_test_pkl_path = get_file_path(
            Wang21_data, X_liq_test_pkl_name)

        self.initialize_model(self.X_cpx_train_pkl_path, self.X_cpx_test_pkl_path,
                              self.X_liq_train_pkl_path, self.X_liq_test_pkl_path)

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
        if self.T_P == "T":
            if X_liq is None:
                X_liq = pd.DataFrame(
                    np.zeros((X.shape[0], len(self.liq_names))), columns=self.liq_names, index=X.index)
            return self.model.predict(X, X_liq)
        else:
            return self.model.predict(X)
