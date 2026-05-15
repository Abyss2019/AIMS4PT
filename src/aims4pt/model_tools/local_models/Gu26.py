"""Equation-based lunar clinopyroxene and clinopyroxene–liquid models."""

from __future__ import annotations

import numpy as np

from aims4pt.data_tools.compositions import cpx_calculation
from aims4pt.model_tools.ModelManager import ModelManager
from aims4pt.model_tools.conventional_model import conventional_model
from aims4pt.utils import normalize_column_names


class T_cpx_only(conventional_model):
    """
    T(K) = 1310.8245 + 11.1071 * P(kbar)
           - 1669.0979 * X_cpx_Ti
           - 153.1122 * X_cpx_Al2O3
           - 3320.4226 * X_cpx_Mn
           + 138.7883 * X_cpx_Mg
           + 1145.8082 * X_cpx_Na
           + 1258.8258 * X_cpx_CaTi
    """

    def __init__(self):
        parameters_dict = {
            'b': 1310.8245,
            'w_P': 11.1071,
            'w_Ti': -1669.0979,
            'w_Al2O3': -153.1122,
            'w_Mn': -3320.4226,
            'w_Mg': 138.7883,
            'w_Na': 1145.8082,
            'w_CaTi': 1258.8258,
        }

        standard_columns = ['SiO2_cpx', 'TiO2_cpx', 'Al2O3_cpx', 'FeOt_cpx',
                            'MnO_cpx', 'MgO_cpx', 'CaO_cpx', 'Na2O_cpx', 'K2O_cpx', 'Cr2O3_cpx']
        self.cpx_only = True  # True or False
        self.require_water = False  # True or False

        super().__init__('thermometer_cpx_only', parameters_dict, standard_columns)

    def predict(self, X):
        X_cpx = X.copy()
        P_kbar = X_cpx['P_kbar']
        X_cpx = normalize_column_names(X_cpx, self.standard_columns)
        X_cpx_params = cpx_calculation(X_cpx)

        p = self.parameters_dict

        T_K = (
            p['b']
            + p['w_P'] * P_kbar
            + p['w_Ti'] * X_cpx_params['TiO2_6OBasis']
            + p['w_Al2O3'] * X_cpx_params['Al2O3_6OBasis']
            + p['w_Mn'] * X_cpx_params['MnO_6OBasis']
            + p['w_Mg'] * X_cpx_params['MgO_6OBasis']
            + p['w_Na'] * X_cpx_params['Na2O_6OBasis']
            + p['w_CaTi'] * X_cpx_params['CaTi']
        )
        
        T_C = T_K - 273.15

        return T_C


class P_cpx_only(conventional_model):
    """
    P(kbar) = -87.4464 + 0.0636 * T(K)
              + 101.6780 * X_cpx_Ti
              + 69.6729 * X_cpx_Al2O3
              - 120.2060 * X_cpx_Ca
              + 120.7142 * X_cpx_DiHd
    """

    def __init__(self):
        parameters_dict = {
            'b': -87.4464,
            'w_T': 0.0636,
            'w_Ti': 101.6780,
            'w_Al2O3': 69.6729,
            'w_Ca': -120.2060,
            'w_DiHd': 120.7142
        }

        standard_columns = ['SiO2_cpx', 'TiO2_cpx', 'Al2O3_cpx', 'FeOt_cpx',
                            'MnO_cpx', 'MgO_cpx', 'CaO_cpx', 'Na2O_cpx', 'K2O_cpx', 'Cr2O3_cpx']


        self.cpx_only = True  # True or False
        self.require_water = False  # True or False
        super().__init__('barometer_cpx_only', parameters_dict, standard_columns)

    def predict(self, X):
        X_cpx = X.copy()
        T_C = X_cpx['T_C']
        T_K = T_C + 273.15
        X_cpx = normalize_column_names(X_cpx, self.standard_columns)
        X_cpx_params  = cpx_calculation(X_cpx)
        p = self.parameters_dict

        P_kbar = (
            p['b']
            + p['w_T'] * T_K
            + p['w_Ti'] * X_cpx_params['TiO2_6OBasis']
            + p['w_Al2O3'] * X_cpx_params['Al2O3_6OBasis']
            + p['w_Ca'] * X_cpx_params['CaO_6OBasis']
            + p['w_DiHd'] * X_cpx_params['DiHd']
        )
        return P_kbar


class Gu26(ModelManager):
    """
    Lunar clinopyroxene-liquid thermobarometer.

    Iterative calculation of T and P using lunar cpx-liquid equations.
    """
    T_model_options = ["T_cpx_only"]
    P_model_options = ["P_cpx_only"]

    def __init__(self, T_P, T_model_name, P_model_name,  iteration_max=100, stop_criteria=1e-2, comments=None):
        '''
        Initialize the model.

        Parameters:
            T_P (str):
                The output type. 'T' for temperature, 'P' for pressure.
            T_model_name (str):
                The name of the temperature model.
                if = "Input" then require "T_C" when predicting.
                Options: 'T_cpx_only'
            P_model_name (str):
                The name of the pressure model.
                if = "Input" then require "P_kbar" when predicting.
                Options: 'P_cpx_only'
            iteration_max (int):
                The maximum number of iterations for the model.
            stop_criteria (float):

                    The stop criteria for the model.
            comments (str):

                    Comments for the model. 
        '''

        model_options = {
            'T_cpx_only': T_cpx_only,
            'P_cpx_only': P_cpx_only
        }

        # check if the T_model and P_model are in the model_options
        if T_model_name not in model_options:
            raise ValueError(
                f"Invalid T_model: {T_model_name}. Available models: {list(model_options.keys())}")
        if P_model_name not in model_options:
            raise ValueError(
                f"Invalid P_model: {P_model_name}. Available models: {list(model_options.keys())}")

        # initialize the models
        model_dict = {
            T_model_name: model_options[T_model_name](),
            P_model_name: model_options[P_model_name]()
        }

        standard_columns_T = model_dict[T_model_name].standard_columns
        standard_columns_P = model_dict[P_model_name].standard_columns
        # combine the two lists and remove duplicates
        standard_columns = list(set(standard_columns_T + standard_columns_P))

        super().__init__(model_dict, standard_columns, comments)

        self.T_P = T_P
        self.iteration_max = iteration_max
        self.stop_criteria = stop_criteria
        self.T_model = model_dict[T_model_name]
        self.P_model = model_dict[P_model_name]

        self.cpx_only = self.T_model.cpx_only and self.P_model.cpx_only
        self.require_water = self.T_model.require_water and self.P_model.require_water

    def switch_PT(self, T_P):
        '''
        Switch the output type of the model.

        Parameters:
            T_P (str):
                The output type. 'T' for temperature, 'P' for pressure.
        '''
        if T_P not in ['T', 'P']:
            raise ValueError("Invalid T_P value. Must be 'T' or 'P'.")
        self.T_P = T_P
        if T_P == 'T':
            self.model = self.T_model
        else:
            self.model = self.P_model

    def predict(self, X):
        '''
        iterate the model until the stop criteria is met.
        Parameters:
            X (pd.DataFrame): 
                The input features。


        Returns:
            pd.Series: 
                The predicted response variable.
        '''

        # Format input and predict
        input = self.format_input(X)[self.standard_columns]

        # iterate the model until the stop criteria is met.
        iteration = 0
        stop_criteria = self.stop_criteria + 1
        # initialize the input with the first iterationsa
        input['T_C'] = 1500
        input['P_kbar'] = 20
        while iteration < self.iteration_max and stop_criteria > self.stop_criteria:
            iteration += 1
            # predict temperature and pressure
            T = self.T_model.predict(input)
            P = self.P_model.predict(input)

            # calculate the stop criteria
            stop_criteria_T = np.max(np.abs(input['T_C'] - T) / T)
            stop_criteria_P = np.max(np.abs(input['P_kbar'] - P) / P)
            stop_criteria = max(stop_criteria_T, stop_criteria_P)

            # update the input with the new temperature and pressure
            input['T_C'] = T
            input['P_kbar'] = P

        # print(f"Iteration: {iteration}, Stop criteria: {stop_criteria}")

        if self.T_P == 'T':
            return T
        else:
            return P
