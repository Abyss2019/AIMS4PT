"""Chicchi et al. (2023) GAIA deep learning thermobarometer wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd

import aims4pt.model_tools.data.Chicchi23 as chicchi_data
from aims4pt.model_tools.ModelManager import ModelManager
from aims4pt.model_tools.model_registry import register_model
from aims4pt.toolkit_utils import get_file_path


@register_model
class Chicchi23(ModelManager):
    '''
    Chicchi, L., Bindi, L., Fanelli, D., & Tommasini, S. (2023). Frontiers of thermobarometry: GAIA, a novel Deep Learning-based tool for volcano plumbing systems. Earth and Planetary Science Letters, 620. https://doi.org/10.1016/j.epsl.2023.118352


    '''

    def __init__(self, T_P, cpx_only, comments=None):
        '''
        Initialize with a model object and standard columns.

        Parameters:
            T_P (str):
                "T" for temperature, "P" for pressure.
            cpx_only (bool):
                If True, only clinopyroxene data is used for prediction.
                If False, both clinopyroxene and melt data are used for prediction.

            comments (str): 
                Additional comments or metadata for the model.
        '''

        super().__init__(comments=comments)
        self.model_name = "Chicchi et al., 2023"
        self.cpx_only = cpx_only

        self.require_water = False
        if T_P == "T":
            self.prediction_column_name = "T_C"
            if self.cpx_only:
                self.uncertainty = 28
            else:
                self.uncertainty = np.nan
        else:
            self.prediction_column_name = "P_kbar"
            if self.cpx_only:
                self.uncertainty = 0.9
            else:
                self.uncertainty = np.nan
        # melt (SiO2, TiO2, Al2O3, FeOt, MnO, MgO, CaO, Na2O, K2O, Cr2O3, P2O5, and H2O) and clinopyroxene (SiO2, TiO2, Al2O3, FeOt, MnO, MgO, CaO, Na2O, K2O, and Cr2O3)
        self.cpx_names = ["SiO2_cpx", "TiO2_cpx", "Al2O3_cpx", "Cr2O3_cpx", "FeOt_cpx",
                          "MnO_cpx", "NiO_cpx", "MgO_cpx",  "CaO_cpx", "Na2O_cpx", "K2O_cpx", ]
        self.liq_names = ["SiO2_liq", "TiO2_liq", "Al2O3_liq", "Cr2O3_liq", "FeOt_liq",
                          "MnO_liq", "NiO_liq", "MgO_liq", "CaO_liq", "Na2O_liq", "K2O_liq", "P2O5_liq"]
        if self.cpx_only:
            standard_columns = self.cpx_names
        else:
            standard_columns = self.cpx_names + self.liq_names
        
        self.standard_columns = standard_columns

        self.model_save_path = None # disposal
        self.models_dir = get_file_path(
            chicchi_data, "model")

        self.T_P = T_P
    

        self.if_support_hydrous = True


        X_cpx_train_pkl_name = r"datapkl/X_conly_cpx_train.pkl" if self.cpx_only else r"datapkl/X_cliq_cpx_train.pkl"
        X_liq_train_pkl_name = r"datapkl/X_conly_liq_train.pkl" if self.cpx_only else r"datapkl/X_cliq_liq_train.pkl"
        self.X_cpx_train_pkl_path = get_file_path(
            chicchi_data, X_cpx_train_pkl_name)
        self.X_liq_train_pkl_path = get_file_path(
            chicchi_data, X_liq_train_pkl_name)


        self.initialize_model(cpx_training_path=self.X_cpx_train_pkl_path,
                              liq_training_path=self.X_liq_train_pkl_path)
        self.if_normalize_liq = True
        
    def format_input(self, X_cpx, X_liq = None):
        '''
        Expand the normalize_column_names function which only support 1 phase a time.

        Default: format_columns_names function.

        Should be overridden in the subclass when 2 or more phases are involved.
        '''
        from aims4pt.utils import normalize_column_names
        from aims4pt.model_tools.data.Chicchi23.script.preprocessing import preprocessing_cpx
        X_cpx_norm_col = ["Index", "sample", "notes", "notes.1", "SiO2_cpx", "TiO2_cpx", "Al2O3_cpx", "Cr2O3_cpx", "FeOt_cpx", "MnO_cpx", "NiO_cpx", "MgO_cpx", "CaO_cpx", "Na2O_cpx", "K2O_cpx", "tot"]
        X_cpx_norm = normalize_column_names(X_cpx, X_cpx_norm_col)
        X_cpx_norm.fillna(0, inplace=True)
        X_cpx_norm = X_cpx_norm[X_cpx_norm_col]
        
        X_cpx_processed = preprocessing_cpx(X_cpx_norm)['input_NN']

        cpx_only_col = ['Index', 'sample', 'notes', 'notes.1','CaTiAl2O6', 'CaTs', 'Es', 'CaCrTs', 'NaCrSi2O6', 'Jd', 'Ae', 'Di', 'Hd', 'En(Mg+Ni)', 'Fs(Fe+Mn)']
        X_cpx_processed = X_cpx_processed[cpx_only_col]

        if self.cpx_only:
            return X_cpx_processed
        else:
            X_liq_norm = normalize_column_names(X_liq, self.liq_names)
            X_liq_norm.fillna(0, inplace=True)
            from aims4pt.data_tools.compositions import calculate_cation_fractions
            oxide_list = ["SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MnO", "NiO", "MgO", "CaO", "Na2O", "K2O", "P2O5"]
            X_liq_processed = calculate_cation_fractions(X_liq_norm, oxide_list)
            liq_col_names =['Si', 'Ti', 'Al', 'Fetot', 'Mg', 'Ca', 'Na', 'K']
            cpx_liq_col = ['Index', 'sample', 'notes', 'notes.1', 'Si', 'Ti', 'Al', 'Fetot', 'Mg', 'Ca', 'Na', 'K', 'CaTiAl2O6', 'CaTs', 'Es',
                           'CaCrTs', 'NaCrSi2O6', 'Jd', 'Ae', 'Di', 'Hd', 'En(Mg+Ni)', 'Fs(Fe+Mn)']
            X_liq_processed_norm = pd.DataFrame()
            map_dict = {"SiO2": "Si", "TiO2": "Ti", "Al2O3": "Al",
                        "FeO": "Fetot",  "MgO": "Mg", "CaO": "Ca", "Na2O": "Na", "K2O": "K"}
            for col in map_dict.keys():
                X_liq_processed_norm[col] = X_liq_processed[col].astype(float)
            X_liq_processed_norm.rename(columns=map_dict, inplace=True)
            X_input = pd.concat([X_cpx_processed, X_liq_processed_norm], axis=1)
            X_input = X_input[cpx_liq_col]
            return X_input

    def save(self):
        pass

    def predict(self, X_cpx, X_liq=None):
        '''
        Predict using the model.

        Parameters:
            X (pd.DataFrame): 
                Input DataFrame. The order of the columns does not matter, but it is recommended to use the standard column names.
                In principle, it only needs to contain all required features; order and exact names are not required,
                but using the standard column order is recommended.

        Returns:
            pd.Series: The predicted values.    
        '''
#         ['Index', 'sample', 'notes', 'notes.1','Si', 'Ti', 'Al', 'Fetot', 'Mg', 'Ca', 'Na', 'K', 'CaTiAl2O6', 'CaTs', 'Es',
#        'CaCrTs', 'NaCrSi2O6', 'Jd', 'Ae', 'Di', 'Hd', 'En(Mg+Ni)', 'Fs(Fe+Mn)']
        X_cpx = X_cpx.copy()

        if self.cpx_only:

            X_input = self.format_input(X_cpx=X_cpx)

            X_input.fillna(0, inplace=True)
            from aims4pt.model_tools.data.Chicchi23.script.functions import easy_predict
            prediction = easy_predict(X_input, T_P=self.T_P, dir=self.models_dir)
            
        elif not self.cpx_only:
            X_liq = X_liq.copy()
            X_input = self.format_input(X_cpx=X_cpx, X_liq=X_liq)
            X_input.fillna(0, inplace=True)
            from aims4pt.model_tools.data.Chicchi23.script.functions_cpx_liq import easy_predict
            prediction = easy_predict(X_input, T_P=self.T_P, dir=self.models_dir)
        else:
            raise ValueError("cpx_only must be True or False.")
        
        # to pd.Series
        prediction = pd.Series(prediction, name=self.prediction_column_name, index=X_cpx.index)
        return prediction



