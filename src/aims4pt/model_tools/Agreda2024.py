"""Ágreda-López et al. (2024) clinopyroxene thermobarometry wrapper."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aims4pt.model_tools.data.Agreda2024 as agreda_data
from aims4pt.model_tools.ModelManager import ModelManager
from aims4pt.model_tools.model_registry import register_model
from aims4pt.toolkit_utils import get_file_path


def _load_ml_pt_workflow():
    from aims4pt.model_tools.data.Agreda2024.ML_PT_Pyworkflow import ML_PT_Pyworkflow
    return ML_PT_Pyworkflow


@register_model
class Agreda2024(ModelManager):
    '''
    Machine Learning Thermo-Barometry for Clinopyroxene and Clinopyroxene-liquid systems.

    This model applies pre-trained models from ML_PT_Pyworkflow to estimate either 
    pressure (kbar) or temperature (°C) based on user-provided geochemical data.

    Reference: https://doi.org/10.1093/petrology/egac126
    '''

    def __init__(self, T_P: str, cpx_only: bool, comments: Optional[str] = None):
        '''
        Initialize the predictor with ML_PT_Pyworkflow.

        Parameters:
            T_P (str): "T" for temperature, "P" for pressure.
            cpx_only (bool): True for cpx_only, False for cpx_liq.
            comments (str): Additional metadata for the model.
        '''
        super().__init__(comments=comments)

        self.model_name = "Ágreda-López et al., 2024"
        self.T_P = T_P
        self.cpx_only = cpx_only
        self.require_water = False

        self.uncertainty = np.nan
        if self.cpx_only and self.T_P == "T":
            self.uncertainty = 57
        elif self.cpx_only and self.T_P == "P":
            self.uncertainty = 2.5
        elif not self.cpx_only and self.T_P == "T":
            self.uncertainty = 36
        elif not self.cpx_only and self.T_P == "P":
            self.uncertainty = 2.1

        self.prediction_column_name = "T_C" if T_P == "T" else "P_kbar"

        self.cpx_names = ['SiO2_cpx', 'TiO2_cpx', 'Al2O3_cpx', 'FeOt_cpx',
                          'MgO_cpx', 'MnO_cpx', 'CaO_cpx',  'Na2O_cpx', 'Cr2O3_cpx']
        self.liq_names = ['SiO2_liq', 'TiO2_liq', 'Al2O3_liq', 'FeOt_liq',
                          'MgO_liq', 'MnO_liq', 'CaO_liq',  'Na2O_liq', 'K2O_liq']

        standard_columns = self.cpx_names if cpx_only else self.cpx_names + self.liq_names
        self.standard_columns = standard_columns

        model = 'cpx_only' if cpx_only else 'cpx_liquid'
        output = 'Temperature' if T_P == 'T' else 'Pressure'

        # Initialize ML_PT_Pyworkflow.
        ml_pt_workflow = _load_ml_pt_workflow()
        scaler, predictor, bias_json = ml_pt_workflow.P_T_predictors(output, model)
        model = {
            'scaler': scaler,
            'predictor': predictor,
            'bias_json': bias_json
        }
        self.scaler = model['scaler']
        self.predictor = model['predictor']
        self.bias_json = model['bias_json']

        X_cpx_train_pkl_name = r"datapkl/X_T_cpx_train.pkl" if self.T_P == "T" else r"datapkl/X_P_cpx_train.pkl"
        X_liq_train_pkl_name = r"datapkl/X_T_liq_train.pkl" if self.T_P == "T" else r"datapkl/X_P_liq_train.pkl"
        self.X_cpx_train_pkl_path = get_file_path(
            agreda_data, X_cpx_train_pkl_name)
        self.X_liq_train_pkl_path = get_file_path(
            agreda_data, X_liq_train_pkl_name)

        X_cpx_test_pkl_name = r"datapkl/X_T_cpx_test.pkl" if self.T_P == "T" else r"datapkl/X_P_cpx_test.pkl"
        X_liq_test_pkl_name = r"datapkl/X_T_liq_test.pkl" if self.T_P == "T" else r"datapkl/X_P_liq_test.pkl"
        self.X_cpx_test_pkl_path = get_file_path(
            agreda_data, X_cpx_test_pkl_name)
        self.X_liq_test_pkl_path = get_file_path(
            agreda_data, X_liq_test_pkl_name)
        self.initialize_model(self.X_cpx_train_pkl_path, self.X_cpx_test_pkl_path,
                              self.X_liq_train_pkl_path, self.X_liq_test_pkl_path)
        
        self.if_normalize_liq = True

    def format_input(
        self,
        X_cpx: pd.DataFrame | None = None,
        X_liq: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        '''
        Format input data to match the expected feature names.

        Parameters:
            X_cpx (pd.DataFrame): Clinopyroxene composition.
            X_liq (pd.DataFrame, optional): Liquid composition.

        Returns:
            pd.DataFrame: Formatted input for prediction.
        '''
        from aims4pt.utils import normalize_column_names

        X_cpx_standard = normalize_column_names(X_cpx, self.cpx_names)
        # X_cpx_standard.rename(columns={col: col + "_cpx" for col in X_cpx_standard.columns}, inplace=True)
        # FeO_cpx to FeOt_cpx
        X_cpx_standard.rename(columns={'FeO_cpx': 'FeOt_cpx'}, inplace=True)

        if self.cpx_only:
            return X_cpx_standard
        else:
            X_liq_standard = normalize_column_names(X_liq, self.liq_names)
            # X_liq_standard.rename(columns={col: col + "_liq" for col in X_liq_standard.columns}, inplace=True)
            # FeO_liq to FeOt_liq
            X_liq_standard.rename(
                columns={'FeO_liq': 'FeOt_liq'}, inplace=True)
            # normalize liquid compositions ele = element / sum of all O
            sum_oxides = X_liq_standard.sum(axis=1)
            X_liq_standard = X_liq_standard.div(sum_oxides, axis=0) * 100.0
            
            return pd.concat([X_cpx_standard, X_liq_standard], axis=1)

    def predict(self, X_cpx: pd.DataFrame, X_liq: pd.DataFrame | None = None) -> pd.Series:
        '''
        Predict temperature or pressure using ML_PT_Pyworkflow.

        Parameters:
            X_cpx (pd.DataFrame): Clinopyroxene composition.
            X_liq (pd.DataFrame, optional): Liquid composition.

        Returns:
            pd.DataFrame: Predicted values with uncertainty bounds.
        '''
        p = 100
        # Preprocess input data.
        output = "T" if self.T_P == "T" else "P"
        # model = 'cpx_only' if self.cpx_only else 'cpx_liquid'

        ml_pt_workflow = _load_ml_pt_workflow()

        input_data = self.format_input(X_cpx, X_liq)
        input_data = ml_pt_workflow.data_imputation(input_data)

        for element in self.standard_columns:
            ml_pt_workflow.replace_zeros(input_data, element)
        Xd = input_data[self.standard_columns]
        X = Xd.to_numpy(dtype=np.float64, copy=True)

        # get training bounds
        pred_max_bound, pred_min_bound, warning = self.get_training_bounds(
            X, input_data)

        std_dev_perc = ml_pt_workflow.Parameters.oxide_rel_err

        K = ml_pt_workflow.Parameters.K_rel_err

        if self.cpx_only:
            std_dev_perc = std_dev_perc

        else:
            std_dev_perc_temp_1 = std_dev_perc
            std_dev_perc_temp_2 = std_dev_perc[:-1]
            std_dev_perc = np.concatenate(
                (std_dev_perc_temp_1, std_dev_perc_temp_2, K), axis=None)

        X_perturb, groups = ml_pt_workflow.input_perturbation(
            X, std_dev_perc, n_perturbations=p)

        X_perturb_s = self.scaler.transform(X_perturb)

        bias_popt_left = np.array(self.bias_json['slope']['left'])
        bias_popt_right = np.array(self.bias_json['slope']['right'])
        ang_left = self.bias_json['angle']['left']
        ang_right = self.bias_json['angle']['right']
        input_name = self.predictor.get_inputs()[0].name
        label_name = self.predictor.get_outputs()[0].name
        y_pred = self.predictor.run(
            [label_name], {input_name: X_perturb_s.astype(np.float32)})[0]

        unit = "kbar" if self.T_P == "P" else "C"

        if y_pred.shape[0] % p != 0:
            raise ValueError("Predictions count is not divisible by perturbation factor.")
        perturbation_blocks = y_pred[:, 0].reshape(-1, p)
        unique_y_pred = np.median(perturbation_blocks, axis=1)
        unique_y_perc_max = np.percentile(perturbation_blocks, 84, axis=1)
        unique_y_perc_min = np.percentile(perturbation_blocks, 16, axis=1)

        bias_temp = ml_pt_workflow.bias_f(unique_y_pred,
                                          ang_left, bias_popt_left,
                                          ang_right, bias_popt_right)

        unique_y_pred_temp = unique_y_pred - bias_temp
        unique_y_perc_max_temp = unique_y_perc_max - bias_temp
        unique_y_perc_min_temp = unique_y_perc_min - bias_temp

        # Bound of the training set
        unique_y_pred = np.minimum(pred_max_bound, np.maximum(
            pred_min_bound, unique_y_pred_temp))
        unique_y_perc_max = np.minimum(pred_max_bound, np.maximum(
            pred_min_bound, unique_y_perc_max_temp))
        unique_y_perc_min = np.minimum(pred_max_bound, np.maximum(
            pred_min_bound, unique_y_perc_min_temp))

        # error = (unique_y_perc_max - unique_y_perc_min)/2
        predictions = pd.DataFrame(
            data=np.column_stack([unique_y_pred, unique_y_perc_max, unique_y_perc_min]),
            columns=[output+'_'+unit, 'Percentile_84', 'Percentile_16'],
            index=X_cpx.index,
        )
        predictions['warning'] = warning

        return predictions[output+'_'+unit]

        # df_orig = input_data
        # df_orig['Sample_ID'] = df_orig['Sample_ID'].astype(str)

        # if model == 'cpx_only':
        #     df_orig = pd.concat([df_orig['Sample_ID'], df_orig[self.standard_columns]], axis=1)
        #     # output_file_name = 'out-files/Results_cpx_only_' + output + '.xlsx'

        # elif model == 'cpx_liquid':
        #     df_orig = pd.concat(
        #         [df_orig['Sample_ID'], df_orig[self.standard_columns]], axis=1)
        #     # output_file_name = 'out-files/Results_cpx_liquid_' + output + '.xlsx'

        # finaldf = pd.concat([df_orig, predictions], axis=1)
        # print(finaldf.head())

        # print(finaldf.head())
        # finaldf.to_excel(output_file_name, sheet_name=output, index=False)

    def plot_predictions(self, predictions):
        ''' Plot the histogram of predictions '''
        # unit = "kbar" if self.T_P == "P" else "C"
        color = '#9BB0C1' if self.T_P == "P" else '#D37676'

        plt.hist(predictions[self.prediction_column_name],
                 color=color, edgecolor='black')
        plt.xlabel(self.prediction_column_name)
        plt.ylabel("Frequency")
        plt.title(f"{self.prediction_column_name} Distribution")
        plt.grid(color='#B2B2B2', linestyle='--', linewidth=0.5, alpha=0.4)
        plt.show()

    def get_training_bounds(self, X, df):
        '''
        get_training_bounds for the model

        '''
        if self.cpx_only and self.T_P == "P":
            pred_max_bound = np.full(len(X), 30.0)
            pred_min_bound = np.zeros(len(X))
            mask = (
                df['Al2O3_cpx'].between(0.250000, 25.760000)
                & df['Na2O_cpx'].between(0.000137, 7.760000)
                & df['CaO_cpx'].between(0.400000, 24.820000)
            )
            warning = pd.Series(np.where(mask, '', 'Input out of bound'), index=df.index)
        elif self.cpx_only and self.T_P == "T":
            pred_max_bound = np.full(len(X), 1700.0)
            pred_min_bound = np.full(len(X), 700.0)
            mask = (
                df['MgO_cpx'].between(0.760000, 31.300000)
                & df['CaO_cpx'].between(0.400000, 24.820000)
                & df['Al2O3_cpx'].between(0.290000, 19.010000)
                & df['FeOt_cpx'].between(1.700000, 34.200000)
                & df['MnO_cpx'].between(0.000179, 2.980000)
            )
            warning = pd.Series(np.where(mask, '', 'Input out of bound'), index=df.index)
        elif not self.cpx_only and self.T_P == "P":
            pred_max_bound = np.full(len(X), 30.0)
            pred_min_bound = np.zeros(len(X))
            mask = (
                df['Al2O3_cpx'].between(0.250000, 25.760000)
                & df['Na2O_cpx'].between(0.000137, 7.760000)
                & df['MgO_liq'].between(0.030000, 18.587513)
            )
            warning = pd.Series(np.where(mask, '', 'Input out of bound'), index=df.index)
        elif not self.cpx_only and self.T_P == "T":
            pred_max_bound = np.full(len(X), 1700.0)
            pred_min_bound = np.full(len(X), 700.0)
            mask = df['MgO_liq'].between(0.030000, 17.723561)
            warning = pd.Series(np.where(mask, '', 'Input out of bound'), index=df.index)
        return pred_max_bound, pred_min_bound, warning
