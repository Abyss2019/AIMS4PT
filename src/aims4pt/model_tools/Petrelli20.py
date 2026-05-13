"""Petrelli et al. (2020) clinopyroxene-bearing magma model wrapper."""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd

import aims4pt.model_tools.data.Petrelli20 as petrelli_data
from aims4pt.model_tools.ModelManager import ModelManager
from aims4pt.model_tools.model_registry import register_model
from aims4pt.toolkit_utils import get_file_path


@register_model
class Petrelli20(ModelManager):
    '''
    Petrelli, M., Caricchi, L., & Perugini, D. (2020). Machine Learning Thermo-Barometry: Application to Clinopyroxene-Bearing Magmas. Journal of Geophysical Research: Solid Earth, 125(9), e2020JB020130. https://doi.org/10.1029/2020JB020130


    '''

    def __init__(self, T_P, cpx_only, comments=None):
        '''
        Initialize with a model object and standard columns.

        Parameters:
            T_P (str):
                "T" for temperature, "P" for pressure.
            cpx_only (bool):
                True for cpx_only, False for cpx_liq.
            comments (str): 
                Additional comments or metadata for the model.
        '''

        super().__init__(comments=comments)
        self.model_name = "Petrelli et al., 2020"

        self.require_water = False
        if T_P == "T":
            self.prediction_column_name = "T_C"
            if cpx_only:
                model_name = "cpx_only_T"
                self.uncertainty = 96
            else:
                model_name = "cpx_liq_T"
                self.uncertainty = 51
                self.require_water = True
        else:
            self.prediction_column_name = "P_kbar"
            if cpx_only:
                model_name = "cpx_only_P"
                self.uncertainty = 3.2
            else:
                model_name = "cpx_liq_P"
                self.uncertainty = 2.9
                self.require_water = True
        # melt (SiO2, TiO2, Al2O3, FeOt, MnO, MgO, CaO, Na2O, K2O, Cr2O3, P2O5, and H2O) and clinopyroxene (SiO2, TiO2, Al2O3, FeOt, MnO, MgO, CaO, Na2O, K2O, and Cr2O3)
        self.cpx_names = ["SiO2_cpx", "TiO2_cpx", "Al2O3_cpx", "FeOt_cpx",
                          "MnO_cpx", "MgO_cpx", "CaO_cpx", "Na2O_cpx", "K2O_cpx", "Cr2O3_cpx"]
        self.liq_names = ["SiO2_liq", "TiO2_liq", "Al2O3_liq", "FeOt_liq", "MnO_liq",
                          "MgO_liq", "CaO_liq", "Na2O_liq", "K2O_liq", "Cr2O3_liq", "P2O5_liq", "H2O_liq"]
        if cpx_only:
            standard_columns = self.cpx_names
        else:
            standard_columns = self.cpx_names + self.liq_names
        
        self.standard_columns = standard_columns

        model_save_path = "model/{}.pkl".format(model_name)
        self.model_save_path = get_file_path(petrelli_data, model_save_path)

        self.T_P = T_P
        self.cpx_only = cpx_only

        self.if_support_hydrous = True

        X_cpx_train_pkl_name = "datapkl/X_cpx_train.pkl"
        X_liq_train_pkl_name = "datapkl/X_liq_train.pkl"
        self.X_cpx_train_pkl_path = get_file_path(
            petrelli_data, X_cpx_train_pkl_name)
        self.X_liq_train_pkl_path = get_file_path(
            petrelli_data, X_liq_train_pkl_name)

        X_cpx_test_pkl_name = "datapkl/X_cpx_test.pkl"
        X_liq_test_pkl_name = "datapkl/X_liq_test.pkl"
        self.X_cpx_test_pkl_path = get_file_path(
            petrelli_data, X_cpx_test_pkl_name)
        self.X_liq_test_pkl_path = get_file_path(
            petrelli_data, X_liq_test_pkl_name)
        self.initialize_model(self.X_cpx_train_pkl_path, self.X_cpx_test_pkl_path,
                              self.X_liq_train_pkl_path, self.X_liq_test_pkl_path)
        
        # X_cpx_all_pkl_name = "datapkl/X_cpx_all.pkl"
        # X_liq_all_pkl_name = "datapkl/X_liq_all.pkl"
        # self.X_cpx_all_pkl_path = get_file_path(
        #     petrelli_data, X_cpx_all_pkl_name)
        # self.X_liq_all_pkl_path = get_file_path(
        #     petrelli_data, X_liq_all_pkl_name)
        # #load all data
        # with open(self.X_cpx_all_pkl_path, "rb") as f:
        #     self.X_cpx_all = pickle.load(f)
        # with open(self.X_liq_all_pkl_path, "rb") as f:
        #     self.X_liq_all = pickle.load(f)
    def format_input(self, X_cpx, X_liq=None):
        '''
        Expand the normalize_column_names function which only support 1 phase a time.

        Default: format_columns_names function.

        Should be overridden in the subclass when 2 or more phases are involved.
        '''
        from aims4pt.utils import normalize_column_names
        X_cpx_standard = normalize_column_names(X_cpx, self.cpx_names)
        if self.cpx_only:
            return X_cpx_standard
        else:
            X_liq_standard = normalize_column_names(X_liq, self.liq_names)
            return pd.concat([X_cpx_standard, X_liq_standard], axis=1)

    def save(self):
        return super().save(self.model_save_path)

    def train(self):
        '''
        Initialize with model.

        '''
        def get_available_data(file_name):
            from aims4pt.utils import normalize_column_names

            myLiquids = pd.read_excel(file_name, usecols="B:M", skiprows=1)
            myLiquids = myLiquids.fillna(0)

            myCPXs = pd.read_excel(file_name, usecols="O:X", skiprows=1)
            myCPXs = myCPXs.fillna(0)
            myCPXs.columns = [c.replace('.1', '') for c in myCPXs.columns]

            Experimental_PT = pd.read_excel(
                file_name, usecols="Z:AA", skiprows=1)
            myLabels = pd.read_excel(file_name, usecols="A", skiprows=1)
            myCPXs = normalize_column_names(myCPXs, self.cpx_names)
            myLiquids = normalize_column_names(myLiquids, self.liq_names)
            # (myCPXs.head())
            return myLiquids, myCPXs, Experimental_PT, myLabels
        self_calibration_path = "data_set/GlobalDataset_Final_rev9_TrainValidation.xlsx"
        self_calibration_path = get_file_path(
            petrelli_data, self_calibration_path)
        X_liq, X_cpx, Experimental_PT, myLabels = get_available_data(
            self_calibration_path)


        X_cpx_training = X_cpx.copy()
        X_cpx_training["T_C"] = Experimental_PT["T_K"] - 273.15
        X_cpx_training["P_kbar"] = Experimental_PT["P_GPa"] * 10
        import pickle
        with open(self.X_cpx_train_pkl_path, "wb") as f:
            pickle.dump(X_cpx_training, f)
        with open(self.X_liq_train_pkl_path, "wb") as f:
            pickle.dump(X_liq, f)

        if self.T_P == "T":
            Y = np.array([Experimental_PT.T_K]).T
        else:
            Y = np.array([Experimental_PT.P_GPa * 10]).T
        if self.cpx_only:
            X = X_cpx.values
        else:
            X = pd.concat([X_cpx, X_liq], axis=1).values

        # Scaling Training Data
        from sklearn.preprocessing import StandardScaler
        # pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import ExtraTreesRegressor

        # parameters
        if self.T_P == "T" and self.cpx_only:
            n_estimators, max_features, random_state = 650, 10, 280
        elif self.T_P == "T" and not self.cpx_only:
            n_estimators, max_features, random_state = 550, 22, 280
        elif self.T_P == "P" and self.cpx_only:
            n_estimators, max_features, random_state = 450, 10, 120
        else:
            n_estimators, max_features, random_state = 350, 22, 80

        # Pipeline
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", ExtraTreesRegressor(n_estimators=n_estimators, criterion='squared_error',
                                              max_features=max_features, random_state=random_state))
        ])

        self.model.fit(X, Y.ravel())
        self.save()

        # Evaluate the model
        test_data_path = "data_set/GlobalDataset_Final_rev9_Test.xlsx"
        test_data_path = get_file_path(petrelli_data, test_data_path)
        X_liq_test, X_cpx_test, Experimental_PT_test, myLabels_test = get_available_data(
            test_data_path)

        

        if self.T_P == "T":
            Y_test = np.array([Experimental_PT_test.T_K]).T
            # K to C
            Y_test = Y_test - 273.15
        else:
            Y_test = np.array([Experimental_PT_test.P_GPa * 10]).T
        if self.cpx_only:
            X_test = X_cpx_test.values
        else:
            X_test = pd.concat([X_cpx_test, X_liq_test], axis=1).values

        X_cpx_test["T_C"] = Experimental_PT_test["T_K"] - 273.15
        X_cpx_test["P_kbar"] = Experimental_PT_test["P_GPa"] * 10
        import pickle
        with open(self.X_cpx_test_pkl_path, "wb") as f:
            pickle.dump(X_cpx_test, f)
        with open(self.X_liq_test_pkl_path, "wb") as f:
            pickle.dump(X_liq_test, f)
        y_pred = self.model.predict(X_test)


        
        if self.T_P == "T":
            y_pred = y_pred - 273.15
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(Y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, y_pred)
        # self.uncertainty = rmse
        print(f"RMSE: {rmse}, R2: {r2}")
        self.initialize_model(self.X_cpx_train_pkl_path, self.X_cpx_test_pkl_path,
                              
                              self.X_liq_train_pkl_path, self.X_liq_test_pkl_path)
        

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
        X_cpx = X_cpx.copy()
        if self.cpx_only:
            input = self.format_input(X_cpx)[self.standard_columns]
        else:
            X_liq = X_liq.copy()
            input = self.format_input(X_cpx, X_liq)[self.standard_columns]
        # save the input fil
        input = input.fillna(0)
        input = input.values
        prediction = self.model.predict(input)
        if self.T_P == "T":
            prediction = prediction - 273.15

        # to pd.Series
        prediction = pd.Series(prediction, name=self.prediction_column_name, index=X_cpx.index)
        return prediction

    def get_feature_importance_df(self):
        '''
        Get the feature importance of the model.

        Returns:
            DataFrame: Feature importance.
            ```importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
            })
        '''

        # get model from the pipeline
        tree_model = self.model.named_steps["regressor"]

        feature_importance = tree_model.feature_importances_
        print(feature_importance)
        importance_df = pd.DataFrame({
            "Feature": self.standard_columns,
            "Importance": feature_importance
        })
        importance_df = importance_df.sort_values(
            by="Importance", ascending=False)
        return importance_df

    def plot_feature_importance(self, ax=None, first_k=None, x_label="Feature Importance", title=None):
        '''
        Plot the feature importance of the model.

        Parameters:

            ax (matplotlib.axes.Axes, optional):
                Axes object to plot on. Default is None.

            first_k (int, optional):
                Number of top features to display. Default is None (display all).

            x_label (str, optional):
                Label for the x-axis. Default is "Feature Importance".
        '''
        from aims4pt.visualization.mode_related_plot import plot_feature_importance
        importance_df = self.get_feature_importance_df()
        title = "Thermometer" if self.T_P == "T" else "Barometer"
        if self.cpx_only:
            title += " (Cpx-only)"
        else:
            title += " (Cpx-Liq)"

        if ax is None:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(dpi=150)

        plot_feature_importance(
            importance_df, title=title, ax=ax, first_k=first_k, x_label=x_label)
        return importance_df
