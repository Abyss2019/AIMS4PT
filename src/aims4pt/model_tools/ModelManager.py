"""Base abstractions for wrapping thermobarometry and classification models."""

from __future__ import annotations

import pickle
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
# print("Pandas imported.")

from aims4pt.utils import filter_oxides, normalize_column_names
# print("Utils imported.")
import aims4pt.model_tools.trained_model.OOD_detectors as OOD_detectors_package
import aims4pt.model_tools.trained_model.Deviation_functions as Deviation_functions_package
import aims4pt.model_tools.trained_model.SHAP as SHAP_package

from aims4pt.toolkit_utils import get_file_path, load_pickle_compat

class BaseModelManager:
    """
    Handles model initialization, saving, loading, and prediction.
    """

    def __init__(self, model=None, standard_columns: Optional[Iterable[str]] = None, comments: Optional[str] = None):
        self.model = model

        self.comments = comments
        self.standard_columns = list(standard_columns) if standard_columns is not None else None
        self.uncertainty = np.nan
        self.model_save_path = None  # Only used for pickle persistence
        self.if_support_hydrous = True

        # self.model_name = None
        # self.T_P = None
        # self.cpx_only = None
        # self.require_water = None

        # self.cpx_names = None
        # self.liq_names = None

        # self.X_cpx_training = None
        # self.X_liq_training = None
        # self.X_cpx_test = None
        # self.X_liq_test = None

        # self.TAS_rock_types = None # e.g., basalt, andesite, rhyolite
        # self.volcanic_rock_series = None # e.g., tholeiitic, calc-alkaline, alkaline

        # self.deviation_function = None
        # self.OOD_detector = None

    def save(self, file_path: str) -> None:
        if self.model is None:
            raise ValueError("No model to save.")
        
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model_pkl(self, file_path: Optional[str] = None) -> None:
        target_path = file_path or self.model_save_path
        if target_path is None:
            raise ValueError("No model_save_path is set for loading.")

        with open(target_path, 'rb') as f:
            self.model = load_pickle_compat(f)

    def initialize_model(
            self,
            cpx_training_path: Optional[str] = None,
            cpx_test_path: Optional[str] = None,
            liq_training_path: Optional[str] = None,
            liq_test_path: Optional[str] = None,
    ) -> None:
        '''
        The universal function to initialize the model.
        Called during initialization.

        Shared setup that is easier to keep together.

        1. load model based on the setting self.model_save_path (if not None)
        2. load training and test data.

        '''
        import os
        self.X_cpx_training = None
        self.X_liq_training = None
        self.X_cpx_test = None
        self.X_liq_test = None
        self.X_cpx_all = None
        self.X_liq_all = None
        self.rock_types = None
        self.volcanic_rock_series = None
        
        self.model_name = self.model_name + f" ({'cpx_only' if self.cpx_only else 'cpx_liq'})"
        # load model path
        if self.model_save_path is not None:
            if not os.path.exists(self.model_save_path):
                ("Model not found in the path: {}".format(self.model_save_path))
                return
            else:
                self.load_model_pkl(self.model_save_path)

        # load training and test data
        if cpx_training_path is not None and not os.path.exists(cpx_training_path):
            return
        if cpx_training_path is not None:
            with open(cpx_training_path, 'rb') as f:
                self.X_cpx_training = pd.read_pickle(f)
                self.X_cpx_all = self.X_cpx_training.copy()
                target_column = "P_kbar" if self.T_P == "P" else "T_C"
                self.y_min = self.X_cpx_training[target_column].min()
                self.y_max = self.X_cpx_training[target_column].max()
                #2.5% – 97.5%
                self.y_min_95 = self.X_cpx_training[target_column].quantile(0.025)
                self.y_max_95 = self.X_cpx_training[target_column].quantile(0.975)

        if liq_training_path is not None:
            from aims4pt.data_tools.rocks import get_TAS_rock_types, get_volcanic_rock_series
            with open(liq_training_path, 'rb') as f:
                self.X_liq_training = pd.read_pickle(f)
                self.X_liq_all = self.X_liq_training.copy()
                self.rock_types = get_TAS_rock_types(self.X_liq_training)
                self.volcanic_rock_series = get_volcanic_rock_series(self.X_liq_training)
        if cpx_test_path is not None:
            with open(cpx_test_path, 'rb') as f:
                self.X_cpx_test = pd.read_pickle(f)
                self.X_cpx_all = pd.concat(
                    [self.X_cpx_training, self.X_cpx_test], axis=0)
                self.X_cpx_all.reset_index(drop=True, inplace=True)
        if liq_test_path is not None:
            with open(liq_test_path, 'rb') as f:
                self.X_liq_test = pd.read_pickle(f)
                self.X_liq_all = pd.concat(
                    [self.X_liq_training, self.X_liq_test], axis=0)
                self.X_liq_all.reset_index(drop=True, inplace=True)
        shap_path = get_file_path(SHAP_package, f"{self.model_name}_{self.T_P}_shap_df_default.pkl")
        feature_importance_path = get_file_path(SHAP_package, f"{self.model_name}_{self.T_P}_feature_importance_default.pkl")
        Deviation_functions_path = get_file_path(Deviation_functions_package, f"{self.T_P}_{self.model_name}_deviation_function_v_ind.pkl")
        OOD_detectors_path = get_file_path(OOD_detectors_package, f"{self.model_name}_{self.T_P}_weighted.pkl")
        if os.path.exists(Deviation_functions_path):
            with open(Deviation_functions_path, "rb") as f:
                self.deviation_function = load_pickle_compat(f)
        if os.path.exists(OOD_detectors_path):
            with open(OOD_detectors_path, "rb") as f:
                self.OOD_detector = load_pickle_compat(f)
        if os.path.exists(shap_path):
            with open(shap_path, "rb") as f:
                self.shap_df = load_pickle_compat(f)
        if os.path.exists(feature_importance_path):
            with open(feature_importance_path, "rb") as f:
                self.feature_importance_df = load_pickle_compat(f)
        # try:
        #     print(self.rock_types)
        # except AttributeError:
        #     raise AttributeError(
        #         "Rock types not found. Please ensure that the liquid phase training data is loaded correctly.")

        # cluster dataset.
        # if cpx_training_path is not None:
        #     cluster_path = os.path.join(
        #         os.path.dirname(cpx_training_path), "training_data_cluster_model.pkl")
        #     if os.path.exists(cluster_path):
        #         with open(cluster_path, 'rb') as f:
        #             self.dataset_cluster_model = pickle.load(f)
        #     else:
        #         normalized_training_dataset = self.get_normed_training_dataset()
        #         additional_info_columns = ["P_kbar", "T_C"]
        #         additional_info = self.X_cpx_training.reindex(
        #             columns=additional_info_columns, 
        #             index=normalized_training_dataset.index
        #         ).fillna(np.nan)
                
        #         # from aims4pt.statistic_tools.nearest_data import my_dataset_cluster
        #         # # additional_info includes P_kbar and T_C if available.

        #         # self.dataset_cluster_model = my_dataset_cluster(normalized_training_dataset, additional_info=additional_info)
        #         # self.dataset_cluster_model.fit()

        #         from aims4pt.statistic_tools.nearest_data import MyDatasetNeighbors

        #         self.dataset_cluster_model = MyDatasetNeighbors(
        #             normalized_training_dataset, additional_info=additional_info)
        #         self.dataset_cluster_model.fit()
                
                

    def format_input(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Format the input DataFrame to match the model's expected feature order.

        Expand the normalize_column_names function which only support 1 phase a time.

                Default: format_columns_names function.

        Parameters:
            X (pd.DataFrame): The input features.

        Returns:
            pd.DataFrame: Formatted DataFrame.
        '''
        if self.standard_columns:
            X = normalize_column_names(X, self.standard_columns)

        return X

    def predict(self, X: pd.DataFrame, X2: Optional[pd.DataFrame] = None):
        '''
        Predict using the model.

        Parameters:
            X (pd.DataFrame): The input features.
            X2 (pd.DataFrame, optional): Additional input features. (Not used in this implementation)

            Returns:
            pd.Series: The predicted response variable.
        '''

        # Format input and predict
        input = self.format_input(X)[self.standard_columns]
        # save the input file

        y = self.model.predict(input)

        return y

class DatasetManager:
    """
    Manages dataset-related operations.

    """

    def get_normed_training_dataset(self):
        """
        Get the normalized training dataset.
        Returns
        -------
        pd.DataFrame
            The normalized training dataset.
        """
        if self.X_cpx_training is None:
            raise ValueError("No training dataset available.")
        if self.X_liq_training is not None and (not self.cpx_only or self.require_water):
            X = pd.concat([self.X_cpx_training, self.X_liq_training], axis=1)
        else:
            X = self.X_cpx_training
        return normalize_column_names(X, self.standard_columns)
    
    
    

    
    

class FeatureManager:
    """
    Manages feature-related operations.

    This class provides methods to set and retrieve feature information,
    as well as to visualize feature importance.

    Example
    -------
    >>> feature_manager = FeatureManager()
    >>> feature_manager.set_feature_info(["SiO2", "Al2O3"])
    >>> feature_manager.set_key_features(["SiO2", "Al2O3"], mode="manual")
    """

    def set_feature_info(self, standard_columns: List[str]) -> None:
        """
        Set the standard columns for the model.

        Parameters
        ----------
        standard_columns : list of str
            List of standard column names to normalize and order.

        Example
        -------
        >>> feature_manager.set_feature_info(["SiO2", "Al2O3"])
        """
        self.standard_columns = standard_columns

    def load_set_feature_info(self, path: str) -> None:
        """
        Load and set feature information from a file.

        Parameters
        ----------
        path : str
            Path to the file containing feature information. Supported formats are CSV and Excel.

        Raises
        ------
        ValueError
            If the file format is not supported.

        Example
        -------
        >>> feature_manager.load_set_feature_info("features.csv")
        """
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith('.xlsx'):
            df = pd.read_excel(path)
        else:
            raise ValueError('Unsupported file format.')

        standard_columns = filter_oxides(df.columns.tolist())
        self.set_feature_info(standard_columns)

    def set_key_features(self, key_features=None, mode="manual", num_features=None, accumulate_threshold=0.9) -> None:
        """
        Set the key features for the model.

        Parameters
        ----------
        key_features : list, optional
            List of key features to be set.
        mode : str, optional
            The mode for setting key features. Can be "manual" or "automated".
            - "manual": The user will set the key features manually.
            - "automated_old": The key features will be set automatically based on the model's feature importance.
            - "automated_kneedle": implement the kneedle algorithm.
            - "automated_accumulate": The key features will be set based on accumulated importance until a 90% threshold is reached.
            - "all": All features will be set as key features.

        Raises
        ------
        ValueError
            If `key_features` is None in manual mode or not a list.

        Example
        -------
        >>> feature_manager.set_key_features(["SiO2", "Al2O3"], mode="manual")
        """

        if mode == "manual":
            if key_features is None:
                raise ValueError("key_features cannot be None in manual mode.")
            if not isinstance(key_features, list):
                raise ValueError("key_features should be a list.")
            self.key_features = key_features

        elif mode == "automated_kneedle":
            if self.feature_importance_df is None:
                print(
                    "Feature importance DataFrame is not available in the model. Setting key features to standard columns.")
                self.key_features = self.standard_columns
                return
                
            # Sort features by importance in descending order
            sorted_features = self.feature_importance_df.sort_values(
                by='Importance', ascending=False)
            if num_features is not None:
                sorted_features = sorted_features.head(num_features)
                key_features = sorted_features['Feature'].tolist()
                self.key_features = key_features
                return

            sum_threshold = 0.68  # 1 sigma
            decline_threshold = 0.7
            sum_of_importance = 0
            key_features = []

            # Initialize the last importance value
            last_importance = None

            for _, feature in sorted_features.iterrows():
                # If cumulative importance reaches the threshold
                if sum_of_importance >= sum_threshold:
                    # Check if the importance has declined beyond the threshold
                    if last_importance is not None and feature['Importance'] / last_importance < decline_threshold:
                        break
                # Update the last importance value
                last_importance = feature['Importance']
                # Accumulate the importance
                sum_of_importance += feature['Importance']
                # Add the feature to the key features list
                key_features.append(feature['Feature'])

            self.key_features = key_features
        elif mode == "all":
            sorted_features = self.feature_importance_df.sort_values(by='Importance', ascending=False)
            self.key_features = sorted_features['Feature'].tolist()
        elif mode == "automated_accumulate":
            if self.feature_importance_df is None:
                print(
                    "Feature importance DataFrame is not available in the model. Setting key features to standard columns.")
                self.key_features = self.standard_columns
                return
            # Sort features by importance in descending order
            sorted_features = self.feature_importance_df.sort_values(
                by='Importance', ascending=False)
            sum_threshold = accumulate_threshold
            sum_of_importance = 0
            key_features = []
            for _, feature in sorted_features.iterrows():
                # If cumulative importance reaches the threshold
                if sum_of_importance >= sum_threshold:
                    break
                # Accumulate the importance
                sum_of_importance += feature['Importance']
                # Add the feature to the key features list
                key_features.append(feature['Feature'])
            self.key_features = key_features
        else:
            raise ValueError("Invalid mode for setting key features.")

    def feature_importance_plot(self, feature_importance_df: Optional[pd.DataFrame] = None, top_n: Optional[int] = None, show_key_features: bool = True):
        """
        Plot the feature importance.

        Parameters
        ----------
        feature_importance_df : pd.DataFrame, optional
            The feature importance DataFrame. If None, will use the `self.feature_importance_df`.
        top_n : int, optional
            The number of top features to plot. If None, will plot all features.
        show_key_features : bool, optional
            Whether to highlight key features in the plot.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object for the plot.

        Example
        -------
        >>> feature_manager.feature_importance_plot(top_n=10, show_key_features=True)
        """
        from aims4pt.visualization.mode_related_plot import plot_feature_importance
        import matplotlib.pyplot as plt

        feature_importance_df = feature_importance_df or self.feature_importance_df
        if feature_importance_df is None:
            raise ValueError("Feature importance DataFrame is not available in the model.")
        colored_features = self.key_features if show_key_features else None
        # print(colored_features)
        ax = plot_feature_importance(
            feature_importance_df,
            title=f"Feature Importance for {self.model_name}",
            first_k=top_n,
            colored_features=colored_features,
        )
        
        return ax
    
    # export compositional range df and text
    def export_composition_range(self, interested_dataset: str = "training",
                                   key_features: bool = True,
                                   sort_by_importance: bool = True,
                                   return_type: str = "df",):
        """
        Export compositional range DataFrame or text.

        Parameters
        ----------
        interested_dataset : str, optional
            The dataset to use for the compositional range. Can be "training" or "test" or "all".
            Default is "training".
        key_features : bool, optional
            Whether to use key features for the compositional range. If True, will use the key features set by `set_key_features`, otherwise will use all features.
            Default is True.
        sort_by_importance : bool, optional
            Whether to sort the compositional range by feature importance. Default is True.
            If True, the features will be sorted by their importance scores before exporting the range.
        return_type : str, optional
            The type of output to return. Can be "df" for DataFrame or "text" for formatted text.
            Default is "df".

        Returns
        -------
        pd.DataFrame or str
            The compositional range DataFrame or formatted text.
        """
        from aims4pt.utils import export_min_max_range_df
        X = pd.DataFrame()
        if interested_dataset == "training":
            if self.X_cpx_training is not None:
                X = self.X_cpx_training.copy()
            if self.X_liq_training is not None and (not self.cpx_only or self.require_water):
                X = pd.concat([X, self.X_liq_training], axis=1)
        elif interested_dataset == "test":
            if self.X_cpx_test is not None:
                X = self.X_cpx_test.copy()
            if self.X_liq_test is not None and (not self.cpx_only or self.require_water):
                X = pd.concat([X, self.X_liq_test], axis=1)
        elif interested_dataset == "all":
            if self.X_cpx_all is not None:
                X = self.X_cpx_all.copy()
            if self.X_liq_all is not None and (not self.cpx_only or self.require_water):
                X = pd.concat([X, self.X_liq_all], axis=1)
            
        if X.empty:
            print("No data available for the specified dataset.")
            return None

        if key_features:
            if self.key_features is None:
                raise ValueError(
                    "Key features are not set. Please set key features using `set_key_features` method.")
            X = X[self.key_features]
        else:
            X = X[self.standard_columns]

        if X.empty:
            raise ValueError(
                "No data available for the specified dataset. Please check the dataset paths.")
        
        try:
            min_max_range_df = export_min_max_range_df(X)
        except Exception as e:
            print(f"X is: {X}")
            raise ValueError(f"Error exporting compositional range: {e}")
        
        if sort_by_importance and hasattr(self, 'feature_importance_df'):
            # Sort the DataFrame by feature importance if available
            if self.feature_importance_df is not None:
                feature_order = self.feature_importance_df['Feature'].tolist()
                min_max_range_df = min_max_range_df[feature_order]
        
        if return_type == "df":
            return min_max_range_df
        elif return_type == "text":
            # Convert DataFrame to formatted text
            text = ""
            for col in min_max_range_df.columns:
                text += f"{col}: {min_max_range_df[col].min().round(1)}–{min_max_range_df[col].max().round(1)}; "
                # remove the last semicolon and space
            text = text[:-2]  # Remove the last "; "
            return text

class SHAPManager:
    """
    Handles SHAP analysis and related methods.
    """

    @staticmethod
    def _import_shap():
        import shap
        return shap

    def get_baseline_value(self, X_data, model_predict, masker, sample_size=1):
        """
        Calculate the baseline value using SHAP.

        This function builds a temporary SHAP Explainer using the provided model prediction function and masker.
        It then computes the SHAP values for the first `sample_size` samples of the input data (X_data) and returns 
        the first element from the Explainer's base_values array as the baseline value. The baseline value is generally
        used as a reference output for tasks such as filling in missing prediction values.

        Parameters
        ----------
        X_data : pd.DataFrame or array-like
            The input data on which to calculate the baseline value.
        model_predict : callable
            The prediction function of the model (e.g., model.predict) to be used for computing SHAP values.
        masker : object
            The SHAP masker object that is used to create the background data for the Explainer.
        sample_size : int, optional
            The number of samples from X_data to use for calculating the baseline value (default is 1).

        Returns
        -------
        baseline_value : scalar or array-like
            The calculated baseline value. For regression tasks, this is typically a scalar, and for classification tasks,
            it might be a one-dimensional array.
        """
        shap = self._import_shap()

        # Create a temporary SHAP Explainer with the provided prediction function and masker.
        temp_explainer = shap.Explainer(model_predict, masker=masker)

        # Select a subset of the input data for baseline calculation.
        # If X_data is a DataFrame, use iloc for slicing; otherwise, assume array-like slicing works.
        if isinstance(X_data, pd.DataFrame):
            sample = X_data.iloc[:sample_size]
        else:
            sample = X_data[:sample_size]

        # Compute the SHAP values for the selected sample(s) silently.
        temp_shap_values = temp_explainer(sample, silent=True)

        # Extract the baseline value (typically the reference model output)
        baseline_value = temp_shap_values.base_values[0]

        return baseline_value

    def shap_calculation_1phases(self, test_X, background_data=None, sampling_test=None, sampling_bg=None,
                                 shap_sort=True,
                                 package_predict_func=True):
        shap = self._import_shap()

        def wrapped_predict_function(X, baseline_value):
            # Original model prediction.
            y_pred = self.predict(X)

            # If the output is pd.Series or pd.DataFrame:
            if isinstance(y_pred, pd.Series):
                y_pred.fillna(baseline_value, inplace=True)
            elif isinstance(y_pred, pd.DataFrame):
                y_pred.fillna(baseline_value, inplace=True)
            else:
                # Assume y_pred is a NumPy array.
                y_pred = np.nan_to_num(y_pred, nan=baseline_value)

            return y_pred

        shap.initjs()
        X = normalize_column_names(
            test_X, self.standard_columns, drop_missing=True)
        if background_data is None:
            background_data = X.copy()

        X.fillna(0, inplace=True)
        background_data.fillna(0, inplace=True)
        background_data = normalize_column_names(
            background_data, self.standard_columns, drop_missing=True)
        if sampling_bg is not None:
            bg_km = shap.kmeans(background_data, sampling_bg, round_values=False).data
            background_data_ = pd.DataFrame(bg_km, columns=background_data.columns)
            background_data = background_data_

        if sampling_test is not None:
            from aims4pt.model_tools.model_utils import random_sample_reduce_data
            X = random_sample_reduce_data(X, sampling_test, 42)

        # y_pred = self.predict(X)
        # y_background = self.predict(background_data)

        # X = X[~y_pred.isna()]
        # background_data = background_data[~y_background.isna()]

        # masker = shap.maskers.Partition(
        #     background_data, clustering='correlation')
        masker = shap.maskers.Independent(
             background_data)
        # get the baseline value
        baseline_value = self.get_baseline_value(X, lambda X: self.predict(
            X).values, masker, sample_size=1)


        if package_predict_func:
            # Use the wrapped predict function with baseline value
            explainer = shap.Explainer(
                lambda X: wrapped_predict_function(
                    X, baseline_value).values,
                masker=masker)
        else:
            # Use the original predict function
            explainer = shap.Explainer(
                lambda X: self.predict(X).values,
                masker=masker,)

        # Compute SHAP values for the entire dataset
        shap_values = explainer(X)

        # Convert to DataFrame
        shap_df = pd.DataFrame(
            shap_values.values,
            columns=X.columns,
            index=X.index[:len(shap_values.values)]
        )



        if package_predict_func:
            # check if baseline_value is the same as explainer
            try:
                assert baseline_value == shap_values.base_values[0]
            except:
                print("Warning: baseline_value is not the same as explainer.")
                print(f"baseline_value: {baseline_value}")
                print(f"explainer: {shap_values.base_values[0]}")
        feature_importance = (
            shap_df
            .replace([np.inf, -np.inf], np.nan)   # Replace infinite values with NaN for safe aggregation
            .abs()                                # Use absolute SHAP values
            .mean(skipna=True)                    # Compute mean importance per feature (column-wise), ignoring NaN
            .sort_values(ascending=False)         # Rank features by global SHAP importance
        )
        if shap_sort:
            feature_order = feature_importance.index.tolist()
            shap_df = shap_df[feature_order]
            X = X[feature_order]
        else:
            feature_order = self.standard_columns
            shap_df = shap_df[feature_order]
            X = X[feature_order]

        shap.summary_plot(shap_df.values, features=X,
                          feature_names=X.columns.tolist(), sort=shap_sort)

        # Compute feature importance
        feature_importance_percent = feature_importance / feature_importance.sum()
        feature_importance_df = pd.DataFrame(
            {'Importance': feature_importance_percent, 'Feature': feature_importance.index})

        return shap_df, feature_importance_df, shap_values

    def shap_calculation_2phases(self, test_X, test_X_liq, background_data, bg_liq, sampling_test,
                                 sampling_bg, shap_sort, package_predict_func):
        shap = self._import_shap()
        """
        Compute SHAP values for two phases and return feature importance.

        Parameters
        ----------
        test_X : pd.DataFrame
            Phase 1 test data.
        test_X_liq : pd.DataFrame
            Phase 2 test data. Liquid phase data.
        background_data : pd.DataFrame or None
            Phase 1 background data; use combined test data if None.
        bg_liq : pd.DataFrame
            Phase 2 background data. Liquid phase background data.
        sampling_test : int or None
            Number of test samples to randomly select; use all if None.
        sampling_bg : int or None
            Number of background samples to randomly select; use all if None.
        shap_sort : bool
            Whether to sort SHAP values by mean absolute value.
        package_predict_func : bool
            Use a wrapped predict function that fills NaN with baseline if True.

        Returns
        -------
        shap_df : pd.DataFrame
            SHAP values for each sample.
        feature_importance_df : pd.DataFrame
            Normalized feature importance.
        shap_values : object
            Raw SHAP values object.
        """
        # Normalize and combine phase data
        X1 = normalize_column_names(
            test_X.copy(), self.cpx_names, drop_missing=True)
        X2 = normalize_column_names(
            test_X_liq.copy(), self.liq_names, drop_missing=True)
        X_combined = pd.concat([X1, X2], axis=1)

        # Process background data
        if background_data is None:
            BG_combined = X_combined.copy()
        else:
            BG1 = normalize_column_names(
                background_data.copy(), self.cpx_names, drop_missing=True)
            BG2 = normalize_column_names(
                bg_liq.copy(), self.liq_names, drop_missing=True)
            BG_combined = pd.concat([BG1, BG2], axis=1)

        phase1_cols = self.cpx_names
        phase2_cols = self.liq_names

        # Fill missing values
        X_combined = X_combined.fillna(0)
        BG_combined = BG_combined.fillna(0)
        # Randomly sample background data if needed
        if sampling_bg is not None:
            bg_km = shap.kmeans(BG_combined, sampling_bg, round_values=False).data
            background_data_ = pd.DataFrame(bg_km, columns=BG_combined.columns)
            BG_combined = background_data_

        if sampling_test is not None:
            from aims4pt.model_tools.model_utils import random_sample_reduce_data
            X_combined = random_sample_reduce_data(
                X_combined, sampling_test, 42)

        # # drop nan in prediction
        # y_pred = self.predict(X_combined[phase1_cols], X_combined[phase2_cols])
        # y_pred = y_pred[~y_pred.isna()]
        # X_combined = X_combined.loc[y_pred.index]
        # bg_pred = self.predict(
        #     BG_combined[phase1_cols], BG_combined[phase2_cols])
        # bg_pred = bg_pred[~bg_pred.isna()]
        # BG_combined = BG_combined.loc[bg_pred.index]

        # Define prediction wrappers

        def predict_wrapper(X: pd.DataFrame):
            return self.predict(X[phase1_cols], X[phase2_cols])

        def wrapped_predict_function(X: pd.DataFrame, baseline_value: float):
            y = predict_wrapper(X)
            if isinstance(y, (pd.Series, pd.DataFrame)):
                return y.fillna(baseline_value)
            return np.nan_to_num(y, nan=baseline_value)

        # Create SHAP masker with correlation clustering
        # masker = shap.maskers.Partition(BG_combined, clustering='correlation')
        masker = shap.maskers.Independent(BG_combined)

        # Compute baseline value
        baseline_value = self.get_baseline_value(
            X_combined,
            lambda X: self.predict(X[phase1_cols], X[phase2_cols]).values,
            masker,
            sample_size=1
        )

        # Select predict function based on flag
        if package_predict_func:
            def predict_fn(X): return wrapped_predict_function(
                X, baseline_value).values
        else:
            def predict_fn(X): return self.predict(
                X[phase1_cols], X[phase2_cols]).values



        explainer = shap.Explainer(predict_fn, masker=masker)
        
        # Compute SHAP values and create DataFrame
        shap_values = explainer(X_combined)
        shap_df = pd.DataFrame(
            shap_values.values,
            columns=X_combined.columns,
            index=X_combined.index[:len(shap_values.values)]
        )

        # Check baseline consistency if using wrapped function
        if package_predict_func:
            import warnings
            try:
                if not np.isclose(baseline_value, shap_values.base_values[0]):
                    warnings.warn(
                        f"Inconsistent baseline: {baseline_value} vs {shap_values.base_values[0]}")
            except Exception:
                warnings.warn("Error comparing baseline values.")

        feature_importance = (
            shap_df
            .replace([np.inf, -np.inf], np.nan)   # Replace infinite values with NaN for safe aggregation
            .abs()                                # Use absolute SHAP values
            .mean(skipna=True)                    # Compute mean importance per feature (column-wise), ignoring NaN
            .sort_values(ascending=False)         # Rank features by global SHAP importance
        )

        # Sort SHAP values if required
        if shap_sort:
            feature_order = feature_importance.index.tolist()
        else:
            feature_order = list(X_combined.columns)
        shap_df = shap_df[feature_order]

        # Plot summary
        shap.summary_plot(shap_df.values, features=X_combined[feature_order],
                          feature_names=feature_order, sort=shap_sort)

        # Compute normalized feature importance
        feature_importance_normalized = feature_importance / feature_importance.sum()
        feature_importance_df = pd.DataFrame({
            'Feature': feature_importance.index,
            'Importance': feature_importance_normalized.values
        })

        return shap_df, feature_importance_df, shap_values

    def shap_calculation(self, test_X: pd.DataFrame, test_X_liq: Optional[pd.DataFrame] = None, background_data: Optional[pd.DataFrame] = None, bg_liq: Optional[pd.DataFrame] = None, sampling_test: Optional[int] = None,
                         sampling_bg: Optional[int] = None, shap_sort: bool = True, package_predict_func: bool = True):
        if background_data is None or background_data.empty:
            self.shap_df = None
            self.feature_importance_df = None
            self.shap_values = None
            print("No background data provided for SHAP calculation.")
            return None, None, None
        
        if not self.cpx_only or self.require_water:
            # 1. cpx-liq method; 2. cpx-only but require water
            shap_df, feature_importance_df, shap_values = self.shap_calculation_2phases(test_X, test_X_liq, background_data, bg_liq, sampling_test,
                                                                                        sampling_bg, shap_sort=shap_sort, package_predict_func=package_predict_func)

        else:
            shap_df, feature_importance_df, shap_values = self.shap_calculation_1phases(test_X, background_data, sampling_test, sampling_bg,
                                                                                        shap_sort=shap_sort, package_predict_func=package_predict_func)

        self.shap_df = shap_df
        print(feature_importance_df.head())
        self.feature_importance_df = feature_importance_df
        self.shap_values = shap_values
        return shap_df, feature_importance_df, shap_values


class AvailabilityManager:
    """
    Manages checking the model availability.
    """

    def P_T_range_check(self, P_min: float = np.nan, P_max: float = np.nan, T_min: float = np.nan, T_max: float = np.nan, check_both: bool = False):
        from aims4pt.statistic_tools.density_region_analysis import range_check
        import logging

        if self.X_cpx_training is None or self.X_cpx_training.empty:
            print("No training data available for range check.")
            return None

        logger = logging.getLogger(__name__)

        def validate_range(min_val, max_val, data, column_name, range_name):
            if not (pd.isna(min_val) or pd.isna(max_val)):
                if min_val > max_val:
                    raise ValueError(
                        f"Invalid {range_name} range: {min_val} > {max_val}.")
            if data is not None:
                result = range_check(
                    (min_val, max_val), data[column_name], quantile_experiments=1, quantile_samples=1)
                if not result:
                    logger.warning(f"{range_name} range check failed.")
                return result
            return True

        check_P = self.T_P == "P" or check_both 
        check_T = self.T_P == "T" or check_both
        pass_check = True

        if check_P:
            try:
                pass_check = pass_check and validate_range(
                    P_min, P_max, self.X_cpx_training, "P_kbar", "Pressure")
            except KeyError:
                print("Pressure column 'P_kbar' not found in training data. Skipping pressure range check.")
                pass_check = pass_check and True

        if check_T:
            try:
                pass_check = pass_check and validate_range(
                    T_min, T_max, self.X_cpx_training, "T_C", "Temperature")
            except KeyError:
                print("Temperature column 'T_C' not found in training data. Skipping temperature range check.")
                pass_check = pass_check and True

        return pass_check

    def get_X_best_exp(self, X, X_liq=None, threshold=45):
        """
        Get the best predicted samples based on the model predictions.
        T_column = "T_C"
        P_column = "P_kbar"

        Parameters
        ----------
        X : pd.DataFrame
            The input features.
        X_liq : pd.DataFrame, optional
            The input features for the liquid phase.
        threshold : float
            The threshold error (tolerating) for selecting the best samples. 

        Returns
        -------
        X_best : pd.DataFrame 
            DataFrame containing the best samples for the first phase.
        X_liq_best : pd.DataFrame (optional)
            DataFrame containing the best samples for the liquid phase.
        """
        # Ensure target column exists
        target_col = "T_C" if self.T_P == "T" else "P_kbar"
        if target_col not in X.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in input DataFrame.")
        
        if np.isnan(threshold):
            threshold = float('inf')
            y_pred = np.zeros(len(X))  # Dummy prediction if threshold is NaN
        else:
            # Get the predicted values
            if X_liq is not None:
                y_pred = self.predict(X, X_liq)
            else:
                y_pred = self.predict(X)

        # Ensure y_pred and X[target_col] have the same length
        if len(y_pred) != len(X):
            raise ValueError(
                "Mismatch between prediction length and input DataFrame length.")

        # Calculate the error
        error = np.abs(y_pred - X[target_col])

        # Get the best samples based on the error threshold
        best_samples = X[error < threshold].copy()

        # Get the corresponding liquid phase samples
        if X_liq is not None:
            best_samples_liq = X_liq[error < threshold].copy()
            return best_samples, best_samples_liq
        else:
            return best_samples

    def multidimensional_KDE_similarity_check(self, data_query, uncertainty=None, data_reference_cpx=None,
                                              data_reference_liq=None,
                                              interested_features=None, confidence_level=0.9973,
                                              feature_weights=True):
        """
        Perform a multidimensional kernel density estimation (KDE) similarity check.

        Parameters
        ----------
        data_query : pd.DataFrame
            The data to be checked against the reference. Pass it in after normalization.
        uncertainty : float, optional
            The uncertainty of the model. If None, the model's reported uncertainty will be used.
        data_reference_cpx : pd.DataFrame, optional
            The reference data for comparison for cpx.
            If None, the model's training+test (cpx) data will be used.
        data_reference_liq : pd.DataFrame, optional
            The reference data for comparison for liquid phase.
            If None, the model's training+test liquid data will be used.
        interested_features : List[str] | None, optional
            The features to be used for the analysis.
            if None, the self.standard_columns will be used.
        confidence_level : float, optional
            The confidence level for the analysis.
            Default is 0.9973 (99.73% confidence level).
        feature_weights : bool, optional
            Whether to apply feature weights in the analysis.
            Default is True.
            Will use the feature importance dataframe saved in the model.

        Returns
        -------
        mask : np.ndarray
            The mask indicating the density region.
        fig : matplotlib.figure.Figure
            The figure object for the plot.
        ax : matplotlib.axes.Axes
            The axes object for the plot.
        kde_paras : dict
            The parameters used in the KDE analysis.
            Dictionary with keys: 'kde', 'log_thresh', 'pass_percent', 'density_mean', 'log_density_mean', 'mean_percentile'.


        """

        from aims4pt.statistic_tools.density_region_analysis import check_density_region

        if uncertainty is None:
            uncertainty = self.uncertainty
            if np.isnan(uncertainty):
                Warning(
                    "Since uncertainty is not accounted for in the model, we cannot identify the best‐fit experimental data; instead, we use all available data.")

        
    
        if data_reference_cpx is None:
            data_reference_cpx = self.X_cpx_all
            # data_reference_cpx = normalize_column_names(
            #     self.X_cpx_all, self.cpx_names, drop_missing=False)
            if not self.cpx_only or self.require_water:
                # if self.X_liq_all is None:
                #     raise ValueError(
                #         "No liquid train or test data available in the model.")
                # data_reference_liq = normalize_column_names(
                #     self.X_liq_all, self.liq_names, drop_missing=False)
                data_reference_liq = self.X_liq_all

        # deal with no input       
        if data_reference_cpx is None or data_reference_cpx.empty:
            print("No reference data input, returning None.")
            kde_para_dict = {
            "kde": None,
            "log_density_array": None,
            "Density percentile": None,
            "log_thresh": None,
            "pass_percent": None,
            "log_density_mean": None,
            "min_percentile": None,
            "median_percentile": None,
            "max_percentile": None,
            }
            return None, None, None, kde_para_dict
        
        # get best n exps
        if not self.cpx_only or self.require_water:
            X_best_cpx, X_best_liq = self.get_X_best_exp(
                data_reference_cpx, data_reference_liq, threshold=uncertainty
            )
            data_reference = pd.concat(
                [X_best_cpx[self.cpx_names], X_best_liq[self.liq_names]], axis=1)
        else:
            X_best_cpx = self.get_X_best_exp(
                data_reference_cpx, threshold=uncertainty
            )
            data_reference = X_best_cpx[self.cpx_names].copy()

        if interested_features is None:
            interested_features = self.feature_importance_df["Feature"]
        if feature_weights:
            if self.feature_importance_df is None:
                raise ValueError(
                    "Feature importance DataFrame is not available in the model.")
            feature_weights: dict = self.feature_importance_df.set_index(
                'Feature')['Importance'].to_dict()
        else:
            feature_weights = None

        (mask, density), (fig, ax), kde_paras = check_density_region(
            data_query=data_query,
            data_reference=data_reference,
            features=interested_features,
            feature_weights=feature_weights,
            confidence_level=confidence_level,
            full_return=True,
            if_show_plot=True,
        )
        return mask, fig, ax, kde_paras


class ModelManager(BaseModelManager, DatasetManager, FeatureManager, SHAPManager, AvailabilityManager):
    '''
    Manage models and handle input formatting.

    This class is designed to work with the existing ml model.
    '''

    def __init__(self, model=None, standard_columns=None, comments=None):
        '''
        Initialize with model and optional feature order/standard columns.

        Parameters:
        model: model object; python model object or fake sci-kit learn model object that wraps an R model (r_model).

        standard_columns (list): List of standard column names to normalize and order. e.g. ['SiO2.n.', 'Al2O3.n.', 'FeO.n.', ...]

        comments (str): Comments about the model or any other information.

        '''
        super().__init__(model, standard_columns, comments)

        # temporary storage
        self.shap_df = None
        self.feature_importance_df = None
        self.shap_values = None
        self.key_features = None

        self.grid_uncertainty = None  # for grid prediction uncertainty

    @staticmethod
    def filter_models(
        model_list,
        model_name=None,
        T_P=None,
        cpx_only=None,
        exact=True,
        raise_if_empty=False,
    ):
        """
        Filter model instances from an existing model pool.

        Parameters
        ----------
        model_list : iterable
            Existing model pool to search. Each model is expected to expose a
            ``model_name`` attribute. ``T_P`` and ``cpx_only`` attributes are
            optional, but models without those attributes will not match when
            the corresponding filter is provided.
        model_name : str, optional
            Target model name. If ``None``, model names are not used as a
            filter. The value is matched against each model's ``model_name``
            attribute, not the class name.
        T_P : {"T", "P"}, optional
            Target thermobarometric parameter. ``"T"`` returns only
            temperature models and ``"P"`` returns only pressure models. If
            ``None``, both temperature and pressure models are allowed.
        cpx_only : bool, optional
            Phase-type filter. ``True`` returns only cpx-only models,
            ``False`` returns only cpx-liq models, and ``None`` does not
            restrict by phase type.
        exact : bool, default True
            Controls model name matching. If ``True``, ``model_name`` must
            exactly match the model's ``model_name`` attribute. If ``False``,
            matching is case-insensitive and allows ``model_name`` to be a
            substring of the model's ``model_name`` attribute.
        raise_if_empty : bool, default False
            If ``True``, raise a ``ValueError`` when no matching model is
            found. If ``False``, return an empty list.

        Returns
        -------
        list
            All models matching the requested filters. The returned list keeps
            the same order as ``model_list``.

        Examples
        --------
        >>> ModelManager.filter_models(model_list, model_name="Petrelli", exact=False)
        >>> ModelManager.filter_models(model_list, T_P="T", cpx_only=True)
        """
        model_list = list(model_list)
        matched_models = []

        for model in model_list:
            current_model_name = getattr(model, "model_name", None)

            if model_name is not None:
                if current_model_name is None:
                    continue

                current_model_name_str = str(current_model_name)
                model_name_str = str(model_name)

                if exact:
                    if current_model_name_str != model_name_str:
                        continue
                elif model_name_str.lower() not in current_model_name_str.lower():
                    continue

            if T_P is not None and getattr(model, "T_P", None) != T_P:
                continue

            if cpx_only is not None and getattr(model, "cpx_only", None) != cpx_only:
                continue

            matched_models.append(model)

        if raise_if_empty and not matched_models:
            available_names = sorted(
                {
                    str(getattr(model, "model_name", type(model).__name__))
                    for model in model_list
                }
            )
            raise ValueError(
                "No models matched the requested filters. "
                f"Available model names: {available_names}"
            )

        return matched_models

    @staticmethod
    def get_model(
        model_list,
        model_name=None,
        T_P=None,
        cpx_only=None,
        exact=True,
        raise_if_empty=False,
    ):
        """
        Return the first model matching filters from an existing model pool.

        Parameters
        ----------
        model_list : iterable
            Existing model pool to search.
        model_name : str, optional
            Target model name matched against each model's ``model_name``
            attribute. If ``None``, model names are not used as a filter.
        T_P : {"T", "P"}, optional
            Target thermobarometric parameter. If ``None``, both temperature
            and pressure models are allowed.
        cpx_only : bool, optional
            Phase-type filter. ``True`` returns only cpx-only models,
            ``False`` returns only cpx-liq models, and ``None`` does not
            restrict by phase type.
        exact : bool, default True
            If ``True``, require exact model name matching. If ``False``, use
            case-insensitive substring matching.
        raise_if_empty : bool, default False
            If ``True``, raise a ``ValueError`` when no matching model is
            found. If ``False``, return ``None``.

        Returns
        -------
        object or None
            The first matching model, or ``None`` if no model matches and
            ``raise_if_empty`` is ``False``.

        Examples
        --------
        >>> ModelManager.get_model(model_list, model_name="Petrelli", exact=False)
        >>> ModelManager.get_model(model_list, T_P="P", cpx_only=False)
        """
        matched_models = ModelManager.filter_models(
            model_list=model_list,
            model_name=model_name,
            T_P=T_P,
            cpx_only=cpx_only,
            exact=exact,
            raise_if_empty=raise_if_empty,
        )
        if not matched_models:
            return None
        return matched_models[0]

    def __str__(self):
        # print class name, T_P,
        # cpx_only, require_water, uncertainty
        text = f'''
        ModelManager: {self.__class__.__name__}
        Model name: {self.model_name}
        interested parameter: {self.T_P}; {"cpx-only" if self.cpx_only else "cpx-liq"}
        require_water: {self.require_water}
        uncertainty: {self.uncertainty} {"℃" if self.T_P == "T" else "kbar"}
        comment: {self.comments}
        '''
        return text

    def __repr__(self):
        return self.__str__()

# test code
if __name__ == "__main__":
    print("Testing ModelManager...")
    
    # model_manager = ModelManager()
    # print(model_manager)
