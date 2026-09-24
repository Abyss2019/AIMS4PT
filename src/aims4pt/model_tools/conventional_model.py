"""Shared lightweight wrapper for equation-based thermobarometry models.

This module keeps simple, equation-driven models consistent with the rest of
the package by standardizing input normalization and documenting expectations
for subclasses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aims4pt.model_tools.ModelManager import ModelManager
from aims4pt.utils import normalize_column_names


class conventional_model:
    """Minimal base class for closed-form thermobarometry equations."""

    def __init__(self, model_name: str, parameters_dict: dict, standard_columns: list[str]):
        """
        Initialize the model metadata.

        Parameters
        ----------
        model_name : str
            Human-readable model name.
        parameters_dict : dict
            Equation parameters (e.g., coefficients) supplied by the model
            author. The class does not alter these values.
        standard_columns : list[str]
            Expected feature names in standardized format (``*_cpx`` / ``*_liq``).
        """
        self.model_name = model_name
        self.parameters_dict = parameters_dict
        self.standard_columns = standard_columns
        self.cpx_names = [oxide for oxide in self.standard_columns if oxide.endswith('_cpx')]
        self.liq_names = ([oxide for oxide in self.standard_columns if oxide.endswith('_liq')] or None)

    def train(self, X: pd.DataFrame, y: pd.Series, train_method=None):
        """Placeholder for fitting equation parameters.

        The default implementation is intentionally empty because many
        conventional models rely on published coefficients rather than
        refitting. Subclasses can supply a custom ``train_method`` when
        re-calibration is required.
        """
        pass
        if train_method:
            self.parameters_dic = train_method(X, y)
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()

    def predict(self, X: pd.DataFrame):
        """Placeholder for predicting with equation parameters."""
        pass
        y = np.zeros(X.shape[0])
        return y

    def process_input(self, X: pd.DataFrame, X_liq: pd.DataFrame | None = None):
        """Normalize clinopyroxene/liquid inputs and append pressure/temperature.

        Parameters
        ----------
        X : pandas.DataFrame
            Required clinopyroxene features. ``P_kbar`` and ``T_C`` columns are
            optional; missing values are filled with zeros on the same index.
        X_liq : pandas.DataFrame, optional
            Liquid features when the equation requires them.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame | None]
            Normalized clinopyroxene inputs plus optional liquid inputs.
        """

        P_kbar = X.get("P_kbar", pd.Series(np.zeros(X.shape[0]), index=X.index))
        T_C = X.get("T_C", pd.Series(np.zeros(X.shape[0]), index=X.index))
        input_cpx = normalize_column_names(X, self.cpx_names)
        input_cpx.fillna(0, inplace=True)
        input_liq = None
        if X_liq is not None and (not self.cpx_only or self.require_water):
            input_liq = normalize_column_names(X_liq, self.liq_names)
            input_liq.fillna(0, inplace=True)
        elif X_liq is None and (self.cpx_only and self.require_water):
            input_liq = pd.DataFrame(
                np.zeros((X.shape[0], len(self.liq_names))), columns=self.liq_names, index=X.index
            )
        input_cpx["P_kbar"] = P_kbar
        input_cpx["T_C"] = T_C
        return input_cpx, input_liq

class iterative_model(ModelManager):
    '''
    This class is designed to handle iterative models that require multiple calculations to converge on a solution.

    Requirements: Wt: SiO2, Wt: TiO2, Wt: Al2O3, Wt: Cr2O3, Wt: FeO, Wt: MnO, Wt: MgO, Wt: CaO, Wt: Na2O, Wt: K2O

    This model requires iterative calculations for pressure (P) and temperature (T). 
    During initialization, the calculation method must be selected.
    To maintain consistency, T and P methods are calculated separately. 
    This is achieved by creating a `Putirka_08_T` object and a `Putirka_08_P` object, ensuring consistent parameters.
    A `switch_PT` method is added to allow switching between T and P outputs.
    '''
    cpx_only_models = {
        "T_models": {
            "hydrous": ["eq32d_T_hydrousVersion"],
            "anhydrous": ["eq32d_T"],
            "both": [],

        },
        "P_models": {
            "hydrous": [],
            "anhydrous": ["eq32a_P"],
            "both": []
        }
    }

    cpx_liq_models = {
        "T_models": {
            "hydrous": [],
            "anhydrous": [],
            "both": ["eq33_T"],
        },
        "P_models": {
            "hydrous": [],
            "anhydrous": [],
            "both": ["eq31_P"]
        }
    }

    iterative_model = True

    model_options = {
    }
    model_name_str = "XXX"
    
    def __init__(self, T_P, T_model_name, P_model_name, iteration_max=50, stop_criteria=1e-2, comments=None):
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
        comments = comments

        if not self.__class__.model_options:  
            model_options = {}
            model_name_str = "XXX"
        else:
            model_options = self.__class__.model_options  
            model_name_str = self.__class__.model_name_str
        

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
        self.iterative_model = True
        self.T_P = T_P
        self.iteration_max = iteration_max
        self.stop_criteria = stop_criteria
        self.T_model = model_dict[T_model_name]
        self.P_model = model_dict[P_model_name]

        self.model_name = f"{model_name_str} {T_model_name}; {P_model_name}"
        self.uncertainty = self.T_model.uncertainty if T_P == 'T' else self.P_model.uncertainty
        self.require_water = self.T_model.require_water or self.P_model.require_water

        self.if_support_hydrous = self.T_model.if_support_hydrous and self.P_model.if_support_hydrous
        self.cpx_only = self.T_model.cpx_only and self.P_model.cpx_only

        self.cpx_names = [oxide for oxide in self.standard_columns if oxide.endswith('_cpx')]
        self.liq_names = [oxide for oxide in self.standard_columns if oxide.endswith('_liq')]

        self.X_cpx_training = self.T_model.X_cpx_train_pkl_path if T_P == 'T' else self.P_model.X_cpx_train_pkl_path
        self.X_liq_training = self.T_model.X_liq_train_pkl_path if T_P == 'T' else self.P_model.X_liq_train_pkl_path

        self.if_normalize_liq = True if not self.cpx_only else False

        self.y_min = getattr(self.T_model, 'y_min', None) if T_P == 'T' else getattr(self.P_model, 'y_min', None)
        self.y_max = getattr(self.T_model, 'y_max', None) if T_P == 'T' else getattr(self.P_model, 'y_max', None)

        self.initialize_model(
            cpx_training_path=self.X_cpx_training,
            liq_training_path=self.X_liq_training,
        )
        self.rock_types = None # Not available yet.



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


            

    def predict(self, X, X_liq=None):
        """
        Iterate the coupled T–P model until the stop criterion is met.

        Parameters
        ----------
        X : pd.DataFrame
            Clinopyroxene input features.
        X_liq : pd.DataFrame, optional
            Liquid input features (if applicable).

        Returns
        -------
        pd.Series
            Predicted temperature (T_C) or pressure (P_kbar), depending on self.T_P.
        """
        # Make defensive copies to avoid mutating external inputs
        input_cpx = X.copy()
        input_liq = X_liq.copy() if X_liq is not None else None

        # If the model is cpx-only but requires water, build a fake liquid with 0 H2O (and other oxides)
        if input_liq is None and (self.cpx_only and self.require_water):
            input_liq = pd.DataFrame(
                np.zeros((X.shape[0], len(self.liq_names))),
                columns=self.liq_names,
                index=X.index,  # keep index aligned with cpx
            )

        # Initialize iteration control
        iteration = 0
        stop_criteria_value = self.stop_criteria + 1.0  # force at least one iteration

        # Initialize T and P fields for the first iteration
        input_cpx["T_C"] = 1000.0
        input_cpx["P_kbar"] = 5.0

        while iteration < self.iteration_max and stop_criteria_value > self.stop_criteria:
            iteration += 1

            # Predict temperature and pressure for current state
            T = self.T_model.predict(input_cpx, input_liq)
            P = self.P_model.predict(input_cpx, input_liq)

            # Ensure T and P are pandas Series aligned with input_cpx
            if not isinstance(T, pd.Series):
                T = pd.Series(T, index=input_cpx.index, name="T_C")
            if not isinstance(P, pd.Series):
                P = pd.Series(P, index=input_cpx.index, name="P_kbar")

            # Compute relative changes for convergence check
            # Small epsilon to avoid division by zero in pathological cases
            eps_T = 1e-9
            eps_P = 1e-9

            rel_change_T = np.abs(input_cpx["T_C"] - T) / (np.abs(T) + eps_T)
            rel_change_P = np.abs(input_cpx["P_kbar"] - P) / (np.abs(P) + eps_P)

            stop_criteria_T = rel_change_T.max()
            stop_criteria_P = rel_change_P.max()
            stop_criteria_value = max(stop_criteria_T, stop_criteria_P)

            # Update the input state with the new T and P
            input_cpx["T_C"] = T
            input_cpx["P_kbar"] = P

        # Post-iteration: handle non-converged samples
        if iteration >= self.iteration_max and stop_criteria_value > self.stop_criteria:
            print(
                f"Warning: Maximum iterations ({self.iteration_max}) reached "
                f"without full convergence (max relative change={stop_criteria_value:.3e})."
            )

            # Recompute final relative changes for each sample
            eps_T = 1e-9
            eps_P = 1e-9
            rel_change_T = np.abs(input_cpx["T_C"] - T) / (np.abs(T) + eps_T)
            rel_change_P = np.abs(input_cpx["P_kbar"] - P) / (np.abs(P) + eps_P)

            # Element-wise OR for "non-converged" mask
            not_converged = (rel_change_T > self.stop_criteria) | (
                rel_change_P > self.stop_criteria
            )

            # Assign NaN to non-converged samples
            T = T.copy()
            P = P.copy()
            T[not_converged] = np.nan
            P[not_converged] = np.nan

        # Return either T or P depending on configuration
        if self.T_P == "T":
            return pd.Series(T, name="T_C")
        else:
            return pd.Series(P, name="P_kbar")
