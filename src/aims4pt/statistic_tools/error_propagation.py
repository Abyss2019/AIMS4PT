"""Analytical error propagation helpers with optional visualization."""

from typing import Dict, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AbsErrorType = Union[Dict[str, float], List[float]]


def _abs_error_to_dict(abs_error: AbsErrorType, columns: Iterable[str]) -> Dict[str, float]:
    """Normalize absolute error inputs to a dictionary keyed by column name."""

    column_list = list(columns)
    if isinstance(abs_error, dict):
        missing = [col for col in column_list if col not in abs_error]
        if missing:
            missing_cols = ", ".join(missing)
            raise ValueError(f"Missing absolute error values for: {missing_cols}")
        return {col: float(abs_error[col]) for col in column_list}

    if isinstance(abs_error, list):
        if len(abs_error) != len(column_list):
            raise ValueError("Length of abs_error list must match number of columns.")
        return dict(zip(column_list, abs_error))

    raise ValueError("abs_error should be a dictionary or a list.")


class ErrorDF:
    """Container for generating random error samples."""

    def __init__(
        self, abs_error_dict: Dict[str, float], n: int, random_state: int | None = 0
    ) -> None:
        rng = np.random.RandomState(random_state) if random_state is not None else np.random
        self.error_df = pd.DataFrame(
            {col: rng.normal(0, error, n) for col, error in abs_error_dict.items()}
        )

    def get_error_df(self) -> pd.DataFrame:
        return self.error_df


def error_propagation(
    df: pd.DataFrame,
    abs_error: AbsErrorType,
    model,
    n: int = 1000,
    plot: bool = False,
    random_state: int | None = 0,
) -> Tuple[float, float]:
    """
    Calculate the mean and standard deviation of the propagated error distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing model features.
    abs_error : dict or list
        Absolute error per column, either as a mapping or a list aligned with ``df``.
    model : object
        Model object exposing a ``predict`` method that accepts a DataFrame.
    n : int, default 1000
        Number of simulated rows used to build the error distribution.
    plot : bool, default False
        Whether to plot the prediction distribution.
    random_state : int or None, default 0
        Seed for reproducible error sampling. Set to ``None`` to draw from the
        global NumPy generator without altering its state.

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of the propagated predictions.
    """

    abs_error_dict = _abs_error_to_dict(abs_error, df.columns)

    # create a DataFrame with random errors
    error_df = ErrorDF(abs_error_dict, n, random_state=random_state).get_error_df()

    # calculate the result of the model with the errors
    perturbed_df = pd.concat([df] * n, ignore_index=True).add(error_df, fill_value=0)

    result = np.asarray(model.predict(perturbed_df)).ravel()
    real_results = np.asarray(model.predict(df)).ravel()

    # calculate the mean and standard deviation of the error distribution
    mean_error = float(np.mean(result))
    std_error = float(np.std(result, ddof=0))

    if plot:
        plt.figure(figsize=(8, 6), dpi=200)
        plt.hist(result, bins=20, color='C0', edgecolor='white')
        y_min, y_max = plt.ylim()
        if real_results.size:
            plt.vlines(
                real_results[0],
                y_min,
                y_max,
                color='k',
                label='Prediction (Original Data Input)',
                linestyle='--',
            )
        plt.xlabel('Predictions')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Distribution of the Predictions')
        plt.show()

    return mean_error, std_error


def error_propagation_AAT(
    df: pd.DataFrame,
    abs_error: AbsErrorType,
    model,
    n: int = 1000,
    plot: bool = False,
    random_state: int | None = 0,
) -> Tuple[float, float]:
    """All-at-a-time error propagation wrapper.

    Parameters mirror :func:`error_propagation`, allowing a shared ``random_state``
    for reproducible sampling.
    """

    mean, sd = error_propagation(df, abs_error, model, n, plot, random_state)
    return mean, sd


def error_propagation_OAT(
    df: pd.DataFrame,
    abs_error: AbsErrorType,
    model,
    n: int = 1000,
    plot: bool = False,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """One-at-a-time error propagation across each feature column.

    Parameters include ``random_state`` for reproducibility and ``plot`` for an
    optional scatter summary of per-feature standard deviations.
    """

    abs_error_dict = _abs_error_to_dict(abs_error, df.columns)

    results = {}
    for column in df.columns:
        abs_error_dict_column = {key: 0 for key in abs_error_dict}
        abs_error_dict_column[column] = abs_error_dict[column]

        mean_error, std_error = error_propagation(
            df, abs_error_dict_column, model, n, plot=False, random_state=random_state
        )
        results[column] = {'mean_error': mean_error, 'std_error': std_error}

    df_results = pd.DataFrame(results).T

    if plot:
        plt.figure(figsize=(8, 6), dpi=200)
        plt.scatter(df_results.index, df_results['std_error'], color='C0', label='σ')
        plt.xlabel('Components')
        plt.ylabel('σ')
        plt.legend()
        plt.show()

    return df_results


def analyze_error_influence(
    df: pd.DataFrame,
    base_abs_error: AbsErrorType,
    model,
    n: int = 1000,
    scale_factors: float = 0.5,
    plot: bool = False,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Analyze how scaling individual analytical errors affects propagation.

    ``random_state`` controls the sampling used when generating synthetic error
    realizations.
    """

    base_abs_error_dict = _abs_error_to_dict(base_abs_error, df.columns)
    keys = list(base_abs_error_dict.keys())

    results = {}
    for key in keys:
        abs_error_dict = base_abs_error_dict.copy()
        abs_error_dict[key] = abs_error_dict[key] * scale_factors

        mean_error, std_error = error_propagation(
            df, abs_error_dict, model, n, plot=False, random_state=random_state
        )
        results[key] = {'mean_error': mean_error, 'std_error': std_error}

    df_results = pd.DataFrame(results).T
    _, ref_std = error_propagation(
        df, base_abs_error_dict, model, n, plot=False, random_state=random_state
    )

    if plot:
        plt.figure(figsize=(8, 6), dpi=200)
        plt.axhline(ref_std, color='k', label='σ (Original Analysis Error)', linestyle='--')
        plt.scatter(
            df_results.index,
            df_results['std_error'],
            color='C0',
            label=f'σ (Scaled Analysis Error by {scale_factors})',
        )
        plt.xlabel('Components')
        plt.xticks(keys)
        plt.ylabel('σ')
        plt.legend()
        plt.show()

    return df_results


def analyze_error_influence_inverse(
    df: pd.DataFrame,
    base_abs_error: AbsErrorType,
    model,
    n: int = 1000,
    scale_factors: float = 0.5,
    plot: bool = False,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Analyze the influence when scaling all errors except one component.

    ``random_state`` controls the sampling used when generating synthetic error
    realizations.
    """

    base_abs_error_dict = _abs_error_to_dict(base_abs_error, df.columns)
    keys = list(base_abs_error_dict.keys())

    results = {}
    for key in keys:
        abs_error_dict = base_abs_error_dict.copy()
        for other_key in keys:
            if other_key != key:
                abs_error_dict[other_key] = abs_error_dict[other_key] * scale_factors

        mean_error, std_error = error_propagation(
            df, abs_error_dict, model, n, plot=False, random_state=random_state
        )
        results[key] = {'mean_error': mean_error, 'std_error': std_error}

    df_results = pd.DataFrame(results).T

    if plot:
        plt.figure(figsize=(8, 6), dpi=200)
        plt.scatter(
            df_results.index,
            df_results['std_error'],
            color='C0',
            label=f'σ (other Errors Scaled by {scale_factors})',
        )
        plt.xlabel('Components')
        plt.xticks(rotation=45)
        plt.ylabel('Standard Deviation (σ)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df_results
