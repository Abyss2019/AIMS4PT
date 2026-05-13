"""Helper utilities for model explainability and background reduction.

This module centralizes SHAP explanations and background sampling helpers so
model wrappers can reuse a consistent, well-documented interface.
"""

from typing import Callable, Dict, Optional, Tuple

import pandas as pd


def shap_calculation(
    model_callable: Callable[[pd.DataFrame], pd.DataFrame],
    test_X: pd.DataFrame,
    y_column: Optional[str] = None,
    background_data: Optional[pd.DataFrame] = None,
    exclude_nan_results: bool = True,
    per_sample_background: bool = False,
    n_neighbors: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate SHAP values while guarding against empty inputs.

    The function standardizes how SHAP explanations are generated across
    models. It supports per-sample background selection (using nearest
    neighbors) with a lightweight cache so repeated neighbor sets reuse the
    same explainer instead of rebuilding it for every row.

    Parameters
    ----------
    model_callable : callable
        Prediction callable accepting a :class:`pandas.DataFrame` and returning
        a Series/DataFrame.
    test_X : pandas.DataFrame
        Input features to explain.
    y_column : str, optional
        Target column to extract from model outputs.
    background_data : pandas.DataFrame, optional
        Background data used by SHAP; defaults to ``test_X``.
    exclude_nan_results : bool, default True
        Exclude rows where model outputs are NaN or zero.
    per_sample_background : bool, default False
        If True, select nearest background samples per explained sample.
    n_neighbors : int, default 50
        Number of neighbors to use for per-sample background selection.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        SHAP values and feature-importance summary.
    """

    import shap
    from sklearn.neighbors import NearestNeighbors

    shap.initjs()

    test_X = test_X.copy()
    background_data = background_data.copy() if background_data is not None else test_X.copy()

    test_X.fillna(0, inplace=True)
    background_data.fillna(0, inplace=True)

    y_pred = model_callable(test_X)
    if y_column is not None:
        y_pred = y_pred[y_column]
    y_background = model_callable(background_data)
    if y_column is not None:
        y_background = y_background[y_column]

    if exclude_nan_results:
        y_pred_series = pd.Series(y_pred, index=test_X.index)
        y_background_series = pd.Series(y_background, index=background_data.index)

        valid_test_mask = ~(y_pred_series.isna() | (y_pred_series == 0))
        valid_background_mask = ~(y_background_series.isna() | (y_background_series == 0))

        test_X = test_X.loc[valid_test_mask]
        background_data = background_data.loc[valid_background_mask]

    if test_X.empty or background_data.empty:
        return pd.DataFrame(), pd.DataFrame(columns=['Importance', 'Feature'])

    # Ensure n_neighbors does not exceed available background samples
    n_neighbors = min(n_neighbors, len(background_data))

    def _build_explainer(background: pd.DataFrame) -> shap.Explainer:
        masker = shap.maskers.Independent(background)
        return shap.Explainer(lambda X: model_callable(X).values, masker=masker)

    if per_sample_background:
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(background_data)
        neighbor_matrix = knn.kneighbors(test_X, return_distance=False)
        explainer_cache: Dict[tuple[int, ...], shap.Explainer] = {}
        shap_values_matrix = []
        for row_idx, neighbor_indices in enumerate(neighbor_matrix):
            key = tuple(neighbor_indices.tolist())
            explainer = explainer_cache.get(key)
            if explainer is None:
                explainer = _build_explainer(background_data.iloc[neighbor_indices])
                explainer_cache[key] = explainer
            shap_values = explainer(test_X.iloc[[row_idx]])
            shap_values_matrix.append(shap_values.values[0])
        shap_values_array = shap_values_matrix
    else:
        explainer = _build_explainer(background_data)
        shap_values = explainer(test_X)
        shap_values_array = shap_values.values

    shap_df = pd.DataFrame(
        shap_values_array,
        columns=test_X.columns,
        index=test_X.index[: len(shap_values_array)],
    )

    feature_order = shap_df.abs().mean().sort_values(ascending=False).index.tolist()
    shap_df = shap_df[feature_order]
    ordered_test_X = test_X[feature_order]

    if not shap_df.empty:
        shap.summary_plot(shap_df.values, features=ordered_test_X)

    feature_importance = shap_df.abs().mean().sort_values(ascending=False)
    feature_importance_percent = feature_importance / feature_importance.sum()
    feature_importance_df = pd.DataFrame(
        {'Importance': feature_importance_percent, 'Feature': feature_importance.index}
    )

    return shap_df, feature_importance_df


def kmeans_reduce_data(data: pd.DataFrame, n_clusters: int, random_state: int = 42) -> pd.DataFrame:
    '''Reduce background data using KMeans clustering from scikit-learn.
    Parameters:
        data (pd.DataFrame): The original background data.
        n_clusters (int): The desired number of cluster centers.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
    Returns:
        pd.DataFrame: A new background dataset composed of the cluster centers.
    Raises:
        Warning: If the number of clusters exceeds the size of the data, all data will be used.
    '''

    import warnings
    from sklearn.cluster import KMeans
    if n_clusters > len(data):
        warnings.warn(
            f"n_clusters ({n_clusters}) exceeds data size ({len(data)}); using all data.")
        n_clusters = len(data)

    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state, n_init='auto')
    kmeans.fit(data)

    return pd.DataFrame(kmeans.cluster_centers_, columns=data.columns).reset_index(drop=True)


def random_sample_reduce_data(data: pd.DataFrame, n_samples: int, random_state: int = 42) -> pd.DataFrame:
    '''Reduce background data using random sampling.
    Parameters:
        data (pd.DataFrame): The original background data.
        n_samples (int): The desired number of samples.
    Returns:
        pd.DataFrame: A new background dataset composed of the sampled data.
    Raises:
        Warning: If the number of samples exceeds the size of the data, all data will be used.
    '''
    import warnings
    if n_samples > len(data):
        warnings.warn(
            f"n_samples ({n_samples}) exceeds data size ({len(data)}); using all data.")
        n_samples = len(data)

    return data.sample(n=n_samples, random_state=random_state).reset_index(drop=True)



