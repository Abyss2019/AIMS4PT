# -*- coding: utf-8 -*-
'''
This module provides functions to fit Gaussian Mixture Models (GMM) to positive-only data columns,
automatically determine the optimal number of components using BIC, and compute confidence intervals for each component.

Functions:
- range_check: Check if the range of experimental data overlaps with the range of natural samples.
- rock_type_check: Check if all rock types in the sample list are covered by the experiment list.
- fit_positive_gmm_extract_ranges: Fit GMM to positive-only data columns and compute confidence intervals.
- check_unimodal_coverage: Check whether a unimodal dataset falls inside the model range.
- fit_kde: Fit a multivariate Kernel Density Estimator (KDE) with cross-validation based bandwidth selection.
- compute_density_threshold: Compute the lower density threshold corresponding to a specified tail fraction.
- is_in_high_density_region: Classify new samples based on their density relative to the threshold.
- check_density_region: Convenience wrapper to flag low-density samples via KDE.
- visualize_density_region: Visualize the density region using contours and scatter plots.

'''


import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, List, Dict
from typing import Sequence, Optional, Tuple
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import numpy as np
from aims4pt.utils import normalize_column_names
from typing import Tuple
import pandas as pd


def range_check(X_samples_or_lim_tuple, X_experiments, quantile_samples=0.9973, quantile_experiments=0.9973):
    """
    Check if the range of experimental data overlaps with the range of natural samples.

    Parameters
    ----------
    X_samples_or_lim_tuple : pd.Series or tuple
        Values from natural samples or a tuple specifying (lower, upper) limits.
        np.nan means no limit.
    X_experiments : pd.Series
        Values from experimental data.
    quantile_samples : float, optional
        Quantile for filtering outliers in natural samples (default: 0.9973, ~3σ).
    quantile_experiments : float, optional
        Quantile for filtering outliers in experimental data (default: 0.9973, ~3σ).

    Returns
    -------
    bool
        True if ranges overlap, False otherwise.
    """
    if isinstance(X_samples_or_lim_tuple, tuple):
        samples_lower, samples_upper = X_samples_or_lim_tuple
    else:
        samples = X_samples_or_lim_tuple[X_samples_or_lim_tuple.notna()]
        samples_lower = samples.quantile((1 - quantile_samples) / 2)
        samples_upper = samples.quantile(1 - (1 - quantile_samples) / 2)

    experiments = X_experiments[X_experiments.notna()]
    experiments_lower = experiments.quantile((1 - quantile_experiments) / 2)
    experiments_upper = experiments.quantile(
        1 - (1 - quantile_experiments) / 2)

    if pd.isna(samples_lower) or samples_upper <= experiments_upper:
        if pd.isna(samples_upper) or samples_lower >= experiments_lower :
            return True
    return False


def rock_type_check(sample_list_of_rock_types, experiment_list_of_rock_types, report=False):
    """
    Check if all rock types in the sample list are covered by the experiment list by TAS classification.

    Parameters
    ----------
    sample_list_of_rock_types : list or str
        List of rock types from natural samples.
    experiment_list_of_rock_types : list
        List of rock types from experimental data.

    Returns
    -------
    bool
        True if all sample rock types are covered by the experiment rock types, False otherwise.
    list (optional)
        List of missing rock types if `report` is True. If no missing types, returns an empty list.
    """
    # 1. Empty experiment list is treated as unconstrained, so it always passes.
    if not experiment_list_of_rock_types:
        if report:
            return True, []
        return True

    # 2. Normalize the input to a list.
    if isinstance(sample_list_of_rock_types, str):
        sample_list_of_rock_types = [sample_list_of_rock_types]

    # 3. Check the subset relationship.
    result = set(sample_list_of_rock_types).issubset(set(experiment_list_of_rock_types))

    if report:
        if result:
            return True, []
        else:
            missing_rock_types = set(sample_list_of_rock_types) - set(experiment_list_of_rock_types)
            return False, list(missing_rock_types)

    return result




def fit_gmm_extract_ranges(data, features, confidence_level=0.954, random_state=0, n_comp=None, st_col_name=None, axes=None, suffix=""):
    """
    Fit Gaussian Mixture Models (GMM) to positive-only data columns (oxides),
    automatically determine the optimal number of components using BIC,
    and compute confidence intervals for each component (μ ± Z·σ, lower bound truncated at 0).

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataset containing oxide values.

    features : list of str
        The list of features column names to analyze.

    confidence_level : float, optional
        Confidence level used to compute confidence intervals (default: 0.954 for ~2σ).
        Could be set to 0.6827 (1σ), 0.954 (2σ), or 0.9973 (3σ), or any other value.

    random_state : int, optional
        Random seed for reproducibility.

    st_col_name : list of str, optional
        Standard names for the columns. If None, will use the column names from the data.

    axes : list of matplotlib.axes.Axes, optional
        Pre-existing axes to plot on. If None, new axes will be created.

    suffix : str, optional
        Suffix to append to the plot labels for clarity.


    Returns
    -------
    axes : list of matplotlib.axes.Axes
        List of axes for the generated plots (one per oxide).

    result_df : pandas.DataFrame
        A DataFrame containing model parameters and confidence intervals per component.

        Example:
            +--------+-----------+--------------+----------+---------+---------+------------+-------------+-------------------+
            | oxide  | component | n_components |   mean   |  sigma  | weight  | range_low  | range_high  | confidence_level  |
            +--------+-----------+--------------+----------+---------+---------+------------+-------------+-------------------+
            | SiO2   |     1     |      2       | 69.64576 | 2.95871 | 0.49482 | 63.74197   | 75.54954    |       0.954       |
            | SiO2   |     2     |      2       | 50.28032 | 5.17188 | 0.50518 | 39.96039   | 60.60025    |       0.954       |
            | Fe2O3  |     1     |      2       | 20.17479 | 1.39989 | 0.50028 | 17.38147   | 22.96811    |       0.954       |
            | Fe2O3  |     2     |      2       |  9.80625 | 1.91756 | 0.49972 |  5.97997   | 13.63254    |       0.954       |
            +--------+-----------+--------------+----------+---------+---------+------------+-------------+-------------------+
    """

    import os

    os.environ["OMP_NUM_THREADS"] = "3"
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm

    # normalize data
    from aims4pt.utils import normalize_column_names
    if isinstance(features, str):
        features = [features]
    if isinstance(st_col_name, str):
        st_col_name = [st_col_name]

    if st_col_name is None:
        data = normalize_column_names(df=data)
    else:
        data = data[st_col_name].copy()
    # Compute Z-score corresponding to the given confidence level
    z = norm.ppf(0.5 + confidence_level / 2)

    # Result container: one row per component per oxide
    results = []

    # Prepare subplot layout based on number of oxides
    num_oxides = len(features)
    from aims4pt.visualization.plot_utils import get_subplot_shape
    rows, cols = get_subplot_shape(num_oxides)

    if axes is None:
        color_shade = "lightgrey"
        color_line = "black"

        fig, axes = plt.subplots(
            rows, cols, figsize=(5 * cols, 4 * rows), dpi=200)
    if not isinstance(axes, np.ndarray):
        color_shade = "lightgrey"
        color_line = "black"
        axes = np.array([axes])

    # Loop through each oxide column
    for idx, feature in enumerate(features):
        # Extract data and filter out non-positive values
        x = np.array(data[feature])
        x = x[np.isfinite(x)]

        # Skip if not enough valid data
        if len(x) < 10:
            print(f"{feature} has too few data points, skipping.")
            continue

        # Automatically determine best number of components using BIC if num_comp is None
        if n_comp is None:
            lowest_bic = np.inf
            best_gmm = None
            best_n_comp = 0
            for n_comp in range(1, 7):
                gmm = GaussianMixture(n_components=n_comp,
                                      random_state=random_state)
                gmm.fit(x.reshape(-1, 1))
                bic = gmm.bic(x.reshape(-1, 1))
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
                    best_n_comp = n_comp
        else:
            # Use provided number of components
            best_n_comp = n_comp
            best_gmm = GaussianMixture(n_components=best_n_comp,
                                       random_state=random_state)
            best_gmm.fit(x.reshape(-1, 1))

        # Plot histogram and component PDFs
        ax = axes[idx]

        ax.hist(x, bins=40, density=True, alpha=0.8,
                label="Data" + " " + suffix, color=color_shade)
        x_plot = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
        total_pdf = np.zeros_like(x_plot)

        # sort the components by mean value
        sorted_indices = np.argsort(best_gmm.means_[:, 0])
        best_gmm.means_ = best_gmm.means_[sorted_indices]
        best_gmm.covariances_ = best_gmm.covariances_[sorted_indices]
        best_gmm.weights_ = best_gmm.weights_[sorted_indices]

        # Loop through each component and plot its contribution
        for comp in range(best_gmm.n_components):
            mu = best_gmm.means_[comp, 0]
            sigma = np.sqrt(best_gmm.covariances_[comp, 0, 0])
            weight = best_gmm.weights_[comp]

            # Confidence interval, truncated at zero
            range_low = mu - z * sigma
            range_high = mu + z * sigma

            # Store result
            results.append({
                "oxide": feature,
                "component": comp + 1,
                "n_components": best_n_comp,
                "mean": mu,
                "sigma": sigma,
                "weight": weight,
                "range_low": range_low,
                "range_high": range_high,
                "confidence_level": confidence_level
            })

            # Plot individual component
            pdf = weight * norm.pdf(x_plot, mu, sigma)
            total_pdf += pdf
            ax.plot(x_plot, pdf,
                    label=f"Comp {comp+1}: μ={mu:.2f}, σ={sigma:.2f}")
            ax.axvline(mu, linestyle="--", color="gray", alpha=0.7)
            # ax.axvline(range_high, linestyle="--", color="gray", alpha=0.7)

        # Plot total GMM PDF
        ax.plot(x_plot, total_pdf, '--', label="Total GMM" +
                " " + suffix, color=color_line)
        ax.set_title(f"{feature} ({best_n_comp} peaks)")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig = axes[0].figure
    fig.suptitle(f"Confidence Level: {confidence_level:.3f} ({z:.1f}σ)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    # Convert results list to DataFrame
    result_df = pd.DataFrame(results)

    return axes, result_df


def check_unimodal_coverage(
    query_df: pd.DataFrame,
    range_df: pd.DataFrame,
    oxides: list[str] | None = None,
    coverage_threshold: float = 0.9
) -> Tuple[bool, pd.DataFrame, pd.DataFrame]:
    """
    Check whether a *unimodal* dataset falls inside the model range.

    Parameters
    ----------
    query_df : pd.DataFrame
        Samples to be validated.
    range_df : pd.DataFrame
        Model range table with **one component per oxide**.
        Must contain columns: ["oxide", "range_low", "range_high"].

        +--------+-----------+--------------+----------+---------+---------+------------+-------------+-------------------+
        | oxide  | component | n_components |   mean   |  sigma  | weight  | range_low  | range_high  | confidence_level  |
        +--------+-----------+--------------+----------+---------+---------+------------+-------------+-------------------+
        | SiO2   |     1     |      2       | 69.64576 | 2.95871 | 0.49482 | 63.74197   | 75.54954    |       0.954       |
        | SiO2   |     2     |      2       | 50.28032 | 5.17188 | 0.50518 | 39.96039   | 60.60025    |       0.954       |
        | Fe2O3  |     1     |      2       | 20.17479 | 1.39989 | 0.50028 | 17.38147   | 22.96811    |       0.954       |
        | Fe2O3  |     2     |      2       |  9.80625 | 1.91756 | 0.49972 |  5.97997   | 13.63254    |       0.954       |
        +--------+-----------+--------------+----------+---------+---------+------------+-------------+-------------------+

    oxides : list[str] | None, default None
        Oxides to validate; if None, use all oxides in ``range_df``.
    coverage_threshold : float, default 0.9
        Fraction of rows that must satisfy *all* oxide ranges
        for the whole dataset to be considered “passed”.

    Returns
    -------
    passed : bool
        True  if coverage ≥ ``coverage_threshold``.
    passed_df : pd.DataFrame
        Rows that pass every‐oxide check.
    failed_df : pd.DataFrame
        Rows that fail at least one oxide check (deduplicated).

    Notes
    -----
    • Only valid for **unimodal** distributions (n_components == 1).  
    • ``range_df`` with multiple components should be pre‑aggregated
      to a single [low, high] interval beforehand.
    """
    # --- 1. normalize_column_names ------------------------------
    query_df = normalize_column_names(query_df.copy())

    # --- 2. oxides --------------------
    avail_oxides = set(range_df["oxide"].unique())
    oxides = avail_oxides if oxides is None else set(oxides)
    if not oxides:
        raise ValueError("No valid oxides found in range_df.")

    # --- 3. mask----------
    mask_all = pd.Series(True, index=query_df.index)

    for ox in oxides:
        range_ox_df = range_df[range_df["oxide"] == ox]
        pass_ox = pd.Series(False, index=range_ox_df.index)
        if range_df.empty:
            raise ValueError(f"Oxide '{ox}' not found in range_df.")
        for component in range_ox_df["component"].unique():
            range_comp_df = range_ox_df[range_ox_df["component"] == component]
            low, high = range_comp_df[["range_low", "range_high"]].values[0]
            pass_ox = query_df[ox].between(low, high) | pass_ox
        mask_all = mask_all & pass_ox

    # --- 4. split ----------------------
    passed_df = query_df[mask_all]
    failed_df = query_df[~mask_all]

    # --- 5. check pass ----------------------
    # == passed_df.shape[0] / len(query_df)
    coverage = mask_all.mean()
    passed = coverage >= coverage_threshold

    return passed, passed_df, failed_df


# ----------------
# Usage example.
if __name__ == "__main__" and False:

    # Simulated example: construct two oxides as mixtures of normal distributions with positive values only.
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    np.random.seed(0)
    size = 300
    data_dict = {
        "SiO2": np.concatenate([
            np.random.normal(50, 5, size),
            np.random.normal(70, 3, size)
        ]),
        "Fe2O3": np.concatenate([
            np.random.normal(10, 2, size),
            np.random.normal(20, 1.5, size)
        ])
    }
    data_dict2 = {
        "SiO2": np.concatenate([
            np.random.normal(24, 5, size),
            np.random.normal(43, 3, size)
        ]),
        "Fe2O3": np.concatenate([
            np.random.normal(98, 2, size),
            np.random.normal(44, 1.5, size)
        ])

    }

    df = pd.DataFrame(data_dict)
    df2 = pd.DataFrame(data_dict2)
    # Truncate negative values, if any.
    for col in df.columns:
        df.loc[df[col] < 0, col] = np.nan

    # Call the method with target oxide columns, a 0.95 confidence level, and random seed 0.
    axes, results_df = fit_positive_gmm_extract_ranges(
        df, features=["SiO2", "Fe2O3"], confidence_level=0.954, random_state=0, suffix="_1")

    axes2, results_df2 = fit_positive_gmm_extract_ranges(
        df2, features=["SiO2", "Fe2O3"], confidence_level=0.954, random_state=0, axes=axes, suffix="_2")
    # plt.show()
    # Check whether the data fall within the model range.
    passed, passed_df, failed_df = check_unimodal_coverage(
        df, results_df, coverage_threshold=0.9)
    print(f"Passed: {passed}")
    print(f"failed_df rows:\n{failed_df}")


# ----------------------------------------------------------------------
# 1. KDE fitting
# ----------------------------------------------------------------------

def fit_kde(
    data: pd.DataFrame,
    bandwidths: Optional[Sequence[float]] = None,
    cv: int = 5,
    kernel: str = "gaussian",

) -> KernelDensity:
    """Fit a multivariate KDE with CV‑based bandwidth selection.

    Parameters
    ----------
    data : pd.DataFrame, shape (n_samples, n_features)
        Reference data used to estimate the joint PDF.
    bandwidths : Sequence[float] | None, optional
        Candidate bandwidths. If *None*, a log‑spaced grid between 0.1 and 10
        is used.
    cv : int, default=5
        Number of folds for cross‑validation.
    kernel : str, default="gaussian"
        Kernel type supported by :class:`sklearn.neighbors.KernelDensity`.

    Returns
    -------
    KernelDensity
        The best KDE estimator fitted on *data*.
    """
    if bandwidths is None:
        bandwidths = np.logspace(-2, 1, num=100)

    def ll_with_penalty(est, X_val, lam=0.3):
        return est.score(X_val)
    
    grid = GridSearchCV(
        KernelDensity(kernel=kernel),
        {"bandwidth": bandwidths},
        cv=cv,
        n_jobs=-1,
        scoring=lambda est, X: ll_with_penalty(est, X, lam=0.3),
    )

    grid.fit(data.to_numpy())
    # plt.figure(figsize=(8, 4))
    # plt.plot(bandwidths, grid.cv_results_["mean_test_score"], marker="o")
    # plt.xlabel("Bandwidth")
    # plt.ylabel("Log-likelihood")
    # plt.show()
    best_bw = grid.best_params_["bandwidth"]
    print(f"Best bandwidth: {best_bw:.3f}")
    print(f"Best CV log-likelihood: {grid.best_score_:.3f} ()")
    
    
    return grid.best_estimator_


# ----------------------------------------------------------------------
# 2. Threshold on reference density
# ----------------------------------------------------------------------
def compute_density_threshold(
    kde: KernelDensity,
    data: pd.DataFrame,
    tail_frac: float = 0.00135,
) -> float:
    """Return the lower density threshold corresponding to *tail_frac*."""
    log_density = kde.score_samples(data.to_numpy())
    density = np.exp(log_density)
    threshold = np.percentile(density, 100 * tail_frac)
    # print(
    #     f"Density threshold ({tail_frac*100:.3f}% quantile): {threshold:.5e}")
    return threshold


# ----------------------------------------------------------------------
# 3. Classify new samples
# ----------------------------------------------------------------------
def is_in_high_density_region(
    data_query: pd.DataFrame,
    kde: KernelDensity,
    density_thresh: float,
) -> Tuple[pd.Series, pd.Series]:
    """Flag whether each query sample lies inside the high‑density region."""
    log_density = kde.score_samples(data_query.to_numpy())
    density = np.exp(log_density)
    mask = density >= density_thresh
    return pd.Series(mask, index=data_query.index), pd.Series(density, index=data_query.index)




def check_density_region(
    data_reference: pd.DataFrame,
    data_query: pd.DataFrame,
    features: Optional[List[str]] = None,
    confidence_level: float = 0.9973,
    cv: int = 5,
    feature_weights: Optional[Dict[str, float]] = None,
    full_return: bool = False,
    if_show_plot: bool = True,
    verbose: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    Convenience wrapper to flag low‑density samples via KDE, with optional per-feature weights.

    Parameters
    ----------
    data_reference : pd.DataFrame, shape (n_samples, n_features)
        Reference data used to estimate the joint PDF.
        Must be normalized first.
    data_query : pd.DataFrame, shape (n_samples, n_features)
        Samples to be classified.
        Must be normalized first.
    features : list[str] | None
        Features to validate; if None, use all columns in data_reference.
    confidence_level : float
        Fraction of samples that should be classified as inliers (default 0.9973).
    cv : int
        Number of folds for KDE bandwidth cross‑validation.
    feature_weights : dict[str, float] | None
        Per‑feature positive weights (higher → feature more important).
        If None, all weights = 1.
    full_return : bool, default False
        If True, returns additional information including the plot and KDE parameters.
    if_show_plot : bool, default True
        If True, displays the density region plot.
    verbose : bool, default False
        If True, prints detailed information about the process.
    

    Returns
    -------
    mask : pd.Series of bool, shape (n_query,)
        True for samples in the high‑density region.
    density_values : pd.Series of float, shape (n_query,)
        Estimated linear density for each query sample.

    If full_return is True, returns:

    (mask, density_values) : tuple
        As above.
    (fig, ax) : tuple
        Matplotlib Figure and Axes objects for the density region plot.
    kde_para_dict : dict
        Dictionary with keys: 'kde', 'log_thresh', 'pass_percent', 'density_mean', 'log_density_mean', 'mean_percentile'.

    Examples
    --------
    mask, density = check_density_region(df_ref, df_qry)
    (mask, density), (fig, ax), info = check_density_region(
        df_ref, df_qry, full_return=True)
    """
    # from sklearn.preprocessing import RobustScaler 
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    # 1. feature subset + fill NA
    if features is None:
        features = list(data_reference.columns)
    ref = data_reference[features].fillna(0).copy()
    qry = data_query[features].fillna(0).copy()

    ref_scaled = scaler.fit_transform(ref)
    qry_scaled = scaler.transform(qry)

    # keep original indices so downstream reindexing does not introduce NaN
    ref_df_scaled = pd.DataFrame(ref_scaled, columns=features, index=data_reference.index)
    qry_df_scaled = pd.DataFrame(qry_scaled, columns=features, index=data_query.index)

    
    # 2. build weight vector and scale features
    if feature_weights is None:
        weights = np.ones(len(features))
    else:
        # default weight=1 for any missing key
        weights = np.array([feature_weights.get(f, 1.0)
                           for f in features], dtype=float)
                           
    scales = np.sqrt(weights)
    ref_df_scaled = ref_df_scaled.mul(scales, axis=1)
    qry_df_scaled = qry_df_scaled.mul(scales, axis=1)


    # 3. fit KDE and compute threshold
    kde = fit_kde(ref_df_scaled, cv=cv)
    tail_frac = 1 - confidence_level
    threshold = compute_density_threshold(kde, ref_df_scaled, tail_frac)

    # 4. classify query samples
    mask, density_values = is_in_high_density_region(qry_df_scaled, kde, threshold)

    # 5. compute report metrics
    pass_percent = mask.mean() * 100
    density_mean = density_values.mean()
    log_density_array = kde.score_samples(qry_df_scaled)
    log_density_mean = log_density_array.mean()

    # relative percentile
    dens_ref = np.exp(kde.score_samples(ref_df_scaled))
    sorted_ref = np.sort(dens_ref)
    percentiles = np.searchsorted(sorted_ref,
                                  density_values.values,
                                  side="right") / len(sorted_ref) * 100
    percentile_series = pd.Series(percentiles, index=density_values.index)
    percentile_min = percentile_series.min()
    percentile_max = percentile_series.max()
    percentile_median = percentile_series.median()

    # 6. print brief report
    if verbose:
        print(
            f"Density threshold ({tail_frac*100:.3f}% quantile): {threshold:.5e} (log‑density)")
        print(f"Coverage: {pass_percent:.2f}% of samples in high‑density region")
        print(f"Average similarity (mean density): {density_mean:.3e}")
        print(f"Average log density: {log_density_mean:.3f}")
        print(f"Average relative percentile: {percentile_median:.2f}%")

    # 7. optional visualization (1D/2D)

    fig, ax = visualize_density_region(ref_df_scaled, qry_df_scaled, kde, threshold,
                                       scales=scales, normalize_scaler =scaler ,plot_original=True)
    if if_show_plot:
        plt.show()

    if full_return:
        kde_para_dict = {
            "kde": kde,
            "log_density_array": log_density_array,
            "Density percentile": percentile_series,
            "log_thresh": np.log(threshold),
            "pass_percent": pass_percent,
            "log_density_mean": log_density_mean,
            "min_percentile": percentile_min,
            "median_percentile": percentile_median,
            "max_percentile": percentile_max,
            "density_of_reference": dens_ref,
            }

        return (mask, density_values), (fig, ax), kde_para_dict

    return mask, density_values




def visualize_density_region(
    data_ref_scaled: pd.DataFrame,
    data_qry_scaled: pd.DataFrame,
    kde: KernelDensity,
    density_thresh: float,
    scales: np.ndarray,
    normalize_scaler: Optional[StandardScaler] = None,
    plot_original: bool = True,
    hue_norm: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize KDE density of reference & query samples with optional original-coordinate plotting.

    Parameters
    ----------
    data_ref_scaled : pd.DataFrame
        Reference data used for KDE scoring (already scaled).
    data_qry_scaled : pd.DataFrame
        Query data used for scoring (already scaled).
    kde : KernelDensity
        Fitted KDE model.
    density_thresh : float
        Density threshold to flag high-density region.
    scales : np.ndarray
        sqrt(weights) used to scale each feature; used for inverse-scaling.
    normalize_scaler : StandardScaler, optional
        StandardScaler used to normalize the data.
    plot_original : bool
        If True, rescale coordinates back to original (unscaled) space for plotting.
    hue_norm : bool
        If True, normalize hue (colorbar) across both groups for comparability.

    Returns
    -------
    fig, ax : Matplotlib figure and axes
    """

    # 1. Calculate density values.
    dens_ref = np.exp(kde.score_samples(data_ref_scaled.values))
    dens_qry = np.exp(kde.score_samples(data_qry_scaled.values))

    # 2. If plotting original-scale data, reverse the scaling while preserving column names.
    if plot_original:
        inv_scales = 1.0 / scales
        plot_ref = data_ref_scaled.mul(inv_scales, axis=1)
        plot_qry = data_qry_scaled.mul(inv_scales, axis=1)
        if normalize_scaler is not None:
            plot_ref = normalize_scaler.inverse_transform(plot_ref)
            plot_qry = normalize_scaler.inverse_transform(plot_qry)
            plot_ref = pd.DataFrame(
                plot_ref, columns=data_ref_scaled.columns)
            plot_qry = pd.DataFrame(
                plot_qry, columns=data_qry_scaled.columns)
    else:
        plot_ref = data_ref_scaled.copy()
        plot_qry = data_qry_scaled.copy()

    plot_ref["density"] = dens_ref
    plot_qry["density"] = dens_qry

    # 3. Determine the hue range.
    if hue_norm:
        combined_density = pd.concat(
            [plot_ref["density"], plot_qry["density"]])
        vmin = combined_density.min()
        vmax = combined_density.max()
        vmin1 = vmin2 = vmin
        vmax1 = vmax2 = vmax
    else:
        vmin1 = plot_ref["density"].min()
        vmax1 = plot_ref["density"].max()
        vmin2 = plot_qry["density"].min()
        vmax2 = plot_qry["density"].max()

    # 4. Start plotting.
    # sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # reference scatter
    edgecolors_ref = np.where(
        plot_ref["density"] > density_thresh, "black", "#1f77b4")
    sns.scatterplot(
        data=plot_ref,
        x=plot_ref.columns[0],
        y=plot_ref.columns[1],
        hue="density",
        palette="Blues",
        hue_norm=(vmin1, vmax1),
        legend=False,
        ax=ax,
        edgecolor=edgecolors_ref,
        marker="o",
        linewidth=0.8,
        s=25,
    )


    # query scatter
    edgecolors_qry = np.where(
        plot_qry["density"] > density_thresh, "black", "#d62728")
    sns.scatterplot(
        data=plot_qry,
        x=plot_qry.columns[0],
        y=plot_qry.columns[1],
        hue="density",
        palette="Reds",
        hue_norm=(vmin2, vmax2),
        legend=False,
        ax=ax,
        edgecolors=edgecolors_qry,
        linewidth=0.8,
        s=25,
        marker="o",
    )

    ax.set_xlabel(plot_ref.columns[0])
    ax.set_ylabel(plot_ref.columns[1])

    # fake scatter for legend
    ax.scatter([], [], color="#1f77b4",
               label="Reference (high density)", edgecolors="black", marker="o")
    ax.scatter([], [], color="#d3e4f3",
               label="Reference (low density)", edgecolors="#1f77b4", marker="o")
    ax.scatter([], [], color="#d62728",
               label="Unknown (high density)", edgecolors="black", marker="o")
    ax.scatter([], [], color="#fcb499",
               label="Unknown (low density)", edgecolors="#d62728", marker="o")

    ax.grid(True, linestyle='--', alpha=0.5)

    plt.legend()

    plt.tight_layout()
    # sns.reset_orig()
    return fig, ax


# ----------------------------------------------------------------------
# 6. Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":

    key_features = ["SiO2", "FeO"]

    df_train = pd.DataFrame({
        "SiO2": np.random.normal(50, 5, 1000),
        "FeO": np.random.normal(10, 2, 1000),
        # "CaO": np.random.normal(20, 3, 1000),
    })
    # df_train2 = pd.DataFrame({
    #     "SiO2": np.random.normal(30, 5, 1000),
    #     "FeO": np.random.normal(20, 2, 1000),
    #     # "CaO": np.random.normal(25, 3, 1000),
    # })
    # df_train = pd.concat([df_train, df_train2], axis=0)

    df_test = pd.DataFrame({
        "SiO2": np.random.normal(40, 5, 100),
        "FeO": np.random.normal(15, 2, 100),
        # "CaO": np.random.normal(25, 3, 100),
    })

    inlier_mask, densities = check_density_region(
        df_train[key_features],
        df_test[key_features],
        features=key_features,
        confidence_level=0.95,
        cv=5,
        feature_weights={"SiO2": 0.9, "FeO": 0.1}
    )
    # print("In‑cluster flags:\n", inlier_mask)
    # print("Density estimates:\n", densities)
