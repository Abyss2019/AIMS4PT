'''
Description:
    This file contains functions for plotting model-related results.

Functions:
    - plot_feature_importance: Plot feature importance.
    - confusion_matrix_plot: Plot confusion matrix.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import colors as mcolors


def plot_feature_importance(importance_df, title=None, first_k=None, colored_features=None, x_label="Feature Importance", ax=None):
    '''
    Plot feature importance.

    Parameters:
        importance_df (DataFrame): 
            DataFrame containing feature importance.
                ```
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance
                })
                ```
        title (str, optional): 
            Title of the plot. Default is None.

        first_k (int, optional): 
            Number of top features to display. Default is None (display all).

        colored_features (list, optional):
            List of features to color differently. Default is None (no special coloring).

        x_label (str, optional): 
            Label for the x-axis. Default is "Feature Importance".

    Returns:
        matplotlib.axes.Axes: 
            Axes displaying the feature-importance bars.
    '''

    importance_df = importance_df.sort_values(
        by="Importance", ascending=False)
    importance_df.reset_index(drop=True, inplace=True)

    # str -> float
    importance_df["Importance"] = importance_df["Importance"].astype(float)
    if first_k is not None:
        importance_df = importance_df.iloc[:first_k]

    # # importance_df["Feature"] to not captialize
    # ox_ = importance_df["Feature"].str.split("_").str[0]
    # phase_ = importance_df["Feature"].str.split("_").str[1].str.lower()
    # importance_df["Feature"] = ox_ + " " + phase_

    # Draw horizontal bar chart
    if ax is None:
        _, ax = plt.subplots(dpi=150, figsize=(8, 6))
    if colored_features is not None:
        # Color list with emphasis on selected features
        colors = [
            "moccasin" if feature not in colored_features else "skyblue" for feature in importance_df["Feature"]]
        edgecolor = [
            "black" if feature not in colored_features else "black" for feature in importance_df["Feature"]]
    else:
        colors = ["skyblue"] * len(importance_df)
        edgecolor = ["black"] * len(importance_df)
    ax.barh(importance_df["Feature"], importance_df["Importance"],
            color=colors, edgecolor=edgecolor)
    for i in range(len(importance_df)):

        # Switch label alignment when horizontal space is limited
        if importance_df["Importance"][i] < 0.08*importance_df["Importance"].max():
            ax.text(importance_df["Importance"][i], i,
                    f"{importance_df['Importance'][i]:.2f}", ha='left', va='center')
        else:
            ax.text(importance_df["Importance"][i], i,
                    f"{importance_df['Importance'][i]:.2f}", ha='right', va='center')
    if x_label is not None:

        ax.set_xlabel(x_label)
    if title is not None:
        ax.set_title(title)

    ax.invert_yaxis()  # Keep highest importance at top
    return ax


def shap_plot(
    shap_values,
    feature_order=None,
    shap_sort=True,
    type="summary_plot",
    sample_idx=0,
    feature_name=None,
    interaction_index="auto",
    max_display=None,
    use_matplotlib=False,
    return_fig_ax=False,
):
    '''
    SHAP values visualization.

    Parameters:
        shap_values (shap.Explanation): 
            SHAP explanation object.

        feature_order (list, optional): 
            Order of feature display. Default is None.

        shap_sort (bool, optional): 
            Sort SHAP values in summary plot. Default is True.

        type (str): 
            Plot type to generate. 
            Options: "summary_plot", "force_plot", "waterfall_plot", 
                     "decision_plot", "dependence_plot".
            Default is "summary_plot".

        sample_idx (int): 
            Index of the sample to visualize for single-sample plots. Default is 0.

        feature_name (str, optional): 
            Feature name for dependence plot. Required if type is "dependence_plot".

        interaction_index (str|int): 
            Feature used for interaction coloring in dependence plot. Default is "auto".

        max_display (int, optional): 
            Maximum number of features to display. If None, show all features.

        use_matplotlib (bool): 
            Whether to use static matplotlib for force plot. Default is False.

        return_fig (bool):
            Whether to return the figure object. Default is False.

    Examples:
        #### - Summary plot
        shap_plot(shap_values, type="summary_plot")

        #### - Force plot for one sample
        shap_plot(shap_values, type="force_plot", sample_idx=0)

        #### - Waterfall plot
        shap_plot(shap_values, type="waterfall_plot", sample_idx=0)

        #### - Decision plot
        shap_plot(shap_values, type="decision_plot", sample_idx=0)

        #### - Dependence plot for specific feature
        shap_plot(shap_values, type="dependence_plot", feature_name="MgO")
    '''
    import shap
    import matplotlib.pyplot as plt

    shap.initjs()

    if max_display is None:
        max_display = shap_values.data.shape[1]  # Set to the total number of features.

    if return_fig_ax:
        show = False
    else:
        show = True


    if type == "summary_plot":
        shap.summary_plot(
            shap_values.values,
            shap_values.data,
            feature_names=shap_values.feature_names,
            sort=shap_sort,
            max_display=max_display,
            show=show,
        )

    elif type == "force_plot":
        return shap.force_plot(
            shap_values.base_values[sample_idx],
            shap_values.values[sample_idx],
            shap_values.data[sample_idx],
            feature_names=shap_values.feature_names,
            matplotlib=use_matplotlib,
            show=show,
        )

    elif type == "waterfall_plot":
        shap.plots._waterfall.waterfall_legacy(
            shap_values[sample_idx], max_display=max_display, show=show
        )

    elif type == "decision_plot":
        shap.decision_plot(
            shap_values.base_values[sample_idx],
            shap_values.values[sample_idx:sample_idx+1],
            feature_names=shap_values.feature_names,
            feature_order=feature_order,
            feature_display_range=slice(None, max_display),
            show=show,
        )

    elif type == "dependence_plot":
        if feature_name is None:
            raise ValueError(
                "feature_name must be provided for dependence_plot.")
        shap.dependence_plot(
            feature_name,
            shap_values.values,
            shap_values.data,
            feature_names=shap_values.feature_names,
            interaction_index=interaction_index,
            show=show,
        )

    else:
        raise ValueError(f"Unsupported plot type: {type}")
    
    if return_fig_ax:
        fig = plt.gcf()
        ax = plt.gca()
        return fig, ax
    return None



def confusion_matrix_plot(y_true=None, y_pred=None, confusion_matrix=None, labels=None, title=None, x_label="Predicted", y_label="Actual", ax=None):
    '''
    Plot confusion matrix.

    Parameters:

        y_true (array-like): 
            True labels.

        y_pred (array-like): 
            Predicted labels.

        confusion_matrix (array-like):
            Confusion matrix.

        labels (array-like): 
            Labels to display on the x-axis and y-axis.

        title (str): 
            Title of the plot.

        ax (matplotlib.axes.Axes): 
            Axes object to plot on.
    '''
    if confusion_matrix is None:

        cm = confusion_matrix(y_true, y_pred)

    else:
        cm = confusion_matrix
    if labels is not None:
        cm = pd.DataFrame(cm, index=labels, columns=labels)
    else:
        cm = pd.DataFrame(cm)

    if ax is not None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax)
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if title is not None:
            ax.set_title(title)
        return ax

    plt.figure(figsize=(8, 6), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _align_shap_and_X(shap_df: pd.DataFrame, X: pd.DataFrame):
    """
    Align shap_df and X by index; keep only common samples and common features.
    """
    common_idx = shap_df.index.intersection(X.index)
    if len(common_idx) == 0:
        raise ValueError("shap_df.index and X.index have no overlap; samples cannot be aligned.")

    shap_aligned = shap_df.loc[common_idx]
    X_aligned = X.loc[common_idx]

    common_features = [c for c in shap_aligned.columns if c in X_aligned.columns]
    if len(common_features) == 0:
        raise ValueError(
            "shap_df.columns and X.columns have no overlap; cannot color by feature values."
        )

    shap_aligned = shap_aligned[common_features]
    X_aligned = X_aligned[common_features]
    return shap_aligned, X_aligned


def plot_shap_summary_dot_colored(
    shap_df: pd.DataFrame,
    X: pd.DataFrame,
    top_n: int = 20,
    figsize=(7, 6),
    s: float = 12,
    alpha: float = 0.7,
    jitter: float = 0.18,
    clip_quantiles=(0.01, 0.99),
    ax=None,
    cmap="coolwarm",
    if_show_colorbar: bool = True,
    units: str = ""
):
    """
    SHAP summary dot plot with feature-value coloring (approximate shap.summary_plot dot).

    Parameters
    ----------
    shap_df : pd.DataFrame
        Rows=samples, cols=features, values=SHAP values (can be +/-).
        Denoted as \phi_{ij}.
    X : pd.DataFrame
        Feature values for the SAME samples & features as shap_df.
        Denoted as x_{ij}.
    top_n : int
        Show top-N features ranked by mean(|SHAP|):
        I_j = (1/n) * sum_i |phi_{ij}|
    jitter : float
        Vertical jitter amplitude (approx beeswarm). 0 disables jitter.
    clip_quantiles : tuple(float, float)
        Robust scaling for coloring. Use (q_low, q_high) quantiles per feature.
        For example, (0.01, 0.99).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new axes is created when None.
    if_show_colorbar : bool
        Whether to show color bar.
    """
    shap_aligned, X_aligned = _align_shap_and_X(shap_df, X)

    # Importance ranking: I_j = mean(|phi|)
    importance = shap_aligned.abs().mean(axis=0).sort_values(ascending=False)
    features = importance.head(top_n).index.tolist()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot from bottom to top for consistent ordering with beeswarm style
    for row_i, feat in enumerate(features[::-1]):
        phi = shap_aligned[feat].to_numpy()
        xval = X_aligned[feat].to_numpy()

        # Robust normalization for color: clip to quantiles for stable color range
        ql, qh = np.nanquantile(xval, clip_quantiles)
        x_clip = np.clip(xval, ql, qh)
        norm = mcolors.Normalize(vmin=ql, vmax=qh) if qh != ql else None

        y0 = np.full_like(phi, row_i, dtype=float)
        if jitter and jitter > 0:
            y0 = y0 + (np.random.rand(phi.size) - 0.5) * 2 * jitter

        sc = ax.scatter(
            phi,
            y0,
            c=x_clip if norm is not None else xval,
            cmap=plt.get_cmap(cmap),
            norm=norm,
            s=s,
            alpha=alpha,
            marker="o",
        )

    ax.set_yticks(range(len(features)))
    # yticklabels: features + normalized feature importance (:.2f)
    importance_normalized = importance / importance.sum()
    ticklabels = [
        f"{feat} ({importance_normalized[feat] * 100:.0f}%)"
        for feat in features[::-1]
    ]

    ax.set_yticklabels(ticklabels)
    ax.set_xlabel(f"SHAP value {units}")
    ax.axvline(0.0, linewidth=1)

    if if_show_colorbar:
        cbar = fig.colorbar(sc, ax=ax)
        if sc.get_array().size:
            if norm is not None:
                cbar.set_ticks([norm.vmin, norm.vmax])
            else:
                data_min = float(np.nanmin(sc.get_array()))
                data_max = float(np.nanmax(sc.get_array()))
                cbar.set_ticks([data_min, data_max])
            cbar.set_ticklabels(["Low", "High"])

    # fig.tight_layout()
    return ax
