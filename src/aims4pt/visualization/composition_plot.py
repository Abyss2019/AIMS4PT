'''
This module contains functions for plotting compositions.

Functions:
-----------

    plot_scatter: 
        Plot scatter plot for each pair of x and y columns.
    plot_new_literature: 
        Plot new literature data along with this study's data.
    compare_natural_exp_dataset:
        Compare natural and experimental data.
    plot_histogram_elements:
        Plot histogram for each element in the data_df.


'''



import seaborn as sns
from aims4pt.data_tools.rocks import get_volcanic_rock_series
from typing import Union, Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aims4pt.constants import (
    ERUPTION_COLOR,
    ERUPTION_MARKER,
    SAMPLE_MARKER,
    SAMPLE_COLOR
)
from aims4pt.toolkit_utils import manually_input_sths, set_diy_color_cycle
from aims4pt.utils import normalize_column_names


def plot_scatter(x, y, xlabel, ylabel, title, df_overall, literature_df, eruption_list, eruption_marker, eruption_color, sample_marker, sample_color, manually_input_x_y_labels=False, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # literature_set = set(literature_df["Source "].unique())
    # this_study_set = set(df_overall["Source "].unique())

    # color cycle

    set_diy_color_cycle()

    for eruption in eruption_list:

        data = literature_df[literature_df["Stratigraphic unit"] == eruption]
        # Literature data with alpha set to 0.5.
        ax.scatter(
            data[x], data[y],
            marker=eruption_marker[eruption],
            s=100,
            edgecolor="k",  # Black edge color.
            facecolors=eruption_color[eruption],
            alpha=0.5,  # Set alpha to 0.5.
            label=f"{eruption} (literature)"
        )

        # print eruption composition range
        print(eruption+" (literature)")
        print(x)
        print(f"{data[x].min():.2f} - {data[x].max():.2f}")
        print(y)
        print(f"{data[y].min():.2f} - {data[y].max():.2f}")

    # add a v line at FeO/MgO = 2.7
    if False:
        ymin = ax.get_ylim()[0]
        ymax = ax.get_ylim()[1]
        ax.vlines(2.7, ymin, ymax, colors='k', linestyles='dashed', alpha=0.5)
        # print(f"FeO/MgO = 2.7: {ymin} - {ymax}")

    # Plot this study's data without distinguishing core and rim.
    for eruption in eruption_list:
        data = df_overall[df_overall["Stratigraphic unit"] == eruption]
        ax.scatter(
            data[x], data[y],
            marker=eruption_marker[eruption],
            s=150,
            edgecolor="k",  # Black edge color.
            facecolors=eruption_color[eruption],
            alpha=1,  # Set to opaque.
            label=(f"{eruption} (this study)")
        )

        # print eruption composition range
        print(eruption + " (this study)")
        print(x)
        print(f"{data[x].min():.2f} - {data[x].max():.2f}")
        print(y)
        print(f"{data[y].min():.2f} - {data[y].max():.2f}")

    # deal new literature data
    for lit_keys, lit_df in kwargs.items():
        temp_eruption_list = set(lit_df["Stratigraphic unit"].unique())
        for eruption in temp_eruption_list:
            data = lit_df[(lit_df["Stratigraphic unit"] == eruption)]
            ax.scatter(
                data[x], data[y],
                marker=eruption_marker[eruption],
                s=100,
                edgecolor="k",  # Black edge color.
                alpha=0.8,  # Set alpha to 0.8.
                label=f"{eruption} ({lit_keys})"
            )

            # print eruption composition range
            print(eruption + " (" + lit_keys + ")")
            print(x)
            print(f"{data[x].min():.2f} - {data[x].max():.2f}")
            print(y)
            print(f"{data[y].min():.2f} - {data[y].max():.2f}")

    if manually_input_x_y_labels:
        # manually input x and y labels
        label_dic = manually_input_sths(
            input_mode="auto", x_label=xlabel, y_label=ylabel)
        xlabel = label_dic["x_label"]
        ylabel = label_dic["y_label"]

    # Create and add the legend.
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# new literature arguments
def plot_new_literature(this_study_df, literature_df, x_y_pairs, manually_input_x_y_labels=False, title=True,
                        eruption_marker=None, eruption_color=None, sample_marker=None, sample_color=None, **kwargs):
    """
    Plot new literature data along with this study's data.

    Parameters:
        this_study_df (pd.DataFrame): 
            This study's data.
        literature_df (pd.DataFrame):
            Literature data to plot.
        x_y_pairs (list of tuples):
            List of tuples containing the x and y column names to plot.
        eruption_marker (dict):
            Dictionary mapping eruption names to marker styles.
        eruption_color (dict):
            Dictionary mapping eruption names to colors.
        sample_marker (dict):
            Dictionary mapping sample names to marker styles.
        sample_color (dict):
            Dictionary mapping sample names to colors.
        **kwargs:
            New literature_df arguments to pass to the plotting function. 
            This kwargs are leave for those literature data that require additional legends, and not want to be included in the "literature" legend.
            Format: literature_name (str): literature_df (pd.DataFrame)
            for example:
            - "Nean et al., 2023": literature_df
    """

    # preporcessing
    this_study_df = this_study_df.dropna(how='all', axis=1)

    # mapping
    # the tag includes the thin section for this study and source for literature
    this_study_df["Tag"] = this_study_df["thin section"]
    literature_df["Tag"] = literature_df["Source "]

    literature_df["Comment"] = literature_df["Sample"]
    this_study_df["Sample"] = this_study_df["Comment"]

    # Position: core->C, rim->R
    literature_df["Position"] = np.where(
        literature_df["Position"].str.contains("core"), "C", "R")

    # sort by comment
    this_study_df = this_study_df.sort_values(by=["Comment", "Position"])
    literature_df = literature_df.sort_values(by=["Comment", "Position"])

    this_study_df["FeO/MgO"] = this_study_df["FeO"] / this_study_df["MgO"]
    literature_df["FeO/MgO"] = literature_df["FeO"] / literature_df["MgO"]

    # set the eruption marker and color
    if eruption_marker is None:
        eruption_marker = ERUPTION_MARKER
    if eruption_color is None:
        eruption_color = ERUPTION_COLOR
    if sample_marker is None:
        sample_marker = SAMPLE_MARKER
    if sample_color is None:
        sample_color = SAMPLE_COLOR

    eruption_list = this_study_df["Stratigraphic unit"].unique()
    eruption_list_2 = literature_df["Stratigraphic unit"].unique()
    eruption_list = np.concatenate((eruption_list, eruption_list_2))
    eruption_list = np.unique(eruption_list)

    # plot
    for x, y in x_y_pairs:
        plot_scatter(
            x=x, y=y,
            xlabel=x, ylabel=y,
            title=f"{x} vs {y}" if title else None,
            df_overall=this_study_df,
            literature_df=literature_df,
            eruption_list=eruption_list,
            eruption_marker=eruption_marker,
            eruption_color=eruption_color,
            sample_marker=sample_marker,
            sample_color=sample_color,
            **kwargs
        )


def compare_natural_exp_dataset(natural_df, exp_df, x_y_pairs, T_color=None, P_color=None, kde=False):
    '''
    Compare natural and experimental data. Useful when applying a Geothermobarometry method to a dataset.
    Harker diagrams are plotted for each pair of x and y columns. Colorbar for temperature or pressure can be added.

    Parameters:

        natural_df (DataFrame): 
            DataFrame of natural data.
        exp_df (DataFrame):
            DataFrame of experimental data.
        x_y_pairs (list):
            List of tuples of x and y columns to compare.
        T_color (str):
            Column name for temperature.
            None by default. No colorbar for temperature.
        P_color (str):
            Column name for pressure.
            None by default. No colorbar for pressure.
    '''
    natural_df = natural_df.copy()
    exp_df = exp_df.copy()
    from ..utils import normalize_column_names
    natural_df = normalize_column_names(natural_df)
    exp_df = normalize_column_names(exp_df)

    from .plot_utils import get_subplot_shape
    n = len(x_y_pairs)
    nrow, ncol = get_subplot_shape(n)
    fig, ax = plt.subplots(nrow, ncol, figsize=(20, 5 * nrow), dpi=150)
    ax = ax.flatten()
    cmap = "YlOrRd"

    for i, (x, y) in enumerate(x_y_pairs):
        ax[i].scatter(natural_df[x], natural_df[y], c='blue', s=40,
                      alpha=0.5, label='natural data', marker="2")
        scatter = None
        if T_color:
            scatter = ax[i].scatter(exp_df[x], exp_df[y], c=exp_df[T_color], s=40,
                                    edgecolors='k', alpha=1, cmap=cmap)

        if P_color:
            scatter = ax[i].scatter(exp_df[x], exp_df[y], c=exp_df[P_color], s=40,
                                    edgecolors='k', alpha=1, cmap=cmap)
        else:
            ax[i].scatter(exp_df[x], exp_df[y], c='grey', s=40,
                          edgecolors='k', alpha=1)

        if kde:
            sns.kdeplot(exp_df, x=x, y=y,
                        ax=ax[i], fill=False, levels=1, color='blue')
        # red star, non-filled

        # Add colorbar
        if scatter:
            cbar = fig.colorbar(scatter, ax=ax[i])
            if T_color:
                cbar.set_label(f'Temperature ({T_color})')
            elif P_color:
                cbar.set_label(f'Pressure ({P_color})')

        ax[i].set_xlabel(x)
        ax[i].set_ylabel(y)
        ax[i].grid(True)

    # legend
    plt.legend()
    plt.show()


def plot_histogram_elements(data_df, specific_oxides=None, kde=True, compared_data=None, title=None, **kwargs):
    '''
    Plot histogram for each element in the data_df.

    Parameters:


        data_df (DataFrame): 
            DataFrame of data.
        specific_oxides (list):
            List of specific oxides to plot.
            None by default. Plot all oxides.
        kde (bool):
            Whether to plot the kernel density estimate.
            True by default.
        compared_data (DataFrame):
            DataFrame of data to compare with.
            None by default. No comparison.
            If included, the compared data will be plotted as kde.
        **kwargs:
            Additional keyword arguments for sns.histplot.
    '''
    data_df = data_df.copy()
    from ..utils import normalize_column_names, filter_oxides
    data_df = normalize_column_names(data_df)

    if specific_oxides is None:
        specific_oxides = filter_oxides(data_df.columns.tolist())

    n = len(specific_oxides)

    from .plot_utils import get_subplot_shape, get_robust_xlim
    nrow, ncol = get_subplot_shape(n)
    fig, ax = plt.subplots(nrow, ncol, figsize=(5*ncol, 5 * nrow), dpi=150)
    ax = ax.flatten()

    if compared_data is not None:
        compared_data = normalize_column_names(compared_data)

    for i, oxide in enumerate(specific_oxides):
        if compared_data is not None:

            sns.histplot(data_df[oxide], kde=False, ax=ax[i], **
                         kwargs, stat='density', label="experimental data")
            # kde only for compared
            sns.kdeplot(compared_data[oxide], ax=ax[i],
                        color='red', label='this study', **kwargs)

        else:
            sns.histplot(data_df[oxide], kde=kde, ax=ax[i], **kwargs)

        ax[i].set_xlabel(oxide + " (wt%)", fontsize=12)
        ax[i].set_ylabel("Frequency")
        ax[i].grid(True)

        if compared_data is not None:
            lower_lim, upper_lim = get_robust_xlim(
                data_df[oxide], compared_data[oxide], allow_negative=False)
        else:
            lower_lim, upper_lim = get_robust_xlim(
                data_df[oxide], allow_negative=False)

        ax[i].set_xlim(lower_lim, upper_lim)

        if i == 0 and compared_data is not None:
            ax[i].legend(loc='upper right', fontsize=12)

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()


def _calculate_tukey_fences(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return None

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _add_tukey_fence_overlay(ax, x_values, y_values, x_label, y_label, fence_axis="both"):
    valid_axes = {"x", "y", "both"}
    if fence_axis not in valid_axes:
        raise ValueError(f"fence_axis must be one of {sorted(valid_axes)}.")

    fence_lines = []
    if fence_axis in {"x", "both"}:
        x_fences = _calculate_tukey_fences(x_values)
        if x_fences is not None:
            lower, upper = x_fences
            ax.axvspan(lower, upper, color="tab:green", alpha=0.08, zorder=0)
            ax.axvline(lower, color="tab:green", linestyle="--", linewidth=1.2, zorder=1)
            ax.axvline(upper, color="tab:green", linestyle="--", linewidth=1.2, zorder=1)
            fence_lines.append(f"{x_label}: {lower:.2f}-{upper:.2f}")

    if fence_axis in {"y", "both"}:
        y_fences = _calculate_tukey_fences(y_values)
        if y_fences is not None:
            lower, upper = y_fences
            ax.axhspan(lower, upper, color="tab:orange", alpha=0.08, zorder=0)
            ax.axhline(lower, color="tab:orange", linestyle="--", linewidth=1.2, zorder=1)
            ax.axhline(upper, color="tab:orange", linestyle="--", linewidth=1.2, zorder=1)
            fence_lines.append(f"{y_label}: {lower:.2f}-{upper:.2f}")

    if fence_lines:
        print(f"Tukey 1.5 x IQR fences ({y_label} vs {x_label}): " + "; ".join(fence_lines))


def _add_transparent_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    legend_items = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label and not label.startswith("_")
    ]
    if not legend_items:
        return None

    handles, labels = zip(*legend_items)
    return ax.legend(
        handles,
        labels,
        frameon=True,
        framealpha=0.75,
        facecolor="white",
        edgecolor="0.7",
        **kwargs,
    )


def _resolve_panel_fence_axis(fence_axis, panel_index):
    if panel_index == 0:
        return fence_axis
    if fence_axis == "both":
        return "y"
    if fence_axis == "x":
        return None
    return fence_axis


def plot_all_Harker_diagrams(
    data_df,
    x_col,
    y_compositions,
    T_color=None,
    P_color=None,
    kde=False,
    ax=None,
    color='grey',
    label=None,
    size=40,
    edgecolors='k',
    distribution_overlay=None,
    fence_axis="both",
):
    '''
    Compare natural and experimental data. Useful when applying a Geothermobarometry method to a dataset.
    Harker diagrams are plotted for each pair of x and y columns. Colorbar for temperature or pressure can be added.

    Parameters:
        data_df (DataFrame): 
            DataFrame of data.
        x_col (str):
            Column name for x-axis.
        y_compositions (list):
            List of column names for y-axis.    
        T_color (str):
            Column name for temperature.
            None by default. No colorbar for temperature.
        P_color (str):
            Column name for pressure.
            None by default. No colorbar for pressure.
        kde (bool):
            Whether to plot the kernel density estimate.
            False by default. Kept for backward compatibility; use
            distribution_overlay="kde" for new code.
        distribution_overlay (str or None):
            Optional reference overlay. Use "kde" for the kernel density
            contour or "tukey_fence" for Tukey's 1.5 x IQR fences.
            If None, kde=True still draws the KDE overlay.
        fence_axis (str):
            Which Tukey fence axes to draw when distribution_overlay is
            "tukey_fence": "x", "y", or "both".
        ax (Axes or array of Axes):
            Axes to plot on.
            If None, new axes will be created.
        color (str):
            Color for the scatter points.
            'grey' by default.
        label (str):
            Label for the scatter points.
            None by default.
    
    returns:
        ax (array of Axes):
            Array of Axes with the plots.
    '''
    _data_df = data_df.copy()

    from ..utils import normalize_column_names
    _data_df_norm = normalize_column_names(_data_df)

    if distribution_overlay is None and kde:
        distribution_overlay = "kde"
    if distribution_overlay not in {None, "kde", "tukey_fence"}:
        raise ValueError('distribution_overlay must be None, "kde", or "tukey_fence".')


    from .plot_utils import get_subplot_shape
    n = len(y_compositions)
    nrow, ncol = get_subplot_shape(n)
    if ax is None:
        fig, ax = plt.subplots(nrow, ncol, figsize=(20, 5 * nrow), dpi=150)
        ax = ax.flatten()
        for extra_ax in ax[n:]:
            fig.delaxes(extra_ax)
        ax = ax[:n]
    else:
        ax = np.array(ax).ravel()
        fig = ax[0].figure
        for extra_ax in ax[n:]:
            fig.delaxes(extra_ax)
        ax = ax[:n]
    # elif len(ax) != n:
    #     raise ValueError("ax should be a list of axes with length equal to the number of y_compositions.")
    cmap = "YlOrRd"
    x = x_col
    for i, y in enumerate(y_compositions):
        scatter = None
        if T_color:
            scatter = ax[i].scatter(_data_df_norm[x], _data_df_norm[y], c=_data_df_norm[T_color], s=40,
                                    edgecolors='k', alpha=1, cmap=cmap, label=label)

        if P_color:
            scatter = ax[i].scatter(_data_df_norm[x], _data_df_norm[y], c=_data_df_norm[P_color], s=40,
                                    edgecolors='k', alpha=1, cmap=cmap, label=label)
        else:
            ax[i].scatter(_data_df_norm[x], _data_df_norm[y], c=color, s=size,
                          edgecolors=edgecolors, alpha=1, label=label)

        if distribution_overlay == "kde":
            sns.kdeplot(_data_df_norm, x=x, y=y,
                        ax=ax[i], fill=False, levels=1, color='blue')
        elif distribution_overlay == "tukey_fence":
            panel_fence_axis = _resolve_panel_fence_axis(fence_axis, i)
            if panel_fence_axis is not None:
                _add_tukey_fence_overlay(
                    ax[i],
                    _data_df_norm[x],
                    _data_df_norm[y],
                    x,
                    y,
                    fence_axis=panel_fence_axis,
                )
        # red star, non-filled

        # Add colorbar
        if scatter:
            cbar = fig.colorbar(scatter, ax=ax[i])
            if T_color:
                cbar.set_label(f'Temperature ({T_color})')
            elif P_color:
                cbar.set_label(f'Pressure ({P_color})')

        ax[i].set_xlabel(x)
        ax[i].set_ylabel(y)
        ax[i].grid(True, color='grey', linestyle='--', linewidth=0.5)

    legend_by_label = {}
    for plot_ax in ax[:n]:
        handles, labels = plot_ax.get_legend_handles_labels()
        for handle, item_label in zip(handles, labels):
            if item_label and not item_label.startswith("_") and item_label not in legend_by_label:
                legend_by_label[item_label] = handle

        legend = plot_ax.get_legend()
        if legend is not None:
            legend.remove()

    for fig_legend in list(fig.legends):
        fig_legend.remove()

    if legend_by_label:
        legend_anchor_ax = ax[0]
        legend_anchor_ax.legend(
            legend_by_label.values(),
            legend_by_label.keys(),
            loc="upper right",
            bbox_to_anchor=(-0.02, 1.0),
            frameon=True,
            framealpha=0.75,
            facecolor="white",
            edgecolor="0.7",
        )

    # plt.show()
    plt.tight_layout(rect=(0.08, 0, 1, 1))
    
    return  ax




def compare_two_datasets(
    df1,
    df2,
    x_y_pairs,
    df1_label="Dataset 1",
    df2_label="Dataset 2",
    df1_color="blue",
    df2_color="black",
    df1_marker="o",
    df2_marker="*",
    df1_alpha=0.5,
    df2_alpha=1.0,
    color_col=None,
    color_label=None,
    cmap="YlOrRd",
    normalize_cols=True,
):
    """
    Compare two datasets on multiple x–y pairs (e.g. Harker / scatter plots).
    Optionally color the second dataset by a given column.

    Parameters
    ----------
    df1 : DataFrame
        First dataset.
    df2 : DataFrame
        Second dataset.
    x_y_pairs : list[(str, str)]
        List of (x, y) columns to plot.
    df1_label : str
        Legend label for Dataset 1.
    df2_label : str
        Legend label for Dataset 2.
    df1_color : str
        Color for Dataset 1.
    df2_color : str
        Base color for Dataset 2 (when color_col is None).
    df1_marker : str
        Marker style for Dataset 1.
    df2_marker : str
        Marker style for Dataset 2.
    df1_alpha : float
        Alpha for Dataset 1.
    df2_alpha : float
        Alpha for Dataset 2.
    color_col : str or None
        Column in df2 used for color mapping.
    color_label : str or None
        Colorbar label. Defaults to color_col.
    cmap : str
        Colormap used when color_col is provided.
    normalize_cols : bool
        If True, normalize column names in both DataFrames.

    """

    df1 = df1.copy()
    df2 = df2.copy()

    if normalize_cols:
        from ..utils import normalize_column_names
        df1 = normalize_column_names(df1)
        df2 = normalize_column_names(df2)

    from .plot_utils import get_subplot_shape
    n = len(x_y_pairs)
    nrow, ncol = get_subplot_shape(n)
    fig, ax = plt.subplots(nrow, ncol, figsize=(20, 5 * nrow), dpi=150)
    ax = ax.flatten()

    # For colorbar usage
    last_scatter = None

    for i, (x, y) in enumerate(x_y_pairs):
        # Dataset 1
        ax[i].scatter(
            df1[x], df1[y],
            s=40,
            c=df1_color,
            alpha=df1_alpha,
            marker=df1_marker,
            label=df1_label,
        )

        # Dataset 2
        if color_col:
            last_scatter = ax[i].scatter(
                df2[x], df2[y],
                s=40,
                c=df2[color_col],
                cmap=cmap,
                alpha=df2_alpha,
                edgecolors="k",
                marker=df2_marker,
                label=df2_label,
            )
        else:
            ax[i].scatter(
                df2[x], df2[y],
                s=40,
                c=df2_color,
                alpha=df2_alpha,
                edgecolors="k",
                marker=df2_marker,
                label=df2_label,
            )

        ax[i].set_xlabel(x)
        ax[i].set_ylabel(y)
        ax[i].grid(True)

    # Optional colorbar
    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=ax)
        cbar.set_label(color_label if color_label else str(color_col))

    # #### Global legend at top ####
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=df1_marker, linestyle='',
               markerfacecolor=df1_color, markersize=10, label=df1_label),
        Line2D([0], [0], marker=df2_marker, linestyle='',
               markerfacecolor=df2_color, markersize=12, label=df2_label),
    ]

    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=14
    )


def plot_glass_TAS_diagram(
    X_liq,
    model="LeMaitreCombined",
    axes=None,
    color=None,
    marker='o',
    label=None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    plot_alkaline_boundary: bool = False,
    kde = False,
    add_TAS_labels: bool = True,
    fill= False,
    **kwargs
):
    """
    Plot the liquid TAS diagram using the pyrolite package.
    
    This method exports liquid phase data (using to_pandas_by_phase('liq')), computes
    "Na2O + K2O", initializes a TAS model, and overlays the TAS classification background
    with a scatter plot of data points.
    
    Parameters
    ----------
    X_liq : DataFrame
        DataFrame containing liquid composition data.
    model : str, optional
        The TAS model to use (default is "LeMaitreCombined").
    axes : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    color : str or array-like, optional
        Color for the scatter points. If None, defaults to 'lightblue'.
    marker : str, optional
        Marker style for the scatter points (default is 'o').
    xlim : tuple(float, float), optional
        Custom SiO2 limits for the TAS background and scatter. Defaults to
        the data-driven bounds if None.
    ylim : tuple(float, float), optional
        Custom Na2O + K2O limits for the TAS background and scatter. Defaults
        to the data-driven upper bound starting at zero if None.

    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the TAS diagram.
    """
    # normalize column names
    liq_df = normalize_column_names(X_liq)
    
    # Compute Na2O + K2O
    liq_df["Na2O + K2O"] = liq_df["Na2O"] + liq_df["K2O"]
    
    if xlim is None:
        x_min = liq_df["SiO2"].min() - 1
        x_max = liq_df["SiO2"].max() + 1
    else:
        x_min, x_max = xlim

    if ylim is None:
        y_min, y_max = 0, liq_df["Na2O + K2O"].max() + 1
    else:
        y_min, y_max = ylim

    from pyrolite.util.classification import TAS
    cm = TAS(which_model=model, xlim=(x_min, x_max), ylim=(y_min, y_max))
    
    if axes is  None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    else:
        ax = axes
        fig = ax.figure
    ax.scatter(liq_df["SiO2"], liq_df["Na2O + K2O"],
                c=color if color is not None else 'lightblue', 
                marker=marker, 
                label=label, **kwargs)
    tas_pred = cm.predict(liq_df[["SiO2", "Na2O + K2O"]])
    cm.add_to_axes(
        ax,
        alpha=1,
        linewidth=0.5,
        zorder=-1,
        add_labels=add_TAS_labels,
        which_labels="volcanic",
        which_ids= np.unique(tas_pred),
        label_at_centroid=True,
        fill=fill,
        facecolor= "lightgrey",
    )
    cm.add_to_axes(
        ax,
        alpha=1,
        linewidth=0.5,
        zorder=0,
        add_labels=add_TAS_labels,
        which_labels="volcanic",
        label_at_centroid=True,
        fill=False,
    )

    # plot Alkaline vs Subalkaline boundary line
    if plot_alkaline_boundary :
        ys = np.linspace(y_min, y_max, 100)
        Na2O_plus_K2O_boundary = (
            -3.3539e-4 * ys**6
            + 1.2030e-2 * ys**5
            - 1.5188e-1 * ys**4
            + 8.6096e-1 * ys**3
            - 2.1111    * ys**2
            + 3.9492    * ys
            + 39.0
        )
        ax.plot(Na2O_plus_K2O_boundary, ys, color='black', linestyle='--', linewidth=2, label='Alkaline/Subalkaline boundary')

    # kde 
    if kde:
        sns.kdeplot(
            data=liq_df,
            x="SiO2",
            y="Na2O + K2O",
            ax=ax,
            fill=False,
            levels=4,
            color='red',
            linewidths=1,
            alpha=0.7
        )
    ax.set_xlabel(r"$\mathrm{SiO_2}$ (wt%)")
    ax.set_ylabel(r"$\mathrm{Na_2O + K_2O}$ (wt%)")

    ax.set_xlim(35, 83)
    # ax.set_ylim(y_min, y_max)

    return ax





def plot_volcanic_rock_series(
    df: Union[pd.DataFrame, pd.Series],
    figsize: Tuple[float, float] = (12, 5),
    alpha: float = 0.8,
    s: float = 30,
    annotate_n: bool = True,
    show_unknown: bool = True,
    return_data: bool = False,
    ax: Optional[np.ndarray] = None,
):
    """
    Visualize volcanic rock series classification following the same logic as get_volcanic_rock_series().

    Panels
    ------
    (A) TAS boundary: Alkaline vs Subalkaline
        x = SiO2 (wt.%), y = Na2O + K2O (wt.%)
        Subalkaline if: SiO2 >= f(A), where A = Na2O + K2O and f is the polynomial used in your code.

    (B) FeO*/MgO vs SiO2 boundary: Calc-alkaline vs Tholeiitic (applied within Subalkaline)
        FeO* = FeO + 0.899 * Fe2O3
        R = FeO* / MgO
        Calc-alkaline if: SiO2 >= 6.4 * R + 42.8

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Must contain (or be normalizable to) SiO2, FeO, MgO, Fe2O3, Na2O, K2O (wt.%).
    figsize : (w, h)
        Figure size if ax is None.
    alpha : float
        Scatter transparency.
    s : float
        Marker size.
    annotate_n : bool
        If True, append sample counts to legend labels.
    show_unknown : bool
        If False, drop samples classified as "unknown" from plots and legend.
    return_data : bool
        If True, also return the per-sample classification table.
    ax : array-like of matplotlib Axes, optional
        Provide existing axes of shape (2,) or (1,2). If None, create a new figure.

    Returns
    -------
    fig, axes  (and optionally result_df)
        fig: matplotlib Figure
        axes: ndarray of Axes with length 2
        result_df (optional): DataFrame with SiO2, A, FeO*, R, is_subalkaline, is_calc_alkaline, series.
    """

    if df is None or (hasattr(df, "empty") and df.empty):
        if return_data:
            return None, None, pd.DataFrame()
        return None, None

    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df = df.copy()

    df = normalize_column_names(
        df,
        ["SiO2", "FeO", "MgO", "Fe2O3", "Na2O", "K2O"],
        missing_fill=0
    )
    df["FeO_star"] = df["FeO"] + df["Fe2O3"] * 0.899  # FeO* = FeO + 0.899 Fe2O3

    A = df["Na2O"].fillna(0) + df["K2O"].fillna(0)  # A = Na2O + K2O
    boundary_poly = (
        -3.3539e-4 * A**6
        + 1.2030e-2 * A**5
        - 1.5188e-1 * A**4
        + 8.6096e-1 * A**3
        - 2.1111    * A**2
        + 3.9492    * A
        + 39.0
    )
    is_subalkaline = df["SiO2"] >= boundary_poly

    R = df["FeO_star"].div(df["MgO"].replace(0, np.nan)).fillna(0)  # R = FeO*/MgO
    is_calc_alkaline = df["SiO2"] >= 6.4 * R + 42.8

    conditions = [
        ~is_subalkaline,
        is_subalkaline & is_calc_alkaline,
        is_subalkaline & ~is_calc_alkaline,
    ]
    choices = ["alkaline", "calc-alkaline", "tholeiitic"]
    series = np.select(conditions, choices, default="unknown")

    result_df = pd.DataFrame({
        "SiO2": df["SiO2"].astype(float),
        "Na2O": df["Na2O"].astype(float),
        "K2O": df["K2O"].astype(float),
        "A_Na2O_plus_K2O": A.astype(float),
        "FeO_star": df["FeO_star"].astype(float),
        "MgO": df["MgO"].astype(float),
        "R_FeOstar_over_MgO": R.astype(float),
        "is_subalkaline": is_subalkaline.values,
        "is_calc_alkaline": is_calc_alkaline.values,
        "series": series
    })

    if not show_unknown:
        result_df = result_df[result_df["series"] != "unknown"].copy()

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        axes = np.array(ax).ravel()
        if len(axes) != 2:
            raise ValueError("ax must contain exactly 2 matplotlib Axes.")
        fig = axes[0].figure

    ax1, ax2 = axes

    A_grid = np.linspace(
        np.nanmin(result_df["A_Na2O_plus_K2O"].values),
        np.nanmax(result_df["A_Na2O_plus_K2O"].values),
        200
    )
    x_boundary = (
        -3.3539e-4 * A_grid**6
        + 1.2030e-2 * A_grid**5
        - 1.5188e-1 * A_grid**4
        + 8.6096e-1 * A_grid**3
        - 2.1111    * A_grid**2
        + 3.9492    * A_grid
        + 39.0
    )
    ax1.plot(x_boundary, A_grid, linewidth=1.5, label="TAS boundary")

    # scatter by series
    counts = result_df["series"].value_counts().to_dict()
    for key, sub in result_df.groupby("series", sort=False):
        lbl = f"{key} (n={counts.get(key, 0)})" if annotate_n else key
        ax1.scatter(
            sub["SiO2"], sub["A_Na2O_plus_K2O"],
            s=s, alpha=alpha, label=lbl
        )
    ax1.set_xlim(35, 83)

    ax1.set_xlabel(r"$\mathrm{SiO_2}$ (wt.%)")
    ax1.set_ylabel(r"$\mathrm{Na_2O}+\mathrm{K_2O}$ (wt.%)")
    ax1.set_title("TAS: Alkaline vs Subalkaline")
    ax1.legend(frameon=False)

    R_grid = np.linspace(
        np.nanmin(result_df["R_FeOstar_over_MgO"].values),
        np.nanmax(result_df["R_FeOstar_over_MgO"].values),
        200
    )
    # Boundary: SiO2 = 6.4 * R + 42.8  -> draw x=6.4R+42.8 vs y=R
    ax2.plot(6.4 * R_grid + 42.8, R_grid, linewidth=1.5, label=r"$\mathrm{SiO_2}=6.4R+42.8$")

    for key, sub in result_df.groupby("series", sort=False):
        lbl = f"{key} (n={counts.get(key, 0)})" if annotate_n else key
        ax2.scatter(
            sub["SiO2"], sub["R_FeOstar_over_MgO"],
            s=s, alpha=alpha, label=lbl
        )

    ax2.set_xlabel(r"$\mathrm{SiO_2}$ (wt.%)")
    ax2.set_ylabel(r"$R=\mathrm{FeO^{\ast}}/\mathrm{MgO}$ (dimensionless)")
    ax2.set_title(r"$\mathrm{FeO^{\ast}}/\mathrm{MgO}$ vs $\mathrm{SiO_2}$: Calc-alkaline vs Tholeiitic")
    ax2.legend(frameon=False)

    ax2.set_xlim(35, 83)
    ax2.set_ylim(0, 250)

    fig.tight_layout()

    if return_data:
        return fig, axes, result_df
    return fig, axes


from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_volcanic_rock_series_overlay(
    df: Union[pd.DataFrame, pd.Series],
    figsize: Tuple[float, float] = (12, 5),
    alpha: float = 0.8,
    s: float = 30,
    show_unknown: bool = True,
    return_data: bool = False,
    ax: Optional[np.ndarray] = None,

    # --- overlay style controls ---
    color=None,
    marker: str = "o",
    label: Optional[str] = None,
    plot_boundaries: bool = True,
    boundary_label: bool = False,
    add_legend: bool = False,

    # --- robustness controls (new and enabled by default) ---
    mgO_floor: float = 0.10,     # wt.% ; avoid FeO*/MgO blow-up when MgO ~ 0
    A_grid_max: float = 20.0,    # Na2O+K2O range for TAS boundary
    R_grid_max: float = 30.0,    # R range for FeO*/MgO boundary
    R_ylim_cap: float = 30.0,    # cap y-limit for readability
    R_ylim_quantile: float = 95.0,  # use upper quantile for y-limit (not dominated by tails)
):
    """
    Overlay-friendly volcanic rock series plotter (2 panels):
      (1) TAS: Na2O+K2O vs SiO2 with alkaline/subalkaline boundary
      (2) FeO*/MgO vs SiO2 with calc-alkaline/tholeiitic boundary

    Key fixes vs your current version:
      - TAS boundary grid uses realistic A range (default 0–20), not 0–200
      - R = FeO*/MgO uses MgO floor to avoid explosion at MgO ~ 0
      - Right-panel y-limits use robust quantile + hard cap so the plot won't "explode"
    """

    if df is None or (hasattr(df, "empty") and df.empty):
        if return_data:
            return None, None, pd.DataFrame()
        return None, None

    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df = df.copy()

    # You already have this helper in your project
    df = normalize_column_names(
        df,
        ["SiO2", "FeO", "MgO", "Fe2O3", "Na2O", "K2O"],
        missing_fill=0
    )

    # FeO* definition
    df["FeO_star"] = df["FeO"] + df["Fe2O3"] * 0.899

    # ---------- TAS boundary (alkaline vs subalkaline) ----------
    A = df["Na2O"].fillna(0) + df["K2O"].fillna(0)
    boundary_poly = (
        -3.3539e-4 * A**6
        + 1.2030e-2 * A**5
        - 1.5188e-1 * A**4
        + 8.6096e-1 * A**3
        - 2.1111    * A**2
        + 3.9492    * A
        + 39.0
    )
    is_subalkaline = df["SiO2"] >= boundary_poly

    # ---------- FeO*/MgO boundary (calc-alkaline vs tholeiitic) ----------
    # Robust R: avoid MgO very close to zero blowing up R to huge numbers
    MgO_safe = df["MgO"].where(df["MgO"] >= mgO_floor, np.nan)
    R = df["FeO_star"].div(MgO_safe)  # keep NaN; do NOT fillna(0)
    is_calc_alkaline = df["SiO2"] >= 6.4 * R + 42.8

    conditions = [
        ~is_subalkaline,
        is_subalkaline & is_calc_alkaline.fillna(False),
        is_subalkaline & ~is_calc_alkaline.fillna(False),
    ]
    choices = ["alkaline", "calc-alkaline", "tholeiitic"]
    series = np.select(conditions, choices, default="unknown")

    result_df = pd.DataFrame({
        "SiO2": df["SiO2"].astype(float),
        "A_Na2O_plus_K2O": A.astype(float),
        "FeO_star": df["FeO_star"].astype(float),
        "MgO": df["MgO"].astype(float),
        "R_FeOstar_over_MgO": R.astype(float),
        "is_subalkaline": is_subalkaline.values,
        "is_calc_alkaline": is_calc_alkaline.fillna(False).values,
        "series": series
    })

    if not show_unknown:
        result_df = result_df[result_df["series"] != "unknown"].copy()

    # ---------- axes ----------
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        axes = np.array(ax).ravel()
        if len(axes) != 2:
            raise ValueError("ax must contain exactly 2 matplotlib Axes.")
        fig = axes[0].figure
    ax1, ax2 = axes
