'''
This script is used to plot thermobarometry results.

The script contains the following functions:
    - box_plot: plot thermobarometry results. Box plot for method comparison.
    - T_P_kde_plot: plot the kernel density estimation (KDE) of the thermobarometry results. The KDE is plotted for all data and for data within specified error ranges.
    - compare_natural_exp_dataset: compare natural and experimental data. Useful when applying a Geothermobarometry method to a dataset.
    - compare_natural_exp_dataset_v_lazy: compare natural and experimental data. Useful when applying a Geothermobarometry method to a dataset. Input data with single phase only.
    - plot_P_T_oxides: plot P-T-oxides diagram. Compare the oxides in data_1 and data_2.
    - plot_error_kde: plot the kernel density estimation (KDE) of the oxides in the cpx_AL24_exp DataFrame. The KDE is plotted for all data and for data within specified error ranges.

'''

import seaborn as sns
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FixedLocator, NullLocator
from matplotlib.transforms import blended_transform_factory
from aims4pt.utils import normalize_column_names
from aims4pt.toolkit_utils import wrap_text
from aims4pt.visualization.plot_utils import get_subplot_shape


MANUSCRIPT_AXIS_LABEL_SIZE = 17
MANUSCRIPT_Y_AXIS_LABEL_SIZE = 18
MANUSCRIPT_TICK_LABEL_SIZE = 14
MANUSCRIPT_MODEL_TICK_LABEL_SIZE = 15
MANUSCRIPT_GROUP_LABEL_SIZE = 17
MANUSCRIPT_LEGEND_SIZE = 14
MANUSCRIPT_PANEL_LABEL_SIZE = 19
MANUSCRIPT_ANNOTATION_SIZE = 13
MANUSCRIPT_RESERVOIR_LABEL_SIZE = 15
MANUSCRIPT_THIS_STUDY_AXIS_LABEL_SIZE = 19
MANUSCRIPT_THIS_STUDY_Y_AXIS_LABEL_SIZE = 20
MANUSCRIPT_THIS_STUDY_TICK_LABEL_SIZE = 16
MANUSCRIPT_THIS_STUDY_MODEL_TICK_LABEL_SIZE = 18
MANUSCRIPT_THIS_STUDY_GROUP_LABEL_SIZE = 20
MANUSCRIPT_THIS_STUDY_ANNOTATION_SIZE = 15
MANUSCRIPT_THIS_STUDY_LEGEND_SIZE = 16


def box_plot(data, P_T,  model_list, model_uncertainty=None, eruption_list=None, title=''):
    """
    Plot thermobarometry results as box plots for method comparison.

    Displays the median line and value, as well as the error space (median ± model uncertainty, if provided).
    The upper and lower bars represent the 0th and 100th percentiles.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing thermobarometry results.
        Required columns: 'eruption' (if multiple eruptions), and model result columns.
        Example table:
            id | T_Putirka08 | P_Putirka08 | T_Putirka96 | P_Putirka96 | eruption
            1  |    1000     |     2.5     |    1100      |     3.0     |   OTT
            2  |    1100     |     3.0     |    1200      |     4.0     |   MTT
            3  |    1200     |     4.0     |    1300      |     5.0     |   YTT
    P_T : str
        "P" for pressure or "T" for temperature mode.
    model_list : list of str
        List of model names (column names) to plot.
        Example: ['T_Putirka08', 'P_Putirka08', 'T_Putirka96', 'P_Putirka96']
    model_uncertainty : dict, optional
        Dictionary mapping model names to their uncertainties.
        If None (default), no uncertainty area is shown.
    eruption_list : list of str, optional
        List of eruption names to plot.
        If None (default), plots all data as a single group.
    title : str, optional
        Title of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object.
    ax : matplotlib.axes.Axes
        The main Axes object.
    bps : list
        List of boxplot objects.
    """
    from aims4pt.toolkit_utils import wrap_text
    one_eruption_mode = True if eruption_list is None or len(
        eruption_list) == 1 else False
    num_models = len(model_list)
    num_eruption = len(eruption_list) if eruption_list is not None else 1
    figure_width = max(2* num_models * num_eruption, 12)

    # Set model colors
    model_colors = {}
    for model_name in model_list:
        # C0 - C9: default color cycle
        model_colors[model_name] = f'C{model_list.index(model_name)}'

    # Filter data
    if one_eruption_mode:
        plot_data = data
        plot_data["eruption"] = "All"
    else:
        plot_data = data[(data['eruption'].isin(eruption_list))]

    fig, ax = plt.subplots(figsize=(figure_width, 6), dpi=300)
    pos = 0
    xticks = []
    xticklabels = []
    x2ticks = []
    x2ticklabels = []

    # Record the position of each eruption event to center the label
    eruption_positions = {}

    # Plot box plots for each model in each eruption event
    bps = []
    if one_eruption_mode:
        eruption_list = ['All']
    for eruption in eruption_list:
        plot_data_eruption = plot_data[plot_data['eruption'] == eruption]
        for model_name in model_list:
            
            if model_name not in plot_data_eruption.columns:
                # fake boxplot
                pos += 1
                x2ticks.append(pos)
                x2ticklabels.append(model_name)
                bp = ax.boxplot([], positions=[pos], widths=0.5,
                                patch_artist=True, meanline=False, showfliers=False,
                                boxprops=dict(facecolor='lightgray', color="black"),
                                medianprops=dict(color="black"),
                                whis=[0, 100],
                                )
                bp['boxes'][0].set_alpha(0.5)
                bps.append(bp)

                print(f'Model {model_name} not found in data for eruption {eruption}. Skipping.')
                continue
            y = model_name
            values = plot_data_eruption[y].dropna().values

            if len(values) == 0:
                print(f'No data for {eruption} - {model_name}')
                continue

            alpha = 0.5 if 'Szy23' in model_name else 1

            color = model_colors[model_name]

            # add a shaded area to indicate the uncertainty of the model
            if model_uncertainty is not None:
                if model_name in model_uncertainty:
                    median_value = np.median(values)
                    ax.fill_between([pos-0.3, pos+0.3], median_value+model_uncertainty[model_name],
                                    median_value-model_uncertainty[model_name], color='gray', alpha=0.3, edgecolor=None)

            bp = ax.boxplot(values,
                            positions=[pos], widths=0.5,
                            patch_artist=True,
                            meanline=False, showfliers=False,
                            boxprops=dict(facecolor=color, color="black"),
                            medianprops=dict(color="black"),
                            whis=[0, 100],

                            )
            bp['boxes'][0].set_alpha(alpha)

            # Display sample size n
            # if model_name == model_list[-1]:
            ax.text(pos, np.min(values) if P_T == 'T' else np.max(values),
                    f'n = {len(values)}', ha='center', va='bottom', fontsize=12, color="#003C78")
            # Display median value
            median_value = np.median(values)
            ax.text(pos, median_value, f'{median_value:.1f}', ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

            if eruption not in eruption_positions:
                eruption_positions[eruption] = []
            eruption_positions[eruption].append(pos)

            # Set model name label
            x2ticks.append(pos)
            # x2 label
            model_label = wrap_text(model_name, 15)  # Wrap text to 10 characters
            x2ticklabels.append(model_label)

            # Update position counter
            pos += 1
            bps.append(bp)


        # Add a gap between different eruption events
        pos += 1

    # Set the labels for eruption events, display one and center it
    for eruption_name, positions in eruption_positions.items():
        center_position = sum(positions) / len(positions)  # Calculate the center position
        xticks.append(center_position)
        xticklabels.append(eruption_name)

    # Set the ticks for the main x-axis and secondary x-axis
    if not one_eruption_mode:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, ha='center', fontsize=12)
    else:
        # remove the ticks for the main x-axis
        ax.set_xticks([])
        ax.set_xticklabels([])

    ax2 = ax.twiny()
    ax2.set_xticks(x2ticks)
    ax2.set_xticklabels(x2ticklabels, ha='center', fontsize=12)

    # Align the range of the main axis and the secondary axis
    ax.set_xlim(-0.5, pos - 1.5)
    ax2.set_xlim(ax.get_xlim())

    if P_T == 'P':
        # Add a new left y-axis and set the depth unit
        ax3 = ax.twinx()  # Create a secondary axis sharing the y-axis
        ax3.set_ylabel('Depth (km)', color='gray')

        # Flip horizontally to the inside
        ax3.yaxis.set_ticks_position('right')  # Keep on the left side
        # Turn the ticks and labels towards the inside of the chart
        ax3.tick_params(axis='y', colors='gray')

        # Synchronize ticks and range
        ax3.set_yticks(ax.get_yticks())
        ax3.set_ylim(ax.get_ylim())
        # Convert to depth and set color
        ax3.set_yticklabels(
            [f'{y * 3.77:.1f}' for y in ax.get_yticks()], color='gray')

        # Invert y-axis
        ax.invert_yaxis()
        ax3.invert_yaxis()

    # Add title and axis labels
    plt.title(title)
    if not one_eruption_mode:
        ax.set_xlabel('Eruption')
    if P_T == 'P':
        ax.set_ylabel('Pressure (Kbar)')
    else:
        ax.set_ylabel('Temperature (°C)')

    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # font size to 12
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()

    
    return fig, ax, bps


def violin_plot(data, P_T,  model_list, model_uncertainty=None, eruption_list=None, title=''):
    """
    Plot thermobarometry results as box plots for method comparison.

    Displays the median line and value, as well as the error space (median ± model uncertainty, if provided).
    The upper and lower bars represent the 0th and 100th percentiles.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing thermobarometry results.
        Required columns: 'eruption' (if multiple eruptions), and model result columns.
        Example table:
            id | T_Putirka08 | P_Putirka08 | T_Putirka96 | P_Putirka96 | eruption
            1  |    1000     |     2.5     |    1100      |     3.0     |   OTT
            2  |    1100     |     3.0     |    1200      |     4.0     |   MTT
            3  |    1200     |     4.0     |    1300      |     5.0     |   YTT
    P_T : str
        "P" for pressure or "T" for temperature mode.
    model_list : list of str
        List of model names (column names) to plot.
        Example: ['T_Putirka08', 'P_Putirka08', 'T_Putirka96', 'P_Putirka96']
    model_uncertainty : dict, optional
        Dictionary mapping model names to their uncertainties.
        If None (default), no uncertainty area is shown.
    eruption_list : list of str, optional
        List of eruption names to plot.
        If None (default), plots all data as a single group.
    title : str, optional
        Title of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object.
    ax : matplotlib.axes.Axes
        The main Axes object.
    bps : list
        List of boxplot objects.
    """
    from aims4pt.toolkit_utils import wrap_text
    one_eruption_mode = True if eruption_list is None or len(
        eruption_list) == 1 else False
    num_models = len(model_list)
    num_eruption = len(eruption_list) if eruption_list is not None else 1
    figure_width = max(2* num_models * num_eruption, 12)

    # Set model colors
    model_colors = {}
    for model_name in model_list:
        # C0 - C9: default color cycle
        model_colors[model_name] = f'C{model_list.index(model_name)}'

    # Filter data
    if one_eruption_mode:
        plot_data = data
        plot_data["eruption"] = "All"
    else:
        plot_data = data[(data['eruption'].isin(eruption_list))]

    fig, ax = plt.subplots(figsize=(figure_width, 6), dpi=300)
    pos = 0
    xticks = []
    xticklabels = []
    x2ticks = []
    x2ticklabels = []

    # Record the position of each eruption event to center the label
    eruption_positions = {}

    # Plot box plots for each model in each eruption event
    bps = []
    if one_eruption_mode:
        eruption_list = ['All']
    for eruption in eruption_list:
        plot_data_eruption = plot_data[plot_data['eruption'] == eruption]
        for model_name in model_list:
            
            if model_name not in plot_data_eruption.columns:
                # fake boxplot
                pos += 1
                x2ticks.append(pos)
                x2ticklabels.append(model_name)
                bp = ax.boxplot([], positions=[pos], widths=0.5,
                                patch_artist=True, meanline=False, showfliers=False,
                                boxprops=dict(facecolor='lightgray', color="black"),
                                medianprops=dict(color="black"),
                                whis=[0, 100],
                                )
                bp['boxes'][0].set_alpha(0.5)
                bp['bodies'] = bp['boxes'].copy()  # Ensure 'bodies' is present
                bps.append(bp)

                print(f'Model {model_name} not found in data for eruption {eruption}. Skipping.')
                continue
            y = model_name
            values = plot_data_eruption[y].dropna().values

            if len(values) == 0:
                print(f'No data for {eruption} - {model_name}. Skipping violin plot.')
                continue

            alpha = 0.5 if 'Szy23' in model_name else 1

            color = model_colors[model_name]

            # add a shaded area to indicate the uncertainty of the model
            if model_uncertainty is not None:
                if model_name in model_uncertainty:
                    median_value = np.median(values)
                    ax.fill_between([pos-0.3, pos+0.3], median_value+model_uncertainty[model_name],
                                    median_value-model_uncertainty[model_name], color='gray', alpha=0.3, edgecolor=None)

            bp = ax.violinplot(
                        values,
                        positions=[pos],
                        widths=0.5,
                        showmeans=False,
                        showmedians=True,    
                        showextrema=False,

                    )
            bp['bodies'][0].set_alpha(alpha)
            bp['bodies'][0].set_facecolor(color)
            bp['bodies'][0].set_edgecolor('black')
            bp["cmedians"].set_color("black")
            bp["cmedians"].set_linewidth(2)

            # Display sample size n
            # if model_name == model_list[-1]:
            ax.text(pos, np.min(values) if P_T == 'T' else np.max(values),
                    f'n = {len(values)}', ha='center', va='top', fontsize=12, color="#003C78")
            # Display median value
            median_value = np.median(values)
            ax.text(pos, median_value, f'{median_value:.1f}', ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')

            if eruption not in eruption_positions:
                eruption_positions[eruption] = []
            eruption_positions[eruption].append(pos)

            # Set model name label
            x2ticks.append(pos)
            # x2 label
            model_label = wrap_text(model_name, 15)  # Wrap text to 10 characters
            x2ticklabels.append(model_label)

            # Update position counter
            pos += 1
            bps.append(bp)


        # Add a gap between different eruption events
        pos += 1

    # Set the labels for eruption events, display one and center it
    for eruption_name, positions in eruption_positions.items():
        center_position = sum(positions) / len(positions)  # Calculate the center position
        xticks.append(center_position)
        xticklabels.append(eruption_name)

    # Set the ticks for the main x-axis and secondary x-axis
    if not one_eruption_mode:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, ha='center', fontsize=12)
    else:
        # remove the ticks for the main x-axis
        ax.set_xticks([])
        ax.set_xticklabels([])

    ax2 = ax.twiny()
    ax2.set_xticks(x2ticks)
    ax2.set_xticklabels(x2ticklabels, ha='center', fontsize=12)

    # Align the range of the main axis and the secondary axis
    ax.set_xlim(-0.5, pos - 1.5)
    ax2.set_xlim(ax.get_xlim())

    if P_T == 'P':
        # Add a new left y-axis and set the depth unit
        ax3 = ax.twinx()  # Create a secondary axis sharing the y-axis
        ax3.set_ylabel('Depth (km)', color='gray')

        # Flip horizontally to the inside
        ax3.yaxis.set_ticks_position('right')  # Keep on the left side
        # Turn the ticks and labels towards the inside of the chart
        ax3.tick_params(axis='y', colors='gray')

        # Synchronize ticks and range
        ax3.set_yticks(ax.get_yticks())
        ax3.set_ylim(ax.get_ylim())
        # Convert to depth and set color
        ax3.set_yticklabels(
            [f'{y * 3.77:.1f}' for y in ax.get_yticks()], color='gray')

        # Invert y-axis
        ax.invert_yaxis()
        ax3.invert_yaxis()

    # Add title and axis labels
    plt.title(title)
    if not one_eruption_mode:
        ax.set_xlabel('Eruption')
    if P_T == 'P':
        ax.set_ylabel('Pressure (Kbar)')
    else:
        ax.set_ylabel('Temperature (°C)')

    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # font size to 12
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()

    
    return fig, ax, bps



def T_P_kde_plot(data,  P_T, model_list, model_uncertainty=None, title=''):
    '''
    Plot the kernel density estimation (KDE) of the thermobarometry results.

    Parameters:
        data (DataFrame): DataFrame containing thermobarometry results.
            >>>Example table:
            ```
            id | model_1 | model_2 | model_3
            ---|---------|---------|---------
            1  |    1000 |     2.5 |    1100
            2  |    1100 |     3.0 |    1200
            3  |    1200 |     4.0 |    1300
            ```
        P_T (str): "P" for pressure or "T" for temperature mode.
        model_list (list): List of model names (column names) to plot.
        model_uncertainty (dict): Dictionary mapping model names to their uncertainties.
        title (str): Title of the plot.
    '''
    model_colors = {}
    for model_name in model_list:
    # C0 - C9: default color cycle
        model_colors[model_name] = f'C{model_list.index(model_name)}'
    # Filter data
    plot_data = data.copy()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    record_kde_max = []

    # Plot KDE for each model in each eruption event
    for model_name in model_list:


        values = plot_data[model_name].dropna().values

        alpha = 0.5 if 'Szy23' in model_name else 1
        color = model_colors[model_name]

        # Plot KDE
        kde = sns.kdeplot(values, ax=ax, color=color, shade=False,
                    alpha=alpha, label=model_name, linewidth=2)
        # Record the maximum value of KDE for setting y-axis limit
        kde_max = np.max(kde.get_lines()[-1].get_ydata())
        record_kde_max.append(kde_max)

    # Set the title and axis labels
    plt.title(title)
    ax.set_xlabel('Pressure (Kbar)' if P_T == 'P' else 'Temperature (°C)')
    ax.set_ylabel('Density')  # KDE density on y-axis
    ax.legend(loc='upper right')  # Display legend
    
    # add error bar
    if model_uncertainty is not None:
        # n_models = len(model_list)
        # min_y_ = ax.get_ylim()[0]
        # max_y_ = ax.get_ylim()[1]
        # min_y = 0.15*(max_y_ - min_y_) + min_y_
        # max_y = 0.85*(max_y_ - min_y_) + min_y_
        for i, model_name in enumerate(model_list):
            predicted_values_mean = np.mean(plot_data[model_name].dropna().values)
            model_y = record_kde_max[i]*0.5  # Use the maximum KDE value for the model
            # model_y = i * (max_y - min_y) / n_models + min_y
            ax.errorbar( x= predicted_values_mean, y=model_y,
                 xerr=model_uncertainty[model_name], fmt='o', color=model_colors[model_name], alpha=0.5)

    # Display grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.8)

    return fig, ax



def compare_natural_exp_dataset(natural_df, exp_df, x_y_pairs, T_color=None, P_color=None, ):
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

    # plot Harker diagram n.
    n = len(x_y_pairs)
    if n == 1:
        ncol = 1
    elif n == 2:
        ncol = 2
    else:
        ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(20, 5 * nrow), dpi=150)
    ax = ax.flatten()
    cmap = "YlOrRd"

    for i, (x, y) in enumerate(x_y_pairs):
        ax[i].scatter(natural_df[x], natural_df[y], c='blue', s=40,
                      alpha=0.5, label='natural data', marker="2")
        scatter = None
        if T_color:
            scatter = ax[i].scatter(exp_df[x], exp_df[y], c=exp_df[T_color], s=40,
                                    edgecolors='k', alpha=1, label='experimental data', cmap=cmap)

        if P_color:
            scatter = ax[i].scatter(exp_df[x], exp_df[y], c=exp_df[P_color], s=40,
                                    edgecolors='k', alpha=1, label='experimental data', cmap=cmap)
        else:
            ax[i].scatter(exp_df[x], exp_df[y], c='grey', s=40,
                          edgecolors='k', alpha=1, label='experimental data')

        sns.kdeplot(natural_df, x=x, y=y,
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
        ax[i].legend()
        ax[i].grid(True)

    plt.show()

# Convenience version.


def compare_natural_exp_dataset_v_lazy(natural_df_one_phase, exp_df_one_phase, x_y_pairs, T_color=None, P_color=None, P_T_range=[None, None]):
    '''
    Compare natural and experimental data. Useful when applying a Geothermobarometry method to a dataset.
    Input data with single phase only.

    Parameters:
        natural_df_one_phase (DataFrame):
            DataFrame of natural data with single phase only.
        exp_df_one_phase (DataFrame):
            DataFrame of experimental data with single phase only.
        x_y_pairs (list):
            List of tuples of x and y columns to compare.
        T_color (str):
            Column name for temperature.
            None by default. No colorbar for temperature.
        P_color (str):
            Column name for pressure.
            None by default. No colorbar for pressure.
        P_T_range (list):
            List of two elements. The first element is the lower limit, the second element is the upper limit.
            None by default. No limit for temperature or pressure.
            Example: [1000, 1200] for temperature range 1000-1200 °C.
            Example: [2, 4] for pressure range 2-4 Kbar.
        
    Example:
    >>> refer to this link:
    >>> /files/example_image/compare_natural_exp_dataset_v_lazy.png
    '''
    from aims4pt.utils import normalize_column_names

    natural_df_norm = normalize_column_names(natural_df_one_phase)
    exp_df_norm = normalize_column_names(exp_df_one_phase)

    if P_T_range[0]:
        if T_color:
            exp_df_norm = exp_df_norm[(exp_df_norm[T_color] >= P_T_range[0])]
        if P_color:
            exp_df_norm = exp_df_norm[(exp_df_norm[P_color] >= P_T_range[0])]
    if P_T_range[1]:
        if T_color:
            exp_df_norm = exp_df_norm[(exp_df_norm[T_color] <= P_T_range[1])]
        if P_color:
            exp_df_norm = exp_df_norm[(exp_df_norm[P_color] <= P_T_range[1])]

    compare_natural_exp_dataset(
        natural_df_norm, exp_df_norm, x_y_pairs, T_color=T_color, P_color=P_color)


def plot_P_T_oxides(data_1, y_1_name, data_2=None, y_2_name=None, title=None, oxides=None, sub_title=False, P_T='P', kde=True):
    '''
    Plot P-T-oxides diagram. Compare the oxides in data_1 and data_2.

    Parameters:

        data_1 (DataFrame):
            DataFrame of data 1.
        y_1_name (str):
            Column name of y in data 1.
        data_2 (DataFrame):
            DataFrame of data 2.
            None by default.
        y_2_name (str):
            Column name of y in data 2.
            None by default.
        title (str):
            Title of the plot.
            None by default.
        oxides (list):
            List of oxides to compare.
            None by default.
        sub_title (bool):
            Whether to add sub title.
            False by default.

    Example:
    >>> refer to this link:
    >>> /files/example_image/plot_P_T_oxides.png
    '''

    from aims4pt.utils import filter_oxides
    from aims4pt.visualization.plot_utils import get_subplot_shape
    data_1 = normalize_column_names(data_1)
    print(data_1.columns)

    if data_2 is not None:
        data_2 = normalize_column_names(data_2)
        print(data_2.columns)

    if oxides is None:
        oxides_1 = filter_oxides(data_1.columns)
        oxides_2 = filter_oxides(
            data_2.columns) if data_2 is not None else oxides_1
        # Use the intersection of datasets 1 and 2.
        oxides = list(set(oxides_1).intersection(set(oxides_2)))

    # plot
    n_rows, n_cols = get_subplot_shape(len(oxides))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    for i, ax in enumerate(axes.flat):
        if i >= len(oxides):
            break
        # exp: oxides[i] vs 'P_kbar'
        sns.scatterplot(data=data_1, x=y_1_name, y=oxides[i], ax=ax)
        # kde
        if kde:
            sns.kdeplot(data=data_1, x=y_1_name,
                        y=oxides[i], ax=ax, color="grey")

        if data_2 is not None:
            sns.scatterplot(data=data_2, x=y_2_name, y=oxides[i], ax=ax)
            # sns.kdeplot(data=data_2, x=y_2_name,
            #             y=oxides[i], ax=ax, color="red")

        ax.set_ylabel(oxides[i]+" (wt%)")
        if P_T == 'P':
            ax.set_xlabel('Pressure (Kbar)')
        else:
            ax.set_xlabel('Temperature (°C)')

        # title
        if sub_title:
            ax.set_title(oxides[i], fontsize=14)

    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_error_kde(data, oxides, error_arrays, difference_column="difference", kde_bandwidth = 1, axes=None, suffix=""):
    '''
    Plot the kernel density estimation (KDE) of the oxides in the DataFrame.
    The KDE is plotted for all data and for data within specified error ranges.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data to plot.
        oxides (list): List of oxide names to plot.
        error_arrays (list): List of error ranges to filter the data.
        difference_column (str): Column name for the difference values.
        kde_bandwidth (float): Bandwidth for the KDE.

    Example:
    >>> refer to this link:
    >>> /files/example_image/plot_error_kde.png

    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # get the color map

    if axes is None:
        color_map = sns.color_palette("Blues", as_cmap=True)
    else:
        color_map = sns.color_palette("Reds", as_cmap=True)
    data = data.copy()
    data = normalize_column_names(data)

    if axes is None:
        n_fig = len(oxides)
        n_rows, n_cols = get_subplot_shape(n_fig)
        
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), dpi=200
        )
    else:
        fig = axes[0].figure
    
    axes = axes.flatten()  # Flatten the subplot array for iteration.

    for i, oxide in enumerate(oxides):
        if i >= len(axes):  # Prevent indexing beyond the number of subplots.
            break

        ax = axes[i]  # Get the current subplot.
        data[oxide] = data[oxide].astype(float)

        # Plot the KDE for all data.
        sns.kdeplot(data[oxide], label=f"all data ({suffix}, n={len(data)})"
                    , color="black", ax=ax, bw_adjust=kde_bandwidth)

        for error in error_arrays:
            data_within_error = data[np.abs(data[difference_column]) < error]

            # Plot the KDE for data within the error range.
            color = color_map(error / np.max(error_arrays))
            sns.kdeplot(
                data_within_error[oxide],
                color=color,
                label=f"difference < ±{error} C, n={len(data)}",
                ax=ax,
                bw_adjust=kde_bandwidth,
            )

        if i == 0:
            ax.legend()
        ax.set_title(f"{oxide}")

    # Hide unused subplots.
    for j in range(len(oxides), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.show()

    return fig, axes



def plot_ood_score_vs_deviation(input_cpx, input_meta, model, input_liq = None, high_calculated_deviation_marker = False, petrological_ood_marker = False, T_P_ood_marker = False, ax = None):
    '''
    Plot OOD score vs deviation for a given model and input data.

    Parameters:
        input_cpx : pd.DataFrame
            Input clinopyroxene compositions.
        input_meta : pd.DataFrame
            Metadata containing true or temperature values.
        model : ThermobarometerModel
            The thermobarometer model to use for predictions.
        input_liq : pd.DataFrame, optional
            Input liquid compositions, if required by the model.
        high_calculated_deviation_marker : bool, optional
            Whether to highlight points with high calculated deviation.
        petrological_ood_marker : bool, optional
            Whether to highlight petrologically consistent OOD points.
        T_P_ood_marker : bool, optional
            Whether to highlight points which are OOD in T or P space.
        ax : matplotlib.axes.Axes, optional 
            Axes object to plot on. If None, a new figure and axes will be created.
    Returns:
        fig : matplotlib.figure.Figure
            The Figure object.
        ax_scatter : matplotlib.axes.Axes
            The Axes object with the scatter plot.
    '''
    model_name = model.model_name
    target_col = "P_kbar" if model.T_P == "P" else "T_C"
    print(f"model's key features: {model.key_features}")

    if petrological_ood_marker:
        from aims4pt.data_tools.rocks import get_TAS_rock_types, get_volcanic_rock_series
        from aims4pt.statistic_tools.density_region_analysis import rock_type_check
        input_meta['TAS_rock_type'] = input_liq.apply(get_TAS_rock_types, axis=1)
        input_meta['volcanic_rock_series'] = input_liq.apply(get_volcanic_rock_series, axis=1)
        tas_mask = [rock_type_check(rock_type, model.rock_types, report= False) for rock_type in input_meta['TAS_rock_type']]
        series_mask = [rock_type_check(rock_series, model.volcanic_rock_series, report= False) for rock_series in input_meta['volcanic_rock_series']]
        petrological_mask = [tas and series for tas, series in zip(tas_mask, series_mask)]
    
    predictions = model.predict(input_cpx, input_liq)
    ood_model = getattr(model, "OOD_detector", None)
    if ood_model is None:
        print(f"No OOD detector found for model {model_name}.")
        ood_mask = np.array([False]*len(input_cpx))
        ood_score = np.array([np.nan]*len(input_cpx))
    else:
        ood_score = ood_model.score(input_cpx, input_liq)
        ood_mask = ood_model.is_ood(input_cpx, input_liq)
    deviation = (input_meta[target_col] - predictions).abs()

    if high_calculated_deviation_marker:
        deviation_function= getattr(model, "deviation_function", None)
        if deviation_function is None:
            print(f"No deviation function found for model {model_name}.")
            deviation_mask = np.array([False]*len(input_cpx))
        else:
            estimate_deviation = deviation_function.predict_deviation(input_cpx, input_liq)
            deviation_mask = estimate_deviation >= model.uncertainty
    
    if T_P_ood_marker:
        y_lim = model.y_lim if hasattr(model, 'y_lim') else -np.inf
        y_max = model.y_max if hasattr(model, 'y_max') else np.inf
        T_P_ood_mask = (predictions < y_lim) | (predictions > y_max)
        



    # Pair up scores with deviations first so dropna keeps them aligned
    split_df = pd.DataFrame({"score": ood_score, "deviation": deviation})
    ood_df = split_df.loc[ood_mask].dropna()
    ind_df = split_df.loc[~ood_mask].dropna()
    ood_x, ood_y = ood_df["score"], ood_df["deviation"]
    ind_x, ind_y = ind_df["score"], ind_df["deviation"]

    if petrological_ood_marker:
        petro_mask = pd.Series(petrological_mask, index=split_df.index)
        petro_df = split_df.loc[~petro_mask].dropna()
        petro_ood_x = petro_df["score"]
        petro_ood_y = petro_df["deviation"]


    if high_calculated_deviation_marker:
        deviation_mask_series = pd.Series(deviation_mask, index=split_df.index)
        deviation_ood_df = split_df.loc[deviation_mask_series].dropna()
        deviation_ood_x = deviation_ood_df["score"]
        deviation_ood_y = deviation_ood_df["deviation"]

    if T_P_ood_marker:
        T_P_ood_mask_series = pd.Series(T_P_ood_mask, index=split_df.index)
        T_P_ood_df = split_df.loc[T_P_ood_mask_series].dropna()
        T_P_ood_x = T_P_ood_df["score"]
        T_P_ood_y = T_P_ood_df["deviation"]

    if ax is not None:
        ax_scatter = ax
        fig = ax.figure
    else:   
        fig, ax_scatter = plt.subplots(figsize=(10, 6), dpi=200)

    # Highlight typical deviation ranges
    if len(ind_x) > 0:
        ind_q = np.quantile(ind_y, [0.25, 0.5, 0.75])
        ind_xmin, ind_xmax = ind_x.min(), ind_x.max()
        ax_scatter.fill_between(
            [ind_xmin, ind_xmax],
            [ind_q[0],  ind_q[0]],
            [ind_q[2],  ind_q[2]],
            color="#FF7F0E",
            alpha=0.18,
            zorder=1,
        )
        ax_scatter.hlines(y=ind_y.mean(), xmin=ind_xmin, xmax=ind_xmax,  color="#FF7F0E", linestyle="-", linewidth=1) #, label="In-Distribution Mean Deviation"

    if len(ood_x) > 0:
        ood_q = np.quantile(ood_y, [0.25, 0.5, 0.75])
        ood_xmin, ood_xmax = ood_x.min(), ood_x.max()
        ax_scatter.hlines(y=ood_y.mean(), xmin=ood_xmin, xmax=ood_xmax,  color="#1F77B4", linestyle="-", linewidth=1) #, label="OOD Mean Deviation"
        ax_scatter.fill_between(
            [ood_xmin, ood_xmax],
            [ood_q[0],  ood_q[0]],
            [ood_q[2],  ood_q[2]],
            color="#1F77B4",
            alpha=0.15,
            zorder=1,
        )

    # In-distribution scatter
    ax_scatter.scatter(
        ind_x,
        ind_y,
        s=30,
        alpha=0.5,
        color="#FF7F0E",
        edgecolors="none",
        label="In-distribution",
        zorder=2,
        rasterized=True,
        
    )

    # OOD scatter
    ax_scatter.scatter(
        ood_x,
        ood_y,
        s=30,
        alpha=0.5,
        color="#1F77B4",
        edgecolors="none",
        label="OOD (machine learning)",
        zorder=3,
        rasterized=True,
        
    )

    if T_P_ood_marker:
        ax_scatter.scatter(
            T_P_ood_x,
            T_P_ood_y,
            s=60,
            alpha=0.7,
            color="orange",
            edgecolors="black",
            label=f"OOD (predicted {model.T_P})",
            zorder=4,
            rasterized=True,
            )
        
    if petrological_ood_marker:
        ax_scatter.scatter(
            petro_ood_x,
            petro_ood_y,
            s=50,
            alpha=0.7,
            color="purple",
            marker = 'o',
            edgecolors="black",
            label="OOD (petrological)",
            zorder=5,
            rasterized=True,
            )

    if high_calculated_deviation_marker:
        ax_scatter.scatter(
            deviation_ood_x,
            deviation_ood_y,
            s=50,
            alpha=0.7,
            color="green",
            edgecolors="black",
            label=f"High estimated deviation (D > uncertainty)",
            zorder=6,
            rasterized=True,
            )




    ax_scatter.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_scatter.axhline(model.uncertainty, color="gray", linestyle="--", linewidth=1, label="Model uncertainty")
    ax_scatter.invert_xaxis()
    ax_scatter.set_xlabel("Signed distance to the boundary", fontsize=12)
    ax_scatter.set_ylabel(f"Absolute deviation ({'kbar' if model.T_P == 'P' else '°C'})", fontsize=12)
    ax_scatter.grid(alpha=0.3, linestyle="--")
    ax_scatter.legend(loc="upper right", fontsize=10)
    # ax_scatter.set_ylim(0, 30)
    # ax_scatter.set_ylim(0, 20)
    if ax is None:
        plt.title(f"OOD Score vs Deviation for {model_name}", fontsize=14)
        plt.tight_layout()
        plt.show()
    return fig, ax_scatter


@dataclass
class ThermobarometryWorkflowBundle:
    """
    Container for one workflow family across pressure/temperature and eruption year.

    Parameters
    ----------
    pressure_2006, pressure_2010, temperature_2006, temperature_2010
        Workflow objects that expose ``prediction_df`` and
        ``get_best_model_series()``.

    Notes
    -----
    This class is optional. The plotting helpers below also accept dictionaries
    with equivalent keys such as ``pressure_2006`` / ``temperature_2010`` or a
    nested form like ``{"P": {"2006": ...}, "T": {"2010": ...}}``.
    """

    pressure_2006: Any
    pressure_2010: Any
    temperature_2006: Any
    temperature_2010: Any


def _coerce_workflow_bundle(bundle: Any, label: str) -> ThermobarometryWorkflowBundle:
    if isinstance(bundle, ThermobarometryWorkflowBundle):
        return bundle

    if not isinstance(bundle, Mapping):
        raise TypeError(
            f"{label} must be a ThermobarometryWorkflowBundle or mapping, got {type(bundle)!r}."
        )

    flat_keys = {"pressure_2006", "pressure_2010", "temperature_2006", "temperature_2010"}
    if flat_keys.issubset(bundle):
        return ThermobarometryWorkflowBundle(
            pressure_2006=bundle["pressure_2006"],
            pressure_2010=bundle["pressure_2010"],
            temperature_2006=bundle["temperature_2006"],
            temperature_2010=bundle["temperature_2010"],
        )

    short_keys = {"P_2006", "P_2010", "T_2006", "T_2010"}
    if short_keys.issubset(bundle):
        return ThermobarometryWorkflowBundle(
            pressure_2006=bundle["P_2006"],
            pressure_2010=bundle["P_2010"],
            temperature_2006=bundle["T_2006"],
            temperature_2010=bundle["T_2010"],
        )

    if "P" in bundle and "T" in bundle:
        pressure_map = bundle["P"]
        temperature_map = bundle["T"]
        if not isinstance(pressure_map, Mapping) or not isinstance(temperature_map, Mapping):
            raise TypeError(f"{label}['P'] and {label}['T'] must be mappings.")

        def _pick_year(mapping: Mapping[Any, Any], year: str) -> Any:
            if year in mapping:
                return mapping[year]
            year_int = int(year)
            if year_int in mapping:
                return mapping[year_int]
            raise KeyError(year)

        return ThermobarometryWorkflowBundle(
            pressure_2006=_pick_year(pressure_map, "2006"),
            pressure_2010=_pick_year(pressure_map, "2010"),
            temperature_2006=_pick_year(temperature_map, "2006"),
            temperature_2010=_pick_year(temperature_map, "2010"),
        )

    raise KeyError(
        f"{label} must provide either flat keys {sorted(flat_keys)}, short keys {sorted(short_keys)}, "
        "or nested keys {'P': {'2006', '2010'}, 'T': {'2006', '2010'}}."
    )


def _coerce_phase_column_groups(columns: Mapping[str, Sequence[str]], label: str) -> dict[str, list[str]]:
    if not isinstance(columns, Mapping):
        raise TypeError(f"{label} must be a mapping with 'cpx_only' and 'cpx_liq' keys.")

    required = {"cpx_only", "cpx_liq"}
    missing = required.difference(columns)
    if missing:
        raise KeyError(f"{label} is missing required keys: {sorted(missing)}")

    return {
        "cpx_only": list(columns["cpx_only"]),
        "cpx_liq": list(columns["cpx_liq"]),
    }


def _get_prediction_df_from_workflow(workflow: Any, workflow_label: str) -> pd.DataFrame:
    if not hasattr(workflow, "prediction_df"):
        raise AttributeError(f"{workflow_label} does not expose prediction_df.")
    prediction_df = workflow.prediction_df
    if not isinstance(prediction_df, pd.DataFrame):
        raise TypeError(f"{workflow_label}.prediction_df must be a pandas DataFrame.")
    return prediction_df


def _get_best_model_series_from_workflow(workflow: Any, workflow_label: str) -> pd.Series:
    if not hasattr(workflow, "get_best_model_series"):
        raise AttributeError(f"{workflow_label} does not expose get_best_model_series().")
    best_model_series = workflow.get_best_model_series()
    if not isinstance(best_model_series, pd.Series):
        raise TypeError(f"{workflow_label}.get_best_model_series() must return a pandas Series.")
    return best_model_series


def _extract_plot_arrays(df: pd.DataFrame, columns: Sequence[str], *, df_label: str) -> list[np.ndarray]:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{df_label} is missing required columns: {missing}")
    return [df[col].dropna().to_numpy() for col in columns]


def _build_uncertainty_dict(model_pool: Optional[Sequence[Any]]) -> dict[str, float]:
    if model_pool is None:
        return {}
    return {
        getattr(model, "model_name", str(i)): getattr(model, "uncertainty", np.nan)
        for i, model in enumerate(model_pool)
    }


def _remove_boxplot_outliers(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size <= 3:
        return arr

    q1 = np.nanpercentile(arr, 25)
    q3 = np.nanpercentile(arr, 75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return arr

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return arr[(arr >= lower) & (arr <= upper)]


def _filter_data_groups(data_groups: Sequence[Sequence[float]]) -> list[np.ndarray]:
    return [_remove_boxplot_outliers(values) for values in data_groups]


def _coerce_result_array(values: Any, label: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        series = values
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] == 1:
            series = values.iloc[:, 0]
        elif "prediction" in values.columns:
            series = values["prediction"]
        else:
            raise TypeError(f"{label} must be a Series, array-like, or single-column DataFrame.")
    else:
        try:
            series = pd.Series(np.asarray(values, dtype=float).ravel())
        except Exception as exc:
            raise TypeError(f"{label} must be numeric and array-like.") from exc

    numeric = pd.to_numeric(series, errors="coerce")
    return _remove_boxplot_outliers(numeric.dropna().to_numpy())


def _compute_rank_pcts_from_best(best_model_series: pd.Series) -> list[tuple[str, float]]:
    counts = best_model_series.dropna().astype(str).value_counts()
    total = counts.sum()
    if total == 0:
        return []
    pcts = counts / total * 100.0
    return list(zip(pcts.index.tolist(), pcts.values.tolist()))


def _model_to_index(columns: Sequence[str], model_name: str) -> Optional[int]:
    try:
        return list(columns).index(model_name)
    except ValueError:
        return None


def _pick_models_for_single_eruption(
    rank_pcts: Sequence[tuple[str, float]],
    threshold: float,
) -> list[str]:
    if len(rank_pcts) == 0:
        return []
    rank1_name, rank1_pct = rank_pcts[0]
    if float(rank1_pct) > threshold or len(rank_pcts) == 1:
        return [rank1_name]
    return [rank1_name, rank_pcts[1][0]]


def _selected_rank_set_from_topk(topk: Sequence[tuple[str, float]], threshold: float) -> set[int]:
    if not topk:
        return set()
    if float(topk[0][1]) > threshold or len(topk) == 1:
        return {0}
    return {0, 1}


def _finite_uncertainty(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except Exception:
        return None
    if not np.isfinite(value) or value < 0:
        return None
    return value


def _clean_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _strip_parens(value: Any) -> str:
    text = _clean_str(value)
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text


def _wrap_label(value: Any, width: int = 12, *, allow_word_break: bool = False) -> str:
    return wrap_text(str(value), max_width=width, allow_word_break=allow_word_break)


def _apply_threshold_xtick_rotation(
    ax: plt.Axes,
    raw_labels: Sequence[Any],
    *,
    rotation_threshold: int = 22,
    long_label_fontsize_scale: float = 0.86,
) -> None:
    for tick_label, raw_label in zip(ax.get_xticklabels(), raw_labels):
        tick_label.set_rotation(0)
        tick_label.set_ha("center")
        tick_label.set_va("top")
        if len(str(raw_label)) > rotation_threshold:
            tick_label.set_fontsize(tick_label.get_fontsize() * long_label_fontsize_scale)
        tick_label.set_linespacing(0.95)


def _pick_kind_mapping(mapping: Optional[Mapping[str, Any]], kind: str) -> Optional[Mapping[str, Any]]:
    if mapping is None:
        return None

    direct_value = mapping.get(kind)
    if direct_value is not None:
        return direct_value

    alias = "pressure" if kind == "P" else "temperature"
    alias_value = mapping.get(alias)
    if alias_value is not None:
        return alias_value

    return None


def _coerce_liquid_kind_results(kind: str, liquid_results: Optional[Mapping[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    kind_mapping = _pick_kind_mapping(liquid_results, kind)
    if kind_mapping is None:
        return {}
    if not isinstance(kind_mapping, Mapping):
        raise TypeError(f"liquid_results[{kind!r}] must be a mapping.")

    coerced: dict[str, dict[str, np.ndarray]] = {}
    for year in ("2006", "2010"):
        year_mapping = kind_mapping.get(year, kind_mapping.get(int(year)))
        if year_mapping is None:
            continue
        if not isinstance(year_mapping, Mapping):
            raise TypeError(f"liquid_results[{kind!r}][{year!r}] must be a mapping.")

        phase_results: dict[str, np.ndarray] = {}
        for phase_name in ("glass", "bulk"):
            if phase_name not in year_mapping or year_mapping[phase_name] is None:
                continue
            phase_results[phase_name] = _coerce_result_array(
                year_mapping[phase_name],
                f"liquid_results[{kind!r}][{year!r}][{phase_name!r}]",
            )

        if phase_results:
            coerced[year] = phase_results

    return coerced


def _liquid_prediction_uncertainty(kind: str, phase_name: str) -> float:
    uncertainty_map = {
        ("P", "glass"): 1.66,
        ("P", "bulk"): 1.29,
        ("T", "glass"): 31.9,
        ("T", "bulk"): 26.2,
    }
    return uncertainty_map[(kind, phase_name)]


def _short_model_name(model_name: str) -> str:
    return _clean_str(model_name).split("(")[0].strip()


def _model_axis_label(model_name: str, kind: Optional[str] = None, *, use_model_abbreviations: bool = False) -> str:
    if not use_model_abbreviations:
        return _short_model_name(model_name)
    if kind is None:
        return _short_model_name(model_name)
    try:
        from paper.scripts.constants_illustration import get_model_abbreviation
    except Exception:
        return _short_model_name(model_name)
    label = get_model_abbreviation(model_name, kind)
    if label == model_name:
        return _short_model_name(model_name)
    return label


def _display_phase_type_label(value: Any) -> str:
    label = _clean_str(value)
    key = label.lower().replace("_", "-").replace(" ", "-")
    if key in {"cpx-only", "clinopyroxene-only"}:
        return "Clinopyroxene-only"
    if key in {"cpx-liq", "cpx-liquid", "clinopyroxene-liquid"}:
        return "Clinopyroxene-liquid"
    return label


def _style_bp_item(
    bp: Mapping[str, Any],
    box_idx: int,
    facecolor: str,
    edgecolor: str = "k",
    lw: float = 1.2,
    alpha: float = 1.0,
    median_color: Optional[str] = None,
    median_lw: Optional[float] = None,
) -> None:
    box = bp["boxes"][box_idx]
    box.set_facecolor(facecolor)
    box.set_alpha(alpha)
    box.set_edgecolor(edgecolor)
    box.set_linewidth(lw)
    box.set_zorder(3)

    median = bp["medians"][box_idx]
    median.set_color(median_color if median_color is not None else edgecolor)
    median.set_linewidth(median_lw if median_lw is not None else lw)
    median.set_zorder(4)

    whisker_low = bp["whiskers"][2 * box_idx]
    whisker_high = bp["whiskers"][2 * box_idx + 1]
    cap_low = bp["caps"][2 * box_idx]
    cap_high = bp["caps"][2 * box_idx + 1]
    for obj in (whisker_low, whisker_high, cap_low, cap_high):
        obj.set_color(edgecolor)
        obj.set_linewidth(lw)
        obj.set_zorder(4)


def _draw_violin(
    ax: plt.Axes,
    data_list: Sequence[Sequence[float]],
    positions: Sequence[float],
    *,
    widths: float = 0.28,
    facecolor: str = "blue",
    edgecolor: str = "k",
    lw: float = 1.2,
    alpha: float = 1.0,
    bw_method: float = 0.25,
) -> Mapping[str, Any]:
    vp = ax.violinplot(
        data_list,
        positions=positions,
        widths=widths,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        bw_method=bw_method,
    )
    for body in vp["bodies"]:
        body.set_facecolor(facecolor)
        body.set_edgecolor(edgecolor)
        body.set_linewidth(lw)
        body.set_alpha(alpha)
        body.set_zorder(3)

    if "cmedians" in vp:
        vp["cmedians"].set_color("k")
        vp["cmedians"].set_linewidth(max(lw, 1.2))
        vp["cmedians"].set_zorder(4)
    return vp


def _add_vertical_uncertainty_band(
    ax: plt.Axes,
    x_center: float,
    values: Sequence[float],
    uncertainty: Any,
    width: float,
    color: str,
    *,
    alpha: float = 0.16,
    zorder: float = 2,
) -> None:
    finite_uncertainty = _finite_uncertainty(uncertainty)
    clean_values = _remove_boxplot_outliers(values)
    if finite_uncertainty is None or clean_values.size == 0:
        return

    median = float(np.nanmedian(clean_values))
    x0 = float(x_center) - width / 2.0
    x1 = float(x_center) + width / 2.0
    y0 = median - finite_uncertainty
    y1 = median + finite_uncertainty
    ax.fill_between([x0, x1], [y0, y0], [y1, y1], color=color, alpha=alpha, zorder=zorder, linewidth=0)
    ax.hlines(median, x0, x1, color=color, lw=1.8, zorder=zorder + 0.2)


def _add_vertical_uncertainty_band_from_median(
    ax: plt.Axes,
    x_center: float,
    median: float,
    uncertainty: Any,
    width: float,
    color: str,
    *,
    alpha: float = 0.16,
    zorder: float = 2,
    median_marker: str = "line",
) -> None:
    finite_uncertainty = _finite_uncertainty(uncertainty)
    if finite_uncertainty is None or median is None or not np.isfinite(median):
        return

    x0 = float(x_center) - width / 2.0
    x1 = float(x_center) + width / 2.0
    y0 = float(median) - finite_uncertainty
    y1 = float(median) + finite_uncertainty
    ax.fill_between([x0, x1], [y0, y0], [y1, y1], color=color, alpha=alpha, zorder=zorder, linewidth=0)
    if median_marker == "dot":
        ax.scatter([float(x_center)], [float(median)], s=22, color=color, zorder=zorder + 0.3)
    else:
        ax.hlines(float(median), x0 + 0.28 * width, x1 - 0.28 * width, color=color, lw=1.4, zorder=zorder + 0.2)


def _add_subgroup_background(
    ax: plt.Axes,
    x_center: float,
    width: float,
    color: str,
    *,
    alpha: float = 0.12,
    zorder: float = 1,
) -> None:
    x0 = float(x_center) - width / 2.0
    x1 = float(x_center) + width / 2.0
    ax.axvspan(x0, x1, color=color, alpha=alpha, zorder=zorder)


def _style_selected_violin(
    vp: Mapping[str, Any],
    body_idx: int,
    *,
    edgecolor: str = "k",
    lw: float = 3.0,
    linestyle: str = "--",
) -> None:
    body = vp["bodies"][body_idx]
    body.set_edgecolor(edgecolor)
    body.set_linewidth(lw)
    body.set_linestyle(linestyle)
    body.set_zorder(4.2)


def _redraw_violin_median(
    ax: plt.Axes,
    x_center: float,
    values: Sequence[float],
    width: float,
    *,
    color: str = "k",
    lw: float = 1.8,
    zorder: float = 4.6,
) -> None:
    clean_values = _remove_boxplot_outliers(values)
    if clean_values.size == 0:
        return
    median = float(np.nanmedian(clean_values))
    half_width = 0.24 * width
    ax.hlines(median, float(x_center) - half_width, float(x_center) + half_width, color=color, lw=lw, zorder=zorder)


def _annotate_above_data(
    ax: plt.Axes,
    x: float,
    data: Sequence[float],
    text: str,
    color: str,
    *,
    y_pad_frac: float = 0.04,
    fontsize: float = 14,
    va: str = "bottom",
    y_shift: float = 0.0,
) -> None:
    clean_values = np.asarray(data, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size == 0:
        return
    y_top = float(np.nanmax(clean_values))
    y_min, y_max = ax.get_ylim()
    y_pad = (y_max - y_min) * y_pad_frac
    ax.text(
        float(x),
        y_top + y_pad + y_shift,
        text,
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        ha="center",
        va=va,
        zorder=10,
    )


def _boxplot_whisker_bounds(data: Sequence[float], whis: float = 1.5) -> Optional[tuple[float, float]]:
    clean_values = np.asarray(data, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size == 0:
        return None
    if clean_values.size <= 3:
        return float(np.nanmin(clean_values)), float(np.nanmax(clean_values))

    q1 = float(np.nanpercentile(clean_values, 25))
    q3 = float(np.nanpercentile(clean_values, 75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return float(np.nanmin(clean_values)), float(np.nanmax(clean_values))

    lower_fence = q1 - whis * iqr
    upper_fence = q3 + whis * iqr
    lower_whisker = float(np.nanmin(clean_values[clean_values >= lower_fence]))
    upper_whisker = float(np.nanmax(clean_values[clean_values <= upper_fence]))
    return lower_whisker, upper_whisker


def _annotate_above_boxplot_whisker(
    ax: plt.Axes,
    x: float,
    data: Sequence[float],
    text: str,
    color: str,
    *,
    y_pad_frac: float = 0.04,
    fontsize: float = 14,
    va: str = "bottom",
    y_shift: float = 0.0,
    whisker: str = "upper",
) -> None:
    bounds = _boxplot_whisker_bounds(data)
    if bounds is None:
        return
    lower_whisker, upper_whisker = bounds
    y_anchor = lower_whisker if whisker == "lower" else upper_whisker
    y_min, y_max = ax.get_ylim()
    y_pad = (y_max - y_min) * y_pad_frac
    ax.text(
        float(x),
        y_anchor + y_pad + y_shift,
        text,
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        ha="center",
        va=va,
        zorder=10,
    )


def _annotate_grouped_box_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: float = 11.5,
    color: str = "0.20",
) -> None:
    ax.text(
        float(x),
        float(y),
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        zorder=6,
        clip_on=True,
    )


def _apply_pressure_depth_axes(
    ax: plt.Axes,
    *,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    pressure_ylim: Optional[tuple[float, float]] = None,
    pressure_ticks: Optional[Sequence[float]] = None,
    depth_tick_step: float = 5,
    depth_max: float = 37,
    label_fontsize: float = MANUSCRIPT_Y_AXIS_LABEL_SIZE,
    tick_labelsize: float = MANUSCRIPT_TICK_LABEL_SIZE,
    depth_label_fontsize: Optional[float] = None,
    depth_tick_labelsize: Optional[float] = None,
) -> plt.Axes:
    depth_label_fontsize = label_fontsize if depth_label_fontsize is None else depth_label_fontsize
    depth_tick_labelsize = tick_labelsize if depth_tick_labelsize is None else depth_tick_labelsize

    ax.set_ylabel("Pressure (kbar)", fontsize=label_fontsize)
    if pressure_ylim is not None:
        ax.set_ylim(*pressure_ylim)
    if pressure_ticks is not None:
        ax.set_yticks(list(pressure_ticks))
    ax.invert_yaxis()

    depth_ticks_km = np.arange(0, depth_max + depth_tick_step, depth_tick_step)
    ax_depth = ax.secondary_yaxis(
        "right",
        functions=(
            lambda pressure_kbar: _pressure_to_depth_km(
                pressure_kbar,
                densities_kg_m3=densities_kg_m3,
                layer_boundaries_km=layer_boundaries_km,
            ),
            lambda depth_km: _depth_to_pressure_kbar(
                depth_km,
                densities_kg_m3=densities_kg_m3,
                layer_boundaries_km=layer_boundaries_km,
            ),
        ),
    )
    ax_depth.yaxis.set_major_locator(FixedLocator(depth_ticks_km))
    ax_depth.set_yticklabels([f"{int(d)}" for d in depth_ticks_km])
    ax_depth.set_ylabel("Depth (km)", rotation=-90, va="bottom", labelpad=10, fontsize=depth_label_fontsize)

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_depth.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", which="minor", length=3, width=0.8)
    ax.tick_params(axis="y", which="major", length=6, width=1.0, labelsize=tick_labelsize)
    ax_depth.tick_params(axis="y", which="minor", length=0)
    ax_depth.tick_params(axis="y", which="major", length=6, width=1.0, labelsize=depth_tick_labelsize)
    return ax_depth


def _validate_multilayer_depth_conversion(
    densities_kg_m3: Optional[Sequence[float]],
    layer_boundaries_km: Optional[Sequence[float]],
) -> None:
    if densities_kg_m3 is None or layer_boundaries_km is None:
        raise ValueError(
            "densities_kg_m3 and layer_boundaries_km must be provided together."
        )


def _pressure_to_depth_km(
    pressure_kbar: Any,
    *,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
) -> Any:
    _validate_multilayer_depth_conversion(densities_kg_m3, layer_boundaries_km)
    from aims4pt.data_tools.crust import (
        DEFAULT_G,
        KBAR_TO_PA,
        M_PER_KM,
        kbar_to_km_multilayer,
    )

    pressure_arr = np.asarray(pressure_kbar, dtype=float)
    depth_km = np.empty_like(pressure_arr, dtype=float)
    positive_mask = pressure_arr >= 0

    if np.any(positive_mask):
        depth_km[positive_mask] = kbar_to_km_multilayer(
            pressure_arr[positive_mask],
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
        )
    if np.any(~positive_mask):
        density_kg_m3 = float(np.asarray(densities_kg_m3, dtype=float)[0])
        depth_km[~positive_mask] = (
            pressure_arr[~positive_mask] * KBAR_TO_PA / (density_kg_m3 * DEFAULT_G) / M_PER_KM
        )

    if np.isscalar(pressure_kbar):
        return float(depth_km)
    return depth_km


def _depth_to_pressure_kbar(
    depth_km: Any,
    *,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
) -> Any:
    _validate_multilayer_depth_conversion(densities_kg_m3, layer_boundaries_km)
    from aims4pt.data_tools.crust import (
        DEFAULT_G,
        KBAR_TO_PA,
        M_PER_KM,
        km_to_kbar_multilayer,
    )

    depth_arr = np.asarray(depth_km, dtype=float)
    pressure_kbar = np.empty_like(depth_arr, dtype=float)
    positive_mask = depth_arr >= 0

    if np.any(positive_mask):
        pressure_kbar[positive_mask] = km_to_kbar_multilayer(
            depth_arr[positive_mask],
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
        )
    if np.any(~positive_mask):
        density_kg_m3 = float(np.asarray(densities_kg_m3, dtype=float)[0])
        pressure_kbar[~positive_mask] = (
            density_kg_m3 * DEFAULT_G * depth_arr[~positive_mask] * M_PER_KM / KBAR_TO_PA
        )

    if np.isscalar(depth_km):
        return float(pressure_kbar)
    return pressure_kbar


def _build_ranked_thermobarometry_state(
    cpx_only_workflows: Any,
    cpx_liq_workflows: Any,
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    liquid_results: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    cpx_only_bundle = _coerce_workflow_bundle(cpx_only_workflows, "cpx_only_workflows")
    cpx_liq_bundle = _coerce_workflow_bundle(cpx_liq_workflows, "cpx_liq_workflows")
    pressure_column_groups = _coerce_phase_column_groups(pressure_columns, "pressure_columns")
    temperature_column_groups = _coerce_phase_column_groups(temperature_columns, "temperature_columns")

    pressure_results = {
        "2006": {
            "cpx_only": _get_prediction_df_from_workflow(cpx_only_bundle.pressure_2006, "cpx_only_workflows.pressure_2006"),
            "cpx_liq": _get_prediction_df_from_workflow(cpx_liq_bundle.pressure_2006, "cpx_liq_workflows.pressure_2006"),
        },
        "2010": {
            "cpx_only": _get_prediction_df_from_workflow(cpx_only_bundle.pressure_2010, "cpx_only_workflows.pressure_2010"),
            "cpx_liq": _get_prediction_df_from_workflow(cpx_liq_bundle.pressure_2010, "cpx_liq_workflows.pressure_2010"),
        },
    }
    temperature_results = {
        "2006": {
            "cpx_only": _get_prediction_df_from_workflow(cpx_only_bundle.temperature_2006, "cpx_only_workflows.temperature_2006"),
            "cpx_liq": _get_prediction_df_from_workflow(cpx_liq_bundle.temperature_2006, "cpx_liq_workflows.temperature_2006"),
        },
        "2010": {
            "cpx_only": _get_prediction_df_from_workflow(cpx_only_bundle.temperature_2010, "cpx_only_workflows.temperature_2010"),
            "cpx_liq": _get_prediction_df_from_workflow(cpx_liq_bundle.temperature_2010, "cpx_liq_workflows.temperature_2010"),
        },
    }

    pressure_ranks = {
        "2006": {
            "cpx_only": _compute_rank_pcts_from_best(
                _get_best_model_series_from_workflow(cpx_only_bundle.pressure_2006, "cpx_only_workflows.pressure_2006")
            ),
            "cpx_liq": _compute_rank_pcts_from_best(
                _get_best_model_series_from_workflow(cpx_liq_bundle.pressure_2006, "cpx_liq_workflows.pressure_2006")
            ),
        },
        "2010": {
            "cpx_only": _compute_rank_pcts_from_best(
                _get_best_model_series_from_workflow(cpx_only_bundle.pressure_2010, "cpx_only_workflows.pressure_2010")
            ),
            "cpx_liq": _compute_rank_pcts_from_best(
                _get_best_model_series_from_workflow(cpx_liq_bundle.pressure_2010, "cpx_liq_workflows.pressure_2010")
            ),
        },
    }
    temperature_ranks = {
        "2006": {
            "cpx_only": _compute_rank_pcts_from_best(
                _get_best_model_series_from_workflow(cpx_only_bundle.temperature_2006, "cpx_only_workflows.temperature_2006")
            ),
            "cpx_liq": _compute_rank_pcts_from_best(
                _get_best_model_series_from_workflow(cpx_liq_bundle.temperature_2006, "cpx_liq_workflows.temperature_2006")
            ),
        },
        "2010": {
            "cpx_only": _compute_rank_pcts_from_best(
                _get_best_model_series_from_workflow(cpx_only_bundle.temperature_2010, "cpx_only_workflows.temperature_2010")
            ),
            "cpx_liq": _compute_rank_pcts_from_best(
                _get_best_model_series_from_workflow(cpx_liq_bundle.temperature_2010, "cpx_liq_workflows.temperature_2010")
            ),
        },
    }

    state = {
        "P": {
            "columns": pressure_column_groups,
            "columns_all": pressure_column_groups["cpx_only"] + pressure_column_groups["cpx_liq"],
            "split_idx": len(pressure_column_groups["cpx_only"]),
            "results": pressure_results,
            "ranks": pressure_ranks,
            "uncertainty": _build_uncertainty_dict(pressure_model_pool),
            "liquid_results": _coerce_liquid_kind_results("P", liquid_results),
        },
        "T": {
            "columns": temperature_column_groups,
            "columns_all": temperature_column_groups["cpx_only"] + temperature_column_groups["cpx_liq"],
            "split_idx": len(temperature_column_groups["cpx_only"]),
            "results": temperature_results,
            "ranks": temperature_ranks,
            "uncertainty": _build_uncertainty_dict(temperature_model_pool),
            "liquid_results": _coerce_liquid_kind_results("T", liquid_results),
        },
    }

    for kind in ("P", "T"):
        kind_state = state[kind]
        kind_state["data"] = {}
        for year in ("2006", "2010"):
            data_groups = []
            for phase_type in ("cpx_only", "cpx_liq"):
                df_label = f"{kind} results {year} {phase_type}"
                df = kind_state["results"][year][phase_type]
                cols = kind_state["columns"][phase_type]
                data_groups.extend(_extract_plot_arrays(df, cols, df_label=df_label))
            kind_state["data"][year] = _filter_data_groups(data_groups)

    return state


def _plot_ranked_this_study_kind_panel(
    ax: plt.Axes,
    kind: str,
    state: Mapping[str, Any],
    *,
    selection_threshold: float,
    pressure_ylim: Optional[tuple[float, float]] = None,
    pressure_ticks: Optional[Sequence[float]] = None,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    depth_tick_step: float = 5,
    depth_max: float = 37,
    use_model_abbreviations: bool = False,
    axis_label_fontsize: float = MANUSCRIPT_AXIS_LABEL_SIZE,
    y_axis_label_fontsize: float = MANUSCRIPT_Y_AXIS_LABEL_SIZE,
    tick_labelsize: float = MANUSCRIPT_TICK_LABEL_SIZE,
    model_tick_labelsize: float = MANUSCRIPT_MODEL_TICK_LABEL_SIZE,
    group_label_fontsize: float = MANUSCRIPT_GROUP_LABEL_SIZE,
    annotation_fontsize: float = MANUSCRIPT_ANNOTATION_SIZE,
) -> None:
    kind_state = state[kind]
    columns_all = kind_state["columns_all"]
    split_idx = kind_state["split_idx"]
    n_cols = len(columns_all)
    positions = np.arange(1, n_cols + 1)
    offset = 0.14
    positions_06 = positions - offset
    positions_10 = positions + offset

    violin_width = 0.28
    edge_lw_default = 1.2
    edge_lw_selected = 3.0
    uncertainty_band_width = violin_width * 0.72
    color_06 = "blue"
    color_10 = "red"
    highlight_06 = "#20b8c5"
    highlight_10 = "#f28e2b"

    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=3, width=0.8)
    ax.tick_params(axis="y", which="major", length=6, width=1.0, labelsize=tick_labelsize)
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, color="0.85", zorder=0)
    ax.grid(True, which="minor", axis="y", linestyle="-", linewidth=0.5, color="0.92", zorder=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            wrap_text(
                _model_axis_label(col, kind, use_model_abbreviations=use_model_abbreviations),
                max_width=12,
            )
            for col in columns_all
        ],
        rotation=0,
        ha="center",
        fontsize=model_tick_labelsize if use_model_abbreviations else tick_labelsize,
    )
    _apply_threshold_xtick_rotation(
        ax,
        [_model_axis_label(col, kind, use_model_abbreviations=use_model_abbreviations) for col in columns_all],
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="x", which="major", labelsize=model_tick_labelsize if use_model_abbreviations else tick_labelsize)

    ax.axvline(split_idx + 0.5, color="0.6", linewidth=1.0, zorder=1)
    ax.text(split_idx / 2 + 0.5, 1.025, _display_phase_type_label("cpx_only"), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=group_label_fontsize, fontweight="bold")
    ax.text((split_idx + n_cols) / 2 + 0.5, 1.025, _display_phase_type_label("cpx_liq"), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=group_label_fontsize, fontweight="bold")

    vp06 = _draw_violin(
        ax,
        kind_state["data"]["2006"],
        positions_06,
        widths=violin_width,
        facecolor=color_06,
        edgecolor="k",
        lw=edge_lw_default,
        alpha=1.0,
        bw_method=0.25,
    )
    vp10 = _draw_violin(
        ax,
        kind_state["data"]["2010"],
        positions_10,
        widths=violin_width,
        facecolor=color_10,
        edgecolor="k",
        lw=edge_lw_default,
        alpha=1.0,
        bw_method=0.25,
    )

    def _selected_index_set(rank_pcts: Sequence[tuple[str, float]], phase_type: str, idx_shift: int) -> set[int]:
        selected = set()
        selected_ranks = _selected_rank_set_from_topk(rank_pcts, threshold=selection_threshold)
        cols_local = kind_state["columns"][phase_type]
        for rank_i, (model_name, _) in enumerate(rank_pcts):
            if rank_i not in selected_ranks:
                continue
            idx_local = _model_to_index(cols_local, model_name)
            if idx_local is not None:
                selected.add(idx_shift + idx_local)
        return selected

    selected_06 = (
        _selected_index_set(kind_state["ranks"]["2006"]["cpx_only"], "cpx_only", 0)
        | _selected_index_set(kind_state["ranks"]["2006"]["cpx_liq"], "cpx_liq", split_idx)
    )
    selected_10 = (
        _selected_index_set(kind_state["ranks"]["2010"]["cpx_only"], "cpx_only", 0)
        | _selected_index_set(kind_state["ranks"]["2010"]["cpx_liq"], "cpx_liq", split_idx)
    )

    def _draw_all_uncertainty(positions_used: np.ndarray, data_all: Sequence[np.ndarray], band_color: str) -> None:
        for idx, model_name in enumerate(columns_all):
            _add_vertical_uncertainty_band(
                ax,
                positions_used[idx],
                data_all[idx],
                kind_state["uncertainty"].get(model_name),
                uncertainty_band_width,
                band_color,
                alpha=1.0,
                zorder=2.15,
            )

    def _annotate_all_pcts(
        rank_pcts: Sequence[tuple[str, float]],
        phase_type: str,
        idx_shift: int,
        positions_used: np.ndarray,
        data_all: Sequence[np.ndarray],
        selected_indices: set[int],
        selected_color: str,
    ) -> None:
        rank_pct_map = {model_name: pct for model_name, pct in rank_pcts}
        y_pad_frac = 0.06 if kind == "P" else 0.03
        y_shift = -0.12 if kind == "P" else 0.0
        for idx_local, model_name in enumerate(kind_state["columns"][phase_type]):
            pct = rank_pct_map.get(model_name)
            if pct is None:
                continue
            idx = idx_shift + idx_local
            is_selected = idx in selected_indices
            _annotate_above_data(
                ax,
                positions_used[idx],
                data_all[idx],
                f"{int(round(pct))}%",
                selected_color if is_selected else "k",
                y_pad_frac=y_pad_frac,
                fontsize=annotation_fontsize + 1 if is_selected else annotation_fontsize,
                y_shift=y_shift,
            )

    def _highlight_selected(
        vp: Mapping[str, Any],
        rank_pcts: Sequence[tuple[str, float]],
        phase_type: str,
        idx_shift: int,
        positions_used: np.ndarray,
        data_all: Sequence[np.ndarray],
        bg_color: str,
    ) -> None:
        selected_ranks = _selected_rank_set_from_topk(rank_pcts, threshold=selection_threshold)
        for rank_i, (model_name, _) in enumerate(rank_pcts):
            if rank_i not in selected_ranks:
                continue
            idx_local = _model_to_index(kind_state["columns"][phase_type], model_name)
            if idx_local is None:
                continue
            idx = idx_shift + idx_local
            _add_subgroup_background(ax, positions_used[idx], violin_width, bg_color, alpha=0.18, zorder=1.5)
            _style_selected_violin(vp, idx, edgecolor="k", lw=edge_lw_selected, linestyle="--")
            _redraw_violin_median(ax, positions_used[idx], data_all[idx], violin_width, color="k", lw=2.0, zorder=4.8)

    _draw_all_uncertainty(positions_06, kind_state["data"]["2006"], "lightgray")
    _draw_all_uncertainty(positions_10, kind_state["data"]["2010"], "lightgray")

    _annotate_all_pcts(
        kind_state["ranks"]["2006"]["cpx_only"], "cpx_only", 0, positions_06, kind_state["data"]["2006"], selected_06, color_06
    )
    _annotate_all_pcts(
        kind_state["ranks"]["2006"]["cpx_liq"], "cpx_liq", split_idx, positions_06, kind_state["data"]["2006"], selected_06, color_06
    )
    _annotate_all_pcts(
        kind_state["ranks"]["2010"]["cpx_only"], "cpx_only", 0, positions_10, kind_state["data"]["2010"], selected_10, color_10
    )
    _annotate_all_pcts(
        kind_state["ranks"]["2010"]["cpx_liq"], "cpx_liq", split_idx, positions_10, kind_state["data"]["2010"], selected_10, color_10
    )

    _highlight_selected(
        vp06, kind_state["ranks"]["2006"]["cpx_only"], "cpx_only", 0, positions_06, kind_state["data"]["2006"], highlight_06
    )
    _highlight_selected(
        vp06, kind_state["ranks"]["2006"]["cpx_liq"], "cpx_liq", split_idx, positions_06, kind_state["data"]["2006"], highlight_06
    )
    _highlight_selected(
        vp10, kind_state["ranks"]["2010"]["cpx_only"], "cpx_only", 0, positions_10, kind_state["data"]["2010"], highlight_10
    )
    _highlight_selected(
        vp10, kind_state["ranks"]["2010"]["cpx_liq"], "cpx_liq", split_idx, positions_10, kind_state["data"]["2010"], highlight_10
    )

    if kind == "P":
        _apply_pressure_depth_axes(
            ax,
            pressure_ylim=pressure_ylim,
            pressure_ticks=pressure_ticks,
            depth_tick_step=depth_tick_step,
            depth_max=depth_max,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
            label_fontsize=y_axis_label_fontsize,
            tick_labelsize=tick_labelsize,
        )
    else:
        ax.set_ylabel("Temperature (°C)", fontsize=y_axis_label_fontsize)
        ax.tick_params(axis="y", labelsize=tick_labelsize)
    ax.xaxis.label.set_size(axis_label_fontsize)


def _load_literature_table(data_or_path: Any, label: str) -> pd.DataFrame:
    if isinstance(data_or_path, pd.DataFrame):
        df = data_or_path.copy()
    else:
        df = pd.read_excel(Path(data_or_path))

    if "plot" in df.columns:
        plot_mask = df["plot"].fillna(False).astype(bool)
        df = df.loc[plot_mask].reset_index(drop=True)
    return df


def _build_method_id(row: pd.Series) -> tuple[str, str, str, str, str]:
    return (
        _clean_str(row.get("category")),
        _clean_str(row.get("type")),
        _strip_parens(row.get("cite")),
        _clean_str(row.get("models")),
        _clean_str(row.get("thermobatometer")),
    )


def _make_lit_xtick_label(row: pd.Series) -> str:
    cite = _strip_parens(row.get("cite"))
    if cite:
        return cite
    thermobarometer = _clean_str(row.get("thermobatometer"))
    if thermobarometer:
        return thermobarometer
    models = _clean_str(row.get("models"))
    if models:
        return models
    return _clean_str(row.get("type")) or "unknown"


def _extract_numeric_ranges(
    row: pd.Series,
    keys: Sequence[tuple[str, str]],
) -> list[tuple[float, float]]:
    ranges = []
    for key_min, key_max in keys:
        value_min = row.get(key_min)
        value_max = row.get(key_max)
        if pd.isna(value_min) or pd.isna(value_max):
            continue
        try:
            ranges.append((float(value_min), float(value_max)))
        except Exception:
            pass
    return ranges


def _pressure_range_keys() -> list[tuple[str, str]]:
    return [
        ("min kbar", "max kbar"),
        ("min kbar2", "max kbar2"),
        ("min kbar3", "max kbar3"),
        ("min kbar (or equivalent)", "max kbar (or equivalent)"),
        ("min kbar (or equivalent)2", "max kbar (or equivalent)2"),
        ("min kbar (or equivalent)3", "max kbar (or equivalent)3"),
    ]


def _depth_range_keys() -> list[tuple[str, str]]:
    return [
        ("min km", "max km"),
        ("min km2", "max km2"),
        ("min km3", "max km3"),
        ("min depth (km)", "max depth (km)"),
        ("min depth (km)2", "max depth (km)2"),
        ("min depth (km)3", "max depth (km)3"),
    ]


def _first_valid_row_value(row: pd.Series, keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        return value
    return np.nan


def _coerce_literature_pressure_source(literature_pressure_source: str) -> str:
    source = _clean_str(literature_pressure_source).lower()
    if source not in {"pressure", "depth", "auto"}:
        raise ValueError(
            "literature_pressure_source must be 'pressure', 'depth', or 'auto'."
        )
    return source


def _convert_depth_range_to_pressure_range(
    depth_range: tuple[float, float],
    *,
    densities_kg_m3: Optional[Sequence[float]],
    layer_boundaries_km: Optional[Sequence[float]],
) -> tuple[float, float]:
    pressures = np.asarray(
        _depth_to_pressure_kbar(
            [depth_range[0], depth_range[1]],
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
        ),
        dtype=float,
    )
    return float(np.nanmin(pressures)), float(np.nanmax(pressures))


def _convert_depth_uncertainty_to_pressure(
    depth_range: tuple[float, float],
    uncertainty_depth_km: Any,
    *,
    densities_kg_m3: Optional[Sequence[float]],
    layer_boundaries_km: Optional[Sequence[float]],
) -> float:
    if pd.isna(uncertainty_depth_km):
        return np.nan
    try:
        uncertainty_depth_km = float(uncertainty_depth_km)
    except Exception:
        return np.nan
    if uncertainty_depth_km < 0:
        return np.nan
    midpoint_depth_km = 0.5 * (float(depth_range[0]) + float(depth_range[1]))
    lower_depth_km = max(0.0, midpoint_depth_km - uncertainty_depth_km)
    upper_depth_km = midpoint_depth_km + uncertainty_depth_km
    lower_pressure, upper_pressure = _convert_depth_range_to_pressure_range(
        (lower_depth_km, upper_depth_km),
        densities_kg_m3=densities_kg_m3,
        layer_boundaries_km=layer_boundaries_km,
    )
    return 0.5 * abs(upper_pressure - lower_pressure)


def _extract_ranges_pressure(
    row: pd.Series,
    *,
    literature_pressure_source: str = "pressure",
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
) -> tuple[list[tuple[float, float]], list[float]]:
    source = _coerce_literature_pressure_source(literature_pressure_source)

    if source in {"pressure", "auto"}:
        pressure_ranges = _extract_numeric_ranges(row, _pressure_range_keys())
        if pressure_ranges or source == "pressure":
            pressure_uncertainty = _first_valid_row_value(
                row,
                ("uncertainty kbar", "uncertainty"),
            )
            return pressure_ranges, [pressure_uncertainty] * len(pressure_ranges)

    depth_ranges = _extract_numeric_ranges(row, _depth_range_keys())
    depth_uncertainty = _first_valid_row_value(row, ("uncertainty km", "uncertainty"))
    pressure_ranges = [
        _convert_depth_range_to_pressure_range(
            depth_range,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
        )
        for depth_range in depth_ranges
    ]
    pressure_uncertainties = [
        _convert_depth_uncertainty_to_pressure(
            depth_range,
            depth_uncertainty,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
        )
        for depth_range in depth_ranges
    ]
    return pressure_ranges, pressure_uncertainties


def _extract_ranges_temperature(row: pd.Series) -> list[tuple[float, float]]:
    value_min = row.get("min")
    value_max = row.get("max")
    if pd.isna(value_min) or pd.isna(value_max):
        return []
    try:
        return [(float(value_min), float(value_max))]
    except Exception:
        return []


def _eruption_color(eruption: str) -> str:
    eruption_lower = _clean_str(eruption).lower()
    if "2006" in eruption_lower and "2010" in eruption_lower:
        return "purple"
    if "2006&2010" in eruption_lower or "2006 & 2010" in eruption_lower:
        return "purple"
    if "2006" in eruption_lower:
        return "blue"
    if "2010" in eruption_lower:
        return "red"
    return "0.2"


def _eruption_slot_offset(eruption: str, delta: float = 0.16) -> float:
    eruption_lower = _clean_str(eruption).lower()
    if ("2006" in eruption_lower) and ("2010" in eruption_lower):
        return 0.0
    if "2006" in eruption_lower:
        return -delta
    if "2010" in eruption_lower:
        return delta
    return 0.0


def _add_category_and_type_bands(
    ax: plt.Axes,
    methods: Sequence[Mapping[str, Any]],
    x_positions: np.ndarray,
    *,
    category_fontsize: float = MANUSCRIPT_GROUP_LABEL_SIZE,
    type_fontsize: float = MANUSCRIPT_ANNOTATION_SIZE,
    category_y: float = 1.04,
    type_y: float = 0.985,
) -> None:
    category_ranges = []
    start = 0
    for i in range(1, len(methods) + 1):
        if i == len(methods) or methods[i]["category"] != methods[start]["category"]:
            category_ranges.append((methods[start]["category"], start, i - 1))
            start = i

    for j, (category, i0, i1) in enumerate(category_ranges):
        x0 = x_positions[i0] - 0.5
        x1 = x_positions[i1] + 0.5
        xc = 0.5 * (x0 + x1)
        ax.text(
            xc,
            category_y,
            _wrap_label(category, width=14, allow_word_break=True),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=category_fontsize,
            fontweight="bold",
        )
        if j < len(category_ranges) - 1:
            ax.axvline(x1, color="0.60", lw=1.2, zorder=1)

    blocks = []
    start = 0
    for i in range(1, len(methods) + 1):
        current_type_label = _display_phase_type_label(methods[start].get("type"))
        next_type_label = None if i == len(methods) else _display_phase_type_label(methods[i].get("type"))
        if i == len(methods) or methods[i]["category"] != methods[start]["category"] or next_type_label != current_type_label:
            blocks.append((methods[start]["category"], current_type_label, start, i - 1))
            start = i

    for _, method_type, i0, i1 in blocks:
        x0 = x_positions[i0] - 0.5
        x1 = x_positions[i1] + 0.5
        ax.axvspan(x0, x1, color="white", alpha=1.0, zorder=0)
        xc = 0.5 * (x0 + x1)
        ax.text(
            xc,
            type_y,
            _wrap_label(method_type, width=14, allow_word_break=True),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=type_fontsize,
            color="0.20",
        )
        ax.axvline(x0, color="0.93", lw=0.9, zorder=1)
        ax.axvline(x1, color="0.93", lw=0.9, zorder=1)


def _add_pressure_reservoir_bands(
    ax: plt.Axes,
    reservoir_bands: Sequence[Mapping[str, Any]],
    *,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    label_fontsize: float = MANUSCRIPT_RESERVOIR_LABEL_SIZE,
    label_x: Optional[float] = None,
) -> None:
    if label_x is None:
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        x_text = 0.985
        ha = "left"
        clip_on = True
    else:
        transform = ax.transData
        x_text = float(label_x)
        ha = "right"
        clip_on = True

    def _format_reservoir_label(raw_label: str) -> str:
        label = raw_label.strip()
        for token in (" km ", "km "):
            if token in label:
                label = label.split(token, 1)[1].strip()
                break
        label = label.removeprefix("<").removeprefix(">").strip()
        if label.endswith("-crustal"):
            label = f"{label}\nreservoir"
        return label

    for band in reservoir_bands:
        min_depth = band.get("min_depth_km", band.get("min km"))
        max_depth = band.get("max_depth_km", band.get("max km"))
        if min_depth is None or max_depth is None:
            continue
        p0, p1 = _convert_depth_range_to_pressure_range(
            (float(min_depth), float(max_depth)),
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
        )
        y0, y1 = sorted((p0, p1))
        color = band.get("color", "0.85")
        alpha = float(band.get("alpha", 0.18))
        hatch = band.get("hatch")
        linestyle = band.get("linestyle", "-")
        edgecolor = band.get("edgecolor", color)
        label = _clean_str(band.get("label"))
        ax.axhspan(
            y0,
            y1,
            xmin=0.0,
            xmax=1.0,
            facecolor=color,
            edgecolor=edgecolor,
            alpha=alpha,
            hatch=hatch,
            linestyle=linestyle,
            linewidth=0.8 if hatch or linestyle != "-" else 0,
            zorder=0.45,
        )
        if label:
            ax.text(
                x_text,
                0.5 * (y0 + y1),
                _format_reservoir_label(label),
                transform=transform,
                ha=ha,
                va="center",
                fontsize=label_fontsize,
                fontweight="bold",
                color=band.get("text_color", "0.20"),
                zorder=1.6,
                clip_on=clip_on,
            )


def _group_literature_methods(
    df: pd.DataFrame,
    *,
    kind: str,
    literature_pressure_source: str = "pressure",
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str, str, str, str], list[pd.Series]] = defaultdict(list)
    for _, row in df.iterrows():
        groups[_build_method_id(row)].append(row)

    methods = []
    for rows in groups.values():
        representative = rows[0]
        eruptions = []
        for row in rows:
            if kind == "P":
                ranges, range_uncertainties = _extract_ranges_pressure(
                    row,
                    literature_pressure_source=literature_pressure_source,
                    densities_kg_m3=densities_kg_m3,
                    layer_boundaries_km=layer_boundaries_km,
                )
            else:
                ranges = _extract_ranges_temperature(row)
                range_uncertainties = [row.get("uncertainty", np.nan)] * len(ranges)
            if len(ranges) == 0:
                continue
            eruptions.append(
                {
                    "eruption": _clean_str(row.get("eruption(s)")),
                    "ranges": ranges,
                    "uncertainty": _first_valid_row_value(
                        row,
                        ("uncertainty", "uncertainty kbar", "uncertainty km"),
                    ),
                    "range_uncertainties": range_uncertainties,
                    "notes": _clean_str(row.get("notes")),
                }
            )
        if len(eruptions) == 0:
            continue
        methods.append(
            {
                "category": _clean_str(representative.get("category")),
                "type": _clean_str(representative.get("type")),
                "label": _make_lit_xtick_label(representative),
                "thermobatometer": _clean_str(representative.get("thermobatometer")),
                "eruptions": eruptions,
            }
        )

    categories = sorted({method["category"] for method in methods})
    category_order = {category: i for i, category in enumerate(categories)}
    methods.sort(key=lambda method: (category_order.get(method["category"], 999), method["type"], method["label"]))
    return methods


def _build_this_study_columns_for_comparison(
    kind: str,
    state: Mapping[str, Any],
    *,
    selection_threshold: float,
    use_model_abbreviations: bool = False,
) -> list[dict[str, Any]]:
    kind_state = state[kind]
    columns = []

    for phase_type in ("cpx_only", "cpx_liq"):
        for year in ("2006", "2010"):
            rank_pcts = kind_state["ranks"][year][phase_type]
            df_pred = kind_state["results"][year][phase_type]
            models_to_plot = _pick_models_for_single_eruption(rank_pcts, threshold=selection_threshold)
            rank_pct_map = {model_name: pct for model_name, pct in rank_pcts}

            for rank_i, model_name in enumerate(models_to_plot):
                if model_name not in df_pred.columns:
                    raise KeyError(f"{kind} results {year} {phase_type} are missing column {model_name!r}.")
                columns.append(
                    {
                        "category": "This study",
                        "type": phase_type,
                        "eruption": year,
                        "model": model_name,
                        "label": _model_axis_label(
                            model_name,
                            kind,
                            use_model_abbreviations=use_model_abbreviations,
                        ),
                        "data": _remove_boxplot_outliers(df_pred[model_name].dropna().to_numpy()),
                        "rank_i": rank_i,
                        "pct": rank_pct_map.get(model_name),
                        "fill": "blue" if year == "2006" else "red",
                        "uncertainty": kind_state["uncertainty"].get(model_name, np.nan),
                    }
                )

    liquid_results = kind_state.get("liquid_results", {})
    if liquid_results:
        liquid_subcolumns = []
        liquid_colors = {"2006": "blue", "2010": "red"}
        for year in ("2006", "2010"):
            year_results = liquid_results.get(year, {})
            for phase_name in ("glass", "bulk"):
                if phase_name not in year_results:
                    continue
                liquid_subcolumns.append(
                    {
                        "eruption": year,
                        "phase_name": phase_name,
                        "data": year_results[phase_name],
                        "fill": liquid_colors[year],
                        "uncertainty": _liquid_prediction_uncertainty(kind, phase_name),
                    }
                )

        if liquid_subcolumns:
            columns.append(
                {
                    "category": "This study",
                    "type": "liquid",
                    "label": "Weber & Blundy, 2024",
                    "subcolumns": liquid_subcolumns,
                }
            )

    return columns


def _plot_this_study_boxes_on_comparison(
    ax: plt.Axes,
    this_cols: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    annotation_fontsize: float = MANUSCRIPT_ANNOTATION_SIZE,
) -> Optional[Mapping[str, Any]]:
    if len(this_cols) == 0:
        return None

    x_centers = np.arange(1, len(this_cols) + 1)
    last_bp: Optional[Mapping[str, Any]] = None

    for i, col in enumerate(this_cols):
        x_center = x_centers[i]

        if "subcolumns" in col:
            subcolumns = list(col["subcolumns"])
            eruption_positions = {"2006": x_center - 0.10, "2010": x_center + 0.10}
            sub_positions = np.array([eruption_positions.get(subcol["eruption"], x_center) for subcol in subcolumns], dtype=float)
            last_bp = ax.boxplot(
                [subcol["data"] for subcol in subcolumns],
                positions=sub_positions,
                widths=0.20,
                showfliers=False,
                patch_artist=True,
            )

            phase_data: dict[str, list[np.ndarray]] = defaultdict(list)
            phase_positions: dict[str, list[float]] = defaultdict(list)
            for j, subcol in enumerate(subcolumns):
                _style_bp_item(last_bp, j, facecolor=subcol["fill"], edgecolor="k", lw=1.1, alpha=1.0)
                _add_vertical_uncertainty_band(
                    ax,
                    sub_positions[j],
                    subcol["data"],
                    subcol.get("uncertainty"),
                    0.23,
                    "lightgray",
                    alpha=1.0,
                    zorder=2.4,
                )
                phase_data[subcol["phase_name"]].append(np.asarray(subcol["data"], dtype=float))
                phase_positions[subcol["phase_name"]].append(float(sub_positions[j]))

            y_min, y_max = ax.get_ylim()
            y_pad = max((y_max - y_min) * 0.03, 0.15 if kind == "P" else 3.0)
            for phase_name, phase_label in (("glass", "glass"), ("bulk", "bulk rock")):
                values = phase_data.get(phase_name, [])
                if not values:
                    continue
                phase_concat = np.concatenate(values)
                phase_concat = phase_concat[np.isfinite(phase_concat)]
                if phase_concat.size == 0:
                    continue
                label_x = float(np.mean(phase_positions.get(phase_name, [x_center])))
                if kind == "P":
                    label_y = float(np.nanmin(phase_concat) - y_pad)
                else:
                    label_y = float(np.nanmax(phase_concat) + y_pad)
                _annotate_grouped_box_label(ax, label_x, label_y, phase_label, fontsize=annotation_fontsize, color="0.15")
            continue

        last_bp = ax.boxplot([col["data"]], positions=[x_center], widths=0.20, showfliers=False, patch_artist=True)
        _style_bp_item(last_bp, 0, facecolor=col["fill"], edgecolor="k", lw=1.2, alpha=1.0)
        _add_vertical_uncertainty_band(
            ax,
            x_center,
            col["data"],
            col.get("uncertainty"),
            0.23,
            "lightgray",
            alpha=1.0,
            zorder=2.4,
        )
        pct = col.get("pct")
        if pct is not None:
            y_pad_frac = 0.06 if kind == "P" else 0.03
            _annotate_above_boxplot_whisker(
                ax,
                x_center,
                col["data"],
                f"{int(round(pct))}%",
                col["fill"],
                y_pad_frac=y_pad_frac,
                fontsize=annotation_fontsize,
            )

    return last_bp


def _build_methods_for_bands(this_cols: Sequence[Mapping[str, Any]], lit_methods: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    this_for_bands = [{"category": col["category"], "type": col["type"], "label": col["label"]} for col in this_cols]
    lit_for_bands = [{"category": method["category"], "type": method["type"], "label": method["label"]} for method in lit_methods]
    return this_for_bands + lit_for_bands


def _is_melts_modeling_method(method: Mapping[str, Any], kind: str) -> bool:
    return (
        kind == "T"
        and _clean_str(method.get("category")).lower() == "modeling"
        and _clean_str(method.get("type")).upper() == "MELTS"
    )


def _format_melts_note(note: Any) -> str:
    note_clean = _clean_str(note)
    if "4 wt." in note_clean:
        return "3 kbar and\n4 wt.% H2O"
    if "6 wt." in note_clean:
        return "3 kbar and\n6 wt.% H2O"
    return _wrap_label(note_clean, width=14)


def _draw_melts_modeling_column(
    ax: plt.Axes,
    xi: float,
    eruptions: Sequence[Mapping[str, Any]],
    *,
    text_fontsize: float = MANUSCRIPT_ANNOTATION_SIZE,
) -> None:
    melts_points = []
    for eruption_item in eruptions:
        for range_min, range_max in eruption_item["ranges"]:
            melts_points.append(
                {
                    "y": 0.5 * (float(range_min) + float(range_max)),
                    "label": _format_melts_note(eruption_item.get("notes", "")),
                }
            )

    if len(melts_points) == 0:
        return

    melts_points = sorted(melts_points, key=lambda item: item["y"], reverse=True)
    melts_color = "#8a0f9c"
    cap_half_width = 0.13
    reference_arrow_length = None

    for idx, point in enumerate(melts_points):
        y = point["y"]
        ax.text(xi, y, point["label"], fontsize=text_fontsize, color="0.10", ha="center", va="center", zorder=4, clip_on=True)

        arrow_start = y - 7.0
        if idx < len(melts_points) - 1:
            arrow_end = melts_points[idx + 1]["y"] + 7.0
            reference_arrow_length = arrow_start - arrow_end
        else:
            arrow_length = reference_arrow_length if reference_arrow_length is not None else 16.0
            arrow_end = arrow_start - arrow_length

        ax.hlines(arrow_start, xi - cap_half_width, xi + cap_half_width, color=melts_color, lw=2.3, zorder=4)
        ax.annotate(
            "",
            xy=(xi, arrow_end),
            xytext=(xi, arrow_start),
            arrowprops=dict(arrowstyle="-|>", color=melts_color, lw=2.3),
            zorder=4,
        )


def _plot_ranked_literature_panel(
    ax: plt.Axes,
    kind: str,
    state: Mapping[str, Any],
    literature_df: pd.DataFrame,
    *,
    selection_threshold: float,
    add_literature_legend: bool,
    pressure_ylim: Optional[tuple[float, float]] = None,
    temperature_ylim: Optional[tuple[float, float]] = None,
    pressure_ticks: Optional[Sequence[float]] = None,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    literature_pressure_source: str = "pressure",
    depth_tick_step: float = 5,
    depth_max: float = 37,
    use_model_abbreviations: bool = False,
    pressure_reservoir_bands: Optional[Sequence[Mapping[str, Any]]] = None,
    axis_label_fontsize: float = MANUSCRIPT_AXIS_LABEL_SIZE,
    y_axis_label_fontsize: float = MANUSCRIPT_Y_AXIS_LABEL_SIZE,
    tick_labelsize: float = MANUSCRIPT_TICK_LABEL_SIZE,
    model_tick_labelsize: float = MANUSCRIPT_MODEL_TICK_LABEL_SIZE,
    group_label_fontsize: float = MANUSCRIPT_GROUP_LABEL_SIZE,
    legend_fontsize: float = MANUSCRIPT_LEGEND_SIZE,
    annotation_fontsize: float = MANUSCRIPT_ANNOTATION_SIZE,
    reservoir_label_fontsize: float = MANUSCRIPT_RESERVOIR_LABEL_SIZE,
) -> None:
    this_cols = _build_this_study_columns_for_comparison(
        kind,
        state,
        selection_threshold=selection_threshold,
        use_model_abbreviations=use_model_abbreviations,
    )
    lit_methods = _group_literature_methods(
        literature_df,
        kind=kind,
        literature_pressure_source=literature_pressure_source,
        densities_kg_m3=densities_kg_m3,
        layer_boundaries_km=layer_boundaries_km,
    )

    methods_all = _build_methods_for_bands(this_cols, lit_methods)
    x_all = np.arange(1, len(methods_all) + 1)
    reservoir_label_x = None
    if kind == "P" and pressure_reservoir_bands:
        reservoir_label_x = len(methods_all) + 1.45
    _add_category_and_type_bands(
        ax,
        methods_all,
        x_all,
        category_fontsize=group_label_fontsize,
        type_fontsize=annotation_fontsize,
    )
    if kind == "P" and pressure_reservoir_bands:
        _add_pressure_reservoir_bands(
            ax,
            pressure_reservoir_bands,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
            label_fontsize=reservoir_label_fontsize,
            label_x=reservoir_label_x,
        )

    n_this = len(this_cols)
    if n_this > 0:
        _plot_this_study_boxes_on_comparison(
            ax,
            this_cols,
            kind=kind,
            annotation_fontsize=annotation_fontsize,
        )

    if len(lit_methods) > 0:
        x_lit = np.arange(1, len(lit_methods) + 1) + n_this
        for i, method in enumerate(lit_methods):
            xi = x_lit[i]
            thermobarometer_text = _clean_str(method.get("thermobatometer"))

            if _is_melts_modeling_method(method, kind):
                _draw_melts_modeling_column(ax, xi, method["eruptions"], text_fontsize=annotation_fontsize)
                continue

            for eruption_item in method["eruptions"]:
                color = _eruption_color(eruption_item["eruption"])
                dx = _eruption_slot_offset(eruption_item["eruption"], delta=0.16)
                ranges = eruption_item["ranges"]
                range_uncertainties = eruption_item.get("range_uncertainties", [])
                offsets_local = np.linspace(-0.06, 0.06, max(1, len(ranges)))

                for k, (range_min, range_max) in enumerate(ranges):
                    xk = xi + dx + offsets_local[k]
                    midpoint = 0.5 * (float(range_min) + float(range_max))
                    uncertainty = (
                        range_uncertainties[k]
                        if k < len(range_uncertainties)
                        else eruption_item.get("uncertainty", np.nan)
                    )
                    _add_vertical_uncertainty_band_from_median(
                        ax,
                        xk,
                        midpoint,
                        uncertainty,
                        0.22,
                        "lightgray",
                        alpha=1.0,
                        zorder=2.2,
                        median_marker="line",
                    )
                    if np.isclose(float(range_min), float(range_max), equal_nan=False):
                        ax.plot(
                            xk,
                            range_min,
                            marker="o",
                            markersize=6.5,
                            color=color,
                            markeredgecolor=color,
                            linestyle="None",
                            zorder=3,
                        )
                    else:
                        ax.vlines(xk, range_min, range_max, color=color, lw=4.0, zorder=3)
                        ax.hlines([range_min, range_max], xk - 0.07, xk + 0.07, color=color, lw=2.4, zorder=3)

                    if _clean_str(method["category"]).lower() == "geothermobarometry" and thermobarometer_text:
                        if kind == "P":
                            y_top = min(range_min, range_max)
                            y_text = y_top - 0.25
                        else:
                            y_top = max(range_min, range_max)
                            y_text = y_top + 15.0
                        ax.text(
                            xk,
                            y_text,
                            _wrap_label(thermobarometer_text, width=14),
                            fontsize=annotation_fontsize,
                            color="0.20",
                            ha="center",
                            va="bottom",
                            zorder=4,
                            clip_on=True,
                        )

    x_right = len(methods_all) + (1.55 if reservoir_label_x is not None else 0.5)
    ax.set_xlim(0.5, x_right)
    ax.set_xticks(x_all)
    raw_xtick_labels = [method["label"] for method in methods_all]
    xtick_labels = [_wrap_label(label, width=12) for label in raw_xtick_labels]
    ax.set_xticklabels(
        xtick_labels,
        rotation=0,
        ha="center",
        fontsize=model_tick_labelsize if use_model_abbreviations else tick_labelsize,
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.tick_params(axis="x", which="major", labelsize=model_tick_labelsize if use_model_abbreviations else tick_labelsize)
    _apply_threshold_xtick_rotation(ax, raw_xtick_labels)

    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.8, color="0.85", zorder=0)
    ax.grid(True, which="minor", axis="y", linestyle="-", linewidth=0.5, color="0.92", zorder=0)

    if kind == "P":
        _apply_pressure_depth_axes(
            ax,
            pressure_ylim=pressure_ylim,
            pressure_ticks=pressure_ticks,
            depth_tick_step=depth_tick_step,
            depth_max=depth_max,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
            label_fontsize=y_axis_label_fontsize,
            tick_labelsize=tick_labelsize,
        )
    else:
        ax.set_ylabel("Temperature (°C)", fontsize=y_axis_label_fontsize)
        if temperature_ylim is not None:
            ax.set_ylim(*temperature_ylim)
        ax.tick_params(axis="y", labelsize=tick_labelsize)
    ax.xaxis.label.set_size(axis_label_fontsize)

    if add_literature_legend:
        handles = [
            Patch(facecolor="blue", edgecolor="k", alpha=1.0, label="2006 (this study)"),
            Patch(facecolor="red", edgecolor="k", alpha=1.0, label="2010 (this study)"),
            Line2D([0], [0], color="purple", lw=4.0, label="2006&2010 (literature)"),
            Line2D([0], [0], color="blue", lw=4.0, label="2006 (literature)"),
            Line2D([0], [0], color="red", lw=4.0, label="2010 (literature)"),
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.005, 0.02),
            frameon=True,
            ncol=3,
            fontsize=legend_fontsize,
            borderpad=0.35,
            handlelength=1.6,
            handletextpad=0.5,
        )


def _apply_this_study_axis_font_sizes(
    fig: plt.Figure,
    *,
    labelsize: float = MANUSCRIPT_THIS_STUDY_AXIS_LABEL_SIZE,
    y_labelsize: float = MANUSCRIPT_THIS_STUDY_Y_AXIS_LABEL_SIZE,
    tick_labelsize: float = MANUSCRIPT_THIS_STUDY_TICK_LABEL_SIZE,
    model_tick_labelsize: float = MANUSCRIPT_THIS_STUDY_MODEL_TICK_LABEL_SIZE,
    legend_fontsize: float = MANUSCRIPT_THIS_STUDY_LEGEND_SIZE,
) -> None:
    for ax in fig.axes:
        ax.xaxis.label.set_size(labelsize)
        ax.yaxis.label.set_size(y_labelsize)
        ax.tick_params(axis="both", labelsize=tick_labelsize)
        ax.tick_params(axis="x", labelsize=model_tick_labelsize)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(legend_fontsize)


def _add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.06,
    y: float = 1.045,
    fontsize: float = MANUSCRIPT_PANEL_LABEL_SIZE,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
        zorder=10,
        clip_on=False,
    )


def plot_ranked_thermobarometry_this_study(
    cpx_only_workflows: Any,
    cpx_liq_workflows: Any,
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    selection_threshold: float = 50.0,
    pressure_ylim: Optional[tuple[float, float]] = None,
    pressure_ticks: Optional[Sequence[float]] = None,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    depth_tick_step: float = 5,
    depth_max: float = 37,
    figsize: tuple[float, float] = (20.5, 16),
    constrained_layout: bool = True,
    add_legend: bool = True,
    panel_labels: tuple[str, str] = ("(a)", "(b)"),
    save_path: Optional[str | Path] = None,
    use_model_abbreviations: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot the notebook-style two-panel "this study" summary figure.

    Parameters
    ----------
    cpx_only_workflows, cpx_liq_workflows
        Workflow bundles for the `cpx_only` and `cpx_liq` branches. Each input
        may be a :class:`ThermobarometryWorkflowBundle` or a mapping with keys
        like ``pressure_2006`` / ``temperature_2010``.
    pressure_columns, temperature_columns
        Mappings with keys ``"cpx_only"`` and ``"cpx_liq"`` whose values are
        the model-result column names to plot in left-to-right order.
    pressure_model_pool, temperature_model_pool : sequence, optional
        Model objects used to draw uncertainty bands. Their ``model_name`` and
        ``uncertainty`` attributes are read when available.
    selection_threshold : float, default 50.0
        If the top-ranked model frequency is above this percentage, only that
        model is highlighted. Otherwise the top two are highlighted.
    pressure_ylim, pressure_ticks : optional
        Optional fixed pressure-axis limits/ticks for the pressure panel.
    save_path : path-like, optional
        If given, save the figure after creation.

    Returns
    -------
    fig, axes
        Matplotlib figure and the two panel axes.
    """
    state = _build_ranked_thermobarometry_state(
        cpx_only_workflows,
        cpx_liq_workflows,
        pressure_columns,
        temperature_columns,
        pressure_model_pool=pressure_model_pool,
        temperature_model_pool=temperature_model_pool,
    )

    fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=constrained_layout)
    _plot_ranked_this_study_kind_panel(
        axes[0],
        "P",
        state,
        selection_threshold=selection_threshold,
        pressure_ylim=pressure_ylim,
        pressure_ticks=pressure_ticks,
        densities_kg_m3=densities_kg_m3,
        layer_boundaries_km=layer_boundaries_km,
        depth_tick_step=depth_tick_step,
        depth_max=depth_max,
        use_model_abbreviations=use_model_abbreviations,
        axis_label_fontsize=MANUSCRIPT_THIS_STUDY_AXIS_LABEL_SIZE,
        y_axis_label_fontsize=MANUSCRIPT_THIS_STUDY_Y_AXIS_LABEL_SIZE,
        tick_labelsize=MANUSCRIPT_THIS_STUDY_TICK_LABEL_SIZE,
        model_tick_labelsize=MANUSCRIPT_THIS_STUDY_MODEL_TICK_LABEL_SIZE,
        group_label_fontsize=MANUSCRIPT_THIS_STUDY_GROUP_LABEL_SIZE,
        annotation_fontsize=MANUSCRIPT_THIS_STUDY_ANNOTATION_SIZE,
    )
    _plot_ranked_this_study_kind_panel(
        axes[1],
        "T",
        state,
        selection_threshold=selection_threshold,
        depth_tick_step=depth_tick_step,
        depth_max=depth_max,
        use_model_abbreviations=use_model_abbreviations,
        axis_label_fontsize=MANUSCRIPT_THIS_STUDY_AXIS_LABEL_SIZE,
        y_axis_label_fontsize=MANUSCRIPT_THIS_STUDY_Y_AXIS_LABEL_SIZE,
        tick_labelsize=MANUSCRIPT_THIS_STUDY_TICK_LABEL_SIZE,
        model_tick_labelsize=MANUSCRIPT_THIS_STUDY_MODEL_TICK_LABEL_SIZE,
        group_label_fontsize=MANUSCRIPT_THIS_STUDY_GROUP_LABEL_SIZE,
        annotation_fontsize=MANUSCRIPT_THIS_STUDY_ANNOTATION_SIZE,
    )

    if add_legend:
        legend_items = [
            Patch(facecolor="blue", edgecolor="k", alpha=1.0, label="2006 eruption"),
            Patch(facecolor="red", edgecolor="k", alpha=1.0, label="2010 eruption"),
            Patch(facecolor="white", edgecolor="k", linestyle="--", linewidth=3.0, label="Selected model"),
        ]
        axes[0].legend(
            handles=legend_items,
            bbox_to_anchor=(0.02, 0.98),
            ncol=2,
            frameon=True,
            fontsize=MANUSCRIPT_THIS_STUDY_LEGEND_SIZE,
            borderpad=0.35,
            handlelength=1.6,
            handletextpad=0.5,
        )

    if panel_labels:
        _add_panel_label(axes[0], panel_labels[0])
        if len(panel_labels) > 1:
            _add_panel_label(axes[1], panel_labels[1], y=1.045)

    _apply_this_study_axis_font_sizes(fig)

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=300)

    return fig, axes


def plot_ranked_thermobarometry_literature_comparison(
    cpx_only_workflows: Any,
    cpx_liq_workflows: Any,
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    pressure_literature: Any,
    temperature_literature: Any,
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    liquid_results: Optional[Mapping[str, Any]] = None,
    selection_threshold: float = 50.0,
    pressure_ylim: Optional[tuple[float, float]] = None,
    temperature_ylim: Optional[tuple[float, float]] = (900, 1200),
    pressure_ticks: Optional[Sequence[float]] = None,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    literature_pressure_source: str = "pressure",
    depth_tick_step: float = 5,
    depth_max: float = 37,
    figsize: tuple[float, float] = (16.5, 16),
    constrained_layout: bool = True,
    add_literature_legend: bool = True,
    panel_labels: tuple[str, str] = ("(a)", "(b)"),
    save_path: Optional[str | Path] = None,
    use_model_abbreviations: bool = False,
    pressure_reservoir_bands: Optional[Sequence[Mapping[str, Any]]] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot the notebook-style two-panel "this study vs literature" comparison.

    Parameters
    ----------
    pressure_literature, temperature_literature
        Either DataFrames or Excel paths. If a ``plot`` column is present it is
        used as a boolean include mask, matching the notebook workflow.

    Returns
    -------
    fig, axes
        Matplotlib figure and the two panel axes.
    """
    state = _build_ranked_thermobarometry_state(
        cpx_only_workflows,
        cpx_liq_workflows,
        pressure_columns,
        temperature_columns,
        pressure_model_pool=pressure_model_pool,
        temperature_model_pool=temperature_model_pool,
        liquid_results=liquid_results,
    )
    pressure_literature_df = _load_literature_table(pressure_literature, "pressure_literature")
    temperature_literature_df = _load_literature_table(temperature_literature, "temperature_literature")

    fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=constrained_layout)
    _plot_ranked_literature_panel(
        axes[0],
        "P",
        state,
        pressure_literature_df,
        selection_threshold=selection_threshold,
        add_literature_legend=add_literature_legend,
        pressure_ylim=pressure_ylim,
        pressure_ticks=pressure_ticks,
        densities_kg_m3=densities_kg_m3,
        layer_boundaries_km=layer_boundaries_km,
        literature_pressure_source=literature_pressure_source,
        depth_tick_step=depth_tick_step,
        depth_max=depth_max,
        use_model_abbreviations=use_model_abbreviations,
        pressure_reservoir_bands=pressure_reservoir_bands,
    )
    _plot_ranked_literature_panel(
        axes[1],
        "T",
        state,
        temperature_literature_df,
        selection_threshold=selection_threshold,
        add_literature_legend=False,
        temperature_ylim=temperature_ylim,
        depth_tick_step=depth_tick_step,
        depth_max=depth_max,
        use_model_abbreviations=use_model_abbreviations,
    )

    if panel_labels:
        _add_panel_label(axes[0], panel_labels[0])
        if len(panel_labels) > 1:
            _add_panel_label(axes[1], panel_labels[1])

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=300)

    return fig, axes


def _show_ranked_thermobarometry_figures(figures: Mapping[str, tuple[plt.Figure, np.ndarray]]) -> None:
    backend = plt.get_backend().lower()
    if backend.endswith("agg"):
        try:
            from IPython.display import display
        except ImportError:
            return

        # Agg cannot show interactively, but notebooks can still render figures.
        for fig, _axes in figures.values():
            display(fig)
        return

    plt.show()


def plot_ranked_thermobarometry_summary(
    cpx_only_workflows: Any,
    cpx_liq_workflows: Any,
    pressure_columns: Mapping[str, Sequence[str]],
    temperature_columns: Mapping[str, Sequence[str]],
    *,
    pressure_model_pool: Optional[Sequence[Any]] = None,
    temperature_model_pool: Optional[Sequence[Any]] = None,
    pressure_literature: Any = None,
    temperature_literature: Any = None,
    liquid_results: Optional[Mapping[str, Any]] = None,
    selection_threshold: float = 50.0,
    pressure_ylim: Optional[tuple[float, float]] = None,
    temperature_ylim: Optional[tuple[float, float]] = (900, 1150),
    pressure_ticks: Optional[Sequence[float]] = None,
    densities_kg_m3: Optional[Sequence[float]] = None,
    layer_boundaries_km: Optional[Sequence[float]] = None,
    literature_pressure_source: str = "pressure",
    depth_tick_step: float = 5,
    depth_max: float = 37,
    this_study_figsize: tuple[float, float] = (20.5, 16),
    literature_figsize: tuple[float, float] = (16.5, 16),
    this_study_save_path: Optional[str | Path] = None,
    literature_save_path: Optional[str | Path] = None,
    show: bool = True,
    use_model_abbreviations: bool = False,
    pressure_reservoir_bands: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, tuple[plt.Figure, np.ndarray]]:
    """
    Convenience wrapper that reproduces both notebook summary figures.

    Parameters
    ----------
    show : bool, default True
        Whether to display the generated figures before returning them.

    Returns
    -------
    dict
        Always includes ``"this_study"``. Includes
        ``"literature_comparison"`` only when both literature tables are
        provided.
    """
    figures = {
        "this_study": plot_ranked_thermobarometry_this_study(
            cpx_only_workflows,
            cpx_liq_workflows,
            pressure_columns,
            temperature_columns,
            pressure_model_pool=pressure_model_pool,
            temperature_model_pool=temperature_model_pool,
            selection_threshold=selection_threshold,
            pressure_ylim=pressure_ylim,
            pressure_ticks=pressure_ticks,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
            depth_tick_step=depth_tick_step,
            depth_max=depth_max,
            figsize=this_study_figsize,
            save_path=this_study_save_path,
            use_model_abbreviations=use_model_abbreviations,
        )
    }

    if pressure_literature is not None and temperature_literature is not None:
        figures["literature_comparison"] = plot_ranked_thermobarometry_literature_comparison(
            cpx_only_workflows,
            cpx_liq_workflows,
            pressure_columns,
            temperature_columns,
            pressure_literature,
            temperature_literature,
            pressure_model_pool=pressure_model_pool,
            temperature_model_pool=temperature_model_pool,
            liquid_results=liquid_results,
            selection_threshold=selection_threshold,
            pressure_ylim=pressure_ylim,
            temperature_ylim=temperature_ylim,
            pressure_ticks=pressure_ticks,
            densities_kg_m3=densities_kg_m3,
            layer_boundaries_km=layer_boundaries_km,
            literature_pressure_source=literature_pressure_source,
            depth_tick_step=depth_tick_step,
            depth_max=depth_max,
            figsize=literature_figsize,
            save_path=literature_save_path,
            use_model_abbreviations=use_model_abbreviations,
            pressure_reservoir_bands=pressure_reservoir_bands,
        )

    if show:
        _show_ranked_thermobarometry_figures(figures)

    return figures



def plot_ood_score_vs_deviation_(input_cpx, input_meta, model, input_liq = None, high_calculated_deviation_marker = False, petrological_ood_marker = False, T_P_ood_marker = False, ax = None):
    '''
    Plot OOD score vs deviation for a given model and input data.

    Parameters:
        input_cpx : pd.DataFrame
            Input clinopyroxene compositions.
        input_meta : pd.DataFrame
            Metadata containing true or temperature values.
        model : ThermobarometerModel
            The thermobarometer model to use for predictions.
        input_liq : pd.DataFrame, optional
            Input liquid compositions, if required by the model.
        high_calculated_deviation_marker : bool, optional
            Whether to highlight points with high calculated deviation.
        petrological_ood_marker : bool, optional
            Whether to highlight petrologically consistent OOD points.
        T_P_ood_marker : bool, optional
            Whether to highlight points which are OOD in T or P space.
        ax : matplotlib.axes.Axes, optional 
            Axes object to plot on. If None, a new figure and axes will be created.
    Returns:
        fig : matplotlib.figure.Figure
            The Figure object.
        ax_scatter : matplotlib.axes.Axes
            The Axes object with the scatter plot.
    '''
    model_name = model.model_name
    target_col = "P_kbar" if model.T_P == "P" else "T_C"
    print(f"model's key features: {model.key_features}")

    if petrological_ood_marker:
        from aims4pt.data_tools.rocks import get_TAS_rock_types, get_volcanic_rock_series
        from aims4pt.statistic_tools.density_region_analysis import rock_type_check
        input_meta['TAS_rock_type'] = input_liq.apply(get_TAS_rock_types, axis=1)
        input_meta['volcanic_rock_series'] = input_liq.apply(get_volcanic_rock_series, axis=1)
        tas_mask = [rock_type_check(rock_type, model.rock_types, report= False) for rock_type in input_meta['TAS_rock_type']]
        series_mask = [rock_type_check(rock_series, model.volcanic_rock_series, report= False) for rock_series in input_meta['volcanic_rock_series']]
        petrological_mask = [tas and series for tas, series in zip(tas_mask, series_mask)]
    
    predictions = model.predict(input_cpx, input_liq)
    ood_model = getattr(model, "OOD_detector", None)
    if ood_model is None:
        print(f"No OOD detector found for model {model_name}.")
        ood_mask = np.array([False]*len(input_cpx))
        ood_score = np.array([np.nan]*len(input_cpx))
    else:
        ood_score = ood_model.score(input_cpx, input_liq)
        ood_mask = ood_model.is_ood(input_cpx, input_liq)
    deviation = (input_meta[target_col] - predictions).abs()

    if high_calculated_deviation_marker:
        deviation_function= getattr(model, "deviation_function", None)
        if deviation_function is None:
            print(f"No deviation function found for model {model_name}.")
            deviation_mask = np.array([False]*len(input_cpx))
        else:
            estimate_deviation = deviation_function.predict_deviation(input_cpx, input_liq)
            deviation_mask = estimate_deviation >= model.uncertainty
    
    if T_P_ood_marker:
        y_lim = model.y_min_95 if hasattr(model, 'y_min_95') else -np.inf
        y_max = model.y_max_95 if hasattr(model, 'y_max_95') else np.inf
        T_P_ood_mask = (predictions < y_lim) | (predictions > y_max)
        



    # Pair up scores with deviations first so dropna keeps them aligned
    split_df = pd.DataFrame({"score": ood_score, "deviation": deviation})
    ood_df = split_df.loc[ood_mask].dropna()
    ind_df = split_df.loc[~ood_mask].dropna()
    ood_x, ood_y = ood_df["score"], ood_df["deviation"]
    ind_x, ind_y = ind_df["score"], ind_df["deviation"]

    if petrological_ood_marker:
        petro_mask = pd.Series(petrological_mask, index=split_df.index)
        petro_df = split_df.loc[~petro_mask].dropna()
        petro_ood_x = petro_df["score"]
        petro_ood_y = petro_df["deviation"]


    if high_calculated_deviation_marker:
        deviation_mask_series = pd.Series(deviation_mask, index=split_df.index)
        deviation_ood_df = split_df.loc[deviation_mask_series].dropna()
        deviation_ood_x = deviation_ood_df["score"]
        deviation_ood_y = deviation_ood_df["deviation"]

    if T_P_ood_marker:
        T_P_ood_mask_series = pd.Series(T_P_ood_mask, index=split_df.index)
        T_P_ood_df = split_df.loc[T_P_ood_mask_series].dropna()
        T_P_ood_x = T_P_ood_df["score"]
        T_P_ood_y = T_P_ood_df["deviation"]

    if ax is not None:
        ax_scatter = ax
        fig = ax.figure
    else:   
        fig, ax_scatter = plt.subplots(figsize=(10, 6), dpi=200)

    # Highlight typical deviation ranges
    if len(ind_x) > 0:
        ind_q = np.quantile(ind_y, [0.25, 0.5, 0.75])
        ind_xmin, ind_xmax = ind_x.min(), ind_x.max()
        ax_scatter.fill_between(
            [ind_xmin, ind_xmax],
            [ind_q[0],  ind_q[0]],
            [ind_q[2],  ind_q[2]],
            color="#FF7F0E",
            alpha=0.18,
            zorder=1,
        )
        ax_scatter.hlines(y=ind_y.mean(), xmin=ind_xmin, xmax=ind_xmax,  color="#FF7F0E", linestyle="-", linewidth=1) #, label="In-Distribution Mean Deviation"

    if len(ood_x) > 0:
        ood_q = np.quantile(ood_y, [0.25, 0.5, 0.75])
        ood_xmin, ood_xmax = ood_x.min(), ood_x.max()
        ax_scatter.hlines(y=ood_y.mean(), xmin=ood_xmin, xmax=ood_xmax,  color="#1F77B4", linestyle="-", linewidth=1) #, label="OOD Mean Deviation"
        ax_scatter.fill_between(
            [ood_xmin, ood_xmax],
            [ood_q[0],  ood_q[0]],
            [ood_q[2],  ood_q[2]],
            color="#1F77B4",
            alpha=0.15,
            zorder=1,
        )

    # In-distribution scatter
    ax_scatter.scatter(
        ind_x,
        ind_y,
        s=30,
        alpha=0.5,
        color="#FF7F0E",
        edgecolors="none",
        marker = "o",
        label="In-distribution",
        zorder=2,
        rasterized=True,
        
    )

    # OOD scatter
    ax_scatter.scatter(
        ood_x,
        ood_y,
        s=30,
        alpha=0.5,
        color="#1F77B4",
        edgecolors="none",
        label="OOD (features)",
        zorder=3,
        rasterized=True,
        
    )

    if T_P_ood_marker:
        ax_scatter.scatter(
            T_P_ood_x,
            T_P_ood_y,
            s=60,
            alpha=0.7,
            color="orange",
            edgecolors="black",
            label=f"OOD (predicted {model.T_P})",
            zorder=4,
            rasterized=True,
            )
        
    if petrological_ood_marker:
        ax_scatter.scatter(
            petro_ood_x,
            petro_ood_y,
            s=50,
            alpha=0.7,
            color="purple",
            edgecolors="black",
            label="OOD (TAS)",
            zorder=5,
            rasterized=True,
            )

    if high_calculated_deviation_marker:
        ax_scatter.scatter(
            deviation_ood_x,
            deviation_ood_y,
            s=50,
            alpha=0.7,
            color="green",
            edgecolors="black",
            label=f"High estimated deviation (D > uncertainty)",
            zorder=6,
            rasterized=True,
            )




    ax_scatter.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_scatter.axhline(model.uncertainty, color="gray", linestyle="--", linewidth=1, label="Model uncertainty")
    ax_scatter.invert_xaxis()
    ax_scatter.set_xlabel("Signed distance to the boundary", fontsize=12)
    ax_scatter.set_ylabel(f"Absolute deviation ({'kbar' if model.T_P == 'P' else '°C'})", fontsize=12)
    ax_scatter.grid(alpha=0.3, linestyle="--")
    ax_scatter.legend(loc="upper right", fontsize=10)
    # ax_scatter.set_ylim(0, 30)
    # ax_scatter.set_ylim(0, 20)
    if ax is None:
        plt.title(f"OOD Score vs Deviation for {model_name}", fontsize=14)
        plt.tight_layout()
        plt.show()
    return fig, ax_scatter
