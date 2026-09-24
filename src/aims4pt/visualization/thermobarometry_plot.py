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
from aims4pt.utils import normalize_column_names
from aims4pt.toolkit_utils import wrap_text
from aims4pt.visualization.plot_utils import get_subplot_shape


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
