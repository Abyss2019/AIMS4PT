# aims4pt/data_tools/data_engineering.py
import pandas as pd
import numpy as np
import re
from aims4pt.utils import normalize_column_names
import matplotlib.pyplot as plt
import seaborn as sns

def compare_natural_exp_dataset (natural_df, exp_df, x_y_pairs, T_color= None, P_color=None, ):
    '''
    Compare natural and experimental data. Useful when applying a Geothermobarometry method to a dataset.
    Harker diagrams are plotted for each pair of x and y columns.

    Parameters:
    -----------
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
    cmap= "YlOrRd"

    for i, (x, y) in enumerate(x_y_pairs):
        ax[i].scatter(natural_df[x], natural_df[y], c='blue', s=40, alpha=0.5, label='natural data', marker="2")
        scatter = None
        if T_color:
            scatter = ax[i].scatter(exp_df[x], exp_df[y], c=exp_df[T_color], s=40, edgecolors='k', alpha=1, label='experimental data', cmap=cmap)

        if P_color:
            scatter = ax[i].scatter(exp_df[x], exp_df[y], c=exp_df[P_color], s=40, edgecolors='k', alpha=1, label='experimental data', cmap=cmap)
        else: 
            ax[i].scatter(exp_df[x], exp_df[y], c='grey', s=40, edgecolors='k', alpha=1, label='experimental data')

        sns.kdeplot(natural_df, x=x, y=y, ax=ax[i], fill=False, levels=1, color='blue')
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
def compare_natural_exp_dataset_v_lazy(natural_df_one_phase, exp_df_one_phase, x_y_pairs, T_color= None, P_color=None, P_T_range = [None, None]):
    '''
    Compare natural and experimental data. Useful when applying a Geothermobarometry method to a dataset.
    Input data with single phase only.

    Parameters:
    -----------
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

    compare_natural_exp_dataset(natural_df_norm, exp_df_norm, x_y_pairs, T_color= T_color, P_color=P_color)
