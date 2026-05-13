# aims4pt/statistic_tools/sensitivity_test.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from aims4pt.visualization.plot_utils import get_subplot_shape


def sensitive_test(df, base_abs_error, model, plot=True, scale_factors=1):
    '''
    Perform a sensitivity analysis by changing the value of a feature and predicting the response variable.

    Parameters:
    df (DataFrame): Input data frame.
    base_abs_error (dict): Dictionary of base absolute errors for each feature.
    model (object): Model object with a predict method.
    plot (bool): Plot the results.
    scale_factors (int): Scale factor for the error.

    Returns:
    pd.DataFrame: Formatted DataFrame containing the corresponding prediction for each feature.
    '''

    if isinstance(base_abs_error, dict):
        keys = list(base_abs_error.keys())
    elif isinstance(base_abs_error, list):
        keys = df.columns.tolist()
        base_abs_error = dict(zip(keys, base_abs_error))

    original_df = df.copy()

    results = {}

    for key in keys:
        wrong_df = df.copy()
        wrong_df[key] = wrong_df[key] + base_abs_error[key]*scale_factors

        # print(wrong_df)
        # calculate T/P

        prediction = model.predict(wrong_df)

        results[key] = prediction

    results = pd.DataFrame(results)
    # set index to scale factors
    results.index = [scale_factors]

    return results


def plot_sensitivity(df, re_error_dict, title='Temperature (℃)'):
    '''
    Plot the sensitivity analysis results.

    Parameters:
    df (DataFrame): Input data frame.
    re_error_dict (dict): Dictionary of relative errors for each feature.
    title (str): Plot title.

    '''
    color_index = range(len(df.index))
    color_map = plt.cm.get_cmap('summer', len(color_index))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    if 0 in df.index:
        known_wt = df.loc[0].mean()
        ax.axhline(y=known_wt, color='black', linestyle='--',
                   label='Baseline ($C_{known}$)')

    for index, i in zip(df.index, color_index):
        T = df.loc[index]
        x = [f"{col}\n({re_error_dict[col]:.2f}%)" for col in df.columns]
        y = T
        # label

        if index == 0:
            label = '$C_{known}$'
        else:
            label = '$C_{known}$' + f"+{index}*error(wt%)"
        ax.scatter(x, y, color=color_map(i), label=label, s=100,
                   edgecolors='black', linewidths=0.5, alpha=0.5)

    if 0 in df.index:
        known_wt = df.loc[0].mean()
        # ax.axhline(y = known_wt, color = 'black', linestyle = '--', label = 'known')

        # Add a new left y-axis and set the depth unit
        ax2 = ax.twinx()  # Create a secondary axis sharing the y-axis
        ax2.set_ylabel('Difference: prediction - Baseline')  # Set the label

        # Flip horizontally to the inside
        ax2.yaxis.set_ticks_position('right')  # Keep on the left side
        # Turn the ticks and labels towards the inside of the chart
        ax2.tick_params(axis='y', colors='gray')

        # Synchronize ticks and range
        ax2.set_yticks(ax.get_yticks())
        ax2.set_ylim(ax.get_ylim())

        text = [
            f'{y - known_wt:.2f}\n({((y - known_wt)/known_wt*100):.01f}%)' for y in ax.get_yticks()]
        # Convert to depth and set color
        ax2.set_yticklabels(text, color='gray')

    ax.set_xlabel('Element\n(relative error %)')
    ax.set_ylabel('Prediction')

    ax.set_title(title)
    # fake data
    ax.legend(title='input')
    # grid for y axis

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    return ax


def plot_sensitivity_magnitude(df, re_error_dict, title='Temperature (℃)'):
    '''
    Plot the sensitivity analysis results.

    Parameters:
    df (DataFrame): Input data frame.
    re_error_dict (dict): Dictionary of relative errors for each feature.
    title (str): Plot title.

    '''

    color_index = range(len(df.index))
    # color_map from light to dark.
    color_map = plt.cm.get_cmap('summer', len(color_index))
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    # if 0 in df.index:
    #     known_wt = df.loc[0].mean()
    #     ax.axhline(y = known_wt, color = 'black', linestyle = '--', label = 'Baseline (know)')

    for index, i in zip(df.index, color_index):
        if index == 0:
            continue
        known_wt = df.loc[0].mean()
        T = abs(df.loc[index]-known_wt)
        x = [f"{col}\n({re_error_dict[col]:.2f}%)" for col in df.columns]
        y = T
        # label
        if index == 0:
            label = '$C_{known}$'
        else:
            label = '$C_{known}$'f"+{index}*error(wt%)"

        ax.scatter(x, y, color=color_map(i), label=label, s=100,
                   edgecolors='black', linewidths=0.5, alpha=0.5)

    ax.set_xlabel('Element\n(relative error %)')
    ax.set_ylabel('Absolute difference: |prediction - Baseline|')

    ax.set_title(title)
    text = [
        f'{y:.2f}\n({abs((y)/known_wt*100):.2f}%)' for y in ax.get_yticks()]
    ax.set_yticklabels(text)
    # fake data
    ax.legend(title='input')
    # grid for y axis

    ax.grid(axis='y', linestyle='--', alpha=0.5)

    ax.set_title(title)


# a overall pipeline

def sensitivity_analysis_pipeline(df, base_re_error_percentage, model, plot=True, scale_factors=[0, 1, 2], title='Temperature (℃)'):
    '''
    Perform a sensitivity analysis by changing the value of a feature and predicting the response variable.

    Parameters:
        df (DataFrame): Input data frame.
        base_re_error (dict): Dictionary of relative errors for each feature.
        model (object): Model object with a predict method.
        plot (bool): Plot the results.
        scale_factors (list): Scale factor for the error.

    Returns:
        pd.DataFrame: Formatted DataFrame containing the corresponding prediction for each feature.
    '''
    base_abs_error = {key: df[key].mean(
    )*base_re_error_percentage[key]/100 for key in base_re_error_percentage.keys()}
    print(base_abs_error)
    results_collection = pd.DataFrame()
    for scale in scale_factors:
        results = sensitive_test(df, base_abs_error, model, plot, scale)
        results_collection = pd.concat([results_collection, results])

    if plot:
        plot_sensitivity(results_collection, base_re_error_percentage, title)
        plot_sensitivity_magnitude(
            results_collection, base_re_error_percentage, title)

    return results_collection



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Helper: subplot shape. Remove this if the project already has get_subplot_shape.
# ------------------------------------------------------------
def get_subplot_shape(n: int, n_cols: int = 3):
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


# ------------------------------------------------------------
# 1) Sensitivity engine: compositional perturbation -> prediction
# ------------------------------------------------------------
def sensitivity_compositional_variation(
    df,
    model,
    step_size_dic,
    step_list=(0, 1, 2),
    normalization="no_normalize",
    verbose=False,
    prediction_reducer="mean",  # "mean" | "median" | None (None -> keep raw)
):
    """
    Perform sensitivity analysis by perturbing selected oxides and predicting.

    Notes
    -----
    - This function returns one row per step.
    - By default, prediction is reduced to a scalar (mean) to avoid shape issues.
    """
    results = pd.DataFrame(columns=["index"] + list(df.columns) + ["prediction"])

    # pre-check required columns for your F/Cl correction
    if normalization in ("normalize_to_100", "normalize_to_original_sum"):
        for col in ("F", "Cl"):
            if col not in df.columns:
                raise KeyError(f"normalization='{normalization}' requires column '{col}' in df.")

    for step in step_list:
        wrong_df = df.copy()
        row = {"index": step}

        # apply perturbations
        for key, delta in step_size_dic.items():
            if key in wrong_df.columns:
                wrong_df[key] = wrong_df[key] + delta * step

        # normalization (your F/Cl O2 correction logic preserved)
        if normalization in ("normalize_to_100", "normalize_to_original_sum"):
            total_or = df.sum(axis=1).mean() - (df["F"].mean()/19 + df["Cl"].mean()/33.45) * 8
            total_new = wrong_df.sum(axis=1).mean() - (wrong_df["F"].mean()/19 + wrong_df["Cl"].mean()/33.45) * 8

            if normalization == "normalize_to_100":
                wrong_df = wrong_df.div(total_new, axis=0) * 100
            elif normalization == "normalize_to_original_sum":
                wrong_df = wrong_df.div(total_new, axis=0) * total_or

        # store mean compositions per step (scalar)
        for key in df.columns:
            row[key] = float(wrong_df[key].mean())

        if verbose:
            step_size_df = wrong_df - df
            print(f"step: {step}; variation:\n{step_size_df}")
            print("input:")
            print(wrong_df)
            if "F" in wrong_df.columns and "Cl" in wrong_df.columns:
                print(
                    f"total:{wrong_df.sum(axis=1).mean()}, "
                    f"minus O2 for F and Cl: {wrong_df.sum(axis=1).mean() - (wrong_df['F'].mean()/19 + wrong_df['Cl'].mean()/33.45)*8}"
                )
            print("----------------------")

        pred = model.predict(wrong_df)

        # reduce prediction to scalar by default (strongly recommended)
        pred_arr = np.asarray(pred).astype(float).ravel()
        if prediction_reducer is None:
            row["prediction"] = pred  # risky: may break DataFrame shape
        elif prediction_reducer == "mean":
            row["prediction"] = float(np.nanmean(pred_arr))
        elif prediction_reducer == "median":
            row["prediction"] = float(np.nanmedian(pred_arr))
        else:
            raise ValueError("prediction_reducer must be 'mean', 'median', or None.")

        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

    results = results.set_index("index")
    return results


# ------------------------------------------------------------
# 2) Error transform (your original logic kept)
# ------------------------------------------------------------
def get_transformed_error(known_df, error_dic):
    total = known_df.sum(axis=1).mean() - (known_df["F"].mean()/19 + known_df["Cl"].mean()/33.45)*8

    # NOTE: known_df[key].values[0].mean() is odd; values[0] is scalar already.
    # Keep your intention but make it robust:
    def first_scalar(col):
        v = known_df[col].iloc[0]
        return float(np.asarray(v).mean())

    max_error_dict_input = {
        key: -total*value/(value-total+first_scalar(key)) for key, value in error_dic.items()
    }

    F_scale = 1 - 8/19
    Cl_scale = 1 - 8/33.45

    max_error_dict_input["Cl"] = error_dic["Cl"]*total / (
        total - Cl_scale*first_scalar("Cl") - Cl_scale*error_dic["Cl"]
    )
    max_error_dict_input["F"] = error_dic["F"]*total / (
        total - F_scale*first_scalar("F") - F_scale*error_dic["F"]
    )

    return max_error_dict_input


# ------------------------------------------------------------
# 3) Plot a single feature on a given ax (one panel)
# ------------------------------------------------------------
def plot_feature_sensitivity(results_df, feature: str, ax=None, label_baseline=True, y_error=None):
    """
    Plot one panel: feature (x) vs prediction (y).
    results_df must be the output of sensitivity_compositional_variation().
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    else:
        fig = ax.figure

    x = results_df[feature].astype(float).values
    y = pd.Series(results_df["prediction"]).astype(float).values

    ax.scatter(x, y, s=70, edgecolors="black", linewidths=0.5, alpha=0.85)

    if label_baseline and 0 in results_df.index:
        x0 = float(results_df.loc[0, feature])
        y0 = float(results_df.loc[0, "prediction"])
        ax.scatter([x0], [y0], s=90, edgecolors="black", linewidths=0.5,
                   alpha=0.95, color="C2", label="Baseline ($C_{known}$)")
        ax.axhline(y=y0, color="black", linestyle="--", linewidth=1)

        if y_error is not None:
            y_min = y0 - y_error
            y_max = y0 + y_error
            ax.set_ylim(y_min, y_max)

    ax.set_xlabel(f"{feature} (wt%)")
    ax.set_ylabel("Prediction")
    ax.set_title(feature)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if label_baseline:
        ax.legend(frameon=False)

    return fig, ax


# ------------------------------------------------------------
# 4) One-parameter analysis (compute + store per-element results)
# ------------------------------------------------------------
def one_parameter_compositional_analysis(
    df,
    model,
    interested_oxides,
    acceptable_error_dict,
    n=5,
    normalization="normalize_to_original_sum",
    n_cols=3,
    make_plots=True,
    step_mode="n_steps",          # "n_steps" | "step_size"
    step_size_dict=None,          # dict: {oxide: step_size_in_wt%}
    max_steps=None,               # optional int cap when step_mode="step_size"
):
    """
    step_mode
    ---------
    - "n_steps": (default) keep old behavior: step = Emax/n for each element.
    - "step_size": use user-provided step_size_dict per element (wt%). Step count is auto:
          n_i = ceil(|Emax| / |step_size|)
      If max_steps is not None, cap n_i by max_steps.

    step_size_dict
    --------------
    Dict mapping oxide -> step size in wt% (e.g., {"SiO2": 0.1, "MgO": 0.02})
    """

    element_error_dict = acceptable_error_dict.copy()
    element_list = df.columns.tolist()
    abs_error_dict_zeros = dict(zip(element_list, [0]*len(element_list)))

    results_collection = pd.DataFrame(
        columns=["element", "-max", "+max", "element error", "rate", "score"]
    )

    acceptable_error_dict_inverse = {k: -v for k, v in acceptable_error_dict.items()}

    acc_err = acceptable_error_dict
    acc_err_inv = acceptable_error_dict_inverse
    if normalization == "normalize_to_original_sum":
        acc_err = get_transformed_error(df, acceptable_error_dict)
        acc_err_inv = get_transformed_error(df, acceptable_error_dict_inverse)

    results_by_element = {}

    fig = None
    axs = None
    if make_plots:
        n_rows, n_cols_ = get_subplot_shape(len(interested_oxides), n_cols=n_cols)
        fig, axs = plt.subplots(n_rows, n_cols_, figsize=(n_cols_*5, n_rows*4), dpi=150)
        axs = np.atleast_1d(axs).ravel()
        for j in range(len(interested_oxides), n_rows*n_cols_):
            fig.delaxes(axs[j])

    if step_mode not in ("n_steps", "step_size"):
        raise ValueError("step_mode must be 'n_steps' or 'step_size'.")

    if step_mode == "step_size" and not isinstance(step_size_dict, dict):
        raise ValueError("When step_mode='step_size', step_size_dict must be a dict {oxide: step_size}.")

    for i, key in enumerate(interested_oxides):
        step_size_dic = abs_error_dict_zeros.copy()
        step_size_dic_inv = abs_error_dict_zeros.copy()

        # ---- Decide Emax (transformed) for this element ----
        if key in acc_err:
            Emax_pos = float(acc_err[key])
            Emax_neg = float(acc_err_inv[key])  # should be negative counterpart
        else:
            # fallback Emax if not provided
            Emax_pos = 0.5
            Emax_neg = -0.5

        # ---- Build step size and step list depending on mode ----
        if step_mode == "n_steps":
            # old behavior: fixed steps, variable step size
            step_size_dic[key] = Emax_pos / n
            step_size_dic_inv[key] = Emax_neg / n
            step_list_pos = range(0, n+1)
            step_list_neg = range(1, n+1)

        else:
            # step_mode == "step_size": user chooses step size, steps auto per element
            delta = float(step_size_dict.get(key, np.nan))
            if not np.isfinite(delta) or delta == 0:
                raise ValueError(f"step_size_dict must provide a non-zero finite step size for '{key}'.")

            # steps needed to cover Emax (ceil)
            n_i = int(np.ceil(abs(Emax_pos) / abs(delta)))
            n_i = max(n_i, 1)
            if max_steps is not None:
                n_i = min(n_i, int(max_steps))

            # user step size controls direction via step_list sign
            step_size_dic[key] = abs(delta)          # positive direction
            step_size_dic_inv[key] = -abs(delta)     # negative direction
            step_list_pos = range(0, n_i+1)
            step_list_neg = range(1, n_i+1)

        df_pos = sensitivity_compositional_variation(
            df, model, step_size_dic=step_size_dic,
            step_list=step_list_pos, normalization=normalization,
            prediction_reducer="mean"
        )
        df_neg = sensitivity_compositional_variation(
            df, model, step_size_dic=step_size_dic_inv,
            step_list=step_list_neg, normalization=normalization,
            prediction_reducer="mean"
        )

        df_ = pd.concat([df_pos, df_neg], axis=0).sort_index()
        results_by_element[key] = df_

        # summary metrics
        y0 = float(df_.loc[0, "prediction"])
        ymin = float(df_["prediction"].min())
        ymax = float(df_["prediction"].max())

        elem_err = float(element_error_dict.get(key, np.nan))
        result_dic = {
            "element": key,
            "-max": ymin - y0,
            "+max": ymax - y0,
            "element error": elem_err,
        }
        result_dic["rate"] = (result_dic["+max"] - result_dic["-max"]) / elem_err / 2 if np.isfinite(elem_err) and elem_err != 0 else np.nan
        result_dic["score"] = (result_dic["+max"] - result_dic["-max"])

        results_collection = pd.concat([results_collection, pd.DataFrame(result_dic, index=[key])])

        if make_plots:
            plot_feature_sensitivity(df_, key, ax=axs[i], label_baseline=True)

    if make_plots:
        fig.tight_layout()
        return results_collection, results_by_element, fig, axs[:len(interested_oxides)]

    return results_collection, results_by_element

# ------------------------------------------------------------
# 5) Standalone multi-panel plotting for selected features
# ------------------------------------------------------------
def plot_selected_features_panels(
    results_by_element: dict,
    features,
    n_cols=3,
    title=None,
    y_error  = None,
):
    """
    Plot selected features in panels using results_by_element from one_parameter_compositional_analysis.

    Parameters
    ----------
    results_by_element : dict[str, pd.DataFrame]
        output from one_parameter_compositional_analysis
    features : list[str]
        which features to plot
    n_cols : int
        number of columns in panel grid
    title : str or None
        figure title
    """
    features = list(features)
    n_rows, n_cols = get_subplot_shape(len(features), n_cols=n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4), dpi=150)
    axs = np.atleast_1d(axs).ravel()

    for i, f in enumerate(features):
        if f not in results_by_element:
            raise KeyError(f"Feature '{f}' not found in results_by_element. Available: {list(results_by_element.keys())}")
        plot_feature_sensitivity(results_by_element[f], f, ax=axs[i], label_baseline=True, y_error=y_error)

    # delete extra axes
    for j in range(len(features), n_rows*n_cols):
        fig.delaxes(axs[j])

    if title:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()


    return fig, axs[:len(features)]

# df = sensitivity_compositional_variation(known_df, bt_T_model, abs_error_dict)
# plot_compositional_variation(df, element_list)
