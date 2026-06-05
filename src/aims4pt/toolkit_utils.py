# aims4pt/toolkit_utils.py

'''
A collection of utility functions for data analysis and visualization.

Functions:
    manually_input_sths(input_mode="auto", **kwargs):
        Collect user inputs for various keys using terminal, GUI, or notebook widgets.
    
    set_diy_color_cycle(ax=None, color_cycle=Custom_cycle):
        Set a custom color cycle for a given axis.
    
    ray_searching_curve(results_df, score):
        Plot the ray-searching curve for a given DataFrame of results.
    
    get_file_path(data_file_package, file_name):
        Get the full file path for a given file name in a specified package.
    
    wrap_text(text: str, max_width: int = 15) -> str:
        Reformat text to fit within a specified width.
    
    find_common_id_from_dataframes(dfs, id_column="id"):
        Find common IDs across multiple DataFrames.
    
    BlankScaler:
        A placeholder scaler that does nothing to the data.
'''



import re
import sys
import textwrap
import tkinter as tk
import pickle
import io
from tkinter import simpledialog
import numpy as np


from IPython.display import display
from matplotlib import pyplot as plt

from aims4pt.constants import Custom_cycle


class LegacyPackageUnpickler(pickle.Unpickler):
    """Load pickle files created before the package was renamed."""

    MODULE_PREFIX_MAP = {
        # Historical singular package name -> current plural package name.
        "my_analysis_tools.data_tool.compositions": "aims4pt.data_tools.compositions",
        "my_analysis_tools.data_tool.crust": "aims4pt.data_tools.crust",
        "my_analysis_tools.data_tool.eq_pairing": "aims4pt.data_tools.eq_pairing",
        "my_analysis_tools.data_tool.equilibrium": "aims4pt.data_tools.equilibrium",
        "my_analysis_tools.data_tool.rocks": "aims4pt.data_tools.rocks",
        "my_analysis_tools.data_tool": "aims4pt.data_tools",

        # Current package groups under the old project root.
        "my_analysis_tools.data_tools": "aims4pt.data_tools",
        "my_analysis_tools.model_tools.data": "aims4pt.model_tools.data",
        "my_analysis_tools.model_tools.trained_model": "aims4pt.model_tools.trained_model",
        "my_analysis_tools.model_tools": "aims4pt.model_tools",
        "my_analysis_tools.reporting": "aims4pt.reporting",
        "my_analysis_tools.statistic_tools": "aims4pt.statistic_tools",
        "my_analysis_tools.visualization": "aims4pt.visualization",

        # Top-level utility modules.
        "my_analysis_tools.constants": "aims4pt.constants",
        "my_analysis_tools.ipynb_utils": "aims4pt.ipynb_utils",
        "my_analysis_tools.test_utils": "aims4pt.test_utils",
        "my_analysis_tools.toolkit_utils": "aims4pt.toolkit_utils",
        "my_analysis_tools.utils": "aims4pt.utils",

        # Last-resort fallback for objects pickled against the old project root.
        "my_analysis_tools": "aims4pt",
    }

    def find_class(self, module, name):
        for old_prefix, new_prefix in self.MODULE_PREFIX_MAP.items():
            if module == old_prefix or module.startswith(f"{old_prefix}."):
                module = f"{new_prefix}{module[len(old_prefix):]}"
                break
        return super().find_class(module, name)


def load_pickle_compat(file_obj):
    """Load pickle files, using legacy module remapping only when needed."""
    data = file_obj.read()
    if b"my_analysis_tools" in data:
        return LegacyPackageUnpickler(io.BytesIO(data)).load()
    return pickle.load(io.BytesIO(data))


def manually_input_sths(input_mode="auto", **kwargs):
    """
    General-purpose function to manually input values for various keys.

    Parameters:
        input_mode : str, optional
            The input mode to use. Options:
            - "terminal": Use terminal input.
            - "gui": Use GUI (Tkinter dialog).
            - "notebook": Use Jupyter Notebook widgets.
            - "auto": Automatically detect environment (default).

        kwargs : dict
            Keys and optional default values for input.
            Example: `manually_input_sths(x_label="default_x", y_label=None)`

    Returns:
        dict
            A dictionary containing user inputs for the provided keys.
            Example: `{'x_label': 'default_x', 'y_label': 'user_input'}`
    """
    import ipywidgets as widgets
    # Helper function for terminal input
    def terminal_input(keys_with_defaults):
        inputs = {}
        for key, default_value in keys_with_defaults.items():
            user_input = input(f"Enter value for '{key}' (default: {default_value}): ")
            inputs[key] = user_input if user_input.strip() else default_value
        return inputs

    # Helper function for GUI input
    def gui_input(keys_with_defaults):
        inputs = {}
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        for key, default_value in keys_with_defaults.items():
            prompt = f"Enter value for '{key}' (default: {default_value}):"
            user_input = simpledialog.askstring("Input", prompt)
            inputs[key] = user_input if user_input else default_value
        root.destroy()
        return inputs

    # Helper function for Notebook input
    def notebook_input(keys_with_defaults):
        inputs = {}
        widgets_dict = {}
        submit_button = widgets.Button(description="Submit")

        def on_submit(change):
            for key in keys_with_defaults.keys():
                inputs[key] = widgets_dict[key].value or keys_with_defaults[key]
            print(f"Inputs collected: {inputs}")

        display_widgets = []
        for key, default_value in keys_with_defaults.items():
            widget = widgets.Text(description=key, placeholder=str(default_value))
            widgets_dict[key] = widget
            display_widgets.append(widget)
        display(*display_widgets, submit_button)
        submit_button.on_click(on_submit)

        # Return placeholder values (interactive input handled separately)
        return {key: f"Default({key})" for key in keys_with_defaults}

    # Automatically detect environment
    if input_mode == "auto":
        try:
            get_ipython = sys.modules['IPython'].get_ipython
            if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':  # Jupyter Notebook
                input_mode = "notebook"
            else:
                input_mode = "terminal"
        except:
            input_mode = "terminal"

    # Prepare keys and defaults
    keys_with_defaults = {key: value for key, value in kwargs.items()}

    # Call the appropriate input function
    if input_mode == "terminal":
        return terminal_input(keys_with_defaults)
    elif input_mode == "gui":
        return gui_input(keys_with_defaults)
    elif input_mode == "notebook":
        return notebook_input(keys_with_defaults)
    else:
        raise ValueError("Invalid input_mode. Choose 'terminal', 'gui', 'notebook', or 'auto'.")


def set_diy_color_cycle(ax = None, color_cycle = Custom_cycle):
    """
    Set a custom color cycle for a given axis.

    Parameters:
        ax : matplotlib.axes.Axes, optional
            The axis to set the color cycle for. If not provided, the current axis is used.
        color_cycle : cycler.Cycler, optional
            The custom color cycle to set.
    """
    if ax is None:
        ax = plt.gca()
    ax.set_prop_cycle(color_cycle)



def ray_searching_curve(results_df, score):
    """
    Plot the ray-searching curve for a given DataFrame of results.

    Parameters:
        results_df : pd.DataFrame
            DataFrame containing the results of ray-searching.
        score : str
            The name of the score column to plot.
    """
    best_scores = results_df[score].cummax()

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(range(1, len(best_scores)+1), best_scores.tolist(), marker="o", label="Best Score")
    plt.xlabel("Searching Times")
    plt.ylabel("Best Score")
    plt.title("Searching Times vs Best Score")
    plt.legend()
    plt.grid(True)
    plt.show()

# Dynamically resolve file location
def get_file_path(data_file_package, file_name):
    """
    Get the full file path for a given file name.

    Parameters:
        data_file_package : str
            The package name containing the data file.\n
            'e.g., "aims4pt.model_tools.data.Petrelli20"'
        
        file_name : str
            The name of the data file.

    Returns:
        pathlib.Path
            The full path to the data file.
    """
    import importlib.resources as pkg_resources
    from pathlib import Path

    if file_name is None:
        return None

    self_calibration_path = Path(pkg_resources.files(data_file_package) / file_name)
    return self_calibration_path



import re
import textwrap
from typing import List



def wrap_text(
    text: str,
    max_width: int = 15,
    *,
    allow_word_break: bool = False,
    hyphen_char: str = "-",
) -> str:
    """
    Wrap a single paragraph to lines <= max_width.

    If allow_word_break=True:
      - Prefer breaking long hyphenated words at existing hyphens.
        Example: "clinopyroxene-only" -> "clinopyroxene-\\nonly"
      - If no suitable existing hyphen is available, split the word using
        fixed-width hyphenation.
        Example: "opportunity" with max_width=6 -> "oppor-\\ntunity".

    Notes:
      - This is a simple fixed-width wrapper, NOT dictionary-based hyphenation.
      - Existing hyphens are preserved and are not duplicated.
      - Explicit line breaks are preserved, and each line is wrapped separately.
    """
    if "\n" in text or "\r" in text:
        return "\n".join(
            wrap_text(
                line,
                max_width=max_width,
                allow_word_break=allow_word_break,
                hyphen_char=hyphen_char,
            )
            for line in text.splitlines()
        )

    # --- original cleanup ---
    text = re.sub(r"(\()", r" \1", text)
    text = re.sub(r"\s+", " ", text).strip()

    if max_width <= len(hyphen_char):
        raise ValueError("max_width must be larger than the hyphen length.")

    if not allow_word_break:
        lines = textwrap.wrap(
            text,
            width=max_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        return "\n".join(lines)

    def split_long_word(word: str) -> List[str]:
        """
        Split one long word into chunks <= max_width.

        Priority:
        1. Break after an existing hyphen within max_width.
        2. If the next character is an existing hyphen, include it and break after it.
        3. Otherwise, insert hyphen_char as a fixed-width hyphenation mark.
        """
        parts: List[str] = []
        w = word

        while len(w) > max_width:
            # 1. Prefer breaking after an existing hyphen within max_width.
            #    Example, max_width=14:
            #    clinopyroxene-only -> clinopyroxene- / only
            candidate = w[:max_width]
            hyphen_positions = [
                m.end()
                for m in re.finditer(re.escape(hyphen_char), candidate)
                if m.start() > 0
            ]

            if hyphen_positions:
                break_pos = hyphen_positions[-1]  # position after the existing hyphen
                parts.append(w[:break_pos])
                w = w[break_pos:]
                continue

            # 2. Otherwise use fixed-width splitting.
            take = max_width - len(hyphen_char)

            # If the next character is already a hyphen, include it rather than
            # adding another hyphen or leaving it at the start of the next line.
            if len(w) > take and w[take] == hyphen_char:
                parts.append(w[: take + 1])
                w = w[take + 1:]
                continue

            # If the chunk already ends with a hyphen, do not add another one.
            if w[:take].endswith(hyphen_char):
                parts.append(w[:take])
                w = w[take:]
                continue

            # Normal fixed-width hyphenation.
            parts.append(w[:take] + hyphen_char)
            w = w[take:]

        if w:
            parts.append(w)

        return parts

    words = text.split(" ")
    lines: List[str] = []
    cur = ""

    def flush_cur():
        nonlocal cur
        if cur:
            lines.append(cur)
            cur = ""

    for word in words:
        if not word:
            continue

        # Long word: split cleanly first.
        if len(word) > max_width:
            pieces = split_long_word(word)

            # To avoid awkward partial packing, flush the current line first.
            flush_cur()

            # All pieces except the last are complete wrapped lines.
            lines.extend(pieces[:-1])

            # The last piece can continue to receive following words.
            cur = pieces[-1]
            continue

        # Normal greedy wrapping.
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= max_width:
            cur = cur + " " + word
        else:
            flush_cur()
            cur = word

    flush_cur()
    return "\n".join(lines)

def fmt_num(x: float) -> str:
    """Pretty formatting: integers as int; otherwise 1 decimal.
    Special case: values that round to 0.0 at 1 decimal -> "0".
    """
    if x is None:
        return "NA"

    x = float(x)

    if not np.isfinite(x):
        return "NA"

    # Collapse -0.0 and tiny values to zero.
    if abs(x) < 5e-12:
        return "0"

    # Display near-integers as integers.
    xr = round(x)
    if abs(x - xr) < 1e-9:
        return str(int(xr))

    # If rounding to one decimal gives 0.0 or -0.0, display 0.
    if round(x, 1) == 0.0:
        return "0"

    return f"{x:.1f}"



def find_common_index_from_dataframes(dfs):
    """
    Find common indices across multiple DataFrames.

    Parameters:
        dfs : list or dict of pd.DataFrame
            A list or dictionary of DataFrames to check for common indices.
            If a dictionary is provided, the values should be DataFrames.

    Returns:
        set
            A set of common IDs found in all DataFrames.
    """
    if isinstance(dfs, dict):
        dfs = list(dfs.values())
    
    if not dfs:
        return set()

    common_ids = set(dfs[0].index)
    for df in dfs[1:]:
        common_ids.intersection_update(df.index)

    return common_ids




from sklearn.base import BaseEstimator, TransformerMixin

class BlankScaler(BaseEstimator, TransformerMixin):
    """
    A placeholder scaler that does nothing to the data.
    Useful for maintaining a consistent pipeline interface.
    """
    def fit(self, X, y=None):
        # No operation; return self
        return self

    def transform(self, X):
        # Return data unchanged
        return X

    def inverse_transform(self, X):
        # Return data unchanged
        return X
# Example usage
if __name__ == "__main__":
        # Example 1: Terminal input with default values
    if False:
        inputs = manually_input_sths(input_mode="gui", x_label="Default X", y_label="Default Y", z_value="No Default")
        print(f"Collected inputs: {inputs}")


    # Example 2: Notebook input (will require Jupyter environment)
    if True:
        model_names = [
            "Putirka2008_T-Putirka2008_P (modelA)",
            "A_very_long_model_name (with_multiple_parts ) example",
            "ShortName",
            "Putirka, 2008 T:eq32d_T_hydrousVersion; P:eq32a_P"
        ]

        wrapped_labels = [wrap_text(name) for name in model_names]

        for lbl in wrapped_labels:
            print("----")
            print(lbl)
