# utils.py
'''
A collection of utility functions for geological data analysis.

Functions:
normalize_column_names: Normalize column names to standard oxide forms and raise errors for ambiguous matches.

merge_phase_dataframes: Merge two DataFrames from different phases and rename columns to include the phase name.

'''

import pandas as pd
import re
from aims4pt.constants import OXIDES_MAPPING as oxides_mapping
from aims4pt.constants import REPORT_INFO as report_info
import numpy as np


def create_column_mapping(standard_names_list):
    """
    Create a column mapping dictionary from standard names and regular expressions.

    Parameters:
    standard_names_list (list): A list of standard names for the columns, in order of preference.

    Returns:
    dict: A dictionary mapping standard names to regular expressions.
    """
    column_mapping = {}
    for name in standard_names_list:
        for base_name, pattern in oxides_mapping.items():
            if re.search(pattern, name, re.IGNORECASE):
                column_mapping[name] = pattern
                break

    if report_info:
        print(f"Column mapping created: {column_mapping}")
    return column_mapping


def normalize_column_names(df, standard_names_list=None, missing_fill=0, drop_missing=False, report_info=report_info):
    """
    Use this carefully, and avoid duplicate entries in the mapping.
    Normalize column names to standard oxide forms and order them according to standard_names_list.
    Logic:
        1. If standard_names_list=None, only normalize the columns and keep all original columns.
        2. If standard_names_list is not None, normalize the columns and keep only columns in
           standard_names_list, preserving that order and filling missing columns.
            If drop_missing=True, do not fill missing columns; drop them instead.

    If not specified, the standard_names_list includes the following oxides:
    "SiO2", "Al2O3", "FeO", "Fe2O3", "MgO", "CaO", "Na2O", "K2O", "TiO2", "MnO", "P2O5",
    and will **keep other columns** which not match the standard names.


    *Warning: Oxides input only (and F), one phase at a time. (e.g. cannot handle 'SiO2_cpx' and 'SiO2_liq' simultaneously)*
    *The output may contain errors. Please review the results carefully.*


    Parameters:

        df (pd.DataFrame or pd.Series or dict):
            Input with potentially non-standard column names.

        standard_names_list (list): 
            List of standard column names to normalize and order.
            - default is None, which includes Oxides: "SiO2", "Al2O3", "FeO", "Fe2O3", "MgO", "CaO", "Na2O", "K2O", "TiO2", "MnO", "P2O5", 
            - "Cr2O3", "NiO", "H2O", "CO2", "BaO", "SrO", "ZrO2", "SO3", "Cl", "F"

        missing_fill (int or float):
            Value to fill for missing columns in the output DataFrame.
            - default is 0.
            - if None, will fill it with NaN.

        drop_missing (bool):
            If True, drop columns that are not in standard_names_list.
            If False, fill missing columns with missing_fill value.


        report_info (bool):
            If True, print information about the column mapping and unmatched columns.
            - default is True, which will print the information.

    ## Limitations:
        - Error/value separation is not automated (e.g., distinguishing 'SiO2' from 'SiO2.error'). 
        You need to manually select the value columns.


    ## Returns:
        pd.DataFrame or pd.Series or dict:
            A DataFrame with standardized and ordered column names.


    """

    df = df.copy()  # Avoid modifying the original DataFrame

    input_type = "DataFrame"
    if not isinstance(df, pd.DataFrame):
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).T
            input_type = "Series"
        elif isinstance(df, dict):
            df = pd.DataFrame(df)
            input_type = "dict"

    column_mapping = create_column_mapping(
        standard_names_list) if standard_names_list else oxides_mapping
    if report_info:
        print("Column mapping:")
        print(column_mapping)
    # add a additional pair to deal with F
    column_mapping['Fe'] = r'(^|[\s_\-\.])Fe([\s_\-\.]|$)'



    new_columns = []
    unmatched_columns = []

    # sp cases for F

    for col in df.columns:
        col = str(col)
        matches = []
        matches = [std for std, pattern in column_mapping.items()
                   if re.search(pattern, col, re.IGNORECASE)]

        # Handling multiple matches
        if len(matches) == 1:
            new_columns.append((col, matches[0]))
            
            
        elif len(matches) > 1:
            # Retain the match with the longest name (most specific)
            max_len = max(len(column_mapping[x]) for x in matches)
            longest_matches = [x for x in matches if len(column_mapping[x]) == max_len]
            if report_info:
                print(
                    f" matches for column '{col}': {matches}. Retaining '{longest_matches}'.")
            for match in longest_matches:
                new_columns.append((col, match))

        else:
            unmatched_columns.append(col)

    if unmatched_columns:
        if report_info:
            print(f"Unmatched columns: {unmatched_columns}")

    if report_info:
        print("Column normalization mapping (original -> new):")
        for orig, new in new_columns:
            print(f"  {orig} -> {new}")

    # Rename columns based on the mapping
    if not standard_names_list:
        new_columns = {orig: new for orig, new in new_columns}
        df = df.rename(columns=new_columns)



    # Reorder columns according to the standard_names_list and fill missing columns with 0
    elif standard_names_list:
        # missing_cols = set(standard_names_list) - set(df.columns)
        new_df = pd.DataFrame(columns=standard_names_list, index=df.index)
        for orig, new in new_columns:
            new_df[new] = df[orig]
        new_df = new_df[standard_names_list]

        if drop_missing:
            # drop missing_cols
            new_df = new_df.dropna(axis=1, how='all')
        else:
            if missing_fill is None:
                missing_fill = np.nan
            with pd.option_context('future.no_silent_downcasting', True):
                new_df = new_df.fillna(missing_fill)
        new_df = new_df.infer_objects(copy=False)
        df = new_df

    # print("Columns normalized and ordered.")
    # print(df.columns)
    if input_type == "Series":
        df = df.iloc[0]
    elif input_type == "dict":
        df = df.iloc[0].to_dict()

    # print original vs new columns

    return df


def filter_oxides(standard_columns):
    """
    find the oxides in the standard columns.
    Parameters:
    standard_columns (list): List of standard column names which is normalized and ordered.

    Returns:
        list: A list of oxides extracted from the standard column names.
    """
    oxides = []
    for col in standard_columns:
        for base_name, pattern in oxides_mapping.items():
            if re.search(pattern, col):
                oxides.append(col)
                break
    return oxides




def get_oxides_list(list_or_df):
    '''
    Get standard oxides list from the a list of oxides_names.

    Parameters:
        list_or_df (list or pd.DataFrame): 
        List of standard column names or DataFrame with columns.

    Returns:
        list: 
        A list of oxides extracted from the standard column names.
    '''

    # create fake dataframe
    if isinstance(list_or_df, list):
        df = pd.DataFrame(columns=list_or_df)
    elif isinstance(list_or_df, pd.DataFrame):
        df = list_or_df.copy()
    # standardize column names into oxides
    standard_df = normalize_column_names(df)
    # keep the oxides

    return filter_oxides(standard_df.columns.tolist())

if __name__ == "__main__":
    
    standard_names_list = ['SiO2.n.',   'F', 'FeO',  'f8098', "Fe2O3", 'Fe', 'Fm', 'Al2O3_cpx', 'FeOt_cpx', 'FeOtot_cpx', 'FeO tot_cpx',]

    #
    oxides = get_oxides_list(standard_names_list)
    print(oxides)

def merge_phase_dataframes(df1, phase1, df2, phase2):
    """
    Merge two DataFrames from different phases and rename columns to include the phase name.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame with element compositions.
    phase1 (str): The name of the phase for df1 (e.g., 'bt', 'melt').
    df2 (pd.DataFrame): The second DataFrame with element compositions.
    phase2 (str): The name of the phase for df2 (e.g., 'bt', 'melt').

    Returns:
    pd.DataFrame: A merged DataFrame with columns renamed to {phase}_{composition}.
    """
    # Add phase prefix to column names
    df1_renamed = df1.rename(columns=lambda x: f"{phase1}_{x}")
    df2_renamed = df2.rename(columns=lambda x: f"{phase2}_{x}")

    # Merge the two DataFrames along columns (axis=1)
    merged_df = pd.concat([df1_renamed, df2_renamed], axis=1)

    return merged_df

    # Example usage

    # Example usage:
    # df1 = pd.DataFrame({'SiO2': [50.3, 51.2], 'TiO2': [0.5, 0.6]})
    # df2 = pd.DataFrame({'SiO2': [48.3, 49.0], 'TiO2': [0.4, 0.5]})
    # merged_df = merge_phase_dataframes(df1, 'bt', df2, 'melt')


def export_min_max_range_df(X_df: pd.DataFrame):
    """
    Export a DataFrame with minimum and maximum values for specified columns.

    Parameters:
        X_df (pd.DataFrame): Input DataFrame.
        col_list (list): List of column names to calculate min and max.

    Returns:
        pd.DataFrame: A DataFrame with columns 'min' and 'max' for each specified column.
    """
    if not isinstance(X_df, pd.DataFrame):
        raise TypeError("X_df must be a pandas DataFrame.")

    
    min_max_df = pd.DataFrame(columns=X_df.columns)
    min_max_df["Index"] = ["min", "max"]
    min_max_df = min_max_df.set_index("Index")
    
    for col in X_df.columns:
        if X_df[col].dtype in [np.float64, np.int64]:
            min_value = X_df[col].min()
            max_value = X_df[col].max()
            min_max_df.loc["min", col] = min_value
            min_max_df.loc["max", col] = max_value
    
    return min_max_df



if __name__ == "__main__":
    # test

    data = {'SiO2_cpx': ["SiO2"],
            'FeO': ["FeO"], 'Fe2O3': ["Fe2O3"], "F_cpx": ["F"],
            'MgO': ["MgO"],  
            'TiO2': ["TiO2"],
            "clinopyroxene_Al2O3.1": ["Al2O3"], 
            "Na2O": ["Na2O"], 
            "K2O": ["K2O"], 
            "MnO": ["MnO"], 
            "P2O5": ["P2O5"],
            # Noise column used to test matching.
            "First": ["First"],
    }
    df = pd.DataFrame(data)
    standard_names_list = ["F_clinopyroxene",  "Fe2O3", "FeO", "Fe", "Fm",
                           "Al2O3_cpx", "FeOt_cpx", "FeOtot_cpx", "FeO tot_cpx"]

    # Normalize column names.
    normalized_df = normalize_column_names(
        df, standard_names_list=standard_names_list, report_info=True, drop_missing=False)
    print(normalized_df)
