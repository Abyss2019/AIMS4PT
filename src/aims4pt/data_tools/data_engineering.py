# aims4pt/data_tools/data_engineering.py
import pandas as pd
import numpy as np
import re
from aims4pt.utils import normalize_column_names, get_oxides_list



# add ratio columns if the dominator is not zero
def _add_ratio(df, oxides):
    '''
    Add ratio columns to the DataFrame.
    
    Parameters:
    -----------
    df (DataFrame): 
        DataFrame to add ratio columns.
    oxides (list): 
        List of oxides.
    '''
    df_copy = df.copy()
    for i in range(len(oxides)):
        for j in range(len(oxides)):
            if i != j:
                if (df_copy[oxides[j]] != 0).all():
                    # remove (WT%) from the column name
                    oxi = oxides[i]
                    oxj = oxides[j]
                    # have inverted ratio?
                    if oxj + '/' + oxi not in df_copy.keys():
                        df_copy[oxi + '/' + oxj] = df_copy[oxides[i]] / df_copy[oxides[j]]
    return df_copy

def add_ratio(df, oxides, random_state=42):
    df = df.copy()
    for i in range(len(oxides)):
        for j in range(len(oxides)):
            if i != j:
                if (df[oxides[j]] != 0).all():
                    oxi = oxides[i].strip()
                    oxj = oxides[j].strip()
                    # have inverted ratio?
                    if oxj + '/' + oxi not in df.keys():
                        df[oxi + '/' + oxj] = df[oxides[i]] / df[oxides[j]]
    return df


def add_ratio_keep_nan_keep_inverse(df, oxides, random_state=42, keep_inverse = True):
    df = df.copy()
    ratio_columns = []
    np.random.seed(random_state)

    # Compute all ratios and store column names
    for i in range(len(oxides)):
        for j in range(len(oxides)):
            if i != j:
                oxi = oxides[i].strip()
                oxj = oxides[j].strip()
                ratio_col = oxi + '/' + oxj
                df[ratio_col] = np.where(df[oxj] != 0, df[oxi] / df[oxj], np.nan)
                ratio_columns.append((ratio_col, oxj + '/' + oxi))

    if keep_inverse:
        return df
    
    # Filter reciprocal pairs
    for ratio1, ratio2 in ratio_columns:
        if ratio2 in df.columns:
            nan_count1 = df[ratio1].isna().sum()
            nan_count2 = df[ratio2].isna().sum()

            # Keep the column with fewer NaN values
            if nan_count1 < nan_count2:
                df.drop(columns=[ratio2], inplace=True)
            elif nan_count1 > nan_count2:
                df.drop(columns=[ratio1], inplace=True)
            else:
                # Randomly drop one if NaN counts are the same
                drop_col = np.random.choice([ratio1, ratio2])
                df.drop(columns=[drop_col], inplace=True)

    return df

def add_ratio_keep_nan_no_inverse(df, oxides, random_state=42, keep_inverse = False):
    df = df.copy()
    ratio_columns = []
    np.random.seed(random_state)

    # Compute all ratios and store column names
    for i in range(len(oxides)):
        for j in range(len(oxides)):
            if i != j:
                oxi = oxides[i].strip()
                oxj = oxides[j].strip()
                ratio_col = oxi + '/' + oxj
                df[ratio_col] = np.where(df[oxj] != 0, df[oxi] / df[oxj], np.nan)
                if i < j:
                    ratio_columns.append((ratio_col, oxj + '/' + oxi))

    if keep_inverse:
        return df
    
    # Filter reciprocal pairs
    for ratio1, ratio2 in ratio_columns:
        if ratio2 in df.columns:
            nan_count1 = df[ratio1].isna().sum()
            nan_count2 = df[ratio2].isna().sum()

            # Keep the column with fewer NaN values
            if nan_count1 < nan_count2:
                df.drop(columns=[ratio2], inplace=True)
            elif nan_count1 > nan_count2:
                df.drop(columns=[ratio1], inplace=True)
            else:
                # Randomly drop one if NaN counts are the same
                drop_col = np.random.choice([ratio1, ratio2])
                df.drop(columns=[drop_col], inplace=True)

    return df

                
def add_ratios_to_dataframe(df, oxides=None):
    '''
    Add ratio columns to the DataFrame.
    Dealing with one phase at a time.
    
    Parameters:
    -----------
    df (pd.DataFrame): 
        DataFrame to add ratio columns, with the original columns.
    oxides (list, optional): 
        List of oxides. If None, it will be generated from the DataFrame columns.
    
    Returns:
    --------
    pd.DataFrame: 
        DataFrame with ratio columns added. The original columns are not changed; 
        the ratio columns will be in the standard names.
    '''
    # Create a copy of the original DataFrame
    df_return = df.copy()
    
    # Standardize column names
    df_standard = normalize_column_names(df)
    
    # Get the list of oxides if not provided
    if oxides is None:
        oxides = get_oxides_list(df_standard.columns.tolist())
    
    # Add ratio columns to the standardized DataFrame
    df_standard_add = _add_ratio(df_standard, oxides)
    
    # Drop the original columns from the DataFrame with added ratios
    df_standard_add = df_standard_add.drop(df_standard.columns, axis=1)
    
    # Merge the ratio columns with the original DataFrame
    df_return = pd.concat([df_return, df_standard_add], axis=1)
    
    return df_return


# training data balancing

# mixture method
def balance_data_mixture(x_train, y_train, random_state=42):
    '''
    Balance the training data by oversampling the minority classes and undersampling the majority classes.
    
    Parameters:
    -----------
    x_train (pd.DataFrame): 
        The features of the training data.
    y_train (pd.Series): 
        The target variable of the training data.
    
    Returns:
    --------
    pd.DataFrame: 
        The balanced features.
    pd.Series: 
        The balanced target variable.
    '''
    pass
    # from imblearn.combine import SMOTETomek

    # sampler = SMOTETomek(random_state= random_state)
    # X_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
    # return X_resampled, y_resampled
