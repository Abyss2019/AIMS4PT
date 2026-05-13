import pandas as pd
import numpy as np
import pickle
import warnings
from pandas.errors import SettingWithCopyWarning

#### PREPROCESSING ####

def preprocessing_cpx(df, suppress_warnings=False):
    # optionally suppress runtime and pandas warnings
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    # drop empty columns
    #df = df.dropna(axis=1, how='all')
    df_1 = df.copy()

    # molecular weight
    mw = {'SiO2': 60.084, 'TiO2': 79.900, 'Al2O3': 101.960, 'Cr2O3': 151.990, 'FeO': 71.846, 'MnO': 70.937,
          'NiO': 74.699, 'MgO': 40.304, 'CaO': 56.079, 'Na2O': 61.979, 'K2O': 94.196, 'Fe2O3': 159.69}
    mw_arr = np.array(list(mw.values()))

    # moles of oxides
    mole_ox = np.array([1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2])

    # compute the correction factor and multiply by it (handle divide-by-zero safely)
    cols = df.columns[4:-1]
    non_numeric = df[cols].map(lambda x: not isinstance(x, (int, float)))
    if non_numeric.any().any():
        print("Non-numeric values found in the following columns:")
        print(non_numeric.loc[:, non_numeric.any()])
    # convert to moles and basis
    df_1[cols] = df[cols] / mw_arr[:-1] * mole_ox
    # safe correction factor: avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sums = df_1[cols].sum(axis=1)
        corr_fact = 4 / sums
    # replace infinities or invalid with NaN
    corr_fact = pd.Series(corr_fact, index=df_1.index).where(pd.notnull(corr_fact), np.nan)
    # apply correction factor
    df_1[cols] = df_1[cols].multiply(corr_fact, axis=0)

    # compute the charge and the difference
    charges_of_column = np.array([4, 4, 3, 3, 2, 2, 2, 2, 2, 1, 1])
    charge = (df_1[df.columns[4:-1]] * charges_of_column).sum(axis=1)
    difference = 12 - charge

    bol = df_1['FeOt_cpx'] < difference
    df_Fe = pd.DataFrame(columns=['Fe2', 'Fe3'], index=df_1.index)

    df_Fe['Fe3'][df_1['FeOt_cpx'] < difference] = df_1['FeOt_cpx'].copy()
    df_Fe['Fe2'][df_1['FeOt_cpx'] < difference] = 0

    df_Fe['Fe3'][df_1['FeOt_cpx'] >= difference] = difference
    df_Fe['Fe2'][df_1['FeOt_cpx'] >= difference] = df_1['FeOt_cpx'].copy() - difference
    
    #  corr vr. 1.1
    df_Fe['Fe3'][difference < 0] = 0
    df_Fe['Fe2'][difference < 0] = df_1['FeOt_cpx'].copy()
   
    df_Fe = df_Fe.astype('float64')

    #  We define a new oxide dataframe with different columns for the two Fe and
    #  we calculate the sum over all oxides (to be used in the check)

    df_ox = df.copy()
    df_ox['Fe2O3'] = df_Fe['Fe3'] / corr_fact * mw['Fe2O3'] / 2
    df_ox['FeO'] = df_Fe['Fe2'] / corr_fact * mw['FeO']
    df_ox = pd.concat([df_ox[df_ox.columns[:8]], df_ox[df_ox.columns[-2:]], df_ox[df_ox.columns[9:-2]]], axis=1)
    df_ox['tot'] = df_ox[df_ox.columns[4:-1]].sum(axis=1)

    # We define a new dataframe for cations
   
    
    df_cat = pd.concat([df_1[df_1.columns[:8]], df_Fe[['Fe3', 'Fe2']], df_1[df_1.columns[9:]]], axis=1)
    old = df_cat.columns[4:-1]
    new = ['Si', 'Ti', 'Al', 'Cr', 'Fe3', 'Fe2', 'Mn', 'Ni', 'Mg', 'Ca', 'Na', 'K']
    rename_dic = {}
    for i in range(12):
        rename_dic[old[i]] = new[i]
    df_cat = df_cat.rename(columns=rename_dic)
    charges_of_column_new = np.array([4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1])
    charge_balanced = (df_cat[df_cat.columns[4:-1]] * charges_of_column_new).sum(axis=1)

    # define dataframe for sites

    df_T = pd.DataFrame(columns=['Si', 'Al', 'Ti', 'Fe3'], index=df_cat.index)
    df_M = pd.DataFrame(columns=['Mg', 'Fe2', 'Fe3', 'Al', 'Ti', 'Cr', 'Ni', 'Mn', 'Ca', 'Na', 'K'], index=df_cat.index)


    df_T['Si'] = df_cat['Si']
    df_T['Al'][df_T['Si'] + df_cat['Al'] >= 2] = 2 - df_T['Si']
    df_T['Al'][df_T['Si'] >= 2] = 0
    df_T['Al'][df_T['Si'] + df_cat['Al'] < 2] = df_cat['Al']
    df_T['Ti'][df_T['Si'] + df_T['Al'] + df_cat['Ti'] >= 2] = 2 - (df_T['Si'] + df_T['Al'])
    df_T['Ti'][df_T['Si'] + df_T['Al'] >= 2] = 0
    df_T['Ti'][df_T['Si'] + df_T['Al'] + df_cat['Ti'] < 2] = df_cat['Ti']
    df_T['Fe3'][df_T['Si'] + df_T['Al'] + df_T['Ti'] + df_cat['Fe3'] >= 2] = 2 - (df_T['Si'] + df_T['Al'] + df_T['Ti'])
    df_T['Fe3'][df_T['Si'] + df_T['Al'] + df_T['Ti'] + df_cat['Fe3'] < 2] = df_cat['Fe3']
    df_T['Fe3'][df_T['Si'] + df_T['Al'] + df_T['Ti'] >= 2] = 0

    df_M = df_cat[df_M.columns]
    df_M[['Al', 'Fe3', 'Ti']] = df_M[['Al', 'Fe3', 'Ti']] - df_T[['Al', 'Fe3', 'Ti']]

    # define dataframe of classification

    df_class = pd.DataFrame(columns=['Fs', 'Wo', 'En', 'Q', 'J'])
    den = df_M[['Fe2', 'Fe3', 'Mn', 'Mg', 'Ca']].sum(axis=1)

    den = den.replace(0, np.nan)  # Avoid division by zero
    
    df_class['Fs'] = df_M[['Fe2', 'Fe3', 'Mn']].sum(axis=1) / den * 100
    df_class['Wo'] = df_M['Ca'] / den * 100
    df_class['En'] = df_M['Mg'] / den * 100
    df_class['Q'] = df_M[['Fe2', 'Ca', 'Mg']].sum(axis=1)
    df_class['J'] = df_M[['Na']].sum(axis=1) * 2

    # compute the components and define the relative dataframe
    # we copy the sites dataframes because we have to update them during the computation

    M = df_M.copy()
    M = M.fillna(0)
    T = df_T.copy()
    if len(T) != len(M):
        raise ValueError("The number of rows in T and M dataframes do not match. Please check the input data.")

    colcomp = ['CaTiAl2O6', 'CaTs', 'Es', 'CaCrTs', 'NaCrSi2O6', 'Jd', 'Ae', 'Di', 'Hd', 'En(Mg+Ni)', 'Fs(Fe+Mn)']
    df_comp = pd.DataFrame(columns=['Index', 'sample', 'notes', 'notes.1'] + colcomp, index=df.index)

    # computation of first component ('CaTiAl2O6')

    df_comp[colcomp[0]][T[['Al', 'Fe3']].sum(axis=1) >= 2 * M['Ti']] = M['Ti']
    df_comp[colcomp[0]][T[['Al', 'Fe3']].sum(axis=1) < 2 * M['Ti']] = T[['Al', 'Fe3']].sum(axis=1) / 2

    # sites dataframe updating

    T['Si'] = T['Si'] + T['Ti']
    T['Al'] = T['Al'] + T['Fe3'] - 2 * df_comp[colcomp[0]]

    M['Ti'] = M['Ti'] - df_comp[colcomp[0]]
    M['Ca'] = M['Ca'] - df_comp[colcomp[0]]
    if len(T) != len(M):
        warnings.warn(
            "预警: T 和 M 的行数不一致，结果可能不可靠，但继续运行。",
            UserWarning
        )
    # computation of 2nd, 3th, 4th and 5th components ('CaTs','Es','CaCrTs','NaCrSi2O6')

    R1 = pd.Series(0.0, index=M.index)
    valid = M['Al'] > 0
    R1.loc[valid] = M.loc[valid, 'Fe3'] / M.loc[valid, 'Al']
    assert len(R1) == len(T), f"R1 与 T 行数不一致: {len(R1)} vs {len(T)}"


    if len(T) != len(M):
        warnings.warn(
            "预警: T 和 M 的行数不一致（后续检查），继续运行。",
            UserWarning
        )
    # if T.index.equals(M.index):
    #     raise ValueError(f"The indices of T and M dataframes do not match. T.index: {T.index}, M.index: {M.index}")
    
    if T.isnull().any().any():
        warnings.warn(
            f'''T missing values:\n{T.isnull().sum()}
              M missing values:\n{M.isnull().sum()}''',
            UserWarning
        )
    try:
        M_al = M['Al']
        T_al = T['Al'] / (R1 + 1)

        mask = M_al.to_numpy() > T_al.to_numpy()
        df_comp.loc[mask, colcomp[1]] = T_al[mask]
    except ValueError as e:
        # 可选：列出前几条不一致的行
        print(">> R1.shape:", R1.shape)

    if len(T) != len(M):
        warnings.warn(
            "预警: T 和 M 的行数不一致，再次检查，继续运行。",
            UserWarning
        )
    df_comp[colcomp[1]][M['Al'] <= T['Al'] / (R1 + 1)] = M['Al']

    df_comp[colcomp[2]][M['Fe3'] > T['Al'] - df_comp[colcomp[1]]] = T['Al'] - df_comp[colcomp[1]]
    df_comp[colcomp[2]][M['Fe3'] <= T['Al'] - df_comp[colcomp[1]]] = M['Fe3']

    df_comp[colcomp[3]][M['Cr'] > T['Al'] - (df_comp[colcomp[2]] + df_comp[colcomp[1]])] = T['Al'] - df_comp[colcomp[2]] - \
                                                                                           df_comp[colcomp[1]]
    df_comp[colcomp[3]][M['Cr'] <= T['Al'] - (df_comp[colcomp[2]] + df_comp[colcomp[1]])] = M['Cr']

    df_comp[colcomp[4]][M['Cr'] - df_comp[colcomp[3]] > M['Na']] = M['Na']
    df_comp[colcomp[4]][M['Cr'] - df_comp[colcomp[3]] <= M['Na']] = M['Cr'] - df_comp[colcomp[3]]

    # sites dataframe updating

    T['Si'] = T['Si'] - (df_comp[colcomp[1]] + df_comp[colcomp[2]] + df_comp[colcomp[3]] + 2 * df_comp[colcomp[4]])
    T['Al'] = T['Al'] - (df_comp[colcomp[1]] + df_comp[colcomp[2]] + df_comp[colcomp[3]])

    M['Fe3'] = M['Fe3'] - df_comp[colcomp[2]]
    M['Al'] = M['Al'] - df_comp[colcomp[1]]
    M['Cr'] = M['Cr'] - df_comp[colcomp[3]] - df_comp[colcomp[4]]
    M['Ca'] = M['Ca'] - df_comp[colcomp[1]] - df_comp[colcomp[2]] - df_comp[colcomp[3]]
    M['Na'] = M['Na'] - df_comp[colcomp[4]]

    # computation of 6th and 7th components ('Jd','Ae')

    df_comp[colcomp[5]][M['Al'] > M['Na'] / (R1 + 1)] = M['Na'] / (R1 + 1)
    df_comp[colcomp[5]][M['Al'] <= M['Na'] / (R1 + 1)] = M['Al']

    df_comp[colcomp[6]][M['Fe3'] > M['Na'] - df_comp[colcomp[5]]] = M['Na'] - df_comp[colcomp[5]]
    df_comp[colcomp[6]][M['Fe3'] <= M['Na'] - df_comp[colcomp[5]]] = M['Fe3']

    # sites dataframe updating

    T['Si'] = T['Si'] - 2 * (df_comp[colcomp[6]] + df_comp[colcomp[5]])

    M['Fe3'] = M['Fe3'] - df_comp[colcomp[6]]
    M['Al'] = M['Al'] - df_comp[colcomp[5]]
    M['Na'] = M['Na'] - (df_comp[colcomp[6]] + df_comp[colcomp[5]])

    # computation of 8th and 9th components ('Di','Hd')

    BOOL1 = pd.DataFrame(columns=['Is_cpx'], index=df.index)
    BOOL1[M['Ca'] > 0] = True
    BOOL1[M['Ca'] <= 0] = False

    R2 = pd.Series(0.0, index=M.index)
    valid2 = M['Mg'] > 0
    R2.loc[valid2] = M.loc[valid2, 'Fe2'] / M.loc[valid2, 'Mg']
    assert len(R2) == len(T), f"R2 与 T 行数不一致: {len(R2)} vs {len(T)}"


    df_comp[colcomp[7]][M['Mg'] > M['Ca'] / (R2 + 1)] = M['Ca'] / (R2 + 1)
    df_comp[colcomp[7]][M['Mg'] <= M['Ca'] / (R2 + 1)] = M['Mg']
    df_comp[colcomp[7]][M['Ca'] <= 0] = 0

    df_comp[colcomp[8]][M['Fe2'] > M['Ca'] - df_comp[colcomp[7]]] = M['Ca'] - df_comp[colcomp[7]]
    df_comp[colcomp[8]][M['Fe2'] <= M['Ca'] - df_comp[colcomp[7]]] = M['Fe2']
    df_comp[colcomp[8]][M['Ca'] <= 0] = 0

    # sites dataframe updating

    T['Si'] = T['Si'] - 2 * (df_comp[colcomp[7]] + df_comp[colcomp[8]])

    M['Mg'] = M['Mg'] - df_comp[colcomp[7]]
    M['Fe2'] = M['Fe2'] - df_comp[colcomp[8]]
    M['Ca'] = M['Ca'] - df_comp[colcomp[8]] - df_comp[colcomp[7]]

    # computation of 10th and 11th components ('En(Mg+Ni)','Fs(Fe+Mn)')

    df_comp[colcomp[9]] = (M['Mg'] + M['Ni']) / 2
    df_comp[colcomp[10]] = (M['Fe2'] + M['Mn']) / 2

    # sites dataframe updating

    T['Si'] = T['Si'] - 2 * (df_comp[colcomp[9]] + df_comp[colcomp[10]])

    Mg_old = M['Mg'].copy()
    Fe2_old = M['Fe2'].copy()

    M['Mg'] = M['Mg'] - (2 * df_comp[colcomp[9]] - M['Ni'])
    M['Fe2'] = M['Fe2'] - (2 * df_comp[colcomp[10]] - M['Mn'])

    M['Ni'] = M['Ni'] - (2 * df_comp[colcomp[9]] - Mg_old)
    M['Mn'] = M['Mn'] - (2 * df_comp[colcomp[10]] - Fe2_old)
    
    
    
    sum_comp = df_comp.sum(axis=1).astype(float).round(3)

    # we define and evaluate the checks

    df_ck = pd.DataFrame(
        columns=['Wo', 'J', 'Fs', 'Wt%', 'components', 'Si apfu', 'CaTiAl2O6', 'T_site', 'M_site', 'charge'],
        index=df.index)

    df_ck['Wo'] = (df_class['Wo'] > 20) & (df_class['Wo'] < 55)
    df_ck['J'] = (df_class['J'] < 1)
    df_ck['Fs'] = (df_class['Fs'] > 5) & (df_class['Fs'] < 50)
    df_ck['Wt%'] = (df_ox['tot'] > 97.5) & (df_ox['tot'] < 102.5)
    df_ck['components'] = (sum_comp > 0.95) & (sum_comp < 1.05)
    df_ck['Si apfu'] = (df_T['Si'] <= 2)
    df_ck['CaTiAl2O6'] = df_comp['CaTiAl2O6'] >= 0
    df_ck['T_site'] = (df_T.sum(axis=1) > 1.95) & (df_T.sum(axis=1) < 2.05)
    df_ck['M_site'] = (df_M.sum(axis=1) > 1.95) & (df_M.sum(axis=1) < 2.05)
    df_ck['charge'] = (charge_balanced > 11.95) & (charge_balanced < 12.05)

    # Global check, if it is False, the element cannot be used in the prediction model
    df_ck['cpx_selection'] = (df_ck == True).all(axis=1)

    # We round the datasets to work with a suitable number of significant digits
    df_comp1 = df_comp.astype('float64').round(3)
    df_comp[['Index', 'sample', 'notes', 'notes.1']]= df_cat[['Index', 'sample', 'notes', 'notes.1']]
    df_comp1[['Index', 'sample', 'notes', 'notes.1']]= df_cat[['Index', 'sample', 'notes', 'notes.1']]
    df_cat = df_cat.round(3)
    df_T = df_T.astype('float64').round(3)
    df_M = df_M.astype('float64').round(3)
    df_class = df_class.round(2)

    # the output dictionary is defined
    output_dictionary = {'components': df_comp1, 'cations': df_cat, 'checks': df_ck, 'classifications': df_class,
                         'site_T': df_T, 'site_M1&2': df_M, 'sum_of_components': sum_comp, 'input_NN': df_comp, 'major':df_ox.round(2)}

    return output_dictionary



def preprocessing_cpx(df, suppress_warnings=False):
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    df_1 = df.copy()

    mw = {'SiO2': 60.084, 'TiO2': 79.900, 'Al2O3': 101.960, 'Cr2O3': 151.990, 'FeO': 71.846, 'MnO': 70.937,
          'NiO': 74.699, 'MgO': 40.304, 'CaO': 56.079, 'Na2O': 61.979, 'K2O': 94.196, 'Fe2O3': 159.69}
    mw_arr = np.array(list(mw.values()))
    mole_ox = np.array([1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2])

    cols = df.columns[4:-1]
    non_numeric = df.loc[:, cols].map(lambda x: not isinstance(x, (int, float)))
    if non_numeric.any().any():
        print("Non-numeric values found in the following columns:")
        print(non_numeric.loc[:, non_numeric.any()])

    df_1.loc[:, cols] = df.loc[:, cols] / mw_arr[:-1] * mole_ox

    with np.errstate(divide='ignore', invalid='ignore'):
        sums = df_1.loc[:, cols].sum(axis=1)
        corr_fact = 4 / sums
    corr_fact = pd.Series(corr_fact, index=df_1.index).where(pd.notnull(corr_fact), np.nan)

    df_1.loc[:, cols] = df_1.loc[:, cols].multiply(corr_fact, axis=0)

    charges_of_column = np.array([4, 4, 3, 3, 2, 2, 2, 2, 2, 1, 1])
    charge = (df_1.loc[:, df.columns[4:-1]] * charges_of_column).sum(axis=1)
    difference = 12 - charge

    mask_lt = df_1['FeOt_cpx'] < difference
    mask_ge = ~mask_lt
    mask_neg = difference < 0

    df_Fe = pd.DataFrame(columns=['Fe2', 'Fe3'], index=df_1.index)
    df_Fe.loc[:, ['Fe2', 'Fe3']] = np.nan

    df_Fe.loc[mask_lt, 'Fe3'] = df_1.loc[mask_lt, 'FeOt_cpx']
    df_Fe.loc[mask_lt, 'Fe2'] = 0

    df_Fe.loc[mask_ge, 'Fe3'] = difference.loc[mask_ge]
    df_Fe.loc[mask_ge, 'Fe2'] = df_1.loc[mask_ge, 'FeOt_cpx'] - difference.loc[mask_ge]

    df_Fe.loc[mask_neg, 'Fe3'] = 0
    df_Fe.loc[mask_neg, 'Fe2'] = df_1.loc[mask_neg, 'FeOt_cpx']

    df_Fe = df_Fe.astype('float64')

    df_ox = df.copy()
    df_ox['Fe2O3'] = df_Fe['Fe3'] / corr_fact * mw['Fe2O3'] / 2
    df_ox['FeO'] = df_Fe['Fe2'] / corr_fact * mw['FeO']
    df_ox = pd.concat([df_ox.iloc[:, :8], df_ox.iloc[:, -2:], df_ox.iloc[:, 9:-2]], axis=1)
    df_ox['tot'] = df_ox.iloc[:, 4:-1].sum(axis=1)

    df_cat = pd.concat([df_1.iloc[:, :8], df_Fe[['Fe3', 'Fe2']], df_1.iloc[:, 9:]], axis=1)

    old = df_cat.columns[4:-1]
    new = ['Si', 'Ti', 'Al', 'Cr', 'Fe3', 'Fe2', 'Mn', 'Ni', 'Mg', 'Ca', 'Na', 'K']
    rename_dic = {old[i]: new[i] for i in range(12)}
    df_cat = df_cat.rename(columns=rename_dic)

    charges_of_column_new = np.array([4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1])
    charge_balanced = (df_cat.loc[:, df_cat.columns[4:-1]] * charges_of_column_new).sum(axis=1)

    index = df.index
    n = len(index)

    numeric_cols = ['Si', 'Ti', 'Al', 'Cr', 'Fe3', 'Fe2', 'Mn', 'Ni', 'Mg', 'Ca', 'Na', 'K']
    cat_numeric = df_cat[numeric_cols].to_numpy(dtype=np.float64, copy=True)

    si = cat_numeric[:, 0]
    ti_cat = cat_numeric[:, 1]
    al_cat = cat_numeric[:, 2]
    cr_cat = cat_numeric[:, 3]
    fe3_cat = cat_numeric[:, 4]
    fe2_cat = cat_numeric[:, 5]
    mn_cat = cat_numeric[:, 6]
    ni_cat = cat_numeric[:, 7]
    mg_cat = cat_numeric[:, 8]
    ca_cat = cat_numeric[:, 9]
    na_cat = cat_numeric[:, 10]
    k_cat = cat_numeric[:, 11]

    T_si = si.copy()
    si_plus_al = si + al_cat
    T_al = np.where(si >= 2, 0, np.where(si_plus_al < 2, al_cat, 2 - si))

    si_plus_al = T_si + T_al
    mask_ti_sum_ge2 = si_plus_al + ti_cat >= 2
    mask_ti_sa_ge2 = si_plus_al >= 2
    T_ti = np.where(mask_ti_sa_ge2, 0, np.where(mask_ti_sum_ge2, 2 - si_plus_al, ti_cat))

    base_sum = T_si + T_al + T_ti
    mask_base_ge2 = base_sum >= 2
    mask_sum_plus_fe3_lt2 = base_sum + fe3_cat < 2
    T_fe3 = np.where(mask_base_ge2, 0, np.where(mask_sum_plus_fe3_lt2, fe3_cat, 2 - base_sum))

    T_vals = np.column_stack([T_si, T_al, T_ti, T_fe3])

    M_vals = np.column_stack([
        mg_cat,
        fe2_cat,
        fe3_cat,
        al_cat,
        ti_cat,
        cr_cat,
        ni_cat,
        mn_cat,
        ca_cat,
        na_cat,
        k_cat,
    ])

    M_vals[:, 3] -= T_al
    M_vals[:, 2] -= T_fe3
    M_vals[:, 4] -= T_ti

    M_for_class = M_vals.copy()
    den = M_for_class[:, 1] + M_for_class[:, 2] + M_for_class[:, 7] + M_for_class[:, 0] + M_for_class[:, 8]
    den[den == 0] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        fs = (M_for_class[:, 1] + M_for_class[:, 2] + M_for_class[:, 7]) / den * 100
        wo = M_for_class[:, 8] / den * 100
        en = M_for_class[:, 0] / den * 100
    q_val = M_for_class[:, 1] + M_for_class[:, 8] + M_for_class[:, 0]
    j_val = M_for_class[:, 9] * 2
    df_class = pd.DataFrame({
        'Fs': fs,
        'Wo': wo,
        'En': en,
        'Q': q_val,
        'J': j_val,
    }, index=index)

    comp_vals = np.full((n, 11), np.nan, dtype=np.float64)

    def filled(arr):
        return np.nan_to_num(arr, nan=0.0, copy=True)

    M_fill = filled(M_vals)
    sum_al_fe3 = np.nansum(T_vals[:, [1, 3]], axis=1)
    comp0 = np.where(sum_al_fe3 >= 2 * M_fill[:, 4], M_fill[:, 4], sum_al_fe3 / 2)
    comp_vals[:, 0] = comp0

    T_vals[:, 0] = T_vals[:, 0] + T_vals[:, 2]
    T_vals[:, 1] = T_vals[:, 1] + T_vals[:, 3] - 2 * comp0
    M_vals[:, 4] -= comp0
    M_vals[:, 8] -= comp0

    M_fill = filled(M_vals)
    R1 = np.zeros(n, dtype=np.float64)
    valid_al = M_fill[:, 3] > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        R1[valid_al] = M_fill[valid_al, 2] / M_fill[valid_al, 3]
    target_cats = np.divide(T_vals[:, 1], R1 + 1, out=np.zeros(n, dtype=np.float64), where=(R1 + 1) != 0)
    comp1 = np.where(M_fill[:, 3] > target_cats, target_cats, M_fill[:, 3])
    comp_vals[:, 1] = comp1

    diff_Ta_c1 = T_vals[:, 1] - comp1
    comp2 = np.where(M_fill[:, 2] > diff_Ta_c1, diff_Ta_c1, M_fill[:, 2])
    comp_vals[:, 2] = comp2

    diff_Ta_c12 = T_vals[:, 1] - comp2 - comp1
    comp3 = np.where(M_fill[:, 5] > diff_Ta_c12, diff_Ta_c12, M_fill[:, 5])
    comp_vals[:, 3] = comp3

    cr_minus_comp3 = M_fill[:, 5] - comp3
    comp4 = np.where(cr_minus_comp3 > M_fill[:, 9], M_fill[:, 9], cr_minus_comp3)
    comp_vals[:, 4] = comp4

    T_vals[:, 0] = T_vals[:, 0] - (comp1 + comp2 + comp3 + 2 * comp4)
    T_vals[:, 1] = T_vals[:, 1] - (comp1 + comp2 + comp3)
    M_vals[:, 2] -= comp2
    M_vals[:, 3] -= comp1
    M_vals[:, 5] -= (comp3 + comp4)
    M_vals[:, 8] -= (comp1 + comp2 + comp3)
    M_vals[:, 9] -= comp4

    M_fill = filled(M_vals)
    target_jd = np.divide(M_fill[:, 9], R1 + 1, out=np.zeros(n, dtype=np.float64), where=(R1 + 1) != 0)
    comp5 = np.where(M_fill[:, 3] > target_jd, target_jd, M_fill[:, 3])
    comp_vals[:, 5] = comp5

    comp6 = np.where(M_fill[:, 2] > (M_fill[:, 9] - comp5), M_fill[:, 9] - comp5, M_fill[:, 2])
    comp_vals[:, 6] = comp6

    T_vals[:, 0] = T_vals[:, 0] - 2 * (comp6 + comp5)
    M_vals[:, 2] -= comp6
    M_vals[:, 3] -= comp5
    M_vals[:, 9] -= (comp6 + comp5)

    M_fill = filled(M_vals)
    R2 = np.zeros(n, dtype=np.float64)
    valid_mg = M_fill[:, 0] > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        R2[valid_mg] = M_fill[valid_mg, 1] / M_fill[valid_mg, 0]
    target_di = np.divide(M_fill[:, 8], R2 + 1, out=np.zeros(n, dtype=np.float64), where=(R2 + 1) != 0)
    comp7 = np.where(M_fill[:, 0] > target_di, target_di, M_fill[:, 0])
    comp7 = np.where(M_fill[:, 8] <= 0, 0, comp7)
    comp_vals[:, 7] = comp7

    comp8 = np.where(M_fill[:, 1] > (M_fill[:, 8] - comp7), M_fill[:, 8] - comp7, M_fill[:, 1])
    comp8 = np.where(M_fill[:, 8] <= 0, 0, comp8)
    comp_vals[:, 8] = comp8

    T_vals[:, 0] = T_vals[:, 0] - 2 * (comp7 + comp8)
    M_vals[:, 0] -= comp7
    M_vals[:, 1] -= comp8
    M_vals[:, 8] -= (comp8 + comp7)

    M_fill = filled(M_vals)
    comp9 = (M_fill[:, 0] + M_fill[:, 6]) / 2
    comp10 = (M_fill[:, 1] + M_fill[:, 7]) / 2
    comp_vals[:, 9] = comp9
    comp_vals[:, 10] = comp10

    T_vals[:, 0] = T_vals[:, 0] - 2 * (comp9 + comp10)
    Mg_old = M_vals[:, 0].copy()
    Fe2_old = M_vals[:, 1].copy()
    M_vals[:, 0] -= (2 * comp9 - M_fill[:, 6])
    M_vals[:, 1] -= (2 * comp10 - M_fill[:, 7])
    M_vals[:, 6] -= (2 * comp9 - Mg_old)
    M_vals[:, 7] -= (2 * comp10 - Fe2_old)

    sum_comp = np.nansum(comp_vals, axis=1).round(3)

    T_cols = ['Si', 'Al', 'Ti', 'Fe3']
    M_cols = ['Mg', 'Fe2', 'Fe3', 'Al', 'Ti', 'Cr', 'Ni', 'Mn', 'Ca', 'Na', 'K']

    df_T = pd.DataFrame(T_vals, columns=T_cols, index=index)
    df_M = pd.DataFrame(M_vals, columns=M_cols, index=index)

    colcomp = ['CaTiAl2O6', 'CaTs', 'Es', 'CaCrTs', 'NaCrSi2O6', 'Jd', 'Ae', 'Di', 'Hd', 'En(Mg+Ni)', 'Fs(Fe+Mn)']
    df_comp_numeric = pd.DataFrame(comp_vals, columns=colcomp, index=index)

    df_ck = pd.DataFrame({
        'Wo': (df_class['Wo'] > 20) & (df_class['Wo'] < 55),
        'J': df_class['J'] < 1,
        'Fs': (df_class['Fs'] > 5) & (df_class['Fs'] < 50),
        'Wt%': (df_ox['tot'] > 97.5) & (df_ox['tot'] < 102.5),
        'components': (sum_comp > 0.95) & (sum_comp < 1.05),
        'Si apfu': df_T['Si'] <= 2,
        'CaTiAl2O6': df_comp_numeric['CaTiAl2O6'] >= 0,
        'T_site': (df_T.sum(axis=1) > 1.95) & (df_T.sum(axis=1) < 2.05),
        'M_site': (df_M.sum(axis=1) > 1.95) & (df_M.sum(axis=1) < 2.05),
        'charge': (charge_balanced > 11.95) & (charge_balanced < 12.05),
    })
    df_ck['cpx_selection'] = df_ck.eq(True).all(axis=1)

    meta_cols = ['Index', 'sample', 'notes', 'notes.1']
    meta_cols = [c for c in meta_cols if c in df_cat.columns]
    df_comp = pd.concat([df_cat[meta_cols], df_comp_numeric], axis=1)
    df_comp1 = pd.concat([df_cat[meta_cols], df_comp_numeric.round(3)], axis=1)

    df_cat = df_cat.round(3)
    df_T = df_T.round(3)
    df_M = df_M.round(3)
    df_class = df_class.round(2)

    output_dictionary = {
        'components': df_comp1,
        'cations': df_cat,
        'checks': df_ck,
        'classifications': df_class,
        'site_T': df_T,
        'site_M1&2': df_M,
        'sum_of_components': pd.Series(sum_comp, index=index),
        'input_NN': df_comp,
        'major': df_ox.round(2),
    }

    return output_dictionary