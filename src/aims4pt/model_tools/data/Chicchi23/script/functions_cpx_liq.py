# import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
# from io import BytesIO
# import base64
# import time
# import tensorflow_hub as hub
import os
from keras.layers import TFSMLayer


# def to_excel(df, index=False, startrow = 0):
#     output = BytesIO()
#     writer = pd.ExcelWriter(output, engine='xlsxwriter')
#     df.to_excel(writer, index=index, startrow=startrow, sheet_name='Sheet1')           
#     workbook = writer.book
#     worksheet = writer.sheets['Sheet1']
#     format1 = workbook.add_format({'num_format': '0.00'})
#     worksheet.set_column('A:A', None, format1)
#     writer.close()
#     processed_data = output.getvalue()
#     return processed_data

# def to_excel_multi_sheet(dic_cal):
#     output = BytesIO()
#     writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
#     for key in dic_cal:   
#         dic_cal[key].to_excel(writer, index=False, sheet_name=key)  
#     writer.save()
#     processed_data = output.getvalue()
#     return processed_data

# def predict(data, dir):    
#     for tg in [0, 1]:
#         if tg == 0:
#             directory = 'Pressure_models'
#             N = 100
#             array_max = [10.0]
#         else:
#             directory = 'Temperature_models'
#             N = 20
#             array_max = [1400.0]

#         directory = os.path.join(dir, directory)

#         targets = ['P (kbar)', 'T (C)']
#         target = targets[tg]
#         names_targets = ['pressure', 'temperature']
#         names_target = names_targets[tg]
#         sect = 'cpx_and_liq'
        
#         # Add a placeholder
#         latest_iteration = st.empty()
#         st.write('Predicting ' + names_target +' ...')
#         bar = st.progress(0)
        
#         #load global variable          
#         #with open(directory + '/mod_' + names_target + '_' + sect + '/Global_variable.pickle', 'rb') as handle:
#             #g = pickle.load(handle)
#             #g = pd.read_pickle(handle)
#         #N = g['N']
#         #array_max = g['array_max']

#         col = data.columns
#         index_col = [col[i] for i in range(0, 4)]
#         df_noindex = data.drop(columns=index_col)

#         if tg == 0:
#             df_output = pd.DataFrame(
#                 columns=index_col[:] + ['mean - ' + targets[0], 'std - ' + targets[0], 'mean - ' + targets[1],
#                                         'std - ' + targets[1]])


#         results = np.zeros((len(df_noindex), N))
#         for e in range(N):
            
#             #update bar
#             latest_iteration.text(f'Applying model n°{e + 1}')
#             bar.progress(int((e + 1)/N*100))
#             time.sleep(0.1)
            
#             #load modell
#             model = tf.keras.models.load_model(
#                 directory + "/mod_" + names_target + '_' + sect + "/Bootstrap_model_" + str(e) + '.h5', custom_objects={"KerasLayer": hub.KerasLayer} )
#             results[:, e] = model(df_noindex.values.astype('float32')).numpy().reshape((len(df_noindex),))

#         results = results * array_max[0]

#         df_output[index_col] = data[index_col]
#         mean =  results.mean(axis=1).round(np.mod(tg+1,2))
#         std = results.std(axis=1).round(np.mod(tg+1,2))
#         if tg == 1:
#             mean = mean.astype('int')
#             std = std.astype('int')
#         df_output['mean - ' + target] = mean
#         df_output['std - ' + target] = std
#     return df_output



def easy_predict(data, T_P,  dir):    
    '''
    Predict the pressure or temperature of the data using the model.
    Parameters:
        data (pd.DataFrame): The input data.
        T_P (str): 'P' for pressure, 'T' for temperature.
        dir (module): The directory where the model is stored.

    Returns:
        pd.Series: The predicted values.
    '''
    
    if T_P == 'P':
        tg = 0
    else:
        tg = 1


    if tg == 0:
        directory = 'Pressure_models'
        N = 100
        array_max = [10.0]
    else:
        directory = 'Temperature_models'
        N = 20
        array_max = [1400.0]

    directory = os.path.join(dir, directory)

    targets = ['P (kbar)', 'T (C)']
    target = targets[tg]
    names_targets = ['pressure', 'temperature']
    names_target = names_targets[tg]
    sect = 'cpx_and_liq'
    
    
    #load global variable          
    #with open(directory + '/mod_' + names_target + '_' + sect + '/Global_variable.pickle', 'rb') as handle:
        #g = pickle.load(handle)
        #g = pd.read_pickle(handle)
    #N = g['N']
    #array_max = g['array_max']

    col = data.columns
    index_col = [col[i] for i in range(0, 4)]
    df_noindex = data.drop(columns=index_col)

    df_output = pd.DataFrame(
        columns=index_col[:] + ['mean - ' + targets[0], 'std - ' + targets[0] if tg == 0 else 
                                'mean - ' + targets[1],
                                'std - ' + targets[1]])


    results = np.zeros((len(df_noindex), N))
    for e in range(N):
        sm_path = directory + "/mod_" + names_target + '_' + sect + "/Bootstrap_model_" + str(e) + '_saved'

        layer = TFSMLayer(
            sm_path,
            call_endpoint="serving_default",
        )

        inp = tf.keras.Input(shape=(19,), dtype="float32", name="dense_input")
        out = layer(inp)
        model = tf.keras.Model(inp, out)
        predict_dict = model.predict(df_noindex.values.astype('float32'), verbose=0)
        key = list(predict_dict.keys())[0]
        results[:, e] = predict_dict[key].reshape((len(df_noindex),))

    results = results * array_max[0]

    df_output[index_col] = data[index_col]
    mean =  results.mean(axis=1).round(np.mod(tg+1,2))
    std = results.std(axis=1).round(np.mod(tg+1,2))
    if tg == 1:
        mean = mean.astype('int')
        std = std.astype('int')
    df_output['mean - ' + target] = mean
    df_output['std - ' + target] = std
    return mean


_MODEL_CACHE = {}

def _get_model_spec(T_P: str):
    """
    Return (subdir, N, scale_max, names_target).
    T_P: 'P' or 'T'
    """
    if T_P == "P":
        return "Pressure_models", 100, 10.0, "pressure"
    elif T_P == "T":
        return "Temperature_models", 20, 1400.0, "temperature"
    else:
        raise ValueError("T_P must be 'P' or 'T'")

def preload_models_cpx_liq(model_root: str, T_P: str, n_features: int, sect: str = "cpx_and_liq"):
    """
    Preload and cache bootstrap models for cpx+liq.
    Cache key: (abs_model_root, T_P, sect, n_features)
    """
    abs_root = os.path.abspath(model_root)
    cache_key = (abs_root, T_P, sect, n_features)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    subdir, N, _, names_target = _get_model_spec(T_P)
    directory = os.path.join(abs_root, subdir)

    models = []
    for e in range(N):
        sm_path = os.path.join(
            directory,
            f"mod_{names_target}_{sect}",
            f"Bootstrap_model_{e}_saved",
        )

        layer = TFSMLayer(sm_path, call_endpoint="serving_default")
        inp = tf.keras.Input(shape=(n_features,), dtype="float32", name="dense_input")
        out = layer(inp)
        model = tf.keras.Model(inp, out)
        models.append(model)

    _MODEL_CACHE[cache_key] = models
    return models

def easy_predict_cpx_liq(data, T_P: str, dir: str):
    """
    cpx+liq version. Return mean only.
    Assumes first 4 columns are index/info; remaining columns are features.
    """
    # split index/info cols vs features
    col = data.columns
    index_col = [col[i] for i in range(0, 4)]
    df_noindex = data.drop(columns=index_col)

    n_features = df_noindex.shape[1]
    sect = "cpx_and_liq"

    # preload (cached)
    models = preload_models_cpx_liq(model_root=dir, T_P=T_P, n_features=n_features, sect=sect)

    # model spec
    _, N, scale_max, _ = _get_model_spec(T_P)

    # predict
    x = df_noindex.values.astype("float32", copy=False)
    results = np.empty((len(df_noindex), N), dtype=np.float32)

    for e, model in enumerate(models):
        predict_dict = model.predict(x, verbose=0)
        key = next(iter(predict_dict.keys()))
        results[:, e] = predict_dict[key].reshape((len(df_noindex),))

    # rescale + mean
    results *= scale_max
    mean = results.mean(axis=1)

    # rounding consistent with your original code:
    # P -> 1 decimal; T -> int
    if T_P == "P":
        mean = np.round(mean, 1)
    else:
        mean = np.round(mean, 0).astype(int)

    return mean

def clear_model_cache():
    """Call this if you updated model files on disk or want to free memory."""
    _MODEL_CACHE.clear()


# @st.cache
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')


# def get_base64(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()


# def set_png_as_page_bg(png_file, opacity=1):
#     bin_str = get_base64(png_file)
#     page_bg_img = '''
#     <style>
#     .stApp {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     background-opacity:opacity;
#     background-repeat: no-repeat;
#     background-attachment: scroll; # doesn't work
#     }
#     </style>
#     ''' % bin_str
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     return


# def plothist(df_output):
#     targets = ['P (kbar)', 'T (C)']
#     col = ['tab:green', 'tab:red']
#     titles = ['P distribution', 'T distribution']
#     fig, ax = plt.subplots(1, 2, figsize=(8, 6))
#     for tg in [0, 1]:
#         x = df_output['mean - ' + targets[tg]].values.reshape(-1, 1)
#         ax[tg].hist(df_output['mean - ' + targets[tg]].values, density=True, edgecolor='k', color=col[tg], label='hist')
#         ax[tg].set_title(titles[tg], fontsize=13)
#         ax[tg].set_xlabel(targets[tg], fontsize=13)
#     fig.tight_layout(pad=2.0)
#     st.pyplot(fig)
