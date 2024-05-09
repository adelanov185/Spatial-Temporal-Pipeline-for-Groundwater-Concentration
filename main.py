#%% Imports
import os
import copy
import pandas as pd
import numpy as np
seed = 1
np.random.seed(seed)

import argparse

import matplotlib.pyplot as plt
import subprocess

import glob
from PIL import Image
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from model import LSTM_Model, CNN_Model, plot_prediction, plot_links
import shap

#%% Command Line Inputs
parser = argparse.ArgumentParser(description="Modeling and explanation pipeline for \"Spatial-Temporal Analysis of Groundwater Well Features from Neural Network Prediction of Hexavalent Chromium Concentration\".")
parser.add_argument("-t", "--target", help = "Name of target well to forecast and explain. Target has to be available in the data after processing.", required=True)
parser.add_argument("-s", "--start", help = "Start date of data to be modeled (in Year-Month-Day format). If not available, will pick the earliest date.", required=True)
parser.add_argument("-e", "--end", help = "End date of data to be modeled (in Year-Month-Day format). If not available, will pick the latest date.", required=False)
parsed = parser.parse_args()

date1 = parsed.start
date2 = parsed.end
specific_well = parsed.target

# If interative python is preferred:
# date1 = '2015-01-01'
# date2 = '2019-12-31'
# specific_well = '199-D5-127'

#%% Read and process inputs
def data_interpolation(available_data, start, end, rollingWindow=45, feature='Concentration'):
    wells = available_data['WellName'].unique()
    
    dfList = []
    for well in wells:
        selection = available_data[available_data['WellName'] == well][feature]
        selection.index = pd.to_datetime(selection.index)
        selection = selection.reindex(pd.date_range(start, end))
        
        selection = selection.resample('D').mean()
        
        if (selection[~selection.isna()].shape[0] > 1):
            selection = selection.interpolate(method='polynomial', order=1)
        
        selection.interpolate(method='linear', order=10, inplace=True)
        selection.name = well
        
        dfList.append(selection)
    
    finals = pd.concat(dfList ,axis=1)
    
    print('Final Well Count without NaNs before rolling mean fill:', finals.shape[1])
    rez = finals.rolling(window=rollingWindow).mean().fillna(method='bfill').fillna(method='ffill')
    rez = rez.loc[:, ~rez.isna().any()]
    print('Final Well Count without NaNs after rolling mean fill:', rez.shape[1])
    return rez

data = pd.read_csv('input/100HRD.csv')

data = data[data['STD_CON_LONG_NAME'] == 'Hexavalent Chromium']
data = data.rename(columns={'SAMP_SITE_NAME': 'WellName', 'STD_VALUE_RPTD': 'Concentration', 'SAMP_DATE_TIME': 'Date'})
data = data[['Date', 'WellName', 'Concentration']]
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

locs = pd.read_csv('input/100AreaWellLocs.csv')
attri = pd.read_csv('input/100AreaWellAttri.csv')
attri_gw = attri[attri['WELL_TYPE'] == 'GROUNDWATER WELL']
groundwater_wells = attri_gw['WELL_NAME'].unique()

data = data[data['WellName'].isin(groundwater_wells)]

selects = []
for well in data['WellName'].unique():
    selection = data[data['WellName'] == well]
    selection = selection.groupby('Date')['Concentration'].mean().to_frame().reset_index()
    selection['WellName'] = well
    selects.append(selection)
data = pd.concat(selects)
data = data.sort_values(by='Date')
data = data.set_index('Date')

if date1 < data.index[0]:
    date1 = data.index[0]
if date2 > data.index[-1]:
    date2 = data.index[-1]

data = data_interpolation(data, date1, date2, rollingWindow=45, feature='Concentration')

if specific_well not in data.columns:
    raise Exception(specific_well + ' not available after processing. Here is a list of available wells: ' + str(data.columns))

y_raw = copy.deepcopy(data[specific_well])
Xx = data.drop(columns=[specific_well], inplace=False)

#%% Train/Test Split and Scaling
x_base, x_test, y_base, y_test = train_test_split(
    Xx, y_raw, test_size=0.2, random_state=seed, shuffle=False
)

predicted_start_date = x_test.index[0].strftime('%Y-%m-%d')

scaler = StandardScaler()
x_base_scaled = scaler.fit_transform(x_base)
x_test_scaled = scaler.transform(x_test)
x_base_scaled_pd = pd.DataFrame(x_base_scaled, columns=x_base.columns, index=x_base.index)
x_test_scaled_pd = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)

#%% Output Directory
directoryName = 'Target_' + specific_well + '\\'
os.makedirs(directoryName, exist_ok=True)

#%% Modeling
shuffle=True
features=Xx.columns
def DLModeling(shuffle, base_df, test_df, y_base, y_test, directoryName, model_iter=10):
    # This is the code to generate LSTM NN results.
    lstm_comparisons, lstm_metrics, scaled_data, LSTM_model = LSTM_Model(base_df, test_df, y_base, y_test, shuffle=shuffle, model_iter=model_iter)
    y_train_pred_comparison, y_val_pred_comparison, y_test_pred_comparison = lstm_comparisons
    lstm_mse, lstm_mae, lstm_rmse = lstm_metrics
    lstm_x_train_scaled, lstm_x_test_scaled, lstm_x_scaler = scaled_data
    plot_prediction(
        y_raw.name, 
        y_train_pred_comparison, y_val_pred_comparison, y_test_pred_comparison, 
        lstm_mae, lstm_mse, lstm_rmse,
        directoryName, model='LSTM'
    )
     
    # This is the code to generate CNN NN results
    cnn_comparisons, cnn_metrics, scaled_data, CNN_model = CNN_Model(base_df, test_df, y_base, y_test, shuffle=shuffle, model_iter=model_iter)
    y_train_pred_comparison, y_val_pred_comparison, y_test_pred_comparison = cnn_comparisons
    cnn_mse, cnn_mae, cnn_rmse = cnn_metrics
    cnn_x_train_scaled, cnn_x_test_scaled, cnn_x_scaler = scaled_data
    plot_prediction(
        y_raw.name,
        y_train_pred_comparison, y_val_pred_comparison, y_test_pred_comparison,
        cnn_mse, cnn_mae, cnn_rmse,
        directoryName, model='CNN'
    )
    
    # Grab best model
    # Decided by majority winner of lowest metrics
    lstm_win_count = 0
    cnn_win_count = 0
    for i in range(len(lstm_metrics)):
        if lstm_metrics[i] < cnn_metrics[i]:
            lstm_win_count += 1
        else:
            cnn_win_count += 1
    if lstm_win_count > cnn_win_count:   
        model = LSTM_model
        DL_model_name = 'LSTM'
        x_train_scaled = lstm_x_train_scaled
        x_test_scaled = lstm_x_test_scaled
        x_scaler = lstm_x_scaler
    else:
        model = CNN_model
        DL_model_name = 'CNN'
        x_train_scaled = cnn_x_train_scaled
        x_test_scaled = cnn_x_test_scaled
        x_scaler = cnn_x_scaler

    return {
        'main': (model, DL_model_name, x_train_scaled, x_test_scaled, x_scaler),
        'LSTM': (
            LSTM_model, # [0]
            lstm_mse, lstm_mae, lstm_rmse, # [1-3]
            lstm_x_train_scaled, lstm_x_test_scaled, lstm_x_scaler, # [4-6]
            lstm_comparisons # [7]
        ),
        'CNN': (
            CNN_model,  # [0]
            cnn_mse, cnn_mae, cnn_rmse, # [1-3]
            cnn_x_train_scaled, cnn_x_test_scaled, cnn_x_scaler, # [4-6]
            cnn_comparisons # [7]
        )
    }

ret = DLModeling(shuffle, x_base_scaled_pd, x_test_scaled_pd, y_base, y_test, directoryName, model_iter=10)
model, DL_model_name, x_train_scaled, x_test_scaled, x_scaler = ret['main']
y_test_pred_comparison = ret[DL_model_name][7][2]['Predicted_Value']

#%% SHAP Explanations
def shapXAI(model, x_train_scaled, x_test_scaled, assertation=True):
    explainer = shap.DeepExplainer(model, x_train_scaled)
    shap_values = explainer.shap_values(x_test_scaled, check_additivity=assertation)[0]
    return (explainer, shap_values)

explainer, shap_values = shapXAI(model, x_train_scaled, x_test_scaled)

shap.summary_plot(shap_values.sum(axis=1), plot_type='bar', feature_names=features, show=False)
plt.savefig(directoryName + 'shap_feature_importance')

#%% SHAP Deltas
def processContributions(shap_values):
    deltas = []
    j = 0
    for i in range(1, shap_values.shape[0]):
        i_sum = shap_values[i].sum(axis=0)
        j_sum = shap_values[j].sum(axis=0)
        delta = i_sum - j_sum
        deltas.append(delta)
        j += 1
    deltas = np.array(deltas)
    
    return deltas

deltas = processContributions(shap_values)

def agg_contributions(shap_values, features, deltas=None, agg_method='sum'):
    if agg_method == 'mean_mag':
        # Mean of magnitudes without direction
        contributions = np.mean(np.abs(shap_values), axis=(0,1))
    elif agg_method == 'sum_mag':
        # Sum of magnitudes without direction
        contributions = np.sum(np.abs(shap_values), axis=(0,1))
    elif agg_method == 'mean':
        # Mean of magnitudes with direction
        contributions = np.mean(shap_values, axis=(0,1))
    elif agg_method == 'sum':
        # cumulative magnitude with direction
        contributions = shap_values.sum(axis=(0,1))
    elif agg_method == 'delta_sum':
        # summing deltas with direction
        contributions = deltas.sum(0)
    
    # Match feature to its mean across every sample at every timestep
    tup = []
    for m in zip(contributions, features):
        tup.append(m)
    
    # Sort features (wells) by their means in descending order
    sorted_fImp_total = sorted(tup, key=lambda lda: lda[0], reverse=True)
    
    return sorted_fImp_total

sorted_fImp_total_mean = agg_contributions(shap_values, features, deltas, agg_method='delta_sum')

#%% Spatial Contribution Visualization
def plot_space(specific_well, sorted_fImp_total_mean, directoryName, specific_well_idx, locs, attri):
    title = 'Delta Sum Contributions: ' + specific_well
        
    f, head, tail = plot_links(
        locs, attri, specific_well_idx,  
        sorted_fImp_total_mean, title=title, 
        seperate_table=False
    )
    
    f.savefig(directoryName + 'PlotSpace_' + specific_well, bbox_inches='tight')
    subprocess.run(['optipng', directoryName + 'PlotSpace_' + specific_well])

specific_well_idx = locs[locs['WELL_NAME'] == specific_well].index[0]
plot_space(specific_well, sorted_fImp_total_mean, directoryName, specific_well_idx, locs, attri)

#%% Contribution Animation
def animate_gif(deltas, features, directoryName, specific_well_idx, locs, attri, start_date, plot_type='link', keep_scale_consitent=False):
    if keep_scale_consitent:
        min_value = np.min(deltas)
        max_value = np.max(deltas)
        file_name = f'{directoryName}{plot_type}_change_over_time_consistent_scale.gif'
    else:
        min_value = None
        max_value = None
        file_name = f'{directoryName}{plot_type}_change_over_time_relative_scale.gif'
    
    date = start_date
    for delta in deltas:
        # Match feature to its mean across every sample at every timestep
        tup = []
        for m in zip(delta, features):
            tup.append(m)
        
        # Sort features (wells) by their means in descending order
        sorted_fImp_total_mean = sorted(tup, key=lambda lda: lda[0], reverse=True)
        
        specific_well = locs.loc[specific_well_idx, 'WELL_NAME']
     
        title = f'Delta Sum Contributions at {date}: {specific_well}'
        f, head, tail = plot_links(
            locs, attri, specific_well_idx, 
            sorted_fImp_total_mean, title=title,
            min_value=min_value, max_value=max_value, seperate_table=False
        )
            
        f.savefig(f'Anim/{date}.png', bbox_inches='tight', format='png')
        
        subprocess.run(['optipng', f'Anim/{date}.png'])
            
        date = datetime.strptime(date, "%Y-%m-%d")
        date = date + timedelta(days=1)
        date = datetime.strftime(date, "%Y-%m-%d")
    
    frames = [Image.open(image) for image in glob.glob("Anim/*.png")]
    frame_one = frames[0]
    frame_one.save(file_name, format="GIF", append_images=frames, save_all=True, duration=100, loop=0)
    
    return

# Start of predicted set after subtracting lookback
predicted_start_date = datetime.strptime(predicted_start_date, "%Y-%m-%d")
predicted_start_date = predicted_start_date + timedelta(days=10+1)
predicted_start_date = datetime.strftime(predicted_start_date, "%Y-%m-%d")

plt.ioff()
animate_gif(deltas, features, directoryName, specific_well_idx, locs, attri, predicted_start_date, plot_type='link', keep_scale_consitent=True)
plt.ion()

print('Finished')