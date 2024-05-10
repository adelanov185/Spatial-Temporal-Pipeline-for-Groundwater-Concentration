#%% Imports
import os
import copy
import pandas as pd
import numpy as np
seed = 1
np.random.seed(seed)

import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tab_transformer_pytorch import TabTransformer
from early_stopping import EarlyStopping
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import DataLoadUtil, plot_prediction

#%% Command Line Inputs
parser = argparse.ArgumentParser(description="TabTransformer pipeline for \"Spatial-Temporal Analysis of Groundwater Well Features from Neural Network Prediction of Hexavalent Chromium Concentration\".")
parser.add_argument("-t", "--target", help = "Name of target well to forecast and explain. Target has to be available in the data after processing.", required=True)
parser.add_argument("-s", "--start", help = "Start date of data to be modeled (in Year-Month-Day format). If not available, will pick the earliest date.", required=True)
parser.add_argument("-e", "--end", help = "End date of data to be modeled (in Year-Month-Day format). If not available, will pick the latest date.", required=True)
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

data = pd.read_csv('../input/100HRD.csv')

data = data[data['STD_CON_LONG_NAME'] == 'Hexavalent Chromium']
data = data.rename(columns={'SAMP_SITE_NAME': 'WellName', 'STD_VALUE_RPTD': 'Concentration', 'SAMP_DATE_TIME': 'Date'})
data = data[['Date', 'WellName', 'Concentration']]
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

locs = pd.read_csv('../input/100AreaWellLocs.csv')
attri = pd.read_csv('../input/100AreaWellAttri.csv')
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
    raise Exception(specific_well + ' not available after processing. Here is a list of available wells: ' + str(list(data.columns)))

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

# Split-shuffle between train and val for evaluation
x_base_scaled, x_val_scaled, y_base, y_val = train_test_split(x_base_scaled_pd, y_base, test_size=.2, random_state=seed, shuffle=True)

#%% Creating Dataloaders for training
batch_size = 32
dataset = DataLoadUtil(x_base_scaled.values, y_base.values)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
dataset_val = DataLoadUtil(x_val_scaled.values, y_val.values)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
dataset_test = DataLoadUtil(x_test_scaled, y_test)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

#%% Output Directory
directoryName = '..\\Target_' + specific_well + '\\TabTransformerResults\\'
os.makedirs(directoryName, exist_ok=True)

#%% Modeling
continous_features = x_base_scaled.shape[1]

model_iterations = 10
model_results = []
for i in range(model_iterations):
    model = TabTransformer(
        categories = (),                        
        num_continuous = continous_features,    
        dim = 32,                               
        dim_out = 1,                            
        depth = 6,                              
        heads = 8,                              
        attn_dropout = 0.1,                     
        ff_dropout = 0.1,                       
        mlp_hidden_mults = (4, 2),              
        mlp_act = nn.Tanh(),                    
        continuous_mean_std = None              
    )

    optimizer = Adam(params=model.parameters())
    batch_number = len(dataloader)
    num_epochs = 10000

    x_categ = torch.Tensor().to(torch.float32)  # No category values
    criterion = nn.L1Loss()
    early_stopping = EarlyStopping(patience=25, verbose=True)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for j, (data_batch, target_batch) in enumerate(dataloader):
            model.train()
            outputs = model(x_categ, data_batch)
            optimizer.zero_grad()
            loss = criterion(outputs, target_batch.reshape((target_batch.shape[0], 1)))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / batch_number}')

        running_loss = 0.0
        for j, (data_batch, target_batch) in enumerate(dataloader_val):
            model.eval()
            outputs = model(x_categ, data_batch)
            loss = criterion(outputs, target_batch.reshape((target_batch.shape[0], 1)))
            running_loss += loss.item()
        print(f'Validation at Epoch {epoch + 1}, Loss: {running_loss / batch_number}')

        early_stopping((running_loss / batch_number), model)
        if early_stopping.early_stop:
            break
    print('Finished TabTransformer Training')

    state = torch.load('checkpoint.pt')
    model.load_state_dict(state)
    model.eval()

    train_batch_num = y_base.shape[0]//batch_size
    ser = pd.Series(index=y_base.index, name=specific_well)
    for j, (data_batch, target_batch) in enumerate(dataloader):
        outputs = model(x_categ, data_batch)
        if j == train_batch_num:
            ser.iloc[(batch_size*j):(batch_size*j)+len(outputs)] = outputs.detach().numpy().reshape(outputs.shape[0])
        else:
            ser.iloc[(batch_size*j):(batch_size*(j+1))] = outputs.detach().numpy().reshape(outputs.shape[0])
    ser.to_csv(directoryName + 'predicted_train.csv')

    val_batch_num = y_val.shape[0]//batch_size
    val_ser = pd.Series(index=y_val.index, name=specific_well)
    for j, (data_batch, target_batch) in enumerate(dataloader_val):
        outputs = model(x_categ, data_batch)
        if j == val_batch_num:
            val_ser.iloc[(batch_size*j):(batch_size*j)+len(outputs)] = outputs.detach().numpy().reshape(outputs.shape[0])
        else:
            val_ser.iloc[(batch_size*j):(batch_size*(j+1))] = outputs.detach().numpy().reshape(outputs.shape[0])
    val_ser.to_csv(directoryName + 'predicted_val.csv')

    test_batch_num = y_test.shape[0]//batch_size
    test_ser = pd.Series(index=y_test.index, name=specific_well)
    for j, (data_batch, target_batch) in enumerate(dataloader_test):
        outputs = model(x_categ, data_batch)
        if j == test_batch_num:
            test_ser.iloc[(batch_size*j):(batch_size*j)+len(outputs)] = outputs.detach().numpy().reshape(outputs.shape[0])
        else:
            test_ser.iloc[(batch_size*j):(batch_size*(j+1))] = outputs.detach().numpy().reshape(outputs.shape[0])
    test_ser.to_csv(directoryName + 'predicted_test.csv')

    train_comparison = pd.DataFrame(index=y_base.index, columns=['Actual_Value', 'Predicted_Value'])
    train_comparison['Actual_Value'] = y_base
    train_comparison['Predicted_Value'] = ser
    val_comparison = pd.DataFrame(index=y_val.index, columns=['Actual_Value', 'Predicted_Value'])
    val_comparison['Actual_Value'] = y_val
    val_comparison['Predicted_Value'] = val_ser
    test_comparison = pd.DataFrame(index=y_test.index, columns=['Actual_Value', 'Predicted_Value'])
    test_comparison['Actual_Value'] = y_test
    test_comparison['Predicted_Value'] = test_ser

    mae = mean_absolute_error(test_comparison['Actual_Value'], test_comparison['Predicted_Value'])
    mse = mean_squared_error(test_comparison['Actual_Value'], test_comparison['Predicted_Value'])
    rmse = mean_squared_error(test_comparison['Actual_Value'], test_comparison['Predicted_Value'], squared=False)

    model_results.append(
        (
            state, 
            (mse, mae, rmse), 
            (train_comparison, val_comparison, test_comparison)
        )
    )

#%% Choosing best model instance
best_result_index = 0
best_metrics = model_results[0][1]
for index, results in enumerate(model_results):
    metrics = results[1]
    
    # Grab best model
    # Decided by majority winner of lowest metrics
    # best_metrics is player1
    player1_win_count = 0
    player2_win_count = 0
    for i in range(len(best_metrics)):
        if best_metrics[i] < metrics[i]:
            player1_win_count += 1
        else:
            player2_win_count += 1    
            
    # Set a new best model to the winning model
    if player1_win_count < player2_win_count:  
        best_result_index = index
        best_metrics = metrics

(train_comparison, val_comparison, test_comparison) = model_results[best_result_index][2]
(mse, mae, rmse) = model_results[best_result_index][1]
state = model_results[best_result_index][0]

#%% Plotting Prediction Performance
plot_prediction(
    specific_well,
    train_comparison, val_comparison, test_comparison, 
    mae, mse, rmse,
    directoryName, model='TabTransformer'
)

#%% Save best model state
torch.save(state, 'checkpoint.pt')

