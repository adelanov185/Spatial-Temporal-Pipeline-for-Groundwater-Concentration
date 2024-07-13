import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime, timedelta
import shap
import subprocess
import glob
from PIL import Image
from utils import plot_links, plot_links_2
import argparse

#%% Read Input
parser = argparse.ArgumentParser(description="DA-LSTM attention analysis for \"Spatial-Temporal Analysis of Groundwater Well Features from Neural Network Prediction of Hexavalent Chromium Concentration\".")
parser.add_argument("-t", "--target", help = "Name of target well to forecast and explain. Target has to be available in the data after processing.", required=True)
parser.add_argument("-s", "--start", help = "Start date of data to be modeled (in Year-Month-Day format). If not available, will pick the earliest date.", required=True)
parser.add_argument("-e", "--end", help = "End date of data to be modeled (in Year-Month-Day format). If not available, will pick the latest date.", required=True)
parsed = parser.parse_args()

date1 = parsed.start
date2 = parsed.end
specific_well = parsed.target

# date1 = '2015-01-01'
# date2 = '2019-12-31'
# specific_well = '199-D5-127'

directory = '..\\Target_' + specific_well + '\\DA-LSTM_Results\\'

#region Attentions
att_feat = np.load(directory + 'f_final_att_tests.npy')
att_temp = np.load(directory + 't_final_att_tests.npy')
att_feat = att_feat[:, -1, :]
abs_max = np.max(np.abs(att_feat))
att_feat = att_feat / abs_max # Max Absolute Normalization

att_temp = att_temp[:, -1, :]
array_min = np.min(att_temp)
array_max = np.max(att_temp)
att_temp = (att_temp - array_min) / (array_max - array_min) # Min-Max Normalization
#endregion

#region Original
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

train_split = int(Xx.shape[0] * 0.8)

lookback = 50
# Sequencing
x_sequences = []
y_sequences = []
for i in range(Xx.shape[0]):
    endIndex = i + lookback
    if endIndex >= Xx.shape[0]:
        break
    sequenceX = Xx[i:endIndex]
    sequenceY = endIndex
    x_sequences.append(sequenceX)
    y_sequences.append(sequenceY)
x_test = np.array(x_sequences)

X = x_test
#%% Trim x to just account for testing year
X = X[train_split:]

pd_comp = pd.read_csv(directory + 'DA-LSTM_predicted.csv')
pd_comp = pd_comp.set_index('COLLECTION_DATE')
#endregion

sample_num = 259 #209 #270 #150

sample_feat_att = att_feat[sample_num]

sample_temp_att = att_temp[sample_num]

sample_tar = pd_comp.iloc[sample_num]
x_sample = X[sample_num]
start_date = datetime.strptime(sample_tar.name, '%Y-%m-%d') - timedelta(days=lookback)
end_date = datetime.strptime(sample_tar.name, '%Y-%m-%d') - timedelta(days=1)
d_range = pd.date_range(start_date.date(), end_date.date()).astype(str)
x_sample_pd = pd.DataFrame(index=d_range, data=x_sample, columns=Xx.columns)
sample_feat_att_pd = pd.DataFrame(index=Xx.columns, data=sample_feat_att)
sample_temp_att_pd = pd.DataFrame(index=d_range, data=sample_temp_att)

#region Standard Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_sample_pd)
plt.ylabel("Cr(VI) µg/L")
plt.xlabel("Dates of Testing Data")
plt.title('Sample Original')
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
plt.savefig(directory + 'Sample_Original', bbox_inches='tight')
plt.show()
#endregion 

#region Explanative Plotting
plt.figure(figsize=(10, 6))
plt.plot(sample_temp_att_pd)
plt.ylabel("Attention Scores")
plt.xlabel("Dates of Testing Data")
plt.title('Sample Explanation')
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
plt.savefig(directory + 'Sample_Explanation', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
title = 'Explantion Location on Prediction'
plt.plot(pd_comp['Predicted_Value'], color='red')
plt.axvline(x=sample_temp_att_pd.index[0], color='blue', linestyle='--')
plt.axvline(x=sample_temp_att_pd.index[-1], color='blue', linestyle='--')
plt.plot(sample_tar.name, sample_tar.iloc[1], 'ro') 
plt.title(title)
plt.ylabel("Cr(VI) µg/L")
plt.xlabel("Dates of Testing Data")
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
plt.savefig(directory + 'Sample_Explanation_Location' , bbox_inches='tight')
plt.show()

diff = list(set(pd_comp['Predicted_Value'].index) - set(sample_temp_att_pd.index))
diff.sort()
twin_temp_att = pd.concat([sample_temp_att_pd, pd.DataFrame(index=diff, columns=sample_temp_att_pd.columns)])
twin_temp_att = twin_temp_att.sort_index()

fig, ax1 = plt.subplots(figsize=(10, 6))
# Plot data on the primary y-axis
ax1.plot(pd_comp['Predicted_Value'], 'r-', label='Original Data')
ax1.plot(sample_tar.name, sample_tar.iloc[1], 'ro') 
ax1.set_xlabel('Dates of Testing Data')
ax1.set_ylabel('Cr(VI) µg/L', color='r')
ax1.tick_params(axis='y', labelcolor='r')

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.plot(twin_temp_att, 'b-', label='Explanations')
ax2.set_ylabel('Attention', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Add legends for both plots
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(1.10, 1))

plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
plt.title('Target Concentration vs Temporal Attentions')
plt.savefig(directory + 'TargetTempAtt', bbox_inches='tight')
plt.show()

fig, ax1 = plt.subplots(figsize=(10, 6))
# Plot data on the primary y-axis
ax1.plot(pd_comp['Predicted_Value'].loc[sample_temp_att_pd.index], 'r-', label='Original Data')
ax1.set_xlabel('Dates of Testing Data')
ax1.set_ylabel('Cr(VI) µg/L', color='r')
ax1.tick_params(axis='y', labelcolor='r')

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.plot(sample_temp_att_pd, 'b-', label='Explanations')
ax2.set_ylabel('Attention', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Add legends for both plots
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(1.10, 1))

plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
plt.title('Target Concentration vs Temporal Attentions Expanded')
plt.savefig(directory + 'TargetTempAttExpanded', bbox_inches='tight')
plt.show()
    
sample_feat_att_pd = sample_feat_att_pd.sort_values(by=0)
driving_wells = sample_feat_att_pd.tail(10)

plt.figure(figsize=(10, 6))
for well in driving_wells.index:
    plt.plot(x_sample_pd[well], label=well)
plt.plot(x_sample_pd[list(set(sample_feat_att_pd.index) - set(driving_wells.index))], alpha=0.1)
plt.ylabel("Attention Scores on Features")
plt.xlabel("Dates of Testing Data")
plt.title('Sample Driving Features')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
plt.savefig(directory + 'SampleFeatureAtt', bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.barh(driving_wells.index, driving_wells.values.reshape(-1))
plt.xticks(rotation=25)
plt.ylabel("Wells")
plt.xlabel("Attention Scores")
plt.title('Sample Feature Attentions')
plt.savefig(directory + 'SampleBarFeatureAtt', bbox_inches='tight')
plt.show()

fig, ax1 = plt.subplots(figsize=(10, 6))
# Plot data on the primary y-axis
for well in driving_wells.index:
    ax1.plot(x_sample_pd[well], label=well)
ax1.set_xlabel('Dates of Testing Data')
ax1.set_ylabel('Cr(VI) µg/L')

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.plot(sample_temp_att_pd, 'b-', label='Temporal Attention')
ax2.set_ylabel('Temporal Attention', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Add legends for both plots
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(1.10, 1))

plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
plt.title('Driving Features with Temporal Attentions')
plt.savefig(directory + 'FeatureTempAttExpanded', bbox_inches='tight')
plt.show()
#endregion

if True:
#region SHAP-like Plotting
    year_feat_att = att_feat

    year_temp_att = att_temp

    meaned = np.mean(np.abs(year_feat_att), axis=0) 
    meaned_norm_pd = pd.DataFrame(index=Xx.columns, data=meaned).sort_values(0)

    plt.figure(figsize=(8, 6))
    plt.barh(meaned_norm_pd.index[-20:], meaned_norm_pd.iloc[-20:, 0])
    plt.xticks(rotation=25)
    plt.ylabel("Wells")
    plt.xlabel("Mean(|Feature Attentions|) of Every Target Prediction'")
    plt.title('Top 20 Average Feature Attentions')
    plt.savefig(directory + 'attention_feature_importance', bbox_inches='tight')
    plt.show()

    def processContributions(arr):
        deltas = []
        j = 0
        for i in range(1, arr.shape[0]):
            delta = arr[i] - arr[j]
            deltas.append(delta)
            j += 1
        deltas = np.array(deltas)
        
        return deltas

    deltas = processContributions(year_feat_att)

    def plot_deltas(deltas, index, features):
        # Use prediction comparison DataFrame's index for x-axis since it already has the testing dates included. 
        # We don't include the first date since we don't have a delta between first date and it's previous date
        deltas_pd = pd.DataFrame(deltas, index=index, columns=features)
        
        # Select deltas head and tail
        maxes = deltas_pd.apply(lambda x: max(x)).sort_values(ascending=False).head(5)
        mins = deltas_pd.apply(lambda x: min(x)).sort_values(ascending=True).head(5)
        z = maxes.index.to_list() + mins.index.to_list()
        z = list(set(z))
        
        # Plot those wells which has the highest or lowest delta contributions
        f, ax = plt.subplots(1, 1, figsize=(12,6), dpi=300)
        ax.set_xlabel("Dates of Testing Data")
        ax.set_ylabel("Feature Attention Deltas")
        ax.plot(deltas_pd[z], label=z)

        ax.set_title('Wells with Highest and Lowest Deltas Across Time')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        f.savefig(f'{directory}DeltasSumHighestLowest.jpg', bbox_inches='tight')

    plot_deltas(deltas, pd_comp.index[1:], Xx.columns)

    def agg_contributions(year_feat_att, features, deltas=None, agg_method='sum'):
        if agg_method == 'mean_mag':
            # Mean of magnitudes without direction
            contributions = np.mean(np.abs(year_feat_att), axis=(0))
        elif agg_method == 'sum_mag':
            # Sum of magnitudes without direction
            contributions = np.sum(np.abs(year_feat_att), axis=(0))
        elif agg_method == 'mean':
            # Mean of magnitudes with direction
            contributions = np.mean(year_feat_att, axis=(0))
        elif agg_method == 'sum':
            # cumulative magnitude with direction
            contributions = year_feat_att.sum(axis=(0))
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

    sorted_fImp_total_mean = agg_contributions(year_feat_att, Xx.columns, deltas, agg_method='mean')

    def plot_space(specific_well, sorted_fImp_total_mean, directoryName, specific_well_idx, locs, attri):
        title = 'Mean Attentions: ' + specific_well

        min_value = np.min(year_feat_att)
        max_value = np.max(year_feat_att)
            
        f, head, tail = plot_links_2(
            locs, attri, specific_well_idx,  
            sorted_fImp_total_mean, title=title,
            max_value=max_value, min_value=min_value,
            seperate_table=False 
        )
        
        f.savefig(directoryName + 'PlotSpace_' + specific_well, bbox_inches='tight')
        subprocess.run(['../optipng', directoryName + 'PlotSpace_' + specific_well + '.png'])

    specific_well_idx = locs[locs['WELL_NAME'] == specific_well].index[0]

    if False:
        # Match feature to its mean across every sample at every timestep
        tup = []
        for m in zip(year_feat_att[sample_num], Xx.columns):
            tup.append(m)
        
        # Sort features (wells) by their means in descending order
        sorted_fImp_total_mean = sorted(tup, key=lambda lda: lda[0], reverse=True)
    
    plot_space(specific_well, sorted_fImp_total_mean, directory, specific_well_idx, locs, attri)
    
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
        
            title = f'Mean Attentions at {date}: {specific_well}'
            f, head, tail = plot_links_2(
                locs, attri, specific_well_idx, 
                sorted_fImp_total_mean, title=title,
                min_value=min_value, max_value=max_value, seperate_table=False
            )
                
            f.savefig(f'../Anim/{date}.png', bbox_inches='tight', format='png')
            
            subprocess.run(['../optipng', f'../Anim/{date}.png'])
                
            date = datetime.strptime(date, "%Y-%m-%d")
            date = date + timedelta(days=1)
            date = datetime.strftime(date, "%Y-%m-%d")
        
        frames = [Image.open(image) for image in glob.glob("../Anim/*.png")]
        frame_one = frames[0]
        frame_one.save(file_name, format="GIF", append_images=frames, save_all=True, duration=100, loop=0)
        
        return

    plt.ioff()
    animate_gif(year_feat_att, Xx.columns, directory, specific_well_idx, locs, attri, pd_comp.index[1], plot_type='link', keep_scale_consitent=True)
    plt.ion()

#endregion