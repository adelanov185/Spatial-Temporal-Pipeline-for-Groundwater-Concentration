import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DataLoadUtil(Dataset):
    def __init__(self, data, target):
        assert data.shape[0] == target.shape[0], "data and target samples should match."
        self.data = data
        self.target= target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        y_sample = self.target[idx]
        
        data = torch.tensor(sample, dtype=torch.float32)
        target = torch.tensor(y_sample, dtype=torch.float32)

        return data, target

def plot_prediction(
        specific_well, 
        y_train_pred_comparison, 
        y_val_pred_comparison, 
        y_test_pred_comparison, 
        mae,
        mse,
        rmse,
        directoryName,
        model='Model'
):
    color = sns.color_palette("tab10")

    full_train = pd.concat([y_train_pred_comparison, y_val_pred_comparison]).sort_index()

    plt.figure(figsize = (10,6), dpi = 200)
    plt.plot(full_train['Actual_Value'], label='Training Data', color=color[0])
    plt.plot(full_train['Predicted_Value'], label='Predicted Training Value', color=color[1])
    plt.legend()
    plt.title(specific_well + ": " + model + " On Training")
    plt.xlabel("Dates of Training Data")
    plt.ylabel("Cr(VI) Concentration")
    plt.savefig(directoryName + specific_well + "_" + model + "Train.png")
    plt.show()
    
    plt.figure(figsize = (10,6), dpi = 200)
    plt.plot(y_test_pred_comparison['Actual_Value'], label='Actual Value', color=color[2])
    plt.plot(y_test_pred_comparison['Predicted_Value'], label='Prediction', color=color[3])
    plt.legend()
    plt.title(
        specific_well + ": " + model + " On Testing," + 
        " MAE: " + str(np.round(mae,3)) + 
        " MSE: " + str(np.round(mse,3)) +
        " RMSE: " + str(np.round(rmse,3))
    )
    plt.xlabel("Dates of Testing Data")
    plt.ylabel("Cr(VI) Concentration")
    
    plt.savefig(directoryName + specific_well + "_" + model + "Test.png")
    plt.show()