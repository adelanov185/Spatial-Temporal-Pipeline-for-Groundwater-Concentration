import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scipy.interpolate import interp1d

def LSTM_Model(x_base, x_test, y_raw, y_test, shuffle=True, model_iter=10):
    #%% Split
    seed = 1
    np.random.seed(seed)              
    tf.random.set_seed(seed)

    lookback = 10

    trainShape0 = x_base.shape[0]
    testShape0 = x_test.shape[0]

    # Adding the some lookback records from training to our test for the first lookback amount of predictions
    x_test = pd.concat([x_base.iloc[-(lookback-1):,:], x_test])
    
    # Making test data into sequences (This stays static)
    x_sequences = []
    y_sequences = []
    for i in range(testShape0):
        endIndex = i + lookback
        if endIndex >= testShape0:
            break
        sequenceX = x_test[i:endIndex]
        sequenceY = endIndex
        x_sequences.append(sequenceX)
        y_sequences.append(sequenceY)
    x_test = np.array(x_sequences)
    y_array = np.array(y_sequences) 
    y_test_date = [y_test.index[i] for i in y_array]
    y_test = np.array([y_test[i] for i in y_array], dtype='float64')

    # Making base training data into sequences
    x_sequences = []
    y_sequences = []
    for i in range(trainShape0):
        endIndex = i + lookback
        if endIndex >= trainShape0:
            break
        sequenceX = x_base[i:endIndex]
        sequenceY = endIndex
        x_sequences.append(sequenceX)
        y_sequences.append(sequenceY)
    x_array = np.array(x_sequences)
    y_array = np.array(y_sequences) 
    
    model_iterations = model_iter
    model_results = []
    for i in range(model_iterations):
        x_train, x_val, y_train_indices, y_val_indices = train_test_split(
            x_array, y_array, test_size=0.2, random_state=seed, shuffle=shuffle
        )
        y_train = np.array([y_raw[i] for i in y_train_indices], dtype='float64')
        y_train_date = [y_raw.index[i] for i in y_train_indices]
        y_val_date = [y_raw.index[i] for i in y_val_indices]
        y_val = np.array([y_raw[i] for i in y_val_indices], dtype='float64')

        #%% LSTM
        l1Nodes = 14
        l2Nodes = 14
        d1Nodes = 14
        d2Nodes = 7
    
        #Input Layer
        lstm1 = LSTM(l1Nodes, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True, name='LSTM_1')
        lstm2 = LSTM(l2Nodes, return_sequences=True, name='LSTM_2')
        flatten = Flatten(name='flatten')
        dense1 = Dense(d1Nodes, name='Dense_1')
        dense2 = Dense(d2Nodes, name='Dense_2')
    
        #Ouput Layer
        outL = Dense(1, activation='relu', name='output')
    
        #Combine Layers
        layers = [lstm1, lstm2, flatten, dense1, dense2, outL]
    
        #Creation
        model = Sequential(layers)
        opt = Adam(learning_rate=0.005)
        model.compile(optimizer=opt, loss='mae')
    
        #%% Fit
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        history = model.fit(x_train, y_train, batch_size=64, epochs=1000, callbacks=[earlyStopping])
    
        #%% Predict
        y_train_pred = model.predict(x_train)
        y_val_pred = model.predict(x_val)
        y_test_pred = model.predict(x_test)
    
    
        #%% Create Label Prediction Result Dataframes
        y_train_pred_comparison = pd.DataFrame(columns=['COLLECTION_DATE', 'Actual_Value'])
        y_train_pred_comparison['COLLECTION_DATE'] = y_train_date
        y_train_pred_comparison['Actual_Value'] = y_train
        y_train_pred_comparison["Predicted_Value"] = y_train_pred
        y_train_pred_comparison.set_index('COLLECTION_DATE', inplace=True)
        y_train_pred_comparison.sort_index(inplace=True)
    
        y_val_pred_comparison = pd.DataFrame(columns=['COLLECTION_DATE', 'Actual_Value'])
        y_val_pred_comparison['COLLECTION_DATE'] = y_val_date
        y_val_pred_comparison['Actual_Value'] = y_val
        y_val_pred_comparison["Predicted_Value"] = y_val_pred
        y_val_pred_comparison.set_index('COLLECTION_DATE', inplace=True)
        y_val_pred_comparison.sort_index(inplace=True)
    
        y_test_pred_comparison = pd.DataFrame(columns=['COLLECTION_DATE', 'Actual_Value'])
        y_test_pred_comparison['COLLECTION_DATE'] = y_test_date
        y_test_pred_comparison['Actual_Value'] = y_test
        y_test_pred_comparison["Predicted_Value"] = y_test_pred
        y_test_pred_comparison.set_index('COLLECTION_DATE', inplace=True)
        y_test_pred_comparison.sort_index(inplace=True)
    
    
        #%% Get Metrics
        mae = mean_absolute_error(y_test_pred_comparison['Actual_Value'], y_test_pred_comparison['Predicted_Value'])
        mse = mean_squared_error(y_test_pred_comparison['Actual_Value'], y_test_pred_comparison['Predicted_Value'])
        rmse = mean_squared_error(y_test_pred_comparison['Actual_Value'], y_test_pred_comparison['Predicted_Value'], squared=False)
        r2 = r2_score(y_test_pred_comparison['Actual_Value'], y_test_pred_comparison['Predicted_Value'])

        model_results.append(
            (
                model, 
                (mse, mae, rmse, r2), 
                (y_train_pred_comparison, y_val_pred_comparison, y_test_pred_comparison),
                (x_train, x_test, None)
            )
        )
        
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
       
    (y_train_pred_comparison, y_val_pred_comparison, y_test_pred_comparison) = model_results[best_result_index][2]
    (mse, mae, rmse, r2) = model_results[best_result_index][1]
    (x_train, x_test, None), model_results[best_result_index][3]
    model = model_results[best_result_index][0]

    return (
        (y_train_pred_comparison, y_val_pred_comparison, y_test_pred_comparison), 
        (mse, mae, rmse, r2), 
        (x_train, x_test, None), 
        model
    )

def CNN_Model(x_base, x_test, y_raw, y_test, shuffle=True, model_iter=10):
    #%% Split
    seed = 1
    np.random.seed(seed)              
    tf.random.set_seed(seed)

    lookback = 10

    trainShape0 = x_base.shape[0]
    testShape0 = x_test.shape[0]

    # Adding the some lookback records from training to our test for the first lookback amount of predictions
    x_test = pd.concat([x_base.iloc[-(lookback-1):,:], x_test])
    
    # Making test data into sequences (This stays static)
    x_sequences = []
    y_sequences = []
    for i in range(testShape0):
        endIndex = i + lookback
        if endIndex >= testShape0:
            break
        sequenceX = x_test[i:endIndex]
        sequenceY = endIndex
        x_sequences.append(sequenceX)
        y_sequences.append(sequenceY)
    x_test = np.array(x_sequences)
    y_array = np.array(y_sequences) 
    y_test_date = [y_test.index[i] for i in y_array]
    y_test = np.array([y_test[i] for i in y_array], dtype='float64')

    # Making base training data into sequences
    x_sequences = []
    y_sequences = []
    for i in range(trainShape0):
        endIndex = i + lookback
        if endIndex >= trainShape0:
            break
        sequenceX = x_base[i:endIndex]
        sequenceY = endIndex
        x_sequences.append(sequenceX)
        y_sequences.append(sequenceY)
    x_array = np.array(x_sequences)
    y_array = np.array(y_sequences) 
    
    model_iterations = model_iter
    model_results = []
    for i in range(model_iterations):
        x_train, x_val, y_train_indices, y_val_indices = train_test_split(
            x_array, y_array, test_size=0.2, random_state=seed, shuffle=shuffle
        )
        y_train = np.array([y_raw[i] for i in y_train_indices], dtype='float64')
        y_train_date = [y_raw.index[i] for i in y_train_indices]
        y_val_date = [y_raw.index[i] for i in y_val_indices]
        y_val = np.array([y_raw[i] for i in y_val_indices], dtype='float64')

        #%% Conv1D
        c1Nodes = 14
        d1Nodes = 7
        
        #Input Layer
        Conv1D1 = Conv1D(filters=c1Nodes, kernel_size=3, activation='tanh', input_shape=(x_train.shape[1], x_train.shape[2]), name='1D_CNN')
        
        #Hidden Layers
        MaxPooling1D1 = MaxPooling1D(pool_size=2, name='1D_MaxPooling')
        flatten = Flatten(name='flatten')
        dense1 = Dense(d1Nodes, name='Dense')
        
        #Ouput Layer
        outL = Dense(1, name='output')
        
        #Combine Layers
        layers = [Conv1D1, MaxPooling1D1, flatten, dense1, outL]
        
        #Creation
        model = Sequential(layers)
        model.compile(loss='mae', optimizer='adam')
        
        #%% Fit
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
        history = model.fit(x_train, y_train,  epochs=1000, batch_size=64, callbacks=[callback])
    
        #%% Predict
        y_train_pred = model.predict(x_train)
        y_val_pred = model.predict(x_val)
        y_test_pred = model.predict(x_test)
    
    
        #%% Create Label Prediction Result Dataframes
        y_train_pred_comparison = pd.DataFrame(columns=['COLLECTION_DATE', 'Actual_Value'])
        y_train_pred_comparison['COLLECTION_DATE'] = y_train_date
        y_train_pred_comparison['Actual_Value'] = y_train
        y_train_pred_comparison["Predicted_Value"] = y_train_pred
        y_train_pred_comparison.set_index('COLLECTION_DATE', inplace=True)
        y_train_pred_comparison.sort_index(inplace=True)
    
        y_val_pred_comparison = pd.DataFrame(columns=['COLLECTION_DATE', 'Actual_Value'])
        y_val_pred_comparison['COLLECTION_DATE'] = y_val_date
        y_val_pred_comparison['Actual_Value'] = y_val
        y_val_pred_comparison["Predicted_Value"] = y_val_pred
        y_val_pred_comparison.set_index('COLLECTION_DATE', inplace=True)
        y_val_pred_comparison.sort_index(inplace=True)
    
        y_test_pred_comparison = pd.DataFrame(columns=['COLLECTION_DATE', 'Actual_Value'])
        y_test_pred_comparison['COLLECTION_DATE'] = y_test_date
        y_test_pred_comparison['Actual_Value'] = y_test
        y_test_pred_comparison["Predicted_Value"] = y_test_pred
        y_test_pred_comparison.set_index('COLLECTION_DATE', inplace=True)
        y_test_pred_comparison.sort_index(inplace=True)
    
    
        #%% Get Metrics
        mae = mean_absolute_error(y_test_pred_comparison['Actual_Value'], y_test_pred_comparison['Predicted_Value'])
        mse = mean_squared_error(y_test_pred_comparison['Actual_Value'], y_test_pred_comparison['Predicted_Value'])
        rmse = mean_squared_error(y_test_pred_comparison['Actual_Value'], y_test_pred_comparison['Predicted_Value'], squared=False)
        r2 = r2_score(y_test_pred_comparison['Actual_Value'], y_test_pred_comparison['Predicted_Value'])

        model_results.append(
            (
                model, 
                (mse, mae, rmse, r2), 
                (y_train_pred_comparison, y_val_pred_comparison, y_test_pred_comparison),
                (x_train, x_test, None)
            )
        )
    
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
 
    (y_train_pred_comparison, y_val_pred_comparison, y_test_pred_comparison) = model_results[best_result_index][2]
    (mse, mae, rmse, r2) = model_results[best_result_index][1]
    (x_train, x_test, None), model_results[best_result_index][3]
    model = model_results[best_result_index][0]
        
    return (
        (y_train_pred_comparison, y_val_pred_comparison, y_test_pred_comparison), 
        (mse, mae, rmse, r2), 
        (x_train, x_test, None), 
        model
    )

def plot_prediction(
        specific_well, 
        y_train_pred_comparison, 
        y_val_pred_comparison, 
        y_test_pred_comparison, 
        mae,
        mse,
        rmse,
        r2,
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
    plt.ylabel("Cr(VI) µg/L")
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
        " RMSE: " + str(np.round(rmse,3)) + 
        " R2: " + str(np.round(r2,3))
    )
    plt.xlabel("Dates of Testing Data")
    plt.ylabel("Cr(VI) µg/L")
    
    plt.savefig(directoryName + specific_well + "_" + model + "Test.png")
    plt.show()

def plot_links(
    g, g2attr, specific_well_idx, sorted_fImp_total, 
    title='Spatial Contributions', consider_distance=False,
    min_value=None, max_value=None, color_sam=None, seperate_table=True
):
    wellNamesByImportanceDesc = [well[1] for well in sorted_fImp_total]
    
    # Grab well considered in modeling only
    modeled_wells = g[g['WELL_NAME'].isin(wellNamesByImportanceDesc)]
    modeled_wells['Contribution'] = modeled_wells.apply(
        lambda x: [value[0] for value in sorted_fImp_total if value[1] == x['WELL_NAME']][0], axis=1
    )

    # Grab target well location and name
    CenterX = g.loc[specific_well_idx]['X']
    CenterY = g.loc[specific_well_idx]['Y']
    
    # Initial plot with aquifers and center wells
    if not seperate_table:
        f, (ax, ax_table) = plt.subplots(2, 1, figsize=(12,10), dpi=500, gridspec_kw={'height_ratios': [2,1]})
    else: 
        f, ax = plt.subplots(figsize=(8,6), dpi=500)

    ax.set_ylabel('Northing', fontsize=20) 
    ax.set_xlabel('Easting', fontsize=20)
    
    aquifer = g[g['WELL_NAME'].isin(
        g2attr.loc[g2attr.index[g2attr['WELL_TYPE'] == 'AQUIFER TUBE'], 'WELL_NAME']
    )]
    
    aqtX = aquifer['X']
    aqtY = aquifer['Y']
    # ax.scatter(aqtX, aqtY, color='r', marker='*', s=142)

    unique_data = dict(zip(aqtX, aqtY))
    aqtX_unique = list(unique_data.keys())
    aqtY_unique = list(unique_data.values())
    aqtX_unique.append(-13305000)
    aqtY_unique.append(5896500)
    aqtX_unique.append(-13304500)
    aqtY_unique.append(5897000)

    #ax.scatter(aqtX_unique, aqtY_unique, color='r', marker='*', s=142)

    # Create an interpolation function
    interp_func = interp1d(aqtX_unique, aqtY_unique, kind='linear')

    # Generate new X values for a smooth line
    x_smooth = np.linspace(min(aqtX), max(aqtX), 500)
    y_smooth = interp_func(x_smooth)

    # Draw a smooth line connecting the points
    ax.plot(x_smooth, y_smooth, color='y', alpha=.5)
    
    ax.scatter(CenterX, CenterY, color='g')
    
    # Add low alpha unmodeled wells for reference
    unmodeled = g.drop(pd.concat([aquifer, modeled_wells]).index)
    unmodeled.drop(specific_well_idx, inplace=True)
    ax.scatter(unmodeled['X'], unmodeled['Y'], color='black', alpha=0.01)
    
    #%% Color interpolation
    modeled_wells = modeled_wells.reindex(modeled_wells['Contribution'].sort_values(ascending=True).index)
    z = modeled_wells['Contribution'].to_list()
    
    # Define the two colors for the gradient
    color_start = np.array([0.1, 0.1, 1.0])  # blue for negative values
    color_end = np.array([1.0, 0.1, 0.1])   # red for positive values
    
    # Create the gradient array
    if min_value is None:
        min_value = np.min(z)
    if max_value is None:
        max_value = np.max(z)
    num_color_samples = len(z)//2 if color_sam is None else color_sam
    if min_value < 0 and max_value > 0:
        # Do not allow for a white color by limiting negative and positive gradient color limits
        gradient_start = np.linspace(color_start, [0.9, 0.9, 1.0], num_color_samples)
        gradient_end = np.linspace([1.0, 0.9, 0.9], color_end, num_color_samples)
        
        # Combine blue and red gradients
        gradient = np.vstack((gradient_start, gradient_end))

        # Interpolate the colors based on the contirbution values
        colors = np.zeros((len(z), 3))
        for i in range(len(z)):
            if z[i] < 0:
                # Reverse indexing from half point in gradient for negative numbers as z is sorted from negative to positve
                # This produces divergent gradient between blue and red, rather than two stacked gradients  idx = num_color_samples - int((z[i] / min_value) * (num_color_samples - 1))
                idx = num_color_samples - int((z[i] / min_value) * (num_color_samples - 1))
                if idx == num_color_samples:
                    idx = num_color_samples - 1
                colors[i, :] = gradient[idx]
            elif z[i] > 0:
                colors[i, :] = gradient[num_color_samples + int((z[i] / max_value) * (num_color_samples - 1))]
            else:
                # Green color for zero values
                colors[i, :] = [0.9, 0.9, 0.9]
    else:
        num_color_samples = len(z)
        colors = np.zeros((len(z), 3))
        if max_value > 0:
            # Do not allow for a white color by limiting negative and positive gradient color limits
            gradient = np.linspace([1.0, 0.9, 0.9], color_end, num_color_samples)
            for i in range(len(z)):
                colors[i, :] = gradient[int((z[i] / max_value) * (num_color_samples - 1))]
        else:
            gradient = np.linspace(color_start, [0.9, 0.9, 1.0], num_color_samples)
            for i in range(len(z)):
                colors[i, :] = gradient[int((z[i] / min_value) * (num_color_samples - 1))]
    
    # Plot contributions
    ax.scatter(modeled_wells['X'], modeled_wells['Y'], c=colors)    
    
    #%% Annotate top 5 impacts on each direction
    if seperate_table:
        f_table, ax_table = plt.subplots(1, 1, figsize=(8,2), dpi=500)
    
    take = 5
    head = modeled_wells.head(take).reset_index(drop=True)
    tail = modeled_wells.tail(take).reset_index(drop=True)
    for i in range(take):
        # Blues
        ax.annotate(head.iloc[i]['WELL_NAME'], (head.iloc[i]['X'], head.iloc[i]['Y']))
        # Reds
        ax.annotate(tail.iloc[(take-1)-i]['WELL_NAME'], (tail.iloc[(take-1)-i]['X'], tail.iloc[(take-1)-i]['Y']))
    # Column names are opposite to object series becuase of how it makes sense when plotted
    tail = tail.reindex(tail['Contribution'].sort_values(ascending=False).index).reset_index(drop=True)
    table_data = pd.DataFrame(data={
            'Tail Well Name': head['WELL_NAME'], 
            'Tail Contribution': head['Contribution'].round(5),
            'Head Well Name': tail['WELL_NAME'],
            'Head Contribution': tail['Contribution'].round(5)
        }
    )
    if seperate_table:
        table = ax_table.table(
            cellText=table_data.values, colLabels=table_data.columns,
            loc='center', cellLoc='center'
        )
    else:
        table = ax_table.table(
            cellText=table_data.values, colLabels=table_data.columns, 
            loc='center', bbox=[-0.05, 0, 1, 0.7]
        )
    
    table.auto_set_font_size(False)  
    ax_table.axis('off')
    
    if not seperate_table:
        f.subplots_adjust(hspace=0)
          
    #%% Radius Plotting and Scaling     
    aquifer['Distance_To_Center'] = aquifer.apply(lambda x: (
        # sqrt((x2-x1)^2 + (y2-y1)^2) is Euclidean Distance
        (x['Y']-CenterY)**2 + (x['X']-CenterX)**2
    ), axis=1)
    closest_aqt = aquifer.loc[aquifer['Distance_To_Center'].idxmin()]
    displace = 200
    ax.annotate('Columbia River', (closest_aqt['X'] - (displace*2), closest_aqt['Y'] + displace), ha='left', rotation=45, fontsize=15)

    constant = 200
    x_min_bound = min(modeled_wells['X'])-constant
    x_max_bound = max(modeled_wells['X'])+constant
    y_min_bound = min(modeled_wells['Y'])-constant
    y_max_bound = max(modeled_wells['Y'])+constant
    ax.set_xlim([x_min_bound, x_max_bound])
    ax.set_ylim([y_min_bound, y_max_bound])
    
    ax.set_aspect('equal', adjustable='datalim')

    ax.set_title(title)
        
    #%% Objective Arrow Direction
    if False:
        # Define the coordinates of the surrounding points
        points = modeled_wells[['X', 'Y']].to_numpy()
        
        # Calculate Euclidean distance of surrounding points to target
        distances = np.linalg.norm(points - np.array([[CenterX, CenterY]]), axis=1)
        
        # Calculate the force vectors for each surrounding point
        if consider_distance:
            # Define the constants that determines the strength of the force based on contribution
            ks = modeled_wells['Contribution'] 
            ks = ks.to_numpy()
            
            forces = (ks[:, np.newaxis]/(distances[:, np.newaxis]**2)) * (points - np.array([[CenterX, CenterY]]))
        else:
            # Define the constants that determines the strength of the force based on contribution
            # Normalize with maximum absolute scaling as we need contributions to retain their original directions
            ks = modeled_wells['Contribution'] / modeled_wells['Contribution'].abs().max()
            ks = ks.to_numpy()
            
            forces = ks[:, np.newaxis] * (points - np.array([[CenterX, CenterY]]))
        
        # Sum the force vectors to get the total force vector on the target point
        total_force = np.sum(forces, axis=0)
        
        # Calculate the direction and magnitude of the vector
        angle_rad = np.arctan2(total_force[1], total_force[0])
        magnitude = np.linalg.norm(total_force)
        
        # Print the results
        print(f"Vector direction: {np.rad2deg(angle_rad)} degrees from x-axis")
        print(f"Vector magnitude: {magnitude}")
        
        # x and y deltas
        dx = total_force[0]
        dy = total_force[1]
        
        # Plot the arrow
        ax.quiver(CenterX, CenterY, dx, dy, angles='xy', scale_units='xy', scale=2, alpha=0.25)
        
        # Annotate magnitude and angle
        if seperate_table:
            ax.text(
                x_min_bound-1000, y_min_bound-900, 
                f'Angle of Direction: {np.rad2deg(angle_rad):.3f} Degrees', 
                bbox=dict(facecolor='white', edgecolor='black')
            )
            ax.text(
                x_min_bound+2700, y_min_bound-900, 
                f'Magnitude of Direction: {magnitude:.3f} (µg*distance)/L', 
                bbox=dict(facecolor='white', edgecolor='black')
            )
          
        else:
            ax_table.text(
                -0.06, 0.76, 
                f'Angle of Direction: {np.rad2deg(angle_rad):.3f} Degrees (From +X-Axis)', 
                bbox=dict(facecolor='white', edgecolor='black')
            )
            ax_table.text(
                0.50, 0.76, 
                f'Magnitude of Direction: {magnitude:.3f}µg/L Per Coordinate Unit', 
                bbox=dict(facecolor='white', edgecolor='black')
            )
             
    #%% Colorbar. Color maps interpolated color
    cmap = LinearSegmentedColormap.from_list('red_blue', colors=colors)
    im = ax.imshow([[1]], cmap=cmap, vmin=min_value, vmax=max_value)
    cbar = f.colorbar(im)
    cbar.set_ticks([min_value, max_value])
    cbar.set_ticklabels([f'{min_value:.4f}', f'{max_value:.4f}'])
    cbar.set_label('Contribution')
    
    #%% Drawing links
    for i in range(tail.shape[0]):
        x = [CenterX, tail.iloc[i]['X']]
        y = [CenterY, tail.iloc[i]['Y']]
        ax.plot(x, y, color=colors[len(colors) - i - 1], linestyle="--")
        
    for i in range(head.shape[0]):
        x = [CenterX, head.iloc[i]['X']]
        y = [CenterY, head.iloc[i]['Y']]
        ax.plot(x, y, color=colors[i], linestyle="--")

    if seperate_table:
        return f, f_table, head, tail
    else:
        return f, head, tail  

