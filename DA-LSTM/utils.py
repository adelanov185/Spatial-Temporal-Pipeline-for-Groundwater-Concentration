import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy.interpolate import interp1d

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
    #ax.scatter(aqtX, aqtY, color='r', marker='*', s=142)

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
            'Tail Attentions': head['Contribution'].round(5),
            'Head Well Name': tail['WELL_NAME'],
            'Head Attentions': tail['Contribution'].round(5)
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
    cbar.set_label('Feature Attention')
    
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

def plot_links_2(
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
    z = np.array(modeled_wells['Contribution'].to_list())
    
    # # Define the two colors for the gradient
    # # color_start = np.array([0.1, 0.1, 1.0])  # blue for negative values
    # color_end = np.array([1.0, 0.1, 0.1])   # red for positive values
    
    # # Create the gradient array
    if min_value is None:
        min_value = np.min(z)
    if max_value is None:
        max_value = np.max(z)
    # num_color_samples = len(z)//2 if color_sam is None else color_sam
    # if min_value < 0 and max_value > 0:
    #     # Do not allow for a white color by limiting negative and positive gradient color limits
    #     #gradient_start = np.linspace(color_start, [0.9, 0.9, 1.0], num_color_samples)
    #     gradient_end = np.linspace([1.0, 0.9, 0.9], color_end, num_color_samples)
        
    #     # Combine blue and red gradients
    #     # gradient = np.vstack((gradient_start, gradient_end))
    #     gradient = gradient_end

    #     # Interpolate the colors based on the contirbution values
    #     colors = np.zeros((len(z), 3))
    #     for i in range(len(z)):
    #         if z[i] < 0:
    #             # Reverse indexing from half point in gradient for negative numbers as z is sorted from negative to positve
    #             # This produces divergent gradient between blue and red, rather than two stacked gradients  idx = num_color_samples - int((z[i] / min_value) * (num_color_samples - 1))
    #             idx = num_color_samples - int((z[i] / min_value) * (num_color_samples - 1))
    #             if idx == num_color_samples:
    #                 idx = num_color_samples - 1
    #             colors[i, :] = gradient[idx]
    #         elif z[i] > 0:
    #             colors[i, :] = gradient[num_color_samples + int((z[i] / max_value) * (num_color_samples - 1))]
    #         else:
    #             # White color for zero values
    #             colors[i, :] = [0.9, 0.9, 0.9]
    # else:
    #     num_color_samples = len(z)
    #     colors = np.zeros((len(z), 3))
    #     if max_value > 0:
    #         # Do not allow for a white color by limiting negative and positive gradient color limits
    #         gradient = np.linspace([1.0, 0.9, 0.9], color_end, num_color_samples)
    #         for i in range(len(z)):
    #             colors[i, :] = gradient[int((z[i] / max_value) * (num_color_samples - 1))]
    #     else:
    #         gradient = np.linspace(color_start, [0.9, 0.9, 1.0], num_color_samples)
    #         for i in range(len(z)):
    #             colors[i, :] = gradient[int((z[i] / min_value) * (num_color_samples - 1))]

    # z_norm = (z - np.min(z)) / (np.max(z) - np.min(z))
    # cmap = plt.get_cmap('Reds')
    # colors = cmap(z_norm)

    #z_norm = (z - np.min(z)) / (np.max(z) - np.min(z))

    z_clip = np.clip(z, np.percentile(z, 1), np.percentile(z, 99))  # Clip extreme values
    # z_norm = (z_clip - np.min(z_clip)) / (np.max(z_clip) - np.min(z_clip))

    # z_sqrt = np.sqrt(z_clip)
    # z_norm = (z_sqrt - np.min(z_sqrt)) / (np.max(z_sqrt) - np.min(z_sqrt))

    z_exp = np.exp(z_clip * 2)
    z_norm = (z_exp - np.min(z_exp)) / (np.max(z_exp) - np.min(z_exp))

    c_range = [(1, 0, 0, 0), (1, 0, 0, 1)]
    cmap = LinearSegmentedColormap.from_list('low_alpha_to_dark_red', c_range)
    colors = cmap(z_norm)
    
    # Plot contributions
    sc = ax.scatter(modeled_wells['X'], modeled_wells['Y'], c=colors)    
    
    #%% Annotate top 5 impacts on each direction
    if seperate_table:
        f_table, ax_table = plt.subplots(1, 1, figsize=(8,2), dpi=500)
    
    take = 5
    # head = modeled_wells.head(take).reset_index(drop=True)
    head = None
    tail = modeled_wells.tail(take).reset_index(drop=True)
    tail_table = modeled_wells.tail(take+5).reset_index(drop=True)
    for i in range(take):
        # Blues
        # ax.annotate(head.iloc[i]['WELL_NAME'], (head.iloc[i]['X'], head.iloc[i]['Y']))
        # Reds
        ax.annotate(tail.iloc[(take-1)-i]['WELL_NAME'], (tail.iloc[(take-1)-i]['X'], tail.iloc[(take-1)-i]['Y']))
    # Column names are opposite to object series becuase of how it makes sense when plotted
    tail_table = tail_table.reindex(tail_table['Contribution'].sort_values(ascending=False).index).reset_index(drop=True)
    table_data = pd.DataFrame(data={
            # 'Tail Well Name': head['WELL_NAME'], 
            # 'Tail Attentions': head['Contribution'].round(5),
            'Head Well Name': tail_table['WELL_NAME'],
            'Head Attentions': tail_table['Contribution'].round(5)
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
    # cmap = LinearSegmentedColormap.from_list('red_blue', colors=colors)
    im = ax.imshow([[1]], cmap=cmap, vmin=min_value, vmax=max_value)
    cbar = f.colorbar(im)
    cbar.set_ticks([min_value, max_value])
    cbar.set_ticklabels([f'{min_value:.4f}', f'{max_value:.4f}'])
    cbar.set_label('Feature Attention')
    
    #%% Drawing links
    for i in range(tail.shape[0]):
        x = [CenterX, tail.iloc[i]['X']]
        y = [CenterY, tail.iloc[i]['Y']]
        ax.plot(x, y, color=colors[len(colors) - i - 1], linestyle="--")
        
    # for i in range(head.shape[0]):
    #     x = [CenterX, head.iloc[i]['X']]
    #     y = [CenterY, head.iloc[i]['Y']]
    #     ax.plot(x, y, color=colors[i], linestyle="--")

    if seperate_table:
        return f, f_table, head, tail
    else:
        return f, head, tail  
