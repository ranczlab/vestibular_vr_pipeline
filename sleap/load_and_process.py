import os
import numpy as np
import pandas as pd
from ellipse import LsqEllipse
from scipy.ndimage import median_filter
from . import horizontal_flip_script

def load_df(path):
    return pd.read_csv(path)

def fill_with_empty_rows_based_on_index(df, new_index_column_name='frame_idx'):
    
    complete_index = pd.Series(range(0, df[new_index_column_name].max() + 1))
    df.set_index(new_index_column_name, inplace=True)
    df = df.reindex(complete_index, fill_value=np.nan)
    df.reset_index(inplace=True)
    df.rename(columns={'index': new_index_column_name}, inplace=True)
    
    return df

def load_videography_data(path):
    
    #print('ℹ️ Load_and_process.load_videography_data() function expects the following format of SLEAP outputs:')
    #print('"VideoData1_1904-01-01T23-59-59.sleap.csv"')
    #print('"..."')
    print('ℹ️ Make sure the SLEAP files follow this convention: VideoData1_1904-01-01T00-00-00.sleap.csv')
    #print('\n')
    #print('RESULTS:')
    
    # Listing filenames in the folders
    vd1_files, read_vd1_dfs, read_vd1_sleap_dfs = [], [], []
    vd2_files, read_vd2_dfs, read_vd2_sleap_dfs = [], [], []
    
    for e in os.listdir(path/'VideoData1'):
        if not '.avi' in e and not '.sleap' in e and e[-4:] != '.slp':
            vd1_files.append(e)
    for e in os.listdir(path/'VideoData2'):
        if not '.avi' in e and not '.sleap' in e and e[-4:] != '.slp':
            vd2_files.append(e)

    vd1_has_sleap, vd2_has_sleap = False, False
    for e in os.listdir(path/'VideoData1'):
        if '.sleap' in e: vd1_has_sleap = True
    for e in os.listdir(path/'VideoData2'):
        if '.sleap' in e : vd2_has_sleap = True
    print(f'\nOutputs of SLEAP found in VideoData1: {vd1_has_sleap}')
    print(f'Outputs of SLEAP found in VideoData2: {vd2_has_sleap}')
    # Remove '.DS_Store' if found in the lists
    vd1_files = [f for f in vd1_files if f != '.DS_Store']
    vd2_files = [f for f in vd2_files if f != '.DS_Store']

    # Sorting filenames chronologically
    #sorted_vd1_files = pd.to_datetime(pd.Series([x.split('_')[1].split('.')[0] for x in vd1_files])).sort_values()
    #sorted_vd2_files = pd.to_datetime(pd.Series([x.split('_')[1].split('.')[0] for x in vd2_files])).sort_values()
    sorted_vd1_files = pd.to_datetime(pd.Series([x.split('_')[1].split('.')[0] for x in vd1_files]), format='%Y-%m-%dT%H-%M-%S').sort_values()
    sorted_vd2_files = pd.to_datetime(pd.Series([x.split('_')[1].split('.')[0] for x in vd2_files]), format='%Y-%m-%dT%H-%M-%S').sort_values()
    
    print(f'Found .csv VideoData logs timestamped at:')
    for ts in sorted_vd1_files.values:
        print('-',ts)
    
    # Reading the csv files in the chronological order
    last_sleap_frame_idx_vd1, last_sleap_frame_idx_vd2 = 0, 0 # to add to consecutive SLEAP logs because frame_idx restarts at 0 in each file
    for row in sorted_vd1_files:
        read_vd1_dfs.append(pd.read_csv(path/'VideoData1'/f"VideoData1_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"))
        if vd1_has_sleap: 
            #read_vd1_sleap_dfs.append(pd.read_csv(path/'VideoData1'/f'VideoData1_{row.strftime('%Y-%m-%dT%H-%M-%S')}.sleap.csv'))
            read_vd1_sleap_dfs.append(
                pd.read_csv(
                    path / 'VideoData1' / f"VideoData1_{row.strftime('%Y-%m-%dT%H-%M-%S')}.sleap.csv"
                )
            )
            read_vd1_sleap_dfs[-1]['frame_idx'] = read_vd1_sleap_dfs[-1]['frame_idx'] + last_sleap_frame_idx_vd1
            last_sleap_frame_idx_vd1 = read_vd1_sleap_dfs[-1]['frame_idx'].iloc[-1] + 1
    for row in sorted_vd2_files:
        read_vd2_dfs.append(pd.read_csv(path/'VideoData2'/f"VideoData2_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"))
        if vd2_has_sleap: 
            read_vd2_sleap_dfs.append(pd.read_csv(path/'VideoData2'/f"VideoData2_{row.strftime('%Y-%m-%dT%H-%M-%S')}.sleap.csv"))
            read_vd2_sleap_dfs[-1]['frame_idx'] = read_vd2_sleap_dfs[-1]['frame_idx'] + last_sleap_frame_idx_vd2
            last_sleap_frame_idx_vd2 = read_vd2_sleap_dfs[-1]['frame_idx'].iloc[-1] + 1
    read_vd1_dfs = pd.concat(read_vd1_dfs).reset_index().drop(columns='index')
    read_vd2_dfs = pd.concat(read_vd2_dfs).reset_index().drop(columns='index')
    if vd1_has_sleap: read_vd1_sleap_dfs = pd.concat(read_vd1_sleap_dfs).reset_index().drop(columns='index')
    if vd2_has_sleap: read_vd2_sleap_dfs = pd.concat(read_vd2_sleap_dfs).reset_index().drop(columns='index')
        
    #print('Reading dataframes finished.')
    
    read_vd1_dfs = read_vd1_dfs.rename(columns={"Value.ChunkData.FrameID": "frame_idx"})
    read_vd2_dfs = read_vd2_dfs.rename(columns={"Value.ChunkData.FrameID": "frame_idx"})
    
    # Resetting frame_idx to start at 0
    read_vd1_dfs['frame_idx'] = read_vd1_dfs['frame_idx'] - read_vd1_dfs['frame_idx'].iloc[0]
    read_vd2_dfs['frame_idx'] = read_vd2_dfs['frame_idx'] - read_vd2_dfs['frame_idx'].iloc[0]
    
    # Filling in the skipped frames (if there are any) with NaN rows
    if vd1_has_sleap:
        if read_vd1_sleap_dfs.index[-1] != read_vd1_sleap_dfs['frame_idx'].iloc[-1]:
            #print(f'VideoData1 SLEAP output: {read_vd1_sleap_dfs['frame_idx'].iloc[-1] + 1} frames registered, but {read_vd1_sleap_dfs.index[-1] + 1} rows found inside file. Filling with empty rows.')
            frame_count = read_vd1_sleap_dfs['frame_idx'].iloc[-1] + 1
            row_count = read_vd1_sleap_dfs.index[-1] + 1
            dropped_frames_vd1 = frame_count - row_count
            print(f"ℹ️ VideoData1 has {dropped_frames_vd1} dropped frames. Filling missing frames with empty rows.")
            read_vd1_sleap_dfs = fill_with_empty_rows_based_on_index(read_vd1_sleap_dfs)
    if vd2_has_sleap:
        if read_vd2_sleap_dfs.index[-1] != read_vd2_sleap_dfs['frame_idx'].iloc[-1]:
            #print(f'VideoData2 SLEAP output: {read_vd2_sleap_dfs['frame_idx'].iloc[-1] + 1} frames registered, but {read_vd2_sleap_dfs.index[-1] + 1} rows found inside file. Filling with empty rows.')
            frame_count = read_vd2_sleap_dfs['frame_idx'].iloc[-1] + 1
            row_count = read_vd2_sleap_dfs.index[-1] + 1
            dropped_frames_vd2 = frame_count - row_count
            print(f"ℹ️ VideoData2 has {dropped_frames_vd2} dropped frames. Filling missing frames with empty rows.")
            read_vd2_sleap_dfs = fill_with_empty_rows_based_on_index(read_vd2_sleap_dfs)
    
    # Merging VideoData csv files and sleap outputs to get access to the HARP timestamps
    vd1_out, vd2_out = read_vd1_dfs[['frame_idx', 'Seconds']], read_vd2_dfs[['frame_idx', 'Seconds']]
    if vd1_has_sleap: vd1_out = pd.merge(read_vd1_dfs[['frame_idx', 'Seconds']], read_vd1_sleap_dfs, on='frame_idx')
    if vd2_has_sleap: vd2_out = pd.merge(read_vd2_dfs[['frame_idx', 'Seconds']], read_vd2_sleap_dfs, on='frame_idx')
    
    return vd1_out, vd2_out, vd1_has_sleap, vd2_has_sleap

def recalculated_coordinates(point_name, df, reference_subtraced_displacements_dict):
    # Recalculates coordinates of a point at each frame, applying the referenced displacements to the coordinates of the very first frame.
    out_array = np.zeros(reference_subtraced_displacements_dict[point_name].shape[0]+1)
    out_array[0] = df[point_name].to_numpy()[0]
    for i, disp in enumerate(reference_subtraced_displacements_dict[point_name]):
        out_array[i+1] = out_array[i] + disp
        
    return out_array

def get_referenced_recalculated_coordinates(df):
    columns_of_interest = ['left.x','left.y','center.x','center.y','right.x','right.y','p1.x','p1.y','p2.x','p2.y','p3.x','p3.y','p4.x','p4.y','p5.x','p5.y','p6.x','p6.y','p7.x','p7.y','p8.x','p8.y']
    active_points_x = ['center.x','p1.x','p2.x','p3.x','p4.x','p5.x','p6.x','p7.x','p8.x']
    active_points_y = ['center.y','p1.y','p2.y','p3.y','p4.y','p5.y','p6.y','p7.y','p8.y']

    coordinates_dict = {key:df[key].to_numpy() for key in columns_of_interest}
    displacements_dict = {k:np.diff(v) for k, v in coordinates_dict.items()} # in [displacement] = [pixels / frame]

    mean_reference_x = np.stack((displacements_dict['left.x'], displacements_dict['right.x'])).mean(axis=0)
    mean_reference_y = np.stack((displacements_dict['left.y'], displacements_dict['right.y'])).mean(axis=0)

    # Subtracting the displacement of the reference points at each frame
    reference_subtraced_displacements_dict = {k:displacements_dict[k]-mean_reference_x for k in active_points_x} | {k:displacements_dict[k]-mean_reference_y for k in active_points_y} # joining the horizontal and vertical dictionaries into one

    reference_subtraced_coordinates_dict = {p:recalculated_coordinates(p, df, reference_subtraced_displacements_dict) for p in active_points_x + active_points_y}

    return reference_subtraced_coordinates_dict

def rotate_points(points, theta):
    # This is for rotating with an angle of positive theta
    # rotation_matrix = np.array([
    #     [np.cos(theta), -np.sin(theta)],
    #     [np.sin(theta), np.cos(theta)]
    # ])
    # This is for rotating with an angle of negative theta
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    rotated_points = points.dot(rotation_matrix.T)
    return rotated_points

def get_rotated_points(point_name, theta, reference_subtraced_coordinates_dict):
    # mean_center_coord = np.stack([reference_subtraced_coordinates_dict[f'center.x'], reference_subtraced_coordinates_dict[f'center.y']], axis=1).mean(axis=0)
    temp_points = np.stack([reference_subtraced_coordinates_dict[f'{point_name}.x'], reference_subtraced_coordinates_dict[f'{point_name}.y']], axis=1)
    temp_mean_center_coord = temp_points.mean(axis=0)
    centered_points = temp_points.copy()
    centered_points[:,0] = centered_points[:,0] - temp_mean_center_coord[0]
    centered_points[:,1] = centered_points[:,1] - temp_mean_center_coord[1]
    rotated_points = rotate_points(centered_points, theta)
    rotated_points[:,0] = rotated_points[:,0] + temp_mean_center_coord[0]
    rotated_points[:,1] = rotated_points[:,1] + temp_mean_center_coord[1]
    return rotated_points

def find_horizontal_axis_angle(df, point1='left', point2='center'):
    # Fits a line between original (unreferenced) left reference point and center of the pupil point, return the angle of the line
    line_fn = np.polyfit(np.hstack([df[f'{point1}.x'].to_numpy(), df[f'{point2}.x'].to_numpy()]), np.hstack([df[f'{point1}.y'].to_numpy(), df[f'{point2}.y'].to_numpy()]), 1)
    line_fn = np.poly1d(line_fn)
    theta = np.arctan(line_fn[1])
    return theta

def moving_average_smoothing(X,k):
    S = np.zeros(X.shape[0])
    for t in range(X.shape[0]):
        if t < k:
            S[t] = np.mean(X[:t+1])
        else:
            S[t] = np.sum(X[t-k:t])/k
    return S

def median_filter_smoothing(X, k):
    return median_filter(X, size=k)

def find_sequential_groups(arr):
    groups = []
    current_group = [arr[0]]
    
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1] + 1:
            current_group.append(arr[i])
        else:
            groups.append(current_group)
            current_group = [arr[i]]
    groups.append(current_group)
    
    return groups

def detect_saccades_per_point_per_direction(rotated_points):

    # Expects rotated_points to be a 1D array representing coordinate time series of one point in one direction (either X or Y)

    displacement_time_series = np.diff(rotated_points) # pixels per frame

    threshold = displacement_time_series.mean() + displacement_time_series.std() * 3 # chosen threshold

    detected_peaks_inds = np.where(np.abs(displacement_time_series) > threshold)[0]

    # Collecting max value deteceted saccades
    # into a nested list = [[saccade_0_index, saccade_0_velocity_amplitude], [saccade_1_index, saccade_1_velocity_amplitude], ...]
    detected_max_saccades = []

    for group in find_sequential_groups(detected_peaks_inds):
        max_amplitude_relative_ind = np.abs(displacement_time_series[group]).argmax()
        max_amplitude_ind = group[max_amplitude_relative_ind]
        max_amplitude_value = displacement_time_series[max_amplitude_ind]
        detected_max_saccades.append([max_amplitude_ind, max_amplitude_value])

    detected_max_saccades = np.array(detected_max_saccades)

    return detected_max_saccades

def get_all_detected_saccades(path):
    active_points = ['center', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
    df = load_df(path)
    reference_subtraced_coordinates_dict = get_referenced_recalculated_coordinates(df)
    theta = find_horizontal_axis_angle(df)

    all_detected_saccades = {point:{"X":[], "Y":[]} for point in active_points}
    for point in active_points:
        rotated_points = get_rotated_points(point, theta, reference_subtraced_coordinates_dict)
        all_detected_saccades[point]["X"] = detect_saccades_per_point_per_direction(rotated_points[:,0])
        all_detected_saccades[point]["Y"] = detect_saccades_per_point_per_direction(rotated_points[:,1])
    
    return all_detected_saccades

def get_eight_points_at_time(data_dict, point_name_list, t):
    points_coord_data = []
    for point in point_name_list:
        points_coord_data.append(data_dict[point][t,:])
    return np.stack(points_coord_data, axis=0)

def get_fitted_ellipse_parameters(coordinates_dict, columns_of_interest):

    # Collecting parameters of the fitted ellipse into an array over the whole recording
    # ellipse_parameters_data contents = (width, height, phi)
    # ellipse_center_points_data = (center_x, center_y)
    ellipse_parameters_data = []
    ellipse_center_points_data = []
    for t in range(coordinates_dict[list(coordinates_dict.keys())[0]].shape[0]):
        reg = LsqEllipse().fit(get_eight_points_at_time(coordinates_dict, columns_of_interest, t))
        center, width, height, phi = reg.as_parameters()
        ellipse_parameters_data.append([width, height, phi])
        ellipse_center_points_data.append(center)
    ellipse_parameters_data = np.array(ellipse_parameters_data)
    ellipse_center_points_data = np.array(ellipse_center_points_data)

    return ellipse_parameters_data, ellipse_center_points_data

def get_coordinates_dict(df, columns_of_interest):
    return {key:df[key].to_numpy() for key in columns_of_interest}

def get_left_right_center_point(coordinates_dict):
    x = np.hstack([coordinates_dict['left.x'], coordinates_dict['right.x']]).mean()
    y = np.hstack([coordinates_dict['left.y'], coordinates_dict['right.y']]).mean()
    return (x, y)

def get_reformatted_coordinates_dict(coordinates_dict, columns_of_interest):
    # Combining separated x and y number arrays into (samples, 2)-shaped array and subtracting the inferred center point from above
    return {p:np.stack([coordinates_dict[f'{p}.x'], coordinates_dict[f'{p}.y']], axis=1) for p in columns_of_interest}

def get_centered_coordinates_dict(coordinates_dict, center_point):
    return {point: arr - center_point for point, arr in coordinates_dict.items()}

def get_rotated_coordinates_dict(coordinates_dict, theta):
    return {point: rotate_points(arr, theta) for point, arr in coordinates_dict.items()}

def create_flipped_videos(path, what_to_flip='VideoData1'):
    
    if len([x for x in os.listdir(path / what_to_flip) if '.flipped.' in x]) != 0:
        print(f'Flipped videos already exist in {path/what_to_flip}. Exiting.')
    else:
        avis = [x for x in os.listdir(path / what_to_flip) if x[-4:] == '.avi']
        for avi in avis:
            horizontal_flip_script.horizontal_flip_avi(path / what_to_flip / avi, path / what_to_flip / f'{avi[:-4]}.flipped.avi')

def find_sequential_groups(arr):

    groups = []
    current_group = [arr[0]]
    
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1] + 1:
            current_group.append(arr[i])
        else:
            groups.append(current_group)
            current_group = [arr[i]]
    groups.append(current_group)
    
    return groups

def detect_saccades_with_threshold(eye_data_stream, threshold_std_times=1):

    harp_time_inds, absolute_positions = eye_data_stream.index, eye_data_stream.values

    framerate = 60
    print(f'Assuming camera frame rate of {framerate} Hz')

    derivative_of_position = np.diff(absolute_positions) * framerate

    threshold = derivative_of_position.mean() + derivative_of_position.std() * threshold_std_times

    detected_peaks_inds = np.where(np.abs(derivative_of_position) > threshold)[0]

    # Correcting for over-counted peaks - collecting max value points within groups crossing the threshold
    # detected_max_saccades is a nested list = [[saccade_0_index, saccade_0_velocity_amplitude], [saccade_1_index, saccade_1_velocity_amplitude], ...]
    detected_max_saccades = []

    for group in find_sequential_groups(detected_peaks_inds):
        max_amplitude_relative_ind = np.abs(derivative_of_position[group]).argmax()
        max_amplitude_ind = group[max_amplitude_relative_ind]
        max_amplitude_value = derivative_of_position[max_amplitude_ind]
        detected_max_saccades.append([max_amplitude_ind, max_amplitude_value])

    detected_max_saccades = np.array(detected_max_saccades)
    detected_max_saccades_inds = (detected_max_saccades[:,0].astype(int))

    total_number = detected_max_saccades.shape[0]

    print(f'Found {total_number} saccades with chosen threshold of {threshold} (mean + {threshold_std_times} times the standard deviation).')

    runtime = harp_time_inds[-1] - harp_time_inds[0]

    print(f'Saccade frequency = {(total_number / runtime) * 60} events per minute')

    return harp_time_inds[detected_max_saccades_inds], detected_max_saccades[:,1]

def calculate_saccade_frequency_within_time_range():
    pass