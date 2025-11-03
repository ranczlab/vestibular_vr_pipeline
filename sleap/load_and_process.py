import os
import numpy as np
import pandas as pd
from ellipse import LsqEllipse
from scipy.ndimage import median_filter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
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
    
    print(f'\n📋 LOADING {len(sorted_vd1_files)} VideoData1 file(s) and {len(sorted_vd2_files)} VideoData2 file(s)\n')
    
    # Reading the csv files in the chronological order
    # Track actual video frame counts for proper offset calculation (fixes concatenated video frame number mismatch)
    # The issue: SLEAP frame_idx is offset by processed frame count, but concatenated video includes ALL frames
    # Solution: Track actual video file frame counts and use them to create proper mapping
    video_frame_counts_vd1, video_frame_counts_vd2 = [], []  # Track frame counts per file
    last_video_frame_offset_vd1, last_video_frame_offset_vd2 = 0, 0  # Cumulative offsets for concatenated video
    last_sleap_frame_idx_vd1, last_sleap_frame_idx_vd2 = 0, 0 # to add to consecutive SLEAP logs because frame_idx restarts at 0 in each file
    
    for row in sorted_vd1_files:
        vd1_df = pd.read_csv(path/'VideoData1'/f"VideoData1_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv")
        read_vd1_dfs.append(vd1_df)
        
        # Calculate actual frame count from VideoData CSV (before reset)
        # Value.ChunkData.FrameID represents actual video frame numbers from acquisition system
        # The number of rows represents all frames in the original video file
        actual_frame_count = len(vd1_df)  # This represents all frames in the original video file
        video_frame_counts_vd1.append(actual_frame_count)
        
        if vd1_has_sleap: 
            vd1_sleap_filename = f"VideoData1_{row.strftime('%Y-%m-%dT%H-%M-%S')}.sleap.csv"
            sleap_df = pd.read_csv(
                path / 'VideoData1' / vd1_sleap_filename
            )
            
            # Determine if SLEAP frame_idx is 0-based (per-file) or FrameID-based (continuous camera counter)
            first_sleap_frame_idx = sleap_df['frame_idx'].iloc[0]
            if len(read_vd1_sleap_dfs) == 0:
                # First file - log what we see to help diagnose
                print(f"ℹ️ VideoData1 SLEAP file {len(read_vd1_sleap_dfs) + 1} ({vd1_sleap_filename}): first frame_idx = {first_sleap_frame_idx}")
            
            # CRITICAL FIX: Gap-fill PER FILE before offsetting to ensure gaps are filled within each file's range
            file_num = len(read_vd1_sleap_dfs) + 1
            max_sleap_frame_idx = sleap_df['frame_idx'].max()
            min_sleap_frame_idx = sleap_df['frame_idx'].min()
            sleap_row_count = len(sleap_df)
            
            # Check if there are gaps (dropped frames) within this file
            # We only gap-fill WITHIN the range that SLEAP actually processed
            if sleap_row_count < max_sleap_frame_idx + 1:
                # There are gaps within the processed range - fill them
                dropped_frames = (max_sleap_frame_idx + 1) - sleap_row_count
                print(f"   ⚠️ Found {dropped_frames} dropped frames within processed range [0-{max_sleap_frame_idx}] in {vd1_sleap_filename}. Filling gaps.")
                sleap_df = fill_with_empty_rows_based_on_index(sleap_df)
            
            if max_sleap_frame_idx + 1 < actual_frame_count:
                missing_at_end = actual_frame_count - (max_sleap_frame_idx + 1)
                print(f"   ⚠️ SLEAP processed {max_sleap_frame_idx + 1} frames, but video has {actual_frame_count} frames.")
                print(f"      Missing {missing_at_end} frames at the end (will extend later).")
            
            # CRITICAL: Offset SLEAP frame_idx by cumulative actual video frame count
            # This assumes SLEAP frame_idx is 0-based per file (restarts at 0 for each file)
            sleap_df['frame_idx'] = sleap_df['frame_idx'] + last_video_frame_offset_vd1
            
            read_vd1_sleap_dfs.append(sleap_df)
            
            # Update offset for next file: use actual video frame count (not SLEAP processed count)
            last_video_frame_offset_vd1 += actual_frame_count
            
            # Also track SLEAP frame_idx max for internal consistency (if needed elsewhere)
            last_sleap_frame_idx_vd1 = read_vd1_sleap_dfs[-1]['frame_idx'].iloc[-1] + 1
            
    for row in sorted_vd2_files:
        vd2_df = pd.read_csv(path/'VideoData2'/f"VideoData2_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv")
        read_vd2_dfs.append(vd2_df)
        
        # Calculate actual frame count from VideoData CSV (before reset)
        actual_frame_count = len(vd2_df)
        video_frame_counts_vd2.append(actual_frame_count)
            
        if vd2_has_sleap: 
            vd2_sleap_filename = f"VideoData2_{row.strftime('%Y-%m-%dT%H-%M-%S')}.sleap.csv"
            sleap_df = pd.read_csv(path/'VideoData2'/vd2_sleap_filename)
            
            # Determine if SLEAP frame_idx is 0-based (per-file) or FrameID-based (continuous camera counter)
            first_sleap_frame_idx = sleap_df['frame_idx'].iloc[0]
            if len(read_vd2_sleap_dfs) == 0:
                # First file - log what we see to help diagnose
                print(f"ℹ️ VideoData2 SLEAP file {len(read_vd2_sleap_dfs) + 1} ({vd2_sleap_filename}): first frame_idx = {first_sleap_frame_idx}")
            
            # CRITICAL FIX: Gap-fill PER FILE before offsetting to ensure gaps are filled within each file's range
            file_num = len(read_vd2_sleap_dfs) + 1
            max_sleap_frame_idx = sleap_df['frame_idx'].max()
            min_sleap_frame_idx = sleap_df['frame_idx'].min()
            sleap_row_count = len(sleap_df)
            
            # Check if there are gaps (dropped frames) within this file
            # We only gap-fill WITHIN the range that SLEAP actually processed
            if sleap_row_count < max_sleap_frame_idx + 1:
                # There are gaps within the processed range - fill them
                dropped_frames = (max_sleap_frame_idx + 1) - sleap_row_count
                print(f"   ⚠️ Found {dropped_frames} dropped frames within processed range [0-{max_sleap_frame_idx}] in {vd2_sleap_filename}. Filling gaps.")
                sleap_df = fill_with_empty_rows_based_on_index(sleap_df)
            
            if max_sleap_frame_idx + 1 < actual_frame_count:
                missing_at_end = actual_frame_count - (max_sleap_frame_idx + 1)
                print(f"   ⚠️ SLEAP processed {max_sleap_frame_idx + 1} frames, but video has {actual_frame_count} frames.")
                print(f"      Missing {missing_at_end} frames at the end (will extend later).")
            
            # CRITICAL: Offset SLEAP frame_idx by cumulative actual video frame count
            # This assumes SLEAP frame_idx is 0-based per file (restarts at 0 for each file)
            sleap_df['frame_idx'] = sleap_df['frame_idx'] + last_video_frame_offset_vd2
            
            read_vd2_sleap_dfs.append(sleap_df)
            
            # Update offset for next file: use actual video frame count (not SLEAP processed count)
            last_video_frame_offset_vd2 += actual_frame_count
            
            # Also track SLEAP frame_idx max for internal consistency (if needed elsewhere)
            last_sleap_frame_idx_vd2 = read_vd2_sleap_dfs[-1]['frame_idx'].iloc[-1] + 1
    
    # CRITICAL FIX: Reset VideoData frame_idx BEFORE concatenation
    # Value.ChunkData.FrameID is a continuous camera counter across files (doesn't restart at 0)
    # Example: File 1 FrameID [26654, ..., 31653], File 2 FrameID [31654, ..., 36653]
    # We need to map to concatenated video frame numbers: [0, 1, 2, ..., N-1]
    # Strategy: Capture first file's starting FrameID, then map all files relative to that
    
    # Capture starting FrameID from first file BEFORE any modifications (for VideoData1)
    first_file_frame_id_vd1 = None
    if len(read_vd1_dfs) > 0:
        # Temporarily rename to access FrameID, then rename back
        temp_df = read_vd1_dfs[0].copy()
        if "Value.ChunkData.FrameID" in temp_df.columns:
            first_file_frame_id_vd1 = temp_df["Value.ChunkData.FrameID"].iloc[0]
    
    # Capture starting FrameID from first file BEFORE any modifications (for VideoData2)
    first_file_frame_id_vd2 = None
    if len(read_vd2_dfs) > 0:
        temp_df = read_vd2_dfs[0].copy()
        if "Value.ChunkData.FrameID" in temp_df.columns:
            first_file_frame_id_vd2 = temp_df["Value.ChunkData.FrameID"].iloc[0]
    
    # Reset frame_idx for each VideoData file before concatenation
    # Since FrameID is continuous across files, we simply subtract the first file's starting FrameID
    # This automatically creates continuous frame_idx values: [0, 1, 2, ..., N-1]
    for i in range(len(read_vd1_dfs)):
        read_vd1_dfs[i] = read_vd1_dfs[i].rename(columns={"Value.ChunkData.FrameID": "frame_idx"})
        if first_file_frame_id_vd1 is not None:
            read_vd1_dfs[i]['frame_idx'] = read_vd1_dfs[i]['frame_idx'] - first_file_frame_id_vd1
    
    for i in range(len(read_vd2_dfs)):
        read_vd2_dfs[i] = read_vd2_dfs[i].rename(columns={"Value.ChunkData.FrameID": "frame_idx"})
        if first_file_frame_id_vd2 is not None:
            read_vd2_dfs[i]['frame_idx'] = read_vd2_dfs[i]['frame_idx'] - first_file_frame_id_vd2
    
    # Now concatenate with properly aligned frame_idx values (matching concatenated video frame numbers)
    read_vd1_dfs = pd.concat(read_vd1_dfs).reset_index().drop(columns='index')
    read_vd2_dfs = pd.concat(read_vd2_dfs).reset_index().drop(columns='index')
    if vd1_has_sleap: read_vd1_sleap_dfs = pd.concat(read_vd1_sleap_dfs).reset_index().drop(columns='index')
    if vd2_has_sleap: read_vd2_sleap_dfs = pd.concat(read_vd2_sleap_dfs).reset_index().drop(columns='index')
        
    #print('Reading dataframes finished.')
    
    # CRITICAL FIX: After gap-filling per file and concatenation, extend to total actual video frame count
    # Gap-filling has already been done per file before concatenation, so now we just need to extend
    # if the maximum frame_idx is less than the total actual frames (due to dropped frames at file boundaries)
    if vd1_has_sleap:
        # Calculate total actual frame count from all video files
        total_actual_frames_vd1 = sum(video_frame_counts_vd1) if video_frame_counts_vd1 else 0
        max_frame_idx_vd1 = read_vd1_sleap_dfs['frame_idx'].max()
        
        # If SLEAP max frame_idx is less than total actual frames, extend with NaN rows
        if max_frame_idx_vd1 < total_actual_frames_vd1 - 1:
            missing_frames = total_actual_frames_vd1 - 1 - max_frame_idx_vd1
            print(f"ℹ️ VideoData1: Extending SLEAP dataframe by {missing_frames} frames to match concatenated video frame count ({total_actual_frames_vd1} total frames)")
            # Create additional rows with NaN values for missing frame_idx values
            extended_frame_idx = pd.Series(range(max_frame_idx_vd1 + 1, total_actual_frames_vd1))
            extended_df = pd.DataFrame({'frame_idx': extended_frame_idx})
            # Add NaN columns for all other SLEAP columns
            for col in read_vd1_sleap_dfs.columns:
                if col != 'frame_idx':
                    extended_df[col] = np.nan
            # Concatenate with existing dataframe
            read_vd1_sleap_dfs = pd.concat([read_vd1_sleap_dfs, extended_df], ignore_index=True)
            # Sort by frame_idx to maintain order
            read_vd1_sleap_dfs = read_vd1_sleap_dfs.sort_values('frame_idx').reset_index(drop=True)
    
    if vd2_has_sleap:
        # Calculate total actual frame count from all video files
        total_actual_frames_vd2 = sum(video_frame_counts_vd2) if video_frame_counts_vd2 else 0
        max_frame_idx_vd2 = read_vd2_sleap_dfs['frame_idx'].max()
        
        # If SLEAP max frame_idx is less than total actual frames, extend with NaN rows
        if max_frame_idx_vd2 < total_actual_frames_vd2 - 1:
            missing_frames = total_actual_frames_vd2 - 1 - max_frame_idx_vd2
            print(f"ℹ️ VideoData2: Extending SLEAP dataframe by {missing_frames} frames to match concatenated video frame count ({total_actual_frames_vd2} total frames)")
            # Create additional rows with NaN values for missing frame_idx values
            extended_frame_idx = pd.Series(range(max_frame_idx_vd2 + 1, total_actual_frames_vd2))
            extended_df = pd.DataFrame({'frame_idx': extended_frame_idx})
            # Add NaN columns for all other SLEAP columns
            for col in read_vd2_sleap_dfs.columns:
                if col != 'frame_idx':
                    extended_df[col] = np.nan
            # Concatenate with existing dataframe
            read_vd2_sleap_dfs = pd.concat([read_vd2_sleap_dfs, extended_df], ignore_index=True)
            # Sort by frame_idx to maintain order
            read_vd2_sleap_dfs = read_vd2_sleap_dfs.sort_values('frame_idx').reset_index(drop=True)
    
    # Merging VideoData csv files and sleap outputs to get access to the HARP timestamps
    # CRITICAL: Use LEFT merge to preserve VideoData frame order and ensure 1:1 mapping
    # This ensures frame_idx positions in the merged dataframe match VideoData frame_idx positions
    vd1_out, vd2_out = read_vd1_dfs[['frame_idx', 'Seconds']], read_vd2_dfs[['frame_idx', 'Seconds']]
    if vd1_has_sleap: 
        # Use left merge to keep VideoData order and ensure each VideoData row gets matched to SLEAP
        # Sort SLEAP by frame_idx first to ensure proper matching
        read_vd1_sleap_dfs_sorted = read_vd1_sleap_dfs.sort_values('frame_idx').reset_index(drop=True)
        vd1_out = pd.merge(read_vd1_dfs[['frame_idx', 'Seconds']], read_vd1_sleap_dfs_sorted, on='frame_idx', how='left')
        # Check for merge issues
        if len(vd1_out) != len(read_vd1_dfs):
            print(f"   ⚠️ WARNING: VideoData1 row count changed during merge!")
            print(f"      Original VideoData: {len(read_vd1_dfs)} rows")
            print(f"      After merge: {len(vd1_out)} rows")
            print(f"      Difference: {len(vd1_out) - len(read_vd1_dfs)} rows")
    if vd2_has_sleap: 
        read_vd2_sleap_dfs_sorted = read_vd2_sleap_dfs.sort_values('frame_idx').reset_index(drop=True)
        vd2_out = pd.merge(read_vd2_dfs[['frame_idx', 'Seconds']], read_vd2_sleap_dfs_sorted, on='frame_idx', how='left')
        # Check for merge issues
        if len(vd2_out) != len(read_vd2_dfs):
            print(f"   ⚠️ WARNING: VideoData2 row count changed during merge!")
            print(f"      Original VideoData: {len(read_vd2_dfs)} rows")
            print(f"      After merge: {len(vd2_out)} rows")
            print(f"      Difference: {len(vd2_out) - len(read_vd2_dfs)} rows")
    
    print('\n' + '='*80)
    print('✅ DIAGNOSTIC SUMMARY: Frame Index Alignment Complete')
    print('='*80)
    if len(vd1_out) > 0:
        print(f'VideoData1 final dataframe: {len(vd1_out)} rows, frame_idx range {vd1_out["frame_idx"].min()}-{vd1_out["frame_idx"].max()}')
    if len(vd2_out) > 0:
        print(f'VideoData2 final dataframe: {len(vd2_out)} rows, frame_idx range {vd2_out["frame_idx"].min()}-{vd2_out["frame_idx"].max()}')
    print('='*80 + '\n')
    
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

def find_horizontal_axis_angle(df, point1='left', point2='center', min_valid_points=50):
    """
    Fits a line between original (unreferenced) reference points to determine horizontal axis angle.
    Filters out NaN values before fitting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing coordinate data
    point1 : str
        First point name (default: 'left')
    point2 : str
        Second point name (default: 'center')
    min_valid_points : int
        Minimum number of valid (non-NaN) point pairs required for fitting (default: 50)
    
    Returns:
    --------
    float : Angle of the fitted line in radians
    
    Raises:
    -------
    ValueError : If insufficient valid data points are available
    """
    # Extract x and y coordinates for both points
    x1 = df[f'{point1}.x'].to_numpy()
    y1 = df[f'{point1}.y'].to_numpy()
    x2 = df[f'{point2}.x'].to_numpy()
    y2 = df[f'{point2}.y'].to_numpy()
    
    # Concatenate coordinates from both points
    x_all = np.hstack([x1, x2])
    y_all = np.hstack([y1, y2])
    
    # Filter out NaN values - keep only points where both x and y are valid
    valid_mask = ~(np.isnan(x_all) | np.isnan(y_all))
    x_valid = x_all[valid_mask]
    y_valid = y_all[valid_mask]
    
    # Check if we have enough valid points
    n_valid = len(x_valid)
    if n_valid < min_valid_points:
        total_points = len(x_all)
        n_invalid = total_points - n_valid
        raise ValueError(
            f"Insufficient valid data points for line fitting. "
            f"Found {n_valid} valid point pairs (required: {min_valid_points}), "
            f"excluded {n_invalid} invalid pairs (NaNs). "
            f"This may be due to too many blink periods or missing tracking data. "
            f"Consider reducing blink detection sensitivity or checking data quality."
        )
    
    # Fit line to valid data points
    line_fn = np.polyfit(x_valid, y_valid, 1)
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

def _get_all_points_pre_extracted(data_dict, point_name_list):
    """
    Pre-extract all points for all frames into a 3D array (n_frames, n_points, 2).
    This is much faster than extracting frame-by-frame in the loop.
    """
    n_frames = data_dict[point_name_list[0]].shape[0]
    n_points = len(point_name_list)
    
    # Pre-allocate array
    points_array = np.zeros((n_frames, n_points, 2))
    
    # Extract all points at once
    for i, point in enumerate(point_name_list):
        points_array[:, i, :] = data_dict[point]
    
    return points_array

def _fit_ellipse_single_frame(points, min_points=5):
    """
    Worker function to fit an ellipse to a single frame's points.
    Filters out NaN values before fitting.
    This function must be at module level to be pickled for multiprocessing.
    
    Args:
        points: numpy array of shape (n_points, 2) containing the coordinates
        min_points: minimum number of valid points required for fitting (default: 5)
        
    Returns:
        tuple: (params, center) where params is [width, height, phi] and center is [x, y]
    """
    # Filter out NaN values - keep only points where both x and y are valid
    valid_mask = ~(np.isnan(points[:, 0]) | np.isnan(points[:, 1]))
    valid_points = points[valid_mask]
    
    # Check if we have enough valid points (ellipse fitting needs at least 5 points)
    if len(valid_points) < min_points:
        return [np.nan, np.nan, np.nan], [np.nan, np.nan]
    
    try:
        reg = LsqEllipse()
        reg.fit(valid_points)
        center, width, height, phi = reg.as_parameters()
        return [width, height, phi], center
    except Exception as e:
        # Return NaN values if fitting fails
        return [np.nan, np.nan, np.nan], [np.nan, np.nan]

def _fit_ellipses_chunk(chunk_data, min_points=5):
    """
    Worker function to fit ellipses to a chunk of frames.
    Filters out NaN values before fitting.
    Reduces pickling overhead by processing multiple frames per worker.
    
    Args:
        chunk_data: numpy array of shape (n_chunk_frames, n_points, 2)
        min_points: minimum number of valid points required for fitting (default: 5)
        
    Returns:
        list of results, each result is (params, center)
    """
    results = []
    for t in range(chunk_data.shape[0]):
        points = chunk_data[t, :, :]
        
        # Filter out NaN values - keep only points where both x and y are valid
        valid_mask = ~(np.isnan(points[:, 0]) | np.isnan(points[:, 1]))
        valid_points = points[valid_mask]
        
        # Check if we have enough valid points (ellipse fitting needs at least 5 points)
        if len(valid_points) < min_points:
            results.append(([np.nan, np.nan, np.nan], [np.nan, np.nan]))
            continue
        
        try:
            reg = LsqEllipse()
            reg.fit(valid_points)
            center, width, height, phi = reg.as_parameters()
            results.append(([width, height, phi], center))
        except Exception as e:
            # Return NaN values if fitting fails
            results.append(([np.nan, np.nan, np.nan], [np.nan, np.nan]))
    return results


def get_fitted_ellipse_parameters(coordinates_dict, columns_of_interest, use_parallel=False, n_workers=None):

    # Collecting parameters of the fitted ellipse into an array over the whole recording
    # ellipse_parameters_data contents = (width, height, phi)
    # ellipse_center_points_data = (center_x, center_y)
    
    # Pre-extract all point data to avoid repeated dictionary lookups and list operations
    all_points = _get_all_points_pre_extracted(coordinates_dict, columns_of_interest)
    
    n_frames = all_points.shape[0]
    
    # Determine number of workers
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # Use parallel processing for larger datasets
    # Note: Parallel processing uses fork on Linux/macOS (works well here)
    # Falls back to sequential for small datasets or single worker
    if use_parallel and n_frames > 100 and n_workers > 1:
        print(f"ℹ️ Fitting ellipses to {n_frames} frames using {n_workers} parallel workers...")
        
        # Chunk frame indices to reduce communication overhead
        # Use smaller number of chunks to minimize per-chunk overhead
        n_chunks = n_workers  # 1 chunk per worker for minimal overhead
        chunk_size = max(1, n_frames // n_chunks)
        chunks = []
        for i in range(0, n_frames, chunk_size):
            end_idx = min(i + chunk_size, n_frames)
            # Send only the data needed for this chunk (views reduce memory usage)
            chunk_data = all_points[i:end_idx, :, :]
            chunks.append(chunk_data)
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            chunk_results = list(executor.map(_fit_ellipses_chunk, chunks))
        
        # Flatten results from chunks
        ellipse_parameters_data = []
        ellipse_center_points_data = []
        for chunk_result in chunk_results:
            for params, center in chunk_result:
                ellipse_parameters_data.append(params)
                ellipse_center_points_data.append(center)
        
        ellipse_parameters_data = np.array(ellipse_parameters_data)
        ellipse_center_points_data = np.array(ellipse_center_points_data)
    else:
        # Sequential processing (original algorithm)
        if use_parallel:
            print(f"ℹ️ Fitting ellipses to {n_frames} frames sequentially (n_workers={n_workers})...")
        
        ellipse_parameters_data = []
        ellipse_center_points_data = []
        
        # Create reg object once to avoid repeated instantiation
        reg = LsqEllipse()
        min_points = 5  # Minimum points required for ellipse fitting
        
        for t in range(n_frames):
            # Direct array indexing instead of function call
            points = all_points[t, :, :]
            
            # Filter out NaN values - keep only points where both x and y are valid
            valid_mask = ~(np.isnan(points[:, 0]) | np.isnan(points[:, 1]))
            valid_points = points[valid_mask]
            
            # Check if we have enough valid points
            if len(valid_points) < min_points:
                ellipse_parameters_data.append([np.nan, np.nan, np.nan])
                ellipse_center_points_data.append([np.nan, np.nan])
                continue
            
            try:
                reg.fit(valid_points)
                center, width, height, phi = reg.as_parameters()
                ellipse_parameters_data.append([width, height, phi])
                ellipse_center_points_data.append(center)
            except Exception as e:
                # Return NaN values if fitting fails
                ellipse_parameters_data.append([np.nan, np.nan, np.nan])
                ellipse_center_points_data.append([np.nan, np.nan])
        
        ellipse_parameters_data = np.array(ellipse_parameters_data)
        ellipse_center_points_data = np.array(ellipse_center_points_data)

    return ellipse_parameters_data, ellipse_center_points_data

def get_coordinates_dict(df, columns_of_interest):
    return {key:df[key].to_numpy() for key in columns_of_interest}

def get_left_right_center_point(coordinates_dict):
    """
    Calculate the center point between left and right eye reference points.
    Filters out NaN values before computing the mean.
    
    Parameters:
    -----------
    coordinates_dict : dict
        Dictionary containing coordinate arrays with keys 'left.x', 'left.y', 'right.x', 'right.y'
    
    Returns:
    --------
    tuple : (x, y) coordinates of the center point
    
    Raises:
    -------
    ValueError : If insufficient valid data points are available
    """
    x_all = np.hstack([coordinates_dict['left.x'], coordinates_dict['right.x']])
    y_all = np.hstack([coordinates_dict['left.y'], coordinates_dict['right.y']])
    
    # Filter out NaN values
    x_valid = x_all[~np.isnan(x_all)]
    y_valid = y_all[~np.isnan(y_all)]
    
    # Check if we have enough valid points
    min_valid_points = 20  # Require at least 20 valid points
    if len(x_valid) < min_valid_points or len(y_valid) < min_valid_points:
        raise ValueError(
            f"Insufficient valid data points for center point calculation. "
            f"Found {len(x_valid)} valid x-coordinates and {len(y_valid)} valid y-coordinates "
            f"(required: {min_valid_points} each). "
            f"This may be due to too many blink periods or missing tracking data."
        )
    
    x = np.mean(x_valid)
    y = np.mean(y_valid)
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