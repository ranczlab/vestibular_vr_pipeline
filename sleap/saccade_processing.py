"""
Saccade detection and processing functions for eye tracking data analysis.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.cm as cm


def detect_saccades_adaptive(df, position_col='X_smooth', velocity_col='vel_x_smooth', 
                             time_col='Seconds', fps=None, k=5, refractory_period=0.1,
                             onset_offset_fraction=0.2, peak_width=1, verbose=True, upward_label=None, downward_label=None):
    """
    Detect saccades using adaptive statistical threshold method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing position and velocity data
    position_col : str
        Column name for smoothed position data (default: 'X_smooth')
    velocity_col : str
        Column name for velocity data (default: 'vel_x_smooth')
    time_col : str
        Column name for time data (default: 'Seconds')
    fps : float, optional
        Frames per second. If None, will be estimated from time_col
    k : float
        Number of standard deviations for adaptive threshold (default: 5)
    refractory_period : float
        Minimum time (in seconds) between saccades (default: 0.1)
    onset_offset_fraction : float
        Fraction of peak velocity threshold for onset/offset detection (default: 0.2)
    peak_width : int
        Minimum peak width in samples for find_peaks (default: 1)
    verbose : bool
        Whether to print detection statistics (default: True)
        
    Returns:
    --------
    tuple : (upward_saccades_df, downward_saccades_df, velocity_threshold)
    """
    # Calculate FPS if not provided
    if fps is None:
        fps = 1 / df[time_col].diff().mean()
    
    # 1. Calculate adaptive thresholds using statistical methods
    abs_vel = df[velocity_col].abs().dropna()
    vel_thresh = abs_vel.mean() + k * abs_vel.std()
    
    if verbose:
        print(f"Adaptive velocity threshold: {vel_thresh:.2f} px/s")
        print(f"  (mean: {abs_vel.mean():.2f} px/s, std: {abs_vel.std():.2f} px/s, k={k})")
        print(f"Saccade duration parameter: {onset_offset_fraction} (saccade ends when velocity < {vel_thresh * onset_offset_fraction:.2f} px/s)")
    
    # 2. Find peaks using scipy's find_peaks with adaptive height
    # Find positive peaks (upward saccades)
    pos_peaks, pos_properties = find_peaks(
        df[velocity_col],
        height=vel_thresh,  # Minimum peak height
        distance=int(fps * refractory_period),  # Minimum distance between peaks (~100ms refractory period)
        width=peak_width  # Minimum peak width in samples
    )
    
    # Find negative peaks (downward saccades) by inverting the signal
    neg_peaks, neg_properties = find_peaks(
        -df[velocity_col],  # Invert to find troughs
        height=vel_thresh,  # Same threshold
        distance=int(fps * refractory_period),
        width=peak_width
    )
    
    # 3. Extract saccade information
    upward_saccades = []
    for peak_idx in pos_peaks:
        peak_time = df.iloc[peak_idx][time_col]
        peak_velocity = df.iloc[peak_idx][velocity_col]
        
        # Find onset and offset by going backward/forward until velocity drops below threshold
        start_idx = peak_idx
        end_idx = peak_idx
        
        # Find onset (go backward) - use threshold_fraction for sensitivity to catch early movement
        while start_idx > 0 and abs(df.iloc[start_idx][velocity_col]) > vel_thresh * onset_offset_fraction:
            start_idx -= 1
        
        # Find offset (go forward) - use FULL threshold since saccade is only above full threshold briefly
        # This is more accurate for low sampling rate where velocity is only above threshold for 1 frame
        while end_idx < len(df) - 1 and abs(df.iloc[end_idx][velocity_col]) > vel_thresh:
            end_idx += 1
        # After loop: end_idx points to last frame with velocity > full threshold
        # Increment to get frame AFTER velocity dropped below full threshold
        if end_idx < len(df) - 1:
            end_idx += 1
        
        onset_time = df.iloc[start_idx][time_col]
        offset_time = df.iloc[end_idx][time_col]
        
        # Calculate amplitude (absolute change in position)
        start_x = df.iloc[start_idx][position_col]
        end_x = df.iloc[end_idx][position_col]
        amplitude = abs(end_x - start_x)
        
        upward_saccades.append({
            'time': peak_time,
            'velocity': peak_velocity,
            'start_time': onset_time,
            'end_time': offset_time,
            'duration': offset_time - onset_time,
            'start_position': start_x,
            'end_position': end_x,
            'amplitude': amplitude
        })
    
    downward_saccades = []
    for peak_idx in neg_peaks:
        peak_time = df.iloc[peak_idx][time_col]
        peak_velocity = df.iloc[peak_idx][velocity_col]
        
        # Find onset and offset
        start_idx = peak_idx
        end_idx = peak_idx
        
        # Find onset (go backward) - use threshold_fraction for sensitivity to catch early movement
        while start_idx > 0 and abs(df.iloc[start_idx][velocity_col]) > vel_thresh * onset_offset_fraction:
            start_idx -= 1
        
        # Find offset (go forward) - use FULL threshold since saccade is only above full threshold briefly
        # This is more accurate for low sampling rate where velocity is only above threshold for 1 frame
        while end_idx < len(df) - 1 and abs(df.iloc[end_idx][velocity_col]) > vel_thresh:
            end_idx += 1
        # After loop: end_idx points to last frame with velocity > full threshold
        # Increment to get frame AFTER velocity dropped below full threshold
        if end_idx < len(df) - 1:
            end_idx += 1
        
        onset_time = df.iloc[start_idx][time_col]
        offset_time = df.iloc[end_idx][time_col]
        
        # Calculate amplitude (absolute change in position)
        start_x = df.iloc[start_idx][position_col]
        end_x = df.iloc[end_idx][position_col]
        amplitude = abs(end_x - start_x)
        
        downward_saccades.append({
            'time': peak_time,
            'velocity': peak_velocity,
            'start_time': onset_time,
            'end_time': offset_time,
            'duration': offset_time - onset_time,
            'start_position': start_x,
            'end_position': end_x,
            'amplitude': amplitude
        })
    
    # Convert to DataFrames for easier handling
    upward_saccades_df = pd.DataFrame(upward_saccades)
    downward_saccades_df = pd.DataFrame(downward_saccades)
    
    if verbose:
        # Format direction labels for output
        upward_str = f"upward ({upward_label})" if upward_label else "upward"
        downward_str = f"downward ({downward_label})" if downward_label else "downward"
        
        print(f"\n✅ Detected {len(pos_peaks)} {upward_str} saccades")
        print(f"✅ Detected {len(neg_peaks)} {downward_str} saccades")
        
        # Print summary statistics (compact format)
        if len(upward_saccades) > 0:
            print(f"Upward saccades - mean velocity: {upward_saccades_df['velocity'].mean():.2f} px/s, mean duration: {upward_saccades_df['duration'].mean():.3f} s, mean amplitude: {upward_saccades_df['amplitude'].mean():.2f} px, std amplitude: {upward_saccades_df['amplitude'].std():.2f} px")
        
        if len(downward_saccades) > 0:
            print(f"Downward saccades - mean velocity: {downward_saccades_df['velocity'].mean():.2f} px/s, mean duration: {downward_saccades_df['duration'].mean():.3f} s, mean amplitude: {downward_saccades_df['amplitude'].mean():.2f} px, std amplitude: {downward_saccades_df['amplitude'].std():.2f} px")
    
    return upward_saccades_df, downward_saccades_df, vel_thresh


def extract_saccade_segment(df, sacc_df, n_before, n_after, direction='upward',
                           position_col='X_smooth', time_col='Seconds'):
    """
    Extract peri-saccade segment for saccades.
    Uses start_time (threshold crossing) as center point: n_before points before, n_after points after.
    Excludes segments with unreasonable time ranges (validation).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the full time series data
    sacc_df : pd.DataFrame
        DataFrame containing saccade information with columns: 'start_time', 'end_time', 'time', 'amplitude'
    n_before : int
        Number of points before threshold crossing to extract
    n_after : int
        Number of points after threshold crossing to extract
    direction : str
        Direction label for saccades ('upward' or 'downward', default: 'upward')
    position_col : str
        Column name for position data (default: 'X_smooth')
    time_col : str
        Column name for time data (default: 'Seconds')
        
    Returns:
    --------
    tuple : (segments, excluded_segments)
        segments: list of DataFrames, one per valid saccade
        excluded_segments: list of dicts with exclusion metadata
    """
    segments = []
    excluded_segments = []  # Track excluded segments for reporting
    
    # Use a global dt estimate for validation (robust to bad slices)
    if len(df) > 1:
        dt_global = df[time_col].diff().median()
        if pd.isna(dt_global) or dt_global <= 0:
            dt_global = df[time_col].diff().mean()
    else:
        dt_global = 0.0167

    for idx, sacc in sacc_df.iterrows():
        start_time = sacc['start_time']
        end_time = sacc['end_time']
        peak_time = sacc['time']
        amplitude = sacc['amplitude']
        
        # Find threshold crossing position (start_time)
        # Work in positional indices to avoid label/position mismatches
        start_pos = int(np.argmin(np.abs(df[time_col].to_numpy() - start_time)))
        
        # Calculate pre/post positions centered on threshold crossing
        pre_start = max(0, start_pos - n_before)
        post_end = min(len(df) - 1, start_pos + n_after)
        
        # Extract segment
        segment = df.iloc[pre_start:post_end + 1].copy()
        
        # Normalize time relative to threshold crossing (start_time)
        segment['Time_rel_threshold'] = segment[time_col] - start_time
        
        # Validate time range
        time_range_min = segment['Time_rel_threshold'].min()
        time_range_max = segment['Time_rel_threshold'].max()
        
        # Calculate expected time span based on global dt and desired window (110% tolerance)
        time_span = time_range_max - time_range_min
        expected_total_points = n_before + 1 + n_after
        expected_time_span = (expected_total_points - 1) * float(dt_global)
        max_allowed_time_span = expected_time_span * 1.1  # 110% of expected
        
        # Check if time range is reasonable (110% of expected span) or if point count is wrong
        # Allow some flexibility for edge cases where we can't extract full window
        is_valid = True
        if time_span > max_allowed_time_span:
            is_valid = False  # Time span too large (likely data gaps)
        elif len(segment) < expected_total_points * 0.9:
            is_valid = False  # Too few points (likely truncated at edges or data gaps)
        
        if not is_valid:
            excluded_segments.append({
                'saccade_id': idx,
                'direction': direction,
                'time_range': (time_range_min, time_range_max),
                'n_points': len(segment),
                'threshold_time': start_time,
                'peak_time': peak_time,
                'amplitude': amplitude
            })
            continue  # Skip this segment
        
        segment['Time_rel_saccade_start'] = segment[time_col] - start_time
        segment['Time_rel_saccade_end'] = segment[time_col] - end_time
        
        # Add metadata
        segment['saccade_id'] = idx
        segment['saccade_direction'] = direction
        segment['saccade_amplitude'] = amplitude
        segment['saccade_peak_velocity'] = sacc['velocity']
        segment['saccade_duration'] = sacc['duration']
        segment['is_saccade_period'] = (segment[time_col] >= start_time) & (segment[time_col] <= end_time)
        
        segments.append(segment)
    
    return segments, excluded_segments


def baseline_saccade_segments(segments, n_before, baseline_n_points=5, 
                              position_col='X_smooth', time_rel_col='Time_rel_threshold'):
    """
    Baseline each saccade using the average value of data points in the baseline window.
    This removes pre-saccade position offsets and centers all saccades at baseline = 0.
    
    Parameters:
    -----------
    segments : list of pd.DataFrame
        List of saccade segment DataFrames
    n_before : int
        Number of points before threshold crossing (used to identify threshold position)
    baseline_n_points : int
        Number of points before threshold crossing to use for baseline (default: 5)
    position_col : str
        Column name for position data to baseline (default: 'X_smooth')
    time_rel_col : str
        Column name for time relative to threshold (default: 'Time_rel_threshold')
        
    Returns:
    --------
    list : List of baselined segment DataFrames
    """
    baseline_start = -n_before  # Relative to threshold crossing (start of extracted window)
    baseline_end = -(n_before - baseline_n_points)  # Relative to threshold crossing
    
    for segment in segments:
        # Find threshold crossing index in segment (where Time_rel_threshold is closest to 0)
        threshold_idx_in_seg = segment[time_rel_col].abs().idxmin()
        threshold_pos_in_seg = segment.index.get_loc(threshold_idx_in_seg)
        
        # Baseline window: last baseline_n_points points before threshold crossing
        baseline_start_idx = max(0, threshold_pos_in_seg - baseline_n_points)
        baseline_end_idx = threshold_pos_in_seg  # Up to but not including threshold crossing
        
        if baseline_end_idx > baseline_start_idx:
            # Extract baseline window
            baseline_indices = range(baseline_start_idx, baseline_end_idx)
            baseline_window_size = len(baseline_indices)
            
            # Calculate baseline as mean position in this window
            baseline_value = segment.iloc[baseline_indices][position_col].mean()
            
            # Subtract baseline from all position values in the segment
            segment['X_smooth_baselined'] = segment[position_col] - baseline_value
            segment['baseline_value'] = baseline_value
        else:
            # If segment is too short, use available early points
            baseline_window_size = baseline_n_points
            if len(segment) > 0:
                baseline_value = segment.iloc[:min(baseline_window_size, len(segment))][position_col].mean()
                segment['X_smooth_baselined'] = segment[position_col] - baseline_value
                segment['baseline_value'] = baseline_value
            else:
                # If no points found, use zero baseline
                segment['X_smooth_baselined'] = segment[position_col]
                segment['baseline_value'] = 0.0
    
    return segments


def filter_outlier_saccades(segments, direction_name):
    """
    Filter out outlier segments using IQR method on amplitude and extreme values.
    
    Parameters:
    -----------
    segments : list of pd.DataFrame
        List of saccade segment DataFrames
    direction_name : str
        Direction label (e.g., 'upward', 'downward')
        
    Returns:
    --------
    tuple : (filtered_segments, outliers_metadata, outlier_segments)
    """
    if len(segments) == 0:
        return [], [], []
    
    # Calculate statistics on amplitudes
    amplitudes = [seg['saccade_amplitude'].iloc[0] for seg in segments]
    q1_amp = np.percentile(amplitudes, 25)
    q3_amp = np.percentile(amplitudes, 75)
    iqr_amp = q3_amp - q1_amp
    lower_bound_amp = q1_amp - 3 * iqr_amp  # Using 3*IQR for more lenient filtering
    upper_bound_amp = q3_amp + 3 * iqr_amp
    
    # Calculate statistics on max absolute position values
    max_pos_values = []
    max_vel_values = []
    for seg in segments:
        max_pos_values.append(np.abs(seg['X_smooth_baselined']).max())
        max_vel_values.append(np.abs(seg['vel_x_smooth']).max())
    
    q1_pos = np.percentile(max_pos_values, 25)
    q3_pos = np.percentile(max_pos_values, 75)
    iqr_pos = q3_pos - q1_pos
    upper_bound_pos = q3_pos + 3 * iqr_pos
    
    q1_vel = np.percentile(max_vel_values, 25)
    q3_vel = np.percentile(max_vel_values, 75)
    iqr_vel = q3_vel - q1_vel
    upper_bound_vel = q3_vel + 3 * iqr_vel
    
    filtered = []
    outliers_metadata = []
    outlier_segments = []
    
    for seg in segments:
        amp = seg['saccade_amplitude'].iloc[0]
        max_pos = np.abs(seg['X_smooth_baselined']).max()
        max_vel = np.abs(seg['vel_x_smooth']).max()
        seg_id = seg['saccade_id'].iloc[0]
        
        # Check if outlier
        is_outlier = (amp < lower_bound_amp or amp > upper_bound_amp or
                     max_pos > upper_bound_pos or
                     max_vel > upper_bound_vel)
        
        if is_outlier:
            outliers_metadata.append({
                'saccade_id': seg_id,
                'amplitude': amp,
                'max_abs_position': max_pos,
                'max_abs_velocity': max_vel,
                'direction': direction_name
            })
            outlier_segments.append(seg)  # Keep the segment for plotting
        else:
            filtered.append(seg)
    
    return filtered, outliers_metadata, outlier_segments


def get_color_mapping(amplitudes):
    """
    Create color mapping from min to max amplitude using plasma colormap.
    
    Parameters:
    -----------
    amplitudes : array-like
        Array of amplitude values
        
    Returns:
    --------
    tuple : (colors_list, min_amp, max_amp)
        colors_list: list of RGB color strings
        min_amp: minimum amplitude value
        max_amp: maximum amplitude value
    """
    amps = np.array(amplitudes)
    min_amp = np.min(amps)
    max_amp = np.max(amps)
    
    # Normalize amplitudes to 0-1 range for colormap
    if max_amp > min_amp:
        normalized_amps = (amps - min_amp) / (max_amp - min_amp)
    else:
        normalized_amps = np.zeros_like(amps)
    
    # Use plasma colormap: lower amplitudes get dark purple/blue, higher amplitudes get bright yellow/magenta
    colors = cm.plasma(normalized_amps)
    return ['rgb({}, {}, {})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors], min_amp, max_amp


def classify_saccades_orienting_vs_compensatory(
    all_saccades_df,
    df,
    fps,
    bout_window=1.5,
    pre_saccade_window=0.3,
    max_intersaccade_interval_for_classification=5.0,
    pre_saccade_velocity_threshold=50.0,
    pre_saccade_drift_threshold=10.0,
    post_saccade_variance_threshold=100.0,
    post_saccade_position_change_threshold_percent=50.0,
    use_adaptive_thresholds=False,
    adaptive_percentile_pre_velocity=75,
    adaptive_percentile_pre_drift=75,
    adaptive_percentile_post_variance=25,
    position_col='X_smooth',
    velocity_col='vel_x_smooth',
    time_col='Seconds',
    verbose=True
):
    """
    Classify saccades as orienting vs compensatory based on temporal clustering and signal features.
    
    Parameters:
    -----------
    all_saccades_df : pd.DataFrame
        Combined DataFrame of all saccades (upward + downward) with columns:
        'time', 'start_time', 'end_time', 'duration', 'amplitude', 'velocity', 'direction'
    df : pd.DataFrame
        Full time series DataFrame with position and velocity data
    fps : float
        Frames per second
    bout_window : float
        Time window (seconds) for grouping saccades into bouts (default: 1.5)
    pre_saccade_window : float
        Time window (seconds) before saccade onset to analyze (default: 0.3)
    max_intersaccade_interval_for_classification : float
        Maximum time (seconds) to extend post-saccade window until next saccade (default: 5.0)
        Post-saccade analysis dynamically extends until next saccade or this maximum
    pre_saccade_velocity_threshold : float
        Velocity threshold (px/s) for detecting pre-saccade drift (default: 50.0)
    pre_saccade_drift_threshold : float
        Position drift threshold (px) before saccade for compensatory classification (default: 10.0)
    post_saccade_variance_threshold : float
        Position variance threshold (px²) after saccade for orienting classification (default: 100.0)
    post_saccade_position_change_threshold_percent : float
        Position change threshold (% of saccade amplitude) for compensatory classification (default: 50.0)
        If post_saccade_position_change > amplitude * threshold_percent, classify as compensatory
    use_adaptive_thresholds : bool
        If True, calculate thresholds from feature distributions using percentiles (default: False)
        If False, use fixed thresholds provided as parameters
    adaptive_percentile_pre_velocity : int
        Percentile for adaptive pre-saccade velocity threshold (default: 75)
        Used as upper percentile for compensatory detection
    adaptive_percentile_pre_drift : int
        Percentile for adaptive pre-saccade drift threshold (default: 75)
        Used as upper percentile for compensatory detection
    adaptive_percentile_post_variance : int
        Percentile for adaptive post-saccade variance threshold (default: 25)
        Used as lower percentile for orienting detection (low variance = stable)
    position_col : str
        Column name for position data (default: 'X_smooth')
    velocity_col : str
        Column name for velocity data (default: 'vel_x_smooth')
    time_col : str
        Column name for time data (default: 'Seconds')
    verbose : bool
        Whether to print classification statistics (default: True)
        
    Returns:
    --------
    pd.DataFrame : Same DataFrame with added columns:
        'saccade_type': 'orienting' or 'compensatory' (ALL saccades are classified)
        'bout_id': ID of bout (None for isolated saccades)
        'bout_size': Number of saccades in bout (1 for isolated)
        'pre_saccade_mean_velocity': Mean velocity before saccade
        'pre_saccade_position_drift': Position change before saccade
        'post_saccade_position_variance': Position variance in dynamic window (until next saccade or max interval)
        'post_saccade_position_change': Total position change in dynamic window (until next saccade or max interval)
    """
    if len(all_saccades_df) == 0:
        return all_saccades_df
    
    # Sort by time to ensure chronological order
    all_saccades_df = all_saccades_df.sort_values('time').reset_index(drop=True)
    
    # Initialize new columns
    all_saccades_df['saccade_type'] = None
    all_saccades_df['bout_id'] = None
    all_saccades_df['bout_size'] = 1
    all_saccades_df['pre_saccade_mean_velocity'] = np.nan
    all_saccades_df['pre_saccade_position_drift'] = np.nan
    all_saccades_df['post_saccade_position_variance'] = np.nan
    all_saccades_df['post_saccade_position_change'] = np.nan
    
    # Stage 1: Temporal clustering (bout detection)
    bout_id = 0
    current_bout_start_idx = None
    
    for i in range(len(all_saccades_df)):
        if i == 0:
            # First saccade starts a potential bout
            current_bout_start_idx = 0
            bout_id += 1
            all_saccades_df.loc[i, 'bout_id'] = bout_id
        else:
            # Check time interval from previous saccade
            time_interval = all_saccades_df.iloc[i]['time'] - all_saccades_df.iloc[i-1]['time']
            
            if time_interval <= bout_window:
                # Within bout window - add to current bout
                all_saccades_df.loc[i, 'bout_id'] = bout_id
            else:
                # Gap > bout_window - start new bout
                bout_id += 1
                all_saccades_df.loc[i, 'bout_id'] = bout_id
                current_bout_start_idx = i
    
    # Calculate bout sizes
    bout_sizes = all_saccades_df.groupby('bout_id').size()
    all_saccades_df['bout_size'] = all_saccades_df['bout_id'].map(bout_sizes)
    
    # Stage 2: Feature extraction for each saccade
    for idx, saccade in all_saccades_df.iterrows():
        start_time = saccade['start_time']
        end_time = saccade['end_time']
        peak_time = saccade['time']
        
        # Find indices in full dataframe
        start_idx = int(np.argmin(np.abs(df[time_col].to_numpy() - start_time)))
        end_idx = int(np.argmin(np.abs(df[time_col].to_numpy() - end_time)))
        
        # Feature 1: Pre-saccade velocity
        # Calculate desired window start time
        desired_pre_window_start = start_time - pre_saccade_window
        
        # Constrain to previous saccade's peak time (if previous saccade exists)
        if idx > 0:
            previous_saccade_peak_time = all_saccades_df.iloc[idx - 1]['time']
            # Use the maximum (most recent) of: desired start time or previous saccade peak
            pre_window_start_time = max(desired_pre_window_start, previous_saccade_peak_time)
        else:
            # First saccade - no constraint, use desired window
            pre_window_start_time = desired_pre_window_start
        
        pre_window_start_idx = max(0, int(np.argmin(np.abs(df[time_col].to_numpy() - pre_window_start_time))))
        pre_saccade_mask = (df[time_col] >= pre_window_start_time) & (df[time_col] < start_time)
        pre_saccade_velocities = df.loc[pre_saccade_mask, velocity_col]
        pre_saccade_mean_vel = pre_saccade_velocities.abs().mean() if len(pre_saccade_velocities) > 0 else 0.0
        
        # Feature 2: Pre-saccade position drift
        pre_saccade_positions = df.loc[pre_saccade_mask, position_col]
        if len(pre_saccade_positions) > 0:
            pre_saccade_drift = abs(pre_saccade_positions.iloc[-1] - pre_saccade_positions.iloc[0])
        else:
            pre_saccade_drift = 0.0
        
        # Feature 3: Post-saccade position stability (DYNAMIC WINDOW)
        # Window extends from saccade offset until next saccade start (or max interval)
        # Find next saccade
        next_saccade_idx = idx + 1
        if next_saccade_idx < len(all_saccades_df):
            next_saccade_start_time = all_saccades_df.iloc[next_saccade_idx]['start_time']
            interval_to_next = next_saccade_start_time - end_time
            # Use interval to next saccade if within max, otherwise use max
            dynamic_window_duration = min(interval_to_next, max_intersaccade_interval_for_classification)
        else:
            # Last saccade - use max interval
            dynamic_window_duration = max_intersaccade_interval_for_classification
        
        post_window_end_time = end_time + dynamic_window_duration
        post_window_end_idx = min(len(df) - 1, int(np.argmin(np.abs(df[time_col].to_numpy() - post_window_end_time))))
        post_saccade_mask = (df[time_col] > end_time) & (df[time_col] <= post_window_end_time)
        post_saccade_positions = df.loc[post_saccade_mask, position_col]
        
        if len(post_saccade_positions) > 1:
            post_saccade_variance = post_saccade_positions.var()
            # Calculate total position change (from end of saccade to end of window)
            post_saccade_change = abs(post_saccade_positions.iloc[-1] - post_saccade_positions.iloc[0])
        else:
            post_saccade_variance = 0.0
            post_saccade_change = 0.0
        
        # Store features
        all_saccades_df.loc[idx, 'pre_saccade_mean_velocity'] = pre_saccade_mean_vel
        all_saccades_df.loc[idx, 'pre_saccade_position_drift'] = pre_saccade_drift
        all_saccades_df.loc[idx, 'post_saccade_position_variance'] = post_saccade_variance
        all_saccades_df.loc[idx, 'post_saccade_position_change'] = post_saccade_change
    
    # Calculate adaptive thresholds if enabled (after feature extraction, before classification)
    if use_adaptive_thresholds:
        # Calculate thresholds from feature distributions using percentiles
        # Filter out NaN values for percentile calculation
        pre_velocities = all_saccades_df['pre_saccade_mean_velocity'].dropna()
        pre_drifts = all_saccades_df['pre_saccade_position_drift'].dropna()
        post_variances = all_saccades_df['post_saccade_position_variance'].dropna()
        
        # Check if we have enough data points (at least 3 for meaningful percentiles)
        min_samples = 3
        
        if len(pre_velocities) >= min_samples:
            pre_saccade_velocity_threshold = np.percentile(pre_velocities, adaptive_percentile_pre_velocity)
        else:
            if verbose:
                print(f"⚠️ Warning: Only {len(pre_velocities)} pre-saccade velocity samples, using fixed threshold")
            # Keep original fixed threshold
            
        if len(pre_drifts) >= min_samples:
            pre_saccade_drift_threshold = np.percentile(pre_drifts, adaptive_percentile_pre_drift)
        else:
            if verbose:
                print(f"⚠️ Warning: Only {len(pre_drifts)} pre-saccade drift samples, using fixed threshold")
            # Keep original fixed threshold
            
        if len(post_variances) >= min_samples:
            post_saccade_variance_threshold = np.percentile(post_variances, adaptive_percentile_post_variance)
        else:
            if verbose:
                print(f"⚠️ Warning: Only {len(post_variances)} post-saccade variance samples, using fixed threshold")
            # Keep original fixed threshold
        
        # Report thresholds used
        if verbose:
            print(f"\n📊 Adaptive Thresholds Calculated (from feature distributions):")
            print(f"  Pre-saccade velocity threshold ({adaptive_percentile_pre_velocity}th percentile): {pre_saccade_velocity_threshold:.2f} px/s")
            print(f"  Pre-saccade drift threshold ({adaptive_percentile_pre_drift}th percentile): {pre_saccade_drift_threshold:.2f} px")
            print(f"  Post-saccade variance threshold ({adaptive_percentile_post_variance}th percentile): {post_saccade_variance_threshold:.2f} px²")
    else:
        if verbose:
            print(f"\n📊 Using Fixed Thresholds:")
            print(f"  Pre-saccade velocity threshold: {pre_saccade_velocity_threshold:.2f} px/s")
            print(f"  Pre-saccade drift threshold: {pre_saccade_drift_threshold:.2f} px")
            print(f"  Post-saccade variance threshold: {post_saccade_variance_threshold:.2f} px²")
    
    # Stage 3: Classification logic
    # Original simple logic (conservative approach):
    # - If in a bout (bout_size >= 2) → automatically compensatory
    # - If isolated → use feature-based classification
    # This is the starting point - can be refined later if needed
    
    for idx, saccade in all_saccades_df.iterrows():
        bout_size = saccade['bout_size']
        pre_vel = saccade['pre_saccade_mean_velocity']
        pre_drift = saccade['pre_saccade_position_drift']
        post_var = saccade['post_saccade_position_variance']
        post_change = saccade['post_saccade_position_change']
        amplitude = saccade['amplitude']
        
        # Classification rules:
        # 1. If in a bout (bout_size >= 2), classify as compensatory
        # 2. If isolated, use features:
        #    - Pre-saccade velocity > threshold OR pre-saccade drift significant → compensatory
        #    - Post-saccade position change > amplitude * threshold_percent → compensatory (eye continues moving)
        #    - Pre-saccade velocity low AND post-saccade stable (variance AND position change) → orienting
        
        if bout_size >= 2:
            # Multiple saccades in bout → compensatory
            all_saccades_df.loc[idx, 'saccade_type'] = 'compensatory'
        else:
            # Isolated saccade - use feature-based classification
            # Calculate position change threshold relative to amplitude
            position_change_threshold = amplitude * (post_saccade_position_change_threshold_percent / 100.0)
            
            # Check for compensatory indicators
            has_pre_saccade_drift = (pre_vel > pre_saccade_velocity_threshold or pre_drift > pre_saccade_drift_threshold)
            has_post_saccade_movement = (post_change > position_change_threshold)
            
            if has_pre_saccade_drift or has_post_saccade_movement:
                # Evidence of drift/compensation → compensatory
                all_saccades_df.loc[idx, 'saccade_type'] = 'compensatory'
            elif (pre_vel <= pre_saccade_velocity_threshold and 
                  post_var < post_saccade_variance_threshold and
                  post_change <= position_change_threshold):
                # Stable before and after → orienting
                all_saccades_df.loc[idx, 'saccade_type'] = 'orienting'
            else:
                # Default: if uncertain, classify as compensatory (conservative)
                all_saccades_df.loc[idx, 'saccade_type'] = 'compensatory'
    
    # Print statistics
    if verbose:
        n_orienting = (all_saccades_df['saccade_type'] == 'orienting').sum()
        n_compensatory = (all_saccades_df['saccade_type'] == 'compensatory').sum()
        n_bouts = all_saccades_df['bout_id'].nunique()
        n_isolated = (all_saccades_df['bout_size'] == 1).sum()
        
        print(f"\n📊 Saccade Classification Results:")
        print(f"  Total saccades: {len(all_saccades_df)}")
        print(f"  Orienting saccades: {n_orienting} ({n_orienting/len(all_saccades_df)*100:.1f}%)")
        print(f"  Compensatory saccades: {n_compensatory} ({n_compensatory/len(all_saccades_df)*100:.1f}%)")
        print(f"  Detected bouts: {n_bouts}")
        print(f"  Isolated saccades: {n_isolated}")
        
        if n_orienting > 0:
            orienting_df = all_saccades_df[all_saccades_df['saccade_type'] == 'orienting']
            print(f"\n  Orienting saccades stats:")
            print(f"    Mean amplitude: {orienting_df['amplitude'].mean():.2f} px")
            print(f"    Mean duration: {orienting_df['duration'].mean():.3f} s")
            print(f"    Mean pre-saccade velocity: {orienting_df['pre_saccade_mean_velocity'].mean():.2f} px/s")
        
        if n_compensatory > 0:
            compensatory_df = all_saccades_df[all_saccades_df['saccade_type'] == 'compensatory']
            print(f"\n  Compensatory saccades stats:")
            print(f"    Mean amplitude: {compensatory_df['amplitude'].mean():.2f} px")
            print(f"    Mean duration: {compensatory_df['duration'].mean():.3f} s")
            print(f"    Mean pre-saccade velocity: {compensatory_df['pre_saccade_mean_velocity'].mean():.2f} px/s")
            print(f"    Mean bout size: {compensatory_df['bout_size'].mean():.2f} saccades")
    
    return all_saccades_df


def analyze_eye_video_saccades(
    df,
    fps,
    video_label,
    k=5,
    refractory_period=0.1,
    onset_offset_fraction=0.2,
    n_before=10,
    n_after=30,
    baseline_n_points=5,
    saccade_smoothing_window=5,
    saccade_peak_width=1,
    upward_label=None,
    downward_label=None,
    classify_orienting_compensatory=True,
    bout_window=1.5,
    pre_saccade_window=0.3,
    max_intersaccade_interval_for_classification=5.0,
    pre_saccade_velocity_threshold=50.0,
    pre_saccade_drift_threshold=10.0,
    post_saccade_variance_threshold=100.0,
    post_saccade_position_change_threshold_percent=50.0,
    use_adaptive_thresholds=False,
    adaptive_percentile_pre_velocity=75,
    adaptive_percentile_pre_drift=75,
    adaptive_percentile_post_variance=25,
):
    """
    Analyze saccades for a given eye video dataframe.
    Performs smoothing, velocity computation, adaptive saccade detection, peri-event extraction, baselining, outlier filtering.
    Optionally classifies saccades as orienting vs compensatory.
    Returns dict of results including all relevant outputs for stats/plots.
    """
    import numpy as np
    import pandas as pd
    # sp is this module (sleap.saccade_processing): use local functions

    # Smoothing
    df = df.copy()
    # Ensure a clean positional index to avoid label/position confusion downstream
    df = df.reset_index(drop=True)
    df['X_smooth'] = (
        df['Ellipse.Center.X']
        .rolling(window=saccade_smoothing_window, center=True)
        .median()
        .bfill()
        .ffill()
    )

    # Compute instantaneous velocity
    df['dt'] = df['Seconds'].diff()
    df['vel_x_original'] = df['Ellipse.Center.X'].diff() / df['dt']
    df['vel_x_smooth'] = df['X_smooth'].diff() / df['dt']

    # Saccade detection
    upward_saccades_df, downward_saccades_df, vel_thresh = detect_saccades_adaptive(
        df,
        position_col='X_smooth',
        velocity_col='vel_x_smooth',
        time_col='Seconds',
        fps=fps,
        k=k,
        refractory_period=refractory_period,
        onset_offset_fraction=onset_offset_fraction,
        peak_width=saccade_peak_width,
        verbose=True,
        upward_label=upward_label,
        downward_label=downward_label
    )

    # Peri-saccade segment extraction
    peri_saccades = []
    all_excluded = []
    if len(upward_saccades_df) > 0:
        upward_segments, upward_excluded = extract_saccade_segment(df, upward_saccades_df, n_before, n_after, direction='upward')
        peri_saccades.extend(upward_segments)
        all_excluded.extend(upward_excluded)
    if len(downward_saccades_df) > 0:
        downward_segments, downward_excluded = extract_saccade_segment(df, downward_saccades_df, n_before, n_after, direction='downward')
        peri_saccades.extend(downward_segments)
        all_excluded.extend(downward_excluded)

    # Baselining
    peri_saccades = baseline_saccade_segments(peri_saccades, n_before, baseline_n_points=baseline_n_points)

    # Filter outliers
    upward_segments_all = [seg for seg in peri_saccades if seg['saccade_direction'].iloc[0] == 'upward']
    downward_segments_all = [seg for seg in peri_saccades if seg['saccade_direction'].iloc[0] == 'downward']
    upward_segments, upward_outliers_meta, upward_outlier_segments = filter_outlier_saccades(upward_segments_all, 'upward')
    downward_segments, downward_outliers_meta, downward_outlier_segments = filter_outlier_saccades(downward_segments_all, 'downward')

    # Classify saccades as orienting vs compensatory (if enabled)
    # Combine upward and downward saccades for classification
    # Preserve original indices for matching
    all_saccades_list = []
    if len(upward_saccades_df) > 0:
        upward_saccades_df_copy = upward_saccades_df.copy()
        upward_saccades_df_copy['direction'] = 'upward'
        upward_saccades_df_copy['original_index'] = upward_saccades_df_copy.index
        upward_saccades_df_copy['is_upward'] = True
        all_saccades_list.append(upward_saccades_df_copy)
    if len(downward_saccades_df) > 0:
        downward_saccades_df_copy = downward_saccades_df.copy()
        downward_saccades_df_copy['direction'] = 'downward'
        downward_saccades_df_copy['original_index'] = downward_saccades_df_copy.index
        downward_saccades_df_copy['is_upward'] = False
        all_saccades_list.append(downward_saccades_df_copy)
    
    if len(all_saccades_list) > 0:
        all_saccades_df = pd.concat(all_saccades_list, ignore_index=True)
        # Store original index mapping before classification (which sorts by time)
        original_idx_map = {}
        for new_idx, row in all_saccades_df.iterrows():
            original_idx_map[new_idx] = {
                'original_index': row['original_index'],
                'is_upward': row['is_upward']
            }
        
        # Classify saccades (if enabled)
        if classify_orienting_compensatory:
            all_saccades_df = classify_saccades_orienting_vs_compensatory(
                all_saccades_df, df, fps,
                bout_window=bout_window,
                pre_saccade_window=pre_saccade_window,
                max_intersaccade_interval_for_classification=max_intersaccade_interval_for_classification,
                pre_saccade_velocity_threshold=pre_saccade_velocity_threshold,
                pre_saccade_drift_threshold=pre_saccade_drift_threshold,
                post_saccade_variance_threshold=post_saccade_variance_threshold,
                post_saccade_position_change_threshold_percent=post_saccade_position_change_threshold_percent,
                use_adaptive_thresholds=use_adaptive_thresholds,
                adaptive_percentile_pre_velocity=adaptive_percentile_pre_velocity,
                adaptive_percentile_pre_drift=adaptive_percentile_pre_drift,
                adaptive_percentile_post_variance=adaptive_percentile_post_variance,
                verbose=True
            )
            # Verify all saccades are classified (all branches in classification logic assign a type)
            unclassified = all_saccades_df[all_saccades_df['saccade_type'].isna()]
            if len(unclassified) > 0:
                print(f"⚠️ WARNING: {len(unclassified)} saccades were not classified!")
            else:
                # Confirm all saccades are classified
                n_total = len(all_saccades_df)
                n_orienting = (all_saccades_df['saccade_type'] == 'orienting').sum()
                n_compensatory = (all_saccades_df['saccade_type'] == 'compensatory').sum()
                if n_orienting + n_compensatory == n_total:
                    print(f"✅ All {n_total} saccades successfully classified ({n_orienting} orienting, {n_compensatory} compensatory)")
        else:
            # Skip classification - all saccades will have no saccade_type
            all_saccades_df['saccade_type'] = None
            all_saccades_df['bout_id'] = None
            all_saccades_df['bout_size'] = 1
            all_saccades_df['pre_saccade_mean_velocity'] = np.nan
            all_saccades_df['pre_saccade_position_drift'] = np.nan
            all_saccades_df['post_saccade_position_variance'] = np.nan
            all_saccades_df['post_saccade_position_change'] = np.nan
        
        # Split back into upward and downward with classification info
        # Match using original_index to preserve correct mapping
        upward_saccade_map = {}  # Maps original index -> classified info
        downward_saccade_map = {}
        
        if len(upward_saccades_df) > 0:
            upward_classified = all_saccades_df[all_saccades_df['is_upward'] == True].copy()
            # Match by original_index to preserve order
            for new_idx, row in upward_classified.iterrows():
                orig_idx = row['original_index']
                upward_saccade_map[orig_idx] = {
                    'saccade_type': row['saccade_type'],
                    'bout_id': row['bout_id'],
                    'bout_size': row['bout_size']
                }
            
            # Merge classification columns back to original DataFrame in correct order
            for orig_idx in upward_saccades_df.index:
                if orig_idx in upward_saccade_map:
                    info = upward_saccade_map[orig_idx]
                    upward_saccades_df.loc[orig_idx, 'saccade_type'] = info['saccade_type']
                    upward_saccades_df.loc[orig_idx, 'bout_id'] = info['bout_id']
                    upward_saccades_df.loc[orig_idx, 'bout_size'] = info['bout_size']
                    upward_saccades_df.loc[orig_idx, 'pre_saccade_mean_velocity'] = upward_classified[upward_classified['original_index'] == orig_idx]['pre_saccade_mean_velocity'].iloc[0]
                    upward_saccades_df.loc[orig_idx, 'pre_saccade_position_drift'] = upward_classified[upward_classified['original_index'] == orig_idx]['pre_saccade_position_drift'].iloc[0]
                    upward_saccades_df.loc[orig_idx, 'post_saccade_position_variance'] = upward_classified[upward_classified['original_index'] == orig_idx]['post_saccade_position_variance'].iloc[0]
                    upward_saccades_df.loc[orig_idx, 'post_saccade_position_change'] = upward_classified[upward_classified['original_index'] == orig_idx]['post_saccade_position_change'].iloc[0]
        
        if len(downward_saccades_df) > 0:
            downward_classified = all_saccades_df[all_saccades_df['is_upward'] == False].copy()
            # Match by original_index to preserve order
            for new_idx, row in downward_classified.iterrows():
                orig_idx = row['original_index']
                downward_saccade_map[orig_idx] = {
                    'saccade_type': row['saccade_type'],
                    'bout_id': row['bout_id'],
                    'bout_size': row['bout_size']
                }
            
            # Merge classification columns back to original DataFrame in correct order
            for orig_idx in downward_saccades_df.index:
                if orig_idx in downward_saccade_map:
                    info = downward_saccade_map[orig_idx]
                    downward_saccades_df.loc[orig_idx, 'saccade_type'] = info['saccade_type']
                    downward_saccades_df.loc[orig_idx, 'bout_id'] = info['bout_id']
                    downward_saccades_df.loc[orig_idx, 'bout_size'] = info['bout_size']
                    downward_saccades_df.loc[orig_idx, 'pre_saccade_mean_velocity'] = downward_classified[downward_classified['original_index'] == orig_idx]['pre_saccade_mean_velocity'].iloc[0]
                    downward_saccades_df.loc[orig_idx, 'pre_saccade_position_drift'] = downward_classified[downward_classified['original_index'] == orig_idx]['pre_saccade_position_drift'].iloc[0]
                    downward_saccades_df.loc[orig_idx, 'post_saccade_position_variance'] = downward_classified[downward_classified['original_index'] == orig_idx]['post_saccade_position_variance'].iloc[0]
                    downward_saccades_df.loc[orig_idx, 'post_saccade_position_change'] = downward_classified[downward_classified['original_index'] == orig_idx]['post_saccade_position_change'].iloc[0]
        
        # Add classification info to segments using saccade_id (which matches original DataFrame index)
        for seg in peri_saccades:
            seg_id = seg['saccade_id'].iloc[0]
            seg_dir = seg['saccade_direction'].iloc[0]
            
            if seg_dir == 'upward' and seg_id in upward_saccade_map:
                sacc_info = upward_saccade_map[seg_id]
                seg['saccade_type'] = sacc_info['saccade_type']
                seg['bout_id'] = sacc_info['bout_id']
                seg['bout_size'] = sacc_info['bout_size']
            elif seg_dir == 'downward' and seg_id in downward_saccade_map:
                sacc_info = downward_saccade_map[seg_id]
                seg['saccade_type'] = sacc_info['saccade_type']
                seg['bout_id'] = sacc_info['bout_id']
                seg['bout_size'] = sacc_info['bout_size']
    else:
        all_saccades_df = pd.DataFrame()

    summary = {
        'upward_saccades_df': upward_saccades_df,
        'downward_saccades_df': downward_saccades_df,
        'all_saccades_df': all_saccades_df,  # Combined DataFrame with classification
        'peri_saccades': peri_saccades,
        'upward_segments': upward_segments,
        'upward_outliers_meta': upward_outliers_meta,
        'upward_outlier_segments': upward_outlier_segments,
        'downward_segments': downward_segments,
        'downward_outliers_meta': downward_outliers_meta,
        'downward_outlier_segments': downward_outlier_segments,
        'vel_thresh': vel_thresh,
        'video_label': video_label,
        'all_excluded': all_excluded,
        'df': df
    }
    return summary

