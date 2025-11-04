"""
Saccade detection and processing functions for eye tracking data analysis.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.cm as cm


def detect_saccades_adaptive(df, position_col='X_smooth', velocity_col='vel_x_smooth', 
                             time_col='Seconds', fps=None, k=5, refractory_period=0.1,
                             onset_offset_fraction=0.2, verbose=True, upward_label=None, downward_label=None):
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
        width=1  # Minimum peak width in samples
    )
    
    # Find negative peaks (downward saccades) by inverting the signal
    neg_peaks, neg_properties = find_peaks(
        -df[velocity_col],  # Invert to find troughs
        height=vel_thresh,  # Same threshold
        distance=int(fps * refractory_period),
        width=1
    )
    
    # 3. Extract saccade information
    upward_saccades = []
    for peak_idx in pos_peaks:
        peak_time = df.iloc[peak_idx][time_col]
        peak_velocity = df.iloc[peak_idx][velocity_col]
        
        # Find onset and offset by going backward/forward until velocity drops below threshold
        start_idx = peak_idx
        end_idx = peak_idx
        
        # Find onset (go backward)
        while start_idx > 0 and abs(df.iloc[start_idx][velocity_col]) > vel_thresh * onset_offset_fraction:
            start_idx -= 1
        
        # Find offset (go forward)
        while end_idx < len(df) - 1 and abs(df.iloc[end_idx][velocity_col]) > vel_thresh * onset_offset_fraction:
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
        
        # Find onset
        while start_idx > 0 and abs(df.iloc[start_idx][velocity_col]) > vel_thresh * onset_offset_fraction:
            start_idx -= 1
        
        # Find offset
        while end_idx < len(df) - 1 and abs(df.iloc[end_idx][velocity_col]) > vel_thresh * onset_offset_fraction:
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
    upward_label=None,
    downward_label=None,
):
    """
    Analyze saccades for a given eye video dataframe.
    Performs smoothing, velocity computation, adaptive saccade detection, peri-event extraction, baselining, outlier filtering.
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
        .rolling(window=5, center=True)
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

    summary = {
        'upward_saccades_df': upward_saccades_df,
        'downward_saccades_df': downward_saccades_df,
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

