"""
Saccade detection and processing functions for eye tracking data analysis.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.cm as cm


def detect_saccades_adaptive(df, position_col='X_smooth', velocity_col='vel_x_smooth', 
                             time_col='Seconds', fps=None, k=5, refractory_period=0.1,
                             onset_offset_fraction=0.2, peak_width=None, peak_width_time=None, verbose=True, upward_label=None, downward_label=None):
    """
    Detect saccades using adaptive statistical threshold method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing position and velocity data
    position_col : str
        Column name for position data (should be raw position for accurate amplitude, default: 'X_smooth')
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
    peak_width : int, optional
        Minimum peak width in samples for find_peaks (deprecated, use peak_width_time instead)
    peak_width_time : float, optional
        Minimum peak width in seconds for find_peaks (default: 0.01 = 10ms if peak_width not provided)
        Note: Typical saccades have peak widths of 5-20ms. Very short values (<5ms) may filter valid saccades.
    verbose : bool
        Whether to print detection statistics (default: True)
        
    Returns:
    --------
    tuple : (upward_saccades_df, downward_saccades_df, velocity_threshold)
    """
    # Calculate FPS if not provided
    if fps is None:
        fps = 1 / df[time_col].diff().mean()
    
    # Convert peak_width_time to points if provided, otherwise use peak_width
    if peak_width_time is not None:
        peak_width = max(1, int(round(peak_width_time * fps)))
    elif peak_width is None:
        # Default: 10ms peak width
        peak_width = max(1, int(round(0.01 * fps)))
    
    # Warn if peak width is very restrictive (less than 5ms equivalent)
    if peak_width_time is not None and peak_width_time < 0.005:
        import warnings
        warnings.warn(
            f"peak_width_time={peak_width_time*1000:.1f}ms may be too restrictive. "
            f"Typical saccade peaks are 5-20ms wide. Consider using 0.01 (10ms) or higher.",
            UserWarning
        )
    
    # 1. Calculate adaptive thresholds using statistical methods
    abs_vel = df[velocity_col].abs().dropna()
    vel_thresh = abs_vel.mean() + k * abs_vel.std()
    frame_idx_available = 'frame_idx' in df.columns
    
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
        # NOTE: Onset/offset detection can be affected by drift and noise. Consider improvements:
        # - Use adaptive thresholds based on local noise levels
        # - Apply smoothing to velocity before threshold detection
        # - Use position-based validation to ensure detected boundaries capture true saccade movement
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
        
        # Calculate amplitude (absolute change in position) and signed displacement
        start_x = df.iloc[start_idx][position_col]
        end_x = df.iloc[end_idx][position_col]
        amplitude = abs(end_x - start_x)
        displacement = end_x - start_x  # Signed displacement: positive = upward, negative = downward
        
        if frame_idx_available:
            start_frame_idx_val = df.iloc[start_idx]['frame_idx']
            peak_frame_idx_val = df.iloc[peak_idx]['frame_idx']
            end_frame_idx_val = df.iloc[end_idx]['frame_idx']
            start_frame_idx = int(start_frame_idx_val) if pd.notna(start_frame_idx_val) else np.nan
            peak_frame_idx = int(peak_frame_idx_val) if pd.notna(peak_frame_idx_val) else np.nan
            end_frame_idx = int(end_frame_idx_val) if pd.notna(end_frame_idx_val) else np.nan
        else:
            start_frame_idx = np.nan
            peak_frame_idx = np.nan
            end_frame_idx = np.nan

        upward_saccades.append({
            'time': peak_time,
            'velocity': peak_velocity,
            'start_time': onset_time,
            'end_time': offset_time,
            'duration': offset_time - onset_time,
            'start_position': start_x,
            'end_position': end_x,
            'amplitude': amplitude,
            'displacement': displacement,  # Signed displacement for direction validation
            'start_idx': int(start_idx),
            'peak_idx': int(peak_idx),
            'end_idx': int(end_idx),
            'start_frame_idx': start_frame_idx,
            'peak_frame_idx': peak_frame_idx,
            'end_frame_idx': end_frame_idx
        })
    
    downward_saccades = []
    for peak_idx in neg_peaks:
        peak_time = df.iloc[peak_idx][time_col]
        peak_velocity = df.iloc[peak_idx][velocity_col]
        
        # Find onset and offset
        # NOTE: Onset/offset detection can be affected by drift and noise. Consider improvements:
        # - Use adaptive thresholds based on local noise levels
        # - Apply smoothing to velocity before threshold detection
        # - Use position-based validation to ensure detected boundaries capture true saccade movement
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
        
        # Calculate amplitude (absolute change in position) and signed displacement
        start_x = df.iloc[start_idx][position_col]
        end_x = df.iloc[end_idx][position_col]
        amplitude = abs(end_x - start_x)
        displacement = end_x - start_x  # Signed displacement: positive = upward, negative = downward
        
        if frame_idx_available:
            start_frame_idx_val = df.iloc[start_idx]['frame_idx']
            peak_frame_idx_val = df.iloc[peak_idx]['frame_idx']
            end_frame_idx_val = df.iloc[end_idx]['frame_idx']
            start_frame_idx = int(start_frame_idx_val) if pd.notna(start_frame_idx_val) else np.nan
            peak_frame_idx = int(peak_frame_idx_val) if pd.notna(peak_frame_idx_val) else np.nan
            end_frame_idx = int(end_frame_idx_val) if pd.notna(end_frame_idx_val) else np.nan
        else:
            start_frame_idx = np.nan
            peak_frame_idx = np.nan
            end_frame_idx = np.nan

        downward_saccades.append({
            'time': peak_time,
            'velocity': peak_velocity,
            'start_time': onset_time,
            'end_time': offset_time,
            'duration': offset_time - onset_time,
            'start_position': start_x,
            'end_position': end_x,
            'amplitude': amplitude,
            'displacement': displacement,  # Signed displacement for direction validation
            'start_idx': int(start_idx),
            'peak_idx': int(peak_idx),
            'end_idx': int(end_idx),
            'start_frame_idx': start_frame_idx,
            'peak_frame_idx': peak_frame_idx,
            'end_frame_idx': end_frame_idx
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
        segment['saccade_displacement'] = sacc.get('displacement', np.nan)  # Signed displacement for direction validation
        segment['saccade_peak_velocity'] = sacc['velocity']
        segment['saccade_duration'] = sacc['duration']
        segment['is_saccade_period'] = (segment[time_col] >= start_time) & (segment[time_col] <= end_time)
        
        segments.append(segment)
    
    return segments, excluded_segments


def baseline_saccade_segments(segments, baseline_window_start_time, baseline_window_end_time,
                              position_col='X_smooth', time_rel_col='Time_rel_threshold'):
    """
    Baseline each saccade using the average value of data points in the baseline window.
    This removes pre-saccade position offsets and centers all saccades at baseline = 0.
    Also recalculates amplitude from baselined position to reflect true saccadic movement.
    
    Parameters:
    -----------
    segments : list of pd.DataFrame
        List of saccade segment DataFrames
    baseline_window_start_time : float
        Start time (seconds) relative to threshold crossing for baseline window (e.g., -0.1 for 100ms before)
    baseline_window_end_time : float
        End time (seconds) relative to threshold crossing for baseline window (e.g., -0.02 for 20ms before)
        Must be less than baseline_window_start_time (more negative = further before threshold)
    position_col : str
        Column name for position data to baseline (default: 'X_smooth')
    time_rel_col : str
        Column name for time relative to threshold (default: 'Time_rel_threshold')
        
    Returns:
    --------
    list : List of baselined segment DataFrames with updated amplitude values
    """
    # Ensure baseline_window_start_time is more negative (further before threshold) than baseline_window_end_time
    if baseline_window_start_time >= baseline_window_end_time:
        raise ValueError(
            f"baseline_window_start_time ({baseline_window_start_time}) must be less than "
            f"baseline_window_end_time ({baseline_window_end_time}) (more negative = further before threshold)"
        )
    
    for segment in segments:
        # Find indices where Time_rel_threshold is within the baseline window
        # Baseline window is between baseline_window_start_time and baseline_window_end_time (both negative)
        # CRITICAL: Only use points BEFORE threshold crossing (Time_rel_threshold < 0) to avoid using post-saccade data
        baseline_mask = (
            (segment[time_rel_col] >= baseline_window_start_time) & 
            (segment[time_rel_col] <= baseline_window_end_time) &
            (segment[time_rel_col] < 0)  # Ensure we only use pre-threshold points
        )
        
        # Filter out NaN values from position data - CRITICAL for robust baselining
        valid_baseline_mask = baseline_mask & segment[position_col].notna()
        
        if valid_baseline_mask.sum() > 0:
            # Extract baseline window with valid (non-NaN) position values
            baseline_indices = segment.index[valid_baseline_mask]
            baseline_positions = segment.loc[baseline_indices, position_col]
            
            # Calculate baseline as mean position in this window (already filtered for NaN)
            baseline_value = baseline_positions.mean()
            
            # Safety check: ensure baseline_value is valid (not NaN)
            if pd.isna(baseline_value):
                # If mean is still NaN (shouldn't happen after filtering, but safety check), use median
                baseline_value = baseline_positions.median()
                if pd.isna(baseline_value):
                    # Last resort: use first valid value before threshold
                    pre_threshold_mask = (segment[time_rel_col] < 0) & segment[position_col].notna()
                    pre_threshold_positions = segment.loc[pre_threshold_mask, position_col]
                    if len(pre_threshold_positions) > 0:
                        baseline_value = pre_threshold_positions.iloc[-1]  # Last point before threshold
                    else:
                        baseline_value = 0.0  # Fallback to zero
            
            # Subtract baseline from all position values in the segment
            segment['X_smooth_baselined'] = segment[position_col] - baseline_value
            segment['baseline_value'] = baseline_value
        else:
            # Fallback: use points BEFORE threshold crossing (Time_rel_threshold < 0)
            # This ensures we never use post-threshold data for baselining
            pre_threshold_mask = (segment[time_rel_col] < 0) & segment[position_col].notna()
            pre_threshold_positions = segment.loc[pre_threshold_mask, position_col]
            
            if len(pre_threshold_positions) > 0:
                # Use mean of all pre-threshold points, or a subset closest to baseline_window_end_time if available
                pre_threshold_times = segment.loc[pre_threshold_mask, time_rel_col]
                
                # Find points closest to baseline_window_end_time (but still before threshold)
                time_distances = (pre_threshold_times - baseline_window_end_time).abs()
                # Use up to 10 points closest to baseline_window_end_time, or all if fewer than 10
                n_points_to_use = min(10, len(pre_threshold_positions))
                closest_indices = time_distances.nsmallest(n_points_to_use).index
                
                baseline_value = segment.loc[closest_indices, position_col].mean()
                
                # Safety check for NaN
                if pd.isna(baseline_value):
                    baseline_value = pre_threshold_positions.mean()
                    if pd.isna(baseline_value):
                        baseline_value = pre_threshold_positions.iloc[-1] if len(pre_threshold_positions) > 0 else 0.0
                
                segment['X_smooth_baselined'] = segment[position_col] - baseline_value
                segment['baseline_value'] = baseline_value
            else:
                # If no pre-threshold points exist (shouldn't happen for valid segments)
                # This is a fallback for edge cases - try to use earliest available points
                # Find the earliest valid point in the segment (should be before threshold for valid segments)
                earliest_valid_idx = None
                for idx in segment.index:
                    if segment.loc[idx, time_rel_col] < 0 and pd.notna(segment.loc[idx, position_col]):
                        earliest_valid_idx = idx
                        break
                
                if earliest_valid_idx is not None:
                    # Use a small window around the earliest valid point
                    earliest_pos = segment.index.get_loc(earliest_valid_idx)
                    window_start = max(0, earliest_pos)
                    window_end = min(len(segment), earliest_pos + 5)  # Use up to 5 points
                    baseline_window = segment.iloc[window_start:window_end][position_col].dropna()
                    if len(baseline_window) > 0:
                        baseline_value = baseline_window.mean()
                        segment['X_smooth_baselined'] = segment[position_col] - baseline_value
                        segment['baseline_value'] = baseline_value
                    else:
                        # Use zero baseline as last resort
                        segment['X_smooth_baselined'] = segment[position_col]
                        segment['baseline_value'] = 0.0
                else:
                    # Truly no valid pre-threshold data - use zero baseline
                    segment['X_smooth_baselined'] = segment[position_col]
                    segment['baseline_value'] = 0.0
        
        # Final safety check: ensure baselined column exists and has valid values
        if 'X_smooth_baselined' not in segment.columns:
            segment['X_smooth_baselined'] = segment[position_col]
            segment['baseline_value'] = 0.0
        
        # Verify baseline was applied correctly (check that baseline_value is not NaN)
        if pd.isna(segment['baseline_value'].iloc[0] if len(segment) > 0 else np.nan):
            # If baseline_value is NaN, recalculate using any available pre-threshold points
            pre_threshold_mask = (segment[time_rel_col] < 0) & segment[position_col].notna()
            if pre_threshold_mask.sum() > 0:
                baseline_value = segment.loc[pre_threshold_mask, position_col].mean()
                if not pd.isna(baseline_value):
                    segment['X_smooth_baselined'] = segment[position_col] - baseline_value
                    segment['baseline_value'] = baseline_value
        
        # Recalculate amplitude from baselined position to reflect true saccadic movement
        # This removes the effect of pre-saccade offset/drift
        if 'X_smooth_baselined' in segment.columns and 'is_saccade_period' in segment.columns:
            # Use saccade period if available (most accurate)
            is_saccade_period = segment['is_saccade_period']
            if isinstance(is_saccade_period, pd.Series):
                saccade_mask = is_saccade_period.values
            else:
                saccade_mask = np.array([bool(is_saccade_period.iloc[0])] * len(segment)) if len(segment) > 0 else np.array([])
            
            if saccade_mask.sum() > 0:
                # Calculate amplitude from baselined position over saccade period
                saccade_positions_baselined = segment.loc[saccade_mask, 'X_smooth_baselined'].dropna().values
                if len(saccade_positions_baselined) > 0:
                    amplitude_baselined = abs(saccade_positions_baselined[-1] - saccade_positions_baselined[0])
                else:
                    # Fallback to original amplitude if no valid baselined positions
                    amplitude_baselined = segment['saccade_amplitude'].iloc[0] if 'saccade_amplitude' in segment.columns else np.nan
            else:
                # Fallback: use first and last baselined positions in segment
                baselined_positions = segment['X_smooth_baselined'].dropna().values
                if len(baselined_positions) > 1:
                    amplitude_baselined = abs(baselined_positions[-1] - baselined_positions[0])
                else:
                    # Fallback to original amplitude
                    amplitude_baselined = segment['saccade_amplitude'].iloc[0] if 'saccade_amplitude' in segment.columns else np.nan
        else:
            # If baselined column doesn't exist, keep original amplitude
            amplitude_baselined = segment['saccade_amplitude'].iloc[0] if 'saccade_amplitude' in segment.columns else np.nan
        
        # Update amplitude with baselined value
        if 'saccade_amplitude' in segment.columns:
            segment['saccade_amplitude'] = amplitude_baselined
        else:
            segment['saccade_amplitude'] = amplitude_baselined
    
    return segments


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
    # COMMENTED OUT: Bout-based classification removed per user request
    # Orienting saccades can also occur in quick bouts (especially pairs), so bout detection
    # alone is not sufficient for classification. Bout detection is kept for analysis purposes
    # but classification now uses only feature-based logic for all saccades.
    # Original logic (now commented):
    # - If in a bout (bout_size >= 2) → automatically compensatory
    # - If isolated → use feature-based classification
    
    # Initialize confidence column
    all_saccades_df['classification_confidence'] = np.nan
    
    for idx, saccade in all_saccades_df.iterrows():
        bout_size = saccade['bout_size']
        pre_vel = saccade['pre_saccade_mean_velocity']
        pre_drift = saccade['pre_saccade_position_drift']
        post_var = saccade['post_saccade_position_variance']
        post_change = saccade['post_saccade_position_change']
        amplitude = saccade['amplitude']
        
        # Classification rules (now applies to ALL saccades, regardless of bout_size):
        # COMMENTED OUT: Bout-based classification
        # if bout_size >= 2:
        #     # Multiple saccades in bout → compensatory
        #     all_saccades_df.loc[idx, 'saccade_type'] = 'compensatory'
        # else:
        #     # Isolated saccade - use feature-based classification
        
        # Feature-based classification (applies to all saccades):
        # - Pre-saccade velocity > threshold OR pre-saccade drift significant → compensatory
        # - Post-saccade position change > amplitude * threshold_percent → compensatory (eye continues moving)
        # - Pre-saccade velocity low AND post-saccade stable (variance AND position change) → orienting
        
        # Calculate position change threshold relative to amplitude
        position_change_threshold = amplitude * (post_saccade_position_change_threshold_percent / 100.0)
        
        # Check for compensatory indicators
        has_pre_saccade_drift = (pre_vel > pre_saccade_velocity_threshold or pre_drift > pre_saccade_drift_threshold)
        has_post_saccade_movement = (post_change > position_change_threshold)
        
        # Calculate confidence score based on feature agreement and strength
        # Confidence ranges from 0.0 (uncertain) to 1.0 (very confident)
        confidence = 0.5  # Start with medium confidence
        
        if has_pre_saccade_drift or has_post_saccade_movement:
            # Evidence of drift/compensation → compensatory
            all_saccades_df.loc[idx, 'saccade_type'] = 'compensatory'
            
            # Calculate confidence for compensatory classification
            compensatory_indicators = 0
            strength_scores = []
            
            # Check pre-saccade velocity indicator
            if pre_vel > pre_saccade_velocity_threshold:
                compensatory_indicators += 1
                # Normalized distance: how far above threshold (clamped to reasonable range)
                if pre_saccade_velocity_threshold > 0:
                    normalized_dist = min((pre_vel - pre_saccade_velocity_threshold) / pre_saccade_velocity_threshold, 2.0)
                    strength_scores.append(normalized_dist)
            
            # Check pre-saccade drift indicator
            if pre_drift > pre_saccade_drift_threshold:
                compensatory_indicators += 1
                # Normalized distance: how far above threshold
                if pre_saccade_drift_threshold > 0:
                    normalized_dist = min((pre_drift - pre_saccade_drift_threshold) / pre_saccade_drift_threshold, 2.0)
                    strength_scores.append(normalized_dist)
            
            # Check post-saccade movement indicator
            if post_change > position_change_threshold:
                compensatory_indicators += 1
                # Normalized distance: how far above threshold
                if position_change_threshold > 0:
                    normalized_dist = min((post_change - position_change_threshold) / position_change_threshold, 2.0)
                    strength_scores.append(normalized_dist)
            
            # Count orienting indicators (disagree with compensatory)
            orienting_indicators = 0
            if pre_vel <= pre_saccade_velocity_threshold and pre_drift <= pre_saccade_drift_threshold:
                orienting_indicators += 1
            if post_var < post_saccade_variance_threshold:
                orienting_indicators += 1
            if post_change <= position_change_threshold:
                orienting_indicators += 1
            
            # Base confidence from feature agreement
            # More compensatory indicators = higher confidence
            # More orienting indicators = lower confidence
            total_indicators = compensatory_indicators + orienting_indicators
            if total_indicators > 0:
                agreement_ratio = compensatory_indicators / total_indicators
                # Map agreement ratio to confidence: 0.5 (balanced) to 0.9 (strong agreement)
                base_confidence = 0.5 + (agreement_ratio - 0.5) * 0.8
            
            # Strength modifier: how strongly features agree
            if len(strength_scores) > 0:
                avg_strength = np.mean(strength_scores)
                # Map strength to modifier: ±0.1 based on average strength
                # Use tanh to cap the modifier
                strength_modifier = np.tanh(avg_strength) * 0.1
                confidence = np.clip(base_confidence + strength_modifier, 0.0, 1.0)
            else:
                confidence = base_confidence
            
            # Rule-based boost: multiple strong indicators
            if compensatory_indicators >= 2 and len(strength_scores) > 0 and np.mean(strength_scores) > 0.5:
                confidence = min(confidence + 0.1, 1.0)
            
        elif (pre_vel <= pre_saccade_velocity_threshold and 
              post_var < post_saccade_variance_threshold and
              post_change <= position_change_threshold):
            # Stable before and after → orienting
            all_saccades_df.loc[idx, 'saccade_type'] = 'orienting'
            
            # Calculate confidence for orienting classification
            orienting_indicators = 0
            strength_scores = []
            
            # Check pre-saccade velocity (should be low)
            if pre_vel <= pre_saccade_velocity_threshold:
                orienting_indicators += 1
                # Normalized distance: how far below threshold
                if pre_saccade_velocity_threshold > 0:
                    normalized_dist = min((pre_saccade_velocity_threshold - pre_vel) / pre_saccade_velocity_threshold, 2.0)
                    strength_scores.append(normalized_dist)
            
            # Check pre-saccade drift (should be low)
            if pre_drift <= pre_saccade_drift_threshold:
                orienting_indicators += 1
                # Normalized distance: how far below threshold
                if pre_saccade_drift_threshold > 0:
                    normalized_dist = min((pre_saccade_drift_threshold - pre_drift) / pre_saccade_drift_threshold, 2.0)
                    strength_scores.append(normalized_dist)
            
            # Check post-saccade variance (should be low)
            if post_var < post_saccade_variance_threshold:
                orienting_indicators += 1
                # Normalized distance: how far below threshold
                if post_saccade_variance_threshold > 0:
                    normalized_dist = min((post_saccade_variance_threshold - post_var) / post_saccade_variance_threshold, 2.0)
                    strength_scores.append(normalized_dist)
            
            # Check post-saccade position change (should be low relative to amplitude)
            if post_change <= position_change_threshold:
                orienting_indicators += 1
                # Normalized distance: how far below threshold
                if position_change_threshold > 0:
                    normalized_dist = min((position_change_threshold - post_change) / position_change_threshold, 2.0)
                    strength_scores.append(normalized_dist)
            
            # Count compensatory indicators (disagree with orienting)
            compensatory_indicators = 0
            if pre_vel > pre_saccade_velocity_threshold or pre_drift > pre_saccade_drift_threshold:
                compensatory_indicators += 1
            if post_change > position_change_threshold:
                compensatory_indicators += 1
            
            # Base confidence from feature agreement
            total_indicators = orienting_indicators + compensatory_indicators
            if total_indicators > 0:
                agreement_ratio = orienting_indicators / total_indicators
                # Map agreement ratio to confidence: 0.5 (balanced) to 0.9 (strong agreement)
                base_confidence = 0.5 + (agreement_ratio - 0.5) * 0.8
            
            # Strength modifier: how strongly features agree
            if len(strength_scores) > 0:
                avg_strength = np.mean(strength_scores)
                # Map strength to modifier: ±0.1 based on average strength
                strength_modifier = np.tanh(avg_strength) * 0.1
                confidence = np.clip(base_confidence + strength_modifier, 0.0, 1.0)
            else:
                confidence = base_confidence
            
            # Rule-based boost: all conditions met strongly
            if orienting_indicators >= 3 and len(strength_scores) > 0 and np.mean(strength_scores) > 0.5:
                confidence = min(confidence + 0.1, 1.0)
            
        else:
            # Default: if uncertain, classify as compensatory (conservative)
            all_saccades_df.loc[idx, 'saccade_type'] = 'compensatory'
            
            # Low confidence for default classification (conflicting or weak indicators)
            confidence = 0.3
        
        # Store confidence score
        all_saccades_df.loc[idx, 'classification_confidence'] = confidence
    
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
        
        # Confidence statistics
        if 'classification_confidence' in all_saccades_df.columns:
            conf_mean = all_saccades_df['classification_confidence'].mean()
            conf_std = all_saccades_df['classification_confidence'].std()
            conf_min = all_saccades_df['classification_confidence'].min()
            conf_max = all_saccades_df['classification_confidence'].max()
            print(f"\n  Classification Confidence:")
            print(f"    Mean: {conf_mean:.3f} ± {conf_std:.3f}")
            print(f"    Range: [{conf_min:.3f}, {conf_max:.3f}]")
            
            # Confidence by type
            if n_orienting > 0:
                orienting_conf = all_saccades_df[all_saccades_df['saccade_type'] == 'orienting']['classification_confidence']
                print(f"    Orienting: {orienting_conf.mean():.3f} ± {orienting_conf.std():.3f}")
            if n_compensatory > 0:
                compensatory_conf = all_saccades_df[all_saccades_df['saccade_type'] == 'compensatory']['classification_confidence']
                print(f"    Compensatory: {compensatory_conf.mean():.3f} ± {compensatory_conf.std():.3f}")
            
            # Count by confidence levels
            high_conf = (all_saccades_df['classification_confidence'] >= 0.7).sum()
            med_conf = ((all_saccades_df['classification_confidence'] >= 0.4) & 
                       (all_saccades_df['classification_confidence'] < 0.7)).sum()
            low_conf = (all_saccades_df['classification_confidence'] < 0.4).sum()
            print(f"    High confidence (≥0.7): {high_conf} ({high_conf/len(all_saccades_df)*100:.1f}%)")
            print(f"    Medium confidence (0.4-0.7): {med_conf} ({med_conf/len(all_saccades_df)*100:.1f}%)")
            print(f"    Low confidence (<0.4): {low_conf} ({low_conf/len(all_saccades_df)*100:.1f}%)")
        
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
    pre_saccade_window_time=None,  # Time (seconds) before threshold crossing to extract
    post_saccade_window_time=None,  # Time (seconds) after threshold crossing to extract
    baseline_window_start_time=None,  # Start time (seconds) relative to threshold crossing for baseline window (e.g., -0.1)
    baseline_window_end_time=None,  # End time (seconds) relative to threshold crossing for baseline window (e.g., -0.02)
    smoothing_window_time=None,  # Time (seconds) for smoothing window
    peak_width_time=None,  # Minimum peak width (seconds) - typically 5-20ms for saccades
    min_saccade_duration=None,  # Minimum saccade duration (seconds) - saccades shorter than this are excluded (default: 0.2)
    # Backward compatibility: old point-based parameters (deprecated)
    n_before=None,
    n_after=None,
    baseline_n_points=None,
    baseline_window_time=None,  # Deprecated: use baseline_window_start_time and baseline_window_end_time instead
    saccade_smoothing_window=None,
    saccade_peak_width=None,
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
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'Ellipse.Center.X' and 'Seconds' columns
    fps : float
        Frames per second (used to convert time-based parameters to points)
    video_label : str
        Label for this video (for logging)
    k : float
        Threshold multiplier for adaptive detection (default: 5)
    refractory_period : float
        Minimum time between saccades in seconds (default: 0.1)
    onset_offset_fraction : float
        Fraction of peak velocity threshold for onset detection (default: 0.2)
    pre_saccade_window_time : float
        Time (seconds) before threshold crossing to extract (default: 0.15)
    post_saccade_window_time : float
        Time (seconds) after threshold crossing to extract (default: 0.5)
    baseline_window_start_time : float
        Start time (seconds) relative to threshold crossing for baseline window (default: -0.1)
        Negative value means before threshold crossing (e.g., -0.1 = 100ms before)
    baseline_window_end_time : float
        End time (seconds) relative to threshold crossing for baseline window (default: -0.02)
        Negative value means before threshold crossing (e.g., -0.02 = 20ms before)
        Must be greater than baseline_window_start_time (less negative, closer to threshold)
    smoothing_window_time : float
        Time (seconds) for smoothing window (default: 0.08)
    peak_width_time : float
        Minimum peak width in seconds for find_peaks (default: 0.01 = 10ms)
        Note: Typical saccades have peak widths of 5-20ms. Very short values (<5ms) may filter valid saccades.
    min_saccade_duration : float
        Minimum saccade segment duration in seconds (default: 0.2 = 200ms)
        Segments shorter than this are excluded as they are likely truncated at recording edges or contain data gaps.
        Note: This filters based on the extracted segment duration (pre_saccade_window_time + post_saccade_window_time),
        not the saccade duration itself. Typical segments are 0.65s (0.15s before + 0.5s after), so 0.2s is conservative.
    ...
    Returns dict of results including all relevant outputs for stats/plots.
    """
    import numpy as np
    import pandas as pd
    # sp is this module (sleap.saccade_processing): use local functions

    # Backward compatibility: Convert old point-based parameters to time-based if provided
    if n_before is not None:
        import warnings
        warnings.warn(
            "Parameter 'n_before' is deprecated. Use 'pre_saccade_window_time' instead. "
            f"Converting {n_before} points to time using fps={fps:.1f}.",
            DeprecationWarning
        )
        if pre_saccade_window_time is None:
            pre_saccade_window_time = n_before / fps
    
    if n_after is not None:
        import warnings
        warnings.warn(
            "Parameter 'n_after' is deprecated. Use 'post_saccade_window_time' instead. "
            f"Converting {n_after} points to time using fps={fps:.1f}.",
            DeprecationWarning
        )
        if post_saccade_window_time is None:
            post_saccade_window_time = n_after / fps
    
    if baseline_n_points is not None:
        import warnings
        warnings.warn(
            "Parameter 'baseline_n_points' is deprecated. Use 'baseline_window_start_time' and 'baseline_window_end_time' instead. "
            f"Converting {baseline_n_points} points to time using fps={fps:.1f}.",
            DeprecationWarning
        )
        if baseline_window_start_time is None:
            # Convert to time: assume baseline window ends just before threshold (0)
            baseline_window_end_time = -0.02  # 20ms before threshold
            baseline_window_start_time = -(baseline_n_points / fps)  # Start further back
        if baseline_window_time is None:
            baseline_window_time = baseline_n_points / fps
    
    if baseline_window_time is not None:
        import warnings
        warnings.warn(
            "Parameter 'baseline_window_time' is deprecated. Use 'baseline_window_start_time' and 'baseline_window_end_time' instead. "
            f"Converting {baseline_window_time}s to start/end times.",
            DeprecationWarning
        )
        if baseline_window_start_time is None:
            baseline_window_start_time = -baseline_window_time  # Start at baseline_window_time before threshold
        if baseline_window_end_time is None:
            baseline_window_end_time = -0.02  # End 20ms before threshold
    
    if saccade_smoothing_window is not None:
        import warnings
        warnings.warn(
            "Parameter 'saccade_smoothing_window' is deprecated. Use 'smoothing_window_time' instead. "
            f"Converting {saccade_smoothing_window} points to time using fps={fps:.1f}.",
            DeprecationWarning
        )
        if smoothing_window_time is None:
            smoothing_window_time = saccade_smoothing_window / fps
    
    if saccade_peak_width is not None:
        import warnings
        warnings.warn(
            "Parameter 'saccade_peak_width' is deprecated. Use 'peak_width_time' instead. "
            f"Converting {saccade_peak_width} points to time using fps={fps:.1f}.",
            DeprecationWarning
        )
        if peak_width_time is None:
            peak_width_time = saccade_peak_width / fps
    
    # Set defaults for time-based parameters if not provided
    if pre_saccade_window_time is None:
        pre_saccade_window_time = 0.15  # 150ms default
    if post_saccade_window_time is None:
        post_saccade_window_time = 0.5  # 500ms default
    if baseline_window_start_time is None:
        baseline_window_start_time = -0.1  # 100ms before threshold default
    if baseline_window_end_time is None:
        baseline_window_end_time = -0.02  # 20ms before threshold default
    if smoothing_window_time is None:
        smoothing_window_time = 0.08  # 80ms default
    if peak_width_time is None:
        peak_width_time = 0.01  # 10ms default (was 1 frame, which could be 8-16ms depending on FPS)
    if min_saccade_duration is None:
        min_saccade_duration = 0.2  # 200ms default - exclude segments shorter than this (typically truncated at edges)
    
    # Convert time-based parameters to points using FPS
    # Round to nearest integer, minimum 1 point
    n_before = max(1, int(round(pre_saccade_window_time * fps)))
    n_after = max(1, int(round(post_saccade_window_time * fps)))
    saccade_smoothing_window = max(1, int(round(smoothing_window_time * fps)))
    saccade_peak_width = max(1, int(round(peak_width_time * fps)))
    
    # Warn if peak_width_time is very restrictive
    if peak_width_time < 0.005:  # Less than 5ms
        import warnings
        warnings.warn(
            f"peak_width_time={peak_width_time*1000:.1f}ms may be too restrictive. "
            f"Typical saccade peaks are 5-20ms wide. Consider using 0.01 (10ms) or higher.",
            UserWarning
        )

    # Preprocessing: Smooth position first (reduces noise before differentiation), then compute velocity
    # Use raw position for accurate amplitude calculations
    df = df.copy()

    # Preserve frame indices so downstream steps can map back to original video frames
    if 'frame_idx' not in df.columns:
        df['frame_idx'] = df.index
    df['original_dataframe_index'] = df.index

    # Ensure a clean positional index to avoid label/position confusion downstream
    df = df.reset_index(drop=True)
    
    # Keep raw position for accurate amplitude calculations
    df['X_raw'] = df['Ellipse.Center.X']
    
    # Smooth position FIRST (before differentiation) - this reduces noise more effectively
    # Differentiation amplifies high-frequency noise, so smoothing before differentiation is crucial
    df['X_smooth'] = (
        df['X_raw']
        .rolling(window=saccade_smoothing_window, center=True)
        .median()
        .bfill()
        .ffill()
    )

    # Compute instantaneous velocity from SMOOTHED position (not raw)
    # This gives smooth velocity because we smoothed before differentiating
    df['dt'] = df['Seconds'].diff()
    df['vel_x_smooth'] = df['X_smooth'].diff() / df['dt']
    
    # Also compute velocity from raw position for reference (noisy, not used for detection)
    df['vel_x_raw'] = df['X_raw'].diff() / df['dt']
    df['vel_x_original'] = df['vel_x_raw']  # Keep for reference

    # Saccade detection
    # Use raw position for accurate amplitude calculations, smooth velocity for detection
    upward_saccades_df, downward_saccades_df, vel_thresh = detect_saccades_adaptive(
        df,
        position_col='X_raw',  # Use raw position for accurate amplitude
        velocity_col='vel_x_smooth',  # Use smoothed velocity for robust detection
        time_col='Seconds',
        fps=fps,
        k=k,
        refractory_period=refractory_period,
        onset_offset_fraction=onset_offset_fraction,
        peak_width_time=peak_width_time,  # Pass time-based parameter
        verbose=True,
        upward_label=upward_label,
        downward_label=downward_label
    )

    # Peri-saccade segment extraction
    # Use raw position for accurate segment data (amplitude will be recalculated from baselined position)
    peri_saccades = []
    all_excluded = []
    if len(upward_saccades_df) > 0:
        upward_segments, upward_excluded = extract_saccade_segment(
            df, upward_saccades_df, n_before, n_after, 
            direction='upward', position_col='X_raw'
        )
        peri_saccades.extend(upward_segments)
        all_excluded.extend(upward_excluded)
    if len(downward_saccades_df) > 0:
        downward_segments, downward_excluded = extract_saccade_segment(
            df, downward_saccades_df, n_before, n_after, 
            direction='downward', position_col='X_raw'
        )
        peri_saccades.extend(downward_segments)
        all_excluded.extend(downward_excluded)

    # Filter segments by minimum segment duration
    # Exclude segments shorter than min_saccade_duration (typically truncated at recording edges or data gaps)
    # This filters based on the actual extracted segment duration, not the saccade duration
    filtered_peri_saccades = []
    excluded_by_duration = []
    excluded_saccade_ids = set()  # Track saccade IDs to remove from DataFrames
    
    for seg in peri_saccades:
        # Calculate segment duration from actual time span
        # Time_rel_threshold is relative to threshold crossing, so duration = max - min
        segment_duration = seg['Time_rel_threshold'].max() - seg['Time_rel_threshold'].min()
        
        if segment_duration >= min_saccade_duration:
            filtered_peri_saccades.append(seg)
        else:
            # Segment too short - exclude it
            seg_id = seg['saccade_id'].iloc[0] if 'saccade_id' in seg.columns else None
            direction = seg['saccade_direction'].iloc[0] if 'saccade_direction' in seg.columns else 'unknown'
            
            excluded_by_duration.append({
                'saccade_id': seg_id,
                'direction': direction,
                'segment_duration': segment_duration,
                'min_required': min_saccade_duration,
                'time_range': (seg['Time_rel_threshold'].min(), seg['Time_rel_threshold'].max())
            })
            
            if seg_id is not None:
                excluded_saccade_ids.add(seg_id)
    
    # Report exclusions
    n_excluded_upward = sum(1 for ex in excluded_by_duration if ex['direction'] == 'upward')
    n_excluded_downward = sum(1 for ex in excluded_by_duration if ex['direction'] == 'downward')
    
    if len(excluded_by_duration) > 0:
        print(f"  Excluded {len(excluded_by_duration)} segment(s) with duration < {min_saccade_duration:.3f}s")
        if n_excluded_upward > 0:
            print(f"    - {n_excluded_upward} upward segment(s)")
        if n_excluded_downward > 0:
            print(f"    - {n_excluded_downward} downward segment(s)")
    
    # Update saccade DataFrames to remove saccades whose segments were filtered
    # This ensures consistency between segments and DataFrames
    if len(excluded_saccade_ids) > 0:
        n_upward_before = len(upward_saccades_df)
        n_downward_before = len(downward_saccades_df)
        
        if len(upward_saccades_df) > 0:
            upward_saccades_df = upward_saccades_df[~upward_saccades_df.index.isin(excluded_saccade_ids)].copy()
            n_upward_removed = n_upward_before - len(upward_saccades_df)
            if n_upward_removed > 0:
                print(f"  Removed {n_upward_removed} upward saccade(s) from DataFrame (segments filtered)")
        
        if len(downward_saccades_df) > 0:
            downward_saccades_df = downward_saccades_df[~downward_saccades_df.index.isin(excluded_saccade_ids)].copy()
            n_downward_removed = n_downward_before - len(downward_saccades_df)
            if n_downward_removed > 0:
                print(f"  Removed {n_downward_removed} downward saccade(s) from DataFrame (segments filtered)")
    
    # Use filtered segments for downstream processing
    peri_saccades = filtered_peri_saccades
    # Add duration-filtered segments to all_excluded for reporting
    all_excluded.extend(excluded_by_duration)

    # Baselining
    # Use raw position for baselining, then recalculate amplitude from baselined position
    # This removes pre-saccade offset/drift and gives true saccadic amplitude
    # Column name 'X_smooth_baselined' kept for backward compatibility
    peri_saccades = baseline_saccade_segments(
        peri_saccades, 
        baseline_window_start_time=baseline_window_start_time,
        baseline_window_end_time=baseline_window_end_time,
        position_col='X_raw'
    )

    # Use all segments (outlier filtering removed per user request)
    upward_segments_all = [seg for seg in peri_saccades if seg['saccade_direction'].iloc[0] == 'upward']
    downward_segments_all = [seg for seg in peri_saccades if seg['saccade_direction'].iloc[0] == 'downward']
    upward_segments = upward_segments_all
    downward_segments = downward_segments_all

    # Update amplitude in DataFrames to match baselined amplitude from segments
    # This ensures consistency between segment amplitude and DataFrame amplitude
    if len(upward_saccades_df) > 0:
        for seg in peri_saccades:
            if seg['saccade_direction'].iloc[0] == 'upward':
                seg_id = seg['saccade_id'].iloc[0]
                if seg_id in upward_saccades_df.index:
                    baselined_amp = seg['saccade_amplitude'].iloc[0]
                    upward_saccades_df.loc[seg_id, 'amplitude'] = baselined_amp
    
    if len(downward_saccades_df) > 0:
        for seg in peri_saccades:
            if seg['saccade_direction'].iloc[0] == 'downward':
                seg_id = seg['saccade_id'].iloc[0]
                if seg_id in downward_saccades_df.index:
                    baselined_amp = seg['saccade_amplitude'].iloc[0]
                    downward_saccades_df.loc[seg_id, 'amplitude'] = baselined_amp

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
                position_col='X_raw',  # Use raw position for accurate drift/variance calculations
                velocity_col='vel_x_smooth',  # Use smoothed velocity for classification
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
        'downward_segments': downward_segments,
        'vel_thresh': vel_thresh,
        'video_label': video_label,
        'all_excluded': all_excluded,
        'df': df
    }
    return summary

