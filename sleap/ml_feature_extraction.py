"""
ML Feature Extraction for Saccade Classification

This module extracts features from detected saccades for machine learning classification.
Features are organized by category as defined in FEATURE_SELECTION_LIST.md.

Note: Velocity and position profile features (Categories E and F) are excluded
for undersampled data (<120 Hz FPS). These can be added when FPS increases significantly.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple


def extract_experiment_id(data_path: Path) -> str:
    """
    Extract experiment ID from data path using Option 2 (readable components).
    
    Extracts: Cohort + condition + animal + date
    Example: "Cohort3_Visual_mismatch_day4_B6J2782_2025-04-28"
    
    Expected path structure:
    .../vestVR/20250409_Cohort3_rotation/Visual_mismatch_day4/B6J2782-2025-04-28T14-22-03
    
    Parameters:
    -----------
    data_path : Path
        Path to experiment data directory
        
    Returns:
    --------
    str : Experiment ID in format "CohortX_Condition_Animal_Date"
    """
    path_str = str(data_path)
    path_parts = Path(data_path).parts
    
    cohort = None
    condition = None
    animal = None
    date = None
    
    # Look for cohort pattern in path parts (e.g., "20250409_Cohort3_rotation")
    for part in path_parts:
        if 'cohort' in part.lower():
            # Extract cohort number (e.g., "Cohort3" from "20250409_Cohort3_rotation")
            match = re.search(r'[Cc]ohort\s*(\d+)', part, re.IGNORECASE)
            if match:
                cohort = f"Cohort{match.group(1)}"
            break
    
    # Look for condition (e.g., "Visual_mismatch_day4")
    # Usually contains underscores and descriptive text, appears before animal-date folder
    for i, part in enumerate(path_parts):
        if '_' in part and ('mismatch' in part.lower() or 'day' in part.lower() or 'rotation' in part.lower() or 'visual' in part.lower()):
            condition = part
            break
    
    # Look for animal ID and date (last directory, format: "B6J2782-2025-04-28T14-22-03")
    if len(path_parts) > 0:
        last_part = path_parts[-1]
        # Try to extract animal ID and date
        # Pattern: AnimalID-DateTime or AnimalID-Date
        match = re.match(r'([A-Z0-9]+)-(\d{4}-\d{2}-\d{2})', last_part)
        if match:
            animal = match.group(1)
            date = match.group(2)  # Extract date part only (YYYY-MM-DD)
    
    # Fallback: use path components if extraction fails
    if not all([cohort, condition, animal, date]):
        # Try alternative extraction methods
        # If still missing components, use hash as fallback
        import hashlib
        path_hash = hashlib.md5(path_str.encode()).hexdigest()[:12]
        
        # Try to use last few path components as readable ID
        if len(path_parts) >= 2:
            # Use last 2-3 components
            readable_parts = '_'.join(path_parts[-3:])
            # Clean up special characters
            readable_parts = re.sub(r'[^\w\-_]', '_', readable_parts)
            return readable_parts[:100]  # Limit length
        
        return f"exp_{path_hash}"
    
    # Clean up components (remove special characters that might cause issues)
    cohort = re.sub(r'[^\w]', '_', cohort)
    condition = re.sub(r'[^\w]', '_', condition)
    animal = re.sub(r'[^\w]', '_', animal)
    
    experiment_id = f"{cohort}_{condition}_{animal}_{date}"
    return experiment_id


def extract_ml_features(
    saccade_results: Dict,
    df: pd.DataFrame,
    fps: float,
    data_path: Path,
    pre_saccade_window: float = 0.3,
    max_intersaccade_interval: float = 5.0,
    position_col: str = 'X_smooth',
    velocity_col: str = 'vel_x_smooth',
    time_col: str = 'Seconds',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract ML features from detected saccades.
    
    Extracts features from Categories A, B, C, D, G, H as selected by user.
    Categories E and F (velocity/position profile) are excluded for undersampled data.
    
    Parameters:
    -----------
    saccade_results : dict
        Dictionary returned by analyze_eye_video_saccades containing:
        - 'all_saccades_df': Combined DataFrame of all saccades
        - 'df': Full time series DataFrame
        - 'peri_saccades': List of peri-saccade segment DataFrames
        - Other metadata
    df : pd.DataFrame
        Full time series DataFrame with position and velocity data
    fps : float
        Frames per second
    data_path : Path
        Path to experiment data (used to generate experiment_id)
    pre_saccade_window : float
        Time window (seconds) before saccade onset for pre-saccade features (default: 0.3)
    max_intersaccade_interval : float
        Maximum time (seconds) for post-saccade window (default: 5.0)
    position_col : str
        Column name for position data (default: 'X_smooth')
    velocity_col : str
        Column name for velocity data (default: 'vel_x_smooth')
    time_col : str
        Column name for time data (default: 'Seconds')
    verbose : bool
        Whether to print progress (default: True)
        
    Returns:
    --------
    pd.DataFrame : Features DataFrame with one row per saccade
        Columns include all extracted features plus:
        - 'experiment_id': Experiment identifier
        - 'saccade_id': Unique saccade ID within experiment
        - 'video_label': Video label (VideoData1/VideoData2)
    """
    # Extract experiment ID
    experiment_id = extract_experiment_id(data_path)
    
    # Get all saccades DataFrame
    if 'all_saccades_df' in saccade_results:
        all_saccades_df = saccade_results['all_saccades_df'].copy()
    else:
        # Combine upward and downward if not already combined
        upward_df = saccade_results.get('upward_saccades_df', pd.DataFrame())
        downward_df = saccade_results.get('downward_saccades_df', pd.DataFrame())
        if len(upward_df) > 0 and len(downward_df) > 0:
            all_saccades_df = pd.concat([upward_df, downward_df], ignore_index=True).sort_values('time').reset_index(drop=True)
        elif len(upward_df) > 0:
            all_saccades_df = upward_df.copy()
        elif len(downward_df) > 0:
            all_saccades_df = downward_df.copy()
        else:
            if verbose:
                print("⚠️ No saccades found in saccade_results")
            return pd.DataFrame()
    
    # Get video label
    video_label = saccade_results.get('video_label', 'Unknown')
    
    # Initialize features DataFrame
    features_df = pd.DataFrame()
    features_df['experiment_id'] = experiment_id
    features_df['saccade_id'] = range(len(all_saccades_df))
    features_df['video_label'] = video_label
    
    # Sort by time to ensure chronological order
    all_saccades_df = all_saccades_df.sort_values('time').reset_index(drop=True)
    
    if verbose:
        print(f"\n📊 Extracting ML features for {len(all_saccades_df)} saccades...")
        print(f"   Experiment ID: {experiment_id}")
        print(f"   Video Label: {video_label}")
    
    # ============================================================================
    # CATEGORY A: Basic Saccade Properties (Already Extracted)
    # ============================================================================
    if verbose:
        print("   Category A: Basic properties...")
    
    features_df['amplitude'] = all_saccades_df['amplitude'].values
    features_df['duration'] = all_saccades_df['duration'].values
    features_df['peak_velocity'] = all_saccades_df['velocity'].values  # Note: 'velocity' column contains peak velocity
    features_df['direction'] = all_saccades_df['direction'].map({'upward': 1, 'downward': -1}).values  # Encode as 1/-1
    features_df['start_time'] = all_saccades_df['start_time'].values
    features_df['end_time'] = all_saccades_df['end_time'].values
    features_df['time'] = all_saccades_df['time'].values  # Peak time
    
    # ============================================================================
    # CATEGORY B: Pre-Saccade Features
    # ============================================================================
    if verbose:
        print("   Category B: Pre-saccade features...")
    
    # Get existing features from classification (if available)
    if 'pre_saccade_mean_velocity' in all_saccades_df.columns:
        features_df['pre_saccade_mean_velocity'] = all_saccades_df['pre_saccade_mean_velocity'].values
    else:
        features_df['pre_saccade_mean_velocity'] = np.nan
    
    if 'pre_saccade_position_drift' in all_saccades_df.columns:
        features_df['pre_saccade_position_drift'] = all_saccades_df['pre_saccade_position_drift'].values
    else:
        features_df['pre_saccade_position_drift'] = np.nan
    
    # Extract new pre-saccade features
    pre_saccade_position_variance = []
    pre_saccade_drift_rate = []
    pre_saccade_window_durations = []
    
    for idx, saccade in all_saccades_df.iterrows():
        start_time = saccade['start_time']
        peak_time = saccade['time']
        
        # Calculate pre-saccade window
        desired_pre_window_start = start_time - pre_saccade_window
        
        # Constrain to previous saccade's peak time (if exists)
        if idx > 0:
            previous_peak_time = all_saccades_df.iloc[idx - 1]['time']
            pre_window_start_time = max(desired_pre_window_start, previous_peak_time)
        else:
            pre_window_start_time = desired_pre_window_start
        
        # Extract pre-saccade data
        pre_mask = (df[time_col] >= pre_window_start_time) & (df[time_col] < start_time)
        pre_positions = df.loc[pre_mask, position_col]
        pre_velocities = df.loc[pre_mask, velocity_col]
        
        # Calculate features
        if len(pre_positions) > 1:
            pre_var = pre_positions.var()
            pre_drift = abs(pre_positions.iloc[-1] - pre_positions.iloc[0])
            # Calculate window duration from time column
            pre_times = df.loc[pre_mask, time_col]
            window_duration = pre_times.iloc[-1] - pre_times.iloc[0] if len(pre_times) > 1 else 0
            if window_duration > 0:
                drift_rate = pre_drift / window_duration
            else:
                drift_rate = 0.0
        else:
            pre_var = 0.0
            drift_rate = 0.0
            window_duration = 0.0
        
        pre_saccade_position_variance.append(pre_var)
        pre_saccade_drift_rate.append(drift_rate)
        pre_saccade_window_durations.append(window_duration)
    
    features_df['pre_saccade_position_variance'] = pre_saccade_position_variance
    features_df['pre_saccade_drift_rate'] = pre_saccade_drift_rate
    features_df['pre_saccade_window_duration'] = pre_saccade_window_durations
    
    # ============================================================================
    # CATEGORY C: Post-Saccade Features
    # ============================================================================
    if verbose:
        print("   Category C: Post-saccade features...")
    
    # Get existing features from classification (if available)
    if 'post_saccade_position_variance' in all_saccades_df.columns:
        features_df['post_saccade_position_variance'] = all_saccades_df['post_saccade_position_variance'].values
    else:
        features_df['post_saccade_position_variance'] = np.nan
    
    if 'post_saccade_position_change' in all_saccades_df.columns:
        features_df['post_saccade_position_change'] = all_saccades_df['post_saccade_position_change'].values
    else:
        features_df['post_saccade_position_change'] = np.nan
    
    # Extract new post-saccade features
    post_saccade_position_change_pct = []
    post_saccade_mean_velocity = []
    post_saccade_drift_rate = []
    post_saccade_window_durations = []
    
    for idx, saccade in all_saccades_df.iterrows():
        end_time = saccade['end_time']
        amplitude = saccade['amplitude']
        
        # Calculate post-saccade window (dynamic - extends until next saccade or max interval)
        next_idx = idx + 1
        if next_idx < len(all_saccades_df):
            next_start_time = all_saccades_df.iloc[next_idx]['start_time']
            interval_to_next = next_start_time - end_time
            dynamic_window_duration = min(interval_to_next, max_intersaccade_interval)
        else:
            dynamic_window_duration = max_intersaccade_interval
        
        post_window_end_time = end_time + dynamic_window_duration
        
        # Extract post-saccade data
        post_mask = (df[time_col] > end_time) & (df[time_col] <= post_window_end_time)
        post_positions = df.loc[post_mask, position_col]
        post_velocities = df.loc[post_mask, velocity_col]
        
        # Calculate features
        if len(post_positions) > 1:
            post_change = abs(post_positions.iloc[-1] - post_positions.iloc[0])
            post_change_pct = (post_change / amplitude * 100) if amplitude > 0 else 0.0
            post_mean_vel = post_velocities.abs().mean() if len(post_velocities) > 0 else 0.0
            # Calculate window duration from time column
            post_times = df.loc[post_mask, time_col]
            window_duration = post_times.iloc[-1] - post_times.iloc[0] if len(post_times) > 1 else 0
            if window_duration > 0:
                drift_rate = post_change / window_duration
            else:
                drift_rate = 0.0
        else:
            post_change_pct = 0.0
            post_mean_vel = 0.0
            drift_rate = 0.0
            window_duration = 0.0
        
        post_saccade_position_change_pct.append(post_change_pct)
        post_saccade_mean_velocity.append(post_mean_vel)
        post_saccade_drift_rate.append(drift_rate)
        post_saccade_window_durations.append(window_duration)
    
    features_df['post_saccade_position_change_pct'] = post_saccade_position_change_pct
    features_df['post_saccade_mean_velocity'] = post_saccade_mean_velocity
    features_df['post_saccade_drift_rate'] = post_saccade_drift_rate
    features_df['post_saccade_window_duration'] = post_saccade_window_durations
    
    # ============================================================================
    # CATEGORY D: Temporal Context Features (excluding bout_density)
    # ============================================================================
    if verbose:
        print("   Category D: Temporal context features...")
    
    # Get existing bout features
    if 'bout_size' in all_saccades_df.columns:
        bout_sizes_array = all_saccades_df['bout_size'].values
        features_df['bout_size'] = bout_sizes_array
    else:
        # Calculate bout_size if not available (simple temporal clustering)
        bout_sizes_array = calculate_bout_sizes(all_saccades_df, bout_window=1.5)
        features_df['bout_size'] = bout_sizes_array
    
    if 'bout_id' in all_saccades_df.columns:
        bout_ids = all_saccades_df['bout_id'].values
    else:
        bout_ids = calculate_bout_ids(all_saccades_df, bout_window=1.5)
    
    # Calculate temporal context features
    time_since_previous = []
    time_until_next = []
    position_in_bout = []
    is_first_in_bout = []
    is_last_in_bout = []
    is_isolated = []
    bout_durations = []
    inter_saccade_interval_mean = []
    inter_saccade_interval_std = []
    
    # Group by bout_id to calculate bout-level features
    bout_groups = {}
    for idx, bout_id in enumerate(bout_ids):
        if bout_id not in bout_groups:
            bout_groups[bout_id] = []
        bout_groups[bout_id].append(idx)
    
    for idx, saccade in all_saccades_df.iterrows():
        current_time = saccade['time']
        current_bout_id = bout_ids[idx]
        
        # Time since previous saccade
        if idx > 0:
            time_since = current_time - all_saccades_df.iloc[idx - 1]['time']
        else:
            time_since = np.nan  # First saccade
        time_since_previous.append(time_since)
        
        # Time until next saccade
        if idx < len(all_saccades_df) - 1:
            time_until = all_saccades_df.iloc[idx + 1]['time'] - current_time
        else:
            time_until = np.nan  # Last saccade
        time_until_next.append(time_until)
        
        # Bout-related features
        bout_indices = bout_groups[current_bout_id]
        try:
            position_in_bout_val = bout_indices.index(idx) + 1  # 1-indexed
        except ValueError:
            # Should not happen, but handle gracefully
            position_in_bout_val = 1
        position_in_bout.append(position_in_bout_val)
        
        is_first = (position_in_bout_val == 1)
        is_first_in_bout.append(is_first)
        
        is_last = (position_in_bout_val == len(bout_indices))
        is_last_in_bout.append(is_last)
        
        # Check if isolated using bout_size array
        bout_size_val = bout_sizes_array[idx]
        is_isolated_val = (bout_size_val == 1)
        is_isolated.append(is_isolated_val)
        
        # Bout duration
        if len(bout_indices) > 1:
            bout_start_time = all_saccades_df.iloc[bout_indices[0]]['start_time']
            bout_end_time = all_saccades_df.iloc[bout_indices[-1]]['end_time']
            bout_duration = bout_end_time - bout_start_time
        else:
            bout_duration = saccade['duration']  # Isolated saccade
        bout_durations.append(bout_duration)
        
        # Inter-saccade intervals within bout
        if len(bout_indices) > 1:
            bout_times = [all_saccades_df.iloc[i]['time'] for i in bout_indices]
            intervals = np.diff(bout_times)
            inter_mean = intervals.mean() if len(intervals) > 0 else np.nan
            inter_std = intervals.std() if len(intervals) > 1 else 0.0
        else:
            inter_mean = np.nan
            inter_std = np.nan
        inter_saccade_interval_mean.append(inter_mean)
        inter_saccade_interval_std.append(inter_std)
    
    features_df['time_since_previous_saccade'] = time_since_previous
    features_df['time_until_next_saccade'] = time_until_next
    features_df['position_in_bout'] = position_in_bout
    features_df['is_first_in_bout'] = np.array(is_first_in_bout).astype(int)
    features_df['is_last_in_bout'] = np.array(is_last_in_bout).astype(int)
    features_df['is_isolated'] = np.array(is_isolated).astype(int)
    features_df['bout_duration'] = bout_durations
    features_df['inter_saccade_interval_mean'] = inter_saccade_interval_mean
    features_df['inter_saccade_interval_std'] = inter_saccade_interval_std
    
    # ============================================================================
    # CATEGORY E & F: Velocity and Position Profile Features
    # ============================================================================
    # NOTE: Excluded for undersampled data (<120 Hz FPS)
    # TODO: Add these features when FPS increases significantly (e.g., to 120 Hz or more)
    # See FEATURE_SELECTION_LIST.md Categories E and F for available features
    
    # ============================================================================
    # CATEGORY G: Amplitude & Direction Consistency Features
    # ============================================================================
    if verbose:
        print("   Category G: Amplitude & direction consistency...")
    
    # Calculate bout-level statistics
    bout_amplitudes = {}
    bout_directions = {}
    for bout_id in set(bout_ids):
        bout_indices = bout_groups[bout_id]
        bout_amps = [all_saccades_df.iloc[i]['amplitude'] for i in bout_indices]
        bout_dirs = [all_saccades_df.iloc[i]['direction'] for i in bout_indices]
        bout_amplitudes[bout_id] = bout_amps
        bout_directions[bout_id] = bout_dirs
    
    amplitude_relative_to_bout_mean = []
    amplitude_consistency_in_bout = []
    direction_relative_to_previous = []
    
    for idx, saccade in all_saccades_df.iterrows():
        current_bout_id = bout_ids[idx]
        current_amplitude = saccade['amplitude']
        current_direction = saccade['direction']
        
        # Amplitude relative to bout mean
        if current_bout_id in bout_amplitudes and len(bout_amplitudes[current_bout_id]) > 1:
            bout_amps = np.array(bout_amplitudes[current_bout_id])
            bout_mean = bout_amps.mean()
            bout_std = bout_amps.std()
            if bout_std > 0:
                amplitude_relative = (current_amplitude - bout_mean) / bout_std
            else:
                amplitude_relative = 0.0
            amplitude_consistency = bout_std
        else:
            amplitude_relative = 0.0  # Isolated saccade
            amplitude_consistency = 0.0
        amplitude_relative_to_bout_mean.append(amplitude_relative)
        amplitude_consistency_in_bout.append(amplitude_consistency)
        
        # Direction relative to previous
        if idx > 0:
            prev_direction = all_saccades_df.iloc[idx - 1]['direction']
            if current_direction == prev_direction:
                direction_relative = 1  # Same direction
            else:
                direction_relative = -1  # Opposite direction
        else:
            direction_relative = 0  # First saccade
        direction_relative_to_previous.append(direction_relative)
    
    features_df['amplitude_relative_to_bout_mean'] = amplitude_relative_to_bout_mean
    features_df['amplitude_consistency_in_bout'] = amplitude_consistency_in_bout
    features_df['direction_relative_to_previous'] = direction_relative_to_previous
    
    # ============================================================================
    # CATEGORY H: Rule-Based Classification Features
    # ============================================================================
    if verbose:
        print("   Category H: Rule-based classification features...")
    
    # Get rule-based classification if available
    # NOTE: Classification encoding scheme (used by both rule-based and ML):
    #   0 = Compensatory
    #   1 = Orienting (Purely Orienting)
    #   2 = Saccade-Fixate (first saccade of compensatory bout)
    #   3 = Non-Saccade (false positives to exclude)
    #   -1 = Unclassified/Unknown
    #
    # Current rule-based classification only produces classes 0 and 1.
    # ML classification will produce all 4 classes (0, 1, 2, 3).
    # When ML predictions are added, they should use the same numeric encoding.
    
    if 'saccade_type' in all_saccades_df.columns:
        # Map rule-based string labels to numeric: compensatory=0, orienting=1
        # Note: Current rule-based classification only has 2 classes
        # ML classification will add: saccade_and_fixate=2, non_saccade=3
        type_map = {'compensatory': 0, 'orienting': 1}
        # Map known types, fill unknown/None/NaN with -1 (Unclassified)
        features_df['rule_based_class'] = all_saccades_df['saccade_type'].map(type_map)
        # Fill NaN values (None, unknown types) with -1
        features_df['rule_based_class'] = features_df['rule_based_class'].fillna(-1).astype(int)
        
        if verbose:
            unique_types = all_saccades_df['saccade_type'].unique()
            print(f"      Found saccade_type values: {unique_types}")
            print(f"      Mapped rule_based_class values: {features_df['rule_based_class'].unique()}")
            print(f"      Class counts: {features_df['rule_based_class'].value_counts().to_dict()}")
    elif 'ml_class' in all_saccades_df.columns:
        # If ML predictions are present, use them directly (should already be numeric 0-3)
        features_df['rule_based_class'] = all_saccades_df['ml_class'].fillna(-1).astype(int)
        if verbose:
            print(f"      Using ML classification (ml_class column)")
            print(f"      ML class values: {features_df['rule_based_class'].unique()}")
            print(f"      Class counts: {features_df['rule_based_class'].value_counts().to_dict()}")
    else:
        if verbose:
            print(f"      ⚠️ Neither 'saccade_type' nor 'ml_class' column found in all_saccades_df")
            print(f"      Available columns: {list(all_saccades_df.columns)[:20]}...")
        features_df['rule_based_class'] = -1  # Not classified
    
    if 'classification_confidence' in all_saccades_df.columns:
        features_df['rule_based_confidence'] = all_saccades_df['classification_confidence'].values
    else:
        features_df['rule_based_confidence'] = np.nan
    
    # Note: 'classification_confidence' column name is already used above
    
    if verbose:
        print(f"✅ Feature extraction complete: {len(features_df)} saccades, {len(features_df.columns)} features")
        print(f"   Feature columns: {list(features_df.columns)}")
    
    return features_df


def calculate_bout_sizes(all_saccades_df: pd.DataFrame, bout_window: float = 1.5) -> np.ndarray:
    """
    Calculate bout sizes using temporal clustering.
    
    Helper function if bout_size is not already in DataFrame.
    """
    bout_sizes = np.ones(len(all_saccades_df), dtype=int)
    bout_id = 0
    current_bout_start_idx = 0
    
    for i in range(len(all_saccades_df)):
        if i == 0:
            bout_id = 1
        else:
            time_interval = all_saccades_df.iloc[i]['time'] - all_saccades_df.iloc[i-1]['time']
            if time_interval > bout_window:
                # New bout
                bout_id += 1
                # Update sizes for previous bout
                bout_size = i - current_bout_start_idx
                bout_sizes[current_bout_start_idx:i] = bout_size
                current_bout_start_idx = i
    
    # Update last bout
    bout_size = len(all_saccades_df) - current_bout_start_idx
    bout_sizes[current_bout_start_idx:] = bout_size
    
    return bout_sizes


def calculate_bout_ids(all_saccades_df: pd.DataFrame, bout_window: float = 1.5) -> np.ndarray:
    """
    Calculate bout IDs using temporal clustering.
    
    Helper function if bout_id is not already in DataFrame.
    """
    bout_ids = np.zeros(len(all_saccades_df), dtype=int)
    bout_id = 0
    
    for i in range(len(all_saccades_df)):
        if i == 0:
            bout_id = 1
            bout_ids[i] = bout_id
        else:
            time_interval = all_saccades_df.iloc[i]['time'] - all_saccades_df.iloc[i-1]['time']
            if time_interval > bout_window:
                bout_id += 1
            bout_ids[i] = bout_id
    
    return bout_ids

