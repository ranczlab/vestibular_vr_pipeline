"""
General processing functions for eye tracking data analysis.
Includes utility functions, blink detection, and data quality analysis.
"""

import numpy as np
import pandas as pd


def min_max_dict(coordinates_dict, columns_of_interest):
    """
    Calculate min and max values for X and Y coordinates from a coordinates dictionary.
    
    Parameters:
    -----------
    coordinates_dict : dict
        Dictionary containing coordinate arrays with keys like 'left.x', 'center.y', etc.
    columns_of_interest : list
        List of point names (without .x/.y suffix) to consider
        
    Returns:
    --------
    tuple : (x_min, x_max, y_min, y_max)
    """
    x_min = min([coordinates_dict[f'{col}.x'][~np.isnan(coordinates_dict[f'{col}.x'])].min() 
                 for col in columns_of_interest])
    x_max = max([coordinates_dict[f'{col}.x'][~np.isnan(coordinates_dict[f'{col}.x'])].max() 
                 for col in columns_of_interest])
    y_min = min([coordinates_dict[f'{col}.y'][~np.isnan(coordinates_dict[f'{col}.y'])].min() 
                 for col in columns_of_interest])
    y_max = max([coordinates_dict[f'{col}.y'][~np.isnan(coordinates_dict[f'{col}.y'])].max() 
                 for col in columns_of_interest])
    return x_min, x_max, y_min, y_max


def find_longest_lowscore_sections(scores, threshold, top_n=5):
    """
    Finds the top_n longest consecutive segments where scores < threshold.
    
    Parameters:
    -----------
    scores : pd.Series
        Series of score values
    threshold : float
        Threshold value below which scores are considered low
    top_n : int
        Number of top sections to return (default: 5)
        
    Returns:
    --------
    list : List of dicts with keys: {'start_idx', 'end_idx', 'length'}
    """
    is_low = scores < threshold
    diff = is_low.astype(int).diff().fillna(0)
    starts = np.where((diff == 1))[0]
    ends = np.where((diff == -1))[0]

    if is_low.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if is_low.iloc[-1]:
        ends = np.append(ends, len(is_low))

    sections = []
    for s, e in zip(starts, ends):
        sections.append({'start_idx': s, 'end_idx': e-1, 'length': e-s})

    top_sections = sorted(sections, key=lambda x: x['length'], reverse=True)[:top_n]
    return top_sections


def find_percentile_for_consecutive_limit(scores, max_consecutive, find_longest_lowscore_sections_fn=None):
    """
    Binary search for the lowest percentile so that the maximum number of
    consecutively excluded frames (scores below the percentile) does not exceed max_consecutive.
    
    Parameters:
    -----------
    scores : pd.Series
        Series of score values
    max_consecutive : int
        Maximum allowed consecutive frames below threshold
    find_longest_lowscore_sections_fn : callable, optional
        Function to find longest low score sections (default: uses find_longest_lowscore_sections)
        
    Returns:
    --------
    tuple : (percentile, threshold_value)
    """
    if find_longest_lowscore_sections_fn is None:
        find_longest_lowscore_sections_fn = find_longest_lowscore_sections
    
    sorted_scores = scores.dropna().sort_values()
    n = len(sorted_scores)
    low, high = 0.0, 0.2  # Search up to the 20th percentile; adjust as needed
    tol = 0.0001

    found_pct = None
    found_value = None

    while (high - low) > tol:
        mid = (low + high) / 2
        thr = sorted_scores.quantile(mid)
        sections = find_longest_lowscore_sections_fn(scores, thr, top_n=1)
        longest = sections[0]['length'] if sections else 0
        if longest > max_consecutive:
            high = mid
        else:
            found_pct = mid
            found_value = thr
            low = mid

    return found_pct, found_value


def find_blink_segments(instance_scores, threshold, min_frames, max_frames):
    """
    Find consecutive segments where instance.score < threshold.
    Filters by duration and returns valid blink segments.
    
    Parameters:
    -----------
    instance_scores : pd.Series
        Series of instance.score values
    threshold : float
        Score threshold below which to consider a blink
    min_frames : int
        Minimum number of consecutive frames to consider a blink
    max_frames : int
        Maximum number of consecutive frames to consider a blink
    
    Returns:
    --------
    list : List of dicts with keys: 'start_idx', 'end_idx', 'length', 'mean_score'
    """
    is_low = instance_scores < threshold
    diff = is_low.astype(int).diff().fillna(0)
    starts = np.where((diff == 1))[0]
    ends = np.where((diff == -1))[0]
    
    # Handle case where the first element is low
    if is_low.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if is_low.iloc[-1]:
        ends = np.append(ends, len(is_low))
    
    segments = []
    for s, e in zip(starts, ends):
        length = e - s
        if min_frames <= length <= max_frames:
            mean_score = instance_scores.iloc[s:e].mean()
            segments.append({
                'start_idx': s,
                'end_idx': e - 1,  # inclusive end
                'length': length,
                'mean_score': mean_score
            })
    
    return segments


def format_time_from_start(seconds_from_start):
    """
    Format seconds as MM:SS from the start of recording.
    
    Parameters:
    -----------
    seconds_from_start : float
        Time in seconds from the beginning of the recording
    
    Returns:
    --------
    str : Formatted as "MM:SS" or "M:SS" if minutes < 10
    """
    minutes = int(seconds_from_start // 60)
    seconds = int(seconds_from_start % 60)
    return f"{minutes}:{seconds:02d}"


def merge_nearby_blinks(segments, merge_window_frames):
    """
    Merge blink segments that are within merge_window_frames of each other.
    
    Parameters:
    -----------
    segments : list of dict
        List of blink segment dictionaries with keys: 'start_idx', 'end_idx', 'length', 'mean_score'
    merge_window_frames : int
        Maximum gap (in frames) between segments to merge them
        
    Returns:
    --------
    list : Merged segments list
    """
    if not segments:
        return segments
    
    # Sort by start_idx
    sorted_segments = sorted(segments, key=lambda x: x['start_idx'])
    merged = [sorted_segments[0]]
    
    for seg in sorted_segments[1:]:
        last = merged[-1]
        gap = seg['start_idx'] - last['end_idx'] - 1
        
        if gap <= merge_window_frames:
            # Merge: extend the last segment
            merged[-1]['end_idx'] = seg['end_idx']
            merged[-1]['length'] = merged[-1]['end_idx'] - merged[-1]['start_idx'] + 1
            merged[-1]['mean_score'] = (last['mean_score'] * last['length'] + 
                                       seg['mean_score'] * seg['length']) / merged[-1]['length']
        else:
            merged.append(seg)
    
    return merged

