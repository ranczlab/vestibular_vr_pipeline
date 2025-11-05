"""
General processing functions for eye tracking data analysis.
Includes utility functions, blink detection, and data quality analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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


def check_manual_match(auto_start, auto_end, manual_blinks_list):
    """
    Check if an auto-detected blink matches any manual blink annotation.
    
    Uses >=40% overlap threshold to handle over-merged manual blinks.
    
    Parameters:
    -----------
    auto_start : int
        Start frame of auto-detected blink
    auto_end : int
        End frame of auto-detected blink
    manual_blinks_list : list of dict or None
        List of manual blink dicts with keys: 'start', 'end', 'num'
        
    Returns:
    --------
    int : 1 if match found, 0 otherwise
    """
    if manual_blinks_list is None:
        return 0
    for manual in manual_blinks_list:
        overlap_start = max(manual['start'], auto_start)
        overlap_end = min(manual['end'], auto_end)
        if overlap_start <= overlap_end:
            overlap_frames = overlap_end - overlap_start + 1
            manual_length = manual['end'] - manual['start'] + 1
            # Use 40% threshold to match the diagnostic comparison logic
            if overlap_frames >= manual_length * 0.40:
                return 1
    return 0


def compare_manual_vs_auto_blinks(blink_segments, video_data, manual_blinks, video_label, debug=False):
    """
    Compare manual blink annotations with auto-detected blinks.
    
    Prints detailed comparison including overlap analysis.
    
    Parameters:
    -----------
    blink_segments : list of dict
        Auto-detected blink segments with keys: 'start_idx', 'end_idx', 'length'
    video_data : pd.DataFrame
        Video data with 'frame_idx' and 'Seconds' columns
    manual_blinks : list of dict or None
        Manual blink annotations with keys: 'start', 'end', 'num'
    video_label : str
        Label for the video (for printing)
    debug : bool
        Whether to print detailed comparison
        
    Returns:
    --------
    list : List of auto-detected blink dicts with frame numbers
    """
    if not debug or manual_blinks is None:
        return []
    
    # Get auto-detected blink frame ranges
    auto_blinks = []
    for i, blink in enumerate(blink_segments, 1):
        start_idx = blink['start_idx']
        end_idx = blink['end_idx']
        if 'frame_idx' in video_data.columns:
            frame_start = int(video_data['frame_idx'].iloc[start_idx])
            frame_end = int(video_data['frame_idx'].iloc[end_idx])
        else:
            frame_start = start_idx
            frame_end = end_idx
        auto_blinks.append({
            'num': i,
            'start': frame_start,
            'end': frame_end,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'length': blink['length']
        })
    
    print(f"\n   🔍 MANUAL vs AUTO-DETECTED BLINK COMPARISON:")
    print(f"      Manual blinks: {len(manual_blinks)}")
    print(f"      Auto-detected blinks: {len(auto_blinks)}")
    
    # Match manual blinks to auto-detected ones (find overlapping frames)
    # Allow matching with multiple auto-detected blinks if manual blink is over-merged
    print(f"\n   Manual → Auto matching (overlap analysis):")
    for manual in manual_blinks:
        manual_length = manual['end'] - manual['start'] + 1
        total_overlap = 0
        matching_auto_blinks = []
        
        # Find all auto-detected blinks that overlap with this manual blink
        for auto in auto_blinks:
            overlap_start = max(manual['start'], auto['start'])
            overlap_end = min(manual['end'], auto['end'])
            if overlap_start <= overlap_end:
                overlap_frames = overlap_end - overlap_start + 1
                total_overlap += overlap_frames
                matching_auto_blinks.append({
                    'auto': auto,
                    'overlap': overlap_frames
                })
        
        # Calculate total overlap percentage
        total_overlap_pct = (total_overlap / manual_length) * 100 if manual_length > 0 else 0
        
        # Match if total overlap is >= 40% (less stringent to handle over-merged manual blinks)
        if total_overlap >= manual_length * 0.40:  # At least 40% overlap
            if len(matching_auto_blinks) == 1:
                # Single match
                match_info = matching_auto_blinks[0]
                best_match = match_info['auto']
                overlap = match_info['overlap']
                start_diff = best_match['start'] - manual['start']
                end_diff = best_match['end'] - manual['end']
                match_str = f"✅ MATCH: Auto blink {best_match['num']} (frames {best_match['start']}-{best_match['end']})"
                match_str += f", {overlap} frames overlap ({total_overlap_pct:.1f}%)"
                if start_diff != 0 or end_diff != 0:
                    match_str += f", offset: start={start_diff:+d}, end={end_diff:+d}"
                print(f"      Manual {manual['num']}: {manual['start']}-{manual['end']} → {match_str}")
            else:
                # Multiple matches (manual blink is over-merged)
                auto_nums = [m['auto']['num'] for m in matching_auto_blinks]
                auto_ranges = [f"{m['auto']['start']}-{m['auto']['end']}" for m in matching_auto_blinks]
                match_str = f"✅ MATCH: Auto blinks {auto_nums} (frames: {', '.join(auto_ranges)})"
                match_str += f", total {total_overlap} frames overlap ({total_overlap_pct:.1f}%)"
                print(f"      Manual {manual['num']}: {manual['start']}-{manual['end']} → {match_str}")
        else:
            print(f"      Manual {manual['num']}: {manual['start']}-{manual['end']} → ❌ NO MATCH FOUND")
    
    # Also check which auto-detected blinks don't have manual matches
    print(f"\n   Auto-detected blinks without manual matches:")
    unmatched_auto = []
    for auto in auto_blinks:
        has_match = False
        for manual in manual_blinks:
            overlap_start = max(manual['start'], auto['start'])
            overlap_end = min(manual['end'], auto['end'])
            if overlap_start <= overlap_end:
                overlap_frames = overlap_end - overlap_start + 1
                manual_length = manual['end'] - manual['start'] + 1
                # Use same threshold as matching logic (40% of manual blink)
                if overlap_frames >= manual_length * 0.40:
                    has_match = True
                    break
        if not has_match:
            unmatched_auto.append(auto)
    
    if len(unmatched_auto) > 0:
        for auto in unmatched_auto:
            print(f"      Auto {auto['num']}: frames {auto['start']}-{auto['end']}, length={auto['length']} frames")
    else:
        print(f"      (all auto-detected blinks have manual matches)")
    
    return auto_blinks


def detect_blinks_for_video(video_data, columns_of_interest, blink_instance_score_threshold,
                            long_blink_warning_ms, min_frames_threshold=4, merge_window_frames=10,
                            fps=None, video_label="", manual_blinks=None, debug=False):
    """
    Complete blink detection pipeline for a single video.
    
    Detects blinks, filters by duration, merges nearby blinks, compares with manual annotations,
    interpolates short blinks, and marks long blinks as NaN.
    
    Parameters:
    -----------
    video_data : pd.DataFrame
        Video data with 'instance.score', 'Seconds', 'frame_idx' columns
    columns_of_interest : list of str
        List of coordinate column names to set to NaN during blinks
    blink_instance_score_threshold : float
        Hard threshold for blink detection (frames with instance.score below this are blinks)
    long_blink_warning_ms : float
        Warn if blinks exceed this duration (in ms)
    min_frames_threshold : int
        Minimum frames for a blink to be kept (shorter ones are interpolated)
    merge_window_frames : int
        Merge blinks within this many frames into blink bouts
    fps : float, optional
        Frames per second (calculated from video_data if not provided)
    video_label : str
        Label for the video (for printing)
    manual_blinks : list of dict or None
        Manual blink annotations for comparison (keys: 'start', 'end', 'num')
    debug : bool
        Whether to print detailed information
        
    Returns:
    --------
    dict : Dictionary with keys:
        - 'blink_segments': List of long blink segments (>= min_frames_threshold)
        - 'short_blink_segments': List of short blink segments (< min_frames_threshold)
        - 'blink_bouts': List of merged blink bouts
        - 'all_blink_segments': List of all detected blink segments
        - 'fps': Calculated or provided FPS
        - 'long_blinks_warnings': List of warnings for long blinks
        - 'total_blink_frames': Total frames marked as blinks
        - 'blink_bout_rate': Blinks per minute
    """
    if debug:
        print(f"\n{video_label} - Blink Detection")
    
    # Calculate FPS if not provided
    if fps is None:
        fps = 1 / video_data["Seconds"].diff().mean()
    
    long_blink_warning_frames = int(long_blink_warning_ms / 1000 * fps)
    
    if debug:
        print(f"  FPS: {fps:.2f}")
        print(f"  Long blink warning threshold: {long_blink_warning_frames} frames ({long_blink_warning_ms}ms)")
    
    # Use hard threshold from user parameters
    blink_threshold = blink_instance_score_threshold
    if debug:
        print(f"  Using hard threshold: {blink_threshold:.4f}")
    
    # Find all blink segments - use very lenient min_frames (1) to capture all segments
    # No filtering by frame count - short blinks are OK to interpolate
    # No merging - we want to preserve good data between separate blinks
    all_blink_segments = find_blink_segments(
        video_data['instance.score'], 
        blink_threshold, 
        min_frames=1,  # Very lenient to capture all segments
        max_frames=999999  # Very high limit - essentially no maximum
    )
    
    # Always print key blink detection stats
    print(f"{video_label} - Found {len(all_blink_segments)} blink segments")
    
    # Filter out blinks shorter than min_frames_threshold frames
    blink_segments = [blink for blink in all_blink_segments if blink['length'] >= min_frames_threshold]
    short_blink_segments = [blink for blink in all_blink_segments if blink['length'] < min_frames_threshold]
    
    # Always print filtering stats
    print(f"  After filtering <{min_frames_threshold} frames: {len(blink_segments)} blink segment(s), {len(short_blink_segments)} short segment(s) will be interpolated")
    
    # Merge blinks within merge_window_frames into blink bouts
    blink_bouts = merge_nearby_blinks(blink_segments, merge_window_frames)
    
    # Always print bout count
    print(f"  After merging blinks within {merge_window_frames} frames: {len(blink_bouts)} blink bout(s)")
    
    # Check for long blinks and warn if needed
    long_blinks_warnings = []
    for i, blink in enumerate(blink_segments, 1):
        start_idx = blink['start_idx']
        end_idx = blink['end_idx']
        start_time = video_data['Seconds'].iloc[start_idx]
        end_time = video_data['Seconds'].iloc[end_idx]
        duration_ms = (end_time - start_time) * 1000
        
        # Warn about very long blinks (may need manual verification)
        if duration_ms > long_blink_warning_ms:
            if 'frame_idx' in video_data.columns:
                frame_start = int(video_data['frame_idx'].iloc[start_idx])
                frame_end = int(video_data['frame_idx'].iloc[end_idx])
            else:
                frame_start = start_idx
                frame_end = end_idx
            long_blinks_warnings.append({
                'blink_num': i,
                'frames': f"{frame_start}-{frame_end}",
                'duration_ms': duration_ms
            })
    
    # Print warnings for long blinks
    if len(long_blinks_warnings) > 0:
        print(f"\n   ⚠️ WARNING: Found {len(long_blinks_warnings)} blink(s) longer than {long_blink_warning_ms}ms:")
        for warn in long_blinks_warnings:
            print(f"      Blink {warn['blink_num']}: frames {warn['frames']}, duration {warn['duration_ms']:.1f}ms - Please verify this is a real blink in the video")
    
    if debug:
        print(f"\n  Detailed blink detection information:")
        print(f"  FPS: {fps:.2f}")
        print(f"  Long blink warning threshold: {long_blink_warning_frames} frames ({long_blink_warning_ms}ms)")
        print(f"  Using hard threshold: {blink_threshold:.4f}")
        print(f"  Detected {len(blink_segments)} blink segment(s)\n")
    
    # Print all detected blinks once (detailed) - only in debug mode
    if debug and len(blink_segments) > 0:
        print(f"  Detected blinks:")
        for i, blink in enumerate(blink_segments, 1):
            start_idx = blink['start_idx']
            end_idx = blink['end_idx']
            
            # Calculate time range
            start_time = video_data['Seconds'].iloc[start_idx]
            end_time = video_data['Seconds'].iloc[end_idx]
            duration_ms = (end_time - start_time) * 1000
            
            # Get actual frame numbers from frame_idx column
            if 'frame_idx' in video_data.columns:
                actual_start_frame = int(video_data['frame_idx'].iloc[start_idx])
                actual_end_frame = int(video_data['frame_idx'].iloc[end_idx])
                frame_info = f"frames {actual_start_frame}-{actual_end_frame}"
            else:
                actual_start_frame = start_idx
                actual_end_frame = end_idx
                frame_info = f"frames {actual_start_frame}-{actual_end_frame}"
            
            print(f"    Blink {i}: {frame_info}, {blink['length']} frames, {duration_ms:.1f}ms, mean score: {blink['mean_score']:.4f}")
        
        # DIAGNOSTIC: Direct comparison of manual vs auto-detected blinks (detailed)
        if manual_blinks is not None:
            compare_manual_vs_auto_blinks(blink_segments, video_data, manual_blinks, video_label, debug=True)
        else:
            print(f"\n   ⚠️ WARNING: Manual blink comparison skipped - manual_blinks not provided")
    elif debug:
        print("  No blinks detected")
    
    # Always interpolate short blinks and mark long blinks (regardless of debug mode)
    total_blink_frames = 0
    short_blink_frames = 0
    
    if len(blink_segments) > 0 or len(short_blink_segments) > 0:
        # Interpolate short blinks first (if any)
        if len(short_blink_segments) > 0:
            if debug:
                print(f"\n  Interpolating short blinks (< {min_frames_threshold} frames):")
            for blink in short_blink_segments:
                start_idx = blink['start_idx']
                end_idx = blink['end_idx']
                video_data.loc[video_data.index[start_idx:end_idx+1], columns_of_interest] = np.nan
                short_blink_frames += blink['length']
            
            # Interpolate all NaNs (this fills short blinks)
            video_data[columns_of_interest] = video_data[columns_of_interest].interpolate(method='linear', limit_direction='both')
            
            if debug:
                print(f"    Interpolated {short_blink_frames} frames from {len(short_blink_segments)} short blink segment(s)")
        elif debug:
            print(f"\n  No short blinks to interpolate")
        
        # Mark long blinks by setting coordinates to NaN (these remain as NaN, not interpolated)
        for blink in blink_segments:
            start_idx = blink['start_idx']
            end_idx = blink['end_idx']
            video_data.loc[video_data.index[start_idx:end_idx+1], columns_of_interest] = np.nan
            total_blink_frames += blink['length']
        
        if debug:
            print(f"  Total long blink frames marked (kept as NaN): {total_blink_frames} frames "
                  f"({total_blink_frames/fps*1000:.1f}ms)")
    
    # Calculate blink bout rate
    if len(video_data) > 0:
        recording_duration_min = (video_data['Seconds'].iloc[-1] - video_data['Seconds'].iloc[0]) / 60
        blink_bout_rate = len(blink_bouts) / recording_duration_min if recording_duration_min > 0 else 0
        if debug and len(blink_segments) > 0:
            print(f"  Blink bout rate: {blink_bout_rate:.2f} blink bouts/minute")
    else:
        blink_bout_rate = 0
    
    return {
        'blink_segments': blink_segments,
        'short_blink_segments': short_blink_segments,
        'blink_bouts': blink_bouts,
        'all_blink_segments': all_blink_segments,
        'fps': fps,
        'long_blinks_warnings': long_blinks_warnings,
        'total_blink_frames': total_blink_frames,
        'blink_bout_rate': blink_bout_rate
    }


def load_manual_blinks(data_path, video_number):
    """
    Load manual blink annotations from CSV file.
    
    Parameters:
    -----------
    data_path : Path
        Path to directory containing the CSV file
    video_number : int
        Video number (1 or 2) to determine filename (Video1_manual_blinks.csv or Video2_manual_blinks.csv)
        
    Returns:
    --------
    list of dict or None : List of blink dicts with keys 'num', 'start', 'end', or None if file doesn't exist or has errors
    """
    manual_blinks_path = data_path / f"Video{video_number}_manual_blinks.csv"
    
    if not manual_blinks_path.exists():
        return None
    
    try:
        manual_blinks_df = pd.read_csv(manual_blinks_path)
        # Expected columns: blink_number, start_frame, end_frame
        if all(col in manual_blinks_df.columns for col in ['blink_number', 'start_frame', 'end_frame']):
            manual_blinks = [
                {'num': int(row['blink_number']), 'start': int(row['start_frame']), 'end': int(row['end_frame'])}
                for _, row in manual_blinks_df.iterrows()
            ]
            print(f"✅ Loaded {len(manual_blinks)} manual blinks for VideoData{video_number} from {manual_blinks_path.name}")
            return manual_blinks
        else:
            print(f"⚠️ WARNING: {manual_blinks_path.name} exists but doesn't have expected columns (blink_number, start_frame, end_frame)")
            return None
    except Exception as e:
        print(f"⚠️ WARNING: Failed to load {manual_blinks_path.name}: {e}")
        return None


def analyze_confidence_scores(video_data, score_columns, score_cutoff, video_label="", debug=False):
    """
    Analyze confidence scores and report top 3 columns with most low-confidence frames.
    
    Parameters:
    -----------
    video_data : pd.DataFrame
        Video data with score columns
    score_columns : list of str
        List of score column names to analyze
    score_cutoff : float
        Threshold below which scores are considered low
    video_label : str
        Label for the video (for printing)
    debug : bool
        Whether to print analysis (only runs if debug=True)
        
    Returns:
    --------
    list : List of tuples (col, count, pct, longest) sorted by count
    """
    if not debug:
        return []
    
    total_points = len(video_data)
    print(f'\nℹ️ {video_label} - Top 3 columns with most frames below {score_cutoff} confidence score:')
    
    stats = []
    for col in score_columns:
        if col in video_data.columns:
            count_below = (video_data[col] < score_cutoff).sum()
            pct_below = (count_below / total_points) * 100 if total_points > 0 else 0
            
            below_mask = video_data[col] < score_cutoff
            longest = 0
            run = 0
            for val in below_mask:
                if val:
                    run += 1
                    if run > longest:
                        longest = run
                else:
                    run = 0
            stats.append((col, count_below, pct_below, longest))
    
    stats.sort(key=lambda x: x[1], reverse=True)
    for i, (col, count, pct, longest) in enumerate(stats[:3]):
        print(f"{video_label} - #{i+1}: {col} | Values below {score_cutoff}: {count} ({pct:.2f}%) | Longest consecutive frame series: {longest}")
    
    return stats


def center_coordinates_to_median(video_data, columns_of_interest, video_label=""):
    """
    Center coordinates to the median pupil centre.
    
    Modifies video_data in place by subtracting median center.x and center.y from all coordinates.
    
    Parameters:
    -----------
    video_data : pd.DataFrame
        Video data with coordinate columns
    columns_of_interest : list of str
        List of coordinate column names to center
    video_label : str
        Label for the video (for printing)
        
    Returns:
    --------
    pd.DataFrame : Copy of video_data after centering
    """
    # Calculate the median of the center x and y points
    mean_center_x = video_data['center.x'].median()
    mean_center_y = video_data['center.y'].median()
    
    print(f"{video_label} - Centering on median pupil centre: \nMean center.x: {mean_center_x}, Mean center.y: {mean_center_y}")
    
    # Translate the coordinates
    for col in columns_of_interest:
        if '.x' in col:
            video_data[col] = video_data[col] - mean_center_x
        elif '.y' in col:
            video_data[col] = video_data[col] - mean_center_y
    
    return video_data.copy()


def filter_low_confidence_points(video_data, point_names, score_cutoff, video_label="", debug=False):
    """
    Filter out low-confidence points and replace with NaN.
    
    Modifies video_data in place by setting x and y coordinates to NaN where score < threshold.
    
    Parameters:
    -----------
    video_data : pd.DataFrame
        Video data with score and coordinate columns
    point_names : list of str
        List of point names (without .x, .y, .score suffix)
    score_cutoff : float
        Score threshold below which points are replaced with NaN
    video_label : str
        Label for the video (for printing)
    debug : bool
        Whether to print statistics
        
    Returns:
    --------
    dict : Dictionary with 'total_low_score', 'max_low_score_channel', 'max_low_score_count'
    """
    total_low_score = 0
    low_score_counts = {}
    
    for point in point_names:
        if f'{point}.score' in video_data.columns:
            # Find indices where score is below threshold
            low_score_mask = video_data[f'{point}.score'] < score_cutoff
            low_score_count = low_score_mask.sum()
            low_score_counts[f'{point}.x'] = low_score_count
            low_score_counts[f'{point}.y'] = low_score_count
            total_low_score += low_score_count * 2  # *2 because we're removing both x and y
            
            # Set x and y to NaN for low confidence points
            video_data.loc[low_score_mask, f'{point}.x'] = np.nan
            video_data.loc[low_score_mask, f'{point}.y'] = np.nan
    
    # Find the channel with the maximum number of low-score points
    max_low_score_channel = max(low_score_counts, key=low_score_counts.get) if low_score_counts else None
    max_low_score_count = low_score_counts[max_low_score_channel] if max_low_score_channel else 0
    
    # Print the channel with the maximum number of low-score points
    if debug:
        if max_low_score_channel:
            print(f"{video_label} - Channel with the maximum number of low-confidence points: {max_low_score_channel}, Number of low-confidence points: {max_low_score_count}")
        print(f"{video_label} - A total number of {total_low_score} low-confidence coordinate values were replaced by interpolation")
    
    return {
        'total_low_score': total_low_score,
        'max_low_score_channel': max_low_score_channel,
        'max_low_score_count': max_low_score_count
    }


def remove_outliers_and_interpolate(video_data, columns_of_interest, outlier_sd_threshold, video_label="", debug=False):
    """
    Remove outliers and interpolate NaN values.
    
    Outliers are defined as values more than outlier_sd_threshold standard deviations from the mean.
    Modifies video_data in place.
    
    Parameters:
    -----------
    video_data : pd.DataFrame
        Video data with coordinate columns
    columns_of_interest : list of str
        List of coordinate column names to process
    outlier_sd_threshold : float
        Number of standard deviations for outlier detection
    video_label : str
        Label for the video (for printing)
    debug : bool
        Whether to print statistics
        
    Returns:
    --------
    dict : Dictionary with 'total_outliers', 'max_outliers_channel', 'max_outliers_count'
    """
    # Calculate the standard deviation for each column of interest
    std_devs = {col: video_data[col].std() for col in columns_of_interest}
    
    # Calculate the number of outliers for each column
    outliers = {col: ((video_data[col] - video_data[col].mean()).abs() > outlier_sd_threshold * std_devs[col]).sum() 
                for col in columns_of_interest}
    
    # Find the channel with the maximum number of outliers
    max_outliers_channel = max(outliers, key=outliers.get) if outliers else None
    max_outliers_count = outliers[max_outliers_channel] if max_outliers_channel else 0
    total_outliers = sum(outliers.values())
    
    # Print the channel with the maximum number of outliers and the number
    if debug:
        if max_outliers_channel:
            print(f"{video_label} - Channel with the maximum number of outliers: {max_outliers_channel}, Number of outliers: {max_outliers_count}")
        print(f"{video_label} - A total number of {total_outliers} outliers were replaced by interpolation")
    
    # Replace outliers by interpolating between the previous and subsequent non-NaN value
    for col in columns_of_interest:
        outlier_indices = video_data[((video_data[col] - video_data[col].mean()).abs() > outlier_sd_threshold * std_devs[col])].index
        video_data.loc[outlier_indices, col] = np.nan
    
    # Interpolate all NaN values (returns new DataFrame, so caller needs to reassign)
    video_data_interpolated = video_data.interpolate(method='linear', limit_direction='both')
    
    return {
        'total_outliers': total_outliers,
        'max_outliers_channel': max_outliers_channel,
        'max_outliers_count': max_outliers_count,
        'video_data_interpolated': video_data_interpolated  # Return interpolated dataframe
    }


def analyze_instance_score_distribution(video_data, blink_instance_score_threshold, fps, video_label="", debug=False, plot=True):
    """
    Analyze instance score distribution and plot histogram.
    
    Parameters:
    -----------
    video_data : pd.DataFrame
        Video data with 'instance.score' column
    blink_instance_score_threshold : float
        Hard threshold for blink detection
    fps : float
        Frames per second (for time calculations)
    video_label : str
        Label for the video (for printing)
    debug : bool
        Whether to print detailed statistics
    plot : bool
        Whether to create histogram plot
        
    Returns:
    --------
    dict : Dictionary with statistics including 'percentile', 'num_low', 'pct_low', 'longest_consecutive', 'low_sections'
    """
    if not debug:
        return {}
    
    # Plot histogram
    if plot:
        plt.figure(figsize=(6, 5))
        plt.hist(video_data['instance.score'].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.axvline(blink_instance_score_threshold, color='red', linestyle='--', linewidth=2, 
                    label=f'Hard threshold = {blink_instance_score_threshold}')
        plt.yscale('log')
        plt.title(f"Distribution of instance.score ({video_label})")
        plt.xlabel("instance.score")
        plt.ylabel("Frequency (log scale)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Calculate statistics
    percentile = (video_data['instance.score'] < blink_instance_score_threshold).sum() / len(video_data) * 100
    num_low = (video_data['instance.score'] < blink_instance_score_threshold).sum()
    total = len(video_data)
    pct_low = (num_low / total) * 100
    
    # Find longest consecutive segments
    low_sections = find_longest_lowscore_sections(video_data['instance.score'], blink_instance_score_threshold, top_n=1)
    longest_consecutive = low_sections[0]['length'] if low_sections else 0
    longest_consecutive_ms = (longest_consecutive / fps) * 1000 if fps and longest_consecutive > 0 else None
    
    # Always print key stats
    print(f"\n{video_label} - Instance Score Threshold Analysis:")
    print(f"  Hard threshold: {blink_instance_score_threshold}")
    print(f"  Frames below threshold: {num_low} / {total} ({pct_low:.2f}%)")
    print(f"  Longest consecutive segment: {longest_consecutive} frames", end="")
    if longest_consecutive_ms:
        print(f" ({longest_consecutive_ms:.1f}ms)")
    else:
        print()
    
    # Detailed stats only in debug mode
    if debug:
        print(f"\n  Detailed statistics:")
        print(f"  Percentile: {percentile:.2f}% (i.e., {percentile:.2f}% of frames have instance.score < {blink_instance_score_threshold})")
        
        # Report the top 5 longest consecutive sections
        low_sections_detailed = find_longest_lowscore_sections(video_data['instance.score'], blink_instance_score_threshold, top_n=5)
        if len(low_sections_detailed) > 0:
            print(f"\n  Top 5 longest consecutive sections where instance.score < threshold:")
            for i, sec in enumerate(low_sections_detailed, 1):
                start_idx = sec['start_idx']
                end_idx = sec['end_idx']
                sec_duration_ms = (sec['length'] / fps) * 1000 if fps else None
                if sec_duration_ms:
                    print(f"    Section {i}: index {start_idx}-{end_idx} (length {sec['length']} frames, {sec_duration_ms:.1f}ms)")
                else:
                    print(f"    Section {i}: index {start_idx}-{end_idx} (length {sec['length']} frames)")
    
    return {
        'percentile': percentile,
        'num_low': num_low,
        'pct_low': pct_low,
        'longest_consecutive': longest_consecutive,
        'longest_consecutive_ms': longest_consecutive_ms,
        'low_sections': low_sections
    }


def plot_instance_score_distributions_combined(video_data1, video_data2, blink_instance_score_threshold, 
                                               has_v1=False, has_v2=False):
    """
    Plot combined histograms for instance.score distributions (both videos in one figure).
    
    Parameters:
    -----------
    video_data1 : pd.DataFrame or None
        Video data 1 with 'instance.score' column
    video_data2 : pd.DataFrame or None
        Video data 2 with 'instance.score' column
    blink_instance_score_threshold : float
        Hard threshold for blink detection
    has_v1 : bool
        Whether video_data1 exists and should be plotted
    has_v2 : bool
        Whether video_data2 exists and should be plotted
    """
    if not (has_v1 or has_v2):
        return
    
    plt.figure(figsize=(12, 5))
    plot_index = 1
    
    if has_v1:
        plt.subplot(1, 2 if has_v2 else 1, plot_index)
        plt.hist(video_data1['instance.score'].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.axvline(blink_instance_score_threshold, color='red', linestyle='--', linewidth=2, 
                    label=f'Hard threshold = {blink_instance_score_threshold}')
        plt.yscale('log')
        plt.title("Distribution of instance.score (VideoData1)")
        plt.xlabel("instance.score")
        plt.ylabel("Frequency (log scale)")
        plt.legend()
        plot_index += 1
    
    if has_v2:
        plt.subplot(1, 2 if has_v1 else 1, plot_index)
        plt.hist(video_data2['instance.score'].dropna(), bins=30, color='salmon', edgecolor='black')
        plt.axvline(blink_instance_score_threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Hard threshold = {blink_instance_score_threshold}')
        plt.yscale('log')
        plt.title("Distribution of instance.score (VideoData2)")
        plt.xlabel("instance.score")
        plt.ylabel("Frequency (log scale)")
        plt.legend()
        plot_index += 1
    
    plt.tight_layout()
    plt.show()

