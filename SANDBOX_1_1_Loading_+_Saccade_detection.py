# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: aeon
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Setup

# %%
import numpy as np
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import gc
import io
from contextlib import redirect_stdout

from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from fastkde.fastKDE import fastKDE
from scipy.stats import linregress
from scipy.signal import butter, filtfilt
from scipy.signal import correlate
from scipy.stats import pearsonr
from scipy.signal import find_peaks

from harp_resources import process, utils
from sleap import load_and_process as lp
from sleap import processing_functions as pf
from sleap import saccade_processing as sp
from sleap.saccade_processing import analyze_eye_video_saccades

# Reload modules to pick up latest changes (useful after code updates)
# Set force_reload_modules = True to always reload, or False to use cached versions
force_reload_modules = True  # Set to False for faster execution when modules haven't changed
if force_reload_modules:
    import importlib
    import sleap.load_and_process
    import sleap.processing_functions
    import sleap.saccade_processing
    importlib.reload(sleap.load_and_process)
    importlib.reload(sleap.processing_functions)
    importlib.reload(sleap.saccade_processing)
    # Re-import aliases after reload
    lp = sleap.load_and_process
    pf = sleap.processing_functions
    sp = sleap.saccade_processing
    from sleap.saccade_processing import analyze_eye_video_saccades

def get_eye_label(key):
    """Return mapped user-viewable eye label for video key."""
    return VIDEO_LABELS.get(key, key)

NaNs_removed = False # keep as false here, it is to checking if NaNs already removed if the notebook cell is rerun


# symbols to use ✅ ℹ️ ⚠️ ❗

# %%
# set up variables and load data 
############################################################################################################

# User-editable friendly labels for plotting and console output:

debug = True  # Set to True to enable debug output across all cells (file loading, processing, etc.)

video1_eye = 'L'  # Options: 'L' or 'R'; which eye does VideoData1 represent? ('L' = Left, 'R' = Right)
plot_QC_timeseries = False
score_cutoff = 0.2 # for filtering out inferred points with low confidence, they get interpolated 
outlier_sd_threshold = 10 # for removing outliers from the data, they get interpolated 

# Pupil diameter filter settings (Butterworth low-pass)
pupil_filter_cutoff_hz = 10  # Hz
pupil_filter_order = 6

# Parameters for blink detection
min_blink_duration_ms = 50  # minimum blink duration in milliseconds
blink_merge_window_ms = 100  # NOT CURRENTLY USED: merge window was removed to preserve good data between separate blinks
long_blink_warning_ms = 2000  # warn if blinks exceed this duration (in ms) - user should verify these are real blinks
blink_instance_score_threshold = 3.8  # hard threshold for blink detection - frames with instance.score below this value are considered blinks, calculated as 9 pupil points *0.2 + left/right as 1   

# for saccades
refractory_period = 0.1  # sec
## Separate adaptive saccade threshold (k) for each video:
k1 = 4  # for VideoData1 (L)
k2 = 5  # for VideoData2 (R)

# for adaptive saccade threshold - Number of standard deviations (adjustable: 2-4 range works well) 
onset_offset_fraction = 0.2  # to determine saccade onset and offset, i.e. o.2 is 20% of the peak velocity
n_before = 10  # Number of points before detection peak to extract for peri-saccade-segments, points, so independent of FPS 
n_after = 30   # Number of points after detection peak to extract

# Additional saccade detection parameters
baseline_n_points = 5  # Number of points before threshold crossing to use for baseline calculation
saccade_smoothing_window = 5  # Rolling median window size for position smoothing (frames)
saccade_peak_width = 1  # Minimum peak width in samples for find_peaks (frames)

plot_saccade_detection_QC = True

# Parameters for orienting vs compensatory saccade classification
classify_orienting_compensatory = True  # Set to True to classify saccades as orienting vs compensatory
bout_window = 1.5  # Time window (seconds) for grouping saccades into bouts
pre_saccade_window = 0.3  # Time window (seconds) before saccade onset to analyze
max_intersaccade_interval_for_classification = 5.0  # Maximum time (seconds) to extend post-saccade window until next saccade for classification
pre_saccade_velocity_threshold = 50.0  # Velocity threshold (px/s) for detecting pre-saccade drift
pre_saccade_drift_threshold = 10.0  # Position drift threshold (px) before saccade for compensatory classification
post_saccade_variance_threshold = 100.0  # Position variance threshold (px²) after saccade for orienting classification
post_saccade_position_change_threshold_percent = 50.0  # Position change threshold (% of saccade amplitude) - if post-saccade change > amplitude * this%, classify as compensatory

# Adaptive threshold parameters (percentile-based)
use_adaptive_thresholds = True  # Set to True to use adaptive thresholds based on feature distributions, False to use fixed thresholds
adaptive_percentile_pre_velocity = 75  # Percentile for pre-saccade velocity threshold (upper percentile for compensatory detection)
adaptive_percentile_pre_drift = 75  # Percentile for pre-saccade drift threshold (upper percentile for compensatory detection)
adaptive_percentile_post_variance = 25  # Percentile for post-saccade variance threshold (lower percentile for orienting detection - low variance = stable)

"""
CLASSIFICATION PARAMETERS EXPLANATION:
======================================

1. bout_window (1.5 seconds)
   - Purpose: Groups saccades that occur within this time window into "bouts"
   - How it works: If two saccades occur within 1.5s of each other, they're grouped into the same bout
   - Reasoning: Compensatory saccades occur in rapid succession during head rotation
   - Typical values: 1.0-2.0 seconds (adjust based on your data)

2. pre_saccade_window (0.3 seconds)
   - Purpose: Time window BEFORE each saccade onset to analyze for drift/stability
   - How it works: Measures mean velocity and position drift in the 0.3s before saccade starts
     * IMPORTANT: The window is constrained to not extend before the peak time of the previous saccade
     * This ensures we only measure the inter-saccade interval, not the period during the previous saccade
   - Reasoning: 
     * Compensatory: Eye drifts slowly before saccade (compensating for head rotation)
     * Orienting: Eye is stable before saccade (at rest before quick shift)
   - Typical values: 0.2-0.5 seconds (should capture pre-saccade behavior)

3. max_intersaccade_interval_for_classification (5.0 seconds)
   - Purpose: Maximum time to extend post-saccade window until the next saccade occurs
   - How it works: 
     * For each saccade, the post-saccade window dynamically extends until the next saccade starts
     * If the next saccade occurs within max_intersaccade_interval_for_classification, use that interval
     * If no next saccade occurs within this time, cap at max_intersaccade_interval_for_classification
     * Measures position change/variance in this dynamic window
   - Reasoning:
     * Compensatory saccades: Eye position continues to change until next compensatory saccade (large position change)
     * Orienting saccades: Eye settles at new position and stays stable (small position change)
   - Typical values: 3-10 seconds (should capture behavior until next saccade or reasonable maximum)
   - This parameter is key for distinguishing compensatory vs orienting based on eye position stability

5. pre_saccade_velocity_threshold (50.0 px/s)
   - Purpose: Threshold for mean absolute velocity in pre-saccade window
   - How it works: If mean(|velocity|) > threshold → evidence of drift (compensatory)
   - Reasoning: Compensatory saccades follow slow drift (velocity > 0)
   - Typical values: 30-100 px/s (adjust based on your sampling rate and noise level)
   - Too low: May misclassify orienting saccades with noise as compensatory
   - Too high: May miss true compensatory drift

6. pre_saccade_drift_threshold (10.0 px)
   - Purpose: Threshold for position change in pre-saccade window
   - How it works: If |position_end - position_start| > threshold → evidence of drift (compensatory)
   - Reasoning: Compensatory saccades follow slow position drift (eye moves slowly)
   - Typical values: 5-20 px (adjust based on your typical saccade amplitudes)
   - Too low: May misclassify orienting saccades with small movements as compensatory
   - Too high: May miss true compensatory drift

7. post_saccade_variance_threshold (100.0 px²)
   - Purpose: Threshold for position variance in post-saccade window
   - How it works: If variance < threshold → stable position (orienting)
   - Reasoning: Orienting saccades settle at new stable position (low variance)
   - Typical values: 50-200 px² (adjust based on your noise level)
   - Too low: May misclassify orienting saccades as compensatory
   - Too high: May misclassify compensatory saccades as orienting
   - Note: Less reliable for saccades in bouts (next saccade may occur soon after)

8. post_saccade_position_change_threshold_percent (50.0%)
   - Purpose: Threshold for position change in dynamic post-saccade window, expressed as percentage of saccade amplitude
   - How it works: 
     * Calculates total position change from end of saccade until next saccade (or max interval)
     * Compares to saccade amplitude: if post_change > amplitude * threshold_percent → compensatory
     * Compensatory saccades: Eye continues moving after saccade (large position change relative to amplitude)
     * Orienting saccades: Eye settles at new position (small position change relative to amplitude)
   - Reasoning: 
     * Compensatory: Eye position continues changing until next compensatory saccade (change comparable to or larger than saccade amplitude)
     * Orienting: Eye settles and stays stable (change much smaller than saccade amplitude)
   - Typical values: 30-70% (adjust based on your data)
   - Too low: May misclassify orienting saccades with small noise as compensatory
   - Too high: May miss true compensatory drift patterns
   - Key advantage: Relative to amplitude, so adapts to different saccade sizes

PRACTICAL TUNING GUIDE (For QC Adjustments):
=============================================
When you find a miscategorized saccade during QC, here's what to adjust:

⚠️ PROBLEM: Two orienting saccades close together are misclassified as compensatory
   → SOLUTION: Decrease bout_window (e.g., 1.5 → 1.0 seconds)
   → WHY: Makes the classifier less likely to group saccades into bouts

⚠️ PROBLEM: Compensatory saccades misclassified as orienting (happens in bouts)
   → SOLUTION: Increase bout_window (e.g., 1.5 → 2.0 seconds)
   → WHY: Better captures rapid compensatory saccade sequences

⚠️ PROBLEM: Compensatory saccades misclassified as orienting (isolated saccades)
   → SOLUTION: Decrease pre_saccade_velocity_threshold or pre_saccade_drift_threshold
   → WHY: Makes it easier to detect slow pre-saccade drift that indicates compensation
   → OR: Increase post_saccade_position_change_threshold_percent (e.g., 50 → 70)
   → WHY: Requires larger post-saccade position change to classify as compensatory

⚠️ PROBLEM: Orienting saccades misclassified as compensatory (too sensitive to drift)
   → SOLUTION: Increase pre_saccade_velocity_threshold or pre_saccade_drift_threshold
   → WHY: Requires stronger evidence of drift before classifying as compensatory
   → OR: Decrease post_saccade_position_change_threshold_percent (e.g., 50 → 30)
   → WHY: Makes it harder to classify based on post-saccade position change

⚠️ PROBLEM: Eye position continues moving after orienting saccades (causes misclassification)
   → SOLUTION: Increase max_intersaccade_interval_for_classification (e.g., 5.0 → 7.0 seconds)
   → WHY: Gives more time for eye to settle, better captures true stability

⚠️ PROBLEM: Using adaptive thresholds but getting inconsistent results
   → SOLUTION: Set use_adaptive_thresholds = False and use fixed thresholds
   → WHY: Fixed thresholds give more predictable, reproducible results

📝 QUICK REFERENCE:
   - bout_window: Controls how close saccades need to be to form a bout (default: 1.5s)
   - pre_saccade_velocity_threshold: How fast eye moves before saccade = compensatory (default: 50 px/s)
   - pre_saccade_drift_threshold: How much position changes before saccade = compensatory (default: 10 px)
   - post_saccade_position_change_threshold_percent: % of saccade amplitude as position change threshold (default: 50%)
   - max_intersaccade_interval_for_classification: Max time to look for next saccade (default: 5.0s)

CLASSIFICATION ORDER AND LOGIC:
================================

Stage 1: Temporal Clustering (Bout Detection)
  - Groups saccades within bout_window into bouts
  - Saccades > bout_window apart start a new bout
  - Result: Each saccade gets a bout_id and bout_size

Stage 2: Feature Extraction
  For each saccade:
    a) Extract pre-saccade features (pre_saccade_window before onset):
       - Mean absolute velocity (pre_saccade_mean_velocity)
       - Position drift (pre_saccade_position_drift)
    b) Extract post-saccade features (DYNAMIC window):
       - Window extends from saccade offset until next saccade start (or max_intersaccade_interval_for_classification)
       - Position variance (post_saccade_position_variance) - measures stability until next saccade
       - Position change (post_saccade_position_change) - total change in position until next saccade
       - This captures whether eye settles (orienting) or continues changing (compensatory)

Stage 3: Classification (ORIGINAL SIMPLE LOGIC - Conservative Starting Point)
  Rule 1: If in a bout (bout_size >= 2)
    - Automatically classify as compensatory
    - Rationale: Compensatory saccades occur in rapid succession during head rotation
  
  Rule 2: If isolated (bout_size == 1)
    - If pre_vel > pre_saccade_velocity_threshold OR pre_drift > pre_saccade_drift_threshold
      → Compensatory (evidence of drift/compensation before saccade)
    - Else if pre_vel ≤ pre_saccade_velocity_threshold AND post_var < post_saccade_variance_threshold
      → Orienting (stable before and stable after saccade)
    - Else
      → Compensatory (conservative default - when uncertain)

RATIONALE FOR ORDER:
====================
1. Temporal clustering first: Identifies which saccades are potentially related (bouts)
2. Feature extraction: Measures key characteristics of each saccade
3. Classification using simple rules:
   - Bouts: Automatically compensatory (conservative - assumes bouts indicate compensatory behavior)
   - Isolated: Feature-based classification (pre-saccade drift + post-saccade stability)

KEY INSIGHT:
============
- Compensatory saccades: Show DRIFT before them (slow compensation for head rotation)
- Orienting saccades: Show STABLE periods before them (eye at rest before quick shift)
- The pre-saccade window captures the inter-saccade interval, which is the key discriminator

NOTE ON CURRENT LOGIC:
======================
This is the original conservative starting point:
- All saccades in bouts (>=2 saccades within bout_window) are classified as compensatory
- Only isolated saccades are classified using features
- This may over-classify compensatory (especially if two orienting saccades happen close together)
- Adjust bout_window or add refinement logic if needed
"""

video2_eye = 'R' if video1_eye == 'L' else 'L' # Automatically assign eye for VideoData2
eye_fullname = {'L': 'Left', 'R': 'Right'} # Map for full names (used in labels)
# Update VIDEO_LABELS based on selection
VIDEO_LABELS = {
    'VideoData1': f"VideoData1 ({video1_eye}: {eye_fullname[video1_eye]})",
    'VideoData2': f"VideoData2 ({video2_eye}: {eye_fullname[video2_eye]})"
}

data_path = Path('/Users/rancze/Documents/Data/vestVR/20250409_Cohort3_rotation/Visual_mismatch_day4/B6J2782-2025-04-28T14-22-03') 
#data_path = Path('/Users/rancze/Documents/Data/vestVR/Cohort1/No_iso_correction/Visual_mismatch_day3/B6J2717-2024-12-10T12-17-03') # only has sleap data 1
save_path = data_path.parent / f"{data_path.name}_processedData"

VideoData1, VideoData2, VideoData1_Has_Sleap, VideoData2_Has_Sleap = lp.load_videography_data(data_path, debug=debug)

# Load manual blink data if available
manual_blinks_v1 = None
manual_blinks_v2 = None

manual_blinks_v1_path = data_path / "Video1_manual_blinks.csv"
manual_blinks_v2_path = data_path / "Video2_manual_blinks.csv"

if manual_blinks_v1_path.exists():
    try:
        manual_blinks_df_v1 = pd.read_csv(manual_blinks_v1_path)
        # Expected columns: blink_number, start_frame, end_frame
        if all(col in manual_blinks_df_v1.columns for col in ['blink_number', 'start_frame', 'end_frame']):
            manual_blinks_v1 = [
                {'num': int(row['blink_number']), 'start': int(row['start_frame']), 'end': int(row['end_frame'])}
                for _, row in manual_blinks_df_v1.iterrows()
            ]
            print(f"✅ Loaded {len(manual_blinks_v1)} manual blinks for VideoData1 from {manual_blinks_v1_path.name}")
        else:
            print(f"⚠️ WARNING: {manual_blinks_v1_path.name} exists but doesn't have expected columns (blink_number, start_frame, end_frame)")
    except Exception as e:
        print(f"⚠️ WARNING: Failed to load {manual_blinks_v1_path.name}: {e}")

if manual_blinks_v2_path.exists():
    try:
        manual_blinks_df_v2 = pd.read_csv(manual_blinks_v2_path)
        # Expected columns: blink_number, start_frame, end_frame
        if all(col in manual_blinks_df_v2.columns for col in ['blink_number', 'start_frame', 'end_frame']):
            manual_blinks_v2 = [
                {'num': int(row['blink_number']), 'start': int(row['start_frame']), 'end': int(row['end_frame'])}
                for _, row in manual_blinks_df_v2.iterrows()
            ]
            print(f"✅ Loaded {len(manual_blinks_v2)} manual blinks for VideoData2 from {manual_blinks_v2_path.name}")
        else:
            print(f"⚠️ WARNING: {manual_blinks_v2_path.name} exists but doesn't have expected columns (blink_number, start_frame, end_frame)")
    except Exception as e:
        print(f"⚠️ WARNING: Failed to load {manual_blinks_v2_path.name}: {e}")

columns_of_interest = ['left.x','left.y','center.x','center.y','right.x','right.y','p1.x','p1.y','p2.x','p2.y','p3.x','p3.y','p4.x','p4.y','p5.x','p5.y','p6.x','p6.y','p7.x','p7.y','p8.x','p8.y']

if VideoData1_Has_Sleap:
    VideoData1 = VideoData1.drop(columns=['track']) # drop the track column as it is empty
    coordinates_dict1_raw=lp.get_coordinates_dict(VideoData1, columns_of_interest)
    FPS_1 = 1 / VideoData1["Seconds"].diff().mean()  # frame rate for VideoData1 TODO where to save it, is it useful?
    print ()
    print(f"{get_eye_label('VideoData1')}: FPS = {FPS_1}")

if VideoData2_Has_Sleap:
    VideoData2 = VideoData2.drop(columns=['track']) # drop the track column as it is empty
    coordinates_dict2_raw=lp.get_coordinates_dict(VideoData2, columns_of_interest)
    FPS_2 = 1 / VideoData2["Seconds"].diff().mean()  # frame rate for VideoData2
    print(f"{get_eye_label('VideoData2')}: FPS = {FPS_2}")


# %%
# plot timeseries of coordinates in browser for both VideoData1 and VideoData2
############################################################################################################
if plot_QC_timeseries:
    print(f'⚠️ Check for long discontinuities and outliers in the data, we will try to deal with them later')
    print(f'ℹ️ Figures open in browser window, takes a bit of time.')

    # Helper list variables
    subplot_titles = (
        "X coordinates for pupil centre and left-right eye corner",
        "Y coordinates for pupil centre and left-right eye corner",
        "X coordinates for iris points",
        "Y coordinates for iris points"
    )
    eye_x = ['left.x', 'center.x', 'right.x']
    eye_y = ['left.y', 'center.y', 'right.y']
    iris_x = ['p1.x', 'p2.x', 'p3.x', 'p4.x', 'p5.x', 'p6.x', 'p7.x', 'p8.x']
    iris_y = ['p1.y', 'p2.y', 'p3.y', 'p4.y', 'p5.y', 'p6.y', 'p7.y', 'p8.y']

    # --- VideoData1 ---
    if VideoData1_Has_Sleap:
        fig1 = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles
        )

        # Row 1: left.x, center.x, right.x
        for col in eye_x:
            fig1.add_trace(go.Scatter(x=VideoData1['Seconds'], y=VideoData1[col], mode='lines', name=col), row=1, col=1)
        # Row 2: left.y, center.y, right.y
        for col in eye_y:
            fig1.add_trace(go.Scatter(x=VideoData1['Seconds'], y=VideoData1[col], mode='lines', name=col), row=2, col=1)
        # Row 3: p1.x ... p8.x
        for col in iris_x:
            fig1.add_trace(go.Scatter(x=VideoData1['Seconds'], y=VideoData1[col], mode='lines', name=col), row=3, col=1)
        # Row 4: p1.y ... p8.y
        for col in iris_y:
            fig1.add_trace(go.Scatter(x=VideoData1['Seconds'], y=VideoData1[col], mode='lines', name=col), row=4, col=1)

        fig1.update_layout(
            height=1200,
            title_text=f"Time series subplots for coordinates [{get_eye_label('VideoData1')}]",
            showlegend=True
        )
        fig1.update_xaxes(title_text="Seconds", row=4, col=1)
        fig1.update_yaxes(title_text="X Position", row=1, col=1)
        fig1.update_yaxes(title_text="Y Position", row=2, col=1)
        fig1.update_yaxes(title_text="X Position", row=3, col=1)
        fig1.update_yaxes(title_text="Y Position", row=4, col=1)

        fig1.show(renderer='browser')

    # --- VideoData2 ---
    if VideoData2_Has_Sleap:
        fig2 = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles
        )
        # Row 1: left.x, center.x, right.x
        for col in eye_x:
            fig2.add_trace(go.Scatter(x=VideoData2['Seconds'], y=VideoData2[col], mode='lines', name=col), row=1, col=1)
        # Row 2: left.y, center.y, right.y
        for col in eye_y:
            fig2.add_trace(go.Scatter(x=VideoData2['Seconds'], y=VideoData2[col], mode='lines', name=col), row=2, col=1)
        # Row 3: p1.x ... p8.x
        for col in iris_x:
            fig2.add_trace(go.Scatter(x=VideoData2['Seconds'], y=VideoData2[col], mode='lines', name=col), row=3, col=1)
        # Row 4: p1.y ... p8.y
        for col in iris_y:
            fig2.add_trace(go.Scatter(x=VideoData2['Seconds'], y=VideoData2[col], mode='lines', name=col), row=4, col=1)

        fig2.update_layout(
            height=1200,
            title_text=f"Time series subplots for coordinates [{get_eye_label('VideoData2')}]",
            showlegend=True
        )
        fig2.update_xaxes(title_text="Seconds", row=4, col=1)
        fig2.update_yaxes(title_text="X Position", row=1, col=1)
        fig2.update_yaxes(title_text="Y Position", row=2, col=1)
        fig2.update_yaxes(title_text="X Position", row=3, col=1)
        fig2.update_yaxes(title_text="Y Position", row=4, col=1)

        fig2.show(renderer='browser')

# %%
# QC plot XY coordinate distributions to visualize outliers 
############################################################################################################

columns_of_interest = ['left', 'right', 'center', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']

# Filter out NaN values and calculate the min and max values for X and Y coordinates for both dict1 and dict2

def min_max_dict(coordinates_dict):
    x_min = min([coordinates_dict[f'{col}.x'][~np.isnan(coordinates_dict[f'{col}.x'])].min() for col in columns_of_interest])
    x_max = max([coordinates_dict[f'{col}.x'][~np.isnan(coordinates_dict[f'{col}.x'])].max() for col in columns_of_interest])
    y_min = min([coordinates_dict[f'{col}.y'][~np.isnan(coordinates_dict[f'{col}.y'])].min() for col in columns_of_interest])
    y_max = max([coordinates_dict[f'{col}.y'][~np.isnan(coordinates_dict[f'{col}.y'])].max() for col in columns_of_interest])
    return x_min, x_max, y_min, y_max

# Only plot panels for 1 and 2 if VideoData1_Has_Sleap and/or VideoData2_Has_Sleap are true

# Compute min/max as before for global axes limits
if VideoData1_Has_Sleap:
    x_min1, x_max1, y_min1, y_max1 = pf.min_max_dict(coordinates_dict1_raw, columns_of_interest)
if VideoData2_Has_Sleap:
    x_min2, x_max2, y_min2, y_max2 = pf.min_max_dict(coordinates_dict2_raw, columns_of_interest)

# Use global min and max for consistency only if both VideoData1_Has_Sleap and VideoData2_Has_Sleap are True
if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    x_min = min(x_min1, x_min2)
    x_max = max(x_max1, x_max2)
    y_min = min(y_min1, y_min2)
    y_max = max(y_max1, y_max2)
elif VideoData1_Has_Sleap:
    x_min, x_max, y_min, y_max = x_min1, x_max1, y_min1, y_max1
elif VideoData2_Has_Sleap:
    x_min, x_max, y_min, y_max = x_min2, x_max2, y_min2, y_max2
else:
    raise ValueError("Neither VideoData1 nor VideoData2 has Sleap data available.")

# Create the figure and axes
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
fig.suptitle(
    f"XY coordinate distribution of different points for {get_eye_label('VideoData1')} and {get_eye_label('VideoData2')} before outlier removal and NaN interpolation", 
    fontsize=14
)

# Define colormap for p1-p8
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']

# Panel 1: left, right, center (dict1)
if 'VideoData1_Has_Sleap' in globals() and VideoData1_Has_Sleap:
    ax[0, 0].set_title(f"{get_eye_label('VideoData1')}: left, right, center")
    ax[0, 0].scatter(coordinates_dict1_raw['left.x'], coordinates_dict1_raw['left.y'], color='black', label='left', s=10)
    ax[0, 0].scatter(coordinates_dict1_raw['right.x'], coordinates_dict1_raw['right.y'], color='grey', label='right', s=10)
    ax[0, 0].scatter(coordinates_dict1_raw['center.x'], coordinates_dict1_raw['center.y'], color='red', label='center', s=10)
    ax[0, 0].set_xlim([x_min, x_max])
    ax[0, 0].set_ylim([y_min, y_max])
    ax[0, 0].set_xlabel('x coordinates (pixels)')
    ax[0, 0].set_ylabel('y coordinates (pixels)')
    ax[0, 0].legend(loc='upper right')
else:
    ax[0, 0].axis('off')

# Panel 2: p1 to p8 (dict1)
if 'VideoData1_Has_Sleap' in globals() and VideoData1_Has_Sleap:
    ax[0, 1].set_title(f"{get_eye_label('VideoData1')}: p1 to p8")
    for idx, col in enumerate(columns_of_interest[3:]):
        ax[0, 1].scatter(coordinates_dict1_raw[f'{col}.x'], coordinates_dict1_raw[f'{col}.y'], color=colors[idx], label=col, s=5)
    ax[0, 1].set_xlim([x_min, x_max])
    ax[0, 1].set_ylim([y_min, y_max])
    ax[0, 1].set_xlabel('x coordinates (pixels)')
    ax[0, 1].set_ylabel('y coordinates (pixels)')
    ax[0, 1].legend(loc='upper right')
else:
    ax[0, 1].axis('off')

# Panel 3: left, right, center (dict2)
if 'VideoData2_Has_Sleap' in globals() and VideoData2_Has_Sleap:
    ax[1, 0].set_title(f"{get_eye_label('VideoData2')}: left, right, center")
    ax[1, 0].scatter(coordinates_dict2_raw['left.x'], coordinates_dict2_raw['left.y'], color='black', label='left', s=10)
    ax[1, 0].scatter(coordinates_dict2_raw['right.x'], coordinates_dict2_raw['right.y'], color='grey', label='right', s=10)
    ax[1, 0].scatter(coordinates_dict2_raw['center.x'], coordinates_dict2_raw['center.y'], color='red', label='center', s=10)
    ax[1, 0].set_xlim([x_min, x_max])
    ax[1, 0].set_ylim([y_min, y_max])
    ax[1, 0].set_xlabel('x coordinates (pixels)')
    ax[1, 0].set_ylabel('y coordinates (pixels)')
    ax[1, 0].legend(loc='upper right')
else:
    ax[1, 0].axis('off')

# Panel 4: p1 to p8 (dict2)
if 'VideoData2_Has_Sleap' in globals() and VideoData2_Has_Sleap:
    ax[1, 1].set_title(f"{get_eye_label('VideoData2')}: p1 to p8")
    for idx, col in enumerate(columns_of_interest[3:]):
        ax[1, 1].scatter(coordinates_dict2_raw[f'{col}.x'], coordinates_dict2_raw[f'{col}.y'], color=colors[idx], label=col, s=5)
    ax[1, 1].set_xlim([x_min, x_max])
    ax[1, 1].set_ylim([y_min, y_max])
    ax[1, 1].set_xlabel('x coordinates (pixels)')
    ax[1, 1].set_ylabel('y coordinates (pixels)')
    ax[1, 1].legend(loc='upper right')
else:
    ax[1, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()


# %%
# Center coordinates, filter low-confidence points, remove outliers, and interpolate
############################################################################################################

# Detect and print confidence scores analysis (runs before any filtering)
#########

if not debug:
    print("ℹ️ Debug output suppressed. Set debug=True to see detailed confidence score analysis.")

score_columns = ['left.score','center.score','right.score','p1.score','p2.score','p3.score','p4.score','p5.score','p6.score','p7.score','p8.score']

# VideoData1 confidence score analysis
if debug and 'VideoData1_Has_Sleap' in globals() and VideoData1_Has_Sleap:
    total_points1 = len(VideoData1)
    print(f'\nℹ️ VideoData1 - Top 3 columns with most frames below {score_cutoff} confidence score:')

    video1_stats = []
    for col in score_columns:
        if col in VideoData1.columns:
            count_below = (VideoData1[col] < score_cutoff).sum()
            pct_below = (count_below / total_points1) * 100 if total_points1 > 0 else 0

            below_mask = VideoData1[col] < score_cutoff
            longest = 0
            run = 0
            for val in below_mask:
                if val:
                    run += 1
                    if run > longest:
                        longest = run
                else:
                    run = 0
            video1_stats.append((col, count_below, pct_below, longest))

    video1_stats.sort(key=lambda x: x[1], reverse=True)
    for i, (col, count, pct, longest) in enumerate(video1_stats[:3]):
        print(f"VideoData1 - #{i+1}: {col} | Values below {score_cutoff}: {count} ({pct:.2f}%) | Longest consecutive frame series: {longest}")

# VideoData2 confidence score analysis
if debug and 'VideoData2_Has_Sleap' in globals() and VideoData2_Has_Sleap:
    total_points2 = len(VideoData2)
    print(f'\nℹ️ VideoData2 - Top 3 columns with most frames below {score_cutoff} confidence score:')

    video2_stats = []
    for col in score_columns:
        if col in VideoData2.columns:
            count_below = (VideoData2[col] < score_cutoff).sum()
            pct_below = (count_below / total_points2) * 100 if total_points2 > 0 else 0

            below_mask = VideoData2[col] < score_cutoff
            longest = 0
            run = 0
            for val in below_mask:
                if val:
                    run += 1
                    if run > longest:
                        longest = run
                else:
                    run = 0
            video2_stats.append((col, count_below, pct_below, longest))

    video2_stats.sort(key=lambda x: x[1], reverse=True)
    for i, (col, count, pct, longest) in enumerate(video2_stats[:3]):
        print(f"VideoData2 - #{i+1}: {col} | Values below {score_cutoff}: {count} ({pct:.2f}%) | Longest consecutive frame series: {longest}")


## Center coordinates to the median pupil centre
columns_of_interest = ['left.x','left.y','center.x','center.y','right.x','right.y','p1.x','p1.y','p2.x','p2.y','p3.x','p3.y','p4.x','p4.y','p5.x','p5.y','p6.x','p6.y','p7.x','p7.y','p8.x','p8.y']

print ()
print("=== Centering coordinates to the median pupil centre ===")
# VideoData1 processing
if 'VideoData1_Has_Sleap' in globals() and VideoData1_Has_Sleap:
    # Calculate the mean of the center x and y points
    mean_center_x1 = VideoData1['center.x'].median()
    mean_center_y1 = VideoData1['center.y'].median()

    print(f"{get_eye_label('VideoData1')} - Centering on median pupil centre: \nMean center.x: {mean_center_x1}, Mean center.y: {mean_center_y1}")

    # Translate the coordinates
    for col in columns_of_interest:
        if '.x' in col:
            VideoData1[col] = VideoData1[col] - mean_center_x1
        elif '.y' in col:
            VideoData1[col] = VideoData1[col] - mean_center_y1

    VideoData1_centered = VideoData1.copy()

# VideoData2 processing
if 'VideoData2_Has_Sleap' in globals() and VideoData2_Has_Sleap:
    # Calculate the mean of the center x and y points
    mean_center_x2 = VideoData2['center.x'].median()
    mean_center_y2 = VideoData2['center.y'].median()

    print(f"{get_eye_label('VideoData2')} - Centering on median pupil centre: \nMean center.x: {mean_center_x2}, Mean center.y: {mean_center_y2}")

    # Translate the coordinates
    for col in columns_of_interest:
        if '.x' in col:
            VideoData2[col] = VideoData2[col] - mean_center_x2
        elif '.y' in col:
            VideoData2[col] = VideoData2[col] - mean_center_y2

    VideoData2_centered = VideoData2.copy()

############################################################################################################
# remove low confidence points (score < threshold)
############################################################################################################
if not NaNs_removed:
    if debug:
        print("\n=== Score-based Filtering - point scores below threshold are replaced by interpolation ===")
        print(f"Score threshold: {score_cutoff}")
    # List of point names (without .x, .y, .score)
    point_names = ['left', 'right', 'center', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']

    # VideoData1 score-based filtering
    if 'VideoData1_Has_Sleap' in globals() and VideoData1_Has_Sleap:
        total_low_score1 = 0
        low_score_counts1 = {}
        for point in point_names:
            if f'{point}.score' in VideoData1.columns:
                # Find indices where score is below threshold
                low_score_mask = VideoData1[f'{point}.score'] < score_cutoff
                low_score_count = low_score_mask.sum()
                low_score_counts1[f'{point}.x'] = low_score_count
                low_score_counts1[f'{point}.y'] = low_score_count
                total_low_score1 += low_score_count * 2  # *2 because we're removing both x and y
                
                # Set x and y to NaN for low confidence points
                VideoData1.loc[low_score_mask, f'{point}.x'] = np.nan
                VideoData1.loc[low_score_mask, f'{point}.y'] = np.nan
        
        # Find the channel with the maximum number of low-score points
        max_low_score_channel1 = max(low_score_counts1, key=low_score_counts1.get)
        max_low_score_count1 = low_score_counts1[max_low_score_channel1]
        
        # Print the channel with the maximum number of low-score points
        if debug:
            print(f"{get_eye_label('VideoData1')} - Channel with the maximum number of low-confidence points: {max_low_score_channel1}, Number of low-confidence points: {max_low_score_count1}")
            print(f"{get_eye_label('VideoData1')} - A total number of {total_low_score1} low-confidence coordinate values were replaced by interpolation")

    # VideoData2 score-based filtering
    if 'VideoData2_Has_Sleap' in globals() and VideoData2_Has_Sleap:
        total_low_score2 = 0
        low_score_counts2 = {}
        for point in point_names:
            if f'{point}.score' in VideoData2.columns:
                # Find indices where score is below threshold
                low_score_mask = VideoData2[f'{point}.score'] < score_cutoff
                low_score_count = low_score_mask.sum()
                low_score_counts2[f'{point}.x'] = low_score_count
                low_score_counts2[f'{point}.y'] = low_score_count
                total_low_score2 += low_score_count * 2  # *2 because we're removing both x and y
                
                # Set x and y to NaN for low confidence points
                VideoData2.loc[low_score_mask, f'{point}.x'] = np.nan
                VideoData2.loc[low_score_mask, f'{point}.y'] = np.nan
        
        # Find the channel with the maximum number of low-score points
        max_low_score_channel2 = max(low_score_counts2, key=low_score_counts2.get)
        max_low_score_count2 = low_score_counts2[max_low_score_channel2]
        
        # Print the channel with the maximum number of low-score points
        if debug:
            print(f"{get_eye_label('VideoData2')} - Channel with the maximum number of low-confidence points: {max_low_score_channel2}, Number of low-confidence points: {max_low_score_count2}")
            print(f"{get_eye_label('VideoData2')} - A total number of {total_low_score2} low-confidence coordinate values were replaced by interpolation")

    ############################################################################################################
    # remove outliers (x times SD)
    # then interpolates on all NaN values (skipped frames, low confidence inference points, outliers)
    ############################################################################################################

    if debug:
        print("\n=== Outlier Analysis - outlier points are replaced by interpolation ===")

    # VideoData1 outlier analysis and interpolation
    if 'VideoData1_Has_Sleap' in globals() and VideoData1_Has_Sleap:
        # Calculate the standard deviation for each column of interest
        std_devs1 = {col: VideoData1[col].std() for col in columns_of_interest}

        # Calculate the number of outliers for each column
        outliers1 = {col: ((VideoData1[col] - VideoData1[col].mean()).abs() > outlier_sd_threshold * std_devs1[col]).sum() for col in columns_of_interest}

        # Find the channel with the maximum number of outliers
        max_outliers_channel1 = max(outliers1, key=outliers1.get)
        max_outliers_count1 = outliers1[max_outliers_channel1]
        total_outliers1 = sum(outliers1.values())

        # Print the channel with the maximum number of outliers and the number
        if debug:
            print(f"{get_eye_label('VideoData1')} - Channel with the maximum number of outliers: {max_outliers_channel1}, Number of outliers: {max_outliers_count1}")
            print(f"{get_eye_label('VideoData1')} - A total number of {total_outliers1} outliers were replaced by interpolation")

        # Replace outliers by interpolating between the previous and subsequent non-NaN value
        for col in columns_of_interest:
            outlier_indices = VideoData1[((VideoData1[col] - VideoData1[col].mean()).abs() > outlier_sd_threshold * std_devs1[col])].index
            VideoData1.loc[outlier_indices, col] = np.nan

        #VideoData1.interpolate(inplace=True)
        VideoData1 = VideoData1.interpolate(method='linear', limit_direction='both')

    # VideoData2 outlier analysis and interpolation
    if 'VideoData2_Has_Sleap' in globals() and VideoData2_Has_Sleap:
        # Calculate the standard deviation for each column of interest
        std_devs2 = {col: VideoData2[col].std() for col in columns_of_interest}

        # Calculate the number of outliers for each column
        outliers2 = {col: ((VideoData2[col] - VideoData2[col].mean()).abs() > outlier_sd_threshold * std_devs2[col]).sum() for col in columns_of_interest}

        # Find the channel with the maximum number of outliers
        max_outliers_channel2 = max(outliers2, key=outliers2.get)
        max_outliers_count2 = outliers2[max_outliers_channel2]
        total_outliers2 = sum(outliers2.values())

        # Print the channel with the maximum number of outliers and the number
        if debug:
            print(f"{get_eye_label('VideoData2')} - Channel with the maximum number of outliers: {max_outliers_channel2}, Number of outliers: {max_outliers_count2}")
            print(f"{get_eye_label('VideoData2')} - A total number of {total_outliers2} outliers were replaced by interpolation")

        # Replace outliers by interpolating between the previous and subsequent non-NaN value
        for col in columns_of_interest:
            outlier_indices = VideoData2[((VideoData2[col] - VideoData2[col].mean()).abs() > outlier_sd_threshold * std_devs2[col])].index
            VideoData2.loc[outlier_indices, col] = np.nan

        #VideoData2.interpolate(inplace=True)
        VideoData2 = VideoData2.interpolate(method='linear', limit_direction='both')

    # Set flag after both VideoData1 and VideoData2 processing is complete
    NaNs_removed = True
else:
    print("=== Interpolation already done, skipping ===")


# %%
# Instance.score distribution and hard threshold for blink detection
############################################################################################################
# Plotting the distribution of instance scores and using hard threshold for blink detection.
# When instance score is low, that's typically because of a blink or similar occlusion, as there are long sequences of low scores.
# Frames with instance.score below the hard threshold are considered potential blinks.

if not debug:
    print("ℹ️ Debug output suppressed. Set debug=True to see detailed instance score distribution analysis.")

if debug:
    print("=" * 80)
    print("INSTANCE.SCORE DISTRIBUTION AND BLINK DETECTION THRESHOLD")
    print("=" * 80)
    print(f"\nHard threshold: instance.score < {blink_instance_score_threshold}")
    print(f"  Frames with instance.score below this threshold will be considered potential blinks.")
    print("=" * 80)

# Only analyze for dataset(s) that exist
has_v1 = "VideoData1_Has_Sleap" in globals() and VideoData1_Has_Sleap
has_v2 = "VideoData2_Has_Sleap" in globals() and VideoData2_Has_Sleap

# Get FPS for time calculations
if has_v1:
    if 'FPS_1' in globals():
        fps_1_for_threshold = FPS_1
    else:
        fps_1_for_threshold = 1 / VideoData1["Seconds"].diff().mean()
else:
    fps_1_for_threshold = None

if has_v2:
    if 'FPS_2' in globals():
        fps_2_for_threshold = FPS_2
    else:
        fps_2_for_threshold = 1 / VideoData2["Seconds"].diff().mean()
else:
    fps_2_for_threshold = None

# Plot histograms with hard threshold marked
fig = None
if has_v1 or has_v2:
    plt.figure(figsize=(12,5))
    plot_index = 1

if has_v1:
    plt.subplot(1, 2 if has_v2 else 1, plot_index)
    plt.hist(VideoData1['instance.score'].dropna(), bins=30, color='skyblue', edgecolor='black')
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
    plt.hist(VideoData2['instance.score'].dropna(), bins=30, color='salmon', edgecolor='black')
    plt.axvline(blink_instance_score_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Hard threshold = {blink_instance_score_threshold}')
    plt.yscale('log')
    plt.title("Distribution of instance.score (VideoData2)")
    plt.xlabel("instance.score")
    plt.ylabel("Frequency (log scale)")
    plt.legend()
    plot_index += 1

if has_v1 or has_v2:
    plt.tight_layout()
    plt.show()

# Report the statistics for available VideoData
# Always show key stats: number/percentile below threshold and longest consecutive segment
if has_v1:
    print(f"\nVideoData1 - Instance Score Threshold Analysis:")
    print(f"  Hard threshold: {blink_instance_score_threshold}")
    
    # Calculate percentile for reference
    v1_percentile = (VideoData1['instance.score'] < blink_instance_score_threshold).sum() / len(VideoData1) * 100
    v1_num_low = (VideoData1['instance.score'] < blink_instance_score_threshold).sum()
    v1_total = len(VideoData1)
    v1_pct_low = (v1_num_low / v1_total) * 100
    
    # Find longest consecutive segments
    low_sections_v1 = pf.find_longest_lowscore_sections(VideoData1['instance.score'], blink_instance_score_threshold, top_n=1)
    longest_consecutive_v1 = low_sections_v1[0]['length'] if low_sections_v1 else 0
    longest_consecutive_v1_ms = (longest_consecutive_v1 / fps_1_for_threshold) * 1000 if fps_1_for_threshold and longest_consecutive_v1 > 0 else None
    
    # Always print key stats
    print(f"  Frames below threshold: {v1_num_low} / {v1_total} ({v1_pct_low:.2f}%)")
    print(f"  Longest consecutive segment: {longest_consecutive_v1} frames", end="")
    if longest_consecutive_v1_ms:
        print(f" ({longest_consecutive_v1_ms:.1f}ms)")
    else:
        print()
    
    # Detailed stats only in debug mode
    if debug:
        print(f"\n  Detailed statistics:")
        print(f"  Percentile: {v1_percentile:.2f}% (i.e., {v1_percentile:.2f}% of frames have instance.score < {blink_instance_score_threshold})")
        
        # Report the top 5 longest consecutive sections
        low_sections_v1 = pf.find_longest_lowscore_sections(VideoData1['instance.score'], blink_instance_score_threshold, top_n=5)
        if len(low_sections_v1) > 0:
            print(f"\n  Top 5 longest consecutive sections where instance.score < threshold:")
            for i, sec in enumerate(low_sections_v1, 1):
                start_idx = sec['start_idx']
                end_idx = sec['end_idx']
                start_time = VideoData1.index[start_idx] if isinstance(VideoData1.index, pd.DatetimeIndex) else start_idx
                end_time = VideoData1.index[end_idx] if isinstance(VideoData1.index, pd.DatetimeIndex) else end_idx
                sec_duration_ms = (sec['length'] / fps_1_for_threshold) * 1000 if fps_1_for_threshold else None
                if sec_duration_ms:
                    print(f"    Section {i}: index {start_idx}-{end_idx} (length {sec['length']} frames, {sec_duration_ms:.1f}ms)")
                else:
                    print(f"    Section {i}: index {start_idx}-{end_idx} (length {sec['length']} frames)")

if has_v2:
    print(f"\nVideoData2 - Instance Score Threshold Analysis:")
    print(f"  Hard threshold: {blink_instance_score_threshold}")
    
    # Calculate percentile for reference
    v2_percentile = (VideoData2['instance.score'] < blink_instance_score_threshold).sum() / len(VideoData2) * 100
    v2_num_low = (VideoData2['instance.score'] < blink_instance_score_threshold).sum()
    v2_total = len(VideoData2)
    v2_pct_low = (v2_num_low / v2_total) * 100
    
    # Find longest consecutive segments
    low_sections_v2 = pf.find_longest_lowscore_sections(VideoData2['instance.score'], blink_instance_score_threshold, top_n=1)
    longest_consecutive_v2 = low_sections_v2[0]['length'] if low_sections_v2 else 0
    longest_consecutive_v2_ms = (longest_consecutive_v2 / fps_2_for_threshold) * 1000 if fps_2_for_threshold and longest_consecutive_v2 > 0 else None
    
    # Always print key stats
    print(f"  Frames below threshold: {v2_num_low} / {v2_total} ({v2_pct_low:.2f}%)")
    print(f"  Longest consecutive segment: {longest_consecutive_v2} frames", end="")
    if longest_consecutive_v2_ms:
        print(f" ({longest_consecutive_v2_ms:.1f}ms)")
    else:
        print()
    
    # Detailed stats only in debug mode
    if debug:
        print(f"\n  Detailed statistics:")
        print(f"  Percentile: {v2_percentile:.2f}% (i.e., {v2_percentile:.2f}% of frames have instance.score < {blink_instance_score_threshold})")
        
        # Report the top 5 longest consecutive sections
        low_sections_v2 = pf.find_longest_lowscore_sections(VideoData2['instance.score'], blink_instance_score_threshold, top_n=5)
        if len(low_sections_v2) > 0:
            print(f"\n  Top 5 longest consecutive sections where instance.score < threshold:")
            for i, sec in enumerate(low_sections_v2, 1):
                start_idx = sec['start_idx']
                end_idx = sec['end_idx']
                start_time = VideoData2.index[start_idx] if isinstance(VideoData2.index, pd.DatetimeIndex) else start_idx
                end_time = VideoData2.index[end_idx] if isinstance(VideoData2.index, pd.DatetimeIndex) else end_idx
                sec_duration_ms = (sec['length'] / fps_2_for_threshold) * 1000 if fps_2_for_threshold else None
                if sec_duration_ms:
                    print(f"    Section {i}: index {start_idx}-{end_idx} (length {sec['length']} frames, {sec_duration_ms:.1f}ms)")
                else:
                    print(f"    Section {i}: index {start_idx}-{end_idx} (length {sec['length']} frames)")

if debug:
    print(f"\n{'='*80}")
    print("Note: This threshold will be used for blink detection in the next cell.")
    print("      Frames with instance.score below this threshold are considered potential blinks.")
    print("=" * 80)



# %%
# Blink detection using instance.score - mark blinks and set coordinates to NaN (keep them as NaN, no interpolation)
############################################################################################################

if not debug:
    print("ℹ️ Debug output suppressed. Set debug=True to see detailed blink detection information.")

# Capture all print output to save to file

class TeeOutput:
    """Output to both stdout and a string buffer"""
    def __init__(self, stdout, buffer):
        self.stdout = stdout
        self.buffer = buffer
    
    def write(self, s):
        self.stdout.write(s)
        self.buffer.write(s)
    
    def flush(self):
        self.stdout.flush()
        self.buffer.flush()

output_buffer = io.StringIO()
original_stdout = sys.stdout
sys.stdout = TeeOutput(original_stdout, output_buffer)

# Run blink detection code with output captured
if debug:
    print("\n=== Blink Detection ===")

# VideoData1 blink detection
if 'VideoData1_Has_Sleap' in globals() and VideoData1_Has_Sleap:
    # Get FPS if available, otherwise will be calculated in function
    fps_1 = FPS_1 if 'FPS_1' in globals() else None
    
    # Get manual blinks if available
    manual_blinks_for_v1 = manual_blinks_v1 if 'manual_blinks_v1' in globals() and manual_blinks_v1 is not None else None
    
    # Run blink detection
    blink_results_v1 = pf.detect_blinks_for_video(
        video_data=VideoData1,
        columns_of_interest=columns_of_interest,
        blink_instance_score_threshold=blink_instance_score_threshold,
        long_blink_warning_ms=long_blink_warning_ms,
        min_frames_threshold=4,
        merge_window_frames=10,
        fps=fps_1,
        video_label=get_eye_label('VideoData1'),
        manual_blinks=manual_blinks_for_v1,
        debug=debug
    )
    
    # Extract results to maintain compatibility with existing variable names
    blink_segments_v1 = blink_results_v1['blink_segments']
    short_blink_segments_v1 = blink_results_v1['short_blink_segments']
    blink_bouts_v1 = blink_results_v1['blink_bouts']
    all_blink_segments_v1 = blink_results_v1['all_blink_segments']
    fps_1 = blink_results_v1['fps']  # Update fps_1 with calculated value
    FPS_1 = fps_1  # Also update global FPS_1 for use elsewhere
    long_blinks_warnings_v1 = blink_results_v1['long_blinks_warnings']

# VideoData2 blink detection
if 'VideoData2_Has_Sleap' in globals() and VideoData2_Has_Sleap:
    # Get FPS if available, otherwise will be calculated in function
    fps_2 = FPS_2 if 'FPS_2' in globals() else None
    
    # Get manual blinks if available
    manual_blinks_for_v2 = manual_blinks_v2 if 'manual_blinks_v2' in globals() and manual_blinks_v2 is not None else None
    
    # Run blink detection
    blink_results_v2 = pf.detect_blinks_for_video(
        video_data=VideoData2,
        columns_of_interest=columns_of_interest,
        blink_instance_score_threshold=blink_instance_score_threshold,
        long_blink_warning_ms=long_blink_warning_ms,
        min_frames_threshold=4,
        merge_window_frames=10,
        fps=fps_2,
        video_label=get_eye_label('VideoData2'),
        manual_blinks=manual_blinks_for_v2,
        debug=debug
    )
    
    # Extract results to maintain compatibility with existing variable names
    blink_segments_v2 = blink_results_v2['blink_segments']
    short_blink_segments_v2 = blink_results_v2['short_blink_segments']
    blink_bouts_v2 = blink_results_v2['blink_bouts']
    all_blink_segments_v2 = blink_results_v2['all_blink_segments']
    fps_2 = blink_results_v2['fps']  # Update fps_2 with calculated value
    FPS_2 = fps_2  # Also update global FPS_2 for use elsewhere
    long_blinks_warnings_v2 = blink_results_v2['long_blinks_warnings']

print("\n✅ Blink detection complete. Blink periods remain as NaN (not interpolated).")

# Compare blink bout timing between VideoData1 and VideoData2 (between eyes)
if ('VideoData1_Has_Sleap' in globals() and VideoData1_Has_Sleap and 
    'VideoData2_Has_Sleap' in globals() and VideoData2_Has_Sleap):
    if debug:
        print("\n" + "="*80)
        print("BLINK BOUT TIMING COMPARISON: VideoData1 vs VideoData2 (Between Eyes)")
        print("="*80)
    
    # Get blink bout frame ranges for both videos (if they exist)
    # Check if blink_bouts variables exist (they are created during blink detection)
    try:
        has_bouts_v1 = 'blink_bouts_v1' in globals() and len(blink_bouts_v1) > 0
    except:
        has_bouts_v1 = False
    
    try:
        has_bouts_v2 = 'blink_bouts_v2' in globals() and len(blink_bouts_v2) > 0
    except:
        has_bouts_v2 = False
    
    if has_bouts_v1 and has_bouts_v2:
        
        # Convert bout indices to frame numbers
        bouts_v1 = []
        for i, bout in enumerate(blink_bouts_v1, 1):
            start_idx = bout['start_idx']
            end_idx = bout['end_idx']
            if 'frame_idx' in VideoData1.columns:
                start_frame = int(VideoData1['frame_idx'].iloc[start_idx])
                end_frame = int(VideoData1['frame_idx'].iloc[end_idx])
            else:
                start_frame = start_idx
                end_frame = end_idx
            bouts_v1.append({
                'num': i,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'length': bout['length']
            })
        
        bouts_v2 = []
        for i, bout in enumerate(blink_bouts_v2, 1):
            start_idx = bout['start_idx']
            end_idx = bout['end_idx']
            if 'frame_idx' in VideoData2.columns:
                start_frame = int(VideoData2['frame_idx'].iloc[start_idx])
                end_frame = int(VideoData2['frame_idx'].iloc[end_idx])
            else:
                start_frame = start_idx
                end_frame = end_idx
            bouts_v2.append({
                'num': i,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'length': bout['length']
            })
        
        # Find concurrent bouts (overlapping in time, synchronized by Seconds)
        concurrent_bouts = []
        v1_independent = []
        v2_independent = []
        
        v2_matched = set()  # Track which VideoData2 bouts have been matched
        
        for bout1 in bouts_v1:
            # Get time range for bout1
            v1_start_time = VideoData1['Seconds'].iloc[bout1['start_idx']]
            v1_end_time = VideoData1['Seconds'].iloc[bout1['end_idx']]
            
            found_match = False
            for bout2 in bouts_v2:
                # Get time range for bout2
                v2_start_time = VideoData2['Seconds'].iloc[bout2['start_idx']]
                v2_end_time = VideoData2['Seconds'].iloc[bout2['end_idx']]
                
                # Check if bouts overlap in time (any overlapping time period)
                overlap_start_time = max(v1_start_time, v2_start_time)
                overlap_end_time = min(v1_end_time, v2_end_time)
                
                if overlap_start_time <= overlap_end_time:
                    # Concurrent - they overlap in time
                    # Calculate overlap duration
                    overlap_duration = overlap_end_time - overlap_start_time
                    
                    concurrent_bouts.append({
                        'v1_num': bout1['num'],
                        'v1_start_frame': bout1['start_frame'],
                        'v1_end_frame': bout1['end_frame'],
                        'v1_start_time': v1_start_time,
                        'v1_end_time': v1_end_time,
                        'v2_num': bout2['num'],
                        'v2_start_frame': bout2['start_frame'],
                        'v2_end_frame': bout2['end_frame'],
                        'v2_start_time': v2_start_time,
                        'v2_end_time': v2_end_time,
                        'overlap_start_time': overlap_start_time,
                        'overlap_end_time': overlap_end_time,
                        'overlap_duration': overlap_duration
                    })
                    v2_matched.add(bout2['num'])
                    found_match = True
                    break
            
            if not found_match:
                v1_independent.append(bout1)
        
        # Find VideoData2 bouts that don't have matches
        for bout2 in bouts_v2:
            if bout2['num'] not in v2_matched:
                v2_independent.append(bout2)
        
        # Calculate statistics
        total_v1_bouts = len(bouts_v1)
        total_v2_bouts = len(bouts_v2)
        total_concurrent = len(concurrent_bouts)
        total_v1_independent = len(v1_independent)
        total_v2_independent = len(v2_independent)
        
        if debug:
            print(f"\nBlink bout counts:")
            print(f"  VideoData1: {total_v1_bouts} blink bout(s)")
            print(f"  VideoData2: {total_v2_bouts} blink bout(s)")
            print(f"  Concurrent: {total_concurrent} bout(s) (overlapping frames)")
            print(f"  VideoData1 only: {total_v1_independent} bout(s)")
            print(f"  VideoData2 only: {total_v2_independent} bout(s)")
            
            if total_v1_bouts > 0 and total_v2_bouts > 0:
                concurrent_pct_v1 = (total_concurrent / total_v1_bouts) * 100
                concurrent_pct_v2 = (total_concurrent / total_v2_bouts) * 100
                print(f"\nConcurrency percentage:")
                print(f"  {concurrent_pct_v1:.1f}% of VideoData1 bouts are concurrent with VideoData2")
                print(f"  {concurrent_pct_v2:.1f}% of VideoData2 bouts are concurrent with VideoData1")
                
                # Calculate timing offsets for concurrent bouts
                if len(concurrent_bouts) > 0:
                    time_offsets_ms = []
                    for cb in concurrent_bouts:
                        # Calculate offset from start times (already in Seconds)
                        offset_ms = (cb['v1_start_time'] - cb['v2_start_time']) * 1000
                        time_offsets_ms.append(offset_ms)
                        cb['time_offset_ms'] = offset_ms
                    
                    mean_offset = np.mean(time_offsets_ms)
                    std_offset = np.std(time_offsets_ms)
                    print(f"\nTiming offset for concurrent bouts:")
                    print(f"  Mean offset (VideoData1 - VideoData2): {mean_offset:.2f} ms")
                    print(f"  Std offset: {std_offset:.2f} ms")
                    print(f"  Range: {min(time_offsets_ms):.2f} to {max(time_offsets_ms):.2f} ms")
            
            # Visualization removed per request
            print("="*80)
    elif has_bouts_v1 or has_bouts_v2:
        print(f"\n⚠️ Cannot compare blink bouts - only one eye has blink bouts detected:")
        if has_bouts_v1:
            print(f"  VideoData1: {len(blink_bouts_v1)} blink bout(s)")
        else:
            print(f"  VideoData1: 0 blink bout(s)")
        if has_bouts_v2:
            print(f"  VideoData2: {len(blink_bouts_v2)} blink bout(s)")
        else:
            print(f"  VideoData2: 0 blink bout(s)")
    else:
        print("\n⚠️ Cannot compare blink bouts - neither video has blink bouts detected")

# Save blink detection results to CSV files
if 'VideoData1_Has_Sleap' in globals() and VideoData1_Has_Sleap:
    if len(blink_segments_v1) > 0:
        # Collect blink information
        blink_data_v1 = []
        manual_blinks_for_csv = None
        if 'manual_blinks_v1' in globals() and manual_blinks_v1 is not None:
            manual_blinks_for_csv = manual_blinks_v1
            
        for i, blink in enumerate(blink_segments_v1, 1):
            start_idx = blink['start_idx']
            end_idx = blink['end_idx']
            
            # Get actual frame numbers from frame_idx column
            if 'frame_idx' in VideoData1.columns:
                first_frame = int(VideoData1['frame_idx'].iloc[start_idx])
                last_frame = int(VideoData1['frame_idx'].iloc[end_idx])
            else:
                first_frame = start_idx
                last_frame = end_idx
            
            # Check if this blink matches a manual one (using function from processing_functions)
            matches_manual = pf.check_manual_match(first_frame, last_frame, manual_blinks_for_csv)
            
            blink_data_v1.append({
                'blink_number': i,
                'first_frame': first_frame,
                'last_frame': last_frame,
                'matches_manual': matches_manual
            })
        
        # Create DataFrame and save to CSV
        blink_df_v1 = pd.DataFrame(blink_data_v1)
        blink_csv_path_v1 = data_path / "blink_detection_VideoData1.csv"
        blink_df_v1.to_csv(blink_csv_path_v1, index=False)
        print(f"\n✅ Blink detection results (VideoData1) saved to: {blink_csv_path_v1}")
        print(f"   Saved {len(blink_data_v1)} blinks")

if 'VideoData2_Has_Sleap' in globals() and VideoData2_Has_Sleap:
    if len(blink_segments_v2) > 0:
        # Collect blink information
        blink_data_v2 = []
        manual_blinks_for_csv = None
        if 'manual_blinks_v2' in globals() and manual_blinks_v2 is not None:
            manual_blinks_for_csv = manual_blinks_v2
            
        for i, blink in enumerate(blink_segments_v2, 1):
            start_idx = blink['start_idx']
            end_idx = blink['end_idx']
            
            # Get actual frame numbers from frame_idx column
            if 'frame_idx' in VideoData2.columns:
                first_frame = int(VideoData2['frame_idx'].iloc[start_idx])
                last_frame = int(VideoData2['frame_idx'].iloc[end_idx])
            else:
                first_frame = start_idx
                last_frame = end_idx
            
            # Check if this blink matches a manual one (using function from processing_functions)
            matches_manual = pf.check_manual_match(first_frame, last_frame, manual_blinks_for_csv)
            
            blink_data_v2.append({
                'blink_number': i,
                'first_frame': first_frame,
                'last_frame': last_frame,
                'matches_manual': matches_manual
            })
        
        # Create DataFrame and save to CSV
        blink_df_v2 = pd.DataFrame(blink_data_v2)
        blink_csv_path_v2 = data_path / "blink_detection_VideoData2.csv"
        blink_df_v2.to_csv(blink_csv_path_v2, index=False)
        print(f"\n✅ Blink detection results (VideoData2) saved to: {blink_csv_path_v2}")
        print(f"   Saved {len(blink_data_v2)} blinks")

print("\n" + "="*80)
print("📹 MANUAL QC CHECK:")
print("="*80)
print("For instructions on how to prepare videos for manual blink detection QC,")
print("see: https://github.com/ranczlab/vestibular_vr_pipeline/issues/86")
print("="*80)

# Restore original stdout and save captured output to file
sys.stdout = original_stdout

# Get the captured output
captured_output = output_buffer.getvalue()

# Save to file in data_path folder
output_file = data_path / "blink_detection_QC.txt"
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    f.write(captured_output)

print(f"\n✅ Blink detection output saved to: {output_file}")



# %%
# QC plot timeseries of interpolation corrected NaN and (TODO low confidence coordinates in browser 
############################################################################################################

if plot_QC_timeseries:
    print(f'ℹ️ Figure opens in browser window, takes a bit of time.')
    
    # VideoData1 QC Plot
    if 'VideoData1_Has_Sleap' in globals() and VideoData1_Has_Sleap:
        fig1 = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"VideoData1 - X coordinates for pupil centre and left-right eye corner",
                f"VideoData1 - Y coordinates for pupil centre and left-right eye corner",
                f"VideoData1 - X coordinates for iris points",
                f"VideoData1 - Y coordinates for iris points"
            )
        )

        # Row 1: Plot left.x, center.x, right.x
        fig1.add_trace(go.Scatter(x=VideoData1['Seconds'], y=VideoData1['left.x'], mode='lines', name='left.x'), row=1, col=1)
        fig1.add_trace(go.Scatter(x=VideoData1['Seconds'], y=VideoData1['center.x'], mode='lines', name='center.x'), row=1, col=1)
        fig1.add_trace(go.Scatter(x=VideoData1['Seconds'], y=VideoData1['right.x'], mode='lines', name='right.x'), row=1, col=1)

        # Row 2: Plot left.y, center.y, right.y
        fig1.add_trace(go.Scatter(x=VideoData1['Seconds'], y=VideoData1['left.y'], mode='lines', name='left.y'), row=2, col=1)
        fig1.add_trace(go.Scatter(x=VideoData1['Seconds'], y=VideoData1['center.y'], mode='lines', name='center.y'), row=2, col=1)
        fig1.add_trace(go.Scatter(x=VideoData1['Seconds'], y=VideoData1['right.y'], mode='lines', name='right.y'), row=2, col=1)

        # Row 3: Plot p.x coordinates for p1 to p8
        for col in ['p1.x', 'p2.x', 'p3.x', 'p4.x', 'p5.x', 'p6.x', 'p7.x', 'p8.x']:
            fig1.add_trace(go.Scatter(x=VideoData1['Seconds'], y=VideoData1[col], mode='lines', name=col), row=3, col=1)

        # Row 4: Plot p.y coordinates for p1 to p8
        for col in ['p1.y', 'p2.y', 'p3.y', 'p4.y', 'p5.y', 'p6.y', 'p7.y', 'p8.y']:
            fig1.add_trace(go.Scatter(x=VideoData1['Seconds'], y=VideoData1[col], mode='lines', name=col), row=4, col=1)

        fig1.update_layout(
            height=1200,
            title_text=f"VideoData1 - Time series subplots for coordinates (QC after interpolation)",
            showlegend=True
        )
        fig1.update_xaxes(title_text="Seconds", row=4, col=1)
        fig1.update_yaxes(title_text="X Position", row=1, col=1)
        fig1.update_yaxes(title_text="Y Position", row=2, col=1)
        fig1.update_yaxes(title_text="X Position", row=3, col=1)
        fig1.update_yaxes(title_text="Y Position", row=4, col=1)

        fig1.show(renderer='browser')
    
    # VideoData2 QC Plot
    if 'VideoData2_Has_Sleap' in globals() and VideoData2_Has_Sleap:
        fig2 = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"VideoData2 - X coordinates for pupil centre and left-right eye corner",
                f"VideoData2 - Y coordinates for pupil centre and left-right eye corner",
                f"VideoData2 - X coordinates for iris points",
                f"VideoData2 - Y coordinates for iris points"
            )
        )

        # Row 1: Plot left.x, center.x, right.x
        fig2.add_trace(go.Scatter(x=VideoData2['Seconds'], y=VideoData2['left.x'], mode='lines', name='left.x'), row=1, col=1)
        fig2.add_trace(go.Scatter(x=VideoData2['Seconds'], y=VideoData2['center.x'], mode='lines', name='center.x'), row=1, col=1)
        fig2.add_trace(go.Scatter(x=VideoData2['Seconds'], y=VideoData2['right.x'], mode='lines', name='right.x'), row=1, col=1)

        # Row 2: Plot left.y, center.y, right.y
        fig2.add_trace(go.Scatter(x=VideoData2['Seconds'], y=VideoData2['left.y'], mode='lines', name='left.y'), row=2, col=1)
        fig2.add_trace(go.Scatter(x=VideoData2['Seconds'], y=VideoData2['center.y'], mode='lines', name='center.y'), row=2, col=1)
        fig2.add_trace(go.Scatter(x=VideoData2['Seconds'], y=VideoData2['right.y'], mode='lines', name='right.y'), row=2, col=1)

        # Row 3: Plot p.x coordinates for p1 to p8
        for col in ['p1.x', 'p2.x', 'p3.x', 'p4.x', 'p5.x', 'p6.x', 'p7.x', 'p8.x']:
            fig2.add_trace(go.Scatter(x=VideoData2['Seconds'], y=VideoData2[col], mode='lines', name=col), row=3, col=1)

        # Row 4: Plot p.y coordinates for p1 to p8
        for col in ['p1.y', 'p2.y', 'p3.y', 'p4.y', 'p5.y', 'p6.y', 'p7.y', 'p8.y']:
            fig2.add_trace(go.Scatter(x=VideoData2['Seconds'], y=VideoData2[col], mode='lines', name=col), row=4, col=1)

        fig2.update_layout(
            height=1200,
            title_text=f"VideoData2 - Time series subplots for coordinates (QC after interpolation)",
            showlegend=True
        )
        fig2.update_xaxes(title_text="Seconds", row=4, col=1)
        fig2.update_yaxes(title_text="X Position", row=1, col=1)
        fig2.update_yaxes(title_text="Y Position", row=2, col=1)
        fig2.update_yaxes(title_text="X Position", row=3, col=1)
        fig2.update_yaxes(title_text="Y Position", row=4, col=1)

        fig2.show(renderer='browser')


# %%
# QC plot XY coordinate distributions after NaN and ( TODO - low confidence inference points) are interpolated 
##############################################################################################################

columns_of_interest = ['left.x','left.y','center.x','center.y','right.x','right.y','p1.x','p1.y','p2.x','p2.y','p3.x','p3.y','p4.x','p4.y','p5.x','p5.y','p6.x','p6.y','p7.x','p7.y','p8.x','p8.y']

# Create coordinates_dict for both datasets
if VideoData1_Has_Sleap:
    coordinates_dict1_processed = lp.get_coordinates_dict(VideoData1, columns_of_interest)
if VideoData2_Has_Sleap:
    coordinates_dict2_processed = lp.get_coordinates_dict(VideoData2, columns_of_interest)

columns_of_interest = ['left', 'right', 'center', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']

# Filter out NaN values and calculate the min and max values for X and Y coordinates for both dict1 and dict2
def min_max_dict(coordinates_dict):
    x_min = min([coordinates_dict[f'{col}.x'][~np.isnan(coordinates_dict[f'{col}.x'])].min() for col in columns_of_interest])
    x_max = max([coordinates_dict[f'{col}.x'][~np.isnan(coordinates_dict[f'{col}.x'])].max() for col in columns_of_interest])
    y_min = min([coordinates_dict[f'{col}.y'][~np.isnan(coordinates_dict[f'{col}.y'])].min() for col in columns_of_interest])
    y_max = max([coordinates_dict[f'{col}.y'][~np.isnan(coordinates_dict[f'{col}.y'])].max() for col in columns_of_interest])
    return x_min, x_max, y_min, y_max

if VideoData1_Has_Sleap:
    x_min1, x_max1, y_min1, y_max1 = min_max_dict(coordinates_dict1_processed)
if VideoData2_Has_Sleap:
    x_min2, x_max2, y_min2, y_max2 = min_max_dict(coordinates_dict2_processed)

# Use global min and max for consistency across subplots
if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    x_min = min(x_min1, x_min2)
    x_max = max(x_max1, x_max2)
    y_min = min(y_min1, y_min2)
    y_max = max(y_max1, y_max2)
elif VideoData1_Has_Sleap:
    x_min, x_max, y_min, y_max = x_min1, x_max1, y_min1, y_max1
elif VideoData2_Has_Sleap:
    x_min, x_max, y_min, y_max = x_min2, x_max2, y_min2, y_max2

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
fig.suptitle(
    f"XY coordinate distribution of different points for {get_eye_label('VideoData1')} and {get_eye_label('VideoData2')} post outlier removal and NaN interpolation",
    fontsize=14
)

# Define colormap for p1-p8
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']

# Panel 1: left, right, center (VideoData1)
if VideoData1_Has_Sleap:
    ax[0, 0].set_title(f"{get_eye_label('VideoData1')}: left, right, center")
    ax[0, 0].scatter(coordinates_dict1_processed['left.x'], coordinates_dict1_processed['left.y'], color='black', label='left', s=10)
    ax[0, 0].scatter(coordinates_dict1_processed['right.x'], coordinates_dict1_processed['right.y'], color='grey', label='right', s=10)
    ax[0, 0].scatter(coordinates_dict1_processed['center.x'], coordinates_dict1_processed['center.y'], color='red', label='center', s=10)
    ax[0, 0].set_xlim([x_min, x_max])
    ax[0, 0].set_ylim([y_min, y_max])
    ax[0, 0].set_xlabel('x coordinates (pixels)')
    ax[0, 0].set_ylabel('y coordinates (pixels)')
    ax[0, 0].legend(loc='upper right')

    # Panel 2: p1 to p8 (VideoData1)
    ax[0, 1].set_title(f"{get_eye_label('VideoData1')}: p1 to p8")
    for idx, col in enumerate(columns_of_interest[3:]):
        ax[0, 1].scatter(coordinates_dict1_processed[f'{col}.x'], coordinates_dict1_processed[f'{col}.y'], color=colors[idx], label=col, s=5)
    ax[0, 1].set_xlim([x_min, x_max])
    ax[0, 1].set_ylim([y_min, y_max])
    ax[0, 1].set_xlabel('x coordinates (pixels)')
    ax[0, 1].set_ylabel('y coordinates (pixels)')
    ax[0, 1].legend(loc='upper right')

# Panel 3: left, right, center (VideoData2)
if VideoData2_Has_Sleap:
    ax[1, 0].set_title(f"{get_eye_label('VideoData2')}: left, right, center")
    ax[1, 0].scatter(coordinates_dict2_processed['left.x'], coordinates_dict2_processed['left.y'], color='black', label='left', s=10)
    ax[1, 0].scatter(coordinates_dict2_processed['right.x'], coordinates_dict2_processed['right.y'], color='grey', label='right', s=10)
    ax[1, 0].scatter(coordinates_dict2_processed['center.x'], coordinates_dict2_processed['center.y'], color='red', label='center', s=10)
    ax[1, 0].set_xlim([x_min, x_max])
    ax[1, 0].set_ylim([y_min, y_max])
    ax[1, 0].set_xlabel('x coordinates (pixels)')
    ax[1, 0].set_ylabel('y coordinates (pixels)')
    ax[1, 0].legend(loc='upper right')

    # Panel 4: p1 to p8 (VideoData2)
    ax[1, 1].set_title(f"{get_eye_label('VideoData2')}: p1 to p8")
    for idx, col in enumerate(columns_of_interest[3:]):
        ax[1, 1].scatter(coordinates_dict2_processed[f'{col}.x'], coordinates_dict2_processed[f'{col}.y'], color=colors[idx], label=col, s=5)
    ax[1, 1].set_xlim([x_min, x_max])
    ax[1, 1].set_ylim([y_min, y_max])
    ax[1, 1].set_xlabel('x coordinates (pixels)')
    ax[1, 1].set_ylabel('y coordinates (pixels)')
    ax[1, 1].legend(loc='upper right')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

# %%
# fit ellipses on the 8 points to determine pupil centre and diameter
############################################################################################################

columns_of_interest = ['left.x','left.y','center.x','center.y','right.x','right.y','p1.x','p1.y','p2.x','p2.y','p3.x','p3.y','p4.x','p4.y','p5.x','p5.y','p6.x','p6.y','p7.x','p7.y','p8.x','p8.y']

# VideoData1 processing
if VideoData1_Has_Sleap:
    print(f"=== VideoData1 Ellipse Fitting for Pupil Diameter ===")
    coordinates_dict1_processed = lp.get_coordinates_dict(VideoData1, columns_of_interest)

    theta1 = lp.find_horizontal_axis_angle(VideoData1, 'left', 'center')
    center_point1 = lp.get_left_right_center_point(coordinates_dict1_processed)

    columns_of_interest_reformatted = ['left', 'right', 'center', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
    remformatted_coordinates_dict1 = lp.get_reformatted_coordinates_dict(coordinates_dict1_processed, columns_of_interest_reformatted)
    centered_coordinates_dict1 = lp.get_centered_coordinates_dict(remformatted_coordinates_dict1, center_point1)
    rotated_coordinates_dict1 = lp.get_rotated_coordinates_dict(centered_coordinates_dict1, theta1)

    columns_of_interest_ellipse = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
    ellipse_parameters_data1, ellipse_center_points_data1 = lp.get_fitted_ellipse_parameters(rotated_coordinates_dict1, columns_of_interest_ellipse)

    average_diameter1 = np.mean([ellipse_parameters_data1[:,0], ellipse_parameters_data1[:,1]], axis=0)

    SleapVideoData1 = process.convert_arrays_to_dataframe(['Seconds', 'Ellipse.Diameter', 'Ellipse.Angle', 'Ellipse.Center.X', 'Ellipse.Center.Y'], [VideoData1['Seconds'].values, average_diameter1, ellipse_parameters_data1[:,2], ellipse_center_points_data1[:,0], ellipse_center_points_data1[:,1]])

# VideoData2 processing
if VideoData2_Has_Sleap:
    print(f"=== VideoData2 Ellipse Fitting for Pupil Diameter ===")
    coordinates_dict2_processed = lp.get_coordinates_dict(VideoData2, columns_of_interest)

    theta2 = lp.find_horizontal_axis_angle(VideoData2, 'left', 'center')
    center_point2 = lp.get_left_right_center_point(coordinates_dict2_processed)

    columns_of_interest_reformatted = ['left', 'right', 'center', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
    remformatted_coordinates_dict2 = lp.get_reformatted_coordinates_dict(coordinates_dict2_processed, columns_of_interest_reformatted)
    centered_coordinates_dict2 = lp.get_centered_coordinates_dict(remformatted_coordinates_dict2, center_point2)
    rotated_coordinates_dict2 = lp.get_rotated_coordinates_dict(centered_coordinates_dict2, theta2)

    columns_of_interest_ellipse = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
    ellipse_parameters_data2, ellipse_center_points_data2 = lp.get_fitted_ellipse_parameters(rotated_coordinates_dict2, columns_of_interest_ellipse)

    average_diameter2 = np.mean([ellipse_parameters_data2[:,0], ellipse_parameters_data2[:,1]], axis=0)

    SleapVideoData2 = process.convert_arrays_to_dataframe(['Seconds', 'Ellipse.Diameter', 'Ellipse.Angle', 'Ellipse.Center.X', 'Ellipse.Center.Y'], [VideoData2['Seconds'].values, average_diameter2, ellipse_parameters_data2[:,2], ellipse_center_points_data2[:,0], ellipse_center_points_data2[:,1]])

############################################################################################################
# Filter pupil diameter using 10 Hz Butterworth low-pass filter
############################################################################################################

# VideoData1 filtering
if VideoData1_Has_Sleap:
    print(f"\n=== Filtering pupil diameter for VideoData1  ===")
    # Butterworth filter parameters - pupil_filter_cutoff_hz low-pass filter
    fs1 = 1 / np.median(np.diff(SleapVideoData1['Seconds']))  # Sampling frequency (Hz)
    order = pupil_filter_order

    b1, a1 = butter(order, pupil_filter_cutoff_hz / (0.5 * fs1), btype='low')
    
    # Handle NaN values before filtering (from blink detection)
    # Replace NaN with forward-fill for filtering purposes only (to avoid filtfilt issues)
    diameter_data = SleapVideoData1['Ellipse.Diameter'].copy()
    # Use ffill() and bfill() instead of deprecated fillna(method='ffill')
    diameter_data_filled = diameter_data.ffill().bfill()
    
    # Apply Butterworth filter
    if not diameter_data_filled.isna().all():
        filtered = filtfilt(b1, a1, diameter_data_filled)
        # Restore NaN values at original NaN positions (from blinks)
        filtered = pd.Series(filtered, index=diameter_data.index)
        filtered[diameter_data.isna()] = np.nan
        SleapVideoData1['Ellipse.Diameter.Filt'] = filtered
    else:
        # If all values are NaN, just copy
        SleapVideoData1['Ellipse.Diameter.Filt'] = diameter_data

# VideoData2 filtering
if VideoData2_Has_Sleap:
    print(f"=== Filtering pupil diameter for VideoData1 ===")
    # Butterworth filter parameters - pupil_filter_cutoff_hz low-pass filter
    fs2 = 1 / np.median(np.diff(SleapVideoData2['Seconds']))  # Sampling frequency (Hz)
    order = pupil_filter_order

    b2, a2 = butter(order, pupil_filter_cutoff_hz / (0.5 * fs2), btype='low')
    
    # Handle NaN values before filtering (from blink detection)
    # Replace NaN with forward-fill for filtering purposes only (to avoid filtfilt issues)
    diameter_data = SleapVideoData2['Ellipse.Diameter'].copy()
    # Use ffill() and bfill() instead of deprecated fillna(method='ffill')
    diameter_data_filled = diameter_data.ffill().bfill()
    
    # Apply Butterworth filter
    if not diameter_data_filled.isna().all():
        filtered = filtfilt(b2, a2, diameter_data_filled)
        # Restore NaN values at original NaN positions (from blinks)
        filtered = pd.Series(filtered, index=diameter_data.index)
        filtered[diameter_data.isna()] = np.nan
        SleapVideoData2['Ellipse.Diameter.Filt'] = filtered
    else:
        # If all values are NaN, just copy
        SleapVideoData2['Ellipse.Diameter.Filt'] = diameter_data

print("✅ Done calculating pupil diameter and angle for both VideoData1 and VideoData2")

# %%
# cross-correlate pupil diameter for left and right eye 
############################################################################################################

if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    # # Create subplots for both comparison and cross-correlation
    # fig = make_subplots(
    #     rows=2, cols=1,
    #     subplot_titles=["Pupil Diameter Comparison", "Cross-Correlation Analysis"],
    #     vertical_spacing=0.15
    # )

    # # Add SleapVideoData1 pupil diameter
    # fig.add_trace(
    #     go.Scatter(
    #         x=SleapVideoData1['Seconds'],
    #         y=SleapVideoData1['Ellipse.Diameter'],
    #         mode='lines',
    #         name=f"VideoData1 Pupil Diameter",
    #         line=dict(color='blue')
    #     ),
    #     row=1, col=1
    # )

    # # Add SleapVideoData2 pupil diameter
    # fig.add_trace(
    #     go.Scatter(
    #         x=SleapVideoData2['Seconds'],
    #         y=SleapVideoData2['Ellipse.Diameter'],
    #         mode='lines',
    #         name=f"VideoData2 Pupil Diameter",
    #         line=dict(color='red')
    #     ),
    #     row=1, col=1
    # )

    # Cross-correlation analysis
    print("=== Cross-Correlation Analysis ===")

    # Get pupil diameter data
    # Use filtered diameter data (with NaN restored at blink positions)
    pupil1 = SleapVideoData1['Ellipse.Diameter.Filt'].values
    pupil2 = SleapVideoData2['Ellipse.Diameter.Filt'].values

    # Handle different lengths by using the shorter dataset length
    min_length = min(len(pupil1), len(pupil2))

    # Truncate both datasets to the same length (preserving time alignment)
    pupil1_truncated = pupil1[:min_length]
    pupil2_truncated = pupil2[:min_length]

    # Remove NaN values for correlation - preserve time alignment by only keeping pairs where BOTH are valid
    # This ensures cross-correlation is computed on temporally aligned data
    valid_mask1 = ~np.isnan(pupil1_truncated)
    valid_mask2 = ~np.isnan(pupil2_truncated)
    valid_mask = valid_mask1 & valid_mask2  # Only use indices where both arrays have valid data

    # Extract aligned pairs (preserves temporal alignment)
    pupil1_clean = pupil1_truncated[valid_mask]
    pupil2_clean = pupil2_truncated[valid_mask]

    # Check if we have enough data
    if len(pupil1_clean) < 2 or len(pupil2_clean) < 2:
        print("❌ Error: Not enough valid data points for correlation analysis")
    else:
        # Z-score normalize both signals before cross-correlation
        # This accounts for different camera magnifications/orientations by comparing relative changes
        # Formula: z = (x - mean) / std
        pupil1_mean = np.mean(pupil1_clean)
        pupil1_std = np.std(pupil1_clean)
        pupil2_mean = np.mean(pupil2_clean)
        pupil2_std = np.std(pupil2_clean)
        
        if pupil1_std > 0 and pupil2_std > 0:
            pupil1_z = (pupil1_clean - pupil1_mean) / pupil1_std
            pupil2_z = (pupil2_clean - pupil2_mean) / pupil2_std
            print(f"Applied z-score normalization to pupil diameter signals (accounts for different camera magnifications)")
            print(f"  VideoData1: mean={pupil1_mean:.2f}, std={pupil1_std:.2f}")
            print(f"  VideoData2: mean={pupil2_mean:.2f}, std={pupil2_std:.2f}")
        else:
            print("⚠️ Warning: Zero variance detected, using raw signals (no normalization)")
            pupil1_z = pupil1_clean
            pupil2_z = pupil2_clean
        
        # Calculate cross-correlation using z-scored signals
        try:
            correlation = correlate(pupil1_z, pupil2_z, mode='full')
            
            # Calculate lags (in samples)
            lags = np.arange(-len(pupil2_z) + 1, len(pupil1_z))
            
            # Convert lags to time (assuming same sampling rate)
            dt = np.median(np.diff(SleapVideoData1['Seconds']))
            lag_times = lags * dt
            
            # Find peak correlation and corresponding lag
            peak_idx = np.argmax(correlation)
            peak_correlation = correlation[peak_idx]
            peak_lag_samples = lags[peak_idx]
            peak_lag_time = lag_times[peak_idx]
            peak_lag_time_display = peak_lag_time # for final QC figure 
            
            print(f"Peak lag (time): {peak_lag_time:.4f} seconds")

        
            # Normalize correlation to [-1, 1] range (for z-scored signals, this is standard normalization)
            norm_factor = np.sqrt(np.sum(pupil1_z**2) * np.sum(pupil2_z**2))
            if norm_factor > 0:
                correlation_normalized = correlation / norm_factor
                peak_correlation_normalized = correlation_normalized[peak_idx]
                print(f"Peak normalized correlation: {peak_correlation_normalized:.4f}")
            else:
                print("❌ Error: Cannot normalize correlation (zero variance)")
                correlation_normalized = correlation
                peak_correlation_normalized = 0
            
            # # Plot cross-correlation
            # fig.add_trace(
            #     go.Scatter(
            #         x=lag_times,
            #         y=correlation_normalized,
            #         mode='lines',
            #         name="Cross-Correlation",
            #         line=dict(color='green')
            #     ),
            #     row=2, col=1
            # )
            
            # # Add vertical line at peak
            # fig.add_vline(
            #     x=peak_lag_time,
            #     line_dash="dash",
            #     line_color="red",
            #     annotation_text=f"Peak: {peak_correlation_normalized:.3f}",
            #     row=2, col=1
            # )
            
        except Exception as e:
            print(f"❌ Error in cross-correlation calculation: {e}")
            # # Add empty trace to maintain plot structure
            # fig.add_trace(
            #     go.Scatter(
            #         x=[0], y=[0],
            #         mode='lines',
            #         name="Cross-Correlation (Error)",
            #         line=dict(color='gray')
            #     ),
            #     row=2, col=1
            # )

    # # Update axes labels
    # fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    # fig.update_yaxes(title_text="Pupil Diameter", row=1, col=1)
    # fig.update_xaxes(title_text="Lag (seconds)", row=2, col=1)
    # fig.update_yaxes(title_text="Normalized Correlation", row=2, col=1)

    # fig.update_layout(
    #     height=800,
    #     width=1000,
    #     title_text=f"SLEAP Pupil Diameter Analysis: Comparison & Cross-Correlation"
    # )

    # fig.show()

    # Additional correlation statistics
    if len(pupil1_clean) >= 2 and len(pupil2_clean) >= 2:
        try:
            # Calculate Pearson correlation coefficient on z-scored signals
            # Note: For z-scored signals, Pearson correlation is equivalent to the normalized cross-correlation at zero lag
            pearson_r, pearson_p = pearsonr(pupil1_z, pupil2_z)
            pearson_r_display = pearson_r
            pearson_p_display = pearson_p
            
            print(f"\n=== Additional Statistics ===")
            print(f"Pearson correlation coefficient: {pearson_r:.2f}")

            # Handle extremely small p-values
            if pearson_p < 1e-300:
                print(f'Pearson p-value: < 1e-300 (extremely significant)')
            else:
                print(f'Pearson p-value: {pearson_p:.5e}')
            
        except Exception as e:
            print(f"❌ Error in additional statistics: {e}")
            pearson_r_display = None
            pearson_p_display = None
    else:
        print("❌ Cannot calculate additional statistics - insufficient data")
else:
    print("Only one eye is present, no pupil diameter cross-correlation can be done")

# %%
# check if Second values match 1:1 between VideoData and SleapVideoData then merge them into VideoData
############################################################################################################

if VideoData1_Has_Sleap is True:
    if VideoData1['Seconds'].equals(SleapVideoData1['Seconds']) is False:
        print(f"❗ {get_eye_label('VideoData1')}: The 'Seconds' columns DO NOT correspond 1:1 between the two DataFrames. This should not happen")
    else:
        VideoData1 = VideoData1.merge(SleapVideoData1, on='Seconds', how='outer')
        del SleapVideoData1

if VideoData2_Has_Sleap is True:
    if VideoData2['Seconds'].equals(SleapVideoData2['Seconds']) is False:
        print(f"❗ {get_eye_label('VideoData2')}: The 'Seconds' columns DO NOT correspond 1:1 between the two DataFrames. This should not happen")
    else:
        VideoData2 = VideoData2.merge(SleapVideoData2, on='Seconds', how='outer')
        del SleapVideoData2
gc.collect()
None

# %%
# Compare SLEAP center.x and .y with fitted ellipse centre distributions for both VideoData1 and VideoData2
############################################################################################################

# ------------------------------------------------------------------
# 1) Compute correlations for VideoData1
# ------------------------------------------------------------------
if VideoData1_Has_Sleap is True:
    print(f"=== VideoData1 Analysis ===")
    slope_x1, intercept_x1, r_value_x1, p_value_x1, std_err_x1 = linregress(
        VideoData1["Ellipse.Center.X"], 
        VideoData1["center.x"]
    )
    r_squared_x1 = r_value_x1**2
    print(f"{get_eye_label('VideoData1')} - R^2 between center point and ellipse center X data: {r_squared_x1:.4f}")

    slope_y1, intercept_y1, r_value_y1, p_value_y1, std_err_y1 = linregress(
        VideoData1["Ellipse.Center.Y"], 
        VideoData1["center.y"]
    )
    r_squared_y1 = r_value_y1**2
    print(f"{get_eye_label('VideoData1')} - R^2 between center point and ellipse center Y data: {r_squared_y1:.4f}")

# ------------------------------------------------------------------
# 2) Compute correlations for VideoData2
# ------------------------------------------------------------------
if VideoData2_Has_Sleap is True:
    print(f"\n=== VideoData2 Analysis ===")
    slope_x2, intercept_x2, r_value_x2, p_value_x2, std_err_x2 = linregress(
        VideoData2["Ellipse.Center.X"], 
        VideoData2["center.x"]
    )
    r_squared_x2 = r_value_x2**2
    print(f"{get_eye_label('VideoData2')} - R^2 between center point and ellipse center X data: {r_squared_x2:.4f}")

    slope_y2, intercept_y2, r_value_y2, p_value_y2, std_err_y2 = linregress(
        VideoData2["Ellipse.Center.Y"], 
        VideoData2["center.y"]
    )
    r_squared_y2 = r_value_y2**2
    print(f"{get_eye_label('VideoData2')} - R^2 between center point and ellipse center Y data: {r_squared_y2:.4f}")

# ------------------------------------------------------------------
# 3) Center of Mass Analysis (if both VideoData1 and VideoData2 are available)
# ------------------------------------------------------------------
if VideoData1_Has_Sleap is True and VideoData2_Has_Sleap is True:
    print(f"\n=== Center of Mass Distance Analysis ===")
    
    # Calculate center of mass (mean) for VideoData1
    com_center_x1 = VideoData1["center.x"].mean()
    com_center_y1 = VideoData1["center.y"].mean()
    com_ellipse_x1 = VideoData1["Ellipse.Center.X"].mean()
    com_ellipse_y1 = VideoData1["Ellipse.Center.Y"].mean()
    
    # Calculate absolute distances for VideoData1
    dist_x1 = abs(com_center_x1 - com_ellipse_x1)
    dist_y1 = abs(com_center_y1 - com_ellipse_y1)
    
    print(f"\n{get_eye_label('VideoData1')}:")
    print(f"  Center of mass for center.x/y: ({com_center_x1:.4f}, {com_center_y1:.4f})")
    print(f"  Center of mass for Ellipse.Center.X/Y: ({com_ellipse_x1:.4f}, {com_ellipse_y1:.4f})")
    print(f"  Absolute distance in X: {dist_x1:.4f} pixels")
    print(f"  Absolute distance in Y: {dist_y1:.4f} pixels")
    
    # Calculate center of mass (mean) for VideoData2
    com_center_x2 = VideoData2["center.x"].mean()
    com_center_y2 = VideoData2["center.y"].mean()
    com_ellipse_x2 = VideoData2["Ellipse.Center.X"].mean()
    com_ellipse_y2 = VideoData2["Ellipse.Center.Y"].mean()
    
    # Calculate absolute distances for VideoData2
    dist_x2 = abs(com_center_x2 - com_ellipse_x2)
    dist_y2 = abs(com_center_y2 - com_ellipse_y2)
    
    print(f"\n{get_eye_label('VideoData2')}:")
    print(f"  Center of mass for center.x/y: ({com_center_x2:.4f}, {com_center_y2:.4f})")
    print(f"  Center of mass for Ellipse.Center.X/Y: ({com_ellipse_x2:.4f}, {com_ellipse_y2:.4f})")
    print(f"  Absolute distance in X: {dist_x2:.4f} pixels")
    print(f"  Absolute distance in Y: {dist_y2:.4f} pixels")

# ------------------------------------------------------------------
# 4) Re-center Ellipse.Center.X and Ellipse.Center.Y using median
# ------------------------------------------------------------------
print("\n=== Re-centering Ellipse.Center coordinates ===")

# Re-center VideoData1 Ellipse.Center coordinates
if VideoData1_Has_Sleap is True:
    # Calculate median
    median_ellipse_x1 = VideoData1["Ellipse.Center.X"].median()
    median_ellipse_y1 = VideoData1["Ellipse.Center.Y"].median()
    
    # Center the coordinates
    VideoData1["Ellipse.Center.X"] = VideoData1["Ellipse.Center.X"] - median_ellipse_x1
    VideoData1["Ellipse.Center.Y"] = VideoData1["Ellipse.Center.Y"] - median_ellipse_y1
    
    print(f"{get_eye_label('VideoData1')} - Re-centered Ellipse.Center using median: ({median_ellipse_x1:.4f}, {median_ellipse_y1:.4f})")

# Re-center VideoData2 Ellipse.Center coordinates
if VideoData2_Has_Sleap is True:
    # Calculate median
    median_ellipse_x2 = VideoData2["Ellipse.Center.X"].median()
    median_ellipse_y2 = VideoData2["Ellipse.Center.Y"].median()
    
    # Center the coordinates
    VideoData2["Ellipse.Center.X"] = VideoData2["Ellipse.Center.X"] - median_ellipse_x2
    VideoData2["Ellipse.Center.Y"] = VideoData2["Ellipse.Center.Y"] - median_ellipse_y2
    
    print(f"{get_eye_label('VideoData2')} - Re-centered Ellipse.Center using median: ({median_ellipse_x2:.4f}, {median_ellipse_y2:.4f})")



# %%
# Make and save summary QC plot using matplotlib with scatter plots for 2D distributions

# Initialize the statistics variables (these are calculated in Cell 11)
try:
    pearson_r_display
except NameError:
    pearson_r_display = None
    pearson_p_display = None
    peak_lag_time_display = None
    print("⚠️ Note: Statistics not found. They should be calculated in Cell 11.")

# Calculate correlation for Ellipse.Center.X between VideoData1 and VideoData2 (if both exist)
pearson_r_center = None
pearson_p_center = None
peak_lag_time_center = None

if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    # Get the Center.X data
    center_x1 = VideoData1['Ellipse.Center.X'].values
    center_x2 = VideoData2['Ellipse.Center.X'].values
    
    min_length = min(len(center_x1), len(center_x2))
    center_x1_truncated = center_x1[:min_length]
    center_x2_truncated = center_x2[:min_length]
    
    valid_mask1 = ~np.isnan(center_x1_truncated)
    valid_mask2 = ~np.isnan(center_x2_truncated)
    valid_mask = valid_mask1 & valid_mask2
    
    center_x1_clean = center_x1_truncated[valid_mask]
    center_x2_clean = center_x2_truncated[valid_mask]
    
    if len(center_x1_clean) >= 2 and len(center_x2_clean) >= 2:
        try:
            # Calculate Pearson correlation
            pearson_r_center, pearson_p_center = pearsonr(center_x1_clean, center_x2_clean)
            
            # Calculate cross-correlation for peak lag
            correlation = correlate(center_x1_clean, center_x2_clean, mode='full')
            lags = np.arange(-len(center_x2_clean) + 1, len(center_x1_clean))
            dt = np.median(np.diff(VideoData1['Seconds']))
            lag_times = lags * dt
            peak_idx = np.argmax(correlation)
            peak_lag_time_center = lag_times[peak_idx]
        except Exception as e:
            print(f"❌ Error calculating Ellipse.Center.X correlation stats: {e}")

# Create the QC summary figure using matplotlib with custom grid layout
fig = plt.figure(figsize=(20, 18))
fig.suptitle(str(data_path), fontsize=16, y=0.995)

# Create a grid layout:
# - Top row (full width): VideoData1 Time Series
# - Second row (full width): VideoData2 Time Series  
# - Third row (two columns): 2D scatter plots (VideoData1 left, VideoData2 right)
# - Fourth row (two columns): Pupil diameter (left), Ellipse.Center.X correlation (right)

gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Panel 1: VideoData1 center coordinates - Time Series (full width)
if VideoData1_Has_Sleap:
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(VideoData1_centered['Seconds'], VideoData1_centered['center.x'],
            linewidth=0.5, c='blue', alpha=0.6, label='center.x original')
    ax1.plot(VideoData1['Seconds'], VideoData1['Ellipse.Center.X'],
            linewidth=0.5, c='red', alpha=0.6, label='Ellipse Center.X')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (pixels)')
    ax1.set_title(f"{get_eye_label('VideoData1')} - center.X Time Series")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# Panel 2: VideoData2 center coordinates - Time Series (full width)
if VideoData2_Has_Sleap:
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(VideoData2_centered['Seconds'], VideoData2_centered['center.x'],
            linewidth=0.5, c='blue', alpha=0.6, label='center.x original')
    ax2.plot(VideoData2['Seconds'], VideoData2['Ellipse.Center.X'],
            linewidth=0.5, c='red', alpha=0.6, label='Ellipse Center.X')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (pixels)')
    ax2.set_title(f"{get_eye_label('VideoData2')} - center.X Time Series")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Panel 3: VideoData1 center coordinates - Scatter plot (left half)
if VideoData1_Has_Sleap:
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Ellipse.Center (blue)
    x_ellipse1 = VideoData1['Ellipse.Center.X'].to_numpy()
    y_ellipse1 = VideoData1['Ellipse.Center.Y'].to_numpy()
    mask1 = ~(np.isnan(x_ellipse1) | np.isnan(y_ellipse1))
    
    ax3.scatter(x_ellipse1[mask1], y_ellipse1[mask1],
               s=1, alpha=0.3, c='blue', label='Ellipse.Center')
    
    # Center (red) - from centered data
    x_center1 = VideoData1_centered['center.x'].to_numpy()
    y_center1 = VideoData1_centered['center.y'].to_numpy()
    mask2 = ~(np.isnan(x_center1) | np.isnan(y_center1))
    
    ax3.scatter(x_center1[mask2], y_center1[mask2],
               s=1, alpha=0.3, c='red', label='center.x original')
    
    ax3.set_xlabel('Center X (pixels)')
    ax3.set_ylabel('Center Y (pixels)')
    ax3.set_title(f"{get_eye_label('VideoData1')} - Center X-Y Distribution (center.X vs Ellipse)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add R² statistics for VideoData1 (bottom left)
    try:
        if 'r_squared_x1' in globals() and 'r_squared_y1' in globals():
            stats_text = f'R² X: {r_squared_x1:.2g}\nR² Y: {r_squared_y1:.2g}'
            ax3.text(0.02, 0.02, stats_text, transform=ax3.transAxes,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9, family='monospace')
    except:
        pass
    
    # Add center of mass distance for VideoData1 (bottom right)
    try:
        if 'dist_x1' in globals() and 'dist_y1' in globals():
            distance_text = f'COM Dist X: {dist_x1:.3g}\nCOM Dist Y: {dist_y1:.3g}'
            ax3.text(0.98, 0.02, distance_text, transform=ax3.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=9, family='monospace')
    except:
        pass

# Panel 4: VideoData2 center coordinates - Scatter plot (right half)
if VideoData2_Has_Sleap:
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Ellipse.Center (blue)
    x_ellipse2 = VideoData2['Ellipse.Center.X'].to_numpy()
    y_ellipse2 = VideoData2['Ellipse.Center.Y'].to_numpy()
    mask3 = ~(np.isnan(x_ellipse2) | np.isnan(y_ellipse2))
    
    ax4.scatter(x_ellipse2[mask3], y_ellipse2[mask3],
               s=1, alpha=0.3, c='blue', label='Ellipse.Center')
    
    # Center (red) - from centered data
    x_center2 = VideoData2_centered['center.x'].to_numpy()
    y_center2 = VideoData2_centered['center.y'].to_numpy()
    mask4 = ~(np.isnan(x_center2) | np.isnan(y_center2))
    
    ax4.scatter(x_center2[mask4], y_center2[mask4],
               s=1, alpha=0.3, c='red', label='center.X Center')
    
    ax4.set_xlabel('Center X (pixels)')
    ax4.set_ylabel('Center Y (pixels)')
    ax4.set_title(f"{get_eye_label('VideoData2')} - Center X-Y Distribution (center.X vs Ellipse)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add R² statistics for VideoData2 (bottom left)
    try:
        if 'r_squared_x2' in globals() and 'r_squared_y2' in globals():
            stats_text = f'R² X: {r_squared_x2:.2g}\nR² Y: {r_squared_y2:.2g}'
            ax4.text(0.02, 0.02, stats_text, transform=ax4.transAxes,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9, family='monospace')
    except:
        pass
    
    # Add center of mass distance for VideoData2 (bottom right)
    try:
        if 'dist_x2' in globals() and 'dist_y2' in globals():
            distance_text = f'COM Dist X: {dist_x2:.3g}\nCOM Dist Y: {dist_y2:.3g}'
            ax4.text(0.98, 0.02, distance_text, transform=ax4.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=9, family='monospace')
    except:
        pass

# Panel 5: Pupil diameter comparison (bottom left)
ax5 = fig.add_subplot(gs[3, 0])
if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    ax5.plot(VideoData1['Seconds'], VideoData1['Ellipse.Diameter.Filt'],
            linewidth=0.5, c='#FF7F00', alpha=0.6, label='VideoData1 Diameter')
    ax5.plot(VideoData2['Seconds'], VideoData2['Ellipse.Diameter.Filt'],
            linewidth=0.5, c='#9370DB', alpha=0.6, label='VideoData2 Diameter')
elif VideoData1_Has_Sleap:
    ax5.plot(VideoData1['Seconds'], VideoData1['Ellipse.Diameter.Filt'],
            linewidth=0.5, c='#FF7F00', alpha=0.6, label='VideoData1 Diameter')
elif VideoData2_Has_Sleap:
    ax5.plot(VideoData2['Seconds'], VideoData2['Ellipse.Diameter.Filt'],
            linewidth=0.5, c='#9370DB', alpha=0.6, label='VideoData2 Diameter')

ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Diameter (pixels)')
ax5.set_title('Pupil Diameter Comparison')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Add statistics text to Panel 5
if pearson_r_display is not None and pearson_p_display is not None and peak_lag_time_display is not None:
    stats_text = (f'Pearson r = {pearson_r_display:.4f}\n'
                  f'Pearson p = {pearson_p_display:.4e}\n'
                  f'Peak lag = {peak_lag_time_display:.4f} s')
    ax5.text(0.98, 0.98, stats_text, transform=ax5.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')
else:
    ax5.text(0.5, 0.5, 'Statistics not available\n(See Cell 11 for correlation analysis)', 
            transform=ax5.transAxes, ha='center', va='center', fontsize=10)

# Panel 6: Ellipse.Center.X comparison (bottom right) with dual y-axis
ax6 = fig.add_subplot(gs[3, 1])
ax6_twin = ax6.twinx()  # Create a second y-axis

if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    # Plot the individual traces
    ax6.plot(VideoData1['Seconds'], VideoData1['Ellipse.Center.X'],
            linewidth=0.5, c='#FF7F00', alpha=0.6, label='VideoData1 Ellipse.Center.X')
    ax6.plot(VideoData2['Seconds'], VideoData2['Ellipse.Center.X'],
            linewidth=0.5, c='#9370DB', alpha=0.6, label='VideoData2 Ellipse.Center.X')
    
    # Plot the difference on the right axis
    # Align the data to the same length and normalize for fair comparison
    min_length = min(len(VideoData1), len(VideoData2))
    
    # Normalize data (z-score) to account for different scales
    center_x1_aligned = VideoData1['Ellipse.Center.X'].iloc[:min_length]
    center_x2_aligned = VideoData2['Ellipse.Center.X'].iloc[:min_length]
    
    # Calculate mean and std for normalization
    mean1 = center_x1_aligned.mean()
    std1 = center_x1_aligned.std()
    mean2 = center_x2_aligned.mean()
    std2 = center_x2_aligned.std()
    
    # Normalize both datasets
    center_x1_norm = (center_x1_aligned - mean1) / std1
    center_x2_norm = (center_x2_aligned - mean2) / std2
    
    # Calculate difference of normalized data
    center_x_diff = center_x1_norm - center_x2_norm
    seconds_aligned = VideoData1['Seconds'].iloc[:min_length]
    ax6_twin.plot(seconds_aligned, center_x_diff,
                  linewidth=0.5, c='green', alpha=0.6, label='Difference (normalized)')
    
elif VideoData1_Has_Sleap:
    ax6.plot(VideoData1['Seconds'], VideoData1['Ellipse.Center.X'],
            linewidth=0.5, c='#FF7F00', alpha=0.6, label='VideoData1 Ellipse.Center.X')
elif VideoData2_Has_Sleap:
    ax6.plot(VideoData2['Seconds'], VideoData2['Ellipse.Center.X'],
            linewidth=0.5, c='#9370DB', alpha=0.6, label='VideoData2 Ellipse.Center.X')

ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Center X (pixels)', color='black')
ax6.set_title('Ellipse.Center.X Comparison')
ax6.tick_params(axis='y', labelcolor='black')
if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    ax6_twin.set_ylabel('Normalized Difference (z-score)', color='green')
    ax6_twin.tick_params(axis='y', labelcolor='green')

# Combine legends from both axes
lines1, labels1 = ax6.get_legend_handles_labels()
if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
else:
    ax6.legend(loc='upper left')

ax6.grid(True, alpha=0.3)

# Add statistics text to Panel 6
if pearson_r_center is not None and pearson_p_center is not None and peak_lag_time_center is not None:
    stats_text = (f'Pearson r = {pearson_r_center:.4f}\n'
                  f'Pearson p = {pearson_p_center:.4e}\n'
                  f'Peak lag = {peak_lag_time_center:.4f} s')
    ax6.text(0.98, 0.98, stats_text, transform=ax6.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')
else:
    ax6.text(0.5, 0.5, 'Statistics not available\n(both eyes required)', 
            transform=ax6.transAxes, ha='center', va='center', fontsize=10)

# Save as PDF (editable vector format)
save_path.mkdir(parents=True, exist_ok=True)
pdf_path = save_path / "Eye_data_QC.pdf"
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"✅ QC figure saved as PDF (editable): {pdf_path}")

# Also save as 600 dpi PNG (high-resolution for printing)
png_path = save_path / "Eye_data_QC.png"
plt.savefig(png_path, dpi=600, bbox_inches='tight', format='png')
print(f"✅ QC figure saved as PNG (600 dpi for printing): {png_path}")


plt.show()


# %%
# Create interactive time series plots using plotly for browser viewing
if plot_QC_timeseries:
    # Create subplots for the time series (3 rows now instead of 2)
    # Need to enable secondary_y for the third panel
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"{get_eye_label('VideoData1')} - center.X Time Series",
            f"{get_eye_label('VideoData2')} - center.X Time Series",
            "Ellipse.Center.X Comparison with Difference"
        ),
        specs=[[{}], [{}], [{"secondary_y": True}]]  # Enable secondary_y for row 3
    )

    # Panel 1: VideoData1 center coordinates - Time Series
    if VideoData1_Has_Sleap:
        fig.add_trace(go.Scatter(
            x=VideoData1_centered['Seconds'],
            y=VideoData1_centered['center.x'],
            mode='lines',
            name='center.x original',
            line=dict(color='blue', width=0.5),
            opacity=0.6
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=VideoData1['Seconds'],
            y=VideoData1['Ellipse.Center.X'],
            mode='lines',
            name='Ellipse Center.X',
            line=dict(color='red', width=0.5),
            opacity=0.6
        ), row=1, col=1)

    # Panel 2: VideoData2 center coordinates - Time Series
    if VideoData2_Has_Sleap:
        fig.add_trace(go.Scatter(
            x=VideoData2_centered['Seconds'],
            y=VideoData2_centered['center.x'],
            mode='lines',
            name='center.x original',
            line=dict(color='blue', width=0.5),
            opacity=0.6
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=VideoData2['Seconds'],
            y=VideoData2['Ellipse.Center.X'],
            mode='lines',
            name='Ellipse Center.X',
            line=dict(color='red', width=0.5),
            opacity=0.6
        ), row=2, col=1)

    # Panel 3: Ellipse.Center.X Comparison with difference
    if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
        # Plot the individual traces
        fig.add_trace(go.Scatter(
            x=VideoData1['Seconds'],
            y=VideoData1['Ellipse.Center.X'],
            mode='lines',
            name='VideoData1 Ellipse.Center.X',
            line=dict(color='#FF7F00', width=0.5),  # Orange
            opacity=0.6
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=VideoData2['Seconds'],
            y=VideoData2['Ellipse.Center.X'],
            mode='lines',
            name='VideoData2 Ellipse.Center.X',
            line=dict(color='#9370DB', width=0.5),  # Purple
            opacity=0.6
        ), row=3, col=1)
        
        # Plot the difference on secondary y-axis
        # Align the data to the same length and normalize for fair comparison
        min_length = min(len(VideoData1), len(VideoData2))
        
        # Normalize data (z-score) to account for different scales
        center_x1_aligned = VideoData1['Ellipse.Center.X'].iloc[:min_length]
        center_x2_aligned = VideoData2['Ellipse.Center.X'].iloc[:min_length]
        
        # Calculate mean and std for normalization
        mean1 = center_x1_aligned.mean()
        std1 = center_x1_aligned.std()
        mean2 = center_x2_aligned.mean()
        std2 = center_x2_aligned.std()
        
        # Normalize both datasets
        center_x1_norm = (center_x1_aligned - mean1) / std1
        center_x2_norm = (center_x2_aligned - mean2) / std2
        
        # Calculate difference of normalized data
        center_x_diff = center_x1_norm - center_x2_norm
        seconds_aligned = VideoData1['Seconds'].iloc[:min_length]
        
        fig.add_trace(go.Scatter(
            x=seconds_aligned,
            y=center_x_diff,
            mode='lines',
            name='Difference (normalized)',
            line=dict(color='green', width=0.5),
            opacity=0.6
        ), row=3, col=1, secondary_y=True)
        
    elif VideoData1_Has_Sleap:
        fig.add_trace(go.Scatter(
            x=VideoData1['Seconds'],
            y=VideoData1['Ellipse.Center.X'],
            mode='lines',
            name='VideoData1 Ellipse.Center.X',
            line=dict(color='#FF7F00', width=0.5),
            opacity=0.6
        ), row=3, col=1)
    elif VideoData2_Has_Sleap:
        fig.add_trace(go.Scatter(
            x=VideoData2['Seconds'],
            y=VideoData2['Ellipse.Center.X'],
            mode='lines',
            name='VideoData2 Ellipse.Center.X',
            line=dict(color='#9370DB', width=0.5),
            opacity=0.6
        ), row=3, col=1)

    # Update layout
    fig.update_layout(
        height=1200,  # Increased height for 3 panels
        title_text=f'{data_path} - Eye Tracking Time Series QC',
        showlegend=True,
        hovermode='x unified'
    )

    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Position (pixels)", row=1, col=1)
    fig.update_yaxes(title_text="Position (pixels)", row=2, col=1)
    fig.update_yaxes(title_text="Center X (pixels)", row=3, col=1)

    # Update secondary y-axis for difference plot
    if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
        fig.update_yaxes(title_text="Normalized Difference (z-score)", row=3, col=1, secondary_y=True)

    # Show in browser
    fig.show(renderer='browser')

    # Also save as HTML
    save_path.mkdir(parents=True, exist_ok=True)
    html_path = save_path / "Eye_data_QC_time_series.html"
    fig.write_html(html_path)
    print(f"✅ Interactive time series plot saved to: {html_path}")


# %%
# save as df to csv to be loaded in the photometry/harp/etc. analysis notebook 
############################################################################################################
# reindex to aeon datetime to be done in the other notebook
 
if VideoData1_Has_Sleap:
    # Save  DataFrame as CSV to proper path and filename
    save_path1 = save_path / "Video_Sleap_Data1" / "Video_Sleap_Data1_1904-01-01T00-00-00.csv"
    save_path1.parent.mkdir(parents=True, exist_ok=True)
    #save_path1.parent.mkdir(parents=True, exist_ok=True)
    VideoData1.to_csv(save_path1)

if VideoData2_Has_Sleap:
    # Save  DataFrame as CSV to proper path and filename
    save_path2 = save_path / "Video_Sleap_Data2" / "Video_Sleap_Data2_1904-01-01T00-00-00.csv"
    save_path2.parent.mkdir(parents=True, exist_ok=True)
    #save_path2.parent.mkdir(parents=True, exist_ok=True)
    VideoData2.to_csv(save_path2)


# %% [markdown]
# # Saccade detection
#

# %%
# # TEMPORARY - param setting 
# # Parameters for orienting vs compensatory saccade classification
# classify_orienting_compensatory = True  # Set to True to classify saccades as orienting vs compensatory
# bout_window = 1.5  # Time window (seconds) for grouping saccades into bouts
# pre_saccade_window = 0.3  # Time window (seconds) before saccade onset to analyze
# max_intersaccade_interval_for_classification = 5.0  # Maximum time (seconds) to extend post-saccade window until next saccade for classification
# pre_saccade_velocity_threshold = 50.0  # Velocity threshold (px/s) for detecting pre-saccade drift
# pre_saccade_drift_threshold = 10.0  # Position drift threshold (px) before saccade for compensatory classification
# post_saccade_variance_threshold = 100.0  # Position variance threshold (px²) after saccade for orienting classification
# post_saccade_position_change_threshold_percent = 50.0  # Position change threshold (% of saccade amplitude) - if post-saccade change > amplitude * this%, classify as compensatory

# # Adaptive threshold parameters (percentile-based)
# use_adaptive_thresholds = False  # Set to True to use adaptive thresholds based on feature distributions, False to use fixed thresholds
# adaptive_percentile_pre_velocity = 75  # Percentile for pre-saccade velocity threshold (upper percentile for compensatory detection)
# adaptive_percentile_pre_drift = 75  # Percentile for pre-saccade drift threshold (upper percentile for compensatory detection)
# adaptive_percentile_post_variance = 25  # Percentile for post-saccade variance threshold (lower percentile for orienting detection - low variance = stable)

# for saccades
refractory_period = 0.1  # sec
## Separate adaptive saccade threshold (k) for each video:
k1 = 6  # for VideoData1 (L)
k2 = 6  # for VideoData2 (R)

# for adaptive saccade threshold - Number of standard deviations (adjustable: 2-4 range works well) 
onset_offset_fraction = 0.2  # to determine saccade onset and offset, i.e. o.2 is 20% of the peak velocity
n_before = 10  # Number of points before detection peak to extract for peri-saccade-segments, points, so independent of FPS 
n_after = 30   # Number of points after detection peak to extract

# Additional saccade detection parameters
baseline_n_points = 5  # Number of points before threshold crossing to use for baseline calculation
saccade_smoothing_window = 5  # Rolling median window size for position smoothing (frames)
saccade_peak_width = 1  # Minimum peak width in samples for find_peaks (frames)

plot_saccade_detection_QC = True


# %%
saccade_results = {}

# Helper: map detected directions (upward/downward) to NT/TN based on eye assignment
# Left eye: upward→NT, downward→TN; Right eye: upward→TN, downward→NT
def get_direction_map_for_video(video_key):
    eye = video1_eye if video_key == 'VideoData1' else video2_eye
    if eye == 'L':
        return {'upward': 'NT', 'downward': 'TN'}
    else:
        return {'upward': 'TN', 'downward': 'NT'}

if VideoData1_Has_Sleap:
    print(f"\n🔎 === Source: ({get_eye_label('VideoData1')}) ===\n")
    df1 = VideoData1[['Ellipse.Center.X', 'Seconds']].copy()
    dir_map_v1 = get_direction_map_for_video('VideoData1')
    saccade_results['VideoData1'] = analyze_eye_video_saccades(
        df1, FPS_1, get_eye_label('VideoData1'),
        k=k1, refractory_period=refractory_period,
        onset_offset_fraction=onset_offset_fraction,
        n_before=n_before, n_after=n_after, baseline_n_points=baseline_n_points,
        saccade_smoothing_window=saccade_smoothing_window,
        saccade_peak_width=saccade_peak_width,
        upward_label=dir_map_v1['upward'],
        downward_label=dir_map_v1['downward'],
        classify_orienting_compensatory=classify_orienting_compensatory,
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
        adaptive_percentile_post_variance=adaptive_percentile_post_variance
    )


if VideoData2_Has_Sleap:
    print(f"\n🔎 === Source: ({get_eye_label('VideoData2')}) ===\n")
    df2 = VideoData2[['Ellipse.Center.X', 'Seconds']].copy()
    dir_map_v2 = get_direction_map_for_video('VideoData2')
    saccade_results['VideoData2'] = analyze_eye_video_saccades(
        df2, FPS_2, get_eye_label('VideoData2'),
        k=k2, refractory_period=refractory_period,
        onset_offset_fraction=onset_offset_fraction,
        n_before=n_before, n_after=n_after, baseline_n_points=baseline_n_points,
        saccade_smoothing_window=saccade_smoothing_window,
        saccade_peak_width=saccade_peak_width,
        upward_label=dir_map_v2['upward'],
        downward_label=dir_map_v2['downward'],
        classify_orienting_compensatory=classify_orienting_compensatory,
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
        adaptive_percentile_post_variance=adaptive_percentile_post_variance
    )

# %%
# ADAPTIVE THRESHOLD DIAGNOSTIC PLOTS (only if debug=True)
#-------------------------------------------------------------------------------
# Plot distributions of classification features to help determine meaningful adaptive thresholds
if debug and len(saccade_results) > 0:
    print("\n📊 Generating adaptive threshold diagnostic plots...")
    
    for video_key, res in saccade_results.items():
        all_saccades_df = res.get('all_saccades_df', pd.DataFrame())
        
        if len(all_saccades_df) == 0:
            print(f"⚠️  No saccades found for {get_eye_label(video_key)}, skipping diagnostic plots")
            continue
        
        # Filter out NaN values for plotting
        pre_vel = all_saccades_df['pre_saccade_mean_velocity'].dropna()
        pre_drift = all_saccades_df['pre_saccade_position_drift'].dropna()
        post_var = all_saccades_df['post_saccade_position_variance'].dropna()
        post_change = all_saccades_df['post_saccade_position_change'].dropna()
        amplitude = all_saccades_df['amplitude'].dropna()
        
        # Calculate post_change / amplitude ratio (for percentage threshold visualization)
        # Align by index to ensure matching
        aligned_indices = post_change.index.intersection(amplitude.index)
        post_change_aligned = post_change.loc[aligned_indices]
        amplitude_aligned = amplitude.loc[aligned_indices]
        post_change_ratio = (post_change_aligned / amplitude_aligned) * 100  # Convert to percentage
        
        # Calculate current thresholds for visualization
        if use_adaptive_thresholds:
            # Calculate adaptive thresholds from current data
            if len(pre_vel) >= 3:
                current_pre_vel_threshold = np.percentile(pre_vel, adaptive_percentile_pre_velocity)
            else:
                current_pre_vel_threshold = pre_saccade_velocity_threshold
            
            if len(pre_drift) >= 3:
                current_pre_drift_threshold = np.percentile(pre_drift, adaptive_percentile_pre_drift)
            else:
                current_pre_drift_threshold = pre_saccade_drift_threshold
            
            if len(post_var) >= 3:
                current_post_var_threshold = np.percentile(post_var, adaptive_percentile_post_variance)
            else:
                current_post_var_threshold = post_saccade_variance_threshold
        else:
            # Use fixed thresholds
            current_pre_vel_threshold = pre_saccade_velocity_threshold
            current_pre_drift_threshold = pre_saccade_drift_threshold
            current_post_var_threshold = post_saccade_variance_threshold
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Adaptive Threshold Diagnostic Plots: {get_eye_label(video_key)}\n'
                    f'(n={len(all_saccades_df)} saccades)', fontsize=14, fontweight='bold')
        
        # Plot 1: Pre-saccade mean velocity
        ax = axes[0, 0]
        if len(pre_vel) > 0:
            ax.hist(pre_vel, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(current_pre_vel_threshold, color='red', linestyle='--', linewidth=2, 
                      label=f'Threshold: {current_pre_vel_threshold:.2f} px/s')
            if use_adaptive_thresholds:
                ax.axvline(np.percentile(pre_vel, 50), color='gray', linestyle=':', linewidth=1, 
                          label=f'Median: {np.percentile(pre_vel, 50):.2f} px/s')
                ax.axvline(np.percentile(pre_vel, 75), color='orange', linestyle=':', linewidth=1, 
                          label=f'75th: {np.percentile(pre_vel, 75):.2f} px/s')
            ax.set_xlabel('Pre-saccade Mean Velocity (px/s)')
            ax.set_ylabel('Count')
            ax.set_title(f'Pre-saccade Velocity Distribution\n'
                        f'{"Adaptive" if use_adaptive_thresholds else "Fixed"} threshold at '
                        f'{adaptive_percentile_pre_velocity if use_adaptive_thresholds else "fixed"}th percentile')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Pre-saccade position drift
        ax = axes[0, 1]
        if len(pre_drift) > 0:
            ax.hist(pre_drift, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
            ax.axvline(current_pre_drift_threshold, color='red', linestyle='--', linewidth=2,
                      label=f'Threshold: {current_pre_drift_threshold:.2f} px')
            if use_adaptive_thresholds:
                ax.axvline(np.percentile(pre_drift, 50), color='gray', linestyle=':', linewidth=1,
                          label=f'Median: {np.percentile(pre_drift, 50):.2f} px')
                ax.axvline(np.percentile(pre_drift, 75), color='orange', linestyle=':', linewidth=1,
                          label=f'75th: {np.percentile(pre_drift, 75):.2f} px')
            ax.set_xlabel('Pre-saccade Position Drift (px)')
            ax.set_ylabel('Count')
            ax.set_title(f'Pre-saccade Drift Distribution\n'
                        f'{"Adaptive" if use_adaptive_thresholds else "Fixed"} threshold at '
                        f'{adaptive_percentile_pre_drift if use_adaptive_thresholds else "fixed"}th percentile')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Post-saccade position variance
        ax = axes[1, 0]
        if len(post_var) > 0:
            ax.hist(post_var, bins=50, alpha=0.7, color='plum', edgecolor='black')
            ax.axvline(current_post_var_threshold, color='red', linestyle='--', linewidth=2,
                      label=f'Threshold: {current_post_var_threshold:.2f} px²')
            if use_adaptive_thresholds:
                ax.axvline(np.percentile(post_var, 25), color='orange', linestyle=':', linewidth=1,
                          label=f'25th: {np.percentile(post_var, 25):.2f} px²')
                ax.axvline(np.percentile(post_var, 50), color='gray', linestyle=':', linewidth=1,
                          label=f'Median: {np.percentile(post_var, 50):.2f} px²')
            ax.set_xlabel('Post-saccade Position Variance (px²)')
            ax.set_ylabel('Count')
            ax.set_title(f'Post-saccade Variance Distribution\n'
                        f'{"Adaptive" if use_adaptive_thresholds else "Fixed"} threshold at '
                        f'{adaptive_percentile_post_variance if use_adaptive_thresholds else "fixed"}th percentile')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Post-saccade position change (as percentage of amplitude)
        ax = axes[1, 1]
        if len(post_change_ratio) > 0:
            ax.hist(post_change_ratio, bins=50, alpha=0.7, color='salmon', edgecolor='black')
            ax.axvline(post_saccade_position_change_threshold_percent, color='red', linestyle='--', 
                      linewidth=2, label=f'Threshold: {post_saccade_position_change_threshold_percent:.1f}%')
            ax.axvline(np.percentile(post_change_ratio, 50), color='gray', linestyle=':', linewidth=1,
                      label=f'Median: {np.percentile(post_change_ratio, 50):.1f}%')
            ax.axvline(np.percentile(post_change_ratio, 75), color='orange', linestyle=':', linewidth=1,
                      label=f'75th: {np.percentile(post_change_ratio, 75):.1f}%')
            ax.set_xlabel('Post-saccade Position Change / Amplitude (%)')
            ax.set_ylabel('Count')
            ax.set_title(f'Post-saccade Position Change Ratio\n'
                        f'Fixed threshold: {post_saccade_position_change_threshold_percent:.1f}% of amplitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n📈 Summary Statistics for {get_eye_label(video_key)}:")
        if len(pre_vel) > 0:
            print(f"  Pre-saccade velocity: mean={pre_vel.mean():.2f}, median={pre_vel.median():.2f}, "
                  f"std={pre_vel.std():.2f} px/s")
        if len(pre_drift) > 0:
            print(f"  Pre-saccade drift: mean={pre_drift.mean():.2f}, median={pre_drift.median():.2f}, "
                  f"std={pre_drift.std():.2f} px")
        if len(post_var) > 0:
            print(f"  Post-saccade variance: mean={post_var.mean():.2f}, median={post_var.median():.2f}, "
                  f"std={post_var.std():.2f} px²")
        if len(post_change_ratio) > 0:
            print(f"  Post-saccade change ratio: mean={post_change_ratio.mean():.1f}%, "
                  f"median={post_change_ratio.median():.1f}%, std={post_change_ratio.std():.1f}%")
        print()

# %%
# VISUALIZE ALL SACCADES - SIDE BY SIDE
#-------------------------------------------------------------------------------
# Plot all upward and downward saccades aligned by time with position and velocity traces

# Create figure with 4 columns in a single row: up pos, up vel, down pos, down vel

for video_key, res in saccade_results.items():
    dir_map = get_direction_map_for_video(video_key)
    label_up = dir_map['upward']
    label_down = dir_map['downward']

    upward_saccades_df = res['upward_saccades_df']
    downward_saccades_df = res['downward_saccades_df']
    peri_saccades = res['peri_saccades']
    upward_segments = res['upward_segments']
    downward_segments = res['downward_segments']
    # Any other variables you need...

    fig_all = make_subplots(
        rows=1, cols=4,
        shared_yaxes=False,  # Each panel can have different y-axis scale
        shared_xaxes=False,
        subplot_titles=(
            f'Position - {label_up} Saccades',
            f'Velocity - {label_up} Saccades',
            f'Position - {label_down} Saccades',
            f'Velocity - {label_down} Saccades'
        )
    )

    # Extract segments for each direction
    upward_segments_all = [seg for seg in peri_saccades if seg['saccade_direction'].iloc[0] == 'upward']
    downward_segments_all = [seg for seg in peri_saccades if seg['saccade_direction'].iloc[0] == 'downward']

    # Filter outliers
    upward_segments, upward_outliers_meta, upward_outlier_segments = sp.filter_outlier_saccades(upward_segments_all, 'upward')
    downward_segments, downward_outliers_meta, downward_outlier_segments = sp.filter_outlier_saccades(downward_segments_all, 'downward')

    if debug:
        print(f"Plotting {len(upward_segments)} {label_up} and {len(downward_segments)} {label_down} saccades...")
        if len(upward_outliers_meta) > 0 or len(downward_outliers_meta) > 0:
            print(f"   Excluded {len(upward_outliers_meta)} {label_up} outlier(s) and {len(downward_outliers_meta)} {label_down} outlier(s)")

    if debug and len(upward_outliers_meta) > 0:
        print(f"\n   {label_up} outliers (first 5):")
        for i, out in enumerate(upward_outliers_meta[:5]):
            pass
        if len(upward_outliers_meta) > 5:
            print(f"      ... and {len(upward_outliers_meta) - 5} more")

    if debug and len(downward_outliers_meta) > 0:
        print(f"\n   {label_down} outliers (first 5):")
        for i, out in enumerate(downward_outliers_meta[:5]):
            pass
        if len(downward_outliers_meta) > 5:
            print(f"      ... and {len(downward_outliers_meta) - 5} more")

    # Plot upward saccades
    for i, segment in enumerate(upward_segments):
        color_opacity = 0.15 if len(upward_segments) > 20 else 0.3
        
        # Position trace (using baselined values)
        fig_all.add_trace(
            go.Scatter(
                x=segment['Time_rel_threshold'],
                y=segment['X_smooth_baselined'],
                mode='lines',
                name=f'Up #{i+1}',
                line=dict(color='green', width=1),
                showlegend=False,
                opacity=color_opacity
            ),
            row=1, col=1
        )
        
        # Velocity trace
        fig_all.add_trace(
            go.Scatter(
                x=segment['Time_rel_threshold'],
                y=segment['vel_x_smooth'],
                mode='lines',
                name=f'Up #{i+1}',
                line=dict(color='green', width=1),
                        showlegend=False,
                opacity=color_opacity
            ),
            row=1, col=2
        )

    # Plot downward saccades
    for i, segment in enumerate(downward_segments):
        color_opacity = 0.15 if len(downward_segments) > 20 else 0.3
        
        # Position trace (using baselined values)
        fig_all.add_trace(
            go.Scatter(
                x=segment['Time_rel_threshold'],
                y=segment['X_smooth_baselined'],
                mode='lines',
                name=f'Down #{i+1}',
                line=dict(color='purple', width=1),
                showlegend=False,
                opacity=color_opacity
            ),
            row=1, col=3
        )
        
        # Velocity trace
        fig_all.add_trace(
            go.Scatter(
                x=segment['Time_rel_threshold'],
                y=segment['vel_x_smooth'],
                mode='lines',
                name=f'Down #{i+1}',
                line=dict(color='purple', width=1),
                showlegend=False,
                opacity=color_opacity
            ),
            row=1, col=4
        )

    # Add mean traces for reference
    if len(upward_segments) > 0:
        # Calculate mean for upward by aligning segments by Time_rel_threshold (not by array index)
        # Find the threshold crossing index (where Time_rel_threshold ≈ 0) for each segment and align based on that
        aligned_positions = []
        aligned_velocities = []
        aligned_times = []
        
        for seg in upward_segments:
            # Find index where Time_rel_threshold is closest to 0 (the threshold crossing)
            threshold_idx = np.abs(seg['Time_rel_threshold'].values).argmin()
            
            # Extract data centered on threshold crossing: n_before points before, threshold crossing, n_after points after
            start_idx = max(0, threshold_idx - n_before)
            end_idx = min(len(seg), threshold_idx + n_after + 1)
            
            # Extract aligned segment
            aligned_seg = seg.iloc[start_idx:end_idx].copy()
            
            # Find where the threshold crossing actually is within the extracted segment
            threshold_in_seg_idx = threshold_idx - start_idx
            
            # Ensure threshold crossing is at index n_before by padding at start if needed
            if threshold_in_seg_idx < n_before:
                # Need to pad at the start to align threshold crossing to index n_before
                pad_length = n_before - threshold_in_seg_idx
                # Estimate time step from original segment for padding
                if len(seg) > 1:
                    dt_est = np.diff(seg['Time_rel_threshold'].values).mean()
                else:
                    dt_est = 0.0083  # default estimate if we can't calculate
                
                # Create padding with NaN values
                pad_times = aligned_seg['Time_rel_threshold'].iloc[0] - dt_est * np.arange(pad_length, 0, -1)
                pad_df = pd.DataFrame({
                    'X_smooth_baselined': [np.nan] * pad_length,
                    'vel_x_smooth': [np.nan] * pad_length,
                    'Time_rel_threshold': pad_times
                })
                aligned_seg = pd.concat([pad_df, aligned_seg.reset_index(drop=True)], ignore_index=True)
            
            aligned_positions.append(aligned_seg['X_smooth_baselined'].values)
            aligned_velocities.append(aligned_seg['vel_x_smooth'].values)
            aligned_times.append(aligned_seg['Time_rel_threshold'].values)
        
        # Find minimum length after alignment
        min_length = min(len(pos) for pos in aligned_positions)
        max_length = max(len(pos) for pos in aligned_positions)
        
        if min_length != max_length and debug:
            print(f"⚠️  Warning: {label_up} segments have variable lengths after alignment ({min_length} to {max_length} points). Using minimum length {min_length}.")
        
        # Truncate all segments to same length and stack
        upward_positions = np.array([pos[:min_length] for pos in aligned_positions])
        upward_velocities = np.array([vel[:min_length] for vel in aligned_velocities])
        upward_times = aligned_times[0][:min_length]  # Use first segment's time values
        
        # Calculate mean across all segments (axis=0 means across segments, keeping time dimension)
        upward_mean_pos = np.mean(upward_positions, axis=0)
        upward_mean_vel = np.mean(upward_velocities, axis=0)
        
        fig_all.add_trace(
            go.Scatter(
                x=upward_times,
                y=upward_mean_pos,
                mode='lines',
                name=f'{label_up} Mean Position',
                line=dict(color='red', width=3)
            ),
            row=1, col=1
        )
        
        fig_all.add_trace(
            go.Scatter(
                x=upward_times,
                y=upward_mean_vel,
                mode='lines',
                name=f'{label_up} Mean Velocity',
                line=dict(color='red', width=3)
            ),
            row=1, col=2
        )

    if len(downward_segments) > 0:
        # Calculate mean for downward by taking mean across all segments at each index position
        # Find minimum length to handle variable-length segments
        min_length_down = min(len(seg) for seg in downward_segments)
        max_length_down = max(len(seg) for seg in downward_segments)
        
        if min_length_down != max_length_down and debug:
            print(f"⚠️  Warning: {label_down} segments have variable lengths ({min_length_down} to {max_length_down} points). Using minimum length {min_length_down}.")
        
        # Stack all segments as arrays (each row is one saccade, columns are time points)
        # Use only first min_length points to ensure all arrays have same shape
        downward_positions = np.array([seg['X_smooth_baselined'].values[:min_length_down] for seg in downward_segments])
        downward_velocities = np.array([seg['vel_x_smooth'].values[:min_length_down] for seg in downward_segments])
        downward_times = downward_segments[0]['Time_rel_threshold'].values[:min_length_down]  # Use first segment's time values
        
        # Calculate mean across all segments (axis=0 means across segments, keeping time dimension)
        downward_mean_pos = np.mean(downward_positions, axis=0)
        downward_mean_vel = np.mean(downward_velocities, axis=0)
        
        fig_all.add_trace(
            go.Scatter(
                x=downward_times,
                y=downward_mean_pos,
                mode='lines',
                name=f'{label_down} Mean Position',
                line=dict(color='purple', width=3)
            ),
            row=1, col=3
        )
        
        fig_all.add_trace(
            go.Scatter(
                x=downward_times,
                y=downward_mean_vel,
                mode='lines',
                name=f'{label_down} Mean Velocity',
                line=dict(color='purple', width=3)
            ),
            row=1, col=4
        )

    # Add vertical line at time=0 (saccade onset)
    for col in [1, 2, 3, 4]:
        fig_all.add_shape(
            type="line",
            x0=0, x1=0,
            y0=-999, y1=999,
            line=dict(color='black', width=1, dash='dash'),
            row=1, col=col
        )

    # Update layout
    fig_all.update_layout(
        title_text=f'All Detected Saccades Overlaid ({get_eye_label(video_key)}) - Position and Velocity Profiles<br><sub>Position traces are baselined (avg of points -{n_before} to -{n_before-5}). All traces in semi-transparent, mean in bold. Time=0 is threshold crossing. {n_before} points before, {n_after} after.</sub>',
        height=500,
        width=1600,
        showlegend=False
    )

    # Calculate x-axis limits based on actual min/max time values from segments (with small padding)
    # Y-axis will use auto-range (no explicit range setting)

    # Collect all time values for proper x-axis scaling
    all_upward_times = []
    all_downward_times = []

    for seg in upward_segments:
        all_upward_times.extend(seg['Time_rel_threshold'].values)

    for seg in downward_segments:
        all_downward_times.extend(seg['Time_rel_threshold'].values)

    # Set x-axis ranges using actual min/max from all segment times (with small padding to show all data)
    padding_factor = 0.02  # 2% padding on each side for readability
    if len(all_upward_times) > 0:
        up_x_range = np.max(all_upward_times) - np.min(all_upward_times)
        padding = up_x_range * padding_factor if up_x_range > 0 else 0.01
        up_x_min = np.min(all_upward_times) - padding
        up_x_max = np.max(all_upward_times) + padding
    else:
        up_x_min, up_x_max = -0.2, 0.4

    if len(all_downward_times) > 0:
        down_x_range = np.max(all_downward_times) - np.min(all_downward_times)
        padding = down_x_range * padding_factor if down_x_range > 0 else 0.01
        down_x_min = np.min(all_downward_times) - padding
        down_x_max = np.max(all_downward_times) + padding
    else:
        down_x_min, down_x_max = -0.2, 0.4

    # Calculate y-axis ranges separately for position and velocity from filtered data
    # Collect position and velocity values from filtered segments
    upward_pos_values = []
    downward_pos_values = []
    upward_vel_values = []
    downward_vel_values = []

    for seg in upward_segments:
        upward_pos_values.extend(seg['X_smooth_baselined'].values)
        # Filter out NaN values when collecting velocity data
        vel_values = seg['vel_x_smooth'].values
        upward_vel_values.extend(vel_values[~np.isnan(vel_values)])

    for seg in downward_segments:
        downward_pos_values.extend(seg['X_smooth_baselined'].values)
        # Filter out NaN values when collecting velocity data
        vel_values = seg['vel_x_smooth'].values
        downward_vel_values.extend(vel_values[~np.isnan(vel_values)])

    # Position: find min/max for upward and downward, use wider range for both panels in row 1
    if len(upward_pos_values) > 0 and len(downward_pos_values) > 0:
        up_pos_min = np.min(upward_pos_values)
        up_pos_max = np.max(upward_pos_values)
        down_pos_min = np.min(downward_pos_values)
        down_pos_max = np.max(downward_pos_values)
        # Use the wider range (smaller min, larger max)
        pos_min = min(up_pos_min, down_pos_min)
        pos_max = max(up_pos_max, down_pos_max)
    elif len(upward_pos_values) > 0:
        pos_min = np.min(upward_pos_values)
        pos_max = np.max(upward_pos_values)
    elif len(downward_pos_values) > 0:
        pos_min = np.min(downward_pos_values)
        pos_max = np.max(downward_pos_values)
    else:
        pos_min, pos_max = -50, 50

    # Velocity: find min/max for upward and downward, use wider range for both panels in row 2
    # Get min and max directly from all velocity traces being plotted, with padding to prevent clipping
    if len(upward_vel_values) > 0 and len(downward_vel_values) > 0:
        # Get actual min/max from all velocity values
        all_vel_min = min(np.min(upward_vel_values), np.min(downward_vel_values))
        all_vel_max = max(np.max(upward_vel_values), np.max(downward_vel_values))

    elif len(upward_vel_values) > 0:
        all_vel_min = np.min(upward_vel_values)
        all_vel_max = np.max(upward_vel_values)

    elif len(downward_vel_values) > 0:
        all_vel_min = np.min(downward_vel_values)
        all_vel_max = np.max(downward_vel_values)

    else:
        all_vel_min, all_vel_max = -1000, 1000
        if debug:
            print(f"   ⚠️  No velocity values found, using default range: [{all_vel_min:.2f}, {all_vel_max:.2f}] px/s")

    # Add padding to prevent clipping (20% padding on each side)
    vel_range = all_vel_max - all_vel_min
    if vel_range > 0:
        padding = vel_range * 0.20  # 20% padding
        vel_min = all_vel_min - padding
        vel_max = all_vel_max + padding
    else:
        # If range is zero or very small, use default padding
        vel_min = all_vel_min - 1.0
        vel_max = all_vel_max + 1.0

    # Update axes - x-axis with ranges, y-axis with explicit ranges based on filtered data
    fig_all.update_xaxes(title_text="Time relative to threshold crossing (s)", range=[up_x_min, up_x_max], row=1, col=2)
    fig_all.update_xaxes(title_text="Time relative to threshold crossing (s)", range=[down_x_min, down_x_max], row=1, col=4)
    fig_all.update_xaxes(title_text="", range=[up_x_min, up_x_max], row=1, col=1)
    fig_all.update_xaxes(title_text="", range=[down_x_min, down_x_max], row=1, col=3)

    # Set explicit y-axis ranges - position panels share same range, velocity panels share same range
    fig_all.update_yaxes(title_text="X Position (px)", range=[pos_min, pos_max], row=1, col=1)
    fig_all.update_yaxes(title_text="X Position (px)", range=[pos_min, pos_max], row=1, col=3)
    fig_all.update_yaxes(title_text="Velocity (px/s)", range=[vel_min, vel_max], row=1, col=2)
    fig_all.update_yaxes(title_text="Velocity (px/s)", range=[vel_min, vel_max], row=1, col=4)


    fig_all.show()

    # Print statistics
    if debug:
        print(f"\n=== OVERLAY SUMMARY ===")
        if len(upward_segments) > 0:
            up_amps = [seg['saccade_amplitude'].iloc[0] for seg in upward_segments]
            up_durs = [seg['saccade_duration'].iloc[0] for seg in upward_segments]
            print(f"{label_up} saccades: {len(upward_segments)}")
            print(f"  Mean amplitude: {np.mean(up_amps):.2f} px")
            print(f"  Mean duration: {np.mean(up_durs):.3f} s")

        if len(downward_segments) > 0:
            down_amps = [seg['saccade_amplitude'].iloc[0] for seg in downward_segments]
            down_durs = [seg['saccade_duration'].iloc[0] for seg in downward_segments]
            print(f"{label_down} saccades: {len(downward_segments)}")
            print(f"  Mean amplitude: {np.mean(down_amps):.2f} px")
            print(f"  Mean duration: {np.mean(down_durs):.3f} s")

        print(f"\n⏱️  Time alignment: All saccades aligned to threshold crossing (Time_rel_threshold=0)")
        if len(all_upward_times) > 0 and len(all_downward_times) > 0:
            # Use the wider range for reporting
            overall_x_min = min(np.min(all_upward_times), np.min(all_downward_times))
            overall_x_max = max(np.max(all_upward_times), np.max(all_downward_times))
            print(f"📏 Window: {n_before} points before + {n_after} points after threshold crossing (actual time range: {overall_x_min:.3f} to {overall_x_max:.3f} s)")
        elif len(all_upward_times) > 0:
            print(f"📏 Window: {n_before} points before + {n_after} points after threshold crossing ({label_up} actual time range: {np.min(all_upward_times):.3f} to {np.max(all_upward_times):.3f} s)")
        elif len(all_downward_times) > 0:
            print(f"📏 Window: {n_before} points before + {n_after} points after threshold crossing ({label_down} actual time range: {np.min(all_downward_times):.3f} to {np.max(all_downward_times):.3f} s)")
        else:
            print(f"📏 Window: {n_before} points before + {n_after} points after threshold crossing")



# %%
# SACCADE AMPLITUDE QC VISUALIZATION
#-------------------------------------------------------------------------------
# 1. Distribution of saccade amplitudes
# 2. Correlation between saccade amplitude and duration
# 3. Peri-saccade segments colored by amplitude (outlier detection)

for video_key, res in saccade_results.items():
    dir_map = get_direction_map_for_video(video_key)
    label_up = dir_map['upward']
    label_down = dir_map['downward']

    upward_saccades_df = res['upward_saccades_df']
    downward_saccades_df = res['downward_saccades_df']
    peri_saccades = res['peri_saccades']   
    upward_segments = res['upward_segments']
    downward_segments = res['downward_segments']
    # Any other variables you need...

    fig_qc = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f'Amplitude Distribution - {label_up} Saccades', 
            f'Amplitude Distribution - {label_down} Saccades',
            f'Amplitude vs Duration - {label_up} Saccades',
            f'Amplitude vs Duration - {label_down} Saccades',
            f'Peri-Saccade Segments - {label_up} (colored by amplitude)',
            f'Peri-Saccade Segments - {label_down} (colored by amplitude)'
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.1,
        row_heights=[0.25, 0.25, 0.5]  # Make segment plots larger
    )

    # 1. Amplitude distributions
    if len(upward_saccades_df) > 0:
        # Histogram for upward saccades
        fig_qc.add_trace(
            go.Histogram(
                x=upward_saccades_df['amplitude'],
                nbinsx=50,
                name=f'{label_up}',
                marker_color='red',
                opacity=0.6
            ),
            row=1, col=1
        )
        
        # Scatter plot for upward saccades
        fig_qc.add_trace(
            go.Scatter(
                x=upward_saccades_df['duration'],
                y=upward_saccades_df['amplitude'],
                mode='markers',
                name=f'{label_up}',
                marker=dict(color='red', size=6, opacity=0.6)
            ),
            row=2, col=1
        )
        
        # Add correlation line for upward saccades
        corr_up = upward_saccades_df[['amplitude', 'duration']].corr().iloc[0, 1]
        z_up = np.polyfit(upward_saccades_df['duration'], upward_saccades_df['amplitude'], 1)
        p_up = np.poly1d(z_up)
        fig_qc.add_trace(
            go.Scatter(
                x=upward_saccades_df['duration'],
                y=p_up(upward_saccades_df['duration']),
                mode='lines',
                name=f'R={corr_up:.2f}',
                line=dict(color='darkgreen', width=2),
                showlegend=False
            ),
            row=2, col=1
        )

    if len(downward_saccades_df) > 0:
        # Histogram for downward saccades
        fig_qc.add_trace(
            go.Histogram(
                x=downward_saccades_df['amplitude'],
                nbinsx=50,
                name=f'{label_down}',
                marker_color='purple',
                opacity=0.6
            ),
            row=1, col=2
        )
        
        # Scatter plot for downward saccades
        fig_qc.add_trace(
            go.Scatter(
                x=downward_saccades_df['duration'],
                y=downward_saccades_df['amplitude'],
                mode='markers',
                name=f'{label_down}',
                marker=dict(color='purple', size=6, opacity=0.6)
            ),
            row=2, col=2
        )
        
        # Add correlation line for downward saccades
        corr_down = downward_saccades_df[['amplitude', 'duration']].corr().iloc[0, 1]
        z_down = np.polyfit(downward_saccades_df['duration'], downward_saccades_df['amplitude'], 1)
        p_down = np.poly1d(z_down)
        fig_qc.add_trace(
            go.Scatter(
                x=downward_saccades_df['duration'],
                y=p_down(downward_saccades_df['duration']),
                mode='lines',
                name=f'R={corr_down:.2f}',
                line=dict(color='darkviolet', width=2),
                showlegend=False
            ),
            row=2, col=2
        )

    # 3. Plot peri-saccade segments colored by amplitude
    # Reuse already-extracted and baselined segments from peri_saccades (no re-extraction or re-baselining)

    # Extract upward and downward segments for QC visualization from already-baselined peri_saccades
    # (Removed extract_qc_segments function - segments are now baselined only once during initial extraction)
    if 'peri_saccades' in globals() and len(peri_saccades) > 0:
        upward_segments_all = [seg for seg in peri_saccades if seg['saccade_direction'].iloc[0] == 'upward']
        downward_segments_all = [seg for seg in peri_saccades if seg['saccade_direction'].iloc[0] == 'downward']
    else:
        upward_segments_all = []
        downward_segments_all = []

    # Plot upward segments
    if len(upward_segments_all) > 0:
        upward_amplitudes = [seg['saccade_amplitude'].iloc[0] for seg in upward_segments_all]
        upward_colors, upward_min_amp, upward_max_amp = sp.get_color_mapping(upward_amplitudes)
        
        for i, (segment, color) in enumerate(zip(upward_segments_all, upward_colors)):
            fig_qc.add_trace(
                go.Scatter(
                    x=segment['Time_rel_threshold'],
                    y=segment['X_smooth_baselined'],
                    mode='lines',
                    name=f'{label_up} #{i+1}',
                    line=dict(color=color, width=1.5),
                    showlegend=False,
                    opacity=0.7,
                    hovertemplate=f'Amplitude: {segment["saccade_amplitude"].iloc[0]:.2f} px<br>' +
                                'Time: %{x:.3f} s<br>' +
                                'Position: %{y:.2f} px<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Add dummy trace for colorbar (hidden but provides colorbar)
        fig_qc.add_trace(
            go.Scatter(
                x=[None],  # Hidden trace
                y=[None],
                mode='markers',
                marker=dict(
                    size=1,
                    color=[upward_min_amp, upward_max_amp],
                    colorscale='Plasma',
                    cmin=upward_min_amp,
                    cmax=upward_max_amp,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=f"Amplitude ({label_up})", side="right"),
                        x=0.47,  # Position to the right of the left column plot
                        xpad=10,
                        len=0.45,  # Set colorbar length relative to subplot
                        y=0.5,  # Center vertically on the subplot
                        yanchor="middle"
                    )
                ),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=3, col=1
        )

    # Plot downward segments
    if len(downward_segments_all) > 0:
        downward_amplitudes = [seg['saccade_amplitude'].iloc[0] for seg in downward_segments_all]
        downward_colors, downward_min_amp, downward_max_amp = sp.get_color_mapping(downward_amplitudes)
        
        for i, (segment, color) in enumerate(zip(downward_segments_all, downward_colors)):
            fig_qc.add_trace(
                go.Scatter(
                    x=segment['Time_rel_threshold'],
                    y=segment['X_smooth_baselined'],
                    mode='lines',
                    name=f'{label_down} #{i+1}',
                    line=dict(color=color, width=1.5),
                    showlegend=False,
                    opacity=0.7,
                    hovertemplate=f'Amplitude: {segment["saccade_amplitude"].iloc[0]:.2f} px<br>' +
                                'Time: %{x:.3f} s<br>' +
                                'Position: %{y:.2f} px<extra></extra>'
                ),
                row=3, col=2
            )
        
        # Add dummy trace for colorbar (hidden but provides colorbar)
        fig_qc.add_trace(
            go.Scatter(
                x=[None],  # Hidden trace
                y=[None],
                mode='markers',
                marker=dict(
                    size=1,
                    color=[downward_min_amp, downward_max_amp],
                    colorscale='Plasma',
                    cmin=downward_min_amp,
                    cmax=downward_max_amp,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=f"Amplitude ({label_down})", side="right"),
                        x=0.97,  # Position to the right of the right column plot
                        xpad=10,
                        len=0.45,  # Set colorbar length relative to subplot
                        y=0.5,  # Center vertically on the subplot
                        yanchor="middle"
                    )
                ),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=3, col=2
        )

    # Update layout
    fig_qc.update_layout(
        title_text=f'Saccade Amplitude QC: Distributions, Correlations, and Segments (Outlier Detection, {get_eye_label(video_key)})',
        height=1200,
        showlegend=False
    )

    # Update axes labels
    fig_qc.update_xaxes(title_text="Amplitude (px)", row=1, col=1)
    fig_qc.update_xaxes(title_text="Amplitude (px)", row=1, col=2)
    fig_qc.update_xaxes(title_text="Duration (s)", row=2, col=1)
    fig_qc.update_xaxes(title_text="Duration (s)", row=2, col=2)
    fig_qc.update_xaxes(title_text="Time relative to threshold crossing (s)", row=3, col=1)
    fig_qc.update_xaxes(title_text="Time relative to threshold crossing (s)", row=3, col=2)
    fig_qc.update_yaxes(title_text="Count", row=1, col=1)
    fig_qc.update_yaxes(title_text="Count", row=1, col=2)
    fig_qc.update_yaxes(title_text="Amplitude (px)", row=2, col=1)
    fig_qc.update_yaxes(title_text="Amplitude (px)", row=2, col=2)
    fig_qc.update_yaxes(title_text="Position (baselined, px)", row=3, col=1)
    fig_qc.update_yaxes(title_text="Position (baselined, px)", row=3, col=2)

    fig_qc.show()

    # Print correlation statistics
    if debug:
        print("\n=== SACCADE AMPLITUDE-DURATION CORRELATION ===\n")
        if upward_saccades_df is not None and len(upward_saccades_df) > 0:
            print(f"{get_eye_label(video_key)} saccades (n={len(upward_saccades_df)}):")
            print(f"  Correlation (amplitude vs duration): {corr_up:.3f}")
            print(f"  Mean amplitude: {upward_saccades_df['amplitude'].mean():.2f} px")
            print(f"  Mean duration: {upward_saccades_df['duration'].mean():.3f} s")
            print(f"  Amp range: {upward_saccades_df['amplitude'].min():.2f} - {upward_saccades_df['amplitude'].max():.2f} px")

        if downward_saccades_df is not None and len(downward_saccades_df) > 0:
            print(f"\n{get_eye_label(video_key)} saccades (n={len(downward_saccades_df)}):")
            print(f"  Correlation (amplitude vs duration): {corr_down:.3f}")
            print(f"  Mean amplitude: {downward_saccades_df['amplitude'].mean():.2f} px")
            print(f"  Mean duration: {downward_saccades_df['duration'].mean():.3f} s")
            print(f"  Amp range: {downward_saccades_df['amplitude'].min():.2f} - {downward_saccades_df['amplitude'].max():.2f} px")



# %%
# VISUALIZE DETECTED SACCADES (Adaptive Method)
#-------------------------------------------------------------------------------
# Create overlay plot showing detected saccades with duration lines and peak arrows
if plot_saccade_detection_QC:
    for video_key, res in saccade_results.items():
        dir_map = get_direction_map_for_video(video_key)
        label_up = dir_map['upward']
        label_down = dir_map['downward']

        upward_saccades_df = res['upward_saccades_df']
        downward_saccades_df = res['downward_saccades_df']
        peri_saccades = res['peri_saccades']   
        upward_segments = res['upward_segments']
        downward_segments = res['downward_segments']
        # Any other variables you need...

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('X Position (px)', 'Velocity (px/s) with Detected Saccades')
        )

        # Add smoothed X position to the first subplot
        fig.add_trace(
            go.Scatter(
                x=res['df']['Seconds'],
                y=res['df']['X_smooth'],
                mode='lines',
                name='Smoothed X',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # Add smoothed velocity to the second subplot
        fig.add_trace(
            go.Scatter(
                x=res['df']['Seconds'],
                y=res['df']['vel_x_smooth'],
                mode='lines',
                name='Smoothed Velocity',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )

        # Add adaptive threshold lines for reference
        fig.add_hline(
            y=res['vel_thresh'],
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text=f"Adaptive threshold (±{res['vel_thresh']:.0f} px/s)",
            row=2, col=1
        )

        fig.add_hline(
            y=-res['vel_thresh'],
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            row=2, col=1
        )

        # Set offset for saccade indicator lines (above the trace for upward, below for downward)
        vel_max = res['df']['vel_x_smooth'].max()
        vel_min = res['df']['vel_x_smooth'].min()
        vel_range = vel_max - vel_min
        line_offset = vel_range * 0.15  # 15% of velocity range

        # Plot upward saccades with duration lines and peak arrows
        if len(upward_saccades_df) > 0:
            for idx, row in upward_saccades_df.iterrows():
                start_time = row['start_time']
                end_time = row['end_time']
                peak_time = row['time']
                peak_velocity = row['velocity']
                
                # Draw horizontal line spanning the saccade duration
                # Line is positioned above the velocity trace
                y_line_pos = vel_max + line_offset
                
                fig.add_shape(
                    type="line",
                    x0=start_time, y0=y_line_pos,
                    x1=end_time, y1=y_line_pos,
                    line=dict(color='green', width=3),
                    row=2, col=1
                )
                
                # Add arrow annotation at the peak position pointing to the actual peak velocity
                # Arrow points from the line to the peak velocity value on the velocity trace
                y_arrow_start = y_line_pos
                y_arrow_end = peak_velocity
                
                fig.add_annotation(
                    x=peak_time,
                    y=y_arrow_start,
                    ax=0,
                    ay=y_arrow_end - y_arrow_start,  # arrow points to peak velocity
                    arrowhead=2,  # filled arrowhead
                    arrowsize=2,
                    arrowwidth=2,
                    arrowcolor='green',
                    row=2, col=1,
                    showarrow=True
                )
            
            # Add legend entry for upward saccades
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    name=f'{label_up} Saccades (duration lines)',
                    marker=dict(symbol='line-ns', size=15, color='green', line=dict(width=3))
                ),
                row=2, col=1
            )

        # Plot downward saccades with duration lines and peak arrows
        if len(downward_saccades_df) > 0:
            for idx, row in downward_saccades_df.iterrows():
                start_time = row['start_time']
                end_time = row['end_time']
                peak_time = row['time']
                peak_velocity = row['velocity']
                
                # Draw horizontal line spanning the saccade duration
                # Line is positioned below the velocity trace
                y_line_pos = vel_min - line_offset
                
                fig.add_shape(
                    type="line",
                    x0=start_time, y0=y_line_pos,
                    x1=end_time, y1=y_line_pos,
                    line=dict(color='purple', width=3),
                    row=2, col=1
                )
                
                # Add arrow annotation at the peak position pointing to the actual peak velocity
                # Arrow points from the line to the peak velocity value on the velocity trace
                y_arrow_start = y_line_pos
                y_arrow_end = peak_velocity
                
                fig.add_annotation(
                    x=peak_time,
                    y=y_arrow_start,
                    ax=0,
                    ay=y_arrow_end - y_arrow_start,  # arrow points to peak velocity
                    arrowhead=2,  # filled arrowhead
                    arrowsize=2,
                    arrowwidth=2,
                    arrowcolor='purple',
                    row=2, col=1,
                    showarrow=True
                )
            
            # Add legend entry for downward saccades
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    name=f'{label_down} Saccades (duration lines)',
                    marker=dict(symbol='line-ns', size=15, color='purple', line=dict(width=3))
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=f'Detected Saccades ({get_eye_label(video_key)}): Duration Lines + Peak Arrows (QC Visualization)',
            height=600,
            showlegend=True,
            legend=dict(x=0.01, y=0.99)
        )

        # Update axes
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="X Position (px)", row=1, col=1)
        fig.update_yaxes(title_text="Velocity (px/s)", row=2, col=1)

        fig.show()


# %%
# VISUALIZE AND ANALYZE SACCADE CLASSIFICATION (Orienting vs Compensatory)
#-------------------------------------------------------------------------------
# Create validation plots and statistical comparisons for saccade classification

for video_key, res in saccade_results.items():
    dir_map = get_direction_map_for_video(video_key)
    label_up = dir_map['upward']
    label_down = dir_map['downward']
    
    all_saccades_df = res.get('all_saccades_df', pd.DataFrame())
    
    if len(all_saccades_df) == 0:
        print(f"\n⚠️ No saccades found for {get_eye_label(video_key)}")
        continue
    
    # Check if classification was performed
    if 'saccade_type' not in all_saccades_df.columns:
        print(f"\n⚠️ Classification not performed for {get_eye_label(video_key)}")
        continue
    
    orienting_saccades = all_saccades_df[all_saccades_df['saccade_type'] == 'orienting']
    compensatory_saccades = all_saccades_df[all_saccades_df['saccade_type'] == 'compensatory']
    
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION ANALYSIS: {get_eye_label(video_key)}")
    print(f"{'='*80}")
    
    # Statistical comparisons
    from scipy import stats
    
    print(f"\n📊 Statistical Comparisons:")
    print(f"  Orienting saccades: {len(orienting_saccades)}")
    print(f"  Compensatory saccades: {len(compensatory_saccades)}")
    
    if len(orienting_saccades) > 0 and len(compensatory_saccades) > 0:
        # Amplitude comparison
        orienting_amps = orienting_saccades['amplitude'].values
        compensatory_amps = compensatory_saccades['amplitude'].values
        amp_stat, amp_p = stats.mannwhitneyu(orienting_amps, compensatory_amps, alternative='two-sided')
        print(f"\n  Amplitude (px):")
        print(f"    Orienting: {orienting_amps.mean():.2f} ± {orienting_amps.std():.2f} (median: {np.median(orienting_amps):.2f})")
        print(f"    Compensatory: {compensatory_amps.mean():.2f} ± {compensatory_amps.std():.2f} (median: {np.median(compensatory_amps):.2f})")
        print(f"    Mann-Whitney U test: U={amp_stat:.1f}, p={amp_p:.4f}")
        
        # Duration comparison
        orienting_durs = orienting_saccades['duration'].values
        compensatory_durs = compensatory_saccades['duration'].values
        dur_stat, dur_p = stats.mannwhitneyu(orienting_durs, compensatory_durs, alternative='two-sided')
        print(f"\n  Duration (s):")
        print(f"    Orienting: {orienting_durs.mean():.3f} ± {orienting_durs.std():.3f} (median: {np.median(orienting_durs):.3f})")
        print(f"    Compensatory: {compensatory_durs.mean():.3f} ± {compensatory_durs.std():.3f} (median: {np.median(compensatory_durs):.3f})")
        print(f"    Mann-Whitney U test: U={dur_stat:.1f}, p={dur_p:.4f}")
        
        # Pre-saccade velocity comparison
        orienting_pre_vel = orienting_saccades['pre_saccade_mean_velocity'].values
        compensatory_pre_vel = compensatory_saccades['pre_saccade_mean_velocity'].values
        pre_vel_stat, pre_vel_p = stats.mannwhitneyu(orienting_pre_vel, compensatory_pre_vel, alternative='two-sided')
        print(f"\n  Pre-saccade velocity (px/s):")
        print(f"    Orienting: {orienting_pre_vel.mean():.2f} ± {orienting_pre_vel.std():.2f} (median: {np.median(orienting_pre_vel):.2f})")
        print(f"    Compensatory: {compensatory_pre_vel.mean():.2f} ± {compensatory_pre_vel.std():.2f} (median: {np.median(compensatory_pre_vel):.2f})")
        print(f"    Mann-Whitney U test: U={pre_vel_stat:.1f}, p={pre_vel_p:.4f}")
        
        # Pre-saccade drift comparison
        orienting_pre_drift = orienting_saccades['pre_saccade_position_drift'].values
        compensatory_pre_drift = compensatory_saccades['pre_saccade_position_drift'].values
        pre_drift_stat, pre_drift_p = stats.mannwhitneyu(orienting_pre_drift, compensatory_pre_drift, alternative='two-sided')
        print(f"\n  Pre-saccade position drift (px):")
        print(f"    Orienting: {orienting_pre_drift.mean():.2f} ± {orienting_pre_drift.std():.2f} (median: {np.median(orienting_pre_drift):.2f})")
        print(f"    Compensatory: {compensatory_pre_drift.mean():.2f} ± {compensatory_pre_drift.std():.2f} (median: {np.median(compensatory_pre_drift):.2f})")
        print(f"    Mann-Whitney U test: U={pre_drift_stat:.1f}, p={pre_drift_p:.4f}")
        
        # Post-saccade variance comparison
        orienting_post_var = orienting_saccades['post_saccade_position_variance'].values
        compensatory_post_var = compensatory_saccades['post_saccade_position_variance'].values
        post_var_stat, post_var_p = stats.mannwhitneyu(orienting_post_var, compensatory_post_var, alternative='two-sided')
        print(f"\n  Post-saccade position variance (px²):")
        print(f"    Orienting: {orienting_post_var.mean():.2f} ± {orienting_post_var.std():.2f} (median: {np.median(orienting_post_var):.2f})")
        print(f"    Compensatory: {compensatory_post_var.mean():.2f} ± {compensatory_post_var.std():.2f} (median: {np.median(compensatory_post_var):.2f})")
        print(f"    Mann-Whitney U test: U={post_var_stat:.1f}, p={post_var_p:.4f}")
        
        # Bout size for compensatory saccades
        if len(compensatory_saccades) > 0:
            bout_sizes = compensatory_saccades['bout_size'].values
            print(f"\n  Bout size (compensatory saccades only):")
            print(f"    Mean: {bout_sizes.mean():.2f} ± {bout_sizes.std():.2f} saccades")
            print(f"    Range: {bout_sizes.min():.0f} - {bout_sizes.max():.0f} saccades")
            print(f"    Median: {np.median(bout_sizes):.0f} saccades")
    else:
        print(f"  ⚠️ Cannot perform statistical comparisons - need both types present")
    
    # Create visualization figure
    fig_class = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Amplitude Distribution',
            'Duration Distribution',
            'Pre-saccade Velocity Distribution',
            'Pre-saccade Position Drift',
            'Post-saccade Position Variance',
            'Bout Size Distribution (Compensatory)'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Row 1, Col 1: Amplitude distributions
    if len(orienting_saccades) > 0:
        fig_class.add_trace(
            go.Histogram(
                x=orienting_saccades['amplitude'],
                nbinsx=30,
                name='Orienting',
                marker_color='blue',
                opacity=0.6
            ),
            row=1, col=1
        )
    if len(compensatory_saccades) > 0:
        fig_class.add_trace(
            go.Histogram(
                x=compensatory_saccades['amplitude'],
                nbinsx=30,
                name='Compensatory',
                marker_color='orange',
                opacity=0.6
            ),
            row=1, col=1
        )
    
    # Row 1, Col 2: Duration distributions
    if len(orienting_saccades) > 0:
        fig_class.add_trace(
            go.Histogram(
                x=orienting_saccades['duration'],
                nbinsx=30,
                name='Orienting',
                marker_color='blue',
                opacity=0.6,
                showlegend=False
            ),
            row=1, col=2
        )
    if len(compensatory_saccades) > 0:
        fig_class.add_trace(
            go.Histogram(
                x=compensatory_saccades['duration'],
                nbinsx=30,
                name='Compensatory',
                marker_color='orange',
                opacity=0.6,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Row 1, Col 3: Pre-saccade velocity distributions
    if len(orienting_saccades) > 0:
        fig_class.add_trace(
            go.Histogram(
                x=orienting_saccades['pre_saccade_mean_velocity'],
                nbinsx=30,
                name='Orienting',
                marker_color='blue',
                opacity=0.6,
                showlegend=False
            ),
            row=1, col=3
        )
    if len(compensatory_saccades) > 0:
        fig_class.add_trace(
            go.Histogram(
                x=compensatory_saccades['pre_saccade_mean_velocity'],
                nbinsx=30,
                name='Compensatory',
                marker_color='orange',
                opacity=0.6,
                showlegend=False
            ),
            row=1, col=3
        )
    
    # Row 2, Col 1: Pre-saccade drift distributions
    if len(orienting_saccades) > 0:
        fig_class.add_trace(
            go.Histogram(
                x=orienting_saccades['pre_saccade_position_drift'],
                nbinsx=30,
                name='Orienting',
                marker_color='blue',
                opacity=0.6,
                showlegend=False
            ),
            row=2, col=1
        )
    if len(compensatory_saccades) > 0:
        fig_class.add_trace(
            go.Histogram(
                x=compensatory_saccades['pre_saccade_position_drift'],
                nbinsx=30,
                name='Compensatory',
                marker_color='orange',
                opacity=0.6,
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Row 2, Col 2: Post-saccade variance distributions
    if len(orienting_saccades) > 0:
        fig_class.add_trace(
            go.Histogram(
                x=orienting_saccades['post_saccade_position_variance'],
                nbinsx=30,
                name='Orienting',
                marker_color='blue',
                opacity=0.6,
                showlegend=False
            ),
            row=2, col=2
        )
    if len(compensatory_saccades) > 0:
        fig_class.add_trace(
            go.Histogram(
                x=compensatory_saccades['post_saccade_position_variance'],
                nbinsx=30,
                name='Compensatory',
                marker_color='orange',
                opacity=0.6,
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Row 2, Col 3: Bout size distribution (compensatory only)
    if len(compensatory_saccades) > 0:
        fig_class.add_trace(
            go.Histogram(
                x=compensatory_saccades['bout_size'],
                nbinsx=20,
                name='Compensatory Bout Size',
                marker_color='orange',
                opacity=0.6,
                showlegend=False
            ),
            row=2, col=3
        )
    else:
        # Add empty trace to maintain layout
        fig_class.add_trace(
            go.Histogram(x=[], name='No compensatory saccades'),
            row=2, col=3
        )
    
    # Update layout
    fig_class.update_layout(
        title_text=f'Saccade Classification Analysis: Orienting vs Compensatory ({get_eye_label(video_key)})',
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    # Update axes labels
    fig_class.update_xaxes(title_text="Amplitude (px)", row=1, col=1)
    fig_class.update_xaxes(title_text="Duration (s)", row=1, col=2)
    fig_class.update_xaxes(title_text="Velocity (px/s)", row=1, col=3)
    fig_class.update_xaxes(title_text="Drift (px)", row=2, col=1)
    fig_class.update_xaxes(title_text="Variance (px²)", row=2, col=2)
    fig_class.update_xaxes(title_text="Bout Size (saccades)", row=2, col=3)
    
    fig_class.update_yaxes(title_text="Count", row=1, col=1)
    fig_class.update_yaxes(title_text="Count", row=1, col=2)
    fig_class.update_yaxes(title_text="Count", row=1, col=3)
    fig_class.update_yaxes(title_text="Count", row=2, col=1)
    fig_class.update_yaxes(title_text="Count", row=2, col=2)
    fig_class.update_yaxes(title_text="Count", row=2, col=3)
    
    fig_class.show()
    
    # Time series visualization with classification
    fig_ts = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('X Position (px)', 'Velocity (px/s) with Classified Saccades')
    )
    
    # Add position trace
    fig_ts.add_trace(
        go.Scatter(
            x=res['df']['Seconds'],
            y=res['df']['X_smooth'],
            mode='lines',
            name='Smoothed X',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add velocity trace
    fig_ts.add_trace(
        go.Scatter(
            x=res['df']['Seconds'],
            y=res['df']['vel_x_smooth'],
            mode='lines',
            name='Smoothed Velocity',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    # Add adaptive threshold lines
    fig_ts.add_hline(
        y=res['vel_thresh'],
        line_dash="dash",
        line_color="green",
        opacity=0.5,
        annotation_text=f"Adaptive threshold (±{res['vel_thresh']:.0f} px/s)",
        row=2, col=1
    )
    fig_ts.add_hline(
        y=-res['vel_thresh'],
        line_dash="dash",
        line_color="green",
        opacity=0.5,
        row=2, col=1
    )
    
    # Calculate offset for saccade indicator lines
    vel_max = res['df']['vel_x_smooth'].max()
    vel_min = res['df']['vel_x_smooth'].min()
    vel_range = vel_max - vel_min
    line_offset = vel_range * 0.15
    
    # Plot orienting saccades (blue)
    orienting_in_df = all_saccades_df[all_saccades_df['saccade_type'] == 'orienting']
    if len(orienting_in_df) > 0:
        for idx, row in orienting_in_df.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            peak_time = row['time']
            peak_velocity = row['velocity']
            
            # Draw horizontal line
            y_line_pos = vel_max + line_offset
            fig_ts.add_shape(
                type="line",
                x0=start_time, y0=y_line_pos,
                x1=end_time, y1=y_line_pos,
                line=dict(color='blue', width=3),
                row=2, col=1
            )
            
            # Add arrow
            fig_ts.add_annotation(
                x=peak_time,
                y=y_line_pos,
                ax=0,
                ay=peak_velocity - y_line_pos,
                arrowhead=2,
                arrowsize=2,
                arrowwidth=2,
                arrowcolor='blue',
                row=2, col=1,
                showarrow=True
            )
        
        # Legend entry
        fig_ts.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                name='Orienting Saccades',
                marker=dict(symbol='line-ns', size=15, color='blue', line=dict(width=3))
            ),
            row=2, col=1
        )
    
    # Plot compensatory saccades (orange)
    compensatory_in_df = all_saccades_df[all_saccades_df['saccade_type'] == 'compensatory']
    if len(compensatory_in_df) > 0:
        for idx, row in compensatory_in_df.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            peak_time = row['time']
            peak_velocity = row['velocity']
            
            # Draw horizontal line (below velocity trace)
            y_line_pos = vel_min - line_offset
            fig_ts.add_shape(
                type="line",
                x0=start_time, y0=y_line_pos,
                x1=end_time, y1=y_line_pos,
                line=dict(color='orange', width=3),
                row=2, col=1
            )
            
            # Add arrow
            fig_ts.add_annotation(
                x=peak_time,
                y=y_line_pos,
                ax=0,
                ay=peak_velocity - y_line_pos,
                arrowhead=2,
                arrowsize=2,
                arrowwidth=2,
                arrowcolor='orange',
                row=2, col=1,
                showarrow=True
            )
        
        # Legend entry
        fig_ts.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                name='Compensatory Saccades',
                marker=dict(symbol='line-ns', size=15, color='orange', line=dict(width=3))
            ),
            row=2, col=1
        )
    
    # Update layout
    fig_ts.update_layout(
        title=f'Time Series with Saccade Classification ({get_eye_label(video_key)})<br><sub>Blue: Orienting, Orange: Compensatory</sub>',
        height=600,
        showlegend=True,
        legend=dict(x=0.01, y=0.99)
    )
    
    # Update axes
    fig_ts.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig_ts.update_yaxes(title_text="X Position (px)", row=1, col=1)
    fig_ts.update_yaxes(title_text="Velocity (px/s)", row=2, col=1)
    
    fig_ts.show()


# %%
