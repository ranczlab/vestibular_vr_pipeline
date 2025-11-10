# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: aeon
#     language: python
#     name: python3
# ---

# ## Setup

# +
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gc
import io

from scipy.stats import linregress
from scipy.signal import butter, filtfilt
from scipy.signal import correlate
from scipy.stats import pearsonr

from harp_resources import process
import aeon.io.api as api
from sleap import load_and_process as lp
from sleap import processing_functions as pf
from sleap import saccade_processing as sp
from sleap.saccade_processing import analyze_eye_video_saccades
from sleap.visualization import plot_all_saccades_overlay, plot_saccade_amplitude_qc
from sleap.annotation_gui import launch_annotation_gui
from sleap.ml_feature_extraction import extract_experiment_id
from sleap.annotation_storage import load_annotations, print_annotation_stats
from sleap.visualization import visualize_ml_features
from sleap.ml_feature_extraction import extract_ml_features

# Reload modules to pick up latest changes (useful after code updates)
# Set force_reload_modules = True to always reload, or False to use cached
# versions
# Set to False for faster execution when modules haven't changed
force_reload_modules = True
if force_reload_modules:
    import importlib
    import sleap.load_and_process
    import sleap.processing_functions
    import sleap.saccade_processing
    import sleap.visualization
    import sleap.annotation_gui
    import sleap.ml_feature_extraction
    import sleap.annotation_storage

    importlib.reload(sleap.load_and_process)
    importlib.reload(sleap.processing_functions)
    importlib.reload(sleap.saccade_processing)
    importlib.reload(sleap.visualization)
    importlib.reload(sleap.annotation_gui)
    importlib.reload(sleap.ml_feature_extraction)
    importlib.reload(sleap.annotation_storage)
    # Re-import aliases after reload
    lp = sleap.load_and_process
    pf = sleap.processing_functions
    sp = sleap.saccade_processing
    from sleap.saccade_processing import analyze_eye_video_saccades
    from sleap.visualization import plot_all_saccades_overlay, plot_saccade_amplitude_qc
    from sleap.annotation_gui import launch_annotation_gui
    from sleap.ml_feature_extraction import extract_experiment_id
    from sleap.annotation_storage import load_annotations, print_annotation_stats


def get_eye_label(key):
    """Return mapped user-viewable eye label for video key."""
    return VIDEO_LABELS.get(key, key)


# Column prefixes that indicate SLEAP-derived eye-tracking data
_SLEAP_EYE_PREFIXES = (
    "left",
    "right",
    "center",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
    "Ellipse",
)

RESAMPLED_DROP_COLUMNS = [
    "pre_saccade_mean_velocity",
    "pre_saccade_position_drift",
    "post_saccade_position_variance",
    "post_saccade_position_change",
    "saccade_start_position",
    "saccade_end_position",
    "Ellipse.Angle",
    "Ellipse.Diameter",
    "instance.score",
    "left.x",
    "left.y",
    "left.score",
    "right.x",
    "right.y",
    "right.score",
    "p1.x",
    "p1.y",
    "p1.score",
    "p2.x",
    "p2.y",
    "p2.score",
    "p3.x",
    "p3.y",
    "p3.score",
    "p4.x",
    "p4.y",
    "p4.score",
    "p5.x",
    "p5.y",
    "p5.score",
    "p6.x",
    "p6.y",
    "p6.score",
    "p7.x",
    "p7.y",
    "p7.score",
    "p8.x",
    "p8.y",
    "p8.score",
    "center.x",
    "center.y",
    "center.score",
]

SACCADE_EXPORT_DROP_COLUMNS = {
    "saccade_id",
    "video_key",
    "eye",
    "direction",
    "direction_label",
    "saccade_start_time",
    "saccade_end_time",
    "saccade_peak_time",
    "saccade_start_frame_idx",
    "saccade_peak_frame_idx",
    "saccade_end_frame_idx",
    "saccade_peak_velocity",
    "saccade_amplitude",
    "saccade_displacement",
    "saccade_duration",
    "saccade_start_position",
    "saccade_end_position",
    "saccade_type",
    "bout_id",
    "bout_size",
    "pre_saccade_mean_velocity",
    "pre_saccade_position_drift",
    "post_saccade_position_variance",
    "post_saccade_position_change",
    "classification_confidence",
    "rule_based_class",
    "rule_based_confidence",
    "merge_frame_idx",
}

def _needs_eye_suffix(column: str) -> bool:
    """Return True if the column should be tagged as eye data during resampling."""
    return any(column.startswith(prefix) for prefix in _SLEAP_EYE_PREFIXES)


def resample_video_dataframe(
    video_df: pd.DataFrame,
    eye_tag: str,
    target_freq_hz: float = None,
    optical_filter_hz: float = None,
) -> pd.DataFrame:
    """Resample a SLEAP video dataframe onto the common time grid."""
    if "Seconds" not in video_df.columns:
        raise ValueError("Video dataframe must contain a 'Seconds' column before resampling.")

    target_freq_hz = target_freq_hz or COMMON_RESAMPLED_RATE
    df_for_resample = video_df.copy()

    # Rename SLEAP-specific columns so the resampler treats them as eye data
    rename_map = {
        col: f"{col}_{eye_tag}"
        for col in df_for_resample.columns
        if col not in {"Seconds", "frame_idx"} and _needs_eye_suffix(col)
    }
    df_for_resample = df_for_resample.rename(columns=rename_map)

    # Convert Seconds to aeon datetime index and drop Seconds before resampling
    df_for_resample.index = pd.to_datetime(df_for_resample["Seconds"].apply(api.aeon))
    df_for_resample = df_for_resample.drop(columns=["Seconds"])

    resampled = process.resample_dataframe(
        df_for_resample,
        target_freq_Hz=target_freq_hz,
        optical_filter_Hz=optical_filter_hz,
    )

    # Restore original column names
    inverse_map = {v: k for k, v in rename_map.items()}
    resampled = resampled.rename(columns=inverse_map)

    # Convert index back to Seconds
    resampled_seconds = (resampled.index - datetime(1904, 1, 1)).total_seconds()
    resampled = resampled.reset_index(drop=True)
    resampled.insert(0, "Seconds", resampled_seconds)

    # Frame indices should remain integers if present
    if "frame_idx" in resampled.columns:
        resampled["frame_idx"] = (
            pd.to_numeric(resampled["frame_idx"], errors="coerce").round().astype("Int64")
        )

    return resampled


def set_aeon_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with a DatetimeIndex derived from Seconds."""
    df_indexed = df.copy()
    df_indexed.index = pd.to_datetime(df_indexed["Seconds"].apply(api.aeon))
    df_indexed.index.name = "aeon_time"
    return df_indexed


def append_aeon_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with an additional aeon_time ISO timestamp column."""
    df_with_time = df.copy()
    if isinstance(df_with_time.index, pd.DatetimeIndex):
        df_with_time["aeon_time"] = df_with_time.index.map(lambda ts: ts.isoformat())
    else:
        df_with_time["aeon_time"] = df_with_time["Seconds"].apply(lambda x: api.aeon(x).isoformat())
    return df_with_time


# keep as false here, it is to checking if NaNs already removed if the
# notebook cell is rerun
NaNs_removed = False

# symbols to use ✅ ℹ️ ⚠️ ❗

# +
# set up variables and load data
##########################################################################

data_path = Path(
    "/Users/rancze/Documents/Data/vestVR/20250409_Cohort3_rotation/Visual_mismatch_day4/B6J2782-2025-04-28T14-22-03"
)
# data_path =
# Path('/Users/rancze/Documents/Data/vestVR/Cohort1/No_iso_correction/Visual_mismatch_day3/B6J2717-2024-12-10T12-17-03') only has sleap data 1 for testing purposes

### MOST commonly changed params for tuning
video1_eye = "L" # Options: 'L' or 'R'; which eye does VideoData1 represent? ('L' = Left,'R' = Right)
debug = False # Set to True to enable debug output and QC plots across all cells (file loading, processing, etc.)
plot_saccade_detection_QC = False
plot_QC_timeseries = False
blink_instance_score_threshold = 3.8 # hard threshold for blink detection - frames with instance.score below this value are considered blinks
k1 = 4.5  # adaptive saccade detction threshold (k * SD) for VideoData1 - 3-6 works well
k2 = 4.5  # adaptive saccade detction threshold (k * SD) for VideoData1 - 3-6 works well
refractory_period = 0.1  # seconds, refractory period for saccade detection
bout_window = 1.5  # Time window (seconds) for grouping saccades into bouts

### SLEAP raw data filtering 
score_cutoff = 0.2 # for filtering out inferred points with low confidence, they get interpolated
outlier_sd_threshold = 10  # for removing outliers from the data, they get interpolated

pupil_filter_cutoff_hz = 10  # Hz, Pupil diameter filter settings (Butterworth low-pass)
pupil_filter_order = 6

### Parameters for blink detection
min_blink_duration_ms = 50  # minimum blink duration in milliseconds
blink_merge_window_ms = 100 # NOT CURRENTLY USED: merge window was removed to preserve good data between separate blinks
long_blink_warning_ms = 2000 # warn if blinks exceed this duration (in ms) - user should verify these are real blinks

### params for for saccade detection and classification 
onset_offset_fraction = 0.2 # to determine saccade onset and offset, i.e. 0.2 is 20% of the peak velocity

# Saccade detection parameters (time-based for FPS independence)

pre_saccade_window_time = 0.15 # Time (seconds) before threshold crossing to extract peri-saccade segment
post_saccade_window_time = 0.5

baseline_window_start_time = -0.06 # Start time (seconds) relative to threshold crossing for baseline window (e.g., -0.1 = 100ms before)
baseline_window_end_time = -0.02 # End time (seconds) relative to threshold crossing for baseline window (e.g., -0.02 = 20ms before)
smoothing_window_time = 0.08 # Time (seconds) for position smoothing window (rolling median)
peak_width_time = 0.005 # Minimum peak width (seconds) for find_peaks - typically 5-20ms for saccades
min_saccade_duration = 0.2 # Minimum saccade segment duration (seconds) - segments shorter than this are excluded (typically truncated at recording edges)

# Parameters for orienting vs compensatory saccade classification
classify_orienting_compensatory = True # this is the rule-based, not ML based, we generally use it 
pre_saccade_window = 0.3 # Time window (seconds) before saccade onset to analyze
max_intersaccade_interval_for_classification = 1.0 # Maximum time (seconds) to extend post-saccade window until next saccade for classification for feature extraction
pre_saccade_velocity_threshold = 50.0 # Velocity threshold (px/s) for detecting pre-saccade drift for feature extraction
pre_saccade_drift_threshold = 10.0 # Position drift threshold (px) before saccade for compensatory classification for feature extraction
post_saccade_variance_threshold = 100.0 # Position variance threshold (px²) after saccade for orienting classification for feature extraction
post_saccade_position_change_threshold_percent = 50.0 # Position change threshold (% of saccade amplitude) - if post-saccade change > amplitude * this%, classify as compensatory for feature extraction

# Adaptive threshold parameters (percentile-based) for 
# Set to True to use adaptive thresholds based on feature distributions, False to use fixed thresholds
use_adaptive_thresholds = True
adaptive_percentile_pre_velocity = 75 # Percentile for pre-saccade velocity threshold (upper percentile for compensatory detection)
adaptive_percentile_pre_drift = 75 # Percentile for pre-saccade drift threshold (upper percentile for compensatory detection)
adaptive_percentile_post_variance = 25 # Percentile for post-saccade variance threshold (lower percentile for orienting detection - low variance = stable)

COMMON_RESAMPLED_RATE = 1000 # Common resampling rate (Hz) used for alignment with other modalities, like photometry, HARP streams, etc. 



# Automatically assign eye for VideoData2
video2_eye = "R" if video1_eye == "L" else "L"
eye_fullname = {"L": "Left", "R": "Right"} # Map for full names (used in labels)
VIDEO_LABELS = { # Update VIDEO_LABELS based on selection
    "VideoData1": f"VideoData1 ({video1_eye}: {eye_fullname[video1_eye]})",
    "VideoData2": f"VideoData2 ({video2_eye}: {eye_fullname[video2_eye]})",
}

save_path = data_path.parent / f"{data_path.name}_processedData"
qc_debug_dir = save_path / "QC_and_debug"
qc_debug_dir.mkdir(parents=True, exist_ok=True)

VideoData1, VideoData2, VideoData1_Has_Sleap, VideoData2_Has_Sleap = (
    lp.load_videography_data(data_path, debug=debug)
)

# Load manual blink data if available
manual_blinks_v1 = pf.load_manual_blinks(data_path, video_number=1)
manual_blinks_v2 = pf.load_manual_blinks(data_path, video_number=2)

columns_of_interest = [
    "left.x",
    "left.y",
    "center.x",
    "center.y",
    "right.x",
    "right.y",
    "p1.x",
    "p1.y",
    "p2.x",
    "p2.y",
    "p3.x",
    "p3.y",
    "p4.x",
    "p4.y",
    "p5.x",
    "p5.y",
    "p6.x",
    "p6.y",
    "p7.x",
    "p7.y",
    "p8.x",
    "p8.y",
]

if VideoData1_Has_Sleap:
    # drop the track column as it is empty
    VideoData1 = VideoData1.drop(columns=["track"])
    coordinates_dict1_raw = lp.get_coordinates_dict(VideoData1, columns_of_interest)
    # frame rate for VideoData1 TODO where to save it, is it useful?
    FPS_1 = 1 / VideoData1["Seconds"].diff().mean()
    print()
    print(f"{get_eye_label('VideoData1')}: FPS = {FPS_1}")

if VideoData2_Has_Sleap:
    # drop the track column as it is empty
    VideoData2 = VideoData2.drop(columns=["track"])
    coordinates_dict2_raw = lp.get_coordinates_dict(VideoData2, columns_of_interest)
    # frame rate for VideoData2
    FPS_2 = 1 / VideoData2["Seconds"].diff().mean()
    print(f"{get_eye_label('VideoData2')}: FPS = {FPS_2}")
# -

# plot timeseries of coordinates in browser for both VideoData1 and VideoData2
##########################################################################
if plot_QC_timeseries:
    print(
        "⚠️ Check for long discontinuities and outliers in the data, we will try to deal with them later"
    )
    print("ℹ️ Figures open in browser window, takes a bit of time.")

    # Helper list variables
    subplot_titles = (
        "X coordinates for pupil centre and left-right eye corner",
        "Y coordinates for pupil centre and left-right eye corner",
        "X coordinates for iris points",
        "Y coordinates for iris points",
    )
    eye_x = ["left.x", "center.x", "right.x"]
    eye_y = ["left.y", "center.y", "right.y"]
    iris_x = ["p1.x", "p2.x", "p3.x", "p4.x", "p5.x", "p6.x", "p7.x", "p8.x"]
    iris_y = ["p1.y", "p2.y", "p3.y", "p4.y", "p5.y", "p6.y", "p7.y", "p8.y"]

    # --- VideoData1 ---
    if VideoData1_Has_Sleap:
        fig1 = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
        )

        # Row 1: left.x, center.x, right.x
        for col in eye_x:
            fig1.add_trace(
                go.Scatter(
                    x=VideoData1["Seconds"], y=VideoData1[col], mode="lines", name=col
                ),
                row=1,
                col=1,
            )
        # Row 2: left.y, center.y, right.y
        for col in eye_y:
            fig1.add_trace(
                go.Scatter(
                    x=VideoData1["Seconds"], y=VideoData1[col], mode="lines", name=col
                ),
                row=2,
                col=1,
            )
        # Row 3: p1.x ... p8.x
        for col in iris_x:
            fig1.add_trace(
                go.Scatter(
                    x=VideoData1["Seconds"], y=VideoData1[col], mode="lines", name=col
                ),
                row=3,
                col=1,
            )
        # Row 4: p1.y ... p8.y
        for col in iris_y:
            fig1.add_trace(
                go.Scatter(
                    x=VideoData1["Seconds"], y=VideoData1[col], mode="lines", name=col
                ),
                row=4,
                col=1,
            )

        fig1.update_layout(
            height=1200,
            title_text=f"Time series subplots for coordinates [{get_eye_label('VideoData1')}]",
            showlegend=True,
        )
        fig1.update_xaxes(title_text="Seconds", row=4, col=1)
        fig1.update_yaxes(title_text="X Position", row=1, col=1)
        fig1.update_yaxes(title_text="Y Position", row=2, col=1)
        fig1.update_yaxes(title_text="X Position", row=3, col=1)
        fig1.update_yaxes(title_text="Y Position", row=4, col=1)

        fig1.show(renderer="browser")

    # --- VideoData2 ---
    if VideoData2_Has_Sleap:
        fig2 = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
        )
        # Row 1: left.x, center.x, right.x
        for col in eye_x:
            fig2.add_trace(
                go.Scatter(
                    x=VideoData2["Seconds"], y=VideoData2[col], mode="lines", name=col
                ),
                row=1,
                col=1,
            )
        # Row 2: left.y, center.y, right.y
        for col in eye_y:
            fig2.add_trace(
                go.Scatter(
                    x=VideoData2["Seconds"], y=VideoData2[col], mode="lines", name=col
                ),
                row=2,
                col=1,
            )
        # Row 3: p1.x ... p8.x
        for col in iris_x:
            fig2.add_trace(
                go.Scatter(
                    x=VideoData2["Seconds"], y=VideoData2[col], mode="lines", name=col
                ),
                row=3,
                col=1,
            )
        # Row 4: p1.y ... p8.y
        for col in iris_y:
            fig2.add_trace(
                go.Scatter(
                    x=VideoData2["Seconds"], y=VideoData2[col], mode="lines", name=col
                ),
                row=4,
                col=1,
            )

        fig2.update_layout(
            height=1200,
            title_text=f"Time series subplots for coordinates [{get_eye_label('VideoData2')}]",
            showlegend=True,
        )
        fig2.update_xaxes(title_text="Seconds", row=4, col=1)
        fig2.update_yaxes(title_text="X Position", row=1, col=1)
        fig2.update_yaxes(title_text="Y Position", row=2, col=1)
        fig2.update_yaxes(title_text="X Position", row=3, col=1)
        fig2.update_yaxes(title_text="Y Position", row=4, col=1)

        fig2.show(renderer="browser")

# +
# QC plot XY coordinate distributions to visualize outliers
##########################################################################
columns_of_interest = [
    "left",
    "right",
    "center",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
]

# Filter out NaN values and calculate the min and max values for X and Y
# coordinates for both dict1 and dict2

def min_max_dict(coordinates_dict):
    x_min = min(
        [
            coordinates_dict[f"{col}.x"][
                ~np.isnan(coordinates_dict[f"{col}.x"])
            ].min()
            for col in columns_of_interest
        ]
    )
    x_max = max(
        [
            coordinates_dict[f"{col}.x"][
                ~np.isnan(coordinates_dict[f"{col}.x"])
            ].max()
            for col in columns_of_interest
        ]
    )
    y_min = min(
        [
            coordinates_dict[f"{col}.y"][
                ~np.isnan(coordinates_dict[f"{col}.y"])
            ].min()
            for col in columns_of_interest
        ]
    )
    y_max = max(
        [
            coordinates_dict[f"{col}.y"][
                ~np.isnan(coordinates_dict[f"{col}.y"])
            ].max()
            for col in columns_of_interest
        ]
    )
    return x_min, x_max, y_min, y_max

# PLOT QC plot XY coordinate distributions to visualize outliers
##########################################################################
if plot_QC_timeseries:
    # Compute min/max as before for global axes limits
    if VideoData1_Has_Sleap:
        x_min1, x_max1, y_min1, y_max1 = pf.min_max_dict(
            coordinates_dict1_raw, columns_of_interest
        )

    if VideoData2_Has_Sleap:
        x_min2, x_max2, y_min2, y_max2 = pf.min_max_dict(
            coordinates_dict2_raw, columns_of_interest
        )

    # Use global min and max for consistency only if both VideoData1_Has_Sleap
    # and VideoData2_Has_Sleap are True
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
        fontsize=14,
    )

    # Define colormap for p1-p8

    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "orange"]

    # Panel 1: left, right, center (dict1)

    if "VideoData1_Has_Sleap" in globals() and VideoData1_Has_Sleap:
        ax[0, 0].set_title(f"{get_eye_label('VideoData1')}: left, right, center")
        ax[0, 0].scatter(
            coordinates_dict1_raw["left.x"],
            coordinates_dict1_raw["left.y"],
            color="black",
            label="left",
            s=10,
        )
        ax[0, 0].scatter(
            coordinates_dict1_raw["right.x"],
            coordinates_dict1_raw["right.y"],
            color="grey",
            label="right",
            s=10,
        )
        ax[0, 0].scatter(
            coordinates_dict1_raw["center.x"],
            coordinates_dict1_raw["center.y"],
            color="red",
            label="center",
            s=10,
        )
        ax[0, 0].set_xlim([x_min, x_max])
        ax[0, 0].set_ylim([y_min, y_max])
        ax[0, 0].set_xlabel("x coordinates (pixels)")
        ax[0, 0].set_ylabel("y coordinates (pixels)")
        ax[0, 0].legend(loc="upper right")
    else:
        ax[0, 0].axis("off")

    # Panel 2: p1 to p8 (dict1)

    if "VideoData1_Has_Sleap" in globals() and VideoData1_Has_Sleap:
        ax[0, 1].set_title(f"{get_eye_label('VideoData1')}: p1 to p8")
        for idx, col in enumerate(columns_of_interest[3:]):
            ax[0, 1].scatter(
                coordinates_dict1_raw[f"{col}.x"],
                coordinates_dict1_raw[f"{col}.y"],
                color=colors[idx],
                label=col,
                s=5,
            )

        ax[0, 1].set_xlim([x_min, x_max])
        ax[0, 1].set_ylim([y_min, y_max])
        ax[0, 1].set_xlabel("x coordinates (pixels)")
        ax[0, 1].set_ylabel("y coordinates (pixels)")
        ax[0, 1].legend(loc="upper right")
    else:
        ax[0, 1].axis("off")

    # Panel 3: left, right, center (dict2)

    if "VideoData2_Has_Sleap" in globals() and VideoData2_Has_Sleap:
        ax[1, 0].set_title(f"{get_eye_label('VideoData2')}: left, right, center")
        ax[1, 0].scatter(
            coordinates_dict2_raw["left.x"],
            coordinates_dict2_raw["left.y"],
            color="black",
            label="left",
            s=10,
        )
        ax[1, 0].scatter(
            coordinates_dict2_raw["right.x"],
            coordinates_dict2_raw["right.y"],
            color="grey",
            label="right",
            s=10,
        )
        ax[1, 0].scatter(
            coordinates_dict2_raw["center.x"],
            coordinates_dict2_raw["center.y"],
            color="red",
            label="center",
            s=10,
        )
        ax[1, 0].set_xlim([x_min, x_max])
        ax[1, 0].set_ylim([y_min, y_max])
        ax[1, 0].set_xlabel("x coordinates (pixels)")
        ax[1, 0].set_ylabel("y coordinates (pixels)")
        ax[1, 0].legend(loc="upper right")
    else:
        ax[1, 0].axis("off")

    # Panel 4: p1 to p8 (dict2)

    if "VideoData2_Has_Sleap" in globals() and VideoData2_Has_Sleap:
        ax[1, 1].set_title(f"{get_eye_label('VideoData2')}: p1 to p8")
        for idx, col in enumerate(columns_of_interest[3:]):
            ax[1, 1].scatter(
                coordinates_dict2_raw[f"{col}.x"],
                coordinates_dict2_raw[f"{col}.y"],
                color=colors[idx],
                label=col,
                s=5,
            )

        ax[1, 1].set_xlim([x_min, x_max])
        ax[1, 1].set_ylim([y_min, y_max])
        ax[1, 1].set_xlabel("x coordinates (pixels)")
        ax[1, 1].set_ylabel("y coordinates (pixels)")
        ax[1, 1].legend(loc="upper right")
    else:
        ax[1, 1].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    plt.show()

# +
# Center coordinates, filter low-confidence points, remove outliers, and interpolate
##########################################################################

# Detect and print confidence scores analysis (runs before any filtering)
#########

if not debug:
    print(
        "ℹ️ Debug output suppressed. Set debug=True to see detailed confidence score analysis."
    )

score_columns = [
    "left.score",
    "center.score",
    "right.score",
    "p1.score",
    "p2.score",
    "p3.score",
    "p4.score",
    "p5.score",
    "p6.score",
    "p7.score",
    "p8.score",
]

# VideoData1 confidence score analysis
if debug and "VideoData1_Has_Sleap" in globals() and VideoData1_Has_Sleap:
    pf.analyze_confidence_scores(
        VideoData1,
        score_columns,
        score_cutoff,
        get_eye_label("VideoData1"),
        debug=debug,
    )

# VideoData2 confidence score analysis
if debug and "VideoData2_Has_Sleap" in globals() and VideoData2_Has_Sleap:
    pf.analyze_confidence_scores(
        VideoData2,
        score_columns,
        score_cutoff,
        get_eye_label("VideoData2"),
        debug=debug,
    )


print()
print("=== Centering coordinates to the median pupil centre ===")
# Reset columns_of_interest to full coordinate column names (needed after
# QC plotting redefined it)
columns_of_interest = [
    "left.x",
    "left.y",
    "center.x",
    "center.y",
    "right.x",
    "right.y",
    "p1.x",
    "p1.y",
    "p2.x",
    "p2.y",
    "p3.x",
    "p3.y",
    "p4.x",
    "p4.y",
    "p5.x",
    "p5.y",
    "p6.x",
    "p6.y",
    "p7.x",
    "p7.y",
    "p8.x",
    "p8.y",
]
# VideoData1 processing
if "VideoData1_Has_Sleap" in globals() and VideoData1_Has_Sleap:
    VideoData1_centered = pf.center_coordinates_to_median(
        VideoData1, columns_of_interest, get_eye_label("VideoData1")
    )

# VideoData2 processing
if "VideoData2_Has_Sleap" in globals() and VideoData2_Has_Sleap:
    VideoData2_centered = pf.center_coordinates_to_median(
        VideoData2, columns_of_interest, get_eye_label("VideoData2")
    )

# remove low confidence points (score < threshold)
##################################################
if not NaNs_removed:
    if debug:
        print(
            "\n=== Score-based Filtering - point scores below threshold are replaced by interpolation ==="
        )
        print(f"Score threshold: {score_cutoff}")
    # List of point names (without .x, .y, .score)
    point_names = [
        "left",
        "right",
        "center",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
    ]

    # VideoData1 score-based filtering
    if "VideoData1_Has_Sleap" in globals() and VideoData1_Has_Sleap:
        pf.filter_low_confidence_points(
            VideoData1,
            point_names,
            score_cutoff,
            get_eye_label("VideoData1"),
            debug=debug,
        )

    # VideoData2 score-based filtering
    if "VideoData2_Has_Sleap" in globals() and VideoData2_Has_Sleap:
        pf.filter_low_confidence_points(
            VideoData2,
            point_names,
            score_cutoff,
            get_eye_label("VideoData2"),
            debug=debug,
        )

    # remove outliers (x times SD)
    # then interpolates on all NaN values (skipped frames, low confidence inference points, outliers)
    ##########################################################################

    if debug:
        print(
            "\n=== Outlier Analysis - outlier points are replaced by interpolation ==="
        )

    # Reset columns_of_interest to full coordinate column names (needed after
    # QC plotting redefined it)
    columns_of_interest = [
        "left.x",
        "left.y",
        "center.x",
        "center.y",
        "right.x",
        "right.y",
        "p1.x",
        "p1.y",
        "p2.x",
        "p2.y",
        "p3.x",
        "p3.y",
        "p4.x",
        "p4.y",
        "p5.x",
        "p5.y",
        "p6.x",
        "p6.y",
        "p7.x",
        "p7.y",
        "p8.x",
        "p8.y",
    ]

    # VideoData1 outlier analysis and interpolation
    if "VideoData1_Has_Sleap" in globals() and VideoData1_Has_Sleap:
        outlier_results_v1 = pf.remove_outliers_and_interpolate(
            VideoData1,
            columns_of_interest,
            outlier_sd_threshold,
            get_eye_label("VideoData1"),
            debug=debug,
        )
        VideoData1 = outlier_results_v1["video_data_interpolated"]

    # VideoData2 outlier analysis and interpolation
    if "VideoData2_Has_Sleap" in globals() and VideoData2_Has_Sleap:
        outlier_results_v2 = pf.remove_outliers_and_interpolate(
            VideoData2,
            columns_of_interest,
            outlier_sd_threshold,
            get_eye_label("VideoData2"),
            debug=debug,
        )
        VideoData2 = outlier_results_v2["video_data_interpolated"]

    # Set flag after both VideoData1 and VideoData2 processing is complete
    NaNs_removed = True
else:
    print("=== Interpolation already done, skipping ===")


# +
# Instance.score distribution and hard threshold for blink detection
##########################################################################
# Plotting the distribution of instance scores and using hard threshold for blink detection.
# When instance score is low, that's typically because of a blink or similar occlusion, as there are long sequences of low scores.
# Frames with instance.score below the hard threshold are considered
# potential blinks.

if not debug:
    print(
        "ℹ️ Debug output suppressed. Set debug=True to see detailed instance score distribution analysis."
    )

if debug:
    print("=" * 80)
    print("INSTANCE.SCORE DISTRIBUTION AND BLINK DETECTION THRESHOLD")
    print("=" * 80)
    print(f"\nHard threshold: instance.score < {blink_instance_score_threshold}")
    print(
        "  Frames with instance.score below this threshold will be considered potential blinks."
    )
    print("=" * 80)

# Only analyze for dataset(s) that exist
has_v1 = "VideoData1_Has_Sleap" in globals() and VideoData1_Has_Sleap
has_v2 = "VideoData2_Has_Sleap" in globals() and VideoData2_Has_Sleap

# Get FPS for time calculations
fps_1_for_threshold = None
fps_2_for_threshold = None
if has_v1:
    fps_1_for_threshold = (
        FPS_1
        if "FPS_1" in globals()
        else (1 / VideoData1["Seconds"].diff().mean() if has_v1 else None)
    )
if has_v2:
    fps_2_for_threshold = (
        FPS_2
        if "FPS_2" in globals()
        else (1 / VideoData2["Seconds"].diff().mean() if has_v2 else None)
    )

# Plot combined histograms
if debug and (has_v1 or has_v2):
    pf.plot_instance_score_distributions_combined(
        VideoData1 if has_v1 else None,
        VideoData2 if has_v2 else None,
        blink_instance_score_threshold,
        has_v1=has_v1,
        has_v2=has_v2,
    )

# Report the statistics for available VideoData
# Always show key stats: number/percentile below threshold and longest
# consecutive segment
if has_v1:
    pf.analyze_instance_score_distribution(
        VideoData1,
        blink_instance_score_threshold,
        fps_1_for_threshold,
        # Already plotted above
        get_eye_label("VideoData1"),
        debug=debug,
        plot=False,
    )

if has_v2:
    pf.analyze_instance_score_distribution(
        VideoData2,
        blink_instance_score_threshold,
        fps_2_for_threshold,
        # Already plotted above
        get_eye_label("VideoData2"),
        debug=debug,
        plot=False,
    )

if debug:
    print(f"\n{'=' * 80}")
    print("Note: This threshold will be used for blink detection in the next cell.")
    print(
        "      Frames with instance.score below this threshold are considered potential blinks."
    )
    print("=" * 80)


# +
# Blink detection using instance.score - mark blinks and set coordinates to NaN (keep them as NaN, no interpolation)
##########################################################################

if not debug:
    print(
        "ℹ️ Debug output suppressed. Set debug=True to see detailed blink detection information."
    )

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
if "VideoData1_Has_Sleap" in globals() and VideoData1_Has_Sleap:
    # Get FPS if available, otherwise will be calculated in function
    fps_1 = FPS_1 if "FPS_1" in globals() else None

    # Get manual blinks if available
    manual_blinks_for_v1 = (
        manual_blinks_v1
        if "manual_blinks_v1" in globals() and manual_blinks_v1 is not None
        else None
    )

    # Run blink detection
    blink_results_v1 = pf.detect_blinks_for_video(
        video_data=VideoData1,
        columns_of_interest=columns_of_interest,
        blink_instance_score_threshold=blink_instance_score_threshold,
        long_blink_warning_ms=long_blink_warning_ms,
        min_frames_threshold=4,
        merge_window_frames=10,
        fps=fps_1,
        video_label=get_eye_label("VideoData1"),
        manual_blinks=manual_blinks_for_v1,
        debug=debug,
    )

    # Extract results to maintain compatibility with existing variable names
    blink_segments_v1 = blink_results_v1["blink_segments"]
    short_blink_segments_v1 = blink_results_v1["short_blink_segments"]
    blink_bouts_v1 = blink_results_v1["blink_bouts"]
    all_blink_segments_v1 = blink_results_v1["all_blink_segments"]
    fps_1 = blink_results_v1["fps"]  # Update fps_1 with calculated value
    FPS_1 = fps_1  # Also update global FPS_1 for use elsewhere
    long_blinks_warnings_v1 = blink_results_v1["long_blinks_warnings"]

# VideoData2 blink detection
if "VideoData2_Has_Sleap" in globals() and VideoData2_Has_Sleap:
    # Get FPS if available, otherwise will be calculated in function
    fps_2 = FPS_2 if "FPS_2" in globals() else None

    # Get manual blinks if available
    manual_blinks_for_v2 = (
        manual_blinks_v2
        if "manual_blinks_v2" in globals() and manual_blinks_v2 is not None
        else None
    )

    # Run blink detection
    blink_results_v2 = pf.detect_blinks_for_video(
        video_data=VideoData2,
        columns_of_interest=columns_of_interest,
        blink_instance_score_threshold=blink_instance_score_threshold,
        long_blink_warning_ms=long_blink_warning_ms,
        min_frames_threshold=4,
        merge_window_frames=10,
        fps=fps_2,
        video_label=get_eye_label("VideoData2"),
        manual_blinks=manual_blinks_for_v2,
        debug=debug,
    )

    # Extract results to maintain compatibility with existing variable names
    blink_segments_v2 = blink_results_v2["blink_segments"]
    short_blink_segments_v2 = blink_results_v2["short_blink_segments"]
    blink_bouts_v2 = blink_results_v2["blink_bouts"]
    all_blink_segments_v2 = blink_results_v2["all_blink_segments"]
    fps_2 = blink_results_v2["fps"]  # Update fps_2 with calculated value
    FPS_2 = fps_2  # Also update global FPS_2 for use elsewhere
    long_blinks_warnings_v2 = blink_results_v2["long_blinks_warnings"]

print("\n✅ Blink detection complete. Blink periods remain as NaN (not interpolated).")

# Compare blink bout timing between VideoData1 and VideoData2 (between eyes)
if (
    "VideoData1_Has_Sleap" in globals()
    and VideoData1_Has_Sleap
    and "VideoData2_Has_Sleap" in globals()
    and VideoData2_Has_Sleap
):
    if debug:
        print("\n" + "=" * 80)
        print("BLINK BOUT TIMING COMPARISON: VideoData1 vs VideoData2 (Between Eyes)")
        print("=" * 80)

    # Get blink bout frame ranges for both videos (if they exist)
    # Check if blink_bouts variables exist (they are created during blink
    # detection)
    try:
        has_bouts_v1 = "blink_bouts_v1" in globals() and len(blink_bouts_v1) > 0
    except BaseException:
        has_bouts_v1 = False

    try:
        has_bouts_v2 = "blink_bouts_v2" in globals() and len(blink_bouts_v2) > 0
    except BaseException:
        has_bouts_v2 = False

    if has_bouts_v1 and has_bouts_v2:
        # Convert bout indices to frame numbers
        bouts_v1 = []
        for i, bout in enumerate(blink_bouts_v1, 1):
            start_idx = bout["start_idx"]
            end_idx = bout["end_idx"]
            if "frame_idx" in VideoData1.columns:
                start_frame = int(VideoData1["frame_idx"].iloc[start_idx])
                end_frame = int(VideoData1["frame_idx"].iloc[end_idx])
            else:
                start_frame = start_idx
                end_frame = end_idx
            bouts_v1.append(
                {
                    "num": i,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "length": bout["length"],
                }
            )

        bouts_v2 = []
        for i, bout in enumerate(blink_bouts_v2, 1):
            start_idx = bout["start_idx"]
            end_idx = bout["end_idx"]
            if "frame_idx" in VideoData2.columns:
                start_frame = int(VideoData2["frame_idx"].iloc[start_idx])
                end_frame = int(VideoData2["frame_idx"].iloc[end_idx])
            else:
                start_frame = start_idx
                end_frame = end_idx
            bouts_v2.append(
                {
                    "num": i,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "length": bout["length"],
                }
            )

        # Find concurrent bouts (overlapping in time, synchronized by Seconds)
        concurrent_bouts = []
        v1_independent = []
        v2_independent = []

        v2_matched = set()  # Track which VideoData2 bouts have been matched

        for bout1 in bouts_v1:
            # Get time range for bout1
            v1_start_time = VideoData1["Seconds"].iloc[bout1["start_idx"]]
            v1_end_time = VideoData1["Seconds"].iloc[bout1["end_idx"]]

            found_match = False
            for bout2 in bouts_v2:
                # Get time range for bout2
                v2_start_time = VideoData2["Seconds"].iloc[bout2["start_idx"]]
                v2_end_time = VideoData2["Seconds"].iloc[bout2["end_idx"]]

                # Check if bouts overlap in time (any overlapping time period)
                overlap_start_time = max(v1_start_time, v2_start_time)
                overlap_end_time = min(v1_end_time, v2_end_time)

                if overlap_start_time <= overlap_end_time:
                    # Concurrent - they overlap in time
                    # Calculate overlap duration
                    overlap_duration = overlap_end_time - overlap_start_time

                    concurrent_bouts.append(
                        {
                            "v1_num": bout1["num"],
                            "v1_start_frame": bout1["start_frame"],
                            "v1_end_frame": bout1["end_frame"],
                            "v1_start_time": v1_start_time,
                            "v1_end_time": v1_end_time,
                            "v2_num": bout2["num"],
                            "v2_start_frame": bout2["start_frame"],
                            "v2_end_frame": bout2["end_frame"],
                            "v2_start_time": v2_start_time,
                            "v2_end_time": v2_end_time,
                            "overlap_start_time": overlap_start_time,
                            "overlap_end_time": overlap_end_time,
                            "overlap_duration": overlap_duration,
                        }
                    )
                    v2_matched.add(bout2["num"])
                    found_match = True
                    break

            if not found_match:
                v1_independent.append(bout1)

        # Find VideoData2 bouts that don't have matches
        for bout2 in bouts_v2:
            if bout2["num"] not in v2_matched:
                v2_independent.append(bout2)

        # Calculate statistics
        total_v1_bouts = len(bouts_v1)
        total_v2_bouts = len(bouts_v2)
        total_concurrent = len(concurrent_bouts)
        total_v1_independent = len(v1_independent)
        total_v2_independent = len(v2_independent)

        if debug:
            print("\nBlink bout counts:")
            print(f"  VideoData1: {total_v1_bouts} blink bout(s)")
            print(f"  VideoData2: {total_v2_bouts} blink bout(s)")
            print(f"  Concurrent: {total_concurrent} bout(s) (overlapping frames)")
            print(f"  VideoData1 only: {total_v1_independent} bout(s)")
            print(f"  VideoData2 only: {total_v2_independent} bout(s)")

            if total_v1_bouts > 0 and total_v2_bouts > 0:
                concurrent_pct_v1 = (total_concurrent / total_v1_bouts) * 100
                concurrent_pct_v2 = (total_concurrent / total_v2_bouts) * 100
                print("\nConcurrency percentage:")
                print(
                    f"  {concurrent_pct_v1:.1f}% of VideoData1 bouts are concurrent with VideoData2"
                )
                print(
                    f"  {concurrent_pct_v2:.1f}% of VideoData2 bouts are concurrent with VideoData1"
                )

                # Calculate timing offsets for concurrent bouts
                if len(concurrent_bouts) > 0:
                    time_offsets_ms = []
                    for cb in concurrent_bouts:
                        # Calculate offset from start times (already in
                        # Seconds)
                        offset_ms = (cb["v1_start_time"] - cb["v2_start_time"]) * 1000
                        time_offsets_ms.append(offset_ms)
                        cb["time_offset_ms"] = offset_ms

                    mean_offset = np.mean(time_offsets_ms)
                    std_offset = np.std(time_offsets_ms)
                    print("\nTiming offset for concurrent bouts:")
                    print(
                        f"  Mean offset (VideoData1 - VideoData2): {mean_offset:.2f} ms"
                    )
                    print(f"  Std offset: {std_offset:.2f} ms")
                    print(
                        f"  Range: {min(time_offsets_ms):.2f} to {max(time_offsets_ms):.2f} ms"
                    )

            # Visualization removed per request
            print("=" * 80)
    elif has_bouts_v1 or has_bouts_v2:
        print(
            "\n⚠️ Cannot compare blink bouts - only one eye has blink bouts detected:"
        )
        if has_bouts_v1:
            print(f"  VideoData1: {len(blink_bouts_v1)} blink bout(s)")
        else:
            print("  VideoData1: 0 blink bout(s)")
        if has_bouts_v2:
            print(f"  VideoData2: {len(blink_bouts_v2)} blink bout(s)")
        else:
            print("  VideoData2: 0 blink bout(s)")
    else:
        print("\n⚠️ Cannot compare blink bouts - neither video has blink bouts detected")

# Save blink detection results to CSV files
if "VideoData1_Has_Sleap" in globals() and VideoData1_Has_Sleap:
    if len(blink_segments_v1) > 0:
        # Collect blink information
        blink_data_v1 = []
        manual_blinks_for_csv = None
        if "manual_blinks_v1" in globals() and manual_blinks_v1 is not None:
            manual_blinks_for_csv = manual_blinks_v1

        for i, blink in enumerate(blink_segments_v1, 1):
            start_idx = blink["start_idx"]
            end_idx = blink["end_idx"]

            # Get actual frame numbers from frame_idx column
            if "frame_idx" in VideoData1.columns:
                first_frame = int(VideoData1["frame_idx"].iloc[start_idx])
                last_frame = int(VideoData1["frame_idx"].iloc[end_idx])
            else:
                first_frame = start_idx
                last_frame = end_idx

            # Check if this blink matches a manual one (using function from
            # processing_functions)
            matches_manual = pf.check_manual_match(
                first_frame, last_frame, manual_blinks_for_csv
            )

            blink_data_v1.append(
                {
                    "blink_number": i,
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                    "matches_manual": matches_manual,
                }
            )

        # Create DataFrame and save to CSV
        blink_df_v1 = pd.DataFrame(blink_data_v1)
        blink_csv_path_v1 = qc_debug_dir / "blink_detection_VideoData1.csv"
        blink_df_v1.to_csv(blink_csv_path_v1, index=False)
        print(
            f"\n✅ Blink detection results (VideoData1) saved to: {blink_csv_path_v1}"
        )
        print(f"   Saved {len(blink_data_v1)} blinks")

if "VideoData2_Has_Sleap" in globals() and VideoData2_Has_Sleap:
    if len(blink_segments_v2) > 0:
        # Collect blink information
        blink_data_v2 = []
        manual_blinks_for_csv = None
        if "manual_blinks_v2" in globals() and manual_blinks_v2 is not None:
            manual_blinks_for_csv = manual_blinks_v2

        for i, blink in enumerate(blink_segments_v2, 1):
            start_idx = blink["start_idx"]
            end_idx = blink["end_idx"]

            # Get actual frame numbers from frame_idx column
            if "frame_idx" in VideoData2.columns:
                first_frame = int(VideoData2["frame_idx"].iloc[start_idx])
                last_frame = int(VideoData2["frame_idx"].iloc[end_idx])
            else:
                first_frame = start_idx
                last_frame = end_idx

            # Check if this blink matches a manual one (using function from
            # processing_functions)
            matches_manual = pf.check_manual_match(
                first_frame, last_frame, manual_blinks_for_csv
            )

            blink_data_v2.append(
                {
                    "blink_number": i,
                    "first_frame": first_frame,
                    "last_frame": last_frame,
                    "matches_manual": matches_manual,
                }
            )

        # Create DataFrame and save to CSV
        blink_df_v2 = pd.DataFrame(blink_data_v2)
        blink_csv_path_v2 = qc_debug_dir / "blink_detection_VideoData2.csv"
        blink_df_v2.to_csv(blink_csv_path_v2, index=False)
        print(
            f"\n✅ Blink detection results (VideoData2) saved to: {blink_csv_path_v2}"
        )
        print(f"   Saved {len(blink_data_v2)} blinks")

print("\n" + "=" * 80)
print("📹 MANUAL QC CHECK:")
print("=" * 80)
print("For instructions on how to prepare videos for manual blink detection QC,")
print("see: https://github.com/ranczlab/vestibular_vr_pipeline/issues/86")
print("=" * 80)

# Restore original stdout and save captured output to file
sys.stdout = original_stdout

# Get the captured output
captured_output = output_buffer.getvalue()

# Save to file in data_path folder
output_file = qc_debug_dir / "blink_detection_QC.txt"
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w") as f:
    f.write(captured_output)

print(f"\n✅ Blink detection output saved to: {output_file}")


# +
# QC plot timeseries of interpolation corrected NaN in browser
##########################################################################

if plot_QC_timeseries:
    print("ℹ️ Figure opens in browser window, takes a bit of time.")

    # VideoData1 QC Plot
    if "VideoData1_Has_Sleap" in globals() and VideoData1_Has_Sleap:
        fig1 = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "VideoData1 - X coordinates for pupil centre and left-right eye corner",
                "VideoData1 - Y coordinates for pupil centre and left-right eye corner",
                "VideoData1 - X coordinates for iris points",
                "VideoData1 - Y coordinates for iris points",
            ),
        )

        # Row 1: Plot left.x, center.x, right.x
        fig1.add_trace(
            go.Scatter(
                x=VideoData1["Seconds"],
                y=VideoData1["left.x"],
                mode="lines",
                name="left.x",
            ),
            row=1,
            col=1,
        )
        fig1.add_trace(
            go.Scatter(
                x=VideoData1["Seconds"],
                y=VideoData1["center.x"],
                mode="lines",
                name="center.x",
            ),
            row=1,
            col=1,
        )
        fig1.add_trace(
            go.Scatter(
                x=VideoData1["Seconds"],
                y=VideoData1["right.x"],
                mode="lines",
                name="right.x",
            ),
            row=1,
            col=1,
        )

        # Row 2: Plot left.y, center.y, right.y
        fig1.add_trace(
            go.Scatter(
                x=VideoData1["Seconds"],
                y=VideoData1["left.y"],
                mode="lines",
                name="left.y",
            ),
            row=2,
            col=1,
        )
        fig1.add_trace(
            go.Scatter(
                x=VideoData1["Seconds"],
                y=VideoData1["center.y"],
                mode="lines",
                name="center.y",
            ),
            row=2,
            col=1,
        )
        fig1.add_trace(
            go.Scatter(
                x=VideoData1["Seconds"],
                y=VideoData1["right.y"],
                mode="lines",
                name="right.y",
            ),
            row=2,
            col=1,
        )

        # Row 3: Plot p.x coordinates for p1 to p8
        for col in ["p1.x", "p2.x", "p3.x", "p4.x", "p5.x", "p6.x", "p7.x", "p8.x"]:
            fig1.add_trace(
                go.Scatter(
                    x=VideoData1["Seconds"], y=VideoData1[col], mode="lines", name=col
                ),
                row=3,
                col=1,
            )

        # Row 4: Plot p.y coordinates for p1 to p8
        for col in ["p1.y", "p2.y", "p3.y", "p4.y", "p5.y", "p6.y", "p7.y", "p8.y"]:
            fig1.add_trace(
                go.Scatter(
                    x=VideoData1["Seconds"], y=VideoData1[col], mode="lines", name=col
                ),
                row=4,
                col=1,
            )

        fig1.update_layout(
            height=1200,
            title_text="VideoData1 - Time series subplots for coordinates (QC after interpolation)",
            showlegend=True,
        )
        fig1.update_xaxes(title_text="Seconds", row=4, col=1)
        fig1.update_yaxes(title_text="X Position", row=1, col=1)
        fig1.update_yaxes(title_text="Y Position", row=2, col=1)
        fig1.update_yaxes(title_text="X Position", row=3, col=1)
        fig1.update_yaxes(title_text="Y Position", row=4, col=1)

        fig1.show(renderer="browser")

    # VideoData2 QC Plot
    if "VideoData2_Has_Sleap" in globals() and VideoData2_Has_Sleap:
        fig2 = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "VideoData2 - X coordinates for pupil centre and left-right eye corner",
                "VideoData2 - Y coordinates for pupil centre and left-right eye corner",
                "VideoData2 - X coordinates for iris points",
                "VideoData2 - Y coordinates for iris points",
            ),
        )

        # Row 1: Plot left.x, center.x, right.x
        fig2.add_trace(
            go.Scatter(
                x=VideoData2["Seconds"],
                y=VideoData2["left.x"],
                mode="lines",
                name="left.x",
            ),
            row=1,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=VideoData2["Seconds"],
                y=VideoData2["center.x"],
                mode="lines",
                name="center.x",
            ),
            row=1,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=VideoData2["Seconds"],
                y=VideoData2["right.x"],
                mode="lines",
                name="right.x",
            ),
            row=1,
            col=1,
        )

        # Row 2: Plot left.y, center.y, right.y
        fig2.add_trace(
            go.Scatter(
                x=VideoData2["Seconds"],
                y=VideoData2["left.y"],
                mode="lines",
                name="left.y",
            ),
            row=2,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=VideoData2["Seconds"],
                y=VideoData2["center.y"],
                mode="lines",
                name="center.y",
            ),
            row=2,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=VideoData2["Seconds"],
                y=VideoData2["right.y"],
                mode="lines",
                name="right.y",
            ),
            row=2,
            col=1,
        )

        # Row 3: Plot p.x coordinates for p1 to p8

        for col in ["p1.x", "p2.x", "p3.x", "p4.x", "p5.x", "p6.x", "p7.x", "p8.x"]:
            fig2.add_trace(
                go.Scatter(
                    x=VideoData2["Seconds"], y=VideoData2[col], mode="lines", name=col
                ),
                row=3,
                col=1,
            )

        # Row 4: Plot p.y coordinates for p1 to p8

        for col in ["p1.y", "p2.y", "p3.y", "p4.y", "p5.y", "p6.y", "p7.y", "p8.y"]:
            fig2.add_trace(
                go.Scatter(
                    x=VideoData2["Seconds"], y=VideoData2[col], mode="lines", name=col
                ),
                row=4,
                col=1,
            )

        fig2.update_layout(
            height=1200,
            title_text="VideoData2 - Time series subplots for coordinates (QC after interpolation)",
            showlegend=True,
        )
        fig2.update_xaxes(title_text="Seconds", row=4, col=1)
        fig2.update_yaxes(title_text="X Position", row=1, col=1)
        fig2.update_yaxes(title_text="Y Position", row=2, col=1)
        fig2.update_yaxes(title_text="X Position", row=3, col=1)
        fig2.update_yaxes(title_text="Y Position", row=4, col=1)

        fig2.show(renderer="browser")


# +
# QC plot XY coordinate distributions after NaN are interpolated
################################################################

columns_of_interest = [
    "left.x",
    "left.y",
    "center.x",
    "center.y",
    "right.x",
    "right.y",
    "p1.x",
    "p1.y",
    "p2.x",
    "p2.y",
    "p3.x",
    "p3.y",
    "p4.x",
    "p4.y",
    "p5.x",
    "p5.y",
    "p6.x",
    "p6.y",
    "p7.x",
    "p7.y",
    "p8.x",
    "p8.y",
]

# Create coordinates_dict for both datasets
if VideoData1_Has_Sleap:
    coordinates_dict1_processed = lp.get_coordinates_dict(
        VideoData1, columns_of_interest
    )

if VideoData2_Has_Sleap:
    coordinates_dict2_processed = lp.get_coordinates_dict(
        VideoData2, columns_of_interest
    )


columns_of_interest = [
    "left",
    "right",
    "center",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
    "p6",
    "p7",
    "p8",
]

# Filter out NaN values and calculate the min and max values for X and Y
# coordinates for both dict1 and dict2

def min_max_dict(coordinates_dict):
    x_min = min(
        [
            coordinates_dict[f"{col}.x"][~np.isnan(coordinates_dict[f"{col}.x"])].min()
            for col in columns_of_interest
        ]
    )

    x_max = max(
        [
            coordinates_dict[f"{col}.x"][~np.isnan(coordinates_dict[f"{col}.x"])].max()
            for col in columns_of_interest
        ]
    )

    y_min = min(
        [
            coordinates_dict[f"{col}.y"][~np.isnan(coordinates_dict[f"{col}.y"])].min()
            for col in columns_of_interest
        ]
    )

    y_max = max(
        [
            coordinates_dict[f"{col}.y"][~np.isnan(coordinates_dict[f"{col}.y"])].max()
            for col in columns_of_interest
        ]
    )

    return x_min, x_max, y_min, y_max


if plot_QC_timeseries:
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
        fontsize=14,
    )

    # Define colormap for p1-p8
    colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "orange"]

    # Panel 1: left, right, center (VideoData1)
    if VideoData1_Has_Sleap:
        ax[0, 0].set_title(f"{get_eye_label('VideoData1')}: left, right, center")

        ax[0, 0].scatter(
            coordinates_dict1_processed["left.x"],
            coordinates_dict1_processed["left.y"],
            color="black",
            label="left",
            s=10,
        )

        ax[0, 0].scatter(
            coordinates_dict1_processed["right.x"],
            coordinates_dict1_processed["right.y"],
            color="grey",
            label="right",
            s=10,
        )

        ax[0, 0].scatter(
            coordinates_dict1_processed["center.x"],
            coordinates_dict1_processed["center.y"],
            color="red",
            label="center",
            s=10,
        )

        ax[0, 0].set_xlim([x_min, x_max])

        ax[0, 0].set_ylim([y_min, y_max])

        ax[0, 0].set_xlabel("x coordinates (pixels)")

        ax[0, 0].set_ylabel("y coordinates (pixels)")

        ax[0, 0].legend(loc="upper right")

        # Panel 2: p1 to p8 (VideoData1)

        ax[0, 1].set_title(f"{get_eye_label('VideoData1')}: p1 to p8")

        for idx, col in enumerate(columns_of_interest[3:]):
            ax[0, 1].scatter(
                coordinates_dict1_processed[f"{col}.x"],
                coordinates_dict1_processed[f"{col}.y"],
                color=colors[idx],
                label=col,
                s=5,
            )

        ax[0, 1].set_xlim([x_min, x_max])

        ax[0, 1].set_ylim([y_min, y_max])

        ax[0, 1].set_xlabel("x coordinates (pixels)")

        ax[0, 1].set_ylabel("y coordinates (pixels)")

        ax[0, 1].legend(loc="upper right")

    # Panel 3: left, right, center (VideoData2)
    if VideoData2_Has_Sleap:
        ax[1, 0].set_title(f"{get_eye_label('VideoData2')}: left, right, center")

        ax[1, 0].scatter(
            coordinates_dict2_processed["left.x"],
            coordinates_dict2_processed["left.y"],
            color="black",
            label="left",
            s=10,
        )

        ax[1, 0].scatter(
            coordinates_dict2_processed["right.x"],
            coordinates_dict2_processed["right.y"],
            color="grey",
            label="right",
            s=10,
        )

        ax[1, 0].scatter(
            coordinates_dict2_processed["center.x"],
            coordinates_dict2_processed["center.y"],
            color="red",
            label="center",
            s=10,
        )

        ax[1, 0].set_xlim([x_min, x_max])

        ax[1, 0].set_ylim([y_min, y_max])

        ax[1, 0].set_xlabel("x coordinates (pixels)")

        ax[1, 0].set_ylabel("y coordinates (pixels)")

        ax[1, 0].legend(loc="upper right")

        # Panel 4: p1 to p8 (VideoData2)

        ax[1, 1].set_title(f"{get_eye_label('VideoData2')}: p1 to p8")

        for idx, col in enumerate(columns_of_interest[3:]):
            ax[1, 1].scatter(
                coordinates_dict2_processed[f"{col}.x"],
                coordinates_dict2_processed[f"{col}.y"],
                color=colors[idx],
                label=col,
                s=5,
            )

        ax[1, 1].set_xlim([x_min, x_max])

        ax[1, 1].set_ylim([y_min, y_max])

        ax[1, 1].set_xlabel("x coordinates (pixels)")

        ax[1, 1].set_ylabel("y coordinates (pixels)")

        ax[1, 1].legend(loc="upper right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

# +
# fit ellipses on the 8 points to determine pupil centre and diameter
##########################################################################

columns_of_interest = [
    "left.x",
    "left.y",
    "center.x",
    "center.y",
    "right.x",
    "right.y",
    "p1.x",
    "p1.y",
    "p2.x",
    "p2.y",
    "p3.x",
    "p3.y",
    "p4.x",
    "p4.y",
    "p5.x",
    "p5.y",
    "p6.x",
    "p6.y",
    "p7.x",
    "p7.y",
    "p8.x",
    "p8.y",
]

# VideoData1 processing
if VideoData1_Has_Sleap:
    print("=== VideoData1 Ellipse Fitting for Pupil Diameter ===")
    coordinates_dict1_processed = lp.get_coordinates_dict(
        VideoData1, columns_of_interest
    )

    theta1 = lp.find_horizontal_axis_angle(VideoData1, "left", "center")
    center_point1 = lp.get_left_right_center_point(coordinates_dict1_processed)

    columns_of_interest_reformatted = [
        "left",
        "right",
        "center",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
    ]
    remformatted_coordinates_dict1 = lp.get_reformatted_coordinates_dict(
        coordinates_dict1_processed, columns_of_interest_reformatted
    )
    centered_coordinates_dict1 = lp.get_centered_coordinates_dict(
        remformatted_coordinates_dict1, center_point1
    )
    rotated_coordinates_dict1 = lp.get_rotated_coordinates_dict(
        centered_coordinates_dict1, theta1
    )

    columns_of_interest_ellipse = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
    ellipse_parameters_data1, ellipse_center_points_data1 = (
        lp.get_fitted_ellipse_parameters(
            rotated_coordinates_dict1, columns_of_interest_ellipse
        )
    )

    average_diameter1 = np.mean(
        [ellipse_parameters_data1[:, 0], ellipse_parameters_data1[:, 1]], axis=0
    )

    SleapVideoData1 = process.convert_arrays_to_dataframe(
        [
            "Seconds",
            "Ellipse.Diameter",
            "Ellipse.Angle",
            "Ellipse.Center.X",
            "Ellipse.Center.Y",
        ],
        [
            VideoData1["Seconds"].values,
            average_diameter1,
            ellipse_parameters_data1[:, 2],
            ellipse_center_points_data1[:, 0],
            ellipse_center_points_data1[:, 1],
        ],
    )

# VideoData2 processing
if VideoData2_Has_Sleap:
    print("=== VideoData2 Ellipse Fitting for Pupil Diameter ===")
    coordinates_dict2_processed = lp.get_coordinates_dict(
        VideoData2, columns_of_interest
    )

    theta2 = lp.find_horizontal_axis_angle(VideoData2, "left", "center")
    center_point2 = lp.get_left_right_center_point(coordinates_dict2_processed)

    columns_of_interest_reformatted = [
        "left",
        "right",
        "center",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
    ]
    remformatted_coordinates_dict2 = lp.get_reformatted_coordinates_dict(
        coordinates_dict2_processed, columns_of_interest_reformatted
    )
    centered_coordinates_dict2 = lp.get_centered_coordinates_dict(
        remformatted_coordinates_dict2, center_point2
    )
    rotated_coordinates_dict2 = lp.get_rotated_coordinates_dict(
        centered_coordinates_dict2, theta2
    )

    columns_of_interest_ellipse = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
    ellipse_parameters_data2, ellipse_center_points_data2 = (
        lp.get_fitted_ellipse_parameters(
            rotated_coordinates_dict2, columns_of_interest_ellipse
        )
    )

    average_diameter2 = np.mean(
        [ellipse_parameters_data2[:, 0], ellipse_parameters_data2[:, 1]], axis=0
    )

    SleapVideoData2 = process.convert_arrays_to_dataframe(
        [
            "Seconds",
            "Ellipse.Diameter",
            "Ellipse.Angle",
            "Ellipse.Center.X",
            "Ellipse.Center.Y",
        ],
        [
            VideoData2["Seconds"].values,
            average_diameter2,
            ellipse_parameters_data2[:, 2],
            ellipse_center_points_data2[:, 0],
            ellipse_center_points_data2[:, 1],
        ],
    )


# Filter pupil diameter using a user set (default 10) Hz Butterworth low-pass filter
##########################################################################

# VideoData1 filtering
if VideoData1_Has_Sleap:
    print("\n=== Filtering pupil diameter for VideoData1  ===")
    # Butterworth filter parameters - pupil_filter_cutoff_hz low-pass filter
    # Sampling frequency (Hz)
    fs1 = 1 / np.median(np.diff(SleapVideoData1["Seconds"]))
    order = pupil_filter_order

    b1, a1 = butter(order, pupil_filter_cutoff_hz / (0.5 * fs1), btype="low")

    # Handle NaN values before filtering (from blink detection)
    # Replace NaN with forward-fill for filtering purposes only (to avoid
    # filtfilt issues)
    diameter_data = SleapVideoData1["Ellipse.Diameter"].copy()
    # Use ffill() and bfill() instead of deprecated fillna(method='ffill')
    diameter_data_filled = diameter_data.ffill().bfill()

    # Apply Butterworth filter
    if not diameter_data_filled.isna().all():
        filtered = filtfilt(b1, a1, diameter_data_filled)
        # Restore NaN values at original NaN positions (from blinks)
        filtered = pd.Series(filtered, index=diameter_data.index)
        filtered[diameter_data.isna()] = np.nan
        SleapVideoData1["Ellipse.Diameter.Filt"] = filtered
    else:
        # If all values are NaN, just copy
        SleapVideoData1["Ellipse.Diameter.Filt"] = diameter_data

# VideoData2 filtering
if VideoData2_Has_Sleap:
    print("=== Filtering pupil diameter for VideoData1 ===")
    # Butterworth filter parameters - pupil_filter_cutoff_hz low-pass filter
    # Sampling frequency (Hz)
    fs2 = 1 / np.median(np.diff(SleapVideoData2["Seconds"]))
    order = pupil_filter_order

    b2, a2 = butter(order, pupil_filter_cutoff_hz / (0.5 * fs2), btype="low")

    # Handle NaN values before filtering (from blink detection)
    # Replace NaN with forward-fill for filtering purposes only (to avoid
    # filtfilt issues)
    diameter_data = SleapVideoData2["Ellipse.Diameter"].copy()
    # Use ffill() and bfill() instead of deprecated fillna(method='ffill')
    diameter_data_filled = diameter_data.ffill().bfill()

    # Apply Butterworth filter
    if not diameter_data_filled.isna().all():
        filtered = filtfilt(b2, a2, diameter_data_filled)
        # Restore NaN values at original NaN positions (from blinks)
        filtered = pd.Series(filtered, index=diameter_data.index)
        filtered[diameter_data.isna()] = np.nan
        SleapVideoData2["Ellipse.Diameter.Filt"] = filtered
    else:
        # If all values are NaN, just copy
        SleapVideoData2["Ellipse.Diameter.Filt"] = diameter_data

print("✅ Done calculating pupil diameter and angle for both VideoData1 and VideoData2")

# +
# cross-correlate pupil diameter for left and right eye
##########################################################################

if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    # Cross-correlation analysis
    print("=== Cross-Correlation Analysis ===")

    # Get pupil diameter data
    # Use filtered diameter data (with NaN restored at blink positions)
    pupil1 = SleapVideoData1["Ellipse.Diameter.Filt"].values
    pupil2 = SleapVideoData2["Ellipse.Diameter.Filt"].values

    # Handle different lengths by using the shorter dataset length
    min_length = min(len(pupil1), len(pupil2))

    # Truncate both datasets to the same length (preserving time alignment)
    pupil1_truncated = pupil1[:min_length]
    pupil2_truncated = pupil2[:min_length]

    # Remove NaN values for correlation - preserve time alignment by only keeping pairs where BOTH are valid
    # This ensures cross-correlation is computed on temporally aligned data
    valid_mask1 = ~np.isnan(pupil1_truncated)
    valid_mask2 = ~np.isnan(pupil2_truncated)
    # Only use indices where both arrays have valid data
    valid_mask = valid_mask1 & valid_mask2

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
            print(
                "Applied z-score normalization to pupil diameter signals (accounts for different camera magnifications)"
            )
            print(f"  VideoData1: mean={pupil1_mean:.2f}, std={pupil1_std:.2f}")
            print(f"  VideoData2: mean={pupil2_mean:.2f}, std={pupil2_std:.2f}")
        else:
            print(
                "⚠️ Warning: Zero variance detected, using raw signals (no normalization)"
            )
            pupil1_z = pupil1_clean
            pupil2_z = pupil2_clean

        # Calculate cross-correlation using z-scored signals
        try:
            correlation = correlate(pupil1_z, pupil2_z, mode="full")

            # Calculate lags (in samples)
            lags = np.arange(-len(pupil2_z) + 1, len(pupil1_z))

            # Convert lags to time (assuming same sampling rate)
            dt = np.median(np.diff(SleapVideoData1["Seconds"]))
            lag_times = lags * dt

            # Find peak correlation and corresponding lag
            peak_idx = np.argmax(correlation)
            peak_correlation = correlation[peak_idx]
            peak_lag_samples = lags[peak_idx]
            peak_lag_time = lag_times[peak_idx]
            peak_lag_time_display = peak_lag_time  # for final QC figure

            print(f"Peak lag (time): {peak_lag_time:.4f} seconds")

            # Normalize correlation to [-1, 1] range (for z-scored signals,
            # this is standard normalization)
            norm_factor = np.sqrt(np.sum(pupil1_z**2) * np.sum(pupil2_z**2))
            if norm_factor > 0:
                correlation_normalized = correlation / norm_factor
                peak_correlation_normalized = correlation_normalized[peak_idx]
                print(f"Peak normalized correlation: {peak_correlation_normalized:.4f}")
            else:
                print("❌ Error: Cannot normalize correlation (zero variance)")
                correlation_normalized = correlation
                peak_correlation_normalized = 0

        except Exception as e:
            print(f"❌ Error in cross-correlation calculation: {e}")

    # Additional correlation statistics
    if len(pupil1_clean) >= 2 and len(pupil2_clean) >= 2:
        try:
            # Calculate Pearson correlation coefficient on z-scored signals
            # Note: For z-scored signals, Pearson correlation is equivalent to
            # the normalized cross-correlation at zero lag
            pearson_r, pearson_p = pearsonr(pupil1_z, pupil2_z)
            pearson_r_display = pearson_r
            pearson_p_display = pearson_p

            print("\n=== Additional Statistics ===")
            print(f"Pearson correlation coefficient: {pearson_r:.2f}")

            # Handle extremely small p-values
            if pearson_p < 1e-300:
                print("Pearson p-value: < 1e-300 (extremely significant)")
            else:
                print(f"Pearson p-value: {pearson_p:.5e}")

        except Exception as e:
            print(f"❌ Error in additional statistics: {e}")
            pearson_r_display = None
            pearson_p_display = None
    else:
        print("❌ Cannot calculate additional statistics - insufficient data")
else:
    print("Only one eye is present, no pupil diameter cross-correlation can be done")

# +
# check if Second values match 1:1 between VideoData and SleapVideoData then merge them into VideoData
##########################################################################

if VideoData1_Has_Sleap is True:
    if VideoData1["Seconds"].equals(SleapVideoData1["Seconds"]) is False:
        print(
            f"❗ {get_eye_label('VideoData1')}: The 'Seconds' columns DO NOT correspond 1:1 between the two DataFrames. This should not happen"
        )
    else:
        VideoData1 = VideoData1.merge(SleapVideoData1, on="Seconds", how="outer")
        del SleapVideoData1

if VideoData2_Has_Sleap is True:
    if VideoData2["Seconds"].equals(SleapVideoData2["Seconds"]) is False:
        print(
            f"❗ {get_eye_label('VideoData2')}: The 'Seconds' columns DO NOT correspond 1:1 between the two DataFrames. This should not happen"
        )
    else:
        VideoData2 = VideoData2.merge(SleapVideoData2, on="Seconds", how="outer")
        del SleapVideoData2
gc.collect()
None

# +
# Compare SLEAP center.x and .y with fitted ellipse centre distributions for both VideoData1 and VideoData2
##########################################################################

# ------------------------------------------------------------------
# 1) Compute correlations for VideoData1
# ------------------------------------------------------------------
if VideoData1_Has_Sleap is True:
    print("=== VideoData1 Analysis ===")
    slope_x1, intercept_x1, r_value_x1, p_value_x1, std_err_x1 = linregress(
        VideoData1["Ellipse.Center.X"], VideoData1["center.x"]
    )
    r_squared_x1 = r_value_x1**2
    print(
        f"{get_eye_label('VideoData1')} - R^2 between center point and ellipse center X data: {r_squared_x1:.4f}"
    )

    slope_y1, intercept_y1, r_value_y1, p_value_y1, std_err_y1 = linregress(
        VideoData1["Ellipse.Center.Y"], VideoData1["center.y"]
    )
    r_squared_y1 = r_value_y1**2
    print(
        f"{get_eye_label('VideoData1')} - R^2 between center point and ellipse center Y data: {r_squared_y1:.4f}"
    )

# ------------------------------------------------------------------
# 2) Compute correlations for VideoData2
# ------------------------------------------------------------------
if VideoData2_Has_Sleap is True:
    print("\n=== VideoData2 Analysis ===")
    slope_x2, intercept_x2, r_value_x2, p_value_x2, std_err_x2 = linregress(
        VideoData2["Ellipse.Center.X"], VideoData2["center.x"]
    )
    r_squared_x2 = r_value_x2**2
    print(
        f"{get_eye_label('VideoData2')} - R^2 between center point and ellipse center X data: {r_squared_x2:.4f}"
    )

    slope_y2, intercept_y2, r_value_y2, p_value_y2, std_err_y2 = linregress(
        VideoData2["Ellipse.Center.Y"], VideoData2["center.y"]
    )
    r_squared_y2 = r_value_y2**2
    print(
        f"{get_eye_label('VideoData2')} - R^2 between center point and ellipse center Y data: {r_squared_y2:.4f}"
    )

# ------------------------------------------------------------------
# 3) Center of Mass Analysis (if both VideoData1 and VideoData2 are available)
# ------------------------------------------------------------------
if VideoData1_Has_Sleap is True and VideoData2_Has_Sleap is True:
    print("\n=== Center of Mass Distance Analysis ===")

    # Calculate center of mass (mean) for VideoData1
    com_center_x1 = VideoData1["center.x"].mean()
    com_center_y1 = VideoData1["center.y"].mean()
    com_ellipse_x1 = VideoData1["Ellipse.Center.X"].mean()
    com_ellipse_y1 = VideoData1["Ellipse.Center.Y"].mean()

    # Calculate absolute distances for VideoData1

    dist_x1 = abs(com_center_x1 - com_ellipse_x1)

    dist_y1 = abs(com_center_y1 - com_ellipse_y1)

    print(f"\n{get_eye_label('VideoData1')}:")

    print(
        f"  Center of mass for center.x/y: ({com_center_x1:.4f}, {com_center_y1:.4f})"
    )

    print(
        f"  Center of mass for Ellipse.Center.X/Y: ({com_ellipse_x1:.4f}, {com_ellipse_y1:.4f})"
    )

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

    print(
        f"  Center of mass for center.x/y: ({com_center_x2:.4f}, {com_center_y2:.4f})"
    )

    print(
        f"  Center of mass for Ellipse.Center.X/Y: ({com_ellipse_x2:.4f}, {com_ellipse_y2:.4f})"
    )

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

    print(
        f"{get_eye_label('VideoData1')} - Re-centered Ellipse.Center using median: ({median_ellipse_x1:.4f}, {median_ellipse_y1:.4f})"
    )

# Re-center VideoData2 Ellipse.Center coordinates
if VideoData2_Has_Sleap is True:
    # Calculate median
    median_ellipse_x2 = VideoData2["Ellipse.Center.X"].median()
    median_ellipse_y2 = VideoData2["Ellipse.Center.Y"].median()

    # Center the coordinates
    VideoData2["Ellipse.Center.X"] = VideoData2["Ellipse.Center.X"] - median_ellipse_x2
    VideoData2["Ellipse.Center.Y"] = VideoData2["Ellipse.Center.Y"] - median_ellipse_y2

    print(
        f"{get_eye_label('VideoData2')} - Re-centered Ellipse.Center using median: ({median_ellipse_x2:.4f}, {median_ellipse_y2:.4f})"
    )


# +
# Make and save summary QC plot using matplotlib with scatter plots for 2D
# distributions

# Initialize the statistics variables (these are calculated in Cell 11)
try:
    pearson_r_display
except NameError:
    pearson_r_display = None
    pearson_p_display = None
    peak_lag_time_display = None
    print("⚠️ Note: Statistics not found. They should be calculated in Cell 11.")

# Visualization and statistics calculation (only if plot_QC_timeseries is True)
if plot_QC_timeseries:
    # Calculate correlation for Ellipse.Center.X between VideoData1 and
    # VideoData2 (if both exist)
    pearson_r_center = None
    pearson_p_center = None
    peak_lag_time_center = None
else:
    pearson_r_center = None
    pearson_p_center = None
    peak_lag_time_center = None


if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    # Get the Center.X data
    center_x1 = VideoData1["Ellipse.Center.X"].values
    center_x2 = VideoData2["Ellipse.Center.X"].values
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
            pearson_r_center, pearson_p_center = pearsonr(
                center_x1_clean, center_x2_clean
            )
            # Calculate cross-correlation for peak lag
            correlation = correlate(center_x1_clean, center_x2_clean, mode="full")
            lags = np.arange(-len(center_x2_clean) + 1, len(center_x1_clean))
            dt = np.median(np.diff(VideoData1["Seconds"]))
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
    ax1.plot(
        VideoData1_centered["Seconds"],
        VideoData1_centered["center.x"],
        linewidth=0.5,
        c="blue",
        alpha=0.6,
        label="center.x original",
    )
    ax1.plot(
        VideoData1["Seconds"],
        VideoData1["Ellipse.Center.X"],
        linewidth=0.5,
        c="red",
        alpha=0.6,
        label="Ellipse Center.X",
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (pixels)")
    ax1.set_title(f"{get_eye_label('VideoData1')} - center.X Time Series")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# Panel 2: VideoData2 center coordinates - Time Series (full width)
if VideoData2_Has_Sleap:
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(
        VideoData2_centered["Seconds"],
        VideoData2_centered["center.x"],
        linewidth=0.5,
        c="blue",
        alpha=0.6,
        label="center.x original",
    )
    ax2.plot(
        VideoData2["Seconds"],
        VideoData2["Ellipse.Center.X"],
        linewidth=0.5,
        c="red",
        alpha=0.6,
        label="Ellipse Center.X",
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (pixels)")
    ax2.set_title(f"{get_eye_label('VideoData2')} - center.X Time Series")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Panel 3: VideoData1 center coordinates - Scatter plot (left half)
if VideoData1_Has_Sleap:
    ax3 = fig.add_subplot(gs[2, 0])

    # Ellipse.Center (blue)
    x_ellipse1 = VideoData1["Ellipse.Center.X"].to_numpy()
    y_ellipse1 = VideoData1["Ellipse.Center.Y"].to_numpy()
    mask1 = ~(np.isnan(x_ellipse1) | np.isnan(y_ellipse1))

    ax3.scatter(
        x_ellipse1[mask1],
        y_ellipse1[mask1],
        s=1,
        alpha=0.3,
        c="blue",
        label="Ellipse.Center",
    )

    # Center (red) - from centered data
    x_center1 = VideoData1_centered["center.x"].to_numpy()
    y_center1 = VideoData1_centered["center.y"].to_numpy()
    mask2 = ~(np.isnan(x_center1) | np.isnan(y_center1))

    ax3.scatter(
        x_center1[mask2],
        y_center1[mask2],
        s=1,
        alpha=0.3,
        c="red",
        label="center.x original",
    )

    ax3.set_xlabel("Center X (pixels)")
    ax3.set_ylabel("Center Y (pixels)")
    ax3.set_title(
        f"{get_eye_label('VideoData1')} - Center X-Y Distribution (center.X vs Ellipse)"
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add R² statistics for VideoData1 (bottom left)
    try:
        if "r_squared_x1" in globals() and "r_squared_y1" in globals():
            stats_text = f"R² X: {r_squared_x1:.2g}\nR² Y: {r_squared_y1:.2g}"
            ax3.text(
                0.02,
                0.02,
                stats_text,
                transform=ax3.transAxes,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                fontsize=9,
                family="monospace",
            )
    except BaseException:
        pass

    # Add center of mass distance for VideoData1 (bottom right)
    try:
        if "dist_x1" in globals() and "dist_y1" in globals():
            distance_text = f"COM Dist X: {dist_x1:.3g}\nCOM Dist Y: {dist_y1:.3g}"
            ax3.text(
                0.98,
                0.02,
                distance_text,
                transform=ax3.transAxes,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                fontsize=9,
                family="monospace",
            )
    except BaseException:
        pass

# Panel 4: VideoData2 center coordinates - Scatter plot (right half)
if VideoData2_Has_Sleap:
    ax4 = fig.add_subplot(gs[2, 1])

    # Ellipse.Center (blue)
    x_ellipse2 = VideoData2["Ellipse.Center.X"].to_numpy()
    y_ellipse2 = VideoData2["Ellipse.Center.Y"].to_numpy()
    mask3 = ~(np.isnan(x_ellipse2) | np.isnan(y_ellipse2))

    ax4.scatter(
        x_ellipse2[mask3],
        y_ellipse2[mask3],
        s=1,
        alpha=0.3,
        c="blue",
        label="Ellipse.Center",
    )

    # Center (red) - from centered data
    x_center2 = VideoData2_centered["center.x"].to_numpy()
    y_center2 = VideoData2_centered["center.y"].to_numpy()
    mask4 = ~(np.isnan(x_center2) | np.isnan(y_center2))

    ax4.scatter(
        x_center2[mask4],
        y_center2[mask4],
        s=1,
        alpha=0.3,
        c="red",
        label="center.X Center",
    )

    ax4.set_xlabel("Center X (pixels)")
    ax4.set_ylabel("Center Y (pixels)")
    ax4.set_title(
        f"{get_eye_label('VideoData2')} - Center X-Y Distribution (center.X vs Ellipse)"
    )
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add R² statistics for VideoData2 (bottom left)
    try:
        if "r_squared_x2" in globals() and "r_squared_y2" in globals():
            stats_text = f"R² X: {r_squared_x2:.2g}\nR² Y: {r_squared_y2:.2g}"
            ax4.text(
                0.02,
                0.02,
                stats_text,
                transform=ax4.transAxes,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                fontsize=9,
                family="monospace",
            )
    except BaseException:
        pass

    # Add center of mass distance for VideoData2 (bottom right)
    try:
        if "dist_x2" in globals() and "dist_y2" in globals():
            distance_text = f"COM Dist X: {dist_x2:.3g}\nCOM Dist Y: {dist_y2:.3g}"
            ax4.text(
                0.98,
                0.02,
                distance_text,
                transform=ax4.transAxes,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                fontsize=9,
                family="monospace",
            )
    except BaseException:
        pass

# Panel 5: Pupil diameter comparison (bottom left)
ax5 = fig.add_subplot(gs[3, 0])
if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    ax5.plot(
        VideoData1["Seconds"],
        VideoData1["Ellipse.Diameter.Filt"],
        linewidth=0.5,
        c="#FF7F00",
        alpha=0.6,
        label="VideoData1 Diameter",
    )
    ax5.plot(
        VideoData2["Seconds"],
        VideoData2["Ellipse.Diameter.Filt"],
        linewidth=0.5,
        c="#9370DB",
        alpha=0.6,
        label="VideoData2 Diameter",
    )
elif VideoData1_Has_Sleap:
    ax5.plot(
        VideoData1["Seconds"],
        VideoData1["Ellipse.Diameter.Filt"],
        linewidth=0.5,
        c="#FF7F00",
        alpha=0.6,
        label="VideoData1 Diameter",
    )
elif VideoData2_Has_Sleap:
    ax5.plot(
        VideoData2["Seconds"],
        VideoData2["Ellipse.Diameter.Filt"],
        linewidth=0.5,
        c="#9370DB",
        alpha=0.6,
        label="VideoData2 Diameter",
    )

ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Diameter (pixels)")
ax5.set_title("Pupil Diameter Comparison")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Add statistics text to Panel 5
if (
    pearson_r_display is not None
    and pearson_p_display is not None
    and peak_lag_time_display is not None
):
    stats_text = (
        f"Pearson r = {pearson_r_display:.4f}\n"
        f"Pearson p = {pearson_p_display:.4e}\n"
        f"Peak lag = {peak_lag_time_display:.4f} s"
    )
    ax5.text(
        0.98,
        0.98,
        stats_text,
        transform=ax5.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=10,
        family="monospace",
    )
else:
    ax5.text(
        0.5,
        0.5,
        "Statistics not available\n(See Cell 11 for correlation analysis)",
        transform=ax5.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )

# Panel 6: Ellipse.Center.X comparison (bottom right) with dual y-axis
ax6 = fig.add_subplot(gs[3, 1])
ax6_twin = ax6.twinx()  # Create a second y-axis

if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    # Plot the individual traces
    ax6.plot(
        VideoData1["Seconds"],
        VideoData1["Ellipse.Center.X"],
        linewidth=0.5,
        c="#FF7F00",
        alpha=0.6,
        label="VideoData1 Ellipse.Center.X",
    )
    ax6.plot(
        VideoData2["Seconds"],
        VideoData2["Ellipse.Center.X"],
        linewidth=0.5,
        c="#9370DB",
        alpha=0.6,
        label="VideoData2 Ellipse.Center.X",
    )

    # Plot the difference on the right axis
    # Align the data to the same length and normalize for fair comparison
    min_length = min(len(VideoData1), len(VideoData2))

    # Normalize data (z-score) to account for different scales
    center_x1_aligned = VideoData1["Ellipse.Center.X"].iloc[:min_length]
    center_x2_aligned = VideoData2["Ellipse.Center.X"].iloc[:min_length]

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
    seconds_aligned = VideoData1["Seconds"].iloc[:min_length]
    ax6_twin.plot(
        seconds_aligned,
        center_x_diff,
        linewidth=0.5,
        c="green",
        alpha=0.6,
        label="Difference (normalized)",
    )

elif VideoData1_Has_Sleap:
    ax6.plot(
        VideoData1["Seconds"],
        VideoData1["Ellipse.Center.X"],
        linewidth=0.5,
        c="#FF7F00",
        alpha=0.6,
        label="VideoData1 Ellipse.Center.X",
    )
elif VideoData2_Has_Sleap:
    ax6.plot(
        VideoData2["Seconds"],
        VideoData2["Ellipse.Center.X"],
        linewidth=0.5,
        c="#9370DB",
        alpha=0.6,
        label="VideoData2 Ellipse.Center.X",
    )

ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Center X (pixels)", color="black")
ax6.set_title("Ellipse.Center.X Comparison")
ax6.tick_params(axis="y", labelcolor="black")
if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    ax6_twin.set_ylabel("Normalized Difference (z-score)", color="green")
    ax6_twin.tick_params(axis="y", labelcolor="green")

# Combine legends from both axes
lines1, labels1 = ax6.get_legend_handles_labels()
if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
else:
    ax6.legend(loc="upper left")

ax6.grid(True, alpha=0.3)

# Add statistics text to Panel 6
if (
    pearson_r_center is not None
    and pearson_p_center is not None
    and peak_lag_time_center is not None
):
    stats_text = (
        f"Pearson r = {pearson_r_center:.4f}\n"
        f"Pearson p = {pearson_p_center:.4e}\n"
        f"Peak lag = {peak_lag_time_center:.4f} s"
    )
    ax6.text(
        0.98,
        0.98,
        stats_text,
        transform=ax6.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=10,
        family="monospace",
    )
else:
    ax6.text(
        0.5,
        0.5,
        "Statistics not available\n(both eyes required)",
        transform=ax6.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )

# Save as PDF (editable vector format)
save_path.mkdir(parents=True, exist_ok=True)
pdf_path = qc_debug_dir / "Eye_data_QC.pdf"
plt.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
print(f"✅ QC figure saved as PDF (editable): {pdf_path}")

# Also save as 600 dpi PNG (high-resolution for printing)
png_path = qc_debug_dir / "Eye_data_QC.png"
plt.savefig(png_path, dpi=600, bbox_inches="tight", format="png")
print(f"✅ QC figure saved as PNG (600 dpi for printing): {png_path}")

if plot_QC_timeseries:
    plt.show()
else:
    plt.close(fig)
# -

# Create interactive time series plots using plotly for browser viewing
if plot_QC_timeseries:
    # Create subplots for the time series (3 rows now instead of 2)
    # Need to enable secondary_y for the third panel
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"{get_eye_label('VideoData1')} - center.X Time Series",
            f"{get_eye_label('VideoData2')} - center.X Time Series",
            "Ellipse.Center.X Comparison with Difference",
        ),
        # Enable secondary_y for row 3
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    # Panel 1: VideoData1 center coordinates - Time Series
    if VideoData1_Has_Sleap:
        fig.add_trace(
            go.Scatter(
                x=VideoData1_centered["Seconds"],
                y=VideoData1_centered["center.x"],
                mode="lines",
                name="center.x original",
                line=dict(color="blue", width=0.5),
                opacity=0.6,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=VideoData1["Seconds"],
                y=VideoData1["Ellipse.Center.X"],
                mode="lines",
                name="Ellipse Center.X",
                line=dict(color="red", width=0.5),
                opacity=0.6,
            ),
            row=1,
            col=1,
        )

    # Panel 2: VideoData2 center coordinates - Time Series
    if VideoData2_Has_Sleap:
        fig.add_trace(
            go.Scatter(
                x=VideoData2_centered["Seconds"],
                y=VideoData2_centered["center.x"],
                mode="lines",
                name="center.x original",
                line=dict(color="blue", width=0.5),
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=VideoData2["Seconds"],
                y=VideoData2["Ellipse.Center.X"],
                mode="lines",
                name="Ellipse Center.X",
                line=dict(color="red", width=0.5),
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

    # Panel 3: Ellipse.Center.X Comparison with difference
    if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
        # Plot the individual traces
        fig.add_trace(
            go.Scatter(
                x=VideoData1["Seconds"],
                y=VideoData1["Ellipse.Center.X"],
                mode="lines",
                name="VideoData1 Ellipse.Center.X",
                line=dict(color="#FF7F00", width=0.5),  # Orange
                opacity=0.6,
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=VideoData2["Seconds"],
                y=VideoData2["Ellipse.Center.X"],
                mode="lines",
                name="VideoData2 Ellipse.Center.X",
                line=dict(color="#9370DB", width=0.5),  # Purple
                opacity=0.6,
            ),
            row=3,
            col=1,
        )

        # Plot the difference on secondary y-axis
        # Align the data to the same length and normalize for fair comparison
        min_length = min(len(VideoData1), len(VideoData2))

        # Normalize data (z-score) to account for different scales
        center_x1_aligned = VideoData1["Ellipse.Center.X"].iloc[:min_length]
        center_x2_aligned = VideoData2["Ellipse.Center.X"].iloc[:min_length]

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
        seconds_aligned = VideoData1["Seconds"].iloc[:min_length]

        fig.add_trace(
            go.Scatter(
                x=seconds_aligned,
                y=center_x_diff,
                mode="lines",
                name="Difference (normalized)",
                line=dict(color="green", width=0.5),
                opacity=0.6,
            ),
            row=3,
            col=1,
            secondary_y=True,
        )

    elif VideoData1_Has_Sleap:
        fig.add_trace(
            go.Scatter(
                x=VideoData1["Seconds"],
                y=VideoData1["Ellipse.Center.X"],
                mode="lines",
                name="VideoData1 Ellipse.Center.X",
                line=dict(color="#FF7F00", width=0.5),
                opacity=0.6,
            ),
            row=3,
            col=1,
        )
    elif VideoData2_Has_Sleap:
        fig.add_trace(
            go.Scatter(
                x=VideoData2["Seconds"],
                y=VideoData2["Ellipse.Center.X"],
                mode="lines",
                name="VideoData2 Ellipse.Center.X",
                line=dict(color="#9370DB", width=0.5),
                opacity=0.6,
            ),
            row=3,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=1200,  # Increased height for 3 panels
        title_text=f"{data_path} - Eye Tracking Time Series QC",
        showlegend=True,
        hovermode="x unified",
    )

    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Position (pixels)", row=1, col=1)
    fig.update_yaxes(title_text="Position (pixels)", row=2, col=1)
    fig.update_yaxes(title_text="Center X (pixels)", row=3, col=1)

    # Update secondary y-axis for difference plot
    if VideoData1_Has_Sleap and VideoData2_Has_Sleap:
        fig.update_yaxes(
            title_text="Normalized Difference (z-score)", row=3, col=1, secondary_y=True
        )

    # Show in browser
    fig.show(renderer="browser")

    # Also save as HTML
    save_path.mkdir(parents=True, exist_ok=True)
    html_path = qc_debug_dir / "Eye_data_QC_time_series.html"
    fig.write_html(html_path)
    print(f"✅ Interactive time series plot saved to: {html_path}")

# # Saccade detection
#

# +
saccade_results = {}

# Helper: map detected directions (upward/downward) to NT/TN based on eye assignment
# Left eye: upward→NT, downward→TN; Right eye: upward→TN, downward→NT


def get_direction_map_for_video(video_key):
    eye = video1_eye if video_key == "VideoData1" else video2_eye
    if eye == "L":
        return {"upward": "NT", "downward": "TN"}
    else:
        return {"upward": "TN", "downward": "NT"}


def get_eye_code_for_video(video_key):
    """Return 'L' or 'R' for the specified video."""
    return video1_eye if video_key == "VideoData1" else video2_eye


if VideoData1_Has_Sleap:
    df1 = VideoData1[["Ellipse.Center.X", "Seconds", "frame_idx"]].copy()
    dir_map_v1 = get_direction_map_for_video("VideoData1")
    saccade_results["VideoData1"] = analyze_eye_video_saccades(
        df1,
        FPS_1,
        get_eye_label("VideoData1"),
        k=k1,
        refractory_period=refractory_period,
        onset_offset_fraction=onset_offset_fraction,
        pre_saccade_window_time=pre_saccade_window_time,
        post_saccade_window_time=post_saccade_window_time,
        baseline_window_start_time=baseline_window_start_time,
        baseline_window_end_time=baseline_window_end_time,
        smoothing_window_time=smoothing_window_time,
        peak_width_time=peak_width_time,
        min_saccade_duration=min_saccade_duration,
        upward_label=dir_map_v1["upward"],
        downward_label=dir_map_v1["downward"],
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
        adaptive_percentile_post_variance=adaptive_percentile_post_variance,
        debug=debug,
    )


if VideoData2_Has_Sleap:
    df2 = VideoData2[["Ellipse.Center.X", "Seconds", "frame_idx"]].copy()
    dir_map_v2 = get_direction_map_for_video("VideoData2")
    saccade_results["VideoData2"] = analyze_eye_video_saccades(
        df2,
        FPS_2,
        get_eye_label("VideoData2"),
        k=k2,
        refractory_period=refractory_period,
        onset_offset_fraction=onset_offset_fraction,
        pre_saccade_window_time=pre_saccade_window_time,
        post_saccade_window_time=post_saccade_window_time,
        baseline_window_start_time=baseline_window_start_time,
        baseline_window_end_time=baseline_window_end_time,
        smoothing_window_time=smoothing_window_time,
        peak_width_time=peak_width_time,
        min_saccade_duration=min_saccade_duration,
        upward_label=dir_map_v2["upward"],
        downward_label=dir_map_v2["downward"],
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
        adaptive_percentile_post_variance=adaptive_percentile_post_variance,
        debug=debug,
    )

# +
# SAVE SACCADE RESULTS TO FILES
##########################################################################
# Build tidy per-saccade summary tables, merge saccade metadata into VideoData,
# then save enriched DataFrames (parquet) and QC summaries (CSV).


def build_saccade_summary(video_key):
    """Create a tidy per-saccade summary table for the specified video."""
    results = saccade_results.get(video_key)
    if results is None:
        return pd.DataFrame()

    direction_map = get_direction_map_for_video(video_key)
    summary_parts = []
    for direction, df in (
        ("upward", results.get("upward_saccades_df")),
        ("downward", results.get("downward_saccades_df")),
    ):
        if df is None or df.empty:
            continue
        temp = df.copy()
        temp["direction"] = direction
        temp["direction_label"] = direction_map.get(direction, direction)
        summary_parts.append(temp)

    if not summary_parts:
        return pd.DataFrame()

    summary = pd.concat(summary_parts, ignore_index=True)
    summary = summary.sort_values(["start_time", "time"]).reset_index(drop=True)
    summary.insert(0, "saccade_id", np.arange(1, len(summary) + 1, dtype=int))
    summary["video_key"] = video_key
    summary["eye"] = get_eye_code_for_video(video_key)

    # Normalise frame/index columns
    summary["merge_frame_idx"] = pd.to_numeric(
        summary.get("start_frame_idx"), errors="coerce"
    ).astype("Int64")
    for col_name in [
        "start_frame_idx",
        "peak_frame_idx",
        "end_frame_idx",
    ]:
        if col_name in summary.columns:
            summary[col_name] = pd.to_numeric(
                summary[col_name], errors="coerce"
            ).astype("Int64")

    rename_map = {
        "time": "saccade_peak_time",
        "velocity": "saccade_peak_velocity",
        "start_time": "saccade_start_time",
        "end_time": "saccade_end_time",
        "duration": "saccade_duration",
        "start_position": "saccade_start_position",
        "end_position": "saccade_end_position",
        "amplitude": "saccade_amplitude",
        "displacement": "saccade_displacement",
        "start_frame_idx": "saccade_start_frame_idx",
        "peak_frame_idx": "saccade_peak_frame_idx",
        "end_frame_idx": "saccade_end_frame_idx",
    }
    summary = summary.rename(
        columns={k: v for k, v in rename_map.items() if k in summary.columns}
    )

    preferred_order = [
        "saccade_id",
        "video_key",
        "eye",
        "direction",
        "direction_label",
        "saccade_start_time",
        "saccade_end_time",
        "saccade_peak_time",
        "saccade_start_frame_idx",
        "saccade_peak_frame_idx",
        "saccade_end_frame_idx",
        "saccade_peak_velocity",
        "saccade_amplitude",
        "saccade_displacement",
        "saccade_duration",
        "saccade_start_position",
        "saccade_end_position",
        "saccade_type",
        "bout_id",
        "bout_size",
        "pre_saccade_mean_velocity",
        "pre_saccade_position_drift",
        "post_saccade_position_variance",
        "post_saccade_position_change",
        "merge_frame_idx",
    ]
    remaining_cols = [col for col in summary.columns if col not in preferred_order]
    summary = summary[preferred_order + remaining_cols]
    return summary


summary_tables = {}


def merge_summary_into_video(video_df, summary_df):
    """Merge tidy saccade summary columns into the per-frame VideoData DataFrame."""
    if summary_df is None or summary_df.empty:
        return video_df

    summary_columns = [col for col in summary_df.columns if col != "merge_frame_idx"]
    columns_to_drop = [col for col in summary_columns if col in video_df.columns]
    if columns_to_drop:
        video_df = video_df.drop(columns=columns_to_drop)

    summary_for_join = summary_df.set_index("merge_frame_idx")
    summary_for_join = summary_for_join.drop(columns=["merge_frame_idx"], errors="ignore")

    merged = (
        video_df.set_index("frame_idx")
        .join(summary_for_join, how="left")
        .reset_index()
        .rename(columns={"index": "frame_idx"})
    )
    return merged


if "VideoData1" in globals() and "VideoData1" in saccade_results:

    summary_v1 = build_saccade_summary("VideoData1")
    if not summary_v1.empty:
        VideoData1 = merge_summary_into_video(VideoData1, summary_v1)
        summary_tables["VideoData1"] = summary_v1.drop(columns=["merge_frame_idx"])
    else:
        summary_tables["VideoData1"] = summary_v1

if "VideoData2" in globals() and "VideoData2" in saccade_results:
    summary_v2 = build_saccade_summary("VideoData2")
    if not summary_v2.empty:
        VideoData2 = merge_summary_into_video(VideoData2, summary_v2)
        summary_tables["VideoData2"] = summary_v2.drop(columns=["merge_frame_idx"])
    else:
        summary_tables["VideoData2"] = summary_v2

# Save enriched VideoData tables as parquet and tidy summaries as CSV for QC
downsampled_output_dir = save_path / "downsampled_data"
downsampled_output_dir.mkdir(parents=True, exist_ok=True)

if "VideoData1" in globals():
    video_dir1 = qc_debug_dir / "Video_Sleap_Data1"
    if debug:
        video_dir1.mkdir(parents=True, exist_ok=True)
    summary_output_path1 = downsampled_output_dir / "VideoData1_saccade_summary.csv"

    drop_for_export = [col for col in SACCADE_EXPORT_DROP_COLUMNS if col in VideoData1.columns]
    VideoData1_export = VideoData1.drop(columns=drop_for_export)

    VideoData1_export_renamed = (
        VideoData1_export.rename(columns={"Ellipse.Diameter.Filt": "Pupil.Diameter"})
        if "Ellipse.Diameter.Filt" in VideoData1_export.columns
        else VideoData1_export
    )

    if debug:
        VideoData1_full_indexed = set_aeon_index(VideoData1_export_renamed)
        append_aeon_time_column(VideoData1_full_indexed).to_csv(
            video_dir1 / "Video_Sleap_Data1_1904-01-01T00-00-00.csv", index=False
        )

    drop_for_resample = [col for col in RESAMPLED_DROP_COLUMNS if col in VideoData1_export.columns]
    VideoData1_resampled_input = VideoData1_export.drop(columns=drop_for_resample)
    VideoData1_resampled = resample_video_dataframe(VideoData1_resampled_input, "eye1")
    if "Ellipse.Diameter.Filt" in VideoData1_resampled.columns:
        VideoData1_resampled = VideoData1_resampled.rename(columns={"Ellipse.Diameter.Filt": "Pupil.Diameter"})
    VideoData1_resampled_indexed = set_aeon_index(VideoData1_resampled)
    if debug:
        append_aeon_time_column(VideoData1_resampled_indexed).to_csv(
            video_dir1 / "Video_Sleap_Data1_1904-01-01T00-00-00_resampled.csv", index=False
        )
    downsampled_video1_path = downsampled_output_dir / "VideoData1_resampled.parquet"
    VideoData1_resampled_indexed.to_parquet(
        downsampled_video1_path, engine="pyarrow", compression="snappy"
    )
    print(f"✅ Saved VideoData1 resampled parquet to {downsampled_video1_path}")

    summary_table = summary_tables.get("VideoData1")
    if summary_table is not None and not summary_table.empty:
        summary_export = summary_table.copy()
        summary_export["aeon_time"] = summary_export["saccade_peak_time"].apply(api.aeon)
        summary_export.to_csv(summary_output_path1, index=False)

if "VideoData2" in globals():
    video_dir2 = qc_debug_dir / "Video_Sleap_Data2"
    if debug:
        video_dir2.mkdir(parents=True, exist_ok=True)
    summary_output_path2 = downsampled_output_dir / "VideoData2_saccade_summary.csv"

    drop_for_export = [col for col in SACCADE_EXPORT_DROP_COLUMNS if col in VideoData2.columns]
    VideoData2_export = VideoData2.drop(columns=drop_for_export)

    VideoData2_export_renamed = (
        VideoData2_export.rename(columns={"Ellipse.Diameter.Filt": "Pupil.Diameter"})
        if "Ellipse.Diameter.Filt" in VideoData2_export.columns
        else VideoData2_export
    )

    if debug:
        VideoData2_full_indexed = set_aeon_index(VideoData2_export_renamed)
        append_aeon_time_column(VideoData2_full_indexed).to_csv(
            video_dir2 / "Video_Sleap_Data2_1904-01-01T00-00-00.csv", index=False
        )

    drop_for_resample = [col for col in RESAMPLED_DROP_COLUMNS if col in VideoData2_export.columns]
    VideoData2_resampled_input = VideoData2_export.drop(columns=drop_for_resample)
    VideoData2_resampled = resample_video_dataframe(VideoData2_resampled_input, "eye2")
    if "Ellipse.Diameter.Filt" in VideoData2_resampled.columns:
        VideoData2_resampled = VideoData2_resampled.rename(columns={"Ellipse.Diameter.Filt": "Pupil.Diameter"})
    VideoData2_resampled_indexed = set_aeon_index(VideoData2_resampled)
    if debug:
        append_aeon_time_column(VideoData2_resampled_indexed).to_csv(
            video_dir2 / "Video_Sleap_Data2_1904-01-01T00-00-00_resampled.csv", index=False
        )
    downsampled_video2_path = downsampled_output_dir / "VideoData2_resampled.parquet"
    VideoData2_resampled_indexed.to_parquet(
        downsampled_video2_path, engine="pyarrow", compression="snappy"
    )
    print(f"✅ Saved VideoData2 resampled parquet to {downsampled_video2_path}")

    summary_table = summary_tables.get("VideoData2")
    if summary_table is not None and not summary_table.empty:
        summary_export = summary_table.copy()
        summary_export["aeon_time"] = summary_export["saccade_peak_time"].apply(api.aeon)
        summary_export.to_csv(summary_output_path2, index=False)


# +
# VISUALIZE ALL SACCADES - SIDE BY SIDE
# -------------------------------------------------------------------------------
# Plot all upward and downward saccades aligned by time with position and velocity traces
# Using extracted visualization function from sleap.visualization

if plot_saccade_detection_QC:
    plot_all_saccades_overlay(
        saccade_results=saccade_results,
        video_labels=VIDEO_LABELS,
        video1_eye=video1_eye,
        video2_eye=video2_eye,
        pre_saccade_window_time=pre_saccade_window_time,
        post_saccade_window_time=post_saccade_window_time,
        debug=debug,
        show_plot=True,
    )


# +
# SACCADE AMPLITUDE QC VISUALIZATION
# -------------------------------------------------------------------------------
# 1. Distribution of saccade amplitudes
# 2. Correlation between saccade amplitude and duration
# 3. Peri-saccade segments colored by amplitude (outlier detection)
# Using extracted visualization function from sleap.visualization

if plot_saccade_detection_QC:
    plot_saccade_amplitude_qc(
        saccade_results=saccade_results,
        video_labels=VIDEO_LABELS,
        video1_eye=video1_eye,
        video2_eye=video2_eye,
        debug=debug,
        show_plot=True,
    )


# +
# DEBUG: BASELINING DIAGNOSTICS
# -------------------------------------------------------------------------------
# Plot distribution of baseline window values before and after baselining
# This helps diagnose why some segments might not be baselined correctly

if debug:
    for video_key, res in saccade_results.items():
        dir_map = get_direction_map_for_video(video_key)
        label_up = dir_map["upward"]
        label_down = dir_map["downward"]

        peri_saccades = res["peri_saccades"]

        if len(peri_saccades) == 0:
            print(
                f"\n⚠️  No saccades found for {get_eye_label(video_key)}, skipping baselining diagnostics"
            )
            continue

        # Extract baseline window statistics for each segment
        baseline_values = []  # What was subtracted
        # Mean position in baseline window BEFORE baselining
        baseline_window_means_before = []
        # Mean position in baseline window AFTER baselining
        baseline_window_means_after = []
        baseline_window_counts = []  # Number of points in baseline window
        segment_directions = []
        segment_ids = []

        # Get baseline window parameters from the function call (use defaults
        # if not available)
        baseline_window_start_time = (
            -0.1
            if "baseline_window_start_time" not in globals()
            else baseline_window_start_time
        )
        baseline_window_end_time = (
            -0.02
            if "baseline_window_end_time" not in globals()
            else baseline_window_end_time
        )

        for seg in peri_saccades:
            seg_id = (
                seg["saccade_id"].iloc[0]
                if "saccade_id" in seg.columns
                else len(baseline_values)
            )
            direction = (
                seg["saccade_direction"].iloc[0]
                if "saccade_direction" in seg.columns
                else "unknown"
            )

            # Get baseline value that was used
            if "baseline_value" in seg.columns:
                baseline_val = seg["baseline_value"].iloc[0]
            else:
                baseline_val = np.nan

            # Find baseline window points (before threshold crossing)
            baseline_mask = (
                (seg["Time_rel_threshold"] >= baseline_window_start_time)
                & (seg["Time_rel_threshold"] <= baseline_window_end_time)
                & (seg["Time_rel_threshold"] < 0)  # Pre-threshold only
            )

            # Get original position values (reconstruct if needed)
            if "X_raw" in seg.columns:
                original_pos_col = "X_raw"
            elif "X_smooth" in seg.columns:
                original_pos_col = "X_smooth"
            else:
                original_pos_col = None

            # Calculate mean in baseline window BEFORE baselining
            if original_pos_col is not None:
                baseline_window_original = seg.loc[
                    baseline_mask, original_pos_col
                ].dropna()
                if len(baseline_window_original) > 0:
                    mean_before = baseline_window_original.mean()
                else:
                    mean_before = np.nan
            else:
                # Reconstruct: original = baselined + baseline_value
                if "X_smooth_baselined" in seg.columns and not pd.isna(baseline_val):
                    baseline_window_baselined = seg.loc[
                        baseline_mask, "X_smooth_baselined"
                    ].dropna()
                    if len(baseline_window_baselined) > 0:
                        mean_before = baseline_window_baselined.mean() + baseline_val
                    else:
                        mean_before = np.nan
                else:
                    mean_before = np.nan

            # Calculate mean in baseline window AFTER baselining
            if "X_smooth_baselined" in seg.columns:
                baseline_window_baselined = seg.loc[
                    baseline_mask, "X_smooth_baselined"
                ].dropna()
                if len(baseline_window_baselined) > 0:
                    mean_after = baseline_window_baselined.mean()
                    n_points = len(baseline_window_baselined)
                else:
                    mean_after = np.nan
                    n_points = 0
            else:
                mean_after = np.nan
                n_points = 0

            baseline_values.append(baseline_val)
            baseline_window_means_before.append(mean_before)
            baseline_window_means_after.append(mean_after)
            baseline_window_counts.append(n_points)
            segment_directions.append(direction)
            segment_ids.append(seg_id)

        # Convert to arrays for easier manipulation
        baseline_values = np.array(baseline_values)
        baseline_window_means_before = np.array(baseline_window_means_before)
        baseline_window_means_after = np.array(baseline_window_means_after)
        baseline_window_counts = np.array(baseline_window_counts)
        segment_directions = np.array(segment_directions)

        # Create diagnostic figure
        fig_baseline = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Baseline Values (What Was Subtracted)",
                "Baseline Window Mean - BEFORE Baselining",
                "Baseline Window Mean - AFTER Baselining",
                "Baseline Window Point Counts",
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        # Plot 1: Baseline values distribution
        for direction in ["upward", "downward"]:
            mask = segment_directions == direction
            if mask.sum() > 0:
                label = label_up if direction == "upward" else label_down
                color = "green" if direction == "upward" else "purple"
                fig_baseline.add_trace(
                    go.Histogram(
                        x=baseline_values[mask],
                        nbinsx=50,
                        name=f"{label}",
                        marker_color=color,
                        opacity=0.6,
                    ),
                    row=1,
                    col=1,
                )

        # Plot 2: Baseline window mean BEFORE baselining
        for direction in ["upward", "downward"]:
            mask = segment_directions == direction
            if mask.sum() > 0:
                label = label_up if direction == "upward" else label_down
                color = "green" if direction == "upward" else "purple"
                valid_mask = mask & ~np.isnan(baseline_window_means_before)
                if valid_mask.sum() > 0:
                    fig_baseline.add_trace(
                        go.Histogram(
                            x=baseline_window_means_before[valid_mask],
                            nbinsx=50,
                            name=f"{label}",
                            marker_color=color,
                            opacity=0.6,
                            showlegend=False,
                        ),
                        row=1,
                        col=2,
                    )

        # Plot 3: Baseline window mean AFTER baselining (should be ~0)
        for direction in ["upward", "downward"]:
            mask = segment_directions == direction
            if mask.sum() > 0:
                label = label_up if direction == "upward" else label_down
                color = "green" if direction == "upward" else "purple"
                valid_mask = mask & ~np.isnan(baseline_window_means_after)
                if valid_mask.sum() > 0:
                    fig_baseline.add_trace(
                        go.Histogram(
                            x=baseline_window_means_after[valid_mask],
                            nbinsx=50,
                            name=f"{label}",
                            marker_color=color,
                            opacity=0.6,
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

        # Plot 4: Baseline window point counts
        for direction in ["upward", "downward"]:
            mask = segment_directions == direction
            if mask.sum() > 0:
                label = label_up if direction == "upward" else label_down
                color = "green" if direction == "upward" else "purple"
                fig_baseline.add_trace(
                    go.Histogram(
                        x=baseline_window_counts[mask],
                        nbinsx=20,
                        name=f"{label}",
                        marker_color=color,
                        opacity=0.6,
                        showlegend=False,
                    ),
                    row=2,
                    col=2,
                )

        # Add vertical line at 0 for "AFTER baselining" plot
        fig_baseline.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            line_width=2,
            opacity=0.7,
            annotation_text="Expected: 0",
            row=2,
            col=1,
        )

        # Update layout
        fig_baseline.update_layout(
            title_text=f"Baselining Diagnostics: {get_eye_label(video_key)}<br><sub>After baselining, baseline window mean should be ~0</sub>",
            height=800,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
        )

        # Update axes labels
        fig_baseline.update_xaxes(title_text="Baseline Value (px)", row=1, col=1)
        fig_baseline.update_xaxes(
            title_text="Mean Position in Baseline Window (px)", row=1, col=2
        )
        fig_baseline.update_xaxes(
            title_text="Mean Position in Baseline Window (px)", row=2, col=1
        )
        fig_baseline.update_xaxes(title_text="Number of Points", row=2, col=2)
        fig_baseline.update_yaxes(title_text="Count", row=1, col=1)
        fig_baseline.update_yaxes(title_text="Count", row=1, col=2)
        fig_baseline.update_yaxes(title_text="Count", row=2, col=1)
        fig_baseline.update_yaxes(title_text="Count", row=2, col=2)

        fig_baseline.show()

        # Print statistics
        print(f"\n{'=' * 80}")
        print(f"BASELINING DIAGNOSTICS: {get_eye_label(video_key)}")
        print(f"{'=' * 80}")

        for direction in ["upward", "downward"]:
            mask = segment_directions == direction
            label = label_up if direction == "upward" else label_down

            if mask.sum() > 0:
                print(f"\n{label} saccades (n={mask.sum()}):")

                # Baseline values
                valid_baseline = baseline_values[mask & ~np.isnan(baseline_values)]
                if len(valid_baseline) > 0:
                    print("  Baseline values (subtracted):")
                    print(f"    Mean: {valid_baseline.mean():.2f} px")
                    print(f"    Std: {valid_baseline.std():.2f} px")
                    print(
                        f"    Range: [{valid_baseline.min():.2f}, {valid_baseline.max():.2f}] px"
                    )
                    print(f"    NaN count: {np.isnan(baseline_values[mask]).sum()}")

                # Baseline window means BEFORE
                valid_before = baseline_window_means_before[
                    mask & ~np.isnan(baseline_window_means_before)
                ]
                if len(valid_before) > 0:
                    print("\n  Baseline window mean BEFORE baselining:")
                    print(f"    Mean: {valid_before.mean():.2f} px")
                    print(f"    Std: {valid_before.std():.2f} px")
                    print(
                        f"    Range: [{valid_before.min():.2f}, {valid_before.max():.2f}] px"
                    )
                    print(
                        f"    NaN count: {np.isnan(baseline_window_means_before[mask]).sum()}"
                    )

                # Baseline window means AFTER (should be ~0)
                valid_after = baseline_window_means_after[
                    mask & ~np.isnan(baseline_window_means_after)
                ]
                if len(valid_after) > 0:
                    print("\n  Baseline window mean AFTER baselining (should be ~0):")
                    print(f"    Mean: {valid_after.mean():.2f} px")
                    print(f"    Std: {valid_after.std():.2f} px")
                    print(
                        f"    Range: [{valid_after.min():.2f}, {valid_after.max():.2f}] px"
                    )
                    print(
                        f"    NaN count: {np.isnan(baseline_window_means_after[mask]).sum()}"
                    )

                    # Count segments that are NOT properly baselined (mean >
                    # 1px away from 0)
                    not_baselined = np.abs(valid_after) > 1.0
                    if not_baselined.sum() > 0:
                        print(
                            f"\n  ⚠️  WARNING: {not_baselined.sum()} segments NOT properly baselined (|mean| > 1px)"
                        )
                        print(
                            f"    Mean values for non-baselined segments: {valid_after[not_baselined]}"
                        )

                    # Show distribution of baseline window means (should all be
                    # ~0)
                    print(
                        "\n  Distribution of baseline window means AFTER baselining:"
                    )
                    print(
                        f"    Segments with |mean| < 0.1 px: {(np.abs(valid_after) < 0.1).sum()}"
                    )
                    print(
                        f"    Segments with 0.1 <= |mean| < 0.5 px: {((np.abs(valid_after) >= 0.1) & (np.abs(valid_after) < 0.5)).sum()}"
                    )
                    print(
                        f"    Segments with 0.5 <= |mean| < 1.0 px: {((np.abs(valid_after) >= 0.5) & (np.abs(valid_after) < 1.0)).sum()}"
                    )
                    print(
                        f"    Segments with |mean| >= 1.0 px: {(np.abs(valid_after) >= 1.0).sum()}"
                    )

                    # Show worst offenders
                    worst_indices = np.argsort(np.abs(valid_after))[
                        -10:
                    ]  # Top 10 worst
                    if len(worst_indices) > 0:
                        print(
                            "\n  Top 10 segments with largest |baseline window mean|:"
                        )
                        for i, idx in enumerate(
                            worst_indices[::-1]
                        ):  # Reverse to show worst first
                            seg_idx = np.where(mask)[0][idx]
                            seg_id = segment_ids[seg_idx]
                            mean_val = valid_after[idx]
                            baseline_val = baseline_values[mask][idx]
                            print(
                                f"    Segment {seg_id}: baseline_window_mean={mean_val:.3f} px, baseline_value={baseline_val:.3f} px"
                            )

                    # CRITICAL CHECK: Verify that baseline_value actually equals baseline_window_mean_before
                    # This checks if baselining logic is correct
                    valid_before_check = baseline_window_means_before[
                        mask & ~np.isnan(baseline_window_means_before)
                    ]
                    if len(valid_before_check) > 0 and len(valid_baseline) > 0:
                        # Check if baseline_value matches
                        # baseline_window_mean_before (should be identical)
                        # Allow small floating point differences
                        mismatches = np.abs(valid_baseline - valid_before_check) > 0.01
                        if mismatches.sum() > 0:
                            print(
                                f"\n  ⚠️  CRITICAL: {mismatches.sum()} segments where baseline_value != baseline_window_mean_before"
                            )
                            print("    This indicates a bug in baselining logic!")
                            for i in np.where(mismatches)[0][:5]:  # Show first 5
                                print(
                                    f"      Segment {i}: baseline_value={valid_baseline[i]:.3f}, baseline_window_mean_before={valid_before_check[i]:.3f}"
                                )
                        else:
                            print(
                                "\n  ✅ Baselining logic check: baseline_value matches baseline_window_mean_before for all segments"
                            )

                # Baseline window point counts
                valid_counts = baseline_window_counts[
                    mask & (baseline_window_counts > 0)
                ]
                if len(valid_counts) > 0:
                    print("\n  Baseline window point counts:")
                    print(f"    Mean: {valid_counts.mean():.1f} points")
                    print(
                        f"    Range: [{valid_counts.min()}, {valid_counts.max()}] points"
                    )
                    print(
                        f"    Segments with 0 points in baseline window: {(baseline_window_counts[mask] == 0).sum()}"
                    )

        print(f"\n{'=' * 80}\n")

        # Additional diagnostic: Check if baselining is actually applied to segments
        # Compare original vs baselined values at the start of segments (before
        # threshold)
        print(f"\n{'=' * 80}")
        print("ADDITIONAL BASELINING CHECK: Comparing segment start positions")
        print(f"{'=' * 80}")

        for direction in ["upward", "downward"]:
            mask = segment_directions == direction
            label = label_up if direction == "upward" else label_down

            if mask.sum() > 0:
                print(f"\n{label} saccades:")

                # Check first few points of each segment (should be ~0 after
                # baselining)
                segment_start_means_original = []
                segment_start_means_baselined = []

                for i, seg in enumerate(peri_saccades):
                    if segment_directions[i] != direction:
                        continue

                    # Get first 5 points before threshold (if available)
                    pre_threshold_mask = seg["Time_rel_threshold"] < 0
                    pre_threshold_seg = seg.loc[pre_threshold_mask].head(5)

                    if len(pre_threshold_seg) > 0:
                        # Get original position
                        if "X_raw" in seg.columns:
                            orig_col = "X_raw"
                        elif "X_smooth" in seg.columns:
                            orig_col = "X_smooth"
                        else:
                            orig_col = None

                        if orig_col is not None:
                            orig_mean = pre_threshold_seg[orig_col].mean()
                            segment_start_means_original.append(orig_mean)

                        # Get baselined position
                        if "X_smooth_baselined" in seg.columns:
                            baselined_mean = pre_threshold_seg[
                                "X_smooth_baselined"
                            ].mean()
                            segment_start_means_baselined.append(baselined_mean)

                if len(segment_start_means_baselined) > 0:
                    segment_start_means_baselined = np.array(
                        segment_start_means_baselined
                    )
                    print(
                        "  Mean position at segment start (first 5 pre-threshold points) AFTER baselining:"
                    )
                    print(f"    Mean: {segment_start_means_baselined.mean():.3f} px")
                    print(f"    Std: {segment_start_means_baselined.std():.3f} px")
                    print(
                        f"    Range: [{segment_start_means_baselined.min():.3f}, {segment_start_means_baselined.max():.3f}] px"
                    )
                    print(
                        f"    Segments with |mean| > 1 px: {(np.abs(segment_start_means_baselined) > 1.0).sum()}"
                    )
                    print(
                        f"    Segments with |mean| > 5 px: {(np.abs(segment_start_means_baselined) > 5.0).sum()}"
                    )

                    # Show worst offenders
                    worst_indices = np.argsort(np.abs(segment_start_means_baselined))[
                        -10:
                    ]
                    if len(worst_indices) > 0:
                        print(
                            "\n  Top 10 segments with largest |start position mean| AFTER baselining:"
                        )
                        for idx in worst_indices[::-1]:
                            print(
                                f"    Segment {idx}: start_mean={segment_start_means_baselined[idx]:.3f} px"
                            )

        print(f"\n{'=' * 80}\n")
# -


# VISUALIZE DETECTED SACCADES (Adaptive Method)
# -------------------------------------------------------------------------------
# Create overlay plot showing detected saccades with duration lines and
# peak arrows
if plot_saccade_detection_QC:
    for video_key, res in saccade_results.items():
        dir_map = get_direction_map_for_video(video_key)
        label_up = dir_map["upward"]
        label_down = dir_map["downward"]

        upward_saccades_df = res["upward_saccades_df"]
        downward_saccades_df = res["downward_saccades_df"]
        peri_saccades = res["peri_saccades"]
        upward_segments = res["upward_segments"]
        downward_segments = res["downward_segments"]
        # Any other variables you need...

        # Optional: Find 5-minute window with highest saccade density
        plot_5min_window = (
            True  # Set to True to plot only highest density 5-minute window
        )
        window_duration = 300  # 5 minutes in seconds

        # Initialize variables for windowing
        best_window_start = None
        best_window_end = None
        best_window_count = 0

        if plot_5min_window and (
            len(upward_saccades_df) > 0 or len(downward_saccades_df) > 0
        ):
            # Combine all saccades and find time window with highest density
            all_saccade_times = []
            if len(upward_saccades_df) > 0:
                all_saccade_times.extend(upward_saccades_df["time"].values)
            if len(downward_saccades_df) > 0:
                all_saccade_times.extend(downward_saccades_df["time"].values)

            if len(all_saccade_times) > 0:
                all_saccade_times = np.array(all_saccade_times)
                time_min = all_saccade_times.min()
                time_max = all_saccade_times.max()

                # Slide window and count saccades in each window
                best_window_start = time_min
                best_window_count = 0
                step_size = 10  # Check every 10 seconds

                for window_start in np.arange(
                    time_min, time_max - window_duration + step_size, step_size
                ):
                    window_end = window_start + window_duration
                    count = np.sum(
                        (all_saccade_times >= window_start)
                        & (all_saccade_times <= window_end)
                    )
                    if count > best_window_count:
                        best_window_count = count
                        best_window_start = window_start

                best_window_end = best_window_start + window_duration

                # Filter data to this window
                df_window = res["df"][
                    (res["df"]["Seconds"] >= best_window_start)
                    & (res["df"]["Seconds"] <= best_window_end)
                ].copy()
                upward_saccades_df_window = upward_saccades_df[
                    (upward_saccades_df["time"] >= best_window_start)
                    & (upward_saccades_df["time"] <= best_window_end)
                ].copy()
                downward_saccades_df_window = downward_saccades_df[
                    (downward_saccades_df["time"] >= best_window_start)
                    & (downward_saccades_df["time"] <= best_window_end)
                ].copy()

                if debug:
                    print(
                        f"\n📊 Highest saccade density window: {best_window_start:.1f}s to {best_window_end:.1f}s"
                    )
                    print(
                        f"   ({best_window_count} saccades in {window_duration / 60:.1f} minutes)"
                    )
                    print(
                        f"   Density: {best_window_count / (window_duration / 60):.1f} saccades/min"
                    )
            else:
                plot_5min_window = False
        else:
            plot_5min_window = False

        # Use windowed data if requested, otherwise use full data
        if plot_5min_window:
            df_plot = df_window
            upward_saccades_df_plot = upward_saccades_df_window
            downward_saccades_df_plot = downward_saccades_df_window
            time_range_text = f" (5-min window: {best_window_start:.1f}-{best_window_end:.1f}s, {best_window_count} saccades)"
        else:
            df_plot = res["df"]
            upward_saccades_df_plot = upward_saccades_df
            downward_saccades_df_plot = downward_saccades_df
            time_range_text = ""

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                "X Position (px)",
                "Velocity (px/s) with Detected Saccades",
            ),
        )

        # Add smoothed X position to the first subplot
        fig.add_trace(
            go.Scatter(
                x=df_plot["Seconds"],
                y=df_plot["X_smooth"],
                mode="lines",
                name="Smoothed X",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Add smoothed velocity to the second subplot
        fig.add_trace(
            go.Scatter(
                x=df_plot["Seconds"],
                y=df_plot["vel_x_smooth"],
                mode="lines",
                name="Smoothed Velocity",
                line=dict(color="red", width=2),
            ),
            row=2,
            col=1,
        )

        # Add adaptive threshold lines for reference
        fig.add_hline(
            y=res["vel_thresh"],
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text=f"Adaptive threshold (±{res['vel_thresh']:.0f} px/s)",
            row=2,
            col=1,
        )

        fig.add_hline(
            y=-res["vel_thresh"],
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            row=2,
            col=1,
        )

        # Calculate position y-axis range for vertical lines
        pos_max = df_plot["X_smooth"].max()
        pos_min = df_plot["X_smooth"].min()
        pos_range = pos_max - pos_min
        # Add small padding to ensure lines span full visible range
        pos_padding = pos_range * 0.05
        pos_y_min = pos_min - pos_padding
        pos_y_max = pos_max + pos_padding

        # Plot upward saccades (TN) as vertical lines on position trace
        if len(upward_saccades_df_plot) > 0:
            for idx, row in upward_saccades_df_plot.iterrows():
                start_time = row["start_time"]
                end_time = row["end_time"]

                # Use rectangle with thin border to show duration span more efficiently
                # Opacity must be set at shape level or via rgba color, not in
                # line dict
                fig.add_shape(
                    type="rect",
                    x0=start_time,
                    y0=pos_y_min,
                    x1=end_time,
                    y1=pos_y_max,
                    # Light green fill with opacity
                    fillcolor="rgba(0,128,0,0.1)",
                    # Green border with opacity via rgba
                    line=dict(color="rgba(0,128,0,0.3)", width=2),
                    row=1,
                    col=1,
                )

            # Add legend entry for upward saccades
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=f"{label_up} Saccades",
                    line=dict(
                        color="rgba(0,128,0,0.3)", width=3
                    ),  # Green with opacity via rgba
                ),
                row=1,
                col=1,
            )

        # Plot downward saccades (NT) as vertical lines on position trace
        if len(downward_saccades_df_plot) > 0:
            for idx, row in downward_saccades_df_plot.iterrows():
                start_time = row["start_time"]
                end_time = row["end_time"]

                # Use rectangle with thin border to show duration span more efficiently
                # Opacity must be set at shape level or via rgba color, not in
                # line dict
                fig.add_shape(
                    type="rect",
                    x0=start_time,
                    y0=pos_y_min,
                    x1=end_time,
                    y1=pos_y_max,
                    # Light purple fill with opacity
                    fillcolor="rgba(128,0,128,0.1)",
                    # Purple border with opacity via rgba
                    line=dict(color="rgba(128,0,128,0.3)", width=2),
                    row=1,
                    col=1,
                )

            # Add legend entry for downward saccades
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=f"{label_down} Saccades",
                    line=dict(
                        color="rgba(128,0,128,0.3)", width=3
                    ),  # Purple with opacity via rgba
                ),
                row=1,
                col=1,
            )

        # Update layout
        fig.update_layout(
            title=f"Detected Saccades ({get_eye_label(video_key)}): Vertical Lines on Position Trace (QC Visualization){time_range_text}",
            height=600,
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
        )

        # Update axes
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="X Position (px)", row=1, col=1)
        fig.update_yaxes(title_text="Velocity (px/s)", row=2, col=1)

        fig.show()


# ADAPTIVE THRESHOLD DIAGNOSTIC PLOTS (only if debug=True)
# -------------------------------------------------------------------------------
# Plot distributions of classification features to help determine
# meaningful adaptive thresholds
if debug and len(saccade_results) > 0:
    print("\n📊 Generating adaptive threshold diagnostic plots...")

    for video_key, res in saccade_results.items():
        all_saccades_df = res.get("all_saccades_df", pd.DataFrame())

        if len(all_saccades_df) == 0:
            print(
                f"⚠️  No saccades found for {get_eye_label(video_key)}, skipping diagnostic plots"
            )
            continue

        # Filter out NaN values for plotting
        pre_vel = all_saccades_df["pre_saccade_mean_velocity"].dropna()
        pre_drift = all_saccades_df["pre_saccade_position_drift"].dropna()
        post_var = all_saccades_df["post_saccade_position_variance"].dropna()
        post_change = all_saccades_df["post_saccade_position_change"].dropna()
        amplitude = all_saccades_df["amplitude"].dropna()

        # Calculate post_change / amplitude ratio (for percentage threshold visualization)
        # Align by index to ensure matching
        aligned_indices = post_change.index.intersection(amplitude.index)
        post_change_aligned = post_change.loc[aligned_indices]
        amplitude_aligned = amplitude.loc[aligned_indices]
        # Convert to percentage
        post_change_ratio = (post_change_aligned / amplitude_aligned) * 100

        # Calculate current thresholds for visualization
        if use_adaptive_thresholds:
            # Calculate adaptive thresholds from current data
            if len(pre_vel) >= 3:
                current_pre_vel_threshold = np.percentile(
                    pre_vel, adaptive_percentile_pre_velocity
                )
            else:
                current_pre_vel_threshold = pre_saccade_velocity_threshold

            if len(pre_drift) >= 3:
                current_pre_drift_threshold = np.percentile(
                    pre_drift, adaptive_percentile_pre_drift
                )
            else:
                current_pre_drift_threshold = pre_saccade_drift_threshold

            if len(post_var) >= 3:
                current_post_var_threshold = np.percentile(
                    post_var, adaptive_percentile_post_variance
                )
            else:
                current_post_var_threshold = post_saccade_variance_threshold
        else:
            # Use fixed thresholds
            current_pre_vel_threshold = pre_saccade_velocity_threshold
            current_pre_drift_threshold = pre_saccade_drift_threshold
            current_post_var_threshold = post_saccade_variance_threshold

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Adaptive Threshold Diagnostic Plots: {get_eye_label(video_key)}\n"
            f"(n={len(all_saccades_df)} saccades)",
            fontsize=14,
            fontweight="bold",
        )

        # Plot 1: Pre-saccade mean velocity
        ax = axes[0, 0]
        if len(pre_vel) > 0:
            ax.hist(pre_vel, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
            ax.axvline(
                current_pre_vel_threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold: {current_pre_vel_threshold:.2f} px/s",
            )
            if use_adaptive_thresholds:
                ax.axvline(
                    np.percentile(pre_vel, 50),
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    label=f"Median: {np.percentile(pre_vel, 50):.2f} px/s",
                )
                ax.axvline(
                    np.percentile(pre_vel, 75),
                    color="orange",
                    linestyle=":",
                    linewidth=1,
                    label=f"75th: {np.percentile(pre_vel, 75):.2f} px/s",
                )
            ax.set_xlabel("Pre-saccade Mean Velocity (px/s)")
            ax.set_ylabel("Count")
            ax.set_title(
                f"Pre-saccade Velocity Distribution\n"
                f"{'Adaptive' if use_adaptive_thresholds else 'Fixed'} threshold at "
                f"{adaptive_percentile_pre_velocity if use_adaptive_thresholds else 'fixed'}th percentile"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 2: Pre-saccade position drift
        ax = axes[0, 1]
        if len(pre_drift) > 0:
            ax.hist(
                pre_drift, bins=50, alpha=0.7, color="lightgreen", edgecolor="black"
            )
            ax.axvline(
                current_pre_drift_threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold: {current_pre_drift_threshold:.2f} px",
            )
            if use_adaptive_thresholds:
                ax.axvline(
                    np.percentile(pre_drift, 50),
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    label=f"Median: {np.percentile(pre_drift, 50):.2f} px",
                )
                ax.axvline(
                    np.percentile(pre_drift, 75),
                    color="orange",
                    linestyle=":",
                    linewidth=1,
                    label=f"75th: {np.percentile(pre_drift, 75):.2f} px",
                )
            ax.set_xlabel("Pre-saccade Position Drift (px)")
            ax.set_ylabel("Count")
            ax.set_title(
                f"Pre-saccade Drift Distribution\n"
                f"{'Adaptive' if use_adaptive_thresholds else 'Fixed'} threshold at "
                f"{adaptive_percentile_pre_drift if use_adaptive_thresholds else 'fixed'}th percentile"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 3: Post-saccade position variance
        ax = axes[1, 0]
        if len(post_var) > 0:
            ax.hist(post_var, bins=50, alpha=0.7, color="plum", edgecolor="black")
            ax.axvline(
                current_post_var_threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold: {current_post_var_threshold:.2f} px²",
            )
            if use_adaptive_thresholds:
                ax.axvline(
                    np.percentile(post_var, 25),
                    color="orange",
                    linestyle=":",
                    linewidth=1,
                    label=f"25th: {np.percentile(post_var, 25):.2f} px²",
                )
                ax.axvline(
                    np.percentile(post_var, 50),
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    label=f"Median: {np.percentile(post_var, 50):.2f} px²",
                )
            ax.set_xlabel("Post-saccade Position Variance (px²)")
            ax.set_ylabel("Count")
            ax.set_title(
                f"Post-saccade Variance Distribution\n"
                f"{'Adaptive' if use_adaptive_thresholds else 'Fixed'} threshold at "
                f"{adaptive_percentile_post_variance if use_adaptive_thresholds else 'fixed'}th percentile"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 4: Post-saccade position change (as percentage of amplitude)
        ax = axes[1, 1]
        if len(post_change_ratio) > 0:
            ax.hist(
                post_change_ratio, bins=50, alpha=0.7, color="salmon", edgecolor="black"
            )
            ax.axvline(
                post_saccade_position_change_threshold_percent,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Threshold: {post_saccade_position_change_threshold_percent:.1f}%",
            )
            ax.axvline(
                np.percentile(post_change_ratio, 50),
                color="gray",
                linestyle=":",
                linewidth=1,
                label=f"Median: {np.percentile(post_change_ratio, 50):.1f}%",
            )
            ax.axvline(
                np.percentile(post_change_ratio, 75),
                color="orange",
                linestyle=":",
                linewidth=1,
                label=f"75th: {np.percentile(post_change_ratio, 75):.1f}%",
            )
            ax.set_xlabel("Post-saccade Position Change / Amplitude (%)")
            ax.set_ylabel("Count")
            ax.set_title(
                f"Post-saccade Position Change Ratio\n"
                f"Fixed threshold: {post_saccade_position_change_threshold_percent:.1f}% of amplitude"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print(f"\n📈 Summary Statistics for {get_eye_label(video_key)}:")
        if len(pre_vel) > 0:
            print(
                f"  Pre-saccade velocity: mean={pre_vel.mean():.2f}, median={pre_vel.median():.2f}, "
                f"std={pre_vel.std():.2f} px/s"
            )
        if len(pre_drift) > 0:
            print(
                f"  Pre-saccade drift: mean={pre_drift.mean():.2f}, median={pre_drift.median():.2f}, "
                f"std={pre_drift.std():.2f} px"
            )
        if len(post_var) > 0:
            print(
                f"  Post-saccade variance: mean={post_var.mean():.2f}, median={post_var.median():.2f}, "
                f"std={post_var.std():.2f} px²"
            )
        if len(post_change_ratio) > 0:
            print(
                f"  Post-saccade change ratio: mean={post_change_ratio.mean():.1f}%, "
                f"median={post_change_ratio.median():.1f}%, std={post_change_ratio.std():.1f}%"
            )
        print()

# +
# VISUALIZE AND ANALYZE SACCADE CLASSIFICATION (Orienting vs Compensatory)
# -------------------------------------------------------------------------------
# Create validation plots and statistical comparisons for saccade
# classification

if plot_saccade_detection_QC:
    for video_key, res in saccade_results.items():
        dir_map = get_direction_map_for_video(video_key)
        label_up = dir_map["upward"]
        label_down = dir_map["downward"]

        all_saccades_df = res.get("all_saccades_df", pd.DataFrame())

        if len(all_saccades_df) == 0:
            print(f"\n⚠️ No saccades found for {get_eye_label(video_key)}")
            continue

        # Check if classification was performed

        if "saccade_type" not in all_saccades_df.columns:
            print(f"\n⚠️ Classification not performed for {get_eye_label(video_key)}")
            continue

        orienting_saccades = all_saccades_df[
            all_saccades_df["saccade_type"] == "orienting"
        ]

        compensatory_saccades = all_saccades_df[
            all_saccades_df["saccade_type"] == "compensatory"
        ]

        print(f"\n{'=' * 80}")

        print(f"CLASSIFICATION ANALYSIS: {get_eye_label(video_key)}")

        print(f"{'=' * 80}")

        # Statistical comparisons

        from scipy import stats

        print("\n📊 Statistical Comparisons:")

        print(f"  Orienting saccades: {len(orienting_saccades)}")

        print(f"  Compensatory saccades: {len(compensatory_saccades)}")

        if len(orienting_saccades) > 0 and len(compensatory_saccades) > 0:
            # Amplitude comparison

            orienting_amps = orienting_saccades["amplitude"].values

            compensatory_amps = compensatory_saccades["amplitude"].values

            amp_stat, amp_p = stats.mannwhitneyu(
                orienting_amps, compensatory_amps, alternative="two-sided"
            )

            print("\n  Amplitude (px):")

            print(
                f"    Orienting: {orienting_amps.mean():.2f} ± {orienting_amps.std():.2f} (median: {np.median(orienting_amps):.2f})"
            )

            print(
                f"    Compensatory: {compensatory_amps.mean():.2f} ± {compensatory_amps.std():.2f} (median: {np.median(compensatory_amps):.2f})"
            )

            print(f"    Mann-Whitney U test: U={amp_stat:.1f}, p={amp_p:.4f}")

            # Duration comparison

            orienting_durs = orienting_saccades["duration"].values

            compensatory_durs = compensatory_saccades["duration"].values

            dur_stat, dur_p = stats.mannwhitneyu(
                orienting_durs, compensatory_durs, alternative="two-sided"
            )

            print("\n  Duration (s):")

            print(
                f"    Orienting: {orienting_durs.mean():.3f} ± {orienting_durs.std():.3f} (median: {np.median(orienting_durs):.3f})"
            )

            print(
                f"    Compensatory: {compensatory_durs.mean():.3f} ± {compensatory_durs.std():.3f} (median: {np.median(compensatory_durs):.3f})"
            )

            print(f"    Mann-Whitney U test: U={dur_stat:.1f}, p={dur_p:.4f}")

            # Pre-saccade velocity comparison

            orienting_pre_vel = orienting_saccades["pre_saccade_mean_velocity"].values

            compensatory_pre_vel = compensatory_saccades[
                "pre_saccade_mean_velocity"
            ].values

            pre_vel_stat, pre_vel_p = stats.mannwhitneyu(
                orienting_pre_vel, compensatory_pre_vel, alternative="two-sided"
            )

            print("\n  Pre-saccade velocity (px/s):")

            print(
                f"    Orienting: {orienting_pre_vel.mean():.2f} ± {orienting_pre_vel.std():.2f} (median: {np.median(orienting_pre_vel):.2f})"
            )

            print(
                f"    Compensatory: {compensatory_pre_vel.mean():.2f} ± {compensatory_pre_vel.std():.2f} (median: {np.median(compensatory_pre_vel):.2f})"
            )

            print(f"    Mann-Whitney U test: U={pre_vel_stat:.1f}, p={pre_vel_p:.4f}")

            # Pre-saccade drift comparison

            orienting_pre_drift = orienting_saccades[
                "pre_saccade_position_drift"
            ].values

            compensatory_pre_drift = compensatory_saccades[
                "pre_saccade_position_drift"
            ].values

            pre_drift_stat, pre_drift_p = stats.mannwhitneyu(
                orienting_pre_drift, compensatory_pre_drift, alternative="two-sided"
            )

            print("\n  Pre-saccade position drift (px):")

            print(
                f"    Orienting: {orienting_pre_drift.mean():.2f} ± {orienting_pre_drift.std():.2f} (median: {np.median(orienting_pre_drift):.2f})"
            )

            print(
                f"    Compensatory: {compensatory_pre_drift.mean():.2f} ± {compensatory_pre_drift.std():.2f} (median: {np.median(compensatory_pre_drift):.2f})"
            )

            print(
                f"    Mann-Whitney U test: U={pre_drift_stat:.1f}, p={pre_drift_p:.4f}"
            )

            # Post-saccade variance comparison

            orienting_post_var = orienting_saccades[
                "post_saccade_position_variance"
            ].values

            compensatory_post_var = compensatory_saccades[
                "post_saccade_position_variance"
            ].values

            post_var_stat, post_var_p = stats.mannwhitneyu(
                orienting_post_var, compensatory_post_var, alternative="two-sided"
            )

            print("\n  Post-saccade position variance (px²):")

            print(
                f"    Orienting: {orienting_post_var.mean():.2f} ± {orienting_post_var.std():.2f} (median: {np.median(orienting_post_var):.2f})"
            )

            print(
                f"    Compensatory: {compensatory_post_var.mean():.2f} ± {compensatory_post_var.std():.2f} (median: {np.median(compensatory_post_var):.2f})"
            )

            print(f"    Mann-Whitney U test: U={post_var_stat:.1f}, p={post_var_p:.4f}")

            # Bout size for compensatory saccades

            if len(compensatory_saccades) > 0:
                bout_sizes = compensatory_saccades["bout_size"].values
                print("\n  Bout size (compensatory saccades only):")
                print(
                    f"    Mean: {bout_sizes.mean():.2f} ± {bout_sizes.std():.2f} saccades"
                )
                print(
                    f"    Range: {bout_sizes.min():.0f} - {bout_sizes.max():.0f} saccades"
                )
                print(f"    Median: {np.median(bout_sizes):.0f} saccades")

            # Classification confidence comparison
            if "classification_confidence" in all_saccades_df.columns:
                orienting_conf = orienting_saccades["classification_confidence"].values

                compensatory_conf = compensatory_saccades[
                    "classification_confidence"
                ].values

                conf_stat, conf_p = stats.mannwhitneyu(
                    orienting_conf, compensatory_conf, alternative="two-sided"
                )

                print("\n  Classification Confidence:")

                print(
                    f"    Orienting: {orienting_conf.mean():.3f} ± {orienting_conf.std():.3f} (median: {np.median(orienting_conf):.3f})"
                )

                print(
                    f"    Compensatory: {compensatory_conf.mean():.3f} ± {compensatory_conf.std():.3f} (median: {np.median(compensatory_conf):.3f})"
                )

                print(f"    Mann-Whitney U test: U={conf_stat:.1f}, p={conf_p:.4f}")

        else:
            print(
                "  ⚠️ Cannot perform statistical comparisons - need both types present"
            )

        # Visualization

        # Create visualization figure

        fig_class = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                "Amplitude Distribution",
                "Duration Distribution",
                "Pre-saccade Velocity Distribution",
                "Pre-saccade Position Drift",
                "Post-saccade Position Variance",
                "Bout Size Distribution (Compensatory)",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        # Row 1, Col 1: Amplitude distributions

        if len(orienting_saccades) > 0:
            fig_class.add_trace(
                go.Histogram(
                    x=orienting_saccades["amplitude"],
                    nbinsx=30,
                    name="Orienting",
                    marker_color="blue",
                    opacity=0.6,
                    histnorm="probability",
                ),
                row=1,
                col=1,
            )

        if len(compensatory_saccades) > 0:
            fig_class.add_trace(
                go.Histogram(
                    x=compensatory_saccades["amplitude"],
                    nbinsx=30,
                    name="Compensatory",
                    marker_color="orange",
                    opacity=0.6,
                    histnorm="probability",
                ),
                row=1,
                col=1,
            )

        # Row 1, Col 2: Duration distributions

        if len(orienting_saccades) > 0:
            fig_class.add_trace(
                go.Histogram(
                    x=orienting_saccades["duration"],
                    nbinsx=30,
                    name="Orienting",
                    marker_color="blue",
                    opacity=0.6,
                    showlegend=False,
                    histnorm="probability",
                ),
                row=1,
                col=2,
            )
        if len(compensatory_saccades) > 0:
            fig_class.add_trace(
                go.Histogram(
                    x=compensatory_saccades["duration"],
                    nbinsx=30,
                    name="Compensatory",
                    marker_color="orange",
                    opacity=0.6,
                    showlegend=False,
                    histnorm="probability",
                ),
                row=1,
                col=2,
            )

        # Row 1, Col 3: Pre-saccade velocity distributions

        if len(orienting_saccades) > 0:
            fig_class.add_trace(
                go.Histogram(
                    x=orienting_saccades["pre_saccade_mean_velocity"],
                    nbinsx=30,
                    name="Orienting",
                    marker_color="blue",
                    opacity=0.6,
                    showlegend=False,
                    histnorm="probability",
                ),
                row=1,
                col=3,
            )

        if len(compensatory_saccades) > 0:
            fig_class.add_trace(
                go.Histogram(
                    x=compensatory_saccades["pre_saccade_mean_velocity"],
                    nbinsx=30,
                    name="Compensatory",
                    marker_color="orange",
                    opacity=0.6,
                    showlegend=False,
                    histnorm="probability",
                ),
                row=1,
                col=3,
            )

        # Row 2, Col 1: Pre-saccade drift distributions

        if len(orienting_saccades) > 0:
            fig_class.add_trace(
                go.Histogram(
                    x=orienting_saccades["pre_saccade_position_drift"],
                    nbinsx=30,
                    name="Orienting",
                    marker_color="blue",
                    opacity=0.6,
                    showlegend=False,
                    histnorm="probability",
                ),
                row=2,
                col=1,
            )

        if len(compensatory_saccades) > 0:
            fig_class.add_trace(
                go.Histogram(
                    x=compensatory_saccades["pre_saccade_position_drift"],
                    nbinsx=30,
                    name="Compensatory",
                    marker_color="orange",
                    opacity=0.6,
                    showlegend=False,
                    histnorm="probability",
                ),
                row=2,
                col=1,
            )

        # Row 2, Col 2: Post-saccade variance distributions

        if len(orienting_saccades) > 0:
            fig_class.add_trace(
                go.Histogram(
                    x=orienting_saccades["post_saccade_position_variance"],
                    nbinsx=30,
                    name="Orienting",
                    marker_color="blue",
                    opacity=0.6,
                    showlegend=False,
                    histnorm="probability",
                ),
                row=2,
                col=2,
            )

        if len(compensatory_saccades) > 0:
            fig_class.add_trace(
                go.Histogram(
                    x=compensatory_saccades["post_saccade_position_variance"],
                    nbinsx=30,
                    name="Compensatory",
                    marker_color="orange",
                    opacity=0.6,
                    showlegend=False,
                    histnorm="probability",
                ),
                row=2,
                col=2,
            )

        # Row 2, Col 3: Bout size distribution (compensatory only)

        if len(compensatory_saccades) > 0:
            fig_class.add_trace(
                go.Histogram(
                    x=compensatory_saccades["bout_size"],
                    nbinsx=20,
                    name="Compensatory Bout Size",
                    marker_color="orange",
                    opacity=0.6,
                    showlegend=False,
                    histnorm="probability",
                ),
                row=2,
                col=3,
            )

        else:
            # Add empty trace to maintain layout
            fig_class.add_trace(
                go.Histogram(x=[], name="No compensatory saccades"), row=2, col=3
            )

        # Update layout

        fig_class.update_layout(
            title_text=f"Saccade Classification Analysis: Orienting vs Compensatory ({get_eye_label(video_key)})",
            height=800,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
        )

        # Update axes labels
        fig_class.update_xaxes(title_text="Amplitude (px)", row=1, col=1)
        fig_class.update_xaxes(title_text="Duration (s)", row=1, col=2)
        fig_class.update_xaxes(title_text="Velocity (px/s)", row=1, col=3)
        fig_class.update_xaxes(title_text="Drift (px)", row=2, col=1)
        fig_class.update_xaxes(title_text="Variance (px²)", row=2, col=2)
        fig_class.update_xaxes(title_text="Bout Size (saccades)", row=2, col=3)

        fig_class.update_yaxes(title_text="Probability", row=1, col=1)
        fig_class.update_yaxes(title_text="Probability", row=1, col=2)
        fig_class.update_yaxes(title_text="Probability", row=1, col=3)
        fig_class.update_yaxes(title_text="Probability", row=2, col=1)
        fig_class.update_yaxes(title_text="Probability", row=2, col=2)
        fig_class.update_yaxes(title_text="Probability", row=2, col=3)

        fig_class.show()

        # Confidence distribution visualization

        if "classification_confidence" in all_saccades_df.columns:
            fig_conf = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    "Classification Confidence Distribution",
                    "Confidence by Saccade Type",
                ),
                horizontal_spacing=0.15,
            )

        # Overall confidence distribution

        fig_conf.add_trace(
            go.Histogram(
                x=all_saccades_df["classification_confidence"],
                nbinsx=30,
                name="All Saccades",
                marker_color="gray",
                opacity=0.7,
                histnorm="probability",
            ),
            row=1,
            col=1,
        )

        # Add vertical lines for confidence thresholds
        fig_conf.add_vline(
            x=0.7,
            line_dash="dash",
            line_color="green",
            annotation_text="High (≥0.7)",
            row=1,
            col=1,
        )
        fig_conf.add_vline(
            x=0.4,
            line_dash="dash",
            line_color="orange",
            annotation_text="Medium (0.4-0.7)",
            row=1,
            col=1,
        )

        # Confidence by type
        if len(orienting_saccades) > 0:
            fig_conf.add_trace(
                go.Histogram(
                    x=orienting_saccades["classification_confidence"],
                    nbinsx=30,
                    name="Orienting",
                    marker_color="blue",
                    opacity=0.6,
                    histnorm="probability",
                ),
                row=1,
                col=2,
            )
        if len(compensatory_saccades) > 0:
            fig_conf.add_trace(
                go.Histogram(
                    x=compensatory_saccades["classification_confidence"],
                    nbinsx=30,
                    name="Compensatory",
                    marker_color="orange",
                    opacity=0.6,
                    histnorm="probability",
                ),
                row=1,
                col=2,
            )

        fig_conf.update_layout(
            title_text=f"Classification Confidence Analysis ({get_eye_label(video_key)})",
            height=400,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
        )

        fig_conf.update_xaxes(title_text="Confidence Score", row=1, col=1)

        fig_conf.update_xaxes(title_text="Confidence Score", row=1, col=2)

        fig_conf.update_yaxes(title_text="Probability", row=1, col=1)

        fig_conf.update_yaxes(title_text="Probability", row=1, col=2)

        fig_conf.show()

        # Time series visualization with classification

        fig_ts = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                "X Position (px)",
                "Velocity (px/s) with Classified Saccades",
            ),
        )

        # Add position trace

        fig_ts.add_trace(
            go.Scatter(
                x=res["df"]["Seconds"],
                y=res["df"]["X_smooth"],
                mode="lines",
                name="Smoothed X",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Add velocity trace

        fig_ts.add_trace(
            go.Scatter(
                x=res["df"]["Seconds"],
                y=res["df"]["vel_x_smooth"],
                mode="lines",
                name="Smoothed Velocity",
                line=dict(color="red", width=2),
            ),
            row=2,
            col=1,
        )

        # Add adaptive threshold lines
        fig_ts.add_hline(
            y=res["vel_thresh"],
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            annotation_text=f"Adaptive threshold (±{res['vel_thresh']:.0f} px/s)",
            row=2,
            col=1,
        )
        fig_ts.add_hline(
            y=-res["vel_thresh"],
            line_dash="dash",
            line_color="green",
            opacity=0.5,
            row=2,
            col=1,
        )

        # Calculate position y-axis range for vertical lines
        pos_max = res["df"]["X_smooth"].max()
        pos_min = res["df"]["X_smooth"].min()
        pos_range = pos_max - pos_min
        # Add small padding to ensure lines span full visible range
        pos_padding = pos_range * 0.05
        pos_y_min = pos_min - pos_padding
        pos_y_max = pos_max + pos_padding

        # Plot orienting saccades (blue) as rectangles on position trace
        orienting_in_df = all_saccades_df[
            all_saccades_df["saccade_type"] == "orienting"
        ]
        if len(orienting_in_df) > 0:
            for idx, row in orienting_in_df.iterrows():
                start_time = row["start_time"]

                end_time = row["end_time"]

                # Use rectangle with thin border to show duration span more
                # efficiently

                # Opacity must be set via rgba color, not in line dict

                fig_ts.add_shape(
                    type="rect",
                    x0=start_time,
                    y0=pos_y_min,
                    x1=end_time,
                    y1=pos_y_max,
                    fillcolor="rgba(0,0,255,0.1)",  # Light blue fill with opacity
                    # Blue border with opacity via rgba
                    line=dict(color="rgba(0,0,255,0.3)", width=2),
                    row=1,
                    col=1,
                )

        # Legend entry

        fig_ts.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="Orienting Saccades",
                line=dict(
                    color="rgba(0,0,255,0.3)", width=3
                ),  # Blue with opacity via rgba
            ),
            row=1,
            col=1,
        )

        # Plot compensatory saccades (orange) as rectangles on position trace

        compensatory_in_df = all_saccades_df[
            all_saccades_df["saccade_type"] == "compensatory"
        ]

        if len(compensatory_in_df) > 0:
            for idx, row in compensatory_in_df.iterrows():
                start_time = row["start_time"]
                end_time = row["end_time"]

                # Use rectangle with thin border to show duration span more
                # efficiently

                # Opacity must be set via rgba color, not in line dict

                fig_ts.add_shape(
                    type="rect",
                    x0=start_time,
                    y0=pos_y_min,
                    x1=end_time,
                    y1=pos_y_max,
                    # Light orange fill with opacity
                    fillcolor="rgba(255,165,0,0.1)",
                    # Orange border with opacity via rgba
                    line=dict(color="rgba(255,165,0,0.3)", width=2),
                    row=1,
                    col=1,
                )

        # Legend entry

        fig_ts.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="Compensatory Saccades",
                line=dict(
                    color="rgba(255,165,0,0.3)", width=3
                ),  # Orange with opacity via rgba
            ),
            row=1,
            col=1,
        )

        # Update layout
        fig_ts.update_layout(
            title=f"Time Series with Saccade Classification ({get_eye_label(video_key)})<br><sub>Blue: Orienting, Orange: Compensatory</sub>",
            height=600,
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
        )

        # Update axes
        fig_ts.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig_ts.update_yaxes(title_text="X Position (px)", row=1, col=1)
        fig_ts.update_yaxes(title_text="Velocity (px/s)", row=2, col=1)

        fig_ts.show()


# +
# ML Feature Extraction and Visualization
# Extract features for ML classification and visualize distributions
##########################################################################


# Extract features for VideoData1 (if available)
features_v1 = None
if VideoData1_Has_Sleap and "VideoData1" in saccade_results:
    print("=" * 80)
    print("Extracting ML features for VideoData1...")
    print("=" * 80)
    # Use the processed df from saccade_results (has X_smooth, vel_x_smooth
    # columns)
    df1_processed = saccade_results["VideoData1"]["df"]
    features_v1 = extract_ml_features(
        saccade_results=saccade_results["VideoData1"],
        df=df1_processed,  # Use processed df with X_smooth, vel_x_smooth
        fps=FPS_1,
        data_path=data_path,
        verbose=True,
    )
    if len(features_v1) > 0:
        features_v1["eye"] = "Left" if video1_eye == "L" else "Right"
        print(
            f"✅ Extracted {len(features_v1)} saccades with {len(features_v1.columns)} features"
        )

# Extract features for VideoData2 (if available)
features_v2 = None
if VideoData2_Has_Sleap and "VideoData2" in saccade_results:
    print("\n" + "=" * 80)
    print("Extracting ML features for VideoData2...")
    print("=" * 80)
    # Use the processed df from saccade_results (has X_smooth, vel_x_smooth
    # columns)
    df2_processed = saccade_results["VideoData2"]["df"]
    features_v2 = extract_ml_features(
        saccade_results=saccade_results["VideoData2"],
        df=df2_processed,  # Use processed df with X_smooth, vel_x_smooth
        fps=FPS_2,
        data_path=data_path,
        verbose=True,
    )
    if len(features_v2) > 0:
        features_v2["eye"] = "Right" if video1_eye == "L" else "Left"
        print(
            f"✅ Extracted {len(features_v2)} saccades with {len(features_v2.columns)} features"
        )

# Combine features from both eyes
if features_v1 is not None and features_v2 is not None:
    features_combined = pd.concat([features_v1, features_v2], ignore_index=True)
    print(
        f"\n✅ Combined features: {len(features_combined)} total saccades ({len(features_v1)} left, {len(features_v2)} right)"
    )
elif features_v1 is not None:
    features_combined = features_v1.copy()
    print(f"\n✅ Using VideoData1 only: {len(features_combined)} saccades")
elif features_v2 is not None:
    features_combined = features_v2.copy()
    print(f"\n✅ Using VideoData2 only: {len(features_combined)} saccades")
else:
    print("\n⚠️ No features extracted - check if saccades were detected")
    features_combined = None


# +
# Visualize Feature Distributions: Panel 1 (Violin Plots by Category) + Panel 2 (Key Features by Class)
##########################################################################

if features_combined is not None and len(features_combined) > 0:
    # Define feature categories
    feature_categories = {
        "Category A: Basic Properties": [
            "amplitude",
            "duration",
            "peak_velocity",
            "direction",
            "start_time",
            "end_time",
            "time",
        ],
        "Category B: Pre-Saccade": [
            "pre_saccade_mean_velocity",
            "pre_saccade_position_drift",
            "pre_saccade_position_variance",
            "pre_saccade_drift_rate",
            "pre_saccade_window_duration",
        ],
        "Category C: Post-Saccade": [
            "post_saccade_position_variance",
            "post_saccade_position_change",
            "post_saccade_position_change_pct",
            "post_saccade_mean_velocity",
            "post_saccade_drift_rate",
            "post_saccade_window_duration",
        ],
        "Category D: Temporal Context": [
            "time_since_previous_saccade",
            "time_until_next_saccade",
            "bout_size",
            "position_in_bout",
            "is_first_in_bout",
            "is_last_in_bout",
            "is_isolated",
            "bout_duration",
            "inter_saccade_interval_mean",
            "inter_saccade_interval_std",
        ],
        "Category G: Amplitude/Direction Consistency": [
            "amplitude_relative_to_bout_mean",
            "amplitude_consistency_in_bout",
            "direction_relative_to_previous",
        ],
        "Category H: Rule-Based Classification": [
            "rule_based_class",
            "rule_based_confidence",
        ],
    }

    # Use the visualization function
    visualize_ml_features(
        features_combined=features_combined,
        feature_categories=feature_categories,
        video_labels=VIDEO_LABELS,
        show_plots=True,
        verbose=True,
    )
else:
    print("⚠️ No features available for visualization. Run feature extraction first.")

# +
# LAUNCH GUI ANNOTATION TOOL
##########################################################################
# Use this cell to launch the GUI for manual annotation of saccades
# The GUI starts with rule-based classifications and allows you to correct
# them to 4 classes

# Get experiment ID
experiment_id = extract_experiment_id(data_path)
print(f"Experiment ID: {experiment_id}")

# Launch GUI with BOTH eyes combined (VideoData1 and VideoData2)
# The GUI will automatically combine saccades from both eyes and display
# them together

# Get features for BOTH eyes (since we're combining both eyes' saccades)
# features_combined already contains features from both VideoData1 and
# VideoData2
video_features = (
    features_combined
    if features_combined is not None and len(features_combined) > 0
    else None
)

# Set annotations file path (save in data_path parent directory)
annotations_file = data_path.parent / "saccade_annotations_master.csv"
# Or use a project-wide location:
# annotations_file = Path('/Users/rancze/Documents/Data/vestVR/saccade_annotations_master.csv')

# Count saccades from both eyes for display
total_saccades = 0
eye_breakdown = {}
if isinstance(saccade_results, dict):
    for key in ["VideoData1", "VideoData2"]:
        if key in saccade_results:
            eye_data = saccade_results[key]
            if "all_saccades_df" in eye_data:
                count = len(eye_data["all_saccades_df"])
                total_saccades += count
                eye_label = VIDEO_LABELS.get(key, key)
                eye_breakdown[eye_label] = count

print(f"\n{'=' * 80}")
print("Launching GUI Annotation Tool (BOTH EYES)")
print(f"{'=' * 80}")
print(f"Experiment ID: {experiment_id}")
print(f"Eyes: {', '.join(eye_breakdown.keys()) if eye_breakdown else 'Unknown'}")
print(f"Total saccades: {total_saccades}")
if eye_breakdown:
    for eye, count in eye_breakdown.items():
        print(f"  - {eye}: {count} saccades")
print(f"Features: {len(video_features) if video_features is not None else 0}")
print(f"Annotations file: {annotations_file}")
print(f"{'=' * 80}\n")
print("ℹ️ Instructions:")
print(
    "  - Use keyboard shortcuts: 1=Compensatory, 2=Orienting, 3=Saccade-and-Fixate, 4=Non-Saccade"
)
print("  - Navigation: . = Next, , = Previous, S = Save")
print("  - Click on saccades in the table to select them")
print("  - The table shows saccades from both eyes (L and R) combined")
print("  - Close the GUI window when done annotating")
print(f"{'=' * 80}\n")

# Launch GUI (this will block until GUI is closed)
# Pass the FULL saccade_results dict so both eyes are combined
launch_annotation_gui(
    # Pass full dict with VideoData1 and VideoData2 keys
    saccade_results=saccade_results,
    features_df=video_features,  # Pass all features (both eyes)
    experiment_id=experiment_id,
    annotations_file_path=annotations_file,
)

# After GUI closes, show statistics
print(f"\n{'=' * 80}")
print("Annotation session complete!")
print(f"{'=' * 80}")

# Load and display annotations
annotations = load_annotations(annotations_file, experiment_id=experiment_id)
if len(annotations) > 0:
    print(f"\n✅ Annotated {len(annotations)} saccades for this experiment")
    print_annotation_stats(annotations_file, experiment_id=experiment_id)
else:
    print("\n⚠️ No annotations saved for this experiment")
