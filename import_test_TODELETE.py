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
force_reload_modules = False
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