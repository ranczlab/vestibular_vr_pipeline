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
# ## 1_3_1 Saccade Detection + Quantification (rebuild)
#
# This notebook includes:
# - input contract + loading of selected eye parquet from `1_2` metadata
# - detection + filtering parameter block
# - preprocessing (`X_raw`, `X_smooth`, `vel_x_smooth`)
# - velocity-threshold saccade detection
# - post-detection filtering (dedup, transient-pair, amplitude)
# - QC plots and manual QC annotation
#

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
from plotly.subplots import make_subplots


# %%
# Cell 1: Input contract + load selected eye data from 1_2 metadata
##########################################################################

# good S/N
data_path = Path(
    "/Users/rancze/Documents/Data/vestVR/20250409_Cohort3_rotation/Visual_mismatch_day4/B6J2783-2025-04-28T14-57-30"
)

# #bad S/N
# data_path = Path(
#     "/Users/rancze/Documents/Data/vestVR/20250409_Cohort3_rotation/Visual_mismatch_day4/B6J2782-2025-04-28T14-22-03"
# )

# data_path = Path(
#     "/Users/rancze/Documents/Data/vestVR/20250409_Cohort3_rotation/Visual_mismatch_day4/B6J2781-2025-04-28T13-45-40"
# )

override_detect_params_with_metadata = True
debug = False

save_path = data_path.parent / f"{data_path.name}_processedData"
downsampled_output_dir = save_path / "downsampled_data"
metadata_path = downsampled_output_dir / "saccade_input_metadata.json"

if not metadata_path.exists():
    raise FileNotFoundError(
        f"Metadata not found at {metadata_path}. Run 1_2_SLEAP_processing.ipynb first."
    )

with open(metadata_path, "r") as f:
    metadata = json.load(f)

required_top_keys = [
    "eye_with_least_low_confidence",
    "video1_eye",
    "video2_eye",
    "outputs",
]
missing_top_keys = [k for k in required_top_keys if k not in metadata]
if missing_top_keys:
    raise KeyError(f"Metadata missing required keys: {missing_top_keys}")

eye_with_least_low_confidence = metadata.get("eye_with_least_low_confidence")
if eye_with_least_low_confidence not in {"VideoData1", "VideoData2"}:
    raise ValueError(
        "metadata['eye_with_least_low_confidence'] is missing or invalid. "
        "Expected 'VideoData1' or 'VideoData2'."
    )

selected_eye_code = (
    metadata.get("video1_eye")
    if eye_with_least_low_confidence == "VideoData1"
    else metadata.get("video2_eye")
)
if selected_eye_code not in {"L", "R"}:
    selected_eye_code = "Unknown"

video_outputs = metadata.get("outputs", {}).get(eye_with_least_low_confidence, {})
video_parquet_path = video_outputs.get("resampled_parquet")
if video_parquet_path is None:
    raise ValueError(
        f"No resampled_parquet path found in metadata for {eye_with_least_low_confidence}."
    )

video_parquet_path = Path(video_parquet_path)
if not video_parquet_path.exists():
    raise FileNotFoundError(
        f"Parquet file not found at {video_parquet_path} for {eye_with_least_low_confidence}."
    )

VideoData = pd.read_parquet(video_parquet_path).reset_index(drop=True)

required_columns = ["Seconds", "Ellipse.Center.X"]
missing_columns = [c for c in required_columns if c not in VideoData.columns]
if missing_columns:
    raise KeyError(f"Loaded VideoData is missing required columns: {missing_columns}")

if VideoData["Seconds"].diff().dropna().le(0).any():
    raise ValueError("Seconds must be strictly increasing.")

if "frame_idx" not in VideoData.columns:
    VideoData["frame_idx"] = np.arange(len(VideoData), dtype=int)

if debug:
    print(f"Loaded metadata: {metadata_path}")
    print(f"Selected eye/video: {eye_with_least_low_confidence} ({selected_eye_code})")
    print(f"Loaded parquet: {video_parquet_path}")
    print(f"VideoData shape: {VideoData.shape}")
    display(VideoData.head())

    # Debug plot (inline): Ellipse.Center.X vs relative time
    time_rel_s = VideoData["Seconds"] - VideoData["Seconds"].iloc[0]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=time_rel_s,
                y=VideoData["Ellipse.Center.X"],
                mode="lines",
                name="Ellipse.Center.X",
                line=dict(width=1),
            )
        ]
    )
    fig.update_layout(
        title=f"Ellipse.Center.X ({eye_with_least_low_confidence}, eye={selected_eye_code})",
        xaxis_title="Relative time (s)",
        yaxis_title="Ellipse.Center.X (px)",
        template="plotly_white",
    )
    fig.show()
else:
    print(
        f"✅ Loaded selected eye/video: {eye_with_least_low_confidence} ({selected_eye_code})"
    )


# %%
# Cell 2: Detection and filtering parameters
##########################################################################

# --- Signal preprocessing ---
smoothing_window_s = 0.08  # median smoothing window before velocity calculation

# --- Velocity-threshold event detection ---
k = 3.4  # single detection sensitivity parameter
peak_width_time_s = 0.005  # minimum peak width for velocity peaks
onset_fraction = (
    0.2  # for saccade duration, start boundary when |vel| falls below this * threshold
)
offset_fraction = (
    0.1  # for saccade duration, end boundary when |vel| falls below this * threshold
)

# --- Post-detection filtering ---
refractory_period_s = 0.2  # to detect transients, transient-pair ISI window (not used in find_peaks spacing)
same_direction_dedup_window_s = (
    refractory_period_s  # collapse close same-direction duplicates
)
transient_pair_max_net_displacement_px = 3.0  # max pair net displacement to classify close opposite-direction transient, i.e. does the transient come back to baseline?
min_saccade_amplitude_px = 3.0  # minimum amplitude to keep final event

# --- Optional downstream analysis/QC controls ---
pre_window_s = 0.15  # peri-event snippet window before event time
post_window_s = 0.5  # peri-event snippet window after event time
baseline_window_start_s = -0.06  # baseline window start (relative to event)
baseline_window_end_s = -0.02  # baseline window end (relative to event)
plot_detection_qc = True  # render Cell 8/9 QC figures

# If enabled, metadata can override Cell 2 defaults from prior runs.
loaded_params = metadata.get("saccade_detection_parameters", {})
if (
    override_detect_params_with_metadata
    and isinstance(loaded_params, dict)
    and len(loaded_params) > 0
):

    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    _overridable_params = {
        "smoothing_window_s": float,
        "k": float,
        "peak_width_time_s": float,
        "onset_fraction": float,
        "offset_fraction": float,
        "refractory_period_s": float,
        "same_direction_dedup_window_s": float,
        "transient_pair_max_net_displacement_px": float,
        "min_saccade_amplitude_px": float,
        "pre_window_s": float,
        "post_window_s": float,
        "baseline_window_start_s": float,
        "baseline_window_end_s": float,
        "plot_detection_qc": _as_bool,
    }

    _local_scope = locals()
    for _key, _caster in _overridable_params.items():
        if _key not in _local_scope or _key not in loaded_params:
            continue
        try:
            _local_scope[_key] = _caster(loaded_params[_key])
        except (TypeError, ValueError):
            pass

    # Propagate overrides back into the local namespace.
    smoothing_window_s = _local_scope["smoothing_window_s"]
    k = _local_scope["k"]
    peak_width_time_s = _local_scope["peak_width_time_s"]
    onset_fraction = _local_scope["onset_fraction"]
    offset_fraction = _local_scope["offset_fraction"]
    refractory_period_s = _local_scope["refractory_period_s"]
    same_direction_dedup_window_s = _local_scope["same_direction_dedup_window_s"]
    transient_pair_max_net_displacement_px = _local_scope[
        "transient_pair_max_net_displacement_px"
    ]
    min_saccade_amplitude_px = _local_scope["min_saccade_amplitude_px"]
    pre_window_s = _local_scope["pre_window_s"]
    post_window_s = _local_scope["post_window_s"]
    baseline_window_start_s = _local_scope["baseline_window_start_s"]
    baseline_window_end_s = _local_scope["baseline_window_end_s"]
    plot_detection_qc = _local_scope["plot_detection_qc"]

    print("ℹ️ Loaded Cell 2 parameter overrides from metadata.")

print("✅ Detection and filtering parameters initialized")


# %%
# Cell 2b: Prepare Cell 2 parameters for metadata save (written in Cell 10)
##########################################################################
cell2_params = {
    # Keep only currently-defined parameters to avoid stale-key crashes.
}


def _add_param_if_defined(target_dict, key, caster):
    scope = globals()
    if key not in scope:
        return
    try:
        target_dict[key] = caster(scope[key])
    except (TypeError, ValueError):
        return


_add_param_if_defined(cell2_params, "smoothing_window_s", float)
_add_param_if_defined(cell2_params, "k", float)
_add_param_if_defined(cell2_params, "peak_width_time_s", float)
_add_param_if_defined(cell2_params, "onset_fraction", float)
_add_param_if_defined(cell2_params, "offset_fraction", float)
_add_param_if_defined(cell2_params, "refractory_period_s", float)
_add_param_if_defined(cell2_params, "same_direction_dedup_window_s", float)
_add_param_if_defined(cell2_params, "transient_pair_max_net_displacement_px", float)
_add_param_if_defined(cell2_params, "min_saccade_amplitude_px", float)
_add_param_if_defined(cell2_params, "pre_window_s", float)
_add_param_if_defined(cell2_params, "post_window_s", float)
_add_param_if_defined(cell2_params, "baseline_window_start_s", float)
_add_param_if_defined(cell2_params, "baseline_window_end_s", float)
_add_param_if_defined(cell2_params, "plot_detection_qc", bool)

print("ℹ️ Cell 2 parameters prepared; they will be saved with Manual QC in Cell 10.")


# %%
# Cell 3: Preprocess eye position for velocity-based saccade detection
##########################################################################

df_work = VideoData[["Seconds", "Ellipse.Center.X", "frame_idx"]].copy()
df_work["X_raw"] = df_work["Ellipse.Center.X"].astype(float)

fps = 1.0 / df_work["Seconds"].diff().median()

smoothing_window_points = max(1, int(round(smoothing_window_s * fps)))
df_work["X_smooth"] = (
    df_work["X_raw"]
    .rolling(window=smoothing_window_points, center=True)
    .median()
    .bfill()
    .ffill()
)

df_work["dt"] = df_work["Seconds"].diff()
eps = np.finfo(float).eps
df_work.loc[df_work["dt"].abs() <= eps, "dt"] = np.nan
df_work.loc[df_work["dt"] <= 0, "dt"] = np.nan

df_work["vel_x_smooth"] = (df_work["X_smooth"].diff() / df_work["dt"]).astype(float)
df_work["vel_x_raw"] = (df_work["X_raw"].diff() / df_work["dt"]).astype(float)
df_work["vel_x_smooth"] = df_work["vel_x_smooth"].where(
    np.isfinite(df_work["vel_x_smooth"]), np.nan
)
df_work["vel_x_raw"] = df_work["vel_x_raw"].where(
    np.isfinite(df_work["vel_x_raw"]), np.nan
)

print(
    f"✅ Preprocessing complete | fps={fps:.3f} | smoothing_window_points={smoothing_window_points}"
)


# %%
# Cell 4: Prepare shared velocity summary for detection
##########################################################################
# Keep one shared velocity summary for thresholding.
abs_vel = df_work["vel_x_smooth"].abs().dropna()
if len(abs_vel) == 0:
    raise ValueError("No finite velocity samples available for saccade detection.")
print("ℹ️ Velocity summary prepared for detection.")


# %%
# Cell 5: Detection run
##########################################################################
v = df_work["vel_x_smooth"].to_numpy()
x = df_work["X_raw"].to_numpy()
t = df_work["Seconds"].to_numpy()
f = df_work["frame_idx"].to_numpy()
peak_width_points = max(1, int(round(peak_width_time_s * fps)))


def _split_transient_pairs(
    events_df: pd.DataFrame, max_isi_s: float, max_net_displacement_px: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split events into kept vs transient-pair filtered (dropped) events."""
    if len(events_df) < 2:
        return events_df.reset_index(drop=True), pd.DataFrame(columns=events_df.columns)

    times = events_df["time"].to_numpy(dtype=float)
    directions = events_df["direction"].to_numpy()
    starts = events_df["start_position"].to_numpy(dtype=float)
    ends = events_df["end_position"].to_numpy(dtype=float)
    keep_mask = np.ones(len(events_df), dtype=bool)

    i = 0
    while i < len(events_df) - 1:
        isi = times[i + 1] - times[i]
        opposite_direction = directions[i] != directions[i + 1]
        pair_net_disp = abs(ends[i + 1] - starts[i])
        if (
            isi <= max_isi_s
            and opposite_direction
            and pair_net_disp <= max_net_displacement_px
        ):
            keep_mask[i] = False
            keep_mask[i + 1] = False
            i += 2
            continue
        i += 1

    kept_df = events_df.loc[keep_mask].reset_index(drop=True)
    dropped_df = events_df.loc[~keep_mask].reset_index(drop=True)
    return kept_df, dropped_df


def _dedupe_same_direction_events(
    events_df: pd.DataFrame, min_isi_s: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep strongest event within close same-direction clusters."""
    if len(events_df) < 2:
        return events_df.reset_index(drop=True), pd.DataFrame(columns=events_df.columns)

    times = events_df["time"].to_numpy(dtype=float)
    directions = events_df["direction"].to_numpy()
    peak_abs_vel = np.abs(events_df["velocity"].to_numpy(dtype=float))
    keep_mask = np.ones(len(events_df), dtype=bool)

    i = 0
    while i < len(events_df):
        cluster_indices = [i]
        j = i + 1
        while (
            j < len(events_df)
            and directions[j] == directions[j - 1]
            and (times[j] - times[j - 1]) <= min_isi_s
        ):
            cluster_indices.append(j)
            j += 1

        if len(cluster_indices) > 1:
            strongest_idx = cluster_indices[
                int(np.argmax(peak_abs_vel[cluster_indices]))
            ]
            for idx in cluster_indices:
                if idx != strongest_idx:
                    keep_mask[idx] = False

        i = j

    kept_df = events_df.loc[keep_mask].reset_index(drop=True)
    dropped_df = events_df.loc[~keep_mask].reset_index(drop=True)
    return kept_df, dropped_df


def _events_for_k(k_value: float) -> tuple[pd.DataFrame, float, float, float]:
    """Run detection for a single k and return events + threshold stats."""
    vel_mean_k = abs_vel.mean()
    vel_std_k = abs_vel.std()
    vel_thresh_k = vel_mean_k + k_value * vel_std_k

    pos_peaks_k, _ = find_peaks(
        v,
        height=vel_thresh_k,
        width=peak_width_points,
    )
    neg_peaks_k, _ = find_peaks(
        -v,
        height=vel_thresh_k,
        width=peak_width_points,
    )

    def _extract_events_k(peak_indices, direction):
        events = []
        for peak_idx in peak_indices:
            start_idx = int(peak_idx)
            end_idx = int(peak_idx)

            while start_idx > 0 and abs(v[start_idx]) > vel_thresh_k * onset_fraction:
                start_idx -= 1

            while (
                end_idx < len(v) - 1
                and abs(v[end_idx]) > vel_thresh_k * offset_fraction
            ):
                end_idx += 1
            if end_idx < len(v) - 1:
                end_idx += 1

            start_x = x[start_idx]
            end_x = x[end_idx]
            displacement = end_x - start_x

            events.append(
                {
                    "direction": direction,
                    "time": float(t[peak_idx]),
                    "velocity": float(v[peak_idx]),
                    "start_time": float(t[start_idx]),
                    "end_time": float(t[end_idx]),
                    "duration": float(t[end_idx] - t[start_idx]),
                    "start_position": float(start_x),
                    "end_position": float(end_x),
                    "amplitude": float(abs(displacement)),
                    "displacement": float(displacement),
                    "start_frame_idx": int(f[start_idx]),
                    "peak_frame_idx": int(f[peak_idx]),
                    "end_frame_idx": int(f[end_idx]),
                }
            )
        return pd.DataFrame(events)

    up_df = _extract_events_k(pos_peaks_k, "upward")
    down_df = _extract_events_k(neg_peaks_k, "downward")
    events_df = pd.concat([up_df, down_df], ignore_index=True)
    if len(events_df) > 0:
        events_df = events_df.sort_values("time").reset_index(drop=True)
    return events_df, vel_thresh_k, vel_mean_k, vel_std_k


k_selected = float(k)
selected_events_df, vel_thresh, vel_mean, vel_std = _events_for_k(k_selected)
all_saccades_df = selected_events_df.copy()
upward_saccades_df = all_saccades_df[all_saccades_df["direction"] == "upward"].copy()
downward_saccades_df = all_saccades_df[
    all_saccades_df["direction"] == "downward"
].copy()

print("✅ Detection complete")
print(f"  k={k_selected:.2f}")
print(f"  vel_thresh={vel_thresh:.2f} px/s")
print(
    f"  upward={len(upward_saccades_df)}, downward={len(downward_saccades_df)}, total={len(all_saccades_df)}"
)


# %%
# Cell 7: Post-detection cleanup (transient filter, then amplitude filter)
##########################################################################
all_saccades_df_pre_filter = all_saccades_df.copy()
upward_saccades_df_pre_filter = upward_saccades_df.copy()
downward_saccades_df_pre_filter = downward_saccades_df.copy()

# 1) Same-direction de-dup first (keep strongest in close clusters).
all_saccades_df_after_detection = all_saccades_df_pre_filter.sort_values(
    "time"
).reset_index(drop=True)
all_saccades_df_after_dedup, dedup_removed_saccades_df = _dedupe_same_direction_events(
    all_saccades_df_after_detection,
    min_isi_s=same_direction_dedup_window_s,
)
n_before_dedup = len(all_saccades_df_after_detection)
n_after_dedup = len(all_saccades_df_after_dedup)
n_removed_dedup = len(dedup_removed_saccades_df)
removed_dedup_pct = (
    (100.0 * n_removed_dedup / n_before_dedup) if n_before_dedup > 0 else 0.0
)
print(
    f"✅ Same-direction de-dup applied (window={same_direction_dedup_window_s:.3f}s) | "
    f"kept={n_after_dedup}/{n_before_dedup}, removed={n_removed_dedup} ({removed_dedup_pct:.1f}%)"
)

# 2) Transient-pair filter second (teal markers in Cell 8).
all_saccades_df, transient_removed_saccades_df = _split_transient_pairs(
    all_saccades_df_after_dedup,
    max_isi_s=refractory_period_s,
    max_net_displacement_px=transient_pair_max_net_displacement_px,
)
n_before_transient = len(all_saccades_df_after_dedup)
n_after_transient = len(all_saccades_df)
n_removed_transient = len(transient_removed_saccades_df)
removed_transient_pct = (
    (100.0 * n_removed_transient / n_before_transient)
    if n_before_transient > 0
    else 0.0
)
print(
    f"✅ Transient-pair filter applied (window={refractory_period_s:.3f}s, "
    f"net_disp<={transient_pair_max_net_displacement_px:.2f} px) | "
    f"kept={n_after_transient}/{n_before_transient}, "
    f"removed={n_removed_transient} ({removed_transient_pct:.1f}%)"
)

# 3) Amplitude filter third (gray markers in Cell 8).
all_saccades_df_pre_amplitude = all_saccades_df.copy()
all_saccades_df = all_saccades_df[
    all_saccades_df["amplitude"] >= min_saccade_amplitude_px
].copy()
all_saccades_df = all_saccades_df.sort_values("time").reset_index(drop=True)
upward_saccades_df = all_saccades_df[all_saccades_df["direction"] == "upward"].copy()
downward_saccades_df = all_saccades_df[
    all_saccades_df["direction"] == "downward"
].copy()
removed_saccades_df = all_saccades_df_pre_amplitude[
    all_saccades_df_pre_amplitude["amplitude"] < min_saccade_amplitude_px
].copy()

n_before_amp = len(all_saccades_df_pre_amplitude)
n_after_amp = len(all_saccades_df)
n_removed_amp = len(removed_saccades_df)
removed_amp_pct = (100.0 * n_removed_amp / n_before_amp) if n_before_amp > 0 else 0.0
print(
    f"✅ Amplitude filter applied (min_saccade_amplitude_px={min_saccade_amplitude_px:.2f} px) | "
    f"kept={n_after_amp}/{n_before_amp}, removed={n_removed_amp} ({removed_amp_pct:.1f}%)"
)

if debug and n_removed_amp > 0:
    display(
        removed_saccades_df[
            [
                "time",
                "direction",
                "velocity",
                "duration",
                "amplitude",
                "start_time",
                "end_time",
            ]
        ].head(30)
    )


# %%
# Cell 8: QC plots (after amplitude filtering)
##########################################################################
if plot_detection_qc:
    print(
        "ℹ️ QC note: In decimated plots, rectangle widths may not always reflect true "
        "saccade duration exactly."
    )
    print("ℹ️ Filled circles: detected saccade points (velocity threshold crossings).")
    print("ℹ️ Teal hollow circles: transient-pair filtered events.")
    print("ℹ️ Gray hollow circles: events dropped by the amplitude threshold.")
    # Fast QC settings
    qc_max_points_global = 30000
    qc_max_points_velocity = 150000
    qc_density_window_s = 30.0
    qc_density_step_s = 10.0
    qc_local_window_s = 30.0

    # 1) Global lightweight overview (decimated traces + event spans/peaks)
    time_rel_s = (df_work["Seconds"] - df_work["Seconds"].iloc[0]).to_numpy()
    n_pts = len(df_work)
    stride_pos = max(1, int(np.ceil(n_pts / qc_max_points_global)))
    stride_vel = max(1, int(np.ceil(n_pts / qc_max_points_velocity)))
    idx_pos = np.arange(0, n_pts, stride_pos)
    idx_vel = np.arange(0, n_pts, stride_vel)

    fig_overlay = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "X position (decimated)",
            "Velocity with threshold",
        ),
    )

    fig_overlay.add_trace(
        go.Scatter(
            x=time_rel_s[idx_pos],
            y=df_work["X_raw"].to_numpy()[idx_pos],
            mode="lines",
            name=f"X_raw (1/{stride_pos} decimation)",
            line=dict(width=1, color="royalblue"),
        ),
        row=1,
        col=1,
    )
    fig_overlay.add_trace(
        go.Scatter(
            x=time_rel_s[idx_vel],
            y=df_work["vel_x_smooth"].to_numpy()[idx_vel],
            mode="lines",
            name=f"vel_x_smooth (1/{stride_vel} decimation)",
            line=dict(width=1, color="firebrick"),
        ),
        row=2,
        col=1,
    )

    fig_overlay.add_hline(
        y=vel_thresh,
        line_dash="dash",
        line_color="green",
        annotation_text=f"+thr ({vel_thresh:.1f})",
        row=2,
        col=1,
    )
    fig_overlay.add_hline(
        y=-vel_thresh,
        line_dash="dash",
        line_color="green",
        annotation_text=f"-thr ({-vel_thresh:.1f})",
        row=2,
        col=1,
    )

    # Direction-colored threshold-crossing markers on position panel.
    # Use start_time (onset crossing) and draw as dots.
    if len(upward_saccades_df) > 0:
        up_x = upward_saccades_df["start_time"] - df_work["Seconds"].iloc[0]
        up_y = np.interp(
            upward_saccades_df["start_time"].to_numpy(dtype=float),
            df_work["Seconds"].to_numpy(dtype=float),
            df_work["X_raw"].to_numpy(dtype=float),
        )
        fig_overlay.add_trace(
            go.Scatter(
                x=up_x,
                y=up_y,
                mode="markers",
                name="upward threshold-crossing",
                marker=dict(color="limegreen", size=7, symbol="circle"),
            ),
            row=1,
            col=1,
        )
    if len(downward_saccades_df) > 0:
        down_x = downward_saccades_df["start_time"] - df_work["Seconds"].iloc[0]
        down_y = np.interp(
            downward_saccades_df["start_time"].to_numpy(dtype=float),
            df_work["Seconds"].to_numpy(dtype=float),
            df_work["X_raw"].to_numpy(dtype=float),
        )
        fig_overlay.add_trace(
            go.Scatter(
                x=down_x,
                y=down_y,
                mode="markers",
                name="downward threshold-crossing",
                marker=dict(color="mediumpurple", size=7, symbol="circle"),
            ),
            row=1,
            col=1,
        )

    # Threshold-crossings filtered as transient pairs (post-hoc refractory logic).
    # Show as teal hollow circles on position panel only (no duration rectangles).
    if (
        "transient_removed_saccades_df" in globals()
        and len(transient_removed_saccades_df) > 0
    ):
        transient_x = (
            transient_removed_saccades_df["start_time"] - df_work["Seconds"].iloc[0]
        )
        transient_y = np.interp(
            transient_removed_saccades_df["start_time"].to_numpy(dtype=float),
            df_work["Seconds"].to_numpy(dtype=float),
            df_work["X_raw"].to_numpy(dtype=float),
        )
        fig_overlay.add_trace(
            go.Scatter(
                x=transient_x,
                y=transient_y,
                mode="markers",
                name="transient-pair filtered threshold-crossing",
                marker=dict(
                    color="rgba(0,0,0,0)",
                    size=8,
                    symbol="circle",
                    line=dict(color="darkcyan", width=1.5),
                ),
            ),
            row=1,
            col=1,
        )

    # Threshold-crossings that were filtered out by amplitude threshold.
    # Show as gray hollow circles on position panel only (no duration rectangles).
    if "removed_saccades_df" in globals() and len(removed_saccades_df) > 0:
        removed_x = removed_saccades_df["start_time"] - df_work["Seconds"].iloc[0]
        removed_y = np.interp(
            removed_saccades_df["start_time"].to_numpy(dtype=float),
            df_work["Seconds"].to_numpy(dtype=float),
            df_work["X_raw"].to_numpy(dtype=float),
        )
        fig_overlay.add_trace(
            go.Scatter(
                x=removed_x,
                y=removed_y,
                mode="markers",
                name="amplitude-filtered threshold-crossing",
                marker=dict(
                    color="rgba(0,0,0,0)",
                    size=8,
                    symbol="circle",
                    line=dict(color="gray", width=1.5),
                ),
            ),
            row=1,
            col=1,
        )

    # Direction-colored event spans only on position panel (start_time->end_time).
    # Build all shapes first, then assign once (much faster than add_shape in a loop).
    if len(all_saccades_df) > 0:
        t0 = float(df_work["Seconds"].iloc[0])
        starts_rel = all_saccades_df["start_time"].to_numpy(dtype=float) - t0
        ends_rel = all_saccades_df["end_time"].to_numpy(dtype=float) - t0
        dirs = all_saccades_df["direction"].to_numpy()

        overlay_shapes = []
        for x0, x1, direction in zip(starts_rel, ends_rel, dirs):
            is_up = direction == "upward"
            overlay_shapes.append(
                dict(
                    type="rect",
                    x0=float(x0),
                    x1=float(x1),
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="y domain",
                    fillcolor="rgba(34,139,34,0.10)"
                    if is_up
                    else "rgba(128,0,128,0.10)",
                    line=dict(
                        color="rgba(34,139,34,0.25)"
                        if is_up
                        else "rgba(128,0,128,0.25)",
                        width=1,
                    ),
                )
            )
        fig_overlay.update_layout(shapes=overlay_shapes)

    fig_overlay.update_layout(
        title=f"Detection QC (global, fast) ({eye_with_least_low_confidence}, eye={selected_eye_code})",
        template="plotly_white",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    fig_overlay.update_xaxes(title_text="Relative time (s)", row=2, col=1)
    fig_overlay.update_yaxes(title_text="X position (px)", row=1, col=1)
    fig_overlay.update_yaxes(title_text="Velocity (px/s)", row=2, col=1)
    fig_overlay.show()

    # Note: regime-stratified local windows were removed by request.

    # 2) Distribution panel (single row, 4 columns):
    #    |peak velocity|, amplitude (kept + filtered-out), duration, ISI(log-x).
    if len(all_saccades_df) > 0:
        n_removed_local = (
            len(removed_saccades_df) if "removed_saccades_df" in globals() else 0
        )
        fig_dist = make_subplots(
            rows=1,
            cols=4,
            subplot_titles=(
                "Peak velocity distribution",
                "Amplitude distribution",
                "Duration distribution",
                "ISI distribution (log x-axis)",
            ),
            horizontal_spacing=0.08,
        )

        fig_dist.add_trace(
            go.Histogram(
                x=all_saccades_df["velocity"].abs(),
                nbinsx=60,
                name="|peak velocity| (kept)",
                marker_color="indianred",
                opacity=0.75,
            ),
            row=1,
            col=1,
        )
        fig_dist.add_trace(
            go.Histogram(
                x=all_saccades_df["amplitude"],
                nbinsx=80,
                name=f"Amplitude kept (n={len(all_saccades_df)})",
                marker_color="steelblue",
                opacity=0.6,
            ),
            row=1,
            col=2,
        )
        if n_removed_local > 0:
            fig_dist.add_trace(
                go.Histogram(
                    x=removed_saccades_df["amplitude"],
                    nbinsx=80,
                    name=f"Amplitude filtered out (n={n_removed_local})",
                    marker_color="indianred",
                    opacity=0.7,
                ),
                row=1,
                col=2,
            )
        fig_dist.add_vline(
            x=min_saccade_amplitude_px,
            line_dash="dash",
            line_color="black",
            row=1,
            col=2,
        )

        fig_dist.add_trace(
            go.Histogram(
                x=all_saccades_df["duration"],
                nbinsx=60,
                name="duration (kept)",
                marker_color="mediumpurple",
                opacity=0.75,
            ),
            row=1,
            col=3,
        )

        isi_kept = (
            np.diff(np.sort(all_saccades_df["time"].to_numpy()))
            if len(all_saccades_df) >= 2
            else np.array([])
        )
        isi_kept = isi_kept[isi_kept > 0]

        if len(isi_kept) > 0:
            isi_min = float(np.min(isi_kept))
            isi_max = float(np.max(isi_kept))
            if np.isclose(isi_min, isi_max):
                bins = np.array([isi_min * 0.8, isi_max * 1.2])
            else:
                bins = np.logspace(np.log10(isi_min), np.log10(isi_max), 80)

            counts_kept, edges_kept = np.histogram(isi_kept, bins=bins)
            centers_kept = np.sqrt(edges_kept[:-1] * edges_kept[1:])
            fig_dist.add_trace(
                go.Bar(
                    x=centers_kept,
                    y=counts_kept,
                    width=np.diff(edges_kept),
                    marker_color="darkcyan",
                    opacity=0.85,
                    name=f"ISI kept (n={len(isi_kept)})",
                ),
                row=1,
                col=4,
            )

            fig_dist.update_xaxes(type="log", row=1, col=4)
            fig_dist.update_xaxes(
                range=[np.log10(0.1), np.log10(max(isi_max, 0.1 * 1.01))],
                row=1,
                col=4,
            )
            fig_dist.add_vline(
                x=refractory_period_s,
                line_dash="dash",
                line_color="red",
                row=1,
                col=4,
            )
        else:
            print("ℹ️ ISI panel empty (need at least 2 kept detected saccades).")

        fig_dist.update_layout(
            template="plotly_white",
            height=380,
            width=1650,
            barmode="overlay",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0
            ),
        )
        fig_dist.update_xaxes(title_text="|peak velocity| (px/s)", row=1, col=1)
        fig_dist.update_xaxes(title_text="Amplitude (px)", row=1, col=2)
        fig_dist.update_xaxes(title_text="Duration (s)", row=1, col=3)
        fig_dist.update_xaxes(title_text="ISI (s, log scale)", row=1, col=4)
        fig_dist.update_yaxes(title_text="Count", row=1, col=1)
        fig_dist.update_yaxes(title_text="Count", row=1, col=2)
        fig_dist.update_yaxes(title_text="Count", row=1, col=3)
        fig_dist.update_yaxes(title_text="Count", row=1, col=4)
        fig_dist.show()
    else:
        print(
            "ℹ️ Distribution panel skipped (no detected saccades after amplitude filtering)."
        )


# %%
# Cell 9: Interactive saccade curation GUI
##########################################################################
import sys
from pathlib import Path as _Path

_sleap_dir = (
    str(_Path(__file__).resolve().parent / "sleap") if "__file__" in dir() else "sleap"
)
if _sleap_dir not in sys.path:
    sys.path.insert(0, str(_Path(_sleap_dir).resolve().parent))

from sleap.saccade_curation import build_curation_gui

curation_widget, curation_state = build_curation_gui(
    df_work=df_work,
    auto_events=all_saccades_df,
    vel_thresh=vel_thresh,
    cell2_params=cell2_params,
    metadata=metadata,
    metadata_path=metadata_path,
    save_dir=downsampled_output_dir,
    eye_label=f"({eye_with_least_low_confidence}, eye={selected_eye_code})",
)
display(curation_widget)

# %%
# Cell 10: Save curated events and metadata (click when curation + analysis are done)
import ipywidgets as _ipyw

_save_btn = _ipyw.Button(
    description="Save curated events & metadata",
    icon="save",
    button_style="success",
    layout=_ipyw.Layout(width="240px"),
)
_save_status = _ipyw.HTML(value="")


def _on_save_click(_):
    try:
        result = curation_state.save(
            downsampled_output_dir, metadata, metadata_path, cell2_params
        )
        _save_status.value = f"<span style='color:green'>✅ {result}</span>"
    except Exception as exc:
        _save_status.value = f"<span style='color:red'>❌ Save failed: {exc}</span>"


_save_btn.on_click(_on_save_click)
display(_ipyw.VBox([_save_btn, _save_status]))
