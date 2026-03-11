# ---
# jupyter:
#   jupytext:
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

# ### 1_4 Saccade classification
#
# - load curated saccade events csv and saccade snipets 1_3_saccade_detection
# - load motor velocity data_filter
# - classify saccades into look around, look-ahead, recentering and anomalous (these are opposite direction saccades during re-centering bouts)
# - manual curration
# - save for MM triggered analysis (check format to work with 2_1)
#

# +
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import binary_dilation

# +
## Cell 1 Data path and loading

debug = False

# ---------------------------------------------------------------------------
# Data path -- match the active session used in 1_3_saccade_detection
# ---------------------------------------------------------------------------
data_path = Path(
    "/Users/rancze/Documents/Data/vestVR/20250409_Cohort3_rotation/Visual_mismatch_day4/B6J2780-2025-04-28T13-10-18"
)

save_path = data_path.parent / f"{data_path.name}_processedData"
downsampled_dir = save_path / "downsampled_data"
classification_output_dir = save_path / "saccade_classification"
classification_output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load curated saccade events
# Index: aeon_time (naive datetime)
# ---------------------------------------------------------------------------
events_path = downsampled_dir / "curated_saccade_events.csv"
if not events_path.exists():
    raise FileNotFoundError(
        f"Curated saccade events not found at {events_path}. "
        "Run 1_3_saccade_detection.ipynb and complete curation first."
    )
saccade_events_df = pd.read_csv(events_path)
if "aeon_time" not in saccade_events_df.columns:
    raise ValueError(
        f"'aeon_time' column missing from {events_path.name}. "
        "Re-run 1_3_saccade_detection.ipynb with the updated curation code."
    )
saccade_events_df["aeon_time"] = pd.to_datetime(saccade_events_df["aeon_time"])
if saccade_events_df["aeon_time"].isna().all():
    raise ValueError(f"All aeon_time values are NaT in {events_path.name}.")
saccade_events_df = saccade_events_df.set_index("aeon_time")

# ---------------------------------------------------------------------------
# Load saccade snippets
# Index: aeon_time (naive datetime)
# ---------------------------------------------------------------------------
snippets_path = downsampled_dir / "curated_saccade_snippets.parquet"
if not snippets_path.exists():
    raise FileNotFoundError(
        f"Saccade snippets not found at {snippets_path}. "
        "Run 1_3_saccade_detection.ipynb and complete curation first."
    )
saccade_snippets_df = pd.read_parquet(snippets_path)
if "aeon_time" not in saccade_snippets_df.columns:
    raise ValueError(
        f"'aeon_time' column missing from {snippets_path.name}. "
        "Re-run 1_3_saccade_detection.ipynb with the updated curation code."
    )
saccade_snippets_df = saccade_snippets_df.set_index("aeon_time")

# ---------------------------------------------------------------------------
# Load motor velocity
# Index: native naive DatetimeIndex from the parquet file
# ---------------------------------------------------------------------------
motor_path = downsampled_dir / "photometry_tracking_encoder_data.parquet"
if not motor_path.exists():
    raise FileNotFoundError(
        f"Motor velocity data not found at {motor_path}. "
        "Run 1_1_Loading_and_Sync first."
    )
turning_df = pd.read_parquet(motor_path, columns=["Motor_Velocity", "Velocity_0Y"])

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(
    f"✅ Loaded {len(saccade_events_df)} curated saccade events from {events_path.name}"
)
print(f"   index: {saccade_events_df.index.min()} → {saccade_events_df.index.max()}")
print(
    f"✅ Loaded saccade snippets: {saccade_snippets_df.shape} "
    f"({saccade_snippets_df['event_idx'].nunique()} events)"
)
print(
    f"   index: {saccade_snippets_df.index.min()} → {saccade_snippets_df.index.max()}"
)
print(f"✅ Loaded motor velocity: {turning_df.shape}")
print(f"   index: {turning_df.index.min()} → {turning_df.index.max()}")
# print(f"ℹ️ Save dir: {classification_output_dir}")

if debug:
    display(saccade_events_df.head())
    display(saccade_snippets_df.head())
    display(turning_df.head())


# +
## Cell 2 QC plots if debug is True

# ---------------------------------------------------------------------------
# Three-row QC figure
#   Row 1: all saccade snippets (X_raw) over the recording timeline
#   Row 2: Motor_Velocity over the same timeline
#   Row 3: Velocity_0Y over the same timeline
# All datasets share a naive DatetimeIndex (aeon_time).
# x-axis is relative time in seconds from the earliest sample.
# ---------------------------------------------------------------------------

# Global t0: earliest timestamp across snippets and turning data
t0 = min(saccade_snippets_df.index.min(), turning_df.index.min())

if debug:
    # --- Colour map keyed on TNT_direction ---
    _colour_map = {"NT": "limegreen", "TN": "mediumpurple"}
    _tnt_by_event = (
        saccade_events_df["TNT_direction"].reindex(
            saccade_snippets_df["event_idx"].unique()
        )
        if "TNT_direction" in saccade_events_df.columns
        else pd.Series(dtype=str)
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Saccade snippets (X position)",
            "Motor velocity",
            "Velocity_0Y",
        ),
    )

    # --- Row 1: saccade snippets ---
    _shown_legend = set()
    for ev_idx, snip in saccade_snippets_df.groupby("event_idx"):
        tnt = _tnt_by_event.get(ev_idx, "") if len(_tnt_by_event) > 0 else ""
        colour = _colour_map.get(tnt, "steelblue")
        legend_key = tnt if tnt else "unknown"
        show_legend = legend_key not in _shown_legend
        _shown_legend.add(legend_key)

        fig.add_trace(
            go.Scatter(
                x=(snip.index - t0).total_seconds(),
                y=snip["X_raw"],
                mode="lines",
                line=dict(width=0.8, color=colour),
                name=legend_key,
                legendgroup=legend_key,
                showlegend=show_legend,
            ),
            row=1,
            col=1,
        )

    # Shared decimation index for rows 2 and 3 (same source DataFrame)
    n_motor = len(turning_df)
    _stride = max(1, int(np.ceil(n_motor / 30_000)))
    _idx = np.arange(0, n_motor, _stride)
    _time_rel = (turning_df.index[_idx] - t0).total_seconds()

    # --- Row 2: Motor_Velocity (decimated) ---
    fig.add_trace(
        go.Scatter(
            x=_time_rel,
            y=turning_df["Motor_Velocity"].to_numpy()[_idx],
            mode="lines",
            line=dict(width=1, color="firebrick"),
            name="Motor velocity",
        ),
        row=2,
        col=1,
    )

    # --- Row 3: Velocity_0Y (decimated) ---
    fig.add_trace(
        go.Scatter(
            x=_time_rel,
            y=turning_df["Velocity_0Y"].to_numpy()[_idx],
            mode="lines",
            line=dict(width=1, color="darkorange"),
            name="Velocity_0Y",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=900,
        title="Saccade QC: snippets, motor velocity & Velocity_0Y",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(title_text="Relative time (s)", row=3, col=1)
    fig.update_yaxes(title_text="X position (px)", row=1, col=1)
    fig.update_yaxes(title_text="Motor velocity (deg/s)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity_0Y (deg/s)", row=3, col=1)
    fig.show()


# +
## Cell 3 Movement bout detection and QC plots
# Movement bout detection via rolling RMS energy envelope + MAD threshold.
#
# Strategy:
#   1. Compute centered rolling RMS of each signal over window rms_window_s.
#      The RMS converts the oscillatory motor signal into a sustained envelope:
#      even when Motor_Velocity crosses zero mid-bout, the RMS stays elevated.
#   2. Threshold = median(RMS) + k * MAD(RMS)/0.6745  (robust, adapts per signal).
#   3. Morphological cleanup: remove/fill bouts shorter than min_bout_duration_s.
#   4. Cross-validation: discard turning bouts with no motor response within max_lag_s
#      (every real turn must cause motor rotation; unmatched turns are false positives).
#
# ── HOW TO FINE-TUNE ────────────────────────────────────────────────────────
#
#   SYMPTOM: motor rotation bouts are missed (rotation visible but not detected)
#     → decrease k_motor  (try steps of −0.5; current default 2.5)
#     → or increase rms_window_s to smooth over brief baseline oscillations
#       that raise the RMS noise floor (try 0.7–1.0 s)
#
#   SYMPTOM: too many false-positive motor bouts (baseline noise flagged as rotation)
#     → increase k_motor  (try steps of +0.5)
#     → or increase rms_window_s  (wider window averages out transient noise)
#
#   SYMPTOM: turning bouts over-detected (small movements flagged as turning)
#     → increase k_turning  (try steps of +0.5; current default 4.0)
#
#   SYMPTOM: real turning bouts missed
#     → decrease k_turning  (try steps of −0.5)
#
#   SYMPTOM: bouts are fragmented (one long bout split into several short ones)
#     → increase min_bout_duration_s to merge close fragments  (try +0.1 s steps)
#     → or increase rms_window_s so the RMS envelope stays elevated longer
#
#   SYMPTOM: turning bouts still detected without corresponding motor rotation
#      after cross-validation removes them
#     → first check k_motor is low enough to detect the missed motor bout
#     → if motor detection is correct but lag is too short, increase max_lag_s
#       (try 2.0 s); decrease it if unrelated later bouts are wrongly linking
#
#   DIAGNOSTIC FIGURE (run the cell):
#     Row 1 histograms — threshold line should sit in the valley between the
#       tall noise peak (left) and the low active-bout tail (right); if the
#       line is inside the noise peak, raise k; if it is past the valley, lower k.
#     Row 2 time series — dotted RMS envelope should visibly clear the solid
#       threshold line during every bout you want to capture.
#     Coloured shading — red = platform rotating, orange = animal turning.
#
# ── PARAMETERS ──────────────────────────────────────────────────────────────
rms_window_s = 1  # RMS integration window (s)
k_motor = (
    4.0  # MAD multiplier for Motor_Velocity  (lower → detects weaker rotation bouts)
)
k_turning = (
    4.5  # MAD multiplier for Velocity_0Y     (higher → fewer false-positive turns)
)
min_movement_bout_duration_s = (
    0.5  # minimum movement bout duration (s); shorter runs are removed
)
max_lag_s = (
    1.0  # cross-validation: max allowed delay between turn onset and motor response
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def rolling_rms(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """Centered rolling RMS energy envelope.

    Uses pandas rolling with center=True for symmetric smoothing.
    NaN-padded edges are filled with the signal's own RMS (conservative).
    """
    s = pd.Series(signal)
    rms = np.sqrt(
        s.pow(2)
        .rolling(window=window_samples, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    return rms


def mad_threshold(rms: np.ndarray, k: float) -> tuple:
    """Robust adaptive threshold from the RMS distribution.

    threshold = median(rms) + k * MAD(rms) / 0.6745

    Returns threshold value and the two components for diagnostics.
    """
    finite = rms[np.isfinite(rms)]
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    sigma_rms = mad / 0.6745
    thresh = med + k * sigma_rms
    return thresh, med, sigma_rms


def remove_short_runs(state: np.ndarray, min_samples: int) -> np.ndarray:
    """Remove contiguous True-runs (and False-gaps) shorter than min_samples.

    Two passes:
      1. Remove short True-runs (noise bursts mistaken for movement).
      2. Fill short False-gaps (very brief dropouts within a sustained active run).
    """
    state = state.copy()
    # Pass 1: remove short active runs
    i = 0
    while i < len(state):
        if state[i]:
            j = i
            while j < len(state) and state[j]:
                j += 1
            if (j - i) < min_samples:
                state[i:j] = False
            i = j
        else:
            i += 1
    # Pass 2: fill short inactive gaps inside an active run
    i = 0
    while i < len(state):
        if not state[i]:
            j = i
            while j < len(state) and not state[j]:
                j += 1
            if (j - i) < min_samples:
                state[i:j] = True
            i = j
        else:
            i += 1
    return state


# ---------------------------------------------------------------------------
# Estimate sampling interval
# ---------------------------------------------------------------------------
_dt_s = pd.Timedelta(turning_df.index.to_series().diff().median()).total_seconds()
rms_window_samples = max(3, int(round(rms_window_s / _dt_s)))
min_movement_bout_samples = max(1, int(round(min_movement_bout_duration_s / _dt_s)))

motor_vals = turning_df["Motor_Velocity"].to_numpy(dtype=float)
vel_vals = turning_df["Velocity_0Y"].to_numpy(dtype=float)

# ---------------------------------------------------------------------------
# Motor_Velocity — rolling RMS + MAD threshold
# ---------------------------------------------------------------------------
motor_rms = rolling_rms(motor_vals, rms_window_samples)
motor_thresh, motor_rms_med, motor_rms_sigma = mad_threshold(motor_rms, k_motor)

motor_state_raw = motor_rms > motor_thresh
motor_state = remove_short_runs(motor_state_raw, min_movement_bout_samples)
turning_df["is_platform_rotating"] = motor_state

# ---------------------------------------------------------------------------
# Velocity_0Y — rolling RMS + MAD threshold
# ---------------------------------------------------------------------------
vel_rms = rolling_rms(vel_vals, rms_window_samples)
vel_thresh, vel_rms_med, vel_rms_sigma = mad_threshold(vel_rms, k_turning)

vel_state_raw = vel_rms > vel_thresh
vel_state = remove_short_runs(vel_state_raw, min_movement_bout_samples)

# ---------------------------------------------------------------------------
# Cross-validation: discard turning bouts with no motor response within lag
# ---------------------------------------------------------------------------
# Every genuine turn must produce motor rotation within max_lag_s.
# Turning bouts that have no is_platform_rotating samples in the window
# [bout_start, bout_end + max_lag_s] are treated as false positives and
# removed from vel_state.  Motor detection is left unchanged.


def reject_unmatched_turns(
    turn_state: np.ndarray,
    motor_state: np.ndarray,
    max_lag_samples: int,
) -> tuple[np.ndarray, int]:
    """Remove turning bouts that lack any motor activity within max_lag_samples.

    For each contiguous True-run in turn_state, checks whether motor_state
    contains at least one True sample in [bout_start, bout_end + max_lag_samples].
    Unmatched bouts are set to False.

    Returns the cleaned state array and the count of rejected bouts.
    """
    state = turn_state.copy()
    n = len(state)
    rejected = 0
    i = 0
    while i < n:
        if state[i]:
            j = i
            while j < n and state[j]:
                j += 1
            # search window: [i, min(j + max_lag_samples, n))
            window_end = min(j + max_lag_samples, n)
            if not motor_state[i:window_end].any():
                state[i:j] = False
                rejected += 1
            i = j
        else:
            i += 1
    return state, rejected


max_lag_samples = max(1, int(round(max_lag_s / _dt_s)))
vel_state, _n_rejected = reject_unmatched_turns(vel_state, motor_state, max_lag_samples)
turning_df["is_animal_turning"] = vel_state

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
_session_s = len(turning_df) * _dt_s
print("── Motor_Velocity threshold (rolling RMS + MAD) ─────────────")
print(
    f"   RMS window      : {rms_window_s * 1000:.0f} ms  ({rms_window_samples} samples)"
)
print(f"   RMS median      : {motor_rms_med:.4f} deg/s")
print(f"   RMS sigma (MAD) : {motor_rms_sigma:.4f} deg/s")
print(f"   k_motor         : {k_motor}  →  threshold {motor_thresh:.4f} deg/s")
_rot_s = motor_state.sum() * _dt_s
print(
    f"   Rotating        : {_rot_s:.1f} s / {_session_s:.1f} s  "
    f"({100 * _rot_s / _session_s:.1f}%)"
)

print("\n── Velocity_0Y threshold (rolling RMS + MAD) ─────────────────")
print(
    f"   RMS window      : {rms_window_s * 1000:.0f} ms  ({rms_window_samples} samples)"
)
print(f"   RMS median      : {vel_rms_med:.4f} deg/s")
print(f"   RMS sigma (MAD) : {vel_rms_sigma:.4f} deg/s")
print(f"   k_turning       : {k_turning}  →  threshold {vel_thresh:.4f} deg/s")
_turn_s = vel_state.sum() * _dt_s
print(
    f"   Turning         : {_turn_s:.1f} s / {_session_s:.1f} s  "
    f"({100 * _turn_s / _session_s:.1f}%)"
)
print(
    f"   Rejected (no motor match) : {_n_rejected} turning run(s)  "
    f"[max_lag={max_lag_s * 1000:.0f} ms]"
)

print(
    f"\n   min_movement_bout_duration : {min_movement_bout_duration_s * 1000:.0f} ms  "
    f"({min_movement_bout_samples} samples)"
)

# ---------------------------------------------------------------------------
# Diagnostic figure
#   Row 1 (left/right) – RMS distributions with threshold lines (log y-scale)
#   Row 2 (colspan 2)  – raw signal time series, dual y-axes, RMS overlaid
#   Row 3 (colspan 2)  – saccade snippets (X_raw), coloured by TNT_direction
#   Rows 2 and 3 share the x-axis (same relative-time scale).
# ---------------------------------------------------------------------------

# Common t0: earliest sample across turning data and snippets
_t0 = min(turning_df.index[0], saccade_snippets_df.index.min())

# Full-resolution relative time (seconds) — used for accurate bout boundaries
_full_time_rel = (turning_df.index - _t0).total_seconds().to_numpy()

# Decimation for time series display
_n = len(turning_df)
_stride = max(1, int(np.ceil(_n / 50_000)))
_idx_d = np.arange(0, _n, _stride)
_time_rel = _full_time_rel[_idx_d]

# Row 1: two RMS histograms side by side.
# Row 2: time series panel (colspan 2), dual y-axes.
# Row 3: saccade snippets (colspan 2).
fig_thresh = make_subplots(
    rows=3,
    cols=2,
    shared_xaxes=False,  # rows 2 & 3 are linked manually via shared_xaxes arg below
    specs=[
        [{"secondary_y": False}, {"secondary_y": False}],
        [{"secondary_y": True, "colspan": 2}, None],
        [{"secondary_y": False, "colspan": 2}, None],
    ],
    row_heights=[0.20, 0.45, 0.35],
    subplot_titles=(
        "Motor_Velocity RMS distribution",
        "Velocity_0Y RMS distribution",
        "Motor_Velocity & Velocity_0Y time series (RMS envelope overlaid)",
        "Saccade snippets (X position, coloured by TNT direction)",
    ),
    vertical_spacing=0.08,
    horizontal_spacing=0.06,
)

# ── Row 1 col 1: Motor RMS histogram (log y) ──────────────────────────────
_motor_rms_finite = motor_rms[np.isfinite(motor_rms)]
_hist_counts, _hist_edges = np.histogram(_motor_rms_finite, bins=200)
fig_thresh.add_trace(
    go.Bar(
        x=0.5 * (_hist_edges[:-1] + _hist_edges[1:]),
        y=_hist_counts,
        marker_color="firebrick",
        opacity=0.6,
        name="Motor RMS",
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig_thresh.add_vline(
    x=motor_thresh,
    line_dash="solid",
    line_color="black",
    line_width=1.5,
    annotation_text=f"Thresh {motor_thresh:.2f}",
    annotation_position="top right",
    row=1,
    col=1,
)

# ── Row 1 col 2: Velocity_0Y RMS histogram (log y) ────────────────────────
_vel_rms_finite = vel_rms[np.isfinite(vel_rms)]
_hist_counts2, _hist_edges2 = np.histogram(_vel_rms_finite, bins=200)
fig_thresh.add_trace(
    go.Bar(
        x=0.5 * (_hist_edges2[:-1] + _hist_edges2[1:]),
        y=_hist_counts2,
        marker_color="darkorange",
        opacity=0.6,
        name="Vel_0Y RMS",
        showlegend=False,
    ),
    row=1,
    col=2,
)
fig_thresh.add_vline(
    x=vel_thresh,
    line_dash="solid",
    line_color="black",
    line_width=1.5,
    annotation_text=f"Thresh {vel_thresh:.2f}",
    annotation_position="top right",
    row=1,
    col=2,
)

# ── Row 2 (colspan 2): Motor_Velocity + RMS on primary y-axis ─────────────
fig_thresh.add_trace(
    go.Scatter(
        x=_time_rel,
        y=motor_vals[_idx_d],
        mode="lines",
        line=dict(width=0.6, color="firebrick"),
        opacity=0.5,
        name="Motor_Velocity",
    ),
    row=2,
    col=1,
    secondary_y=False,
)
fig_thresh.add_trace(
    go.Scatter(
        x=_time_rel,
        y=motor_rms[_idx_d],
        mode="lines",
        line=dict(width=1.2, color="firebrick", dash="dot"),
        name="Motor RMS",
    ),
    row=2,
    col=1,
    secondary_y=False,
)
fig_thresh.add_trace(
    go.Scatter(
        x=[_time_rel[0], _time_rel[-1]],
        y=[motor_thresh, motor_thresh],
        mode="lines",
        line=dict(color="firebrick", width=1.5, dash="solid"),
        showlegend=False,
        hoverinfo="skip",
    ),
    row=2,
    col=1,
    secondary_y=False,
)

# ── Row 2 (colspan 2): Velocity_0Y + RMS on secondary y-axis ──────────────
fig_thresh.add_trace(
    go.Scatter(
        x=_time_rel,
        y=vel_vals[_idx_d],
        mode="lines",
        line=dict(width=0.6, color="darkorange"),
        opacity=0.5,
        name="Velocity_0Y",
    ),
    row=2,
    col=1,
    secondary_y=True,
)
fig_thresh.add_trace(
    go.Scatter(
        x=_time_rel,
        y=vel_rms[_idx_d],
        mode="lines",
        line=dict(width=1.2, color="darkorange", dash="dot"),
        name="Vel_0Y RMS",
    ),
    row=2,
    col=1,
    secondary_y=True,
)
fig_thresh.add_trace(
    go.Scatter(
        x=[_time_rel[0], _time_rel[-1]],
        y=[vel_thresh, vel_thresh],
        mode="lines",
        line=dict(color="darkorange", width=1.5, dash="solid"),
        showlegend=False,
        hoverinfo="skip",
    ),
    row=2,
    col=1,
    secondary_y=True,
)

# ── Row 3 (colspan 2): saccade snippets coloured by TNT_direction ─────────
_colour_map_snip = {"NT": "limegreen", "TN": "mediumpurple"}
_tnt_by_event = (
    saccade_events_df["TNT_direction"].reindex(
        saccade_snippets_df["event_idx"].unique()
    )
    if "TNT_direction" in saccade_events_df.columns
    else pd.Series(dtype=str)
)
_shown_snip_legend = set()
for _ev_idx, _snip in saccade_snippets_df.groupby("event_idx"):
    _tnt = _tnt_by_event.get(_ev_idx, "") if len(_tnt_by_event) > 0 else ""
    _colour = _colour_map_snip.get(_tnt, "steelblue")
    _legend_key = _tnt if _tnt else "unknown"
    _show_leg = _legend_key not in _shown_snip_legend
    _shown_snip_legend.add(_legend_key)
    _snip_t = (_snip.index - _t0).total_seconds()
    fig_thresh.add_trace(
        go.Scatter(
            x=_snip_t,
            y=_snip["X_raw"],
            mode="lines",
            line=dict(width=0.8, color=_colour),
            name=_legend_key,
            legendgroup=_legend_key,
            showlegend=_show_leg,
        ),
        row=3,
        col=1,
    )

# ── State shading: vectorised run detection + single layout update ─────────
# Using np.diff on the full-resolution state finds exact run edges in one
# pass (no Python loop over samples).  All shapes are built as plain dicts
# and assigned to the layout in one call — O(N_runs) instead of O(N_runs²).


def _state_runs_to_shapes(
    state: np.ndarray, time_s: np.ndarray, fillcolor: str, xref: str, yref: str
) -> list:
    """Return shape dicts for every active run (True region) in *state*.

    Works on the full-resolution arrays so run start/end times are exact.
    No Python loop over individual samples.
    """
    # prepend=0 only → output length == len(state), so it indexes time_s cleanly.
    # append=0 was removed because it inflates the array to length N+1.
    # If the session ends while still active, len(starts) > len(ends);
    # use the last timestamp as the closing edge in that case.
    edges = np.diff(state.astype(np.int8), prepend=0)
    starts = time_s[edges == 1]
    ends = time_s[edges == -1]
    if len(starts) > len(ends):
        ends = np.append(ends, time_s[-1])
    return [
        dict(
            type="rect",
            xref=xref,
            yref=yref,
            x0=float(t0),
            x1=float(t1),
            y0=0,
            y1=1,
            fillcolor=fillcolor,
            line_width=0,
        )
        for t0, t1 in zip(starts, ends)
    ]


# Subplot reading order (ignoring secondary_y x-numbering):
#   (1,1)→x1  (1,2)→x2  (2,1)→x3  (3,1)→x4
# Link rows 2 and 3 on the same x-axis (matches/x3).
_ts_xref = "x3"
_ts_yref = "y3 domain"
_snip_xref = "x4"
_snip_yref = "y4 domain"

_all_shapes = (
    _state_runs_to_shapes(
        motor_state, _full_time_rel, "rgba(178,34,34,0.12)", _ts_xref, _ts_yref
    )
    + _state_runs_to_shapes(
        vel_state, _full_time_rel, "rgba(255,140,0,0.12)", _ts_xref, _ts_yref
    )
    + _state_runs_to_shapes(
        motor_state, _full_time_rel, "rgba(178,34,34,0.12)", _snip_xref, _snip_yref
    )
    + _state_runs_to_shapes(
        vel_state, _full_time_rel, "rgba(255,140,0,0.12)", _snip_xref, _snip_yref
    )
)

fig_thresh.update_layout(
    template="plotly_white",
    height=950,
    title=(
        f"Movement threshold QC (rolling RMS + MAD)  |  "
        f"Motor thresh={motor_thresh:.2f} deg/s  |  "
        f"Vel_0Y thresh={vel_thresh:.2f} deg/s  |  "
        f"k_motor={k_motor}, k_turning={k_turning}, window={rms_window_s * 1000:.0f} ms"
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    shapes=_all_shapes,
)
# Histogram axes
fig_thresh.update_xaxes(title_text="RMS (deg/s)", row=1, col=1)
fig_thresh.update_xaxes(title_text="RMS (deg/s)", row=1, col=2)
fig_thresh.update_yaxes(title_text="count", type="log", row=1, col=1)
fig_thresh.update_yaxes(title_text="count", type="log", row=1, col=2)
# Time series axes
fig_thresh.update_xaxes(title_text="", row=2, col=1, matches="x4")
fig_thresh.update_yaxes(
    title_text="Motor_Velocity (deg/s)",
    title_font=dict(color="firebrick"),
    tickfont=dict(color="firebrick"),
    secondary_y=False,
    row=2,
    col=1,
)
fig_thresh.update_yaxes(
    title_text="Velocity_0Y (deg/s)",
    title_font=dict(color="darkorange"),
    tickfont=dict(color="darkorange"),
    secondary_y=True,
    row=2,
    col=1,
)
# Snippet axes
fig_thresh.update_xaxes(title_text="Relative time (s)", row=3, col=1, matches="x3")
fig_thresh.update_yaxes(title_text="X position (px)", row=3, col=1)
fig_thresh.show()


# +
## Cell 4: ISI-based bout detection and saccade classification
#
# Strategy:
#   1. Compute inter-saccade interval (ISI) as the gap from the *end* of one
#      saccade to the *start* of the next, using aeon_start_time / aeon_end_time.
#   2. Link consecutive saccades where ISI <= max_isi_s to form candidate bouts.
#   3. Only runs with >= min_saccade_bout_size saccades become bouts; the rest are
#      look-around saccades.
#   4. Within each bout:
#        - Position 0 (first saccade) → look_ahead
#        - Positions 1..N: majority TNT_direction (among positions 1..N only)
#          determines the bout direction.  Same-direction → recentering.
#          Opposite direction → anomalous.
#        - Tie in majority: direction of position 1 wins.
#
# ── FUTURE PARAMETERS (not yet used) ────────────────────────────────────────
# Two additional features will later be incorporated to refine classification:
#   • Turning state  : is_animal_turning   (Velocity_0Y, Cell 3)
#     → identifies when the animal is self-locomoting (turning the ball)
#   • Rotation state : is_platform_rotating (Motor_Velocity, Cell 3)
#     → identifies when the platform is physically rotating the animal
#   Note: turning events drive rotation events and always precede them due to
#   hardware lag in the motor apparatus.
#
# ── PARAMETERS ──────────────────────────────────────────────────────────────
min_saccade_bout_size = 2  # minimum saccades to form a saccade bout (groups smaller than this → look_around)
max_isi_s = (
    6.0  # max gap (s) from end of saccade[i] to start of saccade[i+1] to be linked
)
turning_gap_tolerance_s = (
    0.5  # turning-state filter: max allowed gap (s) in is_animal_turning within a bout;
    # gaps longer than this split the bout
)

# ---------------------------------------------------------------------------
# Parse aeon_start_time / aeon_end_time columns (stored as ISO strings in CSV)
# ---------------------------------------------------------------------------
_ev = saccade_events_df.sort_index().copy()  # sort by aeon_time (peak time)

for _col in ("aeon_start_time", "aeon_end_time"):
    if _col in _ev.columns and not pd.api.types.is_datetime64_any_dtype(_ev[_col]):
        _ev[_col] = pd.to_datetime(_ev[_col], errors="coerce")

# ---------------------------------------------------------------------------
# ISI: gap from end[i] to start[i+1], vectorised
# ---------------------------------------------------------------------------
_starts = _ev["aeon_start_time"]
_ends = _ev["aeon_end_time"]

# isi_after[i]  = start[i+1] - end[i]   (NaN for last event)
# isi_before[i] = start[i]   - end[i-1] (NaN for first event)
_isi_after = (_starts.shift(-1) - _ends).dt.total_seconds()
_isi_before = (_starts - _ends.shift(1)).dt.total_seconds()

_ev["isi_before_s"] = _isi_before.to_numpy()
_ev["isi_after_s"] = _isi_after.to_numpy()

# ---------------------------------------------------------------------------
# Saccade bout detection: find runs of ISI-linked consecutive saccades
# ---------------------------------------------------------------------------
_n = len(_ev)
_isi_after_arr = _isi_after.to_numpy(dtype=float)

_saccade_bout_id = np.full(_n, -1, dtype=int)
_saccade_bout_pos = np.full(_n, -1, dtype=int)
_current_saccade_bout_id = 0

_i = 0
while _i < _n:
    # Check if saccade _i is linked forward (within threshold)
    if (
        _i < _n - 1
        and np.isfinite(_isi_after_arr[_i])
        and _isi_after_arr[_i] <= max_isi_s
    ):
        _run_start = _i
        _j = _i
        while (
            _j < _n - 1
            and np.isfinite(_isi_after_arr[_j])
            and _isi_after_arr[_j] <= max_isi_s
        ):
            _j += 1
        _run_end = _j  # inclusive
        _run_length = _run_end - _run_start + 1

        if _run_length >= min_saccade_bout_size:
            for _k in range(_run_start, _run_end + 1):
                _saccade_bout_id[_k] = _current_saccade_bout_id
                _saccade_bout_pos[_k] = _k - _run_start
            _current_saccade_bout_id += 1

        _i = _run_end + 1  # skip entire run regardless of threshold
    else:
        _i += 1

_ev["saccade_bout_id"] = _saccade_bout_id
_ev["saccade_bout_position"] = _saccade_bout_pos

# ---------------------------------------------------------------------------
# Turning-state filter: split saccade bouts at is_animal_turning gaps
#
# Strategy:
#   For each ISI-detected bout, query is_animal_turning across the full bout
#   time span.  Any contiguous False run longer than turning_gap_tolerance_s
#   is a split point.  The saccades on each side form candidate sub-groups.
#   Each sub-group is re-validated: must have >= min_saccade_bout_size members
#   with all ISIs still <= max_isi_s.  Valid sub-groups get new sequential
#   saccade_bout_id values; invalid ones are reset to -1 (→ look_around).
#
#   If is_animal_turning is absent from turning_df, the filter is skipped
#   and a warning is printed.
# ---------------------------------------------------------------------------

_n_bouts_split = 0  # bouts that were split by the filter
_n_subgroups_kept = 0  # sub-groups that passed re-validation
_n_saccades_reclassified = 0  # saccades demoted to look_around by the filter

if "is_animal_turning" not in turning_df.columns:
    print(
        "⚠️  is_animal_turning not found in turning_df — "
        "turning-state filter skipped. Run Cell 3 first."
    )
else:
    # Work on fresh mutable copies so we can rebuild from scratch
    _new_bout_id = np.full(_n, -1, dtype=int)
    _new_bout_pos = np.full(_n, -1, dtype=int)
    _next_bid = 0

    # Pre-extract the turning state as a pandas Series for fast .loc slicing
    _turning_series = turning_df["is_animal_turning"]

    # Pre-compute the median sample interval from the turning_df index
    # (_dt_s is already available from Cell 3; use it directly)
    _turning_gap_samples = max(1, int(round(turning_gap_tolerance_s / _dt_s)))

    def _find_turning_split_points(
        bout_start_time: pd.Timestamp,
        bout_end_time: pd.Timestamp,
    ) -> list[pd.Timestamp]:
        """Return timestamps at which long turning-OFF gaps occur within a bout span.

        Each returned timestamp is the first sample of a turning-OFF gap that
        exceeds turning_gap_tolerance_s.  These become split boundaries: saccades
        whose aeon_time is before the gap go to one sub-group, those after go to
        the next.
        """
        window = _turning_series.loc[bout_start_time:bout_end_time]
        if len(window) < 2:
            return []
        arr = window.to_numpy(dtype=bool)
        times = window.index
        split_times = []
        i = 0
        while i < len(arr):
            if not arr[i]:
                j = i
                while j < len(arr) and not arr[j]:
                    j += 1
                gap_len = j - i
                if gap_len > _turning_gap_samples:
                    # Record the start of this gap as the split boundary
                    split_times.append(times[i])
                i = j
            else:
                i += 1
        return split_times

    # Collect the set of positional indices for the current bouts (before filter)
    _pre_filter_bout_count = _current_saccade_bout_id

    for _bid in range(_pre_filter_bout_count):
        _bout_mask = _saccade_bout_id == _bid
        _bout_indices = np.where(_bout_mask)[0]  # positional indices into _ev
        if len(_bout_indices) == 0:
            continue

        # Bout time span
        _b_start = _ev["aeon_start_time"].iloc[_bout_indices[0]]
        _b_end = _ev["aeon_end_time"].iloc[_bout_indices[-1]]

        if pd.isna(_b_start) or pd.isna(_b_end):
            # Can't check turning without valid timestamps — keep bout as-is
            for _k, _pos_idx in enumerate(_bout_indices):
                _new_bout_id[_pos_idx] = _next_bid
                _new_bout_pos[_pos_idx] = _k
            _next_bid += 1
            _n_subgroups_kept += 1
            continue

        _split_times = _find_turning_split_points(_b_start, _b_end)

        if not _split_times:
            # No long gap — bout passes as-is
            for _k, _pos_idx in enumerate(_bout_indices):
                _new_bout_id[_pos_idx] = _next_bid
                _new_bout_pos[_pos_idx] = _k
            _next_bid += 1
            _n_subgroups_kept += 1
            continue

        # Bout must be split — build sub-groups by assigning each saccade
        # to the turning-on region its aeon_time falls in.
        _n_bouts_split += 1

        # Build sorted boundary list: include recording start sentinel and
        # each split point.  Between split_times[i] and split_times[i+1] is
        # one continuous turning-on region (or the end of the bout).
        # We assign a saccade to a sub-group based on the gap AFTER which it
        # falls — i.e. saccades before the first split go to sub-group 0,
        # saccades between split[0] and split[1] go to sub-group 1, etc.
        _boundaries = _split_times  # already sorted (temporal order)

        _subgroups: list[list[int]] = [[]]  # list of positional-index lists
        for _pos_idx in _bout_indices:
            _sac_time = _ev.index[_pos_idx]  # aeon_time (index)
            # Find which sub-group this saccade belongs to
            _sg = 0
            for _bt in _boundaries:
                if _sac_time >= _bt:
                    _sg += 1
                else:
                    break
            # Grow subgroups list if needed
            while len(_subgroups) <= _sg:
                _subgroups.append([])
            _subgroups[_sg].append(_pos_idx)

        # Validate each sub-group and assign IDs
        for _sg_indices in _subgroups:
            if len(_sg_indices) == 0:
                continue
            # ISI check: all consecutive gaps within sub-group must be <= max_isi_s
            _sg_isi_ok = True
            for _ki in range(len(_sg_indices) - 1):
                _isi_val = _isi_after_arr[_sg_indices[_ki]]
                if not np.isfinite(_isi_val) or _isi_val > max_isi_s:
                    _sg_isi_ok = False
                    break

            if len(_sg_indices) >= min_saccade_bout_size and _sg_isi_ok:
                for _k, _pos_idx in enumerate(_sg_indices):
                    _new_bout_id[_pos_idx] = _next_bid
                    _new_bout_pos[_pos_idx] = _k
                _next_bid += 1
                _n_subgroups_kept += 1
            else:
                # Sub-group too small or ISI broken — demote to look_around
                _n_saccades_reclassified += len(_sg_indices)

    # Write filtered arrays back
    _saccade_bout_id = _new_bout_id
    _saccade_bout_pos = _new_bout_pos
    _current_saccade_bout_id = _next_bid

    # Update dataframe columns
    _ev["saccade_bout_id"] = _saccade_bout_id
    _ev["saccade_bout_position"] = _saccade_bout_pos

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
_classes = ["look_around"] * _n
_tnt = (
    _ev["TNT_direction"].to_numpy(dtype=str)
    if "TNT_direction" in _ev.columns
    else np.full(_n, "")
)

for _bid in range(_current_saccade_bout_id):
    _mask = _saccade_bout_id == _bid
    _indices = np.where(_mask)[0]

    # First saccade in saccade bout → look_ahead
    _classes[_indices[0]] = "look_ahead"

    if len(_indices) < 2:
        continue

    # Majority direction among positions 1..N (exclude look-ahead from vote)
    _rest = _indices[1:]
    _rest_tnt = _tnt[_rest]
    _nt_count = int(np.sum(_rest_tnt == "NT"))
    _tn_count = int(np.sum(_rest_tnt == "TN"))

    if _nt_count > _tn_count:
        _majority = "NT"
    elif _tn_count > _nt_count:
        _majority = "TN"
    else:
        # Tie: direction of position 1 (first recentering candidate) decides
        _majority = str(_rest_tnt[0]) if len(_rest_tnt) > 0 else "NT"

    for _ri in _rest:
        _classes[_ri] = "recentering" if _tnt[_ri] == _majority else "anomalous"

# ---------------------------------------------------------------------------
# Non-turning override: force saccades at non-turning times to look_around
#
# A saccade is overridden when is_animal_turning is False for the entire
# 0.5-second window immediately following the saccade peak (aeon_time).
# Using a post-saccade window (rather than a single point) avoids false
# overrides caused by momentary turning-state transitions at the saccade onset.
#
# Skipped silently if is_animal_turning is absent from turning_df.
# ---------------------------------------------------------------------------

_n_non_turning_overridden = 0
_non_turning_window_s = 0.5  # seconds after saccade peak to evaluate turning state

if "is_animal_turning" in turning_df.columns:
    _turning_series = turning_df["is_animal_turning"]
    for _nt_i, _t_peak in enumerate(_ev.index):
        _t_end = _t_peak + pd.Timedelta(seconds=_non_turning_window_s)
        _window = _turning_series.loc[_t_peak:_t_end]
        # Override only when the animal was not turning at all during the window
        if len(_window) > 0 and not _window.any():
            if _classes[_nt_i] != "look_around":
                _classes[_nt_i] = "look_around"
                _saccade_bout_id[_nt_i] = -1
                _saccade_bout_pos[_nt_i] = -1
                _n_non_turning_overridden += 1
    # Update dataframe columns with the corrected bout arrays
    _ev["saccade_bout_id"] = _saccade_bout_id
    _ev["saccade_bout_position"] = _saccade_bout_pos

# ---------------------------------------------------------------------------
# Post-override re-validation and re-classification
#
# The non-turning override may have removed saccades from bouts, which can:
#   (a) leave a bout without a look-ahead (its position-0 saccade was removed)
#   (b) shrink a bout below min_saccade_bout_size
#
# This pass iterates over every surviving bout, dissolves undersized ones,
# re-numbers positions, and re-assigns look_ahead / recentering / anomalous
# using the same majority-direction logic as the initial classification.
# ---------------------------------------------------------------------------

_n_bouts_dissolved_post = 0
_n_lookahead_reassigned = 0

for _bid in sorted(set(_saccade_bout_id[_saccade_bout_id >= 0])):
    _indices = np.where(_saccade_bout_id == _bid)[0]
    # Sort by temporal order (aeon_time index is already sorted, but be safe)
    _indices = _indices[np.argsort(_indices)]

    # Dissolve bouts that are now too small
    if len(_indices) < min_saccade_bout_size:
        for _di in _indices:
            _classes[_di] = "look_around"
            _saccade_bout_id[_di] = -1
            _saccade_bout_pos[_di] = -1
        _n_bouts_dissolved_post += 1
        continue

    # Re-number positions contiguously (0, 1, 2, ...)
    for _new_pos, _idx in enumerate(_indices):
        _saccade_bout_pos[_idx] = _new_pos

    # Re-classify: position 0 is look_ahead
    _prev_class_0 = _classes[_indices[0]]
    _classes[_indices[0]] = "look_ahead"
    if _prev_class_0 != "look_ahead":
        _n_lookahead_reassigned += 1

    if len(_indices) < 2:
        continue

    # Majority direction of positions 1..N (same logic as the initial pass)
    _rest_idx = _indices[1:]
    _rest_tnt_vals = _tnt[_rest_idx]
    _nt_count = int(np.sum(_rest_tnt_vals == "NT"))
    _tn_count = int(np.sum(_rest_tnt_vals == "TN"))
    if _nt_count > _tn_count:
        _majority = "NT"
    elif _tn_count > _nt_count:
        _majority = "TN"
    else:
        _majority = str(_rest_tnt_vals[0]) if len(_rest_tnt_vals) > 0 else "NT"

    for _ri in _rest_idx:
        _classes[_ri] = "recentering" if _tnt[_ri] == _majority else "anomalous"

_ev["saccade_bout_id"] = _saccade_bout_id
_ev["saccade_bout_position"] = _saccade_bout_pos

_ev["saccade_class"] = _classes

# Write results back to the main dataframe
saccade_events_df = _ev

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
_class_counts = saccade_events_df["saccade_class"].value_counts()
_n_saccade_bouts = _current_saccade_bout_id
_saccade_bout_sizes = (
    saccade_events_df[saccade_events_df["saccade_bout_id"] >= 0]
    .groupby("saccade_bout_id")
    .size()
)

print(f"── Saccade classification ──────────────────────────────────────────")
print(
    f"   Parameters : min_saccade_bout_size={min_saccade_bout_size}, max_isi_s={max_isi_s} s"
)
print(f"   Total events : {_n}")
print(f"   Saccade bouts detected : {_n_saccade_bouts}")
if len(_saccade_bout_sizes) > 0:
    print(
        f"   Saccade bout size : mean={_saccade_bout_sizes.mean():.1f}, "
        f"median={_saccade_bout_sizes.median():.0f}, "
        f"min={_saccade_bout_sizes.min()}, max={_saccade_bout_sizes.max()}"
    )
if "is_animal_turning" in turning_df.columns:
    print(
        f"   Turning-state filter  : {_n_bouts_split} bout(s) split  |  "
        f"{_n_subgroups_kept} sub-group(s) kept  |  "
        f"{_n_saccades_reclassified} saccade(s) reclassified as look_around  "
        f"[gap_tolerance={turning_gap_tolerance_s * 1000:.0f} ms]"
    )
    print(
        f"   Non-turning override  : {_n_non_turning_overridden} saccade(s) forced to look_around"
    )
    print(
        f"   Post-override reclass : {_n_bouts_dissolved_post} bout(s) dissolved  |  "
        f"{_n_lookahead_reassigned} look-ahead(s) reassigned"
    )
print()
for _cls in ["look_around", "look_ahead", "recentering", "anomalous"]:
    _cnt = int(_class_counts.get(_cls, 0))
    print(f"   {_cls:<14} : {_cnt:>4}  ({100.0 * _cnt / _n:.1f}%)")

if debug:
    display(
        saccade_events_df[
            [
                "TNT_direction",
                "isi_before_s",
                "isi_after_s",
                "saccade_bout_id",
                "saccade_bout_position",
                "saccade_class",
            ]
        ].head(20)
    )


# +
## Cell 5: Classification QC visualization
#
# Three figures:
#   Figure A – ISI distribution (log x) with bout threshold and parameter info.
#   Figure B – Timeline overview (3 rows):
#                Row 1: saccade snippets coloured by saccade_class, with shaded
#                        rectangles marking each detected bout.
#                Row 2: Motor_Velocity with is_platform_rotating shading.
#                Row 3: Velocity_0Y with is_animal_turning shading.
#   Figure C – Per-bout summary table printed to console.

# Colour map for the five classes (no_saccade added for forward compat)
_CLASS_COLOURS = {
    "look_around": "steelblue",
    "look_ahead": "gold",
    "recentering": "limegreen",
    "anomalous": "crimson",
    "no_saccade": "lightgray",
}

# ---------------------------------------------------------------------------
# Figure A: ISI distribution with threshold line
# ---------------------------------------------------------------------------
_isi_all = saccade_events_df["isi_after_s"].dropna().to_numpy()
_isi_all = _isi_all[_isi_all > 0]

if len(_isi_all) > 0:
    _isi_min = float(np.min(_isi_all))
    _isi_max = float(np.max(_isi_all))
    if np.isclose(_isi_min, _isi_max):
        _isi_bins = np.array([_isi_min * 0.8, _isi_max * 1.2])
    else:
        _isi_bins = np.logspace(np.log10(_isi_min), np.log10(_isi_max), 80)
    _isi_counts, _isi_edges = np.histogram(_isi_all, bins=_isi_bins)
    _isi_centers = np.sqrt(_isi_edges[:-1] * _isi_edges[1:])

    fig_isi = go.Figure()
    fig_isi.add_trace(
        go.Bar(
            x=_isi_centers,
            y=_isi_counts,
            width=np.diff(_isi_edges),
            marker_color="darkcyan",
            opacity=0.85,
            name=f"ISI (n={len(_isi_all)})",
        )
    )
    fig_isi.add_vline(
        x=max_isi_s,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text=f"max_isi_s = {max_isi_s} s",
        annotation_position="top right",
    )
    fig_isi.update_xaxes(type="log", title_text="ISI (s, end→start, log scale)")
    fig_isi.update_yaxes(title_text="Count")
    fig_isi.update_layout(
        template="plotly_white",
        height=360,
        title=(
            f"ISI distribution (end→start)  |  "
            f"max_isi_s={max_isi_s} s, min_saccade_bout_size={min_saccade_bout_size}  |  "
            f"{_current_saccade_bout_id} saccade bouts detected"
        ),
    )
    fig_isi.show()
else:
    print("ℹ️ ISI figure skipped: no finite ISI values available.")

# ---------------------------------------------------------------------------
# Figure B: 3-row timeline
# ---------------------------------------------------------------------------
_t0_cls = min(saccade_snippets_df.index.min(), turning_df.index.min())

fig_cls = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=(
        "Saccade snippets (X position) — coloured by class, bout shading",
        "Motor velocity (platform rotation)",
        "Velocity_0Y (animal turning)",
    ),
)

# ── Row 1: snippets coloured by saccade_class ────────────────────────────
# Only the peri-saccade window is shown: time_rel in [-0.5, +1.0] s
# (= positions 4.5–6.0 s inside the 10 s snippet).
# Render order: recentering first (bottom), then look_around, look_ahead,
# anomalous last (top), so higher-priority classes are never hidden.
_SNIP_REL_LO = -0.5  # seconds relative to saccade peak
_SNIP_REL_HI = 1.0

# Build lookup: event_idx → saccade_class
_ev_sorted = saccade_events_df.sort_index()
_class_by_event = {}
for _row in saccade_snippets_df["event_idx"].unique():
    _class_by_event[_row] = (
        _ev_sorted["saccade_class"].iloc[_row]
        if _row < len(_ev_sorted)
        else "look_around"
    )

# Group snippets by class for ordered rendering
_snips_by_class: dict[str, list] = {c: [] for c in _CLASS_COLOURS}
for _ev_idx, _snip in saccade_snippets_df.groupby("event_idx"):
    _cls = _class_by_event.get(_ev_idx, "look_around")
    # Filter to peri-saccade window using time_rel column
    if "time_rel" in _snip.columns:
        _snip = _snip[
            (_snip["time_rel"] >= _SNIP_REL_LO) & (_snip["time_rel"] <= _SNIP_REL_HI)
        ]
    _snips_by_class.setdefault(_cls, []).append((_ev_idx, _snip))

_LAYER_ORDER = ["recentering", "look_around", "look_ahead", "anomalous", "no_saccade"]
_shown_cls_legend: set = set()
for _cls in _LAYER_ORDER:
    for _ev_idx, _snip in _snips_by_class.get(_cls, []):
        if len(_snip) == 0:
            continue
        _colour = _CLASS_COLOURS.get(_cls, "steelblue")
        _show_leg = _cls not in _shown_cls_legend
        _shown_cls_legend.add(_cls)
        fig_cls.add_trace(
            go.Scatter(
                x=(_snip.index - _t0_cls).total_seconds(),
                y=_snip["X_raw"],
                mode="lines",
                line=dict(width=0.8, color=_colour),
                name=_cls,
                legendgroup=_cls,
                showlegend=_show_leg,
            ),
            row=1,
            col=1,
        )

# ── Row 2: Motor_Velocity (decimated) ────────────────────────────────────
_n_mot = len(turning_df)
_mot_stride = max(1, int(np.ceil(_n_mot / 30_000)))
_mot_idx = np.arange(0, _n_mot, _mot_stride)
_mot_t_rel = (turning_df.index[_mot_idx] - _t0_cls).total_seconds()

fig_cls.add_trace(
    go.Scatter(
        x=_mot_t_rel,
        y=turning_df["Motor_Velocity"].to_numpy()[_mot_idx],
        mode="lines",
        line=dict(width=1, color="firebrick"),
        name="Motor_Velocity",
    ),
    row=2,
    col=1,
)

# ── Row 3: Velocity_0Y (decimated) ───────────────────────────────────────
fig_cls.add_trace(
    go.Scatter(
        x=_mot_t_rel,
        y=turning_df["Velocity_0Y"].to_numpy()[_mot_idx],
        mode="lines",
        line=dict(width=1, color="darkorange"),
        name="Velocity_0Y",
    ),
    row=3,
    col=1,
)

# ── Shading: bouts (Row 1), rotation (Rows 2-3), turning (Rows 2-3) ──────
_full_t_rel = (turning_df.index - _t0_cls).total_seconds().to_numpy()

# Saccade bout shading on Row 1
# Subplot x-axis refs: row1→x, row2→x2, row3→x3 (shared)
_all_cls_shapes = []

# Saccade bout rectangles on snippet panel
_ev_sorted_cls = saccade_events_df.sort_index()
for _bid in range(_current_saccade_bout_id):
    _saccade_bout_rows = _ev_sorted_cls[_ev_sorted_cls["saccade_bout_id"] == _bid]
    if len(_saccade_bout_rows) == 0:
        continue
    _t_start_saccade_bout = (
        _saccade_bout_rows["aeon_start_time"].min() - _t0_cls
    ).total_seconds()
    _t_end_saccade_bout = (
        _saccade_bout_rows["aeon_end_time"].max() - _t0_cls
    ).total_seconds()
    _all_cls_shapes.append(
        dict(
            type="rect",
            xref="x",
            yref="y domain",
            x0=float(_t_start_saccade_bout),
            x1=float(_t_end_saccade_bout),
            y0=0,
            y1=1,
            fillcolor="rgba(200,200,200,0.18)",
            line=dict(color="rgba(100,100,100,0.35)", width=1, dash="dot"),
        )
    )


def _state_to_shapes(
    state_arr: np.ndarray,
    time_arr: np.ndarray,
    fillcolor: str,
    xref: str,
    yref: str,
) -> list:
    """Convert a boolean state array into rect shapes for shading."""
    edges = np.diff(state_arr.astype(np.int8), prepend=0)
    starts = time_arr[edges == 1]
    ends = time_arr[edges == -1]
    if len(starts) > len(ends):
        ends = np.append(ends, time_arr[-1])
    return [
        dict(
            type="rect",
            xref=xref,
            yref=yref,
            x0=float(s),
            x1=float(e),
            y0=0,
            y1=1,
            fillcolor=fillcolor,
            line_width=0,
        )
        for s, e in zip(starts, ends)
    ]


# Rotation and turning shading on rows 2 and 3
if "is_platform_rotating" in turning_df.columns:
    _all_cls_shapes += _state_to_shapes(
        turning_df["is_platform_rotating"].to_numpy(),
        _full_t_rel,
        "rgba(178,34,34,0.12)",
        "x2",
        "y2 domain",
    )
if "is_animal_turning" in turning_df.columns:
    _all_cls_shapes += _state_to_shapes(
        turning_df["is_animal_turning"].to_numpy(),
        _full_t_rel,
        "rgba(255,140,0,0.12)",
        "x2",
        "y2 domain",
    )
if "is_platform_rotating" in turning_df.columns:
    _all_cls_shapes += _state_to_shapes(
        turning_df["is_platform_rotating"].to_numpy(),
        _full_t_rel,
        "rgba(178,34,34,0.12)",
        "x3",
        "y3 domain",
    )
if "is_animal_turning" in turning_df.columns:
    _all_cls_shapes += _state_to_shapes(
        turning_df["is_animal_turning"].to_numpy(),
        _full_t_rel,
        "rgba(255,140,0,0.12)",
        "x3",
        "y3 domain",
    )

fig_cls.update_layout(
    template="plotly_white",
    height=900,
    title=(
        f"Classification overview  |  "
        f"max_isi_s={max_isi_s} s, min_saccade_bout_size={min_saccade_bout_size}  |  "
        f"{_current_saccade_bout_id} saccade bouts"
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    shapes=_all_cls_shapes,
)
fig_cls.update_xaxes(title_text="Relative time (s)", row=3, col=1)
fig_cls.update_yaxes(title_text="X position (px)", row=1, col=1)
fig_cls.update_yaxes(title_text="Motor velocity (deg/s)", row=2, col=1)
fig_cls.update_yaxes(title_text="Velocity_0Y (deg/s)", row=3, col=1)
fig_cls.show()

# ---------------------------------------------------------------------------
# Figure C: per-saccade-bout summary (console)
# ---------------------------------------------------------------------------
print("\n── Per-saccade-bout summary ────────────────────────────────────────")
print(
    f"{'Bout':>5}  {'Size':>5}  {'Maj.dir':>8}  {'L-ahead':>7}  {'Rectr':>6}  {'Anom':>5}"
)
_ev_sorted_cls = saccade_events_df.sort_index()
for _bid in range(_current_saccade_bout_id):
    _bdf = _ev_sorted_cls[_ev_sorted_cls["saccade_bout_id"] == _bid]
    _size = len(_bdf)
    _tnt_vals = (
        _bdf["TNT_direction"].to_numpy(dtype=str)
        if "TNT_direction" in _bdf.columns
        else np.array([])
    )
    _rest_tnt = _tnt_vals[1:] if len(_tnt_vals) > 1 else np.array([])
    _nt = int(np.sum(_rest_tnt == "NT"))
    _tn = int(np.sum(_rest_tnt == "TN"))
    _maj = "NT" if _nt > _tn else ("TN" if _tn > _nt else "tie")
    _n_la = int((_bdf["saccade_class"] == "look_ahead").sum())
    _n_rc = int((_bdf["saccade_class"] == "recentering").sum())
    _n_an = int((_bdf["saccade_class"] == "anomalous").sum())
    print(f"  {_bid:>4}  {_size:>5}  {_maj:>8}  {_n_la:>7}  {_n_rc:>6}  {_n_an:>5}")
print()


# +
## Cell 6: Manual classification curation GUI
#
# Loads the automatic classifications from Cell 4 into an interactive
# 3-panel viewer.  Navigate events with W/Q, jump between bouts with J/K,
# and override any class with keys 1–5 or the buttons.
#
# Keyboard shortcuts:
#   W / Q         — next / previous event
#   J / K         — previous / next bout (jumps to first event of that bout)
#   1             — look_around
#   2             — look_ahead
#   3             — recentering
#   4             — anomalous
#   5             — no_saccade
#
# The window shows ±5 s centred on the saccade peak:
#   Row 1  X position (snippet), coloured by current class, grey rect = bout span
#   Row 2  Motor_Velocity with rotation shading (red)
#   Row 3  Velocity_0Y   with turning shading   (orange)

import sys as _sys
from pathlib import Path as _Path

_sleap_dir = (
    str(_Path(__file__).resolve().parent / "sleap") if "__file__" in dir() else "sleap"
)
if _sleap_dir not in _sys.path:
    _sys.path.insert(0, str(_Path(_sleap_dir).resolve().parent))

from sleap.saccade_classification_gui import build_classification_gui

# Pass the Cell 4 classification parameters so they are persisted on save.
_cls_params = {
    "min_saccade_bout_size": min_saccade_bout_size,
    "max_isi_s": max_isi_s,
    "turning_gap_tolerance_s": turning_gap_tolerance_s,
}

classification_widget, classification_state = build_classification_gui(
    saccade_events_df=saccade_events_df,
    saccade_snippets_df=saccade_snippets_df,
    turning_df=turning_df,
    t0=t0,
    classification_params=_cls_params,
)
display(classification_widget)


# +
## Cell 7: Save classified events and update metadata
#
# Click the button after completing curation in Cell 6.
# Writes:
#   classification_output_dir / classified_saccade_events.csv
#       All original columns from curated_saccade_events.csv plus:
#       saccade_class, saccade_bout_id, saccade_bout_position, isi_before_s, isi_after_s.
#       Events marked no_saccade are retained (filtered downstream).
#   downsampled_dir / saccade_input_metadata.json  (updated in-place)
#       Adds classification_parameters and classification_summary keys.

import ipywidgets as _ipyw

_save_btn = _ipyw.Button(
    description="Save classified events",
    icon="save",
    button_style="success",
    layout=_ipyw.Layout(width="220px"),
)
_save_status = _ipyw.HTML(value="")


def _on_save_click(_):
    try:
        # Load metadata from the downsampled_dir JSON written by 1_3
        _meta_path = downsampled_dir / "saccade_input_metadata.json"
        if _meta_path.exists():
            with open(_meta_path, "r") as _fh:
                _meta = json.load(_fh)
        else:
            _meta = {}

        result = classification_state.save(
            save_dir=classification_output_dir,
            metadata=_meta,
            metadata_path=_meta_path,
            classification_params=_cls_params,
        )
        _save_status.value = f"<span style='color:green'>✅ {result}</span>"
    except Exception as exc:
        _save_status.value = f"<span style='color:red'>❌ Save failed: {exc}</span>"


_save_btn.on_click(_on_save_click)
display(_ipyw.VBox([_save_btn, _save_status]))
print("Can take a few seconds — be patient.")
