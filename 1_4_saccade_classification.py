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
## Cell 1

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
print(f"ℹ️ Save dir: {classification_output_dir}")

if debug:
    display(saccade_events_df.head())
    display(saccade_snippets_df.head())
    display(turning_df.head())


# +
## Cell 2

# ---------------------------------------------------------------------------
# Table: head of curated saccade events
# ---------------------------------------------------------------------------
display(saccade_events_df.head(10))

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
    subplot_titles=("Saccade snippets (X position)", "Motor velocity", "Velocity_0Y"),
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
## Cell 3
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
rms_window_s        = 0.5  # RMS integration window (s)
k_motor             = 2.5  # MAD multiplier for Motor_Velocity  (lower → detects weaker rotation bouts)
k_turning           = 4.0  # MAD multiplier for Velocity_0Y     (higher → fewer false-positive turns)
min_bout_duration_s = 0.3  # minimum bout duration (s); shorter bouts are removed
max_lag_s           = 1.5  # cross-validation: max allowed delay between turn onset and motor response

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def rolling_rms(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """Centered rolling RMS energy envelope.

    Uses pandas rolling with center=True for symmetric smoothing.
    NaN-padded edges are filled with the signal's own RMS (conservative).
    """
    s = pd.Series(signal)
    rms = np.sqrt(s.pow(2).rolling(window=window_samples, center=True, min_periods=1).mean().to_numpy())
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


def remove_short_bouts(state: np.ndarray, min_samples: int) -> np.ndarray:
    """Remove contiguous True-runs (and False-gaps) shorter than min_samples.

    Two passes:
      1. Remove short True-bouts (noise bursts mistaken for movement).
      2. Remove short False-gaps (very brief dropouts within a sustained bout).
    """
    state = state.copy()
    # Pass 1: remove short active bouts
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
    # Pass 2: fill short inactive gaps inside a movement bout
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
rms_window_samples  = max(3, int(round(rms_window_s / _dt_s)))
min_bout_samples    = max(1, int(round(min_bout_duration_s / _dt_s)))

motor_vals = turning_df["Motor_Velocity"].to_numpy(dtype=float)
vel_vals   = turning_df["Velocity_0Y"].to_numpy(dtype=float)

# ---------------------------------------------------------------------------
# Motor_Velocity — rolling RMS + MAD threshold
# ---------------------------------------------------------------------------
motor_rms = rolling_rms(motor_vals, rms_window_samples)
motor_thresh, motor_rms_med, motor_rms_sigma = mad_threshold(motor_rms, k_motor)

motor_state_raw = motor_rms > motor_thresh
motor_state = remove_short_bouts(motor_state_raw, min_bout_samples)
turning_df["is_platform_rotating"] = motor_state

# ---------------------------------------------------------------------------
# Velocity_0Y — rolling RMS + MAD threshold
# ---------------------------------------------------------------------------
vel_rms = rolling_rms(vel_vals, rms_window_samples)
vel_thresh, vel_rms_med, vel_rms_sigma = mad_threshold(vel_rms, k_turning)

vel_state_raw = vel_rms > vel_thresh
vel_state = remove_short_bouts(vel_state_raw, min_bout_samples)

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
print(f"   RMS window      : {rms_window_s*1000:.0f} ms  ({rms_window_samples} samples)")
print(f"   RMS median      : {motor_rms_med:.4f} deg/s")
print(f"   RMS sigma (MAD) : {motor_rms_sigma:.4f} deg/s")
print(f"   k_motor         : {k_motor}  →  threshold {motor_thresh:.4f} deg/s")
_rot_s = motor_state.sum() * _dt_s
print(f"   Rotating        : {_rot_s:.1f} s / {_session_s:.1f} s  "
      f"({100*_rot_s/_session_s:.1f}%)")

print("\n── Velocity_0Y threshold (rolling RMS + MAD) ─────────────────")
print(f"   RMS window      : {rms_window_s*1000:.0f} ms  ({rms_window_samples} samples)")
print(f"   RMS median      : {vel_rms_med:.4f} deg/s")
print(f"   RMS sigma (MAD) : {vel_rms_sigma:.4f} deg/s")
print(f"   k_turning       : {k_turning}  →  threshold {vel_thresh:.4f} deg/s")
_turn_s = vel_state.sum() * _dt_s
print(f"   Turning         : {_turn_s:.1f} s / {_session_s:.1f} s  "
      f"({100*_turn_s/_session_s:.1f}%)")
print(f"   Rejected (no motor match) : {_n_rejected} bout(s)  "
      f"[max_lag={max_lag_s*1000:.0f} ms]")

print(f"\n   min_bout_duration : {min_bout_duration_s*1000:.0f} ms  "
      f"({min_bout_samples} samples)")

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

# ── State shading: vectorised bout detection + single layout update ────────
# Using np.diff on the full-resolution state finds exact bout edges in one
# pass (no Python loop over samples).  All shapes are built as plain dicts
# and assigned to the layout in one call — O(N_bouts) instead of O(N_bouts²).


def _bouts_to_shapes(
    state: np.ndarray, time_s: np.ndarray, fillcolor: str, xref: str, yref: str
) -> list:
    """Return shape dicts for every active bout in *state*.

    Works on the full-resolution arrays so bout start/end times are exact.
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
_ts_xref   = "x3"
_ts_yref   = "y3 domain"
_snip_xref = "x4"
_snip_yref = "y4 domain"

_all_shapes = (
    _bouts_to_shapes(motor_state, _full_time_rel, "rgba(178,34,34,0.12)", _ts_xref, _ts_yref)
    + _bouts_to_shapes(vel_state,   _full_time_rel, "rgba(255,140,0,0.12)",  _ts_xref, _ts_yref)
    + _bouts_to_shapes(motor_state, _full_time_rel, "rgba(178,34,34,0.12)", _snip_xref, _snip_yref)
    + _bouts_to_shapes(vel_state,   _full_time_rel, "rgba(255,140,0,0.12)",  _snip_xref, _snip_yref)
)

fig_thresh.update_layout(
    template="plotly_white",
    height=950,
    title=(
        f"Movement threshold QC (rolling RMS + MAD)  |  "
        f"Motor thresh={motor_thresh:.2f} deg/s  |  "
        f"Vel_0Y thresh={vel_thresh:.2f} deg/s  |  "
        f"k_motor={k_motor}, k_turning={k_turning}, window={rms_window_s*1000:.0f} ms"
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
