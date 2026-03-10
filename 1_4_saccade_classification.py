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
# determine motor velocity threshold using adaptive thresholding, this is used to determine rotating and not-rotating periods which will help with saccade classification
