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
# This notebook currently includes only:
# - input contract + loading of selected eye parquet from `1_2` metadata
# - detection + k-sweep parameter block
# - preprocessing (`X_raw`, `X_smooth`, `vel_x_smooth`)
# - adaptive `k` sweep (coarse + fine) with heuristic selection
# - final detection run at selected `k`
# - optional debug plot of `Ellipse.Center.X`
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

data_path = Path(
    "/Users/rancze/Documents/Data/vestVR/20241125_Cohort1_rotation/Visual_mismatch_day4/B6J2718-2024-12-11T13-49-13"
)

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
# Cell 2: Detection and k-sweep parameters
##########################################################################

k = 4.5
refractory_period_s = 0.1
peak_width_time_s = 0.005
onset_offset_fraction = 0.2

pre_window_s = 0.15
post_window_s = 0.5

baseline_window_start_s = -0.06
baseline_window_end_s = -0.02

min_segment_duration_s = 0.2
min_saccade_amplitude_px = 3.0
k_overdetect_offset = -0.3
stability_guardrail_drop = 5.0
smoothing_window_s = 0.08
plot_detection_qc = True

print("✅ Detection and k-sweep parameters initialized")


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
# Cell 4: Prepare shared velocity summary for k-sweep
##########################################################################
# Keep one shared velocity summary for sweeps
abs_vel = df_work["vel_x_smooth"].abs().dropna()
if len(abs_vel) == 0:
    raise ValueError("No finite velocity samples available for saccade detection.")
print("ℹ️ Velocity summary prepared for k-sweep detection workflow.")


# %%
# Cell 5: k-value sweep (coarse -> fine) for adaptive threshold tuning
##########################################################################
k_coarse_values = np.arange(2.5, 7.0 + 1e-9, 0.5)
near_refractory_factor = 1.2  # "near" refractory if ISI <= refractory * factor
fine_step = 0.1
fine_half_range = 0.5
overdetect_bias_strength = 1  # >0 favors slightly lower k (more detections)
near_refractory_soft_cap_pct = 20.0  # strong penalty if exceeded
k_hard_min = 2.0
k_hard_max = 8.0
max_fine_expansions = 4

v = df_work["vel_x_smooth"].to_numpy()
x = df_work["X_raw"].to_numpy()
t = df_work["Seconds"].to_numpy()
f = df_work["frame_idx"].to_numpy()
peak_width_points = max(1, int(round(peak_width_time_s * fps)))
refractory_points = max(1, int(round(refractory_period_s * fps)))


def _events_for_k(k_value: float) -> tuple[pd.DataFrame, float, float, float]:
    """Run detection for a single k and return events + threshold stats."""
    vel_mean_k = abs_vel.mean()
    vel_std_k = abs_vel.std()
    vel_thresh_k = vel_mean_k + k_value * vel_std_k

    pos_peaks_k, _ = find_peaks(
        v,
        height=vel_thresh_k,
        distance=refractory_points,
        width=peak_width_points,
    )
    neg_peaks_k, _ = find_peaks(
        -v,
        height=vel_thresh_k,
        distance=refractory_points,
        width=peak_width_points,
    )

    def _extract_events_k(peak_indices, direction):
        events = []
        for peak_idx in peak_indices:
            start_idx = int(peak_idx)
            end_idx = int(peak_idx)

            while (
                start_idx > 0
                and abs(v[start_idx]) > vel_thresh_k * onset_offset_fraction
            ):
                start_idx -= 1

            while end_idx < len(v) - 1 and abs(v[end_idx]) > vel_thresh_k:
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


def _evaluate_k_grid(k_values: np.ndarray, label: str) -> pd.DataFrame:
    rows = []
    for kval in k_values:
        events_df, thr, m, s = _events_for_k(float(kval))
        n = len(events_df)
        isi = np.diff(events_df["time"].to_numpy()) if n >= 2 else np.array([])
        pct_near_ref = (
            float(
                np.mean(isi <= (refractory_period_s * near_refractory_factor)) * 100.0
            )
            if len(isi) > 0
            else 0.0
        )
        rows.append(
            {
                "sweep": label,
                "k": float(kval),
                "vel_thresh": float(thr),
                "vel_mean": float(m),
                "vel_std": float(s),
                "n_saccades": int(n),
                "pct_near_refractory": pct_near_ref,
                "median_duration": float(events_df["duration"].median())
                if n > 0
                else np.nan,
                "median_amplitude": float(events_df["amplitude"].median())
                if n > 0
                else np.nan,
            }
        )

    df_metrics = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    # Stability across neighboring k values based on count-change percentage.
    n_vals = df_metrics["n_saccades"].astype(float).to_numpy()
    local_change_pct = []
    for i in range(len(df_metrics)):
        changes = []
        if i > 0 and n_vals[i - 1] > 0:
            changes.append(abs(n_vals[i] - n_vals[i - 1]) / n_vals[i - 1] * 100.0)
        if i < len(df_metrics) - 1 and n_vals[i + 1] > 0:
            changes.append(
                abs(n_vals[i + 1] - n_vals[i]) / n_vals[i] * 100.0
                if n_vals[i] > 0
                else np.nan
            )
        local_change_pct.append(
            float(np.nanmean(changes)) if len(changes) > 0 else np.nan
        )

    # Only keep stability_score (derived from neighboring count consistency).
    df_metrics["stability_score"] = np.clip(
        100.0 - np.array(local_change_pct), 0.0, 100.0
    )
    return df_metrics


def _add_composite_score(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Add selection heuristic score with slight over-detection bias."""
    if len(df_metrics) < 3:
        df_metrics["composite_score"] = np.nan
        return df_metrics

    norm = lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    near_ref_penalty = np.clip(
        (df_metrics["pct_near_refractory"] - near_refractory_soft_cap_pct)
        / max(near_refractory_soft_cap_pct, 1e-9),
        a_min=0.0,
        a_max=None,
    )
    composite_score = (
        norm(df_metrics["stability_score"])
        - norm(df_metrics["pct_near_refractory"])
        + overdetect_bias_strength * norm(df_metrics["n_saccades"])
        - 0.25 * norm(df_metrics["k"])
        - 2.0 * near_ref_penalty
    )
    df_metrics["composite_score"] = composite_score
    return df_metrics


coarse_metrics_df = _evaluate_k_grid(k_coarse_values, label="coarse")
print("✅ Coarse k sweep complete")
if debug:
    display(coarse_metrics_df)

# Heuristic coarse-k candidate for fine sweep center.
if len(coarse_metrics_df) >= 3:
    coarse_metrics_df = _add_composite_score(coarse_metrics_df)
    k_coarse_best = float(
        coarse_metrics_df.loc[coarse_metrics_df["composite_score"].idxmax(), "k"]
    )
else:
    k_coarse_best = float(k)

print(f"Suggested coarse-k center for fine sweep: {k_coarse_best:.2f}")

k_fine_values = np.round(
    np.arange(
        k_coarse_best - fine_half_range,
        k_coarse_best + fine_half_range + 1e-9,
        fine_step,
    ),
    3,
)
k_fine_values = k_fine_values[
    (k_fine_values >= k_hard_min) & (k_fine_values <= k_hard_max)
]
fine_metrics_df = _evaluate_k_grid(k_fine_values, label="fine")
fine_metrics_df = _add_composite_score(fine_metrics_df)

# Adaptive fine-sweep expansion when optimum lands on boundary.
for _ in range(max_fine_expansions):
    if len(fine_metrics_df) < 3 or fine_metrics_df["composite_score"].isna().all():
        break

    idx_best = int(fine_metrics_df["composite_score"].idxmax())
    k_best_now = float(fine_metrics_df.loc[idx_best, "k"])
    k_min_now = float(fine_metrics_df["k"].min())
    k_max_now = float(fine_metrics_df["k"].max())
    at_lower_edge = np.isclose(k_best_now, k_min_now)
    at_upper_edge = np.isclose(k_best_now, k_max_now)

    if not (at_lower_edge or at_upper_edge):
        break

    if at_lower_edge and k_min_now > k_hard_min:
        new_center = max(k_hard_min, k_best_now - fine_half_range)
    elif at_upper_edge and k_max_now < k_hard_max:
        new_center = min(k_hard_max, k_best_now + fine_half_range)
    else:
        break

    k_fine_values = np.round(
        np.arange(
            new_center - fine_half_range, new_center + fine_half_range + 1e-9, fine_step
        ),
        3,
    )
    k_fine_values = k_fine_values[
        (k_fine_values >= k_hard_min) & (k_fine_values <= k_hard_max)
    ]
    fine_metrics_df = _evaluate_k_grid(k_fine_values, label="fine")
    fine_metrics_df = _add_composite_score(fine_metrics_df)

print("✅ Fine k sweep complete")
if debug:
    display(fine_metrics_df)

# Suggested k from final fine sweep
if len(fine_metrics_df) >= 3 and not fine_metrics_df["composite_score"].isna().all():
    k_suggested = float(
        fine_metrics_df.loc[fine_metrics_df["composite_score"].idxmax(), "k"]
    )
else:
    k_suggested = float(k_coarse_best)

print(f"🎯 Suggested k from fine sweep: {k_suggested:.2f}")

# Apply absolute over-detection offset with stability guardrail.
fine_k_vals = fine_metrics_df["k"].to_numpy(dtype=float)
fine_stability_vals = fine_metrics_df["stability_score"].to_numpy(dtype=float)
best_stability = float(np.interp(k_suggested, fine_k_vals, fine_stability_vals))

k_offset_applied = float(k_overdetect_offset)
while True:
    k_candidate = float(np.clip(k_suggested + k_offset_applied, k_hard_min, k_hard_max))
    stability_candidate = float(
        np.interp(k_candidate, fine_k_vals, fine_stability_vals)
    )
    if stability_candidate >= (best_stability - stability_guardrail_drop):
        break
    if k_offset_applied >= 0.0:
        break
    k_offset_applied = min(0.0, k_offset_applied + 0.05)

k_selected = float(np.clip(k_suggested + k_offset_applied, k_hard_min, k_hard_max))
stability_selected = float(np.interp(k_selected, fine_k_vals, fine_stability_vals))

# Run final detection using the guardrail-adjusted selected k for downstream QC/analysis.
selected_events_df, vel_thresh, vel_mean, vel_std = _events_for_k(k_selected)
all_saccades_df = selected_events_df.copy()
upward_saccades_df = all_saccades_df[all_saccades_df["direction"] == "upward"].copy()
downward_saccades_df = all_saccades_df[
    all_saccades_df["direction"] == "downward"
].copy()

print("✅ Final detection re-run complete (using guardrail-adjusted k)")
print(f"  k_suggested={k_suggested:.2f}")
print(
    f"  requested_offset={k_overdetect_offset:+.2f}, applied_offset={k_offset_applied:+.2f}, "
    f"k_selected={k_selected:.2f}"
)
print(
    f"  stability(best={best_stability:.2f}, selected={stability_selected:.2f}, "
    f"guardrail_drop={stability_guardrail_drop:.2f})"
)
print(f"  vel_thresh={vel_thresh:.2f} px/s")
print(
    f"  upward={len(upward_saccades_df)}, downward={len(downward_saccades_df)}, total={len(all_saccades_df)}"
)


# %%
# Cell 6: k-sweep metric curves and interpretation helpers
##########################################################################
fig_k = make_subplots(
    rows=3,
    cols=2,
    subplot_titles=(
        "Number of saccades",
        "% near refractory limit",
        "Median duration",
        "Median amplitude",
        "Stability score",
        "Composite score (selection heuristic)",
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.12,
)


def _add_metric_trace(
    dfm: pd.DataFrame, x_col: str, y_col: str, row: int, col: int, name: str, color: str
):
    fig_k.add_trace(
        go.Scatter(
            x=dfm[x_col],
            y=dfm[y_col],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=6),
        ),
        row=row,
        col=col,
    )


_add_metric_trace(coarse_metrics_df, "k", "n_saccades", 1, 1, "coarse", "royalblue")
_add_metric_trace(fine_metrics_df, "k", "n_saccades", 1, 1, "fine", "darkorange")

_add_metric_trace(
    coarse_metrics_df, "k", "pct_near_refractory", 1, 2, "coarse", "royalblue"
)
_add_metric_trace(
    fine_metrics_df, "k", "pct_near_refractory", 1, 2, "fine", "darkorange"
)

_add_metric_trace(
    coarse_metrics_df, "k", "median_duration", 2, 1, "coarse", "royalblue"
)
_add_metric_trace(fine_metrics_df, "k", "median_duration", 2, 1, "fine", "darkorange")

_add_metric_trace(
    coarse_metrics_df, "k", "median_amplitude", 2, 2, "coarse", "royalblue"
)
_add_metric_trace(fine_metrics_df, "k", "median_amplitude", 2, 2, "fine", "darkorange")

_add_metric_trace(
    coarse_metrics_df, "k", "stability_score", 3, 1, "coarse", "royalblue"
)
_add_metric_trace(fine_metrics_df, "k", "stability_score", 3, 1, "fine", "darkorange")

if "composite_score" in coarse_metrics_df.columns:
    _add_metric_trace(
        coarse_metrics_df, "k", "composite_score", 3, 2, "coarse", "royalblue"
    )
if "composite_score" in fine_metrics_df.columns:
    _add_metric_trace(
        fine_metrics_df, "k", "composite_score", 3, 2, "fine", "darkorange"
    )

for r in (1, 2, 3):
    for c in (1, 2):
        fig_k.update_xaxes(title_text="k", row=r, col=c)

fig_k.update_yaxes(title_text="count", row=1, col=1)
fig_k.update_yaxes(title_text="%", row=1, col=2)
fig_k.update_yaxes(title_text="seconds", row=2, col=1)
fig_k.update_yaxes(title_text="pixels", row=2, col=2)
fig_k.update_yaxes(title_text="0-100", row=3, col=1)
fig_k.update_yaxes(title_text="a.u.", row=3, col=2)

fig_k.add_vline(x=k_suggested, line_dash="dash", line_color="green")
fig_k.add_vline(x=k_selected, line_dash="dot", line_color="purple")
fig_k.update_layout(
    title=(
        f"k-sweep metrics ({eye_with_least_low_confidence}, eye={selected_eye_code}) | "
        f"suggested k={k_suggested:.2f}, selected k={k_selected:.2f}"
    ),
    template="plotly_white",
    width=900,
    height=650,
    font=dict(size=10),
    title_font=dict(size=14),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    legend_font=dict(size=9),
)
fig_k.update_annotations(font=dict(size=10))
fig_k.update_xaxes(tickfont=dict(size=9), title_font=dict(size=10))
fig_k.update_yaxes(tickfont=dict(size=9), title_font=dict(size=10))
fig_k.show()

print("\nSuggested k-selection approach:")
print("- Prefer k values with high stability_score and low % near refractory.")
print(
    "- This heuristic is intentionally biased toward slight over-detection (lower k / more events)."
)
print(
    f"- Strong penalty applied when % near refractory exceeds ~{near_refractory_soft_cap_pct:.0f}%."
)
print("- Fine-sweep range auto-expands if the best k lands on an edge.")
print(
    "- Use median duration/amplitude curves as sanity checks; abrupt jumps suggest unstable detection."
)
print(f"- Current heuristic suggestion: k = {k_suggested:.2f}")
print(
    f"- Final selected k applies absolute offset ({k_overdetect_offset:+.2f}) with "
    f"stability guardrail ({stability_guardrail_drop:.1f} points): k = {k_selected:.2f}"
)


# %%
# Cell 7: Amplitude filter (post-detection cleanup)
##########################################################################
all_saccades_df_pre_filter = all_saccades_df.copy()
upward_saccades_df_pre_filter = upward_saccades_df.copy()
downward_saccades_df_pre_filter = downward_saccades_df.copy()

all_saccades_df = all_saccades_df[
    all_saccades_df["amplitude"] >= min_saccade_amplitude_px
].copy()
all_saccades_df = all_saccades_df.sort_values("time").reset_index(drop=True)
upward_saccades_df = all_saccades_df[all_saccades_df["direction"] == "upward"].copy()
downward_saccades_df = all_saccades_df[
    all_saccades_df["direction"] == "downward"
].copy()
removed_saccades_df = all_saccades_df_pre_filter[
    all_saccades_df_pre_filter["amplitude"] < min_saccade_amplitude_px
].copy()

n_before = len(all_saccades_df_pre_filter)
n_after = len(all_saccades_df)
n_removed = n_before - n_after
removed_pct = (100.0 * n_removed / n_before) if n_before > 0 else 0.0
print(
    f"✅ Amplitude filter applied (min_saccade_amplitude_px={min_saccade_amplitude_px:.2f} px) | "
    f"kept={n_after}/{n_before}, removed={n_removed} ({removed_pct:.1f}%)"
)

if debug and n_removed > 0:
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
# Cell 8: QC plots for selected-k detection (after amplitude filtering)
##########################################################################
if plot_detection_qc:
    # Fast QC settings
    qc_max_points_global = 30000
    qc_max_points_velocity = 150000
    qc_density_window_s = 30.0
    qc_density_step_s = 10.0
    qc_local_window_s = 30.0

    # 1) Global lightweight overview (decimated traces + event rugs)
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
            "Velocity with threshold + event rugs",
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

    # Event rugs are much faster than drawing hundreds of rectangles.
    if len(upward_saccades_df) > 0:
        fig_overlay.add_trace(
            go.Scatter(
                x=upward_saccades_df["time"] - df_work["Seconds"].iloc[0],
                y=np.full(len(upward_saccades_df), vel_thresh * 1.02),
                mode="markers",
                name="upward events (rug)",
                marker=dict(color="limegreen", size=5, symbol="line-ns"),
            ),
            row=2,
            col=1,
        )
    if len(downward_saccades_df) > 0:
        fig_overlay.add_trace(
            go.Scatter(
                x=downward_saccades_df["time"] - df_work["Seconds"].iloc[0],
                y=np.full(len(downward_saccades_df), -vel_thresh * 1.02),
                mode="markers",
                name="downward events (rug)",
                marker=dict(color="mediumpurple", size=5, symbol="line-ns"),
            ),
            row=2,
            col=1,
        )

    # Direction-colored event spans on X-position panel only (row 1).
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
        title=f"Selected-k detection QC (global, fast) ({eye_with_least_low_confidence}, eye={selected_eye_code})",
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

        if len(all_saccades_df) >= 2:
            isi = np.diff(np.sort(all_saccades_df["time"].to_numpy()))
            isi = isi[isi > 0]
            if len(isi) > 0:
                isi_min = float(np.min(isi))
                isi_max = float(np.max(isi))
                if np.isclose(isi_min, isi_max):
                    bins = np.array([isi_min * 0.8, isi_max * 1.2])
                else:
                    bins = np.logspace(np.log10(isi_min), np.log10(isi_max), 80)
                counts, edges = np.histogram(isi, bins=bins)
                centers = np.sqrt(edges[:-1] * edges[1:])
                fig_dist.add_trace(
                    go.Bar(
                        x=centers,
                        y=counts,
                        width=np.diff(edges),
                        marker_color="darkcyan",
                        opacity=0.85,
                        name="ISI (kept)",
                    ),
                    row=1,
                    col=4,
                )
                fig_dist.update_xaxes(type="log", row=1, col=4)
                fig_dist.add_vline(
                    x=refractory_period_s,
                    line_dash="dash",
                    line_color="red",
                    row=1,
                    col=4,
                )
            else:
                print(
                    "ℹ️ ISI panel empty (all ISI values are non-positive after filtering)."
                )
        else:
            print("ℹ️ ISI panel empty (need at least 2 detected saccades).")

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
# Cell 9: Manual full-resolution QC windows (3-column layout)
##########################################################################
# Relative-time windows requested for detailed inspection
manual_qc_windows = [
    (162.0, 164.0, ""),
    (216.0, 218.0, "fast transient"),
    (466.0, 468.0, ""),
    (488.0, 490.0, ""),
    (569.0, 571.0, "blinkdrop"),
    (571.0, 573.0, "blinkdrop"),
    (573.0, 575.0, "blinkdrop"),
    (780.0, 782.0, ""),
    (1210.0, 1212.0, ""),
    (1214.0, 1216.0, ""),
    (1592.0, 1594.0, ""),
]

if plot_detection_qc:
    t0 = float(df_work["Seconds"].iloc[0])
    n_windows = len(manual_qc_windows)
    n_cols = 3
    n_rows = int(np.ceil(n_windows / n_cols))
    n_plot_rows = 2 * n_rows

    # make_subplots assigns subplot titles in row-major order.
    # Build titles with explicit row/col indexing so each panel label matches its data.
    subplot_titles = [""] * (n_plot_rows * n_cols)
    for i, (ws_rel, we_rel, note) in enumerate(manual_qc_windows):
        block_row = i // n_cols
        row_pos = (2 * block_row) + 1
        row_vel = row_pos + 1
        col = i % n_cols + 1
        note_txt = f" ({note})" if note else ""
        base_label = f"{ws_rel:.0f}-{we_rel:.0f}s{note_txt}"
        idx_pos = (row_pos - 1) * n_cols + (col - 1)
        idx_vel = (row_vel - 1) * n_cols + (col - 1)
        subplot_titles[idx_pos] = f"{base_label} - position"
        subplot_titles[idx_vel] = f"{base_label} - velocity"

    # Single paired figure per window: top=position, bottom=velocity
    fig_manual_pairs = make_subplots(
        rows=n_plot_rows,
        cols=n_cols,
        subplot_titles=tuple(subplot_titles),
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    for i, (ws_rel, we_rel, _) in enumerate(manual_qc_windows):
        block_row = i // n_cols
        row_pos = (2 * block_row) + 1
        row_vel = row_pos + 1
        col = i % n_cols + 1
        ws_abs = t0 + ws_rel
        we_abs = t0 + we_rel

        mask = (df_work["Seconds"] >= ws_abs) & (df_work["Seconds"] <= we_abs)
        if not mask.any():
            continue

        x_rel = (df_work.loc[mask, "Seconds"] - t0).to_numpy()
        x_raw = df_work.loc[mask, "X_raw"].to_numpy()
        vel = df_work.loc[mask, "vel_x_smooth"].to_numpy()

        # Position trace (top panel)
        fig_manual_pairs.add_trace(
            go.Scatter(
                x=x_rel,
                y=x_raw,
                mode="lines",
                line=dict(color="royalblue", width=1),
                name="X_raw",
                showlegend=(i == 0),
            ),
            row=row_pos,
            col=col,
        )

        # Velocity trace (bottom panel)
        fig_manual_pairs.add_trace(
            go.Scatter(
                x=x_rel,
                y=vel,
                mode="lines",
                line=dict(color="firebrick", width=1),
                name="vel_x_smooth",
                showlegend=(i == 0),
            ),
            row=row_vel,
            col=col,
        )

        # Threshold lines (velocity panel)
        fig_manual_pairs.add_hline(
            y=vel_thresh,
            line_dash="dash",
            line_color="green",
            line_width=1,
            row=row_vel,
            col=col,
        )
        fig_manual_pairs.add_hline(
            y=-vel_thresh,
            line_dash="dash",
            line_color="green",
            line_width=1,
            row=row_vel,
            col=col,
        )

        # Direction-colored event spans on both position and velocity panels.
        # Use overlap condition so spans appear even if peak is just outside window.
        events_local = all_saccades_df[
            (all_saccades_df["start_time"] <= we_abs)
            & (all_saccades_df["end_time"] >= ws_abs)
        ]
        if len(events_local) > 0:
            for _, ev in events_local.iterrows():
                x0 = max(ws_abs, ev["start_time"]) - t0
                x1 = min(we_abs, ev["end_time"]) - t0
                fill = (
                    "rgba(34,139,34,0.10)"
                    if ev["direction"] == "upward"
                    else "rgba(128,0,128,0.10)"
                )
                line = (
                    "rgba(34,139,34,0.25)"
                    if ev["direction"] == "upward"
                    else "rgba(128,0,128,0.25)"
                )
                fig_manual_pairs.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=fill,
                    line=dict(color=line, width=1),
                    row=row_pos,
                    col=col,
                )
                fig_manual_pairs.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=fill,
                    line=dict(color=line, width=1),
                    row=row_vel,
                    col=col,
                )

    fig_manual_pairs.update_layout(
        title="Manual full-resolution QC windows: Paired position/velocity",
        template="plotly_white",
        height=260 * n_plot_rows,
        width=1400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        font=dict(size=10),
    )

    for rr in range(1, n_plot_rows + 1):
        for cc in range(1, n_cols + 1):
            if rr % 2 == 0:
                fig_manual_pairs.update_xaxes(
                    title_text="Relative time (s)",
                    row=rr,
                    col=cc,
                    tickfont=dict(size=9),
                    title_font=dict(size=10),
                )
            else:
                fig_manual_pairs.update_xaxes(
                    showticklabels=False,
                    row=rr,
                    col=cc,
                )

            fig_manual_pairs.update_yaxes(
                title_text="X position (px)" if rr % 2 == 1 else "Velocity (px/s)",
                row=rr,
                col=cc,
                tickfont=dict(size=9),
                title_font=dict(size=10),
            )
    fig_manual_pairs.show()
