# 1_3_1 Saccade Detection + Quantification

## Workflow

1. **Specify the data directory**  
   In Cell 1, set `data_path` to the experiment folder (e.g. `B6J2783-2025-04-28T14-57-30`). The notebook expects that `1_2_SLEAP_processing.ipynb` has already been run for this session.

2. **Run the notebook**  
   Execute cells in order (Cell 1 through Cell 8).

3. **Check detection quality (Cell 8)**  
   Inspect the QC plots to judge detection quality:
   - **Underdetected** (missed saccades): decrease `k` in Cell 2 (e.g. try 2.5–3.0).
   - **Overdetected** (false positives): increase `k` in Cell 2 (e.g. 3.4 or higher).
   - For noisy recordings, `k ≈ 2.5` often works well; for cleaner data, `k ≈ 3.4` or higher.
   - increse transient_pair_max_net_displacement_px if there are too many noise transients remaining 
   - ultimately there will be manual curation for both over- and underdetection, but overdetection is easier to deal with

4. **Manual curation (Cell 9)**  
   Use the curation GUI:
   - if previously saved curration is available, it will be loaded. 
   - **Events (Q/W):** Step through detected events. Delete non-saccades with **]** and add missed saccades by clicking start and end on the trace.
   - **Zoom:** Use the plot controls to zoom. Autoscaling resets when you press the home (reset zoom) icon and move between events (W/Q).
   - **Gaps (E/R):** Step through gaps (periods with no detected saccades, where missed saccades may occur). Use zoom and two-click to add saccades here.
   - **Jump:** Use the Event # / Gap # inputs and **Go** to jump directly to a specific event or gap.
   - **Done curating:** Click when finished.
   - **Reset:** If the GUI stops working (e.g. you can't add saccades), press Done Curating, then SAVE in Cell 10. Now you can safely re-run Cell 9 without loosing the previous curration. 
   **Notes** 
   - manually add saccades even if post-saccade data points are missing, if it is a clear saccade

5. **Save**  
   Run Cell 10. Click **Save curated events & metadata** when you are ready to write the curated outputs.

---

## Overview

This notebook performs velocity-threshold saccade detection on eye-tracking data from the vestibular VR pipeline, followed by filtering and manual curation.

### Prerequisites

- **Run `1_2_SLEAP_processing.ipynb` first** for the session. It produces:
  - `metadata/saccade_input_metadata.json` (with selected eye and video paths)
  - Resampled parquet files with `Seconds`, `Ellipse.Center.X`, `frame_idx`

### Input

| Variable | Location | Description |
|---------|----------|-------------|
| `data_path` | Cell 1 | Path to the experiment folder (e.g. `.../B6J2783-2025-04-28T14-57-30`). |
| `override_detect_params_with_metadata` | Cell 1 | If `True`, detection parameters from `saccade_input_metadata.json` override Cell 2 defaults. |
| `k` | Cell 2 | Detection sensitivity. Lower = more sensitive (more detections). Higher = more conservative. |
| `smoothing_window_s` | Cell 2 | Median smoothing window (seconds) before velocity. |
| `peak_width_time_s` | Cell 2 | Minimum peak width for velocity peaks. |
| `onset_fraction`, `offset_fraction` | Cell 2 | Threshold fractions for saccade start/end boundaries. |
| `refractory_period_s` | Cell 2 | ISI window for transient-pair filtering. |
| `same_direction_dedup_window_s` | Cell 2 | Window for deduplicating same-direction events. |
| `transient_pair_max_net_displacement_px` | Cell 2 | Max net displacement for transient-pair classification. |
| `min_saccade_amplitude_px` | Cell 2 | Minimum amplitude to keep an event. |

### Pipeline

1. **Cell 1:** Load metadata and selected eye from `1_2`. Load the resampled parquet.
2. **Cell 2:** Set detection and filtering parameters. Optionally load overrides from metadata.
3. **Cell 3:** Preprocess position (`X_raw` → `X_smooth` → `vel_x_smooth`).
4. **Cell 4:** Prepare velocity summary for detection.
5. **Cell 5:** Run velocity-threshold detection (find peaks).
6. **Cell 7:** Post-detection: transient-pair filter, then amplitude filter.
7. **Cell 8:** QC plots (detection, amplitude, duration, ISI).
8. **Cell 9:** Interactive curation GUI (add, delete, navigate events and gaps).
9. **Cell 10:** Save curated events and metadata.

On rerun, if `curated_saccade_events.csv` exists, the curation GUI loads it so your manual edits and deletions are preserved.

### Outputs

- **`curated_saccade_events.csv`** — Curated events with columns: `direction`, `time`, `velocity`, `start_time`, `end_time`, `duration`, `amplitude`, `source`, `TNT_direction`, etc.
- **`curated_saccade_snippets.parquet`** — ±5 s traces around each saccade. Columns: `event_idx`, `time_rel`, `time_abs`, `X_raw`, `frame_idx`.
- **`saccade_input_metadata.json`** — Updated with `saccade_detection_parameters` and `curation_summary`.

### Curation GUI shortcuts

| Key | Action |
|-----|--------|
| **W** / **Q** | Next / previous event |
| **R** / **E** | Next / previous gap |
| **]** | Delete current event |
| **Z** | Undo |
| **Click** | Two-click add: click start, then end of saccade |
