# Eye Tracking Analysis Changes - Summary

## Issues Fixed

### 1. ✅ Eye Position Analysis Now Respects Turn Direction
**Problem**: Eye position was being averaged across left and right turns, which doesn't make sense because eye position moves in opposite directions for opposite turns.

**Solution**: 
- Eye position metrics are now kept **separate for left and right turns**
- Saccade and pupil metrics continue to be averaged across all turns (direction doesn't matter for these)
- New configuration flag `direction_matters` distinguishes these cases

### 2. ✅ Eye Tracking PDF Plots Now Save Correctly
**Problem**: The `eye_tracking_avg_traces.pdf` file was not being saved.

**Solution**:
- Fixed the save path to use explicit `output_path` variable
- Plot now saves correctly to `OUTPUT_DIR / "eye_tracking_avg_traces.pdf"`

### 3. ✅ Better Diagnostics for Empty CSV Files
**Problem**: CSV files were empty but there was no clear indication of why data loading failed.

**Solution**:
- Added comprehensive data loading summary that shows:
  - Total files processed
  - Number of successes vs failures per metric
  - First 3 detailed error messages for each metric
  - Total data points loaded for traces
- Empty dataframe detection before processing

---

## What Changed in the Code

### Data Loading Section (Lines 1442-1600)

**Before**: Generic error handling with minimal feedback
```python
for metric_name, column_name, display_name in eye_metrics_config:
    try:
        df = load_time_series(...)
```

**After**: Direction-aware loading with detailed diagnostics
```python
for metric_name, column_name, display_name, direction_matters in eye_metrics_config:
    try:
        df = load_time_series(...)
        if df.empty:
            raise ValueError(f"Empty dataframe after loading {column_name}")
        n_files_with_data[metric_name] += 1
```

**New Output**:
```
📊 Eye tracking data loading summary:
   Total files processed: 12
   saccade: 10 succeeded, 2 failed
   pupil: 12 succeeded, 0 failed
   eye_position: 8 succeeded, 4 failed
```

### Aggregation Section (Lines 1547-1600)

**Before**: Single aggregation for all metrics
```python
eye_tracking_per_mouse = eye_tracking_df.groupby(["group", "mouse"])
```

**After**: Separate aggregation strategies
```python
# Saccade & Pupil: average across ALL turns
eye_tracking_per_mouse = eye_tracking_df.groupby(["group", "mouse"]).agg(...)

# Eye Position: keep direction SEPARATE
eye_position_per_mouse = eye_tracking_df.groupby(
    ["group", "mouse", "direction"]
).agg(...)
```

**New Files Saved**:
- `saccade_pupil_metrics_per_mouse.csv` (averaged across turns)
- `eye_position_metrics_per_mouse.csv` (separated by turn direction)

### Trace Computation Section (Lines 1652-1721)

**Before**: Single averaging strategy
```python
avg_trace = traces.groupby(["group", "time"]).agg(...)
```

**After**: Direction-aware averaging
```python
if direction_matters:
    # Keep left and right separate
    avg_trace = traces.groupby(["group", "direction", "time"]).agg(...)
else:
    # Average across all turns
    avg_trace = traces.groupby(["group", "time"]).agg(...)
```

### Plotting Section (Lines 1724-1852)

**Before**: 3 panels (saccade, pupil, eye position)
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
```

**After**: 4 panels (saccade, pupil, eye_pos_left, eye_pos_right)
```python
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
# Separate plots for left and right turns
for turn_direction in ["left", "right"]:
    direction_data = eye_pos_df[eye_pos_df["direction"] == turn_direction]
```

### Statistical Comparison Section (Lines 1855-2116)

**Before**: Combined comparison for all 3 metrics
```python
eye_metric_specs = [
    ("saccade_post_mean", ...),
    ("pupil_post_mean", ...),
    ("eye_position_post_mean", ...),  # ❌ Averaged across turns
]
```

**After**: Separate comparisons
```python
# Section 1: Saccade & Pupil (averaged)
eye_metric_specs = [
    ("saccade_post_mean", ...),
    ("pupil_post_mean", ...),
]

# Section 2: Eye Position (by direction)
for turn_direction in ["left", "right"]:
    direction_data = eye_position_per_mouse[
        eye_position_per_mouse["direction"] == turn_direction
    ]
```

**New Files Saved**:
- `saccade_pupil_metric_comparisons.pdf` (2 panels)
- `eye_position_metric_comparisons.pdf` (2 panels: left vs right)
- `saccade_pupil_metrics_ttests.csv`
- `eye_position_metrics_ttests.csv` (includes direction column)

---

## Output Files Summary

### Before (3 eye files):
- ❌ `eye_tracking_metrics_per_file.csv` (mixed directions)
- ❌ `eye_tracking_metrics_per_mouse.csv` (mixed directions)
- ❌ `eye_tracking_metric_comparisons.pdf` (incorrect averaging)

### After (7 eye files):
- ✅ `eye_tracking_metrics_per_file.csv` (all raw metrics)
- ✅ `saccade_pupil_metrics_per_mouse.csv` (averaged across turns)
- ✅ `eye_position_metrics_per_mouse.csv` (separated by direction)
- ✅ `eye_saccade_avg_traces.csv` (time traces)
- ✅ `eye_pupil_avg_traces.csv` (time traces)
- ✅ `eye_position_avg_traces.csv` (time traces with direction)
- ✅ `eye_tracking_avg_traces.pdf` (4-panel plot)
- ✅ `saccade_pupil_metric_comparisons.pdf` (2-panel comparison)
- ✅ `eye_position_metric_comparisons.pdf` (2-panel comparison by direction)
- ✅ `saccade_pupil_metrics_ttests.csv` (statistics)
- ✅ `eye_position_metrics_ttests.csv` (statistics with direction)

---

## Why These Changes Matter

### Scientific Validity
**Eye position is directional**: When the mouse turns left, the eyes move one way. When it turns right, the eyes move the opposite way. Averaging these together would **cancel out the signal** and give meaningless results.

**Example**:
- Left turn: Eye position changes by +10 pixels
- Right turn: Eye position changes by -10 pixels
- ❌ Old average: (10 + -10) / 2 = 0 (looks like no change!)
- ✅ New approach: Analyze left (+10) and right (-10) separately

### Saccade and Pupil Don't Need Direction
- **Saccade probability**: How often saccades occur (doesn't matter which direction)
- **Pupil diameter**: Size of pupil (isotropic, no directional component)
- These metrics are correctly averaged across all turns

---

## How to Use the New Analysis

1. **Saccade & Pupil**: Look at the averaged metrics (same as before, but now clearer naming)
   - Files: `saccade_pupil_metrics_per_mouse.csv`, `saccade_pupil_metric_comparisons.pdf`

2. **Eye Position**: Look at left and right turns separately
   - Files: `eye_position_metrics_per_mouse.csv` (has "direction" column), 
     `eye_position_metric_comparisons.pdf` (2 panels)

3. **Time Traces**: Check the 4-panel plot to see temporal dynamics
   - File: `eye_tracking_avg_traces.pdf`

4. **Diagnostics**: If data is missing, check the console output for loading errors
   - Shows which files failed and why
   - Reports success/failure counts per metric

---

## Testing Recommendations

1. **Verify data loading**:
   ```python
   # Check the diagnostic output
   📊 Eye tracking data loading summary:
      saccade: X succeeded, Y failed
   ```

2. **Check column names**: If all metrics fail, the column names in your CSV might be different
   - Current expected names:
     - `saccade_probability_eye1`
     - `Pupil.Diameter_eye1_Baseline`
     - `Ellipse.Center.X_eye1_Baseline`

3. **Verify eye position separation**: 
   - `eye_position_metrics_per_mouse.csv` should have a "direction" column
   - Should have separate rows for "left" and "right" turns

4. **Check PDFs**:
   - `eye_tracking_avg_traces.pdf`: Should have 4 panels
   - `eye_position_metric_comparisons.pdf`: Should show left vs right separately

---

---

## UPDATE: Saccade Probability Scaling Issue

### Issue
Saccade probability values are showing as very small (e.g., 0.0004-0.0008) instead of the expected 0-1 range.

### Investigation
**The script does NOT apply any scaling** - it loads raw values directly from the CSV column `saccade_probability_eye1`.

The small values could mean:
1. **Per-frame probability**: If this is the probability of a saccade occurring in each individual frame (e.g., at 30 fps), then 0.0008 per frame = 0.024 per second = 2.4% chance per second, which is reasonable
2. **Wrong column**: The column name might be different in your CSV files
3. **Data format issue**: The data might be stored differently than expected

### How to Debug

**Option 1: Check what columns are available**
Set this flag in the script (around line 1463):
```python
SHOW_AVAILABLE_COLUMNS = True  # Set to True to see what columns are in your CSV files
```

This will print all eye-related columns and their value ranges from your first CSV file.

**Option 2: Check the diagnostic output**
When you run the script, look for:
```
📊 Data range diagnostics (raw values from CSV):
   saccade_post_mean: min=0.000400, max=0.000800, mean=0.000600
   ⚠️ WARNING: Saccade probability values are very small (< 0.01)
```

**Option 3: Manually inspect your CSV**
Open one of your CSV files and check:
- What columns contain "saccade" in the name?
- What are the actual values in those columns?
- Are there other saccade-related columns with larger values?

### Possible Solutions

If you find a different column name has the values you expect:
1. Update line 90 in the script:
   ```python
   SACCADE_COLUMN = "your_correct_column_name_here"
   ```

If the values are correct but just small (per-frame probability):
- The values are fine as-is, they just represent per-frame probability
- The plots will show the correct relative changes between conditions

---

## Pupil Diameter Status

✅ **Pupil diameter is ALREADY set to COMBINED across all turns** (not separated by direction).

This is the correct approach because pupil diameter is isotropic (doesn't have a directional component). The configuration is:
```python
("pupil", PUPIL_COLUMN, "Pupil Diameter", False),  # ✓ COMBINED across all turns
```

---

---

## UPDATE 2: Saccade Probability and Pupil Treated as Absolute Values

### Issue
Saccade probability and pupil diameter were being treated like velocity (signed values), but they should be treated as **absolute values** because:
- **Probability**: Always in range 0-1, has no negative values
- **Pupil diameter**: Always positive (it's a size measurement)

The CSV files incorrectly labeled these columns as "velocity" internally, causing confusion.

### What Changed

**Before**: All eye metrics loaded as "velocity" and could have signed values
```python
df = load_time_series(..., value_alias="velocity")
metrics = analyse_turn_direction(df, ...)  # Looked for direction changes (wrong for probability!)
```

**After**: Metrics properly distinguished as absolute vs signed
```python
# Configuration now includes is_absolute flag:
("saccade", SACCADE_COLUMN, "Saccade Probability", False, True),  # ABSOLUTE
("pupil", PUPIL_COLUMN, "Pupil Diameter", False, True),           # ABSOLUTE  
("eye_position", EYE_POSITION_COLUMN, "Eye Position (X)", True, False),  # SIGNED

# Loading applies abs() for probability and diameter:
df = load_time_series(..., value_alias="value")
if is_absolute:
    df["value"] = df["value"].abs()  # Force positive values
```

### Impact on Analysis

1. **Saccade probability**: Now correctly treated as magnitude (0-1 range)
   - No longer analyzes "direction changes" (which made no sense for probability)
   - Uses absolute values throughout
   
2. **Pupil diameter**: Now correctly treated as size (always positive)
   - Takes absolute value to ensure no negative sizes
   
3. **Eye position**: Still correctly treated as signed (can be left/right of center)
   - Keeps original sign for directional analysis

### CSV Column Names Updated

**Internal processing**:
- Changed: `"velocity"` → `"value"` (more accurate generic name)

**CSV output columns** (semantically meaningful names):
- `eye_saccade_avg_traces.csv`: 
  - ✅ `mean_probability`, `sem_probability` (not velocity!)
  - Represents probability of saccade occurrence (0-1 or percentage)
  
- `eye_pupil_avg_traces.csv`: 
  - ✅ `mean_diameter`, `sem_diameter` (not velocity!)
  - Represents pupil diameter/size (pixels or mm)
  
- `eye_position_avg_traces.csv`: 
  - ✅ `mean_position`, `sem_position` (not velocity!)
  - Represents eye position coordinate (pixels or degrees, can be signed)

### Why This Matters

**Before** (incorrect):
- All columns called `mean_velocity` → implied rate of change
- Confusing because probability, diameter, and position are NOT velocities

**After** (correct):
- Each metric has semantically appropriate names
- `mean_probability` → clearly a probability value
- `mean_diameter` → clearly a size measurement  
- `mean_position` → clearly a spatial coordinate

---

## Date: November 11, 2025
## Modified File: `SANDBOX_4_MM_aligned_running_turning_analysis.py`

