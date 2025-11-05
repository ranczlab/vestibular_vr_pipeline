# Saccade Detection: Logic, Parameters, and Tuning Guide

## Overview

Saccade detection uses an adaptive statistical threshold method based on velocity analysis. The pipeline detects fast eye movements, quantifies their amplitude and timing, and optionally classifies them as orienting vs compensatory saccades.

## Detection Algorithm

### Step 1: Data Preprocessing
1. **Position smoothing**: Apply rolling median filter to `Ellipse.Center.X` position data
   - Window size: `saccade_smoothing_window` (default: 5 frames)
   - Purpose: Reduce noise while preserving saccade dynamics

2. **Velocity calculation**: Compute instantaneous velocity from smoothed position
   - Formula: `velocity = diff(position) / diff(time)`

### Step 2: Adaptive Threshold Calculation
- **Method**: Statistical threshold based on velocity distribution
- **Formula**: `threshold = mean(|velocity|) + k * std(|velocity|)`
- **Parameters**:
  - `k1` / `k2`: Number of standard deviations (separate for each eye)
  - **Default**: `k1 = 6`, `k2 = 6` (adjustable: 2-6 range works well)

### Step 3: Peak Detection
- **Method**: Use `scipy.find_peaks()` to find velocity peaks/troughs
- **Parameters**:
  - `height`: Adaptive threshold (from Step 2)
  - `distance`: `fps * refractory_period` (minimum time between peaks)
  - `width`: `saccade_peak_width` (minimum peak width in samples, default: 1)
- **Output**: Detected peak indices for upward and downward saccades

### Step 4: Onset/Offset Detection
For each detected peak:
- **Onset** (backward search):
  - Start at peak, go backward until velocity drops below `threshold * onset_offset_fraction`
  - Uses threshold fraction (0.2) for sensitivity to catch early movement
- **Offset** (forward search):
  - Start at peak, go forward until velocity drops below **full threshold**
  - Increment by 1 frame to get frame AFTER velocity dropped
  - Uses full threshold for accuracy (important for low sampling rate)

### Step 5: Amplitude Calculation
- **Method**: Absolute position change between onset and offset
- **Formula**: `amplitude = |position[end_idx] - position[start_idx]|`

### Step 6: Peri-Saccade Segment Extraction
- Extract `n_before` points before and `n_after` points after the threshold crossing (onset)
- Baseline correction using `baseline_n_points` before threshold crossing

## Input Parameters

### Core Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k1`, `k2` | 6 | Number of standard deviations for adaptive threshold (separate for each eye) |
| `refractory_period` | 0.1 sec | Minimum time between saccade onsets |
| `onset_offset_fraction` | 0.2 | Fraction of peak velocity threshold for onset detection (20% of threshold) |
| `saccade_smoothing_window` | 5 frames | Rolling median window size for position smoothing |
| `saccade_peak_width` | 1 frame | Minimum peak width in samples for `find_peaks()` |

### Segment Extraction Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_before` | 10 points | Points before threshold crossing to extract |
| `n_after` | 30 points | Points after threshold crossing to extract |
| `baseline_n_points` | 5 points | Points before threshold crossing used for baseline calculation |

### Classification Parameters (Optional)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `classify_orienting_compensatory` | True | Enable/disable classification |
| `bout_window` | 1.5 sec | Time window for grouping saccades into bouts |
| `pre_saccade_window` | 0.3 sec | Time window before saccade to analyze |
| `max_intersaccade_interval_for_classification` | 5.0 sec | Max time to extend post-saccade window |
| `pre_saccade_velocity_threshold` | 50.0 px/s | Velocity threshold for pre-saccade drift |
| `pre_saccade_drift_threshold` | 10.0 px | Position drift threshold before saccade |
| `post_saccade_variance_threshold` | 100.0 px² | Position variance threshold after saccade |
| `post_saccade_position_change_threshold_percent` | 50.0% | Position change threshold (% of amplitude) |
| `use_adaptive_thresholds` | True | Use adaptive thresholds based on feature distributions |

## Tuning Guide

### Problem: Under-detection (Missing Saccades)

**Symptoms**: Fewer saccades detected than expected, obvious saccades not detected

**Solutions**:
1. **Decrease `k1` / `k2`** (e.g., 6 → 4 or 5)
   - Makes threshold more sensitive
   - Lower threshold catches smaller/weaker saccades
   - ⚠️ Risk: May increase false positives

2. **Decrease `saccade_peak_width`** (e.g., 1 → 1, already minimal)
   - Allows detection of very brief saccades
   - Only relevant if saccades are extremely fast (< 1 frame at peak)

3. **Decrease `refractory_period`** (e.g., 0.1 → 0.05 sec)
   - Allows detection of rapid saccade sequences
   - ⚠️ Risk: May split single saccades into multiple detections

4. **Decrease `saccade_smoothing_window`** (e.g., 5 → 3 frames)
   - Less smoothing preserves faster dynamics
   - ⚠️ Risk: May increase noise sensitivity

### Problem: Over-detection (Too Many False Positives)

**Symptoms**: Too many saccades detected, noise classified as saccades, small movements detected

**Solutions**:
1. **Increase `k1` / `k2`** (e.g., 6 → 7 or 8)
   - Makes threshold more stringent
   - Filters out smaller movements and noise
   - ⚠️ Risk: May miss genuine small saccades

2. **Increase `saccade_peak_width`** (e.g., 1 → 2 or 3 frames)
   - Requires peaks to be wider (more stable)
   - Filters out brief noise spikes
   - ⚠️ Risk: May miss very fast saccades

3. **Increase `refractory_period`** (e.g., 0.1 → 0.15 or 0.2 sec)
   - Prevents detection of rapid noise fluctuations
   - More physiologically realistic
   - ⚠️ Risk: May miss rapid genuine saccade sequences

4. **Increase `saccade_smoothing_window`** (e.g., 5 → 7 or 9 frames)
   - More smoothing reduces noise
   - ⚠️ Risk: May blur saccade dynamics, especially for fast saccades

### Problem: Incorrect Onset/Offset Timing

**Symptoms**: Saccade duration seems wrong, onset/offset don't align with visual inspection

**Solutions**:
1. **Adjust `onset_offset_fraction`** (default: 0.2)
   - **Decrease** (e.g., 0.2 → 0.1): More sensitive onset detection, catches earlier movement
   - **Increase** (e.g., 0.2 → 0.3): More conservative onset detection, excludes slow pre-movement
   - Note: Onset uses threshold_fraction, offset uses full threshold (fixed for low sampling rate)

2. **For low sampling rate data**: 
   - Current implementation uses full threshold for offset detection (optimal for 1-frame velocity spikes)
   - No adjustment needed - this is handled automatically

### Problem: Incorrect Amplitude Measurements

**Symptoms**: Amplitude seems too small, doesn't match visual inspection

**Potential issues**:
- **Saccade overshoot**: Current calculation uses onset/offset positions, not peak position
  - May underestimate amplitude if saccade overshoots
  - Consider: Use maximum position change within saccade window (future enhancement)

- **Baseline contamination**: Baseline may overlap with previous saccade
  - **Solution**: Decrease `baseline_n_points` if segments are too short
  - Or increase `n_before` to ensure sufficient pre-saccade data

### Problem: Classification Issues (Orienting vs Compensatory)

**See separate classification tuning guide** (parameters listed above)

## Recommended Workflow

1. **Start with default parameters** (see parameter table above)
2. **Run detection** and inspect results visually
3. **If under-detection**: Gradually decrease `k1`/`k2` (try 5, then 4)
4. **If over-detection**: Gradually increase `k1`/`k2` (try 7, then 8)
5. **Fine-tune**: Adjust `refractory_period` and `saccade_smoothing_window` based on data quality
6. **Validate**: Check detected saccades against visual inspection or manual annotations

## Important Notes

- **Low sampling rate**: The code is optimized for low sampling rates where velocity may only exceed threshold for 1 frame. Offset detection uses full threshold for this reason.
- **Separate parameters per eye**: `k1` and `k2` allow different thresholds for left/right eyes if needed
- **Refractory period**: Applied to peak distances, not onset distances (physiological constraint should ideally be on onsets)
- **Adaptive threshold**: Automatically adjusts to your data's velocity distribution - no manual threshold setting needed

## Parameter Location

All parameters are defined in **Cell 2** of `SANDBOX_1_1_Loading_+_Saccade_detection.py` (around lines 2881-2895).

## Example: Tuning for Noisy Data

```python
# For noisy data with many false positives:
k1 = 7  # Increase threshold (was 6)
k2 = 7
saccade_smoothing_window = 7  # More smoothing (was 5)
saccade_peak_width = 2  # Require wider peaks (was 1)
refractory_period = 0.15  # Longer refractory period (was 0.1)
```

## Example: Tuning for Low Sampling Rate

```python
# For very low sampling rate (< 30 Hz):
k1 = 5  # Lower threshold to catch brief saccades (was 6)
k2 = 5
saccade_smoothing_window = 3  # Less smoothing to preserve dynamics (was 5)
saccade_peak_width = 1  # Keep minimal (already optimal)
refractory_period = 0.08  # Shorter refractory period (was 0.1)
```

