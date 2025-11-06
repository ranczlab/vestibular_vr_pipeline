# Saccade Detection: Logic and Flow

## Overview

This document describes the logic and implementation of saccade detection, which identifies rapid eye movements (saccades) from eye-tracking time series data. The detection uses an adaptive statistical threshold method combined with peak detection, followed by post-processing steps including segment extraction, baselining, and outlier filtering.

The detection pipeline consists of multiple stages: preprocessing, adaptive threshold calculation, peak detection, onset/offset determination, segment extraction, duration filtering, baselining, and outlier filtering.

---

## Detection Logic and Flow

### Stage 1: Preprocessing

**Purpose**: Prepare position and velocity signals for robust saccade detection.

**Algorithm**:
1. **Store raw position**: Copy `Ellipse.Center.X` to `X_raw` for accurate amplitude calculations
2. **Smooth position**: Apply rolling median filter to `X_raw` with window size `smoothing_window_time` (default: 80ms)
   - Uses `center=True` for symmetric smoothing
   - Handles edge cases with forward/backward fill
   - Result stored in `X_smooth`
3. **Compute velocity**: Calculate instantaneous velocity from smoothed position
   - `vel_x_smooth = diff(X_smooth) / dt`
   - This produces smooth velocity because position was smoothed before differentiation
4. **Reference velocities**: Also compute `vel_x_raw` and `vel_x_original` from raw position (for reference, not used in detection)

**Key Principle**: Smooth position *before* differentiation to reduce noise. Differentiation amplifies high-frequency noise, so smoothing first is crucial for robust detection.

### Stage 2: Adaptive Threshold Calculation

**Purpose**: Calculate velocity threshold that adapts to the signal characteristics of each recording.

**Algorithm**:
1. Calculate absolute velocity: `abs_vel = abs(vel_x_smooth)`
2. Compute statistics: `mean = mean(abs_vel)`, `std = std(abs_vel)`
3. Calculate adaptive threshold: `vel_thresh = mean + k * std`
   - `k` is the threshold multiplier (default: 5)
   - Higher `k` = stricter threshold = fewer detections
   - Lower `k` = more lenient threshold = more detections

**Rationale**: Each recording has different noise levels and signal characteristics. Adaptive thresholds ensure consistent detection performance across different recording conditions without manual tuning.

### Stage 3: Peak Detection

**Purpose**: Identify velocity peaks that exceed the adaptive threshold, representing potential saccades.

**Algorithm**:
1. **Find positive peaks** (upward saccades):
   - Use `scipy.signal.find_peaks` on `vel_x_smooth`
   - Parameters:
     - `height=vel_thresh`: Minimum peak height
     - `distance=int(fps * refractory_period)`: Minimum distance between peaks (prevents double-detection)
     - `width=peak_width`: Minimum peak width in samples (converted from `peak_width_time`)
2. **Find negative peaks** (downward saccades):
   - Use `scipy.signal.find_peaks` on `-vel_x_smooth` (invert signal to find troughs)
   - Same parameters as positive peaks

**Result**: Arrays of peak indices (`pos_peaks`, `neg_peaks`) representing detected velocity peaks.

### Stage 4: Onset and Offset Detection

**Purpose**: Determine the start and end times of each saccade by finding when velocity crosses thresholds.

**Algorithm** (for each detected peak):

1. **Find onset** (start of saccade):
   - Start at peak index, move backward in time
   - Stop when `abs(velocity) <= vel_thresh * onset_offset_fraction`
   - Uses fraction of threshold (default: 0.2 = 20%) for sensitivity to catch early movement
   - Result: `start_idx`, `start_time`

2. **Find offset** (end of saccade):
   - Start at peak index, move forward in time
   - Stop when `abs(velocity) <= vel_thresh` (full threshold)
   - Uses full threshold since velocity is only above threshold briefly
   - Increment by 1 to get frame after velocity drops below threshold
   - Result: `end_idx`, `end_time`

3. **Calculate saccade properties**:
   - `duration = end_time - start_time`
   - `amplitude = abs(end_position - start_position)` (from raw position)
   - `displacement = end_position - start_position` (signed, for direction validation)
   - `peak_velocity = velocity[peak_idx]`

**Note**: Onset/offset detection can be affected by drift and noise. The current implementation uses fixed thresholds; improvements could include adaptive local thresholds or position-based validation.

### Stage 5: Segment Extraction

**Purpose**: Extract time windows around each detected saccade for detailed analysis.

**Algorithm** (for each saccade):
1. Define extraction window:
   - Start: `start_time - pre_saccade_window_time` (default: 150ms before onset)
   - End: `end_time + post_saccade_window_time` (default: 500ms after offset)
2. Extract data from full DataFrame:
   - Position (raw): `X_raw`
   - Velocity (smoothed): `vel_x_smooth`
   - Time: `Seconds`
3. Calculate relative time: `Time_rel_threshold = Seconds - peak_time`
   - Zero corresponds to threshold crossing (peak velocity)
4. Store metadata:
   - `saccade_id`: Unique identifier
   - `saccade_direction`: 'upward' or 'downward'
   - `saccade_amplitude`: Initial amplitude (will be recalculated after baselining)
   - `saccade_displacement`: Signed displacement
   - `saccade_peak_velocity`: Peak velocity
   - `saccade_duration`: Duration
   - `is_saccade_period`: Boolean mask for saccade period (onset to offset)

**Result**: List of segment DataFrames, one per detected saccade.

### Stage 6: Duration Filtering

**Purpose**: Exclude segments that are too short (likely truncated at recording edges or contain data gaps).

**Algorithm**:
1. For each segment, calculate actual duration: `segment_duration = max(Time_rel_threshold) - min(Time_rel_threshold)`
2. Compare to `min_saccade_duration` (default: 200ms)
3. If `segment_duration < min_saccade_duration`:
   - Exclude segment
   - Track saccade ID for removal from DataFrames
4. Update `upward_saccades_df` and `downward_saccades_df` to remove filtered saccades

**Note**: This filters based on extracted segment duration (pre + post window), not saccade duration itself. Typical segments are ~650ms (150ms before + 500ms after), so 200ms is conservative.

### Stage 7: Baselining

**Purpose**: Remove pre-saccade position offsets by subtracting baseline value, centering all saccades at zero.

**Algorithm** (for each segment):
1. Define baseline window:
   - Start: `baseline_window_start_time` (default: -100ms relative to threshold)
   - End: `baseline_window_end_time` (default: -20ms relative to threshold)
   - **Critical**: Only use points BEFORE threshold crossing (`Time_rel_threshold < 0`)
2. Extract position values in baseline window (filter NaN values)
3. Calculate baseline: `baseline_value = mean(positions_in_window)`
4. Apply baselining: `X_smooth_baselined = X_raw - baseline_value`
5. Recalculate amplitude from baselined position:
   - Use `is_saccade_period` mask to identify saccade period
   - `amplitude = abs(max_position - min_position)` within saccade period
   - Update `saccade_amplitude` in segment metadata

**Fallback Logic**: If baseline window is empty or contains only NaN:
- Use any available pre-threshold points
- Prefer points closest to `baseline_window_end_time`
- Use median or last valid point if mean fails
- Fallback to zero baseline if no pre-threshold data exists

**Result**: All segments have `X_smooth_baselined` column with zero-centered position, and updated amplitude values.

### Stage 8: Outlier Filtering

**Purpose**: Remove spurious detections using statistical criteria and direction validation.

**Algorithm**:
1. **Calculate IQR-based thresholds** (using 3×IQR for lenient filtering):
   - Amplitude: `Q1 - 3×IQR` to `Q3 + 3×IQR`
   - Max absolute position: `Q3 + 3×IQR`
   - Max absolute velocity: `Q3 + 3×IQR`

2. **Direction validation** (for each segment):
   - **Primary check**: Peak velocity sign must match expected direction
     - Upward: `peak_velocity > 0`
     - Downward: `peak_velocity < 0`
   - **Secondary check**: Displacement sign must match expected direction
     - Uses baselined displacement if available (more accurate)
     - Falls back to raw displacement
   - **Agreement check**: If peak velocity and displacement disagree, flag as wrong direction

3. **Outlier classification**: Segment is outlier if:
   - Amplitude outside IQR bounds, OR
   - Max position > threshold, OR
   - Max velocity > threshold, OR
   - Wrong direction (peak velocity or displacement mismatch)

4. **Store metadata**: Outlier segments include detailed reasons (amplitude, position, velocity, wrong_direction)

**Result**: Filtered segments (accepted) and outlier segments (excluded), with metadata for QC.

### Stage 9: Optional Classification

**Purpose**: Classify saccades as orienting vs compensatory (see separate classification documentation).

**Algorithm**: See `SACCADE_CLASSIFICATION_DOCUMENTATION.md`

---

## User-Settable Parameters

### Detection Threshold Parameters

#### `k` (default: 5)
- **Type**: Float
- **Purpose**: Multiplier for adaptive velocity threshold calculation
- **Effect**:
  - **Higher values** (e.g., 6-8): Stricter threshold → fewer detections, higher precision
  - **Lower values** (e.g., 3-4): More lenient threshold → more detections, may include noise
- **Typical range**: 3-8 (3-6 works well for most cases)
- **Location**: Cell 2 (`k1`, `k2` for VideoData1 and VideoData2) and TEMPORARY FOR DEBUGGING cell
- **Note**: Can be set separately for each video to account for different noise levels

#### `refractory_period` (default: 0.1 seconds)
- **Type**: Float (seconds)
- **Purpose**: Minimum time between consecutive saccades (prevents double-detection)
- **Effect**:
  - **Larger values**: More separation required → fewer detections, prevents rapid sequences
  - **Smaller values**: Allows closer saccades → more detections, may detect rapid sequences
- **Typical range**: 0.05 - 0.15 seconds
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell
- **Note**: Typical saccades last 20-100ms, so 100ms refractory period is conservative

#### `onset_offset_fraction` (default: 0.2)
- **Type**: Float (fraction, 0-1)
- **Purpose**: Fraction of peak velocity threshold used for onset detection
- **Effect**:
  - **Higher values** (e.g., 0.3-0.4): Stricter onset detection → shorter saccades, may miss early movement
  - **Lower values** (e.g., 0.1-0.15): More sensitive onset detection → longer saccades, catches early movement
- **Typical range**: 0.15 - 0.3
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell
- **Note**: Offset detection uses full threshold (1.0), onset uses fraction for sensitivity

#### `peak_width_time` (default: 0.01 seconds = 10ms)
- **Type**: Float (seconds)
- **Purpose**: Minimum peak width for `find_peaks` (filters very narrow peaks, likely noise)
- **Effect**:
  - **Larger values** (e.g., 15-20ms): Filter more peaks → fewer detections, removes noise spikes
  - **Smaller values** (e.g., 5ms): Allow narrower peaks → more detections, may include noise
- **Typical range**: 0.005 - 0.02 seconds (5-20ms)
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell
- **Warning**: Values < 5ms may filter valid saccades; typical saccade peaks are 5-20ms wide

### Preprocessing Parameters

#### `smoothing_window_time` (default: 0.08 seconds = 80ms)
- **Type**: Float (seconds)
- **Purpose**: Time window for rolling median filter applied to position before velocity calculation
- **Effect**:
  - **Larger values** (e.g., 100-150ms): More smoothing → smoother velocity, may blur rapid movements
  - **Smaller values** (e.g., 50-60ms): Less smoothing → more responsive, may retain noise
- **Typical range**: 0.05 - 0.15 seconds
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell
- **Note**: Applied to position before differentiation; critical for noise reduction

### Segment Extraction Parameters

#### `pre_saccade_window_time` (default: 0.15 seconds = 150ms)
- **Type**: Float (seconds)
- **Purpose**: Time before saccade onset to extract in segments
- **Effect**:
  - **Larger values**: More pre-saccade context → better baselining, larger segments
  - **Smaller values**: Less pre-saccade context → smaller segments, faster processing
- **Typical range**: 0.1 - 0.3 seconds
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

#### `post_saccade_window_time` (default: 0.5 seconds = 500ms)
- **Type**: Float (seconds)
- **Purpose**: Time after saccade offset to extract in segments
- **Effect**:
  - **Larger values**: More post-saccade context → better analysis of post-saccade behavior
  - **Smaller values**: Less post-saccade context → smaller segments, faster processing
- **Typical range**: 0.3 - 0.8 seconds
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

### Baselining Parameters

#### `baseline_window_start_time` (default: -0.1 seconds = -100ms)
- **Type**: Float (seconds, negative = before threshold)
- **Purpose**: Start time of baseline window relative to threshold crossing
- **Effect**:
  - **More negative** (e.g., -0.15): Earlier baseline → captures longer-term drift
  - **Less negative** (e.g., -0.06): Later baseline → focuses on immediate pre-saccade period
- **Typical range**: -0.15 to -0.06 seconds
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell
- **Note**: Must be < `baseline_window_end_time` (more negative)

#### `baseline_window_end_time` (default: -0.02 seconds = -20ms)
- **Type**: Float (seconds, negative = before threshold)
- **Purpose**: End time of baseline window relative to threshold crossing
- **Effect**:
  - **More negative** (e.g., -0.05): Earlier end → avoids saccade onset contamination
  - **Less negative** (e.g., -0.01): Later end → closer to threshold, may capture early saccade movement
- **Typical range**: -0.05 to -0.01 seconds
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell
- **Note**: Must be > `baseline_window_start_time` (less negative, closer to threshold)

### Filtering Parameters

#### `min_saccade_duration` (default: 0.2 seconds = 200ms)
- **Type**: Float (seconds)
- **Purpose**: Minimum extracted segment duration (filters truncated segments at recording edges)
- **Effect**:
  - **Larger values**: Stricter filtering → fewer segments, removes edge cases
  - **Smaller values**: More lenient filtering → more segments, may include truncated data
- **Typical range**: 0.15 - 0.3 seconds
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell
- **Note**: Filters based on extracted segment duration (pre + post window), not saccade duration. Typical segments are ~650ms, so 200ms is conservative.

---

## Fine-Tuning Guide

### Diagnostic Steps

1. **Run detection with `verbose=True`** to see:
   - Adaptive threshold value and statistics
   - Number of detected saccades (upward and downward)
   - Mean velocity, duration, and amplitude statistics

2. **Examine the detection visualization**:
   - "VISUALIZE DETECTED SACCADES" plot shows detected saccades overlaid on position/velocity traces
   - Check if detections match visual inspection
   - Look for missed saccades (under-detection) or false positives (over-detection)

3. **Review outlier filtering results**:
   - "OUTLIER FILTERING QC" plot shows accepted (blue) vs excluded (red) saccades
   - Check if excluded saccades are truly outliers or valid detections

4. **Examine segment quality**:
   - "VISUALIZE ALL SACCADES" plot shows all segments aligned by threshold crossing
   - Check for proper baselining (segments should start near zero)
   - Check for consistent segment durations

### Common Issues and Solutions

#### Issue 1: Under-Detection (Too Few Saccades)

**Symptoms**: Many obvious saccades are not detected

**Possible Causes & Solutions**:
- **Cause**: `k` too high (threshold too strict)
  - **Solution**: Decrease `k` (e.g., 5 → 4 or 3)
- **Cause**: `peak_width_time` too large (filtering valid peaks)
  - **Solution**: Decrease `peak_width_time` (e.g., 0.01 → 0.005 seconds)
- **Cause**: `refractory_period` too large (preventing rapid sequences)
  - **Solution**: Decrease `refractory_period` (e.g., 0.1 → 0.05 seconds)
- **Cause**: Velocity too noisy (smoothing insufficient)
  - **Solution**: Increase `smoothing_window_time` (e.g., 0.08 → 0.12 seconds)

#### Issue 2: Over-Detection (Too Many False Positives)

**Symptoms**: Many noise spikes or small movements detected as saccades

**Possible Causes & Solutions**:
- **Cause**: `k` too low (threshold too lenient)
  - **Solution**: Increase `k` (e.g., 5 → 6 or 7)
- **Cause**: `peak_width_time` too small (allowing noise spikes)
  - **Solution**: Increase `peak_width_time` (e.g., 0.01 → 0.015 seconds)
- **Cause**: Velocity too noisy (insufficient smoothing)
  - **Solution**: Increase `smoothing_window_time` (e.g., 0.08 → 0.12 seconds)
- **Cause**: Outlier filtering too lenient
  - **Solution**: Check outlier filtering results; may need to adjust IQR multiplier (currently 3×IQR)

#### Issue 3: Incorrect Onset/Offset Detection

**Symptoms**: Saccade boundaries don't match visual inspection (too early/late start or end)

**Possible Causes & Solutions**:
- **Cause**: `onset_offset_fraction` inappropriate
  - **Solution**: Adjust `onset_offset_fraction` (e.g., 0.2 → 0.15 for earlier onset, or 0.25 for later onset)
- **Cause**: Drift affecting threshold detection
  - **Solution**: Consider adaptive local thresholds (not currently implemented)
- **Cause**: Low sampling rate (velocity only briefly above threshold)
  - **Solution**: Current implementation handles this by using full threshold for offset; may need position-based validation

#### Issue 4: Poor Baselining

**Symptoms**: Segments don't start at zero after baselining, or baseline window captures saccade movement

**Possible Causes & Solutions**:
- **Cause**: Baseline window too close to threshold (captures early saccade movement)
  - **Solution**: Make `baseline_window_end_time` more negative (e.g., -0.02 → -0.05 seconds)
- **Cause**: Baseline window too early (captures unrelated drift)
  - **Solution**: Make `baseline_window_start_time` less negative (e.g., -0.1 → -0.06 seconds)
- **Cause**: Insufficient pre-saccade data
  - **Solution**: Increase `pre_saccade_window_time` (e.g., 0.15 → 0.2 seconds)

#### Issue 5: Wrong Direction Detections

**Symptoms**: Saccades detected in wrong direction (e.g., upward peak detected as downward)

**Possible Causes & Solutions**:
- **Cause**: Noise causing peak detection errors
  - **Solution**: Increase smoothing (`smoothing_window_time`) or increase `peak_width_time`
- **Cause**: Drift affecting peak detection
  - **Solution**: Outlier filtering should catch these; check if direction validation is working
- **Note**: Outlier filtering includes direction validation that should catch most wrong-direction detections

#### Issue 6: Segments Truncated at Edges

**Symptoms**: Many segments excluded by duration filtering

**Possible Causes & Solutions**:
- **Cause**: `min_saccade_duration` too strict
  - **Solution**: Decrease `min_saccade_duration` (e.g., 0.2 → 0.15 seconds)
- **Cause**: Recording too short or many gaps
  - **Solution**: This is expected behavior; segments at edges will be filtered
- **Note**: This filtering is intentional to ensure complete segments for analysis

### Step-by-Step Tuning Process

1. **Start with default parameters** and run detection
2. **Examine detection statistics**:
   - Check threshold value (should be reasonable for your data)
   - Check detection counts (compare to visual inspection)
3. **Identify primary issue**:
   - Too few detections? → Decrease `k` or `peak_width_time`
   - Too many detections? → Increase `k` or `peak_width_time`
   - Incorrect boundaries? → Adjust `onset_offset_fraction`
   - Poor baselining? → Adjust baseline window parameters
4. **Make incremental changes** (e.g., ±10-20% of current value)
5. **Re-run detection** and compare results
6. **Check outlier filtering**: Review excluded saccades to ensure they're truly outliers
7. **Iterate** until detection matches visual inspection

### Parameter Interaction

**Important**: Parameters interact with each other:
- **`k` and `smoothing_window_time`**: More smoothing → lower noise → can use lower `k`
- **`peak_width_time` and `refractory_period`**: Both affect minimum separation between detections
- **`baseline_window_*` and `pre_saccade_window_time`**: Baseline window must be within pre-saccade window
- **`min_saccade_duration` and extraction windows**: Must be less than `pre_saccade_window_time + post_saccade_window_time`

---

## Suggested Improvements and Fixes

### 1. **Adaptive Local Thresholds for Onset/Offset**

**Current Issue**: Fixed threshold fraction (`onset_offset_fraction`) may not work well with varying noise levels or drift.

**Suggested Fix**: 
- Calculate local noise level around each saccade
- Use adaptive threshold based on local statistics
- Consider position-based validation (ensure detected boundaries capture true position change)

**Implementation**: Modify onset/offset detection to use local adaptive thresholds or position-based criteria.

### 2. **Position-Based Onset/Offset Validation**

**Current Issue**: Onset/offset detection relies solely on velocity thresholds, which can be affected by drift.

**Suggested Fix**:
- After velocity-based detection, validate using position change
- Ensure detected boundaries capture significant position change (e.g., > 5% of amplitude)
- Extend boundaries if position change is insufficient

**Implementation**: Add position-based validation step after velocity-based onset/offset detection.

### 3. **Multi-Scale Peak Detection**

**Current Issue**: Single `peak_width_time` may miss saccades of different sizes or filter valid peaks.

**Suggested Fix**:
- Use multi-scale peak detection (detect peaks at multiple width scales)
- Combine results from different scales
- Prefer narrower peaks for small saccades, wider peaks for large saccades

**Implementation**: Run `find_peaks` at multiple `width` values and merge results intelligently.

### 4. **Velocity Profile Shape Analysis**

**Current Issue**: Only peak velocity is used; velocity profile shape may distinguish saccades from artifacts.

**Suggested Fix**:
- Analyze velocity profile shape (acceleration, deceleration phases)
- Use profile characteristics to validate detections
- Filter detections with abnormal profiles (e.g., asymmetric, multiple peaks)

**Implementation**: Extract velocity profile features (time to peak, deceleration rate, symmetry) and use for validation.

### 5. **Amplitude-Based Filtering**

**Current Issue**: Amplitude filtering happens after detection; could be used earlier to prevent false positives.

**Suggested Fix**:
- Add minimum amplitude threshold to peak detection
- Filter peaks with insufficient amplitude change
- Use adaptive amplitude threshold based on noise level

**Implementation**: Add amplitude check during peak detection or immediately after onset/offset detection.

### 6. **Direction Consistency Validation**

**Current Issue**: Direction validation happens during outlier filtering; could catch errors earlier.

**Suggested Fix**:
- Validate direction immediately after peak detection
- Check peak velocity sign matches expected direction
- Filter wrong-direction peaks before segment extraction

**Implementation**: Add direction validation step after peak detection, before segment extraction.

### 7. **Refractory Period Adaptation**

**Current Issue**: Fixed refractory period may not work well for rapid sequences or isolated saccades.

**Suggested Fix**:
- Use adaptive refractory period based on saccade duration
- Shorter refractory for small saccades, longer for large saccades
- Consider saccade amplitude in refractory calculation

**Implementation**: Calculate refractory period dynamically based on detected saccade properties.

### 8. **Smoothing Strategy Refinement**

**Current Issue**: Single smoothing window may blur rapid movements or retain noise.

**Suggested Fix**:
- Use adaptive smoothing based on local signal characteristics
- Apply different smoothing for position vs velocity
- Consider using Savitzky-Golay filter for better edge preservation

**Implementation**: Implement adaptive smoothing or multi-stage smoothing strategy.

### 9. **Baseline Window Optimization**

**Current Issue**: Fixed baseline window may not work well for all saccades (e.g., rapid sequences).

**Suggested Fix**:
- Use adaptive baseline window based on pre-saccade velocity
- Avoid baseline window if pre-saccade drift is detected
- Use position-based baseline selection (find stable period)

**Implementation**: Add adaptive baseline window selection based on pre-saccade signal characteristics.

### 10. **Outlier Filtering Improvements**

**Current Issues**:
- IQR-based filtering may not work well for skewed distributions
- Direction validation could be more robust
- No consideration of saccade quality metrics

**Suggested Fix**:
- Use robust statistics (median absolute deviation) instead of IQR
- Add saccade quality metrics (e.g., velocity profile smoothness, position trajectory consistency)
- Use machine learning for outlier detection (train on manually labeled data)

**Implementation**: Enhance outlier filtering with robust statistics, quality metrics, or ML-based detection.

### 11. **Edge Case Handling**

**Current Issues**:
- First/last saccades may have incomplete windows
- Segments at recording edges are filtered but not handled gracefully
- Missing data (NaN) handling could be improved

**Suggested Fix**:
- Add explicit handling for edge cases (use available data, flag incomplete windows)
- Provide warnings for edge cases
- Improve NaN handling in all stages

**Implementation**: Add comprehensive edge case checks and fallback logic throughout detection pipeline.

### 12. **Validation and Testing**

**Current Issue**: No systematic validation against ground truth data.

**Suggested Fix**:
- Add validation mode that compares detections to manual labels
- Calculate accuracy, precision, recall metrics
- Generate detection quality report

**Implementation**: Add `validate_detection` function that accepts ground truth labels and calculates metrics.

### 13. **Real-Time Detection**

**Current Issue**: Detection processes entire recording at once; not suitable for real-time applications.

**Suggested Fix**:
- Implement sliding window approach for real-time detection
- Use buffered processing with minimal delay
- Optimize for low-latency applications

**Implementation**: Refactor detection to work on sliding windows with configurable buffer size.

### 14. **Multi-Dimensional Detection**

**Current Issue**: Only detects horizontal (X) saccades; vertical (Y) saccades ignored.

**Suggested Fix**:
- Extend detection to Y dimension
- Detect 2D saccades (combine X and Y)
- Classify saccade direction in 2D space (up, down, left, right, diagonal)

**Implementation**: Add Y-dimension detection and 2D saccade classification.

---

## Summary

The saccade detection pipeline uses adaptive statistical thresholds and peak detection to identify rapid eye movements from position and velocity time series. Key stages include preprocessing (smoothing), threshold calculation, peak detection, onset/offset determination, segment extraction, duration filtering, baselining, and outlier filtering. Parameters control detection sensitivity, segment extraction, baselining, and filtering. Fine-tuning involves adjusting thresholds based on detection statistics and visual inspection. Future improvements could include adaptive local thresholds, position-based validation, multi-scale detection, velocity profile analysis, and machine learning approaches.

