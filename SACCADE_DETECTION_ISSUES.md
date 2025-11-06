# Saccade Detection: Code Analysis and Potential Issues

## Overview

This document provides a detailed analysis of potential issues, robustness concerns, and suggested improvements for the saccade detection algorithm implemented in `sleap/saccade_processing.py`.

---

## Critical Issues

### 1. NaN and Invalid Data Handling

#### Issue 1.1: Velocity Computation with NaN Values
**Location**: `analyze_eye_video_saccades()` lines 812-814

```python
df['dt'] = df['Seconds'].diff()
df['vel_x_original'] = df['Ellipse.Center.X'].diff() / df['dt']
df['vel_x_smooth'] = df['X_smooth'].diff() / df['dt']
```

**Problems**:
- If `dt` contains zeros (duplicate timestamps), division by zero creates `inf` values
- If `dt` contains NaN values, division results in NaN
- If `Ellipse.Center.X` or `X_smooth` contain NaN, `diff()` propagates NaN
- NaN values in velocity column will cause `find_peaks()` to fail

**Impact**: High - Will cause detection to fail completely if data has gaps or duplicate timestamps

**Recommendation**:
```python
df['dt'] = df['Seconds'].diff()
# Handle zero or invalid dt values
df['dt'] = df['dt'].replace(0, np.nan)
df['dt'] = df['dt'].fillna(df['dt'].median())  # Fill with median if multiple NaN

# Compute velocity with NaN handling
df['vel_x_smooth'] = df['X_smooth'].diff() / df['dt']
# Remove NaN velocities before threshold calculation
df['vel_x_smooth'] = df['vel_x_smooth'].replace([np.inf, -np.inf], np.nan)
```

#### Issue 1.2: Threshold Calculation with Empty Data
**Location**: `detect_saccades_adaptive()` line 49-50

```python
abs_vel = df[velocity_col].abs().dropna()
vel_thresh = abs_vel.mean() + k * abs_vel.std()
```

**Problems**:
- If `abs_vel` is empty after `dropna()`, `mean()` and `std()` will return NaN
- If `std()` is 0 (constant velocity), threshold = mean, which may be too low
- If all velocity values are NaN, entire detection fails

**Impact**: High - Will cause silent failures or incorrect thresholds

**Recommendation**:
```python
abs_vel = df[velocity_col].abs().dropna()
if len(abs_vel) == 0:
    raise ValueError(f"No valid velocity data in {velocity_col}")
if len(abs_vel) < 10:
    warnings.warn(f"Very few velocity samples ({len(abs_vel)}), threshold may be unreliable")

vel_std = abs_vel.std()
if vel_std == 0 or np.isnan(vel_std):
    # Fallback: use fixed threshold based on mean
    vel_thresh = abs_vel.mean() * 2.0  # Or raise error
    warnings.warn("Zero variance in velocity, using fixed threshold")
else:
    vel_thresh = abs_vel.mean() + k * vel_std
```

#### Issue 1.3: Peak Detection with NaN Values
**Location**: `detect_saccades_adaptive()` lines 59-72

**Problems**:
- `find_peaks()` will fail if velocity column contains NaN values
- No explicit NaN removal before peak detection

**Impact**: High - Detection will crash

**Recommendation**:
```python
# Remove NaN values before peak detection
vel_clean = df[velocity_col].fillna(method='bfill').fillna(method='ffill')
# Or use interpolation
# vel_clean = df[velocity_col].interpolate(method='linear')

pos_peaks, pos_properties = find_peaks(
    vel_clean,
    height=vel_thresh,
    distance=int(fps * refractory_period),
    width=peak_width
)
```

---

### 2. Edge Case Handling

#### Issue 2.1: Onset/Offset Detection Boundary Conditions
**Location**: `detect_saccades_adaptive()` lines 85-95

```python
# Find onset (go backward)
while start_idx > 0 and abs(df.iloc[start_idx][velocity_col]) > vel_thresh * onset_offset_fraction:
    start_idx -= 1

# Find offset (go forward)
while end_idx < len(df) - 1 and abs(df.iloc[end_idx][velocity_col]) > vel_thresh:
    end_idx += 1
```

**Problems**:
- If velocity never drops below threshold (e.g., sustained high velocity), `start_idx` stops at 0 and `end_idx` stops at `len(df)-1`
- This creates saccades that span entire recording, which is likely incorrect
- No validation that `start_idx < end_idx`
- No validation that duration is reasonable (e.g., < 1 second)

**Impact**: Medium - Can create spurious long-duration "saccades"

**Recommendation**:
```python
# Add maximum duration check
max_duration = 1.0  # seconds
max_duration_samples = int(fps * max_duration)

# Find onset with max lookback
start_idx = peak_idx
lookback_count = 0
while (start_idx > 0 and 
       abs(df.iloc[start_idx][velocity_col]) > vel_thresh * onset_offset_fraction and
       lookback_count < max_duration_samples):
    start_idx -= 1
    lookback_count += 1

# Find offset with max lookahead
end_idx = peak_idx
lookahead_count = 0
while (end_idx < len(df) - 1 and 
       abs(df.iloc[end_idx][velocity_col]) > vel_thresh and
       lookahead_count < max_duration_samples):
    end_idx += 1
    lookahead_count += 1

# Validate
if end_idx <= start_idx:
    # Invalid saccade - skip
    continue

duration = df.iloc[end_idx][time_col] - df.iloc[start_idx][time_col]
if duration > max_duration:
    # Saccade too long - likely not a saccade
    continue
```

#### Issue 2.2: FPS Calculation Edge Cases
**Location**: `detect_saccades_adaptive()` line 46

```python
if fps is None:
    fps = 1 / df[time_col].diff().mean()
```

**Problems**:
- If all time differences are identical, this works
- If time differences vary significantly, mean may not represent true FPS
- If `diff().mean()` is 0 or NaN, division fails
- Uses mean instead of median (median is more robust to outliers)

**Impact**: Medium - Can cause incorrect refractory period calculations

**Recommendation**:
```python
if fps is None:
    dt_values = df[time_col].diff().dropna()
    if len(dt_values) == 0:
        raise ValueError("Cannot calculate FPS: no valid time differences")
    dt_median = dt_values.median()
    if dt_median <= 0 or np.isnan(dt_median):
        raise ValueError(f"Invalid median dt: {dt_median}")
    fps = 1 / dt_median
```

#### Issue 2.3: Refractory Period Distance Calculation
**Location**: `detect_saccades_adaptive()` line 62

```python
distance=int(fps * refractory_period)
```

**Problems**:
- If `fps * refractory_period < 1`, `int()` rounds to 0
- Distance of 0 allows peaks at adjacent samples, defeating refractory period purpose
- No validation that distance is reasonable

**Impact**: Medium - Can cause double-counting of single saccades

**Recommendation**:
```python
distance = int(fps * refractory_period)
if distance < 1:
    distance = 1  # Minimum 1 sample separation
    warnings.warn(f"Refractory period {refractory_period}s too short for fps {fps:.1f}, using minimum distance=1")
```

---

### 3. Data Quality and Validation

#### Issue 3.1: Missing Input Validation
**Location**: Multiple functions

**Problems**:
- No validation that required columns exist in DataFrame
- No validation that DataFrame is not empty
- No validation that parameters are within reasonable ranges
- No type checking

**Impact**: Medium - Can cause cryptic errors

**Recommendation**:
```python
def detect_saccades_adaptive(df, ...):
    # Input validation
    if df.empty:
        raise ValueError("DataFrame is empty")
    if position_col not in df.columns:
        raise ValueError(f"Column '{position_col}' not found")
    if velocity_col not in df.columns:
        raise ValueError(f"Column '{velocity_col}' not found")
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found")
    
    # Parameter validation
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if refractory_period <= 0:
        raise ValueError(f"refractory_period must be positive, got {refractory_period}")
    if not (0 < onset_offset_fraction <= 1):
        raise ValueError(f"onset_offset_fraction must be in (0, 1], got {onset_offset_fraction}")
```

#### Issue 3.2: NaN Propagation in Amplitude Calculation
**Location**: `detect_saccades_adaptive()` lines 101-103

```python
start_x = df.iloc[start_idx][position_col]
end_x = df.iloc[end_idx][position_col]
amplitude = abs(end_x - start_x)
```

**Problems**:
- If `start_x` or `end_x` are NaN, amplitude becomes NaN
- No check for NaN before storing in results

**Impact**: Medium - NaN amplitudes cause downstream issues

**Recommendation**:
```python
start_x = df.iloc[start_idx][position_col]
end_x = df.iloc[end_idx][position_col]

if pd.isna(start_x) or pd.isna(end_x):
    # Skip this saccade if position data is invalid
    continue

amplitude = abs(end_x - start_x)
```

---

### 4. Segment Extraction Issues

#### Issue 4.1: Time Matching with Duplicate Timestamps
**Location**: `extract_saccade_segment()` line 228

```python
start_pos = int(np.argmin(np.abs(df[time_col].to_numpy() - start_time)))
```

**Problems**:
- If there are duplicate timestamps, `argmin()` returns the first match
- This may not be the correct frame if duplicates exist
- No validation that match is close enough (could match wrong time if data has gaps)

**Impact**: Medium - Can extract wrong segments

**Recommendation**:
```python
time_diffs = np.abs(df[time_col].to_numpy() - start_time)
start_pos = int(np.argmin(time_diffs))
min_diff = time_diffs[start_pos]

# Validate match is close enough (e.g., within 1 frame)
dt_expected = df[time_col].diff().median()
if min_diff > dt_expected * 1.5:
    # Match too far - likely data gap issue
    excluded_segments.append({
        'saccade_id': idx,
        'reason': 'time_match_too_far',
        'min_diff': min_diff,
        ...
    })
    continue
```

#### Issue 4.2: Hardcoded Fallback FPS
**Location**: `extract_saccade_segment()` line 218

```python
else:
    dt_global = 0.0167
```

**Problems**:
- Hardcoded 60 Hz assumption may not match actual data
- If `dt_global` calculation fails, silent fallback to wrong value
- No warning when fallback is used

**Impact**: Low-Medium - Can cause incorrect validation

**Recommendation**:
```python
if len(df) > 1:
    dt_global = df[time_col].diff().median()
    if pd.isna(dt_global) or dt_global <= 0:
        dt_global = df[time_col].diff().mean()
        if pd.isna(dt_global) or dt_global <= 0:
            dt_global = 0.0167  # Fallback to 60 Hz
            warnings.warn("Using fallback dt_global=0.0167s (60 Hz)")
else:
    dt_global = 0.0167
    warnings.warn("Single-row DataFrame, using fallback dt_global=0.0167s (60 Hz)")
```

#### Issue 4.3: Segment Validation Logic
**Location**: `extract_saccade_segment()` lines 253-256

```python
if time_span > max_allowed_time_span:
    is_valid = False  # Time span too large (likely data gaps)
elif len(segment) < expected_total_points * 0.9:
    is_valid = False  # Too few points (likely truncated at edges or data gaps)
```

**Problems**:
- If `dt_global` is wrong (from fallback), `max_allowed_time_span` is wrong
- 90% threshold is arbitrary - may exclude valid segments at edges
- No check for NaN values in segment

**Impact**: Low-Medium - May exclude valid segments or include invalid ones

**Recommendation**:
```python
# Check for NaN values in critical columns
if segment[position_col].isna().any() or segment[time_col].isna().any():
    is_valid = False
    exclusion_reason = 'contains_nan'

# Use more lenient threshold for edge cases
if pre_start == 0 or post_end == len(df) - 1:
    # Edge case - use more lenient validation
    min_points_threshold = expected_total_points * 0.7  # 70% instead of 90%
else:
    min_points_threshold = expected_total_points * 0.9

if len(segment) < min_points_threshold:
    is_valid = False
    exclusion_reason = 'insufficient_points'
```

---

### 5. Baselining Issues

#### Issue 5.1: Index Label vs Position Confusion
**Location**: `baseline_saccade_segments()` lines 314-315

```python
threshold_idx_in_seg = segment[time_rel_col].abs().idxmin()
threshold_pos_in_seg = segment.index.get_loc(threshold_idx_in_seg)
```

**Problems**:
- `idxmin()` returns the index label, not positional index
- If DataFrame has non-numeric or non-sequential index, `get_loc()` may fail
- If multiple rows have same minimum value, `idxmin()` returns first match (may not be correct)

**Impact**: Medium - Can cause incorrect baselining

**Recommendation**:
```python
# Use positional index instead of label
time_rel_values = segment[time_rel_col].abs().values
threshold_pos_in_seg = np.argmin(time_rel_values)

# Verify we found a reasonable match
if time_rel_values[threshold_pos_in_seg] > 0.1:  # Threshold crossing should be close to 0
    warnings.warn(f"Threshold crossing match may be incorrect: {time_rel_values[threshold_pos_in_seg]}")
```

#### Issue 5.2: NaN in Baseline Window
**Location**: `baseline_saccade_segments()` line 327

```python
baseline_value = segment.iloc[baseline_indices][position_col].mean()
```

**Problems**:
- If all baseline window values are NaN, `mean()` returns NaN
- NaN baseline propagates to all baselined values
- No fallback strategy

**Impact**: Medium - Can corrupt baselining

**Recommendation**:
```python
baseline_window = segment.iloc[baseline_indices][position_col]
baseline_window_clean = baseline_window.dropna()

if len(baseline_window_clean) == 0:
    # All NaN - use zero baseline or skip
    baseline_value = 0.0
    warnings.warn(f"All baseline values NaN, using zero baseline")
elif len(baseline_window_clean) < baseline_n_points * 0.5:
    # Too few valid points - use available or warn
    baseline_value = baseline_window_clean.mean()
    warnings.warn(f"Only {len(baseline_window_clean)}/{len(baseline_indices)} baseline points valid")
else:
    baseline_value = baseline_window_clean.mean()
```

---

### 6. Numerical Stability

#### Issue 6.1: Floating Point Comparison in Threshold Detection
**Location**: `detect_saccades_adaptive()` lines 85-90

**Problems**:
- Direct floating point comparison (`>`) without tolerance
- May miss values very close to threshold due to floating point precision

**Impact**: Low - Usually not an issue, but could cause edge cases

**Recommendation**:
```python
# Use small tolerance for floating point comparison
EPSILON = 1e-10
while start_idx > 0 and abs(df.iloc[start_idx][velocity_col]) > vel_thresh * onset_offset_fraction + EPSILON:
    start_idx -= 1
```

---

### 7. Code Quality Improvements

#### Issue 7.1: Code Duplication
**Location**: `detect_saccades_adaptive()` lines 75-114 and 116-155

**Problems**:
- Nearly identical code for upward and downward saccades
- Duplication increases maintenance burden and bug risk

**Impact**: Low - Code quality issue

**Recommendation**:
```python
def extract_saccade_info(df, peak_idx, vel_thresh, onset_offset_fraction, position_col, time_col):
    """Extract onset, offset, and metrics for a single saccade."""
    peak_time = df.iloc[peak_idx][time_col]
    peak_velocity = df.iloc[peak_idx][velocity_col]
    
    # Find onset and offset (shared logic)
    start_idx, end_idx = find_saccade_bounds(
        df, peak_idx, vel_thresh, onset_offset_fraction, velocity_col
    )
    
    # ... rest of extraction logic
    
    return saccade_dict

# Then use for both directions:
for peak_idx in pos_peaks:
    upward_saccades.append(extract_saccade_info(...))
for peak_idx in neg_peaks:
    downward_saccades.append(extract_saccade_info(...))
```

#### Issue 7.2: Magic Numbers
**Location**: Various locations

**Problems**:
- Hardcoded values like `0.0167` (60 Hz), `0.9` (90% threshold), `1.1` (110% tolerance)
- No explanation or named constants

**Impact**: Low - Code clarity issue

**Recommendation**:
```python
# At module level
DEFAULT_FPS = 60.0
DEFAULT_DT = 1.0 / DEFAULT_FPS  # 0.0167
MIN_SEGMENT_POINTS_FRACTION = 0.9
MAX_TIME_SPAN_TOLERANCE = 1.1
MAX_SACCADE_DURATION = 1.0  # seconds
```

#### Issue 7.3: Missing Error Messages
**Location**: Various locations

**Problems**:
- Silent failures or cryptic errors
- No user-friendly error messages

**Impact**: Low-Medium - Debugging difficulty

**Recommendation**: Add descriptive error messages with context

---

## Robustness Recommendations Summary

### High Priority
1. ✅ Add NaN handling in velocity computation
2. ✅ Validate threshold calculation (empty data, zero variance)
3. ✅ Add input validation (columns, parameters)
4. ✅ Handle NaN values before peak detection
5. ✅ Add maximum duration check for onset/offset detection

### Medium Priority
1. ✅ Improve FPS calculation (use median, validate)
2. ✅ Validate refractory period distance calculation
3. ✅ Check for NaN in amplitude calculation
4. ✅ Improve time matching with validation
5. ✅ Fix baselining index confusion
6. ✅ Handle NaN in baseline window

### Low Priority
1. ✅ Reduce code duplication
2. ✅ Replace magic numbers with named constants
3. ✅ Add floating point tolerance for comparisons
4. ✅ Improve error messages

---

## Testing Recommendations

### Unit Tests Needed
1. Test with all NaN velocity data
2. Test with zero variance velocity
3. Test with duplicate timestamps
4. Test with very short/long recordings
5. Test with saccades at recording edges
6. Test with invalid parameters (negative k, etc.)
7. Test with empty DataFrame
8. Test with missing columns

### Integration Tests Needed
1. Test full pipeline with real data
2. Test with data containing blinks (NaN regions)
3. Test with varying noise levels
4. Test with different FPS values

---

## Performance Considerations

### Current Performance
- No major performance bottlenecks identified
- Rolling median is O(n) per window, acceptable for typical data sizes

### Potential Optimizations
1. Pre-compute velocity threshold once instead of recalculating
2. Use vectorized operations where possible
3. Cache FPS calculation
4. Consider numba JIT for critical loops if performance becomes issue

---

## Conclusion

The detection algorithm is generally well-structured but has several robustness issues related to:
- **NaN handling**: Missing throughout preprocessing and detection
- **Edge cases**: Boundary conditions not fully handled
- **Data validation**: Limited input validation
- **Error handling**: Silent failures and cryptic errors

Addressing the high-priority issues will significantly improve robustness and reliability of the detection system.

