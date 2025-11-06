# Saccade Classification: Orienting vs Compensatory

## Overview

This document describes the logic and implementation of saccade classification, which categorizes detected saccades into two types:
- **Orienting saccades**: Voluntary eye movements to explore the environment (typically isolated, stable gaze before and after)
- **Compensatory saccades**: Eye movements that compensate for head/body movement (typically occur in rapid sequences/bouts, show pre-saccade drift)

The classification uses a two-stage approach: (1) temporal clustering to identify saccade bouts, and (2) feature-based classification for isolated saccades.

---

## Classification Logic and Flow

### Stage 1: Temporal Clustering (Bout Detection)

**Purpose**: Identify groups of saccades that occur in rapid succession (bouts), which are strong indicators of compensatory behavior.

**Algorithm**:
1. Sort all saccades chronologically by peak time (`time` column)
2. Iterate through saccades sequentially
3. For each saccade, calculate the time interval from the previous saccade
4. If interval ≤ `bout_window` (default: 1.5 seconds):
   - Add saccade to current bout
5. If interval > `bout_window`:
   - Start a new bout
6. Assign `bout_id` to each saccade and calculate `bout_size` (number of saccades in bout)

**Result**: All saccades are assigned to bouts. Saccades with `bout_size >= 2` are automatically classified as compensatory in Stage 3.

### Stage 2: Feature Extraction

For each saccade, extract four features from the position and velocity time series:

#### Feature 1: Pre-saccade Mean Velocity (`pre_saccade_mean_velocity`)
- **Window**: From `max(previous_saccade_peak_time, start_time - pre_saccade_window)` to `start_time`
- **Calculation**: Mean absolute velocity in this window
- **Interpretation**: High values indicate drift/compensation before saccade

#### Feature 2: Pre-saccade Position Drift (`pre_saccade_position_drift`)
- **Window**: Same as Feature 1
- **Calculation**: Absolute difference between first and last position values
- **Interpretation**: Large drift indicates compensatory movement

#### Feature 3: Post-saccade Position Variance (`post_saccade_position_variance`)
- **Window**: Dynamic window from `end_time` to `min(next_saccade_start_time, end_time + max_intersaccade_interval_for_classification)`
- **Calculation**: Variance of position values in this window
- **Interpretation**: Low variance indicates stable gaze (orienting), high variance indicates continued movement (compensatory)

#### Feature 4: Post-saccade Position Change (`post_saccade_position_change`)
- **Window**: Same as Feature 3
- **Calculation**: Absolute difference between first and last position values
- **Interpretation**: Large change relative to saccade amplitude indicates compensatory behavior

**Note**: The post-saccade window is "dynamic" - it extends until the next saccade starts (if within `max_intersaccade_interval_for_classification`) or uses the maximum interval. This captures the full inter-saccade period for accurate classification.

### Stage 3: Classification Rules

Classification follows a hierarchical decision tree:

```
IF bout_size >= 2:
    → Classify as COMPENSATORY (multiple saccades in rapid succession)
ELSE (isolated saccade):
    Calculate position_change_threshold = amplitude × (post_saccade_position_change_threshold_percent / 100)
    
    IF (pre_saccade_mean_velocity > pre_saccade_velocity_threshold) OR 
       (pre_saccade_position_drift > pre_saccade_drift_threshold):
        → Classify as COMPENSATORY (pre-saccade drift detected)
    
    ELSE IF post_saccade_position_change > position_change_threshold:
        → Classify as COMPENSATORY (eye continues moving after saccade)
    
    ELSE IF (pre_saccade_mean_velocity ≤ pre_saccade_velocity_threshold) AND
            (post_saccade_position_variance < post_saccade_variance_threshold) AND
            (post_saccade_position_change ≤ position_change_threshold):
        → Classify as ORIENTING (stable before and after)
    
    ELSE:
        → Classify as COMPENSATORY (default/conservative)
```

**Key Points**:
- Saccades in bouts are always compensatory
- Isolated saccades require feature-based classification
- The default (else branch) is conservative - uncertain cases are classified as compensatory
- All saccades are classified (no unclassified saccades)

---

## User-Settable Parameters

### Temporal Clustering Parameters

#### `bout_window` (default: 1.5 seconds)
- **Type**: Float (seconds)
- **Purpose**: Maximum time interval between consecutive saccades to be considered part of the same bout
- **Effect**: 
  - **Larger values**: More saccades grouped into bouts → more compensatory classifications
  - **Smaller values**: Fewer bouts → more isolated saccades → more feature-based classifications
- **Typical range**: 1.0 - 2.5 seconds
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

#### `pre_saccade_window` (default: 0.3 seconds)
- **Type**: Float (seconds)
- **Purpose**: Time window before saccade onset to analyze for pre-saccade drift
- **Effect**:
  - **Larger values**: Capture longer-term drift → may detect more compensatory saccades
  - **Smaller values**: Focus on immediate pre-saccade period → more sensitive to short-term drift
- **Typical range**: 0.2 - 0.5 seconds
- **Note**: Window is constrained by previous saccade peak time (if exists) to avoid overlap
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

#### `max_intersaccade_interval_for_classification` (default: 5.0 seconds)
- **Type**: Float (seconds)
- **Purpose**: Maximum duration for post-saccade analysis window (dynamic window extends until next saccade or this maximum)
- **Effect**:
  - **Larger values**: Analyze longer post-saccade periods → better capture slow compensatory movements
  - **Smaller values**: Focus on immediate post-saccade period → faster classification
- **Typical range**: 3.0 - 10.0 seconds
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

### Feature-Based Classification Thresholds

#### `pre_saccade_velocity_threshold` (default: 50.0 px/s)
- **Type**: Float (pixels per second)
- **Purpose**: Velocity threshold for detecting pre-saccade drift
- **Effect**:
  - **Higher values**: Less sensitive to drift → fewer compensatory classifications
  - **Lower values**: More sensitive to drift → more compensatory classifications
- **Typical range**: 30.0 - 100.0 px/s
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

#### `pre_saccade_drift_threshold` (default: 10.0 px)
- **Type**: Float (pixels)
- **Purpose**: Position drift threshold before saccade for compensatory classification
- **Effect**:
  - **Higher values**: Require larger drift → fewer compensatory classifications
  - **Lower values**: Detect smaller drift → more compensatory classifications
- **Typical range**: 5.0 - 20.0 px
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

#### `post_saccade_variance_threshold` (default: 100.0 px²)
- **Type**: Float (pixels squared)
- **Purpose**: Position variance threshold after saccade for orienting classification (low variance = stable = orienting)
- **Effect**:
  - **Higher values**: More lenient stability requirement → more orienting classifications
  - **Lower values**: Stricter stability requirement → fewer orienting classifications
- **Typical range**: 50.0 - 200.0 px²
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

#### `post_saccade_position_change_threshold_percent` (default: 50.0%)
- **Type**: Float (percentage)
- **Purpose**: Position change threshold relative to saccade amplitude for compensatory classification
- **Effect**:
  - **Higher values**: Require larger post-saccade movement → fewer compensatory classifications
  - **Lower values**: Detect smaller post-saccade movement → more compensatory classifications
- **Typical range**: 30.0 - 70.0%
- **Note**: Threshold is calculated as `amplitude × (threshold_percent / 100)`
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

### Adaptive Threshold Parameters

#### `use_adaptive_thresholds` (default: True)
- **Type**: Boolean
- **Purpose**: Enable adaptive threshold calculation from feature distributions
- **Effect**:
  - **True**: Thresholds adapt to data distribution → better performance across different recording conditions
  - **False**: Use fixed thresholds → consistent behavior but may need manual tuning
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

#### `adaptive_percentile_pre_velocity` (default: 75)
- **Type**: Integer (percentile, 0-100)
- **Purpose**: Percentile for adaptive pre-saccade velocity threshold (upper percentile for compensatory detection)
- **Effect**:
  - **Higher percentiles**: Stricter threshold → fewer compensatory classifications
  - **Lower percentiles**: More lenient threshold → more compensatory classifications
- **Typical range**: 70 - 85
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

#### `adaptive_percentile_pre_drift` (default: 75)
- **Type**: Integer (percentile, 0-100)
- **Purpose**: Percentile for adaptive pre-saccade drift threshold (upper percentile for compensatory detection)
- **Effect**:
  - **Higher percentiles**: Stricter threshold → fewer compensatory classifications
  - **Lower percentiles**: More lenient threshold → more compensatory classifications
- **Typical range**: 70 - 85
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

#### `adaptive_percentile_post_variance` (default: 25)
- **Type**: Integer (percentile, 0-100)
- **Purpose**: Percentile for adaptive post-saccade variance threshold (lower percentile for orienting detection - low variance = stable)
- **Effect**:
  - **Lower percentiles**: Stricter stability requirement → fewer orienting classifications
  - **Higher percentiles**: More lenient stability requirement → more orienting classifications
- **Typical range**: 20 - 30
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

### Classification Control

#### `classify_orienting_compensatory` (default: True)
- **Type**: Boolean
- **Purpose**: Enable/disable classification
- **Effect**: If False, classification is skipped and `saccade_type` column is not created
- **Location**: Cell 2 and TEMPORARY FOR DEBUGGING cell

---

## Fine-Tuning Guide

### Diagnostic Steps

1. **Run classification with `debug=True`** to see:
   - Classification statistics (counts, percentages)
   - Feature distributions (if adaptive thresholds enabled)
   - Threshold values used

2. **Examine the diagnostic plots** (generated when `debug=True`):
   - Histograms of pre-saccade velocity, drift, post-saccade variance
   - Current threshold values overlaid on distributions
   - Percentile markers (if adaptive thresholds enabled)

3. **Review the classification visualization**:
   - Time series plot showing orienting (blue) vs compensatory (orange) saccades
   - Check if classifications match visual inspection

### Common Issues and Solutions

#### Issue 1: Too Many Compensatory Classifications

**Symptoms**: Most saccades classified as compensatory, even isolated ones

**Possible Causes & Solutions**:
- **Cause**: `bout_window` too large
  - **Solution**: Decrease `bout_window` (e.g., 1.5 → 1.0 seconds)
- **Cause**: Pre-saccade thresholds too low
  - **Solution**: Increase `pre_saccade_velocity_threshold` (e.g., 50 → 75 px/s) or `pre_saccade_drift_threshold` (e.g., 10 → 15 px)
  - **If using adaptive**: Increase `adaptive_percentile_pre_velocity` or `adaptive_percentile_pre_drift` (e.g., 75 → 80)
- **Cause**: Post-saccade position change threshold too low
  - **Solution**: Increase `post_saccade_position_change_threshold_percent` (e.g., 50 → 60%)
- **Cause**: Post-saccade variance threshold too high (too lenient for orienting)
  - **Solution**: Decrease `post_saccade_variance_threshold` (e.g., 100 → 75 px²)
  - **If using adaptive**: Decrease `adaptive_percentile_post_variance` (e.g., 25 → 20)

#### Issue 2: Too Many Orienting Classifications

**Symptoms**: Saccades that appear compensatory are classified as orienting

**Possible Causes & Solutions**:
- **Cause**: `bout_window` too small
  - **Solution**: Increase `bout_window` (e.g., 1.5 → 2.0 seconds)
- **Cause**: Pre-saccade thresholds too high
  - **Solution**: Decrease `pre_saccade_velocity_threshold` (e.g., 50 → 40 px/s) or `pre_saccade_drift_threshold` (e.g., 10 → 7 px)
  - **If using adaptive**: Decrease `adaptive_percentile_pre_velocity` or `adaptive_percentile_pre_drift` (e.g., 75 → 70)
- **Cause**: Post-saccade position change threshold too high
  - **Solution**: Decrease `post_saccade_position_change_threshold_percent` (e.g., 50 → 40%)
- **Cause**: Post-saccade variance threshold too low (too strict for orienting)
  - **Solution**: Increase `post_saccade_variance_threshold` (e.g., 100 → 150 px²)
  - **If using adaptive**: Increase `adaptive_percentile_post_variance` (e.g., 25 → 30)

#### Issue 3: Misclassification of Isolated Compensatory Saccades

**Symptoms**: Isolated compensatory saccades (with drift) classified as orienting

**Possible Causes & Solutions**:
- **Cause**: Pre-saccade window too short to capture drift
  - **Solution**: Increase `pre_saccade_window` (e.g., 0.3 → 0.4 seconds)
- **Cause**: Pre-saccade thresholds too high
  - **Solution**: Decrease thresholds (see Issue 2)
- **Cause**: Post-saccade window too short
  - **Solution**: Increase `max_intersaccade_interval_for_classification` (e.g., 5.0 → 7.0 seconds)

#### Issue 4: Misclassification of Orienting Saccades in Sequences

**Symptoms**: Rapid orienting saccades (e.g., scanning) incorrectly grouped into bouts

**Possible Causes & Solutions**:
- **Cause**: `bout_window` too large
  - **Solution**: Decrease `bout_window` (e.g., 1.5 → 1.0 seconds)
- **Note**: This is a fundamental limitation - rapid orienting sequences will be classified as compensatory. Consider manual review or additional features (e.g., amplitude consistency) to distinguish.

### Step-by-Step Tuning Process

1. **Start with default parameters** and run classification
2. **Examine diagnostic plots** to understand feature distributions
3. **Identify the primary issue**:
   - Too many compensatory? → Adjust thresholds upward
   - Too many orienting? → Adjust thresholds downward
   - Misclassification of specific cases? → Adjust relevant thresholds
4. **Make incremental changes** (e.g., ±10-20% of current value)
5. **Re-run classification** and compare results
6. **Iterate** until classification matches visual inspection

### Using Adaptive Thresholds

**When to use**: 
- Recording conditions vary across sessions
- Data has different noise levels or signal characteristics
- You want automatic adaptation to data distribution

**How to tune**:
- Adjust percentiles rather than absolute thresholds
- Higher percentiles = stricter thresholds (fewer positives)
- Lower percentiles = more lenient thresholds (more positives)
- Start with defaults (75, 75, 25) and adjust based on diagnostic plots

**When NOT to use**:
- You have specific, well-defined thresholds from literature
- You want consistent behavior across different datasets
- You have very few saccades (<10) - adaptive thresholds may be unreliable

---

## Suggested Improvements and Fixes

### 1. **Direction-Aware Classification**

**Current Issue**: Classification doesn't consider saccade direction (upward vs downward). Compensatory saccades may have direction-specific patterns.

**Suggested Fix**: 
- Add direction-specific thresholds or features
- Consider direction consistency within bouts (compensatory bouts may alternate directions)

**Implementation**: Modify `classify_saccades_orienting_vs_compensatory` to accept direction information and use direction-specific thresholds.

### 2. **Amplitude-Based Features**

**Current Issue**: Classification doesn't use saccade amplitude as a feature, though it's used for relative thresholds.

**Suggested Fix**:
- Add amplitude consistency within bouts (compensatory bouts may have similar amplitudes)
- Use amplitude as a direct feature (orienting saccades may be larger/more variable)

**Implementation**: Calculate amplitude statistics (mean, std) within bouts and use as classification features.

### 3. **Velocity Profile Analysis**

**Current Issue**: Only mean velocity is used; velocity profile shape may distinguish saccade types.

**Suggested Fix**:
- Analyze velocity profile shape (peak velocity, acceleration/deceleration phases)
- Compensatory saccades may have different velocity profiles than orienting

**Implementation**: Extract velocity profile features (peak velocity, time to peak, deceleration rate) and add to classification logic.

### 4. **Post-Saccade Window Refinement**

**Current Issue**: Post-saccade window uses fixed maximum interval; may miss slow compensatory movements or include irrelevant periods.

**Suggested Fix**:
- Use velocity-based window termination (stop when velocity stabilizes)
- Use adaptive window duration based on saccade amplitude/duration

**Implementation**: Add velocity stability detection to dynamically terminate post-saccade window.

### 5. **Confidence Scores**

**Current Issue**: Classification is binary; no confidence measure for uncertain cases.

**Suggested Fix**:
- Calculate confidence scores based on feature agreement
- Flag low-confidence classifications for manual review

**Implementation**: Add `classification_confidence` column (0-1 scale) based on how many features agree with classification.

### 6. **Bout Detection Refinement**

**Current Issue**: Simple time-based bout detection may miss complex patterns.

**Suggested Fix**:
- Consider direction alternation within bouts
- Use amplitude consistency within bouts
- Add minimum bout size requirement (e.g., ≥3 saccades)

**Implementation**: Enhance bout detection with additional criteria beyond time intervals.

### 7. **Feature Normalization**

**Current Issue**: Features are in absolute units (px, px/s); may not scale well across different recording conditions.

**Suggested Fix**:
- Normalize features by saccade amplitude or recording statistics
- Use relative measures (e.g., drift as % of amplitude)

**Implementation**: Add normalization step before classification, or use relative thresholds.

### 8. **Machine Learning Approach**

**Current Issue**: Rule-based classification may not capture complex patterns.

**Suggested Fix**:
- Train a classifier (e.g., random forest, SVM) on manually labeled data
- Use feature importance to guide threshold selection

**Implementation**: Add optional ML-based classification mode with pre-trained model or training capability.

### 9. **Edge Case Handling**

**Current Issues**:
- First/last saccades may have incomplete pre/post windows
- Very short recordings may have insufficient data for adaptive thresholds

**Suggested Fix**:
- Add explicit handling for edge cases (use available data, flag incomplete windows)
- Provide fallback thresholds when adaptive calculation fails

**Implementation**: Add edge case checks and fallback logic throughout classification function.

### 10. **Validation and Testing**

**Current Issue**: No systematic validation against ground truth data.

**Suggested Fix**:
- Add validation mode that compares classifications to manual labels
- Calculate accuracy, precision, recall metrics
- Generate confusion matrix

**Implementation**: Add `validate_classification` function that accepts ground truth labels and calculates metrics.

---

## Summary

The saccade classification system uses temporal clustering and feature-based analysis to distinguish orienting from compensatory saccades. Key parameters control bout detection, feature extraction windows, and classification thresholds. Adaptive thresholds can automatically adjust to data distributions, while fixed thresholds provide consistent behavior. Fine-tuning involves adjusting thresholds based on diagnostic plots and visual inspection of results. Future improvements could include direction-aware features, amplitude analysis, velocity profile analysis, and machine learning approaches.

