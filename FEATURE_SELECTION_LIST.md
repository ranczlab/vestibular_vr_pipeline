# Feature Selection List for ML-Based Saccade Classification

This document lists all potential features for ML-based saccade classification. **Please review and select which features to include** in the initial implementation. Features are organized by category for easy selection.

---

## Feature Categories

### Category A: Basic Saccade Properties (Currently Extracted)
These features are already available from the existing detection pipeline.

| Feature Name | Description | Units | Notes |
|-------------|-------------|-------|-------|
| `amplitude` | Saccade amplitude (position change) | px | Already extracted |
| `duration` | Saccade duration | s | Already extracted |
| `peak_velocity` | Maximum velocity during saccade | px/s | Already extracted |
| `direction` | Saccade direction | categorical (upward/downward) | Already extracted |
| `start_time` | Saccade start time | s | Already extracted |
| `end_time` | Saccade end time | s | Already extracted |
| `time` | Peak time (threshold crossing) | s | Already extracted |

**Recommendation**: ✅ **Include all** - These are fundamental features.

---

### Category B: Pre-Saccade Features (Currently Extracted)
Features describing the period before saccade onset.

| Feature Name | Description | Units | Notes |
|-------------|-------------|-------|-------|
| `pre_saccade_mean_velocity` | Mean absolute velocity in pre-saccade window | px/s | Already extracted, indicates drift |
| `pre_saccade_position_drift` | Absolute position change in pre-saccade window | px | Already extracted, indicates drift |
| `pre_saccade_position_variance` | Position variance in pre-saccade window | px² | **NEW** - Stability measure |
| `pre_saccade_velocity_std` | Standard deviation of velocity in pre-window | px/s | **NEW** - Velocity variability |
| `pre_saccade_drift_rate` | Position drift normalized by window duration | px/s | **NEW** - Drift rate |
| `pre_saccade_max_velocity` | Maximum velocity in pre-saccade window | px/s | **NEW** - Peak drift velocity |
| `pre_saccade_window_duration` | Actual duration of pre-saccade window | s | **NEW** - May vary if constrained by previous saccade |

**Recommendation**: ✅ **Include existing** + consider `pre_saccade_position_variance` and `pre_saccade_drift_rate`.

---

### Category C: Post-Saccade Features (Currently Extracted)
Features describing the period after saccade offset.

| Feature Name | Description | Units | Notes |
|-------------|-------------|-------|-------|
| `post_saccade_position_variance` | Position variance in post-saccade window | px² | Already extracted, indicates stability |
| `post_saccade_position_change` | Absolute position change in post-saccade window | px | Already extracted |
| `post_saccade_position_change_pct` | Position change as % of saccade amplitude | % | **NEW** - Normalized change |
| `post_saccade_mean_velocity` | Mean absolute velocity in post-saccade window | px/s | **NEW** - Post-saccade movement |
| `post_saccade_velocity_std` | Standard deviation of velocity in post-window | px/s | **NEW** - Velocity variability |
| `post_saccade_fixation_quality` | Binary: variance < threshold (stable fixation) | binary | **NEW** - Fixation indicator |
| `post_saccade_window_duration` | Actual duration of post-saccade window | s | **NEW** - May vary if constrained by next saccade |
| `post_saccade_max_velocity` | Maximum velocity in post-saccade window | px/s | **NEW** - Post-saccade drift |

**Recommendation**: ✅ **Include existing** + consider `post_saccade_position_change_pct` and `post_saccade_mean_velocity`.

---

### Category D: Temporal Context Features
Features describing temporal relationships with other saccades.

| Feature Name | Description | Units | Notes |
|-------------|-------------|-------|-------|
| `time_since_previous_saccade` | Time interval from previous saccade | s | **NEW** - Isolation measure |
| `time_until_next_saccade` | Time interval until next saccade | s | **NEW** - Isolation measure |
| `bout_size` | Number of saccades in bout | count | Already extracted |
| `bout_id` | ID of bout (None if isolated) | categorical | Already extracted |
| `position_in_bout` | Position within bout (1st, 2nd, 3rd, etc.) | count | **NEW** - 1 = first, bout_size = last |
| `is_first_in_bout` | Binary: Is this the first saccade in bout? | binary | **NEW** - Important for Saccade-and-Fixate |
| `is_last_in_bout` | Binary: Is this the last saccade in bout? | binary | **NEW** |
| `is_isolated` | Binary: Is this saccade isolated (bout_size=1)? | binary | **NEW** - Important for Orienting |
| `bout_duration` | Total time span of bout | s | **NEW** - From first to last saccade |
| `bout_density` | Saccades per second in bout | saccades/s | **NEW** - Bout intensity |
| `inter_saccade_interval_mean` | Mean interval between saccades in bout | s | **NEW** - Bout rhythm |
| `inter_saccade_interval_std` | Std of intervals between saccades in bout | s | **NEW** - Bout regularity |

**Recommendation**: ✅ **Include all** - Temporal context is crucial for distinguishing classes.

---

### Category E: Velocity Profile Features
Features describing the velocity profile shape during the saccade.

| Feature Name | Description | Units | Notes |
|-------------|-------------|-------|-------|
| `time_to_peak_velocity` | Time from onset to peak velocity | s | **NEW** - Acceleration phase |
| `time_to_peak_velocity_norm` | Time to peak normalized by duration | ratio | **NEW** - 0-1, shape descriptor |
| `acceleration_phase_duration` | Duration of acceleration phase | s | **NEW** - From onset to peak |
| `deceleration_phase_duration` | Duration of deceleration phase | s | **NEW** - From peak to offset |
| `velocity_asymmetry` | Acceleration duration / deceleration duration | ratio | **NEW** - Shape asymmetry |
| `peak_velocity_ratio` | Peak velocity / mean velocity | ratio | **NEW** - Velocity profile sharpness |
| `velocity_smoothness` | Variance of velocity derivative | (px/s²)² | **NEW** - Profile smoothness |
| `acceleration_max` | Maximum acceleration (derivative of velocity) | px/s² | **NEW** - Acceleration strength |
| `deceleration_max` | Maximum deceleration (absolute value) | px/s² | **NEW** - Deceleration strength |

**Recommendation**: ⚠️ **Selective** - Start with `time_to_peak_velocity_norm` and `velocity_asymmetry`. Add others if needed.

---

### Category F: Position Profile Features
Features describing the position trajectory during and around the saccade.

| Feature Name | Description | Units | Notes |
|-------------|-------------|-------|-------|
| `position_change_rate` | Amplitude / duration | px/s | **NEW** - Average velocity |
| `position_smoothness` | Variance of position derivative (velocity) | (px/s)² | **NEW** - Trajectory smoothness |
| `amplitude_normalized` | Amplitude normalized by recording std | ratio | **NEW** - Relative amplitude |
| `trajectory_curvature` | Curvature of position trajectory | 1/px | **NEW** - Complex to compute |
| `position_at_onset` | Position value at saccade onset | px | **NEW** - Context |
| `position_at_offset` | Position value at saccade offset | px | **NEW** - Context |

**Recommendation**: ⚠️ **Selective** - Start with `position_change_rate`. Others may be redundant.

---

### Category G: Amplitude & Direction Consistency Features
Features describing consistency within bouts.

| Feature Name | Description | Units | Notes |
|-------------|-------------|-------|-------|
| `amplitude_relative_to_bout_mean` | (amplitude - bout_mean) / bout_std | ratio | **NEW** - Normalized amplitude |
| `amplitude_consistency_in_bout` | Std of amplitudes in bout | px | **NEW** - Bout consistency |
| `direction_relative_to_previous` | Same (1) or opposite (-1) direction | categorical | **NEW** - Direction alternation |
| `direction_alternation_in_bout` | Fraction of direction changes in bout | ratio | **NEW** - 0-1, alternation pattern |
| `amplitude_ratio_to_previous` | amplitude / previous_amplitude | ratio | **NEW** - Relative size |
| `amplitude_ratio_to_next` | amplitude / next_amplitude | ratio | **NEW** - Relative size |

**Recommendation**: ✅ **Include** `amplitude_relative_to_bout_mean`, `direction_relative_to_previous`, `amplitude_consistency_in_bout`.

---

### Category H: Rule-Based Classification Features
Features from existing rule-based classification (for reference/fallback).

| Feature Name | Description | Units | Notes |
|-------------|-------------|-------|-------|
| `rule_based_class` | Rule-based classification | categorical | **NEW** - Initial label |
| `rule_based_confidence` | Rule-based confidence score | 0-1 | Already extracted |
| `classification_confidence` | Confidence from rule-based classifier | 0-1 | Already extracted |

**Recommendation**: ✅ **Include** - Useful as features and for fallback.

---

### Category I: Contextual Flags
Binary flags for quick filtering and classification.

| Feature Name | Description | Units | Notes |
|-------------|-------------|-------|-------|
| `has_pre_saccade_drift` | Binary: pre_velocity > threshold OR pre_drift > threshold | binary | **NEW** - Compensatory indicator |
| `has_post_saccade_movement` | Binary: post_change > amplitude * threshold | binary | **NEW** - Compensatory indicator |
| `has_stable_fixation` | Binary: post_variance < threshold | binary | **NEW** - Orienting indicator |
| `is_edge_case` | Binary: First or last saccade in recording | binary | **NEW** - Incomplete windows |

**Recommendation**: ⚠️ **Selective** - May be redundant if using raw features, but useful for interpretability.

---

### Category J: Normalized/Relative Features
Features normalized by amplitude or other factors for robustness.

| Feature Name | Description | Units | Notes |
|-------------|-------------|-------|-------|
| `pre_drift_normalized` | pre_drift / amplitude | ratio | **NEW** - Amplitude-independent drift |
| `post_change_normalized` | post_change / amplitude | ratio | **NEW** - Already computed as percentage |
| `velocity_normalized` | peak_velocity / amplitude | 1/s | **NEW** - Velocity efficiency |
| `duration_normalized` | duration / amplitude | s/px | **NEW** - Duration efficiency |

**Recommendation**: ⚠️ **Selective** - Useful if amplitude varies significantly across recordings.

---

## Feature Selection Summary

### Recommended Core Feature Set (Start Here)

**Total: ~30 features**

#### Must Include (Already Extracted):
- All from Category A (7 features)
- `pre_saccade_mean_velocity`, `pre_saccade_position_drift` (Category B)
- `post_saccade_position_variance`, `post_saccade_position_change` (Category C)
- `bout_size`, `bout_id` (Category D)
- `classification_confidence` (Category H)

#### High Priority New Features:
- **Category D**: `time_since_previous_saccade`, `time_until_next_saccade`, `position_in_bout`, `is_first_in_bout`, `is_isolated`, `bout_density`
- **Category B**: `pre_saccade_position_variance`, `pre_saccade_drift_rate`
- **Category C**: `post_saccade_position_change_pct`, `post_saccade_mean_velocity`
- **Category G**: `amplitude_relative_to_bout_mean`, `direction_relative_to_previous`
- **Category E**: `time_to_peak_velocity_norm`, `velocity_asymmetry`

#### Medium Priority (Add if Needed):
- **Category E**: `acceleration_phase_duration`, `deceleration_phase_duration`
- **Category F**: `position_change_rate`
- **Category I**: `has_pre_saccade_drift`, `has_post_saccade_movement`, `has_stable_fixation`

#### Low Priority (Add Later):
- **Category E**: `velocity_smoothness`, `acceleration_max`, `deceleration_max`
- **Category F**: `position_smoothness`, `trajectory_curvature`
- **Category J**: Normalized features (if amplitude normalization needed)

---

## Feature Engineering Notes

### Handling Missing Values
- **Edge cases**: First/last saccades may have incomplete pre/post windows
  - Use available data, flag with `is_edge_case`
  - For `time_since_previous_saccade`: Use large value (e.g., 999) or recording duration
  - For `time_until_next_saccade`: Use large value or recording duration

### Feature Scaling
- **StandardScaler** recommended for neural networks
- Normalize all features to mean=0, std=1
- Save scaler parameters with model for inference

### Categorical Features
- **Direction**: Encode as 0/1 (upward/downward) or -1/1
- **Bout ID**: Can be used as categorical or ignored (bout_size more useful)
- **Rule-based class**: One-hot encode or use as ordinal

### Feature Interactions
- Neural network will learn interactions automatically
- Consider explicit interaction features if needed:
  - `pre_drift * post_variance` (compensatory indicator)
  - `bout_size * bout_density` (bout intensity)

---

## Next Steps

1. **Review this list** and select features to include
2. **Prioritize features** by category (must-have, nice-to-have, future)
3. **Provide feedback** on:
   - Features to add
   - Features to remove
   - Features to modify
   - Any domain-specific features needed

Once you've selected features, we'll implement the feature extraction function with your chosen feature set.

