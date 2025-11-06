# Feature Extraction Implementation Summary

## Selected Features

Based on your selections, the following features have been implemented:

### ✅ Category A: Basic Saccade Properties (7 features)
- `amplitude` - Saccade amplitude (px)
- `duration` - Saccade duration (s)
- `peak_velocity` - Maximum velocity during saccade (px/s)
- `direction` - Encoded as 1 (upward) or -1 (downward)
- `start_time` - Saccade start time (s)
- `end_time` - Saccade end time (s)
- `time` - Peak time / threshold crossing (s)

### ✅ Category B: Pre-Saccade Features (5 features)
- `pre_saccade_mean_velocity` - Mean absolute velocity in pre-window (px/s) [existing]
- `pre_saccade_position_drift` - Position change in pre-window (px) [existing]
- `pre_saccade_position_variance` - Position variance in pre-window (px²) [NEW]
- `pre_saccade_drift_rate` - Drift normalized by window duration (px/s) [NEW]
- `pre_saccade_window_duration` - Actual duration of pre-window (s) [NEW]

### ✅ Category C: Post-Saccade Features (6 features)
- `post_saccade_position_variance` - Position variance in post-window (px²) [existing]
- `post_saccade_position_change` - Position change in post-window (px) [existing]
- `post_saccade_position_change_pct` - Position change as % of amplitude (%) [NEW]
- `post_saccade_mean_velocity` - Mean absolute velocity in post-window (px/s) [NEW]
- `post_saccade_drift_rate` - Drift rate in post-window (px/s) [NEW - as requested]
- `post_saccade_window_duration` - Actual duration of post-window (s) [NEW]

### ✅ Category D: Temporal Context Features (10 features, excluding bout_density)
- `time_since_previous_saccade` - Time interval from previous saccade (s) [NEW]
- `time_until_next_saccade` - Time interval until next saccade (s) [NEW]
- `bout_size` - Number of saccades in bout (count) [existing]
- `bout_id` - ID of bout (categorical) [existing]
- `position_in_bout` - Position within bout (1st, 2nd, etc.) [NEW]
- `is_first_in_bout` - Binary: Is first saccade in bout? [NEW]
- `is_last_in_bout` - Binary: Is last saccade in bout? [NEW]
- `is_isolated` - Binary: Is isolated (bout_size=1)? [NEW]
- `bout_duration` - Total time span of bout (s) [NEW]
- `inter_saccade_interval_mean` - Mean interval between saccades in bout (s) [NEW]
- `inter_saccade_interval_std` - Std of intervals between saccades in bout (s) [NEW]

**Excluded**: `bout_density` (as requested)

### ⏭️ Category E: Velocity Profile Features
**Status**: Excluded for undersampled data (<120 Hz FPS)
**Note**: Code includes TODO comment to add these features when FPS increases significantly (e.g., to 120 Hz or more)

### ⏭️ Category F: Position Profile Features
**Status**: Excluded for undersampled data (<120 Hz FPS)
**Note**: Code includes TODO comment to add these features when FPS increases significantly (e.g., to 120 Hz or more)

### ✅ Category G: Amplitude & Direction Consistency (3 features)
- `amplitude_relative_to_bout_mean` - (amplitude - bout_mean) / bout_std [NEW]
- `amplitude_consistency_in_bout` - Std of amplitudes in bout (px) [NEW]
- `direction_relative_to_previous` - Same (1) or opposite (-1) direction [NEW]

### ✅ Category H: Rule-Based Classification Features (2 features)
- `rule_based_class` - Rule-based classification (0=compensatory, 1=orienting, 2=saccade_and_fixate, 3=non_saccade) [NEW]
- `rule_based_confidence` - Rule-based confidence score (0-1) [existing]

### ❌ Category I: Contextual Flags
**Status**: Excluded (as requested)

### ❌ Category J: Normalized Features
**Status**: Excluded (as requested)

---

## Total Features: ~35 features

**Breakdown:**
- Category A: 7 features
- Category B: 5 features
- Category C: 6 features
- Category D: 10 features
- Category G: 3 features
- Category H: 2 features
- Metadata: 3 features (experiment_id, saccade_id, video_label)

**Total**: ~36 features per saccade

---

## Implementation Details

### Experiment ID Generation (Option 2)
- Extracts readable components from `data_path`
- Format: `CohortX_Condition_Animal_Date`
- Example: `Cohort3_Visual_mismatch_day4_B6J2782_2025-04-28`
- Includes fallback to path hash if extraction fails

### Feature Extraction Function
- **File**: `sleap/ml_feature_extraction.py`
- **Function**: `extract_ml_features()`
- **Helper Functions**: 
  - `extract_experiment_id()` - Generates experiment ID from path
  - `calculate_bout_sizes()` - Calculates bout sizes if not available
  - `calculate_bout_ids()` - Calculates bout IDs if not available

### Edge Case Handling
- First saccade: `time_since_previous_saccade` = NaN
- Last saccade: `time_until_next_saccade` = NaN
- Incomplete windows: Uses available data, flags with NaN where appropriate
- Empty windows: Returns 0.0 for drift/variance features

### Notes on Undersampled Data
- Velocity/position profile features (Categories E & F) are excluded
- Code includes TODO comments indicating where to add these when FPS increases
- Current implementation focuses on temporal and statistical features that work well with lower FPS

---

## Next Steps

1. ✅ Feature extraction function created
2. ⏳ Test feature extraction on sample data
3. ⏳ Implement neural network model
4. ⏳ Create training pipeline
5. ⏳ Build GUI annotation tool

---

## Usage Example

```python
from sleap.ml_feature_extraction import extract_ml_features
from pathlib import Path

# After running analyze_eye_video_saccades()
saccade_results = {...}  # From analyze_eye_video_saccades()
df = VideoData1  # Full time series DataFrame
fps = FPS_1
data_path = Path('/Users/rancze/Documents/Data/vestVR/...')

# Extract features
features_df = extract_ml_features(
    saccade_results=saccade_results,
    df=df,
    fps=fps,
    data_path=data_path,
    verbose=True
)

# features_df now contains all extracted features
# One row per saccade, ~36 columns
```

