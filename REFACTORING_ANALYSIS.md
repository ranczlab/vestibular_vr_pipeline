# Refactoring Analysis: Functions to Extract from Notebook

## Summary
The notebook `SANDBOX_1_1_Loading_+_Saccade_detection.py` is 4429 lines long and contains several functions and repeated code patterns that should be moved to modules.

## Functions Currently Defined in Notebook

### 1. `get_eye_label(key)` - Line 67-69
**Status**: Should be moved to `sleap/processing_functions.py` or `sleap/utils.py`
**Usage**: Used throughout notebook for display labels
**Dependencies**: Uses `VIDEO_LABELS` global dict

### 2. `min_max_dict(coordinates_dict)` - Line 463-468
**Status**: DUPLICATE - Already exists in `sleap/processing_functions.py` (line 10-33)
**Action**: Remove from notebook, use `pf.min_max_dict()` instead
**Note**: Also duplicated at line 1880

### 3. `TeeOutput` class - Line 971-983
**Status**: Should be moved to `sleap/processing_functions.py` or new `sleap/utils.py`
**Usage**: Used for capturing stdout to file (blink detection output)
**Dependencies**: `sys`, `io`

### 4. `check_manual_match()` - Lines 1641-1654 and 1694-1707
**Status**: DUPLICATE - Defined twice (once for v1, once for v2)
**Action**: Extract to `sleap/processing_functions.py` as single function
**Usage**: Checks if auto-detected blink matches manual blink annotation

### 5. `get_direction_map_for_video(video_key)` - Line 2910-2915
**Status**: Should be moved to `sleap/saccade_processing.py` or `sleap/utils.py`
**Usage**: Maps detected directions (upward/downward) to NT/TN based on eye assignment
**Dependencies**: Uses `video1_eye` and `video2_eye` globals

## Code Blocks That Should Be Extracted to Functions

### 6. Manual Blink Loading (Lines 314-349)
**Function Name**: `load_manual_blinks(data_path, video_number)`
**Module**: `sleap/processing_functions.py`
**Description**: Loads manual blink annotations from CSV files
**Returns**: List of blink dicts or None
**Repeated**: Code is duplicated for VideoData1 and VideoData2

### 7. Confidence Score Analysis (Lines 571-623)
**Function Name**: `analyze_confidence_scores(video_data, score_columns, score_cutoff, debug=False)`
**Module**: `sleap/processing_functions.py`
**Description**: Analyzes confidence scores and reports top 3 columns with most low-confidence frames
**Repeated**: Code is duplicated for VideoData1 and VideoData2

### 8. Center Coordinates (Lines 626-663)
**Function Name**: `center_coordinates_to_median(video_data, columns_of_interest)`
**Module**: `sleap/load_and_process.py` or `sleap/processing_functions.py`
**Description**: Centers coordinates to median pupil centre
**Repeated**: Code is duplicated for VideoData1 and VideoData2

### 9. Score-Based Filtering (Lines 668-725)
**Function Name**: `filter_low_confidence_points(video_data, point_names, score_cutoff, debug=False)`
**Module**: `sleap/processing_functions.py`
**Description**: Filters out low-confidence points and replaces with NaN
**Repeated**: Code is duplicated for VideoData1 and VideoData2

### 10. Outlier Analysis and Interpolation (Lines 732-785)
**Function Name**: `remove_outliers_and_interpolate(video_data, columns_of_interest, outlier_sd_threshold, debug=False)`
**Module**: `sleap/processing_functions.py`
**Description**: Removes outliers and interpolates NaN values
**Repeated**: Code is duplicated for VideoData1 and VideoData2

### 11. Instance Score Distribution Analysis (Lines 800-958)
**Function Name**: `analyze_instance_score_distribution(video_data, blink_instance_score_threshold, fps, debug=False)`
**Module**: `sleap/processing_functions.py`
**Description**: Analyzes instance score distribution and plots histogram
**Repeated**: Code is duplicated for VideoData1 and VideoData2

### 12. Blink Detection (Lines 994-1228 and 1231-1465)
**Function Name**: `detect_blinks_for_video(video_data, fps, blink_instance_score_threshold, long_blink_warning_ms, min_frames_threshold, merge_window_frames, columns_of_interest, manual_blinks=None, debug=False)`
**Module**: `sleap/processing_functions.py`
**Description**: Complete blink detection pipeline for a single video
**Repeated**: ~240 lines duplicated between VideoData1 and VideoData2
**Returns**: Dict with blink_segments, blink_bouts, etc.

### 13. Blink Bout Comparison (Lines 1469-1635)
**Function Name**: `compare_blink_bouts(video_data1, blink_bouts_v1, video_data2, blink_bouts_v2, debug=False)`
**Module**: `sleap/processing_functions.py`
**Description**: Compares blink bout timing between two videos
**Note**: Only runs if both videos exist

### 14. Save Blink Detection Results (Lines 1638-1742)
**Function Name**: `save_blink_detection_results(video_data, blink_segments, data_path, video_key, manual_blinks=None)`
**Module**: `sleap/processing_functions.py`
**Description**: Saves blink detection results to CSV
**Repeated**: Code is duplicated for VideoData1 and VideoData2

### 15. Cross-Correlation Analysis (Lines 2099-2252)
**Function Name**: `analyze_pupil_diameter_correlation(video_data1, video_data2, debug=False)`
**Module**: `sleap/processing_functions.py` or new `sleap/correlation_analysis.py`
**Description**: Cross-correlates pupil diameter between two videos
**Returns**: Dict with correlation stats

### 16. QC Figure Creation (Lines 2433-2685)
**Function Name**: `create_eye_data_qc_figure(video_data1, video_data2, video_data1_centered, video_data2_centered, data_path, save_path, video1_has_sleap, video2_has_sleap, pearson_r_display, pearson_p_display, peak_lag_time_display, get_eye_label_fn)`
**Module**: New `sleap/qc_plots.py`
**Description**: Creates comprehensive QC figure with 6 panels
**Note**: Very long function (~250 lines), could be split into sub-functions

### 17. QC Time Series Plot (Lines 2689-2841)
**Function Name**: `create_qc_timeseries_plot(video_data1, video_data2, video_data1_centered, video_data2_centered, video1_has_sleap, video2_has_sleap, data_path, save_path, get_eye_label_fn)`
**Module**: `sleap/qc_plots.py`
**Description**: Creates interactive Plotly time series plots

### 18. Saccade Visualization - All Saccades Overlay (Lines 3128-3527)
**Function Name**: `plot_all_saccades_overlay(saccade_results, video_key, get_eye_label_fn, n_before, n_after, debug=False)`
**Module**: `sleap/saccade_processing.py` or new `sleap/saccade_plots.py`
**Description**: Creates overlay plot of all detected saccades
**Note**: Very long (~400 lines), could be split

### 19. Saccade Amplitude QC Visualization (Lines 3532-3799)
**Function Name**: `plot_saccade_amplitude_qc(saccade_results, video_key, get_eye_label_fn, debug=False)`
**Module**: `sleap/saccade_plots.py`
**Description**: Creates QC plots for saccade amplitude analysis

### 20. Saccade Detection QC Visualization (Lines 3803-3987)
**Function Name**: `plot_saccade_detection_qc(saccade_results, video_key, get_eye_label_fn)`
**Module**: `sleap/saccade_plots.py`
**Description**: Creates QC plot showing detected saccades on velocity trace

### 21. Saccade Classification Visualization (Lines 3991-4425)
**Function Name**: `plot_saccade_classification_analysis(saccade_results, video_key, get_eye_label_fn, debug=False)`
**Module**: `sleap/saccade_plots.py`
**Description**: Creates comprehensive classification analysis plots and statistics
**Note**: Very long (~430 lines), includes statistical analysis

## Recommended Module Structure

```
sleap/
├── load_and_process.py          # Existing - add center_coordinates_to_median
├── processing_functions.py      # Existing - add many functions
├── saccade_processing.py        # Existing - add get_direction_map_for_video
├── utils.py                     # NEW - utility functions
│   ├── get_eye_label()
│   ├── TeeOutput class
│   └── get_direction_map_for_video()
├── qc_plots.py                 # NEW - QC plotting functions
│   ├── create_eye_data_qc_figure()
│   └── create_qc_timeseries_plot()
└── saccade_plots.py            # NEW - saccade visualization
    ├── plot_all_saccades_overlay()
    ├── plot_saccade_amplitude_qc()
    ├── plot_saccade_detection_qc()
    └── plot_saccade_classification_analysis()
```

## Priority Order for Refactoring

### High Priority (Most Duplicated Code)
1. **Blink Detection** (#12) - ~240 lines duplicated
2. **Manual Blink Loading** (#6) - Simple but duplicated
3. **Confidence Score Analysis** (#7) - Duplicated
4. **Score-Based Filtering** (#9) - Duplicated
5. **Outlier Analysis** (#10) - Duplicated
6. **Instance Score Analysis** (#11) - Duplicated
7. **Save Blink Results** (#14) - Duplicated

### Medium Priority (Long Functions)
8. **QC Figure Creation** (#16) - ~250 lines
9. **Saccade Classification Visualization** (#21) - ~430 lines
10. **All Saccades Overlay** (#18) - ~400 lines

### Low Priority (Small Functions)
11. **Utility Functions** (#1, #3, #5) - Small but used frequently
12. **Cross-Correlation** (#15) - Only used once but could be reused
13. **Blink Bout Comparison** (#13) - Only used once

## Estimated Impact

- **Current notebook size**: 4429 lines
- **Estimated reduction**: ~1500-2000 lines (34-45% reduction)
- **Functions to extract**: 21 functions/blocks
- **Duplicated code to eliminate**: ~800-1000 lines

## Notes

1. Some functions depend on global variables (`VIDEO_LABELS`, `video1_eye`, `video2_eye`) - these should be passed as parameters
2. The `get_eye_label()` function uses a global dict - consider refactoring to accept it as parameter
3. Many visualization functions are very long - consider splitting into smaller helper functions
4. Some functions have complex parameter lists - consider using dataclasses or config objects
5. Debug flags are passed through many functions - consider using logging module instead

