# Refactoring Plan: Extract Visualization Functions from Notebook

## Overview
The notebook `SANDBOX_1_1_Loading_+_Saccade_detection.py` is ~5000 lines. We need to extract visualization functions to reduce size and improve maintainability.

## Completed
✅ ML Feature Visualization (~595 lines) → `sleap/visualization.py::visualize_ml_features()`
✅ Notebook updated to use `visualize_ml_features()`

## To Extract (Main Visualization Functions)

### 1. `plot_all_saccades_overlay()` (~445 lines, lines 2100-2545)
**Dependencies:**
- `saccade_results` dict
- `get_direction_map_for_video()` helper
- `get_eye_label()` helper  
- `pre_saccade_window_time`, `post_saccade_window_time`
- `debug` flag
- `make_subplots` from plotly

**Function signature:**
```python
def plot_all_saccades_overlay(
    saccade_results: Dict,
    direction_map_func: Callable,
    get_eye_label_func: Callable,
    pre_saccade_window_time: float,
    post_saccade_window_time: float,
    video_label: str,
    debug: bool = True,
    show_plot: bool = True
) -> None
```

### 2. `plot_saccade_amplitude_qc()` (~268 lines, lines 2548-2816)
**Dependencies:**
- `saccade_results` dict
- `get_direction_map_for_video()` helper
- `get_eye_label()` helper
- `sp.get_color_mapping()` from saccade_processing
- `debug` flag

**Function signature:**
```python
def plot_saccade_amplitude_qc(
    saccade_results: Dict,
    direction_map_func: Callable,
    get_eye_label_func: Callable,
    video_label: str,
    debug: bool = True,
    show_plot: bool = True
) -> None
```

### 3. `plot_baselining_diagnostics()` (~365 lines, lines 3103-3468)
**Dependencies:**
- `saccade_results` dict
- `get_direction_map_for_video()` helper
- `get_eye_label()` helper
- `baseline_window_start_time`, `baseline_window_end_time`
- `debug` flag

**Function signature:**
```python
def plot_baselining_diagnostics(
    saccade_results: Dict,
    direction_map_func: Callable,
    get_eye_label_func: Callable,
    baseline_window_start_time: float,
    baseline_window_end_time: float,
    video_label: str,
    debug: bool = True,
    show_plot: bool = True
) -> None
```

### 4. `plot_saccade_detection_qc()` (~216 lines, lines 3471-3687)
**Dependencies:**
- `saccade_results` dict
- `get_direction_map_for_video()` helper
- `get_eye_label()` helper
- `plot_saccade_detection_QC` flag
- `debug` flag

**Function signature:**
```python
def plot_saccade_detection_qc(
    saccade_results: Dict,
    direction_map_func: Callable,
    get_eye_label_func: Callable,
    video_label: str,
    plot_5min_window: bool = True,
    window_duration: int = 300,
    debug: bool = True,
    show_plot: bool = True
) -> None
```

### 5. `plot_adaptive_threshold_diagnostics()` (~147 lines, lines 3689-3836)
**Dependencies:**
- `saccade_results` dict
- `get_eye_label()` helper
- `use_adaptive_thresholds` flag
- `adaptive_percentile_*` parameters
- `pre_saccade_velocity_threshold`, `pre_saccade_drift_threshold`, `post_saccade_variance_threshold`
- `post_saccade_position_change_threshold_percent`
- `debug` flag

**Function signature:**
```python
def plot_adaptive_threshold_diagnostics(
    saccade_results: Dict,
    get_eye_label_func: Callable,
    use_adaptive_thresholds: bool,
    adaptive_percentile_pre_velocity: int,
    adaptive_percentile_pre_drift: int,
    adaptive_percentile_post_variance: int,
    pre_saccade_velocity_threshold: float,
    pre_saccade_drift_threshold: float,
    post_saccade_variance_threshold: float,
    post_saccade_position_change_threshold_percent: float,
    video_label: str,
    debug: bool = True,
    show_plot: bool = True
) -> None
```

### 6. `plot_classification_analysis()` (~503 lines, lines 3838-4341)
**Dependencies:**
- `saccade_results` dict
- `get_direction_map_for_video()` helper
- `get_eye_label()` helper
- `scipy.stats` for statistical tests
- `debug` flag

**Function signature:**
```python
def plot_classification_analysis(
    saccade_results: Dict,
    direction_map_func: Callable,
    get_eye_label_func: Callable,
    video_label: str,
    debug: bool = True,
    show_plot: bool = True
) -> None
```

## Helper Functions Needed

### `get_direction_map_for_video(video_key: str, video1_eye: str, video2_eye: str) -> Dict[str, str]`
Maps video key to direction labels (upward/downward → NT/TN)

### `get_eye_label(video_key: str, video_labels: Dict[str, str]) -> str`
Gets display label for video

## Implementation Strategy

1. Add helper functions to `visualization.py`
2. Add each visualization function one by one
3. Update notebook to import and call functions
4. Test each function individually
5. Verify notebook still works end-to-end

## Estimated Reduction
- ML Feature Visualization: ~527 lines removed ✅
- Main Visualizations: ~1944 lines to remove
- **Total reduction: ~2471 lines (from ~5000 to ~2529)**

## Notes
- All functions should accept parameters explicitly (no global variable dependencies)
- Functions should be callable independently
- Preserve all existing functionality
- Maintain same plot outputs and diagnostic messages

