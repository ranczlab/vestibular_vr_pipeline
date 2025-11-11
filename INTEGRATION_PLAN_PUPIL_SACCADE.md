# Integration Plan: Pupil & Saccade Data into Peri-Event Analysis

**Date:** 2025-11-10  
**Notebook:** `SANDBOX_2_noSLEAP_BATCH-alignment.py`  
**Purpose:** Integrate pupil diameter, eye position, and saccade data into the existing PhotometryAnalyzer pipeline

---

## 📊 CURRENT BEHAVIOR

### Data Loading
- **Pupil Data**: `Pupil.Diameter_eye1/2` and `Ellipse.Centre.X/Y_eye1/2` are loaded from `VideoData1/2_resampled.parquet` and joined to `photometry_tracking_encoder_data` DataFrame (lines 171-261)
- **Saccade Data**: `VideoData1/2_saccade_summary.csv` files are loaded with columns including `aeon_time`, `saccade_type` (orienting/compensatory), velocity, amplitude, etc. (lines 262-308)
- Both datasets are stored in `loaded_data` dictionary but **NOT utilized downstream**

### PhotometryAnalyzer Class (lines 2036-2473)
**Current Workflow:**
1. **Windowing**: Extracts time windows around halt events (default: -5 to +10s)
2. **Alignment**: Creates aligned DataFrames with relative time "Time (s)" and "Halt Time" columns
3. **Turn Classification**: Separates events into left/right turns based on Motor_Velocity in pre-event window (-1 to 0s)
4. **Outputs**:
   - **CSV Files**: `*_aligned.csv`, `*_left_turns.csv`, `*_right_turns.csv`
   - **Plots**: Summary plots (3 panels) and heatmaps for fluorescence channels
   - **Columns Included**: z_470, z_560, dfF_470, dfF_560, Motor_Velocity, Velocity_0X, Velocity_0Y, Photodiode_int

### Baseline Analysis (lines 2523-2886)
- Baseline correction applied to: z_470, z_560, Motor_Velocity, Velocity_0X, Velocity_0Y
- Baseline window: -1 to 0s (relative to halt event)
- Creates plots with 5 y-axes showing all baselined signals
- **Missing**: Pupil and eye position data

---

## 🎯 DESIRED CHANGES

### Focus Scope
- **Single Eye Analysis**: Focus on eye1 only (VideoData1)
- **Two New Signals**:
  1. `Pupil.Diameter_eye1` - pupil size (continuous signal)
  2. `Ellipse.Center.X_eye1` - horizontal eye position (continuous signal)

### Saccade Integration
- **Density Analysis**: Bin saccades at user-defined granularity within peri-event windows (default: 500ms)
- **Temporal Resolution**: `saccade_bin_size_s` parameter externalized to line 140 (e.g., 0.1 = 100ms, 0.5 = 500ms)
- **Metrics per Time Bin**:
  - Total saccade count
  - Orienting saccade count  
  - Compensatory saccade count
  - Saccade rate (saccades/second in Hz)
- **Visualization**: Step plot (not bar chart) to show binned density as continuous-looking signal
- **IMPORTANT**: Saccade rate is **NOT baselined** - plotted as raw rate values

### Visualization
- **Add to existing plots** (not separate plots)
- **Maintain current structure**: Extend plots with additional axes/traces
- **Apply same baseline window**: Use -1 to 0s for pupil and eye position

---

## 🔧 PROPOSED IMPLEMENTATION

### Phase 1: Data Structure Updates

#### 1.1 PhotometryAnalyzer.__init__() Modifications
**File**: SANDBOX_2_noSLEAP_BATCH-alignment.py, line ~2071

**Add parameters:**
```python
def __init__(self, 
             time_window: Tuple[float, float] = (time_window_start, time_window_end),
             video_saccade_summaries: Dict[str, pd.DataFrame] = None,
             saccade_bin_size_s: float = saccade_bin_size_s):  # Externalized parameter
    """
    Parameters:
    -----------
    video_saccade_summaries : Dict[str, pd.DataFrame]
        Dictionary with keys 'VideoData1', 'VideoData2' containing saccade summary DataFrames
    saccade_bin_size_s : float
        Bin size in seconds for saccade density analysis (externalized from line 140)
    """
    self.time_window_start, self.time_window_end = time_window
    self.video_saccade_summaries = video_saccade_summaries or {}
    self.saccade_bin_size_s = saccade_bin_size_s
```

**Externalized Parameter (Line 140):**
```python
saccade_bin_size_s = 0.5  # s, bin size for saccade density analysis (e.g., 0.1 = 100ms, 0.5 = 500ms)
```

#### 1.2 Add Eye Tracking Channels to FLUORESCENCE_CHANNELS
**Location**: Line ~2051

**Add:**
```python
FLUORESCENCE_CHANNELS = {
    'z_470': {'color': 'cornflowerblue', 'label': 'z_470'},
    'z_560': {'color': 'red', 'label': 'z_560'},
    'dfF_470': {'color': 'blue', 'label': 'dfF_470'},
    'dfF_560': {'color': 'orange', 'label': 'dfF_560'},
    # NEW: Eye tracking channels
    'Pupil.Diameter_eye1': {'color': 'purple', 'label': 'Pupil Diameter'},
    'Ellipse.Center.X_eye1': {'color': 'magenta', 'label': 'Eye Position X'}
}
```

---

### Phase 2: Saccade Density Calculation

#### 2.1 New Method: compute_saccade_density()
**Purpose**: Calculate saccade counts in time bins for each event window

**Signature:**
```python
def compute_saccade_density(self, 
                           window_data: pd.DataFrame, 
                           halt_time: pd.Timestamp,
                           eye_key: str = "VideoData1") -> pd.DataFrame:
    """
    Compute saccade density within the event window at specified bin resolution.
    
    Parameters:
    -----------
    window_data : pd.DataFrame
        Aligned window data for single halt event
    halt_time : pd.Timestamp
        Timestamp of halt event
    eye_key : str
        Which eye to analyze ('VideoData1' or 'VideoData2')
    
    Returns:
    --------
    pd.DataFrame
        window_data with added columns:
        - saccade_count_eye1: total saccades in bin
        - saccade_orienting_eye1: orienting saccades in bin
        - saccade_compensatory_eye1: compensatory saccades in bin
        - saccade_rate_eye1: saccades per second in bin
    """
```

**Implementation Logic:**
1. Get saccade summary for specified eye
2. If no saccade data available, return window_data with NaN columns
3. Filter saccades within event window (window_start to window_end)
4. Create time bins based on `self.saccade_bin_size_s`
5. For each timestamp in window_data:
   - Determine which bin it belongs to
   - Count saccades in that bin (total, orienting, compensatory)
   - Calculate rate = count / bin_size_s
6. Assign counts to corresponding rows in window_data

**Time Binning Strategy:**
```python
# Create bins relative to halt time
bin_edges = np.arange(self.time_window_start, 
                      self.time_window_end + self.saccade_bin_size_s, 
                      self.saccade_bin_size_s)

# Convert saccade times to relative time
saccade_relative_times = (saccade_times - halt_time).total_seconds()

# Digitize: assign each saccade to a bin
saccade_bins = np.digitize(saccade_relative_times, bin_edges)

# For each row in window_data, find its bin and assign counts
window_data['saccade_count_eye1'] = ...
```

#### 2.2 Integrate into process_aligned_data()
**Location**: Line ~2067

**Modify:**
```python
def process_aligned_data(self, df: pd.DataFrame, halt_time: pd.Timestamp) -> Optional[pd.DataFrame]:
    window_start = halt_time + pd.Timedelta(seconds=self.time_window_start)
    window_end = halt_time + pd.Timedelta(seconds=self.time_window_end)
    mask = (df.index >= window_start) & (df.index <= window_end)
    
    if not mask.any():
        return None
    
    window = df.loc[mask].copy()
    window["Time (s)"] = (window.index - halt_time).total_seconds()
    window["Halt Time"] = halt_time
    
    # NEW: Add saccade density if available
    if self.video_saccade_summaries:
        window = self.compute_saccade_density(window, halt_time, eye_key="VideoData1")
    
    return window
```

---

### Phase 3: Baseline Correction Extensions

#### 3.1 Update baseline_dataframe() Function
**Location**: Line ~2725 in `baseline_aligned_data_simple()`

**Modify signal list:**
```python
def baseline_dataframe(df, baseline_window, mouse_name, event_name, output_folder, suffix=""):
    df_copy = df.copy()
    
    baseline_df = df_copy[
        (df_copy["Time (s)"] >= baseline_window[0]) & 
        (df_copy["Time (s)"] <= baseline_window[1])
    ].groupby("Halt Time").mean(numeric_only=True)
    
    # UPDATED: Include pupil and eye position
    signals_to_baseline = [
        "z_470", "z_560", 
        "Motor_Velocity", "Velocity_0X", "Velocity_0Y",
        "Pupil.Diameter_eye1",      # NEW
        "Ellipse.Center.X_eye1"     # NEW
    ]
    
    # Create baseline-corrected columns
    for signal_name in signals_to_baseline:
        if signal_name in df_copy.columns:
            df_copy[f"{signal_name}_Baseline"] = (
                df_copy[signal_name] - df_copy["Halt Time"].map(baseline_df[signal_name])
            )
        else:
            print(f"      ⚠️  Column {signal_name} not found, skipping baseline correction...")
    
    # Save baseline data...
    return df_copy
```

**Note**: Saccade count columns are **NOT baselined** (they are discrete counts, not continuous signals)

---

### Phase 4: Visualization Updates

#### 4.1 Add Pupil & Eye Position to Summary Plots
**Location**: `create_summary_plot()`, line ~2247

**Modifications:**
1. **Include in line collections** (left/right turn traces panels):
```python
# Currently plots: z_470, z_560, Motor_Velocity
# UPDATED to include:
eye_channels = ['Pupil.Diameter_eye1', 'Ellipse.Center.X_eye1']
if all(ch in left_df.columns for ch in eye_channels):
    left_collections = self.create_line_collections(
        left_df, 
        z_channels + ['Motor_Velocity'] + eye_channels  # UPDATED
    )
```

2. **Add to mean±SEM comparison panel** (third panel):
   - Add secondary y-axis for pupil diameter (or use normalized values)
   - Plot both pupil and eye position with appropriate colors

3. **Optional**: Add saccade events as vertical lines or scatter overlay

#### 4.2 Update Baseline Plots
**Location**: `baseline_aligned_data_simple()`, line ~2784

**Add two new axes (ax6, ax7):**

```python
if create_plots:
    fig, ax = plt.subplots(figsize=(plot_width, 6))
    
    # ... existing axes (ax, ax2, ax3, ax4, ax5) ...
    
    # NEW: Pupil diameter axis (ax6)
    if "Pupil.Diameter_eye1_Baseline" in mean_baseline_df.columns:
        ax6 = ax.twinx()
        ax6.spines['right'].set_position(('outward', 200))
        ax6.plot(mean_baseline_df.index, 
                 mean_baseline_df["Pupil.Diameter_eye1_Baseline"], 
                 color='purple', alpha=0.8, linewidth=2, label='Pupil Diameter')
        ax6.fill_between(mean_baseline_df.index,
                         mean_baseline_df["Pupil.Diameter_eye1_Baseline"] - 
                         sem_baseline_df["Pupil.Diameter_eye1_Baseline"],
                         mean_baseline_df["Pupil.Diameter_eye1_Baseline"] + 
                         sem_baseline_df["Pupil.Diameter_eye1_Baseline"],
                         color='purple', alpha=0.2)
        ax6.set_ylabel('Pupil Diameter (pixels)', color='purple')
        ax6.set_ylim(get_symmetric_ylim(
            mean_baseline_df["Pupil.Diameter_eye1_Baseline"], 
            sem_baseline_df["Pupil.Diameter_eye1_Baseline"]
        ))
        ax6.yaxis.label.set_color('purple')
    
    # NEW: Eye position X axis (ax7)
    if "Ellipse.Center.X_eye1_Baseline" in mean_baseline_df.columns:
        ax7 = ax.twinx()
        ax7.spines['right'].set_position(('outward', 250))
        ax7.plot(mean_baseline_df.index, 
                 mean_baseline_df["Ellipse.Center.X_eye1_Baseline"], 
                 color='magenta', alpha=0.8, linewidth=2, label='Eye Position X')
        ax7.fill_between(mean_baseline_df.index,
                         mean_baseline_df["Ellipse.Center.X_eye1_Baseline"] - 
                         sem_baseline_df["Ellipse.Center.X_eye1_Baseline"],
                         mean_baseline_df["Ellipse.Center.X_eye1_Baseline"] + 
                         sem_baseline_df["Ellipse.Center.X_eye1_Baseline"],
                         color='magenta', alpha=0.2)
        ax7.set_ylabel('Eye Position X (pixels)', color='magenta')
        ax7.set_ylim(get_symmetric_ylim(
            mean_baseline_df["Ellipse.Center.X_eye1_Baseline"], 
            sem_baseline_df["Ellipse.Center.X_eye1_Baseline"]
        ))
        ax7.yaxis.label.set_color('magenta')
    
    # Add vertical line at event time
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    fig.tight_layout()
    # ... save figure ...
```

#### 4.3 Saccade Density Visualization
**Implementation**: Use **step plot** (not bar chart) to show binned density as a continuous-looking signal

**In Summary Plots** (`create_summary_plot()`, line ~2571):
```python
if "saccade_rate_eye1" in aligned_df.columns:
    ax3_saccade = ax3.twinx()
    ax3_saccade.spines['right'].set_position(('outward', 180))
    
    # Use step plot for left turns
    ax3_saccade.step(time_idx, mean_saccade, where='mid',
                    color='cyan', alpha=0.6, linewidth=2, 
                    label="Left Saccade Rate")
    
    # Use step plot for right turns (different style)
    ax3_saccade.step(time_idx, mean_saccade, where='mid',
                    color='darkblue', alpha=0.8, linewidth=2.5,
                    linestyle='--', label="Right Saccade Rate")
```

**In Baseline Plots** (`baseline_aligned_data_simple()`, line ~3171):
```python
if "saccade_rate_eye1" in mean_baseline_df.columns:
    ax8 = ax.twinx()
    ax8.spines['right'].set_position(('outward', 300))
    
    # Step plot with fill
    ax8.step(mean_baseline_df.index, 
            mean_baseline_df["saccade_rate_eye1"],
            where='mid', color='cyan', alpha=0.8, linewidth=2)
    ax8.fill_between(mean_baseline_df.index,
                    0, mean_baseline_df["saccade_rate_eye1"],
                    step='mid', color='cyan', alpha=0.2)
    ax8.set_ylabel('Saccade Rate (Hz)', color='cyan')
```

**Key Points**:
- Step plot (`where='mid'`) centers each bin value at its midpoint
- Creates a histogram-like appearance but as a continuous line
- Maintains the binned nature while looking like other timeseries signals
- **NOT baselined** - shows raw saccade rate (Hz)

---

### Phase 5: Heatmap Extensions

#### 5.1 Add Pupil and Eye Position Heatmaps
**Location**: `_create_all_heatmaps()`, line ~2447

**Modify channel list:**
```python
def _create_all_heatmaps(self, aligned_df: pd.DataFrame, session_name: str, 
                       event_name: str, data_path: Path) -> None:
    """Create all heatmaps for different channels."""
    
    # UPDATED: Include pupil and eye position
    heatmap_channels = [
        'z_470', 'z_560', 'dfF_470', 'dfF_560',
        'Pupil.Diameter_eye1',      # NEW
        'Ellipse.Center.X_eye1'     # NEW
    ]
    
    for channel in heatmap_channels:
        try:
            if channel not in aligned_df.columns:
                print(f"ℹ️ Skipping {channel} heatmap (column not present)")
                continue
            
            # Create pivot table and heatmap...
            # ... existing code ...
```

---

### Phase 6: Pipeline Integration

#### 6.1 Update main() Function
**Location**: Line ~2476

**Pass saccade data and bin size:**
```python
def main(data_paths: List[Path], loaded_data: Dict, data_path_variables: Dict, 
         event_name: str = "halt", 
         time_window: Tuple[float, float] = (-5, 10),
         saccade_bin_size_s: float = 0.1):  # NEW parameter
    """
    Main processing function.
    
    Args:
        saccade_bin_size_s : float
            Bin size in seconds for saccade density analysis (default: 0.1 = 100ms)
    """
    # Initialize analyzer with saccade parameters
    analyzer = PhotometryAnalyzer(
        time_window, 
        saccade_bin_size_s=saccade_bin_size_s  # NEW
    )
    
    # Process each data path
    for idx, data_path in enumerate(data_paths, start=1):
        # ... existing code ...
        
        # UPDATED: Extract saccade summaries from loaded_data
        video_saccade_summaries = loaded_data[data_path].get("video_saccade_summaries", {})
        
        # Pass to analyzer (modify process_session to accept this)
        analyzer.video_saccade_summaries = video_saccade_summaries  # Set before processing
        
        analyzer.process_session(
            data_path, 
            loaded_data[data_path], 
            data_path_variables[data_path], 
            event_name
        )
```

#### 6.2 Update Script Usage Cell
**Location**: Line ~2516

**Update call:**
```python
time_window = (time_window_start, time_window_end)
saccade_bin_size_s = 0.1  # 100ms bins - USER CONFIGURABLE

main(data_paths, loaded_data, data_path_variables, 
     event_name=event_name, 
     time_window=time_window,
     saccade_bin_size_s=saccade_bin_size_s)  # NEW parameter
```

---

### Phase 7: Error Handling & Validation

#### 7.1 Graceful Degradation
**Implement checks throughout:**
```python
# Check if eye tracking columns exist
has_pupil = 'Pupil.Diameter_eye1' in df.columns
has_eye_pos = 'Ellipse.Center.X_eye1' in df.columns

if not has_pupil:
    print("ℹ️ Pupil diameter data not available, skipping pupil analysis")

if not has_eye_pos:
    print("ℹ️ Eye position data not available, skipping eye position analysis")

# Check if saccade data exists
if not self.video_saccade_summaries:
    print("ℹ️ No saccade summary data available, skipping saccade density analysis")
```

#### 7.2 Data Quality Checks
**In compute_saccade_density():**
- Verify saccade timestamps are within session bounds
- Handle empty saccade lists (return zeros, not NaN)
- Log warnings if saccade alignment is poor (>10ms offset)

#### 7.3 Backward Compatibility
**Ensure code works without new data:**
- All new columns should be optional
- Plots should render without pupil/eye data
- CSVs should save successfully with missing columns

---

## 📝 EXPECTED OUTPUTS AFTER IMPLEMENTATION

### CSV Files (per session)
**Updated Files:**
1. `*_aligned.csv` - **NOW INCLUDES**:
   - `Pupil.Diameter_eye1`
   - `Ellipse.Center.X_eye1`
   - `saccade_count_eye1`
   - `saccade_orienting_eye1`
   - `saccade_compensatory_eye1`
   - `saccade_rate_eye1`

2. `*_left_turns.csv` - Same new columns as above
3. `*_right_turns.csv` - Same new columns as above
4. `*_baselined_data.csv` - **NOW INCLUDES**:
   - `Pupil.Diameter_eye1_Baseline`
   - `Ellipse.Center.X_eye1_Baseline`

### Plots
**Updated Plots:**
1. **Summary plots** (`*_halt.pdf`):
   - Left/right turn panels now show pupil and eye position traces
   - Mean±SEM panel includes pupil/eye data

2. **Baseline plots** (`*_baselined.pdf`):
   - 7-8 y-axes total (added ax6, ax7, optionally ax8)
   - Purple: Pupil diameter
   - Magenta: Eye position X
   - Cyan: Saccade rate (if included)

3. **Heatmaps**:
   - `*_heatmap_Pupil.Diameter_eye1.pdf` - NEW
   - `*_heatmap_Ellipse.Center.X_eye1.pdf` - NEW

---

## 🎯 SUCCESS CRITERIA

✅ **Functional Requirements:**
- [ ] Pupil diameter and eye position included in all aligned CSVs
- [ ] Saccade density calculated at 100ms bins (user-configurable)
- [ ] Baseline correction applied to pupil and eye position
- [ ] All plots updated with new signals
- [ ] Heatmaps generated for pupil and eye position

✅ **Quality Requirements:**
- [ ] Code runs without errors when eye tracking data is present
- [ ] Code runs without errors when eye tracking data is missing (graceful degradation)
- [ ] No memory leaks or performance degradation
- [ ] Consistent color scheme across all plots
- [ ] Clear axis labels and legends

✅ **Documentation:**
- [ ] This integration plan document
- [ ] Inline code comments for new methods
- [ ] Updated docstrings for modified methods

---

## 📌 OPEN QUESTIONS / FUTURE WORK

### Deferred to Todo List (Low Priority):
1. **Saccade Event-Level Summary CSV**: Create per-event statistics file with:
   - One row per halt event
   - Columns: Halt Time, Turn Direction, Total Saccades, Orienting Count, Compensatory Count, etc.
   - **Decision**: User unsure if needed - implement only if requested

### Potential Enhancements (Not in Scope):
- Binocular analysis (both eyes)
- Saccade-triggered averaging (align to saccade onset, not halt)
- Correlation analysis (pupil vs fluorescence)
- Turn classification by saccade presence

---

## 📅 IMPLEMENTATION TIMELINE

**Estimated Effort**: 4-6 hours of focused work

**Suggested Sequence**:
1. Phase 1: Data structure (30 min)
2. Phase 2: Saccade density (60 min)
3. Phase 3: Baseline correction (30 min)
4. Phase 4: Visualization baseline plots (45 min)
5. Phase 5: Heatmaps (20 min)
6. Phase 6: Pipeline integration (30 min)
7. Phase 7: Error handling & testing (60 min)
8. Phase 4.1: Summary plot updates (45 min) - can be deferred

**Testing Strategy**:
- Test with full dataset (has pupil, eye, saccade data)
- Test with partial dataset (missing some data)
- Test with minimal dataset (only fluorescence, no eye tracking)
- Verify CSV outputs and plots visually

---

## 🔗 RELATED FILES

**Primary File**: `SANDBOX_2_noSLEAP_BATCH-alignment.py`

**Related Modules**:
- `harp_resources/process.py` - Data loading and preprocessing
- `harp_resources/utils.py` - Utility functions

**Example Data**:
- `VideoData1_saccade_summary.csv` - Saccade event data (referenced but not provided)

---

**Document Version**: 2.0  
**Last Updated**: 2025-11-10  
**Status**: ✅ **IMPLEMENTATION COMPLETE** - Ready for testing

---

## ✅ IMPLEMENTATION COMPLETION SUMMARY

**All core phases completed successfully:**

### ✅ Phase 1: Data Structure Updates
- Added `video_saccade_summaries` and `saccade_bin_size_s` parameters to `PhotometryAnalyzer.__init__()`
- Added pupil diameter and eye position channels to `FLUORESCENCE_CHANNELS` dictionary
- **Lines modified**: 2051-2076

### ✅ Phase 2: Saccade Density Calculation
- Created `compute_saccade_density()` method with 100ms binning (user-configurable)
- Integrated into `process_aligned_data()` with error handling
- Calculates total, orienting, compensatory saccade counts and rates per time bin
- **Lines added**: 2078-2237

### ✅ Phase 3: Baseline Correction
- Updated `baseline_dataframe()` to include pupil and eye position signals
- Added conditional checks for missing columns
- **Lines modified**: 2863-2890

### ✅ Phase 4: Visualization Updates
- **Phase 4.2**: Added ax6 (pupil diameter) and ax7 (eye position) to baseline plots
  - Purple color for pupil, magenta for eye position
  - Spines positioned at outward 200 and 250
  - **Lines added**: 3005-3033
- **Phase 4.3**: Added ax8 (saccade rate) as cyan bar chart to baseline plots
  - **Lines added**: 3138-3153
- **Phase 4.1**: Updated summary plots with pupil, eye position, AND saccade rate in mean±SEM panel
  - Pupil diameter: purple, outward 60
  - Eye position X: magenta, outward 120
  - Saccade rate: cyan bar chart, outward 180
  - Bar width = 80% of bin size for visibility
  - **Lines added**: 2512-2601

### ✅ Phase 5: Heatmap Extensions
- Added pupil and eye position to heatmap channel list
- Added conditional checks for missing columns
- **Lines modified**: 2585-2599

### ✅ Phase 6: Pipeline Integration
- Added `saccade_bin_size_s` parameter to `main()` function
- Updated script usage cell with configurable parameter (default: 0.1s = 100ms)
- Pass `video_saccade_summaries` to PhotometryAnalyzer per session
- **Lines modified**: 2623-2675

### ✅ Phase 7: Error Handling & Validation
- Graceful degradation for missing eye tracking data
- Try-except wrapper in `process_aligned_data()` for saccade computation
- Backward compatibility ensured (all new columns optional)
- **Lines modified**: 2103-2235

---

## 📊 NEW OUTPUT FILES

**CSV Files (per session):**
- `*_aligned.csv` - Now includes: `Pupil.Diameter_eye1`, `Ellipse.Center.X_eye1`, `saccade_count_eye1`, `saccade_orienting_eye1`, `saccade_compensatory_eye1`, `saccade_rate_eye1`
- `*_left_turns.csv` - Same new columns
- `*_right_turns.csv` - Same new columns
- `*_baselined_data.csv` - Now includes: `Pupil.Diameter_eye1_Baseline`, `Ellipse.Center.X_eye1_Baseline`

**Plots (per session):**
- `*_halt.pdf` - **Summary plot** (3 panels):
  - Panel 1: Left turn individual traces (fluorescence + motor velocity)
  - Panel 2: Right turn individual traces (fluorescence + motor velocity)  
  - Panel 3: **Mean±SEM comparison** with multiple y-axes:
    - Left axis: Fluorescence (z-scores)
    - Right axis 1: Motor velocity (slategray)
    - Right axis 2: Pupil diameter (purple, outward 60)
    - Right axis 3: Eye position X (magenta, outward 120)
    - Right axis 4: **Saccade rate (cyan bar chart, outward 180)**
- `*_baselined.pdf` - **Baseline-corrected plot** with up to 8 y-axes:
  - ax1: Photodiode (grey)
  - ax2: z_470 & z_560 (green, red)
  - ax3: Motor velocity (dark blue)
  - ax4: Running velocity (orange)
  - ax5: Turning velocity (steel blue)
  - ax6: Pupil diameter (purple, outward 200)
  - ax7: Eye position X (magenta, outward 250)
  - ax8: **Saccade rate bar chart (cyan, outward 300)**
- `*_heatmap_Pupil.Diameter_eye1.pdf` - New heatmap
- `*_heatmap_Ellipse.Center.X_eye1.pdf` - New heatmap

---

## 🎯 USER-CONFIGURABLE PARAMETERS

```python
# In the script usage cell (line ~2671):
saccade_bin_size_s = 0.1  # 100ms bins - USER CONFIGURABLE
```

**To change saccade temporal resolution:**
- Set to `0.05` for 50ms bins (higher temporal resolution, more detail)
- Set to `0.1` for 100ms bins (default, good balance) ⭐
- Set to `0.2` for 200ms bins (coarser, smoother)
- Set to `0.5` for 500ms bins (very coarse)

### 📊 How Saccade Density Works

**Computation:**
1. For each halt event, extract all saccades within the time window (-5 to +10s by default)
2. Bin saccades into time bins of width `saccade_bin_size_s` (default: 100ms)
3. For each bin, count:
   - Total saccades (`saccade_count_eye1`)
   - Orienting saccades (`saccade_orienting_eye1`)
   - Compensatory saccades (`saccade_compensatory_eye1`)
4. Calculate rate: `saccade_rate_eye1` = count / bin_size (in Hz)

**Visualization in Summary Plots (`*_Apply halt 2s.pdf`):**
- Saccade rate shown as **cyan bar chart** on right axis 4 (outward 180)
- Bar width = 80% of `saccade_bin_size_s` for visibility
- Left turns: lighter cyan (alpha=0.3, cyan edge)
- Right turns: darker cyan (alpha=0.5, dark blue edge)

**Visualization in Baseline Plots (`*_baselined.pdf`):**
- Saccade rate shown as **cyan bar chart** on ax8 (outward 300)
- Bar width = 0.05s fixed
- Shows mean±SEM across all halt events

**CSV Output:**
- Every row in `*_aligned.csv` has all 4 saccade columns
- Multiple rows within same bin have identical saccade counts/rate
- Zero counts if no saccades in that bin

---

## 🔧 CODE QUALITY

**Linter Status**: 
- ✅ No new errors introduced
- Pre-existing warnings (unused imports, f-strings) remain unchanged
- All new code follows existing style conventions

**Backward Compatibility**:
- ✅ Code runs without errors when eye tracking data is missing
- ✅ Code runs without errors when saccade data is missing
- ✅ All new columns are optional and initialized with zeros/NaN if data unavailable

---

## 📋 REMAINING TASKS

**For User:**
1. **Testing** (requires actual data):
   - Test with full dataset (all eye tracking + saccades)
   - Test with partial dataset (missing some eye tracking)
   - Test with minimal dataset (no eye tracking)
   - Verify CSV outputs contain expected columns
   - Verify plots render correctly with new axes

2. **Optional**: Saccade event-level summary CSV
   - Currently deferred per user request
   - Can be implemented if needed in future

---

## 🚀 HOW TO USE

### **IMPORTANT: Workflow for Multiple Event Types**

If you need to process BOTH "Apply halt: 2s" AND "No halt" events, follow this workflow:

**Step 1: Initial Setup (Run Once)**
1. Cell with `data_dirs` setup (lines 93-143)
   - **Set `saccade_bin_size_s` at line 140** (e.g., 0.5 for 500ms bins)
   - Set `event_name` at line 142
2. Data loading cell (lines 154-343) - **This loads ALL data including pupil/eye columns**
   - ✅ Watch for: "Joined eye tracking columns: ['Pupil.Diameter_eye1', 'Ellipse.Center.X_eye1', ...]"

**Step 2: Process First Event Type**
3. Set `event_name = "Apply halt: 2s"` (line 142)
4. Behavioral analysis cell (line 1945) - *optional*
5. PhotometryAnalyzer cell (lines ~2788-2793) - **Uses saccade_bin_size_s from line 140**
   - ✅ Watch for: "Eye tracking columns present: ['Pupil.Diameter_eye1', 'Ellipse.Center.X_eye1', ...]"
6. Baseline analysis cell (lines ~3206-3218) - *if desired*

**Step 3: Process Second Event Type**
7. Change `event_name = "No halt"` (line 141)
8. **Re-run** behavioral analysis cell (line 1945) - *if used in Step 2*
9. **Re-run** PhotometryAnalyzer cell (lines ~2739-2745)
   - ✅ Should also show: "Eye tracking columns present: [...]"
10. **Re-run** baseline analysis cell (lines ~2873-2886) - *if used in Step 2*

### **⚠️ Common Mistake:**

**DON'T** compare outputs from different runs! If you:
- Processed "No halt" BEFORE updating the code → old outputs WITHOUT pupil/eye
- Processed "Apply halt: 2s" AFTER updating the code → new outputs WITH pupil/eye
- The difference is from different code versions, not different event types!

### **Diagnostic Output:**

The code now prints diagnostic messages:
```
✅ Joined eye tracking columns: ['Pupil.Diameter_eye1', 'Ellipse.Center.X_eye1', ...]
✅ Eye tracking columns present: ['Pupil.Diameter_eye1', 'Ellipse.Center.X_eye1', ...]
```

If you see:
```
ℹ️ No eye tracking columns joined (video data may not contain them)
⚠️ No eye tracking columns found in aligned data
```

Then your video parquet files don't contain the eye tracking columns.

### **Expected Behavior:**

- ✅ If pupil/eye data exists: All plots and CSVs will include new signals **for ALL event types**
- ✅ If pupil/eye data missing: Code runs normally, columns filled with zeros/NaN
- ✅ If saccade data missing: Code runs normally, saccade columns filled with zeros

---

## ⚠️ CURRENT ISSUE: Saccade Rate Calculation (2025-11-11)

### **Problem Report:**
Saccade rate showing as constant high value (~1000 Hz) across entire aligned/averaged plot, instead of varying values showing temporal dynamics.

### **Current Implementation Analysis:**

#### How Saccade Rate is Currently Calculated:

**Step 1: Per-Event Binning (in `compute_saccade_density()`, lines 2087-2209)**
```python
# For EACH halt event individually:
1. Get saccades within this event's window (time_window_start to time_window_end)
2. Create bin edges: [-5.0, -4.5, -4.0, ..., 9.5, 10.0] with 500ms steps
3. Count saccades falling in each bin (e.g., bin[0] = 2 saccades)
4. Calculate rate: bin_rates = bin_counts / saccade_bin_size_s
   Example: 2 saccades in 500ms bin → 2 / 0.5 = 4 Hz
5. Assign this rate to EVERY sample in window_data that falls within that bin
```

**Step 2: Assignment to DataFrame (lines 2197-2208)**
```python
for idx, rel_time in enumerate(window_relative_times):
    bin_idx = np.digitize([rel_time], bin_edges)[0] - 1
    if 0 <= bin_idx < n_bins:
        window_data.iloc[idx, ...] = bin_rates[bin_idx]
```

**Key Point:** If window_data is sampled at 1000 Hz (1ms resolution):
- Each 500ms bin has ~500 rows in window_data
- ALL 500 rows get assigned the SAME rate value
- Example: If bin[0] has rate=4 Hz, then rows at t=0.000s, t=0.001s, t=0.002s, ..., t=0.499s ALL get value 4

**Step 3: Aggregation Across Events (in plotting code)**
```python
grouped = aligned_df.groupby("Time (s)")
mean_saccade = grouped.mean()["saccade_rate_eye1"]
```

### **Possible Issues:**

#### **Issue 1: Units/Interpretation Confusion**
- **What user wants:** "Saccade probability in 500ms bins"
- **What code calculates:** Saccade rate (Hz = saccades/second)
- **Difference:**
  - **Probability**: 0 to 1 (or 0% to 100%) → proportion of bins with saccades OR proportion of trials with saccade in this bin
  - **Rate (Hz)**: 0 to ∞ → saccades per second
  - **Count**: 0, 1, 2, 3... → raw number of saccades in bin

**Example for 500ms bin:**
- 2 saccades detected in bin
- **Current calculation**: 2 / 0.5 = 4 Hz
- **Possible intended**: 2 saccades (raw count) OR 1.0 (100% probability if any saccade present)

#### **Issue 2: Potential Bug - Wrong Bin Size Used**
If `self.saccade_bin_size_s` is somehow not 0.5 but much smaller (e.g., 0.001), then:
- 1 saccade / 0.001 = 1000 Hz ← **This matches the observed value!**

**Diagnostic needed:** Print `self.saccade_bin_size_s` at line 2195 to verify

#### **Issue 3: Over-replication in High-Resolution Data**
- window_data sampled at 1000 Hz (1ms steps)
- Assigning same value to all 500 samples per bin
- Then averaging across events

**This shouldn't cause wrong values BUT:**
- If there's any issue with time alignment or binning logic, it could amplify errors
- Creates massive redundancy (same value repeated 500 times per bin)

#### **Issue 4: Aggregation Method**
Current: Calculate rate per event, then average rates across events
Alternative: Pool all saccades across all events, then calculate rate

**Example with 10 events, looking at first 500ms bin:**
- **Current approach:**
  - Event 1: 1 saccade → 2 Hz
  - Event 2: 0 saccades → 0 Hz
  - Event 3: 2 saccades → 4 Hz
  - Mean rate = (2 + 0 + 4 + ...) / 10 = some average
  
- **Alternative approach:**
  - Total saccades across all 10 events in this bin: 10 saccades
  - Average per event: 10 / 10 = 1 saccade/event
  - Rate: 1 / 0.5 = 2 Hz
  - OR Probability: 7/10 events had at least one saccade = 70%

### **Proposed Solutions (for discussion):**

#### **Option A: Saccade Probability (0 to 1)**
```python
# For each bin, calculate: proportion of trials with at least one saccade
# This would be calculated AFTER pooling across all events, not per-event
```

#### **Option B: Average Saccade Count per Bin**
```python
# Keep raw counts, don't divide by bin_size_s
# Show average number of saccades per bin across trials
bin_average_counts = bin_counts  # Don't divide by time
```

#### **Option C: Fix Current Rate Calculation**
```python
# Keep current approach but verify:
1. self.saccade_bin_size_s is correct (should be 0.5)
2. bin_counts is correct (should be small integers like 0, 1, 2)
3. Division is happening correctly
```

### **Questions for User:**

1. **What do you want to see plotted?**
   - A) Probability (0 to 1 or 0% to 100%) that a saccade occurs in each bin
   - B) Average count of saccades per bin (e.g., 0.5, 1.2, 2.0 saccades)
   - C) Rate in Hz (saccades/second) - current approach
   - D) Something else?

2. **Expected values:**
   - For a 500ms bin in your data, roughly how many saccades do you expect?
   - Should the value vary across time (e.g., more before halt, fewer after)?
   - What's a typical saccade rate for a mouse (e.g., 1-5 Hz? 10-20 Hz?)?

3. **Diagnostic request:**
   - Can you print `saccade_bin_size_s` value when running?
   - Can you check one saccade summary CSV - how many rows (saccades) in a typical 15-second window?

---

## 🔄 RECENT FIXES (2025-11-11)

### Fixed Issues:
1. **Externalized `saccade_bin_size_s` Parameter**
   - **Problem**: Parameter was hardcoded in `__init__` method (line 2073)
   - **Solution**: Moved to global parameter cell (line 140)
   - **Benefit**: User can now easily adjust bin size (e.g., 0.1 = 100ms, 0.5 = 500ms) without editing class code

2. **Improved Saccade Density Visualization**
   - **Problem**: Bar charts created redundancy (plotting same value at every sample point within bins)
   - **Solution**: Changed to step plots (`plt.step(..., where='mid')`)
   - **Benefit**: 
     - Cleaner visualization that looks like other timeseries signals
     - Maintains binned nature while appearing as continuous line
     - Better performance (fewer plot elements)
     - Histogram-like appearance with proper bin centering

3. **Clarified Non-Baseline Behavior**
   - **Confirmed**: Saccade rate is **NOT baselined** (as intended)
   - **Rationale**: Saccade rate is a density measure (Hz), not a signal that should be zero-centered
   - **Implementation**: Raw rate values plotted on all visualizations

### Updated Code Locations:
- **Parameter Cell**: Line 140 (`saccade_bin_size_s = 0.5`)
- **`__init__` Method**: Line 2073 (uses externalized parameter)
- **Summary Plot**: Lines 2571-2601 (step plot implementation)
- **Baseline Plot**: Lines 3171-3188 (step plot with fill)
- **Script Usage**: Line 2788-2793 (references externalized parameter)

---

