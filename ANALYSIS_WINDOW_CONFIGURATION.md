# Analysis Window Configuration Guide

## Key Configuration Parameter

### `ANALYSIS_WINDOW_MEAN_PEAK` - Primary User-Configurable Window

**Location:** Configuration cell in both `.py` and `.ipynb` files

**Default:** `(0.0, 1.0)` seconds

**Controls:**
- Post-halt mean velocity (turning & running)
- Peak velocity (turning & running)  
- Area under curve (AUC)

## How to Change the Analysis Window

To analyze a different time window after halt onset, simply change this line:

```python
ANALYSIS_WINDOW_MEAN_PEAK = (0.0, 1.0)  # Current: analyze 0-1s after halt
```

### Examples:

**Analyze first 2 seconds:**
```python
ANALYSIS_WINDOW_MEAN_PEAK = (0.0, 2.0)
```

**Analyze 0.5-1.5s window:**
```python
ANALYSIS_WINDOW_MEAN_PEAK = (0.5, 1.5)
```

**Analyze first 500ms:**
```python
ANALYSIS_WINDOW_MEAN_PEAK = (0.0, 0.5)
```

## What Gets Updated Automatically

When you change `ANALYSIS_WINDOW_MEAN_PEAK`, the following are automatically updated:

### 1. **Metric Labels**
Plot titles and axis labels now show the actual window being analyzed:
- ✅ `"Peak velocity 0.0-1.0s (deg/s)"` → Shows your configured window
- ✅ `"Post-halt mean velocity 0.0-1.0s (cm/s)"` → Shows your configured window
- ✅ `"AUC 0.0-1.0s (deg·s)"` → Shows your configured window

### 2. **Data Analysis**
All calculations use your configured window:
- Mean velocity calculation
- Peak detection
- AUC integration
- Statistical comparisons

### 3. **Output Files**
CSV files and plots reflect the configured window in their labels and filenames.

## Relationship to Other Windows

```
Timeline:  |-1.0s -------- 0.0s -------- 1.0s -------- 2.0s -------- 3.0s|
           |             Halt |                                          |
           
BASELINE_WINDOW:           ↑←→↑               (Pre-halt baseline)
ANALYSIS_WINDOW_MEAN_PEAK:      ↑←────────→↑  (YOUR CONFIGURABLE WINDOW)
EXTENDED_RESPONSE_WINDOW:       ↑←──────────────────→↑  (Mean |velocity|)
FULL_RESPONSE_WINDOW:           ↑←────────────────────────────→↑  (Decay)
```

### Other Windows (Less Commonly Changed):

- **`BASELINE_WINDOW`**: Pre-halt baseline period (default: -1 to 0s)
- **`EXTENDED_RESPONSE_WINDOW`**: For mean absolute velocity that combines left/right turns (default: 0-2s)
- **`FULL_RESPONSE_WINDOW`**: For decay fitting (default: 0-3s)
- **`TEMPORAL_*_WINDOW`**: Fine-grained temporal analysis windows

## Terminology Clarification

**Previous (incorrect):** "Post-turn"
**Current (correct):** "Post-halt"

The analysis examines velocity changes **after halt onset**, not after turning behavior. The terminology has been updated throughout the code to accurately reflect this.

## Output Files

When you run the analysis, all outputs will reflect your configured window:

### PDFs:
- `turning_metric_comparisons.pdf` - Shows window in subplot titles
- `running_metric_comparisons.pdf` - Shows window in subplot titles

### CSVs:
- `turning_metrics_ttests.csv` - Metric labels include window
- `running_metrics_ttests.csv` - Metric labels include window

## Quick Start

1. **Open configuration cell** in `SANDBOX_4_MM_aligned_running_turning_analysis.py` or `.ipynb`

2. **Find this line:**
   ```python
   ANALYSIS_WINDOW_MEAN_PEAK = (0.0, 1.0)
   ```

3. **Change to your desired window:**
   ```python
   ANALYSIS_WINDOW_MEAN_PEAK = (0.0, 2.0)  # Example: 2 seconds
   ```

4. **Run the analysis** - All plots and metrics will automatically use your new window!

## Benefits of This Approach

✅ **Single source of truth** - Change one value, updates everywhere

✅ **Clear labeling** - Plots always show which window was used

✅ **No hardcoding** - No need to search through code to change windows

✅ **Reproducible** - Anyone can see exactly what window was analyzed

✅ **Flexible** - Easy to compare different time windows by re-running with different settings

## Common Use Cases

### Immediate Response (0-0.5s)
```python
ANALYSIS_WINDOW_MEAN_PEAK = (0.0, 0.5)
```
Use when interested in the very rapid response immediately after halt onset.

### Standard Analysis (0-1s)
```python
ANALYSIS_WINDOW_MEAN_PEAK = (0.0, 1.0)  # Default
```
Captures the initial response while avoiding longer-term dynamics.

### Extended Response (0-2s)
```python
ANALYSIS_WINDOW_MEAN_PEAK = (0.0, 2.0)
```
Include more sustained response, useful if effects are slower.

### Delayed Window (0.5-1.5s)
```python
ANALYSIS_WINDOW_MEAN_PEAK = (0.5, 1.5)
```
Exclude immediate transient, focus on sustained phase.

## Troubleshooting

**Q: I changed the window but plots still show old values**

A: Make sure to re-run all analysis cells after changing the configuration. The values are calculated when you run the cells.

**Q: Can I have different windows for turning vs running?**

A: Currently they use the same window. If you need different windows, you could create `ANALYSIS_WINDOW_TURNING` and `ANALYSIS_WINDOW_RUNNING` separately (requires code modification).

**Q: What if my data doesn't extend to my chosen window end?**

A: The analysis will use whatever data is available. Check your aligned data files to see the time range available.

## Implementation Details

The configuration works through variable aliasing:

```python
# Your configuration
ANALYSIS_WINDOW_MEAN_PEAK = (0.0, 1.0)

# Internal aliases (don't change these)
EARLY_RESPONSE_WINDOW = ANALYSIS_WINDOW_MEAN_PEAK
POST_WINDOW = EARLY_RESPONSE_WINDOW
PEAK_WINDOW = EARLY_RESPONSE_WINDOW
AUC_WINDOW = EARLY_RESPONSE_WINDOW
```

When you change `ANALYSIS_WINDOW_MEAN_PEAK`, all the derived variables update automatically.

