# Baseline Window Configuration Summary

**Date:** 2025-11-11  
**Script:** `SANDBOX_2_noSLEAP_BATCH-alignment.py`

---

## 📊 Baseline Window Parameters (Line 136-141)

```python
baseline_window = (-6, -4)                    # s, FOR baselining behavioral signals
fluorescence_baseline_window = (-1, 0)        # s, FOR baselining fluorescence signals
```

---

## 📋 Signal-Specific Baseline Windows

| Signal Type | Signals | Baseline Window | Duration | Usage |
|------------|---------|-----------------|----------|-------|
| **Fluorescence** | `z_470`, `z_560`, `dfF_470`, `dfF_560` | **-1 to 0 s** | 1 second | Zero-centered fluorescence during pre-event period |
| **Behavioral - Running** | `Velocity_0X` | **-6 to -4 s** | 2 seconds | Zero-centered running velocity |
| **Behavioral - Pupil** | `Pupil.Diameter_eye1` | **-6 to -4 s** | 2 seconds | Zero-centered pupil diameter |
| **Behavioral - Eye Position** | `Ellipse.Center.X_eye1` | **-6 to -4 s** | 2 seconds | Zero-centered eye position |
| **NOT Baselined** | `Motor_Velocity`, `Velocity_0Y`, `saccade_probability_eye1` | N/A | N/A | Plotted as raw values |

### Applied To:
- ✅ **Aligned Baseline Plots** (`*_baselined.pdf`, `*_baselined_left_turns.pdf`, `*_baselined_right_turns.pdf`)
- ✅ **Heatmaps** (`*_heatmap_z_470.pdf`, `*_heatmap_Pupil.Diameter_eye1.pdf`, etc.)

---

## 🔧 Implementation Details

### Code Location: Lines 3265-3308

```python
def baseline_dataframe(df, baseline_window, fluorescence_baseline_window, ...):
    # Calculate baseline for fluorescence (using fluorescence_baseline_window)
    fluorescence_baseline_df = df[
        (df["Time (s)"] >= fluorescence_baseline_window[0]) & 
        (df["Time (s)"] <= fluorescence_baseline_window[1])
    ].groupby("Halt Time").mean()
    
    # Calculate baseline for other signals (using baseline_window)
    baseline_df = df[
        (df["Time (s)"] >= baseline_window[0]) & 
        (df["Time (s)"] <= baseline_window[1])
    ].groupby("Halt Time").mean()
    
    # Apply to fluorescence signals
    for signal in ["z_470", "z_560", "dfF_470", "dfF_560"]:
        df[f"{signal}_Baseline"] = df[signal] - fluorescence_baseline_df[signal]
    
    # Apply to other signals
    for signal in ["Velocity_0X", "Pupil.Diameter_eye1", "Ellipse.Center.X_eye1"]:
        df[f"{signal}_Baseline"] = df[signal] - baseline_df[signal]
```

---

## 🎯 Rationale

### Why Different Windows?

**Fluorescence signals (-2 to 0 s):**
- ✅ Closer to event onset for more relevant baseline
- ✅ Captures immediate pre-event neural state
- ✅ Reduces influence of earlier behavioral variability
- ✅ Standard practice in photometry analysis

**Behavioral signals (-6 to -4 s):**
- ✅ More stable baseline further from event
- ✅ Avoids potential anticipatory effects
- ✅ Better captures steady-state behavior
- ✅ Consistent with previous analyses

---

## 📈 Output Files

All baseline-corrected signals are saved to CSV with `_Baseline` suffix:
- `z_470_Baseline` (baselined using -2 to 0s)
- `z_560_Baseline` (baselined using -2 to 0s)
- `dfF_470_Baseline` (baselined using -2 to 0s)
- `dfF_560_Baseline` (baselined using -2 to 0s)
- `Velocity_0X_Baseline` (baselined using -6 to -4s)
- `Pupil.Diameter_eye1_Baseline` (baselined using -6 to -4s)
- `Ellipse.Center.X_eye1_Baseline` (baselined using -6 to -4s)

**Raw (non-baselined) columns preserved:**
- `Motor_Velocity` (raw)
- `Velocity_0Y` (raw)
- `saccade_probability_eye1` (raw)

---

## 🔄 Function Flow

```
process_aligned_data_folders()
  ├─ Receives: baseline_window + fluorescence_baseline_window
  │
  └─ Calls: baseline_aligned_data_simple()
      ├─ Receives: baseline_window + fluorescence_baseline_window
      │
      └─ Calls: baseline_dataframe() [3 times]
          ├─ Combined data
          ├─ Left turns
          └─ Right turns
          
          Each call:
            ├─ Computes fluorescence baseline using fluorescence_baseline_window
            ├─ Computes behavioral baseline using baseline_window
            └─ Applies appropriate baseline to each signal type
```

---

## 🆕 HEATMAP BASELINE CONSISTENCY (2025-11-11)

### Previous Behavior
**Problem:** Heatmaps used a **hardcoded** `-1 to 0s` baseline for ALL channels, regardless of signal type.

```python
# OLD CODE (line 2409)
baseline_cols = (pivot_data.columns >= -1) & (pivot_data.columns < 0)  # Hardcoded!
```

**Result:** 
- ✅ Fluorescence heatmaps: Correct baseline (-1 to 0s)
- ❌ Pupil/eye heatmaps: WRONG baseline (used -1 to 0s instead of -6 to -4s)
- ❌ Inconsistent with aligned baseline plots

### Fixed Behavior
**Solution:** Heatmaps now use **channel-specific** baseline windows, matching aligned plots.

**Changes:**
1. **`PhotometryAnalyzer.__init__()`** (lines 2220-2241): Now accepts and stores both baseline windows
2. **`create_heatmap()`** (lines 2398-2422): Now accepts `baseline_window` parameter instead of hardcoded values
3. **`_create_all_heatmaps()`** (lines 2800-2841): Determines which baseline window to use per channel
4. **`main()`** (lines 2844-2866): Passes baseline windows to analyzer
5. **Main execution** (lines 2901-2905): Passes baseline windows to `main()`

**Implementation:**
```python
# NEW CODE (lines 2816-2832)
if channel in fluorescence_channels:
    baseline_window = self.fluorescence_baseline_window  # -1 to 0s
else:  # behavioral channels
    baseline_window = self.baseline_window  # -6 to -4s

self.create_heatmap(pivot_data, session_name, event_name, channel, 
                    heatmap_path, baseline_window)  # Pass appropriate window
```

**Console Output:**
```
Saved z_470 heatmap (baseline: -1s to 0s)
Saved Pupil.Diameter_eye1 heatmap (baseline: -6s to -4s)
```

**Result:**
- ✅ Fluorescence heatmaps: Use -1 to 0s baseline
- ✅ Pupil/eye heatmaps: Use -6 to -4s baseline
- ✅ Consistent with aligned baseline plots
- ✅ User can adjust both windows from parameter cell

---

## ✅ Changed Files

| File | Lines Changed | Description |
|------|---------------|-------------|
| `SANDBOX_2_noSLEAP_BATCH-alignment.py` | 136-141 | Added `fluorescence_baseline_window` parameter |
| `SANDBOX_2_noSLEAP_BATCH-alignment.py` | 2220-2241 | Updated `PhotometryAnalyzer.__init__()` to accept baseline windows |
| `SANDBOX_2_noSLEAP_BATCH-alignment.py` | 2398-2422 | Updated `create_heatmap()` to accept baseline_window parameter |
| `SANDBOX_2_noSLEAP_BATCH-alignment.py` | 2800-2841 | Updated `_create_all_heatmaps()` to use channel-specific baselines |
| `SANDBOX_2_noSLEAP_BATCH-alignment.py` | 2844-2866 | Updated `main()` to accept and pass baseline windows |
| `SANDBOX_2_noSLEAP_BATCH-alignment.py` | 2888-2906 | Updated `process_aligned_data_folders()` signature and docstring |
| `SANDBOX_2_noSLEAP_BATCH-alignment.py` | 2901-2905 | Updated main execution to pass baseline windows |
| `SANDBOX_2_noSLEAP_BATCH-alignment.py` | 3000 | Pass `fluorescence_baseline_window` in function call |
| `SANDBOX_2_noSLEAP_BATCH-alignment.py` | 3231-3261 | Updated `baseline_aligned_data_simple()` signature and docstring |
| `SANDBOX_2_noSLEAP_BATCH-alignment.py` | 3265-3320 | Updated `baseline_dataframe()` to use two separate baseline windows |
| `SANDBOX_2_noSLEAP_BATCH-alignment.py` | 3324-3336 | Pass `fluorescence_baseline_window` to all 3 baseline_dataframe calls |
| `SANDBOX_2_noSLEAP_BATCH-alignment.py` | 3420-3427 | Pass `fluorescence_baseline_window` in main execution |

---

## 🧪 Testing

**Before running analysis, verify parameters:**
```python
print(f"Behavioral baseline window: {baseline_window}")
print(f"Fluorescence baseline window: {fluorescence_baseline_window}")
```

**Expected output:**
```
Behavioral baseline window: (-6, -4)
Fluorescence baseline window: (-2, 0)
```

**After running, check CSV files contain correct baseline columns with reasonable values.**

---

## 📝 Notes

- Both baseline windows have the same duration (2 seconds) but different positions
- Fluorescence signals are more sensitive to timing, hence closer baseline
- All signals in the same category use the same baseline window for consistency
- Parameters are externalized for easy adjustment without code changes

