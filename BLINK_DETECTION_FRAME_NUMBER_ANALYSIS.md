# Blink Detection Frame Number Mismatch Analysis

## Problem Description
When QC'ing blink detection results, the frame numbers reported in `blink_detection_QC.txt` do not match the actual frame numbers in the concatenated video, especially later in the video. Early blinks match correctly, but later ones are offset.

## Root Cause Analysis

### 1. **Index vs Frame Number Confusion**

**In `load_videography_data()` (sleap/load_and_process.py):**
- Lines 75-76, 81-82: When concatenating multiple SLEAP files, the `frame_idx` column is adjusted to be continuous across files
- Line 85-86: After concatenation, `reset_index().drop(columns='index')` is called, which **resets the pandas integer index** to start at 0
- **Result**: The DataFrame has:
  - **Pandas index**: 0, 1, 2, 3, ... (reset to continuous integers)
  - **`frame_idx` column**: Contains actual frame numbers (may have gaps, may not align with index)

**In `find_blink_segments()` (sleap/processing_functions.py):**
- Lines 137-138: Uses `np.where()` on pandas Series indices
- **Returns**: `start_idx` and `end_idx` are **pandas DataFrame integer positions** (0, 1, 2, 3...), NOT actual frame numbers

**In blink detection code (SANDBOX file, lines 747-788):**
- Line 749-750: `VideoData1['Seconds'].iloc[start_idx]` - Uses `.iloc[]` which accesses by pandas position
- Line 770: `VideoData1.loc[VideoData1.index[start_idx:end_idx+1], ...]` - Uses pandas index
- **Line 786**: `print(f"    Blink {i}: frames {start_idx}-{end_idx}...")` - **PRINTS PANDAS INDEX POSITIONS, NOT ACTUAL FRAME NUMBERS!**

### 2. **Why It Gets Worse Over Time**

When multiple videos are concatenated:
- If `fill_with_empty_rows_based_on_index()` (line 105 in load_and_process.py) is called, it inserts NaN rows for missing frames
- These inserted rows have pandas index positions but maintain the correct `frame_idx` values
- **Example scenario:**
  - Video 1 has frames 0-499 (pandas index 0-499, frame_idx 0-499)
  - Video 2 has frames 0-499 (pandas index 500-999 after concat, but frame_idx adjusted to 500-999)
  - If Video 1 had 5 dropped frames (filled with NaN), pandas index still increments, but frame_idx has gaps
  - At pandas index 1000, frame_idx might be 1005 (due to earlier gaps)
  - **Reported blink at "frames 1000-1010" is actually at frame_idx 1005-1015 in the video**

### 3. **Critical Code Locations**

**Issue locations in SANDBOX_1_1_Loading_+_Saccade_detection.py:**

1. **Line 749-750, 774-775, 846-847, 871-872**: Accessing time/data using `.iloc[start_idx]`
   - Uses pandas position, which is correct for data access
   - But frame numbers should come from `frame_idx` column

2. **Line 786, 883**: Printing frame numbers
   ```python
   print(f"    Blink {i}: frames {start_idx}-{end_idx} ({blink['length']} frames, ...")
   ```
   - **BUG**: Prints `start_idx` and `end_idx` which are pandas positions
   - **SHOULD**: Print `VideoData1['frame_idx'].iloc[start_idx]` and `VideoData1['frame_idx'].iloc[end_idx]`

3. **Line 770, 867**: Marking blinks
   ```python
   VideoData1.loc[VideoData1.index[start_idx:end_idx+1], columns_of_interest] = np.nan
   ```
   - This is actually correct - uses pandas index to mark blinks
   - But the printed frame numbers are wrong

### 4. **Potential Additional Issues**

1. **Missing frame_idx column check**: The code assumes `frame_idx` exists but never verifies it
2. **No validation**: No check that pandas index aligns with frame_idx (which it won't if frames were dropped/filled)

## Recommended Fixes

### Primary Fix: Use frame_idx Column for Reporting

Replace frame number reporting to use the actual `frame_idx` column:

```python
# Instead of:
print(f"    Blink {i}: frames {start_idx}-{end_idx} ...")

# Use:
actual_start_frame = VideoData1['frame_idx'].iloc[start_idx]
actual_end_frame = VideoData1['frame_idx'].iloc[end_idx]
print(f"    Blink {i}: frames {actual_start_frame}-{actual_end_frame} (pandas idx {start_idx}-{end_idx}, ...")
```

### Secondary Checks:

1. **Verify frame_idx exists** before using it
2. **Add diagnostic output** showing the relationship between pandas index and frame_idx
3. **Check for gaps** in frame_idx that might cause misalignment

## Files to Modify

1. **SANDBOX_1_1_Loading_+_Saccade_detection.py**:
   - Lines 786-788: VideoData1 blink reporting
   - Lines 883-885: VideoData2 blink reporting
   - Consider adding frame_idx validation at the start of blink detection

2. **Consider modifying load_and_process.py**:
   - Document that frame_idx should be used for video frame references
   - Add warning if pandas index doesn't align with frame_idx after concatenation

