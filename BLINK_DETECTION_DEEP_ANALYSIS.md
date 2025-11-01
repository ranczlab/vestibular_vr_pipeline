# Deep Analysis: Blink Detection Frame Number Mismatch

## Problem Statement
The frame numbers reported in blink detection QC (e.g., "frames 97999-98013") don't match the actual frame numbers in the concatenated video viewed in Fiji, especially later in the video.

## Critical Discovery
When pandas index and frame_idx are the same, it indicates the underlying issue is **NOT** about pandas index vs frame_idx mismatch, but rather about **what frame_idx actually represents** vs **what the concatenated video frame numbers represent**.

## Frame Index Flow Analysis

### 1. Data Loading (load_videography_data)

**VideoData CSV files:**
- Contains `Value.ChunkData.FrameID` which comes from HARP video acquisition system
- Line 90-91: Renamed to `frame_idx`
- Line 94-95: **Reset to start at 0**: `read_vd1_dfs['frame_idx'] = read_vd1_dfs['frame_idx'] - read_vd1_dfs['frame_idx'].iloc[0]`
- This makes frame_idx start at 0 for the first video file

**SLEAP CSV files:**
- Contains `frame_idx` that represents frames processed by SLEAP
- Line 75-76: Adjusted sequentially: `read_vd1_sleap_dfs[-1]['frame_idx'] = read_vd1_sleap_dfs[-1]['frame_idx'] + last_sleap_frame_idx_vd1`
- This makes frame_idx continuous across concatenated files

**Merge operation (line 117):**
- Merges VideoData and SLEAP on `frame_idx`
- The resulting DataFrame has frame_idx values that are continuous across all files

### 2. Gap Filling (fill_with_empty_rows_based_on_index)

**Critical Issue Location: Lines 98-105**

```python
if read_vd1_sleap_dfs.index[-1] != read_vd1_sleap_dfs['frame_idx'].iloc[-1]:
    frame_count = read_vd1_sleap_dfs['frame_idx'].iloc[-1] + 1
    row_count = read_vd1_sleap_dfs.index[-1] + 1
    dropped_frames_vd1 = frame_count - row_count
    read_vd1_sleap_dfs = fill_with_empty_rows_based_on_index(read_vd1_sleap_dfs)
```

**What fill_with_empty_rows_based_on_index does:**
- Creates rows for EVERY frame_idx value from 0 to max(frame_idx)
- Fills gaps with NaN values
- **Result**: If original data had frame_idx: [0, 1, 2, 5, 6, 7] (frames 3-4 missing)
  - After filling: frame_idx: [0, 1, 2, 3, 4, 5, 6, 7] (rows 3-4 are NaN-filled placeholders)

### 3. The Fundamental Mismatch

**frame_idx in DataFrame:**
- Represents a **logical sequence** that includes filled gaps
- Values: 0, 1, 2, 3, 4, 5, ... (continuous, but some rows are NaN-filled placeholders)
- Includes rows for frames that don't exist in the actual video

**Concatenated Video Frame Numbers (in Fiji):**
- Represents **actual video frames** in the concatenated file
- Values: 0, 1, 2, 3, 4, 5, ... (continuous, NO gaps, every number = real frame)
- Does NOT have placeholders for missing frames

**Example Scenario:**
```
Original Video File 1: 1000 frames (some might be dropped during acquisition)
SLEAP processes: 995 frames (5 frames dropped)
frame_idx in SLEAP: 0-994 (but with gaps)
After fill_with_empty_rows: frame_idx 0-994 (with NaN-filled rows for gaps)

Original Video File 2: 1000 frames
SLEAP processes: 998 frames
frame_idx adjusted: 995-1992 (continuous from file 1)
After fill_with_empty_rows: frame_idx 995-1992 (with NaN-filled rows)

DataFrame has: frame_idx 0-1992 (some are NaN placeholders)
Concatenated video has: frame 0-1992 (all real frames)
```

But wait - if SLEAP missed frames, those frames still exist in the original video files. When videos are concatenated, ALL frames from both files are included, including the ones SLEAP didn't process.

**THE REAL PROBLEM:**
When multiple video files are concatenated in Fiji:
- Video 1 has frames 0-999 (1000 frames)
- Video 2 has frames 0-999 (1000 frames)  
- Concatenated video has frames 0-1999 (2000 frames total)

But the frame_idx in the DataFrame was constructed by:
1. Taking SLEAP frame_idx from file 1 (0-994 if 5 frames dropped)
2. Adjusting file 2's frame_idx to start at 995
3. Filling gaps with NaN rows

However, the **actual concatenated video** has all 2000 frames continuously numbered 0-1999. The frame_idx values (0-1992 in this example) don't account for the fact that when concatenating videos, frame numbers are continuous from 0 in the final video.

**Key Insight:**
The `frame_idx` represents the **logical frame position in the combined dataset** (accounting for dropped frames within individual files), but the **concatenated video frame numbers** represent **absolute frame positions in the final concatenated video file** (which includes ALL frames, even those SLEAP didn't process).

## Root Cause Hypothesis

The mismatch occurs because:
1. SLEAP processes individual video files and assigns frame_idx values
2. When videos are concatenated externally (by user in Fiji), the frame numbers reset to 0 for each new video file in the sequence
3. The DataFrame's frame_idx assumes continuous numbering across concatenated files
4. But the actual concatenated video has frame numbers that restart at 0 for each original video file

**Alternative Hypothesis:**
The `Value.ChunkData.FrameID` from VideoData CSV might represent frame numbers from the video acquisition system, which could have different numbering than the actual video file frame numbers.

## Questions to Answer

1. **How does the user concatenate videos?** 
   - Do frame numbers restart at 0 for each original file?
   - Or are they continuous across all files?

2. **What does Value.ChunkData.FrameID actually represent?**
   - Frame number in the original video file?
   - Frame number from the acquisition system (HARP)?
   - Some other identifier?

3. **When SLEAP processes videos, does it use the original video file frame numbers or does it create its own numbering?**

4. **Are there dropped frames?** 
   - The fill_with_empty_rows function suggests there might be frames that SLEAP didn't process
   - But do those frames still exist in the original video files?

## Solution Approach

To fix this, we need to:
1. **Understand the relationship** between:
   - Original video file frame numbers
   - SLEAP frame_idx values  
   - Concatenated video frame numbers
   - Value.ChunkData.FrameID values

2. **Track frame number mapping** through the concatenation process:
   - Keep track of how many frames each original video file has
   - Map DataFrame frame_idx to concatenated video frame numbers

3. **Potentially add a new column** that represents the actual concatenated video frame number, calculated based on:
   - Original file boundaries
   - Frame counts per file
   - Position within the concatenated sequence

