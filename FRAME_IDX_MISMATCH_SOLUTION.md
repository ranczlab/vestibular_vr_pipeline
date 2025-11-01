# Solution: Frame Index Mismatch Between DataFrame and Concatenated Video

## Problem Identified

**User's observations:**
1. Concatenated video: Frame numbers start at 0 and continue continuously (0, 1, 2, 3... up to N)
2. All frames from original files included (even ones SLEAP didn't process)
3. Dropped frames exist in both VideoData1 and VideoData2
4. Early blinks match perfectly, later ones don't
5. Example: Video frame 27871 → Detection outputs 27926 (55 frame offset)

## Root Cause

When `fill_with_empty_rows_based_on_index()` is called (line 105 in load_and_process.py), it:
1. Creates rows for EVERY frame_idx value from 0 to max(frame_idx)
2. Fills gaps with NaN values
3. **These NaN-filled rows are included in the count, but don't correspond to actual video frames**

**The critical bug:**
- Line 76: `last_sleap_frame_idx_vd1 = read_vd1_sleap_dfs[-1]['frame_idx'].iloc[-1] + 1`
- This uses the MAXIMUM frame_idx value (after gap filling) as the offset for the next file
- But if there were dropped frames that got filled with NaN rows, this offset is TOO LARGE

**Example Scenario:**
```
Video File 1:
  - Actual video: 5000 frames (0-4999)
  - SLEAP processed: 4980 frames (20 frames dropped)
  - frame_idx in SLEAP: 0-4979 (with gaps where frames were dropped)
  - After fill_with_empty_rows: Creates rows for frame_idx 0-4979 (including NaN-filled gaps)
  - DataFrame has 4980 rows with frame_idx 0-4979

Video File 2:
  - Actual video: 5000 frames (0-4999)
  - SLEAP processed: 4995 frames (5 frames dropped)
  - frame_idx in SLEAP: 0-4994 (with gaps)
  - Offset calculation: last_sleap_frame_idx_vd1 = 4979 + 1 = 4980
  - frame_idx adjusted: 4980-9974 (4980 + 0 to 4980 + 4994)
  - After fill_with_empty_rows: Creates rows for frame_idx 4980-9974
  - DataFrame has 4995 rows with frame_idx 4980-9974

Total DataFrame: frame_idx 0-9974 (9975 rows)

Concatenated Video:
  - Video 1: frames 0-4999 (5000 frames - ALL frames, including dropped ones)
  - Video 2: frames 5000-9999 (5000 frames - ALL frames)
  - Total: 10000 frames (0-9999)
```

**The mismatch:**
- DataFrame thinks there are 9975 frames (frame_idx 0-9974)
- Actual concatenated video has 10000 frames (0-9999)
- Offset at frame 5000 of video: DataFrame frame_idx ~4979, Video frame 5000
- **Drift accumulates**: By the end, DataFrame frame_idx 9974 → Video frame 9999, but DataFrame thinks it goes higher

## The Actual Problem in the Code

The issue is in how `last_sleap_frame_idx_vd1` is calculated:

**Current (WRONG) approach (line 76):**
```python
last_sleap_frame_idx_vd1 = read_vd1_sleap_dfs[-1]['frame_idx'].iloc[-1] + 1
```
This uses the maximum frame_idx value, which includes NaN-filled placeholders.

**Correct approach should use:**
- The ACTUAL number of frames in the concatenated video files up to that point
- OR: Track the original video file frame counts and use those for offset calculation

## Solution Strategy

We need to calculate frame offsets based on the ACTUAL number of frames in the original video files, not based on the processed frame_idx values.

**Option 1: Use VideoData CSV frame counts**
- The VideoData CSV files contain frame information from the original video files
- Use `Value.ChunkData.FrameID` (before reset) to determine actual frame counts
- Calculate offset based on actual video file lengths

**Option 2: Calculate cumulative frame count from actual video files**
- When loading, count actual frames in each video file
- Use cumulative count for offset calculation

**Option 3: Map frame_idx to video frame numbers**
- Track how many frames each original video file has
- Create a mapping: DataFrame frame_idx → Concatenated video frame number
- Account for dropped frames in the mapping

## Recommended Fix

Since VideoData CSV files contain `Value.ChunkData.FrameID` which comes from the HARP acquisition system and represents the actual frame numbers from the original video files, we should:

1. **Track actual frame counts per file** from VideoData CSV before resetting frame_idx
2. **Use cumulative actual frame counts** for offset calculation instead of frame_idx max values
3. **Store this mapping** so we can convert DataFrame frame_idx back to concatenated video frame numbers

This would ensure that:
- DataFrame frame_idx 4979 → Concatenated video frame 4979 (from video 1)
- DataFrame frame_idx 5000 → Concatenated video frame 5000 (from video 2)

But wait... if Video 1 has 5000 actual frames and Video 2 has 5000 actual frames, then:
- Video 1 frames: 0-4999
- Video 2 frames: 5000-9999 in concatenated video

So if DataFrame frame_idx is 5000, it should map to concatenated video frame 5000, which is the first frame of video 2.

The issue is that if Video 1 only had 4980 processed frames (20 dropped), the DataFrame frame_idx goes 0-4979, then when we offset video 2, we start at 4980. But the concatenated video has frame 4980-4999 from video 1 that weren't processed by SLEAP.

So the fix needs to: **Use the ORIGINAL video file frame counts (from VideoData CSV) to calculate offsets, not the processed SLEAP frame counts.**

