# Critical Bug Analysis: Frame Index Reset Logic

## The Problem

Looking at lines 129 and 132, there's a potential critical bug:

```python
if i == 0:
    read_vd1_dfs[i]['frame_idx'] = read_vd1_dfs[i]['frame_idx'] - read_vd1_dfs[i]['frame_idx'].iloc[0]
else:
    read_vd1_dfs[i]['frame_idx'] = read_vd1_dfs[i]['frame_idx'] - read_vd1_dfs[i]['frame_idx'].iloc[0] + vd1_offset
```

## The Issue

**This assumes that `Value.ChunkData.FrameID` in each CSV file should be reset to start at 0.** 

But what if `Value.ChunkData.FrameID` represents something different? For example:
- If FrameID represents the actual frame number in the ORIGINAL video file (already 0-based), then resetting might be correct
- But if FrameID represents a continuous counter from the acquisition system across all files, resetting would be WRONG

## More Importantly: The Real Issue

**When you reset VideoData frame_idx, you're losing information about what the original frame numbers were.** 

The correct approach should be:
- VideoData CSV File 1: FrameID [0, 1, 2, ..., 4999] → After reset: [0, 1, 2, ..., 4999] ✓
- VideoData CSV File 2: FrameID [0, 1, 2, ..., 4999] → After reset + offset: [5000, 5001, ..., 9999] ✓

BUT: What if VideoData CSV File 2's FrameID doesn't start at 0? What if it continues from File 1?

## Critical Question

**What does `Value.ChunkData.FrameID` actually represent?**
- Frame number in the original video file (restarts at 0 for each file)?
- Continuous frame counter from acquisition system (continues across files)?
- Something else?

## Potential Fix

Instead of resetting based on the first value, we should:
1. For the first file: Map FrameID values to continuous 0-based indices (assuming they represent frame numbers in that file)
2. For subsequent files: We need to know if FrameID restarts at 0 or continues

But actually, wait - if VideoData CSV files are created per video file, and each video file has frames 0-N, then FrameID should restart at 0 for each file. So the reset logic might be correct.

## Another Critical Issue: Gap Filling Logic

The `fill_with_empty_rows_based_on_index` function (line 13-21) creates rows for EVERY frame_idx value from 0 to max. But this happens AFTER concatenation, when frame_idx values have already been adjusted.

For example:
- File 1 SLEAP: frame_idx [0, 1, 2, 5, 6, 7] (frames 3-4 dropped)
- After offset (if file 1): [0, 1, 2, 5, 6, 7]
- File 2 SLEAP: frame_idx [0, 1, 3, 4, 5] (frame 2 dropped)  
- After offset 5000: [5000, 5001, 5003, 5004, 5005]
- After concatenation: [0, 1, 2, 5, 6, 7, 5000, 5001, 5003, 5004, 5005]
- Gap filling creates: [0-7, 5000-5005] - **MISSING 5002!**

This could be a problem! The gap filling happens AFTER concatenation, so it only fills gaps within each file's range, not gaps BETWEEN files!

## The Real Bug

Gap filling should happen PER FILE before concatenation, not after! Otherwise, gaps between file boundaries aren't filled correctly.

