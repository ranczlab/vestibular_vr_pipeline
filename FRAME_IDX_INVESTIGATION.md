# Frame Index Investigation - Critical Questions

## Current Logic Analysis

### VideoData Frame Reset (Lines 126-141)

**File 1:**
- Line 129: `frame_idx = FrameID - FrameID[0]`
- Assumes first FrameID should map to frame 0
- Creates frame_idx: [0, 1, 2, ..., N-1] where N = len(df)

**File 2:**
- Line 132: `frame_idx = FrameID - FrameID[0] + offset`
- Offset = len(file1_df) = actual_frame_count of file 1
- Creates frame_idx: [offset, offset+1, ..., offset+N-1]

## Critical Assumptions Being Made

1. **`len(vd1_df)` equals the number of frames in the video file**
   - This assumes VideoData CSV has one row per video frame
   - What if some frames don't have CSV rows?

2. **`Value.ChunkData.FrameID - FrameID[0]` creates correct 0-based indices**
   - This assumes FrameID[0] represents the first frame we want to map to 0
   - What if FrameID values have gaps?

3. **Frame counts align perfectly between VideoData CSV and actual video files**
   - What if CSV has more/fewer rows than video frames?

## Potential Root Cause

**The issue might be that `len(vd1_df)` (number of CSV rows) doesn't equal the actual number of frames in the video file!**

For example:
- Video file has 5000 frames
- VideoData CSV might have 5000 rows (if every frame is logged)
- OR VideoData CSV might have fewer rows (if some frames weren't logged)
- OR VideoData CSV might have more rows (if there are duplicates or extra entries)

If `len(vd1_df) != actual_video_frame_count`, then:
- Offset calculations will be wrong
- Frame_idx values won't match concatenated video frame numbers

## The Real Question

**How can we determine the actual number of frames in each video file?**

Options:
1. Count rows in VideoData CSV (current approach) - assumes 1:1 mapping
2. Use max(Value.ChunkData.FrameID) - but this might have gaps
3. Count actual frames in the .avi file (would require video reading)
4. Use metadata or file size calculations

## Another Possibility

What if `Value.ChunkData.FrameID` already represents the correct frame numbers we should use (not just within a file, but in some global sense)? Then resetting would be wrong!

For example, if FrameID represents:
- Frame number from the start of the recording session (continuous across files)
- Then we shouldn't reset it - we should use it directly!

But that doesn't make sense if files are processed separately...

Actually, wait. I think the real issue might be simpler: **What if VideoData CSV rows don't correspond 1:1 to video frames?** What if there are extra rows or missing rows?

