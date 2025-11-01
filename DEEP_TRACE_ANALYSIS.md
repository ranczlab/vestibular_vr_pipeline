# Deep Trace: Frame Index Flow Analysis

## User's Observation
- Early blinks match perfectly
- Later blinks don't match
- Example: Video frame 27871 → Detection outputs 27926 (55 frame offset)

## Critical Question
**What does `Value.ChunkData.FrameID` actually represent?**

Looking at line 129 and 132:
```python
read_vd1_dfs[i]['frame_idx'] = read_vd1_dfs[i]['frame_idx'] - read_vd1_dfs[i]['frame_idx'].iloc[0]
```

This assumes that the FIRST row of VideoData CSV corresponds to frame 0 of the video file. But what if:
- `Value.ChunkData.FrameID` doesn't start at 0?
- `Value.ChunkData.FrameID` has gaps?
- `Value.ChunkData.FrameID` represents something other than video file frame numbers?

## Potential Issue #1: Value.ChunkData.FrameID Reset Logic

Lines 129, 132, 138, 140 reset frame_idx by subtracting the first value:
- This assumes the first value corresponds to frame 0
- But if VideoData CSV doesn't start at frame 0, this will be wrong
- The reset creates frame_idx: [0, 1, 2, ..., N-1] for each file
- Then offset is applied: File 1: 0-4999, File 2: 5000-9999

**But wait**: If `Value.ChunkData.FrameID` already represents continuous frame numbers from the acquisition system, and we're resetting it, we might be losing information!

## Potential Issue #2: SLEAP Offset vs VideoData Offset Mismatch

SLEAP offset calculation (line 91):
- Uses `last_video_frame_offset_vd1` which is based on actual frame counts
- For file 2: offset = 5000

VideoData offset calculation (line 132):
- Uses `vd1_offset` which is also based on actual frame counts
- For file 2: offset = 5000

These should match, but let me verify the logic...

Actually, I see a potential bug! Look at line 133:
```python
vd1_offset += video_frame_counts_vd1[i] if i < len(video_frame_counts_vd1) else len(read_vd1_dfs[i])
```

This updates `vd1_offset` AFTER setting the frame_idx. So:
- i=0: frame_idx = original - first, then vd1_offset = 5000
- i=1: frame_idx = original - first + 5000, then vd1_offset = 10000

This looks correct actually.

## Potential Issue #3: Gap Filling After Concatenation

After concatenation, gap filling happens (line 160). This fills gaps WITHIN the SLEAP data based on frame_idx. But the problem might be that when we merge VideoData and SLEAP on frame_idx, there could be mismatches if:
- VideoData has frame_idx values that SLEAP doesn't have (because SLEAP didn't process all frames)
- The merge is an inner join, so only matching frame_idx values are kept

But wait - that should be fine. If VideoData has frame_idx 0-9999 and SLEAP has frame_idx 0-9999 (after extension), they should merge correctly.

## Potential Issue #4: The Real Problem Might Be Different

Actually, wait. The user said the offset gets worse over time. At frame 27871, the offset is 55 frames (27926 - 27871). 

If the offset is consistent (always 55 frames ahead), that suggests a constant shift. But if it gets worse over time, that suggests a cumulative error or drift.

Let me think about what could cause drift:
- If frame_idx offset calculation is wrong, the error would accumulate
- If gap filling creates wrong frame_idx values, that could cause drift
- If the extension logic is wrong, that could cause issues

Actually, I think I need to understand: **What does the concatenated video frame number actually represent?**
- Is it the frame number in the concatenated video file (0, 1, 2, ..., N)?
- Or is it something else?

And **what should frame_idx represent after all the processing?**
- It should match the concatenated video frame numbers (0, 1, 2, ..., N)

## Potential Issue #5: Frame Index Reset Logic Issue

I notice that line 129 resets to start at 0:
```python
read_vd1_dfs[i]['frame_idx'] = read_vd1_dfs[i]['frame_idx'] - read_vd1_dfs[i]['frame_idx'].iloc[0]
```

But what if `Value.ChunkData.FrameID` represents the actual frame number in the original video file (already 0-based)? Then subtracting the first value would be wrong if the first frame isn't frame 0!

For example:
- Video file has frames 0-4999
- VideoData CSV might have FrameID: [0, 1, 2, ..., 4999] → Reset works
- OR VideoData CSV might have FrameID: [100, 101, 102, ..., 5099] → Reset creates wrong values!

This is a critical assumption that might be wrong!

