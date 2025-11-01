# Critical Issue Found: Gap Filling After Concatenation

## Problem Location: Lines 151-167

## The Issue

**`fill_with_empty_rows_based_on_index()` is called AFTER concatenation** (line 146-147), and it fills gaps in the concatenated SLEAP dataframe based on the **SLEAP frame_idx range**, not the **actual video frame range**.

### What happens:

1. **After my fix (lines 89-94)**: SLEAP frame_idx is correctly offset using actual video frame counts
   - Video 1: 5000 actual frames → SLEAP frame_idx offset correctly
   - Video 2: 5000 actual frames → SLEAP frame_idx offset by 5000

2. **Concatenation (line 146)**: SLEAP dataframes are concatenated
   - Combined SLEAP has frame_idx values like: [0, 1, 2, ..., 4979, 5000, 5001, ..., 9994]
   - Note: Frame_idx 4980-4999 might be missing (if Video 1 had dropped frames that SLEAP didn't process)

3. **Gap filling (line 159)**: `fill_with_empty_rows_based_on_index()` is called
   - Takes max frame_idx (e.g., 9994)
   - Creates rows for frame_idx 0 to 9994 (filling gaps)
   - **BUT**: This creates frame_idx values that don't match actual video frames!

### Example Scenario:

**Video 1:**
- Actual video: 5000 frames (0-4999)
- SLEAP processed: 4980 frames (20 dropped)
- After offset: SLEAP frame_idx 0-4979

**Video 2:**
- Actual video: 5000 frames (0-4999)
- SLEAP processed: 4995 frames (5 dropped)
- After offset: SLEAP frame_idx 5000-9994

**After concatenation:**
- SLEAP dataframe has frame_idx: [0-4979, 5000-9994]
- Missing: frame_idx 4980-4999 (these frames exist in video but SLEAP didn't process them)

**After fill_with_empty_rows_based_on_index():**
- Creates rows for frame_idx 0-9994 (filling gap 4980-4999 with NaN)
- This looks correct!

**BUT WAIT - The Real Problem:**

The issue is that `fill_with_empty_rows_based_on_index()` fills gaps **within the SLEAP frame_idx range**, but if there are gaps at the boundaries or if the logic is wrong, it could create frame_idx values that exceed the actual video frame count.

Actually, I think the real issue might be different. Let me reconsider...

**The actual problem might be:**

When `fill_with_empty_rows_based_on_index()` creates rows, it's creating frame_idx values based on the max value in the SLEAP data. But if SLEAP has frame_idx values that were already offset correctly, but then there's a gap filling operation that doesn't account for the actual video structure, it could create frame_idx values that go beyond the actual concatenated video frame count.

Or more likely: The gap filling happens per-file BEFORE concatenation, not after. But wait, looking at the code, it happens after concatenation (line 159 is after line 146).

Actually, I need to check if gap filling should happen per-file before concatenation, not after.

