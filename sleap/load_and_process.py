import os
import numpy as np
import pandas as pd
from ellipse import LsqEllipse
from scipy.ndimage import median_filter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from . import horizontal_flip_script
import matplotlib.pyplot as plt
from pathlib import Path
from harp_resources import process
import aeon.io.api as api


def load_df(path):
    return pd.read_csv(path)


def fill_with_empty_rows_based_on_index(df, new_index_column_name="frame_idx"):
    complete_index = pd.Series(range(0, df[new_index_column_name].max() + 1))
    df.set_index(new_index_column_name, inplace=True)
    df = df.reindex(complete_index, fill_value=np.nan)
    df.reset_index(inplace=True)
    df.rename(columns={"index": new_index_column_name}, inplace=True)

    return df


def load_videography_data(path, debug=False):
    # print('ℹ️ Load_and_process.load_videography_data() function expects the following format of SLEAP outputs:')
    # print('"VideoData1_1904-01-01T23-59-59.sleap.csv"')
    # print('"..."')
    print(
        "ℹ️ Make sure the SLEAP files follow this convention: VideoData1_1904-01-01T00-00-00.sleap.csv"
    )
    # print('\n')
    # print('RESULTS:')

    # Listing filenames in the folders
    # Exclude: .avi files, .sleap files, .slp files, and any hidden files (starting with .)
    vd1_files, read_vd1_dfs, read_vd1_sleap_dfs = [], [], []
    vd2_files, read_vd2_dfs, read_vd2_sleap_dfs = [], [], []

    for e in os.listdir(path / "VideoData1"):
        if (
            not ".avi" in e
            and not ".sleap" in e
            and e[-4:] != ".slp"
            and not e.startswith(".")
        ):
            vd1_files.append(e)
    for e in os.listdir(path / "VideoData2"):
        if (
            not ".avi" in e
            and not ".sleap" in e
            and e[-4:] != ".slp"
            and not e.startswith(".")
        ):
            vd2_files.append(e)

    vd1_has_sleap, vd2_has_sleap = False, False
    for e in os.listdir(path / "VideoData1"):
        if not e.startswith(".") and ".sleap" in e:
            vd1_has_sleap = True
    for e in os.listdir(path / "VideoData2"):
        if not e.startswith(".") and ".sleap" in e:
            vd2_has_sleap = True
    print(f"\nOutputs of SLEAP found in VideoData1: {vd1_has_sleap}")
    print(f"Outputs of SLEAP found in VideoData2: {vd2_has_sleap}")

    # DEBUG: Print collected files before timestamp parsing
    if debug:
        print(f"\n🔍 DEBUG: Files collected for VideoData1 ({len(vd1_files)} files):")
        for f in vd1_files:
            print(f"  - {f}")
            # Show what would be extracted as timestamp
            try:
                extracted = f.split("_")[1].split(".")[0]
                print(f'    → Extracted timestamp string: "{extracted}"')
            except (IndexError, AttributeError) as e:
                print(f"    → ERROR extracting timestamp: {e}")

        print(f"\n🔍 DEBUG: Files collected for VideoData2 ({len(vd2_files)} files):")
        for f in vd2_files:
            print(f"  - {f}")
            # Show what would be extracted as timestamp
            try:
                extracted = f.split("_")[1].split(".")[0]
                print(f'    → Extracted timestamp string: "{extracted}"')
            except (IndexError, AttributeError) as e:
                print(f"    → ERROR extracting timestamp: {e}")

        print(f"\n🔍 DEBUG: Attempting to parse timestamps...")

    # Sorting filenames chronologically
    # sorted_vd1_files = pd.to_datetime(pd.Series([x.split('_')[1].split('.')[0] for x in vd1_files])).sort_values()
    # sorted_vd2_files = pd.to_datetime(pd.Series([x.split('_')[1].split('.')[0] for x in vd2_files])).sort_values()
    sorted_vd1_files = pd.to_datetime(
        pd.Series([x.split("_")[1].split(".")[0] for x in vd1_files]),
        format="%Y-%m-%dT%H-%M-%S",
    ).sort_values()
    sorted_vd2_files = pd.to_datetime(
        pd.Series([x.split("_")[1].split(".")[0] for x in vd2_files]),
        format="%Y-%m-%dT%H-%M-%S",
    ).sort_values()

    # DEBUG: Print files in CHRONOLOGICAL ORDER (as they will be loaded)
    if debug:
        print(
            f"\n🔍 DEBUG: VideoData1 files (will be loaded in this chronological order):"
        )
        for i, ts in enumerate(sorted_vd1_files.values, 1):
            # Convert numpy datetime64 to pandas Timestamp for strftime
            ts_pd = pd.Timestamp(ts)
            filename = f"VideoData1_{ts_pd.strftime('%Y-%m-%dT%H-%M-%S')}.csv"
            print(f"  {i}. {filename} (timestamp: {ts_pd})")

        print(
            f"\n🔍 DEBUG: VideoData2 files (will be loaded in this chronological order):"
        )
        for i, ts in enumerate(sorted_vd2_files.values, 1):
            # Convert numpy datetime64 to pandas Timestamp for strftime
            ts_pd = pd.Timestamp(ts)
            filename = f"VideoData2_{ts_pd.strftime('%Y-%m-%dT%H-%M-%S')}.csv"
            print(f"  {i}. {filename} (timestamp: {ts_pd})")

    print(
        f"\n📋 LOADING {len(sorted_vd1_files)} VideoData1 file(s) and {len(sorted_vd2_files)} VideoData2 file(s)\n"
    )

    # Reading the csv files in the chronological order
    # Track actual video frame counts for proper offset calculation (fixes concatenated video frame number mismatch)
    # The issue: SLEAP frame_idx is offset by processed frame count, but concatenated video includes ALL frames
    # Solution: Track actual video file frame counts and use them to create proper mapping
    video_frame_counts_vd1, video_frame_counts_vd2 = (
        [],
        [],
    )  # Track frame counts per file
    last_video_frame_offset_vd1, last_video_frame_offset_vd2 = (
        0,
        0,
    )  # Cumulative offsets for concatenated video
    last_sleap_frame_idx_vd1, last_sleap_frame_idx_vd2 = (
        0,
        0,
    )  # to add to consecutive SLEAP logs because frame_idx restarts at 0 in each file

    for row in sorted_vd1_files:
        vd1_df = pd.read_csv(
            path / "VideoData1" / f"VideoData1_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"
        )
        read_vd1_dfs.append(vd1_df)

        # Calculate actual frame count from VideoData CSV (before reset)
        # Value.ChunkData.FrameID represents actual video frame numbers from acquisition system
        # The number of rows represents all frames in the original video file
        actual_frame_count = len(
            vd1_df
        )  # This represents all frames in the original video file
        video_frame_counts_vd1.append(actual_frame_count)

        if vd1_has_sleap:
            vd1_sleap_filename = (
                f"VideoData1_{row.strftime('%Y-%m-%dT%H-%M-%S')}.sleap.csv"
            )
            sleap_df = pd.read_csv(path / "VideoData1" / vd1_sleap_filename)

            # Determine if SLEAP frame_idx is 0-based (per-file) or FrameID-based (continuous camera counter)
            first_sleap_frame_idx = sleap_df["frame_idx"].iloc[0]
            if len(read_vd1_sleap_dfs) == 0:
                # First file - log what we see to help diagnose
                print(
                    f"ℹ️ VideoData1 SLEAP file {len(read_vd1_sleap_dfs) + 1} ({vd1_sleap_filename}): first frame_idx = {first_sleap_frame_idx}"
                )

            # CRITICAL FIX: Gap-fill PER FILE before offsetting to ensure gaps are filled within each file's range
            file_num = len(read_vd1_sleap_dfs) + 1
            max_sleap_frame_idx = sleap_df["frame_idx"].max()
            min_sleap_frame_idx = sleap_df["frame_idx"].min()
            sleap_row_count = len(sleap_df)

            # Check if there are gaps (dropped frames) within this file
            # We only gap-fill WITHIN the range that SLEAP actually processed
            if sleap_row_count < max_sleap_frame_idx + 1:
                # There are gaps within the processed range - fill them
                dropped_frames = (max_sleap_frame_idx + 1) - sleap_row_count
                print(
                    f"   ⚠️ Found {dropped_frames} dropped frames within processed range [0-{max_sleap_frame_idx}] in {vd1_sleap_filename}. Filling gaps."
                )
                sleap_df = fill_with_empty_rows_based_on_index(sleap_df)

            if max_sleap_frame_idx + 1 < actual_frame_count:
                missing_at_end = actual_frame_count - (max_sleap_frame_idx + 1)
                print(
                    f"   ⚠️ SLEAP processed {max_sleap_frame_idx + 1} frames, but video has {actual_frame_count} frames."
                )
                print(
                    f"      Missing {missing_at_end} frames at the end (will extend later)."
                )

            # CRITICAL: Offset SLEAP frame_idx by cumulative actual video frame count
            # This assumes SLEAP frame_idx is 0-based per file (restarts at 0 for each file)
            sleap_df["frame_idx"] = sleap_df["frame_idx"] + last_video_frame_offset_vd1

            read_vd1_sleap_dfs.append(sleap_df)

            # Update offset for next file: use actual video frame count (not SLEAP processed count)
            last_video_frame_offset_vd1 += actual_frame_count

            # Also track SLEAP frame_idx max for internal consistency (if needed elsewhere)
            last_sleap_frame_idx_vd1 = read_vd1_sleap_dfs[-1]["frame_idx"].iloc[-1] + 1

    for row in sorted_vd2_files:
        vd2_df = pd.read_csv(
            path / "VideoData2" / f"VideoData2_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"
        )
        read_vd2_dfs.append(vd2_df)

        # Calculate actual frame count from VideoData CSV (before reset)
        actual_frame_count = len(vd2_df)
        video_frame_counts_vd2.append(actual_frame_count)

        if vd2_has_sleap:
            vd2_sleap_filename = (
                f"VideoData2_{row.strftime('%Y-%m-%dT%H-%M-%S')}.sleap.csv"
            )
            sleap_df = pd.read_csv(path / "VideoData2" / vd2_sleap_filename)

            # Determine if SLEAP frame_idx is 0-based (per-file) or FrameID-based (continuous camera counter)
            first_sleap_frame_idx = sleap_df["frame_idx"].iloc[0]
            if len(read_vd2_sleap_dfs) == 0:
                # First file - log what we see to help diagnose
                print(
                    f"ℹ️ VideoData2 SLEAP file {len(read_vd2_sleap_dfs) + 1} ({vd2_sleap_filename}): first frame_idx = {first_sleap_frame_idx}"
                )

            # CRITICAL FIX: Gap-fill PER FILE before offsetting to ensure gaps are filled within each file's range
            file_num = len(read_vd2_sleap_dfs) + 1
            max_sleap_frame_idx = sleap_df["frame_idx"].max()
            min_sleap_frame_idx = sleap_df["frame_idx"].min()
            sleap_row_count = len(sleap_df)

            # Check if there are gaps (dropped frames) within this file
            # We only gap-fill WITHIN the range that SLEAP actually processed
            if sleap_row_count < max_sleap_frame_idx + 1:
                # There are gaps within the processed range - fill them
                dropped_frames = (max_sleap_frame_idx + 1) - sleap_row_count
                print(
                    f"   ⚠️ Found {dropped_frames} dropped frames within processed range [0-{max_sleap_frame_idx}] in {vd2_sleap_filename}. Filling gaps."
                )
                sleap_df = fill_with_empty_rows_based_on_index(sleap_df)

            if max_sleap_frame_idx + 1 < actual_frame_count:
                missing_at_end = actual_frame_count - (max_sleap_frame_idx + 1)
                print(
                    f"   ⚠️ SLEAP processed {max_sleap_frame_idx + 1} frames, but video has {actual_frame_count} frames."
                )
                print(
                    f"      Missing {missing_at_end} frames at the end (will extend later)."
                )

            # CRITICAL: Offset SLEAP frame_idx by cumulative actual video frame count
            # This assumes SLEAP frame_idx is 0-based per file (restarts at 0 for each file)
            sleap_df["frame_idx"] = sleap_df["frame_idx"] + last_video_frame_offset_vd2

            read_vd2_sleap_dfs.append(sleap_df)

            # Update offset for next file: use actual video frame count (not SLEAP processed count)
            last_video_frame_offset_vd2 += actual_frame_count

            # Also track SLEAP frame_idx max for internal consistency (if needed elsewhere)
            last_sleap_frame_idx_vd2 = read_vd2_sleap_dfs[-1]["frame_idx"].iloc[-1] + 1

    # CRITICAL FIX: Reset VideoData frame_idx BEFORE concatenation
    # Value.ChunkData.FrameID is a continuous camera counter across files (doesn't restart at 0)
    # Example: File 1 FrameID [26654, ..., 31653], File 2 FrameID [31654, ..., 36653]
    # We need to map to concatenated video frame numbers: [0, 1, 2, ..., N-1]
    # Strategy: Capture first file's starting FrameID, then map all files relative to that

    # Capture starting FrameID from first file BEFORE any modifications (for VideoData1)
    first_file_frame_id_vd1 = None
    if len(read_vd1_dfs) > 0:
        # Temporarily rename to access FrameID, then rename back
        temp_df = read_vd1_dfs[0].copy()
        if "Value.ChunkData.FrameID" in temp_df.columns:
            first_file_frame_id_vd1 = temp_df["Value.ChunkData.FrameID"].iloc[0]

    # Capture starting FrameID from first file BEFORE any modifications (for VideoData2)
    first_file_frame_id_vd2 = None
    if len(read_vd2_dfs) > 0:
        temp_df = read_vd2_dfs[0].copy()
        if "Value.ChunkData.FrameID" in temp_df.columns:
            first_file_frame_id_vd2 = temp_df["Value.ChunkData.FrameID"].iloc[0]

    # Reset frame_idx for each VideoData file before concatenation
    # Since FrameID is continuous across files, we simply subtract the first file's starting FrameID
    # This automatically creates continuous frame_idx values: [0, 1, 2, ..., N-1]
    for i in range(len(read_vd1_dfs)):
        read_vd1_dfs[i] = read_vd1_dfs[i].rename(
            columns={"Value.ChunkData.FrameID": "frame_idx"}
        )
        if first_file_frame_id_vd1 is not None:
            read_vd1_dfs[i]["frame_idx"] = (
                read_vd1_dfs[i]["frame_idx"] - first_file_frame_id_vd1
            )

    for i in range(len(read_vd2_dfs)):
        read_vd2_dfs[i] = read_vd2_dfs[i].rename(
            columns={"Value.ChunkData.FrameID": "frame_idx"}
        )
        if first_file_frame_id_vd2 is not None:
            read_vd2_dfs[i]["frame_idx"] = (
                read_vd2_dfs[i]["frame_idx"] - first_file_frame_id_vd2
            )

    # Now concatenate with properly aligned frame_idx values (matching concatenated video frame numbers)
    read_vd1_dfs = pd.concat(read_vd1_dfs).reset_index().drop(columns="index")
    read_vd2_dfs = pd.concat(read_vd2_dfs).reset_index().drop(columns="index")

    # CRITICAL FIX: Gap-fill VideoData frame_idx to ensure consecutive values [0, 1, 2, ..., N-1]
    # This fixes issues where Value.ChunkData.FrameID had gaps or duplicates
    if len(read_vd1_dfs) > 0:
        max_frame_idx_vd1 = read_vd1_dfs["frame_idx"].max()
        expected_rows = max_frame_idx_vd1 + 1
        if len(read_vd1_dfs) < expected_rows:
            # Gap-fill missing frame_idx values
            missing_frames = expected_rows - len(read_vd1_dfs)
            if debug:
                print(
                    f"ℹ️ VideoData1: Gap-filling {missing_frames} missing frame_idx values (max frame_idx: {max_frame_idx_vd1}, rows: {len(read_vd1_dfs)})"
                )
            read_vd1_dfs = fill_with_empty_rows_based_on_index(
                read_vd1_dfs, "frame_idx"
            )
        # Validate monotonicity and check for duplicates
        if read_vd1_dfs["frame_idx"].duplicated().any():
            dup_count = read_vd1_dfs["frame_idx"].duplicated().sum()
            print(
                f"   ⚠️ WARNING: VideoData1 has {dup_count} duplicate frame_idx values after gap-filling!"
            )
            print(f"      This should not happen. Frame_idx should be unique.")
        if not read_vd1_dfs["frame_idx"].is_monotonic_increasing:
            print(
                f"   ⚠️ WARNING: VideoData1 frame_idx is not monotonic after gap-filling!"
            )
            print(
                f"      This should not happen. Frame_idx should be strictly increasing."
            )
        # Re-index to ensure consecutive [0, 1, 2, ..., N-1]
        read_vd1_dfs = read_vd1_dfs.sort_values("frame_idx").reset_index(drop=True)
        # Replace frame_idx with consecutive values to ensure [0, 1, 2, ..., N-1]
        read_vd1_dfs["frame_idx"] = range(len(read_vd1_dfs))

    if len(read_vd2_dfs) > 0:
        max_frame_idx_vd2 = read_vd2_dfs["frame_idx"].max()
        expected_rows = max_frame_idx_vd2 + 1
        if len(read_vd2_dfs) < expected_rows:
            # Gap-fill missing frame_idx values
            missing_frames = expected_rows - len(read_vd2_dfs)
            if debug:
                print(
                    f"ℹ️ VideoData2: Gap-filling {missing_frames} missing frame_idx values (max frame_idx: {max_frame_idx_vd2}, rows: {len(read_vd2_dfs)})"
                )
            read_vd2_dfs = fill_with_empty_rows_based_on_index(
                read_vd2_dfs, "frame_idx"
            )
        # Validate monotonicity and check for duplicates
        if read_vd2_dfs["frame_idx"].duplicated().any():
            dup_count = read_vd2_dfs["frame_idx"].duplicated().sum()
            print(
                f"   ⚠️ WARNING: VideoData2 has {dup_count} duplicate frame_idx values after gap-filling!"
            )
            print(f"      This should not happen. Frame_idx should be unique.")
        if not read_vd2_dfs["frame_idx"].is_monotonic_increasing:
            print(
                f"   ⚠️ WARNING: VideoData2 frame_idx is not monotonic after gap-filling!"
            )
            print(
                f"      This should not happen. Frame_idx should be strictly increasing."
            )
        # Re-index to ensure consecutive [0, 1, 2, ..., N-1]
        read_vd2_dfs = read_vd2_dfs.sort_values("frame_idx").reset_index(drop=True)
        # Replace frame_idx with consecutive values to ensure [0, 1, 2, ..., N-1]
        read_vd2_dfs["frame_idx"] = range(len(read_vd2_dfs))
    if vd1_has_sleap:
        read_vd1_sleap_dfs = (
            pd.concat(read_vd1_sleap_dfs).reset_index().drop(columns="index")
        )
    if vd2_has_sleap:
        read_vd2_sleap_dfs = (
            pd.concat(read_vd2_sleap_dfs).reset_index().drop(columns="index")
        )

    # print('Reading dataframes finished.')

    # CRITICAL FIX: After gap-filling per file and concatenation, extend to total actual video frame count
    # Gap-filling has already been done per file before concatenation, so now we just need to extend
    # if the maximum frame_idx is less than the total actual frames (due to dropped frames at file boundaries)
    if vd1_has_sleap:
        # Calculate total actual frame count from all video files
        total_actual_frames_vd1 = (
            sum(video_frame_counts_vd1) if video_frame_counts_vd1 else 0
        )
        max_frame_idx_vd1 = read_vd1_sleap_dfs["frame_idx"].max()

        # If SLEAP max frame_idx is less than total actual frames, extend with NaN rows
        if max_frame_idx_vd1 < total_actual_frames_vd1 - 1:
            missing_frames = total_actual_frames_vd1 - 1 - max_frame_idx_vd1
            print(
                f"ℹ️ VideoData1: Extending SLEAP dataframe by {missing_frames} frames to match concatenated video frame count ({total_actual_frames_vd1} total frames)"
            )
            # Create additional rows with NaN values for missing frame_idx values
            extended_frame_idx = pd.Series(
                range(max_frame_idx_vd1 + 1, total_actual_frames_vd1)
            )
            extended_df = pd.DataFrame({"frame_idx": extended_frame_idx})
            # Add NaN columns for all other SLEAP columns
            for col in read_vd1_sleap_dfs.columns:
                if col != "frame_idx":
                    extended_df[col] = np.nan
            # Concatenate with existing dataframe
            read_vd1_sleap_dfs = pd.concat(
                [read_vd1_sleap_dfs, extended_df], ignore_index=True
            )
            # Sort by frame_idx to maintain order
            read_vd1_sleap_dfs = read_vd1_sleap_dfs.sort_values(
                "frame_idx"
            ).reset_index(drop=True)

    if vd2_has_sleap:
        # Calculate total actual frame count from all video files
        total_actual_frames_vd2 = (
            sum(video_frame_counts_vd2) if video_frame_counts_vd2 else 0
        )
        max_frame_idx_vd2 = read_vd2_sleap_dfs["frame_idx"].max()

        # If SLEAP max frame_idx is less than total actual frames, extend with NaN rows
        if max_frame_idx_vd2 < total_actual_frames_vd2 - 1:
            missing_frames = total_actual_frames_vd2 - 1 - max_frame_idx_vd2
            print(
                f"ℹ️ VideoData2: Extending SLEAP dataframe by {missing_frames} frames to match concatenated video frame count ({total_actual_frames_vd2} total frames)"
            )
            # Create additional rows with NaN values for missing frame_idx values
            extended_frame_idx = pd.Series(
                range(max_frame_idx_vd2 + 1, total_actual_frames_vd2)
            )
            extended_df = pd.DataFrame({"frame_idx": extended_frame_idx})
            # Add NaN columns for all other SLEAP columns
            for col in read_vd2_sleap_dfs.columns:
                if col != "frame_idx":
                    extended_df[col] = np.nan
            # Concatenate with existing dataframe
            read_vd2_sleap_dfs = pd.concat(
                [read_vd2_sleap_dfs, extended_df], ignore_index=True
            )
            # Sort by frame_idx to maintain order
            read_vd2_sleap_dfs = read_vd2_sleap_dfs.sort_values(
                "frame_idx"
            ).reset_index(drop=True)

    # Merging VideoData csv files and sleap outputs to get access to the HARP timestamps
    # CRITICAL: Use LEFT merge to preserve VideoData frame order and ensure 1:1 mapping
    # This ensures frame_idx positions in the merged dataframe match VideoData frame_idx positions
    vd1_out, vd2_out = (
        read_vd1_dfs[["frame_idx", "Seconds"]],
        read_vd2_dfs[["frame_idx", "Seconds"]],
    )
    if vd1_has_sleap:
        # Use left merge to keep VideoData order and ensure each VideoData row gets matched to SLEAP
        # Sort SLEAP by frame_idx first to ensure proper matching
        read_vd1_sleap_dfs_sorted = read_vd1_sleap_dfs.sort_values(
            "frame_idx"
        ).reset_index(drop=True)
        vd1_out = pd.merge(
            read_vd1_dfs[["frame_idx", "Seconds"]],
            read_vd1_sleap_dfs_sorted,
            on="frame_idx",
            how="left",
        )
        # Check for merge issues
        if len(vd1_out) != len(read_vd1_dfs):
            print(f"   ⚠️ WARNING: VideoData1 row count changed during merge!")
            print(f"      Original VideoData: {len(read_vd1_dfs)} rows")
            print(f"      After merge: {len(vd1_out)} rows")
            print(f"      Difference: {len(vd1_out) - len(read_vd1_dfs)} rows")
        # CRITICAL FIX: Validate frame_idx after merge
        if vd1_out["frame_idx"].duplicated().any():
            dup_count = vd1_out["frame_idx"].duplicated().sum()
            print(
                f"   ⚠️ WARNING: VideoData1 has {dup_count} duplicate frame_idx values after merge!"
            )
            print(f"      This should not happen. Frame_idx should be unique.")
        if not vd1_out["frame_idx"].is_monotonic_increasing:
            print(f"   ⚠️ WARNING: VideoData1 frame_idx is not monotonic after merge!")
            print(
                f"      This should not happen. Frame_idx should be strictly increasing."
            )
    if vd2_has_sleap:
        read_vd2_sleap_dfs_sorted = read_vd2_sleap_dfs.sort_values(
            "frame_idx"
        ).reset_index(drop=True)
        vd2_out = pd.merge(
            read_vd2_dfs[["frame_idx", "Seconds"]],
            read_vd2_sleap_dfs_sorted,
            on="frame_idx",
            how="left",
        )
        # Check for merge issues
        if len(vd2_out) != len(read_vd2_dfs):
            print(f"   ⚠️ WARNING: VideoData2 row count changed during merge!")
            print(f"      Original VideoData: {len(read_vd2_dfs)} rows")
            print(f"      After merge: {len(vd2_out)} rows")
            print(f"      Difference: {len(vd2_out) - len(read_vd2_dfs)} rows")
        # CRITICAL FIX: Validate frame_idx after merge
        if vd2_out["frame_idx"].duplicated().any():
            dup_count = vd2_out["frame_idx"].duplicated().sum()
            print(
                f"   ⚠️ WARNING: VideoData2 has {dup_count} duplicate frame_idx values after merge!"
            )
            print(f"      This should not happen. Frame_idx should be unique.")
        if not vd2_out["frame_idx"].is_monotonic_increasing:
            print(f"   ⚠️ WARNING: VideoData2 frame_idx is not monotonic after merge!")
            print(
                f"      This should not happen. Frame_idx should be strictly increasing."
            )

    if debug:
        print("\n" + "=" * 80)
        print("✅ DIAGNOSTIC SUMMARY: Frame Index Alignment Complete")
        print("=" * 80)
        if len(vd1_out) > 0:
            print(
                f"VideoData1 final dataframe: {len(vd1_out)} rows, frame_idx range {vd1_out['frame_idx'].min()}-{vd1_out['frame_idx'].max()}"
            )
        if len(vd2_out) > 0:
            print(
                f"VideoData2 final dataframe: {len(vd2_out)} rows, frame_idx range {vd2_out['frame_idx'].min()}-{vd2_out['frame_idx'].max()}"
            )
        print("=" * 80 + "\n")

    def _ensure_unique_seconds(df, video_label):
        if "Seconds" in df.columns and df["Seconds"].duplicated().any():
            duplicate_count = df["Seconds"].duplicated().sum()
            duplicated_values = df.loc[
                df["Seconds"].duplicated(keep=False), "Seconds"
            ].unique()
            sample_values = ", ".join(f"{val:.6f}" for val in duplicated_values[:5])
            print(
                f"❗ CRITICAL ERROR: Duplicate timestamps detected in '{video_label}'. "
                f"Found {duplicate_count} duplicate entries. Sample duplicate values: {sample_values}"
            )
            raise ValueError(
                f"Duplicate timestamps detected in '{video_label}'. "
                "Please inspect the source data. Duplicate Seconds values break downstream processing."
            )

    _ensure_unique_seconds(vd1_out, "VideoData1")
    _ensure_unique_seconds(vd2_out, "VideoData2")

    return vd1_out, vd2_out, vd1_has_sleap, vd2_has_sleap


def recalculated_coordinates(point_name, df, reference_subtraced_displacements_dict):
    # Recalculates coordinates of a point at each frame, applying the referenced displacements to the coordinates of the very first frame.
    out_array = np.zeros(
        reference_subtraced_displacements_dict[point_name].shape[0] + 1
    )
    out_array[0] = df[point_name].to_numpy()[0]
    for i, disp in enumerate(reference_subtraced_displacements_dict[point_name]):
        out_array[i + 1] = out_array[i] + disp

    return out_array


def get_referenced_recalculated_coordinates(df):
    columns_of_interest = [
        "left.x",
        "left.y",
        "center.x",
        "center.y",
        "right.x",
        "right.y",
        "p1.x",
        "p1.y",
        "p2.x",
        "p2.y",
        "p3.x",
        "p3.y",
        "p4.x",
        "p4.y",
        "p5.x",
        "p5.y",
        "p6.x",
        "p6.y",
        "p7.x",
        "p7.y",
        "p8.x",
        "p8.y",
    ]
    active_points_x = [
        "center.x",
        "p1.x",
        "p2.x",
        "p3.x",
        "p4.x",
        "p5.x",
        "p6.x",
        "p7.x",
        "p8.x",
    ]
    active_points_y = [
        "center.y",
        "p1.y",
        "p2.y",
        "p3.y",
        "p4.y",
        "p5.y",
        "p6.y",
        "p7.y",
        "p8.y",
    ]

    coordinates_dict = {key: df[key].to_numpy() for key in columns_of_interest}
    displacements_dict = {
        k: np.diff(v) for k, v in coordinates_dict.items()
    }  # in [displacement] = [pixels / frame]

    mean_reference_x = np.stack(
        (displacements_dict["left.x"], displacements_dict["right.x"])
    ).mean(axis=0)
    mean_reference_y = np.stack(
        (displacements_dict["left.y"], displacements_dict["right.y"])
    ).mean(axis=0)

    # Subtracting the displacement of the reference points at each frame
    reference_subtraced_displacements_dict = {
        k: displacements_dict[k] - mean_reference_x for k in active_points_x
    } | {
        k: displacements_dict[k] - mean_reference_y for k in active_points_y
    }  # joining the horizontal and vertical dictionaries into one

    reference_subtraced_coordinates_dict = {
        p: recalculated_coordinates(p, df, reference_subtraced_displacements_dict)
        for p in active_points_x + active_points_y
    }

    return reference_subtraced_coordinates_dict


def rotate_points(points, theta):
    # This is for rotating with an angle of positive theta
    # rotation_matrix = np.array([
    #     [np.cos(theta), -np.sin(theta)],
    #     [np.sin(theta), np.cos(theta)]
    # ])
    # This is for rotating with an angle of negative theta
    rotation_matrix = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    rotated_points = points.dot(rotation_matrix.T)
    return rotated_points


def get_rotated_points(point_name, theta, reference_subtraced_coordinates_dict):
    # mean_center_coord = np.stack([reference_subtraced_coordinates_dict[f'center.x'], reference_subtraced_coordinates_dict[f'center.y']], axis=1).mean(axis=0)
    temp_points = np.stack(
        [
            reference_subtraced_coordinates_dict[f"{point_name}.x"],
            reference_subtraced_coordinates_dict[f"{point_name}.y"],
        ],
        axis=1,
    )
    temp_mean_center_coord = temp_points.mean(axis=0)
    centered_points = temp_points.copy()
    centered_points[:, 0] = centered_points[:, 0] - temp_mean_center_coord[0]
    centered_points[:, 1] = centered_points[:, 1] - temp_mean_center_coord[1]
    rotated_points = rotate_points(centered_points, theta)
    rotated_points[:, 0] = rotated_points[:, 0] + temp_mean_center_coord[0]
    rotated_points[:, 1] = rotated_points[:, 1] + temp_mean_center_coord[1]
    return rotated_points


def find_horizontal_axis_angle(df, point1="left", point2="center", min_valid_points=50):
    """
    Fits a line between original (unreferenced) reference points to determine horizontal axis angle.
    Filters out NaN values before fitting.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing coordinate data
    point1 : str
        First point name (default: 'left')
    point2 : str
        Second point name (default: 'center')
    min_valid_points : int
        Minimum number of valid (non-NaN) point pairs required for fitting (default: 50)

    Returns:
    --------
    float : Angle of the fitted line in radians

    Raises:
    -------
    ValueError : If insufficient valid data points are available
    """
    # Extract x and y coordinates for both points
    x1 = df[f"{point1}.x"].to_numpy()
    y1 = df[f"{point1}.y"].to_numpy()
    x2 = df[f"{point2}.x"].to_numpy()
    y2 = df[f"{point2}.y"].to_numpy()

    # Concatenate coordinates from both points
    x_all = np.hstack([x1, x2])
    y_all = np.hstack([y1, y2])

    # Filter out NaN values - keep only points where both x and y are valid
    valid_mask = ~(np.isnan(x_all) | np.isnan(y_all))
    x_valid = x_all[valid_mask]
    y_valid = y_all[valid_mask]

    # Check if we have enough valid points
    n_valid = len(x_valid)
    if n_valid < min_valid_points:
        total_points = len(x_all)
        n_invalid = total_points - n_valid
        raise ValueError(
            f"Insufficient valid data points for line fitting. "
            f"Found {n_valid} valid point pairs (required: {min_valid_points}), "
            f"excluded {n_invalid} invalid pairs (NaNs). "
            f"This may be due to too many blink periods or missing tracking data. "
            f"Consider reducing blink detection sensitivity or checking data quality."
        )

    # Fit line to valid data points
    line_fn = np.polyfit(x_valid, y_valid, 1)
    line_fn = np.poly1d(line_fn)
    theta = np.arctan(line_fn[1])
    return theta


def moving_average_smoothing(X, k):
    S = np.zeros(X.shape[0])
    for t in range(X.shape[0]):
        if t < k:
            S[t] = np.mean(X[: t + 1])
        else:
            S[t] = np.sum(X[t - k : t]) / k
    return S


def median_filter_smoothing(X, k):
    return median_filter(X, size=k)


def find_sequential_groups(arr):
    groups = []
    current_group = [arr[0]]

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            current_group.append(arr[i])
        else:
            groups.append(current_group)
            current_group = [arr[i]]
    groups.append(current_group)

    return groups


def detect_saccades_per_point_per_direction(rotated_points):
    # Expects rotated_points to be a 1D array representing coordinate time series of one point in one direction (either X or Y)

    displacement_time_series = np.diff(rotated_points)  # pixels per frame

    threshold = (
        displacement_time_series.mean() + displacement_time_series.std() * 3
    )  # chosen threshold

    detected_peaks_inds = np.where(np.abs(displacement_time_series) > threshold)[0]

    # Collecting max value deteceted saccades
    # into a nested list = [[saccade_0_index, saccade_0_velocity_amplitude], [saccade_1_index, saccade_1_velocity_amplitude], ...]
    detected_max_saccades = []

    for group in find_sequential_groups(detected_peaks_inds):
        max_amplitude_relative_ind = np.abs(displacement_time_series[group]).argmax()
        max_amplitude_ind = group[max_amplitude_relative_ind]
        max_amplitude_value = displacement_time_series[max_amplitude_ind]
        detected_max_saccades.append([max_amplitude_ind, max_amplitude_value])

    detected_max_saccades = np.array(detected_max_saccades)

    return detected_max_saccades


def get_all_detected_saccades(path):
    active_points = ["center", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
    df = load_df(path)
    reference_subtraced_coordinates_dict = get_referenced_recalculated_coordinates(df)
    theta = find_horizontal_axis_angle(df)

    all_detected_saccades = {point: {"X": [], "Y": []} for point in active_points}
    for point in active_points:
        rotated_points = get_rotated_points(
            point, theta, reference_subtraced_coordinates_dict
        )
        all_detected_saccades[point]["X"] = detect_saccades_per_point_per_direction(
            rotated_points[:, 0]
        )
        all_detected_saccades[point]["Y"] = detect_saccades_per_point_per_direction(
            rotated_points[:, 1]
        )

    return all_detected_saccades


def get_eight_points_at_time(data_dict, point_name_list, t):
    points_coord_data = []
    for point in point_name_list:
        points_coord_data.append(data_dict[point][t, :])
    return np.stack(points_coord_data, axis=0)


def _get_all_points_pre_extracted(data_dict, point_name_list):
    """
    Pre-extract all points for all frames into a 3D array (n_frames, n_points, 2).
    This is much faster than extracting frame-by-frame in the loop.
    """
    n_frames = data_dict[point_name_list[0]].shape[0]
    n_points = len(point_name_list)

    # Pre-allocate array
    points_array = np.zeros((n_frames, n_points, 2))

    # Extract all points at once
    for i, point in enumerate(point_name_list):
        points_array[:, i, :] = data_dict[point]

    return points_array


def _fit_ellipse_single_frame(points, min_points=5):
    """
    Worker function to fit an ellipse to a single frame's points.
    Filters out NaN values before fitting.
    This function must be at module level to be pickled for multiprocessing.

    Args:
        points: numpy array of shape (n_points, 2) containing the coordinates
        min_points: minimum number of valid points required for fitting (default: 5)

    Returns:
        tuple: (params, center) where params is [width, height, phi] and center is [x, y]
    """
    # Filter out NaN values - keep only points where both x and y are valid
    valid_mask = ~(np.isnan(points[:, 0]) | np.isnan(points[:, 1]))
    valid_points = points[valid_mask]

    # Check if we have enough valid points (ellipse fitting needs at least 5 points)
    if len(valid_points) < min_points:
        return [np.nan, np.nan, np.nan], [np.nan, np.nan]

    try:
        reg = LsqEllipse()
        reg.fit(valid_points)
        center, width, height, phi = reg.as_parameters()
        return [width, height, phi], center
    except Exception as e:
        # Return NaN values if fitting fails
        return [np.nan, np.nan, np.nan], [np.nan, np.nan]


def _fit_ellipses_chunk(chunk_data, min_points=5):
    """
    Worker function to fit ellipses to a chunk of frames.
    Filters out NaN values before fitting.
    Reduces pickling overhead by processing multiple frames per worker.

    Args:
        chunk_data: numpy array of shape (n_chunk_frames, n_points, 2)
        min_points: minimum number of valid points required for fitting (default: 5)

    Returns:
        list of results, each result is (params, center)
    """
    results = []
    for t in range(chunk_data.shape[0]):
        points = chunk_data[t, :, :]

        # Filter out NaN values - keep only points where both x and y are valid
        valid_mask = ~(np.isnan(points[:, 0]) | np.isnan(points[:, 1]))
        valid_points = points[valid_mask]

        # Check if we have enough valid points (ellipse fitting needs at least 5 points)
        if len(valid_points) < min_points:
            results.append(([np.nan, np.nan, np.nan], [np.nan, np.nan]))
            continue

        try:
            reg = LsqEllipse()
            reg.fit(valid_points)
            center, width, height, phi = reg.as_parameters()
            results.append(([width, height, phi], center))
        except Exception as e:
            # Return NaN values if fitting fails
            results.append(([np.nan, np.nan, np.nan], [np.nan, np.nan]))
    return results


def get_fitted_ellipse_parameters(
    coordinates_dict, columns_of_interest, use_parallel=False, n_workers=None
):
    # Collecting parameters of the fitted ellipse into an array over the whole recording
    # ellipse_parameters_data contents = (width, height, phi)
    # ellipse_center_points_data = (center_x, center_y)

    # Pre-extract all point data to avoid repeated dictionary lookups and list operations
    all_points = _get_all_points_pre_extracted(coordinates_dict, columns_of_interest)

    n_frames = all_points.shape[0]

    # Determine number of workers
    if n_workers is None:
        n_workers = mp.cpu_count()

    # Use parallel processing for larger datasets
    # Note: Parallel processing uses fork on Linux/macOS (works well here)
    # Falls back to sequential for small datasets or single worker
    if use_parallel and n_frames > 100 and n_workers > 1:
        print(
            f"ℹ️ Fitting ellipses to {n_frames} frames using {n_workers} parallel workers..."
        )

        # Chunk frame indices to reduce communication overhead
        # Use smaller number of chunks to minimize per-chunk overhead
        n_chunks = n_workers  # 1 chunk per worker for minimal overhead
        chunk_size = max(1, n_frames // n_chunks)
        chunks = []
        for i in range(0, n_frames, chunk_size):
            end_idx = min(i + chunk_size, n_frames)
            # Send only the data needed for this chunk (views reduce memory usage)
            chunk_data = all_points[i:end_idx, :, :]
            chunks.append(chunk_data)

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            chunk_results = list(executor.map(_fit_ellipses_chunk, chunks))

        # Flatten results from chunks
        ellipse_parameters_data = []
        ellipse_center_points_data = []
        for chunk_result in chunk_results:
            for params, center in chunk_result:
                ellipse_parameters_data.append(params)
                ellipse_center_points_data.append(center)

        ellipse_parameters_data = np.array(ellipse_parameters_data)
        ellipse_center_points_data = np.array(ellipse_center_points_data)
    else:
        # Sequential processing (original algorithm)
        if use_parallel:
            print(
                f"ℹ️ Fitting ellipses to {n_frames} frames sequentially (n_workers={n_workers})..."
            )

        ellipse_parameters_data = []
        ellipse_center_points_data = []

        # Create reg object once to avoid repeated instantiation
        reg = LsqEllipse()
        min_points = 5  # Minimum points required for ellipse fitting

        for t in range(n_frames):
            # Direct array indexing instead of function call
            points = all_points[t, :, :]

            # Filter out NaN values - keep only points where both x and y are valid
            valid_mask = ~(np.isnan(points[:, 0]) | np.isnan(points[:, 1]))
            valid_points = points[valid_mask]

            # Check if we have enough valid points
            if len(valid_points) < min_points:
                ellipse_parameters_data.append([np.nan, np.nan, np.nan])
                ellipse_center_points_data.append([np.nan, np.nan])
                continue

            try:
                reg.fit(valid_points)
                center, width, height, phi = reg.as_parameters()
                ellipse_parameters_data.append([width, height, phi])
                ellipse_center_points_data.append(center)
            except Exception as e:
                # Return NaN values if fitting fails
                ellipse_parameters_data.append([np.nan, np.nan, np.nan])
                ellipse_center_points_data.append([np.nan, np.nan])

        ellipse_parameters_data = np.array(ellipse_parameters_data)
        ellipse_center_points_data = np.array(ellipse_center_points_data)

    return ellipse_parameters_data, ellipse_center_points_data


def get_coordinates_dict(df, columns_of_interest):
    return {key: df[key].to_numpy() for key in columns_of_interest}


def get_left_right_center_point(coordinates_dict):
    """
    Calculate the center point between left and right eye reference points.
    Filters out NaN values before computing the mean.

    Parameters:
    -----------
    coordinates_dict : dict
        Dictionary containing coordinate arrays with keys 'left.x', 'left.y', 'right.x', 'right.y'

    Returns:
    --------
    tuple : (x, y) coordinates of the center point

    Raises:
    -------
    ValueError : If insufficient valid data points are available
    """
    x_all = np.hstack([coordinates_dict["left.x"], coordinates_dict["right.x"]])
    y_all = np.hstack([coordinates_dict["left.y"], coordinates_dict["right.y"]])

    # Filter out NaN values
    x_valid = x_all[~np.isnan(x_all)]
    y_valid = y_all[~np.isnan(y_all)]

    # Check if we have enough valid points
    min_valid_points = 20  # Require at least 20 valid points
    if len(x_valid) < min_valid_points or len(y_valid) < min_valid_points:
        raise ValueError(
            f"Insufficient valid data points for center point calculation. "
            f"Found {len(x_valid)} valid x-coordinates and {len(y_valid)} valid y-coordinates "
            f"(required: {min_valid_points} each). "
            f"This may be due to too many blink periods or missing tracking data."
        )

    x = np.mean(x_valid)
    y = np.mean(y_valid)
    return (x, y)


def get_reformatted_coordinates_dict(coordinates_dict, columns_of_interest):
    # Combining separated x and y number arrays into (samples, 2)-shaped array and subtracting the inferred center point from above
    return {
        p: np.stack([coordinates_dict[f"{p}.x"], coordinates_dict[f"{p}.y"]], axis=1)
        for p in columns_of_interest
    }


def get_centered_coordinates_dict(coordinates_dict, center_point):
    return {point: arr - center_point for point, arr in coordinates_dict.items()}


def get_rotated_coordinates_dict(coordinates_dict, theta):
    return {point: rotate_points(arr, theta) for point, arr in coordinates_dict.items()}


def create_flipped_videos(path, what_to_flip="VideoData1"):
    if len([x for x in os.listdir(path / what_to_flip) if ".flipped." in x]) != 0:
        print(f"Flipped videos already exist in {path / what_to_flip}. Exiting.")
    else:
        avis = [x for x in os.listdir(path / what_to_flip) if x[-4:] == ".avi"]
        for avi in avis:
            horizontal_flip_script.horizontal_flip_avi(
                path / what_to_flip / avi,
                path / what_to_flip / f"{avi[:-4]}.flipped.avi",
            )


def detect_saccades_with_threshold(eye_data_stream, threshold_std_times=1):
    harp_time_inds, absolute_positions = eye_data_stream.index, eye_data_stream.values

    framerate = 60
    print(f"Assuming camera frame rate of {framerate} Hz")

    derivative_of_position = np.diff(absolute_positions) * framerate

    threshold = (
        derivative_of_position.mean()
        + derivative_of_position.std() * threshold_std_times
    )

    detected_peaks_inds = np.where(np.abs(derivative_of_position) > threshold)[0]

    # Correcting for over-counted peaks - collecting max value points within groups crossing the threshold
    # detected_max_saccades is a nested list = [[saccade_0_index, saccade_0_velocity_amplitude], [saccade_1_index, saccade_1_velocity_amplitude], ...]
    detected_max_saccades = []

    for group in find_sequential_groups(detected_peaks_inds):
        max_amplitude_relative_ind = np.abs(derivative_of_position[group]).argmax()
        max_amplitude_ind = group[max_amplitude_relative_ind]
        max_amplitude_value = derivative_of_position[max_amplitude_ind]
        detected_max_saccades.append([max_amplitude_ind, max_amplitude_value])

    detected_max_saccades = np.array(detected_max_saccades)
    detected_max_saccades_inds = detected_max_saccades[:, 0].astype(int)

    total_number = detected_max_saccades.shape[0]

    print(
        f"Found {total_number} saccades with chosen threshold of {threshold} (mean + {threshold_std_times} times the standard deviation)."
    )

    runtime = harp_time_inds[-1] - harp_time_inds[0]

    print(f"Saccade frequency = {(total_number / runtime) * 60} events per minute")

    return harp_time_inds[detected_max_saccades_inds], detected_max_saccades[:, 1]


def calculate_saccade_frequency_within_time_range():
    pass


def min_max_dict(coordinates_dict, columns_of_interest):
    """
    Calculate min and max values for X and Y coordinates from a coordinates dictionary.

    Parameters:
    -----------
    coordinates_dict : dict
        Dictionary containing coordinate arrays with keys like 'left.x', 'center.y', etc.
    columns_of_interest : list
        List of point names (without .x/.y suffix) to consider

    Returns:
    --------
    tuple : (x_min, x_max, y_min, y_max)
    """
    x_min = min(
        [
            coordinates_dict[f"{col}.x"][~np.isnan(coordinates_dict[f"{col}.x"])].min()
            for col in columns_of_interest
        ]
    )
    x_max = max(
        [
            coordinates_dict[f"{col}.x"][~np.isnan(coordinates_dict[f"{col}.x"])].max()
            for col in columns_of_interest
        ]
    )
    y_min = min(
        [
            coordinates_dict[f"{col}.y"][~np.isnan(coordinates_dict[f"{col}.y"])].min()
            for col in columns_of_interest
        ]
    )
    y_max = max(
        [
            coordinates_dict[f"{col}.y"][~np.isnan(coordinates_dict[f"{col}.y"])].max()
            for col in columns_of_interest
        ]
    )
    return x_min, x_max, y_min, y_max


def find_longest_lowscore_sections(scores, threshold, top_n=5):
    """
    Finds the top_n longest consecutive segments where scores < threshold.

    Parameters:
    -----------
    scores : pd.Series
        Series of score values
    threshold : float
        Threshold value below which scores are considered low
    top_n : int
        Number of top sections to return (default: 5)

    Returns:
    --------
    list : List of dicts with keys: {'start_idx', 'end_idx', 'length'}
    """
    is_low = scores < threshold
    diff = is_low.astype(int).diff().fillna(0)
    starts = np.where((diff == 1))[0]
    ends = np.where((diff == -1))[0]

    if is_low.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if is_low.iloc[-1]:
        ends = np.append(ends, len(is_low))

    sections = []
    for s, e in zip(starts, ends):
        sections.append({"start_idx": s, "end_idx": e - 1, "length": e - s})

    top_sections = sorted(sections, key=lambda x: x["length"], reverse=True)[:top_n]
    return top_sections


def find_percentile_for_consecutive_limit(
    scores, max_consecutive, find_longest_lowscore_sections_fn=None
):
    """
    Binary search for the lowest percentile so that the maximum number of
    consecutively excluded frames (scores below the percentile) does not exceed max_consecutive.

    Parameters:
    -----------
    scores : pd.Series
        Series of score values
    max_consecutive : int
        Maximum allowed consecutive frames below threshold
    find_longest_lowscore_sections_fn : callable, optional
        Function to find longest low score sections (default: uses find_longest_lowscore_sections)

    Returns:
    --------
    tuple : (percentile, threshold_value)
    """
    if find_longest_lowscore_sections_fn is None:
        find_longest_lowscore_sections_fn = find_longest_lowscore_sections

    sorted_scores = scores.dropna().sort_values()
    n = len(sorted_scores)
    low, high = 0.0, 0.2  # Search up to the 20th percentile; adjust as needed
    tol = 0.0001

    found_pct = None
    found_value = None

    while (high - low) > tol:
        mid = (low + high) / 2
        thr = sorted_scores.quantile(mid)
        sections = find_longest_lowscore_sections_fn(scores, thr, top_n=1)
        longest = sections[0]["length"] if sections else 0
        if longest > max_consecutive:
            high = mid
        else:
            found_pct = mid
            found_value = thr
            low = mid

    return found_pct, found_value


def find_blink_segments(instance_scores, threshold, min_frames, max_frames):
    """
    Find consecutive segments where instance.score < threshold.
    Filters by duration and returns valid blink segments.

    Parameters:
    -----------
    instance_scores : pd.Series
        Series of instance.score values
    threshold : float
        Score threshold below which to consider a blink
    min_frames : int
        Minimum number of consecutive frames to consider a blink
    max_frames : int
        Maximum number of consecutive frames to consider a blink

    Returns:
    --------
    list : List of dicts with keys: 'start_idx', 'end_idx', 'length', 'mean_score'
    """
    is_low = instance_scores < threshold
    diff = is_low.astype(int).diff().fillna(0)
    starts = np.where((diff == 1))[0]
    ends = np.where((diff == -1))[0]

    # Handle case where the first element is low
    if is_low.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if is_low.iloc[-1]:
        ends = np.append(ends, len(is_low))

    segments = []
    for s, e in zip(starts, ends):
        length = e - s
        if min_frames <= length <= max_frames:
            mean_score = instance_scores.iloc[s:e].mean()
            segments.append(
                {
                    "start_idx": s,
                    "end_idx": e - 1,  # inclusive end
                    "length": length,
                    "mean_score": mean_score,
                }
            )

    return segments


def format_time_from_start(seconds_from_start):
    """
    Format seconds as MM:SS from the start of recording.

    Parameters:
    -----------
    seconds_from_start : float
        Time in seconds from the beginning of the recording

    Returns:
    --------
    str : Formatted as "MM:SS" or "M:SS" if minutes < 10
    """
    minutes = int(seconds_from_start // 60)
    seconds = int(seconds_from_start % 60)
    return f"{minutes}:{seconds:02d}"


def merge_nearby_blinks(segments, merge_window_frames):
    """
    Merge blink segments that are within merge_window_frames of each other.

    Parameters:
    -----------
    segments : list of dict
        List of blink segment dictionaries with keys: 'start_idx', 'end_idx', 'length', 'mean_score'
        May also contain 'start_frame_idx' and 'end_frame_idx' if frame_idx values were stored
    merge_window_frames : int
        Maximum gap (in frames) between segments to merge them

    Returns:
    --------
    list : Merged segments list
    """
    if not segments:
        return segments

    # Sort by start_idx
    sorted_segments = sorted(segments, key=lambda x: x["start_idx"])
    merged = [sorted_segments[0].copy()]  # Copy to avoid modifying original

    for seg in sorted_segments[1:]:
        last = merged[-1]
        gap = seg["start_idx"] - last["end_idx"] - 1

        if gap <= merge_window_frames:
            # Merge: extend the last segment
            merged[-1]["end_idx"] = seg["end_idx"]
            merged[-1]["length"] = merged[-1]["end_idx"] - merged[-1]["start_idx"] + 1
            merged[-1]["mean_score"] = (
                last["mean_score"] * last["length"] + seg["mean_score"] * seg["length"]
            ) / merged[-1]["length"]
            # Preserve frame_idx values if they exist
            if "start_frame_idx" in seg and "end_frame_idx" in seg:
                merged[-1]["end_frame_idx"] = seg["end_frame_idx"]
                # start_frame_idx stays the same (from first segment)
                if "start_frame_idx" not in merged[-1]:
                    merged[-1]["start_frame_idx"] = last.get(
                        "start_frame_idx", last["start_idx"]
                    )
        else:
            merged.append(seg.copy())  # Copy to avoid modifying original

    return merged


def check_manual_match(auto_start, auto_end, manual_blinks_list):
    """
    Check if an auto-detected blink matches any manual blink annotation.

    Uses >=40% overlap threshold to handle over-merged manual blinks.

    Parameters:
    -----------
    auto_start : int
        Start frame of auto-detected blink
    auto_end : int
        End frame of auto-detected blink
    manual_blinks_list : list of dict or None
        List of manual blink dicts with keys: 'start', 'end', 'num'

    Returns:
    --------
    int : 1 if match found, 0 otherwise
    """
    if manual_blinks_list is None:
        return 0
    for manual in manual_blinks_list:
        overlap_start = max(manual["start"], auto_start)
        overlap_end = min(manual["end"], auto_end)
        if overlap_start <= overlap_end:
            overlap_frames = overlap_end - overlap_start + 1
            manual_length = manual["end"] - manual["start"] + 1
            # Use 40% threshold to match the diagnostic comparison logic
            if overlap_frames >= manual_length * 0.40:
                return 1
    return 0


def compare_manual_vs_auto_blinks(
    blink_segments, video_data, manual_blinks, video_label, debug=False
):
    """
    Compare manual blink annotations with auto-detected blinks.

    Prints detailed comparison including overlap analysis.

    Parameters:
    -----------
    blink_segments : list of dict
        Auto-detected blink segments with keys: 'start_idx', 'end_idx', 'length'
    video_data : pd.DataFrame
        Video data with 'frame_idx' and 'Seconds' columns
    manual_blinks : list of dict or None
        Manual blink annotations with keys: 'start', 'end', 'num'
    video_label : str
        Label for the video (for printing)
    debug : bool
        Whether to print detailed comparison

    Returns:
    --------
    list : List of auto-detected blink dicts with frame numbers
    """
    if not debug or manual_blinks is None:
        return []

    # Get auto-detected blink frame ranges
    auto_blinks = []
    for i, blink in enumerate(blink_segments, 1):
        start_idx = blink["start_idx"]
        end_idx = blink["end_idx"]
        if "frame_idx" in video_data.columns:
            frame_start = int(video_data["frame_idx"].iloc[start_idx])
            frame_end = int(video_data["frame_idx"].iloc[end_idx])
        else:
            frame_start = start_idx
            frame_end = end_idx
        auto_blinks.append(
            {
                "num": i,
                "start": frame_start,
                "end": frame_end,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "length": blink["length"],
            }
        )

    print(f"\n   🔍 MANUAL vs AUTO-DETECTED BLINK COMPARISON:")
    print(f"      Manual blinks: {len(manual_blinks)}")
    print(f"      Auto-detected blinks: {len(auto_blinks)}")

    # Match manual blinks to auto-detected ones (find overlapping frames)
    # Allow matching with multiple auto-detected blinks if manual blink is over-merged
    print(f"\n   Manual → Auto matching (overlap analysis):")
    for manual in manual_blinks:
        manual_length = manual["end"] - manual["start"] + 1
        total_overlap = 0
        matching_auto_blinks = []

        # Find all auto-detected blinks that overlap with this manual blink
        for auto in auto_blinks:
            overlap_start = max(manual["start"], auto["start"])
            overlap_end = min(manual["end"], auto["end"])
            if overlap_start <= overlap_end:
                overlap_frames = overlap_end - overlap_start + 1
                total_overlap += overlap_frames
                matching_auto_blinks.append({"auto": auto, "overlap": overlap_frames})

        # Calculate total overlap percentage
        total_overlap_pct = (
            (total_overlap / manual_length) * 100 if manual_length > 0 else 0
        )

        # Match if total overlap is >= 40% (less stringent to handle over-merged manual blinks)
        if total_overlap >= manual_length * 0.40:  # At least 40% overlap
            if len(matching_auto_blinks) == 1:
                # Single match
                match_info = matching_auto_blinks[0]
                best_match = match_info["auto"]
                overlap = match_info["overlap"]
                start_diff = best_match["start"] - manual["start"]
                end_diff = best_match["end"] - manual["end"]
                match_str = f"✅ MATCH: Auto blink {best_match['num']} (frames {best_match['start']}-{best_match['end']})"
                match_str += f", {overlap} frames overlap ({total_overlap_pct:.1f}%)"
                if start_diff != 0 or end_diff != 0:
                    match_str += f", offset: start={start_diff:+d}, end={end_diff:+d}"
                print(
                    f"      Manual {manual['num']}: {manual['start']}-{manual['end']} → {match_str}"
                )
            else:
                # Multiple matches (manual blink is over-merged)
                auto_nums = [m["auto"]["num"] for m in matching_auto_blinks]
                auto_ranges = [
                    f"{m['auto']['start']}-{m['auto']['end']}"
                    for m in matching_auto_blinks
                ]
                match_str = f"✅ MATCH: Auto blinks {auto_nums} (frames: {', '.join(auto_ranges)})"
                match_str += (
                    f", total {total_overlap} frames overlap ({total_overlap_pct:.1f}%)"
                )
                print(
                    f"      Manual {manual['num']}: {manual['start']}-{manual['end']} → {match_str}"
                )
        else:
            print(
                f"      Manual {manual['num']}: {manual['start']}-{manual['end']} → ❌ NO MATCH FOUND"
            )

    # Also check which auto-detected blinks don't have manual matches
    print(f"\n   Auto-detected blinks without manual matches:")
    unmatched_auto = []
    for auto in auto_blinks:
        has_match = False
        for manual in manual_blinks:
            overlap_start = max(manual["start"], auto["start"])
            overlap_end = min(manual["end"], auto["end"])
            if overlap_start <= overlap_end:
                overlap_frames = overlap_end - overlap_start + 1
                manual_length = manual["end"] - manual["start"] + 1
                # Use same threshold as matching logic (40% of manual blink)
                if overlap_frames >= manual_length * 0.40:
                    has_match = True
                    break
        if not has_match:
            unmatched_auto.append(auto)

    if len(unmatched_auto) > 0:
        for auto in unmatched_auto:
            print(
                f"      Auto {auto['num']}: frames {auto['start']}-{auto['end']}, length={auto['length']} frames"
            )
    else:
        print(f"      (all auto-detected blinks have manual matches)")

    return auto_blinks


def detect_blinks_for_video(
    video_data,
    columns_of_interest,
    blink_instance_score_threshold,
    long_blink_warning_ms,
    min_frames_threshold=4,
    merge_window_frames=10,
    fps=None,
    video_label="",
    manual_blinks=None,
    debug=False,
):
    """
    Complete blink detection pipeline for a single video.

    Detects blinks, filters by duration, merges nearby blinks, compares with manual annotations,
    interpolates short blinks, and marks long blinks as NaN.

    Parameters:
    -----------
    video_data : pd.DataFrame
        Video data with 'instance.score', 'Seconds', 'frame_idx' columns
    columns_of_interest : list of str
        List of coordinate column names to set to NaN during blinks
    blink_instance_score_threshold : float
        Hard threshold for blink detection (frames with instance.score below this are blinks)
    long_blink_warning_ms : float
        Warn if blinks exceed this duration (in ms)
    min_frames_threshold : int
        Minimum frames for a blink to be kept (shorter ones are interpolated)
    merge_window_frames : int
        Merge blinks within this many frames into blink bouts
    fps : float, optional
        Frames per second (calculated from video_data if not provided)
    video_label : str
        Label for the video (for printing)
    manual_blinks : list of dict or None
        Manual blink annotations for comparison (keys: 'start', 'end', 'num')
    debug : bool
        Whether to print detailed information

    Returns:
    --------
    dict : Dictionary with keys:
        - 'blink_segments': List of long blink segments (>= min_frames_threshold)
        - 'short_blink_segments': List of short blink segments (< min_frames_threshold)
        - 'blink_bouts': List of merged blink bouts
        - 'all_blink_segments': List of all detected blink segments
        - 'fps': Calculated or provided FPS
        - 'long_blinks_warnings': List of warnings for long blinks
        - 'total_blink_frames': Total frames marked as blinks
        - 'blink_bout_rate': Blinks per minute
    """
    if debug:
        print(f"\n{video_label} - Blink Detection")

    # Calculate FPS if not provided
    if fps is None:
        fps = 1 / video_data["Seconds"].diff().mean()

    long_blink_warning_frames = int(long_blink_warning_ms / 1000 * fps)

    if debug:
        print(f"  FPS: {fps:.2f}")
        print(
            f"  Long blink warning threshold: {long_blink_warning_frames} frames ({long_blink_warning_ms}ms)"
        )

    # Use hard threshold from user parameters
    blink_threshold = blink_instance_score_threshold
    if debug:
        print(f"  Using hard threshold: {blink_threshold:.4f}")

    # Find all blink segments - use very lenient min_frames (1) to capture all segments
    # No filtering by frame count - short blinks are OK to interpolate
    # No merging - we want to preserve good data between separate blinks
    all_blink_segments = find_blink_segments(
        video_data["instance.score"],
        blink_threshold,
        min_frames=1,  # Very lenient to capture all segments
        max_frames=999999,  # Very high limit - essentially no maximum
    )

    # Always print key blink detection stats
    print(f"{video_label} - Found {len(all_blink_segments)} blink segments")

    # CRITICAL FIX: Store frame_idx values immediately while DataFrame is still valid
    # This prevents issues if the DataFrame is modified later (e.g., by merge operations)
    frame_idx_available = "frame_idx" in video_data.columns

    # CRITICAL FIX: Validate frame_idx before storing blink values
    if frame_idx_available:
        # Check for duplicates
        if video_data["frame_idx"].duplicated().any():
            dup_count = video_data["frame_idx"].duplicated().sum()
            print(
                f"   ⚠️ WARNING: {video_label} has {dup_count} duplicate frame_idx values before blink detection!"
            )
            print(
                f"      This may cause incorrect blink frame numbers. Frame_idx should be unique."
            )
        # Check for monotonicity
        if not video_data["frame_idx"].is_monotonic_increasing:
            print(
                f"   ⚠️ WARNING: {video_label} frame_idx is not monotonic before blink detection!"
            )
            print(
                f"      This may cause incorrect blink frame numbers. Frame_idx should be strictly increasing."
            )

    for blink in all_blink_segments:
        start_idx = blink["start_idx"]
        end_idx = blink["end_idx"]
        if frame_idx_available:
            start_frame_val = video_data["frame_idx"].iloc[start_idx]
            end_frame_val = video_data["frame_idx"].iloc[end_idx]
            # Validate that stored values are valid
            if pd.notna(start_frame_val) and pd.notna(end_frame_val):
                blink["start_frame_idx"] = int(start_frame_val)
                blink["end_frame_idx"] = int(end_frame_val)
                # Additional validation: ensure start <= end
                if blink["start_frame_idx"] > blink["end_frame_idx"]:
                    print(
                        f"   ⚠️ WARNING: Blink segment has start_frame_idx ({blink['start_frame_idx']}) > end_frame_idx ({blink['end_frame_idx']})!"
                    )
                    print(
                        f"      This should not happen. Using positional indices instead."
                    )
                    blink["start_frame_idx"] = start_idx
                    blink["end_frame_idx"] = end_idx
            else:
                # If NaN, use positional indices
                blink["start_frame_idx"] = start_idx
                blink["end_frame_idx"] = end_idx
        else:
            blink["start_frame_idx"] = start_idx
            blink["end_frame_idx"] = end_idx

    # Filter out blinks shorter than min_frames_threshold frames
    blink_segments = [
        blink for blink in all_blink_segments if blink["length"] >= min_frames_threshold
    ]
    short_blink_segments = [
        blink for blink in all_blink_segments if blink["length"] < min_frames_threshold
    ]

    # Always print filtering stats
    print(
        f"  After filtering <{min_frames_threshold} frames: {len(blink_segments)} blink segment(s), {len(short_blink_segments)} short segment(s) will be interpolated"
    )

    # Merge blinks within merge_window_frames into blink bouts
    blink_bouts = merge_nearby_blinks(blink_segments, merge_window_frames)

    # Always print bout count
    print(
        f"  After merging blinks within {merge_window_frames} frames: {len(blink_bouts)} blink bout(s)"
    )

    # Check for long blinks and warn if needed
    long_blinks_warnings = []
    for i, blink in enumerate(blink_segments, 1):
        start_idx = blink["start_idx"]
        end_idx = blink["end_idx"]
        start_time = video_data["Seconds"].iloc[start_idx]
        end_time = video_data["Seconds"].iloc[end_idx]
        duration_ms = (end_time - start_time) * 1000

        # Warn about very long blinks (may need manual verification)
        if duration_ms > long_blink_warning_ms:
            if "frame_idx" in video_data.columns:
                frame_start = int(video_data["frame_idx"].iloc[start_idx])
                frame_end = int(video_data["frame_idx"].iloc[end_idx])
            else:
                frame_start = start_idx
                frame_end = end_idx
            long_blinks_warnings.append(
                {
                    "blink_num": i,
                    "frames": f"{frame_start}-{frame_end}",
                    "duration_ms": duration_ms,
                }
            )

    # Print warnings for long blinks
    if len(long_blinks_warnings) > 0:
        print(
            f"\n   ⚠️ WARNING: Found {len(long_blinks_warnings)} blink(s) longer than {long_blink_warning_ms}ms:"
        )
        for warn in long_blinks_warnings:
            print(
                f"      Blink {warn['blink_num']}: frames {warn['frames']}, duration {warn['duration_ms']:.1f}ms - Please verify this is a real blink in the video"
            )

    if debug:
        print(f"\n  Detailed blink detection information:")
        print(f"  FPS: {fps:.2f}")
        print(
            f"  Long blink warning threshold: {long_blink_warning_frames} frames ({long_blink_warning_ms}ms)"
        )
        print(f"  Using hard threshold: {blink_threshold:.4f}")
        print(f"  Detected {len(blink_segments)} blink segment(s)\n")

    # Print all detected blinks once (detailed) - only in debug mode
    if debug and len(blink_segments) > 0:
        print(f"  Detected blinks:")
        for i, blink in enumerate(blink_segments, 1):
            start_idx = blink["start_idx"]
            end_idx = blink["end_idx"]

            # Calculate time range
            start_time = video_data["Seconds"].iloc[start_idx]
            end_time = video_data["Seconds"].iloc[end_idx]
            duration_ms = (end_time - start_time) * 1000

            # Get actual frame numbers from frame_idx column
            if "frame_idx" in video_data.columns:
                actual_start_frame = int(video_data["frame_idx"].iloc[start_idx])
                actual_end_frame = int(video_data["frame_idx"].iloc[end_idx])
                frame_info = f"frames {actual_start_frame}-{actual_end_frame}"
            else:
                actual_start_frame = start_idx
                actual_end_frame = end_idx
                frame_info = f"frames {actual_start_frame}-{actual_end_frame}"

            print(
                f"    Blink {i}: {frame_info}, {blink['length']} frames, {duration_ms:.1f}ms, mean score: {blink['mean_score']:.4f}"
            )

        # DIAGNOSTIC: Direct comparison of manual vs auto-detected blinks (detailed)
        if manual_blinks is not None:
            compare_manual_vs_auto_blinks(
                blink_segments, video_data, manual_blinks, video_label, debug=True
            )
        else:
            print(
                f"\n   ⚠️ WARNING: Manual blink comparison skipped - manual_blinks not provided"
            )
    elif debug:
        print("  No blinks detected")

    # Always interpolate short blinks and mark long blinks (regardless of debug mode)
    total_blink_frames = 0
    short_blink_frames = 0

    if len(blink_segments) > 0 or len(short_blink_segments) > 0:
        # Interpolate short blinks first (if any)
        if len(short_blink_segments) > 0:
            if debug:
                print(
                    f"\n  Interpolating short blinks (< {min_frames_threshold} frames):"
                )
            for blink in short_blink_segments:
                start_idx = blink["start_idx"]
                end_idx = blink["end_idx"]
                video_data.loc[
                    video_data.index[start_idx : end_idx + 1], columns_of_interest
                ] = np.nan
                short_blink_frames += blink["length"]

            # Interpolate all NaNs (this fills short blinks)
            video_data[columns_of_interest] = video_data[
                columns_of_interest
            ].interpolate(method="linear", limit_direction="both")

            if debug:
                print(
                    f"    Interpolated {short_blink_frames} frames from {len(short_blink_segments)} short blink segment(s)"
                )
        elif debug:
            print(f"\n  No short blinks to interpolate")

        # Mark long blinks by setting coordinates to NaN (these remain as NaN, not interpolated)
        for blink in blink_segments:
            start_idx = blink["start_idx"]
            end_idx = blink["end_idx"]
            video_data.loc[
                video_data.index[start_idx : end_idx + 1], columns_of_interest
            ] = np.nan
            total_blink_frames += blink["length"]

        if debug:
            print(
                f"  Total long blink frames marked (kept as NaN): {total_blink_frames} frames "
                f"({total_blink_frames / fps * 1000:.1f}ms)"
            )

    # Calculate blink bout rate
    if len(video_data) > 0:
        recording_duration_min = (
            video_data["Seconds"].iloc[-1] - video_data["Seconds"].iloc[0]
        ) / 60
        blink_bout_rate = (
            len(blink_bouts) / recording_duration_min
            if recording_duration_min > 0
            else 0
        )
        if debug and len(blink_segments) > 0:
            print(f"  Blink bout rate: {blink_bout_rate:.2f} blink bouts/minute")
    else:
        blink_bout_rate = 0

    return {
        "blink_segments": blink_segments,
        "short_blink_segments": short_blink_segments,
        "blink_bouts": blink_bouts,
        "all_blink_segments": all_blink_segments,
        "fps": fps,
        "long_blinks_warnings": long_blinks_warnings,
        "total_blink_frames": total_blink_frames,
        "blink_bout_rate": blink_bout_rate,
    }


def load_manual_blinks(data_path, video_number):
    """
    Load manual blink annotations from CSV file.

    Parameters:
    -----------
    data_path : Path
        Path to directory containing the CSV file
    video_number : int
        Video number (1 or 2) to determine filename (Video1_manual_blinks.csv or Video2_manual_blinks.csv)

    Returns:
    --------
    list of dict or None : List of blink dicts with keys 'num', 'start', 'end', or None if file doesn't exist or has errors
    """
    manual_blinks_path = data_path / f"Video{video_number}_manual_blinks.csv"

    if not manual_blinks_path.exists():
        return None

    try:
        manual_blinks_df = pd.read_csv(manual_blinks_path)
        # Expected columns: blink_number, start_frame, end_frame
        if all(
            col in manual_blinks_df.columns
            for col in ["blink_number", "start_frame", "end_frame"]
        ):
            manual_blinks = [
                {
                    "num": int(row["blink_number"]),
                    "start": int(row["start_frame"]),
                    "end": int(row["end_frame"]),
                }
                for _, row in manual_blinks_df.iterrows()
            ]
            print(
                f"✅ Loaded {len(manual_blinks)} manual blinks for VideoData{video_number} from {manual_blinks_path.name}"
            )
            return manual_blinks
        else:
            print(
                f"⚠️ WARNING: {manual_blinks_path.name} exists but doesn't have expected columns (blink_number, start_frame, end_frame)"
            )
            return None
    except Exception as e:
        print(f"⚠️ WARNING: Failed to load {manual_blinks_path.name}: {e}")
        return None


def analyze_confidence_scores(
    video_data, score_columns, score_cutoff, video_label="", debug=False
):
    """
    Analyze confidence scores and report top 3 columns with most low-confidence frames.

    Parameters:
    -----------
    video_data : pd.DataFrame
        Video data with score columns
    score_columns : list of str
        List of score column names to analyze
    score_cutoff : float
        Threshold below which scores are considered low
    video_label : str
        Label for the video (for printing)
    debug : bool
        Whether to print analysis (only runs if debug=True)

    Returns:
    --------
    list : List of tuples (col, count, pct, longest) sorted by count
    """
    if not debug:
        return []

    total_points = len(video_data)
    print(
        f"\nℹ️ {video_label} - Top 3 columns with most frames below {score_cutoff} confidence score:"
    )

    stats = []
    for col in score_columns:
        if col in video_data.columns:
            count_below = (video_data[col] < score_cutoff).sum()
            pct_below = (count_below / total_points) * 100 if total_points > 0 else 0

            below_mask = video_data[col] < score_cutoff
            longest = 0
            run = 0
            for val in below_mask:
                if val:
                    run += 1
                    if run > longest:
                        longest = run
                else:
                    run = 0
            stats.append((col, count_below, pct_below, longest))

    stats.sort(key=lambda x: x[1], reverse=True)
    for i, (col, count, pct, longest) in enumerate(stats[:3]):
        print(
            f"{video_label} - #{i + 1}: {col} | Values below {score_cutoff}: {count} ({pct:.2f}%) | Longest consecutive frame series: {longest}"
        )

    return stats


def center_coordinates_to_median(video_data, columns_of_interest, video_label=""):
    """
    Center coordinates to the median pupil centre.

    Modifies video_data in place by subtracting median center.x and center.y from all coordinates.

    Parameters:
    -----------
    video_data : pd.DataFrame
        Video data with coordinate columns
    columns_of_interest : list of str
        List of coordinate column names to center
    video_label : str
        Label for the video (for printing)

    Returns:
    --------
    pd.DataFrame : Copy of video_data after centering
    """
    # Calculate the median of the center x and y points
    mean_center_x = video_data["center.x"].median()
    mean_center_y = video_data["center.y"].median()

    print(
        f"{video_label} - Centering on median pupil centre: \nMean center.x: {mean_center_x}, Mean center.y: {mean_center_y}"
    )

    # Translate the coordinates
    for col in columns_of_interest:
        if ".x" in col:
            video_data[col] = video_data[col] - mean_center_x
        elif ".y" in col:
            video_data[col] = video_data[col] - mean_center_y

    return video_data.copy()


def filter_low_confidence_points(
    video_data, point_names, score_cutoff, video_label="", debug=False
):
    """
    Filter out low-confidence points and replace with NaN.

    Modifies video_data in place by setting x and y coordinates to NaN where score < threshold.

    Parameters:
    -----------
    video_data : pd.DataFrame
        Video data with score and coordinate columns
    point_names : list of str
        List of point names (without .x, .y, .score suffix)
    score_cutoff : float
        Score threshold below which points are replaced with NaN
    video_label : str
        Label for the video (for printing)
    debug : bool
        Whether to print statistics

    Returns:
    --------
    dict : Dictionary with 'total_low_score', 'max_low_score_channel', 'max_low_score_count'
    """
    total_low_score = 0
    low_score_counts = {}

    for point in point_names:
        if f"{point}.score" in video_data.columns:
            # Find indices where score is below threshold
            low_score_mask = video_data[f"{point}.score"] < score_cutoff
            low_score_count = low_score_mask.sum()
            low_score_counts[f"{point}.x"] = low_score_count
            low_score_counts[f"{point}.y"] = low_score_count
            total_low_score += (
                low_score_count * 2
            )  # *2 because we're removing both x and y

            # Set x and y to NaN for low confidence points
            video_data.loc[low_score_mask, f"{point}.x"] = np.nan
            video_data.loc[low_score_mask, f"{point}.y"] = np.nan

    # Find the channel with the maximum number of low-score points
    max_low_score_channel = (
        max(low_score_counts, key=low_score_counts.get) if low_score_counts else None
    )
    max_low_score_count = (
        low_score_counts[max_low_score_channel] if max_low_score_channel else 0
    )

    # Print the channel with the maximum number of low-score points
    if debug:
        if max_low_score_channel:
            print(
                f"{video_label} - Channel with the maximum number of low-confidence points: {max_low_score_channel}, Number of low-confidence points: {max_low_score_count}"
            )
        print(
            f"{video_label} - A total number of {total_low_score} low-confidence coordinate values were replaced by interpolation"
        )

    return {
        "total_low_score": total_low_score,
        "max_low_score_channel": max_low_score_channel,
        "max_low_score_count": max_low_score_count,
    }


def remove_outliers_and_interpolate(
    video_data, columns_of_interest, outlier_sd_threshold, video_label="", debug=False
):
    """
    Remove outliers and interpolate NaN values.

    Outliers are defined as values more than outlier_sd_threshold standard deviations from the mean.
    Modifies video_data in place.

    Parameters:
    -----------
    video_data : pd.DataFrame
        Video data with coordinate columns
    columns_of_interest : list of str
        List of coordinate column names to process
    outlier_sd_threshold : float
        Number of standard deviations for outlier detection
    video_label : str
        Label for the video (for printing)
    debug : bool
        Whether to print statistics

    Returns:
    --------
    dict : Dictionary with 'total_outliers', 'max_outliers_channel', 'max_outliers_count'
    """
    # Calculate the standard deviation for each column of interest
    std_devs = {col: video_data[col].std() for col in columns_of_interest}

    # Calculate the number of outliers for each column
    outliers = {
        col: (
            (video_data[col] - video_data[col].mean()).abs()
            > outlier_sd_threshold * std_devs[col]
        ).sum()
        for col in columns_of_interest
    }

    # Find the channel with the maximum number of outliers
    max_outliers_channel = max(outliers, key=outliers.get) if outliers else None
    max_outliers_count = outliers[max_outliers_channel] if max_outliers_channel else 0
    total_outliers = sum(outliers.values())

    # Print the channel with the maximum number of outliers and the number
    if debug:
        if max_outliers_channel:
            print(
                f"{video_label} - Channel with the maximum number of outliers: {max_outliers_channel}, Number of outliers: {max_outliers_count}"
            )
        print(
            f"{video_label} - A total number of {total_outliers} outliers were replaced by interpolation"
        )

    # Replace outliers by interpolating between the previous and subsequent non-NaN value
    for col in columns_of_interest:
        outlier_indices = video_data[
            (
                (video_data[col] - video_data[col].mean()).abs()
                > outlier_sd_threshold * std_devs[col]
            )
        ].index
        video_data.loc[outlier_indices, col] = np.nan

    # Interpolate all NaN values (returns new DataFrame, so caller needs to reassign)
    video_data_interpolated = video_data.interpolate(
        method="linear", limit_direction="both"
    )

    return {
        "total_outliers": total_outliers,
        "max_outliers_channel": max_outliers_channel,
        "max_outliers_count": max_outliers_count,
        "video_data_interpolated": video_data_interpolated,  # Return interpolated dataframe
    }


def analyze_instance_score_distribution(
    video_data,
    blink_instance_score_threshold,
    fps,
    video_label="",
    debug=False,
    plot=True,
    threshold_label="Applied threshold",
):
    """
    Analyze instance score distribution and plot histogram.

    Parameters:
    -----------
    video_data : pd.DataFrame
        Video data with 'instance.score' column
    blink_instance_score_threshold : float
        Threshold used for blink detection
    fps : float
        Frames per second (for time calculations)
    video_label : str
        Label for the video (for printing)
    debug : bool
        Whether to print detailed statistics
    plot : bool
        Whether to create histogram plot

    Returns:
    --------
    dict : Dictionary with statistics including 'percentile', 'num_low', 'pct_low', 'longest_consecutive', 'low_sections'
    """
    if not debug:
        return {}

    # Plot histogram
    if plot:
        plt.figure(figsize=(6, 5))
        plt.hist(
            video_data["instance.score"].dropna(),
            bins=30,
            color="skyblue",
            edgecolor="black",
        )
        plt.axvline(
            blink_instance_score_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"{threshold_label} = {blink_instance_score_threshold}",
        )
        plt.yscale("log")
        plt.title(f"Distribution of instance.score ({video_label})")
        plt.xlabel("instance.score")
        plt.ylabel("Frequency (log scale)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Calculate statistics
    percentile = (
        (video_data["instance.score"] < blink_instance_score_threshold).sum()
        / len(video_data)
        * 100
    )
    num_low = (video_data["instance.score"] < blink_instance_score_threshold).sum()
    total = len(video_data)
    pct_low = (num_low / total) * 100

    # Find longest consecutive segments
    low_sections = find_longest_lowscore_sections(
        video_data["instance.score"], blink_instance_score_threshold, top_n=1
    )
    longest_consecutive = low_sections[0]["length"] if low_sections else 0
    longest_consecutive_ms = (
        (longest_consecutive / fps) * 1000 if fps and longest_consecutive > 0 else None
    )

    # Always print key stats
    print(f"\n{video_label} - Instance Score Threshold Analysis:")
    print(f"  {threshold_label}: {blink_instance_score_threshold}")
    print(f"  Frames below threshold: {num_low} / {total} ({pct_low:.2f}%)")
    print(f"  Longest consecutive segment: {longest_consecutive} frames", end="")
    if longest_consecutive_ms:
        print(f" ({longest_consecutive_ms:.1f}ms)")
    else:
        print()

    # Detailed stats only in debug mode
    if debug:
        print(f"\n  Detailed statistics:")
        print(
            f"  Percentile: {percentile:.2f}% (i.e., {percentile:.2f}% of frames have instance.score < {blink_instance_score_threshold})"
        )

        # Report the top 5 longest consecutive sections
        low_sections_detailed = find_longest_lowscore_sections(
            video_data["instance.score"], blink_instance_score_threshold, top_n=5
        )
        if len(low_sections_detailed) > 0:
            print(
                f"\n  Top 5 longest consecutive sections where instance.score < threshold:"
            )
            for i, sec in enumerate(low_sections_detailed, 1):
                start_idx = sec["start_idx"]
                end_idx = sec["end_idx"]
                sec_duration_ms = (sec["length"] / fps) * 1000 if fps else None
                if sec_duration_ms:
                    print(
                        f"    Section {i}: index {start_idx}-{end_idx} (length {sec['length']} frames, {sec_duration_ms:.1f}ms)"
                    )
                else:
                    print(
                        f"    Section {i}: index {start_idx}-{end_idx} (length {sec['length']} frames)"
                    )

    return {
        "percentile": percentile,
        "num_low": num_low,
        "pct_low": pct_low,
        "longest_consecutive": longest_consecutive,
        "longest_consecutive_ms": longest_consecutive_ms,
        "low_sections": low_sections,
    }


def plot_instance_score_distributions_combined(
    video_data1,
    video_data2,
    blink_instance_score_threshold,
    has_v1=False,
    has_v2=False,
    threshold_label="Applied threshold",
):
    """
    Plot combined histograms for instance.score distributions (both videos in one figure).

    Parameters:
    -----------
    video_data1 : pd.DataFrame or None
        Video data 1 with 'instance.score' column
    video_data2 : pd.DataFrame or None
        Video data 2 with 'instance.score' column
    blink_instance_score_threshold : float or dict
        Threshold for blink detection. Can be a single float for both plots or
        a dict with per-video values, e.g. {"VideoData1": 3.7, "VideoData2": 3.4}
    has_v1 : bool
        Whether video_data1 exists and should be plotted
    has_v2 : bool
        Whether video_data2 exists and should be plotted
    """
    if not (has_v1 or has_v2):
        return

    plt.figure(figsize=(12, 5))
    plot_index = 1

    if isinstance(blink_instance_score_threshold, dict):
        threshold_v1 = blink_instance_score_threshold.get("VideoData1")
        threshold_v2 = blink_instance_score_threshold.get("VideoData2")
    else:
        threshold_v1 = blink_instance_score_threshold
        threshold_v2 = blink_instance_score_threshold

    if has_v1:
        plt.subplot(1, 2 if has_v2 else 1, plot_index)
        plt.hist(
            video_data1["instance.score"].dropna(),
            bins=30,
            color="skyblue",
            edgecolor="black",
        )
        if threshold_v1 is not None:
            plt.axvline(
                threshold_v1,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"{threshold_label} = {threshold_v1}",
            )
        plt.yscale("log")
        plt.title("Distribution of instance.score (VideoData1)")
        plt.xlabel("instance.score")
        plt.ylabel("Frequency (log scale)")
        plt.legend()
        plot_index += 1

    if has_v2:
        plt.subplot(1, 2 if has_v1 else 1, plot_index)
        plt.hist(
            video_data2["instance.score"].dropna(),
            bins=30,
            color="salmon",
            edgecolor="black",
        )
        if threshold_v2 is not None:
            plt.axvline(
                threshold_v2,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"{threshold_label} = {threshold_v2}",
            )
        plt.yscale("log")
        plt.title("Distribution of instance.score (VideoData2)")
        plt.xlabel("instance.score")
        plt.ylabel("Frequency (log scale)")
        plt.legend()
        plot_index += 1

    plt.tight_layout()
    plt.show()
