#!/usr/bin/env python3
"""
Batch export SLEAP prediction files (.slp) to CSV format matching pipeline requirements.

This script converts SLEAP .slp files to CSV format with:
- Exact column naming: left.x, left.y, center.x, center.y, right.x, right.y, p1.x-p8.y
- Score columns: left.score, center.score, right.score, p1.score-p8.score, instance.score
- frame_idx column (0-based per file)
- Proper file naming: VideoData1_YYYY-MM-DDTHH-MM-SS.sleap.csv
- Batch processing support

Usage:
    python batch_export_sleap_to_csv.py <input_directory> [--output-dir OUTPUT_DIR] [--pattern PATTERN]
    
    Or use as Python module:
    from sleap.batch_export_sleap_to_csv import export_sleap_to_csv
    export_sleap_to_csv('path/to/file.slp', 'output/path')
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import re

try:
    from sleap import Labels
    from sleap.io.video import Video
except ImportError:
    raise ImportError(
        "SLEAP is not installed. Install with: pip install sleap\n"
        "Or use: sleap-convert --format csv -o output.csv input.slp"
    )


# Expected column names matching pipeline requirements
EXPECTED_COLUMNS = [
    'left.x', 'left.y', 'center.x', 'center.y', 'right.x', 'right.y',
    'p1.x', 'p1.y', 'p2.x', 'p2.y', 'p3.x', 'p3.y', 'p4.x', 'p4.y',
    'p5.x', 'p5.y', 'p6.x', 'p6.y', 'p7.x', 'p7.y', 'p8.x', 'p8.y'
]

# Expected score columns matching pipeline requirements
EXPECTED_SCORE_COLUMNS = [
    'left.score', 'center.score', 'right.score',
    'p1.score', 'p2.score', 'p3.score', 'p4.score',
    'p5.score', 'p6.score', 'p7.score', 'p8.score',
    'instance.score'
]

# Node names that should map to our expected columns
NODE_MAPPING = {
    'left': 'left',
    'right': 'right',
    'center': 'center',
    'p1': 'p1', 'p2': 'p2', 'p3': 'p3', 'p4': 'p4',
    'p5': 'p5', 'p6': 'p6', 'p7': 'p7', 'p8': 'p8'
}


def normalize_node_name(node_name):
    """
    Normalize SLEAP node names to match expected format.
    Handles case variations and common naming conventions.
    """
    node_lower = node_name.lower().strip()
    
    # Try direct mapping first
    if node_lower in NODE_MAPPING:
        return NODE_MAPPING[node_lower]
    
    # Try to extract from patterns like "left_eye", "Left", "P1", etc.
    for key, mapped in NODE_MAPPING.items():
        if key.lower() in node_lower or node_lower in key.lower():
            return mapped
    
    # Try to match p1-p8 patterns
    p_match = re.match(r'p?(\d+)', node_lower)
    if p_match:
        num = p_match.group(1)
        if 1 <= int(num) <= 8:
            return f'p{num}'
    
    return None


def extract_timestamp_from_filename(filename):
    """
    Extract timestamp from filename in format YYYY-MM-DDTHH-MM-SS.
    Looks for patterns like VideoData1_2025-04-28T14-22-03.slp
    """
    # Pattern: YYYY-MM-DDTHH-MM-SS or YYYYMMDDTHHMMSS
    patterns = [
        r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})',  # 2025-04-28T14-22-03
        r'(\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})',       # 20250428T142203
        r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',  # 2025-04-28_14-22-03
    ]
    
    for pattern in patterns:
        match = re.search(pattern, str(filename))
        if match:
            ts_str = match.group(1)
            # Normalize to YYYY-MM-DDTHH-MM-SS format
            if 'T' not in ts_str:
                ts_str = ts_str.replace('_', 'T')
            if len(ts_str) == 15:  # YYYYMMDDTHHMMSS
                ts_str = f"{ts_str[:4]}-{ts_str[4:6]}-{ts_str[6:8]}T{ts_str[9:11]}-{ts_str[11:13]}-{ts_str[13:15]}"
            return ts_str
    
    return None


def get_video_timestamp(video_path):
    """
    Try to extract timestamp from video file metadata or filename.
    """
    timestamp = extract_timestamp_from_filename(video_path)
    if timestamp:
        return timestamp
    
    # If video has metadata, try to extract from there
    try:
        video = Video.from_filename(str(video_path))
        # SLEAP videos might have metadata with timestamps
        # This is a fallback - adjust based on your video metadata structure
        if hasattr(video, 'metadata') and video.metadata:
            # Try to extract timestamp from metadata if available
            pass
    except:
        pass
    
    return None


def export_sleap_to_csv(slp_path, output_path=None, video_path=None):
    """
    Export a single SLEAP .slp file to CSV format matching pipeline requirements.
    
    Parameters:
    -----------
    slp_path : Path or str
        Path to .slp file
    output_path : Path or str, optional
        Output CSV path. If None, will be generated from slp_path (same filename with .sleap.csv extension)
    video_path : Path or str, optional
        Path to corresponding video file (currently unused, reserved for future use)
        
    Returns:
    --------
    Path : Path to exported CSV file
    """
    slp_path = Path(slp_path)
    
    if not slp_path.exists():
        raise FileNotFoundError(f"SLEAP file not found: {slp_path}")
    
    if not slp_path.suffix.lower() == '.slp':
        raise ValueError(f"File must be a .slp file: {slp_path}")
    
    print(f"\n📂 Processing: {slp_path.name}")
    
    # Load SLEAP labels
    try:
        labels = Labels.load_file(str(slp_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load SLEAP file {slp_path}: {e}")
    
    if len(labels.videos) == 0:
        raise ValueError(f"No videos found in SLEAP file: {slp_path}")
    
    # Generate output path if not provided - use exact same filename as input
    if output_path is None:
        output_dir = slp_path.parent
        # Use the same filename as the .slp file, just change extension to .sleap.csv
        output_filename = slp_path.stem + ".sleap.csv"
        output_path = output_dir / output_filename
    else:
        output_path = Path(output_path)
        # If output_path is a directory, use the slp filename
        if output_path.is_dir():
            output_filename = slp_path.stem + ".sleap.csv"
            output_path = output_path / output_filename
    
    # Get node names from SLEAP labels
    skeleton = labels.skeleton
    node_names = [node.name for node in skeleton.nodes]
    
    print(f"   Found {len(node_names)} nodes: {', '.join(node_names)}")
    
    # Map nodes to expected column names
    node_to_column = {}
    for node_name in node_names:
        normalized = normalize_node_name(node_name)
        if normalized:
            node_to_column[node_name] = normalized
        else:
            print(f"   ⚠️ Warning: Could not map node '{node_name}' to expected column")
    
    # Check if we have required nodes
    required_nodes = ['left', 'right', 'center'] + [f'p{i}' for i in range(1, 9)]
    missing_nodes = [node for node in required_nodes if node not in node_to_column.values()]
    if missing_nodes:
        print(f"   ⚠️ Warning: Missing expected nodes: {missing_nodes}")
    
    # Build DataFrame - use SLEAP's built-in method if available, otherwise manual extraction
    try:
        # Try using SLEAP's to_dataframe method (if available)
        if hasattr(labels, 'to_dataframe'):
            df_raw = labels.to_dataframe()
            # This should give us a dataframe with columns for each node
            # We'll need to reshape it to match our expected format
            print("   ℹ️ Using SLEAP's to_dataframe() method")
    except:
        pass
    
    # Manual extraction method
    all_frames = []
    max_frame_idx = -1
    
    # Create a mapping of node names to indices
    node_to_idx = {node.name: idx for idx, node in enumerate(skeleton.nodes)}
    
    # Get all labeled frames
    for labeled_frame in labels.labeled_frames:
        frame_idx = labeled_frame.frame_idx
        max_frame_idx = max(max_frame_idx, frame_idx)
        
        row = {'frame_idx': frame_idx}
        
        # Initialize all coordinates to NaN
        for node_name, column_name in node_to_column.items():
            row[f'{column_name}.x'] = np.nan
            row[f'{column_name}.y'] = np.nan
        
        # Initialize all score columns to NaN
        for col in EXPECTED_SCORE_COLUMNS:
            row[col] = np.nan
        
        # Extract coordinates and scores for each instance
        for instance in labeled_frame.instances:
            if instance.skeleton == skeleton:
                # Extract instance score (overall confidence for this instance)
                try:
                    if hasattr(instance, 'score') and instance.score is not None:
                        row['instance.score'] = float(instance.score)
                    elif hasattr(instance, 'confidence') and instance.confidence is not None:
                        row['instance.score'] = float(instance.confidence)
                except (AttributeError, ValueError, TypeError):
                    pass
                
                # Try to get scores array from instance (if available)
                scores_array = None
                try:
                    if hasattr(instance, 'scores') and instance.scores is not None:
                        scores_array = np.array(instance.scores)
                except (AttributeError, ValueError, TypeError):
                    pass
                
                try:
                    # Get points as numpy array - this is the most reliable method
                    points_array = instance.numpy()
                    
                    if points_array is not None and len(points_array.shape) == 2:
                        # points_array shape: (num_nodes, 2) where 2 is [x, y]
                        for node_name, column_name in node_to_column.items():
                            if node_name in node_to_idx:
                                node_idx = node_to_idx[node_name]
                                if node_idx < points_array.shape[0]:
                                    coords = points_array[node_idx]
                                    if not np.isnan(coords[0]) and not np.isnan(coords[1]):
                                        row[f'{column_name}.x'] = float(coords[0])
                                        row[f'{column_name}.y'] = float(coords[1])
                                        
                                        # Extract score from scores array if available
                                        if scores_array is not None and node_idx < len(scores_array):
                                            try:
                                                row[f'{column_name}.score'] = float(scores_array[node_idx])
                                            except (IndexError, ValueError, TypeError):
                                                pass
                
                except Exception as e:
                    pass
                
                # Extract coordinates and scores via node dictionary access (more reliable for scores)
                for node_name, column_name in node_to_column.items():
                    if node_name in node_to_idx:
                        node_idx = node_to_idx[node_name]
                        node_obj = skeleton.nodes[node_idx]
                        
                        try:
                            point = instance[node_obj]
                            if point is not None:
                                # Extract coordinates
                                if hasattr(point, 'x') and hasattr(point, 'y'):
                                    row[f'{column_name}.x'] = float(point.x)
                                    row[f'{column_name}.y'] = float(point.y)
                                
                                # Extract score (only if not already set from scores_array)
                                if np.isnan(row.get(f'{column_name}.score', np.nan)):
                                    if hasattr(point, 'score') and point.score is not None:
                                        row[f'{column_name}.score'] = float(point.score)
                                    elif hasattr(point, 'confidence') and point.confidence is not None:
                                        row[f'{column_name}.score'] = float(point.confidence)
                                    elif hasattr(point, 'visible') and point.visible:
                                        # If point is visible but no score, set to 1.0 (default confidence)
                                        row[f'{column_name}.score'] = 1.0
                        except (KeyError, TypeError, AttributeError):
                            # Fallback: try accessing points directly
                            try:
                                if hasattr(instance, 'points'):
                                    if isinstance(instance.points, (list, tuple)) and node_idx < len(instance.points):
                                        point = instance.points[node_idx]
                                        if point is not None:
                                            if hasattr(point, 'x') and hasattr(point, 'y'):
                                                row[f'{column_name}.x'] = float(point.x)
                                                row[f'{column_name}.y'] = float(point.y)
                                            
                                            # Extract score (only if not already set)
                                            if np.isnan(row.get(f'{column_name}.score', np.nan)):
                                                if hasattr(point, 'score') and point.score is not None:
                                                    row[f'{column_name}.score'] = float(point.score)
                                                elif hasattr(point, 'confidence') and point.confidence is not None:
                                                    row[f'{column_name}.score'] = float(point.confidence)
                            except (IndexError, AttributeError, TypeError):
                                pass
        
        all_frames.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_frames)
    
    # Ensure we have all expected columns (fill with NaN if missing)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
            print(f"   ⚠️ Added missing column: {col} (filled with NaN)")
    
    # Ensure we have all expected score columns (fill with NaN if missing)
    for col in EXPECTED_SCORE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
            print(f"   ⚠️ Added missing column: {col} (filled with NaN)")
    
    # Reorder columns: frame_idx first, then expected columns, then score columns
    column_order = ['frame_idx'] + EXPECTED_COLUMNS + EXPECTED_SCORE_COLUMNS
    df = df[column_order]
    
    # Sort by frame_idx
    df = df.sort_values('frame_idx').reset_index(drop=True)
    
    # Fill gaps in frame_idx (if frames are missing, add rows with NaN)
    if len(df) > 0:
        expected_frames = set(range(df['frame_idx'].min(), df['frame_idx'].max() + 1))
        actual_frames = set(df['frame_idx'].values)
        missing_frames = expected_frames - actual_frames
        
        if missing_frames:
            print(f"   ℹ️ Found {len(missing_frames)} missing frames (will be filled with NaN)")
            missing_rows = []
            for frame_idx in sorted(missing_frames):
                row = {'frame_idx': frame_idx}
                for col in EXPECTED_COLUMNS + EXPECTED_SCORE_COLUMNS:
                    row[col] = np.nan
                missing_rows.append(row)
            
            missing_df = pd.DataFrame(missing_rows)
            df = pd.concat([df, missing_df], ignore_index=True)
            df = df.sort_values('frame_idx').reset_index(drop=True)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"   ✅ Exported to: {output_path.name}")
    print(f"      Frames: {len(df)}, frame_idx range: {df['frame_idx'].min()}-{df['frame_idx'].max()}")
    
    return output_path


def batch_export_directory(input_dir, output_dir=None, pattern="*.slp", recursive=False):
    """
    Batch export all .slp files in a directory.
    
    Parameters:
    -----------
    input_dir : Path or str
        Directory containing .slp files
    output_dir : Path or str, optional
        Output directory. If None, exports to same directory as .slp files
    pattern : str
        File pattern to match (default: "*.slp")
    recursive : bool
        Whether to search recursively in subdirectories
        
    Returns:
    --------
    list : List of exported CSV file paths
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .slp files
    if recursive:
        slp_files = list(input_dir.rglob(pattern))
    else:
        slp_files = list(input_dir.glob(pattern))
    
    if len(slp_files) == 0:
        print(f"⚠️ No .slp files found in {input_dir} matching pattern '{pattern}'")
        return []
    
    print(f"\n🔍 Found {len(slp_files)} .slp file(s) to process\n")
    
    exported_files = []
    failed_files = []
    
    for slp_file in sorted(slp_files):
        try:
            if output_dir:
                output_path = output_dir / f"{slp_file.stem}.sleap.csv"
            else:
                output_path = None
            
            exported = export_sleap_to_csv(
                slp_file, 
                output_path=output_path
            )
            exported_files.append(exported)
            
        except Exception as e:
            print(f"   ❌ Failed to export {slp_file.name}: {e}")
            failed_files.append((slp_file, str(e)))
    
    print(f"\n{'='*80}")
    print(f"✅ Batch export complete!")
    print(f"   Successfully exported: {len(exported_files)} file(s)")
    if failed_files:
        print(f"   Failed: {len(failed_files)} file(s)")
        for slp_file, error in failed_files:
            print(f"      - {slp_file.name}: {error}")
    print(f"{'='*80}\n")
    
    return exported_files


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Batch export SLEAP .slp files to CSV format matching pipeline requirements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all .slp files in a directory
  python batch_export_sleap_to_csv.py /path/to/sleap/files
  
  # Export to specific output directory
  python batch_export_sleap_to_csv.py /path/to/sleap/files --output-dir /path/to/output
  
  # Export only VideoData1 files
  python batch_export_sleap_to_csv.py /path/to/sleap/files --pattern "*VideoData1*.slp"
  
  # Export recursively
  python batch_export_sleap_to_csv.py /path/to/sleap/files --recursive
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing .slp files (or path to single .slp file)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for CSV files (default: same as input)'
    )
    
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*.slp',
        help='File pattern to match (default: "*.slp")'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search recursively in subdirectories'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    
    # Handle single file
    if input_path.is_file() and input_path.suffix.lower() == '.slp':
        try:
            export_sleap_to_csv(
                input_path,
                output_path=args.output_dir
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    # Handle directory
    elif input_path.is_dir():
        try:
            batch_export_directory(
                input_path,
                output_dir=args.output_dir,
                pattern=args.pattern,
                recursive=args.recursive
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        print(f"❌ Error: {input_path} is not a valid .slp file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()

