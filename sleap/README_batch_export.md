# Batch Export SLEAP Predictions to CSV

This script converts SLEAP `.slp` prediction files to CSV format matching the exact requirements of the vestibular_vr_pipeline.

## Features

- ✅ **Exact column matching**: Exports columns as `left.x`, `left.y`, `center.x`, `center.y`, `right.x`, `right.y`, `p1.x`-`p8.y`
- ✅ **frame_idx column**: Creates 0-based `frame_idx` column matching pipeline expectations
- ✅ **Automatic naming**: Generates filenames like `VideoData1_2025-04-28T14-22-03.sleap.csv`
- ✅ **Batch processing**: Process multiple files at once
- ✅ **Gap filling**: Automatically fills missing frames with NaN
- ✅ **Node mapping**: Handles various node naming conventions

## Installation

Make sure SLEAP is installed:
```bash
pip install sleap
```

## Usage

### Command Line

```bash
# Export all .slp files in a directory
python sleap/batch_export_sleap_to_csv.py /path/to/sleap/files

# Export to specific output directory
python sleap/batch_export_sleap_to_csv.py /path/to/sleap/files --output-dir /path/to/output

# Export only VideoData1 files
python sleap/batch_export_sleap_to_csv.py /path/to/sleap/files --pattern "*VideoData1*.slp"

# Export recursively through subdirectories
python sleap/batch_export_sleap_to_csv.py /path/to/sleap/files --recursive

# Specify VideoData number explicitly
python sleap/batch_export_sleap_to_csv.py /path/to/sleap/files --video-data-number 1
```

### Python API

```python
from sleap.batch_export_sleap_to_csv import export_sleap_to_csv, batch_export_directory
from pathlib import Path

# Export single file
export_sleap_to_csv(
    'path/to/predictions.slp',
    output_path='path/to/VideoData1_2025-04-28T14-22-03.sleap.csv',
    video_data_number=1
)

# Batch export directory
batch_export_directory(
    '/path/to/sleap/files',
    output_dir='/path/to/output',
    pattern='*.slp',
    video_data_number=1,
    recursive=True
)
```

## Expected Output Format

The script generates CSV files with exactly these columns:

```
frame_idx,left.x,left.y,center.x,center.y,right.x,right.y,p1.x,p1.y,p2.x,p2.y,...,p8.x,p8.y
0,123.45,456.78,234.56,567.89,345.67,678.90,...
1,124.12,457.23,235.01,568.45,346.12,679.56,...
...
```

## Node Name Mapping

The script automatically maps SLEAP node names to expected columns:
- `left`, `left_eye`, `Left` → `left`
- `right`, `right_eye`, `Right` → `right`
- `center`, `Center`, `pupil_center` → `center`
- `p1`, `P1`, `point1`, `iris_1` → `p1`
- ... (same for p2-p8)

## File Naming

The script extracts timestamps from filenames or generates them automatically:
- Input: `VideoData1_2025-04-28T14-22-03.slp` → Output: `VideoData1_2025-04-28T14-22-03.sleap.csv`
- Input: `predictions.slp` → Output: `VideoData1_2025-04-28T14-22-03.sleap.csv` (uses current time if timestamp not found)

## Troubleshooting

### Missing columns warning
If you see warnings about missing columns, check that your SLEAP model has nodes named:
- `left`, `right`, `center`
- `p1`, `p2`, `p3`, `p4`, `p5`, `p6`, `p7`, `p8`

### Timestamp extraction
If timestamps aren't extracted correctly, manually specify the output filename:
```python
export_sleap_to_csv(
    'file.slp',
    output_path='VideoData1_2025-04-28T14-22-03.sleap.csv'
)
```

### Missing frames
The script automatically fills gaps in frame_idx with NaN values, which matches pipeline behavior.

