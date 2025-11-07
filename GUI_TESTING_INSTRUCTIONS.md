# Testing the GUI Annotation Tool

## Step 1: Install PyQt5

First, install PyQt5 in your conda environment:

```bash
# Activate your environment
conda activate aeon  # or aeon_ml if you're using that

# Install PyQt5
pip install PyQt5
```

## Step 2: Prepare Your Data

The GUI needs:
1. `saccade_results` - from `analyze_eye_video_saccades()`
2. `features_df` (optional) - from `extract_ml_features()`
3. `experiment_id` - from `extract_experiment_id()`

These should already be available in your notebook after running saccade detection.

## Step 3: Add GUI Launch Cell to Your Notebook

Add this cell at the end of your notebook (after feature extraction):

```python
# %%
# LAUNCH GUI ANNOTATION TOOL
############################################################################################################

from sleap.annotation_gui import launch_annotation_gui
from sleap.ml_feature_extraction import extract_experiment_id

# Get experiment ID
experiment_id = extract_experiment_id(data_path)
print(f"Experiment ID: {experiment_id}")

# Choose which video to annotate (VideoData1 or VideoData2)
# For now, let's use VideoData1 if available
video_key = 'VideoData1'  # Change to 'VideoData2' if you want to annotate the other eye

if video_key in saccade_results:
    # Get saccade results for this video
    video_saccade_results = saccade_results[video_key]
    
    # Get features for this video (if available)
    video_features = None
    if features_combined is not None and len(features_combined) > 0:
        # Filter features for this video
        if 'video_label' in features_combined.columns:
            video_label_match = VIDEO_LABELS.get(video_key, video_key)
            video_features = features_combined[
                features_combined['video_label'].str.contains(video_key.split('VideoData')[1], na=False)
            ].copy()
        else:
            # If no video_label column, use all features (will work but less ideal)
            video_features = features_combined.copy()
    
    # Set annotations file path
    annotations_file = data_path.parent / 'saccade_annotations_master.csv'
    # Or use a project-wide location:
    # annotations_file = Path('/path/to/project/data/annotations/saccade_annotations_master.csv')
    
    print(f"\n{'='*80}")
    print(f"Launching GUI Annotation Tool")
    print(f"{'='*80}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Video: {video_key} ({VIDEO_LABELS.get(video_key, video_key)})")
    print(f"Saccades: {len(video_saccade_results.get('all_saccades_df', pd.DataFrame()))}")
    print(f"Annotations file: {annotations_file}")
    print(f"{'='*80}\n")
    
    # Launch GUI
    launch_annotation_gui(
        saccade_results=video_saccade_results,
        features_df=video_features,
        experiment_id=experiment_id,
        annotations_file_path=annotations_file
    )
else:
    print(f"⚠️ {video_key} not found in saccade_results")
```

## Step 4: Run the Cell

When you run the cell, the GUI window should open. You'll see:

**Left Panel:**
- Table with all saccades (ID, Time, Amplitude, Duration, Rule-Based Label, User Label)
- Statistics showing total/annotated/remaining
- Classification buttons (1-4)
- Navigation buttons
- Notes field

**Right Panel:**
- Time series plot (position + velocity) with saccades highlighted
- Peri-saccade segment plot for selected saccade

## Step 5: Annotate Saccades

1. **Select a saccade** from the table (or use navigation buttons)
2. **View the plots** to see the saccade traces
3. **Classify** using buttons or keyboard shortcuts:
   - `1` = Compensatory
   - `2` = Orienting
   - `3` = Saccade-and-Fixate
   - `4` = Non-Saccade
4. **Add notes** (optional) in the notes field
5. **Save** with `S` key or Save button
6. **Navigate** with `N` (Next) or `P` (Previous)

## Step 6: Check Annotations

After annotating, check the CSV file:

```python
from sleap.annotation_storage import load_annotations, print_annotation_stats

# Load annotations
annotations = load_annotations(annotations_file, experiment_id=experiment_id)
print(f"\nAnnotated {len(annotations)} saccades:")
print(annotations[['saccade_id', 'user_label', 'time', 'amplitude']].head(10))

# Print statistics
print_annotation_stats(annotations_file, experiment_id=experiment_id)
```

## Troubleshooting

### GUI doesn't open
- Check PyQt5 is installed: `python -c "import PyQt5; print('OK')"`
- Check for errors in the notebook output

### No saccades shown
- Verify `saccade_results` contains data
- Check `all_saccades_df` has rows

### Plots are empty
- Verify `df` has 'Seconds', 'X_smooth', 'vel_x_smooth' columns
- Check `peri_saccades` list is not empty

### Can't save annotations
- Check file path is writable
- Check experiment_id is valid (no special characters)

## Quick Test (Minimal)

If you want to test quickly without your full data:

```python
# Minimal test
from sleap.annotation_gui import launch_annotation_gui
import pandas as pd
import numpy as np

# Create minimal dummy data
df = pd.DataFrame({
    'Seconds': np.linspace(0, 100, 1000),
    'X_smooth': np.cumsum(np.random.randn(1000) * 0.5),
    'vel_x_smooth': np.random.randn(1000) * 50
})

saccades_df = pd.DataFrame({
    'saccade_id': [1, 2, 3],
    'time': [20, 50, 80],
    'start_time': [19.9, 49.9, 79.9],
    'end_time': [20.1, 50.1, 80.1],
    'amplitude': [30, 45, 25],
    'duration': [0.1, 0.12, 0.08],
    'peak_velocity': [200, 250, 180],
    'saccade_type': ['compensatory', 'orienting', 'compensatory'],
    'classification_confidence': [0.8, 0.9, 0.7],
    'saccade_direction': ['upward', 'downward', 'upward']
})

# Create dummy segments
peri_saccades = []
for i, saccade in saccades_df.iterrows():
    t_rel = np.linspace(-0.15, 0.5, 65)
    segment = pd.DataFrame({
        'Time_rel_threshold': t_rel,
        'X_smooth_baselined': np.random.randn(65) * 5,
        'vel_x_smooth': np.random.randn(65) * 50,
        'saccade_id': saccade['saccade_id']
    })
    peri_saccades.append(segment)

saccade_results = {
    'df': df,
    'all_saccades_df': saccades_df,
    'peri_saccades': peri_saccades
}

# Launch GUI
launch_annotation_gui(
    saccade_results=saccade_results,
    experiment_id='test_experiment',
    annotations_file_path='test_annotations.csv'
)
```

This will open a GUI with 3 dummy saccades for testing.

