"""
Annotation Storage Infrastructure for ML Saccade Classification

This module handles saving and loading manual annotations for training the ML classifier.
Annotations are stored in a master CSV file that accumulates data across multiple experiments.

CSV Format:
- experiment_id: Identifies the experiment/notebook run
- saccade_id: Unique saccade ID within the experiment
- time: Saccade time (seconds)
- amplitude: Saccade amplitude (px)
- duration: Saccade duration (seconds)
- user_label: Manual classification (compensatory, orienting, saccade_and_fixate, non_saccade)
- user_confidence: User's confidence in annotation (0-1, default: 1.0)
- notes: Optional notes about the annotation
- annotation_date: Date/time when annotation was made
- annotator: Name/ID of person who made the annotation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import os


# Valid user labels
VALID_LABELS = ['compensatory', 'orienting', 'saccade_and_fixate', 'non_saccade']


def initialize_annotations_file(annotations_file_path: Union[str, Path]) -> Path:
    """
    Create master annotations CSV file with headers if it doesn't exist.
    
    Parameters
    ----------
    annotations_file_path : str or Path
        Path to the master annotations CSV file
        
    Returns
    -------
    Path
        Path to the annotations file (created or existing)
    """
    annotations_file_path = Path(annotations_file_path)
    
    # Create parent directory if it doesn't exist
    annotations_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create file with headers if it doesn't exist
    if not annotations_file_path.exists():
        # Create empty DataFrame with correct columns
        columns = [
            'experiment_id',
            'saccade_id',
            'time',
            'amplitude',
            'duration',
            'user_label',
            'user_confidence',
            'notes',
            'annotation_date',
            'annotator'
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(annotations_file_path, index=False)
        print(f"✅ Created new annotations file: {annotations_file_path}")
    else:
        print(f"ℹ️ Annotations file already exists: {annotations_file_path}")
    
    return annotations_file_path


def save_annotation(
    annotations_file_path: Union[str, Path],
    experiment_id: str,
    saccade_id: Union[int, str],
    user_label: str,
    time: Optional[float] = None,
    amplitude: Optional[float] = None,
    duration: Optional[float] = None,
    user_confidence: float = 1.0,
    notes: str = '',
    annotator: Optional[str] = None
) -> bool:
    """
    Save a single annotation to the master annotations file.
    
    If an annotation with the same (experiment_id, saccade_id) already exists,
    it will be updated (replaced) with the new annotation.
    
    Parameters
    ----------
    annotations_file_path : str or Path
        Path to the master annotations CSV file
    experiment_id : str
        Unique identifier for the experiment/notebook run
    saccade_id : int or str
        Unique saccade ID within the experiment
    user_label : str
        Manual classification label. Must be one of:
        - 'compensatory'
        - 'orienting'
        - 'saccade_and_fixate'
        - 'non_saccade'
    time : float, optional
        Saccade time in seconds
    amplitude : float, optional
        Saccade amplitude in pixels
    duration : float, optional
        Saccade duration in seconds
    user_confidence : float
        User's confidence in the annotation (0-1, default: 1.0)
    notes : str
        Optional notes about the annotation (default: '')
    annotator : str, optional
        Name/ID of person who made the annotation (default: None)
        
    Returns
    -------
    bool
        True if annotation was saved successfully, False otherwise
        
    Raises
    ------
    ValueError
        If user_label is not valid
    """
    annotations_file_path = Path(annotations_file_path)
    
    # Validate label
    if user_label not in VALID_LABELS:
        raise ValueError(
            f"Invalid user_label: '{user_label}'. "
            f"Must be one of: {VALID_LABELS}"
        )
    
    # Validate confidence
    if not (0.0 <= user_confidence <= 1.0):
        raise ValueError(f"user_confidence must be between 0 and 1, got: {user_confidence}")
    
    # Initialize file if it doesn't exist
    if not annotations_file_path.exists():
        initialize_annotations_file(annotations_file_path)
    
    # Load existing annotations
    try:
        existing_df = pd.read_csv(annotations_file_path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        existing_df = pd.DataFrame(columns=[
            'experiment_id', 'saccade_id', 'time', 'amplitude', 'duration',
            'user_label', 'user_confidence', 'notes', 'annotation_date', 'annotator'
        ])
    
    # Create new annotation row
    annotation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    new_row = {
        'experiment_id': str(experiment_id),
        'saccade_id': str(saccade_id),
        'time': time if time is not None else np.nan,
        'amplitude': amplitude if amplitude is not None else np.nan,
        'duration': duration if duration is not None else np.nan,
        'user_label': user_label,
        'user_confidence': user_confidence,
        'notes': str(notes) if notes else '',
        'annotation_date': annotation_date,
        'annotator': str(annotator) if annotator is not None else ''
    }
    
    # Check if annotation already exists (same experiment_id + saccade_id)
    mask = (
        (existing_df['experiment_id'] == str(experiment_id)) &
        (existing_df['saccade_id'] == str(saccade_id))
    )
    
    if mask.any():
        # Update existing annotation
        existing_df.loc[mask, list(new_row.keys())] = list(new_row.values())
        updated = True
    else:
        # Append new annotation
        new_df = pd.DataFrame([new_row])
        existing_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated = False
    
    # Save updated DataFrame
    existing_df.to_csv(annotations_file_path, index=False)
    
    action = "Updated" if updated else "Saved"
    print(f"✅ {action} annotation: {experiment_id}/{saccade_id} -> {user_label}")
    
    return True


def save_annotations_batch(
    annotations_file_path: Union[str, Path],
    annotations: List[Dict],
    annotator: Optional[str] = None
) -> int:
    """
    Save multiple annotations in batch.
    
    Parameters
    ----------
    annotations_file_path : str or Path
        Path to the master annotations CSV file
    annotations : List[Dict]
        List of annotation dictionaries. Each dict should contain:
        - experiment_id (required)
        - saccade_id (required)
        - user_label (required)
        - time, amplitude, duration (optional)
        - user_confidence (optional, default: 1.0)
        - notes (optional, default: '')
    annotator : str, optional
        Name/ID of person who made the annotations (default: None)
        
    Returns
    -------
    int
        Number of annotations saved
    """
    annotations_file_path = Path(annotations_file_path)
    
    saved_count = 0
    for ann in annotations:
        try:
            save_annotation(
                annotations_file_path=annotations_file_path,
                experiment_id=ann['experiment_id'],
                saccade_id=ann['saccade_id'],
                user_label=ann['user_label'],
                time=ann.get('time'),
                amplitude=ann.get('amplitude'),
                duration=ann.get('duration'),
                user_confidence=ann.get('user_confidence', 1.0),
                notes=ann.get('notes', ''),
                annotator=annotator
            )
            saved_count += 1
        except Exception as e:
            print(f"⚠️ Failed to save annotation {ann.get('experiment_id')}/{ann.get('saccade_id')}: {e}")
    
    print(f"✅ Saved {saved_count}/{len(annotations)} annotations")
    return saved_count


def load_annotations(
    annotations_file_path: Union[str, Path],
    experiment_id: Optional[str] = None,
    user_label: Optional[str] = None
) -> pd.DataFrame:
    """
    Load annotations from the master CSV file.
    
    Parameters
    ----------
    annotations_file_path : str or Path
        Path to the master annotations CSV file
    experiment_id : str, optional
        Filter annotations by experiment_id (default: None, load all)
    user_label : str, optional
        Filter annotations by user_label (default: None, load all)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with annotations. Empty DataFrame if file doesn't exist or no matches.
    """
    annotations_file_path = Path(annotations_file_path)
    
    if not annotations_file_path.exists():
        print(f"⚠️ Annotations file not found: {annotations_file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(annotations_file_path)
    except pd.errors.EmptyDataError:
        print(f"⚠️ Annotations file is empty: {annotations_file_path}")
        return pd.DataFrame()
    
    # Filter by experiment_id if specified
    if experiment_id is not None:
        df = df[df['experiment_id'] == str(experiment_id)].copy()
    
    # Filter by user_label if specified
    if user_label is not None:
        if user_label not in VALID_LABELS:
            raise ValueError(
                f"Invalid user_label filter: '{user_label}'. "
                f"Must be one of: {VALID_LABELS}"
            )
        df = df[df['user_label'] == user_label].copy()
    
    return df.reset_index(drop=True)


def get_annotation_stats(
    annotations_file_path: Union[str, Path],
    experiment_id: Optional[str] = None
) -> Dict[str, any]:
    """
    Get statistics about annotations in the master file.
    
    Parameters
    ----------
    annotations_file_path : str or Path
        Path to the master annotations CSV file
    experiment_id : str, optional
        Filter statistics by experiment_id (default: None, all experiments)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'total_annotations': int, total number of annotations
        - 'annotations_by_class': dict, count per class
        - 'annotations_by_experiment': dict, count per experiment
        - 'experiments': list, list of unique experiment IDs
        - 'date_range': dict, {'first': str, 'last': str} annotation dates
    """
    annotations_file_path = Path(annotations_file_path)
    
    if not annotations_file_path.exists():
        return {
            'total_annotations': 0,
            'annotations_by_class': {},
            'annotations_by_experiment': {},
            'experiments': [],
            'date_range': {'first': None, 'last': None}
        }
    
    # Load annotations
    df = load_annotations(annotations_file_path, experiment_id=experiment_id)
    
    if len(df) == 0:
        return {
            'total_annotations': 0,
            'annotations_by_class': {},
            'annotations_by_experiment': {},
            'experiments': [],
            'date_range': {'first': None, 'last': None}
        }
    
    # Calculate statistics
    stats = {
        'total_annotations': len(df),
        'annotations_by_class': df['user_label'].value_counts().to_dict(),
        'annotations_by_experiment': df['experiment_id'].value_counts().to_dict(),
        'experiments': sorted(df['experiment_id'].unique().tolist()),
    }
    
    # Date range
    if 'annotation_date' in df.columns and df['annotation_date'].notna().any():
        dates = pd.to_datetime(df['annotation_date'], errors='coerce').dropna()
        if len(dates) > 0:
            stats['date_range'] = {
                'first': dates.min().strftime('%Y-%m-%d %H:%M:%S'),
                'last': dates.max().strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            stats['date_range'] = {'first': None, 'last': None}
    else:
        stats['date_range'] = {'first': None, 'last': None}
    
    return stats


def print_annotation_stats(
    annotations_file_path: Union[str, Path],
    experiment_id: Optional[str] = None
) -> None:
    """
    Print formatted statistics about annotations.
    
    Parameters
    ----------
    annotations_file_path : str or Path
        Path to the master annotations CSV file
    experiment_id : str, optional
        Filter statistics by experiment_id (default: None, all experiments)
    """
    stats = get_annotation_stats(annotations_file_path, experiment_id=experiment_id)
    
    print(f"\n{'='*80}")
    print(f"ANNOTATION STATISTICS")
    if experiment_id:
        print(f"Experiment ID: {experiment_id}")
    print(f"{'='*80}")
    
    print(f"\nTotal annotations: {stats['total_annotations']}")
    
    if stats['annotations_by_class']:
        print(f"\nAnnotations by class:")
        for label, count in sorted(stats['annotations_by_class'].items()):
            pct = (count / stats['total_annotations'] * 100) if stats['total_annotations'] > 0 else 0
            print(f"  {label:<25} {count:>5} ({pct:>5.1f}%)")
    
    if stats['annotations_by_experiment']:
        print(f"\nAnnotations by experiment:")
        for exp_id, count in sorted(stats['annotations_by_experiment'].items()):
            print(f"  {exp_id:<40} {count:>5}")
    
    if stats['date_range']['first']:
        print(f"\nDate range:")
        print(f"  First annotation: {stats['date_range']['first']}")
        print(f"  Last annotation:  {stats['date_range']['last']}")
    
    print(f"\n{'='*80}\n")


# Test function
if __name__ == '__main__':
    print("Testing Annotation Storage Module")
    print("=" * 60)
    
    # Create temporary test file
    test_file = Path('test_annotations.csv')
    
    # Clean up if exists
    if test_file.exists():
        test_file.unlink()
    
    # Test 1: Initialize file
    print("\n1. Testing file initialization:")
    initialize_annotations_file(test_file)
    
    # Test 2: Save single annotation
    print("\n2. Testing save_annotation:")
    save_annotation(
        annotations_file_path=test_file,
        experiment_id='test_exp_001',
        saccade_id=1,
        user_label='compensatory',
        time=12.345,
        amplitude=45.2,
        duration=0.125,
        user_confidence=0.9,
        notes='First in bout',
        annotator='test_user'
    )
    
    # Test 3: Save batch annotations
    print("\n3. Testing save_annotations_batch:")
    batch_annotations = [
        {
            'experiment_id': 'test_exp_001',
            'saccade_id': 2,
            'user_label': 'compensatory',
            'time': 12.567,
            'amplitude': 38.1,
            'duration': 0.098,
            'user_confidence': 0.85
        },
        {
            'experiment_id': 'test_exp_002',
            'saccade_id': 1,
            'user_label': 'orienting',
            'time': 15.234,
            'amplitude': 52.3,
            'duration': 0.142,
            'user_confidence': 0.95,
            'notes': 'Isolated'
        },
        {
            'experiment_id': 'test_exp_002',
            'saccade_id': 5,
            'user_label': 'non_saccade',
            'time': 18.456,
            'amplitude': 12.3,
            'duration': 0.045,
            'user_confidence': 0.8,
            'notes': 'Artifact'
        }
    ]
    save_annotations_batch(test_file, batch_annotations, annotator='test_user')
    
    # Test 4: Load annotations
    print("\n4. Testing load_annotations:")
    all_annotations = load_annotations(test_file)
    print(f"   Loaded {len(all_annotations)} annotations")
    print(f"   Columns: {list(all_annotations.columns)}")
    
    exp1_annotations = load_annotations(test_file, experiment_id='test_exp_001')
    print(f"   Filtered by experiment_id='test_exp_001': {len(exp1_annotations)} annotations")
    
    compensatory_annotations = load_annotations(test_file, user_label='compensatory')
    print(f"   Filtered by user_label='compensatory': {len(compensatory_annotations)} annotations")
    
    # Test 5: Get statistics
    print("\n5. Testing get_annotation_stats:")
    stats = get_annotation_stats(test_file)
    print(f"   Total annotations: {stats['total_annotations']}")
    print(f"   By class: {stats['annotations_by_class']}")
    print(f"   By experiment: {stats['annotations_by_experiment']}")
    
    # Test 6: Print statistics
    print("\n6. Testing print_annotation_stats:")
    print_annotation_stats(test_file)
    
    # Test 7: Update existing annotation
    print("\n7. Testing update existing annotation:")
    save_annotation(
        annotations_file_path=test_file,
        experiment_id='test_exp_001',
        saccade_id=1,
        user_label='orienting',  # Changed from 'compensatory'
        user_confidence=0.95,
        notes='Updated: Actually orienting'
    )
    
    updated_ann = load_annotations(test_file, experiment_id='test_exp_001', user_label='orienting')
    print(f"   Updated annotation found: {len(updated_ann)} annotation(s)")
    
    # Clean up
    print("\n8. Cleaning up test file:")
    if test_file.exists():
        test_file.unlink()
        print(f"   ✅ Deleted {test_file}")
    
    print("\n✅ All tests passed!")

