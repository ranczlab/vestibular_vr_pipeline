"""
Visualization functions for saccade analysis.

This module contains plotting functions extracted from notebooks to improve
code organization and reusability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Dict, Optional, List, Tuple


def visualize_ml_features(
    features_combined: pd.DataFrame,
    feature_categories: Dict[str, List[str]],
    video_labels: Dict[str, str],
    show_plots: bool = True,
    verbose: bool = True
) -> None:
    """
    Visualize ML feature distributions with violin plots and class comparisons.
    
    Creates two panels:
    - Panel 1: Feature distributions by category (violin plots, left/right eye, colored by class)
    - Panel 2: Key features by rule-based classification (class comparison)
    
    Parameters
    ----------
    features_combined : pd.DataFrame
        Combined features DataFrame from both eyes (or single eye)
        Must contain columns: 'eye', 'rule_based_class', and feature columns
    feature_categories : dict
        Dictionary mapping category names to lists of feature names
        Example: {'Category A: Basic Properties': ['amplitude', 'duration', ...]}
    video_labels : dict
        Dictionary mapping video keys to display labels
        Example: {'VideoData1': 'VideoData1 (L: Left)', 'VideoData2': 'VideoData2 (R: Right)'}
    show_plots : bool
        Whether to display plots (default: True)
    verbose : bool
        Whether to print diagnostic information (default: True)
    """
    if features_combined is None or len(features_combined) == 0:
        if verbose:
            print("⚠️ No features available for visualization. Run feature extraction first.")
        return
    
    # Get all feature columns (exclude metadata columns)
    metadata_cols = ['experiment_id', 'saccade_id', 'video_label', 'eye']
    all_feature_cols = [col for col in features_combined.columns if col not in metadata_cols]
    
    # Filter to only numeric features (exclude categorical like direction if encoded as string)
    numeric_features = []
    for col in all_feature_cols:
        if features_combined[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Check if column has valid numeric data
            if features_combined[col].notna().sum() > 0:
                numeric_features.append(col)
    
    if verbose:
        print(f"\n📊 Preparing visualizations for {len(numeric_features)} numeric features...")
        print(f"   Total saccades: {len(features_combined)}")
        if 'eye' in features_combined.columns:
            eye_counts = features_combined['eye'].value_counts()
            print(f"   By eye: {dict(eye_counts)}")
        if 'rule_based_class' in features_combined.columns:
            class_counts = features_combined['rule_based_class'].value_counts()
            print(f"   By rule-based class: {dict(class_counts)}")
    
    # Map rule_based_class to readable labels BEFORE normalization
    # (rule_based_class should NOT be normalized - it's a categorical label)
    if verbose:
        print("\n" + "="*80)
        print("🔍 DIAGNOSTIC: Checking data before visualization")
        print("="*80)
    
    features_normalized = features_combined.copy()
    
    # Save original rule_based_class values before any processing
    if 'rule_based_class' in features_normalized.columns:
        # Round to nearest integer to handle any float conversion issues
        features_normalized['rule_based_class'] = features_normalized['rule_based_class'].round().astype(int)
        # Classification encoding scheme (used by both rule-based and ML):
        #   0 = Compensatory
        #   1 = Orienting (Purely Orienting)
        #   2 = Saccade-Fixate (first saccade of compensatory bout)
        #   3 = Non-Saccade (false positives to exclude)
        #   -1 = Unclassified/Unknown
        # Current rule-based classification only produces classes 0 and 1.
        # ML classification will produce all 4 classes (0, 1, 2, 3).
        class_map = {0: 'Compensatory', 1: 'Orienting', 2: 'Saccade-Fixate', 3: 'Non-Saccade', -1: 'Unclassified'}
        features_normalized['rule_based_class_label'] = features_normalized['rule_based_class'].map(class_map)
        # Fill any unmapped values with 'Unclassified'
        features_normalized['rule_based_class_label'] = features_normalized['rule_based_class_label'].fillna('Unclassified')
        if verbose:
            print(f"✅ rule_based_class column found")
            print(f"   Unique rule_based_class values: {sorted(features_normalized['rule_based_class'].unique())}")
            print(f"   Unique rule_based_class_label values: {features_normalized['rule_based_class_label'].unique()}")
            print(f"   Counts by label:")
            for label, count in features_normalized['rule_based_class_label'].value_counts().items():
                print(f"     - {label}: {count}")
            
            # Additional diagnostic: check for NaN values
            nan_count = features_normalized['rule_based_class'].isna().sum()
            if nan_count > 0:
                print(f"   ⚠️ WARNING: {nan_count} rows have NaN rule_based_class values")
    else:
        if verbose:
            print("❌ 'rule_based_class' column NOT found in features_normalized")
            print(f"   Available columns: {list(features_normalized.columns)[:20]}...")  # Show first 20
    
    # Normalize features using StandardScaler (as model will use)
    # Exclude rule_based_class from normalization (it's categorical, not numeric)
    # Only normalize numeric features, keep metadata columns as-is
    scaler = StandardScaler()
    
    # Exclude rule_based_class and rule_based_class_label from normalization
    features_to_normalize = [col for col in numeric_features if col not in ['rule_based_class', 'rule_based_class_label']]
    
    # Normalize each feature separately (handle NaN values)
    for col in features_to_normalize:
        valid_mask = features_normalized[col].notna()
        if valid_mask.sum() > 1:  # Need at least 2 values to normalize
            valid_data = features_normalized.loc[valid_mask, col].values.reshape(-1, 1)
            normalized_data = scaler.fit_transform(valid_data).flatten()
            features_normalized.loc[valid_mask, col] = normalized_data
    
    # Diagnostic: Check eye column
    if verbose:
        if 'eye' in features_normalized.columns:
            print(f"\n✅ eye column found")
            print(f"   Unique eye values: {features_normalized['eye'].unique()}")
            print(f"   Counts by eye:")
            for eye, count in features_normalized['eye'].value_counts().items():
                print(f"     - {eye}: {count}")
        else:
            print("\n❌ 'eye' column NOT found in features_normalized")
        
        # Diagnostic: Check numeric features
        print(f"\n✅ Numeric features check")
        print(f"   Total numeric features: {len(numeric_features)}")
        print(f"   Features normalized: {len(features_to_normalize)}")
        print(f"   Features excluded from normalization: {len(numeric_features) - len(features_to_normalize)}")
        print(f"   Total rows in features_normalized: {len(features_normalized)}")
        if len(numeric_features) > 0:
            print(f"   First 10 numeric features: {numeric_features[:10]}")
    
    # ============================================================================
    # PANEL 1: Feature Distributions by Category (Violin Plots)
    # ============================================================================
    if verbose:
        print("\n" + "="*80)
        print("PANEL 1: Feature Distributions by Category (Violin Plots)")
        print("="*80)
    
    # Prepare data for plotting: melt features into long format
    plot_data_list = []
    
    if verbose:
        print(f"\n🔍 DIAGNOSTIC: Preparing Panel 1 data")
        print(f"   Iterating over {len(features_normalized)} rows")
        print(f"   Processing {len(features_to_normalize)} normalized features (excluding rule_based_class)")
    
    rows_processed = 0
    features_with_data = 0
    
    for idx, row in features_normalized.iterrows():
        eye_val = row.get('eye', 'Unknown')
        class_label = row.get('rule_based_class_label', 'Unknown')
        # Handle NaN values - convert to 'Unclassified'
        if pd.isna(class_label) or class_label == 'Unknown':
            class_label = 'Unclassified'
        
        # Only iterate over normalized features (exclude rule_based_class)
        for feat in features_to_normalize:
            feat_val = row[feat]
            if pd.notna(feat_val):
                features_with_data += 1
                # Determine category
                category = 'Other'
                for cat_name, cat_features in feature_categories.items():
                    if feat in cat_features:
                        category = cat_name
                        break
                
                plot_data_list.append({
                    'Feature': feat,
                    'Category': category,
                    'Value': feat_val,
                    'Eye': eye_val,
                    'Rule_Based_Class': class_label
                })
        rows_processed += 1
    
    plot_df = pd.DataFrame(plot_data_list)
    if verbose:
        print(f"   ✅ Processed {rows_processed} rows")
        print(f"   ✅ Created {len(plot_data_list)} data points")
        print(f"   ✅ Features with non-NaN values: {features_with_data}")
        print(f"   ✅ plot_df shape: {plot_df.shape}")
        
        if len(plot_df) > 0:
            print(f"   ✅ Columns in plot_df: {list(plot_df.columns)}")
            print(f"   ✅ Unique Eye values: {plot_df['Eye'].unique()}")
            print(f"   ✅ Unique Rule_Based_Class values: {plot_df['Rule_Based_Class'].unique()}")
            print(f"   ✅ Unique Categories: {plot_df['Category'].unique()}")
            print(f"   ✅ Sample data (first 3 rows):")
            print(plot_df.head(3).to_string())
        else:
            print("   ❌ ERROR: plot_data_list is EMPTY!")
            print("   This means no data points were created for plotting")
    
    # Create subplots for each category
    categories = list(feature_categories.keys()) + ['Other']
    n_categories = len([c for c in categories if len(plot_df[plot_df['Category'] == c]) > 0])
    
    if verbose:
        print(f"\n🔍 DIAGNOSTIC: Panel 1 categories")
        print(f"   Total categories defined: {len(categories)}")
        print(f"   Categories with data: {n_categories}")
        for cat in categories:
            cat_data = plot_df[plot_df['Category'] == cat]
            if len(cat_data) > 0:
                unique_features = cat_data['Feature'].unique()
                print(f"   ✅ {cat}: {len(cat_data)} data points, {len(unique_features)} unique features")
                if len(unique_features) <= 5:
                    print(f"      Features: {list(unique_features)}")
            else:
                print(f"   ❌ {cat}: NO DATA")
    
    # Calculate number of features per category
    features_per_category = {}
    for cat in categories:
        cat_features = plot_df[plot_df['Category'] == cat]['Feature'].unique()
        features_per_category[cat] = len(cat_features)
    
    # Create figure with subplots for each category
    # Use 2 columns, multiple rows
    n_cols = 2
    n_rows = int(np.ceil(n_categories / n_cols))
    
    if n_categories == 0:
        if verbose:
            print("\n❌ ERROR: No categories have data - Panel 1 will be empty!")
            print("   Skipping Panel 1 figure creation")
    else:
        if verbose:
            print(f"\n   Creating Panel 1 figure: {n_rows} rows x {n_cols} cols = {n_rows * n_cols} subplots")
        
        if show_plots:
            fig_dist, axes_dist = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
            if n_rows == 1:
                axes_dist = axes_dist.reshape(1, -1)
            axes_dist = axes_dist.flatten()
            
            plot_idx = 0
            for cat_idx, category in enumerate(categories):
                cat_data = plot_df[plot_df['Category'] == category]  # Fixed: use 'category' not 'cat'
                if len(cat_data) == 0:
                    continue
                
                cat_features = sorted(cat_data['Feature'].unique())
                if len(cat_features) == 0:
                    continue
                
                if verbose:
                    print(f"   🔍 Plotting category '{category}' with {len(cat_features)} features (plot_idx={plot_idx})")
                
                ax = axes_dist[plot_idx]
                plot_data_cat = cat_data.copy()
                
                # Color palette for rule-based classes
                class_palette = {
                    'Compensatory': '#FF7F00',      # Orange
                    'Orienting': '#1F77B4',         # Blue
                    'Saccade-Fixate': '#2CA02C',    # Green
                    'Non-Saccade': '#D62728',       # Red
                    'Unknown': '#7F7F7F'            # Gray
                }
                
                # Create positions for features
                n_features = len(cat_features)
                x_positions = np.arange(n_features)
                
                # For each feature, create side-by-side violins for Left/Right
                # Position: feature_idx - 0.2 (Left), feature_idx + 0.2 (Right)
                # Color by class (overlay multiple classes)
                
                for feat_idx, feat in enumerate(cat_features):
                    feat_data = plot_data_cat[plot_data_cat['Feature'] == feat]
                    
                    for eye in ['Left', 'Right']:
                        eye_data = feat_data[feat_data['Eye'] == eye]
                        if len(eye_data) == 0:
                            continue
                        
                        # Position: base position + eye offset
                        base_pos = feat_idx
                        eye_offset = -0.2 if eye == 'Left' else 0.2
                        position = base_pos + eye_offset
                        
                        # For each class, create a separate violin
                        for class_label in sorted(eye_data['Rule_Based_Class'].unique()):
                            class_data = eye_data[eye_data['Rule_Based_Class'] == class_label]
                            if len(class_data) > 0:
                                # Small offset for class within eye group
                                class_offsets = {
                                    'Compensatory': -0.06,
                                    'Orienting': -0.02,
                                    'Saccade-Fixate': 0.02,
                                    'Non-Saccade': 0.06,
                                    'Unknown': 0.0
                                }
                                class_offset = class_offsets.get(class_label, 0.0)
                                final_position = position + class_offset
                                
                                # Create violin data
                                values = class_data['Value'].values
                                if len(values) > 1:  # Need at least 2 points for violin
                                    # Use matplotlib violinplot for this group
                                    parts = ax.violinplot(
                                        [values],
                                        positions=[final_position],
                                        widths=0.08,
                                        showmeans=False,
                                        showmedians=True
                                    )
                                    
                                    # Color by class
                                    color = class_palette.get(class_label, '#7F7F7F')
                                    for pc in parts['bodies']:
                                        pc.set_facecolor(color)
                                        pc.set_alpha(0.7)
                                    if parts['cmedians']:
                                        parts['cmedians'].set_color('black')
                                        parts['cmedians'].set_linewidth(1.5)
                
                # Set x-axis
                ax.set_xticks(x_positions)
                ax.set_xticklabels(cat_features, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Normalized Feature Value', fontsize=10)
                ax.set_title(f'{category}\n({len(cat_features)} features)', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
                
                # Add vertical lines to separate features
                for feat_idx in range(n_features - 1):
                    ax.axvline(x=feat_idx + 0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
                
                plot_idx += 1
                if verbose:
                    print(f"   ✅ Plotted category '{category}' (subplot {plot_idx}/{n_categories})")
            
            # Hide unused subplots
            for idx in range(plot_idx, len(axes_dist)):
                axes_dist[idx].axis('off')
            
            if verbose:
                print(f"\n🔍 DIAGNOSTIC: Panel 1 summary")
                print(f"   Total categories plotted: {plot_idx}")
                print(f"   Total subplots created: {len(axes_dist)}")
            
            plt.suptitle('Feature Distributions by Category: Left vs Right Eye (Colored by Rule-Based Class)', 
                         fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plt.show()
    
    # ============================================================================
    # PANEL 2: Rule-Based Classification Comparison (Key Features by Class)
    # ============================================================================
    if verbose:
        print("\n" + "="*80)
        print("PANEL 2: Rule-Based Classification Comparison (Key Features)")
        print("="*80)
    
    # Select key features that are most informative for classification
    # Focus on features that distinguish between classes
    key_features_for_class_comparison = [
        'amplitude', 'duration', 'peak_velocity',
        'pre_saccade_mean_velocity', 'pre_saccade_position_drift',
        'post_saccade_position_variance', 'post_saccade_position_change',
        'bout_size', 'time_since_previous_saccade', 'time_until_next_saccade',
        'is_first_in_bout', 'is_isolated', 'bout_duration',
        'amplitude_relative_to_bout_mean', 'rule_based_confidence'
    ]
    
    # Filter to only features that exist in the data (and were normalized)
    available_key_features = [f for f in key_features_for_class_comparison if f in features_to_normalize]
    
    if verbose:
        print(f"\n🔍 DIAGNOSTIC: Preparing Panel 2 data")
        print(f"   Key features requested: {len(key_features_for_class_comparison)}")
        print(f"   Available key features: {len(available_key_features)}")
        print(f"   Has rule_based_class_label column: {'rule_based_class_label' in features_normalized.columns}")
    
    if len(available_key_features) > 0 and 'rule_based_class_label' in features_normalized.columns:
        # Prepare data for Panel 2: focus on class comparison
        class_plot_data = []
        rows_skipped_unknown = 0
        rows_processed = 0
        
        for idx, row in features_normalized.iterrows():
            class_label = row.get('rule_based_class_label', 'Unknown')
            # Handle NaN values - convert to 'Unclassified'
            if pd.isna(class_label) or class_label == 'Unknown':
                class_label = 'Unclassified'
            
            if class_label == 'Unclassified' or pd.isna(class_label):
                rows_skipped_unknown += 1
                continue
            
            for feat in available_key_features:
                feat_val = row[feat]
                if pd.notna(feat_val):
                    class_plot_data.append({
                        'Feature': feat,
                        'Value': feat_val,
                        'Rule_Based_Class': class_label
                    })
            rows_processed += 1
        
        class_plot_df = pd.DataFrame(class_plot_data)
        
        if verbose:
            print(f"   ✅ Processed {rows_processed} rows")
            print(f"   ⚠️ Skipped {rows_skipped_unknown} rows with 'Unknown' or NaN class label")
            print(f"   ✅ Created {len(class_plot_data)} data points")
            print(f"   ✅ class_plot_df shape: {class_plot_df.shape}")
            
            if len(class_plot_df) > 0:
                print(f"   ✅ Columns in class_plot_df: {list(class_plot_df.columns)}")
                print(f"   ✅ Unique Rule_Based_Class values: {class_plot_df['Rule_Based_Class'].unique()}")
                print(f"   ✅ Unique Features: {class_plot_df['Feature'].unique()}")
                print(f"   ✅ Sample data (first 3 rows):")
                print(class_plot_df.head(3).to_string())
            else:
                print("   ❌ ERROR: class_plot_data is EMPTY!")
                print("   Checking why:")
                print(f"     - Total rows in features_normalized: {len(features_normalized)}")
                if 'rule_based_class_label' in features_normalized.columns:
                    class_label_counts = features_normalized['rule_based_class_label'].value_counts()
                    print(f"     - Class label counts:")
                    for label, count in class_label_counts.items():
                        print(f"       - {label}: {count}")
                    unknown_count = (features_normalized['rule_based_class_label'].isin(['Unknown']) | features_normalized['rule_based_class_label'].isna()).sum()
                    print(f"     - Rows with 'Unknown' or NaN: {unknown_count}")
        
        # Check if we have data to plot
        if len(class_plot_df) == 0 or 'Feature' not in class_plot_df.columns:
            if verbose:
                print("\n❌ Cannot create Panel 2: No data available for class comparison")
        else:
            if verbose:
                print(f"\n   ✅ Creating Panel 2 figure with {len(available_key_features)} features")
            # Create figure for Panel 2
            n_key_features = len(available_key_features)
            n_cols_p3 = 3
            n_rows_p3 = int(np.ceil(n_key_features / n_cols_p3))
            
            if verbose:
                print(f"   Figure layout: {n_rows_p3} rows x {n_cols_p3} cols = {n_rows_p3 * n_cols_p3} subplots")
            
            if show_plots:
                fig_class, axes_class = plt.subplots(n_rows_p3, n_cols_p3, figsize=(18, 5 * n_rows_p3))
                if n_rows_p3 == 1:
                    axes_class = axes_class.reshape(1, -1)
                axes_class = axes_class.flatten()
                
                # Color palette for classes
                class_palette_p3 = {
                    'Compensatory': '#FF7F00',      # Orange
                    'Orienting': '#1F77B4',         # Blue
                    'Saccade-Fixate': '#2CA02C',    # Green
                    'Non-Saccade': '#D62728',       # Red
                    'Unknown': '#7F7F7F'            # Gray
                }
                
                for feat_idx, feat in enumerate(available_key_features):
                    ax = axes_class[feat_idx]
                    feat_data = class_plot_df[class_plot_df['Feature'] == feat]
                    
                    if len(feat_data) == 0:
                        ax.axis('off')
                        continue
                    
                    # Create violin plots grouped by class
                    classes = sorted(feat_data['Rule_Based_Class'].unique())
                    positions = np.arange(len(classes))
                    
                    violin_data_by_class = []
                    for class_label in classes:
                        class_values = feat_data[feat_data['Rule_Based_Class'] == class_label]['Value'].values
                        if len(class_values) > 1:  # Need at least 2 points for violin
                            violin_data_by_class.append(class_values)
                        else:
                            violin_data_by_class.append([])
                    
                    # Plot violins
                    parts = ax.violinplot(
                        violin_data_by_class,
                        positions=positions,
                        widths=0.6,
                        showmeans=False,
                        showmedians=True
                    )
                    
                    # Color violins by class
                    for i, (pc, class_label) in enumerate(zip(parts['bodies'], classes)):
                        color = class_palette_p3.get(class_label, '#7F7F7F')
                        pc.set_facecolor(color)
                        pc.set_alpha(0.7)
                    
                    if parts['cmedians']:
                        parts['cmedians'].set_color('black')
                        parts['cmedians'].set_linewidth(1.5)
                    
                    # Set labels
                    ax.set_xticks(positions)
                    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
                    ax.set_ylabel('Normalized Feature Value', fontsize=10)
                    ax.set_title(feat, fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
                
                # Hide unused subplots
                for idx in range(n_key_features, len(axes_class)):
                    axes_class[idx].axis('off')
                
                plt.suptitle('Key Features by Rule-Based Classification (Class Comparison)', 
                             fontsize=14, fontweight='bold', y=0.995)
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                plt.show()
                
                if verbose:
                    print(f"\n✅ Panel 2 complete: Compared {len(available_key_features)} key features across classes")
    else:
        if verbose:
            print("⚠️ Cannot create Panel 2: Missing rule-based classification labels or key features")
    
    # Create legend for rule-based classes
    if show_plots:
        fig_legend, ax_legend = plt.subplots(figsize=(8, 1))
        color_map = {
            'Compensatory': '#FF7F00',
            'Orienting': '#1F77B4',
            'Saccade-Fixate': '#2CA02C',
            'Non-Saccade': '#D62728',
            'Unknown': '#7F7F7F'
        }
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=label) 
                           for label, color in color_map.items()]
        ax_legend.legend(handles=legend_elements, loc='center', ncol=5)
        ax_legend.axis('off')
        plt.title('Rule-Based Classification Colors', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    if verbose:
        print("\n✅ Feature visualization complete!")
        print(f"   Normalized {len(numeric_features)} features")
        print(f"   Visualized {n_categories} categories")
        print(f"   Created 2 panels: Category Distributions, Class Comparison")


# ============================================================================
# Helper Functions
# ============================================================================

def get_eye_label(video_key: str, video_labels: Dict[str, str]) -> str:
    """
    Return mapped user-viewable eye label for video key.
    
    Parameters
    ----------
    video_key : str
        Key identifying the video (e.g., 'VideoData1', 'VideoData2')
    video_labels : dict
        Dictionary mapping video keys to display labels
        Example: {'VideoData1': 'VideoData1 (L: Left)', 'VideoData2': 'VideoData2 (R: Right)'}
    
    Returns
    -------
    str
        Display label for the video
    """
    return video_labels.get(video_key, video_key)


def get_direction_map_for_video(video_key: str, video1_eye: str, video2_eye: str) -> Dict[str, str]:
    """
    Get direction mapping for a video based on which eye it represents.
    
    Parameters
    ----------
    video_key : str
        Key identifying the video (e.g., 'VideoData1', 'VideoData2')
    video1_eye : str
        Eye label for VideoData1 ('L' or 'R')
    video2_eye : str
        Eye label for VideoData2 ('L' or 'R')
    
    Returns
    -------
    dict
        Dictionary mapping 'upward' and 'downward' to their labels
        For left eye: {'upward': 'NT', 'downward': 'TN'}
        For right eye: {'upward': 'TN', 'downward': 'NT'}
    """
    eye = video1_eye if video_key == 'VideoData1' else video2_eye
    if eye == 'L':
        return {'upward': 'NT', 'downward': 'TN'}
    else:
        return {'upward': 'TN', 'downward': 'NT'}


# ============================================================================
# Main Visualization Functions
# ============================================================================

def plot_all_saccades_overlay(
    saccade_results: Dict[str, Dict],
    video_labels: Dict[str, str],
    video1_eye: str,
    video2_eye: str,
    pre_saccade_window_time: float,
    post_saccade_window_time: float,
    debug: bool = True,
    show_plot: bool = True
) -> None:
    """
    Plot all upward and downward saccades aligned by time with position and velocity traces.
    
    Creates a figure with 4 columns in a single row: upward position, upward velocity,
    downward position, downward velocity. All saccades are overlaid with semi-transparent
    traces, and mean traces are shown in bold.
    
    Parameters
    ----------
    saccade_results : dict
        Dictionary containing saccade analysis results for each video
        Each entry should contain:
        - 'df': DataFrame with 'Seconds' column for FPS calculation
        - 'upward_saccades_df': DataFrame with upward saccades
        - 'downward_saccades_df': DataFrame with downward saccades
        - 'peri_saccades': List of DataFrames with peri-saccade segments
        - 'upward_segments': List of DataFrames with upward segments
        - 'downward_segments': List of DataFrames with downward segments
    video_labels : dict
        Dictionary mapping video keys to display labels
    video1_eye : str
        Eye label for VideoData1 ('L' or 'R')
    video2_eye : str
        Eye label for VideoData2 ('L' or 'R')
    pre_saccade_window_time : float
        Time (seconds) before threshold crossing to extract
    post_saccade_window_time : float
        Time (seconds) after threshold crossing to extract
    debug : bool
        Whether to print debug information (default: True)
    show_plot : bool
        Whether to display the plot (default: True)
    """
    # Calculate point-based values from time-based parameters for visualization
    # (These are calculated from the first video's FPS - should be similar for both videos)
    n_before = None
    n_after = None
    
    for video_key, res in saccade_results.items():
        # Get FPS from the results (stored in df)
        fps_for_viz = 1 / res['df']['Seconds'].diff().mean()
        n_before = max(1, int(round(pre_saccade_window_time * fps_for_viz)))
        n_after = max(1, int(round(post_saccade_window_time * fps_for_viz)))
        break  # Only need to calculate once, FPS should be similar for both videos
    
    for video_key, res in saccade_results.items():
        dir_map = get_direction_map_for_video(video_key, video1_eye, video2_eye)
        label_up = dir_map['upward']
        label_down = dir_map['downward']
        
        upward_saccades_df = res['upward_saccades_df']
        downward_saccades_df = res['downward_saccades_df']
        peri_saccades = res['peri_saccades']
        upward_segments = res['upward_segments']
        downward_segments = res['downward_segments']
        
        fig_all = make_subplots(
            rows=1, cols=4,
            shared_yaxes=False,  # Each panel can have different y-axis scale
            shared_xaxes=False,
            subplot_titles=(
                f'Position - {label_up} Saccades',
                f'Velocity - {label_up} Saccades',
                f'Position - {label_down} Saccades',
                f'Velocity - {label_down} Saccades'
            )
        )
        
        # Extract segments for each direction
        upward_segments_all = [seg for seg in peri_saccades if seg['saccade_direction'].iloc[0] == 'upward']
        downward_segments_all = [seg for seg in peri_saccades if seg['saccade_direction'].iloc[0] == 'downward']
        
        # Use all segments without filtering (outlier filtering removed per user request)
        upward_segments = upward_segments_all
        downward_segments = downward_segments_all
        
        if debug:
            print(f"Plotting {len(upward_segments)} {label_up} and {len(downward_segments)} {label_down} saccades...")
        
        # Plot upward saccades
        for i, segment in enumerate(upward_segments):
            color_opacity = 0.15 if len(upward_segments) > 20 else 0.3
            
            # Position trace (using baselined values)
            fig_all.add_trace(
                go.Scatter(
                    x=segment['Time_rel_threshold'],
                    y=segment['X_smooth_baselined'],
                    mode='lines',
                    name=f'Up #{i+1}',
                    line=dict(color='green', width=1),
                    showlegend=False,
                    opacity=color_opacity
                ),
                row=1, col=1
            )
            
            # Velocity trace
            fig_all.add_trace(
                go.Scatter(
                    x=segment['Time_rel_threshold'],
                    y=segment['vel_x_smooth'],
                    mode='lines',
                    name=f'Up #{i+1}',
                    line=dict(color='green', width=1),
                    showlegend=False,
                    opacity=color_opacity
                ),
                row=1, col=2
            )
        
        # Plot downward saccades
        for i, segment in enumerate(downward_segments):
            color_opacity = 0.15 if len(downward_segments) > 20 else 0.3
            
            # Position trace (using baselined values)
            fig_all.add_trace(
                go.Scatter(
                    x=segment['Time_rel_threshold'],
                    y=segment['X_smooth_baselined'],
                    mode='lines',
                    name=f'Down #{i+1}',
                    line=dict(color='purple', width=1),
                    showlegend=False,
                    opacity=color_opacity
                ),
                row=1, col=3
            )
            
            # Velocity trace
            fig_all.add_trace(
                go.Scatter(
                    x=segment['Time_rel_threshold'],
                    y=segment['vel_x_smooth'],
                    mode='lines',
                    name=f'Down #{i+1}',
                    line=dict(color='purple', width=1),
                    showlegend=False,
                    opacity=color_opacity
                ),
                row=1, col=4
            )
        
        # Add mean traces for reference
        if len(upward_segments) > 0:
            # Calculate mean for upward by aligning segments by Time_rel_threshold
            aligned_positions = []
            aligned_velocities = []
            aligned_times = []
            
            for seg in upward_segments:
                # Find index where Time_rel_threshold is closest to 0 (the threshold crossing)
                threshold_idx = np.abs(seg['Time_rel_threshold'].values).argmin()
                
                # Extract data centered on threshold crossing: n_before points before, threshold crossing, n_after points after
                start_idx = max(0, threshold_idx - n_before)
                end_idx = min(len(seg), threshold_idx + n_after + 1)
                
                # Extract aligned segment
                aligned_seg = seg.iloc[start_idx:end_idx].copy()
                
                # Find where the threshold crossing actually is within the extracted segment
                threshold_in_seg_idx = threshold_idx - start_idx
                
                # Ensure threshold crossing is at index n_before by padding at start if needed
                if threshold_in_seg_idx < n_before:
                    # Need to pad at the start to align threshold crossing to index n_before
                    pad_length = n_before - threshold_in_seg_idx
                    # Estimate time step from original segment for padding
                    if len(seg) > 1:
                        dt_est = np.diff(seg['Time_rel_threshold'].values).mean()
                    else:
                        dt_est = 0.0083  # default estimate if we can't calculate
                    
                    # Create padding with NaN values
                    pad_times = aligned_seg['Time_rel_threshold'].iloc[0] - dt_est * np.arange(pad_length, 0, -1)
                    pad_df = pd.DataFrame({
                        'X_smooth_baselined': [np.nan] * pad_length,
                        'vel_x_smooth': [np.nan] * pad_length,
                        'Time_rel_threshold': pad_times
                    })
                    aligned_seg = pd.concat([pad_df, aligned_seg.reset_index(drop=True)], ignore_index=True)
                
                aligned_positions.append(aligned_seg['X_smooth_baselined'].values)
                aligned_velocities.append(aligned_seg['vel_x_smooth'].values)
                aligned_times.append(aligned_seg['Time_rel_threshold'].values)
            
            # Find minimum length after alignment
            min_length = min(len(pos) for pos in aligned_positions)
            max_length = max(len(pos) for pos in aligned_positions)
            
            if min_length != max_length and debug:
                print(f"⚠️  Warning: {label_up} segments have variable lengths after alignment ({min_length} to {max_length} points). Using minimum length {min_length}.")
            
            # Truncate all segments to same length and stack
            upward_positions = np.array([pos[:min_length] for pos in aligned_positions])
            upward_velocities = np.array([vel[:min_length] for vel in aligned_velocities])
            upward_times = aligned_times[0][:min_length]  # Use first segment's time values
            
            # Calculate mean across all segments (axis=0 means across segments, keeping time dimension)
            # Use nanmean to handle NaN values properly (for segments that finish early)
            upward_mean_pos = np.nanmean(upward_positions, axis=0)
            upward_mean_vel = np.nanmean(upward_velocities, axis=0)
            
            fig_all.add_trace(
                go.Scatter(
                    x=upward_times,
                    y=upward_mean_pos,
                    mode='lines',
                    name=f'{label_up} Mean Position',
                    line=dict(color='red', width=3)
                ),
                row=1, col=1
            )
            
            fig_all.add_trace(
                go.Scatter(
                    x=upward_times,
                    y=upward_mean_vel,
                    mode='lines',
                    name=f'{label_up} Mean Velocity',
                    line=dict(color='red', width=3)
                ),
                row=1, col=2
            )
        
        if len(downward_segments) > 0:
            # Calculate mean for downward by taking mean across all segments at each index position
            # Find minimum length to handle variable-length segments
            min_length_down = min(len(seg) for seg in downward_segments)
            max_length_down = max(len(seg) for seg in downward_segments)
            
            if min_length_down != max_length_down and debug:
                print(f"⚠️  Warning: {label_down} segments have variable lengths ({min_length_down} to {max_length_down} points). Using minimum length {min_length_down}.")
            
            # Stack all segments as arrays (each row is one saccade, columns are time points)
            # Use only first min_length points to ensure all arrays have same shape
            downward_positions = np.array([seg['X_smooth_baselined'].values[:min_length_down] for seg in downward_segments])
            downward_velocities = np.array([seg['vel_x_smooth'].values[:min_length_down] for seg in downward_segments])
            downward_times = downward_segments[0]['Time_rel_threshold'].values[:min_length_down]  # Use first segment's time values
            
            # Calculate mean across all segments (axis=0 means across segments, keeping time dimension)
            # Use nanmean to handle NaN values properly (for segments that finish early)
            downward_mean_pos = np.nanmean(downward_positions, axis=0)
            downward_mean_vel = np.nanmean(downward_velocities, axis=0)
            
            fig_all.add_trace(
                go.Scatter(
                    x=downward_times,
                    y=downward_mean_pos,
                    mode='lines',
                    name=f'{label_down} Mean Position',
                    line=dict(color='red', width=4)
                ),
                row=1, col=3
            )
            
            fig_all.add_trace(
                go.Scatter(
                    x=downward_times,
                    y=downward_mean_vel,
                    mode='lines',
                    name=f'{label_down} Mean Velocity',
                    line=dict(color='red', width=4)
                ),
                row=1, col=4
            )
        
        # Add vertical line at time=0 (saccade onset)
        for col in [1, 2, 3, 4]:
            fig_all.add_shape(
                type="line",
                x0=0, x1=0,
                y0=-999, y1=999,
                line=dict(color='black', width=1, dash='dash'),
                row=1, col=col
            )
        
        # Update layout
        eye_label = get_eye_label(video_key, video_labels)
        fig_all.update_layout(
            title_text=f'All Detected Saccades Overlaid ({eye_label}) - Position and Velocity Profiles<br><sub>Position traces are baselined (avg of points -{n_before} to -{n_before-5}). All traces in semi-transparent, mean in bold. Time=0 is threshold crossing. {n_before} points before, {n_after} after.</sub>',
            height=500,
            width=1600,
            showlegend=False
        )
        
        # Collect all time values for proper x-axis scaling
        all_upward_times = []
        all_downward_times = []
        
        for seg in upward_segments:
            all_upward_times.extend(seg['Time_rel_threshold'].values)
        
        for seg in downward_segments:
            all_downward_times.extend(seg['Time_rel_threshold'].values)
        
        # Set x-axis ranges using actual min/max from all segment times (with small padding to show all data)
        padding_factor = 0.02  # 2% padding on each side for readability
        if len(all_upward_times) > 0:
            up_x_range = np.max(all_upward_times) - np.min(all_upward_times)
            padding = up_x_range * padding_factor if up_x_range > 0 else 0.01
            up_x_min = np.min(all_upward_times) - padding
            up_x_max = np.max(all_upward_times) + padding
        else:
            up_x_min, up_x_max = -0.2, 0.4
        
        if len(all_downward_times) > 0:
            down_x_range = np.max(all_downward_times) - np.min(all_downward_times)
            padding = down_x_range * padding_factor if down_x_range > 0 else 0.01
            down_x_min = np.min(all_downward_times) - padding
            down_x_max = np.max(all_downward_times) + padding
        else:
            down_x_min, down_x_max = -0.2, 0.4
        
        # Calculate y-axis ranges separately for position and velocity from filtered data
        # Collect position and velocity values from filtered segments
        upward_pos_values = []
        downward_pos_values = []
        upward_vel_values = []
        downward_vel_values = []
        
        for seg in upward_segments:
            # Filter out NaN values when collecting position data
            pos_values = seg['X_smooth_baselined'].values
            upward_pos_values.extend(pos_values[~np.isnan(pos_values)])
            # Filter out NaN values when collecting velocity data
            vel_values = seg['vel_x_smooth'].values
            upward_vel_values.extend(vel_values[~np.isnan(vel_values)])
        
        for seg in downward_segments:
            # Filter out NaN values when collecting position data
            pos_values = seg['X_smooth_baselined'].values
            downward_pos_values.extend(pos_values[~np.isnan(pos_values)])
            # Filter out NaN values when collecting velocity data
            vel_values = seg['vel_x_smooth'].values
            downward_vel_values.extend(vel_values[~np.isnan(vel_values)])
        
        # Position: find min/max for upward and downward, use wider range for both panels
        if len(upward_pos_values) > 0 and len(downward_pos_values) > 0:
            # Use actual min/max from all position values
            up_pos_min = np.min(upward_pos_values)
            up_pos_max = np.max(upward_pos_values)
            down_pos_min = np.min(downward_pos_values)
            down_pos_max = np.max(downward_pos_values)
            # Use the wider range (smaller min, larger max)
            pos_min = min(up_pos_min, down_pos_min)
            pos_max = max(up_pos_max, down_pos_max)
        elif len(upward_pos_values) > 0:
            pos_min = np.min(upward_pos_values)
            pos_max = np.max(upward_pos_values)
        elif len(downward_pos_values) > 0:
            pos_min = np.min(downward_pos_values)
            pos_max = np.max(downward_pos_values)
        else:
            pos_min, pos_max = -50, 50
        
        # Add padding to position range (20% padding on each side) to prevent clipping
        pos_range = pos_max - pos_min
        if pos_range > 0:
            padding = pos_range * 0.20  # 20% padding
            pos_min = pos_min - padding
            pos_max = pos_max + padding
        else:
            # If range is zero or very small, use default padding
            pos_min = pos_min - 10.0
            pos_max = pos_max + 10.0
        
        # Velocity: find min/max for upward and downward, use wider range for both panels in row 2
        # Get min and max directly from all velocity traces being plotted, with padding to prevent clipping
        if len(upward_vel_values) > 0 and len(downward_vel_values) > 0:
            # Get actual min/max from all velocity values
            all_vel_min = min(np.min(upward_vel_values), np.min(downward_vel_values))
            all_vel_max = max(np.max(upward_vel_values), np.max(downward_vel_values))
        elif len(upward_vel_values) > 0:
            all_vel_min = np.min(upward_vel_values)
            all_vel_max = np.max(upward_vel_values)
        elif len(downward_vel_values) > 0:
            all_vel_min = np.min(downward_vel_values)
            all_vel_max = np.max(downward_vel_values)
        else:
            all_vel_min, all_vel_max = -1000, 1000
            if debug:
                print(f"   ⚠️  No velocity values found, using default range: [{all_vel_min:.2f}, {all_vel_max:.2f}] px/s")
        
        # Add padding to prevent clipping (20% padding on each side)
        vel_range = all_vel_max - all_vel_min
        if vel_range > 0:
            padding = vel_range * 0.20  # 20% padding
            vel_min = all_vel_min - padding
            vel_max = all_vel_max + padding
        else:
            # If range is zero or very small, use default padding
            vel_min = all_vel_min - 1.0
            vel_max = all_vel_max + 1.0
        
        # Update axes - x-axis with ranges, y-axis with explicit ranges based on filtered data
        fig_all.update_xaxes(title_text="Time relative to threshold crossing (s)", range=[up_x_min, up_x_max], row=1, col=2)
        fig_all.update_xaxes(title_text="Time relative to threshold crossing (s)", range=[down_x_min, down_x_max], row=1, col=4)
        fig_all.update_xaxes(title_text="", range=[up_x_min, up_x_max], row=1, col=1)
        fig_all.update_xaxes(title_text="", range=[down_x_min, down_x_max], row=1, col=3)
        
        # Set explicit y-axis ranges - position panels share same range, velocity panels share same range
        fig_all.update_yaxes(title_text="X Position (px)", range=[pos_min, pos_max], row=1, col=1)
        fig_all.update_yaxes(title_text="X Position (px)", range=[pos_min, pos_max], row=1, col=3)
        fig_all.update_yaxes(title_text="Velocity (px/s)", range=[vel_min, vel_max], row=1, col=2)
        fig_all.update_yaxes(title_text="Velocity (px/s)", range=[vel_min, vel_max], row=1, col=4)
        
        if show_plot:
            fig_all.show()
        
        # Print statistics
        if debug:
            print(f"\n=== OVERLAY SUMMARY ===")
            if len(upward_segments) > 0:
                up_amps = [seg['saccade_amplitude'].iloc[0] for seg in upward_segments]
                up_durs = [seg['saccade_duration'].iloc[0] for seg in upward_segments]
                print(f"{label_up} saccades: {len(upward_segments)}")
                print(f"  Mean amplitude: {np.mean(up_amps):.2f} px")
                print(f"  Mean duration: {np.mean(up_durs):.3f} s")
            
            if len(downward_segments) > 0:
                down_amps = [seg['saccade_amplitude'].iloc[0] for seg in downward_segments]
                down_durs = [seg['saccade_duration'].iloc[0] for seg in downward_segments]
                print(f"{label_down} saccades: {len(downward_segments)}")
                print(f"  Mean amplitude: {np.mean(down_amps):.2f} px")
                print(f"  Mean duration: {np.mean(down_durs):.3f} s")
            
            print(f"\n⏱️  Time alignment: All saccades aligned to threshold crossing (Time_rel_threshold=0)")
            if len(all_upward_times) > 0 and len(all_downward_times) > 0:
                # Use the wider range for reporting
                overall_x_min = min(np.min(all_upward_times), np.min(all_downward_times))
                overall_x_max = max(np.max(all_upward_times), np.max(all_downward_times))
                print(f"📏 Window: {n_before} points before + {n_after} points after threshold crossing (actual time range: {overall_x_min:.3f} to {overall_x_max:.3f} s)")
            elif len(all_upward_times) > 0:
                print(f"📏 Window: {n_before} points before + {n_after} points after threshold crossing ({label_up} actual time range: {np.min(all_upward_times):.3f} to {np.max(all_upward_times):.3f} s)")
            elif len(all_downward_times) > 0:
                print(f"📏 Window: {n_before} points before + {n_after} points after threshold crossing ({label_down} actual time range: {np.min(all_downward_times):.3f} to {np.max(all_downward_times):.3f} s)")
            else:
                print(f"📏 Window: {n_before} points before + {n_after} points after threshold crossing")


def plot_saccade_amplitude_qc(
    saccade_results: Dict[str, Dict],
    video_labels: Dict[str, str],
    video1_eye: str,
    video2_eye: str,
    debug: bool = True,
    show_plot: bool = True
) -> None:
    """
    Create QC visualization for saccade amplitudes.
    
    Creates a figure with 3 rows x 2 columns showing:
    1. Distribution of saccade amplitudes (histograms)
    2. Correlation between saccade amplitude and duration (scatter plots)
    3. Peri-saccade segments colored by amplitude (outlier detection)
    
    Parameters
    ----------
    saccade_results : dict
        Dictionary containing saccade analysis results for each video
        Each entry should contain:
        - 'upward_saccades_df': DataFrame with upward saccades
        - 'downward_saccades_df': DataFrame with downward saccades
        - 'peri_saccades': List of DataFrames with peri-saccade segments
        - 'upward_segments': List of DataFrames with upward segments
        - 'downward_segments': List of DataFrames with downward segments
    video_labels : dict
        Dictionary mapping video keys to display labels
    video1_eye : str
        Eye label for VideoData1 ('L' or 'R')
    video2_eye : str
        Eye label for VideoData2 ('L' or 'R')
    debug : bool
        Whether to print debug information (default: True)
    show_plot : bool
        Whether to display the plot (default: True)
    """
    from sleap import saccade_processing as sp
    
    for video_key, res in saccade_results.items():
        dir_map = get_direction_map_for_video(video_key, video1_eye, video2_eye)
        label_up = dir_map['upward']
        label_down = dir_map['downward']
        
        upward_saccades_df = res['upward_saccades_df']
        downward_saccades_df = res['downward_saccades_df']
        peri_saccades = res['peri_saccades']
        upward_segments = res['upward_segments']
        downward_segments = res['downward_segments']
        
        fig_qc = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                f'Amplitude Distribution - {label_up} Saccades', 
                f'Amplitude Distribution - {label_down} Saccades',
                f'Amplitude vs Duration - {label_up} Saccades',
                f'Amplitude vs Duration - {label_down} Saccades',
                f'Peri-Saccade Segments - {label_up} (colored by amplitude)',
                f'Peri-Saccade Segments - {label_down} (colored by amplitude)'
            ),
            vertical_spacing=0.10,
            horizontal_spacing=0.1,
            row_heights=[0.25, 0.25, 0.5]  # Make segment plots larger
        )
        
        # 1. Amplitude distributions
        if len(upward_saccades_df) > 0:
            # Histogram for upward saccades
            fig_qc.add_trace(
                go.Histogram(
                    x=upward_saccades_df['amplitude'],
                    nbinsx=50,
                    name=f'{label_up}',
                    marker_color='red',
                    opacity=0.6
                ),
                row=1, col=1
            )
            
            # Scatter plot for upward saccades
            fig_qc.add_trace(
                go.Scatter(
                    x=upward_saccades_df['duration'],
                    y=upward_saccades_df['amplitude'],
                    mode='markers',
                    name=f'{label_up}',
                    marker=dict(color='red', size=6, opacity=0.6)
                ),
                row=2, col=1
            )
            
            # Add correlation line for upward saccades
            corr_up = upward_saccades_df[['amplitude', 'duration']].corr().iloc[0, 1]
            z_up = np.polyfit(upward_saccades_df['duration'], upward_saccades_df['amplitude'], 1)
            p_up = np.poly1d(z_up)
            fig_qc.add_trace(
                go.Scatter(
                    x=upward_saccades_df['duration'],
                    y=p_up(upward_saccades_df['duration']),
                    mode='lines',
                    name=f'R={corr_up:.2f}',
                    line=dict(color='darkgreen', width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        if len(downward_saccades_df) > 0:
            # Histogram for downward saccades
            fig_qc.add_trace(
                go.Histogram(
                    x=downward_saccades_df['amplitude'],
                    nbinsx=50,
                    name=f'{label_down}',
                    marker_color='purple',
                    opacity=0.6
                ),
                row=1, col=2
            )
            
            # Scatter plot for downward saccades
            fig_qc.add_trace(
                go.Scatter(
                    x=downward_saccades_df['duration'],
                    y=downward_saccades_df['amplitude'],
                    mode='markers',
                    name=f'{label_down}',
                    marker=dict(color='purple', size=6, opacity=0.6)
                ),
                row=2, col=2
            )
            
            # Add correlation line for downward saccades
            corr_down = downward_saccades_df[['amplitude', 'duration']].corr().iloc[0, 1]
            z_down = np.polyfit(downward_saccades_df['duration'], downward_saccades_df['amplitude'], 1)
            p_down = np.poly1d(z_down)
            fig_qc.add_trace(
                go.Scatter(
                    x=downward_saccades_df['duration'],
                    y=p_down(downward_saccades_df['duration']),
                    mode='lines',
                    name=f'R={corr_down:.2f}',
                    line=dict(color='darkviolet', width=2),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # 3. Plot peri-saccade segments colored by amplitude
        # Extract upward and downward segments for QC visualization from already-baselined peri_saccades
        upward_segments_all = [seg for seg in peri_saccades if seg['saccade_direction'].iloc[0] == 'upward']
        downward_segments_all = [seg for seg in peri_saccades if seg['saccade_direction'].iloc[0] == 'downward']
        
        # Plot upward segments
        if len(upward_segments_all) > 0:
            upward_amplitudes = [seg['saccade_amplitude'].iloc[0] for seg in upward_segments_all]
            upward_colors, upward_min_amp, upward_max_amp = sp.get_color_mapping(upward_amplitudes)
            
            for i, (segment, color) in enumerate(zip(upward_segments_all, upward_colors)):
                fig_qc.add_trace(
                    go.Scatter(
                        x=segment['Time_rel_threshold'],
                        y=segment['X_smooth_baselined'],
                        mode='lines',
                        name=f'{label_up} #{i+1}',
                        line=dict(color=color, width=1.5),
                        showlegend=False,
                        opacity=0.7,
                        hovertemplate=f'Amplitude: {segment["saccade_amplitude"].iloc[0]:.2f} px<br>' +
                                    'Time: %{x:.3f} s<br>' +
                                    'Position: %{y:.2f} px<extra></extra>'
                    ),
                    row=3, col=1
                )
            
            # Add dummy trace for colorbar (hidden but provides colorbar)
            fig_qc.add_trace(
                go.Scatter(
                    x=[None],  # Hidden trace
                    y=[None],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=[upward_min_amp, upward_max_amp],
                        colorscale='Plasma',
                        cmin=upward_min_amp,
                        cmax=upward_max_amp,
                        showscale=True,
                        colorbar=dict(
                            title=dict(text=f"Amplitude ({label_up})", side="right"),
                            x=0.47,  # Position to the right of the left column plot
                            xpad=10,
                            len=0.45,  # Set colorbar length relative to subplot
                            y=0.5,  # Center vertically on the subplot
                            yanchor="middle"
                        )
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=3, col=1
            )
        
        # Plot downward segments
        if len(downward_segments_all) > 0:
            downward_amplitudes = [seg['saccade_amplitude'].iloc[0] for seg in downward_segments_all]
            downward_colors, downward_min_amp, downward_max_amp = sp.get_color_mapping(downward_amplitudes)
            
            for i, (segment, color) in enumerate(zip(downward_segments_all, downward_colors)):
                fig_qc.add_trace(
                    go.Scatter(
                        x=segment['Time_rel_threshold'],
                        y=segment['X_smooth_baselined'],
                        mode='lines',
                        name=f'{label_down} #{i+1}',
                        line=dict(color=color, width=1.5),
                        showlegend=False,
                        opacity=0.7,
                        hovertemplate=f'Amplitude: {segment["saccade_amplitude"].iloc[0]:.2f} px<br>' +
                                    'Time: %{x:.3f} s<br>' +
                                    'Position: %{y:.2f} px<extra></extra>'
                    ),
                    row=3, col=2
                )
            
            # Add dummy trace for colorbar (hidden but provides colorbar)
            fig_qc.add_trace(
                go.Scatter(
                    x=[None],  # Hidden trace
                    y=[None],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=[downward_min_amp, downward_max_amp],
                        colorscale='Plasma',
                        cmin=downward_min_amp,
                        cmax=downward_max_amp,
                        showscale=True,
                        colorbar=dict(
                            title=dict(text=f"Amplitude ({label_down})", side="right"),
                            x=0.97,  # Position to the right of the right column plot
                            xpad=10,
                            len=0.45,  # Set colorbar length relative to subplot
                            y=0.5,  # Center vertically on the subplot
                            yanchor="middle"
                        )
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=3, col=2
            )
        
        # Update layout
        eye_label = get_eye_label(video_key, video_labels)
        fig_qc.update_layout(
            title_text=f'Saccade Amplitude QC: Distributions, Correlations, and Segments (Outlier Detection, {eye_label})',
            height=1200,
            showlegend=False
        )
        
        # Update axes labels
        fig_qc.update_xaxes(title_text="Amplitude (px)", row=1, col=1)
        fig_qc.update_xaxes(title_text="Amplitude (px)", row=1, col=2)
        fig_qc.update_xaxes(title_text="Duration (s)", row=2, col=1)
        fig_qc.update_xaxes(title_text="Duration (s)", row=2, col=2)
        fig_qc.update_xaxes(title_text="Time relative to threshold crossing (s)", row=3, col=1)
        fig_qc.update_xaxes(title_text="Time relative to threshold crossing (s)", row=3, col=2)
        fig_qc.update_yaxes(title_text="Count", row=1, col=1)
        fig_qc.update_yaxes(title_text="Count", row=1, col=2)
        fig_qc.update_yaxes(title_text="Amplitude (px)", row=2, col=1)
        fig_qc.update_yaxes(title_text="Amplitude (px)", row=2, col=2)
        fig_qc.update_yaxes(title_text="Position (baselined, px)", row=3, col=1)
        fig_qc.update_yaxes(title_text="Position (baselined, px)", row=3, col=2)
        
        if show_plot:
            fig_qc.show()
        
        # Print correlation statistics
        if debug:
            print("\n=== SACCADE AMPLITUDE-DURATION CORRELATION ===\n")
            if upward_saccades_df is not None and len(upward_saccades_df) > 0:
                eye_label = get_eye_label(video_key, video_labels)
                print(f"{eye_label} saccades (n={len(upward_saccades_df)}):")
                print(f"  Correlation (amplitude vs duration): {corr_up:.3f}")
                print(f"  Mean amplitude: {upward_saccades_df['amplitude'].mean():.2f} px")
                print(f"  Mean duration: {upward_saccades_df['duration'].mean():.3f} s")
                print(f"  Amp range: {upward_saccades_df['amplitude'].min():.2f} - {upward_saccades_df['amplitude'].max():.2f} px")
            
            if downward_saccades_df is not None and len(downward_saccades_df) > 0:
                eye_label = get_eye_label(video_key, video_labels)
                print(f"\n{eye_label} saccades (n={len(downward_saccades_df)}):")
                print(f"  Correlation (amplitude vs duration): {corr_down:.3f}")
                print(f"  Mean amplitude: {downward_saccades_df['amplitude'].mean():.2f} px")
                print(f"  Mean duration: {downward_saccades_df['duration'].mean():.3f} s")
                print(f"  Amp range: {downward_saccades_df['amplitude'].min():.2f} - {downward_saccades_df['amplitude'].max():.2f} px")

