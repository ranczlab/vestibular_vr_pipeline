# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: aeon2
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Multi-Cohort Running and Turning Behavior Analysis
#
# This notebook analyzes multiple cohort behavioral analysis CSV files (merged together) and generates summary plots showing:
# - Running velocity (cm/s) with average and SEM per experiment day
# - Time spent running (percentage) per experiment day
# - Turning velocity (deg/s) with average and SEM per experiment day
# - Total turn distance per experiment day **normalized by session duration (deg/minute)**
# - Time spent turning (percentage) per experiment day
#
# Individual mouse values are shown with consistent colors across days and cohorts. Analysis is performed across all cohorts grouped by experiment day type.
#
# **Note:** Distance metrics are normalized by total recording time to account for differences in session duration across experiment days. This ensures fair comparison between training days and experiment days with different recording lengths.
#

# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


# %%
# Configuration
#----------------------------
# Paths to cohort CSV files (can specify multiple cohorts)
# CSV files should be located 2 levels above _processedData folder
cohort_csv_paths = [
    Path('/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20250409_Cohort3_rotation/cohort_behavioral_analysis.csv').expanduser(),
    # Path('/Users/nora/Desktop/for_poster/cohort_3/cohort_behavioral_analysis.csv').expanduser(),
    # Add more cohort CSV paths here as needed
    # Path('/path/to/cohort2/cohort_behavioral_analysis.csv').expanduser(),
]

# Analysis options
EXCLUDE_VISUAL_MISMATCH = True  # Set to True to exclude visual mismatch experiments from main analysis (they will be analyzed separately)

# Output directory for plots (use first cohort's parent directory or specify a common location)
if len(cohort_csv_paths) > 0:
    # Use parent directory of first cohort
    base_output_dir = cohort_csv_paths[0].parent
else:
    base_output_dir = Path('.')

# Create subfolder for all outputs
output_dir = base_output_dir / 'cohort_behavioral_analysis_outputs'
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Number of cohort CSV files: {len(cohort_csv_paths)}")
for i, path in enumerate(cohort_csv_paths, 1):
    print(f"  Cohort {i}: {path}")
print(f"\nOutput directory: {output_dir}")
print(f"\nExclude visual mismatch from main analysis: {EXCLUDE_VISUAL_MISMATCH}")


# %%
# Load and merge multiple cohort CSV files
#----------------------------
dfs = []
cohort_names = []

for i, csv_path in enumerate(cohort_csv_paths, 1):
    if not csv_path.exists():
        print(f"⚠️ Warning: Cohort CSV not found at {csv_path}, skipping...")
        continue
    
    # Load CSV
    df_temp = pd.read_csv(csv_path)
    
    # Extract cohort directory name from path (e.g., "20250409_Cohort3_rotation" or "Cohort1_rotation")
    # Use the parent directory name as the cohort identifier
    cohort_name = csv_path.parent.name
    
    # If parent is just the filename or empty, try grandparent
    if not cohort_name or cohort_name == csv_path.parent:
        cohort_name = csv_path.parent.parent.name
    
    # If still not found, use a default name
    if not cohort_name or cohort_name == '.':
        cohort_name = f"Cohort{i}"
    
    # Add cohort identifier column
    df_temp['Cohort'] = cohort_name
    
    dfs.append(df_temp)
    cohort_names.append(cohort_name)
    
    print(f"✅ Loaded {cohort_name} CSV: {len(df_temp)} rows from {csv_path.name}")

# Merge all dataframes
if len(dfs) == 0:
    raise ValueError("No valid cohort CSV files found!")

df = pd.concat(dfs, ignore_index=True)
print(f"\n✅ Merged {len(dfs)} cohort CSV files")
print(f"   Total rows: {len(df)}")
print(f"   Cohorts: {', '.join(cohort_names)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nExperiment days: {sorted(df['Experiment_Day'].unique())}")
print(f"\nNumber of animals per cohort:")
print(df.groupby('Cohort')['Animal_ID'].nunique())
print(f"\nTotal unique animals: {df['Animal_ID'].nunique()}")


# %%
# Data preparation and conversion
#----------------------------
# Convert running velocity from m/s to cm/s (multiply by 100)
# Note: These columns may not exist if all data is visual mismatch (block-level only)
if 'running_velocity_avg_m_per_s' in df.columns:
    df['running_velocity_avg_cm_per_s'] = df['running_velocity_avg_m_per_s'] * 100
elif 'Visual_mismatch' not in str(df['Experiment_Day'].unique()):
    print("⚠️ Warning: 'running_velocity_avg_m_per_s' column not found")

if 'running_velocity_sd_m_per_s' in df.columns:
    df['running_velocity_sd_cm_per_s'] = df['running_velocity_sd_m_per_s'] * 100
elif 'Visual_mismatch' not in str(df['Experiment_Day'].unique()):
    print("⚠️ Warning: 'running_velocity_sd_m_per_s' column not found")

# Turning velocity is in degrees/s (column name updated to reflect this)
if 'turning_velocity_avg_deg_per_s' not in df.columns:
    # Check for legacy column name and convert if found
    if 'turning_velocity_avg_m_per_s' in df.columns:
        print("⚠️ Warning: Legacy column name 'turning_velocity_avg_m_per_s' found. Values are in deg/s. Please regenerate cohort CSV with updated column names.")
        df['turning_velocity_avg_deg_per_s'] = df['turning_velocity_avg_m_per_s']
    elif 'Visual_mismatch' not in str(df['Experiment_Day'].unique()):
        print("⚠️ Warning: 'turning_velocity_avg_deg_per_s' column not found")

# Normalize cumulative metrics by total recording time (block duration)
# This accounts for different session durations across experiment days
# Use running_block_duration_seconds or turning_block_duration_seconds (they should be the same)
if 'running_block_duration_seconds' in df.columns:
    total_time_col = 'running_block_duration_seconds'
elif 'turning_block_duration_seconds' in df.columns:
    total_time_col = 'turning_block_duration_seconds'
else:
    total_time_col = None
    # Only show warning if not all visual mismatch (which has block-level times)
    if 'Visual_mismatch' not in str(df['Experiment_Day'].unique()):
        print("⚠️ Warning: No block duration column found. Cannot normalize distances.")

if total_time_col is not None:
    # Convert seconds to minutes for normalization
    df['total_time_minutes'] = df[total_time_col] / 60.0
    
    # Normalize running distance: m per minute
    if 'running_distance_travelled_m' in df.columns:
        df['running_distance_travelled_m_per_minute'] = df['running_distance_travelled_m'] / df['total_time_minutes']
        print("✅ Created normalized running distance (m/minute)")
    else:
        print("⚠️ Warning: 'running_distance_travelled_m' column not found")
    
    # Normalize turning distance: degrees per minute (values already stored in degrees)
    turning_distance_col = None
    if 'turning_distance_turned_deg' in df.columns:
        turning_distance_col = 'turning_distance_turned_deg'
    elif 'turning_distance_turned_m' in df.columns:
        turning_distance_col = 'turning_distance_turned_m'
        print("⚠️ Detected legacy column 'turning_distance_turned_m'; treating values as degrees. Please regenerate cohort CSVs to use '_deg'.")

    if turning_distance_col is not None:
        df['turning_distance_turned_deg_per_minute'] = df[turning_distance_col] / df['total_time_minutes']
        print("✅ Created normalized turning distance (deg/minute)")
    else:
        print("⚠️ Warning: turning distance column not found")
    
    # Display summary of session durations
    print(f"\n📊 Session duration summary (minutes):")
    print(f"   Mean: {df['total_time_minutes'].mean():.2f} minutes")
    print(f"   Min: {df['total_time_minutes'].min():.2f} minutes")
    print(f"   Max: {df['total_time_minutes'].max():.2f} minutes")
    print(f"\n   By experiment day:")
    for day in sorted(df['Experiment_Day'].unique()):
        day_times = df[df['Experiment_Day'] == day]['total_time_minutes']
        print(f"   {day}: {day_times.mean():.2f} ± {day_times.std():.2f} minutes (n={len(day_times)})")
else:
    # Create dummy normalized columns if normalization not possible
    df['running_distance_travelled_m_per_minute'] = df.get('running_distance_travelled_m', np.nan)
    df['turning_distance_turned_deg_per_minute'] = df.get('turning_distance_turned_deg', df.get('turning_distance_turned_m', np.nan))
    # Only show warning if not all visual mismatch (which has block-level normalization)
    if 'Visual_mismatch' not in str(df['Experiment_Day'].unique()):
        print("⚠️ Using non-normalized distances (no time normalization available)")

# Separate visual mismatch experiments if configured
#----------------------------
if EXCLUDE_VISUAL_MISMATCH:
    # Identify visual mismatch experiment days
    visual_mismatch_days = [day for day in df['Experiment_Day'].unique() if 'visual_mismatch' in day.lower() or 'Visual_mismatch' in day]
    
    if visual_mismatch_days:
        print(f"\n📊 Visual mismatch experiment days detected: {visual_mismatch_days}")
        df_visual_mismatch = df[df['Experiment_Day'].isin(visual_mismatch_days)].copy()
        df_main = df[~df['Experiment_Day'].isin(visual_mismatch_days)].copy()
        print(f"   Main analysis dataset: {len(df_main)} rows (excluding visual mismatch)")
        print(f"   Visual mismatch dataset: {len(df_visual_mismatch)} rows (will be analyzed separately)")
    else:
        print(f"\n⚠️ No visual mismatch experiment days found in data")
        df_visual_mismatch = pd.DataFrame()
        df_main = df.copy()
else:
    df_main = df.copy()
    df_visual_mismatch = pd.DataFrame()
    print(f"\n⚠️ Visual mismatch exclusion is disabled - all experiments will be included in main analysis")

# Check required columns (only if df_main is not empty)
if len(df_main) > 0:
    required_cols = ['Animal_ID', 'Experiment_Day', 'running_velocity_avg_cm_per_s', 
                     'running_distance_travelled_m_per_minute', 'running_time_percentage',
                     'turning_velocity_avg_deg_per_s', 'turning_distance_turned_deg_per_minute', 
                     'turning_time_percentage']
    missing_cols = [col for col in required_cols if col not in df_main.columns]

    if missing_cols:
        print(f"⚠️ Warning: Missing columns in df_main: {missing_cols}")
        print(f"Available columns: {list(df_main.columns)}")
    else:
        print("\n✅ All required columns found (including normalized distances)")
else:
    print("\n⚠️ df_main is empty - all experiments are visual mismatch (will be analyzed separately)")


# %%
# Calculate summary statistics per experiment day
#----------------------------
def calculate_cohort_stats(df: pd.DataFrame, experiment_day: str) -> Dict:
    """Calculate mean and SEM for all mice on a given experiment day."""
    day_data = df[df['Experiment_Day'] == experiment_day].copy()
    
    if len(day_data) == 0:
        return None
    
    n = len(day_data)
    
    # Calculate SEM = SD / sqrt(n)
    def sem(x):
        return x.std() / np.sqrt(n) if n > 1 else 0
    
    stats = {
        'experiment_day': experiment_day,
        'n_mice': n,
        # Running stats
        'running_velocity_avg_cm_per_s_mean': day_data['running_velocity_avg_cm_per_s'].mean(),
        'running_velocity_avg_cm_per_s_sem': sem(day_data['running_velocity_avg_cm_per_s']),
        'running_distance_travelled_m_per_minute_mean': day_data['running_distance_travelled_m_per_minute'].mean(),
        'running_distance_travelled_m_per_minute_sem': sem(day_data['running_distance_travelled_m_per_minute']),
        'running_time_percentage_mean': day_data['running_time_percentage'].mean(),
        'running_time_percentage_sem': sem(day_data['running_time_percentage']),
        # Turning stats
        'turning_velocity_avg_deg_per_s_mean': day_data['turning_velocity_avg_deg_per_s'].mean(),
        'turning_velocity_avg_deg_per_s_sem': sem(day_data['turning_velocity_avg_deg_per_s']),
        'turning_distance_turned_deg_per_minute_mean': day_data['turning_distance_turned_deg_per_minute'].mean(),
        'turning_distance_turned_deg_per_minute_sem': sem(day_data['turning_distance_turned_deg_per_minute']),
        'turning_time_percentage_mean': day_data['turning_time_percentage'].mean(),
        'turning_time_percentage_sem': sem(day_data['turning_time_percentage'])
    }
    
    return stats

# Calculate stats for all experiment days (excluding visual mismatch if configured)
if len(df_main) > 0:
    experiment_days = sorted(df_main['Experiment_Day'].unique())
    cohort_stats = []

    for day in experiment_days:
        stats = calculate_cohort_stats(df_main, day)
        if stats:
            cohort_stats.append(stats)

    cohort_stats_df = pd.DataFrame(cohort_stats)
    print("\nCohort Summary Statistics (Mean ± SEM) - Normalized by session duration:")
    print(cohort_stats_df.to_string(index=False))
else:
    cohort_stats_df = pd.DataFrame()
    print("\n⚠️ No non-visual mismatch experiments to calculate cohort stats")


# %%
# Assign consistent colors to each mouse using gnuplot2 palette
#----------------------------
def assign_mouse_colors(df: pd.DataFrame) -> Dict[str, str]:
    """Assign a consistent color to each mouse across all days using gnuplot2 palette."""
    unique_mice = sorted(df['Animal_ID'].unique())
    n_colors = len(unique_mice)
    
    # Use gnuplot2 color palette (similar to gnuplot's default palette)
    # gnuplot2 is a rainbow-like palette going from blue to red
    # Avoid the white end (value 1.0) by using 0 to 0.95 instead of 0 to 1
    # This ensures all colors are visible and distinct
    colors = plt.cm.gnuplot2(np.linspace(0, 0.95, n_colors))
    
    mouse_colors = {mouse: colors[i] for i, mouse in enumerate(unique_mice)}
    return mouse_colors

# Assign mouse colors from full dataset (not just df_main)
# This ensures colors are available for visual mismatch and eye metrics analysis
if len(df) > 0:
    mouse_colors = assign_mouse_colors(df)
    print(f"✅ Assigned colors to {len(mouse_colors)} mice using gnuplot2 palette")
    print(f"\nMouse colors:")
    for mouse, color in list(mouse_colors.items())[:10]:  # Show first 10
        print(f"  {mouse}: {color}")
else:
    mouse_colors = {}
    print(f"⚠️ No mice in dataset to assign colors")


# %%
# Create comprehensive plot
#----------------------------
def plot_cohort_behavioral_analysis(df: pd.DataFrame, cohort_stats_df: pd.DataFrame, 
                                     mouse_colors: Dict[str, str], output_path: Path):
    """Create a comprehensive plot showing all running and turning metrics with individual mice and cohort averages."""
    
    experiment_days = sorted(df['Experiment_Day'].unique())
    n_days = len(experiment_days)
    
    # Create figure with 2 rows and 3 columns (6 subplots total)
    # Total height should be 19 cm including x-axis labels
    # Convert cm to inches: 19 cm = 7.48 inches
    fig, axes = plt.subplots(2, 3, figsize=(17,9))
    
    x_pos = np.arange(n_days)
    
    # Helper function to plot a single metric
    def plot_metric(ax, means, sems, individual_data_dict, title, ylabel):
        """Helper function to plot a metric with individual mice and mean ± SEM."""
        # Plot individual mice
        for mouse, mouse_values in individual_data_dict.items():
            color = mouse_colors[mouse]
            ax.plot(x_pos, mouse_values, 'o-', color=color, alpha=0.7, 
                    linewidth=1.5, markersize=5, label=mouse)
        
        # Plot mean line (black)
        ax.plot(x_pos, means, '-', color='black', linewidth=2.5, zorder=10)
        
        # Plot SEM as opaque grey fill
        ax.fill_between(x_pos, means - sems, means + sems, 
                       color='grey', alpha=0.5, zorder=9)
        
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=15, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(experiment_days, rotation=20, ha='right', fontsize=15)
        # Remove grid
        ax.grid(False)
    
    # Collect individual mouse data for each metric
    
    # ========== ROW 1: RUNNING METRICS ==========
    
    # Plot 1: Running Velocity (cm/s)
    ax1 = axes[0, 0]
    running_velocities_dict = {}
    for mouse in df['Animal_ID'].unique():
        mouse_data = df[df['Animal_ID'] == mouse]
        mouse_velocities = []
        for day in experiment_days:
            day_mouse_data = mouse_data[mouse_data['Experiment_Day'] == day]
            if len(day_mouse_data) > 0:
                mouse_velocities.append(day_mouse_data['running_velocity_avg_cm_per_s'].values[0])
            else:
                mouse_velocities.append(np.nan)
        running_velocities_dict[mouse] = mouse_velocities
    
    means = cohort_stats_df['running_velocity_avg_cm_per_s_mean'].values
    sems = cohort_stats_df['running_velocity_avg_cm_per_s_sem'].values
    plot_metric(ax1, means, sems, running_velocities_dict, 
               'Running Velocity per Experiment Day', 'Running Velocity (cm/s)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    
    # Plot 2: Total Run Distance (m/minute) - Normalized by session duration
    ax2 = axes[0, 1]
    running_distances_dict = {}
    for mouse in df['Animal_ID'].unique():
        mouse_data = df[df['Animal_ID'] == mouse]
        mouse_distances = []
        for day in experiment_days:
            day_mouse_data = mouse_data[mouse_data['Experiment_Day'] == day]
            if len(day_mouse_data) > 0:
                mouse_distances.append(day_mouse_data['running_distance_travelled_m_per_minute'].values[0])
            else:
                mouse_distances.append(np.nan)
        running_distances_dict[mouse] = mouse_distances
    
    means = cohort_stats_df['running_distance_travelled_m_per_minute_mean'].values
    sems = cohort_stats_df['running_distance_travelled_m_per_minute_sem'].values
    plot_metric(ax2, means, sems, running_distances_dict,
               'Total Run Distance per Experiment Day (Normalized)', 'Run Distance (m/min)')
    
    # Plot 3: Time Spent Running (Percentage)
    ax3 = axes[0, 2]
    running_percentages_dict = {}
    for mouse in df['Animal_ID'].unique():
        mouse_data = df[df['Animal_ID'] == mouse]
        mouse_percentages = []
        for day in experiment_days:
            day_mouse_data = mouse_data[mouse_data['Experiment_Day'] == day]
            if len(day_mouse_data) > 0:
                mouse_percentages.append(day_mouse_data['running_time_percentage'].values[0])
            else:
                mouse_percentages.append(np.nan)
        running_percentages_dict[mouse] = mouse_percentages
    
    means = cohort_stats_df['running_time_percentage_mean'].values
    sems = cohort_stats_df['running_time_percentage_sem'].values
    plot_metric(ax3, means, sems, running_percentages_dict,
               'Time Spent Running per Experiment Day', 'Time Spent Running (%)')
    
    # ========== ROW 2: TURNING METRICS ==========
    
    # Plot 4: Turning Velocity (deg/s)
    ax4 = axes[1, 0]
    turning_velocities_dict = {}
    for mouse in df['Animal_ID'].unique():
        mouse_data = df[df['Animal_ID'] == mouse]
        mouse_velocities = []
        for day in experiment_days:
            day_mouse_data = mouse_data[mouse_data['Experiment_Day'] == day]
            if len(day_mouse_data) > 0:
                mouse_velocities.append(day_mouse_data['turning_velocity_avg_deg_per_s'].values[0])
            else:
                mouse_velocities.append(np.nan)
        turning_velocities_dict[mouse] = mouse_velocities
    
    means = cohort_stats_df['turning_velocity_avg_deg_per_s_mean'].values
    sems = cohort_stats_df['turning_velocity_avg_deg_per_s_sem'].values
    plot_metric(ax4, means, sems, turning_velocities_dict,
               'Turning Velocity per Experiment Day', 'Turning Velocity (deg/s)')
    
    # Plot 5: Total Turn Distance (deg/minute) - Normalized by session duration
    ax5 = axes[1, 1]
    turning_distances_dict = {}
    for mouse in df['Animal_ID'].unique():
        mouse_data = df[df['Animal_ID'] == mouse]
        mouse_distances = []
        for day in experiment_days:
            day_mouse_data = mouse_data[mouse_data['Experiment_Day'] == day]
            if len(day_mouse_data) > 0:
                mouse_distances.append(day_mouse_data['turning_distance_turned_deg_per_minute'].values[0])
            else:
                mouse_distances.append(np.nan)
        turning_distances_dict[mouse] = mouse_distances
    
    means = cohort_stats_df['turning_distance_turned_deg_per_minute_mean'].values
    sems = cohort_stats_df['turning_distance_turned_deg_per_minute_sem'].values
    plot_metric(ax5, means, sems, turning_distances_dict,
               'Total Turn Distance per Experiment Day (Normalized)', 'Turn Distance (deg/min)')
    
    # Plot 6: Time Spent Turning (Percentage)
    ax6 = axes[1, 2]
    turning_percentages_dict = {}
    for mouse in df['Animal_ID'].unique():
        mouse_data = df[df['Animal_ID'] == mouse]
        mouse_percentages = []
        for day in experiment_days:
            day_mouse_data = mouse_data[mouse_data['Experiment_Day'] == day]
            if len(day_mouse_data) > 0:
                mouse_percentages.append(day_mouse_data['turning_time_percentage'].values[0])
            else:
                mouse_percentages.append(np.nan)
        turning_percentages_dict[mouse] = mouse_percentages
    
    means = cohort_stats_df['turning_time_percentage_mean'].values
    sems = cohort_stats_df['turning_time_percentage_sem'].values
    plot_metric(ax6, means, sems, turning_percentages_dict,
               'Time Spent Turning per Experiment Day', 'Time Spent Turning (%)')
    
    # Only add xlabel to bottom row plots
    for ax in [ax4, ax5, ax6]:
        ax.set_xlabel('')  # Remove xlabel as requested
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"✅ Saved plot to: {output_path}")
    plt.close(fig)

# Create the plot (using main dataframe, excluding visual mismatch if configured)
# Use simplified filename based on cohort directory names
if len(df_main) > 0 and len(cohort_stats_df) > 0:
    cohort_str = "_".join(cohort_names)
    plot_path = output_dir / f"{cohort_str}_averages.svg"
    plot_cohort_behavioral_analysis(df_main, cohort_stats_df, mouse_colors, plot_path)
else:
    print("⚠️ Skipping main plot generation (no non-visual mismatch data available)")


# %%
# Save summary statistics to CSV
#----------------------------
if len(cohort_stats_df) > 0:
    # Use simplified filename based on cohort directory names
    cohort_str = "_".join(cohort_names)
    summary_csv_path = output_dir / f"{cohort_str}_averages.csv"
    cohort_stats_df.to_csv(summary_csv_path, index=False)
    print(f"✅ Saved summary statistics to: {summary_csv_path}")

    # Display the summary
    print("\nSummary Statistics (Mean ± SEM) - Combined across all cohorts:")
    print(cohort_stats_df.to_string(index=False))

    # Also show breakdown by cohort if multiple cohorts
    if len(cohort_names) > 1 and len(df_main) > 0:
        print(f"\n\nBreakdown by Cohort and Experiment Day:")
        for cohort in cohort_names:
            cohort_data = df_main[df_main['Cohort'] == cohort]
            print(f"\n{cohort}:")
            print(f"  Number of animals: {cohort_data['Animal_ID'].nunique()}")
            print(f"  Experiment days: {sorted(cohort_data['Experiment_Day'].unique())}")
else:
    print("⚠️ No summary statistics to save (no non-visual mismatch data)")


# %%
# Visual Mismatch Analysis: First Block vs Last Block Comparison
#----------------------------
if EXCLUDE_VISUAL_MISMATCH and len(df_visual_mismatch) > 0:
    print("\n" + "="*80)
    print("VISUAL MISMATCH ANALYSIS: First Block vs Last Block Comparison")
    print("="*80)
    
    # Prepare first_block and last_block data
    # Convert velocities from m/s to cm/s for first_block
    if 'first_block_running_velocity_avg_m_per_s' in df_visual_mismatch.columns:
        df_visual_mismatch['first_block_running_velocity_avg_cm_per_s'] = df_visual_mismatch['first_block_running_velocity_avg_m_per_s'] * 100
    
    # Convert last_block running velocity from m/s to cm/s
    if 'last_block_running_velocity_avg_m_per_s' in df_visual_mismatch.columns:
        df_visual_mismatch['last_block_running_velocity_avg_cm_per_s'] = df_visual_mismatch['last_block_running_velocity_avg_m_per_s'] * 100
    
    # Prepare turning velocities (should already be in deg/s with updated column names)
    # Check for legacy column names and convert if needed
    if 'first_block_turning_velocity_avg_deg_per_s' not in df_visual_mismatch.columns:
        if 'first_block_turning_velocity_avg_m_per_s' in df_visual_mismatch.columns:
            print("⚠️ Warning: Legacy column 'first_block_turning_velocity_avg_m_per_s' found. Values are in deg/s. Please regenerate cohort CSV.")
            df_visual_mismatch['first_block_turning_velocity_avg_deg_per_s'] = df_visual_mismatch['first_block_turning_velocity_avg_m_per_s']
    
    if 'last_block_turning_velocity_avg_deg_per_s' not in df_visual_mismatch.columns:
        if 'last_block_turning_velocity_avg_m_per_s' in df_visual_mismatch.columns:
            print("⚠️ Warning: Legacy column 'last_block_turning_velocity_avg_m_per_s' found. Values are in deg/s. Please regenerate cohort CSV.")
            df_visual_mismatch['last_block_turning_velocity_avg_deg_per_s'] = df_visual_mismatch['last_block_turning_velocity_avg_m_per_s']
    
    # Normalize distances by block time for first_block
    if 'first_block_running_block_duration_seconds' in df_visual_mismatch.columns:
        df_visual_mismatch['first_block_total_time_minutes'] = df_visual_mismatch['first_block_running_block_duration_seconds'] / 60.0
        if 'first_block_running_distance_travelled_m' in df_visual_mismatch.columns:
            df_visual_mismatch['first_block_running_distance_m_per_minute'] = (
                df_visual_mismatch['first_block_running_distance_travelled_m'] / 
                df_visual_mismatch['first_block_total_time_minutes']
            )
    first_block_turn_col = None
    if 'first_block_turning_distance_turned_deg' in df_visual_mismatch.columns:
        first_block_turn_col = 'first_block_turning_distance_turned_deg'
    elif 'first_block_turning_distance_turned_m' in df_visual_mismatch.columns:
        first_block_turn_col = 'first_block_turning_distance_turned_m'
        print("⚠️ Detected legacy column 'first_block_turning_distance_turned_m'; treating values as degrees. Please regenerate cohort CSVs to use '_deg'.")

    if first_block_turn_col is not None:
        df_visual_mismatch['first_block_turning_distance_deg_per_minute'] = (
            df_visual_mismatch[first_block_turn_col] / 
            df_visual_mismatch['first_block_total_time_minutes']
        )
    
    # For last_block - calculate time and normalize distances
    # Note: With fixed SANDBOX_2, last_block_running_block_duration_seconds is the actual duration of last block only
    if 'last_block_running_block_duration_seconds' in df_visual_mismatch.columns:
        # Use last_block duration directly (no longer cumulative after SANDBOX_2 fix)
        df_visual_mismatch['last_block_total_time_minutes'] = (
            df_visual_mismatch['last_block_running_block_duration_seconds'] / 60.0
        ).clip(lower=0.1)  # Avoid division by zero
        print(f"✅ Using last_block duration directly from data")
    else:
        # Fallback: use first_block duration as estimate
        if 'first_block_running_block_duration_seconds' in df_visual_mismatch.columns:
            df_visual_mismatch['last_block_total_time_minutes'] = df_visual_mismatch['first_block_running_block_duration_seconds'] / 60.0
            print("⚠️ Warning: Using first_block duration as fallback for last_block")
        else:
            df_visual_mismatch['last_block_total_time_minutes'] = 5.0  # Default fallback
            print("⚠️ Warning: Using default 5 minutes for last_block duration")
    
    # Calculate last_block running distance - prioritize direct column, then calculation
    if 'last_block_running_distance_travelled_m' in df_visual_mismatch.columns:
        # Use direct last_block distance column
        df_visual_mismatch['last_block_running_distance_m_per_minute'] = (
            df_visual_mismatch['last_block_running_distance_travelled_m'] / 
            df_visual_mismatch['last_block_total_time_minutes']
        )
    elif all(col in df_visual_mismatch.columns for col in ['running_distance_travelled_m', 'first_block_running_distance_travelled_m']):
        # Calculate as difference between total and first block
        last_block_distance = (
            df_visual_mismatch['running_distance_travelled_m'] - 
            df_visual_mismatch['first_block_running_distance_travelled_m']
        )
        df_visual_mismatch['last_block_running_distance_m_per_minute'] = (
            last_block_distance / df_visual_mismatch['last_block_total_time_minutes']
        )
        # Handle negative values (shouldn't happen, but safety check)
        df_visual_mismatch['last_block_running_distance_m_per_minute'] = df_visual_mismatch['last_block_running_distance_m_per_minute'].clip(lower=0)
    
    # Calculate last_block turning distance - prioritize direct column, then calculation
    last_block_turn_col = None
    if 'last_block_turning_distance_turned_deg' in df_visual_mismatch.columns:
        last_block_turn_col = 'last_block_turning_distance_turned_deg'
    elif 'last_block_turning_distance_turned_m' in df_visual_mismatch.columns:
        last_block_turn_col = 'last_block_turning_distance_turned_m'
        print("⚠️ Detected legacy column 'last_block_turning_distance_turned_m'; treating values as degrees. Please regenerate cohort CSVs to use '_deg'.")

    if last_block_turn_col is not None:
        # Use direct last_block distance column
        df_visual_mismatch['last_block_turning_distance_deg_per_minute'] = (
            df_visual_mismatch[last_block_turn_col] / 
            df_visual_mismatch['last_block_total_time_minutes']
        )
    else:
        # Calculate as difference between total and first block
        total_turn_col = None
        first_turn_col = None
        if all(col in df_visual_mismatch.columns for col in ['turning_distance_turned_deg', 'first_block_turning_distance_turned_deg']):
            total_turn_col = 'turning_distance_turned_deg'
            first_turn_col = 'first_block_turning_distance_turned_deg'
        elif all(col in df_visual_mismatch.columns for col in ['turning_distance_turned_m', 'first_block_turning_distance_turned_m']):
            total_turn_col = 'turning_distance_turned_m'
            first_turn_col = 'first_block_turning_distance_turned_m'
            print("⚠️ Detected legacy turning distance columns ('*_m'); treating values as degrees. Please regenerate cohort CSVs to use '_deg'.")

        if total_turn_col and first_turn_col:
            last_block_turn_distance = (
                df_visual_mismatch[total_turn_col] - 
                df_visual_mismatch[first_turn_col]
            )
            df_visual_mismatch['last_block_turning_distance_deg_per_minute'] = (
                last_block_turn_distance / df_visual_mismatch['last_block_total_time_minutes']
            )
            # Handle negative values (shouldn't happen, but safety check)
            df_visual_mismatch['last_block_turning_distance_deg_per_minute'] = df_visual_mismatch['last_block_turning_distance_deg_per_minute'].clip(lower=0)
    
    # Debug: Check if all columns were created
    print(f"\n✅ Prepared visual mismatch data for {len(df_visual_mismatch)} rows")
    print(f"\n📊 Checking velocity calculations:")
    print(f"   first_block_running_velocity_avg_cm_per_s: {df_visual_mismatch.get('first_block_running_velocity_avg_cm_per_s', pd.Series()).notna().sum() if 'first_block_running_velocity_avg_cm_per_s' in df_visual_mismatch.columns else 0}/{len(df_visual_mismatch)} non-NaN")
    print(f"   last_block_running_velocity_avg_cm_per_s: {df_visual_mismatch.get('last_block_running_velocity_avg_cm_per_s', pd.Series()).notna().sum() if 'last_block_running_velocity_avg_cm_per_s' in df_visual_mismatch.columns else 0}/{len(df_visual_mismatch)} non-NaN")
    print(f"\n📊 Checking distance calculations:")
    print(f"   first_block_running_distance_m_per_minute: {df_visual_mismatch.get('first_block_running_distance_m_per_minute', pd.Series()).notna().sum() if 'first_block_running_distance_m_per_minute' in df_visual_mismatch.columns else 0}/{len(df_visual_mismatch)} non-NaN")
    print(f"   last_block_running_distance_m_per_minute: {df_visual_mismatch.get('last_block_running_distance_m_per_minute', pd.Series()).notna().sum() if 'last_block_running_distance_m_per_minute' in df_visual_mismatch.columns else 0}/{len(df_visual_mismatch)} non-NaN")
    print(f"   first_block_turning_distance_deg_per_minute: {df_visual_mismatch.get('first_block_turning_distance_deg_per_minute', pd.Series()).notna().sum() if 'first_block_turning_distance_deg_per_minute' in df_visual_mismatch.columns else 0}/{len(df_visual_mismatch)} non-NaN")
    print(f"   last_block_turning_distance_deg_per_minute: {df_visual_mismatch.get('last_block_turning_distance_deg_per_minute', pd.Series()).notna().sum() if 'last_block_turning_distance_deg_per_minute' in df_visual_mismatch.columns else 0}/{len(df_visual_mismatch)} non-NaN")
else:
    print("\n⚠️ Visual mismatch analysis skipped")
    df_vm_melted = None


# %%
# Visual Mismatch Analysis: First Block vs Last Block Comparison
#----------------------------
# Note: Cell 9 already prepared the data with all conversions. 
# This cell creates the melted dataframe and calculates statistics.
if EXCLUDE_VISUAL_MISMATCH and len(df_visual_mismatch) > 0:
    print("\n" + "="*80)
    print("VISUAL MISMATCH ANALYSIS: First Block vs Last Block Comparison")
    print("="*80)
    
    # Ensure we have the necessary conversions (Cell 9 should have done this, but double-check)
    # Convert last_block running velocity from m/s to cm/s if not already done
    if 'last_block_running_velocity_avg_m_per_s' in df_visual_mismatch.columns and 'last_block_running_velocity_avg_cm_per_s' not in df_visual_mismatch.columns:
        df_visual_mismatch['last_block_running_velocity_avg_cm_per_s'] = df_visual_mismatch['last_block_running_velocity_avg_m_per_s'] * 100
        print("✅ Converted last_block_running_velocity to cm/s")
    
    # Debug: Check what columns we have
    print(f"\n🔍 Debug: Checking available columns in df_visual_mismatch:")
    last_block_cols = [col for col in df_visual_mismatch.columns if 'last_block' in col.lower()]
    print(f"   Last block columns: {len(last_block_cols)}")
    for col in ['last_block_running_velocity_avg_cm_per_s', 'last_block_running_distance_m_per_minute', 
                'last_block_turning_distance_deg_per_minute', 'last_block_turning_velocity_avg_deg_per_s']:
        if col in df_visual_mismatch.columns:
            non_nan = df_visual_mismatch[col].notna().sum()
            print(f"   ✓ {col}: {non_nan}/{len(df_visual_mismatch)} non-NaN")
        else:
            print(f"   ✗ {col}: NOT FOUND")
    
    # Data should already be prepared by Cell 9, but ensure distances are calculated if missing
    # (Cell 9 handles this, so this is just a safety check)
    if 'last_block_running_distance_m_per_minute' not in df_visual_mismatch.columns:
        # This shouldn't happen if Cell 9 ran correctly, but handle it just in case
        print("⚠️ Warning: last_block distances not found, attempting to calculate...")
        # Use the same logic as Cell 9 (simplified after SANDBOX_2 fix)
        if 'last_block_running_block_duration_seconds' in df_visual_mismatch.columns:
            df_visual_mismatch['last_block_total_time_minutes'] = (
                df_visual_mismatch['last_block_running_block_duration_seconds'] / 60.0
            ).clip(lower=0.1)
            
            if 'last_block_running_distance_travelled_m' in df_visual_mismatch.columns:
                df_visual_mismatch['last_block_running_distance_m_per_minute'] = (
                    df_visual_mismatch['last_block_running_distance_travelled_m'] / 
                    df_visual_mismatch['last_block_total_time_minutes']
                )
            
            if 'last_block_turning_distance_turned_deg' in df_visual_mismatch.columns:
                df_visual_mismatch['last_block_turning_distance_deg_per_minute'] = (
                    df_visual_mismatch['last_block_turning_distance_turned_deg'] / 
                    df_visual_mismatch['last_block_total_time_minutes']
                )
            elif 'last_block_turning_distance_turned_m' in df_visual_mismatch.columns:
                print("⚠️ Detected legacy column 'last_block_turning_distance_turned_m'; treating values as degrees. Please regenerate cohort CSVs to use '_deg'.")
                df_visual_mismatch['last_block_turning_distance_deg_per_minute'] = (
                    df_visual_mismatch['last_block_turning_distance_turned_m'] / 
                    df_visual_mismatch['last_block_total_time_minutes']
                )
    
    # Create a long-format dataframe for easier plotting (first_block vs last_block)
    vm_melted_data = []
    
    for _, row in df_visual_mismatch.iterrows():
        animal_id = row['Animal_ID']
        exp_day = row['Experiment_Day']
        cohort = row['Cohort']
        
        # First block metrics
        vm_melted_data.append({
            'Animal_ID': animal_id,
            'Experiment_Day': exp_day,
            'Cohort': cohort,
            'Block': 'First Block',
            'running_velocity_cm_per_s': row.get('first_block_running_velocity_avg_cm_per_s', np.nan),
            'running_distance_m_per_minute': row.get('first_block_running_distance_m_per_minute', np.nan),
            'running_time_percentage': row.get('first_block_running_time_percentage', np.nan),
            'turning_velocity_deg_per_s': row.get('first_block_turning_velocity_avg_deg_per_s', np.nan),
            'turning_distance_deg_per_minute': row.get('first_block_turning_distance_deg_per_minute', np.nan),
            'turning_time_percentage': row.get('first_block_turning_time_percentage', np.nan) if 'first_block_turning_time_percentage' in row.index else np.nan,
        })
        
        # Last block metrics
        vm_melted_data.append({
            'Animal_ID': animal_id,
            'Experiment_Day': exp_day,
            'Cohort': cohort,
            'Block': 'Last Block',
            'running_velocity_cm_per_s': row.get('last_block_running_velocity_avg_cm_per_s', np.nan),
            'running_distance_m_per_minute': row.get('last_block_running_distance_m_per_minute', np.nan),
            'running_time_percentage': row.get('last_block_running_time_percentage', np.nan) if 'last_block_running_time_percentage' in row.index else np.nan,
            'turning_velocity_deg_per_s': row.get('last_block_turning_velocity_avg_deg_per_s', np.nan),
            'turning_distance_deg_per_minute': row.get('last_block_turning_distance_deg_per_minute', np.nan),
            'turning_time_percentage': row.get('last_block_turning_time_percentage', np.nan) if 'last_block_turning_time_percentage' in row.index else np.nan,
        })
    
    df_vm_melted = pd.DataFrame(vm_melted_data)
    
    print(f"\n✅ Prepared visual mismatch data:")
    print(f"   Total rows: {len(df_vm_melted)}")
    print(f"   Animals: {df_vm_melted['Animal_ID'].nunique()}")
    print(f"   Experiment days: {sorted(df_vm_melted['Experiment_Day'].unique())}")
    print(f"\nAvailable metrics:")
    print(f"   Running velocity: {'running_velocity_cm_per_s' in df_vm_melted.columns}")
    print(f"   Running distance: {'running_distance_m_per_minute' in df_vm_melted.columns}")
    print(f"   Running time %: {'running_time_percentage' in df_vm_melted.columns}")
    print(f"   Turning velocity: {'turning_velocity_deg_per_s' in df_vm_melted.columns}")
    print(f"   Turning distance: {'turning_distance_deg_per_minute' in df_vm_melted.columns}")
    print(f"   Turning time %: {'turning_time_percentage' in df_vm_melted.columns}")
    
    # Debug: Check last_block data in melted dataframe
    print(f"\n🔍 Debugging last_block data in df_vm_melted:")
    for metric in ['running_velocity_cm_per_s', 'running_distance_m_per_minute', 'turning_distance_deg_per_minute']:
        if metric in df_vm_melted.columns:
            last_block_data = df_vm_melted[(df_vm_melted['Block'] == 'Last Block')][metric]
            first_block_data = df_vm_melted[(df_vm_melted['Block'] == 'First Block')][metric]
            last_non_nan = last_block_data.notna().sum()
            first_non_nan = first_block_data.notna().sum()
            total_count = len(last_block_data)
            print(f"   {metric}:")
            print(f"      First Block: {first_non_nan}/{total_count} non-NaN")
            print(f"      Last Block: {last_non_nan}/{total_count} non-NaN")
            if last_non_nan == 0:
                print(f"      ⚠️ All last_block values are NaN!")
else:
    print("\n⚠️ Visual mismatch analysis skipped (EXCLUDE_VISUAL_MISMATCH=False or no visual mismatch data found)")
    df_vm_melted = pd.DataFrame()

# Calculate statistics for visual mismatch: First Block vs Last Block
#----------------------------
if EXCLUDE_VISUAL_MISMATCH and len(df_vm_melted) > 0 and 'Block' in df_vm_melted.columns:
    def calculate_block_stats(df_melted: pd.DataFrame, metric: str, block: str) -> Dict:
        """Calculate mean and SEM for a given metric and block."""
        block_data = df_melted[(df_melted['Block'] == block) & (df_melted[metric].notna())]
        if len(block_data) == 0:
            return None
        
        n = len(block_data)
        def sem(x):
            return x.std() / np.sqrt(n) if n > 1 and x.std() > 0 else 0
        
        values = block_data[metric]
        return {
            'block': block,
            'n': n,
            'mean': values.mean(),
            'sem': sem(values),
            'std': values.std()
        }
    
    # Calculate stats for each metric and block
    metrics_to_analyze = {
        'running_velocity_cm_per_s': 'Running Velocity (cm/s)',
        'running_distance_m_per_minute': 'Running Distance (m/min)',
        'running_time_percentage': 'Running Time (%)',
        'turning_velocity_deg_per_s': 'Turning Velocity (deg/s)',
        'turning_distance_deg_per_minute': 'Turning Distance (deg/min)',
        'turning_time_percentage': 'Turning Time (%)'
    }
    
    vm_stats = []
    for metric, metric_label in metrics_to_analyze.items():
        if metric in df_vm_melted.columns:
            # Prepare paired data for t-test (matching by Animal_ID and Experiment_Day)
            paired_data = []
            for animal_id in df_vm_melted['Animal_ID'].unique():
                for exp_day in df_vm_melted['Experiment_Day'].unique():
                    animal_day_data = df_vm_melted[
                        (df_vm_melted['Animal_ID'] == animal_id) & 
                        (df_vm_melted['Experiment_Day'] == exp_day)
                    ]
                    first_val = animal_day_data[(animal_day_data['Block'] == 'First Block')][metric].values
                    last_val = animal_day_data[(animal_day_data['Block'] == 'Last Block')][metric].values
                    if len(first_val) > 0 and len(last_val) > 0 and not np.isnan(first_val[0]) and not np.isnan(last_val[0]):
                        paired_data.append({
                            'Animal_ID': animal_id,
                            'Experiment_Day': exp_day,
                            'first_block': first_val[0],
                            'last_block': last_val[0]
                        })
            
            if len(paired_data) > 1:
                paired_df = pd.DataFrame(paired_data)
                first_vals = paired_df['first_block'].values
                last_vals = paired_df['last_block'].values
                
                # Perform paired t-test
                t_stat, p_value = ttest_rel(first_vals, last_vals)
                
                first_stats = calculate_block_stats(df_vm_melted, metric, 'First Block')
                last_stats = calculate_block_stats(df_vm_melted, metric, 'Last Block')
                
                if first_stats and last_stats:
                    vm_stats.append({
                        'metric': metric,
                        'metric_label': metric_label,
                        'first_block_mean': first_stats['mean'],
                        'first_block_sem': first_stats['sem'],
                        'first_block_n': first_stats['n'],
                        'last_block_mean': last_stats['mean'],
                        'last_block_sem': last_stats['sem'],
                        'last_block_n': last_stats['n'],
                        'difference': last_stats['mean'] - first_stats['mean'],
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'n_paired': len(paired_data)
                    })
    
    vm_stats_df = pd.DataFrame(vm_stats)
    if len(vm_stats_df) > 0:
        print("\n📊 Visual Mismatch Statistics: First Block vs Last Block")
        display_cols = ['metric_label', 'first_block_mean', 'first_block_sem', 'last_block_mean', 'last_block_sem', 'difference', 'p_value']
        print(vm_stats_df[display_cols].to_string(index=False))
        print("\n📈 Paired t-test results:")
        for _, row in vm_stats_df.iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
            print(f"   {row['metric_label']}: t={row['t_statistic']:.3f}, p={row['p_value']:.4f} {sig} (n={row['n_paired']})")
    else:
        print("\n⚠️ No statistics calculated - check available metrics")
        vm_stats_df = pd.DataFrame()
else:
    vm_stats_df = pd.DataFrame()


# %%
# Visualize Visual Mismatch: First Block vs Last Block Comparison
#----------------------------
if EXCLUDE_VISUAL_MISMATCH and len(df_vm_melted) > 0 and 'Block' in df_vm_melted.columns:
    def plot_vm_block_comparison(df_melted: pd.DataFrame, vm_stats_df: pd.DataFrame, output_dir: Path):
        """Create plots comparing first block vs last block for visual mismatch experiments."""
        
        metrics_to_plot = {
            'running_velocity_cm_per_s': ('Running Velocity', 'Running Velocity (cm/s)'),
            'running_distance_m_per_minute': ('Running Distance', 'Distance (m/min)'),
            'running_time_percentage': ('Running Time', 'Time (%)'),
            'turning_velocity_deg_per_s': ('Turning Velocity', 'Turning Velocity (deg/s)'),
            'turning_distance_deg_per_minute': ('Turning Distance', 'Distance (deg/min)'),
            'turning_time_percentage': ('Turning Time', 'Time (%)')
        }
        
        # Get available metrics
        available_metrics = {k: v for k, v in metrics_to_plot.items() if k in df_melted.columns}
        
        if len(available_metrics) == 0:
            print("⚠️ No metrics available for plotting")
            return
        
        # Create figure with subplots
        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Assign colors to mice for consistency
        vm_mouse_colors = assign_mouse_colors(df_visual_mismatch)
        
        for idx, (metric, (title, ylabel)) in enumerate(available_metrics.items()):
            ax = axes[idx]
            
            # Get data for this metric
            metric_data = df_melted[df_melted[metric].notna()].copy()
            
            if len(metric_data) == 0:
                ax.text(0.5, 0.5, f'No data for {title}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=14, fontweight='bold')
                continue
            
            # Prepare data for comparison plot (no bars, just points and lines)
            blocks = ['First Block', 'Last Block']
            x_pos = np.arange(len(blocks))
            
            # Get unique mice and experiment days
            unique_mice = sorted(metric_data['Animal_ID'].unique())
            exp_days = sorted(metric_data['Experiment_Day'].unique())
            
            # Plot individual mouse lines (like main plot style)
            for mouse in unique_mice:
                mouse_data = metric_data[metric_data['Animal_ID'] == mouse]
                mouse_values = []
                for block in blocks:
                    block_mouse_data = mouse_data[mouse_data['Block'] == block]
                    if len(block_mouse_data) > 0:
                        # Average across experiment days if multiple
                        mouse_values.append(block_mouse_data[metric].mean())
                    else:
                        mouse_values.append(np.nan)
                
                color = vm_mouse_colors.get(mouse, 'gray')
                ax.plot(x_pos, mouse_values, 'o-', color=color, alpha=0.7, 
                       linewidth=1.5, markersize=5, label=mouse if idx == 0 else "")
            
            # Plot mean ± SEM line (like main plot style, no bars)
            if len(vm_stats_df) > 0 and metric in vm_stats_df['metric'].values:
                stats_row = vm_stats_df[vm_stats_df['metric'] == metric].iloc[0]
                
                means = [stats_row['first_block_mean'], stats_row['last_block_mean']]
                sems = [stats_row['first_block_sem'], stats_row['last_block_sem']]
                
                # Plot mean line (black)
                ax.plot(x_pos, means, '-', color='black', linewidth=2.5, zorder=10)
                
                # Plot SEM as opaque grey fill
                ax.fill_between(x_pos, np.array(means) - np.array(sems), np.array(means) + np.array(sems), 
                               color='grey', alpha=0.5, zorder=9)
                
                # Add p-value annotation if significant
                if stats_row['p_value'] < 0.05:
                    sig_text = "***" if stats_row['p_value'] < 0.001 else "**" if stats_row['p_value'] < 0.01 else "*"
                    y_max = max(means) + max(sems)
                    ax.text(0.5, y_max * 1.1, sig_text, ha='center', va='bottom', fontsize=14, fontweight='bold')
            
            ax.set_xlabel('Block', fontsize=15, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=15, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(blocks, fontsize=12, rotation=20, ha='right')
            ax.grid(False)  # Match main plot style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if idx == 0 and len(unique_mice) <= 15:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        
        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Visual Mismatch: First Block vs Last Block Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save plot
        cohort_str = "_".join(cohort_names)
        plot_path = output_dir / f"{cohort_str}_visual_mismatch_blocks_comparison.svg"
        plt.savefig(plot_path, format='svg', bbox_inches='tight')
        print(f"✅ Saved visual mismatch plot to: {plot_path}")
        plt.close(fig)
    
    # Create the plot
    plot_vm_block_comparison(df_vm_melted, vm_stats_df, output_dir)
    
    # Save visual mismatch statistics to CSV
    if len(vm_stats_df) > 0:
        cohort_str = "_".join(cohort_names)
        vm_stats_csv_path = output_dir / f"{cohort_str}_visual_mismatch_stats.csv"
        vm_stats_df.to_csv(vm_stats_csv_path, index=False)
        print(f"✅ Saved visual mismatch statistics to: {vm_stats_csv_path}")
else:
    print("\n⚠️ Visual mismatch visualization skipped (no data available)")


# %%
# Optional: Create individual plots for each metric (larger, more detailed)
#----------------------------
def plot_individual_metric(df: pd.DataFrame, cohort_stats_df: pd.DataFrame,
                          mouse_colors: Dict[str, str], metric: str, 
                          metric_label: str, ylabel: str, output_path: Path):
    """Create a detailed plot for a single metric."""
    
    experiment_days = sorted(df['Experiment_Day'].unique())
    n_days = len(experiment_days)
    x_pos = np.arange(n_days)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot individual mice
    for mouse in df['Animal_ID'].unique():
        mouse_data = df[df['Animal_ID'] == mouse]
        mouse_values = []
        
        for day in experiment_days:
            day_mouse_data = mouse_data[mouse_data['Experiment_Day'] == day]
            if len(day_mouse_data) > 0:
                mouse_values.append(day_mouse_data[metric].values[0])
            else:
                mouse_values.append(np.nan)
        
        color = mouse_colors[mouse]
        ax.plot(x_pos, mouse_values, 'o-', color=color, alpha=0.7, 
                linewidth=2, markersize=8, label=mouse)
    
    # Plot cohort average ± SEM
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"
    
    if mean_col in cohort_stats_df.columns and sem_col in cohort_stats_df.columns:
        means = cohort_stats_df[mean_col].values
        sems = cohort_stats_df[sem_col].values
        
        ax.errorbar(x_pos, means, yerr=sems, fmt='o-', color='black', 
                   linewidth=4, markersize=12, capsize=8, capthick=3, 
                   label='Cohort Mean ± SEM', zorder=10)
    
    ax.set_xlabel('Experiment Day', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_label} per Experiment Day', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(experiment_days, rotation=45, ha='right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"✅ Saved plot to: {output_path}")
    plt.close(fig)

# Create individual plots (optional - uncomment if needed)
# plot_individual_metric(df, cohort_stats_df, mouse_colors, 
#                       'running_velocity_avg_cm_per_s', 'Running Velocity', 
#                       'Running Velocity (cm/s)', 
#                       output_dir / "running_velocity_detailed.svg")
# 
# plot_individual_metric(df, cohort_stats_df, mouse_colors,
#                       'running_distance_travelled_m_per_minute', 'Total Run Distance (Normalized)',
#                       'Run Distance (m/min)',
#                       output_dir / "running_distance_detailed.svg")
# 
# plot_individual_metric(df, cohort_stats_df, mouse_colors,
#                       'running_time_percentage', 'Time Spent Running',
#                       'Time Spent Running (%)',
#                       output_dir / "running_time_percentage_detailed.svg")
# 
# plot_individual_metric(df, cohort_stats_df, mouse_colors,
#                       'turning_velocity_avg_deg_per_s', 'Turning Velocity',
#                       'Turning Velocity (deg/s)',
#                       output_dir / "turning_velocity_detailed.svg")
# 
# plot_individual_metric(df, cohort_stats_df, mouse_colors,
#                       'turning_distance_turned_deg_per_minute', 'Total Turn Distance (Normalized)',
#                       'Turn Distance (deg/min)',
#                       output_dir / "turning_distance_detailed.svg")
# 
# plot_individual_metric(df, cohort_stats_df, mouse_colors,
#                       'turning_time_percentage', 'Time Spent Turning',
#                       'Time Spent Turning (%)',
#                       output_dir / "turning_time_percentage_detailed.svg")


# %%
# Eye Metrics Analysis: First Block vs Last Block Comparison (All Experiments)
#----------------------------
print("\n" + "="*80)
print("EYE METRICS ANALYSIS: First Block vs Last Block Comparison")
print("="*80)

# Check if eye data columns are available in the merged dataframe
saccade_cols = [col for col in df.columns if 'saccade' in col.lower() and 'first_block' in col]
pupil_cols = [col for col in df.columns if 'pupil' in col.lower() and 'first_block' in col]

if len(saccade_cols) > 0 or len(pupil_cols) > 0:
    print(f"✅ Found {len(saccade_cols)} saccade columns and {len(pupil_cols)} pupil columns")
    
    # Define eye metrics to analyze (using combined saccades and both eyes for pupil)
    eye_metrics_config = {
        # Saccade metrics (using combined)
        'saccade_combined_total': {
            'first_col': 'first_block_saccade_combined_total',
            'last_col': 'last_block_saccade_combined_total',
            'label': 'Total Saccades (Combined)',
            'ylabel': 'Saccade Count',
            'normalize': False
        },
        'saccade_combined_orienting_rate': {
            'first_col': 'first_block_saccade_combined_orienting_rate_per_min',
            'last_col': 'last_block_saccade_combined_orienting_rate_per_min',
            'label': 'Orienting Saccade Rate (Combined)',
            'ylabel': 'Rate (per min)',
            'normalize': False
        },
        'saccade_combined_compensatory_rate': {
            'first_col': 'first_block_saccade_combined_compensatory_rate_per_min',
            'last_col': 'last_block_saccade_combined_compensatory_rate_per_min',
            'label': 'Compensatory Saccade Rate (Combined)',
            'ylabel': 'Rate (per min)',
            'normalize': False
        },
        'saccade_combined_orienting_pct': {
            'first_col': 'first_block_saccade_combined_orienting_pct',
            'last_col': 'last_block_saccade_combined_orienting_pct',
            'label': 'Orienting Saccades (%)',
            'ylabel': 'Percentage (%)',
            'normalize': False
        },
        # Pupil metrics (both eyes)
        'pupil_eye1_velocity_corr': {
            'first_col': 'first_block_pupil_eye1_velocity_corr',
            'last_col': 'last_block_pupil_eye1_velocity_corr',
            'label': 'Pupil-Velocity Correlation (Eye 1)',
            'ylabel': 'Correlation (r)',
            'normalize': False
        },
        'pupil_eye2_velocity_corr': {
            'first_col': 'first_block_pupil_eye2_velocity_corr',
            'last_col': 'last_block_pupil_eye2_velocity_corr',
            'label': 'Pupil-Velocity Correlation (Eye 2)',
            'ylabel': 'Correlation (r)',
            'normalize': False
        },
    }
    
    # Create melted dataframe for eye metrics analysis
    eye_melted_data = []
    
    for _, row in df.iterrows():
        animal_id = row['Animal_ID']
        exp_day = row['Experiment_Day']
        cohort = row['Cohort']
        
        # Check if we have eye data for this row
        has_eye_data = any(col in row.index and pd.notna(row.get(col)) 
                          for metric_info in eye_metrics_config.values() 
                          for col in [metric_info['first_col'], metric_info['last_col']])
        
        if not has_eye_data:
            continue
        
        # First block metrics
        first_block_metrics = {'Animal_ID': animal_id, 'Experiment_Day': exp_day, 
                              'Cohort': cohort, 'Block': 'First Block'}
        for metric_key, metric_info in eye_metrics_config.items():
            if metric_info['first_col'] in row.index:
                first_block_metrics[metric_key] = row.get(metric_info['first_col'], np.nan)
            else:
                first_block_metrics[metric_key] = np.nan
        
        eye_melted_data.append(first_block_metrics)
        
        # Last block metrics
        last_block_metrics = {'Animal_ID': animal_id, 'Experiment_Day': exp_day,
                             'Cohort': cohort, 'Block': 'Last Block'}
        for metric_key, metric_info in eye_metrics_config.items():
            if metric_info['last_col'] in row.index:
                last_block_metrics[metric_key] = row.get(metric_info['last_col'], np.nan)
            else:
                last_block_metrics[metric_key] = np.nan
        
        eye_melted_data.append(last_block_metrics)
    
    df_eye_melted = pd.DataFrame(eye_melted_data)
    
    if len(df_eye_melted) > 0:
        print(f"\n✅ Prepared eye metrics data:")
        print(f"   Total rows: {len(df_eye_melted)}")
        print(f"   Animals: {df_eye_melted['Animal_ID'].nunique()}")
        print(f"   Experiment days: {sorted(df_eye_melted['Experiment_Day'].unique())}")
        print(f"\nAvailable metrics:")
        for metric_key, metric_info in eye_metrics_config.items():
            if metric_key in df_eye_melted.columns:
                non_nan = df_eye_melted[metric_key].notna().sum()
                print(f"   {metric_info['label']}: {non_nan}/{len(df_eye_melted)} non-NaN")
    else:
        print("\n⚠️ No eye metrics data found in any experiments")
        df_eye_melted = pd.DataFrame()
else:
    print("\n⚠️ No eye metrics columns found in CSV data")
    df_eye_melted = pd.DataFrame()
    eye_metrics_config = {}


# %%
# Calculate statistics for eye metrics: First Block vs Last Block
#----------------------------
if len(df_eye_melted) > 0 and 'Block' in df_eye_melted.columns:
    print("\n" + "="*80)
    print("EYE METRICS STATISTICS: First Block vs Last Block")
    print("="*80)
    
    def calculate_eye_block_stats(df_melted: pd.DataFrame, metric: str, block: str) -> Dict:
        """Calculate mean and SEM for a given eye metric and block."""
        block_data = df_melted[(df_melted['Block'] == block) & (df_melted[metric].notna())]
        if len(block_data) == 0:
            return None
        
        n = len(block_data)
        def sem(x):
            return x.std() / np.sqrt(n) if n > 1 and x.std() > 0 else 0
        
        values = block_data[metric]
        return {
            'block': block,
            'n': n,
            'mean': values.mean(),
            'sem': sem(values),
            'std': values.std()
        }
    
    # Calculate stats for each eye metric and block
    eye_stats = []
    for metric_key, metric_info in eye_metrics_config.items():
        if metric_key in df_eye_melted.columns:
            # Prepare paired data for t-test (matching by Animal_ID and Experiment_Day)
            paired_data = []
            for animal_id in df_eye_melted['Animal_ID'].unique():
                for exp_day in df_eye_melted['Experiment_Day'].unique():
                    animal_day_data = df_eye_melted[
                        (df_eye_melted['Animal_ID'] == animal_id) & 
                        (df_eye_melted['Experiment_Day'] == exp_day)
                    ]
                    first_val = animal_day_data[(animal_day_data['Block'] == 'First Block')][metric_key].values
                    last_val = animal_day_data[(animal_day_data['Block'] == 'Last Block')][metric_key].values
                    if len(first_val) > 0 and len(last_val) > 0 and not np.isnan(first_val[0]) and not np.isnan(last_val[0]):
                        paired_data.append({
                            'Animal_ID': animal_id,
                            'Experiment_Day': exp_day,
                            'first_block': first_val[0],
                            'last_block': last_val[0]
                        })
            
            if len(paired_data) > 1:
                paired_df = pd.DataFrame(paired_data)
                first_vals = paired_df['first_block'].values
                last_vals = paired_df['last_block'].values
                
                # Perform paired t-test
                t_stat, p_value = ttest_rel(first_vals, last_vals)
                
                first_stats = calculate_eye_block_stats(df_eye_melted, metric_key, 'First Block')
                last_stats = calculate_eye_block_stats(df_eye_melted, metric_key, 'Last Block')
                
                if first_stats and last_stats:
                    eye_stats.append({
                        'metric': metric_key,
                        'metric_label': metric_info['label'],
                        'first_block_mean': first_stats['mean'],
                        'first_block_sem': first_stats['sem'],
                        'first_block_n': first_stats['n'],
                        'last_block_mean': last_stats['mean'],
                        'last_block_sem': last_stats['sem'],
                        'last_block_n': last_stats['n'],
                        'difference': last_stats['mean'] - first_stats['mean'],
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'n_paired': len(paired_data)
                    })
    
    eye_stats_df = pd.DataFrame(eye_stats)
    if len(eye_stats_df) > 0:
        print("\n📊 Eye Metrics Statistics: First Block vs Last Block")
        display_cols = ['metric_label', 'first_block_mean', 'first_block_sem', 'last_block_mean', 'last_block_sem', 'difference', 'p_value']
        print(eye_stats_df[display_cols].to_string(index=False))
        print("\n📈 Paired t-test results:")
        for _, row in eye_stats_df.iterrows():
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
            print(f"   {row['metric_label']}: t={row['t_statistic']:.3f}, p={row['p_value']:.4f} {sig} (n={row['n_paired']})")
        
        # Save eye metrics statistics to CSV
        cohort_str = "_".join(cohort_names)
        eye_stats_csv_path = output_dir / f"{cohort_str}_eye_metrics_stats.csv"
        eye_stats_df.to_csv(eye_stats_csv_path, index=False)
        print(f"\n✅ Saved eye metrics statistics to: {eye_stats_csv_path}")
    else:
        print("\n⚠️ No eye metrics statistics calculated - check available metrics")
        eye_stats_df = pd.DataFrame()
else:
    eye_stats_df = pd.DataFrame()


# %%
# Visualize Eye Metrics: First Block vs Last Block Comparison
#----------------------------
if len(df_eye_melted) > 0 and 'Block' in df_eye_melted.columns and len(eye_stats_df) > 0:
    print("\n" + "="*80)
    print("CREATING EYE METRICS PLOTS")
    print("="*80)
    
    def plot_eye_block_comparison(df_melted: pd.DataFrame, eye_stats_df: pd.DataFrame, 
                                  eye_metrics_config: Dict, mouse_colors: Dict, output_dir: Path):
        """Create plots comparing first block vs last block for eye metrics."""
        
        # Get available metrics (only those with data)
        available_metrics = {k: v for k, v in eye_metrics_config.items() 
                           if k in df_melted.columns and df_melted[k].notna().sum() > 0}
        
        if len(available_metrics) == 0:
            print("⚠️ No eye metrics available for plotting")
            return
        
        # Create figure with subplots
        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Use existing mouse colors from main analysis
        for idx, (metric, metric_info) in enumerate(available_metrics.items()):
            ax = axes[idx]
            
            # Get data for this metric
            metric_data = df_melted[df_melted[metric].notna()].copy()
            
            if len(metric_data) == 0:
                ax.text(0.5, 0.5, f'No data for {metric_info["label"]}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric_info['label'], fontsize=14, fontweight='bold')
                continue
            
            # Prepare data for comparison plot
            blocks = ['First Block', 'Last Block']
            x_pos = np.arange(len(blocks))
            
            # Get unique mice
            unique_mice = sorted(metric_data['Animal_ID'].unique())
            
            # Plot individual mouse lines (like main plot style)
            for mouse in unique_mice:
                mouse_data = metric_data[metric_data['Animal_ID'] == mouse]
                mouse_values = []
                for block in blocks:
                    block_mouse_data = mouse_data[mouse_data['Block'] == block]
                    if len(block_mouse_data) > 0:
                        # Average across experiment days if multiple
                        mouse_values.append(block_mouse_data[metric].mean())
                    else:
                        mouse_values.append(np.nan)
                
                color = mouse_colors.get(mouse, 'gray')
                ax.plot(x_pos, mouse_values, 'o-', color=color, alpha=0.7, 
                       linewidth=1.5, markersize=5, label=mouse if idx == 0 else "")
            
            # Plot mean ± SEM line (like main plot style)
            if len(eye_stats_df) > 0 and metric in eye_stats_df['metric'].values:
                stats_row = eye_stats_df[eye_stats_df['metric'] == metric].iloc[0]
                
                means = [stats_row['first_block_mean'], stats_row['last_block_mean']]
                sems = [stats_row['first_block_sem'], stats_row['last_block_sem']]
                
                # Plot mean line (black)
                ax.plot(x_pos, means, '-', color='black', linewidth=2.5, zorder=10)
                
                # Plot SEM as opaque grey fill
                ax.fill_between(x_pos, np.array(means) - np.array(sems), 
                               np.array(means) + np.array(sems), 
                               color='grey', alpha=0.5, zorder=9)
                
                # Add p-value annotation if significant
                if stats_row['p_value'] < 0.05:
                    sig_text = "***" if stats_row['p_value'] < 0.001 else "**" if stats_row['p_value'] < 0.01 else "*"
                    y_max = max(means) + max(sems)
                    ax.text(0.5, y_max * 1.1, sig_text, ha='center', va='bottom', 
                           fontsize=14, fontweight='bold')
            
            ax.set_xlabel('Block', fontsize=15, fontweight='bold')
            ax.set_ylabel(metric_info['ylabel'], fontsize=15, fontweight='bold')
            ax.set_title(metric_info['label'], fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(blocks, fontsize=12, rotation=20, ha='right')
            ax.grid(False)  # Match main plot style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if idx == 0 and len(unique_mice) <= 15:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        
        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Eye Metrics: First Block vs Last Block Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save plot
        cohort_str = "_".join(cohort_names)
        plot_path = output_dir / f"{cohort_str}_eye_metrics_blocks_comparison.svg"
        plt.savefig(plot_path, format='svg', bbox_inches='tight')
        print(f"✅ Saved eye metrics plot to: {plot_path}")
        plt.close(fig)
    
    # Create the plot (use mouse colors from main analysis)
    plot_eye_block_comparison(df_eye_melted, eye_stats_df, eye_metrics_config, 
                              mouse_colors, output_dir)
else:
    print("\n⚠️ Eye metrics visualization skipped (no data available)")


# %%
# Visualize Saccade Proportions: Orienting vs Compensatory (First vs Last Block)
#----------------------------
if len(df_eye_melted) > 0 and 'Block' in df_eye_melted.columns:
    # Check if we have saccade proportion data
    has_orienting = 'saccade_combined_orienting_pct' in df_eye_melted.columns
    has_compensatory = any('compensatory_pct' in col for col in df_eye_melted.columns)
    
    if has_orienting:
        print("\n" + "="*80)
        print("CREATING SACCADE PROPORTION PLOTS")
        print("="*80)
        
        def plot_saccade_proportions(df_melted: pd.DataFrame, mouse_colors: Dict, output_dir: Path):
            """Create stacked bar plots showing proportion of orienting vs compensatory saccades."""
            
            # Get data for saccade proportions
            saccade_data = df_melted[
                df_melted['saccade_combined_orienting_pct'].notna()
            ].copy()
            
            if len(saccade_data) == 0:
                print("⚠️ No saccade proportion data available")
                return
            
            # Calculate compensatory percentage (100 - orienting)
            saccade_data['saccade_combined_compensatory_pct_calc'] = 100 - saccade_data['saccade_combined_orienting_pct']
            
            # Create figure with two subplots: one for individual mice, one for cohort average
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # ========== Plot 1: Individual Mice ==========
            ax1 = axes[0]
            
            blocks = ['First Block', 'Last Block']
            unique_mice = sorted(saccade_data['Animal_ID'].unique())
            x_pos = np.arange(len(blocks))
            width = 0.8 / len(unique_mice)  # Width of bars
            
            # Plot each mouse as a separate grouped bar
            for mouse_idx, mouse in enumerate(unique_mice):
                mouse_data = saccade_data[saccade_data['Animal_ID'] == mouse]
                
                orienting_vals = []
                compensatory_vals = []
                
                for block in blocks:
                    block_data = mouse_data[mouse_data['Block'] == block]
                    if len(block_data) > 0:
                        # Average across experiment days if multiple
                        orienting_vals.append(block_data['saccade_combined_orienting_pct'].mean())
                        compensatory_vals.append(block_data['saccade_combined_compensatory_pct_calc'].mean())
                    else:
                        orienting_vals.append(0)
                        compensatory_vals.append(0)
                
                color = mouse_colors.get(mouse, 'gray')
                offset = (mouse_idx - len(unique_mice)/2) * width + width/2
                
                # Stacked bars for each mouse
                ax1.bar(x_pos + offset, orienting_vals, width, 
                       label=f'{mouse}' if mouse_idx < 10 else '', 
                       color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax1.bar(x_pos + offset, compensatory_vals, width, 
                       bottom=orienting_vals, color=color, alpha=0.3, 
                       edgecolor='black', linewidth=0.5)
            
            ax1.set_ylabel('Percentage (%)', fontsize=15, fontweight='bold')
            ax1.set_title('Saccade Proportions by Mouse', fontsize=14, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(blocks, fontsize=12, rotation=0)
            ax1.set_ylim(0, 100)
            ax1.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax1.grid(False)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            if len(unique_mice) <= 10:
                ax1.legend(loc='upper left', fontsize=8, ncol=2)
            
            # Add text labels
            ax1.text(0.02, 0.98, 'Orienting', transform=ax1.transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax1.text(0.02, 0.02, 'Compensatory', transform=ax1.transAxes, 
                    fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # ========== Plot 2: Cohort Average ==========
            ax2 = axes[1]
            
            # Calculate mean proportions for each block
            mean_orienting = []
            sem_orienting = []
            mean_compensatory = []
            sem_compensatory = []
            
            for block in blocks:
                block_data = saccade_data[saccade_data['Block'] == block]
                
                # Calculate mean and SEM
                orienting_vals = block_data['saccade_combined_orienting_pct'].values
                compensatory_vals = block_data['saccade_combined_compensatory_pct_calc'].values
                
                n = len(orienting_vals)
                mean_orienting.append(orienting_vals.mean())
                mean_compensatory.append(compensatory_vals.mean())
                sem_orienting.append(orienting_vals.std() / np.sqrt(n) if n > 1 else 0)
                sem_compensatory.append(compensatory_vals.std() / np.sqrt(n) if n > 1 else 0)
            
            # Create stacked bars
            bar_width = 0.6
            bars1 = ax2.bar(x_pos, mean_orienting, bar_width, 
                           label='Orienting', color='steelblue', 
                           edgecolor='black', linewidth=1.5)
            bars2 = ax2.bar(x_pos, mean_compensatory, bar_width, 
                           bottom=mean_orienting, label='Compensatory', 
                           color='coral', edgecolor='black', linewidth=1.5)
            
            # Add error bars (only for orienting, as total is always 100%)
            ax2.errorbar(x_pos, mean_orienting, yerr=sem_orienting, 
                        fmt='none', ecolor='black', capsize=5, capthick=2, linewidth=2)
            
            # Add percentage labels on bars
            for i, (ori, comp) in enumerate(zip(mean_orienting, mean_compensatory)):
                # Orienting label
                ax2.text(x_pos[i], ori/2, f'{ori:.1f}%', 
                        ha='center', va='center', fontweight='bold', fontsize=12)
                # Compensatory label
                ax2.text(x_pos[i], ori + comp/2, f'{comp:.1f}%', 
                        ha='center', va='center', fontweight='bold', fontsize=12)
            
            ax2.set_ylabel('Percentage (%)', fontsize=15, fontweight='bold')
            ax2.set_title('Cohort Average Saccade Proportions', fontsize=14, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(blocks, fontsize=12, rotation=0)
            ax2.set_ylim(0, 100)
            ax2.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.legend(loc='upper right', fontsize=12)
            ax2.grid(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            plt.suptitle('Saccade Type Proportions: First Block vs Last Block', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            cohort_str = "_".join(cohort_names)
            plot_path = output_dir / f"{cohort_str}_saccade_proportions.svg"
            plt.savefig(plot_path, format='svg', bbox_inches='tight')
            print(f"✅ Saved saccade proportions plot to: {plot_path}")
            plt.close(fig)
            
            # Print summary statistics
            print("\n📊 Saccade Proportion Summary:")
            for i, block in enumerate(blocks):
                print(f"\n{block}:")
                print(f"   Orienting: {mean_orienting[i]:.2f} ± {sem_orienting[i]:.2f}%")
                print(f"   Compensatory: {mean_compensatory[i]:.2f} ± {sem_compensatory[i]:.2f}%")
        
        # Create the plot
        plot_saccade_proportions(df_eye_melted, mouse_colors, output_dir)
    else:
        print("\n⚠️ Saccade proportion visualization skipped (no orienting % data available)")
else:
    print("\n⚠️ Saccade proportion visualization skipped (no eye data available)")


# %%
