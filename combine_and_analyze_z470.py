"""
Combine and Analyze Mouse z470 Change Data

This script:
1. Combines multiple mouse-specific *_z470_change.csv files
2. Extracts experiment day information from paths
3. Performs 2-way repeated measures ANOVA
4. Creates publication-quality boxplots separated by experiment day
"""

import pandas as pd
import numpy as np
import glob
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# For statistics
try:
    import pingouin as pg
except ImportError:
    print("Installing pingouin for statistical analysis...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pingouin'])
    import pingouin as pg

# Set plotting style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

#%% Load and Combine Data

# Base directories containing mouse-specific subdirectories
base_dirs = [
    "/Volumes/RanczLab2/EXP_2_fluoxetine_open_loop/",
    "/Volumes/RanczLab2/EXP_2_open_loop/"
]

# Find all CSV files matching the pattern
all_csv_files = []
for base_dir in base_dirs:
    csv_files = glob.glob(os.path.join(base_dir, "*/*_z470_change.csv"))
    # Filter out hidden files
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("._")]
    all_csv_files.extend(csv_files)

print(f"Found {len(all_csv_files)} CSV files")
for f in all_csv_files:
    print(f"  - {f}")

#%% Extract Experiment Information and Combine

def extract_experiment_info(file_path):
    """Extract experiment day and type from file path."""
    # Get the parent directory name
    parent_dir = Path(file_path).parent.name
    
    # Try to extract date (YYYY-MM-DD format) from parent directory
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', parent_dir)
    exp_date = date_match.group(1) if date_match else None
    
    # If no date in parent, check sibling directories for dates
    if not exp_date:
        parent_path = Path(file_path).parent.parent
        for sibling in parent_path.iterdir():
            if sibling.is_dir() and parent_dir in sibling.name:
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', sibling.name)
                if date_match:
                    exp_date = date_match.group(1)
                    break
    
    # If still no date, use "Unknown" as placeholder
    if not exp_date:
        exp_date = "Unknown_Date"
    
    # Determine experiment type from base directory path
    exp_type = "fluoxetine" if "fluoxetine" in file_path.lower() else "control"
    
    return exp_date, exp_type

# Read and combine all CSV files
dfs = []
for csv_file in all_csv_files:
    df = pd.read_csv(csv_file)
    
    # Extract experiment info
    exp_date, exp_type = extract_experiment_info(csv_file)
    df['Exp_Day'] = exp_date
    df['Experiment_Type'] = exp_type
    df['File_Path'] = csv_file
    
    dfs.append(df)
    print(f"Loaded: {os.path.basename(csv_file)} | Mouse: {df['Mouse'].values[0]} | Day: {exp_date} | Type: {exp_type}")

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined dataset shape: {combined_df.shape}")
print(f"Unique mice: {combined_df['Mouse'].unique()}")
print(f"Unique experiment days: {combined_df['Exp_Day'].unique()}")
print(f"Unique experiment types: {combined_df['Experiment_Type'].unique()}")

# Debug: Show first few rows
print("\nFirst few rows of combined data:")
print(combined_df[['Mouse', 'Exp_Day', 'Experiment_Type', 'First_5min_Mean', 'Last_5min_Mean']].head(10))

# Check for NaN values
print(f"\nNaN check:")
print(f"  Exp_Day has {combined_df['Exp_Day'].isna().sum()} NaN values")
print(f"  First_5min_Mean has {combined_df['First_5min_Mean'].isna().sum()} NaN values")
print(f"  Last_5min_Mean has {combined_df['Last_5min_Mean'].isna().sum()} NaN values")

#%% Prepare Data for Statistical Analysis

# Reshape data for repeated measures ANOVA
# We need long format: Mouse | Exp_Day | Experiment_Type | Time_Block | Z470_Mean

# Create long format dataframe
first_block = combined_df[['Mouse', 'Exp_Day', 'Experiment_Type', 'First_5min_Mean']].copy()
first_block['Time_Block'] = 'First'
first_block.rename(columns={'First_5min_Mean': 'Z470_Mean'}, inplace=True)

last_block = combined_df[['Mouse', 'Exp_Day', 'Experiment_Type', 'Last_5min_Mean']].copy()
last_block['Time_Block'] = 'Last'
last_block.rename(columns={'Last_5min_Mean': 'Z470_Mean'}, inplace=True)

# Combine
data_long = pd.concat([first_block, last_block], ignore_index=True)

# No need to filter - we now use "Unknown_Date" as placeholder instead of None
print(f"Data before any filtering: {len(data_long)} rows")

print(f"\nLong format data shape: {data_long.shape}")
print(f"\nFirst few rows:")
print(data_long.head(10))

# Check data ranges
print(f"\nZ470_Mean statistics:")
print(data_long['Z470_Mean'].describe())

# Create a numeric day identifier if needed
unique_days = sorted(data_long['Exp_Day'].unique())
day_mapping = {day: f"Day_{i+1}" for i, day in enumerate(unique_days)}
data_long['Day_Label'] = data_long['Exp_Day'].map(day_mapping)

print(f"\nDay mapping: {day_mapping}")
print(f"\nData distribution by Time_Block:")
print(data_long.groupby('Time_Block')['Z470_Mean'].describe())

#%% Perform 2-Way Repeated Measures ANOVA

print("=" * 80)
print("2-WAY REPEATED MEASURES ANOVA")
print("=" * 80)
print("\nFactors:")
print("  - Time_Block (within-subject): First vs Last")
print("  - Exp_Day (within-subject): Different experimental days")
print("\nDependent Variable: Z470_Mean")
print("\n" + "=" * 80)

# Get output directory
output_dir = os.path.dirname(all_csv_files[0]) if all_csv_files else "."
parent_dir = os.path.dirname(output_dir)

# Perform RM ANOVA
try:
    anova_result = pg.rm_anova(
        data=data_long,
        dv='Z470_Mean',
        within=['Time_Block', 'Day_Label'],
        subject='Mouse',
        detailed=True
    )
    
    print("\nANOVA Results:")
    print(anova_result)
    
    # Save results
    stats_file = os.path.join(parent_dir, "z470_anova_results.csv")
    anova_result.to_csv(stats_file, index=False)
    print(f"\nANOVA results saved to: {stats_file}")
    
except Exception as e:
    print(f"\nError performing RM ANOVA: {e}")
    print("\nThis might occur if:")
    print("  - Not all mice have data for all days")
    print("  - There's only one experiment day")
    print("\nTrying alternative analysis...")
    
    # Fall back to paired t-test for Time_Block only
    print("\n" + "=" * 80)
    print("PAIRED T-TEST (Time_Block: First vs Last)")
    print("=" * 80)
    
    ttest_result = pg.ttest(
        combined_df['First_5min_Mean'],
        combined_df['Last_5min_Mean'],
        paired=True
    )
    print("\nT-test Results:")
    print(ttest_result)
    
    stats_file = os.path.join(parent_dir, "z470_ttest_results.csv")
    ttest_result.to_csv(stats_file, index=False)
    print(f"\nT-test results saved to: {stats_file}")

#%% Post-hoc Tests (if multiple days)

if len(unique_days) > 1:
    print("\n" + "=" * 80)
    print("POST-HOC PAIRWISE COMPARISONS")
    print("=" * 80)
    
    try:
        # Pairwise comparisons for Time_Block within each Day
        posthoc = pg.pairwise_tests(
            data=data_long,
            dv='Z470_Mean',
            within=['Time_Block', 'Day_Label'],
            subject='Mouse',
            padjust='bonf'
        )
        
        print("\nPost-hoc Results:")
        print(posthoc)
        
        posthoc_file = os.path.join(parent_dir, "z470_posthoc_results.csv")
        posthoc.to_csv(posthoc_file, index=False)
        print(f"\nPost-hoc results saved to: {posthoc_file}")
    except Exception as e:
        print(f"Error in post-hoc analysis: {e}")
else:
    print("\nOnly one experiment day found - skipping post-hoc tests")

#%% Create Publication-Quality Plots

# Debug: Check data before plotting
print("\n" + "=" * 80)
print("DATA CHECK BEFORE PLOTTING")
print("=" * 80)
print(f"Number of unique days: {len(unique_days)}")
print(f"Unique days: {unique_days}")
print(f"Total data points in long format: {len(data_long)}")
print(f"\nData by day and time block:")
for day in unique_days:
    day_data = data_long[data_long['Exp_Day'] == day]
    first_count = len(day_data[day_data['Time_Block'] == 'First'])
    last_count = len(day_data[day_data['Time_Block'] == 'Last'])
    print(f"  {day}: First={first_count}, Last={last_count}")
print("=" * 80)

fig, ax = plt.subplots(figsize=(10, 6))

# Define colors
colors = {'First': 'lightblue', 'Last': 'lightcoral'}

# If multiple days, create grouped boxplots
if len(unique_days) > 1:
    # Create positions for grouped boxplots
    positions = []
    labels = []
    
    for i, day in enumerate(unique_days):
        day_label = day_mapping[day]
        positions.extend([i*3, i*3+1])
        labels.extend([f"{day_label}\nFirst", f"{day_label}\nLast"])
    
    # Prepare data for boxplot
    data_to_plot = []
    for day in unique_days:
        day_data = data_long[data_long['Exp_Day'] == day]
        first_data = day_data[day_data['Time_Block'] == 'First']['Z470_Mean'].values
        last_data = day_data[day_data['Time_Block'] == 'Last']['Z470_Mean'].values
        print(f"\n{day} - First: {len(first_data)} points, Last: {len(last_data)} points")
        if len(first_data) > 0:
            print(f"  First range: {first_data.min():.3f} to {first_data.max():.3f}")
        if len(last_data) > 0:
            print(f"  Last range: {last_data.min():.3f} to {last_data.max():.3f}")
        data_to_plot.append(first_data)
        data_to_plot.append(last_data)
    
    print(f"\nTotal datasets to plot: {len(data_to_plot)}")
    print(f"Non-empty datasets: {sum(1 for d in data_to_plot if len(d) > 0)}")
    
    # Create boxplot
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                     patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    
    # Color boxes
    for i, patch in enumerate(bp['boxes']):
        if i % 2 == 0:
            patch.set_facecolor(colors['First'])
        else:
            patch.set_facecolor(colors['Last'])
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
else:
    # Single day - simpler plot
    first_data = data_long[data_long['Time_Block'] == 'First']['Z470_Mean'].values
    last_data = data_long[data_long['Time_Block'] == 'Last']['Z470_Mean'].values
    
    print(f"\nSingle day plot:")
    print(f"  First: {len(first_data)} points")
    print(f"  Last: {len(last_data)} points")
    if len(first_data) > 0:
        print(f"  First range: {first_data.min():.3f} to {first_data.max():.3f}")
    if len(last_data) > 0:
        print(f"  Last range: {last_data.min():.3f} to {last_data.max():.3f}")
    
    data_to_plot = [first_data, last_data]
    
    bp = ax.boxplot(data_to_plot, positions=[0, 1], widths=0.6,
                     patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
    
    bp['boxes'][0].set_facecolor(colors['First'])
    bp['boxes'][1].set_facecolor(colors['Last'])
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['First 5 min', 'Last 5 min'])

# Overlay individual points with connecting lines
print("\nOverlaying individual mouse data...")
for i, day in enumerate(unique_days):
    day_data = data_long[data_long['Exp_Day'] == day]
    
    # Plot individual mice with connecting lines
    for mouse in day_data['Mouse'].unique():
        mouse_data = day_data[day_data['Mouse'] == mouse]
        first_vals = mouse_data[mouse_data['Time_Block'] == 'First']['Z470_Mean'].values
        last_vals = mouse_data[mouse_data['Time_Block'] == 'Last']['Z470_Mean'].values
        
        if len(first_vals) == 0 or len(last_vals) == 0:
            print(f"Warning: Mouse {mouse} missing data for day {day}")
            continue
            
        first_val = first_vals[0]
        last_val = last_vals[0]
        
        if len(unique_days) > 1:
            x_pos = [i*3, i*3+1]
        else:
            x_pos = [0, 1]
        
        # Add jitter
        x_jitter = [x + np.random.uniform(-0.1, 0.1) for x in x_pos]
        
        # Plot line
        ax.plot(x_jitter, [first_val, last_val], 'o-', alpha=0.5, linewidth=1.5, markersize=6)

ax.set_ylabel('Z470 Mean (normalized)', fontsize=14, fontweight='bold')
ax.set_xlabel('Time Block', fontsize=14, fontweight='bold')
ax.set_title('Z470 Signal: First vs Last 5-min Block\n(Separated by Experiment Day)', 
             fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save figure
plot_file = os.path.join(parent_dir, "z470_boxplot_by_day.pdf")
plt.savefig(plot_file, format='pdf', bbox_inches='tight', dpi=300)
print(f"\nPlot saved to: {plot_file}")

plt.show()

#%% Summary Statistics by Day

# Calculate summary statistics
summary_stats = data_long.groupby(['Day_Label', 'Time_Block'])['Z470_Mean'].agg([
    'count', 'mean', 'std', 'sem', 'median',
    ('q25', lambda x: x.quantile(0.25)),
    ('q75', lambda x: x.quantile(0.75))
]).reset_index()

print("\n" + "=" * 80)
print("SUMMARY STATISTICS BY DAY AND TIME BLOCK")
print("=" * 80)
print(summary_stats)

# Save summary statistics
summary_file = os.path.join(parent_dir, "z470_summary_statistics.csv")
summary_stats.to_csv(summary_file, index=False)
print(f"\nSummary statistics saved to: {summary_file}")

#%% Save Combined Data

# Save combined wide format data
combined_file = os.path.join(parent_dir, "combined_z470_change.csv")
combined_df.to_csv(combined_file, index=False)
print(f"\nCombined data (wide format) saved to: {combined_file}")

# Save long format data for further analysis
long_file = os.path.join(parent_dir, "z470_data_long_format.csv")
data_long.to_csv(long_file, index=False)
print(f"Long format data saved to: {long_file}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nAll output files saved to: {parent_dir}")
print("\nGenerated files:")
print("  - combined_z470_change.csv (wide format data)")
print("  - z470_data_long_format.csv (long format for stats)")
print("  - z470_anova_results.csv or z470_ttest_results.csv (statistical results)")
print("  - z470_summary_statistics.csv (descriptive stats)")
print("  - z470_boxplot_by_day.pdf (publication-quality plot)")
if len(unique_days) > 1:
    print("  - z470_posthoc_results.csv (post-hoc comparisons)")

