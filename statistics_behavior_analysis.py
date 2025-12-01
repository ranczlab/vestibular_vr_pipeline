"""
Script to combine metrics from Visual Mismatch day3 and day4 experiments
and perform two-way ANOVA analysis.

This script:
1. Loads CSV files from both experimental day folders
2. Combines data across experimental days
3. Performs two-way ANOVA (factors: experimental_day × condition)
4. Creates visualizations comparing the 4 conditions (day3 No halt, day3 Apply halt, day4 No halt, day4 Apply halt)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# Set plotting style (matching SANDBOX_4, with larger fonts)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "legend.title_fontsize": 15,
})
sns.set_theme(style="whitegrid")

# Define paths
BASE_PATH = Path("/Volumes/RanczLab2/DATA_NEW")
DAY3_PATH = BASE_PATH / "turning_analysis__Cohort1_rotation_Visual_mismatch_day3__Cohort3_rotation_Visual_mismatch_day3"
DAY4_PATH = BASE_PATH / "turning_analysis__Cohort1_rotation_Visual_mismatch_day4__Cohort3_rotation_Visual_mismatch_day4"

# Output directory - one level above data directories (in DATA_NEW)
OUTPUT_DIR = BASE_PATH / "two_way_anova_analysis__Visual_mismatch_day3_vs_day4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TWO-WAY ANOVA ANALYSIS: Visual Mismatch Day 3 vs Day 4")
print("=" * 80)
print()

# Exclude mouse B6J2723 from eye metrics (no reliable eye tracking data)
EXCLUDED_MICE_EYE_TRACKING = ['B6J2723']


def assign_mouse_colors_consistent(mouse_ids):
    """Assign consistent colors to mice using gnuplot2 palette (matching SANDBOX_4)."""
    normalized = [str(mouse) for mouse in mouse_ids]
    unique_mice = sorted(dict.fromkeys(normalized))
    if not unique_mice:
        return OrderedDict()
    palette = sns.color_palette("gnuplot2", len(unique_mice))
    return OrderedDict((mouse, palette[idx]) for idx, mouse in enumerate(unique_mice))


def load_and_combine_saccade_pupil_metrics():
    """Load and combine saccade and pupil metrics from both days."""
    print("📊 Loading saccade and pupil metrics...")
    
    # Load day3
    day3 = pd.read_csv(DAY3_PATH / "saccade_pupil_metrics_per_mouse.csv")
    day3['experimental_day'] = 'day3'
    
    # Load day4
    day4 = pd.read_csv(DAY4_PATH / "saccade_pupil_metrics_per_mouse.csv")
    day4['experimental_day'] = 'day4'
    
    # Combine
    combined = pd.concat([day3, day4], ignore_index=True)
    
    # Rename columns for clarity
    combined = combined.rename(columns={
        'group': 'condition',
        'saccade_post_mean': 'saccade_probability_post_halt',
        'saccade_peak': 'saccade_probability_peak',
        'pupil_post_mean': 'pupil_diameter_post_halt',
        'pupil_peak': 'pupil_diameter_peak'
    })
    
    # Exclude specified mice from eye tracking analysis
    if 'mouse' in combined.columns:
        before_exclusion = len(combined)
        combined = combined[~combined['mouse'].isin(EXCLUDED_MICE_EYE_TRACKING)]
        after_exclusion = len(combined)
        n_excluded = before_exclusion - after_exclusion
        if n_excluded > 0:
            print(f"  ⚠️ Excluded {n_excluded} records for mice: {', '.join(EXCLUDED_MICE_EYE_TRACKING)}")
    
    print(f"  ✓ Loaded {len(day3)} records from day3")
    print(f"  ✓ Loaded {len(day4)} records from day4")
    print(f"  ✓ Combined: {len(combined)} total records (after exclusions)")
    print()
    
    return combined


def load_and_combine_running_metrics():
    """Load and combine running metrics from both days."""
    print("🏃 Loading running metrics...")
    
    # Load day3
    day3 = pd.read_csv(DAY3_PATH / "running_metrics_combined.csv")
    day3['experimental_day'] = 'day3'
    
    # Load day4
    day4 = pd.read_csv(DAY4_PATH / "running_metrics_combined.csv")
    day4['experimental_day'] = 'day4'
    
    # Combine
    combined = pd.concat([day3, day4], ignore_index=True)
    
    # Rename columns for clarity
    combined = combined.rename(columns={
        'group': 'condition',
        'post_mean': 'running_velocity_post_halt',
        'pre_mean': 'running_velocity_pre_halt',
        'peak_velocity_abs_1s': 'running_velocity_peak_1s'
    })
    
    print(f"  ✓ Loaded {len(day3)} records from day3")
    print(f"  ✓ Loaded {len(day4)} records from day4")
    print(f"  ✓ Combined: {len(combined)} total records")
    print()
    
    return combined


def load_and_combine_turning_metrics():
    """Load and combine turning metrics from both days."""
    print("🔄 Loading turning metrics...")
    
    # Load day3
    day3 = pd.read_csv(DAY3_PATH / "turning_metrics_combined.csv")
    day3['experimental_day'] = 'day3'
    
    # For day4, check if the file exists; if not, try alternative file
    day4_file = DAY4_PATH / "turning_metrics_combined.csv"
    if not day4_file.exists():
        print("  ⚠️ Warning: turning_metrics_combined.csv not found in day4")
        print("  Attempting to use turning_velocity_difference.csv instead...")
        day4_diff = pd.read_csv(DAY4_PATH / "turning_velocity_difference.csv")
        
        # Reshape from wide to long format
        day4 = pd.DataFrame()
        for _, row in day4_diff.iterrows():
            # Apply halt row
            day4 = pd.concat([day4, pd.DataFrame({
                'mouse': [row['mouse']],
                'condition': ['Apply halt'],
                'mean_abs_velocity_0_2s': [row['Apply halt']],
                'experimental_day': ['day4']
            })], ignore_index=True)
            
            # No halt row
            day4 = pd.concat([day4, pd.DataFrame({
                'mouse': [row['mouse']],
                'condition': ['No halt'],
                'mean_abs_velocity_0_2s': [row['No halt']],
                'experimental_day': ['day4']
            })], ignore_index=True)
        
        print("  ✓ Converted turning_velocity_difference.csv to long format")
    else:
        day4 = pd.read_csv(day4_file)
        day4['experimental_day'] = 'day4'
        day4 = day4.rename(columns={'group': 'condition'})
    
    # Rename group to condition if needed for day3
    if 'group' in day3.columns:
        day3 = day3.rename(columns={'group': 'condition'})
    
    # Combine
    combined = pd.concat([day3, day4], ignore_index=True)
    
    print(f"  ✓ Loaded {len(day3)} records from day3")
    print(f"  ✓ Loaded {len(day4)} records from day4")
    print(f"  ✓ Combined: {len(combined)} total records")
    print()
    
    return combined


def load_and_combine_baseline_saccade():
    """Load and combine baseline saccade metrics from both days (if available)."""
    print("📍 Loading baseline saccade density...")
    
    # Load day3
    day3_file = DAY3_PATH / "baseline_saccade_density_per_mouse.csv"
    if not day3_file.exists():
        print("  ⚠️ Warning: baseline_saccade_density_per_mouse.csv not found in day3")
        return None
    
    day3 = pd.read_csv(day3_file)
    day3['experimental_day'] = 'day3'
    
    # Load day4 if available
    day4_file = DAY4_PATH / "baseline_saccade_density_per_mouse.csv"
    if not day4_file.exists():
        print("  ⚠️ Warning: baseline_saccade_density_per_mouse.csv not found in day4")
        print("  Returning only day3 data")
        if 'group' in day3.columns:
            day3 = day3.rename(columns={'group': 'condition'})
        # Exclude mice from day3 data
        if 'mouse' in day3.columns:
            before_exclusion = len(day3)
            day3 = day3[~day3['mouse'].isin(EXCLUDED_MICE_EYE_TRACKING)]
            n_excluded = before_exclusion - len(day3)
            if n_excluded > 0:
                print(f"  ⚠️ Excluded {n_excluded} records for mice: {', '.join(EXCLUDED_MICE_EYE_TRACKING)}")
        return day3
    
    day4 = pd.read_csv(day4_file)
    day4['experimental_day'] = 'day4'
    
    # Rename columns
    if 'group' in day3.columns:
        day3 = day3.rename(columns={'group': 'condition'})
    if 'group' in day4.columns:
        day4 = day4.rename(columns={'group': 'condition'})
    
    # Combine
    combined = pd.concat([day3, day4], ignore_index=True)
    
    # Exclude specified mice from eye tracking analysis
    if 'mouse' in combined.columns:
        before_exclusion = len(combined)
        combined = combined[~combined['mouse'].isin(EXCLUDED_MICE_EYE_TRACKING)]
        after_exclusion = len(combined)
        n_excluded = before_exclusion - after_exclusion
        if n_excluded > 0:
            print(f"  ⚠️ Excluded {n_excluded} records for mice: {', '.join(EXCLUDED_MICE_EYE_TRACKING)}")
    
    print(f"  ✓ Loaded {len(day3)} records from day3")
    print(f"  ✓ Loaded {len(day4)} records from day4")
    print(f"  ✓ Combined: {len(combined)} total records (after exclusions)")
    print()
    
    return combined


def perform_two_way_anova(data, dependent_var, title=""):
    """
    Perform two-way ANOVA with factors: experimental_day × condition
    
    Parameters:
    -----------
    data : DataFrame
        Data containing dependent_var, experimental_day, and condition columns
    dependent_var : str
        Name of the dependent variable column
    title : str
        Title for the analysis (for printing)
    
    Returns:
    --------
    dict : Dictionary containing ANOVA results and descriptive statistics
    """
    print(f"\n{'=' * 80}")
    print(f"TWO-WAY ANOVA: {title}")
    print(f"Dependent Variable: {dependent_var}")
    print(f"{'=' * 80}")
    
    # Remove rows with missing values
    clean_data = data[[dependent_var, 'experimental_day', 'condition', 'mouse']].dropna()
    
    if len(clean_data) == 0:
        print(f"  ⚠️ No valid data for {dependent_var}")
        return None
    
    print(f"\nSample sizes:")
    print(clean_data.groupby(['experimental_day', 'condition']).size())
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics:")
    desc_stats = clean_data.groupby(['experimental_day', 'condition'])[dependent_var].agg([
        'count', 'mean', 'std', 'sem'
    ]).round(4)
    print(desc_stats)
    
    # Two-way ANOVA
    try:
        # Fit the model
        formula = f'{dependent_var} ~ C(experimental_day) + C(condition) + C(experimental_day):C(condition)'
        model = ols(formula, data=clean_data).fit()
        anova_table = anova_lm(model, typ=2)
        
        print(f"\nTwo-Way ANOVA Results:")
        print(anova_table)
        
        # Extract p-values
        p_day = anova_table.loc['C(experimental_day)', 'PR(>F)']
        p_condition = anova_table.loc['C(condition)', 'PR(>F)']
        p_interaction = anova_table.loc['C(experimental_day):C(condition)', 'PR(>F)']
        
        print(f"\nSignificance Summary:")
        print(f"  Main effect of experimental_day: p = {p_day:.4f} {'***' if p_day < 0.001 else '**' if p_day < 0.01 else '*' if p_day < 0.05 else 'ns'}")
        print(f"  Main effect of condition: p = {p_condition:.4f} {'***' if p_condition < 0.001 else '**' if p_condition < 0.01 else '*' if p_condition < 0.05 else 'ns'}")
        print(f"  Interaction (day × condition): p = {p_interaction:.4f} {'***' if p_interaction < 0.001 else '**' if p_interaction < 0.01 else '*' if p_interaction < 0.05 else 'ns'}")
        
        # Post-hoc comparisons if main effects or interaction are significant
        if p_day < 0.05 or p_condition < 0.05 or p_interaction < 0.05:
            print(f"\nPost-hoc Pairwise Comparisons:")
            
            # Create all group combinations
            groups = clean_data.groupby(['experimental_day', 'condition'])
            group_names = list(groups.groups.keys())
            
            posthoc_results = []
            for i, g1 in enumerate(group_names):
                for g2 in group_names[i+1:]:
                    data1 = groups.get_group(g1)[dependent_var]
                    data2 = groups.get_group(g2)[dependent_var]
                    
                    if len(data1) > 1 and len(data2) > 1:
                        t_stat, p_val = stats.ttest_ind(data1, data2)
                        mean_diff = data1.mean() - data2.mean()
                        
                        posthoc_results.append({
                            'comparison': f"{g1} vs {g2}",
                            'mean_diff': mean_diff,
                            't_stat': t_stat,
                            'p_value': p_val,
                            'significant': 'Yes' if p_val < 0.05 else 'No'
                        })
            
            posthoc_df = pd.DataFrame(posthoc_results)
            print(posthoc_df.to_string(index=False))
        
        results = {
            'dependent_var': dependent_var,
            'title': title,
            'anova_table': anova_table,
            'descriptive_stats': desc_stats,
            'clean_data': clean_data,
            'p_day': p_day,
            'p_condition': p_condition,
            'p_interaction': p_interaction
        }
        
        if p_day < 0.05 or p_condition < 0.05 or p_interaction < 0.05:
            results['posthoc'] = posthoc_df
        
        return results
        
    except Exception as e:
        print(f"  ⚠️ Error performing ANOVA: {e}")
        return None


def plot_two_way_interaction(data, dependent_var, title, ylabel, filename):
    """
    Create interaction plot for two-way ANOVA showing all 4 conditions.
    
    Parameters:
    -----------
    data : DataFrame
        Data containing dependent_var, experimental_day, condition, and mouse columns
    dependent_var : str
        Name of the dependent variable
    title : str
        Plot title
    ylabel : str
        Y-axis label
    filename : str
        Output filename
    """
    # Remove missing values
    plot_data = data[[dependent_var, 'experimental_day', 'condition', 'mouse']].dropna()
    
    if len(plot_data) == 0:
        print(f"  ⚠️ No valid data to plot for {dependent_var}")
        return
    
    # Create figure with subplots - more compact size
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Assign consistent mouse colors
    mouse_colors = assign_mouse_colors_consistent(plot_data['mouse'].dropna().unique())
    
    # Left plot: Paired line plot with individual mice (matching SANDBOX_4 style)
    ax = axes[0]
    
    # Create a combined group variable for x-axis
    plot_data['group'] = plot_data['experimental_day'] + ' - ' + plot_data['condition']
    group_order = ['day3 - No halt', 'day3 - Apply halt', 'day4 - No halt', 'day4 - Apply halt']
    
    # Filter to only include groups that exist in the data
    group_order = [g for g in group_order if g in plot_data['group'].values]
    
    # Calculate means and SEMs
    summary = plot_data.groupby('group')[dependent_var].agg(['mean', 'sem']).reindex(group_order)
    
    # Plot setup
    x_pos = np.arange(len(group_order))
    
    # Plot individual mice with connecting lines (matching SANDBOX_4 style)
    for mouse in plot_data['mouse'].dropna().unique():
        mouse_data = plot_data[plot_data['mouse'] == mouse]
        values = []
        for group in group_order:
            group_vals = mouse_data[mouse_data['group'] == group][dependent_var]
            if len(group_vals) > 0:
                values.append(group_vals.iloc[0])
            else:
                values.append(np.nan)
        
        # Only plot if we have some values
        if not all(np.isnan(values)):
            ax.plot(x_pos, values, marker='o', markersize=8, linewidth=1.1, alpha=0.65,
                   color=mouse_colors.get(mouse, '#1f77b4'), zorder=2)
    
    # Add group mean with SEM shading
    mean_values = summary['mean'].to_numpy(dtype=float)
    sem_values = summary['sem'].to_numpy(dtype=float)
    valid_mask = np.isfinite(mean_values) & np.isfinite(sem_values)
    if valid_mask.any():
        x_valid = x_pos[valid_mask]
        mean_valid = mean_values[valid_mask]
        sem_valid = sem_values[valid_mask]
        ax.fill_between(x_valid, mean_valid - sem_valid, mean_valid + sem_valid,
                        color='#b3b3b3', alpha=0.3, zorder=1, linewidth=0)
    
    # Plot mean line (matching SANDBOX_4 style)
    ax.errorbar(x_pos, mean_values, yerr=sem_values, fmt='o-', color='#333333',
               linewidth=2.1, markersize=8, capsize=4, label='Mean ± SEM', zorder=3)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_order, rotation=45, ha='right')
    ax.set_xlim(-0.3, len(group_order) - 1 + 0.31)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', axis='y', linestyle=':', linewidth=0.7)
    
    # Right plot: Interaction plot (line plot)
    ax = axes[1]
    
    # Calculate means for each experimental_day × condition combination
    interaction_data = plot_data.groupby(['experimental_day', 'condition'])[dependent_var].agg(['mean', 'sem']).reset_index()
    
    # Plot lines for each condition (matching SANDBOX_4 style)
    for condition in ['No halt', 'Apply halt']:
        condition_data = interaction_data[interaction_data['condition'] == condition]
        color = '#2ca02c' if condition == 'No halt' else '#d62728'  # Green/Red like SANDBOX_4
        
        ax.errorbar(condition_data['experimental_day'], condition_data['mean'],
                   yerr=condition_data['sem'],
                   marker='o', markersize=8, linewidth=2.5,
                   color=color, label=condition, capsize=5,
                   capthick=2)
    
    ax.set_xlabel('Experimental Day')
    ax.set_ylabel(ylabel)
    ax.set_title('Interaction Plot')
    ax.legend(title='Condition', frameon=True)
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot: {filename}")
    plt.close()


def load_per_mouse_turning_timeseries():
    """
    Load per-mouse turning velocity time series from aligned data files.
    Returns dictionary: {day_name: {mouse: {condition: DataFrame with time and velocity}}}
    """
    print("📊 Loading per-mouse turning velocity time series...")
    
    per_mouse_data = {}
    TIME_COLUMN = "Time (s)"
    VELOCITY_COLUMN = "Motor_Velocity"
    
    # Event file suffixes
    EVENT_SUFFIXES = [
        "_Apply halt_2s_right_turns_baselined_data.csv",
        "_Apply halt_2s_left_turns_baselined_data.csv",
        "_No halt_right_turns_baselined_data.csv",
        "_No halt_left_turns_baselined_data.csv",
    ]
    
    def infer_condition(suffix):
        if "no_halt" in suffix.lower() or "no halt" in suffix.lower():
            return "No halt"
        if "apply" in suffix.lower():
            return "Apply halt"
        return "Unknown"
    
    # Load from both day folders
    for day_path, day_name in [(DAY3_PATH, 'day3'), (DAY4_PATH, 'day4')]:
        per_mouse_data[day_name] = {}
        
        # Find aligned data folders (one level up from turning_analysis folder)
        # The structure is: DATA_NEW/CohortX_rotation/Visual_mismatch_dayX/
        # We need to go back and find the actual data folders
        base_data_dirs = [
            BASE_PATH / "Cohort1_rotation" / "Visual_mismatch_day3",
            BASE_PATH / "Cohort3_rotation" / "Visual_mismatch_day3",
        ] if day_name == 'day3' else [
            BASE_PATH / "Cohort1_rotation" / "Visual_mismatch_day4",
            BASE_PATH / "Cohort3_rotation" / "Visual_mismatch_day4",
        ]
        
        for data_dir in base_data_dirs:
            if not data_dir.exists():
                continue
            
            # Find mouse directories
            for mouse_dir in data_dir.iterdir():
                if not mouse_dir.is_dir() or mouse_dir.name.endswith("_processedData"):
                    continue
                
                mouse_id = mouse_dir.name.split("-")[0]
                aligned_dir = mouse_dir / f"{mouse_dir.name}_processedData" / "aligned_data"
                
                if not aligned_dir.exists():
                    continue
                
                if mouse_id not in per_mouse_data[day_name]:
                    per_mouse_data[day_name][mouse_id] = {}
                
                # Load event files for this mouse
                for suffix in EVENT_SUFFIXES:
                    csv_path = aligned_dir / f"{mouse_id}{suffix}"
                    if not csv_path.exists():
                        continue
                    
                    condition = infer_condition(suffix)
                    if condition == "Unknown":
                        continue
                    
                    try:
                        df = pd.read_csv(csv_path)
                        if TIME_COLUMN in df.columns and VELOCITY_COLUMN in df.columns:
                            # Extract time and velocity, use absolute velocity
                            ts = df[[TIME_COLUMN, VELOCITY_COLUMN]].copy()
                            ts = ts.rename(columns={TIME_COLUMN: 'time', VELOCITY_COLUMN: 'velocity'})
                            ts['abs_velocity'] = ts['velocity'].abs()
                            
                            # Average across all turns for this condition
                            if condition not in per_mouse_data[day_name][mouse_id]:
                                per_mouse_data[day_name][mouse_id][condition] = []
                            per_mouse_data[day_name][mouse_id][condition].append(ts)
                    except Exception as e:
                        continue
        
        # Average across all turns for each mouse × condition
        for mouse_id in per_mouse_data[day_name]:
            for condition in per_mouse_data[day_name][mouse_id]:
                if per_mouse_data[day_name][mouse_id][condition]:
                    # Concatenate all turns
                    all_turns = pd.concat(per_mouse_data[day_name][mouse_id][condition], ignore_index=True)
                    # Average by time
                    avg_trace = all_turns.groupby('time')['abs_velocity'].mean().reset_index()
                    avg_trace = avg_trace.rename(columns={'abs_velocity': 'velocity'})
                    per_mouse_data[day_name][mouse_id][condition] = avg_trace
    
    n_mice = sum(len(mice) for mice in per_mouse_data.values())
    print(f"  ✓ Loaded per-mouse traces for {n_mice} mice")
    
    return per_mouse_data


def compute_turning_velocity_difference(turning_data):
    """
    Compute turning velocity difference (Apply halt - No halt) per mouse per experimental day.
    Uses velocity at 1 second after halt (t=1.0s) from per-mouse time series data.
    
    Returns:
    --------
    DataFrame with columns: mouse, experimental_day, velocity_difference
    """
    print("🔄 Computing turning velocity difference (Apply halt - No halt) at t=1.0s...")
    
    # Try to load per-mouse time series
    per_mouse_data = load_per_mouse_turning_timeseries()
    
    diff_records = []
    
    # Extract velocity at t=1.0s for each mouse
    for day_name in ['day3', 'day4']:
        if day_name not in per_mouse_data:
            continue
        
        for mouse_id, mouse_conditions in per_mouse_data[day_name].items():
            apply_halt_trace = mouse_conditions.get('Apply halt')
            no_halt_trace = mouse_conditions.get('No halt')
            
            if apply_halt_trace is not None and no_halt_trace is not None:
                # Extract velocity at t=1.0s
                def get_velocity_at_1s(trace):
                    if trace is None or trace.empty:
                        return None
                    # Find closest time point to 1.0s
                    time_diff = (trace['time'] - 1.0).abs()
                    closest_idx = time_diff.idxmin()
                    closest_time = trace.loc[closest_idx, 'time']
                    
                    if abs(closest_time - 1.0) < 0.1:  # Within 100ms
                        return trace.loc[closest_idx, 'velocity']
                    else:
                        # Interpolate
                        trace_sorted = trace.sort_values('time')
                        if 1.0 >= trace_sorted['time'].min() and 1.0 <= trace_sorted['time'].max():
                            return np.interp(1.0, trace_sorted['time'], trace_sorted['velocity'])
                    return None
                
                apply_vel_1s = get_velocity_at_1s(apply_halt_trace)
                no_halt_vel_1s = get_velocity_at_1s(no_halt_trace)
                
                if pd.notna(apply_vel_1s) and pd.notna(no_halt_vel_1s):
                    diff_val = apply_vel_1s - no_halt_vel_1s
                    diff_records.append({
                        'mouse': mouse_id,
                        'experimental_day': day_name,
                        'velocity_difference': diff_val
                    })
    
    if not diff_records:
        print("  ⚠️ Warning: Could not load per-mouse time series. Falling back to aggregated metric.")
        # Fallback to using aggregated metric
        pivot = turning_data.pivot_table(
            index=['mouse', 'experimental_day'],
            columns='condition',
            values='mean_abs_velocity_0_2s',
            aggfunc='mean'
        ).reset_index()
        
        if 'Apply halt' not in pivot.columns or 'No halt' not in pivot.columns:
            print("  ⚠️ Warning: Missing Apply halt or No halt data")
            return pd.DataFrame()
        
        pivot['velocity_difference'] = pivot['Apply halt'] - pivot['No halt']
        diff_df = pivot[['mouse', 'experimental_day', 'velocity_difference']].copy()
        diff_df = diff_df.dropna(subset=['velocity_difference'])
        print(f"  ✓ Computed differences (using 0-2s mean) for {len(diff_df)} mouse-day combinations")
        print()
        return diff_df
    
    diff_df = pd.DataFrame(diff_records)
    diff_df = diff_df.dropna(subset=['velocity_difference'])
    
    print(f"  ✓ Computed differences at t=1.0s for {len(diff_df)} mouse-day combinations")
    print()
    
    return diff_df


def analyze_turning_velocity_difference(diff_df):
    """
    Analyze turning velocity difference between day3 and day4 using unpaired t-test.
    Includes all mice (not just those with data for both days).
    
    Parameters:
    -----------
    diff_df : DataFrame
        Data with columns: mouse, experimental_day, velocity_difference
    """
    print("\n" + "=" * 80)
    print("TURNING VELOCITY DIFFERENCE ANALYSIS")
    print("(Apply halt - No halt) at t=1.0s for Day 3 vs Day 4")
    print("=" * 80)
    
    if diff_df.empty:
        print("  ⚠️ No difference data available")
        return None
    
    # Descriptive statistics
    print("\nDescriptive Statistics:")
    desc_stats = diff_df.groupby('experimental_day')['velocity_difference'].agg([
        'count', 'mean', 'std', 'sem'
    ]).round(4)
    print(desc_stats)
    
    # Separate day3 and day4 data (include all mice, not just paired)
    day3_data = diff_df[diff_df['experimental_day'] == 'day3']['velocity_difference'].dropna()
    day4_data = diff_df[diff_df['experimental_day'] == 'day4']['velocity_difference'].dropna()
    
    if len(day3_data) == 0 or len(day4_data) == 0:
        print("  ⚠️ Warning: Missing day3 or day4 data")
        return None
    
    print(f"\nUnpaired t-test:")
    print(f"  Day 3: n = {len(day3_data)} mice")
    print(f"  Day 4: n = {len(day4_data)} mice")
    
    # Perform unpaired t-test
    t_stat, p_value = stats.ttest_ind(day3_data, day4_data)
    mean_diff = day4_data.mean() - day3_data.mean()
    
    print(f"  Mean difference (day4 - day3): {mean_diff:.4f} deg/s")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    
    results = {
        'n_day3': len(day3_data),
        'n_day4': len(day4_data),
        'mean_diff_day3': float(day3_data.mean()),
        'mean_diff_day4': float(day4_data.mean()),
        'mean_change': float(mean_diff),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'desc_stats': desc_stats,
        'day3_data': day3_data,
        'day4_data': day4_data
    }
    
    return results


def load_turning_velocity_timeseries():
    """
    Load turning velocity time series data from both experimental days.
    Returns dictionary with time series for each condition × day combination.
    """
    print("📊 Loading turning velocity time series...")
    
    timeseries_data = {}
    
    # Try to load from average traces files if they exist
    for day_path, day_name in [(DAY3_PATH, 'day3'), (DAY4_PATH, 'day4')]:
        # Check for average traces file (from SANDBOX_4 output)
        avg_traces_file = day_path / "turning_velocity_avg_traces.csv"
        
        if avg_traces_file.exists():
            try:
                traces = pd.read_csv(avg_traces_file)
                # Check if it has the expected columns
                if 'time' in traces.columns and 'group' in traces.columns:
                    # Find mean and sem columns (could be named differently)
                    mean_col = None
                    sem_col = None
                    for col in traces.columns:
                        if 'mean' in col.lower() and 'velocity' in col.lower():
                            mean_col = col
                        elif 'sem' in col.lower() and 'velocity' in col.lower():
                            sem_col = col
                    
                    # Fallback to common column names
                    if mean_col is None:
                        if 'mean_velocity' in traces.columns:
                            mean_col = 'mean_velocity'
                        elif 'velocity' in traces.columns:
                            mean_col = 'velocity'
                    
                    if sem_col is None:
                        if 'sem_velocity' in traces.columns:
                            sem_col = 'sem_velocity'
                    
                    if mean_col:
                        for condition in ['No halt', 'Apply halt']:
                            subset = traces[traces['group'] == condition].copy()
                            if not subset.empty:
                                key = f"{day_name}_{condition}"
                                # Use absolute values for turning velocity
                                subset['abs_velocity'] = subset[mean_col].abs()
                                if sem_col and sem_col in subset.columns:
                                    result_df = subset[['time', 'abs_velocity', sem_col]].copy()
                                    result_df = result_df.rename(columns={sem_col: 'sem_velocity'})
                                else:
                                    result_df = subset[['time', 'abs_velocity']].copy()
                                    result_df['sem_velocity'] = 0
                                
                                result_df = result_df.rename(columns={'abs_velocity': 'mean_velocity'})
                                timeseries_data[key] = result_df
                                print(f"  ✓ Loaded {day_name} {condition}: {len(subset)} time points")
            except Exception as e:
                print(f"  ⚠️ Could not load {avg_traces_file}: {e}")
    
    if not timeseries_data:
        print("  ⚠️ No time series data found. Check for turning_velocity_avg_traces.csv in output folders.")
    
    return timeseries_data


def plot_turning_velocity_difference(diff_df, diff_results, turning_data):
    """
    Plot turning velocity difference analysis and grand averages (time series).
    
    Parameters:
    -----------
    diff_df : DataFrame
        Difference data (mouse, experimental_day, velocity_difference)
    diff_results : dict
        Results from analyze_turning_velocity_difference
    turning_data : DataFrame
        Original turning data for grand averages
    """
    if diff_df.empty:
        print("  ⚠️ No data to plot")
        return
    
    # Load time series data
    timeseries_data = load_turning_velocity_timeseries()
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left plot: Velocity difference (day3 vs day4)
    ax = axes[0]
    
    # Assign consistent mouse colors
    mouse_colors = assign_mouse_colors_consistent(diff_df['mouse'].dropna().unique())
    
    # Plot individual mice (all mice, not just paired)
    if diff_results and 'day3_data' in diff_results and 'day4_data' in diff_results:
        day3_data = diff_results['day3_data']
        day4_data = diff_results['day4_data']
        x_pos = np.array([0, 1])
        
        # Get mouse IDs for each day from original dataframe
        day3_df = diff_df[diff_df['experimental_day'] == 'day3']
        day4_df = diff_df[diff_df['experimental_day'] == 'day4']
        
        # Plot all individual data points with jitter
        # Day 3
        for idx, row in day3_df.iterrows():
            val = row['velocity_difference']
            mouse_id = row['mouse']
            if pd.notna(val):
                x_jitter = np.random.normal(0, 0.04)
                ax.scatter(0 + x_jitter, val, marker='o', s=50, alpha=0.65,
                          color=mouse_colors.get(mouse_id, '#1f77b4'), zorder=2)
        
        # Day 4
        for idx, row in day4_df.iterrows():
            val = row['velocity_difference']
            mouse_id = row['mouse']
            if pd.notna(val):
                x_jitter = np.random.normal(0, 0.04)
                ax.scatter(1 + x_jitter, val, marker='o', s=50, alpha=0.65,
                          color=mouse_colors.get(mouse_id, '#1f77b4'), zorder=2)
        
        # Plot group means
        mean_day3 = day3_data.mean()
        mean_day4 = day4_data.mean()
        sem_day3 = day3_data.sem()
        sem_day4 = day4_data.sem()
        
        # Add SEM shading
        ax.fill_between([0, 1], 
                        [mean_day3 - sem_day3, mean_day4 - sem_day4],
                        [mean_day3 + sem_day3, mean_day4 + sem_day4],
                        color='#b3b3b3', alpha=0.3, zorder=1, linewidth=0)
        
        # Plot mean line
        ax.errorbar(x_pos, [mean_day3, mean_day4], 
                   yerr=[sem_day3, sem_day4],
                   fmt='o-', color='#333333', linewidth=2.1, markersize=8,
                   capsize=4, label='Mean ± SEM', zorder=3)
        
        # Add statistics text
        p_val = diff_results['p_value']
        if p_val < 0.001:
            p_text = "p < 0.001***"
        elif p_val < 0.01:
            p_text = f"p = {p_val:.3f}**"
        elif p_val < 0.05:
            p_text = f"p = {p_val:.3f}*"
        else:
            p_text = f"p = {p_val:.3f} ns"
        
        n_day3 = diff_results['n_day3']
        n_day4 = diff_results['n_day4']
        ax.text(0.5, 0.98, f"n = {n_day3} (day3), {n_day4} (day4)\n{p_text}",
               transform=ax.transAxes, ha='center', va='top',
               fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                     edgecolor='grey', alpha=0.8))
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Day 3', 'Day 4'])
    ax.set_xlim(-0.3, 1.31)
    ax.set_ylabel('Velocity Difference (deg/s)\n(Apply halt - No halt)')
    ax.set_title('Turning Velocity Difference\nat t=1.0s: Day 3 vs Day 4')
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, which='both', axis='y', linestyle=':', linewidth=0.7)
    
    # Right plot: Grand averages of absolute turning velocity (time series)
    ax = axes[1]
    
    # Plot time series for each condition × day combination
    plot_order = [
        ('day3', 'No halt', '#2ca02c', 0.6, '--'),
        ('day3', 'Apply halt', '#d62728', 0.6, '--'),
        ('day4', 'No halt', '#2ca02c', 1.0, '-'),
        ('day4', 'Apply halt', '#d62728', 1.0, '-'),
    ]
    
    if timeseries_data:
        # Plot time series
        for day, condition, color, alpha, linestyle in plot_order:
            key = f"{day}_{condition}"
            if key in timeseries_data:
                ts = timeseries_data[key]
                ax.plot(ts['time'], ts['mean_velocity'], 
                       color=color, linewidth=1.5, alpha=alpha, linestyle=linestyle,
                       label=f'{day} - {condition}')
                
                # Add SEM shading
                if 'sem_velocity' in ts.columns:
                    upper = ts['mean_velocity'] + ts['sem_velocity'].fillna(0)
                    lower = ts['mean_velocity'] - ts['sem_velocity'].fillna(0)
                    ax.fill_between(ts['time'], lower, upper, 
                                   color=color, alpha=0.15)
    else:
        # Fallback: show message if no time series data
        ax.text(0.5, 0.5, 'Time series data not available\n(Check for turning_velocity_avg_traces.csv)',
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
    
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Turning Velocity (deg/s)')
    ax.set_title('Grand Average Turning Velocity\n(Time Series)')
    ax.legend(fontsize=10, loc='best', frameon=True)
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "turning_velocity_difference_analysis.pdf", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot: turning_velocity_difference_analysis.pdf")
    plt.close()


def main():
    """Main analysis pipeline."""
    
    # Print exclusion information
    if EXCLUDED_MICE_EYE_TRACKING:
        print("\n" + "⚠️  " * 20)
        print(f"NOTE: Excluding mice from ALL eye tracking metrics: {', '.join(EXCLUDED_MICE_EYE_TRACKING)}")
        print("These mice will be INCLUDED in running and turning velocity analyses.")
        print("⚠️  " * 20 + "\n")
    
    # 1. Load and combine data
    print("\n" + "=" * 80)
    print("STEP 1: LOADING AND COMBINING DATA")
    print("=" * 80 + "\n")
    
    saccade_pupil_data = load_and_combine_saccade_pupil_metrics()
    running_data = load_and_combine_running_metrics()
    turning_data = load_and_combine_turning_metrics()
    baseline_saccade_data = load_and_combine_baseline_saccade()
    
    # Save combined datasets
    saccade_pupil_data.to_csv(OUTPUT_DIR / "combined_saccade_pupil_metrics.csv", index=False)
    running_data.to_csv(OUTPUT_DIR / "combined_running_metrics.csv", index=False)
    turning_data.to_csv(OUTPUT_DIR / "combined_turning_metrics.csv", index=False)
    if baseline_saccade_data is not None:
        baseline_saccade_data.to_csv(OUTPUT_DIR / "combined_baseline_saccade_density.csv", index=False)
    
    print("\n✓ All combined datasets saved to:", OUTPUT_DIR)
    
    # 2. Perform two-way ANOVAs
    print("\n" + "=" * 80)
    print("STEP 2: TWO-WAY ANOVA ANALYSES")
    print("=" * 80)
    
    all_results = []
    
    # Pupil diameter analyses
    print("\n" + "-" * 80)
    print("PUPIL DIAMETER METRICS")
    print("-" * 80)
    
    pupil_post_results = perform_two_way_anova(
        saccade_pupil_data, 
        'pupil_diameter_post_halt',
        'Pupil Diameter Post-Halt (mean over 0-2s)'
    )
    if pupil_post_results:
        all_results.append(pupil_post_results)
        plot_two_way_interaction(
            saccade_pupil_data,
            'pupil_diameter_post_halt',
            'Pupil Diameter Post-Halt',
            'Pupil Diameter (a.u.)',
            'pupil_diameter_post_halt_two_way_anova.pdf'
        )
    
    pupil_peak_results = perform_two_way_anova(
        saccade_pupil_data,
        'pupil_diameter_peak',
        'Pupil Diameter Peak (max over 0-1s)'
    )
    if pupil_peak_results:
        all_results.append(pupil_peak_results)
        plot_two_way_interaction(
            saccade_pupil_data,
            'pupil_diameter_peak',
            'Pupil Diameter Peak',
            'Pupil Diameter (a.u.)',
            'pupil_diameter_peak_two_way_anova.pdf'
        )
    
    # Saccade probability analyses
    print("\n" + "-" * 80)
    print("SACCADE PROBABILITY METRICS")
    print("-" * 80)
    
    saccade_post_results = perform_two_way_anova(
        saccade_pupil_data,
        'saccade_probability_post_halt',
        'Saccade Probability Post-Halt (mean over 0-2s)'
    )
    if saccade_post_results:
        all_results.append(saccade_post_results)
        plot_two_way_interaction(
            saccade_pupil_data,
            'saccade_probability_post_halt',
            'Saccade Probability Post-Halt',
            'Saccade Probability',
            'saccade_probability_post_halt_two_way_anova.pdf'
        )
    
    saccade_peak_results = perform_two_way_anova(
        saccade_pupil_data,
        'saccade_probability_peak',
        'Saccade Probability Peak (max over 0-1s)'
    )
    if saccade_peak_results:
        all_results.append(saccade_peak_results)
        plot_two_way_interaction(
            saccade_pupil_data,
            'saccade_probability_peak',
            'Saccade Probability Peak',
            'Saccade Probability',
            'saccade_probability_peak_two_way_anova.pdf'
        )
    
    # Running velocity analysis (mean absolute velocity after halt)
    print("\n" + "-" * 80)
    print("RUNNING VELOCITY METRICS")
    print("-" * 80)
    
    running_post_results = perform_two_way_anova(
        running_data,
        'running_velocity_post_halt',
        'Running Velocity Post-Halt (mean over 0-2s)'
    )
    if running_post_results:
        all_results.append(running_post_results)
        plot_two_way_interaction(
            running_data,
            'running_velocity_post_halt',
            'Running Velocity Post-Halt (Absolute)',
            'Running Velocity (cm/s)',
            'running_velocity_post_halt_two_way_anova.pdf'
        )
    
    running_peak_results = perform_two_way_anova(
        running_data,
        'running_velocity_peak_1s',
        'Running Velocity Peak (max absolute over 0-1s)'
    )
    if running_peak_results:
        all_results.append(running_peak_results)
        plot_two_way_interaction(
            running_data,
            'running_velocity_peak_1s',
            'Running Velocity Peak (Absolute)',
            'Peak Velocity (cm/s)',
            'running_velocity_peak_two_way_anova.pdf'
        )
    
    # Turning velocity analysis (mean absolute velocity 0-2s)
    print("\n" + "-" * 80)
    print("TURNING VELOCITY METRICS")
    print("-" * 80)
    
    turning_results = perform_two_way_anova(
        turning_data,
        'mean_abs_velocity_0_2s',
        'Turning Velocity (mean absolute over 0-2s)'
    )
    if turning_results:
        all_results.append(turning_results)
        plot_two_way_interaction(
            turning_data,
            'mean_abs_velocity_0_2s',
            'Turning Velocity (Absolute)',
            'Turning Velocity (deg/s)',
            'turning_velocity_two_way_anova.pdf'
        )
    
    # Turning velocity difference analysis (Apply halt - No halt)
    print("\n" + "-" * 80)
    print("TURNING VELOCITY DIFFERENCE")
    print("-" * 80)
    
    turning_diff_df = compute_turning_velocity_difference(turning_data)
    if not turning_diff_df.empty:
        # Save difference data
        turning_diff_df.to_csv(OUTPUT_DIR / "turning_velocity_difference_per_mouse.csv", index=False)
        print("✓ Saved: turning_velocity_difference_per_mouse.csv")
        
        # Analyze difference
        turning_diff_results = analyze_turning_velocity_difference(turning_diff_df)
        
        # Plot difference and grand averages
        if turning_diff_results:
            plot_turning_velocity_difference(turning_diff_df, turning_diff_results, turning_data)
            
            # Save difference statistics
            diff_stats = pd.DataFrame([{
                'metric': 'Turning Velocity Difference (Apply halt - No halt) at t=1.0s',
                'test_type': 'Unpaired t-test',
                'time_point': '1.0s post-halt',
                'n_day3': turning_diff_results['n_day3'],
                'n_day4': turning_diff_results['n_day4'],
                'mean_diff_day3': turning_diff_results['mean_diff_day3'],
                'mean_diff_day4': turning_diff_results['mean_diff_day4'],
                'mean_change_day4_minus_day3': turning_diff_results['mean_change'],
                't_statistic': turning_diff_results['t_statistic'],
                'p_value': turning_diff_results['p_value'],
                'significant': 'Yes' if turning_diff_results['p_value'] < 0.05 else 'No'
            }])
            diff_stats.to_csv(OUTPUT_DIR / "turning_velocity_difference_stats.csv", index=False)
            print("✓ Saved: turning_velocity_difference_stats.csv")
    
    # Baseline saccade density (if available)
    if baseline_saccade_data is not None and 'condition' in baseline_saccade_data.columns:
        print("\n" + "-" * 80)
        print("BASELINE SACCADE DENSITY")
        print("-" * 80)
        
        baseline_results = perform_two_way_anova(
            baseline_saccade_data,
            'baseline_saccade_probability',
            'Baseline Saccade Probability'
        )
        if baseline_results:
            all_results.append(baseline_results)
            plot_two_way_interaction(
                baseline_saccade_data,
                'baseline_saccade_probability',
                'Baseline Saccade Probability',
                'Saccade Probability',
                'baseline_saccade_probability_two_way_anova.pdf'
            )
    
    # 3. Save summary of all ANOVA results
    print("\n" + "=" * 80)
    print("STEP 3: SAVING SUMMARY RESULTS")
    print("=" * 80 + "\n")
    
    summary_rows = []
    for result in all_results:
        if result is not None:
            summary_rows.append({
                'Metric': result['title'],
                'Variable': result['dependent_var'],
                'p_experimental_day': result['p_day'],
                'p_condition': result['p_condition'],
                'p_interaction': result['p_interaction'],
                'sig_day': '***' if result['p_day'] < 0.001 else '**' if result['p_day'] < 0.01 else '*' if result['p_day'] < 0.05 else 'ns',
                'sig_condition': '***' if result['p_condition'] < 0.001 else '**' if result['p_condition'] < 0.01 else '*' if result['p_condition'] < 0.05 else 'ns',
                'sig_interaction': '***' if result['p_interaction'] < 0.001 else '**' if result['p_interaction'] < 0.01 else '*' if result['p_interaction'] < 0.05 else 'ns'
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "two_way_anova_summary.csv", index=False)
    print("✓ Saved ANOVA summary to: two_way_anova_summary.csv")
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - combined_*.csv (combined datasets)")
    print("  - *_two_way_anova.pdf (interaction plots)")
    print("  - two_way_anova_summary.csv (summary table)")
    print("  - turning_velocity_difference_per_mouse.csv (difference data)")
    print("  - turning_velocity_difference_stats.csv (difference statistics)")
    print("  - turning_velocity_difference_analysis.pdf (difference plot + grand averages)")
    print()


if __name__ == "__main__":
    main()

