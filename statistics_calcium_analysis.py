"""
Compare mean_fluorescence_2_to_8s for Visual Mismatch Day 3 and Day 4
between Cohort1 and Cohort3. 
- Performs separate comparisons for Day 3 and Day 4
- Also performs combined Day 3 & 4 comparison
Performs t-tests and saves statistics for all comparisons
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import OrderedDict

# Set plotting style (matching project style)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "legend.title_fontsize": 20,
})
sns.set_theme(style="whitegrid")

# File paths (same files contain both day3 and day4 data, filtered by condition column)
cohort1_file = "/Volumes/RanczLab2/DATA_NEW/analysis results/calcium_analysis-cohort1/Cohort1_Visual_mismatch_day3_mean_fluorescence_2_to_8s_paired_condition_comparison_plot_data.csv"
cohort3_file = "/Volumes/RanczLab2/DATA_NEW/analysis results/calcium_analysis_cohort3/Cohort3_Visual_mismatch_day3_mean_fluorescence_2_to_8s_unpaired_condition_comparison_plot_data.csv"

# Output directory
output_dir = "/Volumes/RanczLab2/DATA_NEW/analysis results"
os.makedirs(output_dir, exist_ok=True)


def plot_cohort_comparison(cohort1_values, cohort3_values, cohort1_mice, cohort3_mice,
                          cohort1_mean, cohort3_mean, cohort1_sem, cohort3_sem,
                          p_value, output_path, title=None):
    """
    Create a comparison plot showing individual data points and group means.
    
    Parameters:
    -----------
    cohort1_values : array-like
        Values for Cohort1
    cohort3_values : array-like
        Values for Cohort3
    cohort1_mice : array-like
        Mouse IDs for Cohort1
    cohort3_mice : array-like
        Mouse IDs for Cohort3
    cohort1_mean : float
        Mean for Cohort1
    cohort3_mean : float
        Mean for Cohort3
    cohort1_sem : float
        SEM for Cohort1
    cohort3_sem : float
        SEM for Cohort3
    p_value : float
        p-value from t-test
    output_path : str
        Path to save the plot
    title : str, optional
        Custom title for the plot. If None, uses default title.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Assign consistent colors to mice (matching project style)
    all_mice = list(cohort1_mice) + list(cohort3_mice)
    unique_mice = sorted(dict.fromkeys(all_mice))
    if unique_mice:
        palette = sns.color_palette("gnuplot2", len(unique_mice))
        mouse_colors = OrderedDict((mouse, palette[idx]) for idx, mouse in enumerate(unique_mice))
    else:
        mouse_colors = {}
    
    x_pos = np.array([0, 1])
    x_labels = ['C1', 'C3']
    
    # Plot individual data points with jitter
    np.random.seed(42)  # For reproducible jitter
    
    # Cohort1 points
    for i, (val, mouse) in enumerate(zip(cohort1_values, cohort1_mice)):
        x_jitter = np.random.normal(0, 0.04)
        ax.scatter(0 + x_jitter, val, marker='o', s=50, alpha=0.65,
                  color=mouse_colors.get(mouse, '#1f77b4'), zorder=2)
    
    # Cohort3 points
    for i, (val, mouse) in enumerate(zip(cohort3_values, cohort3_mice)):
        x_jitter = np.random.normal(0, 0.04)
        ax.scatter(1 + x_jitter, val, marker='o', s=50, alpha=0.65,
                  color=mouse_colors.get(mouse, '#1f77b4'), zorder=2)
    
    # Add SEM shading
    ax.fill_between([0, 1],
                    [cohort1_mean - cohort1_sem, cohort3_mean - cohort3_sem],
                    [cohort1_mean + cohort1_sem, cohort3_mean + cohort3_sem],
                    color='#b3b3b3', alpha=0.3, zorder=1, linewidth=0)
    
    # Plot mean line with error bars
    ax.errorbar(x_pos, [cohort1_mean, cohort3_mean],
               yerr=[cohort1_sem, cohort3_sem],
               fmt='o-', color='#333333', linewidth=2.1, markersize=8,
               capsize=4, label='Mean ± SEM', zorder=3)
    
    # Calculate dynamic y-axis limits based on data range
    all_values = np.concatenate([cohort1_values, cohort3_values])
    # Include error bars in range calculation
    data_min = min(np.min(all_values), 
                   cohort1_mean - cohort1_sem, 
                   cohort3_mean - cohort3_sem)
    data_max = max(np.max(all_values), 
                   cohort1_mean + cohort1_sem, 
                   cohort3_mean + cohort3_sem)
    
    # Add padding (15% of range on each side, but at least some minimum padding)
    data_range = data_max - data_min
    if data_range > 0:
        padding = max(data_range * 0.15, abs(data_max) * 0.05)  # At least 5% of max value
    else:
        padding = abs(data_max) * 0.1 if data_max != 0 else 0.1
    
    y_min = data_min - padding
    y_max = data_max + padding
    
    # Ensure y_min is not too negative if all data is positive
    if data_min >= 0 and y_min < 0:
        y_min = -data_max * 0.05  # Small negative padding for positive-only data
    
    # Add statistics text
    if p_value < 0.001:
        p_text = "p < 0.001***"
    elif p_value < 0.01:
        p_text = f"p = {p_value:.3f}**"
    elif p_value < 0.05:
        p_text = f"p = {p_value:.3f}*"
    else:
        p_text = f"p = {p_value:.3f} ns"
    
    n_cohort1 = len(cohort1_values)
    n_cohort3 = len(cohort3_values)
    ax.text(0.5, 0.98, f"n = {n_cohort1} (Cohort1), {n_cohort3} (Cohort3)\n{p_text}",
           transform=ax.transAxes, ha='center', va='top',
           fontsize=20, bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                 edgecolor='grey', alpha=0.8))
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=20)
    ax.set_xlim(-0.3, 1.31)
    ax.set_ylabel('MeanFluo', fontsize=20)
    if title is None:
        ax.set_title('Mean Fluorescence (2-8s)', fontsize=20)
    else:
        ax.set_title(title, fontsize=20)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which='both', axis='y', linestyle=':', linewidth=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot: {Path(output_path).name}")
    plt.close()


# Load data
print("Loading data...")
print("=" * 80)

# Load data from files (same files contain both day3 and day4 data)
print("\nLoading data files...")
df_cohort1 = pd.read_csv(cohort1_file)
df_cohort3 = pd.read_csv(cohort3_file)

# Check available conditions
print(f"\nAvailable conditions in Cohort1 file:")
print(df_cohort1['condition'].unique())
print(f"\nAvailable conditions in Cohort3 file:")
print(df_cohort3['condition'].unique())

# Filter for Apply_halt condition - try different possible condition names
# Common variations: 'Apply_halt_day3', 'Apply halt day3', 'Apply_halt_day4', 'Apply halt day4'
cohort1_day3 = df_cohort1[df_cohort1['condition'].str.contains('Apply.*halt.*day3', case=False, na=False)].copy()
cohort3_day3 = df_cohort3[df_cohort3['condition'].str.contains('Apply.*halt.*day3', case=False, na=False)].copy()

cohort1_day4 = df_cohort1[df_cohort1['condition'].str.contains('Apply.*halt.*day4', case=False, na=False)].copy()
cohort3_day4 = df_cohort3[df_cohort3['condition'].str.contains('Apply.*halt.*day4', case=False, na=False)].copy()

# Get unique values per mouse for day3 (in case of duplicates)
cohort1_day3_unique = cohort1_day3.groupby('mouse')['value'].first().reset_index()
cohort1_day3_unique['experimental_day'] = 'day3'
cohort3_day3_unique = cohort3_day3.groupby('mouse')['value'].first().reset_index()
cohort3_day3_unique['experimental_day'] = 'day3'

print(f"\n  Cohort1 Day3: {len(cohort1_day3_unique)} mice")
print(f"  Cohort3 Day3: {len(cohort3_day3_unique)} mice")

# Get unique values per mouse for day4 (in case of duplicates)
cohort1_day4_unique = cohort1_day4.groupby('mouse')['value'].first().reset_index()
cohort1_day4_unique['experimental_day'] = 'day4'
cohort3_day4_unique = cohort3_day4.groupby('mouse')['value'].first().reset_index()
cohort3_day4_unique['experimental_day'] = 'day4'

print(f"  Cohort1 Day4: {len(cohort1_day4_unique)} mice")
print(f"  Cohort3 Day4: {len(cohort3_day4_unique)} mice")

# Combine Day 3 and Day 4 data per cohort (include all values, not just unique mice)
print("\nCombining Day 3 and Day 4 data per cohort...")
cohort1_combined = pd.concat([cohort1_day3_unique, cohort1_day4_unique], ignore_index=True)
cohort3_combined = pd.concat([cohort3_day3_unique, cohort3_day4_unique], ignore_index=True)

# Extract values for t-test (combined cohorts - includes all data points from both days)
cohort1_values = cohort1_combined['value'].values
cohort3_values = cohort3_combined['value'].values

# Get unique mice for plotting (to assign consistent colors)
cohort1_combined_unique = cohort1_combined.groupby('mouse')['value'].first().reset_index()
cohort3_combined_unique = cohort3_combined.groupby('mouse')['value'].first().reset_index()

print(f"\nCohort1 Combined (Day3+Day4) - Total data points: {len(cohort1_values)}")
print(f"  Unique mice: {len(cohort1_combined_unique)}")
print(f"  Mean: {np.mean(cohort1_values):.6f}, Std: {np.std(cohort1_values, ddof=1):.6f}")

print(f"\nCohort3 Combined (Day3+Day4) - Total data points: {len(cohort3_values)}")
print(f"  Unique mice: {len(cohort3_combined_unique)}")
print(f"  Mean: {np.mean(cohort3_values):.6f}, Std: {np.std(cohort3_values, ddof=1):.6f}")

# Also show breakdown by day for reference
print("\nBreakdown by day:")
print(f"  Cohort1 Day3: n={len(cohort1_day3_unique)}, mean={np.mean(cohort1_day3_unique['value']):.6f}")
print(f"  Cohort1 Day4: n={len(cohort1_day4_unique)}, mean={np.mean(cohort1_day4_unique['value']):.6f}")
print(f"  Cohort3 Day3: n={len(cohort3_day3_unique)}, mean={np.mean(cohort3_day3_unique['value']):.6f}")
print(f"  Cohort3 Day4: n={len(cohort3_day4_unique)}, mean={np.mean(cohort3_day4_unique['value']):.6f}")

# ============================================================================
# DAY 3 SEPARATE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("DAY 3 SEPARATE ANALYSIS")
print("=" * 80)

cohort1_day3_values = cohort1_day3_unique['value'].values
cohort3_day3_values = cohort3_day3_unique['value'].values

print(f"\nCohort1 Day3 - n={len(cohort1_day3_values)}")
print(f"  Mean: {np.mean(cohort1_day3_values):.6f}, Std: {np.std(cohort1_day3_values, ddof=1):.6f}")

print(f"\nCohort3 Day3 - n={len(cohort3_day3_values)}")
print(f"  Mean: {np.mean(cohort3_day3_values):.6f}, Std: {np.std(cohort3_day3_values, ddof=1):.6f}")

mean_diff_day3 = np.mean(cohort1_day3_values) - np.mean(cohort3_day3_values)
print(f"\nMean difference Day3 (Cohort1 - Cohort3): {mean_diff_day3:.6f}")

# Perform t-tests for Day 3
t_stat_day3, p_value_day3 = stats.ttest_ind(cohort1_day3_values, cohort3_day3_values)
t_stat_welch_day3, p_value_welch_day3 = stats.ttest_ind(cohort1_day3_values, cohort3_day3_values, equal_var=False)

# Calculate effect size for Day 3
pooled_std_day3 = np.sqrt(((len(cohort1_day3_values) - 1) * np.var(cohort1_day3_values, ddof=1) + 
                           (len(cohort3_day3_values) - 1) * np.var(cohort3_day3_values, ddof=1)) / 
                          (len(cohort1_day3_values) + len(cohort3_day3_values) - 2))
cohens_d_day3 = mean_diff_day3 / pooled_std_day3 if pooled_std_day3 > 0 else 0

print(f"\n=== Day 3 T-Test Results ===")
print(f"Standard t-test:")
print(f"  t-statistic: {t_stat_day3:.6f}")
print(f"  p-value: {p_value_day3:.6f}")
print(f"  Significant (p < 0.05): {'Yes' if p_value_day3 < 0.05 else 'No'}")

print(f"\nWelch's t-test (unequal variances):")
print(f"  t-statistic: {t_stat_welch_day3:.6f}")
print(f"  p-value: {p_value_welch_day3:.6f}")
print(f"  Significant (p < 0.05): {'Yes' if p_value_welch_day3 < 0.05 else 'No'}")

print(f"\nEffect size (Cohen's d): {cohens_d_day3:.6f}")

# ============================================================================
# DAY 4 SEPARATE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("DAY 4 SEPARATE ANALYSIS")
print("=" * 80)

cohort1_day4_values = cohort1_day4_unique['value'].values
cohort3_day4_values = cohort3_day4_unique['value'].values

print(f"\nCohort1 Day4 - n={len(cohort1_day4_values)}")
print(f"  Mean: {np.mean(cohort1_day4_values):.6f}, Std: {np.std(cohort1_day4_values, ddof=1):.6f}")

print(f"\nCohort3 Day4 - n={len(cohort3_day4_values)}")
print(f"  Mean: {np.mean(cohort3_day4_values):.6f}, Std: {np.std(cohort3_day4_values, ddof=1):.6f}")

mean_diff_day4 = np.mean(cohort1_day4_values) - np.mean(cohort3_day4_values)
print(f"\nMean difference Day4 (Cohort1 - Cohort3): {mean_diff_day4:.6f}")

# Perform t-tests for Day 4
t_stat_day4, p_value_day4 = stats.ttest_ind(cohort1_day4_values, cohort3_day4_values)
t_stat_welch_day4, p_value_welch_day4 = stats.ttest_ind(cohort1_day4_values, cohort3_day4_values, equal_var=False)

# Calculate effect size for Day 4
pooled_std_day4 = np.sqrt(((len(cohort1_day4_values) - 1) * np.var(cohort1_day4_values, ddof=1) + 
                           (len(cohort3_day4_values) - 1) * np.var(cohort3_day4_values, ddof=1)) / 
                          (len(cohort1_day4_values) + len(cohort3_day4_values) - 2))
cohens_d_day4 = mean_diff_day4 / pooled_std_day4 if pooled_std_day4 > 0 else 0

print(f"\n=== Day 4 T-Test Results ===")
print(f"Standard t-test:")
print(f"  t-statistic: {t_stat_day4:.6f}")
print(f"  p-value: {p_value_day4:.6f}")
print(f"  Significant (p < 0.05): {'Yes' if p_value_day4 < 0.05 else 'No'}")

print(f"\nWelch's t-test (unequal variances):")
print(f"  t-statistic: {t_stat_welch_day4:.6f}")
print(f"  p-value: {p_value_welch_day4:.6f}")
print(f"  Significant (p < 0.05): {'Yes' if p_value_welch_day4 < 0.05 else 'No'}")

print(f"\nEffect size (Cohen's d): {cohens_d_day4:.6f}")

# ============================================================================
# COMBINED DAY 3 + DAY 4 ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("COMBINED DAY 3 + DAY 4 ANALYSIS")
print("=" * 80)

# Calculate difference
mean_diff = np.mean(cohort1_values) - np.mean(cohort3_values)
print(f"\nMean difference (Cohort1 - Cohort3): {mean_diff:.6f}")

# Perform t-test (independent samples, two-tailed)
t_stat, p_value = stats.ttest_ind(cohort1_values, cohort3_values)

# Also perform Welch's t-test (unequal variances)
t_stat_welch, p_value_welch = stats.ttest_ind(cohort1_values, cohort3_values, equal_var=False)

# Calculate effect size (Cohen's d)
pooled_std = np.sqrt(((len(cohort1_values) - 1) * np.var(cohort1_values, ddof=1) + 
                      (len(cohort3_values) - 1) * np.var(cohort3_values, ddof=1)) / 
                     (len(cohort1_values) + len(cohort3_values) - 2))
cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

print(f"\n=== T-Test Results ===")
print(f"Standard t-test:")
print(f"  t-statistic: {t_stat:.6f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Significant (p < 0.05): {'Yes' if p_value < 0.05 else 'No'}")

print(f"\nWelch's t-test (unequal variances):")
print(f"  t-statistic: {t_stat_welch:.6f}")
print(f"  p-value: {p_value_welch:.6f}")
print(f"  Significant (p < 0.05): {'Yes' if p_value_welch < 0.05 else 'No'}")

print(f"\nEffect size (Cohen's d): {cohens_d:.6f}")

# Create summary statistics dataframe for all comparisons
summary_stats = pd.DataFrame({
    'comparison': [
        'Cohort1 vs Cohort3 - Day 3',
        'Cohort1 vs Cohort3 - Day 4',
        'Cohort1 vs Cohort3 - Combined Day3+Day4'
    ],
    'cohort1_n': [
        len(cohort1_day3_values),
        len(cohort1_day4_values),
        len(cohort1_values)
    ],
    'cohort1_mean': [
        np.mean(cohort1_day3_values),
        np.mean(cohort1_day4_values),
        np.mean(cohort1_values)
    ],
    'cohort1_std': [
        np.std(cohort1_day3_values, ddof=1),
        np.std(cohort1_day4_values, ddof=1),
        np.std(cohort1_values, ddof=1)
    ],
    'cohort1_sem': [
        stats.sem(cohort1_day3_values),
        stats.sem(cohort1_day4_values),
        stats.sem(cohort1_values)
    ],
    'cohort3_n': [
        len(cohort3_day3_values),
        len(cohort3_day4_values),
        len(cohort3_values)
    ],
    'cohort3_mean': [
        np.mean(cohort3_day3_values),
        np.mean(cohort3_day4_values),
        np.mean(cohort3_values)
    ],
    'cohort3_std': [
        np.std(cohort3_day3_values, ddof=1),
        np.std(cohort3_day4_values, ddof=1),
        np.std(cohort3_values, ddof=1)
    ],
    'cohort3_sem': [
        stats.sem(cohort3_day3_values),
        stats.sem(cohort3_day4_values),
        stats.sem(cohort3_values)
    ],
    'mean_difference': [mean_diff_day3, mean_diff_day4, mean_diff],
    't_statistic': [t_stat_day3, t_stat_day4, t_stat],
    'p_value': [p_value_day3, p_value_day4, p_value],
    't_statistic_welch': [t_stat_welch_day3, t_stat_welch_day4, t_stat_welch],
    'p_value_welch': [p_value_welch_day3, p_value_welch_day4, p_value_welch],
    'cohens_d': [cohens_d_day3, cohens_d_day4, cohens_d],
    'significant_05': [
        p_value_day3 < 0.05,
        p_value_day4 < 0.05,
        p_value < 0.05
    ],
    'significant_01': [
        p_value_day3 < 0.01,
        p_value_day4 < 0.01,
        p_value < 0.01
    ],
    'significant_001': [
        p_value_day3 < 0.001,
        p_value_day4 < 0.001,
        p_value < 0.001
    ]
})

# Create detailed results dataframe with individual mouse data (including day info)
detailed_results = pd.DataFrame({
    'cohort': ['Cohort1'] * len(cohort1_combined) + ['Cohort3'] * len(cohort3_combined),
    'mouse': list(cohort1_combined['mouse']) + list(cohort3_combined['mouse']),
    'experimental_day': list(cohort1_combined['experimental_day']) + list(cohort3_combined['experimental_day']),
    'main_peak_residual_auc': list(cohort1_combined['value']) + list(cohort3_combined['value'])
})

# Save results
stats_output_file = os.path.join(output_dir, "cohort1_vs_cohort3_all_days_ttest_statistics_MEANFLUO2-8.csv")
detailed_output_file = os.path.join(output_dir, "cohort1_vs_cohort3_combined_day3+day4_detailed_data_MEANFLUO2-8.csv")

summary_stats.to_csv(stats_output_file, index=False)
detailed_results.to_csv(detailed_output_file, index=False)

# Create plots for Day 3
print("\n=== Creating Day 3 Plot ===")
plot_output_day3 = os.path.join(output_dir, "cohort1_vs_cohort3_day3_comparison_MEANFLUO2-8.pdf")
cohort1_mean_day3 = np.mean(cohort1_day3_values)
cohort3_mean_day3 = np.mean(cohort3_day3_values)
cohort1_sem_day3 = stats.sem(cohort1_day3_values)
cohort3_sem_day3 = stats.sem(cohort3_day3_values)

plot_cohort_comparison(
    cohort1_day3_values, cohort3_day3_values,
    cohort1_day3_unique['mouse'].values, cohort3_day3_unique['mouse'].values,
    cohort1_mean_day3, cohort3_mean_day3, cohort1_sem_day3, cohort3_sem_day3,
    p_value_day3, plot_output_day3,
    title='Mean Fluorescence (2-8s) - Day 3'
)

# Create plots for Day 4
print("\n=== Creating Day 4 Plot ===")
plot_output_day4 = os.path.join(output_dir, "cohort1_vs_cohort3_day4_comparison_MEANFLUO2-8.pdf")
cohort1_mean_day4 = np.mean(cohort1_day4_values)
cohort3_mean_day4 = np.mean(cohort3_day4_values)
cohort1_sem_day4 = stats.sem(cohort1_day4_values)
cohort3_sem_day4 = stats.sem(cohort3_day4_values)

plot_cohort_comparison(
    cohort1_day4_values, cohort3_day4_values,
    cohort1_day4_unique['mouse'].values, cohort3_day4_unique['mouse'].values,
    cohort1_mean_day4, cohort3_mean_day4, cohort1_sem_day4, cohort3_sem_day4,
    p_value_day4, plot_output_day4,
    title='Mean Fluorescence (2-8s) - Day 4'
)

# Create combined plot
print("\n=== Creating Combined Day 3+4 Plot ===")
plot_output_combined = os.path.join(output_dir, "cohort1_vs_cohort3_combined_day3+day4_comparison_MEANFLUO2-8.pdf")
cohort1_mean = np.mean(cohort1_values)
cohort3_mean = np.mean(cohort3_values)
cohort1_sem = stats.sem(cohort1_values)
cohort3_sem = stats.sem(cohort3_values)

plot_cohort_comparison(
    cohort1_values, cohort3_values,
    cohort1_combined['mouse'].values, cohort3_combined['mouse'].values,
    cohort1_mean, cohort3_mean, cohort1_sem, cohort3_sem,
    p_value, plot_output_combined
)

print(f"\n=== Results Saved ===")
print(f"Summary statistics (all days) saved to: {stats_output_file}")
print(f"Detailed data saved to: {detailed_output_file}")
print(f"Day 3 plot saved to: {plot_output_day3}")
print(f"Day 4 plot saved to: {plot_output_day4}")
print(f"Combined plot saved to: {plot_output_combined}")

