# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: aeon
#     language: python
#     name: python3
# ---

# %%
#---------------------------------------------------------------------------------------------------#
# IMPORTS
#---------------------------------------------------------------------------------------------------#
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import cm
import plotly.io as pio
import plotly.express as px
import plotly.subplots as sp
import math
import re
from pprint import pprint
import pickle
from plotly.subplots import make_subplots
from scipy.stats import mode, pearsonr, norm, ttest_rel, ttest_ind, friedmanchisquare, wilcoxon
from scipy.integrate import cumulative_trapezoid
from scipy.signal import correlate, find_peaks
from scipy.optimize import curve_fit
import gc  # garbage collector for removing large variables from memory instantly 
import importlib  # for force updating changed packages 
import seaborn as sns
import warnings
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Tuple, Iterable, List
warnings.filterwarnings('ignore')

try:
    from statsmodels.stats.anova import AnovaRM, anova_lm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multitest import multipletests
    STATS_MODELS_AVAILABLE = True
except ImportError:
    STATS_MODELS_AVAILABLE = False
    AnovaRM = None  # type: ignore
    anova_lm = None  # type: ignore
    ols = None  # type: ignore
    multipletests = None  # type: ignore
    print("⚠️ statsmodels not available. Repeated measures ANOVA will fall back to non-parametric tests.")

# Interactive widgets for dropdowns
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    print("⚠️ ipywidgets not available. Install with: pip install ipywidgets")
    WIDGETS_AVAILABLE = False
# %config Completer.use_jedi = False  # Fixes autocomplete issues
# %config InlineBackend.figure_format = 'retina'  # Improves plot resolution

# %% [markdown]
# # Aligned Data Analysis Notebook
#
# Analyze aligned `.csv` files generated in **SANDBOX_2** to produce mouse-level averages, grand averages with SEM, and a suite of post-alignment calcium signal metrics. The notebook also supports interactive plotting and the new halt vs no-halt comparison workflow.
#
# ## What this notebook does
# - Loads aligned data for your selected cohort(s) and mice
# - Computes per-mouse mean ± SEM traces for key columns
# - Builds grand averages with SEM across mice
# - Extracts signal features (peak, onset, decay tau) for `z_470_Baseline` and `z_560_Baseline`
# - Calculates the 2–8 s post-alignment mean for `z_560_Baseline`
# - Optionally saves per-mouse and cohort-level summary CSVs
# - Provides interactive grand-average plotting widgets (when `ipywidgets` is available)
# - Compares halt vs no-halt conditions with paired statistics and plotting
#
# ## How to use it
# 1. Review **Configuration** and update cohort, mouse selection, event names, and data directories.
# 2. Run the **Data Loading** cell to load aligned data.
# 3. Execute the **Analysis** cells to compute metrics and grand averages.
# 4. (Optional) Enable saving flags to export CSVs with per-mouse metrics and grand averages.
# 5. Use the **Interactive Grand Average Plotting** cell to explore previously saved grand-average CSVs.
# 6. Run the **Halt vs No Halt Condition Comparison** section to generate paired plots, statistics, and export CSV summaries per metric.
#
# ## Outputs
# - **Grand averages CSV:** time-point means plus SEM for each selected column.
# - **Signal metric CSVs:** per-mouse feature metrics (peak, onset, decay tau, 2–8 s mean) saved in each experiment folder.
# - **Comparison exports:** per-metric statistics, plot data, and figures saved to `calcium_analysis/` within the source cohort directory.
#
# %%
#CONFIGURATION SECTION
# ---------------------------------------------------------------------------------------------------#
# Configure all settings here before running the analysis.

# Cohort selection
COHORT_OPTIONS = {
    "Cohort1": {
        "mice": ['B6J2717', 'B6J2718', 'B6J2719', 'B6J2721', 'B6J2722'],
        "identifier": "Cohort1"
    },
    "Cohort3": {
        "mice": ["B6J2780", "B6J2781", "B6J2783", "B6J2782"],
        "identifier": "Cohort3"
    }
}

# Select cohort
cohort_identifier = "Cohort1"  # Options: "Cohort1", "Cohort3

# which part of the notebook to run 
# 1st run the save_signal_metrics first for "Apple_halt_2s", 
# 2nd run the for "No_halt" - 
# NB: input  files may need to be rename, i.e. removing spurious / or ; characters
# 2nd run the generate_condition_comparison, then change cohorts 
SAVE_SIGNAL_METRICS = False
GENERATE_CONDITION_COMPARISON = True



# Select which animals to process (subset of the cohort's available mice).
# Leave empty list [] to process all mice in the cohort.
selected_mice: List[str] = []  # Example: ['B6J2717', 'B6J2718'] or [] for all

# Data columns to analyze
selected_columns = [
    'z_560_Baseline'
]

# ------------------------------------------------------------------
# SINGLE-CONDITION ANALYSIS SETUP
# ------------------------------------------------------------------
# Configure the 1️⃣ primary dataset whose aligned traces will be loaded and
# summarised into metrics/grand averages in the first half of the notebook.
COMPUTE_PER_MOUSE_ANALYSIS_OF_CALCIUM_DATA = {
    # Friendly name used in status prints / saved filenames.
    "label": "Apply halt 2s", # change for Apply halt 2s / No halt 
    # Suffix appended to each mouse ID to locate aligned CSVs inside `aligned_data/`.
    "event_suffix": "_Apply halt_2s_baselined_data.csv", # change for halt / nohalt  "_Apply halt_2s_baselined_data.csv" "_No halt_baselined_data.csv"
    # One or more experiment-day directories that contain the aligned traces.
    "data_dirs": [
        Path('/Volumes/RanczLab/Cohort1_rotation/Visual_mismatch_day3').expanduser(), # change per cohort
        Path('/Volumes/RanczLab/Cohort1_rotation/Visual_mismatch_day4').expanduser(),
    ],
}

event_name: str = COMPUTE_PER_MOUSE_ANALYSIS_OF_CALCIUM_DATA["event_suffix"]
PRIMARY_ANALYSIS_LABEL: str = COMPUTE_PER_MOUSE_ANALYSIS_OF_CALCIUM_DATA["label"]
DATA_DIRS: List[Path] = [
    Path(p).expanduser() for p in COMPUTE_PER_MOUSE_ANALYSIS_OF_CALCIUM_DATA["data_dirs"]
]
if not DATA_DIRS:
    raise ValueError("COMPUTE_PER_MOUSE_ANALYSIS_OF_CALCIUM_DATA must provide at least one data directory.")

SAVE_CSV = True      # Save grand averages with SEM as CSV file for plotting traces

SIGNAL_METRICS_OUTPUT_SUBDIR = DATA_DIRS[0].parent.parent

# Signal feature metric settings (incorporated from SANDBOX 4 workflow)
SIGNAL_METRIC_COLUMNS = [
    'z_560_Baseline',
]
SIGNAL_METRIC_FEATURES = (
    'peak',
    'onset_time',
    'decay_tau1',
    'mean_fluorescence_2_to_8s',
    'main_peak_residual_auc',
    'offset_residual_auc',
    'offset_peak_amplitude'
)

# Configure the post-alignment analysis window (seconds relative to alignment time).
# Adjust POST_ALIGNMENT_WINDOW_DURATION to control the length of the comparison interval.
POST_ALIGNMENT_WINDOW_START = 0.0 #aligment means halt time here  
POST_ALIGNMENT_WINDOW_DURATION = 2.0
POST_ALIGNMENT_WINDOW = (
    POST_ALIGNMENT_WINDOW_START,
    POST_ALIGNMENT_WINDOW_START + POST_ALIGNMENT_WINDOW_DURATION,
)

# Offset response detection settings (Strategy 3: Hybrid residual + peak detection)
OFFSET_BASELINE_WINDOW = (1.9, 2.0)  # Baseline window for direct peak amplitude calculation
OFFSET_ANALYSIS_START = 2.0          # Start analyzing offset response after main peak
OFFSET_ANALYSIS_END = 8.0            # End of offset analysis window
OFFSET_AUC_WINDOW = (2.0, 6.0)       # Window for area-under-curve calculation on residuals
DECAY_FIT_START_OFFSET = 0.0         # Extra time (seconds) added to peak time before starting exponential fit (0.0 = fit from peak)

# ------------------------------------------------------------------
# CONDITION COMPARISON SETUP
# ------------------------------------------------------------------
# Configure the 2️⃣ cross-condition comparison module (halt vs no-halt, etc.).
# Each entry stands on its own so it is easy to enable/disable or point to
# different folders without touching the single-condition settings above.  
# Each key defines one condition: give it a unique name, the event suffix to load,
# the directories the aligned traces live in, and an optional list of already
# computed metrics CSVs (leave empty to recompute on the fly).
COMPARE_CALCIUM_METRICS_ACROSS_CONDITIONS = OrderedDict({    # CHANGE according to cohort 
    "Apply_halt_day3": {
        "event_name": "_Apply halt_2s_baselined_data.csv",
        "data_dirs": [
            Path('/Volumes/RanczLab/Cohort1_rotation/Visual_mismatch_day3').expanduser(),
        ],
        "label": "Apply halt day 3",
        "metrics_csvs": [],
    },
    "Apply_halt_day4": {
        "event_name": "_Apply halt_2s_baselined_data.csv",
        "data_dirs": [
            Path('/Volumes/RanczLab/Cohort1_rotation/Visual_mismatch_day4').expanduser(),
        ],
        "label": "Apply halt day 4",
        "metrics_csvs": [],
    },
    "No_halt": {
        "event_name": "_No halt_baselined_data.csv",
        "data_dirs": [
            Path('/Volumes/RanczLab/Cohort1_rotation/Visual_mismatch_day3').expanduser(),
            Path('/Volumes/RanczLab/Cohort1_rotation/Visual_mismatch_day4').expanduser(),
        ],
        "label": "No halt",
        "metrics_csvs": [],
    },
})

COMPARISON_METRICS = ( # FIXME EDE TO ADD EXTRA FEATURES (aad function cell ANALYSIS functions  )
    "peak",
    "onset_time",
    "decay_tau1",
    "mean_fluorescence_2_to_8s",
    "main_peak_residual_auc",
    "offset_residual_auc",
    "offset_peak_amplitude"
)

# ---------------------------------------------------------------------------------------------------#
# # Auto-configure based on cohort selection
# ---------------------------------------------------------------------------------------------------#
if cohort_identifier in COHORT_OPTIONS:
    cohort_info = COHORT_OPTIONS[cohort_identifier]
    available_mice = cohort_info["mice"]
    if not selected_mice:  # If empty, use all mice
        selected_mice = available_mice
    else:  # Filter to only include valid mice for the selected cohort
        filtered_mice = [m for m in selected_mice if m in available_mice]
        ignored_mice = [m for m in selected_mice if m not in available_mice]
        selected_mice = filtered_mice
        if ignored_mice:
            print(f"⏭️ Ignoring mice not in cohort {cohort_identifier}: {ignored_mice}")
    print(f"✅ Cohort: {cohort_identifier}")
    print(f"✅ Available mice: {available_mice}")
    print(f"✅ Selected mice: {selected_mice}")
else:
    raise ValueError(f"Invalid cohort_identifier: {cohort_identifier}. Must be one of {list(COHORT_OPTIONS.keys())}")

# Save options
# SAVE_PICKLE = False  # Save results as pickle file (deprecated - use SAVE_ANIMAL_CSV instead)
# FIXME THIS IS REGENERATING SOMETHING BUT IT DOES NOT MAKE SENSE! SAVE_ANIMAL_CSV = False  # Save averaged mismatch aligned data for each animal as CSV
# FIXME WE WILL NOT USE THESE PLOTS FOR NOW!!! GENERATE_PLOTS = True  # Generate plots
# # # Columns to plot
# columns_to_plot = [
#     'Velocity_0X_Baseline', 'Motor_Velocity_Baseline', 
#     'z_470_Baseline', 'z_560_Baseline'
# ]
# # Pre/post comparison plotting options
# FIXME WE WILL NOT USE THESE PLOTS FOR NOW!!! PLOT_PREPOST_FROM_RESULTS = True  # Generate pre/post plots from freshly computed results
# FIXME NEED TO VERIFY WHAT IS GOING ON LOAD_EXISTING_PREPOST_CSV = False  # Load a previously created cohort_aligned_data_analysis.csv
# FIXME VERIFY WHAT IS GOING ONEXISTING_PREPOST_CSV_PATH = Path('/Users/nora/Desktop/for_poster/cohort_3/cohort_aligned_data_analysis.csv').expanduser()
# PREPOST_SAVE_DIR = None  # Optional custom directory to save pre/post plots

# %%
# DATA LOADING
#---------------------------------------------------------------------------------------------------#
def load_aligned_data(data_dirs, event_name, selected_mice, allowed_mice=None):
    """
    Load aligned data from every CSV that matches the requested event suffix across
    all provided directories. Multiple matching files per mouse (and per experiment
    day) are preserved so downstream steps can aggregate or analyse them per day.
    
    Parameters
    ----------
    data_dirs : list
        Directories that should be searched (recursively) for aligned CSV files.
    event_name : str
        Event name suffix (e.g. "_No halt_baselined_data.csv").
    selected_mice : list[str]
        Explicit list of mice to include. If empty, all cohort mice are considered.
    allowed_mice : list[str] | None
        Full set of cohort-allowed mice (used when selected_mice is empty).
    
    Returns
    -------
    dict
        Dictionary keyed by mouse name containing loaded dataframes and metadata.
    """
    if selected_mice:
        selected_mice_set: Optional[set[str]] = set(selected_mice)
    elif allowed_mice:
        selected_mice_set = set(allowed_mice)
    else:
        selected_mice_set = None

    event_suffix = event_name or ""
    if event_suffix and not event_suffix.lower().endswith(".csv"):
        print(f"⚠️ Event suffix '{event_suffix}' does not end with '.csv'; loading may fail.")

    data_dir_paths: List[Path] = [Path(p).expanduser().resolve() for p in data_dirs]
    per_mouse_records: Dict[str, List[Dict[str, object]]] = {}
    discovered_mice: set[str] = set()

    for base_dir in data_dir_paths:
        if not base_dir.exists():
            print(f"⚠️ Data directory not found: {base_dir}")
            continue

        try:
            csv_candidates = list(base_dir.rglob("*.csv"))
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Failed to scan directory {base_dir}: {exc}")
            continue

        for csv_path in csv_candidates:
            if not csv_path.is_file() or csv_path.name.startswith("._"):
                continue
            if event_suffix and not csv_path.name.endswith(event_suffix):
                continue

            resolved_csv = csv_path.resolve()
            mouse_id = (
                resolved_csv.name[:-len(event_suffix)] if event_suffix else resolved_csv.stem
            )
            if not mouse_id:
                continue

            discovered_mice.add(mouse_id)
            if selected_mice_set and mouse_id not in selected_mice_set:
                continue

            try:
                aligned_df = pd.read_csv(resolved_csv)
            except FileNotFoundError:
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ Failed to read {resolved_csv}: {exc}")
                continue

            if aligned_df.empty:
                print(f"⚠️ Empty CSV skipped: {resolved_csv}")
                continue

            experiment_dir = next(
                (candidate for candidate in data_dir_paths if candidate in resolved_csv.parents),
                None,
            )
            experiment_day = experiment_dir.name if experiment_dir else resolved_csv.parent.name

            per_mouse_records.setdefault(mouse_id, []).append(
                {
                    "dataframe": aligned_df,
                    "csv_path": resolved_csv,
                    "aligned_dir": resolved_csv.parent,
                    "experiment_dir": experiment_dir,
                    "experiment_day": experiment_day,
                }
            )
            print(f"✅ Loaded {mouse_id}: {resolved_csv}")

    if selected_mice_set:
        missing = sorted(selected_mice_set - discovered_mice)
        if missing:
            print(f"⏭️ No matching files found for selected mice: {missing}")

    loaded_data: Dict[str, Dict[str, object]] = {}
    for mouse_name, records in per_mouse_records.items():
        aligned_dirs = [record["aligned_dir"] for record in records]
        source_csvs = [record["csv_path"] for record in records]
        experiment_dirs = [
            record["experiment_dir"] for record in records if record.get("experiment_dir") is not None
        ]
        loaded_data[mouse_name] = {
            "mouse_name": mouse_name,
            "records": records,
            "dataframes": [record["dataframe"] for record in records],
            "data_path": aligned_dirs[0] if aligned_dirs else None,
            "aligned_dirs": aligned_dirs,
            "source_csvs": source_csvs,
            "experiment_dirs": experiment_dirs,
        }

    if not loaded_data:
        print("⚠️ No aligned CSV files were loaded. Please verify the configuration.")

    return loaded_data

# Load the data
loaded_data = load_aligned_data(DATA_DIRS, event_name, selected_mice, available_mice)


# %%
# OUTPUT DIRECTORY HELPERS
#---------------------------------------------------------------------------------------------------#

def determine_main_data_dir(loaded_data, data_dirs, cohort_identifier):
    """Select the base data directory corresponding to the loaded cohort data."""
    data_dir_paths = [Path(p).expanduser().resolve() for p in data_dirs]

    loaded_base_dirs = []
    for entry in loaded_data.values():
        if not isinstance(entry, dict):
            continue
        candidate_paths: List[Path] = []
        records = entry.get("records", [])
        for record in records:
            experiment_dir = record.get("experiment_dir")
            if experiment_dir:
                candidate_paths.append(Path(experiment_dir).resolve())
            else:
                aligned_dir = record.get("aligned_dir")
                if aligned_dir:
                    candidate_paths.append(Path(aligned_dir).resolve().parent.parent)
        if not candidate_paths and entry.get("data_path"):
            candidate_paths.append(Path(entry["data_path"]).resolve())
        for candidate in candidate_paths:
            if candidate not in loaded_base_dirs:
                loaded_base_dirs.append(candidate)

    if not loaded_base_dirs:
        for candidate in data_dir_paths:
            if candidate.exists():
                return candidate
        return Path.cwd()

    cohort_id = (cohort_identifier or "").lower()
    if cohort_id:
        for candidate in data_dir_paths:
            candidate_resolved = Path(candidate).resolve()
            if candidate_resolved in loaded_base_dirs and cohort_id in str(candidate_resolved).lower():
                return candidate_resolved

    for candidate in data_dir_paths:
        candidate_resolved = Path(candidate).resolve()
        if candidate_resolved in loaded_base_dirs:
            return candidate_resolved

    return loaded_base_dirs[0]

def extract_experiment_day(base_dir: Path) -> str:
    """Derive experiment day label from the selected base directory."""
    if not base_dir or not isinstance(base_dir, Path):
        return "unknown"
    return base_dir.name or "unknown"


def sanitize_for_filename(value: Optional[str], fallback: str = "unknown") -> str:
    """Convert a string into a filesystem-friendly fragment."""
    if value is None:
        return fallback
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return cleaned or fallback


def build_grand_average_filename(cohort_id: str, experiment_day: str, event_name: str) -> str:
    """Compose the output filename for grand average CSV exports."""
    cohort_part = sanitize_for_filename(cohort_id, "cohort")
    day_part = sanitize_for_filename(experiment_day, "day")
    event_label = sanitize_for_filename(event_name.replace(".csv", ""), "event")
    return f"{cohort_part}_{day_part}_grand_averages_with_sem_{event_label}.csv"



# %%
# ANALYSIS FUNCTIONS
#---------------------------------------------------------------------------------------------------#

def compute_mouse_means_and_grand_average(loaded_data, selected_columns, main_data_dir, selected_mice):
    """
    Compute means per mouse and grand averages across selected mice for selected columns.
    
    Parameters:
    loaded_data (dict): Dictionary with data paths as keys and mouse data as values
    selected_columns (list): List of column names to analyze
    main_data_dir (str/Path): Main directory to save results
    selected_mice (list): List of mouse names to include in the grand average
    
    Returns:
    tuple: (mean_data_per_mouse, sem_data_per_mouse, grand_averages, grand_sems, mouse_to_data_path)
        mouse_to_data_path: Dictionary mapping mouse names to their data_path (aligned_data folder)
    """
    
    main_data_dir = Path(main_data_dir)
    
    loaded_mouse_names = [data['mouse_name'] for data in loaded_data.values()]
    loaded_mouse_set = set(loaded_mouse_names)

    if selected_mice:
        filtered_selection = [m for m in selected_mice if m in loaded_mouse_set]
        ignored_selection = [m for m in selected_mice if m not in loaded_mouse_set]
        if ignored_selection:
            print(f"⏭️ Ignoring selected mice without loaded data: {ignored_selection}")
        selected_mice = filtered_selection
    else:
        selected_mice = loaded_mouse_names

    if not selected_mice:
        print("⚠️ No mice available for grand average computation.")
        return {}, {}, pd.DataFrame(), pd.DataFrame(), {}
    
    print(f"Processing selected columns: {selected_columns}")
    
    # Step 1: Compute mean and SEM for each mouse
    mean_data_per_mouse = {}
    sem_data_per_mouse = {}
    mouse_to_data_path = {}  # Track data_path for each mouse
    
    for mouse_name, entry in loaded_data.items():
        if mouse_name not in selected_mice:
            continue

        frames: List[pd.DataFrame] = []
        if isinstance(entry, dict):
            records = entry.get("records", [])
            if records:
                for record in records:
                    df = record.get("dataframe")
                    if isinstance(df, pd.DataFrame):
                        frames.append(df.copy())
            elif entry.get("dataframes"):
                frames = [
                    df.copy()
                    for df in entry.get("dataframes", [])
                    if isinstance(df, pd.DataFrame)
                ]

        if not frames:
            print(f"⚠️ No data frames available for {mouse_name}, skipping...")
            continue

        print(f"Processing mouse: {mouse_name} ({len(frames)} file(s))")

        combined_df = pd.concat(frames, ignore_index=True, sort=False)
        if combined_df.empty:
            print(f"⚠️ Combined dataframe empty for {mouse_name}, skipping...")
            continue

        if "Time (s)" not in combined_df.columns:
            print(f"⚠️  'Time (s)' column not found for {mouse_name}, skipping...")
            continue

        available_columns = [col for col in selected_columns if col in combined_df.columns]
        missing_columns = [col for col in selected_columns if col not in combined_df.columns]
        if missing_columns:
            print(f"⚠️  Missing columns for {mouse_name}: {missing_columns}")
        if not available_columns:
            print(f"⚠️  No requested columns available for {mouse_name}, skipping...")
            continue

        numeric_selected = []
        for col in available_columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")
            if col != "Time (s)" and pd.api.types.is_numeric_dtype(combined_df[col]):
                numeric_selected.append(col)
        combined_df["Time (s)"] = pd.to_numeric(combined_df["Time (s)"], errors="coerce")
        combined_df = combined_df.dropna(subset=["Time (s)"])

        if not numeric_selected:
            print(f"⚠️  No numeric columns found for {mouse_name}")
            continue

        grouped = combined_df.groupby("Time (s)")
        mean_df = grouped[numeric_selected].mean()
        sem_df = grouped[numeric_selected].sem()

        mean_data_per_mouse[mouse_name] = mean_df
        sem_data_per_mouse[mouse_name] = sem_df

        data_path_value = None
        if isinstance(entry, dict):
            if entry.get("records"):
                first_record = next(
                    (record for record in entry["records"] if record.get("aligned_dir")), None
                )
                if first_record:
                    data_path_value = Path(first_record["aligned_dir"])
            if data_path_value is None and entry.get("data_path"):
                data_path_value = Path(entry["data_path"])

        mouse_to_data_path[mouse_name] = data_path_value

        print(f"✅ Processed {len(numeric_selected)} columns for {mouse_name}")
    
    # Step 2: Compute grand averages across selected mice
    print(f"\n📊 Computing grand averages across {len(mean_data_per_mouse)} selected mice...")
    
    # Get all unique time points
    all_time_points = set()
    for mouse_data in mean_data_per_mouse.values():
        all_time_points.update(mouse_data.index)
    all_time_points = sorted(list(all_time_points))
    
    # Get all columns that were successfully processed
    all_processed_columns = set()
    for mouse_data in mean_data_per_mouse.values():
        all_processed_columns.update(mouse_data.columns)
    all_processed_columns = sorted(list(all_processed_columns))
    
    print(f"Time points: {len(all_time_points)} from {min(all_time_points):.2f}s to {max(all_time_points):.2f}s")
    print(f"Processed columns: {all_processed_columns}")
    
    # Create grand average DataFrame
    grand_averages = pd.DataFrame(index=all_time_points, columns=all_processed_columns)
    grand_averages.index.name = 'Time (s)'
    
    grand_sems = pd.DataFrame(index=all_time_points, columns=all_processed_columns)
    grand_sems.index.name = 'Time (s)'
    
    # Compute grand averages for each column and time point
    for col in all_processed_columns:
        for time_point in all_time_points:
            # Collect data from all selected mice for this time point and column
            mouse_values = []
            for mouse_name, mouse_data in mean_data_per_mouse.items():
                if time_point in mouse_data.index and col in mouse_data.columns:
                    value = mouse_data.loc[time_point, col]
                    if not pd.isna(value):
                        mouse_values.append(value)
            
            if len(mouse_values) > 0:
                grand_averages.loc[time_point, col] = np.mean(mouse_values)
                if len(mouse_values) > 1:
                    grand_sems.loc[time_point, col] = np.std(mouse_values) / np.sqrt(len(mouse_values))
                else:
                    grand_sems.loc[time_point, col] = 0
    
    return mean_data_per_mouse, sem_data_per_mouse, grand_averages, grand_sems, mouse_to_data_path

def analyze_mice_data(loaded_data, selected_columns, main_data_dir):
    """
    Complete analysis workflow: compute means, grand averages, save CSV, and create plots.
    
    Parameters:
    loaded_data (dict): Your loaded_data dictionary
    selected_columns (list): List of column names to analyze (including 'Time (s)')
    main_data_dir (str/Path): Main directory to save results
    
    Returns:
    dict: Complete results including individual and grand averages
    """
    
    print(f"\n{'='*60}")
    print(f"MOUSE DATA ANALYSIS")
    print(f"{'='*60}")
    
    # Use the selected_mice from the configuration (defined in Cell 2)
    # Compute means and grand averages
    mean_data_per_mouse, sem_data_per_mouse, grand_averages, grand_sems, mouse_to_data_path = compute_mouse_means_and_grand_average(
        loaded_data, selected_columns, main_data_dir, selected_mice
    )

    # Print summary
    print(f"\n📊 ANALYSIS COMPLETE:")
    print(f"   • Number of mice analyzed: {len(mean_data_per_mouse)}")
    print(f"   • Mouse names: {list(mean_data_per_mouse.keys())}")
    print(f"   • Columns processed: {list(grand_averages.columns)}")
    print(f"   • Time range: {grand_averages.index.min():.2f}s to {grand_averages.index.max():.2f}s")
    print(f"   • Files saved in: {main_data_dir}")
    
    # Return all results
    results = {
        'mean_data_per_mouse': mean_data_per_mouse,
        'sem_data_per_mouse': sem_data_per_mouse,
        'grand_averages': grand_averages,
        'grand_sems': grand_sems,
        'mouse_to_data_path': mouse_to_data_path,
    }
    
    return results


# %%
# SAVE GRAND AVERAGES CSV
# Determine main data directory (first existing entry in DATA_DIRS, otherwise current working dir)
_existing_data_dirs = [Path(p).expanduser() for p in DATA_DIRS if Path(p).expanduser().exists()]
if 'loaded_data' in locals():
    main_data_dir = determine_main_data_dir(loaded_data, DATA_DIRS, cohort_identifier)
else:
    main_data_dir = _existing_data_dirs[0] if _existing_data_dirs else Path.cwd()
main_data_dir = Path(main_data_dir).resolve()
experiment_day = extract_experiment_day(main_data_dir)

# Run analysis and keep results available for later cells/plots
if loaded_data:
    results = analyze_mice_data(loaded_data, selected_columns, main_data_dir)
else:
    print("⚠️ No data loaded. Please check your configuration and data paths.")
    results = None
if results and SAVE_CSV:
    # Create a DataFrame combining grand averages and SEMs
    grand_avg_with_sem = results['grand_averages'].copy()
    for col in results['grand_sems'].columns:
        grand_avg_with_sem[f'{col}_SEM'] = results['grand_sems'][col]

    # Generate filename with cohort and experiment day context
    csv_filename = main_data_dir / build_grand_average_filename(
        cohort_identifier,
        experiment_day,
        event_name,
    )

    # Save the DataFrame to a CSV file (retain Time (s) as index)
    grand_avg_with_sem.to_csv(csv_filename)
    print(f"✅ Grand averages with SEM saved to: {csv_filename}")
else:
    print("⏭️  Skipping CSV save (SAVE_CSV=False or no results)")


# %%
# INTERACTIVE GRAND AVERAGE PLOTTING
#---------------------------------------------------------------------------------------------------#

def find_csv_files(base_paths):
    """Find all CSV files containing 'grand_averages' in the given paths."""
    csv_files = set()
    for base_path in base_paths:
        path = Path(base_path)
        if path.exists():
            for csv_file in path.rglob('*.csv'):
                name = csv_file.name
                if name.startswith('._'):
                    continue
                if 'grand_averages' in name:
                    csv_files.add(str(csv_file))
    return sorted(csv_files)

def get_available_columns(df):
    """Get available data columns (excluding Time and SEM columns)."""
    # Filter out Time column and SEM columns
    data_cols = [col for col in df.columns 
                 if col != 'Time (s)' and not col.endswith('_SEM')]
    return sorted(data_cols)

def get_column_label(column_name):
    """Generate a readable label for a column."""
    # Map common column names to readable labels
    label_map = {
        'z_470': 'GRAB-5HT3.0 (z-score)',
        'z_470_Baseline': 'GRAB-5HT3.0 (z-score)',
        'z_560': 'RGeco1a (z-score)',
        'z_560_Baseline': 'RGeco1a (z-score)',
        'Velocity_0X': 'Running speed',
        'Velocity_0X_Baseline': 'Running speed',
        'Motor_Velocity': 'Motor Velocity',
        'Motor_Velocity_Baseline': 'Motor Velocity',
    }
    return label_map.get(column_name, column_name)

def get_axis_label(column_name):
    """Generate axis label based on column name."""
    if 'z_' in column_name or 'z-score' in column_name.lower():
        return 'z-score'
    elif 'Velocity' in column_name or 'velocity' in column_name.lower():
        return 'Running speed (m/s)'
    elif 'Motor' in column_name:
        return 'Motor velocity (m/s)'
    else:
        return column_name

def should_use_right_axis(column_name):
    """Determine if a column should be plotted on the right axis."""
    # Velocity and Motor columns go on right axis
    return 'Velocity' in column_name or 'Motor' in column_name


def infer_series_metadata(file_path: str) -> dict:
    """Extract cohort and experiment day heuristically from a CSV path."""
    path = Path(file_path)
    parts = [p for p in path.parts]

    cohort = next((p for p in parts if 'Cohort' in p), 'Unknown')
    experiment_day = next((p for p in parts if 'Visual_mismatch' in p or 'Vestibular_mismatch' in p), 'UnknownDay')

    # Normalise formatting
    cohort_clean = cohort.replace('_', ' ')
    day_clean = experiment_day.replace('_', ' ')
    return {'cohort': cohort_clean, 'experiment_day': day_clean}


def select_series_color(cohort: str) -> str:
    """Choose a colour based on cohort name."""
    cohort_lower = cohort.lower()
    if 'cohort3' in cohort_lower:
        return '#8B0000'  # dark red
    if 'cohort1' in cohort_lower:
        return '#FF0000'  # red
    return 'slategray'


def plot_grand_averages_multi_series(series_list):
    """Plot multiple series from selected grand-average CSV files on a shared figure."""
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'legend.fontsize': 7,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8
    })

    if not series_list:
        print("⚠️ No series selected for plotting.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = None

    left_axis_series = []
    right_axis_series = []

    for series in series_list:
        file_path = series.get('file')
        column = series.get('column')
        label = series.get('label')

        if not file_path or not column:
            continue

        try:
            abs_file_path = Path(file_path).resolve()
            df = pd.read_csv(abs_file_path)
        except Exception:
            continue

        if column not in df.columns:
            continue

        metadata = infer_series_metadata(file_path)
        colour = series.get('color') or select_series_color(metadata['cohort'])

        label_parts = [get_column_label(column)]
        if metadata['cohort'] != 'Unknown':
            label_parts.append(metadata['cohort'])
        if metadata['experiment_day'] != 'UnknownDay':
            label_parts.append(metadata['experiment_day'])
        plot_label = label if label else ' | '.join(label_parts)

        use_right = should_use_right_axis(column)

        if use_right:
            if ax2 is None:
                ax2 = ax.twinx()
            ax_handle = ax2
            collection = right_axis_series
        else:
            ax_handle = ax
            collection = left_axis_series

        ax_handle.plot(
            df['Time (s)'],
            df[column],
            label=plot_label,
            color=colour,
            linestyle=series.get('linestyle', '-'),
            alpha=1,
        )

        sem_column = f'{column}_SEM'
        if sem_column in df.columns:
            ax_handle.fill_between(
                df['Time (s)'],
                df[column] - df[sem_column],
                df[column] + df[sem_column],
                color=colour,
                alpha=0.1 if not use_right else 0.2,
            )

        collection.append(series)

    if left_axis_series:
        first_col = left_axis_series[0]['column']
        ax.set_ylabel(get_axis_label(first_col), fontname='Arial', fontsize=9, color='black')
        ax.tick_params(axis='y', labelcolor='black')

    if right_axis_series and ax2:
        first_col = right_axis_series[0]['column']
        ax2.set_ylabel(get_axis_label(first_col), fontname='Arial', fontsize=9, color='slategray')
        ax2.tick_params(axis='y', labelcolor='slategray')

    ax.axvspan(0, 2, color='gray', alpha=0.2, label='Visual mismatch (0-2s)')
    ax.set_title('Grand Averages Comparison', fontname='Arial', fontsize=10)
    ax.set_xlabel('Time (s)', fontname='Arial', fontsize=9)

    lines, labels = ax.get_legend_handles_labels()
    if ax2:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right',
                  prop={'family': 'Arial', 'size': 8})
    else:
        ax.legend(lines, labels, loc='upper right', prop={'family': 'Arial', 'size': 8})

    plt.tight_layout()

    return fig

def plot_grand_averages_single(df, label, y1_col=None, y2_col=None):
    """Plot grand averages from a single CSV file with selected columns."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = None
    
    # Plot y1 column on left axis
    if y1_col and y1_col in df.columns:
        ax.plot(df['Time (s)'], df[y1_col], label=get_column_label(y1_col), 
                color='green', alpha=1)
        if f'{y1_col}_SEM' in df.columns:
            ax.fill_between(df['Time (s)'],
                           df[y1_col] - df[f'{y1_col}_SEM'],
                           df[y1_col] + df[f'{y1_col}_SEM'],
                           color='green', alpha=0.1)
        ax.set_ylabel(get_axis_label(y1_col), fontname='DejaVu Sans', fontsize=10, color='black')
        ax.tick_params(axis='y', labelcolor='black')
    
    # Plot y2 column on right axis (if different from y1)
    if y2_col and y2_col in df.columns and y2_col != y1_col:
        ax2 = ax.twinx()
        ax2.plot(df['Time (s)'], df[y2_col], label=get_column_label(y2_col), 
                color='slategray', alpha=1)
        if f'{y2_col}_SEM' in df.columns:
            ax2.fill_between(df['Time (s)'],
                            df[y2_col] - df[f'{y2_col}_SEM'],
                            df[y2_col] + df[f'{y2_col}_SEM'],
                            color='slategray', alpha=0.2)
        ax2.set_ylabel(get_axis_label(y2_col), fontname='DejaVu Sans', fontsize=10, color='slategray')
        ax2.tick_params(axis='y', labelcolor='slategray')
    
    ax.axvspan(0, 2, color='gray', alpha=0.2, label='Visual mismatch (0-2s)')
    ax.set_title(f'Grand Averages: {label}', fontname='DejaVu Sans', fontsize=12)
    ax.set_xlabel('Time (s)', fontname='DejaVu Sans', fontsize=10)
    
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    if ax2:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right', 
                 prop={'family': 'DejaVu Sans', 'size': 10})
    else:
        ax.legend(lines, labels, loc='upper right', prop={'family': 'DejaVu Sans', 'size': 10})
    
    plt.tight_layout()
    plt.show()

def plot_grand_averages_comparison(df1, df2, label1, label2, y1_col=None, y2_col=None):
    """Plot grand averages comparing two CSV files with selected columns."""
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = None
    
    # Plot y1 column on left axis
    if y1_col:
        if y1_col in df1.columns:
            ax.plot(df1['Time (s)'], df1[y1_col], label=f"{get_column_label(y1_col)} ({label1})", 
                    color='green', alpha=1)
            if f'{y1_col}_SEM' in df1.columns:
                ax.fill_between(df1['Time (s)'],
                               df1[y1_col] - df1[f'{y1_col}_SEM'],
                               df1[y1_col] + df1[f'{y1_col}_SEM'],
                               color='green', alpha=0.1)
        
        if y1_col in df2.columns:
            ax.plot(df2['Time (s)'], df2[y1_col], label=f"{get_column_label(y1_col)} ({label2})", 
                    color='orange', alpha=1, linestyle='--')
            if f'{y1_col}_SEM' in df2.columns:
                ax.fill_between(df2['Time (s)'],
                               df2[y1_col] - df2[f'{y1_col}_SEM'],
                               df2[y1_col] + df2[f'{y1_col}_SEM'],
                               color='orange', alpha=0.1)
        
        ax.set_ylabel(get_axis_label(y1_col), fontname='DejaVu Sans', fontsize=10, color='black')
        ax.tick_params(axis='y', labelcolor='black')
    
    # Plot y2 column on right axis (if different from y1)
    if y2_col and y2_col != y1_col:
        ax2 = ax.twinx()
        
        if y2_col in df1.columns:
            ax2.plot(df1['Time (s)'], df1[y2_col], label=f"{get_column_label(y2_col)} ({label1})", 
                    color='slategray', alpha=1)
            if f'{y2_col}_SEM' in df1.columns:
                ax2.fill_between(df1['Time (s)'],
                                df1[y2_col] - df1[f'{y2_col}_SEM'],
                                df1[y2_col] + df1[f'{y2_col}_SEM'],
                                color='slategray', alpha=0.2)
        
        if y2_col in df2.columns:
            ax2.plot(df2['Time (s)'], df2[y2_col], label=f"{get_column_label(y2_col)} ({label2})", 
                    color='slategray', alpha=1, linestyle='--')
            if f'{y2_col}_SEM' in df2.columns:
                ax2.fill_between(df2['Time (s)'],
                                df2[y2_col] - df2[f'{y2_col}_SEM'],
                                df2[y2_col] + df2[f'{y2_col}_SEM'],
                                color='slategray', alpha=0.2)
        
        ax2.set_ylabel(get_axis_label(y2_col), fontname='DejaVu Sans', fontsize=10, color='slategray')
        ax2.tick_params(axis='y', labelcolor='slategray')
    
    ax.axvspan(0, 2, color='gray', alpha=0.2, label='Visual mismatch (0-2s)')
    ax.set_title('Grand Averages with SEM', fontname='DejaVu Sans', fontsize=12)
    ax.set_xlabel('Time (s)', fontname='DejaVu Sans', fontsize=10)
    
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    if ax2:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right', 
                 prop={'family': 'DejaVu Sans', 'size': 10})
    else:
        ax.legend(lines, labels, loc='upper right', prop={'family': 'DejaVu Sans', 'size': 10})
    
    plt.tight_layout()
    plt.show()

def plot_grand_averages_interactive():
    """Create interactive plot with dropdowns for multiple CSV file and column selection."""
    
    # Find available CSV files using the same paths as DATA_DIRS
    base_paths = DATA_DIRS if DATA_DIRS else []
    
    csv_files = find_csv_files(base_paths)
    
    if not csv_files:
        return
    
    if WIDGETS_AVAILABLE:
        # Store series widgets
        series_widgets = []
        series_container = widgets.VBox([])
        
        # Store current figure for saving
        current_fig = [None]  # Use list to allow modification in nested functions
        
        # Color options for different series
        color_options = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        linestyle_options = ['-', '--', '-.', ':']
        
        def create_series_row(series_id):
            """Create a row of widgets for selecting file and column."""
            file_dropdown = widgets.Dropdown(
                options=['None'] + csv_files,
                value='None',
                description=f'File {series_id}:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='400px')
            )
            
            col_dropdown = widgets.Dropdown(
                options=[''],
                value='',
                description='Column:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='250px')
            )
            
            remove_btn = widgets.Button(
                description='Remove',
                button_style='danger',
                layout=widgets.Layout(width='80px', height='30px')
            )
            
            def update_columns(change):
                """Update column dropdown when file changes."""
                if change['new'] and change['new'] != 'None':
                    try:
                        # Clear the current column value first
                        col_dropdown.value = ''
                        
                        # Read the new file
                        df = pd.read_csv(change['new'])
                        available_cols = get_available_columns(df)
                        
                        # Update options and set to first available column
                        col_dropdown.options = available_cols
                        if available_cols:
                            col_dropdown.value = available_cols[0]
                        else:
                            col_dropdown.value = ''
                    except Exception:
                        col_dropdown.options = ['']
                        col_dropdown.value = ''
                else:
                    col_dropdown.options = ['']
                    col_dropdown.value = ''
            
            def remove_series(b):
                """Remove this series row."""
                for i, (f, c, r, _) in enumerate(series_widgets):
                    if f == file_dropdown:
                        series_widgets.pop(i)
                        update_series_display()
                        break
            
            file_dropdown.observe(update_columns, names='value')
            remove_btn.on_click(remove_series)
            
            return file_dropdown, col_dropdown, remove_btn, widgets.HBox([file_dropdown, col_dropdown, remove_btn])
        
        def update_series_display():
            """Update the display of all series rows."""
            children = [row[3] for row in series_widgets]
            series_container.children = children
        
        def add_series(b):
            """Add a new series row."""
            series_id = len(series_widgets) + 1
            widgets_row = create_series_row(series_id)
            series_widgets.append(widgets_row)
            update_series_display()
        
        def plot_all_series(b):
            """Plot all selected series."""
            with output:
                clear_output(wait=True)
                series_list = []
                seen = set()

                for file_dropdown, col_dropdown, _, _ in series_widgets:
                    file_path = file_dropdown.value
                    column = col_dropdown.value

                    key = (file_path, column)
                    if (
                        file_path
                        and file_path != 'None'
                        and column
                        and column != ''
                        and key not in seen
                    ):
                        seen.add(key)
                        series_list.append({
                            'file': file_path,
                            'column': column,
                            'linestyle': linestyle_options[(len(series_list) // len(color_options)) % len(linestyle_options)]
                        })

                if not series_list:
                    return

                try:
                    fig = plot_grand_averages_multi_series(series_list)
                    current_fig[0] = fig  # Store figure for saving
                    if fig is not None:
                        display(fig)
                        plt.close(fig)
                except Exception:
                    pass

        def save_current_plot(b):
            """Save the current plot to a file."""
            if current_fig[0] is None:
                return

            # Generate filename based on selected series
            series_names = []
            first_file_path = None

            for file_dropdown, col_dropdown, _, _ in series_widgets:
                file_path = file_dropdown.value
                column = col_dropdown.value
                if file_path and file_path != 'None' and column and column != '':
                    if first_file_path is None:
                        first_file_path = file_path
                    file_label = Path(file_path).parent.name
                    series_names.append(f"{file_label}_{column}")

            if series_names:
                filename_base = "_vs_".join(series_names[:3])  # Limit filename length
                if len(series_names) > 3:
                    filename_base += f"_and_{len(series_names)-3}_more"
            else:
                filename_base = "grand_averages_plot"

            # Get save directory from the first CSV file being plotted
            if first_file_path:
                save_dir = Path(first_file_path).parent
            else:
                save_dir = Path(DATA_DIRS[0]) if DATA_DIRS else Path.cwd()

            # Save as PDF and PNG
            pdf_path = save_dir / f"{filename_base}.pdf"
            png_path = save_dir / f"{filename_base}.png"

            try:
                current_fig[0].savefig(pdf_path, dpi=300, bbox_inches='tight')
                current_fig[0].savefig(png_path, dpi=300, bbox_inches='tight')
            except Exception:
                pass
        
        # Create initial series row
        add_series(None)
        
        # Buttons
        add_btn = widgets.Button(
            description='+ Add Series',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        add_btn.on_click(add_series)
        
        plot_button = widgets.Button(
            description='Plot All Series',
            button_style='success',
            layout=widgets.Layout(width='200px')
        )

        def plot_and_show(_):
            plot_all_series(_)

        plot_button.on_click(plot_and_show)
        
        save_button = widgets.Button(
            description='Save Plot',
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        save_button.on_click(save_current_plot)
        
        output = widgets.Output()
        
        display(widgets.VBox([
            widgets.HTML("<h3>Select multiple files and columns to plot:</h3>"),
            widgets.HTML("<p><i>Add multiple series to compare columns from different files on the same graph.</i></p>"),
            series_container,
            widgets.HBox([add_btn, plot_button, save_button]),
            output
        ]))
    else:
        return

# Run interactive plotting
plot_grand_averages_interactive()


# %%
# SIGNAL METRIC ANALYSIS UTILITIES
#---------------------------------------------------------------------------------------------------#

def _estimate_baseline(times: np.ndarray, values: np.ndarray) -> float:
    """
    Estimate the baseline from late post-stimulus period.
    
    Uses the median of the last 2 seconds of data to robustly estimate baseline.
    Falls back to pre-stimulus period if late data is unavailable.
    """
    if times.size == 0 or values.size == 0:
        return 0.0
    
    # Try late post-stimulus period (last 2 seconds of data)
    time_range = times[-1] - times[0]
    if time_range >= 2.0:
        late_mask = times >= (times[-1] - 2.0)
        if late_mask.any():
            late_values = values[late_mask]
            late_values = late_values[np.isfinite(late_values)]
            if late_values.size > 0:
                return float(np.median(late_values))
    
    # Fallback: use pre-stimulus period if available (times < 0)
    pre_mask = times < 0.0
    if pre_mask.any():
        pre_values = values[pre_mask]
        pre_values = pre_values[np.isfinite(pre_values)]
        if pre_values.size > 0:
            return float(np.median(pre_values))
    
    # Last resort: use median of first 10% of data
    n_baseline = max(3, int(0.1 * times.size))
    baseline_values = values[:n_baseline]
    baseline_values = baseline_values[np.isfinite(baseline_values)]
    if baseline_values.size > 0:
        return float(np.median(baseline_values))
    
    return 0.0


def _double_exponential_model(t: np.ndarray, A1: float, tau1: float, A2: float, tau2: float) -> np.ndarray:
    """Double exponential decay model (without baseline, added separately)."""
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)


def _estimate_double_exponential_decay(
    times: np.ndarray,
    values: np.ndarray,
    peak_index: int,
    baseline: float,
    fit_start_offset: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    Fit a double-exponential decay with baseline:
    y(t) = baseline + A1*exp(-t/tau1) + A2*exp(-t/tau2)
    
    Parameters
    ----------
    times : np.ndarray
        Time points
    values : np.ndarray
        Signal values
    peak_index : int
        Index of the peak in the arrays
    baseline : float
        Baseline level
    fit_start_offset : float
        Additional time (seconds) to add to peak time before starting fit (default: 0.0)
    
    Returns
    -------
    Tuple[float, float, float, float]
        (tau1, A1, tau2, A2) or (nan, nan, nan, nan) if fitting fails
    """
    if times.size == 0 or values.size == 0:
        return float(np.nan), float(np.nan), float(np.nan), float(np.nan)
    
    peak_value = values[peak_index]
    if np.isnan(peak_value):
        return float(np.nan), float(np.nan), float(np.nan), float(np.nan)
    
    # Use post-peak data for fitting, optionally offset from peak time
    peak_time = times[peak_index]
    fit_start_time = peak_time + fit_start_offset
    
    # Find data points at or after the fit start time
    fit_mask = times >= fit_start_time
    if not fit_mask.any():
        return float(np.nan), float(np.nan), float(np.nan), float(np.nan)
    
    post_times = times[fit_mask]
    post_values = values[fit_mask]
    
    if post_times.size < 6:  # Need at least 6 points for 4-parameter fit
        return float(np.nan), float(np.nan), float(np.nan), float(np.nan)
    
    # Subtract baseline
    baseline_corrected = post_values - baseline
    total_amplitude = peak_value - baseline
    
    if total_amplitude <= 0:
        return float(np.nan), float(np.nan), float(np.nan), float(np.nan)
    
    # Relative times from fit start time
    rel_times = post_times - fit_start_time
    
    # Initial parameter guesses
    # Fast component: ~70% of amplitude, tau ~0.3-0.5s
    # Slow component: ~30% of amplitude, tau ~2-4s
    A1_guess = 0.7 * total_amplitude
    tau1_guess = 0.4
    A2_guess = 0.3 * total_amplitude
    tau2_guess = 2.5
    
    p0 = [A1_guess, tau1_guess, A2_guess, tau2_guess]
    
    # Bounds: amplitudes positive, tau1 < tau2 (fast < slow)
    bounds_lower = [0.0, 0.05, 0.0, 0.5]  # tau1 > 0.05s, tau2 > 0.5s
    bounds_upper = [total_amplitude * 2, 2.0, total_amplitude * 2, 10.0]  # tau1 < 2s, tau2 < 10s
    
    try:
        params, _ = curve_fit(
            _double_exponential_model,
            rel_times,
            baseline_corrected,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000,
        )
        A1, tau1, A2, tau2 = params
        
        # Ensure tau1 < tau2 (fast < slow)
        if tau1 > tau2:
            tau1, tau2 = tau2, tau1
            A1, A2 = A2, A1
        
        return float(tau1), float(A1), float(tau2), float(A2)
    
    except (RuntimeError, ValueError):
        # If double-exponential fails, return NaN
        return float(np.nan), float(np.nan), float(np.nan), float(np.nan)


def _estimate_decay_tau_with_baseline(
    times: np.ndarray, 
    values: np.ndarray, 
    peak_index: int,
    baseline: float,
) -> Tuple[float, float]:
    """
    Fit a single exponential decay with baseline: y(t) = baseline + amplitude * exp(-t/tau)
    
    This is kept for backward compatibility but is now deprecated in favor of
    _estimate_double_exponential_decay.
    
    Returns
    -------
    Tuple[float, float]
        (tau, amplitude) or (nan, nan) if fitting fails
    """
    if times.size == 0 or values.size == 0:
        return float(np.nan), float(np.nan)

    peak_value = values[peak_index]
    if np.isnan(peak_value):
        return float(np.nan), float(np.nan)

    post_times = times[peak_index:]
    post_values = values[peak_index:]
    if post_times.size < 3:
        return float(np.nan), float(np.nan)

    # Subtract baseline to get amplitude decay
    baseline_corrected = post_values - baseline
    
    # Initial amplitude is peak minus baseline
    initial_amplitude = peak_value - baseline
    if initial_amplitude <= 0:
        return float(np.nan), float(np.nan)

    rel_times = post_times - post_times[0]
    normalized = baseline_corrected / initial_amplitude

    # Only use positive normalized values for log fitting
    mask = (
        np.isfinite(rel_times)
        & np.isfinite(normalized)
        & (rel_times >= 0.0)
        & (normalized > 0.0)
    )
    mask[0] = False  # exclude the peak itself to avoid log(1)=0 dominating

    if mask.sum() < 2:
        return float(np.nan), float(np.nan)

    try:
        slope, intercept = np.polyfit(rel_times[mask], np.log(normalized[mask]), 1)
    except np.linalg.LinAlgError:
        return float(np.nan), float(np.nan)

    if slope >= 0:
        return float(np.nan), float(np.nan)

    tau = float(-1.0 / slope)
    # Recalculate amplitude from intercept: log(normalized) = log(amp_factor) + slope*t
    amplitude = float(initial_amplitude * np.exp(intercept))
    
    return tau, amplitude


def _estimate_decay_tau(times: np.ndarray, values: np.ndarray, peak_index: int) -> float:
    """
    Legacy function for backward compatibility.
    Estimates tau assuming baseline=0 (for main peak analysis in 0-2s window).
    """
    if times.size == 0 or values.size == 0:
        return float(np.nan)

    peak_value = values[peak_index]
    if np.isnan(peak_value) or peak_value == 0.0:
        return float(np.nan)

    post_times = times[peak_index:]
    post_values = values[peak_index:]
    if post_times.size < 3:
        return float(np.nan)

    rel_times = post_times - post_times[0]
    normalized = post_values / peak_value

    mask = (
        np.isfinite(rel_times)
        & np.isfinite(normalized)
        & (rel_times >= 0.0)
        & (normalized > 0.0)
    )
    mask[0] = False  # exclude the peak itself to avoid log(1)=0 dominating

    if mask.sum() < 2:
        return float(np.nan)

    try:
        slope, intercept = np.polyfit(rel_times[mask], np.log(normalized[mask]), 1)
    except np.linalg.LinAlgError:
        return float(np.nan)

    if slope >= 0:
        return float(np.nan)

    return float(-1.0 / slope)


def _compute_offset_residual_metrics(
    times: np.ndarray,
    values: np.ndarray,
    peak_index: int,
    peak_value: float,
    tau1: float,
    A1: float,
    tau2: float,
    A2: float,
    baseline: float,
    offset_start: float,
    offset_end: float,
    auc_window: Tuple[float, float],
) -> Dict[str, float]:
    """
    Strategy 3: Compute offset response metrics using double-exponential residual analysis.
    
    This function implements double-exponential decay subtraction to detect offset responses
    (late peaks or shoulders) that occur after the main calcium transient.
    
    Algorithm:
    1. Reconstruct expected double-exponential decay from peak using fitted parameters
    2. Subtract fitted decay from observed trace to get residuals
    3. Compute area under curve for positive residuals only (removes negative fluctuations)
    
    Model: y(t) = baseline + A1*exp(-(t-t_peak)/tau1) + A2*exp(-(t-t_peak)/tau2)
    
    Parameters
    ----------
    times : np.ndarray
        Time points (relative to alignment)
    values : np.ndarray
        Signal values (z-scores)
    peak_index : int
        Index of the main peak
    peak_value : float
        Amplitude of the main peak
    tau1 : float
        Fast decay time constant (seconds)
    A1 : float
        Fast component amplitude
    tau2 : float
        Slow decay time constant (seconds)
    A2 : float
        Slow component amplitude
    baseline : float
        Baseline level (pre-stimulus)
    offset_start : float
        Start time for offset analysis (typically 2.0s)
    offset_end : float
        End time for offset analysis (typically 8.0s)
    auc_window : Tuple[float, float]
        Time window for AUC calculation (typically 2.0-3.0s)
        
    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - 'residual_auc': Area under positive residuals
    """
    # Initialize return values
    result = {
        "residual_auc": np.nan,
    }
    
    # Check if double-exponential parameters are valid
    if not np.isfinite(tau1) or tau1 <= 0:
        return result
    if not np.isfinite(A1):
        return result
    
    # Restrict to offset analysis window
    offset_mask = (times >= offset_start) & (times <= offset_end)
    if not offset_mask.any():
        return result
    
    offset_times = times[offset_mask]
    offset_values = values[offset_mask]
    
    # Reconstruct baseline-corrected double-exponential decay:
    # y(t) = baseline + A1*exp(-(t - t_peak)/tau1) + A2*exp(-(t - t_peak)/tau2)
    peak_time = times[peak_index]
    time_from_peak = offset_times - peak_time
    expected_decay = baseline + A1 * np.exp(-time_from_peak / tau1) + A2 * np.exp(-time_from_peak / tau2)
    
    # Compute residuals (observed - expected)
    residuals = offset_values - expected_decay
    
    # Compute area under curve for positive residuals only
    auc_mask = (offset_times >= auc_window[0]) & (offset_times <= auc_window[1])
    if auc_mask.any():
        auc_times = offset_times[auc_mask]
        auc_residuals = residuals[auc_mask]
        
        # Keep only positive residuals
        positive_residuals = np.maximum(auc_residuals, 0.0)
        
        # Integrate using trapezoidal rule
        if positive_residuals.size > 1:
            if np.any(positive_residuals > 0):
                auc = float(np.trapz(positive_residuals, auc_times))
                result["residual_auc"] = auc
            else:
                # No positive residuals found - set to 0 instead of NaN
                result["residual_auc"] = 0.0
        else:
            # Not enough points for integration - set to 0
            result["residual_auc"] = 0.0
    else:
        # No data in AUC window - set to 0
        result["residual_auc"] = 0.0
    
    return result


def _detect_offset_peak_direct(
    times: np.ndarray,
    values: np.ndarray,
    baseline_window: Tuple[float, float],
    detection_start: float,
    detection_end: float,
) -> float:
    """
    Strategy 1: Direct peak detection with baseline subtraction.
    
    This is a fallback method when exponential fitting fails. It detects
    the maximum value after detection_start and subtracts a baseline.
    
    Parameters
    ----------
    times : np.ndarray
        Time points
    values : np.ndarray
        Signal values
    baseline_window : Tuple[float, float]
        Time window for baseline calculation (e.g., 1.9-2.0s)
    detection_start : float
        Start time for peak detection
    detection_end : float
        End time for peak detection
        
    Returns
    -------
    float
        Peak amplitude relative to baseline (NaN if detection fails)
    """
    # Compute baseline
    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    if not baseline_mask.any():
        return np.nan
    baseline_value = float(np.nanmean(values[baseline_mask]))
    
    # Find peak in detection window
    detection_mask = (times >= detection_start) & (times <= detection_end)
    if not detection_mask.any():
        return np.nan
    
    detection_values = values[detection_mask]
    peak_value = float(np.nanmax(detection_values))
    
    # Return amplitude above baseline
    return peak_value - baseline_value


def _logistic_rise_model(t: np.ndarray, baseline: float, amplitude: float, t_half: float, tau: float) -> np.ndarray:
    return baseline + amplitude / (1.0 + np.exp(-(t - t_half) / tau))


def _estimate_logistic_onset(times: np.ndarray, values: np.ndarray) -> Optional[float]:
    if times.size < 6:
        return None

    sorted_idx = np.argsort(times)
    times_sorted = times[sorted_idx]
    values_sorted = values[sorted_idx]

    amplitude_guess = float(np.nanmax(values_sorted) - np.nanmin(values_sorted))
    baseline_guess = float(np.nanpercentile(values_sorted, 5))
    amplitude_guess = float(np.nanmax(values_sorted) - baseline_guess)
    if not np.isfinite(amplitude_guess) or amplitude_guess <= 0:
        return None

    above_half = np.where(values_sorted >= baseline_guess + 0.5 * amplitude_guess)[0]
    if above_half.size:
        t_half_guess = float(times_sorted[above_half[0]])
    else:
        t_half_guess = float(times_sorted[len(times_sorted) // 2])

    tau_guess = max((times_sorted[-1] - times_sorted[0]) / 4.0, 1e-3)

    lower_bounds = [-np.inf, 0.0, times_sorted[0] - 1.0, 1e-3]
    upper_bounds = [np.inf, amplitude_guess * 10.0, times_sorted[-1] + 1.0, max((times_sorted[-1] - times_sorted[0]) * 2.0, 1e-2)]

    try:
        params, _ = curve_fit(
            _logistic_rise_model,
            times_sorted,
            values_sorted,
            p0=[baseline_guess, amplitude_guess, t_half_guess, tau_guess],
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return None

    baseline_fit, amplitude_fit, t_half_fit, tau_fit = params
    if amplitude_fit <= 0 or tau_fit <= 0:
        return None

    onset_time = float(t_half_fit - tau_fit * np.log(9.0))
    if not np.isfinite(onset_time):
        return None

    if onset_time < times_sorted[0] - 0.5:
        onset_time = times_sorted[0]
    if onset_time > times_sorted[-1]:
        onset_time = times_sorted[-1]

    return onset_time


def _prepare_mouse_signal_frame(mouse_df: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    """Return tidy DataFrame with time/value pairs for a given signal."""
    if signal_column not in mouse_df.columns:
        return pd.DataFrame(columns=["time", "value"])

    times = pd.to_numeric(mouse_df.index, errors="coerce")
    values = pd.to_numeric(mouse_df[signal_column], errors="coerce")
    df_signal = pd.DataFrame({"time": times, "value": values}).dropna()

    if df_signal.empty:
        return df_signal

    return df_signal.sort_values("time")


def _compute_signal_metrics_for_mouse(
    mouse_df: pd.DataFrame,
    signal_column: str,
    window: Tuple[float, float],
) -> Dict[str, float]:
    """Compute peak, logistic 10% onset time, and decay tau for a mouse signal.
    
    Note: window should extend through offset analysis period (e.g., 0-8s),
    but main peak detection is restricted to the early period (0-2s).
    """
    df_signal = _prepare_mouse_signal_frame(mouse_df, signal_column)

    if df_signal.empty:
        return {"peak": np.nan, "onset_time": np.nan, "decay_tau1": np.nan}

    window_mask = (df_signal["time"] >= window[0]) & (df_signal["time"] <= window[1])
    window_df = df_signal.loc[window_mask]

    if window_df.empty:
        return {"peak": np.nan, "onset_time": np.nan, "decay_tau1": np.nan}

    times = window_df["time"].to_numpy(dtype=float)
    values = window_df["value"].to_numpy(dtype=float)

    finite_mask = np.isfinite(times) & np.isfinite(values)
    times = times[finite_mask]
    values = values[finite_mask]

    if values.size == 0:
        return {"peak": np.nan, "onset_time": np.nan, "decay_tau1": np.nan}

    smooth_window = min(7, values.size)
    if smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        smoothed = np.convolve(values, kernel, mode="same")
    else:
        smoothed = values.copy()

    candidate = smoothed
    
    # Restrict peak detection to main peak window (0 to POST_ALIGNMENT_WINDOW_DURATION)
    # to avoid detecting offset responses as the main peak
    main_peak_window_end = POST_ALIGNMENT_WINDOW_START + POST_ALIGNMENT_WINDOW_DURATION
    main_peak_mask = times <= main_peak_window_end
    candidate_peak_search = candidate[main_peak_mask]
    times_peak_search = times[main_peak_mask]
    
    if candidate_peak_search.size == 0:
        return {"peak": np.nan, "onset_time": np.nan, "decay_tau1": np.nan}
    
    candidate_std = float(np.nanstd(candidate_peak_search)) if candidate_peak_search.size else 0.0
    prominence = max(0.05, 0.15 * candidate_std) if candidate_std > 0 else 0.05

    peaks, _ = find_peaks(candidate_peak_search, prominence=prominence)
    if peaks.size == 0:
        peaks, _ = find_peaks(candidate_peak_search)

    if peaks.size == 0:
        peak_index_in_search = int(np.nanargmax(candidate_peak_search))
    else:
        best_idx = int(np.argmax(candidate_peak_search[peaks]))
        peak_index_in_search = int(peaks[best_idx])
    
    # Convert back to index in full array
    peak_index = peak_index_in_search

    peak_index = max(0, min(peak_index, candidate_peak_search.size - 1))

    peak_value_raw = values[peak_index]
    peak_value_smooth = candidate_peak_search[peak_index]
    effective_peak_value = max(peak_value_raw, peak_value_smooth)

    if not np.isfinite(effective_peak_value) or effective_peak_value <= 0:
        return {"peak": 0.0, "onset_time": np.nan, "decay_tau1": np.nan}

    peak_value = float(effective_peak_value)

    # Estimate baseline from full data (for offset analysis)
    baseline = _estimate_baseline(times, candidate)
    
    # Fit double-exponential decay using full data (for offset analysis)
    tau1, A1, tau2, A2 = _estimate_double_exponential_decay(
        times, candidate, peak_index, baseline, DECAY_FIT_START_OFFSET
    )
    
    # For main peak metrics, report only tau1 (fast component) as decay_tau1
    decay_tau1 = tau1

    # Use only main peak window for onset detection
    logistic_onset = _estimate_logistic_onset(times_peak_search[: peak_index + 1], candidate_peak_search[: peak_index + 1])

    if logistic_onset is not None:
        onset_time = float(logistic_onset)
    else:
        onset_time = np.nan
        onset_threshold = peak_value * 0.1

        for idx in range(0, peak_index + 1):
            current_value = candidate_peak_search[idx]
            if not np.isfinite(current_value):
                continue
            if current_value >= onset_threshold:
                if idx == 0:
                    onset_time = times_peak_search[idx]
                else:
                    prev_time = times_peak_search[idx - 1]
                    prev_value = candidate_peak_search[idx - 1]
                    if not np.isfinite(prev_value) or prev_value == current_value:
                        onset_time = times_peak_search[idx]
                    else:
                        fraction = (onset_threshold - prev_value) / (current_value - prev_value)
                        onset_time = prev_time + fraction * (times_peak_search[idx] - prev_time)
                break

    # Compute offset response metrics (Strategy 3: Double-exponential residual analysis)
    offset_metrics = _compute_offset_residual_metrics(
        times,
        candidate,
        peak_index,
        peak_value,
        tau1,
        A1,
        tau2,
        A2,
        baseline,
        OFFSET_ANALYSIS_START,
        OFFSET_ANALYSIS_END,
        OFFSET_AUC_WINDOW,
    )
    
    # Fallback to direct peak detection if residual analysis fails
    offset_peak_amplitude = np.nan
    if not np.isfinite(offset_metrics.get("residual_max", np.nan)):
        offset_peak_amplitude = _detect_offset_peak_direct(
            times,
            values,
            OFFSET_BASELINE_WINDOW,
            OFFSET_ANALYSIS_START,
            OFFSET_ANALYSIS_END,
        )
    
    # Compute main peak residual AUC (residuals in the fitting window)
    main_peak_residual_auc = np.nan
    if np.isfinite(decay_tau1) and decay_tau1 > 0 and np.isfinite(peak_value):
        main_peak_start = POST_ALIGNMENT_WINDOW_START
        main_peak_end = POST_ALIGNMENT_WINDOW_START + POST_ALIGNMENT_WINDOW_DURATION
        main_peak_mask = (times >= main_peak_start) & (times <= main_peak_end) & (times >= times[peak_index])
        
        if main_peak_mask.any():
            main_peak_times = times[main_peak_mask]
            main_peak_values = values[main_peak_mask]
            
            # Reconstruct exponential fit for main peak window (using fast tau)
            peak_time = times[peak_index]
            time_from_peak = main_peak_times - peak_time
            expected_decay = peak_value * np.exp(-time_from_peak / decay_tau1)
            
            # Compute residuals
            main_peak_residuals = main_peak_values - expected_decay
            
            # Integrate positive residuals only (like offset_residual_auc)
            positive_residuals = np.maximum(main_peak_residuals, 0.0)
            
            if main_peak_times.size > 1:
                if np.any(positive_residuals > 0):
                    main_peak_residual_auc = float(np.trapz(positive_residuals, main_peak_times))
                else:
                    # No positive residuals - set to 0 instead of NaN
                    main_peak_residual_auc = 0.0
            else:
                # Not enough points - set to 0
                main_peak_residual_auc = 0.0
        else:
            # No data in window - set to 0
            main_peak_residual_auc = 0.0
    
    return {
        "peak": peak_value,
        "onset_time": float(onset_time),
        "decay_tau1": float(decay_tau1),
        "main_peak_residual_auc": main_peak_residual_auc,
        "offset_residual_auc": offset_metrics.get("residual_auc", np.nan),
        "offset_peak_amplitude": offset_peak_amplitude,
    }


def sem(values) -> float:
    """Compute standard error of the mean, ignoring NaNs."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n <= 1:
        return float(np.nan)
    return float(arr.std(ddof=1) / np.sqrt(n))

def compute_signal_metrics_from_results(
    mean_data_per_mouse: Dict[str, pd.DataFrame],
    signal_columns: list[str],
    window: Tuple[float, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-mouse and summary signal metrics directly from loaded results."""
    records = []

    for mouse_id, mouse_df in mean_data_per_mouse.items():
        if mouse_df is None or mouse_df.empty:
            continue

        for signal in signal_columns:
            metrics = _compute_signal_metrics_for_mouse(mouse_df, signal, window)
            records.append({"mouse": mouse_id, "signal": signal, **metrics})

    metrics_df = pd.DataFrame(records)
    if metrics_df.empty:
        return metrics_df, pd.DataFrame(), pd.DataFrame()

    for feature_name in SIGNAL_METRIC_FEATURES:
        if feature_name not in metrics_df.columns:
            metrics_df[feature_name] = np.nan

    summary_records = []
    for signal_name, group in metrics_df.groupby("signal"):
        for metric_name in SIGNAL_METRIC_FEATURES:
            values = pd.to_numeric(group[metric_name], errors="coerce").dropna()
            summary_records.append(
                {
                    "signal": signal_name,
                    "metric": metric_name,
                    "n": int(values.size),
                    "mean": float(values.mean()) if values.size else np.nan,
                    "sem": sem(values) if values.size else np.nan,
                }
            )

    summary_df = pd.DataFrame(summary_records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["signal", "metric"]).reset_index(drop=True)
        summary_pivot = summary_df.pivot(index="metric", columns="signal", values=["mean", "sem"])
    else:
        summary_pivot = pd.DataFrame()

    return metrics_df, summary_df, summary_pivot

def save_signal_metrics_to_csv(
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    file_prefix: str,
    cohort: Optional[str] = None,
    experiment_day: Optional[str] = None,
) -> Path:
    """Save per-mouse signal metrics to disk (summary is not written)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_save = metrics.copy()
    if cohort:
        metrics_to_save.insert(0, "Cohort", cohort)
    if experiment_day:
        insert_idx = 1 if cohort else 0
        metrics_to_save.insert(insert_idx, "Experiment_Day", experiment_day)

    per_mouse_path = output_dir / f"{file_prefix}_calcium_analysis_per_mouse_metrics.csv"
    metrics_to_save.to_csv(per_mouse_path, index=False)

    return per_mouse_path


def _make_signal_metrics_file_prefix(event_name: str) -> str:
    """Create a filesystem-friendly prefix from the configured event name."""
    if not event_name:
        return "signal_metrics"
    label = event_name.replace(".csv", "")
    label = label.strip("_")
    label = label.replace(" ", "_")
    return label or "signal_metrics"

def compute_signal_window_mean_per_mouse(
    mean_data_per_mouse: Dict[str, pd.DataFrame],
    signal_column: str,
    window_start: float,
    window_end: float,
) -> pd.DataFrame:
    """Compute per-mouse mean of a signal within a specified time window."""
    records = []
    for mouse_id, mouse_df in mean_data_per_mouse.items():
        if mouse_df is None or signal_column not in mouse_df.columns:
            continue

        times = pd.to_numeric(mouse_df.index, errors="coerce")
        if np.all(np.isnan(times)):
            continue

        mask = (times >= window_start) & (times <= window_end)
        if not mask.any():
            continue

        window_values = pd.to_numeric(mouse_df.loc[mask, signal_column], errors="coerce").dropna()
        if window_values.empty:
            continue

        records.append(
            {
                "mouse": mouse_id,
                "mean": float(window_values.mean()),
                "n_samples": int(window_values.size),
            }
        )

    return pd.DataFrame(records)


def extract_mouse_dataframes_from_loaded(
    loaded_data: Dict[object, Dict[str, object]],
    signal_columns: Optional[Iterable[str]] = None,
    time_column: str = "Time (s)",
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path]]:
    """Convert the raw loaded data into per-mouse DataFrames and track their origins."""
    mouse_frames: Dict[str, list[pd.DataFrame]] = {}
    mouse_sources: Dict[str, Path] = {}

    for entry in loaded_data.values():
        mouse_name = entry.get("mouse_name") if isinstance(entry, dict) else None
        if not mouse_name:
            continue

        frames: List[pd.DataFrame] = []
        if isinstance(entry, dict):
            records = entry.get("records", [])
            if records:
                for record in records:
                    df = record.get("dataframe")
                    if isinstance(df, pd.DataFrame):
                        frames.append(df.copy())
            elif entry.get("dataframes"):
                raw_frames = entry.get("dataframes", [])
                for df in raw_frames:
                    if isinstance(df, pd.DataFrame):
                        frames.append(df.copy())
        else:
            columns = [key for key in entry.keys() if key not in {"mouse_name", "data_path"}]  # type: ignore[union-attr]
            if time_column in entry:
                columns = [time_column] + [col for col in columns if col != time_column]
            if signal_columns is not None:
                allowed = set(signal_columns)
                columns = [col for col in columns if col == time_column or col in allowed]
            data = {col: entry[col] for col in columns if col in entry}
            if data:
                frames = [pd.DataFrame(data)]

        if not frames:
            continue

        cleaned_frames: List[pd.DataFrame] = []
        for df in frames:
            if time_column in df.columns:
                df = df.set_index(time_column)
                df.index.name = time_column
            df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            if df.empty:
                continue
            cleaned_frames.append(df.sort_index())

        if not cleaned_frames:
            continue

        mouse_frames.setdefault(mouse_name, []).extend(cleaned_frames)

        data_paths = []
        if isinstance(entry, dict):
            records = entry.get("records", [])
            if records:
                for record in records:
                    aligned_dir = record.get("aligned_dir")
                    if aligned_dir:
                        data_paths.append(Path(aligned_dir))
            if not data_paths:
                data_paths = entry.get("aligned_dirs") or []
            if not data_paths and entry.get("data_path"):
                data_paths = [entry["data_path"]]
        if data_paths and mouse_name not in mouse_sources:
            mouse_sources[mouse_name] = Path(data_paths[0]).resolve()

    consolidated: Dict[str, pd.DataFrame] = {}
    for mouse, frames in mouse_frames.items():
        if len(frames) == 1:
            consolidated[mouse] = frames[0]
        else:
            combined = pd.concat(frames)
            combined = combined.groupby(combined.index).mean().sort_index()
            consolidated[mouse] = combined

    return consolidated, mouse_sources



def assign_mouse_colors_consistent(mouse_ids: Iterable[str]) -> Dict[str, tuple]:
    """Assign consistent colors to mice using the gnuplot2 palette."""
    normalized_ids = [str(mouse) for mouse in mouse_ids]
    mouse_list = sorted(dict.fromkeys(normalized_ids))
    if not mouse_list:
        return OrderedDict()
    count = len(mouse_list)
    if count == 1:
        sample_points = np.array([0.6])
    else:
        sample_points = np.linspace(0.12, 0.95, count)
    color_map = plt.cm.gnuplot2
    colors: list[tuple] = []
    for point in sample_points:
        candidate = color_map(point)
        attempts = 0
        while np.all(np.asarray(candidate[:3]) <= 0.15) and attempts < 5:
            point = min(point + 0.08, 0.99)
            candidate = color_map(point)
            attempts += 1
        colors.append(candidate)
    return OrderedDict((mouse, colors[idx]) for idx, mouse in enumerate(mouse_list))


def cm_to_inches(*dimensions_cm: float) -> Tuple[float, ...]:
    """Convert one or more centimetre measurements to inches."""
    return tuple(dim / 2.54 for dim in dimensions_cm)


def holm_adjust(p_values: List[float]) -> List[float]:
    """Perform Holm-Bonferroni correction without statsmodels."""
    if not p_values:
        return []
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, (m - rank) * p_values[idx])
    # Ensure monotonicity
    for i in range(m - 2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])
    return adjusted.tolist()


def _discover_metric_csvs(
    data_dirs: Iterable[Path],
    event_name: Optional[str],
) -> list[Path]:
    """Search data directories for per-mouse metrics CSVs."""
    discovered: list[Path] = []
    candidate_names: list[str] = []
    event_label = None
    if event_name:
        prefix = _make_signal_metrics_file_prefix(event_name)
        candidate_names.append(f"{prefix}_calcium_analysis_per_mouse_metrics.csv")
        event_label = prefix.lower()
    candidate_names.append("calcium_analysis_per_mouse_metrics.csv")

    for base in data_dirs:
        base_path = Path(base).expanduser()
        if not base_path.exists():
            continue
        # Direct name matches first
        for name in candidate_names:
            direct = base_path / name
            if direct.name.startswith("._"):
                continue
            if event_label and event_label not in direct.stem.lower():
                continue
            if direct.exists():
                discovered.append(direct.resolve())
        # Recursive search (limit depth to avoid huge scans)
        for csv_file in base_path.rglob("*_calcium_analysis_per_mouse_metrics.csv"):
            if csv_file.name.startswith("._"):
                continue
            if event_label and event_label not in csv_file.stem.lower():
                continue
            discovered.append(csv_file.resolve())
    # Remove duplicates while preserving order
    unique: list[Path] = []
    seen = set()
    for path in discovered:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def compute_condition_signal_metrics(
    condition_label: str,
    condition_cfg: Dict[str, object],
    selected_mice: List[str],
    allowed_mice: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """Compute signal metrics for a given condition (event + data directories)."""
    event_name_cfg = condition_cfg.get("event_name")
    metrics_csvs_cfg = condition_cfg.get("metrics_csvs", [])
    data_dirs_cfg = condition_cfg.get("data_dirs", [])

    metrics_csv_paths: list[Path] = []
    for csv_entry in metrics_csvs_cfg:
        csv_path = Path(csv_entry).expanduser()
        if csv_path.exists():
            metrics_csv_paths.append(csv_path.resolve())
        else:
            print(f"⚠️ Metrics CSV not found for condition '{condition_label}': {csv_path}")

    if not metrics_csv_paths and data_dirs_cfg:
        discovered = _discover_metric_csvs(data_dirs_cfg, event_name_cfg)
        if discovered:
            metrics_csv_paths.extend(discovered)

    if metrics_csv_paths:
        frames: list[pd.DataFrame] = []
        sources: Dict[str, Path] = {}
        for csv_path in metrics_csv_paths:
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ Failed to read metrics CSV '{csv_path}': {exc}")
                continue
            if "mouse" not in df.columns or "signal" not in df.columns:
                print(f"⚠️ CSV '{csv_path}' missing required columns ('mouse', 'signal'). Skipping.")
                continue
            df["mouse"] = df["mouse"].astype(str)
            if selected_mice:
                df = df[df["mouse"].isin(selected_mice)]
            elif allowed_mice:
                df = df[df["mouse"].isin(allowed_mice)]
            if df.empty:
                continue
            numeric_cols = [col for col in df.columns if col not in {"mouse", "signal", "Cohort", "Experiment_Day"}]
            df_filtered = df[["mouse", "signal"] + numeric_cols].copy()
            df_filtered["condition"] = condition_label
            df_filtered["condition_label"] = condition_cfg.get("label", condition_label)
            df_filtered["event_name"] = event_name_cfg or csv_path.stem
            df_filtered["source_csv"] = str(csv_path)
            frames.append(df_filtered)
            csv_parent = csv_path.parent
            for mouse_name in df_filtered["mouse"].unique():
                sources.setdefault(mouse_name, csv_parent)
        if not frames:
            print(f"⚠️ No usable entries found in metrics CSVs for condition '{condition_label}'.")
            return pd.DataFrame(), {}

        metrics_df = pd.concat(frames, ignore_index=True)
        value_columns = [
            col for col in metrics_df.columns
            if col not in {"mouse", "signal", "condition", "condition_label", "event_name", "source_csv"}
        ]
        grouped = (
            metrics_df.groupby(
                ["mouse", "signal", "condition", "condition_label", "event_name"],
                as_index=False,
            )[value_columns]
            .mean(numeric_only=True)
        )
        return grouped, sources

    # Fall back to loading aligned data and computing metrics if CSVs not available
    if not data_dirs_cfg:
        print(f"⚠️ No data directories configured for condition '{condition_label}'.")
        return pd.DataFrame(), {}

    loaded = load_aligned_data(
        data_dirs_cfg,
        event_name_cfg,
        selected_mice,
        allowed_mice,
    )
    if not loaded:
        print(f"⚠️ No data loaded for condition '{condition_label}'.")
        return pd.DataFrame(), {}

    frames, sources = extract_mouse_dataframes_from_loaded(
        loaded,
        signal_columns=SIGNAL_METRIC_COLUMNS,
    )
    if not frames:
        print(f"⚠️ No signal frames available for condition '{condition_label}'.")
        return pd.DataFrame(), sources

    # Analysis window must extend through offset analysis period to capture all data
    metrics_df, _, _ = compute_signal_metrics_from_results(
        frames,
        SIGNAL_METRIC_COLUMNS,
        (POST_ALIGNMENT_WINDOW_START, OFFSET_ANALYSIS_END),
    )
    if metrics_df.empty:
        print(f"⚠️ No signal metrics computed for condition '{condition_label}'.")
        return metrics_df, sources

    window_start = POST_ALIGNMENT_WINDOW_START + 2.0
    window_end = POST_ALIGNMENT_WINDOW_START + 8.0
    z560_window_stats = compute_signal_window_mean_per_mouse(
        frames,
        "z_560_Baseline",
        window_start,
        window_end,
    )
    if not z560_window_stats.empty and "z_560_Baseline" in metrics_df["signal"].unique():
        mean_map = z560_window_stats.set_index("mouse")["mean"]
        mask = metrics_df["signal"] == "z_560_Baseline"
        metrics_df.loc[mask, "mean_fluorescence_2_to_8s"] = metrics_df.loc[
            mask, "mouse"
        ].map(mean_map)

    metrics_df["condition"] = condition_label
    metrics_df["condition_label"] = condition_cfg.get("label", condition_label)
    metrics_df["event_name"] = event_name_cfg

    return metrics_df, sources


def _paired_sem(values: pd.Series) -> float:
    """Compute SEM for paired samples."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def compute_repeated_measures_stats(
    metric_df: pd.DataFrame,
    condition_order: List[str],
    signal_order: List[str],
    metric_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Run repeated or non-paired ANOVA depending on data completeness."""
    notes: List[str] = []
    summary_records: List[Dict[str, object]] = []
    pairwise_records: List[Dict[str, object]] = []

    clean_df = metric_df.dropna(subset=["value"]).copy()
    if clean_df.empty:
        notes.append(f"No valid data available for metric '{metric_name}'.")
        return pd.DataFrame(), pd.DataFrame(), notes

    condition_levels_present = [
        cond for cond in condition_order if cond in clean_df["condition"].unique()
    ]
    if not condition_levels_present:
        condition_levels_present = sorted(clean_df["condition"].unique())

    signal_levels_present = [
        sig for sig in signal_order if sig in clean_df["signal"].unique()
    ]
    if not signal_levels_present:
        signal_levels_present = sorted(clean_df["signal"].unique())

    clean_df["Condition"] = pd.Categorical(
        clean_df["condition"], categories=condition_levels_present, ordered=True
    )
    clean_df["Signal"] = pd.Categorical(
        clean_df["signal"], categories=signal_levels_present, ordered=True
    )
    clean_df["mouse"] = clean_df["mouse"].astype(str)
    clean_df["Mouse"] = clean_df["mouse"]
    clean_df["Value"] = clean_df["value"]

    mice_per_condition = {
        cond: sorted(set(clean_df.loc[clean_df["condition"] == cond, "Mouse"]))
        for cond in condition_levels_present
    }
    unique_counts = {cond: len(mice) for cond, mice in mice_per_condition.items()}
    same_count = bool(unique_counts) and len(set(unique_counts.values())) == 1

    if same_count:
        common_mice = set.intersection(
            *(set(mice) for mice in mice_per_condition.values())
        )
        expected_count = next(iter(unique_counts.values()))
        paired_design = len(common_mice) == expected_count and expected_count > 0
        if not paired_design:
            notes.append(
                "Mice counts match across conditions, but identities differ; "
                "using non-paired analyses."
            )
    else:
        common_mice = set()
        paired_design = False
        notes.append(
            "Detected unequal mouse counts across conditions; switching to non-paired analyses."
        )

    anova_performed = False

    if paired_design and STATS_MODELS_AVAILABLE:
        try:
            anova_model = AnovaRM(
                clean_df,
                depvar="Value",
                subject="Mouse",
                within=["Condition", "Signal"],
            )
            anova_result = anova_model.fit()
            anova_table = anova_result.anova_table.reset_index().rename(
                columns={"index": "effect"}
            )
            for _, row in anova_table.iterrows():
                summary_records.append(
                    {
                        "analysis": "AnovaRM",
                        "metric": metric_name,
                        "effect": row["effect"],
                        "num_df": row.get("Num DF"),
                        "den_df": row.get("Den DF"),
                        "F": row.get("F Value"),
                        "p_value": row.get("Pr > F"),
                    }
                )
            anova_performed = True
        except Exception as exc:  # noqa: BLE001
            notes.append(
                f"Repeated measures ANOVA failed for metric '{metric_name}': {exc}"
            )

    if paired_design and not anova_performed:
        pivot = clean_df.pivot_table(
            index="mouse", columns=["signal", "condition"], values="value"
        )
        pivot = pivot.dropna()
        if pivot.shape[0] >= 3:
            try:
                fried_stat, fried_p = friedmanchisquare(
                    *[pivot[col] for col in pivot.columns]
                )
                summary_records.append(
                    {
                        "analysis": "Friedman",
                        "metric": metric_name,
                        "effect": "condition_x_signal",
                        "df": pivot.shape[1] - 1,
                        "statistic": fried_stat,
                        "p_value": fried_p,
                    }
                )
                notes.append(
                    "Used Friedman test because repeated measures ANOVA was unavailable."
                )
                anova_performed = True
            except Exception as exc:  # noqa: BLE001
                notes.append(
                    f"Friedman test failed for metric '{metric_name}': {exc}"
                )
        else:
            notes.append(
                "Not enough paired samples for Friedman test "
                f"({pivot.shape[0]} mice with complete data)."
            )

    if not paired_design and STATS_MODELS_AVAILABLE and not anova_performed:
        try:
            between_df = clean_df.copy()
            between_df["Condition"] = between_df["Condition"].astype(str)
            between_df["Signal"] = between_df["Signal"].astype(str)
            n_signals = between_df["Signal"].nunique()
            n_conditions = between_df["Condition"].nunique()
            if n_conditions < 2 and n_signals < 2:
                notes.append(
                    "Insufficient variability across conditions/signals for non-paired ANOVA."
                )
            else:
                if n_signals > 1 and n_conditions > 1:
                    formula = "Value ~ C(Condition) * C(Signal)"
                    analysis_label = "TwoWay_ANOVA"
                elif n_signals > 1:
                    formula = "Value ~ C(Signal)"
                    analysis_label = "OneWay_ANOVA_Signal"
                else:
                    formula = "Value ~ C(Condition)"
                    analysis_label = "OneWay_ANOVA_Condition"
                model = ols(formula, data=between_df).fit()
                anova_table = anova_lm(model, typ=2).reset_index().rename(
                    columns={"index": "effect", "df": "df", "PR(>F)": "p_value"}
                )
                for _, row in anova_table.iterrows():
                    summary_records.append(
                        {
                            "analysis": analysis_label,
                            "metric": metric_name,
                            "effect": row["effect"],
                            "df": row.get("df"),
                            "F": row.get("F"),
                            "p_value": row.get("p_value"),
                        }
                    )
                anova_performed = True
        except Exception as exc:  # noqa: BLE001
            notes.append(
                f"Non-paired ANOVA failed for metric '{metric_name}': {exc}"
            )

    if not anova_performed:
        notes.append("No omnibus statistical test could be completed for this metric.")

    if paired_design:
        for signal in signal_levels_present:
            subset = clean_df[clean_df["signal"] == signal]
            pivot = subset.pivot(index="mouse", columns="condition", values="value")
            pivot = pivot.dropna()
            if len(condition_levels_present) >= 2:
                cond_pairs = [
                    (condition_levels_present[i], condition_levels_present[j])
                    for i in range(len(condition_levels_present))
                    for j in range(i + 1, len(condition_levels_present))
                ]
            else:
                cond_pairs = []
            for cond_a, cond_b in cond_pairs:
                if cond_a not in pivot.columns or cond_b not in pivot.columns:
                    continue
                paired = pivot[[cond_a, cond_b]].dropna()
                if paired.shape[0] < 2:
                    notes.append(
                        f"Not enough data for paired comparison between {cond_a} and {cond_b} "
                        f"in signal {signal} (n={paired.shape[0]})."
                    )
                    continue
                diff = paired[cond_a] - paired[cond_b]
                try:
                    t_stat, p_val = ttest_rel(paired[cond_a], paired[cond_b])
                    test_used = "paired_ttest"
                except Exception:  # noqa: BLE001
                    try:
                        t_stat, p_val = wilcoxon(paired[cond_a], paired[cond_b])
                        test_used = "wilcoxon"
                    except Exception as exc:  # noqa: BLE001
                        notes.append(
                            f"Failed to compute paired test for {cond_a} vs {cond_b} "
                            f"in signal {signal}: {exc}"
                        )
                        continue
                effect = (
                    diff.mean() / diff.std(ddof=1)
                    if diff.std(ddof=1) not in (0, np.nan)
                    else np.nan
                )
                pairwise_records.append(
                    {
                        "comparison": "condition_within_signal",
                        "metric": metric_name,
                        "signal": signal,
                        "level_a": cond_a,
                        "level_b": cond_b,
                        "statistic": t_stat,
                        "p_value": p_val,
                        "n": paired.shape[0],
                        "effect_size_cohens_d": effect,
                        "test": test_used,
                    }
                )

        if len(signal_levels_present) >= 2:
            signal_pairs = [
                (signal_levels_present[i], signal_levels_present[j])
                for i in range(len(signal_levels_present))
                for j in range(i + 1, len(signal_levels_present))
            ]
        else:
            signal_pairs = []

        for condition in condition_levels_present:
            subset = clean_df[clean_df["condition"] == condition]
            pivot = subset.pivot(index="mouse", columns="signal", values="value").dropna()
            for signal_a, signal_b in signal_pairs:
                if signal_a not in pivot.columns or signal_b not in pivot.columns:
                    continue
                paired = pivot[[signal_a, signal_b]].dropna()
                if paired.shape[0] < 2:
                    notes.append(
                        f"Not enough data for paired comparison between {signal_a} and "
                        f"{signal_b} within condition {condition} (n={paired.shape[0]})."
                    )
                    continue
                diff = paired[signal_a] - paired[signal_b]
                try:
                    t_stat, p_val = ttest_rel(paired[signal_a], paired[signal_b])
                    test_used = "paired_ttest"
                except Exception:  # noqa: BLE001
                    try:
                        t_stat, p_val = wilcoxon(paired[signal_a], paired[signal_b])
                        test_used = "wilcoxon"
                    except Exception as exc:  # noqa: BLE001
                        notes.append(
                            f"Failed to compute signal comparison within {condition}: {exc}"
                        )
                        continue
                effect = (
                    diff.mean() / diff.std(ddof=1)
                    if diff.std(ddof=1) not in (0, np.nan)
                    else np.nan
                )
                pairwise_records.append(
                    {
                        "comparison": "signal_within_condition",
                        "metric": metric_name,
                        "condition": condition,
                        "level_a": signal_a,
                        "level_b": signal_b,
                        "statistic": t_stat,
                        "p_value": p_val,
                        "n": paired.shape[0],
                        "effect_size_cohens_d": effect,
                        "test": test_used,
                    }
                )
    else:
        for signal in signal_levels_present:
            subset = clean_df[clean_df["signal"] == signal]
            if len(condition_levels_present) >= 2:
                cond_pairs = [
                    (condition_levels_present[i], condition_levels_present[j])
                    for i in range(len(condition_levels_present))
                    for j in range(i + 1, len(condition_levels_present))
                ]
            else:
                cond_pairs = []
            for cond_a, cond_b in cond_pairs:
                group_a = subset[subset["condition"] == cond_a]["value"].dropna().to_numpy(dtype=float)
                group_b = subset[subset["condition"] == cond_b]["value"].dropna().to_numpy(dtype=float)
                if group_a.size < 2 or group_b.size < 2:
                    notes.append(
                        f"Not enough data for unpaired comparison between {cond_a} and {cond_b} "
                        f"in signal {signal} (n={group_a.size} vs {group_b.size})."
                    )
                    continue
                t_stat, p_val = ttest_ind(group_a, group_b, equal_var=False)
                var_a = group_a.var(ddof=1)
                var_b = group_b.var(ddof=1)
                pooled_denom = ((group_a.size - 1) * var_a + (group_b.size - 1) * var_b)
                pooled_denom = pooled_denom / (group_a.size + group_b.size - 2) if (group_a.size + group_b.size - 2) > 0 else np.nan
                pooled_sd = np.sqrt(pooled_denom) if pooled_denom > 0 else np.nan
                effect = (
                    (group_a.mean() - group_b.mean()) / pooled_sd
                    if pooled_sd not in (0, np.nan)
                    else np.nan
                )
                pairwise_records.append(
                    {
                        "comparison": "condition_within_signal",
                        "metric": metric_name,
                        "signal": signal,
                        "level_a": cond_a,
                        "level_b": cond_b,
                        "statistic": t_stat,
                        "p_value": p_val,
                        "n": int(group_a.size + group_b.size),
                        "effect_size_cohens_d": effect,
                        "test": "unpaired_ttest",
                    }
                )
        notes.append(
            "Pairwise comparisons performed with independent-samples tests due to missing mice across conditions."
        )

    if pairwise_records:
        p_vals = [rec["p_value"] for rec in pairwise_records]
        if multipletests is not None:
            _, p_adj, _, _ = multipletests(p_vals, method="holm")
        else:
            p_adj = holm_adjust(p_vals)
            notes.append("Applied manual Holm correction (statsmodels not available).")
        for rec, adj in zip(pairwise_records, p_adj):
            rec["p_value_holm"] = adj
            rec["significant_0_05"] = adj < 0.05

    summary_df = pd.DataFrame(summary_records)
    pairwise_df = pd.DataFrame(pairwise_records)
    return summary_df, pairwise_df, notes


def plot_condition_metric(
    metric_name: str,
    metric_df: pd.DataFrame,
    condition_order: List[str],
    signal_order: List[str],
    mouse_colors: Dict[str, tuple],
    connect_lines: bool = True,
) -> plt.Figure:
    """Generate paired plots for a given metric across conditions."""
    signals_present = [sig for sig in signal_order if sig in metric_df["signal"].unique()]
    if not signals_present:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            f"No data available for metric {metric_name}",
            ha="center",
            va="center",
        )
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(
        1,
        len(signals_present),
        figsize=(6 * len(signals_present), 5),
        sharey=True,
    )
    if len(signals_present) == 1:
        axes = [axes]

    for ax, signal in zip(axes, signals_present):
        subset = metric_df[metric_df["signal"] == signal]
        for mouse, color in mouse_colors.items():
            mouse_data = (
                subset[subset["mouse"] == mouse]
                .set_index("condition")
                .reindex(condition_order)
            )
            mouse_series = mouse_data["value"]
            if connect_lines and mouse_series.isna().any():
                continue
            valid_series = mouse_series.dropna()
            if valid_series.empty:
                continue
            x_indices = np.array([condition_order.index(cond) for cond in valid_series.index])
            y_values = valid_series.values
            if connect_lines:
                ax.plot(
                    x_indices,
                    y_values,
                    marker="o",
                    linewidth=1.5,
                    alpha=0.8,
                    color=color,
                    label=mouse,
                )
            else:
                jitter_seed = (hash((mouse, signal)) % 11) - 5
                jitter = jitter_seed * 0.03
                ax.scatter(
                    x_indices + jitter,
                    y_values,
                    marker="o",
                    s=55,
                    alpha=0.85,
                    color=color,
                    label=mouse,
                )
        means = subset.groupby("condition")["value"].mean().reindex(condition_order)
        sems = subset.groupby("condition")["value"].apply(_paired_sem).reindex(
            condition_order
        )
        x_axis = np.arange(len(condition_order))
        errorbar_common = dict(
            yerr=sems.values,
            color="black",
            linewidth=2.5,
            capsize=6,
            capthick=2,
            markersize=10,
            label="Mean ± SEM",
        )
        if connect_lines:
            mean_container = ax.errorbar(
                x_axis,
                means.values,
                fmt="-o",
                **errorbar_common,
            )
        else:
            mean_container = ax.errorbar(
                x_axis,
                means.values,
                fmt="o",
                linestyle="none",
                **errorbar_common,
            )
        ax.set_xticks(np.arange(len(condition_order)))
        ax.set_xticklabels(
            [subset[subset["condition"] == cond]["condition_label"].iloc[0]
             if not subset[subset["condition"] == cond].empty else cond
             for cond in condition_order],
            rotation=15,
            ha="right",
        )
        ax.set_title(signal.replace("_Baseline", "").replace("_", " "))
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_xlabel("Condition")
    axes[0].set_ylabel(metric_name.replace("_", " ").title())

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=min(len(unique), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.suptitle(f"{metric_name.replace('_', ' ').title()} Comparison", y=1.02)
    return fig



# %%
# RUN SIGNAL METRIC ANALYSIS
#---------------------------------------------------------------------------------------------------#

mouse_signal_frames_by_day: Dict[str, Dict[str, pd.DataFrame]] = {}
signal_metrics_by_day: Dict[str, pd.DataFrame] = {}
signal_metrics_summary_by_day: Dict[str, pd.DataFrame] = {}
signal_metrics_summary_pivot_by_day: Dict[str, pd.DataFrame] = {}
per_day_output_dirs: Dict[str, Path] = {}

def _prepare_frame_for_signals(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "Time (s)" not in df.columns:
        return None
    working = df.copy()
    working["Time (s)"] = pd.to_numeric(working["Time (s)"], errors="coerce")
    working = working.dropna(subset=["Time (s)"])
    if working.empty:
        return None
    working = working.sort_values("Time (s)")
    working = working.set_index("Time (s)")
    working.index.name = "Time (s)"
    working = working.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if working.empty:
        return None
    return working

if 'loaded_data' in locals() and loaded_data:
    per_day_frames = defaultdict(lambda: defaultdict(list))
    per_day_sources: Dict[str, set[Path]] = defaultdict(set)

    for mouse_name, entry in loaded_data.items():
        if not isinstance(entry, dict):
            continue
        records = entry.get("records", [])
        for record in records:
            df = record.get("dataframe")
            if not isinstance(df, pd.DataFrame):
                continue
            prepared = _prepare_frame_for_signals(df)
            if prepared is None:
                continue
            day_label = record.get("experiment_day") or extract_experiment_day(
                record.get("experiment_dir")
            )
            per_day_frames[day_label][mouse_name].append(prepared)

            experiment_dir = record.get("experiment_dir")
            aligned_dir = record.get("aligned_dir")
            if experiment_dir:
                per_day_sources[day_label].add(Path(experiment_dir))
            elif aligned_dir:
                per_day_sources[day_label].add(Path(aligned_dir).parent.parent)

    for day_label, mouse_map in per_day_frames.items():
        consolidated: Dict[str, pd.DataFrame] = {}
        for mouse, frames in mouse_map.items():
            if not frames:
                continue
            combined = pd.concat(frames)
            combined = combined.groupby(combined.index).mean().sort_index()
            consolidated[mouse] = combined
        if consolidated:
            mouse_signal_frames_by_day[day_label] = consolidated
            if per_day_sources.get(day_label):
                per_day_output_dirs[day_label] = next(iter(per_day_sources[day_label]))

results_available = locals().get("results")
if not mouse_signal_frames_by_day and isinstance(results_available, dict):
    fallback_frames = results_available.get("mean_data_per_mouse", {})
    if fallback_frames:
        mouse_signal_frames_by_day["combined"] = fallback_frames
        source_map = results_available.get("mouse_to_data_path", {})
        if isinstance(source_map, dict):
            for mouse, path in source_map.items():
                if path:
                    per_day_output_dirs.setdefault("combined", Path(path).parent.parent)

if not mouse_signal_frames_by_day:
    print("⏭️  Skipping signal metric analysis (no loaded data available).")
else:
    # Analysis window must extend through offset analysis period to capture all data
    analysis_window = (POST_ALIGNMENT_WINDOW_START, OFFSET_ANALYSIS_END)
    window_str = f"[{analysis_window[0]:.2f}, {analysis_window[1]:.2f}] s"
    print(f"\n{'='*60}")
    print(f"SIGNAL METRIC ANALYSIS (window = {window_str})")
    print(f"Note: Main peak detection uses 0-{POST_ALIGNMENT_WINDOW_DURATION}s, offset analysis uses {OFFSET_ANALYSIS_START}-{OFFSET_ANALYSIS_END}s")
    print(f"{'='*60}")

    file_prefix = _make_signal_metrics_file_prefix(event_name)
    window_start = POST_ALIGNMENT_WINDOW_START + 2.0
    window_end = POST_ALIGNMENT_WINDOW_START + 6.0

    for day_label, mouse_frames in mouse_signal_frames_by_day.items():
        if not mouse_frames:
            continue

        metrics_df, summary_df, summary_pivot = compute_signal_metrics_from_results(
            mouse_frames,
            SIGNAL_METRIC_COLUMNS,
            analysis_window,
        )

        z560_window_stats = compute_signal_window_mean_per_mouse(
            mouse_frames,
            "z_560_Baseline",
            window_start,
            window_end,
        )

        if not z560_window_stats.empty:
            z560_summary = pd.DataFrame(
                [
                    {
                        "signal": "z_560_Baseline",
                        "metric": "mean_fluorescence_2_to_8s",
                        "n": int(z560_window_stats["mean"].count()),
                        "mean": float(z560_window_stats["mean"].mean()),
                        "sem": sem(z560_window_stats["mean"]),
                    }
                ]
            )
            summary_df = pd.concat([summary_df, z560_summary], ignore_index=True)
            mask = metrics_df["signal"] == "z_560_Baseline"
            metrics_df.loc[mask, "mean_fluorescence_2_to_8s"] = metrics_df.loc[
                mask, "mouse"
            ].map(z560_window_stats.set_index("mouse")["mean"])
            print(
                f"{day_label}: z_560_Baseline mean fluorescence (2-8s post) = "
                f"{z560_summary.at[0, 'mean']:.6f} ± {z560_summary.at[0, 'sem']:.6f} "
                f"(SEM, n={int(z560_summary.at[0, 'n'])})"
            )
        else:
            print(f"{day_label}: ⚠️ No z_560_Baseline data between 2 and 8 seconds post alignment.")

        if not summary_df.empty:
            summary_df = summary_df.sort_values(["signal", "metric"]).drop_duplicates(
                subset=["signal", "metric"], keep="last"
            )
            summary_pivot = summary_df.pivot(index="metric", columns="signal", values=["mean", "sem"])

        signal_metrics_by_day[day_label] = metrics_df
        signal_metrics_summary_by_day[day_label] = summary_df
        signal_metrics_summary_pivot_by_day[day_label] = summary_pivot

        if metrics_df.empty:
            print(f"{day_label}: ⚠️ No signal metrics were computed. Check signal names and analysis window.")
            continue

        n_mice = metrics_df["mouse"].nunique()
        n_signals = metrics_df["signal"].nunique()
        print(f"{day_label}: ✅ Computed signal metrics for {n_mice} mice across {n_signals} signals.")

        output_dir = per_day_output_dirs.get(day_label, main_data_dir)
        experiment_day_label = day_label or extract_experiment_day(output_dir)

        if SAVE_SIGNAL_METRICS:
            per_mouse_path = save_signal_metrics_to_csv(
                metrics_df,
                summary_df,
                output_dir,
                file_prefix,
                cohort_identifier,
                experiment_day_label,
            )
            print(f"{day_label}: Saved per-mouse metrics to {per_mouse_path}")

        if not summary_pivot.empty:
            print(f"\n{day_label}: Signal metric summary (mean/sem):")
            display(summary_pivot)


# %%
# SANITY CHECK: VISUALISE ONSET DETECTION FOR ONE MOUSE
#---------------------------------------------------------------------------------------------------#
def plot_onset_detection_sanity_check(
    mouse_id: str,
    signal_column: str,
    window: Tuple[float, float],
    mouse_frames: Dict[str, pd.DataFrame],
    reference_label: str,
) -> None:
    if mouse_id not in mouse_frames:
        print(f"⚠️ Mouse '{mouse_id}' not found in the available frames: {sorted(mouse_frames.keys())}")
        return

    mouse_df = mouse_frames[mouse_id]
    if signal_column not in mouse_df.columns:
        print(f"⚠️ Signal '{signal_column}' not present for mouse '{mouse_id}'.")
        return

    df_signal = _prepare_mouse_signal_frame(mouse_df, signal_column)
    if df_signal.empty:
        print(f"⚠️ No data available for mouse '{mouse_id}' and signal '{signal_column}'.")
        return

    window_mask = (df_signal["time"] >= window[0]) & (df_signal["time"] <= window[1])
    window_df = df_signal.loc[window_mask]
    if window_df.empty:
        print(f"⚠️ Window {window} yielded no data points for mouse '{mouse_id}'.")
        return

    times = window_df["time"].to_numpy(dtype=float)
    values = window_df["value"].to_numpy(dtype=float)
    finite_mask = np.isfinite(times) & np.isfinite(values)
    times = times[finite_mask]
    values = values[finite_mask]
    if values.size == 0:
        print(f"⚠️ No finite values available for plotting (mouse '{mouse_id}').")
        return

    smooth_window = min(7, values.size)
    if smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        smoothed = np.convolve(values, kernel, mode="same")
    else:
        smoothed = values.copy()

    candidate = smoothed
    candidate_std = float(np.nanstd(candidate)) if candidate.size else 0.0
    prominence = max(0.05, 0.15 * candidate_std) if candidate_std > 0 else 0.05

    peaks, _ = find_peaks(candidate, prominence=prominence)
    if peaks.size == 0:
        peaks, _ = find_peaks(candidate)
    if peaks.size == 0:
        peak_index = int(np.nanargmax(candidate))
    else:
        best_idx = int(np.argmax(candidate[peaks]))
        peak_index = int(peaks[best_idx])
    peak_index = max(0, min(peak_index, values.size - 1))

    peak_value_raw = values[peak_index]
    peak_value_smooth = candidate[peak_index]
    peak_value = max(peak_value_raw, peak_value_smooth)

    logistic_onset = _estimate_logistic_onset(times[: peak_index + 1], candidate[: peak_index + 1])
    if logistic_onset is not None:
        onset_time = float(logistic_onset)
        onset_threshold = peak_value * 0.1
    else:
        onset_time = np.nan
        onset_threshold = peak_value * 0.1
        for idx in range(0, peak_index + 1):
            current_value = candidate[idx]
            if not np.isfinite(current_value):
                continue
            if current_value >= onset_threshold:
                if idx == 0:
                    onset_time = times[idx]
                else:
                    prev_time = times[idx - 1]
                    prev_value = candidate[idx - 1]
                    if not np.isfinite(prev_value) or prev_value == current_value:
                        onset_time = times[idx]
                    else:
                        fraction = (onset_threshold - prev_value) / (current_value - prev_value)
                        onset_time = prev_time + fraction * (times[idx] - prev_time)
                break

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, values, color="#1f77b4", alpha=0.35, label="Raw")
    ax.plot(times, candidate, color="#1f77b4", linewidth=2, label="Smoothed")
    ax.axhline(onset_threshold, color="orange", linestyle="--", label="10% peak threshold")
    ax.scatter(times[peak_index], candidate[peak_index], color="crimson", marker="*", s=120, label="Peak")
    if np.isfinite(onset_time):
        ax.axvline(onset_time, color="green", linestyle="--", label=f"Onset {onset_time:.3f}s")
    else:
        ax.text(0.05, 0.9, "Onset not found", transform=ax.transAxes, color="red", fontsize=12)

    ax.set_title(f"Onset sanity check | {reference_label}\nMouse: {mouse_id} | Signal: {signal_column}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Baseline z-score")
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right")
    plt.tight_layout()
    plt.show()


if 'mouse_signal_frames_by_day' in locals() and mouse_signal_frames_by_day:
    first_day_label = next(iter(mouse_signal_frames_by_day.keys()))
    frames_for_sanity = mouse_signal_frames_by_day[first_day_label]
    sanity_mouse = selected_mice[0] if selected_mice else next(iter(frames_for_sanity.keys()))
    plot_onset_detection_sanity_check(
        sanity_mouse,
        'z_560_Baseline',
        POST_ALIGNMENT_WINDOW,
        frames_for_sanity,
        reference_label=f"{cohort_identifier} | {first_day_label}",
    )
else:
    print("⚠️ No mouse signal frames available for onset sanity check.")


# %%
# DIAGNOSTIC: EXPONENTIAL FIT AND RESIDUAL VISUALIZATION
#---------------------------------------------------------------------------------------------------#
def plot_exponential_fit_and_residuals_per_mouse(
    mouse_id: str,
    mouse_df: pd.DataFrame,
    signal_column: str,
    metrics: Dict[str, float],
    cohort_id: str,
    day_label: str,
) -> go.Figure:
    """
    Create diagnostic plot showing calcium trace, exponential fit, and residuals.
    
    Parameters
    ----------
    mouse_id : str
        Mouse identifier
    mouse_df : pd.DataFrame
        Mouse data with signal column (indexed by time)
    signal_column : str
        Signal column name (e.g., 'z_560_Baseline')
    metrics : Dict[str, float]
        Computed metrics containing peak, decay_tau1, etc.
    cohort_id : str
        Cohort identifier for title
    day_label : str
        Experiment day label for title
    
    Returns
    -------
    go.Figure
        Plotly figure with calcium trace, fit, and residuals
    """
    # Extract time and signal
    if signal_column not in mouse_df.columns:
        print(f"⚠️ Signal '{signal_column}' not found for mouse {mouse_id}")
        return None
    
    times = pd.to_numeric(mouse_df.index, errors="coerce")
    values = pd.to_numeric(mouse_df[signal_column], errors="coerce")
    
    # Create clean dataframe
    df_clean = pd.DataFrame({"time": times, "value": values}).dropna()
    if df_clean.empty:
        print(f"⚠️ No valid data for mouse {mouse_id}")
        return None
    
    df_clean = df_clean.sort_values("time")
    times_clean = df_clean["time"].to_numpy()
    values_clean = df_clean["value"].to_numpy()
    
    # Get metrics
    peak_value = metrics.get("peak", np.nan)
    decay_tau1 = metrics.get("decay_tau1", np.nan)
    onset_time = metrics.get("onset_time", np.nan)
    main_peak_residual_auc = metrics.get("main_peak_residual_auc", np.nan)
    offset_residual_auc = metrics.get("offset_residual_auc", np.nan)
    
    # Find peak index
    peak_mask = (times_clean >= POST_ALIGNMENT_WINDOW[0]) & (times_clean <= POST_ALIGNMENT_WINDOW[1])
    if not peak_mask.any():
        print(f"⚠️ No data in analysis window for mouse {mouse_id}")
        return None
    
    window_values = values_clean[peak_mask]
    window_times = times_clean[peak_mask]
    peak_idx_window = np.argmax(window_values)
    peak_time = window_times[peak_idx_window]
    
    # Estimate baseline and fit double-exponential
    baseline = _estimate_baseline(times_clean, values_clean)
    
    # Find peak index in full data
    peak_idx_full = np.argmin(np.abs(times_clean - peak_time))
    
    # Fit double-exponential decay
    tau1, A1, tau2, A2 = _estimate_double_exponential_decay(
        times_clean, values_clean, peak_idx_full, baseline, DECAY_FIT_START_OFFSET
    )
    
    # Reconstruct double-exponential decay (from fit start time onward)
    exponential_fit = np.full_like(times_clean, np.nan)
    residuals = np.full_like(times_clean, np.nan)
    
    if np.isfinite(tau1) and tau1 > 0 and np.isfinite(A1):
        # Only compute exponential from fit start time onward (peak + offset)
        fit_start_time = peak_time + DECAY_FIT_START_OFFSET
        post_fit_mask = times_clean >= fit_start_time
        time_from_fit_start = times_clean[post_fit_mask] - fit_start_time
        exponential_fit[post_fit_mask] = (
            baseline + 
            A1 * np.exp(-time_from_fit_start / tau1) + 
            A2 * np.exp(-time_from_fit_start / tau2)
        )
        residuals[post_fit_mask] = values_clean[post_fit_mask] - exponential_fit[post_fit_mask]
    else:
        print(f"⚠️ Invalid tau1 ({tau1}) or A1 ({A1}) for mouse {mouse_id}")
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f"Calcium Trace & Double-Exponential Fit",
            f"Residuals (Observed - Expected)"
        ),
        row_heights=[0.6, 0.4]
    )
    
    # Top panel: Calcium trace and exponential fit
    fig.add_trace(
        go.Scatter(
            x=times_clean,
            y=values_clean,
            mode='lines',
            name='Calcium trace',
            line=dict(color='black', width=2),
            showlegend=True
        ),
        row=1, col=1
    )
    
    if np.any(np.isfinite(exponential_fit)):
        fig.add_trace(
            go.Scatter(
                x=times_clean,
                y=exponential_fit,
                mode='lines',
                name='Double-exponential fit',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Mark peak
    if np.isfinite(peak_value):
        fig.add_trace(
            go.Scatter(
                x=[peak_time],
                y=[peak_value],
                mode='markers',
                name='Peak',
                marker=dict(color='crimson', size=12, symbol='star'),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Mark onset
    if np.isfinite(onset_time):
        fig.add_vline(
            x=onset_time,
            line=dict(color='green', width=2, dash='dot'),
            annotation_text=f"Onset: {onset_time:.2f}s",
            annotation_position="top",
            row=1, col=1
        )
    
    # Mark fit start time (if offset from peak)
    if DECAY_FIT_START_OFFSET != 0.0:
        fit_start_time = peak_time + DECAY_FIT_START_OFFSET
        fig.add_vline(
            x=fit_start_time,
            line=dict(color='purple', width=2, dash='dashdot'),
            annotation_text=f"Fit start: {fit_start_time:.2f}s",
            annotation_position="bottom",
            row=1, col=1
        )
    
    # Highlight main peak window (used for fitting & main_peak_residual_auc)
    fig.add_vrect(
        x0=POST_ALIGNMENT_WINDOW[0],
        x1=POST_ALIGNMENT_WINDOW[1],
        fillcolor="lightgreen",
        opacity=0.15,
        layer="below",
        line_width=0,
        annotation_text="Main peak fit window",
        annotation_position="top right",
        row=1, col=1
    )
    
    # Highlight offset analysis window
    fig.add_vrect(
        x0=OFFSET_ANALYSIS_START,
        x1=OFFSET_ANALYSIS_END,
        fillcolor="lightblue",
        opacity=0.2,
        layer="below",
        line_width=0,
        annotation_text="Offset window",
        annotation_position="top left",
        row=1, col=1
    )
    
    # Bottom panel: Residuals
    if np.any(np.isfinite(residuals)):
        fig.add_trace(
            go.Scatter(
                x=times_clean,
                y=residuals,
                mode='lines',
                name='Residuals',
                line=dict(color='blue', width=2),
                showlegend=True
            ),
            row=2, col=1
        )
        
        # Highlight positive residuals in main peak window (fit window)
        main_peak_mask = (times_clean >= peak_time) & (times_clean <= POST_ALIGNMENT_WINDOW[1])
        if main_peak_mask.any():
            main_peak_times_plot = times_clean[main_peak_mask]
            main_peak_residuals_plot = residuals[main_peak_mask]
            main_positive_mask = main_peak_residuals_plot > 0
            
            if main_positive_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=main_peak_times_plot[main_positive_mask],
                        y=main_peak_residuals_plot[main_positive_mask],
                        mode='lines',
                        name='Main peak residual AUC',
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 0, 0.2)',
                        line=dict(color='green', width=3),
                        showlegend=True
                    ),
                    row=2, col=1
                )
        
        # Highlight positive residuals in offset AUC window (only where AUC is calculated)
        offset_auc_mask = (times_clean >= OFFSET_AUC_WINDOW[0]) & (times_clean <= OFFSET_AUC_WINDOW[1])
        if offset_auc_mask.any():
            offset_auc_times = times_clean[offset_auc_mask]
            offset_auc_residuals = residuals[offset_auc_mask]
            positive_mask = offset_auc_residuals > 0
            
            if positive_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=offset_auc_times[positive_mask],
                        y=offset_auc_residuals[positive_mask],
                        mode='lines',
                        name='Offset residual AUC',
                        fill='tozeroy',
                        fillcolor='rgba(255, 165, 0, 0.3)',
                        line=dict(color='orange', width=3),
                        showlegend=True
                    ),
                    row=2, col=1
                )
                
                # Add marker showing the AUC value at midpoint of window
                if np.isfinite(offset_residual_auc):
                    midpoint_time = (OFFSET_AUC_WINDOW[0] + OFFSET_AUC_WINDOW[1]) / 2
                    # Find the max positive residual in the window for marker placement
                    max_positive_residual = float(np.max(offset_auc_residuals[positive_mask]))
                    fig.add_trace(
                        go.Scatter(
                            x=[midpoint_time],
                            y=[max_positive_residual],
                            mode='markers+text',
                            name=f'AUC={offset_residual_auc:.3f}',
                            marker=dict(color='darkorange', size=14, symbol='star'),
                            text=[f'AUC={offset_residual_auc:.3f}'],
                            textposition='top center',
                            textfont=dict(size=12, color='darkorange'),
                            showlegend=True
                        ),
                        row=2, col=1
                    )
        
        # Zero line
        fig.add_hline(
            y=0,
            line=dict(color='gray', width=1, dash='dash'),
            row=2, col=1
        )
        
        # Highlight offset window
        fig.add_vrect(
            x0=OFFSET_ANALYSIS_START,
            x1=OFFSET_ANALYSIS_END,
            fillcolor="lightblue",
            opacity=0.2,
            layer="below",
            line_width=0,
            row=2, col=1
        )
        
        # Highlight offset AUC integration window
        fig.add_vrect(
            x0=OFFSET_AUC_WINDOW[0],
            x1=OFFSET_AUC_WINDOW[1],
            fillcolor="yellow",
            opacity=0.15,
            layer="below",
            line_width=0,
            annotation_text="Offset AUC window",
            annotation_position="bottom left",
            row=2, col=1
        )
    
    # Update layout
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="z-score", row=1, col=1)
    fig.update_yaxes(title_text="Residual z-score", row=2, col=1)
    
    # Add title with metrics
    fit_offset_text = f" | Fit offset: +{DECAY_FIT_START_OFFSET:.2f}s" if DECAY_FIT_START_OFFSET != 0.0 else ""
    metrics_text = (
        f"Mouse: {mouse_id} | {cohort_id} | {day_label}<br>"
        f"<sub>Peak: {peak_value:.3f} | Baseline: {baseline:.3f}{fit_offset_text}<br>"
        f"Fast: τ1={tau1:.3f}s, A1={A1:.3f} | Slow: τ2={tau2:.3f}s, A2={A2:.3f}<br>"
        f"Main peak residual AUC: {main_peak_residual_auc:.3f} | "
        f"Offset AUC: {offset_residual_auc:.3f}</sub>"
    )
    
    fig.update_layout(
        title_text=metrics_text,
        height=800,
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top'),
        hovermode='x unified'
    )
    
    return fig


def generate_diagnostic_plots_for_all_mice(
    mouse_signal_frames: Dict[str, pd.DataFrame],
    signal_metrics: pd.DataFrame,
    signal_column: str,
    cohort_id: str,
    day_label: str,
    show_plots: bool = True,
    save_plots: bool = False,
    output_dir: Optional[Path] = None,
):
    """
    Generate diagnostic plots for all mice showing exponential fits and residuals.
    
    Parameters
    ----------
    mouse_signal_frames : Dict[str, pd.DataFrame]
        Dictionary mapping mouse IDs to their signal dataframes
    signal_metrics : pd.DataFrame
        Computed metrics for each mouse/signal
    signal_column : str
        Signal column to analyze
    cohort_id : str
        Cohort identifier
    day_label : str
        Experiment day label
    show_plots : bool
        Whether to display plots (default: True)
    save_plots : bool
        Whether to save plots as HTML (default: False)
    output_dir : Optional[Path]
        Directory to save plots (required if save_plots=True)
    """
    if not mouse_signal_frames:
        print("⚠️ No mouse signal frames available for diagnostic plotting.")
        return
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC PLOTS: Exponential Fit & Residuals")
    print(f"{'='*60}")
    print(f"Signal: {signal_column}")
    print(f"Generating plots for {len(mouse_signal_frames)} mice...")
    print()
    
    if save_plots and output_dir is None:
        print("⚠️ save_plots=True but no output_dir provided. Plots will not be saved.")
        save_plots = False
    
    if save_plots:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for mouse_id in sorted(mouse_signal_frames.keys()):
        mouse_df = mouse_signal_frames[mouse_id]
        
        # Get metrics for this mouse
        mouse_metrics_rows = signal_metrics[
            (signal_metrics["mouse"] == mouse_id) & 
            (signal_metrics["signal"] == signal_column)
        ]
        
        if mouse_metrics_rows.empty:
            print(f"⚠️ No metrics found for mouse {mouse_id}, skipping...")
            continue
        
        # Convert first row to dict
        metrics_dict = mouse_metrics_rows.iloc[0].to_dict()
        
        # Generate plot
        fig = plot_exponential_fit_and_residuals_per_mouse(
            mouse_id,
            mouse_df,
            signal_column,
            metrics_dict,
            cohort_id,
            day_label,
        )
        
        if fig is None:
            continue
        
        # Display plot
        if show_plots:
            fig.show()
        
        # Save plot
        if save_plots:
            filename = f"{sanitize_for_filename(cohort_id)}_{sanitize_for_filename(day_label)}_{mouse_id}_{signal_column}_fit_diagnostic.html"
            output_path = output_dir / filename
            fig.write_html(str(output_path))
            print(f"✅ Saved: {output_path.name}")
    
    print(f"\n✅ Diagnostic plotting complete!")


# Run diagnostic plots if data available
if 'mouse_signal_frames_by_day' in locals() and mouse_signal_frames_by_day:
    if 'signal_metrics_by_day' in locals() and signal_metrics_by_day:
        # Plot for ALL available days
        print(f"\n{'='*60}")
        print(f"GENERATING DIAGNOSTIC PLOTS FOR ALL DAYS")
        print(f"{'='*60}")
        
        for day_label in mouse_signal_frames_by_day.keys():
            print(f"\nProcessing day: {day_label}")
            mouse_frames = mouse_signal_frames_by_day[day_label]
            metrics_df = signal_metrics_by_day.get(day_label)
            
            if metrics_df is not None and not metrics_df.empty:
                # Determine output directory for this day
                day_output_dir = per_day_output_dirs.get(day_label, main_data_dir) / "diagnostic_plots"
                
                generate_diagnostic_plots_for_all_mice(
                    mouse_frames,
                    metrics_df,
                    'z_560_Baseline',
                    cohort_identifier,
                    day_label,
                    show_plots=True,
                    save_plots=True,
                    output_dir=day_output_dir,
                )
                print(f"✅ Completed diagnostic plots for {day_label}")
            else:
                print(f"⚠️ No signal metrics available for {day_label}")
        
        print(f"\n{'='*60}")
        print(f"ALL DIAGNOSTIC PLOTS COMPLETE")
        print(f"{'='*60}")
    else:
        print("⚠️ signal_metrics_by_day not available for diagnostic plotting.")
else:
    print("⚠️ No mouse signal frames available for diagnostic plotting.")


# %%
# FINAL CONDITION METRIC PLOT (WITH CONNECTING LINES)
#---------------------------------------------------------------------------------------------------#
def plot_condition_metric(
    metric_name: str,
    metric_df: pd.DataFrame,
    condition_order: List[str],
    signal_order: List[str],
    mouse_colors: Dict[str, tuple],
    connect_lines: bool = True,
    no_line_conditions: Optional[List[str]] = None,
) -> plt.Figure:
    """Generate scatter plots for a given metric across conditions.

    When connect_lines is True, mice present in all conditions are shown with
    connecting lines; otherwise individual points are plotted without lines.
    
    Parameters
    ----------
    no_line_conditions : Optional[List[str]]
        List of conditions that should always be plotted as individual points
        without connecting lines (e.g., combined conditions like "day3+day4")
    """
    styled_mouse_colors = OrderedDict((str(mouse), color) for mouse, color in mouse_colors.items())
    if no_line_conditions is None:
        no_line_conditions = []
    
    style_context = {
        "font.size": 15,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "legend.fontsize": 10,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    }

    with plt.rc_context(style_context):
        signals_present = [sig for sig in signal_order if sig in metric_df["signal"].unique()]
        width_cm = 5.0 * max(1, len(signals_present))
        height_cm = 7.0
        fig_width_in, fig_height_in = cm_to_inches(width_cm, height_cm)

        if not signals_present:
            fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))
            ax.text(
                0.5,
                0.5,
                f"No data available for metric {metric_name}",
                ha="center",
                va="center",
            )
            ax.axis("off")
            return fig

        fig, axes = plt.subplots(
            1,
            len(signals_present),
            figsize=(fig_width_in, fig_height_in),
            sharey=True,
        )
        if len(signals_present) == 1:
            axes = [axes]

        positions = np.arange(len(condition_order), dtype=float)
        mean_legend_handle = None
        mouse_handles: OrderedDict[str, object] = OrderedDict()
        
        # Separate conditions into paired (with lines) and unpaired (no lines)
        paired_conditions = [c for c in condition_order if c not in no_line_conditions]
        unpaired_conditions = [c for c in condition_order if c in no_line_conditions]

        for ax, signal in zip(axes, signals_present):
            subset = metric_df[metric_df["signal"] == signal].copy()
            if subset.empty:
                ax.axis("off")
                continue

            subset["condition"] = pd.Categorical(
                subset["condition"], categories=condition_order, ordered=True
            )
            subset = subset.sort_values(["mouse", "condition"])

            # Plot paired conditions (with connecting lines if connect_lines=True)
            if connect_lines and paired_conditions:
                for mouse, color in styled_mouse_colors.items():
                    mouse_subset = subset[subset["mouse"] == mouse]
                    if mouse_subset.empty:
                        continue
                    # Filter to only paired conditions to avoid duplicate index issues
                    mouse_subset_paired = mouse_subset[mouse_subset["condition"].isin(paired_conditions)]
                    if mouse_subset_paired.empty:
                        continue
                    mouse_series = mouse_subset_paired.set_index("condition")["value"]
                    # Only plot paired conditions with lines
                    paired_series = mouse_series.reindex(paired_conditions)
                    if paired_series.isna().any():
                        continue
                    paired_values = paired_series.to_numpy(dtype=float)
                    paired_positions = np.array([condition_order.index(c) for c in paired_conditions])
                    line, = ax.plot(
                        paired_positions,
                        paired_values,
                        marker="o",
                        linestyle="-",
                        color=color,
                        linewidth=1.4,
                        markersize=5,
                        alpha=0.7,
                        zorder=2,
                    )
                    if mouse not in mouse_handles:
                        mouse_handles[mouse] = line
            elif not connect_lines:
                # Original behavior when connect_lines=False (unpaired design)
                for mouse, color in styled_mouse_colors.items():
                    mouse_subset = subset[subset["mouse"] == mouse]
                    if mouse_subset.empty:
                        continue
                    
                    jitter = (hash((mouse, signal)) % 5 - 2) * 0.04
                    
                    # Iterate through rows directly to handle any duplicate condition entries
                    for _, row in mouse_subset.iterrows():
                        cond = row["condition"]
                        value = row["value"]
                        
                        if cond not in condition_order:
                            continue
                        if pd.isna(value):
                            continue
                        
                        cond_idx = condition_order.index(cond)
                        point = ax.plot(
                            positions[cond_idx] + jitter,
                            float(value),
                            marker="o",
                            linestyle="None",
                            color=color,
                            markersize=6,
                            alpha=0.8,
                            zorder=2,
                        )[0]
                        if mouse not in mouse_handles:
                            mouse_handles[mouse] = point
            
            # Plot unpaired conditions (no lines, just individual points)
            # These conditions may have duplicate mouse entries (e.g., combined day3+day4)
            if unpaired_conditions:
                for mouse, color in styled_mouse_colors.items():
                    mouse_subset = subset[subset["mouse"] == mouse]
                    if mouse_subset.empty:
                        continue
                    # Filter to only unpaired conditions
                    mouse_subset_unpaired = mouse_subset[mouse_subset["condition"].isin(unpaired_conditions)]
                    if mouse_subset_unpaired.empty:
                        continue
                    
                    jitter = (hash((mouse, signal)) % 5 - 2) * 0.04
                    
                    # Iterate through rows directly to handle duplicate condition entries
                    for _, row in mouse_subset_unpaired.iterrows():
                        cond = row["condition"]
                        value = row["value"]
                        
                        if cond not in condition_order:
                            continue
                        if pd.isna(value):
                            continue
                        
                        cond_idx = condition_order.index(cond)
                        ax.plot(
                            positions[cond_idx] + jitter,
                            float(value),
                            marker="o",
                            linestyle="None",
                            color=color,
                            markersize=6,
                            alpha=0.8,
                            zorder=2,
                        )

            means = subset.groupby("condition")["value"].mean().reindex(condition_order)
            sems = subset.groupby("condition")["value"].apply(_paired_sem).reindex(condition_order)
            mean_values = means.to_numpy(dtype=float)
            sem_values = sems.to_numpy(dtype=float)
            sem_values = np.where(np.isfinite(sem_values), sem_values, 0.0)
            valid_mask = np.isfinite(mean_values)
            positions_valid = positions[valid_mask]
            mean_values_valid = mean_values[valid_mask]
            sem_values_valid = sem_values[valid_mask]
            ax.fill_between(
                positions_valid,
                mean_values_valid - sem_values_valid,
                mean_values_valid + sem_values_valid,
                color="0.7",
                alpha=0.25,
                zorder=3,
            )
            mean_line, = ax.plot(
                positions_valid,
                mean_values_valid,
                color="black",
                linewidth=2.0,
                marker="o",
                markersize=6,
                zorder=4,
            )
            if mean_legend_handle is None:
                mean_legend_handle = mean_line

            condition_labels = [
                subset[subset["condition"] == cond]["condition_label"].iloc[0]
                if not subset[subset["condition"] == cond].empty
                else cond
                for cond in condition_order
            ]

            ax.set_xticks(positions)
            ax.set_xticklabels(condition_labels, rotation=15, ha="right")
            ax.set_xlabel("Condition", fontfamily="Arial")
            clean_title = signal.replace("_Baseline", "").replace("_", " ")
            ax.set_title(clean_title, fontfamily="Arial")
            ax.tick_params(axis="both", labelsize=15)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontfamily("Arial")
            ax.grid(False)

        axes[0].set_ylabel(metric_name.replace("_", " ").title(), fontfamily="Arial")

        legend_handles: List[object] = []
        legend_labels: List[str] = []
        if mouse_handles:
            legend_handles.extend(mouse_handles.values())
            legend_labels.extend(mouse_handles.keys())
        if mean_legend_handle is not None:
            legend_handles.append(mean_legend_handle)
            legend_labels.append("Mean ± SEM")

        if legend_handles:
            legend_columns = 1 if len(legend_labels) <= 10 else 2
            legend = axes[0].legend(
                legend_handles,
                legend_labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
                borderaxespad=0.0,
                prop={"family": "Arial", "size": 10},
                ncol=legend_columns,
            )
            legend._legend_box.sep = 4

        fig.tight_layout(rect=(0, 0, 0.78, 1))
        return fig



# %%
# CONDITION COMPARISON (DAY 3 / DAY 4 / NO HALT)
#---------------------------------------------------------------------------------------------------#
if GENERATE_CONDITION_COMPARISON:
    print(f"\n{'='*60}")
    print("CONDITION COMPARISON")
    print(f"{'='*60}")

    TARGET_SIGNAL = "z_560_Baseline"
    TARGET_METRICS = [
        "peak",
        "onset_time",
        "decay_tau1",
        "mean_fluorescence_2_to_8s",
        "main_peak_residual_auc",
        "offset_residual_auc",
    ]
    CONDITION_ORDER = ["Apply_halt_day3", "Apply_halt_day4", "No_halt"]
    
    # Metrics that should EXCLUDE "No halt" condition from plots and statistics
    METRICS_EXCLUDE_NO_HALT = [
        "offset_residual_auc",
        "decay_tau1",
        "onset_time",
    ]
    
    print(f"🎯 TARGET_METRICS to process: {TARGET_METRICS}")
    print(f"🚫 METRICS_EXCLUDE_NO_HALT: {METRICS_EXCLUDE_NO_HALT}")

    condition_metrics_frames: Dict[str, pd.DataFrame] = {}
    condition_sources_map: Dict[str, Dict[str, Path]] = {}

    for condition_key, cfg in COMPARE_CALCIUM_METRICS_ACROSS_CONDITIONS.items():
        metrics_df_condition, sources_condition = compute_condition_signal_metrics(
            condition_key,
            cfg,
            selected_mice,
            available_mice,
        )
        if metrics_df_condition.empty:
            print(f"⚠️ Skipping condition '{condition_key}' (no metrics computed).")
            continue
        if TARGET_SIGNAL not in metrics_df_condition["signal"].unique():
            print(
                f"⚠️ Skipping condition '{condition_key}' (no '{TARGET_SIGNAL}' signal present)."
            )
            continue
        condition_metrics_frames[condition_key] = metrics_df_condition
        condition_sources_map[condition_key] = sources_condition
        print(
            f"✅ Condition '{condition_key}' loaded: "
            f"{metrics_df_condition['mouse'].nunique()} mice, "
            f"{metrics_df_condition['signal'].nunique()} signals."
        )

    if len(condition_metrics_frames) < 2:
        print("⚠️ At least two conditions with valid data are required for comparison. Skipping analysis.")
    else:
        combined_metrics_wide = pd.concat(
            condition_metrics_frames.values(), ignore_index=True
        )

        long_df = combined_metrics_wide.melt(
            id_vars=[
                "mouse",
                "signal",
                "condition",
                "condition_label",
                "event_name",
            ],
            value_vars=TARGET_METRICS,
            var_name="metric",
            value_name="value",
        )
        long_df = (
            long_df.groupby(
                ["mouse", "signal", "condition", "condition_label", "event_name", "metric"],
                as_index=False,
            )["value"]
            .mean()
        )
        long_df = long_df.dropna(subset=["value"])
        long_df = long_df[long_df["signal"] == TARGET_SIGNAL].copy()
        
        print(f"\n📊 After melting and filtering for {TARGET_SIGNAL}:")
        print(f"   Total rows: {len(long_df)}")
        print(f"   Available metrics: {sorted(long_df['metric'].unique())}")
        print(f"   Available conditions: {sorted(long_df['condition'].unique())}")
        print(f"\n   Metric value counts:")
        for metric in sorted(long_df['metric'].unique()):
            metric_rows = len(long_df[long_df['metric'] == metric])
            print(f"      {metric}: {metric_rows} rows")

        if long_df.empty:
            print("⚠️ No z_560_Baseline data available for comparison; skipping analysis.")
        else:
            preferred_order = CONDITION_ORDER
            available_conditions = [
                cond for cond in preferred_order if cond in long_df["condition"].unique()
            ]

            if len(available_conditions) < 2:
                print("⚠️ Not enough conditions with data for comparison; skipping analysis.")
            else:
                signal_order = [TARGET_SIGNAL]

                all_mice_for_colors = sorted(
                    {
                        str(mouse)
                        for df in condition_metrics_frames.values()
                        for mouse in df["mouse"].astype(str).unique()
                    }
                )
                global_mouse_colors = assign_mouse_colors_consistent(all_mice_for_colors)

                resolved_main_dir = Path(main_data_dir).resolve()
                cohort_dir = resolved_main_dir.parent if resolved_main_dir.parent != resolved_main_dir else resolved_main_dir

                comparison_output_dir = cohort_dir / "calcium_analysis"
                comparison_output_dir.mkdir(parents=True, exist_ok=True)

                outputs_written = False

                summary_rows: List[Dict[str, object]] = []
                stats_records: List[pd.DataFrame] = []

                for metric_name in TARGET_METRICS:
                    print(f"\n{'='*50}")
                    print(f"🔍 PROCESSING METRIC: {metric_name}")
                    print(f"{'='*50}")
                    
                    # Determine which conditions to include for this metric
                    if metric_name in METRICS_EXCLUDE_NO_HALT:
                        # Exclude "No_halt" condition for specific metrics
                        condition_subset = [
                            cond for cond in preferred_order 
                            if cond in long_df["condition"].unique() and cond != "No_halt"
                        ]
                        print(f"📊 Metric '{metric_name}': excluding 'No_halt' condition from analysis")
                    else:
                        condition_subset = [
                            cond for cond in preferred_order if cond in long_df["condition"].unique()
                        ]
                    
                    print(f"   Available conditions in long_df: {sorted(long_df['condition'].unique())}")
                    print(f"   Condition subset for analysis: {condition_subset}")
                    
                    if len(condition_subset) < 2:
                        print(
                            f"⚠️ Metric '{metric_name}' does not have enough conditions for comparison; skipping."
                        )
                        continue

                    metric_long_full = long_df[
                        (long_df["metric"] == metric_name)
                        & (long_df["condition"].isin(condition_subset))
                    ].copy()
                    
                    print(f"   Rows in metric_long_full before processing: {len(metric_long_full)}")
                    print(f"   Unique mice: {sorted(metric_long_full['mouse'].unique()) if not metric_long_full.empty else 'None'}")
                    
                    # For offset_residual_auc only: create combined day3+day4 condition
                    add_combined_column = False
                    if metric_name == "offset_residual_auc":
                        print(f"   🎯 This is offset_residual_auc - checking for combined column...")
                        print(f"   'Apply_halt_day3' in subset: {'Apply_halt_day3' in condition_subset}")
                        print(f"   'Apply_halt_day4' in subset: {'Apply_halt_day4' in condition_subset}")
                        
                        if "Apply_halt_day3" in condition_subset and "Apply_halt_day4" in condition_subset:
                            # Create combined condition with all day3 and day4 data
                            combined_data = metric_long_full[
                                metric_long_full["condition"].isin(["Apply_halt_day3", "Apply_halt_day4"])
                            ].copy()
                            print(f"   ✅ Creating combined column with {len(combined_data)} data points")
                            combined_data["condition"] = "Apply_halt_day3+day4"
                            combined_data["condition_label"] = "Apply halt day 3+4"
                            
                            # Append combined data to metric_long_full
                            metric_long_full = pd.concat([metric_long_full, combined_data], ignore_index=True)
                            
                            # Add combined condition to subset (at the end)
                            condition_subset = condition_subset + ["Apply_halt_day3+day4"]
                            add_combined_column = True
                            print(f"   📊 Added combined 'day3+day4' column")
                            print(f"   Updated condition_subset: {condition_subset}")
                        else:
                            print(f"   ⚠️ Cannot create combined column - missing day3 or day4")
                    
                    if metric_long_full.empty:
                        print(f"⚠️ No data for metric '{metric_name}', skipping.")
                        continue
                    metric_long_full["mouse"] = metric_long_full["mouse"].astype(str)
                    
                    print(f"   Final rows in metric_long_full: {len(metric_long_full)}")

                    mice_by_condition = {
                        cond: set(metric_long_full.loc[metric_long_full["condition"] == cond, "mouse"])
                        for cond in condition_subset
                    }
                    common_mice = set.intersection(*(mice_by_condition[cond] for cond in condition_subset)) if mice_by_condition else set()
                    common_mice_list = sorted(common_mice)
                    counts_equal = len({len(mice_by_condition[cond]) for cond in condition_subset if mice_by_condition[cond]}) == 1
                    has_complete_overlap = (
                        bool(common_mice_list)
                        and counts_equal
                        and all(len(common_mice_list) == len(mice_by_condition[cond]) for cond in condition_subset)
                    )

                    if has_complete_overlap:
                        analysis_mode = "paired"
                        metric_long = metric_long_full[metric_long_full["mouse"].isin(common_mice_list)].copy()
                    else:
                        analysis_mode = "unpaired"
                        metric_long = metric_long_full.copy()
                    paired_mice_count = len(common_mice_list)

                    if metric_long.empty:
                        print(f"⚠️ After alignment, no usable data for metric '{metric_name}'. Skipping.")
                        continue

                    condition_label_map = (
                        metric_long.drop_duplicates(subset=["condition", "condition_label"])
                        .set_index("condition")["condition_label"]
                        .to_dict()
                    )
                    metric_long["condition_label"] = metric_long["condition"].map(condition_label_map)
                    metric_long["condition"] = pd.Categorical(
                        metric_long["condition"], categories=condition_subset, ordered=True
                    )

                    mouse_colors = OrderedDict(
                        (mouse, global_mouse_colors[mouse])
                        for mouse in global_mouse_colors
                        if mouse in metric_long["mouse"].unique()
                    )

                    if not mouse_colors:
                        print(
                            f"⚠️ No color assignments available for metric '{metric_name}'. Skipping."
                        )
                        continue

                    print(
                        f"Analyzing metric '{metric_name}' using {analysis_mode} design across conditions: {condition_subset}"
                    )

                    mode_notes = []
                    if analysis_mode == "paired":
                        mode_notes.append(
                            "Paired design: restricted to mice present in every condition."
                        )
                    else:
                        mode_notes.append(
                            "Unpaired design: mouse rosters differ across conditions."
                        )

                    # For statistics, exclude combined conditions (which have duplicate mice)
                    stats_condition_subset = [c for c in condition_subset if "+" not in c]
                    stats_metric_long = metric_long[metric_long["condition"].isin(stats_condition_subset)].copy()
                    
                    if add_combined_column:
                        print(f"   📊 Statistics will use conditions: {stats_condition_subset}")
                        print(f"   📊 (excluding combined column from stats to avoid duplicate mice)")
                        mode_notes.append(
                            f"Combined condition '{[c for c in condition_subset if '+' in c]}' "
                            "included in plot and descriptive statistics but excluded from "
                            "inferential tests (ANOVA/pairwise) to avoid duplicate mouse entries."
                        )
                    
                    summary_df, pairwise_df, analysis_notes = compute_repeated_measures_stats(
                        stats_metric_long,
                        stats_condition_subset,
                        signal_order,
                        metric_name,
                    )
                    analysis_notes = mode_notes + analysis_notes

                    # Determine which conditions should not have connecting lines
                    no_line_conditions_for_plot = []
                    if add_combined_column:
                        no_line_conditions_for_plot = ["Apply_halt_day3+day4"]
                        print(f"   📊 Will plot with no_line_conditions: {no_line_conditions_for_plot}")
                    
                    print(f"   🎨 Creating plot for {metric_name}...")
                    fig = plot_condition_metric(
                        metric_name,
                        metric_long,
                        condition_subset,
                        signal_order,
                        mouse_colors,
                        connect_lines=(analysis_mode == "paired"),
                        no_line_conditions=no_line_conditions_for_plot,
                    )

                    cohort_part = sanitize_for_filename(cohort_identifier, "cohort")
                    day_part = sanitize_for_filename(experiment_day, "day")
                    metric_part = sanitize_for_filename(metric_name, metric_name)

                    plot_filename = (
                        comparison_output_dir
                        / f"{cohort_part}_{day_part}_{metric_part}_{analysis_mode}_condition_comparison.pdf"
                    )
                    print(f"   💾 Saving plot to: {plot_filename}")
                    fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    print(f"   ✅ Plot saved successfully")

                    plot_data_path = (
                        comparison_output_dir
                        / f"{cohort_part}_{day_part}_{metric_part}_{analysis_mode}_condition_comparison_plot_data.csv"
                    )
                    metric_long.sort_values(
                        ["signal", "mouse", "condition"]
                    ).to_csv(plot_data_path, index=False)

                    stats_frames = []
                    if not summary_df.empty:
                        stats_frames.append(summary_df)
                    if not pairwise_df.empty:
                        stats_frames.append(pairwise_df)
                    if analysis_notes:
                        stats_frames.append(
                            pd.DataFrame(
                                {
                                    "analysis": ["note"] * len(analysis_notes),
                                    "metric": [metric_name] * len(analysis_notes),
                                    "note": analysis_notes,
                                }
                            )
                        )

                    if stats_frames:
                        stats_output = pd.concat(stats_frames, ignore_index=True)
                    else:
                        stats_output = pd.DataFrame(
                            {
                                "analysis": ["note"],
                                "metric": [metric_name],
                                "note": ["No statistics computed."],
                            }
                        )

                    stats_output["cohort"] = cohort_identifier
                    stats_output["experiment_day"] = experiment_day
                    stats_output["analysis_mode"] = analysis_mode
                    stats_output["paired_mice_n"] = paired_mice_count
                    stats_output["unique_mice_total"] = metric_long["mouse"].nunique()
                    stats_output_path = (
                        comparison_output_dir
                        / f"{cohort_part}_{day_part}_{metric_part}_{analysis_mode}_condition_comparison_stats.csv"
                    )
                    stats_output.to_csv(stats_output_path, index=False)

                    stats_records.append(
                        stats_output.assign(
                            plot_filename=plot_filename.name,
                            plot_data_filename=plot_data_path.name,
                            stats_filename=stats_output_path.name,
                        )
                    )

                    print(
                        f"📄 Metric '{metric_name}': saved plot ({plot_filename.name}), "
                        f"data ({plot_data_path.name}), stats ({stats_output_path.name})"
                    )

                    for cond in condition_subset:
                        cond_values = metric_long.loc[
                            metric_long["condition"] == cond, "value"
                        ]
                        n_obs = int(cond_values.count())
                        mean_val = float(cond_values.mean()) if n_obs else np.nan
                        if n_obs > 1:
                            sem_val = float(cond_values.std(ddof=1) / np.sqrt(n_obs))
                        else:
                            sem_val = 0.0 if n_obs == 1 else np.nan
                        summary_rows.append(
                            {
                                "metric": metric_name,
                                "condition": cond,
                                "condition_label": condition_label_map.get(cond, cond),
                                "mean": mean_val,
                                "sem": sem_val,
                                "n": n_obs,
                            }
                        )

                    outputs_written = True

                if outputs_written:
                    summary_table = pd.DataFrame(summary_rows)
                    if not summary_table.empty:
                        resolved_main_dir = Path(main_data_dir).resolve()
                        cohort_dir = (
                            resolved_main_dir.parent
                            if resolved_main_dir.parent != resolved_main_dir
                            else resolved_main_dir
                        )
                        comparison_output_dir = cohort_dir / "calcium_analysis"
                        comparison_output_dir.mkdir(parents=True, exist_ok=True)
                        comparison_summary_path = (
                            comparison_output_dir
                            / f"{sanitize_for_filename(cohort_identifier)}_{sanitize_for_filename(experiment_day)}_z560_metric_summary.csv"
                        )
                        summary_table.sort_values(["metric", "condition"]).to_csv(
                            comparison_summary_path, index=False
                        )
                        print(f"✅ Summary table saved to {comparison_summary_path}")
                    if stats_records:
                        combined_stats = pd.concat(stats_records, ignore_index=True)
                        combined_stats_path = (
                            comparison_output_dir
                            / f"{sanitize_for_filename(cohort_identifier)}_{sanitize_for_filename(experiment_day)}_z560_condition_stats_all.csv"
                        )
                        combined_stats.to_csv(combined_stats_path, index=False)
                        print(f"✅ Combined stats saved to {combined_stats_path}")
                    print(
                        f"✅ Calcium analysis outputs saved to: {comparison_output_dir}"
                    )
                else:
                    print("⚠️ No condition comparison outputs were generated.")
else:
    print("⏭️  Condition comparison disabled (GENERATE_CONDITION_COMPARISON=False)")


# %%
# GRAND AVERAGE VS PER-MOUSE PEAK OVERLAY
#---------------------------------------------------------------------------------------------------#
def plot_grand_average_vs_peaks(
    grand_mean: pd.Series,
    grand_sem: pd.Series,
    per_mouse_metrics: pd.DataFrame,
    signal: str,
    condition_label: str,
    cohort_identifier: str,
    experiment_day: str,
    output_dir: Path,
    reference_time: Optional[float] = None,
) -> Optional[Path]:
    """Plot the grand-average trace with per-mouse peak overlay and save to disk."""
    if grand_mean is None or grand_mean.empty:
        print(f"⚠️ Grand-average data unavailable for signal '{signal}'.")
        return None

    if grand_sem is None or grand_sem.empty:
        grand_sem = pd.Series(0.0, index=grand_mean.index)

    peaks = (
        per_mouse_metrics[per_mouse_metrics["signal"] == signal]["peak"].apply(pd.to_numeric, errors="coerce")
        if "signal" in per_mouse_metrics.columns
        else pd.Series(dtype=float)
    )
    peaks = peaks.dropna()
    if peaks.empty:
        print(f"⚠️ No per-mouse peak values available for signal '{signal}'.")
        return None

    peak_mean = float(peaks.mean())
    peak_sem = float(peaks.std(ddof=1) / np.sqrt(len(peaks))) if len(peaks) > 1 else 0.0

    time_points = pd.to_numeric(grand_mean.index, errors="coerce")
    values = pd.to_numeric(grand_mean.values, errors="coerce")
    sem_values = pd.to_numeric(grand_sem.reindex(grand_mean.index).fillna(0).values, errors="coerce")

    valid_mask = np.isfinite(time_points) & np.isfinite(values) & np.isfinite(sem_values)
    time_points = time_points[valid_mask]
    values = values[valid_mask]
    sem_values = sem_values[valid_mask]

    if time_points.size == 0:
        print(f"⚠️ Grand-average data for signal '{signal}' contains no finite values.")
        return None

    if reference_time is None:
        peak_idx = int(np.argmax(values))
        reference_time = float(time_points[peak_idx])

    jitter = np.linspace(-0.15, 0.15, len(peaks)) if len(peaks) > 1 else np.array([0.0])
    scatter_x = reference_time + jitter

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_points, values, color="black", linewidth=2, label="Grand average")
    ax.fill_between(time_points, values - sem_values, values + sem_values, color="grey", alpha=0.25, label="Grand avg ± SEM")

    band_bottom = peak_mean - peak_sem
    band_top = peak_mean + peak_sem
    ax.axhspan(band_bottom, band_top, color="#d62728", alpha=0.15, label="Per-mouse peak mean ± SEM")
    ax.axhline(peak_mean, color="#d62728", linestyle="--", linewidth=2)
    ax.scatter(scatter_x, peaks.values, color="#d62728", edgecolors="black", linewidths=0.8, s=55, zorder=5, label="Per-mouse peaks")

    annotation = (
        f"Per-mouse peak mean = {peak_mean:.3f}\n"
        f"SEM = {peak_sem:.3f}\n"
        f"n = {len(peaks)}"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.set_title(f"{signal.replace('_', ' ')} | {condition_label}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z-score")
    ax.grid(True, alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="lower right")

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_part = sanitize_for_filename(cohort_identifier, "cohort")
    day_part = sanitize_for_filename(experiment_day, "day")
    signal_part = sanitize_for_filename(signal, signal)
    condition_part = sanitize_for_filename(condition_label, "condition")
    output_path = output_dir / f"{cohort_part}_{day_part}_{signal_part}_{condition_part}_grand_vs_peaks.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Grand-average vs peaks plot saved to: {output_path}")
    return output_path


def _compute_grand_average_for_condition(
    condition_key: str,
    target_signal: str,
    target_mice: Optional[List[str]] = None,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    cfg = COMPARE_CALCIUM_METRICS_ACROSS_CONDITIONS.get(condition_key)
    if cfg is None:
        print(f"⚠️ Unknown condition '{condition_key}' — skipping grand-average overlay.")
        return None, None

    data_dirs = cfg.get("data_dirs", [])
    event_name_cond = cfg.get("event_name")
    if not data_dirs:
        print(f"⚠️ No data directories configured for condition '{condition_key}'.")
        return None, None

    mice_subset = []
    if target_mice:
        mice_subset = sorted({str(m) for m in target_mice})
    elif 'condition_metrics_frames' in locals():
        metrics_frame = condition_metrics_frames.get(condition_key)
        if metrics_frame is not None and not metrics_frame.empty:
            mice_subset = sorted(metrics_frame['mouse'].astype(str).unique())

    loaded = load_aligned_data(
        data_dirs,
        event_name_cond,
        mice_subset,
        available_mice,
    )
    if not loaded:
        print(f"⚠️ Unable to load aligned data for condition '{condition_key}'.")
        return None, None

    frames, _ = extract_mouse_dataframes_from_loaded(
        loaded,
        signal_columns=[target_signal],
    )
    if not frames:
        print(f"⚠️ No aligned traces available for condition '{condition_key}'.")
        return None, None

    series_list = []
    for mouse_id, df in frames.items():
        if target_signal not in df.columns:
            continue
        series = df[target_signal].astype(float).dropna()
        if series.empty:
            continue
        if series.index.duplicated().any():
            series = series.groupby(series.index).mean()
        series.name = str(mouse_id)
        series_list.append(series)

    if not series_list:
        print(f"⚠️ Condition '{condition_key}' lacks usable '{target_signal}' traces.")
        return None, None

    combined = pd.concat(series_list, axis=1)
    combined = combined.sort_index()

    grand_mean = combined.mean(axis=1)
    grand_sem = combined.sem(axis=1)
    if grand_sem.isna().all():
        grand_sem = pd.Series(0.0, index=grand_mean.index)
    else:
        grand_sem = grand_sem.fillna(0.0)

    return grand_mean, grand_sem


if 'condition_metrics_frames' not in locals() or not condition_metrics_frames:
    print("⚠️ Condition metrics unavailable; run the comparison cell before plotting overlays.")
else:
    target_signal = 'z_560_Baseline'
    resolved_main_dir = Path(main_data_dir).resolve()
    cohort_dir = resolved_main_dir.parent if resolved_main_dir.parent != resolved_main_dir else resolved_main_dir
    overlay_output_dir = cohort_dir / 'calcium_analysis'
    overlay_output_dir.mkdir(parents=True, exist_ok=True)

    apply_candidates = [
        key for key in ("Apply_halt_day4", "Apply_halt_day3")
        if key in condition_metrics_frames
    ]
    overlay_plan: List[str] = []
    if apply_candidates:
        overlay_plan.append(apply_candidates[0])
    if 'No_halt' in condition_metrics_frames:
        overlay_plan.append('No_halt')

    if not overlay_plan:
        print("⚠️ No suitable conditions found for grand-average overlays.")
    else:
        for condition_key in overlay_plan:
            per_mouse_metrics = condition_metrics_frames[condition_key]
            if per_mouse_metrics is None or per_mouse_metrics.empty:
                print(f"⚠️ No per-mouse metrics available for '{condition_key}'. Skipping overlay.")
                continue

            mice_for_condition = per_mouse_metrics['mouse'].astype(str).unique().tolist()
            grand_mean, grand_sem = _compute_grand_average_for_condition(
                condition_key,
                target_signal,
                mice_for_condition,
            )
            if grand_mean is None or grand_mean.empty:
                print(f"⚠️ Could not compute grand average for condition '{condition_key}'.")
                continue

            condition_label_for_plot = COMPARE_CALCIUM_METRICS_ACROSS_CONDITIONS[condition_key]["label"]
            plot_grand_average_vs_peaks(
                grand_mean,
                grand_sem,
                per_mouse_metrics,
                target_signal,
                condition_label_for_plot,
                cohort_identifier,
                experiment_day,
                overlay_output_dir,
                reference_time=POST_ALIGNMENT_WINDOW_START + POST_ALIGNMENT_WINDOW_DURATION / 2,
            )

