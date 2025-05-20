# to run, go to the repo directory and run "python3 run_notebook_batch.py" from terminal

import papermill as pm
from pathlib import Path
import os
import json
import sys

notebook_path = Path("SANDBOX_2_noSLEAP_BATCH-alignment_papermill.ipynb")
output_dir = Path("./temp_output")  # One level up from notebook's location
output_dir.mkdir(parents=True, exist_ok=True) # Ensure the output directory exists

#--------------------------------------------------------------------------------
# DEFAULT PARAMETERS - These will be used unless overridden for specific sessions
#--------------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "Time_window_start": -6,            # s, FOR PLOTTING PURPOSES
    "Time_window_end": 10,              # s, FOR PLOTTING PURPOSES
    "Baseline_window": (-1, 0),         # s, FOR baselining averages 
    "Plot_width": 14,
    "Event_name": "Apply halt: 2s",     # "Apply halt: 2s" or "No halt"
    "Vestibular_mismatch": False,
    "Common_resampled_rate": 1000,      # in Hz
    "Plot_fig1": False,
    "Has_sleap": False                  # Will be auto-detected
}

#--------------------------------------------------------------------------------
# DATA DIRECTORY
#--------------------------------------------------------------------------------
# Default path - change this to your actual data directory
default_data_dir = Path('~/Desktop/RANCZLAB-NAS/data/ONIX/20250409_Cohort3_rotation/Visual_mismatch_day3').expanduser()

# Allow override from command line
if len(sys.argv) > 1:
    data_dir = Path(sys.argv[1]).expanduser()
    print(f"Using data directory from command line: {data_dir}")
else:
    data_dir = default_data_dir
    print(f"Using default data directory: {data_dir}")

# Verify data directory exists
if not data_dir.exists():
    print(f"❌ Error: Data directory {data_dir} does not exist!")
    sys.exit(1)

# Find the actual data paths by looking for directories with processedData/downsampled_data
data_paths = []
for p in data_dir.iterdir():
    if p.is_dir() and not p.name.endswith('_processedData'):
        # Extract session name for potential parameter overrides
        session_name = p.name
        
        # Look for the corresponding processed data directory
        processed_dir = p.parent / f"{session_name}_processedData"
        downsampled_dir = processed_dir / "downsampled_data"
        
        if downsampled_dir.exists() and downsampled_dir.is_dir():
            # Verify required files exist before adding to processing list
            required_files = [
                "photometry_tracking_encoder_data.parquet",
                "camera_photodiode_data.parquet",
                "experiment_events.parquet",
                "photometry_info.parquet",
                "session_settings.parquet"
            ]
            
            files_exist = True
            for filename in required_files:
                file_path = downsampled_dir / filename
                if not file_path.exists():
                    print(f"❌ Warning: Required file {filename} missing in {downsampled_dir}")
                    files_exist = False
                    break
            
            if files_exist:
                data_paths.append((session_name, downsampled_dir))
            else:
                print(f"⚠️ Skipping {session_name} due to missing required files")

# Sort by session name
data_paths.sort(key=lambda x: x[0])

#--------------------------------------------------------------------------------
# EXECUTION
#--------------------------------------------------------------------------------

# Print some debug info
print(f"Found {len(data_paths)} data paths to process")
if data_paths:
    print("First 3 sessions:")
    for i, (session_name, path) in enumerate(data_paths[:3], 1):
        print(f"{i}. {session_name}: {path}")
        # Verify data files exist
        parquet_files = list(path.glob("*.parquet"))
        print(f"   Found {len(parquet_files)} parquet files")
        # Print actual strings for debugging path issues
        print(f"   Path as string: {str(path)}")
else:
    print("❌ No data paths found! Check the folder structure.")

print(f"Notebook path: {notebook_path}")
print(f"Notebook exists: {notebook_path.exists()}")

if not notebook_path.exists():
    print(f"❌ Error: Notebook file {notebook_path} does not exist!")
    sys.exit(1)

for idx, (session_name, path) in enumerate(data_paths, 1):
    # Create a unique output notebook name
    output_notebook = output_dir / f"output_{session_name}.ipynb"
    
    # Check if Sleap data exists
    sleap_data_path1 = path.parent / "Video_Sleap_Data1"
    sleap_data_path2 = path.parent / "Video_Sleap_Data2"
    has_sleap = sleap_data_path1.exists() or sleap_data_path2.exists()
    
    # Start with the default parameters
    params = DEFAULT_PARAMS.copy()
    
    # Convert Path object to string to avoid serialization issues
    params["Data_path"] = str(path)
    params["Has_sleap"] = has_sleap
    
    print(f"\n============================================================")
    print(f"ℹ️  Processing {idx}/{len(data_paths)}: {session_name}")
    print(f"ℹ️  Data path: {path}")
    print(f"ℹ️  Data path as string: {str(path)}")
    print(f"ℹ️  Sleap data available: {has_sleap}")
    print(f"ℹ️  Parameters:")
    for key, value in params.items():
        print(f"   - {key}: {value}")
    
    try:
        # Execute the notebook with papermill
        pm.execute_notebook(
            notebook_path,
            output_notebook,
            parameters=params,
            log_output=True,
            report_mode=True  # Makes papermill more verbose about parameter injection
        )
        print(f"✅ Successfully executed notebook for {session_name}")
        print(f"✅ Output saved to: {output_notebook}")
    except Exception as e:
        print(f"❌ Error processing {session_name}: {str(e)}")
        # Print stack trace for debugging
        import traceback
        traceback.print_exc()

print("\n✅ Processing complete!")