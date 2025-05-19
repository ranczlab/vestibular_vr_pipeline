# to run, go to the repo directory and run "python3 1_run_iterative_notebook1.py" from terminal

import papermill as pm
from pathlib import Path

notebook_path = Path("SANDBOX_2_noSLEAP_BATCH-alignment_papermill.ipynb")
output_dir = notebook_path.parent.parent / "temp_output"  # One level up from notebook's location
output_dir.mkdir(parents=True, exist_ok=True) # Ensure the output directory exists

#--------------------------------------------------------------------------------
# variables to be injected at runtime
#-------------------------------------------------------------------------------- 
#--------------------------------------------------------------------------------
data_dir = Path('~/RANCZLAB-NAS/data/ONIX/20250409_Cohort3_rotation/Visual_mismatch_day3').expanduser() # injects at runtime by papermill or define explicetily below
time_window_start = -6  # s, FOR PLOTTING PURPOSES
time_window_end = 10  # s, FOR PLOTTING PURPOSES
baseline_window = (-1, 0) # s, FOR baselining averages 
plot_width= 14

event_name = "Apply halt: 2s" #Apply halt: 2s, No halt
vestibular_mismatch = False
common_resampled_rate = 1000 #in Hz
plot_fig1 = False

#-------------------------------------------------------------------------------- 

for path in data_paths:
    output_notebook = output_dir / f"output_{path.name}.ipynb"
    # Check if Sleap data exists
    sleap_data_path1 = path.parent / f"{path.name}_processedData" / "Video_Sleap_Data1"
    sleap_data_path2 = path.parent / f"{path.name}_processedData" / "Video_Sleap_Data2"
    if sleap_data_path1.exists() or sleap_data_path2.exists():
        has_sleap = True
    else:
        has_sleap = False
    print(f"ℹ️  Running notebook for: {path}\nℹ️  Sleap data is {has_sleap}")
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        parameters={
            "Data_path" = Path(Data_path) # injects at runtime by papermill or define explicetily below
            "Time_window_start" = -6  # s, FOR PLOTTING PURPOSES
            "Time_window_end" = 10  # s, FOR PLOTTING PURPOSES
            "Baseline_window" = (-1, 0) # s, FOR baselining averages 
            "Plot_width"= 14

            "Event_name" = "Apply halt: 2s" #Apply halt: 2s, No halt
            "Vestibular_mismatch" = False
            "Common_resampled_rate" = 1000 #in Hz
            "Plot_fig1" = False
            "Has_sleap": has_sleap,    # Static parameter
        }
    )

    print(f"Saved output to: {output_notebook}")
