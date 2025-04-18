# to run, go to the repo directory and run "python3 1_run_iterative_notebook1.py" from terminal

import papermill as pm
from pathlib import Path

notebook_path = Path("1_Loading_and_Sync_Cohort1+_batch.ipynb")
output_dir = notebook_path.parent.parent / "temp_output"  # One level up from notebook's location
output_dir.mkdir(parents=True, exist_ok=True) # Ensure the output directory exists

#--------------------------------------------------------------------------------
# variables to be injected at runtime
#-------------------------------------------------------------------------------- 
# data path (recording day directory)
# data_dir = Path('/Users/rancze/Documents/Data/vestVR/Cohort1/Visual_Mismatch_day3')
data_dir = Path('/Volumes/RanczLab2/20250409_Cohort3_rotation/Training_day1')
data_paths = [Path(p) for p in data_dir.iterdir() if p.is_dir() and not p.name.endswith('_processedData')]
data_paths.sort()
# static variables 
vestibular_mismatch = False # if it is vestibular or visual MM, can be improved later with different experiment types, vestibular here means that the MM events are taken from the experiment_events instead of the photodiode
event_name = "Apply halt: 2s" # event name to be used for the MM events; for visual "Apply Halt 2s" for vestibular "DrumWithReverseflow block started"
sensor_resolution = 3100 #cpi, inferred empirical value from unit testing notebook  
ball_radius = 0.1 # meters 
optical_filter_Hz=40 #filter cutoff for Optical tracking and encoder signals
common_resampled_rate = 1000 #in Hz
save_full_asynchronous_data = True #saves alldata before resampling
# helper flags 
has_heartbeat = True # is this the same as cohort2 or more complicated? see discussion in repo 
cohort0 = False
cohort2 = True
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
            "Data_path": str(path),  # Iterative parameter
            "Vestibular_mismatch": vestibular_mismatch,  # Static parameter
            "Event_name": event_name,   # Static parameter
            "Sensor_resolution": sensor_resolution,   # Static parameter
            "Ball_radius": ball_radius,   # Static parameter
            "Optical_filter_Hz": optical_filter_Hz,   # Static parameter
            "Common_resampled_rate": common_resampled_rate,  # Static parameter
            "Save_full_asynchronous_data": save_full_asynchronous_data,   # Static parameter
            "Has_heartbeat": has_heartbeat,   # Static parameter
            "Cohort0": cohort0,    # Static parameter
            "Cohort2": cohort2,    # Static parameter
            "Has_sleap": has_sleap,    # Static parameter
        }
    )

    print(f"Saved output to: {output_notebook}")
