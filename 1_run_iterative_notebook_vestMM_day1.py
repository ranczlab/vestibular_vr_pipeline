# script to run analysis on a whole folder of recordings of the same type 
# it moans it does not find a cell tagged parameters when running, but that is only cosmetic and it is a drag to do it in VS Code, needs to be tagged manually in Jupyter

import papermill as pm
from pathlib import Path

notebook_path = Path("1_Loading_and_Sync_Cohort1.ipynb")
output_dir = notebook_path.parent.parent / "temp_output"  # One level up from notebook's location
output_dir.mkdir(parents=True, exist_ok=True) # Ensure the output directory exists

#--------------------------------------------------------------------------------
# variables to be injected at runtime
#-------------------------------------------------------------------------------- 
# data path (recording day directory)
data_dir = Path('/Users/rancze/Documents/Data/vestVR/Cohort1/VestibularMismatch_day1')
data_paths = [Path(p) for p in data_dir.iterdir() if p.is_dir() and not p.name.endswith('_processedData')]
# static variables 
vestibular_mismatch = True # if it is vestibular or visual MM, can be improved later with different experiment types, vestibular here means that the MM events are taken from the experiment_events instead of the photodiode
event_name = "DrumWithReverseflow block started" # event name to be used for the MM events
sensor_resolution = 3100 #cpi, inferred empirical value from unit testing notebook  
ball_radius = 0.1 # meters 
optical_filter_Hz=40 #filter cutoff for Optical tracking and encoder signals
common_resampled_rate = 1000 #in Hz
save_full_asynchronous_data = True #saves alldata before resampling
# helper flags 
has_heartbeat = False # is this the same as cohort2 or more complicated? see discussion in repo 
cohort0 = False
cohort2 = False 
#-------------------------------------------------------------------------------- 

for path in data_paths:
    output_notebook = output_dir / f"output_{path.name}.ipynb"
    
    print(f"Running notebook for: {path}")
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        #parameters={'DATA_PATH': str(path)}
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
        }
    )

    print(f"Saved output to: {output_notebook}")
