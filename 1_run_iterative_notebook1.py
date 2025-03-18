import papermill as pm
from pathlib import Path

notebook_path = Path("1_Loading_and_Sync_Cohort1.ipynb")
output_dir = notebook_path.parent.parent / "temp_output"  # One level up from notebook's location

# Ensure the output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# FIXME processedData after photometry processing directory name does not have session, only animal naje, not good
# make sure it's saved properly in photometry processing OR USE 0_rename_processedData_directories.py 
data_dir = Path('/Users/rancze/Documents/Data/vestVR/Cohort1/Visual_mismatch_day4')
data_paths = [Path(p) for p in data_dir.iterdir() if p.is_dir() and not p.name.endswith('_processedData')]

# data_paths = [
#     Path('/Users/rancze/Documents/Data/vestVR/Cohort1/Visual_mismatch_day3/B6J2717-2024-12-10T12-17-03'),
#     Path('/Users/rancze/Documents/Data/vestVR/Cohort1/Visual_mismatch_day3/B6J2718-2024-12-10T12-57-02'),
#     Path('/Users/rancze/Documents/Data/vestVR/Cohort1/Visual_mismatch_day3/B6J2719-2024-12-10T13-36-31'),
#     Path('/Users/rancze/Documents/Data/vestVR/Cohort1/Visual_mismatch_day3/B6J2721-2024-12-10T14-18-54'),
#     Path('/Users/rancze/Documents/Data/vestVR/Cohort1/Visual_mismatch_day3/B6J2722-2024-12-10T14-58-52'),
#     Path('/Users/rancze/Documents/Data/vestVR/Cohort1/Visual_mismatch_day3/B6J2723-2024-12-10T15-36-13')
# ]

for path in data_paths:
    output_notebook = output_dir / f"output_{path.name}.ipynb"
    
    print(f"Running notebook for: {path}")
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        parameters={'DATA_PATH': str(path)}
    )

    print(f"Saved output to: {output_notebook}")
