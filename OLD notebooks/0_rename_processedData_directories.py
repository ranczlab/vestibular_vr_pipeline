#wrangle directory renaming 
from pathlib import Path

# Define the parent directory containing all subdirectories
data_dir = Path('/Users/rancze/Documents/Data/vestVR/Cohort1/Visual_mismatch_day4')

# Find all directories
dir_dict = {}

# Group directories by their first 7 characters
for dir_path in data_dir.iterdir():
    if dir_path.is_dir():
        key = dir_path.name[:7]  # First 7 characters
        if key not in dir_dict:
            dir_dict[key] = {"raw": None, "processed": None}

        if dir_path.name.endswith("_processedData"):
            dir_dict[key]["processed"] = dir_path
        else:
            dir_dict[key]["raw"] = dir_path

# Rename processed directories
for key, pair in dir_dict.items():
    if pair["raw"] and pair["processed"]:
        new_name = f"{pair['raw'].name}_processedData"
        new_path = data_dir / new_name

        if not new_path.exists():
            pair["processed"].rename(new_path)
            print(f"Renamed: {pair['processed'].name} → {new_name}")
        else:
            print(f"Skipping {pair['processed'].name}, {new_name} already exists.")
