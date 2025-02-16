import os
import json
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from dotmap import DotMap
from sklearn.linear_model import LinearRegression

import harp
from aeon.io.reader import Reader, Csv, Harp
import aeon.io.api as api

import csv

#FIXME list
# some functions are very verbose, e.g. printing which dataset is loaded many times, giving how long it took (find and remove these time.time())
# when everything works, delete commented out functions

class SessionData(Reader):
    """Extracts metadata information from a settings .jsonl file."""

    def __init__(self, pattern):
        super().__init__(pattern, columns=["metadata"], extension="jsonl")

    def read(self, file, print_contents=True):
        """Returns metadata for the specified epoch."""
        with open(file) as fp:
            metadata = [json.loads(line) for line in fp] 

        data = {
            "metadata": [DotMap(entry['value']) for entry in metadata]
        }
        timestamps = [api.aeon(entry['seconds']) for entry in metadata]

        return pd.DataFrame(data, index=timestamps, columns=self.columns)
        # Pretty print if needed
        if print_contents:
            print(json.dumps(data, indent=4))


#not used 
# class PhotometryReader(Csv):
#     def __init__(self, pattern):
#         #super().__init__(pattern, columns=["Time", "Events", "CH1-410", "CH1-470", "CH1-560", "U"], extension="csv")
#         super().__init__(pattern, columns=["TimeStamp", "dfF_470", "dfF_560", "dfF_410"], extension="csv")
#         self._rawcolumns = self.columns

#     def read(self, file):
#         data = pd.read_csv(file, header=1, names=self._rawcolumns)
#         data.set_index("Time", inplace=True)
#         return data
    

class Video(Csv):
    def __init__(self, pattern):
        #super().__init__(pattern, columns = ["HardwareCounter", "HardwareTimestamp", "FrameIndex", "Path", "Epoch"], extension="csv")
        super().__init__(pattern, columns = ["HardwareCounter", "HardwareTimestamp", "FrameIndex"], extension="csv") #we don't need the path and epoch
        self._rawcolumns = ["Time"] + self.columns[0:2]

    def read(self, file):
        data = pd.read_csv(file, header=0, names=self._rawcolumns)
        data["FrameIndex"] = data.index
        #we don't need the path and epoch
        #data["Path"] = os.path.splitext(file)[0] + ".avi" #we d
        #data["Epoch"] = file.parts[-3]
        data["Time"] = data["Time"].transform(lambda x: api.aeon(x))
        data.set_index("Time", inplace=True)
        return data


class TimestampedCsvReader(Csv):
    def __init__(self, pattern, columns):
        super().__init__(pattern, columns, extension="csv")
        self._rawcolumns = ["Time"] + columns

    def read(self, file):
        data = pd.read_csv(file, header=0, names=self._rawcolumns)
        data["Seconds"] = data["Time"]
        data["Time"] = data["Time"].transform(lambda x: api.aeon(x))
        data.set_index("Time", inplace=True)
        return data
    
    
class OnixDigitalReader(Csv): #multiple files aware
    def __init__(self, pattern, columns):
        super().__init__(pattern, columns, extension="csv")
        self._rawcolumns = columns

    def read(self, file):
        try:
            processed_lines = []
            
            with open(file, 'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    
                    parts = line.strip().split(',')
                    
                    processed_line = {
                        'Seconds': float(parts[0]),
                        'Value.Clock': float(parts[1]),
                        'Value.HubClock': float(parts[2]),
                        'Value.DigitalInputs': parts[3].strip(),
                        'Value.Buttons': parts[-1]
                    }
                    processed_lines.append(processed_line)
            
            # Create DataFrame
            data = pd.DataFrame(processed_lines)
            
            # Keep only specified columns, no duplicates
            data = data[self._rawcolumns].copy()
            
            # Rename columns
            column_mapping = {
                'Value.Clock': 'Clock',
                'Value.HubClock': 'HubClock',
                'Value.DigitalInputs': 'DigitalInputs0'
            }
            data = data.rename(columns=column_mapping)
            
            # Check if DigitalInputs0 contains numbers
            try:
                data['DigitalInputs0'] = pd.to_numeric(data['DigitalInputs0'])
                is_numeric = True
            except ValueError:
                is_numeric = False
            
            # Create PhotometrySyncState column
            if is_numeric:
                data['PhotometrySyncState'] = data['DigitalInputs0'].apply(lambda x: x == 255)
            else:
                data['PhotometrySyncState'] = data['DigitalInputs0'].apply(lambda x: x.strip() == 'Pin0')
            
            # Transform to Time
            data["Time"] = data["Seconds"].apply(api.aeon)
            data.set_index("Time", inplace=True)
            
            return data
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            print("Data sample at error:", data.head() if 'data' in locals() else "No data loaded")
            raise
        
        
def load_2(reader: Reader, root: Path, verbose=False) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(reader.pattern).joinpath(reader.pattern)}_*.{reader.extension}"
    files = glob(pattern)
    
    # Create list of (file, timestamp) tuples
    file_timestamps = []
    for f in files:
        # Extract timestamp from filename (format: YYYY-MM-DDThh-mm-ss)
        timestamp = Path(f).stem.split('_')[-1]
        file_timestamps.append((f, timestamp))
    
    # Sort files by timestamp
    file_timestamps.sort(key=lambda x: x[1])
    sorted_files = [f[0] for f in file_timestamps]
    
    if verbose:
        print("\nFiles sorted by timestamp:")
        for f in sorted_files:
            print(f"  {Path(f).name}")
    
    data = []
    for file in sorted_files:
        if verbose:
            print(f"\nLoading: {Path(file).name}")
        data.append(reader.read(Path(file)))
    
    return pd.concat(data)


def load(reader: Reader, root: Path) -> pd.DataFrame: #used to load H1 & H2 registers
    root = Path(root)
    pattern = f"{root.joinpath(root.name)}_{reader.register.address}_*.bin"
    data = [reader.read(file) for file in glob(pattern)]
    return pd.concat(data)
    

def load_harp(reader: Harp, root: Path) -> pd.DataFrame: #multiple files aware, but not used (we use load_regiters instead)
    root = Path(root)
    pattern = f"{root.joinpath(root.name)}_{reader.register.address}_*.bin"
    print(pattern)
    data = [reader.read(file) for file in glob(pattern)]
    return pd.concat(data)


def concat_digi_events(series_low: pd.DataFrame, series_high: pd.DataFrame) -> pd.DataFrame:
    """Concatenate seperate high and low dataframes to produce on/off vector"""
    data_off = ~series_low[series_low==True]
    data_on = series_high[series_high==True]
    return pd.concat([data_off, data_on]).sort_index()


def read_ExperimentEvents(path):
    filenames = os.listdir(path/'ExperimentEvents')
    filenames = [x for x in filenames if x[:16]=='ExperimentEvents'] # filter out other (hidden) files
    sorted_filenames = pd.to_datetime(pd.Series([x.split('_')[1].split('.')[0] for x in filenames])).sort_values()
    read_dfs = []
    try:
        for row in sorted_filenames:
            read_dfs.append(pd.read_csv(path/'ExperimentEvents'/f"ExperimentEvents_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"))
        return pd.concat(read_dfs).reset_index().drop(columns='index')
    except pd.errors.ParserError as e:
        filename = f"ExperimentEvents_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"
        print(f'Tokenisation failed for file "{filename}".\n')
        print(f'Exact description of error: {e}')
        print('Likely due to extra commas in the "Value" column of ExperimentEvents. Please manually remove and run again.')
        return None
    except Exception as e:
        print('Reading failed:', e)
        return None


# def read_OnixAnalogFrameCount(path): #multiple files aware, but we use load_2 instead
#     filenames = os.listdir(path/'OnixAnalogFrameCount')
#     filenames = [x for x in filenames if x[:20]=='OnixAnalogFrameCount'] # filter out other (hidden) files
#     sorted_filenames = pd.to_datetime(pd.Series([x.split('_')[1].split('.')[0] for x in filenames])).sort_values()
#     read_dfs = []
#     for row in sorted_filenames:
#         read_dfs.append(pd.read_csv(path/'OnixAnalogFrameCount'/f"OnixAnalogFrameCount_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"))
#     return pd.concat(read_dfs).reset_index().drop(columns='index')


def read_OnixAnalogData(dataset_path, channels=[0], binarise=False, method='adaptive', refractory = 300, flip=True, verbose=False):
    """Read and process Onix analog data.
    
    Args:
        dataset_path: Path to data
        channels: List of channels to read [0-11]
        binarise: Whether to binarize data
        method: 'adaptive' or 'threshold'
        flip: Flip binary values
        verbose: Print processing details
    """
    start_time = time.time()
    arrays_to_concatenate = []
    files_to_read = [x for x in os.listdir(dataset_path/'OnixAnalogData')]
    
    def extract_number(filename):
        return int(filename.split('_')[-1].split('.')[0])
    
    # Sort files
    files_to_read.sort(key=extract_number)
    
    for filename in files_to_read:
        with open(dataset_path/'OnixAnalogData'/filename, 'rb') as f:
            photo_diode = np.fromfile(f, dtype=np.int16)

            try:
                photo_diode = np.reshape(photo_diode, (-1, 12))[:, channels]
            except:
                print(f'ERROR: Cannot reshape loaded "{filename}" binary file into [-1, 12] shape. Continuing with non-reshaped data.')
            
            arrays_to_concatenate.append(photo_diode)

    # Concatenate arrays
    photo_diode = np.concatenate(arrays_to_concatenate)

    # Return raw data if binarize=False
    if not binarise:
        return photo_diode

    if binarise:
        # Store original shape
        original_shape = photo_diode.shape
        
        # Ensure 1D array
        if photo_diode.ndim > 1:
            photo_diode = photo_diode.flatten()
        
        if method == 'adaptive':
            try:
                threshold = np.percentile(photo_diode, 1)
                threshold = threshold * 1.2
                if verbose:
                    print(f"Adaptive threshold: {threshold}")
            except:
                if verbose:
                    print("Using default threshold")
                threshold = 120
        else:  # hard threshold
            threshold = 120
            if verbose:
                print(f"Using hard threshold: {threshold}")

        # Binarize
        photo_diode = photo_diode <= threshold
        photo_diode = photo_diode.astype(bool)
        
        # Flip if requested
        if flip:
            photo_diode = ~photo_diode
        
        # Convert to int for transition detection
        signal_int = photo_diode.astype(int)
        diff_signal = np.diff(signal_int)
        
        # Find falling edges with refractory period
        fall_indices = np.where(diff_signal < 0)[0]
        valid_falls = []
        last_fall = -300  # Initialize before first possible transition
        
        for idx in fall_indices:
            if idx - last_fall > refractory:  # Check refractory period
                valid_falls.append(idx)
                last_fall = idx
        
        falling_edges = len(valid_falls)
        
        if verbose:
            print(f"Number of falling edges detected: {falling_edges}")
            #if falling_edges > 0:
               # print(f"First 5 falling edge indices: {valid_falls[:5]}")
            if falling_edges == 0:
                print("Warning: No falling edges detected. Check threshold value and signal.")
        
        return photo_diode
    
    return photo_diode


def read_OnixAnalogClock(dataset_path): #multiple files aware
    
    arrays_to_concatenate = []
    files_to_read = [x for x in os.listdir(dataset_path/'OnixAnalogClock')]
    
    def extract_number(filename):
        return int(filename.split('_')[-1].split('.')[0])
    
    # Sort the files based on the extracted number
    files_to_read.sort(key=extract_number)
    
    for filename in files_to_read:
        with open(dataset_path/'OnixAnalogClock'/filename, 'rb') as f:
            clock_data = np.fromfile(f, dtype=np.uint64)
            arrays_to_concatenate.append(clock_data) 
    
    output = np.concatenate(arrays_to_concatenate)


    return output


def read_SessionSettings(dataset_path, print_contents=False):
    path = Path(dataset_path)
    session_settings_path = path / 'SessionSettings'
    
    # List and filter only JSONL files
    jsonl_files = [f for f in os.listdir(session_settings_path) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {session_settings_path}")

    jsonl_file_path = session_settings_path / jsonl_files[0]  # Pick the first valid .jsonl file, there should be only one!

    # Open and read the jsonl file line by line
    with open(jsonl_file_path, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.strip()  # Remove whitespace/newlines
            if not line: 
                continue  # Skip empty lines

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line: {line[:100]}... Error: {e}")
                continue
            
            # Pretty print if needed
            if print_contents:
                print(json.dumps(data, indent=4))

    return data

# def read_fluorescence(photometry_data_path):
#     try:
#         Fluorescence = pd.read_csv(photometry_data_path/'Processed_Fluorescence.csv', skiprows=1, index_col=False)
#     except FileNotFoundError:
#         Fluorescence = pd.read_csv(photometry_data_path/'Fluorescence.csv', skiprows=1, index_col=False)
        
#     if 'Unnamed: 5' in Fluorescence.columns: Fluorescence = Fluorescence.drop(columns='Unnamed: 5')
#     return Fluorescence

# def read_fluorescence_events(photometry_data_path):
#     Events = pd.read_csv(photometry_data_path/'Events.csv', skiprows=0, index_col=False)
#     return Events


def load_registers(dataset_path, dataframe=True, verbose = False):
    """Load register data and return as either dictionary or DataFrame
    Args:
        dataset_path: Path to data directory
        dataframe: If True returns DataFrame, if False returns dict
    """
    h1_dict, h2_dict = load_register_paths(dataset_path, verbose)
    
    h1_data_streams = {}
    for register in h1_dict.keys():
        data_stream = load(get_register_object(register, 'h1'), dataset_path/'HarpDataH1')
        
        if data_stream.columns.shape[0] > 1:
            for col_name in data_stream.columns:
                h1_data_streams[f'{col_name}({register})'] = data_stream[col_name]
        elif data_stream.columns.shape[0] == 1:
            h1_data_streams[f'{data_stream.columns[0]}({register})'] = data_stream
        else:
            raise ValueError("Loaded data stream does not contain supported number of columns")
    
    h2_data_streams = {}
    for register in h2_dict.keys():
        data_stream = load(get_register_object(register, 'h2'), dataset_path/'HarpDataH2')
        
        if data_stream.columns.shape[0] > 1:
            for col_name in data_stream.columns:
                h2_data_streams[f'{col_name}({register})'] = data_stream[col_name]
        elif data_stream.columns.shape[0] == 1:
            h2_data_streams[f'{data_stream.columns[0]}({register})'] = data_stream[data_stream.columns[0]]
        else:
            raise ValueError("Loaded data stream does not contain supported number of columns")
    
    # Convert DataFrames to Series
    for streams in [h1_data_streams, h2_data_streams]:
        for stream_name, stream in streams.items():
            if isinstance(stream, pd.DataFrame):
                try:
                    streams[stream_name] = pd.Series(data=stream.values.squeeze(), index=stream.index)
                except:
                    raise ValueError(f"Failed to convert register {stream_name} to Series")
    
    if dataframe:
        # Return as DataFrame
        all_series = {**h1_data_streams, **h2_data_streams}
        return pd.DataFrame(all_series)
    else:
        # Return as dict
        return {'H1': h1_data_streams, 'H2': h2_data_streams}
  
    
def load_register_paths(dataset_path, verbose=False):
    
    if not os.path.exists(dataset_path/'HarpDataH1') or not os.path.exists(dataset_path/'HarpDataH2'):
        raise FileNotFoundError(f"'HarpDataH1' or 'HarpDataH2' folder was not found in {dataset_path}.")
    h1_folder = dataset_path/'HarpDataH1'
    h2_folder = dataset_path/'HarpDataH2'
    
    if verbose: print("\nProcessing H1 files:")
    h1_files = os.listdir(h1_folder)
    h1_files = [f for f in h1_files if f.split('_')[0] == 'HarpDataH1']
    if verbose: print("Found H1 files:", h1_files)
    
    h1_dict = {}
    for filename in h1_files:
        register = int(filename.split('_')[1])
        if register not in h1_dict:
            h1_dict[register] = []
        h1_dict[register].append(h1_folder/filename)
        if verbose: print(f"Added to register {register}: {filename}")
    
    # Sort files by timestamp for each register
    for register in h1_dict:
        h1_dict[register].sort()  # Files sort by timestamp naturally
        if verbose:
            print(f"\nSorted files for H1 register {register}:")
            for f in h1_dict[register]:
                print(f"  {f.name}")
    
    if verbose: print("\nProcessing H2 files:")
    h2_files = os.listdir(h2_folder)
    h2_files = [f for f in h2_files if f.split('_')[0] == 'HarpDataH2']
    if verbose: print("Found H2 files:", h2_files)
    
    h2_dict = {}
    for filename in h2_files:
        register = int(filename.split('_')[1])
        if register not in h2_dict:
            h2_dict[register] = []
        h2_dict[register].append(h2_folder/filename)
        if verbose: print(f"Added to register {register}: {filename}")
    
    # Sort files by timestamp for each register
    for register in h2_dict:
        h2_dict[register].sort()  # Files sort by timestamp naturally
        if verbose:
            print(f"\nSorted files for H2 register {register}:")
            for f in h2_dict[register]:
                print(f"  {f.name}")
            
    # Print final dictionary contents
    if verbose:
        print("\nFinal dictionaries:")
        print("H1 Dictionary:")
        for reg in sorted(h1_dict.keys()):
            print(f"  Register {reg}:")
            for f in h1_dict[reg]:
                print(f"    {f.name}")
                
        print("\nH2 Dictionary:")
        for reg in sorted(h2_dict.keys()):
            print(f"  Register {reg}:")
            for f in h2_dict[reg]:
                print(f"    {f.name}")
    
    return h1_dict, h2_dict
    
    
def get_register_object(register_number, harp_board='h1'):
    h1_reader = harp.create_reader(f'harp_resources/h1-device.yml', epoch=harp.REFERENCE_EPOCH)
    h2_reader = harp.create_reader(f'harp_resources/h2-device.yml', epoch=harp.REFERENCE_EPOCH)
    reference_dict = {
        'h1': {
            32: h1_reader.Cam0Event,
            33: h1_reader.Cam1Event,
            38: h1_reader.StartAndStop,
            46: h1_reader.OpticalTrackingRead
        },
        'h2': {
            38: h2_reader.Encoder,
            39: h2_reader.AnalogInput,
            42: h2_reader.ImmediatePulses
        }
    }
    
    if register_number not in reference_dict[harp_board]:
        raise KeyError(f"Register number {register_number} not found for harp board {harp_board}")
    
    return reference_dict[harp_board][register_number]
    

def detect_and_remove_outliers(df, x_column, y_column, verbose=False):
    """
    Used for onix_harp as the Harp clock (very rarely) produces outliers.
    Detects and removes outliers from a DataFrame using IQR and residual-based methods.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        x_column (str): The column name for the independent variable.
        y_column (str): The column name for the dependent variable.
        verbose (bool): If True, prints the list of outliers. If False, prints the number of outliers found.
    
    Returns:
        pd.DataFrame: A DataFrame with outliers removed.
    """
    # Step 1: Detect outliers using IQR (initial filtering)
    def detect_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data[column] < lower_bound) | (data[column] > upper_bound)
    
    # Apply IQR-based outlier detection
    x_outliers = detect_outliers_iqr(df, x_column)
    y_outliers = detect_outliers_iqr(df, y_column)
    initial_outliers = df[x_outliers | y_outliers]
    filtered_data = df[~(x_outliers | y_outliers)]

    # Step 2: Fit linear model on filtered data
    X_filtered = filtered_data[x_column].values.reshape(-1, 1)
    y_filtered = filtered_data[y_column].values
    model = LinearRegression()
    model.fit(X_filtered, y_filtered)

    # Step 3: Detect outliers based on residuals
    y_pred = model.predict(X_filtered)
    residuals = y_filtered - y_pred
    residual_threshold = 3 * np.std(residuals)  # Use 3 * std as threshold
    residual_outliers_mask = np.abs(residuals) > residual_threshold

    # Add residual-based outliers back to outliers
    residual_outliers = filtered_data[residual_outliers_mask]
    final_outliers = pd.concat([initial_outliers, residual_outliers])

    # Remove outliers from the original DataFrame
    cleaned_data = df[~df.index.isin(final_outliers.index)]

    # Verbose output
    if not final_outliers.empty:
        if verbose:
            print(f"Outliers detected")
            print(final_outliers[[x_column, y_column]])
            print(f"Warning: {len(final_outliers)} outliers detected and removed.")
        else:
            print(f"Warning: {len(final_outliers)} outliers detected and removed.")

    return cleaned_data


def load_streams_from_h5(data_path):
    # File path to read the HDF5 file
    input_file = data_path/f'resampled_streams_{data_path.parts[-1]}.h5'

    if not os.path.exists(input_file):
        print(f'ERROR: {input_file} does not exist.')
        return None

    # Open the HDF5 file to read data
    with h5py.File(input_file, 'r') as h5file:
        # Read the common index (which was saved as Unix timestamps)
        common_index = h5file['HARP_timestamps'][:]
        
        # Convert Unix timestamps back to pandas DatetimeIndex
        # common_index = pd.to_datetime(common_index)
        
        # Initialize the dictionary to reconstruct the data
        reconstructed_streams = {}
        
        # Iterate through the groups (sources) in the file
        for source_name in h5file.keys():
            if source_name == 'HARP_timestamps':
                # Skip the 'common_index' dataset, it's already loaded
                continue
            
            # Initialize a sub-dictionary for each source
            reconstructed_streams[source_name] = {}
            
            # Get the group (source) and iterate over its datasets (streams)
            source_group = h5file[source_name]
            
            for stream_name in source_group.keys():
                # Read the stream data
                stream_data = source_group[stream_name][:]
                
                # Reconstruct the original pd.Series with the common index
                reconstructed_streams[source_name][stream_name] = pd.Series(data=stream_data, index=common_index)

    return reconstructed_streams