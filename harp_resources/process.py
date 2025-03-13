import os
import copy
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from datetime import datetime, timedelta
from scipy.signal import correlate
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy import signal  # for Lomb-Scargle PSD
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import re # for photometry alignment
from dotmap import DotMap
import json

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import plotly.subplots as sp  # Keep sp for other parts of your code
make_subplots = sp.make_subplots  # Explicitly define make_subplots

import time
import logging

from . import utils
import aeon.io.api as api

def downsample_numpy(arr, factor, method="mean"):
    """
    Downsamples a NumPy array by a given factor.
    
    Args:
        arr (np.ndarray): Input array.
        factor (int): Downsampling factor.
        method (str): 'mean', 'median', or 'first' for downsampling strategy.
        
    Returns:
        np.ndarray: Downsampled array.
    """
    if method == "mean":
        return arr[:len(arr)//factor * factor].reshape(-1, factor).mean(axis=1)
    elif method == "median":
        return np.median(arr[:len(arr)//factor * factor].reshape(-1, factor), axis=1)
    elif method == "first":
        return arr[::factor]
    else:
        raise ValueError("Invalid method. Choose 'mean', 'median', or 'first'.")

def running_unit_conversion(running_array): #for ball linear movement
    resolution = 12000 # counts per inch
    inches_per_count = 1 / resolution
    meters_per_count = 0.0254 * inches_per_count #inch to meter conversion 
    #dt = 0.01 # for OpticalTrackingRead0Y(46) -this is sensor specific. current sensor samples at 100 hz 
    #linear_velocity = meters_per_count / dt # meters per second per count; this makes no sense as array will go back to datetime indexed harp_streams df
    return running_array * meters_per_count

def encoder_unit_conversion(encoder_array, home_position): #for ball linear movement
    encoder_resolution = 4000 # counts per revolution
    gear_ratio = 6 # motor to platform 
    wrap_period = 65536  # (32767 - (-32768) + 1)

    encoder_unwrapped = np.unwrap(encoder_array, discont=wrap_period / 2)
    position = (encoder_unwrapped - home_position) / (encoder_resolution * gear_ratio / 360)
    wrapped_position = np.mod(position, 360) # wrap to 0-360 degrees
    return wrapped_position 

def get_encoder_home_position(experiment_events, harp_streams, event="Homing platform", column="Encoder(38)"):
    """
    Finds the first occurrence of `event` in `experiment_events`, then finds the next event in the DataFrame.
    Retrieves the corresponding next valid values from the specified column in `harp_streams`.

    Args:
        experiment_events (pd.DataFrame): DataFrame with datetime index and event labels.
        harp_streams (pd.DataFrame): DataFrame with datetime index containing the target column.
        event (str): The event to search for (default: "Homing platform").
        column (str): The target column in `harp_streams` (default: "Encoder(38)").

    Returns:
        tuple: (encoder_value_event, encoder_value_next_event), where each value is the first valid
               value found in `harp_streams` after the corresponding event timestamp.
    """
    # Step 1: Find the first occurrence of the specified event
    event_rows = experiment_events[experiment_events["Event"] == event]
    
    if event_rows.empty:
        return None, None  # Event not found

    # Step 2: Extract the first timestamp where the event occurs
    first_event_idx = event_rows.index[0]

    # Step 3: Find the next event that occurs AFTER the first one
    next_event_df = experiment_events.loc[first_event_idx:].iloc[1:]  # Start after the first event
    next_event_df = next_event_df[next_event_df["Event"] != event]  # Ignore repeated events

    # Step 4: Extract the timestamp of the next event
    next_event_idx = next_event_df.index[0] if not next_event_df.empty else None

    # Function to find the next valid timestamp in harp_streams
    def get_next_valid_value(timestamp, df, column):
        if timestamp is None:
            return None  # No valid timestamp found
        
        # Get valid timestamps where column is not NaN
        valid_indices = df.index[df[column].notna()]
        pos = valid_indices.searchsorted(timestamp)  # Find next valid timestamp

        if pos < len(valid_indices):  # Check if we found a valid index
            return df.loc[valid_indices[pos], column]

        return None  # No valid index found

    # Step 3: Get values from harp_streams
    encoder_value_event = get_next_valid_value(first_event_idx, harp_streams, column)
    encoder_value_next_event = get_next_valid_value(next_event_idx, harp_streams, column)

    return encoder_value_event, encoder_value_next_event

def turning_unit_conversion(turning_array): # for ball rotation
    resolution = 12000 # counts per inch
    inches_per_count = 1 / resolution
    meters_per_count = 0.0254 * inches_per_count
    #dt = 0.01 # for OpticalTrackingRead0Y(46) -this is sensor specific. current sensor samples at 100 hz 
    turning_array = turning_array * meters_per_count #/ dt # meters per second per count 
    
    ball_radius = 0.1 # meters 
    turning_array = turning_array / ball_radius # radians per second per count
    turning_array = turning_array * (180 / np.pi) # degrees per second per count
    
    return turning_array

def photometry_harp_onix_synchronisation(
    onix_digital, 
    onix_harp,
    photometry_events,
    photometry_data,
    photodiode_df, 
    verbose=True
):
    output = {}

    #---------------------
    # Onix alingment 
    #---------------------

    # Find the time mapping/warping between onix and harp clock
    clock = onix_harp["Clock"]
    harp = onix_harp["HarpTime"]
    o_m, o_b = np.polyfit(clock, harp, 1)
    onix_to_harp_seconds = lambda x: x * o_m + o_b
    onix_to_harp = lambda x: api.aeon(onix_to_harp_seconds(x))
    harp_to_onix = lambda x: (x - o_b) / o_m
     
    # Calculate R-squared value
    y_pred = onix_to_harp_seconds(clock)
    ss_res = np.sum((harp - y_pred) ** 2)
    ss_tot = np.sum((harp - np.mean(harp)) ** 2)
    r_squared_harp_to_onix = 1 - (ss_res / ss_tot)

    if r_squared_harp_to_onix < 0.999:
        print(f"❗WARNING: R-squared value between Clock and HarpTime in onix_harp is {r_squared_harp_to_onix:.6f}. Could be an issue with the dataset.")
    output["onix_to_harp"] = onix_to_harp
    output["harp_to_onix"] = harp_to_onix

    #---------------------
    # Photometry alignment
    #---------------------
    # Find the time mapping/warping between photometry and harp clock
    # Drop rows with NaN values from photometry_events and convert to seconds
    nan_count = photometry_events.isna().sum().sum()
    if verbose and nan_count > 0:
        print(f"❗WARNING:, {nan_count} NaNs detected, check Events'csv")
    photometry_sync_events = photometry_events.dropna()
    photometry_sync_events = photometry_sync_events[photometry_sync_events['Name'] == 'Input1']  # Filter to Input1
    photometry_sync_events = photometry_sync_events["State"]

    # Truncate photometry to onix_digital as photometry recording starts before the sync signal 
    min_length = min(len(photometry_sync_events), len(onix_digital["Clock"]))
    photometry_sync_events = photometry_sync_events.iloc[:min_length]
    num_truncated = len(photometry_sync_events) - min_length
    if num_truncated > 1:
        print(f"❗WARNING: Photometry_sync_events was truncated by {num_truncated} points, more than the expected single sync pulse.")
    onix_digital["PhotometrySyncState"] = ~onix_digital["PhotometrySyncState"]  # Flip sync line to match photometry FIXME why is this necessary?

    # Define conversion functions between timestamps (photometry to harp)
    m, b = np.polyfit(photometry_sync_events.index, onix_digital["Clock"], 1)
    photometry_to_onix = lambda x: x * m + b
    photometry_to_harp = lambda x: onix_to_harp(photometry_to_onix(x))
    onix_to_photometry = lambda x: (x - b) / m #FIXME used???
    
    # Calculate R-squared value
    y_pred = photometry_to_onix(photometry_sync_events.index)
    ss_res = np.sum((onix_digital["Clock"] - y_pred) ** 2)
    ss_tot = np.sum((onix_digital["Clock"] - np.mean(onix_digital["Clock"])) ** 2)
    r_squared_photometry_to_onix = 1 - (ss_res / ss_tot)
    
    if r_squared_photometry_to_onix < 0.999:
        print(f"❗WARNING: R-squared value between photometry and onix_digital is {r_squared_photometry_to_onix:.6f}. Could be an issue with the dataset.")

    output["photometry_to_onix"] = photometry_to_onix
    output["photometry_to_harp"] = photometry_to_harp
    
    if verbose:
        print(f"for onix to harp o_m: {o_m}, o_b: {o_b}")
        print(f"R-squared value for harp to onix: {r_squared_harp_to_onix}")
        print(f"for photometry to harp m: {m}, b: {b}")
        print(f"R-squared value for photometry to onix: {r_squared_photometry_to_onix}")

    #---------------------
    # Align photometry_data  
    #---------------------   

    # Align photometry_data, photodiode_df, photometry_sync_events and onix_digital events to harp
    photometry_aligned = photometry_data.copy()
    photometry_aligned.index = photometry_to_harp(photometry_data.index)
    photometry_aligned.index = photometry_aligned.index.round("us")  # Convert to microseconds
    photometry_aligned.index.name = 'Time'  # Rename the index to 'Time'

    photodiode_aligned = photodiode_df.copy()
    photodiode_aligned.index = onix_to_harp(photodiode_aligned.index)
    photodiode_aligned.index = photodiode_aligned.index.round("us")  # Convert to microseconds
    photodiode_aligned.index.name = 'Time'  # Rename the index to 'Time'

    photometry_sync_aligned = photometry_to_harp(photometry_sync_events.index) #for plotting only 
    onix_digital_aligned = onix_to_harp(onix_digital["Clock"]) #for plotting only

    # Plotting
    if verbose:
        print(f"{len(photometry_sync_events)} events found")
        if not photometry_sync_events.empty:
            if len(photometry_sync_events) > 15:
                print("Plotting the first 15 sync events")
            plot_limit = 15
            plot_limit = min(15, len(photometry_sync_events))
            limited_index = photometry_sync_events.index[:plot_limit]
            limited_values = photometry_sync_events.values[:plot_limit]
    
            # Plot the first 15 aligned events
            limited_photometry_sync_aligned = photometry_sync_aligned[:plot_limit]
            limited_onix_digital_aligned = onix_digital_aligned[:plot_limit]
            limited_photometry_sync_state = onix_digital["PhotometrySyncState"][:plot_limit]
    
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    
            # First subplot (replacing the original ax1)
            ax1.set_xlabel("Harp Time (s)")
            ax1.set_ylabel("Event State", color="tab:blue")
            ax1.step(limited_photometry_sync_aligned, limited_values, where="mid", color="tab:blue", label="Photometry Events")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
    
            ax1_twin = ax1.twinx()
            ax1_twin.set_ylabel("Sync Line", color="tab:orange")
            ax1_twin.step(limited_onix_digital_aligned, limited_photometry_sync_state, where="mid", linestyle="--", color="tab:orange", label="Onix Digital Sync Line")
            ax1_twin.tick_params(axis="y", labelcolor="tab:orange")
    
            ax1.legend(loc="upper left")
            ax1_twin.legend(loc="upper right")
            ax1.set_title("Aligned Photometry Events and Onix Digital Sync Line to Harp Time")
    
            # Second subplot
            ax2.set_xlabel("Photometry Time(s)")
            ax2.set_ylabel("Onix Digital Clock", color="tab:orange")
            ax2.scatter(photometry_sync_events.index, onix_digital["Clock"], color="tab:orange", label="Onix Digital Clock")
            ax2.tick_params(axis="y", labelcolor="tab:orange")
            ax2.legend(loc="upper right")
            ax2.set_title("Photometry Events vs Onix Digital Clock")
    
            fig.tight_layout()
            plt.show()
        else:
            print(f"❗Something's wrong, short recording? Found only {len(photometry_sync_events)} sync events.")
    
    return output, photometry_aligned, photodiode_aligned

def get_global_minmax_timestamps(stream_dict, print_all=False, verbose=False):
    first_timestamps, last_timestamps = {}, {}

    # Check and fix all DataFrames in stream_dict for non-monotonic datetime indices
    for source_name, stream_source in stream_dict.items():
        for register, df in stream_source.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                print(f"❗❗❗ Warning: Index in {source_name}/{register} is not a DatetimeIndex. This shouldn't happen. Skipping.")
                continue  # Skip non-datetime indexed DataFrames
            
            # Detect non-consecutive datetime indices
            incorrect_rows = df.index.to_series().diff().lt(pd.Timedelta(0))
            if incorrect_rows.any():
                print(f"❗❗❗ Warning: Non-consecutive datetime index in {source_name}/{register} at time {df.index[incorrect_rows].tolist()}.")
                print("❗ If in video_data1 or 2, then deleting first row which should fix it.") 
                print("❗ Otherwise, we are in trouble and need to explore. See https://github.com/neurogears/vestibular-vr/issues/120") 
                
                # ✅ Fix: Drop the first row and explicitly replace the original DataFrame
                df_fixed = df.iloc[1:].copy()

                # ✅ Fix: Adjust the FrameIndex column by subtracting 1
                if 'FrameIndex' in df_fixed.columns:
                    df_fixed['FrameIndex'] -= 1
                
                # ✅ Explicitly update stream_dict
                stream_dict[source_name][register] = df_fixed

                # ✅ Confirm fix (for debugging)
                if verbose:
                    print(f"✅ Successfully corrected {source_name}/{register}: First timestamp is now {df_fixed.index[0]}.")

    # Extract first and last timestamps from corrected DataFrames
    for source_name, stream_source in stream_dict.items():
        first_timestamps[source_name] = {k: v.index[0] for k, v in stream_source.items()}
        last_timestamps[source_name] = {k: v.index[-1] for k, v in stream_source.items()}

    joint_first_timestamps, joint_last_timestamps = [], []
    first_dfs, last_dfs = {}, {}
    first_df_names, last_df_names = {}, {}

    for source_name, stream_source in first_timestamps.items():
        for register, timestamp in stream_source.items():
            joint_first_timestamps.append(timestamp)
            first_dfs[timestamp] = stream_dict[source_name][register]
            first_df_names[timestamp] = f"{source_name}/{register}"

    for source_name, stream_source in last_timestamps.items():
        for register, timestamp in stream_source.items():
            joint_last_timestamps.append(timestamp)
            last_dfs[timestamp] = stream_dict[source_name][register]
            last_df_names[timestamp] = f"{source_name}/{register}"

    joint_first_timestamps = pd.DataFrame(joint_first_timestamps)
    joint_last_timestamps = pd.DataFrame(joint_last_timestamps)

    # ✅ Ensure the true minimum is selected
    global_first_timestamp = min(joint_first_timestamps[0])  
    global_last_timestamp = max(joint_last_timestamps[0])  

    # ✅ Ensure correct mapping to the first and last timestamps
    correct_first_entry = {ts: df_name for ts, df_name in first_df_names.items() if ts == global_first_timestamp}
    global_first_df_name = correct_first_entry[global_first_timestamp]
    global_first_whichdf = first_dfs[global_first_timestamp]

    correct_last_entry = {ts: df_name for ts, df_name in last_df_names.items() if ts == global_last_timestamp}
    global_last_df_name = correct_last_entry[global_last_timestamp]
    global_last_whichdf = last_dfs[global_last_timestamp]

    if verbose:
        print(f'Global first timestamp: {global_first_timestamp}')
        print(f'Global first timestamp from: {global_first_df_name}')
        print(f'Global last timestamp: {global_last_timestamp}')
        print(f'Global last timestamp from: {global_last_df_name}')
        print(f'Global length: {global_last_timestamp - global_first_timestamp}')

    if print_all:
        for source_name in stream_dict.keys():
            print(f'\n{source_name}')
            for key in first_timestamps[source_name].keys():
                print(f'{key}: \n\tfirst  {first_timestamps[source_name][key]} \n\tlast   {last_timestamps[source_name][key]} \n\tlength {last_timestamps[source_name][key] - first_timestamps[source_name][key]} \n\tmean difference between timestamps {stream_dict[source_name][key].index.to_series().diff().mean()}')

    return global_first_timestamp, global_last_timestamp, stream_dict  # ✅ Return updated stream_dict

def pad_dataframe_with_global_timestamps(df, global_min_datetime, global_max_datetime):
    """
    Adds rows at global_min_datetime and global_max_datetime if they don't exist in the DataFrame.
    Preserves original data types but converts integer columns to nullable Int64 to handle NaN values.
    Boolean columns will be padded with False values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex to be padded
    global_min_datetime : pd.Timestamp
        Global minimum datetime to add if df starts later
    global_max_datetime : pd.Timestamp
        Global maximum datetime to add if df ends earlier
        
    Returns:
    --------
    pd.DataFrame
        Padded DataFrame with preserved data types
    """
    # First, convert all int64 columns to Int64 (nullable integer type)
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_integer_dtype(df_copy[col].dtype):
            df_copy[col] = df_copy[col].astype(pd.Int64Dtype())
    
    # Now we can add padding rows without type conversion issues
    rows_to_add = []
    
    # Function to get appropriate padding value based on column dtype
    def get_padding_value(col_name):
        if pd.api.types.is_bool_dtype(df_copy[col_name].dtype):
            if col_name == 'Photodiode':
                return True  # Use True for 'Photodiode' boolean column
            return False  # Use False for other boolean columns
        else:
            return pd.NA  # Use pd.NA for other columns
    
    # Check if the first index is greater than the global minimum datetime
    if df_copy.index[0] > global_min_datetime:
        # Create a new row with the global minimum datetime, with appropriate padding values
        rows_to_add.append(pd.DataFrame({col: [get_padding_value(col)] for col in df_copy.columns}, 
                                     index=[global_min_datetime]))
    
    # Check if the last index is less than the global maximum datetime
    if df_copy.index[-1] < global_max_datetime:
        # Create a new row with the global maximum datetime, with appropriate padding values
        rows_to_add.append(pd.DataFrame({col: [get_padding_value(col)] for col in df_copy.columns},
                                     index=[global_max_datetime]))
    
    # If we have rows to add, concatenate them with the original dataframe
    #FIXME DEBUG this as it gives future warning, suggestive that some columns are all NaNs which should not be the case
    if rows_to_add:
        df_copy = pd.concat([df_copy] + rows_to_add)
        df_copy = df_copy.sort_index()
    
    # Ensure boolean columns remain boolean
    for col in df_copy.columns:
        if pd.api.types.is_bool_dtype(df_copy[col].dtype):
            df_copy[col] = df_copy[col].astype(bool)
        
    return df_copy

def upsample_photometry(df, target_freq=1000):
    """
    Upsample photometry data to a target frequency while preserving high-frequency details.
    
    Parameters:
        df (pd.DataFrame): Original dataframe with non-uniform timestamps.
        target_freq (int): Target frequency in Hz (default 1000Hz).
        
    Returns:
        pd.DataFrame: Upsampled dataframe with uniform timestamps.
    """
    # Extract time and data columns
    time = df.index.values  # The index is the time
    data = df.values  # All other columns are signal data

    # Generate new uniform time vector
    new_time = np.arange(time[0], time[-1], 1 / target_freq)

    # Perform Akima interpolation (more robust for unevenly spaced data)
    interpolated_data = np.zeros((len(new_time), data.shape[1]))
    for i in range(data.shape[1]):
        interp_func = Akima1DInterpolator(time, data[:, i])
        interpolated_data[:, i] = interp_func(new_time)

    # Create new DataFrame with interpolated data and updated index
    new_df = pd.DataFrame(interpolated_data, columns=df.columns, index=new_time)

    return new_df

# Low-pass filter function 
def low_pass_filter(data, cutoff_freq, sample_rate, order=2):
    nyquist_freq = 0.5 * sample_rate
    if cutoff_freq >= nyquist_freq:
        raise ValueError(f"cutoff_freq must be less than the Nyquist frequency ({nyquist_freq} Hz)")
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Make index strictly monotonic
def make_index_monotonic(index, min_diff_threshold=pd.Timedelta('1us')):
    index_diff = index.to_series().diff().fillna(pd.Timedelta('0ns'))
    adjusted_index = index.to_series().copy()
    
    for i in range(1, len(adjusted_index)):
        if index_diff.iloc[i] < min_diff_threshold:
            adjusted_index.iloc[i] = adjusted_index.iloc[i-1] + min_diff_threshold

    return adjusted_index.astype(index.dtype)

# Circular interpolation for encoder values
def circular_interpolation(x, y, new_x):
    """ Interpolates circular data (angles in degrees, modulo 360). """
    interp_func = interp1d(x, np.unwrap(np.radians(y)), kind='linear', fill_value="extrapolate")
    interpolated = np.degrees(np.mod(interp_func(new_x), 2 * np.pi))  # Re-wrap after interpolation
    return interpolated

# Filter angular velocity, then reintegrate to get smooth angles
def filter_angular_velocity(angles, time, cutoff_freq, sample_rate):
    """ Filters the angular velocity, then reintegrates to get smooth angles. """
    velocity = np.gradient(np.unwrap(np.radians(angles)), time)  # Compute angular velocity
    velocity_filtered = low_pass_filter(velocity, cutoff_freq, sample_rate)  # Filter velocity
    smoothed_angles = np.degrees(np.cumsum(velocity_filtered) * np.mean(np.diff(time)))  # Reintegrate
    return smoothed_angles % 360  # Re-wrap angles

# Resampling function
def resample_column(column, new_index, method, optical_filter_Hz, sample_rate):
    logging.info(f"Resampling column: {column.name}")
    
    # Ensure the index is strictly monotonic
    column.index = make_index_monotonic(column.index)

    if "Encoder" in column.name:  # Special handling for encoder signals
        temp_col = column.ffill().bfill()  # Handle missing values
        interpolated_col = circular_interpolation(
            column.index.astype(np.int64) / 10**9, temp_col, new_index.astype(np.int64) / 10**9
        )
        # Apply velocity filtering & reintegrate
        smoothed_angles = filter_angular_velocity(interpolated_col, new_index.astype(np.int64) / 10**9, optical_filter_Hz, sample_rate)
        return smoothed_angles

    elif method == 'linear':
        temp_col = column.ffill().bfill()  # Ensure forward fill and back fill to avoid NaNs
        if temp_col.isnull().any():
            logging.warning(f"Column {column.name} still has NaNs after forward and backward fill.")
        interpolated_col = np.interp(new_index.astype(np.int64) / 10**9, column.index.astype(np.int64) / 10**9, temp_col)
        # Apply low-pass filter
        filtered_col = low_pass_filter(interpolated_col, optical_filter_Hz, sample_rate)
        return filtered_col

    elif method == 'fluor':
        return column.bfill().reindex(new_index, method='nearest')  # For fluorescence data

# Resampling function for entire DataFrame
def resample_dataframe(df, target_freq_Hz=1000, optical_filter_Hz=33):
    """
    Resample a DataFrame with:
    - Circular interpolation & velocity filtering for Encoder signals.
    - Linear interpolation for Optical & Position signals.
    - No interpolation for fluorescence signals (already upsampled).
    - Ensures strict monotonicity and prevents artifacts.
    """
    start_time = time.time()

    # Ensure the index is strictly increasing
    df.index = make_index_monotonic(df.index)

    logging.info(f"Index adjustment took {time.time() - start_time:.2f} seconds")

    # Create a regular time grid
    time_interval = f"{1/target_freq_Hz}s"
    new_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=time_interval)

    logging.info(f"Time grid creation took {time.time() - start_time:.2f} seconds")

    # Identify column types
    encoder_cols = [col for col in df.columns if "Encoder" in col]
    linear_cols = [col for col in df.columns if "Position" in col and col not in encoder_cols]
    fluorescence_cols = [col for col in df.columns if col not in encoder_cols + linear_cols]

    # Create an empty DataFrame
    resampled_df = pd.DataFrame(index=new_index)

    logging.info(f"Empty DataFrame creation took {time.time() - start_time:.2f} seconds")

    # Resample in parallel
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        encoder_futures = {executor.submit(resample_column, df[col], new_index, 'encoder', optical_filter_Hz, target_freq_Hz): col for col in encoder_cols}
        linear_futures = {executor.submit(resample_column, df[col], new_index, 'linear', optical_filter_Hz, target_freq_Hz): col for col in linear_cols}
        fluor_futures = {executor.submit(resample_column, df[col], new_index, 'fluor', optical_filter_Hz, target_freq_Hz): col for col in fluorescence_cols}

        for future in encoder_futures:
            col = encoder_futures[future]
            resampled_df[col] = future.result()

        for future in linear_futures:
            col = linear_futures[future]
            resampled_df[col] = future.result()

        for future in fluor_futures:
            col = fluor_futures[future]
            resampled_df[col] = future.result()

    logging.info(f"Resampling took {time.time() - start_time:.2f} seconds")

    return resampled_df

def plot_figure_1(df, session_name, save_path, common_resampled_rate, photodiode_halts, save_figure = False, show_figure = False, downsample_factor=20):
    """
    Plot specific time series data from the DataFrame in a browser window.
    Fixes missing X-axis and Y-axis movement plots by correctly assigning y-axes per subplot.
    """

    # Unwrap the position and encoder data to handle wrapping around 360 degrees
    position_unwrapped = np.unwrap(df['Position_0Y'].values * np.pi / 180) * 180 / np.pi
    encoder_unwrapped = np.unwrap(df['Encoder'].values * np.pi / 180) * 180 / np.pi

    # Compute cross-correlation efficiently
    cross_corr = correlate(
        position_unwrapped, 
        encoder_unwrapped, 
        mode='full', method='fft'
    )

    # Find negative peak
    negative_peak_index = np.argmin(cross_corr)
    negative_peak_position = negative_peak_index - len(position_unwrapped) + 1

    # Convert the position of the negative peak to microseconds
    negative_peak_position_ms = negative_peak_position * (1e3 / common_resampled_rate)

    # Downsample data for plotting efficiency
    df_downsampled = df.iloc[::downsample_factor].copy()
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, 
        vertical_spacing=0.1, subplot_titles=(
            'X Position (m)', 'Y position (degrees)', 
            f"Encoder Data (degrees, neg corr peak with posY: {negative_peak_position_ms:.2f} ms)", 'Photometry Data (z-score)'),
        specs=[[{"secondary_y": True}],  # Allow second y-axis for X movement
               [{"secondary_y": True}],  # Allow second y-axis for Y movement
               [{}], [{}]]  # Single y-axis for Encoder & Photometry
    )

    # Colors for the different metrics
    colors = {'Position': 'blue', 'Velocity': 'green', 'Acceleration': 'red'}

    # 1. Plot X-axis movement (Now with properly assigned y-axes)
    fig.add_trace(go.Scatter(x=df_downsampled.index, y=df_downsampled['Position_0X'], 
                             mode='lines', name='Position_0X', line=dict(color=colors['Position'])), 
                  row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=df_downsampled.index, y=df_downsampled['Velocity_0X'], 
                             mode='lines', name='Velocity_0X', line=dict(color=colors['Velocity'], width=1, dash='solid')), 
                  row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=df_downsampled.index, y=df_downsampled['Acceleration_0X'], 
                             mode='lines', name='Acceleration_0X', line=dict(color=colors['Acceleration'], width=1, dash='solid')), 
                  row=1, col=1, secondary_y=True)

    # 2. Plot Y-axis movement (Fixed missing data)
    fig.add_trace(go.Scatter(x=df_downsampled.index, y=df_downsampled['Position_0Y'], 
                             mode='lines', name='Position_0Y', line=dict(color=colors['Position'])), 
                  row=2, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=df_downsampled.index, y=df_downsampled['Velocity_0Y'], 
                             mode='lines', name='Velocity_0Y', line=dict(color=colors['Velocity'], width=1, dash='solid')), 
                  row=2, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(x=df_downsampled.index, y=df_downsampled['Acceleration_0Y'], 
                             mode='lines', name='Acceleration_0Y', line=dict(color=colors['Acceleration'], width=1, dash='solid')), 
                  row=2, col=1, secondary_y=True)

    # Adjust transparency by setting the opacity of the traces
    fig.update_traces(opacity=0.5, selector=dict(name='Velocity_0X'))
    fig.update_traces(opacity=0.5, selector=dict(name='Acceleration_0X'))
    fig.update_traces(opacity=0.5, selector=dict(name='Velocity_0Y'))
    fig.update_traces(opacity=0.5, selector=dict(name='Acceleration_0Y'))

    # 3. Plot Encoder data
    encoder_cols = [col for col in df.columns if 'Encoder' in col]
    for col in encoder_cols:
        if col in df_downsampled.columns:
            fig.add_trace(go.Scatter(x=df_downsampled.index, y=df_downsampled[col], 
                                     mode='lines', name=col), row=3, col=1)

    # 4. Plot Photometry data (z_470, z_560)
    photometry_cols = ['z_470', 'z_560']
    for col in photometry_cols:
        if col in df_downsampled.columns:
            fig.add_trace(go.Scatter(x=df_downsampled.index, y=df_downsampled[col], 
                                     mode='lines', name=col, line=dict(width=1)), row=4, col=1)

    max_z_470 = df['z_560'].max()
    first_edge = True
    for edge in photodiode_halts:
        if edge in df.index:
            fig.add_trace(go.Scatter(
                x=[edge], 
                y=[max_z_470],  # Use the max value of the z_470 signal
                mode='markers', 
                name='Photodiode Halt' if first_edge else '',  # Show legend once
                marker=dict(color='black', size=5, symbol='circle-open'),
                showlegend=first_edge  # Hide legend for subsequent traces
            ), row=4, col=1)
            first_edge = False  # Disable legend for subsequent traces

    # Update layout
    fig.update_layout(
        height=800, width=1000, 
        title_text=f'Position, Velocity, Acceleration, Encoder and Photometry Data - {session_name}'
    )
    if save_figure:
        fig.write_image(save_path / "figure1.png", scale=3)  # Adjust scale for higher resolution

    if show_figure:
        fig.show()

def safe_to_json(x): # for session_settings saving
    if isinstance(x, DotMap):
        return json.dumps(x.toDict())  # Convert DotMap to JSON string
    elif isinstance(x, dict):  
        return json.dumps(x)  # Convert plain dictionaries to JSON
    return x  # Leave strings unchanged

def safe_from_json(x): # for session_settings  loading 
    if isinstance(x, str):  # Only attempt to decode if it's a string
        return DotMap(json.loads(x))  
    return x  # If it's already a dictionary or DotMap, keep it as is

def analyze_photodiode(photodiode_aligned, experiment_events, event_name, plot=True):

    # Calculate the difference in the photodiode signal (no need for ['Photodiode'])
    photodiode_diff = photodiode_aligned.astype(int).diff()

    # Identify the falling and rising edges
    falling_edge_timestamps = photodiode_diff[photodiode_diff == -1].index
    rising_edge_timestamps = photodiode_diff[photodiode_diff == 1].index

    # Implement a refractory period of X seconds; #this is some old remnant code and I just made the refractory period super small, so it does not throuw out any edges
    valid_falling_edges = []
    refractory_period = pd.Timedelta(seconds=0.0005) 
    last_edge_time = falling_edge_timestamps[0] - refractory_period  # Initialize before first edge

    for edge_time in falling_edge_timestamps:
        if edge_time - last_edge_time >= refractory_period:
            valid_falling_edges.append(edge_time)
            last_edge_time = edge_time

    falling_edges_count = len(valid_falling_edges)
    halt_count = experiment_events[experiment_events["Event"] == event_name].shape[0]

    if falling_edges_count == halt_count:
        print(f"✅ {halt_count} events found. Matching number of photodiode falling edges and '{event_name}' events.")
    if falling_edges_count != halt_count:
        print(f"❗ WARNING: Falling edges ({falling_edges_count}) and {event_name} events ({halt_count}) do not match. Number of events: {falling_edges_count}.")
        print(f"ℹ️ This happens occasionally. The following section prints the extra falling edges and their durations and report of if they were removed.")
 
        # Step 1: Detect Falling and Rising Edges
        #already done above 

        # Step 2: Find Extra Falling Edges (No Corresponding Event)
        if "Time" in experiment_events.columns and not isinstance(experiment_events.index, pd.DatetimeIndex):
            experiment_events["Time"] = pd.to_datetime(experiment_events["Time"])
            experiment_events = experiment_events.set_index("Time")

        halt_timestamps = experiment_events.loc[experiment_events["Event"] == event_name].index
        if halt_timestamps.empty:
            print("❗ Warning: No halt events found. Skipping extra falling edge detection.")
            extra_falling_edges = falling_edge_timestamps  # Skip filtering
        else:
            extra_falling_edges = falling_edge_timestamps[~falling_edge_timestamps.isin(halt_timestamps)]

        # Step 3: Match Falling Edges to Next Rising Edge and Find Duration
        falling_rising_data = []

        for falling_edge in falling_edge_timestamps:
            next_rising_edges = rising_edge_timestamps[rising_edge_timestamps > falling_edge]
            if not next_rising_edges.empty:
                rising_edge = next_rising_edges.min()  # ✅ Select the nearest rising edge
                duration = (rising_edge - falling_edge).total_seconds()
            else:
                print(f"⚠️ No rising edge found after {falling_edge}. Setting duration to NaN.")
                rising_edge = None
                duration = None
      
            # Append data
            falling_rising_data.append({
                "Falling Edge Time": falling_edge,
                "Rising Edge Time": rising_edge,
                "Duration (s)": duration
            })

        # Convert list to DataFrame
        falling_rising_df = pd.DataFrame(falling_rising_data)

        # Step 4: Find Preceding Experiment Events for Each Falling Edge
        falling_rising_df["Preceding Event"] = None
        falling_rising_df["Event Time Difference (s)"] = None

        for i, row in falling_rising_df.iterrows():
            falling_edge_time = row["Falling Edge Time"]
            preceding_events = experiment_events[experiment_events.index < falling_edge_time]
            if not preceding_events.empty:
                last_event_time = preceding_events.index[-1]
                last_event_name = preceding_events.iloc[-1]["Event"]
                time_difference = (falling_edge_time - last_event_time).total_seconds()
                
                # Update DataFrame
                falling_rising_df.at[i, "Preceding Event"] = last_event_name
                falling_rising_df.at[i, "Event Time Difference (s)"] = time_difference

        # Step 5: Print Only Rows Where Preceding Event is Not event_name

        filtered_df = falling_rising_df[falling_rising_df["Preceding Event"] != event_name]
        print(filtered_df[["Falling Edge Time", "Duration (s)", "Preceding Event", "Event Time Difference (s)"]].to_string(index=False))

        # Step 6: Plot Only Traces Where Preceding Event is Not event_name
        filtered_falling_edges = filtered_df["Falling Edge Time"]

        # Define the number of traces per row
        traces_per_row = 5

        # Correct time window: -1s to +5s relative to the falling edge
        time_before = pd.Timedelta(seconds=1)
        time_after = pd.Timedelta(seconds=5)

        # Calculate the number of rows needed
        num_rows = -(-len(filtered_falling_edges) // traces_per_row)  # Equivalent to math.ceil

        # Create a subplot figure with multiple rows
        fig = make_subplots(rows=num_rows, cols=traces_per_row, shared_xaxes=True, shared_yaxes=True,
                            subplot_titles=[f"Falling Edge {i+1}" for i in range(len(filtered_falling_edges))])

        # Determine a global x-axis range (ensuring uniform scale)
        x_min = -0.2
        x_max = 0.5

        # Loop through the filtered falling edges and plot them in subplots
        for i, edge_time in enumerate(filtered_falling_edges):
            row = (i // traces_per_row) + 1
            col = (i % traces_per_row) + 1

            start_time = edge_time - time_before
            end_time = edge_time + time_after
            
            window_df = pd.DataFrame(photodiode_aligned.loc[start_time:end_time]).copy()

            # Convert timestamps to relative time (seconds from falling edge)
            window_df["Relative Time"] = (window_df.index - edge_time).total_seconds()

            # Add trace to subplot
            fig.add_trace(
                go.Scatter(x=window_df["Relative Time"], y=window_df["Photodiode_int"], 
                        mode="lines", line=dict(color="blue"), name=f"Trace {i+1}"),
                row=row, col=col
            )

            # Mark the falling edge (always at time 0)
            fig.add_trace(
                go.Scatter(x=[0], y=[0], mode="markers",
                        marker=dict(size=8, color="red"), name="Falling Edge"),
                row=row, col=col
            )

        # Update layout
        fig.update_layout(
            title="Photodiode Signal Traces Around Extra Falling Edges",
            height=num_rows * 250,
            showlegend=False,
            xaxis=dict(range=[x_min, x_max], title="Time (s) relative to Falling Edge"),
            yaxis=dict(title="Photodiode Signal"),
        )

        # Apply consistent x-axis range to all subplots
        for i in range(1, num_rows * traces_per_row + 1):
            fig.update_xaxes(range=[x_min, x_max], row=((i-1) // traces_per_row) + 1, col=(i-1) % traces_per_row + 1)

        # Show the figure
        fig.show()

    # Calculate the minimum, average, and maximum time differences
    time_differences = []
    halt_events = experiment_events[experiment_events["Event"] == event_name].index
    for event_time in halt_events:
        start_time = event_time - pd.Timedelta(seconds=0.5)
        end_time = event_time + pd.Timedelta(seconds=2.5)
        subset = photodiode_aligned[start_time:end_time]
        relative_time = (subset.index - event_time).total_seconds()

        # Determine the time difference between the event_time and the first 0 value following it
        zero_crossings = relative_time[subset == 0]
        if not zero_crossings.empty:
            time_difference = zero_crossings[0]
            time_differences.append(time_difference)

    if time_differences:
        min_diff = min(time_differences) * 1000  # Convert to milliseconds
        avg_diff = (sum(time_differences) / len(time_differences)) * 1000  # Convert to milliseconds
        max_diff = max(time_differences) * 1000  # Convert to milliseconds
    else:
        min_diff = avg_diff = max_diff = None

    # Print the minimum, average, and maximum time differences
    if time_differences:
        print(f"time difference between photodiode and experimenet events:")
        print(f"min {min_diff:.1f} ms. avg {avg_diff:.1f} ms. max {max_diff:.1f} ms.")
    else:
        print("No zero crossings found following the events.")

    if plot:
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))

        # Plot the falling edges and the events on the first subplot
        ax1.plot(photodiode_aligned.index, photodiode_aligned, label='Photodiode Signal')
        for edge_time in valid_falling_edges:
            ax1.axvline(x=edge_time, color='red', linestyle='--', linewidth=0.5, label='Falling Edge' if edge_time == valid_falling_edges[0] else '')
        for event_time in halt_events:
            nearest_time = photodiode_aligned.index.asof(event_time)
            ax1.plot(nearest_time, photodiode_aligned.loc[nearest_time], 'o', color='red', markersize=8, fillstyle='none')
        ax1.set_title('Photodiode Signal with Falling Edges and Halt Events')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Photodiode Signal')
        ax1.legend(loc='lower left')

        # Plot the triggered data on the second subplot
        for event_time in halt_events:
            start_time = event_time - pd.Timedelta(seconds=0.5)
            end_time = event_time + pd.Timedelta(seconds=2.5)
            subset = photodiode_aligned[start_time:end_time]
            relative_time = (subset.index - event_time).total_seconds()
            ax2.plot(relative_time, subset, label='Photodiode Signal' if event_time == halt_events[0] else '')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Event Trigger' if event_time == halt_events[0] else '')
        ax2.set_title(f'{event_name} Triggered Plot')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Photodiode Signal')
        ax2.legend(loc='lower left')

        plt.tight_layout()
        plt.show()

    # Ensure extra_falling_edges is always defined
    if 'extra_falling_edges' not in locals():
        extra_falling_edges = pd.DatetimeIndex([])  # Ensure it's a valid DatetimeIndex

    # Convert min_diff and max_diff from ms to Timedelta
    min_tolerance = pd.Timedelta(milliseconds=min_diff)
    max_tolerance = pd.Timedelta(milliseconds=max_diff)

    # Identify extra falling edges that are NOT within the expected tolerance window
    refined_extra_falling_edges = []
    for edge in valid_falling_edges:
        within_window = any((halt_time + min_tolerance) <= edge <= (halt_time + max_tolerance) for halt_time in halt_events)
        if not within_window:
            refined_extra_falling_edges.append(edge)

    refined_extra_falling_edges = pd.DatetimeIndex(refined_extra_falling_edges)

    # Remove refined extra falling edges from valid_falling_edges
    filtered_valid_falling_edges = [edge for edge in valid_falling_edges if edge not in refined_extra_falling_edges]
    removed_count = len(valid_falling_edges) - len(filtered_valid_falling_edges)

    print(f"✅ {removed_count} extra falling edges (outside {min_diff:.1f}ms - {max_diff:.1f}ms delay window) were removed before returning.")
  
    # Return filtered falling edges
    return filtered_valid_falling_edges, min_diff, avg_diff, max_diff

def check_exp_events(experiment_events, photometry_info, verbose = True):
    mouse_name = photometry_info.loc[photometry_info["Parameter"] == "mousename", "Value"].values[0]
    if verbose:
        print (f"ℹ️ Mousename: {mouse_name}")

    event_counts = experiment_events["Event"].value_counts()
    if verbose:
        print("ℹ️ Unique events and their counts:")
        print(event_counts)

    block_events = experiment_events[
        experiment_events["Event"].str.contains("block started|Block timer elapsed", case=False, na=False)
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning
    block_events["Time Difference"] = block_events.index.to_series().diff().dt.total_seconds()
    block_events = block_events.drop(columns=['Seconds'])

    if verbose:
        print("ℹ️ block events")
        print(block_events)


def compute_Lomb_Scargle_psd(data_df, freq_min=0.001, freq_max=10**6, num_freqs=1000, normalise=True):
    freqs = np.linspace(freq_min, freq_max, num_freqs)
    x = (data_df.index - data_df.index[0]).total_seconds().to_numpy()
    x = data_df.index
    y = data_df.values
    if y.ndim != 1: y = y[:,0]
    psd = signal.lombscargle(x, y, freqs, normalize=normalise)
    return freqs, psd

def convert_datetime_to_seconds(timestamp_input):
    if type(timestamp_input) == datetime or type(timestamp_input) == pd.DatetimeIndex:
        return (timestamp_input - utils.harp.REFERENCE_EPOCH).total_seconds()
    else:
        return timestamp_input.apply(lambda x: (x - utils.harp.REFERENCE_EPOCH).total_seconds())

def convert_seconds_to_datetime(seconds_input):
        return utils.harp.REFERENCE_EPOCH + timedelta(seconds=seconds_input)

def reformat_dataframe(input_df, name, index_column_name='Seconds', data_column_name='Data'):

    if input_df[index_column_name].values.dtype == np.dtype('<M8[ns]'):
        return pd.Series(data=input_df[data_column_name].values, 
                          index=input_df[index_column_name], 
                          name=name)
    else:
        return pd.Series(data=input_df[data_column_name].values, 
                            index=input_df[index_column_name].apply(convert_seconds_to_datetime), 
                            name=name)

def convert_arrays_to_dataframe(list_of_names, list_of_arrays):
    return pd.DataFrame({list_of_names[i]: list_of_arrays[i] for i in range(len(list_of_names))})

def convert_stream_from_datetime_to_seconds(stream):
    return pd.Series(data=stream.values, index=convert_datetime_to_seconds(stream.index))

def convert_all_streams_from_datetime_to_seconds(streams):
    for source_name in streams.keys():
        for stream_name in streams[source_name].keys():
            streams[source_name][stream_name] = convert_stream_from_datetime_to_seconds(streams[source_name][streams])
    return streams

def add_stream(streams, source_name, new_stream, new_stream_name):
    if not source_name in streams.keys():
        streams[source_name] = {}
        
    streams[source_name][new_stream_name] = new_stream
    
    return streams

def reformat_and_add_many_streams(streams, dataframe, source_name, stream_names, index_column_name='Seconds'):
    for stream_name in stream_names:
        new_stream = reformat_dataframe(dataframe, stream_name, index_column_name, data_column_name=stream_name)
        streams = add_stream(streams, source_name, new_stream, stream_name)
    return streams

def select_from_photodiode_data(onix_analog_clock, OnixAnalogData, hard_start_time, harp_end_time, conversions):

    start_time = time()

    start_onix_time = conversions['harp_to_onix_clock'](hard_start_time)
    end_onix_time = conversions['harp_to_onix_clock'](harp_end_time)
    indices = np.where(np.logical_and(onix_analog_clock >= start_onix_time, onix_analog_clock <= end_onix_time))

    x, y = conversions['onix_to_harp_timestamp'](onix_analog_clock[indices]), OnixAnalogData[indices]

    print(f'Selection of photodiode data finished in {time() - start_time:.2f} seconds.')

    return x, y

def save_streams_as_h5(data_path, resampled_streams, streams_to_save_pattern={'H1': ['OpticalTrackingRead0X(46)', 'OpticalTrackingRead0Y(46)'], 'H2': ['Encoder(38)'], 'Photometry': ['CH1-410', 'CH1-470', 'CH1-560'], 'SleapVideoData1': ['Ellipse.Diameter', 'Ellipse.Center.X', 'Ellipse.Center.Y'], 'SleapVideoData2': ['Ellipse.Diameter', 'Ellipse.Center.X', 'Ellipse.Center.Y'], 'ONIX': ['Photodiode']}):

    start_time = time()

    stream_data_to_be_saved = {}

    for source_name in streams_to_save_pattern.keys():
        if source_name in resampled_streams.keys():
            stream_data_to_be_saved[source_name] = {}
            for stream_name in streams_to_save_pattern[source_name]:
                if stream_name in resampled_streams[source_name].keys():
                    stream_data_to_be_saved[source_name][stream_name] = resampled_streams[source_name][stream_name]
                else:
                    print(f'{stream_name} was included in "streams_to_save_pattern", but cannot be found inside of {source_name} source of resampled streams.')
        else:
            print(f'{source_name} was included in "streams_to_save_pattern", but cannot be found inside of resampled streams.')
            
    common_index = convert_datetime_to_seconds(next(iter(stream_data_to_be_saved.values()))[next(iter(stream_data_to_be_saved[next(iter(stream_data_to_be_saved.keys()))]))].index)

    output_file = data_path/f'resampled_streams_{data_path.parts[-1]}.h5'

    # Open an HDF5 file to save data
    with h5py.File(output_file, 'w') as h5file:
        # Save the common index once
        h5file.create_dataset('HARP_timestamps', data=common_index.values)
        
        # Iterate over the dictionary and save each stream
        for source_name, stream_dict in stream_data_to_be_saved.items():
            # Create a group for each source
            source_group = h5file.create_group(source_name)
            
            for stream_name, stream_data in stream_dict.items():
                # Save each stream as a dataset within the source group
                source_group.create_dataset(stream_name, data=stream_data.values)

    print(f'Data saved as H5 file in {time() - start_time:.2f} seconds to {output_file}.')
        
def add_experiment_events(data_dict, events_dict, mouse_info):
    # Iterate over each mouse key in the dictionaries
    for mouse_key in data_dict:
        # Retrieve the main and event DataFrames
        main_df = data_dict[mouse_key]
        event_df = events_dict[mouse_key]

        # Ensure both indices are sorted
        main_df = main_df.sort_index()
        event_df = event_df.sort_index()

        # Perform a merge_asof on the index to add 'Value' as 'ExperimentEvents' with backward matching
        merged_df = pd.merge_asof(
            main_df,
            event_df[['Value']],  # Only select the 'Value' column from event_df
            left_index=True,
            right_index=True,
            direction='backward',
            tolerance=0  # Adjust tolerance for matching on the index
        )

        # Rename the 'Value' column to 'ExperimentEvents'
        if 'ExperimentEvents' in merged_df.columns:
            merged_df['ExperimentEvents'] = merged_df.pop('Value')  # Replace existing column with the new 'Value' column
            print(f'Pre-existing ExperimentEvents column was replaced with new for {mouse_key}')
        else:
            merged_df = merged_df.rename(columns={'Value': 'ExperimentEvents'})  # Add new column
            print(f'Added new ExperimentEvents for {mouse_key}')

        # Add metadata from event_df
        #merged_df['Experiment'] = event_df['experiment'].unique()[0]
        #merged_df['Session'] = event_df['session'].unique()[0]

        # Add mouse ID, sex, and brain area
        mouse_info_name = mouse_key[:4]
        merged_df['mouseID'] = mouse_info_name
        merged_df['sex'] = mouse_info[mouse_info_name]['sex']
        merged_df['area'] = mouse_info[mouse_info_name]['area']

        # Update the dictionary with the merged DataFrame
        data_dict[mouse_key] = merged_df

    return data_dict

def add_no_halt_column(data_dict, events_dict):
    # Iterate over each mouse in the dictionaries
    for mouse_key in data_dict:
        main_df = data_dict[mouse_key]  # Large DataFrame
        event_df = events_dict[mouse_key]  # Small DataFrame

        # Ensure the index of the event_df is named 'Seconds' and is numeric (milliseconds)
        event_df.index.name = 'Seconds'

        # Create a new column 'No_halt' in the main_df
        main_df['No_halt'] = False

        # Filter the 'No halt' events from event_df
        no_halt_events = event_df[event_df['Value'] == 'No halt']

        # Use pd.merge_asof to match the nearest milliseconds from main_df index to event_df index
        merged_df = pd.merge_asof(
            main_df,
            no_halt_events[['Value']],  # Only bring in the 'Value' column where 'No halt' appears
            left_index=True,  # main_df has time in its index
            right_index=True,  # no_halt_events has time in its index (both in ms)
            direction='backward',  # Choose closest event on or before the timestamp
            tolerance=0.00005  # Match down to 4 decimals
        )

        # Explicitly convert 'Value' to string and fill NaN with 'False'
        main_df['No_halt'] = (merged_df['Value'].astype(str).fillna('') == 'No halt')

        # Update the dictionary with the modified DataFrame
        data_dict[mouse_key] = main_df

        print('No_halt events added to', mouse_key)

        # Verification
        event_len = len(events_dict[mouse_key].loc[events_dict[mouse_key].Value == 'No halt'])
        data_len = len(data_dict[mouse_key].loc[data_dict[mouse_key].No_halt == True])
        if event_len != data_len:
            print(f'For {mouse_key}, the number of actual no-halt events is {event_len} and the number of True values in the data now is {data_len}')
        
        if event_len == data_len:
            print(f'  Correct number of no-halt events for {mouse_key}\n')

    return data_dict

def add_block_columns(df, event_df):
    # Iterate through each index and event value in event_df
    prev_column = None  # Tracks the column currently being filled as True
    for idx, event in event_df['Value'].items():
        if 'block started' in event:
            print(event)
            # Create a new column in df, filling with False initially
            column_name = event.split()[0]+'_block'
            df[column_name] = False

            # If there was a previous column being filled as True, set it to False up to this point
            if prev_column is not None:
                df.loc[:idx, prev_column] = False

            # Set the new column to True starting from this index
            df.loc[idx:, column_name] = True
            prev_column = column_name  # Track the events

        elif 'Block timer elapsed' in event:
    
            # If there's a current active block, set its values to False up to this point
            if prev_column is not None:
                df.loc[idx:, prev_column] = False

                prev_column = None  # Reset current column tracker

    # Ensure that any remaining True blocks are set to False after their end
    #if current_column is not None:
     #   df.loc[:, current_column] = False
    for col in df:
        if 'block started' in col:
            df.rename({col: f'{col.split()[0]}_block'}, inplace = True)
    
    return df

def check_block_overlap(data_dict):
    for mouse, df in data_dict.items():
        # Choose columns that end with _block
        block_columns = df.filter(regex='_block')
        # Check if any row has more than one `True` at the same time in the `_block` columns
        no_overlap = (block_columns.sum(axis=1) <= 1).all()
        # Check if each `_block` column has at least one `True` value
        all_columns_true = block_columns.any().all()
        if no_overlap and all_columns_true:
            print(f'For {mouse}: No overlapping True values, and each _block column has at least one True value')
        elif no_overlap and not all_columns_true:
            print(f'Not all block columns contains True Values for {mouse}')
        elif not no_overlap and all_columns_true:
            print(f'There are some overlap between the blocks {mouse}')

def downsample_data(df, time_col='Seconds', interval=0.001):
    '''
    Uses pandas resample and aggregate functions to downsample the data to the desired interval. 
    * Note: Aggregation functions must be applied for each variable that is to be included.
    https://pandas.pydata.org/docs/reference/api/pandas.core.resample.Resampler.aggregate.html
    * Note: because the donwsampling keeps the first non-NaN value in each interval, some values could be lost.
    '''
    # Convert the Seconds column to a TimedeltaIndex
    df = df.set_index(pd.to_timedelta(df[time_col], unit='s'))

    #define aggregation functions for all possible columns
    aggregation_functions = {
        '470_dfF': 'mean', # takes the mean signal of the datapoints going into each new downsampled datapoint
        '560_dfF': 'mean',
        'movementX': 'mean',
        'movementY': 'mean',
        'event': 'any', # events column is a bool, and if there is any True values in the interval, the downsampled datapoint will be True
        'ExperimentEvents': lambda x: x.dropna().iloc[0] if not x.dropna().empty else None, #first non-NaN value in the interval 
        'Experiment': 'first', # All values should be the same, so it can always just take the first string value
        'Session': 'first',
        'mouseID': 'first',
        'sex': 'first',
        'area': 'first',
        'No_halt': 'any', 
        'LinearMismatch_block': 'any', 
        'LinearPlaybackMismatch_block': 'any',
        'LinearRegular_block': 'any',
        'LinearClosedloopMismatch_block':'any',
        'LinearRegularMismatch_block':'any',
        'LinearNormal_block':'any',
    }

    # Filter aggregation_functions to only include columns present in df
    aggregation_functions = {key: func for key, func in aggregation_functions.items() if key in df.columns}

    print('downsampling...')
    # Resample with the specified interval and apply the filtered aggregations
    downsampled_df = df.resample(f'{interval}s').agg(aggregation_functions)

    # Reset the index to make the Seconds column normal again
    downsampled_df = downsampled_df.reset_index()
    downsampled_df[time_col] = downsampled_df[time_col].dt.total_seconds()  # Convert Timedelta back to seconds

    # Forward fill for categorical columns if needed, only if they exist in downsampled_df
    categorical_cols = ['Experiment', 'Session', 'mouseID', 'sex', 'area']
    for col in categorical_cols:
        if col in downsampled_df.columns:
            downsampled_df[col] = downsampled_df[col].ffill()

    # Remove consecutive duplicate values in the 'ExperimentEvents' column, if it exists
    if 'ExperimentEvents' in downsampled_df.columns:
        downsampled_df['ExperimentEvents'] = downsampled_df['ExperimentEvents'].where(
            downsampled_df['ExperimentEvents'] != downsampled_df['ExperimentEvents'].shift()
        )

    return downsampled_df

def test_event_numbers(downsampled_data, original_data, mouse):
    '''
    Counts number of True values in the No_halt columns in the original and the downsampled data
    This will indicate whether information was lost in the downsampling.
    If the original events somehow has been upsampled previously (for example if the tolerance was set too high in add_experiment_events()), 
    repeatings of the same event can also lead to fewer True events in the downsampled df.
    '''
    nohalt_down = len(downsampled_data.loc[downsampled_data['No_halt']==True])
    nohalt_original = len(original_data.loc[original_data['No_halt']==True])
    if nohalt_down != nohalt_original:
        print(f'mouse{mouse}')
        print(f'There are actually {nohalt_original} no-halts, but the downsampled data only contains {nohalt_down}')
        print('Should re-run the downsampling. Try changing interval lenght. Othewise, consider not downsampling\n')
    if nohalt_down == nohalt_original:
        print(f'mouse{mouse}')
        print(f'There are {nohalt_original} no-halts, and downsampled data contains {nohalt_down}\n')

def load_h5_streams_to_dict(data_paths):
    '''
    Takes list of H5 file paths and, loads streams into dictionary, and save to dictionary named by mouse ID
    '''
    #dict to save streams:
    reconstructed_dict = {} 
    # File path to read the HDF5 file
    for input_file in data_paths:
        name = input_file.split('/')[-1][-7:-3] # Given that file name is of format: resampled_streams_2024-08-22T13-13-15_B3M6.h5 
        
        if not os.path.exists(input_file):
            print(f'ERROR: {input_file} does not exist.')
            return None
    
        # Open the HDF5 file to read data
        with h5py.File(input_file, 'r') as h5file:
            #print(f'reconstructing streams for mouse {input_file.split('/')[-1][-7:-3]}, from session folder: {input_file.split('/')[-3]}')
            mouse_id_x = input_file.split("/")[-1][-7:-3]
            session_folder_x = input_file.split("/")[-3]
            print(f"reconstructing streams for mouse {mouse_id_x}, from session folder: {session_folder_x}")
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
                
        reconstructed_dict[name] = reconstructed_streams
        print(f'  --> {mouse_id_x} streams reconstructed and added to dictionary \n')
            

    return reconstructed_dict

def moving_average_smoothing(X,k): # X is input np array, k is the window size
    S = np.zeros(X.shape[0])
    for t in range(X.shape[0]):
        if t < k:
            S[t] = np.mean(X[:t+1])
        else:
            S[t] = np.sum(X[t-k:t])/k
    return S

def plot_detail(data_stream_df, dataset_name, register, sample_num_to_plot=25):
    
    resampled_data_stream_df = resample_stream(data_stream_df)
    
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,15))
    
    fig.suptitle(f'DATASET [{dataset_name}] REGISTER [{register}]')
    
    ax[0][0].plot(data_stream_df, alpha=0.75)
    ax[0][0].plot(resampled_data_stream_df, alpha=0.75)
    ax[0][0].set_title('Full signal')
    ax[0][0].set_xlabel('Timestamp')
    ax[0][0].set_ylabel('Signal Magnitude')
    ax[0][0].set_xticklabels(ax[0][0].get_xticklabels(), rotation=-45)
    ax[0][0].grid()
    
    ax[1][0].plot(data_stream_df[:sample_num_to_plot], alpha=0.75)
    ax[1][0].scatter(data_stream_df[:sample_num_to_plot].index, data_stream_df[:sample_num_to_plot], s=25)
    filtered_resampled_df = resampled_data_stream_df[resampled_data_stream_df.index < data_stream_df.index[sample_num_to_plot]]
    ax[1][0].plot(filtered_resampled_df, alpha=0.75)
    ax[1][0].scatter(filtered_resampled_df.index, filtered_resampled_df, s=25, alpha=0.25)
    ax[1][0].set_xlabel('Timestamp')
    ax[1][0].set_ylabel('Signal Magnitude')
    ax[1][0].set_title(f'Zoom into first {sample_num_to_plot} timepoints')
    ax[1][0].set_xticks(data_stream_df[:sample_num_to_plot].index)
    ax[1][0].set_xticklabels(data_stream_df[:sample_num_to_plot].index.strftime('%H:%M:%S.%f'), rotation=-90)
    ax[1][0].grid()
    print('First five original timestamps:')
    for ts in data_stream_df[:5].index.to_list(): print(ts)
    print('\nFirst five resampled timestamps:')
    for ts in resampled_data_stream_df[:5].index.to_list(): print(ts)
    
    ax[1][1].plot(data_stream_df[-sample_num_to_plot:], alpha=0.75)
    ax[1][1].scatter(data_stream_df[-sample_num_to_plot:].index, data_stream_df[-sample_num_to_plot:], s=25)
    filtered_resampled_df = resampled_data_stream_df[resampled_data_stream_df.index >= data_stream_df.index[-sample_num_to_plot]]
    ax[1][1].plot(filtered_resampled_df, alpha=0.75)
    ax[1][1].scatter(filtered_resampled_df.index, filtered_resampled_df, s=25, alpha=0.25)
    ax[1][1].set_xlabel('Timestamp')
    ax[1][1].set_ylabel('Signal Magnitude')
    ax[1][1].set_title(f'Zoom into last {sample_num_to_plot} timepoints')
    ax[1][1].set_xticks(data_stream_df[-sample_num_to_plot:].index)
    ax[1][1].set_xticklabels(data_stream_df[-sample_num_to_plot:].index.strftime('%H:%M:%S.%f'), rotation=-90)
    ax[1][1].grid()
    
    inter_timestamp_invervals = np.diff(data_stream_df.index).astype(np.uint32) * (10**-9) # converted to seconds
    ax[2][0].hist(inter_timestamp_invervals, bins=50)
    ax[2][0].set_title('Histogram of intervals between timestamps')
    ax[2][0].set_xlabel('Inter-timestamp interval (seconds)')
    ax[2][0].set_ylabel('Count')
    ax[2][0].set_xticklabels(ax[2][0].get_xticklabels(), rotation=-45)
    ax[2][0].grid()
    
    plt.tight_layout()
    plt.show()

def plot_dataset(dataset_path):
    registers = utils.load_registers(dataset_path)
    h1_data_streams, h2_data_streams = registers['H1'], registers['H2']
    for register, register_stream in h1_data_streams.items():
        plot_detail(register_stream, dataset_path.name, register=str(register))
    for register, register_stream in h2_data_streams.items():
        plot_detail(register_stream, dataset_path.name, register=str(register))

# def read_ExperimentEvents(path):
#     filenames = os.listdir(path/'ExperimentEvents')
#     filenames = [x for x in filenames if x[:16]=='ExperimentEvents'] # filter out other (hidden) files
#     date_strings = [x.split('_')[1].split('.')[0] for x in filenames] 
#     sorted_filenames = pd.to_datetime(date_strings, format='%Y-%m-%dT%H-%M-%S').sort_values()
#     read_dfs = []
#     try:
#         for row in sorted_filenames:
#             read_dfs.append(pd.read_csv(path/'ExperimentEvents'/f"ExperimentEvents_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"))
#         return pd.concat(read_dfs).reset_index().drop(columns='index')
#     except pd.errors.ParserError as e:
#         filename = f"ExperimentEvents_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"
#         print(f'Tokenisation failed for file "{filename}".\n')
#         print(f'Exact description of error: {e}')
#         print('Likely due to extra commas in the "Value" column of ExperimentEvents. Please manually remove and run again.')
#         return None
#     except Exception as e:
#         print('Reading failed:', e)
#         return None

# def resample_stream(data_stream_df, resampling_period='0.1ms', method='linear'):
#     return data_stream_df.resample(resampling_period).last().interpolate(method=method)

# def resample_index(index, freq):
#     """Resamples each day in the daily `index` to the specified `freq`.

#     Parameters
#     ----------
#     index : pd.DatetimeIndex
#         The daily-frequency index to resample
#     freq : str
#         A pandas frequency string which should be higher than daily

#     Returns
#     -------
#     pd.DatetimeIndex
#         The resampled index

#     """
#     assert isinstance(index, pd.DatetimeIndex)
#     start_date = index.min()
#     end_date = index.max() + pd.DateOffset(days=1)
#     resampled_index = pd.date_range(start_date, end_date, freq=freq)[:-1]
#     series = pd.Series(resampled_index, resampled_index.floor('D'))
#     return pd.DatetimeIndex(series.loc[index].values)

#  def align_fluorescence_first_approach(fluorescence_df, onixdigital_df): #DEPRECATED, FIXME could be useful for Cohort0?
#     # Aligns Fluorescence signal using the HARP timestamps from onix_digital and interpolation
#     # Steps:
#     # - Selecting the rows where there are photometry synchronisation events occurring
#     # - Getting the values from 'Seconds' column of onix_digital and setting them to Fluorescence dataframe
#     # - Estimating the very first and the very last 'Seconds' value based on timestamps of the photometry software ('TimeStamp' column)
#     # - Applying default Pandas interpolation
#     # - based on https://github.com/neurogears/vestibular-vr/issues/76
#     # - FIXME relays on serious assumptions, do not use (or maybe for Cohort 0)?
    
#     start_time = time()

#     fluorescence_df = copy.deepcopy(fluorescence_df)
    
#     # Adding a new column
#     fluorescence_df['Seconds'] = np.nan
    
#     # Setting the rows of Seconds column where there are events with HARP timestamp values from onix_digital
#     fluorescence_df.loc[fluorescence_df['Events'].notna(), 'Seconds'] = onixdigital_df['Seconds'].values
    
#     # estimate the very first and very last values of Seconds column in Fluorescence to be able to interpolate between
#     first_val_to_insert = fluorescence_df[fluorescence_df['Events'].notna()].iloc[0]['Seconds'] - fluorescence_df[fluorescence_df['Events'].notna()].iloc[0]['TimeStamp'] / 1000
#     # first_val_to_insert = Seconds value of the first Event to occur - seconds elapsed since start of recording (converted from ms)
#     last_val_to_insert = fluorescence_df[fluorescence_df['Events'].notna()].iloc[-1]['Seconds'] + (fluorescence_df.iloc[-1]['TimeStamp'] / 1000 - fluorescence_df[fluorescence_df['Events'].notna()].iloc[-1]['TimeStamp'] / 1000)
#     # last_val_to_insert = Seconds value of the last Event to occur + seconds elapsed between the last row of Fluorescence and the last event to occur
    
#     fluorescence_df.loc[0, 'Seconds'] = first_val_to_insert
#     fluorescence_df.loc[-1, 'Seconds'] = last_val_to_insert
    
#     fluorescence_df[['Seconds']] = fluorescence_df[['Seconds']].interpolate()

#     print(f'Fluorescence alignment finished in {time() - start_time:.2f} seconds.')
    
#     return fluorescence_df

# def calculate_photodiode_falling_edges(df, experiment_events, event_name):
#     # Calculate the difference in the photodiode signal
#     photodiode_diff = df['Photodiode'].astype(int).diff()

#     # Identify the falling edges (True to False transitions)
#     falling_edges = photodiode_diff[photodiode_diff == -1].index.to_numpy()

#     # Apply a refractory period of 0.5s using numpy
#     if falling_edges.size > 0:
#         refractory_period = pd.Timedelta(seconds=0.5)
#         valid_falling_edges = falling_edges[np.insert(np.diff(falling_edges) >= refractory_period, 0, True)]

#         # Remove last falling edge if necessary
#         if len(valid_falling_edges) > 0:
#             valid_falling_edges = valid_falling_edges[:-1]
#     else:
#         valid_falling_edges = np.array([])

#     # Count events
#     falling_edges_count = len(valid_falling_edges)
#     halt_count = (experiment_events["Event"] == event_name).sum()
#     if falling_edges_count == halt_count:
#         print(f"✅ Matching number of photodiode falling edges and ''{event_name}'' events: {halt_count}")
#     if falling_edges_count != halt_count:
#         print(f"❗ Warning: Falling edges ({falling_edges_count}) and {event_name} events ({halt_count}) do not match. Number of events: {falling_edges_count}. Is the event type the right event?")

#     return valid_falling_edges