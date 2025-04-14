import os
import copy
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from datetime import datetime, timedelta
from scipy.signal import correlate
from scipy import signal  # for Lomb-Scargle PSD
import re # for photometry alignment

from . import utils
import aeon.io.api as api

def resample_stream(data_stream_df, resampling_period='0.1ms', method='linear'):
    return data_stream_df.resample(resampling_period).last().interpolate(method=method)

def resample_index(index, freq):
    """Resamples each day in the daily `index` to the specified `freq`.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The daily-frequency index to resample
    freq : str
        A pandas frequency string which should be higher than daily

    Returns
    -------
    pd.DatetimeIndex
        The resampled index

    """
    assert isinstance(index, pd.DatetimeIndex)
    start_date = index.min()
    end_date = index.max() + pd.DateOffset(days=1)
    resampled_index = pd.date_range(start_date, end_date, freq=freq)[:-1]
    series = pd.Series(resampled_index, resampled_index.floor('D'))
    return pd.DatetimeIndex(series.loc[index].values)

def get_timepoint_info(registers_dict, print_all=False):
    
    # Finding the very first and very last timestamp across all streams
    first_timestamps, last_timestamps = {}, {}
    for source_name, register_source in registers_dict.items():
        first_timestamps[source_name] = {k:v.index[0] for k,v in register_source.items()}
        last_timestamps[source_name] = {k:v.index[-1] for k,v in register_source.items()}
    
    # Saving global first and last timestamps across the sources and the registers
    joint_first_timestamps, joint_last_timestamps = [], []
    
    for register_source in first_timestamps.values():
        for register in register_source.values():
            joint_first_timestamps = joint_first_timestamps + [register]
    joint_first_timestamps = pd.DataFrame(joint_first_timestamps)
    
    for register_source in last_timestamps.values():
        for register in register_source.values():
            joint_last_timestamps = joint_last_timestamps + [register]
    joint_last_timestamps = pd.DataFrame(joint_last_timestamps)
    
    global_first_timestamp = joint_first_timestamps.iloc[joint_first_timestamps[0].argmin()][0]
    global_last_timestamp = joint_last_timestamps.iloc[joint_last_timestamps[0].argmax()][0]
    
    if print_all:
        print(f'Global first timestamp: {global_first_timestamp}')
        print(f'Global last timestamp: {global_last_timestamp}')
        print(f'Global length: {global_last_timestamp - global_first_timestamp}')
        
        for source_name in registers_dict.keys():
            print(f'\n{source_name}')
            for key in first_timestamps[source_name].keys():
                print(f'{key}: \n\tfirst  {first_timestamps[source_name][key]} \n\tlast   {last_timestamps[source_name][key]} \n\tlength {last_timestamps[source_name][key] - first_timestamps[source_name][key]} \n\tmean difference between timestamps {registers_dict[source_name][key].index.diff().mean()}')
    
    return global_first_timestamp, global_last_timestamp, first_timestamps, last_timestamps

def pad_and_resample(streams_dict, resampling_period='0.1ms', method='linear'):

    start_time = time()
    
    streams_dict = copy.deepcopy(streams_dict)
    
    # Padding: Getting the global first/last timepoints, adding them to every stream that starts/ends later/earler
    # Resampling + linear interpolation between points
    
    first_timestamp, last_timestamp, _, _ = get_timepoint_info(streams_dict)
    
    for source_name, source_element in streams_dict.items():
        for stream_name, stream in source_element.items():
            if stream.dtype==bool:
                dummy_value = 1
            else:
                dummy_value = 0
            # Check if global first and last timestamps already exist in a given stream
            if stream.index[0] != first_timestamp:
                # Create new element with the earliest timestamp
                new_start = pd.Series([dummy_value], index=[first_timestamp]).astype(stream.dtype)
                # Append the new element to the Series
                stream = pd.concat([new_start, stream])
                stream = stream.sort_index()
            if stream.index[-1] != last_timestamp:
                # Create new element with the latest timestamp
                new_end = pd.Series([dummy_value], index=[last_timestamp]).astype(stream.dtype)
                # Append the new element to the Series
                stream = pd.concat([stream, new_end])
                stream = stream.sort_index()

            # Resampling and interpolation
            if stream.dtype==bool:
                streams_dict[source_name][stream_name] = resample_stream(stream, resampling_period=resampling_period, method='nearest').astype(bool)
            else:
                streams_dict[source_name][stream_name] = resample_stream(stream, resampling_period=resampling_period, method=method)
    
    print(f'Padding and resampling finished in {time() - start_time:.2f} seconds.')

    return streams_dict

def compute_Lomb_Scargle_psd(data_df, freq_min=0.001, freq_max=10**6, num_freqs=1000, normalise=True):
    freqs = np.linspace(freq_min, freq_max, num_freqs)
#     x = (data_df.index - data_df.index[0]).total_seconds().to_numpy()
    x = data_df.index
    y = data_df.values
    if y.ndim != 1: y = y[:,0]
    psd = signal.lombscargle(x, y, freqs, normalize=normalise)
    return freqs, psd

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

def align_fluorescence_first_approach(fluorescence_df, onixdigital_df): #DEPRECATED, could be useful for Cohort0?
    # Aligns Fluorescence signal using the HARP timestamps from onix_digital and interpolation
    # Steps:
    # - Selecting the rows where there are photometry synchronisation events occurring
    # - Getting the values from 'Seconds' column of onix_digital and setting them to Fluorescence dataframe
    # - Estimating the very first and the very last 'Seconds' value based on timestamps of the photometry software ('TimeStamp' column)
    # - Applying default Pandas interpolation
    # - based on https://github.com/neurogears/vestibular-vr/issues/76
    # - FIXME relays on serious assumptions, do not use (or maybe for Cohort 0)?
    
    start_time = time()

    fluorescence_df = copy.deepcopy(fluorescence_df)
    
    # Adding a new column
    fluorescence_df['Seconds'] = np.nan
    
    # Setting the rows of Seconds column where there are events with HARP timestamp values from onix_digital
    fluorescence_df.loc[fluorescence_df['Events'].notna(), 'Seconds'] = onixdigital_df['Seconds'].values
    
    # estimate the very first and very last values of Seconds column in Fluorescence to be able to interpolate between
    first_val_to_insert = fluorescence_df[fluorescence_df['Events'].notna()].iloc[0]['Seconds'] - fluorescence_df[fluorescence_df['Events'].notna()].iloc[0]['TimeStamp'] / 1000
    # first_val_to_insert = Seconds value of the first Event to occur - seconds elapsed since start of recording (converted from ms)
    last_val_to_insert = fluorescence_df[fluorescence_df['Events'].notna()].iloc[-1]['Seconds'] + (fluorescence_df.iloc[-1]['TimeStamp'] / 1000 - fluorescence_df[fluorescence_df['Events'].notna()].iloc[-1]['TimeStamp'] / 1000)
    # last_val_to_insert = Seconds value of the last Event to occur + seconds elapsed between the last row of Fluorescence and the last event to occur
    
    fluorescence_df.loc[0, 'Seconds'] = first_val_to_insert
    fluorescence_df.loc[-1, 'Seconds'] = last_val_to_insert
    
    fluorescence_df[['Seconds']] = fluorescence_df[['Seconds']].interpolate()

    print(f'Fluorescence alignment finished in {time() - start_time:.2f} seconds.')
    
    return fluorescence_df

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


def photometry_harp_onix_synchronisation(
    photodiode,
    onix_analog_clock, 
    onix_analog_framecount, 
    onix_digital, 
    onix_harp,
    photometry_events, 
    verbose=True
):
    start_time = time()
    output = {}

    # Efficient upsampling of framecount
    upsample_factor = 100
    df = onix_analog_framecount
    new_index = np.linspace(0, len(df) - 1, len(df) * upsample_factor)
    upsampled_framecount = pd.DataFrame(index=new_index)
    for col in df.columns:
        upsampled_framecount[col] = np.interp(new_index, np.arange(len(df)), df[col])

    # Check onix_analog shapes for consistency
    data_len = photodiode.shape[0]
    clock_len = onix_analog_clock.shape[0]
    framecount_len = len(upsampled_framecount)
    
    if data_len != framecount_len or clock_len != framecount_len:
        offset = framecount_len - clock_len
        upsampled_framecount = upsampled_framecount.iloc[offset:]
        print(f"Warning: analog_data and _framecount mismatch by {offset}! Should be OK, but see https://github.com/neurogears/vestibular-vr/issues/81 for more information.")
    else:
        if verbose:
            print("onix_analog shapes are consistent!")

    # Find the time mapping/warping between onix and harp clock
    clock = onix_harp["Clock"]
    harp = onix_harp["HarpTime"]
    o_m, o_b = np.polyfit(clock, harp, 1)
    onix_to_harp_seconds = lambda x: x * o_m + o_b
    onix_to_harp_timestamp = lambda x: api.aeon(onix_to_harp_seconds(x))
    harp_to_onix_clock = lambda x: (x - o_b) / o_m
    
    # Calculate R-squared value
    y_pred = onix_to_harp_seconds(clock)
    ss_res = np.sum((harp - y_pred) ** 2)
    ss_tot = np.sum((harp - np.mean(harp)) ** 2)
    r_squared_harp_to_onix = 1 - (ss_res / ss_tot)

    output["onix_to_harp_timestamp"] = onix_to_harp_timestamp
    output["harp_to_onix_clock"] = harp_to_onix_clock
    output["r_squared_harp_to_onix"] = r_squared_harp_to_onix

    if r_squared_harp_to_onix < 0.999:
        print(f"Warning: R-squared value between Clock and HarpTime in onix_harp is {r_squared_harp_to_onix:.6f}. Could be an issue with the dataset.")
    
    # Drop rows with NaN values
    nan_count = photometry_events.isna().sum().sum()
    if verbose and nan_count > 0:
        print(f"WARNING, {nan_count} NaNs detected, check Events'csv")
    photometry_sync_events = photometry_events.dropna()
    photometry_sync_events["TimeStamp"] = photometry_sync_events["TimeStamp"] / 1000  # Convert to seconds
    photometry_sync_events.set_index("TimeStamp", inplace=True)
    photometry_sync_events = photometry_sync_events["State"]

    # Truncate photometry to onix_digital
    min_length = min(len(photometry_sync_events), len(onix_digital["Clock"]))
    photometry_sync_events = photometry_sync_events.iloc[:min_length]
    num_truncated = len(photometry_sync_events) - min_length
    if num_truncated > 1:
        print(f"WARNING Photometry_sync_events was truncated by {num_truncated} points, more than the expected 1.")
    
    onix_digital["sync_line"] = 1 - (onix_digital["PhotometrySyncState"] & 1)  # Flip sync line
    
    # Plotting
    if verbose:
        print(f"{len(photometry_sync_events)} events found")
        if not photometry_sync_events.empty:
            if len(photometry_sync_events) > 15:
                print (" Many events detected, plotting the first 15")    
            plot_limit = 15
            plot_limit = min(15, len(photometry_sync_events))
            limited_index = photometry_sync_events.index[:plot_limit]
            limited_values = photometry_sync_events.values[:plot_limit]
            
            plt.figure()
            plt.step(limited_index, limited_values)
            plt.xlabel("Time (s)")
            plt.ylabel("Event State")
            plt.title("Photometry Events")
            plt.show()
        else:
            print(f"Not enough events to plot: found {len(photometry_sync_events)} events.")
        
        plt.figure()
        plt.step(onix_digital["Clock"][:plot_limit], onix_digital["sync_line"][:plot_limit])
        plt.xlabel("Clock")
        plt.ylabel("Sync Line")
        plt.title("Onix Digital Sync Line")
        plt.show()

        plt.figure()
        plt.scatter(photometry_sync_events.index, onix_digital["Clock"]) #plot whole range to see if there are outliers 
        plt.xlabel("Photometry Time(s)")
        plt.ylabel("Onix Digital Clock")
        plt.title("Photometry Events vs Onix Digital Clock")
        plt.show()
    
    # Define conversion functions between timestamps (onix to harp)
    m, b = np.polyfit(photometry_sync_events.index, onix_digital["Clock"], 1)
    photometry_to_onix_time = lambda x: x * m + b
    photometry_to_harp_time = lambda x: onix_to_harp_timestamp(photometry_to_onix_time(x))
    onix_time_to_photometry = lambda x: (x - b) / m
    
    # Calculate R-squared value
    y_pred = photometry_to_onix_time(photometry_sync_events.index)
    ss_res = np.sum((onix_digital["Clock"] - y_pred) ** 2)
    ss_tot = np.sum((onix_digital["Clock"] - np.mean(onix_digital["Clock"])) ** 2)
    r_squared_photometry_to_onix = 1 - (ss_res / ss_tot)
    output["r_squared_photometry_to_onix"] = r_squared_photometry_to_onix
    
    if r_squared_photometry_to_onix < 0.999:
        print(f"Warning: R-squared value between photometry and onix_digital is {r_squared_photometry_to_onix:.6f}. Could be an issue with the dataset.")
    
    if verbose:
        print(f"R-squared value for HarpTime to OnixClock: {r_squared_harp_to_onix}")
        print(f"R-squared value for photometry to onix: {r_squared_photometry_to_onix}")
        
    return output, photometry_sync_events, harp_to_onix_clock, onix_time_to_photometry, onix_to_harp_timestamp, photometry_to_harp_time


def select_from_photodiode_data(onix_analog_clock, OnixAnalogData, hard_start_time, harp_end_time, conversions):

    start_time = time()

    start_onix_time = conversions['harp_to_onix_clock'](hard_start_time)
    end_onix_time = conversions['harp_to_onix_clock'](harp_end_time)
    indices = np.where(np.logical_and(onix_analog_clock >= start_onix_time, onix_analog_clock <= end_onix_time))

    x, y = conversions['onix_to_harp_timestamp'](onix_analog_clock[indices]), OnixAnalogData[indices]

    print(f'Selection of photodiode data finished in {time() - start_time:.2f} seconds.')

    return x, y

def running_unit_conversion(running_array): #for ball linear movement
    resolution = 12000 # counts per inch
    inches_per_count = 1 / resolution
    meters_per_count = 0.0254 * inches_per_count
    dt = 0.01 # for OpticalTrackingRead0Y(46) -this is sensor specific. current sensor samples at 100 hz 
    linear_velocity = meters_per_count / dt # meters per second per count
    
    return running_array * linear_velocity

def rotation_unit_conversion(rotation_array): # for ball rotation
    resolution = 12000 # counts per inch
    inches_per_count = 1 / resolution
    meters_per_count = 0.0254 * inches_per_count
    dt = 0.01 # for OpticalTrackingRead0Y(46) -this is sensor specific. current sensor samples at 100 hz 
    linear_velocity = meters_per_count / dt # meters per second per count
    
    ball_radius = 0.1 # meters 
    angular_velocity = linear_velocity / ball_radius # radians per second per count
    angular_velocity = angular_velocity * (180 / np.pi) # degrees per second per count
    
    return rotation_array * angular_velocity

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

def read_ExperimentEvents(path):
    filenames = os.listdir(path/'ExperimentEvents')
    filenames = [x for x in filenames if x[:16]=='ExperimentEvents'] # filter out other (hidden) files
    date_strings = [x.split('_')[1].split('.')[0] for x in filenames] 
    sorted_filenames = pd.to_datetime(date_strings, format='%Y-%m-%dT%H-%M-%S').sort_values()
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

    
def moving_average_smoothing(X,k):
    S = np.zeros(X.shape[0])
    for t in range(X.shape[0]):
        if t < k:
            S[t] = np.mean(X[:t+1])
        else:
            S[t] = np.sum(X[t-k:t])/k
    return S

def running_unit_conversion(running_array):
    resolution = 12000 # counts per inch
    inches_per_count = 1 / resolution
    meters_per_count = 0.0254 * inches_per_count
    dt = 0.01 # for OpticalTrackingRead0Y(46)
    linear_velocity = meters_per_count / dt # meters per second per count
    
    # ball_radius = 0.1 # meters 
    # angular_velocity = linear_velocity / ball_radius # radians per second per count
    # angular_velocity = angular_velocity * (180 / np.pi) # degrees per second per count
    # print(angular_velocity)
    
    return running_array * linear_velocity * 100


