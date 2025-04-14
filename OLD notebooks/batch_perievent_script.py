import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from harp_resources import utils

# PARAMETERS
range_around_halt = [-5,11]
baselining_time_range = [-1, 0] # in relation to the halt event
photometry_type = 'GRAB'
# photometry_type = 'GCaMP8m'

apply_baseline = True

if apply_baseline:
    baseline_string = f'bsln{abs(baselining_time_range[0])}sec'
else:
    baseline_string = 'nobaseline'


def select_perievent_segment(trace, event_times, range_around_event):
    
    selected_trace_chunks = []
    
    for event_time in event_times:
        
        start = event_time + range_around_event[0]
        end = event_time + range_around_event[1]
        
        selected_trace_chunks.append(trace.loc[start:end])
        
    return np.array(selected_trace_chunks)

def baseline_subtract_trace_on_selected_range(time_array, trace_array, time_range):
    inds = np.where(np.logical_and(time_array >= time_range[0], time_array < time_range[1]))
    baselines = trace_array[:, inds].squeeze().mean(axis=1)
    baselines = baselines.repeat(trace_array.shape[1]).reshape(trace_array.shape)
    return trace_array - baselines

def run(data_path, photometry_path):

    data_path = Path(data_path)
    photometry_path = Path(photometry_path)
                        
    resampled_streams = utils.load_streams_from_h5(data_path)

    running = resampled_streams['H1']['OpticalTrackingRead0X(46)']
    photometry = resampled_streams['Photometry']['470_dfF']
    photodiode = resampled_streams['ONIX']['Photodiode']

    eye_data_exists = False

    if 'SleapVideoData1' in resampled_streams.keys():
        eye_movements_horizontal = resampled_streams['SleapVideoData1']['Ellipse.Center.X']
        eye_movements_vertical = resampled_streams['SleapVideoData1']['Ellipse.Center.Y']
        averaged_pupil_diameter_stream = resampled_streams['SleapVideoData1']['Ellipse.Diameter']
        eye_data_exists = True
    elif 'SleapVideoData2' in resampled_streams.keys():
        eye_movements_horizontal = resampled_streams['SleapVideoData2']['Ellipse.Center.X']
        eye_movements_vertical = resampled_streams['SleapVideoData2']['Ellipse.Center.Y']
        averaged_pupil_diameter_stream = resampled_streams['SleapVideoData2']['Ellipse.Diameter']
        eye_data_exists = True

    ExperimentEvents = utils.read_ExperimentEvents(data_path)
    SessionSettings = utils.read_SessionSettings(data_path, print_contents=False)

    # Finding which blocks contain halts (by looking at photodiode signal)
    blocks_names = [x['alias'] for x in SessionSettings['value']["blocks"]]
    block_starting_times = list(ExperimentEvents[ExperimentEvents.Value.isin([x+' block started' for x in blocks_names])].Seconds.values)
    block_starting_times.append(ExperimentEvents.iloc[-1].Seconds)

    blocks_with_halts_inds = []

    for i in range(len(blocks_names)):
        photodiode_segment = photodiode.loc[block_starting_times[i]:block_starting_times[i+1]]
        if True in np.unique(photodiode_segment==0) and blocks_names[i] != 'LinearNormal':
            blocks_with_halts_inds.append(i)

    # Looking through the halt containing blocks to create plots for each
    for block_i in blocks_with_halts_inds:
        A = block_starting_times[block_i]
        B = block_starting_times[block_i+1]

        # Find halt times from photodiode signal
        photodiode_stream = photodiode.loc[A:B]
        t = photodiode_stream.index
        selected_photodiode_data = photodiode_stream.values

        photodiode_low_state_times = t[np.where(selected_photodiode_data==0)].to_numpy()
        intervals_between_states = np.diff(photodiode_low_state_times)

        threshold = intervals_between_states.mean() + 1 * intervals_between_states.std()

        inds = np.where(intervals_between_states >= threshold)[0] + 1
        halt_times = photodiode_low_state_times[inds]

         # Selecting perievent segments
        selected_chunks = {}
        select_perievent_segment_func = lambda x: select_perievent_segment(x, halt_times, range_around_halt)
        selected_chunks[f'{photometry_type} df/F'] = select_perievent_segment_func(photometry)
        selected_chunks['Running'] = select_perievent_segment_func(running)
        if eye_data_exists: selected_chunks['Horizontal eye movement'] = select_perievent_segment_func(eye_movements_horizontal)
        if eye_data_exists: selected_chunks['Vertical eye movement'] = select_perievent_segment_func(eye_movements_vertical)
        if eye_data_exists: selected_chunks['Pupil diameter'] = select_perievent_segment_func(averaged_pupil_diameter_stream)

        # Baselining the selected segments with the defined range in relation to the halt event
        t = np.linspace(range_around_halt[0], range_around_halt[1], selected_chunks[f'{photometry_type} df/F'].shape[1])
        if apply_baseline:
            for name, trace in selected_chunks.items():
                selected_chunks[name] = baseline_subtract_trace_on_selected_range(t, trace, baselining_time_range)

        # Avoiding to baseline the Photodiode
        selected_chunks['Photodiode'] = select_perievent_segment_func(photodiode)

        if eye_data_exists:
            ylabels = [
                f'{photometry_type} df/F (%)',
                'speed (cm/s)',
                'horizontal coordinate (pixels)',
                'vertical coordinate (pixels)',
                'pupli diameter (pixels)',
                'photodiode state'
            ]
        else:
            ylabels = [
                f'{photometry_type} df/F (%)',
                'speed (cm/s)',
                'photodiode state'
            ]

        ########################### Individual Traces Plot ###########################

        fig, ax = plt.subplots(nrows=len(selected_chunks), ncols=1, figsize=(12,(len(selected_chunks)-1)*6))

        fig.suptitle(f'{data_path.parts[-1]} {blocks_names[block_i]} {halt_times.shape[0]} halts')

        for i, (label, trace) in enumerate(selected_chunks.items()):
            t = np.linspace(range_around_halt[0], range_around_halt[1], trace.shape[1])
            ax_traces = []
            for j in range(trace.shape[0]):
                ax_traces.append(ax[i].plot(t, trace[j, :], c='black', alpha=0.7))
            ax_traces[-1][0].set_label(label) # assign a label to a single trace only
            ax[i].add_patch(patches.Rectangle((0, ax[i].get_ylim()[0]), 1, ax[i].get_ylim()[1]-ax[i].get_ylim()[0], edgecolor='none',facecolor='red', alpha=0.3))
            ax[i].legend()
            ax[i].set_xlabel('time from halt (s)')
            ax[i].set_ylabel(ylabels[i])

        # plt.show()
        fig.savefig(data_path/f'{data_path.parts[-1]}_{blocks_names[block_i]}_{baseline_string}_individual_perievent_traces.png')

        ########################### Averaged Traces Plot ###########################

        fig, ax = plt.subplots(nrows=len(selected_chunks), ncols=1, figsize=(12,(len(selected_chunks)-1)*6))

        fig.suptitle(f'{data_path.parts[-1]} {blocks_names[block_i]} {halt_times.shape[0]} halts')

        for i, (label, traces) in enumerate(selected_chunks.items()):
            t = np.linspace(range_around_halt[0], range_around_halt[1], traces.shape[1])
            mean_trace = traces.mean(axis=0)
            std_trace = traces.std(axis=0)
            ax[i].plot(t, mean_trace, label=label, c='black', alpha=0.7)
            ax[i].fill_between(t, mean_trace - std_trace, mean_trace + std_trace, color='gray', alpha=0.3)
            ax[i].add_patch(patches.Rectangle((0, ax[i].get_ylim()[0]), 1, ax[i].get_ylim()[1]-ax[i].get_ylim()[0], edgecolor='none',facecolor='red', alpha=0.3))
            ax[i].legend()
            ax[i].set_xlabel('time from halt (s)')
            ax[i].set_ylabel(ylabels[i])

        # ax[0].set_title(data_path.parts[-1], fontsize=16)
        # plt.show()
        fig.savefig(data_path/f'{data_path.parts[-1]}_{blocks_names[block_i]}_{baseline_string}_averaged_perievent_traces.png')

        ########################### Saving data into CSV file ###########################
        # _ = selected_chunks.pop('Photodiode')
        combined_dictionaries = {'time': np.linspace(range_around_halt[0], range_around_halt[1], selected_chunks['Running'].shape[1])} | {f'{k}_mean':v.mean(axis=0) for k,v in selected_chunks.items()} | {f'{k}_mean_minus_sd':v.mean(axis=0)-v.std(axis=0) for k,v in selected_chunks.items()} | {f'{k}_mean_plus_sd':v.mean(axis=0)+v.std(axis=0) for k,v in selected_chunks.items()}
        out_df = pd.DataFrame(combined_dictionaries)
        out_df.to_csv(data_path/f'{data_path.parts[-1]}_{blocks_names[block_i]}_{baseline_string}_averaged_perievent_data.csv')

GRAB_MMclosed_Regular_day1_path = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/GRAB_MMclosed&Regular_220824/'
GRAB_MMclosed_Regular_day2_path = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/GRAB_MMclosed&Regular_230824/'
GRAB_MMclosed_open_day1_path = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/GRAB_MMclosed&open_190824/'
GRAB_MMclosed_open_day2_path = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/GRAB_MMclosed&open_200824/'

G8_MMclosed_Regular_day1 = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/MMclosed&Regular_120824/'
G8_MMclosed_Regular_day2 = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/MMclosed&Regular_130824/'
G8_MMclosed_open_day1 = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/MMclosed&open_070824/'
G8_MMclosed_open_day2 = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/MMclosed&open_080824/'

paths = [

    # [G8_MMclosed_Regular_day1+'2024-08-12T13-26-45_B2M4',  G8_MMclosed_Regular_day1+'photometry/B2M4_MMclosed&Regular_day1/2024_08_12-15_35_54'],
    # [G8_MMclosed_Regular_day1+'2024-08-12T14-35-55_B2M5',  G8_MMclosed_Regular_day1+'photometry/B2M5_MMclosed&Regular_day1/2024_08_12-16_41_58'],
    # [G8_MMclosed_Regular_day1+'2024-08-12T15-23-18_B3M1',  G8_MMclosed_Regular_day1+'photometry/B3M1_MMclosed&Regular_day1/2024_08_12-17_29_03'],
    # [G8_MMclosed_Regular_day1+'2024-08-12T16-03-29_B3M2',  G8_MMclosed_Regular_day1+'photometry/B3M2_MMclosed&Regular_day1/2024_08_12-18_08_18'],
    # [G8_MMclosed_Regular_day1+'2024-08-12T16-51-16_B3M3',  G8_MMclosed_Regular_day1+'photometry/B3M3_MMclosed&Regular_day1/2024_08_12-18_57_17'],

    # [G8_MMclosed_Regular_day2+'2024-08-13T09-44-04_B2M4',  G8_MMclosed_Regular_day2+'photometry/B2M4_MMclosed&Regular_day2/2024_08_13-11_53_03'],
    # [G8_MMclosed_Regular_day2+'2024-08-13T10-36-27_B2M5',  G8_MMclosed_Regular_day2+'photometry/B2M5_MMclosed&Regular_day2/2024_08_13-12_39_37'],
    # [G8_MMclosed_Regular_day2+'2024-08-13T11-20-23_B3M1',  G8_MMclosed_Regular_day2+'photometry/B3M1_MMclosed&Regular_day2/2024_08_13-13_25_15'],
    # [G8_MMclosed_Regular_day2+'2024-08-13T12-07-21_B3M2',  G8_MMclosed_Regular_day2+'photometry/B3M2_MMclosed&Regular_day2/2024_08_13-14_12_24'],
    # [G8_MMclosed_Regular_day2+'2024-08-13T12-53-01_B3M3',  G8_MMclosed_Regular_day2+'photometry/B3M3_MMclosed&Regular_day2/2024_08_13-14_57_35'],

    # [G8_MMclosed_open_day1+'2024-08-07T09-46-19_B2M5',  G8_MMclosed_open_day1+'photometry/B2M5_MMclosed&open_day1/2024_08_07-11_57_14'],
    # [G8_MMclosed_open_day1+'2024-08-07T10-40-45_B2M4',  G8_MMclosed_open_day1+'photometry/B2M4_MMclosed&open_day1/2024_08_07-12_57_09'],
    # [G8_MMclosed_open_day1+'2024-08-07T11-45-03_B3M3',  G8_MMclosed_open_day1+'photometry/B3M3_MMclosed&open_day1/2024_08_07-13_49_19'],
    # [G8_MMclosed_open_day1+'2024-08-07T13-40-07_B3M1',  G8_MMclosed_open_day1+'photometry/B3M1_MMclosed&open_day1/2024_08_07-15_45_13'],
    # [G8_MMclosed_open_day1+'2024-08-07T14-22-22_B3M2',  G8_MMclosed_open_day1+'photometry/B3M2_MMclosed&open_day1/2024_08_07-16_29_25'],

    # [G8_MMclosed_open_day2+'2024-08-08T08-22-12_B2M5',  G8_MMclosed_open_day2+'photometry/B2M5_MMclosed&open_day2/2024_08_08-10_27_15'],
    # [G8_MMclosed_open_day2+'2024-08-08T09-20-54_B2M4',  G8_MMclosed_open_day2+'photometry/B2M4_MMclosed&open_day2/2024_08_08-11_24_00'],
    # [G8_MMclosed_open_day2+'2024-08-08T10-05-26_B3M3',  G8_MMclosed_open_day2+'photometry/B3M3_MMclosed&open_day2/2024_08_08-12_08_29'],
    # [G8_MMclosed_open_day2+'2024-08-08T11-01-22_B3M1',  G8_MMclosed_open_day2+'photometry/B3M1_MMclosed&open_day2/2024_08_08-13_04_27'],
    # [G8_MMclosed_open_day2+'2024-08-08T12-03-57_B3M2',  G8_MMclosed_open_day2+'photometry/B3M2_MMclosed&open_day2/2024_08_08-14_07_03'],


    [GRAB_MMclosed_Regular_day1_path+'2024-08-22T10-51-25_B2M6',  GRAB_MMclosed_Regular_day1_path+'photometry/B2M6_MMclosed&Regular_day1/2024_08_22-12_52_53'],
    [GRAB_MMclosed_Regular_day1_path+'2024-08-22T12-30-24_B3M7',  GRAB_MMclosed_Regular_day1_path+'photometry/B3M7_MMclosed&Regular_day1/2024_08_22-14_34_09'],
    [GRAB_MMclosed_Regular_day1_path+'2024-08-22T13-53-51_B3M4',  GRAB_MMclosed_Regular_day1_path+'photometry/B3M4_MMclosed&Regular_day1/2024_08_22-15_57_19'],
    [GRAB_MMclosed_Regular_day1_path+'2024-08-22T11-30-52_B3M8',  GRAB_MMclosed_Regular_day1_path+'photometry/B3M8_MMclosed&Regular_day1/2024_08_22-13_34_31'],
    [GRAB_MMclosed_Regular_day1_path+'2024-08-22T13-13-15_B3M6',  GRAB_MMclosed_Regular_day1_path+'photometry/B3M6_MMclosed&Regular_day1/2024_08_22-15_16_40'],

    [GRAB_MMclosed_Regular_day2_path+'2024-08-23T11-31-26_B2M6',  GRAB_MMclosed_Regular_day2_path+'photometry/B2M6_MMclosed&Regular_day2/2024_08_23-13_35_38'],
    [GRAB_MMclosed_Regular_day2_path+'2024-08-23T12-51-05_B3M7',  GRAB_MMclosed_Regular_day2_path+'photometry/B3M7_MMclosed&Regular_day2/2024_08_23-14_54_36'],
    [GRAB_MMclosed_Regular_day2_path+'2024-08-23T12-10-53_B3M8',  GRAB_MMclosed_Regular_day2_path+'photometry/B3M8_MMclosed&Regular_day2/2024_08_23-14_14_47'],
    [GRAB_MMclosed_Regular_day2_path+'2024-08-23T13-32-34_B3M6',  GRAB_MMclosed_Regular_day2_path+'photometry/B3M6_MMclosed&Regular_day2/2024_08_23-15_36_40'],
    [GRAB_MMclosed_Regular_day2_path+'2024-08-23T14-14-16_B3M4',  GRAB_MMclosed_Regular_day2_path+'photometry/B3M4_MMclosed&Regular_day2/2024_08_23-16_17_49'],

    [GRAB_MMclosed_open_day1_path+'2024-08-19T12-09-08_B2M6',  GRAB_MMclosed_open_day1_path+'photometry/B2M6_MMclosed&open_day1/2024_08_19-14_13_28'],
    [GRAB_MMclosed_open_day1_path+'2024-08-19T12-52-52_B3M8',  GRAB_MMclosed_open_day1_path+'photometry/B3M8_MMclosed&open_day1/2024_08_19-14_57_32'],
    [GRAB_MMclosed_open_day1_path+'2024-08-19T13-34-41_B3M7',  GRAB_MMclosed_open_day1_path+'photometry/B3M7_MMclosed&open_day1/2024_08_19-15_39_15'],
    [GRAB_MMclosed_open_day1_path+'2024-08-19T14-18-09_B3M6',  GRAB_MMclosed_open_day1_path+'photometry/B3M6_MMclosed&open_day1/2024_08_19-16_21_55'],
    [GRAB_MMclosed_open_day1_path+'2024-08-19T14-58-24_B3M4',  GRAB_MMclosed_open_day1_path+'photometry/B3M4_MMclosed&open_day1/2024_08_19-17_02_21'],

    [GRAB_MMclosed_open_day2_path+'2024-08-20T09-29-38_B2M6',  GRAB_MMclosed_open_day2_path+'photometry/B2M6_MMclosed&open_day2/2024_08_20-11_34_54'],
    [GRAB_MMclosed_open_day2_path+'2024-08-20T10-11-27_B3M8',  GRAB_MMclosed_open_day2_path+'photometry/B3M8_MMclosed&open_day2/2024_08_20-12_14_52'],
    [GRAB_MMclosed_open_day2_path+'2024-08-20T10-50-54_B3M7',  GRAB_MMclosed_open_day2_path+'photometry/B3M7_MMclosed&open_day2/2024_08_20-12_55_41'],
    [GRAB_MMclosed_open_day2_path+'2024-08-20T11-40-52_B3M6',  GRAB_MMclosed_open_day2_path+'photometry/B3M6_MMclosed&open_day2/2024_08_20-13_44_06'],
    [GRAB_MMclosed_open_day2_path+'2024-08-20T12-23-34_B3M4',  GRAB_MMclosed_open_day2_path+'photometry/B3M4_MMclosed&open_day2/2024_08_20-14_27_00']
]


for i, path_set in enumerate(paths):
    if os.path.exists(Path(path_set[0])/f'resampled_streams_{Path(path_set[0]).parts[-1]}.h5'):
        # print(f'DATASET {i}: STARTING')
        # print(f'DATA_PATH: {path_set[0]}')
        # print(f'PHOTOMETRY PATH: {path_set[1]}')
        # run(path_set[0], path_set[1])
        # print(f'DATASET {i}: FINISHING\n')
        try:
            print(f'DATASET {i}: STARTING')
            print(f'DATA_PATH: {path_set[0]}')
            print(f'PHOTOMETRY PATH: {path_set[1]}')
            run(path_set[0], path_set[1])
            print(f'DATASET {i}: FINISHING\n')
        except Exception as e:
            print(f'DATASET {i} FAILED. ERROR: {e}\n')
    else:
        print(f'DATASET {i}: NO DATA FOUND')
        print(f'DATA_PATH: {path_set[0]}')
        print(f'PHOTOMETRY PATH: {path_set[1]}')
        print(f'DATASET {i}: CONTINUING...\n')