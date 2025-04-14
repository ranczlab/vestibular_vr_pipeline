import os
import numpy as np
from pathlib import Path
import pandas as pd

from harp_resources import process, utils
from sleap import load_and_process as lp

def run(data_path, photometry_path):
    data_path = Path(data_path)
    photometry_path = Path(photometry_path)
    VideoData1, VideoData2, VideoData1_Has_Sleap, VideoData2_Has_Sleap = lp.load_videography_data(data_path)

    if VideoData1_Has_Sleap:
        VideoData1 = VideoData1.interpolate()

        columns_of_interest = ['left.x','left.y','center.x','center.y','right.x','right.y','p1.x','p1.y','p2.x','p2.y','p3.x','p3.y','p4.x','p4.y','p5.x','p5.y','p6.x','p6.y','p7.x','p7.y','p8.x','p8.y']
        coordinates_dict=lp.get_coordinates_dict(VideoData1, columns_of_interest)

        theta = lp.find_horizontal_axis_angle(VideoData1, 'left', 'center')
        center_point = lp.get_left_right_center_point(coordinates_dict)

        columns_of_interest = ['left', 'right', 'center', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
        remformatted_coordinates_dict = lp.get_reformatted_coordinates_dict(coordinates_dict, columns_of_interest)
        centered_coordinates_dict = lp.get_centered_coordinates_dict(remformatted_coordinates_dict, center_point)
        rotated_coordinates_dict = lp.get_rotated_coordinates_dict(centered_coordinates_dict, theta)

        columns_of_interest = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
        ellipse_parameters_data, ellipse_center_points_data = lp.get_fitted_ellipse_parameters(rotated_coordinates_dict, columns_of_interest)

        average_diameter = np.mean([ellipse_parameters_data[:,0], ellipse_parameters_data[:,1]], axis=0)

        SleapVideoData1 = process.convert_arrays_to_dataframe(['Seconds', 'Ellipse.Diameter', 'Ellipse.Angle', 'Ellipse.Center.X', 'Ellipse.Center.Y'], [VideoData1['Seconds'].values, average_diameter, ellipse_parameters_data[:,2], ellipse_center_points_data[:,0], ellipse_center_points_data[:,1]])

    if VideoData2_Has_Sleap:
        VideoData2 = VideoData2.interpolate()

        columns_of_interest = ['left.x','left.y','center.x','center.y','right.x','right.y','p1.x','p1.y','p2.x','p2.y','p3.x','p3.y','p4.x','p4.y','p5.x','p5.y','p6.x','p6.y','p7.x','p7.y','p8.x','p8.y']
        coordinates_dict=lp.get_coordinates_dict(VideoData2, columns_of_interest)

        theta = lp.find_horizontal_axis_angle(VideoData2, 'left', 'center')
        center_point = lp.get_left_right_center_point(coordinates_dict)

        columns_of_interest = ['left', 'right', 'center', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
        remformatted_coordinates_dict = lp.get_reformatted_coordinates_dict(coordinates_dict, columns_of_interest)
        centered_coordinates_dict = lp.get_centered_coordinates_dict(remformatted_coordinates_dict, center_point)
        rotated_coordinates_dict = lp.get_rotated_coordinates_dict(centered_coordinates_dict, theta)

        columns_of_interest = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
        ellipse_parameters_data, ellipse_center_points_data = lp.get_fitted_ellipse_parameters(rotated_coordinates_dict, columns_of_interest)

        average_diameter = np.mean([ellipse_parameters_data[:,0], ellipse_parameters_data[:,1]], axis=0)

        SleapVideoData2 = process.convert_arrays_to_dataframe(['Seconds', 'Ellipse.Diameter', 'Ellipse.Angle', 'Ellipse.Center.X', 'Ellipse.Center.Y'], [VideoData2['Seconds'].values, average_diameter, ellipse_parameters_data[:,2], ellipse_center_points_data[:,0], ellipse_center_points_data[:,1]])

    
    conversions = process.calculate_conversions_second_approach(data_path, photometry_path, verbose=False)

    streams = utils.load_registers(data_path)

    # Photometry = utils.read_fluorescence(photometry_path)
    # Photometry['HARP Timestamps'] = conversions['photometry_to_harp_time'](Photometry['TimeStamp'])

    Photometry = pd.read_csv(photometry_path/'preprocessed.csv')
    Photometry['TimeStamp'] = Photometry['TimeStamp'] * 1000
    Photometry['HARP Timestamps'] = conversions['photometry_to_harp_time'](Photometry['TimeStamp'])

    OnixAnalogClock = utils.read_OnixAnalogClock(data_path)
    OnixAnalogData = utils.read_OnixAnalogData(data_path, binarise=True)

    photodiode_series = pd.Series(OnixAnalogData[:,0], index=conversions['onix_to_harp_timestamp'](OnixAnalogClock))

    # Adding Photometry, Eye Movements and Photodiode to the streams
    # streams = process.reformat_and_add_many_streams(streams, Photometry, 'Photometry', ['CH1-410', 'CH1-470', 'CH1-560'], index_column_name='HARP Timestamps')
    streams = process.reformat_and_add_many_streams(streams, Photometry, 'Photometry', ['470_dfF','560_dfF','410_dfF'], index_column_name='HARP Timestamps')
    if VideoData1_Has_Sleap: streams = process.reformat_and_add_many_streams(streams, SleapVideoData1, 'SleapVideoData1', ['Ellipse.Diameter', 'Ellipse.Angle', 'Ellipse.Center.X', 'Ellipse.Center.Y'])
    if VideoData2_Has_Sleap: streams = process.reformat_and_add_many_streams(streams, SleapVideoData2, 'SleapVideoData2', ['Ellipse.Diameter', 'Ellipse.Angle', 'Ellipse.Center.X', 'Ellipse.Center.Y'])
    streams = process.add_stream(streams, 'ONIX', photodiode_series, 'Photodiode')

    resampled_streams = process.pad_and_resample(streams, resampling_period='0.1ms', method='linear')

    resampled_streams['H1']['OpticalTrackingRead0X(46)'] = process.running_unit_conversion(resampled_streams['H1']['OpticalTrackingRead0X(46)']*100)
    resampled_streams['H1']['OpticalTrackingRead0Y(46)'] = process.rotation_unit_conversion(resampled_streams['H1']['OpticalTrackingRead0Y(46)'])

    streams_to_save_pattern = {'H1': ['OpticalTrackingRead0X(46)', 'OpticalTrackingRead0Y(46)'], 'H2': ['Encoder(38)'], 'Photometry': ['470_dfF','560_dfF','410_dfF'], 'SleapVideoData1': ['Ellipse.Diameter', 'Ellipse.Center.X', 'Ellipse.Center.Y'], 'SleapVideoData2': ['Ellipse.Diameter', 'Ellipse.Center.X', 'Ellipse.Center.Y'], 'ONIX': ['Photodiode']}

    process.save_streams_as_h5(data_path, resampled_streams, streams_to_save_pattern)


GRAB_MMclosed_Regular_day1_path = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/GRAB_MMclosed&Regular_220824/'
GRAB_MMclosed_Regular_day2_path = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/GRAB_MMclosed&Regular_230824/'
GRAB_MMclosed_open_day1_path = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/GRAB_MMclosed&open_190824/'
GRAB_MMclosed_open_day2_path = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/GRAB_MMclosed&open_200824/'

G8_MMclosed_Regular_day1 = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/MMclosed&Regular_120824/'
G8_MMclosed_Regular_day2 = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/MMclosed&Regular_130824/'
G8_MMclosed_open_day1 = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/MMclosed&open_070824/'
G8_MMclosed_open_day2 = '/home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/MMclosed&open_080824/'

paths = [

    [G8_MMclosed_Regular_day1+'2024-08-12T13-26-45_B2M4',  G8_MMclosed_Regular_day1+'photometry/B2M4_MMclosed&Regular_day1/2024_08_12-15_35_54'],
    [G8_MMclosed_Regular_day1+'2024-08-12T14-35-55_B2M5',  G8_MMclosed_Regular_day1+'photometry/B2M5_MMclosed&Regular_day1/2024_08_12-16_41_58'],
    [G8_MMclosed_Regular_day1+'2024-08-12T15-23-18_B3M1',  G8_MMclosed_Regular_day1+'photometry/B3M1_MMclosed&Regular_day1/2024_08_12-17_29_03'],
    [G8_MMclosed_Regular_day1+'2024-08-12T16-03-29_B3M2',  G8_MMclosed_Regular_day1+'photometry/B3M2_MMclosed&Regular_day1/2024_08_12-18_08_18'],
    [G8_MMclosed_Regular_day1+'2024-08-12T16-51-16_B3M3',  G8_MMclosed_Regular_day1+'photometry/B3M3_MMclosed&Regular_day1/2024_08_12-18_57_17'],

    [G8_MMclosed_Regular_day2+'2024-08-13T09-44-04_B2M4',  G8_MMclosed_Regular_day2+'photometry/B2M4_MMclosed&Regular_day2/2024_08_13-11_53_03'],
    [G8_MMclosed_Regular_day2+'2024-08-13T10-36-27_B2M5',  G8_MMclosed_Regular_day2+'photometry/B2M5_MMclosed&Regular_day2/2024_08_13-12_39_37'],
    [G8_MMclosed_Regular_day2+'2024-08-13T11-20-23_B3M1',  G8_MMclosed_Regular_day2+'photometry/B3M1_MMclosed&Regular_day2/2024_08_13-13_25_15'],
    [G8_MMclosed_Regular_day2+'2024-08-13T12-07-21_B3M2',  G8_MMclosed_Regular_day2+'photometry/B3M2_MMclosed&Regular_day2/2024_08_13-14_12_24'],
    [G8_MMclosed_Regular_day2+'2024-08-13T12-53-01_B3M3',  G8_MMclosed_Regular_day2+'photometry/B3M3_MMclosed&Regular_day2/2024_08_13-14_57_35'],

    [G8_MMclosed_open_day1+'2024-08-07T09-46-19_B2M5',  G8_MMclosed_open_day1+'photometry/B2M5_MMclosed&open_day1/2024_08_07-11_57_14'],
    [G8_MMclosed_open_day1+'2024-08-07T10-40-45_B2M4',  G8_MMclosed_open_day1+'photometry/B2M4_MMclosed&open_day1/2024_08_07-12_57_09'],
    [G8_MMclosed_open_day1+'2024-08-07T11-45-03_B3M3',  G8_MMclosed_open_day1+'photometry/B3M3_MMclosed&open_day1/2024_08_07-13_49_19'],
    [G8_MMclosed_open_day1+'2024-08-07T13-40-07_B3M1',  G8_MMclosed_open_day1+'photometry/B3M1_MMclosed&open_day1/2024_08_07-15_45_13'],
    [G8_MMclosed_open_day1+'2024-08-07T14-22-22_B3M2',  G8_MMclosed_open_day1+'photometry/B3M2_MMclosed&open_day1/2024_08_07-16_29_25'],

    [G8_MMclosed_open_day2+'2024-08-08T08-22-12_B2M5',  G8_MMclosed_open_day2+'photometry/B2M5_MMclosed&open_day2/2024_08_08-10_27_15'],
    [G8_MMclosed_open_day2+'2024-08-08T09-20-54_B2M4',  G8_MMclosed_open_day2+'photometry/B2M4_MMclosed&open_day2/2024_08_08-11_24_00'],
    [G8_MMclosed_open_day2+'2024-08-08T10-05-26_B3M3',  G8_MMclosed_open_day2+'photometry/B3M3_MMclosed&open_day2/2024_08_08-12_08_29'],
    [G8_MMclosed_open_day2+'2024-08-08T11-01-22_B3M1',  G8_MMclosed_open_day2+'photometry/B3M1_MMclosed&open_day2/2024_08_08-13_04_27'],
    [G8_MMclosed_open_day2+'2024-08-08T12-03-57_B3M2',  G8_MMclosed_open_day2+'photometry/B3M2_MMclosed&open_day2/2024_08_08-14_07_03'],


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
    try:
        print(f'EXTRACTING DATA FROM DATASET {i}')
        print(f'DATA_PATH: {path_set[0]}')
        print(f'PHOTOMETRY PATH: {path_set[1]}')

        data_path = Path(path_set[0])
        output_file = data_path/f'resampled_streams_{data_path.parts[-1]}.h5'
        if os.path.exists(output_file):
            print(f'PREVIOUSLY SAVED "{output_file}" ALREADY EXISTS. SKIPPING...\n')
        else:
            run(path_set[0], path_set[1])
            print(f'DATASET {i}: FINISHING\n')
    except Exception as e:
        print(f'DATASET {i} FAILED. ERROR: {e}\n')


# G8

# MMclosed-Regular_day1
# /home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/MMclosed&Regular_120824
# 2024-08-12T13-26-45_B2M4  B2M4_MMclosed&Regular_day1/2024_08_12-15_35_54
# 2024-08-12T14-35-55_B2M5  B2M5_MMclosed&Regular_day1/2024_08_12-16_41_58
# 2024-08-12T15-23-18_B3M1  B3M1_MMclosed&Regular_day1/2024_08_12-17_29_03
# 2024-08-12T16-03-29_B3M2  B3M2_MMclosed&Regular_day1/2024_08_12-18_08_18
# 2024-08-12T16-51-16_B3M3  B3M3_MMclosed&Regular_day1/2024_08_12-18_57_17

# MMclosed-Regular_day2
# /home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/MMclosed&Regular_130824
# 2024-08-13T09-44-04_B2M4  B2M4_MMclosed&Regular_day2/2024_08_13-11_53_03
# 2024-08-13T10-36-27_B2M5  B2M5_MMclosed&Regular_day2/2024_08_13-12_39_37
# 2024-08-13T11-20-23_B3M1  B3M1_MMclosed&Regular_day2/2024_08_13-13_25_15
# 2024-08-13T12-07-21_B3M2  B3M2_MMclosed&Regular_day2/2024_08_13-14_12_24
# 2024-08-13T12-53-01_B3M3  B3M3_MMclosed&Regular_day2/2024_08_13-14_57_35

# MMclosed-open_day1
# /home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/MMclosed&open_070824
# 2024-08-07T09-46-19_B2M5  B2M5_MMclosed&open_day1/2024_08_07-11_57_14
# 2024-08-07T10-40-45_B2M4  B2M4_MMclosed&open_day1/2024_08_07-12_57_09
# 2024-08-07T11-45-03_B3M3  B3M3_MMclosed&open_day1/2024_08_07-13_49_19
# 2024-08-07T13-40-07_B3M1  B3M1_MMclosed&open_day1/2024_08_07-15_45_13
# 2024-08-07T14-22-22_B3M2  B3M2_MMclosed&open_day1/2024_08_07-16_29_25

# MMclosed-open_day2
# /home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/MMclosed&open_080824
# 2024-08-08T08-22-12_B2M5  B2M5_MMclosed&open_day2/2024_08_08-10_27_15
# 2024-08-08T09-20-54_B2M4  B2M4_MMclosed&open_day2/2024_08_08-11_24_00
# 2024-08-08T10-05-26_B3M3  B3M3_MMclosed&open_day2/2024_08_08-12_08_29
# 2024-08-08T11-01-22_B3M1  B3M1_MMclosed&open_day2/2024_08_08-13_04_27
# 2024-08-08T12-03-57_B3M2  B3M2_MMclosed&open_day2/2024_08_08-14_07_03




# GRAB

# MMclosed-Regular_day1
# /home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/GRAB_MMclosed&Regular_220824
# 2024-08-22T10-51-25_B2M6  B2M6_MMclosed&Regular_day1/2024_08_22-12_52_53
# 2024-08-22T12-30-24_B3M7  B3M7_MMclosed&Regular_day1/2024_08_22-14_34_09
# 2024-08-22T13-53-51_B3M4  B3M4_MMclosed&Regular_day1/2024_08_22-15_57_19
# 2024-08-22T11-30-52_B3M8  B3M8_MMclosed&Regular_day1/2024_08_22-13_34_31
# 2024-08-22T13-13-15_B3M6  B3M6_MMclosed&Regular_day1/2024_08_22-15_16_40

# MMclosed-Regular_day2
# /home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/GRAB_MMclosed&Regular_230824
# 2024-08-23T11-31-26_B2M6  B2M6_MMclosed&Regular_day2/2024_08_23-13_35_38
# 2024-08-23T12-51-05_B3M7  B3M7_MMclosed&Regular_day2/2024_08_23-14_54_36
# 2024-08-23T12-10-53_B3M8  B3M8_MMclosed&Regular_day2/2024_08_23-14_14_47
# 2024-08-23T13-32-34_B3M6  B3M6_MMclosed&Regular_day2/2024_08_23-15_36_40
# 2024-08-23T14-14-16_B3M4  B3M4_MMclosed&Regular_day2/2024_08_23-16_17_49

# MMclosed-open_day1
# /home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/GRAB_MMclosed&open_190824
# 2024-08-19T12-09-08_B2M6  B2M6_MMclosed&open_day1/2024_08_19-14_13_28
# 2024-08-19T12-52-52_B3M8  B3M8_MMclosed&open_day1/2024_08_19-14_57_32
# 2024-08-19T13-34-41_B3M7  B3M7_MMclosed&open_day1/2024_08_19-15_39_15
# 2024-08-19T14-18-09_B3M6  B3M6_MMclosed&open_day1/2024_08_19-16_21_55
# 2024-08-19T14-58-24_B3M4  B3M4_MMclosed&open_day1/2024_08_19-17_02_21

# MMclosed-open_day2
# /home/ikharitonov/RANCZLAB-NAS/data/ONIX/20240730_Mismatch_Experiment/GRAB_MMclosed&open_200824
# 2024-08-20T09-29-38_B2M6  B2M6_MMclosed&open_day2/2024_08_20-11_34_54
# 2024-08-20T10-11-27_B3M8  B3M8_MMclosed&open_day2/2024_08_20-12_14_52
# 2024-08-20T10-50-54_B3M7  B3M7_MMclosed&open_day2/2024_08_20-12_55_41
# 2024-08-20T11-40-52_B3M6  B3M6_MMclosed&open_day2/2024_08_20-13_44_06
# 2024-08-20T12-23-34_B3M4  B3M4_MMclosed&open_day2/2024_08_20-14_27_00