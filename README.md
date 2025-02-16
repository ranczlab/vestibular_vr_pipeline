# VestibularVR Analysis Pipeline

This is the general pipeline for loading, preprocessing, aligning, quality checking and applying basic analysis to the data recorded on the <a href=https://ranczlab.github.io/RPM/>RPM</a> (e.g. running) using <a href=https://harp-tech.org/index.html>HARP devices</a>, eye movements data derived from <a href=https://sleap.ai/>SLEAP</a> and neural data (fiber photometry, Neuropixels).

## Installation

The code mainly relies on <a href=https://github.com/harp-tech/harp-python>harp-python</a> and <a href=https://github.com/SainsburyWellcomeCentre/aeon_mecha>aeon_mecha</a> packages. The proposed setup is to first create an Anaconda environment for _aeon\_mecha_, install it and then install _harp-python_ inside of this same environment. Optional packages required by some of the example Jupyter notebooks, but not essential for the main pipeline, are cv2, ffmpeg.

### Create anaconda environment and add it to jupyter

```python
conda create -n aeon
conda activate aeon
conda install -c anaconda ipykernel
python3 -m ipykernel install --user --name=aeon
```

### Install _aeon\_mecha_
As of 2025/01, aeon\_mecha only works with python 3.11 and not later python versions. 

```python
conda install python=3.11
git clone https://github.com/SainsburyWellcomeCentre/aeon_mecha.git
cd aeon_mecha
python -m pip install -e .
```
In macOS if you get an error message, use 
`conda install pip` before the last line

### Install _harp-python_

```python
pip install harp-python
```

### Install SLEAP

```python
pip install sleap
```

### Install other packages

```python
pip install lsq-ellipse
pip install h5py
pip install opencv-python
pip install pympler # usefull for monitoring memory during dev
```

## Repository contents

```
📜demo_pipeline.ipynb   -->   main example of pipeline usage and synchronisation
📜grab_figure.ipynb
📂harp_resources
 ┣ 📄utils.py   -->   functions for data loading
 ┣ 📄process.py   -->   functions for converting, resampling, padding, aligning, plotting data
 ┣ 📄h1-device.yml   -->   H1 manifest file
 ┗ 📄h2-device.yml   -->   H2 manifest file
 ┗ 📂notebooks
    ┣ 📜load_example.ipynb
    ┣ 📜demo_synchronisation.ipynb
    ┣ 📜Treshold_exploration_Hilde.ipynb
    ┣ 📜comparing_clocked_nonclocked_data.ipynb
    ┗ 📜prepare_playback_file.ipynb
📂sleap
 ┣ 📄load_and_process.py   -->   main functions for SLEAP preprocessing pipeline
 ┣ 📄add_avi_visuals.py   -->   overlaying SLEAP points on top of the video and saving as a new one for visual inspection
 ┣ 📄horizontal_flip_script.py   -->   flipping avi videos horizontally using OpenCV
 ┣ 📄registration.py   -->   attempt at applying registration from CaImAn to get rid of motion artifacts (https://github.com/flatironinstitute/CaImAn/blob/main/demos/notebooks/demo_multisession_registration.ipynb)
 ┣ 📄upscaling.py   -->   attempt at applying LANCZOS upsampling to avi videos using OpenCV to minimise SLEAP jitter
 ┗ 📂notebooks
    ┣ 📜batch_analysis.ipynb
    ┣ 📜ellipse_analysis.ipynb   -->   visualising SLEAP preprocessing outputs
    ┣ 📜jitter.ipynb   -->   quantifying jitter inherent to SLEAP
    ┣ 📜light_reflection_motion_correction.ipynb   -->   segmentation of light reflection in the eye using OpenCV (unused)
    ┣ 📜saccades_analysis.ipynb   -->   step by step SLEAP data preprocessing (now inside of load_and_process.py + initial saccade detection
    ┗ 📜upsampling_jitter_analysis.ipynb   -->   loading SLEAP outputs from LANCZOS upsampling tests
```

## Conventions

__Saving SLEAP outputs:__

When exporting SLEAP inference outputs (in SLEAP window >> File >> Export Analysis CSV >> Current Video), save the file in the same directory as the analysed video (has to be manually located) under following naming convention:

e.g. _VideoData2_1904-01-14T04-00-00.sleap.csv_

## Functions available

### HARP Resources

__utils.py__:
- ```load_registers(dataset_path) >> returns {'H1': {'OpticalTrackingRead0X(46)': [...], ...}, 'H2': {'AnalogInput(39)': [...], ...}```
- ```read_ExperimentEvents(dataset_path) >> returns pd.DataFrame```
- ```read_OnixDigital(dataset_path) >> returns pd.DataFrame```
- ```read_OnixAnalogData(dataset_path) >> returns pd.DataFrame```
- ```read_OnixAnalogFrameCount(dataset_path) >> returns pd.DataFrame```
- ```read_OnixAnalogClock(dataset_path) >> returns pd.DataFrame```
- ```read_fluorescence(photometry_path) >> returns pd.DataFrame```
- ```read_fluorescence_events(photometry_path) >> returns pd.DataFrame```

__process.py__:
- ```resample_stream(data_stream_df, resampling_period='0.1ms', method='linear') >> resamples pd.DataFrame according to the specified method```
- ```resample_index(index, freq) >> resamples pd.DatetimeIndex according to the specified freq parameter```
- ```get_timepoint_info(registers_dict, print_all=False) >> prints all timepoint information from streams loaded with utils.load_registers```
- ```pad_and_resample(registers_dict, resampling_period='0.1ms', method='linear') >> adds padding and applies process.resample_stream to all streams loaded with utils.load_registers```
- ```plot_dataset(dataset_path) >> plotting function useful to visualise the effects of resampling on each stream```
- ```convert_datetime_to_seconds(timestamp_input) >> convert from datetime representation to seconds representation of HARP timestamps```
- ```convert_seconds_to_datetime(seconds_input) >> inverse of process.convert_datetime_to_seconds```
- ```reformat_and_add_many_streams(streams, dataframe, source_name, stream_names, index_column_name='Seconds') >> takes the input pd.DataFrame, converts to the accepted format and adds it the the streams dictionary```
- ```convert_arrays_to_dataframe(list_of_names, list_of_arrays) >> converts named arrays into pd.DataFrame```
- ```align_fluorescence_first_approach(fluorescence_df, onixdigital_df) >> alignment using the HARP timestamps in OnixDigital and photometry software timestamps (obsolete)```
- ```calculate_conversions_second_approach(data_path, photometry_path=None, verbose=True) >> calculates ONIX-HARP, HARP-ONIX, Photometry-HARP, ONIX-Photometry timestamp conversion functions according to this issue https://github.com/neurogears/vestibular-vr/issues/76```
- ```select_from_photodiode_data(OnixAnalogClock, OnixAnalogData, hard_start_time, harp_end_time, conversions) >> selects a segment of photodiode data```

### SLEAP

__load\_and_process.py__:
- ```load_videography_data(dataset_path) >> scans through VideoData1&2 folders, concatenates log files and searches for SLEAP outputs```
- ```get_coordinates_dict(df, columns_of_interest) >> converts the pd.DataFrame of SLEAP outputs to accepted format```
- ```find_horizontal_axis_angle(df, point1='left', point2='center') >> infers the horizontal axis from the average of the coordinates of the two specified points```
- ```get_left_right_center_point(coordinates_dict) >> gets the average center point between the coordinates of the two specified points```
- ```get_reformatted_coordinates_dict(coordinates_dict, columns_of_interest) >> unifies 'x' and 'y' coordinate arrays corresponding to one 'point' into a single array of shape [sample_number, 2]```
- ```get_centered_coordinates_dict(coordinates_dict, center_point) >> centers the coordinates according to the center point calculated with load_and_process.get_left_right_center_point```
- ```get_rotated_coordinates_dict(coordinates_dict, theta) >> rotates the previously centered coordinates by the angle calculated with load_and_process.find_horizontal_axis_angle```
- ```get_fitted_ellipse_parameters(coordinates_dict, columns_of_interest) >> fits at ellipse to the points designating the circumference of the pupil, returns its center point coordinates, width, height and angle```
- ```create_flipped_videos(path, what_to_flip='VideoData1') >> uses OpenCV to flip avi videos horizontally```
- ```get_all_detected_saccades(path) >> finds saccades based on heuristics and motion referenced points (obsolete)```
