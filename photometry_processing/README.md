# Photometry_preprocessing
 For preprocessing photometry signal
 #### 1) preprocess_functions.py 
 All funcitons needed to perform preprocessing, saving a summary figure and output data files for further processing in the vestibular_vr_pipeline
 #### 2) preprocess_in_notebook_or_batch.ipynb 
 Example jupyter notebook for preprocessing and viewing an experiment step by step, or in a batch for a set of experiments 
 #### 3) run_preprocessing.py
 Example for running preprocessing from terminal. Created before Jan 2025 refactoring. If needed. should be updated based on the batch processing notebook. 

## Required data folder structure
From [issue 25](https://github.com/ikharitonov/vestibular_vr_pipeline/issues/25) in vestibular_vr_pipeline. 
- CohortX (numbered cohort of animals) 
  - experimentType_day (e.g. VestibularMismatch_day1)
    - root_data directory (animalID_yyyy-mm-ddThh-mm-ss)
      - All Bonsai acquired data (HarpData, ONIX, ExperimentEvents, SessionSettings, VideoData)
      - photometry (fluorescence_unaligned.csv, etc...)
     - root_results directory (animalID_yyyy-mm-ddThh-mm-ss_processedData)
       - SLEAP-output (output files from SLEAP inference) 
       - photometry-processed 
       - extracted ONIX data combined with synchronised photometry and eye data (single hdf5 file?)
       - figures 

## preprocess_functions.py - Not updated after Jan 2025 refactoring 
  Class of funcitons 
  Contains all funcitons needed to analyse photometry signal.
  - Uses Fluorescence_unaligned.csv and Events.csv to align all data to timestamps of 470 nm recordings. Can be changed in function create_basics().
  - in create_basics() first 15 seconds are cut in line data = rawdata[rawdata["TimeStamp"] > 15000]  - can be changed, but keep in mind initial bleaching and the effect on later detrending of data
  - Extracts events with extract_events() func into an events property that can later be saved them in a new Events.csv where each event is aligned to the main data as a boolean.
  - Can perform butterworth low-pass filter with cut-off frequency based on sensor on at ~Nyquist frequncy. List of sensors must be updated in the low_pass_filt() function.
  - detrent() can be called to fit a double exponetial to each of the signals, subtracting this to adjust for bleaching during the recording.
  - movement_correct() can be called on to use the 410 isosbestic to motion correct the 470 nm signal.
  - z-scoring can be performed by subtracting the median and dividing by standard deviation of the entire signal 
    -> set z_score(motion = True) if motion corrected signal should be used, default False
  - deltaF_F() can be called to get delta F over F of  signal, using the decay curve from detrend() as basline, and gets the percent change relative to this throughout recording.
    -> set deltaF_F(motion = True) if motion corrected signal should be used, default False
  - add_crucial_info() adds information on date, brain area recorded from, and mouse sex to the data - practical for later analysis
  - write_preprocessed_csv(Onix_align = True), call to save all data as csv files: Processed_fluorescence.csv and Events.csv
    -> Set Onix_align = False if you wish to save events recorded in the fluorescence software, and are not going to align to Onix data
    

  1) Initiate an object from the class: object = preprocess(path, sensors), providing a path to the photometry data
  2) Assign new values and properties to object by calling functions and assinging them by: object.property = function()
      - information is provided regarding whether values need to be added
  3) Some functions can only be called on if other fucntions has already been called and assigned properties to the object
  4) Save to csv files that can be read into pandas dataframes
     
