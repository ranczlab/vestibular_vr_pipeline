import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import traceback #for debugging purposes

#FIXME cleanup list 
#Events handling is confusing, and probably unnecessary complicated (see issue #4 on repo)
    #the current example notebook and batch notebook does not use extract_events
    #remove extract_events function as not needed
    #remove events from create_basic as not needed? waiting for Hilde's input 
#dF/F calculation uses Akam method, uses exp fit as baseline, this does not make much sense to me, but results look right 
    # Based on: https://github.com/ThomasAkam/photometry_preprocessing
#keeps raw_data (and data?) when actually not needed for further processing (not a big memory load, but untidy)
#Different functions use different names internally eg signals or traces  


class preprocess:
    def __init__(self, path, sensors):
        """
        Initializes the preprocess class with the given path and sensors.
        Parameters:
        path (str): The path to the root_data directory (contains all acquired data)
        sensors (list): List of sensors used in the experiment.
        """
        self.path = os.path.join(path, '')
        
         # Create save path one level up with _processedData appended
        parent_dir = os.path.dirname(os.path.dirname(self.path))
        folder_name = os.path.basename(os.path.dirname(self.path)) + "_processedData/photometry"
        self.save_path = os.path.join(parent_dir, folder_name)
        # Create save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        
        # Update path to point to photometry data folder
        self.path = os.path.join(path, 'photometry', '')
        self.sensors = sensors
        self.colors = ['mediumseagreen', 'indianred', 'mediumpurple', 'steelblue']


    def show_structure(self):
        """
        Display readable structure of photometry object
        Useful for debugging and understanding the object's contents
        """
        # Get attributes without dunders
        attributes = [attr for attr in dir(self) if not attr.startswith('__')]
        
        # Organize by type
        dataframes = []
        methods = []
        other = []
        
        for attr in attributes:
            if isinstance(getattr(self, attr), pd.DataFrame):
                dataframes.append(attr)
            elif callable(getattr(self, attr)):
                methods.append(attr)
            else:
                other.append(attr)
                
        # Print organized structure
        print("\n=== DataFrames ===")
        for df in sorted(dataframes):
            shape = getattr(self, df).shape
            print(f"{df}: {shape[0]} rows × {shape[1]} columns")
            
        print("\n=== Properties ===")
        for prop in sorted(other):
            print(f"{prop}: {type(getattr(self, prop)).__name__}")
            
        print("\n=== Methods ===")
        print(", ".join(sorted(methods)))
    
    
    def get_info(self):
        """
        Reads the Fluorescence.csv file and extracts experiment information.
        Returns:
        dict: A dictionary containing experiment information.
        """
        with open(os.path.join(self.path, 'Fluorescence.csv'), newline='') as f:
            reader = csv.reader(f)
            row1 = next(reader)

        info_str = str(row1[0]).replace(";", ",").replace('true', 'True').replace("false", "False")
        info = eval(info_str)

        return info


    def create_basic(self, cutend = False, cutstart = False, target_area = 'X', motion = False):
        '''
        populates self.info with the experiment information
        reads fluorescence and event csv files 
        alignes all channels to the same timestamp (470 nm)
        does the same for events, but this makes no sense, it's some historical remnant (and not used later anyway)
        returns the raw data, data, data_seconds and signals FIXME confusing names, signals is used going froward, but then what is data? 
        '''
        
        info = self.info
        event_path = self.path + 'Events.csv'  # events with precise timestamps
        fluorescence_path = self.path + 'Fluorescence-unaligned.csv'  # fluorescence with precise timestamps

        # get parameters from path name
        mousename = self.path.split('/')[-3][:7]
        session = self.path.split('/')[-3][8:]
        if '&' in session:
            session = session.replace('&', '-') 
        experiments = self.path.split('/')[-4][:]
               
        #Adding mousename, target_area and experiment type to info  
        if 'mousename' not in self.info:
            self.info['mousename'] = ()
        self.info['mousename']= mousename
        if 'target_area' not in self.info:
            self.info['target_area'] = ()
        self.info['target_area'] = target_area
        if 'experiment_type' not in self.info:
            self.info['experiment_type'] = ()
        self.info['experiment_type'] = experiments
        if 'motion_correction' not in self.info:
            self.info['motion_correction'] = ()
        self.info['motion_correction'] = motion
        
        #print the mousename and session to output, epsecially usefull for batch processing 
        print(f'\n\033[1mPreprocessing data for {self.info["mousename"]} in session {session}...\033[0m\n')
        
        #reading the csv files into pandas dataframes
        events = pd.read_csv(event_path) 
        fluorescence = pd.read_csv(fluorescence_path)
        
        #rename the 'Name' column in events to 'Event'
        events.rename(columns={'Name': 'Event'}, inplace=True)
        
        # Create and fill separate dataframes for each wavelength
        df_470 = fluorescence[fluorescence['Lights'] == 470][['TimeStamp', 'Channel1']].rename(columns={'Channel1': '470'})
        df_410 = fluorescence[fluorescence['Lights'] == 410][['TimeStamp', 'Channel1']].rename(columns={'Channel1': '410'})
        df_560 = fluorescence[fluorescence['Lights'] == 560][['TimeStamp', 'Channel1']].rename(columns={'Channel1': '560'})

        # Merge the '410' and '560' dataframes with the '470' dataframe based on nearest timestamps 
        # All will be shifted to match the 470 nm signal
        df_final = pd.merge_asof(df_470, df_410, on='TimeStamp', direction='backward')
        df_final = pd.merge_asof(df_final, df_560, on='TimeStamp', direction='backward')

        # Fill nan values or handle missing data
        # first forward filling and then backwards filling in case of nans at the end of the columns
        df_final['410'] = df_final['410'].ffill().bfill()
        df_final['560'] = df_final['560'].ffill().bfill()

        #merge events and fluorescence
        #is merged to match the 470 signal timestamps as closely as possible,
        # meaning that the event timestamps may be slightly shifted
        # FIXME in practice, Events.csv saved at the end is exactly the same as the original Events.csv file, so unsure if pd.merge.asof is necessary
        if len(events) < 1:
            print("WARNING: No events were found. Check for missing sync signal.")
            rawdata = df_final
            rawdata = rawdata.loc[:, ~rawdata.columns.str.contains('^Unnamed')]
            #data = rawdata[rawdata["TimeStamp"] > 30] FIXME if this line is needed, weird magic number plus there should always be events 
            data = rawdata
        else:
            rawdata = pd.merge_asof(df_final, events[['TimeStamp', 'Event', 'State']], on='TimeStamp', direction='backward') #FIXME this is not saved in the Events.csv file
            rawdata = rawdata.loc[:,~rawdata.columns.str.contains('^Unnamed')]  # sometimes an Unnamed column has appeared...
            # removing first 15 seconds because of bleaching
            if cutstart:
                # Remove initial bleaching period (first 15 seconds)
                data = rawdata[rawdata["TimeStamp"] > 15000]  
            else:
                data = rawdata
        if cutend == True:
            data = data.drop(data.tail(300).index) #This can be done if fiber was by mistake removed before NOTE why 300?
        
        data_seconds = pd.DataFrame([second for second in data['TimeStamp'] / 1000], columns=['TimeStamp'])
        signals = pd.DataFrame()
        if info['Light']['Led470Enable'] == True:
            signals['470'] = data['470']
        if info['Light']['Led560Enable'] == True:
            signals['560'] = data['560']
        if info['Light']['Led410Enable'] == True:
            signals['410'] = data['410']

        #Adding events as booleans: Create a new column for each unique event in the 'Name' column
        unique_events = events['Event'].unique()
        data = data.copy()
        for event in unique_events:
            # Initialize the event-specific column with False, using loc to avoid SettingWithCopyWarning
            data.loc[:, f"{event}_event"] = False
            
            # Filter the events for this specific event name
            event_rows = events[events['Event'] == event]
            
            #Get the transitions from 0 to 1
            transitions = event_rows[(event_rows['State'].shift(1) == 0) & (event_rows['State'] == 1)]

            #Get the timestamps for both 0 and 1 states
            start_timestamps = event_rows[event_rows['State'] == 0].loc[event_rows['State'].shift(-1) == 1, 'TimeStamp'].values
            end_timestamps = event_rows[event_rows['State'] == 1].loc[event_rows['State'].shift(1) == 0, 'TimeStamp'].values

            # For each time range, modify the corresponding values in another DataFrame (e.g., 'other_df')
            for start, end in zip(start_timestamps, end_timestamps):
                mask = (data['TimeStamp'] >= start) & (data['TimeStamp'] <= end)
                data.loc[mask, f"{event}_event"] = True 
 
        return(rawdata, data, data_seconds, signals)
        

    def low_pass_filt(self, method = "auto", plot=False, x_start=None, x_end=None, savefig = False):
        """
        Apply low-pass filter to signals.
        
        Parameters:
        plot (bool): Whether to plot the filtered signals
        method (str): Method to determine cutoff frequency. Options: 'auto', 'sensor'
        x_start (float, optional): Start time for x-axis in seconds. If None, uses minimum time.
        x_end (float, optional): End time for x-axis in seconds. If None, uses maximum time.
        """
        
        signals = self.signals
        sensors = self.sensors
        filtered = pd.DataFrame(index=self.signals.index)
        # Set x-axis limits
        if x_start is None:
            x_start = self.data_seconds['TimeStamp'].min()
        if x_end is None:
            x_end = self.data_seconds['TimeStamp'].max()
        
        # The RWD software records the Fps for all recorded channels combined
        sample_rate = self.info['Fps'] / len(sensors) #get sample_rate per channel
        
        # Add keys if they don't exist
        if 'filtering_Wn' not in self.info:
            self.info['filtering_Wn'] = {}
        if 'filtering_method' not in self.info:
            self.info['filtering_method'] = {}
        
        for signal in signals:
            sensor = sensors[signal]
            
            # Determine cutoff frequency and filter order
            if method == 'auto':
                Wn = np.floor(sample_rate / 2.01)  # ~Nyquist frequency
                self.info['filtering_method'][signal] = 'auto'
            elif method == 'sensor': 
                if sensor == 'G8m':
                    Wn = 16
                elif sensor == 'rG1':
                    Wn = 16 
                elif sensor == 'g5-HT3':
                    Wn = 16
                elif isinstance(sensor, int):
                    Wn = 100 / (sensor / 3)
                else:
                    Wn = 1000 / (int(input("Enter sensor half decay time: ")) / 3)
                self.info['filtering_method'][signal] = 'sensor_specific'
                
            print(f'Filtering {signal} with method {method} at {Wn} Hz')
            
            # Store the Wn value for this signal in 'filtering_Wn'
            self.info['filtering_Wn'][signal] = [Wn]
            
            try:
                # Design the filter
                b, a = butter(2, Wn, btype='low', fs=sample_rate)
                # Apply the filter
                filtered[f'filtered_{signal}'] = filtfilt(b, a, signals[signal])
                self.info['filtering_Wn'][signal].append(True)
            
            except ValueError:
                print(f'Wn is set to {Wn} for the sensor {sensor}, but samplig rate {sample_rate} Hz is less than 2 * {Wn}. No filtering done.')
                self.info['filtering_Wn'][signal].append(False)
                # Copy the unfiltered signal to the filtered DataFrame
                filtered[f'filtered_{signal}'] = signals[signal]
        
        if plot:
            num_signals = len(signals.columns)
            fig, axes = plt.subplots(num_signals, 1, figsize=(12, 8), sharex=True)
            
            # If there's only one signal, `axes` won't be a list
            if num_signals == 1:
                axes = [axes]
            
            for ax, signal, color in zip(axes, signals, self.colors):
                # Create mask using data index
                mask = (self.data_seconds.index >= self.data_seconds[self.data_seconds['TimeStamp'] >= x_start].index[0]) & \
                    (self.data_seconds.index <= self.data_seconds[self.data_seconds['TimeStamp'] <= x_end].index[-1])
                
                # Plot data
                ax.plot(self.data_seconds['TimeStamp'], signals[signal], label='Original', color=color, alpha=1, linewidth=1)
                ax.plot(self.data_seconds['TimeStamp'], filtered[f'filtered_{signal}'], label='Filtered', color="black", linewidth=0.5, alpha=0.5)
                
                # Set x limits
                ax.set_xlim([x_start, x_end])
                
                # Set y limits based on visible range
                visible_orig = signals[signal][mask]
                visible_filt = filtered[f'filtered_{signal}'][mask]
                y_min = min(visible_orig.min(), visible_filt.min())
                y_max = max(visible_orig.max(), visible_filt.max())
                ax.set_ylim([y_min, y_max])
                
                ax.set_title(f'Signal: {signal}')
                ax.set_ylabel('fluorescence')
                ax.legend()
            
            axes[-1].set_xlabel('Seconds')
            fig.suptitle(f'Low-pass Filtered {self.info["mousename"]} with method: {method}')
            plt.tight_layout()
            
             # Save plot to file
            if savefig:
                fig.savefig(self.save_path + f'/low-pass_filtered_{self.info["mousename"]}.png', dpi=150)  # Lower DPI to save time
            
            plt.show()
            plt.close(fig)
            
        return filtered

    def detrend(self, plot=False, method='divisive', savefig = False):
        '''
        Detrends the filtered data using a double exponential fit
        Method can be divisive (returns dF/F by construction) or subtractive (returns raw signal values)
        '''
        
        try:
            traces = self.filtered
        except:
            print('No filtered signal was found')
            traces = self.signals
            
        detrended = pd.DataFrame(index=traces.index)  # Initialize with proper index
        exp_fits = pd.DataFrame()
        
        if 'detrend_params' not in self.info:
            self.info['detrend_params'] = {}   
        if 'detrend_method' not in self.info:
            self.info['detrend_method'] = {}
        self.info['detrend_method'] = method
            
        def double_exponential(time, amp_const, amp_fast, amp_slow, tau_slow, tau_multiplier):
            '''
            Based on: https://github.com/ThomasAkam/photometry_preprocessing
            Compute a double exponential function with constant offset.
            Parameters:
            t       : Time vector in seconds.
            const   : Amplitude of the constant offset.
            amp_fast: Amplitude of the fast component.
            amp_slow: Amplitude of the slow component.
            tau_slow: Time constant of slow component in seconds.
            tau_multiplier: Time constant of fast component relative to slow.
            '''
            tau_fast = tau_slow * tau_multiplier
            return amp_const + amp_slow * np.exp(-time / tau_slow) + amp_fast * np.exp(-time / tau_fast)

        # Fit curve to signal.
        for trace in traces:
            max_sig = np.max(traces[trace])
            min_sig = np.min(traces[trace])
            inital_params = [max_sig*0.5, max_sig*0.1, min_sig*0.8, 1, 0.1]
            bounds = ([0, 0, 0, 0, 0],
                  [max_sig, max_sig, max_sig, 36000, 1])
            try:
                signal_parms, _ = curve_fit(double_exponential, self.data_seconds['TimeStamp'], traces[trace], p0=inital_params, bounds=bounds, maxfev=1000) # parm_cov is not used
            except RuntimeError:
                print('Could not fit exponential fit for: \n', self.path)
                pass
            signal_expfit = double_exponential(self.data_seconds['TimeStamp'], *signal_parms)
            print(f'Detrending {trace} with params: [{", ".join(f"{x:.3g}" for x in signal_parms)}]') # prints to 3 significant digits 
            self.info['detrend_params'][trace]=signal_parms
            if method == "subtractive":
                signal_detrended = traces[trace].reset_index(drop=True) - signal_expfit.reset_index(drop=True)
            if method == "divisive":
                signal_detrended = traces[trace].reset_index(drop=True) / signal_expfit.reset_index(drop=True)
            # Add detrended signal to DataFrame
            detrended[f'detrend_{trace}'] = signal_detrended
            exp_fits[f'expfit{trace[-4:]}'] = signal_expfit

        #Plotting the detrended data
        if plot and len(detrended.columns) > 0:  # Only plot if there's data

            # Plotting the filtered data with the exponential fit
            fig, axs = plt.subplots(len(traces.columns), figsize=(15, 10), sharex=True)
            color_count = 0
            for trace, exp, ax in zip(traces.columns, exp_fits.columns, axs):
                # Plot original data
                line1 = ax.plot(self.data_seconds, traces[trace], c=self.colors[color_count], label=trace, alpha=0.7)
                color_count += 1
                
                # Create twin axis and plot exp fit
                ax2 = ax.twinx()
                line2 = ax2.plot(self.data_seconds, exp_fits[exp], c='black', label=exp)
                
                # Get combined y limits
                y_min = min(traces[trace].min(), exp_fits[exp].min())
                y_max = max(traces[trace].max(), exp_fits[exp].max())
                
                # Set same limits for both axes
                ax.set_ylim([y_min, y_max])
                ax2.set_ylim([y_min, y_max])
                
                ax.set(ylabel='fluorescence')
                ax2.set(ylabel='exponential fit')
                lns = line1 + line2
                labs = [l.get_label() for l in lns]
                ax.legend(lns, labs, loc=0)
            
            axs[-1].set(xlabel='seconds')
            fig.suptitle(f'exponential fit {self.info["mousename"]}')
            if savefig:
                plt.savefig(self.save_path + f'/exp-fit_{self.info["mousename"]}.png', dpi=300)
            
            fig, axs = plt.subplots(len(detrended.columns), figsize = (15, 10), sharex=True)
            color_count = 0
            for column, ax in zip(detrended.columns, axs):
                ax.plot(self.data_seconds, detrended[column], c=self.colors[color_count], label=column)
                if method == "subtractive":
                    ax.set(ylabel='data detrended (raw A.U.)')
                if method == "divisive":
                    ax.set(ylabel='data detrended (dF/F)')
                color_count += 1
                ax.legend()
            axs[-1].set(xlabel='seconds')
            fig.suptitle(f'detrended data {self.info["mousename"]}, with method: {method}')
            if savefig:
                plt.savefig(self.save_path + f'/Detrended_data_{self.info["mousename"]}.png', dpi=300)
        
        print(detrended.columns.tolist())  # Print the column names in the filtered DataFrame
        return detrended, exp_fits


    def motion_correct(self, plot=False, iso_ch = 410, signal_ch = 470):
        """
        Performs motion correction on photometry signals using linear regression between iso_ch and signal_ch channels.
        
        Args:
            plot (bool): If True, generates a correlation plot between 410nm and signal_ch signals.
        
        Returns:
            pandas.DataFrame: Contains all signals with motion-corrected signal_ch and original other channel signals.
                            Returns empty DataFrame if correction fails.
        """
        # Check if we have detrended data
        if not hasattr(self, 'detrended'):
            print('No detrended data found. Run detrend() method first.')
            return pd.DataFrame()
        
        if 'motion_correction_slope' not in self.info:
            self.info['motion_correction_slope'] = []
            self.info['motion_correction_R-squared'] = []
            self.info['motion_correction_isosbestic'] = iso_ch
            self.info['motion_correction_signal'] = signal_ch
    
        data = self.detrended
        corrected_data = pd.DataFrame()
        iso_channel = f'detrend_filtered_{iso_ch}'
        signal_channel = f'detrend_filtered_{signal_ch}'
        
        try:
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x=data[iso_channel], 
                                                                    y=data[signal_channel])
            self.info['motion_correction_slope'] = slope
            self.info['motion_correction_R-squared'] = r_value**2
    
            # Generate correlation plot if requested
            if plot:
                fig, ax = plt.subplots(figsize=(15, 10))
                plt.scatter(data[iso_channel][::5], data[signal_channel][::5], 
                           alpha=0.1, marker='.')
                x = np.array(ax.get_xlim())
                ax.plot(x, intercept + slope * x)
                ax.set_xlabel(iso_ch)
                ax.set_ylabel(signal_ch)
                ax.set_title(f'{iso_ch} nm - {signal_ch} nm correlation {self.info["mousename"]}.')
                plt.rcParams.update({'font.size': 18})
    
            # Apply motion correction if enabled and slope is positive
            if self.info['motion_correction'] == False:
                print('Motion correction NOT applied, plot (if present) for information only.')
                # Copy all original signals
                for col in data.columns:
                    corrected_data[col] = data[col]
                return corrected_data
    
            if slope > 0:
                # Calculate motion-corrected signal,the correction calculated using the signal_ch/iso_ch linear fit is applied to the signal channel only 
                control_corr = intercept + slope * data[iso_channel] #
                corrected_410 = data['detrend_filtered_410']
                if signal_ch == 470:
                    corrected_470 = data['detrend_filtered_470'] - control_corr
                    corrected_560 = data['detrend_filtered_560']
                if signal_ch == 560:
                    corrected_470 = data['detrend_filtered_470']
                    corrected_560 = data['detrend_filtered_560'] - control_corr
                
                #Create output DataFrame with all signals
                for col in data.columns:
                    if '470' in col:
                        corrected_data['motion_corr_470'] = corrected_470
                    elif '410' in col:
                        corrected_data['motion_corr_410'] = corrected_410
                    elif '560' in col:
                        corrected_data['motion_corr_560'] = corrected_560 

                self.info['motion_correction'] = True
                print(f'Motion correction APPLIED to all channels based on iso {iso_ch} and singal {signal_ch}. slope = {slope:.3f} and R-squared = {r_value**2:.3f}')
                return corrected_data

                #debugging 
                # Add new plot showing motion corrected signals
                if plot:
                    fig, ax = plt.subplots(figsize=(15, 10))
                    for col in corrected_data.columns:
                        ax.plot(self.data_seconds['TimeStamp'], 
                            corrected_data[col], 
                            label=col,
                            alpha=0.7)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Fluorescence')
                    ax.set_title(f'Motion Corrected Signals - {self.info["mousename"]}')
                    ax.legend()
                    plt.tight_layout()
                #debugging   
                    return corrected_data
            
            else:
                print('WARNING: signal was not motion corrected as slope <= 0')
                print(f'Intercept: {intercept}, Slope: {slope}')
                self.info['motion_correction'] = False
                # Copy all original signals FIXME why si this needed, just return data no?
                for col in data.columns:
                    corrected_data[col] = data[col]
                return corrected_data
                
    
        except KeyError:
            print('Linear fit failed, the original data is returned')
            # Print full traceback
            traceback.print_exc()
            self.info['motion_correction'] = False
            # Copy all original signals FIXME why si this needed, just return data no?
            for col in data.columns:
                corrected_data[col] = data[col]
            return corrected_data


    def get_deltaF_F(self, plot=False, savefig=False):
        '''
        Input:
        - the detrended signal as delta F
        - The decay curve estimate as baseline fluorescence (FIXME how does this make sense? same in Akam notebook) 
        Calculates deltaF / F if detrending was subtractive, otherwise returns the detrended (or motion corrected) signal as dF/F
        :returns
        - dF/F signals
        '''
        dF_F = pd.DataFrame()
    
        if self.info['detrend_method'] == 'subtractive':
            print('dF/F: The method used for detrending was subtractive, deltaF/F is calculated now.')
            if self.info['motion_correction'] == False:
                main_data = self.detrended
                for signal, fit in zip(main_data, self.exp_fits):
                    F = self.exp_fits[fit]
                    deltaF = main_data[signal]
                    signal_dF_F = deltaF / F
                    dF_F[f'{signal[-3:]}_dfF'] = signal_dF_F
            elif self.info['motion_correction'] == True:
                if self.motion_corrected is None:
                    print("Warning: motion_corrected is None. Skipping motion correction.")
                else:
                    for signal in self.motion_corrected.columns:
                        dF_F[f'{signal[-3:]}_dfF'] = self.motion_corrected[signal]
                    main_data = self.motion_corrected
                    for signal, fit in zip(main_data, self.exp_fits):
                        F = self.exp_fits[fit]
                        deltaF = main_data[signal]
                        signal_dF_F = deltaF / F
                        dF_F[f'{signal[-3:]}_dfF'] = signal_dF_F

    
        if self.info['detrend_method'] == 'divisive': #already dF/F and was motion corrected  
            print('dF/F: Only doing motion correction if needed, as divisive detrending already resulted in deltaF/F')
            if self.info['motion_correction'] == False:
                # Copy all detrended signals to dF_F
                for signal in self.detrended.columns:
                    dF_F[f'{signal[-3:]}_dfF'] = self.detrended[signal]
            
            if self.info['motion_correction'] == True:
                # Copy all motion corrected signals to dF_F
                for signal in self.motion_corrected.columns:
                    dF_F[f'{signal[-3:]}_dfF'] = self.motion_corrected[signal]

    
        if plot:
            fig, axs = plt.subplots(len(dF_F.columns), figsize=(15, 10), sharex=True)
            color_count = 0
            if len(dF_F.columns) > 1:
                for column, ax in zip(dF_F.columns, axs):
                    ax.plot(self.data_seconds, dF_F[column], c=self.colors[color_count], label=column)
                    ax.set(xlabel='seconds', ylabel='dF/F')
                    ax.legend()
                    color_count += 1
            else:
                axs.plot(self.data_seconds, dF_F, c=self.colors[2])
                axs.set(xlabel='seconds', ylabel='dF/F')
            fig.suptitle(f'Delta F/F {self.info["mousename"]}')
    
            if savefig:
                plt.savefig(self.save_path + f'/deltaf-F_figure_{self.info["mousename"]}.png', dpi=300)
    
        return dF_F


    def z_score(self, plot=False, savefig=False):
        '''
        Z-scoring of signal traces
        Gives the signal strength in terms of standard deviation units
        '''
        zscored_data = pd.DataFrame()
    
        if self.info['motion_correction'] == True: 
            print('z-scoring motion corrected data')
            signals = self.motion_corrected
            for signal in signals:
                signal_corrected = signals[signal]
                zscored_data[f'z_{signal[-3:]}'] = (signal_corrected - np.median(signal_corrected)) / np.std(signal_corrected)
    
        if self.info['motion_correction'] == False:
            print('z-scoring non-motion corrected data') 
            signals = self.detrended
            for signal in signals:
                signal_corrected = signals[signal]
                zscored_data[f'z_{signal[-3:]}'] = (signal_corrected - np.median(signal_corrected)) / np.std(signal_corrected)
    
        zscored_data = zscored_data.reset_index(drop=True)
        
        if plot:
            fig, axs = plt.subplots(len(zscored_data.columns), figsize=(15, 10), sharex=True)
            color_count = 0
            if len(zscored_data.columns) > 1:
                for column, ax in zip(zscored_data.columns, axs):
                    ax.plot(self.data_seconds, zscored_data[column], c=self.colors[color_count], label=column)
                    ax.set(xlabel='seconds', ylabel='z-scored dF/F or raw')
                    ax.legend()
                    color_count += 1
            else:
                axs.plot(self.data_seconds, zscored_data, c=self.colors[2])
                axs.set(xlabel='seconds', ylabel='z-scored dF/F or raw')
                axs.legend()
    
            method_label = "raw" if self.info['detrend_method'] == "subtractive" else "dF/F"
            fig.suptitle(f'Z-scored {method_label} data {self.info["mousename"]}')
            
            if savefig:
                plt.savefig(self.save_path + f'/zscored_figure_{self.info["mousename"]}.png', dpi=300)
    
        return zscored_data


    def write_info_csv(self):
        '''
        Writes the available info into a csv file that can be read into a dictionary
        # To read it back:
        with open(f'{filename}_info.csv') as csv_file:
            reader = csv.reader(csv_file)
            info = dict(reader)
        '''
        path = self.save_path
        info = self.info
        with open(f'{path}/info.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in info.items():
                writer.writerow([key, value])
        print('Info.csv saved')
    
    
    def cross_correlate_signals(self, col1='470', col2='560', plot = False):
        """Cross-correlate specified signals, find absolute peak in ±5s window, and plot the result."""
        # Validate inputs
        valid_columns = ['470', '560', '410']
        if col1 not in valid_columns or col2 not in valid_columns:
            raise ValueError(f"Invalid columns. Valid options are: {valid_columns}")
        
        # Extract signals
        z_col1 = self.zscored[f'z_{col1}']
        z_col2 = self.zscored[f'z_{col2}']
        
        # Compute cross-correlation
        cross_corr = np.correlate(z_col1 - np.mean(z_col1), z_col2 - np.mean(z_col2), mode='full')
        lags = np.arange(-len(z_col1) + 1, len(z_col1))
        
        # Define ±5 second window
        sampling_rate = 1 / np.mean(np.diff(self.data_seconds['TimeStamp']))
        window_size = int(5 * sampling_rate)
        center = len(cross_corr) // 2
        window = cross_corr[center - window_size:center + window_size + 1]
        window_lags = lags[center - window_size:center + window_size + 1] / sampling_rate  # Convert to seconds
        
        # Find absolute peak in the window
        peak_idx = np.argmax(np.abs(window))
        peak_value = window[peak_idx]
        peak_lag = window_lags[peak_idx]
        if plot: 
            # Plot cross-correlation
            plt.figure(figsize=(10, 5))
            plt.plot(window_lags, window, label='Cross-correlation')
            plt.axvline(x=peak_lag, color='r', linestyle='--', label=f'Peak at {peak_lag:.2f}s')
            plt.scatter([peak_lag], [peak_value], color='r')
            plt.title(f'Cross-correlation between z_{col1} and z_{col2}')
            plt.xlabel('Lag (s)')
            plt.ylabel('Cross-correlation')
            plt.legend()
            plt.grid()
            plt.show()
        
        # Add to self.info
        if 'cross_correlation' not in self.info:
            self.info['cross_correlation'] = {}
        self.info['cross_correlation'][f'{col1}_{col2}'] = {
            'peak_value': peak_value,
            'peak_lag': peak_lag,
            'columns': (col1, col2)
        }
        # Report peak value
        peak_value = round(window[peak_idx], 2)
        peak_lag = round(window_lags[peak_idx], 2)
        print(f'Peak cross-correlation value: {peak_value} at lag: {peak_lag:.2f}s')
    

    def write_preprocessed_csv(self, Onix_align = True, motion = False):
        '''
        Saves the preprocessed data with renamed coluns and the events into csv files
        '''
        # Combine the base data
        final_df = pd.concat([self.data_seconds.reset_index(drop=True),
                              self.deltaF_F.reset_index(drop=True),
                              self.zscored.reset_index(drop=True),], axis=1)
        
        # Remove unnamed columns
        final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')]
        
        # Create rename mapping for dfF columns
        rename_dict = {
            '470_dfF': 'dfF_470',
            '560_dfF': 'dfF_560', 
            '410_dfF': 'dfF_410'
        }
        
        # Rename columns
        final_df = final_df.rename(columns=rename_dict)
        
        # Save with renamed columns
        final_df.to_csv(self.save_path + '/Processed_fluorescence.csv', index=False)
        print('Processed_fluorescence.csv saved')
        
        # Handle events if self.events exists
        if hasattr(self, 'events') and isinstance(self.events, pd.DataFrame) and (Onix_align ==False):
            # Save the events DataFrame separately NOTE unclear when would this ever be useful, these events are force-aligned to the 470 signal and not the original signal, so not useful for ONIX alignment
            self.events.to_csv(self.save_path + '/Events.csv', index=False)
            print('Events detected and saved.')
            
        if Onix_align ==True:
            print('Original Events.csv saved to Events.csv to be used for ONIX alingment')
            event_path = self.path + 'Events.csv'  # events with precise timestamps
            events = pd.read_csv(event_path)
            events.to_csv(self.save_path + '/Events.csv', index = False)

        mpl.pyplot.close()
        
    
        import matplotlib.patheffects as path_effects
        
        
    def plot_all_signals(self, sensors, plot_info=''):
        """
        Generates a comprehensive figure of signal processing steps on an A4 page, 
        including exponential fits, ΔF/F signals, Z-scored signals, and cross-correlation plots.
        Inputs:
            sensors (list): List of sensors used in the experiment.
            plot_info (str, optional): Additional information to include in the plot title. Default is an empty string.
        Outputs:
            Saves the generated figure to the specified save path.
        """
        # A4 dimensions and setup
        A4_WIDTH = 8.27
        A4_HEIGHT = 11.69
        SMALL_SIZE = 6
        MEDIUM_SIZE = 8
        TITLE_SIZE = 6  # Adjust this value to make the titles smaller
        
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
        plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        
        n_signals = len(self.signals.columns)
        n_rows = n_signals * 3 + 1  # Add one row for the cross-correlation plot
        
        # Create figure
        fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
        gs = fig.add_gridspec(n_rows, 1, hspace=0.5)
        
        for idx, signal in enumerate(self.signals.columns):
            # 1. Exponential fit
            ax1 = fig.add_subplot(gs[idx*3])
            line1 = ax1.plot(self.data_seconds['TimeStamp'], self.filtered[f'filtered_{signal}'],
                    color=self.colors[idx], alpha=1, label=f'Filtered {signal}', linewidth=0.5)
            line2 = ax1.plot(self.data_seconds['TimeStamp'], self.exp_fits[f'expfit_{signal[-3:]}'],
                    color='black', alpha=1, label='Exponential fit', linewidth=1)
            ax1.set_title(f'Exponential Fit - {sensors[signal[-3:]]}', fontsize=TITLE_SIZE)
            ax1.legend(loc='upper right')
            
            # 2. dF/F signal
            ax2 = fig.add_subplot(gs[idx*3 + 1])
            line3 = ax2.plot(self.data_seconds['TimeStamp'], self.deltaF_F[f'{signal[-3:]}_dfF'],
                    color=self.colors[idx], linewidth=0.2)
            ax2.set_title(f'ΔF/F Signal - {sensors[signal[-3:]]}', fontsize=TITLE_SIZE)
            
            # 3. Z-scored signal
            ax3 = fig.add_subplot(gs[idx*3 + 2])
            line4 = ax3.plot(self.data_seconds['TimeStamp'], self.zscored[f'z_{signal[-3:]}'],
                    color=self.colors[idx], linewidth=0.2)
            ax3.set_title(f'Z-scored Signal - {sensors[signal[-3:]]}', fontsize=TITLE_SIZE)
        
        
        # Add cross-correlation plot
        ax_cc = fig.add_subplot(gs[-1])
        col1, col2 = '470', '560'
        z_col1 = self.zscored[f'z_{col1}']
        z_col2 = self.zscored[f'z_{col2}']
        cross_corr = np.correlate(z_col1 - np.mean(z_col1), z_col2 - np.mean(z_col2), mode='full')
        lags = np.arange(-len(z_col1) + 1, len(z_col1))
        sampling_rate = 1 / np.mean(np.diff(self.data_seconds['TimeStamp']))
        window_size = int(5 * sampling_rate)
        center = len(cross_corr) // 2
        window = cross_corr[center - window_size:center + window_size + 1]
        window_lags = lags[center - window_size:center + window_size + 1] / sampling_rate
        peak_idx = np.argmax(np.abs(window))
        peak_value = round(window[peak_idx], 2)
        peak_lag = round(window_lags[peak_idx], 2)
        line5 = ax_cc.plot(window_lags, window, label='Cross-correlation')
        ax_cc.axvline(x=peak_lag, color='r', linestyle='--', label=f'Peak at {peak_lag:.2f}s')
        ax_cc.scatter([peak_lag], [peak_value], color='r')
        ax_cc.set_title(f'Cross-correlation between {sensors[col1]} and {sensors[col2]}', fontsize=TITLE_SIZE)
        ax_cc.set_xlabel('Lag (s)')
        ax_cc.set_ylabel('Cross-correlation')
        ax_cc.legend()
        ax_cc.grid()
        
        # Final formatting
        plt.xlabel('Time (s)')
        filter_method = self.info['filtering_method'][signal]
        detrend_method = self.info['detrend_method']
        motion_correction = self.info['motion_correction']
        fig_title = (
        f'Signal Processing Steps - {self.info["mousename"]} {plot_info}\n'
        f'Filtering: {filter_method}, Detrending: {detrend_method}, '
        f'Motion Correction: {motion_correction}'
        )
        fig.suptitle(f'Signal Processing Steps - {self.info["mousename"]} {plot_info}', fontsize=MEDIUM_SIZE + 2, y=0.98)
        # Add iso_ch and signal_ch if motion correction is True
        if motion_correction:
            iso_ch = self.info.get('motion_correction_isosbestic', 'N/A')
            signal_ch = self.info.get('motion_correction_signal', 'N/A')
            fig.text(0.5, 0.95, f'Filtering: {filter_method}, Detrending: {detrend_method}, Motion Correction: {motion_correction}, Iso: {iso_ch}, Signal: {signal_ch}', ha='center', fontsize=MEDIUM_SIZE)
        else:
            fig.text(0.5, 0.95, f'Filtering: {filter_method}, Detrending: {detrend_method}, Motion Correction: {motion_correction}', ha='center', fontsize=MEDIUM_SIZE)
        
        plt.tight_layout()
        
        # Save figures in multiple formats
        base_path = f'{self.save_path}/all_signals_{self.info["mousename"]}'
        plt.savefig(f'{base_path}.png', bbox_inches='tight', dpi=600)
        plt.savefig(f'{base_path}.svg', bbox_inches='tight', format='svg')
        plt.close()
        
        
    def extract_events(self): #FIXME any use for this?    
        """
        Assigns a boolean data column for each unique event type indicating 
        whether the event occurred at each time point (True/False). 
        Removes 'Event' and 'State' columns after processing.
        Saves the updated data DataFrame to self.data.
        
        Returns:
            DataFrame with 'Event' and 'State' columns (original values) before removal.
        """
        # Copy the data for processing
        data = self.data.copy()
    
        # List to hold unique event names for reference (optional)
        events = []
    
        # Check if the Event column exists and is not empty
        if 'Event' not in data.columns or data['Event'].isna().all():
            print("There are no recorded events.")

        else:
            events = pd.DataFrame()
            for col in data.columns:
                if 'event' in col:
                    events[col]= data[col]
            
        return events#data[[event for event in events]]

    #ratiometric dF/F function added sept. 2025
    def get_ratiometric_dfF(self, baseline='median', plot=True, savefig=True):
        """
        Compute ratiometric dF/F using 470 nm over 560 nm signals.
        
        F(t) = 470F(t) / 560F(t)
        dF/F(t) = (F(t) - F0) / F0

        Args:
            baseline (str or float): Method to calculate baseline F0.
                - 'median': uses the median of F(t)
                - 'mean': uses the mean of F(t)
                - float: uses a user-provided scalar baseline
            plot (bool): Whether to plot the ratiometric signal
            savefig (bool): Whether to save the plot

        Returns:
            pandas.DataFrame: DataFrame with ratiometric F and dF/F signals
        """
        # # Check that both channels exist
        # if '470' not in self.signals.columns or '560' not in self.signals.columns:
        #     raise ValueError("Both 470 and 560 channels must be available in self.signals.")

        # Compute raw ratio
        ratio = self.detrended['detrend_filtered_470'] / self.detrended['detrend_filtered_560']

        # Determine baseline F0
        if baseline == 'median':
            F0 = np.median(ratio)
        elif baseline == 'mean':
            F0 = np.mean(ratio)
        elif isinstance(baseline, (int, float)):
            F0 = baseline
        else:
            raise ValueError("Invalid baseline option. Choose 'median', 'mean', or a numeric value.")

        # Compute ΔF/F
        df_f = (ratio - F0) / F0

        # Save to object
        self.ratiometric = pd.DataFrame({
            'F_ratio': ratio,
            'dfF_ratio': df_f
        })

        if plot:
            fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            axs[0].plot(self.data_seconds['TimeStamp'], ratio, color='steelblue')
            axs[0].axhline(F0, color='red', linestyle='--', label=f'Baseline F0={F0:.3f}')
            axs[0].set_ylabel("F (470/560)")
            axs[0].legend()

            axs[1].plot(self.data_seconds['TimeStamp'], df_f, color='darkgreen')
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("ΔF/F (ratio)")
            axs[1].set_title("Ratiometric ΔF/F")

            plt.suptitle(f'Ratiometric ΔF/F for {self.info.get("mousename", "unknown")}')
            plt.tight_layout()

            if savefig:
                plt.savefig(self.save_path + f'/ratiometric_dfF_{self.info["mousename"]}.png', dpi=300)

            plt.show()
        
        return self.ratiometric


    def analyze_time_windows(self,
                            first_window_start=10,  # minutes
                            first_window_end=15,    # minutes
                            last_window_duration=5,  # minutes
                            signal_channel='470',
                            save_figure=True,
                            save_results_csv=True):
        """
        Analyze preprocessed photometry data and extract mean z-scores for specific time windows.
        This method works on the already-processed data in memory.
        
        Parameters:
        -----------
        first_window_start : float
            Start time of first window in minutes (default: 10)
        first_window_end : float
            End time of first window in minutes (default: 15)
        last_window_duration : float
            Duration of last window in minutes (default: 5)
        signal_channel : str
            Which signal channel to analyze (default: '470')
        save_figure : bool
            Whether to save a figure showing the analysis (default: True)
        save_results_csv : bool
            Whether to save results to CSV files (default: True)
            
        Returns:
        --------
        dict : Dictionary containing analysis results
        """
        
        print(f"\n{'='*80}")
        print(f"Analyzing time windows for preprocessed photometry data")
        print(f"{'='*80}\n")
        
        # Check if we have the required data
        if not hasattr(self, 'zscored') or not hasattr(self, 'data_seconds'):
            raise AttributeError("No z-scored data found. Run the full preprocessing pipeline first.")
        
        # Extract time and z-scored signal
        time_seconds = self.data_seconds['TimeStamp'].values
        z_signal_col = f'z_{signal_channel}'
        
        if z_signal_col not in self.zscored.columns:
            raise ValueError(f"Signal channel {signal_channel} not found in z-scored data. Available: {self.zscored.columns.tolist()}")
        
        z_signal = self.zscored[z_signal_col].values
        
        # Get mouse name
        mouse_name = self.info.get('mousename', 'unknown')
        
        # Get total recording duration
        total_duration_sec = time_seconds[-1]
        total_duration_min = total_duration_sec / 60
        
        print(f"\nRecording duration: {total_duration_min:.2f} minutes ({total_duration_sec:.2f} seconds)")
        print(f"Total data points: {len(time_seconds)}")
        print(f"Sampling rate: ~{len(time_seconds)/total_duration_sec:.2f} Hz")
        
        # Extract mean z-score for first window
        first_start_sec = first_window_start * 60
        first_end_sec = first_window_end * 60
        
        if total_duration_sec < first_end_sec:
            print(f"\nWARNING: Recording is too short for the {first_window_start}-{first_window_end} min window!")
            print(f"Recording ends at {total_duration_min:.2f} minutes")
            first_window_mean = np.nan
            first_window_std = np.nan
            first_window_median = np.nan
            first_75th = np.nan
            first_90th = np.nan
            first_max = np.nan
            first_n_samples = 0
        else:
            first_mask = (time_seconds >= first_start_sec) & (time_seconds <= first_end_sec)
            first_window_data = z_signal[first_mask]
            first_window_mean = np.mean(first_window_data)
            first_window_std = np.std(first_window_data)
            first_window_median = np.median(first_window_data)
            first_75th = np.percentile(first_window_data, 75)
            first_90th = np.percentile(first_window_data, 90)
            first_max = np.max(first_window_data)
            first_n_samples = len(first_window_data)
            
            print(f"\n{'='*80}")
            print(f"FIRST WINDOW ({first_window_start}-{first_window_end} min):")
            print(f"{'='*80}")
            print(f"Time range: {first_start_sec:.1f} - {first_end_sec:.1f} seconds")
            print(f"Data points: {first_n_samples}")
            print(f"Mean z-score: {first_window_mean:.4f}")
            print(f"Median z-score: {first_window_median:.4f}")
            print(f"Std z-score: {first_window_std:.4f}")
        
        # Extract mean z-score for last window
        last_start_sec = total_duration_sec - (last_window_duration * 60)
        
        if total_duration_sec < (last_window_duration * 60):
            print(f"\nWARNING: Recording is shorter than {last_window_duration} minutes!")
            last_window_mean = np.nan
            last_window_std = np.nan
            last_window_median = np.nan
            last_75th = np.nan
            last_90th = np.nan
            last_max = np.nan
            last_n_samples = 0
        else:
            last_mask = time_seconds >= last_start_sec
            last_window_data = z_signal[last_mask]
            last_window_mean = np.mean(last_window_data)
            last_window_std = np.std(last_window_data)
            last_window_median = np.median(last_window_data)
            last_75th = np.percentile(last_window_data, 75)
            last_90th = np.percentile(last_window_data, 90)
            last_max = np.max(last_window_data)
            last_n_samples = len(last_window_data)
            
            print(f"\n{'='*80}")
            print(f"LAST WINDOW (last {last_window_duration} min):")
            print(f"{'='*80}")
            print(f"Time range: {last_start_sec:.1f} - {total_duration_sec:.1f} seconds")
            print(f"Data points: {last_n_samples}")
            print(f"Mean z-score: {last_window_mean:.4f}")
            print(f"Median z-score: {last_window_median:.4f}")
            print(f"Std z-score: {last_window_std:.4f}")
        
        # Create visualization
        if save_figure:
            fig, ax = plt.subplots(figsize=(15, 6))
            
            # Plot full z-scored signal
            time_minutes = time_seconds / 60
            ax.plot(time_minutes, z_signal, color='steelblue', linewidth=0.5, alpha=0.7, 
                   label=f'Z-scored {signal_channel}')
            
            # Highlight first window
            if not np.isnan(first_window_mean):
                ax.axvspan(first_window_start, first_window_end, alpha=0.2, color='green', 
                          label=f'First window ({first_window_start}-{first_window_end} min)\nMean: {first_window_mean:.3f}')
                ax.axhline(first_window_mean, color='green', linestyle='--', linewidth=2,
                          xmin=(first_window_start/total_duration_min), xmax=(first_window_end/total_duration_min))
            
            # Highlight last window
            if not np.isnan(last_window_mean):
                last_start_min = last_start_sec / 60
                ax.axvspan(last_start_min, total_duration_min, alpha=0.2, color='orange',
                          label=f'Last window (last {last_window_duration} min)\nMean: {last_window_mean:.3f}')
                ax.axhline(last_window_mean, color='orange', linestyle='--', linewidth=2,
                          xmin=(last_start_min/total_duration_min), xmax=1.0)
            
            ax.set_xlabel('Time (minutes)', fontsize=14)
            ax.set_ylabel(f'Z-scored {signal_channel} signal', fontsize=14)
            ax.set_title(f'Photometry Time Window Analysis - {mouse_name}', fontsize=16)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            fig_path = os.path.join(self.save_path, f'{mouse_name}_time_windows_analysis.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {fig_path}")
            plt.close()
        
        # Calculate changes and percent changes
        change_mean = last_window_mean - first_window_mean
        change_75th = last_75th - first_75th
        
        if first_window_mean != 0:
            percent_change_mean = (change_mean / abs(first_window_mean)) * 100
        else:
            percent_change_mean = np.nan
        
        if first_75th != 0:
            percent_change_75th = (change_75th / abs(first_75th)) * 100
        else:
            percent_change_75th = np.nan
        
        # Compile results
        results = {
            'Mouse': mouse_name,
            'Signal_Channel': signal_channel,
            'First_5min_Mean': first_window_mean,
            'Last_5min_Mean': last_window_mean,
            'First_5min_Median': first_window_median,
            'Last_5min_Median': last_window_median,
            'First_5min_75th_Percentile': first_75th,
            'Last_5min_75th_Percentile': last_75th,
            'First_5min_90th_Percentile': first_90th,
            'Last_5min_90th_Percentile': last_90th,
            'First_5min_Max': first_max,
            'Last_5min_Max': last_max,
            'Change_Mean': change_mean,
            'Percent_Change_Mean': percent_change_mean,
            'Change_75th_Percentile': change_75th,
            'Percent_Change_75th_Percentile': percent_change_75th,
            'First_5min_N_Samples': first_n_samples,
            'Last_5min_N_Samples': last_n_samples,
            'Recording_Duration_Min': total_duration_min,
            'First_Window_Start_Min': first_window_start,
            'First_Window_End_Min': first_window_end,
            'Last_Window_Duration_Min': last_window_duration,
        }
        
        # Save results to CSV if requested
        if save_results_csv:
            # Create detailed results DataFrame
            results_df = pd.DataFrame([results])
            
            # Save detailed results
            output_filename = f"{mouse_name}_z{signal_channel}_time_windows.csv"
            output_path = os.path.join(self.save_path, output_filename)
            results_df.to_csv(output_path, index=False)
            print(f"\nDetailed results saved to: {output_path}")
            
            # Create summary in format matching original script
            summary_results = {
                'Mouse': mouse_name,
                'First_5min_Mean': results['First_5min_Mean'],
                'Last_5min_Mean': results['Last_5min_Mean'],
                'First_5min_Median': results['First_5min_Median'],
                'Last_5min_Median': results['Last_5min_Median'],
                'First_5min_75th_Percentile': results['First_5min_75th_Percentile'],
                'Last_5min_75th_Percentile': results['Last_5min_75th_Percentile'],
                'First_5min_90th_Percentile': results['First_5min_90th_Percentile'],
                'Last_5min_90th_Percentile': results['Last_5min_90th_Percentile'],
                'First_5min_Max': results['First_5min_Max'],
                'Last_5min_Max': results['Last_5min_Max'],
                'Change_Mean': results['Change_Mean'],
                'Percent_Change_Mean': results['Percent_Change_Mean'],
                'Change_75th_Percentile': results['Change_75th_Percentile'],
                'Percent_Change_75th_Percentile': results['Percent_Change_75th_Percentile'],
                'First_5min_N_Samples': results['First_5min_N_Samples'],
                'Last_5min_N_Samples': results['Last_5min_N_Samples'],
            }
            
            summary_df = pd.DataFrame([summary_results])
            
            # Save summary to parent directory (where mouse data folder is)
            parent_dir = os.path.dirname(os.path.dirname(self.save_path))
            summary_filename = f"{mouse_name}_z{signal_channel}_change.csv"
            summary_output_path = os.path.join(parent_dir, summary_filename)
            summary_df.to_csv(summary_output_path, index=False)
            print(f"Summary saved to: {summary_output_path}")
        
        # Print final summary
        print("\n" + "#"*80)
        print("#" + " "*78 + "#")
        print("#" + " "*22 + "TIME WINDOW ANALYSIS COMPLETE" + " "*27 + "#")
        print("#" + " "*78 + "#")
        print("#"*80)
        print(f"\nMouse: {mouse_name}")
        print(f"Signal: z_{signal_channel}")
        print(f"Recording duration: {total_duration_min:.2f} minutes")
        print(f"\nMean z-score ({first_window_start}-{first_window_end} min): {first_window_mean:.4f}")
        print(f"Mean z-score (last {last_window_duration} min): {last_window_mean:.4f}")
        print(f"\nChange (absolute): {change_mean:.4f}")
        print(f"Change (percent): {percent_change_mean:.2f}%")
        print("\n" + "#"*80)
        
        return results
