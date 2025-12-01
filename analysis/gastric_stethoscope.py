from math import floor
from sonpy import lib as sp
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns
import neurokit2 as nk
import tempfile
import yaml

from spike_extraction_trim import spike_extraction, spike_extraction_df


def gastric_stethoscope_clean_plot(df, output_yaml_path, save_dir, figsize=(20, 6)):
    try:        
        #Load channel data
        signal_data = df['stethoscope_amplitude'].dropna().values  # Get channel data, dropping NaNs
        
        #Load sample rate from yaml file
        with open(output_yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        sample_rate = config['stethoscope']['sampling_rate']
        
        
        ## FILTERING
        # Apply a notch filter at 50 Hz to remove mains interference
        nyquist = 0.5 * sample_rate
        notch_freq = 50.0
        notch_w0 = notch_freq / nyquist
        b_notch, a_notch = signal.iirnotch(notch_w0, Q=30)  # Quality factor of 30
        notched_signal = signal.filtfilt(b_notch, a_notch, signal_data)
    
        # Bandpass filter specifications (100 Hz to 500 Hz)
        low = 100
        high = 499
        b, a = signal.butter(3, [low, high], btype='bandpass', fs=sample_rate, analog=False)
        notched_butter_signal = signal.filtfilt(b, a, notched_signal)
        
        # Remove the mean from the cleaned signal
        notched_butter_demean_signal = notched_butter_signal - np.mean(notched_butter_signal)
    
        df['stethoscope_amplitude_clean'] = notched_butter_demean_signal
        cleaned_file_name = save_dir + 'GASTRIC_stethoscope_cleaned_channel_data.tsv'
        df.to_csv(cleaned_file_name, sep = '\t', index=False)
        #print(df)

        # Define keywords for categories
        category_keywords = {
            'START': ['start', ],   # Anything that signals start of trial
            'INHALE1': ['Inhale1'],
            'INHALE2': ['Inhale2'],
            'RESPONSE': ['Response'],
            'CONFIDENCE': ['confidence']
        }

        # Function to categorize a marker dynamically
        def categorize_marker(marker):
            for category, keywords in category_keywords.items():
                if any(keyword in marker for keyword in keywords):
                    return category
            return 'OTHER'

        # Apply the function to the markers column
        df['category'] = df['markers'].apply(categorize_marker)

        # Optional: filter only main categories
        filtered_df = df[df['category'] != 'OTHER']
        
        # Task markers (these come from the 'markers' column)
        task_markers = {
            'LIVE START': 'live_start_end',
            'LIVE END': 'live_start_end',
            'REC START': 'rec_start_end',
            'REC END': 'rec_start_end',
            'START TASK': 'task_start',
            'END TASK': 'task_end',
            'QUESTION': 'question',
        }
        
        # Map markers in 'markers' column to task categories
        df['category'] = df['markers'].map(task_markers).fillna('other')
        
        # Filter out 'other' categories (for tasks)
        filtered_df = df[df['category'] != 'other']
        
        # Identify intervals for shading
        shading_intervals = {'live_start_end': [], 'rec_start_end': []}
        
        start_markers = {'live_start_end': 'LIVE START', 'rec_start_end': 'REC START'}
        end_markers = {'live_start_end': 'LIVE END', 'rec_start_end': 'REC END'}
        
        for category, start_marker in start_markers.items():
            end_marker = end_markers[category]
            
            # Find start and end times for each category
            starts = filtered_df[filtered_df['markers'] == start_marker]['time_since_start_min']
            ends = filtered_df[filtered_df['markers'] == end_marker]['time_since_start_min']
            
            # Pair start and end times for shading
            shading_intervals[category] = list(zip(starts, ends))
        
        # Marker positions for tasks and questions
        marker_positions = list(filtered_df['time_minutes'])
        
        # Plot
        plt.figure(figsize = figsize)
        plt.title('Original vs Cleaned Stethoscope Data')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Amplitude')
        
        # Add transparent shading for intervals
        color_map = {
            'live_start_end': '#3e3e3e',
            'rec_start_end': '#a1a1a1',
            'question': 'grey',
            'task_start': 'black',  
            'task_end': 'black'   
        }
        
        for category, intervals in shading_intervals.items():
            for start, end in intervals:
                plt.axvspan(start, end, color=color_map[category], alpha=0.2)
        
        # Add vertical lines for task markers
        for _, row in filtered_df.iterrows():
            if row['category'] == 'question':
                plt.axvline(x=row['time_since_start_min'], color=color_map['question'], linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Plot the original and cleaned signal
        plt.plot(df['time_since_start_min'].dropna().values, df['stethoscope_amplitude'].dropna().values, label='Original Signal', color='#ceebc7', alpha=0.6, lw=1.0)
        plt.plot(df['time_since_start_min'], notched_butter_demean_signal, label='Filtered Signal', color='#51C151', alpha=1.0, lw=0.2)
        plt.xlim(min(marker_positions), max(marker_positions))
        plt.ylim(min(df['stethoscope_amplitude'].dropna().values), max(df['stethoscope_amplitude'].dropna().values))

        #legend
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color=color_map['live_start_end'], lw=2, label='LIVE START/END'))
        handles.append(plt.Line2D([0], [0], color=color_map['rec_start_end'], lw=2, label='REC START/END'))
        handles.append(plt.Line2D([0], [0], color=color_map['question'], linestyle='--', lw=2, label='QUESTION'))
        labels.append('LIVE START/END')
        labels.append('REC START/END')
        labels.append('QUESTION')
        plt.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()

        plt_name = 'GASTRIC_stethoscope.png'
        plt_save_path = save_dir +  '/'+ plt_name
        plt.savefig(plt_save_path, dpi=300, bbox_inches='tight')
        #plt.show()

        return df, plt_save_path

    except Exception as e:
        # Handle exception and create fallback image
        print(f"An error occurred: {e}")
        df = df
        plt_name = 'GASTRIC_stethoscope.png'

        # Create FAILED figure with green shading
        plt.figure(figsize=figsize, facecolor='white')
        plt.gca().set_facecolor('white')
        plt.text(0.5, 0.5, f"Data to produce {plt_name} not found",
                 fontsize=14, ha='center', va='center', color='#ceebc7')

        # Save the fallback image
        fallback_path = 'GASTRICstethoscope_FAIL_image.png'
        plt_save_path = save_dir +  '/' + fallback_path
        plt.axis('off') 
        plt.savefig(plt_save_path, bbox_inches='tight', facecolor='white')
    
        return df, plt_save_path



def gastric_compute_freq(df, save_dir, figsize=(20,4)):
    try:
        
        #Load channel data
        signal_data = df['stethoscope_amplitude_clean']
        print(signal_data)
        # Compute the FFT
        fft_result = np.fft.fft(signal_data)
        
        # Compute the frequencies corresponding to the FFT
        frequencies = np.fft.fftfreq(len(signal_data), d=0.001)  # d= sampling interval
        
        # Compute the amplitude spectrum
        amplitude = np.abs(fft_result)
        
        # Plot frequency vs. amplitude
        plt.figure(figsize=figsize)
        plt.plot(frequencies[:len(frequencies)//2], amplitude[:len(frequencies)//2], lw = 0.2, color='#51C151')  # Plot only positive frequencies
        plt.title("Frequency vs Amplitude")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        
        plt.tight_layout()

        plt_name = 'GASTRIC_stethoscope_freq.png'
        plt_save_path = save_dir +  '/'+ plt_name
        plt.savefig(plt_save_path, dpi=300, bbox_inches='tight')
        #plt.show()

    
    except Exception as e:
        # Handle exception and create fallback image
        print(f"An error occurred: {e}")
        
        plt_name = 'GASTRIC_stethoscope_freq.png'

        # Create FAILED figure with green shading
        plt.figure(figsize=figsize, facecolor='white')
        plt.gca().set_facecolor('white')
        plt.text(0.5, 0.5, f"Data to produce {plt_name} not found",
                 fontsize=14, ha='center', va='center', color='#ceebc7')

        # Save the fallback image
        fallback_path = 'GASTRIC_stethoscope_freq_FAIL_image.png'
        plt_save_path = save_dir +  '/' + fallback_path
        plt.axis('off') 
        plt.savefig(plt_save_path, bbox_inches='tight', facecolor='white')
    
        return plt_save_path

##USAGE:

#task = 'GASTRIC'
#spike_file = '/Users/hsavage/OneDrive - University College London/EMBODY/Tasks/gastric_pilots/036_gastric.smrx'
#save_dir = '/Users/hsavage/Desktop/MultipleSub/sub-001/'
#trimmed_df, output_yaml_path = spike_extraction_df(task, spike_file, save_dir)
#print(trimmed_df)
#print(output_yaml_path)
#df, plt_save_path = gastric_stethoscope_clean_plot(trimmed_df, output_yaml_path, save_dir)
#plt_save_path = gastric_compute_freq(df, save_dir)