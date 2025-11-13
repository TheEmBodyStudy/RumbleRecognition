from math import floor
from sonpy import lib as sp
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk
import tempfile
import yaml


def spike_extraction(task, spike_file, participant_id, save_dir):
    
    """
    Function for extracting spike data from a specified spike file and saving it to a text file.

    Parameters
    ----------
    spike_file : str
        Path to the spike file that contains the data to be processed.
    save_dir : str
        Directory path where the output file with extracted data will be saved.

    Notes
    -----
    This function reads spike data from the given `spike_file` and processes it to extract two types of data:
    1. **Channel data**: Numerical data from ADC channels, which are rounded to 5 decimal places.
    2. **Text markers**: Time-based markers and their associated strings from text-mark channels.
    
    The extracted data for each channel is stored in a dictionary format with the following information:
    - Channel index
    - Sampling rate
    - Units of the data
    - Data values (either numerical or text markers)

    Finally, the function saves the extracted data in a formatted text file at the specified `save_dir` location.
    The file is named `channel_data_output.txt`, but this name may need to be adjusted for specific use cases (e.g., adding a participant number).

    Returns
    -------
    channel_data : dict
        A dictionary containing the extracted data for all channels, with channel titles as keys and relevant data (index, sampling rate, units, and data) as values.
    """
    
    # Initialise empty dictionary to store channel and text marker data
    channel_data = {}
    textmark_data = {
        "ticks": [],
        "markers": []
    }
    
    # Open file
    MyFile = sp.SonFile(spike_file, True)
    
    for i in range(MyFile.MaxChannels()):
        if MyFile.ChannelType(i) == sp.DataType.Adc:
            channel_idx = i
            channel_title = MyFile.GetChannelTitle(i)
            channel_rate = MyFile.GetIdealRate(i)
    
            dMaxTime = MyFile.ChannelMaxTime(i) * MyFile.GetTimeBase()
            bInputNumber = dMaxTime
            dPeriod = MyFile.ChannelDivide(i) * MyFile.GetTimeBase()
            nPoints = floor(bInputNumber / dPeriod)
            data = MyFile.ReadFloats(i, nPoints, 0)
    
            # Round numerical data to 5 decimal places
            rounded_data = [round(float(d), 5) for d in data]
    
            channel_data[channel_title] = {
                "channel_index": channel_idx,
                "sampling_rate": channel_rate,
                "units": MyFile.GetChannelUnits(i),
                "data": rounded_data
            }
    
        if MyFile.ChannelType(i) == sp.DataType.TextMark:
            channel_idx = i
            channel_title = MyFile.GetChannelTitle(i)
            array = MyFile.ReadTextMarks(i, 99999, 0) # Read first 99999 text markers
    
            for mark in range(len(array)):
                tick = round((array[mark].Tick) / 100)
                marker = array[mark].GetString()
                textmark_data["ticks"].append(tick) # Gets time in ticks
                textmark_data["markers"].append(marker) # Gets text string
    
            channel_data[channel_title] = {
                "channel_index": channel_idx,
                "sampling_rate": MyFile.GetIdealRate(i),
                "units": MyFile.GetChannelUnits(i),
                "data": textmark_data
            }
    
    output_file_path = save_dir + f"/{participant_id}_{task}_channel_data_output.txt" 
    
    # Save the final dictionary to a text file
    with open(output_file_path, "w") as output_file:
        for channel_title, channel_info in channel_data.items():
            output_file.write(f"Channel Title: {channel_title}\n")
            output_file.write(f"Channel Index: {channel_info['channel_index']}\n")
            output_file.write(f"Sampling Rate: {channel_info['sampling_rate']}\n")
            output_file.write(f"Units: {channel_info['units']}\n")
            output_file.write("Data:\n")
    
            if isinstance(channel_info['data'], dict):
                for key, values in channel_info['data'].items():
                    if isinstance(values[0], (int, float)):
                        formatted_values = [f"{v:.5f}" for v in values]
                    else:
                        formatted_values = values
                    output_file.write(f"{key}: {formatted_values}\n")
            else:
                for value in channel_info['data']:
                    if isinstance(value, (int, float)):
                        output_file.write(f"{value:.5f}\n")
                    else:
                        output_file.write(f"{value}\n")
    
    print(f"Channel data has been saved to {output_file_path}")

    return channel_data



def spike_extraction_df(task, spike_file, participant_id, save_dir, plots = True):
    """
    Function for extracting spike data, converting it into a DataFrame, and saving trimmed data and metadata.

    Parameters
    ----------
    task : str
        The task name (used for saving metadata with task-specific names).
    spike_file : str
        Path to the spike file containing the data to be processed.
    save_dir : str
        Directory path where the output files will be saved.

    Notes
    -----
    This function extends the `spike_extraction` function to create a structured DataFrame from the extracted spike data.
    It performs the following operations:
    1. **Extracts data**: It uses `spike_extraction` to retrieve channel data from the spike file.
    2. **Creates a DataFrame**: 
        - The function creates a DataFrame with a `ticks_index` as a common index across all channels.
        - Amplitude data for numerical channels is added as columns.
        - Marker channel data (text markers) is merged based on the common `ticks_index`.
    3. **Calculates raw time**: A `raw_time_s` column is created based on the highest `ticks_index` divided by 1000 (to convert to seconds).
    4. **Trims data**: 
        - The DataFrame is trimmed based on the indices of "START TASK" and "END TASK" markers.
        - A new column, `time_since_start_s`, is added, along with `time_since_start_min` for convenience.
    5. **Saves DataFrames**:
        - The untrimmed DataFrame is saved to a CSV file (`untrimmed_channel_data.csv`).
        - The trimmed DataFrame is saved to a CSV file (`trimmed_channel_data.csv`).
    6. **Saves metadata**: A YAML file containing metadata about each channel (index, sampling rate, and units) is saved in the specified `save_dir`.

    Returns
    -------
    trimmed_df : pandas.DataFrame
        The trimmed DataFrame containing data between the "START TASK" and "END TASK" markers, with additional time-related columns.
    
    Example usage: 
    -------
    task = 'GASTRIC'
    spike_file = '/Users/hsavage/036_gastric.smrx'
    save_dir = '/Users/hsavage/Desktop/'
    trimmed_df = spike_extraction_df(task, spike_file, save_dir)
    """
    
    channel_data = spike_extraction(task, spike_file, participant_id, save_dir)
    
    #Create the base DataFrame with the common index
    common_index = max(len(info["data"]) for info in channel_data.values() if isinstance(info["data"], list))
    df = pd.DataFrame({"ticks_index": range(common_index)})

    #Add amplitude data for each channel
    for channel_title, channel_info in channel_data.items():
        if isinstance(channel_info["data"], list):  # Check if this is a numerical channel
            amplitude_series = pd.Series(channel_info["data"], index=range(len(channel_info["data"])))
            df[f"{channel_title}_amplitude"] = amplitude_series

    #Add marker channel data
    for channel_title, channel_info in channel_data.items():
        if isinstance(channel_info["data"], dict) and "ticks" in channel_info["data"]:  # Marker channel
            markers_df = pd.DataFrame({
                "ticks_index": channel_info["data"]["ticks"],
                "markers": channel_info["data"]["markers"]
            })
            df = pd.merge(df, markers_df, on="ticks_index", how="left")

    #Calculate raw_time based on the highest sampling rate
    #highest_sampling_rate = max(
    #    channel_info["sampling_rate"] for channel_title, channel_info in channel_data.items() 
    #    if "markers" not in channel_info["data"]
    #)
    
    df["raw_time_s"] = df["ticks_index"] / 1000
    
    #Save untrimmed df
    untrimmed_file_name = save_dir + f"{participant_id}_" + task + '_untrimmed_channel_data.tsv'
    df.to_csv(untrimmed_file_name, sep = '\t', index=False)
    # Locate the index values for START TASK and END TASK + Trim the DataFrame based on index range
    if task == 'Heartbeat Task':
        start_index = df[df["markers"] == "M_CS"].index[-1]
        end_index = df[df["markers"] == "M_CE"].index[0]
    # elif task == 'EMOTION':

    #     # If task initialised before recording started, set start_index to 0
    #     if df[df["markers"].str.contains('START_EXPT', case=False, na=False)].empty:
    #         start_index = 0
    #     else:
    #         start_index = df[df["markers"].str.contains('START_EXPT', case=False, na=False)].index[-1]
        
    #     # If participant did not complete the task, set end_index to the last index
    #     if df[df["markers"].str.contains('END_EXPT', case=False, na=False)].empty:
    #         end_index = df.index[-1]
    #     else:
    #         end_index = df[df["markers"].str.contains('END_EXPT', case=False, na=False)].index[0]
    else:
        # If task initialised before recording started, set start_index to 0
        if df[df["markers"].str.contains('START_EXPT', case=False, na=False)].empty:
            start_index = 0
        else:
            start_index = df[df["markers"].str.contains('START_EXPT', case=False, na=False)].index[-1]
        
        # If participant did not complete the task, set end_index to the last index
        if df[df["markers"].str.contains('END_EXPT', case=False, na=False)].empty:
            end_index = df.index[-1]
        else:
            end_index = df[df["markers"].str.contains('END_EXPT', case=False, na=False)].index[0]

    trimmed_df = df.loc[start_index:end_index].copy()
    # Add a new column 'time_since_start' to the trimmed DataFrame
    raw_time_column = "raw_time_s"
    start_time = df.loc[start_index, raw_time_column]
    trimmed_df.loc[:, "time_since_start_s"] = trimmed_df[raw_time_column] - start_time
    trimmed_df.loc[:, "time_since_start_min"] = trimmed_df["time_since_start_s"] / 60

    # Print and save the trimmed DataFrame
    #print(trimmed_df)
    trimmed_file_name = save_dir + f"{participant_id}_" + task + '_trimmed_channel_data.tsv'
    trimmed_df.to_csv(trimmed_file_name, sep = '\t', index=False)

    preprocessed_file_name = save_dir + f"{participant_id}_" + task + '_preprocessed_channel_data.tsv'
    trimmed_df.to_csv(preprocessed_file_name, sep = '\t', index=False)
    
    #Save metadata to YAML
    metadata = {
        channel_title: {
            "channel_index": channel_info["channel_index"],
            "sampling_rate": channel_info["sampling_rate"],
            "units": channel_info["units"]
        }
        for channel_title, channel_info in channel_data.items()
    }
    output_yaml_path = save_dir + f"{participant_id}_" + task + '_spike_channel_info.yaml'
    with open(output_yaml_path, "w") as yaml_file:
        yaml.dump(metadata, yaml_file, default_flow_style=False)


    if plots:
        if task == 'Rumble Recognition':
            amplitude_columns = [
            "ECG_amplitude",
            "EGG_amplitude",  
            "BB1_amplitude", 
            "BB2_amplitude",
            "stethoscope_amplitude",
            ]
        elif task == 'Respiration':
            amplitude_columns = [
            "ECG_amplitude",
            "EGG_amplitude",  
            "BB1_amplitude", 
            "BB2_amplitude",
            "spirometer_amplitude",
            "fiSYS_amplitude",
            "fiDIA_amplitude",
            "finoHR_amplitude",
            "PhysioCal_amplitude",
            ]
        elif task in ['Active Stand', 'Video Task', 'Social Judgement']:
            amplitude_columns = [
            "ECG_amplitude",
            "EGG_amplitude",  
            "BB1_amplitude", 
            "BB2_amplitude",
            "fiSYS_amplitude",
            "fiDIA_amplitude",
            "finoHR_amplitude",
            "PhysioCal_amplitude",
            ]
        else:
            print("No plots available for this task.")

        time_column = "time_since_start_min"
        
        # Find the minimum and maximum values across all amplitude columns
        min_amplitude = min(trimmed_df[amplitude_column].min() for amplitude_column in amplitude_columns)
        max_amplitude = max(trimmed_df[amplitude_column].max() for amplitude_column in amplitude_columns)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(trimmed_df[time_column], trimmed_df[amplitude_columns[0]], label=amplitude_columns[0], color="blue", linestyle="-")
        plt.plot(trimmed_df[time_column], trimmed_df[amplitude_columns[1]], label=amplitude_columns[1], color="red", linestyle="-")
        plt.plot(trimmed_df[time_column], trimmed_df[amplitude_columns[2]], label=amplitude_columns[2], color="orange", linestyle="-")
        plt.plot(trimmed_df[time_column], trimmed_df[amplitude_columns[3]], label=amplitude_columns[3], color="green", linestyle="-")
        plt.plot(trimmed_df[time_column], trimmed_df[amplitude_columns[4]], label=amplitude_columns[4], color="purple", linestyle="-")
        
        # Add vertical lines at marker positions
        marker_times = trimmed_df.loc[trimmed_df["markers"].notna(), time_column]
        for time in marker_times:
            plt.axvline(x=time, color="red", linestyle="--", linewidth=1, alpha=0.7)
            
        # Set y-axis limits
        plt.ylim(min_amplitude, max_amplitude)
        
        # Labels and title
        plt.xlabel("Raw Time (s)", fontsize=12)
        plt.ylabel("Amplitude", fontsize=12)
        plt.title("Amplitude vs Raw Time", fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=10, loc="upper right")
        
        # Show the plot
        plt.show()

        
    return preprocessed_file_name, output_yaml_path


