import numpy as np
import pandas as pd

import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import os

def get_offset_timecodes(search_string, offset_seconds):
    # Read the .ods file.
    df = pd.read_excel(r"C:\Users\olath\Downloads\elite_time_codes.ods", engine="odf")
    
    # Filter rows where "Event Description" contains the specified string. 
    filtered_rows = df[df['Event Description'].str.contains(search_string, na=False, regex=False)]
    
    # Extract "Seconds in video" and add the offset.
    offset_values = filtered_rows['Seconds in video'] + offset_seconds
    
    # Convert the updated seconds to 'hh:mm:ss' format.
    offset_timecodes = []
    for seconds in offset_values:
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        offset_timecodes.append(f"{hours:02}:{minutes:02}:{seconds:02}")
    
    return offset_timecodes

def extract_data_from_timestamps(input_csv, timestamps):
    # Load the CSV file into a DataFrame.
    data = pd.read_csv(input_csv)
    
    # Initialize list to store the extracted data for each timestamp.
    extracted_data_list = []

    # Loop over each timestamp.
    for timestamp in timestamps:
        # Find the index of the equivalent timestamp in the dataset.
        idx = data[data['DateTime'] == '2023-07-18 '+timestamp+'.999+00:00'].index[0]

        # Extract given number of rows before and after the found index.
        extracted_data = data.iloc[idx - 640: idx + 640]

        # Append the extracted data to the list.
        extracted_data_list.append(extracted_data)

    # Concatenate all extracted dataframes.
    combined_data = pd.concat(extracted_data_list)

    # Return the combined data. 
    return combined_data


def create_and_save_spectrograms(df, window_size, step_size, filename):
    # Making directory if not exists.
    if not os.path.exists(filename):
        os.makedirs(filename)

    # Extracting accelerometer readings.
    accel_x = df[' Vert Accelerometer'].to_numpy()
    accel_y = df[' Lat Accelerometer'].to_numpy()
    accel_z = df[' Long Accelerometer'].to_numpy()
    
    # Number of windows.
    num_windows = (len(df) - window_size) // step_size + 1
    
    # Sampling rate of the accelerometer.
    fs = 256  # Hz
    
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        # Extracting windowed data.
        window_x = accel_x[start_idx:end_idx]
        window_y = accel_y[start_idx:end_idx]
        window_z = accel_z[start_idx:end_idx]
        
        # Creating spectrograms.
        f_x, t_x, Sxx_x = spectrogram(window_x, window = 'hamming', fs=fs, nperseg = 128)
        f_y, t_y, Sxx_y = spectrogram(window_y, window = 'hamming', fs=fs, nperseg = 128)
        f_z, t_z, Sxx_z = spectrogram(window_z, window = 'hamming', fs=fs, nperseg = 128)

        # Concatenating spectrograms.
        combined_spectrogram = np.concatenate((Sxx_x, Sxx_y, Sxx_z), axis=-1)
        
        # Save each combined spectrogram as a separate file.
        
        plt.figure(figsize=(6, 6))
        plt.imshow(10 * np.log10(combined_spectrogram), aspect='auto', cmap='viridis')
        
        # Turn off axis numbers and ticks.
        plt.axis('off')
        
        filepath = os.path.join(filename, filename+f'_{i}.png')
        
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
        plt.close()



input_csv_path = r'C:\Users\olath\Downloads\transfer_77906_files_034aaaa9\Trimmed_Synched_LinearDriftCorrect\Elite1-acc_eq-linear_drift_correct.csv'

example_search_string = "*01A"
#Correct offset of EQ is 57703
example_offset_seconds = 57715

window_size = 1280
step_size = 1280
out_file_name = 'elite1_eq_nonspikes_5sec'


timestamps = get_offset_timecodes(example_search_string, example_offset_seconds)
#print(timestamps)
extracted_data = extract_data_from_timestamps(input_csv_path, timestamps)
#print(extracted_data)
create_and_save_spectrograms(extracted_data[[" Vert Accelerometer", " Lat Accelerometer", " Long Accelerometer"]], window_size, step_size, out_file_name)


