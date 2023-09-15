# -*- coding: utf-8 -*-
# relative_duration.py
"""
Author:   Raghuveer Parthasarathy
Created on Sun Jun 25 17:03:10 2023
Last modified on September 15, 2023

Description
-----------

Contains a function to read the summary behavior analysis CSV and return the 
relative durations for each behavior for each dataset (total time relative
to total experiment duration), as well as the mean and std. dev. across 
datasets. (Last rows.)

"main" asks for file name information, and calls the above function.

"""

import csv
import numpy as np
import os

def get_mean_std_overRows(rows, colIdx):
    """
    At column colIdx, get all values from rows, calc. mean and std dev
    For mean, std. dev, ignore values that are NaN or Inf

    """
    vals = np.array([row[colIdx] for row in rows[1:]]).astype(float)
    OKvals = vals[np.isfinite(vals)]
    mean_vals = np.mean(OKvals)
    std_vals = np.std(OKvals)
    return (vals, mean_vals, std_vals)
    
def calc_relative_durations_csv(input_file, output_file, fps=25.0, 
                                startCol = np.array([]), 
                                endCol = np.array([])):
    """
    Inputs: 
    input_file : The csv file noting behavior counts and duration for each dataset,
                 created by the main behavior program. Likely name 
                 'behavior_count.csv', but may be good to rename (manually).
    output_file : filename for the output CSV file. Again, each row = each dataset.
                  Columns = each behavior. Last rows are summary mean, std. 
                  Suggested name: 'behavior_relDurations.csv'
    fps : frame rate of experiments (frames/sec), 
                  for proper normalization of durations. Probably 25.0
    startCol : First column Index No of Duration data in CSV file; 
               default is to examine the header rows for "Duration"
    endCol : last column Index No +1 of Duration data in CSV file 
               default is to examine the header rows for "Duration"
                  
    Output:
        Writes a CSV file, with filename output_file
    
    
    Code mostly from ChatGPT3.5 with two manual fixes. June 25, 2023.
    See notes (assess_relative_durations_notes_25June2023.txt)
    
    """
    # Read the input CSV file
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    # Extract header, dataset, difference in fish length, inter-fish distance,
    # and angle cross-correlation stats, and calc. stats on these
    header = rows[0]
    Ndatasets = len(rows)-1
    print(' ')
    print('Number of datasets: ', Ndatasets)
    dataset = [row[0] for row in rows[1:]]
    
    # Determine start, end columns if not input
    durationIndices = [index for index, item in enumerate(header) if "duration" in item.lower()]
    startCol = durationIndices[0]
    endCol = durationIndices[-1]        

    (length_delta, mean_length_delta, std_length_delta) = get_mean_std_overRows(rows, 2)
    (interfish_distance, mean_interfish_distance, std_interfish_distance) = get_mean_std_overRows(rows, 3)
    (xcorr_mean, mean_xcorr_mean, std_xcorr_mean) = get_mean_std_overRows(rows, 4)
    #(xcorr_std, mean_xcorr_std, std_xcorr_std) = get_mean_std_overRows(rows, 5)
    #(xcorr_skew, mean_xcorr_skew, std_xcorr_skew) = get_mean_std_overRows(rows, 6)
    
    
    # Calculate relative_duration for each behavior in each row
    relative_duration = np.array([[float(val) / float(row[1]) / fps \
                                   for val in row[startCol:endCol]] for row in rows[1:]])
    
    # Extract modified header for the new CSV file
    new_header = [header[0]] + [header[2]] + [header[3]] + [header[4]] + \
        [col.replace(' Duration', '') for col in header[startCol:endCol]]
                 # deleted after [4] + [header[5]] + [header[6]] + \
    
    # Write the processed data to a new CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write modified header
        writer.writerow(new_header)
        
        # Write dataset and relative_duration rows
        for i in range(len(dataset)):
            writer.writerow([dataset[i]] + [length_delta[i]]  +
                            [interfish_distance[i]] + 
                            [xcorr_mean[i]]  + 
                            relative_duration[i].tolist())
        # + [xcorr_std[i]] + [xcorr_skew[i]]
        
        # Write blank row
        writer.writerow([])
        
        # Write mean row
        mean_row = ['Mean']
        mean_row += [mean_length_delta]
        mean_row += [mean_interfish_distance]
        mean_row += [mean_xcorr_mean]
        #mean_row += [mean_xcorr_std]
        #mean_row += [mean_xcorr_skew]
        mean_row += np.mean(relative_duration, axis=0).tolist()
        writer.writerow(mean_row)
        
        # Write standard deviation row
        std_dev_row = ['Std. Dev.']
        std_dev_row += [std_length_delta]
        std_dev_row += [std_interfish_distance]
        std_dev_row += [std_xcorr_mean]
        #std_dev_row += [std_xcorr_std]
        #std_dev_row += [std_xcorr_skew]
        std_dev_row += np.std(relative_duration, axis=0).tolist()
        writer.writerow(std_dev_row)

        # Write standard error of mean row
        sem_row = ['Std. Error of Mean']
        sem_row += [std_length_delta/np.sqrt(Ndatasets)]
        sem_row += [std_interfish_distance/np.sqrt(Ndatasets)]
        sem_row += [std_xcorr_mean/np.sqrt(Ndatasets)]
        #std_dev_row += [std_xcorr_std]   write for sem if uncommenting
        #std_dev_row += [std_xcorr_skew]
        sem_row += (np.std(relative_duration, axis=0) / np.sqrt(Ndatasets)).tolist()
        writer.writerow(sem_row)

if __name__ == '__main__':
    """
    Get input and output file names; call calc_relative_duration()
    """
    
    print('Current directory: ', os.getcwd())
    input_directory = input('Input directory containing .csv file (blank for current dir.): ')
    if input_directory != '':
        os.chdir(input_directory)
    
    input_file = input('Input CSV file name (blank for "behavior_count.csv"): ')
    if input_file == '':
        input_file = 'behavior_count.csv'
    output_file = input_file.replace('_count', '_relDuration') # default
    temp_output_file = input('Output CSV file name (blank for ' + output_file + ': ')
    if temp_output_file != '':
        output_file = temp_output_file
    fps = input('frames per second (blank for 25.0): ')
    if fps == '':
        fps = '25.0'
    fps = float(fps)

    calc_relative_durations_csv(input_file, output_file, fps)
