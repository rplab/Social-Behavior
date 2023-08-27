# -*- coding: utf-8 -*-
# calc_relative_duration.py
"""
Author:   Raghuveer Parthasarathy
Created on Sun Jun 25 17:03:10 2023
Last modified on August 26, 2023

Description
-----------

A function to read the summary behavior analysis CSV and return the 
relative durations for each behavior for each dataset (total time relative
to total experiment duration), as well as the mean and std. dev. across 
datasets. (Last rows.)

Inputs: 
input_file : The csv file noting behavior counts and duration for each dataset,
             created by the main behavior program. Likely name 
             'behavior_count.csv', but may be good to rename (manually).
output_file : filename for the output CSV file. Again, each row = each dataset.
              Columns = each behavior. Last rows are summary mean, std. 
              Suggested name: 'behavior_relDurations.csv'
fps : frame rate of experiments (frames/sec), 
              for proper normalization of durations. Probably 25.0
startCol : first column of Duration behaviors, default 22
endCol : last column+1 of Duration behaviors, default = 34 
              
Output:
    Writes a CSV file, with filename output_file


Code mostly from ChatGPT3.5 with two manual fixes. June 25, 2023.
See notes (assess_relative_durations_notes_25June2023.txt)
"""

import csv
import numpy as np

def make_relative_durations_csv(input_file, output_file, fps, 
                                startCol = 22, endCol = 36):
    # Read the input CSV file
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    # Extract header, dataset, difference in fish length, and inter-fish distance
    header = rows[0]
    dataset = [row[0] for row in rows[1:]]
    length_delta = np.array([row[2] for row in rows[1:]]).astype(float)
    mean_length_delta = np.mean(length_delta)
    std_length_delta = np.std(length_delta)
    interfish_distance = np.array([row[3] for row in rows[1:]]).astype(float)
    mean_interfish_distance = np.mean(interfish_distance)
    std_interfish_distance = np.std(interfish_distance)
    
    # Calculate relative_duration for each behavior in each row
    relative_duration = np.array([[float(val) / float(row[1]) / fps \
                                   for val in row[startCol:endCol]] for row in rows[1:]])
    
    # Extract modified header for the new CSV file
    new_header = [header[0]] + [header[2]] + [header[3]] + \
        [col.replace(' Duration', '') for col in header[startCol:endCol]]
    
    # Write the processed data to a new CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write modified header
        writer.writerow(new_header)
        
        # Write dataset and relative_duration rows
        for i in range(len(dataset)):
            writer.writerow([dataset[i]] + [length_delta[i]]  +
                            [interfish_distance[i]] + 
                            relative_duration[i].tolist())
        
        # Write blank row
        writer.writerow([])
        
        # Write mean row
        mean_row = ['Mean']
        mean_row += [mean_length_delta]
        mean_row += [mean_interfish_distance]
        mean_row += np.mean(relative_duration, axis=0).tolist()
        writer.writerow(mean_row)
        
        # Write standard deviation row
        std_dev_row = ['Std. Dev.']
        std_dev_row += [std_length_delta]
        std_dev_row += [std_interfish_distance]
        std_dev_row += np.std(relative_duration, axis=0).tolist()
        writer.writerow(std_dev_row)


if __name__ == '__main__':
    main()
