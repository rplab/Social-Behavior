# -*- coding: utf-8 -*-
# calc_relative_duration.py
"""
Author:   Raghuveer Parthasarathy
Created on Sun Jun 25 17:03:10 2023
Last modified on Sun Jun 25 17:03:10 2023

Description
-----------

A function to read the summary behavior analysis CSV and return the 
relative durations for each behavior for each dataset (total time relative
to total experiment duration), as well as the mean and std. dev. across 
datasets. (Last rows.)

Code from ChatGPT3.5 with two manual fixes. June 25, 2023.
See notes (assess_relative_durations_notes_25June2023.txt)
"""

import csv
import numpy as np

def process_csv(input_file, output_file, fps):
    # Read the input CSV file
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    # Extract header, dataset, and inter-fish distance
    header = rows[0]
    dataset = [row[0] for row in rows[1:]]
    interfish_distance = np.array([row[2] for row in rows[1:]]).astype(float)
    mean_interfish_distance = np.mean(interfish_distance)
    std_interfish_distance = np.std(interfish_distance)
    
    # Calculate relative_duration for each behavior in each row
    relative_duration = np.array([[float(val) / float(row[1]) / fps for val in row[13:23]] for row in rows[1:]])
    
    # Extract modified header for the new CSV file
    new_header = [header[0]] + [header[2]] + \
        [col.replace(' Duration', '') for col in header[13:23]]
    
    # Write the processed data to a new CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write modified header
        writer.writerow(new_header)
        
        # Write dataset and relative_duration rows
        for i in range(len(dataset)):
            writer.writerow([dataset[i]] + [interfish_distance[i]] + \
                            relative_duration[i].tolist())
        
        # Write blank row
        writer.writerow([])
        
        # Write mean row
        mean_row = ['Mean']
        mean_row += [mean_interfish_distance]
        mean_row += np.mean(relative_duration, axis=0).tolist()
        writer.writerow(mean_row)
        
        # Write standard deviation row
        std_dev_row = ['Std. Dev.']
        std_dev_row += [std_interfish_distance]
        std_dev_row += np.std(relative_duration, axis=0).tolist()
        writer.writerow(std_dev_row)


