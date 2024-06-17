# -*- coding: utf-8 -*-
# test_reading.py
"""
Author:   Raghuveer Parthasarathy
Created on Thu Jun 15 21:02:33 2023
Last modified on Thu Jun 15 21:02:33 2023

Description
-----------
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from toolkit import *

cwd = os.getcwd() # Current working directory

folder_path, allCSVfileNames = get_CSV_folder_and_filenames() # Get folder containing CSV files
print(f'\n\n All {len(allCSVfileNames)} CSV files starting with "results": ')
print(allCSVfileNames)


# initialize a list of dictionaries
datasets = [{} for j in range(len(allCSVfileNames))]

os.chdir(folder_path)

arenaCentersPathName = 'C:/Users/Raghu/Documents/Experiments and Projects\Zebrafish behavior'
arenaCentersFilename = 'ArenaCenters_SocPref_3456.csv'
arenaCentersLocation = os.path.join(arenaCentersPathName, arenaCentersFilename)

for j, CSVfileName in enumerate(allCSVfileNames):
    print(j)
    print(allCSVfileNames[j])
    print(CSVfileName)
    datasets[j]["CSVfilename"] = CSVfileName
    datasets[j]["dataset_name"] = get_dataset_name(CSVfileName)

    arenaCenter = get_ArenaCenter(datasets[j]["dataset_name"], 
                                  arenaCentersLocation)

    if arenaCenter is None:
        print("No rows contain the input string.")
    else:
        sixth_column_value, seventh_column_value = arenaCenter
        print("Sixth column value:", sixth_column_value)
        print("Seventh column value:", seventh_column_value)


    
    N_columns = 26 # number of columns in CSV file
        
    # load the whole CSV file into a tuple of 2 arrays
    all_data_tuple = load_data(CSVfileName, 0, N_columns) 
    # make a single numpy array with all the data (all columns of CSV)
    all_data = np.zeros((all_data_tuple[0].shape[0], 
                        all_data_tuple[0].shape[1], 
                        len(all_data_tuple)))
    all_data[:,:,0] = all_data_tuple[0]
    all_data[:,:,1] = all_data_tuple[1]
    frameArray = all_data_tuple[2]
    Nframes = len(frameArray)
    print('Number of frames: ', Nframes)

print('Hello')
print(datasets[1]["dataset_name"])

os.chdir(cwd)
