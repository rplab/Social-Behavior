# !/usr/bin/env python3  
# IO_toolkit.py
# -*- coding: utf-8 -*- 
"""
Author:   Raghuveer Parthasarathy
Created on Mon Aug 25 20:59:37 2025
Last modified Sept. 9, 2025 -- Raghuveer Parthasarathy

Description
-----------

Author:   Raghuveer Parthasarathy
Version ='2.0': 
First version created by  : Estelle Trieu, 9/7/2022
Major modifications by Raghuveer Parthasarathy, May-July 2023
Last modified by Raghuveer Parthasarathy, October 5, 2025

Module containing functions for handling data files, configuration
files, and output files -- reading and writing.
Formerly in toolkit.py

"""

import numpy as np
import csv
import os
from pathlib import Path
import pickle
import pandas as pd
import yaml
import tkinter as tk
import tkinter.filedialog as filedialog
from time import perf_counter
import re

from toolkit import get_Nfish

# Function to get a valid path from the user (base Path or config path)
def get_basePath():
    """
    Ask the user for the "base" Path that either contains all the 
    CSV trajectory files or that contains "subgroup" folders 
    with the subgroup CSV files.  Either text input or, if blank,
    provide a dialog box.
    Verify that the basePath contains:
        expt_config.yaml, analysis_parameters.yaml, CSVcolumns.yaml

    """
    print('\n\nSelect the "CSV files" folder for this experiment')
    while True:
        user_input = input("Type the full path name; leave empty for dialog box: ")
        if user_input.lower() == '':
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            selected_path = tk.filedialog.askdirectory(title="Select CSV files folder")
        else:
            selected_path = user_input
        if os.path.isdir(selected_path):
            # It's a valid path
            # Check that it contains the required config files
            if not (os.path.isfile(os.path.join(selected_path, "expt_config.yaml")) and
                    os.path.isfile(os.path.join(selected_path, "analysis_parameters.yaml")) and
                    os.path.isfile(os.path.join(selected_path, "CSVcolumns.yaml"))):
                print(f'\n\nFolder {selected_path} does not contain all three config files:')
                print('      expt_config.yaml, analysis_parameters.yaml, CSVcolumns.yaml')
                if not os.path.isfile(os.path.join(selected_path, "expt_config.yaml")):
                    print('expt_config.yaml is missing; maybe the filename has an "s" at the end?')
                print("Invalid path. Please try again.")
            else: 
                return selected_path
        else:
            print("Invalid path. Please try again.")


def get_loading_option():
    """
    Prompt the user to select a loading option (from CSVs or Pickle)
    
    Returns
    -------
    loading_option : str
        The selected loading option.
    """
    options = {
        1: "load_from_CSV",
        2: "load_from_pickle"
        # Add more options here as needed
    }
    
    print("\nLoading options:")
    print("   load_from_CSV -- Reads CSV files. Do this for the first analysis of an experiment")
    print("   load_from_pickle -- Loads trajectories from pre-existing pickle file; re-analyzes")
    print("\nSelect loading option:")
    for key, value in options.items():
        print(f"   {key}: {value} (default)" if key == 1 else f"   {key}: {value}")
    
    try:
        user_input = input("Enter the number corresponding to your choice: ")
        selected_option = int(user_input) if user_input else 1
    except ValueError:
        selected_option = 1
    
    loading_option = options.get(selected_option, "load_from_CSV")
    return loading_option


def get_valid_file(fileTypeString = 'Config File'):
    """
    Check if the file+path exists; if not, dialog box.

    Parameters
    ----------
    fileTypeString : String, to use in prompt text

    Returns
    -------
    selected_file : selected path+file

    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    while True:
        titleString = f'Select {fileTypeString}'
        selected_file = tk.filedialog.askopenfilename(title=titleString)
        if os.path.isfile(selected_file):
            return selected_file
        else:
            print(f"Invalid file. Please select a valid {fileTypeString}.")
             


def load_expt_config(config_path, config_file):
    """ 
    Loads the experimental configuration file
    Image scale and arena centers paths will be appended to config_path
        (same as basePath in main code).
    Inputs:
        config_path, config_file: path and file name of the yaml config file
    Outputs:
        expt_config : dictionary of configuration information
    """
    config_file_full = os.path.join(config_path, config_file)
    # Note that we have already checked that basePath contains all three 
    # config files, including expt_config.yaml
    # Check if the config file exists; dialog box if not
    
    with open(config_file_full, 'r') as f:
        all_config = yaml.safe_load(f)
    
    # There need only be one experimental configuration in the config file,
    # but allow for the possibility of more
    all_expt_names = list(all_config.keys())

    if len(all_expt_names)> 1:
        print('More...')
        print('\n\nALl experiments in this config file: ')
        for j, key in enumerate(all_expt_names):
            print(f'  {j}: {key}')
        expt_choice = input('Select experiment (name string or number): ')
        # Note that we're not checking if the choice is valid, i.e. if in 
        # all_expt_names (if a string) or if in 0...len(all_expt_names) (if 
        # a string that can be converted to an integer.)
        try:
            # Is the input string just an integer? Try integer...
            expt_config = all_config[all_expt_names[int(expt_choice)]]
        except:
            # Must be a string
            expt_config = all_config[all_expt_names[expt_choice]]
    else:
        expt_config = all_config[all_expt_names[0]]

    # Experiment Name; None if this isn't specified 
    if ("expt_name" not in expt_config.keys()):
        expt_config['expt_name'] = None
    
    # Image scale file name and path, if specified
    if ("imageScaleFilename" in expt_config.keys()) and \
        (expt_config['imageScaleFilename'] is not None):
        if ("imageScalePathName" in expt_config.keys()):
            if expt_config['imageScalePathName'] is not None:
                imageScalePathNameFull = os.path.join(config_path, 
                                                      expt_config['imageScalePathName'])
            else:
                imageScalePathNameFull = config_path
        expt_config['imageScaleLocation'] = os.path.join(imageScalePathNameFull, 
                                             expt_config['imageScaleFilename'])            
    else:
        expt_config['imageScaleLocation'] = None    

    if ("arenaCentersFilename" in expt_config.keys()):
        if expt_config['arenaCentersFilename'] != None:
            print('config_path: ', config_path)
            print('for arenaCentersPathName: ', 
                  expt_config['arenaCentersPathName'])
            print('for arenaCentersFilename: ', 
                  expt_config['arenaCentersFilename'])
            if expt_config['arenaCentersPathName'] is not None:
                expt_config['arenaCentersLocation'] = os.path.join(config_path,
                                                               expt_config['arenaCentersPathName'], 
                                                               expt_config['arenaCentersFilename'])
            else:
                expt_config['arenaCentersLocation'] = os.path.join(config_path,
                                                               expt_config['arenaCentersFilename'])
        else:
            expt_config['arenaCentersLocation'] = None
    else:
        expt_config['arenaCentersLocation'] = None
    
    return expt_config



    
def load_analysis_parameters(basePath, params_file):
    """ 
    Loads the analysis parameters file
    Inputs:
        basePath, params_file: path and file name of the yaml parameter file
    Outputs:
        params : dictionary of analysis parameters
    """
    params_file_full = os.path.join(basePath, params_file)
    # Note that we already checked if this exists
    with open(params_file_full, 'r') as f:
        all_param = yaml.safe_load(f)
    params = all_param['params']

    return params
        

def check_analysis_parameters(params):
    """
    Checks that all the keys in the analysis parameters file exist.
    If a key doesn't exist, prompts the user for its value with defaults from the file.
    Performs various checks
    Inputs:
        params : dictionary of analysis parameters
    Outputs:
        params : dictionary of analysis parameters
    """
    
    # Define the required keys and their default values
    required_keys = {
        "edge_rejection_threshold_mm": None,  
        "edge_proximity_threshold_mm": 5,
        "cos_theta_AP_threshold": -0.7,
        "cos_theta_tangent_threshold": 0.34,
        "cos_theta_90_thresh": 0.26,
        "cosSeeingAngle": 0.5,
        "perp_windowsize": 2,
        "perp_maxDistance_mm": 12.0,
        "contact_distance_threshold_mm": 2.5,
        "contact_inferred_distance_threshold_mm": 3.5,
        "contact_inferred_window": 3,
        "tail_rub_ws": 2,
        "tailrub_maxTailDist_mm": 2.0,
        "tailrub_maxHeadDist_mm": 12.5,
        "cos_theta_antipar": -0.8,
        "bend_min_deg": 10,
        "bend_Jmax_deg": 50, 
        "bend_Cmin_deg": 100, 
        "angle_xcorr_windowsize": 25,
        "approach_speed_threshold_mm_second": 20,
        "approach_cos_angle_thresh": 0.5,
        "approach_min_frame_duration": [2,2],
        "motion_speed_threshold_mm_second": 9,
        "proximity_threshold_mm": 7,
        "output_subFolder": 'Analysis',
        "allDatasets_ExcelFile": 'behavior_counts.xlsx', 
        "allDatasets_markFrames_ExcelFile":  'behaviors_in_each_frame.xlsx'
        # "another_key": default_value,
    }
    
    # Check for missing keys and prompt the user for their values
    for key, default_value in required_keys.items():
        if key not in params:
            user_input = input(f"Enter value for {key} (default: {default_value}): ")
            params[key] = user_input if user_input else default_value
    
    # Set edge rejection criterion to None 'None', and if negative
    if isinstance(params["edge_rejection_threshold_mm"], str):
        if params["edge_rejection_threshold_mm"].lower() == 'none':
            params["edge_rejection_threshold_mm"] = None
        else:
            raise ValueError("edge_rejection_threshold_mm : only allowed string is None") 
    if isinstance(params["edge_rejection_threshold_mm"], float):
        if params["edge_rejection_threshold_mm"] < 0.0:
            params["edge_rejection_threshold_mm"] = None
    
    return params



def load_global_expt_config(config_path, config_file):
    """ 
    Loads the global experimental configuration file, which points to the
    experiment-specific configuration files
    
    PLACEHOLDER
    To be written.
    Modify all_expt_configs to just contain basePath for the various experiments
    Note that this uses get_valid_file(); it's the only function that does;
       will need to load this in behaviors_main.py

    Inputs:
        config_path, config_file: path and file name of the yaml config file
    Outputs:
        expt_config : dictionary of configuration information
    """
    config_file_full = os.path.join(config_path, config_file)
    # Check if the config file exists; dialog box if not
    if not os.path.isfile(config_file_full):
        print(f"The config file '{config_file_full}' does not exist.")
        config_file_full = get_valid_file(fileTypeString = 'Config File')
    
    with open(config_file_full, 'r') as f:
        all_config = yaml.safe_load(f)
    all_expt_names = list(all_config.keys())
    print('\n\nAll experiments: ')
    for j, key in enumerate(all_expt_names):
        print(f'  {j}: {key}')

    #----
    # MODIFY HERE: As below, select an experiment, but just use this path info, 
    # and load the expt config file with load_expt_config
        
    expt_choice = input('Select experiment (name string or number): ')
    # Note that we're not checking if the choice is valid, i.e. if in 
    # all_expt_names (if a string) or if in 0...len(all_expt_names) (if 
    # a string that can be converted to an integer.)
    try:
        # Is the input string just an integer? Try integer...
        expt_config = all_config[all_expt_names[int(expt_choice)]]
    except:
        # Must be a string
        expt_config = all_config[all_expt_names[expt_choice]]
    
    return expt_config           

def get_CSV_filenames(basePath, expt_config, startString="results"):
    """
    Select subgroup (if applicable) and get a list of all CSV files 
    whose names start with startString, probably "results," 
    in the basePath previously specified

    Inputs:
        basePath : folder containing folders with CSV files for analysis;
                    dataPathMain will be appended to this.
                    Required, even if dataPathFull overwrites it
        expt_config : dictionary containing subGroup info (if it exists)
        startString : the string that all CSV files to be considered should
                        start with. Default "results"
    Returns:
        A tuple containing
        - dataPath : the folder path containing CSV files
        - allCSVfileNames : a list of all CSV files with names 
                            starting with startString (probably "results")
        - subGroupName : Path name of the subGroup; None if no subgroups
    
    """

    if ('subGroups' in expt_config.keys()) and (expt_config['subGroups'] != None):
        print('\nSub-Experiments:')
        for j, subGroup in enumerate(expt_config['subGroups']):
            print(f'  {j}: {subGroup}')
        subGroup_choice = input('Select sub-experiment (string or number): ')
        try:
            subGroupName = expt_config['subGroups'][int(subGroup_choice)]
        except:
            subGroupName = expt_config['subGroups'][subGroup_choice]
        dataPath = os.path.join(basePath, subGroupName)
    else:
        subGroupName = None
        dataPath = basePath
        
    # Validate the folder path
    while not os.path.isdir(dataPath):
        print("Invalid data path. Please try again (manual entry).")
        dataPath = input("Enter the folder path: ")

    print("Selected folder path: ", dataPath)
    
    # Make a list of all relevant CSV files in the folder
    allCSVfileNames = []
    for filename in os.listdir(dataPath):
        if (filename.endswith('.csv') and filename.startswith(startString)):
            allCSVfileNames.append(filename)

    return dataPath, allCSVfileNames, subGroupName



def get_dataset_name(CSVfileName):
    """ 
    Extract the "dataset name" from the CSV filename. Delete
    "results_SocPref_", "_ALL.csv"; keep everything else
    E.g. file name "results_SocPref_3c_2wpf_k2_ALL.csv" gives
      dataset_name = "3c_2wpf_k2""
    
    Returns:
        dataset_name : string
    """
    dataset_name = CSVfileName.replace("results_SocPref_", '')
    dataset_name = dataset_name.replace("_ALL", '')
    dataset_name = dataset_name.replace(".csv", '')
    return dataset_name



def set_outputFile_params(params, expt_config, subGroupName):
    """
    Fill in keys in params corresponding to output folders, Excel file names

    Parameters
    ----------
    params : dictionary of analysis parameters.
    expt_config : dictionary of experiment parameters.
    subGroupName : subgroup name (String) or None if no subgroup

    Returns
    -------
    params : dictionary of analysis parameters.

    """
    # Append experiment and subGroup names to Analysis output folder names
    if subGroupName is None:
        params['output_subFolder'] =  expt_config['expt_name'] + '_' + \
            params['output_subFolder']
    else:
        params['output_subFolder'] =  expt_config['expt_name'] + '_' + \
            subGroupName + '_' + params['output_subFolder']
    
    # Add subgroup name (if it exists) and the experiment name (i.e.
    # folder name) to the output Excel file names, appending these to
    # (1) "behavior_counts.xlsx" (or whatever params["allDatasets_ExcelFile"]
    #     is) for summary statistics of each behavior for each dataset
    # (2) "behaviors_in_each_frame.xlsx" (or whatever params["allDatasets_markFrames_ExcelFile"])
    #     is for marking behaviors in each frame
    base_name1, extension1 = os.path.splitext(params["allDatasets_ExcelFile"])
    if subGroupName is None:
        params["allDatasets_ExcelFile"] = f"{expt_config['expt_name']}_{base_name1}{extension1}"
    else:
        params["allDatasets_ExcelFile"] = f"{expt_config['expt_name']}_{subGroupName}_{base_name1}{extension1}" 
    print(f"Modifying output allDatasets_ExcelFile file name to be: {params['allDatasets_ExcelFile']}")
    base_name, extension = os.path.splitext(params["allDatasets_markFrames_ExcelFile"])
    if subGroupName is None:
        params["allDatasets_markFrames_ExcelFile"] = f"{expt_config['expt_name']}_{base_name}{extension}"
    else:
        params["allDatasets_markFrames_ExcelFile"] = f"{expt_config['expt_name']}_{subGroupName}_{base_name}{extension}"
    print(f"Modifying output allDatasets_markFrames_ExcelFile file name to be: {params['allDatasets_markFrames_ExcelFile']}")
    
    
    # If there are subgroups, modify the output Excel file name for
    # summary statistics of each behavior for each dataset -- instead
    # of "behavior_counts.xlsx" (or whatever params["allDatasets_ExcelFile"]
    # currently is), append subGroupName
    if not subGroupName==None:
        # base_name, extension = os.path.splitext(params["allDatasets_ExcelFile"])
        params["allDatasets_ExcelFile"] = f"{base_name1}_{subGroupName}{extension1}"
        print(f"Modifying output allDatasets_ExcelFile file name to be: {params['allDatasets_ExcelFile']}")

    return params


def load_all_position_data(allCSVfileNames, expt_config, CSVcolumns,
                           dataPath, params):
    """
    For all CSV files in the list, call load_data() to load all position
    data, and determine general parameters such as fps and scale
    Inputs:
        allCSVfileNames : a list of all CSV file names to consider
        expt_config : experiment configuration dictionary
        CSVcolumns : CSV column dictionary (what's in what column)
        dataPath : Path containing data files
        params : dictionary of analysis parameters 
                (used only for plotAllPositions())
        
    Returns:
        all_position_data : list of numpy arrays with position information for
                    each dataset. Nframes x Ncolumns x Nfish
        datasets : list of dictionaries, one for each dataset. 
                    datasets[j] contains information for dataset j.
    
    """
    # Number of datasets
    N_datasets = len(allCSVfileNames)

    # initialize a list of numpy arrays and a list of dictionaries
    all_position_data = [{} for j in range(N_datasets)]
    datasets = [{} for j in range(N_datasets)]
    os.chdir(dataPath)

    # For each dataset, get general properties and load all position data
    for j, CSVfileName in enumerate(allCSVfileNames):
        datasets[j]["CSVfilename"] = CSVfileName
        datasets[j]["dataset_name"] = get_dataset_name(CSVfileName)
        datasets[j]["image_scale"] = float(get_imageScale(datasets[j]["dataset_name"], 
                                                    expt_config))
        datasets[j]["fps"] = expt_config["fps"]
        
        # Get arena center, subtracting image position offset
        datasets[j]["arena_center"] = get_ArenaCenter(datasets[j]["dataset_name"], 
                                                      expt_config)
        # Estimate center location of Arena
        # datasets[j]["arena_center"] = estimate_arena_center(all_position_data[j],
        #                                                    CSVcolumns["head_column_x"],
        #                                                    CSVcolumns["head_column_y"])

        # Load all the position information as a numpy array
        print('Loading dataset: ', datasets[j]["dataset_name"])
        all_position_data[j], datasets[j]["frameArray"] = \
            load_data(CSVfileName, CSVcolumns["N_columns"]) 
        datasets[j]["Nframes"] = len(datasets[j]["frameArray"])
        datasets[j]["Nfish"] = all_position_data[j].shape[2]
        print('   ', 'Number of frames: ', datasets[j]["Nframes"] )
        datasets[j]["total_time_seconds"] = (np.max(datasets[j]["frameArray"]) - \
            np.min(datasets[j]["frameArray"]) + 1.0) / datasets[j]["fps"]
        print('   ', 'Total duration: ', datasets[j]["total_time_seconds"], 'seconds')
    

    return all_position_data, datasets
    
def load_data(CSVfileName, N_columns):
    """
    Loads position data from a CSV file and returns a single array
    containing position information for all fish
    (position, angle, body markers etc.)
    Works for any number of fish -- infers this from the first column
    Also returns frame numbers (first column of CSV), checking that 
    the frame number array is the same for each fish id of the 
    dataset.
    Checks that frame numbers are consecutive integers from 1 to Nframes 
    for each ID; raises an Error otherwise.

    Args:
        CSVfileName (str): CSV file name with tracking data
        N_columns: number of columns (probably 26).

    Returns:
        all_data : a single numpy array with all the data 
                   (all columns of CSV)
                   Rows = frames
                   Col = CSV columns
                   Layers = fish (Nfish)
        frameArray : array of all frame numbers
    """
    data = np.genfromtxt(CSVfileName, delimiter=',')
    id_numbers = data[:, 0]
    frame_numbers = data[:, 1]
    unique_ids = np.unique(id_numbers)
    Nfish = len(unique_ids)
    # print('Number of fish: ', Nfish)
    
    # (1) Check that the set of frame numbers is the same for all ID numbers
    frame_sets = [set(frame_numbers[id_numbers == id]) for id in unique_ids]
    if not all(frame_set == frame_sets[0] for frame_set in frame_sets):
        raise ValueError("Frame numbers are not consistent across all ID numbers")

    # (2) Get the number of unique frame numbers, and an array of all frame numbers
    frameArray = np.sort(np.array(list(frame_sets[0]), dtype=int))
    Nframes = len(frame_sets[0])

    # (3) Check that frame numbers are consecutive integers from 1 to Nframes for each ID
    for id in unique_ids:
        id_frames = np.sort(frame_numbers[id_numbers == id])
        if not np.array_equal(id_frames, np.arange(1, Nframes + 1)):
            raise ValueError(f"Frame numbers for ID {id} are not consecutive from 1 to {Nframes}")

    # (4) Create the all_data array
    all_data = np.zeros((Nframes, data.shape[1], Nfish))
    for j, id in enumerate(unique_ids):
        id_data = data[id_numbers == id]
        # sort by frame number, though these should be in order.
        sorted_indices = np.argsort(id_data[:, 1])
        all_data[:, :, j] = id_data[sorted_indices]

    return all_data, frameArray



def get_ArenaCenter(dataset_name, expt_config):
    """ 
    Extract the x,y positions of the Arena centers from the 
    arenaCentersFilename CSV -- previously tabulated.
    Image offsets also previously tabulated, "arenaCentersColumns" 
         columns of offsetPositionsFilename
    Uses first column in the arenaCenters file to match to dataset name

    Inputs:
        dataset_name :
        expt_config : configuration dictionary. Contains:
            arenaCentersFilename: csv file name with arena centers 
                (and one header row). If None, estimate centers from well
                offsets
            arenaCentersColumns : columns containing image offsets
            offsetPositionsFilename : csv file name with well offset positions
            datasetRemoveStrings: strings to remove from file and dataset names
    
    Returns:
        arenaCenterCorrected: tuple of x, y positions of arena Center
    Returns none if no rows match the input dataset_name, 
        and error if >1 match
        
    """
    datasetColumns = [0] # First column only for matching dataset names.
    
    # Find the row of this dataset in the well offset data file
    matching_offset_rows = []
    with open(expt_config['offsetPositionsFilename'], 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if match_dataset_name(dataset_name, row, 
                                  removeStrings = expt_config['datasetRemoveStrings'],
                                  datasetColumns = datasetColumns)==True:
                matching_offset_rows.append(row)
                
    if len(matching_offset_rows) == 0:
        # No matching rows in the offset file were found.
        arenaOffset = None
        print(f"\nFilename: {expt_config['offsetPositionsFilename']}")
        raise ValueError(f"get_ArenaCenter: No rows contain the input dataset_name: {dataset_name}")
    elif len(matching_offset_rows) > 1:
        raise ValueError("get_ArenaCenter: Multiple rows contain the input dataset_name string")
    else:
        arenaOffset = np.array((matching_offset_rows[0][1], 
                                matching_offset_rows[0][2])).astype(float)
        # There's offset data, now load or estimate arena center
        if expt_config['arenaCentersLocation'] != None:
            # Find the uncorrected arena positions
            matching_rows = []
            with open(expt_config['arenaCentersLocation'], 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header row
                for row in reader:
                    if match_dataset_name(dataset_name, row, 
                                          removeStrings = expt_config['datasetRemoveStrings'],
                                          datasetColumns = datasetColumns)==True:
                        matching_rows.append(row)
        
                if len(matching_rows) == 0:
                    arenaCenterUncorrected = None
                elif len(matching_rows) > 1:
                    arenaCenterUncorrected = None
                    raise ValueError("get_ArenaCenter: Multiple rows contain the input dataset_name string")
                else:
                    arenaCenterUncorrected = np.array((matching_rows[0][expt_config['arenaCentersColumns'][0]], 
                                                       matching_rows[0][expt_config['arenaCentersColumns'][1]])).astype(float)
                    arenaCenterCorrected = arenaCenterUncorrected - arenaOffset
        else:
             # Estimate arena positions based on well offset positions
             arenaCenterCorrected = np.array((matching_offset_rows[0][3], 
                                    matching_offset_rows[0][4]), 
                                             dtype=float)/2.0

    if arenaOffset is not None:
        return arenaCenterCorrected
    else:
        return None
        


def get_imageScale(dataset_name, expt_config):
    """ 
    Extract the image scale (um/px) from the config file (same for all datasets
    in this experiment) or from the imageScaleFilename CSV 

    Inputs:
        dataset_name : name of dataset
        expt_config : dictionary that contains as keys
            imageScale : image scale, um/px
            OR
            imageScaleLocation : Path and filename of CSV file containing 
                             image scale information
            imageScaleColumn : column (0-index) with image scale
            datasetColumns: columns to concatenate to match dataset name .
            datasetRemoveStrings: strings to remove from file and dataset names

    Returns:
        image scale (um/px)
    Returns none if no rows match the input dataset_name, 
        and error if >1 match
        
    Code partially from GPT3-5 (openAI)
    """
    
    if ("imageScale" in expt_config.keys()) and \
        expt_config['imageScale'] != None:
            return expt_config['imageScale']
    else:
        # Read from CSV file
        matching_rows = []
        with open(expt_config['imageScaleLocation'], 'r', 
                  encoding='Latin1') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                if match_dataset_name(dataset_name, row, 
                                      removeStrings = expt_config['datasetRemoveStrings'],
                                      datasetColumns = expt_config['datasetColumns']) \
                        == True:
                    matching_rows.append(row)
    
        if len(matching_rows) == 0:
            print('Dataset name: ', dataset_name)
            raise ValueError(f"get_imageScale: No row found with the input string {dataset_name}")
        elif len(matching_rows) > 1:
            # print(dataset_name, ' in rows: ', matching_rows[:][0])
            raise ValueError("get_imageScale: Multiple rows contain the input dataset_name string")
        else:
            return matching_rows[0][expt_config['imageScaleColumn']]
        



def match_dataset_name(dataset_name, row_array, 
                       removeStrings = [], datasetColumns = [0]):
    """ 
    Test whether the string formed from the datasetColumns columns of 
         row_array[] (appending '_') between columns) matches the string 
         dataset_name, modifying both strings by deleting
         the strings in the list removeStrings).
    
    Return True if dataset_name ad row_array match (given modifications);
    False otherwise
    
    To be used by get_imageScale() and get_ArenaCenter()
    
    """
    
    # Modifications to dataset_name
    mod_dataset_name = dataset_name
    for remove_string in removeStrings:
        mod_dataset_name = mod_dataset_name.replace(remove_string, '')
        
    # Append another column to row[0], if desired, with underscore
    mod_row_array = row_array[datasetColumns[0]]
    for j in range(1, len(datasetColumns)):
        mod_row_array = mod_row_array + '_' + row_array[datasetColumns[j]]

    #print('mod_row_array, initial: ', mod_row_array)
    # Modifications to row_array
    for remove_string in removeStrings:
        mod_row_array = mod_row_array.replace(remove_string, '')
        #print('  - mod_row_array: ', mod_row_array)
    
    #print('mod_dataset_name: ', mod_dataset_name)
    #print('mod_row_array: ', mod_row_array)
    if mod_dataset_name == mod_row_array:
        return True
    else:
        # print('dataset name: ', mod_dataset_name, ' row array: ', mod_row_array)
        return False
    
        
def get_output_pickleFileNames(expt_name, subGroupName = None):
    """
    Get / construct pickle file name. 
    First construct the "base" filename, then filenames for
    both Pickle file outputs, appending "positionData" and "datasets"
    
    Parameters
    ----------
    expt_name : experiment name, probably expt_config['expt_name']
    subGroupName : subGroupName, None if no subgroups

    Returns
    -------
    pickleFileNames : tuple of three strings, name of pickle files for
                     "positionData" and "datasets" pickle files, and "base" name

    """                              
    print('\nEnter the base filename for the output pickle files;')
    print('   Do not include ".pickle"; appended later.')
    print('   Enter "none" to skip pickle output.')
    # Append experiment and subGroup names to Analysis output folder
    if subGroupName is None:
        defaultPickleFileNameBase =  expt_name 
    else:
        defaultPickleFileNameBase =  expt_name + '_' + subGroupName
    pickleFileNameBase = input(f'   Press Enter for default {defaultPickleFileNameBase}: ')
    if pickleFileNameBase == '':
        pickleFileNameBase = defaultPickleFileNameBase
    
    # Make sure doesn't end with .pickle
    if pickleFileNameBase.endswith('.pickle'):
            pickleFileNameBase = pickleFileNameBase[:-7]
    
    pickleFileNames = (pickleFileNameBase + "_positionData.pickle", 
                       pickleFileNameBase + "_datasets.pickle",
                       pickleFileNameBase)
    
    return pickleFileNames
    

def write_pickle_file(dict_for_pickle, dataPath, outputFolderName, pickleFileName):
    """
    Write Pickle file containing a dictionary of variables in the analysis folder
    
    Parameters
    ----------
    dict_for_pickle : dictionary of variables to save in the Pickle file
    dataPath : CSV data path
    outputFolderName : output path, should be params['output_subFolder'],
                       appended to dataPath
    pickleFileName : string, filename, including .pickle
    Returns
    -------
    None.
    """
    # Use pathlib for better path handling
    pickle_folder = Path(dataPath) / outputFolderName
    pickle_file_path = pickle_folder / pickleFileName
    
    # Check path length and shorten if necessary
    if len(str(pickle_file_path)) > 250:  # Conservative limit for Windows
        print(f"Warning: Path too long ({len(str(pickle_file_path))} chars). Shortening...")
        
        # Try shortening the filename first
        name_part, ext_part = pickleFileName.rsplit('.', 1) if '.' in pickleFileName else (pickleFileName, '')
        max_filename_len = 250 - len(str(pickle_folder)) - 1  # -1 for path separator
        
        if max_filename_len > 20:  # Need reasonable minimum
            shortened_name = name_part[:max_filename_len-len(ext_part)-1] + '.' + ext_part if ext_part else name_part[:max_filename_len]
            pickle_file_path = pickle_folder / shortened_name
            print(f"Shortened filename to: {shortened_name}")
        else:
            # If even shortening filename doesn't work, use a generic name
            shortened_name = f"datasets_{hash(pickleFileName) % 10000}.pickle"
            pickle_file_path = pickle_folder / shortened_name
            print(f"Using generic filename: {shortened_name}")
    
    try:
        # Create output directory with exist_ok=True to handle race conditions
        pickle_folder.mkdir(parents=True, exist_ok=True)
        
        # Double-check the directory was created
        if not pickle_folder.exists():
            raise FileNotFoundError(f"Failed to create directory: {pickle_folder}")
        
        # Save the pickle file
        print(f'\nWriting pickle file: {pickle_file_path.name} in {pickle_folder}\n')
        with open(pickle_file_path, 'wb') as handle:
            pickle.dump(dict_for_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickle file saved successfully.")
        
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        print(f"Failed path: {pickle_file_path}")
        print(f"Path length: {len(str(pickle_file_path))} characters")
        print("Possible causes:")
        print("- Path too long for Windows (>260 chars)")
        print("- Parent directory doesn't exist or is inaccessible")
        print("- Invalid characters in path")
        
        # Try one more fallback with a very short name
        fallback_name = f"data_{hash(str(pickle_file_path)) % 1000}.pkl"
        fallback_path = pickle_folder / fallback_name
        print(f"Attempting fallback with short filename: {fallback_name}")
        
        try:
            with open(fallback_path, 'wb') as handle:
                pickle.dump(dict_for_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Fallback successful! File saved as: {fallback_path}")
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            raise e  # Raise original error
            
    except PermissionError as e:
        print(f"PermissionError: {e}")
        print(f"Check write permissions for: {pickle_folder}")
        raise
    except OSError as e:
        print(f"OSError: {e}")
        print(f"Path length: {len(str(pickle_file_path))} characters")
        if len(str(pickle_file_path)) > 250:
            print("This is likely due to Windows path length limitations.")
            print("Consider using shorter folder/file names or enabling long path support.")
        raise


def load_and_assign_from_pickle():
    """
    Calls load_dict_from_pickle() and assign_variables_from_dict()
    to load *two* pickle files and assign variables.
    Asks users for filenames.
    
    Inputs:
        None
        
    Outputs
        all_position_data : all position data, from first pickle file
        variable_tuple : tuple of variables, from the second pickle file
    
    """
    print('\n\nLoading from Pickle.')
    print('\n   Note that this requires *two* pickle files:')
    print('     (1) position data, probably in the CSV folder')
    print('     (2) "datasets" and other information, probably in Analysis folder')
    print('For each, enter the full path or just the filename; leave empty for a dialog box.')
    print('\n')
    pickleFileName1 = input('(1) Pickle file name for position data; blank for dialog box: ')
    if pickleFileName1 == '': pickleFileName1 = None
    pos_dict = load_dict_from_pickle(pickleFileName=pickleFileName1)
    all_position_data = assign_variables_from_dict(pos_dict, inputSet = 'positions')
    pickleFileName2 = input('(2) Pickle file name for datasets etc.; blank for dialog box: ')
    if pickleFileName2 == '': pickleFileName2 = None
    data_dict = load_dict_from_pickle(pickleFileName=pickleFileName2)
    variable_tuple = assign_variables_from_dict(data_dict, inputSet = 'datasets')
    
    return all_position_data, variable_tuple
    
def load_dict_from_pickle(pickleFileName=None):
    """
    Load contents from pickle file
    Assumes pickle file contains a dictionary of variables;
    returns these variables as a dictionary.
    
    Parameters
    ----------
    pickleFileName : pickle file name; can include path to append to basePath

    Returns
    -------
    dict_of_variables : dictionary containing all variables in the Pickle file
    """

    badFile = True  # for verifying
    while badFile:
        if pickleFileName is None or pickleFileName == '':
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            titleString = 'Select pickle file'
            pickleFileName = filedialog.askopenfilename(title=titleString,
                                                        filetypes=[("pickle files", "*.pickle")])
        else:
            if not os.path.isabs(pickleFileName):
                # Get parent folder information
                # Get path from dialog
                root = tk.Tk()
                root.withdraw()  # Hide the root window
                basePath = filedialog.askdirectory(title=f"Select folder containing specified pickle file, {pickleFileName}")
                pickleFileName = os.path.join(basePath, pickleFileName)
        if os.path.isfile(pickleFileName):
            badFile = False
        else:
            print(f"\n\nInvalid pickle file path or name: {pickleFileName}")
            print("Please try again; will force dialog box.")
            pickleFileName = None

    print(f'\nOpening pickle file: {pickleFileName}')
    print('\n')
    with open(pickleFileName, 'rb') as handle:
        dict_of_variables = pickle.load(handle)

    return dict_of_variables

def assign_variables_from_dict(dict_of_variables, inputSet):
    """
    Assign dictionary elements loaded from pickle file, from load_dict_from_pickle(),
    to variables. Hard-coded variables for each "inputSet" (positions or datasets etc.)
    Input
        dict_of_variables : dictionary containing all variables in the Pickle file
        inputSet : 'positions' or 'datasets' for the different pickle files
    Outputs
        variable_tuple : tuple of variables
    """
    
    if inputSet == 'positions':
        variable_tuple = dict_of_variables['all_position_data']
    elif inputSet == 'datasets': 
        datasets = dict_of_variables['datasets']
        CSVcolumns = dict_of_variables['CSVcolumns']
        expt_config = dict_of_variables['expt_config']
        params = dict_of_variables['params']
        basePath = dict_of_variables['basePath']
        dataPath = dict_of_variables['dataPath']
        subGroupName = dict_of_variables['subGroupName']
        
        # Number of datasets
        N_datasets = len(datasets)
        Nfish = get_Nfish(datasets)
   
        variable_tuple = (datasets, CSVcolumns, expt_config, params, N_datasets,
                          Nfish, basePath, dataPath, subGroupName)
    else :
        raise ValueError('Invalid inputSet for assigning variables')

    return variable_tuple


def combine_expts_from_pickle():
    """
    Load data from multiple experiments, each with two pickle files,
    combine the variables, and save the output to two new pickle files.
    
    Asks user for number of experiments to combine and loads data from
    each experiment's pickle files. Combines position data and datasets,
    verifies consistency of parameters, and creates new combined output.
    
    Also can write new Excel, CSV, YAML files based on the composite 
    dataset – optional, asks user.
    
    Returns
    -------
    None
    """
    # Get number of experiments to combine
    while True:
        try:
            N_expts = int(input('\nHow many experiments would you like to combine? '))
            if N_expts > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid integer.")
    
    # Initialize lists to store data from all experiments
    all_position_data_combined = []
    datasets_combined = []
    all_Nfish = []
    all_expt_configs = []
    all_params = []
    all_subGroupNames = []
    
    # Load data from each experiment
    for i in range(N_expts):
        print(f'\nLoading data for experiment {i+1} of {N_expts}')
        pos_data, var_tuple = load_and_assign_from_pickle()
        
        # Unpack variables from tuple
        (datasets, CSVcolumns, expt_config, params, N_datasets, 
         Nfish, basePath, dataPath, subGroupName) = var_tuple
        
        # Verify datasets length matches N_datasets
        if len(datasets) != N_datasets:
            raise ValueError(f'Experiment {i+1}: Length of datasets ({len(datasets)}) ' 
                           f'does not match N_datasets ({N_datasets})')
        
        # Store data for later combining/comparison
        all_position_data_combined.extend(pos_data)
        datasets_combined.extend(datasets)
        all_Nfish.append(Nfish)
        all_expt_configs.append(expt_config)
        all_params.append(params)
        all_subGroupNames.append(subGroupName)
    
    # Verify Nfish is the same for all experiments
    if len(set(all_Nfish)) != 1:
        raise ValueError(f'Nfish varies across experiments: {all_Nfish}')
    Nfish = all_Nfish[0]
    
    # Combine expt_config dictionaries
    combined_expt_config = {}
    for key in all_expt_configs[0].keys():
        if key in ['imageScaleLocation', 'arenaCentersLocation']:
            combined_expt_config[key] = None
        else:
            values = [config[key] for config in all_expt_configs]
            if len(set(str(v) for v in values)) > 1:  # Convert to string for comparison
                print(f'\nDifferent values found for expt_config["{key}"]:')
                for i, v in enumerate(values):
                    print(f'  Experiment {i+1}: {v}')
                value = input('Enter value to use for combined dataset: ')
                # Try to convert input to original type
                orig_type = type(values[0])
                try:
                    combined_expt_config[key] = orig_type(value)
                except ValueError:
                    combined_expt_config[key] = value
            else:
                combined_expt_config[key] = values[0]
    
    # Combine params dictionaries
    combined_params = {}
    for key in all_params[0].keys():
        if key == 'output_subFolder':
            continue  # Skip this key as it will be set later
        values = [p[key] for p in all_params]
        if len(set(str(v) for v in values)) > 1:  # Convert to string for comparison
            print(f'\nDifferent values found for params["{key}"]:')
            for i, v in enumerate(values):
                print(f'  Experiment {i+1}: {v}')
            value = input('Enter value to use for combined dataset: ')
            # Try to convert input to original type
            orig_type = type(values[0])
            try:
                combined_params[key] = orig_type(value)
            except ValueError:
                combined_params[key] = value
        else:
            combined_params[key] = values[0]
    
    # Handle subGroupName
    if len(set(all_subGroupNames)) > 1:
        combined_subGroupName = all_subGroupNames
    else:
        combined_subGroupName = all_subGroupNames[0]
    
    # Get base filename for output
    basePickleFileName = input('\nEnter base name for output pickle files; will add .pickle etc.: ')
    outputPickleFileNames = [
        f'{basePickleFileName}_positionData.pickle',
        f'{basePickleFileName}_datasets.pickle'
    ]
    
    # Get output path with option for text input or dialog
    print('\nEnter output path or leave blank for dialog box (create folder if necessary):')
    outputPath = input('Path: ').strip()
    if not outputPath:
        root = tk.Tk()
        root.withdraw()
        outputPath = filedialog.askdirectory(title="Select output directory; create if necessary")
        if not outputPath:  # If user cancels dialog
            raise ValueError("No output path selected")
    
    # Get subfolder name with default "Analysis"
    subfolderName = input('\nEnter name for analysis subfolder; default "Analysis": ').strip()
    if not subfolderName:
        subfolderName = "Analysis"
    
    # Create full output paths
    outputSubPath = os.path.join(outputPath, subfolderName)
    
    # Create subfolder if it doesn't exist
    if not os.path.exists(outputSubPath):
        os.makedirs(outputSubPath)
    
    # Create and write output pickle files
    variables_dict = {'all_position_data': all_position_data_combined}
    write_pickle_file(variables_dict, dataPath=outputPath,
                     outputFolderName='',
                     pickleFileName=outputPickleFileNames[0])
    
    # Update combined_params with output_subFolder
    combined_params['output_subFolder'] = outputSubPath
    
    variables_dict = {
        'datasets': datasets_combined,
        'CSVcolumns': CSVcolumns,
        'expt_config': combined_expt_config,
        'params': combined_params,
        'basePath': outputPath,  # Use outputPath as the new basePath
        'dataPath': None,
        'subGroupName': combined_subGroupName
    }
    write_pickle_file(variables_dict, dataPath=outputSubPath,
                     outputFolderName='',
                     pickleFileName=outputPickleFileNames[1])
    
    print('\nPickle files successfully combined and written to:')
    print(f'  {os.path.join(outputPath, outputPickleFileNames[0])}')
    print(f'  {os.path.join(outputSubPath, outputPickleFileNames[1])}')
    
    outputExcel = input('\nOutput Excel, CSV, and YAML files? y=yes, n=no: ')
    if outputExcel.lower()=='y':
        write_CSV_Excel_YAML(expt_config = combined_expt_config,
                             params = combined_params, 
                             dataPath = outputPath, 
                             datasets = datasets_combined)


def write_output_files(params, output_path, datasets):
    """
    Write the output files (several) for all datasets
    Inputs:
        params : analysis parameters; we use the output file pathinfo
        output_path : output path, probably os.path.join(dataPath, params['output_subFolder']
        datasets : list of dictionaries: all dataset and analysis output
        
    Outputs:
        None (multiple file outputs)
    """
    
    t_writing_all_start = perf_counter()
    
    print('\n\nWriting output files...')
    N_datasets = len(datasets)
    
    # Convert to Path object for better handling
    output_path = Path(output_path)
    
    # Create output directory, if it doesn't exist
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory: {e}")
        print(f"Path: {output_path}")
        print(f"Path length: {len(str(output_path))} characters")
        raise

    # Store original directory to restore later (instead of os.chdir)
    original_cwd = Path.cwd()

    try:
        # behaviors (events) to write. (Superset)
        Nfish = datasets[0]["Nfish"] # number of fish, take from the first set;
                                 # don't bother checking if same for all
        key_list = ["close_pair", "perp_noneSee", 
                    "perp_oneSees", "perp_bothSee", 
                    "perp_larger_fish_sees", "perp_smaller_fish_sees", 
                    "contact_any", "contact_head_body", 
                    "contact_larger_fish_head", "contact_smaller_fish_head", 
                    "contact_inferred", "tail_rubbing", "maintain_proximity", 
                    "anyPairBehavior"]
        for j in range(Nfish):
            key_list.extend([f"Cbend_Fish{j}"])
        key_list.extend(["Cbend_any"])  # formerly had a condition "if Nfish > 1:"
        for j in range(Nfish):
            key_list.extend([f"Rbend_Fish{j}"])
        key_list.extend(["Rbend_any"])  
        for j in range(Nfish):
            key_list.extend([f"Jbend_Fish{j}"])
        key_list.extend(["Jbend_any"]) # formerly had a condition "if Nfish > 1:"
        for j in range(Nfish):
            key_list.extend([f"approaching_Fish{j}"])
        if Nfish > 1:
            key_list.extend(["approaching_any"])
            key_list.extend(["approaching_all"])
        for j in range(Nfish):
            key_list.extend([f"fleeing_Fish{j}"])
        if Nfish > 1:
            key_list.extend(["fleeing_any"])
            key_list.extend(["fleeing_all"])
        for j in range(Nfish):
            key_list.extend([f"isMoving_Fish{j}"])
        key_list.extend(["isMoving_any", "isMoving_all"]) # formerly had a condition "if Nfish > 1:"
        for j in range(Nfish):
            key_list.extend([f"isBending_Fish{j}"])
        key_list.extend(["isBending_any", "isBending_all"]) # formerly had a condition "if Nfish > 1:"
        for j in range(Nfish):
            key_list.extend([f"isActive_Fish{j}"])
        key_list.extend(["isActive_any", "isActive_all"]) # formerly had a condition "if Nfish > 1:"
        for j in range(Nfish):
            key_list.extend([f"close_to_edge_Fish{j}"])
        key_list.extend(["close_to_edge_any", "close_to_edge_all"]) # formerly had a condition "if Nfish > 1:"
        key_list.extend(["edge_frames", "bad_bodyTrack_frames"])
        # Remove any keys that are not in the first dataset, for example
        # two-fish behaviors if that dataset was for single fish data
        key_list_revised = [key for key in key_list if key in datasets[0]]
        
        # Mark frames for each dataset
        # Create the ExcelWriter object with path validation
        excel_filename = sanitize_filename(params['allDatasets_markFrames_ExcelFile'])
        excel_file_path = output_path / excel_filename
        
        # Check if full path length is reasonable
        if len(str(excel_file_path)) > 250:  # Conservative limit
            # Try shortening the filename
            name, ext = excel_filename.rsplit('.', 1)
            shortened_name = name[:100] + '_shortened.' + ext
            excel_file_path = output_path / shortened_name
            print(f"Warning: Excel filename was too long, shortened to: {shortened_name}")
        
        try:
            writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')
        except (FileNotFoundError, OSError, PermissionError) as e:
            print(f"Error creating Excel file: {e}")
            print(f"Attempted path: {excel_file_path}")
            print(f"Path length: {len(str(excel_file_path))} characters")
            
            # Fallback: try with a very short filename
            fallback_name = "behav_frame.xlsx"
            excel_file_path = output_path / fallback_name
            print(f"Trying fallback filename: {fallback_name}")
            writer = pd.ExcelWriter(excel_file_path, engine='xlsxwriter')

        # Call the function to write frames for each dataset
        print('   Marking behaviors in each frame for each dataset.')
        t_mark_frames_start = perf_counter()
        for j in range(N_datasets):
            # Excel worksheet name handling - more robust sanitization
            dataset_name = str(datasets[j]["dataset_name"])
            sheet_name = sanitize_sheet_name(dataset_name)
            
            mark_behavior_frames_Excel(writer, datasets[j], key_list_revised, 
                                       sheet_name)
        t_mark_frames_end = perf_counter()
        print(f'       ... {N_datasets} x mark_behavior_frames_Excel: ' + 
              f'time {t_mark_frames_end - t_mark_frames_start:.1f} s')
        
        # Save and close the Excel file
        try:
            writer.close()
        except Exception as e:
            print(f"Warning: Error closing Excel writer: {e}")
            # Try to save manually if possible
            try:
                writer.save()
            except:
                pass

        
        print('   Writing summary text file and basic measurements for each dataset.')
        # For each dataset, summary text file and basic measurements    
        t_writing_summary_start = perf_counter()
        
        # Change to output directory for the text file functions
        # (assuming they expect to write in current directory)
        os.chdir(output_path)
        
        try:
            for j in range(N_datasets):
                # Write for this dataset: summary in text file
                write_behavior_txt_file(datasets[j], key_list_revised)
                # Write for this dataset: frame-by-frame "basic measurements"
                write_basicMeasurements_txt_file(datasets[j])
        finally:
            # Always restore original directory
            os.chdir(original_cwd)
            
        t_writing_summary_end = perf_counter()
        print( f'   ... Elapsed time {t_writing_summary_end - t_writing_summary_start:.1f} s')
            
            
        # Excel workbook for summary of all behavior counts, durations,
        # relative durations, and relative durations normalized to activity
        print('   Writing summary file of all behavior counts, durations to ' + 
              params['allDatasets_ExcelFile'])
        initial_keys = ["dataset_name", "fps", "image_scale",
                        "total_time_seconds", "close_pair_fraction", 
                        "speed_mm_s_mean", "speed_whenMoving_mm_s_mean",
                        "bout_rate_bpm", "bout_duration_s", "bout_ibi_s",
                        "fish_length_Delta_mm_mean", 
                        "head_head_distance_mm_mean", "closest_distance_mm_mean",
                        "AngleXCorr_mean"]
        initial_strings = ["Dataset", "Frames per sec", 
                           "Image scale (um/px)",
                           "Total Time (s)", "Fraction of time in proximity", 
                           "Mean speed (mm/s)", "Mean moving speed (mm/s)", 
                           "Mean bout rate (/min)", "Mean bout duration (s)",
                           "Mean inter-bout interval (s)",
                           "Mean difference in fish lengths (mm)", 
                           "Mean head-head dist (mm)", "Mean closest distance (mm)",
                           "AngleXCorr_mean"]

        # Remove any keys that are not in the first dataset, for example
        # two-fish behaviors if that dataset was for single fish data
        # Also remove the corresponding strings
        initial_keys_revised = []
        initial_strings_revised = []
        
        for key, name in zip(initial_keys, initial_strings):
            if key in datasets[0]:
                initial_keys_revised.append(key)
                initial_strings_revised.append(name)

        # Sanitize the Excel filename for the summary file
        summary_excel_filename = sanitize_filename(params["allDatasets_ExcelFile"])
        summary_excel_path = output_path / summary_excel_filename
        
        # Check path length for summary file too
        if len(str(summary_excel_path)) > 250:
            name, ext = summary_excel_filename.rsplit('.', 1)
            shortened_name = name[:100] + '_summary.' + ext
            summary_excel_path = output_path / shortened_name
            print(f"Warning: Summary Excel filename was too long, shortened to: {shortened_name}")

        write_behaviorCounts_Excel(str(summary_excel_path), 
                                  datasets, key_list_revised, 
                                  initial_keys_revised, initial_strings_revised)
        
        # Return the full path for use by add_statistics_to_excel
        return str(summary_excel_path)
        
    except Exception as e:
        print(f"Error in write_output_files: {e}")
        raise
    finally:
        # Ensure we're back in the original directory
        try:
            os.chdir(original_cwd)
        except:
            pass
    
    t_writing_all_end = perf_counter()
    print('   Done writing output files. ' + 
          f'Elapsed time {t_writing_all_end - t_writing_all_start:.1f} s')


def sanitize_filename(filename):
    """
    Sanitize filename by removing or replacing invalid characters
    """
    if not filename:
        return "output_file"
    
    # Remove or replace invalid characters for Windows/Unix
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', str(filename))
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure filename isn't empty after sanitization
    if not sanitized:
        sanitized = "output_file"
    
    return sanitized


def sanitize_sheet_name(dataset_name):
    """
    Sanitize and truncate Excel sheet name to meet Excel's requirements
    """
    if not dataset_name:
        return "Sheet1"
    
    # Excel sheet names cannot contain: [ ] : * ? / \
    invalid_chars = r'[\[\]:*?/\\]'
    sanitized = re.sub(invalid_chars, '_', str(dataset_name))
    
    # Remove leading/trailing spaces
    sanitized = sanitized.strip()
    
    # Truncate to 31 characters (Excel limit)
    if len(sanitized) > 31:
        sanitized = sanitized[-31:]  # Take last 31 chars as in original
    
    # Ensure sheet name isn't empty
    if not sanitized:
        sanitized = "Sheet1"
    
    return sanitized

        
def write_behavior_txt_file(dataset, key_list):
    """
    Creates a txt file of the relevant window frames and event durations
    for a set of social behaviors in a given *single dataset*
    Output text file name: dataset_name + .txt, one per dataset,
    in Analysis output folder

    Inputs:
        dataset  : dictionary with all dataset info
        key_list : list of dictionary keys corresponding to each 
                   behavior to write frame information for

    Returns:
        N/A
    """
    with open(f"{dataset['dataset_name']}.txt", "w") as results_file:
        results_file.write(f"{dataset['dataset_name']}\n")
        results_file.write("Experimental parameters\n")
        results_file.write(f"   Image scale (um/px): {dataset['image_scale']:.1f}\n")
        results_file.write(f"   frames per second: {dataset['fps']:.1f}\n")
        results_file.write(f"   Number of frames: {dataset['Nframes']}\n")
        results_file.write(f"   Duration: {dataset['total_time_seconds']:.1f} s\n")
        results_file.write("Tracking Properties\n")
        results_file.write(f"   Number of spans of continuous good tracking: {dataset['good_tracking_spans']['number']}\n")
        results_file.write(f"   Mean length of spans of continuous good tracking: {dataset['good_tracking_spans']['mean_frames']:.1f} frames\n")
        results_file.write(f"   Minimum length of spans of continuous good tracking: {dataset['good_tracking_spans']['min_frames']} frames\n")
        results_file.write(f"   Maximum length of spans of continuous good tracking: {dataset['good_tracking_spans']['max_frames']} frames\n")
        results_file.write("Basic Properties\n")
        results_file.write(f"   Mean fish length: {dataset['fish_length_mm_mean']:.3f} mm\n")
        results_file.write(f"   Mean fish speed: {dataset['speed_mm_s_mean']:.3f} mm/s\n")
        results_file.write(f"   Mean fish speed when moving > threshold: {dataset['speed_whenMoving_mm_s_mean']:.3f} mm/s\n")
        results_file.write(f"   Bout rate: {dataset['bout_rate_bpm']:.1f} bouts/minute\n")
        results_file.write(f"   Mean bout duration: {dataset['bout_duration_s']:.2f} seconds\n")
        results_file.write(f"   Mean inter-bout interval: {dataset['bout_ibi_s']:.2f} seconds\n")
        if dataset["Nfish"]==2:
            results_file.write(f"   Mean difference in fish length: {dataset['fish_length_Delta_mm_mean']:.3f} mm\n")
            results_file.write(f"   Mean head-to-head distance: {dataset['head_head_distance_mm_mean']:.3f} mm\n")
            results_file.write(f"   Mean closest distance: {dataset['closest_distance_mm_mean']:.3f} mm\n")
            results_file.write(f"   Fraction of time in proximity: {dataset['close_pair_fraction']:.3f}\n")
        for k in key_list:
            outString = f'{k} N_events: {dataset[k]["combine_frames"].shape[1]}\n' + \
                    f'{k} Total N_frames: {dataset[k]["total_duration"]}\n' + \
                    f'{k} frames: {dataset[k]["combine_frames"][0,:]}\n' + \
                    f'{k} durations: {dataset[k]["combine_frames"][1,:]}\n'
            results_file.write(outString)


def write_basicMeasurements_txt_file(dataset):
    """
    Creates a txt file of "basic" speed and distance measurements for
    a given *single dataset* at each frame.
    Assesses what to write given number of fish. (For example,
            don't attempt inter-fish distance if Nfish==1)
    Rows = Frames
    Columns = 
        Head-to-head distance (mm) ["head_head_distance_mm"]
        Closest inter-fish distance (mm) ["closest_distance_mm"]
        Speed of each fish (mm/s); frame-to-frame speed, recorded as 0 for the first frame. ["speed_array_mm_s"]
        Relative orientation, i.e. angle between heading and 
            head-to-head vector, for each fish (radians) ["relative_orientation"]
        Relative orientation, i.e. angle between heading and 
            head-to-head vector, for each fish (radians) ["relative_orientation"]
        Relative_heading_angle: The difference in heading angle between the 
            two fish (in range [0, pi]), radians. ["relative_heading_angle"]
        Edge flag (1 or 0) ["edge_frames"]
        Bad track (1 or 0) ["bad_bodyTrack_frames"]

    Caution: 
        Number of fish hard-coded as 2
        Key names hard coded!
        
    Output text file name: dataset_name + _basicMeasurements.txt, 
    one per dataset, in Analysis output folder

    Inputs:
        dataset : dictionary with all dataset info
        key_list_basic : list of dictionary keys corresponding 
                         to each measurement to write (some 
                         are for multiple fish)

    Returns:
        N/A
    """
    Nframes = dataset["Nframes"] # number of frames
    Nfish = dataset["Nfish"] # number of fish
    frames = np.arange(1, Nframes+1)
    
    # Prepare edge and tracking flags more efficiently
    EdgeFlag = np.zeros(Nframes, dtype=int)
    if len(dataset["edge_frames"]['raw_frames']) > 0:
        EdgeFlagIdx = dataset["edge_frames"]['raw_frames'] - 1
        EdgeFlag[EdgeFlagIdx] = 1
        
    BadTrackFlag = np.zeros(Nframes, dtype=int)
    BadTrackIdx = dataset["bad_bodyTrack_frames"]['raw_frames'] - 1
    BadTrackFlag[BadTrackIdx] = 1

    # Create headers
    headers = ["frame"]
    if Nfish == 2:
        headers.extend(["head_head_distance_mm", "closest_distance_mm", "relative_heading_angle"])
        headers.extend([f"rel_orientation_rad_Fish{j}" for j in range(Nfish)])
    
    headers.extend([f"speed_mm_s_Fish{j}" for j in range(Nfish)])
    headers.extend([f"r_fromCtr_mm_Fish{j}" for j in range(Nfish)])
    headers.extend(["edge flag", "bad tracking"])

    # Create CSV data more efficiently with string building
    # Prepare buffer for rows
    rows = []
    for j in range(Nframes):
        row_parts = [f"{frames[j]}"]
        
        if Nfish == 2:
            row_parts.extend([
                f"{dataset['head_head_distance_mm'][j].item():.3f}",
                f"{dataset['closest_distance_mm'][j].item():.3f}",
                f"{dataset['relative_heading_angle'][j].item():.3f}"
            ])
            row_parts.extend([f"{dataset['relative_orientation'][j, k].item():.3f}" for k in range(Nfish)])
        
        row_parts.extend([f"{dataset['speed_array_mm_s'][j, k].item():.3f}" for k in range(Nfish)])
        row_parts.extend([f"{dataset['radial_position_mm'][j, k].item():.3f}" for k in range(Nfish)])
        row_parts.extend([f"{EdgeFlag[j]}", f"{BadTrackFlag[j]}"])
        
        rows.append(",".join(row_parts))
    
    # Write the CSV file in one operation
    with open(f"{dataset['dataset_name']}_basicMeasurements.csv", "w", newline="") as f:
        f.write(",".join(headers) + "\n")
        f.write("\n".join(rows))
    
    
def write_CSV_Excel_YAML(expt_config, params, dataPath, datasets):
    """
    Write the output files (CSV, Excel, and YAML (parameters))
    Calls write_output_files(), add_statistics_to_excel()
    
    expt_config : experimental configuration dictionary
    params : analysis parameters
    dataPath : primary output folder, e.g. with CSV files
    datasets : list of dictionaries of datasets
    
    """
    print(f'\nWriting to dataPath: {dataPath}')
    # Write the output files (CSV, Excel) - THIS WAS MISSING!
    output_path = os.path.join(dataPath, params['output_subFolder'])
    excel_file_path = write_output_files(params, output_path, datasets)
    
    # Modify the Excel file with statistics
    print(f'\nAdding statistics to Excel file: {excel_file_path}')
    add_statistics_to_excel(excel_file_path)
    
    # Write a YAML file with parameters  
    more_param_output = {'dataPath': dataPath}
    all_outputs = expt_config | params | more_param_output
    print('\nWriting output YAML file.')
    yaml_path = os.path.join(output_path, 'all_params.yaml')
    with open(yaml_path, 'w') as file:
        yaml.dump(all_outputs, file)
        


def mark_behavior_frames_Excel(writer, dataset, key_list, sheet_name):
    """
    Create and fill in a sheet in an existing Excel file, marking all frames 
    with behaviors found in this dataset.
    
    Optimized Aug. 2025. Uses numpy arrays and vectorized operation.
    
    Inputs:
        writer (pandas.ExcelWriter): The ExcelWriter object representing the Excel file.
        dataset (dict): Dictionary with all dataset info.
        key_list (list): List of dictionary keys corresponding to each behavior to write.
        sheet_name (str): Name of the sheet to be created.
        
    Returns:
        N/A
        
    """
    
    maxFrame = int(np.max(dataset["frameArray"]))
    
    # Pre-allocate numpy array (much faster than DataFrame operations)
    # Use object dtype to store strings efficiently
    n_behaviors = len(key_list)
    data_array = np.full((maxFrame, n_behaviors), '', dtype='U17')
    
    # Process all behaviors using vectorized operations
    for col_idx, k in enumerate(key_list):
        behavior_data = dataset[k]["combine_frames"]
        
        # Skip if no events
        if behavior_data.shape[1] == 0:
            continue
            
        # Extract frame ranges
        start_frames = behavior_data[0, :].astype(int) - 1  # Convert to 0-indexed
        durations = behavior_data[1, :].astype(int)
        
        # Create boolean mask for all frames that should be marked
        mask = np.zeros(maxFrame, dtype=bool)
        
        # Vectorized approach to mark ranges
        for start, duration in zip(start_frames, durations):
            end = min(start + duration, maxFrame)  # Ensure we don't exceed bounds
            if start >= 0 and start < maxFrame:
                mask[start:end] = True
        
        # Apply marking where mask is True
        data_array[mask, col_idx] = 'X'.center(17)
    
    # Create DataFrame from pre-populated array (much faster)
    frames = np.arange(1, maxFrame + 1)
    df = pd.DataFrame(data_array, columns=key_list)
    df.insert(0, 'Frame', frames)
    
    # Write to Excel in single operation
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    

def write_behaviorCounts_Excel(ExcelFileName, datasets, key_list, 
                              initial_keys, initial_strings):
    """
    Creates an Excel with summary statistics of each behavior for each dataset, 
    indicating the number of events, duration (number of frames), 
    and relative duration of each of the behaviors. 
    
    The function also calculates durations relative to the number of 
    "active" frames, both for "isActive_any" and "isActive_all". 
    These are not saved outside this function.
    
    Each row is one dataset. Each column is one behavior; 
    the first few columns are general dataset properties. 
    In addition, separated by a blank row, each sheet contains 
    statistics over the datasets (mean, std. dev., and s.e.m.).
        
    Output text file name: dataset_name + _basicMeasurements.txt, 
    one per dataset, in Analysis output folder

    Inputs:
        ExcelFileName : File name to write
        datasets : list of dictionaries with behavior information,
        key_list : list of behavior dictionary keys to write; note that each 
                    contains Nframes, durations, and relative durations
        initial_keys : keys to write that are single values, not behavior
                    dictionaries 
        initial_strings : column header strings corresponding to initial_keys

    Returns:
        N/A
    """
    t_write_behavCounts_start = perf_counter()

    Ndatasets = len(datasets)

    # Calculate relative durations in one pass
    for j in range(Ndatasets):
        # Calculate these values once per dataset
        active_any_total = datasets[j]["isActive_any"]["total_duration"]
        active_all_total = datasets[j]["isActive_all"]["total_duration"]
        
        for key in key_list:
            total_duration = datasets[j][key]["total_duration"]
            datasets[j][key]['rel_duration_active_any'] = total_duration / active_any_total 
            datasets[j][key]['rel_duration_active_all'] = total_duration / active_all_total
    
    # Create a new Excel writer object
    writer = pd.ExcelWriter(ExcelFileName, engine='xlsxwriter')

    # Define the sheet names and corresponding data keys
    sheets = {
        "N_events": "N_events",
        "Durations (frames)": "total_duration",
        "Relative Durations": "relative_duration",
        "Durations rel Any Active": "rel_duration_active_any",
        "Durations rel All Active": "rel_duration_active_all"
    }

    # Process all sheets in one pass
    for sheet_name, data_key in sheets.items():
        # Create data for the dataframe
        data = [
            [datasets[j].get(key, {}) for key in initial_keys] + 
            [datasets[j][key][data_key] for key in key_list]
            for j in range(Ndatasets)
        ]
        
        # Create DataFrame and write to Excel
        df = pd.DataFrame(data, columns=initial_strings + key_list)
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save the Excel file
    writer.close()
    t_write_behavCounts_end = perf_counter()
    print('      write_behaviorCounts_Excel: ' + 
          f'time {t_write_behavCounts_end - t_write_behavCounts_start:.1f} s')


def add_statistics_to_excel(file_path='behaviorCounts.xlsx'):
    """
    Modify the Excel sheet containing behavior counts to include
    summary statistics for all datasets (e.g. average for 
    each behavior)
    
    Parameters:
    -----------
    file_path : str or Path
        Full path to the Excel file (can be relative or absolute)
    """
    # Convert to Path object for better handling
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        # Try to find it in current directory if only filename was provided
        if not file_path.is_absolute():
            cwd_file = Path.cwd() / file_path.name
            if cwd_file.exists():
                file_path = cwd_file
            else:
                print(f"Error: Excel file not found at: {file_path}")
                print(f"Current working directory: {Path.cwd()}")
                print(f"Also tried: {cwd_file}")
                
                # List files in current directory for debugging
                try:
                    files = list(Path.cwd().glob("*.xlsx"))
                    if files:
                        print(f"Excel files in current directory: {[f.name for f in files]}")
                    else:
                        print("No .xlsx files found in current directory")
                except:
                    pass
                
                raise FileNotFoundError(f"Could not find Excel file: {file_path}")
    
    print(f"Processing Excel file: {file_path}")
    
    try:
        # Load the Excel file
        xls = pd.ExcelFile(file_path)
        
        # Create a temporary file for writing (to avoid conflicts)
        temp_file = file_path.parent / f"temp_{file_path.name}"
        writer = pd.ExcelWriter(temp_file, engine='openpyxl')
        
        for sheet_name in xls.sheet_names:
            try:
                # Read the sheet
                df = pd.read_excel(xls, sheet_name=sheet_name)
                
                # Separate non-numeric and numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) == 0:
                    print(f"Warning: No numeric columns found in sheet '{sheet_name}', skipping statistics")
                    # Just write the original data
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    continue
                
                # Calculate statistics for numeric columns
                mean = df[numeric_cols].mean()
                std = df[numeric_cols].std()
                n = df[numeric_cols].count()
                sem = std / np.sqrt(n)
                
                # Write the original data first
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Get the worksheet to add statistics
                worksheet = writer.sheets[sheet_name]
                
                # Add a blank row, then statistics
                start_row = len(df) + 3  # +2 for header and +1 for blank row
                
                # Create statistics data
                stats_data = [
                    ['Mean'] + [mean.get(col, '') for col in numeric_cols],
                    ['Std. Dev.'] + [std.get(col, '') for col in numeric_cols],
                    ['Std. Error of Mean'] + [sem.get(col, '') for col in numeric_cols],
                    ['N datasets'] + [n.get(col, '') for col in numeric_cols]
                ]
                
                # Write statistics with proper column alignment
                non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
                
                for i, row_data in enumerate(stats_data):
                    # Start with the statistic name
                    worksheet.cell(row=start_row + i, column=1, value=row_data[0])
                    
                    # Fill in empty cells for non-numeric columns
                    for j, col in enumerate(non_numeric_cols[1:], start=2):  # Skip first col (statistic name)
                        worksheet.cell(row=start_row + i, column=j, value='')
                    
                    # Add the numeric values in the correct columns
                    numeric_start_col = len(non_numeric_cols) + 1
                    for j, value in enumerate(row_data[1:]):
                        if value != '':  # Only write non-empty values
                            worksheet.cell(row=start_row + i, column=numeric_start_col + j, value=value)
                
                # Adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if cell.value is not None:
                                length = len(str(cell.value))
                                if length > max_length:
                                    max_length = length
                        except:
                            pass
                    
                    # Set width with some padding
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 for very long headers
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                    
            except Exception as e:
                print(f"Error processing sheet '{sheet_name}': {e}")
                # Still write the original data even if statistics fail
                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                except:
                    print(f"Could not even write original data for sheet '{sheet_name}'")
        
        # Save and close the writer
        writer.close()
        xls.close()
        
        # Replace the original file with the updated one
        if file_path.exists():
            file_path.unlink()  # Delete original
        temp_file.rename(file_path)  # Rename temp to original
        
        print(f"Successfully added statistics to: {file_path}")
        
    except Exception as e:
        print(f"Error in add_statistics_to_excel: {e}")
        # Clean up temp file if it exists
        temp_file = file_path.parent / f"temp_{file_path.name}"
        if temp_file.exists():
            temp_file.unlink()
        raise

