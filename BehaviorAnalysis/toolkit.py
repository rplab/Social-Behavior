# !/usr/bin/env python3  
# toolkit.py
# -*- coding: utf-8 -*- 
"""
Author:   Raghuveer Parthasarathy
Version ='2.0': 
First version created by  : Estelle Trieu, 9/7/2022
Major modifications by Raghuveer Parthasarathy, May-July 2023
Last modified by Rghuveer Parthasarathy, Oct. 17, 2024

Description
-----------

Module containing functions for handling data files and performing
various non-behavioral analyses.
Includes: 
    - Load expt config file
    - get lists of files to load
    - load data, 
    - assess proximity to the edge, 
    - assess bad frames, 
    - link_weighted(): re-do fish IDs (track linkage)
    - repair_double_length_fish() : split fish that are 2L in length into two fish
    - auto- and cross-correlation, for single and all datasets
    - etc.
"""

import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pickle
import pandas as pd
import yaml
import tkinter as tk
import tkinter.filedialog
from scipy import signal

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
                # Check path name
                last_part_of_path = os.path.basename(os.path.normpath(selected_path))
                if "csv files" not in last_part_of_path.lower():
                    verifyName = input('Chosen folder name does not include "CSV files." Continue? (y/n): ') 
                    if verifyName.lower() == 'y':
                        return selected_path
                else:
                    return selected_path
        else:
            print("Invalid path. Please try again.")


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
    print('\n\nALl experiments: ')
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
    


def get_CSV_filenames(basePath, expt_config, startString="results"):
    """
    Select subgroup (if applicable) and get a list of all CSV files 
    whose names start with 
    startString, probably "results," in the basePath previously specified

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


def load_all_position_data(allCSVfileNames, expt_config, CSVcolumns,
                           dataPath, params, showAllPositions=False):
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
        showAllPositions : if True, plotAllPositions() will be called
                            to show all head positions, 
                            dish edge in a separate figure for each dataset.
        
    Returns:
        datasets : dictionaries for each dataset. datasets[j] contains
                    all the information for dataset j.
    
    """
    # Number of datasets
    N_datasets = len(allCSVfileNames)

    # initialize a list of dictionaries for datasets
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
        # datasets[j]["arena_center"] = estimate_arena_center(datasets[j]["all_data"],
        #                                                    CSVcolumns["head_column_x"],
        #                                                    CSVcolumns["head_column_y"])

        # Load all the position information as a numpy array
        print('Loading dataset: ', datasets[j]["dataset_name"])
        datasets[j]["all_data"], datasets[j]["frameArray"] = \
            load_data(CSVfileName, CSVcolumns["N_columns"]) 
        datasets[j]["Nframes"] = len(datasets[j]["frameArray"])
        datasets[j]["Nfish"] = datasets[j]["all_data"].shape[2]
        print('   ', 'Number of frames: ', datasets[j]["Nframes"] )
        datasets[j]["total_time_seconds"] = (np.max(datasets[j]["frameArray"]) - \
            np.min(datasets[j]["frameArray"]) + 1.0) / datasets[j]["fps"]
        print('   ', 'Total duration: ', datasets[j]["total_time_seconds"], 'seconds')
    
        # (Optional) Show all head positions, and arena center, and dish edge. 
        #    (& close threshold)
        if showAllPositions:
            plotAllPositions(datasets[j], CSVcolumns, expt_config['arena_radius_mm'], 
                             params["arena_edge_threshold_mm"])

    return datasets
    
def load_data(CSVfileName, N_columns):
    """
    Loads position data from a CSV file and returns a single array
    containing both information for all fish
    (position, angle, body markers etc.)
    Load all columns (0 to N_columns-1)
    Works for any number of fish -- infers this from the first column
    Also returns frame numbers (first column of CSV), checking that 
    the frame number array is the same for each fish section of the 
    dataset.

    Args:
        CSVfileName (str): CSV file name with tracking data
        N_columns: number of columns (almost certainly 26).

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
    # Check that the number of rows (i.e. frames) is the same for each fish
    unique_ids = np.unique(id_numbers)
    Nfish = len(unique_ids)
    print('Number of fish: ', Nfish)
    
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
        sorted_indices = np.argsort(id_data[:, 1])
        all_data[:, :, j] = id_data[sorted_indices]

    return all_data, frameArray


def fix_heading_angles(datasets, CSVcolumns):
    """ 
    Fix the heading angles -- rather than the strangely quantized angles from
    ZebraZoom, calculate the angle from arctan(y[1]-y[2], x[1]-x[2]) 
    See notes Sept. 2024
    
    Inputs
        datasets : list of all dataset dictionaries. 
                    Note "all_data" is Nframes x data columns x Nfish
        CSVcolumns : CSV column dictionary (what's in what column)
                                            
    Outputs
        datasets: contains new "heading_angle" key in each dataset
                    datasets[j]["heading_angle"]
                                            
    """
    x2 = CSVcolumns["body_column_x_start"]+1 # x position #2
    x3 = CSVcolumns["body_column_x_start"]+2 # x position #3
    y2 = CSVcolumns["body_column_y_start"]+1 # y position #2
    y3 = CSVcolumns["body_column_y_start"]+2 # y position #3
    for j in range(len(datasets)):
        datasets[j]['heading_angle'] = np.arctan2(datasets[j]["all_data"][:,y2,:]-datasets[j]["all_data"][:,y3,:], 
                                                  datasets[j]["all_data"][:,x2,:]-datasets[j]["all_data"][:,x3,:])
        # put in range [0, 2*pi]; not actually necessary
        datasets[j]['heading_angle'][datasets[j]['heading_angle'] < 0.0] += 2*np.pi
    return datasets



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

def make_frames_dictionary(frames, frames_to_remove, behavior_name,
                           Nframes):
    """
    Make a dictionary of raw (original) frames, frames with "bad" 
    frames removed, combined (adjacent) frames + durations,
    total durations, and relative durations
    
    Calls remove_frames()
    Calls combine_events()
    Inputs:
        frames (int) : 1D array of frame numbers
        frames_to_remove : tuple of 1D arrays of frame numbers to remove
        behavior_name : name to assign to the behavior (string)
        Nframes : number of frames (probably datasets[j]['Nframes'])
        
    Outputs:
        frames_dict : dictionary with keys
            behavior_name : name of the behavior (string)
            raw_frames : original (frames), 1D array
            edit_frames : frames with "bad" frames removed, 1D array
            combine_frames : 2 x N array using combine_events, frame numbers
                and durations
            N_events : number of events (simply the length
                                         of the second row of combine_frames)
            total_duration : scalar, sum of durations (frames)
            relative_duration : scalar, relative duration, i.e.
                total_duration / Nframes
    """
    # necessary to initialize the dictionary this way?
    keys = {"raw_frames", "edit_frames", "combine_frames", 
            "total_duration", "behavior_name", "relative_duration"}
    frames_dict = dict([(key, []) for key in keys])
    frames_dict["behavior_name"] = behavior_name
    frames_dict["raw_frames"] = frames
    frames_dict["edit_frames"] = \
        remove_frames(frames,  frames_to_remove)
    frames_dict["combine_frames"] = \
        combine_events(frames_dict["edit_frames"])
    frames_dict["total_duration"] = np.sum(frames_dict["combine_frames"][1,:])
    frames_dict["N_events"] = frames_dict["combine_frames"].shape[1]
    frames_dict["relative_duration"] = frames_dict["total_duration"] / Nframes
    return frames_dict


def remove_frames(frames, frames_to_remove, dilate_frames=np.array([0])):
    """
    Remove from frames values that appear in frames_to_remove, 
    and optionally dilate the set of frames to remove.
    
    Inputs:
        frames (numpy.ndarray): 1D array of frame numbers
        frames_to_remove (tuple): tuple of 1D arrays of frame numbers 
                                    to remove
        dilate_frames (numpy.ndarray): 1D array of integers, 
            where each non-zero value 'j' will cause frames
            'all_frames_to_remove + j' to be added to 
            the set of frames to remove. (Leave as zero for no dilation)
        
    Outputs:
        frames_edit (numpy.ndarray): 1D array of (remaining) frame numbers
    """
    # Combine all frame numbers in frames_to_remove into a single set

    # Exit if nothing to remove
    if len(frames_to_remove) == 0:
        return frames
    
    # Concatenate only if there is a tuple of arrays
    if type(frames_to_remove) is tuple:
        all_frames_to_remove = np.unique(np.concatenate(frames_to_remove))
    else:
        all_frames_to_remove = np.unique(frames_to_remove)
    
    # Dilate the set of frames to remove if dilate_frames is provided
    all_frames_to_remove_temp = all_frames_to_remove.copy()
    for j in dilate_frames:
        if j != 0:
            all_frames_to_remove = np.unique(np.concatenate(
                (all_frames_to_remove, all_frames_to_remove_temp + j)))
    
    # Remove the frames to be removed from the original frames
    frames_edit = np.setdiff1d(frames, all_frames_to_remove)
    
    return frames_edit

def dilate_frames(frames, dilate_frames=np.array([0])):
    """
    "dilate" the array of frame numbers.
    
    Inputs:
        frames (numpy.ndarray): 1D array of frame numbers, for example
            bad tracking frames
        dilate_frames (numpy.ndarray): 1D array of integers, 
            where each non-zero value 'j' will cause frames
            'frames + j' to be added to the array of frames
            the set of frames to remove. (Leave as zero for no dilation)
        
    Outputs:
        frames_edit (numpy.ndarray): 1D array of unique frame numbers
    """
    
    frames_edit = frames.copy()
    
    # Dilate the set of frames to remove if dilate_frames is provided
    for j in dilate_frames:
        if j != 0:
            frames_edit = np.unique(np.concatenate((frames_edit, frames + j)))
   
    return frames_edit


def combine_events(events):
    """
    Given an array of frame numbers, return an arrays of frame numbers 
    with adjacent frames combined and duration numbers 
    corresponding to the duration of adjacent frames.
    For example, frames…
    	1, 5, 12, 13, 14, 17, 22, 23, 34, 40
    Would be returned as frame numbers
        5, 12, 17, 22, 34, 40
    And durations
        1, 3, 1, 2, 1, 1
    See Raghu's notes May 28, 2023, and example_get_runs_and_durations.py 
        for a description of the method

    Args:
        events (array): an array of frame numbers corresponding to events
                           (e.g. frames of detected circling, 
                            tail-rubbing, 90-degree events).

    Returns:
        combined_events_and_durations (array) : a 2 x N array; Row 1
            are uninque, non-adjacent frame numbers, and Row 2 are the 
            corresponding durations.
            and duration numbers corresponding to the duration of adjacent frames.
    """
    d_events = np.diff(events)
    idx1 = np.array(np.where(d_events==1)).flatten() + 1  # index + 1 of flat regions
    idx_keep = np.setdiff1d(np.arange(len(events)), idx1) # indexes of x to keep
    unique_events = events[idx_keep]
    durations = np.ones_like(unique_events)
    for j in range(len(idx1)-1,-1, -1):
        # find the closest idx_keep under idx1[j]
        close_idx = max(idx_keep[idx1[j] > idx_keep])
        duration_idx = np.array(np.where(idx_keep == close_idx)).flatten()
        durations[duration_idx] += 1
    
    combined_events_and_durations = np.stack((events[idx_keep], durations))
    return combined_events_and_durations

def get_edge_frames_dictionary(datasets, params, arena_radius_mm, CSVcolumns):
    """ 
    identify frames in which the head position of one or more fish is close
    to the dish edge (within threshold)
    Note that radial coordinate has already been calculated (get_polar_coords() )
    
    Inputs:
        datasets : all datasets, dictionary 
        params : analysis parameters
        arena_radius_mm :arena_radius in mm
        CSVcolumns : CSV column name dictionary
        
    Output:
        datasets : all datasets, dictionary; now with ["edge_frames"]
                    key for each datasets[j]; see documentation
    """
    # Number of datasets
    N_datasets = len(datasets)
    print('N_datasets: ', N_datasets)
    for j in range(N_datasets):
        print('Finding edge frames in Dataset: ', datasets[j]["dataset_name"])
        # Identify frames in which one or both fish are too close to the edge
        # First keep as an array, then convert into a dictionary that includes
        #    durations of edge events, etc.
        # Don't bother keeping array of distance to edge, 
        #    since radial_position_mm and arena_radius contains this infomation
        edge_frames = get_edge_frames(datasets[j], params, arena_radius_mm, 
                                      CSVcolumns["head_column_x"],
                                      CSVcolumns["head_column_y"])
        
        datasets[j]["edge_frames"] = make_frames_dictionary(edge_frames, (), 
                                                            behavior_name='Edge frames',
                                                            Nframes=datasets[j]['Nframes'])
        print('   Number of edge frames: ', len(datasets[j]["edge_frames"]["raw_frames"]))
    
    return datasets

    
def get_edge_frames(dataset, params, arena_radius_mm, xcol=3, ycol=4):
    """ 
    identify frames in which the head position of one or more fish is close
    to the dish edge (within threshold)
    Note that radial coordinate has already been calculated (get_polar_coords() )
    
    Inputs:
        dataset : dataset dictionary. Note "all_data" is 
                  Nframes x data columns x Nfish
        params : parameters, including edge-closeness threshold
        arena_radius_mm :arena_radius in mm
        xcol, ycol = column indices (0==first) of the x and y head 
                        position columns
        
    Output:
        near_edge_frames : array of frame numbers (not index numbers!)
    """

    r_mm = dataset["radial_position_mm"] # distance from center, mm
    # True if close to edge
    near_edge = (arena_radius_mm - r_mm) < params["arena_edge_threshold_mm"]
    near_edge = np.any(near_edge, axis=1)

    near_edge_frames = dataset["frameArray"][np.where(near_edge)]
    
    return near_edge_frames

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
    
    
def estimate_arena_center(alldata, xcol=3, ycol=4):
    """ 
    Estimate the arena center position as the midpoint of the x-y range.
    
    
    Input: 
        alldata = numpy array of positions, 
        xcol, ycol = column indices (0==first) of the x and y head 
                        position columns
    Output:
        xc, yc : tuple of center position, px
    """
    xmax = np.max(alldata[:,xcol,:])
    xmin = np.min(alldata[:,xcol,:])
    ymax = np.max(alldata[:,ycol,:])
    ymin = np.min(alldata[:,ycol,:])
    xc = 0.5*(xmax+xmin)
    yc = 0.5*(ymax+ymin)
    return (xc, yc)



def get_badTracking_frames_dictionary(datasets, params, CSVcolumns, tol=0.001):
    """ 
    identify frames in which head or body tracking of one or more fish 
    is bad (zero values)
    
    Inputs:
        datasets : all datasets, dictionary 
        params : analysis parameters
        CSVcolumns : CSV column name dictionary
        tol : tolerance for determining "zeros" in position information
        
    Output:
        datasets : all datasets, dictionary; now with ["bad_headTrack_frames"]
                    and ["bad_bodyTrack_frames"]
                    keys for each datasets[j]; see documentation
    """
    # Number of datasets
    N_datasets = len(datasets)
    print('N_datasets: ', N_datasets)
    for j in range(N_datasets):
        print('Finding bad tracking frames in Dataset: ', datasets[j]["dataset_name"])
        # Identify frames in which tracking is bad; separately consider head, body
        # Note that body is the most general of these -- use this for criteria
        # First keep as an array, then convert into a dictionary that includes
        #    durations of bad tracking events, etc.
        bad_headTrack_frames = get_bad_headTrack_frames(datasets[j], params, 
                                     CSVcolumns["head_column_x"],
                                     CSVcolumns["head_column_y"], tol)
        datasets[j]["bad_headTrack_frames"] = make_frames_dictionary(bad_headTrack_frames, 
                                                                     (), 
                                                                     behavior_name='Bad head track frames',
                                                                     Nframes=datasets[j]['Nframes'])
        print('   Number of bad head tracking frames: ', len(datasets[j]["bad_headTrack_frames"]["raw_frames"]))
        bad_bodyTrack_frames = get_bad_bodyTrack_frames(datasets[j], params, 
                                     CSVcolumns["body_column_x_start"],
                                     CSVcolumns["body_column_y_start"],
                                     CSVcolumns["body_Ncolumns"], 0.001)
        datasets[j]["bad_bodyTrack_frames"] = make_frames_dictionary(bad_bodyTrack_frames, (), 
                                                                     behavior_name='Bad track frames',
                                                                     Nframes=datasets[j]['Nframes'])
        print('   Number of bad body tracking frames: ', len(datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
    
    return datasets

        
    
def get_bad_headTrack_frames(dataset, params, xcol=3, ycol=4, tol=0.001):
    """ 
    identify frames in which the head position of one or more fish is 
    zero, indicating bad tracking
    
    Inputs:
        dataset : dataset dictionary. Note "all_data" is 
                  Nframes x data columns x Nfish
        params : parameters
        xcol, ycol = column indices (0==first) of the x and y head 
                        position columns
        tol : tolerance for "zero" (bad tracking), pixels
        
    Output:
        bad_headTrack_frames : array of frame numbers (not index numbers!)
    """
    x = dataset["all_data"][:,xcol,:]
    y = dataset["all_data"][:,ycol,:]
    # True if any of x, y is zero; 
    xy_zero = np.logical_or(np.abs(x)<tol, np.abs(y)<tol)
    bad_headTrack = np.any(xy_zero, axis=1)

    bad_headTrack_frames = dataset["frameArray"][np.where(bad_headTrack)]
    return bad_headTrack_frames

    
def get_bad_bodyTrack_frames(dataset, params, body_column_x_start=6, 
                             body_column_y_start=16, body_Ncolumns=10, 
                             tol=0.001):
    """ 
    identify frames in which tracking failed, as indicated by either of:
    (i) any body position of one or more fish is zero, or
    (ii) the distance between positions 1 and 2 (head-body) is more than
         3 times the mean distance between positions j and j+1 
         for j = 2 to 9
    
    Inputs:
        dataset : dataset dictionary. Note "all_data" is 
                  Nframes x data columns x Nfish
        params : parameters
        body_column_{x,y}_start" : column indices (0==first) of the x and y 
                    body position column
        body_Ncolumns : 10 # number of body datapoints
        tol : tolerance for "zero", pixels
        
    Output:
        bad_bodyTrack_frames : array of frame numbers (not index numbers!)
    """
    x = dataset["all_data"][:,body_column_x_start:(body_column_x_start+body_Ncolumns),:]
    y = dataset["all_data"][:,body_column_y_start:(body_column_y_start+body_Ncolumns),:]
    # True if any of x, y is zero; 
    xy_zero = np.logical_or(np.abs(x)<tol, np.abs(y)<tol)
    # Look for any across body positions, and across fish
    bad_bodyTrack = np.any(xy_zero, axis=(1,2))

    # Look at distance between head and body, compare to body-body distances
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    dr_12 = np.sqrt(dx[:,0,:]**2 + dy[:,0,:]**2)
    dr_body = np.sqrt(dx[:,1:,:]**2 + dy[:,1:,:]**2)
    mean_dr_body = np.mean(dr_body,axis=1)
    bad_12_distance = np.any(dr_12 > 3*mean_dr_body, axis=1)
    # Flag frames with either zeros or large 1-2 distances.
    badidx = np.where(np.logical_or(bad_bodyTrack, bad_12_distance))
    bad_bodyTrack_frames = np.array(dataset["frameArray"][badidx])
    # print(bad_bodyTrack_frames)
    
    return bad_bodyTrack_frames
    
def wrap_to_pi(x):
    # shift values of numpy array "x" to [-pi, pi]
    # x must be an array, not a single number
    x_wrap=np.remainder(x, 2*np.pi)
    mask = np.abs(x_wrap)>np.pi
    x_wrap[mask] -= 2*np.pi * np.sign(x_wrap[mask])
    return x_wrap


def plotAllPositions(dataset, CSVcolumns, arena_radius_mm, 
                     arena_edge_mm = 0):
    """
    Plot head x and y positions for each fish, in all frames
    also dish center and edge
    
    Inputs:
        dataset : dictionary with all dataset info
        CSVcolumns : CSV column information (dictionary)
        arena_radius_mm
        arena_edge_mm : threshold distance from arena_radius to illustrate
    
    Outputs: none
    
    """
    plt.figure()
    plt.scatter(dataset["all_data"][:,CSVcolumns["head_column_x"],0].flatten(), 
                dataset["all_data"][:,CSVcolumns["head_column_y"],0].flatten(), color='m', marker='x')
    plt.scatter(dataset["all_data"][:,CSVcolumns["head_column_x"],1].flatten(), 
                dataset["all_data"][:,CSVcolumns["head_column_y"],1].flatten(), color='darkturquoise', marker='x')
    plt.scatter(dataset["arena_center"][0], dataset["arena_center"][1], 
                color='red', s=100, marker='o')
    Npts = 360
    cos_phi = np.cos(2*np.pi*np.arange(Npts)/Npts).reshape((Npts, 1))
    sin_phi = np.sin(2*np.pi*np.arange(Npts)/Npts).reshape((Npts, 1))
    R_px = arena_radius_mm*1000/dataset["image_scale"]
    R_closeEdge_px = (arena_radius_mm-arena_edge_mm)*1000/dataset["image_scale"]
    arena_ring = dataset["arena_center"] + R_px*np.hstack((cos_phi, sin_phi))
    edge_ring = dataset["arena_center"] + R_closeEdge_px*np.hstack((cos_phi, sin_phi))
    #arena_ring_x = dataset["arena_center"][0] + arena_radius_mm*1000/dataset["image_scale"]*cos_phi
    #arena_ring_y = dataset["arena_center"][1] + arena_radius_mm*1000/dataset["image_scale"]*sin_phi
    plt.plot(arena_ring[:,0], arena_ring[:,1], c='orangered', linewidth=3.0)
    if arena_edge_mm > 1e-9:
        plt.plot(edge_ring[:,0], edge_ring[:,1], c='lightcoral', linewidth=3.0)
    plt.title(dataset["dataset_name"] )
    plt.axis('equal')


def write_pickle_file(list_for_pickle, dataPath, outputFolderName, 
                      pickleFileName):
    """
    Write Pickle file containing datasets, etc., in the analysis folder
    
    Parameters
    ----------
    list_for_pickle : list of datasets to save in the Pickle file
    dataPath : CSV data path
    outputFolderName : output path, should be params['output_subFolder']
    pickleFileName : string, filename, including .pickle

    Returns
    -------
    None.

    """
    pickle_folder = os.path.join(dataPath, outputFolderName)
    
    # Create output directory, if it doesn't exist
    pickle_folder = os.path.join(dataPath, outputFolderName)
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)

    print(f'\nWriting pickle file: {pickleFileName}\n')
    with open(os.path.join(pickle_folder, pickleFileName), 'wb') as handle:
        pickle.dump(list_for_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
def write_output_files(params, dataPath, datasets):
    """
    Write the output files (several) for all datasets
    Inputs:
        params : analysis parameters; we use the output file pathinfo
        dataPath : path containing CSV input files
        datasets : list of dictionaries: all dataset and analysis output
        
    Outputs:
        None (multiple file outputs)
    """
    
    print('\n\nWriting output files...')
    N_datasets = len(datasets)
    
    # Create output directory, if it doesn't exist
    output_path = os.path.join(dataPath, params['output_subFolder'])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Go to analysis folder
    os.chdir(output_path)

    # Write to individual text files, individual Excel sheets, 
    # and summary CSV file

    # behaviors (events) to write. (Superset)
    Nfish = datasets[0]["Nfish"] # number of fish, take from the first set;
                             # don't bother checking if same for all
    key_list = ["close_pair", "perp_noneSee", 
                "perp_oneSees", "perp_bothSee", 
                "perp_larger_fish_sees", "perp_smaller_fish_sees", 
                "contact_any", "contact_head_body", 
                "contact_larger_fish_head", "contact_smaller_fish_head", 
                "contact_inferred", "tail_rubbing"]
    for j in range(Nfish):
        key_list.extend([f"Cbend_Fish{j}"])
    key_list.extend(["Cbend_any"])  # formerly had a condition "if Nfish > 1:"
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
    key_list.extend(["edge_frames", "bad_bodyTrack_frames"])
    # Remove any keys that are not in the first dataset, for example
    # two-fish behaviors if that dataset was for single fish data
    key_list_revised = [key for key in key_list if key in datasets[0]]

    # Mark frames for each dataset
    # Create the ExcelWriter object
    excel_file = os.path.join(output_path, 
                              params['allDatasets_markFrames_ExcelFile'])
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    # Call the function to write frames for each dataset
    for j in range(N_datasets):
        # Annoyingly, Excel won't allow a worksheet name that's
        # more than 31 characters! Force it to use the last 31.
        sheet_name = datasets[j]["dataset_name"][-31:]
        mark_behavior_frames_Excel(writer, datasets[j], key_list_revised, 
                                   sheet_name)
    # Save and close the Excel file
    writer.close()

    # Excel workbook for summary of all behavior counts, durations
    print('File for collecting all behavior counts: ', 
          params['allDatasets_ExcelFile'])
    initial_keys = ["dataset_name", "fps", "image_scale",
                    "total_time_seconds", "close_pair_fraction", 
                    "speed_mm_s_mean", "speed_whenMoving_mm_s_mean",
                    "bout_rate_bpm", "bout_duration_s", "bout_ibi_s",
                    "fish_length_Delta_mm_mean", 
                    "head_head_distance_mm_mean", 
                    "AngleXCorr_mean"]
    initial_strings = ["Dataset", "Frames per sec", 
                       "Image scale (um/s)",
                       "Total Time (s)", "Fraction of time in proximity", 
                       "Mean speed (mm/s)", "Mean moving speed (mm/s)", 
                       "Mean bout rate (/min)", "Mean bout duration (s)",
                       "Mean inter-bout interval (s)",
                       "Mean difference in fish lengths (mm)", 
                       "Mean head-head dist (mm)", 
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

    write_behaviorCounts_Excel(params["allDatasets_ExcelFile"], 
                              datasets, key_list_revised, 
                              initial_keys_revised, initial_strings_revised)

    # For each dataset, summary  text file and basic measurements    
    for j in range(N_datasets):
        
        # Write for this dataset: summary in text file
        write_behavior_txt_file(datasets[j], key_list_revised)
        
        # Write for this dataset: frame-by-frame "basic measurements"
        write_basicMeasurements_txt_file(datasets[j])

        
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
        results_file.write(f"   Duration: {dataset['total_time_seconds']:.1f} s\n")
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
    Creates a txt file of "basic" speed and distance measurements
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
    EdgeFlag = np.zeros((Nframes,),dtype=int)
    EdgeFlagIdx = dataset["edge_frames"]['raw_frames'] - 1
    EdgeFlag[EdgeFlagIdx] = 1
    BadTrackFlag = np.zeros((Nframes,),dtype=int)
    BadTrackIdx = dataset["bad_bodyTrack_frames"]['raw_frames'] - 1
    BadTrackFlag[BadTrackIdx] = 1

    # Create headers list
    headers = ["frame"]
    if Nfish == 2:
        headers.extend(["head_head_distance_mm", "closest_distance_mm",
                       "relative_heading_angle"])
        for j in range(Nfish):
            headers.extend([f"rel_orientation_rad_Fish{j}"])
    for j in range(Nfish):
        headers.extend([f"speed_mm_s_Fish{j}"])
    for j in range(Nfish):
        headers.extend([f"r_fromCtr_mm_Fish{j}"])
    headers.extend(["edge flag", "bad tracking"])

    # Create a list of rows
    rows = []
    
    for j in range(Nframes):
        row = ["{:d}".format(frames[j])]
        if Nfish == 2:
            row.extend(["{:.3f}".format(dataset["head_head_distance_mm"][j].item()),
                        "{:.3f}".format(dataset["closest_distance_mm"][j].item()),
                        "{:.3f}".format(dataset["relative_heading_angle"][j].item())])
            for k in range(Nfish):
                row.extend(["{:.3f}".format(dataset["relative_orientation"][j, k].item())])
        for k in range(Nfish):
            row.extend(["{:.3f}".format(dataset["speed_array_mm_s"][j, k].item())])
        for k in range(Nfish):
            row.extend(["{:.3f}".format(dataset["radial_position_mm"][j, k].item())])
        row.extend(["{:d}".format(EdgeFlag[j]),
                    "{:d}".format(BadTrackFlag[j])])
        rows.append(",".join(row))
    
    # Write the CSV file
    with open(f"{dataset['dataset_name']}_basicMeasurements.csv", "w", newline="") as f:
        f.write(",".join(headers) + "\n")
        f.write("\n".join(rows))
    

def mark_behavior_frames_Excel(writer, dataset, key_list, sheet_name):
    """
    Create and fill in a sheet in an existing Excel file, marking all frames with behaviors
    found in this dataset.
    Args:
        writer (pandas.ExcelWriter): The ExcelWriter object representing the Excel file.
        dataset (dict): Dictionary with all dataset info.
        key_list (list): List of dictionary keys corresponding to each behavior to write.
        sheet_name (str): Name of the sheet to be created.
    Returns:
        N/A
    """
    
    # Create an empty dataframe with column names
    df = pd.DataFrame(columns=['Frame'] + key_list)
    
    # Add frame numbers to the 'Frame' column
    maxFrame = int(np.max(dataset["frameArray"]))
    df['Frame'] = range(1, maxFrame + 1)
    
    # Fill the dataframe with 'X' for each behavior
    for k in key_list:
        for run_idx in range(dataset[k]["combine_frames"].shape[1]):
            start_frame = dataset[k]["combine_frames"][0, run_idx]
            duration = dataset[k]["combine_frames"][1, run_idx]
            end_frame = start_frame + duration - 1
            df.loc[start_frame-1:end_frame-1, k] = 'X'.center(17)
    
    # Write the dataframe to the Excel file as a new sheet
    df.to_excel(writer, sheet_name=sheet_name, index=False)


def write_behaviorCounts_Excel(ExcelFileName, datasets, key_list, 
                              initial_keys, initial_strings):
    # Create a new Excel writer object
    writer = pd.ExcelWriter(ExcelFileName, engine='xlsxwriter')

    # Define the sheet names and corresponding data keys
    sheets = {
        "N_events": "N_events",
        "Durations (frames)": "total_duration",
        "Relative Durations": "relative_duration"
    }

    for sheet_name, data_key in sheets.items():
        # Prepare data for the current sheet
        data = []
        for j in range(len(datasets)):
            row = [datasets[j][key] for key in initial_keys]
            for key in key_list:
                row.append(datasets[j][key][data_key])
            data.append(row)

        # Create a DataFrame
        df = pd.DataFrame(data, columns=initial_strings + key_list)

        # Write the DataFrame to the Excel sheet
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save the Excel file
    writer.close()


def add_statistics_to_excel(file_path='behaviorCounts.xlsx'):
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    writer = pd.ExcelWriter(file_path, engine='openpyxl', mode='a')
    
    # Get the existing workbook
    book = writer.book
    
    for sheet_name in xls.sheet_names:
        # Read the sheet
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Separate non-numeric and numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate statistics for numeric columns
        mean = df[numeric_cols].mean()
        std = df[numeric_cols].std()
        n = df[numeric_cols].count()
        sem = std / np.sqrt(n)
        
        # Prepare statistics rows
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Std. Dev.', 'Std. Error of Mean', 'N datasets']
        })
        stats_df = pd.concat([stats_df, pd.DataFrame({
            col: [mean[col], std[col], sem[col], n[col]] for col in numeric_cols
        })], axis=1)
        
        # Clear the existing sheet
        if sheet_name in book.sheetnames:
            book.remove(book[sheet_name])
        
        # Write the original data
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Get the worksheet
        worksheet = writer.sheets[sheet_name]
        
        # Write statistics rows
        start_row = len(df) + 3  # +3 to leave a blank row
        for i, row in enumerate(stats_df.itertuples(index=False), start=start_row):
            for j, value in enumerate(row):
                cell = worksheet.cell(row=i, column=j+1, value=value)
        
        # Adjust column widths
        for idx, column in enumerate(worksheet.columns, start=1):
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    
    # Save the workbook
    writer.close()


def calc_distances_matrix(pos_current, pos_previous):
    """
    "Distances matrix" is the sum of inter-body distances between
    # each fish in one frame and each fish in another frame.

    Parameters
    ----------
    pos_current : numpy array; x and y body positions of each
                  fish in the "current" frame Shape: (Npos, Nfish, 2 (x and y))
    pos_previous :  numpy array; x and y body positions of each
                  fish in the "previous" frame Shape: (Npos, Nfish, 2 (x and y))

    Returns
    -------
    distances_matrix : matrix of the sum of interbody distances
            for fish j in current frame to fish k in the previous frame

    """
            
    # Avoiding loop using NumPy broadcasting and vectorized operations 
    # (From ChatGPT 3.5; tested in temp_testlinks.py)
    distances_matrix = np.sum(np.linalg.norm(pos_current[:, :, np.newaxis, :] 
                                             - pos_previous[:, np.newaxis, :, :], 
                                             axis=-1), axis=0)
    return distances_matrix



def link_weighted(pos_input, CSVcolumns, tol=0.001):
    """
    Re-do fish IDs (track linkage) based on whole-body distances
       and other weighted measures.
    Assesses bad tracking (zeros in track data), redundantly with
       get_bad_bodyTrack_frames, etc., but simple and self-contained
    Allow frame gap of bad tracking

    Author:   Raghuveer Parthasarathy
    Created on Thu Jan  4 11:26:30 2024
    Last modified on Thu Jan  4 11:26:30 2024
    
    Description
    -----------
    
    Inputs:
        pos_input: all the position information for a given expt.
            Possibly from dataset["all_data"] 
            Rows = frame numbers
            Columns = x, y, angle data -- see CSVcolumns
            Dim 3 = fish (2 fish)
        CSVcolumns: information on what the columns of pos_input are
           (same as columsn of dataset["all_data"] in other functions)
        tol : tolerance for "zero" (bad tracking), pixels
    
        
    Outputs:
        newIDs
    """
    
    # Number of frames, and number of fish
    Nframes = pos_input.shape[0]
    Nfish = pos_input.shape[2]
    
    # All positions: Nframes x N body positions x Nfish arrays for x, y
    body_x = pos_input[:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = pos_input[:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]

    # Identify frames with good tracking (head, body values nonzero)
    # Each array is Nframes x Nfish
    good_track_head = np.logical_and(body_x[:,0,:] > tol, 
                                     body_y[:,0,:] > tol)
    good_track_body = np.logical_and(np.all(body_x > tol, axis=1), 
                                     np.all(body_y > tol, axis=1))
    
    # Verify that first frame information looks ok: no zeros
    if np.any(good_track_head[0,:] == False):
        raise ValueError("Bad tracking (head) in the first frame! -- link_weighted()")
    if np.any(good_track_body[0,:] == False):
        raise ValueError("Bad tracking (head) in the first frame! -- link_weighted()")
    
    # Initial IDs
    IDs = np.tile(np.arange(Nfish), (Nframes, 1))    
    # IDs after re-linking; initialize to same
    newIDs = IDs.copy()
    
    # Keep track of last index with good body tracking of all fish
    last_good_index = 0
    # Iterate over frames starting from the second frame
    for j in range(1, Nframes):
        if np.all(good_track_body[j-1,:]) and np.all(good_track_body[j,:]):
            # Link between the current frame and the previous frame
            pos_current = np.stack([body_x[j, :, :], body_y[j, :, :]], axis=-1)  # Shape: (Npos, Nfish, 2 (x and y))
            pos_previous = np.stack([body_x[j-1, :, :], body_y[j-1, :, :]], axis=-1)  # Shape: (Npos, Nfish, 2)
            
            # Get distances_matrix, the sum of inter-body distances
            # between each fish in frame j and each fish in frame j-1
            distances_matrix = calc_distances_matrix(pos_current, 
                                                     pos_previous)
                        
            # Use the distances_matrix for assignment of IDs.
            # Note that this can be generalized to a weighted score 
            # incorporating other factors; calculate a weighted sum of
            # distances_matrix and whatever else, and apply the 
            # linear sum assignment to that.
            
            # Use linear_sum_assignment to find the optimal assignment
            row_indices, col_indices = linear_sum_assignment(distances_matrix)
            newIDs[j, :] = col_indices
            
            last_good_index = j # this frame is good
            
        elif np.all(good_track_body[j,:]):
            # Current frame is good, but previous is not
            # Link between the current frame and last good frame
            # See comments above
            pos_current = np.stack([body_x[j, :, :], body_y[j, :, :]], axis=-1)  # Shape: (Npos, Nfish, 2 (x and y))
            pos_previous = np.stack([body_x[last_good_index, :, :], body_y[last_good_index, :, :]], axis=-1)  # Shape: (Npos, Nfish, 2)
            distances_matrix = calc_distances_matrix(pos_current, 
                                                     pos_previous)
            row_indices, col_indices = linear_sum_assignment(distances_matrix)
            print(f'j = {j}, gap = {j - last_good_index - 1}')
            newIDs[j, :] = col_indices
            

    switchIndexes= np.where(np.any(IDs != newIDs, axis=1))[0].flatten()
    print("Frames for switched IDs (==index + 1)\n", switchIndexes+1)
    return IDs, newIDs
        

def repair_head_positions(datasets, CSVcolumns, tol=0.001):
    """ 
    Ignore "Index 0" head position and replace it with the interpolated 
    position from "Index 1-3" body positions. (Fixing ZebraZoom's unreliable
    head positions.) 
    Could use to replace the heading angle with the interpolated angle
    (see commented code), but there are offset issues, and I'll just use the
    previously-written fix_heading_angles()
    
    Check that body positions aren't zero; ignore otherwise
    Redundant code with get_bad_bodyTrack_frames(), for determining 
       nonzero position data
       
    Inputs:
        datasets : list of all dataset dictionaries. 
                    Note "all_data" is Nframes x data columns x Nfish
        CSVcolumns: information on what the columns of dataset["all_data"] are
        tol : tolerance for "zero" (bad tracking), pixels

    Output:
        datasets_repaired: In each, new ["all_data"] with 
                            repaired head positions in both "head_column_{x,y}"
                            and the first body column for x, y (same)
                                            
    """

    Npositions = CSVcolumns["body_Ncolumns"]
    
    # Create a new list for the repaired datasets
    # Avoiding confusing copy-in-place. Necessary?
    datasets_repaired = []

    for j in range(len(datasets)):
        dataset = datasets[j]
    
        # x and y are shape Nframes x Npositions x Nfish
        x = dataset["all_data"][:,CSVcolumns["body_column_x_start"] : 
                                (CSVcolumns["body_column_x_start"]+Npositions),:]
        y = dataset["all_data"][:,CSVcolumns["body_column_y_start"] :
                                (CSVcolumns["body_column_y_start"]+Npositions),:]
    
        # True if all x, y are nonzero; shape Nframes x Nfish
        good_bodyTrack = np.logical_and(np.all(np.abs(x)>tol, axis=1), 
                                        np.all(np.abs(y)>tol, axis=1))
        
        # Interpolation code, Claude.
       
        # Create the design matrix for linear regression
        X = np.array([1, 2, 3])
        X_design = np.vstack([np.ones_like(X), X]).T
        
        # Compute the pseudoinverse of X_design
        X_pinv = np.linalg.pinv(X_design)
        
        # Perform linear regression for x where mask is True
        beta_x = np.einsum('ij,kjl->kil', X_pinv, x[:, 1:4, :])
        x_interp = beta_x[:, 0, :] + beta_x[:, 1, :] * 0  # Interpolate at index 0
        x[:, 0, :] = np.where(good_bodyTrack, x_interp, x[:, 0, :])
        
        # Perform linear regression for y where mask is True
        beta_y = np.einsum('ij,kjl->kil', X_pinv, y[:, 1:4, :])
        y_interp = beta_y[:, 0, :] + beta_y[:, 1, :] * 0  # Interpolate at index 0
        y[:, 0, :] = np.where(good_bodyTrack, y_interp, y[:, 0, :])
        
        # Repair head position
        # Keeping same column redundancy as ZebraZoom
        dataset_repaired = dataset.copy()
        dataset_repaired["all_data"][:,CSVcolumns["body_column_x_start"],:] = x[:, 0, :]
        dataset_repaired["all_data"][:,CSVcolumns["body_column_y_start"],:] = y[:, 0, :]  
        dataset_repaired["all_data"][:,CSVcolumns["head_column_x"],:] = x[:, 0, :]
        dataset_repaired["all_data"][:,CSVcolumns["head_column_y"],:] = y[:, 0, :]    
        '''
        # Calculate heading angle using slopes from linear regression
        
        See Oct. 2024 notes
        Offset issue I don't feel like fixing (Oct. 16, 2024)
        # Get mean differences to determine quadrant
        mean_diff_x = np.mean(np.diff(x[:, 1:4, :], axis=1), axis=1)
        mean_diff_y = np.mean(np.diff(y[:, 1:4, :], axis=1), axis=1)
        
        # Calculate heading angle using arctan2 with both slopes and mean differences
        heading_angle = np.arctan2(beta_y[:, 1, :], beta_x[:, 1, :])
        heading_angle[heading_angle < 0.0] += 2*np.pi
            
        # Apply the mask to heading angle
        heading_angle = np.where(good_bodyTrack, heading_angle, np.nan)
    
        # Repair
        dataset_repaired["heading_angle"] = heading_angle
        '''
        
        datasets_repaired.append(dataset_repaired)
    
    return datasets_repaired

    
def repair_disjoint_heads(dataset, CSVcolumns, Dtol=3.0, tol=0.001):
    """ 
    Fix tracking data in which a fish has a "disjoint head" -- the head
    position is far from the body positions.
    
    Criteria:
    (i)  All head and body positions are nonzero
    (ii) the distance between positions 0 and 1 (head-body) is more than
         Dtol times the mean distance between positions j and j+1 
         for j = 1 to 9
    If these are met, replace position 0 with a point the same distance
    and orientation from point 1 as point 1 is from point 2 (i.e. linear
    extrapolation).
    
    Redundant code with get_bad_bodyTrack_frames(), for determining 
       nonzero position data
       
    Inputs:
        dataset : dataset dictionary. Note "all_data" contains all position
                  data, and has shape (Nframes x data columns x 2 fish)
        CSVcolumns: information on what the columns of dataset["all_data"] are
        Dtol : tolerance for head-body separation, default 3x mean
               separation distance between other body positions
        tol : tolerance for "zero" (bad tracking), pixels
        
    Output:
        dataset_repaired : overwrites ["all_data"] with repaired head positions
    """
    
    Npositions = CSVcolumns["body_Ncolumns"]
    # .copy() to avoid repairing in place
    # x and y are shape Nframes x Npositions x Nfish
    x = dataset["all_data"][:,CSVcolumns["body_column_x_start"] : 
                            (CSVcolumns["body_column_x_start"]+Npositions),:].copy()
    y = dataset["all_data"][:,CSVcolumns["body_column_y_start"] :
                            (CSVcolumns["body_column_y_start"]+Npositions),:].copy()
    angles = dataset["heading_angle"].copy()

    # True if all x, y are nonzero; shape Nframes x Nfish
    good_bodyTrack = np.logical_and(np.all(np.abs(x)>tol, axis=1), 
                                    np.all(np.abs(y)>tol, axis=1))
    
    # Look at distance between head and body, compare to body-body distances
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    dr_01 = np.sqrt(dx[:,0,:]**2 + dy[:,0,:]**2) # head-body distance
    dr_body = np.sqrt(dx[:,1:,:]**2 + dy[:,1:,:]**2)
    mean_dr_body = np.mean(dr_body,axis=1)
    bad_01_distance = dr_01 > Dtol*mean_dr_body # disjoint head; Nframes x Nfish
    # Identify frames with large head-body distance but good (nonzero) tracking
    # Will make a separate array for each fish, for readability
    Nfish = good_bodyTrack.shape[1]
    for j in range(Nfish):
        disjoint_idx = np.where(np.logical_and(good_bodyTrack[:,j], 
                                               bad_01_distance[:,j]))[0]
        # disjoint_head_idx = np.array(dataset["frameArray"][badidx])
        # print(disjoint_idx)
        # print(x[2108,0,j])
        x[disjoint_idx,0,j] = 1.25*x[disjoint_idx,1,j] - 0.25*x[disjoint_idx,2,j]
        y[disjoint_idx,0,j] = 1.25*y[disjoint_idx,1,j] - 0.25*y[disjoint_idx,2,j]
        # print(x[2108,0,j])
        angles[disjoint_idx,j] = np.arctan2(y[disjoint_idx,1,j]- y[disjoint_idx,2,j], 
                                            x[disjoint_idx,1,j]- x[disjoint_idx,2,j])

    # Repair
    dataset_repaired = dataset.copy()
    dataset_repaired["all_data"][:,CSVcolumns["body_column_x_start"] : 
                        (CSVcolumns["body_column_x_start"]+Npositions),:] = x
    dataset_repaired["all_data"][:,CSVcolumns["body_column_y_start"] :
                        (CSVcolumns["body_column_y_start"]+Npositions),:] = y
    dataset_repaired["heading_angle"] = angles
    
    return dataset_repaired

def repair_double_length_fish(dataset, CSVcolumns, 
                              lengthFactor = [1.5, 2.5], tol=0.001):
    """ 
    Fix tracking data in which there is only one identified fish, with the 
    10 body positions spanning two actual fish and overall length 
    roughly twice the actual single fish length.
    Replace one fish with the first 5 positions, interpolated to 10 pts
    and the other with the second 5, interpolated along with the heading 
    angle
    
    Inputs:
        dataset : dataset dictionary. Note "all_data" contains all position
                  data, and has shape (Nframes x data columns x 2 fish)
                  Note that dataset["fish_length_array_mm"] 
                     contains fish lengths (mm)
        CSVcolumns: information on what the columns of dataset["all_data"] are
        lengthFactor : a list with two values; 
                       split fish into two if length is between these
                       factors of median fish length
        tol : tolerance for "zero" (bad tracking), pixels
        
    Output:
        dataset_repaired : overwrites ["all_data"] with repaired head 
        positions
    """
    
    # median fish length (px) for each fish; take average across fish
    mean_fish_length_mm = np.mean(np.median(dataset["fish_length_array_mm"], axis=0))
    print('mean fish length (mm): ', mean_fish_length_mm)
    
    # .copy() to avoid repairing in place
    Npositions = CSVcolumns["body_Ncolumns"]
    x = dataset["all_data"][:,CSVcolumns["body_column_x_start"] : 
                            (CSVcolumns["body_column_x_start"]+Npositions),:].copy()
    y = dataset["all_data"][:,CSVcolumns["body_column_y_start"] :
                            (CSVcolumns["body_column_y_start"]+Npositions),:].copy()
    
    # True if all x, y are zero; shape Nframes x Nfish
    good_bodyTrack = np.logical_and(np.all(np.abs(x)>tol, axis=1), 
                                    np.all(np.abs(y)>tol, axis=1))
    # Indices of frames in which  only one fish was tracked 
    rows_with_one_tracked = np.where(np.sum(good_bodyTrack, axis=1) == 1)[0]
    print('Frame indexes with one fish \n', rows_with_one_tracked)
    # Column indices (i.e. fish) where True values exist
    oneFish_indices = np.argmax(good_bodyTrack[rows_with_one_tracked,:], axis=1)
    print('One fish indices\n', oneFish_indices)
    # Calculate length ratios
    lengthRatios = dataset["fish_length_array_mm"][rows_with_one_tracked, 
                                                oneFish_indices] / mean_fish_length_mm
    # Find frame indices where length ratios meet the condition
    doubleLength_indices = rows_with_one_tracked[np.logical_and(lengthFactor[0] < lengthRatios, 
                             lengthRatios < lengthFactor[1])]

    # Column indices (i.e. fish) where True values exist, only for these
    # frames. Using same variable name
    oneFish_indices = np.argmax(good_bodyTrack[doubleLength_indices,:], axis=1)

    # Repair
    # Note that it doesn't matter which ID is which, since we'll re-link later
    dataset_repaired = dataset.copy()
    midPosition = int(np.floor(Npositions/2.0))  # 5 for usual 10 body positions
    interpIndices = np.linspace(0, Npositions-1, num=midPosition).astype(int) 
    for j, frameIdx in enumerate(doubleLength_indices):
        print('Double length Frame Idx: ', frameIdx)
        # one fish from the first 5 positions.
        x_first = x[frameIdx,0:midPosition,oneFish_indices[j]]
        y_first = y[frameIdx,0:midPosition,oneFish_indices[j]]
        # print('Frame Index: ', frameIdx)
        print('which fish: ', oneFish_indices[j])
        print(x_first)
        print(y_first)
        # print(np.arange(0,midPosition))
        x0_new = np.interp(np.arange(0,Npositions), interpIndices, x_first)
        y0_new = np.interp(np.arange(0,Npositions), interpIndices, y_first)
        angles0_new = np.arctan2(y0_new[0]- y0_new[2], x0_new[0]- x0_new[2])
        
        # the other fish from the last 5 positions.
        x_last = x[frameIdx,midPosition:, oneFish_indices[j]]
        y_last = y[frameIdx,midPosition:, oneFish_indices[j]]
        x1_new = np.interp(np.arange(0,Npositions), interpIndices, x_last)
        y1_new = np.interp(np.arange(0,Npositions), interpIndices, y_last)
        angles1_new = np.arctan2(y1_new[0]- y1_new[2], x1_new[0]- x1_new[2])

        dataset_repaired["all_data"][frameIdx,CSVcolumns["body_column_x_start"] : 
                            (CSVcolumns["body_column_x_start"]+Npositions),0] = x0_new
        dataset_repaired["all_data"][frameIdx,CSVcolumns["body_column_y_start"] :
                            (CSVcolumns["body_column_y_start"]+Npositions),0] = y0_new
        dataset_repaired["all_data"][frameIdx,CSVcolumns["body_column_x_start"] : 
                            (CSVcolumns["body_column_x_start"]+Npositions),1] = x1_new
        dataset_repaired["all_data"][frameIdx,CSVcolumns["body_column_y_start"] :
                            (CSVcolumns["body_column_y_start"]+Npositions),1] = y1_new
        dataset_repaired["heading_angle"][:, 0] = angles0_new
        dataset_repaired["heading_angle"][:, 1] = angles1_new
    
    return dataset_repaired
 

def combine_all_values_constrained(datasets, keyName='speed_array_mm_s', 
                                   keyIdx = None,
                                   constraintKey=None, constraintRange=None,
                                   constraintIdx = 0, dilate_plus1=True):
    """
    Loop through each dataset, get values of some numerical property
    in datasets[j][keyName], and collect all these in a list of 
    numpy arrays, one array per dataset. 
	Ignore, in each dataset, "bad tracking" frames. 
       If "dilate_plus1" is True, dilate the bad frames +1; do this for speed values, since bad tracking affects adjacent frames!
    if datasets[j][keyName] is multi-dimensional, return the
        multidimensional array for each dataset
        (i.e. a list of these multidimensional arrays as output)
        unless keyIdx is specified, indicating the column to extract,
        or an operation like "min", "max", "mean"
    Optional: combine only 
        if the corresponding values of the 'constraintKey' (index or instructions
        given by constraintIdx) are within 
        the 'constraintRange'. For example: get all speed values for frames
        in which inter-fish-distance is below 5 mm.
        If 'constraintRange' is empty or None, do not apply the constraint.
    
    Return a list of numpy arrays containing all values.

    List contains one numpy array per dataset, of size (# values, # fish d.o.f.)
        E.g. if there are two fish, speed would have 2 dof (speed of each fish)
                and inter-fish distance would have 1 dof
    (Can later concatenate from all datasets into one numpy array with 
     "np.concatenate()").
    Output can be used, for example, for making a histogram of speeds or 
    inter-fish distance.

    
    Parameters
    ----------
    datasets : list of dictionaries containing all analysis.
    keyName : the key to combine (e.g. "speed_array_mm_s")
    keyIdx : integer or string, or None, used by get_values_subset(). 
                If keyIdx is:
                    None: If datasets[j][keyName] is a multidimensional array, 
                       return the full array (minus bad frames, constraints)
                    an integer: extract that column
                       (e.g. datasets[12]["speed_array_mm_s"][:,keyIdx])
                    a string , use the operation "min", "max",
                       or "mean", along axis==1 (e.g. for fastest fish)
    constraintKey : the key to use for the constraint (e.g. "interfish_distance_mm")
                    datasets[j][constraintKey] should be a single, 1D numpy array
                    or should be a (N, Nfish) numpy array with the column
                    to use specified by constraintIdx (see below)
                    If None, don't apply constraint. (Still remove bad Tracking)
    constraintRange : a numpy array with two numerical elements specifying the range to filter on
    constraintIdx : integer or string.
                    If the constraint is a multidimensional array, will use
                    dimension constraintIdx if constraintIdx is an integer 
                    (e.g. datasets[12]["speed_array_mm_s"][:,constraintIdx])
                    or the "operation" if constraintIdx is a string,
                    "min", "max", or "mean",
                    (e.g. max of datasets[12]["speed_array_mm_s"] along axis==1
                     if constraintIdx is "max")
                    If None, won't apply constraint
    dilate_plus1 : If True, dilate the bad frames +1; see above.
    
    Returns
    -------
    values_all_constrained : list of numpy arrays of all values in all 
       datasets that satisfy the constraint; can concatenate.

    """
    Ndatasets = len(datasets)
    # print(f'\nCombining values of {keyName} for {Ndatasets} datasets...')
    
    
    values_all_constrained = []
    
    if constraintKey is None or constraintRange is None \
                             or len(constraintRange) != 2:
        # If constraintRange is empty or invalid, return all values,
        # minus those from bad tracking frames.
        for j in range(Ndatasets):
            frames = datasets[j]["frameArray"]
            badTrackFrames = datasets[j]["bad_bodyTrack_frames"]["raw_frames"]
            if dilate_plus1:
                dilate_badTrackFrames = np.concatenate((badTrackFrames, badTrackFrames + 1))
                bad_frames_set = set(dilate_badTrackFrames)
            else:
                bad_frames_set = set(badTrackFrames)
            
            good_frames_mask = np.isin(frames, list(bad_frames_set), invert=True)
            values = get_values_subset(datasets[j][keyName], keyIdx)
            values_this_set = values[good_frames_mask, ...]
            values_all_constrained.append(values_this_set)
        
        return values_all_constrained
    
    # print(f'    ... with constraint on {constraintKey}')
    for j in range(Ndatasets):
        frames = datasets[j]["frameArray"]
        badTrackFrames = datasets[j]["bad_bodyTrack_frames"]["raw_frames"]
        if dilate_plus1:
            dilate_badTrackFrames = np.concatenate((badTrackFrames, badTrackFrames + 1))
            bad_frames_set = set(dilate_badTrackFrames)
        else:
            bad_frames_set = set(badTrackFrames)
        
        # Filter values based on constraint
        good_frames_mask = np.isin(frames, list(bad_frames_set), invert=True)
        constraint_array = get_values_subset(datasets[j][constraintKey], 
                                             constraintIdx)
        constrained_mask = (constraint_array >= constraintRange[0]) \
                            & (constraint_array <= constraintRange[1])
        constrained_mask = constrained_mask.flatten()
        values = get_values_subset(datasets[j][keyName], keyIdx)
        values_this_set = values[good_frames_mask & constrained_mask, ...]
        values_all_constrained.append(values_this_set)
    
    return values_all_constrained


def get_values_subset(values_all, idx):
    """
    Extract subset of a values array, e.g. datasets[j][constraintKey]
    to use later, e.g. as a constraint array, specifying which column to
    use, or min or max values.

    Parameters
    ----------
    values_all : numpy array, probably from datasets[j][constraintKey]
                 if using this to extract an array to use as a constraint,
                 of shape (N,) or shape (N,1), or (N, Nfish)
                 If shape[1]=1, output values_all_constrained = values_all
                 
    idx : either an integer or a string, probably from keyIdx or constraintIdx.
          If idx is:
             None: ignore idx; don't apply constraint; return values_all
             an integer: the column of values_all to use as the 
                constraint array
             a string, use the operation "min", "max", or "mean", 
                along axis==1 (e.g. for fastest fish), or or "all" to use 
                "all" (same as idx==None)
          if None, ignore idx; don't apply constraint

    Returns
    -------
    values_all_constrained : numpy array, the subset of values_all,
                or min or max, etc. Same axis=0 shape as values_all

    """
    if values_all.ndim == 1:
        # Only one array, so output = input; ignore idx
        values_all_constrained = values_all
    elif idx is None:
        # No constraint
        values_all_constrained = values_all    
    else: 
        if type(idx)==str:
            if idx == 'min':
                values_all_constrained = np.min(values_all, axis=1)
            elif idx == 'max':
                values_all_constrained = np.max(values_all, axis=1)
            elif idx == 'mean':
                values_all_constrained = np.mean(values_all, axis=1)
            elif idx == 'all':
                values_all_constrained = values_all
            else:
                raise ValueError(f"Invalid index string {idx}")
        else:
            if (idx+1) > values_all.shape[1]:
                raise ValueError(f"subset index {idx} is too large for the size" + 
                                 f"of the array, {values_all.shape[1]}")
            values_all_constrained =  values_all[:,idx]

    return values_all_constrained
    

def plot_probability_distr(x_list, bin_width=1.0, bin_range=[None, None], 
                           xlabelStr='x', titleStr='Probability density',
                           yScaleType = 'log', flatten_dataset = False,
                           ylim = None, polarPlot = False, 
                           outputFileName = None):
    """
    Plot the probability distribution (normalized histogram) 
    for each array in x_list (semi-transparent)
    and for the concatenated array of all items in x_list (black)
    Can plot in polar coordinates – useful for angle distributions.
    Inputs:
       x_list : list of numpy arrays
       bin_width : bin width
       bin_range : list of smallest and largest bin edges; if None
                   use min and max of all arrays combined. 
                   Or if None and polarPlot==True use nearest
                   integer multiple of pi below/above min and max 
       xlabelStr : string for x axis label
       titleStr : string for title
       yScaleType : either "log" or "linear"
       flatten_dataset : if true, flatten each dataset's array for 
                           individual dataset plots. If false, plot each
                           array column (fish, probably) separately
       ylim : (optional) tuple of min, max y-axis limits
       polarPlot : if True, use polar coordinates for the histogram.
                   Will not plot x and y labels.
                   Strongly reommended to set y limit (which will set r limit)
       outputFileName : if not None, save the figure with this filename 
                       (include extension)
    """ 
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(polar=polarPlot)
    
    # Concatenate all arrays for the overall, combined plot
    x_all = np.concatenate([arr.flatten() for arr in x_list])
    
    slight_theta_shift = 0.01 # for axis range of polar plots
    # Determine bin range if not provided
    if bin_range[0] is None:
        if polarPlot==True:
            bin_range[0] = (np.nanmin(x_all) // np.pi)*np.pi # floor * pi
        else:
            bin_range[0] = x_all.min()
    if bin_range[1] is None:
        if polarPlot==True:
            # next mult of pi
            bin_range[1] = (((np.nanmax(x_all)-slight_theta_shift) // np.pi)+1)*np.pi
        else:
            bin_range[1] = np.nanmax(x_all)
    Nbins = np.round((bin_range[1] - bin_range[0])/bin_width + 1).astype(int)
    bin_edges = np.linspace(bin_range[0], bin_range[1], num=Nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot individual distributions
    alpha_each = np.max((0.7/len(x_list), 0.05))
    for i, x in enumerate(x_list):
        if flatten_dataset:
            x = x.flatten()
        if x.ndim==1:
            Nsets = 1
            counts, _ = np.histogram(x.flatten(), bins=bin_edges)
            prob_dist = counts / np.sum(counts) / bin_width
            datasetLabel = f'Dataset {i+1}'
            ax.plot(bin_centers, prob_dist, color='black', 
                    alpha=alpha_each, label=datasetLabel)
        else:
            Nsets = x.shape[1] # number of measures per dataset, e.g. number of fish
            for j in range(Nsets):
                counts, _ = np.histogram(x[:,j].flatten(), bins=bin_edges)
                prob_dist = counts / np.sum(counts) / bin_width
                datasetLabel = f'Dataset {i+1}: {j+1}'
                ax.plot(bin_centers, prob_dist, color='black', 
                        alpha=alpha_each, label=datasetLabel)

    
    # Plot concatenated distribution
    counts_all, _ = np.histogram(x_all, bins=bin_edges)
    prob_dist_all = counts_all / np.sum(counts_all) / bin_width
    plt.plot(bin_centers, prob_dist_all, color='black', linewidth=2, 
             label='All Datasets')
    if not polarPlot:
        plt.xlabel(xlabelStr, fontsize=16)
        plt.ylabel('Probability density', fontsize=16)
    plt.title(titleStr, fontsize=18)
    if polarPlot:
        ax.set_thetalim(bin_range[0], bin_range[1])
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale(yScaleType)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.legend()
    plt.show()
    if outputFileName != None:
        plt.savefig(outputFileName, bbox_inches='tight')
    
def calculate_autocorr(data, n_lags):
    """
    Calculate autocorrelation for the entire dataset.
    Helper function for calculate_value_autocorr_oneSet()
    """
    data_centered = data - np.mean(data)
    autocorr = signal.correlate(data_centered, data_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr /= (np.var(data) * len(data))
    return autocorr[:n_lags]


def calculate_block_autocorr(data, n_lags, window_size):
    """
    Calculate autocorrelation using non-overlapping blocks.
    Helper function for calculate_value_autocorr_oneSet()
    """
    num_blocks = len(data) // window_size
    block_autocorrs = []
    
    for i in range(num_blocks):
        start = i * window_size
        end = (i + 1) * window_size
        block = data[start:end]
        block_centered = block - np.mean(block)
        block_autocorr = signal.correlate(block_centered, block_centered, mode='full')
        block_autocorr = block_autocorr[len(block_autocorr)//2:]
        block_autocorr /= (np.var(block) * len(block))
        block_autocorrs.append(block_autocorr[:n_lags])
    
    # Average the autocorrelations from all blocks
    avg_autocorr = np.mean(block_autocorrs, axis=0)
    
    return avg_autocorr


def calculate_crosscorr(data1, data2, n_lags):
    """Calculate cross-correlation for the entire dataset."""
    data1_centered = data1 - np.mean(data1)
    data2_centered = data2 - np.mean(data2)
    crosscorr = signal.correlate(data1_centered, data2_centered, mode='full')
    crosscorr = crosscorr[len(crosscorr)//2-n_lags//2:len(crosscorr)//2+n_lags//2+1]
    crosscorr /= (np.std(data1) * np.std(data2) * len(data1))
    return crosscorr

def calculate_block_crosscorr(data1, data2, n_lags, window_size):
    """Calculate cross-correlation using non-overlapping blocks."""
    num_blocks = len(data1) // window_size
    block_crosscorrs = []
    
    for i in range(num_blocks):
        start = i * window_size
        end = (i + 1) * window_size
        block1 = data1[start:end]
        block2 = data2[start:end]
        block1_centered = block1 - np.mean(block1)
        block2_centered = block2 - np.mean(block2)
        block_crosscorr = signal.correlate(block1_centered, block2_centered, mode='full')
        block_crosscorr = block_crosscorr[len(block_crosscorr)//2-n_lags//2:len(block_crosscorr)//2+n_lags//2+1]
        block_crosscorr /= (np.std(block1) * np.std(block2) * len(block1))
        block_crosscorrs.append(block_crosscorr)
    
    return np.mean(block_crosscorrs, axis=0)

def calculate_value_corr_oneSet(dataset, keyName='speed_array_mm_s', 
                                corr_type='auto', dilate_plus1=True, 
                                t_max=10, t_window=None):
    """
    For a *single* dataset, calculate the auto or cross-correlation of the numerical
    property in the given key (e.g. speed)
    Ignore "bad tracking" frames. If a frame is in the bad tracking list,
    replace the value with a Gaussian random number with the same mean, 
    std. dev. as the frames not in the bad tracking list. This avoids
    having to figure out how to deal with non-uniform time lags.
    If "dilate_plus1" is True, dilate the bad frames +1.
    Output is a numpy array with dim 1 corresponding to each fish.
    
    Parameters
    ----------
    dataset : single analysis dataset
    keyName : the key to combine (e.g. "speed_array_mm_s")
    corr_type : 'auto' for autocorrelation, 'cross' for cross-correlation (only for Nfish==2)
    dilate_plus1 : If True, dilate the bad frames +1
    t_max : max time to consider for autocorrelation, seconds.
    t_window : size of sliding window in seconds. If None, don't use a sliding window.
    
    Returns
    -------
    corr : correlation of desired property, numpy array of
                    shape (#time lags + 1 , Nfish) for autocorrelation
                    shape (#time lags + 1 , 1) for autocorrelation
    t_lag : time lag array, seconds (including zero)
    """
    value_array = dataset[keyName]
    Nframes, Nfish = value_array.shape
    fps = dataset["fps"]
    badTrackFrames = dataset["bad_bodyTrack_frames"]["raw_frames"]
    if dilate_plus1:
        dilate_badTrackFrames = dilate_frames(badTrackFrames, 
                                              dilate_frames=np.array([1]))
        bad_frames_set = set(dilate_badTrackFrames)
    else:
        bad_frames_set = set(badTrackFrames)
     
    if corr_type == 'auto':
        t_lag = np.arange(0, t_max + 1.0/fps, 1.0/fps)
        n_lags = len(t_lag)
        corr = np.zeros((n_lags, Nfish))
    elif corr_type == 'cross':
        t_lag = np.arange(-t_max, t_max + 1.0/fps, 1.0/fps)
        n_lags = len(t_lag)
        if Nfish != 2:
            raise ValueError("Cross-correlation is only supported for Nfish==2")
        corr = np.zeros(n_lags)
    else:
        raise ValueError("corr_type must be 'auto' or 'cross'")
    
    for fish in range(Nfish):
        fish_value = value_array[:, fish].copy()
        
        good_frames = [speed for i, speed in enumerate(fish_value) if i not in bad_frames_set]
        mean_value = np.mean(good_frames)
        std_value = np.std(good_frames)
        
        for frame in bad_frames_set:
            if frame < Nframes:
                fish_value[frame] = np.random.normal(mean_value, std_value)

        if corr_type == 'auto':
            if t_window is None:
                fish_corr = calculate_autocorr(fish_value, n_lags)
            else:
                window_size = int(t_window * fps)
                fish_corr = calculate_block_autocorr(fish_value, n_lags, 
                                                     window_size)
            corr[:, fish] = fish_corr
            
        
    if corr_type == 'cross':
        if t_window is None:
            corr = calculate_crosscorr(value_array[:, 0], value_array[:, 1], n_lags)
        else:
            window_size = int(t_window * fps)
            corr = calculate_block_crosscorr(value_array[:, 0], value_array[:, 1], n_lags, window_size)
    
    return corr, t_lag


def calculate_value_corr_all(datasets, keyName = 'speed_array_mm_s',
                             corr_type='auto', dilate_plus1 = True, 
                             t_max = 10, t_window = None, fpstol = 1e-6):
    """
    Loop through each dataset, call calculate_value_corr_oneSet() to
    calculate the auto- or cross-corrlation of the numerical
    property in the given key (e.g. speed
    and collect all these in a list of numpy arrays. 
    Note that calculate_value_autocorr_all() ignores bad tracking frames.
    

    List contains one numpy array per dataset, of size (# values, # fish d.o.f.)
        E.g. if there are two fish, speed would have 2 dof (speed of each fish)
                and inter-fish distance would have 1 dof
    
    Parameters
    ----------
    datasets : list of dictionaries containing all analysis. 
	keyName : the key to combine (e.g. "speed_array_mm_s")
    corr_type : 'auto' for autocorrelation, 'cross' for cross-correlation (only for Nfish==2)
	dilate_plus1 :  If "dilate_plus1" is True, dilate the bad frames +1; see above. 
    t_max : max time to consider for autocorrelation, seconds. 
                (Calculates for all time lags, but just ouputs to t_max.) 
                Uses "fps" key to convert time to frames.
    t_window : size of sliding window in seconds. If None, don't use a sliding window.
    fpstol = relative tolerance for checking that all fps are the same 

    Returns
    -------
    autocorr_all : list of numpy arrays of all autocorrelations in all datasets
    t_lag : time lag array, seconds (including zero); 
       should be same for all datasets -- check if fps is same 
       (to tolerance fpstol for all datasets)

    """
    Ndatasets = len(datasets)
    print(f'\nCombining {corr_type}-correlations of {keyName} for {Ndatasets} datasets')
    autocorr_all = []
    # Check that all fps are the same, so that all time lag arrays will be the same
    fps_all = np.zeros((Ndatasets,))
    for j in range(Ndatasets):
        fps_all = datasets[j]["fps"]
    good_fps = np.abs(np.std(fps_all)/np.mean(fps_all)) < fpstol
    
    if not good_fps:
        raise ValueError("fps values are not the same across datasets!")

    for j in range(Ndatasets):
        (autocorr_one, t_lag) = calculate_value_corr_oneSet(datasets[j], 
                                            keyName = keyName, 
                                            corr_type=corr_type,
                                            dilate_plus1 = dilate_plus1, 
                                            t_max = t_max, t_window = t_window)
        autocorr_all.append(autocorr_one)
        
    return autocorr_all, t_lag



def plot_function_allSets(y_list, x_array = None, 
                           xlabelStr='x', ylabelStr='y', titleStr='Value',
                           average_in_dataset = False,
                           outputFileName = None):
    """
    Plot some function that has been calculated for all datasets, 
    such as the autocorrelation, for the average array from all 
    items in y_list and all individual items (semi-transparent)
    x_array (x values) to plot will be the same for all datasets
    
    Inputs:
       y_list : list of numpy arrays
       x_array : x-values, same for all datasets. If None, just use indexes
       ylabelStr : string for x axis label
       titleStr : string for title
       average_in_dataset : if true, average each dataset's arrays for 
                            the individual dataset plots
       outputFileName : if not None, save the figure with this filename 
                       (include extension)
    """ 
    # Ensure y_list is a list of numpy arrays
    if not isinstance(y_list, list):
        raise ValueError("y_list must be a list of numpy arrays")

    # Determine Nfish based on the shape of the first array in y_list
    # Note that Nfish is actually # dof
    if len(y_list[0].shape) == 1:
        Nfish = 1
    else:
        Nfish = y_list[0].shape[1]

    # Determine N (number of x points)
    N = y_list[0].shape[0]

    # Create x_array if not provided
    if x_array is None:
        x_array = np.arange(0, N)
    elif len(x_array) != N:
        raise ValueError("Length of x_array must match the number of rows in y arrays")

    if Nfish == 1:
        y_mean = np.mean([y.flatten() for y in y_list], axis=0)
    else:
        y_mean = np.mean([np.mean(y, axis=1) for y in y_list], axis=0)
    
    if x_array is None:
        x_array = np.arange(len(y_list[1]))

    plt.figure(figsize=(12, 6))

    # Plot individual y data
    alpha_each = np.max((0.7/len(y_list), 0.05))
    for y in y_list:
        if Nfish==1:
            plt.plot(x_array, y, color='black', alpha=alpha_each)            
        else:
            for fish in range(Nfish):
                plt.plot(x_array, y[:, fish], color='black', alpha=alpha_each)

    # Plot y_mean
    plt.plot(x_array, y_mean, color='black', linewidth=2, label='Mean')

    plt.xlabel(xlabelStr, fontsize=16)
    plt.ylabel(ylabelStr, fontsize=16)
    plt.title(titleStr, fontsize=20)    # plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.legend()
    plt.show()
    if outputFileName != None:
        plt.savefig(outputFileName, bbox_inches='tight')

def make_2D_histogram(datasets, keyNames = ('speed_array_mm_s', 'head_head_distance_mm'), 
                      keyIdx = (None, None), 
                      constraintKey=None, constraintRange=None, constraintIdx = 0,
                      dilate_plus1=True, bin_ranges=None, Nbins=(20,20),
                      titleStr = None, colorRange = None, outputFileName = None):
    """
    Create a 2D histogram plot of the values from two keys in the 
    given datasets. Combine all the values across datasets.
    Uses combine_all_values_constrained() to pick a subset of the keys or to 
    apply a constraint to the values to plot (optional)
    
    Parameters:
    datasets (list): List of dictionaries containing the analysis data.
    keyNames (tuple of 2 str): Key names for the first and second values to plot.
    keyIdx : tuple of 2 items, integer or string or None indicating subset of 
             value array, using get_values_subset(datasets[j][keyName], keyIdx)
            If keyIdx is:
                None: If datasets[j][keyName] is a multidimensional array, 
                   return the full array (minus bad frames, constraints)
                an integer: extract that column
                   (e.g. datasets[12]["speed_array_mm_s"][:,keyIdx])
                a string , use the operation "min", "max",
                   or "mean", along axis==1 (e.g. for fastest fish)
    constraintKey (str): Key name for the constraint, 
        or None to use no constraint. Apply the same constraint to both keys.
        see combine_all_values_constrained()
    constraintRange (np.ndarray): Numpy array with two elements specifying the constraint range, or None to use no constraint.
    constraintIdx : integer or string.
                    If the constraint is a multidimensional array, will use
                    dimension constraintIdx if constraintIdx is an integer 
                    or the "operation" if constraintIdx is a string,
                    "min", "max", or "mean",
                    If None, won't apply constraint    
                    see combine_all_values_constrained()
    dilate_plus1 (bool): If True, dilate the bad frames +1; see above.
    bin_ranges (tuple): Optional tuple of two lists, specifying the (min, max) range for the bins of value1 and value2.
    Nbins (tuple): Optional tuple of two integers, number of bins for value1 and value2
    titleStr : title string; if None use Key names
    colorRange : Optional tuple of (vmin, vmax) for the histogram "v axis" range
    outputFileName : if not None, save the figure with this filename 
                     (include extension)    
    Returns:
    None
    """
    if len(keyNames) != 2:
        raise ValueError("There must be two keys for the 2D histogram!") 
        
    # Get the values for each key with the constraint applied
    values1 = combine_all_values_constrained(datasets, keyNames[0], keyIdx=keyIdx[0],
                                             constraintKey=constraintKey, 
                                             constraintRange=constraintRange, 
                                             constraintIdx=constraintIdx,
                                             dilate_plus1=dilate_plus1)
    values2 = combine_all_values_constrained(datasets, keyNames[1], keyIdx=keyIdx[1],
                                             constraintKey=constraintKey, 
                                             constraintRange=constraintRange, 
                                             constraintIdx=constraintIdx,
                                             dilate_plus1 = dilate_plus1)
    
    # Flatten the values and handle different dimensions
    values1_all = []
    values2_all = []
    for v1, v2 in zip(values1, values2):
        M1 = 1 if v1.ndim == 1 else v1.shape[1] if v1.ndim > 1 else 1 if v1.ndim == 2 and v1.shape[1] == 1 else None
        M2 = 1 if v2.ndim == 1 else v2.shape[1] if v2.ndim > 1 else 1 if v2.ndim == 2 and v2.shape[1] == 1 else None
        
        if M1 is None or M2 is None or ((M1 != M2) and min(M1, M2) > 1):
            print(f'M values: {M1}, {M2}')
            raise ValueError("Values for the two keys are not commensurate. 2D histogram cannot be created.")
        
        if M1 > 1 and M2 == 1:
            Nfish = M1
            values2_all.append(np.repeat(v2.flatten(), Nfish))
            values1_all.append(v1.flatten())
        elif M2 > 1 and M1 == 1:
            Nfish = M2
            values1_all.append(np.repeat(v1.flatten(), Nfish))
            values2_all.append(v2.flatten())
        else:
            values1_all.append(v1.flatten())
            values2_all.append(v2.flatten())
    
    # Concatenate the flattened values
    values1_all = np.concatenate(values1_all)
    values2_all = np.concatenate(values2_all)
    
    # Determine the bin ranges
    if bin_ranges is None:
        value1_min, value1_max = np.nanmin(values1_all), np.nanmax(values1_all)
        value2_min, value2_max = np.nanmin(values2_all), np.nanmax(values2_all)
        # print('Values: ', value1_min, value1_max, value2_min, value2_max)
        # Expand a bit!
        d1 = (value1_max - value1_min)/Nbins[0]
        d2 = (value2_max - value2_min)/Nbins[1]
        value1_min = value1_min - d1/2.0
        value1_max = value1_max + d1/2.0
        value2_min = value2_min - d2/2.0
        value2_max = value2_max + d2/2.0
    else:
        value1_min, value1_max = bin_ranges[0]
        value2_min, value2_max = bin_ranges[1]
    
    # Create the 2D histogram
    fig, ax = plt.subplots(figsize=(8, 6))

    hist, xedges, yedges = np.histogram2d(values1_all, values2_all, 
                                          bins=Nbins, 
                                          range=[(value1_min, value1_max), 
                                                 (value2_min, value2_max)])
    # Normalize the histogram
    hist = hist / hist.sum()
    
    # Plot the 2D histogram
    # X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
    X, Y = np.meshgrid(0.5*(xedges[1:] + xedges[:-1]), 
                       0.5*(yedges[1:] + yedges[:-1]), indexing='ij')
    # Create the 2D histogram and the colorbar
    if colorRange is None:
        cbar = fig.colorbar(ax.pcolormesh(X, Y, hist, shading='nearest'), ax=ax)
    else:
        cbar = fig.colorbar(ax.pcolormesh(X, Y, hist, shading='nearest',
                               vmin=colorRange[0], vmax = colorRange[1]), ax=ax)
            
    ax.set_xlabel(keyNames[0], fontsize=16)
    ax.set_ylabel(keyNames[1], fontsize=16)
    if titleStr is None:
        titleStr = f'2D Histogram of {keyNames[0]} vs {keyNames[1]}'
    ax.set_title(titleStr, fontsize=18)
    cbar.set_label('Normalized Count')
    plt.show()
    if outputFileName != None:
        plt.savefig(outputFileName, bbox_inches='tight')


def behaviorFrameCount_one(dataset, keyList, normalizeCounts = False):
    """
    Count behaviors at each frame for one dataset.
    
    Parameters:
        dataset : single dictionary containing the analysis data.
        keyList : a list of strings that contains behavior key names to count
        normalizeCounts : if True, normalize by Nframes to return a probability, 
                    rather than a count, of behavior events
                    
    Returns: 
        behaviorCount : (Nframes,) array, total count these behaviors at each frame
                        if normalizeCounts==True, this is counts/Nframes
        behaviorCount_eachBehavior : (Nframes x Nkeys) count of each behavior (0 or 1) at each frame
        
    """
    Nkeys = len(keyList)
    Nframes = dataset["Nframes"]
    behaviorCount_eachBehavior = np.zeros((Nframes, Nkeys), dtype=int)
    
    for i, key in enumerate(keyList):
        if key in dataset and "edit_frames" in dataset[key]:
            edit_frames = dataset[key]["edit_frames"]
            for frame in edit_frames:
                if frame < Nframes:
                    behaviorCount_eachBehavior[int(frame), i] += 1
    if normalizeCounts:
        behaviorCount = np.sum(behaviorCount_eachBehavior, axis=1) / Nframes
    else:
        behaviorCount = np.sum(behaviorCount_eachBehavior, axis=1)
    
    return behaviorCount, behaviorCount_eachBehavior

def behaviorFrameCount_all(datasets, keyList, 
                           behaviorLabel='behavCount', normalizeCounts=False):
    """
    Count behaviors at each frame for all datasets.
    Modifies "datasets" list of dictionaries, can use make_2D_histogram()
    
    Parameters:
        datasets (list): List of dictionaries containing the analysis data.
        keyList : a list of strings that contains behavior key names to count
        behaviorLabel : string, name of the new key of counts to put in each dataset item
        normalizeCounts : if True, normalize by Nframes to return a probability, 
                    rather than a count, of behavior events
                    
    Returns: 
        datasets (list): List of dictionaries containing the analysis data.
            This is the same as the input dataset, with an additional key
            "behaviorLabel" of the counts of the given behaviors.
            Note that this does not copy datasets!
        
    """
    for dataset in datasets:
        behaviorCounts, _ = behaviorFrameCount_one(dataset, keyList, normalizeCounts=normalizeCounts)
        dataset[behaviorLabel] = behaviorCounts
    
    return datasets
