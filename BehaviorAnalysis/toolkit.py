# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
"""
Author:   Raghuveer Parthasarathy
Version ='2.0': 
First version created by  : Estelle Trieu, 9/7/2022
Major modifications by Raghuveer Parthasarathy, May-July 2023
Last modified by Rghuveer Parthasarathy, July 21, 2024

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

def load_expt_config(config_path, config_file):
    """ 
    Loads the experimental configuration file
    Asks user for the experiment being examined
    Inputs:
        config_path, config_file: path and file name of the yaml config file
    Outputs:
        expt_config : dictionary of configuration information
    """
    with open(os.path.join(config_path, config_file), 'r') as f:
        all_config = yaml.safe_load(f)
    all_expt_names = list(all_config.keys())
    print('\n\nALl experiments: ')
    for j, key in enumerate(all_expt_names):
        print(f'  {j}: {key}')
    expt_choice = input('Select experiment (name string or number): ')
    # Note that we're not checking if the choice is valid, i.e. if in 
    # all_expt_names (if a string) or if in 0...len(all_expt_names) (if 
    # a string that can be converted to an integer.)
    try:
        # Is the input string just an integer?
        expt_config = all_config[all_expt_names[int(expt_choice)]]
    except:
        expt_config = all_config[all_expt_names[expt_choice]]
    expt_config['imageScaleLocation'] = os.path.join(expt_config['imageScalePathName'], 
                                                     expt_config['imageScaleFilename'])

    if ("arenaCentersFilename" in expt_config.keys()):
        if expt_config['arenaCentersFilename'] != None:
            expt_config['arenaCentersLocation'] = os.path.join(expt_config['arenaCentersPathName'], 
                                                           expt_config['arenaCentersFilename'])
        else:
            expt_config['arenaCentersLocation'] = None
    else:
        expt_config['arenaCentersLocation'] = None
    
    return expt_config
    

def get_CSV_folder_and_filenames(expt_config, startString="results"):
    """
    Get the folder path containing CSV files, either from the
    configuration file, or asking user for the folder path
    Also get list of all CSV files whose names start with 
    startString, probably "results"

    Inputs:
        expt_config : dictionary containing dataPathMain (or None to ask user)
                        as well as subGroup info (optional)
        startString : the string that all CSV files to be considered should
                        start with. Default "results"
    Returns:
        A tuple containing
        - dataPath : the folder path containing CSV files
        - allCSVfileNames : a list of all CSV files with names 
                            starting with startString (probably "results")
        - subGroupName : Path name of the subGroup; None if no subgroups
    
    """
    
    if expt_config['dataPathMain'] == None:
        dataPath = input("Enter the folder for CSV files, or leave empty for cwd: ")
        if dataPath=='':
            dataPath = os.getcwd() # Current working directory
    else:
        # Load path from config file
        if ('subGroups' in expt_config.keys()):
            dataPathMain = expt_config['dataPathMain']
            print('\nSub-Experiments:')
            for j, subGroup in enumerate(expt_config['subGroups']):
                print(f'  {j}: {subGroup}')
            subGroup_choice = input('Select sub-experiment (string or number): ')
            try:
                subGroupName = expt_config['subGroups'][int(subGroup_choice)]
            except:
                subGroupName = expt_config['subGroups'][subGroup_choice]
            dataPath = os.path.join(dataPathMain, subGroupName)
        else:
            subGroupName = None
            dataPath = expt_config['dataPathMain']
        
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


def remove_frames(frames, frames_to_remove):
    """
    Remove from frames values that appear in frames_to_remove
    
    Inputs:
        frames (int) : 1D array of frame numbers
        frames_to_remove : tuple of 1D arrays of frame numbers to remove
        
    Outputs:
        frames_edit : 1D array of frame numbers
    """

    for j in range(len(frames_to_remove)):
        frames = np.setdiff1d(frames, frames_to_remove[j])

    frames_edit = frames
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
        # Also keep Nframes x Nfish=2 array of distance to edge, mm
        edge_frames, datasets[j]["d_to_edge_mm"] \
            = get_edge_frames(datasets[j], params, arena_radius_mm, 
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
    
    Inputs:
        dataset : dataset dictionary. Note "all_data" is 
                  Nframes x data columns x Nfish
        params : parameters
        arena_radius_mm :arena_radius in mm
        xcol, ycol = column indices (0==first) of the x and y head 
                        position columns
        
    Output:
        near_edge_frames : array of frame numbers (not index numbers!)
        d_to_edge_mm : r_edge - r_fish, Nframes x Nfish array, mm
    """
    x = dataset["all_data"][:,xcol,:]
    y = dataset["all_data"][:,ycol,:]
    dx = x - dataset["arena_center"][0]
    dy = y - dataset["arena_center"][1]
    dr = np.sqrt(dx**2 + dy**2)
    # r_edge - r_fish, Nframes x Nfish array, units = mm:
    d_to_edge_mm = arena_radius_mm - dr*dataset["image_scale"]/1000.0
    # True if close to edge
    near_edge = d_to_edge_mm < params["arena_edge_threshold_mm"]
    near_edge = np.any(near_edge, axis=1)

    near_edge_frames = dataset["frameArray"][np.where(near_edge)]
    
    return near_edge_frames, d_to_edge_mm

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
    
    if ("imageScale" in expt_config.keys()):
        if expt_config['imageScale'] != None:
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


def write_pickle_file(list_for_pickle, dataPath, outputFolderName, pickleFileName):
    """
    Write Pickle file containing datasets, etc., in the analysis folder
    
    Parameters
    ----------
    list_for_pickle : list of datasets to save in the Pickle file
    dataPath : CSV data path
    outputFolderName : output path, should be params['output_subFolder']
    pickleFileName : string, filename -- will append .pickle

    Returns
    -------
    None.

    """
    pickle_folder = os.path.join(dataPath, outputFolderName)
    
    # Create output directory, if it doesn't exist
    pickle_folder = os.path.join(dataPath, outputFolderName)
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)

    pickleFileName = pickleFileName + '.pickle'
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
    key_list = ["perp_noneSee", 
                "perp_oneSees", "perp_bothSee", 
                "perp_larger_fish_sees", 
                "perp_smaller_fish_sees", 
                "contact_any", "contact_head_body", 
                "contact_larger_fish_head", "contact_smaller_fish_head", 
                "contact_inferred", "tail_rubbing", 
                "Cbend_Fish0", "Cbend_Fish1", "Jbend_Fish0", "Jbend_Fish1",
                "approaching_Fish0", "approaching_Fish1", 
                "fleeing_Fish0", "fleeing_Fish1", 
                "isMoving_any", "isMoving_all", 
                "edge_frames", "bad_bodyTrack_frames"]
    # Remove any keys that are not in the first dataset, for example
    # two-fish behaviors if that dataset was for single fish data
    key_list_revised = [key for key in key_list if key in datasets[0]]

    # Mark frames for each dataset
    # Create the ExcelWriter object
    excel_file = os.path.join(output_path, params['allDatasets_markFrames_ExcelFile'])
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
                    "total_time_seconds",
                    "speed_mm_s_mean", "speed_whenMoving_mm_s_mean",
                    "fish_length_Delta_mm_mean", 
                    "head_head_distance_mm_mean", 
                    "AngleXCorr_mean"]
    initial_strings = ["Dataset", "Frames per sec", 
                       "Image scale (um/s)",
                       "Total Time (s)", 
                       "Mean speed (mm/s)", "Mean moving speed (mm/s)", 
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
        dataset : dictionary with all dataset info
        key_list : list of dictionary keys corresponding to each behavior to write

    Returns:
        N/A
    """
    with open(f"{dataset['dataset_name']}.txt", "w") as results_file:
        results_file.write(f"{dataset['dataset_name']}\n")
        results_file.write(f"   Image scale (um/px): {dataset['image_scale']:.1f}\n")
        results_file.write(f"   frames per second: {dataset['fps']:.1f}\n")
        results_file.write(f"   Duration: {dataset['total_time_seconds']:.1f} s\n")
        results_file.write(f"   Mean fish length: {dataset['fish_length_mm_mean']:.3f} mm\n")
        if dataset["Nfish"]==2:
            results_file.write(f"   Mean difference in length: {dataset['fish_length_Delta_mm_mean']:.3f} mm\n")
            results_file.write(f"   Mean head-to-head distance: {dataset['head_head_distance_mm_mean']:.3f} mm\n")
            results_file.write(f"   Mean closest distance: {dataset['closest_distance_mm_mean']:.3f} mm\n")
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
        Distance to edge, each fish (mm); ["d_to_edge_mm"]
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
        headers.extend(["head_head_distance_mm", "closest_distance_mm"])
        for j in range(Nfish):
            headers.extend([f"rel_orientation_rad_Fish{j}"])
    for j in range(Nfish):
        headers.extend([f"speed_mm_s_Fish{j}"])
    for j in range(Nfish):
        headers.extend([f"d_to_edge_mm_Fish{j}"])
    headers.extend(["edge flag", "bad tracking"])

    # Create a list of rows
    rows = []
    
    for j in range(Nframes):
        row = ["{:d}".format(frames[j])]
        if Nfish == 2:
            row.extend(["{:.3f}".format(dataset["head_head_distance_mm"][j].item()),
                        "{:.3f}".format(dataset["closest_distance_mm"][j].item())])
            for k in range(Nfish):
                row.extend(["{:.3f}".format(dataset["relative_orientation"][j, k].item())])
        for k in range(Nfish):
            row.extend(["{:.3f}".format(dataset["speed_array_mm_s"][j, k].item())])
        for k in range(Nfish):
            row.extend(["{:.3f}".format(dataset["d_to_edge_mm"][j, k].item())])
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
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
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
    angles = dataset["all_data"][:, CSVcolumns["angle_data_column"], :].copy()

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
    dataset_repaired["all_data"][:, CSVcolumns["angle_data_column"], :] = angles
    
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
        dataset_repaired["all_data"][:, CSVcolumns["angle_data_column"], 0] = angles0_new
        dataset_repaired["all_data"][:, CSVcolumns["angle_data_column"], 1] = angles1_new
    
    return dataset_repaired
        