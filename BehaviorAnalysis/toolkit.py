# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
"""
Author:   Raghuveer Parthasarathy
Version ='2.0': 
First versions created By  : Estelle Trieu, 9/7/2022
Major modifications by Raghuveer Parthasarathy, May-July 2023
Last modified Sept. 15, 2023 -- Raghu Parthasarathy

Description
-----------

Module containing functions to get lists of files to load, 
load data, assess proximity to the edge, assess bad frames, etc.

"""

import numpy as np
import csv
import os
import matplotlib.pyplot as plt


def get_CSV_folder_and_filenames():
    """
    Asks user for the folder path containing CSV files; returns this
    and a list of all CSV files whose names start with "results"

    Returns:
        A tuple containing
        - folder_path : the folder path containing CSV files
        - allCSVfileNames : all CSV Files with names starting with 'results'
    
    """
    
    folder_path = input("Enter the folder path for CSV files, or leave empty for cwd: ")
    
    if folder_path=='':
        folder_path = os.getcwd() # Current working directory
        
    # Validate the folder path
    while not os.path.isdir(folder_path):
        print("Invalid folder path. Please try again.")
        folder_path = input("Enter the folder path: ")

    print("Selected folder path: ", folder_path)
    
    # Make a list of all relevant CSV files in the folder
    allCSVfileNames = []
    for filename in os.listdir(folder_path):
        if (filename.endswith('.csv') and filename.startswith('results')):
            allCSVfileNames.append(filename)

    return folder_path, allCSVfileNames

    
def load_data(CSVfileName, N_columns):
    """
    Loads position data from a CSV file and returns a single array
    containing both fish's information (position, angle, body markers
    etc.
    Load all columns (0 to N_columns-1)
    Also returns frame numbers (first column of CSV), checking that 
    the frame number array is the same for both dataset halves.

    Args:
        CSVfileName (str): CSV file name with tracking data
        N_columns: number of columns (almost certainly 26).

    Returns:
        all_data : a single numpy array with all the data 
                   (all columns of CSV)
                   Rows = frames
                   Col = CSV columns
                   Layers = fish (2)
        frameArray : array of all frame numbers
    """
    data = np.genfromtxt(CSVfileName, delimiter=',')
    Nrows = data.shape[0] # number of rows
    if np.mod(Nrows,2) == 1:
        print('Error! number of rows is odd. load_data in toolkit.py')
        print('    Pausing; press Enter or (recommended) Control-C')
        input('    : ')
    half_size = int(Nrows/2)
    fish1_data = data[:half_size][:, np.r_[0:N_columns]] 
    fish2_data = data[half_size:][:, np.r_[0:N_columns]]
    
    # make a single numpy array with all the data (all columns of CSV)
    all_data = np.zeros((fish1_data.shape[0], fish1_data.shape[1], 2))
    all_data[:,:,0] = fish1_data
    all_data[:,:,1] = fish2_data
    
    # Check that frame numbers are the same for each array, and that
    # there are no gaps in frames, and that the first frame is 1
    frames_1 = all_data[:,1,0]
    frames_2 = all_data[:,1,1]
    if not(np.array_equal(frames_1, frames_2)):
        print('load_data: frame arrays are bad; not equal for both fish!')
        print('    Pausing; press Enter or (recommended) Control-C')
        input('    : ')
    elif np.max(np.diff(frames_1)>1.01):
        print('load_data: there is a gap in frame numbers!')
        print('    Behavior analysis is not written to accomodate this.')
        print('    Pausing; press Enter or (recommended) Control-C')
        input('    : ')
    elif np.min(np.abs(frames_1-1.0)) > 0.01:
        print('load_data: first frame is not numbered 1!')
        print('    Behavior analysis *might* work, but has not been tested for this.')
        print('    Pausing; press Enter to continue, or Control-C')
        input('    : ')
    else:
        # all is fine; force integer
        frameArray = frames_1.astype(int)

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



def make_frames_dictionary(frames, frames_to_remove):
    """
    Make a dictionary of raw (original) frames, frames with "bad" 
    frames removed, and combined (adjacent) frames + durations
    
    Calls remove_frames()
    Calls combine_events()
    Inputs:
        frames (int) : 1D array of frame numbers
        frames_to_remove : tuple of 1D arrays of frame numbers to remove
        
    Outputs:
        frames_dict : dictionary with keys
            raw_frames : original (frames), 1D array
            edit_frames : frames with "bad" frames removed, 1D array
            combine_frames : 2 x N array using combine_events, frame numbers
                and durations
            total_duration : scalar, sum of durations
    """
    # necessary to initialize the dictionary this way?
    keys = {"raw_frames", "edit_frames", "combine_frames", 
            "total_duration", "behavior_name"}
    frames_dict = dict([(key, []) for key in keys])
    frames_dict["raw_frames"] = frames
    frames_dict["edit_frames"] = \
        remove_frames(frames,  frames_to_remove)
    frames_dict["combine_frames"] = \
        combine_events(frames_dict["edit_frames"])
    frames_dict["total_duration"] = np.sum(frames_dict["combine_frames"][1,:])
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


def get_ArenaCenter(dataset_name, arenaCentersFilename, offsetPositionsFilename):
    """ 
    Extract the x,y positions of the Arena centers from the 
    arenaCentersFilename CSV -- previously tabulated.
    Image offsets also previously tabulated, first and second columns of
    offsetPositionsFilename

    Inputs:
        
    Returns:
        tuple of x, y positions
    Returns none if no rows match the input dataset_name, 
        and error if >1 match
        
    Code partially from GPT3-5 (openAI), for center positions
    Then I crudely duplicate it for offset positions
    """
    matching_rows = []

    with open(arenaCentersFilename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row

        for row in reader:
            # remove "_light" and "_dark" to allow 5b datasets
            mod_dataset_name = dataset_name.replace('_light', '')
            mod_dataset_name = mod_dataset_name.replace('_dark', '')
            if mod_dataset_name == row[0].replace('SocPref_', ''):
                matching_rows.append(row)

        if len(matching_rows) == 0:
            arenaCenterUncorrected = None
        elif len(matching_rows) > 1:
            raise ValueError("get_ArenaCenter: Multiple rows contain the input dataset_name string")
        else:
            arenaCenterUncorrected = np.array((matching_rows[0][5], 
                                               matching_rows[0][6])).astype(float)
            
    matching_offset_rows = []
    with open(offsetPositionsFilename, 'r') as file:
        reader = csv.reader(file)

        for row in reader:
            # remove "_light" and "_dark" to allow 5b datasets
            mod_dataset_name = dataset_name.replace('_light', '')
            mod_dataset_name = mod_dataset_name.replace('_dark', '')
            thisRow0 = row[0].replace('results_SocPref_', '')
            thisRow0 = thisRow0.replace('_ALL', '')
            if mod_dataset_name == thisRow0:
                matching_offset_rows.append(row)

        if len(matching_offset_rows) == 0:
            arenaOffset = None
            return None
        elif len(matching_offset_rows) > 1:
            raise ValueError("get_ArenaCenter: Multiple rows contain the input dataset_name string")
        else:
            arenaOffset = np.array((matching_offset_rows[0][1], matching_offset_rows[0][2])).astype(float)
    
    if (arenaCenterUncorrected is not None) and (arenaOffset is not None):
        return arenaCenterUncorrected - arenaOffset
    else:
        return None
        
    
def get_edge_frames(dataset, params, arena_radius_mm, xcol=3, ycol=4):
    """ 
    identify frames in which the head position of one or more fish is close
    to the dish edge (within threshold)
    
    Inputs:
        dataset : dataset dictionary. Note "all_data" is 
                  Nframes x data columns x 2 fish
        params : parameters
        arena_radius_mm :arena_radius in mm
        xcol, ycol = column indices (0==first) of the x and y head 
                        position columns
        
    Output:
        near_edge_frames : array of frame numbers (not index numbers!)
    """
    x = dataset["all_data"][:,xcol,:]
    y = dataset["all_data"][:,ycol,:]
    dx = x - dataset["arena_center"][0]
    dy = y - dataset["arena_center"][1]
    dr = np.sqrt(dx**2 + dy**2)
    # True if close to edge
    near_edge = (arena_radius_mm*1000/dataset["image_scale"] - dr) < \
        params["arena_edge_threshold_mm"]*1000/dataset["image_scale"]
    near_edge = np.any(near_edge, axis=1)

    near_edge_frames = dataset["frameArray"][np.where(near_edge)]
    return near_edge_frames

def get_imageScale(dataset_name, imageScaleFilename):
    """ 
    Extract the image scale (um/px) from 
    imageScaleFilename CSV -- previously tabulated

    Inputs:
        dataset_name : name of dataset
        imageScaleFilename : name of CSV file containing image scale 
                             information
    Returns:
        image scale (um/px)
    Returns none if no rows match the input dataset_name, 
        and error if >1 match
        
    Code partially from GPT3-5 (openAI)
    """
    matching_rows = []

    with open(imageScaleFilename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row

        for row in reader:
            # remove "_light" and "_dark" to allow 5b datasets
            mod_dataset_name = dataset_name.replace('_light', '')
            mod_dataset_name = mod_dataset_name.replace('_dark', '')
            if mod_dataset_name == row[0].replace('SocPref_', ''):
                matching_rows.append(row)

        if len(matching_rows) == 0:
            return None
        elif len(matching_rows) > 1:
            # print(dataset_name, ' in rows: ', matching_rows[:][0])
            raise ValueError("get_imageScale: Multiple rows contain the input dataset_name string")
        else:
            return matching_rows[0][4]


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



def get_interfish_distance(all_data, CSVcolumns):
    """
    Get the inter-fish distance (distance between head positions)
        in each frame 
    Input:
        all_data : all position data, from dataset["all_data"]
        CSVcolumns
    Output
        interfish_distance : Nframes x 1 array
    """
    
    x = all_data[:,CSVcolumns["pos_data_column_x"],:] # x, both fish
    y = all_data[:,CSVcolumns["pos_data_column_y"],:] # y, both fish
    dx = np.diff(x)
    dy = np.diff(y)
    interfish_distance = np.sqrt(dx**2 + dy**2)
    return interfish_distance
        
def get_fish_lengths(all_data, CSVcolumns):
    """
    Get the length of each fish in each frame (sum of all segments)
    Input:
        all_data : all position data, from dataset["all_data"]
        CSVcolumns
    Output
        fish_lengths : Nframes x 2 array
    """
    xstart = int(CSVcolumns["body_column_x_start"])
    xend =int(CSVcolumns["body_column_x_start"])+int(CSVcolumns["body_Ncolumns"])
    ystart = int(CSVcolumns["body_column_y_start"])
    yend = int(CSVcolumns["body_column_y_start"])+int(CSVcolumns["body_Ncolumns"])
    dx = np.diff(all_data[:,xstart:xend,:], axis=1)
    dy = np.diff(all_data[:,ystart:yend,:], axis=1)
    dr = np.sqrt(dx**2 + dy**2)
    fish_lengths = np.sum(dr,axis=1)
    return fish_lengths
            

def get_bad_headTrack_frames(dataset, params, xcol=3, ycol=4, tol=0.001):
    """ 
    identify frames in which the head position of one or more fish is 
    zero, indicating bad tracking
    
    Inputs:
        dataset : dataset dictionary. Note "all_data" is 
                  Nframes x data columns x 2 fish
        params : parameters
        xcol, ycol = column indices (0==first) of the x and y head 
                        position columns
        tol : tolerance for "zero", pixels
        
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
                  Nframes x data columns x 2 fish
        params : parameters
        body_column_{x,y}_start" : column indices (0==first) of the x and y 
                    body position column
        body_Ncolumns : 10 # number of body datapoints
        tol : tolerance for "zero", pixels
        
    Output:
        bad_headTrack_frames : array of frame numbers (not index numbers!)
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
    plt.scatter(dataset["all_data"][:,CSVcolumns["pos_data_column_x"],0].flatten(), 
                dataset["all_data"][:,CSVcolumns["pos_data_column_y"],0].flatten(), color='m', marker='x')
    plt.scatter(dataset["all_data"][:,CSVcolumns["pos_data_column_x"],1].flatten(), 
                dataset["all_data"][:,CSVcolumns["pos_data_column_y"],1].flatten(), color='darkturquoise', marker='x')
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

    


        
def write_behavior_txt_file(dataset, key_list):
    """
    Creates a txt file of the relevant window frames and event durations
    for a set of social behaviors in a given single dataset
    Output text file name: dataset_name + .txt

    Args:
        dataset : dictionary with all dataset info
        key_list : list of dictionary keys corresponding to each behavior to write

    Returns:
        N/A
    """
    with open(f"{dataset['dataset_name']}.txt", "w") as results_file:
        results_file.write(f"{dataset['dataset_name']}\n")
        results_file.write(f"   Duration: {dataset['total_time_seconds']:.1f} s\n")
        results_file.write(f"   Mean length: {dataset['fish_length_mean']:.2f} px\n")
        results_file.write(f"   Mean difference in length: {dataset['fish_length_Delta_mean']:.2f} px\n")
        results_file.write(f"   Mean inter-fish distance: {dataset['inter-fish_distance_mean']:.2f} px\n")
        for k in key_list:
            outString = f'{k} N_events: {dataset[k]["combine_frames"].shape[1]}\n' + \
                    f'{k} Total N_frames: {dataset[k]["total_duration"]}\n' + \
                    f'{k} frames: {dataset[k]["combine_frames"][0,:]}\n' + \
                    f'{k} durations: {dataset[k]["combine_frames"][1,:]}\n'
            results_file.write(outString)

def mark_behavior_frames_Excel(markFrames_workbook, dataset, key_list):
    """
    Create and fill in sheet in Excel marking all frames with behaviors
    found in this dataset

    Args:
        markFrames_workbook : Excel workbook 
        dataset : dictionary with all dataset info
        key_list : list of dictionary keys corresponding to each behavior to write

    Returns:
        N/A
    """
    sheet1 = markFrames_workbook.add_worksheet(dataset["dataset_name"])
    ascii_uppercase = list(map(chr, range(65, 91)))
    
    # Headers 
    sheet1.write('A1', 'Frame') 
    for j, k in enumerate(key_list):
        sheet1.write(f'{ascii_uppercase[j+1]}1', k) 
        
    # All frame numbers
    maxFrame = int(np.max(dataset["frameArray"]))
    for j in range(1,maxFrame+1):
        sheet1.write(f'A{j+1}', str(j))

    # Each behavior
    for j, k in enumerate(key_list):
        for run_idx in  range(dataset[k]["combine_frames"].shape[1]):
            for duration_idx in range(dataset[k]["combine_frames"][1,run_idx]):
                sheet1.write(f'{ascii_uppercase[j+1]}{dataset[k]["combine_frames"][0,run_idx]+duration_idx+1}', 
                         "X".center(17))
                