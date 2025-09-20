# !/usr/bin/env python3  
# toolkit.py
# -*- coding: utf-8 -*- 
"""
Author:   Raghuveer Parthasarathy
Version ='2.0': 
First version created by  : Estelle Trieu, 9/7/2022
Major modifications by Raghuveer Parthasarathy, May-July 2023
Last modified by Rghuveer Parthasarathy, Sept. 14, 2025

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
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
import tkinter as tk
from tkinter import ttk
from scipy import signal
import warnings



def get_Nfish(datasets):
    # Check that the number of fish is the same for all datasets; note this
    Nfish_values = [dataset.get("Nfish") for dataset in datasets]
    if len(set(Nfish_values)) != 1:
        raise ValueError("Not all datasets have the same 'Nfish' value")
    Nfish = Nfish_values[0]
    print(f'Number of fish: {Nfish}')
    return Nfish



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
    frames_dict["edit_frames"] = remove_frames(frames, frames_to_remove)
    frames_dict["combine_frames"] = combine_events(frames_dict["edit_frames"])
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
                           (e.g. contact, tail-rubbing, 90-degree events).

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


def dilate_frames(frames, dilate_frames=np.array([0])):
    """
    "dilate" the array of frame numbers.
    Note: doesn't check that dilated frame numbers are valid.
    
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
            frames_edit = np.concatenate((frames_edit, frames + j))
    frames_edit = np.unique(frames_edit)
   
    return frames_edit



def get_edgeRejection_frames(dataset, params, arena_radius_mm):
    """ 
    Identify frames to reject in which the head position of one or more 
    fish is close to the dish edge (within threshold)
    Note that radial coordinate has already been calculated (get_polar_coords() )
    If there is no edge-rejection threshold, return empty numpy array
    
    Inputs:
        dataset : dataset dictionary. Note "all_data" is 
                  Nframes x data columns x Nfish
        params : parameters, including edge-closeness threshold
        arena_radius_mm :arena_radius in mm
         
    Output:
        edge_rejection_frames : numpy array of frame numbers (not index numbers!)
    """
    params["edge_rejection_threshold_mm"] 
    if params["edge_rejection_threshold_mm"] is not None:
        print('\n\nUsing edge threshold: ' , params["edge_rejection_threshold_mm"])
        r_mm = dataset["radial_position_mm"] # distance from center, mm
        # True if close to edge
        near_edge = (arena_radius_mm - r_mm) < params["edge_rejection_threshold_mm"]
        near_edge = np.any(near_edge, axis=1)
        edge_rejection_frames = dataset["frameArray"][np.where(near_edge)]
    else:
        edge_rejection_frames = np.array([])
    
    return edge_rejection_frames


def get_edgeRejection_frames_dictionary(datasets, params, arena_radius_mm):
    """ 
    Calls get_edgeRejection_frames() to make a dictionary with
    frames for rejecting behavior, in which the head position 
    of one or more fish is close to the dish edge (within threshold)
    Note that radial coordinate has already been calculated (get_polar_coords() )
    
    Inputs:
        datasets : all datasets, dictionary 
        params : analysis parameters
        arena_radius_mm :arena_radius in mm
        
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
        edge_frames = get_edgeRejection_frames(datasets[j], params, arena_radius_mm)
        
        datasets[j]["edge_frames"] = make_frames_dictionary(edge_frames, (), 
                                                            behavior_name='Edge frames',
                                                            Nframes=datasets[j]['Nframes'])
        #print('   Number of edge frames to reject: ', len(datasets[j]["edge_frames"]["raw_frames"]))
    
    return datasets
    
    
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



def get_badTracking_frames_dictionary(all_position_data, datasets, params, CSVcolumns, tol=0.001):
    """ 
    identify frames in which head or body tracking of one or more fish 
    is bad (zero values)
    
    Inputs:
        all_position_data : basic position information for all datasets, list of numpy arrays
        datasets : all datasets, list of dictionaries 
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
        bad_headTrack_frames = get_bad_headTrack_frames(all_position_data[j], 
                                                        datasets[j]["frameArray"], 
                                                        params,
                                                        CSVcolumns["head_column_x"],
                                                        CSVcolumns["head_column_y"],
                                                        tol)
        datasets[j]["bad_headTrack_frames"] = make_frames_dictionary(bad_headTrack_frames, 
                                                                     (), 
                                                                     behavior_name='Bad head track frames',
                                                                     Nframes=datasets[j]['Nframes'])
        print('   Number of bad head tracking frames: ', len(datasets[j]["bad_headTrack_frames"]["raw_frames"]))
        bad_bodyTrack_frames = get_bad_bodyTrack_frames(all_position_data[j], 
                                                        datasets[j]["frameArray"], 
                                                        params, 
                                                        CSVcolumns["body_column_x_start"],
                                                        CSVcolumns["body_column_y_start"], 
                                                        CSVcolumns["body_Ncolumns"], 
                                                        0.001)
        datasets[j]["bad_bodyTrack_frames"] = make_frames_dictionary(bad_bodyTrack_frames, (), 
                                                                     behavior_name='Bad track frames',
                                                                     Nframes=datasets[j]['Nframes'])
        print('   Number of bad body tracking frames: ', len(datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
    
    return datasets

        
    
def get_bad_headTrack_frames(position_data, frameArray, params, 
                             xcol=3, ycol=4, tol=0.001):
    """ 
    identify frames in which the head position of one or more fish is 
    zero, indicating bad tracking
    
    Inputs:
        position_data : position data for this dataset, presumably all_position_data[j]
        frameArray : numpy array of frame numbers, presumably dataset[j]["frameArray"]
        dataset : dataset dictionary. Note "all_data" is 
                  Nframes x data columns x Nfish
        params : parameters
        xcol, ycol = column indices (0==first) of the x and y head 
                        position columns
        tol : tolerance for "zero" (bad tracking), pixels
        
    Output:
        bad_headTrack_frames : array of frame numbers (not index numbers!)
    """
    x = position_data[:,xcol,:]
    y = position_data[:,ycol,:]
    # True if any of x, y is zero; 
    xy_zero = np.logical_or(np.abs(x)<tol, np.abs(y)<tol)
    bad_headTrack = np.any(xy_zero, axis=1)

    bad_headTrack_frames = frameArray[np.where(bad_headTrack)]
    return bad_headTrack_frames

    
def get_bad_bodyTrack_frames(position_data, frameArray, params, body_column_x_start=6, 
                             body_column_y_start=16, body_Ncolumns=10, 
                             tol=0.001):
    """ 
    identify frames in which tracking failed, as indicated by either of:
    (i) any body position of one or more fish is zero, or
    (ii) the distance between positions 1 and 2 (head-body) is more than
         3 times the mean distance between positions j and j+1 
         for j = 2 to 9
    
    Inputs:
        position_data : position data for this dataset, presumably all_position_data[j]
        frameArray : numpy array of frame numbers, presumably dataset[j]["frameArray"]
        params : parameters
        body_column_{x,y}_start" : column indices (0==first) of the x and y 
                    body position column
        body_Ncolumns : 10 # number of body datapoints
        tol : tolerance for "zero", pixels
        
    Output:
        bad_bodyTrack_frames : array of frame numbers (not index numbers!)
    """
    x = position_data[:,body_column_x_start:(body_column_x_start+body_Ncolumns),:]
    y = position_data[:,body_column_y_start:(body_column_y_start+body_Ncolumns),:]
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
    bad_bodyTrack_frames = np.array(frameArray[badidx])
    # print(bad_bodyTrack_frames)
    
    return bad_bodyTrack_frames
    
def wrap_to_pi(x):
    # shift values of numpy array "x" to [-pi, pi]
    # x must be an array, not a single number
    x_wrap = np.remainder(x, 2*np.pi)
    mask = np.abs(x_wrap)>np.pi
    x_wrap[mask] -= 2*np.pi * np.sign(x_wrap[mask])
    return x_wrap


def calc_good_tracking_spans(dataset, verbose = False):
    """
    For a given dataset, identify continuous frames of good 
    (i.e. not bad) tracking. Calculate the number of such spans, 
    and their min, max, and mean duration
    
    Parameters:
    -----------
    dataset : single dataset dictionary. Containing (among other things)
        - 'bad_bodyTrack_frames': dict with 'raw_frames' containing frame numbers with bad tracking
          (Note: frame numbers start from 1, not 0)
    verbose : if True, print various things
        
    Returns:
    --------
    good_tracking_spans : dictionary with keys
        "number" : number of continuous-frame good tracking spans
        "mean_frames" : mean number of frames of good tracking spans
        "min_frames" : minimum number of frames of good tracking spans (int)
        "max_frames" : maximum number of frames of good tracking spans (int)
        
    
    """
    bad_frames = dataset["bad_bodyTrack_frames"]["raw_frames"]
    
    # Create a boolean array indicating good frames (True) and bad frames (False)
    # Accounting for the offset between frame numbers and array indices
    
    good_frames = np.ones(dataset["Nframes"], dtype=bool)
    for frame in bad_frames:
        if 1 <= frame <= dataset["Nframes"]:  # Ensure the frame number is valid
            good_frames[frame-1] = False  # Adjust for the offset
    
    # Find continuous spans of good frames
    spans = []
    span_start = None
    
    # Initialize dictionary
    good_tracking_spans = {
        'number': 0,
        'mean_frames': np.nan,
        'min_frames': np.nan,
        'max_frames': np.nan,
    }
    
    for i in range(dataset["Nframes"]):
        if good_frames[i] and span_start is None:
            span_start = i
        elif not good_frames[i] and span_start is not None:
            spans.append((span_start, i-1))
            span_start = None
    
    # Handle the case where the last span extends to the end
    if span_start is not None:
        spans.append((span_start, dataset["Nframes"]-1))
    
    # If no good spans found, return the initialized dictionary (number = 0)
    if not spans:
        print("No good tracking spans found.")
        return good_tracking_spans
    
    # Find the largest span 
    largest_span = max(spans, key=lambda x: x[1] - x[0] + 1)
    largest_span_start, largest_span_end = largest_span
    
    good_tracking_spans["number"] = len(spans)
    span_length = np.zeros((len(spans),))
    
    if verbose:
        print(f"Number of spans of continuous good tracking: {good_tracking_spans['number']}")
        print(f"Largest span: frames {largest_span_start+1}-{largest_span_end+1}")
    
    # Evaluate each span 
    for j, (span_start, span_end) in enumerate(spans):
        span_length[j] = span_end - span_start + 1
    
    good_tracking_spans['mean_frames'] = np.mean(span_length)
    good_tracking_spans['min_frames'] = int(np.min(span_length))
    good_tracking_spans['max_frames'] = int(np.max(span_length))
    
    return good_tracking_spans



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



def find_smallest_good_idx(N, bad_frames):
    """
    Find the smallest element of range(N) that is not in bad_frames
    """
    bad_frames_set = set(bad_frames)
    for x in range(N):
        if x not in bad_frames_set:
            return x
        
    
def relink_fish_ids(position_data, dataset, CSVcolumns, 
                    verbose = False):
    """
    Relink fish IDs across frames based on minimizing the Euclidean distance 
    between body positions in consecutive good tracking frames.
    Swaps position data to match the re-linking.
    Also swaps heading angle in dataset
    
    Parameters:
    -----------
    position_data : numpy.ndarray
        Position data for a single dataset with shape (Nframes, Ncolumns, Nfish)
        probably input as all_position_data[j]
    dataset : dict
        Dataset information for one dataset, containing bad_bodyTrack_frames
        probably input as datasets[j]
    CSVcolumns : dict
        Dictionary specifying the column indices for body positions
    verbose : if true, show the number of re-assigned frames
        
    Returns:
    --------
    tuple
        - position_data : numpy.ndarray: Updated position data with 
            corrected fish IDs
        - heading_angles : Nframes x Nfish, updated heading angle data 
                            corresponding to corrected fish IDs
    """
    
    # Check first frame number, for indexing
    if min(dataset['frameArray']) != 1:
        raise ValueError('relink_fish_ids requires first frame no. to be 1')
        
    # Create a copy of the data to avoid modifying the original
    new_position_data = position_data.copy()
    heading_angles = dataset["heading_angle"].copy()
    
    # Extract bad tracking frames
    bad_frames = np.array(dataset["bad_bodyTrack_frames"]["raw_frames"])
        
    # Get column indices for body positions
    body_x_start = CSVcolumns["body_column_x_start"]
    body_y_start = CSVcolumns["body_column_y_start"]
    body_n_cols = CSVcolumns["body_Ncolumns"]
    
    body_x_end = body_x_start + body_n_cols
    body_y_end = body_y_start + body_n_cols
    
    # Get the number of frames and fish
    n_frames, _, n_fish = position_data.shape
    
    # Find the first good frame (hopefully 1, but not necessarily)
    first_good_frame = find_smallest_good_idx(dataset["Nframes"], 
                                             dataset["bad_bodyTrack_frames"]["raw_frames"])
    # This variable will also be used as the last known good frame index 
    # Check if first frame has good tracking
    if first_good_frame > 1:
        print('Dataset: ', dataset['dataset_name'], ' starts with bad frames.')
        print('   Start relinking at frame: ', first_good_frame)
    
    # frames whose new assignment differs from frame 0
    reassigned_frames = np.array([])
    
    # Process each frame starting from the one after the first good frame
    last_good_frame = first_good_frame
    for frame in range(first_good_frame + 1, n_frames + 1):
        # Skip bad frames
        if frame in bad_frames:
            continue
        
        # Get body positions for current and last good frame
        # Note -1 for indexing
        body_x_current = new_position_data[frame-1, body_x_start:body_x_end, :]
        body_y_current = new_position_data[frame-1, body_y_start:body_y_end, :]
        
        body_x_previous = new_position_data[last_good_frame-1, body_x_start:body_x_end, :]
        body_y_previous = new_position_data[last_good_frame-1, body_y_start:body_y_end, :]
        
        # Stack x and y coordinates to get positions
        pos_current = np.stack([body_x_current, body_y_current], axis=-1)    # Shape: (Npos, Nfish, 2)
        pos_previous = np.stack([body_x_previous, body_y_previous], axis=-1) # Shape: (Npos, Nfish, 2)
        
        # Calculate distance matrix between all fish in current and previous frame
        # For each body position point, calculate distance between current and previous
        distances_matrix = np.zeros((n_fish, n_fish))
        
        for i in range(n_fish):
            for j in range(n_fish):
                # Calculate Euclidean distance between all body points of fish i in current frame
                # and fish j in previous frame
                point_distances = np.linalg.norm(pos_current[:, i, :] - pos_previous[:, j, :], axis=1)
                # L2 norm
                distances_matrix[i, j] = np.sqrt(np.sum(point_distances**2)) 
                    # sqrt is unnecessary, but I'd like this to be comparable to distance
        
        # Find optimal assignment that minimizes total distance
        row_indices, col_indices = linear_sum_assignment(distances_matrix)
        
        # If the assignment is different from identity, reorder the fish IDs
        if not np.array_equal(col_indices, np.arange(n_fish)):
            # Note this frame
            reassigned_frames = np.append(reassigned_frames, frame)
            
            # Create a new array with reordered fish
            thisFrame_position_data = np.zeros_like(new_position_data[frame-1,:,:])
            thisFrame_heading_angles = np.zeros_like(heading_angles[frame-1,:])
            
            # Reorder fish IDs according to assignment
            for new_id, old_id in enumerate(col_indices):
                thisFrame_position_data[:, new_id] = new_position_data[frame-1, :, old_id]
                thisFrame_heading_angles[new_id] = heading_angles[frame-1, old_id]
            
            # Update the position data for this frame
            new_position_data[frame-1] = thisFrame_position_data
            heading_angles[frame-1] = thisFrame_heading_angles
        
        # Update last good frame
        last_good_frame = frame
        
    reassigned_block_endFrames = reassigned_frames[np.where(
        np.diff(reassigned_frames)>1.0)[0]]
    # print(reassigned_block_endFrames)
    num_reassigned_blocks = len(reassigned_block_endFrames)
    if verbose:
        print(f'Re-linked dataset {dataset["dataset_name"]}; {dataset["Nframes"]} frames.')
        print(f'   re-assigned {num_reassigned_blocks} blocks.')
        print(reassigned_block_endFrames)
    return new_position_data, heading_angles



def relink_fish_ids_all_datasets(all_position_data, datasets, CSVcolumns,
                                 verbose = False):
    """
    Apply the fish ID relinking to all datasets.
    
    Parameters:
    -----------
    all_position_data : list of numpy.ndarray
        List of position data arrays for each experiment
    datasets : list of dict
        List of dataset information for each experiment
    CSVcolumns : dict
        Dictionary specifying the column indices for body positions
    verbose : if true, show the number of re-assigned frames
        
    Returns:
    --------
    corrected_data -- list of numpy.ndarray
        Updated "all_position_data" with corrected fish IDs for all experiments
    datasets -- datasets list of dictionaries with heading_angle updated
    """
    corrected_data = []
    
    for data_idx in range(len(all_position_data)):
        corrected_exp_data, heading_angles = \
            relink_fish_ids(all_position_data[data_idx], 
            datasets[data_idx], CSVcolumns)
        corrected_data.append(corrected_exp_data)
        datasets[data_idx]["heading_angle"] = heading_angles
    
    return corrected_data, datasets


def repair_head_positions(all_position_data, CSVcolumns, tol=0.001):
    """ 
    Ignore "Index 0" head position and replace it with the interpolated 
    position from "Index 1-3" body positions. (Fixing ZebraZoom's unreliable
    head positions.) 
    Could use to replace the heading angle with the interpolated angle
    (see commented code), but there are offset issues, and I'll just use the
    previously-written repair_heading_angles()
    
    Check that body positions aren't zero; ignore otherwise
    Redundant code with get_bad_bodyTrack_frames(), for determining 
       nonzero position data
       
    Inputs:
        all_position_data : basic position information for all datasets, 
                            list of numpy arrays, each Nframes x data columns x Nfish
        CSVcolumns: information on what the columns of all_position_data are
        tol : tolerance for "zero" (bad tracking), pixels

    Output:
        all_position_data_repaired: New all_position_data with 
                            repaired head positions in both "head_column_{x,y}"
                            and the first body column for x, y (same)
                                            
    """

    Npositions = CSVcolumns["body_Ncolumns"]
    
    # Create a new list for the repaired datasets
    # Avoiding confusing copy-in-place. Necessary?
    all_position_data_repaired = []

    for j in range(len(all_position_data)):
        position_data = all_position_data[j]
    
        # x and y are shape Nframes x Npositions x Nfish
        x = position_data[:,CSVcolumns["body_column_x_start"] : 
                                (CSVcolumns["body_column_x_start"]+Npositions),:]
        y = position_data[:,CSVcolumns["body_column_y_start"] :
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
        position_data_repaired = position_data.copy()
        position_data_repaired[:,CSVcolumns["body_column_x_start"],:] = x[:, 0, :]
        position_data_repaired[:,CSVcolumns["body_column_y_start"],:] = y[:, 0, :]  
        position_data_repaired[:,CSVcolumns["head_column_x"],:] = x[:, 0, :]
        position_data_repaired[:,CSVcolumns["head_column_y"],:] = y[:, 0, :]    
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
        all_position_data_repaired.append(position_data_repaired)
    
    return all_position_data_repaired


def repair_heading_angles(all_position_data, datasets, CSVcolumns):
    """ 
    Fix the heading angles -- rather than the strangely quantized angles from
    ZebraZoom, calculate the angle from arctan(y[1]-y[2], x[1]-x[2]) 
    See notes Sept. 2024
    
    Inputs
        all_position_data : basic position information for all datasets, list of numpy arrays
        datasets : list of all dataset dictionaries. 
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
        datasets[j]['heading_angle'] = np.arctan2(all_position_data[j][:,y2,:]-all_position_data[j][:,y3,:], 
                                                  all_position_data[j][:,x2,:]-all_position_data[j][:,x3,:])
        # put in range [0, 2*pi]; not actually necessary
        datasets[j]['heading_angle'][datasets[j]['heading_angle'] < 0.0] += 2*np.pi
    return datasets


    

def repair_double_length_fish(all_position_data, datasets, CSVcolumns, 
                              lengthFactor = [1.5, 2.5], tol=0.001):
    """ 
    Fix tracking data in which there is only one identified fish, with the 
    10 body positions having an overall length 
    roughly twice the actual single fish length (median), therefore
    likely spanning two actual fish.
    Replace one fish with the first 5 positions, interpolated to 10 pts
    and the other with the second 5 positions, interpolated
    Also recalculate the heading angle; call repair_heading_angle() for
    consistency (uses 2nd and 3rd positions).
    
    Inputs:
        all_position_data : basic position information for all datasets, 
                            list of numpy arrays, each Nframes x data columns x Nfish
                            **Note** sending all_position_data as input overwrites
                            (in place); input a copy if you don't want to do this
        datasets : list of all dataset dictionaries. 
                   Note that datasets[j]["fish_length_array_mm"] contains fish lengths (mm)
                   **Note** sending all_position_data as input overwrites
                   (in place); input a copy if you don't want to do this
        CSVcolumns: information on what the columns of all_position_data are
        lengthFactor : a list with two values; 
                       split fish into two if length is between these
                       factors of median fish length
        tol : tolerance for "zero" (bad tracking), pixels
        
    Output:
        all_position_data : positions, with repaired two-fish positions
        datasets : overwrites ["heading_angle"] with repaired heading angles,
                    and revises fish lengths.
    """
    
    print('Repairing "double-length" fish.')
    for j in range(len(datasets)):
        # median fish length (px) for each fish; take average across fish
        mean_fish_length_mm = np.mean(np.median(datasets[j]["fish_length_array_mm"], axis=0))
        
        # .copy() to avoid repairing in place
        Npositions = CSVcolumns["body_Ncolumns"]

        # body position data
        x = all_position_data[j][:,CSVcolumns["body_column_x_start"] : 
                                (CSVcolumns["body_column_x_start"]+Npositions),:].copy()
        y = all_position_data[j][:,CSVcolumns["body_column_y_start"] :
                                (CSVcolumns["body_column_y_start"]+Npositions),:].copy()
        
        # True if all x, y are nonzero; shape Nframes x Nfish
        good_bodyTrack = np.logical_and(np.all(np.abs(x)>tol, axis=1), 
                                        np.all(np.abs(y)>tol, axis=1))
        # Indices of frames in which only one fish was tracked 
        rows_with_one_tracked = np.where(np.sum(good_bodyTrack, axis=1) == 1)[0]
        # Column indices (i.e. fish) where True values exist
        oneFish_indices = np.argmax(good_bodyTrack[rows_with_one_tracked,:], 
                                    axis=1)
        # print('       One fish indices\n', oneFish_indices)
        # Calculate length ratios
        lengthRatios = datasets[j]["fish_length_array_mm"][rows_with_one_tracked, 
                                   oneFish_indices] / mean_fish_length_mm
        # Find frame indices where length ratios meet the condition
        doubleLength_indices = rows_with_one_tracked[np.logical_and(lengthFactor[0] < lengthRatios, 
                                 lengthRatios < lengthFactor[1])]
    
        # Column indices (i.e. fish) where True values exist, only for these
        # frames. Using same variable name
        oneFish_indices = np.argmax(good_bodyTrack[doubleLength_indices,:], 
                                    axis=1)
    
        print(f'Dataset {datasets[j]["dataset_name"]}: median fish length (mm): {mean_fish_length_mm:.2f}')
        print(f'   {len(rows_with_one_tracked)} frames with one fish.')
        print(f'   {len(doubleLength_indices)} frames with one double-length fish.')
        # Repair
        # Note that it doesn't matter which ID is which, since we'll re-link later
        position_data_repaired = all_position_data[j].copy()
        dataset_repaired = datasets[j].copy()
        midPosition = int(np.floor(Npositions/2.0))  # 5 for usual 10 body positions
        interpIndices = np.linspace(0, Npositions-1, num=midPosition).astype(int) 
        for k, frameIdx in enumerate(doubleLength_indices):
            # print('     Double length Frame Idx: ', frameIdx)
            # one fish from the first 5 positions.
            x_first = x[frameIdx,0:midPosition,oneFish_indices[k]]
            y_first = y[frameIdx,0:midPosition,oneFish_indices[k]]
            # print('Frame Index: ', frameIdx)
            #print('     which fish: ', oneFish_indices[k])
            # print(np.arange(0,midPosition))
            x0_new = np.interp(np.arange(0,Npositions), interpIndices, x_first)
            y0_new = np.interp(np.arange(0,Npositions), interpIndices, y_first)
            
            # the other fish from the last 5 positions.
            x_last = x[frameIdx,midPosition:, oneFish_indices[k]]
            y_last = y[frameIdx,midPosition:, oneFish_indices[k]]
            x1_new = np.interp(np.arange(0,Npositions), interpIndices, x_last)
            y1_new = np.interp(np.arange(0,Npositions), interpIndices, y_last)
    
            position_data_repaired[frameIdx,CSVcolumns["body_column_x_start"] : 
                                (CSVcolumns["body_column_x_start"]+Npositions),0] = x0_new
            position_data_repaired[frameIdx,CSVcolumns["body_column_y_start"] :
                                (CSVcolumns["body_column_y_start"]+Npositions),0] = y0_new
            position_data_repaired[frameIdx,CSVcolumns["body_column_x_start"] : 
                                (CSVcolumns["body_column_x_start"]+Npositions),1] = x1_new
            position_data_repaired[frameIdx,CSVcolumns["body_column_y_start"] :
                                (CSVcolumns["body_column_y_start"]+Npositions),1] = y1_new
            
        all_position_data[j] = position_data_repaired
        datasets[j] = dataset_repaired
    
    datasets = repair_heading_angles(all_position_data, datasets, CSVcolumns)
    
    return all_position_data, datasets
 

def combine_all_values_constrained(datasets, keyName='speed_array_mm_s', 
                                   keyIdx = None,
                                   constraintKey=None, constraintRange=None,
                                   constraintIdx = 0, dilate_minus1=True):
    """
    Loop through each dataset, get values of some numerical property
    in datasets[j][keyName], and collect all these in a list of 
    numpy arrays, one array per dataset. 
	Ignore, in each dataset, "bad tracking" frames. 
       If "dilate_minus1" is True, dilate the bad frames -1; 
       do this for speed values, since bad tracking affects adjacent frames!
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
    dilate_minus1 : If True, dilate the bad frames -1; see above.
    
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
            idx_offset = min(frames) # should be 1, but ensure
            badTrackFrames = datasets[j]["bad_bodyTrack_frames"]["raw_frames"]
            if dilate_minus1:
                dilate_badTrackFrames = np.concatenate((badTrackFrames, badTrackFrames - 1))
                bad_frames = np.unique(dilate_badTrackFrames)
            else:
                bad_frames = badTrackFrames
                
            good_frames_mask = np.isin(frames-idx_offset, 
                                       bad_frames-idx_offset, invert=True)
            values = get_values_subset(datasets[j][keyName], keyIdx)
            values_this_set = values[good_frames_mask, ...]
            values_all_constrained.append(values_this_set)
        
        return values_all_constrained
    
    # print(f'    ... with constraint on {constraintKey}')
    for j in range(Ndatasets):
        frames = datasets[j]["frameArray"]
        idx_offset = min(frames) # should be 1, but ensure
        badTrackFrames = datasets[j]["bad_bodyTrack_frames"]["raw_frames"]
        if dilate_minus1:
            dilate_badTrackFrames = np.concatenate((badTrackFrames, badTrackFrames - 1))
            bad_frames = np.unique(dilate_badTrackFrames)
        else:
            bad_frames = badTrackFrames
        
        # Filter values based on constraint
        good_frames_mask = np.isin(frames-idx_offset, 
                                   bad_frames-idx_offset, invert=True)
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
                           xlim = None, ylim = None, polarPlot = False, 
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
       xlim : (optional) tuple of min, max x-axis limits
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
            bin_range[0] = np.nanmin(x_all)
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
    if xlim is not None:
        plt.xlim(xlim)
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
        block_autocorr = signal.correlate(block_centered, block_centered, 
                                          mode='full')
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
        block_crosscorr = signal.correlate(block1_centered, block2_centered, 
                                           mode='full', method='direct')
        block_crosscorr = block_crosscorr[len(block_crosscorr)//2-n_lags//2:len(block_crosscorr)//2+n_lags//2+1]
        block_crosscorr /= (np.std(block1) * np.std(block2) * len(block1))
        block_crosscorrs.append(block_crosscorr)
    
    return np.mean(block_crosscorrs, axis=0)

def calculate_value_corr_oneSet(dataset, keyName='speed_array_mm_s', 
                                corr_type='auto', dilate_minus1=True, 
                                t_max=10, t_window=None):
    """
    For a *single* dataset, calculate the auto or cross-correlation of the numerical
    property in the given key (e.g. speed).
    Cross-correlation is valid only for Nfish==2 (verified)
    Ignore "bad tracking" frames. If a frame is in the bad tracking list,
    replace the value with a Gaussian random number with the same mean, 
    std. dev. as the frames not in the bad tracking list. This avoids
    having to figure out how to deal with non-uniform time lags.
    NOTE: replaces values for *both* fish if a frame is a bad-tracking frame, 
       even if one of the fish is properly tracked. (Simpler to implement, and
       also both fish's values may be unreliable)
    If "dilate_minus1" is True, dilate the bad frames -1.
    Output is a numpy array with dim 1 corresponding to each fish, for auto-
    correlation, and a 1D numpy array for cross-correlation.
    
    Parameters
    ----------
    dataset : single analysis dataset
    keyName : the key to combine (e.g. "speed_array_mm_s")
    corr_type : 'auto' for autocorrelation, 'cross' for cross-correlation (only for Nfish==2)
    dilate_minus1 : If True, dilate the bad frames -1
    t_max : max time to consider for autocorrelation, seconds.
    t_window : size of sliding window in seconds. 
        If None, don't use a sliding window.
        Using a sliding window is strongly recommended!
    
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
    if dilate_minus1:
        dilate_badTrackFrames = dilate_frames(badTrackFrames, 
                                              dilate_frames=np.array([-1]))
        bad_frames_set = set(dilate_badTrackFrames)
    else:
        bad_frames_set = set(badTrackFrames)
     
    if corr_type == 'auto':
        t_lag = np.arange(0, t_max + 1.0/fps, 1.0/fps)
        n_lags = len(t_lag)
        corr = np.zeros((n_lags, Nfish))
    elif corr_type == 'cross':
        if Nfish != 2:
            raise ValueError("Cross-correlation is only supported for Nfish==2")
        t_lag = np.arange(-t_max, t_max + 1.0/fps, 1.0/fps)
        n_lags = len(t_lag)
        corr = np.zeros(n_lags)
    else:
        raise ValueError("corr_type must be 'auto' or 'cross'")
    
    # Lowest frame number (should be 1)   
    idx_offset = min(dataset["frameArray"])
    # Replace values from bad tracking frames with mean, std.
    fish_value = value_array.copy()
    for fish in range(Nfish):
        good_values = [val for i, val in enumerate(fish_value[:,fish]) \
                       if (i + idx_offset) not in bad_frames_set]
        mean_value = np.mean(good_values)
        std_value = np.std(good_values)
        for frame in bad_frames_set:
            if frame < Nframes:
                fish_value[frame - idx_offset, fish] = \
                    np.random.normal(mean_value, std_value)

    if corr_type == 'auto':
        for fish in range(Nfish):
            if t_window is None:
                fish_corr = calculate_autocorr(fish_value[:, fish], n_lags)
            else:
                window_size = int(t_window * fps)
                fish_corr = calculate_block_autocorr(fish_value[:, fish],  
                                                     n_lags, window_size)
            corr[:, fish] = fish_corr
            
    if corr_type == 'cross':
        if t_window is None:
            corr = calculate_crosscorr(fish_value[:, 0], fish_value[:, 1], 
                                       n_lags)
        else:
            window_size = int(t_window * fps)
            corr = calculate_block_crosscorr(fish_value[:, 0], 
                                             fish_value[:, 1], n_lags, 
                                             window_size)
    
    return corr, t_lag


def calculate_value_corr_all(datasets, keyName = 'speed_array_mm_s',
                             corr_type='auto', dilate_minus1 = True, 
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
	dilate_minus1 :  If "dilate_minus1" is True, dilate the bad frames -1; see above. 
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
    
    get_fps(datasets, fpstol = 1e-6) # will give error if not valid

    for j in range(Ndatasets):
        print(f'  {j},', end='')
        (autocorr_one, t_lag) = calculate_value_corr_oneSet(datasets[j], 
                                            keyName = keyName, 
                                            corr_type=corr_type,
                                            dilate_minus1 = dilate_minus1, 
                                            t_max = t_max, t_window = t_window)
        autocorr_all.append(autocorr_one)
    print('\n')
    return autocorr_all, t_lag



def calculate_value_corr_oneSet_binned(dataset, keyName='speed_array_mm_s',
                                       binKeyName = 'head_head_distance_mm',
                                       bin_value_min=0, bin_value_max=50.0, 
                                       bin_width=5.0, t_max=2.0, 
                                       t_window=10.0, dilate_minus1=True):
    """
    Calculate the cross-correlation (between fish) of a property such
    as speed (default) binned by another property (default head-to-head distance)
    for a single dataset.
    
    Parameters
    ----------
    dataset : single analysis dataset (must include "fps")
    keyName : the key to calculate cross-correlation for (e.g. "speed_array_mm_s")
    binKeyName: the key to use for binning (e.g. "head_head_distance_mm");
        Recommend either "head_head_distance_mm" or "closest_distance_mm"
    bin_value_min : minimum value for binning (e.g. min distance, mm)
    bin_value_max : maximum value for binning (e.g. max distance, mm)
    bin_width : width of bins (e.g. for distance, mm)
    t_max : max time to consider for cross-correlation, seconds
    t_window : size of sliding window in seconds
    dilate_minus1 : If True, dilate the bad frames -1
    
    Returns
    -------
    binned_crosscorr : 2D numpy array of shape (n_bins, n_time_lags)
                      Each row is the average cross-correlation for that bin
    bin_centers : array of bin centers (e.g. distance values)
    t_lag : time lag array, seconds
    bin_counts : number of windows contributing to each distance bin
    """
    
    # Get data
    value_array = dataset[keyName]
    bin_value_array = dataset[binKeyName]
    Nframes, Nfish = value_array.shape
    fps = dataset["fps"]
    
    if Nfish != 2:
        raise ValueError("Cross-correlation binning is only supported for Nfish==2")
    
    # Handle bad tracking frames
    badTrackFrames = dataset["bad_bodyTrack_frames"]["raw_frames"]
    if dilate_minus1:
        dilate_badTrackFrames = dilate_frames(badTrackFrames, 
                                              dilate_frames=np.array([-1]))
        bad_frames_set = set(dilate_badTrackFrames)
    else:
        bad_frames_set = set(badTrackFrames)
    
    # Set up time parameters
    window_size = int(t_window * fps)
    t_lag = np.arange(-t_max, t_max + 1.0/fps, 1.0/fps)
    n_lags = len(t_lag)
    
    # Set up distance bins
    bins = np.arange(bin_value_min, bin_value_max + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    n_bins = len(bin_centers)
    
    # Initialize output arrays
    binned_crosscorr = np.zeros((n_bins, n_lags))
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_sums = np.zeros((n_bins, n_lags))
    
    # Process data in non-overlapping windows
    num_windows = Nframes // window_size
    
    for window_idx in range(num_windows):
        start = window_idx * window_size
        end = (window_idx + 1) * window_size
        
        # Check if this window has too many bad frames (skip if >50% are bad)
        window_bad_frames = sum(1 for frame in range(start, end) if frame in bad_frames_set)
        if window_bad_frames / window_size > 0.5:
            print('\ncalculate_value_corr_oneSet_binned:')
            print('  Too many bad frames.')
            continue
            
        # Get window data
        window_data1 = value_array[start:end, 0].copy()
        window_data2 = value_array[start:end, 1].copy()
        window_binValue = bin_value_array[start:end].copy()
        
        # Replace bad frames with random values based on good frame statistics
        good_indices1 = [i for i in range(len(window_data1)) if (start + i) not in bad_frames_set]
        good_indices2 = [i for i in range(len(window_data2)) if (start + i) not in bad_frames_set]
        good_indices_dist = [i for i in range(len(window_binValue)) if (start + i) not in bad_frames_set]
        
        if len(good_indices1) < window_size // 4:  # Skip if too few good frames
            print('calculate_value_corr_oneSet_binned:')
            print('  Too few good frames.')
            continue
            
        mean_val1 = np.mean(window_data1[good_indices1])
        std_val1 = np.std(window_data1[good_indices1])
        mean_val2 = np.mean(window_data2[good_indices2])
        std_val2 = np.std(window_data2[good_indices2])
        mean_dist = np.mean(window_binValue[good_indices_dist])
        
        # Replace bad frames
        for i in range(len(window_data1)):
            if (start + i) in bad_frames_set:
                window_data1[i] = np.random.normal(mean_val1, std_val1)
                window_data2[i] = np.random.normal(mean_val2, std_val2)
                window_binValue[i] = mean_dist  # Use mean distance for bad frames
        
        # Calculate cross-correlation for this window
        window_crosscorr = calculate_crosscorr(window_data1, window_data2, n_lags)
        
        # Calculate mean distance for this window
        mean_window_binValue = np.mean(window_binValue)
        
        # Find which distance bin this window belongs to
        bin_idx = np.digitize(mean_window_binValue, bins) - 1
        
        # Make sure bin index is valid
        if 0 <= bin_idx < n_bins:
            bin_sums[bin_idx] += window_crosscorr
            bin_counts[bin_idx] += 1
                
    # Calculate averages for each bin
    for bin_idx in range(n_bins):
        if bin_counts[bin_idx] > 0:
            binned_crosscorr[bin_idx] = bin_sums[bin_idx] / bin_counts[bin_idx]
        else:
            binned_crosscorr[bin_idx] = np.nan  # No data for this bin
    
    return binned_crosscorr, bin_centers, t_lag, bin_counts


def calculate_value_corr_all_binned(datasets, keyName='speed_array_mm_s',
                                    binKeyName = 'head_head_distance_mm',
                                    bin_value_min=0, bin_value_max=50.0, 
                                    bin_width=5.0, t_max=2.0, 
                                    t_window=10.0, dilate_minus1=True, fpstol=1e-6):
    """
    Calculate distance-binned cross-correlations for all datasets.
    
    Parameters
    ----------
    datasets : list of dictionaries containing all analysis
    keyName : the key to calculate cross-correlation for
    binKeyName: the key to use for binning (e.g. "head_head_distance_mm");
        Recommend either "head_head_distance_mm" or "closest_distance_mm"
    bin_value_min : minimum value for binning (e.g. min distance, mm)
    bin_value_max : maximum value for binning (e.g. max distance, mm)
    bin_width : width of bins (e.g. for distance, mm)
    t_max : max time to consider for cross-correlation, seconds
    t_window : size of sliding window in seconds
    dilate_minus1 : If True, dilate the bad frames -1
    fpstol : relative tolerance for checking that all fps are the same
    
    Returns
    -------
    binned_crosscorr_all : list of 2D numpy arrays, one per dataset
    bin_centers : array of bin centers (e.g. distance values)
    t_lag : time lag array, seconds
    bin_counts_all : list of bin count arrays, one per dataset
    """
    
    Ndatasets = len(datasets)
    print(f'\nCalculating binned cross-correlations of {keyName} for {Ndatasets} datasets')
    print(f'Bin Value range: {bin_value_min}-{bin_value_max} mm, bin width: {bin_width} mm')
    
    # Check that all datasets have the same fps
    get_fps(datasets, fpstol=fpstol)  # will give error if not valid
    
    binned_crosscorr_all = []
    bin_counts_all = []
    
    for j in range(Ndatasets):
        print(f'Processing dataset {j+1}/{Ndatasets}...', end=' ')
        
        binned_crosscorr, bin_centers, t_lag, bin_counts = \
            calculate_value_corr_oneSet_binned(datasets[j], 
                                               keyName=keyName,
                                               binKeyName = binKeyName,
                                               bin_value_min=bin_value_min,
                                               bin_value_max=bin_value_max,
                                               bin_width=bin_width,
                                               t_max=t_max,
                                               t_window=t_window,
                                               dilate_minus1=dilate_minus1)
        
        binned_crosscorr_all.append(binned_crosscorr)
        bin_counts_all.append(bin_counts)
        
        # Report how many bins had data
        valid_bins = np.sum(bin_counts > 0)
        print(f'{valid_bins}/{len(bin_centers)} bins with data')
    
    return binned_crosscorr_all, bin_centers, t_lag, bin_counts_all


def plot_waterfall_binned_crosscorr(binned_crosscorr_all, bin_centers, t_lag,
                                     bin_counts_all=None, xlabelStr='Time lag (s)', 
                                     titleStr='Distance-Binned Cross-correlation',
                                     offset_scale=0.5,  
                                     colormap='RdBu_r', unit_string = 'mm', 
                                     outputFileName=None,
                                     vmin=None, vmax=None, 
                                     plot_heatmap = False, 
                                     heatmap_ylabelStr='Distance (mm)'):
    """
    Create a waterfall plot of binned cross-correlations averaged 
    across all datasets.
    
    Parameters
    ----------
    binned_crosscorr_all : list of 2D numpy arrays from calculate_value_corr_all_binned
    bin_centers : array of bin centers (e.g. distance values)
    t_lag : time lag array
    bin_counts_all : list of bin count arrays (optional, for weighting)
    xlabelStr : x-axis label
    titleStr : plot title
    outputFileName : if not None, save figure with this filename
    colormap : colormap for the waterfall plot
    unit_string : string to append to labels on waterfall plot, indicating units
        for the binning values. Use '' for none
    offset_scale : vertical offset between traces as fraction of 
        max cross-correlation range
    vmin, vmax : color scale limits (if None, use data range)
    plot_heatmap : if True, also make a heatmap plot.
    heatmap_ylabelStr : y-axis label for heatmap
    """
    
    # Calculate average across all datasets
    n_datasets = len(binned_crosscorr_all)
    n_bins, n_time_lags = binned_crosscorr_all[0].shape
    
    # Initialize arrays for weighted average
    total_crosscorr = np.zeros((n_bins, n_time_lags))
    total_weights = np.zeros(n_bins)
    
    # Calculate weighted average if bin counts provided, otherwise simple average
    for i, binned_crosscorr in enumerate(binned_crosscorr_all):
        if bin_counts_all is not None and bin_counts_all[i] is not None:
            weights = bin_counts_all[i]
        else:
            weights = np.ones(n_bins)
            
        for bin_idx in range(n_bins):
            if not np.isnan(binned_crosscorr[bin_idx, 0]) and weights[bin_idx] > 0:
                total_crosscorr[bin_idx] += binned_crosscorr[bin_idx] * weights[bin_idx]
                total_weights[bin_idx] += weights[bin_idx]
    
    # Calculate final averages
    avg_crosscorr = np.zeros((n_bins, n_time_lags))
    for bin_idx in range(n_bins):
        if total_weights[bin_idx] > 0:
            avg_crosscorr[bin_idx] = total_crosscorr[bin_idx] / total_weights[bin_idx]
        else:
            avg_crosscorr[bin_idx] = np.nan
    
    # Debug: Print information about the data
    print(f"Bin centers: {bin_centers[:5]}...{bin_centers[-5:]}")
    print(f"Array shape: {avg_crosscorr.shape}")
    print(f"First few rows have data: {[not np.all(np.isnan(avg_crosscorr[i])) for i in range(min(5, avg_crosscorr.shape[0]))]}")
    print(f"Last few rows have data: {[not np.all(np.isnan(avg_crosscorr[i])) for i in range(max(0, avg_crosscorr.shape[0]-5), avg_crosscorr.shape[0])]}")

    # Waterfall plot with constant offset and distance labels
    fig, ax = plt.subplots(figsize=(7, 12))
    
    valid_bins = ~np.all(np.isnan(avg_crosscorr), axis=1)
    valid_bin_indices = np.where(valid_bins)[0]
    n_valid_bins = len(valid_bin_indices)
    
    
    if n_valid_bins > 0:
        # Calculate constant offset between traces based on correlation data range
        data_range = np.nanmax(avg_crosscorr) - np.nanmin(avg_crosscorr)
        offset = offset_scale * data_range
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_valid_bins))
        
        traces_plotted = 0
        for i, bin_idx in enumerate(valid_bin_indices):
            # Use constant offset
            y_values = avg_crosscorr[bin_idx] + i * offset
            ax.plot(t_lag, y_values, color=colors[i], linewidth=1.5,
                    label=f'{bin_centers[bin_idx]:.1f} ' + unit_string)
            
            # Add text label showing distance bin center
            # Place text at the right edge of the plot
            text_x = t_lag[-1] * 0.95  # 95% along the x-axis
            text_y = np.nanmean(y_values)  # Use mean y-value of the trace
            ax.text(text_x, text_y, f'{bin_centers[bin_idx]:.1f}' + unit_string, 
                    fontsize=10, verticalalignment='center', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            traces_plotted += 1
        
        ax.set_xlabel(xlabelStr, fontsize=14)
        ax.set_ylabel('Cross-correlation + offset', fontsize=14)
        ax.set_title(f'{titleStr}', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        # Add legend for a subset of traces to avoid clutter
        if traces_plotted <= 10:
            ax.legend(fontsize=10, loc='upper left')  # Move legend to avoid text labels
        else:
            # Show only every nth trace in legend
            step = max(1, traces_plotted // 8)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::step], labels[::step], fontsize=10, loc='upper left')
    
    plt.tight_layout()
    
    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight')
    
    plt.show()

    # Print summary statistics
    valid_bins = ~np.all(np.isnan(avg_crosscorr), axis=1)  # Bins that have at least some valid data
    print('\nSummary:')
    print(f'Number of datasets: {n_datasets}')
    print(f'Bins with data: {np.sum(valid_bins)}/{len(bin_centers)}')
    print(f'Traces plotted in waterfall: {traces_plotted}')


    if plot_heatmap:
        # Left panel: 2D color plot
        fig, ax2 = plt.subplots(figsize=(8, 8))
        
        if vmin is None or vmax is None:
            valid_data = avg_crosscorr[~np.isnan(avg_crosscorr)]
            if len(valid_data) > 0:
                data_range = np.percentile(valid_data, [5, 95])
                vmin = vmin or data_range[0]
                vmax = vmax or data_range[1]
            else:
                vmin, vmax = -1, 1
    
        if len(valid_data) > 0:
            print(f'Cross-correlation range: {np.min(valid_data):.3f} to {np.max(valid_data):.3f}')
        
        # Create a copy of the data where NaN values are set to a neutral value for display
        display_data = avg_crosscorr.copy()
        display_data[np.isnan(display_data)] = 0  # Set NaN to 0 for visualization
        
        # Calculate bin edges for proper extent
        bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 5
        bin_edges_min = bin_centers[0] - bin_width/2
        bin_edges_max = bin_centers[-1] + bin_width/2
        
        print(f"Extent: [{t_lag[0]}, {t_lag[-1]}, {bin_edges_min}, {bin_edges_max}]")
        
        im = ax2.imshow(display_data, aspect='auto', origin='lower', 
                        extent=[t_lag[0], t_lag[-1], bin_edges_min, bin_edges_max],
                        cmap=colormap, vmin=vmin, vmax=vmax)
        
        ax2.set_xlabel(xlabelStr, fontsize=14)
        ax2.set_ylabel(heatmap_ylabelStr, fontsize=14)
        ax2.set_title(f'{titleStr}', fontsize=16)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Cross-correlation', fontsize=12)
    
        plt.tight_layout()
        
        if outputFileName is not None:
            plt.savefig('heatmap_' + outputFileName, bbox_inches='tight')
        
        plt.show()
    
    


def get_fps(datasets, fpstol = 1e-6):
    """
    Get the fps (frames per second) from all the datasets. These should
    all be the same; verify this.
    
    Parameters:
    datasets : list of dictionaries containing all analysis. 
    fpstol = relative tolerance for checking that all fps are the same 
    
    Returns:
    average of all the (identical) fps values.    
    """
    if len(datasets) == 0:
        return None
    Ndatasets = len(datasets)
    # Check that all fps are the same, so that all time lag arrays will be the same
    fps_all = np.zeros((Ndatasets,))
    for j in range(Ndatasets):
        fps_all = datasets[j]["fps"]
    good_fps = np.abs(np.std(fps_all)/np.mean(fps_all)) < fpstol
    
    if not good_fps:
        raise ValueError("fps values are not the same across datasets!")
    
    return np.mean(fps_all)


def plot_function_allSets(y_list, x_array = None, 
                           xlabelStr='x', ylabelStr='y', titleStr='Value',
                           average_in_dataset = False,
                           xlim = None, ylim = None,
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
       xlim : (optional) tuple of min, max x-axis limits
       ylim : (optional) tuple of min, max y-axis limits
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
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()
    if outputFileName != None:
        plt.savefig(outputFileName, bbox_inches='tight')

def make_2D_histogram(datasets, keyNames = ('speed_array_mm_s', 'head_head_distance_mm'), 
                      keyIdx = (None, None), 
                      constraintKey=None, constraintRange=None, constraintIdx = 0,
                      dilate_minus1=True, bin_ranges=None, Nbins=(20,20),
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
    dilate_minus1 (bool): If True, dilate the bad frames -1; see above.
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
                                             dilate_minus1=dilate_minus1)
    values2 = combine_all_values_constrained(datasets, keyNames[1], keyIdx=keyIdx[1],
                                             constraintKey=constraintKey, 
                                             constraintRange=constraintRange, 
                                             constraintIdx=constraintIdx,
                                             dilate_minus1 = dilate_minus1)
    
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


def get_behavior_key_list(datasets):
    """
    Get a list of all behaviors
    Only looks at datasets[0]; gets the names of all sub-dictionaries
        that contain the key "combine_frames" , since all behaviors
        should have this.
    
    Inputs
    ----------
    datasets : dictionary; All datasets to analyze

    Returns
    -------
                             
    behavior_key_list : list of all behaviors considered 

    To use: behavior_key_list = get_behavior_key_list(datasets)
    """

    # Get list of all behaviors; examine the first dataset
    behavior_key_list = []
    
    # Iterate through each dictionary in the dataset
    for key, value in datasets[0].items():
        # Check if the dictionary has the key "combine_frames" 
        # (Could have also done this with "behavior_name")
        if isinstance(value, dict) and "combine_frames" in value:
            print('Adding behavior key:   ', key)
            # Add the value of "behavior_name" to the behavior_key_list list
            behavior_key_list.append(key)
    #print('PARTIAL BEHAVIORS!')
    #behavior_key_list = ["approaching_Fish0", "approaching_Fish1",
    #                    "fleeing_Fish0", "fleeing_Fish1"]
    
    return behavior_key_list
    

def select_items_dialog(behavior_key_list, default_keys=['perp_noneSee', 
        'perp_oneSees', 'perp_bothSee', 'contact_any', 'contact_head_body', 
        'contact_inferred', 'tail_rubbing', 'maintain_proximity', 
        'Cbend_Fish0', 'Cbend_Fish1', 
        'Jbend_Fish0', 'Jbend_Fish1', 'Rbend_Fish0', 'Rbend_Fish1', 
        'isActive_any', 'isMoving_any', 'approaching_Fish0', 
        'approaching_Fish1', 'fleeing_Fish0', 'fleeing_Fish1']):

    """
    Creates a dialog with checkboxes for each item in a list of strings,
    e.g. the keys of all possible behaviors in behavior_key_list.
    Default selection is based on default_keys.
    Returns a list of selected items.
    written by Claude 3.5 Sonnet
    
    Args:
        behavior_key_list (list): List of strings to display as options
        default_keys (list, optional): List of strings to select by default
    
    Returns:
        list: Selected items
    """
    if default_keys is None:
        default_keys = []
    
    # Create a separate Tk instance to avoid console freezing
    root = tk.Tk()
    root.title("Select Items")
    root.geometry("600x1000+100+100")
    
    # Initialize result with an empty list
    result = []
    
    frame = ttk.Frame(root, padding="10")
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Label at the top
    ttk.Label(frame, text="Select items:").pack(anchor=tk.W, pady=(0, 10))
    
    # Create variables to track selection state
    var_dict = {}
    for item in behavior_key_list:
        var = tk.BooleanVar(root)
        # Explicitly set default values
        if item in default_keys:
            var.set(True)
        else:
            var.set(False)
        var_dict[item] = var
    
    # Create a canvas with scrollbar for many items
    canvas = tk.Canvas(frame)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Add checkboxes for each item
    for item in behavior_key_list:
        ttk.Checkbutton(
            scrollable_frame, 
            text=item, 
            variable=var_dict[item]
        ).pack(anchor=tk.W, pady=2)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # OK and Cancel buttons
    button_frame = ttk.Frame(root)
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    
    def on_ok():
        nonlocal result
        result = [item for item, var in var_dict.items() if var.get()]
        root.quit()  # Use quit instead of destroy
    
    def on_cancel():
        root.quit()  # Use quit instead of destroy
    
    ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=5)
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    # Start the main loop
    root.mainloop()
    
    # After mainloop ends, destroy the window
    root.destroy()
    
    return result



def timeShift(position_data, dataset, CSVcolumns, fishIdx = 1, 
              frameShift = 7500):
    """
    Cyclicly shift all position data for one fish (fishIdx) by some number of 
    frames such that the fish position and heading in frame startFrame is
    the same, within tolerance, after shifting.
    For "randomizing" trajectory data, as a control for social interactions.
    Note that frames and index values are offset by 1
    Require some minimum number of frames to shift
    
    Note that the shifted start / end will have a large discontinuity in 
    positions. This can be “fixed” by marking the shifted start/end frames 
    for the "bad tracking" flag, with head positions as zeros.

    Parameters
    ----------
    position_data : basic position information for this dataset, numpy array
    dataset : dataset dictionary of all behavior information for a given expt.
    CSVcolumns: information on what the columns of position_data are
    fishIdx : index of fish to shift
    frameShift : number of frames by which to shift fishIdx's position and 
        heading data. 


    Returns
    -------
    position_data : cyclicly shifted position columns for fishIdx
    dataset : cyclicly shifted heading angle for fishIdx

    """
    
    if frameShift > dataset["Nframes"]:
        raise ValueError(f"frameShift {frameShift} is larger than the number of frames!")
                
    # Cyclicly shift position and heading data by shift_idx
    new_dataset = dataset.copy()
    new_position_data = position_data.copy()
    new_dataset["heading_angle"][:,fishIdx] = \
        np.roll(dataset["heading_angle"][:,fishIdx], frameShift, axis=0)
    # Note that position data cols 0 and 1 are ID no and frame no., so don't 
    # shift these (shouldn't matter...)
    new_position_data[:, 2:, fishIdx] = \
        np.roll(position_data[:, 2:, fishIdx], frameShift, axis=0)
    
    # Mark shifted frame and the frame before it as zeros for 
    # head and body posititions, to flag bad tracking
    new_position_data[frameShift-1:frameShift+1,CSVcolumns["head_column_x"],fishIdx] = 0.0
    new_position_data[frameShift-1:frameShift+1,CSVcolumns["head_column_y"],fishIdx] = 0.0
    new_position_data[frameShift-1:frameShift+1,CSVcolumns["body_column_x_start"]:CSVcolumns["body_Ncolumns"]+1,fishIdx] = 0.0
    new_position_data[frameShift-1:frameShift+1,CSVcolumns["body_column_y_start"]:CSVcolumns["body_Ncolumns"]+1,fishIdx] = 0.0
    
    return new_position_data, new_dataset

    
def fit_gaussian_mixture(x, n_gaussian=3, init_means=None, random_state=42):
    """
    Fit a Gaussian Mixture Model to data with outliers and identify the component
    initialized with the median value.
    
    Written by Claude Sonnet 3.7, with some modifications
    
    Ignore warnings, to avoid printing
        UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input data array containing values with outliers
    n_gaussian : int, default=3
        Number of Gaussian components to fit
    init_means : array-like, default=None
        Initial means for GMM components. If None, will use 
           [median, 0.75*median, 3*median]
           Note that the output (properties of the "main" peak) will be the
           properties of the peak with the *first* initialization value,
           so the order of init_means matters!
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple of float : Mean and std. dev of the component 
        initialized at the first value
    """
    
    # Ignore all warnings of type UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)

    # Handle case where there's not enough data
    if len(x) < n_gaussian or len(np.unique(x)) < n_gaussian:
        return np.median(x)  # Fallback to median if not enough data
    
    # Initialize the model
    gmm = GaussianMixture(
        n_components=n_gaussian,
        covariance_type='full',
        random_state=random_state,
        max_iter=100
    )
    
    # Reshape data for scikit-learn
    X = x.reshape(-1, 1)
    
    # Use custom initialization
    if init_means is None:
        median_val = np.median(x)
        # Initialize means at median, 0.75*median, and 3*median
        init_means = np.array([
            [median_val],
            [0.75 * median_val],
            [3 * median_val]
        ])
    
    # Only use as many initial means as requested components
    init_means = init_means[:n_gaussian]
    
    # Reshape; make an array
    init_means = np.array(init_means).reshape(-1,1) 
    
    # Set the initial parameters
    gmm.means_init = init_means
    
    # Fit the model
    try:
        gmm.fit(X)
        # Get component information
        means = gmm.means_.flatten()
        gmm_sigmas = np.sqrt(gmm.covariances_.flatten())
        # Return the mean of the component initialized at the first 
        #component in our initialization; should be the median
        # Reset all filters to default
        warnings.resetwarnings()
        return means[0], gmm_sigmas[0]
    except Exception as e:
        # Fallback to median if GMM fitting fails
        print(f'\nGMM fitting fails! Exception {e}\n Returning median, std.')
        # Reset all filters to default
        warnings.resetwarnings()
        return np.median(x), np.std(x)