# -*- coding: utf-8 -*-
# behavior_correlations.py
"""
Author:   Raghuveer Parthasarathy
Created on Wed Sept. 6, 2023
Last modified on Sept. 11, 2025

Description
-----------

Contains function(s) for calculating the correlation between different 
behavior events
Also contains functions for plotting quantitative properties in intervals
around behavior events


"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy.stats import linregress
import scipy.stats as st
import pickle
from itertools import cycle
from IO_toolkit import load_and_assign_from_pickle
from toolkit import get_fps, get_values_subset

def calc_correlations_with_defaults():
    """
    Run all the correlation (conditional probability) code, with default
    values. See "Correlation between behavior events notes v2" document,
    "To Run" section.
    
    Assumes that we’ve previously run the behavior identification analysis, 
    and that the output is saved in pickle files.
    Previously: Run behavior_correlations.py
    
    See documentation, and all the constituent functions, for details.
    
    Inputs
    ----------
    datasets : dictionary; All datasets to analyze

    Returns
    -------
                             
    behav_corr_allSets, binCenters, behavior_key_list, fps, corr_asymm,
    behavior_key_list_subset
    """
    all_position_data, variable_tuple = load_and_assign_from_pickle()
    (datasets, CSVcolumns, expt_config, params, N_datasets, Nfish,
             basePath, dataPath, subGroupName) = variable_tuple
    fps = get_fps(datasets, fpstol = 1e-6)
    behavior_key_list = ['Cbend_Fish0', 'Cbend_Fish1', 'Rbend_Fish0', 
                         'Rbend_Fish1', 'Jbend_Fish0', 'Jbend_Fish1', 
                         'perp_noneSee', 'perp_oneSees', 'perp_bothSee', 
                         'contact_any', 'tail_rubbing', 'maintain_proximity',
                         'approaching_Fish0', 'approaching_Fish1', 
                         'approaching_any', 
                         'fleeing_Fish0', 'fleeing_Fish1', 'fleeing_any']
    
    # Check that these keys are in datasets[0]
    filtered_list = [key for key in behavior_key_list if key in datasets[0]]
    # Print elements that are in behavior_key_list but not in dict_data
    missing_keys = [key for key in behavior_key_list if key not in datasets[0]]
    if missing_keys:
        print("Missing keys from behavior_key_list (not in datasets[0]):", 
              missing_keys)
    behavior_key_list = filtered_list

    binWidthFrames = 5 # bin size for delays (number of frames)
    halfFrameRange = 50 # max frame delay to consider
    # Calculate frame-delays between each event pair for each dataset.
    print('\nCalculating frame delays and binning...')
    behav_corr = calcDeltaFramesEvents(datasets, behavior_key_list, 
                                       max_delta_frame = 250)
    # Bin the frame delays, and calculate behavior correlations and probabilities for each dataset.
    behav_corr, binCenters = bin_deltaFrames(behav_corr, 
                                             behavior_key_list, 
                                             binWidthFrames = binWidthFrames, 
                                             halfFrameRange = halfFrameRange, 
                                             deleteDeltaFrames = True)
    behav_corr = calc_pAB(behav_corr, behavior_key_list, binCenters)
    
    # Combine behavior correlations and probabilities for all datasets
    print('\nCombining correlations for all datasets...')
    behav_corr_allSets = calcBehavCorrAllSets(behav_corr, behavior_key_list, binCenters)
    
    behavior_key_list_subset = ['Cbend_Fish0', 'Cbend_Fish1', 'Rbend_Fish0', 
                         'Rbend_Fish1', 'Jbend_Fish0', 'Jbend_Fish1', 
                         'perp_noneSee', 'perp_oneSees', 'perp_bothSee', 
                         'contact_any', 'tail_rubbing', 'maintain_proximity', 
                         'approaching_Fish0', 'approaching_Fish1', 
                         'fleeing_Fish0', 'fleeing_Fish1']
    # Check that these keys are in datasets[0]
    filtered_list = [key for key in behavior_key_list_subset if key in datasets[0]]
    # Print elements that are in behavior_key_list but not in dict_data
    missing_keys = [key for key in behavior_key_list_subset if key not in datasets[0]]
    if missing_keys:
        print("Missing keys from behavior_key_list_subset (not in datasets[0]):", 
              missing_keys)
    behavior_key_list_subset = filtered_list

    corr_asymm = calc_corr_asymm(behav_corr_allSets['DeltaCorr'], 
                                 behavior_key_list_subset, binCenters, 
                                 maxFrameDelay = None, normalization = 'none')
    
    return behav_corr, behav_corr_allSets, binCenters, behavior_key_list, \
        corr_asymm, fps, behavior_key_list_subset



def apply_behavior_constraints(dataset, behavior_A, 
                               min_duration_behavior = None, min_duration_fr=0, 
                               behavior_C=None, C_delta_f=(0, 1), 
                               constraintKey=None, constraintRange=None, 
                               constraintIdx=0, 
                               use_abs_value_constraint = False):
    """
    Apply all constraints to behavior A events and return filtered event frames.
    
    Parameters
    ----------
    dataset : dict, single dataset
    behavior_A : str, behavior name to constrain
    min_duration_behavior : str, behavior key for which to apply the minimum 
        duration; None for none and 'all' for all "Behavior A"s
    min_duration_fr : int, minimum duration constraint
    behavior_C : str or None, constraining behavior name
    C_delta_f : tuple, frame range for behavior C constraint
    constraintKey : str or None, key for quantitative constraint
    constraintRange : tuple or None, (min, max) for quantitative constraint
    constraintIdx : int or str, index/operation for constraint
    use_abs_value_constraint : bool, default False
                    If True, use absolute value of the quantitative constraint
                    property before applying constraints or combining values. 
                    Useful for signed angles (relative orientation, bending).    
    
    Returns
    -------
    filtered_frames : numpy array of behavior A start frames that meet all constraints
    """
    # Start with all behavior A events
    bA_start_frames = dataset[behavior_A]["combine_frames"][0].copy()
    bA_durations = dataset[behavior_A]["combine_frames"][1].copy()
    
    # Apply minimum duration constraint
    if (min_duration_behavior is not None) and (min_duration_fr > 0):
        if (min_duration_behavior == behavior_A) or (min_duration_behavior.lower() == 'all'):
            duration_mask = bA_durations >= min_duration_fr
            bA_start_frames = bA_start_frames[duration_mask]
            bA_durations = bA_durations[duration_mask]
            print(f'     {behavior_A}, After min duration constraint: {len(bA_start_frames)} events remain')
    
    # Apply behavior C constraint
    if behavior_C is not None and behavior_C in dataset:
        bC_start_frames = dataset[behavior_C]["combine_frames"][0]
        
        # For each behavior A event, check if any behavior C event occurs within range
        valid_A_mask = np.zeros(len(bA_start_frames), dtype=bool)
        for i, frameA in enumerate(bA_start_frames):
            # Calculate frame differences (C - A)
            frame_diffs = bC_start_frames - frameA
            # Check if any C event is within the specified range
            in_range = (frame_diffs >= C_delta_f[0]) & (frame_diffs <= C_delta_f[1])
            valid_A_mask[i] = np.any(in_range)
        
        bA_start_frames = bA_start_frames[valid_A_mask]
        print(f'     {behavior_A}, After behavior C constraint: {len(bA_start_frames)} events remain')
    
    # Apply quantitative constraint
    if constraintKey is not None and constraintRange is not None and constraintKey in dataset:
        # Check if we still have events to constrain
        if len(bA_start_frames) == 0:
            print(f'   {behavior_A}, After quantitative constraint: 0 events remain (no events to constrain)')
        else:
            try:
                # Get constraint values at behavior A start frames
                constraint_data = dataset[constraintKey]
                
                # Apply keyIdx operation to get the relevant constraint values
                constraint_values = get_values_subset(constraint_data, 
                                                      constraintIdx, 
                                                      use_abs_value = use_abs_value_constraint)
                    
                if len(constraint_values) == 0:
                    print(f'   Warning: Constraint data {constraintKey} is empty')
                    bA_start_frames = np.array([])
                else:
                    # For each behavior A event, check constraint at start frame
                    valid_constraint_mask = np.zeros(len(bA_start_frames), dtype=bool)
                    for i, frameA in enumerate(bA_start_frames):
                        frame_idx = int(frameA) - 1  # Convert to 0-based indexing
                        if 0 <= frame_idx < len(constraint_values):
                            try:
                                # Handle different array shapes and types
                                if constraint_values.ndim > 1 and constraint_values.shape[1] > 0:
                                    val_array = constraint_values[frame_idx]
                                    if hasattr(val_array, 'size') and val_array.size > 0:
                                        val = val_array.flatten()[0]
                                    else:
                                        val = np.nan
                                else:
                                    val = constraint_values[frame_idx]
                                    if hasattr(val, '__len__') and len(val) > 0:
                                        val = val[0]
                                
                                # Check if value is valid (not NaN) and within range
                                if not np.isnan(val):
                                    if use_abs_value_constraint:
                                        valid_constraint_mask[i] = \
                                            (constraintRange[0] <= np.abs(val) <= constraintRange[1])
                                    else:
                                        valid_constraint_mask[i] = \
                                            (constraintRange[0] <= val <= constraintRange[1])
                                
                            except (IndexError, TypeError, AttributeError) as e:
                                print(f'   Warning: Could not access constraint value at frame {frameA}: {e}')
                                valid_constraint_mask[i] = False
                        else:
                            # Frame index out of bounds
                            valid_constraint_mask[i] = False
                    
                    bA_start_frames = bA_start_frames[valid_constraint_mask]
                
                print(f'     {behavior_A}, After quantitative constraint: {len(bA_start_frames)} events remain')
            except Exception as e:
                print(f'   Warning: Error applying quantitative constraint: {e}')
                print('   Continuing without quantitative constraint for this dataset')
    
    return bA_start_frames


def calcDeltaFramesEvents(datasets, behavior_key_list, max_delta_frame = None,
                          min_duration_behavior = None, min_duration_fr=0, 
                         behavior_C=None, C_delta_f=(0, 1), constraintKey=None, 
                         constraintRange=None, constraintIdx=0, 
                         use_abs_value = False):
    """
    Calculate the delay between behavior "events" with optional constraints.

    For each event, note the relative time (delay in the number of frames) 
    of all other events, past and future, of the same and
    of different types. Save this list of frame delays (deltaFrames).
    
    Consider the relative times of the start of any runs of a given 
    behavior. For example, if one run of behavior A is at frames 3, 4, 5, 
    and behavior B is at frames 10, 11, the relative time of B is +7 (only)
    from A; only +7 is recorded. If Behavior A is 3, 4, 5, 15, 16, and 
    B is 10, 11, 12, then +7 (3 to 10) and -6 (16 to 10) are recorded.
    Note that positive numbers for corr_BA mean that B occurs after A. 
    
    Allows constraints:
    1. Minimum duration constraint for behavior A events
    2. Behavior C co-occurrence constraint 
    3. Quantitative property constraint at behavior A start frames
    
    
    Inputs
    ----------
    datasets : dictionary; All datasets to analyze
    behavior_key_list : list of all behavior to consider
    max_delta_frame : maximum Delta Frame value to save; ignore Df > this. 
                        Default None, but recommended to avoid memory errors!
    min_duration_behavior : str, behavior key for which to apply the minimum 
        duration; None for none and 'all' for all "Behavior A"s
    min_duration_fr : integer, minimum duration in frames for behavior A events
    behavior_C : string or None, constraining behavior name
    C_delta_f : tuple, frame range for behavior C constraint (start_offset, end_offset)
    constraintKey : string or None, key for quantitative constraint
    constraintRange : tuple or None, (min, max) values for quantitative constraint
    constraintIdx : integer or string, index/operation for multi-dimensional constraint
    use_abs_value : bool, default False
                    If True, use absolute value of the quantitative constraint
                    property before applying constraints or combining values. 
                    Useful for signed angles (relative orientation, bending).    

    Returns
    -------
    behav_corr : list of dictionaries; behav_corr[j] is for dataset j
        behav_corr_j[behavior_A]['pA'] is the probability of 
        behav_corr_j[behavior_A][behavior_B]["deltaFrames"]
              
    """
    
    print('Calculating delays between behavior "events" with constraints...')
    if (min_duration_behavior is not None) and (min_duration_fr > 0):
        print(f'  Minimum duration constraint for {min_duration_behavior}: {min_duration_fr} frames')
    if behavior_C is not None:
        print(f'  Behavior C constraint: {behavior_C} within {C_delta_f} frames')
    if constraintKey is not None and constraintRange is not None:
        print(f'  Quantitative constraint: {constraintKey} in range {constraintRange}')
    
    # Number of datasets
    N_datasets = len(datasets)
    
    # initialize nested dictionaries for datasets
    behav_corr = [{} for j in range(N_datasets)]
    for j in range(N_datasets):
        # keep the "dataset_name" key
        behav_corr_j = behav_corr[j]
        behav_corr_j["dataset_name"] = datasets[j]["dataset_name"]
        behav_corr_j["Nframes"] = datasets[j]["Nframes"]
        print('   Dataset ', j, '; ', behav_corr[j]["dataset_name"])
        for bA in behavior_key_list:
            behav_corr_j[bA] = {"allDeltaFrames": np.array([])}
            # initialize empty array:
            for bB in behavior_key_list:
                behav_corr_j[bA][bB] = {"deltaFrames": np.array([])}

        # Calculate frame delays and append to each deltaFrames list
        for behavior_A in behavior_key_list:
            # Apply all constraints to get filtered behavior A events
            bA_frames_filtered = apply_behavior_constraints(
                dataset = datasets[j], behavior_A = behavior_A, 
                min_duration_behavior = min_duration_behavior, 
                min_duration_fr=min_duration_fr,
                behavior_C=behavior_C, 
                C_delta_f=C_delta_f,
                constraintKey=constraintKey,
                constraintRange=constraintRange,
                constraintIdx=constraintIdx
            )
            
            # Marginal probability based on filtered events
            N_A = len(bA_frames_filtered)
            behav_corr_j[behavior_A]['pA'] = N_A / datasets[j]["Nframes"]
            behav_corr_j[behavior_A]['pA_unc'] = np.sqrt(N_A) / datasets[j]["Nframes"]
                        
            for behavior_B in behavior_key_list:
                # For each dataset, note each event and calculate the delay between
                # this and other events of both the same and different behaviors
                # Note: positive deltaFrames means behavior B is *after* behavior A
                bB_frames = datasets[j][behavior_B]["combine_frames"][0]
                
                if len(bA_frames_filtered) > 0 and len(bB_frames) > 0:
                    deltaFrames_temp = bB_frames[:, None] - bA_frames_filtered # all at once!
                    
                    if max_delta_frame is not None:
                        deltaFrames_temp = deltaFrames_temp[deltaFrames_temp <= 
                                                            max_delta_frame]
                    if behavior_A == behavior_B:
                        deltaFrames_temp = deltaFrames_temp[deltaFrames_temp != 0]

                    behav_corr_j[behavior_A][behavior_B]["deltaFrames"] = \
                        np.append(behav_corr_j[behavior_A][behavior_B]["deltaFrames"], 
                                  deltaFrames_temp.flatten())
                    # All the frame delays (for all Behaviors B)
                    behav_corr_j[behavior_A]["allDeltaFrames"] = \
                        np.append(behav_corr_j[behavior_A]["allDeltaFrames"], 
                                  deltaFrames_temp.flatten())
    # Note that behav_corr[j] is the same as behav_corr_j
    
    return behav_corr

def bin_deltaFrames(behav_corr, behavior_key_list, binWidthFrames = 25, 
                          halfFrameRange = 15000, deleteDeltaFrames = True):
    """
    Bin the "deltaFrames" delays between events of behaviors A, B
    for each dataset.
    Force the central bin to be centered at deltaFrames = 0
    This can take ~20 minutes for all keys and 60 datasets! 
    
    Inputs
    ----------
    behav_corr : list of dictionaries; behav_corr[j] is for dataset j
                 First key: dataset name
                 First key: "Nframes"; value: datasets[j]["Nframes"]
                 First key: behavior A (string)
                 Second key: "allDeltaFrames" for behavior A
                 Second key:  behavior B (string) under behavior A
                 Second key: "pA" : under behavior A, 
                             marginal probability for behavior A (i.e. 
                             simple probability indep of other behaviors)
                             "pA_unc" uncertainty of pA, assuming Poisson distr.
                 Third key: under behavior B; "deltaFrames", the frame delays  
                             between A, B for all events.
    behavior_key_list : list of all behaviors to consider
    binWidthFrames : width of bins for histogram, number of frames
        Default 25, usual 25 fps, so this is 1 second
    halfFrameRange : max possible DeltaFrames to bin; make bins from
        -halfFrameRange to +halfFrameRange, forcing a bin centered at 0.
        Default: 15000, which is probably the total number of frames
    deleteDeltaFrames : remove the deltaFrames key (all the frame-delays
                        for all A-B event pairs) from behav_corr, 
                        and remove "allDeltaFrames" (all frame-delays for A)
                        to save space
    
    Returns
    -------
    behav_corr : dictionary of dictionaries, updated
                 Third key (new): : "counts" : counts in each bin for 
                                     A, B at deltaFrames
    binCenters: bin centers (frames), from bin edges used for histogram
    """
    # for histogram of deltaFrames. Bin centers and edges. 
    # Force 0.0 to be at the center of a bin
    binCenters2 = np.arange(0.0, halfFrameRange+binWidthFrames, binWidthFrames)
    binCenters1 = -1.0*np.flipud(binCenters2)[:-1]
    binCenters = np.concatenate((binCenters1, binCenters2))
    binEdges = np.concatenate((binCenters - binWidthFrames/2.0, 
                               np.array([np.max(binCenters)+binWidthFrames/2.0])))
    
    # Number of datasets
    N_datasets = len(behav_corr)
    for j in range(N_datasets):
        print('Dataset ', j, '; ', behav_corr[j]["dataset_name"])
        # Calculate counts for each bin of frame delays
        for behavior_A in behavior_key_list:
            if deleteDeltaFrames:
                del behav_corr[j][behavior_A]["allDeltaFrames"]
            for behavior_B in behavior_key_list:
                # Histogram counts for deltaFrames_AB
                behav_corr[j][behavior_A][behavior_B]["counts"] = \
                    np.histogram(behav_corr[j][behavior_A][behavior_B]["deltaFrames"], 
                                 bins=binEdges)[0] # [0] to only get the counts array                   
                if deleteDeltaFrames:
                    del behav_corr[j][behavior_A][behavior_B]["deltaFrames"]

    return behav_corr, binCenters


def calc_pAB(behav_corr, behavior_key_list, binCenters):
    """
    From the binned "deltaFrames" delays between events of behaviors A, B
    for each dataset, calculate the normalized p(B (Delta t) | A) for each 
    frame delay, and the relative probability DeltaCorr = p(B | A) - p(B)
    These are both probability per frame.
    
    Inputs
    ----------
    behav_corr : nested dictionary, output by calcDeltaFramesEvents()
                 and then bin_deltaFrames()
    behavior_key_list : list of all behaviors, used in calcDeltaFramesEvents()
    binCenters : bin centers (frames), from bin edges used for histogram.
                 Also used to calculate binWidthFrames 


        
    Returns
    -------
    behav_corr : dictionary of dictionaries, updated
                 Third key: under behavior B; "pAB"
                 Fourth key: under "pAB", ["C"] and ["C_unc"], normalized 
                     p(B (Delta t) | A), numpy array corresponding to each
                     deltaFrames bin, and its uncertainty
                 Third key: under behavior B; "DeltaCorr"
                 Fourth key: under "DeltaCorr", ["C"] and ["C_unc"], normalized 
                     p(B (Delta t) | A) - p(B), numpy array corresponding to each
                     deltaFrames bin, and its uncertainty
    """
    
    # Number of datasets
    N_datasets = len(behav_corr)
    
    # Infer binWidthFrames from binCenters
    binWidthFrames = np.mean(np.diff(binCenters))

    for j in range(N_datasets):
        for bA in behavior_key_list:
            NA = behav_corr[j][bA]['pA']*behav_corr[j]["Nframes"] # No. of A events
            # Initialize the nested dictionaries
            for bB in behavior_key_list:
                behav_corr[j][bA][bB]["pAB"] = {'C': {}, 'C_unc': {}}
                behav_corr[j][bA][bB]["DeltaCorr"] = {'C': {}, 'C_unc': {}}
        
                # Calculate probability p(B | A)
                if NA > 0:
                    behav_corr[j][bA][bB]["pAB"]['C'] = \
                        behav_corr[j][bA][bB]['counts'] / NA / binWidthFrames
                    behav_corr[j][bA][bB]["pAB"]['C_unc'] = \
                        np.sqrt(behav_corr[j][bA][bB]['counts']) / NA / binWidthFrames
                else:
                    behav_corr[j][bA][bB]["pAB"]['C'] = np.nan
                    behav_corr[j][bA][bB]["pAB"]['C_unc'] = np.nan
                        
                # Calculate association probability p(B | A) - pB
                behav_corr[j][bA][bB]["DeltaCorr"]['C'] = \
                    behav_corr[j][bA][bB]["pAB"]['C'] - behav_corr[j][bB]["pA"]
                behav_corr[j][bA][bB]["DeltaCorr"]['C_unc'] = \
                    np.sqrt(behav_corr[j][bA][bB]["pAB"]['C_unc']**2 + 
                            behav_corr[j][bB]["pA_unc"]**2)

    return behav_corr



def calcBehavCorrAllSets(behav_corr, behavior_key_list, binCenters, 
                         n_bootstrap=1000):

    """
    Average the correlations for each behavior pair across datasets
    Averages DeltaCorr = p(B | A) - p(A)
        and  pAB = p(B | A)
        each binned by Delta Frames
        (Redundant code for these two)
    Calculate uncertainty by bootstrap resampling
    
    Parameters
    ----------
    behav_corr : nested dictionary, output by calcDeltaFramesEvents()
                 and then bin_deltaFrames() and calc_pAB()
    behavior_key_list : list of all behaviors, used in calcDeltaFramesEvents()
    binCenters : bin centers (frames), from bin edges used for histogram

    Returns
    -------
    behav_corr_allSets : Likelihood of behaviors preceding / following
                         other behaviors. 
                         Dictionary. 
                         First key: 'DeltaCorr', pAB
                         Second Key [behaviorA]
                         Third Key [behaviorB]
                         Fourth key: {[C], [C_unc]}
                         Then [C] and [C_unc] for the probability and its
                           uncertainty, each a numpy array, C being the 
                           relative likelihood of event B following 
                           event A in each of the frame delay bins with 
                           center binCenters
                         ...['DeltaCorr']...['C'] = p(B | A) - p(A), which is 
                         where a numpy array for each DeltaFrames bin,
                         with each p normalized.


    """
    # Number of datasets
    N_datasets = len(behav_corr)
    # Number of bins
    Nbins = len(binCenters)

    # Create array to hold correlations for all datasets
    behav_DeltaCorr_array = np.zeros((N_datasets, len(behavior_key_list), 
                                 len(behavior_key_list), Nbins), dtype=float)
    behav_pAB_array = np.zeros((N_datasets, len(behavior_key_list), 
                                 len(behavior_key_list), Nbins), dtype=float)
    for j in range(N_datasets):
        for ibA, bA in enumerate(behavior_key_list):
            for ibB, bB in enumerate(behavior_key_list):
                behav_DeltaCorr_array[j, ibA, ibB, :] = \
                    behav_corr[j][bA][bB]["DeltaCorr"]["C"]
                behav_pAB_array[j, ibA, ibB, :] = \
                    behav_corr[j][bA][bB]["pAB"]["C"]
                    
    # Average across all datasets
    behav_DeltaCorr_array_mean = np.nanmean(behav_DeltaCorr_array, axis=0)
    behav_pAB_array_mean = np.nanmean(behav_pAB_array, axis=0)
    
    # Initialize arrays for bootstrap uncertainty calculation
    bootstrap_DeltaCorr_array = np.zeros((n_bootstrap, len(behavior_key_list), 
                                   len(behavior_key_list), Nbins))
    bootstrap_pAB_array = np.zeros((n_bootstrap, len(behavior_key_list), 
                                   len(behavior_key_list), Nbins))
    
    # Bootstrap resampling
    print('... Bootstrap resampling... ')
    for i in range(n_bootstrap):
        # Sample dataset indices with replacement
        sample_indices = np.random.choice(np.arange(N_datasets), 
                                          size=N_datasets, replace=True)
        
        # Get the bootstrap sample arrays
        bootstrap_DeltaCorr_sample = behav_DeltaCorr_array[sample_indices, :, :, :]
        bootstrap_pAB_sample = behav_pAB_array[sample_indices, :, :, :]
        
        # Average across resampled datasets
        bootstrap_DeltaCorr_array_avg = np.nanmean(bootstrap_DeltaCorr_sample, axis=0)
        bootstrap_pAB_array_avg = np.nanmean(bootstrap_pAB_sample, axis=0)
                
        # Calculate normalized values for this bootstrap sample
        for ibA, bA in enumerate(behavior_key_list):
            for ibB, bB in enumerate(behavior_key_list):
                bootstrap_DeltaCorr_array[i, ibA, ibB, :] = \
                    bootstrap_DeltaCorr_array_avg[ibA, ibB, :]
                bootstrap_pAB_array[i, ibA, ibB, :] = \
                    bootstrap_pAB_array_avg[ibA, ibB, :]
                    
    # Calculate standard deviation for each metric across bootstrap samples
    bootstrap_DeltaCorr_uncertainty = np.nanstd(bootstrap_DeltaCorr_array, axis=0)
    bootstrap_pAB_uncertainty = np.nanstd(bootstrap_pAB_array, axis=0)

    # Put this into a nested dictionary
    # initialize nested dictionaries for datasets
    behav_corr_allSets = {"DeltaCorr": {}, "pAB": {}}
    
    for ibA, bA in enumerate(behavior_key_list):
        # Initialize the nested dictionaries
        behav_corr_allSets["DeltaCorr"][bA] = {}
        behav_corr_allSets["pAB"][bA] = {}
        for ibB, bB in enumerate(behavior_key_list):
            behav_corr_allSets["DeltaCorr"][bA][bB] = {'C': {}, 'C_unc': {}}
            behav_corr_allSets["pAB"][bA][bB] = {'C': {}, 'C_unc': {}}
            
            # Means for each normalization method
            behav_corr_allSets["DeltaCorr"][bA][bB]['C'] = \
                behav_DeltaCorr_array_mean[ibA, ibB, :]
            behav_corr_allSets["pAB"][bA][bB]['C'] = \
                behav_pAB_array_mean[ibA, ibB, :]
            
            # Uncertainties
            behav_corr_allSets["DeltaCorr"][bA][bB]['C_unc'] = \
                bootstrap_DeltaCorr_uncertainty[ibA, ibB, :] 
            behav_corr_allSets["pAB"][bA][bB]['C_unc'] = \
                bootstrap_pAB_uncertainty[ibA, ibB, :] 
     
    return behav_corr_allSets


def calc_significant_correlations(behav_dict, behavior_key_list, 
                                  binCenters, CL = 0.95,
                                  frameDelayRange = (-np.inf, np.inf)):
    """
    Calculate whether each correlation of behaviors A, B in C are
    significant relative to the null correlation in C_null at the desired
    confidence level. See "What correlations are significant?" in Correlation
    notes; March 7, 2025

    Parameters
    ----------
    behav_dict : dictionary of correlations (i.e. probabilities of B|A); 
        probably behav_corr_allSets['DeltaCorr'] 
        E.g. behav_corr_allSets['DeltaCorr'][behaviorA][ behaviorB]['C'] 
            is a numpy array of probabilities at each binCenters
    behavior_key_list : list of all behaviors, used in calcDeltaFramesEvents()
    binCenters : numpy array; center of each DeltaFrames bin.
    CL : desired confidence level
    frameDelayRange : Tuple of frame delay range to consider; 
        restrict true values to to binCenters >= frameDelayRange(0) and <= ...(1)
    
    
    Returns
    -------
    behav_corr_significant : each item of behav_corr_significant[A][B] is
       be a numpy array of boolean values for each binCenters Δframe, 
       true if the probability C['normSimple'][A][B]['C'] at that Δt is 
       significantly different from the null C_null['normSimple'][A][B]. 
    validDelays : boolean numpy array of same size as binCenters, True if
        these frame delay values are in the range given by frameDelayRange
    """
    behav_corr_significant = {}
    z_star = st.norm.ppf(CL)
    validDelays = (binCenters >= frameDelayRange[0]) & (binCenters 
                                                        <= frameDelayRange[1])

    printDiagnostics = False
    for ibA, bA in enumerate(behavior_key_list):
        behav_corr_significant[bA] = {}
        for ibB, bB in enumerate(behavior_key_list):
            UCB = behav_dict[bA][bB]['C'] + z_star*behav_dict[bA][bB]['C_unc']
            LCB = behav_dict[bA][bB]['C'] - z_star*behav_dict[bA][bB]['C_unc']
            # Significance at confidence level CL, for the frame delays to be
            # considered.
            behav_corr_significant[bA][bB] = (((behav_dict[bA][bB]['C'] > 0) & (LCB > 0)) | 
                                              ((behav_dict[bA][bB]['C'] < 0) & (UCB < 0))) & validDelays
            if printDiagnostics & (bA == 'perp_oneSees'):
                print('Here. ', bA, bB)
                print(behav_dict[bA][bB]['C'])
                print(behav_dict[bA][bB]['C_unc'])
                print(LCB)
                print(binCenters)
                print(behav_corr_significant[bA][bB])
                _ = input('junk input to terminate ')
    return behav_corr_significant, validDelays
    

def calc_corr_asymm(behav_corr_allSets, behavior_key_list, binCenters, 
                    maxFrameDelay = None, normalization = 'none'):
    """
    Calculate the temporal asymmetry in the correlation function between 
    each pair of behaviors. Probably use "DeltaCorr" = p(B|A) - P(B) as the 
    correlation to consider.
    Ignore uncertainties

    Parameters
    ----------
    behav_corr_allSets : dictionary of correlations for each behavior pair. 
        E.g. for DeltaCorr, input behav_corr_allSets['DeltaCorr'], a 
             numpy array of association probabilities at each binCenters
        
    behavior_key_list : list of all behaviors, used in calcDeltaFramesEvents()

    binCenters : numpy array; center of each DeltaFrames bin.
        
    maxFrameDelay : the abs. value max frame delay to consider ; Default None: 
        consider the full range over which correlations were calculated
        
    normalization: normalization of the correlation asymmetry, string; ignore case
        'none' : no normalization
        'sum' : normalize by sum of + , - times
        'rms' : normalize by RMS of correlation values

    Returns
    -------
    corr_asymm : dictionary of correlation asymmetry for each behavior pair

    """
    
    normalization = normalization.lower() # lowercase.
    
    # Calculate the asymmetries. Use a nested dictionary
    # initialize nested dictionaries for datasets
    # There's probably a better way to do this...
    if maxFrameDelay is None:
        maxFrameDelay = np.max(binCenters) + 0.001 # add a slight offset
    corr_asymm = {}
    for ibA, bA in enumerate(behavior_key_list):
        corr_asymm[bA] = {}
        thisProbA = behav_corr_allSets[bA]
        for ibB, bB in enumerate(behavior_key_list):
            thisProbAplus = np.sum(thisProbA[bB]['C'][(binCenters>0) & 
                                                 (binCenters <= maxFrameDelay)])
            thisProbAminus = np.sum(thisProbA[bB]['C'][(binCenters<0) &
                                                  (binCenters >= -1.0*maxFrameDelay)])
            probDifference = thisProbAplus - thisProbAminus
            if normalization == 'sum':
                denominator = thisProbAplus + thisProbAminus
            elif normalization == 'rms':
                denominator = np.std(thisProbA[bB]['C'][(binCenters <= maxFrameDelay) 
                                                        & (binCenters >= -1.0*maxFrameDelay)])
            elif normalization != 'none' :
                raise ValueError("Invalid normalization option!")
            else:
                denominator = 1.0
            if denominator > 0.0:
                corr_asymm[bA][bB] = probDifference / denominator
            else:
                corr_asymm[bA][bB] = 0.0 # technically undefined, but I want zeros
                
            """
            print('Here.')
            print(thisProbAplus)
            print(thisProbAminus)
            print(corr_asymm[bA][bB])
            x = input('asdf : ')
            """

    return corr_asymm



def plot_behaviorCorrelation(behav_corr_dict, binCenters, 
                             behavior_key_list, behaviorA, behaviorB='',
                             titleString = '', 
                             fps = 1.0, plotShadedUnc = False,
                             xlim = None, ylim = None,
                             outputFileName = None):
    """
    Plot Behavior B likelihood following/preceding Behavior A
    Can plot a single A-B pair, or all B for a given A
    If a single A-B pair, also include the mean value as a dashed line
    Plots all types of normalizations (simple, across Behaviors, and across Time)
    
    Inputs:
        behav_corr_dict : a behavior correlation dictionary, with probability
                            of behaviors preceding / following other behaviors.
                            Can be e.g. behav_corr_allSets['DeltaCorr'], 
                            in any case having
                            the structure [behaviorA][behaviorB]{[C], [C_unc]}. 
    binCenters : bin centers (frames), for plotting
    behavior_key_list : list of behaviors to plot. (Can be a subset of all)
    behaviorA , B: (string) Behavior A, B, to plot, if  just plotting 
                   one pair. Leave B empty ('') to skip plotting
                   a single A-B pair, and only plot all pairs
    titleString : title string for plots, probably describing the dictionary
                    or normalization type, e.g. "simple norm" or 
                    "normAcrossBehavior" or "normAcrossTime" or "Delta C"
    fps : frames per second (probably 25, but default to 1)
    plotShadedUnc : Boolean, if true plot shaded error bands from correlation
                    uncertainties; default False
    xlim : tuple of x axis limits, None for auto (defaults)
    ylim : tuple of y axis limits, None for auto (defaults)
    outputFileName : if not None, save figure with this filename
    """

    # Just one AB pair    
    if not behaviorB=='':

        plt.figure(figsize=(6,5))
        corrAB = behav_corr_dict[behaviorA][behaviorB]['C']
        plt.plot(binCenters/fps, corrAB, color='mediumvioletred')
        meanCorr = np.mean(behav_corr_dict[behaviorA][behaviorB]['C'])
        plt.plot(binCenters/fps, meanCorr*np.ones(binCenters.shape), 
                 linestyle='dashed', color='orchid')
        if plotShadedUnc:
            corrABunc = behav_corr_dict[behaviorA][behaviorB]['C_unc']
            plt.fill_between(binCenters/fps, corrAB - corrABunc, 
                             corrAB + corrABunc, color='mediumvioletred', alpha=0.3)
        plt.xlabel(r'$\Delta$t (s)', fontsize=20)
        plt.ylabel('Probability', fontsize=20)
        plt.title(f'{titleString}: {behaviorA} then {behaviorB}', fontsize=22)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    
    # cmap_name = 'viridis' 
    cmap = plt.cm.rainbow(np.linspace(0, 1, len(behavior_key_list))) 
    # cmap = plt.cm.tab10(np.linspace(0, 1, len(behavior_key_list))) 
    # cmap = plt.colormaps[cmap_name]
    
    # line styles
    lines = ["-","--","-.",":"]
    linecycler = cycle(lines) 

    # All behavior pairs
    
    plt.figure(figsize=(12, 9))
    for j, bB in enumerate(behavior_key_list):
        corrAB = behav_corr_dict[behaviorA][bB]['C']
        plt.plot(binCenters/fps, corrAB, color=cmap[j,:], label=bB, 
                 linewidth=2.0, linestyle = next(linecycler))
        if plotShadedUnc:
            corrABunc = behav_corr_dict[behaviorA][bB]['C_unc']
            plt.fill_between(binCenters/fps, corrAB - corrABunc, 
                             corrAB + corrABunc, color=cmap[j,:], alpha=0.3)
    plt.xlabel(r'$\Delta$t (s)', fontsize=20)
    plt.ylabel('Probability', fontsize=20)
    plt.title(f'{titleString}: {behaviorA} then each behavior', fontsize=22)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    if xlim is not None:
        plt.xlim(xlim)
        
    if ylim is not None:
        plt.ylim(ylim)

    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight')    
    
    # For export, Mar. 1, 2024
    """
    plt.plot(np.array([0.0, 0.0]), np.array([0.0, 0.06]), linestyle=':', 
             color='gray')
    plt.ylim((0.0, 0.06))
    plt.xlim((-1.0, 1.0))
    plt.savefig('behavior_correlations_raw.eps', dpi=300)
    """


def plot_corr_asymm(corr_asymm, crange = None, titleString = '', 
                    outputFileName = None):
    """
    plot a 2D heatmap of correlation temporal asymmetry for each behavior pair

    Parameters
    ----------
    corr_asymm : dictionary, for each type of correlation, of correlation
                    asymmetry for each behavior pair
    crange : tuple of max, min correlation to which to scale the colormap
    titleString : title string for the plot
    outputFileName : if not None, save figure with this filename
    
    Returns
    -------
    None.

    """
    # Extract the behavior keys for the axes
    behaviorkeys = list(corr_asymm.keys())

    # Create a matrix of the values
    matrix = np.array([[corr_asymm[key1][key2] for key2 in behaviorkeys] for key1 in behaviorkeys])

    # Color range; am not checking that crange is a tuple with 2 elements
    # If not input, set to +/- max(abs), so it's centered at 0
    if crange is None:
        vmax = np.max(np.abs(matrix))
        vmin = -1.0*vmax
    else:
        vmin = crange[0]
        vmax = crange[1]
        
    # Plot the heatmap using matplotlib
    plt.figure(figsize=(12, 9))
    plt.imshow(matrix, cmap='coolwarm', interpolation='nearest', vmin=vmin, 
               vmax=vmax)
    plt.colorbar()
    
    # Add labels to the axes
    plt.xticks(ticks=np.arange(len(behaviorkeys)), labels=behaviorkeys, rotation=90)
    plt.yticks(ticks=np.arange(len(behaviorkeys)), labels=behaviorkeys)
    
    # Add title and axis labels
    plt.title(f'{titleString} time asymmetry', fontsize=22)
    plt.xlabel('Behavior', fontsize=18)
    plt.ylabel('Behavior', fontsize=18)

    # Show the plot
    plt.show()
    
    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight')    
    



def save_correlation_pickle(outputPickleFileName, behav_corr, behav_corr_allSets, 
                            binCenters, behavior_key_list, fps, corr_asymm, 
                            outputPicklePathName = None):
    """
    Save behavior correlation output to a pickle file

    Inputs: 
    outputPickleFileName : name of the Pickle file in which to save 
                            outputs. Will append ".pickle" if not there
                            Note that pickle file will be large!
    variables to be saved in the .pickle file
    outputPicklePathName : The output path; if None, ignore
    """
    
    list_for_pickle = [behav_corr, behav_corr_allSets, binCenters, 
                                behavior_key_list, fps, corr_asymm]

    # add .pickle if it's not there.
    # Check if there is a "." in the file name string
    if "." in outputPickleFileName:
        # Split the string at the last "."
        ext = outputPickleFileName.rsplit(".", 1)[-1]
    else:
        # Return an empty string if there is no "."
        ext = ""
    if ext.lower() != 'pickle':
        outputPickleFileName = outputPickleFileName + '.pickle'
        
    if outputPicklePathName is not None:
        outputPickleFileName = os.path.join(outputPicklePathName, 
                                            outputPickleFileName)

    print(f'\nWriting pickle file: {outputPickleFileName}\n')
    with open(outputPickleFileName, 'wb') as handle:
        pickle.dump(list_for_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)



def get_duration_info(CSVfilename = ''):
    """
    Read the CSV output by calc_relative_durations_csv(), 
    containing fish length differences
    
    Inputs:
        CSVfilename : CSV file name containing dataset names (col 1) 
            and relative durations. Reading stops at the first blank
            row, and so will ignore the mean, etc.
            Leave empty to list csv files in the current directory that 
            have 'relDuration' in the file name. If there's only one,
            use it as default.

    Outputs:
        duration_data : Pandas dataframe. Indexed by dataset name 
            (remove '_2wpf', '_light', '_dark')
            Can, for example, print all info for a dataset with:
                print(duration_data.loc['3b_k7'])
            Can, for example, plot all sets' contact duration vs. fish
                length difference with:
                plt.scatter(duration_data['Mean difference in fish lengths (px)'], duration_data['Contact (any)'], marker='o')
        
    Raghuveer Parthasarathy
    Sept. 17, 2023
    """
    
    print('Current directory: ', os.getcwd())
    input_directory = input('Input directory containing .csv file (blank for current dir.): ')
    if input_directory == '':
        input_directory = os.getcwd()
        
    if CSVfilename=='':
        # List all files in the directory
        file_list = os.listdir(input_directory)
        # Filter files with "relDuration" in their filename and .csv extension
        filtered_files = [file for file in file_list if "relDuration" in file and file.endswith('.csv')]
        if len(filtered_files)==1:
            print('File found: ', filtered_files)
            use_ff = input('Use this CSV file? (y or n): ')
            if use_ff.lower() == 'y':
                CSVfilename = filtered_files[0]
            else: 
                CSVfilename = input('Enter the CSV filename, including .csv: ')
        else:
            # Print the filtered file names
            print('Suggested files: ')
            for file_name in filtered_files:
                print(file_name)
            CSVfilename = input('Enter the CSV filename, including .csv: ')
    CSVfilename = os.path.join(input_directory, CSVfilename)

    # Initialize an empty DataFrame to store the data
    duration_data = pd.DataFrame()

    # Open the CSV file for reading
    with open(CSVfilename, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
    
        # Read the header row
        header = next(csvreader)
    
        # Check if the header is present and has at least one column
        if header:
            # Get the column names from the header (not including the dataset names)
            column_names = header[1:]
    
            # Read the data row by row
            for row in csvreader:
                if not any(row):
                    break # exit the loop; don't read further if the row is empty

                # Extract the name (first column) and convert it to a string
                name = row[0]
                # Remove "2wpf", "_light", or "_dark" from the dataset name
                name = name.replace('_2wpf', '').replace('_light', '').replace('_dark', '')
    
                # Extract the durations (remaining columns) and convert them to floats
                durations_row = [float(dataval) for dataval in row[1:]]
    
                # Create a temporary DataFrame for the current row
                temp_df = pd.DataFrame([durations_row], columns=column_names, 
                                       index=[name])
                # Append the temporary DataFrame to the main DataFrame
                duration_data = pd.concat([duration_data, temp_df], ignore_index=False)
    
    Ndatasets = len(duration_data)
    print(f'Number of datasets: {Ndatasets}')
    
    # Print the extracted column names
    print(' ')
    print("Column Names:", column_names)

    return duration_data



def length_difference_correlation(CSVfilename = '', behavior_to_plot=''):
    """
    Calls get_duration_info to read the CSV output by 
    calc_relative_durations_csv(), containing fish length differences, etc.
    Plot / examine correlations between the fish length differences 
    and the durations of each behavior.
    
    Consider the subset of datasets in Laura Desban's list of sets
    for which the program-derived difference in fish lengths is similar
    to the manually-derived list.
    
    Inputs:
        CSVfilename : CSV file name containing dataset names (col 1) 
            and relative durations. Reading stops at the first blank
            row, and so will ignore the mean, etc.
            Leave empty to list csv files in the current directory that 
            have 'relDuration' in the file name. If there's only one,
            use it as default.
        behavior_to_plot : (string) name of the behavior to plot. Leave
            blank for none. Must exactly match column headings in CSV file,
            e.g. '90deg-One'
        
    Outputs:
        
    Raghuveer Parthasarathy
    Sept. 17, 2023
    """
    
    duration_data = get_duration_info(CSVfilename)  # Will ask for CSV file info
    
    # Datasets with manually verified lengths that agree with 
    # the automated lengths
    good_datasets = ['3b_k11', '3b_k3', '3b_k5', '3b_k9', '3b_nk1', 
                     '3b_nk6','3c_k11', '3c_k1', '3c_k2', '3c_k3', 
                     '3c_k4', '3c_k5','3c_k7', '3c_nk1', '3c_nk6', 
                     '5b_k2', '3b_k10', '3b_k1','3b_k4', '3b_nk11', 
                     '3b_nk2', '3b_nk3', '3c_k10','3c_k6', '3c_nk2', 
                     '3c_nk4', '3c_nk7', '3c_nk8','3c_nk9', '5b_k10', 
                     '5b_k4', '5b_k7', '5b_k9','5b_nk2', '5b_nk3', 
                     '5b_nk5', '5b_nk6', '5b_nk8']
    print(f'Number of possible datasets to consider: {len(good_datasets)}')
    
    # Let's make a filtered set of only the data in the "good_datasets" list
    filtered_df = duration_data[duration_data.index.isin(good_datasets)]
    
    # Length differences
    length_diff = filtered_df["Mean difference in fish lengths (px)"]
    print(f'Number of filtered datasets: {len(length_diff)}')
    
    print('here')
    print(filtered_df.columns)
    print(behavior_to_plot)
    # Check if "behavior_to_plot" column exists in the filtered DataFrame
    if behavior_to_plot in filtered_df.columns:
        # Extract the data from the specified columns
        behavior_values = filtered_df[behavior_to_plot]
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(length_diff, behavior_values)
        # Print the regression results
        print("Linear Regression Results:")
        print(f"Slope: {slope:.5f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"R-squared: {r_value**2:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Standard Error: {std_err:.5f}")
    
        # Create a scatter plot
        plt.figure()
        plt.scatter(length_diff, behavior_values, marker='o', 
                    color='deepskyblue', label=behavior_to_plot)

        # Add the linear regression line to the plot
        plt.plot(np.array(length_diff), intercept + slope * np.array(length_diff), 
                 color='steelblue', label='Linear Regression')
        
        # Add labels and a legend
        plt.xlabel("Length Difference (px)")
        plt.ylabel(behavior_to_plot + ' duration')
        plt.title(f'{behavior_to_plot}. Slope: {slope:.2e} +/- {std_err:.2e} ({100*std_err/slope:.0f}%)')
        plt.legend()
    
        # Show the plot
        plt.show()
        
    else:
        print("The specified behavior is not found in the filtered DataFrame.")

    return duration_data



def array_stats(x, stats_axis = 0):
    """
    Array stats, ignore NaN (note that this isn't accounted for in the "N" for
                             s.e.m., but that should be a tiny effect)
    If the array has only one row (along stats_axis), std and sem are NaNs
    
    Inputs
    --------
    x : Numpy array x
    
    Returns
    --------
    stats : dictionary with keys
        "mean" : mean of x along axis 0
        "std" : standard deviation
        "sem" : standard error of the mean
        
    """
    
    mean_val = np.nanmean(x, axis=stats_axis)
    if x.shape[stats_axis]==1:
        # Fill manually with NaNs to avoid a warning!
        std_val = np.full_like(mean_val, np.nan)
        sem_val = np.full_like(mean_val, np.nan)
    else:
        std_val = np.nanstd(x, axis=stats_axis, ddof=1)  
        sem_val = std_val / np.sqrt(x.shape[stats_axis])

    stats = {
        "mean": mean_val,
        "std": std_val,
        "sem": sem_val
    }
    return stats

    
def pool_quant_property_from_behavior_1dataset(dataset, 
                                               behavior = 'maintain_proximity',
                                               qproperty = 'relative_orientation',
                                               duration_s = (-0.2, 0.6),
                                               fishID = (0,1),
                                               use_abs_value = False,
                                               min_duration_fr = 0,
                                               behavior_C = None,
                                               C_delta_f = (0, 1), 
                                               constraintKey = None,
                                               constraintRange = None,
                                               constraintIdx = 0,
                                               use_abs_value_constraint = False):
    """
    Tabulate all values of a quantitative property in the vicinity of the start
    of a particular behavior event, for some time duration before and after.
    Also calculate statistics (mean, std, sem)
    Acts on a single dataset.
    Supports constraints, similar to calcDeltaFramesEvents()
    See July 2025 notes
    
    Inputs
    ----------
    dataset : single dataset dictionary; probably datasets[j]
              must include "fps", "Nframes" 
    behavior : behavior from the start of which to evaluate quantitative properties
    q_property : quantitative property to examine
    duration_s : duration to consider, seconds. Can be:
                 - tuple (start_offset, end_offset) where start_offset is typically negative
                   to include time before behavior start, end_offset is positive for time after
                 - scalar: treated as (0, duration_s)
    fishID : fishID to consider for each output axis==2 coordinates, 
             or instructions for this. Ignored if the quantity
             isn't individual specific (e.g. head-head distance)
             If fish-specific, should be a tuple. Each element
             can be 0 or 1, keeping usual id assignment,
             or a string 'phi_low' or 'phi_high' for low and
             high absolute value of relative orientation, probably indicating 
             approaching or fleeing fish.
    use_abs_value : bool, default False
                    If True, use absolute value of the quantitative 
                    property before applying constraints or combining values. 
                    Useful for signed angles (relative orientation, bending).    
    min_duration_fr : int, minimum duration constraint in frames for behavior events
                     (default 0 = no constraint)
    behavior_C : str or None, co-occurrence constraint behavior name
                Only consider behavior events if behavior_C occurs 
                within C_delta_f frames
    C_delta_f : tuple, frame range for behavior C constraint (start_offset, end_offset)
               Default (0, 1) means behavior_C must occur within 0 to 1 frames after behavior
    constraintKey : str or None, key for quantitative constraint at behavior start frames
    constraintRange : tuple or None, (min, max) values for quantitative constraint
    constraintIdx : int or str, index/operation for multi-dimensional constraint
                   (0 for first column, 'min'/'max'/'mean' for operations)
    use_abs_value_constraint : bool, default False
                    If True, use absolute value of the quantitative constraint
                    property before applying constraints or combining values. 
                    Useful for signed angles (relative orientation, bending).    

    Returns
    -------
    all_qprop_stats :  dictionary containing mean, std dev, s.e.m. of all_qprop, 
                        shape of each key ("mean", etc."): 
                                           (duration_frames, # fish vals)
    all_qprop : numpy array containing all quantitative property values for
                        each detected behavior event. 
                        Shape: (# of events, duration_frames, # fish vals)
    t_array : array of time values corresponding to the duration, seconds
    
        
    """

    # print('   Dataset ', dataset["dataset_name"])
    
    # Check that index 0 corresponds to frame 1. (I should write
    #    more generally...)
    if dataset["frameArray"][0] != 1:
        raise ValueError('Error! First frame of frameArray should be 1')
        
    # Handle duration_s parameter 
    if isinstance(duration_s, tuple):
        start_offset_s, end_offset_s = duration_s
    else:
        # Treat scalar as (0, duration_s)
        start_offset_s, end_offset_s = 0, duration_s
    
    # Convert time offsets to frame offsets
    start_offset_frames = np.round(start_offset_s * dataset["fps"]).astype(int)
    end_offset_frames = np.round(end_offset_s * dataset["fps"]).astype(int)
    
    # Total duration in frames
    duration_frames = end_offset_frames - start_offset_frames

    # All initial frames and durations of the behavior
    b_start_frames = dataset[behavior]["combine_frames"][0].copy()
    b_durations = dataset[behavior]["combine_frames"][1].copy()

    # Apply minimum duration constraint
    if min_duration_fr > 0:
        print(f'     {behavior}, initially {len(b_start_frames)} events. ', end='')
        duration_mask = b_durations >= min_duration_fr
        b_start_frames = b_start_frames[duration_mask]
        b_durations = b_durations[duration_mask]
        print(f'After min duration constraint: {len(b_start_frames)} events')
    
    # Apply behavior C co-occurrence constraint
    if behavior_C is not None and behavior_C in dataset:
        bC_start_frames = dataset[behavior_C]["combine_frames"][0]
        
        # For each behavior event, check if any behavior C event occurs within range
        valid_mask = np.zeros(len(b_start_frames), dtype=bool)
        for i, frameA in enumerate(b_start_frames):
            # Calculate frame differences (C - A)
            frame_diffs = bC_start_frames - frameA
            # Check if any C event is within the specified range
            in_range = (frame_diffs >= C_delta_f[0]) & (frame_diffs <= C_delta_f[1])
            valid_mask[i] = np.any(in_range)
        
        b_start_frames = b_start_frames[valid_mask]
        print(f'     {behavior}, After behavior C constraint: {len(b_start_frames)} events remain')
        
    # Apply quantitative constraint
    if constraintKey is not None and constraintRange is not None and constraintKey in dataset:
        if len(b_start_frames) == 0:
            print(f'     {behavior} 0 events remain (no events to constrain)')
        else:
            try:
                # Get constraint values at behavior start frames
                constraint_data = dataset[constraintKey]
                
                # Apply keyIdx operation to get the relevant constraint values
                constraint_values = get_values_subset(constraint_data, 
                                                      constraintIdx,
                                                      use_abs_value = use_abs_value_constraint)
                    
                if len(constraint_values) == 0:
                    print(f'   Warning: Constraint data {constraintKey} is empty')
                    b_start_frames = np.array([])
                else:
                    print(f'     {behavior}, initially {len(b_start_frames)} events. ', end='')
                    # For each behavior event, check constraint at start frame
                    valid_constraint_mask = np.zeros(len(b_start_frames), dtype=bool)
                    for i, frameA in enumerate(b_start_frames):
                        frame_idx = int(frameA) - 1  # Convert to 0-based indexing
                        if 0 <= frame_idx < len(constraint_values):
                            try:
                                # Handle different array shapes and types
                                if constraint_values.ndim > 1 and constraint_values.shape[1] > 0:
                                    val_array = constraint_values[frame_idx]
                                    if hasattr(val_array, 'size') and val_array.size > 0:
                                        val = val_array.flatten()[0]
                                    else:
                                        val = np.nan
                                else:
                                    val = constraint_values[frame_idx]
                                    if hasattr(val, '__len__') and len(val) > 0:
                                        val = val[0]
                                
                                # Check if value is valid (not NaN) and within range
                                if not np.isnan(val):
                                    if use_abs_value_constraint:
                                        valid_constraint_mask[i] = (constraintRange[0] <= np.abs(val) <= constraintRange[1])
                                    else:
                                        valid_constraint_mask[i] = (constraintRange[0] <= val <= constraintRange[1])

                                
                            except (IndexError, TypeError, AttributeError) as e:
                                print(f'   Warning: Could not access constraint value at frame {frameA}: {e}')
                                valid_constraint_mask[i] = False
                        else:
                            # Frame index out of bounds
                            valid_constraint_mask[i] = False
                    
                    b_start_frames = b_start_frames[valid_constraint_mask]
                
                print(f'After quantitative constraint: {len(b_start_frames)} events')
            except Exception as e:
                print(f'   Warning: Error applying quantitative constraint: {e}')
                print('   Continuing without quantitative constraint for this dataset')

    
    # Keep only frames that are within valid bounds considering both start and end offsets
    # Start frame + start_offset must be >= 0 (since frameArray starts at frame 1, index 0)
    # Start frame + end_offset must be <= Nframes
    valid_bounds_mask = ((b_start_frames + start_offset_frames >= 1) & 
                        (b_start_frames + end_offset_frames <= dataset["Nframes"]))
    b_start_frames = b_start_frames[valid_bounds_mask]
    
    # Should be integers, but I think might sometimes be saved as float (?)
    b_start_frames = b_start_frames.astype(int)
    
    N_events = len(b_start_frames)
    
    if N_events == 0:
        # No (valid) occurrences of this behavior
        print(f'     {behavior}, No valid events after applying constraints')
        return None, None, None
    
    print(f'     {behavior}, Final number of events: {N_events}')
    
    # Assess how many values per frame the quantitative property has (one 
    # per fish, or just one). Initialize array
    # squeeze to reduce dimensionality of shape==1 axis, for single-value properties
    if len(dataset[qproperty].squeeze().shape) > 1:
        Nval = dataset[qproperty].squeeze().shape[-1]
        all_qprop = np.zeros((N_events, duration_frames+1, Nval))
    else:
        Nval = 1
        all_qprop = np.zeros((N_events, duration_frames+1))

    # Get bad tracking frames for NaN replacement
    bad_frames = None
    if 'bad_bodyTrack_frames' in dataset and 'raw_frames' in dataset['bad_bodyTrack_frames']:
        bad_frames = dataset['bad_bodyTrack_frames']['raw_frames']
        
    for j in range(N_events):
        # Calculate actual start and end indices for this event
        actual_start_idx = b_start_frames[j] - 1 + start_offset_frames  # -1 for 0-based indexing
        actual_end_idx = b_start_frames[j] - 1 + end_offset_frames + 1  # +1 for inclusive end
        
        # Get the actual frame numbers for this time window
        frame_numbers = np.arange(b_start_frames[j] + start_offset_frames, 
                                 b_start_frames[j] + end_offset_frames + 1)
      
        if Nval > 1:
            if fishID == (0,1):
                idx_array = np.array([0, 1]) # use fish IDs
            elif fishID == ('phi_low','phi_high') :
                rel_orient = dataset['relative_orientation'][b_start_frames[j]-1,:]
                idx_array = np.argsort(np.abs(rel_orient))
            else:
                raise ValueError('fishID value not recognized')
            for k in range(Nval):
                if use_abs_value:
                    all_qprop[j, :, k] = np.abs(
                        dataset[qproperty][actual_start_idx : 
                                                            actual_end_idx, 
                                                            idx_array[k]])
                else:                        
                    all_qprop[j, :, k] = dataset[qproperty][actual_start_idx : 
                                                            actual_end_idx, 
                                                            idx_array[k]]
                # Replace values with NaN for bad tracking frames
                if bad_frames is not None:
                    bad_mask = np.isin(frame_numbers, bad_frames)
                    all_qprop[j, bad_mask, k] = np.nan

        else:
            all_qprop[j, :] = dataset[qproperty][actual_start_idx : actual_end_idx].squeeze()
            # Replace values with NaN for bad tracking frames
            if bad_frames is not None:
                bad_mask = np.isin(frame_numbers, bad_frames)
                all_qprop[j, bad_mask] = np.nan
                
    # print(all_qprop[:,:,0])
    
    # Average across all events
    all_qprop_stats = array_stats(all_qprop, stats_axis = 0)
    
    # time array
    t_array = np.arange(start_offset_frames, end_offset_frames+1)/dataset["fps"]

    
    return all_qprop_stats, all_qprop, t_array


    
def pool_quant_property_from_behavior_all_datasets(datasets, 
                                               behavior = 'maintain_proximity',
                                               qproperty = 'relative_orientation',
                                               duration_s = (-0.2, 0.6),
                                               fishID = (0,1),
                                               use_abs_value = False, 
                                               min_duration_fr = 0,
                                               behavior_C = None,
                                               C_delta_f = (0, 1),
                                               constraintKey = None,
                                               constraintRange = None,
                                               constraintIdx = 0,
                                               use_abs_value_constraint = False):
    """
    Tabulate all values of a quantitative property following the initiation
    of a particular behavior event, for some time duration.
    Also calculate statistics (mean, std, sem)
    Acts on all datasets by calling 
    pool_quant_property_from_behavior_1dataset for each dataset
    Average & stats across datasets; two versions, each dataset mean
    gets equal weight, and each event (pooled from all datasets) gets
    equal weight.
    
    Inputs
    ----------
    datasets : list of all dataset dictionaries
              must include "fps", "Nframes" 
    behavior : behavior from the start of which to evaluate quantitative properties
    q_property : quantitative property to examine
    duration_s : duration to consider, seconds. Can be:
                 - tuple (start_offset, end_offset) where start_offset is typically negative
                   to include time before behavior start, end_offset is positive for time after
                 - scalar: treated as (0, duration_s)
    fishID : fishID to consider for each output axis==2 coordinates, 
             or instructions for this. Ignored if the quantity
             isn't individual specific (e.g. head-head distance)
             If fish-specific, should be a tuple. Each element
             can be 0 or 1, keeping usual id assignment,
             or a string 'phi_low' or 'phi_high' for low and
             high absolute value of relative orientation, probably indicating 
             approaching or fleeing fish.
    use_abs_value : bool, default False
                    If True, use absolute value of the quantitative 
                    property before applying constraints or combining values. 
                    Useful for signed angles (relative orientation, bending).    
    min_duration_fr : int, minimum duration constraint in frames for behavior events
                     (default 0 = no constraint)
    behavior_C : str or None, co-occurrence constraint behavior name
                Only consider behavior events if behavior_C occurs within C_delta_f frames
    C_delta_f : tuple, frame range for behavior C constraint (start_offset, end_offset)
               Default (0, 1) means behavior_C must occur within 0 to 1 frames after behavior
    constraintKey : str or None, key for quantitative constraint at behavior start frames
    constraintRange : tuple or None, (min, max) values for quantitative constraint
    constraintIdx : int or str, index/operation for multi-dimensional constraint
                   (0 for first column, 'min'/'max'/'mean' for operations)
    use_abs_value_constraint : bool, default False
                    If True, use absolute value of the quantitative constraint
                    property before applying constraints or combining values. 
                    Useful for signed angles (relative orientation, bending).    
                   
    Returns
    -------
    all_qprop_stats_eachEvent :  
        mean, std, s.e.m. of all_qprop, i.e. all events concatenated; 
        shape (duration_frames, # fish vals)
    all_qprop_stats_eachSet :  dictionary containing
        mean, std, s.e.m. of all_qprop_means, i.e. stats of each dataset's mean val.; 
    all_qprop_means :  mean of each dataset's all_qprop, 
                shape (duration_frames, # fish vals)
    all_qprop : numpy array containing all quantitative property values for
                each detected behavior event, concatenated from all datasets
                Shape: (# of events, duration_frames, # fish vals)
    t_array : array of time values corresponding to the duration, seconds
    
    """

    # Print constraint information if constraints are being applied
    if min_duration_fr > 0:
        print(f'  Applying minimum duration constraint: {min_duration_fr} frames')
    if behavior_C is not None:
        print(f'  Applying behavior C constraint: {behavior_C} within {C_delta_f} frames')
    if constraintKey is not None and constraintRange is not None:
        print(f'  Applying quantitative constraint: {constraintKey} in range {constraintRange}')

    # Assess how many values per frame the quantitative property has (one 
    # per fish, or just one). Initialize array. Use datasets[0] to get the
    # number of values (e.g. 1 per fish)
    # Don't use len(datasets) as the first-dimension shape, since some datasets
    # may not have the behavior

    all_qprop_list = [] # Don't know the shape, so will use a list
    all_qprop_means_list = [] # Don't know the shape, so will use a list
    
    print(f'Pooling quantitative properties around {behavior} from {len(datasets)} datasets.')
    for j in range(len(datasets)):
        if datasets[j]["fps"] != datasets[0]["fps"]:
            raise ValueError("Error: fps not consistent between datasets")
        # print('   Dataset ', j, '; ', datasets[j]["dataset_name"])
        qprop_stats1, all_qprop1, t_array = \
            pool_quant_property_from_behavior_1dataset(
                                datasets[j],
                                behavior = behavior,
                                qproperty = qproperty,
                                duration_s = duration_s,
                                fishID = fishID,
                                min_duration_fr = min_duration_fr,
                                behavior_C = behavior_C,
                                C_delta_f = C_delta_f,
                                constraintKey = constraintKey,
                                constraintRange = constraintRange,
                                constraintIdx = constraintIdx, 
                                use_abs_value = use_abs_value)
        if qprop_stats1 is not None:
            all_qprop_list.append(all_qprop1)
            all_qprop_means_list.append(np.expand_dims(qprop_stats1["mean"], 
                                                       axis=0))    
    if len(all_qprop_list) == 0:
        print('Warning: No datasets had valid events after applying constraints')
        return None, None, None, None, None
    
    all_qprop = np.concatenate(all_qprop_list, axis=0)
    all_qprop_means = np.concatenate(all_qprop_means_list, axis=0)
        
    # Average across all events
    print(f'Total number of behavior events: {all_qprop.shape[0]}')
    all_qprop_stats_eachEvent = array_stats(all_qprop, stats_axis = 0)
    # Average across all the mean values of each dataset
    print(f'Total number of datasets: {all_qprop_means.shape[0]}')
    all_qprop_stats_eachSet = array_stats(all_qprop_means, stats_axis = 0)
    
    
    return all_qprop_stats_eachEvent, all_qprop_stats_eachSet, \
            all_qprop_means, all_qprop, t_array


def plot_quant_property_array(all_qprop_stats, all_qprop, idx = None, 
                              t_array = None, titleString = None, 
                              yLabelString = 'Property', 
                              xLabelString = 'Time (fr. or s?)',
                              ylim = None, 
                              outputFileName = None):
    """
    Plot the pooled quantitative property data calculated by
    pool_quant_property_from_behavior_all_datasets()
    See that file for details.
    
    idx : quant property array index to plot. 
          if None, plot each quantitative property array
    
    outputFileName : if not None, save figure with this filename
    """


    # Plot each row as a gray line, and mean / std as colored with band
    if t_array.any() == None:
        t_array = np.arange(all_qprop_stats.shape[1])
        
    if (idx!=None) and (len(all_qprop_stats["mean"].squeeze().shape)) == 1:
        print('\nIndex given, but quant. property has only one value. Ignoring...')
        idx = None
        
    plt.figure(figsize = (10,10))
    if idx == None:
        for j in range(all_qprop.shape[0]):
            plt.plot(t_array, all_qprop[j,:], color='gray', alpha=0.5)
        plt.plot(t_array, all_qprop_stats["mean"], color='orange', label='Mean')
        plt.fill_between(t_array, 
                         all_qprop_stats["mean"] - all_qprop_stats["std"], 
                         all_qprop_stats["mean"] + all_qprop_stats["std"], 
                         color='goldenrod', alpha=0.25, label='Std Dev')
        plt.fill_between(t_array, 
                         all_qprop_stats["mean"] - all_qprop_stats["sem"], 
                         all_qprop_stats["mean"] + all_qprop_stats["sem"], 
                         color='goldenrod', alpha=0.5, label='S.E.M.')
    else:
        for j in range(all_qprop.shape[0]):
            plt.plot(t_array, all_qprop[j,:,idx], color='gray', alpha=0.5)
        plt.plot(t_array, all_qprop_stats["mean"][:,idx], color='orange', label='Mean')
        plt.fill_between(t_array, 
                         all_qprop_stats["mean"][:,idx] - all_qprop_stats["std"][:,idx], 
                         all_qprop_stats["mean"][:,idx] + all_qprop_stats["std"][:,idx], 
                         color='goldenrod', alpha=0.25, label='Std Dev')
        plt.fill_between(t_array, 
                         all_qprop_stats["mean"][:,idx] - all_qprop_stats["sem"][:,idx], 
                         all_qprop_stats["mean"][:,idx] + all_qprop_stats["sem"][:,idx], 
                         color='goldenrod', alpha=0.5, label='S.E.M.')
    if ylim is not None:
        plt.ylim(ylim)
    # If the time-range spans zero, add a dotted line at zero
    if (np.min(t_array) < 0.0) and (np.max(t_array) > 0.0):
        ymin, ymax = plt.ylim()
        plt.vlines(x=0, ymin=ymin, ymax=ymax, colors='dodgerblue', 
                   linestyles='dashed', linewidth = 3.0, alpha = 0.5)
    plt.xlabel(xLabelString, fontsize = 16)
    plt.ylabel(yLabelString, fontsize = 16)
    if titleString is not None:
        plt.title(titleString, fontsize = 18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight')
    

def plot_qprop_pair_scatter(all_qprop, t_array, alpha=0.4, figsize=(8, 6),
                               xlabel='Fish ID 0', ylabel='Fish ID 1', 
                               title='Quant. Property, 2 fish Over Time',
                               cmap='magma', marker_size=20, show_colorbar=True):
    """
    Plot the quantitative property of one fish vs. the other, using the
    concatenated all_qprop array.
    
    for each j in range(all_qprop.shape[0]), and each f in 
    range(all_qprop.shape[0]), plots as a point 
    (all_qprop[j,f,0], all_qprop[j,f,1)), 
    Color by time value. 
    
    Code mostly from Claude Sonnet 4
                                                                                                                                                                                                                                                        
    Inputs:
        
    all_qprop : numpy array containing all quantitative property values for
                each detected behavior event, concatenated from all datasets
                Shape: (# of events, duration_frames, # fish vals)
    t_array : array of time values corresponding to the duration, seconds
    ...
    cmap : str or colormap, default 'magma'
        Colormap to use for time coloring
    marker_size : float, default 20
        Size of scatter points
    show_colorbar : bool, default True
        Whether to show colorbar
    """
    
    if all_qprop.ndim != 3 or all_qprop.shape[2] != 2:
        raise ValueError("all_qprop must have shape (Ne, Nt, 2)")
    
    if len(t_array) != all_qprop.shape[1]:
        raise ValueError("t_array length must match all_qprop time dimension")
    
    Ne, Nt, _ = all_qprop.shape
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Flatten the data for efficient plotting
    x_data = all_qprop[:, :, 0].flatten()  # Fish 0 values
    y_data = all_qprop[:, :, 1].flatten()  # Fish 1 values
    
    # Create time values for each point (repeat t_array for each event)
    time_data = np.tile(t_array, Ne)
    
    # Create scatter plot with time-based coloring
    scatter = ax.scatter(x_data, y_data, c=time_data, cmap=cmap, 
                        alpha=alpha, s=marker_size)
    
    # Create colorbar if requested
    if show_colorbar:
        # Use explicit normalization for better colorbar appearance
        norm = plt.Normalize(vmin=t_array.min(), vmax=t_array.max())
        scatter.set_norm(norm)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time (s)', rotation=270, labelpad=15)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax