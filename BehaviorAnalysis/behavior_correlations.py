# -*- coding: utf-8 -*-
# behavior_correlations.py
"""
Author:   Raghuveer Parthasarathy
Created on Wed Sep  6 13:38:21 2023
Last modified on Wed Sep  6 13:38:21 2023

Description
-----------

Contains function(s) for calculating the correlation between different 
behavior events.
For each event, note relative time of all other events of same and of different types. Save this, combine for datasets. Merge lists of numbers, histogram. See asymmetry. Fit / get moments?
Consider the relative times of the start of any runs of a given behavior. For example, if one run of behavior A is at frames 3, 4, 5, and behavior B is at frames 10, 11, the relative time of B is +7 (only) from A; only +7 is recorded. If Behavior A is 3, 4, 5, 15, 16, and B is 10, 11, 12, then +7 (3 to 10) and -6 (16 to 10) are recorded.

Inputs:
    
Outputs:
    

"""

import numpy as np
import matplotlib.pyplot as plt


def calcDeltaFramesEvents(datasets, binWidthFrames = 25, frameRange = (-15000, 15000)):
    """
    Calculate the correlation between behavior “events” – i.e. 
    likelihood that one behavior precedes / follows another. 
    For each event, note the relative time (delay in the number of frames) 
    of all other events, past and future, of the same and
    of different types. Save this list of frame delays (deltaFrames); 
    combine for datasets. 
    
    Consider the relative times of the start of any runs of a given 
    behavior. For example, if one run of behavior A is at frames 3, 4, 5, 
    and behavior B is at frames 10, 11, the relative time of B is +7 (only)
    from A; only +7 is recorded. If Behavior A is 3, 4, 5, 15, 16, and 
    B is 10, 11, 12, then +7 (3 to 10) and -6 (16 to 10) are recorded.
    Note that positive numbers for corr_BA mean that B occurs after A. 

    Parameters
    ----------
    datasets : dictionary
        All datasets to analyze
    binWidth : width of bins for histogram, number of frames
        Default 25, usual 25 fps, so this is 1 second
    frameRange : (-15000, 15000) min and max possible DeltaFrames,
        which should be +/- total number of frames
    

    Returns
    -------
    behav_corr : dictionary of dictionaries
                 Each element contains "deltaFrames", a 
                 2D (nested) dictionary of the frame delay between
                 behaviors A, B for all events.
    binCenters: bin centers, from bin edges used for histogram
    behavior_key_list : list of all behaviors

    """

    behavior_key_list = ["perpendicular_noneSee", 
                        "perpendicular_oneSees", "perpendicular_bothSee", 
                        "perpendicular_larger_fish_sees", 
                        "perpendicular_smaller_fish_sees", 
                        "contact_any", "contact_head_body", 
                        "contact_larger_fish_head", "contact_smaller_fish_head", 
                        "contact_inferred", "tail_rubbing", "bending"]
    
    # Keep correlation results in a separate dictionary, "behav_corr"
    # Number of datasets
    N_datasets = len(datasets)
    
    # for histogram
    binEdges = np.arange(frameRange[0], frameRange[1], binWidthFrames)
    binCenters = 0.5*(binEdges[1:] + binEdges[:-1])
    
    # initialize a list of dictionaries for datasets
    behav_corr = [{} for j in range(N_datasets)]
    for j in range(N_datasets):
        # keep the dataset_name" key
        behav_corr[j]["dataset_name"] = datasets[j]["dataset_name"]
        # create a 2D (nested) dictionary
        behav_corr[j]["deltaFrames"] = {}
        # And more nested dictionaries, for bin counts
        behav_corr[j]["counts"] = {}
        behav_corr[j]["norm_counts"] = {}
        for behavior_A in behavior_key_list:
            behav_corr[j]["deltaFrames"][behavior_A] = {}
            behav_corr[j]["counts"][behavior_A] = {}
            behav_corr[j]["norm_counts"][behavior_A] = {}
        # ... with an empty list at each element
        for behavior_A in behavior_key_list:
            for behavior_B in behavior_key_list:
                behav_corr[j]["deltaFrames"][behavior_A][behavior_B] = np.array([])
                behav_corr[j]["counts"][behavior_A][behavior_B] = np.array([])
                behav_corr[j]["norm_counts"][behavior_A][behavior_B] = np.array([])
        # Calculate frame delays and append to each deltaFrames list
        for behavior_A in behavior_key_list:
            behav_corr[j]["allDeltaFrames"] = np.array([])
            for behavior_B in behavior_key_list:
                # For each dataset, note each event and calculate the delay between
                # this and other events of both the same and different behaviors
                # Note: positive deltaFrames means behavior A is *after* behavior B
                bA_frames = datasets[j][behavior_A]["combine_frames"][0]
                bB_frames = datasets[j][behavior_B]["combine_frames"][0]
                for k in bA_frames:
                    deltaFrames_temp = bB_frames - k
                    if behavior_A == behavior_B:
                        deltaFrames_temp = deltaFrames_temp[deltaFrames_temp != 0]
                    if behavior_A == "perpendicular_noneSee" and behavior_B == "contact_any": 
                        print('j  =' , j)
                        print(behavior_A)
                        print(behavior_B)
                        print('k = ', k, '  bB_frames = ', bB_frames)
                        print('Delta Frames: ', deltaFrames_temp.flatten())
                    behav_corr[j]["deltaFrames"][behavior_A][behavior_B] = \
                        np.append(behav_corr[j]["deltaFrames"][behavior_A][behavior_B], 
                                  deltaFrames_temp.flatten())
                    # All the frame delays (for all Behaviors B)
                    behav_corr[j]["allDeltaFrames"] = \
                        np.append(behav_corr[j]["allDeltaFrames"], 
                                  deltaFrames_temp.flatten())
                behav_corr[j]["counts"][behavior_A][behavior_B] = \
                    np.histogram(behav_corr[j]["deltaFrames"][behavior_A][behavior_B], 
                                 bins=binEdges)[0] # [0] to just get the counts array
            # Histogram counts of all the behaviors' Frame Delays, rel. to A
            behav_corr[j]["counts_all"] = np.histogram(behav_corr[j]["allDeltaFrames"], 
                                                       bins=binEdges)[0]
            # Normalize the deltaFrames_AB by the total for all deltaFrames
            for behavior_B in behavior_key_list:
                behav_corr[j]["norm_counts"][behavior_A][behavior_B] = \
                    behav_corr[j]["counts"][behavior_A][behavior_B] / \
                        behav_corr[j]["counts_all"]

    return behav_corr, binCenters, behavior_key_list

def calcBehavCorrAllSets(behav_corr, behavior_key_list, binCenters):
    """
    Average the normalized correlations (binned Delta Frames for a 
            given behavior, relative to all the behaviors) across datasets

    Parameters
    ----------
    behav_corr : nested dictionary, output by calcDeltaFramesEvents()
    behavior_key_list : list of all behaviors, used in calcDeltaFramesEvents()
    binCenters : bin centers, for plotting

    Returns
    -------
    behav_corr_allSets : 

    """
    # Number of datasets
    N_datasets = len(behav_corr)
        # initialize a list of dictionaries for datasets
    behav_corr_allSets = {}
    # create a 2D (nested) dictionary
    behav_corr_allSets["norm_counts"] = {}
    for behavior_A in behavior_key_list:
        behav_corr_allSets["norm_counts"][behavior_A] = {}
        for behavior_B in behavior_key_list:
            behav_corr_allSets["norm_counts"][behavior_A][behavior_B] = \
                np.zeros((len(binCenters),N_datasets))
            for j in range(N_datasets):
                behav_corr_allSets["norm_counts"][behavior_A][behavior_B][:,j] = \
                    behav_corr[j]["norm_counts"][behavior_A][behavior_B]
                print(behav_corr[j]["norm_counts"][behavior_A][behavior_B])
    
    return behav_corr_allSets