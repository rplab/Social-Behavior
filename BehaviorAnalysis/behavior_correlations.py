# -*- coding: utf-8 -*-
# behavior_correlations.py
"""
Author:   Raghuveer Parthasarathy
Created on Wed Sept. 6, 2023
Last modified on June 24, 2025

Description
-----------

Contains function(s) for calculating the correlation between different 
behavior events

Inputs:
    
Outputs:
    

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy.stats import linregress
import scipy.stats as st
import pickle
from toolkit import load_and_assign_from_pickle, get_fps

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
    behav_corr = calcDeltaFramesEvents(datasets, behavior_key_list)
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


    
def calcDeltaFramesEvents(datasets, behavior_key_list):
    """
    Calculate the delay between behavior “events” – i.e. 
    the intervals that one behavior precedes / follows another. 
    For each event, note the relative time (delay in the number of frames) 
    of all other events, past and future, of the same and
    of different types. Save this list of frame delays (deltaFrames).
    
    Consider the relative times of the start of any runs of a given 
    behavior. For example, if one run of behavior A is at frames 3, 4, 5, 
    and behavior B is at frames 10, 11, the relative time of B is +7 (only)
    from A; only +7 is recorded. If Behavior A is 3, 4, 5, 15, 16, and 
    B is 10, 11, 12, then +7 (3 to 10) and -6 (16 to 10) are recorded.
    Note that positive numbers for corr_BA mean that B occurs after A. 
    
    Faster w/ ChatGPT revisions! Avoid repeated access to a nested dictionary,
    and avoid looping through frames
    
    Inputs
    ----------
    datasets : dictionary; All datasets to analyze
    behavior_key_list : list of all behavior to consider

    Returns
    -------
    behav_corr : list of dictionaries; behav_corr[j] is for dataset j
                 First key: dataset name
                 First key: "Nframes"; value: datasets[j]["Nframes"]
                 First key: behavior A (string)
                 Second key: "allDeltaFrames" for behavior A
                 Second key:  behavior B (string) under behavior A
                 Second key: "pA" : under behavior A, 
                             marginal probability for behavior A (i.e. 
                             simple probability indep of other behaviors).
                             Normalized by total number of frames, so 
                             probability per frame.
                             "pA_unc" uncertainty of pA, assuming Poisson distr.
                 Third key: under behavior B; "deltaFrames", the frame delays  
                             between A, B for all events.
                             
        
    To use: behav_corr = calcDeltaFramesEvents(datasets, behavior_key_list)
    """

    
    print('Calculating delays between behavior “events” ...')
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
            bA_frames = datasets[j][behavior_A]["combine_frames"][0]
            # Marginal probability
            N_A = len(bA_frames)
            behav_corr_j[behavior_A]['pA'] = N_A / datasets[j]["Nframes"]
            behav_corr_j[behavior_A]['pA_unc'] = np.sqrt(N_A) / datasets[j]["Nframes"]
                        
            for behavior_B in behavior_key_list:
                # For each dataset, note each event and calculate the delay between
                # this and other events of both the same and different behaviors
                # Note: positive deltaFrames means behavior A is *after* behavior B
                bB_frames = datasets[j][behavior_B]["combine_frames"][0]
                deltaFrames_temp = bB_frames[:, None] - bA_frames # all at once!

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
                               np.array([np.max(binCenters)+binWidthFrames/2.0 - 1])))
    
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
                x = input('junk input to terminate ')
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
                             fps = 1.0, plotShadedUnc = False):
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
        plt.title(f'{behaviorA} then {behaviorB}; {titleString}', fontsize=22)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    
    # cmap_name = 'viridis' 
    cmap = plt.cm.rainbow(np.linspace(0, 1, len(behavior_key_list))) 
    # cmap = plt.colormaps[cmap_name]
    
    # All behavior pairs
    
    plt.figure(figsize=(8, 6))
    for j, bB in enumerate(behavior_key_list):
        corrAB = behav_corr_dict[behaviorA][bB]['C']
        plt.plot(binCenters/fps, corrAB, color=cmap[j,:], label=bB, linewidth=2.0)
        if plotShadedUnc:
            corrABunc = behav_corr_dict[behaviorA][bB]['C_unc']
            plt.fill_between(binCenters/fps, corrAB - corrABunc, 
                             corrAB + corrABunc, color=cmap[j,:], alpha=0.3)
    plt.xlabel(r'$\Delta$t (s)', fontsize=20)
    plt.ylabel('Probability', fontsize=20)
    plt.title(f'{behaviorA} then each behavior; {titleString}', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()

    # For export, Mar. 1, 2024
    """
    plt.plot(np.array([0.0, 0.0]), np.array([0.0, 0.06]), linestyle=':', 
             color='gray')
    plt.ylim((0.0, 0.06))
    plt.xlim((-1.0, 1.0))
    plt.savefig('behavior_correlations_raw.eps', dpi=300)
    """


def plot_corr_asymm(corr_asymm, crange = None, titleString = ''):
    """
    plot a 2D heatmap of correlation temporal asymmetry for each behavior pair

    Parameters
    ----------
    corr_asymm : dictionary, for each type of correlation, of correlation
                    asymmetry for each behavior pair
    crange : tuple of max, min correlation to which to scale the colormap

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
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='coolwarm', interpolation='nearest', vmin=vmin, 
               vmax=vmax)
    plt.colorbar()
    
    # Add labels to the axes
    plt.xticks(ticks=np.arange(len(behaviorkeys)), labels=behaviorkeys, rotation=90)
    plt.yticks(ticks=np.arange(len(behaviorkeys)), labels=behaviorkeys)
    
    # Add title and axis labels
    plt.title(f'{titleString} time asymmetry', fontsize=18)
    plt.xlabel('Behavior', fontsize=16)
    plt.ylabel('Behavior', fontsize=16)

    # Show the plot
    plt.show()



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