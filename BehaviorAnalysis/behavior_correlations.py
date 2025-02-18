# -*- coding: utf-8 -*-
# behavior_correlations.py
"""
Author:   Raghuveer Parthasarathy
Created on Wed Sept. 6, 2023
Last modified on February 18, 2025

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
import pickle
import tkinter as tk
from tkinter import ttk


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

def calcDeltaFramesEvents(datasets):
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
    
    Faster w/ ChatGPT input! Avoid repeated access to a nested dictionary,
    and avoid looping through frames

    Inputs
    ----------
    datasets : dictionary
        All datasets to analyze
    binWidthFrames : width of bins for histogram, number of frames
        Default 25, usual 25 fps, so this is 1 second
    frameRange : (-15000, 15000) min and max possible DeltaFrames,
        which should be +/- total number of frames
    

    Returns
    -------
    behav_corr : dictionary of dictionaries
                 First key: dataset
                 Second and third keys: behaviors A, B
                 Third key: "allDeltaFrames" for behavior A
                 Fourth key: "deltaFrames", the frame delays between A, B 
                             for all events.
    behavior_key_list : list of all behaviors considered

    To use: behav_corr, behavior_key_list = calcDeltaFramesEvents(datasets)
    """

    behavior_key_list = ["perp_noneSee", 
                        "perp_oneSees", "perp_bothSee", 
                        "perp_larger_fish_sees", 
                        "perp_smaller_fish_sees", 
                        "contact_any", "contact_head_body", 
                        "contact_larger_fish_head", "contact_smaller_fish_head", 
                        "contact_inferred", "tail_rubbing", 
                        "Cbend_Fish0", "Cbend_Fish1", 
                        "Jbend_Fish0", "Jbend_Fish1", 
                        "Rbend_Fish0", "Rbend_Fish1", 
                        "isActive_any", "isMoving_any",
                        "approaching_Fish0", "approaching_Fish1",
                        "fleeing_Fish0", "fleeing_Fish1",
                        ]
    #behavior_key_list = ["approaching_Fish0", "approaching_Fish1",
    #                    "fleeing_Fish0", "fleeing_Fish1"]
    #print('PARTIAL BEHAVIORS!')
    
    # Number of datasets
    N_datasets = len(datasets)
    
    # initialize nested dictionaries for datasets
    behav_corr = [{} for j in range(N_datasets)]
    for j in range(N_datasets):
        # keep the dataset_name" key
        behav_corr_j = behav_corr[j]
        behav_corr_j["dataset_name"] = datasets[j]["dataset_name"]
        print('Dataset ', j, '; ', behav_corr[j]["dataset_name"])
        for bA in behavior_key_list:
            behav_corr_j[bA] = {"allDeltaFrames": np.array([])}
            for bB in behavior_key_list:
                behav_corr_j[bA][bB] = {"deltaFrames": np.array([])}

        # Calculate frame delays and append to each deltaFrames list
        for behavior_A in behavior_key_list:
            bA_frames = datasets[j][behavior_A]["combine_frames"][0]
                        
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
    
    return behav_corr, behavior_key_list


def bin_deltaFrames(behav_corr, behavior_key_list, binWidthFrames = 25, 
                          halfFrameRange = 15000, 
                          outputPickleFileName = None):
    """
    Bin the "deltaFrames" delays between events of behaviors A, B
    for each dataset.
    Force the central bin to be centered at deltaFrames = 0
    This can take ~20 minutes for all keys and 60 datasets, so save outputs
    in outPickleFileName.pickle if input isn't None
    
    Inputs
    ----------
    behav_corr : dictionary of dictionaries
                 First key: dataset
                 Second and third keys: behaviors A, B
                 Third key: "allDeltaFrames" for behavior A
                 Fourth key: "deltaFrames", the frame delays between A, B 
                             for all events.
    behavior_key_list : list of all behaviors
    binWidthFrames : width of bins for histogram, number of frames
        Default 25, usual 25 fps, so this is 1 second
    halfFrameRange : max possible DeltaFrames to bin; make bins from
        -halfFrameRange to +halfFrameRange, forcing a bin centered at 0.
        Default: 15000, which is probablly the total number of frames
    outputPickleFileName : name of the Pickle file in which to save 
                            outputs. Will append ".pickle"
                            Note that pickle file will be very large!
                            Default: None (don't save) (recommended)
    
        
    Returns
    -------
    behav_corr : dictionary of dictionaries, updated
                 First key: dataset
                 Second and third keys: behaviors A, B
                 Third key: "allDeltaFrames" for behavior A
                 Fourth key (unchanged): "deltaFrames", the frame delays 
                             between A, B for all events.
                 Fourth key (new): : 
                     "counts" : counts in each bin for A, B at deltaFrames
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
        # Calculate frame delays and append to each deltaFrames list
        for behavior_A in behavior_key_list:
            for behavior_B in behavior_key_list:
                # Histogram counts for deltaFrames_AB
                behav_corr[j][behavior_A][behavior_B]["counts"] = \
                    np.histogram(behav_corr[j][behavior_A][behavior_B]["deltaFrames"], 
                                 bins=binEdges)[0] # [0] to only get the counts array                   

    if outputPickleFileName != None:
        list_for_pickle = [behav_corr, binCenters, behavior_key_list]
        outputPickleFileName = outputPickleFileName + '.pickle'
        print(f'\nWriting pickle file: {outputPickleFileName}\n')
        with open(outputPickleFileName, 'wb') as handle:
            pickle.dump(list_for_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return behav_corr, binCenters



def calcBehavCorrAllSets(behav_corr, behavior_key_list, binCenters):
    """
    Average the correlations (binned Delta Frames for a 
            given behavior, relative to all the behaviors) across datasets,
            normalizing the pooled counts (not normalizing per datset)
    
    Parameters
    ----------
    behav_corr : nested dictionary, output by calcDeltaFramesEvents()
                 and then bin_deltaFrames()
    behavior_key_list : list of all behaviors, used in calcDeltaFramesEvents()
    binCenters : bin centers

    Returns
    -------
    behav_corr_allSets : Likelihood of behaviors preceding / following
                         other behaviors. Three initial keys:
                        behav_corr_allSets['normSimple']
                        behav_corr_allSets['normAcrossBehavior']
                        behav_corr_allSets['normAcrossTime']
                        For each next two keys are behaviors A, B; 
                        contains a numpy array of the relative likelihood
                        of event B following event A in each of the frame 
                        delay bins with center BinCenters

    """
    
    # Number of datasets
    N_datasets = len(behav_corr)
    # Number of bins
    Nbins = len(binCenters)

    behav_corr_acrossBehavior_array = np.zeros((N_datasets, 
                                                len(behavior_key_list),
                                                len(behavior_key_list), 
                                                Nbins), dtype=float)
    behav_corr_acrossTime_array = np.zeros((N_datasets, 
                                                len(behavior_key_list),
                                                len(behavior_key_list), 
                                                Nbins), dtype=float)
    for j in range(N_datasets):
        for ibA, bA in enumerate(behavior_key_list):
            for ibB, bB in enumerate(behavior_key_list):
                behav_corr_acrossBehavior_array[j, ibA, ibB, :] = \
                    behav_corr[j][bA][bB]["counts"]
                behav_corr_acrossTime_array[j, ibA, ibB, :] = \
                    behav_corr[j][bA][bB]["counts"]
    
    # Pool across all datasets
    behav_corr_array_sum = np.sum(behav_corr_acrossBehavior_array, axis=0)
    # Sum over behavior B (for each frame delay)
    behav_corr_sum_behaviorB = np.sum(behav_corr_array_sum, axis=1)
    # Sum over time (for each behavior B)
    behav_corr_sum_time = np.sum(behav_corr_array_sum, axis=2)
    # for normalizing by overall time (frames) considered
    frameRange = np.max(binCenters)- np.min(binCenters)

    # Put this into a nested dictionary
    # initialize nested dictionaries for datasets
    # There's probably a better way to do this...
    behav_corr_allSets = {'normSimple' : {}, 
                          'normAcrossBehavior' : {}, 'normAcrossTime': {}}
    for ibA, bA in enumerate(behavior_key_list):
        behav_corr_allSets['normSimple'][bA] = {}
        behav_corr_allSets['normAcrossBehavior'][bA] = {}
        behav_corr_allSets['normAcrossTime'][bA] = {}
        for ibB, bB in enumerate(behavior_key_list):
            # print(ibA, bA, ibB, bB)
            behav_corr_allSets['normSimple'][bA][bB] = \
                behav_corr_array_sum[ibA, ibB, :] / (N_datasets * frameRange)
            behav_corr_allSets['normAcrossBehavior'][bA][bB] = \
                behav_corr_array_sum[ibA, ibB, :] / \
                behav_corr_sum_behaviorB[ibA, :]
            behav_corr_allSets['normAcrossTime'][bA][bB] = \
                behav_corr_array_sum[ibA, ibB, :] / \
                behav_corr_sum_time[ibA,ibB]
     
    return behav_corr_allSets


def calc_corr_asymm(behav_corr_allSets, behavior_key_list, binCenters, 
                    maxFrameDelay = None):
    """
    Calculate the temporal asymmetry in the correlation function between 
    each pair of behaviors
    Note that different correlations only differ in normalization, so doesn't
    matter what we use. Will use 'normSimple'.

    Parameters
    ----------
    behav_corr_allSets : dictionary of each type of correlation for each 
        behavior pair. 
        E.g. for simple probability, 
            behav_corr_allSets['normSimple'][behaviorA][ behaviorB] 
            is a numpy array of probabilities at each binCenters
        
    behavior_key_list : list of all behaviors, used in calcDeltaFramesEvents()

    binCenters : numpy array
        center of each Delta t bin.
        
    maxFrameDelay : the abs. value max frame delay to consider ; Default None: 
        consider the full range over which correlations were calculated

    Returns
    -------
    corr_asymm : dictionary of correlation asymmetry for each behavior pair

    """
    # Calculate the asymmetries. Use a nested dictionary
    # initialize nested dictionaries for datasets
    # There's probably a better way to do this...
    if maxFrameDelay is None:
        maxFrameDelay = np.max(binCenters) + 0.001 # add a slight offset
    corr_asymm = {}
    for ibA, bA in enumerate(behavior_key_list):
        corr_asymm[bA] = {}
        thisProbA = behav_corr_allSets['normSimple'][bA]
        for ibB, bB in enumerate(behavior_key_list):
            thisProbAplus = np.sum(thisProbA[bB][(binCenters>0) & 
                                                 (binCenters <= maxFrameDelay)])
            thisProbAminus = np.sum(thisProbA[bB][(binCenters<0) &
                                                  (binCenters >= -1.0*maxFrameDelay)])
            if (thisProbAplus + thisProbAminus) > 0.0:
                corr_asymm[bA][bB] = (thisProbAplus - thisProbAminus) / \
                    (thisProbAplus + thisProbAminus)
            else:
                corr_asymm[bA][bB] = 0.0 # technically undefined, but I want zeros
            # print(corr_asymm[bA][bB])

    return corr_asymm



def plot_behaviorCorrelation(behav_corr_allSets, binCenters, 
                             behavior_key_list, behaviorA, behaviorB='',
                             fps = 1.0):
    """ 
    Plot Behavior B likelihood following/preceding Behavior A
    Can plot a single A-B pair, or all B for a given A
    If a single A-B pair, also include the mean value as a dashed line
    Plots all types of normalizations (simple, across Behaviors, and across Time)
    
    Inputs:
    behav_corr_allSets : Likelihood of behaviors preceding / following
                         other behaviors. From calcBehavCorrAllSets()
    binCenters : bin centers (frames), for plotting
    behaviorA , B: (string) Behavior A, B, to plot, if  just plotting 
                   one pair. Leave B empty ('') to skip plotting
                   a single A-B pair, and only plot all pairs
    behavior_key_list : list of behaviors to plot. (Can be a subset of all)
    fps : frames per second (probably 25, but default to 1)
    """

    # Just one AB pair    
    if not behaviorB=='':

        plt.figure(figsize=(6,5))
        plt.plot(binCenters/fps, behav_corr_allSets['normSimple'][behaviorA][behaviorB],
                 color='mediumvioletred')
        meanCorr = np.mean(behav_corr_allSets['normSimple'][behaviorA][behaviorB])
        plt.plot(binCenters/fps, meanCorr*np.ones(binCenters.shape), 
                 linestyle='dashed', color='orchid')
        plt.xlabel(r'$\Delta$t (s)', fontsize=20)
        plt.ylabel('Relative likelihood', fontsize=20)
        plt.title(f'{behaviorA} then {behaviorB}; simple norm.', fontsize=22)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.figure(figsize=(6,5))
        plt.plot(binCenters/fps, behav_corr_allSets['normAcrossBehavior'][behaviorA][behaviorB],
                 color='darkturquoise')
        meanCorr = np.mean(behav_corr_allSets['normAcrossBehavior'][behaviorA][behaviorB])
        plt.plot(binCenters/fps, meanCorr*np.ones(binCenters.shape), 
                 linestyle='dashed', color='turquoise')
        plt.xlabel(r'$\Delta$t (s)', fontsize=20)
        plt.ylabel('Relative likelihood', fontsize=20)
        plt.title(f'{behaviorA} then {behaviorB}; norm. across behaviors.', fontsize=22)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.figure(figsize=(6,5))
        plt.plot(binCenters/fps, behav_corr_allSets['normAcrossTime'][behaviorA][behaviorB],
                 color='darkorange')
        meanCorr = np.mean(behav_corr_allSets['normAcrossTime'][behaviorA][behaviorB])
        plt.plot(binCenters/fps, meanCorr*np.ones(binCenters.shape), 
                 linestyle='dashed', color='gold')
        plt.xlabel(r'$\Delta$t (s)', fontsize=20)
        plt.ylabel('Relative likelihood; norm. across time', fontsize=20)
        plt.title(f'{behaviorA} then {behaviorB}; norm. across time.', fontsize=22)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    
    # All pairs
    
    plt.figure(figsize=(8, 6))
    for bB in behavior_key_list:
        plt.plot(binCenters/fps, behav_corr_allSets['normSimple'][behaviorA][bB], 
                 label=bB, linewidth=2.0)
    plt.xlabel(r'$\Delta$t (s)', fontsize=16)
    plt.ylabel('Relative likelihood', fontsize=16)
    plt.title(f'{behaviorA} then each behavior; simple norm', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    plt.figure(figsize=(8, 6))
    for bB in behavior_key_list:
        plt.plot(binCenters/fps, behav_corr_allSets['normAcrossBehavior'][behaviorA][bB], 
                 label=bB, linewidth=2.0)
    plt.xlabel(r'$\Delta$t (s)', fontsize=16)
    plt.ylabel('Relative likelihood', fontsize=16)
    plt.title(f'{behaviorA} then each behavior; norm. across behaviors', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    plt.figure(figsize=(8, 6))
    for bB in behavior_key_list:
        plt.plot(binCenters/fps, behav_corr_allSets['normAcrossTime'][behaviorA][bB], 
                 label=bB, linewidth=2.0)
    plt.xlabel(r'$\Delta$t (s)', fontsize=16)
    plt.ylabel('Relative likelihood', fontsize=16)
    plt.title(f'{behaviorA} then each behavior; norm. across time', fontsize=18)
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


def plot_corr_asymm(corr_asymm, crange = None):
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
    if crange is None:
        vmin = None
        vmax = None
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
    plt.title('Correlation asymmetry', fontsize=18)
    plt.xlabel('Behavior', fontsize=16)
    plt.ylabel('Behavior', fontsize=16)

    # Show the plot
    plt.show()



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



def select_items_dialog(behavior_key_list, default_keys=['perp_noneSee', 
        'perp_oneSees', 'perp_bothSee', 'contact_any', 'contact_head_body', 
        'contact_inferred', 'tail_rubbing', 'Cbend_Fish0', 'Cbend_Fish1', 
        'Jbend_Fish0', 'Jbend_Fish1', 'Rbend_Fish0', 'Rbend_Fish1', 
        'isActive_any', 'isMoving_any', 'approaching_Fish0', 
        'approaching_Fish1', 'fleeing_Fish0', 'fleeing_Fish1']):

    """
    Creates a dialog with checkboxes for each item in behavior_key_list.
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
    root.geometry("400x500")
    
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