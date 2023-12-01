# -*- coding: utf-8 -*-
# behavior_correlations.py
"""
Author:   Raghuveer Parthasarathy
Created on Wed Sep  6 13:38:21 2023
Last modified on Nov. 26, 2023

Description
-----------

Contains function(s) for calculating the correlation between different 
behavior events, and comparing relative durations of events across 
experiments.

Inputs:
    
Outputs:
    

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from relative_duration import get_duration_info


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
                 Third key: "deltaFrames", the frame delays between A, B 
                             for all events.
    behavior_key_list : list of all behaviors considered

    """

    behavior_key_list = ["perpendicular_noneSee", 
                        "perpendicular_oneSees", "perpendicular_bothSee", 
                        "perpendicular_larger_fish_sees", 
                        "perpendicular_smaller_fish_sees", 
                        "contact_any", "contact_head_body", 
                        "contact_larger_fish_head", "contact_smaller_fish_head", 
                        "contact_inferred", "tail_rubbing", 
                        "Cbend_Fish0", "Cbend_Fish1", 
                        "Jbend_Fish0", "Jbend_Fish1", 
                        "approaching_Fish0", "approaching_Fish1",
                        "fleeing_Fish0", "fleeing_Fish1",
                        ]
    
    # Number of datasets
    N_datasets = len(datasets)
    
    # initialize nested dictionaries for datasets
    # There's probably a better way to do this...
    behav_corr = [{} for j in range(N_datasets)]
    for j in range(N_datasets):
        # keep the dataset_name" key
        behav_corr[j]["dataset_name"] = datasets[j]["dataset_name"]
        print('Dataset ', j, '; ', behav_corr[j]["dataset_name"])
        for bA in behavior_key_list:
            behav_corr[j][bA] = {}
            behav_corr[j][bA]["allDeltaFrames"] = np.array([])
            for bB in behavior_key_list:
                behav_corr[j][bA][bB] = {}
                behav_corr[j][bA][bB]["deltaFrames"] = np.array([])
        # Calculate frame delays and append to each deltaFrames list
        for behavior_A in behavior_key_list:
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
                    # if behavior_A == "perpendicular_noneSee" and behavior_B == "contact_any": 
                    #     print('Diagnostics: ', behav_corr[j]["dataset_name"])
                    #     print(behavior_A)
                    #     print(behavior_B)
                    #     print('bA frame k = ', k, '  bB_frames = ', bB_frames)
                    #     print('Delta Frames: ', deltaFrames_temp.flatten())
                    behav_corr[j][behavior_A][behavior_B]["deltaFrames"] = \
                        np.append(behav_corr[j][behavior_A][behavior_B]["deltaFrames"], 
                                  deltaFrames_temp.flatten())
                    # All the frame delays (for all Behaviors B)
                    behav_corr[j][behavior_A]["allDeltaFrames"] = \
                        np.append(behav_corr[j][behavior_A]["allDeltaFrames"], 
                                  deltaFrames_temp.flatten())
    
    return behav_corr, behavior_key_list


def bin_deltaFrames(behav_corr, behavior_key_list, binWidthFrames = 25, 
                          halfFrameRange = 15000):
    """
    Bin the "deltaFrames" delays between events of behaviors A, B
    for each dataset.
    Force the central bin to be centered at deltaFrames = 0
    
    Inputs
    ----------
    behav_corr : dictionary of dictionaries
                 First key: dataset
                 Second and third keys: behaviors A, B
                 Third key: "deltaFrames", the frame delays between A, B 
                             for all events.
    behavior_key_list : list of all behaviors
    binWidthFrames : width of bins for histogram, number of frames
        Default 25, usual 25 fps, so this is 1 second
    halfFrameRange : max possible DeltaFrames to bin; make bins from
        -halfFrameRange to +halfFrameRange, forcing a bin centered at 0.
        Default: 15000, which is probablly the total number of frames
    
        
    Returns
    -------
    behav_corr : dictionary of dictionaries
                 First key: dataset
                 Second and third keys: behaviors A, B
                 Fourth: "counts_normAcrossBehavior" and 
                         "counts_normAcrossTime" , normalized likelihoods
    binCenters: bin centers, from bin edges used for histogram
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
        # Calculate frame delays and append to each deltaFrames list
        for behavior_A in behavior_key_list:
            for behavior_B in behavior_key_list:
                # Histogram counts for deltaFrames_AB
                behav_corr[j][behavior_A][behavior_B]["counts"] = \
                    np.histogram(behav_corr[j][behavior_A][behavior_B]["deltaFrames"], 
                                 bins=binEdges)[0] # [0] to just get the counts array
            # Histogram counts of all the behaviors' Frame Delays, rel. to A
            behav_corr[j][behavior_A]["counts_all"] = \
                np.histogram(behav_corr[j][behavior_A]["allDeltaFrames"], 
                                                       bins=binEdges)[0]
            # Normalize the deltaFrames_AB 
            # (1) by the total for all deltaFrames
            for behavior_B in behavior_key_list:
                behav_corr[j][behavior_A][behavior_B]["counts_normAcrossBehavior"] = \
                    behav_corr[j][behavior_A][behavior_B]["counts"] / \
                        behav_corr[j][behavior_A]["counts_all"]
                behav_corr[j][behavior_A][behavior_B]["counts_normAcrossTime"] = \
                    behav_corr[j][behavior_A][behavior_B]["counts"] / \
                        np.sum(behav_corr[j][behavior_A][behavior_B]["counts"])
    
    return behav_corr, binCenters
    
def calcBehavCorrAllSets(behav_corr, behavior_key_list, binCenters):
    """
    Average the normalized correlations (binned Delta Frames for a 
            given behavior, relative to all the behaviors) across datasets

    Parameters
    ----------
    behav_corr : nested dictionary, output by calcDeltaFramesEvents()
                 and then bin_deltaFrames()
    behavior_key_list : list of all behaviors, used in calcDeltaFramesEvents()
    binCenters : bin centers, for plotting

    Returns
    -------
    behav_corr_allSets : Likelihood of behaviors preceding / following
                         other behaviors. Two initial keys:
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
                    behav_corr[j][bA][bB]["counts_normAcrossBehavior"]
                behav_corr_acrossTime_array[j, ibA, ibB, :] = \
                    behav_corr[j][bA][bB]["counts_normAcrossTime"]

    # Average over all datasets
    behav_corr_acrossBehavior_array_mean = \
        np.nanmean(behav_corr_acrossBehavior_array, axis=0)
    behav_corr_acrossTime_array_mean = \
        np.nanmean(behav_corr_acrossTime_array, axis=0)

    # Put this into a nested dictionary
    # initialize nested dictionaries for datasets
    # There's probably a better way to do this...
    behav_corr_allSets = {'normAcrossBehavior' : {}, 'normAcrossTime': {}}
    for ibA, bA in enumerate(behavior_key_list):
        behav_corr_allSets['normAcrossBehavior'][bA] = {}
        behav_corr_allSets['normAcrossTime'][bA] = {}
        for ibB, bB in enumerate(behavior_key_list):
            behav_corr_allSets['normAcrossBehavior'][bA][bB] = \
                behav_corr_acrossBehavior_array_mean[ibA, ibB, :]
            behav_corr_allSets['normAcrossTime'][bA][bB] = \
                behav_corr_acrossTime_array_mean[ibA, ibB, :]

    return behav_corr_allSets


def plot_behaviorCorrelation(behav_corr_allSets, binCenters, 
                             behavior_key_list, behaviorA, behaviorB=''):
    """ 
    Plot Behavior B likelihood following/preceding Behavior A
    Can plot a single A-B pair, or all B for a given A
    Plots both types of normalizations (across Behaviors and across Time)
    
    Inputs:
    behav_corr_allSets : Likelihood of behaviors preceding / following
                         other behaviors. From calcBehavCorrAllSets()
    binCenters : bin centers, for plotting
    behaviorA , B: (string) Behavior A, B, to plot, if  just plotting 
                   one pair. Leave B empty ('') to skip plotting
                   a single A-B pair, and only plot all pairs
    behavior_key_list : list of behaviors to plot. (Can be a subset of all)
    """

    # Just one AB pair    
    if not behaviorB=='':
        plt.figure()
        plt.plot(binCenters, behav_corr_allSets['normAcrossBehavior'][behaviorA][behaviorB],
                 color='darkturquoise')
        plt.xlabel('Delta Frames')
        plt.ylabel('Relative likelihood')
        plt.title(f'{behaviorA} then {behaviorB}; norm. across behaviors.')
        
        plt.figure()
        plt.plot(binCenters, behav_corr_allSets['normAcrossTime'][behaviorA][behaviorB],
                 color='darkorange')
        plt.xlabel('Delta Frames')
        plt.ylabel('Relative likelihood; norm. across time')
        plt.title(f'{behaviorA} then {behaviorB}; norm. across time.')
    
    # All pairs
    plt.figure(figsize=(7, 6), dpi=100)
    for bB in behavior_key_list:
        plt.plot(binCenters, behav_corr_allSets['normAcrossBehavior'][behaviorA][bB], 
                 label=bB)
    plt.xlabel('Delta Frames')
    plt.ylabel('Relative likelihood')
    plt.title(f'{behaviorA} then each behavior; norm. across behaviors')
    plt.legend()

    plt.figure(figsize=(7, 6), dpi=100)
    for bB in behavior_key_list:
        plt.plot(binCenters, behav_corr_allSets['normAcrossTime'][behaviorA][bB], 
                 label=bB)
    plt.xlabel('Delta Frames')
    plt.ylabel('Relative likelihood')
    plt.title(f'{behaviorA} then each behavior; norm. across time')
    plt.legend()

def compare_relative_durations(CSVfilename1 = '', CSVfilename2 = '',):
    """
    Calls get_duration_info to read the CSV output by 
    calc_relative_durations_csv(), containing fish length differences, etc.
    for *two* different experiments.
    Plot the mean of each behavior for each experiment vs. the other.
    
    Ignore CSV columns for length, "smaller" or "larger" fish, etc.
        column names (keys) hard-coded in 'keysToIgnore'

    Inputs:
        CSVfilename1, CSVfilename2 : CSV file names containing 
            dataset names (col 1) 
            and relative durations. Reading stops at the first blank
            row, and so will ignore the mean, etc.
            Leave empty to list csv files in the current directory that 
            have 'relDuration' in the file name. If there's only one,
            use it as default.

    Outputs:
        
    Raghuveer Parthasarathy
    Sept. 17, 2023           
    Last modified Sept. 18, 2023
    """
    
    duration_data1 = get_duration_info(CSVfilename1)  # Will ask for CSV file info
    duration_data2 = get_duration_info(CSVfilename2)  # Will ask for CSV file info
    
    # Initialize lists to store statistics
    mean_values1 = []    # To store the means of df1
    mean_values2 = []    # To store the means of df2
    std_dev1 = []        # To store the standard deviations of df1
    std_dev2 = []        # To store the standard deviations of df2
    
    keysToIgnore = ['Mean difference in fish lengths (px)', 
                    'Mean Inter-fish dist (px)',
                    'Angle XCorr mean', '90deg-largerSees', 
                    '90deg-smallerSees', 'Contact (Larger fish head-body)',
                    'Contact (Smaller fish head-body)', 
                    'Contact (inferred)']
    key_list = list(set(duration_data1.columns) - set(keysToIgnore))

    for key in key_list:
        # Note that the keys must be the same in both dataframes -- not 
        # checking this!
        # Calculate mean and standard deviation for df1
        mean1 = duration_data1[key].mean()
        std1 = duration_data1[key].std()
        mean_values1.append(mean1)
        std_dev1.append(std1)

        # Calculate mean and standard deviation for df2
        mean2 = duration_data2[key].mean()
        std2 = duration_data2[key].std()
        mean_values2.append(mean2)
        std_dev2.append(std2)        
        
    # Calculate standard errors
    sem_values1 = [std / np.sqrt(len(duration_data1)) for std in std_dev1]
    sem_values2 = [std / np.sqrt(len(duration_data2)) for std in std_dev2]

    # Create a scatter plot of the means with error bars
    plt.figure()
    plt.errorbar(mean_values1, mean_values2, xerr=sem_values1, 
                 yerr=sem_values2, fmt='o', markersize=16, capsize=5, 
                 markeredgecolor = 'darkturquoise', 
                 markerfacecolor='paleturquoise',
                 label='Means with Error Bars')

    # Add labels and a legend
    plt.xlabel('Mean of Expt. 1', fontsize=18)
    plt.ylabel('Mean of Expt. 2', fontsize=18)
    plt.title('Relative Duration of Behaviors', fontsize=20)
    # plt.legend()

    # Display the key names next to data points
    for i, key in enumerate(key_list):
        plt.annotate(key, (mean_values1[i], mean_values2[i]), 
                     xytext=(-30,5), textcoords='offset points',
                     rotation=-45)

    # Show the plot
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
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
