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
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from relative_duration import get_duration_info


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



def compare_relative_durations(CSVfilename1 = '', CSVfilename2 = '',):
    """
    Calls get_duration_info to read the CSV output by 
    calc_relative_durations_csv(), containing fish length differences, etc.
    for *two* different experiments.
    Plot the mean of each behavior for each experiment vs. the other.
    
    Ignore CSV columns for length, etc.; column names (keys) hard-coded
    in 'keysToIgnore'

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
                    'Angle XCorr mean']
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
                 label='Means with Error Bars')

    # Add labels and a legend
    plt.xlabel('Mean of Expt. 1', fontsize=18)
    plt.ylabel('Mean of Expt. 2', fontsize=18)
    plt.title('Relative Duration of Behaviors', fontsize=20)
    # plt.legend()

    # Display the key names next to data points
    for i, key in enumerate(key_list):
        plt.annotate(key, (mean_values1[i], mean_values2[i]), 
                     xytext=(5,5), textcoords='offset points')

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
        plt.scatter(length_diff, behavior_values, marker='o', color='b', label=behavior_to_plot)

        # Add the linear regression line to the plot
        plt.plot(np.array(length_diff), intercept + slope * np.array(length_diff), 
                 color='r', label='Linear Regression')
        
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
