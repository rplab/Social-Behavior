# -*- coding: utf-8 -*-
# compare_experiment_behaviors.py
"""
Author:   Raghuveer Parthasarathy
Created on Fri Dec. 1, 2023
Last modified on May 6, 2024

Description
-----------

Code to read relative durations output for various datasets and plot
summary stats versus each other.

See the "Behavior analysis pipeline v2" document, 
section "Compare experiment sets", for description of the code and process.

Inputs:
    None.
    - manually modify main() to specify the path and file names corresponding
        to each experiment name
    - run this program
    
Outputs:
    - Makes, saves graphs

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_comparison(dataframes, mean_sem,  
                    dataLabels = ('Set1', 'Set2'), 
                    logPlot = False, addDiagonals = True,
                    showTextLabels = False, showLegend = True,
                    outputFileName = None):
    """
    Plot the mean and s.e.m. of behavior frequencies found in 
    datasets 1 and 2 versus each other; set 2 vs set 1
    Values previously calculated and extracted from appropriate CSV rows.

    Parameters
    ----------
    dataframes : tuple of dataframes 1 and 2
    mean_sem : tuple of tuple of mean and sem of each behavior for dataset 1, 2    file_path2 : path and file for dataset 2 CSV
    dataLabels : tuple of dataLabel1, 2 for labeling plots
    logPlot : if True, use log axes
    addDiagonals : if True, add a gray line at the diagonal
    showTextLabels : if true, text by symbols
    showLegend : if true, show a legend
    outputFileName : filename for figure output, if not None
    
    Returns
    -------
    None.

    """    
    
    df1, df2 = dataframes
    mean_sem_1, mean_sem_2 = mean_sem
    dataLabel_1, dataLabel_2 = dataLabels
    
    # Plotting
    plt.figure(figsize=(10, 8))
    for j in range(len(mean_sem_1[0])):
        plt.errorbar(mean_sem_1[0][j], mean_sem_2[0][j], 
                     xerr=mean_sem_1[1][j], 
                     yerr=mean_sem_2[1][j], fmt='o', 
                     capsize=5, markersize=12,
                     label = f"  {df1.columns[j + 1]}") #color='darkorange'
        
    # plt.errorbar(mean_sem_1[0], mean_sem_2[0], xerr=mean_sem_1[1], 
    #              yerr=mean_sem_2[1], fmt='o', 
    #              capsize=5, markersize=12, color='darkorange')

    if logPlot:
        plt.xscale('log')
        plt.yscale('log')

    # Make the axis limits the same, and equal to the min & max
    current_x_limits = plt.xlim()
    current_y_limits = plt.ylim()
    new_x_min = min(current_x_limits[0], current_y_limits[0])
    new_y_min = new_x_min
    new_x_max = max(current_x_limits[1], current_y_limits[1])
    new_y_max = new_x_max
    plt.xlim(new_x_min, new_x_max)
    plt.ylim(new_y_min, new_y_max)
    
    if addDiagonals:
        plt.plot((new_x_min, new_x_max), (new_y_min, new_y_max), 
                 color='gray', linewidth=2.0, linestyle='dotted')
        plt.plot((new_x_min, new_x_max), 0.1*np.array((new_y_min, new_y_max)), 
                 color='lightgray', linewidth=2.0, linestyle='dotted')

    if showLegend:
        plt.legend()
    
    if showTextLabels:
        # Add text annotations for each point
        for i, (x, y) in enumerate(zip(mean_sem_1[0], mean_sem_2[0])):
            plt.text(x, y, f"  {df1.columns[i + 1]}", fontsize=10, ha='left', 
                     va='bottom')  #, rotation=-45

    # Set labels and title
    plt.xlabel(dataLabel_1, fontsize=16)
    plt.ylabel(dataLabel_2, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Relative Durations of Behaviors", fontsize=18)

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    # Display the plot
    plt.show()
    
    if outputFileName != None:
        plt.savefig(outputFileName, bbox_inches='tight')


def scatter_plots_with_error_bars(dataframes, mean_sem, 
                                  dataLabels = ('Set1', 'Set2'),
                                  showLegend = True, outputFileName = None):
    """
    Plot the ratios of behavior frequencies (ratio of means
    with uncertainty) found in datasets 1 and 2 versus each other; 
    mean 1 / mean 2
    Use bootstrap for uncertainty, since sems are large, and simple
    error propagation is asymmetric.

    Parameters
    ----------
    dataframes : tuple of dataframes 1 and 2
    mean_sem : tuple of tuple of mean and sem of each behavior for dataset 1, 2    file_path2 : path and file for dataset 2 CSV
    dataLabels : tuple of dataLabel1, 2 for labeling plots
    showLegend : if true, show a legend
    outputFileName : filename for figure output, if not None
    
    Returns
    -------
    None.

    """    

    df1, df2 = dataframes
    mean_sem_1, mean_sem_2 = mean_sem
    dataLabel_1, dataLabel_2 = dataLabels

    # Calculate ratio and uncertainty
    ratios = mean_sem_1[0] / mean_sem_2[0]  # ratio of means
    
    #ratio_uncertainty = ratios * ((mean_sem_1[1] / mean_sem_1[0])**2 + 
    #                              (mean_sem_2[1] / mean_sem_2[0])**2)**0.5
    
    ratios, r_unc_lower, r_unc_upper = \
        ratio_with_sim_uncertainty(mean_sem_2[0], mean_sem_2[1],
                                         mean_sem_1[0], mean_sem_1[1])

    # Create scatter plots with error bars for each column
    plt.figure(figsize=(9, 7))
    for j in range(len(ratios)):
        plt.errorbar(j, ratios[j], 
                     yerr=np.vstack((r_unc_lower[j], r_unc_upper[j])), 
                     label = f"  {df1.columns[j + 1]}", fmt='o', capsize=5,
                     markersize=12)

    plt.xticks(range(len(ratios)), list(df1.columns[1:]), fontsize=14, 
               rotation=45, ha='right')
    # ax = plt.gca()
    # ax.set_xticklabels(list(df1.columns[1:]), fontsize=14, rotation=45)
    plt.yticks(fontsize=14)

    # Set labels and title
    # plt.xlabel("Behaviors", fontsize=16)
    plt.ylabel(f"Ratio: {dataLabel_1} / {dataLabel_2}", fontsize=16)
    plt.title("Ratios of Behavior Durations", fontsize=16)

    current_x_limits = plt.xlim()
    # dashed line at ratio = 1
    plt.plot(current_x_limits, (1.0, 1.0), 
             color='gray', linewidth=2.0, linestyle='dotted')

    # Display the plot
    if showLegend:
        plt.legend()
    plt.show()
    
    if outputFileName != None:
        plt.savefig(outputFileName, bbox_inches='tight')
    

def extract_mean_stats(df):
    # Input: pandas dataframe
    # Extract the row that contains the mean and s.e.m.,
    # and extract the mean and s.e.m.
    # Return mean of each behavior, s.e.m. of each behavior as a tuple
    #    of arrays

    # Extract relevant rows (mean, s.e.m.) by the first column
    mean_row= df[df.iloc[:, 0] == "Mean"]
    sem_row = df[df.iloc[:, 0] == "Std. Error of Mean"]

    # Extract values from DataFrames
    mean_val = mean_row.iloc[:, 1:].values.flatten().astype(float)
    sem_val = sem_row.iloc[:, 1:].values.flatten().astype(float)

    return mean_val, sem_val


def ratio_with_sim_uncertainty(x, sigx, y, sigy, n_samples=10000):
    """
    Calculate the ratio of y to x with asymmetric uncertainties 
       estimated from simulated normal distribution.
    x are the mean values for various behaviors, with s.e.m. sigx;
    y are the mean values for various behaviors, with s.e.m. sigy;
    To estimate uncertainty, make a Gaussian random distribution 
        and draw x, y pairs; calculate median (not mean, to avoid
        skew and negative numbers) and upper and lower 1 sigma
        percentiles of this.
    A bit silly -- should do a bootstrap on the original data that x 
    and y came from -- but this is a fine estimate.
    
    Parameters:
        x (array-like): Array of x values, one per behavior
        sigx (array-like): Array of s.e.m. of each x value.
        y (array-like): like x.
        sigy (array-like): like x.
        n_samples (int): Number of samples to generate. Default 10000.
    
    Returns:
        r_mean (float): Mean ratio of y to x for each behavior
        r_lower (float): Lower bound of the uncertainty in the ratio.
        r_upper (float): Upper bound of the uncertainty in the ratio.
    """
    
    N_behaviors = len(x)
    
    # Initialize array to store ratios
    ratios = np.zeros(N_behaviors)
    r_lower = np.zeros(N_behaviors)
    r_upper = np.zeros(N_behaviors)
    
    # Bootstrap resampling
    for j in range(N_behaviors):
        # random values
        x_sim = np.random.normal(loc=x[j], scale=sigx[j], size=n_samples)
        y_sim = np.random.normal(loc=y[j], scale=sigy[j], size=n_samples)
        r_sim = y_sim / x_sim
                
        # Calculate ratios for resampled data
        ratios[j] = np.median(r_sim)
        r_lower[j] = ratios[j] - np.percentile(r_sim, 16)
        r_upper[j] = np.percentile(r_sim, 84) - ratios[j]
        
    return ratios, r_lower, r_upper


#%%

if __name__ == '__main__':
    
    
    exptNameList = ['TwoWeekLightDark2023', 'CholicAcid_Jan2024', 
                    'Solitary_Cohoused_March2024', 
                    'Shank3_Feb2024', 'Infected_housing_March2024', 
                    'XGF_May2024', 'Infected_May2023', 'TwoWeekLight2023_TimeFlip']
    
    # Ask the user to indicate the experiment name, constrained 
    exptName = input("\n\nChoose a value for exptName (options: {}): ".format(', '.join(exptNameList)))
    # Check if the user's choice is in the list
    while exptName not in exptNameList:
        print("Invalid choice. Choose a value of exptName from the list.")
        exptName = input("Choose a value for exptName (options: {}): ".format(', '.join(exptNameList)))

    
    print('\n\nExperiment being plotted: ', exptName)

    # Specify the paths and file names
    baseDir = r"C:\Users\Raghu\Documents\Experiments and Projects\Misc\Zebrafish behavior\CSV files and outputs"

    if exptName == 'TwoWeekLightDark2023':
        print('\nTwo week old fish, light and dark 2023\n')
        path1 = baseDir + r"\2 week old - pairs"
        file1 = r"behavior_relDuration_2week_light_26Nov2023.csv"
        path2 = baseDir + r"\2 week old - pairs in the dark"
        file2 = r"behavior_relDuration_2week_dark_26Nov2023.csv"
        dataLabel1 = 'Zebrafish, in light'
        dataLabel2 = 'Zebrafish, in dark'

    if exptName == 'CholicAcid_Jan2024':
        print('\nCholic Acid Jan. 2024 \n')
        path1 = baseDir + r"\2 week old - pairs - cholic acid\Condition1"
        file1 = r"behavior_relDuration_condition1_31Jan2024.csv"
        path2 = baseDir + r"\2 week old - pairs - cholic acid\Condition2"
        file2 = r"behavior_relDuration_condition2_31Jan2024.csv"
        dataLabel1 = 'Fish, condition 1'
        dataLabel2 = 'Fish, condition 2'

    if exptName == 'Solitary_Cohoused_March2024':
        print('\nSolitary and Co-Housed, March 2024 \n')
        path1 = r"C:\Users\Raghu\Documents\Experiments and Projects\Misc\Zebrafish behavior\CSV files and outputs\TwoWeekOld_Solitary_CoHoused_1_3-2-2024\Condition1"
        file1 = r"behavior_relDuration_condition1_3March2024.csv"
        path2 = r"C:\Users\Raghu\Documents\Experiments and Projects\Misc\Zebrafish behavior\CSV files and outputs\TwoWeekOld_Solitary_CoHoused_1_3-2-2024\Condition2"
        file2 = r"behavior_relDuration_condition2_3March2024.csv"
        dataLabel1 = '(1) Co-housed'
        dataLabel2 = '(2) Solitary'

    if exptName == 'Shank3_Feb2024':
        print('\nShank3, Genotypes 1 and 2 2024\n')
        path1 = baseDir + r"\2 week old - pairs with shank3 mutations\Genotype 1"
        file1 = r"behavior_relDuration_G1.csv"
        #path2 = baseDir + r"\2 week old - pairs with shank3 mutations\Genotype 2"
        #file2 = r"behavior_relDuration_G2.csv"
        path2 = baseDir + r"\2 week old - pairs with shank3 mutations\Genotype 3"
        file2 = r"behavior_relDuration_G3.csv"
        dataLabel1 = 'Genotype 1 WT'
        # dataLabel2 = 'Genotype 2 Shank3a/b'
        dataLabel2 = 'Mixed Genotypes'

    if exptName == 'Infected_housing_March2024':
        print('\nVibrio infection, Solitary and Co-housed, March 2024\n')
        # Manually indicate particular subsets (input)
        subExptNameList = ['C1_G1 + C1_G2', 'C2_G1 + C2_G2']
        
        # Ask the user to indicate the sub-experiment name, constrained 
        subExptName = input("\n\nChoose a value for the sub-expt (options: {}): ".format(', '.join(subExptNameList)))
        # Check if the user's choice is in the list
        while subExptName not in subExptNameList:
            print("Invalid choice. Choose a value from the list.")
            subExptName = input("\n\nChoose a value for the sub-expt (options: {}): ".format(', '.join(exptNameList)))

        if subExptName == 'C1_G1 + C1_G2':
            path1 = baseDir + r"\2 week old - infected versus non-infected, solitary versus co-housed pairs" \
                + r"\C1_G1_nonInfect_coHoused"
            file1 = r"behavior_relDuration_c1g1.csv"
            path2 = baseDir + r"\2 week old - infected versus non-infected, solitary versus co-housed pairs" \
                + r"\C1_G2_nonInfect_solitary"
            file2 = r"behavior_relDuration_c1g2.csv"
            dataLabel1 = 'C1_G1_nonInfected_coHoused'
            dataLabel2 = 'C1_G2_nonInfected_solitary'

        if subExptName == 'C2_G1 + C2_G2':
            path1 = baseDir + r"\2 week old - infected versus non-infected, solitary versus co-housed pairs" \
                + r"\C2_G1_Infected_coHoused"
            file1 = r"behavior_relDuration_c2g1.csv"
            path2 = baseDir + r"\2 week old - infected versus non-infected, solitary versus co-housed pairs" \
                + r"\C2_G2_Infected_solitary"
            file2 = r"behavior_relDuration_c2g2.csv"
            dataLabel1 = 'C2_G1_Infected_coHoused'
            dataLabel2 = 'C2_G2_Infected_solitary'

    if exptName == 'XGF_May2024':
        print('\nCVZ and XGF, May 2024\n')
        path1 = baseDir + r"\2 week old - conventionalized versus ex-germ-free fish\XGF_filteredCSVs_1"
        file1 = r"behavior_relDuration_g1.csv"
        path2 = baseDir + r"\2 week old - conventionalized versus ex-germ-free fish\XGF_filteredCSVs_2"
        file2 = r"behavior_relDuration_g2.csv"
        dataLabel1 = 'CVZ'
        dataLabel2 = 'XGF'
        
    if exptName == 'Infected_May2023':
        print('\nVibrio Infected and non-Infected\n')
        path1 = baseDir + r"\2 week old - infected versus non-infected pairs\infected_nonInfected_1"
        file1 = r"behavior_relDuration_g1.csv"
        path2 = baseDir + r"\2 week old - infected versus non-infected pairs\infected_nonInfected_2"
        file2 = r"behavior_relDuration_g2.csv"
        # path2 = baseDir + r"\2 week old - infected versus non-infected pairs\infected_nonInfected_3"
        # file2 = r"behavior_relDuration_g3.csv"
        dataLabel1 = 'Non-Infected'
        dataLabel2 = 'Infected WT Vibrio'
        # dataLabel2 = 'Infected dACD Vibrio'
        
        
    if exptName == 'TwoWeekLight2023_TimeFlip':
        print('\nTwoWeekLight 2023, and Fish1 Time Flipped\n')
        path1 = baseDir + r"\2 week old - pairs"
        file1 = r"behavior_relDuration_2week_light_26Nov2023.csv"
        path2 = baseDir + r"\2 week old - Fish1 FLIPPED in TIME"
        file2 = r"behavior_relDuration_TimeFlipped.csv"
        dataLabel1 = 'In light'
        dataLabel2 = 'In light; *F1 time flipped*'
        
    file_path1 = os.path.join(path1, file1)
    file_path2 = os.path.join(path2, file2)

    # Read CSV files into Pandas DataFrames
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    print('\nColumns in dataset 1:')
    print(df1.columns.tolist())

    # Specify columns to exclude
    exclude_columns = ["Mean difference in fish lengths (px)", 
                       "Mean Inter-fish dist (px)", "Angle XCorr mean"]
    
    # exclude more columns
    exclude_more = False
    if exclude_more:
        exclude_columns.extend(['Jbend Fish0', 'Jbend Fish1', 
                                'Fish0 Flees', 'Fish1 Flees',
                                'Fish0 Approaches', 'Fish1 Approaches'])
    exclude_more2 = True
    if exclude_more2:
        exclude_columns.extend(['90deg-largerSees', '90deg-smallerSees',
                                'Cbend Fish0', 'Cbend Fish1',
                                'Contact (Larger fish head-body)', 
                                'Contact (Smaller fish head-body)'])
    print('Columns excluded: ', exclude_columns)
    
    # Exclude specified columns
    if exclude_columns:
        df1 = df1.drop(columns=exclude_columns, errors='ignore')
        df2 = df2.drop(columns=exclude_columns, errors='ignore')

    mean_sem_1 = extract_mean_stats(df1)
    mean_sem_2 = extract_mean_stats(df2)
    
    # Call the function to plot the comparison
    plot_comparison((df1, df2), (mean_sem_1, mean_sem_2), 
                    (dataLabel1, dataLabel2),
                    logPlot = True, showTextLabels = False, 
                    showLegend = True,
                    outputFileName = exptName + '_relBehaviorPlot.eps')
    
    # Call the function to create scatter plots with error bars
    # Because uncertainties in mean values are large and asymmetric,
    # use bootstrap resampling (separate function)
    # Note that I'm plotting set 2/ set 1, to match "y / x" from the earlier 
    # graph
    scatter_plots_with_error_bars((df2, df1), (mean_sem_2, mean_sem_1),
                                  (dataLabel2, dataLabel1),
                                  showLegend = False,
                                  outputFileName = exptName + \
                                      '_relBehaviorRatios.eps')
