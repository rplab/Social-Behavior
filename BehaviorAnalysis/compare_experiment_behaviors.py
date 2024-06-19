# -*- coding: utf-8 -*-
# compare_experiment_behaviors.py
"""
Author:   Raghuveer Parthasarathy
Created on Fri Dec. 1, 2023
Last modified on June 17, 2024

Description
-----------

Code to read relative durations output for various datasets and plot
summary stats versus each other.

See the "Behavior analysis pipeline v2" document, 
section "Compare experiment sets", 
for description of the code and process.
Also "Behavior Code Revisions November 2023" document

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

import tkinter as tk
from tkinter import filedialog

def select_excel_file(j=1, initial_dir=None):
    # Dialog box: Select Excel file no j; return file name and path
    # Returns file_path, file_name
    
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', True)

    # Set the title for the file dialog
    title = f"Select File no. {j}"

    # Show the file dialog and get the selected file path
    file_path = filedialog.askopenfilename(
        title=title,
        initialdir=initial_dir,
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )

    # Check if a file was selected
    if file_path:
        # Get the file name from the path
        file_name = os.path.basename(file_path)
        return file_path, file_name
    else:
        return None, None



def get_data_label_and_higher_folder(file_path):
    # Determine these from folder names; user input if not readable
    # Code from Claude3
    
    # Normalize the path (this converts forward slashes to the OS-specific separator)
    normalized_path = os.path.normpath(file_path)
    
    # Split the path into components
    path_components = []
    while True:
        normalized_path, folder = os.path.split(normalized_path)
        if folder:
            path_components.append(folder)
        else:
            if normalized_path:
                path_components.append(normalized_path)
            break
    path_components.reverse()  # Reverse to get top-level folders first
    
    # Check if the lowermost folder is "Analysis" (case-insensitive)
    if len(path_components) >= 2 and path_components[-2].lower() == "analysis":
        data_label = path_components[-3] if len(path_components) >= 3 else None
        print(f"Extracted dataLabel: {data_label}")
    else:
        data_label = input("The lowermost folder is not 'Analysis'. Please enter a string for dataLabel: ")
    
    # Get the name of the folder two levels above "Analysis"
    if len(path_components) >= 5 and path_components[-2].lower() == "analysis":
        higher_folder_path_list = path_components[:-3]
        higher_folder_path = ''
        for f in higher_folder_path_list:
            higher_folder_path += f + os.sep
    else:
        higher_folder_path = None
        print("Warning: Could not determine higher_folder_path. The file structure might not be as expected.")
    
    return data_label, higher_folder_path

def getFilenames_and_Labels():

    exptNameList = ['TwoWeekLightDark2023', 'CholicAcid_Jan2024', 
                    'Solitary_Cohoused_March2024', 
                    'Shank3_Feb2024', 'Infected_housing_March2024', 
                    'XGF_May2024', 'Infected_May2023', 
                    'TwoWeekLight2023_TimeFlip']
    
    # Ask the user to indicate the experiment name, constrained 
    exptName = input("\n\nChoose a value for exptName (options: {}): ".format(', '.join(exptNameList)))
    # Check if the user's choice is in the list
    while exptName not in exptNameList:
        print("Invalid choice. Choose a value of exptName from the list.")
        exptName = input("Choose a value for exptName (options: {}): ".format(', '.join(exptNameList)))
    
    
    print('\n\nExperiment being plotted: ', exptName)
    
    # Specify the paths and file names
    baseDir = r"C:\Users\Raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs"
    
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
        path1 = r"C:\Users\Raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\TwoWeekOld_Solitary_CoHoused_1_3-2-2024\Condition1"
        file1 = r"behavior_relDuration_condition1_3March2024.csv"
        path2 = r"C:\Users\Raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\TwoWeekOld_Solitary_CoHoused_1_3-2-2024\Condition2"
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
        subGroupNameList = ['C1_G1 + C1_G2', 'C2_G1 + C2_G2']
        
        # Ask the user to indicate the sub-experiment name, constrained 
        subGroupName = input("\n\nChoose a value for the Group (options: {}): ".format(', '.join(subGroupNameList)))
        # Check if the user's choice is in the list
        while subGroupName not in subGroupNameList:
            print("Invalid choice. Choose a value from the list.")
            subGroupName = input("\n\nChoose a value for the Group (options: {}): ".format(', '.join(exptNameList)))
    
        if subGroupName == 'C1_G1 + C1_G2':
            path1 = baseDir + r"\2 week old - infected versus non-infected, solitary versus co-housed pairs" \
                + r"\C1_G1_nonInfect_coHoused"
            file1 = r"behavior_relDuration_c1g1.csv"
            path2 = baseDir + r"\2 week old - infected versus non-infected, solitary versus co-housed pairs" \
                + r"\C1_G2_nonInfect_solitary"
            file2 = r"behavior_relDuration_c1g2.csv"
            dataLabel1 = 'C1_G1_nonInfected_coHoused'
            dataLabel2 = 'C1_G2_nonInfected_solitary'
    
        if subGroupName == 'C2_G1 + C2_G2':
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
        
    file_paths = (os.path.join(path1, file1), os.path.join(path2, file2))
    dataLabels = (dataLabel1, dataLabel2)
    
    return file_paths, dataLabels


def read_behavior_Excel(file_path):
    """
    Reads an Excel file, loading the sheet called "Relative Durations" 
    into dataframe df:
    In addition, first checks that "Relative Durations" exists. 
    If it does not, gives an error and print the sheets 
    in the Excel file
    Code mostly from Claude3


    Parameters
    ----------
    file_path : file name and path

    Returns
    -------
    df : pandas dataframe
    
    """
    
    try:
        # Read the Excel file
        excel_file = pd.ExcelFile(file_path)
        
        # Check if "Relative Durations" sheet exists
        if "Relative Durations" in excel_file.sheet_names:
            # Load the "Relative Durations" sheet into df1
            df = excel_file.parse("Relative Durations")
        else:
            # If the sheet doesn't exist, raise an error
            raise ValueError(f"Sheet 'Relative Durations' not found. Available sheets: {excel_file.sheet_names}")
            
    except ValueError as e:
        print(f"Error: {e}")
        df = None  # Set df1 to None if the desired sheet is not found
    
    return df


def verify_and_get_column_headings(df1, df2):
    # Get column headings for both DataFrames
    # Code from Claude3
    headings1 = list(df1.columns)
    headings2 = list(df2.columns)

    # Check if the headings are identical
    if headings1 == headings2:
        return headings1
    else:
        # If headings don't match, find the differences
        diff1 = set(headings1) - set(headings2)
        diff2 = set(headings2) - set(headings1)
        
        error_message = "Column headings are not the same.\n"
        if diff1:
            error_message += f"Columns in df1 but not in df2: {diff1}\n"
        if diff2:
            error_message += f"Columns in df2 but not in df1: {diff2}"
        
        raise ValueError(error_message)


    
def plot_comparison(dataframes, exclude_from_loglog,  
                    dataLabels = ('Set1', 'Set2'), 
                    logPlot = False, addDiagonals = True,
                    showTextLabels = False, showLegend = True,
                    outputFileName = None):
    """
    Plot the mean and s.e.m. of behavior frequencies found in 
    datasets 1 and 2 versus each other; Plot set 2 vs set 1
    Values previously calculated; call extract_mean_stats() to 
    extract from appropriate dataframe (CSV) rows.

    Parameters
    ----------
    dataframes : tuple of dataframes 1 and 2
    exclude_from_loglog = list of columns (keys) to ignore in plot
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
    df1 = df1.drop(columns=exclude_from_loglog, errors='ignore')
    df2 = df2.drop(columns=exclude_from_ratio, errors='ignore')

    # tuple of mean and sem of each behavior
    mean_sem_1 = extract_mean_stats(df1)
    mean_sem_2 = extract_mean_stats(df2)

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


def scatter_plots_with_error_bars(dataframes, exclude_from_ratio, 
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
    exclude_from_loglog = list of columns (keys) to ignore in plot
    dataLabels : tuple of dataLabel1, 2 for labeling plots
    showLegend : if true, show a legend
    outputFileName : filename for figure output, if not None
    
    Returns
    -------
    None.

    """    

    df1, df2 = dataframes
    df1 = df1.drop(columns=exclude_from_ratio, errors='ignore')
    df2 = df2.drop(columns=exclude_from_ratio, errors='ignore')

    # tuple of mean and sem of each behavior
    mean_sem_1 = extract_mean_stats(df1)
    mean_sem_2 = extract_mean_stats(df2)

    dataLabel_1, dataLabel_2 = dataLabels

    # Calculate ratio and uncertainty
    ratios = mean_sem_1[0] / mean_sem_2[0]  # ratio of means
    
    #ratio_uncertainty = ratios * ((mean_sem_1[1] / mean_sem_1[0])**2 + 
    #                              (mean_sem_2[1] / mean_sem_2[0])**2)**0.5
    
    r = ratio_with_sim_uncertainty(mean_sem_2[0], mean_sem_2[1],
                                   mean_sem_1[0], mean_sem_1[1])
    r_unc_lower = r[1]  # ignore r[0], re-sampled mean
    r_unc_upper = r[2]

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
    

def extract_mean_stats(df, meanString = "Mean", 
                       semString = "Std. Error of Mean"):
    # Input: pandas dataframe
    # Extract the row that contains the mean and s.e.m.,
    # and extract the mean and s.e.m.
    # Return mean of each behavior, s.e.m. of each behavior as a tuple
    #    of arrays

    # Extract relevant rows (mean, s.e.m.) by the first column
    mean_row= df[df.iloc[:, 0] == meanString]
    sem_row = df[df.iloc[:, 0] == semString]

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


def getOutputPath():
    # Ask user for the output path (for plots)
    # If empty, dialog box
    
    outputPath = input('Enter the path (folder) for output, or leave blank to use a dialog box: ')

    if outputPath:
        return outputPath
    else:
        root = tk.Tk()
        root.withdraw()
        outputPath = filedialog.askdirectory(title="Select a folder")
        return outputPath

    
#%%

if __name__ == '__main__':
    
    # (file_paths, dataLabels) = getFilenames_and_Labels()
    
    
    # initial_directory = "C:/Users/YourUsername/Documents"  # Optional: Specify an initial directory
    
    print('\n\nWARNING: ')
    print('    File dialog box may be hidden. Look! ')
    print('    Also: you may need to close figure windows.\n')
    file_path1, file_name1 = select_excel_file(1)
    data_label1, higher_folder_path = \
        get_data_label_and_higher_folder(file_path1)
    file_path2, file_name2 = select_excel_file(2, initial_dir=higher_folder_path)
    data_label2, higher_folder_path = \
        get_data_label_and_higher_folder(file_path2)
    file_paths = (file_path1, file_path2)
    dataLabels = [data_label1, data_label2]

    # Output plot, and location
    outputPath = getOutputPath()
    baseName0 = input('Base file name for comparison outputs.\n' + \
                       '    Include image extension (e.g. "exptGraphs.eps"): ')
    baseName, out_ext = os.path.splitext(baseName0)
    baseName = baseName.split('.')[0]

    # Read Relative Durations sheets into Pandas DataFrames
    df1 = read_behavior_Excel(file_paths[0])
    df2 = read_behavior_Excel(file_paths[1])

    # Verify column heading:
    try:
        column_headings = verify_and_get_column_headings(df1, df2)
        print(f"\nColumn headings: {column_headings}")
    except ValueError as e:
        print(f"Error: {e}")

    # Specify columns to exclude for plots
    exclude_from_all = ['Frames per sec', 'Image scale (um/s)', 
                        'Total Time (s)',
                        'Mean difference in fish lengths (mm)']
    exclude_more_from_all = True
    if exclude_more_from_all:
        exclude_from_all.extend(['perp_larger_fish_sees',
                                 'perp_smaller_fish_sees',
                                 'contact_larger_fish_head',
                                 'contact_smaller_fish_head',
                                 'Cbend_Fish0', 'Cbend_Fish1',
                                 'bad_bodyTrack_frames'])
        
    also_exclude_from_loglog = ['Mean difference in fish lengths (mm)',
                           'Mean head-head dist (mm)',
                           "AngleXCorr_mean"]
    exclude_from_loglog = exclude_from_all + also_exclude_from_loglog
    also_exclude_from_ratio = ['Mean difference in fish lengths (mm)',
                           'Mean head-head dist (mm)',
                           "AngleXCorr_mean"]    
    exclude_from_ratio = exclude_from_all + also_exclude_from_ratio
    

    # Call the function for log-log plot of the comparison
    new_baseName = baseName + '_relBehaviorRatios' 
    outputFileName = os.path.join(outputPath, new_baseName + out_ext)

    plot_comparison((df1, df2), exclude_from_loglog, 
                    (dataLabels[0], dataLabels[1]),
                    logPlot = True, showTextLabels = False, 
                    showLegend = True,
                    outputFileName = outputFileName)
    
    # Call the function to create scatter plots with error bars
    # Because uncertainties in mean values are large and asymmetric,
    # use bootstrap resampling (separate function)
    # Note that I'm plotting set 2/ set 1, to match "y / x" from the earlier 
    # graph
    new_baseName = baseName + '_relBehaviorPlot' 
    outputFileName = os.path.join(outputPath, new_baseName + out_ext)
    scatter_plots_with_error_bars((df2, df1), exclude_from_ratio,
                                  (dataLabels[1], dataLabels[0]),
                                  showLegend = False,
                                  outputFileName = outputFileName)
