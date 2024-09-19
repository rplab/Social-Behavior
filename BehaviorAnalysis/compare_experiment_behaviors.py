# -*- coding: utf-8 -*-
# compare_experiment_behaviors.py
"""
Author:   Raghuveer Parthasarathy
Created on Fri Dec. 1, 2023
Last modified on September 19, 2024

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
    - Outputs a CSV file of statistics for experiments

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import stats
import csv

import tkinter as tk
from tkinter import filedialog

def get_two_filePaths_and_labels():
    """
    Calls select_excel_file(), get_data_label_and_higher_folder(),
    to get information for two "behavior_counts"
    Excel summary files, and a location for outputs
    Returns
        file_paths : tuple of two full input (Excel) file paths
        dataLabels : list of two labels for the two files
    """
    print('\n\nWARNING: ')
    print('    File dialog box may be hidden. Look! ')
    print('    Also: you may need to close figure windows.\n')
    file_path1 = select_excel_file(1)
    data_label1, higher_folder_path = \
        get_data_label_and_higher_folder(file_path1)
    file_path2 = select_excel_file(2, initial_dir=higher_folder_path)
    data_label2, higher_folder_path = \
        get_data_label_and_higher_folder(file_path2)
    file_paths = (file_path1, file_path2)
    dataLabels = [data_label1, data_label2]

    return file_paths, dataLabels

def select_excel_file(j=1, initial_dir=None):
    # Dialog box: Select Excel file no j; return file name and path
    # Returns file_path (full, including file name)
    
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
        return file_path
    else:
        return None



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
    # Check that both are the same
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




def plot_comparison(stats_both, exclude_from_loglog,  
                    dataLabels = ('Set1', 'Set2'), 
                    logPlot = False, addDiagonals = True,
                    showTextLabels = False, showLegend = True,
                    outputFileName = None):
    """
    Plot the mean and s.e.m. of behavior frequencies found in 
    datasets 1 and 2 versus each other; Plot set 2 vs set 1
    
    Parameters
    ----------
    stats_both : tuple of stats1, stats2, dictionaries containing 
                 statistics for each column
    exclude_from_loglog : list of columns (keys) to ignore in plot
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
    
    stats1, stats2 = stats_both
    
    # Filter out excluded columns
    stats1 = {k: v for k, v in stats1.items() if k not in exclude_from_loglog}
    stats2 = {k: v for k, v in stats2.items() if k not in exclude_from_loglog}
    
    # Extract means and SEMs
    columns = list(stats1.keys())
    means1 = [stats1[col]['mean'] for col in columns]
    sems1 = [stats1[col]['sem'] for col in columns]
    means2 = [stats2[col]['mean'] for col in columns]
    sems2 = [stats2[col]['sem'] for col in columns]

    dataLabel_1, dataLabel_2 = dataLabels
    
    # Plotting
    plt.figure(figsize=(10, 8))
    for j, col in enumerate(columns):
        plt.errorbar(means1[j], means2[j], 
                     xerr=sems1[j], 
                     yerr=sems2[j], fmt='o', 
                     capsize=5, markersize=12,
                     label = f"  {col}")

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
        for i, (x, y) in enumerate(zip(means1, means2)):
            plt.text(x, y, f"  {columns[i]}", fontsize=10, ha='left', 
                     va='bottom')

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
    
    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight')



def scatter_plots_with_error_bars(stats_both, exclude_from_ratio, 
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
    stats_both : tuple of stats1, stats2, dictionaries containing 
                 statistics for each column
    exclude_from_ratio : list of columns (keys) to ignore in plot
    dataLabels : tuple of dataLabel1, 2 for labeling plots
    showLegend : if true, show a legend
    outputFileName : filename for figure output, if not None
    
    Returns
    -------
    None.
    """    

    stats1, stats2 = stats_both 

    # Filter out excluded columns
    stats1 = {k: v for k, v in stats1.items() if k not in exclude_from_ratio}
    stats2 = {k: v for k, v in stats2.items() if k not in exclude_from_ratio}
    
    # Extract means and SEMs
    columns = list(stats1.keys())
    means1 = [stats1[col]['mean'] for col in columns]
    sems1 = [stats1[col]['sem'] for col in columns]
    means2 = [stats2[col]['mean'] for col in columns]
    sems2 = [stats2[col]['sem'] for col in columns]

    dataLabel_1, dataLabel_2 = dataLabels

    # Calculate ratio and uncertainty
    ratios = np.array(means1) / np.array(means2)  # ratio of means
    
    r = ratio_with_sim_uncertainty(means2, sems2, means1, sems1)
    r_unc_lower = r[1]  # ignore r[0], re-sampled mean
    r_unc_upper = r[2]

    # Create scatter plots with error bars for each column
    plt.figure(figsize=(9, 7))
    for j, col in enumerate(columns):
        plt.errorbar(j, ratios[j], 
                     yerr=np.vstack((r_unc_lower[j], r_unc_upper[j])), 
                     label = f"  {col}", fmt='o', capsize=5,
                     markersize=12)

    plt.xticks(range(len(ratios)), columns, fontsize=14, 
               rotation=45, ha='right')
    plt.yticks(fontsize=14)

    # Set labels and title
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
    
    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight')
        

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
    # Ask user for the output path (for plots and CSV file)
    # If empty, dialog box
    
    outputPath = input('Enter the path (folder) for output, or leave blank to use a dialog box: ')

    if outputPath:
        return outputPath
    else:
        root = tk.Tk()
        root.withdraw()
        outputPath = filedialog.askdirectory(title="Select a folder")
        return outputPath

def keep_dataframe_to_blank_line(df):
    """
    keep only the 
    rows to the blank line (if there is one) in the dataframe. 
    (This means we will ignore the previously calculated statistics;
    will re-calculate)

    Parameters
    ----------
    df : dataframe read from Excel file

    Returns
    -------
    df : dataframe, modified

    """
    # Find the index of the first blank row
    blank_row_index = df.index[df.isnull().all(axis=1)].min()
    
    if pd.isna(blank_row_index):
        # If no blank row found, use all rows
        return df
    else:
        # Use only rows before the blank row
        return df.iloc[:blank_row_index]

def calc_stats_dataframes(df, exclude_columns):
    """
    Calculate statistics for columns in a single pandas dataframe.
    
    Parameters:
        df: processed dataframe
        exclude_columns: list of columns to ignore for analysis
    
    Returns:
        stats_dict: dictionary with keys corresponding to 'column_name', 'mean', 'N', 'std', and 'sem'
    """
    stats_dict = {}
    # Exclude 'Dataset' column, if not already excluded
    if 'Dataset' not in exclude_columns:
        exclude_columns.append('Dataset')
        
    for column in df.columns:
        if column not in exclude_columns:
            column_stats = {
                'column_name': column,
                'mean': df[column].mean(),
                'N': df[column].count(),
                'std': df[column].std(),
                'sem': stats.sem(df[column].dropna())
            }
            stats_dict[column] = column_stats
    
    return stats_dict

def stat_comparisons(df1, df2, exclude_columns):
    """
    Perform statistical comparisons between two pandas dataframes.
    
    Parameters:
        df1, df2: processed dataframes
        exclude_columns: list of columns to ignore for comparison
    
    Returns:
        stat_tests: dictionary with keys 'p_MWU', 'p_KS' corresponding to 
                    the mannwhitneyu and ks_2samp tests
    """
    stat_tests = {}
    
    for column in df1.columns:
        if column not in exclude_columns:
            # Perform Mann-Whitney U test
            _, p_mwu = stats.mannwhitneyu(df1[column].dropna(), df2[column].dropna())
            
            # Perform Kolmogorov-Smirnov test
            _, p_ks = stats.ks_2samp(df1[column].dropna(), df2[column].dropna())
            
            stat_tests[column] = {
                'p_MWU': p_mwu,
                'p_KS': p_ks
            }
    
    return stat_tests

def analyze_dataframes(df1, df2, exclude_columns):
    """ 
    Compare dataframes df1, df2. Consider each column and calculate statistics
    For each dataframe, statistical tests comparing the corresponding
    columns; Mann-Whitney U test (tests the median being different) and 
    Kolmogorov-Smirnov test (distributions being different)
    Returns all results, as a list
    
    Parameters:
        df2 and df2: processed dataframes (see preprocess_dataframe())
        exclude_columns: list of columns to ignore for comparison
    
    Returns:
        results : list of column name, mean1, std1, std1, std2, \
            sem1, sem2, p_mwu, p_ks
        headers : list of strings to use for CSV output headers.
            Write this here so that if we modify the stats, we don't 
            forget to modify the headers
    """
    # Preprocess dataframes; Keep only the rows to the blank line.
    df1 = keep_dataframe_to_blank_line(df1)
    df2 = keep_dataframe_to_blank_line(df2)
    
    # Exclude 'Dataset' column, if not already excluded
    if 'Dataset' not in exclude_columns:
        exclude_columns.append('Dataset')
    
    results = []
    
    for column in df1.columns:
        if column not in exclude_columns:
            # Calculate statistics for df1
            mean1 = df1[column].mean()
            n1 = df1[column].count()
            std1 = df1[column].std()
            sem1 = stats.sem(df1[column].dropna())
            
            # Calculate statistics for df2
            mean2 = df2[column].mean()
            n2 = df2[column].count()
            std2 = df2[column].std()
            sem2 = stats.sem(df2[column].dropna())
            
            # Perform Mann-Whitney U test
            _, p_mwu = stats.mannwhitneyu(df1[column].dropna(), df2[column].dropna())
            
            # Perform Kolmogorov-Smirnov test
            _, p_ks = stats.ks_2samp(df1[column].dropna(), df2[column].dropna())
            
            # Append results
            results.append([
                column, mean1, mean2, n1, n2, std1, std2, \
                    sem1, sem2, p_mwu, p_ks
            ])

    # Corresponding headers to use for CSV output
    headers = ['column_name', 'mean_df1', 'mean_df2', 'N_df1', 'N_df2', 
               'std_df1', 'std_df2', 'sem_df1', 'sem_df2', 'p_MWU', 'p_KS']    
    return results, headers



def write_results_to_csv(stats1, stats2, output_file, stat_tests=None):
    """ 
    Writes stats results from two calc_stats_dataframes() 
    outputs and optional stat_tests to a CSV file
    
    Parameters:
        stats1, stats2 : dictionaries with keys corresponding to column names, 
                         each containing a dictionary of statistics
        output_file : string, path to the output CSV file
        stat_tests : optional, dictionary with statistical test results
    """
    required_keys = {'column_name', 'mean', 'N', 'std', 'sem'}

    # Verify keys in stats1 and stats2
    for st in [stats1, stats2]:
        for column, column_stats in st.items():
            if set(column_stats.keys()) != required_keys:
                raise ValueError(f"Invalid keys in stats dictionary. Expected {required_keys}, "
                                 f"but got {set(column_stats.keys())} for column {column}")

    headers = ['column_name', 
               'mean_1', 'N_1', 'std_1', 'sem_1',
               'mean_2', 'N_2', 'std_2', 'sem_2']
    
    if stat_tests is not None:
        headers.extend(['p_MWU', 'p_KS'])
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for column in stats1.keys():
            if column in stats2:
                row = [
                    column,
                    '{:.3G}'.format(stats1[column]['mean']),
                    '{:.3G}'.format(stats2[column]['mean']),
                    str(stats1[column]['N']),
                    str(stats2[column]['N']),
                    '{:.3G}'.format(stats1[column]['std']),
                    '{:.3G}'.format(stats2[column]['std']),
                    '{:.3G}'.format(stats1[column]['sem']),
                    '{:.3G}'.format(stats2[column]['sem'])
                ]
                
                if stat_tests is not None and column in stat_tests:
                    row.extend([
                        '{:.3G}'.format(stat_tests[column]['p_MWU']),
                        '{:.3G}'.format(stat_tests[column]['p_KS'])
                    ])
                
                writer.writerow(row)


def load_and_combine_dataframes(outputStats = True, compareSets = False):
    """ 
    Combine datasets from two experiments. Asks user for files, etc.
    Loads Excel files and writes the concatenated dataset to a new Excel file.
    If outputStats == True, calculate the statistics for 
        each column of the concatenated dataset and write these also
        to the Excel file.
    Optional: if compareSets == True, perform statistical tests on the 
        two sets, which the user might use to evaluate similarity.
        Outputs a CSV file.
    
    Inputs:
        outputStats  : if true, calculate the statistics for 
                each column of the concatenated dataset
        compareSets : If true, calc stats on sets and run statistical
    
    Returns:
        df : dataframe
        writes Excel file, and (optional) CSV file
    
    """
    
    file_paths, dataLabels = get_two_filePaths_and_labels()
    
    # Read Relative Durations sheets into Pandas DataFrames
    df1 = read_behavior_Excel(file_paths[0])
    df2 = read_behavior_Excel(file_paths[1])

    # Verify column headings:
    verify_and_get_column_headings(df1, df2)
    
    # Remove statistics rows from dataframes -- will recalculate later
    # if needed
    print("Removing statistics rows from dataframes -- will recalculate")
    df1 = keep_dataframe_to_blank_line(df1)
    df2 = keep_dataframe_to_blank_line(df2)
    
    # Output location
    print('Enter the output folder for the combined dataset (Excel file)')
    outputPath = getOutputPath()
    
    if compareSets:
        print('Comparing the two datasets. (Output -> CSV)')
        exclude_from_all = []
        # Analyze (compare + stats)
        stats1 = calc_stats_dataframes(df1, exclude_from_all)
        stats2 = calc_stats_dataframes(df2, exclude_from_all)
        stat_tests = stat_comparisons(df1, df2, exclude_from_all)
        # Output CSV file name for statistical tests
        fileNameCSV_stats = \
            input('STAT TEST: Output CSV file name, including ".csv", e.g. "compare_stats.csv"): ')
        stats_output_file = os.path.join(outputPath, fileNameCSV_stats)
        write_results_to_csv(stats1, stats2, 
                             output_file=stats_output_file, 
                             stat_tests=stat_tests)

    df = pd.concat([df1, df2], axis=0)

    if outputStats:
        # Calculate stats for the concatenated dataframe
        stats_cat = calc_stats_dataframes(df, [])
        
        # Create a dataframe from the stats dictionary
        # Create a dataframe from the stats dictionary
        stats_df = pd.DataFrame({
            col: {
                'mean': stats_cat[col]['mean'],
                'N': stats_cat[col]['N'],
                'std': stats_cat[col]['std'],
                'sem': stats_cat[col]['sem']
            } for col in stats_cat
        })
        
        # Reorder the index to have 'mean', 'std', 'sem' , 'N' in this order
        stats_df = stats_df.reindex(['mean', 'std', 'sem', 'N'])
        
        # Reset the index to make the stat names a regular column
        stats_df = stats_df.reset_index().rename(columns={'index': 'Dataset'})

    # Output Excel file name
    inputstr = 'Output Excel file name; can omit ".xlsx"' + \
        '\n  Leave blank for "behavior_counts_combined.xlsx": '
    fileNameExcel = input(inputstr)
    if fileNameExcel=='':
        fileNameExcel = 'behavior_counts_combined.xlsx'
    root, ext = os.path.splitext(fileNameExcel)
    if ext != '.xlsx':
        fileNameExcel = fileNameExcel + '.xlsx'
    Excel_output_file = os.path.join(outputPath, fileNameExcel)
    

    # Create a pandas ExcelWriter object
    with pd.ExcelWriter(Excel_output_file, engine='openpyxl') as writer:
        # Write the concatenated dataframe to the first sheet
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        
        # Get the last row of the existing data
        last_row = len(df)
        
        if outputStats:    
            # Write the stats dataframe, starting two rows below the last data row
            stats_df.to_excel(writer, sheet_name='Sheet1', startrow=last_row+2, index=False, header=False)

    return df

    
#%%

if __name__ == '__main__':
        
    file_paths, dataLabels = get_two_filePaths_and_labels()
    
    # Read Relative Durations sheets into Pandas DataFrames
    df1 = read_behavior_Excel(file_paths[0])
    df2 = read_behavior_Excel(file_paths[1])

    # Verify column headings:
    column_headings = verify_and_get_column_headings(df1, df2)
    
    # Remove statistics rows from dataframes -- will recalculate later
    print("Removing statistics rows from dataframes -- will recalculate")
    df1 = keep_dataframe_to_blank_line(df1)
    df2 = keep_dataframe_to_blank_line(df2)
    
    # Output location
    outputPath = getOutputPath()
    
    # Output CSV file name
    fileNameCSV = input('Output CSV file name, including ".csv", e.g. "compare_stats.csv"): ')

    # Output plot "base" file name
    baseName0 = input('Base file name for comparison outputs.\n' + \
                       '    Include image extension (e.g. "exptGraphs.eps"): ')
    baseName, out_ext = os.path.splitext(baseName0)
    baseName = baseName.split('.')[0]

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
                                 'bad_bodyTrack_frames'])
        
    also_exclude_from_loglog = ['Mean difference in fish lengths (mm)',
                           'Mean head-head dist (mm)',
                           "AngleXCorr_mean"]
    exclude_from_loglog = exclude_from_all + also_exclude_from_loglog
    also_exclude_from_ratio = ['Mean difference in fish lengths (mm)',
                           'Mean head-head dist (mm)',
                           "AngleXCorr_mean"]    
    exclude_from_ratio = exclude_from_all + also_exclude_from_ratio
    
    # Analyze (compare + stats)
    stats1 = calc_stats_dataframes(df1, exclude_from_all)
    stats2 = calc_stats_dataframes(df2, exclude_from_all)
    stat_tests = stat_comparisons(df1, df2, exclude_from_all)
    stats_output_file = os.path.join(outputPath, fileNameCSV)
    write_results_to_csv(stats1, stats2, 
                         output_file=stats_output_file, stat_tests=stat_tests)
    
    #%% Plots
    
    # Call the function for log-log plot of the comparison
    new_baseName = baseName + '_relBehaviorLogLog' 
    outputFileName = os.path.join(outputPath, new_baseName + out_ext)

    plot_comparison((stats1, stats2), exclude_from_loglog, 
                    (dataLabels[0], dataLabels[1]),
                    logPlot = True, showTextLabels = False, 
                    showLegend = True,
                    outputFileName = outputFileName)
    
    # Call the function to create scatter plots with error bars
    # Because uncertainties in mean values are large and asymmetric,
    # use bootstrap resampling (separate function)
    # Note that I'm plotting stats of set 2/ set 1,
    # to match "y / x" from the earlier graph
    new_baseName = baseName + '_relBehaviorRatios' 
    outputFileName = os.path.join(outputPath, new_baseName + out_ext)
    scatter_plots_with_error_bars((stats2, stats1), exclude_from_ratio,
                                  (dataLabels[1], dataLabels[0]),
                                  showLegend = False,
                                  outputFileName = outputFileName)

