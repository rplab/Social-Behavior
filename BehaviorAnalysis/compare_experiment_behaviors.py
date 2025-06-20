# -*- coding: utf-8 -*-
# compare_experiment_behaviors.py
"""
Author:   Raghuveer Parthasarathy
Created on Fri Dec. 1, 2023
Last modified on June 18, 2025

Description
-----------

Code to read relative durations output for various datasets and plot
summary stats versus each other.

See the "Behavior analysis pipeline v2" document, 
section "Compare experiment sets", 
for description of the code and process.
Also "Behavior Code Revisions November 2023" document

Inputs:
    None; run this program
    
Outputs:
    - Makes, saves graphs
    - Outputs a Excel file of statistics for experiments

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas as pd
from scipy import stats
import tkinter as tk
from tkinter import filedialog

from toolkit import select_items_dialog

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
    
    # Check if the folder name ends with "_Analysis" (case-insensitive)
    if len(path_components) >= 2 and path_components[-2].lower().endswith("_analysis"):
        data_label = path_components[-2][:-9]  # Remove "_Analysis" from the folder name
        print(f"Extracted dataLabel: {data_label}")
    else:
        data_label = input("The folder name does not end with '_Analysis'. Please enter a string for dataLabel: ")
        
    # Get the name of the folder two levels above "Analysis"
    if len(path_components) >= 5 and path_components[-2].lower().endswith("_analysis"):
        higher_folder_path_list = path_components[:-3]
        higher_folder_path = ''
        for f in higher_folder_path_list:
            higher_folder_path += f + os.sep
    else:
        higher_folder_path = None
        print("Warning: Could not determine higher_folder_path. The file structure might not be as expected.")
    
    return data_label, higher_folder_path

def read_excel_sheets(file_path):
    """
    Reads all sheets from an Excel file into a dictionary of dataframes.
    
    Parameters
    ----------
    file_path : str
        Path to the Excel file
        
    Returns
    -------
    dict
        Dictionary with sheet names as keys and dataframes as values
    """
    try:
        # Read all sheets into a dictionary of dataframes
        excel_file = pd.ExcelFile(file_path)
        dfs = {}
        for sheet_name in excel_file.sheet_names:
            dfs[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name)
        return dfs
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def get_common_sheets(file_path1, file_path2):
    """
    Get list of sheet names common to both Excel files.
    
    Parameters
    ----------
    file_path1, file_path2 : str
        Paths to the Excel files
        
    Returns
    -------
    list
        List of common sheet names
    """
    excel1 = pd.ExcelFile(file_path1)
    excel2 = pd.ExcelFile(file_path2)
    
    sheets1 = excel1.sheet_names
    sheets2 = excel2.sheet_names
    
    common_sheets = [x for x in sheets1 if x in sheets2] 
    # avoiding set for sheets1, 2 and intersection, to preserve order
    #   list(sheets1.intersection(sheets2))
    print(f"\nCommon sheets found: {common_sheets}")
    return common_sheets


def select_sheets(common_sheets):
    """
    Ask user for the list of sheet names to use.
    
    Parameters
    ----------
    common_sheets: List of common sheet names
        
    Returns
    -------
    list
        sheets_to_use: List of sheet names to use
    """
    # Print the numbered list of sheets
    for i, sheet in enumerate(common_sheets, 1):
        print(f"{i}. {sheet}")
    
    # Ask the user which sheets to use, default is all
    user_input = input("Enter the numbers of the sheets to use (comma-separated), or press Enter to use all: ")
    
    # If user presses Enter, use all sheets
    if not user_input:
        return common_sheets
    
    # Convert user input to a list of integers
    selected_indices = [int(x) for x in user_input.split(",")]
    
    # Get the selected sheets based on indices
    sheets_to_use = [common_sheets[i-1] for i in selected_indices]
    
    return sheets_to_use




def write_results_to_excel(stats1, stats2, output_file, 
                           sheet_name, stat_tests, column_ratios):
    """
    Writes stats results to a sheet in an Excel file.
    Similar to write_results_to_csv but writes to Excel.
    
    Parameters
    ----------
    stats1, stats2 : dict
        Dictionaries with statistics for each dataset column
    output_file : str
        Path to output Excel file
    sheet_name : str
        Name of the sheet to write to
    column_ratios: list of dictionaries with ratios of set2/set1 values 
                    and uncertainties; (make None to ignore)
    stat_tests : dict, (make None to ignore)
        Dictionary with statistical test results
    """
    required_keys = {'column_name', 'mean', 'N', 'std', 'sem'}

    # Verify keys in stats1 and stats2
    for st in [stats1, stats2]:
        for column, column_stats in st.items():
            if set(column_stats.keys()) != required_keys:
                raise ValueError(f"Invalid keys in stats dictionary for {column}")

    # Create lists for DataFrame
    data = []
    headers = ['column_name', 
               'mean_1', 'N_1', 'std_1', 'sem_1',
               'mean_2', 'N_2', 'std_2', 'sem_2']

    if column_ratios is not None:
        headers.extend(['ratio', 'ratio_unc', 'ratio_uncLower', 'ratio_uncUpper'])
    
    if stat_tests is not None:
        headers.extend(['p_MWU', 'p_KS'])
            
    for column in stats1.keys():
        if column in stats2:
            row = [
                column,
                stats1[column]['mean'],
                stats1[column]['N'],
                stats1[column]['std'],
                stats1[column]['sem'],
                stats2[column]['mean'],
                stats2[column]['N'],
                stats2[column]['std'],
                stats2[column]['sem']
            ]
            if column_ratios is not None:
                if column in column_ratios:
                    row.extend([
                        column_ratios[column]['mean'],
                        column_ratios[column]['uncertainty'],
                        column_ratios[column]['unc_lower'],
                        column_ratios[column]['unc_upper']
                    ])
                else:
                    row.extend([' ',' ',' ',' '])
            if stat_tests is not None and column in stat_tests:
                row.extend([
                    stat_tests[column]['p_MWU'],
                    stat_tests[column]['p_KS']
                ])
            
            data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    # Write to Excel
    with pd.ExcelWriter(output_file, engine="openpyxl", 
                        mode='a' if os.path.exists(output_file) else 'w',
                        if_sheet_exists="replace" if os.path.exists(output_file) else None) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        
def plot_comparison(stats_both, df_both, include_columns, colors,
                    dataLabels = ('Set1', 'Set2'),
                    logPlot = True, addDiagonals = True,
                    showTextLabels = False, showLegend = True,
                    titleString = "Behavior Comparison",
                    outputFileName = None):
    """
    Plot the mean and s.e.m. of behavior frequencies found in 
    datasets 1 and 2 versus each other; Plot set 2 vs set 1.
    Also plots individual datapoints from both datasets as open circles
    in matching colors.
    
    Parameters
    ----------
    stats_both : tuple of stats1, stats2, dictionaries containing 
                 statistics for each column
    df_both : tuple of df1, df2, the original dataframes
    include_columns : list of columns (keys) to include in plot
    dataLabels : tuple of dataLabel1, 2 for labeling plots
    colors : N x 4, colors to use for plots
    logPlot : if True, use log axes
    addDiagonals : if True, add a gray line at the diagonal
    showTextLabels : if true, text by symbols
    showLegend : if true, show a legend
    titleString : String for plot title
    outputFileName : filename for figure output, if not None
    
    Returns
    -------
    None.
    """    
    
    stats1, stats2 = stats_both
    df1, df2 = df_both
    
    # Filter out excluded columns
    stats1 = {k: v for k, v in stats1.items() if k in include_columns}
    stats2 = {k: v for k, v in stats2.items() if k in include_columns}
    
    # Extract means and SEMs
    columns = list(stats1.keys())
    means1 = [stats1[col]['mean'] for col in columns]
    sems1 = [stats1[col]['sem'] for col in columns]
    means2 = [stats2[col]['mean'] for col in columns]
    sems2 = [stats2[col]['sem'] for col in columns]

    dataLabel_1, dataLabel_2 = dataLabels
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    
    # Get the default color cycle
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    
    # Plot means with error bars and individual points for each column
    for j, col in enumerate(columns):
        
        # Get individual points from each dataset
        points1 = df1[col].dropna().values
        points2 = df2[col].dropna().values
        
        # Plot horizontal spread at mean2
        plt.scatter(points1, np.full_like(points1, means2[j]), 
                   alpha=0.3, s=60, color=colors[j,:], 
                   facecolors='none', edgecolors=colors[j,:])
        # Plot vertical spread at mean1
        plt.scatter(np.full_like(points2, means1[j]), points2, 
                   alpha=0.3, s=60, color=colors[j,:], 
                   facecolors='none', edgecolors=colors[j,:])
        
        # Plot mean with error bars
        plt.errorbar(means1[j], means2[j], 
                     xerr=sems1[j], 
                     yerr=sems2[j], fmt='o', 
                     capsize=5, markersize=16,
                     color=colors[j,:], ecolor = colors[j,:], 
                     label=f"  {col}")

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
    plt.title(titleString, fontsize=18)

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
        
    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight')

    # Display the plot
    plt.show()


def plot_cols_side_by_side(stats_both, df_both, include_columns, colors, 
                          offset_x=0.1, dataLabels=('Set1', 'Set2'), 
                          titleString = 'Behaviors, side-by-side',
                          outputFileName=None, logYscale = True):
    """
    Plot the data points and statistics from two datasets side by side for each column.
    For each column, plots scatter points and mean±sem, with dataset 1 slightly left 
    and dataset 2 slightly right of the column position.
    
    Parameters:
    -----------
    stats_both : tuple of dicts
        (stats1, stats2) where each contains column statistics with 'mean' and 'sem'
    df_both : tuple of pandas.DataFrame
        (df1, df2) containing the raw data for each dataset
    include_columns : list
        Column names to include in plot
    offset_x : float, optional
        Horizontal offset for separating the two datasets (default: 0.25)
    dataLabels : tuple of str, optional
        Labels for the two datasets (default: ('Set1', 'Set2'))
    titleString : plot title
    outputFileName : str, optional
        If provided, save the plot to this path
    """
    stats1, stats2 = stats_both
    df1, df2 = df_both
    
    # Filter out excluded columns
    stats1 = {k: v for k, v in stats1.items() if k in include_columns}
    stats2 = {k: v for k, v in stats2.items() if k in include_columns}
    
    # Extract columns and prepare figure
    columns = list(stats1.keys())
    plt.figure(figsize=(12, 6))
    
    # Plot each column
    for j, col in enumerate(columns):
         
        # Get data points and statistics
        points1 = df1[col].dropna().values
        points2 = df2[col].dropna().values
        mean1, sem1 = stats1[col]['mean'], stats1[col]['sem']
        mean2, sem2 = stats2[col]['mean'], stats2[col]['sem']
        
        # Plot dataset 1 (left side)
        plt.scatter([j-offset_x] * len(points1), points1, 
                   marker='x', color=colors[j,:], alpha=0.5)
        plt.errorbar(j-offset_x, mean1, yerr=sem1, 
                    marker='<', color=colors[j,:], markersize=10,
                    capsize=5)
        
        # Plot dataset 2 (right side)
        plt.scatter([j+offset_x] * len(points2), points2, 
                   marker='x', color=colors[j,:], alpha=0.5)
        plt.errorbar(j+offset_x, mean2, yerr=sem2, 
                    marker='>', color=colors[j,:], markersize=10,
                    capsize=5)
    
    # Customize the plot
    plt.xticks(range(len(columns)), columns, rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    custom_lines = [
        plt.Line2D([0], [0], marker='<', color='gray', linestyle='none',
                  markersize=10, label=dataLabels[0]),
        plt.Line2D([0], [0], marker='>', color='gray', linestyle='none',
                  markersize=10, label=dataLabels[1])
    ]
    plt.legend(handles=custom_lines)
    
    # Set labels and title
    plt.xlabel('Behaviors')
    plt.ylabel('Duration or Number')
    plt.title(titleString)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # logarithmic y scale
    if logYscale:
        plt.yscale('log')
        
    # Save if filename provided
    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight')
    
    plt.show()
    

def plot_behavior_ratio(column_ratios, colors, 
                        dataLabels = ('Set1', 'Set2'),
                        showLegend = True, titleSheet = '', 
                        ylim = None,
                        outputFileName = None):
    """
    Plot the ratios of behavior frequencies (ratio of means
    with uncertainty) found in datasets 1 and 2 versus each other;
    Ratios and uncertainties previously calculated, calculate_column_ratios()
    mean 2 / mean 1

    Parameters
    ----------
    column_ratios : dictionary of df2/df2 column ratios and uncertainties
    colors : array of colors for plot
    dataLabels : tuple of dataLabel1, 2 for labeling plots
    showLegend : if true, show a legend
    titleSheet : append to title of plot
    ylim : (optional) tuple of min, max y-axis limits
    outputFileName : filename for figure output, if not None
    
    Returns
    -------
    None.
    """    

    dataLabel_1, dataLabel_2 = dataLabels
    plt.figure()
    
    # Create scatter plots with error bars for each column
    for j, col in enumerate(column_ratios.keys()):
        plt.errorbar(j, column_ratios[col]['mean'], 
                     yerr=np.vstack((column_ratios[col]['unc_lower'], 
                                     column_ratios[col]['unc_lower'])), 
                     label = f"  {col}", fmt='o', capsize=5,
                     markerfacecolor = colors[j,:], 
                     markeredgecolor = colors[j,:], 
                     ecolor = colors[j,:], 
                     markersize=14)

    column_names = column_ratios.keys()
    if ylim is not None:
        plt.ylim(ylim)
    plt.xticks(range(len(column_ratios)), column_names, fontsize=14, 
               rotation=45, ha='right')
    plt.yticks(fontsize=14)

    # Set labels and title
    plt.ylabel(f"Ratio: {dataLabel_2} / {dataLabel_1}", fontsize=16)
    plt.title(f"{titleSheet} Ratios of Behaviors", fontsize=16)

    current_x_limits = plt.xlim()
    # dashed line at ratio = 1
    plt.plot(current_x_limits, (1.0, 1.0), 
             color='gray', linewidth=2.0, linestyle='dotted')

    if showLegend:
        plt.legend()
    
    if outputFileName is not None:
        plt.savefig(outputFileName, bbox_inches='tight')

    # Display the plot
    plt.show()
        

def calculate_column_ratios(df1, df2, Nbootstrap=1000, include_columns=[]):
    """
    Calculate the mean and uncertainty of the ratio of values between 
    two DataFrames using Bootstrap resampling. (because uncertainties in mean 
                                                values are large and asymmetric)
    
    Parameters:
    df1 (pandas.DataFrame): The first DataFrame, from experiment 1
    df2 (pandas.DataFrame): The second DataFrame, from experiment 2
    Nbootstrap (int): Number of samples to generate for Bootstrapping. Default 1000.
    include_columns: list of columns to include for analysis

    Returns:
    column_ratios: dictionary, keys being column name, with sub-keys being:
        - mean: the mean of the ratios
        - uncertainty: the uncertainty of the ratio (std. dev. of bootstrap resamplings) 
        - unc_lower: lower uncertainty (16th pct. of bootstrap resamplings) 
        - unc_upper: upper uncertainty (84th pct. of bootstrap resamplings)
    """
    # Verify that the DataFrames have the same column headings
    if not df1.columns.equals(df2.columns):
        raise ValueError("DataFrames must have the same column headings.")
        
    # Columns to analyze
    analyze_columns = [col for col in df1.columns if col in include_columns]
        
    column_ratios = {}

    for col in analyze_columns:
        values1 = df1[col].values
        values2 = df2[col].values
        
        # Are all the array values zero?
        all_zero = all(value == 0 for value in values1) and \
            all(value == 0 for value in values2)
        
        if all_zero:
            column_ratios[col] = {
                'mean': np.nan,
                'uncertainty': np.nan,
                'unc_lower': np.nan,
                'unc_upper': np.nan
            }            
        else:
            # Initialize arrays to store bootstrap results
            ratios = np.zeros(Nbootstrap)
            # Perform bootstrap resampling
            for j in range(Nbootstrap):
                # Resample with replacement
                sample1 = np.random.choice(values1, size=len(values1), replace=True)
                sample2 = np.random.choice(values2, size=len(values2), replace=True)
                
                # Calculate ratio of means for this resampling
                ratios[j] = np.mean(sample2) / np.mean(sample1)
                
            # Remove zeros or inf
            ratios = ratios[np.isfinite(ratios)]
            ratios = ratios[np.abs(ratios)>0.0]
            
            # Store results for this column
            thismean = np.mean(ratios)
            column_ratios[col] = {
                'mean': thismean,
                'uncertainty': np.std(ratios),
                'unc_lower': thismean - np.percentile(ratios, 16),
                'unc_upper': np.percentile(ratios, 84) - thismean
            }
        

    printOutput = False
    if printOutput:
        for k in ['mean', 'uncertainty', 'unc_lower' , 'unc_upper']:
            print(f'{k}: ')
            for col in analyze_columns:
                print(f'{column_ratios[col][k]:.3f},  ', end='')
            print('\n')

    return column_ratios


def getOutputPath(file_paths = None):
    # Ask user for the output path (for plots and CSV file)
    # If empty, dialog box
    # Inputs: file_paths, tuple of both input full file paths, to determine
    #    default for dialog box
    # Output: output path name
    
    outputPath = input('Enter the path (folder) for output, or leave blank to use a dialog box: ')

    if outputPath:
        return outputPath
    else:
        root = tk.Tk()
        root.withdraw()
        # default will be lowest common path
        if file_paths is None:
            outputPath = filedialog.askdirectory(title="Select a folder")
        else:
            default_output_path = common_lowermost_folder(file_paths[0], file_paths[1])
            print(f'Default output path: {default_output_path}')
            outputPath = filedialog.askdirectory(title="Select a folder",
                                                 initialdir = default_output_path)
        return outputPath

def common_lowermost_folder(path1, path2):
    # Get the lower-most common folder of two paths
    # Split the paths into components
    parts1 = path1.split(os.sep)
    parts2 = path2.split(os.sep)
    
    # Initialize the common path
    common_path = []
    
    # Iterate over the components and find the common ones
    for part1, part2 in zip(parts1, parts2):
        if part1 == part2:
            common_path.append(part1)
        else:
            break
    
    # Return the lowermost common folder
    return os.sep.join(common_path)



def get_all_columns(file_path, sheet_name, exclude_from_all=[]):
    """
    Get all possible columns to use, based on the columns in the 
    specified sheet of an Excel file
    
    Inputs:
        file_path : full file path of the Excel spreadsheet
        sheet_name : sheet name
        exclude_from_all : columns to exclude (list of strings)
        
    Outputs
        columns_all : list of strings (column names, i.e. behavior names)
    
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    columns_all = df.columns.to_list()
    
    # Remove columns, e.g. "Dataset" and "Source"
    for x in exclude_from_all:
        if x in columns_all: columns_all.remove(x)

    return columns_all



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
    Note that each column is a property or behavior
    
    Parameters:
        df: processed Pandas dataframe
        exclude_columns: list of columns to ignore for analysis
    
    Returns:
        stats_dict: dictionary with keys corresponding to 'column_name', 'mean', 'N', 'std', and 'sem'
    """
    stats_dict = {}
    # Exclude 'Dataset' column, if not already excluded; also 'Source'
    if 'Dataset' not in exclude_columns:
        exclude_columns.append('Dataset')
    if 'Source' not in exclude_columns:
        exclude_columns.append('Source')
        
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
    # Exclude 'Source' column, if not already excluded
    if 'Source' not in exclude_columns:
        exclude_columns.append('Source')
    
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



def compare_datasets(file_paths, dataLabels, common_sheets, comparison_file_path):
    """
    Perform statistical tests on the two sets, which the user might use 
    to evaluate similarity.
    Outputs an Excel file.    
    
    Called by load_and_combine_dataframes(), in which file paths etc. 
    are specified
    
    Parameters:
        file_paths: list of file paths for the datasets
        dataLabels: list of data labels for the datasets
        common_sheets: list of common sheets between the datasets
        comparison_file_path: path to the output Excel file for comparisons
    """
    
    print('Comparing the two datasets. (Output -> Excel)')
    exclude_from_all = []
    
    with pd.ExcelWriter(comparison_file_path) as comp_writer:
        for sheet_name in common_sheets:
            # Read sheets from both files
            df1 = pd.read_excel(file_paths[0], sheet_name=sheet_name)
            df2 = pd.read_excel(file_paths[1], sheet_name=sheet_name)
            
            # Remove statistics rows if present
            df1 = keep_dataframe_to_blank_line(df1)
            df2 = keep_dataframe_to_blank_line(df2)
            
            # Analyze (compare + stats)
            stats1 = calc_stats_dataframes(df1, exclude_from_all)
            stats2 = calc_stats_dataframes(df2, exclude_from_all)
            stat_tests = stat_comparisons(df1, df2, exclude_from_all)
            
            # Create comparison dataframe
            comparison_data = []
            for column in stats1.keys():
                if column in stats2:
                    row = [
                        '{:.3G}'.format(stats1[column]['mean']),
                        '{:.3G}'.format(stats2[column]['mean']),
                        stats1[column]['N'],
                        stats2[column]['N'],
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
                    comparison_data.append([column] + row)
            
            headers = ['column_name', 'mean_1', 'mean_2', 'N_1', 'N_2', 'std_1', 'std_2', 'sem_1', 'sem_2']
            if stat_tests is not None:
                headers.extend(['p_MWU', 'p_KS'])
            
            comparison_df = pd.DataFrame(comparison_data, columns=headers)
            
            # Write comparison dataframe to Excel
            comparison_df.to_excel(comp_writer, sheet_name=sheet_name, index=False)



def load_and_combine_dataframes(outputStats = True, compareSets = False):
    """ 
    Combine datasets from two experiments. 
    Runs on all sheets in inputs
    Asks user for files, etc.
    Combine datasets’ outputs from two different experiments, 
        concatenating information in the summary Excel files. 
    Creates a new composite Excel file. Adds "dataLabel" column at the end
    Optional. If outputStats == True, calculate the statistics for 
        each column of the concatenated dataset and write these also
        to the Excel file.
    Optional: if compareSets == True, call compare_datasets() to 
        perform statistical tests on the two sets, 
        which the user might use to evaluate similarity.
        Outputs an Excel file.
    
    Inputs:
        outputStats  : if true, calculate the statistics for 
                each column of the concatenated dataset
        compareSets : If true, calc stats on sets and run statistical tests
    
    Returns:
        df : dataframe
        writes Excel file, and (optional) CSV file
    
    """
    
    file_paths, dataLabels = get_two_filePaths_and_labels()
    
    # Get common sheets between the files
    common_sheets = get_common_sheets(file_paths[0], file_paths[1])
    
    if not common_sheets:
        print("No common sheets found between the files.")
        return None
    
    # Ask user for sheets to use
    sheets_to_use = select_sheets(common_sheets)

    # Output location
    print('Enter the output folder for the combined dataset')
    outputPath = getOutputPath(file_paths)
    
    # Output Excel file name
    inputstr = 'Output Excel file name; can omit ".xlsx"' + \
        '\n  Leave blank for "behavior_counts_combined.xlsx": '
    fileNameExcel = input(inputstr)
    if fileNameExcel == '':
        fileNameExcel = 'behavior_counts_combined.xlsx'
    root, ext = os.path.splitext(fileNameExcel)
    if ext != '.xlsx':
        fileNameExcel = fileNameExcel + '.xlsx'
    Excel_output_file = os.path.join(outputPath, fileNameExcel)
    
    # Process each common sheet
    with pd.ExcelWriter(Excel_output_file) as writer:
        for sheet_name in sheets_to_use:
            # Read sheets from both files
            df1 = pd.read_excel(file_paths[0], sheet_name=sheet_name)
            df2 = pd.read_excel(file_paths[1], sheet_name=sheet_name)
            
            # Remove statistics rows if present
            df1 = keep_dataframe_to_blank_line(df1)
            df2 = keep_dataframe_to_blank_line(df2)
            
            # Add dataLabel column
            df1['Source'] = dataLabels[0]
            df2['Source'] = dataLabels[1]
            
            # Combine dataframes
            df_combined = pd.concat([df1, df2], axis=0)
            
            if outputStats:
                # Calculate stats for the concatenated dataframe
                stats_combined = calc_stats_dataframes(df_combined, [])
                
                # Create stats dataframe
                stats_df = pd.DataFrame({
                    col: {
                        'mean': stats_combined[col]['mean'],
                        'N': stats_combined[col]['N'],
                        'std': stats_combined[col]['std'],
                        'sem': stats_combined[col]['sem']
                    } for col in stats_combined
                })
                
                # Reorder and format stats
                stats_df = stats_df.reindex(['mean', 'std', 'sem', 'N'])
                stats_df = stats_df.reset_index().rename(columns={'index': 'Dataset'})
                
                # Write combined data and stats to Excel
                df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
                last_row = len(df_combined)
                stats_df.to_excel(writer, sheet_name=sheet_name, 
                                startrow=last_row+2, index=False, header=False)
            else:
                df_combined.to_excel(writer, sheet_name=sheet_name, index=False)

    if compareSets:
        inputstr = 'Stat. tests: Output Excel file name (including ".xlsx").' + \
                    '\n  Leave blank for "compare_stats.xlsx": '
        comparison_file_name = input(inputstr)
        if comparison_file_name == '':
            comparison_file_name = 'compare_stats.xlsx'
        root, ext = os.path.splitext(comparison_file_name)
        if ext != '.xlsx':
            comparison_file_name = comparison_file_name + '.xlsx'
        comparison_file_path = os.path.join(outputPath, comparison_file_name)
        compare_datasets(file_paths, dataLabels, sheets_to_use, comparison_file_path)

    return True

#%%

def main():
        
    print('\n\nCompare experiment behaviors, using output spreadsheets.')
    file_paths, dataLabels = get_two_filePaths_and_labels()

    # Get common sheets
    common_sheets = get_common_sheets(file_paths[0], file_paths[1])
        
    if not common_sheets:
        print("No common sheets found between the files.")
        return
    
    # Ask user for sheets to use
    sheets_to_use = select_sheets(common_sheets)

    # Output location
    outputPath = getOutputPath(file_paths)

    # Output Excel file name
    fileNameExcel = input('Output Excel file name (can omit ".xlsx"): ')
    if not fileNameExcel.endswith('.xlsx'):
        fileNameExcel += '.xlsx'
    excel_output_file = os.path.join(outputPath, fileNameExcel)
    
    # Output plot base name
    baseName0 = input('Base file name for comparison outputs.\n' + \
                     '    Include image extension (e.g. "exptGraphs.eps", or .png): ')
    baseName, out_ext = os.path.splitext(baseName0)
    baseName = baseName.split('.')[0]
    
    # Get the columns (behaviors) to use, based on the first 
    # sheet of the first experiment
    # (Assumes these are the same for all!)
    # Automatically exclude some columns from calculations and plots
    exclude_from_all = ['Dataset', 'Source', 'Frames per sec', 
                        'Image scale (um/px)', 'Total Time (s)',
                        'Mean difference in fish lengths (mm)']
    columns_all = get_all_columns(file_paths[0], sheets_to_use[0], 
                                  exclude_from_all=exclude_from_all) 
    
    # Get behaviors or properties to include in the ratio plot and the log-log plot
    default_keys_Plot = ['perp_noneSee', 'perp_oneSees', 'perp_bothSee',
                         'contact_any', 'contact_head_body',
                         'tail_rubbing', 'maintain_proximity', 
                         'anyPairBehavior', 'Cbend_any',
                         'Rbend_any', 'Jbend_any', 'approaching_any',
                         'fleeing_any', 'isMoving_any', 'isActive_any',
                         'close_to_edge_any']
    default_keys_ratioPlot = default_keys_Plot + \
                             ['Fraction of time in proximity', 
                              'Mean bout rate (/min)',	'Mean head-head dist (mm)']
    print('Select behaviors for log-log plot. (May need to enlarge dialog box)')
    behavior_key_list_loglog = select_items_dialog(columns_all, 
                                            default_keys = default_keys_Plot)
    print('Select behaviors for ratio plot. (May need to enlarge dialog box)')
    behavior_key_list_ratio = select_items_dialog(columns_all, 
                                            default_keys = default_keys_ratioPlot)

    # Color cycle; define here to use consistently.
    colors = cm.tab20b(np.linspace(0, 1, len(behavior_key_list_ratio)))

    # Process each common sheet
    for sheet_name in sheets_to_use:
        print(f"\nProcessing sheet: {sheet_name}")
        
        # Read sheets
        df1 = pd.read_excel(file_paths[0], sheet_name=sheet_name)
        df2 = pd.read_excel(file_paths[1], sheet_name=sheet_name)
        
        # Remove statistics rows
        df1 = keep_dataframe_to_blank_line(df1)
        df2 = keep_dataframe_to_blank_line(df2)
        
        # Analyze (compare + calculate stats)
        stats1 = calc_stats_dataframes(df1, exclude_from_all)
        stats2 = calc_stats_dataframes(df2, exclude_from_all)
        stat_tests = stat_comparisons(df1, df2, exclude_from_all)
        
        column_ratios = calculate_column_ratios(df1, df2, 
                                                Nbootstrap=1000, 
                                                include_columns = behavior_key_list_ratio)

        # Write results to Excel
        write_results_to_excel(stats1, stats2, excel_output_file, 
                             sheet_name, stat_tests, column_ratios)
        
        # Generate plots with sheet-specific filenames
        plot_output_name = f"{baseName}_{sheet_name}_relBehaviorLogLog{out_ext}"
        plot_output_path = os.path.join(outputPath, plot_output_name)
        
        # Plot comparison of datasets, 1 vs 2, showing means and individual pts
        plot_comparison((stats1, stats2), (df1, df2), behavior_key_list_loglog, 
                       colors = colors, dataLabels = (dataLabels[0], dataLabels[1]), 
                       logPlot=True, showTextLabels=False, 
                       showLegend=True, titleString = sheet_name + ' Comparison', 
                       outputFileName=plot_output_path)
        
        ratio_output_name = f"{baseName}_{sheet_name}_relBehaviorRatios{out_ext}"
        ratio_output_path = os.path.join(outputPath, ratio_output_name)
        
        # Call the function to plot, for each behavior, ratio of 
        # behavior value with errorbars.
        # Note that I'm plotting stats of set 2/ set 1,
        # to match "y / x" from the earlier graph
        plot_behavior_ratio(column_ratios, colors = colors,
                            dataLabels = (dataLabels[0], dataLabels[1]),
                            showLegend=False, titleSheet = sheet_name, 
                            outputFileName=ratio_output_path)
        
        sidebyside_output_name = f"{baseName}_{sheet_name}_BehaviorSideBySide{out_ext}"
        sidebyside_output_path = os.path.join(outputPath, sidebyside_output_name)
        # Plot values side-by-side for each behavior
        plot_cols_side_by_side((stats1, stats2), (df1, df2), 
                               include_columns = behavior_key_list_ratio, 
                                  offset_x=0.1, colors = colors,
                                  dataLabels=(dataLabels[0], dataLabels[1]), 
                                  outputFileName=sidebyside_output_path,
                                  titleString = sheet_name + ' side-by-side Comparison',
                                  logYscale = True) 

if __name__ == '__main__':
    main()
    