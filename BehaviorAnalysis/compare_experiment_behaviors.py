# -*- coding: utf-8 -*-
# compare_experiment_behaviors.py
"""
Author:   Raghuveer Parthasarathy
Created on Fri Dec  1 07:27:01 2023
Last modified on Fri Dec  1 07:27:01 2023

Description
-----------

Code to read relative durations output for various datasets and plot
summary stats versus each other.

Plotting code largely from ChatGPT (3.5)!

Inputs:
    
Outputs:
    

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_comparison(path1, file1, path2, file2, exclude_columns=None):
    # Combine paths and file names using os.path.join()
    file_path1 = os.path.join(path1, file1)
    file_path2 = os.path.join(path2, file2)

    # Read CSV files into Pandas DataFrames
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # Exclude specified columns
    if exclude_columns:
        df1 = df1.drop(columns=exclude_columns, errors='ignore')
        df2 = df2.drop(columns=exclude_columns, errors='ignore')

    # Extract relevant rows by the first column
    mean_row1 = df1[df1.iloc[:, 0] == "Mean"]
    sem_row1 = df1[df1.iloc[:, 0] == "Std. Error of Mean"]

    mean_row2 = df2[df2.iloc[:, 0] == "Mean"]
    sem_row2 = df2[df2.iloc[:, 0] == "Std. Error of Mean"]

    # Extract values from DataFrames
    means1 = mean_row1.iloc[:, 1:].values.flatten().astype(float)
    sem1 = sem_row1.iloc[:, 1:].values.flatten().astype(float)

    means2 = mean_row2.iloc[:, 1:].values.flatten().astype(float)
    sem2 = sem_row2.iloc[:, 1:].values.flatten().astype(float)

    # Plotting
    plt.figure()
    plt.errorbar(means1, means2, xerr=sem1, yerr=sem2, fmt='o', 
                 capsize=5, markersize=12, color='darkmagenta')

    # Add text annotations for each point
    for i, (x, y) in enumerate(zip(means1, means2)):
        plt.text(x, y, f"  {df1.columns[i + 1]}", fontsize=8, ha='left', 
                 va='bottom')  #, rotation=-45

    # Set labels and title
    plt.xlabel(f"Mean ({file1})", fontsize=12)
    plt.ylabel(f"Mean ({file2})", fontsize=12)
    plt.title("Comparison of Mean Relative Durations", fontsize=14)

    #plt.plot(np.arange(0.0, 0.05, 0.001), np.arange(0.0, 0.05, 0.001), linestyle='dashed', color='gray')
    
    #plt.xlim(0, 0.04)
    #plt.ylim(0, 0.01)
    
    # Display the plot
    plt.show()



def scatter_plots_with_error_bars(path1, file1, path2, file2, exclude_columns=None):
    # Combine paths and file names using os.path.join()
    file_path1 = os.path.join(path1, file1)
    file_path2 = os.path.join(path2, file2)

    # Read CSV files into Pandas DataFrames
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # Exclude specified columns
    if exclude_columns:
        df1 = df1.drop(columns=exclude_columns, errors='ignore')
        df2 = df2.drop(columns=exclude_columns, errors='ignore')

    # Extract relevant rows by the first column
    mean_row1 = df1[df1.iloc[:, 0] == "Mean"]
    sem_row1 = df1[df1.iloc[:, 0] == "Std. Error of Mean"]

    mean_row2 = df2[df2.iloc[:, 0] == "Mean"]
    sem_row2 = df2[df2.iloc[:, 0] == "Std. Error of Mean"]

    # Extract values from DataFrames
    means1 = mean_row1.iloc[:, 1:].values.flatten().astype(float)
    sem1 = sem_row1.iloc[:, 1:].values.flatten().astype(float)

    means2 = mean_row2.iloc[:, 1:].values.flatten().astype(float)
    sem2 = sem_row2.iloc[:, 1:].values.flatten().astype(float)

    # Calculate ratio and uncertainty
    ratios = means1 / means2
    ratio_uncertainty = ratios * ((sem1 / means1)**2 + (sem2 / means2)**2)**0.5
    
    # Create scatter plots with error bars for each column
    plt.figure()
    for i, (col_name, ratio, uncertainty) in enumerate(zip(df1.columns[1:], ratios, ratio_uncertainty)):
        plt.errorbar([i], [ratio], yerr=[uncertainty], label=col_name, fmt='o', capsize=5)

    # Set labels and title
    plt.xlabel("Column Index")
    plt.ylabel("Ratio of Mean (File2) to Mean (File1)")
    plt.title("Scatter Plots with Error Bars of Mean Ratios for Each Column")

    # Display the plot
    plt.legend()
    plt.show()
    

# Specify the paths and file names
Path1 = r"C:\Users\Raghu\Documents\Experiments and Projects\Misc\Zebrafish behavior\CSV files and outputs\2 week old - pairs"
File1 = r"behavior_relDuration_2week_light_26Nov2023.csv"
Path2 = r"C:\Users\Raghu\Documents\Experiments and Projects\Misc\Zebrafish behavior\CSV files and outputs\2 week old - pairs in the dark"
File2 = r"behavior_relDuration_2week_dark_26Nov2023.csv"



# Specify columns to exclude
exclude_columns = ["Mean difference in fish lengths (px)", 
                   "Mean Inter-fish dist (px)", "Angle XCorr mean"]

# Call the function to plot the comparison
plot_comparison(Path1, File1, Path2, File2, exclude_columns)

# Call the function to create scatter plots with error bars
scatter_plots_with_error_bars(Path1, File1, Path2, File2, exclude_columns)
