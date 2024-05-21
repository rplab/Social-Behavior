# -*- coding: utf-8 -*-
# calc_transition_matrix.py
"""
Author:   Raghuveer Parthasarathy
Created on Thu Apr  4 06:39:18 2024
Major modifications April 24, 2024
Last modified on April 24, 2024

Description
-----------

Using previously-tabulated behavior events to calculate the probabilities of 
"transitions" between behaviors. Ignores time information; just sequences.

A behavior can have a transition to more than one other behavior, if more 
than one other behavior starts during its run or within max_frame_gap afterwards

See "Transition Matrix Notes" document.
Most code from Claude 3 (AI)

Inputs:
    
Outputs:
    

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_transition_count_matrix(filename, behavior_labels_subset, max_frame_gap=0):
    # Initialize the transition_count_matrix
    N_behaviors_to_use = len(behavior_labels_subset)
    transition_count_matrix = np.zeros((N_behaviors_to_use, N_behaviors_to_use, 0), dtype=int)

    # Read the Excel file
    with pd.ExcelFile(filename) as xls:
        # Get the number of sheets
        Nsheets = len(xls.sheet_names)

        # Resize the transition_count_matrix to match the number of sheets
        transition_count_matrix = np.resize(transition_count_matrix, (N_behaviors_to_use, N_behaviors_to_use, Nsheets))

        for sheet_idx, sheet_name in enumerate(xls.sheet_names):
            # Read the sheet into a DataFrame
            print('Sheet: ', sheet_name)
            df = xls.parse(sheet_name)

            # Check if all items in behavior_labels_subset are column names in the sheet
            all_columns = df.columns[1:]  # Exclude the 'Frame' column
            missing_columns = set(behavior_labels_subset) - set(all_columns)
            if missing_columns:
                raise ValueError(f"The following columns were not found in sheet '{sheet_name}': {', '.join(missing_columns)}")

            # Reorder the columns of df
            ordered_columns = ['Frame'] + [col for col in behavior_labels_subset if col in all_columns]
            df = df[ordered_columns]

            # Get the runs for all behaviors in this sheet
            all_runs = {}
            for col_idx, col_name in enumerate(df.columns[1:], start=1):
                # print(' Column: ', col_name)
                runs = get_runs(df.iloc[:, col_idx].replace(r'[\s\t]+', '', regex=True))
                all_runs[col_idx] = runs

            # Iterate over the runs and update the transition_count_matrix
            for col_idx, col_runs in all_runs.items():
                col_name = df.columns[col_idx]
                for start, end in col_runs:
                    # For each run of this behavior
                    for other_col_idx, other_runs in all_runs.items():
                        if col_idx != other_col_idx:
                            other_col_name = df.columns[other_col_idx]
                            for other_start, other_end in other_runs:
                                if start + 1 <= other_start <= end + max_frame_gap + 1:
                                    transition_count_matrix[col_idx - 1, other_col_idx - 1, sheet_idx] += 1

    return transition_count_matrix

def get_runs(series):
    """Helper function to get the runs of consecutive 'X' in a Series"""
    runs = []
    start = None
    prev_value = ''
    for idx, value in series.items():
        if pd.notnull(value) and value == 'X' and prev_value != 'X':
            start = idx
        elif (pd.isnull(value) or value != 'X') and prev_value == 'X':
            runs.append((start, idx - 1))
        prev_value = 'X' if pd.notnull(value) and value == 'X' else ''
    if prev_value == 'X':
        runs.append((start, series.index[-1]))
    return runs



def plot_heatmap(data, behavior_labels, max_value=None, plot_diagonals=True,
                 showValues = False, showTitle = False):
    fig, ax = plt.subplots(figsize=(10, 8))

    if not plot_diagonals:
        diagonal_mask = np.zeros_like(data, dtype=bool)
        np.fill_diagonal(diagonal_mask, True)
        data = np.ma.masked_array(data, mask=diagonal_mask)

    # Create a new mask for NaN values
    nan_mask = np.isnan(data)
    data = np.ma.masked_array(data, mask=nan_mask)

    im = ax.imshow(data, cmap='hot', vmax=max_value)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Transition Probability', rotation=-90, va="bottom", fontsize=18)

    # Set ticks and labels using behavior_labels
    ax.set_xticks(np.arange(len(behavior_labels)))
    ax.set_yticks(np.arange(len(behavior_labels)))
    ax.set_xticklabels(behavior_labels, fontsize=16, rotation=-45, ha='right')
    ax.set_yticklabels(behavior_labels, fontsize=16, rotation=-45, va='bottom')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if showValues:
        # Loop over data dimensions and create text annotations
        for i in range(len(data)):
            for j in range(len(data)):
                if plot_diagonals or i != j:
                    if np.ma.is_masked(data[i, j]):
                        # Display "nan" in the cell if it's masked (NaN)
                        text = ax.text(j, i, "nan", ha="center", va="center", color="k")
                    else:
                        text = ax.text(j, i, f'{data[i, j]:.2e}', ha="center", va="center", color="deepskyblue")

    # Set the color of the NaN cells to medium grey
    ax.matshow(nan_mask, cmap='Greys', alpha=0.1)

    if not plot_diagonals:
        # Set the color of the diagonal cells to medium grey
        ax.matshow(diagonal_mask, cmap='Greys', alpha=0.1)

    if showTitle:
        ax.set_title("Transition Probabilities", fontsize=18)
    fig.tight_layout()
    plt.show()



def get_behavior_labels(filename):
    # Read the first sheet of the Excel file into a DataFrame
    sheet_df = pd.read_excel(filename, sheet_name=0)

    # Extract the column names, excluding the 'Frame' column
    behavior_labels = [col for col in sheet_df.columns if col != 'Frame']

    return behavior_labels
