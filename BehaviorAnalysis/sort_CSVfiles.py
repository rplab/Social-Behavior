# -*- coding: utf-8 -*-
# sort_CSVfiles.py
"""
Author:   Raghuveer Parthasarathy
Created on Mon May  6 09:56:13 2024
Modified by: Claude (3.5 Sonnet)
Last modified on January 10, 2025

Description
-----------

Code to move CSV files to destination folders, sorting by group or condition 
   code as indicated in the Excel file. For each CSV file, examines the file
   name and reads the "Trial_ID" and "Pair_ID" columns in the Excel file to 
   find the matching row. Uses the "group_code_label" heading in the Excel 
   file to sort; or ignore group codes.
   Excel file must be in the source MAT folder; it is copied to 
   the CSV parent folder
User is prompted for input paths and Excel filename, with dialog box fallbacks.

Only moves files if the "include1_label" is 1
Optional: only move if "include2_label" is one -- for additional filtering; 
   make "None" to avoid this.
Copies the wellOffsetPositionsCSVfile.csv into each CSV sub-folder.
Copies the Excel file to the "parent" CSV folder, and makes CSV files with 
   the appropriate rows of this Excel file in each sub-folder.
All this is done by process_excel_and_csvs(), with the inputs at the end
   of this .py file

Inputs:
    None; folder names, etc., as user inputs
    
Outputs:
    None; moves CSV files and creates experiment-info CSV files that are a 
          subset of the original info file.

"""

import os
import pandas as pd
import shutil
import numpy as np
from tkinter import filedialog
import tkinter as tk

def get_user_input(prompt, is_file=False, is_save=False, initial_dir=None, file_types=None, default_value=None):
    """Get input from user with dialog box fallback if empty input."""
    if default_value is not None:
        prompt = f"{prompt} (default: {default_value}): "
    user_input = input(prompt).strip()
    
    if not user_input and default_value is not None:
        return default_value
        
    if not user_input:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        if is_file and not is_save:
            user_input = filedialog.askopenfilename(
                initialdir=initial_dir,
                title="Select file",
                filetypes=file_types if file_types else [("All files", "*.*")]
            )
        elif is_file and is_save:
            user_input = filedialog.asksaveasfilename(
                initialdir=initial_dir,
                title="Save file as",
                filetypes=file_types if file_types else [("All files", "*.*")]
            )
        else:
            user_input = filedialog.askdirectory(
                initialdir=initial_dir,
                title="Select directory"
            )
    
    return user_input if user_input else None


def get_optional_parameter(prompt, default_value):
    """Get optional parameter with support for None value"""
    print(f"\n{prompt}")
    print(f"Current default: {default_value}")
    print("Options:")
    print("1. Keep default")
    print("2. Set to None")
    print("3. Enter new value")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        return default_value
    elif choice == "2":
        return None
    elif choice == "3":
        return input("Enter new value: ").strip()
    else:
        print("Invalid choice. Using default.")
        return default_value
    
def process_excel_and_csvs(source_path, excel_file, mainCSV_path, subfolder_name, 
                          group_code_label, include1_label, 
                          include2_label=None, 
                          excludeCSVList=['wellOffsetPositionsCSVfile.csv'], 
                          wellOffsetPositionsCSVfilename = 'wellOffsetPositionsCSVfile'):

    excel_fileFull = os.path.join(source_path, excel_file)

    # Load the Excel file
    try:
        xl = pd.ExcelFile(excel_fileFull)
        sheet_names = xl.sheet_names
        if len(sheet_names) > 1:
            print(f"Using only the first sheet: {sheet_names[0]}")
        df = xl.parse(sheet_names[0])
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    # Copy the Excel file to the parent CSV folder
    try:
        shutil.copy(excel_fileFull, mainCSV_path)
        print(f"Successfully copied Excel file to: {mainCSV_path}")
    except Exception as e:
        print(f"Error copying Excel file: {e}")
        return

    # Handle wellOffsetPositionsCSVfile
    source_well_file = os.path.join(source_path, wellOffsetPositionsCSVfilename)
    dest_well_file = os.path.join(mainCSV_path, wellOffsetPositionsCSVfilename)
    
    # First try to copy from source to main destination
    if os.path.exists(source_well_file):
        try:
            shutil.copy(source_well_file, dest_well_file)
            print(f"Successfully copied {wellOffsetPositionsCSVfilename} to main destination folder")
        except Exception as e:
            print(f"Error copying {wellOffsetPositionsCSVfilename} to main folder: {e}")
    else:
        print(f"\nWarning: {wellOffsetPositionsCSVfilename} not found in source folder: {source_path}")

    # Find unique Group_Code values and enforce integer type, ignoring NaN and Inf
    if group_code_label is None:
        group_codes = [1]
    else:
        group_code_series = df[group_code_label].dropna().replace([np.inf, -np.inf], np.nan).dropna().astype(int)
        group_codes = group_code_series.unique()
    
    # Create subfolders for each group and process files
    for group_code in group_codes:
        if group_code_label is None:
            subfolder_path = mainCSV_path
        else:
            subfolder_path = os.path.join(mainCSV_path, 
                                        f"{subfolder_name}_{group_code}")
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Copy wellOffsetPositionsCSVfile to subfolder if it exists in main destination
        if os.path.exists(dest_well_file):
            try:
                shutil.copy(dest_well_file, subfolder_path)
                print(f"Copied {wellOffsetPositionsCSVfilename} to {os.path.basename(subfolder_path)}")
            except Exception as e:
                print(f"Error copying {wellOffsetPositionsCSVfilename} to subfolder {os.path.basename(subfolder_path)}: {e}")

        # Create CSV file for the subset of Excel info
        info_csv = os.path.join(subfolder_path, 
                               f"{os.path.splitext(excel_file)[0]}_set_{group_code}.csv")
        print(f"\nGroup {group_code}. Creating subset CSV file: {info_csv}")

        # Process CSV files and build subset DataFrame
        subset_df = pd.DataFrame()
        for csv_file in os.listdir(source_path):
            if csv_file.endswith(".csv") and csv_file not in excludeCSVList:
                csv_path = os.path.join(source_path, csv_file)
                csv_basename = os.path.splitext(csv_file)[0]
                
                try:
                    trial_id, pair_id = csv_basename.rsplit('_', 1)
                    pair_id = int(pair_id.split('_')[-1])
                except ValueError:
                    print(f'\nError parsing filename: {csv_file}')
                    pair_id = int(input('Enter the pair ID number (integer): '))
                
                # Find matching row in Excel sheet
                if group_code_label is None:
                    row_mask = (df["Trial_ID"].apply(lambda x: str(x) in trial_id)) & \
                             (df["Pair_ID"] == pair_id)
                else:
                    row_mask = (df["Trial_ID"].apply(lambda x: str(x) in trial_id)) & \
                             (df["Pair_ID"] == pair_id) & \
                             (df[group_code_label] == group_code)
                
                row = df.loc[row_mask]

                if not row.empty:
                    use_row = True
                    if include2_label is not None:
                        use_row = (row[include1_label].iloc[0] == 1) and \
                                 (row[include2_label].iloc[0] == 1)
                    else:
                        use_row = row[include1_label].iloc[0] == 1

                    if use_row:
                        try:
                            shutil.move(csv_path, subfolder_path)
                            subset_df = pd.concat([subset_df, row], ignore_index=True)
                        except Exception as e:
                            print(f"Error moving file {csv_file}: {e}")

        # Save subset DataFrame to CSV
        try:
            subset_df.to_csv(info_csv, index=False)
            print(f"Successfully created subset CSV: {info_csv}")
        except Exception as e:
            print(f"Error creating subset CSV: {e}")


def main():
    print("\n=== CSV File Sorting Script ===\n")
    
    # Get source path
    sourceMATpath = get_user_input("Enter source MAT folder path (or press Enter for dialog): ", 
                                  is_file=False)
    if sourceMATpath is None:
        print("No source path selected. Exiting.")
        return

    # Get destination parent path
    destination_parent_path = get_user_input("Enter destination CSV *Parent* folder path (or press Enter for dialog): ", 
                                           is_file=False)
    if destination_parent_path is None:
        print("No destination parent path selected. Exiting.")
        return

    # Create destination CSV folder path using the source folder name
    source_folder_name = os.path.basename(os.path.normpath(sourceMATpath))
    destination_mainCSV_path = os.path.join(destination_parent_path, source_folder_name)

    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_mainCSV_path):
        print(f'Creating CSV folder: {destination_mainCSV_path}')
        os.makedirs(destination_mainCSV_path, exist_ok=True)
    else:
        print(f'CSV folder already exists: {destination_mainCSV_path}')
        proceed = input("Folder already exists. Continue? (y/n): ").lower().strip()
        if proceed != 'y':
            print("Operation cancelled.")
            return

    # Get Excel filename (should be in sourceMATpath)
    excel_fileName = get_user_input("Enter Excel filename (or press Enter for dialog): ", 
                                  is_file=True, 
                                  initial_dir=sourceMATpath,
                                  file_types=[("Excel files", "*.xlsx *.xls")])
    if excel_fileName is None:
        print("No Excel file selected. Exiting.")
        return
    
    # Extract just the filename if full path was provided
    excel_fileName = os.path.basename(excel_fileName)

    print("\n=== Parameter Configuration ===")
    
    # Get all other parameters with defaults
    group_code_label = get_optional_parameter(
        "Group code label (column name in Excel file for grouping)",
        "Group_Code"
    )
    
    include1_label = get_optional_parameter(
        "Primary include label (column name for inclusion criteria)",
        default_value="Include"
    )
    
    include2_label = get_optional_parameter(
        "Secondary include label (optional additional filter column)",
        None
    )
    
    subfolder_name = get_optional_parameter(
        "Subfolder name prefix",
        default_value="Group"
    )
    
    wellOffsetPositionsCSVfilename = get_user_input(
        "Well offset positions CSV filename",
        default_value="wellOffsetPositionsCSVfile.csv"
    )
    
    # Build exclude list
    default_exclude = [wellOffsetPositionsCSVfilename, 
                      os.path.splitext(excel_fileName)[0] + '.csv']
    print("\nDefault files to exclude:", default_exclude)
    additional_excludes = input("Enter additional files to exclude (comma-separated) or press Enter for none: ")
    
    excludeCSVList = default_exclude
    if additional_excludes:
        excludeCSVList.extend([x.strip() for x in additional_excludes.split(',')])

    print("\n=== Configuration Summary ===")
    print(f"Source path: {sourceMATpath}")
    print(f"Destination parent path: {destination_parent_path}")
    print(f"Destination CSV folder: {destination_mainCSV_path}")
    print(f"Excel file: {excel_fileName}")
    print(f"Group code label: {group_code_label}")
    print(f"Include labels: {include1_label}, {include2_label}")
    print(f"Subfolder name: {subfolder_name}")
    print(f"Well offset filename: {wellOffsetPositionsCSVfilename}")
    print(f"Excluded files: {excludeCSVList}")
    
    proceed = input("\nProceed with these settings? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Operation cancelled.")
        return

    # Process the files
    process_excel_and_csvs(sourceMATpath, excel_fileName,
                          mainCSV_path=destination_mainCSV_path,
                          subfolder_name=subfolder_name,
                          group_code_label=group_code_label,
                          include1_label=include1_label,
                          include2_label=include2_label,
                          excludeCSVList=excludeCSVList,
                          wellOffsetPositionsCSVfilename=wellOffsetPositionsCSVfilename)

if __name__ == "__main__":
    main()