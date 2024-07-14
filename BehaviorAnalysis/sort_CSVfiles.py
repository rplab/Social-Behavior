# -*- coding: utf-8 -*-
# sort_CSVfiles.py
"""
Author:   Raghuveer Parthasarathy
Created on Mon May  6 09:56:13 2024
Last modified on June 17, 2024

Description
-----------

Code to move CSV files to destination folders, sorting by group or condition 
   code as indicated in the Excel file. For each CSV file, examines the file
   name and reads the "Trial_ID" and "Pair_ID" columns in the Excel file to 
   find the matching row. Uses the "group_code_label" heading in the Excel 
   file to sort; or ignore group codes.
   Excel file must be in the source MAT folder; it is copied to 
   the CSV parent folder
Only moves files if the "include1_label" is 1
Optional: only move if "include2_label" is one -- for additional filtering; 
   make "None" to avoid this.
Copies the wellOffsetPositionsCSVfile.csv into each CSV sub-folder.
Copies the Excel file to the "parent" CSV folder, and makes CSV files with 
   the appropriate rows of this Excel file in each sub-folder.
All this is done by process_excel_and_csvs(), with the inputs at the end
   of this .py file
Some code from Claude3 (AI)

Inputs:
    None; folder names, etc., hard-coded at the end.
    
Outputs:
    None; moves CSV files and creates experiment-info CSV files that are a 
          subset of the original info file.

"""

import os
import pandas as pd
import shutil
import numpy as np

def process_excel_and_csvs(source_path, excel_file, mainCSV_path, subfolder_name, 
                           group_code_label, include1_label, 
                           include2_label=None, 
                           excludeCSVList=['wellOffsetPositionsCSVfile.csv'], 
                           wellOffsetPositionsCSVfilename = 'wellOffsetPositionsCSVfile'):

    excel_fileFull= os.path.join(source_path, excel_file)

    # Load the Excel file
    xl = pd.ExcelFile(excel_fileFull)
    sheet_names = xl.sheet_names
    if len(sheet_names) > 1:
        print(f"Using only the first sheet: {sheet_names[0]}")
    df = xl.parse(sheet_names[0])

    # Copy the Excel file to the parent CSV folder
    shutil.copy(excel_fileFull, mainCSV_path)

    wellOffsetPositionsCSVfileFull = os.path.join(mainCSV_path, wellOffsetPositionsCSVfilename)

    # Find unique Group_Code values and enforce integer type, ignoring NaN and Inf
    # If there are no group codes, use the value "1" just to keep 
    if group_code_label is None:
        group_codes = [1]
    else:
        group_code_series = df[group_code_label].dropna().replace([np.inf, -np.inf], np.nan).dropna().astype(int)
        group_codes = group_code_series.unique()
    
    # Create subfolders for each group (or just use main folder if there are not groups)
    # Copy wellOffsetPositionsCSVfile.csv if it exists
    for group_code in group_codes:
        if group_code_label is None:
            subfolder_path = mainCSV_path
        else:
            subfolder_path = os.path.join(mainCSV_path, 
                                          f"{subfolder_name}_{group_code}")
        os.makedirs(subfolder_path, exist_ok=True)
        if os.path.exists(wellOffsetPositionsCSVfileFull):
            shutil.copy(wellOffsetPositionsCSVfileFull, subfolder_path)
        else:
            print('\n\nwellOffsetPositionsCSVfile does not exist! Cannot copy.')

        # Create CSV file name for the subset of the Excel info file
        info_csv = os.path.join(subfolder_path, f"{os.path.basename(excel_file).split('.')[0]}_set_{group_code}.csv")
        print(f"\n Group {group_code}. Destination CSV (subset of Excel file): {info_csv}")

        # Process CSV files, move them, and append Excel file rows to info_csv
        subset_df = pd.DataFrame()
        for csv_file in os.listdir(source_path):
            if csv_file.endswith(".csv") and csv_file not in excludeCSVList:
                csv_path = os.path.join(source_path, csv_file)
                csv_basename = os.path.splitext(csv_file)[0]
                trial_id, pair_id = csv_basename.rsplit('_', 1)
                try:
                    pair_id = int(pair_id.split('_')[-1])
                except:
                    print('\nsort_CSVfiles.py')
                    print('**Error: cannot determine pair ID**') 
                    print('trial_id: ', trial_id)
                    print('pair_id: ', pair_id)
                    pair_id = input('Enter the pair ID number (integer): ')
                    pair_id = int(pair_id)
                
                # Find the row in the Excel sheet that matches the CSV file name
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
                        use_row = (row[include1_label].iloc[0] == 1) and (row[include2_label].iloc[0] == 1)
                    else:
                        use_row = row[include1_label].iloc[0] == 1

                    if use_row:
                        # Make ".copy" if testing:
                        shutil.move(csv_path, subfolder_path)
                        subset_df = pd.concat([subset_df, row], ignore_index=True)

        # Output the subset dataframe as a CSV with a header row
        subset_df.to_csv(info_csv, index=False)
        
#%% Main part

basePath = r'C:\Users\Raghu\Documents\Experiments and Projects\Zebrafish behavior'

# Replace these:
excel_fileName = r'SocPref_6a-b_AnalysisRaghu.xlsx'
sourceMATpath = basePath + r'\MAT files\2 week old - single fish in the dark'
destination_mainCSV_path = basePath + r'\CSV files and outputs\2 week old - single fish in the dark'
group_code_label =  None # e.g. 'Genotype', or 'Group_Code'. Use None [no quotes] to ignore
include1_label = 'Include' # Header of the "include" column
include2_label = None  # use None to avoid additional filtering, or 'Filter' to filter
subfolder_name = 'Genotype' # will append the group code to this for sub-folders; probably make this the same as group_code_label
wellOffsetPositionsCSVfilename = 'wellOffsetPositionsCSVfile.csv' # Probably don't need to change
excludeCSVList = ['wellOffsetPositionsCSVfile.csv', 
                  'SocDef_Shank3_AnalysisRaghu.csv']  # ignore these CSVs

if not os.path.exists(destination_mainCSV_path):
    # Make the parent CSV folder
    print('Parent CSV folder does not exist; creating it.')
    os.makedirs(destination_mainCSV_path, exist_ok=False)

process_excel_and_csvs(sourceMATpath, excel_fileName, 
                       mainCSV_path = destination_mainCSV_path,
                       subfolder_name = subfolder_name, 
                       group_code_label = group_code_label, 
                       include1_label = include1_label, include2_label = include2_label, 
                       excludeCSVList = excludeCSVList, 
                       wellOffsetPositionsCSVfilename = wellOffsetPositionsCSVfilename)
