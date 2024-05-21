# -*- coding: utf-8 -*-
# sort_CSVfiles.py
"""
Author:   Raghuveer Parthasarathy
Created on Mon May  6 09:56:13 2024
Last modified on Mon May  6 09:56:13 2024

Description
-----------

Inputs:
    
Outputs:
    

"""

import os
import pandas as pd
import shutil
import numpy as np


def process_excel_and_csvs(excel_file, all_csv_path, subfolder_name, 
                           group_code_label, include1_label, 
                           include2_label=None, 
                           excludeCSVList=['wellOffsetPositionsCSVfile.csv']):
    # Load the Excel file
    xl = pd.ExcelFile(excel_file)
    sheet_names = xl.sheet_names
    if len(sheet_names) > 1:
        print(f"Using only the first sheet: {sheet_names[0]}")
    df = xl.parse(sheet_names[0])

    # Find unique Group_Code values and enforce integer type, ignoring NaN and Inf
    group_code_series = df[group_code_label].dropna().replace([np.inf, -np.inf], np.nan).dropna().astype(int)
    group_codes = group_code_series.unique()

    # Get the parent folder of all_csv_path
    parent_folder = os.path.dirname(all_csv_path)

    # Create subfolders and copy wellOffsetPositionsCSVfile.csv if it exists
    for group_code in group_codes:
        subfolder_path = os.path.join(parent_folder, f"{subfolder_name}_{group_code}")
        os.makedirs(subfolder_path, exist_ok=True)
        wellOffsetPositionsCSVfile = os.path.join(all_csv_path, "wellOffsetPositionsCSVfile.csv")
        if os.path.exists(wellOffsetPositionsCSVfile):
            shutil.copy(wellOffsetPositionsCSVfile, subfolder_path)

        # Create output_CSV file
        output_csv = os.path.join(subfolder_path, f"{os.path.basename(excel_file).split('.')[0]}_subset_{group_code}.csv")

        # Process CSV files and append rows to output_CSV
        for csv_file in os.listdir(all_csv_path):
            if csv_file.endswith(".csv") and csv_file not in excludeCSVList:
                csv_path = os.path.join(all_csv_path, csv_file)
                csv_basename = os.path.splitext(csv_file)[0]
                trial_id, pair_id = csv_basename.rsplit('_', 1)
                pair_id = int(pair_id.split('_')[-1])

                # Find the row in the Excel sheet that matches the CSV file name
                row_mask = (df["Trial_ID"].apply(lambda x: str(x) in trial_id)) & (df["Pair_ID"] == pair_id) & (df[group_code_label] == group_code)
                row = df.loc[row_mask]
                # print('here trial_id ', trial_id)
                # print('here pair_id ', pair_id)
                # print(df["Trial_ID"].apply(lambda x: str(x) in trial_id))
                # print(df["Pair_ID"])
                # print(df["Pair_ID"] == int(pair_id))
                # print('here 2')
                # print(group_code_series)
                # asdf = input('asdf')

                if not row.empty:
                    use_row = True
                    if include2_label is not None:
                        use_row = (row[include1_label].iloc[0] == 1) and (row[include2_label].iloc[0] == 1)
                    else:
                        use_row = row[include1_label].iloc[0] == 1

                    if use_row:
                        shutil.copy(csv_path, subfolder_path)
                        row.to_csv(output_csv, mode='a', header=False, index=False)
                        print(row.values.tolist()[0])
                        print(f"Destination file: {output_csv}")

        # Remove the blank first column in the output_CSV file
        output_df = pd.read_csv(output_csv, header=None)
        output_df.to_csv(output_csv, header=False, index=False)
# Usage

excel_fileName = r'SocDef_XGF_AnalysisRaghu_Filtered.xlsx'
excel_pathName = r'C:\Users\Raghu\Documents\Experiments and Projects\Misc\Zebrafish behavior\CSV files and outputs\2 week old - conventionalized versus ex-germ-free fish'
excel_fileFull= os.path.join(excel_pathName, excel_fileName)

all_csv_path = r'C:\Users\Raghu\Documents\Experiments and Projects\Misc\Zebrafish behavior\CSV files and outputs\2 week old - conventionalized versus ex-germ-free fish\All_CSV'

subfolder_name = 'XGF_filteredCSVs'
group_code_label = 'Group_Code'

process_excel_and_csvs(excel_fileFull, all_csv_path = all_csv_path,
                       subfolder_name = subfolder_name, 
                       group_code_label = group_code_label, 
                       include1_label = 'Include', include2_label = 'Filter', 
                       excludeCSVList = 'wellOffsetPositionsCSVfile.csv')
