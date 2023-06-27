# -*- coding: utf-8 -*-
# split_behavior_CSVs.py
"""
Author:   Raghuveer Parthasarathy
Created on Sat Jun 24 19:48:56 2023
Last modified on Sat Jun 24 19:48:56 2023

Description
-----------
Program to read all CSVs with names starting with string startString, in which 
  frame numbers are the second column, and split them into two CSVs such that
  frames 1 through frameSplit are in one CSV and frames (frameSplit+1)-End
  are in the other. In the second CSV, subtract frameSplit from all frame numbers.

Parameters: Modify code with the intended values
    startString : beginning string of CSV filenames to consider
    outString{1,2} : string appended to filenames for the output CSVs
    frameSplit : (int) see the description above; last frame of first split

"""

import os
import csv

# Parameters
startString = 'results_SocPref_5b'
outString1 = '_light'
outString2 = '_dark'
frameSplit = int(7500)


cwd = os.getcwd() # Current working directory

folder_path = input("Enter the folder path for CSV files, or leave empty for cwd: ")
    
# Validate the folder path
while not os.path.isdir(folder_path):
    print("Invalid folder path. Please try again.")
    folder_path = input("Enter the folder path: ")

os.chdir(folder_path)
    
# Make a list of all relevant CSV files in the folder
allCSVfileNames = []
for filename in os.listdir(folder_path):
    if filename.endswith('.csv') and filename.startswith(startString):
        allCSVfileNames.append(filename)

for CSVfileName in allCSVfileNames:
    print(CSVfileName)
    CSVbase = CSVfileName.replace('.csv', '')
    outFileName1 = CSVbase + outString1 + '.csv'
    outFileName2 = CSVbase + outString2 + '.csv'

    # Code from ChatGPT3.5, modified
    with open(CSVfileName, 'r') as file:
        csv_reader = csv.reader(file)
        rows_out1 = []
        rows_out2 = []

        for row in csv_reader:
            if len(row) > 1 and row[1].isdigit():
                if int(row[1]) <= frameSplit:
                    rows_out1.append(row)
                else:
                    row[1] = int(float(row[1]) - float(frameSplit))
                    rows_out2.append(row)
    
    with open(outFileName1, 'w', newline='') as file_out1:
        csv_writer_out1 = csv.writer(file_out1)
        csv_writer_out1.writerows(rows_out1)
    
    with open(outFileName2, 'w', newline='') as file_out2:
        csv_writer_out2 = csv.writer(file_out2)
        csv_writer_out2.writerows(rows_out2)
    

        
# Return to original directory
os.chdir(cwd)
    