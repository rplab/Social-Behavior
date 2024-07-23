# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# behaviors_main.py
"""
#----------------------------------------------------------------------------
# Raghuveer Parthasarathy (2023-2024)
# First version  : Estelle Trieu 9/19/2022
# Re-written by : Raghuveer Parthasarathy (2023)
# version ='2.0' Raghuveer Parthasarathy -- begun May 2023; see notes.
# last modified: Raghuveer Parthasarathy, July 21, 2024
# ---------------------------------------------------1------------------------
"""

import os
import numpy as np
import yaml
from toolkit import load_expt_config, get_CSV_folder_and_filenames, \
    load_all_position_data, make_frames_dictionary, \
     get_edge_frames_dictionary, \
        get_badTracking_frames_dictionary, \
        write_output_files, write_pickle_file, \
        add_statistics_to_excel
from behavior_identification_single import get_single_fish_characterizations
# , \
#    get_fish_lengths, get_fish_speeds, getTailAngle, get_Cbend_frames, get_Jbend_frames
from behavior_identification import extract_behaviors, \
    get_basic_two_fish_characterizations
    # calcOrientationXCorr, get_relative_orientation
# from behavior_identification import extract_behaviors, get_contact_frames, \
#     get_inferred_contact_frames, get_90_deg_frames, \
#     get_tail_rubbing_frames, get_isMoving_frames, \
#     calcOrientationXCorr, get_approach_flee_frames, \
#     get_relative_orientation



# ---------------------------------------------------------------------------


    
    
def main():
    """
    Main function for calling data reading functions, 
    basic analysis functions, and behavior analysis functions for all 
    CSV files in a set.
    """

    # The main folder containing configuration and parameter files.
    basePath = r'C:\Users\Raghu\Documents\Experiments and Projects\Zebrafish behavior'

    cwd = os.getcwd() # Note the current working directory

    # Load experiment configuration file
    config_path = os.path.join(basePath, r'CSV files and outputs')
    config_file = 'all_expt_configs.yaml'
    expt_config = load_expt_config(config_path, config_file)
    
    # Get CSV column info from configuration file
    CSVinfo_path = os.path.join(basePath, r'CSV files and outputs')
    CSVinfo_file = 'CSVcolumns.yaml'
    with open(os.path.join(CSVinfo_path, CSVinfo_file), 'r') as f:
        all_CSV = yaml.safe_load(f)
    CSVcolumns = all_CSV['CSVcolumns']

    # Get behavior analysis parameter info from configuration file
    params_path = os.path.join  (basePath, r'CSV files and outputs')
    params_file = 'analysis_parameters.yaml'
    with open(os.path.join(params_path, params_file), 'r') as f:
        all_param = yaml.safe_load(f)
    params = all_param['params']
    
    # Get folder containing CSV files, and all "results" CSV filenames
    # Note that dataPath is the path containing CSVs, which 
    # may be a subgroup path
    dataPath, allCSVfileNames, subGroupName = \
        get_CSV_folder_and_filenames(expt_config) 
    print(f'\n\n All {len(allCSVfileNames)} CSV files starting with "results": ')
    print(allCSVfileNames)
    
    # If there are subgroups, modify the output Excel file name for
    # summary statistics of each behavior for each dataset -- instead
    # of "behavior_counts.xlsx" (or whatever params["allDatasets_ExcelFile"]
    # currently is), append subGroupName
    if not subGroupName==None:
        base_name, extension = os.path.splitext(params["allDatasets_ExcelFile"])
        params["allDatasets_ExcelFile"] = f"{base_name}_{subGroupName}{extension}"
        print(f"Modifying output allDatasets_ExcelFile file name to be: {params['allDatasets_ExcelFile']}")
    
    print('\nEnter the filename for an output pickle file (w/ all datasets).')
    pickleFileName = input('   Will append .pickle. Leave blank for none.: ')
    
    # Number of datasets
    N_datasets = len(allCSVfileNames)
        
    # For display    
    showAllPositions = False

    # load all position data and determine general parameters 
    # such as fps and scale
    datasets = load_all_position_data(allCSVfileNames, expt_config, 
                                      CSVcolumns, dataPath, params, 
                                      showAllPositions)

    # Check that the number of fish is the same for all datasets; note this
    Nfish_values = [dataset.get("Nfish") for dataset in datasets]
    if len(set(Nfish_values)) != 1:
        raise ValueError("Not all datasets have the same 'Nfish' value")
    Nfish = Nfish_values[0]
    
    # Identify close-to-edge frames for each dataset
    # Call get_edge_frames for each datasets[j] and put results in a 
    # dictionary that includes durations of edge events, etc.
    datasets = get_edge_frames_dictionary(datasets, params, 
                                          expt_config['arena_radius_mm'],
                                          CSVcolumns)
    
    # Identify bad-tracking frames for each dataset
    # Call get_bad_headTrack_frames and get_bad_bodyTrack_frames
    # for each datasets[j] and put results in a 
    # dictionary that includes durations of events, etc.
    datasets = get_badTracking_frames_dictionary(datasets, params, 
                                          CSVcolumns, tol=0.001)
    
    # For each dataset, characterizations that involve single fish
    # (e.g. fish length, bending, speed)
    datasets = get_single_fish_characterizations(datasets, CSVcolumns,
                                                 expt_config, params)
    
    # For each dataset, perform “basic” two-fish characterizations 
    # such as inter-fish distance, if Nfish > 1. 
    if Nfish==2:
        datasets = get_basic_two_fish_characterizations(datasets, CSVcolumns,
                                                     expt_config, params)
    

    # For each dataset, identify social behaviors
    if Nfish > 1:
        for j in range(N_datasets):
            
            print('Identifying two-fish behaviors for Dataset: ', 
                  datasets[j]["dataset_name"])
            
            perp_noneSee_frames, \
                    perp_oneSees_frames, \
                    perp_bothSee_frames, \
                    perp_larger_fish_sees_frames, \
                    perp_smaller_fish_sees_frames, \
                    contact_any_frames, \
                    contact_head_body_frames, \
                    contact_larger_fish_head, contact_smaller_fish_head, \
                    contact_inferred_frames, tail_rubbing_frames, \
                    approaching_frames, fleeing_frames \
                    = extract_behaviors(datasets[j], params, CSVcolumns)
            # removed "circling_frames," from the list
            
            # For each behavior, a dictionary containing frames, 
            # frames with "bad" elements removed
            # and a 2xN array of initial frames and durations
            # Use the behavior key name as the 
            # Loop through a list of key names and arrays
            # The list of keys could be outside the loop, but for clarity
            # I'll be redundant, at least for now.
            behavior_keys = ('perp_noneSee', 'perp_oneSees', 
                             'perp_bothSee', 'perp_larger_fish_sees',
                             'perp_smaller_fish_sees', 
                             'contact_any', 'contact_head_body', 
                             'contact_larger_fish_head', 'contact_smaller_fish_head',
                             'contact_inferred', 'tail_rubbing',
                             'approaching_Fish0', 'approaching_Fish1',
                             'fleeing_Fish0', 'fleeing_Fish1')
            behavior_arrays = (perp_noneSee_frames, perp_oneSees_frames, 
                         perp_bothSee_frames,
                         perp_larger_fish_sees_frames,
                         perp_smaller_fish_sees_frames,
                         contact_any_frames, contact_head_body_frames,
                         contact_larger_fish_head, contact_smaller_fish_head,
                         contact_inferred_frames, tail_rubbing_frames,
                         approaching_frames[0], approaching_frames[1],
                         fleeing_frames[0], fleeing_frames[1])
    
            for b_key, b_array in zip(behavior_keys, behavior_arrays):
                datasets[j][b_key] = make_frames_dictionary(b_array,
                                              (datasets[j]["edge_frames"]["raw_frames"],
                                               datasets[j]["bad_bodyTrack_frames"]["raw_frames"]),
                                               behavior_name = b_key,
                                               Nframes=datasets[j]['Nframes'])

    # Write pickle file containing all datasets (optional)
    if pickleFileName != '':
        list_for_pickle = [datasets, CSVcolumns, expt_config, params]
        write_pickle_file(list_for_pickle, dataPath, 
                          params['output_subFolder'], pickleFileName)
    
    # Write the output files (CSV, Excel)
    write_output_files(params, dataPath, datasets)
    
    # Modify the Excel sheets with behavior counts to include
    # summary statistics for all datasets (e.g. average for 
    # each behavior)
    add_statistics_to_excel(params['allDatasets_ExcelFile'])
    
    # Write a YAML file with parameters, combining expt_config,
    # analysis parameters, and dataPath of subgroup
    more_param_output = dict({'dataPath': dataPath})
    all_outputs = expt_config | params | more_param_output
    with open('all_params.yaml', 'w') as file:
        yaml.dump(all_outputs, file)
    
    # Return to original directory
    os.chdir(cwd)
    
    
if __name__ == '__main__':
    main()
