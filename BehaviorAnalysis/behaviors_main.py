# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# behaviors_main.py
"""
#----------------------------------------------------------------------------
# Raghuveer Parthasarathy (2023)
# Created By  : Estelle Trieu 9/19/2022
# Re-written by : Raghuveer Parthasarathy (2023)
# version ='2.0' Raghuveer Parthasarathy -- begun May 2023; see notes.
# last modified: Raghuveer Parthasarathy, July 9, 2024
# ---------------------------------------------------1------------------------
"""

import csv
import os
import numpy as np
from time import perf_counter
import pickle
import yaml
from toolkit import get_CSV_folder_and_filenames, load_data, \
    get_dataset_name, make_frames_dictionary, remove_frames, \
        combine_events, get_ArenaCenter, get_edge_frames, get_imageScale, \
        estimate_arena_center, get_interfish_distance, \
        get_fish_lengths, get_fish_speeds, \
        get_bad_headTrack_frames, get_bad_bodyTrack_frames, \
        plotAllPositions, write_output_files, write_pickle_file, \
        write_behavior_txt_file, mark_behavior_frames_Excel, \
        add_statistics_to_excel
from behavior_identification import get_contact_frames, \
    get_inferred_contact_frames, get_90_deg_frames, \
    get_tail_rubbing_frames, get_isMoving_frames, \
    get_Cbend_frames, get_Jbend_frames, \
    calcOrientationXCorr, get_approach_flee_frames, \
    get_relative_orientation

from scipy.stats import skew

# ---------------------------------------------------------------------------


def extract_behaviors(dataset, params, CSVcolumns): 
    """
    Calls functions to identify frames corresponding to each two-fish
    behavioral motif.
    
    Inputs:
        dataset : dictionary, with keys like "all_data" containing all 
                    position data
        params : parameters for behavior criteria
        CSVcolumns : CSV column parameters
    Outputs:
        arrays of all frames in which the various behaviors are found:
            perp_noneSee, perp_oneSees, 
            perp_bothSee, contact_any, contact_head_body, 
            contact_larger_fish_head, contact_smaller_fish_head,
            contact_inferred, tail_rubbing_frames

    """
    
    # Timer
    t1_start = perf_counter()

    # Arrays of head, body positions; angles. 
    # Last dimension = fish (so arrays are Nframes x {1 or 2}, Nfish==2)
    head_pos_data = dataset["all_data"][:,CSVcolumns["head_column_x"]:CSVcolumns["head_column_y"]+1, :]
        # head_pos_data is Nframes x 2 (x and y positions) x 2 (Nfish) array of head positions
    angle_data = dataset["all_data"][:,CSVcolumns["angle_data_column"], :]
    # body_x and _y are the body positions, each of size Nframes x 10 x 2 (fish)
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
        
    Nframes = np.shape(head_pos_data)[0] 

    t1_2 = perf_counter()
    print(f'   t1_2 start 90degree analysis: {t1_2 - t1_start:.2f} seconds')
    # 90-degrees 
    perp_maxHeadDist_px = params["perp_maxHeadDist_mm"]*1000/dataset["image_scale"]
    orientation_dict = get_90_deg_frames(head_pos_data, angle_data, 
                                         Nframes, params["perp_windowsize"], 
                                         params["cos_theta_90_thresh"], 
                                         perp_maxHeadDist_px,
                                         params["cosSeeingAngle"], 
                                         dataset["fish_length_array_mm"])
    perp_noneSee = orientation_dict["noneSee"]
    perp_oneSees = orientation_dict["oneSees"]
    perp_bothSee = orientation_dict["bothSee"]
    perp_larger_fish_sees = orientation_dict["larger_fish_sees"]
    perp_smaller_fish_sees = orientation_dict["smaller_fish_sees"]
 
    t1_3 = perf_counter()
    print(f'   t1_3 start contact analysis: {t1_3 - t1_start:.2f} seconds')
    # Any contact, or head-body contact
    contact_inferred_distance_threshold_px = params["contact_inferred_distance_threshold_mm"]*1000/dataset["image_scale"]
    contact_dict = get_contact_frames(body_x, body_y, dataset["closest_distance_mm"],
                                params["contact_inferred_distance_threshold_mm"], 
                                dataset["image_scale"],
                                dataset["fish_length_array_mm"])
    contact_any = contact_dict["any_contact"]
    contact_head_body = contact_dict["head-body"]
    contact_larger_fish_head = contact_dict["larger_fish_head_contact"]
    contact_smaller_fish_head = contact_dict["smaller_fish_head_contact"]
    contact_inferred_frames = get_inferred_contact_frames(dataset,
                        params["contact_inferred_window"],                                  
                        contact_inferred_distance_threshold_px)

    t1_4 = perf_counter()
    print(f'   t1_4 start tail-rubbing analysis: {t1_4 - t1_start:.2f} seconds')
    # Tail-rubbing
    tailrub_maxTailDist_px = params["tailrub_maxTailDist_mm"]*1000/dataset["image_scale"]
    tail_rubbing_frames = get_tail_rubbing_frames(body_x, body_y, 
                                          dataset["head_head_distance_mm"], 
                                          angle_data, 
                                          params["tail_rub_ws"], 
                                          tailrub_maxTailDist_px, 
                                          params["cos_theta_antipar"], 
                                          params["tailrub_maxHeadDist_mm"])

    t1_5 = perf_counter()
    print(f'   t1_5 start approaching / fleeing analysis: {t1_5 - t1_start:.2f} seconds')
    # Approaching or fleeing
    (approaching_frames, fleeing_frames) = get_approach_flee_frames(dataset, 
                                                CSVcolumns, 
                                                speed_threshold_mm_s = params["approach_speed_threshold_mm_second"],
                                                min_frame_duration = params["approach_min_frame_duration"],
                                                cos_angle_thresh = params["approach_cos_angle_thresh"])


    t1_end = perf_counter()
    print(f'   t1_end end analysis: {t1_end - t1_start:.2f} seconds')

    # removed "circling_wfs," from the list

    return perp_noneSee, perp_oneSees, \
        perp_bothSee, perp_larger_fish_sees, \
        perp_smaller_fish_sees, \
        contact_any, contact_head_body, contact_larger_fish_head, \
        contact_smaller_fish_head, contact_inferred_frames, \
        tail_rubbing_frames, approaching_frames, fleeing_frames

    
def load_expt_config(config_path, config_file):
    """ 
    Loads the experimental configuration file
    Asks user for the experiment being examined
    Inputs:
        config_path, config_file: path and file name of the yaml config file
    Outputs:
        expt_config : dictionary of configuration information
    """
    with open(os.path.join(config_path, config_file), 'r') as f:
        all_config = yaml.safe_load(f)
    all_expt_names = list(all_config.keys())
    print('\n\nALl experiments: ')
    for j, key in enumerate(all_expt_names):
        print(f'  {j}: {key}')
    expt_choice = input('Select experiment (name string or number): ')
    # Note that we're not checking if the choice is valid, i.e. if in 
    # all_expt_names (if a string) or if in 0...len(all_expt_names) (if 
    # a string that can be converted to an integer.)
    try:
        # Is the input string just an integer?
        expt_config = all_config[all_expt_names[int(expt_choice)]]
    except:
        expt_config = all_config[all_expt_names[expt_choice]]
    expt_config['imageScaleLocation'] = os.path.join(expt_config['imageScalePathName'], 
                                                     expt_config['imageScaleFilename'])

    if ("arenaCentersFilename" in expt_config.keys()):
        if expt_config['arenaCentersFilename'] != None:
            expt_config['arenaCentersLocation'] = os.path.join(expt_config['arenaCentersPathName'], 
                                                           expt_config['arenaCentersFilename'])
        else:
            expt_config['arenaCentersLocation'] = None
    else:
        expt_config['arenaCentersLocation'] = None
    
    return expt_config
    
    
    
def main():
    """
    Main function for calling reading functions, basic analysis functions,
    and behavior extraction functions for all CSV files in a set 
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
    params_path = os.path.join(basePath, r'CSV files and outputs')
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
        
    # initialize a list of dictionaries for datasets
    datasets = [{} for j in range(N_datasets)]
    os.chdir(dataPath)

    # For display    
    showAllPositions = False

    # For each dataset, get general properties and load all position data
    for j, CSVfileName in enumerate(allCSVfileNames):
        datasets[j]["CSVfilename"] = CSVfileName
        datasets[j]["dataset_name"] = get_dataset_name(CSVfileName)
        datasets[j]["image_scale"] = float(get_imageScale(datasets[j]["dataset_name"], 
                                                    expt_config))
        datasets[j]["fps"] = expt_config["fps"]
        
        # Get arena center, subtracting image position offset
        datasets[j]["arena_center"] = get_ArenaCenter(datasets[j]["dataset_name"], 
                                                      expt_config)
        # Estimate center location of Arena
        # datasets[j]["arena_center"] = estimate_arena_center(datasets[j]["all_data"],
        #                                                    CSVcolumns["head_column_x"],
        #                                                    CSVcolumns["head_column_y"])

        # Load all the position information as a numpy array
        print('Loading dataset: ', datasets[j]["dataset_name"])
        datasets[j]["all_data"], datasets[j]["frameArray"] = \
            load_data(CSVfileName, CSVcolumns["N_columns"]) 
        datasets[j]["Nframes"] = len(datasets[j]["frameArray"])
        print('   ', 'Number of frames: ', datasets[j]["Nframes"] )
        datasets[j]["total_time_seconds"] = (np.max(datasets[j]["frameArray"]) - \
            np.min(datasets[j]["frameArray"]) + 1.0) / datasets[j]["fps"]
        print('   ', 'Total duration: ', datasets[j]["total_time_seconds"], 'seconds')


        # (Optional) Show all head positions, and arena center, and dish edge. 
        #    (& close threshold)
        if showAllPositions:
            plotAllPositions(datasets[j], CSVcolumns, expt_config['arena_radius_mm'], 
                             params["arena_edge_threshold_mm"])

    # For each dataset, identify close-to-edge and bad-tracking frames
    for j in range(N_datasets):
        print('Dataset: ', datasets[j]["dataset_name"])
        # Identify frames in which one or both fish are too close to the edge
        # First keep as an array, then convert into a dictionary that includes
        #    durations of edge events, etc.
        # Also keep Nframes x Nfish=2 array of distance to edge, mm
        edge_frames, datasets[j]["d_to_edge_mm"] \
            = get_edge_frames(datasets[j], params, expt_config['arena_radius_mm'], 
                                      CSVcolumns["head_column_x"],
                                      CSVcolumns["head_column_y"])
        
        datasets[j]["edge_frames"] = make_frames_dictionary(edge_frames, (), 
                                                            behavior_name='Edge frames',
                                                            Nframes=datasets[j]['Nframes'])
        print('   Number of edge frames: ', len(datasets[j]["edge_frames"]["raw_frames"]))
        
        # Identify frames in which tracking is bad; separately consider head, body
        # Note that body is the most general of these -- use this for criteria
        bad_headTrack_frames = get_bad_headTrack_frames(datasets[j], params, 
                                     CSVcolumns["head_column_x"],
                                     CSVcolumns["head_column_y"], 0.001)
        datasets[j]["bad_headTrack_frames"] = make_frames_dictionary(bad_headTrack_frames, 
                                                                     (), 
                                                                     behavior_name='Bad head track frames',
                                                                     Nframes=datasets[j]['Nframes'])
        print('   Number of bad head tracking frames: ', len(datasets[j]["bad_headTrack_frames"]["raw_frames"]))
        bad_bodyTrack_frames = get_bad_bodyTrack_frames(datasets[j], params, 
                                     CSVcolumns["body_column_x_start"],
                                     CSVcolumns["body_column_y_start"],
                                     CSVcolumns["body_Ncolumns"], 0.001)
        datasets[j]["bad_bodyTrack_frames"] = make_frames_dictionary(bad_bodyTrack_frames, (), 
                                                                     behavior_name='Bad track frames',
                                                                     Nframes=datasets[j]['Nframes'])
        print('   Number of bad body tracking frames: ', len(datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))


    # For each dataset, characterizations that involve single fish
    # (e.g. fish length, bending, speed)
    for j in range(N_datasets):
        print('Single-fish characterizations for Dataset: ', 
                  datasets[j]["dataset_name"])
        # Get the length of each fish in each frame (sum of all segments)
        # Nframes x 2 array
        datasets[j]["fish_length_array_mm"] = \
            get_fish_lengths(datasets[j]["all_data"], 
                             datasets[j]["image_scale"], CSVcolumns)
            
        # Get the speed of each fish in each frame (frame-to-frame
        # displacement of head position, um/s); 0 for first frame
        # Nframes x 2 array
        datasets[j]["speed_array_mm_s"] = \
            get_fish_speeds(datasets[j]["all_data"], CSVcolumns, 
                            datasets[j]["image_scale"], expt_config['fps'])
            
        # Frames with speed above threshold (i.e. moving fish)
        # "_each" is a dictionary with a key for each fish, 
        #     each containing a numpy array of frames in which that fish is moving
        # "_any" is a numpy array of frames in which any fish is moving
        # "_all" is a numpy array of frames in which all fish are moving
        
        isMoving_frames_each, isMoving_frames_any, isMoving_frames_all = \
            get_isMoving_frames(datasets[j], params["motion_speed_threshold_mm_second"])

        # For movement indicator, for "any" and "all" fish,
        # make a dictionary containing frames, removing frames with 
        # "bad" elements. Unlike other characteristics, need to "dilate"
        # bad tracking frames, since each bad tracking frame affects 
        # prior and subsequent speed measure.
        # Also makes a 2xN array of initial frames and durations 
        isMoving_keys = ('isMoving_any', 'isMoving_all')
        isMoving_arrays = (isMoving_frames_any, isMoving_frames_all)
        for isMoving_key, isMoving_array in zip(isMoving_keys, isMoving_arrays):
            badTrFramesRaw = np.array(datasets[j]["bad_bodyTrack_frames"]["raw_frames"]).astype(int)
            dilate_badTrackFrames = np.concatenate((badTrFramesRaw,
                                                   badTrFramesRaw + 1))
            dilate_badTrackFrames = np.unique(dilate_badTrackFrames)
            datasets[j][isMoving_key] = make_frames_dictionary(isMoving_array,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           dilate_badTrackFrames),
                                          behavior_name = isMoving_key,
                                          Nframes=datasets[j]['Nframes'])


        # Frames with C-bends and J-bends; each is a dictionary with two 
        # keys, one for each fish, each containing a numpy array of frames
        Cbend_frames_each = get_Cbend_frames(datasets[j], CSVcolumns, 
                                             params["Cbend_threshold"])
        Jbend_frames_each = get_Jbend_frames(datasets[j], CSVcolumns, 
                                             (params["Jbend_rAP"], 
                                              params["Jbend_cosThetaN"], 
                                              params["Jbend_cosThetaNm1"]))
        # numpy array of frames with C-bend for *any* fish
        Cbend_frames_any = np.unique(np.concatenate(list(Cbend_frames_each.values())))
        # numpy array of frames with J-bend for *any* fish
        Jbend_frames_any = np.unique(np.concatenate(list(Jbend_frames_each.values())))
        
        # For C-bends and J-bends, for each fish and for any fish,
        # make a dictionary containing frames, 
        # remove frames with "bad" elements
        # Also makes a 2xN array of initial frames and durations, though
        # the duration of bends is probably 1 almost always
        CJ_keys = ('Cbend_Fish0', 'Cbend_Fish1', 'Cbend_any',
                   'Jbend_Fish0', 'Jbend_Fish1', 'Jbend_any')
        CJ_arrays = (Cbend_frames_each[0], Cbend_frames_each[1], 
                     Cbend_frames_any,
                     Jbend_frames_each[0], Jbend_frames_each[1], 
                     Jbend_frames_any)
        for CJ_key, CJ_array in zip(CJ_keys, CJ_arrays):
            datasets[j][CJ_key] = make_frames_dictionary(CJ_array,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]),
                                          behavior_name = CJ_key,
                                          Nframes=datasets[j]['Nframes'])
            
    # For each dataset, measure inter-fish distance (head-head and
    # closest points) in each frame, and calculate a
    # sliding window cross-correlation of heading angles
    for j in range(N_datasets):

        # Get the inter-fish distance (distance between head positions) in 
        # each frame (Nframes x 1 array). Units = mm
        datasets[j]["head_head_distance_mm"], datasets[j]["closest_distance_mm"] = \
            get_interfish_distance(datasets[j]["all_data"], CSVcolumns,
                                   datasets[j]["image_scale"])
        
        # Relative orientation of fish (angle between heading and
        # connecting vector). Nframes x Nfish==2 array; radians
        datasets[j]["relative_orientation"] = \
            get_relative_orientation(datasets[j], CSVcolumns)   
        
        # Get the sliding window cross-correlation of heading angles
        datasets[j]["xcorr_array"] = \
            calcOrientationXCorr(datasets[j]["all_data"], CSVcolumns, 
                                 params["angle_xcorr_windowsize"])

        
    # For each dataset, exclude bad tracking frames from calculations of:
    # the mean inter-fish distance, 
    # the mean fish length, 
    # the mean and std. absolute difference in fish length, 
    for j in range(N_datasets):
        print('Removing bad frames from stats for length, distance, for Dataset: ', 
              datasets[j]["dataset_name"])
        goodIdx = np.where(np.in1d(datasets[j]["frameArray"], 
                                   datasets[j]["bad_bodyTrack_frames"]["raw_frames"], 
                                   invert=True))[0]
        goodLengthArray = datasets[j]["fish_length_array_mm"][goodIdx]
        goodHHDistanceArray = datasets[j]["head_head_distance_mm"][goodIdx]
        goodClosestDistanceArray = datasets[j]["closest_distance_mm"][goodIdx]
        datasets[j]["fish_length_mm_mean"] = np.mean(goodLengthArray)
        datasets[j]["fish_length_Delta_mm_mean"] = np.mean(np.abs(np.diff(goodLengthArray, 1)))
        datasets[j]["fish_length_Delta_mm_std"] = np.std(np.abs(np.diff(goodLengthArray, 1)))
        datasets[j]["head_head_distance_mm_mean"] = np.mean(goodHHDistanceArray)
        datasets[j]["closest_distance_mm_mean"] = np.mean(goodClosestDistanceArray)
        print(f'   Mean fish length: {datasets[j]["fish_length_mm_mean"]:.3f} mm')
        print('   Mean +/- std. of difference in fish length: ', 
              f'{datasets[j]["fish_length_Delta_mm_mean"]:.3f} +/- ',
              f'{datasets[j]["fish_length_Delta_mm_std"]:.3f} mm')
        print(f'   Mean head-to-head distance {datasets[j]["head_head_distance_mm_mean"]:.2f} mm')
        print(f'   Mean closest distance {datasets[j]["closest_distance_mm_mean"]:.2f} px')
    
    # For each dataset, exclude bad tracking frames from the calculation
    # of the mean angle-heading cross-correlation
    # if the bad frames occur anywhere in the sliding window
    for j in range(N_datasets):
        print('Removing bad frames from stats for angle xcorr for Dataset: ', 
              datasets[j]["dataset_name"])
        badFrames = datasets[j]["bad_bodyTrack_frames"]["raw_frames"]
        expandBadFrames = []
        for k in range(len(badFrames)):
            # make a list of all frames that are within windowsize prior to bad frames
            expandBadFrames = np.append(expandBadFrames, 
                                        np.arange(badFrames[k] - 
                                                  params["angle_xcorr_windowsize"]+1, 
                                                  badFrames[k]))
        expandBadFrames = np.unique(expandBadFrames)    
        goodIdx = np.where(np.in1d(datasets[j]["frameArray"], 
                                   expandBadFrames, invert=True))[0]
        goodXCorrArray = datasets[j]["xcorr_array"][goodIdx]
        # Also limit to finite values
        finiteGoodXCorrArray = goodXCorrArray[np.isfinite(goodXCorrArray)]
        datasets[j]["AngleXCorr_mean"] = np.mean(finiteGoodXCorrArray)
        # print(f'   Mean heading angle XCorr: {datasets[j]["AngleXCorr_mean"]:.4f}')
        # Calculating the std dev and skew, but won't write to CSV
        datasets[j]["AngleXCorr_std"] = np.std(finiteGoodXCorrArray)
        datasets[j]["AngleXCorr_skew"] = skew(finiteGoodXCorrArray)
        # print(f'   (Not in CSV) std, skew heading angle XCorr: {datasets[j]["AngleXCorr_std"]:.4f}, {datasets[j]["AngleXCorr_skew"]:.4f}')


    # For each dataset, identify two-fish behaviors
    for j in range(N_datasets):
        
        print('Identifying behaviors for Dataset: ', 
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
