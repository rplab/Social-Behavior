# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
# behaviors_main.py
"""
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Re-written by : Raghuveer Parthasarathy (2023)
# Created Date: 9/19/2022
# version ='2.0' Raghuveer Parthasarathy
# last modified: Raghuveer Parthasarathy, November 26, 2023
# ---------------------------------------------------------------------------
"""

import csv
import os
import numpy as np
import xlsxwriter
from time import perf_counter
import pickle
from toolkit import get_CSV_folder_and_filenames, load_data, \
    get_dataset_name, make_frames_dictionary, remove_frames, \
        combine_events, get_ArenaCenter, get_edge_frames, get_imageScale, \
        estimate_arena_center, get_interfish_distance, \
        get_fish_lengths, get_bad_headTrack_frames, get_bad_bodyTrack_frames, \
        plotAllPositions, write_behavior_txt_file, mark_behavior_frames_Excel
from behavior_identification import get_contact_frames, \
    get_inferred_contact_frames, get_90_deg_frames, \
    get_tail_rubbing_frames, get_Cbend_frames, get_Jbend_frames, \
    calcOrientationXCorr, get_approach_flee_frames

import matplotlib.pyplot as plt
from scipy.stats import skew

# ---------------------------------------------------------------------------

def defineParameters():
    """ 
    Defines parameters for behavior identification, 
    identifies columns of trajectory CSV files
    """

    params = {
        "arena_edge_threshold_mm" : 5,
        "circle_windowsize" : 25,
        "circle_fit_threshold" : 0.25,
        "circle_distance_threshold": 240,
        "cos_theta_AP_threshold" : -0.7,
        "cos_theta_tangent_threshold" : 0.34,
        "motion_threshold" : 2.0,
        "cos_theta_90_thresh" : 0.26,
        "cosSeeingAngle" : 0.5,
        "perp_windowsize" : 2,
        "perp_maxHeadDist_mm" : 17.0,
        "contact_distance_threshold_mm" : 2.5,
        "contact_inferred_distance_threshold_mm" : 3.5,
        "contact_inferred_window" : 3,
        "tail_rub_ws" : 2,
        "tailrub_maxTailDist_mm" : 2.0,
        "tailrub_maxHeadDist_mm": 12.5,
        "cos_theta_antipar": -0.8,
        "Cbend_threshold" : 2/np.pi, 
        "Jbend_rAP" : 0.98,
        "Jbend_cosThetaN" : 0.34,
        "Jbend_cosThetaNm1" : 0.7,
        "angle_xcorr_windowsize" : 25,
        "approach_speed_threshold_mm_second" : 20,
        "approach_cos_angle_thresh" : 0.5,
        "approach_min_frame_duration" : (2,2)
    }
    
    # Specify columns of the CSV files with fish trajectory information
    CSVcolumns = {    
        "N_columns" : 26, # total number of columns in CSV file
        "pos_data_column_x" : 3, # head position x column (first col == 0)
        "pos_data_column_y" : 4, # head position y column (first col == 0)
        "angle_data_column" : 5, # angle (radians) column (first col == 0)
        "body_column_x_start" : 6, # starting column for body x positions (first==0)
        "body_column_y_start" : 16, # starting column for body x positions (first==0)
        "body_Ncolumns" : 10 # number of body datapoints
    }
    
    return  params, CSVcolumns

def define_imageParameters(exptName):
    """
    Defines parameters related to imaging, such as filenames
    with arena coordinates, frame rate, arena radius.
    
    Write options for different experiments, done with different parameters
    
    Input: 
        exptName = one of 'TwoWeek2023', 'CA2024'
    """

    fps = 25.0  # frames per second
    arena_radius_mm = 25.0  # arena radius, mm

    if exptName == 'TwoWeek2023':
        # Image scale, to be read from file
        imageScalePathName = 'C:/Users/Raghu/Documents/Experiments and Projects/Misc/Zebrafish behavior'
        imageScaleFilename = 'ArenaCenters_SocPref_3456.csv'
        imageScaleLocation = os.path.join(imageScalePathName, 
                                          imageScaleFilename)
        imageScaleColumn  = 4 # column (0-indexed) with image scale
        
        # Arena center locations
        arenaCentersPathName = 'C:/Users/Raghu/Documents/Experiments and Projects/Misc/Zebrafish behavior'
        arenaCentersFilename = 'ArenaCenters_SocPref_3456.csv'
        arenaCentersLocation = os.path.join(arenaCentersPathName, arenaCentersFilename)
        arenaCentersColumns = (5,6) # columns (0-indexed) with x,y arena centers
        
        # filename of CSV file *in each data folder* with image offset filename
        offsetPositionsFilename = 'wellOffsetPositionsCSVfile.csv'
        
        return fps, arena_radius_mm, imageScaleLocation, imageScaleColumn, \
            arenaCentersLocation, arenaCentersColumns, offsetPositionsFilename

    elif exptName == 'CA2024':
        # Image scale, to be read from file
        imageScalePathName = 'C:/Users/Raghu/Documents/Experiments and Projects/Misc/Zebrafish behavior/CSV files and outputs/2 week old - pairs - cholic acid'
        imageScaleFilename = 'ImageScale_SocPrefBA_3b_all.csv'
        imageScaleLocation = os.path.join(imageScalePathName, 
                                          imageScaleFilename)
        imageScaleColumn  = 4 # column (0-indexed) with image scale
        
        # Arena center locations
        arenaCentersLocation = None # estimate from well offset positions
        arenaCentersColumns = None
        
        # filename of CSV file *in each data folder* with image offset filename
        offsetPositionsFilename = 'wellOffsetPositionsCSVfile.csv'
        
        return fps, arena_radius_mm, imageScaleLocation, imageScaleColumn, \
            arenaCentersLocation, arenaCentersColumns, offsetPositionsFilename

    elif exptName == 'Solitary_Cohoused_March2024':
        # Image scale, to be read from file
        imageScalePathName = 'C:/Users/Raghu/Documents/Experiments and Projects/Misc/Zebrafish behavior/CSV files and outputs/TwoWeekOld_Solitary_CoHoused_1_3-2-2024'
        imageScaleFilename = 'SocDef_Solitary_AnalysisRaghu.csv'
        imageScaleLocation = os.path.join(imageScalePathName, 
                                          imageScaleFilename)
        imageScaleColumn  = 4 # column (0-indexed) with image scale
        
        # Arena center locations
        arenaCentersLocation = None # estimate from well offset positions
        arenaCentersColumns = None
        
        # filename of CSV file *in each data folder* with image offset filename
        offsetPositionsFilename = 'wellOffsetPositionsCSVfile.csv'
        
        return fps, arena_radius_mm, imageScaleLocation, imageScaleColumn, \
            arenaCentersLocation, arenaCentersColumns, offsetPositionsFilename

    else:
        raise ValueError("define_imageParameters: Bad experiment set")



def extract_behaviors(dataset, params, CSVcolumns, fps): 
    """
    Calls functions to identify frames corresponding to each two-fish
    behavioral motif.
    
    Inputs:
        dataset : dictionary, with keys like "all_data" containing all 
                    position data
        params : parameters for behavior criteria
        CSVcolumns : CSV column parameters
        fps : frames per second. From defineParameters; same for all datasets
    Outputs:
        arrays of all frames in which the various behaviors are found:
            perpendicular_noneSee, perpendicular_oneSees, 
            perpendicular_bothSee, contact_any, contact_head_body, 
            contact_larger_fish_head, contact_smaller_fish_head,
            contact_inferred, tail_rubbing_frames

    """
    
    # Timer
    t1_start = perf_counter()

    # Arrays of head, body positions; angles. 
    # Last dimension = fish (so arrays are Nframes x {1 or 2}, Nfish==2)
    pos_data = dataset["all_data"][:,CSVcolumns["pos_data_column_x"]:CSVcolumns["pos_data_column_y"]+1, :]
        # pos_data is Nframes x 2 (x and y positions) x 2 (Nfish) array of head positions
    angle_data = dataset["all_data"][:,CSVcolumns["angle_data_column"], :]
    # body_x and _y are the body positions, each of size Nframes x 10 x 2 (fish)
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
        
    Nframes = np.shape(pos_data)[0] 

    # REMOVE CIRCLING
    # t1_1 = perf_counter()
    # print(f'   t1_1 start circling analysis: {t1_1 - t1_start:.2f} seconds')
    # # Circling 
    # circling_wfs = get_circling_frames(pos_data, dataset["inter-fish_distance"], 
    #                                angle_data, Nframes, params["circle_windowsize"], 
    #                                params["circle_fit_threshold"], 
    #                                params["cos_theta_AP_threshold"], 
    #                                params["cos_theta_tangent_threshold"], 
    #                                params["motion_threshold"], 
    #                                params["circle_distance_threshold"])
    
    t1_2 = perf_counter()
    print(f'   t1_2 start 90degree analysis: {t1_2 - t1_start:.2f} seconds')
    # 90-degrees 
    perp_maxHeadDist_px = params["perp_maxHeadDist_mm"]*1000/dataset["image_scale"]
    orientation_dict = get_90_deg_frames(pos_data, angle_data, 
                                         Nframes, params["perp_windowsize"], 
                                         params["cos_theta_90_thresh"], 
                                         perp_maxHeadDist_px,
                                         params["cosSeeingAngle"], 
                                         dataset["fish_length_array"])
    perpendicular_noneSee = orientation_dict["noneSee"]
    perpendicular_oneSees = orientation_dict["oneSees"]
    perpendicular_bothSee = orientation_dict["bothSee"]
    perpendicular_larger_fish_sees = orientation_dict["larger_fish_sees"]
    perpendicular_smaller_fish_sees = orientation_dict["smaller_fish_sees"]
 
    t1_3 = perf_counter()
    print(f'   t1_3 start contact analysis: {t1_3 - t1_start:.2f} seconds')
    # Any contact, or head-body contact
    contact_distance_threshold_px = params["contact_distance_threshold_mm"]*1000/dataset["image_scale"]
    contact_inferred_distance_threshold_px = params["contact_inferred_distance_threshold_mm"]*1000/dataset["image_scale"]
    contact_dict = get_contact_frames(body_x, body_y,  
                                contact_distance_threshold_px, 
                                dataset["fish_length_array"])
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
    tailrub_maxHeadDist_px = params["tailrub_maxHeadDist_mm"]*1000/dataset["image_scale"]
    tailrub_maxTailDist_px = params["tailrub_maxTailDist_mm"]*1000/dataset["image_scale"]
    tail_rubbing_frames = get_tail_rubbing_frames(body_x, body_y, 
                                          dataset["inter-fish_distance"], 
                                          angle_data, 
                                          params["tail_rub_ws"], 
                                          tailrub_maxTailDist_px, 
                                          params["cos_theta_antipar"], 
                                          tailrub_maxHeadDist_px)

    t1_5 = perf_counter()
    print(f'   t1_5 start approaching / fleeing analysis: {t1_5 - t1_start:.2f} seconds')
    # Approaching or fleeing
    speed_threshold_px_frame = params["approach_speed_threshold_mm_second"]\
                                     /dataset["image_scale"]/fps*1000
    (approaching_frames, fleeing_frames) = get_approach_flee_frames(dataset, 
                                                CSVcolumns, 
                                                speed_threshold_px_frame = speed_threshold_px_frame,
                                                min_frame_duration = params["approach_min_frame_duration"],
                                                cos_angle_thresh = params["approach_cos_angle_thresh"])


    t1_end = perf_counter()
    print(f'   t1_end end analysis: {t1_end - t1_start:.2f} seconds')

    # removed "circling_wfs," from the list

    return perpendicular_noneSee, perpendicular_oneSees, \
        perpendicular_bothSee, perpendicular_larger_fish_sees, \
        perpendicular_smaller_fish_sees, \
        contact_any, contact_head_body, contact_larger_fish_head, \
        contact_smaller_fish_head, contact_inferred_frames, \
        tail_rubbing_frames, approaching_frames, fleeing_frames

    
    
def main():
    """
    Main function for calling reading functions, basic analysis functions,
    and behavior extraction functions for all CSV files in a set 
    """
    
    exptNameList = ['TwoWeek2023', 'CA2024', 'Solitary_Cohoused_March2024']
    
    # Ask the user to indicate the experiment name, constrained 
    exptName = input("\n\nChoose a value for exptName (options: {}): ".format(', '.join(exptNameList)))
    # Check if the user's choice is in the list
    while exptName not in exptNameList:
        print("Invalid choice. Choose a value of exptName from the list.")
        exptName = input("Choose a value for exptName (options: {}): ".format(', '.join(exptNameList)))
    
    showAllPositions = False
    
    params, CSVcolumns = defineParameters()
    
    fps, arena_radius_mm, imageScaleLocation, imageScaleColumn, \
        arenaCentersLocation, arenaCentersColumns, \
        offsetPositionsFilename = define_imageParameters(exptName)
    
    cwd = os.getcwd() # Current working directory
    
    data_path, allCSVfileNames = get_CSV_folder_and_filenames() # Get folder containing CSV files
    print(f'\n\n All {len(allCSVfileNames)} CSV files starting with "results": ')
    print(allCSVfileNames)
    
    print('\nEnter the filename for an output pickle file (w/ all datasets).')
    pickleFileName = input('   Will append .pickle. Leave blank for none.: ')
    
    # Number of datasets
    N_datasets = len(allCSVfileNames)
    
    # initialize a list of dictionaries for datasets
    datasets = [{} for j in range(N_datasets)]
    os.chdir(data_path)

    # For each dataset, get general properties and load all position data
    for j, CSVfileName in enumerate(allCSVfileNames):
        datasets[j]["CSVfilename"] = CSVfileName
        datasets[j]["dataset_name"] = get_dataset_name(CSVfileName)
        datasets[j]["image_scale"] = float(get_imageScale(datasets[j]["dataset_name"], 
                                                    imageScaleLocation, imageScaleColumn))
        # Load all the position information as a numpy array
        print('Loading dataset: ', datasets[j]["dataset_name"])
        datasets[j]["all_data"], datasets[j]["frameArray"] = \
            load_data(CSVfileName, CSVcolumns["N_columns"]) 
        datasets[j]["Nframes"] = len(datasets[j]["frameArray"])
        print('   ', 'Number of frames: ', datasets[j]["Nframes"] )
        datasets[j]["total_time_seconds"] = (np.max(datasets[j]["frameArray"]) - \
            np.min(datasets[j]["frameArray"]) + 1.0) / fps
        print('   ', 'Total duration: ', datasets[j]["total_time_seconds"], 'seconds')
        
        # Get arena center, subtracting image position offset
        datasets[j]["arena_center"] = get_ArenaCenter(datasets[j]["dataset_name"], 
                                                    arenaCentersLocation,
                                                    arenaCentersColumns, 
                                                    offsetPositionsFilename)
        # Estimate center location of Arena
        # datasets[j]["arena_center"] = estimate_arena_center(datasets[j]["all_data"],
        #                                                    CSVcolumns["pos_data_column_x"],
        #                                                    CSVcolumns["pos_data_column_y"])
    
        # Show all head positions, and arena center, and dish edge. 
        #    (& close threshold)
        if showAllPositions:
            plotAllPositions(datasets[j], CSVcolumns, arena_radius_mm, 
                             params["arena_edge_threshold_mm"])

    # For each dataset, identify close-to-edge and bad-tracking frames
    for j in range(N_datasets):
        print('Dataset: ', datasets[j]["dataset_name"])
        # Identify frames in which one or both fish are too close to the edge
        # First keep as an array, then convert into a dictionary that includes
        #    durations of edge events, etc.
        edge_frames = get_edge_frames(datasets[j], params, 
                                                     arena_radius_mm, 
                                                     CSVcolumns["pos_data_column_x"],
                                                     CSVcolumns["pos_data_column_y"])
        
        datasets[j]["edge_frames"] = make_frames_dictionary(edge_frames, ())
        
        # print(datasets[j]["edge_frames"])
        print('   Number of edge frames: ', len(datasets[j]["edge_frames"]["raw_frames"]))
        
        # Identify frames in which tracking is bad; separately consider head, body
        # Note that body is the most general of these -- use this for criteria
        bad_headTrack_frames = get_bad_headTrack_frames(datasets[j], params, 
                                     CSVcolumns["pos_data_column_x"],
                                     CSVcolumns["pos_data_column_y"], 0.001)
        datasets[j]["bad_headTrack_frames"] = make_frames_dictionary(bad_headTrack_frames, ())
        print('   Number of bad head tracking frames: ', len(datasets[j]["bad_headTrack_frames"]["raw_frames"]))
        bad_bodyTrack_frames = get_bad_bodyTrack_frames(datasets[j], params, 
                                     CSVcolumns["body_column_x_start"],
                                     CSVcolumns["body_column_y_start"],
                                     CSVcolumns["body_Ncolumns"], 0.001)
        datasets[j]["bad_bodyTrack_frames"] = make_frames_dictionary(bad_bodyTrack_frames, ())
        print('   Number of bad body tracking frames: ', len(datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))


    # For each dataset, characterizations that involve single fish
    # (e.g. length, bending)
    for j in range(N_datasets):
        print('Single-fish characterizations for Dataset: ', 
                  datasets[j]["dataset_name"])
        # Get the length of each fish in each frame (sum of all segments)
        # Nframes x 2 array
        datasets[j]["fish_length_array"] = \
            get_fish_lengths(datasets[j]["all_data"], CSVcolumns)
            
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
        # the duration of bends should be 1
        datasets[j]["Cbend_Fish0"] = make_frames_dictionary(Cbend_frames_each[0],
                                      (datasets[j]["edge_frames"]["raw_frames"],
                                       datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
        datasets[j]["Cbend_Fish1"] = make_frames_dictionary(Cbend_frames_each[1],
                                      (datasets[j]["edge_frames"]["raw_frames"],
                                       datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
        datasets[j]["Cbend_any"] = make_frames_dictionary(Cbend_frames_any,
                                      (datasets[j]["edge_frames"]["raw_frames"],
                                       datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
        datasets[j]["Jbend_Fish0"] = make_frames_dictionary(Jbend_frames_each[0],
                                      (datasets[j]["edge_frames"]["raw_frames"],
                                       datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
        datasets[j]["Jbend_Fish1"] = make_frames_dictionary(Jbend_frames_each[1],
                                      (datasets[j]["edge_frames"]["raw_frames"],
                                       datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
        datasets[j]["Jbend_any"] = make_frames_dictionary(Jbend_frames_any,
                                      (datasets[j]["edge_frames"]["raw_frames"],
                                       datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
        
            
    # For each dataset, identify inter-fish distance and
    # sliding window cross-correlation of heading angles
    for j in range(N_datasets):

        # Get the inter-fish distance (distance between head positions) in 
        # each frame (Nframes x 1 array)
        datasets[j]["inter-fish_distance"] = \
            get_interfish_distance(datasets[j]["all_data"], CSVcolumns)
        
        # Get the sliding window cross-correlation of heading angles
        datasets[j]["xcorr_array"] = \
            calcOrientationXCorr(datasets[j]["all_data"], CSVcolumns, 
                                 params["angle_xcorr_windowsize"])

        
    # For each dataset, exclude bad tracking frames from calculations of:
    # the mean inter-fish distance, 
    # the mean fish length, 
    # the mean and std. absolute difference in fish length, 
    for j in range(N_datasets):
        print('Removing bad frames from stats for length, distance for Dataset: ', 
              datasets[j]["dataset_name"])
        goodIdx = np.where(np.in1d(datasets[j]["frameArray"], 
                                   datasets[j]["bad_bodyTrack_frames"]["raw_frames"], 
                                   invert=True))[0]
        goodLengthArray = datasets[j]["fish_length_array"][goodIdx]
        goodDistanceArray = datasets[j]["inter-fish_distance"][goodIdx]
        datasets[j]["fish_length_mean"] = np.mean(goodLengthArray)
        datasets[j]["fish_length_Delta_mean"] = np.mean(np.abs(np.diff(goodLengthArray, 1)))
        datasets[j]["fish_length_Delta_std"] = np.std(np.abs(np.diff(goodLengthArray, 1)))
        datasets[j]["inter-fish_distance_mean"] = np.mean(goodDistanceArray)
        print(f'   Mean fish length: {datasets[j]["fish_length_mean"]:.2f} px')
        print('   Mean +/- std. of difference in fish length: ', 
              f'{datasets[j]["fish_length_Delta_mean"]:.2f} +/- {datasets[j]["fish_length_Delta_std"]:.2f}px')
        print(f'   Mean inter-fish distance {datasets[j]["inter-fish_distance_mean"]:.2f} px')
    
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
            
            perpendicular_noneSee_frames, \
                    perpendicular_oneSees_frames, \
                    perpendicular_bothSee_frames, \
                    perpendicular_larger_fish_sees_frames, \
                    perpendicular_smaller_fish_sees_frames, \
                    contact_any_frames, \
                    contact_head_body_frames, \
                    contact_larger_fish_head, contact_smaller_fish_head, \
                    contact_inferred_frames, tail_rubbing_frames, \
                    approaching_frames, fleeing_frames \
                    = extract_behaviors(datasets[j], params, CSVcolumns, fps)
            # removed "circling_frames," from the list
            
            # For each behavior, a dictionary containing frames, 
            # frames with "bad" elements removed
            # and a 2xN array of initial frames and durations
            # I could replace this with a loop through a list of keys, 
            # like below, but I'd have to include the names of variables like 
            # "contact_head_body_frames," so it wouldn't be much more elegant.
            datasets[j]["perpendicular_noneSee"] = make_frames_dictionary(perpendicular_noneSee_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["perpendicular_oneSees"] = make_frames_dictionary(perpendicular_oneSees_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["perpendicular_bothSee"] = make_frames_dictionary(perpendicular_bothSee_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["perpendicular_bothSee"] = make_frames_dictionary(perpendicular_bothSee_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["perpendicular_larger_fish_sees"] = make_frames_dictionary(perpendicular_larger_fish_sees_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["perpendicular_smaller_fish_sees"] = make_frames_dictionary(perpendicular_smaller_fish_sees_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["contact_any"] = make_frames_dictionary(contact_any_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["contact_head_body"] = make_frames_dictionary(contact_head_body_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["contact_larger_fish_head"] = make_frames_dictionary(contact_larger_fish_head,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["contact_smaller_fish_head"] = make_frames_dictionary(contact_smaller_fish_head,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["contact_inferred"] = make_frames_dictionary(contact_inferred_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["tail_rubbing"] = make_frames_dictionary(tail_rubbing_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            # For approaching and fleeing, again a dictionary of frames,
            # for each fish. approaching_Fish0 means Fish0 is approaching Fish 1
            datasets[j]["approaching_Fish0"] = make_frames_dictionary(approaching_frames[0],
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["approaching_Fish1"] = make_frames_dictionary(approaching_frames[1],
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["fleeing_Fish0"] = make_frames_dictionary(fleeing_frames[0],
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
            datasets[j]["fleeing_Fish1"] = make_frames_dictionary(fleeing_frames[1],
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))


            # delete "circling"
            # datasets[j]["circling"] = make_frames_dictionary(circling_frames,
            #                               (datasets[j]["edge_frames"]["raw_frames"],
            #                                datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))

    # Write pickle file containing all datasets
    if pickleFileName != '':
        list_for_pickle = [datasets, CSVcolumns, fps, arena_radius_mm, 
                           params]
        pickleFileName = pickleFileName + '.pickle'
        print(f'\nWriting pickle file: {pickleFileName}\n')
        with open(pickleFileName, 'wb') as handle:
            pickle.dump(list_for_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    # Write to individual text files, individual Excel sheets, and summary CSV file
    allDatasets_markFrames_ExcelFile = 'behaviors_in_each_frame.xlsx'
    markFrames_workbook = xlsxwriter.Workbook(allDatasets_markFrames_ExcelFile)  
    allDatasetsCSVfileName = 'behavior_count.csv' 
    print('File for collecting all behavior counts: ', allDatasetsCSVfileName)
    with open(allDatasetsCSVfileName, "w", newline='') as results_file:
        # behaviors (events) to write
        key_list = ["perpendicular_noneSee", 
                    "perpendicular_oneSees", "perpendicular_bothSee", 
                    "perpendicular_larger_fish_sees", 
                    "perpendicular_smaller_fish_sees", 
                    "contact_any", "contact_head_body", 
                    "contact_larger_fish_head", "contact_smaller_fish_head", 
                    "contact_inferred", "tail_rubbing", 
                    "Cbend_Fish0", "Cbend_Fish1", "Jbend_Fish0", "Jbend_Fish1",
                    "approaching_Fish0", "approaching_Fish1", 
                    "fleeing_Fish0", "fleeing_Fish1", 
                    "edge_frames", "bad_bodyTrack_frames"]
        # removed "circling"
        
        # For summary file
        writer=csv.writer(results_file, delimiter=',')
        writer.writerow(['Dataset', 'Total Time (s)', 
                         'Mean difference in fish lengths (px)', 
                         'Mean Inter-fish dist (px)', 
                         'Angle XCorr mean', 
                         # 'Angle XCorr std dev', 
                         # 'Angle XCorr skew', 
                         '90deg-None N_Events', 
                         '90deg-One N_Events', '90deg-Both N_Events', 
                         '90deg-largerSees N_events', '90deg-smallerSees N_events', 
                         'Contact (any) N_Events', 
                         'Contact (head-body) N_Events', 
                         'Contact (Larger fish head-body) N_Events', 
                         'Contact (Smaller fish head-body) N_Events', 
                         'Contact (inferred) N_events', 'Tail-Rub N_Events', 
                         'Cbend Fish0 N_Events', 'Cbend Fish1 N_Events',
                         'Jbend Fish0 N_Events', 'Jbend Fish1 N_Events',
                         'Fish0 Approaches N_Events', 'Fish1 Approaches N_Events',
                         'Fish0 Flees N_Events', 'Fish1 Flees N_Events',
                         'Dish edge N_Events', 'Bad tracking N_events', 
                         '90deg-None Duration', 
                         '90deg-One Duration', '90deg-Both Duration', 
                         '90deg-largerSees Duration', '90deg-smallerSees Duration', 
                         'Contact (any) Duration', 
                         'Contact (head-body) Duration', 
                         'Contact (Larger fish head-body) Duration', 
                         'Contact (Smaller fish head-body) Duration', 
                         'Contact (inferred) Duration', 'Tail-Rub Duration', 
                         'Cbend Fish0 Duration', 'Cbend Fish1 Duration', 
                         'Jbend Fish0 Duration', 'Jbend Fish1 Duration', 
                         'Fish0 Approaches Duration', 'Fish1 Approaches Duration',
                         'Fish0 Flees Duration', 'Fish1 Flees Duration',
                         'Dish edge Duration', 'Bad Tracking Duration'])
        # removed 'Circling N_Events', 'Circling Duration', 
        
        for j in range(N_datasets):
            
            # Write for this dataset: summary in text file
            write_behavior_txt_file(datasets[j], key_list)

            # Write for this dataset: sheet in Excel marking all frames with behaviors
            mark_behavior_frames_Excel(markFrames_workbook, datasets[j], 
                                       key_list)

            # Append information to the CSV describing all datasets
            writer = csv.writer(results_file)
            list_to_write = [datasets[j]["dataset_name"]]
            list_to_write.append(datasets[j]["total_time_seconds"])
            list_to_write.append(datasets[j]["fish_length_Delta_mean"])
            list_to_write.append(datasets[j]["inter-fish_distance_mean"])
            list_to_write.append(datasets[j]["AngleXCorr_mean"])
            # list_to_write.append(datasets[j]["AngleXCorr_std"])
            # list_to_write.append(datasets[j]["AngleXCorr_skew"])
            for k in key_list:
                list_to_write.append(datasets[j][k]["combine_frames"].shape[1])
            for k in key_list:
                list_to_write.append(datasets[j][k]["total_duration"])
            writer.writerow(list_to_write)                 
        
        # Close the Excel file
        markFrames_workbook.close() 
        # Return to original directory
        os.chdir(cwd)
    


    
if __name__ == '__main__':
    main()
