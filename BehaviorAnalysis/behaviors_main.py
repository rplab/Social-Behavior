# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/19/2022
# version ='2.0'
# last modified: Raghuveer Parthasarathy, June 23, 2023
# ---------------------------------------------------------------------------

import csv
import os
import numpy as np
import xlsxwriter
from time import perf_counter
from toolkit import *
from circling import *
from ninety_deg import *
from contact import *
from tail_rubbing import *
from visualize_behaviors import *

# ---------------------------------------------------------------------------

def defineParameters():
    """ 
    Defines parameters for behavior identification, sets filenames 
    for arena coordinates, identifies columns of trajectory CSV files
    """

    fps = 25.0  # frames per second
    arena_radius_mm = 25.0  # arena radius

    params = {
        "arena_edge_threshold_mm" : 5,
        "circ_windowsize" : 25,
        "circle_fit_threshold" : 0.25,
        "circ_head_dist": 240,
        "cos_theta_AP_threshold" : -0.7,
        "cos_theta_tangent_threshold" : 0.34,
        "motion_threshold" : 2.0,
        "90_ws" : 4,
        "cos_theta_90_thresh" : 0.17,
        "90_head_dist" : 300,
        "contact_distance" : 10,
        "contact_inferred_window" : 3,
        "tail_rub_ws" : 2,
        "tail_dist" : 30,
        "tail_rub_head_dist": 220,
        "tail_anti_low": -1,
        "tail_anti_high": -0.8,
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
    
    # Image scale, read from file
    imageScalePathName = 'C:/Users/Raghu/Documents/Experiments and Projects/Misc/Zebrafish behavior'
    imageScaleFilename = 'ArenaCenters_SocPref_3456.csv'
    imageScaleLocation = os.path.join(imageScalePathName, imageScaleFilename)
    
    # Arena center locations -- if these are tabulated elsewhere.
    arenaCentersPathName = 'C:/Users/Raghu/Documents/Experiments and Projects/Misc/Zebrafish behavior'
    arenaCentersFilename = 'ArenaCenters_SocPref_3456.csv'
    arenaCentersLocation = os.path.join(arenaCentersPathName, arenaCentersFilename)
    # filename of CSV file *in each data folder* with image offset filename
    offsetPositionsFilename = 'wellOffsetPositionsCSVfile.csv'

    return fps, arena_radius_mm, params, CSVcolumns, imageScaleLocation, \
        arenaCentersLocation, offsetPositionsFilename


def main():
    """
    Main function for calling reading functions, basic analysis functions,
    and behavior extraction functions for all CSV files in a set 
    """
    
    showAllPositions = False
    
    fps, arena_radius_mm, params, CSVcolumns, imageScaleLocation, \
        arenaCentersLocation, offsetPositionsFilename = defineParameters()
    
    cwd = os.getcwd() # Current working directory
    
    folder_path, allCSVfileNames = get_CSV_folder_and_filenames() # Get folder containing CSV files
    print(f'\n\n All {len(allCSVfileNames)} CSV files starting with "results": ')
    print(allCSVfileNames)
    
    # Number of datasets
    N_datasets = len(allCSVfileNames)
    
    # initialize a list of dictionaries for datasets
    datasets = [{} for j in range(N_datasets)]
    os.chdir(folder_path)

    # For each dataset, get general properties and load all position data
    for j, CSVfileName in enumerate(allCSVfileNames):
        datasets[j]["CSVfilename"] = CSVfileName
        datasets[j]["dataset_name"] = get_dataset_name(CSVfileName)
        datasets[j]["image_scale"] = np.float(get_imageScale(datasets[j]["dataset_name"], 
                                                    imageScaleLocation))
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
                                                    offsetPositionsFilename)
        # print([d['arena_center'] for d in datasets])  # All arena centers

        # Estimate center location of Arena
        # datasets[j]["arena_center"] = estimate_arena_center(datasets[j]["all_data"],
        #                                                    CSVcolumns["pos_data_column_x"],
        #                                                    CSVcolumns["pos_data_column_y"])
    
        # Show all head positions, and arena center, and dish edge. 
        #    (& close threshold)
        if showAllPositions:
            plotAllPositions(datasets[j], CSVcolumns, arena_radius_mm, 
                             params["arena_edge_threshold_mm"])

        # Get the inter-fish distance (distance between head positions) in 
        # each frame (Nframes x 1 array)
        datasets[j]["inter-fish_distance"] = \
            get_interfish_distance(datasets[j]["all_data"], CSVcolumns)
        
        # Get the length of each fish in each frame (sum of all segments)
        # Nframes x 2 array
        datasets[j]["fish_length_array"] = \
            get_fish_lengths(datasets[j]["all_data"], CSVcolumns)
        plotFishLengths = False
        if plotFishLengths:
            plt.figure()
            plt.plot(np.arange(datasets[j]["fish_length_array"].shape[0]), 
                     datasets[j]["fish_length_array"][:,0], c='magenta')
            plt.plot(np.arange(datasets[j]["fish_length_array"].shape[0]), 
                     datasets[j]["fish_length_array"][:,1], c='darkturquoise')
            plt.title(datasets[j]["dataset_name"] )

            
    # For each dataset, identify edge and bad-tracking frames
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

        
    # For each dataset, the mean inter-fish distance and fish length, 
    # not counting bad-tracking frames
    for j in range(N_datasets):
        print('Dataset: ', datasets[j]["dataset_name"])
        goodIdx = np.where(np.setdiff1d(datasets[j]["frameArray"], datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))
        goodLengthArray = datasets[j]["fish_length_array"][goodIdx]
        goodDistanceArray = datasets[j]["inter-fish_distance"][goodIdx]
        datasets[j]["fish_length_mean"] = np.mean(goodLengthArray)
        datasets[j]["inter-fish_distance_mean"] = np.mean(goodDistanceArray)
        print(f'   Mean fish length {datasets[j]["fish_length_mean"]:.2f} px')
        print(f'   Mean inter-fish distance {datasets[j]["inter-fish_distance_mean"]:.2f} px')
    
    # For each dataset, identify behaviors
    for j in range(N_datasets):
        
            print('Dataset name: ', datasets[j]["dataset_name"])
            
            circling_frames, noneSee90_frames, oneSees90_frames, \
                    bothSee90_frames, contact_any_frames, \
                    contact_head_body_frames, contact_inferred_frames, \
                    tail_rubbing_frames = \
                    extract_behaviors(datasets[j], params, CSVcolumns)
            
            # For each behavior, a dictionary containing frames, 
            # frames with "bad" elements removed
            # and a 2xN array of initial frames and durations
            # I could replace this with a loop through a list of keys, 
            # like below, but I'd have to include the names of strings like 
            # "circling frames," so it wouldn't be much more elegant.
            datasets[j]["circling"] = make_frames_dictionary(circling_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]))
            datasets[j]["noneSee90"] = make_frames_dictionary(noneSee90_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]))
            datasets[j]["oneSees90"] = make_frames_dictionary(oneSees90_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]))
            datasets[j]["bothSee90"] = make_frames_dictionary(bothSee90_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]))
            datasets[j]["contact_any"] = make_frames_dictionary(contact_any_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]))
            datasets[j]["contact_head_body"] = make_frames_dictionary(contact_head_body_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]))
            datasets[j]["contact_inferred"] = make_frames_dictionary(contact_inferred_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]))
            datasets[j]["tail_rubbing"] = make_frames_dictionary(tail_rubbing_frames,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]))


    # Write to individual text files, individual Excel sheets, and summary CSV file
    allDatasets_markFrames_ExcelFile = 'behaviors_in_each_frame.xlsx'
    markFrames_workbook = xlsxwriter.Workbook(allDatasets_markFrames_ExcelFile)  
    allDatasetsCSVfileName = 'behavior_count.csv' 
    print('File for collecting all behavior counts: ', allDatasetsCSVfileName)
    makeDiagram = False
    with open(allDatasetsCSVfileName, "w", newline='') as results_file:
        # behaviors to write
        key_list = ["circling", "noneSee90", "oneSees90", "bothSee90", 
                    "contact_any", "contact_head_body", "contact_inferred", 
                    "tail_rubbing", "edge_frames", "bad_bodyTrack_frames"]
        
        # For summary file
        writer=csv.writer(results_file, delimiter=',')
        writer.writerow(['Dataset', 'Total Time (s)', 
                         'Mean Inter-fish dist (px)', 
                         'Circling N_Events', '90deg-None N_Events', 
                         '90deg-One N_Events', '90deg-Both N_Events', 
                         'Contact (any) N_Events', 
                         'Contact (head-body) N_Events', 
                         'Contact (inferred) N_events', 'Tail-Rub N_Events', 
                         'Dish edge N_Events', 'Bad tracking N_events', 
                         'Circling Duration', '90deg-None Duration', 
                         '90deg-One Duration', '90deg-Both Duration', 
                         'Contact (any) Duration', 
                         'Contact (head-body) Duration', 
                         'Contact (inferred) Duration', 'Tail-Rub Duration', 
                         'Dish edge Duration', 'Bad Tracking Duration'])
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
            list_to_write.append(datasets[j]["inter-fish_distance_mean"])
            for k in key_list:
                list_to_write.append(datasets[j][k]["combine_frames"].shape[1])
            for k in key_list:
                list_to_write.append(datasets[j][k]["total_duration"])
            writer.writerow(list_to_write)                 
        
        # Close the Excel file
        markFrames_workbook.close() 
        # Return to original directory
        os.chdir(cwd)
    
def extract_behaviors(dataset, params, CSVcolumns): 
    """
    FuncFunction for identifying frames (or frame windows) 
    exhibiting corresponding to each behavior..
    
    Inputs:
        dataset : dictionary, with keys like "all_data" containing all 
                    position data
        params : parameters for behavior criteria
        CSVcolumns : CSV column parameters
    Outputs:
        arrays of all initial frames (row 1) and durations (row 2)
                     in which the various behaviors are 
                     found: circling_wfs, noneSee90, oneSees90, bothSee90, 
                     contact_any, contact_head_body, contact_inferred, 
                     tail_rubbing_wf

    """
    
    # Timer
    t1_start = perf_counter()

    # Arrays of head, body positions; angles. 
    # Last dimension = fish (so arrays are Nframes x {1 or 2}, Nfish==2)
    pos_data = dataset["all_data"][:,CSVcolumns["pos_data_column_x"]:CSVcolumns["pos_data_column_y"]+1, :]
        # pos_data is Nframes x 2 (x and y positions) x 2 (Nfish) array of head positions
    angle_data = dataset["all_data"][:,CSVcolumns["angle_data_column"], :]
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
        
    # Number of frames should be the same for all loaded data (); not checking
    Nframes = np.shape(pos_data)[0] 
        
    # Visualize fish positions in some range of frames (optional)
    makeVisualization = False
    if makeVisualization:
        startFrame = 3710  #3710 cirlcing in 3b_k2
        endFrame = startFrame+20
        visualize_fish(body_x, body_y, dataset["frameArray"], 
                       startFrame, endFrame, dataset["dataset_name"])

    t1_1 = perf_counter()
    print(f'   t1_1 start circling analysis: {t1_1 - t1_start:.2f} seconds')
    # Circling 
    circling_wfs = get_circling_wf(pos_data, dataset["inter-fish_distance"], 
                                   angle_data, Nframes, params["circ_windowsize"], 
                                   params["circle_fit_threshold"], 
                                   params["cos_theta_AP_threshold"], 
                                   params["cos_theta_tangent_threshold"], 
                                   params["motion_threshold"], 
                                   params["circ_head_dist"])
    
    t1_2 = perf_counter()
    print(f'   t1_2 start 90degree analysis: {t1_2 - t1_start:.2f} seconds')
    # 90-degrees 
    orientation_dict = get_90_deg_wf(pos_data[:,:,0], pos_data[:,:,1], angle_data[:,0], 
    angle_data[:,1], Nframes, params["90_ws"], params["cos_theta_90_thresh"], 
    params["90_head_dist"])
    noneSee90 = orientation_dict["noneSee"]
    oneSees90 = orientation_dict["oneSees"]
    bothSee90 = orientation_dict["bothSee"]
 
    t1_3 = perf_counter()
    print(f'   t1_3 start contact analysis: {t1_3 - t1_start:.2f} seconds')
    # Any contact, or head-body contact
    contact_dict = get_contact_wf(body_x[:,:,0], body_x[:,:,1], 
                                body_y[:,:,0], body_y[:,:,1], Nframes, 
                                params["contact_distance"])
    contact_any = contact_dict["any_contact"]
    contact_head_body = contact_dict["head-body"]
    contact_inferred_frames = get_inferred_contact_frames(dataset,
                        params["contact_inferred_window"],                                  
                        2.0*params["contact_distance"])

    t1_4 = perf_counter()
    print(f'   t1_4 start tail-rubbing analysis: {t1_4 - t1_start:.2f} seconds')
    # Tail-rubbing
    tail_rubbing_wf = get_tail_rubbing_wf(body_x, body_y, 
                                          dataset["inter-fish_distance"], 
                                          angle_data, 
                                          params["tail_rub_ws"], 
                                          params["tail_dist"], 
    params["tail_anti_high"], params["tail_rub_head_dist"])

    t1_5 = perf_counter()
    print(f'   t1_5 end analysis: {t1_5 - t1_start:.2f} seconds')

    return circling_wfs, noneSee90, oneSees90, bothSee90, \
        contact_any, contact_head_body, contact_inferred_frames, \
        tail_rubbing_wf


def singleDataset_output(dataset, key_list, makeDiagram=False): 
    """
    Export frames for various behaviors: summary to text file, and 
        frame-by-frame in a large CSV file
    
    Inputs:
        dataset : dictionary with all dataset info
        key_list : list of behavior names as keys to dataset dictionaries, 
                   for export
        arrays of all initial frames (row 1) and durations (row 2)
                     in which the various behaviors are 
                     found: circling_wfs, noneSee90, one, both, contact_any, 
                     contact_head_body, tail_rubbing_wf
        makeDiagram : if true, make a figure (diagram) [DISABLED]
    Outputs:
        None

    """
    
    write_behavior_txt_file(dataset_name, key_list)
    if makeDiagram:
        print('makeDiagram in singleDataset_output is disabled.')
        
    get_excel_file(dataset_name, circling_wfs, noneSee90, oneSees90, bothSee90, contact_any,
                   contact_head_body, tail_rubbing_wf)


    
if __name__ == '__main__':
    main()
