# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/19/2022
# version ='1.0'
# last modified: Raghuveer Parthasarathy, May 23, 2023
# ---------------------------------------------------------------------------
import re
import csv
from circling import *
from ninety_deg import *
from contact import *
from tail_rubbing import *
from visualize_behaviors import *

# ---------------------------------------------------------------------------
    
params = {
    "circ_ws" : 10,
    "circ_rmse" : 25,
    "circ_head_dist": 150,
    "circ_anti_low" : -1,
    "circ_anti_high" : -0.9,
    "90_ws" : 10,
    "theta_90_thresh" : 0.1,
    "90_head_dist" : 300,
    "contact_ws" : 1,
    "contact_dist" : 10,
    "tail_rub_ws" : 2,
    "tail_dist" : 30,
    "tail_rub_head_dist": 150,
    "tail_anti_low": -1,
    "tail_anti_high": -0.8,
}

def main():
    """
    Main function for calling the behavior extraction functions
    for all CSV files in a set 
    """
    
    cwd = os.getcwd() # Current working directory
    folder_path, allCSVfileNames = get_CSV_folder_and_filenames() # Get folder containing CSV files
    allDatasetsCSVfileName = 'behavior_count.csv'
    print('\n\n All CSV files starting with "results": ')
    print(allCSVfileNames)
    
    os.chdir(folder_path)
    print('File for collecting all behavior counts: ', allDatasetsCSVfileName)
    makeDiagram = False
    with open(allDatasetsCSVfileName, "w", newline='') as results_file:
        writer=csv.writer(results_file, delimiter=',')
        writer.writerow(['Dataset', 'Circling N_Events', '90deg-None N_Events', \
                         '90deg-One N_Events', '90deg-Both N_Events', 'Contact (any) N_Events', \
                         'Contact (head-body) N_Events', 'Tail-Rub N_Events', \
                         'Circling Duration', '90deg-None Duration', \
                         '90deg-One Duration', '90deg-Both Duration', 'Contact (any) Duration', \
                         'Contact (head-body) Duration', 'Tail-Rub Duration'])
        for CSVfileName in allCSVfileNames:
            # Examine each CSV file of trajectories
            # Identify events
            # Each behavior event detection function returns a 2xN array 
            #    of frames and durations
            print('Filename: ', CSVfileName)
            dataset_name, circling_wfs, none, one, both, \
                contact_any, head_body, tail_rubbing_wf = \
                    extract_behaviors(CSVfileName, makeDiagram)
            
            # Write output files for this dataset (only)
            singleDataset_CSVoutput(dataset_name, circling_wfs, none, 
                                    one, both, contact_any, head_body, 
                                    tail_rubbing_wf, makeDiagram)
            
            # Append to the CSV describing all datasets
            writer = csv.writer(results_file)
            writer.writerow([dataset_name, \
                             circling_wfs.shape[1], \
                             none.shape[1], one.shape[1], both.shape[1], \
                             contact_any.shape[1], head_body.shape[1], \
                             tail_rubbing_wf.shape[1], \
                             np.sum(circling_wfs[1,:]), \
                             np.sum(none[1,:]), np.sum(one[1,:]), \
                             np.sum(both[1,:]), \
                             np.sum(contact_any[1,:]), \
                             np.sum(head_body[1,:]), \
                             np.sum(tail_rubbing_wf[1,:])])
    
    os.chdir(cwd)
    
def extract_behaviors(CSVfileName, makeDiagram=True): 
    """
    Function for finding and displaying window frames 
    corresponding to all the different social behaviors.
    
    Inputs:
        CSVfileName : string, CSV file name with tracking data,
                      for example 'results_SocPref_3c_2wpf_k2_ALL.csv'
        makeDiagram : if true, make a figure (diagram)
    Outputs:
        dataset_name: name of the data set (based on CSVfileName input)
        arrays of all initial frames (row 1) and durations (row 2)
                     in which the various behaviors are 
                     found: circling_wfs, none, one, both, contact_any, 
                     head_body, tail_rubbing_wf

    """
    
    pos_data = load_data(CSVfileName, 3, 5)
    angle_data = load_data(CSVfileName, 5, 6)
    contact_x = load_data(CSVfileName, 6, 16)
    contact_y = load_data(CSVfileName, 16, 26)

    fish1_pos = pos_data[0]
    fish2_pos = pos_data[1]
    fish1_angle_data = angle_data[0]
    fish2_angle_data = angle_data[1]
    fish1_contact_x = contact_x[0]
    fish2_contact_x = contact_x[1]
    fish1_contact_y = contact_y[0]
    fish2_contact_y = contact_y[1]

    # End of array should be the same for all loaded data
    end_of_arr = np.shape(pos_data)[1] 
    dataset_regex = re.search('\d[a-z]_\d[a-z]{3}_[a-z]{1,2}[0-9]{1,2}', 
         CSVfileName)
    if dataset_regex is None:
        # try this instead to ignore "2wpf_" for example
        dataset_regex = re.search('\d[a-z]_[a-z]{1,2}[0-9]{1,2}',
                              CSVfileName)
    dataset_name = dataset_regex.group()

    # Circling 
    circling_wfs = get_circling_wf(fish1_pos, fish2_pos, fish1_angle_data, 
    fish2_angle_data, end_of_arr, params["circ_ws"], 
    params["circ_rmse"], params["circ_anti_high"],
    params["circ_head_dist"])
    
    # 90-degrees 
    orientation_dict = get_90_deg_wf(fish1_pos, fish2_pos, fish1_angle_data, 
    fish2_angle_data, end_of_arr, params["90_ws"], params["theta_90_thresh"], 
    params["90_head_dist"])
    none = orientation_dict["none"]
    one = orientation_dict["1"]
    both = orientation_dict["both"]
 
    # Any contact
    contact_wf = get_contact_wf(fish1_contact_x, fish2_contact_x, 
    fish1_contact_y, fish2_contact_y, end_of_arr, params["contact_ws"], 
    params["contact_dist"])
    contact_any = contact_wf["any_contact"]
    head_body = contact_wf["head-body"]

    # Tail-rubbing
    tail_rubbing_wf = get_tail_rubbing_wf(contact_x[0], contact_x[1], 
    contact_y[0], contact_y[1], fish1_pos, fish2_pos, fish1_angle_data, 
    fish2_angle_data, end_of_arr, params["tail_rub_ws"], params["tail_dist"], 
    params["tail_anti_high"], params["tail_rub_head_dist"])


    return dataset_name, circling_wfs, none, one, both, \
        contact_any, head_body, tail_rubbing_wf


def singleDataset_CSVoutput(dataset_name, circling_wfs, none, one, both, 
                            contact_any, head_body, tail_rubbing_wf, 
                            makeDiagram=True): 
    """
    Function for finding and displaying window frames 
    corresponding to all the different social behaviors.
    
    Inputs:
        dataset_name : string, CSV file name to output. Should be 
                       created by extract_behaviors()
        arrays of all initial frames (row 1) and durations (row 2)
                     in which the various behaviors are 
                     found: circling_wfs, none, one, both, contact_any, 
                     head_body, tail_rubbing_wf
        makeDiagram : if true, make a figure (diagram)
    Outputs:
        None

    """
    
    get_txt_file(dataset_name, circling_wfs, none, one, both, contact_any, 
    head_body, tail_rubbing_wf)
    if makeDiagram:
        get_diagram(dataset_name, circling_wfs, none, one, both, contact_any, 
                    head_body, tail_rubbing_wf)
    get_excel_file(dataset_name, circling_wfs, none, one, both, contact_any, 
    head_body, tail_rubbing_wf)


    
if __name__ == '__main__':
    main()
