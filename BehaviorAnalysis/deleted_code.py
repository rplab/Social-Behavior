# -*- coding: utf-8 -*-
# deleted_code.py
"""
Author:   Raghuveer Parthasarathy
Created on Wed Jul  5 17:54:16 2023
Last modified on August 14, 2024

Description
-----------

Misc. deleted code

"""


#%% Circling

# In extract_behaviors:
    
    # CIRCLING
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
    
    params = {
        "arena_edge_threshold_mm" : 5,
        "circle_windowsize" : 25,
        "circle_fit_threshold" : 0.25,
        "circle_distance_threshold": 240,


#%%

def extract_single_fish_behaviors(dataset, params, CSVcolumns): 
    """
    Calls functions to identify frames corresponding to single-fish behaviors, 
    such as length or J-bends. 

    Inputs:
        dataset : dictionary, with keys like "all_data" containing all 
                    position data
        params : parameters for behavior criteria
        CSVcolumns : CSV column parameters
    Outputs:
        arrays of all frames in which the various behaviors are found:
            Cbend_frames,
            Jbend_frames

    """
    # C-bend
    Cbend_frames_each = get_Cbend_frames(dataset, CSVcolumns, 
                                    params["Cbend_threshold"])
    # numpy array of frames with C-bend for *any* fish
    Cbend_frames = np.unique(np.concatenate(list(Cbend_frames_each.values())))

    # J-bend
    Jbend_frames_each = get_Jbend_frames(dataset, CSVcolumns, 
                                    (params["Jbend_rAP"], 
                                     params["Jbend_cosThetaN"], 
                                     params["Jbend_cosThetaNm1"]))
    # numpy array of frames with J-bend for *any* fish
    Jbend_frames = np.unique(np.concatenate(list(Jbend_frames_each.values())))
    
    return Cbend_frames, Jbend_frames


## From get_90_deg_frames()
        # signs of cross products
        fish1xfish2 = np.sign(np.cross(fish1_vector, fish2_vector))
        fish1xconnect = np.sign(np.cross(fish1_vector, connecting_vector_norm))
        fish2xconnect = np.sign(np.cross(fish2_vector, connecting_vector_norm))
        orientation_type = get_orientation_type((fish1xfish2, fish1xconnect,
                                                 fish2xconnect))

        if orientation_type == "oneSees":
            dh_angle_12 = np.arctan2(connecting_vector[1], connecting_vector[0])
            dh_angle_21 = dh_angle_12 + np.pi
            fish1sees = np.cos(fish_angle_data[idx,0] - dh_angle_12)>0
            fish2sees = np.cos(fish_angle_data[idx,1] - dh_angle_21)>0
            if fish1sees and fish2sees:
                print(' ')
                print('Error in get_90_deg_frames()!')
                print(f'  fish 1 sees cosTheta = {fish1sees}\n')
                print(f'  fish 2 sees cosTheta = {fish2sees}\n')
                print('"oneSees" orientation, but both fish see (see code)')
                print('   fish1 angle degrees: ', fish_angle_data[idx,0]*180/np.pi)
                print('   fish2 angle degrees: ', fish_angle_data[idx,1]*180/np.pi)
                print('   dh_angle_12 degrees: ', dh_angle_12*180/np.pi)
                print('   dh_angle_21 degrees: ', dh_angle_21*180/np.pi)
                print('   fish 1 head position: ', fish_pos[idx,:,0])
                print('   fish 2 head position: ', fish_pos[idx,:,1])
                print('This should not happen. Enter to continue, or Control-C')
                input('--- ')
            else:
                print('fish 1 and fish 2 sees: ', fish1sees,' , ', fish2sees)
                largerFishIdx = np.argmax(fish_length_array[idx,:])
                if largerFishIdx==0 and fish1sees:
                    orientations["larger_fish_sees"].append(idx+1)
                else:
                    orientations["smaller_fish_sees"].append(idx+1)


def get_orientation_type(sign_tuple):
    """
    Returns the orientation type of two fish
    given the sign of their respective (a, b, c) vectors.

    Args:
        orientation_tuple (tuple): a tuple of signs of the cross-products
        between two fish:
            fish1xfish2, fish1xconnect, fish2xconnect
    
    Returns:
        (str): "noneSee", "oneSees", or "bothSee".
        
    DELETED July 5, 2023; see "Behavior Code Revisions July 2023,"
    Perpendicular Orientations section

    """
    # Orientations grouped according to the sign
    # of their respective cross products
    switcher = {
        (1,1,1)   : "noneSee",
        (-1,-1,-1): "noneSee",
        (-1,-1,1) : "oneSees",
        (1,1,-1)  : "oneSees",
        (-1,1,-1) : "oneSees",
        (1,-1,1)  : "oneSees",
        (-1,1,1)  : "bothSee",
        (1,-1,-1) : "bothSee"
    }
    return switcher.get(sign_tuple)


def mark_behavior_frames_Excel(markFrames_workbook, dataset, key_list):
    """
    Create and fill in sheet in Excel marking all frames with behaviors
    found in this dataset

    Args:
        markFrames_workbook : Excel workbook 
        dataset : dictionary with all dataset info
        key_list : list of dictionary keys corresponding to each behavior to write

    Returns:
        N/A
    """
    
    # Annoyingly, Excel won't allow a worksheet name that's
    # more than 31 characters! Force it to use the last 31.
    sheet_name = dataset["dataset_name"]
    sheet_name = sheet_name[-31:]
    sheet1 = markFrames_workbook.add_worksheet(sheet_name)
    ascii_uppercase = list(map(chr, range(65, 91)))
    
    # Headers 
    sheet1.write('A1', 'Frame') 
    for j, k in enumerate(key_list):
        sheet1.write(f'{ascii_uppercase[j+1]}1', k) 
        
    # All frame numbers
    maxFrame = int(np.max(dataset["frameArray"]))
    for j in range(1,maxFrame+1):
        sheet1.write(f'A{j+1}', str(j))

    # Each behavior
    for j, k in enumerate(key_list):
        for run_idx in  range(dataset[k]["combine_frames"].shape[1]):
            for duration_idx in range(dataset[k]["combine_frames"][1,run_idx]):
                sheet1.write(f'{ascii_uppercase[j+1]}{dataset[k]["combine_frames"][0,run_idx]+duration_idx+1}', 
                         "X".center(17))


def combine_all_speeds(datasets):
    """
    Loop through each dataset, get speed array values for all fish, 
    avoiding bad frames (dilated +1), collect all these in a list of 
    numpy arrays. One list per dataset (flattened across fish.)
    (Can concatenate into one numpy array with 
                   "np.concatenate()"). .
    For making a histogram of speeds

    Parameters
    ----------
    datasets : list of dictionaries containing all analysis. 

    Returns
    -------
    speeds_mm_s_all : list of numpy arrays of all speeds in all datasets

    """
    Ndatasets = len(datasets)
    speeds_mm_s_all = []
    for j in range(Ndatasets):
        frames = datasets[j]["frameArray"]
        badTrackFrames = datasets[j]["bad_bodyTrack_frames"]["raw_frames"]
        dilate_badTrackFrames = np.concatenate((badTrackFrames,
                                               badTrackFrames + 1))
        bad_frames_set = set(dilate_badTrackFrames) # faster lookup
        # Calculate mean speed excluding bad tracking frames
        good_frames_mask = np.isin(frames, list(bad_frames_set), invert=True)
        speeds_this_set = datasets[j]["speed_array_mm_s"][good_frames_mask, :].flatten()
        speeds_mm_s_all.append(speeds_this_set)
        
    return speeds_mm_s_all


def calculate_value_autocorr_oneSet(dataset, keyName='speed_array_mm_s', 
                                    dilate_plus1=True, t_max=10, 
                                    t_window=None):
    """
    For a *single* dataset, calculate the autocorrelation of the numerical
    property in the given key (e.g. speed)
    Ignore "bad tracking" frames. If "dilate_plus1" is True, dilate the bad frames +1.
    Output is a numpy array with dim 1 corresponding to each fish.
    
    Parameters
    ----------
    dataset : single analysis dataset
    keyName : the key to combine (e.g. "speed_array_mm_s")
    dilate_plus1 : If True, dilate the bad frames +1
    t_max : max time to consider for autocorrelation, seconds.
    t_window : size of sliding window in seconds. If None, don't use a sliding window.
    
    Returns
    -------
    autocorr_one : autocorrelation of desired property, numpy array of
                    shape (#time lags + 1 , Nfish)
    t_lag : time lag array, seconds (including zero)
    """
    value_array = dataset[keyName]
    Nframes, Nfish = value_array.shape
    fps = dataset["fps"]
    badTrackFrames = dataset["bad_bodyTrack_frames"]["raw_frames"]
    if dilate_plus1:
        dilate_badTrackFrames = np.concatenate((badTrackFrames, badTrackFrames + 1))
        bad_frames_set = set(dilate_badTrackFrames)
    else:
        bad_frames_set = set(badTrackFrames)
     
    t_lag = np.arange(0, t_max + 1.0/fps, 1.0/fps)
    n_lags = len(t_lag)
    
    autocorr = np.zeros((n_lags, Nfish))
    
    for fish in range(Nfish):
        fish_value = value_array[:, fish].copy()
        
        good_frames = [speed for i, speed in enumerate(fish_value) if i not in bad_frames_set]
        mean_value = np.mean(good_frames)
        std_value = np.std(good_frames)
        
        for frame in bad_frames_set:
            if frame < Nframes:
                fish_value[frame] = np.random.normal(mean_value, std_value)
        
        if t_window is None:
            fish_autocorr = calculate_autocorr(fish_value, n_lags)
        else:
            window_size = int(t_window * fps)
            fish_autocorr = calculate_block_autocorr(fish_value, n_lags, 
                                                     window_size)
        
        autocorr[:, fish] = fish_autocorr
    
    return autocorr, t_lag
