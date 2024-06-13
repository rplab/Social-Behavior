# -*- coding: utf-8 -*-
# deleted_code.py
"""
Author:   Raghuveer Parthasarathy
Created on Wed Jul  5 17:54:16 2023
Last modified on Nov. 21, 2023

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