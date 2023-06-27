# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/7/2022
# version ='2.0': Major modifications by Raghuveer Parthasarathy, May-June 2023
# last modified: Raghuveer Parthasarathy June 10, 2023
# ---------------------------------------------------------------------------
from toolkit import *
# ---------------------------------------------------------------------------

def get_90_deg_wf(fish1_pos, fish2_pos, fish1_angle_data, fish2_angle_data, 
Nframes, window_size, cos_theta_90_thresh, head_dist_thresh):
    """
    Returns an array of frames for 90-degree orientation events.
    Each frame represents the starting  frame for 90-degree
       orientation events that span some window (parameter window_size).

    Args:
        fish1_pos (array): a 2D array of (x, y) head positions for fish1. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish2_pos (array): a 2D array of (x, y) head positions for fish2. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish1_angle_data (array): a 1D array of angles at each window frame
                                  for fish1.
        fish2_angle_data (array): a 1D array of angles at each window frame
                                  for fish2.

        Nframes (int): Number of frames (typically 15,000.)
        
        window_size (int)      : window size for which circling is averaged over.
        cos_theta_90_thresh (float): the cosine(angle) threshold for 90-degree orientation.
        head_dist_thresh (int) : head distance threshold for the two fish.

    Returns:
        orientations (dict): a dictionary of arrays of window frames for different 
                             90-degree orientation types:
                      
                      - "noneSee": none of the fish see each other.
                      - "oneSees"   : one fish sees the other.
                      - "bothSee": both fish see each other.
    """
    idx_1, idx_2 = 0, window_size
    orientations = {"noneSee": [], 
                    "oneSees": [], 
                    "bothSee": []}
    
    # cos_theta for all frames
    cos_theta = np.cos(fish1_angle_data - fish2_angle_data)
    cos_theta_criterion = (np.abs(cos_theta) < cos_theta_90_thresh)
    
    # head-head distance, and distance vector for all frames
    dh_vec = fish2_pos - fish1_pos  # also used later, for the connecting vector
    head_separation = np.sqrt(np.sum(dh_vec**2, axis=1))
    head_separation_criterion = (head_separation < head_dist_thresh)

    # All criteria (and), in each frame
    all_criteria_frame = np.logical_and(cos_theta_criterion, head_separation_criterion)
    all_criteria_window = np.zeros(all_criteria_frame.shape, dtype=bool) # initialize to false
    # Check that criteria are met through the frame window. 
    # Will loop rather than doing some clever Boolean product of offset arrays
    for j in range(all_criteria_frame.shape[0]-window_size+1):
        all_criteria_window[j] =  all_criteria_frame[j:j+window_size].all()
    
    # Indexes (frames - 1) where the criteria are met throughout the window
    ninety_degree_idx = np.array(np.where(all_criteria_window==True))[0,:].flatten() + 1
    # Not sure why the [0,:] is needed, but otherwise returns additional zeros.

    # For each 90 degree event, calculate sign of cross product of fish vectors 
    # for the starting frame, to determine orientation type 
    # Could have done this for all frames and just kept those that met 
    # the above criteria; not sure which is faster
    for idx in ninety_degree_idx:
        fish1_vector = np.array((np.cos(fish1_angle_data[idx]), np.sin(fish1_angle_data[idx])))
        fish2_vector = np.array((np.cos(fish2_angle_data[idx]), np.sin(fish2_angle_data[idx])))
        connecting_vector = dh_vec[idx]
        connecting_vector_norm = connecting_vector / np.linalg.norm(connecting_vector)
        
        # signs of cross products
        fish1xfish2 = np.sign(np.cross(fish1_vector, fish2_vector))
        fish1xconnect = np.sign(np.cross(fish1_vector, connecting_vector_norm))
        fish2xconnect = np.sign(np.cross(fish2_vector, connecting_vector_norm))
        orientation_type = get_orientation_type((fish1xfish2, fish1xconnect,
                                                 fish2xconnect))
        if not(orientation_type is None):
            # make sure it's not "None," for example from positions==0
            orientations[orientation_type].append(idx+1) 


    orientations["noneSee"] = np.array(orientations["noneSee"])
    orientations["oneSees"] = np.array(orientations["oneSees"])
    orientations["bothSee"] = np.array(orientations["bothSee"])
    return orientations


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
