# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/7/2022
# version ='1.0'
# ---------------------------------------------------------------------------
from toolkit import *
# ---------------------------------------------------------------------------

def get_90_deg_wf(fish1_pos, fish2_pos, fish1_angle_data, fish2_angle_data, 
end, window_size, theta_90_thresh, head_dist):
    """
    Returns an array of window frames for 90-degree orientation events.
    Each window frame represents the ENDING window frame for 90-degree
    orientation events within some range of window frames specified by 
    the parameter window_size.

    Args:
        fish1_pos (array): a 2D array of (x, y) positions for fish1. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish2_pos (array): a 2D array of (x, y) positions for fish2. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish1_angle_data (array): a 1D array of angles at each window frame
                                  for fish1.
        fish2_angle_data (array): a 1D array of angles at each window frame
                                  for fish2.

        end (int): end of the array for both fish (typically 15,000 window frames.)
        
        window_size (int)      : window size for which circling is averaged over.
        theta_90_thresh (float): the angle threshold for 90-degree orientation.
        head_dist_thresh (int) : head distance threshold for the two fish.

    Returns:
        orientations (dict): a dictionary of arrays of window frames for different 
                             90-degree orientation types:
                      
                      - "none": none of the fish see each other.
                      - "1"   : one fish sees the other.
                      - "both": both fish see each other.
    """
    idx_1, idx_2 = 0, window_size
    orientations = {"none": [], 
                    "1": [], 
                    "both": []}

    while idx_2 <= end:   # end of the array for both fish
        # Get position and angle data for x window frames 
        fish1_positions = fish1_pos[idx_1:idx_2]
        fish2_positions = fish2_pos[idx_1:idx_2]
        fish1_angles = fish1_angle_data[idx_1:idx_2]
        fish2_angles = fish2_angle_data[idx_1:idx_2]
       
        theta_90 = np.abs(get_cos_angle(fish1_angles, fish2_angles))

        fish_vectors = get_fish_vectors(fish1_angles, fish2_angles)
        fish1_vector, fish2_vector = fish_vectors[0], fish_vectors[1]
        connecting_vector = get_connecting_vector(fish1_positions, fish2_positions)

        # Calculate signs of cross product of fish vectors 
        # to determine orientation type 
        fish1xfish2 = np.sign(np.cross(fish1_vector, fish2_vector))
        fish1xconnect = np.sign(np.cross(fish1_vector, connecting_vector))
        fish2xconnect = np.sign(np.cross(fish2_vector, connecting_vector))
        orientation_type = get_orientation_type((fish1xfish2, fish1xconnect,
        fish2xconnect))
        
        if (theta_90 < theta_90_thresh and orientation_type in orientations.keys() 
        and get_head_distance(fish1_positions, fish2_positions) < head_dist):
            orientations[orientation_type].append(idx_2+1) 
        
        # Update the index variables to track 90-degree events 
        # for the next x window frames of size window_size
        idx_1 += 1
        idx_2 += 1

    orientations["none"] = combine_events(np.array(orientations["none"]))
    orientations["1"] = combine_events(np.array(orientations["1"]))
    orientations["both"] = combine_events(np.array(orientations["both"]))
    return orientations


def get_connecting_vector(fish1_positions, fish2_positions):
    """
    Returns the "c" vector between two fish.

    Args:
        fish1_positions (array): a 2D array of (x, y) positions for fish1 over
                                 some number of window frames. The array has 
                                 form [[x1, y1], [x2, y2], [x3, y3],...].

        fish2_positions (array): a 2D array of (x, y) positions for fish2 over
                                 some number of window frames. The array has 
                                 form [[x1, y1], [x2, y2], [x3, y3],...].
       
    Returns:
        normalized_vector (array): the normalized "c" vector between two fish.
    """
    connecting_vector = np.array((np.mean(fish2_positions[:, 0] - fish1_positions[:, 0]), 
    np.mean(fish2_positions[:, 1] - fish1_positions[:, 1])))
    normalized_vector = connecting_vector / np.linalg.norm(connecting_vector)
    return normalized_vector
    

def get_orientation_type(sign_tuple):
    """
    Returns the orientation type of two fish
    given the sign of their respective 
    (a, b, c) vectors.

    Args:
        orientation_tuple (tuple): a tuple of signs between two fish.
    
    Returns:
        (str): "none", "one", or "both.

    """
    # Orientations grouped according to the sign
    # of their respective cross products
    switcher = {
        (1,1,1)   : "none",
        (-1,-1,-1): "none",
        (-1,1,1)  : "none",
        (1,-1,1)  : "both",
        (1,1,-1)  : "1",
        (-1,-1,1) : "1",
        (1,-1,-1) : "both",
        (-1,1,-1) : "1"
    }
    return switcher.get(sign_tuple)
