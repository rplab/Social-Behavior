# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='1.0'
# ---------------------------------------------------------------------------
from math import sqrt
from toolkit import *
from circle_fit_taubin import TaubinSVD
# ---------------------------------------------------------------------------

def get_reciprocal_radius(radius):
    """
    Returns 1/radius given the radius.

    Args:
        radius (float): the radius of the circle
                        created by the two fish during circling.

    Returns:
        The reciprocal radius (float).
    """
    return 1 / radius


def get_distances(data_for_x_wf, taubin_output, idx):
    """
    Returns an array of fish distances for some window size.
    
    Args:
        data_for_x_wf (array): a 2D array of (x, y) positions
                               for two fish of size  2*window size.
        taubin_output (tuple): (x_center, y_center, radius).
        idx (int): the current window frame index.
    
    Returns:
        An array of interfish distances.
    """
    distance = sqrt((data_for_x_wf[idx][0] - taubin_output[0])**2 + 
                (data_for_x_wf[idx][1] - taubin_output[1])**2)
    return (distance - taubin_output[2])**2


def get_rmse(distances_lst):
    """
    Returns the RMSE of an array of distances.

    Args:
        distances_lst: an array of distances.

    Returns:
        The RMSE (float).
    """
    return sqrt(np.sum(distances_lst))


def get_circling_wf(fish1_pos, fish2_pos, fish1_angle_data, fish2_angle_data,
end, window_size, rad_thresh, rmse_thresh, anti_low, anti_high, head_dist_thresh):
    """
    Returns an array of window frames for circling behavior. Each window
    frame represents the ENDING window frame for circling within some range 
    of window frames specified by the parameter window_size. E.g, if 
    window_size = 10 and a circling window frame is 210, then circling
    occured from frames 200-210.
    
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
        
        window_size (int)     : window size for which circling is averaged over.
        rad_thresh (float)    : radius threshold for circling.
        rmse_thresh (int)     : RMSE threshold for circling.
        anti_low (float)      : antiparallel orientation lower bound.
        anti_high (float)     : antiparallel orientation upper bound. 
        head_dist_thresh (int): head distance threshold for the two fish.

    Returns:
        circling_wf (array): a 1D array of circling window frames.
    """
    idx_1, idx_2 = 0, window_size
    circling_wf = np.array([])
    
    while idx_2 <= end:   # end of the array for both fish
        # Get position and angle data for x window frames 
        fish1_positions = fish1_pos[idx_1:idx_2]
        fish2_positions = fish2_pos[idx_1:idx_2]
        fish1_angles = fish1_angle_data[idx_1:idx_2]
        fish2_angles = fish2_angle_data[idx_1:idx_2]

        head_temp = np.concatenate((fish1_positions, fish2_positions), axis=0)
        taubin_output = TaubinSVD(head_temp)  # output gives (x_c, y_c, r)
        reciprocal_radius = get_reciprocal_radius(taubin_output[2])

        # Fit the distances between the two fish to a circle
        # for every x window frames of size window_size
        distances_temp = np.empty(2 * window_size)
        for i in range(2 * window_size):
            distances_temp[i] = get_distances(head_temp, 
            taubin_output, i)  

        rmse = get_rmse(distances_temp)

        if (reciprocal_radius >= rad_thresh and rmse < rmse_thresh and 
        (check_antiparallel_criterion(fish1_positions, fish2_positions, 
        fish1_angles, fish2_angles, anti_low, anti_high, head_dist_thresh))): 
            circling_wf = np.append(circling_wf, idx_2)
    
        # Update the index variables to track circling for the 
        # next x window frames of size window_size
        idx_1, idx_2 = idx_1+window_size, idx_2+window_size
    return circling_wf
