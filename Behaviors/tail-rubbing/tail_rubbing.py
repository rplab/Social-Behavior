# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='1.0'
# ---------------------------------------------------------------------------
from toolkit import *
# ------------------------------------------------------------------------------

def get_tail_rubbing_wf(body1_x, body2_x, body1_y, body2_y, fish1_pos, fish2_pos,
fish1_angle_data, fish2_angle_data, end, window_size, tail_dist, tail_anti_angle, 
head_dist_thresh): 
    """
    Returns an array of tail-rubbing window frames.

    Args:
        body1_x (array): a 1D array of x positions along the 10 body markers of fish1.
        body2_x (array): a 1D array of x positions along the 10 body markers of fish2.
        body1_y (array): a 1D array of y positions along the 10 body markers of fish1.
        body2_y (array): a 1D array of y positions along the 10 body markers of fish2.
        fish1_pos (array): a 2D array of (x, y) positions for fish1. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish2_pos (array): a 2D array of (x, y) positions for fish1. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish1_angle_data (array): a 1D array of angles at each window frame
                                  for fish1.
        fish2_angle_data (array): a 1D array of angles at each window frame
                                  for fish2.

        end (int): end of the array for both fish (typically 15,000 window frames.) 
        window_size (int): window size for which circling is averaged over.
        tail_dist (int): tail distance threshold for the two fish. 
        tail_anti_angle (float): antiparallel orientation angle threshold. 
        head_dist_thresh (int): head distance threshold for the two fish. 

    Returns:
        tail_rubbing_wf (array): a 1D array of tail-rubbing window frames.
    """
    idx_1, idx_2 = 0, window_size
    tail_rubbing_wf = []

    while idx_2 <= end:   # end of the array for both fish
        # Get 4 posterior body markers for x window frames 
        tail1_x = np.average(body1_x[:, -4:][idx_1:idx_2], axis=0)
        tail2_x = np.average(body2_x[:, -4:][idx_1:idx_2], axis=0)
        tail1_y = np.average(body1_y[:, -4:][idx_1:idx_2], axis=0)
        tail2_y = np.average(body2_y[:, -4:][idx_1:idx_2], axis=0)

        # Get position and angle data for x window frames 
        fish1_positions = fish1_pos[idx_1:idx_2]
        fish2_positions = fish2_pos[idx_1:idx_2]
        fish1_angles = fish1_angle_data[idx_1:idx_2]
        fish2_angles = fish2_angle_data[idx_1:idx_2]
      
        for j in range(4):   # get at least four body points in contact
            min_dist = get_min_tail_distances(tail1_x, tail2_x, tail1_y,
            tail2_y, j)
            if (min_dist[0] < tail_dist and min_dist[1] < tail_dist or 
            min_dist[2] < tail_dist and min_dist[3] < tail_dist and
            check_antiparallel_criterion(fish1_positions, fish2_positions, fish1_angles, 
            fish2_angles, tail_anti_angle, head_dist_thresh)):
                if idx_1 not in tail_rubbing_wf:
                    tail_rubbing_wf.append(idx_1+1)

        # Update the index variables to track tail-rubs for the 
        # next x window frames of size window_size
        idx_1 += 1
        idx_2 += 1
    return combine_events(np.array(tail_rubbing_wf))


def get_min_tail_distances(pos1_x, pos2_x, pos1_y, pos2_y, j):
    """
    Returns the minimum tail distances between two fish for two 
    different body markers.

    Args:
        pos1_x (array): a 1D array of tail x positions for fish1 averaged over some window size.
        pos2_x (array): a 1D array of tail x positions for fish2 averaged over some window size.
        pos1_y (array): a 1D array of tail y positions for fish1 averaged over some window size.
        pos2_y (array): a 1D array of tail y positions for fish2 averaged over some window size.
        j (int): current index.

    Returns:
        A tuple of minimum distances (min1, min2, min3, min4).
            min1 (float): minimum x distance for the first body marker. 
            min2 (float): minimum y distance for the first body marker.
            min3 (float): minimum x distance for the second body marker.
            min4 (float): minimum y distance for the second body marker.
    """
    dist_matrix1 = np.sqrt((pos1_x[j] - pos2_x)**2 + 
    (pos1_y[j] - pos2_y)**2)
    dist_matrix2 = np.sqrt((pos1_x - pos2_x[j])**2 + 
    (pos1_y - pos2_y[j])**2)
    min1, min2 = np.partition(dist_matrix1, 1)[0:2]  # two smallest vals
    min3, min4 = np.partition(dist_matrix2, 1)[0:2]
    return min1, min2, min3, min4
