
# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='1.0'
# ---------------------------------------------------------------------------
from toolkit import *
# ------------------------------------------------------------------------------

def get_contact_wf(body1_x, body2_x, body1_y, body2_y, end, window_size,
contact_dist):
    """
    Returns a dictionary of window frames for different types of
    contact between two fish.

    Args:
        body1_x (array): a 1D array of x positions along the 10 body markers of fish1.
        body2_x (array): a 1D array of x positions along the 10 body markers of fish2.
        body1_y (array): a 1D array of y positions along the 10 body markers of fish1.
        body2_y (array): a 1D array of y positions along the 10 body markers of fish2.
        end (int): end of the array for both fish (typically 15,000 window frames.) 
        window_size (int): window size for which circling is averaged over.
        contact_dist (int): the contact distance threshold.

    Returns:
        contact_wf (dictionary): a dictionary of arrays of different contact types.
    """
    idx_1, idx_2 = 0, window_size
    contact_wf = {"any": np.array([]), "head-body": np.array([])}

    for i in range(end // window_size):
        # Head-body contact
        if (np.min(np.sqrt((body1_x[idx_1][0] - body2_x[idx_1])**2 + 
        (body1_y[idx_1][0] - body2_y[idx_1])**2)) < contact_dist or 
        np.min(np.sqrt((body1_x[idx_1] - body2_x[idx_1][0])**2 + 
        (body1_y[idx_1] - body2_y[idx_1][0])**2) < contact_dist)):
             contact_wf["head-body"] = np.append(contact_wf["head-body"], idx_2)
             contact_wf["any"] = np.append(contact_wf["any"], idx_2)

        # Any contact 
        for j in range(1, 10):  # there are ten markers on the fish bodies
            if (np.min(np.sqrt((body1_x[idx_1][j] - body2_x[idx_1])**2 + 
            (body1_y[idx_1][j] - body2_y[idx_1])**2)) < contact_dist or 
            np.min(np.sqrt((body1_x[idx_1] - body2_x[idx_1][j])**2 + 
            (body1_y[idx_1] - body2_y[idx_1][j])**2) < contact_dist)) and (idx_2 not in
            contact_wf["any"]):
                contact_wf["any"] = np.append(contact_wf["any"], idx_2)

        # Update the index variables to track contact for the 
        # next x window frames of size window_size
        idx_1, idx_2 = idx_1+window_size, idx_2+window_size
    return contact_wf


def get_tail_rubbing_wf(body1_x, body2_x, body1_y, body2_y, fish1_pos, fish2_pos,
fish1_angle_data, fish2_angle_data, end, window_size, tail_dist, tail_anti_low, 
tail_anti_high, head_dist): 
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
        tail_anti_low (float): antiparallel orientation lower bound.
        tail_anti_high (float): antiparallel orientation upper bound. 
        head_dist_thresh (int): head distance threshold for the two fish. 

    Returns:
        tail_rubbing_wf (array): a 1D array of tail-rubbing window frames.
    """
    idx_1, idx_2 = 0, window_size
    tail_rubbing_wf = np.array([])

    for i in range (end // window_size):
        tail1_x = np.average(body1_x[:, -4:][idx_1:idx_2], axis=0)
        tail2_x = np.average(body2_x[:, -4:][idx_1:idx_2], axis=0)
        tail1_y = np.average(body1_y[:, -4:][idx_1:idx_2], axis=0)
        tail2_y = np.average(body2_y[:, -4:][idx_1:idx_2], axis=0)
      
        for j in range(4):   # get at least four body points in contact
            min_dist = get_min_tail_distances(tail1_x, tail2_x, tail1_y,
            tail2_y, j)
            if (min_dist[0] < tail_dist and min_dist[1] < tail_dist or 
            min_dist[2] < tail_dist and min_dist[3] < tail_dist and
            check_antiparallel_criterion(fish1_angle_data, fish2_angle_data, 
            idx_1, idx_2, tail_anti_low, tail_anti_high, fish1_pos, fish2_pos,
            head_dist)):
                if idx_2 not in tail_rubbing_wf:
                    tail_rubbing_wf = np.append(tail_rubbing_wf, idx_2)

        # Update the index variables to track tail-rubs for the 
        # next x window frames of size window_size
        idx_1, idx_2 = idx_1+window_size, idx_2+window_size
    return tail_rubbing_wf


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
