
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
    dist_matrix1 = np.sqrt((pos1_x[j] - pos2_x)**2 + 
    (pos1_y[j] - pos2_y)**2)
    dist_matrix2 = np.sqrt((pos1_x - pos2_x[j])**2 + 
    (pos1_y - pos2_y[j])**2)
    min1, min2 = np.partition(dist_matrix1, 1)[0:2]  # two smallest vals
    min3, min4 = np.partition(dist_matrix2, 1)[0:2]
    return min1, min2, min3, min4
