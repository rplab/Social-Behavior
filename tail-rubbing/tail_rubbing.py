# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='1.0'
# ---------------------------------------------------------------------------
from toolkit import *
# ------------------------------------------------------------------------------

def get_contact_wf(body1_x, body2_x, body1_y, body2_y, window_size):
    idx_1, idx_2, wf = 0, window_size, window_size
    contact_wf = {"any": np.array([]), "head-body": np.array([])}

    for i in range(15000):
        # Head-body contact
        if (np.min(np.sqrt((body1_x[idx_1][0] - body2_x[idx_1])**2 + 
        (body1_y[idx_1][0] - body2_y[idx_1])**2)) < 20 or 
        np.min(np.sqrt((body1_x[idx_1] - body2_x[idx_1][0])**2 + 
        (body1_y[idx_1] - body2_y[idx_1][0])**2) < 20)):
             contact_wf["head-body"] = np.append(contact_wf["head-body"], wf)
             contact_wf["any"] = np.append(contact_wf["any"], wf)

        # Any contact 
        for j in range(1, 10):
            if (np.min(np.sqrt((body1_x[idx_1][j] - body2_x[idx_1])**2 + 
            (body1_y[idx_1][j] - body2_y[idx_1])**2)) < 20 or 
            np.min(np.sqrt((body1_x[idx_1] - body2_x[idx_1][j])**2 + 
            (body1_y[idx_1] - body2_y[idx_1][j])**2) < 20)) and (wf not in
            contact_wf["any"]):
                contact_wf["any"] = np.append(contact_wf["any"], wf)
    
        idx_1, idx_2, wf = idx_1+window_size, idx_2+window_size, wf+window_size
    return contact_wf


def get_tail_rubbing_wf(body1_x, body2_x, body1_y, body2_y, pos_data,
angle_data, window_size): 
    idx_1, idx_2, wf = 0, window_size, window_size
    tail_rubbing_wf = np.array([])

    for i in range (15000 // window_size):
        tail1_x = np.average(body1_x[:, -4:][idx_1:idx_2], axis=0)
        tail2_x = np.average(body2_x[:, -4:][idx_1:idx_2], axis=0)
        tail1_y = np.average(body1_y[:, -4:][idx_1:idx_2], axis=0)
        tail2_y = np.average(body2_y[:, -4:][idx_1:idx_2], axis=0)
      
        for j in range(4):
            min_dist = get_min_tail_distances(tail1_x, tail2_x, tail1_y,
            tail2_y, idx_1, j)
            if (min_dist[0] < 40 and min_dist[1] < 40 or 
            min_dist[2] < 40 and min_dist[3] < 40 and
            check_antiparallel_criterion(angle_data, idx_1, idx_2, -1, 
            -0.8, pos_data[0], pos_data[1])):
                if wf not in tail_rubbing_wf:
                    tail_rubbing_wf = np.append(tail_rubbing_wf, wf)

        idx_1, idx_2, wf = idx_1+window_size, idx_2+window_size, wf+window_size
    return tail_rubbing_wf


def get_min_tail_distances(pos1_x, pos2_x, pos1_y, pos2_y, idx_1, j):
    dist_matrix1 = np.sqrt((pos1_x[j] - pos2_x)**2 + 
    (pos1_y[j] - pos2_y)**2)
    dist_matrix2 = np.sqrt((pos1_x - pos2_x[j])**2 + 
    (pos1_y - pos2_y[j])**2)
    min1, min2 = np.partition(dist_matrix1, 1)[0:2]  # two smallest vals
    min3, min4 = np.partition(dist_matrix2, 1)[0:2]
    return min1, min2, min3, min4
