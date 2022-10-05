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

def get_reciprocal_radius(taubin_output):
    return 1 / taubin_output[2]


def get_distances(data_for_x_wf, taubin_output, idx):
    distance = sqrt((data_for_x_wf[idx][0] - taubin_output[0])**2 + 
                (data_for_x_wf[idx][1] - taubin_output[1])**2)
    return (distance - taubin_output[2])**2


def get_rmse(distances_lst):
    return sqrt(np.sum(distances_lst))


def get_circling_wf(fish1_pos, fish2_pos, angle_data, end, window_size,
rad_thresh, rmse_thresh, anti_low, anti_high):
    idx_1, idx_2 = 0, window_size
    circling_wf = np.array([])
    
    while idx_2 <= end:   # end of the array for both fish
        head_temp = np.concatenate((fish1_pos[idx_1:idx_2], 
        fish2_pos[idx_1:idx_2]), axis=0)
        taubin_output = TaubinSVD(head_temp)  # output gives (x_c, y_c, r)
        reciprocal_radius = get_reciprocal_radius(taubin_output)
        distances_temp = np.empty(2 * window_size)
        for i in range(2 * window_size):
            distances_temp[i] = get_distances(head_temp, 
            taubin_output, i)  

        rmse = get_rmse(distances_temp)

        if (reciprocal_radius >= rad_thresh and rmse < rmse_thresh and 
        (check_antiparallel_criterion(angle_data, idx_1, idx_2, anti_low, 
        anti_high, fish1_pos, fish2_pos))): 
            circling_wf = np.append(circling_wf, idx_2)
    
        idx_1, idx_2 = idx_1+window_size, idx_2+window_size
    return circling_wf
    
