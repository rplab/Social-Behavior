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


def get_circling_wf(fish1_pos, fish2_pos, angle_data, window_size):
    idx_1, idx_2, wf = 0, window_size, window_size
    circling_wf = np.array([])
    
    while idx_2 <= 15000:   # end of the array for both fish
        head_temp = np.concatenate((fish1_pos[idx_1:idx_2], 
        fish2_pos[idx_1:idx_2]), axis=0)
        taubin_output = TaubinSVD(head_temp)  # output gives (x_c, y_c, r)
        reciprocal_radius = get_reciprocal_radius(taubin_output)
        distances_temp = np.empty(2 * window_size)
        for i in range(2 * window_size):
            distances_temp[i] = get_distances(head_temp, 
            taubin_output, i)  

        rmse = get_rmse(distances_temp)

        if (reciprocal_radius >= 0.005 and rmse < 25 and 
        (check_antiparallel_criterion(angle_data, idx_1, idx_2, -1, -0.9, 
        fish1_pos, fish2_pos))): 
            circling_wf = np.append(circling_wf, wf)
    
        idx_1, idx_2, wf = idx_1+window_size, idx_2+window_size, wf+window_size
    return circling_wf
