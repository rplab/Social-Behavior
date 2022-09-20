# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/7/2022
# version ='1.0'
# ---------------------------------------------------------------------------
import numpy as np
# ---------------------------------------------------------------------------

def load_data(dir, idx_1, idx_2):
    data = np.genfromtxt(dir, delimiter=',')
    fish1_data = data[:15000][:, np.r_[idx_1:idx_2]]
    fish2_data = data[15000:][:, np.r_[idx_1:idx_2]]
    return fish1_data, fish2_data


def get_antiparallel_angle(angle_data, idx_1, idx_2):
    fish1_data = angle_data[0][idx_1:idx_2]
    fish2_data = angle_data[1][idx_1:idx_2]
    theta_avg = np.mean(np.cos(np.subtract(fish1_data, fish2_data)))
    return theta_avg


def get_head_distance(fish1_pos, fish2_pos, idx_1, idx_2):
    head_dist_matrix = np.sqrt((fish2_pos[:, 0] - fish1_pos[:, 0])**2 + 
    (fish2_pos[:, 1] - fish1_pos[:, 1])**2)
    dist_temp = head_dist_matrix[idx_1:idx_2]
    avg_dist = np.average(dist_temp, axis=0)
    return avg_dist
    

def check_antiparallel_criterion(angle_data, idx_1, idx_2, lower_threshold,
upper_threshold, fish1_pos, fish2_pos):
    angle = get_antiparallel_angle(angle_data, idx_1, idx_2)
    head_distance = get_head_distance(fish1_pos, fish2_pos, idx_1, idx_2)

    if lower_threshold <= angle < upper_threshold and head_distance < 150:
        res = True
    else:
        res = False
    return res
