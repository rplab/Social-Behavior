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


def get_antiparallel_angle(fish1_angle_data, fish2_angle_data, idx_1, idx_2):
    fish1_data = fish1_angle_data[idx_1:idx_2]
    fish2_data = fish2_angle_data[idx_1:idx_2]
    theta_avg = np.mean(np.cos(np.subtract(fish1_data, fish2_data)))
    return theta_avg


def get_head_distance(fish1_pos, fish2_pos, idx_1, idx_2):
    head_dist_matrix = np.sqrt((fish2_pos[:, 0] - fish1_pos[:, 0])**2 + 
    (fish2_pos[:, 1] - fish1_pos[:, 1])**2)
    dist_temp = head_dist_matrix[idx_1:idx_2]
    avg_dist = np.average(dist_temp, axis=0)
    return avg_dist


def get_head_distance_traveled(fish_pos, idx_1, idx_2):
    initial_x = fish_pos[:, 0][idx_1]
    initial_y = fish_pos[:, 1][idx_1]
    final_x = fish_pos[:, 0][idx_2]
    final_y = fish_pos[:, 1][idx_2]
    dist = np.sqrt((final_x - initial_x)**2 + (final_y - initial_y)**2)
    return dist


def get_fish_vectors(fish1_angle_data, fish2_angle_data, idx_1, idx_2):
    fish1_data = fish1_angle_data[idx_1:idx_2]
    fish2_data = fish2_angle_data[idx_1:idx_2]
    fish1_vector = np.array((np.mean(np.cos(fish1_data)), 
    np.mean(np.sin(fish1_data))))
    fish2_vector = np.array((np.mean(np.cos(fish2_data)), 
    np.mean(np.sin(fish2_data))))
    return np.array((fish1_vector, fish2_vector))


def check_antiparallel_criterion(fish1_angle_data, fish2_angle_data,
idx_1, idx_2, lower_threshold, upper_threshold, fish1_pos, fish2_pos,
head_dist_threshold):
    angle = get_antiparallel_angle(fish1_angle_data, fish2_angle_data, idx_1, idx_2)
    head_distance = get_head_distance(fish1_pos, fish2_pos, idx_1, idx_2)

    if (lower_threshold <= angle < upper_threshold and 
    head_distance < head_dist_threshold):
        res = True
    else:
        res = False
    return res

