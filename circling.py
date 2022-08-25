# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='1.0'
# ---------------------------------------------------------------------------
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from circle_fit_taubin import TaubinSVD
# ---------------------------------------------------------------------------

def load_data(dir, idx_1, idx_2):
    data = np.genfromtxt(dir, delimiter=',')
    fish1_data = data[:15000][:, np.r_[idx_1:idx_2]]
    fish2_data = data[15000:][:, np.r_[idx_1:idx_2]]
    return fish1_data, fish2_data


def get_reciprocal_radius(taubin_output):
    return 1 / taubin_output[2]


def get_distances(data_for_x_wf, taubin_output, idx):
    distance = sqrt((data_for_x_wf[idx][0] - taubin_output[0])**2 + 
                (data_for_x_wf[idx][1] - taubin_output[1])**2)
    return (distance - taubin_output[2])**2


def get_rmse(distances_lst):
    return sqrt(np.sum(distances_lst))


def get_circling_window_frames(circling_data, angle_data, window_size):
    fish1_data = circling_data[0]
    fish2_data = circling_data[1]
    idx_1, idx_2, wf = 0, window_size, window_size
    circling_wf = np.array([])
    
    while idx_2 <= 15000:   # end of the array for both fish
        head_temp = np.concatenate((fish1_data[idx_1:idx_2], 
        fish2_data[idx_1:idx_2]), axis=0)
        taubin_output = TaubinSVD(head_temp)  # output gives (x_c, y_c, r)
        reciprocal_radius = get_reciprocal_radius(taubin_output)
        distances_temp = np.empty(2 * window_size)
        for i in range(2 * window_size):
            distances_temp[i] = get_distances(head_temp, 
            taubin_output, i)  

        rmse = get_rmse(distances_temp)

        if (reciprocal_radius >= 0.005 and rmse < 25 and 
        -1 <= get_angles(angle_data, idx_1, idx_2) < -0.9): 
            circling_wf = np.append(circling_wf, wf)
    
        idx_1, idx_2, wf = idx_1+window_size, idx_2+window_size, wf+window_size
    return circling_wf
            

def get_angles(angle_data, idx_1, idx_2):
    fish1_data = angle_data[0][idx_1:idx_2]
    fish2_data = angle_data[1][idx_1:idx_2]
    theta_avg = np.mean(np.subtract(fish1_data, fish2_data))
    return theta_avg


#         if np.abs(theta_avg) < 0.1:
#             theta_90 = np.append(theta_90, theta_avg)

def main():
    circling_data = load_data("results_SocPref_3c_2wpf_k1_ALL.csv", 3, 5)
    angle_data = load_data("results_SocPref_3c_2wpf_k1_ALL.csv", 5, 6)

    circling_wfs = get_circling_window_frames(circling_data, angle_data, 10)
    print(circling_wfs)
