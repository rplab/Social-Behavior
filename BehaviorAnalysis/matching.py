# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 11/3/2022
# version ='1.0'
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from toolkit import *
# ---------------------------------------------------------------------------
'''
A naive first-attempt at quantifying the "reorientation" behavior. 
Module still IN-PROGRESS.
'''

def get_reorientation_wfs(fish1_pos, fish2_pos, fish1_angle_data, fish2_angle_data, 
end, window_size, theta_90_thresh, head_dist):
    idx_1, idx_2 = 0, window_size
    counter = 0 
    reorientation_wfs = np.array([])

    while idx_2 <= end:   # end of the array for both fish
        theta_90 = np.abs(get_antiparallel_angle(fish1_angle_data, fish2_angle_data, 
        idx_1, idx_2))
        head_temp = get_head_distance(fish1_pos, fish2_pos, idx_1, idx_2)

        if (theta_90 < theta_90_thresh and head_temp < head_dist):
            counter += 1 

        if idx_2 % 40 == 0:
            if counter >= 35:
                reorientation_wfs = np.append(reorientation_wfs, idx_2).astype(int)
            counter = 0
                
        # Update the index variables to track 90-degree events 
        # for the next x window frames of size window_size
        idx_1, idx_2 = idx_1+window_size, idx_2+window_size
  
    return reorientation_wfs

def main():
    pos_data = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 3, 5)
    angle_data = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 5, 6)

    fish1_pos = pos_data[0]
    fish2_pos = pos_data[1]
    fish1_angle_data = angle_data[0]
    fish2_angle_data = angle_data[1]
    end_of_arr = np.shape(pos_data)[1] 

    get_reorientation_wfs(fish1_pos, fish2_pos, fish1_angle_data, fish2_angle_data, 
    end_of_arr, 1, 0.3, 300)


if __name__ == "__main__":
    main()
