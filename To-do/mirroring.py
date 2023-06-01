# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 2/3/2023
# version ='1.0'
# ---------------------------------------------------------------------------
import numpy as np
from toolkit import *
# ---------------------------------------------------------------------------
'''
A naive first-attempt at quantifying the "mirroring" behavior. 
Module still IN-PROGRESS.
'''

def get_mirroring_wfs(fish1_pos, fish2_pos, fish1_angle_data, fish2_angle_data, 
end, window_size, theta_90_thresh, head_dist_low, head_dist_high):

    idx_1, idx_2 = 0, window_size
    mirroring_wfs = np.array([])

    while idx_2 <= end:   # end of the array for both fish
        # Get position and angle data for x window frames 
        fish1_positions = fish1_pos[idx_1:idx_2]
        fish2_positions = fish2_pos[idx_1:idx_2]
        fish1_angles = fish1_angle_data[idx_1:idx_2]
        fish2_angles = fish2_angle_data[idx_1:idx_2]
       
        theta_90 = get_antiparallel_angle(fish1_angles, fish2_angles)

        head_temp = get_head_distance(fish1_positions, fish2_positions)

        if ((0 < theta_90 < theta_90_thresh) and 
            (head_dist_low < head_temp < head_dist_high)):
            mirroring_wfs = np.append(mirroring_wfs, idx_2)
                
        # Update the index variables to track mirroring events 
        # for the next x window frames of size window_size
        idx_1, idx_2 = idx_1+window_size, idx_2+window_size
  
    return mirroring_wfs

def main():
    pos_data = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 3, 5)
    angle_data = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 5, 6)

    fish1_pos = pos_data[0]
    fish2_pos = pos_data[1]
    fish1_angle_data = angle_data[0]
    fish2_angle_data = angle_data[1]
    end_of_arr = np.shape(pos_data)[1] 

    print(get_mirroring_wfs(fish1_pos, fish2_pos, fish1_angle_data, fish2_angle_data, 
    end_of_arr, 10, 0.96, 150, 200))



if __name__ == "__main__":
    main()
