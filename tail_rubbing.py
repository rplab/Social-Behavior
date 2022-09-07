# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='1.0'
# ---------------------------------------------------------------------------
import numpy as np
from circling import load_data, get_angles
# ---------------------------------------------------------------------------
  
def get_tail_rubbing_wf(fish1_x, fish2_x, fish1_y, fish2_y, 
angle_data, window_size): 
    idx_1, idx_2, wf = 0, window_size, window_size
    tail_rubbing_wf = np.array([])
    dist_matrix = np.sqrt((fish1_x - fish2_x)**2 + (fish1_y - fish2_y)**2)
  
    for array in dist_matrix[::window_size]:
        dist_temp = dist_matrix[idx_1:idx_2]
        avg_dist_array = np.average(dist_temp, axis=0)
        
        if (np.min(avg_dist_array) < 65 and
        -1 <= get_angles(angle_data, idx_1, idx_2) < -0.8): 
            tail_rubbing_wf = np.append(tail_rubbing_wf, wf)

        idx_1, idx_2, wf = idx_1+window_size, idx_2+window_size, wf+window_size
    return tail_rubbing_wf


def main():
    tail_rubbing_x = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 6, 16)
    tail_rubbing_y = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 16, 26)
    angle_data = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 5, 6)

    tail_rubbing_wf = get_tail_rubbing_wf(tail_rubbing_x[0], 
    tail_rubbing_x[1], tail_rubbing_y[0], tail_rubbing_y[1],
    angle_data, 4)
    print(tail_rubbing_wf)
    
    

if __name__ == '__main__':
    main()
