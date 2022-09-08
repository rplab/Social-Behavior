# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/7/2022
# version ='1.0'
# ---------------------------------------------------------------------------
import numpy as np
from toolkit import *
# ---------------------------------------------------------------------------

def get_90_deg_wf(pos_data, angle_data, window_size):
    idx_1, idx_2, wf = 0, window_size, window_size
    orientations = {
        "bin1": [], "bin2": [], "bin3": [], "bin4": [],
        "bin5": [], "bin6": [], "bin7": [], "bin8": [],
    }

    while idx_2 <= 15000:   # end of the array for both fish
        theta_90 = np.abs(get_opposite_orientation_angle(angle_data, 
        idx_1, idx_2))
        fish_vectors = get_fish_vectors(angle_data, idx_1, idx_2)
        fish1_vector, fish2_vector = fish_vectors[0], fish_vectors[1]
        connecting_vector = get_connecting_vector(pos_data, idx_1, idx_2)
        fish1xfish2 = np.sign(np.cross(fish1_vector, fish2_vector))
        fish1xconnect = np.sign(np.cross(fish1_vector, connecting_vector))
        fish2xconnect = np.sign(np.cross(fish2_vector, connecting_vector))
        orientation_type = get_orientation_type((fish1xfish2, fish1xconnect,
        fish2xconnect))
        
        if theta_90 < 0.1 and orientation_type in orientations.keys():
            orientations[orientation_type].append(wf)
        
        idx_1, idx_2, wf = idx_1+window_size, idx_2+window_size, wf+window_size
    return orientations


def get_fish_vectors(angle_data, idx_1, idx_2):
    fish1_data = angle_data[0][idx_1:idx_2]
    fish2_data = angle_data[1][idx_1:idx_2]
    fish1_vector = np.array((np.mean(np.cos(fish1_data)), 
    np.mean(np.sin(fish1_data))))
    fish2_vector = np.array((np.mean(np.cos(fish2_data)), 
    np.mean(np.sin(fish2_data))))
    return np.array((fish1_vector, fish2_vector))


def get_connecting_vector(pos_data, idx_1, idx_2):
    fish1_data = pos_data[0][idx_1:idx_2]
    fish2_data = pos_data[1][idx_1:idx_2]
    connecting_vector = np.array((np.mean(fish2_data[:, 0] - fish1_data[:, 0]), 
    np.mean(fish2_data[:, 1] - fish1_data[:, 1])))
    normalized_vector = connecting_vector / np.linalg.norm(connecting_vector)
    return normalized_vector
    

def get_orientation_type(sign_tuple):
    switcher = {
        (1,1,1)   : "bin1",
        (-1,-1,-1): "bin2",
        (-1,1,1)  : "bin3",
        (1,-1,1)  : "bin4",
        (1,1,-1)  : "bin5",
        (-1,-1,1) : "bin6",
        (1,-1,-1) : "bin7",
        (-1,1,-1) : "bin8"
    }
    return switcher.get(sign_tuple)
    
    
def main():
    pos_data = load_data("results_SocPref_3c_2wpf_k1_ALL.csv", 3, 5)
    angle_data = load_data("results_SocPref_3c_2wpf_k1_ALL.csv", 5, 6)
    orientation_dict = get_90_deg_wf(pos_data, angle_data, 10)
    print(orientation_dict)
   



if __name__ == '__main__':
    main()
