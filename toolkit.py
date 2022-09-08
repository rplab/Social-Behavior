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


def get_opposite_orientation_angle(angle_data, idx_1, idx_2):
    fish1_data = angle_data[0][idx_1:idx_2]
    fish2_data = angle_data[1][idx_1:idx_2]
    theta_avg = np.mean(np.cos(np.subtract(fish1_data, fish2_data)))
    return theta_avg
