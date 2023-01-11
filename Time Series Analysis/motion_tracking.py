# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#-------------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 11/2/2022
# version ='1.0'
# ------------------------------------------------------------------------------
import numpy as np
from toolkit import get_head_distance_traveled, get_fish_vectors
import statsmodels.api as sm
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------

# ------------------------- MODULE IN PROGRESS!!! ------------------------------
def get_speed(fish1_pos, fish2_pos, end, window_size):
    idx_1, idx_2 = 0, window_size
    array_idx = 0 
    arr_size = (15000 // window_size) - 1
    fish1_speeds = np.empty(arr_size)
    fish2_speeds = np.empty(arr_size)
  
    while idx_2 < end:   # end of the array for both fish
        # Speed = change in distance / change in time; 
        # Speed expressed in distance/second 
        # 1 second = 25 wfs
        fish1_speeds[array_idx] = (get_head_distance_traveled(fish1_pos, idx_1, idx_2) / window_size) * 25
        fish2_speeds[array_idx] = (get_head_distance_traveled(fish2_pos, idx_1, idx_2) / window_size) * 25

        array_idx += 1
        idx_1, idx_2 = idx_1+window_size, idx_2+window_size
    return (fish1_speeds, fish2_speeds)


def get_velocity(fish1_speeds, fish2_speeds, fish1_angle_data, fish2_angle_data, 
    end, window_size):
    idx_1, idx_2 = 0, window_size
    speed_idx = 0
    array_idx = 0 
    arr_size = (15000 // window_size) - 1
    fish1_velocities_mag = np.empty(arr_size)
    fish2_velocities_mag = np.empty(arr_size)
  
    while idx_2 < end:   # end of the array for both fish
        # The fish speeds are already averaged over 
        # x window frames, so we just need to access 
        # each element sequentially  
        fish1_speed, fish2_speed = fish1_speeds[speed_idx], fish2_speeds[speed_idx]

        # Averaging over x window frames is 
        # done in the get_fish_vectors function
        fish1_angles = fish1_angle_data[idx_1:idx_2]
        fish2_angles = fish2_angle_data[idx_1:idx_2]
        fish_vectors_tuple = get_fish_vectors(fish1_angles, fish2_angles)

        # Vectors are unit vectors, so there is 
        # no need divide final velocity vector 
        # by the magnitude of the direction vector
        fish1_vector, fish2_vector = fish_vectors_tuple[0], fish_vectors_tuple[1]
        # velocity = (speed / magnitude of direction vector) * direction vector 
        fish1_velocity_vector = fish1_speed * fish1_vector  
        fish2_velocity_vector = fish2_speed * fish2_vector 
        fish1_velocity_mag = np.sqrt(fish1_velocity_vector[0]**2 + fish1_velocity_vector[1]**2)
        fish2_velocity_mag = np.sqrt(fish2_velocity_vector[0]**2 + fish2_velocity_vector[1]**2)

        fish1_velocities_mag[array_idx] = fish1_velocity_mag
        fish2_velocities_mag[array_idx] = fish2_velocity_mag
        
        speed_idx += 1
        array_idx += 1
        idx_1, idx_2 = idx_1+window_size, idx_2+window_size
    return (fish1_velocities_mag, fish2_velocities_mag)


def get_angle(fish1_angle, fish2_angle, end, window_size):
    idx_1, idx_2 = 0, window_size
    array_idx = 0 
    arr_size = (15000 // window_size) - 1
    fish1_angles = np.empty(arr_size)
    fish2_angles = np.empty(arr_size)
  
    while idx_2 < end:   # end of the array for both fish
        fish1_angles[array_idx] = np.cos(np.mean(fish1_angle[idx_1:idx_2])) 
        fish2_angles[array_idx] = np.cos(np.mean(fish2_angle[idx_1:idx_2])) 

        array_idx += 1
        idx_1, idx_2 = idx_1+window_size, idx_2+window_size
    return (fish1_angles, fish2_angles)
