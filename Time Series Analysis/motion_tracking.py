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
from scipy.interpolate import make_interp_spline
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

    
def motion_frames_plots(fish1_motion, fish2_motion, dataset_name, block_size):
    # Normalize both velocity fish data sets by their mean
    fish1_norm = (fish1_motion - np.mean(fish1_motion)) / np.mean(fish1_motion)
    fish2_norm = (fish2_motion - np.mean(fish2_motion)) / np.mean(fish2_motion)

    # All velocity arrays are of the same size
    lag_arr = np.arange(0, np.size(fish1_norm))

    plt.figure()
    plt.title(f"{dataset_name}: Velocity Correlation; block_size = {block_size}")
    idx_1, idx_2 = 0, block_size

    for i in range(0, np.size(fish1_norm)+1, block_size):
    # Calculate the velocity cross correlation
        cross_corr = sm.tsa.stattools.ccf(fish1_norm[idx_1:idx_2], 
        fish2_norm[idx_1:idx_2], adjusted=False)
        
        plt.plot(lag_arr[idx_1:idx_2], cross_corr)

        idx_1, idx_2 = idx_1+block_size, idx_2+block_size
    plt.show()


def correlation_plots(fish1_motion, fish2_motion, fish1_angles, fish2_angles,
    motion, dataset_name, end, window_size, reg=0, corr=0, auto=1, shuff=0, angle=0):
    # Note: Either fish speed OR velocity can be passed as the first 
    # two parameters 
    if motion == 's':
        type = 'Speed'
    else:
        type = 'Velocity'

    # Normalize both speed fish data sets by their mean
    fish1_norm = (fish1_motion - np.mean(fish1_motion)) / np.mean(fish1_motion)
    fish2_norm = (fish2_motion - np.mean(fish2_motion)) / np.mean(fish2_motion)

    # Calculate the motion cross correlation
    cross_corr = sm.tsa.stattools.ccf(fish1_norm, fish2_norm, adjusted=False)
    cross_corr_max = np.max(cross_corr)
    cross_corr_min = np.min(cross_corr)

    # All correlation arrays are of the same size
    lag_arr = np.arange(0, np.size(cross_corr))

    # Regular nonscaled plot of fish motion
    if reg == 1:
        wfs = np.arange(0, end-window_size, window_size)
        plt.figure()
        plt.title(f"{dataset_name}: Nonscaled {type} Plot Between Two Fish")
        plt.xlabel(f"{type}; WS = {window_size}")
        plt.plot(wfs, fish1_motion, color='turquoise')
        plt.plot(wfs, fish2_motion, color='mediumslateblue')

    # Motion Cross-Correlation Plot
    if corr == 1:
        # Superimpose smoothed frames on
        # top of cross correlation plot
        lag_smooth = lag_arr[0:np.size(cross_corr)]
        cross_corr_smooth = cross_corr[0:np.size(cross_corr)]

        xnew = np.linspace(lag_smooth.min(), lag_smooth.max(), 50)
        spl = make_interp_spline(lag_smooth, cross_corr_smooth)
        ynew = spl(xnew)

        plt.figure()
        plt.title(f"{dataset_name}: {type} Cross-Correlation Plot Between Two Fish")
        plt.xlabel(f"Lag; WS = {window_size}")
        plt.ylabel("Correlation Coefficients")
        plt.plot(lag_arr, cross_corr, color='royalblue')
        plt.plot(xnew, ynew, color='red')

    # Motion Autocorrelation Plots
    if auto == 1:
        # Calculate autocorrelations
        fish1_auto_corr = sm.tsa.stattools.ccf(fish1_norm, fish1_norm, adjusted=False)
        fish2_auto_corr = sm.tsa.stattools.ccf(fish2_norm, fish2_norm, adjusted=False)

        plt.figure()
        plt.title(f"{dataset_name}: {type} Autocorrelation Plot Between Two Fish [Fish1]")
        plt.ylim(cross_corr_min, cross_corr_max)
        plt.xlabel(f"Lag; WS = {window_size}")
        plt.ylabel("Correlation Coefficients")
        plt.plot(lag_arr, fish1_auto_corr, color='turquoise')

        plt.figure()
        plt.title(f"{dataset_name}: {type} Autocorrelation Plot Between Two Fish [Fish2]")
        plt.ylim(cross_corr_min, cross_corr_max)
        plt.xlabel(f"Lag; WS = {window_size}")
        plt.ylabel("Correlation Coefficients")
        plt.plot(lag_arr, fish2_auto_corr, color='mediumslateblue')

    # Speed Shuffled Cross-Correlation Plot
    if shuff == 1:
        # Calculate the cross correlation of fish1
        # with a permutation of fish2 -- resulting 
        # plot should be centered at 0              
        fish2_shuffled = np.random.permutation(fish2_norm)
        
        shuffled_cross_corr = sm.tsa.stattools.ccf(fish1_norm, 
        fish2_shuffled, adjusted=False)

        plt.figure()
        plt.title(f"{dataset_name}: Shuffled {type} Cross-Correlation Plot Between Two Fish")
        plt.xlabel(f"Lag; WS = {window_size}")
        plt.ylabel("Correlation Coefficients")
        plt.plot(lag_arr, shuffled_cross_corr, color='hotpink')

    # Angle Cross-Correlation Plot
    if angle == 1:
        # Calculate the angle cross correlation
        cross_corr = sm.tsa.stattools.ccf(fish1_angles, fish2_angles, adjusted=False)

        # Superimpose smoothed frames on
        # top of cross correlation plot
        lag_smooth = lag_arr[0:np.size(cross_corr)]
        cross_corr_smooth = cross_corr[0:np.size(cross_corr)]

        xnew = np.linspace(lag_smooth.min(), lag_smooth.max(), 20)
        spl = make_interp_spline(lag_smooth, cross_corr_smooth)
        ynew = spl(xnew)

        plt.figure()
        plt.title(f"{dataset_name}: Angle Cross-Correlation Plot Between Two Fish")
        plt.xlabel(f"Lag; WS = {window_size}")
        plt.ylabel("Correlation Coefficients")
        plt.plot(lag_arr, cross_corr, color='royalblue')
        plt.plot(xnew, ynew, color='red')

    plt.show()
