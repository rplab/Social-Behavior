# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 11/2/2022
# version ='1.0'
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import re
from lmfit import minimize, Parameters, fit_report
from scipy import optimize
from toolkit import *
import statsmodels.api as sm
from scipy.interpolate import make_interp_spline
# ---------------------------------------------------------------------------
def get_speed(fish1_pos, fish2_pos, end, window_size):
    idx_1, idx_2 = 0, window_size
    fish1_speeds = np.array([])
    fish2_speeds = np.array([])
  
    while idx_2 < end:   # end of the array for both fish
        # Speed = change in distance / change in time; 
        # Speed expressed in distance/second 
        # 1 second = 25 wfs
        fish1_speeds = np.append(fish1_speeds, 
        (get_head_distance_traveled(fish1_pos, idx_1, idx_2) * 25) / window_size) 
        fish2_speeds = np.append(fish2_speeds, 
        (get_head_distance_traveled(fish2_pos, idx_1, idx_2) * 25) / window_size)

        idx_1, idx_2 = idx_1+window_size, idx_2+window_size

    return (fish1_speeds, fish2_speeds)


def get_angle(fish1_angle, fish2_angle, end, window_size):
    idx_1, idx_2 = 0, window_size
    fish1_angles = np.array([])
    fish2_angles = np.array([])
  
    while idx_2 < end:   # end of the array for both fish
        fish1_angles = np.append(fish1_angles, 
        (np.cos(np.mean(fish1_angle[idx_1:idx_2])))) 
        fish2_angles = np.append(fish2_angles, 
        (np.cos(np.mean(fish2_angle[idx_1:idx_2])))) 

        idx_1, idx_2 = idx_1+window_size, idx_2+window_size

    return fish1_angles, fish2_angles


def get_velocity_mag(fish1_speeds, fish2_speeds, fish1_angle_data, fish2_angle_data, 
    end, window_size):
    speed_idx = 0
    idx_1, idx_2 = 0, window_size
    fish1_velocities_mag = np.array([])
    fish2_velocities_mag = np.array([])
  
    while idx_2 < end:   # end of the array for both fish

        # The fish speeds are already averaged over 
        # x window frames, so we just need to access 
        # each element sequentially  
        fish1_speed, fish2_speed = fish1_speeds[speed_idx], fish2_speeds[speed_idx]

        # Averaging over x window frames is 
        # done in the get_fish_vectors function
        fish_vectors_tuple = get_fish_vectors(fish1_angle_data, fish2_angle_data, 
        idx_1, idx_2)   

        # Vectors are unit vectors, so there is 
        # no need divide final velocity vector 
        # by the magnitude of the direction vector
        fish1_vector, fish2_vector = fish_vectors_tuple[0], fish_vectors_tuple[1]
        # velocity = (speed / magnitude of direction vector) * direction vector 
        fish1_velocity_vector = fish1_speed * fish1_vector  
        fish2_velocity_vector = fish2_speed * fish2_vector 
        fish1_velocity_mag = np.sqrt(fish1_velocity_vector[0]**2 + fish1_velocity_vector[1]**2)
        fish2_velocity_mag = np.sqrt(fish2_velocity_vector[0]**2 + fish2_velocity_vector[1]**2)

        fish1_velocities_mag = np.append(fish1_velocities_mag, fish1_velocity_mag)
        fish2_velocities_mag = np.append(fish2_velocities_mag, fish2_velocity_mag)
        
        speed_idx += 1
        idx_1, idx_2 = idx_1+window_size, idx_2+window_size
    return (fish1_velocities_mag, fish2_velocities_mag)


def obj_func(params, c, t):
    # c0 = params['c0']
    # t_decay = params['decay']
    model = params[0]*np.exp(-t/params[1])
    likelihood = (c - model)**2
    res = -1 * np.sum(likelihood)       # minimize negative of function
    return res 


def minimizer(fish1_velocity, fish2_velocity, dataset_name, block_size):
    # TO-DO: Check if I implemented this function correctly with Raghu
    # params = Parameters()
    # params.add('c0', value=0.04)
    # params.add('decay', value=0.007)
    params_guess = np.array((0.06, 0.001))

    fish1_norm = (fish1_velocity - np.mean(fish1_velocity)) / np.mean(fish1_velocity)
    fish2_norm = (fish2_velocity - np.mean(fish2_velocity)) / np.mean(fish2_velocity)
    # All velocity arrays are of the same size
    lag_arr = np.arange(0, np.size(fish1_norm))

    idx_1, idx_2 = 0, block_size
    for i in range(0, np.size(fish1_norm)+1, block_size):
        # Calculate the velocity cross correlation
        cross_corr = sm.tsa.stattools.ccf(fish1_norm[idx_1:idx_2], 
        fish2_norm[idx_1:idx_2], adjusted=False)
        true_params = optimize.minimize(obj_func, params_guess, 
        args=(cross_corr, lag_arr[idx_1:idx_2]))
        print(true_params['x'])

        idx_1, idx_2 = idx_1+block_size, idx_2+block_size
    

def velocity_frames_plots(fish1_velocity, fish2_velocity, dataset_name, block_size):
    # Normalize both velocity fish data sets by their mean
    fish1_norm = (fish1_velocity - np.mean(fish1_velocity)) / np.mean(fish1_velocity)
    fish2_norm = (fish2_velocity - np.mean(fish2_velocity)) / np.mean(fish2_velocity)

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


def correlation_plots(fish1_speed, fish2_speed, fish1_angles, fish2_angles,
    dataset_name, end, window_size, reg=1, corr=1, auto=1, shuff=1, angle=1):
    # Note: Either fish speed OR velocity can be passed as the first 
    # two parameters 
    
    # Normalize both speed fish data sets by their mean
    fish1_norm = (fish1_speed - np.mean(fish1_speed)) / np.mean(fish1_speed)
    fish2_norm = (fish2_speed - np.mean(fish2_speed)) / np.mean(fish2_speed)

    # Calculate the speed cross correlation
    cross_corr = sm.tsa.stattools.ccf(fish1_norm, fish2_norm, adjusted=False)
    cross_corr_max = np.max(cross_corr)
    cross_corr_min = np.min(cross_corr)

    # All correlation arrays are of the same size
    lag_arr = np.arange(0, np.size(cross_corr))

    # Regular nonscaled plot of fish speed
    if reg == 1:
        wfs = np.arange(0, end-window_size, window_size)
        plt.figure()
        plt.title(f"{dataset_name}: Nonscaled Velocity Plot Between Two Fish")
        plt.xlabel(f"Velocity [distance/second]; WS = {window_size}")
        plt.plot(wfs, fish1_speed, color='turquoise')
        plt.plot(wfs, fish2_speed, color='mediumslateblue')

    # Speed Cross-Correlation Plot
    if corr == 1:
        # Superimpose smoothed frames on
        # top of cross correlation plot
        lag_smooth = lag_arr[0:np.size(cross_corr)]
        cross_corr_smooth = cross_corr[0:np.size(cross_corr)]

        xnew = np.linspace(lag_smooth.min(), lag_smooth.max(), 50)
        spl = make_interp_spline(lag_smooth, cross_corr_smooth)
        ynew = spl(xnew)

        plt.figure()
        plt.title(f"{dataset_name}: Velocity Cross-Correlation Plot Between Two Fish")
        plt.xlabel(f"Lag; WS = {window_size}")
        plt.ylabel("Correlation Coefficients")
        plt.plot(lag_arr, cross_corr, color='royalblue')
        plt.plot(xnew, ynew, color='red')

    # Speed Autocorrelation Plots
    if auto == 1:
        # Calculate autocorrelations
        fish1_auto_corr = sm.tsa.stattools.ccf(fish1_norm, fish1_norm, adjusted=False)
        fish2_auto_corr = sm.tsa.stattools.ccf(fish2_norm, fish2_norm, adjusted=False)

        plt.figure()
        plt.title(f"{dataset_name}: Velocity Autocorrelation Plot Between Two Fish [Fish1]")
        plt.ylim(cross_corr_min, cross_corr_max)
        plt.xlabel(f"Lag; WS = {window_size}")
        plt.ylabel("Correlation Coefficients")
        plt.plot(lag_arr, fish1_auto_corr, color='turquoise')

        plt.figure()
        plt.title(f"{dataset_name}: Velocity Autocorrelation Plot Between Two Fish [Fish2]")
        plt.ylim(cross_corr_min, cross_corr_max)
        plt.xlabel(f"Lag; WS = {window_size}")
        plt.ylabel("Correlation Coefficients")
        plt.plot(lag_arr, fish2_auto_corr, color='mediumslateblue')

    # Speed Shuffled Cross-Correlation Plot
    if shuff == 1:
        # Calculate the cross correlation of fish1
        # with a permutation of fish2 -- resulting 
        # plot should be cenetered at 0              
        fish2_shuffled = np.random.permutation(fish2_norm)
        
        shuffled_cross_corr = sm.tsa.stattools.ccf(fish1_norm, 
        fish2_shuffled, adjusted=False)

        plt.figure()
        plt.title(f"{dataset_name}: Shuffled Velocity Cross-Correlation Plot Between Two Fish")
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

   
def main():
    dataset = "results_SocPref_3c_2wpf_nk3_ALL.csv"
    pos_data = load_data(dataset, 3, 5)
    angle_data = load_data(dataset, 5, 6)
    window_size = 1
    dataset_name = re.search('\d[a-z]_\d[a-z]{3}_[a-z]{1,2}\d', dataset).group()
    end_of_arr = np.shape(pos_data)[1] 

    fish_speeds_tuple = get_speed(pos_data[0], pos_data[1], end_of_arr,
    window_size)
    fish1_speed = fish_speeds_tuple[0]
    fish2_speed = fish_speeds_tuple[1]
    fish_angles_tuple = get_angle(angle_data[0], angle_data[1], end_of_arr, 
    window_size)
    fish1_angles = fish_angles_tuple[0]
    fish2_angles = fish_angles_tuple[1]
    
    fish_velocities_tuple = get_velocity_mag(fish1_speed, fish2_speed, angle_data[0],
    angle_data[1], end_of_arr, window_size)

    fish1_velocities = fish_velocities_tuple[0]
    fish2_velocities = fish_velocities_tuple[1]

    # minimizer(fish1_velocities, fish2_velocities, dataset_name, 200)

    # velocity_frames_plots(fish1_velocities, fish2_velocities, dataset_name, 3000)

    correlation_plots(fish1_velocities, fish2_velocities, fish1_angles, 
    fish2_angles, dataset_name, end_of_arr, window_size)
    # correlation_plots(fish1_speed, fish2_speed, fish1_angles, 
    # fish2_angles, dataset_name, end_of_arr, window_size)

    plt.show()


if __name__ == "__main__":
    main()
