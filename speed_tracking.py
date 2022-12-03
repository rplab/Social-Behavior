# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 11/2/2022
# version ='1.0'
# ---------------------------------------------------------------------------
import re
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
from scipy import optimize
from toolkit import *
import statsmodels.api as sm
from scipy.interpolate import make_interp_spline
# ---------------------------------------------------------------------------
marker = itertools.cycle((',', '+', '.', 'o', '*', '1', 'P', 'd')) 
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

def obj_func(params, c0, t):
    # params = [c0, tau]
    model = params[0]*np.exp((-1)*t / params[1])
    squared_error = (c0 - model)**2
    res = np.sum(squared_error)       
    return res 

def normalize_data(fish_data):
    return (fish_data - np.mean(fish_data)) / np.mean(fish_data)

def get_c0_and_tau(cross_corr, block_size, lag_arr):
    params_guess = np.array((np.max(cross_corr), block_size))        
    estimated_params = optimize.minimize(obj_func, params_guess, 
    args=(cross_corr, lag_arr), bounds = ((0, None), (0, None)))
    return estimated_params

def get_c_alpha_arr(c0_gauss_dist_range, tau, cross_corr, lag_arr):
    # Create a range of testable c_alpha and tau_alpha values centered 
    # at (c_alpha, chi_squared_min) or (tau_alpha, chi_squared_min)
    c_chi_squared = np.array([])
    for i in range(np.shape(c0_gauss_dist_range)[0]):
        c_chi_squared = np.append(c_chi_squared, 
        obj_func((c0_gauss_dist_range[i], tau), cross_corr, lag_arr))
    return c_chi_squared

def get_tau_alpha_arr(tau_gauss_dist_range, c0, cross_corr, lag_arr):
    tau_chi_squared = np.array([])
    for i in range(np.shape(tau_gauss_dist_range)[0]):
        tau_chi_squared = np.append(tau_chi_squared, 
        obj_func((c0, tau_gauss_dist_range[i]), cross_corr, lag_arr))
    return tau_chi_squared

def chi_squared_plots(fish1_velocity, fish2_velocity, dataset_name, block_size,
entire=1):
    fish1_norm = normalize_data(fish1_velocity)
    fish2_norm = normalize_data(fish2_velocity)

    # All velocity arrays are of the same size
    lag_arr = np.arange(0, np.size(fish1_norm))
   
    if entire == 1:
        cross_corr = sm.tsa.stattools.ccf(fish1_norm, fish2_norm, adjusted=False)
        estimated_params = get_c0_and_tau(cross_corr, block_size, lag_arr)
        c0 = estimated_params['x'][0]
        tau = estimated_params['x'][1]
        chi_squared_min = obj_func((c0, tau), cross_corr, lag_arr)

        # Get the corresponding chi square values for c_alpha
        c0_gauss_dist_range = np.arange(0, 2.5 * c0, c0 / 50)
        c_chi_squared = get_c_alpha_arr(c0_gauss_dist_range, tau, cross_corr, lag_arr)

        # Get the corresponding chi square values for tau_alpha
        tau_gauss_dist_range = np.arange(tau / 4, 2 * tau, tau / 8)
        tau_chi_squared = get_tau_alpha_arr(tau_gauss_dist_range, c0, cross_corr, lag_arr)
       
        # Plot c_alpha or tau_alpha vs. chi^2
        plt.figure()
        plt.title(f"chi^2 vs. c_alpha; min: (c0={c0:.2f}, c_chi_squared_min={chi_squared_min:.2f})")
        plt.xlabel("c_alpha")
        plt.ylabel("chi^2")
        plt.scatter(c0_gauss_dist_range, c_chi_squared, color='purple')
        plt.plot(c0, chi_squared_min, marker='x', color='red')

        plt.figure()
        plt.title(f"chi^2 vs. tau; min: (tau={tau:.2f}, tau_chi_squared_min={chi_squared_min:.2f})")
        plt.xlabel("tau")
        plt.ylabel("chi^2")
        plt.scatter(tau_gauss_dist_range, tau_chi_squared, color='green')
        plt.plot(tau, chi_squared_min, marker='x', color='red')
        plt.show()
    
    # c_alpha or tau_alpha vs. chi^2 Plots for Data Blocks
    idx_1, idx_2 = 0, block_size
    plt.figure()
    for i in range(0, np.size(fish1_norm)+1, block_size):
        # Calculate the velocity cross correlation
        cross_corr = sm.tsa.stattools.ccf(fish1_norm[idx_1:idx_2], 
        fish2_norm[idx_1:idx_2], adjusted=False)
        curr_lag_arr = lag_arr[idx_1:idx_2]
        estimated_params = get_c0_and_tau(cross_corr, block_size, curr_lag_arr)

        c0 = estimated_params['x'][0]
        tau = estimated_params['x'][1]

        chi_squared_min = obj_func((c0, tau), cross_corr, curr_lag_arr)

        # Create a range of testable c_alpha and tau_alpha values centered 
        # at (c_alpha, chi_squared_min) or (tau_alpha, chi_squared_min)
        c0_gauss_dist_range = np.arange(0, 2 * chi_squared_min, chi_squared_min / 4)
        tau_gauss_dist_range = np.arange(tau / 4, 2 * tau, tau / 8)

        # Get the corresponding chi square values for c_alpha
        c_chi_squared = get_c_alpha_arr(c0_gauss_dist_range, tau, cross_corr, curr_lag_arr)
       
        plt.scatter(c0_gauss_dist_range, c_chi_squared, marker = next(marker))
        plt.plot(c0, chi_squared_min, marker='x', color='red')

        idx_1, idx_2 = idx_1+block_size, idx_2+block_size
    plt.show()

def coarse_time_plots(fish1_velocity, fish2_velocity, dataset_name, block_size, 
entire_fit=1):
    fish1_norm = (fish1_velocity - np.mean(fish1_velocity)) / np.mean(fish1_velocity)
    fish2_norm = (fish2_velocity - np.mean(fish2_velocity)) / np.mean(fish2_velocity)

    # All velocity arrays are of the same size
    lag_arr = np.arange(0, np.size(fish1_norm))

    # Cross correlation graph of the ENTIRE dataset
    # with fit function superimposed on top
    if entire_fit == 1:
        cross_corr = sm.tsa.stattools.ccf(fish1_norm, fish2_norm, adjusted=False)
        estimated_params = get_c0_and_tau(cross_corr, block_size, lag_arr)
        c0 = estimated_params['x'][0]
        tau = estimated_params['x'][1]

        plt.figure()
        plt.title(f"{dataset_name}")
        plt.plot(lag_arr, cross_corr, color='orange', label=f"cross correlation")
        plt.plot(lag_arr, c0 * np.exp(-lag_arr / tau), color='blue', 
        label=f'fit c0={c0:.4f}')
        plt.legend()
    
    
    idx_1, idx_2 = 0, block_size
    c0_arr = np.array([])
    tau_arr = np.array([])
    uncertainty_arr = np.array([])
    correlation_coefficients = np.array([])

    # Cross correlation graph of blocks of dataset
    # with fit function superimposed on top
    plt.figure()

    for i in range(0, np.size(fish1_norm)+1, block_size):
        # Calculate the velocity cross correlation
        cross_corr = sm.tsa.stattools.ccf(fish1_norm[idx_1:idx_2], 
        fish2_norm[idx_1:idx_2], adjusted=False)
        curr_lag_arr = lag_arr[idx_1:idx_2]

        params_guess = np.array((np.max(cross_corr), block_size))
        estimated_params = optimize.minimize(obj_func, params_guess,    # minimize function
        args=(cross_corr, curr_lag_arr), bounds = ((0, None), (0, None)))

        c0 = estimated_params['x'][0]
        tau =  estimated_params['x'][1]

        c0_arr = np.append(c0_arr, estimated_params['x'][0])
        tau_arr = np.append(tau_arr, estimated_params['x'][1])
        uncertainty_arr = np.append(uncertainty_arr, obj_func((c0, tau), cross_corr, curr_lag_arr) / 2)

        curr_fit = c0 * np.exp((-1) * curr_lag_arr / tau)

        # Correlation coefficient is found at value of 
        # function fit with tau=0
        correlation_coefficients = np.append(correlation_coefficients, np.max(curr_fit))

        plt.plot(curr_lag_arr, cross_corr, color='peachpuff')
        plt.plot(curr_lag_arr, curr_fit)
        idx_1, idx_2 = idx_1+block_size, idx_2+block_size
    
    plt.title(f"{dataset_name}: Correlation fit; block_size = {block_size}")

    # Plot of Coarse Time vs. Correlation Coefficients 
    plt.figure()
    x = np.arange(np.size(correlation_coefficients))
    plt.scatter(x, correlation_coefficients, color='cyan')
    plt.errorbar(x, correlation_coefficients, yerr=uncertainty_arr, fmt="*", color="r")
    plt.title(f"{dataset_name}: Coarse Time vs. Correlation Coefficient")
    plt.xlabel(f"Coarse Time; block_size={block_size}")
    plt.ylabel("Correlation Coefficient")

    # Plot of Coarse Time vs. Tau Values 
    plt.figure()
    x = np.arange(np.size(tau_arr))
    plt.scatter(x, tau_arr, color='purple')
    plt.title(f"{dataset_name}: Coarse Time vs. Tau")
    plt.errorbar(x, tau_arr, yerr=uncertainty_arr, fmt="*", color="r")
    plt.xlabel(f"Coarse Time; block_size={block_size}")
    plt.ylabel("Tau")

    # Table of Correlation Coefficients & Tau Values 
    plt.figure()
    df = pd.DataFrame(np.column_stack((correlation_coefficients, tau_arr)), 
    columns=['c0', 'tau'])
    plt.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.axis('off')

    plt.show()
    

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
    dataset = "results_SocPref_3c_2wpf_k2_ALL.csv"
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

    chi_squared_plots(fish1_velocities, fish2_velocities, dataset_name, 1875)

    # velocity_frames_plots(fish1_velocities, fish2_velocities, dataset_name, 3000)

    # correlation_plots(fish1_velocities, fish2_velocities, fish1_angles, 
    # fish2_angles, dataset_name, end_of_arr, window_size)
    # correlation_plots(fish1_speed, fish2_speed, fish1_angles, 
    # fish2_angles, dataset_name, end_of_arr, window_size)

    # plt.show()


if __name__ == "__main__":
    main()
