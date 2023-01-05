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
import numpy as np
from toolkit import *
import statsmodels.api as sm
from scipy.ndimage import uniform_filter1d # for 1D boxcar smoothing
from scipy.optimize import curve_fit  # for curve fitting
from scipy.interpolate import make_interp_spline
# ---------------------------------------------------------------------------

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


def decay_exp_model(t, C0, tau):
    # inputs:
    #    t : independent variable, probably time array
    #    C0 : amplitude
    #    tau : decay time
    c = C0*np.exp(-t/tau)
    return c


def get_param_guess(cross_corr, lag_arr):
    # Get Paramater Guess 
    smooth_filterSize = 10
    c_smooth = uniform_filter1d(cross_corr, size=smooth_filterSize)

    # For linear regression of log(c), will ignore negative values
    c_smooth_nonNeg = c_smooth[c_smooth>0]
    t_cNonNeg = lag_arr[c_smooth>0]

    c_smooth_nonNeg_shape = np.shape(c_smooth_nonNeg)
    t_cNonNeg_shape = np.shape(t_cNonNeg)

    if t_cNonNeg_shape != c_smooth_nonNeg_shape:
        np.reshape(t_cNonNeg, c_smooth_nonNeg_shape)

    # regression: log(c) = param[0] + param[1]*t
    #    so c_0 = exp(param[0]) and tau = -1/param[1]
    regression_param = np.linalg.lstsq(np.stack((t_cNonNeg, np.ones_like(t_cNonNeg)), axis=1), 
                                    np.log(c_smooth_nonNeg), rcond=None)[0]
    param_guess = (np.exp(regression_param[1]), -1.0/regression_param[0])
    return param_guess


def coarse_time_plots(fish1_motion, fish2_motion, dataset_name, block_size, 
entire_fit=0):
    fish1_norm = (fish1_motion - np.mean(fish1_motion)) / np.mean(fish1_motion)
    fish2_norm = (fish2_motion - np.mean(fish2_motion)) / np.mean(fish2_motion)
    
    lag_arr = np.arange(0, np.size(fish1_norm))

    # Cross correlation graph of the ENTIRE dataset
    # with curve fit superimposed on top
    if entire_fit == 1:
        # Calculate the velocity cross correlation
        cross_corr = sm.tsa.stattools.ccf(fish1_norm, fish2_norm, adjusted=False)
        param_guess = get_param_guess(cross_corr, lag_arr)
        popt, pcov = curve_fit(decay_exp_model, lag_arr, cross_corr, 
                    p0 = param_guess, bounds = ((0, 0), (np.Inf, np.Inf)))
        param_err = np.sqrt(np.diag(pcov))
        c0 = popt[0]
        tau = popt[1]
        c0_unc = param_err[0]
        tau_unc = param_err[1]
        
        plt.figure()
        plt.title(f"{dataset_name}")
        plt.plot(lag_arr, cross_corr, color='orange', label=f"cross correlation")
        plt.plot(lag_arr, c0 * np.exp(-lag_arr / tau), color='blue', 
        label=f'fit c0={c0:.4f}; unc={c0_unc:.4f}')
        plt.legend()
    
    idx_1, idx_2 = 0, block_size
    # similar to a block of lag_arr, but we use
    # only the first block_size frames instead 
    param_time = np.arange(0, block_size)  
    array_idx = 0 
    arr_size = (15000 // block_size)
    c0_arr = np.empty(arr_size)
    tau_arr = np.empty(arr_size)
    c0_unc_arr = np.empty(arr_size)
    tau_unc_arr = np.empty(arr_size)
    correlation_coefficients = np.empty(arr_size)
    
    plt.figure()
    for i in range(0, np.size(fish1_norm)+1, block_size):
        # Calculate the velocity cross correlation
        cross_corr = sm.tsa.stattools.ccf(fish1_norm[idx_1:idx_2], 
        fish2_norm[idx_1:idx_2], adjusted=False)
        curr_lag_arr = lag_arr[idx_1:idx_2]

        if np.shape(cross_corr) == np.shape(param_time):
            param_guess = get_param_guess(cross_corr, param_time)
        else:
            param_time = param_time[:block_size-1]
            param_guess = get_param_guess(cross_corr, param_time)
       
        # Fix parameter guess to be within the bounds of curve fit
        if param_guess[1] < 0:
            param_guess = (param_guess[0], -1 * param_guess[1])

        popt, pcov = curve_fit(decay_exp_model, param_time, cross_corr, 
                    p0 = param_guess, bounds = ((0, 0), (np.Inf, np.Inf)))
        param_err = np.sqrt(np.diag(pcov))
        c0 = popt[0]
        tau = popt[1]
        c0_unc = param_err[0]
        tau_unc = param_err[1]
     
        c0_arr[array_idx] = c0
        tau_arr[array_idx] = tau
        c0_unc_arr[array_idx] = c0_unc
        tau_unc_arr[array_idx] = tau_unc
        
        curr_fit = c0 * np.exp((-1) * param_time / tau)

        # Correlation coefficient is found at value of 
        # function fit with tau=0
        correlation_coefficients[array_idx] = np.max(curr_fit)

        plt.plot(curr_lag_arr, cross_corr, color='peachpuff')
        plt.plot(curr_lag_arr, curr_fit)

        array_idx += 1
        idx_1, idx_2 = idx_1+block_size, idx_2+block_size
    plt.title(f"{dataset_name}: Correlation fit; block_size = {block_size}")

    # Plot of Coarse Time vs. Correlation Coefficients 
    plt.figure()
    x = np.arange(np.size(c0_arr))
    plt.scatter(x, c0_arr, color='cyan')
    plt.errorbar(x, c0_arr, yerr=c0_unc_arr, fmt="*", color="r")
    plt.title(f"{dataset_name}: Coarse Time vs. C0")
    plt.xlabel(f"Coarse Time; block_size={block_size}")
    plt.ylabel("C0")
  
    # Plot of Coarse Time vs. Tau Values 
    plt.figure()
    plt.scatter(x, tau_arr, color='purple')
    plt.errorbar(x, tau_arr, yerr=tau_unc_arr, fmt="*", color="r")
    plt.title(f"{dataset_name}: Coarse Time vs. Tau")
    plt.xlabel(f"Coarse Time; block_size={block_size}")
    plt.ylabel("Tau")

    # Table of Correlation Coefficients & Tau Values 
    plt.figure()
    df = pd.DataFrame(np.column_stack((correlation_coefficients, c0_unc_arr,
    tau_arr, tau_unc_arr)), columns=['c0', 'c0 uncertainty', 'tau', 'tau uncertainty'])
    plt.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.axis('off')

    plt.show()
   
    
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

   
def main():
    dataset = "results_SocPref_3c_2wpf_k2_ALL.csv"
    pos_data = load_data(dataset, 3, 5)
    angle_data = load_data(dataset, 5, 6)
    window_size = 10
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
    
    fish_velocities_tuple = get_velocity(fish1_speed, fish2_speed, angle_data[0],
    angle_data[1], end_of_arr, window_size)
    fish1_velocities = fish_velocities_tuple[0]
    fish2_velocities = fish_velocities_tuple[1]

    # coarse_time_plots(fish1_velocities, fish2_velocities, dataset_name, 1500)
    # velocity_frames_plots(fish1_velocities, fish2_velocities, dataset_name, 3000)
    correlation_plots(fish1_velocities, fish2_velocities, fish1_angles, fish2_angles,
    'v', dataset_name, end_of_arr, window_size)

    # correlation_plots(fish1_speed, fish2_speed, fish1_angles, 
    # fish2_angles, dataset_name, end_of_arr, window_size)

    # plt.show()


if __name__ == "__main__":
    main()