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
from toolkit import *
import statsmodels.api as sm
from scipy.interpolate import make_interp_spline
# ---------------------------------------------------------------------------

def get_speed(fish1_pos, fish2_pos, end, window_size):
    idx_1, idx_2 = 0, window_size
    fish1_speed = np.array([])
    fish2_speed = np.array([])
  
    while idx_2 < end:   # end of the array for both fish
        # Velocity = change in distance / change in time; 
        # Velocity expressed in distance/second 
        # 1 second = 25 wfs
        fish1_speed = np.append(fish1_speed, 
        (get_head_distance_traveled(fish1_pos, idx_1, idx_2) * 25) / window_size) 
        fish2_speed = np.append(fish2_speed, 
        (get_head_distance_traveled(fish2_pos, idx_1, idx_2) * 25) / window_size)

        idx_1, idx_2 = idx_1+window_size, idx_2+window_size

    return fish1_speed, fish2_speed

def correlation_plots(fish1_speed, fish2_speed, dataset_name, window_size,
reg=0, corr=0, auto=0, shuff=0, smooth=1):
    
    # Normalize both fish data sets by their mean
    fish1_norm = (fish1_speed - np.mean(fish1_speed)) / np.mean(fish1_speed)
    fish2_norm = (fish2_speed - np.mean(fish2_speed)) / np.mean(fish2_speed)

    # Calculate the cross correlation
    cross_corr = sm.tsa.stattools.ccf(fish1_norm, fish2_norm, adjusted=False)

    # Calculate autocorrelations
    fish1_auto_corr = sm.tsa.stattools.ccf(fish1_norm, fish1_norm, adjusted=False)
    fish2_auto_corr = sm.tsa.stattools.ccf(fish2_norm, fish2_norm, adjusted=False)

    # Calculate the cross correlation with
    # shuffled datasets to ensure it's correct
    # fish1_shuffled = np.random.permutation(fish1_norm)              
    fish2_shuffled = np.random.permutation(fish2_norm)
    
    shuffled_cross_corr = sm.tsa.stattools.ccf(fish1_norm, 
    fish2_shuffled, adjusted=False)
    
    # All correlation arrays are of the same size
    lag_arr = np.arange(0, np.size(cross_corr))
  
    # Regular nonscaled plot of fish velocities
    if reg == 1:
        wfs = np.arange(0, np.size(fish1_speed), window_size)
        plt.figure()
        plt.title(f"{dataset_name}: Nonscaled Speed Plot Between Two Fish")
        plt.xlabel("Velocity [distance/second]")
        plt.plot(wfs, fish1_speed, color='turquoise')
        plt.plot(wfs, fish2_speed, color='mediumslateblue')

    # Cross-Correlation Plot
    if corr == 1:
        plt.figure()
        plt.title(f"{dataset_name}: Cross-Correlation Plot Between Two Fish")
        plt.xlabel("Lag")
        plt.ylabel("Correlation Coefficients")
        plt.plot(lag_arr, cross_corr, color='royalblue')

    # Autocorrelation Plots
    if auto == 1:
        plt.figure()
        plt.title(f"{dataset_name}: Autocorrelation Plot Between Two Fish [Fish1]")
        plt.xlabel("Lag")
        plt.ylabel("Correlation Coefficients")
        plt.plot(lag_arr, fish1_auto_corr, color='turquoise')

        plt.figure()
        plt.title(f"{dataset_name}: Autocorrelation Plot Between Two Fish [Fish2]")
        plt.xlabel("Lag")
        plt.ylabel("Correlation Coefficients")
        plt.plot(lag_arr, fish1_auto_corr, color='mediumslateblue')

    # Shuffled Cross-Correlation Plot
    if shuff == 1:
        plt.figure()
        plt.title(f"{dataset_name}: Shuffled Cross-Correlation Plot Between Two Fish")
        plt.xlabel("Lag")
        plt.ylabel("Correlation Coefficients")
        plt.plot(lag_arr, shuffled_cross_corr, color='mediumslateblue')

    # Smoothed version of the Cross-Correlation 
    # plot for the first 100 or so data points
    if smooth == 1: 
        lag_smooth = lag_arr[0:51]
        cross_corr_smooth = cross_corr[0:51]

        xnew = np.linspace(lag_smooth.min(), lag_smooth.max(), 800)
        spl = make_interp_spline(lag_smooth, cross_corr_smooth)
        ynew = spl(xnew)
        
        plt.title(f"{dataset_name}: Smoothed Cross-Correlation Plot")
        plt.xlabel("Lag")
        plt.ylabel("Correlation Coefficients")
        plt.plot(xnew, ynew,color='mediumslateblue')

    plt.show()
   
  

def main():
    pos_data = load_data("results_SocPref_3c_2wpf_nk2_ALL.csv", 3, 5)
    dataset_name = re.search('\d[a-z]_\d[a-z]{3}_[a-z]{1,2}\d', 
    'results_SocPref_3c_2wpf_nk1_ALL.csv').group()
    end_of_arr = np.shape(pos_data)[1] 

    fish_speeds_tuple = get_speed(pos_data[0], pos_data[1], end_of_arr, 1)
    
    fish1_speed = fish_speeds_tuple[0]
    fish2_speed = fish_speeds_tuple[1]
    correlation_plots(fish1_speed, fish2_speed, dataset_name, 1)


if __name__ == "__main__":
    main()
