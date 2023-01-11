# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu & Raghu Parthasarathy
# Created Date: 1/5/2022
# version ='1.0'
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from toolkit import normalize_by_mean
import statsmodels.api as sm
from motion_tracking import * 
from scipy.ndimage import uniform_filter1d # for 1D boxcar smoothing
from scipy.optimize import curve_fit  # for curve fitting
# ---------------------------------------------------------------------------
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
    fish1_norm = normalize_by_mean(fish1_motion)
    fish2_norm = normalize_by_mean(fish2_motion)
    
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
   
