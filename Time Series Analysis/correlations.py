# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#-------------------------------------------------------------------------------
# Created By  : Estelle Trieu & Raghu Parthasarathy
# Created Date: 1/3/2023
# version ='1.0'
# ------------------------------------------------------------------------------
import re
import matplotlib.pyplot as plt
import numpy as np
from motion_tracking import *
from toolkit import load_data 
from toolkit import normalize_by_mean
# ------------------------------------------------------------------------------

# ------------------------- MODULE IN PROGRESS!!! ------------------------------
def calc_acf(y):
    # Calculate the autocorrelation function of y.
    # Use numpy correlate, which will return + and - lag
    # ACF is symmetric, so just take the second half
    # Normalize by var(y)*N
    N = len(y)
    y_acf = np.correlate(y, y, mode='full')[(N-1):]
    y_acf = y_acf/np.var(y)/N
    return y_acf


def calc_AR1_residual(x, alpha):
    # subtracting the AR(1) model with parameter alpha from series x
    x = x - np.mean(x)
    for j in np.arange(1, len(x)):
        x[j] = x[j] - alpha*(x[j-1])
    return x


def acf_plots(fish1_motion, fish2_motion, motion, dataset_name, N, est_only=True):
    # Note: Either fish speed OR velocity can be passed as the first 
    # two parameters 
    if motion == 's':
        type = 'Speed'
    elif motion == 'v':
        type = 'Velocity'
    else: 
        type = 'Angle'

    # Normalize both speed fish data sets by their mean
    fish1_norm = normalize_by_mean(fish1_motion)
    fish2_norm = normalize_by_mean(fish2_motion)

    # Calculate autocorrelations
    fish1_acf = calc_acf(fish1_norm)
    fish2_acf = calc_acf(fish2_norm)

    lag_max = int(np.round(N/5)) # Just plot acf to this point

    # Case when there are fewer points in the autocorrelations
    # than in the lag_max value 
    fish1_acf_size = np.shape(fish1_acf)[0]
    if fish1_acf_size < lag_max:
        lag_max = fish1_acf_size
    t_lag = np.arange(lag_max)
        
    # Estimate autoregression parameter
    # The approximate estimator for alpha is acf[1]/acf[0]
    # Asymptotically, acf[0] = 1 by our normalization, 
    # but in practice
    alpha1_est = fish1_acf[1]/fish1_acf[0]
    alpha2_est = fish2_acf[1]/fish2_acf[0]

    if est_only == True:
        return (alpha1_est, alpha2_est)

    # Approximate uncertainty in alpha
    alpha1_est_sigma = np.sqrt((1-alpha1_est**2)/len(fish1_norm))
    alpha2_est_sigma = np.sqrt((1-alpha2_est**2)/len(fish2_norm))

    # one sigma value for alpha == 0
    alpha_zero_sigma = 1/np.sqrt(len(fish1_norm))

    print('Estimated parameters: ')
    print(f'   alpha1_est : {alpha1_est:.3f} +/- {alpha1_est_sigma:.3f}')
    print(f'   alpha2_est : {alpha2_est:.3f} +/- {alpha2_est_sigma:.3f}')
    print(f'   two sigma (95% CI) value for alpha == 0: {2*alpha_zero_sigma:.3f}')

    plt.figure()
    plt.title(f"{dataset_name} {type}")
    plt.plot(t_lag, fish1_acf[0:lag_max], label='est ACF x1', color=[0.2, 0.3, 1.0])
    plt.plot(t_lag, fish2_acf[0:lag_max], label='est ACF x2', color=[0.2, 0.9, 0.2])
    plt.plot(t_lag, (-1/lag_max + 2/np.sqrt(lag_max))*np.ones_like(t_lag), 
            color=[0.4, 0.4, 0.4], label='95% conf. of 0')
    plt.plot(t_lag, (-1/lag_max - 2/np.sqrt(lag_max))*np.ones_like(t_lag), 
            color=[0.4, 0.4, 0.4])

    plt.xlabel('lag')
    plt.ylabel('Autocorrelation function')
    plt.legend()


def ccf_plots(fish1_motion, fish2_motion, motion, dataset_name, N):
    # Note: Either fish speed OR velocity can be passed as the first 
    # two parameters 
    if motion == 's':
        type = 'Speed'
    elif motion == 'v':
        type = 'Velocity'
    else: 
        type = 'Angle'

    # Normalize both speed fish data sets by their mean
    fish1_norm = normalize_by_mean(fish1_motion)
    fish2_norm = normalize_by_mean(fish2_motion)

    # Unfiltered cross-correlation
    N = len(fish1_norm)
    ccf = np.correlate(fish1_norm, fish2_norm, mode='full')/np.std(fish1_norm)/np.std(fish2_norm)/N

    # Filter x1, x2 by subtracting the AR(1) model
    alpha_est_tuple = acf_plots(fish1_motion, fish2_motion, motion, dataset_name, 
    N, est_only=True)
    fish1p = calc_AR1_residual(fish1_norm, alpha_est_tuple[0])
    fish2p = calc_AR1_residual(fish2_norm, alpha_est_tuple[1])

    # Cross-correlation of the filtered signals
    ccf_filtered = np.correlate(fish1p, fish2p, mode='full')/np.std(fish1p)/np.std(fish2p)/N

    # one sigma value for no cross-correlation
    ccf_zero_sigma = 1/np.sqrt(len(fish1_norm))

    lag_max = int(np.round(N/5)) # Just plot ccf to +/- this point
    k_array = np.arange(-lag_max, lag_max + 1)
    k_array_ind = k_array + N - 1

    # 3 px, simple average 1D smoothing kernel
    smooth_kernel = np.array([0.25, 0.5, 0.25])

    plt.figure()
    plt.plot(k_array, ccf[k_array_ind], label='original x1, x2')
    plt.plot(k_array, ccf_filtered[k_array_ind], label='filtered x1p, x2p')
    plt.plot(k_array, np.convolve(ccf_filtered[k_array_ind], 
                                smooth_kernel, mode='same'), 
            label='filtered x1p, x2p; Smoothed', color=[0.7, 0.2, 0.1])
    plt.plot(k_array, 2*ccf_zero_sigma*np.ones_like(k_array_ind), 
            color=[0.4, 0.4, 0.4], label='95% conf. of 0')
    plt.plot(k_array, -2*ccf_zero_sigma*np.ones_like(k_array_ind), 
            color=[0.4, 0.4, 0.4])
    plt.xlabel('lag, k')
    plt.ylabel('Cross-correlation')
    plt.title(f"{dataset_name} {type}")
    plt.legend()
    
    # Find correlation coefficient at k=0
    orignal = np.interp(0, k_array, ccf[k_array_ind])
    filtered = np.interp(0, k_array, ccf_filtered[k_array_ind])
    smooth = np.interp(0, k_array, np.convolve(ccf_filtered[k_array_ind], smooth_kernel, mode='same'))

    print(f"original: {orignal}")
    print(f"filtered: {filtered}")
    print(f"smooth: {smooth}")
    print(f"bottom: {(-2*ccf_zero_sigma*np.ones_like(k_array_ind))[0]}")

    


def main():
    dataset = "results_SocPref_3c_2wpf_nk3_ALL.csv"
    pos_data = load_data(dataset, 3, 5)
    angle_data = load_data(dataset, 5, 6)
    window_size = 10
    dataset_name = re.search('\d[a-z]_\d[a-z]{3}_[a-z]{1,2}\d', dataset).group()
    end_of_arr = np.shape(pos_data)[1] 

    # fish_speeds_tuple = get_speed(pos_data[0], pos_data[1], end_of_arr,
    # window_size)
    # fish1_speed = fish_speeds_tuple[0]
    # fish2_speed = fish_speeds_tuple[1]

    fish_angles_tuple = get_angle(angle_data[0], angle_data[1], end_of_arr, 
    window_size)
    fish1_angles = fish_angles_tuple[0]
    fish2_angles = fish_angles_tuple[1]
    
    # fish_velocities_tuple = get_velocity(fish1_speed, fish2_speed, angle_data[0],
    # angle_data[1], end_of_arr, window_size)
    # fish1_velocities = fish_velocities_tuple[0]
    # fish2_velocities = fish_velocities_tuple[1]

    # acf_plots(fish1_speed, fish2_speed, 's', dataset_name, end_of_arr, False)

    ccf_plots(fish1_angles, fish2_angles, 'a', dataset_name, end_of_arr)

    plt.show()

    


if __name__ == "__main__":
    main()