# -*- coding: utf-8 -*-
# sim_correlated_vectors.py
"""
Author:   Raghuveer Parthasarathy
Created on Mon Dec 12 11:27:36 2022
Last modified on Dec. 15, 2022

Description
-----------
Generating artificial datasets with tunable auto- and cross-correlation,
to see assess how our analysis handles them.
"""

import numpy as np
import matplotlib.pyplot as plt

#%% random numbers

np.random.seed()

#%% Parameters

N = 500 # number of datapoints
alpha = 0.9 # autoregression coefficient

#%% Random, uncorrelated variables. Normal distribution.
z1_mean = 0.0
z_std = 0.2
z2_mean = 0.0
z1 = np.random.normal(z1_mean, z_std, (N,))
z2 = np.random.normal(z2_mean, z_std, (N,))

#%% Autocorrelated variables

def make_autoregress1_vector(x, z, alpha):
    # Make autocorrelated vector y -- first-order autoregression
    # random component (z) and component correlated with the 
    #    single timepoint before it, 
    # alpha: autoregression coefficient
    for j in range(1,N):
        x[j] = alpha*x[j-1] + z[j]
    return x

# Autocorrelate
x1 = z1.copy()
x1 = make_autoregress1_vector(x1, z1, alpha)
x2 = z2.copy()
x2 = make_autoregress1_vector(x2, z2, alpha)
    

#%% Plot Time Series

t = np.arange(N)
plt.figure()
plt.plot(t, x1, color=(0.2, 0.4, 0.9), label='x1')
plt.plot(t, x2, color=(0.3, 0.8, 0.3), label='x2')
plt.xlabel('time (index)')
plt.legend()


#%% Calculate autocorrelation function
# Autocorrelation. 

def calc_acf(y):
    # Calculate the autocorrelation function of y.
    # Use numpy correlate, which will return + and - lag
    # ACF is symmetric, so just take the second half
    # Normalize by var(y)*N
    N = len(y)
    y_acf = np.correlate(y, y, mode='full')[(N-1):]
    y_acf = y_acf/np.var(y)/N
    return y_acf

    
x1_acf = calc_acf(x1)
x2_acf = calc_acf(x2)

lag_max = int(np.round(N/5)) # Just plot acf to this point
t_lag = np.arange(lag_max)
plt.figure()
plt.plot(t_lag, x1_acf[0:lag_max], label='est ACF x1', color=[0.2, 0.3, 1.0])
plt.plot(t_lag, x2_acf[0:lag_max], label='est ACF x2', color=[0.2, 0.9, 0.2])
plt.plot(t_lag, alpha**t_lag, label='theory', color=[0.9, 0.6, 0.2])
plt.plot(t_lag, (-1/lag_max + 2/np.sqrt(lag_max))*np.ones_like(t_lag), 
         color=[0.4, 0.4, 0.4], label='95% conf. of 0')
plt.plot(t_lag, (-1/lag_max - 2/np.sqrt(lag_max))*np.ones_like(t_lag), 
         color=[0.4, 0.4, 0.4])

plt.xlabel('lag')
plt.ylabel('Autocorrelation function')
plt.legend()

#%% Estimate autoregression parameter

# The approximate estimator for alpha is acf[1]/acf[0]! 
#   Asymptotically, acf[0] = 1 by our normalization, but in practice
#   it's not exactly 1

alpha1_est = x1_acf[1]/x1_acf[0]
alpha2_est = x2_acf[1]/x2_acf[0]
# Approximate uncertainty in alpha
alpha1_est_sigma = np.sqrt((1-alpha1_est**2)/len(x1))
alpha2_est_sigma = np.sqrt((1-alpha2_est**2)/len(x2))

# one sigma value for alpha == 0
alpha_zero_sigma = 1/np.sqrt(len(x1))

print(f'True Parameter: alpha = {alpha:.3f}')

print('Estimated parameters: ')
print(f'   alpha1_est : {alpha1_est:.3f} +/- {alpha1_est_sigma:.3f}')
print(f'   alpha2_est : {alpha2_est:.3f} +/- {alpha2_est_sigma:.3f}')
print(f'   two sigma (95% CI) value for alpha == 0: {2*alpha_zero_sigma:.3f}')

#%% Assess *apparent* cross-correlation

# Note that x1, x2 are uncorrelated!
N = len(x1)
ccf = np.correlate(x1, x2, mode='full')/np.std(x1)/np.std(x2)/N

# one sigma value for no cross-correlation
ccf_zero_sigma = 1/np.sqrt(len(x1))

k_array = np.arange(-(N-1),N)
plt.figure()
plt.plot(k_array, ccf)
plt.plot(k_array, ccf_zero_sigma*np.ones_like(k_array), color=[0.4, 0.4, 0.4])
plt.plot(k_array, -ccf_zero_sigma*np.ones_like(k_array), color=[0.4, 0.4, 0.4])
plt.xlabel('lag, k')
plt.ylabel('Cross-correlation')

#%% Cross-correlated vectors

rho_C = 0.3 # cross-correlation magnitude

# Make Cross-correlated vectors
y1 = x1.copy()
y2 = x1*rho_C + x2*np.sqrt(1 - rho_C**2)

#%% Plots

t = np.arange(N)
plt.figure()
plt.plot(t, y1, color=(0.6, 0.8, 0.7), label='x1')
#plt.plot(t, x2, color=(0.7, 0.6, 0.8), label='x2')
plt.plot(t, y2, color=(0.2, 0.8, 0.9), label='y1')
#plt.plot(t, y2, color=(0.3, 0.8, 0.7), label='y2')
#plt.plot(t, v1, color=(0.8, 0.2, 0.1), label='v1==y1', linewidth=2.0)
#plt.plot(t, v2, color=(0.9, 0.6, 0.2), label='v2', linewidth=2.0)
plt.xlabel('time (index)')
plt.legend()
