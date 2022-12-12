# -*- coding: utf-8 -*-
# sim_correlated_vectors.py
"""
Author:   Raghuveer Parthasarathy
Created on Mon Dec 12 11:27:36 2022
Last modified on Mon Dec 12 11:27:36 2022

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

N = 100 # number of datapoints
rho_A = 0.7 # autocorrelation magnitude, instantaneous
rho_C = 0.3 # cross-correlation magnitude, instantaneous

# Random, uncorrelated variables. Normal distribution.
# Note that variances need to be equal for the resulting correlation to 
#    equal rho.
# Will use zero-mean x-series
x1_mean = 0.0
x_std = 0.2
x2_mean = 0.0
x1 = np.random.normal(x1_mean, x_std, (N,))
x2 = np.random.normal(x2_mean, x_std, (N,))

def autocorr_y(y, rho):
    # Make each element of y correlated with the one before it, 
    # with correlation coefficient rho
    for j in range(1,N):
        y[j] = y[j-1]*rho + y[j]*np.sqrt(1 - rho**2)
    return y
    
# Autocorrelate
y1 = x1.copy()
y1 = autocorr_y(y1, rho_A)
y2 = x2.copy()
y2 = autocorr_y(y2, rho_A)
    

# Cross-correlate
v1 = y1.copy()
v2 = y1*rho_C + y2*np.sqrt(1 - rho_C**2)

#%% Plots

t = np.arange(N)
plt.figure()
plt.plot(t, x1, color=(0.6, 0.8, 0.7), label='x1')
plt.plot(t, x2, color=(0.7, 0.6, 0.8), label='x2')
plt.plot(t, y1, color=(0.2, 0.8, 0.9), label='y1')
plt.plot(t, y2, color=(0.3, 0.8, 0.7), label='y2')
plt.plot(t, v1, color=(0.8, 0.2, 0.1), label='v1==y1', linewidth=2.0)
plt.plot(t, v2, color=(0.9, 0.6, 0.2), label='v2', linewidth=2.0)
plt.xlabel('time (index)')
plt.legend()




