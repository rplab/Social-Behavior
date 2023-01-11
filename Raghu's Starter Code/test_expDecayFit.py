# -*- coding: utf-8 -*-
# test_expDecayFit.py
"""
Author:   Raghuveer Parthasarathy
Created on Sun Dec  4 09:43:53 2022
Last modified on Sun Dec  4, 2022

Description
-----------
Generate a decaying exponential with noise and fit to determine
decay amplitude, decay time.
Goal: What methods work?
See also notes document, "Fitting a decaying exponential notes"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.ndimage import uniform_filter1d # for 1D boxcar smoothing
from scipy.optimize import curve_fit  # for curve fitting

def obj_func(params, c, t):
    # Inputs:
    #    params = [c0, tau]
    model = params[0]*np.exp((-1)*t / params[1])
    squared_error = (c - model)**2
    res = np.sum(squared_error)  
    return res 

def create_decayExp(c0 = 1, tau = 2000, N = 15000, eta = 0.1):
    # Create an array that has a decaying exponential form:
    #   c(t) = c0 * exp(-t/tau) + noise
    # where noise = Gaussian random 
    # Inputs 
    #    c0 : amplitude of exponential (i.e. t=0 value)
    #    tau : decay time
    #    N  : number of points
    #    eta : std. dev. of noise
    t = np.arange(N)
    c = c0*np.exp(-t/tau) + np.random.normal(0.0, scale = eta, size=t.shape)
    return c

def get_c0_and_tau(cross_corr, param_guess, lag_arr):
    # From Estelle; modified!
    # Fit decaying exponential to cross_corr
    # Inputs: 
    #    cross_corr : array of cross-corre,ation values
    #    param_guess : guess for initial value of *both* amplitude and decay time
    #    lag_array: array of lag times (almost certainly = indices of cross_corr)
    # Use 1e-10 or c0_guess/1000 as the tolerance for convergence
    estimated_params = optimize.minimize(obj_func, param_guess, 
                                         args=(cross_corr, lag_arr), 
                                         bounds = ((0, None), (0, None)),
                                         tol = np.min(np.array((1e-10, param_guess[0]/1000))), 
                                         options={'maxiter': 100, 'disp': True})
    # print(estimated_params)  # useful to see how many iterations it took!
    return estimated_params

def get_chi2_c0range(c0_range, tau, cross_corr, lag_arr):
    # Calculate chi^2 over a range of c_alpha and values centered 
    # Inputs:
    #    c0_range : range of c0 values
    #    tau : tau value (fixed) 
    #    cross_corr : array of cross-corre,ation values
    #    lag_array: array of lag times (almost certainly = indices of cross_corr)
    # Outputs
    #    c_chi_squared : array of chi^2 values
    c_chi_squared = np.zeros(c0_range.shape)
    for i in range(np.shape(c0_range)[0]):
        c_chi_squared[i] =  obj_func((c0_range[i], tau), cross_corr, lag_arr)
    return c_chi_squared

c0 = 0.05
tau= 2000
N = 15000
eta = 0.005
t = np.arange(N, dtype='float')
c = create_decayExp(c0, tau, N, eta)

#%% Initial parameter guess

# Initial guesses for parameters: use a simple linear regression, first
# smoothing the data

smooth_filterSize = 10
c_smooth = uniform_filter1d(c, size=smooth_filterSize)

# For linear regression of log(c), will ignore negative values
c_smooth_nonNeg = c_smooth[c_smooth>0]
t_cNonNeg = t[c_smooth>0]
plt.figure()
plt.scatter(t, c, c='c', label='data')
plt.plot(t_cNonNeg, c_smooth_nonNeg, c='k', label='smoothed')

# regression: log(c) = param[0] + param[1]*t
#    so c_0 = exp(param[0]) and tau = -1/param[1]
regression_param = np.linalg.lstsq(np.stack((t_cNonNeg, np.ones_like(t_cNonNeg)), axis=1), 
                              np.log(c_smooth_nonNeg), rcond=None)[0]
param_guess = (np.exp(regression_param[1]), -1.0/regression_param[0])

print('Initial parameter guess from linear fit: ')
print(f'    c0={param_guess[0]:.4f}, tau={param_guess[1]:.4f}')
plt.plot(t, param_guess[0]*np.exp(-t/param_guess[1]), 'g-', 
         label='linear fit')

#%% Fit: nonlinear least squares

estimated_params = get_c0_and_tau(c, param_guess, t)
c0_fit = estimated_params['x'][0]
tau_fit = estimated_params['x'][1]
min_chi2 = obj_func((c0_fit, tau_fit), c, t)

print('Nonlinear least squares: minimizing chi^2')
print(f'   N = {N}')
print(f'   c0 True = {c0:.4f}, estimated = {c0_fit:.4f}')
print(f'   tau True = {tau:.3f}, estimated = {tau_fit:.3f}')
print(f'   min chi^2 = {min_chi2:.4e}')

#%% Fit: scipy non-linear least squares, to use its error assessment
#        for parameters via the covariance matrix
    
def decay_exp_model(t, C0, tau):
    # inputs:
    #    t : independent variable, probably time array
    #    C0 : amplitude
    #    tau : decay time
    c = C0*np.exp(-t/tau)
    return c

# Use scipy's curve fitting. Note bounds syntax
# Again using the linear fit values for the initial guess.
popt, pcov = curve_fit(decay_exp_model, t, c, 
                       p0 = param_guess, 
                       bounds = ((0, 0), (np.Inf, np.Inf)))
c0_fit_scipy = popt[0]
tau_fit_scipy = popt[1]
param_err = np.sqrt(np.diag(pcov))
c0_fit_scipy_unc = param_err[0]
tau_fit_scipy_unc = param_err[1]

print('Nonlinear least squares: scipy curve fit')
print(f'   N = {N}')
print(f'   c0 True = {c0:.3f}, estimated = {c0_fit_scipy:.4f} +/- {c0_fit_scipy_unc:.4f}')
print(f'   tau True = {tau:.2f}, estimated = {tau_fit_scipy:.3f} +/- {tau_fit_scipy_unc:.3f}')


#%% Plot chi^2 vs. c0

c0_range= np.arange(0, 2.5*c0_fit, c0_fit/50)
chi2_c0_array = get_chi2_c0range(c0_range, tau_fit, c, t)


#%% Plots

plt.figure()
plt.scatter(t, c, c='c', label='data')
plt.plot(t, c_smooth, c='k', label='smoothed')
plt.plot(t, c0*np.exp(-t/tau_fit), 'm-', label='fit')
plt.plot(t, param_guess[0]*np.exp(-t/param_guess[1]), 'g-', 
         label='linear fit')

plt.ylabel('c')
plt.xlabel('t')
plt.title(f'N = {N}')
plt.legend()

plt.figure()
plt.scatter(c0_range, chi2_c0_array, c='k', marker='x')
plt.scatter(c0_fit, np.min(chi2_c0_array), c='r', marker='o')
plt.title(f'N = {N}')
plt.xlabel('c0')
plt.ylabel('chi^2')

ylim = plt.gca().get_ylim()
ylim = (0, ylim[1]) # force ylim to be zero
plt.ylim(ylim)
xlim = plt.gca().get_xlim()
plt.plot(xlim, 2*min_chi2*np.array((1, 1)), ':')
