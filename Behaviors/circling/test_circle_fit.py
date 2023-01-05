#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By      : Raghuveer Parthasarathy (MATLAB source code)
# Created Date    : May 20, 2022
# Translated By   : Estelle Trieu (Python)
# Translation Date: June 22, 2022
#----------------------------------------------------------------------------
""" Simple program to illustrate and test the usage of the following function
to fit a "best fit" circle to a set of points: 

    CircleFitByTaubin.m
    G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                Space Curves Defined By Implicit Equations, With 
                Applications To Edge And Range Image Segmentation",
    IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
 
 Downloaded from Mathworks: 
 https://www.mathworks.com/matlabcentral/fileexchange/22678-circle-fit-taubin-method?s_tid=srchtitle

 Doesn't perform any analysis of goodness of fit; just makes a plot

 For more information, see
 https://people.cas.uab.edu/~mosya/cl/
 https://blogs.mathworks.com/pick/2009/04/10/fitting-circles-again-and-ellipses/
 https://people.cas.uab.edu/~mosya/cl/CM1nova.pdf -- esp. Appendix 1
""" 
# ---------------------------------------------------------------------------
import numpy
import matplotlib.pyplot as plt
from circle_fit_taubin import TaubinSVD
# ---------------------------------------------------------------------------

# Create a set of points, scattered around a circle
theta = [i * (numpy.pi / 180) for i in range(360)] # angle values, radians
N = len(theta) # number of points

# True values (i.e. coordinates at these angles exactly on a circle)
R = 2.5  # radius
x_ctr = 1.0  # center x position
y_ctr = 0.5  # center y position

noise_std = R / 10  # standard deviation of "R" values
R_rand = R + noise_std * numpy.random.normal(R, noise_std, N) # N Gaussian-distributed points

x = x_ctr + R_rand * numpy.cos(theta)
y = y_ctr + R_rand * numpy.sin(theta)

XY = [x, y]

# Output: Par = [xc yc R] is the fitting circle:
taubin_input = []
for pair in zip(*XY):
    taubin_input.append([pair[0], pair[1]])


# XY = numpy.row_stack((x, y))
# Par = TaubinSVD(XY)

Par = TaubinSVD(taubin_input)


# Display the points and the best-fit circle
plt.figure()

plt.plot(x, y, 'o')
plt.plot(Par[0] + Par[2] * numpy.cos(theta), Par[1] + Par[2] * numpy.sin(theta), '-')

plt.show()
