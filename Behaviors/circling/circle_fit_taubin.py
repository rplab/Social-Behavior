#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By      : Ron Shaar, Lori J, and Kevin Gaastra
# Last Commit     : March 28, 2017
# ---------------------------------------------------------------------------
""" Function to fit a "best fit" circle to a set of points: 

    G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                Space Curves Defined By Implicit Equations, With 
                Applications To Edge And Range Image Segmentation",
    IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
 
 Downloaded from GitHub: 
 https://github.com/PmagPy/PmagPy/blob/master/SPD/lib/lib_curvature.py
""" 
# ---------------------------------------------------------------------------
from __future__ import division
from past.utils import old_div
import numpy
# ---------------------------------------------------------------------------

def TaubinSVD(XY):
    """
    algebraic circle fit
    input: list [[x_1, y_1], [x_2, y_2], ....]
    output: a, b, r.  a and b are the center of the fitting circle, and r is the radius
     Algebraic circle fit by Taubin
      G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                  Space Curves Defined By Implicit Equations, With
                  Applications To Edge And Range Image Segmentation",
      IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
    """
    XY = numpy.array(XY)
    X = XY[:,0] - numpy.mean(XY[:,0]) # norming points by x avg
    Y = XY[:,1] - numpy.mean(XY[:,1]) # norming points by y avg
    centroid = [numpy.mean(XY[:,0]), numpy.mean(XY[:,1])]
    Z = X * X + Y * Y  
    Zmean = numpy.mean(Z)
    Z0 = old_div((Z - Zmean), (2. * numpy.sqrt(Zmean)))
    ZXY = numpy.array([Z0, X, Y]).T
    U, S, V = numpy.linalg.svd(ZXY, full_matrices=False) # 
    V = V.transpose()
    A = V[:,2]
    A[0] = old_div(A[0], (2. * numpy.sqrt(Zmean)))
    A = numpy.concatenate([A, [(-1. * Zmean * A[0])]], axis=0)
    a, b = (-1 * A[1:3]) / A[0] / 2 + centroid 
    r = numpy.sqrt(A[1]*A[1]+A[2]*A[2]-4*A[0]*A[3])/abs(A[0])/2;
    return a,b,r
