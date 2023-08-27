# -*- coding: utf-8 -*-
# misc_behaviors.py
"""
Author:   Raghuveer Parthasarathy
Created on Fri Jul 21 10:12:09 2023
Last modified on July 22 2023

Description
-----------

For testing various behavior identification methods
"""

import numpy as np
import matplotlib.pyplot as plt

    



def simple_unwrap(angles):
    """
    Simple unwrapping, for cases in which I can't use np.unwrap (e.g. steps
    in element number)
    Only unwraps by 1 * 2*pi

    Parameters
    ----------
    angles : (numpy array of float); angles (radians)

    Returns
    -------
    angles_out : unwrapped, radians

    """
    angles_out = angles
    angles_out[angles > np.pi] -= 2*np.pi
    angles_out[angles < -np.pi] += 2*np.pi
    return angles_out

