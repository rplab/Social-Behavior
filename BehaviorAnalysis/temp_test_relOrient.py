# -*- coding: utf-8 -*-
# temp_test_relOrient.py
"""
Author:   Raghuveer Parthasarathy
Created on Fri Aug 23 07:49:08 2024
Last modified on Fri Aug 23 07:49:08 2024

Description
-----------

For test of get_relative_orientation() on uniform angles and positions

Inputs:
    
Outputs:
    

"""

import numpy as np
import matplotlib.pyplot as plt
from behavior_identification import get_relative_orientation

# For test of get_relative_orientation() on uniform angles and positions
CSVcolumns = {"angle_data_column": 0, "head_column_x": 1, "head_column_y": 2}

Nframes = 15000
Nfish = 2
angles_random = np.random.uniform(low=0, high=2.0*np.pi, size=(Nframes,Nfish))
x_random = np.random.uniform(low=0, high=1000, size=(Nframes,Nfish))
y_random = np.random.uniform(low=0, high=1000, size=(Nframes,Nfish))

dataset = {"all_data": np.stack((angles_random, x_random, y_random), axis=1)}

relative_orientation = get_relative_orientation(dataset, CSVcolumns)

plt.figure()
plt.hist(relative_orientation[:,0], bins=30, alpha = 0.4)
plt.hist(relative_orientation[:,1], bins=30, alpha = 0.4)
plt.xlabel('Relative Orientation (rad.)', fontsize=14)