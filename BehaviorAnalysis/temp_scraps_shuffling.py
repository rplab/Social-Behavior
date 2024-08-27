# -*- coding: utf-8 -*-
# file_name.py
"""
Author:   Raghuveer Parthasarathy
Created on Fri Aug 23 09:42:54 2024
Last modified on Fri Aug 23 09:42:54 2024

Description
-----------

Inputs:
    
Outputs:
    

"""

import numpy as np
import matplotlib.pyplot as plt
import os


#%% Replace Fish 1 with shuffled Fish 0.
# Note: flawed because IDs are not maintained

datasets_light_shuffle0 = datasets.copy()

for j in range(len(datasets)):
    datasets_light_shuffle0[j]["all_data"][:,2:,1] = \
        np.random.permutation(datasets_light_shuffle0[j]["all_data"][:,2:,0])
    datasets_light_shuffle0[j]["relative_orientation"] = \
            get_relative_orientation(datasets_light_shuffle0[j], CSVcolumns)  

# Get relative orientation, plot
    
relorient_all_shuffle0 = combine_all_values_constrained(datasets_light_shuffle0, keyName='relative_orientation', dilate_plus1 = False)
plot_probability_distr(relorient_all_shuffle0, bin_width = 0.1, bin_range = [0, None], xlabelStr = 'Rel. orientation (radians)', titleStr = 'Rel. orientation, 2 week light, 0+shuffle0')

#%% Replace all heading angles with random number.

datasets_light_random_heading = datasets.copy()

headingColumn = CSVcolumns["angle_data_column"]
for j in range(len(datasets)):
    datasets_light_random_heading[j]["all_data"][:,headingColumn,:] = \
        np.random.uniform(low=0.0, high=2*np.pi, 
                          size = (datasets_light_random_heading[j]["all_data"].shape[0],
                                  datasets_light_random_heading[j]["all_data"].shape[2]))
    datasets_light_random_heading[j]["relative_orientation"] = \
            get_relative_orientation(datasets_light_random_heading[j], CSVcolumns)  

# Get relative orientation, plot
    
relorient_all_random_heading = combine_all_values_constrained(datasets_light_random_heading, keyName='relative_orientation', dilate_plus1 = False)
plot_probability_distr(relorient_all_random_heading, bin_width = 0.1, bin_range = [0, None], 
                       xlabelStr = 'Rel. orientation (radians)', 
                       titleStr = 'Rel. orientation, 2 week light, randomHeading')
