# -*- coding: utf-8 -*-
# make_Fig5B_turning.py

"""
Author:   Raghuveer Parthasarathy
Created on April 15, 2026
Last modified April 15, 2026 -- Raghuveer Parthasarathy

Description
-----------

For making "Figure 5B", turning angle v
distance and relative orientation

Based on compare_all_2wkData.py
Calls load_all_expt_data() from compare_all_2wkData.py
to load pickle files of two week old zebrafish behavior data. 
Then calls other functions to make plots.

Datasets and colors:
    Pairs, Light (color darkorange)
    Pairs, Light, Time-shifted Fish0 (color saddlebrown)
    Pairs, Dark (color cornflowerblue)
    Pairs, Dark, Time-shifted Fish0 (color darkblue)
    Single, Light (gold)
    Single, Dark (slategrey)


Instructions:

Modify the expBaseStrm et., for each dataset of interest.

"""

import os
import numpy as np
from behavior_identification import make_turning_angle_plots
from IO_toolkit import plot_2D_heatmap, slice_2D_histogram

from compare_all_2wkData import load_all_expt_data


parentPath = r'C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs'


#%% Experiment list

_PAIRS_PATH   = r'2 week old - Sept2025 control pairs in dark vs light New Tracking'

expt_list = [
    {'expBaseStr': 'TwoWk_Sept2025',
     'cond_str':   'Light_Cond_2',
     'exptName':   'TwoWk_Light',
     'plot_color': 'darkorange',
     'CSVPath':    _PAIRS_PATH},
    {'expBaseStr': 'TwoWk_Sept2025_TS0',
     'cond_str':   'Light_Cond_2',
     'exptName':   'TwoWk_Light_TIMESHIFT0',
     'plot_color': 'saddlebrown',
     'CSVPath':    _PAIRS_PATH},
    {'expBaseStr': 'TwoWk_Sept2025',
     'cond_str':   'Dark_Cond_1',
     'exptName':   'TwoWk_Dark',
     'plot_color': 'cornflowerblue',
     'CSVPath':    _PAIRS_PATH},
    {'expBaseStr': 'TwoWk_Sept2025_TS0',
     'cond_str':   'Dark_Cond_1',
     'exptName':   'TwoWk_Dark_TIMESHIFT0',
     'plot_color': 'darkblue',
     'CSVPath':    _PAIRS_PATH},
]

all_expts = load_all_expt_data(expt_list, parentPath)

#%% Make turning plots for 5B (not difference plots)

closeFigures = True
if closeFigures:
    print('Turning angle plots: Closing Figure Windows.')

distance_type =  'head_head_distance' # or head_head_distance
if distance_type == 'closest_distance':
    shortDistanceStr = 'ClDist'
elif distance_type == 'head_head_distance':
    shortDistanceStr = 'HHDist'
else:
    raise ValueError('Invalide distance type')

for exptName in all_expts.keys():
    saved_pair_turning_outputs = make_turning_angle_plots(
                            all_expts[exptName]['datasets'], 
                            exptName = exptName,
                            distance_type = distance_type,
                            color = all_expts[exptName]['plot_color'], 
                            Nbins = (19,25),
                            cmap = 'berlin',
                            outputFileNameBase = f'{exptName}', 
                            outputFileNameExt = 'svg',
                            closeFigures = closeFigures,
                            writeCSVs = False)
    all_expts[exptName]["turn_2Dhist_mean"] = saved_pair_turning_outputs[0]
    all_expts[exptName]["turn_2Dhist_sem"] = saved_pair_turning_outputs[1]
    all_expts[exptName]["turn_2Dhist_X"] = saved_pair_turning_outputs[2]
    all_expts[exptName]["turn_2Dhist_Y"] = saved_pair_turning_outputs[3]   

#%% Differences of turning angle or bending angle or relative orientation. 

makeSlicePlots = False
if makeSlicePlots==False:
    print('Not making slices of difference plots.')

# distance_type should be defined earlier
if distance_type == 'closest_distance':
    distanceStr = 'Closest Distance'
    ylabelStr = 'Closest Distance (mm)'
elif distance_type == 'head_head_distance':
    distanceStr = 'HH Distance'
    ylabelStr = 'Head-Head Distance (mm)'
else:
    raise ValueError('Invalide distance type')
    
optionString = 'Turning'
keyString = 'turn'
mask_by_sem_limit_degrees = 2.0 # show points with s.e.m. < this
colorRange = (-1.5*np.pi/180.0, 1.5*np.pi/180.0)
xlim = (-np.pi, np.pi) # for relative orientation
zlim = (-4*np.pi/180, 4*np.pi/180)
# Mesh is the same for all datasets
X = all_expts['TwoWk_Light']["turn_2Dhist_X"]
Y = all_expts['TwoWk_Light']["turn_2Dhist_Y"]
unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi]
xlabelStr = 'Relative Orientation (deg)'
clabelStr= f'Mean {optionString} Angle (degrees)'

cmap = 'berlin' # 'RdYlBu_r'
plot_type = 'heatmap'
xlabelStr = 'Relative Orientation (degrees)'
outputExtension = '.svg' # for saving images

outputFileNameBase = f'Difference in {optionString} Angle, ({distanceStr}), '
mask_by_sem_limit = (mask_by_sem_limit_degrees * np.pi/180.0
                        if mask_by_sem_limit_degrees is not None else None)

# Each tuple: (title_label, filename_label, exptKey_A, exptKey_B, slice_color)
comparisons = [
    ('Light - TimeShift Light', 'Light - TS0Light', 'TwoWk_Light', 'TwoWk_Light_TIMESHIFT0', 'peru'),
    ('Light - Dark',            'Light - Dark',     'TwoWk_Light', 'TwoWk_Dark',             'darkseagreen'),
    ('Dark - TimeShift Dark',   'Dark - TS0Dark',   'TwoWk_Dark',  'TwoWk_Dark_TIMESHIFT0',  'blue'),
]

for cmp_title, cmp_fname, keyA, keyB, slice_color in comparisons:
    d = (all_expts[keyA][f"{keyString}_2Dhist_mean"] -
            all_expts[keyB][f"{keyString}_2Dhist_mean"])
    sem_A = all_expts[keyA][f"{keyString}_2Dhist_sem"]
    sem_B = all_expts[keyB][f"{keyString}_2Dhist_sem"]
    d_unc = np.sqrt(sem_A**2 + sem_B**2) if (sem_A is not None and sem_B is not None) else None

    if mask_by_sem_limit_degrees is None:
        titleStr = f'{cmp_title}: {optionString} Angle Probability'
    else:
        titleStr = f'{cmp_title}: {optionString} Angle; unc. < {mask_by_sem_limit_degrees:.1f} deg'

    plot_2D_heatmap(d, X, Y, Z_unc=d_unc,
                    titleStr=titleStr, xlabelStr=xlabelStr, ylabelStr=ylabelStr,
                    clabelStr=clabelStr, colorRange=colorRange, cmap=cmap,
                    unit_scaling_for_plot=unit_scaling_for_plot,
                    mask_by_sem_limit=mask_by_sem_limit,
                    outputFileName=outputFileNameBase + cmp_fname + outputExtension,
                    closeFigure=False)

    if makeSlicePlots:
        for d_range in [(0.0, 5.0), (5.0, 15.0)]:
            if d_range[0] == 0.0:
                title_dist = f'{cmp_title}: {optionString} Angle for d < {d_range[1]:.2f} mm'
            else:
                title_dist = f'{cmp_title}: {optionString} Angle for {d_range[0]:.1f} < d < {d_range[1]:.1f} mm'
            slice_2D_histogram(d, X, Y, d_unc,
                                slice_axis='x', other_range=d_range,
                                titleStr=title_dist, xlabelStr=xlabelStr,
                                zlabelStr=clabelStr, ylabelStr=ylabelStr,
                                zlim=zlim, xlim=xlim,
                                plot_z_zero_line=True, plot_vert_zero_line=True,
                                unit_scaling_for_plot=unit_scaling_for_plot,
                                color=slice_color,
                                outputFileName=outputFileNameBase + f'{cmp_fname} {d_range[0]:.1f}-{d_range[1]:.1f} mm' + outputExtension,
                                closeFigure=False)