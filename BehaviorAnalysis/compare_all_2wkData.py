# -*- coding: utf-8 -*-
# compare_all_2wkData.py

"""
Author:   Raghuveer Parthasarathy
Created on Wed Oct 15 08:59:05 2025
Last modified Feb. 6, 2026 -- Raghuveer Parthasarathy

Description
-----------

Load pickle files of two week old zebrafish behavior data. 
Make plots, for comparison.

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
from IO_toolkit import load_and_assign_from_pickle, \
    combine_images_to_tiff, plot_2D_heatmap, slice_2D_histogram
from behavior_identification import make_pair_fish_plots,  \
    make_bending_angle_plots, make_pair_1D_v_distance_plots, \
    make_turning_angle_plots
from behavior_identification_single import make_single_fish_plots
from toolkit import get_fps 
from behavior_correlations import plot_behaviorCorrelation,  \
    calc_corr_asymm, plot_corr_asymm, calcDeltaFramesEvents, bin_deltaFrames, calc_pAB, calcBehavCorrAllSets

#%%

def load_expt_data(pickleFileName1, pickleFileName2, exptName, 
                           plot_color, all_expts):
    
    all_position_data, variable_tuple = load_and_assign_from_pickle(pickleFileName1, 
                                                                    pickleFileName2)
    (datasets, CSVcolumns, expt_config, params, N_datasets, Nfish,
             basePath, dataPath, subGroupName) = variable_tuple
    
    
    all_expts[exptName] = {
        "datasets" : variable_tuple[0],
        "all_position_data" : all_position_data.copy(),
        "CSVcolumns" : variable_tuple[1],
        "expt_config" : variable_tuple[2],
        "params" : variable_tuple[3],
        "N_datasets" : variable_tuple[4],
        "Nfish" : variable_tuple[5],
        "basePath" : variable_tuple[6],
        "dataPath" : variable_tuple[7],
        "subGroupName" : variable_tuple[8],
        "plot_color" : plot_color
        }
    
    del all_position_data
    del variable_tuple
    
    return all_expts

#%%

all_expts = {}

parentPath = r'C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs'


#%% Two week old pairs, Light

# Modify these:
expBaseStr = 'TwoWk_Sept2025'  # for file paths and names
cond_str = 'Light_Cond_2'
exptName = 'TwoWk_Light'
plot_color = 'darkorange'

CSVPath = r'2 week old - Sept2025 control pairs in dark vs light New Tracking'
pickleFileName1 = os.path.join(parentPath, CSVPath, cond_str, 
                               f'{expBaseStr}_{cond_str}_positionData.pickle')
AnalysisPath = f'{expBaseStr}_{cond_str}_Analysis'
# Note: should be "_datasets" but probably file path was too long
pickleFileName2 = os.path.join(parentPath, CSVPath, cond_str, AnalysisPath, 
                               f'{expBaseStr}_{cond_str}_datas.pickle')

all_expts = load_expt_data(pickleFileName1, pickleFileName2, exptName, 
                           plot_color, all_expts)


#%% Two week old pairs, Light -- TIMESHIFT fish 0

expBaseStr = 'TwoWk_Sept2025_TS0'
cond_str = 'Light_Cond_2'
exptName = 'TwoWk_Light_TIMESHIFT0'
plot_color = 'saddlebrown'

CSVPath = r'2 week old - Sept2025 control pairs in dark vs light New Tracking'
pickleFileName1 = os.path.join(parentPath, CSVPath, cond_str, 
                               f'{expBaseStr}_{cond_str}_positionData.pickle')
AnalysisPath = f'{expBaseStr}_{cond_str}_Analysis'

# Note: should be "_datasets" but probably file path was too long
pickleFileName2 = os.path.join(parentPath, CSVPath, cond_str, AnalysisPath, 
                               f'{expBaseStr}_{cond_str}_.pickle')

all_expts = load_expt_data(pickleFileName1, pickleFileName2, exptName, 
                           plot_color, all_expts)



#%% Two week old pairs, Dark

# Modify these:
expBaseStr = 'TwoWk_Sept2025'  # for file paths and names
cond_str = 'Dark_Cond_1'
exptName = 'TwoWk_Dark'
plot_color = 'cornflowerblue'

CSVPath = r'2 week old - Sept2025 control pairs in dark vs light New Tracking'
pickleFileName1 = os.path.join(parentPath, CSVPath, cond_str, 
                               f'{expBaseStr}_{cond_str}_positionData.pickle')
AnalysisPath = f'{expBaseStr}_{cond_str}_Analysis'
pickleFileName2 = os.path.join(parentPath, CSVPath, cond_str, AnalysisPath, 
                               f'{expBaseStr}_{cond_str}_datasets.pickle')

all_expts = load_expt_data(pickleFileName1, pickleFileName2, exptName, 
                           plot_color, all_expts)


#%% Two week old pairs, Dark  -- TIMESHIFT fish 0

expBaseStr = 'TwoWk_Sept2025_TS0'
cond_str = 'Dark_Cond_1'
exptName = 'TwoWk_Dark_TIMESHIFT0'
plot_color = 'darkblue'

CSVPath = r'2 week old - Sept2025 control pairs in dark vs light New Tracking'
pickleFileName1 = os.path.join(parentPath, CSVPath, cond_str, 
                               f'{expBaseStr}_{cond_str}_positionData.pickle')
AnalysisPath = f'{expBaseStr}_{cond_str}_Analysis'

# Note: should be "_datasets" but probably file path was too long
pickleFileName2 = os.path.join(parentPath, CSVPath, cond_str, AnalysisPath, 
                               f'{expBaseStr}_{cond_str}_dat.pickle')

all_expts = load_expt_data(pickleFileName1, pickleFileName2, exptName, 
                           plot_color, all_expts)

#%% Two week old single fish, Light

# Modify these:
expBaseStr = 'TwoWkSingle'  # for file paths and names
cond_str = 'Light_Cond_2'
exptName = 'TwoWk_Light_Single'
plot_color = 'gold'

CSVPath = r'2 week old - Sept2025 control single in dark vs light New Tracking'
pickleFileName1 = os.path.join(parentPath, CSVPath, cond_str, 
                               f'{expBaseStr}_{cond_str}_positionData.pickle')
AnalysisPath = f'{expBaseStr}_{cond_str}_Analysis'
pickleFileName2 = os.path.join(parentPath, CSVPath, cond_str, AnalysisPath, 
                               f'{expBaseStr}_{cond_str}_datasets.pickle')

all_expts = load_expt_data(pickleFileName1, pickleFileName2, exptName, 
                           plot_color, all_expts)

#%% Two week old single fish, Dark

# Modify these:
expBaseStr = 'TwoWkSingle'  # for file paths and names
cond_str = 'Dark_Cond_1'
exptName = 'TwoWk_Dark_Single'
plot_color = 'slategrey'

CSVPath = r'2 week old - Sept2025 control single in dark vs light New Tracking'
pickleFileName1 = os.path.join(parentPath, CSVPath, cond_str, 
                               f'{expBaseStr}_{cond_str}_positionData.pickle')
AnalysisPath = f'{expBaseStr}_{cond_str}_Analysis'
pickleFileName2 = os.path.join(parentPath, CSVPath, cond_str, AnalysisPath, 
                               f'{expBaseStr}_{cond_str}_datasets.pickle')

all_expts = load_expt_data(pickleFileName1, pickleFileName2, exptName, 
                           plot_color, all_expts)



#%% Make single fish plots (a lot!)

makeSingleFishPlots = False
if makeSingleFishPlots:
    
    closeFigures = True
    if closeFigures:
        print('Single fish plots: Closing Figure Windows.')
    
    for exptName in all_expts.keys():
        make_single_fish_plots(all_expts[exptName]['datasets'], 
                           exptName = exptName,
                           color = all_expts[exptName]['plot_color'], 
                           outputFileNameBase = f'JUNK{exptName} single_properties',
                           outputFileNameExt = 'png',
                           closeFigures = closeFigures,
                           writeCSVs = False)

#%% Make all pair plots (a lot!)
# Only for Nfish ==2

closeFigures = True
if closeFigures:
    print('Pair plots: Closing Figure Windows.')
    
distance_type = 'closest_distance' # or head_head_distance
#distance_type = 'head_head_distance' 

for exptName in all_expts.keys():
    if all_expts[exptName]['Nfish'] == 2:

        """
        make_pair_fish_plots(all_expts[exptName]['datasets'], 
                             exptName = exptName,
                             color = all_expts[exptName]['plot_color'], 
                             outputFileNameBase = f'{exptName} pair_properties', 
                             outputFileNameExt = 'png',
                             closeFigures = closeFigures,
                             writeCSVs = False)

        saved_pair_bending_outputs = make_bending_angle_plots(
                             all_expts[exptName]['datasets'], 
                             exptName = exptName,
                             distance_type = distance_type,
                             bending_threshold_deg = all_expts[exptName]['params']['bend_min_deg'],
                             color = all_expts[exptName]['plot_color'], 
                             outputFileNameBase = f'{exptName} pair_properties', 
                             outputFileNameExt = 'png',
                             closeFigures = closeFigures,
                             writeCSVs = False)
        all_expts[exptName]["bend_2Dhist_mean"] = saved_pair_bending_outputs[0]
        all_expts[exptName]["bend_2Dhist_sem"] = saved_pair_bending_outputs[1]
        all_expts[exptName]["bend_2Dhist_X"] = saved_pair_bending_outputs[2]
        all_expts[exptName]["bend_2Dhist_Y"] = saved_pair_bending_outputs[3]

        """ 
        saved_pair_turning_outputs = make_turning_angle_plots(
                             all_expts[exptName]['datasets'], 
                             exptName = exptName,
                             distance_type = distance_type,
                             color = all_expts[exptName]['plot_color'], 
                             outputFileNameBase = f'{exptName} pair_properties', 
                             outputFileNameExt = 'png',
                             closeFigures = closeFigures,
                             writeCSVs = False)
        all_expts[exptName]["turn_2Dhist_mean"] = saved_pair_turning_outputs[0]
        all_expts[exptName]["turn_2Dhist_sem"] = saved_pair_turning_outputs[1]
        all_expts[exptName]["turn_2Dhist_X"] = saved_pair_turning_outputs[2]
        all_expts[exptName]["turn_2Dhist_Y"] = saved_pair_turning_outputs[3]    
#%% Make pair plots of behavior v distance
# Only for Nfish ==2

makePairDistancePlots = False
if makePairDistancePlots:
    closeFigures = True
    if closeFigures:
        print('Pair plots of behavior v distance: Closing Figure Windows.')
        
    distanceKey = 'closest_distance_mm'      # can make it be 'head_head_distance_mm'
    for exptName in all_expts.keys():
        if all_expts[exptName]['Nfish'] == 2:
            make_pair_1D_v_distance_plots(all_expts[exptName]['datasets'], 
                                          exptName = exptName,
                                          distanceKey=distanceKey,
                                          bin_range=(0.0, 50.0), Nbins=20,
                                          color = all_expts[exptName]['plot_color'],
                                          outputFileNameBase = f'{exptName} behavior', 
                                          outputFileNameExt = 'png',
                                          closeFigures = closeFigures,
                                          writeCSVs = False)


#%% Make combination (multipage) images from individual PNGs

makeCombinationImages= False
if makeCombinationImages:
    fileNameStringsToCombine = ['speed', 'angularSpeed', 'radialpos', 'heading_angle',
                                'radialAlignment_angle', 'boutSpeed',
                                'speedAutocorr', 'distance_head_head', 
                                'distance_closest', 'rel_heading_angle',
                                'rel_orientation', 'rel_orientation_sum', 
                                'rel_orientation_abs_sum',
                                'orientation_distance_2D', 'IBI_v_dist_and_radialpos',
                                'IBI_v_dist_r_interior', 'IBI_v_dist_r_edge',
                                'bendAngle_distance_orientation_2D',
                                'bendAngle_v_orientation_small', 
                                'bendAngle_v_orientation_small_asymm',
                                'bendAngle_v_orientation_middle',
                                'bendAngle_v_orientation_middle_asymm',
                                'movingSpeed_v_HHdistance_orientation_2D',
                                'movingSpeed_v_HHdistance_lowOrientation',
                                'speedCrosscorr', 'speedCrosscorrDistBinned']
                                
    excludeStrings = ['angular', '', 'IBI', '', 
                      '', '', '', '',  
                      '', '', 'sum', 'abs', '', 
                      '', '', '', '', '', 'asymm', '', 'asymm', '',
                      '', '', 'DistBinned', '']
    
    if len(fileNameStringsToCombine) != len(excludeStrings):
        raise ValueError('List lengths not equal for combine_images strings.')
    for s, excl_s in zip(fileNameStringsToCombine, excludeStrings):
        print(f'Combining {s}')
        combine_images_to_tiff(filenamestring = s, 
                               path = r'C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\Code_current', 
                               ext = 'png', exclude_string=excl_s)
    

#%% Differences of turning angle or bending angle

# distance_type should be defined earlier

if distance_type == 'closest_distance':
    distanceStr = 'Closest Distance'
    ylabelStr = 'Closest Distance (mm)'
elif distance_type == 'head_head_distance':
    distanceStr = 'HH Distance'
    ylabelStr = 'Head-Head Distance (mm)'
else:
    raise ValueError('Invalide distance type')
    
calcDifferences = 'turning' # 'turning', 'bending', or 'none' [anything else]

if (calcDifferences == 'turning') or (calcDifferences == 'bending'):
    
    if calcDifferences == 'turning':
        optionString = 'Turning'
        keyString = 'turn'
        mask_by_sem_limit_degrees = 8.0 # show points with s.e.m. < this
        colorRange = (-3*np.pi/180.0, 3*np.pi/180.0)
        zlim = (-4*np.pi/180, 4*np.pi/180)
        # Mesh is the same for all datasets
        X = all_expts['TwoWk_Light']["turn_2Dhist_X"]
        Y = all_expts['TwoWk_Light']["turn_2Dhist_Y"]
    elif calcDifferences == 'bending':
        optionString = 'Bending'
        keyString = 'bend'
        mask_by_sem_limit_degrees = 6.0 # show points with s.e.m. < this
        colorRange = (-6*np.pi/180.0, 6*np.pi/180.0)
        zlim = (-15*np.pi/180, 15*np.pi/180)
        # Mesh is the same for all datasets
        X = all_expts['TwoWk_Light']["bend_2Dhist_X"]
        Y = all_expts['TwoWk_Light']["bend_2Dhist_Y"]
    else:
        raise ValueError('Calc differences must be bending or turning')
        
    unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
    cmap = 'RdYlBu_r'
    plot_type = 'heatmap'
    clabelStr= f'Mean {optionString} Angle (degrees)'
    xlabelStr = 'Relative Orientation (degrees)'

    outputFileNameBase = f'Difference in {optionString} Angle, ({distanceStr}), '

    # ylabelStr = 'Head-Head Distance (mm)'
    outputFileNameBase = f'Difference in {optionString} Angle, ({distanceStr}), '

    # Difference between light and time-shifted light
    dLight = all_expts['TwoWk_Light'][f"{keyString}_2Dhist_mean"] - \
        all_expts['TwoWk_Light_TIMESHIFT0'][f"{keyString}_2Dhist_mean"]
    dLight_unc = np.sqrt(all_expts['TwoWk_Light'][f"{keyString}_2Dhist_sem"]**2 + \
        all_expts['TwoWk_Light_TIMESHIFT0'][f"{keyString}_2Dhist_mean"]**2)
    titleStr = f'Light - TimeShift Light: {optionString} Angle; unc. < {mask_by_sem_limit_degrees:.1f} deg'
    outputFileName = outputFileNameBase + 'Light - TS0Light.png'

    plot_2D_heatmap(dLight, X, Y, Z_unc=dLight_unc,
                   titleStr=titleStr, xlabelStr=xlabelStr, ylabelStr=ylabelStr, 
                   clabelStr=clabelStr, colorRange = colorRange, cmap=cmap,
                   unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                   mask_by_sem_limit = mask_by_sem_limit_degrees*np.pi/180.0,
                   outputFileName=outputFileName, 
                   closeFigure=False)

    # Slice along bend or turn angle binned by distance and orientation, 
    # orientation axis, constrain distance: distance < 2.5 mm
    d_range = (0.0, 2.5)
    xlabelStr = 'Relative Orientation (deg)'
    titleStr = f'Light - TimeShift Light: {optionString} Angle for d < {d_range[1]:.2f} mm'
    zlabelStr = f'Mean {optionString} Angle (degrees)'
    xlim = (-np.pi, np.pi)
    color = 'peru'
    outputFileName = outputFileNameBase + 'Light - TS0Light small distance.png'
    slice_2D_histogram(dLight, X, Y, dLight_unc, 
                       slice_axis = 'x', other_range = d_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = ylabelStr, zlim = zlim, xlim = xlim, 
                       plot_z_zero_line = True, plot_vert_zero_line = True,
                       unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                       color = color, outputFileName=outputFileName,
                       closeFigure=False)
    # Slice along bend or turn angle binned by distance and orientation, 
    # orientation axis, constrain distance: distance 5 to 15 mm
    d_range = (5.0, 15.0)
    titleStr = f'Light - TimeShift Light: {optionString} Angle for {d_range[0]:.1f} < d < {d_range[1]:.1f} mm'
    outputFileName = outputFileNameBase + 'Light - TS0Light middle distance.png'
    slice_2D_histogram(dLight, X, Y, dLight_unc, 
                       slice_axis = 'x', other_range = d_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = ylabelStr, zlim = zlim, xlim = xlim, 
                       plot_z_zero_line = True, plot_vert_zero_line = True,
                       unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                       color = color, outputFileName=outputFileName,
                       closeFigure=False)


    # Difference between light and dark
    dLightDark = all_expts['TwoWk_Light'][f"{keyString}_2Dhist_mean"] - \
        all_expts['TwoWk_Dark'][f"{keyString}_2Dhist_mean"]
    dLightDark_unc = np.sqrt(all_expts['TwoWk_Light'][f"{keyString}_2Dhist_sem"]**2 + \
        all_expts['TwoWk_Dark'][f"{keyString}_2Dhist_mean"]**2)
    titleStr = f'Light - Dark: {optionString} Angle; unc. < {mask_by_sem_limit_degrees:.1f} deg'
    outputFileName = outputFileNameBase + 'Light - Dark.png'
        
    plot_2D_heatmap(dLightDark, X, Y, Z_unc=dLightDark_unc,
                   titleStr=titleStr, xlabelStr=xlabelStr, ylabelStr=ylabelStr, 
                   clabelStr=clabelStr, colorRange = colorRange, cmap=cmap,
                   unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                   mask_by_sem_limit = mask_by_sem_limit_degrees*np.pi/180.0,
                   outputFileName=outputFileName, 
                   closeFigure=False)
    # Slice along bend or turn angle binned by distance and orientation, 
    # orientation axis, constrain distance: distance < 2.5 mm
    d_range = (0.0, 2.5)
    xlabelStr = 'Relative Orientation (deg)'
    titleStr = f'Light - Dark: {optionString} Angle for d < {d_range[1]:.2f} mm'
    zlabelStr = f'Mean {optionString} Angle (degrees)'
    xlim = (-np.pi, np.pi)
    color = 'darkseagreen'
    outputFileName = outputFileNameBase + 'Light - Dark small distance.png'
    slice_2D_histogram(dLightDark, X, Y, dLightDark_unc, 
                       slice_axis = 'x', other_range = d_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = ylabelStr, zlim = zlim, xlim = xlim, 
                       plot_z_zero_line = True, plot_vert_zero_line = True,
                       unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                       color = color, outputFileName=outputFileName,
                       closeFigure=False)
    # Slice along bend or turn angle binned by distance and orientation, 
    # orientation axis, constrain distance: distance 5 to 15 mm
    d_range = (0.0, 2.5)
    titleStr = f'Light - Dark: {optionString} Angle for {d_range[0]:.1f} < d < {d_range[1]:.1f} mm'
    outputFileName = outputFileNameBase + 'Light - Dark middle distance.png'
    slice_2D_histogram(dLightDark, X, Y, dLightDark_unc, 
                       slice_axis = 'x', other_range = d_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = ylabelStr, zlim = zlim, xlim = xlim, 
                       plot_z_zero_line = True, plot_vert_zero_line = True,
                       unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                       color = color, outputFileName=outputFileName,
                       closeFigure=False)
    
    # Difference between dark and time-shifted dark
    dDark = all_expts['TwoWk_Dark'][f"{keyString}_2Dhist_mean"] - \
        all_expts['TwoWk_Dark_TIMESHIFT0'][f"{keyString}_2Dhist_mean"]
    dDark_unc = np.sqrt(all_expts['TwoWk_Dark'][f"{keyString}_2Dhist_sem"]**2 + \
        all_expts['TwoWk_Dark_TIMESHIFT0'][f"{keyString}_2Dhist_mean"]**2)
    titleStr = f'Difference in {optionString} Angle: Dark - TimeShift Dark; unc. < {mask_by_sem_limit_degrees:.1f} deg'
    outputFileName = outputFileNameBase + 'Dark - TS0Dark.png'
    plot_2D_heatmap(dDark, X, Y, Z_unc=dDark_unc,
                   titleStr=titleStr, xlabelStr=xlabelStr, ylabelStr=ylabelStr, 
                   clabelStr=clabelStr, colorRange = colorRange, cmap=cmap,
                   unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                   mask_by_sem_limit = mask_by_sem_limit_degrees*np.pi/180.0,
                   outputFileName=outputFileName, 
                   closeFigure=False)

    # Slice along bend or turn angle binned by distance and orientation, 
    # orientation axis, constrain distance: distance < 2.5 mm
    d_range = (0.0, 2.5)
    xlabelStr = 'Relative Orientation (deg)'
    titleStr = f'Dark - Timeshift Dark: {optionString} Angle for d < {d_range[1]:.2f} mm'
    zlabelStr = f'Mean {optionString} Angle (degrees)'
    xlim = (-np.pi, np.pi)
    color = 'blue'
    outputFileName = outputFileNameBase + 'Dark - TS0Dark small distance.png'
    slice_2D_histogram(dDark, X, Y, dDark_unc, 
                       slice_axis = 'x', other_range = d_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = ylabelStr, zlim = zlim, xlim = xlim, 
                       plot_z_zero_line = True, plot_vert_zero_line = True,
                       unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                       color = color, outputFileName=outputFileName,
                       closeFigure=False)
    # Slice along bend or turn angle binned by distance and orientation, 
    # orientation axis, constrain distance: distance 5 to 15 mm
    d_range = (5.0, 15.0)
    titleStr = f'Dark - Timeshift Dark: {optionString} Angle for {d_range[0]:.1f} < d < {d_range[1]:.1f} mm'
    outputFileName = outputFileNameBase + 'Dark - TS0Dark middle distance.png'
    slice_2D_histogram(dDark, X, Y, dDark_unc, 
                       slice_axis = 'x', other_range = d_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = ylabelStr, zlim = zlim, xlim = xlim, 
                       plot_z_zero_line = True, plot_vert_zero_line = True,
                       unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                       color = color, outputFileName=outputFileName,
                       closeFigure=False)


#%% Correlations of pair behaviors
# Note that this is slow!

calculateCorrelations = False

if calculateCorrelations:
    
    behavior_key_list = ['perp_noneSee', 'perp_oneSees', 'perp_bothSee', 
                         'contact_any', 'tail_rubbing_AP', 'tail_rubbing_P', 
                         'maintain_proximity', 'approaching_Fish0', 'approaching_Fish1', 
                         'fleeing_Fish0', 'fleeing_Fish1', 
                         'approaching_Fish_lowRelOrient0',  'approaching_Fish_lowRelOrient1', 
                         'fleeing_Fish_lowRelOrient0',  'fleeing_Fish_lowRelOrient1', 
                         'isActive_Fish_lowRelOrient0', 'isActive_Fish_lowRelOrient1', 
                         'isBending_Fish_lowRelOrient0', 'isBending_Fish_lowRelOrient1', 
                         'isMoving_Fish_lowRelOrient0', 'isMoving_Fish_lowRelOrient1', 
                         'Cbend_Fish_lowRelOrient0', 'Cbend_Fish_lowRelOrient1', 
                         'Jbend_Fish_lowRelOrient0', 'Jbend_Fish_lowRelOrient1', 
                         'Rbend_Fish_lowRelOrient0',  'Rbend_Fish_lowRelOrient1']
    behavior_key_list_subset1 = ['maintain_proximity', 'approaching_Fish_lowRelOrient0', 'approaching_Fish_lowRelOrient1', 'fleeing_Fish_lowRelOrient0', 'fleeing_Fish_lowRelOrient1', 'isBending_Fish_lowRelOrient0', 'isBending_Fish_lowRelOrient1', 'isMoving_Fish_lowRelOrient0', 'isMoving_Fish_lowRelOrient1']
    behavior_key_list_subset2 = ['perp_noneSee', 'perp_oneSees', 'perp_bothSee', 
                                 'contact_any', 'tail_rubbing_AP', 'tail_rubbing_P',
                                 'maintain_proximity', 'Cbend_Fish_lowRelOrient0', 
                                 'Cbend_Fish_lowRelOrient1']
    behavior_key_list_subset = behavior_key_list_subset1 + behavior_key_list_subset2
    
    binWidthFrames = 1 # bin size for delays (number of frames)
    halfFrameRange = 50 # max frame delay to consider
    
    for exptName in all_expts.keys():
        if all_expts[exptName]['Nfish'] == 2:

            print(f'\n\n -- {exptName} -- \n')
            this_datasets = datasets=all_expts[exptName]['datasets']
            fps = get_fps(this_datasets, fpstol = 1e-6)
            # behavior_key_list = get_behavior_key_list(datasets=all_expts[exptName]['datasets'])
            min_duration_behavior = 'maintain_proximity'
            min_duration_fr = 13
            behav_corr = calcDeltaFramesEvents(datasets = this_datasets, 
                                               behavior_key_list = behavior_key_list, 
                                               max_delta_frame = 150, 
                                               min_duration_behavior = min_duration_behavior , 
                                               min_duration_fr = min_duration_fr)
            behav_corr, binCenters = bin_deltaFrames(behav_corr, behavior_key_list, 
                                                     binWidthFrames = binWidthFrames, halfFrameRange = halfFrameRange, deleteDeltaFrames = True)
            behav_corr = calc_pAB(behav_corr, behavior_key_list, binCenters)
            behav_corr_allSets = calcBehavCorrAllSets(behav_corr, behavior_key_list, 
                                                      binCenters)
            exType = 'Light '  # e.g. 'Light ' include space at the end 
            behaviorA = 'maintain_proximity'
            behaviorB = ''
            plot_behaviorCorrelation(behav_corr_allSets['DeltaCorr'], binCenters, 
                                     behavior_key_list_subset1, behaviorA, behaviorB, 
                                     titleString = f'{exptName}' + r'$\Delta$P', 
                                     fps = fps, 
                                     plotShadedUnc = True, 
                                     outputFileName = f'{exptName}_behav_after_{behaviorA}_1.png')
            plot_behaviorCorrelation(behav_corr_allSets['DeltaCorr'], binCenters, 
                                     behavior_key_list_subset2, behaviorA, behaviorB, 
                                     titleString = f'{exptName}' + r'$\Delta$P', 
                                     fps = fps, 
                                     plotShadedUnc = True, 
                                     outputFileName = f'{exptName}_behav_after_{behaviorA}_2.png')
            behaviorA = 'contact_any'
            behaviorB = ''
            plot_behaviorCorrelation(behav_corr_allSets['DeltaCorr'], binCenters, 
                                     behavior_key_list_subset1, behaviorA, behaviorB, 
                                     titleString = f'{exptName}' + r'$\Delta$P', 
                                     fps = fps, 
                                     plotShadedUnc = True, 
                                     outputFileName = f'{exptName}_behav_after_{behaviorA}_1.png')
            plot_behaviorCorrelation(behav_corr_allSets['DeltaCorr'], binCenters, 
                                     behavior_key_list_subset2, behaviorA, behaviorB, 
                                     titleString = f'{exptName}' + r'$\Delta$P', 
                                     fps = fps, 
                                     plotShadedUnc = True, 
                                     outputFileName = f'{exptName}_behav_after_{behaviorA}_2.png')
            corr_asymm = calc_corr_asymm(behav_corr_allSets['DeltaCorr'], 
                                         behavior_key_list_subset, binCenters, 
                                         maxFrameDelay = None, normalization = 'none')
            plot_corr_asymm(corr_asymm, crange=None, 
                            titleString = f'{exptName}' + r'corr_asymm_$\Delta$P', 
                            outputFileName = f'{exptName}_corr_asymm.png')
else:
    print('\n\nNot calculating correlations.')