# -*- coding: utf-8 -*-
# compare_all_2wkData.py

"""
Author:   Raghuveer Parthasarathy
Created on Wed Oct 15 08:59:05 2025
Last modified Nov. 5, 2025 -- Raghuveer Parthasarathy

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
from IO_toolkit import load_and_assign_from_pickle
from behavior_identification import make_pair_fish_plots
from behavior_identification_single import make_single_fish_plots
from toolkit import get_fps, make_2D_histogram
from behavior_correlations import plot_behaviorCorrelation,  \
    calc_corr_asymm, plot_corr_asymm, calcDeltaFramesEvents, bin_deltaFrames, calc_pAB, calcBehavCorrAllSets
import matplotlib.pyplot as plt

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
pickleFileName2 = os.path.join(parentPath, CSVPath, cond_str, AnalysisPath, 
                               f'{expBaseStr}_{cond_str}_datasets.pickle')

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
"""
closeFigures = True
if closeFigures:
    print('Single fish plots: Closing Figure Windows.')

for exptName in all_expts.keys():
    make_single_fish_plots(all_expts[exptName]['datasets'], 
                       exptName = exptName,
                       color = all_expts[exptName]['plot_color'], 
                       outputFileNameBase = f'{exptName} single_properties',
                       outputFileNameExt = 'png',
                       closeFigures = closeFigures)
"""

#%% Make all pair plots (a lot!)
# Only for Nfish ==2

closeFigures = True
if closeFigures:
    print('Pair plots: Closing Figure Windows.')
    
for exptName in all_expts.keys():
    if all_expts[exptName]['Nfish'] == 2:
        saved_pair_outputs = make_pair_fish_plots(
                             all_expts[exptName]['datasets'], 
                             exptName = exptName,
                             plot_each_dataset = True, 
                             color = all_expts[exptName]['plot_color'], 
                             outputFileNameBase = f'{exptName} pair_properties', 
                             outputFileNameExt = 'png',
                             closeFigures = closeFigures)
        #all_expts[exptName]["bend_2Dhist_mean"] = saved_pair_outputs[0]
        #all_expts[exptName]["bend_2Dhist_sem"] = saved_pair_outputs[1]
        #all_expts[exptName]["bend_2Dhist_X"] = saved_pair_outputs[2]
        #all_expts[exptName]["bend_2Dhist_Y"] = saved_pair_outputs[3]
    
"""
exptName = 'TwoWk_Light'
make_pair_fish_plots(all_expts[exptName]['datasets'], 
                     exptName = exptName,
                     plot_each_dataset = True, 
                     color = all_expts[exptName]['plot_color'], 
                     outputFileNameBase = f'{exptName} pair_properties', 
                     outputFileNameExt = 'png')
"""


#%% Differences of bending angle

calcDifferences = False

if calcDifferences:
    # Differences
    X = all_expts['TwoWk_Light']["bend_2Dhist_X"]
    Y = all_expts['TwoWk_Light']["bend_2Dhist_Y"]
    dLight = all_expts['TwoWk_Light']["bend_2Dhist_mean"] - \
        all_expts['TwoWk_Light_TIMESHIFT0']["bend_2Dhist_mean"]
    fig, ax = plt.subplots(figsize=(8, 6))
    cbar = fig.colorbar(ax.pcolormesh(X, Y, dLight, shading='nearest', 
                                      cmap = 'RdYlBu', vmin=-0.05, vmax=0.05), ax=ax)
    plt.title('Light - Light_TS0')
    plt.savefig('Delta_bending_Light - Light_TS0.png', bbox_inches='tight')
    
    dLightDark = all_expts['TwoWk_Light']["bend_2Dhist_mean"] - \
        all_expts['TwoWk_Dark']["bend_2Dhist_mean"]
    fig, ax = plt.subplots(figsize=(8, 6))
    cbar = fig.colorbar(ax.pcolormesh(X, Y, dLightDark, shading='nearest', 
                                      cmap = 'RdYlBu', vmin=-0.05, vmax=0.05), ax=ax)
    plt.title('Light - Dark')
    plt.savefig('Delta_bending_Light - Dark.png', bbox_inches='tight')
    



#%% Correlations of pair behaviors
# Note that this is slow!

calculateCorrelations = False

if calculateCorrelations:
    
    behavior_key_list = ['perp_noneSee', 'perp_oneSees', 'perp_bothSee', 
                         'contact_any', 'tail_rubbing', 'maintain_proximity', 
                         'approaching_Fish0', 'approaching_Fish1', 
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
    behavior_key_list_subset2 = ['perp_noneSee', 'perp_oneSees', 'perp_bothSee', 'contact_any', 'tail_rubbing', 'maintain_proximity', 'Cbend_Fish_lowRelOrient0', 'Cbend_Fish_lowRelOrient1']
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