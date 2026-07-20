# interactive

#%%
import numpy as np

#%% Head-Head distance probability distribution

import matplotlib.pyplot as plt

from behavior_plots import make_pair_fish_plots

from IO_toolkit import load_and_assign_from_pickle, plot_probability_distr
from toolkit import combine_all_values_constrained

plt.ion() 

mainPathName = r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs"

# Pairs light
pickleFileName1=r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_positionData.pickle"
pickleFileName2=r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_Analysis\TwoWk_Sept2025_Light_Cond_2_datasets.pickle"

# pairs dark
#pickleFileName1 = os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_Dark_Cond_1_positionData.pickle")
#pickleFileName2 = os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_Dark_Cond_1_Analysis\TwoWk_Sept2025_Dark_Cond_1_datasets.pickle")

all_position_data, variable_tuple = load_and_assign_from_pickle(pickleFileName1 = pickleFileName1, 
                pickleFileName2 = pickleFileName2)

(datasets, CSVcolumns, expt_config, params, N_datasets, Nfish, 
 basePath, dataPath, subGroupName) = variable_tuple

outputFileName = 'Pairs_Light_HHDistance.png'
outputCSVFileName = None
closeFigures = False
color = 'darkorange'
exptName = 'Pairs_Light'

# head-head distance histogram
head_head_mm_all = combine_all_values_constrained(datasets, 
                                                    keyName='head_head_distance_mm', 
                                                    dilate_minus1 = False)
plot_probability_distr(head_head_mm_all, bin_width = 0.5, 
                        bin_range = [0, None], 
                        color = color,
                        yScaleType = 'linear',
                        plot_each_dataset = False,
                        plot_sem_band = True,
                        xlim = (-1.0, 50.0), ylim = (0.0, 0.05),
                        xlabelStr = 'Head-head distance (mm)', 
                        titleStr = f'{exptName}: head-head distance (mm)',
                        outputFileName = outputFileName,
                        closeFigure = closeFigures,
                        outputCSVFileName = outputCSVFileName)

plt.ioff()             # turn blocking back on for the final hold
plt.show()       

# %%
import matplotlib.pyplot as plt
from IO_toolkit import load_and_assign_from_pickle, plot_probability_distr
from toolkit import combine_all_values_constrained


plt.ion() 

#pairs light
pickleFileName1=r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_positionData.pickle"
pickleFileName2=r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_Analysis\TwoWk_Sept2025_Light_Cond_2_datasets.pickle"

all_position_data, variable_tuple = load_and_assign_from_pickle(pickleFileName1 = pickleFileName1, 
                pickleFileName2 = pickleFileName2)
(datasets, CSVcolumns, expt_config, params, N_datasets, Nfish, 
 basePath, dataPath, subGroupName) = variable_tuple

outputFileName = 'Pairs_Light_radialPosition.png'
outputCSVFileName = None
closeFigures = False
color = 'darkorange'
exptName = 'Pairs_Light'


# Radial position histogram
radial_position_mm_all = combine_all_values_constrained(datasets, 
                                                    keyName='radial_position_mm', 
                                                    dilate_minus1 = False)
plot_probability_distr(radial_position_mm_all, bin_width = 0.5, 
                        bin_range = [0, None],
                        color = color,
                        yScaleType = 'linear',
                        plot_each_dataset = False,
                        plot_sem_band = True,
                        normalize_by_inv_bincenter = True,
                        ylim = (-0.025, 0.5),
                        xlabelStr = 'Radial position (mm)', 
                        titleStr = f'{exptName}: Probability Distr.: r',
                        outputFileName = outputFileName,
                        closeFigure=closeFigures,
                        outputCSVFileName=outputCSVFileName)

plt.ioff()             # turn blocking back on for the final hold
plt.show()       

# %% Revise datasets with IBI properties

import os
from IO_toolkit import revise_datasets

# single light
pickleFileName1 = r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_positionData.pickle"
pickleFileName2 = r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_Analysis\TwoWkSingle_Light_Cond_2_datasets.pickle"

# single dark
#pickleFileName1=r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_positionData.pickle"
#pickleFileName2=r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_Analysis\TwoWkSingle_Dark_Cond_1_datasets.pickle"

#pairs light
#pickleFileName1=r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_positionData.pickle"
#pickleFileName2=r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_Analysis\TwoWk_Sept2025_Light_Cond_2_datasets.pickle"

# pairs timeshifted light
mainPathName = r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs"
#pickleFileName1 = os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_TS0_Light_Cond_2_positionData.pickle")
#pickleFileName2 = os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_TS0_Light_Cond_2_Analysis\TwoWk_Sept2025_TS0_Light_Cond_2_.pickle")

# pairs dark
#pickleFileName1 = os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_Dark_Cond_1_positionData.pickle")
#pickleFileName2 = os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_Dark_Cond_1_Analysis\TwoWk_Sept2025_Dark_Cond_1_datasets.pickle")

# pairs timeshifted dark
#pickleFileName1 = os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_TS0_Dark_Cond_1_positionData.pickle")
#pickleFileName2 = os.path.join(mainPathName, r"2 week old - Sept2025 control pairs in dark vs light New Tracking\Dark_Cond_1\TwoWk_Sept2025_TS0_Dark_Cond_1_Analysis\TwoWk_Sept2025_TS0_Dark_Cond_1_dat.pickle")

revise_datasets(keys_to_modify=["IBI_properties"], pickleFileName1 = pickleFileName1, 
                pickleFileName2 = pickleFileName2, 
                writePickleOutput =  True)

print('\nDone')

#%% Plot a few inter-bout interval histograms, including Delta_theta
import matplotlib.pyplot as plt

from behavior_plots import plot_interbout_histogram

from IO_toolkit import load_and_assign_from_pickle, plot_probability_distr
from toolkit import combine_all_values_constrained

plt.ion() 

# Single Light
pickleFileName1 = r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_positionData.pickle"
pickleFileName2 = r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_Analysis\TwoWkSingle_Light_Cond_2_datasets.pickle"
all_position_data, variable_tuple = load_and_assign_from_pickle(pickleFileName1 = pickleFileName1, 
                pickleFileName2 = pickleFileName2)

(datasets, CSVcolumns, expt_config, params, N_datasets, Nfish, 
 basePath, dataPath, subGroupName) = variable_tuple

"""
plot_interbout_histogram(datasets, key='Delta_s_mm', ax=None, bins=None, yscale='log',
                             color='steelblue', titleStr='Delta_s_mm',
                             outputFileName=None, closeFigure=False)

plot_interbout_histogram(datasets, key='turning_angle_IBI', ax=None, bins=None, yscale='linear',
                             color='darkorange', titleStr='Turning Angle (rad)',
                             outputFileName=None, closeFigure=False)
"""

outputFileName = 'Single_Light_Delta_theta.png'

plot_interbout_histogram(datasets, key='Delta_theta', ax=None, bins=None, yscale='linear',
                             color='darkorange', titleStr='Delta_theta (rad)',
                             outputFileName=outputFileName, closeFigure=False)

plt.ioff()             # turn blocking back on for the final hold
plt.show()       


# %% Radial Probability Distribution p(r) (area normalized)

import matplotlib.pyplot as plt
from IO_toolkit import load_and_assign_from_pickle, plot_probability_distr
from toolkit import combine_all_values_constrained


plt.ion() 

# single light
pickleFileName1 = r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_positionData.pickle"
pickleFileName2 = r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Light_Cond_2\TwoWkSingle_Light_Cond_2_Analysis\TwoWkSingle_Light_Cond_2_datasets.pickle"

# single dark
# pickleFileName1=r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_positionData.pickle"
# pickleFileName2=r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control single in dark vs light New Tracking\Dark_Cond_1\TwoWkSingle_Dark_Cond_1_Analysis\TwoWkSingle_Dark_Cond_1_datasets.pickle"


all_position_data, variable_tuple = load_and_assign_from_pickle(pickleFileName1 = pickleFileName1, 
                pickleFileName2 = pickleFileName2)
(datasets, CSVcolumns, expt_config, params, N_datasets, Nfish, 
 basePath, dataPath, subGroupName) = variable_tuple

outputFileName = 'Single_Light_radialPosition.png'
outputCSVFileName = None
closeFigures = False
color = 'darkorange' # 'navy' #
exptName = 'Single_Light'


# Radial position histogram
radial_position_mm_all = combine_all_values_constrained(datasets, 
                                                    keyName='radial_position_mm', 
                                                    dilate_minus1 = False)
plot_probability_distr(radial_position_mm_all, bin_width = 1.0, 
                        bin_range = [0, None],
                        color = color,
                        yScaleType = 'linear',
                        plot_each_dataset = False,
                        plot_sem_band = True,
                        normalize_by_inv_bincenter = True,
                        ylim = (-0.005, 0.25),
                        xlabelStr = 'Radial position (mm)', 
                        titleStr = f'{exptName}: Probability Distr.: r',
                        outputFileName = outputFileName,
                        closeFigure=closeFigures,
                        outputCSVFileName=outputCSVFileName)

plt.ioff()             # turn blocking back on for the final hold
plt.show()       


#%% Turning angle 2D histogram, and asymmetric sum

import matplotlib.pyplot as plt
import numpy as np

from behavior_plots import make_pair_fish_plots, \
    make_interbout_turning_angle_plots

from IO_toolkit import load_and_assign_from_pickle, plot_probability_distr
from toolkit import combine_all_values_constrained

plt.ion() 

# Pairs light
pickleFileName1=r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_positionData.pickle"
pickleFileName2=r"C:\Users\raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs\2 week old - Sept2025 control pairs in dark vs light New Tracking\Light_Cond_2\TwoWk_Sept2025_Light_Cond_2_Analysis\TwoWk_Sept2025_Light_Cond_2_datasets.pickle"

all_position_data, variable_tuple = load_and_assign_from_pickle(pickleFileName1 = pickleFileName1, 
                pickleFileName2 = pickleFileName2)

(datasets, CSVcolumns, expt_config, params, N_datasets, Nfish, 
 basePath, dataPath, subGroupName) = variable_tuple

outputFileNameBase = 'Pairs_Light_TurningAngle.png'
outputCSVFileName = 'Pairs_Light_TurningAngle_AsymmSum.csv'
closeFigures = False
color = 'darkorange'
exptName = 'Pairs_Light'

Nbins = (21, 15) # n_relorient_bins, n_dHH_bins
arena_radius_mm = 25.0
build_kinematic_bins = False
kinematic_r_bin_size_mm = 2.0
kinematic_dHH_bin_size_mm = 2.0


saved_pair_outputs = make_interbout_turning_angle_plots(
    datasets,
    exptName='pair_light',
    angle_type='Delta_theta',   # displacement-direction change (the sim frame)
    distance_type='head_head_distance',
    Nbins=Nbins,
    mask_by_sem_limit_degrees=8.0,
    colorRange=(-20*np.pi/180.0, 20*np.pi/180.0),
    cmap='berlin',
    plot_type_2D='heatmap',
    outputFileNameBase=outputFileNameBase,
    closeFigures=True,
    outputCSVFileName=None)
turn_2Dhist_mean = saved_pair_outputs[0]
turn_2Dhist_std = saved_pair_outputs[2]
rel_orient_bins = saved_pair_outputs[3]
dHH_bins = saved_pair_outputs[4]

dHH = dHH_bins[0,:]
relOrient = rel_orient_bins[:,0]

print('Done.')
print('Shape: ', turn_2Dhist_mean.shape)

# Asymmetric sum
if len(rel_orient_bins) % 2 == 0:
    # Verify odd number of relOrient bins
    raise ValueError('Must have odd number of relative orientation bins, or rewrite code.')
    turn_sum = None
else:
    # subtract second  half from  first half
    nRO = int((len(relOrient)+1)/2.0)
    print(nRO)
    turn_diff = np.zeros((nRO, Nbins[1]), dtype=turn_2Dhist_mean.dtype)
    firstHalf = turn_2Dhist_mean[:nRO, :]
    secondHalf = turn_2Dhist_mean[:nRO-1:-1,:]
    turn_diff[:-1, :] = secondHalf - firstHalf[:-1, :]
    # center bin zero by definition
    turn_diff[-1, :] = np.zeros_like(firstHalf[-1,:])
    turn_sum_relOrient = np.mean(turn_diff[3:8], axis=0)
    turn_sum_relOrient_sem = np.std(turn_diff[3:8], axis=0)/np.sqrt(nRO)

print('here')
print(turn_sum_relOrient)

print(dHH.shape)
print(turn_sum_relOrient.shape)

print('dHH: ')
print(dHH)

plt.figure()
plt.imshow(turn_2Dhist_mean, cmap='berlin',
           vmin=-0.5, vmax=0.5)
plt.title('Mean turning angle')

plt.figure()
plt.imshow(turn_diff, cmap='berlin',
           vmin=-1.0, vmax=1.0)

plt.figure()
plt.errorbar(dHH, turn_sum_relOrient, yerr=turn_sum_relOrient_sem,  fmt='o',
             capsize=7, markersize = 14, color = 'dodgerblue', ecolor = 'navy')
logisticCurve = -0.3 + 0.85/(1 + np.exp((dHH - 30.0)/5.0))
plt.plot(dHH, logisticCurve, color='darkorange')
plt.xlabel('Inter-fish distance (mm)', fontsize=12)
plt.ylabel('Asymm Turning Angle Difference (rad)', fontsize=12)

plt.ioff()             # turn blocking back on for the final hold
plt.show()       

#%% Poisson



# %%

import os
import tifffile as tiff

mainPath = r'C:\Users\raghu\Documents\Experiments and Projects\Light Sheet Microscope\Sheet shifting deconvolution\Sheet shifting decon_test 23Feb2023\slice20'
filePath = os.path.join(mainPath, 'slice20_MMStack_Pos0.ome.tif')

im = tiff.imread(filePath)

with tiff.TiffFile(filePath) as tif:
    print(tif.series[0].shape)
    print(tif.series[0].axes)
    
    # Access custom metadata (OME-XML, ImageJ tags, etc.)
    ome_xml = tif.ome_metadata
    print(ome_xml)

print(im.shape)
# %% Behavior vs distance

# Plot the probability of a an approach (any fish) as a 
# function of inter-fish distance
# Decide what inter-fish distance measure you want to use. (Head-head or closest).

from IO_toolkit import load_and_assign_from_pickle, calculate_property_1Dbinned
from behavior_plots import plot_property_1Dbinned

# Load datasets as usual:
all_position_data, variable_tuple = load_and_assign_from_pickle()
# follow the prompts. then
(datasets, CSVcolumns, expt_config, params, N_datasets, Nfish, 
 basePath, dataPath, subGroupName) = variable_tuple

# Use calculate_property_1Dbinned() and plot_property_1Dbinned(), 
# as shown below. 
# Note the options for outputting a figure (outputFileName) and / or a 
# CSV file (outputCSVFileName):

keyName = 'Jbend_any'
binKeyName = 'closest_distance_mm'
bin_range = (0.0, 50.0)
Nbins = 20
titleStr = f'{keyName} probability v distance'
xlabelStr = 'Closest distance (mm)'
ylabelStr = f'{keyName} probability'
binned_mean, bin_centers, binned_mean_each_dataset, binned_mean_each_fish = \
    calculate_property_1Dbinned(datasets, keyName= keyName, keyIdx='all',
                                key_is_a_behavior = True, binKeyName=binKeyName,
                                bin_range=bin_range, Nbins=Nbins,
                                dilate_minus1= False)
plot_property_1Dbinned(binned_mean, bin_centers, binned_mean_each_dataset,
                           plot_each_dataset=True, plot_sem_band=True,
                           titleStr=titleStr, xlabelStr=xlabelStr, ylabelStr=ylabelStr,
                           color='black', xlim=None, ylim=None,
                           unit_scaling_for_plot=None,
                           outputFileName='J_and_distance.svg', 
                           closeFigure=False,
                           outputCSVFileName='J_and_distance.csv')

# %%

from IO_toolkit import load_and_assign_from_pickle, calculate_property_1Dbinned
from behavior_plots import make_pair_fish_plots

# Load datasets as usual:
all_position_data, variable_tuple = load_and_assign_from_pickle()
# follow the prompts. then
(datasets, CSVcolumns, expt_config, params, N_datasets, Nfish, 
 basePath, dataPath, subGroupName) = variable_tuple

exptName = 'two_wk_lt_test'
make_pair_fish_plots(datasets, 
                        exptName = exptName,
                        distance_type = 'closest_distance',
                        color = 'darkorange', 
                        outputFileNameBase = f'{exptName}_pair_properties', 
                        outputFileNameExt = 'png',
                        closeFigures = False,
                        writeCSVs = False)
# %%
