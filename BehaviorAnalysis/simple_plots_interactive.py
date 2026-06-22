# interactive

#%%
import numpy as np

#%%
import matplotlib.pyplot as plt

from behavior_plots import make_pair_fish_plots

from IO_toolkit import load_and_assign_from_pickle, plot_probability_distr
from toolkit import combine_all_values_constrained

plt.ion() 

_ = input('Press Enter. ')

all_position_data, variable_tuple = load_and_assign_from_pickle()

(datasets, CSVcolumns, expt_config, params, N_datasets, Nfish, 
 basePath, dataPath, subGroupName) = variable_tuple

outputFileName = 'Pairs_Light_HHDistance.png'
outputCSVFileName = 'Pairs_Light_distance_head_head.csv'
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

all_position_data, variable_tuple = load_and_assign_from_pickle()
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

# %%

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
# %%

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

# %%
