# -*- coding: utf-8 -*-
# load_and_visualize_behaviors.py
"""
Author:   Raghuveer Parthasarathy
Created on Mon Jul 10 18:09:34 2023
Last modified on July 21, 2023

Description
-----------

Contains the function visualize_fish for plotting fish body positions 
   and trajectories, and a program to load datasets, select frames to plot, 
   etc.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

from misc_behaviors import coOrientation
from behavior_identification import get_bent_frames



def visualize_fish(dataset, CSVcolumns, startFrame, endFrame):
    """
    Plot fish body positions (position 1 == head) over some range of frames
    body{x, y} are Nframes x 10 x Nfish=2 arrays of x and y positions.
    Color by frame, with one fish markers in "cool" colormap (cyan to magenta)
    and the other "summer_r" (reversed summer, yellow to green).

    Parameters
    ----------
    dataset: dataset dictionary of all behavior information for a given expt.
        Note that dataset["all_data"] contains all the position information
        Rows = frame numbers
        Columns = x, y, angle data -- see CSVcolumns
        Dim 3 = fish (2 fish)
    CSVcolumns: information on what the columns of dataset["all_data"] are
    startFrame : starting frame number to plot
    endFrame : ending frame number to plot

    Returns
    -------
    None.

    """
    
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
    frameArray = dataset["frameArray"]
    
    cmap1 = matplotlib.cm.get_cmap('cool')
    cmap2 = matplotlib.cm.get_cmap('summer_r')
    fig = plt.figure()
    print(fig)
    ax = plt.axes()
    for j in range(startFrame, endFrame+1):
        isBadFrame = np.isin(j,  dataset["bad_bodyTrack_frames"]["raw_frames"])
        relativej = (j-startFrame)/(endFrame-startFrame+1)
        plot_one_fish(fig, body_x[frameArray==j,:,0], body_y[frameArray==j,:,0], 
                      frameArray, cmap1, relativej, isBadFrame)
        plot_one_fish(fig, body_x[frameArray==j,:,1], body_y[frameArray==j,:,1], 
                      frameArray, cmap2, relativej, isBadFrame)
    ax.set_aspect('equal', adjustable='box')
    plt.draw()
    plt.title(f'{dataset["dataset_name"]}: frames {startFrame} to {endFrame}')


def plot_one_fish(fig, body_x, body_y, frameArray, cmap, relativej, isBadFrame):
    """
    Plots head (marker) and body (line) of one fish in a single frame,
    with color set by "relativej" = [0,1] position in colormap. 
    Also a Black X if isBadFrame == True
    Inputs
        body_x, _y = array of # body positions x 2==Nfish body positions in 
                 the frame to plot
    #    
    """ 
    plt.figure(fig.number)
    # head of fish
    plt.plot(body_x.flatten()[0], body_y.flatten()[0], color=cmap(relativej), marker='o',
             markersize=12)
    # 'x' if this frame is in "bad frames"
    if isBadFrame:
        plt.plot(body_x.flatten()[0], body_y.flatten()[0], color='black', marker='x',
                 markersize=12)
    # Body of fish
    plt.plot(body_x.flatten(), body_y.flatten(), color=cmap(relativej), linestyle='solid')

    
print(' ')
#pickleFileName = input('Pickle file name; Will append .pickle: ')

#pickleFileName = pickleFileName + '.pickle'

# pickleFileName = r'C:\Users\Raghu\Documents\Experiments and Projects\Misc\Zebrafish behavior\CSV files\temp\temp.pickle'

pickleFileName  = r'C:\Users\Raghu\Documents\Experiments and Projects\Misc\Zebrafish behavior\CSV files\2 week old - pairs\all_2wk_light.pickle'
 
# pickleFileName  = r'C:\Users\Raghu\Documents\Experiments and Projects\Misc\Zebrafish behavior\CSV files\2 week old - pairs in the dark\all_2wk_dark.pickle'


with open(pickleFileName, 'rb') as handle:
    b = pickle.load(handle)

# Assign variables
datasets = b[0]
CSVcolumns = b[1]
fps = b[2]    
arena_radius_mm = b[3]
params = b[4]

print('All dataset names:')
for j in range(len(datasets)):
    print('   ' , datasets[j]["dataset_name"])
    
whichDataset = '3b_nk5'
chosenSet = None
for j in range(len(datasets)):
    if datasets[j]["dataset_name"]==whichDataset:
        chosenSet = datasets[j]

visualize_fish(chosenSet, CSVcolumns, 7430, 7490) # 7430, 7490

bendingThreshold = 2/np.pi
get_bent_frames(chosenSet, CSVcolumns, bendingThreshold)
    
coorientation_window_size = 50
coorientation_threshold = 0.9
makeDiagnosticPlots = True
coOrientation(chosenSet, CSVcolumns, coorientation_threshold, 
              params["circle_distance_threshold"], coorientation_window_size, 
              makeDiagnosticPlots)
