# -*- coding: utf-8 -*-
# load_and_visualize_behaviors.py
"""
Author:   Raghuveer Parthasarathy
Created on Mon Jul 10 18:09:34 2023
Last modified on Sept. 1, 2024

Description
-----------

Functions for loading data from a pickle file, and visualizing (diagnostics)

Contains the functions 
    - visualize_fish for plotting fish body positions and trajectories
    - loadAllFromPickle to load datasets from a pickle file
    - a program (main function) to load datasets from the pickle file,
      select frames to plot, etc.
"""

import numpy as np
import matplotlib  # should replace this by just matplotlib.colormaps
import matplotlib.pyplot as plt
import os
import pickle
import tkinter as tk
import tkinter.filedialog

from toolkit import link_weighted, repair_disjoint_heads, \
    repair_double_length_fish, repair_head_positions

def loadAllFromPickle(pickleFileName = None, 
                      basePath = None):
    """

    Load contents from Pickle file
    Assumes pickle file contains datasets, CSVcolumns, expt_config
    If pickleFileName is a full path name, load it
    If pickleFileName is a fileName or partial path and basePath exists, join and load
    If pickleFileName is empty, provide a dialog box
    
    Parameters
    ----------
    pickleFileName : pickle file name; can include path to append to basePath
    basePath = None : main path for behavior analysis

    Returns
    -------
    datasets : all datasets in the Pickle file
    CSVcolumns : see behaviors_main()
    expt_config : contains fps, arena_radius_mm, etc.
    params : analysis parameters; see behaviors_main()
    """

    badFile = True # for verifying
    while badFile:
        if pickleFileName == None or pickleFileName == '':
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            titleString = 'Select pickle file'
            pickleFileName = tk.filedialog.askopenfilename(title=titleString,
                                                          filetypes=[("pickle files", "*.pickle")])
        else:
            if not os.path.isabs(pickleFileName):
                # Get parent folder information
                if basePath == None or basePath == '':
                    # Get path from dialog
                    root = tk.Tk()
                    root.withdraw()  # Hide the root window
                    basePath = tk.filedialog.askdirectory(title=f"Select folder containing specified pickle file, {pickleFileName}")
                pickleFileName = os.path.join(basePath, pickleFileName)
        if os.path.isfile(pickleFileName):
            badFile = False
        else:
            print("\n\nInvalid pickle file path or name.")
            print("Please try again; will force dialog box.")
            pickleFileName = None

    with open(pickleFileName, 'rb') as handle:
        b = pickle.load(handle)

    # Assign variables
    datasets = b[0]
    CSVcolumns = b[1]
    expt_config = b[2]
    params = b[3]
    # fps = b[2]    
    # arena_radius_mm = b[3]
    
    return datasets, CSVcolumns, expt_config, params


def visualize_fish(dataset, CSVcolumns, startFrame, endFrame):
    """
    Plot fish body positions (position 1 == head) over some range of frames
       body{x, y} are Nframes x 10 x Nfish=2 arrays of x and y positions.
    Calls plot_one_fish() to plot. Note that plot_one_fish flips "y"
       to match movie orientation
    Color by frame, with Fish 0 markers in the "cool" colormap (cyan to magenta)
       and Fish 1 markers in the "summer_r" (reversed summer, yellow to green) 
       colormap
    Also plot Arena Center (black circle)
         
    Marker for head = circle (Fish 0) and Diamond (Fish 1); 
       x's for bad frames

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
    
    cmap1 = matplotlib.colormaps.get_cmap('cool')
    cmap2 = matplotlib.colormaps.get_cmap('summer_r')
    fig = plt.figure()
    ax = plt.axes()
    for j in range(startFrame, endFrame+1):
        isBadFrame = np.isin(j,  dataset["bad_bodyTrack_frames"]["raw_frames"])
        relativej = (j-startFrame)/(endFrame-startFrame+1)
        plot_one_fish(fig, body_x[frameArray==j,:,0], body_y[frameArray==j,:,0], 
                      frameArray, cmap1, relativej, isBadFrame, marker='o')
        plot_one_fish(fig, body_x[frameArray==j,:,1], body_y[frameArray==j,:,1], 
                      frameArray, cmap2, relativej, isBadFrame, marker='D')
    # plot arena center
    plt.scatter(dataset['arena_center'][0], -1.0*dataset['arena_center'][1],
                marker='o', facecolor='gray', color='black', s = 140)
    ax.set_aspect('equal', adjustable='box')
    plt.draw()
    plt.title(f'{dataset["dataset_name"]}: frames {startFrame} to {endFrame}')


def plot_one_fish(fig, body_x, body_y, frameArray, cmap, 
                  relativej, isBadFrame, marker='o'):
    """
    Plots head (marker) and body (line) of one fish in a single frame,
       with color set by "relativej" = [0,1] position in colormap. 
    Flip "y" to match movies (in datasets, y is row no.)
    Also a Black X if isBadFrame == True
    Inputs
        body_x, _y = array of # body positions x 2==Nfish body positions in 
                 the frame to plot
    #    
    """ 
    plt.figure(fig.number)
    # head of fish
    plt.plot(body_x.flatten()[0], -1.0*body_y.flatten()[0], color=cmap(relativej), 
             markersize=12, marker=marker)
    # 'x' if this frame is in "bad frames"
    if isBadFrame:
        plt.plot(body_x.flatten()[0], -1.0*body_y.flatten()[0], color='black', marker='x',
                 markersize=12)
    # Body of fish
    plt.plot(body_x.flatten(), -1.0*body_y.flatten(), color=cmap(relativej), linestyle='solid')


def flag_possible_IDswitch(dataset, CSVcolumns):
    """
    Assess possible flags for switched fish IDs
    Consider angle, head-head distance, etc.
    
    Unfinished diagnostic function to determine candidate frames for 
    flipped Fish IDs.
    
    Raghu: 4 Dec. 2023
    
    Parameters
    ----------
    dataset: dataset dictionary of all behavior information for a given expt.
        Note that dataset["all_data"] contains all the position information
        Rows = frame numbers
        Columns = x, y, angle data -- see CSVcolumns
        Dim 3 = fish (2 fish)
    CSVcolumns: information on what the columns of dataset["all_data"] are

    Returns
    -------
    wrongID_head : frames with possible wrong IDs, from considering head positions
    wrongID_body : : frames with possible wrong IDs, from considering all body positions

    """
    # All heading angles
    cos_diff_angle = np.cos(np.diff(dataset["heading_angle"], n=1, axis=0))
    #plt.figure()
    #plt.hist(cos_diff_angle[:,0], bins=50)
    #plt.hist(cos_diff_angle[:,1], bins=50)
    cos_diff_thresh = 0.0
    large_angle_change_flag = np.argwhere(cos_diff_angle < cos_diff_thresh)
    print('large angle change shape:', large_angle_change_flag.shape)
    
    # all lengths
    fish_lengths = dataset["fish_length_array_mm"]
    diff_fish_length = np.diff(fish_lengths, n=1, axis=0)
    diff_fish_length_fraction = np.abs(diff_fish_length) / fish_lengths[:-1,:]
    diff_fish_length_fraction_thresh = 0.2
    diff_fish_length_flag = np.argwhere(diff_fish_length_fraction > \
                                        diff_fish_length_fraction_thresh)
    
    # All positions
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
    
    #print('\n\n\n*** DIAGNOSTIC! ***\n')
    #body_x = body_x[0:500,:,:]
    #body_y = body_y[0:500,:,:]

    # Difference across adjacent frames for positions for the same fish IDs
    d_all_same = np.sqrt(np.diff(body_x, n=1, axis=0)**2 + np.diff(body_y, n=1, axis=0)**2)
    # Difference across adjacent frames for positions for alternating fish IDs
    # There's probably a clever way to do this, but my attempts fail, so I'll loop:
    d_all_01 = np.zeros((body_x.shape[0]-1, body_x.shape[1]))    
    d_all_10 = np.zeros((body_x.shape[0]-1, body_x.shape[1]))    
    for j in range(body_x.shape[0]-1):
        d_all_01[j,:] = np.sqrt((body_x[j+1, :, (j+1)%2] - body_x[j, :, j%2])**2 + 
                                  (body_y[j+1, :, (j+1)%2] - body_y[j, :, j%2])**2)
        d_all_10[j,:] = np.sqrt((body_x[j+1, :, j%2] - body_x[j, :, (j+1)%2])**2 + 
                                  (body_y[j+1, :, j%2] - body_y[j, :, (j+1)%2])**2)
    
    # Does switching ID reduce total difference?
    switch_difference = (d_all_01 + d_all_10) - np.sum(d_all_same, axis=2)
    # Considering just head positions:
    wrongID_head = np.argwhere(switch_difference[:,0] < 0).flatten()
    print('Head only')
    for j in range(wrongID_head.shape[0]):
        print(f'{wrongID_head[j]}: same {d_all_same[wrongID_head[j],0,0]:.2f}, {d_all_same[wrongID_head[j],0,1]:.2f}, ', 
              f'switch {d_all_01[wrongID_head[j],0]:.2f}, {d_all_10[wrongID_head[j],0]:.2f}')
    # Considering total body positions:
    wrongID_body = np.argwhere(np.sum(switch_difference, axis=1) < 0).flatten()
    print('Total body')
    for j in range(wrongID_body.shape[0]):
        print(f'{wrongID_body[j]}: same {np.sum(d_all_same[wrongID_body[j],:,0]):.2f},  ',
              f'{np.sum(d_all_same[wrongID_body[j],:,1]):.2f}, ', 
              f'switch {np.sum(d_all_01[wrongID_body[j],:]):.2f}, ',
              f'{np.sum(d_all_10[wrongID_body[j],:]):.2f}, ', 
              f'Difference {np.sum(switch_difference, axis=1)[wrongID_body[j]]:.2f}')
    
    
    #plt.figure()
    #plt.hist(diff_fish_length_fraction[:,0], bins=100)
    #plt.hist(diff_fish_length_fraction[:,1], bins=100)
    #plt.yscale('log')

    return wrongID_head, wrongID_body

def how_many_both_approaching_frames(dataset):
    """
    Note frames in which both Fish are approaching each other

    Parameters
    ----------
    dataset: dataset dictionary of all behavior information for a given expt.
        Note that dataset["all_data"] contains all the position information
        Rows = frame numbers
        Columns = x, y, angle data -- see CSVcolumns
        Dim 3 = fish (2 fish)

    Returns
    -------
    
    Number of frames in which both fish are classified as approaching each other
    
    None.

    """
    # Identify frames in which both fish are approaching
    approach_Fish0_rawFrames = dataset["approaching_Fish0"]["raw_frames"]
    approach_Fish1_rawFrames = dataset["approaching_Fish1"]["raw_frames"]
    approach_both_rawFrames = np.intersect1d(approach_Fish0_rawFrames, 
                                             approach_Fish1_rawFrames)
    print('\n')
    print(dataset["dataset_name"], 
          ': Frames in which both fish are approaching (raw Frames):')
    print('Number of frames: ', len(approach_both_rawFrames), 'out of ',
          len(approach_Fish0_rawFrames), ' and ', len(approach_Fish1_rawFrames))
    print(approach_both_rawFrames)

    return len(approach_both_rawFrames)


if __name__ == '__main__':
    
    basePath = r'C:\Users\Raghu\Documents\Experiments and Projects\Zebrafish behavior\CSV files and outputs'
    
    picklePath  = basePath + r'\2 week old - pairs TestSubset\Analysis'
    pickleFileName = os.path.join(picklePath, r'test.pickle')
    
    datasets, CSVcolumns, expt_config, params = \
        loadAllFromPickle(pickleFileName = pickleFileName) # or None
    
    print('\nAll dataset names:')
    for j in range(len(datasets)):
        print('   ' , datasets[j]["dataset_name"])
    
    whichDataset = '3b_k7'
    chosenSet = None
    for j in range(len(datasets)):
        if datasets[j]["dataset_name"]==whichDataset:
            chosenSet = datasets[j]
    
    showSpeedGraphs = False
    if showSpeedGraphs:
        plt.figure()
        plt.plot(chosenSet['speed_array_mm_s'][:,0], label='Fish 0')
        plt.plot(chosenSet['speed_array_mm_s'][:,1], label='Fish 1')
        plt.xlabel('Frame number')
        plt.ylabel('Speed (mm/s)')
        plt.legend()
        
        plt.figure()
        plt.hist(chosenSet['speed_array_mm_s'][:,0], 1000)
        plt.hist(chosenSet['speed_array_mm_s'][:,1], 1000)
        plt.xlabel('Speed (mm/s)')
    
        plt.figure()
        plt.plot(chosenSet['speed_array_mm_s'][:,0], label='Fish 0 speed')
        plt.plot((180.0/np.pi)*chosenSet['tail_angle_rad'][:,0], label='Fish 0 tail angle')
        plt.xlabel('Frame number')
        plt.ylabel('Speed (mm/s) or angle (deg.)')
        plt.legend()

        plt.figure()
        plt.hist(chosenSet['tail_angle_rad'][:,0], 1000)
        plt.hist(chosenSet['tail_angle_rad'][:,1], 1000)
        plt.xlabel('Tail angle (rad)')
        
        plt.figure()
        plt.hist2d(np.log10(chosenSet['speed_array_mm_s'][:,0]), 
                   np.log10((180.0/np.pi)*chosenSet['tail_angle_rad'][:,0]), 
                   bins=25, range=[[0, 2], [0, 2]])
        #, range=[[0, 50], [0, 180]]
        plt.xlabel('log10 [Speed (mm/s)]', fontsize=16)
        plt.ylabel('log10 [Tail angle (deg)]', fontsize=16)
        plt.title(whichDataset, fontsize=20)

        plt.figure()
        plt.hist2d(chosenSet['speed_array_mm_s'][:,0], 
                   (180.0/np.pi)*chosenSet['tail_angle_rad'][:,0], 
                   bins=25, range=[[0, 100], [0, 180]],
                   vmin=0, vmax=100)
        #, range=[[0, 50], [0, 180]]
        plt.xlabel('Speed (mm/s)', fontsize=16)
        plt.ylabel('Tail angle (deg)', fontsize=16)
        plt.title(whichDataset, fontsize=20)


    startFrame = 180
    endFrame = 191
    visualize_fish(chosenSet, CSVcolumns, 
                   startFrame=startFrame, endFrame=endFrame) # 7430, 7490
    print(f'Values at frame {startFrame}')
    print('Radial position, each fish:', chosenSet["radial_position_mm"][startFrame,:])
    print('Polar angle, each fish:', chosenSet["polar_angle_rad"][startFrame,:])
    print('Heading angle, each fish:', chosenSet["all_data"][startFrame,5,:])
    print('Radial alignment, each fish:', chosenSet["radial_alignment_rad"][startFrame,:])
    
    # (wrongID_head, wrongID_body) = flag_possible_IDswitch(chosenSet, CSVcolumns)

    # frameRange = 3
    # startFrame = wrongID_body[0] - frameRange
    # endFrame = wrongID_body[0] + frameRange
    # visualize_fish(chosenSet, CSVcolumns, 
    #                startFrame=startFrame, endFrame=endFrame) 

    print('Repair Head Positions')
    print('Angle 2107 1:', chosenSet["all_data"][2107,5,1])
    repair_heads = True
    if repair_heads:
        j = 345
        print('Before ...')
        print(chosenSet["all_data"][j,CSVcolumns["body_column_x_start"],0])
        x0 = chosenSet["all_data"][:,CSVcolumns["body_column_x_start"],0].copy()
        chosenSet = repair_head_positions(chosenSet, CSVcolumns, tol=0.001)
        print('After ...')
        print(chosenSet["all_data"][j,CSVcolumns["body_column_x_start"],0])
        x = chosenSet["all_data"][:,CSVcolumns["body_column_x_start"],0]
        angle = chosenSet["heading_angle"][:,0]
        print(f'Mean x[0] difference: {np.mean(x-x0):.2f} px')
        print(f'Std. dev. of x[0] difference: {np.std(x-x0):.2f} px')
        plt.figure()
        plt.scatter(x0, x-x0)
        plt.xlabel('Original x[0]')
        plt.ylabel('Repaired x[0] - Original x[0]')



    print('Here 0')
    print('Angle 2107 1:', chosenSet["all_data"][2107,5,1])
    # Fix disjoint heads
    fix_disjoint_heads = False
    if fix_disjoint_heads:
        chosenSet = repair_disjoint_heads(chosenSet, CSVcolumns, 
                              Dtol=3.0, tol=0.001)
    # body_column_x_start=6
    # body_column_y_start=16
    # body_Ncolumns=10
    # print('Here')
    # print(chosenSet["all_data"][2107,body_column_x_start:(body_column_x_start+body_Ncolumns),1])
    # print(chosenSet["all_data"][2107,5,1])

    # Fix double length
    body_column_x_start=6
    body_column_y_start=16
    body_Ncolumns=10
    idx_to_plot = 2106
    fix_double_length = False
    if fix_double_length:
        print('Here before double length fix')
        print(chosenSet["all_data"][idx_to_plot,body_column_x_start:(body_column_x_start+body_Ncolumns),0])
        print(chosenSet["all_data"][idx_to_plot,body_column_x_start:(body_column_x_start+body_Ncolumns),1])
        plt.figure()
        plt.plot(chosenSet["all_data"][idx_to_plot,body_column_x_start:(body_column_x_start+body_Ncolumns),0],
                   -chosenSet["all_data"][idx_to_plot,body_column_y_start:(body_column_y_start+body_Ncolumns),0], 
                   color='olivedrab', marker='x')
        plt.plot(0.25+chosenSet["all_data"][idx_to_plot,body_column_x_start:(body_column_x_start+body_Ncolumns),1],
                   0.25-chosenSet["all_data"][idx_to_plot,body_column_y_start:(body_column_y_start+body_Ncolumns),1], 
                   color='magenta', marker='x')
        lengthFactor = [1.5, 2.5]
        chosenSet = repair_double_length_fish(chosenSet, CSVcolumns, 
                              lengthFactor = lengthFactor, tol=0.001)
        print('Here after double length fix')
        print(chosenSet["all_data"][idx_to_plot,body_column_x_start:(body_column_x_start+body_Ncolumns),0])
        print(chosenSet["all_data"][idx_to_plot,body_column_x_start:(body_column_x_start+body_Ncolumns),1])
        print(chosenSet["all_data"][idx_to_plot,5,0])
        print(chosenSet["all_data"][idx_to_plot,5,1])
        plt.plot(chosenSet["all_data"][idx_to_plot,body_column_x_start:(body_column_x_start+body_Ncolumns),0],
                   -chosenSet["all_data"][idx_to_plot,body_column_y_start:(body_column_y_start+body_Ncolumns),0], 
                   color='limegreen', marker='o')
        plt.plot(chosenSet["all_data"][idx_to_plot,body_column_x_start:(body_column_x_start+body_Ncolumns),1],
                   -chosenSet["all_data"][idx_to_plot,body_column_y_start:(body_column_y_start+body_Ncolumns),1], 
                   color='hotpink', marker='o')


    # Fix IDs
    fixIDs = False
    if fixIDs:        
        IDs, newIDs = link_weighted(chosenSet["all_data"], CSVcolumns)
    
    # Find frames in which both fish are approaching each other
    findApproaching = False
    if findApproaching==True:
        n_bothApproachFrames = np.zeros(len(datasets))
        for j in range(len(datasets)):
            n_bothApproachFrames[j] = how_many_both_approaching_frames(datasets[j])
        print(f'\n\nAverage n_bothApproachFrames: {np.mean(n_bothApproachFrames):.1f}')
        
    body_column_x_start=6
    body_column_y_start=16
    body_Ncolumns=10
    dx = np.abs(np.diff(chosenSet["all_data"][:,body_column_x_start:(body_column_x_start+body_Ncolumns),1], axis=1))
    d0 = np.median(dx, axis=0)[0]
    d1 = np.median(dx, axis=0)[1]
    print('dx: ', d0, d1)

    vizFish = False
    if vizFish:
        startFrame = 2105
        endFrame = 2112
        visualize_fish(chosenSet, CSVcolumns, 
                       startFrame=startFrame, endFrame=endFrame) 
        print(chosenSet["all_data"][idx_to_plot,body_column_x_start:(body_column_x_start+body_Ncolumns),0])
        print(chosenSet["all_data"][idx_to_plot,body_column_y_start:(body_column_y_start+body_Ncolumns),0])
        print(chosenSet["all_data"][idx_to_plot,body_column_x_start:(body_column_x_start+body_Ncolumns),1])
        print(chosenSet["all_data"][idx_to_plot,body_column_y_start:(body_column_y_start+body_Ncolumns),1])
        