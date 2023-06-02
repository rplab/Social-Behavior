# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/7/2022
# version ='1.0'
# last modified: Raghuveer Parthasarathy, May 29, 2023
# ---------------------------------------------------------------------------
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import numpy as np
# ---------------------------------------------------------------------------

def get_CSV_folder_and_filenames():
    """
    Gets the folder path containing CSV files, and a list of all CSV files
    whose names start with “results""
    """
    
    folder_path = input("Enter the folder path for CSV files, or leave empty for cwd: ")
    
    if folder_path=='':
        folder_path = os.getcwd() # Current working directory
        
    # Validate the folder path
    while not os.path.isdir(folder_path):
        print("Invalid folder path. Please try again.")
        folder_path = input("Enter the folder path: ")

    print("Selected folder path: ", folder_path)
    
    # Make a list of all CSV files in the folder
    allCSVfileNames = []
    print("Noting CSV Files with names starting with 'results': ")
    for filename in os.listdir(folder_path):
        if (filename.endswith('.csv') and filename.startswith('results')):
            allCSVfileNames.append(filename)

    return folder_path, allCSVfileNames


def load_data(dir, idx_1, idx_2):
    """
    Returns the correct data array for fish1 and fish2 
    from a specified dataset. The data array may contain
    values for position, angle, body marker information, 
    etc. depending on the range specified by the two
    input indices. 

    Args:
        dir (str)  : the dataset's name.
        idx_1 (int): the beginning index from
                     which data should be loaded.
        idx_2 (int): the ending index from
                     which data should be loaded.

    Returns:
        A tuple containing the following arrays:
            fish1_data: data array for fish1.
            fish2_data: data arary for fish2.
    """
    data = np.genfromtxt(dir, delimiter=',')
    Nrows = data.shape[0] # number of rows
    if np.mod(Nrows,2) == 1:
        print('Error! number of rows is odd. load_data in toolkit.py')
    half_size = int(Nrows/2)
    fish1_data = data[:half_size][:, np.r_[idx_1:idx_2]] 
    fish2_data = data[half_size:][:, np.r_[idx_1:idx_2]]
    return fish1_data, fish2_data


def get_cos_angle(fish1_angles, fish2_angles):
    """
    Returns the angle for two fish to be in a antiparallel 
    orientation averaged over some window size. 

    Args:
        fish1_angles (array): a 1D array of angles for x 
                              window frames for fish1.
        fish2_angles (array): a 1D array of angles for x 
                              window frames for fish2.

    Returns:
        theta_avg (float): the angle for two fish to be in a 
                           antiparallel orientation averaged over 
                           some window size.
    """
    cos_angle_avg = np.mean(np.cos(np.subtract(fish1_angles, fish2_angles)))
    return cos_angle_avg



def get_head_distance(fish1_positions, fish2_positions):
    """
    Returns the mean head distance between two fish averaged 
    over some window size.

    Args:
        fish1_positions (array): a 2D array of (x, y) positions for fish1 for x
                                 window frames. The array has form 
                                 [[x1, y1], [x2, y2], [x3, y3],...].
        fish2_positions (array): a 2D array of (x, y) positions for fish2 for x
                                 window frames. The array has form 
                                 [[x1, y1], [x2, y2], [x3, y3],...].

    Returns:
        avg_dist (float) : head distance between two fish averaged 
                           over some window size.
    """
    head_dist = np.sqrt((fish2_positions[:, 0] - fish1_positions[:, 0])**2 + 
    (fish2_positions[:, 1] - fish1_positions[:, 1])**2)
    avg_dist = np.average(head_dist, axis=0)
    return avg_dist


def get_head_distance_traveled(fish_pos, idx_1, idx_2):
    """
    Returns the relative head distance traveled by a
    single fish averaged over some window size.

    Args:
        fish_pos (array): a 2D array of (x, y) positions for a fish. The
                          array has form [[x1, y1], [x2, y2], [x3, y3],...].
        idx_1 (int)     : starting index.
        idx_2 (int)     : ending index.

    Returns:
        dist (float)    : relative head distance traveled by a single fish 
                          averaged over some window size.
    """
    initial_x = fish_pos[:, 0][idx_1]
    initial_y = fish_pos[:, 1][idx_1]
    final_x = fish_pos[:, 0][idx_2]
    final_y = fish_pos[:, 1][idx_2]
    dist = np.sqrt((final_x - initial_x)**2 + (final_y - initial_y)**2)
    return dist


def get_fish_vectors(fish1_angles, fish2_angles):
    """
    Returns a vector in the form (cos(theta), sin(theta)) for fish1
    and fish2 given angle (theta) averaged over some window size.

    Args:
        fish1_angles (array): a 1D array of angles for x 
                              window frames for fish1.
        fish2_angles (array): a 1D array of angles for x 
                              window frames for fish2.

    Returns:
        A 1D array of vectors of the form [fish1_vector, fish2_vector]. 
    """
    fish1_vector = np.array((np.mean(np.cos(fish1_angles)), 
    np.mean(np.sin(fish1_angles))))
    fish2_vector = np.array((np.mean(np.cos(fish2_angles)), 
    np.mean(np.sin(fish2_angles))))
    return np.array((fish1_vector, fish2_vector))


def check_antiparallel_criterion(fish1_positions, fish2_positions, fish1_angles, 
fish2_angles, lower_threshold, upper_threshold, head_dist_threshold):
    """
    Returns True if two fish are antiparallel to each other;
    False otherwise.

    Args:
        fish1_positions (array): a 2D array of (x, y) positions for fish1 for x
                                 window frames. The array has form 
                                 [[x1, y1], [x2, y2], [x3, y3],...].
        fish2_positions (array): a 2D array of (x, y) positions for fish2 for x
                                 window frames. The array has form 
                                 [[x1, y1], [x2, y2], [x3, y3],...].
        fish1_angles (array): a 1D array of angles for x window frames for fish1.
        fish2_angles (array): a 1D array of angles for x window frames for fish2.
        lower_threshold (float)  : lower bound for antiparallel angle.
        upper_threshold (float)  : upper bound for antiparallel angle.
        head_dist_threshold (int): head distance threshold for the two fish.

    Returns:
        True if the fish are antiparallel; False otherwise.
    """
    cos_angle = get_cos_angle(fish1_angles, fish2_angles)
    head_distance = get_head_distance(fish1_positions, fish2_positions)

    if (lower_threshold <= cos_angle < upper_threshold and 
    head_distance < head_dist_threshold):
        res = True
    else:
        res = False
    return res


def normalize_by_mean(motion):
    """
    Returns an array of values normalized by its mean.
    *Most often used in correlations.py*

    Args:
        motion (array): a 1D array of fish motion of speed,
                        velocity, or angle. 

    Returns:
        normalized (array) : the 1D fish motion array normalized by its speed.
    """
    normalized = (motion - np.mean(motion)) / np.mean(motion)
    return normalized


def combine_events(events):
    """
    Given an array of frame numbers, return an arrays of frame numbers 
    with adjacent frames combined and duration numbers 
    corresponding to the duration of adjacent frames.
    For example, frames…
    	1, 5, 12, 13, 14, 17, 22, 23, 34, 40
    Would be returned as frame numbers
        5, 12, 17, 22, 34, 40
    And durations
        1, 3, 1, 2, 1, 1
    See Raghu's notes May 28, 2023, and example_get_runs_and_durations.py 
        for a description of the method

    Args:
        events (array): an array of frame numbers corresponding to events
                           (e.g. frames of detected circling, 
                            tail-rubbing, 90-degree events).

    Returns:
        combined_events_and_durations (array) : a 2 x N array; Row 1
            are uninque, non-adjacent frame numbers, and Row 2 are the 
            corresponding durations.
            and duration numbers corresponding to the duration of adjacent frames.
    """
    d_events = np.diff(events)
    idx1 = np.array(np.where(d_events==1)).flatten() + 1  # index + 1 of flat regions
    idx_keep = np.setdiff1d(np.arange(len(events)), idx1) # indexes of x to keep
    unique_events = events[idx_keep]
    durations = np.ones_like(unique_events)
    for j in range(len(idx1)-1,-1, -1):
        # find the closest idx_keep under idx1[j]
        close_idx = max(idx_keep[idx1[j] > idx_keep])
        duration_idx = np.array(np.where(idx_keep == close_idx)).flatten()
        durations[duration_idx] += 1
    
    combined_events_and_durations = np.stack((events[idx_keep], durations))
    return combined_events_and_durations



def get_mean_body_size(body1_x, body2_x, body1_y, body2_y, end):
    fish1_lens, fish2_lens = [], []
    fish1_coords = np.stack((body1_x, body1_y), axis=-1)
    fish2_coords = np.stack((body2_x, body2_y), axis=-1)

    for idx, (fish1_arr, fish2_arr) in enumerate(zip(fish1_coords, fish2_coords)):
        fish1_sum, fish2_sum = 0, 0

        # Don't calculate body length for frames with bad tracking
        if np.all(fish1_arr[1:] == 0) or np.all(fish2_arr[1:] == 0):
            continue
        else:
            for idx_ in range(9): # Every fish has 10 body markers
                fish1_sum += np.linalg.norm(fish1_arr[idx_] - fish1_arr[idx_ + 1])
                fish2_sum += np.linalg.norm(fish2_arr[idx_] - fish2_arr[idx_ + 1])
            fish1_lens.append(fish1_sum)
            fish2_lens.append(fish2_sum)
        
    fish1_lens, fish2_lens = np.array(fish1_lens), np.array(fish2_lens)
    
    fish1_mean, fish2_mean = np.mean(fish1_lens), np.mean(fish2_lens)
    fish1_std, fish2_std = np.std(fish1_lens), np.std(fish2_lens)
    return (fish1_mean, fish2_mean, fish1_std, fish2_std)
        

def rand_jitter(arr):
    '''
    Adds a small random number to each element in a given
    array. Source: https://stackoverflow.com/questions/8671808/
    matplotlib-avoiding-overlapping-datapoints-in-a-scatter-dot
    -beeswarm-plot.

    Args:
        arr (array) : the array to be modified.

    Returns         : an array whose values are each shifted by 
                      a small random number. 
    
    '''
    return arr + (np.random.uniform(-1, 1, np.size(arr)) * 0.50)


def jitter(x, y, s=40, c='b', marker='o', cmap=None, norm=None, vmin=None, 
           vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    '''
    Plots points on a scatter plot with a small "jitter"
    to x-coordinates to avoid overlapping points. Source: 
    https://stackoverflow.com/questions/8671808/matplotlib
    -avoiding-overlapping-datapoints-in-a-scatter-dot-
    beeswarm-plot

    Args:
        x (array): x coordinates.
        y (array): y coordinates.
        c        : color of points.
        marker   : point shape.
                ...

        (See matplotlib documentation for 
        docs of remaining optional parameters)

    Returns:
        N/A
    '''
    return plt.scatter(rand_jitter(x), y, 
                      s=s, c=c, marker=marker, cmap=cmap, norm=norm, 
                      vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, 
                      **kwargs)

