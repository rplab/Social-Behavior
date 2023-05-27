# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/7/2022
# version ='1.0'
# ------------------------------------------------------------------------------
import os
import re
from random import uniform
from itertools import tee
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
import numpy as np
# ------------------------------------------------------------------------------
two_week_light_centers = {
    '3b_k2': (982.889, 1123.366), '3b_k3': (989.147, 1105.233),
    '3b_k4': (1027.275, 1116.705), '3b_k5': (961.83, 1116.671),
    '3b_k6': (1003.415, 1098.898), '3b_k7': (994.207, 1123.541),
    '3b_k8': (1018.048, 1123.059), '3b_k9': (991.903, 1132.162),
    '3b_k10': (1012.457, 1119.279),'3b_k11': (959.601, 1135.131),
    '3b_k12': (1018.355, 1096.428),'3b_nk1': (995.128, 1131.831),
    '3b_nk2': (968.158, 1120.104), '3b_nk3': (994.503, 1113.68),
    '3b_nk4': (1002.748, 1113.869), '3b_nk5': (988.359, 1134.509),
    '3b_nk6': (993.954, 1107.492), '3b_nk7': (986.441, 1116.652),
    '3b_nk8': (979.737, 1122.867), '3b_nk9': (1009.418, 1108.563),
    '3b_nk10': (1023.671, 1101.803), '3b_nk11': (986.681, 1108.271),
    '3c_2wpf_k1': (1005.402, 1027.511), '3c_2wpf_k2': (986.591, 1019.49),
    '3c_2wpf_k3': (1013.251, 1022.029), '3c_2wpf_k4': (1000.811, 1027.525),
    '3c_2wpf_k5': (993.104, 1009.799), '3c_2wpf_k6': (998.651, 1007.324),
    '3c_2wpf_k7': (974.895, 1030.236), '3c_2wpf_k8': (1001.056, 1036.838),
    '3c_2wpf_k9': (1013.074, 1039.872), '3c_2wpf_k10': (1027.721, 1025.174),
    '3c_2wpf_k11': (989.75, 977.209), '3c_2wpf_nk1': (998.203, 1034.07),
    '3c_2wpf_nk2': (1030.515, 1042.13), '3c_2wpf_nk3': (1012.823, 1029.737),
    '3c_2wpf_nk4': (981.186, 1024.613), '3c_2wpf_nk5': (1006.868, 1015.557),
    '3c_2wpf_nk6': (1004.324, 1025.355), '3c_2wpf_nk7': (1006.978, 1013.005),
    '3c_2wpf_nk8': (1004.194, 1024.702), '3c_2wpf_nk9': (1012.747, 1021.973),
    '3c_2wpf_nk10': (1042.187, 1037.164) 
}

two_week_dark_centers = {
    '5a_k1' : (469.674, 492.335), '5a_k2' : (462.286, 463.342), 
    '5a_k3' : (466.653, 489.392), '5a_k5' : (479.459, 465.974), 
    '5a_k6' : (489.104, 477.871), '5a_k7' : (468.044, 461.357), 
    '5a_k8' : (470.267, 473.77), '5a_k9' : (471.121, 468.212), 
    '5a_k10' : (484.749, 478.691), '5a_nk1' : (473.103, 472.041), 
    '5a_nk2' : (481.051, 460.597), '5a_nk3' : (477.29, 468.048), 
    '5a_nk4' : (470.988, 468.913), '5a_nk5' : (473.09, 463.224), 
    '5a_nk6' : (478.253, 462.838), '5a_nk7' : (470.131, 490.965), 
    '5a_nk8' : (480.611, 479.219), '5a_nk9' : (473.563, 464.297), 
    '5a_nk10' : (477.911, 500.252), 
}

def load_data(dir, file_path, dataset_type, dist_thresh):
    """
    Loads files from a directory into arrays,
    removes all window frames where at least one 
    of the two fish is close to the edge of the 
    petri dish, and stores the resulting arrays
    into .npz compressed files.

    Args:
        dir (str)          : the folder directory that 
                             contains the files.
        file_path (str)    : the folder directory to which
                             the data arrays are stored. 
        dataset_type (str) : e.g. 2 week light or 6 week. 
        dist_thresh        : the distance cutoff from the 
                             center of the petri dish.

    Returns:
        N/A
    """
    if dataset_type == '2 week light':
        centers = two_week_light_centers
    else:
        centers = two_week_dark_centers

    # Recurse through all folders 
    for (root, dirs, files) in os.walk(dir, topdown=True):
        # Recurse through each file in every folder
        for name in files:
            dataset = os.path.join(root, name)
            dataset_name = re.search('(\d[a-z]+_)?\d[a-z]+_[a-z]+\d', dataset).group()    # Fix Regex; see Raghu's email
            data = np.loadtxt(dataset, delimiter=",")
            fish1_data = data[:15000]
            fish2_data = data[15000:]

            # Calculate distance from center for each fish;
            # If calculated distance greater than threshold, remove
            # window frame from both arrays
            if dataset_name in centers.keys():
                center = list(centers[dataset_name])
                wfs_to_remove = []
                
                for idx in range(15000):   # Each 2D array is originally of size 15000
                    if (np.linalg.norm(fish1_data[idx][3:5] - center) > dist_thresh
                    or (np.linalg.norm(fish2_data[idx][3:5] - center) > dist_thresh)):
                        wfs_to_remove.append(idx)
                
                wfs_to_remove = np.array(wfs_to_remove)
                fish1_data = np.delete(fish1_data, wfs_to_remove, axis=0)
                fish2_data = np.delete(fish2_data, wfs_to_remove, axis=0)
                    
                np.savez_compressed(f"{file_path}\{dataset_name}", fish1_data, fish2_data)

                
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
    Returns the head distance between two fish averaged 
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
    angle = get_cos_angle(fish1_angles, fish2_angles)
    head_distance = get_head_distance(fish1_positions, fish2_positions)

    if (lower_threshold <= angle < upper_threshold and 
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


def combine_events(event_arr):
    """
    Combine adjacent window frames into a single event with 
    an associated duration.

    Args:
        event_arr (array): an array of social behavior window frames
                           (e.g. circling, tail-rubbing, 90-degree events).

    Returns:
        combined (array) : a modified array of social behavior window frames
                           where adjacent window frames are classified as a 
                           single event. 
    
    """
    combined = []

    for k, g in groupby(enumerate(event_arr), lambda x:x[0]-x[1]):
        group = (map(itemgetter(1), g))     # Adjacent frames have a diff of 1
        group = list(map(int, group))

        # Adjacent window frames have format (start, finish, duration)
        # if len(group) > 1:
        #     combined.append((group[0], group[-1], group[-1] - group[0]))
        # else:
        #     combined.append(group[0])
        combined.append(group[0])

    return np.array(combined)


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


