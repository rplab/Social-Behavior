# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/7/2022
# version ='1.0'
# ---------------------------------------------------------------------------
import numpy as np
# ---------------------------------------------------------------------------

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
    fish1_data = data[:15000][:, np.r_[idx_1:idx_2]] 
    fish2_data = data[15000:][:, np.r_[idx_1:idx_2]]
    return fish1_data, fish2_data


def get_antiparallel_angle(fish1_angle_data, fish2_angle_data, idx_1, idx_2):
    """
    Returns the angle for two fish to be in a antiparallel 
    orientation averaged over some window size. 

    Args:
        fish1_angle_data (array): a 1D array of angles at each 
                                  window frame for fish1.
        fish2_angle_data (array): a 1D array of angles at each 
                                  window frame for fish2.
        idx_1 (int)             : starting index.
        idx_2 (int)             : ending index.

    Returns:
        theta_avg (float): the angle for two fish to be in a 
                           antiparallel orientation averaged over 
                           some window size.
    """
    fish1_data = fish1_angle_data[idx_1:idx_2]
    fish2_data = fish2_angle_data[idx_1:idx_2]
    theta_avg = np.mean(np.cos(np.subtract(fish1_data, fish2_data)))
    return theta_avg


def get_head_distance(fish1_pos, fish2_pos, idx_1, idx_2):
    """
    Returns the head distance between two fish averaged 
    over some window size.

    Args:
        fish1_pos (array): a 2D array of (x, y) positions for fish1. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish2_pos (array): a 2D array of (x, y) positions for fish2. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        idx_1 (int)      : starting index.
        idx_2 (int)      : ending index.

    Returns:
        avg_dist (float) : head distance between two fish averaged 
                           over some window size.
    """
    head_dist_matrix = np.sqrt((fish2_pos[:, 0] - fish1_pos[:, 0])**2 + 
    (fish2_pos[:, 1] - fish1_pos[:, 1])**2)
    dist_temp = head_dist_matrix[idx_1:idx_2]
    avg_dist = np.average(dist_temp, axis=0)
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


def get_fish_vectors(fish1_angle_data, fish2_angle_data, idx_1, idx_2):
    """
    Returns a vector in the form (cos(theta), sin(theta)) for fish1
    and fish2 given angle (theta) averaged over some window size.

    Args: 
        fish1_angle_data (array): a 1D array of angles at each window frame
                                  for fish1.
        fish2_angle_data (array): a 1D array of angles at each window frame
                                  for fish2.
        idx_1 (int)             : starting index.
        idx_2 (int)             : ending index.

    Returns:
        A 1D array of vectors of the form [fish1_vector, fish2_vector]. 
    """
    fish1_data = fish1_angle_data[idx_1:idx_2]
    fish2_data = fish2_angle_data[idx_1:idx_2]
    fish1_vector = np.array((np.mean(np.cos(fish1_data)), 
    np.mean(np.sin(fish1_data))))
    fish2_vector = np.array((np.mean(np.cos(fish2_data)), 
    np.mean(np.sin(fish2_data))))
    return np.array((fish1_vector, fish2_vector))


def check_antiparallel_criterion(fish1_angle_data, fish2_angle_data,
idx_1, idx_2, lower_threshold, upper_threshold, fish1_pos, fish2_pos,
head_dist_threshold):
    """
    Returns True if two fish are antiparallel to each other;
    False otherwise.

    Args:
        fish1_angle_data (array) : a 1D array of angles at each window frame
                                  for fish1.
        fish2_angle_data (array) : a 1D array of angles at each window frame
                                  for fish2.
        idx_1 (int)              : starting index.
        idx_2 (int)              : ending index.
        lower_threshold (float)  : lower bound for antiparallel angle.
        upper_threshold (float)  : upper bound for antiparallel angle.
        fish1_pos (array)        : a 2D array of (x, y) positions for fish1. The
                                  array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish2_pos (array)        : a 2D array of (x, y) positions for fish2. The
                                  array has form [[x1, y1], [x2, y2], [x3, y3],...].
        head_dist_threshold (int): head distance threshold for the two fish.

    Returns:
        True if the fish are antiparallel; False otherwise.
    """
    angle = get_antiparallel_angle(fish1_angle_data, fish2_angle_data, idx_1, idx_2)
    head_distance = get_head_distance(fish1_pos, fish2_pos, idx_1, idx_2)

    if (lower_threshold <= angle < upper_threshold and 
    head_distance < head_dist_threshold):
        res = True
    else:
        res = False
    return res
