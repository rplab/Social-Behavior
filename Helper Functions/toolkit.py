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


def get_antiparallel_angle(fish1_angles, fish2_angles):
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
    theta_avg = np.mean(np.cos(np.subtract(fish1_angles, fish2_angles)))
    return theta_avg


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
    angle = get_antiparallel_angle(fish1_angles, fish2_angles)
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
