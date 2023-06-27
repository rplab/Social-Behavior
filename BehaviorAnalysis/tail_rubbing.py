# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='2.0': Major modifications by Raghuveer Parthasarathy, May-June 2023
# Major changes: May 24, 2023, Raghu Parthasarathy
# Last modified May 28, 2023, Raghu Parthasarathy
# ---------------------------------------------------------------------------
from toolkit import *
# ------------------------------------------------------------------------------

def get_tail_rubbing_wf(body_x, body_y, head_separation,
                        fish_angle_data, window_size, tail_dist, 
                        tail_anti_high, head_dist_thresh): 
    """
    Returns an array of tail-rubbing window frames.

    Args:
        body_x (array): a 3D array of x positions along the 10 body markers, 
                        at each frame. Dimensions [frames, body markers, fish]
        body_y (array): a 3D array of y positions along the 10 body markers, 
                        at each frame. Dimensions [frames, body markers, fish]
        head_separation (array): a 2D array of inter-fish head separations,
                        previously calculated. 
        fish_angle_data (array): a 2D array of angles at each window frame
                                  (dim 0) for each fish (dim 1).

        window_size (int): window size for which circling is averaged over.
        tail_dist (int): tail distance threshold for the two fish. .
        tail_anti_high (float): antiparallel orientation upper bound. 
        head_dist_thresh (int): head distance threshold for the two fish. 

    Returns:
        tail_rubbing_frames: a 1D array of tail-rubbing window frames
    """
    
    N_postPoints = 4 # number of posterior-most body markers 
                          # to consider. (This should be a function input...)
    Nframes = fish_angle_data.shape[0]  # better than having "end" as an input
    
    # In each frame, see if tails are close
    # Examine all frames in the dataset, not just the window
    close_tails = np.zeros((Nframes,1), dtype=bool) # initialize
    for idx in range(Nframes):  # all frames
        # Get N_postPoints posterior body markers for this frame
        tail1 = np.array([body_x[:, -N_postPoints:,0][idx],
                          body_y[:, -N_postPoints:,0][idx]])
        tail2 = np.array([body_x[:, -N_postPoints:,1][idx],
                          body_y[:, -N_postPoints:,1][idx]])
        d0 = np.subtract.outer(tail1[0,:], tail2[0,:]) # all pairs of subtracted x positions (NposxNpos matrix)
        d1 = np.subtract.outer(tail1[1,:], tail2[1,:]) # all pairs of subtracted y positions (NposxNpos matrix)
        dtail = np.sqrt(d0**2 + d1**2) # Euclidean distance matrix, all points
        smallest_two_d = np.partition(dtail.flatten(), 1)[0:2]
        close_tails[idx] = np.max(smallest_two_d) < tail_dist
        # close_tails[idx] is true if the closest two tail positions in frame [idx] are < tail_dist apart
        
    close_tails = close_tails.flatten()
        
    # cos(angle between headings) for each frame
    # Should be antiparallel, so cos(theta) < threshold 
    #   (ideally cos(theta)==-1)
    cos_theta = np.cos(fish_angle_data[:,0] - fish_angle_data[:,1])
    angle_criterion = (cos_theta < tail_anti_high).flatten()

    # Assess head separation, for each frame
    head_separation_criterion = (head_separation < head_dist_thresh).flatten()
    
    # All criteria (and), in each frame
    all_criteria_frame = np.logical_and(close_tails, angle_criterion, head_separation_criterion)
    all_criteria_window = np.zeros(all_criteria_frame.shape, dtype=bool) # initialize to false
    # Check that criteria are met through the frame window. 
    # Will loop rather than doing some clever Boolean product of offset arrays
    for j in range(all_criteria_frame.shape[0]-window_size+1):
        all_criteria_window[j] =  all_criteria_frame[j:j+window_size].all()

    tail_rubbing_frames = np.array(np.where(all_criteria_window==True))[0,:].flatten() + 1
    # Not sure why the [0,:] is needed, but otherwise returns additional zeros.
    
    return tail_rubbing_frames

