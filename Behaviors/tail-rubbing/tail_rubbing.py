
# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='2.0'
# Major changes: May 24, 2023, Raghu Parthasarathy
# Last modified May 28, 2023, Raghu Parthasarathy
# ---------------------------------------------------------------------------
from toolkit import *
# ------------------------------------------------------------------------------

def get_tail_rubbing_wf(body1_x, body2_x, body1_y, body2_y, fish1_pos, fish2_pos,
fish1_angle_data, fish2_angle_data, end, window_size, tail_dist, 
tail_anti_high, head_dist_thresh): 
    """
    Returns an array of tail-rubbing window frames.

    Args:
        body1_x (array): a 2D array of x positions along the 10 body markers of fish1, at each frame
        body2_x (array): a 2D array of x positions along the 10 body markers of fish2, at each frame
        body1_y (array): a 2D array of y positions along the 10 body markers of fish1, at each frame
        body2_y (array): a 2D array of y positions along the 10 body markers of fish2, at each frame
        fish1_pos (array): a 2D array of (x, y) positions for fish1. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish2_pos (array): a 2D array of (x, y) positions for fish1. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish1_angle_data (array): a 1D array of angles at each window frame
                                  for fish1.
        fish2_angle_data (array): a 1D array of angles at each window frame
                                  for fish2.

        end (int): end of the array for both fish (typically 15,000 window frames.) 
           NOTE: This is unused and should be deleted
        window_size (int): window size for which circling is averaged over.
        tail_dist (int): tail distance threshold for the two fish. .
        tail_anti_high (float): antiparallel orientation upper bound. 
        head_dist_thresh (int): head distance threshold for the two fish. 

    Returns:
        tail_rubbing_wf_combined: a 2D array of tail-rubbing window frames
                                  (row 1) and event durations (row 2)
                                  "combined" to merge adjacent frames
    """
    
    N_postPoints = 4 # number of posterior-most body markers 
                          # to consider. (This should be a function input...)
    Nframes = len(fish1_angle_data)  # better than having "end" as an input
    
    # In each frame, see if tails are close
    # Examine all frames in the dataset, not just the window
    close_tails = np.zeros((Nframes,1), dtype=bool) # initialize
    showDiagnosticList = False
    for idx in range(Nframes):  # all frames
        # Get N_postPoints posterior body markers for this frame
        tail1 = np.array([body1_x[:, -N_postPoints:][idx],
                          body1_y[:, -N_postPoints:][idx]])
        tail2 = np.array([body2_x[:, -N_postPoints:][idx],
                          body2_y[:, -N_postPoints:][idx]])
        d0 = np.subtract.outer(tail1[0,:], tail2[0,:]) # all pairs of subtracted x positions (NposxNpos matrix)
        d1 = np.subtract.outer(tail1[1,:], tail2[1,:]) # all pairs of subtracted y positions (NposxNpos matrix)
        dtail = np.sqrt(d0**2 + d1**2) # Euclidean distance matrix, all points
        smallest_two_d = np.partition(dtail.flatten(), 1)[0:2]
        close_tails[idx] = np.max(smallest_two_d) < tail_dist
        # close_tails[idx] is true if the closest two tail positions in frame [idx] are < tail_dist apart
        if showDiagnosticList:
            print(smallest_two_d)
            print(close_tails[idx])
            x = input('asd')
        
    close_tails = close_tails.flatten()
        
    # cos(angle between headings) for each frame
    # Should be antiparallel, so cos(theta) < threshold 
    #   (ideally cos(theta)==-1)
    cos_theta = np.cos(fish1_angle_data - fish2_angle_data)
    angle_criterion = (cos_theta < tail_anti_high).flatten()

    # Head separation, for each frame
    dh = fish1_pos - fish2_pos
    head_separation = np.sqrt(np.sum(dh**2, axis=1))
    head_separation_criterion = (head_separation < head_dist_thresh)

    #temp = np.logical_or(close_tails, angle_criterion, head_separation_criterion)
    #print(np.sum(angle_criterion))
    #print(np.sum(head_separation_criterion))
    #print(np.sum(close_tails))
    #print(np.sum(temp))
    #print('Hello 2)')
    
    # All criteria (and), in each frame
    all_criteria_frame = np.logical_and(close_tails, angle_criterion, head_separation_criterion)
    all_criteria_window = np.zeros(all_criteria_frame.shape, dtype=bool) # initialize to true
    # Check that criteria are met through the frame window. 
    # Will loop rather than doing some clever Boolean product of offset arrays
    for j in range(all_criteria_frame.shape[0]-window_size+1):
        all_criteria_window[j] =  all_criteria_frame[j:j+window_size].all()

    tail_rubbing_wf = np.array(np.where(all_criteria_window==True))[0,:].flatten() + 1
    # Not sure why the [0,:] is needed, but otherwise returns additional zeros.
    tail_rubbing_wf_combined = combine_events(np.array(tail_rubbing_wf).flatten())
    
    return tail_rubbing_wf_combined

