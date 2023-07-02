# -*- coding: utf-8 -*-
# behavior_identification.py
"""
Author:   Raghuveer Parthasarathy
Version ='2.0': 
First versions created By  : Estelle Trieu, 5/26/2022
Major modifications by Raghuveer Parthasarathy, May-July 2023
Last modified June 2, 2023 -- Raghu Parthasarathy

Description
-----------

Module containing all zebrafish pair behavior identification functions:
    - Circling 
    - Contact
    - 90-degree orientation
    - Tail rubbing

Requires prior import of TaubinSVD() from circle_fit_taubin (for circling)
Requires prior import of numpy as np

"""

import numpy as np
from circle_fit_taubin import TaubinSVD


def get_circling_frames(fish_pos, head_separation, fish_angle_data,
                    Nframes, window_size, circle_fit_threshold, 
                    cos_theta_AP_threshold, 
                    cos_theta_tangent_threshold, motion_threshold, 
                    head_dist_thresh):
    """
    Returns an array of window frames for circling behavior. Each window
    frame represents the STARTING window frame for circling within some range 
    of window frames specified by the parameter window_size. E.g, if 
    window_size = 10 and a circling window frame is 210, then circling
    occured from frames 210-219.
    
    Args:
        fish_pos (array): a 3D array of (x, y) positions for each fish1. 
                    In dimensions 1 and 2 the Nframes x 2 array has the
                    form [[x1, y1], [x2, y2], [x3, y3],...].
                    Dimension 3 = each fish
        head_separation (array): a 2D array of inter-fish head separations,
                        previously calculated. 
        fish_angle_data (array): a 2D array of angles at each window frame
                                  (dim 0) for each fish (dim 1).

        Nframes (int): Number of frames (typically 15,000.)
        
        window_size (int)     : window size for which circling is averaged over.
        circle_fit_threshold (float)     : relative RMSE radius threshold for circling.
        cos_theta_AP_threshold (float)     : antiparallel orientation upper bound for cos(theta) 
        cos_theta_tangent_threshold (float): the cosine(angle) threshold for tangency to the circle
        motion_threshold (float): root mean square frame-to-frame displacement threshold
        head_dist_thresh (int): head distance threshold for the two fish.

    Returns:
        circling_wf (array): a 1D array of circling window frames.
    """
    circling_wf = []
    
    # Assess head-head distance for all frames
    # dh_vec = fish2_pos - fish1_pos  
    # head_separation = np.sqrt(np.sum(dh_vec**2, axis=1))
    head_separation_criterion = (head_separation < head_dist_thresh)
    
    # To save computation time, we're going to consider only the windows
    # with starting frames that meet the head-separation criterion
    # and in which at least one fish is moving.
    # Head separation criterion:
    close_fish_idx = np.array(np.where(head_separation_criterion)) # indexes where met
    # remove starting frames that are within a window-distance from the last frame
    possible_idx = np.delete(close_fish_idx, np.where(
                            close_fish_idx  > (Nframes - window_size)))

    # Evaluate whether at least one fish is moving
    not_moving_idx = []
    for idx in possible_idx:
        fish1_dr = np.diff(fish_pos[idx:idx+window_size,:,0], axis=0)
        fish2_dr = np.diff(fish_pos[idx:idx+window_size,:,1], axis=0)
        fish1_rms = np.sqrt(np.mean(np.sum(fish1_dr**2, axis=1), axis=0))
        fish2_rms = np.sqrt(np.mean(np.sum(fish2_dr**2, axis=1), axis=0))
        # "Not moving" if either fish is not moving
        if (fish1_rms < motion_threshold) or (fish2_rms < motion_threshold):
            not_moving_idx.append(idx) 
    possible_idx = np.setdiff1d(possible_idx, not_moving_idx)

    for idx in possible_idx:
        # Get head position and angle data for all frames in this window
        fish1_positions = fish_pos[idx:idx+window_size,:,0]
        fish2_positions = fish_pos[idx:idx+window_size,:,1]
        fish1_angles = fish_angle_data[idx:idx+window_size,0]
        fish2_angles = fish_angle_data[idx:idx+window_size,1]

        # Array of both fish's head positions (x,y) in this frame window; 
        #   shape 2*window_size x 2
        head_positions = np.concatenate((fish1_positions, fish2_positions), axis=0)
        # Fit to a circle
        taubin_output = TaubinSVD(head_positions)  # output gives (x_c, y_c, r)

        # Goodness of fit to circle
        # Assess distance between each head position and the best-fit circle
        dxy = head_positions - taubin_output[0:2] # x, y distance to center
        dR = np.sqrt(np.sum(dxy**2,1)) # radial distance to center
        # RMS error: difference beteween dR and R (fit radius)
        rmse = np.sqrt(np.mean((dR - taubin_output[2])**2))
        rmse_criterion = rmse < (circle_fit_threshold * taubin_output[2])
        
        # Head separation, for all frames in the window
        dh_window = head_separation[idx:idx+window_size]
        head_separation_window = np.sqrt(np.sum(dh_window**2, axis=1))
        head_separation_window_criterion = \
            (head_separation_window < head_dist_thresh).all()
        
        # Should be antiparallel, so cos(theta) < threshold (ideally cos(theta)==-1)
        cos_theta = np.cos(fish1_angles - fish2_angles)
        angle_criterion = (cos_theta < cos_theta_AP_threshold).all()
        
        # Radius of the best-fit circle should be less than the mean 
        # distance between the heads of the fish over the frame window.
        circle_size_criterion = \
            (taubin_output[2] < np.mean(head_separation_window))

        # Each fish heading should be tangent to the circle
        R1 = fish1_positions - taubin_output[0:1]
        R2 = fish2_positions - taubin_output[0:1]
        n1 = np.column_stack((np.cos(fish1_angles), np.sin(fish1_angles)))
        n2 = np.column_stack((np.cos(fish2_angles), np.sin(fish2_angles)))
        # Dot product for each frame's values; this can be done with 
        # matrix multiplication. Normalize R1, R2 (so result is cos(theta))
        n1dotR1 = np.matmul(n1, R1.transpose()) / np.linalg.norm(R1, axis=1)
        n2dotR2 = np.matmul(n2, R2.transpose()) / np.linalg.norm(R2, axis=1)
        tangent_criterion = np.logical_and((np.abs(n1dotR1) 
                                            < cos_theta_tangent_threshold).all(), 
                                           (np.abs(n2dotR2) 
                                            < cos_theta_tangent_threshold).all())
        
        showDiagnosticPlots = False
        if (rmse_criterion and head_separation_window_criterion
                and angle_criterion and tangent_criterion and circle_size_criterion):
            circling_wf.append(idx+1)  # append the starting frame number
            
            if showDiagnosticPlots:
                print('idx: ', idx, ', rmse: ', rmse)
                print('fish 1 angles: ', fish1_angles)
                print('fish 2 angles: ', fish2_angles)
                print('Cos Theta: ', cos_theta)
                plt.figure()
                plt.plot(fish1_positions[:,0], fish1_positions[:,1], 'x')
                plt.plot(fish2_positions[:,0], fish2_positions[:,1], '+')
                plt.plot(taubin_output[0], taubin_output[1], 'o')
                xplot = np.zeros((200,))
                yplot = np.zeros((200,))
                for k in range(200):
                    xplot[k] = taubin_output[0] + taubin_output[2]*np.cos(k*2*np.pi/200)
                    yplot[k] = taubin_output[1] + taubin_output[2]*np.sin(k*2*np.pi/200)
                plt.plot(xplot, yplot, '-')
                pltInput = input('Press Enter to move on, or "n" to stop after this dataset, or control-C')
                showDiagnosticPlots = (pltInput.lower() == 'n')
                plt.close()
    
    return np.array(circling_wf).astype(int)



def get_contact_frames(body1_x, body2_x, body1_y, body2_y, Nframes, contact_distance):
    """
    Returns a dictionary of window frames for different 
    contact between two fish: any body positions, or head-body contact

    Args:
        body1_x (array): a 2D array of x positions along the 10 body markers of fish1.
        body2_x (array): a 2D array of x positions along the 10 body markers of fish2.
        body1_y (array): a 2D array of y positions along the 10 body markers of fish1.
        body2_y (array): a 2D array of y positions along the 10 body markers of fish2.
        Nframes (int): Number of frames (typically 15,000.)
        contact_distance: the contact distance threshold.

    Returns:
        contact_dict (dictionary): a dictionary of arrays of different contact types.
    """
    contact_dict = {"any_contact": [], "head-body": []}

    for idx in range(Nframes):

        # Any contact: look at closest element of distance matrix between
        # body points
        d0 = np.subtract.outer(body1_x[idx], body2_x[idx]) # all pairs of subtracted x positions
        d1 = np.subtract.outer(body1_y[idx], body2_y[idx]) # all pairs of subtracted y positions
        d = np.sqrt(d0**2 + d1**2) # Euclidean distance matrix, all points
        if np.min(d) < contact_distance:
            contact_dict["any_contact"].append(idx+1)
            
        # Head-body contact
        d_head1_body2 = np.sqrt((body1_x[idx][0] - body2_x[idx])**2 + 
                                (body1_y[idx][0] - body2_y[idx])**2)
        d_head2_body1 = np.sqrt((body2_x[idx][0] - body1_x[idx])**2 + 
                                (body2_y[idx][0] - body1_y[idx])**2)
        if (np.min(d_head1_body2) < contact_distance) or \
            (np.min(d_head2_body1) < contact_distance):
                contact_dict["head-body"].append(idx+1)
                # Note that "any contact" will be automatically satisfied.

    return contact_dict


def get_inferred_contact_frames(dataset, frameWindow, contact_dist):
    """
    Returns an array of frames corresponding to inferred contact, in which
    tracking is bad (zeros values) but inter-fish distance was decreasing
    over some number of preceding frames and was below-threshold immediately 
    before the bad tracking.
    
    Args:
        dataset : dictionary with all dataset info
        frameWindow : number of preceding frames to examine
        contact_dist: the contact distance threshold.

    Returns:
        inf_contact_frames: 1D of array of frames with inferred contact. 
           These are the frames immediately preceding the start of a bad
           tracking run.
    """
    inf_contact_frames = []
    
    # Consider the start of each run of bad frames.
    for badFrame in dataset["bad_bodyTrack_frames"]["combine_frames"][0,:]:
        precedingFrames = np.arange(badFrame-frameWindow,badFrame)
        # check that these frames exist, and aren't part of other runs of bad Frames
        okFrames = np.all(np.isin(precedingFrames, dataset["frameArray"])) and \
                   not(np.any(np.isin(precedingFrames, dataset["bad_bodyTrack_frames"]["raw_frames"])))
        if okFrames:
            # note need to switch order in np.isin()
            this_distance = np.array(dataset["inter-fish_distance"]\
                                     [np.where(np.isin(dataset["frameArray"], precedingFrames))])
            # Check decreasing, and check that last element is within threshold
            if (this_distance[-1]<contact_dist) and np.all(this_distance[:-1] >= this_distance[1:]):
                inf_contact_frames.append(precedingFrames[-1])

    return inf_contact_frames



def get_90_deg_frames(fish1_pos, fish2_pos, fish1_angle_data, fish2_angle_data, 
Nframes, window_size, cos_theta_90_thresh, head_dist_thresh):
    """
    Returns an array of frames for 90-degree orientation events.
    Each frame represents the starting  frame for 90-degree
       orientation events that span some window (parameter window_size).

    Args:
        fish1_pos (array): a 2D array of (x, y) head positions for fish1. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish2_pos (array): a 2D array of (x, y) head positions for fish2. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish1_angle_data (array): a 1D array of angles at each window frame
                                  for fish1.
        fish2_angle_data (array): a 1D array of angles at each window frame
                                  for fish2.

        Nframes (int): Number of frames (typically 15,000.)
        
        window_size (int)      : window size for which circling is averaged over.
        cos_theta_90_thresh (float): the cosine(angle) threshold for 90-degree orientation.
        head_dist_thresh (int) : head distance threshold for the two fish.

    Returns:
        orientations (dict): a dictionary of arrays of window frames for different 
                             90-degree orientation types:
                      
                      - "noneSee": none of the fish see each other.
                      - "oneSees"   : one fish sees the other.
                      - "bothSee": both fish see each other.
    """
    orientations = {"noneSee": [], 
                    "oneSees": [], 
                    "bothSee": []}
    
    # cos_theta for all frames
    cos_theta = np.cos(fish1_angle_data - fish2_angle_data)
    cos_theta_criterion = (np.abs(cos_theta) < cos_theta_90_thresh)
    
    # head-head distance, and distance vector for all frames
    dh_vec = fish2_pos - fish1_pos  # also used later, for the connecting vector
    head_separation = np.sqrt(np.sum(dh_vec**2, axis=1))
    head_separation_criterion = (head_separation < head_dist_thresh)

    # All criteria (and), in each frame
    all_criteria_frame = np.logical_and(cos_theta_criterion, head_separation_criterion)
    all_criteria_window = np.zeros(all_criteria_frame.shape, dtype=bool) # initialize to false
    # Check that criteria are met through the frame window. 
    # Will loop rather than doing some clever Boolean product of offset arrays
    for j in range(all_criteria_frame.shape[0]-window_size+1):
        all_criteria_window[j] =  all_criteria_frame[j:j+window_size].all()
    
    # Indexes (frames - 1) where the criteria are met throughout the window
    ninety_degree_idx = np.array(np.where(all_criteria_window==True))[0,:].flatten() + 1
    # Not sure why the [0,:] is needed, but otherwise returns additional zeros.

    # For each 90 degree event, calculate sign of cross product of fish vectors 
    # for the starting frame, to determine orientation type 
    # Could have done this for all frames and just kept those that met 
    # the above criteria; not sure which is faster
    for idx in ninety_degree_idx:
        fish1_vector = np.array((np.cos(fish1_angle_data[idx]), np.sin(fish1_angle_data[idx])))
        fish2_vector = np.array((np.cos(fish2_angle_data[idx]), np.sin(fish2_angle_data[idx])))
        connecting_vector = dh_vec[idx]
        connecting_vector_norm = connecting_vector / np.linalg.norm(connecting_vector)
        
        # signs of cross products
        fish1xfish2 = np.sign(np.cross(fish1_vector, fish2_vector))
        fish1xconnect = np.sign(np.cross(fish1_vector, connecting_vector_norm))
        fish2xconnect = np.sign(np.cross(fish2_vector, connecting_vector_norm))
        orientation_type = get_orientation_type((fish1xfish2, fish1xconnect,
                                                 fish2xconnect))
        if not(orientation_type is None):
            # make sure it's not "None," for example from positions==0
            orientations[orientation_type].append(idx+1) 


    orientations["noneSee"] = np.array(orientations["noneSee"])
    orientations["oneSees"] = np.array(orientations["oneSees"])
    orientations["bothSee"] = np.array(orientations["bothSee"])
    return orientations


def get_orientation_type(sign_tuple):
    """
    Returns the orientation type of two fish
    given the sign of their respective (a, b, c) vectors.

    Args:
        orientation_tuple (tuple): a tuple of signs of the cross-products
        between two fish:
            fish1xfish2, fish1xconnect, fish2xconnect
    
    Returns:
        (str): "noneSee", "oneSees", or "bothSee".

    """
    # Orientations grouped according to the sign
    # of their respective cross products
    switcher = {
        (1,1,1)   : "noneSee",
        (-1,-1,-1): "noneSee",
        (-1,-1,1) : "oneSees",
        (1,1,-1)  : "oneSees",
        (-1,1,-1) : "oneSees",
        (1,-1,1)  : "oneSees",
        (-1,1,1)  : "bothSee",
        (1,-1,-1) : "bothSee"
    }
    return switcher.get(sign_tuple)


def get_tail_rubbing_frames(body_x, body_y, head_separation,
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
