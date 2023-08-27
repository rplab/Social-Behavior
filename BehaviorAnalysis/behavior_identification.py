# -*- coding: utf-8 -*-
# behavior_identification.py
"""
Author:   Raghuveer Parthasarathy
Version ='2.0': 
First versions created By  : Estelle Trieu, 5/26/2022
Major modifications by Raghuveer Parthasarathy, May-July 2023
Last modified July 23, 2023 -- Raghu Parthasarathy

Description
-----------

Module containing all zebrafish pair behavior identification functions:
    - Circling 
    - Contact
    - 90-degree orientation
    - Tail rubbing

Requires prior import of TaubinSVD() from circle_fit_taubin (for circling)

"""

import numpy as np
# from circle_fit_taubin import TaubinSVD

def get_circling_frames(fish_pos, head_separation, fish_angle_data,
                    Nframes, window_size, circle_fit_threshold, 
                    cos_theta_AP_threshold, 
                    cos_theta_tangent_threshold, motion_threshold, 
                    head_distance_thresh):
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
        
        window_size (int)     : number of frames over which circling is assessed.
        circle_fit_threshold (float)     : relative RMSE radius threshold for circling.
        cos_theta_AP_threshold (float)     : antiparallel orientation upper bound for cos(theta) 
        cos_theta_tangent_threshold (float): the cosine(angle) threshold for tangency to the circle
        motion_threshold (float): root mean square frame-to-frame displacement threshold
        head_distance_thresh (int): head distance threshold for the two fish.

    Returns:
        circling_wf (array): a 1D array of circling window frames.
    """
    circling_wf = []
    
    # Assess head-head distance for all frames
    # dh_vec = fish2_pos - fish1_pos  
    # head_separation = np.sqrt(np.sum(dh_vec**2, axis=1))
    head_separation_criterion = (head_separation < head_distance_thresh)
    
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
            (head_separation_window < head_distance_thresh).all()
        
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



def get_contact_frames(body_x, body_y, contact_distance, fish_length_array):
    """
    Returns a dictionary of window frames for different 
    contact between two fish: any body positions, or head-body contact
    
    Assumes frames are contiguous, as should have been checked earlier.

    Args:
        body_x (array): a 3D array (Nframes x 10 x 2 fish) of x positions along the 10 body markers.
        body_y (array): a 3D array (Nframes x 10 x 2 fish) of y positions along the 10 body markers.
        contact_distance: the contact distance threshold.
        fish_length_array: Nframes x 2 array of fish lengths in each frame

    Returns:
        contact_dict (dictionary): a dictionary of arrays of different 
            contact types: any_contact, head-body contact (a subset)
            larger or smaller fish head contact (a subset)
    """
    contact_dict = {"any_contact": [], "head-body": [], 
                    "larger_fish_head_contact": [], 
                    "smaller_fish_head_contact": []}

    for idx in range(body_x.shape[0]):

        # Any contact: look at closest element of distance matrix between
        # body points
        d0 = np.subtract.outer(body_x[idx,:,0], body_x[idx,:,1]) # all pairs of subtracted x positions
        d1 = np.subtract.outer(body_y[idx,:,0], body_y[idx,:,1]) # all pairs of subtracted y positions
        d = np.sqrt(d0**2 + d1**2) # Euclidean distance matrix, all points
        if np.min(d) < contact_distance:
            contact_dict["any_contact"].append(idx+1)
            
        # Head-body contact
        d_head1_body2 = np.sqrt((body_x[idx,0,0] - body_x[idx,:,1])**2 + 
                                (body_y[idx,0,0] - body_y[idx,:,1])**2)
        d_head2_body1 = np.sqrt((body_x[idx,0,1] - body_x[idx,:,0])**2 + 
                                (body_y[idx,0,1] - body_y[idx,:,0])**2)
        fish1_hb_contact = np.min(d_head1_body2) < contact_distance
        fish2_hb_contact = np.min(d_head2_body1) < contact_distance
        
        if fish1_hb_contact or fish2_hb_contact:
            contact_dict["head-body"].append(idx+1)
            # Note that "any contact" will be automatically satisfied.
            if not (fish1_hb_contact and fish2_hb_contact):
                # Only one is making head-body contact
                largerFishIdx = np.argmax(fish_length_array[idx,:])
                if largerFishIdx==0 and fish1_hb_contact:
                    contact_dict["larger_fish_head_contact"].append(idx+1)
                else:
                    contact_dict["smaller_fish_head_contact"].append(idx+1)

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



def get_90_deg_frames(fish_pos, fish_angle_data, Nframes, window_size, 
                      cos_theta_90_thresh, head_distance_thresh, cosSeeingAngle, 
                      fish_length_array):
    """
    Returns an array of frames for 90-degree orientation events.
    Each frame represents the starting  frame for 90-degree
       orientation events that span some window (parameter window_size).

    Args:
        fish_pos (array): a 3D array of (x, y) head positions for both fish.
                          Nframes x 2 [x, y] x 2 fish 
        fish1_angle_data (array): a 2D array of angles; Nframes x 2 fish

        Nframes (int): Number of frames (typically 15,000.)
        
        window_size (int)      : window size for which circling is averaged over.
        cos_theta_90_thresh (float): the cosine(angle) threshold for 90-degree orientation.
        head_distance_thresh (int) : head distance threshold for the two fish.
        fish_length_array: Nframes x 2 array of fish lengths in each frame

    Returns:
        orientations (dict): a dictionary of arrays of window frames for different 
                             90-degree orientation types:
                      
                      - "noneSee": none of the fish see each other.
                      - "oneSees"   : one fish sees the other.
                      - "bothSee": both fish see each other.
                      - "larger_fish_sees", "smaller_fish_sees" : subset of
                         oneSees; the larger or smaller of the fish see the other
    """
    orientations = {"noneSee": [], 
                    "oneSees": [], 
                    "bothSee": [],
                    "larger_fish_sees": [], 
                    "smaller_fish_sees": []}

    # cos_theta for all frames
    cos_theta = np.cos(fish_angle_data[:,0] - fish_angle_data[:,1])
    cos_theta_criterion = (np.abs(cos_theta) < cos_theta_90_thresh)
    
    # head-head distance, and distance vector for all frames
    dh_vec = fish_pos[:,:,1] - fish_pos[:,:,0]  # also used later, for the connecting vector
    head_separation = np.sqrt(np.sum(dh_vec**2, axis=1))
    head_separation_criterion = (head_separation < head_distance_thresh)

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

    # For each 90 degree event, determine the orientation type -- whether
    # 0, 1, or both fish are in the forward half-plane of the other
    # Could have done this for all frames and just kept those that met 
    # the above criteria; not sure which is faster, but this makes testing
    # easier.
    for idx in ninety_degree_idx:
        # (dx, dy) from fish 1 to 2, normalized to unit length
        # Angle of the connecting vector from fish 1 to 2
        dh_angle_12 = np.arctan2(dh_vec[idx,1], dh_vec[idx,0])
        fish1sees = np.cos(fish_angle_data[idx,0] - dh_angle_12) >= cosSeeingAngle
        fish2sees = np.cos(fish_angle_data[idx,1] - dh_angle_12) <= -1.0*cosSeeingAngle
        if fish1sees or fish2sees:
            if fish1sees and fish2sees:
                orientations["bothSee"].append(idx+1)
            else:
                orientations["oneSees"].append(idx+1)
                # For determining whether the larger or smaller fish is the one 
                # that "sees", if only one does
                largerFishIdx = np.argmax(fish_length_array[idx,:])
                if largerFishIdx==0 and fish1sees:
                    orientations["larger_fish_sees"].append(idx+1)
                else:
                    orientations["smaller_fish_sees"].append(idx+1)
        else:
            orientations["noneSee"].append(idx+1)

    orientations["bothSee"] = np.array(orientations["bothSee"])
    orientations["noneSee"] = np.array(orientations["noneSee"])
    orientations["oneSees"] = np.array(orientations["oneSees"])
    orientations["larger_fish_sees"] = np.array(orientations["larger_fish_sees"])
    orientations["smaller_fish_sees"] = np.array(orientations["smaller_fish_sees"])
    
    return orientations



def get_tail_rubbing_frames(body_x, body_y, head_separation,
                        fish_angle_data, window_size, tail_dist, 
                        tail_anti_high, head_distance_thresh): 
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
        head_distance_thresh (int): head distance threshold for the two fish. 

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
    head_separation_criterion = (head_separation < head_distance_thresh).flatten()
    
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



def get_bent_frames(dataset, CSVcolumns, bending_threshold = 2/np.pi):
    """ 
    Find frames in which one or more fish are bent (beyond threshold)
    Bending is determined by ratio of head to tail-end distance / overall 
    fish length (sum of segments); bend = ratio < threshold
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        CSVcolumns : information on what the columns of dataset["all_data"] are
        bending_threshold : consider a fish bent if chord/arc < this threshold
                         Default 2/pi (0.637) corresponds to a semicircle shape
                         For a circle, chord/arc = sin(theta/2) / (theta/2)
    Output : list of frames with bending < bending_threshold for any fish
    """
    
    fish_length = dataset["fish_length_array"]  # length in each frame, Nframes x Nfish==2 array
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
    fish_head_tail_distance = np.sqrt((body_x[:,0,:]-body_x[:,-1,:])**2 + 
                                      (body_y[:,0,:]-body_y[:,-1,:])**2) # Nframes x Nfish==2 array
    bend_ratio = fish_head_tail_distance/fish_length # Nframes x Nfish==2 array
    bend_criterion = np.any(bend_ratio < bending_threshold, axis=1) # True if either fish is bent
    # print('number of bent frames: ', np.sum(bend_criterion))
    # print('number of both-bent frames: ', np.sum(np.all(bend_ratio < bending_threshold, axis=1)))
    
    # Nframes = np.shape(fish_length)[0] 
    # plt.figure()
    # plt.plot(range(Nframes), bend_ratio[:,0], color='magenta', label='Fish 1')
    # plt.plot(range(Nframes), bend_ratio[:,1], color='olivedrab', label='Fish 2')
    
    # plt.figure()
    # plt.hist(bend_ratio[:,0], bins=50, color='magenta', label='Fish 1')
    
    bendingFrames = np.array(np.where(bend_criterion)).flatten() + 1
    
    return bendingFrames


def calcOrientationXCorr(dataset, CSVcolumns, window_size = 25, makeDiagnosticPlots = False):
    """
    Heading angle Co-orientation behavior; see July 2023 notes
    Calculate cross-correlation of fish heading angles, over a sliding window
    xcorr at frame j is the normalized cross-correlation over the window 
        *ending* at j
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        CSVcolumns : information on what the columns of dataset["all_data"] are
        window_size : number frames for sliding window
        makeDiagnosticPlots : if true, plot angles, xcorr
    Outputs : 
        xcorr : normalized cross-correlation at each frame (for the sliding 
                            window frames *ending* at that frame)
        
    To do:
        [not important] Make xcorr use a sliding window *starting* at that frame
        [not important] Make functions and eliminate redundant code

    """
    angle_data = dataset[:,CSVcolumns["angle_data_column"], :]
    Nframes = np.shape(angle_data)[0] 
    
    # Cross-correlation of angles over a sliding window
    # (Method of sliding sums from
    #    https://stackoverflow.com/questions/12709853/python-running-cumulative-sum-with-a-given-window)
    # Should make this a function...
    # First unwrap to avoid large jumps
    angles_u1 = np.unwrap(angle_data[:,0], axis=0).flatten()
    angles_u2 = np.unwrap(angle_data[:,1], axis=0)
    # The running mean of each set of angles
    angles_1_mean = np.cumsum(angles_u1)
    angles_1_mean[window_size:] = angles_1_mean[window_size:] - angles_1_mean[:-window_size]
    angles_1_mean = angles_1_mean/window_size
    angles_2_mean = np.cumsum(angles_u2)
    angles_2_mean[window_size:] = angles_2_mean[window_size:] - angles_2_mean[:-window_size]
    angles_2_mean = angles_2_mean/window_size
    angles_1_meanSub = angles_u1 - angles_1_mean
    angles_2_meanSub = angles_u2 - angles_2_mean
    xcorr_num = np.cumsum(angles_1_meanSub * angles_2_meanSub) # numerator of cross-correlation
    xcorr_num[window_size:] = xcorr_num[window_size:] - xcorr_num[:-window_size]
    xcorr_mag1 = np.cumsum(angles_1_meanSub**2)
    xcorr_mag1[window_size:] = xcorr_mag1[window_size:] - xcorr_mag1[:-window_size]
    xcorr_mag2 = np.cumsum(angles_2_meanSub**2)
    xcorr_mag2[window_size:] = xcorr_mag2[window_size:] - xcorr_mag2[:-window_size]
    xcorr = xcorr_num / np.sqrt(xcorr_mag1 * xcorr_mag2)
        
    
    if makeDiagnosticPlots:
        print('Diagnostic plots')
        xlimits =  (7000,8000) # (12000, 12050) #

        plt.figure()
        plt.plot(range(Nframes), angle_data[:,0]*180/np.pi, color='magenta', label='Fish 1')
        plt.plot(range(Nframes), angle_data[:,1]*180/np.pi, color='olivedrab', label='Fish 2')
        plt.title('Angles of fish 1, 2; degrees')
        plt.xlabel('Frame')
        plt.xlim(xlimits)
        plt.legend()
    
        plt.figure()
        plt.hist(xcorr, 40, color='deepskyblue', edgecolor='steelblue', linewidth=1.2)
        titleString = 'Cross-correlation, Dataset: ' + dataset["dataset_name"]
        plt.title(titleString)
        plt.xlabel('Angle Cross-correlation')
        

        plt.figure()
        plt.plot(range(Nframes), xcorr, color='navy')
        titleString = 'Cross-correlation of heading angles, Dataset: ' + dataset["dataset_name"]
        plt.title(titleString)
        plt.xlim(xlimits)
        plt.xlabel('Frame')
        plt.xlabel('Xcorr')
        # plt.ylim((0, 10))

    
    return xcorr