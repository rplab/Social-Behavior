# -*- coding: utf-8 -*-
# behavior_identification.py
"""
Author:   Raghuveer Parthasarathy
Version ='2.0': 
First versions created By  : Estelle Trieu, 5/26/2022
Major modifications by Raghuveer Parthasarathy, May-July 2023
Last modified May 19, 2024 -- Raghu Parthasarathy

Description
-----------

Module containing all zebrafish pair behavior identification functions:
    - Contact
    - 90-degree orientation
    - Tail rubbing
    - (and more)

"""
import matplotlib.pyplot as plt

import numpy as np
# from circle_fit_taubin import TaubinSVD


def get_contact_frames(body_x, body_y, closest_distance_mm, 
                       contact_distance_threshold, 
                       fish_length_array):
    """
    Returns a dictionary of window frames for different 
    contact between two fish: any body positions, or head-body contact
    
    Assumes frames are contiguous, as should have been checked earlier.

    Args:
        body_x (array): a 3D array (Nframes x 10 x 2 fish) of x positions along the 10 body markers. (px)
        body_y (array): a 3D array (Nframes x 10 x 2 fish) of y positions along the 10 body markers. (px)
        closest_distance_mm (array) : 1D array of closest distance between fish (px)
        contact_distance_threshold: the contact distance threshold, *px*
        fish_length_array: Nframes x 2 array of fish lengths in each frame, mm
                            Only used for identifying the larger fish.

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
        # body points, previously calculated
        if closest_distance_mm[idx] < contact_distance_threshold:
            contact_dict["any_contact"].append(idx+1)
            
        # Head-body contact
        d_head1_body2 = np.sqrt((body_x[idx,0,0] - body_x[idx,:,1])**2 + 
                                (body_y[idx,0,0] - body_y[idx,:,1])**2)
        d_head2_body1 = np.sqrt((body_x[idx,0,1] - body_x[idx,:,0])**2 + 
                                (body_y[idx,0,1] - body_y[idx,:,0])**2)
        fish1_hb_contact = np.min(d_head1_body2) < contact_distance_threshold
        fish2_hb_contact = np.min(d_head2_body1) < contact_distance_threshold
        
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


def get_inferred_contact_frames(dataset, frameWindow, contact_dist_mm):
    """
    Returns an array of frames corresponding to inferred contact, in which
    tracking is bad (zeros values) but inter-fish distance was decreasing
    over some number of preceding frames and was below-threshold immediately 
    before the bad tracking.
    
    Args:
        dataset : dictionary with all dataset info, including head-head
                  distance (mm)
        frameWindow : number of preceding frames to examine
        contact_dist_mm: the contact distance threshold, *mm*

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
            this_distance = np.array(dataset["head_head_distance_mm"]\
                                     [np.where(np.isin(dataset["frameArray"], precedingFrames))])
            # Check decreasing, and check that last element is within threshold
            if (this_distance[-1]<contact_dist_mm) and \
                    np.all(this_distance[:-1] >= this_distance[1:]):
                inf_contact_frames.append(precedingFrames[-1])

    return inf_contact_frames



def get_90_deg_frames(fish_head_pos, fish_angle_data, Nframes, window_size, 
                      cos_theta_90_thresh, perp_maxHeadDist, cosSeeingAngle, 
                      fish_length_array):
    """
    Returns an array of frames for 90-degree orientation events.
    Each frame represents the starting  frame for 90-degree
       orientation events that span some window (parameter window_size).

    Args:
        fish_head_pos (array): a 3D array of (x, y) head positions for both fish.
                          Nframes x 2 [x, y] x 2 fish 
        fish1_angle_data (array): a 2D array of angles; Nframes x 2 fish

        Nframes (int): Number of frames (typically 15,000.)
        
        window_size (int)      : window size for which circling is averaged over.
        cos_theta_90_thresh (float): the cosine(angle) threshold for 90-degree orientation.
        perp_maxHeadDist (float): inter-fish head distance threshold, *px*
        fish_length_array: Nframes x 2 array of fish lengths in each frame, mm
                            Only used for identifying the larger fish.

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
    dh_vec = fish_head_pos[:,:,1] - fish_head_pos[:,:,0]  # also used later, for the connecting vector
    head_separation = np.sqrt(np.sum(dh_vec**2, axis=1))
    head_separation_criterion = (head_separation < perp_maxHeadDist)
    
    # All criteria (and), in each frame
    all_criteria_frame = np.logical_and(cos_theta_criterion, 
                                        head_separation_criterion)
    all_criteria_window = np.zeros(all_criteria_frame.shape, 
                                   dtype=bool) # initialize to false
    # Check that criteria are met through the frame window. 
    # Will loop rather than doing some clever Boolean product of offset arrays
    for j in range(all_criteria_frame.shape[0]-window_size+1):
        all_criteria_window[j] =  all_criteria_frame[j:j+window_size].all()
    
    # Indexes (frames - 1) where the criteria are met throughout the window
    ninety_degree_idx = np.array(np.where(all_criteria_window==True))[0,:].flatten()
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
                        fish_angle_data, window_size, tailRub_maxTailDist, 
                        cos_theta_antipar, tailRub_maxHeadDist): 
    """
    Returns an array of tail-rubbing window frames.

    Args:
        body_x (array): a 3D array of x positions along the 10 body markers, 
                        at each frame. Dimensions [frames, body markers, fish]
        body_y (array): a 3D array of y positions along the 10 body markers, 
                        at each frame. Dimensions [frames, body markers, fish]
        head_separation (array): a 2D array of inter-fish head separations,
                        previously calculated. (mm)
        fish_angle_data (array): a 2D array of angles at each window frame
                                  (dim 0) for each fish (dim 1).

        window_size (int): number of frames for which tail-rubbing criterion must be met
        tailRub_maxTailDist (float): tail distance threshold for the two fish. *px*
        cos_theta_antipar (float): antiparallel orientation upper bound for cos(theta) 
        tailRub_maxHeadDist (float): head distance threshold for the 
                         two fish. *mm*

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
        close_tails[idx] = np.max(smallest_two_d) < tailRub_maxTailDist
        # close_tails[idx] is true if the closest two tail positions in frame [idx] are < tailRub_maxTailDist apart
        
    close_tails = close_tails.flatten()
        
    # cos(angle between headings) for each frame
    # Should be antiparallel, so cos(theta) < threshold 
    #   (ideally cos(theta)==-1)
    cos_theta = np.cos(fish_angle_data[:,0] - fish_angle_data[:,1])
    angle_criterion = (cos_theta < cos_theta_antipar).flatten()

    # Assess head separation, for each frame
    head_separation_criterion = (head_separation < tailRub_maxHeadDist).flatten()
    
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



def get_Cbend_frames(dataset, CSVcolumns, Cbend_threshold = 2/np.pi):
    """ 
    Find frames in which a fish is sharply bent (C-bend)
    Bending is determined by ratio of head to tail-end distance / overall 
    fish length (sum of segments); bend = ratio < threshold
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        CSVcolumns : information on what the columns of dataset["all_data"] are
        Cbend_threshold : consider a fish bent if chord/arc < this threshold
                         Default 2/pi (0.637) corresponds to a semicircle shape
                         For a circle, chord/arc = sin(theta/2) / (theta/2)
    Output : 
        Cbend_frames : dictionary with two keys, 0 and 1, each of which
                       contains a numpy array of frames with 
                       identified C-bend frames for fish 0 and fish 1, 
                       i.e. with bending < Cbend_threshold
    """
    
    # length in each frame, Nframes x Nfish==2 array, mm so convert
    # to px using image scale (um/px)
    fish_length_px = dataset["fish_length_array_mm"] * 1000.0 / dataset["image_scale"]  
    
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
    fish_head_tail_distance = np.sqrt((body_x[:,0,:]-body_x[:,-1,:])**2 + 
                                      (body_y[:,0,:]-body_y[:,-1,:])**2) # Nframes x Nfish==2 array
    Cbend_ratio = fish_head_tail_distance/fish_length_px # Nframes x Nfish==2 array
    Cbend = Cbend_ratio < Cbend_threshold # # True if fish is bent; Nframes x Nfish==2 array
    
    # Dictionary containing C-bend frames for each fish
    Cbend_frames = {0: np.array(np.where(Cbend[:,0])).flatten() + 1, 
                         1: np.array(np.where(Cbend[:,1])).flatten() + 1}

    # Cbend_criterion = np.any(Cbend_ratio < Cbend_threshold, axis=1) # True if either fish is bent
    # Cbend_frames = np.array(np.where(Cbend_criterion)).flatten() + 1
    
    return Cbend_frames


def get_Jbend_frames(dataset, CSVcolumns, JbendThresholds = (0.98, 0.34, 0.70)):
    """ 
    Find frames in which one or more fish have a J-bend: straight anterior
    and bent posterior.
    A J-bend is defined by:
        - body points 1-5 are linearly correlated: |Pearson r| > JbendThresholds[0]  
          Note that this avoids chord / arc distance issues with point #2
          sometimes being anterior of #1
        - cos(angle) between (points 9-10) and heading angle < JbendThresholds[1]
        - cos(angle) between (points 8-9) and heading angle < JbendThresholds[2]
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        CSVcolumns : information on what the columns of dataset["all_data"] are
        JbendThresholds : see J-bend definition above
    Output : 
        Jbend_frames : dictionary with two keys, 0 and 1, each of which
                       contains a numpy array of frames with 
                       identified J-bend frames for fish 0 and fish 1

    """
    
    midColumn = int(CSVcolumns["body_Ncolumns"]/2)
    # print('midColumn should be 5: ', midColumn)
    
    # All body positions, as in C-bending function
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
    
    angle_data = dataset["all_data"][:,CSVcolumns["angle_data_column"], :]

    # Angle between each pair of points and the heading angle
    segment_angles = np.zeros((body_x.shape[0], body_x.shape[1]-1, body_x.shape[2]))
    for j in range(segment_angles.shape[1]):
        segment_angles[:, j, :] = np.arctan2(body_y[:,j+1,:]-body_y[:,j,:], 
                          body_x[:,j+1,:]-body_x[:,j,:])
    
    # mean values, repeated to allow subtraction
    mean_x = np.mean(body_x[:,0:midColumn,:], axis=1)
    mean_x = np.swapaxes(np.tile(mean_x, (midColumn, 1, 1)), 0, 1)
    mean_y = np.mean(body_y[:,0:midColumn,:], axis=1)
    mean_y = np.swapaxes(np.tile(mean_y, (midColumn, 1, 1)), 0, 1)
    Npts = midColumn # number of points
    cov_xx = np.sum((body_x[:,0:midColumn,:]-mean_x)*(body_x[:,0:midColumn,:]-mean_x), 
                    axis=1)/(Npts-1)
    cov_yy = np.sum((body_y[:,0:midColumn,:]-mean_y)*(body_y[:,0:midColumn,:]-mean_y), 
                    axis=1)/(Npts-1)
    cov_xy = np.sum((body_x[:,0:midColumn,:]-mean_x)*(body_y[:,0:midColumn,:]-mean_y), 
                    axis=1)/(Npts-1)
    Tr = cov_xx + cov_yy
    DetCov = cov_xx*cov_yy - cov_xy**2
    
    # Two eigenvalues for each frame, each fish
    eig_array = np.zeros((Tr.shape[0], Tr.shape[1], 2))
    eig_array[:,:,0]  = Tr/2.0 + np.sqrt((Tr**2)/4.0 - DetCov)
    eig_array[:,:,1] = Tr/2.0 - np.sqrt((Tr**2)/4.0 - DetCov)
    anterior_straight_var = np.max(eig_array, axis=2)/np.sum(eig_array, axis=2)
    anterior_straight_criterion = anterior_straight_var \
        > JbendThresholds[0] # Nframes x Nfish==2 array; Boolean 

    # Evaluate angle between last pair of points and the heading angle
    cos_angle_last_heading = np.cos(segment_angles[:,-1,:] - angle_data)
    cos_angle_last_criterion = np.abs(cos_angle_last_heading) < JbendThresholds[1]

    # Evaluate the angle between second-last pair of points and the heading angle
    cos_angle_2ndlast_heading = np.cos(segment_angles[:,-2,:] - angle_data)
    cos_angle_2ndlast_criterion = np.abs(cos_angle_2ndlast_heading) < JbendThresholds[2]
    
    allCriteria = np.all(np.stack((anterior_straight_criterion, 
                           cos_angle_last_criterion, cos_angle_2ndlast_criterion), 
                           axis=2), axis=2) # for each fish, all criteria must be true    
    
    # Dictionary containing J-bend frames for each fish
    Jbend_frames = {0: np.array(np.where(allCriteria[:,0])).flatten() + 1, 
                         1: np.array(np.where(allCriteria[:,1])).flatten() + 1}
    
    return Jbend_frames


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

def get_approach_flee_frames(dataset, CSVcolumns, 
                           speed_threshold_mm_s = 20, 
                           cos_angle_thresh = 0.5,
                           min_frame_duration = (2, 2)):
    """ 
    Find frames in which a fish is rapidly approaching the other fish,
        or fleeing from the other fish
    Approaching is defined by the following all being true over 
        a frame interval of at least min_frame_duration[0]
        - speed > speed_threshold
        - closest distance btw fish is decreasing
        - cos(angle) between the fish heading and the vector to the closest 
          point on the other fish is greater than cos_angle_thresh
    Fleeing is defined by the following all being true over 
        a frame interval of min_frame_duration[1]
        - speed > speed_threshold
        - closest distance btw fish is increasing
        - cos(angle) between the fish heading and the vector to the closest 
          point on the other fish is less than cos_angle_thresh 
          (NOT -cos_angle_thresh -- doesn't have to be moving directly
           away; allow anything not approaching)
    Note that speed has previously been calculated 
        (dataset['speed_array_mm_s'], Nframes x Nfish==2 array)
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        CSVcolumns : information on what the columns of dataset["all_data"] are
        min_frame_duration : number of frames over which condition must be met.
                      Tuple, for Approaching [0] and Fleeing [1]
                      Default (2, 2)
        speed_threshold_mm_s : speed threshold, mm/s, default 20
        cos_angle_thresh : min cosine of angle between heading and 
                      vector to other fish to consider as approaching. 
                      For approaching, cos(angle) must be > cos_angle_thres
                      For fleeing, cos(angle) must be < cos_angle_thresh 
                         (not -cos_angle_thresh -- allow very wide range)
                      Default 0.5 (60 degrees)
    Output : 
        approaching_frames : dictionary with two keys, 0 and 1, each of 
                       which contains a numpy array of frames in which
                       fish 0 is approaching and fish 1 is approaching, resp.
        fleeinging_frames : dictionary with two keys, 0 and 1, each of 
                       which contains a numpy array of frames in which
                       fish 0 is fleeing and fish 1 is fleeing, resp.
    """

    # All body positions, as in C-bending function
    angle_data = dataset["all_data"][:,CSVcolumns["angle_data_column"], :]
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
    
    Nframes = body_x.shape[0]

    # Minimum head-body distances: closest element of distance matrix 
    # between head of one fish and body of the other.
    # Also angles; head to closest point on other fish
    # Array of Nframes x Nfish==2 values
    # First col. = Fish 0 head to Fish 1 body; Second 1 head to 0 body
    d_head_body = np.zeros((Nframes, 2), dtype = float)
    angle_head_body = np.zeros((Nframes, 2), dtype = float)
    for idx in range(Nframes):
        d_head_body[idx, 0] = np.min(np.sqrt((body_x[idx,0,0] - body_x[idx,:,1])**2 + 
                                (body_y[idx,0,0] - body_y[idx,:,1])**2))
        d_head_body[idx, 1] = np.min(np.sqrt((body_x[idx,0,1] - body_x[idx,:,0])**2 + 
                                (body_y[idx,0,1] - body_y[idx,:,0])**2))
        # Also want the index of the closest position, for calculating approach angle
        d_argmin_head0_body1 = np.argmin(np.sqrt((body_x[idx,0,0] - body_x[idx,:,1])**2 + 
                                (body_y[idx,0,0] - body_y[idx,:,1])**2))
        d_argmin_head1_body0 = np.argmin(np.sqrt((body_x[idx,0,1] - body_x[idx,:,0])**2 + 
                                (body_y[idx,0,1] - body_y[idx,:,0])**2))
        angle_head_body[idx, 0] = np.arctan2(body_y[idx,d_argmin_head0_body1,1] - body_y[idx,0,0], 
                                            body_x[idx,d_argmin_head0_body1,1] - body_x[idx,0,0])
        angle_head_body[idx, 1] = np.arctan2(body_y[idx,d_argmin_head1_body0,0] - body_y[idx,0,1], 
                                            body_x[idx,d_argmin_head1_body0,0] - body_x[idx,0,1])
    
    # Cosine of angle between heading and closest distance vector
    cos_angle_head_body = np.cos(angle_head_body - angle_data)
    print('Cosine angle shape: ', cos_angle_head_body.shape)
    
    # frame-to-frame change in distance between fish (px)
    delta_dheadbody = d_head_body[1:,:] - d_head_body[:-1, :]
    
    # a redundant variable, but useful for plots
    speed = dataset['speed_array_mm_s']
    
    # Is the distance decreasing over min_frame_duration[0] frames (from the
    # initial frame)? Is it increasing?
    # Boolean array; [0] is for head0-body1; [1] is head1-body0
    indices = np.arange(delta_dheadbody.shape[0] - min_frame_duration[0] + 1)[:, None] + \
        np.arange(min_frame_duration[0])
    distance_decr_over_window = np.zeros((Nframes, 2), dtype=bool)
    distance_decr_over_window[:(-min_frame_duration[0]),:] = \
        np.all(delta_dheadbody[indices,:] < 0, axis=1)    
    indices = np.arange(delta_dheadbody.shape[0] - min_frame_duration[1] + 1)[:, None] + \
        np.arange(min_frame_duration[1])
    distance_incr_over_window = np.zeros((Nframes, 2), dtype=bool)
    distance_incr_over_window[:(-min_frame_duration[1]),:] = \
        np.all(delta_dheadbody[indices,:] > 0, axis=1)    
    
    # All the criteria for approaching. (All must be true)
    # Nframes x Nfish==2 array, Boolean. (Could just multiply, but this
    # might be clearer to read)
    approaching = np.all(np.stack((speed > speed_threshold_mm_s, 
                                  cos_angle_head_body > cos_angle_thresh, 
                                  distance_decr_over_window), axis=2),
                         axis=2) 
    # All the criteria for fleeing. (All 
    fleeing = np.all(np.stack((speed > speed_threshold_mm_s, 
                                  cos_angle_head_body < cos_angle_thresh, 
                                  distance_incr_over_window), axis=2),
                         axis=2) 

    # Dictionary containing approaching frames for each fish
    approaching_frames = {0: np.array(np.where(approaching[:,0])).flatten() + 1, 
                         1: np.array(np.where(approaching[:,1])).flatten() + 1}
    # Dictionary containing fleeing frames for each fish
    fleeing_frames = {0: np.array(np.where(fleeing[:,0])).flatten() + 1, 
                         1: np.array(np.where(fleeing[:,1])).flatten() + 1}
    
    makeDiagnosticPlots = False
    if makeDiagnosticPlots:
        
        xlimits = (90, 170)
        plt.figure()
        plt.plot(range(Nframes), speed[:,0], label='Fish 0')
        plt.plot(range(Nframes), speed[:,1], label='Fish 1')
        plt.xlabel('Frame')
        plt.ylabel('Speed (px/frame)')
        plt.title(dataset['dataset_name'])
        plt.legend()
        plt.xlim(xlimits[0], xlimits[1])
        
        plt.figure()
        plt.plot(range(Nframes), d_head_body[:,0], label='head0-body1')
        plt.plot(range(Nframes), d_head_body[:,1], label='head1-body0')
        plt.xlabel('Frame')
        plt.ylabel('Inter-fish distance (px)')
        plt.title(dataset['dataset_name'])
        plt.legend()
        plt.xlim(xlimits[0], xlimits[1])
        
        plt.figure()
        plt.plot(range(Nframes), approaching[:,0], color='royalblue', 
                 label='Fish0 approaching Fish1')
        plt.plot(range(Nframes), approaching[:,1], color='darkorange', 
                 label='Fish1 approaching Fish0')
        plt.plot(range(Nframes), -1.0*(fleeing[:,0].astype(int)), 
                 color='lightseagreen', 
                 label='Fish0 fleeing from Fish1')
        plt.plot(range(Nframes), -1.0*(fleeing[:,1].astype(int)), 
                 color='gold', 
                 label='Fish1 fleeing from Fish0')
        plt.title(dataset['dataset_name'])
        plt.ylabel('Approaching (+1), Fleeing (-1)')
        plt.legend()
        plt.xlim(xlimits[0], xlimits[1])

    return approaching_frames, fleeing_frames
    

def get_relative_orientation(dataset, CSVcolumns):
    """ 
    Calculate the relative orientation of each fish with respect to the
    head-to-head vector to the other fish.
    Note that speed has previously been calculated 
        (dataset['speed_array_mm_s'], Nframes x Nfish==2 array)
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        CSVcolumns : information on what the columns of dataset["all_data"] are
    Output : 
        relative_orientation : numpy array Nframes x Nfish==2 of 
            relative orientation (phi), radians, for fish 0 and fish 1
    """
    # All heading angles
    angle_data = dataset["all_data"][:,CSVcolumns["angle_data_column"], :]
    # all head positions
    head_pos_data = dataset["all_data"][:,CSVcolumns["head_column_x"]:CSVcolumns["head_column_y"]+1, :]
        # head_pos_data is Nframes x 2 (x and y positions) x 2 (Nfish) array of head positions

    # head-head distance, px, and distance vector for all frames
    dh_vec = head_pos_data[:,:,1] - head_pos_data[:,:,0]  # also used later, for the connecting vector
    
    v0 = np.stack((np.cos(angle_data[:, 0]), 
                   np.sin(angle_data[:, 0])), axis=1)
    v1 = np.stack((np.cos(angle_data[:, 1]), 
                   np.sin(angle_data[:, 1])), axis=1)
    
    dot_product_0 = np.sum(v0 * dh_vec, axis=1)
    magnitude_product_0 = np.linalg.norm(v0, axis=1) * np.linalg.norm(dh_vec, axis=1)
    phi0 = np.arccos(dot_product_0 / magnitude_product_0)
    
    dot_product_1 = np.sum(v1 * -dh_vec, axis=1)
    magnitude_product_1 = np.linalg.norm(v1, axis=1) * np.linalg.norm(dh_vec, axis=1)
    phi1 = np.arccos(dot_product_1 / magnitude_product_1)
    
    relative_orientation = np.stack((phi0, phi1), axis=1)  
    
    return relative_orientation

    
