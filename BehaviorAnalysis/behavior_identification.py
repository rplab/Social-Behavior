# -*- coding: utf-8 -*-
# behavior_identification.py
"""
Author:   Raghuveer Parthasarathy
Version ='2.0': 
First versions created By  : Estelle Trieu, 5/26/2022
Major modifications by Raghuveer Parthasarathy, May-July 2023
Last modified Sept. 6, 2024 -- Raghu Parthasarathy

Description
-----------

Module containing all zebrafish pair behavior identification functions:
    - extract_behaviors(), which calls all the other functions
    - Contact
    - 90-degree orientation
    - Tail rubbing
    - (and more)

"""
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np
from toolkit import wrap_to_pi, combine_all_values_constrained, \
    plot_probability_distr, make_2D_histogram,calculate_value_corr_all, \
    plot_function_allSets, behaviorFrameCount_all
from behavior_identification_single import average_bout_trajectory_allSets
from scipy.stats import skew
# from circle_fit_taubin import TaubinSVD


def get_basic_two_fish_characterizations(datasets, CSVcolumns,
                                             expt_config, params):
    """
    For each dataset, perform “basic” two-fish characterizations  
        (e.g. inter-fish distance, relative orientation, 
         relative heading alignment)
    
    Inputs:
        datasets : dictionaries for each dataset
        CSVcolumns : CSV column information (dictionary)
        expt_config : dictionary of configuration information
        params : dictionary of all analysis parameters
    Returns:
        datasets : dictionaries for each dataset. datasets[j] contains
                    all the information for dataset j.
    """
    
    # Number of datasets
    N_datasets = len(datasets)
    
    print('Basic two-fish characterizations: ')
    # For each dataset, measure inter-fish distance (head-head and
    # closest points) in each frame and the relative orientation, 
    # and calculate a sliding window cross-correlation of heading angles
    for j in range(N_datasets):

        # Get the inter-fish distance (distance between head positions) in 
        # each frame (Nframes x 1 array). Units = mm
        datasets[j]["head_head_distance_mm"], datasets[j]["closest_distance_mm"] = \
            get_interfish_distance(datasets[j]["all_data"], CSVcolumns,
                                   datasets[j]["image_scale"])
        
        # Relative orientation of fish (angle between heading and
        # connecting vector). Nframes x Nfish==2 array; radians
        datasets[j]["relative_orientation"] = \
            get_relative_orientation(datasets[j], CSVcolumns)   

        # Relative orientation of fish (angle between heading and
        # connecting vector). Nframes x Nfish==2 array; radians
        datasets[j]["relative_heading_angle"] = \
            get_relative_heading_angle(datasets[j], CSVcolumns)   
        
        # Get the sliding window cross-correlation of heading angles
        datasets[j]["xcorr_array"] = \
            calcOrientationXCorr(datasets[j], CSVcolumns, 
                                 params["angle_xcorr_windowsize"])
        
    # For each dataset, exclude bad tracking frames from calculations of
    # the mean and std. absolute difference in fish length
    # the mean inter-fish distance
    # exclude bad tracking frames from the 
    # calculation of the mean angle-heading cross-correlation
    # if the bad frames occur anywhere in the sliding window
    for j in range(N_datasets):
        print('Dataset: ', datasets[j]["dataset_name"])
        print('   Removing bad frames from stats for inter-fish distance')
        goodIdx = np.where(np.in1d(datasets[j]["frameArray"], 
                                   datasets[j]["bad_bodyTrack_frames"]["raw_frames"], 
                                   invert=True))[0]
        goodHHDistanceArray = datasets[j]["head_head_distance_mm"][goodIdx]
        goodClosestDistanceArray = datasets[j]["closest_distance_mm"][goodIdx]
        datasets[j]["head_head_distance_mm_mean"] = np.mean(goodHHDistanceArray)
        datasets[j]["closest_distance_mm_mean"] = np.mean(goodClosestDistanceArray)
        print(f'   Mean head-to-head distance {datasets[j]["head_head_distance_mm_mean"]:.2f} mm')
        print(f'   Mean closest distance {datasets[j]["closest_distance_mm_mean"]:.2f} px')

        goodLengthArray = datasets[j]["fish_length_array_mm"][goodIdx]
        datasets[j]["fish_length_Delta_mm_mean"] = np.mean(np.abs(np.diff(goodLengthArray, 1)))
        datasets[j]["fish_length_Delta_mm_std"] = np.std(np.abs(np.diff(goodLengthArray, 1)))
        print('   Mean +/- std. of difference in fish length: ', 
              f'{datasets[j]["fish_length_Delta_mm_mean"]:.3f} +/- ',
              f'{datasets[j]["fish_length_Delta_mm_std"]:.3f} mm')
    
        print('Removing bad frames from stats for angle xcorr')
        badFrames = datasets[j]["bad_bodyTrack_frames"]["raw_frames"]
        expandBadFrames = []
        for k in range(len(badFrames)):
            # make a list of all frames that are within windowsize prior to bad frames
            expandBadFrames = np.append(expandBadFrames, 
                                        np.arange(badFrames[k] - 
                                                  params["angle_xcorr_windowsize"]+1, 
                                                  badFrames[k]))
        expandBadFrames = np.unique(expandBadFrames)    
        goodIdx = np.where(np.in1d(datasets[j]["frameArray"], 
                                   expandBadFrames, invert=True))[0]
        goodXCorrArray = datasets[j]["xcorr_array"][goodIdx]
        # Also limit to finite values
        finiteGoodXCorrArray = goodXCorrArray[np.isfinite(goodXCorrArray)]
        datasets[j]["AngleXCorr_mean"] = np.mean(finiteGoodXCorrArray)
        # print(f'   Mean heading angle XCorr: {datasets[j]["AngleXCorr_mean"]:.4f}')
        # Calculating the std dev and skew, but won't write to CSV
        datasets[j]["AngleXCorr_std"] = np.std(finiteGoodXCorrArray)
        datasets[j]["AngleXCorr_skew"] = skew(finiteGoodXCorrArray)
        # print(f'   (Not in CSV) std, skew heading angle XCorr: {datasets[j]["AngleXCorr_std"]:.4f}, {datasets[j]["AngleXCorr_skew"]:.4f}')

    return datasets


    
def get_interfish_distance(all_data, CSVcolumns, image_scale):
    """
    Get the inter-fish distance (calculated both as the distance 
        between head positions and as the closest distance)
        in each frame 
    Input:
        all_data : all position data, from dataset["all_data"]
        CSVcolumns : CSV column information (dictionary)
        image_scale : scale, um/px; from dataset["image_scale"]
    Output
        head_head_distance_mm : head-head distance (mm), Nframes x 1 array 
        closest_distance_mm : closest distance (mm), Nframes x 1 array
    """
    
    # head-head distance
    head_x = all_data[:,CSVcolumns["head_column_x"],:] # x, both fish
    head_y = all_data[:,CSVcolumns["head_column_y"],:] # y, both fish
    dx = np.diff(head_x)
    dy = np.diff(head_y)
    # distance, mm
    head_head_distance_mm = (np.sqrt(dx**2 + dy**2))*image_scale/1000.0
    
    # body-body distance, for all pairs of points
    body_x = all_data[:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = all_data[:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
    closest_distance_mm = np.zeros((body_x.shape[0],1))
    for idx in range(body_x.shape[0]):
        d0 = np.subtract.outer(body_x[idx,:,0], body_x[idx,:,1]) # all pairs of subtracted x positions
        d1 = np.subtract.outer(body_y[idx,:,0], body_y[idx,:,1]) # all pairs of subtracted y positions
        d = np.sqrt(d0**2 + d1**2) # Euclidean distance matrix, all points
        closest_distance_mm[idx] = np.min(d)*image_scale/1000.0 # mm
    
    return head_head_distance_mm, closest_distance_mm

def extract_behaviors(dataset, params, CSVcolumns): 
    """
    Calls functions to identify frames corresponding to each two-fish
    behavioral motif in a single dataset.
    
    Inputs:
        dataset : dictionary, with keys like "all_data" containing all 
                    position data
        params : parameters for behavior criteria
        CSVcolumns : CSV column parameters
    Outputs:
        arrays of all frames in which the various behaviors are found:
            perp_noneSee, perp_oneSees, 
            perp_bothSee, contact_any, contact_head_body, 
            contact_larger_fish_head, contact_smaller_fish_head,
            contact_inferred, tail_rubbing_frames

    """
    
    # Timer
    t1_start = perf_counter()

    # Arrays of head, body positions; angles. 
    # Last dimension = fish (so arrays are Nframes x {1 or 2}, Nfish==2)
    head_pos_data = dataset["all_data"][:,CSVcolumns["head_column_x"]:CSVcolumns["head_column_y"]+1, :]
        # head_pos_data is Nframes x 2 (x and y positions) x 2 (Nfish) array of head positions
    angle_data = dataset["heading_angle"]
    # body_x and _y are the body positions, each of size Nframes x 10 x 2 (fish)
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
        
    t1_2 = perf_counter()
    print(f'   t1_2 start 90degree analysis: {t1_2 - t1_start:.2f} seconds')
    # 90-degrees 
    orientation_dict = get_90_deg_frames(head_pos_data, angle_data, 
                                         dataset["closest_distance_mm"].flatten(),
                                         params["perp_windowsize"], 
                                         params["cos_theta_90_thresh"], 
                                         params["perp_maxDistance_mm"],
                                         params["cosSeeingAngle"], 
                                         dataset["fish_length_array_mm"])
    perp_noneSee = orientation_dict["noneSee"]
    perp_oneSees = orientation_dict["oneSees"]
    perp_bothSee = orientation_dict["bothSee"]
    perp_larger_fish_sees = orientation_dict["larger_fish_sees"]
    perp_smaller_fish_sees = orientation_dict["smaller_fish_sees"]
 
    t1_3 = perf_counter()
    print(f'   t1_3 start contact analysis: {t1_3 - t1_start:.2f} seconds')
    # Any contact, or head-body contact
    contact_inferred_distance_threshold_px = params["contact_inferred_distance_threshold_mm"]*1000/dataset["image_scale"]
    contact_dict = get_contact_frames(body_x, body_y, dataset["closest_distance_mm"],
                                params["contact_inferred_distance_threshold_mm"], 
                                dataset["image_scale"],
                                dataset["fish_length_array_mm"])
    contact_any = contact_dict["any_contact"]
    contact_head_body = contact_dict["head-body"]
    contact_larger_fish_head = contact_dict["larger_fish_head_contact"]
    contact_smaller_fish_head = contact_dict["smaller_fish_head_contact"]
    contact_inferred_frames = get_inferred_contact_frames(dataset,
                        params["contact_inferred_window"],                                  
                        contact_inferred_distance_threshold_px)
    # Include inferred contact frames in "any" contact.
    contact_any = np.unique(np.concatenate((contact_any, 
                                            contact_inferred_frames),0))

    t1_4 = perf_counter()
    print(f'   t1_4 start tail-rubbing analysis: {t1_4 - t1_start:.2f} seconds')
    # Tail-rubbing
    tailrub_maxTailDist_px = params["tailrub_maxTailDist_mm"]*1000/dataset["image_scale"]
    tail_rubbing_frames = get_tail_rubbing_frames(body_x, body_y, 
                                          dataset["head_head_distance_mm"], 
                                          angle_data, 
                                          params["tail_rub_ws"], 
                                          tailrub_maxTailDist_px, 
                                          params["cos_theta_antipar"], 
                                          params["tailrub_maxHeadDist_mm"])

    t1_5 = perf_counter()
    print(f'   t1_5 start approaching / fleeing analysis: {t1_5 - t1_start:.2f} seconds')
    # Approaching or fleeing
    (approaching_frames, fleeing_frames) = get_approach_flee_frames(dataset, 
                                                CSVcolumns, 
                                                speed_threshold_mm_s = params["approach_speed_threshold_mm_second"],
                                                min_frame_duration = params["approach_min_frame_duration"],
                                                cos_angle_thresh = params["approach_cos_angle_thresh"])


    t1_end = perf_counter()
    print(f'   t1_end end analysis: {t1_end - t1_start:.2f} seconds')

    # removed "circling_wfs," from the list

    return perp_noneSee, perp_oneSees, \
        perp_bothSee, perp_larger_fish_sees, \
        perp_smaller_fish_sees, \
        contact_any, contact_head_body, contact_larger_fish_head, \
        contact_smaller_fish_head, contact_inferred_frames, \
        tail_rubbing_frames, approaching_frames, fleeing_frames

    


def get_contact_frames(body_x, body_y, closest_distance_mm, 
                       contact_distance_threshold_mm, 
                       image_scale, fish_length_array):
    """
    Returns a dictionary of window frames for different 
    contact between two fish: any body positions, or head-body contact
    
    Assumes frames are contiguous, as should have been checked earlier.

    Args:
        body_x (array): a 3D array (Nframes x 10 x 2 fish) of x positions along the 10 body markers. (px)
        body_y (array): a 3D array (Nframes x 10 x 2 fish) of y positions along the 10 body markers. (px)
        closest_distance_mm (array) : 1D array of closest distance between fish (mm)
        contact_distance_threshold_mm: the contact distance threshold, *mm*
        image_scale : um/px, from datasets[j]["image_scale"]
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
        if closest_distance_mm[idx] < contact_distance_threshold_mm:
            contact_dict["any_contact"].append(idx+1)
            
        # Head-body contact
        d_head1_body2 = np.sqrt((body_x[idx,0,0] - body_x[idx,:,1])**2 + 
                                (body_y[idx,0,0] - body_y[idx,:,1])**2)
        d_head2_body1 = np.sqrt((body_x[idx,0,1] - body_x[idx,:,0])**2 + 
                                (body_y[idx,0,1] - body_y[idx,:,0])**2)
        fish1_hb_contact = np.min(d_head1_body2) < contact_distance_threshold_mm*1000/image_scale
        fish2_hb_contact = np.min(d_head2_body1) < contact_distance_threshold_mm*1000/image_scale
        
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



def get_90_deg_frames(fish_head_pos, fish_angle_data, 
                      closest_distance_mm, window_size, 
                      cos_theta_90_thresh, perp_maxDistance_mm, cosSeeingAngle, 
                      fish_length_array):
    """
    Returns an array of frames for 90-degree orientation events.
    Each frame represents the starting  frame for 90-degree
       orientation events that span some window (parameter window_size).

    Args:
        fish_head_pos (array): a 3D array of (x, y) head positions for both fish.
                          Nframes x 2 [x, y] x 2 fish 
        fish_angle_data (array): a 2D array of angles; Nframes x 2 fish
        closest_distance_mm : array of closest distance, mm, between fish (Nframes, 1)
        window_size (int)      : window size for which condition is averaged over.
        cos_theta_90_thresh (float): the cosine(angle) threshold for 90-degree orientation.
        perp_maxDistance_mm (float): inter-fish distance threshold, mm
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
                
    # head-to-head distance vector for all frames
    dh_vec = fish_head_pos[:,:,1] - fish_head_pos[:,:,0]  # also used later, for the connecting vector
    
    # closeness criterion
    separation_criterion = (closest_distance_mm < perp_maxDistance_mm)
    
    # All criteria (and), in each frame
    all_criteria_frame = np.logical_and(cos_theta_criterion, 
                                        separation_criterion)

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
    Nframes = np.shape(dataset["heading_angle"])[0] 
    
    # Cross-correlation of angles over a sliding window
    # (Method of sliding sums from
    #    https://stackoverflow.com/questions/12709853/python-running-cumulative-sum-with-a-given-window)
    # Should make this a function...
    # First unwrap to avoid large jumps
    angles_u1 = np.unwrap(dataset["heading_angle"][:,0], axis=0).flatten()
    angles_u2 = np.unwrap(dataset["heading_angle"][:,1], axis=0)
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
        plt.plot(range(Nframes), dataset["heading_angle"][:,0]*180/np.pi, color='magenta', label='Fish 1')
        plt.plot(range(Nframes), dataset["heading_angle"][:,1]*180/np.pi, color='olivedrab', label='Fish 2')
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
    angle_data = dataset["heading_angle"]
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
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        CSVcolumns : information on what the columns of dataset["all_data"] are
    Output : 
        relative_orientation : numpy array Nframes x Nfish==2 of 
            relative orientation (phi), radians, for fish 0 and fish 1
    """
    # All heading angles
    angle_data = dataset["heading_angle"]
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


def get_relative_heading_angle(dataset, CSVcolumns):
    """ 
    Calculate the difference in heading angle between the two fish 
    (in range [0, pi]).
    Requires Nfish = 2 (verified)
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        CSVcolumns : information on what the columns of dataset["all_data"] are
    Output : 
        relative_heading_angle : numpy array of shape (Nframes,), radians
    """
    if dataset["Nfish"] != 2:
        raise ValueError("Error for relative heading angle: Nfish must be 2")

    # All heading angles
    angle_data = dataset["heading_angle"]
    
    relative_heading_angle = np.abs(wrap_to_pi(angle_data[:,1]-angle_data[:,0]))
    
    return relative_heading_angle


def make_pair_fish_plots(datasets, outputFileNameBase = 'pair_fish',
                           outputFileNameExt = 'png'):
    """
    makes several useful "pair" plots -- i.e. plots of characteristics 
    of pairs of fish.
    Note that there are lots of parameter values that are hard-coded; this
    function is probably more useful to read than to run, pasting and 
    modifying its code.
    
    Inputs:
        datasets : dictionaries for each dataset
        outputFileNameBase : base file name for figure output; if None,
                             won't save a figure file
        outputFileNameExt : extension for figure output (e.g. 'eps' or 'png')

    Outputs:

    """
    
    verifyPairs = True
    for j in range(len(datasets)):
        if datasets[j]["Nfish"] != 2:
            verifyPairs = False
    if verifyPairs==False:
        raise ValueError('Error in make_pair_fish_plots; Nfish must be 2 !')


    # head-head distance histogram
    head_head_mm_all = combine_all_values_constrained(datasets, 
                                                     keyName='head_head_distance_mm', 
                                                     dilate_plus1 = False)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_distance_head_head' + '.' + outputFileNameExt
    else:
        outputFileName = None
    plot_probability_distr(head_head_mm_all, bin_width = 0.5, 
                           bin_range = [0, None], yScaleType = 'linear',
                           xlabelStr = 'Head-head distance (mm)', 
                           titleStr = 'Probability distribution: head-head distance (mm)',
                           outputFileName = outputFileName)

    # closest distance histogram
    closest_distance_mm_all = combine_all_values_constrained(datasets, 
                                                     keyName='closest_distance_mm', 
                                                     dilate_plus1 = False)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_distance_closest' + '.' + outputFileNameExt
    else:
        outputFileName = None
    plot_probability_distr(closest_distance_mm_all, bin_width = 0.5, 
                           bin_range = [0, None], yScaleType = 'linear',
                           xlabelStr = 'Closest distance (mm)', 
                           titleStr = 'Probability distribution: closest distance (mm)',
                           outputFileName = outputFileName)

    # Relative heading angle histogram
    relative_heading_angle_all = combine_all_values_constrained(datasets, 
                                                 keyName='relative_heading_angle', 
                                                 dilate_plus1 = False)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_rel_heading_angle' + '.' + outputFileNameExt
    else:
        outputFileName = None
    bin_width = np.pi/30
    plot_probability_distr(relative_heading_angle_all, bin_width = bin_width,
                           bin_range=[None, None], yScaleType = 'linear',
                           polarPlot = True,
                           titleStr = 'Relative Heading Angle',
                           ylim = (0, 0.6),
                           outputFileName = outputFileName)

    # Relative orientation angle histogram
    relative_orientation_angle_all = combine_all_values_constrained(datasets, 
                                                 keyName='relative_orientation', 
                                                 dilate_plus1 = False)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_rel_orientation' + '.' + outputFileNameExt
    else:
        outputFileName = None
    bin_width = np.pi/30
    plot_probability_distr(relative_orientation_angle_all, bin_width = bin_width,
                           bin_range=[None, None], yScaleType = 'linear',
                           polarPlot = True,
                           titleStr = 'Relative Orientation Angle',
                           ylim = (0, 0.6),
                           outputFileName = outputFileName)

    # 2D histogram of speed and head-head distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_speed_distance_2D' + '.' + outputFileNameExt
    else:
        outputFileName = None
    make_2D_histogram(datasets, keyNames = ('speed_array_mm_s', 
                                            'head_head_distance_mm'), 
                          keyIdx = (None, None), 
                          dilate_plus1=True, bin_ranges=None, Nbins=(20,20),
                          titleStr = 'speed and hh distance', 
                          colorRange = (0, 0.01),
                          outputFileName = outputFileName)


    # 2D histogram of heading alignment and head-head distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_heading_distance_2D' + '.' + outputFileNameExt
    else:
        outputFileName = None
    make_2D_histogram(datasets, keyNames = ('relative_heading_angle', 
                                            'head_head_distance_mm'), 
                          keyIdx = (None, None), 
                          dilate_plus1=True, bin_ranges=None, Nbins=(20,20),
                          titleStr = 'heading angle and hh distance', outputFileName = outputFileName)

    # 2D histogram of relative orientation and head-head distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_orientation_distance_2D' + '.' + outputFileNameExt
    else:
        outputFileName = None
    make_2D_histogram(datasets, keyNames = ('relative_orientation', 
                                            'head_head_distance_mm'), 
                          keyIdx = (None, None), 
                          dilate_plus1=True, bin_ranges=None, Nbins=(20,20),
                          titleStr = 'orientation angle and hh distance', outputFileName = outputFileName)


    # Speed of the "other" fish vs. time for bouts
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_boutSpeed_other' + '.' + outputFileNameExt
    else:
        outputFileName = None
    average_bout_trajectory_allSets(datasets, keyName = "speed_array_mm_s", 
                                    keyIdx = 'other', t_range_s=(-1.0, 2.0), 
                                    titleStr = 'Bout Speed, other fish', makePlot=True,
                                    outputFileName = outputFileName)

    # Speed vs. time for bouts, distance constraint
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_boutSpeed_close' + '.' + outputFileNameExt
    else:
        outputFileName = None
    max_d = 5.0 # mm, for constraint
    average_bout_trajectory_allSets(datasets, keyName = "speed_array_mm_s", 
                                    keyIdx = None, t_range_s=(-1.0, 2.0), 
                                    constraintKey='head_head_distance_mm', 
                                    constraintRange=(0, 5.0), 
                                    titleStr = f'Bout Speed, d < {max_d:.1f} mm', 
                                    makePlot=True,
                                    outputFileName = outputFileName)
    
    # Speed cross-correlation function
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_speedCrosscorr' + '.' + outputFileNameExt
    else:
        outputFileName = None
    speed_cc_all, t_lag = \
        calculate_value_corr_all(datasets, keyName = 'speed_array_mm_s',
                                 corr_type='cross', dilate_plus1 = True, 
                                 t_max = 3.0, t_window = 10.0, fpstol = 1e-6)
    plot_function_allSets(speed_cc_all, t_lag, xlabelStr='time (s)', 
                          ylabelStr='Speed Cross-correlation', 
                          titleStr='Speed Cross-correlation', 
                          average_in_dataset = True,
                          outputFileName = outputFileName)
    
    # 2D histogram of C- and J-bend frequencies (combined) vs head-head distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_CJ_distance_2D' + '.' + outputFileNameExt
    else:
        outputFileName = None
    keyList = ['Cbend_any', 'Jbend_any']
    datasets = behaviorFrameCount_all(datasets, keyList, 'CJcombined')
    make_2D_histogram(datasets, keyNames = ('head_head_distance_mm', 
                      'CJcombined'), Nbins=(15,10), 
                      constraintKey='CJcombined', constraintRange=(0.5,100), 
                      colorRange=(0, 0.001), outputFileName = outputFileName)

    