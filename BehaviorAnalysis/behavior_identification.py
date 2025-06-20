# -*- coding: utf-8 -*-
# behavior_identification.py
"""
Author:   Raghuveer Parthasarathy
Version ='2.0': 
First versions created By  : Estelle Trieu, 5/26/2022
Major modifications by Raghuveer Parthasarathy, May-July 2023
Last modified June 20, 2025 -- Raghu Parthasarathy

Description
-----------

Module containing all zebrafish pair behavior identification functions:
    - extract_pair_behaviors(), which calls all the other functions
    - Contact
    - 90-degree orientation
    - Tail rubbing
    - Maintaining proximity
    - (and more)

"""
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np
from toolkit import wrap_to_pi, combine_all_values_constrained, \
    plot_probability_distr, make_2D_histogram,calculate_value_corr_all, \
    plot_function_allSets, behaviorFrameCount_all, make_frames_dictionary
from behavior_identification_single import average_bout_trajectory_allSets
from scipy.stats import skew
import itertools
from scipy.ndimage import binary_closing, binary_opening
# from circle_fit_taubin import TaubinSVD


def get_basic_two_fish_characterizations(all_position_data, datasets, CSVcolumns,
                                             expt_config, params):
    """
    For each dataset, perform “basic” two-fish characterizations  
        (e.g. inter-fish distance, head-head vector, relative orientation, 
         relative heading alignment)
    
    Inputs:
        all_position_data : basic position information for all datasets, list of numpy arrays
        datasets : all datasets, list of dictionaries 
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
            get_interfish_distance(all_position_data[j], CSVcolumns,
                                   datasets[j]["image_scale"])

        # Get the inter-fish head-hed vector (Fish 0 - Fish 1) in 
        # each frame (Nframes x 2 array). Units = px
        datasets[j]["head_head_vec_px"] = \
            calc_head_head_vector(all_position_data[j], CSVcolumns)

        # Get frames in which the inter-fish distance is small (i.e. less
        # than the threshold value of the "proximity_threshold_mm" parameter)
        close_pair_frames =  get_close_pair_frames(datasets[j]["closest_distance_mm"],
                                                  datasets[j]["frameArray"], 
                                                  proximity_threshold_mm = params["proximity_threshold_mm"])
        # make a dictionary containing frames, removing frames with 
        # "bad" elements. Also makes a 2xN array of initial frames and durations, 
        # as usual for these behavior dictionaries
        badTrackFrames = np.array(datasets[j]["bad_bodyTrack_frames"]["raw_frames"]).astype(int)
        datasets[j]["close_pair"] = make_frames_dictionary(close_pair_frames,
                                      badTrackFrames,
                                      behavior_name = "close_pair",
                                      Nframes=datasets[j]['Nframes'])
        # Fraction of time that the pairs are close (excluding bad tracking)
        datasets[j]["close_pair_fraction"] = len(datasets[j]["close_pair"]["edit_frames"]) \
                                             / (datasets[j]['Nframes'] - 
                                                len(datasets[j]["bad_bodyTrack_frames"]["raw_frames"]))

        # Relative orientation of fish (angle between heading and
        # connecting vector). Nframes x Nfish==2 array; radians
        datasets[j]["relative_orientation"] = \
            get_relative_orientation(all_position_data[j], datasets[j], CSVcolumns)   

        # Sum of relative orientation (Calculate for any Nfish, though only
        # meaningful for two.)
        datasets[j]["relative_orientation_sum"] = \
            np.sum(datasets[j]["relative_orientation"], axis=1)  
        
        # Relative the difference in heading angle between the two fish,
        # range [0, pi]). Nframes x 1 array; radians
        datasets[j]["relative_heading_angle"] = \
            get_relative_heading_angle(datasets[j], CSVcolumns)   

        # Get the sliding window cross-correlation of heading angles
        datasets[j]["xcorr_array"] = \
            calcOrientationXCorr(datasets[j], params["angle_xcorr_windowsize"])
        
    # For each dataset, exclude bad tracking frames from calculations of
    # - the mean and std. absolute difference in fish length
    # - the mean inter-fish distance
    # exclude bad tracking frames from the 
    #    calculation of the mean angle-heading cross-correlation
    #    if the bad frames occur anywhere in the sliding window
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


    
def get_interfish_distance(position_data, CSVcolumns, image_scale):
    """
    Get the inter-fish distance (calculated both as the distance 
        between head positions and as the closest distance)
        in each frame 
    Input:
        position_data : all position data for this dataset, from all_position_data[j]
        CSVcolumns : CSV column information (dictionary)
        image_scale : scale, um/px; from dataset["image_scale"]
    Output
        head_head_distance_mm : head-head distance (mm), Nframes x 1 array 
        closest_distance_mm : closest distance (mm), Nframes x 1 array
    """
    
    # head-head distance
    head_x = position_data[:,CSVcolumns["head_column_x"],:] # x, both fish
    head_y = position_data[:,CSVcolumns["head_column_y"],:] # y, both fish
    dx = np.diff(head_x)
    dy = np.diff(head_y)
    # distance, mm
    head_head_distance_mm = (np.sqrt(dx**2 + dy**2))*image_scale/1000.0
    
    # body-body distance, for all pairs of points
    body_x = position_data[:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = position_data[:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
    closest_distance_mm = np.zeros((body_x.shape[0],1))
    for idx in range(body_x.shape[0]):
        d0 = np.subtract.outer(body_x[idx,:,0], body_x[idx,:,1]) # all pairs of subtracted x positions
        d1 = np.subtract.outer(body_y[idx,:,0], body_y[idx,:,1]) # all pairs of subtracted y positions
        d = np.sqrt(d0**2 + d1**2) # Euclidean distance matrix, all points
        closest_distance_mm[idx] = np.min(d)*image_scale/1000.0 # mm
    
    return head_head_distance_mm, closest_distance_mm



def get_close_pair_frames(distance_mm, frameArray, proximity_threshold_mm):
    """ 
    Find frames in which the fish are close to each other (closest distance <
        proximity_threshold_mm)
    Note that inter-fish distance has previously been calculated 
    Inputs:
        distance_mm: inter-fish distance array , mm
                    (either dataset["closest_distance_mm"]
                     or dataset["head_head_distance_mm"]: the inter-fish distance calculated as the closest distance between any inter-fish positions in each frame; mm; array of length Nframes)
        frameArray : Array of frames for this dataset
        proximity_threshold_mm : proximity threshold, mm
    Output : 
        close_pair_frames : numpy array of frames in which inter-fish 
                            distance is < threshold
    """

    # Check that frameArray and distance have the same size
    if len(distance_mm) != len(frameArray):
        raise ValueError("Error: inter-fish distance and frameArray don't have the same size.")

    close_pair_frames = frameArray.flatten()[distance_mm.flatten() < 
                                             proximity_threshold_mm]
    return close_pair_frames


    
def extract_pair_behaviors(pair_behavior_frames, position_data, dataset, 
                           params, CSVcolumns): 
    """
    For a single dataset, calls functions to identify frames corresponding 
    to each two-fish behavioral motif.
    
    Inputs:
        pair_behavior_frames : dictionary in which each key is a behavior,
                    initialized with values all being empty integer arrays
        position_data : position data for this dataset, presumably all_position_data[j]
        dataset : dictionary with all dataset info, presumably datasets[j]
        params : parameters for behavior criteria
        CSVcolumns : CSV column parameters
    Outputs:
        pair_behavior_frames : dictionary in which each key is a behavior string
            and each value is a numpy array of frames identified for that behavior:
            Keys defined outside this function; should be:
                perp_noneSee, perp_oneSees, 
                perp_bothSee, contact_any, contact_head_body, 
                contact_larger_fish_head, contact_smaller_fish_head,
                contact_inferred, tail_rubbing, maintain_proximity, 
                approaching_Fish0, approaching_Fish1, approaching_any, 
                approaching_all,
                fleeing_Fish0, fleeing_Fish1, fleeing_any, fleeing_all
            
    """
    
    # Timer
    t1_start = perf_counter()

    # Arrays of head, body positions abd angles, which are used by multiple 
    # functions.
    # Last dimension = fish (so array shapes are Nframes x {1 or 2}, Nfish==2)
    head_pos_data = position_data[:,CSVcolumns["head_column_x"]:CSVcolumns["head_column_y"]+1, :]
        # head_pos_data is Nframes x 2 (x and y positions) x 2 (Nfish) array of head positions
    angle_data = dataset["heading_angle"]
    # body_x and _y are the body positions, each of size Nframes x 10 x Nfish
    body_x = position_data[:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = position_data[:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
        
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
    pair_behavior_frames['perp_noneSee'] = orientation_dict["noneSee"]
    pair_behavior_frames['perp_oneSees'] = orientation_dict["oneSees"]
    pair_behavior_frames['perp_bothSee'] = orientation_dict["bothSee"]
    pair_behavior_frames['perp_larger_fish_sees'] = orientation_dict["larger_fish_sees"]
    pair_behavior_frames['perp_smaller_fish_sees'] = orientation_dict["smaller_fish_sees"]
 
    t1_3 = perf_counter()
    print(f'   t1_3 start contact analysis: {t1_3 - t1_start:.2f} seconds')
    # Any contact, or head-body contact
    contact_dict = get_contact_frames(position_data, dataset, CSVcolumns,
                                params["contact_distance_threshold_mm"])
    pair_behavior_frames['contact_any'] = contact_dict["any_contact"]
    pair_behavior_frames['contact_head_body'] = contact_dict["head-body"]
    pair_behavior_frames['contact_larger_fish_head'] = contact_dict["larger_fish_head_contact"]
    pair_behavior_frames['contact_smaller_fish_head'] = contact_dict["smaller_fish_head_contact"]
    pair_behavior_frames['contact_inferred'] = \
        get_inferred_contact_frames(position_data, dataset, CSVcolumns, 
                        params["contact_inferred_window"],                                  
                        params["contact_inferred_distance_threshold_mm"],
                        pair_behavior_frames['contact_any'])
    # Include inferred contact frames in "any" contact.
    pair_behavior_frames['contact_any'] = np.unique(np.concatenate((pair_behavior_frames['contact_any'], 
                                            pair_behavior_frames['contact_inferred']),0))

    t1_4 = perf_counter()
    print(f'   t1_4 start tail-rubbing analysis: {t1_4 - t1_start:.2f} seconds')
    # Tail-rubbing
    tailrub_maxTailDist_px = params["tailrub_maxTailDist_mm"]*1000/dataset["image_scale"]

    pair_behavior_frames['tail_rubbing'] = get_tail_rubbing_frames(body_x, body_y, 
                                          dataset["head_head_distance_mm"], 
                                          angle_data, 
                                          params["tail_rub_ws"], 
                                          tailrub_maxTailDist_px, 
                                          params["cos_theta_antipar"], 
                                          params["tailrub_maxHeadDist_mm"])

    t1_5 = perf_counter()
    print(f'   t1_5 start approaching / fleeing analysis: {t1_5 - t1_start:.2f} seconds')
    # Approaching or fleeing
    (approaching_frames, fleeing_frames, approaching_frames_any, \
        approaching_frames_all, fleeing_frames_any, fleeing_frames_all) = \
             get_approach_flee_frames(position_data, dataset, 
                                      CSVcolumns, 
                                      speed_threshold_mm_s = params["approach_speed_threshold_mm_second"],
                                      min_frame_duration = params["approach_min_frame_duration"],
                                      cos_angle_thresh = params["approach_cos_angle_thresh"])
    pair_behavior_frames['approaching_Fish0'] = approaching_frames[0]
    pair_behavior_frames['approaching_Fish1'] = approaching_frames[1]
    pair_behavior_frames['approaching_any'] = approaching_frames_any
    pair_behavior_frames['approaching_all'] = approaching_frames_all
    pair_behavior_frames['fleeing_Fish0'] = fleeing_frames[0]
    pair_behavior_frames['fleeing_Fish1'] = fleeing_frames[1]
    pair_behavior_frames['fleeing_any'] = fleeing_frames_any
    pair_behavior_frames['fleeing_all'] = fleeing_frames_all

    t1_6 = perf_counter()
    print(f'   t1_6 start maintaining proximity analysis: {t1_6 - t1_start:.2f} seconds')
    # Maintaining proximity
    pair_behavior_frames['maintain_proximity'] = \
        get_maintain_proximity_frames(position_data, dataset, CSVcolumns, params)

    
    t1_end = perf_counter()
    print(f'   t1_end end analysis: {t1_end - t1_start:.2f} seconds')

    # removed "circling_wfs," from the list

    return pair_behavior_frames


                            
def get_contact_frames(position_data, dataset, CSVcolumns, contact_distance_threshold_mm):
    """
    Returns a dictionary of window frames for contact between two fish, 
    which can be close distance between any fish body positions or 
    head-body contact
    
    Assumes frames are contiguous, as should have been checked earlier.

    Args:
        position_data : position data for this dataset, presumably all_position_data[j]
        dataset : dictionary with all dataset info, including head-head
                  distance (mm), dataset["closest_distance_mm"],
                  dataset["image_scale"], 
                  dataset["fish_length_array_mm"] (used for identifying the larger fish),
                  and all positions
        CSVcolumns : CSV column parameters
        contact_distance_threshold_mm: the contact distance threshold, *mm*
        image_scale : um/px, from datasets[j]["image_scale"]

    Returns:
        contact_dict (dictionary): a dictionary of arrays of different 
            contact types: any_contact, head-body contact (a subset)
            larger or smaller fish head contact (a subset)
    """
    
    # body_x and _y are the body positions, each of size Nframes x 10 x Nfish
    body_x = position_data[:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = position_data[:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
    
    contact_dict = {"any_contact": [], "head-body": [], 
                    "larger_fish_head_contact": [], 
                    "smaller_fish_head_contact": []}

    for idx in range(body_x.shape[0]):

        # Any contact: look at closest element of distance matrix between
        # body points, previously calculated
        if dataset["closest_distance_mm"][idx] < contact_distance_threshold_mm:
            contact_dict["any_contact"].append(idx+1)
        
        # Head-body contact
        d_head1_body2 = np.sqrt((body_x[idx,0,0] - body_x[idx,:,1])**2 + 
                                (body_y[idx,0,0] - body_y[idx,:,1])**2)
        d_head2_body1 = np.sqrt((body_x[idx,0,1] - body_x[idx,:,0])**2 + 
                                (body_y[idx,0,1] - body_y[idx,:,0])**2)
        fish1_hb_contact = np.min(d_head1_body2) < contact_distance_threshold_mm*1000/dataset["image_scale"]
        fish2_hb_contact = np.min(d_head2_body1) < contact_distance_threshold_mm*1000/dataset["image_scale"]
        
        if fish1_hb_contact or fish2_hb_contact:
            contact_dict["head-body"].append(idx+1)
            # Note that "any contact" will be automatically satisfied.
            if not (fish1_hb_contact and fish2_hb_contact):
                # Only one is making head-body contact
                largerFishIdx = np.argmax(dataset["fish_length_array_mm"][idx,:])
                if largerFishIdx==0 and fish1_hb_contact:
                    contact_dict["larger_fish_head_contact"].append(idx+1)
                else:
                    contact_dict["smaller_fish_head_contact"].append(idx+1)

    return contact_dict


def get_inferred_contact_frames(position_data, dataset, CSVcolumns, frameWindow, 
                                contact_inferred_distance_threshold_mm,
                                contact_any):
    """
    Returns an array of frames corresponding to inferred contact.
    This can be either of the following:
    (i) tracking is bad (zeros values) but inter-fish distance was decreasing
       over some number of preceding frames and was below the contact
       threshold immediately before the bad tracking.
    (ii) frames with bad tracking in which fish are close 
       (separation < contact_distance_threshold) before and after the bad tracking 
       frames, and in which fish have not moved much during this period, 
       accounting for possibly switched Track IDs 
       (total distance < Nfish*contact_inferred_distance_threshold).
    If there are no bad frames, return an empty array
    
    Args:
        position_data : position data for this dataset, presumably all_position_data[j]
        dataset : dictionary with all dataset info, including head-head
                  distance (mm)
        CSVcolumns : CSV column parameters
        frameWindow : number of preceding frames to examine
        contact_inferred_distance_threshold_mm: the inferred contact distance threshold, *mm*
        contact_any : list of frames in which there is a contact event (previously calculated)

    Returns:
        inf_contact_frames: 1D of array of frames with inferred contact. 
           These are the frames immediately preceding the start of a bad
           tracking run.
           
    To do:
        Make inferred 1 and inferred 2 into separate functions
    """
    
    # body positions
    body_x = position_data[:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = position_data[:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]

    # shouldn't need to be a set, but can't hurt.
    badFrames_firstFrames = set(dataset["bad_bodyTrack_frames"]["combine_frames"][0,:])
    
    ## Method (i)
    inf_contact_frames_1 = []
    # Consider the start of each run of bad frames.
    for badFrame in badFrames_firstFrames:
        precedingFrames = np.arange(badFrame-frameWindow,badFrame)
        # check that these frames exist, and aren't part of other runs of bad Frames
        okFrames = np.all(np.isin(precedingFrames, dataset["frameArray"])) and \
                   not(np.any(np.isin(precedingFrames, dataset["bad_bodyTrack_frames"]["raw_frames"])))
        if okFrames:
            # note need to switch order in np.isin()
            this_distance = np.array(dataset["head_head_distance_mm"]\
                                     [np.where(np.isin(dataset["frameArray"], 
                                                       precedingFrames))])
            # Check decreasing, and check that last element is within threshold
            if (this_distance[-1]<contact_inferred_distance_threshold_mm) and \
                    np.all(this_distance[:-1] >= this_distance[1:]):
                inf_contact_frames_1.append(precedingFrames[-1])

    ## Method (ii)
    # shouldn't need to be a set, but can't hurt.
    badFrames_allFrames = set(dataset["bad_bodyTrack_frames"]["edit_frames"])

    # If there are no bad frames, return the empty array. FUNCTION ENDS!
    if not badFrames_allFrames:
        return np.array(inf_contact_frames_1)

    contact_inferred_distance_threshold_px = contact_inferred_distance_threshold_mm*1000/dataset["image_scale"]
    inf_contact_frames_2 = []
    Nfish = body_x.shape[2]

    # Find contiguous blocks of badFrames
    contact_set = set(contact_any)
    blocks = []
    current_block = []
    
    minFrame = int(max(1, min(badFrames_allFrames, default = 1)))
    maxFrame = int(min(body_x.shape[0]+1, max(badFrames_allFrames)))
    for f in range(minFrame, maxFrame):
        # print('testing f: ', f)
        if f in badFrames_allFrames:
            # If current frame is a badFrame
            if not current_block:
                # Start a new block if it's empty
                current_block = [f]
            else:
                # Extend current block if previous frame was also in badFrame
                if f == current_block[-1] + 1:
                    current_block.append(f)
        else:
            # If current frame is not a badFrame
            if current_block:
                # Check if the block is bounded by contact frames
                if (current_block[0]-1 in contact_set) and (current_block[-1]+1 in contact_set):
                    blocks.append(current_block)
                # Reset current block
                current_block = []
    
    # Step 2: Refine inferred contacts
    refined_contacts = []
    if blocks:
        refined_contacts = []
        for block in blocks:
            # Find frames immediately before and after the block
            before_frame = block[0] - 1
            after_frame = block[-1] + 1
            
            # Check if these frames are valid
            if before_frame < 0 or after_frame >= body_x.shape[0]:
                continue
            
            # Get the indices of before_frame and after_frame in dataset["frameArray"]
            before_frame_idx = np.where(dataset["frameArray"] == before_frame)[0][0]
            after_frame_idx = np.where(dataset["frameArray"] == after_frame)[0][0]

            # Calculate head distances for each fish pair
            closest_distances = []
            min_total_distance = np.inf
            best_mapping = None
            
            # Generate all possible one-to-one mappings
            for mapping in itertools.permutations(range(Nfish)):
                total_distance = 0
                distances = []
                for j in range(Nfish):
                    head_before_j = (body_x[before_frame_idx, 0, j], 
                                     body_y[before_frame_idx, 0, j])
                    head_after_k = (body_x[after_frame_idx, 0, mapping[j]], 
                                    body_y[after_frame_idx, 0, mapping[j]])
                    dist = np.sqrt(
                        (head_before_j[0] - head_after_k[0])**2 + 
                        (head_before_j[1] - head_after_k[1])**2
                    )
                    distances.append(dist)
                    total_distance += dist
                
                # Check if this mapping has the minimal total distance
                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    best_mapping = distances
            
            closest_distances = best_mapping
            
            # Check if total closest distances exceed threshold
            if np.sum(closest_distances) <= (Nfish * contact_inferred_distance_threshold_px):
                refined_contacts.extend(block)
                
        inf_contact_frames_2 = np.array(sorted(refined_contacts), dtype=int)

    ## Combine, and remove duplication of frames
    if len(inf_contact_frames_2) > 0:
        inf_contact_frames = np.concatenate((inf_contact_frames_1, 
                                             inf_contact_frames_2), axis=0)
    else:
        inf_contact_frames = inf_contact_frames_1
            
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


def get_maintain_proximity_frames(position_data, dataset, CSVcolumns, params):
    """
    Returns an array of frames for "maintaining proximity" events, in which fish
    maintain proximity while moving, over some duration

    Args:
        position_data : basic position information for this dataset, numpy array
        dataset : dataset dictionary of all behavior information for a given expt.
            contains "heading_angle"
        CSVcolumns: information on what the columns of position_data are
        params : dictionary of all analysis parameters -- will use speed 
            and proximity thresholds
        max_gap_s : maximum gap in matching criterion to allow (s). At 25 fps,
            0.08 s = 2 frames.
        min_duration_s : min duration that matching criteria must be met for the
            behavior to be recorded. At 25 fps, 0.6 s = 15 frames  

    Returns:
        maintain_proximity_frames: a 1D array of frames in which the
            maintaining proximity conditions are met.

    """
    
    # Criteria, evaluated in each frame.
    speed_criterion = np.any(dataset["speed_array_mm_s"] > 
                             params["motion_speed_threshold_mm_second"], 
                             axis=1)
    # Closing to remove small gaps
    N_smallgap = np.round(params["max_motion_gap_s"]*dataset["fps"]).astype(int)
    ste_smallgap = np.ones((N_smallgap+1,), dtype=bool)
    speed_criterion_closed = binary_closing(speed_criterion, ste_smallgap)

    distance_criterion = dataset["closest_distance_mm"].flatten() < \
        params["proximity_threshold_mm"]
    all_criteria_0 = speed_criterion_closed & distance_criterion

    # Opening to enforce min. duration
    N_duration = np.round(params["min_proximity_duration_s"]*dataset["fps"]).astype(int)
    ste_duration = np.ones((N_duration+1,), dtype=bool)
    all_criteria = binary_opening(all_criteria_0, ste_duration)

    startFrame = np.min(dataset["frameArray"])
    maintain_proximity_frames = np.array(np.where(all_criteria==True))[0,:].flatten() + startFrame
    # Not sure why the [0,:] is needed, but otherwise returns additional zeros.
    
    plot_diagnostic = False
    if plot_diagnostic:
        # Figures
        xlim = (1300, 1800)
        frames = np.arange(dataset["Nframes"])
        
        Nplots = 3
        fig, axs = plt.subplots(Nplots, 1, figsize=(20, 14), sharex=True)
        
        fno = 0
        # Plot 1: speed
        axs[fno].plot(frames, dataset["speed_array_mm_s"][:, 0])
        axs[fno].plot(frames, dataset["speed_array_mm_s"][:, 1])
        axs[fno].plot(frames, np.ones_like(frames) * params["motion_speed_threshold_mm_second"],
                      linestyle='dotted', color='gray')
        axs[fno].set_ylabel('Speed (mm/s)')
        axs[fno].set_title(f'{dataset["dataset_name"]}: Speed', fontsize = 20)
        axs[fno].set_xlim(xlim)
        
    
        # Plot 4: Distances
        fno = fno + 1
        axs[fno].plot(frames, dataset["head_head_distance_mm"], color='peru', label='Head-Head distance')
        axs[fno].plot(frames, dataset["closest_distance_mm"].flatten(), color='darkturquoise', label='Closest distance')
        axs[fno].plot(frames, np.ones_like(frames) * params["proximity_threshold_mm"], linestyle='dotted', color='gray')
        axs[fno].set_ylabel('distance, mm')
        axs[fno].set_title(f'{dataset["dataset_name"]}: Distances', fontsize = 20)
        axs[fno].legend()
        axs[fno].set_xlim(xlim)
        axs[fno].set_xlabel('frame', fontsize = 18)  # X-axis label added here
        
        # Plot 4: Criteria
        fno = fno + 1
        axs[fno].plot(frames, speed_criterion, color='olive', label='Speed')
        axs[fno].plot(frames, 0.95*distance_criterion, color='tomato', label='Distance')
        axs[fno].plot(frames, 1.05*all_criteria_0, color='khaki', label='All_0')
        axs[fno].plot(frames, 1.1*all_criteria, color='magenta', label='All')
        axs[fno].set_ylabel('Criteria, boolean')
        axs[fno].set_title(f'{dataset["dataset_name"]}: Criteria', fontsize = 20)
        axs[fno].legend()
        axs[fno].set_xlim(xlim)
        axs[fno].set_xlabel('frame', fontsize = 18)  # X-axis label added here
    
        plt.tight_layout()
        plt.show()

    return maintain_proximity_frames

def calcOrientationXCorr(dataset, window_size = 25, makeDiagnosticPlots = False):
    """
    Heading angle Co-orientation behavior; see July 2023 notes
    Calculate cross-correlation of fish heading angles, over a sliding window
    xcorr at frame j is the normalized cross-correlation over the window 
        *ending* at j
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
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


def get_approach_flee_frames(position_data, dataset, CSVcolumns, 
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
        position_data : position data for this dataset, presumably all_position_data[j]
        dataset: dataset dictionary of all behavior information for a given expt.
        CSVcolumns : information on what the columns of position_data are
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
        approaching_frames_any, _all : frames in which any fish, all fish are approaching
        fleeing_frames_any, _all : frames in which any fish, all fish are fleeing
    """

    # All body positions, as in C-bending function
    angle_data = dataset["heading_angle"]
    body_x = position_data[:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = position_data[:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
    
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
    
    # Approaching and fleeing for "any" and "all" fish (Boolean)
    # any fish
    approaching_any  = np.any(approaching, axis=1)
    approaching_frames_any = np.where(approaching_any)[0] + 1
    fleeing_any  = np.any(fleeing, axis=1)
    fleeing_frames_any = np.where(fleeing_any)[0] + 1
    # all fish
    approaching_all  = np.all(approaching, axis=1)
    approaching_frames_all = np.where(approaching_all)[0] + 1
    fleeing_all  = np.all(fleeing, axis=1)
    fleeing_frames_all = np.where(fleeing_all)[0] + 1

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

    return approaching_frames, fleeing_frames, approaching_frames_any, \
        approaching_frames_all, fleeing_frames_any, fleeing_frames_all
    

def calc_head_head_vector(position_data, CSVcolumns):
    """ 
    Calculate the head-to-head vector, px, in each frame.
    Valid only for Nfish==2; return None otherwise
    
    Inputs:
        position_data : position data for this dataset, presumably all_position_data[j]
        CSVcolumns : information on what the columns of position_data are
    Output : 
        dh_vec_px : vector from fish 0 to fish 1, px, in each frame.
        
        numpy array Nframes x 1 of Head[1] - Head[0]
    """
    
    # all head positions
    head_pos_data = position_data[:,CSVcolumns["head_column_x"]:CSVcolumns["head_column_y"]+1, :]
        # head_pos_data is Nframes x 2 (x and y positions) x 2 (Nfish) array of head positions
    
    if head_pos_data.shape[2]!=2:
        return None
    else:
        # head-head distance vector for all frames, px
        dh_vec_px = head_pos_data[:,:,1] - head_pos_data[:,:,0]  
        return dh_vec_px

    
def get_relative_orientation(position_data, dataset, CSVcolumns):
    """ 
    Calculate the relative orientation of each fish with respect to the
    head-to-head vector to the other fish.
    Inputs:
        position_data : position data for this dataset, presumably all_position_data[j]
        dataset: dataset dictionary of all behavior information for a given expt.
                Includes "heading_angle" from repair_heading_angles() 
        CSVcolumns : information on what the columns of position_data are
    Output : 
        relative_orientation : numpy array Nframes x Nfish==2 of 
            relative orientation (phi), radians, for fish 0 and fish 1
    """
    # All heading angles
    angle_data = dataset["heading_angle"]

    dh_vec_px = calc_head_head_vector(position_data, CSVcolumns)
    
    v0 = np.stack((np.cos(angle_data[:, 0]), 
                   np.sin(angle_data[:, 0])), axis=1)
    v1 = np.stack((np.cos(angle_data[:, 1]), 
                   np.sin(angle_data[:, 1])), axis=1)
    
    dot_product_0 = np.sum(v0 * dh_vec_px, axis=1)
    magnitude_product_0 = np.linalg.norm(v0, axis=1) * np.linalg.norm(dh_vec_px, axis=1)
    phi0 = np.arccos(dot_product_0 / magnitude_product_0)
    
    dot_product_1 = np.sum(v1 * -dh_vec_px, axis=1)
    magnitude_product_1 = np.linalg.norm(v1, axis=1) * np.linalg.norm(dh_vec_px, axis=1)
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
        CSVcolumns : information on what the columns of position_data are
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

    # Sum of relative orientation angles histogram
    relative_orientation_sum__all = combine_all_values_constrained(datasets, 
                                                 keyName='relative_orientation_sum', 
                                                 dilate_plus1 = False)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_rel_orientation_sum' + '.' + outputFileNameExt
    else:
        outputFileName = None
    bin_width = np.pi/60
    plot_probability_distr(relative_orientation_sum__all, bin_width = bin_width,
                           bin_range=[None, None], yScaleType = 'linear',
                           polarPlot = False,
                           titleStr = 'Sum of Relative Orientation Angles',
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

    