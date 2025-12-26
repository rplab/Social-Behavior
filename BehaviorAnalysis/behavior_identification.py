# -*- coding: utf-8 -*-
# behavior_identification.py
"""
Author:   Raghuveer Parthasarathy
Version ='2.0': 
First versions created By  : Estelle Trieu, 5/26/2022
Major modifications by Raghuveer Parthasarathy, May-July 2023
Last modified December 25, 2025 -- Raghu Parthasarathy

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
    calculate_value_corr_all,  make_frames_dictionary, \
    remove_frames, combine_events, calculate_value_corr_all_binned, \
    dilate_frames, get_values_subset, repair_heading_angles
from IO_toolkit import plot_probability_distr, plot_2D_heatmap, \
    plot_2Darray_linePlots, make_2D_histogram, slice_2D_histogram, \
    plot_function_allSets, plot_waterfall_binned_crosscorr, \
    calculate_property_1Dbinned, \
    get_plot_and_CSV_filenames, simple_write_CSV
from behavior_identification_single import average_bout_trajectory_allSets, \
    calc_bend_angle
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
                                                  proximity_threshold_mm = \
                                                      min(params["proximity_threshold_mm"]))
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
        # Also the fish indexes in each frame ordered by relative 
        # orientation angle, low to high,
        # Also the sum of relative orientation angles, signed and sum(abs())
        datasets[j]["relative_orientation"], \
            datasets[j]["rel_orient_rankIdx"], \
            datasets[j]["relative_orientation_sum"], \
            datasets[j]["relative_orientation_abs_sum"] = \
            get_relative_orientation(all_position_data[j], datasets[j], CSVcolumns)   

        # Relative difference in heading angle between the two fish,
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

    # Arrays of head, body positions, used by multiple functions.
    # Last dimension = fish (so array shapes are (Nframes, {1 or 2}, Nfish==2)
    # body_x and _y are the body positions, each of size Nframes x 10 x Nfish
    body_x = position_data[:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = position_data[:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
        
    t1_2 = perf_counter()
    print(f'   t1_2 start 90degree analysis: {t1_2 - t1_start:.2f} seconds')
    # 90-degrees 
    orientation_dict = get_90_deg_frames(position_data, dataset, CSVcolumns,
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
                                          dataset["heading_angle"], 
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



def get_90_deg_frames(position_data, dataset, CSVcolumns, window_size, 
                      cos_theta_90_thresh, perp_maxDistance_mm, cosSeeingAngle, 
                      fish_length_array):
    """
    Returns an array of frames for 90-degree orientation events.
    Each frame represents the starting  frame for 90-degree
       orientation events that span {window_size} frames.

    Args:
        position_data : position data for this dataset, presumably all_position_data[j]
        dataset: dataset dictionary of all behavior information for a given expt.
                probably datasets[j])
                Includes "heading_angle" from repair_heading_angles(), a 2D array of heading angles; Nframes x 2 fish
                Include "closest_distance_mm", array of closest distance, mm, between fish (Nframes, )
        CSVcolumns : information on what the columns of position_data are

        fish_head_pos (array): a 3D array of (x, y) head positions for both fish.
                          Nframes x 2 [x, y] x 2 fish 

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

    fish_angle_data = dataset["heading_angle"]
    closest_distance_mm = dataset["closest_distance_mm"].flatten()
                                     
    # cos_theta for all frames
    cos_theta = np.cos(fish_angle_data[:,0] - fish_angle_data[:,1])
    cos_theta_criterion = (np.abs(cos_theta) < cos_theta_90_thresh)
                    
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
    
    # Calculate angle between the heading angle of a fish and the vector
    # to each body position of every other fish, in each frame.
    # The functions get_relative_orientation_to_body_positions() and
    # and calc_head_body_vectors() are written generally for arbitrary numbers
    # of fish. Here, instead of making a large array for arbitrary numbers of
    # fish, I'll just calculate for fish idx = 0 and = 1 (i.e. Nfish = 2)
    # Shape: (Nframes, Nbodycolumns, Nfish-1)
    f0_angles_to_body = get_relative_orientation_to_body_positions(head_idx = 0, 
                            position_data = position_data, dataset = dataset, 
                            CSVcolumns = CSVcolumns)
    f1_angles_to_body = get_relative_orientation_to_body_positions(head_idx = 1, 
                            position_data = position_data, dataset = dataset, 
                            CSVcolumns = CSVcolumns)
                                                                      
    # For each 90 degree event, determine from the orientation type 
    # whether 0, 1, or both fish are in the "field of view" of the other.
    # Could have done this for all frames and just kept those that met 
    # the above criteria; that's probably faster, but this makes testing
    # easier.
    for idx in ninety_degree_idx:
        # Assess the angle between fish k's heading angle and each body position
        # of the other fish.
        fish0sees = np.any(np.cos(f0_angles_to_body[idx, :, 0]) 
                           >= cosSeeingAngle)
        fish1sees = np.any(np.cos(f1_angles_to_body[idx, :, 0]) 
                           >= cosSeeingAngle)

        if fish0sees or fish1sees:
            if fish0sees and fish1sees:
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
    Returns an array of frames for "maintaining proximity" events, 
    in which fish maintain proximity while moving, over some duration

    Args:
        position_data : basic position information for this dataset, numpy array
        dataset : dataset dictionary of all behavior information for a given expt.
            contains "heading_angle"
        CSVcolumns: information on what the columns of position_data are
        params : dictionary of all analysis parameters -- will use speed 
            and proximity thresholds. Includes:
            proximity_threshold_mm :list of two items, min and max range to 
                consider for proximity
            proximity_distance_measure : what distance measure to use (closest
                    or head_to_head)
            max_motion_gap_s : maximum gap in matching criterion to allow (s). 
                At 25 fps, 0.5 s = 12.5 frames.
            min_proximity_duration_s : min duration that matching criteria 
                    must be met for the behavior to be recorded (s).
                    Leave as zero (or < 1 frame) for no minimum.

    Returns:
        maintain_proximity_frames: a 1D array of frames in which the
            maintaining proximity conditions are met.

    """
    
    if params["proximity_distance_measure"] == 'closest':
        distance_key = "closest_distance_mm"
    elif params["proximity_distance_measure"] == 'head_to_head':
        distance_key = "head_head_distance_mm"
    else:
        raise ValueError('Invalid distance measure!')
            
    # Criteria, evaluated in each frame.
    speed_criterion = np.any(dataset["speed_array_mm_s"] > 
                             params["motion_speed_threshold_mm_second"], 
                             axis=1)

    # Closing to remove small gaps
    N_smallgap = np.round(params["max_motion_gap_s"]*dataset["fps"]).astype(int)
    ste_smallgap = np.ones((N_smallgap+1,), dtype=bool)
    speed_criterion_closed = binary_closing(speed_criterion, ste_smallgap)

    # Distance
    distance_criterion = (dataset[distance_key].flatten() > \
                          params["proximity_threshold_mm"][0]) & \
                         (dataset[distance_key].flatten() < \
                          params["proximity_threshold_mm"][1])
                             
    all_criteria = speed_criterion_closed & distance_criterion

    # Opening to enforce min. duration
    if params["min_proximity_duration_s"]*dataset["fps"] >= 1.0:
        # minimum of at least one frame
        N_duration = np.round(params["min_proximity_duration_s"]*dataset["fps"]).astype(int)
        ste_duration = np.ones((N_duration+1,), dtype=bool)
        all_criteria = binary_opening(all_criteria, ste_duration)

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
        
    
        # Plot 4: Criteria
        fno = fno + 1
        axs[fno].plot(frames, speed_criterion, color='olive', label='Speed')
        axs[fno].plot(frames, 0.95*distance_criterion, color='tomato', label='Distance')
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
                    numpy array, shape (Nframes, 2) of Head[1] - Head[0]
    """
    
    # all head positions
    head_pos_data = position_data[:, [CSVcolumns["head_column_x"],
                                      CSVcolumns["head_column_y"]], :]
        # head_pos_data is Nframes x 2 (x and y positions) x 2 (Nfish) 
        # array of head positions
    
    if head_pos_data.shape[2]!=2:
        return None
    else:
        # head-head distance vector for all frames, px
        dh_vec_px = head_pos_data[:,:,1] - head_pos_data[:,:,0]  
        return dh_vec_px


def calc_head_body_vectors(head_idx, position_data, CSVcolumns):
    """ 
    Calculate the vector from the head of fish head_idx to each body
    position of each other fish, in each frame.
    
    Inputs:
        head_idx : index of the fish whose head to consider. Must be in
                    range(Nfish). (Nfish determined as position_data.shape[2)
        position_data : position data for this dataset, presumably 
                    all_position_data[j]
        CSVcolumns : information on what the columns of position_data are
    Output : 
        head_body_vec_px : vector from fish head_idx to each body position
                           of each other fish "k", in each frame.
                           Body[k] - Head[head_idx]
                           Units: px
                           numpy array
                           shape (Nframes, N body columns, 2 = {x, y}, Nfish-1)
    """
    
    Nfish = position_data.shape[2]
    if head_idx not in range(Nfish):
        raise ValueError("calc_head_body_vectors: invalid index")
    
    # head position. Shape (Nframes, 2 = {x, y})
    head_pos = np.zeros((position_data.shape[0], 2), dtype=float)
    head_pos[:,0] = position_data[:, CSVcolumns["head_column_x"], head_idx]
    head_pos[:,1] = position_data[:, CSVcolumns["head_column_y"], head_idx]

    # all body positions of other fish.
    # Shape (Nframes, N body columns, 2 = {x, y}, Nfish-1)
    mask = np.ones((Nfish,), dtype=bool)
    mask[head_idx] = False
    body_pos = np.zeros((position_data.shape[0], CSVcolumns["body_Ncolumns"], 
                         2, Nfish-1), dtype=float)
    body_pos[:,:,0,:] = position_data[:, CSVcolumns["body_column_x_start"] : 
                                         (CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), 
                                         mask]
    body_pos[:,:,1,:] = position_data[:, CSVcolumns["body_column_y_start"] : 
                                         (CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), 
                                         mask]
    
    head_repeat = np.repeat(head_pos[:, np.newaxis, :], 
                            CSVcolumns["body_Ncolumns"], axis=1)
    head_repeat = np.repeat(head_repeat[:, :, :, np.newaxis], 
                            Nfish-1, axis=3)
    
    head_body_vec_px = body_pos - head_repeat
    return head_body_vec_px


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
            Returns signed angles in range [-π, π] (Oct. 4, 2025 modification).
        rel_orient_rankIdx : Fish indexes in each frame ordered by 
            relative orientation angle, low to high. If, for example, 
            this is [1, 0] for a given frame, fish 1 has the lower 
            relative orientation angle. Shape (Nframes, Nfish==2)
        relative_orientation_sum 
            Sum of relative orientation angles
        relative_orientation_abs_sum 
            Sum of absolute values of relative orientation angles
    
    Note: Valid only for Nfish==2. (Checks this.) Could expand to
        arbitrary Nfish, giving a matrix of relative orientation values.
    """
    
    if dataset["Nfish"] != 2:
        raise ValueError('Error: Relative orientation valid only for Nfish==2')

    # All heading angles
    angle_data = dataset["heading_angle"]

    dh_vec_px = calc_head_head_vector(position_data, CSVcolumns)
    
    # Unit vectors for heading directions
    v0 = np.stack((np.cos(angle_data[:, 0]), 
                   np.sin(angle_data[:, 0])), axis=1)
    v1 = np.stack((np.cos(angle_data[:, 1]), 
                   np.sin(angle_data[:, 1])), axis=1)

    # Normalize dh_vec_px
    dh_norm = np.linalg.norm(dh_vec_px, axis=1, keepdims=True)
    dh_unit = dh_vec_px / dh_norm

    # Calculate dot products for magnitude
    dot_product_0 = np.sum(v0 * dh_unit, axis=1)
    dot_product_1 = np.sum(v1 * -dh_unit, axis=1)

    # Calculate unsigned angles
    phi0_unsigned = np.arccos(dot_product_0)
    phi1_unsigned = np.arccos(dot_product_1)
    
    # Calculate cross products to determine sign (z-component for 2D)
    # cross_z = v_x * dh_y - v_y * dh_x
    cross_z_0 = v0[:, 0] * dh_unit[:, 1] - v0[:, 1] * dh_unit[:, 0]
    cross_z_1 = v1[:, 0] * (-dh_unit[:, 1]) - v1[:, 1] * (-dh_unit[:, 0])
    
    # Apply sign: positive if cross product in -z, negative if in +z
    phi0 = np.where(cross_z_0 >= 0, -phi0_unsigned, phi0_unsigned)
    phi1 = np.where(cross_z_1 >= 0, -phi1_unsigned, phi1_unsigned)
    
    relative_orientation = np.stack((phi0, phi1), axis=1)

    # Rank by absolute value of relative orientation
    rel_orient_rankIdx = np.argsort(np.abs(relative_orientation), axis=1)

    # Sum of relative orientation (Calculate for any Nfish, though only
    # meaningful for two.)
    relative_orientation_sum = np.sum(relative_orientation, axis=1)  
        
    # Sum of absolute values of relative orientation angles
    # (Calculate for any Nfish, though only meaningful for two.)
    relative_orientation_abs_sum = np.sum(np.abs(relative_orientation), axis=1)  
    
    return relative_orientation, rel_orient_rankIdx, relative_orientation_sum, \
        relative_orientation_abs_sum


def get_relative_orientation_to_body_positions(head_idx, position_data, 
                                               dataset, CSVcolumns):
    """
    Calculates the angle between a single fish’s heading angle and 
    each body position of each other fish, in each frame. 
    Inputs:
        head_idx : index of the fish whose head to consider. Must be in
                    range(Nfish). (Nfish determined as position_data.shape[2)
        position_data : position data for this dataset, presumably all_position_data[j]
        dataset: dataset dictionary of all behavior information for a given expt.
                Includes "heading_angle" from repair_heading_angles() 
        CSVcolumns : information on what the columns of position_data are
    Output : 
        relative_orientation_to_body : numpy array, 
            shape (Nframes, Nbodycolumns, Nfish-1), of 
            relative orientation, radians, for fish head_idx to each
            body position of each other fish
    """
    Nfish = position_data.shape[2]

    # All heading angles for this fish
    angle_data = dataset["heading_angle"][:, head_idx]

    # All the head-body vectors to other fish
    # shape (Nframes, N body columns, 2 = {x, y}, Nfish-1)
    head_body_vec = calc_head_body_vectors(head_idx, position_data, CSVcolumns)
    
    # unit vector in heading angle direction of fish head_idx
    v = np.stack((np.cos(angle_data[:]), 
                  np.sin(angle_data[:])), axis=1)
    v_repeat = np.repeat(v[:, np.newaxis, :], 
                            CSVcolumns["body_Ncolumns"], axis=1)
    v_repeat = np.repeat(v_repeat[:, :, :, np.newaxis], 
                            Nfish-1, axis=3)
    
    dot_product = np.sum(v_repeat * head_body_vec, axis=2)
    magnitude_product = np.linalg.norm(v_repeat, axis=2) * \
                        np.linalg.norm(head_body_vec, axis=2)
    relative_orientation_to_body = np.arccos(dot_product / magnitude_product)
        
    return relative_orientation_to_body


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




def calculate_IBI_binned_by_distance(datasets, distance_key='closest_distance_mm',
                                     bin_distance_min=0, bin_distance_max=50.0, 
                                     bin_width=5.0, 
                                     constraintKey=None, constraintRange=None,
                                     constraintIdx=None, use_abs_value_constraint=False,
                                     dilate_minus1=False,
                                     outlier_std=3.0,
                                     makePlot=True, plot_each_dataset=False,
                                     ylim=None, titleStr=None, plotColor='black',
                                     outputFileName=None, closeFigure=False,
                                     outputCSVFileName = None):
    
    """
    Calculate mean inter-bout interval (IBI) binned by mean inter-fish distance
    during the interval. Plot (optional) and write a CSV (optional).
    
    Requires Nfish = 2. Calculates IBI for each fish in each dataset.
    Returns average over all fish, average for each dataset (averaged over both fish),
    and individual fish averages.
    
    The distance_key is used for binning (primary filter on mean distance during IBI).
    Optional: constraintKey/constraintRange provides additional constraint
    (e.g., only consider IBIs where mean radial position is within some range).

    Returns - average over all fish
            - average for each dataset (averaged over both fish)
            - and individual fish averages.
    Could be used to bin IBI by any other quantitative property also
            by changing "distance_key", as long as the property
            has one value per frame, like Nfish==2 inter-fish distance.
        
    Parameters
    ----------
    datasets : list of dataset dictionaries
    distance_key : str, either 'head_head_distance_mm' or 'closest_distance_mm'
    bin_distance_min, bin_distance_max : float, distance range for binning (mm)
    bin_width : float, width of distance bins (mm)
    constraintKey : str or None
        Additional constraint key (e.g., 'radial_position_mm')
    constraintRange : tuple or None
        (min, max) range for additional constraint
    constraintIdx : int, str, or None
        Which fish/operation to use for constraint
    use_abs_value_constraint : bool
        If True, use absolute value of constraint
    dilate_minus1 : bool, if True dilate bad frames by -1
    outlier_std : for each fish's list if IBIs, remove IBI values > outlier_std
                  from the mean. (In rare cases, very high values, probably
                  due to bad tracking.)
    makePlot : bool, make a plot if true
    plot_each_dataset : (bool) if True, plot the IBI vs. distance for each 
                  dataset (values averaged over each fish)
    ylim : tuple, ylimits for plot; None for auto
    titleStr : string, title for plot
    outputFileName : string, for saving the plot (default None -- don't save)
    closeFigure : (bool) if True, if makePlot is True, close the figure
                        after creating it.
    outputCSVFileName : if not None, save to a CSV file the following (columns):
                - plotted "X" positions (bin_centers)
                - plotted mean "Y" positions (binned_IBI[:,0] == mean_IBI)
                - Standard deviation (binned_IBI[:,1] == std_IBI)
                - Standard error of the mean (binned_IBI[:,2] == sem_IBI)
                - Each individual dataset's mean IBI (binned_IBI_each_dataset)
                  (not each individual fish)
    
    Returns
    -------
    binned_IBI : numpy array of shape (n_bins, 3) 
                 containing [mean_IBI, std_IBI, sem_IBI] for each bin; 
                 stats are calculated across fish
    bin_centers : array of bin center distances (mm)
    binned_IBI_each_dataset : numpy array of shape (Ndatasets, n_bins)
             with the mean IBI for each dataset (averaged over both fish), 
             in each bin
    binned_IBI_each_fish : numpy array of shape (nfish_total, n_bins)
             with the mean IBI for each fish, each bin

    """
    
    # Check that Nfish ==2
    for j in range(len(datasets)):
        dataset = datasets[j]    
        Nfish = datasets[j]["Nfish"]
        if Nfish != 2:
            raise ValueError("IBI and distance binning is only supported for Nfish==2")

    # Set up distance bins
    bins = np.arange(bin_distance_min, bin_distance_max + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    n_bins = len(bin_centers)
    binned_IBI = np.zeros((n_bins,))
    
    # Number of datasets; initialize array
    Ndatasets = len(datasets)
    binned_IBI_each_dataset = np.zeros((Ndatasets, n_bins))

    # Number of fish; initialize array
    nfish_total = len(datasets)*Nfish # Nfish = 2 fish per dataset
    binned_IBI_each_fish = np.zeros((nfish_total, n_bins))
                        
    print('\nBinning inter-bout intervals by distance.... ', end='')
    for j in range(Ndatasets):
        print(f' {j}... ', end='')
        dataset = datasets[j]    
        # Lowest frame number (should be 1)   
        idx_offset = min(dataset["frameArray"])
                 
        # Handle bad tracking frames
        badTrackFrames = dataset["bad_bodyTrack_frames"]["raw_frames"]
        if dilate_minus1:
            dilate_badTrackFrames = dilate_frames(badTrackFrames, 
                                                  dilate_frames=np.array([-1]))
            bad_frames_set = set(dilate_badTrackFrames)
        else:
            bad_frames_set = set(badTrackFrames)

        for k in range(Nfish):
            # Initialize lists to collect IBIs for each bin
            # Each element will be a list of IBI values for that bin
            bin_IBI_lists = [[] for _ in range(n_bins)]
    
            # Start frames and durations for bouts (i.e. motion)
            moving_frameInfo = dataset[f"isActive_Fish{k}"]["combine_frames"]
    
            # number of inter-bout intervals for this fish 
            n_ibi = moving_frameInfo.shape[1]-1 
            for jj in range(n_ibi):
                all_ibi_frames = np.arange(moving_frameInfo[0,jj] + moving_frameInfo[1,jj],
                                           moving_frameInfo[0,jj+1])
                ibi_s = (len(all_ibi_frames)+1) / dataset["fps"]
                all_ibi_distances = dataset[distance_key][(all_ibi_frames-idx_offset)]
                
                # Replace bad frames with NaN
                for i in range(len(all_ibi_frames)):
                    if (all_ibi_frames[i]) in bad_frames_set:
                        all_ibi_distances[i] = np.nan

                distance_mm = np.nanmean(all_ibi_distances)
                
                # Check additional constraint if provided
                if constraintKey is not None and constraintRange is not None:
                    # Get constraint values for this IBI
                    constraint_data = dataset[constraintKey]
                    all_ibi_constraint_values = constraint_data[(all_ibi_frames - idx_offset)]
                    
                    # Handle multi-dimensional constraint data
                    if all_ibi_constraint_values.ndim > 1:
                        # Use get_values_subset to extract the right fish/operation
                        constraint_subset = get_values_subset(
                            all_ibi_constraint_values,
                            keyIdx=constraintIdx,
                            use_abs_value=use_abs_value_constraint
                        )
                    else:
                        constraint_subset = all_ibi_constraint_values
                        if use_abs_value_constraint:
                            constraint_subset = np.abs(constraint_subset)
                    
                    # Calculate mean constraint value during this IBI
                    mean_constraint_value = np.nanmean(constraint_subset)
                    
                    # Check if constraint is satisfied
                    if use_abs_value_constraint:
                        if not (constraintRange[0] <= np.abs(mean_constraint_value) <= constraintRange[1]):
                            continue
                    else:
                        if not (constraintRange[0] <= mean_constraint_value <= constraintRange[1]):
                            continue

                # Find which distance bin this window belongs to
                bin_idx = np.digitize(distance_mm, bins) - 1
                # Make sure bin index is valid, and store the IBI value
                if 0 <= bin_idx < n_bins:
                    bin_IBI_lists[bin_idx].append(ibi_s)

            # Calculate weighted averages for each bin; neglect NaNs
            for bin_idx in range(n_bins):
                if len(bin_IBI_lists[bin_idx]) > 1:
                    # require 2 points, for std. dev.
                    bin_IBI_array = np.stack(bin_IBI_lists[bin_idx], axis=0)
                    valid_pts = abs(bin_IBI_array - np.nanmean(bin_IBI_array)) \
                        <= (outlier_std*np.nanstd(bin_IBI_array))
                    binned_IBI_each_fish[j*Nfish + k, bin_idx] = \
                        np.nanmean(bin_IBI_array[valid_pts])
                else:
                    # No data for this bin
                    binned_IBI_each_fish[j*Nfish + k, bin_idx] = np.nan  
           
        # Combine fish IBIs to get average per dataset (simple average, not weighted)
        binned_IBI_each_dataset[j,:] = np.nanmean(binned_IBI_each_fish[j*Nfish : j*Nfish + 2, :], axis=0)
            
    print('... done.\n')
    
    # Average over all fish.
    binned_IBI = np.zeros((n_bins, 3))
    binned_IBI[:,0] = np.nanmean(binned_IBI_each_fish, axis=0)
    binned_IBI[:,1] = np.nanstd(binned_IBI_each_fish, axis=0)
    binned_IBI[:,2] = np.nanstd(binned_IBI_each_fish, axis=0) / np.sqrt(nfish_total)
    
    xlabelStr= f'Distance: {distance_key}' # will also use for CSV
    if makePlot:
        fig = plt.figure()
        plt.errorbar(bin_centers, binned_IBI[:,0], binned_IBI[:,2], 
                     fmt='o', capsize=7, markersize=12,
                     color = plotColor, ecolor = plotColor)

        if plot_each_dataset:
            alpha_each = np.max((0.7/Ndatasets, 0.15))
            for i in range(Ndatasets):
                plt.plot(bin_centers, binned_IBI_each_dataset[i,:], color=plotColor, 
                            alpha=alpha_each)

        plt.title(titleStr)
        plt.xlabel(xlabelStr)
        plt.ylabel('Mean IBI (s)')
        plt.xlim((bin_distance_min, bin_distance_max))
        if ylim is not None:
            plt.ylim(ylim)

        if outputFileName != None:
            plt.savefig(outputFileName, bbox_inches='tight')

        if closeFigure:
            plt.close(fig)
            
    # If only 1 bin, flatten
    if binned_IBI.shape[0]==1:
        binned_IBI = binned_IBI.flatten()

    # Output points to CSV (optional)
    if outputCSVFileName is not None:
        header_strings = [xlabelStr.replace(',', '_'), 
                          'mean_IBI (s)', 'std_IBI (s)', 'sem_IBI (s)']
        for j in range(Ndatasets):
            header_strings.append(f'IBI_Dataset_{j+1}')
        list_to_output = [binned_IBI[:, i] if binned_IBI.ndim == 2 \
                          else binned_IBI[i] for i in range(3)]
        for j in range(Ndatasets):
            list_to_output.append(binned_IBI_each_dataset[j,:],)
        simple_write_CSV(bin_centers, 
                         list_to_output, 
                         filename =  outputCSVFileName, 
                         header_strings=header_strings)
        
    return binned_IBI, bin_centers, binned_IBI_each_dataset, binned_IBI_each_fish



def calculate_IBI_binned_by_2D_keys(datasets, 
                                     key1='closest_distance_mm',
                                     key2='radial_position_mm',
                                     bin_ranges=((0.0, 50.0), (0.0, 25.0)), 
                                     Nbins=(20, 20),
                                     constraintKey=None, constraintRange=None,
                                     constraintIdx=None, use_abs_value_constraint=False,
                                     dilate_minus1=False,
                                     outlier_std=3.0,
                                     makePlot=True, 
                                     titleStr=None, xlabelStr=None, ylabelStr=None,
                                     cmap='RdYlBu_r',
                                     colorRange=None,
                                     plot_type = 'heatmap',
                                     outputFileName=None,
                                     closeFigure=False):
    """
    Calculate mean inter-bout interval (IBI) binned by two quantitative keys
    (e.g., inter-fish distance and radial position).
    Creates a 2D heatmap showing mean IBI in each bin.
        
    Both key1 and key2 are used for binning (primary filters on mean values 
                                             during IBI).
    Optional: constraintKey/constraintRange provides additional constraint
    (e.g., only consider IBIs where some other property is within some range).
    
    Requires Nfish==2.
    Returns average over all fish, average for each dataset, 
    and individual fish averages.
    
    
    Parameters
    ----------
    datasets : list of dataset dictionaries
    key1 : str, first key for binning (default 'closest_distance_mm')
    key2 : str, second key for binning (default 'radial_position_mm')
    bin_ranges : tuple of two tuples, ((key1_min, key1_max), (key2_min, key2_max))
                 If None, will auto-determine from data
    Nbins : tuple of two ints, number of bins for (key1, key2)
    constraintKey : str or None
        Additional constraint key (not used for binning)
    constraintRange : tuple or None
        (min, max) range for additional constraint
    constraintIdx : int, str, or None
        Which fish/operation to use for constraint
    use_abs_value_constraint : bool
        If True, use absolute value of constraint
    dilate_minus1 : bool, if True dilate bad frames by -1
    outlier_std : for each fish's list of IBIs in each bin, remove IBI values 
                  > outlier_std from the mean. (In rare cases, very high values,
                  probably due to bad tracking.)
    makePlot : bool, make a plot if true
    titleStr : string, title for plot (if None, auto-generate)
    xlabelStr, ylabelStr : x and y axis labels. If None, use key names
    cmap : string, colormap for heatmap (default 'RdYlBu_r')
    colorRange : tuple (vmin, vmax) for color scale, None for auto
    plot_type : str, 'heatmap' or 'line_plots'
                   Determines which plotting function to use.
    outputFileName : string, for saving the plot (default None -- don't save)
    closeFigure : bool, if True, close the figure after creating it
    
    Returns
    -------
    binned_IBI : 3D array (Nbins[0] x Nbins[1] x 3) containing 
                 [mean_IBI, std_IBI, sem_IBI] for each bin;
                 stats are calculated across fish
    X, Y : 2D arrays from meshgrid for bin centers
    binned_IBI_each_dataset : 3D numpy array of shape (Ndatasets, Nbins[0], Nbins[1])
             with the mean IBI for each dataset (averaged over both fish), 
             in each 2D bin
    binned_IBI_each_fish : 3D array (nfish_total, Nbins[0], Nbins[1])
                          with the mean IBI for each fish, each bin
    """

    # Check that Nfish ==2
    for j in range(len(datasets)):
        dataset = datasets[j]    
        Nfish = datasets[j]["Nfish"]
        if Nfish != 2:
            raise ValueError("IBI and distance binning is only supported for Nfish==2")

    # Extract bin ranges
    key1_min, key1_max = bin_ranges[0]
    key2_min, key2_max = bin_ranges[1]
    
    # Calculate bin edges and centers
    key1_edges = np.linspace(key1_min, key1_max, Nbins[0] + 1)
    key2_edges = np.linspace(key2_min, key2_max, Nbins[1] + 1)
    
    # Create meshgrid for bin centers
    key1_centers = 0.5 * (key1_edges[1:] + key1_edges[:-1])
    key2_centers = 0.5 * (key2_edges[1:] + key2_edges[:-1])
    X, Y = np.meshgrid(key1_centers, key2_centers, indexing='ij')
    
    # Number of datasets; initialize array to store per-dataset binned IBI values
    Ndatasets = len(datasets)
    binned_IBI_each_dataset = np.zeros((Ndatasets, Nbins[0], Nbins[1]))
    binned_IBI_each_dataset[:] = np.nan  # Initialize with NaN
    
    # Initialize array to store per-fish binned IBI values
    nfish_total = len(datasets) * Nfish  # 2 fish per dataset
    binned_IBI_each_fish = np.zeros((nfish_total, Nbins[0], Nbins[1]))
    binned_IBI_each_fish[:] = np.nan  # Initialize with NaN
    
    print(f'\nBinning inter-bout intervals by {key1} and {key2}.... ', end='')
    
    for j in range(len(datasets)):
        print(f' {j}... ', end='')
        dataset = datasets[j]
        
        # Lowest frame number (should be 1)
        idx_offset = min(dataset["frameArray"])
        
        # Handle bad tracking frames
        badTrackFrames = dataset["bad_bodyTrack_frames"]["raw_frames"]
        if dilate_minus1:
            dilate_badTrackFrames = dilate_frames(badTrackFrames, 
                                                  dilate_frames=np.array([-1]))
            bad_frames_set = set(dilate_badTrackFrames)
        else:
            bad_frames_set = set(badTrackFrames)
        
        for k in range(Nfish):
            # Initialize lists to collect IBIs for each 2D bin
            # Each element will be a list of IBI values for that bin
            bin_IBI_lists = [[[] for _ in range(Nbins[1])] for _ in range(Nbins[0])]
            
            # Start frames and durations for bouts (i.e. motion)
            moving_frameInfo = dataset[f"isActive_Fish{k}"]["combine_frames"]
            
            # Number of inter-bout intervals for this fish
            n_ibi = moving_frameInfo.shape[1] - 1
            
            for jj in range(n_ibi):
                # Get all frames in this inter-bout interval
                all_ibi_frames = np.arange(
                    moving_frameInfo[0, jj] + moving_frameInfo[1, jj],
                    moving_frameInfo[0, jj+1]
                )
                
                # Calculate IBI duration in seconds
                ibi_s = (len(all_ibi_frames) + 1) / dataset["fps"]
                
                # Get key1 values for this interval
                all_key1_values = dataset[key1][(all_ibi_frames - idx_offset)]
                
                # Get key2 values for this interval
                # Handle case where key2 might be per-fish (2D array)
                key2_data = dataset[key2]
                if key2_data.ndim == 2:
                    # Use the current fish's values
                    all_key2_values = key2_data[(all_ibi_frames - idx_offset), k]
                else:
                    # Single value per frame
                    all_key2_values = key2_data[(all_ibi_frames - idx_offset)]
                
                # Replace bad frames with NaN
                for i in range(len(all_ibi_frames)):
                    if all_ibi_frames[i] in bad_frames_set:
                        all_key1_values[i] = np.nan
                        all_key2_values[i] = np.nan
                
                # Calculate mean values for this IBI
                mean_key1 = np.nanmean(all_key1_values)
                mean_key2 = np.nanmean(all_key2_values)
                
                # Only include if both key values are valid
                if not (np.isnan(mean_key1) or np.isnan(mean_key2)):
                    # Check additional constraint if provided
                    if constraintKey is not None and constraintRange is not None:
                        # Get constraint values for this IBI
                        constraint_data = dataset[constraintKey]
                        all_ibi_constraint_values = constraint_data[(all_ibi_frames - idx_offset)]
                        
                        # Handle multi-dimensional constraint data
                        if all_ibi_constraint_values.ndim > 1:
                            # Use get_values_subset to extract the right fish/operation
                            constraint_subset = get_values_subset(
                                all_ibi_constraint_values,
                                keyIdx=constraintIdx,
                                use_abs_value=use_abs_value_constraint
                            )
                        else:
                            constraint_subset = all_ibi_constraint_values
                            if use_abs_value_constraint:
                                constraint_subset = np.abs(constraint_subset)
                        
                        # Calculate mean constraint value during this IBI
                        mean_constraint_value = np.nanmean(constraint_subset)
                        
                        # Check if constraint is satisfied
                        if use_abs_value_constraint:
                            if not (constraintRange[0] <= np.abs(mean_constraint_value) <= constraintRange[1]):
                                continue
                        else:
                            if not (constraintRange[0] <= mean_constraint_value <= constraintRange[1]):
                                continue
                    
                    # Find which bins this IBI belongs to
                    bin_idx1 = np.digitize(mean_key1, key1_edges) - 1
                    bin_idx2 = np.digitize(mean_key2, key2_edges) - 1
                    
                    # Make sure bin indices are valid
                    if (0 <= bin_idx1 < Nbins[0]) and (0 <= bin_idx2 < Nbins[1]):
                        bin_IBI_lists[bin_idx1][bin_idx2].append(ibi_s)
                        
            # Calculate weighted averages for each bin; neglect NaNs and outliers
            for bin_idx1 in range(Nbins[0]):
                for bin_idx2 in range(Nbins[1]):
                    if len(bin_IBI_lists[bin_idx1][bin_idx2]) > 1:
                        # Require 2 points for std. dev.
                        bin_IBI_array = np.array(bin_IBI_lists[bin_idx1][bin_idx2])
                        mean_val = np.nanmean(bin_IBI_array)
                        std_val = np.nanstd(bin_IBI_array)
                        valid_pts = np.abs(bin_IBI_array - mean_val) <= (outlier_std * std_val)
                        binned_IBI_each_fish[j*Nfish + k, bin_idx1, bin_idx2] = \
                            np.nanmean(bin_IBI_array[valid_pts])
                    elif len(bin_IBI_lists[bin_idx1][bin_idx2]) == 1:
                        # Only one data point, use it
                        binned_IBI_each_fish[j*Nfish + k, bin_idx1, bin_idx2] = \
                            bin_IBI_lists[bin_idx1][bin_idx2][0]
                    # else: remains NaN (no data for this bin)

        # Combine fish IBIs to get average per dataset (simple average, not weighted)
        binned_IBI_each_dataset[j, :, :] = \
            np.nanmean(binned_IBI_each_fish[j*Nfish : j*Nfish + 2, :, :], axis=0)
        
    print('... done')

    # Calculate overall statistics
    binned_IBI = np.zeros((Nbins[0], Nbins[1], 3))
    binned_IBI[:, :, 0] = np.nanmean(binned_IBI_each_fish, axis=0)  # mean
    binned_IBI[:, :, 1] = np.nanstd(binned_IBI_each_fish, axis=0)   # std
    binned_IBI[:, :, 2] = np.nanstd(binned_IBI_each_fish, axis=0) / np.sqrt(nfish_total)  # sem
    
    
    
    if makePlot:
        
        if titleStr is None:
            titleStr = f'Mean IBI binned by {key1} and {key2}'
        if xlabelStr is None:
            xlabelStr = key1
        if ylabelStr is None:
            ylabelStr = key2

        # Choose plotting function based on plot_type
        # For line_plots, ignore the colorRange, but use sem as error bars
        if plot_type.lower() == 'linePlots':
            plot_2Darray_linePlots(binned_IBI[:, :, 0], X, Y, 
                                   Z_unc=binned_IBI[:, :, 2],
                           titleStr=titleStr, 
                           xlabelStr=xlabelStr, ylabelStr=ylabelStr, 
                           clabelStr='Mean IBI (s)',
                           colorRange=None, cmap=cmap,
                           unit_scaling_for_plot=[1.0, 1.0, 1.0],
                           mask_by_sem_limit=None,
                           outputFileName=outputFileName, closeFigure=closeFigure)
        else: # default to heatmap
            plot_2D_heatmap(binned_IBI[:, :, 0], X, Y, Z_unc=None,
                               titleStr=titleStr, 
                               xlabelStr=xlabelStr, ylabelStr=ylabelStr, 
                               clabelStr='Mean IBI (s)',
                               colorRange=colorRange, cmap=cmap,
                               unit_scaling_for_plot=[1.0, 1.0, 1.0],
                               mask_by_sem_limit=None,
                               outputFileName=outputFileName, 
                               closeFigure=closeFigure)
    
    return binned_IBI, X, Y, binned_IBI_each_dataset, binned_IBI_each_fish



def calculate_interfish_bout_lags(datasets):
    """
    Calculate the delays between the start of a bout for one fish and
    the start of a bout for the other fish. 
    For each fish i, the time between the start of a bout and the start of the
        next bout of fish j that occurs at the same or a later time.
    Tabulate all of these, across fish; return a list, one set of delays per
    dataset
    
    Requires Nfish==2
    
    Parameters
    ----------
    datasets : list of dataset dictionaries
    distance_key : str, either 'head_head_distance_mm' or 'closest_distance_mm'
    bin_distance_min, bin_distance_max : float, distance range for binning (mm)
    bin_width : float, width of distance bins (mm)
    dilate_minus1 : bool, if True dilate bad frames by -1
    
    Returns
    -------
    interfish_bout_lags_s : list of arrays of inter-fish bout lags (s)
        in each dataset (plot using plot_probability_distr)
    """

    interfish_bout_lags_s = []
    
    print('\nDetermining inter-fish bout delays... ', end='')
    for j in range(len(datasets)):
        print(f' {j}... ', end='')
        dataset = datasets[j]    
        Nfish = dataset["Nfish"]
        if Nfish != 2:
            raise ValueError("IBI and distance binning is only supported for Nfish==2")
        
        # list of bout start frames for each fish
        startFrames0 = dataset["isActive_Fish0"]["combine_frames"][0,:]
        startFrames1 = dataset["isActive_Fish1"]["combine_frames"][0,:]
        
        lags_list = []
        for s0 in startFrames0:
            later_s1 = startFrames1[startFrames1 >= s0]
            if len(later_s1)> 0:
                min_diff_later_s1 = np.min(later_s1 - s0)
                lags_list.append(min_diff_later_s1)
        for s1 in startFrames1:
            later_s0 = startFrames0[startFrames0 >= s1]
            if len(later_s0)> 0:
                min_diff_later_s0 = np.min(later_s0 - s1)
                lags_list.append(min_diff_later_s0)
            
        interfish_bout_lags_s.append(np.array(lags_list)/dataset["fps"])
    print(' Done')
    
    return interfish_bout_lags_s
    

def make_pair_fish_plots(datasets, exptName = '', color = 'black',
                         plot_type_2D = 'heatmap',
                         outputFileNameBase = 'pair_fish', 
                         outputFileNameExt = 'png',
                         closeFigures = False,
                         writeCSVs = False):
    """
    Makes several useful "pair" plots -- i.e. plots of characteristics 
    of pairs of fish.
    Note that there are lots of parameter values that are hard-coded; this
    function is probably more useful to read than to run, pasting and 
    modifying its code.
    Bending angle plots extracted and moved to make_bending_angle_plots()
    
    Inputs:
        datasets : dictionaries for each dataset
        exptName : (string) Experiment name, to append to titles.
        color: plot color (uses alpha for indiv. dataset colors)
        plot_type_2D : str, 'heatmap' or 'line_plots'
                    Which plotting function make_2D_histogram() will use
                    ('heatmap' or 'line_plots')
        outputFileNameBase : base file name for figure output; if None,
                             won't save a figure file
        outputFileNameExt : extension for figure output (e.g. 'eps' or 'png')
        closeFigures : (bool) if True, close a figure after creating it.
        writeCSVs : (bool) Used by various functions; if true, output plotted 
                            points to a CSV file. See code for filenames

    Outputs:
        None

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
                                                     dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_distance_head_head', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    plot_probability_distr(head_head_mm_all, bin_width = 0.5, 
                           bin_range = [0, None], 
                           color = color,
                           yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           xlim = (-1.0, 50.0), ylim = (-0.005, 0.05),
                           xlabelStr = 'Head-head distance (mm)', 
                           titleStr = f'{exptName}: head-head distance (mm)',
                           outputFileName = outputFileName,
                           closeFigure = closeFigures,
                           outputCSVFileName = outputCSVFileName)

    # closest distance histogram
    closest_distance_mm_all = combine_all_values_constrained(datasets, 
                                                     keyName='closest_distance_mm', 
                                                     dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_distance_closest', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    plot_probability_distr(closest_distance_mm_all, bin_width = 0.5, 
                           bin_range = [0, None], 
                           color = color,
                           yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           xlim = (-1.0, 50.0), ylim = (-0.005, 0.15),
                           xlabelStr = 'Closest distance (mm)', 
                           titleStr = f'{exptName}: closest distance (mm)',
                           outputFileName = outputFileName,
                           closeFigure = closeFigures,
                           outputCSVFileName = outputCSVFileName)

    # Relative heading angle histogram
    relative_heading_angle_all = combine_all_values_constrained(datasets, 
                                                 keyName='relative_heading_angle', 
                                                 dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_rel_heading_angle', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    bin_width = np.pi/30
    plot_probability_distr(relative_heading_angle_all, bin_width = bin_width,
                           bin_range=[None, None], 
                           color = color,
                           yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           polarPlot = True,
                           titleStr = f'{exptName}: Relative Heading Angle',
                           ylim = (0, 0.6),
                           outputFileName = outputFileName,
                           closeFigure = closeFigures,
                           outputCSVFileName = outputCSVFileName)

    # Relative orientation angle histogram
    relative_orientation_angle_all = combine_all_values_constrained(datasets, 
                                                 keyName='relative_orientation', 
                                                 dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_rel_orientation', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    bin_width = np.pi/30
    plot_probability_distr(relative_orientation_angle_all, 
                                  bin_width = bin_width,
                                  bin_range=[None, None], 
                                  color = color,
                                  yScaleType = 'linear',
                                  plot_each_dataset = False,
                                  plot_sem_band = True,
                                  polarPlot = True,
                                  titleStr = f'{exptName}: Relative Orientation Angle',
                                  ylim = (0, 0.6),
                                  outputFileName = outputFileName,
                                  closeFigure = closeFigures,
                                  outputCSVFileName = outputCSVFileName)

    # Sum of relative orientation angles histogram
    relative_orientation_sum_all = combine_all_values_constrained(datasets, 
                                                 keyName='relative_orientation_sum', 
                                                 dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_rel_orientation_sum', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    bin_width = np.pi/60
    plot_probability_distr(relative_orientation_sum_all, bin_width = bin_width,
                           bin_range=[None, None], 
                           color = color,
                           yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           polarPlot = False,
                           titleStr = f'{exptName}: Sum of Relative Orientation Angles',
                           xlabelStr = 'Sum of Rel. Orient. Angles (rad)',
                           ylim = (0, 0.6), xlim = (-6.3, 6.3),
                           outputFileName = outputFileName,
                           closeFigure = closeFigures,
                           outputCSVFileName = outputCSVFileName)

    # Sum of absolute value of relative orientation angles histogram
    relative_orientation_abs_sum_all = combine_all_values_constrained(datasets, 
                                                 keyName='relative_orientation_abs_sum', 
                                                 dilate_minus1 = False)
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_rel_orientation_abs_sum', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    bin_width = np.pi/60
    plot_probability_distr(relative_orientation_abs_sum_all, bin_width = bin_width,
                           bin_range=[None, None], 
                           color = color,
                           yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           polarPlot = False,
                           titleStr = f'{exptName}: Sum of Relative Orientation Angles',
                           xlabelStr = 'Sum of Rel. Orient. Angles (rad)',
                           ylim = (0, 0.6), xlim = (0.0, 6.3),
                           outputFileName = outputFileName,
                           closeFigure = closeFigures,
                           outputCSVFileName = outputCSVFileName)

    """
    # 2D histogram of heading alignment and head-head distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_heading_distance_2D' + '.' + outputFileNameExt
    else:
        outputFileName = None
    # use heatmap as plot type; for 2D histogram slicing along Y is not helpful
    make_2D_histogram(datasets, keyNames = ('head_head_distance_mm', 
                                            'relative_heading_angle'), 
                          keyIdx = (None, None), 
                          dilate_minus1=False, 
                          bin_ranges=((0.0, 50.0), (0.0, 3.142)), 
                          Nbins=(20,20),
                          colorRange = (0.0, 0.007),
                          titleStr = f'{exptName}: heading angle and hh distance', 
                          cmap = 'viridis',
                          plot_type = 'heatmap',
                          outputFileName = outputFileName,
                          closeFigure = closeFigures)
    """

    # 2D histogram of abs(relative orientation) and head-head distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_orientation_distance_2D' + '.' + outputFileNameExt
    else:
        outputFileName = None
    # use heatmap as plot type; for 2D histogram slicing along Y is not helpful
    make_2D_histogram(datasets, keyNames = ('head_head_distance_mm', 
                                            'relative_orientation'), 
                          keyIdx = (None, None), 
                          use_abs_value = (False, True),
                          dilate_minus1=False, bin_ranges=((0.0, 50.0), 
                                                           (0.0, 3.142)), 
                          Nbins=(20,20), cmap = 'viridis',
                          colorRange = (0.0, 0.0075),
                          clabelStr = 'Probability',
                          titleStr = f'{exptName}: abs(Rel. orient.) and hh distance', 
                          plot_type = 'heatmap',
                          outputFileName = outputFileName,
                          closeFigure = closeFigures)

    """
    # Speed of the "other" fish vs. time for bouts
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_boutSpeed_other' + '.' + outputFileNameExt
    else:
        outputFileName = None
    average_bout_trajectory_allSets(datasets, keyName = "speed_array_mm_s", 
                                    keyIdx = 'other', t_range_s=(-1.0, 2.0), 
                                    titleStr = f'{exptName}: Bout Speed, other fish', makePlot=True,
                                    outputFileName = outputFileName,
                                    closeFigure = closeFigures)
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
                                    outputFileName = outputFileName,
                                    closeFigure = closeFigures)
    """
        
    """
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
                      colorRange=(0, 0.002), cmap = 'viridis', 
                      titleStr = f'{exptName}: CJ probability', 
                      outputFileName = outputFileName,
                      closeFigure = closeFigures)
    
    # Inter-bout interval (IBI) binned by inter-fish distance.
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_IBI_v_distance' + '.' + outputFileNameExt
    else:
        outputFileName = None
    calculate_IBI_binned_by_distance(
        datasets=datasets, 
        distance_key='head_head_distance_mm',bin_distance_min=0, 
        bin_distance_max=50.0, bin_width=5.0, dilate_minus1=False,
        outlier_std = 3.0,
        makePlot = True, ylim = (0.2, 0.55), titleStr = exptName, 
        plotColor = color,
        outputFileName = outputFileName,
        closeFigure = closeFigures)

    # Inter-bout interval (IBI) binned by radial position.
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_IBI_v_r' + '.' + outputFileNameExt
    else:
        outputFileName = None
    calculate_IBI_binned_by_distance(
        datasets=datasets, 
        distance_key='radial_position_mm',bin_distance_min=0, 
        bin_distance_max=25.0, bin_width=5.0, dilate_minus1=False,
        makePlot = True, ylim = (0.2, 0.55), titleStr = exptName, 
        plotColor = color,
        outputFileName = f'{exptName}_IBI_v_r.png',
        closeFigure = closeFigures)
    """
    
    # Inter-bout interval (IBI) binned by inter-fish distance *and* 
    # radial position
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_IBI_v_dist_and_radialpos' + '.' + outputFileNameExt
    else:
        outputFileName = None
    binned_IBI, X, Y, binned_IBI_each_dataset, binned_IBI_each_fish = \
        calculate_IBI_binned_by_2D_keys(datasets=datasets, 
                                     key1='head_head_distance_mm',
                                     key2='radial_position_mm',
                                     bin_ranges=((0.0, 50.0), (0.0, 25.0)), 
                                     Nbins=(12, 12),
                                     dilate_minus1=False,
                                     outlier_std=3.0,
                                     makePlot=True, 
                                     titleStr = f'{exptName}: IBI vs. head-head distance, radial pos.', 
                                     cmap='viridis_r',
                                     colorRange = (0.0, 0.6),
                                     plot_type = plot_type_2D,
                                     outputFileName=outputFileName,
                                     closeFigure=closeFigures)
        
    # Slice along IBI binned by distance and r, for r < 22 mm
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_IBI_v_dist_r_interior' + '.' + outputFileNameExt
    else:
        outputFileName = None
    r_range = (15.0, 22.0)
    titleStr = f'{exptName}: Average IBI for {r_range[0]:.1f} r < {r_range[1]:.1f} mm'
    xlabelStr = 'Closest Distance (mm)'
    ylabelStr = 'Radial position (mm)'
    zlabelStr = 'Average IBI (s)'
    xlim = (0.0, 50.0)
    zlim = (0.0, 0.6)
    color = color
    slice_2D_histogram(binned_IBI[:,:,0], X, Y, binned_IBI[:,:,2], 
                       slice_axis = 'x', other_range = r_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = ylabelStr, zlim = zlim, xlim = xlim,
                       color = color, outputFileName=outputFileName,
                       closeFigure=closeFigures)

    # Slice along IBI binned by distance and r, for r >= 22 mm
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_IBI_v_dist_r_edge' + '.' + outputFileNameExt
    else:
        outputFileName = None
    r_range = (22.0, np.inf)
    titleStr = f'{exptName}: Average IBI for r >= {r_range[0]:.1f} mm'
    xlabelStr = 'Closest Distance (mm)'
    ylabelStr = 'Radial position (mm)'
    zlabelStr = 'Average IBI (s)'
    xlim = (0.0, 50.0)
    zlim = (0.0, 0.6)
    color = color
    slice_2D_histogram(binned_IBI[:,:,0], X, Y, binned_IBI[:,:,2], 
                       slice_axis = 'x', other_range = r_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = ylabelStr, zlim = zlim, xlim = xlim, 
                       color = color, outputFileName=outputFileName,
                       closeFigure=closeFigures)

    
    # Average above-threshold speed versus distance and relative orientation
    # Hard-code speed threshold 
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_movingSpeed_v_HHdistance_orientation_2D' + '.' + outputFileNameExt
    else:
        outputFileName = None
    speed_threshold = 9.0 # mm/s 
    keyNames = ('head_head_distance_mm', 'relative_orientation')
    keyIdx = (None, None)
    use_abs_value = (False, True)
    keyNameC = 'speed_array_mm_s'
    keyIdxC = None
    constraintKey= 'speed_array_mm_s'
    constraintRange=(speed_threshold, np.inf)
    constraintIdx = None
    use_abs_value_constraint = False
    bin_ranges= ((0.0, 50.0), (0.0, 3.142))
    Nbins = (20, 12)
    titleStr = f'{exptName}: Avg speed when > {constraintRange[0]:.1f} mm/s'
    xlabelStr = 'Head-Head Distance (mm)'
    ylabelStr = 'Relative Orientation (rad)'
    colorRange = (constraintRange[0], 40.0)
    hist_speed, X, Y, hist_speed_sem = make_2D_histogram(
        datasets,
        keyNames = keyNames, keyIdx = keyIdx, use_abs_value = use_abs_value,
        keyNameC = keyNameC, keyIdxC =keyIdxC, 
        constraintKey = constraintKey, constraintRange = constraintRange, 
        constraintIdx = constraintIdx, 
        use_abs_value_constraint = use_abs_value_constraint,
        dilate_minus1=True, bin_ranges = bin_ranges, Nbins = Nbins, 
        titleStr = titleStr,
        clabelStr= 'Mean above-threshold speed (mm/s)',
        xlabelStr = xlabelStr, ylabelStr = ylabelStr, 
        colorRange = colorRange, 
        plot_type = plot_type_2D,
        outputFileName = outputFileName, 
        closeFigure = closeFigures)

    # Slice along above-threshold speed versus distance and relative orientation binned by distance and orientation, 
    # orientation axis, constrain orientation angle to be < 60 degres
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_movingSpeed_v_HHdistance_lowOrientation' + '.' + outputFileNameExt
    else:
        outputFileName = None
    orientation_range = (0.0, np.pi/3.0)
    titleStr = f'{exptName}: Avg speed when > {speed_threshold:.1f} mm/s, ' + \
        f'for |rel. orient| < {180.0*orientation_range[1]/np.pi:.3f} rad'
    xlabelStr = 'Head-Head Distance (mm)'
    ylabelStr = 'Rel. Orient. Angle (rad)'
    zlabelStr = 'Average speed (mm/s)'
    xlim = (0.0, 50.0)
    zlim = (0.0, 90.0)
    color = color
    slice_2D_histogram(hist_speed, X, Y, hist_speed_sem, 
                       slice_axis = 'x', other_range = orientation_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = ylabelStr, zlim = zlim, xlim = xlim, 
                       plot_z_zero_line = False,
                       plot_vert_zero_line = False,
                       color = color, outputFileName=outputFileName,
                       closeFigure=closeFigures)


    # Speed cross-correlation function
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames('_speedCrosscorr', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    speed_cc_all, t_lag = \
        calculate_value_corr_all(datasets, keyName = 'speed_array_mm_s',
                                 corr_type='cross', dilate_minus1 = True, 
                                 t_max = 2.0, t_window = 5.0, fpstol = 1e-6)
    plot_function_allSets(speed_cc_all, t_lag, xlabelStr='time (s)', 
                          ylabelStr='Speed Cross-correlation', 
                          titleStr=f'{exptName}: Speed Cross-correlation', 
                          ylim = (-0.03, 0.2),
                          color = color,
                          plot_each_dataset = True, 
                          average_in_dataset = True,
                          outputFileName = outputFileName,
                          closeFigure = closeFigures,
                          outputCSVFileName = outputCSVFileName)
    
    # Waterfall plot of speed cross-correlations binned by distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_speedCrosscorrDistBinned' + '.' + outputFileNameExt
    else:
        outputFileName = None
    binned_crosscorr_all, bin_centers, t_lag, bin_counts_all = \
        calculate_value_corr_all_binned(datasets, keyName='speed_array_mm_s', 
                                        binKeyName = 'closest_distance_mm', 
                                        bin_value_min = 0.0, bin_value_max = 50.0, 
                                        bin_width=5.0, t_max=2.0, t_window=5.0,
                                        dilate_minus1=True)
    plot_waterfall_binned_crosscorr(binned_crosscorr_all, bin_centers, t_lag,
                                    bin_counts_all=bin_counts_all, 
                                    xlabelStr='Time lag (s)',
                                    titleStr=f'{exptName}: Closest Distance-Binned Cross-correlation',
                                    outputFileName=outputFileName,
                                    closeFigure = closeFigures)

    return None


def make_pair_1D_v_distance_plots(datasets, exptName = '', 
                                  distanceKey='closest_distance_mm', 
                                  bin_range=(0.0, 50.0), Nbins=20,
                                  color = 'black',
                                  outputFileNameBase = 'pair_', 
                                  outputFileNameExt = 'png',
                                  closeFigures = False,
                                  writeCSVs = False):
    """
    Makes several useful plots of something vs. inter-fish distance
    
    Inputs:
        datasets : dictionaries for each dataset
        exptName : (string) Experiment name, to append to titles.
        distanceKey : (string) which distance key to use, 
                      'closest_distance_mm' or 'head_head_distance_mm'
        bin_range : tuple,  (min, max) for binning
        Nbins : int,  Number of bins
        color: plot color (uses alpha for indiv. dataset colors)
        outputFileNameBase : base file name for figure output; if None,
                             won't save a figure file
        outputFileNameExt : extension for figure output (e.g. 'eps' or 'png')
        closeFigures : (bool) if True, close a figure after creating it.
        writeCSVs : (bool) Used by various functions; if true, output plotted 
                            points to a CSV file. See code for filenames

    Outputs:
        None

    """
        
    verifyPairs = True
    for j in range(len(datasets)):
        if datasets[j]["Nfish"] != 2:
            verifyPairs = False
    if verifyPairs==False:
        raise ValueError('Error in make_pair_1D_v_distance_plots; Nfish must be 2 !')
    
    if distanceKey=='closest_distance_mm':
        xlabelStr = 'Closest distance (mm)'
        distance_abbrev = 'CL'
    elif distanceKey=='head_head_distance_mm':
        xlabelStr = 'Head-Head distance (mm)'
        distance_abbrev = 'HH'
    else: 
        raise ValueError('Invalid distance key.')
        
    # Perpendicular one-sees
    keyName = 'perp_oneSees'
    titleStr = f'{keyName} probability v distance'
    ylabelStr = f'{keyName} probability'
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames(
                                            f'_{keyName}_v_{distance_abbrev}distance', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    _, _, _ = calculate_property_1Dbinned(datasets, 
                                          keyName= keyName, 
                                          key_is_a_behavior = True, 
                                          binKeyName=distanceKey,
                                          bin_range=bin_range, Nbins=Nbins,
                                          dilate_minus1= False,
                                          makePlot=True, titleStr=titleStr,
                                          xlabelStr= xlabelStr, 
                                          ylabelStr= ylabelStr,
                                          color=color, 
                                          outputFileName=outputFileName, 
                                          closeFigure=closeFigures)

    # J-Bend
    keyName = 'Jbend_any'
    titleStr = f'{keyName} probability v distance'
    ylabelStr = f'{keyName} probability'
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames(f'_{keyName}_v_{distance_abbrev}distance', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    _, _, _ = calculate_property_1Dbinned(datasets, 
                                          keyName= keyName, 
                                          key_is_a_behavior = True, 
                                          binKeyName=distanceKey,
                                          bin_range=bin_range, Nbins=Nbins,
                                          dilate_minus1= False,
                                          makePlot=True, titleStr=titleStr,
                                          xlabelStr= xlabelStr, 
                                          ylabelStr= ylabelStr,
                                          color=color, 
                                          outputFileName=outputFileName, 
                                          closeFigure=closeFigures)

    # C-Bend
    keyName = 'Cbend_any'
    titleStr = f'{keyName} probability v distance'
    ylabelStr = f'{keyName} probability'
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames(f'_{keyName}_v_{distance_abbrev}distance', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    _, _, _ = calculate_property_1Dbinned(datasets, 
                                          keyName= keyName, 
                                          key_is_a_behavior = True, 
                                          binKeyName=distanceKey,
                                          bin_range=bin_range, Nbins=Nbins,
                                          dilate_minus1= False,
                                          makePlot=True, titleStr=titleStr,
                                          xlabelStr= xlabelStr, 
                                          ylabelStr= ylabelStr,
                                          color=color, 
                                          outputFileName=outputFileName, 
                                          closeFigure=closeFigures)

    # R-Bend
    keyName = 'Rbend_any'
    titleStr = f'{keyName} probability v distance'
    ylabelStr = f'{keyName} probability'
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames(f'_{keyName}_v_{distance_abbrev}distance', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    _, _, _ = calculate_property_1Dbinned(datasets, 
                                          keyName= keyName, 
                                          key_is_a_behavior = True, 
                                          binKeyName=distanceKey,
                                          bin_range=bin_range, Nbins=Nbins,
                                          dilate_minus1= False,
                                          makePlot=True, titleStr=titleStr,
                                          xlabelStr= xlabelStr, 
                                          ylabelStr= ylabelStr,
                                          color=color, 
                                          outputFileName=outputFileName, 
                                          closeFigure=closeFigures)

    # isActive (moving or bending)
    keyName = 'isActive_any'
    titleStr = f'{keyName} probability v distance'
    ylabelStr = f'{keyName} probability'
    outputFileName, outputCSVFileName = get_plot_and_CSV_filenames(f'_{keyName}_v_{distance_abbrev}distance', 
                                            outputFileNameBase, 
                                            outputFileNameExt, writeCSVs)
    _, _, _ = calculate_property_1Dbinned(datasets, 
                                          keyName= keyName, 
                                          key_is_a_behavior = True, 
                                          binKeyName=distanceKey,
                                          bin_range=bin_range, Nbins=Nbins,
                                          dilate_minus1= False,
                                          makePlot=True, titleStr=titleStr,
                                          xlabelStr= xlabelStr, 
                                          ylabelStr= ylabelStr,
                                          color=color, 
                                          outputFileName=outputFileName, 
                                          closeFigure=closeFigures)


    return None



def make_bending_angle_plots(datasets, exptName = '', distance_type = None,
                             bending_threshold_deg = 0.0,
                             color = 'black',
                             plot_type_2D = 'heatmap',
                             outputFileNameBase = 'pair_fish', 
                             outputFileNameExt = 'png',
                             closeFigures = False,
                             writeCSVs = False):
    
    
    """
    Makes several useful plots of bending angle properties for 
    of pairs of fish.
    
    Removed from make_pair_fish_plots()
        
    Inputs:
        datasets : dictionaries for each dataset
        exptName : (string) Experiment name, to append to titles.
        bending_threshold_deg : (float) for a plot of mean bending angle
                constrained to abs(angle) > threshold, use this threshold.
                Input in degrees. 
        distance_type : string, either closest_distance or head_head_distance
                Default is None; user should think about this!
        plot_each_dataset : (bool) if True, plot the prob. distr. for each array               
        color: plot color (uses alpha for indiv. dataset colors)
        plot_type_2D : str, 'heatmap' or 'line_plots'
                    Which plotting function make_2D_histogram() will use
                    ('heatmap' or 'line_plots')
        outputFileNameBase : base file name for figure output; if None,
                             won't save a figure file
        outputFileNameExt : extension for figure output (e.g. 'eps' or 'png')
        closeFigures : (bool) if True, close a figure after creating it.
        writeCSVs : (bool) Used by various functions; if true, output plotted 
                            points to a CSV file. See code for filenames

    Outputs:
        saved_pair_outputs : list, containing
            0 : bend_2Dhist_mean, mean 2D bending angle histogram
            1 : bend_2Dhist_std, std dev for 2D bending angle histogram
            2: bin positions ("X") for head_head_distance_mm for 2D bending angle histogram
            3: bin positions ("Y") for relative orientation for 2D bending angle histogram

    To do:
        Redundant code for slicing, symmetrization. Probably not worth cleaning up.
        
    """
    
    # Make sure of distance measure being used
    if not (distance_type==None or distance_type == 'closest_distance' or \
            distance_type == 'head_head_distance'):
        print('\nDistance measure must be "closest_distance" or "head_head_distance".\n')
        distance_type = None
    if distance_type == None:
        distance_type_choice = 0
        while not ((distance_type_choice  == 1) or (distance_type_choice  == 2)):
            distance_type_choice = int(input('Choose distance measure ' + 
                                             '\n  (1) closest_distance ' + 
                                             '\n  (2) head_head_distance' +
                                             '\nEnter "1" or "2": '))
        if distance_type_choice==1:
            distance_type = 'closest_distance'
        else:
            distance_type = 'head_head_distance'

    # Strings for file output, labels
    if distance_type == 'closest_distance':
        distance_file_string = 'closestDistance'
        distanceLabelStr = 'Closest Distance (mm)'
    elif distance_type == 'head_head_distance':
        distance_file_string = 'headHeadDistance'
        distanceLabelStr = 'Head-Head Distance (mm)'
    else:
        raise ValueError('Invalid distance type')
    
        
    saved_pair_outputs = []

    
    verifyPairs = True
    for j in range(len(datasets)):
        if datasets[j]["Nfish"] != 2:
            verifyPairs = False
    if verifyPairs==False:
        raise ValueError('Error in make_pair_fish_plots; Nfish must be 2 !')


    # 2D plot of mean bending angle vs. relative orientation and distance
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + \
            f'_bendAngle_{distance_file_string}_orientation_2D' + \
                '.' + outputFileNameExt
    else:
        outputFileName = None
    mask_by_sem_limit_degrees = 2.0 # show points with s.e.m. < this
    use_abs_value = (False, False)
    titleStr = f'{exptName}: Bend Angle; unc. < {mask_by_sem_limit_degrees:.1f} deg'
    # Save the output 2D histograms, for use later.
    bend_2Dhist_mean, X, Y, bend_2Dhist_sem = make_2D_histogram(
        datasets,
        keyNames = ('relative_orientation', f'{distance_type}_mm'),
        keyIdx = (None, None), 
        use_abs_value = use_abs_value,
        keyNameC = 'bend_angle', keyIdxC = None,
        colorRange = (-12*np.pi/180.0, 12*np.pi/180.0),
        dilate_minus1= False, 
        bin_ranges = ((-np.pi, np.pi), (0.0, 30.0)), Nbins = (19,15), 
        titleStr = titleStr,
        clabelStr= 'Mean Bending Angle (degrees)',
        xlabelStr = 'Relative Orientation (degrees)',
        ylabelStr = distanceLabelStr, 
        mask_by_sem_limit = mask_by_sem_limit_degrees*np.pi/180.0,
        unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
        cmap = 'RdYlBu_r', 
        plot_type = plot_type_2D,
        outputFileName = outputFileName,
        closeFigure = closeFigures)
    saved_pair_outputs.append(bend_2Dhist_mean)
    saved_pair_outputs.append(bend_2Dhist_sem)
    saved_pair_outputs.append(X)
    saved_pair_outputs.append(Y)

    # Slice bend angle binned by distance and orientation, along the
    # orientation axis, distance slice: distance < 2.5 mm
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + f'_bendAngle_v_orientation_small_{distance_file_string}' + '.' + outputFileNameExt
    else:
        outputFileName = None
    d_range = (0.0, 2.5)
    xlabelStr = 'Relative Orientation (deg)'
    titleStr = f'{exptName}: Bend Angle for d < {d_range[1]:.2f} mm'
    zlabelStr = 'Mean Bending Angle (degrees)'
    xlim = (-np.pi, np.pi)
    zlim = (-15*np.pi/180, 15*np.pi/180)
    color = color
    slice_2D_histogram(bend_2Dhist_mean, X, Y, bend_2Dhist_sem, 
                       slice_axis = 'x', other_range = d_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = distanceLabelStr, zlim = zlim, xlim = xlim, 
                       plot_z_zero_line = True,
                       plot_vert_zero_line = True,
                       unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                       color = color, outputFileName=outputFileName,
                       closeFigure=closeFigures)

    # Symmetrize the above bending angle / relative orientation graph,
    # taking theta[theta > 0] - theta[theta < 0]
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + f'_bendAngle_v_orientation_small_{distance_file_string}_asymm' + '.' + outputFileNameExt
    else:
        outputFileName = None
    midXind = int((X.shape[0] - 1)/2.0)
    if np.abs(X[midXind, 0]) > 1e-6:
        print('"X" array is not centered at zero. Will not symmetrize.')
    else:
        bend_2Dhist_mean_symm = 0.5*(bend_2Dhist_mean[midXind:,:] - 
                                     np.flipud(bend_2Dhist_mean[:(midXind+1),:]))
        bend_2Dhist_sem_symm = np.sqrt(bend_2Dhist_sem[midXind:,:]**2 +
                                       np.flipud(bend_2Dhist_sem[:(midXind+1),:])**2)/np.sqrt(2)
        X_symm = X[midXind:,:]
        Y_symm = Y[midXind:,:]
        slice_2D_histogram(bend_2Dhist_mean_symm, X_symm, Y_symm,
                           bend_2Dhist_sem_symm, 
                           slice_axis = 'x', other_range = d_range, 
                           titleStr = titleStr, xlabelStr = f'|{xlabelStr}|', 
                           zlabelStr = zlabelStr + ' toward Other',
                           ylabelStr = distanceLabelStr, zlim = zlim, 
                           xlim = (0.0, xlim[1]), 
                           plot_z_zero_line = True,
                           plot_vert_zero_line = False,
                           unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                           color = color, outputFileName=outputFileName,
                           closeFigure=closeFigures)

    # Slice along bend angle binned by distance and orientation, 
    # orientation axis, constrain distance: 5 mm < distance < 15 mm
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + f'_bendAngle_v_orientation_middle_{distance_file_string}' + '.' + outputFileNameExt
    else:
        outputFileName = None
    d_range = (5.0, 15.0)
    xlabelStr = 'Relative Orientation (deg)'
    titleStr = f'{exptName}: Bend Angle for {d_range[0]:.1f} < d < {d_range[1]:.1f} mm'
    zlabelStr = 'Mean Bending Angle (degrees)'
    xlim = (-np.pi, np.pi)
    zlim = (-15*np.pi/180, 15*np.pi/180)
    color = color
    slice_2D_histogram(bend_2Dhist_mean, X, Y, bend_2Dhist_sem, 
                       slice_axis = 'x', other_range = d_range, 
                       titleStr = titleStr, xlabelStr = xlabelStr, 
                       zlabelStr = zlabelStr,
                       ylabelStr = distanceLabelStr, zlim = zlim, xlim = xlim, 
                       plot_z_zero_line = True,
                       plot_vert_zero_line = True,
                       unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                       color = color, outputFileName=outputFileName,
                       closeFigure=closeFigures)

    # Symmetrize the above bending angle / relative orientation graph,
    # taking theta[theta > 0] - theta[theta < 0]
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + f'_bendAngle_v_orientation_middle_{distance_file_string}_asymm' + '.' + outputFileNameExt
    else:
        outputFileName = None
    midXind = int((X.shape[0] - 1)/2.0)
    if np.abs(X[midXind, 0]) > 1e-6:
        print('"X" array is not centered at zero. Will not symmetrize.')
    else:
        bend_2Dhist_mean_symm = 0.5*(bend_2Dhist_mean[midXind:,:] - 
                                     np.flipud(bend_2Dhist_mean[:(midXind+1),:]))
        bend_2Dhist_sem_symm = np.sqrt(bend_2Dhist_sem[midXind:,:]**2 +
                                       np.flipud(bend_2Dhist_sem[:(midXind+1),:])**2)/np.sqrt(2)
        X_symm = X[midXind:,:]
        Y_symm = Y[midXind:,:]
        slice_2D_histogram(bend_2Dhist_mean_symm, X_symm, Y_symm,
                           bend_2Dhist_sem_symm, 
                           slice_axis = 'x', other_range = d_range, 
                           titleStr = titleStr, xlabelStr = f'|{xlabelStr}|', 
                           zlabelStr = zlabelStr + ' toward Other',
                           ylabelStr = distanceLabelStr, zlim = zlim, 
                           xlim = (0.0, xlim[1]), 
                           plot_z_zero_line = True,
                           plot_vert_zero_line = False,
                           unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
                           color = color, outputFileName=outputFileName,
                           closeFigure=closeFigures)

    
    # 2D plot of mean bending angle vs. relative orientation and distance,
    # Constrained to bending angle > minimum threshold for "bending"
    # presumably params['bend_min_deg']
    if bending_threshold_deg > 0.0:
        if outputFileNameBase is not None:
            outputFileName = outputFileNameBase + \
                f'_bendAngle_above{bending_threshold_deg:.0f}deg_{distance_file_string}_orientation_2D' + \
                    '.' + outputFileNameExt
        else:
            outputFileName = None
        mask_by_sem_limit_degrees = 8.0 # show points with s.e.m. < this
        use_abs_value = (False, False)
        titleStr = f'{exptName}: Bend Angle >{bending_threshold_deg:.0f}deg; unc. < {mask_by_sem_limit_degrees:.1f}deg'
        # Save the output 2D histograms, for use later.
        bend_2Dhist_mean, X, Y, bend_2Dhist_sem = make_2D_histogram(
            datasets,
            keyNames = ('relative_orientation', f'{distance_type}_mm'),
            keyIdx = (None, None), 
            use_abs_value = use_abs_value,
            keyNameC = 'bend_angle', keyIdxC = None,
            colorRange = (-45*np.pi/180.0, 45*np.pi/180.0),
            dilate_minus1= False, 
            constraintKey = 'bend_angle', 
            constraintRange = ((np.pi/180.0)*bending_threshold_deg, np.inf), 
            constraintIdx = None, use_abs_value_constraint = True,
            bin_ranges = ((-np.pi, np.pi), (0.0, 30.0)), Nbins = (19,15), 
            titleStr = titleStr,
            clabelStr= 'Mean Bending Angle (degrees)',
            xlabelStr = 'Relative Orientation (degrees)',
            ylabelStr = distanceLabelStr, 
            mask_by_sem_limit = mask_by_sem_limit_degrees*np.pi/180.0,
            unit_scaling_for_plot = [180.0/np.pi, 1.0, 180.0/np.pi],
            cmap = 'RdYlBu_r', 
            plot_type = plot_type_2D,
            outputFileName = outputFileName,
            closeFigure = closeFigures)    
    
    return saved_pair_outputs

def make_rel_orient_rank_key(dataset, behavior_key_string):
    """
    Create new behavior keys based on relative orientation rank ordering
    and populate the "raw frames" sub-keys.
    
    For a single dataset, creates new keys like "behavior_key_string_lowRelOrient{m}" 
    where m is the rank index (0 = lowest relative orientation, 1 = highest).
    These new keys contain behavior events for whichever fish has rank m at the 
    time of each behavior event.
    
    If rel_orient_rankIdx key doesn't exist in dataset, create it based on simple
    rank ordering

    
    Parameters
    ----------
    dataset : dictionary containing behavior analysis, probably from datasets[j]
                Note that this only needs "raw frames", since the behavior dictionary
                will be made later. Must contain either "rel_orient_rankIdx"
                or 'relative_orientation' keys.
    behavior_key_string : string, base behavior name (e.g., "Rbend_Fish")
        The function will look for keys like "{behavior_key_string}{k}" where k
        are the fish indices found in rel_orient_rankIdx
    
    Returns
    -------
    None -- modifies dataset in place, creating dataset[new_key]["raw_frames"]
    
    """
     
    # Check that rel_orient_rankIdx exists
    if "rel_orient_rankIdx" not in dataset:
        print(f"Key 'rel_orient_rankIdx' not found in dataset {dataset['dataset_name']}")
        print('   Calculating this now...')
        dataset['rel_orient_rankIdx'] = np.argsort(np.abs(dataset['relative_orientation']), 
                                                   axis=1)
    rel_orient_rankIdx = dataset["rel_orient_rankIdx"]
    
    # Get unique fish indices from first frame
    unique_fish_indices = np.unique(rel_orient_rankIdx[0, :])
    # print(f"Dataset {dataset_name}: Found fish indices {unique_fish_indices}")
    
    # Check that corresponding behavior keys exist
    behavior_keys_fish = []
    for fish_idx in unique_fish_indices:
        behavior_key = f"{behavior_key_string}{fish_idx}"
        if behavior_key not in dataset:
            raise ValueError(f"Behavior key '{behavior_key}' not found in dataset {dataset['dataset_name']}")
        behavior_keys_fish.append(behavior_key)
    
    # Number of rank positions (should equal number of fish)
    n_ranks = len(unique_fish_indices)
    
    # Initialize new rank-based behavior dictionaries
    for rank_idx in range(n_ranks):
        new_key = f"{behavior_key_string}_lowRelOrient{rank_idx}"
        dataset[new_key] = {
            "behavior_name": new_key,
            "raw_frames": np.array([])
        }
    
    # Collect all behavior frames from all fish, with fish identity
    all_behavior_frames_with_fish = []
    for fish_idx in unique_fish_indices:
        behavior_key = f"{behavior_key_string}{fish_idx}"
        behavior_frames = dataset[behavior_key]["raw_frames"]
        for frame in behavior_frames:
            all_behavior_frames_with_fish.append((int(frame), fish_idx))
    
    # Sort by frame number
    all_behavior_frames_with_fish.sort(key=lambda x: x[0])
    
    # For each behavior frame, determine which rank that fish has
    for frame, fish_idx in all_behavior_frames_with_fish:
        # Get the rank ordering for this frame (convert to 0-based indexing)
        frame_idx = frame - 1
        
        if 0 <= frame_idx < len(rel_orient_rankIdx):
            # Find which rank position this fish occupies
            rank_ordering = rel_orient_rankIdx[frame_idx, :]
            
            # Find the rank of this fish (0 = lowest rel orient, etc.)
            fish_rank = np.where(rank_ordering == fish_idx)[0]
            
            if len(fish_rank) > 0:
                rank = fish_rank[0]
                
                # Add this frame to the appropriate rank-based behavior
                new_key = f"{behavior_key_string}_lowRelOrient{rank}"
                dataset[new_key]["raw_frames"] = np.append(
                    dataset[new_key]["raw_frames"], frame)
    
    for rank_idx in range(n_ranks):
        new_key = f"{behavior_key_string}_lowRelOrient{rank_idx}"
        
        # Sort raw_frames and convert to integers
        raw_frames = np.sort(dataset[new_key]["raw_frames"]).astype(int)
        dataset[new_key]["raw_frames"] = raw_frames
        
    
    
def make_rel_orient_rank_keys_allDatasets(datasets, behavior_key_string, 
                              keep_all_frames = False):
    """
    Create new behavior keys based on abs. value of relative orientation 
    rank ordering.
    NOTE: This function is to be used if behaviors have already been analyzed,
    and dataset keys already exist. For new datasets, these relative orientation
    keys are already calculated.
    See notes Sept. 2025; keeping sign of relative orientation Oct. 2025
    
    For each dataset, creates new keys like "behavior_key_string_lowRelOrient{m}" 
    where m is the rank index (0 = lowest relative orientation, 1 = highest).
    These new keys contain behavior events for whichever fish has rank m at the 
    time of each behavior event.
    
    Parameters
    ----------
    datasets : list of dictionaries containing all analysis
    behavior_key_string : string, base behavior name (e.g., "Rbend_Fish")
        The function will look for keys like "{behavior_key_string}{k}" where k
        are the fish indices found in rel_orient_rankIdx
    keep_all_frames : If True, no no frames will be removed. If false,
        remove bad tracking and edge rejection frames
    
    Returns
    -------
    None (modifies datasets in place)
    
    Raises
    ------
    ValueError : if rel_orient_rankIdx key doesn't exist or if expected 
                behavior keys don't exist
    """
    
    for j in range(len(datasets)):
        dataset = datasets[j]
        dataset_name = dataset.get("dataset_name", f"dataset_{j}")
        
        # Check that rel_orient_rankIdx exists
        if "rel_orient_rankIdx" not in dataset:
            print(f"Key 'rel_orient_rankIdx' not found in dataset {dataset_name}")
            print('   Calculating this now...')
            dataset['rel_orient_rankIdx'] = np.argsort(np.abs(dataset['relative_orientation']), 
                                                       axis=1)
        
        rel_orient_rankIdx = dataset["rel_orient_rankIdx"]
        
        # Get unique fish indices from first frame
        unique_fish_indices = np.unique(rel_orient_rankIdx[0, :])
        print(f"Dataset {j} ({dataset_name}): Found fish indices {unique_fish_indices}")
        
        # Check that corresponding behavior keys exist
        behavior_keys_fish = []
        for fish_idx in unique_fish_indices:
            behavior_key = f"{behavior_key_string}{fish_idx}"
            if behavior_key not in dataset:
                raise ValueError(f"Behavior key '{behavior_key}' not found in dataset {dataset_name}")
            behavior_keys_fish.append(behavior_key)
        
        # Number of rank positions (should equal number of fish)
        n_ranks = len(unique_fish_indices)
        
        # Initialize new rank-based behavior dictionaries
        for rank_idx in range(n_ranks):
            new_key = f"{behavior_key_string}_lowRelOrient{rank_idx}"
            dataset[new_key] = {
                "behavior_name": new_key,
                "raw_frames": np.array([]),
                "edit_frames": np.array([]),
                "combine_frames": np.array([[], []]),
                "N_events": 0,
                "total_duration": 0,
                "relative_duration": 0.0
            }
        
        # Collect all behavior frames from all fish, with fish identity
        all_behavior_frames_with_fish = []
        for fish_idx in unique_fish_indices:
            behavior_key = f"{behavior_key_string}{fish_idx}"
            behavior_frames = dataset[behavior_key]["raw_frames"]
            for frame in behavior_frames:
                all_behavior_frames_with_fish.append((int(frame), fish_idx))
        
        # Sort by frame number
        all_behavior_frames_with_fish.sort(key=lambda x: x[0])
        
        # For each behavior frame, determine which rank that fish has
        for frame, fish_idx in all_behavior_frames_with_fish:
            # Get the rank ordering for this frame (convert to 0-based indexing)
            frame_idx = frame - 1
            
            if 0 <= frame_idx < len(rel_orient_rankIdx):
                # Find which rank position this fish occupies
                rank_ordering = rel_orient_rankIdx[frame_idx, :]
                
                # Find the rank of this fish (0 = lowest rel orient, etc.)
                fish_rank = np.where(rank_ordering == fish_idx)[0]
                
                if len(fish_rank) > 0:
                    rank = fish_rank[0]
                    
                    # Add this frame to the appropriate rank-based behavior
                    new_key = f"{behavior_key_string}_lowRelOrient{rank}"
                    dataset[new_key]["raw_frames"] = np.append(
                        dataset[new_key]["raw_frames"], frame)
        
        # Process each new rank-based behavior key using existing helper functions
        for rank_idx in range(n_ranks):
            new_key = f"{behavior_key_string}_lowRelOrient{rank_idx}"
            
            # Sort raw_frames and convert to integers
            raw_frames = np.sort(dataset[new_key]["raw_frames"]).astype(int)
            dataset[new_key]["raw_frames"] = raw_frames
            
            # Use remove_frames() function to get edit_frames
            if keep_all_frames:
                dataset[new_key]["edit_frames"] = raw_frames.copy()
            else:
                frames_to_remove = (datasets[j]["edge_frames"]["raw_frames"],
                                             datasets[j]["bad_bodyTrack_frames"]["raw_frames"])
                dataset[new_key]["edit_frames"] = remove_frames(raw_frames, 
                                                                frames_to_remove)
            
            # Use combine_events() function to get combine_frames
            if len(dataset[new_key]["edit_frames"]) > 0:
                dataset[new_key]["combine_frames"] = combine_events(dataset[new_key]["edit_frames"])
                dataset[new_key]["N_events"] = dataset[new_key]["combine_frames"].shape[1]
                dataset[new_key]["total_duration"] = np.sum(dataset[new_key]["combine_frames"][1, :])
            else:
                dataset[new_key]["combine_frames"] = np.array([[], []]).reshape(2, 0)
                dataset[new_key]["N_events"] = 0
                dataset[new_key]["total_duration"] = 0
            
            # Calculate relative duration
            if "Nframes" in dataset and dataset["Nframes"] > 0:
                dataset[new_key]["relative_duration"] = (
                    dataset[new_key]["total_duration"] / dataset["Nframes"])
            else:
                dataset[new_key]["relative_duration"] = 0.0
            
            #print(f"  Created {new_key}: {dataset[new_key]['N_events']} events, "
            #      f"total duration {dataset[new_key]['total_duration']} frames")
    
    print(f"\nSuccessfully created relative orientation rank keys for "
          f"behavior '{behavior_key_string}' across {len(datasets)} datasets.")
    
    return None
    
def recalculate_angles(all_position_data, datasets, CSVcolumns, 
                      keys_to_modify=["relative_orientation"]):
    """
    Recalculate angle-related quantities in datasets.
    
    Useful for updating old pickle files after modifications to angle 
    calculation functions (e.g., changing from unsigned to signed angles).
    
    Parameters
    ----------
    all_position_data : list of numpy arrays
        Position data for all datasets
    datasets : list of dictionaries
        All datasets
    CSVcolumns : dictionary
        CSV column information
    keys_to_modify : list of str
        List of keys to recalculate. Must be subset of:
        ["relative_orientation", "bend_angle", "heading_angle"]
        Default: ["relative_orientation"]
    
    Returns
    -------
    datasets : list of dictionaries
        Updated datasets with recalculated values
    
    Notes
    -----
    - If "relative_orientation" is in keys_to_modify:
      Recalculates datasets[j]["relative_orientation"] and 
      datasets[j]["rel_orient_rankIdx"] using get_relative_orientation()
      Also calculates datasets[j]["relative_orientation_sum"] and
      datasets[j]["relative_orientation_abs_sum"]
    
    - If "heading_angle" is in keys_to_modify:
      Recalculates datasets[j]["heading_angle"] using repair_heading_angles()
    
    - If "bend_angle" is in keys_to_modify:
      Recalculates datasets[j]["bend_angle"] using calc_bend_angle()
    """
    
    # Validate keys_to_modify
    valid_keys = ["relative_orientation", "bend_angle", "heading_angle"]
    for key in keys_to_modify:
        if key not in valid_keys:
            raise ValueError(f"Invalid key '{key}' in keys_to_modify. " + 
                           f"Must be subset of {valid_keys}")
    
    print(f"\nRecalculating angles for keys: {keys_to_modify}")
    
    # Recalculate heading_angle if requested
    if "heading_angle" in keys_to_modify:
        print("  Recalculating heading_angle for all datasets...")
        datasets = repair_heading_angles(all_position_data, datasets, CSVcolumns)
        print("  Done.")
    
    # Recalculate relative_orientation if requested
    if "relative_orientation" in keys_to_modify:
        print("  Recalculating relative_orientation for all datasets...")
        for j in range(len(datasets)):
            datasets[j]["relative_orientation"], \
                datasets[j]["rel_orient_rankIdx"], \
                datasets[j]["relative_orientation_sum"], \
                datasets[j]["relative_orientation_abs_sum"] = \
                get_relative_orientation(all_position_data[j], 
                                         datasets[j], CSVcolumns)
        print("  Done.")
    
    # Recalculate bend_angle if requested
    if "bend_angle" in keys_to_modify:
        print("  Recalculating bend_angle for all datasets...")
        for j in range(len(datasets)):
            datasets[j]["bend_angle"] = calc_bend_angle(all_position_data[j], 
                                                        CSVcolumns)
        print("  Done.")
    
    print("All angle recalculations complete.\n")
    
    return datasets
