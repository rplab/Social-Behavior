# -*- coding: utf-8 -*-
# behavior_identification_single.py
"""
Author:   Raghuveer Parthasarathy
Split from behavior_identification.py on July 22, 2024
Last modified Nov. 12, 2025 -- Raghu Parthasarathy

Description
-----------

Module containing all behavior identification or characterization functions
that apply to single fish:
    - Fish length
    - Radial position
    - Polar angle
    - Radial alignment angle
    - Fish speed
    - Bout statistics
    - C-bend
    - J-bend
    - "is Moving"
    - average_bout_trajectory_allSets() and average_bout_trajectory_oneSet()
    - (and more)
Also a function that makes several useful "single fish" plots -- i.e. 
plots of characteristics of individual fish, which may be in multi-fish 
experiments
    - make_single_fish_plots()

"""

import numpy as np
import matplotlib.pyplot as plt
from toolkit import make_frames_dictionary, dilate_frames, wrap_to_pi,\
    combine_all_values_constrained, get_values_subset, \
    calculate_value_corr_all, fit_gaussian_mixture
from IO_toolkit import plot_probability_distr, plot_function_allSets

def get_coord_characterizations(all_position_data, datasets, 
                                CSVcolumns, expt_config, params):
    """
    For each dataset, simple coordinate characterizations
        (polar coordinates, radial alignment)
    
    Inputs:
        all_position_data : basic position information for all datasets, list of numpy arrays
        datasets : all datasets, info, list of dictionaries 
        CSVcolumns : CSV column information (dictionary)
        expt_config : dictionary of configuration information
        params : dictionary of all analysis parameters
    Returns:
        datasets : dictionaries for each dataset. Now with coord keys.
    """
    
    # Number of datasets
    N_datasets = len(datasets)
    print('Coordinate characterizations: ')
    for j in range(N_datasets):
        print('    for Dataset: ', datasets[j]["dataset_name"])
        
        # Get the radial position (distance to center) of each fish in each
        # frame.
        polar_coords = get_polar_coords(all_position_data[j], 
                                          CSVcolumns, 
                                          datasets[j]["arena_center"], 
                                          datasets[j]["image_scale"])
        datasets[j]["radial_position_mm"] = polar_coords[0]
        datasets[j]["polar_angle_rad"] = polar_coords[1]
        
        # Get the radial alignment angle, i.e. the heading angle relative 
        # to the radial vector. Put in [-pi, pi]
        radial_alignment = datasets[j]["polar_angle_rad"] - \
                           datasets[j]["heading_angle"]
        datasets[j]["radial_alignment_rad"] = wrap_to_pi(radial_alignment)
    return datasets


def get_single_fish_characterizations(all_position_data, datasets, CSVcolumns,
                                             expt_config, params):
    """
    For each dataset, characterizations that involve single fish
        (e.g. bending angle, C-bend or J-bend, speed, mean fish length)
        Fish length calculation for each frame
           moved out of this, to allow double-length repair
    Inputs:
        all_position_data : basic position information for all datasets, list of numpy arrays
        datasets : all datasets, info, list of dictionaries 
        CSVcolumns : CSV column information (dictionary)
        expt_config : dictionary of configuration information
        params : dictionary of all analysis parameters
    Returns:
        datasets : dictionaries for each dataset. datasets[j] contains
                    all the information for dataset j.
    """
    
    # Number of datasets
    N_datasets = len(datasets)
    
    print('Single-fish characterizations: ')
    for j in range(N_datasets):
        print('    for Dataset: ', datasets[j]["dataset_name"])
                
        # Get the speed of each fish in each frame (frame-to-frame
        # displacement of head position, um/s); 0 for first frame
        # Nframes x 2 array
        datasets[j]["speed_array_mm_s"] = \
            get_fish_speeds(all_position_data[j], CSVcolumns, 
                            datasets[j]["image_scale"], expt_config['fps'])
            
        # Frames with speed above threshold (i.e. moving fish)
        # "_each" is a dictionary with a key for each fish, 
        #     each containing a numpy array of frames in which that fish is moving
        # "_any" is a numpy array of frames in which any fish is moving
        # "_all" is a numpy array of frames in which all fish are moving
        isMoving_frames_each, isMoving_frames_any, isMoving_frames_all = \
            get_isMoving_frames(datasets[j], 
                                params["motion_speed_threshold_mm_second"])

        # Get the angular speed of each fish in each frame (frame-to-frame
        # change in heading angle, abs. val., rad/s); 0 for first frame
        # Nframes x 2 array
        datasets[j]["angular_speed_array_rad_s"] = \
            get_fish_angular_speeds(datasets[j])
        
        # Frames that are close to the edge of the dish 
        # (weak "edge_proximity_threshold_mm" criterion)
        close_to_edge_frames_each, close_to_edge_frames_any, \
            close_to_edge_frames_all = get_isCloseToEdge_frames(datasets[j], 
                                                                params["edge_proximity_threshold_mm"], 
                                                                expt_config['arena_radius_mm'])
        # For edge closeness indicator, for each fish and for "any" and "all" fish,
        # make a dictionary containing frames, removing frames with 
        # "bad" elements. Like "isMoving" arrays
        iCloseEdgeEach = [close_to_edge_frames_each[key] for key in sorted(close_to_edge_frames_each.keys())]
        iCloseEdge_arrays = iCloseEdgeEach + [close_to_edge_frames_any] + [close_to_edge_frames_all] 
        # Create the second tuple of strings
        iCloseEdge_keys = tuple(f'close_to_edge_Fish{i}' for i in range(len(iCloseEdgeEach))) \
            + ('close_to_edge_any',)  + ('close_to_edge_all', )
        for iCL_key, iCL_array in zip(iCloseEdge_keys, iCloseEdge_arrays):
            datasets[j][iCL_key] = make_frames_dictionary(iCL_array,
                                          np.array(datasets[j]["bad_bodyTrack_frames"]["raw_frames"]).astype(int),
                                          behavior_name = iCL_key,
                                          Nframes=datasets[j]['Nframes'])

            
        # Calculate the bending angle: supplement of the angle between 
        # the front half and the back half
        datasets[j]["bend_angle"] = calc_bend_angle(all_position_data[j], CSVcolumns)

        isBending_frames_each, isBending_frames_any, isBending_frames_all = \
            get_isBending_frames(datasets[j], params["bend_min_deg"])
            
        # Calculate "isActive" for each fish as *either* moving or bending
        isActive_frames_each = {}
        for k in isMoving_frames_each.keys():
            # Concatenate the two arrays of frames
            combined_frames = np.concatenate([isMoving_frames_each[k], 
                                              isBending_frames_each[k]])
            # Get unique frames and store in new dictionary
            isActive_frames_each[k] = np.unique(combined_frames)     
        isActive_frames_any = np.unique(np.concatenate([isMoving_frames_any, 
                                             isBending_frames_any]))
        isActive_frames_all = np.unique(np.concatenate([isMoving_frames_all, 
                                             isBending_frames_all]))
            
        # For movement indicator, and activity indicator,
        # for each fish and for "any" and "all" fish,
        # make a dictionary containing frames, removing frames with 
        # "bad" elements. Unlike other characteristics, need to "dilate"
        # bad tracking frames, since each bad tracking frame j affects 
        # speed j and speed j+1.
        # Also makes a 2xN array of initial frames and durations, as usual
        # for these dictionaries
        # Code to allow an arbitrary number of fish, and consequently an 
        #    arbitrary number of elements isMoving_frames_each dictionary
        iMe_arrays = [isMoving_frames_each[key] for key in sorted(isMoving_frames_each.keys())]
        iAe_arrays = [isActive_frames_each[key] for key in sorted(isActive_frames_each.keys())]
        iMA_arrays = iMe_arrays + [isMoving_frames_any] + [isMoving_frames_all] \
                   + iAe_arrays + [isActive_frames_any] + [isActive_frames_all]
        # Create the second tuple of strings
        iMA_keys = tuple(f'isMoving_Fish{i}' for i in range(len(isMoving_frames_each))) \
            + ('isMoving_any',)  + ('isMoving_all', ) \
            + tuple(f'isActive_Fish{i}' for i in range(len(isActive_frames_each))) \
            + ('isActive_any',)  + ('isActive_all', ) 

        for iMA_key, iMA_array in zip(iMA_keys, iMA_arrays):
            badTrFramesRaw = np.array(datasets[j]["bad_bodyTrack_frames"]["raw_frames"]).astype(int)
            dilate_badTrackFrames = dilate_frames(badTrFramesRaw, 
                                                  dilate_frames=np.array([1]))
            datasets[j][iMA_key] = make_frames_dictionary(iMA_array,
                                          dilate_badTrackFrames,
                                          behavior_name = iMA_key,
                                          Nframes=datasets[j]['Nframes'])

        # For bending indicator, for each fish and for "any" and "all" fish,
        # make a dictionary containing frames, removing frames with 
        # "bad" elements. Like "isMoving" arrays
        iBe_arrays = [isBending_frames_each[key] for key in sorted(isBending_frames_each.keys())]
        iB_arrays = iBe_arrays + [isBending_frames_any] + [isBending_frames_all] 
        # Create the second tuple of strings
        iB_keys = tuple(f'isBending_Fish{i}' for i in range(len(isBending_frames_each))) \
            + ('isBending_any',)  + ('isBending_all', )

        for iB_key, iB_array in zip(iB_keys, iB_arrays):
            datasets[j][iB_key] = make_frames_dictionary(iB_array,
                                          np.array(datasets[j]["bad_bodyTrack_frames"]["raw_frames"]).astype(int),
                                          behavior_name = iB_key,
                                          Nframes=datasets[j]['Nframes'])
            
        # Get frames with the various categories of bends
        # C-bends, R-bends, and J-bends; each is a dictionary with Nfish 
        # keys, one for each fish, each containing a numpy array of frames
        # Note that "isBending" (minimal threshold) has already been determined
        
        Jbend_frames_each, Rbend_frames_each, Cbend_frames_each = \
            get_bend_behavior_frames(datasets[j], params)
        
        # numpy array of frames with C-bend for *any* fish
        Cbend_frames_any = np.unique(np.concatenate(Cbend_frames_each))
        Rbend_frames_any = np.unique(np.concatenate(Rbend_frames_each))
        Jbend_frames_any = np.unique(np.concatenate(Jbend_frames_each))
        
        # For C-bends, R-bends, and J-bends, for each fish and for any fish,
        # make a dictionary containing frames, 
        # remove frames with "bad" elements
        # Also makes a 2xN array of initial frames and durations, though
        # the duration of bends is probably 1 almost always
        # Code to allow an arbitrary number of fish, and consequently an 
        #    arbitrary number of elements in the Cbend_frames_each, 
        #    RBend_frames_each, and Jbend_frames_each dictionaries

        CRJ_arrays = Cbend_frames_each + [Cbend_frames_any] \
                     + Rbend_frames_each + [Rbend_frames_any] \
                     + Jbend_frames_each + [Jbend_frames_any] 

        # Create the second tuple of strings
        CRJ_keys = tuple(f'Cbend_Fish{i}' for i in range(len(Cbend_frames_each))) + ('Cbend_any',) \
                  + tuple(f'Rbend_Fish{i}' for i in range(len(Rbend_frames_each))) + ('Rbend_any',) \
                  + tuple(f'Jbend_Fish{i}' for i in range(len(Jbend_frames_each))) + ('Jbend_any',)

        for CRJ_key, CRJ_array in zip(CRJ_keys, CRJ_arrays):
            datasets[j][CRJ_key] = make_frames_dictionary(CRJ_array,
                                          datasets[j]["bad_bodyTrack_frames"]["raw_frames"],
                                          behavior_name = CRJ_key,
                                          Nframes=datasets[j]['Nframes'])
        # Average speed (averaged over fish), and average speed only in 
        # moving frames. get_mean_speed() will dilate bad tracks by +1
        speed_mean_all, speed_mean_moving = \
            get_mean_speed(datasets[j]["speed_array_mm_s"], 
                           isMoving_frames_each, datasets[j]["bad_bodyTrack_frames"]["raw_frames"])
        # average over fish, since ID is unreliable
        datasets[j]["speed_mm_s_mean"] = np.mean(speed_mean_all)
        datasets[j]["speed_whenMoving_mm_s_mean"] = np.mean(speed_mean_moving)


        # Bout statistics.
        bouts_N, bout_duration_s,  bout_rate_bpm, bout_ibi_s = \
            get_bout_statistics(datasets[j])
        # average over fish, since ID is unreliable
        datasets[j]["bouts_N"] = np.mean(bouts_N)
        datasets[j]["bout_duration_s"] = np.mean(bout_duration_s)
        datasets[j]["bout_rate_bpm"] = np.mean(bout_rate_bpm)
        datasets[j]["bout_ibi_s"] = np.mean(bout_ibi_s)

        # Get tail angle of each fish in each frame 
        # Nframes x Nfish array
        # (Oct. 20, 2024: Not doing anything with this.)
        datasets[j]["tail_angle_rad"] = \
            getTailAngle(all_position_data[j], CSVcolumns, 
                         datasets[j]["heading_angle"])
       
        # For each dataset, exclude bad tracking frames from calculation of
        # the mean fish length. 
        # Also use Gaussian mixture model to get mean fish length (in good frames)
        print('Removing bad frames from stats for fish length, for Dataset: ', 
              datasets[j]["dataset_name"])
        goodIdx = np.where(np.in1d(datasets[j]["frameArray"], 
                                   datasets[j]["bad_bodyTrack_frames"]["raw_frames"], 
                                   invert=True))[0]
        goodLengthArray = datasets[j]["fish_length_array_mm"][goodIdx]
        datasets[j]["fish_length_mm_mean"] = np.mean(goodLengthArray)
        print(f'   Mean fish length: {datasets[j]["fish_length_mm_mean"]:.3f} mm')
        # Fit gmm
        datasets[j]["fish_length_mm_gmm_mean"] = get_fish_lengths_mean_gmm(datasets[j]) 

    return datasets

def get_fish_lengths_mean_gmm(dataset, verbose = False):
    """
    Fit each fish's length data, for good tracking frames only, to 
    a Gaussian mixture model and return the mean of the component near 
    the median value.
    
    input:
        dataset : single dataset (probably datasets[j]))
        verbose : if True, print lengths
    return:
        numpy array of shape (Nfish, 1) containing the mean length, mm
    """
    goodIdx = np.where(np.in1d(dataset["frameArray"], 
                               dataset["bad_bodyTrack_frames"]["raw_frames"], 
                               invert=True))[0]
    goodLengthArray = dataset["fish_length_array_mm"][goodIdx]
    mean_length_mm_gmm = np.zeros((dataset["Nfish"],))
    for k in range(dataset["Nfish"]):
        length_med = np.median(goodLengthArray[:,k])
        init_means = [length_med, 0.75*length_med, 3.0*length_med]
        mean_length_mm_gmm[k] = \
            fit_gaussian_mixture(goodLengthArray[:,k], n_gaussian=3, 
                                 init_means=init_means)[0]
        if verbose:
            # Note [k,0] needed for formatting to work
            print(f'   Mean fish length from gmm for fish {k}: ' + 
                  f'{mean_length_mm_gmm[k,0]:.3f} mm')
        
    return mean_length_mm_gmm
    
def get_fish_lengths(position_data, image_scale, CSVcolumns):
    """
    Get the length of each fish in each frame (sum of all segments)
    Input:
        position_data : basic position information for this dataset, numpy array
        image_scale : scale, um/px; from dataset["image_scale"]
        CSVcolumns : CSV column information (dictionary)
    Output
        fish_lengths : (mm) Nframes x Nfish array of fish lengths
    """
    xstart = int(CSVcolumns["body_column_x_start"])
    xend =int(CSVcolumns["body_column_x_start"])+int(CSVcolumns["body_Ncolumns"])
    ystart = int(CSVcolumns["body_column_y_start"])
    yend = int(CSVcolumns["body_column_y_start"])+int(CSVcolumns["body_Ncolumns"])
    dx = np.diff(position_data[:,xstart:xend,:], axis=1)
    dy = np.diff(position_data[:,ystart:yend,:], axis=1)
    dr = np.sqrt(dx**2 + dy**2)
    fish_lengths = np.sum(dr,axis=1)*image_scale/1000.0 # mm
    return fish_lengths
            
def get_fish_speeds(position_data, CSVcolumns, image_scale, fps):
    """
    Get the speed of each fish in each frame (frame-to-frame
        displacement of head position)
    Assign to the later frame; zeros for the first frame.
    Input:
        position_data : basic position information for this dataset, numpy array
        CSVcolumns : CSV column information (dictionary)
        image_scale : scale, um/px; from dataset["image_scale"]
        fps : frames/second, from expt_config
    Output
        speed_array_mm_s  : (mm/s) Nframes x Nfish array
    """
    head_x = position_data[:,CSVcolumns["head_column_x"],:] # x, both fish
    head_y = position_data[:,CSVcolumns["head_column_y"],:] # y, both fish
    # Frame-to-frame speed, px/frame
    dr = np.sqrt((head_x[1:,:] - head_x[:-1,:])**2 + 
                    (head_y[1:,:] - head_y[:-1,:])**2)
    speed_array_mm_s = dr*image_scale*fps/1000.0
    # to make Nframes x Nfish set as 0 for the first frame
    speed_array_mm_s = np.append(np.zeros((1, position_data.shape[2])), 
                                 speed_array_mm_s, axis=0)
    
    return speed_array_mm_s

def get_fish_angular_speeds(dataset):
    """
    Get the angular speed of each fish in each frame (abs val of frame-to-frame
        difference in heading angle), in [0, pi]
    Assign to the later frame; zeros for the first frame.
    Input:
        dataset : single dictionary containing the analysis data, including
                    keys "heading_angle" and "fps"
    Output
        angular_speed_array_rad_s  : (mm/s) Nframes x Nfish array
    """
    delta_angle = np.diff(dataset["heading_angle"], axis=0)
    # wrap to [-pi, pi]
    delta_angle = (delta_angle + np.pi) % (2.0*np.pi) - np.pi
    angular_speed_rad_fr = np.abs(delta_angle)
    
    angular_speed_array_rad_s = angular_speed_rad_fr * dataset["fps"]
    # to make Nframes x Nfish set as 0 for the first frame
    angular_speed_array_rad_s = np.append(np.zeros((1, angular_speed_rad_fr.shape[1])), 
                                 angular_speed_array_rad_s, axis=0)
    
    return angular_speed_array_rad_s

def get_isMoving_frames(dataset, motion_speed_threshold_mm_s = 10.0):
    """ 
    Find frames in which one or more fish have an above-threshold speed.
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        motion_speed_threshold_mm_s : speed threshold, mm/s
    Output : 
        isMoving_frames...
           _each:  dictionary with a key for each fish, 
                   each containing a numpy array of frames in which 
                   that fish is moving
           _any : numpy array of frames in which any fish is moving
           _all : numpy array of frames in which all fish are moving
    """
    
    speed_data = dataset["speed_array_mm_s"] # Nframes x Nfish array of speeds
    Nfish = dataset["Nfish"]
    isMoving = speed_data > motion_speed_threshold_mm_s

    # Dictionary containing "is moving" frames for each fish
    isMoving_frames_each = {}
    for fish in range(Nfish):
        isMoving_frames_each[fish] = dataset['frameArray'][isMoving[:, fish]] 
        
    # any fish
    any_fish_moving  = np.any(isMoving, axis=1)
    isMoving_frames_any = dataset['frameArray'][any_fish_moving]
    
    # all fish
    all_fish_moving  = np.all(isMoving, axis=1)
    isMoving_frames_all = dataset['frameArray'][all_fish_moving]

    return isMoving_frames_each, isMoving_frames_any, isMoving_frames_all

def get_mean_speed(speed_array_mm_s, isMoving_frames_each, badTrackFrames):
    """
    Calculate mean fish speed, ignoring bad-tracking frames
    Note that if *any* fish has bad tracking, the frame is ignored for *all* fish
    Returns mean, and mean only for frames that meet the isMoving criterion
    *NOT* averaged over fish -- do this externally if needed.
    
    Inputs:
        speed_array_mm_s : speed of each fish, frame-to-frame, mm/second,
                            probably input from datasets[j]["speed_array_mm_s"]
        isMoving_frames_each : Dictionary containing "is moving" frames
                                for each fish. Note index = frame no - 1
        badTrackFrames : frames with bad tracking, probably input as
                         datasets[j]["bad_bodyTrack_frames"]["raw_frames"]

    Returns: 
        Tuple of
        - speed_mean_all: (1 x Nfish) mean speed, mm/s
        - speed_mean_moving: (1 x Nfish) mean speed for "isMoving" frames, mm/s
    """
    Nframes = speed_array_mm_s.shape[0]
    frames = np.arange(1, Nframes+1) # all frame numbers
    Nfish = speed_array_mm_s.shape[1] # number of fish
    
    badTrackFrames = np.array(badTrackFrames).astype(int)
    dilate_badTrackFrames = np.concatenate((badTrackFrames,
                                           badTrackFrames - 1))
    bad_frames_set = set(dilate_badTrackFrames) # faster lookup -- caution use "list" for "isin"
    
    # Calculate mean speed excluding bad tracking frames
    good_frames_mask = np.isin(frames, list(bad_frames_set), invert=True)
    speed_mean_all = np.mean(speed_array_mm_s[good_frames_mask, :], axis=0).reshape(1, Nfish)
    
    # Calculate mean speed for moving frames, excluding bad tracking frames
    speed_mean_moving = np.zeros((1, Nfish))
    for j in range(Nfish):
        moving_frames = set(isMoving_frames_each[j]) - bad_frames_set
        moving_mask = np.isin(frames, list(moving_frames))
        speed_mean_moving[0, j] = np.mean(speed_array_mm_s[moving_mask, j])
    
    return speed_mean_all, speed_mean_moving


def get_isCloseToEdge_frames(dataset, edge_proximity_threshold_mm, arena_radius_mm):
    """ 
    Find frames in which fish are close to the edge (not for rejection of behaviors).
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        edge_proximity_threshold_mm : edge-closeness threshold, mm
        expt_config['arena_radius_mm'] :arena_radius in mm
    Output : 
        close_to_edge_frames...
           _each:  dictionary with a key for each fish, 
                   each containing a numpy array of frames in which 
                   that fish is close to the edge
           _any : numpy array of frames in which any fish is close to the edge
           _all : numpy array of frames in which all fish are close to the edge
    """
    
    Nfish = dataset["Nfish"]
    r_mm = dataset["radial_position_mm"] # distance from center, mm, each fish
    
    # True if close to edge
    close_to_edge = (arena_radius_mm - r_mm) < edge_proximity_threshold_mm
    
    # Dictionary containing "close_to_edge" frames for each fish
    close_to_edge_frames_each = {}
    for fish in range(Nfish):
        close_to_edge_frames_each[fish] = dataset['frameArray'][close_to_edge[:, fish]] 
        
    # any fish
    any_fish_close  = np.any(close_to_edge, axis=1)
    close_to_edge_frames_any = dataset['frameArray'][any_fish_close]
    
    # all fish
    all_fish_close  = np.all(close_to_edge, axis=1)
    close_to_edge_frames_all = dataset['frameArray'][all_fish_close]

    return close_to_edge_frames_each, close_to_edge_frames_any, close_to_edge_frames_all



def getTailAngle(position_data, CSVcolumns, heading_angles):
    """
    Calculate the tail angle:  angle of last two body points 
        relative to heading angle (zero if aligned); 
        absolute value, radians.
    Input:
        position_data : basic position information for this dataset, numpy array
        CSVcolumns : CSV column information (dictionary)
        heading_angles : all heading angles (Nframes x Nfish); dataset[j]["heading_angles"]
    Output
        tail_angle  : (rad) Nframes x Nfish array
    """
    # Final tail positions, for each frame, for each fish
    # Each array is Nframes x (2 body points) x (Nfish==2)
    tailpos_x = position_data[:,CSVcolumns["body_column_x_start"] + CSVcolumns["body_Ncolumns"] - 3: \
                           CSVcolumns["body_column_x_start"] + CSVcolumns["body_Ncolumns"] - 1,:]
    tailpos_y = position_data[:,CSVcolumns["body_column_y_start"] + CSVcolumns["body_Ncolumns"] - 3: \
                           CSVcolumns["body_column_y_start"] + CSVcolumns["body_Ncolumns"] - 1,:]
    
    # tail angle, lab frame
    dy = np.diff(tailpos_y, axis=1).squeeze()
    dx = np.diff(tailpos_x, axis=1).squeeze()
    tail_angle_lab = np.arctan2(dy, dx).squeeze()
    # flip to align with heading angle
    tail_angle_lab = tail_angle_lab + np.pi
    # Make 2D, to work with single-fish data
    if tail_angle_lab.ndim == 1:
        tail_angle_lab = tail_angle_lab.reshape(-1, 1)
    
    # tail angle relative to heading angle
    tail_angle = np.abs(tail_angle_lab - heading_angles)

    return tail_angle

def getTailCurvature(position_data, CSVcolumns, image_scale):
    """
    Calculate the curvature of the fish posterior (last 5 body
        datapoints) at each frame, for each fish.
    
    NOTE: Not used; not robust!
    
    Input:
        position_data : basic position information for this dataset, numpy array
        CSVcolumns : CSV column information (dictionary)
        image_scale : scale, um/px; from dataset["image_scale"]
    Output
        curvature_invmm  : (1/mm) Nframes x Nfish array
    """
    
    # All tail positions, for each frame, for each fish
    # Each array is Nframes x (Nbodypts/2==5) x (Nfish==2)
    tailpos_x = position_data[:,CSVcolumns["body_column_x_start"] : \
                           CSVcolumns["body_column_x_start"] + int(CSVcolumns["body_Ncolumns"]/2),:]
    tailpos_y = position_data[:,CSVcolumns["body_column_y_start"] : \
                           CSVcolumns["body_column_y_start"] + int(CSVcolumns["body_Ncolumns"]/2),:]
    
    # Calculate mean curvature. I'm sure there's a way to vectorize
    # this, but I won't bother. (Claude fails at this...)
    curvatures = np.zeros((tailpos_x.shape[0], tailpos_x.shape[2]))
    
    for frame in range(tailpos_x.shape[0]):
        for fish in range(tailpos_x.shape[2]):
            x = tailpos_x[frame, :, fish]
            y = tailpos_y[frame, :, fish]
            
            # bypass if all zeros (bad tracking)
            if np.min(x) > 0.0:
                # Flip if needed to avoid Rank warning for polyfit
                if np.var(x) > np.var(y):
                    # Fit a quadratic function (y = ax^2 + bx + c)
                    coeffs = np.polyfit(x, y, 2)
                    a, b, _ = coeffs
                    # curvature at each point
                    curvature = np.abs(2 * a) / (1 + (2 * a * x + b)**2)**(3/2)
                else:
                    # Fit a quadratic function (y = ax^2 + bx + c)
                    coeffs = np.polyfit(y, x, 2)
                    a, b, _ = coeffs
                    # curvature at each point
                    curvature = np.abs(2 * a) / (1 + (2 * a * y + b)**2)**(3/2)
                    
                # mean curvature 
                curvatures[frame, fish] = np.mean(curvature)
                
    curvature_invmm = curvatures / image_scale
    
    return curvature_invmm

def get_bout_statistics(dataset):
    """
    Calculate basic properties of bouts, defined as frames in which the
    "moving" or "bending" criterion is met (isActive_Fish{j})
    Note that "isActive" already disregards bad tracking frames
    Calculates:
        # Number of bouts for each fish
		# Mean Duration of bouts for each fish k, seconds
		# bout rate (bouts per minute) for each fish
		# Mean inter-bout-interval (s) for each fish

    *NOT* averaged over fish -- do this externally if needed.
    
    Inputs:
        dataset : single dictionary containing the analysis data, including
                    key "isActive_Fish{j}" for each fish j; also fps and
                    total duration
    Outputs: 
        Tuple of
        - bouts_N: (Nfish,) Number of bouts 
        - bout_duration_s: (Nfish,)  Mean Duration of bouts, seconds
        - bout_rate_bpm: (Nfish,) bout rate (bouts per minute)
        - bout_ibi_s: (Nfish,) mean inter-bout interval (seconds)

    """
    Nfish = dataset["Nfish"]
    bouts_N = np.zeros((Nfish,))
    bout_duration_s = np.zeros((Nfish,))
    bout_rate_bpm = np.zeros((Nfish,))
    bout_ibi_s = np.zeros((Nfish,))
  
    for k in range(Nfish):
        # Start frames and durations for bouts (i.e. motion)
        moving_frameInfo = dataset[f"isActive_Fish{k}"]["combine_frames"]

        # Number of bouts for fish k
        bouts_N[k] = moving_frameInfo.shape[1]
      
        # Mean Duration of bouts for fish k, seconds
        bout_duration_s[k] = np.mean(moving_frameInfo[1,:]) / dataset["fps"]
  		
        # bout rate (bouts per minute) for each fish
        bout_rate_bpm[k] = bouts_N[k] *60.0 / dataset["total_time_seconds"]
  		
        # Mean inter-bout-interval (s) for each fish
        endFrames = moving_frameInfo[0,:] + moving_frameInfo[1,:] - 1
        bout_ibi_s[k] = np.mean(moving_frameInfo[0,1:]-endFrames[:-1]) / dataset["fps"]

    return bouts_N, bout_duration_s, bout_rate_bpm, bout_ibi_s
    
def get_polar_coords(position_data, CSVcolumns, arena_center, image_scale):
    """
    Get the fish head position in polar coordinates relative to the
        arena center, for each fish in each frame.
        "y" defined as decreasing downward, so polar angle = atan(y,x)
        increases Clockwise from East.
    Input:
        position_data : basic position information for this dataset, numpy array
        CSVcolumns : CSV column information (dictionary)
        arena_center : tuple of (x,y) positions of the Arena Center,
                       probably stored in dataset["arena_center"]
        image_scale : scale, um/px; from dataset["image_scale"]
    Output
        polar_coords : tuple of 
            radial_position_mm  : (mm) Nframes x Nfish array
            polar_angle_rad  : (radians) Nframes x Nfish array
    """
    head_x = position_data[:,CSVcolumns["head_column_x"],:] # x, all fish, Nframes x Nfish
    head_y = position_data[:,CSVcolumns["head_column_y"],:] # y, all fish, Nframes x Nfish
    # Radial position, px
    dx = head_x - arena_center[0]
    dy = head_y - arena_center[1]
    r = np.sqrt(dx**2 + dy**2)
    radial_position_mm = r*image_scale/1000.0
    polar_angle_rad = np.arctan2(dy, dx)
    
    return radial_position_mm, polar_angle_rad


def fit_y_eq_Bx_simple(x, y):
    """
    function to fit the equation y = Bx to arrays [x], [y] 
    (i.e. a line with intercept at the origin)
    Called by calc_bend_angle()
    Acts on dimension 0; applies to all dimensions
    Based on Raghu's fityeqbx.m
    Returns B only; not uncertainty. 
    No y uncertainties.
    Doesn't check that x is nonzero (But returns inf if zero)
    A simple function!
    Inputs:
        x : x array; acts on dimension 0
        y : y array
    Outputs
        B : slope
    """
    sxx = np.sum(x**2, axis=0)
    sxy = np.sum(x*y, axis=0)
    
    # To avoid a runtime warning for sxx==0, don't do
    # np.where(sxx != 0, sxy / sxx, np.inf)
    # np.divide(sxy, sxx, out=np.full_like(sxy, np.inf), where=sxx != 0
              
    return np.divide(sxy, sxx, out=np.full_like(sxy, np.inf), where=sxx != 0)
    
    
def calc_bend_angle(position_data, CSVcolumns, M=None):
    """
    Calculate the bending angle for each fish in each frame, as
    the best-fit line from points (x[M], y[M]) to (x[0], y[0]) 
    and the best-fit line from points (x[M], y[M]) to (x[-1], y[-1]), 
    Return the angle between these points in the range [0, pi].
    
    Bend angle defined as pi minus opening angle, 
    so that a straight fish has angle 0.
    
    inputs
        position_data : basic position information for this dataset, numpy array
        CSVcolumns : information on what the columns of position_data are
        M : the index of the midpoint; will use N/2 if not input
     
    output
        bend_angle : in range [-π, π]; shape (Nframes, Nfish)
        As of Oct. 2025: signed angles in range [-π, π], with sign indicating
        left/right bend relative to heading direction.
    """
    
    x = position_data[:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    y = position_data[:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]

    Nframes, N, Nfish = x.shape
    if M is None:
        M = round(N/2)
    
    # Prepare output array
    bend_angle = np.zeros((Nframes, Nfish))
    
    # Calculate the opening angle
    for j in range(Nframes):
        # Calculate best-fit line for first segment (front half), for each fish
        x1 = x[j, M::-1, :] # shape (N/2, Nfish)
        y1 = -1.0*y[j, M::-1, :] # shape (N/2, Nfish); -1 since increasing down
        slope1 = fit_y_eq_Bx_simple(x1-x1[0,:], y1-y1[0,:])
        safe_slope1 = np.where(np.isinf(slope1) | np.isnan(slope1), 0, slope1) # 0 if infinite.

        # determine four-quadrant angle from sign of avg step (not elegant!)
        signx = np.sign(np.mean(np.diff(x1, axis=0), axis=0))
        signy = np.sign(np.mean(np.diff(y1, axis=0), axis=0))
        
        # Calculate the unit vector for the heading, allowing infinite slope
        # Numerator components
        numerator_x = np.where(np.isinf(slope1), 0, signx)
        numerator_y = np.where(np.isinf(slope1) | np.isnan(slope1), 1, 
                               signy * np.abs(safe_slope1))

        # Denominator: sqrt(1 + slope^2), safely
        denominator = np.sqrt(1 + safe_slope1**2)

        # np.divide to avoid invalid value warnings
        vector1 = np.vstack([
            np.divide(numerator_x, denominator, out=np.zeros_like(numerator_x), where=denominator != 0),
            np.divide(numerator_y, denominator, out=np.zeros_like(numerator_y), where=denominator != 0)
        ])
                     
        # Calculate best-fit line for second segment (back half)
        x2 = x[j, M:, :] # shape (N/2, Nfish)
        y2 = -1.0*y[j, M:, :] # shape (N/2, Nfish); -1 since increasing down
        slope2 = fit_y_eq_Bx_simple(x2-x2[0,:], y2-y2[0,:])
        safe_slope2 = np.where(np.isinf(slope2) | np.isnan(slope2), 0, slope2) # 0 if infinite.
        # determine four-quadrant angle (not elegant!)
        signx = np.sign(np.mean(np.diff(x2, axis=0), axis=0))
        signy = np.sign(np.mean(np.diff(y2, axis=0), axis=0))
        # Calculate the unit vector for the heading, allowing infinite slope
        # Numerator components
        numerator_x = np.where(np.isinf(slope2), 0, signx)
        numerator_y = np.where(np.isinf(slope2) | np.isnan(slope2), 1, 
                               signy * np.abs(safe_slope2))

        # Denominator: sqrt(1 + slope^2), safely
        denominator = np.sqrt(1 + safe_slope2**2)
        # np.divide to avoid invalid value warnings
        vector2 = np.vstack([
            np.divide(numerator_x, denominator, out=np.zeros_like(numerator_x), where=denominator != 0),
            np.divide(numerator_y, denominator, out=np.zeros_like(numerator_y), where=denominator != 0)
        ])

        # Calculate unsigned opening angle
        # Clip to be safe, but this shouldn't be necessary
        opening_angle = np.arccos(np.clip(np.sum(vector1 * vector2, axis=0), 
                                          -1.0, 1.0))
        unsigned_bend = np.pi - opening_angle
        
        # Calculate cross product to determine sign
        # Positive cross product = back half points right of heading
        cross_z = vector1[0, :] * vector2[1, :] - vector1[1, :] * vector2[0, :]
        
        # Apply sign
        bend_angle[j, :] = np.where(cross_z >= 0, unsigned_bend, -unsigned_bend)
    
    return bend_angle


def get_isBending_frames(dataset, bend_angle_threshold_deg = 10.0):
    """ 
    Find frames in which one or more fish have an above-threshold bend angle.
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
        bend_angle_threshold_deg : bend angle threshold, degrees
    Output : 
        isBending_frames...
           _each:  dictionary with a key for each fish, 
                   each containing a numpy array of frames in which 
                   that fish has bend_angle > threshold
           _any : numpy array of frames in which any fish has bend_angle > threshold
           _all : numpy array of frames in which all fish have bend_angle > threshold
    """
    
    bend_angle = dataset["bend_angle"] # Nframes x Nfish array of bend angles
    Nfish = dataset["Nfish"]
    isBending = np.abs(bend_angle) > (np.pi/180)*bend_angle_threshold_deg

    # Dictionary containing "is Bending" frames for each fish
    isBending_frames_each = {}
    for fish in range(Nfish):
        isBending_frames_each[fish] = dataset['frameArray'][isBending[:, fish]]
        
    # any fish
    any_fish_bending  = np.any(isBending, axis=1)
    isBending_frames_any = dataset['frameArray'][any_fish_bending]
    
    # all fish
    all_fish_bending  = np.all(isBending, axis=1)
    isBending_frames_all = dataset['frameArray'][all_fish_bending] 

    return isBending_frames_each, isBending_frames_any, isBending_frames_all


def get_bend_behavior_frames(dataset, params):
    """
    Analyze bend behaviors for each fish in the dataset.
    Find frames in which one or more fish have a J, R, or C-bend.
    Find blocks of frames with bend angle above the minimum threshold, and then
    identify this block as J, C, or R based on the maximum bend angle in that block
    
    Inputs:
        dataset: dataset dictionary of all behavior information for a given expt.
                 Must include 'bend_angle' (radians) and 'frameArray'
        params: Dictionary containing bend angle thresholds in degrees
    Returns:
        Jbend_frames: Frames with J-bends for each fish
        Rbend_frames: Frames with R-bends for each fish
        Cbend_frames: Frames with C-bends for each fish
        Each of these is a dictionary with keys for each fish number (typically
                       0, 1) each of which contains a numpy array of frames with 
                       identified bend frames for each fish
    
    """
    # Extract parameters
    bend_min_rad = params['bend_min_deg']*np.pi/180.0
    bend_Cmin_rad = params['bend_Cmin_deg']*np.pi/180.0
    bend_Jmax_rad = params['bend_Jmax_deg']*np.pi/180.0
    
    # Get number of fish and frames
    Nfish = dataset['bend_angle'].shape[1]
    Nframes = dataset['bend_angle'].shape[0]
    
    # Initialize output arrays
    Jbend_frames = [None] * Nfish
    Rbend_frames = [None] * Nfish
    Cbend_frames = [None] * Nfish
    
    # Iterate through each fish
    for fish in range(Nfish):
        # Initialize boolean arrays
        isJbend = np.zeros(Nframes, dtype=bool)
        isRbend = np.zeros(Nframes, dtype=bool)
        isCbend = np.zeros(Nframes, dtype=bool)
        
        # Find contiguous blocks of frames above bend_min
        above_min = np.abs(dataset['bend_angle'][:, fish]) > bend_min_rad
        above_min_int = above_min.astype(int)
        transitions = np.diff(above_min_int)
        
        # Find start and end of contiguous blocks
        # Find block starts (transitions from 0 to 1); index values
        block_starts = np.where(transitions == 1)[0] + 1
        # Find block ends (transitions from 1 to 0)
        block_ends = np.where(transitions == -1)[0]
        
        # Handle edge cases if the array starts or ends with True
        if above_min[0]:
            block_starts = np.concatenate(([0], block_starts))
        if above_min[-1]:
            block_ends = np.concatenate((block_ends, [len(above_min) - 1]))
    
        # Ensure starts and ends match
        assert len(block_starts) == len(block_ends), "Mismatch in block starts and ends"

        # Analyze each contiguous block
        for start, end in zip(block_starts, block_ends):
            # Handle single-frame blocks and multi-frame blocks
            block_max = np.max(np.abs(
                dataset['bend_angle'][start:end+1, fish]))\
                if start < end else np.abs(dataset['bend_angle'][start, fish])
            if block_max > bend_Cmin_rad:
                # C-bend
                isCbend[start:end+1] = True
            elif bend_Jmax_rad < block_max <= bend_Cmin_rad:
                # R-bend
                isRbend[start:end+1] = True
            elif block_max <= bend_Jmax_rad:
                # J-bend
                isJbend[start:end+1] = True
 
        # Store frames for each bend type
        Jbend_frames[fish] = dataset['frameArray'][isJbend]
        Rbend_frames[fish] = dataset['frameArray'][isRbend]
        Cbend_frames[fish] = dataset['frameArray'][isCbend]
    
    return Jbend_frames, Rbend_frames, Cbend_frames


                      
def average_bout_trajectory_oneSet(dataset, keyName = "speed_array_mm_s", 
                                   keyIdx = None, t_range_s=(-0.5, 2.0), 
                                   use_abs_value = False,
                                   makePlot=False, 
                                   constraintKey=None,
                                   constraintRange=None,
                                   constraintIdx = None,
                                   use_abs_value_constraint = False):
    """
    Tabulates some quantity from a dataset, typically speed
    dataset["speed_array_mm_s"], around each onset of a bout 
    ("isActive" == True) in the time interval specified by t_range_s. 
    Consider each fish in the dataset, and combines the 
    bout information for that fish unless keyIdx is not None
    (see below; recommended to keep as None!).
    Optional: Only consider bouts for which the value of constraintKey 
    at the start frame (first isActive frame) is between constraintRange[0]
    and constraintRange[1]. Calls get_values_subset(); can apply operations
    like mean on constraint, or particular index for constraint array
    
    Parameters:
        dataset : single dictionary containing the analysis data.
        keyName : the key to combine (default speed, "speed_array_mm_s")
        keyIdx  : integer or string, or None, used by get_values_subset(). 
                  **Note:** recommended to use keyIdx==None; if not, think
                  carefully about the output. Note that "other" is an 
                  additional keyIdx possibility not used by get_values_subset()
                  If keyIdx is:
                    None: If datasets[j][keyName] is a multidimensional array, 
                       return the full array (minus bad frames, constraints)
                    an integer: extract that column
                       (e.g. datasets[12]["speed_array_mm_s"][:,keyIdx])
                    the string "min", "max", or "mean"; perform this operation
                       along axis==1 (e.g. max for the fastest fish)
                    the string "other": If Nfish == 2, extract the value
                       for the *other* fish, i.e. fish 1 for fish0 bouts
                       and fish 0 for fish1 bouts. 
        t_range_s : time interval, seconds, around bout offset around which to tabulate data
        use_abs_value : bool, default False
                    If True, use absolute value of the quantitative 
                    property before applying constraints or combining values. 
                    Useful for signed angles (relative orientation, bending).
        makePlot : if True, plot for each fish speed vs. time relative to bout start
        constraintKey (str): Key name for the constraint, 
            or None to use no constraint. Apply the same constraint to both keys.
            see combine_all_values_constrained()
        constraintRange (np.ndarray): Numpy array with two elements specifying the constraint range, or None to use no constraint.
        constraintIdx : integer or string, or None, used by get_values_subset(). 
                    If constraintIdx is:
                        None: won't apply constraint
                        an integer: if the constraint is a multidimensional 
                            array, will use that column (e.g. fish # constraintIdx)
                        a string: use the operation "min", "max",
                           or "mean", along axis==1 (e.g. for fastest fish)
                        see combine_all_values_constrained()
        use_abs_value_constraint : bool, default False
                    If True, use absolute value of the quantitative constraint
                    property before applying constraints or combining values. 
                    Useful for signed angles (relative orientation, bending).    
    
    Returns:
        avg_values : list of length Nfish, each of which contains
            mean (column 0) and sem (column 1) of the values (typically
            speed) in the time interval around a bout start
    """
        
    Nfish = dataset["Nfish"]
    if keyIdx is not None and keyIdx.lower() == 'other' and Nfish != 2:
        raise ValueError('Error: Nfish must be 2 for keyIdx being "other".')

    fps = dataset["fps"]
    t_min_s, t_max_s = t_range_s
    frames_to_plot = int((t_max_s - t_min_s) * fps + 1)
    start_offset = int(t_min_s * fps)

    if (keyIdx is not None) and (keyIdx.lower() != 'other') :
        print('Warning: keyIdx is not None (or "other"), so bout trajectory will not')
        print("         match the fish that is moving. Be sure you know what you're doing!")
    
    if constraintKey is not None and constraintRange is not None:
        if len(constraintRange) != 2 or not all(isinstance(x, (int, float)) for x in constraintRange):
            raise ValueError("constraintRange must be a numpy array with two numerical elements")
    
    avg_values = []
    
    for k in range(Nfish):
        # Start frames and durations for bouts (i.e. motion)
        moving_frameInfo = dataset[f"isActive_Fish{k}"]["combine_frames"]
        
        # Remove columns based on conditions (start and end of dataset)
        valid_columns = (
            (moving_frameInfo[0,:] + (t_max_s * fps + 1) <= dataset["Nframes"]) &
            (moving_frameInfo[0,:] + (t_min_s * fps) >= 1)
        )
        moving_frameInfo = moving_frameInfo[:, valid_columns]
        
        # Number of events
        N_events = moving_frameInfo.shape[1]
        
        all_values = []
        
        # Get the values, e.g. speed, for the moving fish
        if keyIdx is None:
            # Consider the fish that's moving (recommended!)
            thisKeyIdx = k
        elif keyIdx.lower() == 'other':
            # already checked that Nfish = 2.
            # Choose the other fish
            tempFish = np.arange(0, Nfish)
            thisKeyIdx = int(tempFish[tempFish != k][0])
        else:
            thisKeyIdx = keyIdx
        values_fullRange = get_values_subset(dataset[keyName], thisKeyIdx,
                                             use_abs_value = use_abs_value)
        
        if constraintKey is not None:
            # constraint key may be multidimensional
            constraint_array = get_values_subset(dataset[constraintKey], 
                                                 constraintIdx,
                                                 use_abs_value = use_abs_value_constraint)
        
        for j in range(N_events):
            start_frame = moving_frameInfo[0,j] + start_offset
            end_frame = start_frame + frames_to_plot
            
            # Check constraint if provided
            if constraintKey is not None and constraintRange is not None:
                constraint_value = constraint_array[moving_frameInfo[0,j]]
                if not (constraintRange[0] <= constraint_value <= constraintRange[1]):
                    continue
            if (values_fullRange.ndim > 1) and (values_fullRange.shape[1]>1):
                # Not recommended; probably won't happen
                values = values_fullRange[start_frame:end_frame, thisKeyIdx]
            else:
                values = values_fullRange[start_frame:end_frame]
            all_values.append(values)
        
        if all_values:
            """
            if constraintKey is not None:
                print(f'{N_events} events; {len(all_values)} meeting criteria.')
            else:
                print(f'{N_events} events.')
            """
            all_values = np.array(all_values)
            mean_value = np.mean(all_values, axis=0)
            std_error = np.std(all_values, axis=0) / np.sqrt(len(all_values))
            
            avg_values.append(np.column_stack((mean_value, std_error)))
            
            if makePlot:
                time_array_s = np.linspace(t_min_s, t_max_s, frames_to_plot)
                plt.figure(figsize=(10, 6))
                plt.plot(time_array_s, mean_value, 'k-', label=f'Fish {k}')
                plt.fill_between(time_array_s, mean_value - std_error, mean_value + std_error, alpha=0.3)
                plt.xlabel('Time from bout start (s)', fontsize=18)
                if keyName == 'speed_array_mm_s':
                    ylabelStr = 'Speed (mm/s)'
                    titleStr = f'Avg. Bout Speed, Fish {k}, band=SEM'
                else:
                    ylabelStr = keyName
                    titleStr = f'Avg. {keyName}, Fish {k}, band=SEM'
                plt.ylabel(ylabelStr, fontsize=18)
                plt.title(titleStr, fontsize=18)
                plt.legend()
                plt.show()
        else:
            avg_values.append(np.array([]))
    
    return avg_values



def average_bout_trajectory_allSets(datasets, t_range_s=(-0.5, 2.0), 
                                    keyName = "speed_array_mm_s", 
                                    keyIdx = None, 
                                    use_abs_value = False,
                                    constraintKey=None,
                                    constraintRange=None,
                                    constraintIdx = None,
                                    use_abs_value_constraint = False,
                                    makePlot=False, color = 'black',
                                    ylim = None, titleStr = None,
                                    outputFileName = None,
                                    closeFigure = False):
    """
    For all datasets, call average_bout_trajectory_oneSet() 
    to tabulate some quantity, typically speed
    dataset["speed_array_mm_s"], around each onset of a bout 
    ("isActive" == True) in the time interval specified by t_range_s. 
    Note that this considers each fish in the dataset, and combines
    the bout information for that fish unless keyIdx is not None
    (see below; recommended to keep as None!).
    
    Optional: only consider bouts for which the value of constraintKey 
    at the start frame (first isActive frame) is between constraintRange[0]
    and constraintRange[1]. Calls get_values_subset(); can apply operations
    like mean on constraint
    
    Parameters:
        datasets (list): List of dictionaries containing the analysis data.
        keyName : the key to combine (default speed, "speed_array_mm_s")
        keyIdx  : integer or string, or None, used by get_values_subset(). 
                  **Note:** recommended to use keyIdx==None; if not, think
                  carefully about the output. Note that "other" is an 
                  additional keyIdx possibility not used by get_values_subset()
                  If keyIdx is:
                    None: If datasets[j][keyName] is a multidimensional array, 
                       return the full array (minus bad frames, constraints)
                    an integer: extract that column
                       (e.g. datasets[12]["speed_array_mm_s"][:,keyIdx])
                    the string "min", "max", or "mean"; perform this operation
                       along axis==1 (e.g. max for the fastest fish)
                    the string "other": If Nfish == 2, extract the value
                       for the *other* fish, i.e. fish 1 for fish0 bouts
                       and fish 0 for fish1 bouts. 
        use_abs_value : bool, default False
                    If True, use absolute value of the quantitative 
                    property before applying constraints or combining values. 
                    Useful for signed angles (relative orientation, bending).
        t_range_s : time interval, seconds, around bout offset around which to tabulate data
        constraintKey (str): Key name for the constraint, 
            or None to use no constraint. Apply the same constraint to both keys.
            see combine_all_values_constrained()
        constraintRange (np.ndarray): Numpy array with two elements specifying the constraint range, or None to use no constraint.
        constraintIdx : integer or string, or None, used by get_values_subset(). 
                    If constraintIdx is:
                        None: won't apply constraint
                        an integer: if the constraint is a multidimensional 
                            array, will use that column (e.g. fish # constraintIdx)
                        a string: use the operation "min", "max",
                           or "mean", along axis==1 (e.g. for fastest fish)
                        see combine_all_values_constrained()
        use_abs_value_constraint : bool, default False
                    If True, use absolute value of the quantitative constraint
                    property before applying constraints or combining values. 
                    Useful for signed angles (relative orientation, bending)        makePlot : if True, plot avg speed vs. time relative to bout start
        makePlot : if True, make a plot
        color : plot color (string)
        ylim : Optional ymin, ymax for plot
        titleStr : title string for plot
        outputFileName : if not None, and makePlot is True,
                        save the figure with this filename (include extension)
        closeFigure : (bool) if True, if makePlot is True, close the figure
                        after creating it.
        
    Returns:
        avg_values : numpy array with mean, standard deviation, 
            and s.e.m. as columns.
    """

    all_mean_values = []
    all_fps = [] # To check if all fps are the same
    
    for dataset in datasets:
        avg_value_dataset = average_bout_trajectory_oneSet(dataset, 
                                                        keyName = keyName, 
                                                        keyIdx = keyIdx,
                                                        use_abs_value = use_abs_value,
                                                        t_range_s = t_range_s,
                                                        makePlot=False, 
                                                        constraintKey=constraintKey, 
                                                        constraintRange=constraintRange,
                                                        use_abs_value_constraint = use_abs_value_constraint)
        all_fps.append(dataset["fps"])
        
        for fish_values in avg_value_dataset:
            if fish_values.size > 0:  # Check if the array is not empty
                all_mean_values.append(fish_values[:, 0])  # Append only the mean speeds
    
    if not all_mean_values:
        return np.array([])  # Return empty array if no valid speeds were found

    if np.std(all_fps) / np.mean(all_fps) > 0.001:
        raise ValueError("fps not the same for all datasets!")
    fps = np.mean(all_fps)

    all_mean_values = np.array(all_mean_values)
    
    mean_values = np.mean(all_mean_values, axis=0)
    std_dev = np.std(all_mean_values, axis=0)
    sem = std_dev / np.sqrt(len(all_mean_values))
    
    avg_values = np.column_stack((mean_values, std_dev, sem))

    if makePlot:
        if titleStr is None:
            if keyName == 'speed_array_mm_s':
                ylabelStr = 'Speed (mm/s)'
                titleStr = 'Avg. Bout Speed, all datasets'
            else:
                ylabelStr = keyName
                titleStr = f'Avg. {keyName}, all datasets'        
        if keyName == 'speed_array_mm_s':
            ylabelStr = 'Speed (mm/s)'
        else:
            ylabelStr = keyName
        t_min_s, t_max_s = t_range_s
        frames_to_plot = int((t_max_s - t_min_s) * fps + 1)
        time_array = np.linspace(t_min_s, t_max_s, frames_to_plot) #array of time points to consider
        fig = plt.figure(figsize=(10, 6))
        plt.plot(time_array, mean_values, '-', color = color)
        plt.fill_between(time_array, mean_values - std_dev, mean_values + std_dev, alpha=0.2)
        plt.fill_between(time_array, mean_values - sem, mean_values + sem, alpha=0.4)
        plt.xlabel('Time from bout start (s)', fontsize=18)
        plt.ylabel(ylabelStr, fontsize=18)
        plt.title(titleStr, fontsize=18)
        if ylim is not None:
            plt.ylim(ylim)
        plt.show()
        if outputFileName != None:
            plt.savefig(outputFileName, bbox_inches='tight')
        if closeFigure:
            plt.close(fig)
            
    return avg_values



def make_single_fish_plots(datasets, exptName = '', color = 'black',
                           outputFileNameBase = 'single_fish',
                           outputFileNameExt = 'png',
                           closeFigures = False):
    """
    makes several useful "single fish" plots -- i.e. 
    plots of characteristics of individual fish, which may be in multi-fish 
    experiments
    Note that there are lots of parameter values that are hard-coded; this
    function is probably more useful to read than to run, pasting and 
    modifying its code.
    
    Inputs:
        datasets : dictionaries for each dataset
        exptName : (string) Experiment name, to append to titles.
        color: plot color (uses alpha for indiv. dataset colors)
        outputFileNameBase : base file name for figure output; if None,
                             won't save a figure file
        outputFileNameExt : extension for figure output (e.g. 'eps' or 'png')
        closeFigures : (bool) if True, close a figure after creating it.

    Outputs:

    """
    
    # Speed histogram
    speeds_mm_s_all = combine_all_values_constrained(datasets, 
                                                     keyName='speed_array_mm_s', 
                                                     dilate_minus1 = True)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_speed' + '.' + outputFileNameExt
    else:
        outputFileName = None
    plot_probability_distr(speeds_mm_s_all, bin_width = 1.0, 
                           bin_range = [0, None], 
                           ylim = (0.001, 0.5), xlim = (0.0, 70.0),
                           color = color,
                           yScaleType = 'log',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           xlabelStr = 'Speed (mm/s)', 
                           titleStr = f'{exptName}: Probability Distr.: Speed',
                           outputFileName = outputFileName,
                           closeFigure=closeFigures)

    # Angular_speed histogram
    angular_speeds_rad_s_all = combine_all_values_constrained(datasets, 
                                                     keyName='angular_speed_array_rad_s', 
                                                     dilate_minus1 = True)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_angularSpeed' + '.' + outputFileNameExt
    else:
        outputFileName = None
    plot_probability_distr(angular_speeds_rad_s_all, bin_width = 1.0, 
                           bin_range = [0, None], 
                           color = color, 
                           yScaleType = 'log',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           ylim = (0.001, 0.5), xlim = (0.0, 40.0),
                           xlabelStr = 'Angular Speed (rad/s)', 
                           titleStr = f'{exptName}: Probability Distr.: Angular Speed',
                           outputFileName = outputFileName,
                           closeFigure=closeFigures)


    # Radial position histogram
    radial_position_mm_all = combine_all_values_constrained(datasets, 
                                                     keyName='radial_position_mm', 
                                                     dilate_minus1 = False)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_radialpos' + '.' + outputFileNameExt
    else:
        outputFileName = None
    plot_probability_distr(radial_position_mm_all, bin_width = 0.5, 
                           bin_range = [0, None],
                           color = color,
                           yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           normalize_by_inv_bincenter = True,
                           ylim = (-0.025, 0.5),
                           xlabelStr = 'Radial position (mm)', 
                           titleStr = f'{exptName}: Probability Distr.: r',
                           outputFileName = outputFileName,
                           closeFigure=closeFigures)
    
    # Heading angle histogram
    heading_angle_all = combine_all_values_constrained(datasets, 
                                                 keyName='heading_angle', 
                                                 dilate_minus1 = False)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_heading_angle' + '.' + outputFileNameExt
    else:
        outputFileName = None
    bin_width = np.pi/30
    plot_probability_distr(heading_angle_all, bin_width = bin_width,
                           bin_range=[None, None], yScaleType = 'linear',
                           polarPlot = True,
                           color = color,
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           titleStr = f'{exptName}: Heading Angle',
                           ylim = (0, 0.3),
                           unit_scaling_for_plot = 1.0,
                           outputFileName = outputFileName,
                           closeFigure=closeFigures)
    
    # Radial alignment angle
    radial_alignment_all = combine_all_values_constrained(datasets, 
                                                     keyName='radial_alignment_rad', 
                                                     dilate_minus1 = False)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_radialAlignment_angle' + '.' + outputFileNameExt
    else:
        outputFileName = None
    bin_width = np.pi/30
    plot_probability_distr(radial_alignment_all, bin_width = bin_width,
                           bin_range=[None, None], yScaleType = 'linear',
                           plot_each_dataset = False,
                           plot_sem_band = True,
                           polarPlot = True,
                           color = color,
                           titleStr = f'{exptName}: Radial alignment angle (rad)',
                           ylim = (0, 0.6),
                           outputFileName = outputFileName,
                           closeFigure=closeFigures)

    # Speed vs. time for bouts
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_boutSpeed' + '.' + outputFileNameExt
    else:
        outputFileName = None
    average_bout_trajectory_allSets(datasets, keyName = "speed_array_mm_s", 
                                    keyIdx = None, t_range_s=(-1.0, 2.0), 
                                    titleStr = f'{exptName}: Bout Speed', 
                                    makePlot=True,
                                    color = color,
                                    outputFileName = outputFileName,
                                    closeFigure=closeFigures)

    # speed autocorrelation function
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_speedAutocorr' + '.' + outputFileNameExt
    else:
        outputFileName = None
    speed_ac_all, t_lag = \
        calculate_value_corr_all(datasets, keyName = 'speed_array_mm_s',
                                 corr_type='auto', dilate_minus1 = True, 
                                 t_max = 3.0, t_window = 10.0, fpstol = 1e-6)
    plot_function_allSets(speed_ac_all, t_lag, xlabelStr='time (s)', 
                          ylabelStr='Speed autocorrelation', 
                          titleStr = f'{exptName}: Speed autocorrelation', 
                          color = color,
                          xlim = (-0.1, 2.0),
                          average_in_dataset = True,
                          outputFileName = outputFileName,
                          closeFigure=closeFigures)

    