# -*- coding: utf-8 -*-
# behavior_identification_single.py
"""
Author:   Raghuveer Parthasarathy
Split from behavior_identification.py on July 22, 2024
Last modified Sept. 12, 2024 -- Raghu Parthasarathy

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
    combine_all_values_constrained, get_values_subset, plot_probability_distr, \
    calculate_value_corr_all, plot_function_allSets


def get_coord_characterizations(datasets, CSVcolumns,
                                             expt_config, params):
    """
    For each dataset, simple coordinate characterizations
        (polar coordinates, radial alignment)
    
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
    print('Coordinate characterizations: ')
    for j in range(N_datasets):
        print('    for Dataset: ', datasets[j]["dataset_name"])
        
        # Get the radial position (distance to center) of each fish in each
        # frame.
        polar_coords = get_polar_coords(datasets[j]["all_data"], 
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

def get_single_fish_characterizations(datasets, CSVcolumns,
                                             expt_config, params):
    """
    For each dataset, characterizations that involve single fish
        (e.g. fish length, bending, speed)
    
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
    
    print('Single-fish characterizations: ')
    for j in range(N_datasets):
        print('    for Dataset: ', datasets[j]["dataset_name"])
        # Get the length of each fish in each frame (sum of all segments)
        # Nframes x Nfish array
        datasets[j]["fish_length_array_mm"] = \
            get_fish_lengths(datasets[j]["all_data"], 
                             datasets[j]["image_scale"], CSVcolumns)
                
        # Get the speed of each fish in each frame (frame-to-frame
        # displacement of head position, um/s); 0 for first frame
        # Nframes x 2 array
        datasets[j]["speed_array_mm_s"] = \
            get_fish_speeds(datasets[j]["all_data"], CSVcolumns, 
                            datasets[j]["image_scale"], expt_config['fps'])
            
        # Frames with speed above threshold (i.e. moving fish)
        # "_each" is a dictionary with a key for each fish, 
        #     each containing a numpy array of frames in which that fish is moving
        # "_any" is a numpy array of frames in which any fish is moving
        # "_all" is a numpy array of frames in which all fish are moving
        
        isMoving_frames_each, isMoving_frames_any, isMoving_frames_all = \
            get_isMoving_frames(datasets[j], params["motion_speed_threshold_mm_second"])
            
        # For movement indicator, for each fish and for "any" and "all" fish,
        # make a dictionary containing frames, removing frames with 
        # "bad" elements. Unlike other characteristics, need to "dilate"
        # bad tracking frames, since each bad tracking frame j affects 
        # speed j and speed j+1.
        # Also makes a 2xN array of initial frames and durations, as usual
        # for these dictionaries
        # Code to allow an arbitrary number of fish, and consequently an 
        #    arbitrary number of elements isMoving_frames_each dictionary
        
        
        iMe_arrays = [isMoving_frames_each[key] for key in sorted(isMoving_frames_each.keys())]
        iM_arrays = iMe_arrays + [isMoving_frames_any] + [isMoving_frames_all] 
        # Create the second tuple of strings
        iM_keys = tuple(f'isMoving_Fish{i}' for i in range(len(isMoving_frames_each))) + ('isMoving_any',) \
                  + ('isMoving_all', )

        for iM_key, iM_array in zip(iM_keys, iM_arrays):
            badTrFramesRaw = np.array(datasets[j]["bad_bodyTrack_frames"]["raw_frames"]).astype(int)
            dilate_badTrackFrames = dilate_frames(badTrFramesRaw, 
                                                  dilate_frames=np.array([1]))
            datasets[j][iM_key] = make_frames_dictionary(iM_array,
                                          dilate_badTrackFrames,
                                          behavior_name = iM_key,
                                          Nframes=datasets[j]['Nframes'])

        # Bout statistics.
        bouts_N, bout_duration_s,  bout_rate_bpm, bout_ibi_s = \
            get_bout_statistics(datasets[j])
        # average over fish, since ID is unreliable
        datasets[j]["bouts_N"] = np.mean(bouts_N)
        datasets[j]["bout_duration_s"] = np.mean(bout_duration_s)
        datasets[j]["bout_rate_bpm"] = np.mean(bout_rate_bpm)
        datasets[j]["bout_ibi_s"] = np.mean(bout_ibi_s)

        # Average speed (averaged over fish), and average speed only in moving frames
        speed_mean_all, speed_mean_moving = \
            get_mean_speed(datasets[j]["speed_array_mm_s"], 
                           isMoving_frames_each, datasets[j]["bad_bodyTrack_frames"]["raw_frames"])
        # average over fish, since ID is unreliable
        datasets[j]["speed_mm_s_mean"] = np.mean(speed_mean_all)
        datasets[j]["speed_whenMoving_mm_s_mean"] = np.mean(speed_mean_moving)


        # Get tail angle of each fish in each frame 
        # Nframes x Nfish array
        datasets[j]["tail_angle_rad"] = \
            getTailAngle(datasets[j]["all_data"], CSVcolumns, 
                         datasets[j]["heading_angle"])
        

        # Get frames with C-bends and J-bends; each is a dictionary with two 
        # keys, one for each fish, each containing a numpy array of frames
        Cbend_frames_each = get_Cbend_frames(datasets[j], CSVcolumns, 
                                             params["Cbend_threshold"])
        Jbend_frames_each = get_Jbend_frames(datasets[j], CSVcolumns, 
                                             (params["Jbend_rAP"], 
                                              params["Jbend_cosThetaN"], 
                                              params["Jbend_cosThetaNm1"]))
        # numpy array of frames with C-bend for *any* fish
        Cbend_frames_any = np.unique(np.concatenate(list(Cbend_frames_each.values())))
        # numpy array of frames with J-bend for *any* fish
        Jbend_frames_any = np.unique(np.concatenate(list(Jbend_frames_each.values())))
        
        # For C-bends and J-bends, for each fish and for any fish,
        # make a dictionary containing frames, 
        # remove frames with "bad" elements
        # Also makes a 2xN array of initial frames and durations, though
        # the duration of bends is probably 1 almost always
        # Code to allow an arbitrary number of fish, and consequently an 
        #    arbitrary number of elements in the Cbend_frames_each and 
        #    Jbend_frames_each dictionaries

        Ce_arrays = [Cbend_frames_each[key] for key in sorted(Cbend_frames_each.keys())]
        Je_arrays = [Jbend_frames_each[key] for key in sorted(Jbend_frames_each.keys())]
        CJ_arrays = Ce_arrays + [Cbend_frames_any] + Je_arrays + [Jbend_frames_any] 
        # Create the second tuple of strings
        CJ_keys = tuple(f'Cbend_Fish{i}' for i in range(len(Cbend_frames_each))) + ('Cbend_any',) \
                  + tuple(f'Jbend_Fish{i}' for i in range(len(Jbend_frames_each))) + ('Jbend_any',)

        for CJ_key, CJ_array in zip(CJ_keys, CJ_arrays):
            datasets[j][CJ_key] = make_frames_dictionary(CJ_array,
                                          (datasets[j]["edge_frames"]["raw_frames"],
                                           datasets[j]["bad_bodyTrack_frames"]["raw_frames"]),
                                          behavior_name = CJ_key,
                                          Nframes=datasets[j]['Nframes'])

    # For each dataset, exclude bad tracking frames from calculation of
    # the mean fish length
    # Also exclude bad tracking, dilated one frame as in isMoving,
    #    from calculation of mean speed (over-cautious: if any fish tracking 
    #    is bad, ignore the frame for mean speed calculation)
    for j in range(N_datasets):
        print('Removing bad frames from stats for fish length, for Dataset: ', 
              datasets[j]["dataset_name"])
        goodIdx = np.where(np.in1d(datasets[j]["frameArray"], 
                                   datasets[j]["bad_bodyTrack_frames"]["raw_frames"], 
                                   invert=True))[0]
        goodLengthArray = datasets[j]["fish_length_array_mm"][goodIdx]
        datasets[j]["fish_length_mm_mean"] = np.mean(goodLengthArray)
        print(f'   Mean fish length: {datasets[j]["fish_length_mm_mean"]:.3f} mm')

    return datasets

    
def get_fish_lengths(all_data, image_scale, CSVcolumns):
    """
    Get the length of each fish in each frame (sum of all segments)
    Input:
        all_data : all position data, from dataset["all_data"]
        image_scale : scale, um/px; from dataset["image_scale"]
        CSVcolumns : CSV column information (dictionary)
    Output
        fish_lengths : (mm) Nframes x Nfish array of fish lengths
    """
    xstart = int(CSVcolumns["body_column_x_start"])
    xend =int(CSVcolumns["body_column_x_start"])+int(CSVcolumns["body_Ncolumns"])
    ystart = int(CSVcolumns["body_column_y_start"])
    yend = int(CSVcolumns["body_column_y_start"])+int(CSVcolumns["body_Ncolumns"])
    dx = np.diff(all_data[:,xstart:xend,:], axis=1)
    dy = np.diff(all_data[:,ystart:yend,:], axis=1)
    dr = np.sqrt(dx**2 + dy**2)
    fish_lengths = np.sum(dr,axis=1)*image_scale/1000.0 # mm
    return fish_lengths
            
def get_fish_speeds(all_data, CSVcolumns, image_scale, fps):
    """
    Get the speed of each fish in each frame (frame-to-frame
        displacement of head position)
    Input:
        all_data : all position data, from dataset["all_data"]
        CSVcolumns : CSV column information (dictionary)
        image_scale : scale, um/px; from dataset["image_scale"]
        fps : frames/second, from expt_config
    Output
        speed_array_mm_s  : (mm/s) Nframes x Nfish array
    """
    head_x = all_data[:,CSVcolumns["head_column_x"],:] # x, both fish
    head_y = all_data[:,CSVcolumns["head_column_y"],:] # y, both fish
    # Frame-to-frame speed, px/frame
    dr = np.sqrt((head_x[1:,:] - head_x[:-1,:])**2 + 
                    (head_y[1:,:] - head_y[:-1,:])**2)
    speed_array_mm_s = dr*image_scale*fps/1000.0
    # to make Nframes x Nfish set as 0 for the first frame
    speed_array_mm_s = np.append(np.zeros((1, all_data.shape[2])), 
                                 speed_array_mm_s, axis=0)
    
    return speed_array_mm_s


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
        isMoving_frames_each[fish] = np.array(np.where(isMoving[:, fish])).flatten() + 1
    
    # any fish
    any_fish_moving  = np.any(isMoving, axis=1)
    isMoving_frames_any = np.where(any_fish_moving)[0] + 1
    
    # all fish
    all_fish_moving  = np.all(isMoving, axis=1)
    isMoving_frames_all = np.where(all_fish_moving)[0] + 1

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
                                           badTrackFrames + 1))
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
    
def getTailAngle(all_data, CSVcolumns, heading_angles):
    """
    Calculate the tail angle:  angle of last two body points 
        relative to heading angle (zero if aligned); 
        absolute value, radians.
    Input:
        all_data : all position data, from dataset["all_data"]
        CSVcolumns : CSV column information (dictionary)
        heading_angles : all heading angles (Nframes x Nfish); dataset[j]["heading_angles"]
    Output
        tail_angle  : (rad) Nframes x Nfish array
    """
    # Final tail positions, for each frame, for each fish
    # Each array is Nframes x (2 body points) x (Nfish==2)
    tailpos_x = all_data[:,CSVcolumns["body_column_x_start"] + CSVcolumns["body_Ncolumns"] - 3: \
                           CSVcolumns["body_column_x_start"] + CSVcolumns["body_Ncolumns"] - 1,:]
    tailpos_y = all_data[:,CSVcolumns["body_column_y_start"] + CSVcolumns["body_Ncolumns"] - 3: \
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
    # print('\ndy')
    # print(dy[95:106,0])
    # print('\ndx')
    # print(dx[95:106,0])
    # print('\nTail angle lab')
    # print(tail_angle_lab[95:106,0]*180.0/np.pi)
    # print('\nHeading angle')
    # print(heading_angles[95:106,0]*180.0/np.pi)
    # print('\nTail angle')
    # print(tail_angle[95:106,0]*180.0/np.pi)
    # x = input('cont? ')
    return tail_angle

def getTailCurvature(all_data, CSVcolumns, image_scale):
    """
    Calculate the curvature of the fish posterior (last 5 body
        datapoints) at each frame, for each fish.
    
    NOTE: Not used; not robust!
    
    Input:
        all_data : all position data, from dataset["all_data"]
        CSVcolumns : CSV column information (dictionary)
        image_scale : scale, um/px; from dataset["image_scale"]
    Output
        curvature_invmm  : (1/mm) Nframes x Nfish array
    """
    
    # All tail positions, for each frame, for each fish
    # Each array is Nframes x (Nbodypts/2==5) x (Nfish==2)
    tailpos_x = all_data[:,CSVcolumns["body_column_x_start"] : \
                           CSVcolumns["body_column_x_start"] + int(CSVcolumns["body_Ncolumns"]/2),:]
    tailpos_y = all_data[:,CSVcolumns["body_column_y_start"] : \
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
    "moving" criterion is met (isMoving_Fish{j})
    Note that "isMoving" already disregards bad tracking frames
    Calculates:
        # Number of bouts for each fish
		# Mean Duration of bouts for each fishk, seconds
		# bout rate (bouts per minute) for each fish
		# Mean inter-bout-interval (s) for each fish

    *NOT* averaged over fish -- do this externally if needed.
    
    Inputs:
        dataset : single dictionary containing the analysis data, including
                    key "isMoving_Fish{j}" for each fish j; also fps and
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
        moving_frameInfo = dataset[f"isMoving_Fish{k}"]["combine_frames"]

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
    

def get_polar_coords(all_data, CSVcolumns, arena_center, image_scale):
    """
    Get the fish head position in polar coordinates relative to the
        arena center, for each fish in each frame.
        "y" defined as decreasing downward, so angle = atan(-y,x)
    Input:
        all_data : all position data, from dataset["all_data"]
        CSVcolumns : CSV column information (dictionary)
        arena_center : tuple of (x,y) positions of the Arena Center,
                       probably stored in dataset["arena_center"]
        image_scale : scale, um/px; from dataset["image_scale"]
    Output
        polar_coords : tuple of 
            radial_position_mm  : (mm) Nframes x Nfish array
            polar_angle_rad  : (radians) Nframes x Nfish array
    """
    head_x = all_data[:,CSVcolumns["head_column_x"],:] # x, all fish, Nframes x Nfish
    head_y = all_data[:,CSVcolumns["head_column_y"],:] # y, all fish, Nframes x Nfish
    # Radial position, px
    dx = head_x - arena_center[0]
    dy = head_y - arena_center[1]
    r = np.sqrt(dx**2 + dy**2)
    radial_position_mm = r*image_scale/1000.0
    polar_angle_rad = np.arctan2(-1.0*dy, dx)
    
    return radial_position_mm, polar_angle_rad

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
        Cbend_frames : dictionary with a number of keys equal to the 
                       number of fish, each of which
                       contains a numpy array of frames with 
                       identified C-bend frames, 
                       i.e. with bending < Cbend_threshold
    """
    
    # length in each frame, Nframes x Nfish array, mm so convert
    # to px using image scale (um/px)
    fish_length_px = dataset["fish_length_array_mm"] * 1000.0 / dataset["image_scale"]  
    Nfish = fish_length_px.shape[1]
    
    body_x = dataset["all_data"][:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    body_y = dataset["all_data"][:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]
    fish_head_tail_distance = np.sqrt((body_x[:,0,:]-body_x[:,-1,:])**2 + 
                                      (body_y[:,0,:]-body_y[:,-1,:])**2) # Nframes x Nfish array
    Cbend_ratio = fish_head_tail_distance/fish_length_px # Nframes x Nfish==2 array
    Cbend = Cbend_ratio < Cbend_threshold # # True if fish is bent; Nframes x Nfish array
    
    # Dictionary containing Cbend_frames frames for each fish
    Cbend_frames = {}
    for fish in range(Nfish):
        Cbend_frames[fish] = np.array(np.where(Cbend[:, fish])).flatten() + 1

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
    
    angle_data = dataset["heading_angle"]
    Nfish = angle_data.shape[1]

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

    # Dictionary containing Jbend_frames frames for each fish
    Jbend_frames = {}
    for fish in range(Nfish):
        Jbend_frames[fish] = np.array(np.where(allCriteria[:, fish])).flatten() + 1

    return Jbend_frames


                      
def average_bout_trajectory_oneSet(dataset, keyName = "speed_array_mm_s", 
                                   keyIdx = None, t_range_s=(-0.5, 2.0), 
                                   makePlot=False, 
                                   constraintKey=None,
                                   constraintRange=None,
                                   constraintIdx = None):
    """
    Tabulates some quantity from a dataset, typically speed
    dataset["speed_array_mm_s"], around each onset of a bout 
    ("isMoving" == True) in the time interval specified by t_range_s. 
    Consider each fish in the dataset, and combines the 
    bout information for that fish unless keyIdx is not None
    (see below; recommended to keep as None!).
    Optional: Only consider bouts for which the value of constraintKey 
    at the start frame (first isMoving frame) is between constraintRange[0]
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
        moving_frameInfo = dataset[f"isMoving_Fish{k}"]["combine_frames"]
        
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
            thisKeyIdx = tempFish[tempFish != k][0]
        else:
            thisKeyIdx = keyIdx
        values_fullRange = get_values_subset(dataset[keyName], thisKeyIdx)
        
        if constraintKey is not None:
            # constraint key may be multidimensional
            constraint_array = get_values_subset(dataset[constraintKey], 
                                                 constraintIdx)
        
        for j in range(N_events):
            start_frame = moving_frameInfo[0,j] + start_offset
            end_frame = start_frame + frames_to_plot
            
            # Check constraint if provided
            if constraintKey is not None and constraintRange is not None:
                constraint_value = constraint_array[moving_frameInfo[0,j]]
                if not (constraintRange[0] <= constraint_value <= constraintRange[1]):
                    continue
            if values_fullRange.ndim > 1:
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
                                    keyIdx = None, constraintKey=None,
                                    constraintRange=None,
                                    constraintIdx = None,
                                    makePlot=False, 
                                    ylim = None, titleStr = None,
                                    outputFileName = None):
    """
    For all datasets, call average_bout_trajectory_oneSet() 
    to tabulate some quantity, typically speed
    dataset["speed_array_mm_s"], around each onset of a bout 
    ("isMoving" == True) in the time interval specified by t_range_s. 
    Note that this considers each fish in the dataset, and combines
    the bout information for that fish unless keyIdx is not None
    (see below; recommended to keep as None!).
    
    Optional: only consider bouts for which the value of constraintKey 
    at the start frame (first isMoving frame) is between constraintRange[0]
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
        makePlot : if True, plot avg speed vs. time relative to bout start
        ylim : Optional ymin, ymax for plot
        titleStr : title string for plot
        outputFileName : if not None, and makePlot is True,
                        save the figure with this filename (include extension)
    
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
                                                        t_range_s = t_range_s,
                                                        makePlot=False, 
                                                        constraintKey=constraintKey, 
                                                        constraintRange=constraintRange)
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
        plt.figure(figsize=(10, 6))
        plt.plot(time_array, mean_values, 'k-')
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
            
    return avg_values



def make_single_fish_plots(datasets, outputFileNameBase = 'single_fish',
                           outputFileNameExt = 'png'):
    """
    makes several useful "single fish" plots -- i.e. 
    plots of characteristics of individual fish, which may be in multi-fish 
    experiments
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
    
    # Speed histogram
    speeds_mm_s_all = combine_all_values_constrained(datasets, 
                                                     keyName='speed_array_mm_s', 
                                                     dilate_plus1 = True)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_speed' + '.' + outputFileNameExt
    else:
        outputFileName = None
    plot_probability_distr(speeds_mm_s_all, bin_width = 1.0, 
                           bin_range = [0, None], 
                           xlabelStr = 'Speed (mm/s)', 
                           titleStr = 'Probability distribution: Speed',
                           outputFileName = outputFileName)

    # Radial position histogram
    radial_position_mm_all = combine_all_values_constrained(datasets, 
                                                     keyName='radial_position_mm', 
                                                     dilate_plus1 = False)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_radialpos' + '.' + outputFileNameExt
    else:
        outputFileName = None
    plot_probability_distr(radial_position_mm_all, bin_width = 0.5, 
                           bin_range = [0, None], yScaleType = 'linear',
                           xlabelStr = 'Radial position (mm)', 
                           titleStr = 'Probability distribution: r (not normalized!)',
                           outputFileName = outputFileName)
    
    # Heading angle histogram
    heading_angle_all = combine_all_values_constrained(datasets, 
                                                 keyName='heading_angle', 
                                                 dilate_plus1 = False)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_heading_angle' + '.' + outputFileNameExt
    else:
        outputFileName = None
    bin_width = np.pi/30
    plot_probability_distr(heading_angle_all, bin_width = bin_width,
                           bin_range=[None, None], yScaleType = 'linear',
                           polarPlot = True,
                           titleStr = 'Heading Angle',
                           ylim = (0, 0.3),
                           outputFileName = outputFileName)
    
    # Radial alignment angle
    radial_alignment_all = combine_all_values_constrained(datasets, 
                                                     keyName='radial_alignment_rad', 
                                                     dilate_plus1 = False)
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_radialAlignment_angle' + '.' + outputFileNameExt
    else:
        outputFileName = None
    bin_width = np.pi/30
    plot_probability_distr(radial_alignment_all, bin_width = bin_width,
                           bin_range=[None, None], yScaleType = 'linear',
                           polarPlot = True,
                           titleStr = 'Radial alignment angle (rad)',
                           ylim = (0, 0.3),
                           outputFileName = outputFileName)

    # Speed vs. time for bouts
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_boutSpeed' + '.' + outputFileNameExt
    else:
        outputFileName = None
    average_bout_trajectory_allSets(datasets, keyName = "speed_array_mm_s", 
                                    keyIdx = None, t_range_s=(-1.0, 2.0), 
                                    titleStr = 'Bout Speed', makePlot=True,
                                    outputFileName = outputFileName)

    # speed autocorrelation function
    if outputFileNameBase is not None:
        outputFileName = outputFileNameBase + '_speedAutocorr' + '.' + outputFileNameExt
    else:
        outputFileName = None
    speed_ac_all, t_lag = \
        calculate_value_corr_all(datasets, keyName = 'speed_array_mm_s',
                                 corr_type='auto', dilate_plus1 = True, 
                                 t_max = 3.0, t_window = 10.0, fpstol = 1e-6)
    plot_function_allSets(speed_ac_all, t_lag, xlabelStr='time (s)', 
                          ylabelStr='Speed autocorrelation', 
                          titleStr='Speed autocorrelation', 
                          average_in_dataset = True,
                          outputFileName = outputFileName)

    