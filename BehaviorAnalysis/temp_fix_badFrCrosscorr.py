# -*- coding: utf-8 -*-
"""
Author:   Raghuveer Parthasarathy
Created on Sat Sep 20 00:05:45 2025
Last modified Sat Sep 20 00:05:45 2025 -- Raghuveer Parthasarathy

Description
-----------

"""

def calculate_value_corr_oneSet(dataset, keyName='speed_array_mm_s', 
                                corr_type='auto', dilate_plus1=True, 
                                t_max=10, t_window=None):
    """
    For a *single* dataset, calculate the auto or cross-correlation of the numerical
    property in the given key (e.g. speed).
    Cross-correlation is valid only for Nfish==2 (verified)
    Ignore "bad tracking" frames. If a frame is in the bad tracking list,
    replace the value with a Gaussian random number with the mean and std. dev.
    calculated from good frames in a local window around the bad frame.
    NOTE: replaces values for *both* fish if a frame is a bad-tracking frame, 
       even if one of the fish is properly tracked. (Simpler to implement, and
       also both fish's values may be unreliable)
    If "dilate_plus1" is True, dilate the bad frames +1.
    Output is a numpy array with dim 1 corresponding to each fish, for auto-
    correlation, and a 1D numpy array for cross-correlation.
    
    Parameters
    ----------
    dataset : single analysis dataset
    keyName : the key to combine (e.g. "speed_array_mm_s")
    corr_type : 'auto' for autocorrelation, 'cross' for cross-correlation (only for Nfish==2)
    dilate_plus1 : If True, dilate the bad frames +1
    t_max : max time to consider for autocorrelation, seconds.
    t_window : size of sliding window in seconds. If None, don't use a sliding window.
    
    Returns
    -------
    corr : correlation of desired property, numpy array of
                    shape (#time lags + 1 , Nfish) for autocorrelation
                    shape (#time lags + 1 , 1) for cross-correlation
    t_lag : time lag array, seconds (including zero)
    """
    value_array = dataset[keyName]
    Nframes, Nfish = value_array.shape
    fps = dataset["fps"]
    badTrackFrames = dataset["bad_bodyTrack_frames"]["raw_frames"]
    if dilate_plus1:
        dilate_badTrackFrames = dilate_frames(badTrackFrames, 
                                              dilate_frames=np.array([-1]))
        bad_frames_set = set(dilate_badTrackFrames)
    else:
        bad_frames_set = set(badTrackFrames)
     
    if corr_type == 'auto':
        t_lag = np.arange(0, t_max + 1.0/fps, 1.0/fps)
        n_lags = len(t_lag)
        corr = np.zeros((n_lags, Nfish))
    elif corr_type == 'cross':
        if Nfish != 2:
            raise ValueError("Cross-correlation is only supported for Nfish==2")
        t_lag = np.arange(-t_max, t_max + 1.0/fps, 1.0/fps)
        n_lags = len(t_lag)
        corr = np.zeros(n_lags)
    else:
        raise ValueError("corr_type must be 'auto' or 'cross'")
    
    # Lowest frame number (should be 1)   
    idx_offset = min(dataset["frameArray"])
    
    # Replace values from bad tracking frames with local mean, std.
    fish_value = value_array.copy()
    
    # Calculate window size for local statistics
    if t_window is not None:
        w = int(t_window * fps)
    else:
        # If no t_window specified, use a reasonable default (e.g., 2 seconds)
        w = int(2.0 * fps)
    
    half_w = w // 2
    
    for fish in range(Nfish):
        for frame in bad_frames_set:
            frame_idx = frame - idx_offset  # Convert to 0-based indexing
            
            if 0 <= frame_idx < Nframes:
                # Define local window bounds, ensuring they're within data range
                start_idx = max(0, frame_idx - half_w)
                end_idx = min(Nframes, frame_idx + half_w + 1)
                
                # Get good values in the local window
                local_good_values = []
                for local_idx in range(start_idx, end_idx):
                    local_frame = local_idx + idx_offset
                    if local_frame not in bad_frames_set:
                        local_good_values.append(fish_value[local_idx, fish])
                
                # If we have enough good values in the local window, use local stats
                if len(local_good_values) >= 3:  # Need at least 3 points for reasonable stats
                    mean_value = np.mean(local_good_values)
                    std_value = np.std(local_good_values)
                    if std_value == 0:  # Avoid division by zero
                        std_value = 1e-10
                else:
                    # Fall back to global statistics if local window has too few good values
                    all_good_values = [val for i, val in enumerate(fish_value[:,fish]) 
                                     if (i + idx_offset) not in bad_frames_set]
                    if len(all_good_values) > 0:
                        mean_value = np.mean(all_good_values)
                        std_value = np.std(all_good_values)
                        if std_value == 0:
                            std_value = 1e-10
                    else:
                        # Last resort: use original value (shouldn't happen in practice)
                        continue
                
                # Replace with random value using local or global statistics
                fish_value[frame_idx, fish] = np.random.normal(mean_value, std_value)

    if corr_type == 'auto':
        for fish in range(Nfish):
            if t_window is None:
                fish_corr = calculate_autocorr(fish_value[:, fish], n_lags)
            else:
                window_size = int(t_window * fps)
                fish_corr = calculate_block_autocorr(fish_value[:, fish],  
                                                     n_lags, window_size)
            corr[:, fish] = fish_corr
            
    if corr_type == 'cross':
        if t_window is None:
            corr = calculate_crosscorr(fish_value[:, 0], fish_value[:, 1], 
                                       n_lags)
        else:
            window_size = int(t_window * fps)
            corr = calculate_block_crosscorr(fish_value[:, 0], 
                                             fish_value[:, 1], n_lags, 
                                             window_size)
    
    return corr, t_lag