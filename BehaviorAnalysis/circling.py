# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='2.0': Major modifications by Raghuveer Parthasarathy, May-June 2023
# Last modified June 25, 2023, Raghu Parthasarathy
# ---------------------------------------------------------------------------
from toolkit import *
from circle_fit_taubin import TaubinSVD
# ---------------------------------------------------------------------------

def get_circling_wf(fish_pos, head_separation, fish_angle_data,
                    Nframes, window_size, circle_fit_threshold, cos_theta_AP_threshold, 
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
                and angle_criterion and tangent_criterion):
            circling_wf.append(idx+1)  # append the starting frame number
            
            if showDiagnosticPlots:
                print('idx: ', idx_1, ', rmse: ', rmse)
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
