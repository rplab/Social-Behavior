# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='1.0'
# Last modified May 28, 2023, Raghu Parthasarathy
# ---------------------------------------------------------------------------
from math import sqrt
from toolkit import *
from circle_fit_taubin import TaubinSVD
# ---------------------------------------------------------------------------


def get_circling_wf(fish1_pos, fish2_pos, fish1_angle_data, fish2_angle_data,
end, window_size, rmse_thresh, cos_theta_threshold, head_dist_thresh):
    """
    Returns an array of window frames for circling behavior. Each window
    frame represents the STARTING window frame for circling within some range 
    of window frames specified by the parameter window_size. E.g, if 
    window_size = 10 and a circling window frame is 210, then circling
    occured from frames 200-210.
    
    Args:
        fish1_pos (array): a 2D array of (x, y) positions for fish1. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].
        fish2_pos (array): a 2D array of (x, y) positions for fish2. The
                           array has form [[x1, y1], [x2, y2], [x3, y3],...].

        fish1_angle_data (array): a 1D array of angles at each window frame
                                  for fish1.
        fish2_angle_data (array): a 1D array of angles at each window frame
                                  for fish2.

        end (int): end of the array for both fish (typically 15,000 window frames.)
        
        window_size (int)     : window size for which circling is averaged over.
        rmse_thresh (int)     : RMSE threshold for circling.
        cos_theta_threshold (float)     : antiparallel orientation upper bound for cos(theta) 
        head_dist_thresh (int): head distance threshold for the two fish.

    Returns:
        circling_wf (array): a 1D array of circling window frames.
    """
    idx_1, idx_2 = 0, window_size
    circling_wf = []
    
    while idx_2 <= end:   # end of the array for both fish
        # Get head position and angle data for all window frames 
        fish1_positions = fish1_pos[idx_1:idx_2]
        fish2_positions = fish2_pos[idx_1:idx_2]
        fish1_angles = fish1_angle_data[idx_1:idx_2]
        fish2_angles = fish2_angle_data[idx_1:idx_2]

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
        
        # Head separation, for all points
        dh = fish1_positions - fish2_positions
        head_separation = np.sqrt(np.sum(dh**2, axis=1))
        head_separation_criterion = (head_separation < head_dist_thresh).all()
        
        # Should be antiparallel, so cos(theta) < threshold (ideally cos(theta)==-1)
        cos_theta = np.cos(fish1_angles - fish2_angles)
        angle_criterion = (cos_theta < cos_theta_threshold).all()
        
        # Radius of the best-fit circle should be less than the mean 
        # distance between the heads of the fish over the frame window.
        circle_size_criterion = (taubin_output[2] < np.mean(head_separation))
        
        showDiagnosticPlots = False
        
        # try taubin_output[2]/10 

        if (rmse < rmse_thresh and head_separation_criterion
                and angle_criterion):
            circling_wf.append(idx_1+1)  # append the starting frame number
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
    
        # Update the index variables to track circling for the 
        # next x window frames of size window_size
        idx_1 += 1
        idx_2 += 1
    return combine_events(np.array(circling_wf))
