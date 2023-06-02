
# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#-------------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='1.0'
# Last modified May 28, 2023 -- Raghu Parthasarathy
# ------------------------------------------------------------------------------
from toolkit import *
# ------------------------------------------------------------------------------

def get_contact_wf(body1_x, body2_x, body1_y, body2_y, end, window_size,
contact_dist):
    """
    Returns a dictionary of window frames for different types of
    contact between two fish.

    Args:
        body1_x (array): a 1D array of x positions along the 10 body markers of fish1.
        body2_x (array): a 1D array of x positions along the 10 body markers of fish2.
        body1_y (array): a 1D array of y positions along the 10 body markers of fish1.
        body2_y (array): a 1D array of y positions along the 10 body markers of fish2.
        end (int): end of the array for both fish (typically 15,000 window frames.) 
        window_size (int): window size for which circling is averaged over.
        contact_dist (int): the contact distance threshold.

    Returns:
        contact_wf (dictionary): a dictionary of arrays of different contact types.
    """
    idx_1, idx_2 = 0, window_size
    contact_wf = {"any_contact": [], "head-body": []}

    while idx_2 <= end:   # end of the array for both fish
        # Head-body contact
        if (np.min(np.sqrt((body1_x[idx_1][0] - body2_x[idx_1])**2 + 
        (body1_y[idx_1][0] - body2_y[idx_1])**2)) < contact_dist or 
        np.min(np.sqrt((body1_x[idx_1] - body2_x[idx_1][0])**2 + 
        (body1_y[idx_1] - body2_y[idx_1][0])**2) < contact_dist)):
             contact_wf["head-body"].append(idx_2)
             contact_wf["any_contact"].append(idx_2+1)
        else:
            # Look for any contact 
            any_contact_boolean = False
            for j in range(1, 10):  # there are ten markers on the fish bodies
                if (np.min(np.sqrt((body1_x[idx_1][j] - body2_x[idx_1])**2 + 
                   (body1_y[idx_1][j] - body2_y[idx_1])**2)) < contact_dist or 
                   np.min(np.sqrt((body1_x[idx_1] - body2_x[idx_1][j])**2 + 
                   (body1_y[idx_1] - body2_y[idx_1][j])**2) < contact_dist)) and (idx_2 not in
                   contact_wf["any_contact"]):
                       any_contact_boolean = True
            if any_contact_boolean:
                contact_wf["any_contact"].append(idx_2+1)

        # Update the index variables to track contact for the 
        # next x window frames of size window_size
        idx_1 += 1
        idx_2 += 1

    contact_wf["any_contact"] = combine_events(np.array(contact_wf["any_contact"]))
    contact_wf["head-body"] = combine_events(np.array(contact_wf["head-body"]))

    return contact_wf
