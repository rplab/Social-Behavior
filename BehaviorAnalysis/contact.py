
# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#-------------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='2.0': Major modifications by Raghuveer Parthasarathy, May-June 2023
# Last modified June 23, 2023 -- Raghu Parthasarathy
# ------------------------------------------------------------------------------
from toolkit import *
# ------------------------------------------------------------------------------

def get_contact_wf(body1_x, body2_x, body1_y, body2_y, Nframes, contact_distance):
    """
    Returns a dictionary of window frames for different 
    contact between two fish: any body positions, or head-body contact

    Args:
        body1_x (array): a 2D array of x positions along the 10 body markers of fish1.
        body2_x (array): a 2D array of x positions along the 10 body markers of fish2.
        body1_y (array): a 2D array of y positions along the 10 body markers of fish1.
        body2_y (array): a 2D array of y positions along the 10 body markers of fish2.
        Nframes (int): Number of frames (typically 15,000.)
        contact_distance: the contact distance threshold.

    Returns:
        contact_dict (dictionary): a dictionary of arrays of different contact types.
    """
    contact_dict = {"any_contact": [], "head-body": []}

    for idx in range(Nframes):

        # Any contact: look at closest element of distance matrix between
        # body points
        d0 = np.subtract.outer(body1_x[idx], body2_x[idx]) # all pairs of subtracted x positions
        d1 = np.subtract.outer(body1_y[idx], body2_y[idx]) # all pairs of subtracted y positions
        d = np.sqrt(d0**2 + d1**2) # Euclidean distance matrix, all points
        if np.min(d) < contact_distance:
            contact_dict["any_contact"].append(idx+1)
            
        # Head-body contact
        d_head1_body2 = np.sqrt((body1_x[idx][0] - body2_x[idx])**2 + 
                                (body1_y[idx][0] - body2_y[idx])**2)
        d_head2_body1 = np.sqrt((body2_x[idx][0] - body1_x[idx])**2 + 
                                (body2_y[idx][0] - body1_y[idx])**2)
        if (np.min(d_head1_body2) < contact_distance) or \
            (np.min(d_head2_body1) < contact_distance):
                contact_dict["head-body"].append(idx+1)
                # Note that "any contact" will be automatically satisfied.

    return contact_dict


def get_inferred_contact_frames(dataset, frameWindow, contact_dist):
    """
    Returns an array of frames corresponding to inferred contact, in which
    tracking is bad (zeros values) but inter-fish distance was decreasing
    over some number of preceding frames and was below-threshold immediately 
    before the bad tracking.
    
    Args:
        dataset : dictionary with all dataset info
        frameWindow : number of preceding frames to examine
        contact_dist: the contact distance threshold.

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
            this_distance = np.array(dataset["inter-fish_distance"]\
                                     [np.where(np.isin(dataset["frameArray"], precedingFrames))])
            # Check decreasing, and check that last element is within threshold
            if (this_distance[-1]<contact_dist) and np.all(this_distance[:-1] >= this_distance[1:]):
                inf_contact_frames.append(precedingFrames[-1])

    return inf_contact_frames
