# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#-------------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/1/2023
# version ='1.0'
# ------------------------------------------------------------------------------
from toolkit import *
# ------------------------------------------------------------------------------

def get_tracking_error_wf(body1_x, body2_x, body1_y, body2_y, end, contact_dist):
    idx = 0
    errors = {"bad-tracking": [], "too-close": []}

    # Bad-tracking
    while idx < end:  
        # If there are errors in the x positions, then we know
        # bad-tracking has occurred 
        if np.all(body1_x[idx][1:] == 0) or np.all(body2_x[idx][1:] == 0):
            errors['bad-tracking'].append(idx+1)
        idx += 1
    
    errors['bad-tracking'] = combine_events(np.array(errors['bad-tracking']))

    # Too-close
    for wf in errors['bad-tracking']:
        # if isinstance(wf, tuple):  # type tuple
        #     idx_1 = wf[0] - 1   
        # else:                   # otherwise type int
        #     idx_1 = wf - 1
        idx_1 = wf - 1

        # If bad-tracking has occurred and the fish are close 
        # to each other in the previous window frame, then a contact
        # event has likely occured 
        for j in range(1, 10):  # there are ten markers on the fish bodies
            if (np.min(np.sqrt((body1_x[idx_1][j] - body2_x[idx_1])**2 + 
            (body1_y[idx_1][j] - body2_y[idx_1])**2)) < contact_dist or 
            np.min(np.sqrt((body1_x[idx_1] - body2_x[idx_1][j])**2 + 
            (body1_y[idx_1] - body2_y[idx_1][j])**2) < contact_dist)):
                errors['too-close'].append(idx_1+1)

    errors["too-close"] = np.unique(np.array(errors["too-close"]))

    return errors
