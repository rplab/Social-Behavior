# -*- coding: utf-8 -*-
"""
Author:   Raghuveer Parthasarathy
Created on Sat Oct  4 19:57:30 2025
Last modified Sat Oct  4 19:57:30 2025 -- Raghuveer Parthasarathy

Description
-----------

"""

def get_relative_orientation(position_data, dataset, CSVcolumns):
    """ 
    Calculate the relative orientation of each fish with respect to the
    head-to-head vector to the other fish.
    Now returns signed angles in range [-π, π].
    """
    
    if dataset["Nfish"] != 2:
        raise ValueError('Error: Relative orientation valid only for Nfish==2')

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
    phi0_unsigned = np.arccos(np.clip(dot_product_0, -1.0, 1.0))
    phi1_unsigned = np.arccos(np.clip(dot_product_1, -1.0, 1.0))
    
    # Calculate cross products to determine sign (z-component for 2D)
    # cross_z = v_x * dh_y - v_y * dh_x
    cross_z_0 = v0[:, 0] * dh_unit[:, 1] - v0[:, 1] * dh_unit[:, 0]
    cross_z_1 = v1[:, 0] * (-dh_unit[:, 1]) - v1[:, 1] * (-dh_unit[:, 0])
    
    # Apply sign: positive if cross product in +z, negative if in -z
    phi0 = np.where(cross_z_0 >= 0, phi0_unsigned, -phi0_unsigned)
    phi1 = np.where(cross_z_1 >= 0, phi1_unsigned, -phi1_unsigned)
    
    relative_orientation = np.stack((phi0, phi1), axis=1)
    
    # Rank by absolute value of relative orientation
    rel_orient_rankIdx = np.argsort(np.abs(relative_orientation), axis=1)
    
    return relative_orientation, rel_orient_rankIdx

def calc_bend_angle(position_data, CSVcolumns, M=None):
    """
    Calculate the bending angle for each fish in each frame.
    Now returns signed angles in range [-π, π], with sign indicating
    left/right bend relative to heading direction.
    """
    
    x = position_data[:, CSVcolumns["body_column_x_start"]:(CSVcolumns["body_column_x_start"]+CSVcolumns["body_Ncolumns"]), :]
    y = position_data[:, CSVcolumns["body_column_y_start"]:(CSVcolumns["body_column_y_start"]+CSVcolumns["body_Ncolumns"]), :]

    Nframes, N, Nfish = x.shape
    if M is None:
        M = round(N/2)
    
    angle = np.zeros((Nframes, Nfish))
    
    for j in range(Nframes):
        # First segment (front half)
        x1 = x[j, M::-1, :]
        y1 = -1.0*y[j, M::-1, :]
        slope1 = fit_y_eq_Bx_simple(x1-x1[0,:], y1-y1[0,:])
        safe_slope1 = np.where(np.isinf(slope1) | np.isnan(slope1), 0, slope1)
        
        signx = np.sign(np.mean(np.diff(x1, axis=0), axis=0))
        signy = np.sign(np.mean(np.diff(y1, axis=0), axis=0))
        
        numerator_x = np.where(np.isinf(slope1), 0, signx)
        numerator_y = np.where(np.isinf(slope1) | np.isnan(slope1), 1, 
                               signy * np.abs(safe_slope1))
        denominator = np.sqrt(1 + safe_slope1**2)
        
        vector1 = np.vstack([
            np.divide(numerator_x, denominator, out=np.zeros_like(numerator_x), where=denominator != 0),
            np.divide(numerator_y, denominator, out=np.zeros_like(numerator_y), where=denominator != 0)
        ])
        
        # Second segment (back half)
        x2 = x[j, M:, :]
        y2 = -1.0*y[j, M:, :]
        slope2 = fit_y_eq_Bx_simple(x2-x2[0,:], y2-y2[0,:])
        safe_slope2 = np.where(np.isinf(slope2) | np.isnan(slope2), 0, slope2)
        
        signx = np.sign(np.mean(np.diff(x2, axis=0), axis=0))
        signy = np.sign(np.mean(np.diff(y2, axis=0), axis=0))
        
        numerator_x = np.where(np.isinf(slope2), 0, signx)
        numerator_y = np.where(np.isinf(slope2) | np.isnan(slope2), 1, 
                               signy * np.abs(safe_slope2))
        denominator = np.sqrt(1 + safe_slope2**2)
        
        vector2 = np.vstack([
            np.divide(numerator_x, denominator, out=np.zeros_like(numerator_x), where=denominator != 0),
            np.divide(numerator_y, denominator, out=np.zeros_like(numerator_y), where=denominator != 0)
        ])
        
        # Calculate unsigned opening angle
        opening_angle = np.arccos(np.clip(np.sum(vector1 * vector2, axis=0), -1.0, 1.0))
        unsigned_bend = np.pi - opening_angle
        
        # Calculate cross product to determine sign
        # Positive cross product = back half points right of heading
        cross_z = vector1[0, :] * vector2[1, :] - vector1[1, :] * vector2[0, :]
        
        # Apply sign
        angle[j, :] = np.where(cross_z >= 0, unsigned_bend, -unsigned_bend)
    
    return angle
