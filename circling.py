# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 5/26/2022
# version ='1.0'
# ---------------------------------------------------------------------------
import csv
from math import sqrt, isclose
from tkinter import N
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from circle_fit_taubin import TaubinSVD
# ---------------------------------------------------------------------------

def load_data(dir):
    data = np.genfromtxt(dir, delimiter=',')
    fish1_data = data[:15000]
    fish2_data = data[15000:]
    # fish1_data = data[:10]
    # fish2_data = data[15000:15010]
    return fish1_data, fish2_data


def get_reciprocal_radius(taubin_output):
    return 1 / taubin_output[2]


def get_distances(data_for_x_wf, taubin_output, idx):
    distance = sqrt((data_for_x_wf[idx][0] - taubin_output[0])**2 + 
                (data_for_x_wf[idx][1] - taubin_output[1])**2)
    return (distance - taubin_output[2])**2


def get_rmse(distances_lst):
    return sqrt(np.sum(distances_lst))


def get_circling_window_frames(fish1_data, fish2_data, window_size):
    # frame_numbers = [x for x in range(15000) if x % num_window_frames == 0]
    head_count = 0
    theta_count = 0 
    circling_wf = np.array([])
    theta_90 = np.array([])
    head_pos_for_x_wf = np.empty((2 * window_size, 2), int)
    theta_for_x_wf = np.empty((window_size, 2), float)

    for paired_data in zip(fish1_data, fish2_data):
        head_pos_for_x_wf[head_count] = paired_data[0][3:5]    # x,y pos fish1
        head_count += 1
        head_pos_for_x_wf[head_count] = paired_data[1][3:5]    # x,y pos fish2
        head_count += 1

        theta_for_x_wf[theta_count] = np.cos(
            paired_data[0][5] - paired_data[1][5])
        theta_count += 1

        if paired_data[0][1] % window_size == 0:
            head_count = 0
            theta_count = 0
            taubin_output = TaubinSVD(head_pos_for_x_wf)  # output gives (x_c, y_c, r)
            reciprocal_radius = get_reciprocal_radius(taubin_output)
            theta_avg = np.mean(theta_for_x_wf)
            
            distances_for_x_wf = np.empty(2 * window_size)
            for i in range(2 * window_size):
                distances_for_x_wf[i] = get_distances(head_pos_for_x_wf, 
                taubin_output, i)  
            
            rmse = get_rmse(distances_for_x_wf)

            if np.abs(theta_avg) < 0.1:
                # theta_90 = np.append(theta_90, paired_data[0][1])
                plt.plot(paired_data[0][1], 1, 'ro')
                # plt.annotate(f"wf {paired_data[0][1]}", (paired_data[0][1], 1))

        
            if (reciprocal_radius >= 0.005 and rmse < 25 and 
                np.abs(theta_avg) < 0.1):  
                # circling_wf = np.append(circling_wf, paired_data[0][1])
                plt.plot(paired_data[0][1], 2, 'o')

            # Empty the array for the next x window frames 
            head_pos_for_x_wf = np.empty((2 * window_size, 2), int)  
            theta_for_x_wf = np.empty((window_size, 2), float)

    # plt.figure()
    # plt.plot(theta_90)
    # plt.plot(circling_wf)
    plt.show()




    # return circling_wf
            


#     # plt.plot(wf, angles)
#     # plt.figure()
#     # plt.title(f"RMSE vs. Reciprocal Radii Per {num_window_frames} WFs")
#     # plt.xlabel('RMSE')
#     # plt.ylabel('1/r')
#     # plt.scatter(rmses, reciprocal_radii)
#     # plt.show()

#     # plt.show()
#     return circling_wf


  
    

def main():
    data = load_data("results_SocPref_3c_2wpf_k1_ALL.csv")
    fish1_data = data[0]
    fish2_data = data[1]

    circling_wfs = get_circling_window_frames(fish1_data, fish2_data, 10)
    print(circling_wfs)
  
    
   
    # angle_wfs = []
    # for paired_data in paired_data_lst:
    #     if -0.2 < np.cos(float(paired_data[5]) - float(paired_data[31])) < 0.2:
    #         angle_wfs.append(paired_data[1])

    # for wf in circling_wfs:
    #     for angle in angle_wfs:
    #         if isclose(int(wf), int(angle), rel_tol=10):
    #             plt.title(f"Window Frames w/90 Degree Orientation")
    #             plt.xlabel('Frame #')
    #             # plt.ylabel('Angle [cos(theta_1 - theta_2)]')
    #             plt.plot(angle, 1, 'ro')
    #             plt.plot(wf, 2, 'o')
            
    #         ninety_wf = paired_data[1]

    #         for wf in circling_wfs:
    #             if isclose(int(ninety_wf), int(wf), rel_tol=10):
    #                 plt.title(f"Window Frames w/90 Degree Orientation")
    #                 plt.xlabel('Frame #')
    #                 # plt.ylabel('Angle [cos(theta_1 - theta_2)]')
    #                 plt.plot(ninety_wf, 1, 'ro')
    #                 plt.plot(wf, 2, 'o')
    # plt.show()



if __name__ == "__main__":
    main()
        
