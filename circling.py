################################################################################
import csv
from math import sqrt, isclose
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from circle_fit_taubin import TaubinSVD
################################################################################

def load_data(dir):
    data = []
    with open(dir, 'r') as file:
        csv_file = csv.reader(file)
        for row in csv_file:
            data.append(row)
    return data


def get_paired_data(data_lst):
    fish1_data = data_lst[:15000]
    fish2_data = data_lst[15000:]

    # Put 1 timeframe for both fish into a list
    paired_data_lst = list(map(list.__add__, fish1_data, fish2_data))    
    return paired_data_lst 


def get_reciprocal_radius(taubin_output):
    return 1 / taubin_output[2]


def get_distances(data_for_x_wf, taubin_output, idx):
    distance = sqrt((data_for_x_wf[idx][0] - taubin_output[0])**2 + 
                (data_for_x_wf[idx][1] - taubin_output[1])**2)
    return (distance - taubin_output[2])**2


def get_rmse(distances_lst):
    return sqrt(np.sum(distances_lst))


def get_circling_window_frames(paired_data_lst, num_window_frames):
    head_positions_for_x_wf = []
    # rmses = []
    # frame_numbers = [x for x in range(15000) if x % num_window_frames == 0]
    orientation_angles_for_x_wf = []   
    # reciprocal_radii = []
    circling_wf = []
    # angles = []
    # wf = []


    for paired_data in paired_data_lst:
        # x,y positions of fish1 are at the 3rd and 4th indices 
        # x,y positions of fish2 are at the 29th and 30th indices
        head_positions_for_x_wf.append(list(map(int,map(float,paired_data[3:5]))))   
        head_positions_for_x_wf.append(list(map(int,map(float,paired_data[29:31])))) 

        # Orientation of fish1 is at the 5th index; 31st index is for fish2
        orientation_angles_for_x_wf.append(
            np.cos(float(paired_data[5]) - float(paired_data[31])))

        if int(paired_data[1]) % num_window_frames == 0:               
            taubin_output = TaubinSVD(head_positions_for_x_wf)  # output gives (x_c, y_c, r)
            reciprocal_radius = get_reciprocal_radius(taubin_output)
            # reciprocal_radii.append(reciprocal_radius) 
            orientation_angle = mean(orientation_angles_for_x_wf)

            # if -0.2 < orientation_angle < 0.2:
            #     angles.append(orientation_angle)
            #     wf.append(taubin_output[1])
                # plt.plot(taubin_output[1], orientation_angle, 'ro')
            
            distances_for_x_wf = []
            for i in range(2 * num_window_frames):
                distances_for_x_wf.append(
                    get_distances(head_positions_for_x_wf, taubin_output, i))   
            
            rmse = get_rmse(distances_for_x_wf)
            # rmses.append(rmse)
        
            if (reciprocal_radius >= 0.005 and rmse < 25 and 
                -1 <= orientation_angle <= -0.8):  
                circling_wf.append(paired_data[1])

            # Empty the list for the next x window frames 
            head_positions_for_x_wf = []     
            orientation_angles_for_x_wf = []   

    # plt.plot(wf, angles)
    # plt.figure()
    # plt.title(f"RMSE vs. Reciprocal Radii Per {num_window_frames} WFs")
    # plt.xlabel('RMSE')
    # plt.ylabel('1/r')
    # plt.scatter(rmses, reciprocal_radii)
    # plt.show()

    # plt.show()
    return circling_wf


  
    

def main():
    data = load_data("results_SocPref_3c_2wpf_k1_ALL.csv")
    paired_data_lst = get_paired_data(data)
    circling_wfs = get_circling_window_frames(paired_data_lst, 10)
    print(circling_wfs)
    
   
    angle_wfs = []
    for paired_data in paired_data_lst:
        if -0.2 < np.cos(float(paired_data[5]) - float(paired_data[31])) < 0.2:
            angle_wfs.append(paired_data[1])

    for wf in circling_wfs:
        for angle in angle_wfs:
            if isclose(int(wf), int(angle), rel_tol=10):
                plt.title(f"Window Frames w/90 Degree Orientation")
                plt.xlabel('Frame #')
                # plt.ylabel('Angle [cos(theta_1 - theta_2)]')
                plt.plot(angle, 1, 'ro')
                plt.plot(wf, 2, 'o')
            
            ninety_wf = paired_data[1]

            for wf in circling_wfs:
                if isclose(int(ninety_wf), int(wf), rel_tol=10):
                    plt.title(f"Window Frames w/90 Degree Orientation")
                    plt.xlabel('Frame #')
                    # plt.ylabel('Angle [cos(theta_1 - theta_2)]')
                    plt.plot(ninety_wf, 1, 'ro')
                    plt.plot(wf, 2, 'o')
    plt.show()



if __name__ == "__main__":
    main()
