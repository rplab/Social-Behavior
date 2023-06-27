# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#-------------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 11/2/2022
# version ='1.0'
# ------------------------------------------------------------------------------
import re
import numpy as np
from exp_decay import *
from motion_tracking import *
from toolkit import load_data
# ------------------------------------------------------------------------------

# ------------------------- MODULE IN PROGRESS!!! ------------------------------

def main():
    dataset = "results_SocPref_3c_2wpf_k2_ALL.csv"
    pos_data = load_data(dataset, 3, 5)
    angle_data = load_data(dataset, 5, 6)
    window_size = 10
    dataset_name = re.search('\d[a-z]_\d[a-z]{3}_[a-z]{1,2}\d', dataset).group()
    end_of_arr = np.shape(pos_data)[1] 

    fish_speeds_tuple = get_speed(pos_data[0], pos_data[1], end_of_arr,
    window_size)
    fish1_speed = fish_speeds_tuple[0]
    fish2_speed = fish_speeds_tuple[1]

    fish_angles_tuple = get_angle(angle_data[0], angle_data[1], end_of_arr, 
    window_size)
    fish1_angles = fish_angles_tuple[0]
    fish2_angles = fish_angles_tuple[1]
    
    fish_velocities_tuple = get_velocity(fish1_speed, fish2_speed, angle_data[0],
    angle_data[1], end_of_arr, window_size)
    fish1_velocities = fish_velocities_tuple[0]
    fish2_velocities = fish_velocities_tuple[1]

    coarse_time_plots(fish1_velocities, fish2_velocities, dataset_name, 1500)

    # velocity_frames_plots(fish1_velocities, fish2_velocities, dataset_name, 3000)
    correlation_plots(fish1_velocities, fish2_velocities, fish1_angles, fish2_angles,
    'v', dataset_name, end_of_arr, window_size)

    # correlation_plots(fish1_speed, fish2_speed, fish1_angles, 
    # fish2_angles, dataset_name, end_of_arr, window_size)

    # plt.show()


if __name__ == "__main__":
    main()
