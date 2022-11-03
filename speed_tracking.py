import matplotlib.pyplot as plt
import numpy as np
from toolkit import *

def get_speed(pos_data, end, window_size):
    idx_1, idx_2 = 0, window_size
    wfs = np.arange(0, end-window_size, window_size)
    
    fish1_speed = np.array([])
    fish2_speed = np.array([])
    while idx_2 < end:   # end of the array for both fish
        # Speed = distance / time
        fish1_speed = np.append(fish1_speed, 
        get_head_distance_traveled(pos_data[0], idx_1, idx_2) / window_size) 
        fish2_speed = np.append(fish2_speed, 
        get_head_distance_traveled(pos_data[1], idx_1, idx_2) / window_size)

        idx_1, idx_2 = idx_1+window_size, idx_2+window_size
    
    plt.figure()
    plt.title(f"Speed Tracking for 3c_2wpf_nk1; WS={window_size}")
    plt.plot(wfs, fish1_speed, color='purple')
    plt.plot(wfs, fish2_speed, color='pink')
    plt.show()


def main():
    pos_data = load_data("results_SocPref_3c_2wpf_nk2_ALL.csv", 3, 5)
    end_of_arr = np.shape(pos_data)[1] 

    for i in range(3,20,3):
        get_speed(pos_data, end_of_arr, i)


if __name__ == "__main__":
    main()