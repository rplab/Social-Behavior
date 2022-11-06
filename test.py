# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 10/5/2022
# version ='1.0'
# ---------------------------------------------------------------------------
import re
from circling import *
from ninety_deg import *
from tail_rubbing import *
from visuals import *
# ---------------------------------------------------------------------------
params = {
    "circ_ws" : 10,     
    "circ_rad" : 0.005, 
    "circ_rmse" : 25,  
    "circ_anti_low" : -1,   
    "circ_anti_high" : -0.9,
    "circ_head_dist": 150,
    "90_ws" : 10,   
    "theta_90_thresh" : 0.1,  
    "90_head_dist" : 300,     
    "contact_ws" : 1, 
    "contact_dist" : 20, 
    "tail_rub_ws" : 4,  
    "tail_dist" : 40,   
    "tail_head_dist" : 150,
    "tail_anti_low" : -1,   
    "tail_anti_high": -0.8,
}

def tester(param_name, mean, std, N):
    pos_data = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 3, 5)
    angle_data = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 5, 6)
    contact_x = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 6, 16)
    contact_y = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 16, 26)

    # End of array should be the same for all loaded data
    end_of_arr = np.shape(pos_data)[1]  

    # Create an array of testable paramater values 
    if ('ws' in param_name or 'rmse' in param_name or'dist' in param_name):
        arr = np.unique(np.random.normal(loc=mean, scale=std, 
        size=N).astype(int))
        params_arr = arr[arr >= 1]  # Only values >= 1 are valid
    elif 'anti_high' in param_name:
        arr = np.unique(np.random.normal(loc=mean, scale=std, 
        size=N).astype(float))
        params_arr = arr[arr > -1]  # Only values > -1 are valid
    else:
        arr = np.unique(np.random.normal(loc=mean, scale=std, 
        size=N).astype(float))
        params_arr = arr[arr > 0]  # Only values > 0 are valid

    # Call the appropriate function
    if re.match('circ', param_name):
        num_events = np.array([])
        for param in params_arr:
            num_events = np.append(num_events,
            np.size(get_circling_func(param_name, param, pos_data, angle_data, 
            end_of_arr)))
        return (params_arr, num_events)
    if re.match('90', param_name):
        none = np.array([])
        one = np.array([])
        both = np.array([])
        for param in params_arr:
            res_arr = get_90_deg_func(param_name, param, pos_data, angle_data, 
            end_of_arr)
            none = np.append(none, np.size(res_arr["none"]))
            one = np.append(one, np.size(res_arr["1"]))
            both = np.append(both, np.size(res_arr["both"]))
        return (params_arr, (none, one, both))
    if re.match("contact", param_name):
        any = np.array([])
        head_body = np.array([])
        for param in params_arr:
            res_arr = get_contact_func(param_name, param, contact_x, 
            contact_y, end_of_arr)
            any = np.append(any, np.size(res_arr["any"]))
            head_body = np.append(head_body, np.size(res_arr["head-body"]))
        return (params_arr, (any, head_body))
    if re.match('tail', param_name):
        num_events = np.array([])
        for param in params_arr:
            num_events = np.append(num_events,
            np.size(get_circling_func(param_name, param, pos_data, angle_data, 
            end_of_arr)))
        return (params_arr, num_events)
    

def get_circling_func(param_name, param_val, pos_data, angle_data, end_of_arr):
    if param_name == "circ_ws": 
        res = get_circling_wf(pos_data[0], pos_data[1], angle_data[0], 
        angle_data[1], end_of_arr, param_val, params["circ_rad"], 
        params["circ_rmse"], params["circ_anti_low"], params["circ_anti_high"],
        params["circ_head_dist"])
    elif param_name == "circ_rad": 
        res = get_circling_wf(pos_data[0], pos_data[1], angle_data[0], 
        angle_data[1], end_of_arr, params["circ_ws"], param_val, 
        params["circ_rmse"], params["circ_anti_low"], params["circ_anti_high"],
        params["circ_head_dist"])
    elif param_name == "circ_rmse": 
        res = get_circling_wf(pos_data[0], pos_data[1], angle_data[0], 
        angle_data[1], end_of_arr, params["circ_ws"], params["circ_rad"], 
        param_val, params["circ_anti_low"], params["circ_anti_high"],
        params["circ_head_dist"])
    elif param_name == "circ_anti_high": 
        res = get_circling_wf(pos_data[0], pos_data[1], angle_data[0], 
        angle_data[1], end_of_arr, params["circ_ws"], params["circ_rad"], 
        params["circ_rmse"], params["circ_anti_low"], param_val,
        params["circ_head_dist"])
    else:   # vary circ_head_dist parameter
        res = get_circling_wf(pos_data[0], pos_data[1], angle_data[0], 
        angle_data[1], end_of_arr, params["circ_ws"], params["circ_rad"], 
        params["circ_rmse"], params["circ_anti_low"], params["circ_anti_high"],
        param_val)
    return res


def get_90_deg_func(param_name, param_val, pos_data, angle_data, end_of_arr):
    if param_name == "90_ws": 
        res = get_90_deg_wf(pos_data[0], pos_data[1], angle_data[0], angle_data[1],
        end_of_arr, param_val, params["theta_90_thresh"], params["90_head_dist"])
    elif param_name == "theta_90_thresh": 
        res = get_90_deg_wf(pos_data[0], pos_data[1], angle_data[0], angle_data[1],
        end_of_arr, params["90_ws"], param_val, params["90_head_dist"])
    else:   # vary 90_head_dist parameter
        res = get_90_deg_wf(pos_data[0], pos_data[1], angle_data[0], angle_data[1],
        end_of_arr, params["90_ws"], params["theta_90_thresh"], param_val)
    return res


def get_contact_func(param_name, param_val, contact_x, contact_y, end_of_arr):
    if param_name == "contact_ws": 
        res = get_contact_wf(contact_x[0], contact_x[1], contact_y[0], 
        contact_y[1], end_of_arr, param_val, params["contact_dist"])
    if param_name == "contact_dist": 
        res = get_contact_wf(contact_x[0], contact_x[1], contact_y[0], 
        contact_y[1], end_of_arr, params["contact_ws"], param_val)
    return res


def get_tail_rub_func(param_name, param_val, pos_data, angle_data, contact_x, 
contact_y, end_of_arr):
    if param_name == "tail_rub_ws": 
        res = get_tail_rubbing_wf(contact_x[0], contact_x[1], contact_y[0], 
        contact_y[1], pos_data, angle_data, end_of_arr, param_val, 
        params["tail_dist"], params["tail_anti_low"], params["tail_anti_high"],
        params["tail_head_dist"])
    elif param_name == "tail_dist":
        res = get_tail_rubbing_wf(contact_x[0], contact_x[1], contact_y[0], 
        contact_y[1], pos_data, angle_data, end_of_arr, params["tail_rub_ws"], 
        param_val, params["tail_anti_low"], params["tail_anti_high"],
        params["tail_head_dist"])
    elif param_name == "tail_anti_high":
        res = get_tail_rubbing_wf(contact_x[0], contact_x[1], contact_y[0], 
        contact_y[1], pos_data, angle_data, end_of_arr, params["tail_rub_ws"], 
        params["tail_dist"], params["tail_anti_low"], param_val,
        params["tail_head_dist"])
    else:      # vary the tail_head_dist parameter
        res = get_tail_rubbing_wf(contact_x[0], contact_x[1], contact_y[0], 
        contact_y[1], pos_data, angle_data, end_of_arr, params["tail_rub_ws"], 
        params["tail_dist"], params["tail_anti_low"], params["tail_anti_high"],
        param_val)
    return res


def plot(param_name, params_arr, res_array):
    if re.match("circ", param_name) or re.match("tail", param_name):
        plt.figure()
        plt.title(f"{param_name} vs. Number of Events")
        plt.xlabel(f"{param_name}")
        plt.ylabel(f"Number of Events")
        plt.plot(params_arr, res_array)
    if re.match("90", param_name):
        plt.figure()
        plt.title(f"{param_name} vs. Number of Events")
        plt.xlabel(f"{param_name}")
        plt.ylabel(f"Number of Events")
        plt.plot(params_arr, res_array[0], color='green')
        plt.plot(params_arr, res_array[1], color='blue')
        plt.plot(params_arr, res_array[2], color='red')
    if re.match("contact", param_name):
        plt.figure()
        plt.title(f"{param_name} vs. Number of Events")
        plt.xlabel(f"{param_name}")
        plt.ylabel(f"Number of Events")
        plt.plot(params_arr, res_array[0], color='purple')
        plt.plot(params_arr, res_array[1], color='pink')
    plt.show()


def main():
    res = tester("90_ws", 10, 2, 100)
    plot("90_ws", res[0], res[1])




if __name__ == '__main__':
    main()