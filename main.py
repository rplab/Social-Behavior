# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/19/2022
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
    "rad_thresh" : 0.005,
    "rmse_thresh" : 25,
    "circ_anti_low" : -1,
    "circ_anti_high" : -0.9,
    "90_ws" : 10,
    "theta_90_thresh" : 0.1,
    "90_head_dist" : 300,
    "contact_ws" : 1,
    "contact_dist" : 20,
    "tail_rub_ws" : 4,
    "tail_dist" : 40,
    "tail_anti_low": -1,
    "tail_anti_high": -0.8,
}

def main(): 
    pos_data = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 3, 5)
    angle_data = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 5, 6)
    contact_x = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 6, 16)
    contact_y = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 16, 26)

    # End of array should be the same for all loaded data
    end_of_arr = np.shape(pos_data)[1] 
    dataset_name = re.search('\d[a-z]_\d[a-z]{3}_[a-z]{1,2}\d', 
    'results_SocPref_3c_2wpf_nk1_ALL.csv').group()

    circling_wfs = get_circling_wf(pos_data[0], pos_data[1], angle_data, 
    end_of_arr, params["circ_ws"], params["rad_thresh"], params["rmse_thresh"],
    params["circ_anti_low"], params["circ_anti_high"])

    # 90-degrees 
    orientation_dict = get_90_deg_wf(pos_data, angle_data, end_of_arr,
    params["90_ws"], params["theta_90_thresh"], params["90_head_dist"])
    none = orientation_dict["none"]
    one = orientation_dict["1"]
    both = orientation_dict["both"]

    # Any contact
    contact_wf = get_contact_wf(contact_x[0], contact_x[1], contact_y[0], 
    contact_y[1], end_of_arr, params["contact_ws"], params["contact_dist"])
    any = contact_wf["any"]
    head_body = contact_wf["head-body"]

    # Tail-rubbing
    tail_rubbing_wf = get_tail_rubbing_wf(contact_x[0], contact_x[1], 
    contact_y[0], contact_y[1], pos_data, angle_data, end_of_arr, 
    params["tail_rub_ws"], params["tail_dist"], params["tail_anti_low"],
    params["tail_anti_high"])

    get_txt_file(dataset_name, circling_wfs, none, one, both, any, 
    head_body, tail_rubbing_wf)
    get_diagram(dataset_name, circling_wfs, none, one, both, any, 
    head_body, tail_rubbing_wf)
    get_excel_file(dataset_name, circling_wfs, none, one, both, any, 
    head_body, tail_rubbing_wf)




if __name__ == '__main__':
    main()
