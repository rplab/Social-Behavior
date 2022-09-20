# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/19/2022
# version ='1.0'
# ---------------------------------------------------------------------------
from circling import *
from nintey_degrees import *
from tail_rubbing import *
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------

def get_results_txt(circling, none, one, both, any, head_body, tail_rub):
    with open("results.txt", "w") as results_file:
        results_file.write(f"Circling:\n {circling}\n\n")
        results_file.write(f"90-degrees:\n none: {none} \n\n one: {one} " +
        f"\n\n both: {both} \n\n")
        results_file.write(f"Any Contact:\n {any} \n\nHead-Body Contact:\n " +
        f"{head_body} \n\nTail-rubbing:\n {tail_rub}")


def get_diagram(circling, none, one, both, any, head_body, tail_rub):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
    plt.title("Results Diagram")
    plt.scatter(np.ones(circling.size), circling, color='blue')
    plt.scatter(np.ones(none.size)*2, none, color='purple')
    plt.scatter(np.ones(one.size)*3, one, color='orange')
    plt.scatter(np.ones(both.size)*4, both, color='green')
    plt.scatter(np.ones(any.size)*5, any, color='red')
    plt.scatter(np.ones(head_body.size)*6, head_body, color='hotpink')
    plt.scatter(np.ones(tail_rub.size)*7, tail_rub, color='cyan')
    ax.set_xticklabels(['filler', 'circling', 'none', 'one', 'both', 
    'any','head-body', 'tail-rub'])
    plt.savefig('Results_Fig.png')
   

def main():
    pos_data = load_data("results_SocPref_3c_2wpf_k1_ALL.csv", 3, 5)
    angle_data = load_data("results_SocPref_3c_2wpf_k1_ALL.csv", 5, 6)
    contact_x = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 6, 16)
    contact_y = load_data("results_SocPref_3c_2wpf_nk1_ALL.csv", 16, 26)

    circling_wfs = get_circling_wf(pos_data[0], pos_data[1], 
    angle_data, 10)

    # 90-degrees 
    orientation_dict = get_90_deg_wf(pos_data, angle_data, 10)
    none = orientation_dict["none"]
    one = orientation_dict["1"]
    both = orientation_dict["both"]

    # Any contact
    contact_wf = get_contact_wf(contact_x[0], contact_x[1], 
    contact_y[0], contact_y[1], 1)
    any = contact_wf["any"]
    head_body = contact_wf["head-body"]

    # Tail-rubbing
    tail_rubbing_wf = get_tail_rubbing_wf(contact_x[0], contact_x[1], 
    contact_y[0], contact_y[1], pos_data, angle_data, 4)

    get_results_txt(circling_wfs, none, one, both, any, head_body,
    tail_rubbing_wf)
    get_diagram(circling_wfs, none, one, both, any, head_body, 
    tail_rubbing_wf)




if __name__ == '__main__':
    main()
