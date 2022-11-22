# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/19/2022
# version ='1.0'
# ---------------------------------------------------------------------------
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------

def get_txt_file(dataset_name, circling, none, one, both, any, 
head_body, tail_rub):
    """
    Displays an txt file of the relevant window frames for a
    set of social behaviors.

    Args:
        circling (array) : an array of circling window frames.
        none (array)     : an arary of 90-degree orientation (none) window frames.
        one (array)      : an arary of 90-degree orientation (one) window frames.
        both (array)     : an arary of 90-degree orientation (both) window frames.
        any (array)      : an array of fish contact (any) window frames.
        head_body (array): an array of fish contact (head-body) window frames.
        tail_rub (array) : an array of tail-rubbing window frames.

    Returns:
        N/A
    """
    with open(f"{dataset_name}.txt", "w") as results_file:
        results_file.write(f"Circling:\n {circling}\n\n")
        results_file.write(f"90-degrees:\n none: {none} \n\n one: {one} " +
        f"\n\n both: {both} \n\n")
        results_file.write(f"Any Contact:\n {any} \n\nHead-Body Contact:\n " +
        f"{head_body} \n\nTail-rubbing:\n {tail_rub}")


def get_excel_file(dataset_name, circling, none, one, both, any, 
head_body, tail_rub):
    """
    Displays an excel file of the relevant window frames for a
    set of social behaviors.

    Args:
        circling (array): an array of circling window frames.
        none (array)     : an arary of 90-degree orientation (none) window frames.
        one (array)      : an arary of 90-degree orientation (one) window frames.
        both (array)     : an arary of 90-degree orientation (both) window frames.
        any (array)      : an array of fish contact (any) window frames.
        head_body (array): an array of fish contact (head-body) window frames.
        tail_rub (array) : an array of tail-rubbing window frames.

    Returns:
        N/A
    """
    # Initialize notebook
    workbook = xlsxwriter.Workbook(f'{dataset_name}_excel_file.xlsx')  
    sheet1 = workbook.add_worksheet('Sheet 1')

    # Headers 
    sheet1.write('A1', 'Frame') 
    sheet1.write('B1', 'circling')
    sheet1.write('C1', 'none')
    sheet1.write('D1', 'one')
    sheet1.write('E1', 'both')
    sheet1.write('F1', 'any')
    sheet1.write('G1', 'head-body')
    sheet1.write('H1', 'tail-rub')

    # Set frame numbers 
    for i in range(1, 15001):
        sheet1.write(f'A{i+1}', str(i))

        if i in circling:
            sheet1.write(f'B{i+1}', "X".center(17))
        if i in none:
            sheet1.write(f'C{i+1}', "X".center(17))
        if i in one:
            sheet1.write(f'D{i+1}', "X".center(17))
        if i in both:
            sheet1.write(f'E{i+1}', "X".center(17))
        if i in any:
            sheet1.write(f'F{i+1}', "X".center(17))
        if i in head_body:
            sheet1.write(f'G{i+1}', "X".center(17))
        if i in tail_rub:
            sheet1.write(f'H{i+1}', "X".center(17))

    workbook.close() 


def get_diagram(dataset_name, circling, none, one, both, any, 
head_body, tail_rub):
    """
    Displays a sequencing plot for a set of social behaviors.

    Args:
        circling (array): an array of circling window frames.
        none (array)     : an arary of 90-degree orientation (none) window frames.
        one (array)      : an arary of 90-degree orientation (one) window frames.
        both (array)     : an arary of 90-degree orientation (both) window frames.
        any (array)      : an array of fish contact (any) window frames.
        head_body (array): an array of fish contact (head-body) window frames.
        tail_rub (array) : an array of tail-rubbing window frames.

    Returns:
        N/A
    """
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
    plt.title(f"{dataset_name} Figure")
    plt.xlabel("Event Type")
    plt.scatter(np.ones(circling.size), circling, color='blue')
    plt.scatter(np.ones(none.size)*2, none, color='purple')
    plt.scatter(np.ones(one.size)*3, one, color='orange')
    plt.scatter(np.ones(both.size)*4, both, color='green')
    plt.scatter(np.ones(any.size)*5, any, color='red')
    plt.scatter(np.ones(head_body.size)*6, head_body, color='hotpink')
    plt.scatter(np.ones(tail_rub.size)*7, tail_rub, color='cyan')
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(['circling', 'none', 'one', 'both', 
    'any','head-body', 'tail-rub'])
    plt.savefig(f"{dataset_name}_Fig.png")
