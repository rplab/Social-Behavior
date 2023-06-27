# !/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Estelle Trieu 
# Created Date: 9/19/2022
# version ='2.0' Raghuveer Parthasarathy
# last modified June 23, 2023 by Raghu Parthasarathy
# ---------------------------------------------------------------------------
import numpy as np
import xlsxwriter
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------


    
def get_excel_file(dataset_name, circling, none, one, both, any_contact, 
head_body, tail_rub):
    """
    Creates an excel file of the relevant window frames for a
    set of social behaviors.

    Args:
        circling (array) : an array of circling window frames.
        none (array)     : an array of 90-degree orientation (none) window frames.
        one (array)      : an array of 90-degree orientation (one) window frames.
        both (array)     : an array of 90-degree orientation (both) window frames.
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
    sheet1.write('B1', 'Circling')
    sheet1.write('C1', 'None')
    sheet1.write('D1', 'One')
    sheet1.write('E1', 'Both')
    sheet1.write('F1', 'Any Contact')
    sheet1.write('G1', 'Head-body')
    sheet1.write('H1', 'Tail-rub')

    # All frame numbers
    # I don't like hard-coding this...
    for j in range(1,15001):
        sheet1.write(f'A{j+1}', str(j))
    
    # Each behavior
    for j in  range(circling.shape[1]):
        for k in range(circling[1,j]):
            sheet1.write(f'B{circling[0,j]+k+1}', "X".center(17))
    for j in  range(none.shape[1]):
        for k in range(none[1,j]):
            sheet1.write(f'C{none[0,j]+k+1}', "X".center(17))
    for j in  range(one.shape[1]):
        for k in range(one[1,j]):
            sheet1.write(f'D{one[0,j]+k+1}', "X".center(17))
    for j in  range(both.shape[1]):
        for k in range(both[1,j]):
            sheet1.write(f'E{both[0,j]+k+1}', "X".center(17))
    for j in  range(any_contact.shape[1]):
        for k in range(any_contact[1,j]):
            sheet1.write(f'F{any_contact[0,j]+k+1}', "X".center(17))
    for j in  range(head_body.shape[1]):
        for k in range(head_body[1,j]):
            sheet1.write(f'G{head_body[0,j]+k+1}', "X".center(17))
    for j in  range(tail_rub.shape[1]):
        for k in range(tail_rub[1,j]):
            sheet1.write(f'H{tail_rub[0,j]+k+1}', "X".center(17))

    workbook.close() 


def get_diagram(dataset_name, circling, none, one, both, any_contact, 
head_body, tail_rub):
    """
    Displays a sequencing plot for a set of social behaviors.

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
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
    plt.title(f"{dataset_name} Figure")
    plt.xlabel("Event Type")
    plt.scatter(np.ones(circling.size), circling, color='blue')
    plt.scatter(np.ones(none.size)*2, none, color='purple')
    plt.scatter(np.ones(one.size)*3, one, color='orange')
    plt.scatter(np.ones(both.size)*4, both, color='green')
    plt.scatter(np.ones(any_contact.size)*5, any_contact, color='red')
    plt.scatter(np.ones(head_body.size)*6, head_body, color='hotpink')
    plt.scatter(np.ones(tail_rub.size)*7, tail_rub, color='cyan')
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(['circling', 'none', 'one', 'both', 
    'any','head-body', 'tail-rub'])
    plt.savefig(f"{dataset_name}_Fig.png")
    
