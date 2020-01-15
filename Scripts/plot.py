# -*- coding: utf-8 -*-

"""
	This is a Python3 Script
	Created by: Xuanqi HUANG
	Date: 2018/11/11
	
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def barplot_stakced(data, x, y, hue, hue_order=None, order=None, ax=None):
    """
    Input:
        Data: pandas.DataFrame
        x: str, Column's name
        y: str, Column's name
        hue:str, Column's name
        hue_order: list, index
        order: list, index
    Output:
        ax: matplotlib.pyplot.ax
    """
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    
    if ax == None:
        fig, ax = plt.subplots()
        
    if hue_order == None:
        hue_order = data[hue].unique()
    else:
        hue_order_all = data[hue].unique()
        for i in hue_order:
            if i not in hue_order_all:
                return '[hue_order] contain unexpected value :[%s]'%(str(i))
    
    if order == None:
        order = data[x].unique()
    else:
        order_all = data[x].unique()
        for i in order :
            if i not in order_all:
                return '[order] contain unexpected value :[%s]'%(str(i))
    
    # reformat data into list
    reformed_data = [[0 for i in range(len(order))]]
    
    for i in range(len(hue_order)):
        data_temp = data.loc[data[hue] == hue_order[i]].set_index(keys=x).loc[order, y].values.tolist()
        reformed_data.append(data_temp)
    
    #plot stacked bar
    for i in range(len(hue_order)):
        ax.bar(x=order, height=reformed_data[i+1], bottom=reformed_data[i])
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(hue_order)
    
    return ax