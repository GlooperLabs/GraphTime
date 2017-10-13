#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:06:34 2017

@author: alex

Opens a saved regularisation path solution

"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
#sys.path.append('../')
from graphtime import *
from graphtime.simulate import *
from graphtime.utils import get_edges, get_change_points, plot_data_with_cps, visualise_path

# Open path solution
with open('sol_file.pkl', 'rb') as input:
    exp_dict = pickle.load(input)

# Extract individual parts from sol file
path = exp_dict['path']
DGS = exp_dict['DGS']
y = exp_dict['y']
GT_Thetas = exp_dict['GT_Thetas']
lam1 = exp_dict['lam1']
lam2 = exp_dict['lam2']

visualise_path(path,lam1,lam2,metric='Fscore')

#plot_data_with_cps(y, cp_hist, ymin=-5, ymax=5)
#plot_edge_dif(gfgl.sparse_Theta)