#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:49:25 2017
The aim is to examine GraphTime performance over a fixed dimension p, but 
increasing data T. The number and relative position of changepoints is fixed.

@author: Alex Gibberd
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
#sys.path.append('../')
from graphtime import *
from graphtime.simulate import *
from graphtime.utils import get_edges, get_change_points, plot_data_with_cps, kernel_smooth


#sigma = np.load('data/sigma.npy')
#sigma_inv = np.load('data/sigma_inv.npy')
T = 90 # Steps
cps = [30,60]   # Location of changepoints
K = len(cps)    # Number of changepoints
P = 10 # Variables
S = 5 # Active Edges
eps = 0.000001 # Edge threshold epsilon

#edges = get_edges(sigma_inv[0], eps)
#change_points = get_change_points(sigma_inv, eps)

DGS = DynamicGraphicalModel(P, seed=2)
DGS.generate_graphs(n_edges_list=[S, S, S])
y = DGS.sample(T, changepoints=cps)

S = kernel_smooth(y, 10)

#DGS.draw();
plot_data_with_cps(y, cps, ymin=-5, ymax=5)
