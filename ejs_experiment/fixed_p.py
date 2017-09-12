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
T = [150,450,900] # Steps
# The location of changepoints is given as a proportion of T
cp_rel = [0.33,0.66]
#cps = [150,300]   # Location of changepoints
K = len(cps)    # Number of changepoints
P = 4 # Variables
n = 2 # Active Edges
eps = 0.000001 # Edge threshold epsilon
Nexp = len(T)   # Number of experimnents to perform
#edges = get_edges(sigma_inv[0], eps)
#change_points = get_change_points(sigma_inv, eps)

DGS = DynamicGraphicalModel(P, seed=2)
DGS.generate_graphs(n_edges_list=[n, n, n])

for nexp in range(Nexp):
    Tn = T[nexp]
    y = DGS.sample(Tn, changepoints=[np.ceil(i*Tn) for i in cp_rel])
    
    # The lambdas will be selected from a grid.
    # Need to give GFGL a path option to evaluate multiple lambdas...
    lam1 = 0.1;
    lam2 = 8;
    
    gfgl = GroupFusedGraphLasso(lambda1=lam1,lambda2=lam2,verbose=True,
                                tol=1e-4, max_iter=500,pre_smooth=10)
    gfgl.fit(y)
    
    cp_hist = get_change_points(gfgl.sparse_Theta, 1e-2)
    change_points = [i for i, cp in enumerate(cp_hist) if cp > 0]
    plot_data_with_cps(y, cp_hist, ymin=-5, ymax=5)
    plot_edge_dif(gfgl.sparse_Theta)
