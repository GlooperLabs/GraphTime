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
import pickle
#sys.path.append('../')
from graphtime import *
from graphtime.simulate import *
from graphtime.utils import get_edges, get_change_points, plot_data_with_cps, visualise_path


#sigma = np.load('data/sigma.npy')
#sigma_inv = np.load('data/sigma_inv.npy')
T = [150] # Steps
# The location of changepoints is given as a proportion of T
cp_rel = [0.33,0.66]
#cps = [150,300]   # Location of changepoints
K = len(cp_rel)    # Number of changepoints
P = 10 # Variables
s = 10 # Active Edges
eps = 0.000001 # Edge threshold epsilon
Nexp = len(T)   # Number of experimnents to perform
#edges = get_edges(sigma_inv[0], eps)
#change_points = get_change_points(sigma_inv, eps)

DGS = DynamicGraphicalModel(P, seed=2)
DGS.generate_graphs(n_edges_list=[s, s, s])

# LOOP OVER TIMELENGTHS
for nexp in range(Nexp):
    Tn = T[nexp]
    # Sample data from Graphical Model (also ouptuts ND array of ground truth thetas)
    [y, GT_Thetas] = DGS.sample(Tn, changepoints=[np.ceil(i*Tn) for i in cp_rel], 
                   ret_dgm=True)
    
    # EXPLORE SOLUTION PATH
    # The lambdas will be selected from a grid.
    # Need to give GFGL a path option to evaluate multiple lambdas...
    Nl1 = 10
    Nl2 = 5
    lam1 = np.linspace(1, 0.01,Nl1)
    lam2 = np.linspace(2,1,Nl2)
    #lam1 = np.logspace(1,-1,Nl1)
    #lam2 = np.logspace(1,0,Nl2)
    path = []
    k=0
    # Sparse solutions are generally found faseter, so start off with large lambda1
    for i in range(len(lam1)):
        for j in range(len(lam2)):
            # Initial solution is set to previous iterate
            if k>0:
                Isol = path[k-1].sol
            else:
                Isol = None
                
            # Don't pre-smooth??
            gfgl = GroupFusedGraphLasso(lambda1=lam1[i], lambda2=lam2[j], verbose=True,
                                        tol=1e-4, max_iter=500, pre_smooth=10, init_sol=Isol)
            gfgl.fit(y) # Main estimation routine...adds sparse_Theta and changepoints
            gfgl.evaluate(y,GT_Thetas) # Computes summary statistics of solution
            path.append(gfgl)    # Append object to path
            k=k+1   
            
# Create dictionary which summarises the solution            
exp_dict = {}
exp_dict['path'] = path
exp_dict['DGS'] = DGS
exp_dict['y'] = y
exp_dict['GT_Thetas'] = GT_Thetas
exp_dict['lam1'] = lam1
exp_dict['lam2'] = lam2

# Write to file
with open('sol_file.pkl', 'wb') as output:
    pickle.dump(exp_dict, output, pickle.HIGHEST_PROTOCOL)
    

