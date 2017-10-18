#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 08:49:46 2017

This is an experiment to compare optimisation time with and without pre-smoothing
We need to demonstrate that the same or similar solution is found in both cases
but the pre-smoothed version converges faster

@author: alex
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
import time
#sys.path.append('../')
from graphtime import *
from graphtime.simulate import *
from graphtime.utils import get_edges, get_change_points, plot_data_with_cps, visualise_path

T = 50 # Steps
cp = 25
K = len(cp_rel)    # Number of changepoints
P = 5 # Variables
s = 5 # Active Edges
eps = 0.000001 # Edge threshold epsilon
Nexp = 3
Nl1 = 4
Nl2 = 2
lam1 = np.linspace(1, 0.01,Nl1)
lam2_smooth = np.linspace(2,1,Nl2)
lam2_raw = np.linspace(40,20,Nl2)
    
DGS = DynamicGraphicalModel(P, seed=2)
DGS.generate_graphs(n_edges_list=[s,s])

# Strategy is to simulate data, run Nexp=20 experiments
# choose average lambda1,lambda2 which minimises average (over Nexp) 
# frobenius distance between Theta_pre_smooth and Theta_non_smooth
delta = np.zeros([Nexp,Nl1*Nl2,Nl1*Nl2])    # Init empty set for tracking differences in Theta
dt_smooth = np.zeros([Nexp,Nl1,Nl2])
dt_raw = np.zeros([Nexp,Nl1,Nl2])

for n in range(Nexp):
    [y, GT_Thetas] = DGS.sample(T, changepoints=[cp], ret_dgm=True)
    path_smooth = []
    path_raw = []
    k_raw =0
    k_smooth=0
    print('Experiment',n)
    # Compute solutions over grids for smoothing and raw
    for i in range(Nl1):
        for j in range(Nl2):
            # For warm start
            if k_smooth>0:
                Isol_smooth = path_smooth[k_smooth-1].sol
            else:
                Isol_smooth = None
                
            gfgl = GroupFusedGraphLasso(lambda1=lam1[i], lambda2=lam2_smooth[j], verbose=True,
                                        tol=1e-4, max_iter=500, pre_smooth=10, init_sol=Isol_smooth)
            start_time = time.time()
            gfgl.fit(y) # Main estimation routine...adds sparse_Theta and changepoints
            end_time = time.time()
            dt_smooth[n,i,j] = end_time-start_time
            
            gfgl.evaluate(y,GT_Thetas) # Computes summary statistics of solution
            path_smooth.append(gfgl)    # Append object to path
            k_smooth=k_smooth+1   
            
        # Do the loop for the raw (no pre-smoothing) version
        for j in range(Nl2):
            # For warm start
            if k_raw>0:
                Isol_raw = path_raw[k_raw-1].sol
            else:
                Isol_raw = None
            gfgl = GroupFusedGraphLasso(lambda1=lam1[i], lambda2=lam2_raw[j], verbose=True,
                                        tol=1e-4, max_iter=2000, pre_smooth=0, init_sol=Isol_raw)
            start_time = time.time()
            gfgl.fit(y) # Main estimation routine...adds sparse_Theta and changepoints
            end_time = time.time()
            dt_raw[n,i,j] = end_time-start_time
            
            gfgl.evaluate(y,GT_Thetas) # Computes summary statistics of solution
            path_raw.append(gfgl)    # Append object to path
            k_raw=k_raw+1
            
    # Compute the difference between solutions along path
    for k_r in range(Nl1*Nl2):
        for k_s in range(Nl1*Nl2):
            delta[n,k_r, k_s] = np.linalg.norm(path_smooth[k_r].sparse_Theta.flatten()-
                                               path_raw[k_s].sparse_Theta.flatten(),2)
                        
            
# End of main loop
# Take average over distances to find min
mean_delta = np.mean(delta,axis=0)  # Compress over the Nexp dimension
min_delta = np.min(mean_delta)      # Returns actual minimum (mean) difference in solutions
min_idx = np.argmin(mean_delta)     # Returns index across flattend KxK array
kidx = np.unravel_index(min_idx,(K_smooth,K_raw))
lambda_s_id = np.unravel_index(kidx[0],(Nl1,Nl2))   # Get the lambda pair for smoothed case
lambda_r_id = np.unravel_index(kidx[1],(Nl2,Nl2))   # Get lambda pair for raw case

