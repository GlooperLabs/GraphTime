#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 18:24:33 2017
Analysis of fMRI data during task on/off 
@author: alex
"""

import numpy as np
import sys
import pickle
from graphtime import *
from graphtime.simulate import *
from graphtime.utils import get_edges, get_change_points, plot_data_with_cps, visualise_path
import scipy.io as sio

# Derived from hemodynamic response function (task off/on/off/on.../off)
T = 126;
# Could be out +-1 due to differencing in pre-processing..
cps = [3, 17, 31, 42, 56, 66, 80, 91, 105, 115];
P = 18;
Nsub = 28;
# Pre-processing performed in MATLAB
# Difference the raw signal along time for each patient
# => We model the correlation in "changes" in the fMRI signal
# Z-score over patients
# Input at this point is set of empirical covariance matrices (PxP)

# Read in pre-computed empirical covariance
Data = sio.loadmat('bio_experiment/fmri_cov.mat')
Semp = Data['Cz']  # Extract empirical cov from data

Nl1 = 4
Nl2 = 2
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
            
        gfgl = GroupFusedGraphLasso(lambda1=lam1[i], lambda2=lam2[j], verbose=True,
                                    tol=1e-4, max_iter=500, pre_smooth=0, init_sol=Isol)
        gfgl.fit(Semp) # Main estimation routine...adds sparse_Theta and changepoints
        gfgl.evaluate() # Computes summary statistics of solution
        path.append(gfgl)    # Append object to path
        k=k+1   
            
# Create dictionary which summarises the solution            
exp_dict = {}
exp_dict['path'] = path
#exp_dict['DGS'] = DGS
#exp_dict['y'] = y
exp_dict['Semp'] = Semp
# exp_dict['GT_Thetas'] = GT_Thetas
exp_dict['lam1'] = lam1
exp_dict['lam2'] = lam2

# Write to file
with open('sol_file.pkl', 'wb') as output:
    pickle.dump(exp_dict, output, pickle.HIGHEST_PROTOCOL)