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
sys.path.append('../')
from graphtime import *
from graphtime.utils import get_edges, get_change_points, plot_data_with_cps

for p in [5]:
    s = 0.2*p # Sparsity for this size of experiment
    # Simulate graphs
    for k in [0,1,2]:
        DGS = DynamicGraphicalModel(p, seed=2)
        DGS.graphs = ErdosRenyiPrecisionGraph(DGS.n_vertices, s)