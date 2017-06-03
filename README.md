## GraphTime
[![Build Status](https://travis-ci.org/GlooperLabs/GraphTime.svg?branch=master)](https://travis-ci.org/GlooperLabs/GraphTime)
[![codecov.io](https://codecov.io/gh/GlooperLabs/GraphTime/coverage.svg?branch=master)](https://codecov.io/gh/GlooperLabs/GraphTime?branch=master)
[![PyPI version](https://badge.fury.io/py/graphtime.svg)](https://badge.fury.io/py/graphtime)

Graphtime is a python package to estimate dynamic graphical models from
time series data. We are working on visualisation tools, further optimization,
and parameter estimation (similar to the standard Graphical Lasso).

#### Requirements

- numpy
- matplotlib

#### Installation

    pip install graphtime
    
#### Usage

```Python
from graphtime import GroupFusedGraphLasso
from graphtime.simulate import DynamicGraphicalModel
from graphtime.utils import get_change_points, plot_data_with_cps

# 1. Create Example Data
n_vertices = 10
n_edges = [10, 5]  # amount of edges in two sparse graphical models
T = 60  # sample length
changepoints = [30]  # change point indices
DGS = DynamicGraphicalModel(n_vertices, seed=2)
DGS.create_graphs(n_edges, use_seed=True)
X = DGS.sample(T, changepoints, use_seed=True)

# 2. Fit the Group Fused Graphical Lasso
gfgl = GroupFusedGraphLasso(lambda1=.25, lambda2=20, max_iter=500)
gfgl.fit(X)

cp_hist = get_change_points(gfgl.sparse_Theta, 1e-5, *X.shape)
change_points = [i for i, cp in enumerate(cp_hist) if cp > 0]
plot_data_with_cps(X, change_points, ymin=-5, ymax=5)
```

![image](https://cloud.githubusercontent.com/assets/7715036/26752693/6c631fde-4856-11e7-8d3f-9fba77047db1.png)

As set in our simulation, the actual changepoint was at t=30. The algorithm
is able to find the changepoint despite the visually obscure data.