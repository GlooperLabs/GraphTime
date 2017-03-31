## GraphTime

Graphtime is a python package to estimate dynamical graphical models from
time series data. Further, it is planned to add visualisation tools and
more examples in the future.

#### Dependencies

- numpy
- matplotlib

#### Installation

    pip install graphtime
    
### Usage

```Python
import numpy as np
from graphtime import GroupFusedGraphLasso
from graphtime.utils import get_change_points, plot_data_with_cps

y = np.load('data/y.npy') # load example data

gfgl = GroupFusedGraphLasso(lambda1=.15, lambda2=25, max_iter=500)
gfgl.fit(y)

cp_hist = get_change_points(gfgl.sparse_Theta, 0.01, *y.shape)
change_points = [i for i, cp in enumerate(cp_hist) if cp > 0]
plot_data_with_cps(y, change_points, ymin=-5, ymax=5)
```

![image](https://cloud.githubusercontent.com/assets/7715036/24554698/c636472a-162e-11e7-99a1-6d6a8a3c49f8.png)

The actual change points are at t=30 and t=60.