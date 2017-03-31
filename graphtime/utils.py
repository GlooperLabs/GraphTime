import matplotlib.pyplot as plt
import numpy as np


def get_edges(G, eps, P=None):
    # get edges of adjacency matrix G
    P = P or G.shape[0]
    return [(i, j) for i in range(P-1) for j in range(i+1, P) if abs(G[i, j]) > eps]


def get_change_points(Theta, eps, T=None, P=None):
    # calculate histogram of change points of T adjacency matrices
    T = T or Theta.shape[0]
    P = P or Theta.shape[1]
    # difference between consecutive adjacency matrices
    Delta_Theta = np.diff(Theta, axis=0)
    return [len(get_edges(G, eps, P)) for G in Delta_Theta]


def plot_data_with_cps(data, cps, ymin, ymax):
    plt.plot(data, alpha=0.5)
    for cp in cps:
        plt.plot([cp, cp], [ymin, ymax], 'k-')
    plt.axis([0, len(data), ymin, ymax], 'k-')
    plt.show()