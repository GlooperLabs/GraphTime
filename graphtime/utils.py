import matplotlib.pyplot as plt
import numpy as np


def get_edges(G, eps, P=None):
    # get edges of adjacency matrix G
    # TODO: make use of array mask here
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


def precision(G_est, G_true, eps=1e-6):
    assert len(G_est) == len(G_true)
    n = len(G_est)
    precision = 0
    for i in range(n):
        est_edges = set(get_edges(G_est[i], eps))
        gt_edges = set(get_edges(G_true[i], eps))
        n_joint = len(est_edges.intersection(gt_edges))
        precision += n_joint / len(est_edges)
    return precision / n


def recall(G_est, G_true, eps=1e-6):
    assert len(G_est) == len(G_true)
    n = len(G_est)
    recall = 0
    for i in range(n):
        est_edges = set(get_edges(G_est[i], eps))
        gt_edges = set(get_edges(G_true[i], eps))
        n_joint = len(est_edges.intersection(gt_edges))
        recall += n_joint / len(gt_edges)
    return recall / n


def f1_score(G_est, G_true, eps=1e-6):
    prec = precision(G_est, G_true, eps)
    rec = recall(G_est, G_true, eps)
    return (2*prec*rec) / (prec+rec)


# BELOW IS UNTESTED
# def evalFit(Theta, X):
#     """ Reports model fit chacteristics of a given estimated dynamic
#     graphical model.

#     Inputs:
#     Theta -- Sparse estimate of precision
#     X -- raw data

#     Outputs:
#     Lt -- vector of likelihood for each timepoint
#     bic -- complexity adjusted measure of estimation performance
#     sparsity -- vector of solution sparsity (for each timepoint)
#     """

#     T = Theta.shape[0]
#     P = Theta.shape[1]

#     S = np.zeros((T, P, P))
#     # Init metrics, track for each time-point
#     bic = sparsity = np.zeros(T)
#     for t in range(0, T):
#         sparsity[t] = get_dof(Theta, thresh)
#         # Single sample outer product ala empirical covariance
#         S[t] = np.linalg.outer(X[t, :], X[t, :])

#     Lt = getLike(Theta, S)

#     # This may work with a moving average smoother
#     # but needs to be updated to take into account whole dataset
#     # for t in range(0, T):
#     #     According to standard BIC
#     #     bic[t] = (-(2 * Lt) + (sparsity[t] *
#     #                  np.log(2 * M + 1)))

#     return (Lt, bic, sparsity)

# def getLike(Theta, S, thresh=0.00001):
#     """ Finds likelihood and risk of estimated covariance given a set of
#     empirical (unregularised) covariance matrices"""

#     # A threshold for counting sparsity
#     T = Theta.shape[0]
#     Lt = np.zeros(T)

#     # I think this is correct up to a factor of 2
#     for t in range(0, T):
#         # The likelihood is calculated at each time point
#         Lt[t] = np.log(np.linalg.det(Theta[t])) - np.trace(
#             np.dot(Theta[t], S[t]))

#     return Lt

# def get_dof(Theta, thresh, P=None):
#     """ This works, checked (28/3/2017)
#     get edges of adjacency matrix Theta
#     Can probably just use len(get_edges(Theta))?
#     """

#     P = P or Theta.shape[0]

#     count = 0
#     for i in range(P - 1):
#         for j in range(i + 1, P):
#             if Theta[i, j] > thresh:
#                 count = count + 1

#     #Count diagonals
#     count = count + P

#     return count
