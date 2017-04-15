from multiprocessing import Pool, cpu_count
from itertools import product, repeat
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


def precision(G_est, G_true, eps=1e-6, per_ts=False):
    assert len(G_est) == len(G_true)
    n = len(G_est)
    precision = [] if per_ts else 0
    for i in range(n):
        est_edges = set(get_edges(G_est[i], eps))
        gt_edges = set(get_edges(G_true[i], eps))
        n_joint = len(est_edges.intersection(gt_edges))
        latest = n_joint / len(est_edges) if n_joint > 0 else 0
        precision += [latest] if per_ts else latest
    return np.array(precision) if per_ts else precision / n


def recall(G_est, G_true, eps=1e-6, per_ts=False):
    assert len(G_est) == len(G_true)
    n = len(G_est)
    recall = [] if per_ts else 0
    for i in range(n):
        est_edges = set(get_edges(G_est[i], eps))
        gt_edges = set(get_edges(G_true[i], eps))
        n_joint = len(est_edges.intersection(gt_edges))
        latest = n_joint / len(gt_edges) if n_joint > 0 else 0
        recall += [latest] if per_ts else latest
    return np.array(recall) if per_ts else recall / n


def f1_score(G_est, G_true, eps=1e-6, per_ts=False):
    prec = precision(G_est, G_true, eps, per_ts)
    rec = recall(G_est, G_true, eps, per_ts)
    with np.errstate(divide='ignore', invalid='ignore'):
        nom = 2 * prec * rec
        den = prec + rec
        f1 = np.nan_to_num(np.true_divide(nom, den))
    return f1


def performance(G_est, G_true):
    prec = precision(G_est, G_true)
    rec = recall(G_est, G_true)
    f1 = f1_score(G_est, G_true)
    return prec, rec, f1


def grid_search(model, y, G_true, lam1s, lam2s, tol=1e-4, max_iter=500,
                gamma1=1, gamma2=1, gammaw=1, n_processes='max'):
    performances = dict()
    n_processes = max(cpu_count()-1, 1) if n_processes == 'max' else n_processes
    settings = repeat((gamma1, gamma2, gammaw, tol, max_iter))
    lam_cross = product(lam1s, lam2s)
    with Pool(n_processes) as pool:
        arguments = [(model, y, G_true, *lams, *args)
                     for lams, args in zip(lam_cross, settings)]
        perfs = pool.starmap(evaluate, arguments)
    for perf in perfs:
        performances.update(perf)
    return performances


def evaluate(model, y, G_true, lam1, lam2, gamma1, gamma2, gammaw, tol, max_iter):
    dglm = model(lam1, lam2, gamma1, gamma2, gammaw, tol, max_iter)
    dglm.fit(y)
    return {(lam1, lam2): performance(dglm.sparse_Theta, G_true)}


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
