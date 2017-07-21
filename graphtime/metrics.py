import numpy as np
from graphtime.utils import get_edges


def precision(Theta_true, Theta_est, eps=1e-6, per_ts=False):
    """Compute the precision of graph recovery given true and estimated precision matrices.
    It is possible to average the precision score over all timesteps or get a list of
    precision scores for each timestep. For a global precision score, i.e. weighted
    average of precisions by the amount of detected edges, see `global_precision`.

    Parameters
    ----------
    Theta_true : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    Theta_est : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    eps : float
    per_ts : bool
        whether to compute average or per timestep precision

    Returns
    -------
    ndarray or float
        precision list or single precision value
    """
    assert len(Theta_est) == len(Theta_true)
    n = len(Theta_est)
    precision = [] if per_ts else 0
    for i in range(n):
        est_edges = set(get_edges(Theta_est[i], eps))
        gt_edges = set(get_edges(Theta_true[i], eps))
        n_joint = len(est_edges.intersection(gt_edges))
        latest = n_joint / len(est_edges) if len(est_edges) > 0 else 1
        precision += [latest] if per_ts else latest
    return np.array(precision) if per_ts else precision / n


def global_precision(Theta_true, Theta_est, eps=1e-6):
    """Compute precision where true positives are the number of correctly estimated
    edges and the denominator (true positives + false positives) is the amount of
    detected edges.

    Parameters
    ----------
    Theta_true : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    Theta_est : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    eps : float
        threshold to consider precision entry as edge

    Returns
    -------
    float precision
    """
    assert len(Theta_est) == len(Theta_true)
    n = len(Theta_est)
    tps, dets = 0, 0
    for i in range(n):
        est_edges = set(get_edges(Theta_est[i], eps))
        gt_edges = set(get_edges(Theta_true[i], eps))
        n_joint = len(est_edges.intersection(gt_edges))
        tps, dets = tps + n_joint, dets + len(est_edges)
    return tps / dets if dets > 0 else 1


def recall(Theta_true, Theta_est, eps=1e-6, per_ts=False):
    """Compute the recall of graph recovery given true and estimated precision matrices.
    It is possible to average the recall score over all timesteps or get a list of
    recall scores for each timestep. For a global recall score, i.e. weighted
    average of recall by the amount of detected and true edges, see `global_recall`.

    Parameters
    ----------
    Theta_true : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    Theta_est : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    eps : float
    per_ts : bool
        whether to compute average or per timestep recall

    Returns
    -------
    ndarray or float
        recall list or single precision value
    """
    assert len(Theta_est) == len(Theta_true)
    n = len(Theta_est)
    recall = [] if per_ts else 0
    for i in range(n):
        est_edges = set(get_edges(Theta_est[i], eps))
        gt_edges = set(get_edges(Theta_true[i], eps))
        n_joint = len(est_edges.intersection(gt_edges))
        latest = n_joint / len(gt_edges) if len(gt_edges) > 0 else 1
        recall += [latest] if per_ts else latest
    return np.array(recall) if per_ts else recall / n


def global_recall(Theta_true, Theta_est, eps=1e-6):
    """Compute precision where true positives are the number of correctly estimated
    edges and the denominator (true positives + false positives) is the amount of
    detected edges.

    Parameters
    ----------
    Theta_true : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    Theta_est : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    eps : float
        threshold to consider precision entry as edge

    Returns
    -------
    float precision
    """
    assert len(Theta_est) == len(Theta_true)
    n = len(Theta_est)
    tps, exps = 0, 0
    for i in range(n):
        est_edges = set(get_edges(Theta_est[i], eps))
        gt_edges = set(get_edges(Theta_true[i], eps))
        n_joint = len(est_edges.intersection(gt_edges))
        tps, exps = tps + n_joint, exps + len(gt_edges)
    return tps / exps if exps > 0 else 1


def f_score(Theta_true, Theta_est, beta=1, eps=1e-6, per_ts=False):
    """Compute f1 score in the same manner as `precision` and `recall`.
    Therefore see those two functions for the respective waiting and per_ts
    explanation.

    Parameters
    ----------
    Theta_true : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    Theta_est : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    beta : float (default 1)
        beta value of the F score to be computed
    eps : float
    per_ts : bool
        whether to compute average or per timestep recall

    Returns
    -------
    ndarray or float
        recall list or single precision value
    """
    prec = precision(Theta_true, Theta_est, eps, per_ts=True)
    rec = recall(Theta_true, Theta_est, eps, per_ts=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        nom = (1 + beta**2) * prec * rec
        print(beta**2 * prec)
        den = beta**2 * prec + rec
        f = np.nan_to_num(np.true_divide(nom, den))
    return f if per_ts else np.sum(f) / len(Theta_true)


def global_f_score(Theta_true, Theta_est, beta=1, eps=1e-6):
    """In line with `global_precision` and `global_recall`, compute the
    global f score given true and estimated graphical structures. The
    f score has the only parameter beta.

    Parameters
    ----------
    Theta_true : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    Theta_est : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    beta : float (default 1)
        beta value of the F score to be computed
    eps : float
    per_ts : bool
        whether to compute average or per timestep recall

    Returns
    -------
    float f-beta score
    """
    assert Theta_est.shape == Theta_true.shape
    d = Theta_true.shape[1]
    n = len(Theta_est)
    tps = fps = fns = tns = 0
    for i in range(n):
        est_edges = set(get_edges(Theta_est[i], eps))
        gt_edges = set(get_edges(Theta_true[i], eps))
        n_joint = len(est_edges.intersection(gt_edges))
        tps += n_joint
        fps += len(est_edges) - n_joint
        fns += len(gt_edges) - n_joint
        tns += d**2 - d - tps - fps - fns
    nom = (1 + beta**2) * tps
    denom = nom + beta**2 * fns + fps
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.nan_to_num(np.true_divide(nom, denom))
        return f


def n_estimated_edges(Theta_est, eps=1e-6, per_ts=True):
    """Sums up the number of edges in each individual precision graph and by default
    sums the number for each graph over the whole timeseries. per_ts allows to change
    this so only a scalar is returned

    Parameters
    ----------
    Theta_est : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    eps : float
    per_ts : bool
        whether to return array or sum the amount of edges up

    Returns
    -------
    list for edges or int for global sum
    """
    n_edges = [len(get_edges(G, eps)) for G in Theta_est]
    return n_edges if per_ts else sum(n_edges)


def changepoint_density(Theta_est, eps=1e-6, per_ts=True):
    """Sums up the number of edges changed either per timestep or globally. A changed
    edge exists at t if the weight of edge changes between timestep t-1 and t where the
    prior graph is equal to the first, so no changepoint is possible at t=1.

    Parameters
    ----------
    Theta_est : 3D ndarray, shape (timesteps, n_vertices, n_vertices)
    eps : float
    per_ts : bool
        whether to return array or sum the amount of edges up

    Returns
    -------
    list for changepoints or int for global sum
    """
    cps = [len(get_edges(Theta_est[t-1] - Theta_est[t], eps))
           for t in range(1, len(Theta_est))]
    cps = [0] + cps
    return cps if per_ts else sum(cps)


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
