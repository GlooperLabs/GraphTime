import matplotlib.pyplot as plt
import numpy as np


def get_edges(G, eps):
    P = G.shape[0]
    return [(i, j) for i in range(P-1) for j in range(i+1, P)
            if abs(G[i, j]) >= eps]


def get_change_points(Thetas, eps):
    cps = [i for i in range(1, len(Thetas)) if not
           np.allclose(Thetas[i-1], Thetas[i], atol=eps, rtol=0)]
    return cps


def plot_data_with_cps(data, cps, ymin=None, ymax=None):
    ymin = np.min(data) if not ymin else ymin
    ymax = np.max(data) if not ymax else ymax
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data, alpha=0.5)
    ax.set_ylabel('Values')
    ax.set_xlabel('Timestep')
    for cp in cps:
        ax.plot([cp, cp], [ymin, ymax], 'k-')
    ax.set_xlim([0, len(data)])
    ax.set_ylim([ymin, ymax])
    return fig


def soft_threshold(X, thresh):
    """Proximal mapping of l1-norm results in soft-thresholding. Therefore, it is required
    for the optimisation of the GFGL or IFGL.

    Parameters
    ----------
    X : ndarray
        input data of arbitrary shape
    thresh : float
        threshold value

    Returns
    -------
    ndarray soft threshold applied
    """
    return (np.absolute(X) - thresh).clip(0) * np.sign(X)


def scale_standard(X):
    """Scale data X to unit variance and zero mean. Useful before using any graphical
    estimator

    Parameters
    ----------
    X : 2D ndarray, shape (timesteps, variables)

    Returns
    -------
    X : 2d ndarray, shape (timesteps, variables)
    """
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return X


"""
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
"""