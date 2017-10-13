import matplotlib.pyplot as plt
import numpy as np


def get_edges(G, eps):
    P = G.shape[0]
    return [(i, j) for i in range(P-1) for j in range(i+1, P)
            if abs(G[i, j]) >= eps]

def binary_class_eval(est,truth):
    """ Reports true positive, false positive, false negative and true negatives.
    Input is two lists of equal length populated by one or zero"""
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(est)): 
        if truth[i]==est[i]==1:
           TP += 1
    for i in range(len(est)): 
        if est[i]==1 and truth[i]!=est[i]:
           FP += 1
    for i in range(len(est)): 
        if truth[i]==est[i]==0:
           TN += 1
    for i in range(len(est)): 
        if est[i]==0 and truth[i]!=est[i]:
           FN += 1

    return(TP, FP, TN, FN)

def graph_F_score(G_est,G_true,beta):
    """Reports the F_beta score of a graph estimate"""
    P = G_est.shape[0]
    iu1 = np.triu_indices(P,1)  # Get upper off-diag components
    Gest = G_est[iu1]
    Gtrue = G_true[iu1]
    #Gest = np.extract(1 - np.eye(P),G_est)
    #Gtrue = np.extract(1 - np.eye(P),G_true)
    [tp,fp,tn,fn]=binary_class_eval(Gest,Gtrue)
    bottom = ((1 + (beta ** 2)) * tp ) + ((beta ** 2) * fn) + fp
    top = (1 + (beta **2)) * tp
    if bottom != 0:
        F = top/bottom
    else:
        F = 0
        
    return F

def graph_F_score_dynamic(G_est,G_true,beta):
    """Dynamic version of graph_F_score. Produces an average F_score and 
    time-varying F_score"""
    T = G_est.shape[0]
    F = np.zeros([T])
    for t in range(T):
        F[t] = graph_F_score(G_est[t],G_true[t],beta)
    
    Favg = np.mean(F)
    return(Favg, F)
    
def get_likelihood(Theta_est,Sigma):
    """ Calculate likelihood for dynamic GGM """
    T = len(Theta_est)
    L=0
    for t in range(T):
        L = L + (np.log(np.linalg.det(Theta_est[t])) - 
                 np.trace( Sigma[t].dot(Theta_est[t]) ) )
    # Normalise Likelihood for data points
    L = L/T
    return L

def get_CPerr(cp_est,cp_true):
    """ Finds minimax cp error as per Kolar/Xing etc """
    Ktrue = len(cp_true)
    Kest = len(cp_est)
    min_err = 10000*np.ones(Ktrue)  # Set min error to be intially large
    for k in range(Ktrue):
        # Find closest estimate to truth k
        for l in range(Kest):
            err = np.abs(cp_est[l]-cp_true[k])
            if err<min_err[k]:
                min_err[k] = err
        
    return min_err.max()
                
    
def get_BIC(Theta_est,Sigma,dof):
    """ Calculates BIC based on estimate and ground-truth precision matrices
    This does not calculate dof, you need to pass it"""
    T = Theta_est.shape[0]
    BIC = -2*get_likelihood(Theta_est,Sigma) + dof*np.log(T)
    return BIC

def get_change_points(Thetas, eps):
    cps = [i for i in range(1, len(Thetas)) if not
           np.allclose(Thetas[i-1], Thetas[i], atol=eps, rtol=0)]
    return cps

def get_edge_dif(Thetas,diag=False):
    """A function to summarise variation in the estimated matrices
    """
    T = Thetas.shape[0]
    P = Thetas.shape[1]
    dif = np.zeros(T)
    if diag==False:
        for t in range(1,T):
            dif[t] = np.linalg.norm(np.extract(1 - np.eye(P),
                                    Thetas[t]-Thetas[t-1]),2)         
    else:
        for t in range(1,T):
            dif[t] = np.linalg.norm(Thetas[t]-Thetas[t-1],'fro')
    return dif

def plot_edge_dif(Thetas,ymin=None,ymax=None):
    
    dif_off_diag = get_edge_dif(Thetas,diag=False)
    dif_diag = get_edge_dif(Thetas,diag=True)
    
    ymin = np.min(dif_diag) if not ymin else ymin
    ymax = np.max(dif_diag) if not ymax else ymax
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dif_off_diag, alpha=0.5)
    ax.plot(dif_diag,alpha=0.5)
    ax.set_ylabel('Precision Diff')
    ax.set_xlabel('Timestep')
    return fig

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

def visualise_path(path, lam1, lam2, metric='Fscore'):
    """ Visualise the solution path of estimator in terms of contour plot"""
    k=0
    Z = np.zeros([len(lam1),len(lam2)])
    # Reshape path to match lambda grid
    for i in range(len(lam1)):
        for j in range(len(lam2)):
            if metric=='Fscore':
                Z[i,j] = path[k].Favg
                k=k+1
            elif metric=='BIC':
                Z[i,j] = path[k].BIC
            elif metric=='CPerr':
                Z[i,j] = path[k].CPerr
                            
    print('hello')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cs = ax.contourf(lam1,lam2,Z.T)
    ax.set_ylabel('Lambda 2')
    ax.set_xlabel('Lambda 1')
    fig.colorbar(cs)
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
    X : 2D ndarray, shape (timesteps, variables)
    """
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return X

def kernel_smooth(X, M):
    """Outputs a smoothed empirical covariance matrix with moving window. This
    assumes that the window is square and does not apply any weighting to data
    within the kernel.
    
    Parameters
    ----------
    X : 2D ndarray, shape (timesteps, variables)
    M : int
        width of kernel smoothing window

    Returns
    -------
    S : 3D ndarray, shape (timesteps, variables, variables)
    """
    T, P = X.shape
    S = np.zeros((T, P, P))
    # Split into three parts (end regions and central block)
    for t in range(M):
        S[t] = (X[0:t+M,:].T).dot(X[0:t+M,:])/(t+M)
    for t in range(M,T-M):
        S[t] = (X[t-M:t+M,:].T).dot(X[t-M:t+M,:])/(2*M+1)
    for t in range(T-M,T):
        S[t] = (X[t:-1,:].T).dot(X[t:-1,:])/((T-t)+M)
        
    return S
        
    
    

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