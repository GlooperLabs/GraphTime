import numpy as np
import matplotlib.pyplot as plt


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


def soft_threshold(X, thresh):
    # according to http://www.simonlucey.com/soft-thresholding/
    return (np.absolute(X) - thresh).clip(0) * np.sign(X)


def convert_to_sparse(V1, W):
    T, P, _ = V1.shape
    Theta = np.zeros(V1.shape)
    Theta[0] = V1[0]
    for t in range(T-1):
        Theta[t+1] = Theta[t] + W[t]
    return Theta


def gtADMM(X, lambda1, lambda2, gamma_v1, gamma_v2, gamma_w, max_iter, tol, smoother='GFGL', verbose=False):
    assert smoother in ['GFGL', 'IFGL']
    T, P = X.shape
    tolDual = tol
    tolPrime = tol
    # don't use pre-smoothing
    S = np.zeros((T, P, P))
    for t in range(T):
        S[t] = X[t] * X[t].T
    # dev with kernel pre-smoothing maybe?

    # auxiliary variables
    V1 = np.zeros((T, P, P))
    V2 = np.zeros((T - 1, P, P))
    # differencing aux variable
    W = np.zeros((T - 1, P, P))

    # Initialise Theta (Primal)
    U = np.zeros((T, P, P))
    for t in range(T):
        U[t] = np.eye(P)

    # change auxiliaries
    V1 = U.copy()
    V2 = U[:T - 1].copy()

    # init dual variables
    dV1 = V1.copy()
    dV2 = V2.copy()
    dW = W.copy()

    n_iter = 0

    # init convergence criteria
    eps_primal = [tolPrime + 1]
    eps_dual = [tolDual + 1]

    # ADMM loop
    while (eps_primal[n_iter] > tolPrime and eps_dual[n_iter] > tolDual and n_iter < max_iter):
        # solve step 1 data-update through eigendecomposition
        for t in range(T):
            sum_gamma = gamma_v1 + gamma_v2
            # Construct Gamma
            # (gv1.*(V1(:,:,t)-dV1(:,:,t))+gv2.*(V2(:,:,t)-dV2(:,:,t)))./(gv1+gv2)
            if t < T - 1:
                Gamma = (gamma_v1 * (V1[t] - dV1[t]) + gamma_v2 * (V2[t] - dV2[t])) / sum_gamma
            else:  # in last round
                Gamma = gamma_v1 * (V1[t] - dV1[t])

            gbar = sum_gamma / 2
            Sd, L = np.linalg.eig(S[t] * -2 * gbar * Gamma)  # eig in werte and vectors

            E = np.zeros(P)
            for r in range(P):
                sr = Sd[r]
                theta_r = (sr - np.sqrt(sr ** 2 + 8 * gbar)) / (-4 * gbar)
                E[r] = theta_r

            E = np.diag(E)
            U[t] = L.dot(E).dot(L.T) # reconstruct Theta

        # update auxiliary precision estimates
        Gamma = U[0] + dV1[0]
        V1[0] = soft_threshold(Gamma, lambda1 / gamma_v1)

        """
    parfor t=1:T-1
        V2(:,:,t) = squeeze( gv2.*(U(:,:,t)+dV2(:,:,t))+...
        gw.*(V1(:,:,t+1)-W(:,:,t)+dW(:,:,t)))./(gw+gv2);
    end
        end"""

        # parallelise
        for t in range(1, T):
            Gamma = (gamma_v1 * (U[t] + dV1[t]) + gamma_w * (V2[t-1] + W[t-1] - dW[t-1])) / (gamma_v1 + gamma_w)
            GammaOD = Gamma - np.eye(P) * Gamma  # remove diags
            # soft threshold without diag, re-add diag afterwards
            V1[t] = soft_threshold(GammaOD, lambda1 / (gamma_v1 + gamma_w)) + np.eye(P) * Gamma

        # update V2
        for t in range(T - 1):
            V2[t] = (gamma_v2 * (U[t] + dV2[t]) + gamma_w * (V1[t + 1] - W[t] + dW[t])) / (gamma_v2 + gamma_w)

        # UPdate W auxiliary with soft-thresh or group norm thresholding
        # TODO: Smooth all elements?
        if smoother == 'GFGL':
            for t in range(1, T):
                Gamma = V1[t] - V2[t - 1] + dW[t - 1]
                FGammaN = np.linalg.norm(Gamma, ord='fro')
                # perform projection
                W[t - 1] = (Gamma / FGammaN) * np.maximum(FGammaN - (lambda2 / gamma_w), 0)
        else:
            for t in range(1, T):
                Gamma = V1[t] - V2[t - 1] + dW[t - 1]
                for i in range(P):
                    # TODO: smooth only off-diagonal?
                    for j in range(i, P):
                        W[t - 1, i, j] = soft_threshold(Gamma[i, j], lambda2 / gamma_w)
                        W[t - 1, j, i] = W[t - 1, i, j]

        # update dual variables
        dV1_old = dV1
        dV2_old = dV2

        # FIXME: correct indexing?
        dV1 = dV1_old + U - V1
        dV2 = dV2_old + U[:T - 1] - V2
        dW = dW + V1[1:T] - V2 - W

        # check dual and primal feasability
        epsD1 = epsD2 = epsP1 = epsP2 = 0

        # calculate convergence metrics (parallelize)
        for t in range(T - 1):
            # dual and primal feasability
            norm_delta_dV2 = np.linalg.norm(dV2[t] - dV2_old[t], ord='fro')
            epsD2 = epsD2 + norm_delta_dV2 ** 2
            norm_delta_UV2 = np.linalg.norm(U[t] - V2[t], ord='fro')
            epsP2 = epsP2 + norm_delta_UV2 ** 2

        for t in range(T):
            norm_delta_dV1 = np.linalg.norm(dV1[t] - dV1_old[t], ord='fro')
            epsD1 = epsD1 + norm_delta_dV1 ** 2
            norm_delta_UV1 = np.linalg.norm(U[t] - V1[t], ord='fro')
            epsP1 = epsP1 + norm_delta_UV1 ** 2

        eps_dual.append(epsD1 + epsD2)
        eps_primal.append(epsP1 + epsP2)

        if verbose:
            print('iteration', n_iter, 'Prime: ', eps_primal[-1], ' Dual: ', eps_dual[-1])

        n_iter += 1

    return V1, convert_to_sparse(U, W), n_iter, eps_primal, eps_dual



if __name__ == '__main__':
    y = np.load('data/y.npy')
    T, P = y.shape
    verbose = True
    smoother = 'GFGL'
    tol = 1e-4
    max_iter = 500
    gammas = [1, 1, 1]  # gamma_V1, gamma_V2, gamma_W
    lambda1G = 0.15
    lambda2G = 25
    lambda1I = 0.25
    lambda2I = 2
    Theta, sparse_Theta, n_iter, eps_primal, eps_dual = gtADMM(y, lambda1G, lambda2G, gammas[0], gammas[1], gammas[2], max_iter, tol, smoother, verbose)
    print(Theta.shape, sparse_Theta.shape)
    change_point_hist = get_change_points(sparse_Theta, 0.01, T, P)
    change_points = [i for i, cp in enumerate(change_point_hist) if cp > 0]
    plot_data_with_cps(y, change_points, ymin=-5, ymax=5)
