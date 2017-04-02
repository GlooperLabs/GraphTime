import numpy as np


class DynamicGraphLasso:

    def __init__(self, lambda1, lambda2, gamma1=1, gamma2=1, gammaw=1, tol=1e-4,
                 max_iter=100, verbose=False):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gammaw = gammaw
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X):
        if type(self) == DynamicGraphLasso:
            subs = ', '.join([c.__name__ for c in
                              DynamicGraphLasso.__subclasses__()])
            raise NotImplementedError('Baseclass! - use ' +  subs)
        T, P = X.shape
        # don't use pre-smoothing
        S = np.zeros((T, P, P))
        for t in range(T):
            x = X[t].reshape(-1, 1)
            S[t] = x.dot(x.T)
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

        # init convergence criteria
        eps_primal = [self.tol + 1]
        eps_dual = [self.tol + 1]

        n_iter = 0
        while (eps_primal[n_iter] > self.tol and eps_dual[n_iter] > self.tol
               and n_iter < self.max_iter):

            # Copy old primal estimate to check convergence (later)
            U_old = U.copy()
            # solve step 1 data-update through eigen decomposition
            for t in range(T):
                sum_gamma = self.gamma1 + self.gamma2
                if t < T - 1:
                    Gamma = (self.gamma1 * (V1[t] - dV1[t]) +
                             self.gamma2 * (V2[t] - dV2[t])) / sum_gamma
                else:
                    Gamma = self.gamma1 * (V1[t] - dV1[t])

                gbar = sum_gamma / 2
                SGbar = S[t] - sum_gamma * Gamma
                Sd, L = np.linalg.eig(SGbar)  # eig values and vectors

                E = np.zeros(P)
                for r in range(P):
                    sr = Sd[r]
                    theta_r = (sr - np.sqrt(sr ** 2 + 8 * gbar)) / (-4 * gbar)
                    E[r] = theta_r

                E = np.diag(E)
                U[t] = L.dot(E).dot(L.T)  # reconstruct Theta

            # update auxiliary precision estimates
            Gamma = U[0] + dV1[0]
            V1[0] = soft_threshold(Gamma, self.lambda1 / self.gamma1)

            # update V1
            for t in range(1, T):
                gamma1w = self.gamma1 + self.gammaw

                Gamma = (self.gamma1 * (U[t] + dV1[t]) +
                         self.gammaw * (V2[t - 1] + W[t - 1] - dW[t - 1])) \
                        / gamma1w
                # remove diags
                GammaOD = Gamma - np.eye(P) * Gamma
                # soft threshold
                V1[t] = soft_threshold(GammaOD, self.lambda1 / gamma1w)
                # re-add diag
                V1[t] += np.eye(P) * Gamma

            # update V2
            for t in range(T-1):
                gamma2w = self.gamma2 + self.gammaw
                V2[t] = (self.gamma2 * (U[t] + dV2[t]) +
                         self.gammaw * (V1[t + 1] - W[t] + dW[t])) \
                        / gamma2w

            # Update W auxiliary
            self.smooth(W, T, P, V1, V2, dW)

            # update dual variables
            dV1_old = dV1.copy()
            dV2_old = dV2.copy()

            # FIXME: correct indexing?
            dV1 = dV1_old + U - V1
            dV2 = dV2_old + U[:T - 1] - V2
            dW = dW + V1[1:T] - V2 - W

            # check dual and primal feasability
            epsD1 = epsD2 = epsP1 = 0

            # There was an error here where the primal and dual convergence
            # criteria are the same
            # calculate convergence metrics (parallelize)
            for t in range(T - 1):
                # dual and primal feasability
                # There should be two dual norms but only one primal.
                norm_delta_dV2 = np.linalg.norm(dV2[t] - dV2_old[t], ord='fro')
                epsD2 = epsD2 + norm_delta_dV2 ** 2

            for t in range(T):
                norm_delta_dV1 = np.linalg.norm(dV1[t] - dV1_old[t], ord='fro')
                epsD1 = epsD1 + norm_delta_dV1 ** 2
                # Calculate convergence (in U, the primal variable)
                norm_delta_U = np.linalg.norm(U[t] - U_old[t], ord='fro')
                epsP1 = epsP1 + norm_delta_U ** 2
              
            eps_dual.append(epsD1 + epsD2)
            eps_primal.append(epsP1)

            if self.verbose:
                print('iteration', n_iter, 'Prime: ', eps_primal[-1], ' Dual: ', eps_dual[-1])

            n_iter += 1

        self.n_iter = n_iter
        self.eps_primal = eps_primal
        self.eps_dual = eps_dual
        self.Theta = V1
        self.sparse_Theta = convert_to_sparse(U, W)
        return self

    def smooth(self, W, T, P, V1, V2, dW):
        raise NotImplementedError


class GroupFusedGraphLasso(DynamicGraphLasso):

    def smooth(self, W, T, P, V1, V2, dW):
        lamgw = self.lambda2 / self.gammaw
        for t in range(1, T):
            Gamma = V1[t] - V2[t - 1] + dW[t - 1]
            FGammaN = np.linalg.norm(Gamma, ord='fro')
            # perform projection
            W[t - 1] = (Gamma / FGammaN) * np.maximum(FGammaN - lamgw, 0)


class IndepFusedGraphLasso(DynamicGraphLasso):

    def smooth(self, W, T, P, V1, V2, dW):
        lamgw = self.lambda2 / self.gammaw
        for t in range(1, T):
            Gamma = V1[t] - V2[t - 1] + dW[t - 1]
            self.soft_thresh(W, t, P, Gamma, lamgw)

    def soft_thresh(self, W, t, P, Gamma, lamgw):
        for i in range(P):
            for j in range(i, P):
                W[t - 1, i, j] = soft_threshold(Gamma[i, j], lamgw)
                W[t - 1, j, i] = W[t - 1, i, j]


def soft_threshold(X, thresh):
    return (np.absolute(X) - thresh).clip(0) * np.sign(X)


def convert_to_sparse(V1, W):
    T, P, _ = V1.shape
    Theta = np.zeros(V1.shape)
    Theta[0] = V1[0]
    for t in range(T-1):
        Theta[t+1] = Theta[t] + W[t]
    return Theta
