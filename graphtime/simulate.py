import random
import numpy as np
from graphtime.utils import get_edges


class DynamicGraphicalModel:
    """Representation of a dynamic graph, i.e. a list of
    graphs that represent a process

    Parameters
    ----------
    n_vertices : int
        number of vertices
    labels : list[str]
        list of string denoting the names of vertices
    seed : int
        random seed to use when creating random graphs
        or time series samples from the graphs

    Attributes
    ----------
    graphs : list of ErdosRenyiPrecisionGraph
    """

    def __init__(self, n_vertices, labels=None, seed=None):
        if labels is not None:
            assert len(labels) == n_vertices
        self.n_vertices = n_vertices
        self.labels = labels
        self.seed = seed
        self.graphs = None

    @property
    def n_graphs(self):
        if self.graphs is not None:
            return len(self.graphs)
        return 0

    def create_graphs(self, n_edges_list, use_seed=True):
        """For each number of edges (n_edges) in n_edges_list create
        an Erdos Renyi Precision Graph that allows us to sample
        from later.

        Parameters
        ----------
        n_edges : list[int] or int
            list of number of edges for each graph or scalar
            if only one graph is wanted
        use_seed : bool
            indicates if seed shall be reset
        """
        if use_seed and self.seed is not None:
            random.seed(self.seed)

        n_edges = n_edges_list if type(n_edges_list) is list \
            else [n_edges_list]

        self.graphs = [ErdosRenyiPrecisionGraph(self.n_vertices, n_es)
                       for n_es in n_edges]

    def sample(self, T, changepoints=None, uniform=True, ret_cps=False, use_seed=True):
        """Sample from the created ER Precision graphs. If uniform,
        each graph will approximately generate the same amount of
        samples depending on T. Otherwise a list of changepoints
        must be given indicating where we have to sample from the
        respective graph.

        Parameters
        ----------
        T: int
            number of timesteps T >> #Graphs should hold
        changepoints : list[int]
            list of changepoints where each value is between 0 and T
            and strictly monotonically increasing
        uniform : bool
            indicates if each graph should have approx. same
            amount of samples (last one might have less dep. on T)
        ret_cps : bool
            whether to return list of true changepoints or not
        use_seed : bool
            indicates if seed shall be used

        Returns
        -------
        S: 2D ndarray, shape (timesteps, vertices)
            Sample of length T from the models graphs
        changepoints : optional, list[int]
            list of actual changepoints
        """
        if self.graphs is None:
            raise RuntimeError('First have to create graphs (create_graphs)')
        if T < len(self.graphs):
            raise ValueError('Each graph has to contribute at least one sample')

        if changepoints is not None:
            if len(changepoints) != self.n_graphs - 1:
                raise ValueError('Need one changepoint more than the number of Graphs')
        elif uniform:
            changepoints = self.uniform_changepoints(T, self.n_graphs)
        else:
            raise ValueError('Either Changepoints have to be specified or uniform')

        if use_seed and self.seed is not None:
            np.random.seed(self.seed)

        S = np.zeros((T, self.n_vertices))
        mu = np.zeros(self.n_vertices)
        for i, g in enumerate(self._graph_indices(T, changepoints)):
            S[i] = np.random.multivariate_normal(mu, self.graphs[g].Sigma)

        if ret_cps:
            return S, list(changepoints)
        return S

    @staticmethod
    def uniform_changepoints(T, n_inter):
        """Return optimal uniform changepoints for n intervals of data

        We have to deal with the special cases of divisions with rest
        for the intervals.

        Parameters
        ----------
        T : number of timesteps
        n_inter : number of intervals

        Returns
        -------
        changepoints : list[int]
            changepoint indices
        """
        raw_dist = T / n_inter
        fair_dist = round(raw_dist)
        if fair_dist < raw_dist:
            return range(fair_dist, T-fair_dist, fair_dist)
        else:
            return range(fair_dist, T, fair_dist)

    @staticmethod
    def _graph_indices(T, changepoints):
        """Describes which graphs are active for each time
        by returning a list with the graphs indices

        Parameters
        ----------
        T : int
            number of total timesteps
        changepoints : list[int]
            list of changepoint indices

        Yields
        ------
        Graph indices for all t < T
        """
        graph = count = 0
        for cp in changepoints:
            while count < cp:
                count += 1
                yield graph
            graph += 1
        while count < T:
            count += 1
            yield graph


class ErdosRenyiPrecisionGraph:
    """Creates and Erdos Renyi Adjacency Matrix on initialisation,
    which conforms to the definition of a precision matrix. The
    precision matrix is created based on the defined Erdos
    Renyi graph G(n, M), so drawn uniform at random from all
    graphs with n vertices and M active edges.
    The variance, i.e. the diagonal entries of Theta's inverse,
    are all ones.

    Parameters
    ----------
    n_vertices : int
        number of vertices of the graph
    n_edges : int
        number of edges the graph should have
    seed: int
        seed to use when creating the ER random graph
    eps: float
        threshold below which values are considered 0

    Attributes
    ----------
    Theta : 2D ndarray of shape (n_vertices, n_vertices)
        Precision Matrix of a Covariance considered as Adjacency
        Matrix
    Sigma : 2D ndarray of shape (n_vertices, n_vertices
        Covariance Matrix that corresponds to Theta
        Sigma = Theta^(-1)
    """

    def __init__(self, n_vertices, n_edges, seed=None, eps=1e-10):
        self.n_vertices = n_vertices
        self.n_edges = n_edges
        self.Theta = None
        while not self.is_PSD:
            self.Theta = self.make_precision_graph(seed)
        self.Theta, self.Sigma = self.scale_variance(self.Theta, eps)

    @property
    def is_PSD(self):
        """Convenience property to check if the Precision (and thus also
        the Covariance matrix) is positive semidefinite

        Returns
        -------
        bool indicating if Theta is positive semidefinite"""
        if self.Theta is None:
            return False
        try:
            np.linalg.cholesky(self.Theta)
            return True
        except np.linalg.LinAlgError:
            return False

    @property
    def gexf(self):
        """Associates a GEXF file with the corresponding
        precision matrix. Useful for plotting

        Returns
        -------
        gexf: str
            Gexf formatted graph string
        """
        gexf = '<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">'
        gexf = gexf + '<graph mode="static" defaultedgetype="undirected">'
        gexf = gexf + '<nodes>'

        for nid in range(self.Theta.shape[0]):
            gexf = gexf + '<node id="' + str(nid) + '" />'

        gexf = gexf + '</nodes><edges>'  # Finish adding nodes

        # Find active edges above thresh
        thresh = 0.00001
        eid = 0
        # Uses function from utils
        edges = get_edges(self.Theta, thresh)
        while eid < len(edges):
            gexf = gexf + '<edge id="' + str(edges[eid]) + '">'
            eid += 1

        gexf = gexf + '</edges></graph></gexf>'  # Close off file..
        return gexf

    def make_precision_graph(self, seed):
        """Create an Erdos Renyi Graph G(n_vertices, n_edges) and
        converts it to a valid precision matrix with the same
        sparsity structure

        Parameters
        ----------
        seed: int
            random seed

        Returns
        -------
        G: 2D ndarray shape (n_vertices, n_vertices)
            Precision graph with n_edges active edges
        """
        if seed is not None:
            random.seed(seed)
        G = 0.5 * np.eye(self.n_vertices)
        nodes = list(range(self.n_vertices))
        edges = 0

        while edges < self.n_edges:
            n1 = random.choice(nodes)
            n2 = random.choice(nodes)
            if n1 == n2 or G[n1, n2] != 0:
                continue
            sign = random.choice((1, -1))
            e_val = sign * (random.random() / 2 + 0.5)
            G[n1, n2] = G[n2, n1] = e_val
            G[n1, n1] += abs(e_val)
            G[n2, n2] += abs(e_val)
            edges += 1

        return G

    @staticmethod
    def scale_variance(Theta, eps):
        """Allows to scale a Precision Matrix such that its
        corresponding covariance has unit variance

        Parameters
        ----------
        Theta: ndarray
            Precision Matrix
        eps: float
            values to threshold to zero

        Returns
        -------
        Theta: ndarray
            Precision of rescaled Sigma
        Sigma: ndarray
            Sigma with ones on diagonal
        """
        Sigma = np.linalg.inv(Theta)
        V = np.diag(np.sqrt(np.diag(Sigma) ** -1))
        Sigma = V.dot(Sigma).dot(V.T)  # = VSV
        Theta = np.linalg.inv(Sigma)
        Theta[np.abs(Theta) <= eps] = 0.
        return Theta, Sigma
