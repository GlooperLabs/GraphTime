import numpy as np
from unittest import TestCase
from graphtime.utils import plot_data_with_cps, get_change_points, get_edges, soft_threshold, \
    scale_standard


class UtilsTest(TestCase):

    def test_plot_cps(self):
        data = np.random.random((10, 10))
        cps = [5]
        fig = plot_data_with_cps(data, cps)
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]
        # dimension + changepoint
        self.assertEqual(len(ax.lines), 11)

    def test_get_edges(self):
        # Note: Undirected graphs!
        G = np.zeros((4, 4))
        G[1, 0] = G[0, 1] = 1
        G[2, 1] = G[1, 2] = 1
        edges = get_edges(G, eps=1e-7)
        expected = [(0, 1), (1, 2)]
        self.assertEqual(edges, expected)

    def test_changepoints(self):
        Theta1 = np.array([[1, 0, .3],
                           [0, .5, 0],
                           [.3, 0, 1]])
        Theta2 = np.array([[1, .3, 0],
                           [.3, .5, 0],
                           [0, 0, 1]])
        Thetas = np.zeros((10, 3, 3))
        for i in range(5):
            Thetas[i] = Theta1
        for i in range(5, 10):
            Thetas[i] = Theta2
        cps = get_change_points(Thetas, 1e-7)
        self.assertEqual(cps, [5])

    def test_soft_threshold(self):
        x = -0.5
        lam = 0.01
        exp = - max(0.5 - lam, 0)
        self.assertEqual(soft_threshold(x, lam), exp)
        X = np.array([0.3, -0.2, -0.25])
        lam = 0.2
        exp = np.array([0.1, 0, -0.05])
        self.assertTrue(np.allclose(soft_threshold(X, lam), exp))

    def test_scaling(self):
        X = np.array([0.5, 0.2, -0.9, 0.2, 3])
        X_scale = scale_standard(X)
        self.assertTrue(np.allclose(X_scale.mean(), 0))
        self.assertTrue(np.allclose(X_scale.std(), 1))
