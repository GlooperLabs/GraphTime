import numpy as np
from unittest import TestCase
from graphtime.utils import plot_data_with_cps, get_change_points, get_edges


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

