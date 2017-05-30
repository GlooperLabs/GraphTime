import unittest
import numpy as np

from graphtime.simulate import ErdosRenyiPrecisionGraph, \
    DynamicGraphSimulation
from graphtime.utils import get_edges


class ErdosRenyiTest(unittest.TestCase):

    def test_init(self):
        n_verts = 5
        n_edges = 2
        ER_graph = ErdosRenyiPrecisionGraph(n_verts, n_edges)
        self.assertEqual(ER_graph.n_edges, n_edges)
        self.assertEqual(ER_graph.n_vertices, n_verts)
        self.assertEqual(ER_graph.Theta.shape, (n_verts, n_verts))
        self.assertEqual(ER_graph.Sigma.shape, (n_verts, n_verts))
        print(ER_graph.Sigma)
        print(ER_graph.Theta)

    def test_seed(self):
        n_verts, n_edges = 4, 3
        ER1 = ErdosRenyiPrecisionGraph(n_verts, n_edges, seed=7)
        ER2 = ErdosRenyiPrecisionGraph(n_verts, n_edges, seed=7)
        self.assertTrue((ER1.Sigma == ER2.Sigma).all())
        self.assertTrue((ER1.Theta == ER2.Theta).all())

    def test_active_edges(self):
        n_verts, n_edges = 5, 3
        ER = ErdosRenyiPrecisionGraph(n_verts, n_edges)
        adj_list = get_edges(ER.Theta, eps=1e-7)
        self.assertEqual(len(adj_list), n_edges)

    def test_unit_scale(self):
        n_verts, n_edges = 5, 3
        ER = ErdosRenyiPrecisionGraph(n_verts, n_edges)
        unit = np.ones(5)
        self.assertTrue(np.allclose(ER.Sigma.diagonal(), unit, atol=1e-7))

    def test_psd(self):
        n_verts, n_edges = 5, 3
        ER = ErdosRenyiPrecisionGraph(n_verts, n_edges)
        self.assertTrue(ER.is_PSD)
        # manipulate Theta
        ER.Theta = np.diag([-1, 1])
        self.assertFalse(ER.is_PSD)


class DynamicGraphTest(unittest.TestCase):
    pass

