import unittest
import numpy as np

from graphtime.simulate import ErdosRenyiPrecisionGraph, \
    DynamicGraph
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

    def test_init(self):
        n_vertices = 3
        labels = ['l1', 'l2', 'l3']
        DGS = DynamicGraph(n_vertices, labels)
        self.assertEqual(DGS.n_vertices, n_vertices)
        self.assertEqual(DGS.labels, labels)
        with self.assertRaises(AssertionError):
            DynamicGraph(5, ['l1'])

    def test_properties(self):
        n_vertices = 3
        DGS = DynamicGraph(n_vertices)
        self.assertEqual(DGS.n_graphs, 0)
        DGS.graphs = [1, 2, 3]
        DGS.changepoints = [2, 3]
        self.assertEqual(DGS.n_graphs, 3)

    def test_graph_indices(self):
        T = 10
        changepoints = [3, 6]
        expected = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        self.assertEqual(len(expected), T)
        calc = list(DynamicGraph._graph_indices(T, changepoints))
        self.assertEqual(calc, expected)

    def test_single_graph_creation(self):
        n_vertices = 4
        DGS = DynamicGraph(n_vertices, seed=7)
        self.assertIsNone(DGS.graphs)
        n_edges = 3
        DGS.create_graphs(n_edges)
        self.assertEqual(DGS.n_graphs, 1)
        self.assertEqual(DGS.graphs[0].n_vertices, 4)
        self.assertEqual(DGS.graphs[0].n_edges, 3)

    def test_multiple_graph_creation(self):
        n_vertices = 4
        DGS = DynamicGraph(n_vertices, seed=7)
        self.assertIsNone(DGS.graphs)
        n_edges_list = [2, 4, 1]
        DGS.create_graphs(n_edges_list)
        for i, n_edges in enumerate(n_edges_list):
            self.assertEqual(DGS.graphs[i].n_edges, n_edges)
            self.assertEqual(DGS.graphs[i].n_vertices, 4)

    def test_creation_seed(self):
        n_verts, n_edges = 4, 3
        n_edges_list = [2, 4, 1]
        DGS1 = DynamicGraph(n_verts, seed=7)
        DGS2 = DynamicGraph(n_verts, seed=7)
        DGS1.create_graphs(n_edges_list)
        DGS2.create_graphs(n_edges_list)
        for i in range(len(n_edges_list)):
            self.assertTrue(np.allclose(DGS1.graphs[i].Theta, DGS2.graphs[i].Theta))

    def test_sampling_wo_graph(self):
        DGS = DynamicGraph(5)
        with self.assertRaises(RuntimeError):
            DGS.sample(10)

    def test_sampling_too_short(self):
        DGS = DynamicGraph(5)
        n_edges_list = [2, 4, 1]
        DGS.create_graphs(n_edges_list)
        with self.assertRaises(ValueError):
            DGS.sample(len(n_edges_list) - 1)

    def test_sampling_invalid_changepoints(self):
        DGS = DynamicGraph(5)
        n_edges_list = [2, 4, 1]
        changepoints = [5]
        DGS.create_graphs(n_edges_list)
        with self.assertRaises(ValueError):
            DGS.sample(10, changepoints)

    def test_sampling_no_changepoints(self):
        DGS = DynamicGraph(5)
        n_edges_list = [2, 4, 1]
        DGS.create_graphs(n_edges_list)
        with self.assertRaises(ValueError):
            DGS.sample(10, uniform=False, changepoints=None)
