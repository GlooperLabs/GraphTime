import unittest
import numpy as np

from graphtime.simulate import ErdosRenyiPrecisionGraph, \
    DynamicGraphicalModel, PrecisionGraph
from graphtime.utils import get_edges


class GraphTest(unittest.TestCase):
    G = np.array([[1, 1, 0],
                  [1, 1, 0],
                  [0, 0, 1]])

    def test_init(self):
        Graph = PrecisionGraph(self.G, self.G)
        self.assertEqual(Graph.n_edges, 1)
        self.assertEqual(Graph.n_vertices, 3)

    def test_nxconvert(self):
        Graph = PrecisionGraph(self.G, self.G)
        nxgraph = Graph.nxGraph
        self.assertEqual(nxgraph.edges(), [(1, 2)])


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
        Theta = np.diag([-1, 1])
        self.assertFalse(ErdosRenyiPrecisionGraph.is_PSD(Theta))


class DynamicGraphTest(unittest.TestCase):

    def test_init(self):
        n_vertices = 3
        labels = ['l1', 'l2', 'l3']
        DGS = DynamicGraphicalModel(n_vertices, labels)
        self.assertEqual(DGS.n_vertices, n_vertices)

    def test_properties(self):
        n_vertices = 3
        DGS = DynamicGraphicalModel(n_vertices)
        self.assertEqual(DGS.n_graphs, 0)
        DGS.graphs = [1, 2, 3]
        DGS.changepoints = [2, 3]
        self.assertEqual(DGS.n_graphs, 3)

    def test_graph_indices(self):
        T = 10
        changepoints = [3, 6]
        expected = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        self.assertEqual(len(expected), T)
        calc = list(DynamicGraphicalModel._graph_indices(T, changepoints))
        self.assertEqual(calc, expected)

    def test_single_graph_creation(self):
        n_vertices = 4
        DGS = DynamicGraphicalModel(n_vertices, seed=7)
        self.assertIsNone(DGS.graphs)
        n_edges = 3
        DGS.generate_graphs(n_edges)
        self.assertEqual(DGS.n_graphs, 1)
        self.assertEqual(DGS.graphs[0].n_vertices, 4)
        self.assertEqual(DGS.graphs[0].n_edges, 3)

    def test_multiple_graph_creation(self):
        n_vertices = 4
        DGS = DynamicGraphicalModel(n_vertices, seed=7)
        self.assertIsNone(DGS.graphs)
        n_edges_list = [2, 4, 1]
        DGS.generate_graphs(n_edges_list)
        for i, n_edges in enumerate(n_edges_list):
            self.assertEqual(DGS.graphs[i].n_edges, n_edges)
            self.assertEqual(DGS.graphs[i].n_vertices, 4)

    def test_creation_seed(self):
        n_verts, n_edges = 4, 3
        n_edges_list = [2, 4, 1]
        DGS1 = DynamicGraphicalModel(n_verts, seed=7)
        DGS2 = DynamicGraphicalModel(n_verts, seed=7)
        DGS1.generate_graphs(n_edges_list)
        DGS2.generate_graphs(n_edges_list)
        for i in range(len(n_edges_list)):
            self.assertTrue(np.allclose(DGS1.graphs[i].Theta, DGS2.graphs[i].Theta))

    def test_sampling_wo_graph(self):
        DGS = DynamicGraphicalModel(5)
        with self.assertRaises(RuntimeError):
            DGS.sample(10)

    def test_sampling_too_short(self):
        DGS = DynamicGraphicalModel(5)
        n_edges_list = [2, 4, 1]
        DGS.generate_graphs(n_edges_list)
        with self.assertRaises(ValueError):
            DGS.sample(len(n_edges_list) - 1)

    def test_sampling_invalid_changepoints(self):
        DGS = DynamicGraphicalModel(5)
        n_edges_list = [2, 4, 1]
        changepoints = [5]
        DGS.generate_graphs(n_edges_list)
        with self.assertRaises(ValueError):
            DGS.sample(10, changepoints)

    def test_sampling_no_changepoints(self):
        DGS = DynamicGraphicalModel(5)
        n_edges_list = [2, 4, 1]
        DGS.generate_graphs(n_edges_list)
        with self.assertRaises(ValueError):
            DGS.sample(10, uniform=False)

    def test_uniform_changepoints(self):
        DGS = DynamicGraphicalModel(5)
        changepoints = DGS.uniform_changepoints(7, 3)
        exp_changepoints = [2, 4]
        self.assertEqual(list(changepoints), exp_changepoints)

    def test_sample_format(self):
        n_verts, n_edges, T = 4, 3, 15
        n_edges_list = [2, 4, 1]
        DGS = DynamicGraphicalModel(n_verts, seed=7)
        DGS.generate_graphs(n_edges_list)
        S = DGS.sample(T)
        self.assertEqual(S.shape, (T, n_verts))

    def test_sampling_with_seed(self):
        n_verts, n_edges = 4, 3
        n_edges_list = [2, 4, 1]
        DGS = DynamicGraphicalModel(n_verts, seed=7)
        DGS.generate_graphs(n_edges_list)
        sample1 = DGS.sample(10, use_seed=True)
        sample2 = DGS.sample(10, use_seed=True)
        self.assertTrue(np.allclose(sample1, sample2))

    def test_sampling_wo_seed(self):
        n_verts, n_edges = 4, 3
        n_edges_list = [2, 4, 1]
        DGS = DynamicGraphicalModel(n_verts, seed=7)
        DGS.generate_graphs(n_edges_list)
        sample1 = DGS.sample(10, use_seed=False)
        sample2 = DGS.sample(10, use_seed=False)
        self.assertTrue(~np.allclose(sample1, sample2))

    def test_changepoint_return(self):
        n_verts, n_edges, T = 4, 3, 15
        n_edges_list = [2, 4, 1]
        DGS = DynamicGraphicalModel(n_verts, seed=7)
        DGS.generate_graphs(n_edges_list)
        S, cps = DGS.sample(T, ret_cps=True)
        self.assertEqual(len(cps), len(n_edges_list)-1)

    def test_from_Thetas(self):
        Theta1 = np.array([[1,  0, .3],
                           [0,  .5, 0],
                           [.3, 0,  1]])
        Theta2 = np.array([[1,  .3, 0],
                           [.3, .5, 0],
                           [0,  0,  1]])
        Thetas = np.zeros((10, 3, 3))
        for i in range(5):
            Thetas[i] = Theta1
        for i in range(5, 10):
            Thetas[i] = Theta2
        DGM = DynamicGraphicalModel.from_Thetas(Thetas)
        self.assertEqual(len(DGM.graphs), 2)
        G1, G2 = DGM.graphs
        self.assertEqual(G1.n_edges, 1)
        self.assertEqual(G2.n_edges, 1)
        self.assertTrue(np.allclose(G1.Theta, Theta1))
        self.assertTrue(np.allclose(G2.Theta, Theta2))

    def test_from_Thetassertion(self):
        Theta1 = np.array([[1, 0, 1],
                           [0, 1, 0],
                           [1, 0, 1]])
        with self.assertRaises(ValueError):
            DynamicGraphicalModel.from_Thetas(Theta1)
        with self.assertRaises(ValueError):
            DynamicGraphicalModel.from_Thetas(Theta1[:, np.newaxis, :])

    def test_drawing(self):
        # shallow integration test to ensure drawing still works
        DGM = DynamicGraphicalModel(5)
        DGM.generate_graphs([2, 3])
        fig = DGM.draw('circular')
        self.assertEqual(len(fig.axes), 2)
