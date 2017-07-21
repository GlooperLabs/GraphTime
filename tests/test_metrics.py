from unittest import TestCase
import numpy as np
from graphtime.metrics import precision, global_precision, recall, global_recall, \
    f_score, global_f_score, changepoint_density, n_estimated_edges


nx = np.newaxis


class SupervisedMetrics(TestCase):

    Theta_true1 = np.array([
        [1, 0.2, 0],
        [0.2, 1, 0],
        [0,   0, 1]])
    Theta_true2 = np.array([
        [1,   0,  0.2],
        [0,   1,  0.1],
        [0.2, 0.1,  1]])

    Theta_pred1 = np.array([
        [1, 0.2, 0],
        [0.2, 1, 0],
        [0,   0, 1]])
    Theta_pred2 = np.array([
        [1,   0.2, 0.2],
        [0.2,   1, 0.1],
        [0.2, 0.1,   1]])
    Theta_pred3 = np.array([
        [1,   0, 0.2],
        [0,   1,   0],
        [0.2, 0,  1]])

    @property
    def standard_case(self):
        """Create standard testcase from Thetas defined in this Testcase. The following
        metrics can be calculated by hand and should match the computations:

        precisions: [1, 1, 0, 2/3, 1]
        recalls: [1, 1, 0, 1, 0.5]
        f1s: [1, 1, 0, 0.8, 2/3]
        tps: 1 + 1 + 0 + 2 + 1 = 5
        fps: 0 + 0 + 1 + 1 + 0 = 2
        fns: 0 + 0 + 2 + 0 + 1 = 3
        tns: 2 + 2 + 0 + 0 + 1 = 5
        """
        Theta_true = np.vstack([
            np.repeat(self.Theta_true1[nx, :, :], 2, axis=0),
            np.repeat(self.Theta_true2[nx, :, :], 3, axis=0)
        ])
        Theta_pred = np.vstack([
            np.repeat(self.Theta_pred1[nx, :, :], 3, axis=0),
            self.Theta_pred2[nx, :, :],
            self.Theta_pred3[nx, :, :]
        ])
        return Theta_true, Theta_pred


    def test_ts_precision(self):
        prec = precision(*self.standard_case, per_ts=True)
        exp = np.array([1., 1., 0., 2./3., 1.])
        self.assertTrue(np.allclose(exp, prec))

    def test_avg_precision(self):
        prec = precision(*self.standard_case, per_ts=False)
        exp = np.array([1., 1., 0., 2. / 3., 1.])
        self.assertEqual(exp.mean(), prec)

    def test_global_precision(self):
        prec = global_precision(*self.standard_case)
        exp = 5 / (5 + 2)
        self.assertEqual(exp, prec)

    def test_ts_recall(self):
        rec = recall(*self.standard_case, per_ts=True)
        exp = np.array([1, 1, 0, 1, 0.5])
        self.assertTrue(np.allclose(exp, rec))

    def test_avg_recall(self):
        rec = recall(*self.standard_case, per_ts=False)
        exp = np.array([1, 1, 0, 1, 0.5])
        self.assertTrue(exp.mean(), rec)

    def test_global_recall(self):
        rec = global_recall(*self.standard_case)
        exp = 5 / (5 + 3)
        self.assertEqual(exp, rec)

    def test_ts_fscore(self):
        fscore = f_score(*self.standard_case, per_ts=True)
        exp = np.array([1, 1, 0, 0.8, 2./3.])
        self.assertTrue(np.allclose(exp, fscore))

    def test_ts_f2score(self):
        f2score = f_score(*self.standard_case, per_ts=True, beta=2)
        exp = np.array([1, 1, 0, 0.909090, 0.55555])
        print(exp, f2score)
        self.assertTrue(np.allclose(exp, f2score))

    def test_avg_fscore(self):
        fscore = f_score(*self.standard_case, per_ts=False)
        exp = np.array([1, 1, 0, 0.8, 2. / 3.])
        self.assertTrue(exp.mean(), fscore)

    def test_global_fscore(self):
        fscore = global_f_score(*self.standard_case)
        exp = (2 * 5) / (2 * 5 + 3 + 2)
        self.assertEqual(fscore, exp)

    def test_global_f2score(self):
        f2score = global_f_score(*self.standard_case, beta=2)
        exp = (5 * 5) / (5 * 5 + 4 * 3 + 2)
        self.assertEqual(exp, f2score)


class UnsupervisedMetrics(TestCase):

    Theta_pred1 = np.array([
        [1, 0.2, 0],
        [0.2, 1, 0],
        [0, 0, 1]])
    Theta_pred2 = np.array([
        [1, 0.2, 0.2],
        [0.2, 1, 0.1],
        [0.2, 0.1, 1]])

    @property
    def standard_case(self):
        Theta_pred = np.vstack([
            np.repeat(self.Theta_pred1[nx, :, :], 3, axis=0),
            np.repeat(self.Theta_pred2[nx, :, :], 2, axis=0)
        ])
        return Theta_pred

    def test_ts_edge_count(self):
        n_est = n_estimated_edges(self.standard_case, per_ts=True)
        exp = [1, 1, 1, 3, 3]
        self.assertEqual(exp, n_est)

    def test_edge_count(self):
        n_est = n_estimated_edges(self.standard_case, per_ts=False)
        exp = 3 * 1 + 2 * 3
        self.assertEqual(exp, n_est)

    def test_ts_changepoint_count(self):
        cps = changepoint_density(self.standard_case, per_ts=True)
        exp = [0, 0, 0, 2, 0]
        self.assertEqual(exp, cps)

    def test_changepoint_count(self):
        cps = changepoint_density(self.standard_case, per_ts=False)
        exp = 2
        self.assertEqual(exp, cps)
