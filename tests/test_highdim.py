"""Checks for the high-dimensional test.  Run with:  python -m unittest discover -s tests"""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor_pilot.bench import KLD, InnerProduct, Net, Quadratic  # noqa: E402
from monitor_pilot.highdim import load_bench, run_bench, summarise  # noqa: E402
from monitor_pilot.protocols import (Events, _box, n_updates, run_filter,  # noqa: E402
                                     run_zone, true_values)


def net(kind, d=6, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [d, 8, 5, 1]
    return Net([(rng.normal(scale=0.6, size=(a, b)), rng.normal(scale=0.2, size=b))
                for a, b in zip(sizes[:-1], sizes[1:])], kind)


def walk(seed, T=400, k=6, d=8, own=0.05):
    rng = np.random.default_rng(seed)
    return 1.0 + np.cumsum(rng.normal(scale=own, size=(T, k, d)), axis=0)


def as_events(X):
    """The same stream, one node per step in round-robin order."""
    T, k, d = X.shape
    return Events(X[0], np.tile(np.arange(k), T - 1), X[1:].reshape(-1, d))


class Functions(unittest.TestCase):
    def cases(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(6, 6))
        return [(InnerProduct(6), rng.normal(size=6)), (Quadratic(A), rng.normal(size=6)),
                (KLD(6), rng.uniform(0.05, 0.4, size=6)),
                (net("tanh"), rng.normal(size=6)), (net("relu"), rng.normal(size=6))]

    def test_derivatives_match_finite_differences(self):
        h = 1e-6
        for f, x in self.cases():
            g = np.array([(f.f(x + h * e)[0] - f.f(x - h * e)[0]) / (2 * h) for e in np.eye(6)])
            H = np.array([(f.grad(x + h * e) - f.grad(x - h * e)) / (2 * h) for e in np.eye(6)])
            self.assertLess(np.abs(g - f.grad(x)).max(), 1e-6, f.name)
            self.assertLess(np.abs(H - f.hess(x)).max(), 1e-5, f.name)

    def test_closed_form_eigenvalues_of_the_relu_network(self):
        f, rng = net("relu"), np.random.default_rng(2)
        for _ in range(20):
            x = rng.normal(size=6)
            lam = np.linalg.eigvalsh(f.hess(x))
            lo, hi = f.eig_range(x)
            self.assertAlmostEqual(lo, min(lam[0], 0.0), places=12)
            self.assertAlmostEqual(hi, max(lam[-1], 0.0), places=12)
        self.assertIsNone(net("tanh").eig_range(rng.normal(size=6)))

    def test_lipschitz_constant_bounds_the_change_in_the_networks(self):
        rng = np.random.default_rng(3)
        for kind in ("tanh", "relu"):
            f = net(kind)
            L = f.lipschitz()
            x, y = rng.normal(size=(200, 6)), rng.normal(size=(200, 6))
            self.assertTrue(np.all(np.abs(f.f(x) - f.f(y))
                                   <= L * np.linalg.norm(x - y, axis=1) + 1e-12))

    def test_neighbourhood_is_clipped_to_the_domain(self):
        lo, hi = _box(KLD(4), np.array([0.05, 0.5, 0.95, 0.2]), 0.1)
        np.testing.assert_allclose(lo, [0.0, 0.4, 0.85, 0.1])
        np.testing.assert_allclose(hi, [0.15, 0.6, 1.0, 0.3])


class EventStreams(unittest.TestCase):
    def test_events_replay_the_dense_stream(self):
        X = walk(0)
        E = as_events(X)
        k = X.shape[1]
        self.assertEqual(n_updates(E), n_updates(X))
        np.testing.assert_allclose(E.means()[::k], X.mean(axis=1), atol=1e-12)
        for t in range(len(E)):
            cur = E[t]
            if t % k == 0:
                np.testing.assert_allclose(cur, X[t // k])
        with self.assertRaises(IndexError):
            E[5]

    def test_guaranteed_filter_holds_on_an_event_stream(self):
        f = net("relu", d=8)
        E = as_events(walk(1))
        F = true_values(E, f)
        eps = 0.5 * F.std()
        run = run_filter(E, f, eps, eps / f.lipschitz(), F)
        self.assertEqual(run.viol_rate, 0.0)
        self.assertLessEqual(run.msg_ratio, 1.0)

    def test_periodic_reporting_on_an_event_stream_counts_node_updates(self):
        f = InnerProduct(8)
        E = as_events(walk(2))
        run = run_filter(E, f, 1.0, 0.0, period=4)
        self.assertAlmostEqual(run.msg_ratio, 0.25, places=2)

    def test_safe_zones_keep_the_error_bound_for_a_constant_hessian(self):
        f = InnerProduct(8)
        for X in (walk(3), as_events(walk(3))):
            F = true_values(X, f)
            tight = run_zone(X, f, 0.3 * F.std(), F_true=F)
            wide = run_zone(X, f, 3.0 * F.std(), F_true=F)
            self.assertEqual(tight.viol_rate, 0.0)
            self.assertEqual(wide.viol_rate, 0.0)
            self.assertLess(wide.msg_ratio, tight.msg_ratio)


class Pipeline(unittest.TestCase):
    def test_load_run_and_summarise_a_small_benchmark(self):
        X = walk(4, T=600, k=5, d=6, own=0.02)
        E = as_events(X)
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(os.path.join(tmp, "inner_product.npz"), X0=E.X0, idx=E.idx, V=E.V,
                     rounds=True, eps=0.05, nodes=5, d=6)
            b = load_bench(tmp, "inner_product")
        self.assertEqual(len(b.tune) + len(b.eval), 600)
        np.testing.assert_allclose(b.eval[0], X[len(b.tune)])
        self.assertTrue(np.isnan(b.automon_ratio))
        rows = pd.DataFrame(run_bench(b, audit_p=(0.02,), log=lambda *_: None))
        self.assertEqual(set(rows.method), {"periodic", "filter-fixed", "zone-reimplemented",
                                            "filter-calibrated"})
        fixed = rows[rows.method == "filter-fixed"].sort_values("param")
        self.assertGreater(fixed.viol_rate.iloc[-1], 0.2)          # grid reaches failure
        self.assertTrue((np.diff(fixed.msg_ratio) <= 1e-12).all())  # looser = fewer messages
        summ = summarise(rows, [b], p=0.02)
        self.assertEqual(len(summ), 3)
        self.assertTrue((summ.filter_zero_viol <= summ.periodic_zero_viol).all())
        self.assertTrue((summ.zone_reimplemented_viol == 0).all())


if __name__ == "__main__":
    unittest.main()

