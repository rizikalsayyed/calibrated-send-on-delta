"""Sanity checks.  Run with:  python -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor_pilot.experiment import lipschitz  # noqa: E402
from monitor_pilot.functions import TanhMLP, Variance  # noqa: E402
from monitor_pilot.protocols import (Calibration, run_central, run_filter,  # noqa: E402
                                     run_periodic, run_zone, true_values)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def streams(seed, T=1500, k=8, d=1, common=0.02, own=0.03):
    """Random-walk readings with a shared and a per-node component."""
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.normal(scale=common, size=(T, 1, d)), axis=0)
            + np.cumsum(rng.normal(scale=own, size=(T, k, d)), axis=0))


def variance_case(seed):
    Z = streams(seed)
    f = Variance()
    X = f.local(Z)
    f.scale = X.reshape(-1, 2).std(axis=0)
    F = true_values(X, f)
    return X, f, F, 0.2 * F.std()


def mlp_case(seed):
    Z = streams(seed, d=3, common=0.01, own=0.01)
    f = TanhMLP(3, seed=seed + 1)
    F = true_values(Z, f)
    return Z, f, F, 0.1 * F.std()


class Functions(unittest.TestCase):
    def test_mlp_derivatives_match_finite_differences(self):
        f, rng, h = TanhMLP(4, seed=3), np.random.default_rng(0), 1e-5
        x = rng.normal(size=4)
        g = [(f.f(x + h * e)[0] - f.f(x - h * e)[0]) / (2 * h) for e in np.eye(4)]
        H = [(f.grad(x + h * e) - f.grad(x - h * e)) / (2 * h) for e in np.eye(4)]
        self.assertLess(np.abs(np.array(g) - f.grad(x)).max(), 1e-8)
        self.assertLess(np.abs(np.array(H) - f.hess(x)).max(), 1e-7)

    def test_variance_equals_the_spread_across_nodes(self):
        Z = streams(1, T=50)
        X = Variance.local(Z)
        np.testing.assert_allclose(Variance().f(X.mean(axis=1)), Z[..., 0].var(axis=1),
                                   atol=1e-12)

    def test_lipschitz_constant_bounds_the_network(self):
        f, rng = TanhMLP(5, seed=2), np.random.default_rng(1)
        x, y = rng.normal(size=(300, 5)), rng.normal(size=(300, 5))
        self.assertTrue(np.all(np.abs(f.f(x) - f.f(y))
                               <= lipschitz(f) * np.linalg.norm(x - y, axis=1) + 1e-12))

    def test_fit_reduces_the_error(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(2000, 3))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1]
        f = TanhMLP(3, seed=0)
        before = np.mean((f.f(X) - y) ** 2)
        f.fit(X, y, epochs=60)
        self.assertLess(np.mean((f.f(X) - y) ** 2), 0.2 * before)


class Protocols(unittest.TestCase):
    def test_central_reporting_is_exact_and_costs_one(self):
        X, f, F, eps = variance_case(0)
        run = run_central(X, f, eps)
        self.assertEqual(run.viol_rate, 0.0)
        self.assertAlmostEqual(run.msg_ratio, 1.0)

    def test_periodic_cost(self):
        X, f, F, eps = variance_case(0)
        self.assertAlmostEqual(run_periodic(X, f, eps, 4, F).msg_ratio, 0.25, places=2)

    def test_looser_filter_sends_fewer_messages(self):
        X, f, F, eps = variance_case(1)
        m = [run_filter(X, f, eps, d, F).msg_ratio for d in (0.01, 0.1, 1.0)]
        self.assertTrue(m[0] >= m[1] >= m[2])

    def test_lipschitz_filter_never_violates(self):
        for seed in range(3):
            X, f, F, eps = mlp_case(seed)
            run = run_filter(X, f, eps, eps / lipschitz(f), F)
            self.assertEqual(run.viol_rate, 0.0)

    def test_safe_zones_never_violate_for_the_variance(self):
        for seed in range(3):
            X, f, F, eps = variance_case(seed)
            self.assertEqual(run_zone(X, f, eps, F_true=F).viol_rate, 0.0)

    def test_audit_schedule_ignores_the_data(self):
        a = Calibration(alpha=0.05, p=0.02, seed=7)
        np.testing.assert_array_equal(a.schedule(500), a.schedule(500))
        self.assertAlmostEqual(a.schedule(200000).mean(), 0.02, places=2)

    def test_audited_violation_share_respects_the_bound(self):
        for seed in range(3):
            X, f, F, eps = variance_case(seed)
            for run in (run_filter, run_zone):
                cal = Calibration(alpha=0.05, p=0.05, seed=seed)
                kw = dict(delta=2.0) if run is run_filter else {}   # start far too loose
                r = run(X, f, eps, F_true=F, cal=cal, **kw)
                slack = (0.0 - cal.u_min + cal.eta) / (cal.eta * r.audits)
                self.assertLessEqual(r.audit_violations / r.audits, cal.alpha + slack)

    def test_calibration_tightens_a_tolerance_that_is_too_loose(self):
        X, f, F, eps = variance_case(2)
        loose = run_filter(X, f, eps, 2.0, F)
        cal = run_filter(X, f, eps, 2.0, F, cal=Calibration(alpha=0.05, p=0.05))
        self.assertLess(cal.viol_rate, loose.viol_rate)
        self.assertLess(cal.u_trace[-1], 0.0)


@unittest.skipUnless(os.path.exists(os.path.join(DATA, "intel.npz")), "data/ not built")
class Cases(unittest.TestCase):
    def test_cases_build_with_the_documented_splits(self):
        from monitor_pilot.cases import build_case
        c = build_case(DATA, "intel", "variance")
        T = len(c.X_tune) / 0.1
        self.assertAlmostEqual(len(c.X_eval) / T, 0.6, places=2)
        self.assertGreater(c.eps, 0)


if __name__ == "__main__":
    unittest.main()
