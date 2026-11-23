"""Checks for the calibration variants.  Run with:  python -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor_pilot.bench import InnerProduct  # noqa: E402
import pandas as pd  # noqa: E402

from monitor_pilot.calib import VARIANTS, make, overall, run_study, summarise  # noqa: E402
from monitor_pilot.highdim import Bench  # noqa: E402
from monitor_pilot.protocols import Calibration, run_filter, true_values  # noqa: E402


def walk(seed, T=3000, k=6, d=8, own=0.05):
    rng = np.random.default_rng(seed)
    return 1.0 + np.cumsum(rng.normal(scale=own, size=(T, k, d)), axis=0)


class Loss(unittest.TestCase):
    def test_near_miss_loss_is_never_below_the_indicator(self):
        e = np.linspace(0, 3, 601)
        hit = (e > 1 + 1e-9).astype(float)
        np.testing.assert_array_equal(Calibration(0.01, 0.01).loss(e), hit)
        for ramp in (0.9, 0.75, 0.5, 0.0):
            loss = Calibration(0.01, 0.01, ramp=ramp).loss(e)
            self.assertTrue(np.all(loss >= hit))
            self.assertTrue(np.all((loss >= 0) & (loss <= 1)))
            self.assertTrue(np.all(np.diff(loss) >= 0))
            self.assertTrue(np.all(loss[e <= ramp] == 0))

    def test_front_loaded_schedule_keeps_the_audit_budget(self):
        flat = Calibration(0.01, 0.01, seed=3)
        front = Calibration(0.01, 0.01, seed=3, boost=4.0)
        T = 400000
        self.assertAlmostEqual(front.rates(T).mean(), 0.01, places=6)
        a = front.schedule(T)
        np.testing.assert_array_equal(a, front.schedule(T))        # fixed in advance
        self.assertAlmostEqual(a.mean(), flat.schedule(T).mean(), delta=0.001)
        self.assertGreater(a[:T // 10].mean(), 3 * a[T // 10:].mean())

    def test_every_variant_builds(self):
        for v in VARIANTS:
            cal = make(v, 0.02, 0.01, 1)
            self.assertEqual(cal.alpha, 0.02)
            self.assertGreaterEqual(cal.p, 0.01)


class Guarantee(unittest.TestCase):
    def test_at_the_floor_every_change_is_reported(self):
        X, f = walk(0, T=400), InnerProduct(8)
        F = true_values(X, f)
        cal = Calibration(alpha=0.05, p=0.0, u_min=0.0)             # starts at the floor
        run = run_filter(X, f, 0.1 * F.std(), 5.0, F, cal=cal)
        self.assertEqual(run.max_err, 0.0)
        self.assertAlmostEqual(run.msg_ratio, 1.0, places=6)

    def test_mean_audited_loss_respects_the_bound_for_all_variants(self):
        f = InnerProduct(8)
        for seed in range(3):
            X = walk(seed)
            F = true_values(X, f)
            eps = 0.05 * F.std()
            for v in VARIANTS:
                cal = make(v, 0.02, 0.03, seed)
                run = run_filter(X, f, eps, 50.0, F, cal=cal)        # starts far too loose
                self.assertGreater(run.audits, 30)
                self.assertLessEqual(run.audit_loss / run.audits,
                                     cal.alpha + cal.slack(run.audits))
                self.assertLessEqual(run.audit_violations, run.audit_loss + 1e-9)

    def test_near_miss_loss_settles_at_a_tighter_tolerance(self):
        f = InnerProduct(8)
        u_ind, u_near = [], []
        for seed in range(4):
            X = walk(seed, T=6000)
            F = true_values(X, f)
            eps = 0.05 * F.std()
            for ramp, store in ((1.0, u_ind), (0.5, u_near)):
                cal = Calibration(alpha=0.05, p=0.05, seed=seed, ramp=ramp)
                run = run_filter(X, f, eps, 1.0, F, cal=cal)
                store.append(np.mean(run.u_trace[len(run.u_trace) // 2:]))
        self.assertLess(np.mean(u_near), np.mean(u_ind))


class Study(unittest.TestCase):
    def test_study_runs_and_summarises(self):
        X, f = walk(5, T=2500), InnerProduct(8)
        eps = 0.05 * true_values(X, f).std()
        b = Bench("toy", f, X[:500], X[500:], eps, 8, 6, np.nan, np.nan, np.nan, np.nan,
                  np.nan, True)
        runs = pd.DataFrame(run_study(b, seeds=range(2), alphas=(0.05,), p=0.05,
                                      variants=("indicator", "near-miss 0.5"),
                                      log=lambda *_: None))
        self.assertEqual(len(runs), 4)
        summ = summarise(runs)
        self.assertTrue(summ.bound_held.all())
        ind, near = (summ[summ.variant == v].iloc[0] for v in ("indicator", "near-miss 0.5"))
        self.assertAlmostEqual(ind.msgs_vs_indicator, 1.0)
        self.assertLessEqual(near.viol, ind.viol)
        over = overall(summ)
        self.assertEqual(list(over.benchmarks), [1, 1])


if __name__ == "__main__":
    unittest.main()
