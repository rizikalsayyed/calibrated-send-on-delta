"""Checks for Stage 5.  Run with:  python -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor_pilot import ablation as A  # noqa: E402
from monitor_pilot.bench import InnerProduct  # noqa: E402
from monitor_pilot.protocols import Calibration, run_filter, true_values  # noqa: E402


def walk(seed, T=3000, k=6, d=8, own=0.05):
    rng = np.random.default_rng(seed)
    return 1.0 + np.cumsum(rng.normal(scale=own, size=(T, k, d)), axis=0)


class Energy(unittest.TestCase):
    def test_time_on_air_matches_the_semtech_calculator(self):
        # SF7, 125 kHz, CR 4/5, 8-symbol preamble, explicit header, CRC on
        self.assertAlmostEqual(A.time_on_air_ms(10), 41.216, places=2)
        self.assertAlmostEqual(A.time_on_air_ms(51, sf=12), 2465.792, places=1)

    def test_battery_life_falls_with_messages_and_is_capped_by_sleep(self):
        lives = [A.battery_years(m, 60, 6) for m in (0.0, 0.05, 0.3, 1.0)]
        self.assertTrue(all(a > b for a, b in zip(lives, lives[1:])))
        self.assertAlmostEqual(lives[0], A.BATTERY_MAH / A.LORA_SLEEP_MA / 8766, places=6)


class CalibratedPeriodic(unittest.TestCase):
    def test_period_grows_when_safe_and_bound_holds(self):
        f = InnerProduct(8)
        X = walk(1)
        F = true_values(X, f)
        eps = 0.3 * F.std()
        fixed = run_filter(X, f, eps, 0.0, F, period=2)
        cal = Calibration(alpha=0.05, p=0.03, seed=1, ramp=0.75)
        run = run_filter(X, f, eps, 0.0, F, period=2, cal=cal)
        self.assertLessEqual(run.audit_loss / run.audits, cal.alpha + cal.slack(run.audits))
        self.assertNotAlmostEqual(run.msg_ratio, fixed.msg_ratio, places=3)

    def test_floor_means_every_change_is_reported(self):
        f = InnerProduct(8)
        X = walk(2, T=300)
        F = true_values(X, f)
        cal = Calibration(alpha=0.05, p=0.0, u_min=0.0)
        run = run_filter(X, f, 0.1 * F.std(), 0.0, F, period=50, cal=cal)
        self.assertEqual(run.max_err, 0.0)


if __name__ == "__main__":
    unittest.main()
