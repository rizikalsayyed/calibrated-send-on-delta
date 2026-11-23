"""Checks for the air-quality alert case.  Run with:  python -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor_pilot.alerts import ALERT, decision_metrics, episodes  # noqa: E402


class Decisions(unittest.TestCase):
    def test_episodes(self):
        flag = np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)
        self.assertEqual(episodes(flag), [(1, 3), (4, 5), (7, 8)])

    def test_identical_estimates_agree(self):
        ref = np.array([100, 160, 170, 140, 155, 90.0])
        m = decision_metrics(ref, ref)
        self.assertEqual(m["differing_hours"], 0)
        self.assertEqual(m["episodes"], 2)
        self.assertEqual(m["episodes_missed"], 0)

    def test_delay_miss_and_false_alert(self):
        ref = np.array([100, 160, 170, 180, 100, 155, 100, 100.0])
        est = np.array([100, 140, 160, 180, 100, 140, 100, 151.0])
        m = decision_metrics(ref, est)
        self.assertEqual(m["missed_hours"], 2)
        self.assertEqual(m["false_hours"], 1)
        self.assertEqual(m["episodes_missed"], 1)
        self.assertEqual(m["missed_episode_hours"], 1)
        self.assertEqual(m["false_episodes"], 1)
        self.assertEqual(m["mean_onset_delay_h"], 1.0)
        self.assertGreater(ALERT, 140)


if __name__ == "__main__":
    unittest.main()
