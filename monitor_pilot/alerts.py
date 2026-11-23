"""Stage 6: a sustainability case, city-wide air-quality alerts in Beijing.

The coordinator runs the Stage-2 forecasting network on the average of the 12
stations' standardised 24-hour means and issues a "heavy pollution" alert when
the forecast regional 24-hour mean PM2.5, three hours ahead, reaches
150 ug/m3 (the upper limit of China's "moderately polluted" AQI band,
HJ 633-2012; AQI 200).  Reference: the same rule run on every station's every
hourly reading (full reporting).  Each method gives the coordinator its own
estimate of the forecast; the question is how often its alert differs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .ablation import RADII, battery_years
from .calib import start_tolerance
from .cases import DATASETS, TRAIN, TUNE, _trailing_mean, build_case
from .highdim import _sweep, _typical_step
from .protocols import Calibration, run_filter, run_zone, true_values

LEVELS = (35.0, 75.0, 115.0, 150.0, 250.0)    # 24-h PM2.5 AQI breakpoints, ug/m3
ALERT = 150.0
RAMP = 0.75


@dataclass
class AlertCase:
    func: object
    X_tune: np.ndarray
    X_eval: np.ndarray
    mu: float                 # PM2.5 standardisation (ug/m3)
    sd: float
    actual: np.ndarray        # realised regional 24-h mean PM2.5, 3 h ahead (ug/m3)
    times: pd.DatetimeIndex
    horizon: int

    def ug(self, v):
        return np.asarray(v) * self.sd + self.mu


def load(data_dir: str) -> AlertCase:
    c = build_case(data_dir, "beijing", "mlp")
    raw = np.load(os.path.join(data_dir, "beijing.npz"))["X"].astype(float)
    file, w, h = DATASETS["beijing"]
    Z = _trailing_mean(raw, w)
    a, b = int(TRAIN * len(Z)), int((TRAIN + TUNE) * len(Z))
    mu, sd = float(Z[:a, :, 0].mean()), float(Z[:a, :, 0].std())
    regional = Z[:, :, 0].mean(axis=1)
    actual = np.full(len(Z), np.nan)
    actual[:-h] = regional[h:]
    times = pd.date_range("2013-03-01 00:00", periods=len(Z), freq="h")
    return AlertCase(c.func, c.X_tune, c.X_eval, mu, sd, actual[b:], times[b:], h)


def episodes(flag: np.ndarray):
    """(start, end) index pairs of runs of True, end exclusive."""
    d = np.diff(np.concatenate([[0], flag.astype(int), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def decision_metrics(ref: np.ndarray, est: np.ndarray) -> dict:
    """Alert decisions from an estimate compared with full reporting (both in ug/m3)."""
    r, e = ref >= ALERT, est >= ALERT
    lr, le = np.digitize(ref, LEVELS), np.digitize(est, LEVELS)
    eps_ = episodes(r)
    delays, missed, missed_len = [], 0, []
    for s, t in eps_:
        hit = np.nonzero(e[s:t])[0]
        if len(hit):
            delays.append(int(hit[0]))
        else:
            missed += 1
            missed_len.append(int(t - s))
    false_eps = sum(1 for s, t in episodes(e) if not r[s:t].any())
    return dict(alert_hours=int(r.sum()), differing_hours=int((r != e).sum()),
                missed_hours=int((r & ~e).sum()), false_hours=int((~r & e).sum()),
                level_differs=float(np.mean(lr != le)), episodes=len(eps_),
                episodes_missed=missed, missed_episode_hours=sum(missed_len),
                false_episodes=false_eps,
                mean_onset_delay_h=float(np.mean(delays)) if delays else np.nan,
                max_onset_delay_h=int(max(delays)) if delays else 0)


def run_case(case: AlertCase, tolerances=(2.5, 5.0, 10.0), alphas=(0.01, 0.05),
             seeds=(0, 1, 2), p=0.01, log=print) -> List[dict]:
    f, X = case.func, case.X_eval
    F = true_values(X, f)
    Ft = true_values(case.X_tune, f)
    ref = case.ug(F)
    rows = []

    def add(method, run, tol, **kw):
        """Two decision rules on the same estimate: plain, and with a guard band
        (alert once the estimate is within one tolerance of the threshold)."""
        est = case.ug(run.est)
        base = dict(tolerance_ug=tol, msg_ratio=run.msg_ratio, viol_rate=run.viol_rate,
                    battery_years=battery_years(run.msg_ratio, 1.0, X.shape[2]), **kw)
        rows.append(dict(method=method, rule="plain", **base, **decision_metrics(ref, est)))
        rows.append(dict(method=method, rule="guard band", **base,
                         **decision_metrics(ref, est + tol)))

    for tol in tolerances:
        eps = tol / case.sd
        Fz = Ft
        r = min(RADII, key=lambda r_: run_zone(case.X_tune, f, eps, r=r_, F_true=Fz).msgs)
        add("safe zones (AutoMon re-implementation)", run_zone(X, f, eps, r=r, F_true=F), tol,
            param=r)
        tune = _sweep(case.X_tune, f, eps, Ft, _typical_step(case.X_tune))
        for alpha in alphas:
            d0 = start_tolerance(tune, Calibration(alpha, p, ramp=RAMP), eps)
            for seed in seeds:
                cal = Calibration(alpha=alpha, p=p, seed=seed, ramp=RAMP)
                add(f"send-on-delta, calibrated to {alpha:.0%}",
                    run_filter(X, f, eps, d0, F, cal=cal), tol, alpha=alpha, seed=seed, param=d0)
        log(f"    tolerance {tol} ug/m3 done")
    full = dict(method="full reporting", rule="plain", tolerance_ug=0.0, msg_ratio=1.0,
                viol_rate=0.0,
                battery_years=battery_years(1.0, 1.0, X.shape[2]),
                **decision_metrics(ref, ref))
    rows.insert(0, full)
    return rows


def forecast_skill(case: AlertCase) -> dict:
    """How well the full-reporting alert anticipates the realised 24-h mean."""
    ref = case.ug(true_values(case.X_eval, case.func))
    ok = ~np.isnan(case.actual)
    fc, act = ref[ok] >= ALERT, case.actual[ok] >= ALERT
    return dict(hours=int(ok.sum()), actual_alert_hours=int(act.sum()),
                forecast_alert_hours=int(fc.sum()),
                hit_rate=float((fc & act).sum() / max(act.sum(), 1)),
                false_alarm_ratio=float((fc & ~act).sum() / max(fc.sum(), 1)),
                mae_ug=float(np.mean(np.abs(ref[ok] - case.actual[ok]))))


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    cols = ["msg_ratio", "viol_rate", "battery_years", "differing_hours", "missed_hours",
            "false_hours", "level_differs", "episodes", "episodes_missed",
            "missed_episode_hours", "false_episodes", "mean_onset_delay_h",
            "max_onset_delay_h", "alert_hours"]
    g = rows.groupby(["tolerance_ug", "method", "rule"], sort=False)
    out = g[cols].mean().reset_index()
    out["max_onset_delay_h"] = g.max_onset_delay_h.max().values
    out["episodes_missed_worst"] = g.episodes_missed.max().values
    return out


def episode_traces(case: AlertCase, tol: float = 5.0, alpha: float = 0.01, seed: int = 0):
    """Full-reporting forecast, calibrated send-on-delta estimate and per-step
    messages of both methods, for the timeline figure."""
    f, X = case.func, case.X_eval
    F = true_values(X, f)
    Ft = true_values(case.X_tune, f)
    eps = tol / case.sd
    tune = _sweep(case.X_tune, f, eps, Ft, _typical_step(case.X_tune))
    d0 = start_tolerance(tune, Calibration(alpha, 0.01, ramp=RAMP), eps)
    sod = run_filter(X, f, eps, d0, F, cal=Calibration(alpha, 0.01, seed=seed, ramp=RAMP))
    r = min(RADII, key=lambda r_: run_zone(case.X_tune, f, eps, r=r_, F_true=Ft).msgs)
    zone = run_zone(X, f, eps, r=r, F_true=F)
    return pd.DataFrame(dict(time=case.times, full=case.ug(F), sod=case.ug(sod.est),
                             zone=case.ug(zone.est), actual=case.actual))
