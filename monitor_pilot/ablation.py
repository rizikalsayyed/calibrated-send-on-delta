"""Stage 5: energy, tolerance sweep and ablations.

energy        messages -> charge per node -> battery life, with the LoRaWAN
              class-A model of Casals et al. (Sensors 17(10):2364, 2017).
tolerance     every method at several tolerances, so that no conclusion
              rests on one tolerance that happens to suit one method.
ablations     (a) calibrated periodic reporting against calibrated
              send-on-delta, both with the near-miss rule, to separate what
              the reporting trigger contributes from what calibration
              contributes; (b) the audit rate.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .calib import start_tolerance
from .highdim import Bench, _sweep, _typical_step
from .protocols import Calibration, Events, run_filter, run_periodic, run_zone, true_values

RAMP = 0.75                      # near-miss rule recommended in Stage 4
SEEDS = (0, 1, 2)
PERIODS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
RADII = (0.1, 0.4, 1.6)

# ------------------------------------------------------------------- energy
# Casals et al. 2017, Table 5 (unconfirmed uplink, class A), currents in mA,
# durations in ms; transmission at SF7 / 125 kHz, duration from the LoRa
# time-on-air formula; 3.6 V supply and 2400 mAh battery as in the paper.
LORA_STATES = [("wake up", 168.2, 22.1), ("radio preparation", 83.8, 13.3),
               ("wait 1st window", 983.3, 27.0), ("1st receive window", 12.29, 38.1),
               ("2nd receive window", 33.0, 35.0), ("radio off", 147.4, 13.2),
               ("postprocessing", 268.0, 21.0), ("turn off", 38.6, 13.3)]
LORA_TX_MA, LORA_SLEEP_MA, BATTERY_MAH, VOLT = 83.0, 0.045, 2400.0, 3.6
LORAWAN_OVERHEAD = 13            # MHDR 1 + FHDR 7 + FPort 1 + MIC 4 bytes
BYTES_PER_VALUE = 2              # 16-bit fixed point per vector coordinate


def time_on_air_ms(payload: int, sf: int = 7, bw: float = 125e3, cr: int = 1,
                   preamble: int = 8) -> float:
    """LoRa time on air (Semtech AN1200.13), explicit header, CRC on."""
    t_sym = 2 ** sf / bw * 1e3
    de = 1 if t_sym > 16 else 0
    n = 8 + max(math.ceil((8 * payload - 4 * sf + 28 + 16) / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25) * t_sym + n * t_sym


def charge_per_message_mc(d: int, sf: int = 7) -> float:
    """Charge of one class-A uplink carrying a d-dimensional vector, in mC."""
    payload = LORAWAN_OVERHEAD + BYTES_PER_VALUE * d
    rx1 = {n: t for n, t, _ in LORA_STATES}["1st receive window"]
    wait2 = 1000.0 - rx1         # Eq. (4): the 2nd window opens 1 s after the 1st
    q = sum(t * i for _, t, i in LORA_STATES) + wait2 * 27.1
    return (q + time_on_air_ms(payload, sf) * LORA_TX_MA) / 1e3


def battery_years(msg_ratio: float, updates_per_hour: float, d: int) -> float:
    """Battery life of a node whose radio sends msg_ratio * updates messages.

    Every message, up- or downlink, is charged as one uplink transaction; this
    is an upper bound for a downlink, which in class A rides in a receive
    window.  Sensing and the microcontroller's own work are the same for every
    method and are left out, so lives are upper bounds, as in the source paper.
    """
    q_mah = charge_per_message_mc(d) / 3600.0       # mC -> mAh (1 mAh = 3.6 C)
    i_avg = LORA_SLEEP_MA + msg_ratio * updates_per_hour * q_mah
    return BATTERY_MAH / i_avg / 8766.0


# --------------------------------------------------------------- calibrated

def _cal(alpha, p, seed, per_step):
    return Calibration(alpha=alpha, p=p * per_step, seed=seed, ramp=RAMP)


def _periodic_any(X, f, eps, period, F):
    if isinstance(X, Events):
        return run_filter(X, f, eps, 0.0, F, period=period)
    return run_periodic(X, f, eps, period, F)


def calibrated_runs(b: Bench, alphas: Sequence[float], p: float = 0.01,
                    seeds: Sequence[int] = SEEDS, periodic: bool = True,
                    F: Optional[np.ndarray] = None) -> List[dict]:
    """Near-miss calibrated send-on-delta (and periodic) on one benchmark."""
    f, eps, X = b.func, b.eps, b.eval
    F = true_values(X, f) if F is None else F
    per_step = 1.0 / b.nodes if isinstance(X, Events) else 1.0
    Ft = true_values(b.tune, f)
    tune_d = _sweep(b.tune, f, eps, Ft, _typical_step(b.tune))
    tune_p = [(q, run_filter(b.tune, f, eps, 0.0, Ft, period=q)) for q in PERIODS] \
        if periodic else []
    rows = []
    for alpha in alphas:
        probe = _cal(alpha, p, 0, 1.0)
        d0 = start_tolerance(tune_d, probe, eps)
        ok = [q for q, r in tune_p if probe.loss(r.err / eps).mean() <= alpha]
        q0 = max(ok) if ok else 1
        for seed in seeds:
            run = run_filter(X, f, eps, d0, F, cal=_cal(alpha, p, seed, per_step))
            rows.append(dict(method="send-on-delta, calibrated", alpha=alpha, p=p, seed=seed,
                             param=d0, msg_ratio=run.msg_ratio, viol_rate=run.viol_rate,
                             audits=run.audits))
            if periodic:
                run = run_filter(X, f, eps, 0.0, F, period=q0,
                                 cal=_cal(alpha, p, seed, per_step))
                rows.append(dict(method="periodic, calibrated", alpha=alpha, p=p, seed=seed,
                                 param=q0, msg_ratio=run.msg_ratio, viol_rate=run.viol_rate,
                                 audits=run.audits))
    return rows


def best_radius(b: Bench, eps: float) -> Optional[float]:
    if b.func.const_hessian:
        return None
    Ft = true_values(b.tune, b.func)
    return min(RADII, key=lambda r: run_zone(b.tune, b.func, eps, r=r, F_true=Ft).msgs)


def tolerance_sweep(b: Bench, mults: Sequence[float], alphas=(0.01, 0.05), p=0.01,
                    calibrate: bool = True, zone: bool = True, log=print) -> List[dict]:
    """Every method at eps * mult for each mult."""
    rows = []
    eps0 = b.eps
    F = true_values(b.eval, b.func)
    for m in mults:
        bm = Bench(**{**b.__dict__, "eps": eps0 * m})
        X, f, eps = bm.eval, bm.func, bm.eps
        base = dict(case=b.name, d=b.d, tol_mult=m, eps=eps)
        if zone:
            r = best_radius(bm, eps)
            z = run_zone(X, f, eps, r=r, F_true=F)
            rows.append({**base, "method": "safe zones (AutoMon re-implementation)",
                         "msg_ratio": z.msg_ratio, "viol_rate": z.viol_rate,
                         "param": np.nan if r is None else r})
        for q in PERIODS:
            run = _periodic_any(X, f, eps, q, F)
            rows.append({**base, "method": "periodic, fixed", "param": q,
                         "msg_ratio": run.msg_ratio, "viol_rate": run.viol_rate})
        for d, run in _sweep(X, f, eps, F, _typical_step(bm.tune)):
            rows.append({**base, "method": "send-on-delta, fixed", "param": d,
                         "msg_ratio": run.msg_ratio, "viol_rate": run.viol_rate})
        if calibrate:
            for r in calibrated_runs(bm, alphas, p, periodic=False, F=F):
                rows.append({**base, **r})
        log(f"    {b.name} x{m}: done")
    return rows


def ablation(b: Bench, alphas=(0.01, 0.05), audit_p=(0.0025, 0.005, 0.01, 0.02, 0.04),
             log=print) -> List[dict]:
    """Calibrated periodic vs send-on-delta at 1% audits; audit-rate sweep."""
    F = true_values(b.eval, b.func)
    rows = [{**dict(case=b.name, d=b.d, study="trigger"), **r}
            for r in calibrated_runs(b, alphas, 0.01, F=F)]
    log(f"    {b.name}: trigger ablation done")
    for p in audit_p:
        rows += [{**dict(case=b.name, d=b.d, study="audit rate"), **r}
                 for r in calibrated_runs(b, alphas, p, periodic=False, F=F)]
    log(f"    {b.name}: audit-rate sweep done")
    return rows


# ------------------------------------------------------------------ summary

def hindsight(g: pd.DataFrame, method: str, level: float = 0.0) -> float:
    d = g[(g.method == method) & (g.viol_rate <= level + 1e-12)]
    return float(d.msg_ratio.min()) if len(d) else np.nan


def summarise_tolerance(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (case, m), g in rows.groupby(["case", "tol_mult"], sort=False):
        z = g[g.method.str.startswith("safe zones")]
        row = dict(case=case, d=g.d.iloc[0], tol_mult=m,
                   safe_zones=z.msg_ratio.iloc[0] if len(z) else np.nan,
                   safe_zones_viol=z.viol_rate.iloc[0] if len(z) else np.nan,
                   periodic_zero_viol=hindsight(g, "periodic, fixed"),
                   send_on_delta_zero_viol=hindsight(g, "send-on-delta, fixed"))
        for a in (0.01, 0.05):
            c = g[(g.method == "send-on-delta, calibrated") & np.isclose(g.alpha, a)]
            row[f"calibrated_{a:g}"] = c.msg_ratio.mean() if len(c) else np.nan
            row[f"calibrated_{a:g}_viol"] = c.viol_rate.mean() if len(c) else np.nan
            row[f"calibrated_{a:g}_viol_worst"] = c.viol_rate.max() if len(c) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def summarise_ablation(rows: pd.DataFrame) -> pd.DataFrame:
    g = rows.groupby(["case", "study", "method", "alpha", "p"], sort=False)
    out = g.agg(msgs=("msg_ratio", "mean"), viol=("viol_rate", "mean"),
                viol_worst=("viol_rate", "max"), audits=("audits", "mean"),
                seeds=("seed", "size")).reset_index()
    out["within_target"] = out.viol <= out.alpha
    return out
