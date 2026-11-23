"""AutoMon's own benchmarks: load the rebuilt streams and run the filters."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .bench import KLD, InnerProduct, Net, Quadratic
from .protocols import (Calibration, Events, n_updates, run_filter, run_periodic,
                        run_zone, true_values)

TUNE = 0.2                       # leading share of each stream used for tuning only
PERIODS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
ALPHAS = (0.01, 0.02, 0.05)
SEEDS = (0, 1, 2)
LABELS = {"inner_product": "Inner product", "quadratic": "Quadratic form",
          "MLP_40": "MLP-40", "MLP_80": "MLP-80", "air_quality": "KLD air quality",
          "intrusion_detection": "DNN intrusion detection"}
# neighbourhood size in AutoMon's configuration, used when its tuned value is missing
CONFIG_R = {"MLP_40": 0.4, "MLP_80": 0.4, "air_quality": 1.0, "intrusion_detection": 0.5}
LONG = ("inner_product_long", "quadratic_long", "MLP_40_long", "MLP_80_long")
LABELS.update({n: LABELS[n[:-5]] + ", long" for n in LONG})


@dataclass
class Bench:
    name: str
    func: object
    tune: object            # stream (dense array or Events) for tuning
    eval: object            # stream for all reported numbers
    eps: float
    d: int
    nodes: int
    automon_ratio: float    # original AutoMon, messages / node updates on eval
    automon_viol: float
    automon_mean_err: float
    automon_max_err: float
    automon_r: float
    automon_const_hessian: bool


def _function(name: str, z) -> object:
    if name == "inner_product":
        return InnerProduct(int(z["d"]))
    if name == "quadratic":
        return Quadratic(z["H"])
    if name == "air_quality":
        return KLD(int(z["d"]))
    n = sum(1 for key in z.files if key.startswith("W"))
    layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
    return Net(layers, "relu" if name == "intrusion_detection" else "tanh")


def load_bench(folder: str, name: str) -> Bench:
    z = np.load(os.path.join(folder, f"{name}.npz"), allow_pickle=False)
    base = re.sub(r"_(long|first\d+)$", "", name)
    func, eps, k = _function(base, z), float(z["eps"]), int(z["nodes"])
    X0, idx, V = z["X0"], z["idx"], z["V"]
    has_am = "am_err" in z.files
    am_err = z["am_err"] if has_am else np.full(len(idx) + 1, np.nan)
    am_cum = z["am_cum"].astype(float) if has_am else np.full(len(idx) + 1, np.nan)
    if bool(z["rounds"]):                       # every node updates once per round
        R = len(idx) // k
        X = np.empty((R + 1, k, X0.shape[1]))
        X[0] = X0
        cur = X0.copy()
        for r in range(R):
            sl = slice(r * k, (r + 1) * k)
            cur[idx[sl]] = V[sl]
            X[r + 1] = cur
        a = int(TUNE * len(X))
        tune, ev = X[:a], X[a:]
    else:                                       # one node per step
        full = Events(X0, idx, V)
        a = int(TUNE * len(full))
        state = X0.copy()
        for j in range(a - 1):
            state[idx[j]] = V[j]
        tune = Events(X0, idx[:a - 1], V[:a - 1])
        ev = Events(state, idx[a - 1:], V[a - 1:])
    n = min(len(am_err), len(ev) + a)
    err = am_err[a:n]
    msgs = am_cum[n - 1] - am_cum[a - 1]
    ratio = float(msgs / n_updates(ev))
    short = os.path.join(folder, f"{base}.npz")
    if name != base and os.path.exists(short):   # long stream: AutoMon ran on the
        zs = np.load(short)                      # published stream of the same generator
        if "am_err" in zs.files:
            has_am, err, z = True, zs["am_err"], zs
            ratio = float(zs["am_msgs"]) / len(zs["idx"])
    return Bench(name, func, tune, ev, eps, int(z["d"]), k,
                 automon_ratio=ratio,
                 automon_viol=float(np.mean(err > eps * (1 + 1e-9))) if has_am else np.nan,
                 automon_mean_err=float(err.mean() / eps),
                 automon_max_err=float(err.max() / eps),
                 automon_r=float(z["am_r"]) if has_am else np.nan,
                 automon_const_hessian=bool(z["am_const_hessian"]) if has_am else False)


def _typical_step(X) -> float:
    """Median size of one node update, to place the tolerance grid."""
    if isinstance(X, Events):
        prev, out = X.X0.copy(), []
        for j in range(min(len(X.idx), 20000)):
            i = X.idx[j]
            out.append(np.linalg.norm(X.V[j] - prev[i]))
            prev[i] = X.V[j]
        steps = np.array(out)
    else:
        steps = np.linalg.norm(np.diff(X, axis=0), axis=2).ravel()
    steps = steps[steps > 0]
    return float(np.median(steps)) if len(steps) else 1.0


def _sweep(X, f, eps, F, step, stop=0.2, base=27, cap=70):
    """Send-on-delta over a growing tolerance grid.

    Starts with ``base`` values, step * geomspace(0.25, 400), and keeps
    widening by the same factor until more than ``stop`` of the steps violate
    or no node reports any more.
    """
    ratio = (400 / 0.25) ** (1 / (base - 1))
    out, d = [], 0.25 * step
    for i in range(cap):
        run = run_filter(X, f, eps, d, F)
        out.append((d, run))
        if i >= base - 1 and (run.viol_rate > stop or run.msgs <= X.shape[1]):
            break
        d *= ratio
    return out


def run_bench(b: Bench, audit_p=(0.01, 0.02), calibrate: bool = True, zone: bool = True,
              log=print) -> List[dict]:
    """All methods on one benchmark.

    ``audit_p`` is the share of *rounds* that are audited, a round being one
    update per node.  On a stream where one node changes per step the audit
    probability per step is therefore audit_p / nodes, so that audits cost the
    same share of messages (2 * audit_p) on every benchmark.
    """
    f, eps, X = b.func, b.eps, b.eval
    F = true_values(X, f)
    step = _typical_step(b.tune)
    per_step = 1.0 / b.nodes if isinstance(X, Events) else 1.0
    rows: List[dict] = []

    def add(method, run, **kw):
        rows.append(dict(case=b.name, d=b.d, nodes=b.nodes, steps=len(X), method=method,
                         msg_ratio=run.msg_ratio, viol_rate=run.viol_rate,
                         mean_err=run.mean_err, max_err=run.max_err,
                         audits=run.audits, **kw))

    for p in PERIODS:
        run = (run_filter(X, f, eps, 0.0, F, period=p) if isinstance(X, Events)
               else run_periodic(X, f, eps, p, F))
        add("periodic", run, param=p)
    for d, run in _sweep(X, f, eps, F, step):
        add("filter-fixed", run, param=d)
    L = f.lipschitz()
    if L is not None:
        add("filter-guaranteed", run_filter(X, f, eps, eps / L, F), param=eps / L)
    log(f"    {b.name}: fixed settings done")
    if zone:
        base = re.sub(r"_(long|first\d+)$", "", b.name)
        r = None if f.const_hessian else (
            b.automon_r if np.isfinite(b.automon_r) else CONFIG_R[base])
        run = run_zone(X, f, eps, r=r, F_true=F)
        add("zone-reimplemented", run, param=np.nan if r is None else r)
        log(f"    {b.name}: safe zones (re-implementation) {run.msg_ratio:.3f}, "
            f"{run.full_syncs} full syncs, violations {run.viol_rate:.4f}")
    if calibrate:
        Ft = true_values(b.tune, f)
        tune_viol = [(d, run.viol_rate) for d, run in _sweep(b.tune, f, eps, Ft, step)]
        for alpha in ALPHAS:
            ok = [d for d, v in tune_viol if v <= alpha]
            d0 = max(ok) if ok else float(tune_viol[0][0])
            for p in audit_p:
                for seed in SEEDS:
                    cal = Calibration(alpha=alpha, p=p * per_step, seed=seed)
                    add("filter-calibrated", run_filter(X, f, eps, d0, F, cal=cal),
                        alpha=alpha, p=p, seed=seed, param=d0)
        log(f"    {b.name}: calibrated runs done")
    return rows


def summarise(runs: pd.DataFrame, benches: List[Bench], p: float = 0.01) -> pd.DataFrame:
    """One row per benchmark and target violation rate (messages / node updates).

    ``automon`` is the original code (on the published stream of the benchmark;
    for the long synthetic streams that is a different draw of the same
    generator).  ``zone_reimplemented`` is this package's safe-zone protocol on
    the evaluated stream itself.  ``safe_zone`` is the original where it ran and
    the re-implementation otherwise.
    """
    out = []
    runs = runs.reindex(columns=list(dict.fromkeys(list(runs.columns) + ["alpha", "p", "seed"])))
    for b in benches:
        g = runs[runs.case == b.name]
        guar = g[g.method == "filter-guaranteed"]
        zr = g[g.method == "zone-reimplemented"]

        def best(method, level):
            d = g[(g.method == method) & (g.viol_rate <= level + 1e-12)]
            return float(d.msg_ratio.min()) if len(d) else np.nan

        zone = float(zr.msg_ratio.iloc[0]) if len(zr) else np.nan
        original = np.isfinite(b.automon_ratio)
        base = dict(case=b.name, d=b.d, nodes=b.nodes, steps=len(b.eval),
                    safe_zone=b.automon_ratio if original else zone,
                    safe_zone_source="original" if original else "re-implementation",
                    automon=b.automon_ratio, automon_viol=b.automon_viol,
                    automon_max_err=b.automon_max_err,
                    zone_reimplemented=zone,
                    zone_reimplemented_viol=zr.viol_rate.iloc[0] if len(zr) else np.nan,
                    filter_guaranteed=guar.msg_ratio.iloc[0] if len(guar) else np.nan,
                    filter_zero_viol=best("filter-fixed", 0.0),
                    periodic_zero_viol=best("periodic", 0.0))
        for alpha in ALPHAS:
            fc = g[(g.method == "filter-calibrated") & (g.alpha == alpha) & (g.p == p)]
            out.append(dict(base, alpha=alpha,
                            periodic_hindsight=best("periodic", alpha),
                            filter_hindsight=best("filter-fixed", alpha),
                            filter_calibrated=fc.msg_ratio.mean() if len(fc) else np.nan,
                            filter_calibrated_viol=fc.viol_rate.mean() if len(fc) else np.nan,
                            filter_calibrated_viol_max=fc.viol_rate.max() if len(fc) else np.nan,
                            audits=fc.audits.mean() if len(fc) else np.nan))
    return pd.DataFrame(out)
