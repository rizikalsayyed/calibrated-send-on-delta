"""Stage 7 (revision): experiments requested in review.

zones        safe zones (AutoMon re-implementation) on every case, on the same
             stream as calibrated send-on-delta, with messages split by
             direction; and safe zones with the same audit calibration
             (near-miss rule).
sod          calibrated send-on-delta with messages split by direction (the
             same runs as the calibration study, re-run with counters), the
             offline-tuned tolerance without audits, and sensitivity to eta
             and u_max.
loss         lossy uplinks: independent or bursty losses, unconfirmed or
             confirmed reports.
scale        network size: the Intel deployment resampled to 10-1000 nodes.
retrain      the deployment networks trained on the first 30% only, so that
             the tuning period is out of sample.
"""
from __future__ import annotations

import os
from typing import Iterator, List

import numpy as np
import pandas as pd

from .calib import pilot_bench, start_tolerance
from .cases import DATASETS, TRAIN, TUNE, _trailing_mean
from .functions import TanhMLP, Variance
from .highdim import CONFIG_R, Bench, _sweep, _typical_step, load_bench
from .protocols import Calibration, Events, Run, run_filter, run_zone, true_values

RAMP = 0.75
DEPLOY = ("beijing/variance", "beijing/mlp", "intel/variance", "intel/mlp")
BENCH = ("inner_product_long", "quadratic_long", "MLP_40_long", "MLP_80_long",
         "air_quality", "intrusion_detection")
CASES = DEPLOY + BENCH
# neighbourhood sizes chosen for the original AutoMon runs on the deployments
DEPLOY_R = {"beijing/mlp": 0.4, "intel/mlp": 0.1}


def load_case(name: str, data_dir: str, bench_dir: str) -> Bench:
    return pilot_bench(data_dir, name) if "/" in name else load_bench(bench_dir, name)


def radius(b: Bench):
    if b.func.const_hessian:
        return None
    if "/" in b.name:
        return DEPLOY_R[b.name]
    if np.isfinite(b.automon_r):
        return float(b.automon_r)
    base = b.name.replace("_long", "")
    return CONFIG_R[base]


def per_step(b: Bench) -> float:
    return 1.0 / b.nodes if isinstance(b.eval, Events) else 1.0


def counters(run: Run) -> dict:
    """Message counts per node update, overall and for the busiest node."""
    k = run.nodes
    upd = run.updates or run.rounds * k
    per_node_updates = upd / k
    load_all = (run.node_up + run.node_down) / per_node_updates
    load_a = (run.node_up + run.node_unsol) / per_node_updates
    return dict(msg_ratio=run.msg_ratio, viol_rate=run.viol_rate, max_err=run.max_err,
                up_ratio=run.up / upd, down_ratio=run.down / upd,
                unsolicited_ratio=run.unsolicited / upd,
                busiest_all=float(load_all.max()), busiest_classA=float(load_a.max()),
                mean_classA=float(load_a.mean()), audits=run.audits,
                audit_loss_share=run.audit_loss / max(run.audits, 1), lost=run.lost)


def _done(path: str, keys: List[str]) -> set:
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(map(tuple, d[keys].astype(str).values))


def _append(path: str, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# -------------------------------------------------------------- safe zones

def zones(data_dir, bench_dir, out, cases=CASES, seeds=(0, 1, 2), alpha=0.01,
          p=0.01, calibrated=True, log=print):
    path = os.path.join(out, "zones.csv")
    keys = ["case", "method", "seed"]
    done = _done(path, keys)
    for name in cases:
        b = load_case(name, data_dir, bench_dir)
        F = true_values(b.eval, b.func)
        r = radius(b)
        if (name, "zone", "-1") not in done:
            run = run_zone(b.eval, b.func, b.eps, r=r, F_true=F)
            _append(path, dict(case=name, method="zone", seed=-1, alpha=np.nan, r=r,
                               **counters(run)))
            log(f"  {name}: zone {run.msg_ratio:.4f} viol {run.viol_rate:.4f}")
        if not calibrated:
            continue
        for seed in seeds:
            if (name, "zone-calibrated", str(seed)) in done:
                continue
            cal = Calibration(alpha, p * per_step(b), seed=seed, ramp=RAMP)
            run = run_zone(b.eval, b.func, b.eps, r=r, F_true=F, cal=cal)
            _append(path, dict(case=name, method="zone-calibrated", seed=seed, alpha=alpha,
                               r=r, **counters(run)))
            log(f"  {name}: calibrated zone seed {seed} {run.msg_ratio:.4f} "
                f"viol {run.viol_rate:.4f}")


# ---------------------------------------------------------- send-on-delta

def _start(b: Bench, alpha: float, p: float):
    Ft = true_values(b.tune, b.func)
    tune = _sweep(b.tune, b.func, b.eps, Ft, _typical_step(b.tune))
    d0 = start_tolerance(tune, Calibration(alpha, p, ramp=RAMP), b.eps)
    ok = [d for d, run in tune if run.viol_rate <= alpha]
    d_off = max(ok) if ok else float(tune[0][0])
    return d0, d_off


def sod(data_dir, bench_dir, out, cases=CASES, alphas=(0.01, 0.05), p=0.01, log=print):
    rows = []
    for name in cases:
        b = load_case(name, data_dir, bench_dir)
        F = true_values(b.eval, b.func)
        nseeds = 5 if name == "intrusion_detection" else 10
        for alpha in alphas:
            d0, d_off = _start(b, alpha, p)
            run = run_filter(b.eval, b.func, b.eps, d_off, F)
            rows.append(dict(case=name, method="sod-offline", alpha=alpha, seed=-1,
                             eta=np.nan, u_max=np.nan, delta=d_off, **counters(run)))
            for seed in range(nseeds):
                cal = Calibration(alpha, p * per_step(b), seed=seed, ramp=RAMP)
                run = run_filter(b.eval, b.func, b.eps, d0, F, cal=cal)
                rows.append(dict(case=name, method="sod-calibrated", alpha=alpha, seed=seed,
                                 eta=0.5, u_max=np.log(100), delta=d0, **counters(run)))
            if alpha == 0.01:
                for eta in (0.25, 1.0):
                    for seed in range(3):
                        cal = Calibration(alpha, p * per_step(b), seed=seed, ramp=RAMP, eta=eta)
                        run = run_filter(b.eval, b.func, b.eps, d0, F, cal=cal)
                        rows.append(dict(case=name, method="sod-eta", alpha=alpha, seed=seed,
                                         eta=eta, u_max=np.log(100), delta=d0, **counters(run)))
                for um in (np.log(10), np.log(1000)):
                    for seed in range(3):
                        cal = Calibration(alpha, p * per_step(b), seed=seed, ramp=RAMP, u_max=um)
                        run = run_filter(b.eval, b.func, b.eps, d0, F, cal=cal)
                        rows.append(dict(case=name, method="sod-umax", alpha=alpha, seed=seed,
                                         eta=0.5, u_max=um, delta=d0, **counters(run)))
        log(f"  {name}: send-on-delta runs done")
        pd.DataFrame(rows).to_csv(os.path.join(out, "sod.csv"), index=False)
    return rows


def loss(data_dir, bench_dir, out, cases=CASES, rates=(0.0, 0.01, 0.05, 0.1, 0.2),
         models=("bernoulli", "ge"), alpha=0.01, p=0.01, seeds=(0, 1, 2), log=print):
    path = os.path.join(out, "loss.csv")
    done = _done(path, ["case", "model", "loss", "confirmed", "seed"])
    for name in cases:
        b = load_case(name, data_dir, bench_dir)
        F = true_values(b.eval, b.func)
        d0, _ = _start(b, alpha, p)
        for model in models:
            for q in rates:
                if q == 0 and model != models[0]:
                    continue
                for confirmed in (False, True):
                    for seed in seeds:
                        key = (name, model, str(q), str(confirmed), str(seed))
                        if key in done:
                            continue
                        cal = Calibration(alpha, p * per_step(b), seed=seed, ramp=RAMP)
                        run = run_filter(b.eval, b.func, b.eps, d0, F, cal=cal, loss=q,
                                         loss_model=model, confirmed=confirmed,
                                         loss_seed=seed)
                        n = max(run.audits, 1)
                        _append(path, dict(case=name, model=model, loss=q,
                                           confirmed=confirmed, seed=seed, alpha=alpha,
                                           bound=alpha + cal.slack(run.audits),
                                           **counters(run)))
        log(f"  {name}: loss runs done")


# ------------------------------------------------------------ network size

def scaled_intel(data_dir: str, function: str, k: int, seed: int = 0, max_lag: int = 30):
    """The Intel deployment with k nodes.

    Up to 37 nodes are distinct motes (a random subset for k < 37).  Beyond 37,
    each extra node replays a randomly chosen mote shifted by 1 to ``max_lag``
    minutes, so every node has a realistic but distinct series.  The case is
    then built exactly as the 37-mote case (splits, standardisation, network,
    tolerance).
    """
    raw = np.load(os.path.join(data_dir, "intel.npz"))["X"].astype(float)
    rng = np.random.default_rng(seed)
    T, m, _ = raw.shape
    motes = rng.permutation(m)[:min(k, m)].tolist()
    lags = [0] * len(motes)
    while len(motes) < k:
        motes.append(int(rng.integers(m)))
        lags.append(int(rng.integers(1, max_lag + 1)))
    Z = np.stack([raw[max_lag - lg: T - lg, j] for j, lg in zip(motes, lags)], axis=1)
    file, w, h = DATASETS["intel"]
    Z = _trailing_mean(Z, w)
    T = len(Z)
    a, b = int(TRAIN * T), int((TRAIN + TUNE) * T)
    mu, sd = Z[:a].mean(axis=(0, 1)), Z[:a].std(axis=(0, 1))
    Z = (Z - mu) / sd
    if function == "variance":
        f = Variance()
        X = f.local(Z[..., :1])
        f.scale = X[:a].reshape(-1, 2).std(axis=0)
    else:
        mz = Z.mean(axis=1)
        f = TanhMLP(Z.shape[2], seed=0).fit(mz[:b - h], mz[h:b, 0], seed=0)
        X = Z
    Fv = f.f(X[:b].mean(axis=1))
    eps = 0.1 * float(Fv.std())
    nan = float("nan")
    return Bench(f"intel-k{k}/{function}", f, X[a:b], X[b:], eps, X.shape[2], k,
                 nan, nan, nan, nan, nan, bool(f.const_hessian))


def scale(data_dir, out, sizes=(10, 37, 100, 300, 1000), functions=("variance", "mlp"),
          alpha=0.01, p=0.01, seeds=(0, 1, 2), log=print):
    path = os.path.join(out, "scale.csv")
    done = _done(path, ["k", "function", "method", "seed"])
    for function in functions:
        for k in sizes:
            b = scaled_intel(data_dir, function, k)
            F = true_values(b.eval, b.func)
            d0, _ = _start(b, alpha, p)
            for seed in seeds:
                if (str(k), function, "sod-calibrated", str(seed)) in done:
                    continue
                cal = Calibration(alpha, p, seed=seed, ramp=RAMP)
                run = run_filter(b.eval, b.func, b.eps, d0, F, cal=cal)
                _append(path, dict(k=k, function=function, method="sod-calibrated",
                                   seed=seed, **counters(run)))
            if (str(k), function, "zone", "-1") not in done:
                r = None if b.func.const_hessian else DEPLOY_R["intel/mlp"]
                run = run_zone(b.eval, b.func, b.eps, r=r, F_true=F)
                _append(path, dict(k=k, function=function, method="zone", seed=-1,
                                   **counters(run)))
            log(f"  k={k} {function} done")


# ------------------------------------------------------------- retraining

def retrain(data_dir, out, alphas=(0.01, 0.05), p=0.01, seeds=range(10), log=print):
    """Deployment networks fitted on the first 30% of rounds only."""
    from .cases import readings
    rows = []
    for ds in ("beijing", "intel"):
        Z = readings(data_dir, ds)
        T = len(Z)
        a, b_ = int(TRAIN * T), int((TRAIN + TUNE) * T)
        h = DATASETS[ds][2]
        m = Z.mean(axis=1)
        f = TanhMLP(Z.shape[2], seed=0).fit(m[:a - h], m[h:a, 0], seed=0)
        Fv = f.f(Z[:b_].mean(axis=1))
        eps = 0.1 * float(Fv.std())
        nan = float("nan")
        b = Bench(f"{ds}/mlp-train30", f, Z[a:b_], Z[b_:], eps, Z.shape[2], Z.shape[1],
                  nan, nan, nan, nan, nan, False)
        F = true_values(b.eval, f)
        for alpha in alphas:
            d0, _ = _start(b, alpha, p)
            for seed in seeds:
                run = run_filter(b.eval, f, eps, d0, F,
                                 cal=Calibration(alpha, p, seed=seed, ramp=RAMP))
                rows.append(dict(case=b.name, alpha=alpha, seed=seed, **counters(run)))
        log(f"  {ds}: retrained network done")
    pd.DataFrame(rows).to_csv(os.path.join(out, "retrain.csv"), index=False)
    return rows
