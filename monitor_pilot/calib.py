"""Calibration study: how to keep the realised violation rate at a small target.

The audit rule moves the log-tolerance by eta * (alpha - loss) at every audit.
With the plain indicator loss its long-run audited violation share equals
alpha, so any start-up transient pushes a finite run above the target.  Three
remedies are compared, all with the same form of guarantee:

  near-miss loss   the loss rises linearly from 0 at ramp * tolerance to 1 at
                   the tolerance.  It is never below the indicator, so bounding
                   its mean bounds the violation share, and near-misses give a
                   signal long before violations do.
  front-loading    the same expected number of audits, four times denser in
                   the first tenth of the stream (fixed in advance).
  more audits      twice the audit rate.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .highdim import Bench, _sweep, _typical_step
from .protocols import Calibration, Events, run_filter, true_values

ALPHAS = (0.01, 0.02, 0.05)
AUDIT_P = 0.01
# name -> Calibration settings (p_mult multiplies the audit rate)
VARIANTS = {
    "indicator": dict(ramp=1.0),
    "indicator, 2x audits": dict(ramp=1.0, p_mult=2.0),
    "indicator, front-loaded": dict(ramp=1.0, boost=4.0),
    "near-miss 0.9": dict(ramp=0.9),
    "near-miss 0.75": dict(ramp=0.75),
    "near-miss 0.5": dict(ramp=0.5),
    "near-miss 0.75, front-loaded": dict(ramp=0.75, boost=4.0),
    "near-miss 0.5, front-loaded": dict(ramp=0.5, boost=4.0),
}


def pilot_bench(data_dir: str, name: str, eps_rel: float = 0.1) -> Bench:
    """A Stage-2 case ("beijing/mlp", ...) in the form the study expects."""
    from .cases import build_case
    dataset, function = name.split("/")
    c = build_case(data_dir, dataset, function, eps_rel)
    nan = float("nan")
    return Bench(name, c.func, c.X_tune, c.X_eval, c.eps, c.X_eval.shape[2],
                 c.X_eval.shape[1], nan, nan, nan, nan, nan, bool(c.func.const_hessian))


def make(variant: str, alpha: float, p: float, seed: int) -> Calibration:
    kw = dict(VARIANTS[variant])
    return Calibration(alpha=alpha, p=p * kw.pop("p_mult", 1.0), seed=seed, **kw)


def start_tolerance(tune_runs, cal: Calibration, eps: float) -> float:
    """Largest tolerance whose mean loss on the tuning stream is within alpha."""
    ok = [d for d, run in tune_runs if cal.loss(run.err / eps).mean() <= cal.alpha]
    return max(ok) if ok else float(tune_runs[0][0])


def run_study(b: Bench, seeds=range(10), variants=tuple(VARIANTS), alphas=ALPHAS,
              p: float = AUDIT_P, log=print) -> List[dict]:
    f, eps, X = b.func, b.eps, b.eval
    F = true_values(X, f)
    per_step = 1.0 / b.nodes if isinstance(X, Events) else 1.0
    tune_runs = _sweep(b.tune, f, eps, true_values(b.tune, f), _typical_step(b.tune))
    rows: List[dict] = []
    for variant in variants:
        for alpha in alphas:
            d0 = start_tolerance(tune_runs, make(variant, alpha, p, 0), eps)
            for seed in seeds:
                cal = make(variant, alpha, p, seed)
                p_round = cal.p
                cal.p = cal.p * per_step
                run = run_filter(X, f, eps, d0, F, cal=cal)
                n = max(run.audits, 1)
                u = np.array(run.u_trace) if run.u_trace else np.zeros(1)
                rows.append(dict(
                    case=b.name, variant=variant, ramp=cal.ramp, boost=cal.boost,
                    p=p_round, alpha=alpha, seed=seed, start_tolerance=d0,
                    msg_ratio=run.msg_ratio, viol_rate=run.viol_rate, max_err=run.max_err,
                    audits=run.audits, audit_viol_share=run.audit_violations / n,
                    audit_loss_share=run.audit_loss / n, bound=alpha + cal.slack(run.audits),
                    u_final=float(u[-1]), u_lowest=float(u.min()),
                    hit_floor=bool(u.min() <= cal.u_min)))
        log(f"    {b.name}: {variant} done")
    return rows


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Per benchmark, variant and target: realised rate and messages over seeds."""
    base = (runs[runs.variant == "indicator"].groupby(["case", "alpha"]).msg_ratio.mean()
            .rename("indicator_msgs"))
    runs = runs.assign(within=runs.viol_rate <= runs.alpha,
                       bound_ok=runs.audit_loss_share <= runs.bound + 1e-12)
    g = runs.groupby(["case", "variant", "alpha"], sort=False)
    out = g.agg(seeds=("seed", "size"), msgs=("msg_ratio", "mean"),
                viol=("viol_rate", "mean"), viol_worst=("viol_rate", "max"),
                audits=("audits", "mean"), audit_viol_share=("audit_viol_share", "mean"),
                audit_loss_share=("audit_loss_share", "mean"), bound=("bound", "mean"),
                hit_floor=("hit_floor", "mean"), runs_within_target=("within", "mean"),
                bound_held=("bound_ok", "all")).reset_index()
    out = out.merge(base, on=["case", "alpha"], how="left")
    out["msgs_vs_indicator"] = out.msgs / out.indicator_msgs
    return out


def overall(summ: pd.DataFrame) -> pd.DataFrame:
    """Per variant and target, across benchmarks."""
    summ = summ.assign(within=summ.viol <= summ.alpha)
    g = summ.groupby(["variant", "alpha"], sort=False)
    return g.agg(benchmarks=("case", "size"), benchmarks_within_target=("within", "sum"),
                 viol_mean=("viol", "mean"), viol_highest=("viol", "max"),
                 viol_worst_run=("viol_worst", "max"),
                 runs_within_target=("runs_within_target", "mean"),
                 msgs_vs_indicator=("msgs_vs_indicator", "mean"),
                 msgs_vs_indicator_max=("msgs_vs_indicator", "max"),
                 bound_held=("bound_held", "all")).reset_index()
