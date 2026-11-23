"""Stage 2: sweeps over every method on one case, and the summary table."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .cases import Case
from .functions import TanhMLP
from .protocols import (Calibration, run_central, run_filter, run_periodic, run_zone,
                        true_values)

PERIODS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
DELTAS = np.geomspace(0.01, 1.6, 23)
THETAS = (1.5, 2.0, 3.0, 5.0)
RADII = (0.1, 0.4, 1.6)
ALPHAS = (0.01, 0.02, 0.05)
AUDIT_P = (0.005, 0.01, 0.02)
SEEDS = (0, 1, 2)


def lipschitz(func) -> Optional[float]:
    """Global Lipschitz constant where one exists (the tanh network)."""
    if isinstance(func, TanhMLP):
        return float(np.sum(np.abs(func.w2) * np.linalg.norm(func.W1, axis=1)))
    return None


def tune_delta(case: Case, alpha: float) -> float:
    """Largest send-on-delta tolerance that met alpha on the tuning rounds."""
    X, f = case.X_tune, case.func
    F = true_values(X, f)
    ok = [d for d in DELTAS if run_filter(X, f, case.eps, d, F).viol_rate <= alpha]
    return float(max(ok)) if ok else float(DELTAS[0])


def tune_radius(case: Case) -> Optional[float]:
    """Neighbourhood size with the fewest messages on the tuning rounds."""
    if case.func.const_hessian:
        return None
    X, f = case.X_tune, case.func
    F = true_values(X, f)
    return min(RADII, key=lambda r: run_zone(X, f, case.eps, r=r, F_true=F).msgs)


def run_case(case: Case, eps_rel: float, fast: bool = False, log=print) -> List[dict]:
    X, f, eps = case.X_eval, case.func, case.eps
    F = true_values(X, f)
    rows: List[dict] = []
    seeds = SEEDS[:1] if fast else SEEDS
    audit_p = (0.01,) if fast else AUDIT_P

    def add(method, run, **kw):
        rows.append(dict(case=case.name, eps_rel=eps_rel, method=method,
                         msg_ratio=run.msg_ratio, viol_rate=run.viol_rate,
                         mean_err=run.mean_err, max_err=run.max_err,
                         full_syncs=run.full_syncs, audits=run.audits,
                         audit_viol_share=run.audit_violations / max(run.audits, 1), **kw))

    add("central", run_central(X, f, eps))
    for p in PERIODS:
        add("periodic", run_periodic(X, f, eps, p, F), param=p)
    for d in DELTAS:
        add("filter-fixed", run_filter(X, f, eps, d, F), param=d)
    L = lipschitz(f)
    if L is not None:
        add("filter-guaranteed", run_filter(X, f, eps, eps / L, F), param=eps / L)
    r = tune_radius(case)
    zone = run_zone(X, f, eps, r=r, F_true=F)
    add("zone", zone, param=np.nan if r is None else r)
    log(f"    {case.name} eps {eps_rel}: zone {zone.msg_ratio:.3f}, r={r}")
    for th in THETAS:
        add("zone-relaxed", run_zone(X, f, eps, r=r, theta=th, F_true=F), param=th)
    for alpha in ALPHAS:
        d0 = tune_delta(case, alpha)
        for p in audit_p:
            for seed in seeds:
                cal = Calibration(alpha=alpha, p=p, seed=seed)
                add("filter-calibrated", run_filter(X, f, eps, d0, F, cal=cal),
                    alpha=alpha, p=p, seed=seed, param=d0)
                cal = Calibration(alpha=alpha, p=p, seed=seed)
                add("zone-calibrated", run_zone(X, f, eps, r=r, F_true=F, cal=cal),
                    alpha=alpha, p=p, seed=seed, param=np.nan if r is None else r)
    log(f"    {case.name} eps {eps_rel}: calibrated runs done")
    return rows


def summarise(runs: pd.DataFrame, automon: Optional[pd.DataFrame] = None,
              p: float = 0.01) -> pd.DataFrame:
    """Messages by method per case, tolerance and target violation rate.

    ``automon`` is the original code (automon.csv) where it ran, else the
    re-implementation (``zone``).
    """
    out = []
    for (case, eps_rel), g in runs.groupby(["case", "eps_rel"], sort=False):
        zone = g[g.method == "zone"].iloc[0]
        am, src, am_viol = zone.msg_ratio, "re-implementation", zone.viol_rate
        if automon is not None and len(automon):
            a = automon[(automon.case == case) & np.isclose(automon.eps_rel, eps_rel)]
            if len(a):
                am, src, am_viol = a.msg_ratio.iloc[0], "original", a.viol_rate.iloc[0]
        guar = g[g.method == "filter-guaranteed"]

        def best(method, level):
            d = g[(g.method == method) & (g.viol_rate <= level + 1e-12)]
            return float(d.msg_ratio.min()) if len(d) else np.nan

        for alpha in sorted(g.alpha.dropna().unique()):
            fc = g[(g.method == "filter-calibrated") & (g.alpha == alpha) & (g.p == p)]
            zc = g[(g.method == "zone-calibrated") & (g.alpha == alpha) & (g.p == p)]
            row = dict(case=case, eps_rel=eps_rel, alpha=alpha, automon=am,
                       automon_viol=am_viol, automon_source=src, zone_reimpl=zone.msg_ratio,
                       filter_guaranteed=guar.msg_ratio.iloc[0] if len(guar) else np.nan,
                       periodic_hindsight=best("periodic", alpha),
                       filter_hindsight=best("filter-fixed", alpha),
                       zone_calibrated=zc.msg_ratio.mean(), zone_calibrated_viol=zc.viol_rate.mean(),
                       filter_calibrated=fc.msg_ratio.mean(),
                       filter_calibrated_viol=fc.viol_rate.mean(),
                       filter_calibrated_viol_max=fc.viol_rate.max(),
                       audits=fc.audits.mean())
            row["saving_vs_automon"] = 1 - row["filter_calibrated"] / am
            out.append(row)
    return pd.DataFrame(out)
