#!/usr/bin/env python3
"""The original AutoMon library on the Stage-2 cases.

    git clone https://github.com/hsivan/automon AUTOMON_DIR
    pip install autograd pyzmq cvxpy
    python run_automon_baseline.py --automon AUTOMON_DIR --data data --out results

Writes results/automon.csv: messages per node update and violation rate on the
evaluation rounds of each case, with AutoMon's own message count.  The
neighbourhood size for the network is the one the re-implementation tuned.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from monitor_pilot.automon_glue import ArrayGenerator, prepare  # noqa: E402
from monitor_pilot.cases import ALL_CASES, build_case  # noqa: E402
from monitor_pilot.experiment import tune_radius  # noqa: E402


def automon_function(func):
    import autograd.numpy as anp
    if func.name == "variance":
        return lambda x: x[1] - x[0] ** 2
    W1, b1, w2, b2 = func.W1, func.b1, func.w2, func.b2
    return lambda x: anp.dot(w2, anp.tanh(anp.dot(W1, x) + b1)) + b2


def run_automon(X, func, eps, r):
    from automon import AutomonCoordinator, AutomonNode
    from test_utils.test_utils import run_test
    f = automon_function(func)
    k, d = X.shape[1], X.shape[2]
    kw = dict(max_f_val=func.f_max, min_f_val=func.f_min)
    nodes = [AutomonNode(i, f, d=d, **kw) for i in range(k)]
    coord = AutomonCoordinator(k, f, error_bound=eps, d=d, neighborhood_size=r, **kw)
    run_test(ArrayGenerator(X), coord, nodes, None)
    st = coord.statistics
    err = np.array(st.function_approximation_error, dtype=float)
    return dict(msgs=int(st._total_msgs_for_disabled_broadcast()), err=err,
                full_syncs=len(st.full_sync_history))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--automon", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cases", nargs="*", default=ALL_CASES)
    ap.add_argument("--eps-rel", type=float, nargs="*", default=[0.1])
    a = ap.parse_args()
    prepare(a.automon)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "automon.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    for name in a.cases:
        for eps_rel in a.eps_rel:
            t0 = time.time()
            case = build_case(a.data, *name.split("/"), eps_rel=eps_rel)
            r = tune_radius(case)
            res = run_automon(case.X_eval, case.func, case.eps, r)
            X = case.X_eval
            rows = [x for x in rows if not (x["case"] == name and np.isclose(x["eps_rel"], eps_rel))]
            rows.append(dict(case=name, eps_rel=eps_rel, r=r, msgs=res["msgs"],
                             msg_ratio=res["msgs"] / (X.shape[0] * X.shape[1]),
                             viol_rate=float(np.mean(res["err"] > case.eps * (1 + 1e-9))),
                             max_err=float(res["err"].max() / case.eps),
                             full_syncs=res["full_syncs"], seconds=round(time.time() - t0)))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
