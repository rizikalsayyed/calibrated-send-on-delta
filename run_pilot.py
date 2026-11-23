#!/usr/bin/env python3
"""Stage 2: every method on the Beijing and Intel cases.

    python run_pilot.py --data data --out results
    python run_pilot.py --out results --cases intel/variance --fast

Writes results/runs<tag>.csv (one row per simulated run); with --merge, or
after a plain run, also runs.csv, summary.csv, fig_messages.png and
fig_guarantee.png.  results/automon.csv, if present (run_automon_baseline.py),
supplies the original AutoMon's numbers.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from monitor_pilot.cases import ALL_CASES, build_case  # noqa: E402
from monitor_pilot.experiment import run_case, summarise  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cases", nargs="*", default=ALL_CASES)
    ap.add_argument("--eps-rel", type=float, nargs="*", default=[0.1])
    ap.add_argument("--fast", action="store_true", help="one seed and one audit rate")
    ap.add_argument("--tag", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if not a.merge:
        rows = []
        for name in a.cases:
            for eps_rel in a.eps_rel:
                case = build_case(a.data, *name.split("/"), eps_rel=eps_rel)
                print(f"{name} eps {eps_rel}: {case.X_eval.shape} evaluation rounds", flush=True)
                rows.extend(run_case(case, eps_rel, fast=a.fast,
                                     log=lambda m: print(m, flush=True)))
                pd.DataFrame(rows).to_csv(os.path.join(a.out, f"runs{a.tag or '_main'}.csv"),
                                          index=False)
    parts = [pd.read_csv(os.path.join(a.out, f)) for f in sorted(os.listdir(a.out))
             if f.startswith("runs") and f.endswith(".csv") and f != "runs.csv"]
    runs = pd.concat(parts, ignore_index=True)
    keys = [c for c in ("case", "eps_rel", "method", "param", "alpha", "p", "seed") if c in runs]
    runs = runs.drop_duplicates(keys)
    runs.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    path = os.path.join(a.out, "automon.csv")
    automon = pd.read_csv(path) if os.path.exists(path) else None
    summ = summarise(runs, automon)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    pd.set_option("display.width", 220)
    cols = ["automon", "automon_source", "filter_guaranteed", "periodic_hindsight",
            "filter_hindsight", "filter_calibrated", "filter_calibrated_viol",
            "saving_vs_automon"]
    print(summ.set_index(["case", "eps_rel", "alpha"])[cols].round(3).to_string())
    if not a.no_plots:
        from monitor_pilot.plots import make_figures
        print("wrote", make_figures(summ, runs, a.out))


if __name__ == "__main__":
    main()
