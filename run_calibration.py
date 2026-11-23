#!/usr/bin/env python3
"""Calibration study: variants of the audit rule at small target rates.

    python run_calibration.py --bench data/bench --out results_calibration

Needs the benchmark streams built by run_automon_benchmarks.py (with --tiles 20
for the synthetic ones).  The same study on the Stage-2 cases, which played no
part in choosing the settings:

    python run_calibration.py --pilot-data data --out results_calibration/heldout \
           --cases beijing/variance beijing/mlp intel/variance intel/mlp

Writes

    results_calibration/runs.csv       one row per calibrated run
    results_calibration/summary.csv    per benchmark, variant and target
    results_calibration/overall.csv    per variant and target, across benchmarks
    results_calibration/fig_calibration.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from monitor_pilot.calib import (VARIANTS, overall, pilot_bench, run_study,  # noqa: E402
                                 summarise)
from monitor_pilot.highdim import load_bench  # noqa: E402

CASES = ("inner_product_long", "quadratic_long", "MLP_40_long", "MLP_80_long",
         "air_quality", "intrusion_detection")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--out", default="results_calibration")
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--pilot-data", default="",
                    help="folder with beijing.npz and intel.npz: run on the Stage-2 cases "
                         "named in --cases (e.g. beijing/mlp) instead of the benchmarks")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if not a.merge:
        rows = []
        for name in a.cases:
            b = pilot_bench(a.pilot_data, name) if a.pilot_data else load_bench(a.bench, name)
            print(f"{name}: d={b.d}, {b.nodes} nodes, {len(b.eval)} evaluation steps",
                  flush=True)
            rows.extend(run_study(b, seeds=range(a.seeds), variants=a.variants,
                                  log=lambda m: print(m, flush=True)))
            pd.DataFrame(rows).to_csv(os.path.join(a.out, f"runs{a.tag or '_main'}.csv"),
                                      index=False)
    parts = [pd.read_csv(os.path.join(a.out, f)) for f in sorted(os.listdir(a.out))
             if f.startswith("runs") and f.endswith(".csv")]
    runs = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["case", "variant", "alpha", "seed"])
    if a.merge:
        runs.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    summ = summarise(runs)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    over = overall(summ)
    over.to_csv(os.path.join(a.out, "overall.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\nAcross benchmarks (rates are shares of steps outside the tolerance):")
    print(over.round(4).to_string(index=False))
    if not a.no_plots:
        from monitor_pilot.plots import fig_calibration
        print("wrote", fig_calibration(summ, a.out))


if __name__ == "__main__":
    main()
