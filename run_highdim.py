#!/usr/bin/env python3
"""High-dimensional test on AutoMon's own benchmarks.

    python run_automon_benchmarks.py --automon AUTOMON_DIR --out data/bench
    python run_highdim.py --bench data/bench --out results_highdim

The first command rebuilds each benchmark stream and runs the original AutoMon
on it.  This one runs periodic reporting and send-on-delta (fixed, Lipschitz
bound, calibrated by audits) on the same streams and writes

    results_highdim/runs.csv      one row per simulated run
    results_highdim/summary.csv   messages by method, per benchmark and target
    results_highdim/fig_highdim.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from monitor_pilot.highdim import LABELS, load_bench, run_bench, summarise  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--out", default="results_highdim")
    ap.add_argument("--cases", nargs="*", default=list(LABELS))
    ap.add_argument("--min-steps", type=int, default=5000,
                    help="calibrate by audits only on streams at least this long")
    ap.add_argument("--no-zone", action="store_true",
                    help="skip the safe-zone re-implementation (slow for the networks)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    names = [n for n in a.cases if os.path.exists(os.path.join(a.bench, f"{n}.npz"))]
    benches = [load_bench(a.bench, n) for n in names]
    if not a.merge:
        rows = []
        for b in benches:
            print(f"{b.name}: d={b.d}, {b.nodes} nodes, {len(b.eval)} evaluation steps, "
                  f"AutoMon {b.automon_ratio:.3f}", flush=True)
            rows.extend(run_bench(b, calibrate=len(b.eval) >= a.min_steps,
                                  zone=not a.no_zone))
            pd.DataFrame(rows).to_csv(os.path.join(a.out, f"runs{a.tag or '_main'}.csv"),
                                      index=False)
    parts = [pd.read_csv(os.path.join(a.out, f)) for f in sorted(os.listdir(a.out))
             if f.startswith("runs") and f.endswith(".csv")]
    runs = pd.concat(parts, ignore_index=True)
    keys = [c for c in ("case", "method", "param", "alpha", "p", "seed") if c in runs]
    runs = runs.drop_duplicates(keys)
    if a.merge:
        runs.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    summ = summarise(runs, [b for b in benches if b.name in set(runs.case)])
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    cols = ["d", "steps", "automon", "zone_reimplemented", "filter_guaranteed", "filter_zero_viol",
            "filter_hindsight", "filter_calibrated", "filter_calibrated_viol"]
    pd.set_option("display.width", 220)
    print("\nMessages as a share of every node update being reported:")
    print(summ.set_index(["case", "alpha"])[cols].round(3).to_string())
    if not a.no_plots:
        from monitor_pilot.plots import fig_highdim
        print("wrote", fig_highdim(summ, a.out))


if __name__ == "__main__":
    main()
