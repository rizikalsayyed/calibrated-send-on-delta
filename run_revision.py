#!/usr/bin/env python3
"""Stage 7: experiments requested in review (see monitor_pilot/revision.py).

    python run_revision.py --part sod|loss|retrain|scale|zones --out results_revision
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitor_pilot import revision as R  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", required=True,
                    choices=["sod", "loss", "retrain", "scale", "zones", "zones-plain"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--out", default="results_revision")
    ap.add_argument("--cases", nargs="*", default=list(R.CASES))
    ap.add_argument("--sizes", type=int, nargs="*", default=[10, 37, 100, 300, 1000])
    ap.add_argument("--functions", nargs="*", default=["variance", "mlp"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    if a.part == "sod":
        R.sod(a.data, a.bench, a.out, a.cases, log=log)
    elif a.part == "loss":
        R.loss(a.data, a.bench, a.out, a.cases, log=log)
    elif a.part == "retrain":
        R.retrain(a.data, a.out, log=log)
    elif a.part == "scale":
        R.scale(a.data, a.out, tuple(a.sizes), tuple(a.functions), log=log)
    elif a.part == "zones":
        R.zones(a.data, a.bench, a.out, a.cases, seeds=tuple(a.seeds), log=log)
    else:
        R.zones(a.data, a.bench, a.out, a.cases, calibrated=False, log=log)


if __name__ == "__main__":
    main()
