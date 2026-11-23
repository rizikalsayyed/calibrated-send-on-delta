#!/usr/bin/env python3
"""Stage 5: tolerance sweep, trigger and audit-rate ablations, battery life.

    python run_stage5.py --part tolerance --data data --bench data/bench --out results_stage5
    python run_stage5.py --part ablation  --data data --bench data/bench --out results_stage5
    python run_stage5.py --part report    --out results_stage5

Parts write tolerance_runs<tag>.csv and ablation_runs<tag>.csv; ``report``
merges them, writes the summaries, energy.csv and the figures.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from monitor_pilot import ablation as A  # noqa: E402
from monitor_pilot.calib import pilot_bench  # noqa: E402
from monitor_pilot.highdim import LABELS, load_bench  # noqa: E402

STAGE2 = ("beijing/variance", "beijing/mlp", "intel/variance", "intel/mlp")
TOL_BENCH = ("inner_product", "quadratic", "MLP_40", "air_quality")
ABL_BENCH = ("inner_product_long", "quadratic_long", "MLP_40_long", "MLP_80_long",
             "air_quality", "intrusion_detection")
# node updates per hour: native rate for the two real datasets, a nominal
# one-per-minute for AutoMon's benchmarks (which have no time unit)
RATE = {"beijing": 1.0, "intel": 60.0, "air_quality": 1.0}   # air_quality is hourly Beijing data


def bench_for(name, a):
    return pilot_bench(a.data, name) if "/" in name else load_bench(a.bench, name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=("tolerance", "ablation", "report"), required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--bench", default="data/bench")
    ap.add_argument("--out", default="results_stage5")
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    if a.part == "tolerance":
        rows = []
        for name in a.cases or (STAGE2 + TOL_BENCH):
            b = bench_for(name, a)
            stage2 = "/" in name
            mults = (0.5, 1.0, 2.0, 3.0, 5.0) if stage2 else (0.5, 1.0, 2.0)
            log(f"{name}: {len(b.eval)} evaluation steps")
            rows += A.tolerance_sweep(b, mults, calibrate=len(b.eval) >= 5000, log=log)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, f"tolerance_runs{a.tag}.csv"),
                                      index=False)
    elif a.part == "ablation":
        rows = []
        for name in a.cases or (STAGE2 + ABL_BENCH):
            b = bench_for(name, a)
            log(f"{name}: {len(b.eval)} evaluation steps")
            rows += A.ablation(b, log=log)
            pd.DataFrame(rows).to_csv(os.path.join(a.out, f"ablation_runs{a.tag}.csv"),
                                      index=False)
    else:
        report(a.out)


def _merge(out, stem):
    files = sorted(glob.glob(os.path.join(out, f"{stem}_runs*.csv")))
    parts = [pd.read_csv(f) for f in files if not f.endswith(f"{stem}_runs.csv")]
    if parts:
        runs = pd.concat(parts, ignore_index=True)
        runs.to_csv(os.path.join(out, f"{stem}_runs.csv"), index=False)
        for f in files:
            if not f.endswith(f"{stem}_runs.csv"):
                os.remove(f)
    return pd.read_csv(os.path.join(out, f"{stem}_runs.csv"))


def label(case):
    return LABELS.get(case, case.replace("/", ", ").replace("mlp", "network"))


def report(out):
    from monitor_pilot.plots import fig_tolerance, fig_ablation, fig_energy
    pd.set_option("display.width", 230)
    tol = A.summarise_tolerance(_merge(out, "tolerance"))
    tol.to_csv(os.path.join(out, "tolerance_summary.csv"), index=False)
    abl_runs = _merge(out, "ablation")
    abl = A.summarise_ablation(abl_runs)
    abl.to_csv(os.path.join(out, "ablation_summary.csv"), index=False)
    print(tol.round(4).to_string(index=False))
    print(abl.round(4).to_string(index=False))

    # battery life: safe zones (from the Stage 2/3 results) vs calibrated send-on-delta
    here = os.path.dirname(os.path.abspath(__file__))
    s2 = pd.read_csv(os.path.join(here, "results", "summary.csv"))
    s3 = pd.read_csv(os.path.join(here, "results_highdim", "summary.csv"))
    zones = {r.case: r.automon for r in s2[s2.eps_rel == 0.1].itertuples()}
    zones.update({r.case: r.safe_zone for r in s3.itertuples()})
    rows = []
    trig = abl[(abl.study == "trigger") & (abl.method == "send-on-delta, calibrated")]
    for r in trig.itertuples():
        base = r.case.replace("_long", "")
        rate = RATE.get(r.case.split("/")[0], 60.0)
        d = int(abl_runs[abl_runs.case == r.case].d.iloc[0])
        z = zones.get(base, np.nan)
        rows.append(dict(case=r.case, d=d, updates_per_hour=rate, alpha=r.alpha,
                         full_msgs=1.0, safe_zone_msgs=z, calibrated_msgs=r.msgs,
                         calibrated_viol=r.viol,
                         full_years=A.battery_years(1.0, rate, d),
                         safe_zone_years=A.battery_years(z, rate, d) if np.isfinite(z) else np.nan,
                         calibrated_years=A.battery_years(r.msgs, rate, d)))
    energy = pd.DataFrame(rows)
    energy["gain_vs_safe_zone"] = energy.calibrated_years / energy.safe_zone_years
    energy.to_csv(os.path.join(out, "energy.csv"), index=False)
    print(energy.round(3).to_string(index=False))
    for fn, args in ((fig_tolerance, (tol, out)), (fig_ablation, (abl, out)),
                     (fig_energy, (energy, out))):
        print("wrote", fn(*args))


if __name__ == "__main__":
    main()
