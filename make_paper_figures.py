#!/usr/bin/env python3
"""Draw every figure of the paper from the result CSVs, without in-figure
titles, and trim the white margins.

    python make_paper_figures.py TABLE_DIR OUT_DIR

TABLE_DIR holds energy_numbers.csv written by make_paper_tables.py.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from monitor_pilot import plots as P  # noqa: E402

R = lambda *p: os.path.join(HERE, *p)  # noqa: E731


def trim(path, pad=12):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    bg = a[0, 0]
    mask = np.abs(a - bg).sum(axis=2) > 18
    ys, xs = np.nonzero(mask)
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad, a.shape[1]), min(ys.max() + pad, a.shape[0]))
    im.crop(box).save(path)


def main(tab_dir, out):
    os.makedirs(out, exist_ok=True)
    P.TITLES = False
    paths = [
        P.fig_calibration(pd.read_csv(R("results_calibration", "summary.csv")), out),
        P.fig_tolerance(pd.read_csv(R("results_stage5", "tolerance_summary.csv")), out),
        P.fig_ablation(pd.read_csv(R("results_stage5", "ablation_summary.csv")), out),
        P.fig_alerts(pd.read_csv(R("results_alerts", "summary.csv")), out),
        P.fig_episode(pd.read_csv(R("results_alerts", "traces.csv"), parse_dates=["time"]), out),
        P.fig_energy2(pd.read_csv(os.path.join(tab_dir, "energy_numbers.csv")), out),
        P.fig_network(pd.read_csv(R("results_revision", "loss.csv")),
                      pd.read_csv(R("results_revision", "scale.csv")), out),
    ]
    for p_ in paths:
        trim(p_)
        print("wrote", p_)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
