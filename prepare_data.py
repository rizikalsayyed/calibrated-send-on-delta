#!/usr/bin/env python3
"""Build the compact arrays in data/ from the two public datasets.

    python prepare_data.py --beijing PRSA_DIR --intel INTEL_DATA_TXT --out data

Beijing Multi-Site Air Quality (UCI): the 12 PRSA_Data_*_20130301-20170228.csv
files.  -> data/beijing.npz, X of shape (hours, 12 stations, 6):
PM2.5, PM10, NO2, O3, TEMP, WSPM.  Gaps are filled by linear interpolation in time.

Intel Berkeley Research Lab: data.txt (date time epoch moteid temperature
humidity light voltage).  -> data/intel.npz, X of shape (minutes, motes, 3):
temperature, humidity, log(1 + light), on 1-minute bins from 2004-02-28 01:00
to 2004-03-21 00:00, motes with readings in at least 60% of the bins.
Physically impossible readings are dropped; gaps carry the last reading forward.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

BEIJING_COLS = ["PM2.5", "PM10", "NO2", "O3", "TEMP", "WSPM"]
INTEL_COLS = ["temperature", "humidity", "log_light"]
START, END = "2004-02-28 01:00", "2004-03-21 00:00"


def beijing(folder):
    files = sorted(glob.glob(os.path.join(folder, "**", "PRSA_Data_*_20130301-20170228.csv"),
                             recursive=True))
    assert len(files) == 12, f"expected 12 station files, found {len(files)}"
    out, names = [], []
    for f in files:
        df = pd.read_csv(f)
        df.index = pd.to_datetime(df[["year", "month", "day", "hour"]])
        v = df[BEIJING_COLS].astype(float).interpolate(limit_direction="both")
        out.append(v.to_numpy())
        names.append(str(df.station.iloc[0]))
    return np.stack(out, axis=1).astype(np.float32), names


def intel(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, on_bad_lines="skip",
                     names=["date", "time", "epoch", "mote", "temperature", "humidity",
                            "light", "voltage"])
    df = df.dropna(subset=["mote", "temperature", "humidity", "light"])
    df["ts"] = pd.to_datetime(df.date + " " + df.time, errors="coerce", format="mixed")
    df = df.dropna(subset=["ts"])
    df = df[(df.temperature > -10) & (df.temperature < 60) & (df.humidity >= 0) &
            (df.humidity <= 100) & (df.light >= 0)]
    df["mote"] = df.mote.astype(int)
    df["log_light"] = np.log1p(df.light)
    df["bin"] = df.ts.dt.floor("1min")
    idx = pd.date_range(START, END, freq="1min", inclusive="left")
    df = df[(df.bin >= idx[0]) & (df.bin <= idx[-1])]
    g = df.groupby(["mote", "bin"])[INTEL_COLS].mean()
    cover = g.groupby(level=0).size() / len(idx)
    motes = sorted(cover[cover >= 0.6].index)
    out = []
    for m in motes:
        s = g.loc[m].reindex(idx).ffill().bfill()
        out.append(s.to_numpy())
    return np.stack(out, axis=1).astype(np.float32), motes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--beijing", required=True)
    ap.add_argument("--intel", required=True)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X, names = beijing(a.beijing)
    np.savez_compressed(os.path.join(a.out, "beijing.npz"), X=X, stations=names,
                        columns=BEIJING_COLS)
    print("beijing", X.shape)
    X, motes = intel(a.intel)
    np.savez_compressed(os.path.join(a.out, "intel.npz"), X=X, motes=motes, columns=INTEL_COLS)
    print("intel", X.shape)


if __name__ == "__main__":
    main()
