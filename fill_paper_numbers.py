#!/usr/bin/env python3
"""Fill the @TOKENS@ of the manuscript source from the result CSVs.

    python fill_paper_numbers.py TABLE_DIR SOURCE.tex OUT.tex

TABLE_DIR is where make_paper_tables.py wrote main_numbers.csv and
energy_numbers.csv.  Every token must be filled, or the script fails.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(HERE, "results_revision", *p)  # noqa: E731
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def p(v, nd=None):
    v = 100 * v
    if nd is None:
        nd = 0 if abs(v) >= 9.95 else 1
    return f"{v:.{nd}f}"


def rng(a, b, fmt=lambda v: f"{v:.1f}"):
    a, b = fmt(a), fmt(b)
    return a if a == b else f"{a}--{b}"


def numbers(tab_dir):
    m = pd.read_csv(os.path.join(tab_dir, "main_numbers.csv"))
    e = pd.read_csv(os.path.join(tab_dir, "energy_numbers.csv"))
    t = {}
    red = m.red.dropna()
    t["REDRANGE"] = f"{p(red.min())}--{p(red.max())}\\%"
    hs = m.hs.dropna()
    t["HSRANGE"] = f"{p(hs.min())}--{p(hs.max())}\\%"
    c = m.dropna(subset=["amc"])
    ch = c.amc / c.zone - 1
    ratio = c.amc / c.m1
    t["AMCALTEXT"] = (f"it changed AutoMon's messages by between {p(ch.min())}\\% and "
                      f"+{p(ch.max())}\\% in the {WORDS[len(c)]} cases where we ran it, and "
                      f"calibrated AutoMon still needed {rng(ratio.min(), ratio.max())} times the "
                      "messages of calibrated send-on-delta").replace("between -", "between $-$")
    t["OFFMISS1"] = WORDS[int((m.off_v > 0.01).sum())]
    w = m.loc[m.off_v.idxmax()]
    t["OFFWORST1"] = f"a realised {p(w.off_v)}\\% ({w['name']})"
    t["OFFMISS5"] = WORDS[int((m.off5_v > 0.05).sum())]
    # step size
    s = pd.read_csv(R("sod.csv"))
    s = s[np.isclose(s.alpha, 0.01)]
    base = s[(s.method == "sod-calibrated") & (s.seed < 3)].groupby("case").msg_ratio.mean()
    for eta, tok in ((0.25, "ETA25"), (1.0, "ETA1")):
        g = s[(s.method == "sod-eta") & np.isclose(s.eta, eta)].groupby("case").agg(
            m=("msg_ratio", "mean"), v=("viol_rate", "mean"))
        rel = g.m / base - 1
        sign = lambda v: (f"$-${p(-v)}" if v < 0 else f"+{p(v)}")  # noqa: E731
        t[tok] = f"{sign(rel.min())}\\% and {sign(rel.max())}\\%"
        if eta == 0.25:
            assert (g.v <= 0.01).all()
        else:
            over = g[g.v > 0.01]
            t["ETA1WORST"] = ", ".join(f"{p(v, 1)}\\%" for v in over.v)
            t["ETA1OVER"] = WORDS[len(over)]
    r = pd.read_csv(R("retrain.csv"))
    r = r[np.isclose(r.alpha, 0.01)].groupby("case").viol_rate.mean()
    t["RETRAIN"] = " and ".join(f"{p(v, 2)}\\%" for v in r.sort_index())
    # loss
    L = pd.read_csv(R("loss.csv"))
    cs = L.groupby(["model", "loss", "confirmed", "case"]).agg(
        v=("viol_rate", "mean"), a=("mean_classA", "mean")).reset_index()
    b0 = cs[(cs.loss == 0) & (~cs.confirmed)].set_index("case").a
    cs["rel"] = cs.a / cs.case.map(b0)
    sel = lambda mo, q, cf: cs[(cs.model == mo) & np.isclose(cs.loss, q) & (cs.confirmed == cf)]  # noqa
    t["L5IND"] = f"{p(sel('bernoulli', 0.05, False).v.mean(), 1)}\\%"
    g20 = sel("ge", 0.2, False)
    t["L20GE"] = f"{p(g20.v.mean(), 1)}\\%"
    t["L20GEOVER"] = WORDS[int((g20.v > 0.01).sum())]
    t["L20GEWORST"] = f"{p(g20.v.max(), 1)}\\%"
    conf = lambda q: pd.concat([sel("bernoulli", q, True), sel("ge", q, True)]).rel.mean() - 1  # noqa
    t["CONF5"] = f"{p(conf(0.05))}\\%"
    t["CONF20"] = f"{p(conf(0.2))}\\%"
    for q in (0.01, 0.05, 0.1, 0.2):          # confirmed reports: violations as without loss
        for mo in ("bernoulli", "ge"):
            a_ = sel(mo, q, True).set_index("case").v
            b_ = cs[(cs.loss == 0) & (~cs.confirmed)].set_index("case").v
            assert np.allclose(a_.sort_index(), b_.sort_index()), (mo, q)
    # network size
    S = pd.read_csv(R("scale.csv")).groupby(["function", "k", "method"]).agg(
        m=("msg_ratio", "mean"), v=("viol_rate", "mean")).reset_index()
    so, zo = S[S.method == "sod-calibrated"], S[S.method == "zone"]
    t["SCSOD"] = f"{p(so.m.min())}--{p(so.m.max())}\\%"
    t["SCZONE"] = f"{p(zo.m.min())}--{p(zo.m.max())}\\%"
    t["SCVIOL"] = f"{p(so.v.min())}--{p(so.v.max())}\\%"
    # energy
    mi = e[e.per_hour == 60]
    it = e[e.case.str.startswith("intel")]
    bn = mi[~mi.case.str.startswith("intel")]
    hr = e[e.per_hour == 1]
    t["INTELFULL"] = f"{it.full_mean.iloc[0]:.2f}"
    t["INTELAM"] = rng(it.zone_mean.min(), it.zone_mean.max(), lambda v: f"{v:.2f}")
    t["INTELSOD"] = rng(it.sod_mean.min(), it.sod_mean.max())
    t["INTELGAIN"] = rng(it.gain.min(), it.gain.max())
    t["INTELBUSY"] = rng(it.gain_busiest.min(), it.gain_busiest.max())
    t["BENCHGAIN"] = rng(bn.gain.min(), bn.gain.max()) + " times"
    t["HOURGAIN"] = rng(hr.gain.min(), hr.gain.max()) + " times"
    t["ALLGAIN"] = rng(mi.gain_all.min(), mi.gain_all.max()) + " times"
    t["IDLEGAIN"] = rng(mi.gain_idle.min(), mi.gain_idle.max()) + " times"
    t["GWAM"] = f"{100 * mi.gw_zone.max():.1f}\\%"
    t["GWSOD"] = f"{100 * mi.gw_sod.max():.2f}\\%"
    return t


def main(tab_dir, src, dst):
    t = numbers(tab_dir)
    s = open(src).read()
    for k_, v in t.items():
        s = s.replace(f"@{k_}@", v)
    left = re.findall(r"@[A-Z0-9]+@", s)
    if left:
        raise SystemExit(f"unfilled tokens: {sorted(set(left))}")
    open(dst, "w").write(s)
    for k_, v in sorted(t.items()):
        print(f"{k_:12s} {v}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
