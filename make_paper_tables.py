#!/usr/bin/env python3
"""Write the LaTeX tables of the paper from the result CSVs (no number is typed by hand).

    python make_paper_tables.py OUT_DIR
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(HERE, *p)  # noqa: E731


def pct(v, dash="--"):
    if v is None or not np.isfinite(v):
        return dash
    v = 100 * v
    return f"{v:.0f}" if v >= 9.95 else f"{v:.1f}"


NAMES = {"beijing/variance": "Beijing, variance", "beijing/mlp": "Beijing, network",
         "intel/variance": "Intel, variance", "intel/mlp": "Intel, network",
         "inner_product": "Inner product", "quadratic": "Quadratic form",
         "MLP_40": "MLP-40", "MLP_80": "MLP-80", "air_quality": "KLD air quality",
         "intrusion_detection": "DNN intrusion detection"}


def main_table(out):
    s2 = pd.read_csv(R("results", "summary.csv"))
    s2 = s2[(s2.eps_rel == 0.1)].drop_duplicates("case").set_index("case")
    am = pd.read_csv(R("results", "automon.csv")).set_index("case")
    tol = pd.read_csv(R("results_stage5", "tolerance_summary.csv"))
    tol = tol[tol.tol_mult == 1.0].set_index("case")
    ho = pd.read_csv(R("results_calibration", "heldout", "summary.csv"))
    cb = pd.read_csv(R("results_calibration", "summary.csv"))
    hd = pd.read_csv(R("results_highdim", "summary.csv"))
    hd = hd.drop_duplicates("case").set_index("case")
    rows = []
    for case in NAMES:
        stage2 = "/" in case
        if stage2:
            d, k = (2 if "variance" in case else (6 if "beijing" in case else 3),
                    12 if "beijing" in case else 37)
            zone, src = am.loc[case, "msg_ratio"], "o"
            per, sod = tol.loc[case, "periodic_zero_viol"], tol.loc[case, "send_on_delta_zero_viol"]
            lip = s2.loc[case, "filter_guaranteed"]
            cal = ho[ho.variant == "near-miss 0.75"]
            key = case
        else:
            h = hd.loc[case]
            d, k = int(h.d), int(h.nodes)
            zone, src = h.safe_zone, ("o" if h.safe_zone_source == "original" else "r")
            per, sod, lip = h.periodic_zero_viol, h.filter_zero_viol, h.filter_guaranteed
            cal = cb[cb.variant == "near-miss 0.75"]
            key = case if case in ("air_quality", "intrusion_detection") else case + "_long"
        c1 = cal[(cal.case == key) & np.isclose(cal.alpha, 0.01)].iloc[0]
        c5 = cal[(cal.case == key) & np.isclose(cal.alpha, 0.05)].iloc[0]
        rows.append((NAMES[case], d, k, zone, src, per, sod, lip, c1.msgs, c1.viol, c5.msgs,
                     c5.viol, 1 - c1.msgs / zone))
    lines = []
    for i, (n, d, k, z, src, per, sod, lip, m1, v1, m5, v5, red) in enumerate(rows):
        mark = "" if src == "o" else "$^{\\dagger}$"
        lines.append(f"{n} & {d} & {k} & {pct(z)}{mark} & {pct(per)} & {pct(sod)} & "
                     f"{pct(lip, 'n/a')} & {pct(m1)} ({pct(v1)}) & {pct(m5)} ({pct(v5)}) & "
                     f"{pct(red)} \\\\")
        if i == 3:
            lines.append("\\midrule")
    open(os.path.join(out, "tab_main.tex"), "w").write("\n".join(lines) + "\n\\bottomrule\n")
    reds = [r[-1] for r in rows]
    return dict(red_min=min(reds), red_max=max(reds))


def variants_table(out):
    order = ["indicator", "indicator, 2x audits", "indicator, front-loaded", "near-miss 0.9",
             "near-miss 0.75", "near-miss 0.5"]
    label = {"indicator": "Indicator (plain rule)", "indicator, 2x audits": "Indicator, audits doubled",
             "indicator, front-loaded": "Indicator, audits front-loaded",
             "near-miss 0.9": "Near-miss, $c=0.9$", "near-miss 0.75": "Near-miss, $c=0.75$",
             "near-miss 0.5": "Near-miss, $c=0.5$"}
    o = pd.read_csv(R("results_calibration", "overall.csv"))
    h = pd.read_csv(R("results_calibration", "heldout", "overall.csv"))
    lines = []
    for v in order:
        a = o[(o.variant == v) & np.isclose(o.alpha, 0.01)].iloc[0]
        b = h[(h.variant == v) & np.isclose(h.alpha, 0.01)].iloc[0]
        lines.append(f"{label[v]} & {pct(a.viol_mean, '')} & {int(a.benchmarks_within_target)}/{int(a.benchmarks)} & "
                     f"{pct(a.runs_within_target)} & {a.msgs_vs_indicator:.2f} & "
                     f"{int(b.benchmarks_within_target)}/{int(b.benchmarks)} & {b.msgs_vs_indicator:.2f} \\\\")
    open(os.path.join(out, "tab_variants.tex"), "w").write("\n".join(lines) + "\n\\bottomrule\n")


def energy_table(out):
    e = pd.read_csv(R("results_stage5", "energy.csv"))
    e = e[np.isclose(e.alpha, 0.01)]
    lines = []
    for r in e.itertuples():
        case = r.case.replace("_long", "")
        rate = "1/h" if r.updates_per_hour == 1 else "1/min"
        lines.append(f"{NAMES[case]} & {rate} & {r.full_years:.2f} & {r.safe_zone_years:.2f} & "
                     f"{r.calibrated_years:.2f} & {r.gain_vs_safe_zone:.1f} \\\\")
    open(os.path.join(out, "tab_energy.tex"), "w").write("\n".join(lines) + "\n\\bottomrule\n")


def alerts_table(out):
    s = pd.read_csv(R("results_alerts", "summary.csv"))
    s = s[s.tolerance_ug.isin([0.0, 5.0])]
    lab = {"full reporting": "Full reporting", "safe zones (AutoMon re-implementation)": "Safe zones (AutoMon)",
           "send-on-delta, calibrated to 1%": "Calibrated send-on-delta, $\\alpha=1\\%$",
           "send-on-delta, calibrated to 5%": "Calibrated send-on-delta, $\\alpha=5\\%$"}
    lines = []
    for r in s.itertuples():
        name = lab[r.method] + (", guard band" if r.rule == "guard band" else "")
        delay = "0" if r.mean_onset_delay_h == 0 else f"{r.mean_onset_delay_h:.1f}"
        lines.append(f"{name} & {pct(r.msg_ratio)} & {r.battery_years:.1f} & {r.missed_hours:.0f} & "
                     f"{r.false_hours:.0f} & {r.episodes_missed:.1f} & {r.false_episodes:.1f} & {delay} \\\\")
    open(os.path.join(out, "tab_alerts.tex"), "w").write("\n".join(lines) + "\n\\bottomrule\n")


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    print(main_table(out))
    variants_table(out)
    energy_table(out)
    alerts_table(out)
