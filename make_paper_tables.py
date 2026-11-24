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


CASES = list(NAMES)
LONG = {"inner_product", "quadratic", "MLP_40", "MLP_80"}
# dimension, updates per hour, nodes
SPEC = {"beijing/variance": (2, 1, 12), "beijing/mlp": (6, 1, 12),
        "intel/variance": (2, 60, 37), "intel/mlp": (3, 60, 37),
        "inner_product": (40, 60, 10), "quadratic": (40, 60, 10), "MLP_40": (40, 60, 10),
        "MLP_80": (80, 60, 10), "air_quality": (20, 1, 12), "intrusion_detection": (41, 60, 9)}


def key(case):
    """Name of a case in the revision results (joined streams for synthetic ones)."""
    return case + "_long" if case in LONG else case


def zones():
    """Safe-zone runs of the revision, merged from all worker folders."""
    parts = []
    for root, _, files in os.walk(R("results_revision")):
        if "zones.csv" in files:
            parts.append(pd.read_csv(os.path.join(root, "zones.csv")))
    z = pd.concat(parts).drop_duplicates(["case", "method", "seed"])
    plain = z[z.method == "zone"].set_index("case")
    cal = z[z.method == "zone-calibrated"].groupby("case").agg(
        msgs=("msg_ratio", "mean"), viol=("viol_rate", "mean"), seeds=("seed", "size"))
    return plain, cal


def sod_runs():
    return pd.read_csv(R("results_revision", "sod.csv"))


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
    zp, zc = zones()
    sd = sod_runs()
    off = sd[sd.method == "sod-offline"].set_index(["case", "alpha"])
    rows = []
    for case in CASES:
        d, _, k = SPEC[case]
        stage2 = "/" in case
        kk = key(case)
        if stage2:
            zone, src = am.loc[case, "msg_ratio"], "o"
            per, sod = tol.loc[case, "periodic_zero_viol"], tol.loc[case, "send_on_delta_zero_viol"]
            lip = s2.loc[case, "filter_guaranteed"]
            cal = ho[ho.variant == "near-miss 0.75"]
        else:
            h = hd.loc[kk]
            if np.isfinite(h.safe_zone):
                zone, src = h.safe_zone, ("o" if h.safe_zone_source == "original" else "r")
            elif kk in zp.index:
                zone, src = zp.loc[kk, "msg_ratio"], "r"
            else:
                zone, src = np.nan, "r"
            per = hd.loc[case].periodic_zero_viol          # published stream
            sod, lip = h.filter_zero_viol, h.filter_guaranteed
            cal = cb[cb.variant == "near-miss 0.75"]
        c1 = cal[(cal.case == kk) & np.isclose(cal.alpha, 0.01)].iloc[0]
        c5 = cal[(cal.case == kk) & np.isclose(cal.alpha, 0.05)].iloc[0]
        amc = zc.loc[kk] if kk in zc.index else None
        o1 = off.loc[(kk, 0.01)]
        o5 = off.loc[(kk, 0.05)]
        rows.append(dict(case=case, name=NAMES[case], d=d, k=k, zone=zone, src=src, per=per,
                         sod=sod, lip=lip, amc=None if amc is None else amc.msgs,
                         amc_v=None if amc is None else amc.viol,
                         off=o1.msg_ratio, off_v=o1.viol_rate, off5=o5.msg_ratio,
                         off5_v=o5.viol_rate, m1=c1.msgs, v1=c1.viol, m5=c5.msgs, v5=c5.viol,
                         red=1 - c1.msgs / zone, hs=sod / zone))
    lines = []
    for i, r in enumerate(rows):
        mark = "" if r["src"] == "o" else "$^{\\dagger}$"
        amc = "n/r" if r["amc"] is None else f"{pct(r['amc'])} ({pct(r['amc_v'])})"
        lines.append(f"{r['name']} & {r['d']} & {r['k']} & {pct(r['zone'])}{mark} & {pct(r['per'])} & "
                     f"{pct(r['sod'])} & {pct(r['lip'], 'n/a')} & {amc} & "
                     f"{pct(r['off'])} ({pct(r['off_v'])}) & {pct(r['m1'])} ({pct(r['v1'])}) & "
                     f"{pct(r['m5'])} ({pct(r['v5'])}) & {pct(r['red'])} \\\\")
        if i == 3:
            lines.append("\\midrule")
    open(os.path.join(out, "tab_main.tex"), "w").write("\n".join(lines) + "\n\\bottomrule\n")
    return pd.DataFrame(rows)


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


def energy_rows():
    from monitor_pilot.ablation import battery_years, time_on_air_ms
    zp, _ = zones()
    sd = sod_runs()
    c = sd[(sd.method == "sod-calibrated") & np.isclose(sd.alpha, 0.01)].groupby("case").mean(
        numeric_only=True)
    rows = []
    for case in CASES:
        d, rate, k = SPEC[case]
        kk = key(case)
        if kk not in zp.index:
            continue
        z, s_ = zp.loc[kk], c.loc[kk]
        by = lambda v, idle=None: battery_years(v, rate, d, idle_ma=idle)  # noqa: E731
        tau = time_on_air_ms(13 + 2 * d) / 1000.0
        rows.append(dict(
            case=case, label=NAMES[case], rate="1/h" if rate == 1 else "1/min", per_hour=rate,
            full_mean=by(1.0), full_busiest=by(1.0),
            zone_mean=by(z.mean_classA), zone_busiest=by(z.busiest_classA),
            sod_mean=by(s_.mean_classA), sod_busiest=by(s_.busiest_classA),
            zone_all=by(z.msg_ratio), sod_all=by(s_.msg_ratio),
            zone_idle=by(z.mean_classA, 1.5), sod_idle=by(s_.mean_classA, 1.5),
            gw_zone=z.down_ratio * k * rate * tau / 3600.0,
            gw_sod=s_.down_ratio * k * rate * tau / 3600.0,
            zone_unsol=z.unsolicited_ratio, zone_down=z.down_ratio))
    e = pd.DataFrame(rows)
    e["gain"] = e.sod_mean / e.zone_mean
    e["gain_busiest"] = e.sod_busiest / e.zone_busiest
    e["gain_all"] = e.sod_all / e.zone_all
    e["gain_idle"] = e.sod_idle / e.zone_idle
    return e


def energy_table(out):
    e = energy_rows()
    lines = []
    for r in e.itertuples():
        lines.append(f"{r.label} & {r.rate} & {r.full_mean:.2f} & {r.zone_mean:.2f} ({r.zone_busiest:.2f}) & "
                     f"{r.sod_mean:.2f} ({r.sod_busiest:.2f}) & {r.gain:.1f} & {100 * r.gw_zone:.2f} \\\\")
    open(os.path.join(out, "tab_energy.tex"), "w").write("\n".join(lines) + "\n\\bottomrule\n")
    return e


def alerts_table(out):
    s = pd.read_csv(R("results_alerts", "summary.csv"))
    s = s[s.tolerance_ug.isin([0.0, 5.0])]
    lab = {"full reporting": "Full reporting", "safe zones (AutoMon re-implementation)": "Safe zones",
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
    sys.path.insert(0, HERE)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    pd.set_option("display.width", 250)
    m = main_table(out)
    print(m.round(4).to_string())
    m.to_csv(os.path.join(out, "main_numbers.csv"), index=False)
    variants_table(out)
    e = energy_table(out)
    print(e.round(3).to_string())
    e.to_csv(os.path.join(out, "energy_numbers.csv"), index=False)
    alerts_table(out)
