"""Figures for the pilot (matplotlib, static PNG)."""
from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _main(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the main configuration: unsmoothed cases at the first tolerance."""
    d = df[~df.case.str.contains("smooth")]
    return d[d.eps_rel == d.eps_rel.min()] if len(d) else d


def fig_messages(summ: pd.DataFrame, out: str, alpha: float = 0.05) -> str:
    """Messages by method for each case, at one target violation rate."""
    s = _main(summ)
    s = s[s.alpha == alpha]
    methods = [("automon", "AutoMon (safe zones)"),
               ("periodic_hindsight", "Periodic, best in hindsight"),
               ("filter_hindsight", "Send-on-delta, best in hindsight"),
               ("filter_guaranteed", "Send-on-delta, Lipschitz bound"),
               ("filter_calibrated", f"Send-on-delta, calibrated to {alpha:.0%}")]
    notes = {"filter_guaranteed": "no bound exists"}
    n = len(s)
    ncol = min(2, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol + 2.0, 2.5 * nrow + 1.0),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (_, row) in zip(axes.ravel(), s.iterrows()):
        vals = [100 * row[m] for m, _ in methods]
        y = np.arange(len(methods))[::-1]
        colors = [ORANGE if m == "automon" else AQUA if m == "filter_calibrated" else BLUE
                  for m, _ in methods]
        ax.barh(y, [0 if np.isnan(v) else v for v in vals], height=0.55, color=colors)
        top = max(110.0, np.nanmax(vals) * 1.15)
        for yi, v, (m, _) in zip(y, vals, methods):
            txt = notes.get(m, "none found") if np.isnan(v) else (
                f"{v:.0f}%" if v >= 10 else f"{v:.1f}%")
            if m == "automon" and not np.isnan(v):
                txt += "" if row.automon_source == "original" else "  (re-implemented)"
            ax.text(top * 0.012 if np.isnan(v) else v + top * 0.012, yi, txt, va="center",
                    fontsize=8, color=INK2 if np.isnan(v) else INK)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in methods], fontsize=8.5, color=INK)
        ax.set_xlim(0, top)
        ax.set_title(row.case, fontsize=9.5, color=INK, loc="left")
        _style(ax, "x")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.supxlabel("Messages, as a share of every node reporting every round (audits included). "
                  "Hindsight rows: cheapest setting within the target on the evaluated rounds.",
                  fontsize=8, color=INK2)
    fig.suptitle("Send-on-delta needs fewer messages than safe zones on both datasets",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out, "fig_messages.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_guarantee(runs: pd.DataFrame, out: str) -> str:
    """Realised violation rate of every calibrated run against its target."""
    d = runs[runs.method == "filter-calibrated"].copy()
    d = d[d.eps_rel == d.eps_rel.min()]
    ps = sorted(d.p.unique())
    fig, axes = plt.subplots(1, len(ps), figsize=(3.6 * len(ps) + 0.8, 3.4), sharey=True,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    alphas = sorted(d.alpha.unique())
    rng = np.random.default_rng(0)
    for ax, p in zip(axes[0], ps):
        g = d[d.p == p]
        for i, a in enumerate(alphas):
            v = 100 * g[g.alpha == a].viol_rate.to_numpy()
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=16, color=BLUE, alpha=0.7,
                       linewidths=0)
            ax.plot([i - 0.3, i + 0.3], [100 * a] * 2, color=ORANGE, lw=1.6)
            ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=INK, lw=1.2)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:.0%}" for a in alphas])
        ax.set_title(f"audits at {p:.1%} of rounds", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("target", fontsize=8.5, color=INK2)
        _style(ax, "y")
    axes[0][0].set_ylabel("rounds outside tolerance, %", fontsize=8.5, color=INK2)
    fig.suptitle("Realised violation rates track the target on average, with a wide spread "
                 "at the lowest audit rate", x=0.01, ha="left", fontsize=11, color=INK,
                 fontweight="bold")
    fig.text(0.01, 0.01, "Dots: one run (case x seed).  Orange: target.  Black: mean.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    path = os.path.join(out, "fig_guarantee.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def make_figures(summ: pd.DataFrame, runs: pd.DataFrame, out: str) -> List[str]:
    return [fig_messages(summ, out), fig_guarantee(runs, out)]


def fig_highdim(summ: pd.DataFrame, out: str, alpha: float = 0.05) -> str:
    """AutoMon's own benchmarks: messages by method."""
    from .highdim import LABELS
    s = summ[summ.alpha == alpha]
    s = s[~s.case.str.contains("_first")].set_index("case")
    for c in [c for c in s.index if c.endswith("_long")]:
        if c[:-5] in s.index:              # published stream, plus calibration from the long one
            for col in ("filter_calibrated", "filter_calibrated_viol"):
                s.loc[c[:-5], col] = s.loc[c, col]
            s = s.drop(index=c)
    order = {c: i for i, c in enumerate(LABELS)}
    s = s.assign(_o=[order.get(c, 99) for c in s.index]).sort_values("_o").reset_index()
    methods = [("safe_zone", "AutoMon (safe zones)"),
               ("periodic_zero_viol", "Periodic, best period in hindsight"),
               ("filter_zero_viol", "Send-on-delta, best tolerance in hindsight"),
               ("filter_guaranteed", "Send-on-delta, Lipschitz bound"),
               ("filter_calibrated", f"Send-on-delta, calibrated to {alpha:.0%}")]
    notes = {"filter_guaranteed": "no bound exists", "filter_calibrated": "stream too short",
             "safe_zone": "not run"}
    n = len(s)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol + 2.2, 2.7 * nrow + 1.3),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (_, row) in zip(axes.ravel(), s.iterrows()):
        vals = [100 * row[m] for m, _ in methods]
        y = np.arange(len(methods))[::-1]
        colors = [ORANGE if m == "safe_zone" else AQUA if m == "filter_calibrated" else BLUE
                  for m, _ in methods]
        ax.barh(y, [0 if np.isnan(v) else v for v in vals], height=0.55, color=colors)
        for yi, v, (m, _) in zip(y, vals, methods):
            txt = notes.get(m, "none found") if np.isnan(v) else (
                f"{v:.0f}%" if v >= 10 else f"{v:.1f}%")
            if m == "safe_zone" and row.get("safe_zone_source") != "original" \
                    and not np.isnan(v):
                txt += "  (re-implemented)"
            if m == "filter_calibrated" and not np.isnan(v):
                txt += f"  (off tolerance {100 * row.filter_calibrated_viol:.1f}% of the time)"
            ax.text(1.5 if np.isnan(v) else v + 1.5, yi, txt, va="center", fontsize=8,
                    color=INK2 if np.isnan(v) else INK)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for _, lab in methods], fontsize=8.5, color=INK)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100%"])
        label = LABELS.get(row.case, row.case)
        ax.set_title(f"{label}  (d = {int(row.d)})", fontsize=9.5, color=INK, loc="left")
        _style(ax, "x")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.supxlabel("Messages, as a share of every node update being reported.  Hindsight rows: "
                  "cheapest setting with no violation on the evaluated stream.  AutoMon: original "
                  "code unless marked.\nOnly the calibrated row is allowed to exceed the "
                  "tolerance.  For the four synthetic benchmarks it is run on 20 joined draws of "
                  "the generator; the published streams are too short.",
                  fontsize=8, color=INK2)
    fig.suptitle("On AutoMon's own benchmarks, tuned or calibrated send-on-delta needs fewer "
                 "messages than safe zones",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(out, "fig_highdim.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_calibration(summ: pd.DataFrame, out: str, alpha: float = 0.01,
                    title: str = "") -> str:
    """Calibration variants at one target: realised rate per benchmark, and message cost."""
    from .calib import VARIANTS
    s = summ[np.isclose(summ.alpha, alpha)]
    variants = [v for v in VARIANTS if v in set(s.variant)]
    if not title:        # cheapest variant that keeps every benchmark in target, at every target
        within = (summ.viol <= summ.alpha).groupby(summ.variant).all()
        ok = [(s[s.variant == v].msgs_vs_indicator.mean(), v) for v in variants if within[v]]
        if ok:
            cost, v = min(ok)
            title = (f"The '{v}' loss keeps every benchmark within the {alpha:.0%} target "
                     f"for {cost - 1:.0%} more messages")
    y = np.arange(len(variants))[::-1]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 0.52 * len(variants) + 2.1),
                                 gridspec_kw=dict(width_ratios=[1.55, 1]), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    top = 0.0
    for yi, v in zip(y, variants):
        d = s[s.variant == v]
        ax.scatter(100 * d.viol, np.full(len(d), yi), s=34, color=BLUE, alpha=0.75,
                   linewidths=0, zorder=3)
        ax.scatter([100 * d.viol.mean()], [yi], marker="|", s=260, color=INK, zorder=4)
        top = max(top, 100 * d.viol.max())
        cost = d.msgs_vs_indicator.mean()
        bx.barh(yi, cost, height=0.5, color=BLUE)
        bx.text(cost + 0.03, yi, f"x{cost:.2f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(100 * alpha, color=ORANGE, lw=1.4, zorder=2)
    ax.text(100 * alpha, len(variants) - 0.45, f" target {alpha:.0%}", color=ORANGE,
            fontsize=8.5, va="bottom", ha="left")
    hi = max(top, 100 * alpha) * 1.12
    ax.set_xlim(-0.02 * hi, hi)
    ax.set_ylim(-0.6, len(variants) - 0.2)
    ax.set_yticks(y)
    ax.set_yticklabels(variants, fontsize=9, color=INK)
    ax.set_xlabel("Share of steps outside the tolerance, %  (dot = one benchmark, "
                  "mean over seeds;  tick = mean of the benchmarks)", fontsize=8.5, color=INK2)
    bx.axvline(1.0, color=INK2, lw=0.8)
    bx.set_xlim(0, max(1.6, s.groupby("variant").msgs_vs_indicator.mean().max() * 1.2))
    bx.set_xlabel("Messages, relative to the indicator rule\n(mean of the benchmarks)",
                  fontsize=8.5, color=INK2)
    for a in (ax, bx):
        _style(a, "x")
    fig.suptitle(title or f"Calibration variants at a {alpha:.0%} target", x=0.01, ha="left",
                 fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out, "fig_calibration.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def _case_label(case: str) -> str:
    from .highdim import LABELS
    if case in LABELS:
        return LABELS[case].replace(", long", "")
    ds, fn = case.split("/")
    return f"{ds.capitalize()}, {'network' if fn == 'mlp' else fn}"


def fig_tolerance(tol: pd.DataFrame, out: str) -> str:
    """Messages against the tolerance, per case."""
    cases = list(dict.fromkeys(tol.case))
    ncol = 4
    nrow = int(np.ceil(len(cases) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.7 * nrow + 1.1), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    series = [("safe_zones", "Safe zones (AutoMon)", ORANGE, "-"),
              ("periodic_zero_viol", "Periodic, best in hindsight", INK2, ":"),
              ("send_on_delta_zero_viol", "Send-on-delta, best in hindsight", BLUE, "--"),
              ("calibrated_0.01", "Send-on-delta, calibrated to 1%", AQUA, "-")]
    for ax, case in zip(axes.ravel(), cases):
        g = tol[tol.case == case].sort_values("tol_mult")
        for col, lab, c, ls in series:
            v = 100 * g[col]
            if v.notna().any():
                ax.plot(g.tol_mult, v, ls, color=c, lw=1.8, marker="o", ms=3.5, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(g.tol_mult)
        ax.set_xticklabels([f"x{m:g}" for m in g.tol_mult])
        ax.minorticks_off()
        yt = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100) if
              0.6 * np.nanmin(100 * g[[s[0] for s in series]].to_numpy()) <= t <=
              1.6 * np.nanmax(100 * g[[s[0] for s in series]].to_numpy())]
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{t:g}%" for t in yt])
        ax.set_title(f"{_case_label(case)}  (d = {int(g.d.iloc[0])})", fontsize=9,
                     color=INK, loc="left")
        _style(ax, "both")
    for ax in axes.ravel()[len(cases):]:
        ax.set_visible(False)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.01, 0.955))
    fig.supxlabel("Tolerance, as a multiple of the default (0.1 of the monitored value's "
                  "spread for Beijing and Intel; AutoMon's configured value for its benchmarks)."
                  "  Messages as a share of full reporting, log scale.", fontsize=8, color=INK2)
    fig.suptitle("Send-on-delta stays cheaper than safe zones at every tolerance tested",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(out, "fig_tolerance.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_ablation(abl: pd.DataFrame, out: str, alpha: float = 0.01) -> str:
    """(a) what the trigger contributes; (b) the audit rate."""
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(13.2, 4.4),
                                     gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    fig.patch.set_facecolor(SURFACE)
    t = abl[(abl.study == "trigger") & np.isclose(abl.alpha, alpha)]
    cases = list(dict.fromkeys(t.case))
    y = np.arange(len(cases))[::-1]
    for yi, c in zip(y, cases):
        g = t[t.case == c].set_index("method")
        a, b = 100 * g.loc["periodic, calibrated", "msgs"], 100 * g.loc["send-on-delta, calibrated", "msgs"]
        ax.plot([b, a], [yi, yi], color=GRID, lw=2.2, zorder=1)
        ax.scatter([a], [yi], color=INK2, s=34, zorder=2,
                   label="Periodic, calibrated" if yi == y[0] else None)
        ax.scatter([b], [yi], color=AQUA, s=34, zorder=3,
                   label="Send-on-delta, calibrated" if yi == y[0] else None)
        ax.text(max(a, b) * 1.12, yi, f"x{a / b:.1f}", va="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([_case_label(c) for c in cases], fontsize=8.5, color=INK)
    ax.set_xlabel("Messages, % of full reporting (log)", fontsize=8.5, color=INK2)
    ax.set_title(f"(a) Same calibration, different trigger (target {alpha:.0%})",
                 fontsize=9.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlim(right=ax.get_xlim()[1] * 2.2)
    ticks = [t for t in (2, 5, 10, 20, 50, 100) if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks])
    ax.minorticks_off()
    _style(ax, "x")
    r = abl[(abl.study == "audit rate") & np.isclose(abl.alpha, alpha)]
    ref = r[np.isclose(r.p, 0.01)].set_index("case").msgs
    for c, g in r.groupby("case", sort=False):
        g = g.sort_values("p")
        bx.plot(100 * g.p, g.msgs / ref[c], color=BLUE, alpha=0.35, lw=1)
        cx.plot(100 * g.p, 100 * g.viol, color=BLUE, alpha=0.35, lw=1)
    m = r.assign(rel=r.msgs / r.case.map(ref)).groupby("p")[["rel", "viol"]].mean()
    bx.plot(100 * m.index, m.rel, color=INK, lw=2, marker="o", ms=4, label="mean of cases")
    cx.plot(100 * m.index, 100 * m.viol, color=INK, lw=2, marker="o", ms=4)
    cx.axhline(100 * alpha, color=ORANGE, lw=1.4)
    cx.text(100 * m.index.max(), 100 * alpha, " target", color=ORANGE, fontsize=8, va="bottom",
            ha="right")
    for z in (bx, cx):
        z.set_xscale("log")
        z.set_xticks(100 * m.index)
        z.set_xticklabels([f"{100 * p:g}%" for p in m.index])
        z.minorticks_off()
        z.set_xlabel("Audit rate, share of rounds", fontsize=8.5, color=INK2)
        _style(z, "y")
    bx.set_title("(b) Messages, relative to 1% audits", fontsize=9.5, color=INK, loc="left")
    cx.set_title("(c) Steps outside tolerance, %", fontsize=9.5, color=INK, loc="left")
    bx.legend(frameon=False, fontsize=8)
    fig.suptitle("The send-on-delta trigger, not the calibration alone, carries the saving",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out, "fig_ablation.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_energy(energy: pd.DataFrame, out: str, alpha: float = 0.01) -> str:
    """Battery life per node under the LoRaWAN model."""
    e = energy[np.isclose(energy.alpha, alpha)]
    cases = list(dict.fromkeys(e.case))
    y = np.arange(len(cases))[::-1]
    fig, ax = plt.subplots(figsize=(9.6, 0.5 * len(cases) + 1.9))
    fig.patch.set_facecolor(SURFACE)
    h = 0.26
    for off, col, c, lab in ((h, "full_years", INK2, "Full reporting"),
                             (0, "safe_zone_years", ORANGE, "Safe zones (AutoMon)"),
                             (-h, "calibrated_years", AQUA, f"Send-on-delta, calibrated to {alpha:.0%}")):
        ax.barh(y + off, e[col], height=h * 0.92, color=c, label=lab)
    for yi, row in zip(y, e.itertuples()):
        ax.text(row.calibrated_years + 0.06, yi - h, f"{row.calibrated_years:.1f} y", va="center",
                fontsize=7.5, color=INK)
        if np.isfinite(row.safe_zone_years):
            ax.text(row.safe_zone_years + 0.06, yi, f"{row.safe_zone_years:.1f} y", va="center",
                    fontsize=7.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{_case_label(c)}  ({'1/h' if r == 1 else '1/min'})"
                        for c, r in zip(cases, e.updates_per_hour)], fontsize=8.5, color=INK)
    ax.set_xlabel("Battery life per node, years (2400 mAh, LoRaWAN class A, SF7; upper bound)",
                  fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, "x")
    fig.suptitle("At one reading per minute, calibrated send-on-delta multiplies battery life",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out, "fig_energy.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_alerts(summ: pd.DataFrame, out: str) -> str:
    """Messages and alert errors by method and tolerance."""
    s = summ[summ.method != "full reporting"].copy()
    tols = sorted(s.tolerance_ug.unique())
    methods = [m for m in dict.fromkeys(s.method)]
    colors = {m: (ORANGE if m.startswith("safe") else AQUA if "1%" in m else BLUE) for m in methods}
    short = {m: ("Safe zones (AutoMon)" if m.startswith("safe") else
                 m.replace("send-on-delta, calibrated to", "Send-on-delta,")) for m in methods}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    fig.patch.set_facecolor(SURFACE)
    w = 0.8 / len(methods)
    x = np.arange(len(tols))
    for i, m in enumerate(methods):
        g = s[(s.method == m) & (s.rule == "plain")].set_index("tolerance_ug").reindex(tols)
        gg = s[(s.method == m) & (s.rule == "guard band")].set_index("tolerance_ug").reindex(tols)
        xs = x + (i - (len(methods) - 1) / 2) * w
        axes[0].bar(xs, 100 * g.msg_ratio, w * 0.9, color=colors[m], label=short[m])
        axes[1].bar(xs, g.missed_hours, w * 0.9, color=colors[m])
        axes[1].scatter(xs, gg.missed_hours, marker="_", s=180, color=INK, zorder=3,
                        label="with guard band" if i == 0 else None)
        axes[2].bar(xs, g.false_hours, w * 0.9, color=colors[m])
        axes[2].scatter(xs, gg.false_hours, marker="_", s=180, color=INK, zorder=3)
    axes[0].axhline(100, color=INK2, lw=1, ls=":")
    axes[0].text(x[-1] + 0.45, 101, "full reporting", ha="right", va="bottom", fontsize=8,
                 color=INK2)
    titles = ["(a) Messages, % of full reporting", "(b) Alert hours missed",
              "(c) False alert hours"]
    for ax, t in zip(axes, titles):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t_:g} ug/m3" for t_ in tols])
        ax.set_xlabel("Tolerance on the forecast", fontsize=8.5, color=INK2)
        ax.set_title(t, fontsize=9.5, color=INK, loc="left")
        _style(ax, "y")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    n = int(summ.alert_hours.iloc[0])
    fig.suptitle(f"Beijing heavy-pollution alerts, Oct 2014 - Feb 2017 ({n} alert hours under "
                 "full reporting)", x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out, "fig_alerts.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_episode(tr: pd.DataFrame, out: str, start: str = "2015-11-26",
                end: str = "2016-01-05", threshold: float = 150.0) -> str:
    """Timeline of one winter: the coordinator's estimate under each method."""
    d = tr[(tr.time >= start) & (tr.time < end)]
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    fig.patch.set_facecolor(SURFACE)
    alert = d.full >= threshold
    ax.fill_between(d.time, 0, 1, where=alert, transform=ax.get_xaxis_transform(),
                    color=ORANGE, alpha=0.10, lw=0, label="alert under full reporting")
    ax.plot(d.time, d.actual, color=INK2, lw=0.8, ls=":", label="realised 24-h mean, 3 h later")
    ax.plot(d.time, d.full, color=INK, lw=1.4, label="forecast, full reporting")
    ax.plot(d.time, d.sod, color=AQUA, lw=1.1, label="forecast, calibrated send-on-delta (1%)")
    ax.axhline(threshold, color=ORANGE, lw=1.3)
    ax.text(d.time.iloc[0], threshold, " alert threshold 150 ug/m3", color=ORANGE, fontsize=8,
            va="bottom")
    ax.set_ylabel("Regional 24-h mean PM2.5, ug/m3", fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    _style(ax, "y")
    ax.set_ylim(top=max(d[["full", "actual"]].max()) * 1.25)
    fig.suptitle("Winter 2015-16: the send-on-delta estimate tracks the full-reporting forecast "
                 "through Beijing's worst smog episodes", x=0.01, ha="left", fontsize=11,
                 color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out, "fig_episode.png")
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path
