#!/usr/bin/env python3
"""Stage 6: city-wide air-quality alerts on the Beijing network.

    python run_alerts.py --data data --out results_alerts

Writes runs.csv (one row per run), summary.csv (per tolerance and method),
skill.csv (how well the full-reporting forecast anticipates real pollution),
fig_alerts.png and fig_episode.png.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from monitor_pilot import alerts as AL  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_alerts")
    ap.add_argument("--tolerances", type=float, nargs="*", default=[2.5, 5.0, 10.0])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    case = AL.load(a.data)
    skill = AL.forecast_skill(case)
    pd.DataFrame([skill]).to_csv(os.path.join(a.out, "skill.csv"), index=False)
    print("forecast skill:", json.dumps({k: round(v, 3) for k, v in skill.items()}))
    rows = pd.DataFrame(AL.run_case(case, tuple(a.tolerances), log=lambda m: print(m, flush=True)))
    rows.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    summ = AL.summarise(rows)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))
    from monitor_pilot.plots import fig_alerts, fig_episode
    tr = AL.episode_traces(case)
    tr.to_csv(os.path.join(a.out, "traces.csv"), index=False)
    print("wrote", fig_alerts(summ, a.out), fig_episode(tr, a.out))


if __name__ == "__main__":
    main()
