# Calibrated send-on-delta

Code, data and results for the paper

> R. Al-Sayyed, *Energy-efficient monitoring of learned functions in IoT sensor
> networks: calibrated send-on-delta with a long-run error guarantee*
> (manuscript submitted to *Internet of Things*, Elsevier).

A coordinator tracks a function of the average of many sensors' readings (a
variance, a neural-network forecast, an anomaly score) within a tolerance,
without every sensor reporting every round. Each node reports only when its
own reading has moved by more than a tolerance (send-on-delta), and the
coordinator calibrates that tolerance online from rare random audits. The
repository compares this with geometric monitoring (AutoMon, SIGMOD 2022) on
two public sensor deployments and on AutoMon's own benchmarks, converts
messages to LoRaWAN battery life, and runs a Beijing air-quality alert case
study. Every number and table in the paper is produced by the scripts here
(`make_paper_tables.py` writes the paper's tables from the result files,
`make_paper_figures.py` draws its figures, and `fill_paper_numbers.py` fills
the numbers quoted in the text).

The work is organised in stages, each with its own script and results folder:

| Stage | Script | Results |
| --- | --- | --- |
| 2. Beijing and Intel deployments | `run_pilot.py`, `run_automon_baseline.py` | `results/` |
| 3. AutoMon's benchmarks | `run_automon_benchmarks.py`, `run_highdim.py` | `results_highdim/` |
| 4. Calibration rules at small targets | `run_calibration.py` | `results_calibration/` |
| 5. Tolerance sweep, ablations, battery life | `run_stage5.py` | `results_stage5/` |
| 6. Beijing heavy-pollution alerts | `run_alerts.py` | `results_alerts/` |
| 7. Revision: matched streams, calibrated safe zones, offline tuning, message directions, packet loss, network size | `run_revision.py` | `results_revision/` |

(Stage 1, an earlier single-sensor study, is not part of the paper.)

## Data

`data/beijing.npz` and `data/intel.npz` are compact arrays derived from two
public datasets by `prepare_data.py`; please cite the originals:

- Beijing Multi-Site Air Quality (UCI Machine Learning Repository, CC BY 4.0):
  S. Zhang, B. Guo, A. Dong, J. He, Z. Xu, S. X. Chen, Cautionary tales on
  air-quality improvement in Beijing, *Proc. R. Soc. A* 473 (2017) 20170457.
- Intel Berkeley Research Lab data: P. Bodik, W. Hong, C. Guestrin, S. Madden,
  M. Paskin, R. Thibaux (2004), http://db.csail.mit.edu/labdata/labdata.html.

AutoMon's benchmark streams are not shipped (about 500 MB); Stage 3 explains
how to rebuild them from the AutoMon repository (https://github.com/hsivan/automon).

## Run it

```
pip install -r requirements.txt
python prepare_data.py --beijing PRSA_DIR --intel INTEL_DATA_TXT --out data
python run_pilot.py --data data --out results          # four main cases, ~1 hour
python run_pilot.py --out results --cases intel/variance --fast   # one case, quick
python -m unittest discover -s tests
```

Useful options: `--eps-rel 0.1 0.3` (tolerances), `--cases intel-smooth/mlp`,
`--tag _x` (write `runs_x.csv` so several runs can go in parallel) and
`--merge` (combine all `runs*.csv` into `runs.csv`, summarise, plot).

The shipped `results/` combine three runs: the four main cases at tolerance
0.1, the same cases at 0.3 with `--fast`, and the two `intel-smooth` cases
with `--fast`.

Outputs in `results/`:

| File | Content |
| --- | --- |
| `summary.csv` | messages by method for each case, tolerance and target violation rate |
| `runs.csv` | every simulated run, one row each |
| `automon.csv` | the original AutoMon library on the same cases (optional, see below) |
| `fig_messages.png` | messages by method, per case |
| `fig_guarantee.png` | realised violation rate against the target |

## What is simulated

Each round every node gets a new local vector. The coordinator holds an
estimate of f(average vector). A round is a **violation** when the estimate is
off by more than the tolerance. Messages are counted one per node-coordinator
message, for every method alike (see the header of `monitor_pilot/protocols.py`).

**Cases** (`monitor_pilot/cases.py`)

| Dataset | Rounds, nodes | Local vector |
| --- | --- | --- |
| `beijing` | hourly, 12 stations | 24-hour mean of six standardised readings |
| `intel` | 1 minute, 37 motes | standardised temperature, humidity, log light |
| `intel-smooth` | same | 30-minute means |

| Function | Meaning | Why it is here |
| --- | --- | --- |
| `variance` | spread of one reading across sensors | concave: safe zones have a hard guarantee |
| `mlp` | small tanh network forecasting the regional mean | non-convex: safe zones have no formal guarantee |

The first 30% of rounds fit the standardisation, the next 10% are for tuning,
and all reported numbers come from the last 60%. The network (16 tanh units)
forecasts the regional mean of the first reading 3 hours ahead (Beijing) or
30 minutes ahead (Intel) and is fitted on the first 40%. The tolerance is a share (`--eps-rel`) of the monitored value's
spread before the evaluation period.

**Methods** (`monitor_pilot/protocols.py`)

| Name in outputs | What it does | Guarantee |
| --- | --- | --- |
| `central` | every node reports every round | exact |
| `periodic` | every node reports every P rounds | none |
| `filter-fixed` | a node reports when its vector moved more than delta since its last report | none |
| `filter-guaranteed` | same, delta = tolerance / Lipschitz constant (network only) | always within tolerance |
| `zone` | safe zones with slack and lazy sync, re-implementing AutoMon | hard for variance, none for the network |
| `zone-relaxed` | same with the tolerance inflated | none |
| `filter-calibrated`, `zone-calibrated` | tolerance scaled online from random audits | long-run violation share at most alpha |

**Calibration.** With probability p per round, on a schedule fixed in advance,
all nodes report. The coordinator sees the true error and multiplies the
tolerance by exp(eta x (alpha - violated)). Shrinking the tolerance far enough
makes every node report every round, so the audited violation share cannot
stay above alpha. An audit costs two messages per node. The starting tolerance
for the filter is the largest one that met alpha on the tuning rounds.

## The original AutoMon as a baseline

`run_automon_baseline.py` drives the original library on the same cases:

```
git clone https://github.com/hsivan/automon AUTOMON_DIR
pip install autograd pyzmq cvxpy
python run_automon_baseline.py --automon AUTOMON_DIR --out results
```

It is slow for the network cases. On the cases checked, the bundled
re-implementation and the original agree closely; compare `zone` in
`runs.csv` with `automon.csv`.

## Stage 2 result (`results/summary.csv`, `fig_messages.png`)

Tolerance 0.1 of the monitored value's spread, audits at 1% of rounds, target
5%, three seeds. Messages as a share of every node reporting every round.

| Case | AutoMon (original code) | Send-on-delta, Lipschitz bound | Send-on-delta, calibrated to 5% | Realised rate outside tolerance | Fewer messages than AutoMon |
| --- | --- | --- | --- | --- | --- |
| Beijing, variance | 126% | no bound exists | 32% | 4.6% | 75% |
| Beijing, network | 116% | 100% | 31% | 3.8% | 73% |
| Intel, variance | 49% | no bound exists | 5.2% | 4.3% | 89% |
| Intel, network | 37% | 66% | 6.1% | 4.9% | 83% |

- Across all ten case/tolerance combinations and targets, calibrated
  send-on-delta needed 65 to 89% fewer messages than safe zones.
- The safe-zone re-implementation matches the original AutoMon to within 0.2
  points on all four cases (`automon.csv` vs `zone` in `runs.csv`).
- **Changed from October:** on Beijing the safe zones now send more messages
  than full reporting (126%, 116%; 87% and 94% in October), and the Lipschitz
  filter no longer beats AutoMon on Intel (66% against 37%; 8.1% against 27%
  in October). The calibrated filter's own message levels are close to
  October's (32/31/5.2/6.1% against 26/31/4.6/4.7%).

## Stage 3: AutoMon's own benchmarks (higher dimension)

The cases above have 2 to 6 dimensions per node, which favours send-on-delta.
This stage repeats the comparison on the benchmarks AutoMon was published
with, 20 to 80 dimensions, where safe zones should be at their strongest.

```
git clone https://github.com/hsivan/automon AUTOMON_DIR      # add the data files its README lists
pip install autograd pyzmq cvxpy
python run_automon_benchmarks.py --automon AUTOMON_DIR --out data/bench            # streams + original AutoMon
python run_automon_benchmarks.py --automon AUTOMON_DIR --out data/bench --tiles 20 \\
       --cases inner_product quadratic MLP_40 MLP_80                               # long synthetic streams
python run_highdim.py --bench data/bench --out results_highdim
```

`data/bench` is not shipped (about 500 MB). The rebuilt streams reproduce the
function values published in the AutoMon repository to 1e-15 on the four
benchmarks that have them (inner product, quadratic, air quality, intrusion
detection).

| Benchmark | Function | d | Nodes | Stream evaluated |
| --- | --- | --- | --- | --- |
| `inner_product` | x . y | 40 | 10 | synthetic, 801 rounds |
| `quadratic` | x^T H x, indefinite | 40 | 10 | synthetic, 801 rounds |
| `MLP_40`, `MLP_80` | trained tanh network | 40, 80 | 10 | synthetic, 801 rounds |
| `air_quality` | KL divergence of two histograms | 20 | 12 | Beijing PM, 23,841 rounds |
| `intrusion_detection` | trained ReLU network, sigmoid output | 41 | 9 | KDD Cup 99, 242,237 single-node updates |

The first 20% of every stream is used for tuning only. Tolerances are the
ones in AutoMon's configuration files. `<name>_long` is 20 fresh draws of the
same synthetic generator joined end to end (16,016 evaluated rounds), because
1,000 rounds hold too few audits to calibrate anything.

### Result (`results_highdim/summary.csv`, `fig_highdim.png`)

Messages as a share of every node update being reported. No row except the
last was ever outside the tolerance.

| Benchmark | AutoMon | Periodic, hindsight | Send-on-delta, hindsight | Send-on-delta, Lipschitz | Send-on-delta, calibrated to 5% (realised) |
| --- | --- | --- | --- | --- | --- |
| Inner product, d=40 | 15% | 25% | 6.5% | no bound | 6.0% (5.5%) |
| Quadratic, d=40 | 66% | 100% | 16% | no bound | 5.3% (4.3%) |
| MLP-40 | 17% | 25% | 6.0% | 100% | 6.8% (6.9%) |
| MLP-80 | 26% (re-implemented) | 25% | 8.5% | 100% | 7.6% (4.4%) |
| KLD air quality, d=20 | 18% | 1.6% | 1.9% | no bound | 2.0% (1.0%) |
| DNN intrusion, d=41 | 29% (re-implemented) | 50% | 7.3% | 51% | 2.5% (4.9%) |

How to read it:

- **AutoMon** is the original code on the published stream. It did not finish
  in the time available on two benchmarks: about 50 s per full sync for the
  intrusion network (509 full syncs are needed) and still tuning after 40
  CPU-minutes on MLP-80. Those two rows come from the safe-zone protocol in
  this package (`zone-reimplemented`), run with AutoMon's configured
  neighbourhood size. Where both ran, the two agree within one point:
  15.1 / 15.7, 66.3 / 67.3, 17.2 / 17.0, 17.9 / 18.1, and 11.8 / 11.8 on the
  first 30,000 intrusion updates.
- **Hindsight** rows use the cheapest period or tolerance that gave no
  violation on the evaluated stream itself. They show what the mechanism can
  reach, not something a deployed system could pick in advance.
- **Lipschitz** is the only cheap rule here with a hard guarantee, and it does
  not survive the dimension: the bound is so loose that every node reports
  every round on the two tanh networks.
- **Calibrated** is the only row allowed to violate. For the four synthetic
  benchmarks it is run on the `_long` streams. Audits are 1% of rounds and
  their messages are counted.
- **Drift.** On the intrusion stream the tolerance tuned on the first 20% was
  outside the tolerance 15% of the time afterwards. Calibration brought that
  to 1.8% against a 1% target and 4.9% against 5%.

Limits specific to this stage:

- **Small targets are overshot.** With 160 to 270 audits per run, a 1% target
  gave 1.4 to 1.8% on five of the six benchmarks (worst single run 2.5%), and
  a 5% target gave 1.0 to 6.9% (worst run 9.5%). The guarantee is on the
  audited share and its finite-sample slack is several points at this length.
- **The comparison is not at equal error.** AutoMon never left the tolerance;
  the calibrated runs are allowed to, and the hindsight rows are tuned on the
  data they are scored on.
- **Joined synthetic streams** jump at each join. Send-on-delta reports
  through a jump; periodic reporting cannot, so periodic is scored on the
  published streams only.
- **Periodic reporting already beats safe zones on the air-quality
  benchmark** (1.6% against 18%), so that benchmark says little about
  send-on-delta in particular.
- **On an update-at-a-time stream** (intrusion detection) the audit rate is
  per round of updates, that is audit probability 1% / nodes per update.

## Stage 4: keeping small targets

Stage 3 left one weak spot: against a 1% target the calibrated runs realised
1.4 to 1.8%. The cause is in the rule itself. With the plain indicator loss
(1 if the audited error is over the tolerance, else 0) the long-run audited
violation share *equals* the target, so any start-up transient pushes a finite
run above it, and a violation is seen only once in a hundred audits.

```
python run_calibration.py --bench data/bench --out results_calibration
python run_calibration.py --pilot-data data --out results_calibration/heldout \\
       --cases beijing/variance beijing/mlp intel/variance intel/mlp
```

Three remedies were compared (`monitor_pilot/calib.py`), all with the same
form of guarantee:

| Remedy | What changes |
| --- | --- |
| near-miss loss | the loss rises linearly from 0 at `ramp` x tolerance to 1 at the tolerance, so near-misses move the tolerance before violations occur. It is never below the indicator, so a bound on its mean is a bound on the violation share. |
| front-loading | the same expected number of audits, four times denser in the first tenth of the stream |
| more audits | twice the audit rate |

Result on the six benchmarks, target 1%, audits at 1% of rounds, 10 seeds (5
for intrusion detection). `results_calibration/overall.csv` has all targets.

| Rule | Share of steps outside tolerance (mean of benchmarks) | Benchmarks within target | Runs within target | Messages vs indicator |
| --- | --- | --- | --- | --- |
| indicator (Stages 2 and 3) | 1.16% | 1 of 6 | 38% | x1.00 |
| indicator, 2x audits | 0.85% | 3 of 6 | 65% | x1.36 |
| indicator, front-loaded | 1.07% | 1 of 6 | 55% | x1.01 |
| near-miss 0.9 | 0.80% | 4 of 6 | 68% | x1.04 |
| **near-miss 0.75** | 0.32% | 6 of 6 | 95% | x1.13 |
| near-miss 0.5 | 0.06% | 6 of 6 | 100% | x1.45 |
| near-miss 0.75, front-loaded | 0.39% | 6 of 6 | 95% | x1.12 |
| near-miss 0.5, front-loaded | 0.09% | 6 of 6 | 100% | x1.35 |

- **Near-miss 0.75 is the recommended default.** It was within target on every
  benchmark at all three targets (1%, 2%, 5%) for 6 to 13% more messages.
  About 95% of individual runs were within target; the worst single run was
  1.5% against 1% and 6.0% against 5%.
- **Front-loading does not help** and doubling the audits helps a little at a
  third more messages.
- **Held-out check.** Ramp 0.75 was picked after seeing the benchmark results,
  so the study was repeated on the four Stage-2 cases, which played no part in
  that choice (`results_calibration/heldout/`). On the rebuilt Stage-2 data,
  near-miss 0.75 was within target in 11 of 12 case/target combinations (the
  exception: Beijing variance at 1%, realised 1.07%), for 5 to 17% more
  messages, with 50 to 100% of runs within target. Near-miss 0.5 was within
  target in all 12, with 90 to 100% of runs within target, for 16 to 30% more
  messages. The indicator rule was within target in 3 of the 12.
- **Against AutoMon** with near-miss 0.75 and a 1% target, messages are 52 to
  88% lower on the six benchmarks (7.2 / 15 / 7.8 / 8.1 / 3.1 / 3.6% against
  15 / 66 / 17 / 26 / 18 / 29%), with 0.0 to 0.8% of steps outside tolerance.

What this does and does not show:

- **The formal guarantee is unchanged**, not tighter: mean audited loss at most
  alpha + (u0 - u_min + eta) / (eta x audits), which is still several points
  at 150 to 270 audits. It held in all 2,280 runs. That realised rates sit
  well under the target is an empirical result on ten cases.
- **A correction to the rule.** Writing a test for this stage showed that
  clipping the log-tolerance at its floor broke the bound when the starting
  tolerance was far too loose. The clip is removed and the floor now means
  "every change is reported". No run in any stage reached the floor.

## Stage 5: tolerance, trigger, audit rate and battery life

```
python run_stage5.py --part tolerance --data data --bench data/bench --out results_stage5
python run_stage5.py --part ablation  --data data --bench data/bench --out results_stage5
python run_stage5.py --part report    --out results_stage5
```

All calibrated runs here use the Stage 4 rule (near-miss loss from 75% of the
tolerance), audits at 1% of rounds unless stated, three seeds.

**Tolerance sweep** (`tolerance_summary.csv`, `fig_tolerance.png`). Every
method at 0.5 to 5 times the Stage 2 tolerance, and 0.5 to 2 times AutoMon's
configured tolerance on four of its benchmarks.

- Calibrated send-on-delta needed 58 to 86% fewer messages than safe zones at
  a 1% target, and 65 to 89% fewer at 5%, at every tolerance on the six cases
  where it ran. The mean share of steps outside tolerance was at most 1.10%
  against 1% (Beijing variance at x5) and 4.4% against 5%.
- Send-on-delta with the best tolerance in hindsight needed 5 to 59% of the
  safe zones' messages on all eight cases and all tolerances.
- On Beijing, safe zones cost more than full reporting up to about twice the
  default tolerance, and less beyond it, so the Stage 2 result does not hinge
  on a tolerance where safe zones fail.

**Trigger ablation** (`ablation_summary.csv`, `fig_ablation.png` a). The same
calibration applied to a reporting period instead of a send-on-delta
tolerance. Calibrated periodic reporting needed 1.2 to 5.2 times the messages
at a 1% target (1.1 to 3.1 times at 5%), and it missed its target in 8 of 20
case/target combinations, against 0 of 20 for send-on-delta. The saving comes
from the trigger, not from calibration alone. The exception is the
air-quality benchmark (x1.2), where periodic reporting was already competitive.

**Audit rate** (`fig_ablation.png` b, c). Relative to audits at 1% of rounds,
0.25% audits cut messages by 15% and 4% audits raised them by 67% (1% target).
All ten cases stayed within a 1% target from 1% audits upward; at 0.25%, two
of ten exceeded it (worst 2.0%). One percent is the recommended default.

**Battery life** (`energy.csv`, `fig_energy.png`). Messages are converted with
the LoRaWAN class-A model of Casals et al. (Sensors 17(10):2364, 2017):
measured state currents from its Table 5, SF7 / 125 kHz, time on air from the
Semtech formula for a 13-byte header plus 2 bytes per vector coordinate,
45 uA sleep, 2400 mAh. Every message, up- or downlink, is charged as one
uplink, and sensing is left out, so all lives are upper bounds. Node updates
are one per hour for the Beijing data (including the air-quality benchmark)
and one per minute for Intel and, nominally, for AutoMon's other benchmarks.

| Case | Updates | Full reporting | Safe zones | Calibrated to 1% | Gain over safe zones |
| --- | --- | --- | --- | --- | --- |
| Beijing, variance | 1/h | 4.2 y | 3.9 y | 5.1 y | x1.3 |
| Beijing, network | 1/h | 4.2 y | 4.0 y | 5.1 y | x1.3 |
| Intel, variance | 1/min | 0.22 y | 0.43 y | 2.0 y | x4.7 |
| Intel, network | 1/min | 0.22 y | 0.56 y | 1.8 y | x3.2 |
| Inner product | 1/min | 0.20 y | 1.1 y | 1.9 y | x1.7 |
| Quadratic form | 1/min | 0.20 y | 0.29 y | 1.0 y | x3.4 |
| MLP-40 | 1/min | 0.20 y | 0.98 y | 1.8 y | x1.8 |
| MLP-80 | 1/min | 0.18 y | 0.62 y | 1.6 y | x2.5 |
| KLD air quality | 1/h | 4.1 y | 5.6 y | 6.0 y | x1.07 |
| DNN intrusion detection | 1/min | 0.20 y | 0.61 y | 3.0 y | x4.9 |

At one reading per hour the radio is a small part of the budget (sleep alone
caps life at 6.1 years), so the gain is modest; at one per minute it is 1.7 to
4.9 times.

## Stage 6: city-wide air-quality alerts (sustainability case)

```
python run_alerts.py --data data --out results_alerts
```

The coordinator runs the Stage-2 forecasting network on the average of the 12
Beijing stations and issues a heavy-pollution alert when the forecast regional
24-hour mean PM2.5, three hours ahead, reaches 150 ug/m3 (AQI 200 under China's
HJ 633-2012). The reference is the same rule with every station reporting
every hour. Evaluation: October 2014 to February 2017, 21,039 hours, 66 alert
episodes and 2,831 alert hours under full reporting. Three seeds per
calibrated method; audits at 1% of hours.

The forecast is useful in its own right: under full reporting it caught 95% of
the hours that really reached the threshold, with 7% false alarms and a mean
absolute error of 6.3 ug/m3 (`skill.csv`). So its own error is larger than any
of the tolerances below.

At a tolerance of 5 ug/m3 on the forecast (`summary.csv`, `fig_alerts.png`):

| Method | Messages | Battery (1/h) | Alert hours missed | False alert hours | Episodes missed | Mean onset delay |
| --- | --- | --- | --- | --- | --- | --- |
| Full reporting | 100% | 4.2 y | 0 | 0 | 0 of 66 | 0 h |
| Safe zones (AutoMon) | 128% | 3.9 y | 13 | 4 | 0 | 0.2 h |
| Send-on-delta, calibrated to 1% | 48% | 5.0 y | 34 | 8 | 1 (2 h long) | 0.5 h |
| Same, with guard band | 48% | 5.0 y | 0 | 128 | 0 | 0 h |

- **Guard band.** Alerting once the estimate is within one tolerance of the
  threshold costs no messages. It removed every missed alert hour, at the
  price of 128 extra alert hours (4.5% of the alert time) and 5 short false
  episodes.
- **Tolerances 2.5 and 10 ug/m3** behave the same way. At 10 ug/m3
  send-on-delta calibrated to 1% needs 29% of the messages (safe zones 92%)
  and the guard band again leaves no missed hours.
- **Battery** gains are modest at one reading per hour (5.0 against 3.9
  years), as in Stage 5. Per-minute sensing, as in the Intel data, is where the
  method pays most.
- `fig_episode.png` shows winter 2015-16, including the December 2015 red
  alerts: the send-on-delta estimate is visually indistinguishable from full
  reporting.


## Stage 7: revision experiments

```
python run_revision.py --part zones-plain --bench data/bench --out results_revision
python run_revision.py --part zones       --bench data/bench --out results_revision
python run_revision.py --part sod         --bench data/bench --out results_revision
python run_revision.py --part retrain     --out results_revision
python run_revision.py --part loss        --bench data/bench --out results_revision
python run_revision.py --part scale       --out results_revision
python make_paper_tables.py paper_tables
python make_paper_figures.py paper_tables paper_figures
```

- **Message directions.** `protocols.py` now records uplinks, downlinks, and
  downlinks to nodes that did not transmit in the same round ("unsolicited",
  which a LoRaWAN class-A node cannot receive without an extra uplink), per
  node. Totals are unchanged from earlier stages (checked run by run).
- **Matched streams.** Safe zones are re-run on the joined synthetic streams,
  so AutoMon and calibrated send-on-delta are compared on the same data.
- **Calibrated safe zones** use the same audits and near-miss rule.
- **Offline-tuned send-on-delta** keeps the tolerance tuned on the tuning
  period, with no audits.
- **Packet loss.** `run_filter(..., loss=q, loss_model="bernoulli"|"ge",
  confirmed=...)`: unconfirmed reports can be lost silently; confirmed reports
  are retransmitted until acknowledged; audit reports are always confirmed.
- **Network size.** `revision.scaled_intel` resamples the Intel deployment to
  10-1000 nodes (extra nodes replay a mote shifted by 1-30 minutes).
- **Energy.** `ablation.battery_years(..., idle_ma=...)` gives the
  sensitivity to the current drawn while waiting for the receive windows.

## Limits of this pilot

- **Two datasets, two functions.** Enough for a go/no-go, not for a paper.
- **Messages, not joules.** No radio or energy model, and no message sizes.
- **Short runs for small targets.** A few hundred audits cannot pin down a 1%
  rate. The guarantee is long-run; its finite-sample slack is large here.
- **Different kinds of guarantee are compared.** Safe zones bound every round
  (for functions they cover); calibration bounds the long-run share.
- **Audits assume all nodes can report at agreed times** and that the
  coordinator can send each node a new tolerance.
- **Hindsight rows** (`periodic`, `filter-fixed` picked after the fact) show
  what tuning could reach; no deployed system can pick them in advance.

## Data

`data/*.npz` are compact arrays built by `prepare_data.py` from two public
datasets: Beijing Multi-Site Air Quality (UCI Machine Learning Repository) and
the Intel Berkeley Research Lab sensor data. For a paper, download the
originals, cite them, and rebuild with `prepare_data.py`.

## Layout

```
run_pilot.py                 command-line entry point
run_automon_baseline.py      optional: original AutoMon on the same cases
prepare_data.py              raw public files -> data/*.npz
monitor_pilot/cases.py       datasets, functions, splits
monitor_pilot/functions.py   variance and the tanh network
monitor_pilot/protocols.py   all protocols and the calibration rule
monitor_pilot/experiment.py  sweeps and the summary table
monitor_pilot/plots.py       figures
tests/test_pilot.py          sanity checks
run_automon_benchmarks.py    stage 3: rebuild AutoMon's benchmark streams, run the original on them
run_highdim.py               stage 3: all other methods on those streams
monitor_pilot/automon_glue.py  helpers to drive the original AutoMon code
monitor_pilot/bench.py       stage 3: the benchmark functions in NumPy
monitor_pilot/highdim.py     stage 3: loading, sweeps, summary
tests/test_highdim.py        stage 3: checks
results_highdim/             stage 3: runs.csv, summary.csv, fig_highdim.png
run_calibration.py           stage 4: calibration variants at small targets
monitor_pilot/calib.py       stage 4: variants, study, summary
tests/test_calibration.py    stage 4: checks
results_calibration/         stage 4: runs, summary, overall, figure, heldout/
run_stage5.py                stage 5: tolerance sweep, ablations, battery life
monitor_pilot/ablation.py    stage 5: LoRaWAN energy model, sweeps, ablations
tests/test_stage5.py         stage 5: checks
results_stage5/              stage 5: runs, summaries, energy.csv, three figures
run_alerts.py                stage 6: Beijing heavy-pollution alerts
monitor_pilot/alerts.py      stage 6: alert rule, decision metrics, traces
tests/test_alerts.py         stage 6: checks
results_alerts/              stage 6: runs, summary, skill, traces, two figures
run_revision.py              stage 7: experiments requested in review
monitor_pilot/revision.py    stage 7: safe zones on matched streams, calibrated safe zones,
                             offline-tuned send-on-delta, eta/u_max sensitivity, packet loss,
                             network size, networks fitted on the first 30% only
tests/test_revision.py       stage 7: checks
results_revision/            stage 7: zones.csv (and worker folders z2-z4), sod.csv, loss.csv,
                             scale.csv, retrain.csv
make_paper_figures.py        all figures of the paper, without in-figure titles
fill_paper_numbers.py        numbers quoted in the paper's text, from the result files
```
