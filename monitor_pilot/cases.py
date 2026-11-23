"""Datasets, monitored functions and splits for the Stage-2 cases.

Rounds are split in time:  the first 30% fit the standardisation (and, with
the next 10%, the network);  30-40% are for tuning;  the last 60% give every
reported number.  The tolerance is ``eps_rel`` times the spread of the
monitored value over the first 40% of rounds.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .functions import TanhMLP, Variance

TRAIN, TUNE = 0.30, 0.10
# name -> (file, trailing window in rounds, forecast horizon of the network in rounds)
DATASETS = {"beijing": ("beijing", 24, 3), "intel": ("intel", 1, 30),
            "intel-smooth": ("intel", 30, 30)}
STEPS = {"beijing": "1 hour", "intel": "1 minute"}
FUNCTIONS = ("variance", "mlp")
ALL_CASES = [f"{d}/{f}" for d in ("beijing", "intel") for f in FUNCTIONS]
SMOOTH_CASES = [f"intel-smooth/{f}" for f in FUNCTIONS]


@dataclass
class Case:
    name: str
    func: object
    X_tune: np.ndarray      # local vectors (rounds, nodes, d) used for tuning only
    X_eval: np.ndarray      # local vectors used for every reported number
    eps: float              # error tolerance
    f_std: float            # spread of the monitored value before evaluation
    step: str

    @property
    def nodes(self) -> int:
        return self.X_eval.shape[1]


def _trailing_mean(Z: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return Z.astype(float)
    c = np.cumsum(np.concatenate([np.zeros((1,) + Z.shape[1:]), Z]), axis=0)
    T = len(Z)
    lo = np.maximum(np.arange(1, T + 1) - w, 0)
    n = (np.arange(1, T + 1) - lo)[:, None, None]
    return (c[1:] - c[lo]) / n


def readings(data_dir: str, dataset: str) -> np.ndarray:
    """Standardised, windowed readings (T, nodes, features)."""
    file, w, _ = DATASETS[dataset]
    Z = np.load(os.path.join(data_dir, f"{file}.npz"))["X"].astype(float)
    Z = _trailing_mean(Z, w)
    a = int(TRAIN * len(Z))
    mu = Z[:a].mean(axis=(0, 1))
    sd = Z[:a].std(axis=(0, 1))
    return (Z - mu) / sd


def build_case(data_dir: str, dataset: str, function: str, eps_rel: float = 0.1) -> Case:
    Z = readings(data_dir, dataset)
    T = len(Z)
    a, b = int(TRAIN * T), int((TRAIN + TUNE) * T)
    if function == "variance":
        f = Variance()
        X = f.local(Z[..., :1])
        f.scale = X[:a].reshape(-1, 2).std(axis=0)
    elif function == "mlp":
        h = DATASETS[dataset][2]
        m = Z.mean(axis=1)
        f = TanhMLP(Z.shape[2], seed=0).fit(m[:b - h], m[h:b, 0], seed=0)
        X = Z
    else:
        raise ValueError(function)
    F = f.f(X[:b].mean(axis=1))
    f_std = float(F.std())
    return Case(f"{dataset}/{function}", f, X[a:b], X[b:], eps_rel * f_std, f_std,
                STEPS[DATASETS[dataset][0]])
