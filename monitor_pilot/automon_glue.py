"""Helpers for driving the original AutoMon code (github.com/hsivan/automon).

AutoMon was written for NumPy 1.x; ``prepare`` restores the few names it
uses that NumPy 2 removed, selects autograd for differentiation, and puts the
clone on the import path.  Nothing here changes AutoMon's algorithm.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as onp


def prepare(automon_dir: str) -> None:
    for name, val in (("alltrue", onp.all), ("sometrue", onp.any), ("product", onp.prod),
                      ("float", float), ("int", int), ("bool", bool)):
        if not hasattr(onp, name):
            setattr(onp, name, val)
    os.environ["AUTO_DIFFERENTIATION_TOOL"] = "AutoGrad"
    path = os.path.abspath(automon_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    logging.disable(logging.ERROR)


def autograd_net(layers, kind: str):
    """A network as an autograd function of one vector (AutoMon differentiates it)."""
    import autograd.numpy as np
    Ws = [(onp.asarray(W, float), onp.asarray(b, float)) for W, b in layers]

    def f(x):
        h = x
        for W, b in Ws[:-1]:
            z = np.dot(h, W) + b
            h = np.tanh(z) if kind == "tanh" else np.maximum(z, 0.0)
        z = np.dot(h, Ws[-1][0]) + Ws[-1][1]
        return z[0] if kind == "tanh" else 1.0 / (1.0 + np.exp(-z[0]))
    return f


class ArrayGenerator:
    """Feeds a dense stream (rounds, nodes, d) to AutoMon's test loop, round-robin."""

    def __init__(self, X):
        self.X = onp.asarray(X, float)
        self.T, self.num_nodes, self.d = self.X.shape
        self.state = "monitoring"
        self.reset()

    def reset(self):
        self.cur = self.X[0].copy()
        self.j = 0

    def get_num_iterations(self):
        return self.T

    def get_local_vector(self, i):
        return self.cur[i].copy()

    def get_global_vector(self):
        return self.cur.mean(axis=0)

    def get_num_samples_left(self):
        return (self.T - 1) * self.num_nodes - self.j

    def get_next_data_point(self):
        r, i = divmod(self.j, self.num_nodes)
        self.cur[i] = self.X[r + 1, i]
        self.j += 1
        return self.cur[i].copy(), i
