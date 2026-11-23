"""Functions of the average local vector that the coordinator monitors.

Each function offers
    f(P)              values for a batch of points P (n, d)
    grad(x), hess(x)  exact derivatives at one point
    const_hessian     True when the Hessian does not depend on x
    f_min, f_max      known range of f (used to skip one side of a safe zone)
    scale             per-coordinate scale of local vectors, for send-on-delta
"""
from __future__ import annotations

import numpy as np


class Variance:
    """Spread of one reading across nodes.

    Node i holds (z_i, z_i^2); with a = mean z and b = mean z^2 the variance is
    f = b - a^2, a concave function with a constant Hessian.
    """
    name, d, const_hessian = "variance", 2, True
    f_min, f_max = 0.0, np.inf
    domain = None

    def __init__(self):
        self.scale = np.ones(2)

    @staticmethod
    def local(Z: np.ndarray) -> np.ndarray:
        """Readings (T, k, 1) -> local vectors (T, k, 2)."""
        z = Z[..., 0]
        return np.stack([z, z ** 2], axis=-1)

    def f(self, P):
        P = np.atleast_2d(P)
        return P[:, 1] - P[:, 0] ** 2

    def grad(self, x):
        return np.array([-2.0 * x[0], 1.0])

    def hess(self, x):
        return np.array([[-2.0, 0.0], [0.0, 0.0]])


class TanhMLP:
    """One hidden tanh layer: f(x) = w2 . tanh(W1 x + b1) + b2 (non-convex)."""
    name, const_hessian = "mlp", False
    f_min, f_max = -np.inf, np.inf
    domain = None

    def __init__(self, d: int, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.d = d
        self.W1 = rng.normal(scale=1.0 / np.sqrt(d), size=(hidden, d))
        self.b1 = rng.normal(scale=0.1, size=hidden)
        self.w2 = rng.normal(scale=1.0 / np.sqrt(hidden), size=hidden)
        self.b2 = 0.0
        self.scale = np.ones(d)

    def _h(self, P):
        return np.tanh(P @ self.W1.T + self.b1)

    def f(self, P):
        return self._h(np.atleast_2d(P)) @ self.w2 + self.b2

    def grad(self, x):
        h = self._h(x[None])[0]
        return self.W1.T @ (self.w2 * (1.0 - h ** 2))

    def hess(self, x):
        h = self._h(x[None])[0]
        c = self.w2 * (-2.0 * h * (1.0 - h ** 2))
        return (self.W1.T * c) @ self.W1

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 400, lr: float = 0.01,
            batch: int = 256, seed: int = 0) -> "TanhMLP":
        """Least-squares fit with Adam, in plain NumPy."""
        rng = np.random.default_rng(seed)
        params = [self.W1, self.b1, self.w2, np.array([self.b2], float)]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, t = 0.9, 0.999, 0
        n = len(X)
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, batch):
                j = order[s:s + batch]
                xb, yb = X[j], y[j]
                h = np.tanh(xb @ params[0].T + params[1])
                r = (h @ params[2] + params[3][0]) - yb
                gz = (r[:, None] * params[2]) * (1 - h ** 2) * (2.0 / len(j))
                grads = [gz.T @ xb, gz.sum(0), h.T @ r * (2.0 / len(j)),
                         np.array([2.0 * r.mean()])]
                t += 1
                for i, g in enumerate(grads):
                    m[i] = b1 * m[i] + (1 - b1) * g
                    v[i] = b2 * v[i] + (1 - b2) * g * g
                    params[i] -= lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(v[i] / (1 - b2 ** t)) + 1e-8)
        self.W1, self.b1, self.w2, self.b2 = params[0], params[1], params[2], float(params[3][0])
        return self
