"""AutoMon's own benchmark functions, in plain NumPy.

Same interface as ``functions.py``: f(P) for a batch, grad(x), hess(x),
const_hessian, f_min/f_max, scale, and lipschitz() where a global constant
exists (None otherwise).
"""
from __future__ import annotations

import numpy as np


class _Base:
    f_min, f_max = -np.inf, np.inf
    const_hessian = False
    domain = None            # (low, high) box that local vectors live in, if any
    scale = 1.0

    def lipschitz(self):
        return None


class InnerProduct(_Base):
    """f(x, y) = x . y on the two halves of the vector (constant Hessian)."""
    name, const_hessian = "inner_product", True

    def __init__(self, d: int):
        self.d, self.h = d, d // 2

    def f(self, P):
        P = np.atleast_2d(P)
        return np.sum(P[:, :self.h] * P[:, self.h:], axis=1)

    def grad(self, x):
        return np.concatenate([x[self.h:], x[:self.h]])

    def hess(self, x):
        Z, I = np.zeros((self.h, self.h)), np.eye(self.h)
        return np.block([[Z, I], [I, Z]])


class Quadratic(_Base):
    """f(x) = x^T H x (constant Hessian, indefinite)."""
    name, const_hessian = "quadratic", True

    def __init__(self, H: np.ndarray):
        self.H, self.d = np.asarray(H, float), len(H)

    def f(self, P):
        P = np.atleast_2d(P)
        return np.einsum("nd,de,ne->n", P, self.H, P)

    def grad(self, x):
        return (self.H + self.H.T) @ x

    def hess(self, x):
        return self.H + self.H.T


class KLD(_Base):
    """Smoothed Kullback-Leibler divergence of two histograms (AutoMon's air quality).

    The vector holds P (first half) and Q (second half); f = sum p log(p / q)
    with p = P + e, q = Q + e.  Local vectors live in [0, 1].
    """
    name = "kld"
    domain = (0.0, 1.0)

    def __init__(self, d: int, smooth: float = 1.0 / (200 * 12)):
        self.d, self.h, self.e = d, d // 2, smooth

    def f(self, P):
        P = np.atleast_2d(P)
        p, q = P[:, :self.h] + self.e, P[:, self.h:] + self.e
        return np.sum(p * np.log(p / q), axis=1)

    def grad(self, x):
        p, q = x[:self.h] + self.e, x[self.h:] + self.e
        return np.concatenate([np.log(p / q) + 1.0, -p / q])

    def hess(self, x):
        p, q = x[:self.h] + self.e, x[self.h:] + self.e
        return np.block([[np.diag(1.0 / p), np.diag(-1.0 / q)],
                         [np.diag(-1.0 / q), np.diag(p / q ** 2)]])


class Net(_Base):
    """Feed-forward network with weights W (in, out) per layer.

    kind = "tanh":  tanh hidden layers, linear output   (AutoMon's MLP-d)
    kind = "relu":  ReLU hidden layers, sigmoid output  (AutoMon's intrusion DNN)
    """

    def __init__(self, layers, kind: str):
        self.layers = [(np.asarray(W, float), np.asarray(b, float)) for W, b in layers]
        self.kind, self.name = kind, "mlp" if kind == "tanh" else "dnn"
        self.d = self.layers[0][0].shape[0]
        if kind == "relu":
            self.f_min, self.f_max = 0.0, 1.0

    def _forward(self, P):
        h, acts = P, []
        for W, b in self.layers[:-1]:
            z = h @ W + b
            h = np.tanh(z) if self.kind == "tanh" else np.maximum(z, 0.0)
            acts.append(h)
        W, b = self.layers[-1]
        return (h @ W + b)[..., 0], acts

    def f(self, P):
        z, _ = self._forward(np.atleast_2d(P))
        return z if self.kind == "tanh" else 1.0 / (1.0 + np.exp(-z))

    def _grad_logit(self, x):
        z, acts = self._forward(x[None])
        g = self.layers[-1][0][:, 0]
        for (W, _), h in zip(self.layers[-2::-1], acts[::-1]):
            dh = (1.0 - h[0] ** 2) if self.kind == "tanh" else (h[0] > 0).astype(float)
            g = W @ (g * dh)
        return g, float(z[0])

    def grad(self, x):
        g, z = self._grad_logit(x)
        if self.kind == "relu":
            s = 1.0 / (1.0 + np.exp(-z))
            g = g * s * (1.0 - s)
        return g

    def hess(self, x):
        if self.kind == "relu":           # piecewise linear inside the sigmoid
            g, z = self._grad_logit(x)
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1 - s) * (1 - 2 * s) * np.outer(g, g)
        step, d = 1e-5, self.d            # central differences of the exact gradient
        H = np.empty((d, d))
        for i in range(d):
            e = np.zeros(d)
            e[i] = step
            H[i] = (self.grad(x + e) - self.grad(x - e)) / (2 * step)
        return 0.5 * (H + H.T)

    def eig_range(self, x):
        """Extreme Hessian eigenvalues in closed form (ReLU network only).

        There the Hessian is c * g g^T, so its eigenvalues are c * |g|^2 and 0.
        """
        if self.kind != "relu":
            return None
        g, z = self._grad_logit(x)
        s = 1.0 / (1.0 + np.exp(-z))
        v = float(s * (1 - s) * (1 - 2 * s) * (g @ g))
        return min(v, 0.0), max(v, 0.0)

    def lipschitz(self):
        L = 1.0 if self.kind == "tanh" else 0.25
        for W, _ in self.layers:
            L *= np.linalg.norm(W, 2)
        return float(L)
