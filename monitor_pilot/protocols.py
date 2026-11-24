"""Monitoring protocols and the audit-based calibration rule.

A coordinator tracks f(average of the nodes' local vectors) and must stay
within a tolerance eps.  Messages are counted the same way for every method:

  * a node report, a violation report or a new tolerance sent to a node: 1
  * the coordinator asking a node for its vector and the reply:          2
  * a full synchronisation: one report from every node not yet heard,
    plus one message per node to distribute the new reference
  * an audit: every node reports and receives its new tolerance (2 per node,
    less any reports already sent in that round)

``msg_ratio`` is messages divided by node updates, so central reporting
(every node reports every update) is 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize


# ------------------------------------------------------------- calibration

@dataclass
class Calibration:
    alpha: float                 # target share of rounds in violation
    p: float                     # audit probability per round
    eta: float = 0.5             # step size on the log-tolerance
    u_min: float = np.log(0.02)  # at or below this, fall back to full reporting.
    # u is deliberately not clipped from below: after a loss at the floor it must
    # earn its way back up, which is what makes the bound in ``slack`` hold.
    u_max: float = np.log(100.0)
    seed: int = 0
    ramp: float = 1.0            # loss rises from 0 at ramp*eps to 1 at eps (1 = indicator)
    boost: float = 1.0           # audit rate multiplier during the first boost_frac of rounds
    boost_frac: float = 0.1

    def rates(self, rounds: int) -> np.ndarray:
        """Audit probability per round; the expected number of audits is p * rounds."""
        if self.boost == 1.0:
            return np.full(rounds, self.p)
        assert self.boost * self.boost_frac < 1.0
        head = int(self.boost_frac * rounds)
        low = self.p * (1.0 - self.boost * self.boost_frac) / (1.0 - self.boost_frac)
        return np.where(np.arange(rounds) < head, self.p * self.boost, low)

    def schedule(self, rounds: int) -> np.ndarray:
        """Audit rounds, drawn before the run and independent of the data."""
        return np.random.default_rng(self.seed).random(rounds) < self.rates(rounds)

    def loss(self, e_rel):
        """Loss of an error given in units of the tolerance; never below 1[e_rel > 1]."""
        e_rel = np.asarray(e_rel, float)
        hit = (e_rel > 1.0 + 1e-9).astype(float)
        if self.ramp >= 1.0:
            return hit
        return np.maximum(hit, np.clip((e_rel - self.ramp) / (1.0 - self.ramp), 0.0, 1.0))

    def slack(self, audits: int, u0: float = 0.0) -> float:
        """Finite-sample slack: mean audited loss <= alpha + slack."""
        return (u0 - self.u_min + self.eta) / (self.eta * max(audits, 1))


# ------------------------------------------------------------------ streams

class Events:
    """A stream in which one node changes per step (not all nodes per round).

    ``X0`` holds the start vectors (nodes, d); step t >= 1 sets node
    ``idx[t-1]`` to ``V[t-1]``.  Indexing must be sequential: ``stream[t]``
    returns the current (nodes, d) matrix after applying step t.
    """

    def __init__(self, X0, idx, V):
        self.X0, self.idx, self.V = np.asarray(X0, float), np.asarray(idx), np.asarray(V, float)
        self.shape = (len(self.idx) + 1,) + self.X0.shape
        self.updates = len(self.idx) + len(self.X0)
        self._x, self._t = None, -1

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, t):
        if t == 0:
            self._x, self._t = self.X0.copy(), 0
        elif t == self._t + 1:
            self._x[self.idx[t - 1]] = self.V[t - 1]
            self._t = t
        elif t != self._t:
            raise IndexError("Events must be read in order")
        return self._x

    def means(self) -> np.ndarray:
        """Average vector after every step, shape (steps, d)."""
        k = len(self.X0)
        delta = np.empty_like(self.V)
        for i in range(k):
            pos = np.nonzero(self.idx == i)[0]
            prev = np.vstack([self.X0[i][None], self.V[pos[:-1]]]) if len(pos) else None
            if len(pos):
                delta[pos] = (self.V[pos] - prev) / k
        m0 = self.X0.mean(axis=0)
        return np.vstack([m0[None], m0 + np.cumsum(delta, axis=0)])


def n_updates(X) -> int:
    """Node updates in a stream: every node every round, or one per event."""
    if isinstance(X, Events):
        return X.updates
    return int(X.shape[0] * X.shape[1])


@dataclass
class Run:
    name: str
    rounds: int
    nodes: int
    eps: float
    updates: int = 0
    msgs: int = 0
    err: np.ndarray = field(default_factory=lambda: np.zeros(0))
    full_syncs: int = 0
    lazy_syncs: int = 0
    audits: int = 0
    audit_violations: int = 0
    audit_loss: float = 0.0
    u_trace: list = field(default_factory=list)
    est: np.ndarray = field(default_factory=lambda: np.zeros(0))   # coordinator's value
    # message directions (revision): uplinks, downlinks, downlinks to a node that
    # did not transmit in the same round (impossible in LoRaWAN class A without
    # an extra uplink from that node), and per-node counts
    up: int = 0
    down: int = 0
    unsolicited: int = 0
    node_up: np.ndarray = field(default_factory=lambda: np.zeros(0))
    node_down: np.ndarray = field(default_factory=lambda: np.zeros(0))
    node_unsol: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lost: int = 0

    @property
    def msg_ratio(self) -> float:
        """Messages as a share of full reporting (one per node update)."""
        return self.msgs / (self.updates or self.rounds * self.nodes)

    @property
    def viol_rate(self) -> float:
        return float(np.mean(self.err > self.eps * (1 + 1e-9)))

    @property
    def mean_err(self) -> float:
        return float(np.mean(self.err) / self.eps)

    @property
    def max_err(self) -> float:
        return float(np.max(self.err) / self.eps)


def true_values(X, func) -> np.ndarray:
    return func.f(X.means() if isinstance(X, Events) else X.mean(axis=1))


# ----------------------------------------------------------------- baselines

def run_central(X, func, eps) -> Run:
    T, k, _ = X.shape
    return Run("central", T, k, eps, n_updates(X), msgs=n_updates(X), err=np.zeros(T))


def run_periodic(X, func, eps, period: int, F_true=None) -> Run:
    T, k, _ = X.shape
    F_true = true_values(X, func) if F_true is None else F_true
    last = (np.arange(T) // period) * period
    return Run(f"periodic/{period}", T, k, eps, n_updates(X),
               msgs=k * len(np.unique(last)), err=np.abs(F_true - F_true[last]))


class _Channel:
    """Uplink losses: independent (Bernoulli) or bursty (Gilbert-Elliott, one
    two-state chain per node, advanced at each transmission attempt; every
    attempt in the bad state is lost, mean burst length ``burst`` attempts)."""

    def __init__(self, k, loss, model="bernoulli", burst=3.0, seed=0):
        self.loss, self.model = float(loss), model
        self.rng = np.random.default_rng(10_000 + seed)
        self.bad = np.zeros(k, bool)
        self.b2g = 1.0 / burst
        self.g2b = self.loss * self.b2g / max(1.0 - self.loss, 1e-12)
        if model == "ge" and self.loss > 0:          # start in the stationary state
            self.bad = self.rng.random(k) < self.loss

    def attempt(self, nodes):
        """Delivered flags for one transmission attempt by each of ``nodes``."""
        if self.loss <= 0:
            return np.ones(len(nodes), bool)
        if self.model == "bernoulli":
            return self.rng.random(len(nodes)) >= self.loss
        r = self.rng.random(len(nodes))
        b = self.bad[nodes]
        b = np.where(b, r >= self.b2g, r < self.g2b)
        self.bad[nodes] = b
        return ~b

    def until_delivered(self, nodes):
        """Attempts per node when each node retransmits until delivered."""
        att = np.zeros(len(nodes), int)
        todo = np.arange(len(nodes))
        while len(todo):
            att[todo] += 1
            ok = self.attempt(nodes[todo])
            todo = todo[~ok]
        return att


def run_filter(X, func, eps, delta: float, F_true=None,
               cal: Optional[Calibration] = None, period: int = 0,
               weights: Optional[np.ndarray] = None, loss: float = 0.0,
               loss_model: str = "bernoulli", burst: float = 3.0,
               confirmed: bool = False, loss_seed: int = 0) -> Run:
    """Send-on-delta per node; ``delta`` is in units of ``func.scale``.

    A node reports when the distance between its vector and its last report
    exceeds the tolerance.  ``weights`` (d,) rescales each coordinate first, so
    that movement the function is insensitive to counts for less.  With
    ``period`` > 0 a node instead reports every ``period``-th time its vector
    changes (periodic reporting for streams where nodes update one at a time).
    With ``cal`` the tolerance (or the period) is multiplied by exp(u) and u is
    calibrated by audits.

    Lossy uplinks (revision): with ``loss`` > 0 each report is lost with that
    probability (``loss_model`` "bernoulli" or "ge").  A node does not know its
    report was lost unless ``confirmed``, in which case it retransmits until the
    gateway's acknowledgement (one downlink in the node's receive window)
    arrives.  Audit reports are always confirmed, so an audit re-synchronises
    every node.  The new u is sent in the receive window of each node's audit
    report.
    """
    T, k, _ = X.shape
    F_true = true_values(X, func) if F_true is None else F_true
    run = Run("filter", T, k, eps, n_updates(X), err=np.zeros(T), est=np.zeros(T))
    audit = cal.schedule(T) if cal else None
    ch = _Channel(k, loss, loss_model, burst, loss_seed)
    u = 0.0
    scale = func.scale if weights is None else func.scale / weights
    last = X[0].copy()                 # node side: last vector each node sent
    coord = X[0].copy()                # coordinator side: last vector received
    since = np.zeros(k, dtype=int)
    node_up, node_down = np.ones(k), np.zeros(k)
    for t in range(1, T):
        x = X[t]
        d = x - last
        if period:
            if isinstance(X, Events):
                since[X.idx[t - 1]] += 1
            else:
                since += 1
            if cal is None:
                send = since >= period
            elif u <= cal.u_min:                # at the floor every change is reported
                send = since >= 1
            else:                               # calibrated period: period * exp(u)
                send = since >= max(1.0, period * np.exp(u))
            since[send] = 0
        else:                                   # at the floor every change is reported
            tol = 0.0 if cal is not None and u <= cal.u_min else delta * np.exp(u)
            send = np.sqrt(np.sum((d / scale) ** 2, axis=1)) > tol
        idx = np.nonzero(send)[0]
        if len(idx):
            last[idx] = x[idx]
            if confirmed:
                node_up[idx] += ch.until_delivered(idx)
                node_down[idx] += 1                         # acknowledgement
                coord[idx] = x[idx]
            else:
                ok = ch.attempt(idx)
                node_up[idx] += 1
                coord[idx[ok]] = x[idx[ok]]
                run.lost += int((~ok).sum())
        v = func.f(coord.mean(axis=0))[0]
        e = abs(F_true[t] - v)
        run.est[t] = v
        if cal is not None and audit[t]:
            loss_ = float(cal.loss(e / eps))
            u = float(min(u + cal.eta * (cal.alpha - loss_), cal.u_max))   # no lower clip
            rest = np.setdiff1d(np.arange(k), idx)          # nodes not yet heard this round
            if confirmed or loss > 0:
                node_up[rest] += ch.until_delivered(rest)
            else:
                node_up[rest] += 1
            node_down += 1                                  # new u, in each receive window
            last = x.copy()
            coord = x.copy()
            run.audits += 1
            run.audit_violations += int(e > eps * (1 + 1e-9))
            run.audit_loss += loss_
            run.u_trace.append(u)
            e = 0.0
            run.est[t] = F_true[t]
        run.err[t] = e
    run.est[0] = F_true[0]
    run.node_up, run.node_down, run.node_unsol = node_up, node_down, np.zeros(k)
    run.up, run.down = int(node_up.sum()), int(node_down.sum())
    run.msgs = run.up + run.down
    return run


# ------------------------------------------------- safe zones (AutoMon-style)

def _box(func, x0, r):
    """Neighbourhood x0 +- r, clipped to the function's domain if it has one."""
    lo, hi = x0 - r, x0 + r
    dom = getattr(func, "domain", None)
    if dom is not None:
        lo, hi = np.maximum(lo, dom[0]), np.minimum(hi, dom[1])
    return lo, hi


def _extreme_eigs(func, x0, r):
    """Smallest and largest Hessian eigenvalue in the box x0 +- r.

    Same search as AutoMon: L-BFGS-B from x0, five iterations.  It is a
    heuristic for non-convex functions, which is why no guarantee follows.
    A function may offer ``eig_range(x)`` (smallest, largest eigenvalue in
    closed form); the search is the same, only each evaluation is cheaper.
    """
    box = list(zip(*_box(func, x0, r)))
    if getattr(func, "eig_range", None) is not None and func.eig_range(x0) is not None:
        low, high = (lambda z: func.eig_range(z)[0]), (lambda z: -func.eig_range(z)[1])
    else:
        low = lambda z: np.linalg.eigvalsh(func.hess(z))[0]          # noqa: E731
        high = lambda z: -np.linalg.eigvalsh(func.hess(z))[-1]       # noqa: E731
    lo = minimize(low, x0, bounds=box, method="L-BFGS-B", options={"maxiter": 5})
    hi = minimize(high, x0, bounds=box, method="L-BFGS-B", options={"maxiter": 5})
    lo, hi = float(lo.fun), float(-hi.fun)
    return (-np.inf if np.isnan(lo) else lo), (np.inf if np.isnan(hi) else hi)


class Zone:
    """Local constraint shared by all nodes between two full syncs."""

    def __init__(self, func, x0, tol, r=None):
        self.func, self.x0, self.r = func, x0, r
        self.f0 = float(func.f(x0)[0])
        self.L, self.U = self.f0 - tol, self.f0 + tol
        self.g = func.grad(x0)
        lam, V = np.linalg.eigh(func.hess(x0))
        d = len(x0)
        if func.const_hessian:                               # ADCD-E
            self.r = None
            self.convex = abs(lam[0]) <= lam[-1]
            keep = np.minimum(lam, 0) if self.convex else np.maximum(lam, 0)
            self.D = (V * keep) @ V.T
        else:                                                # ADCD-X
            lo, hi = _extreme_eigs(func, x0, r)
            lo_m, hi_p = min(lo, 0.0), max(hi, 0.0)
            self.convex = abs(lam[0] + 2 * abs(lo_m)) <= abs(lam[-1] - 2 * hi_p)
            self.D = (lo_m if self.convex else hi_p) * np.eye(d)
            self.box = _box(func, x0, r)
        self.skip_upper = self.U >= func.f_max
        self.skip_lower = self.L <= func.f_min

    def check(self, P):
        """For points (n, d): inside flags and origin (1 box, 2 zone, 3 faulty)."""
        delta = P - self.x0
        with np.errstate(all="ignore"):       # points outside the domain give nan
            fP = self.func.f(P)               # and are caught by the box test below
        q = 0.5 * np.einsum("nd,de,ne->n", delta, self.D, delta)
        lin = delta @ self.g
        if self.convex:
            upper = fP - q <= self.U
            lower = -q <= self.f0 + lin - self.L
        else:
            upper = q <= -self.f0 - lin + self.U
            lower = -fP + q <= -self.L
        if self.skip_upper:
            upper = np.ones(len(P), bool)
        if self.skip_lower:
            lower = np.ones(len(P), bool)
        bounds = ((self.L <= fP) | self.skip_lower) & ((fP <= self.U) | self.skip_upper)
        origin = np.zeros(len(P), int)
        origin[~bounds] = 3
        origin[~(upper & lower)] = 2
        if self.r is not None:
            origin[np.any((P < self.box[0]) | (P > self.box[1]), axis=1)] = 1
        return origin == 0, origin


def run_zone(X, func, eps, r: Optional[float] = None, theta: float = 1.0,
             F_true=None, cal: Optional[Calibration] = None,
             lazy_max: float = 0.5) -> Run:
    """Safe-zone monitoring with drift slack and lazy (LRU) synchronisation.

    ``theta`` multiplies the error tolerance used to build the zones
    (theta = 1 is the standard method).  With ``cal`` the multiplier is
    exp(u) * theta and u is calibrated by audits.
    """
    T, k, d = X.shape
    F_true = true_values(X, func) if F_true is None else F_true
    run = Run("zone", T, k, eps, n_updates(X), err=np.zeros(T), est=np.full(T, np.nan))
    audit = cal.schedule(T) if cal else None
    s_max = int(np.round(lazy_max * k))
    st = dict(r=r, nb=0, u=0.0, central=False)
    lru = np.zeros(k)

    def full_sync(x):
        if st["r"] is not None and st["nb"] > 5 * k:
            st["r"] *= 2.0
            st["nb"] = 0
        x0 = x.mean(axis=0)
        run.full_syncs += 1
        return Zone(func, x0, theta * np.exp(st["u"]) * eps, st["r"]), x - x0

    zone, slack = full_sync(X[0])
    run.est[0] = zone.f0
    nu, nd, nx = np.ones(k), np.zeros(k), np.zeros(k)     # up, down, unsolicited down
    for t in range(1, T):
        x = X[t]
        if st["central"]:                       # fallback: everyone reports
            nu += 1
            run.est[t] = F_true[t]
            if audit[t]:
                st["u"] = min(st["u"] + cal.eta * cal.alpha, cal.u_max)
                run.audits += 1
                run.u_trace.append(st["u"])
                if st["u"] > cal.u_min:
                    st["central"] = False
                    zone, slack = full_sync(x)
                    nd += 1                     # new zone, in each receive window
            continue
        P = x - slack
        ok, origin = zone.check(P)
        heard = np.zeros(k, bool)               # nodes that transmitted this round
        if not ok.all():
            viol = ~ok
            nu[viol] += 1
            heard |= viol
            for o in origin[viol]:
                st["nb"] = st["nb"] + 1 if o == 1 else 0
            have = viol.copy()
            eager = bool((origin[viol] == 3).any()) or \
                (st["r"] is not None and st["nb"] > 5 * k)
            while not eager:
                S = np.nonzero(have)[0]
                if len(S) <= s_max:
                    m = P[S].mean(axis=0)
                    if zone.check(m[None])[0][0]:
                        slack[S] = x[S] - m
                        nd[S] += 1              # new slack; every node in S has sent
                        lru[S] += 1
                        run.lazy_syncs += 1
                        break
                if len(S) >= s_max:
                    eager = True
                    lru[:] = 0
                    break
                rest = np.nonzero(~have)[0]
                j = rest[np.argsort(lru[rest], kind="stable")[0]]
                have[j] = True
                nd[j] += 1                      # request to a silent node
                nx[j] += 1
                nu[j] += 1                      # its reply
                heard[j] = True
            if eager:
                miss = ~have
                nd[miss] += 1                   # requests to silent nodes
                nx[miss] += 1
                nu[miss] += 1                   # replies
                nd += 1                         # new zone to every node
                zone, slack = full_sync(x)
        e = abs(F_true[t] - zone.f0)
        run.est[t] = zone.f0
        if cal is not None and audit[t]:
            loss = float(cal.loss(e / eps))
            st["u"] = float(min(st["u"] + cal.eta * (cal.alpha - loss), cal.u_max))
            run.audits += 1
            run.audit_violations += int(e > eps * (1 + 1e-9))
            run.audit_loss += loss
            run.u_trace.append(st["u"])
            nu += 1                             # every node reports (shared schedule)
            nd += 1                             # and receives the new zone and u
            if st["u"] <= cal.u_min:
                st["central"] = True
            else:
                zone, slack = full_sync(x)
            e = 0.0
            run.est[t] = F_true[t]
        run.err[t] = e
    run.node_up, run.node_down, run.node_unsol = nu, nd, nx
    run.up, run.down, run.unsolicited = int(nu.sum()), int(nd.sum()), int(nx.sum())
    run.msgs = run.up + run.down
    return run
