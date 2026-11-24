"""Stage 7: message directions, lossy uplinks and the network-size builder."""
import numpy as np

from monitor_pilot.functions import Variance
from monitor_pilot.protocols import Calibration, run_filter, run_zone, true_values


def _case(T=400, k=8, seed=0):
    rng = np.random.default_rng(seed)
    z = np.cumsum(rng.normal(scale=0.05, size=(T, k, 1)), axis=0)
    f = Variance()
    X = f.local(z)
    f.scale = X.reshape(-1, 2).std(axis=0)
    F = true_values(X, f)
    return f, X, F, 0.1 * float(F.std())


def test_directions_add_up():
    f, X, F, eps = _case()
    cal = Calibration(0.01, 0.05, seed=1, ramp=0.75)
    for run in (run_filter(X, f, eps, 0.2, F, cal=cal), run_zone(X, f, eps, F_true=F),
                run_zone(X, f, eps, F_true=F, cal=cal)):
        assert run.up + run.down == run.msgs
        assert run.node_up.sum() == run.up and run.node_down.sum() == run.down
        assert run.unsolicited <= run.down


def test_send_on_delta_has_no_unsolicited_downlinks():
    f, X, F, eps = _case()
    run = run_filter(X, f, eps, 0.2, F, cal=Calibration(0.01, 0.05, seed=2, ramp=0.75))
    assert run.unsolicited == 0
    assert run.down == run.audits * X.shape[1]


def test_confirmed_reports_reproduce_lossless_errors():
    f, X, F, eps = _case()
    cal = lambda: Calibration(0.01, 0.05, seed=3, ramp=0.75)  # noqa: E731
    base = run_filter(X, f, eps, 0.2, F, cal=cal())
    for model in ("bernoulli", "ge"):
        lossy = run_filter(X, f, eps, 0.2, F, cal=cal(), loss=0.2, loss_model=model,
                           confirmed=True)
        assert np.allclose(lossy.err, base.err)
        assert lossy.up >= base.up


def test_unconfirmed_loss_leaves_stale_copies():
    f, X, F, eps = _case()
    run = run_filter(X, f, eps, 0.05, F, loss=0.5)
    assert run.lost > 0
    base = run_filter(X, f, eps, 0.05, F)
    assert run.viol_rate >= base.viol_rate
