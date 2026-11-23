#!/usr/bin/env python3
"""Rebuild AutoMon's benchmark streams and run the original AutoMon on them.

    git clone https://github.com/hsivan/automon AUTOMON_DIR   # add the data files its README lists
    pip install autograd pyzmq cvxpy
    python run_automon_benchmarks.py --automon AUTOMON_DIR --out data/bench
    python run_automon_benchmarks.py --automon AUTOMON_DIR --out data/bench --tiles 20 \
           --cases inner_product quadratic MLP_40 MLP_80 --skip-automon

For each benchmark writes <name>.npz: the local-vector stream (X0, idx, V), the
tolerance, the function's parameters and, unless --skip-automon, AutoMon's
message count, error per step, true values and neighbourhood size.
"""
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as onp  # noqa: E402

from monitor_pilot.automon_glue import autograd_net, prepare  # noqa: E402

CASES = ("inner_product", "quadratic", "MLP_40", "MLP_80", "air_quality", "intrusion_detection")


def record(gen):
    """Replay a data generator and return its local-vector stream."""
    k, d = gen.num_nodes, gen.d
    X0 = onp.array([gen.get_local_vector(i) for i in range(k)], dtype=float)
    n = gen.get_num_samples_left()
    idx, V = onp.empty(n, dtype=onp.int32), onp.empty((n, d))
    for j in range(n):
        lv, i = gen.get_next_data_point()
        idx[j], V[j] = i, lv
    return X0, idx, V


def build(name, root, tmp, limit=0):
    from test_utils import data_generator as dg
    from test_utils.functions_to_monitor import (func_inner_product, func_kld,
                                                 get_func_quadratic)
    from test_utils.test_utils import read_config_file
    folder = os.path.join(root, "datasets", name)
    conf = read_config_file(folder)
    if limit:
        conf["num_iterations"] = limit
    common = dict(num_iterations=conf["num_iterations"], num_nodes=conf["num_nodes"],
                  d=conf["d"], sliding_window_size=conf["sliding_window_size"],
                  num_iterations_for_tuning=conf.get("num_iterations_for_tuning", 0))
    params, single, tune = {}, False, False
    if name == "inner_product":
        gen = dg.DataGeneratorInnerProduct(data_file_name="data_file.txt",
                                           test_folder=folder, **common)
        func = func_inner_product
    elif name == "quadratic":
        H = onp.loadtxt(os.path.join(folder, "H_matrix.txt"), dtype=onp.float32)
        gen = dg.DataGeneratorQuadratic(data_file_name="data_file.txt",
                                        test_folder=folder, **common)
        func, params = get_func_quadratic(H), dict(H=H)
    elif name.startswith("MLP"):
        net = onp.load(os.path.join(folder, "net_params.npy"), allow_pickle=True)
        layers = [l for l in net if len(l) == 2]
        gen = dg.DataGeneratorMlp(data_file_name="data_file.txt", test_folder=folder,
                                  **common)
        func, tune = autograd_net(layers, "tanh"), True
        params = {f"W{i}": onp.asarray(W) for i, (W, _) in enumerate(layers)}
        params.update({f"b{i}": onp.asarray(b) for i, (_, b) in enumerate(layers)})
    elif name == "air_quality":
        gen = dg.DataGeneratorKldAirQuality(test_folder=tmp, **common)
        func, tune = func_kld, True
    elif name == "intrusion_detection":
        net = onp.load(os.path.join(folder, "net_params_intrusion.npy"), allow_pickle=True)
        layers = [l for l in net if len(l) == 2]
        gen = dg.DataGeneratorDnnIntrusionDetection(test_folder=tmp, **common)
        func, single = autograd_net(layers, "relu"), True
        params = {f"W{i}": onp.asarray(W) for i, (W, _) in enumerate(layers)}
        params.update({f"b{i}": onp.asarray(b) for i, (_, b) in enumerate(layers)})
    else:
        raise ValueError(name)
    return conf, gen, func, params, single, tune


def tiles(name, root, out, n_tiles):
    """A long stream for a synthetic benchmark: fresh draws of its generator, joined.

    Each tile is one full run of AutoMon's generator for that benchmark with a
    new random seed (same length, noise and motion as the published stream).
    Written as ``<name>_long.npz`` without AutoMon results.
    """
    from test_utils import data_generator as dg
    from test_utils.test_utils import read_config_file
    folder = os.path.join(root, "datasets", name)
    conf = read_config_file(folder)
    k, d = conf["num_nodes"], conf["d"]
    cls = {"inner_product": dg.DataGeneratorInnerProduct, "quadratic": dg.DataGeneratorQuadratic,
           "MLP_40": dg.DataGeneratorMlp, "MLP_80": dg.DataGeneratorMlp}[name]
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for t in range(n_tiles):
            onp.random.seed(1000 + t)
            gen = cls(num_iterations=conf["num_iterations"], num_nodes=k, d=d, test_folder=tmp,
                      sliding_window_size=conf["sliding_window_size"],
                      num_iterations_for_tuning=conf.get("num_iterations_for_tuning", 0))
            for f in os.listdir(tmp):
                os.remove(os.path.join(tmp, f))
            X0, idx, V = record(gen)
            cur, rounds = X0.copy(), [X0.copy()]
            for r in range(len(idx) // k):
                sl = slice(r * k, (r + 1) * k)
                cur[idx[sl]] = V[sl]
                rounds.append(cur.copy())
            parts.append(onp.array(rounds))
    X = onp.concatenate(parts)
    base = onp.load(os.path.join(out, f"{name}.npz"))
    keep = {key: base[key] for key in base.files
            if key[0] in "WbH" and not key.startswith("tune")}
    onp.savez(os.path.join(out, f"{name}_long.npz"), X0=X[0],
              idx=onp.tile(onp.arange(k, dtype=onp.int32), len(X) - 1),
              V=X[1:].reshape(-1, d), rounds=True, eps=conf["error_bound"], nodes=k, d=d,
              **keep)
    print(f"{name}_long: {len(X)} rounds from {n_tiles} tiles", flush=True)


def run(name, root, out, skip_automon=False, limit=0):
    from test_utils.object_factory import get_objects
    from test_utils.test_utils import run_test
    from test_utils.tune_neighborhood_size import tune_neighborhood_size
    from automon import AutomonCoordinator, AutomonNode
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        conf, gen, func, params, single, tune = build(name, root, tmp, limit)
        gen.reset()
        X0, idx, V = record(gen)
        save = dict(X0=X0, idx=idx, V=V, rounds=not single, eps=conf["error_bound"],
                    nodes=conf["num_nodes"], d=conf["d"], **params)
        if conf.get("num_iterations_for_tuning", 0):
            gen.set_neighborhood_tuning_state()
            tX0, tidx, tV = record(gen)
            save.update(tune_X0=tX0, tune_idx=tidx, tune_V=tV)
            gen.set_monitoring_state()
        print(f"{name}: stream {V.shape}, {conf}  ({time.time() - t0:.0f}s)", flush=True)
        if not skip_automon:
            gen.reset()
            coord, nodes = get_objects(AutomonNode, AutomonCoordinator, conf, func)
            if tune:
                tune_neighborhood_size(coord, nodes, conf, gen, single)
            run_test(gen, coord, nodes, None, b_single_sample_per_round=single)
            st = coord.statistics
            err = onp.array(st.function_approximation_error, dtype=float)
            save.update(am_msgs=st._total_msgs_for_disabled_broadcast(), am_err=err,
                        am_r=float(coord.neighborhood_size),
                        am_const_hessian=bool(coord.b_hessian_const),
                        am_full_syncs=len(st.full_sync_history),
                        am_true=onp.array(st.real_function_value, dtype=float),
                        am_cum=onp.array(st.cumulative_msg_for_broadcast_disabled))
            print(f"{name}: AutoMon msgs {save['am_msgs']} of {len(idx)} updates "
                  f"({save['am_msgs'] / len(idx):.3f}), violations "
                  f"{(err > conf['error_bound'] * (1 + 1e-9)).mean():.4f}, max err "
                  f"{err.max() / conf['error_bound']:.3f}, neighbourhood {save['am_r']}, "
                  f"const Hessian {save['am_const_hessian']}, full syncs "
                  f"{save['am_full_syncs']}  ({time.time() - t0:.0f}s)", flush=True)
        os.makedirs(out, exist_ok=True)
        onp.savez(os.path.join(out, f"{name}_first{limit}.npz" if limit else f"{name}.npz"),
                  **save)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--automon", required=True, help="folder of the AutoMon clone")
    ap.add_argument("--out", default="data/bench")
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--skip-automon", action="store_true",
                    help="only rebuild the streams")
    ap.add_argument("--tiles", type=int, default=0,
                    help="also write <name>_long.npz for the synthetic benchmarks, "
                         "joining this many fresh draws of the generator")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only this many leading iterations of the stream and save "
                         "the result as <name>_first<limit>.npz")
    a = ap.parse_args()
    prepare(a.automon)
    for name in a.cases:
        if not (a.tiles and os.path.exists(os.path.join(a.out, f"{name}.npz"))):
            run(name, os.path.abspath(a.automon), a.out, a.skip_automon, a.limit)
        if a.tiles and name in ("inner_product", "quadratic", "MLP_40", "MLP_80"):
            tiles(name, os.path.abspath(a.automon), a.out, a.tiles)


if __name__ == "__main__":
    main()
