"""
Benchmark 2: solution quality on QUBO / MaxCut / portfolio instances, n = 16 .. 80.

TCI-bits applies TCI directly to the objective E(x) (no QAOA).  The portfolio kind is
Markowitz selection of k = n // 4 assets (risk aversion 1, linear return, budget
penalty; ``tciqaoa.portfolio_qubo``) on hybridq's own ``make_portfolio_data``.

Compares, on identical instances (hybridq's own generators):
  hybridq-SB    hybridq.solve_qubo(core='sb')   (simulated bifurcation)
  hybridq-QA    hybridq.solve_qubo(core='qa')   (simulated quantum annealing)
  SA, Tabu      the classical baselines shipped in hybridq/examples.py
  LS-multistart steepest-descent from random starts (control for TCI's local polish)
  TCI-bits      tciqaoa.tci_minimize_bits (this repo)

Reference energy: exact brute force for n <= 22, otherwise the best energy any
method found on that instance ("best known").

Usage:
  HYBRIDQ_PATH=/path/to/QAOA-inspired-optimisation python benchmarks/bench_solvers.py
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tciqaoa.problems import (brute_force, maxcut_qubo, random_graph, random_qubo,
                              local_search, qubo_energy, portfolio_data, portfolio_qubo)
from tciqaoa.tci_bits import tci_minimize_bits

HYB = os.environ.get("HYBRIDQ_PATH")
if HYB:
    sys.path.insert(0, HYB)
import hybridq  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "hyb_examples", os.path.join(os.path.dirname(hybridq.__file__), "..", "examples.py"))
hyb_examples = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyb_examples)


def ls_multistart(Q, n_starts, seed, time_budget=None):
    """Random-restart steepest descent; with time_budget, restart until the time is used."""
    rng = np.random.default_rng(seed)
    best = (None, np.inf)
    t0, k = time.perf_counter(), 0
    while (k < n_starts) if time_budget is None else (time.perf_counter() - t0 < time_budget):
        k += 1
        x, e, _ = local_search(rng.integers(0, 2, Q.shape[0]), Q)
        if e < best[1]:
            best = (x, e)
    return best


def run_instance(kind, n, seed, ls_starts):
    if kind == "maxcut":
        Q = maxcut_qubo(random_graph(n, 0.4, seed=seed))
    elif kind == "qubo":
        Q = random_qubo(n, 0.5, seed=seed)
    else:
        mu, sigma = portfolio_data(n, seed=seed)
        Q = portfolio_qubo(mu, sigma, n // 4)
    out = {}

    def timed(name, fn):
        t = time.perf_counter()
        x = fn()
        out[name] = {"energy": qubo_energy(x, Q), "time": time.perf_counter() - t}

    timed("hybridq-SB", lambda: hybridq.solve_qubo(Q, core="sb", verbose=False, seed=seed).x)
    timed("hybridq-QA", lambda: hybridq.solve_qubo(Q, core="qa", verbose=False, seed=seed).x)
    timed("SA", lambda: hyb_examples.classical_sa(Q, seed=seed)[0])
    timed("Tabu", lambda: hyb_examples.tabu_search(Q, seed=seed)[0])
    timed("LS-multistart", lambda: ls_multistart(Q, ls_starts, seed)[0])
    tci = {}
    timed("TCI-bits", lambda: tci.update(tci_minimize_bits(Q, seed=seed)) or tci["x"])
    out["TCI-bits"]["n_energy_evals"] = tci["n_energy_evals"]
    out["TCI-bits"]["max_rank"] = tci["max_rank"]
    # control: same wall-clock as TCI, spent on random-restart local search only
    timed("LS-time-matched", lambda: ls_multistart(Q, 0, seed + 1000, out["TCI-bits"]["time"])[0])

    if n <= 22:
        ref, exact = brute_force(Q)[1], True
    else:
        ref, exact = min(v["energy"] for v in out.values()), False
    for v in out.values():
        v["hit"] = bool(v["energy"] <= ref + 1e-9)
        v["gap"] = float((v["energy"] - ref) / abs(ref)) if ref else 0.0
    return {"kind": kind, "n": n, "seed": seed, "ref": ref, "exact": exact, "methods": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 20, 30, 50, 80])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--ls-starts", type=int, default=100)
    ap.add_argument("--kinds", nargs="+", default=["maxcut", "qubo"])
    ap.add_argument("--out", default="results/solvers.json")
    a = ap.parse_args()
    rows = []
    for kind in a.kinds:
        for n in a.sizes:
            for s in range(a.seeds):
                r = run_instance(kind, n, s, a.ls_starts)
                rows.append(r)
                hits = " ".join(f"{k}:{'Y' if v['hit'] else '.'}" for k, v in r["methods"].items())
                print(f"{kind:6s} n={n:3d} seed={s}  {hits}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
