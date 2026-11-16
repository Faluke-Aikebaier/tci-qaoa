"""
Benchmark 7: TCI directly on the objective, for a whole efficient frontier.

Portfolio selection of k = n // 4 assets on hybridq's ``make_portfolio_data`` (sector
covariance), Markowitz cost with linear return and a budget penalty
(``tciqaoa.portfolio_qubo``), at 16 risk-aversion values lambda = 0.1 .. 5 (log
spaced; hybridq's frontier range).  No QAOA: every method minimises E_lambda(x).

Methods (all solve the 16 problems of one frontier):
  TCI-bits/λ     tci_minimize_bits from scratch at every lambda
  TCI-sweep      tci_sweep_bits: ONE TCI over (lambda bits, asset bits)
  SA/λ           hybridq's classical simulated annealing at every lambda
  LS/λ           100 random-restart steepest descents at every lambda
  LS-cont        steepest descent from the previous lambda's best + 10 random starts
                 (the classical warm start along a sweep)
  LS-matched     random-restart steepest descent with TCI-sweep's wall time, split
                 evenly over the 16 values (control: what TCI adds over its polish)

Reference: exact brute force for n <= 22, otherwise the best any method found.
Evaluations count full energy evaluations (a 1-flip local-search step counts 1/n).

Usage:  HYBRIDQ_PATH=/path/to/QAOA-inspired-optimisation python benchmarks/bench_frontier.py
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tciqaoa import (brute_force, local_search, portfolio_data, portfolio_qubo,  # noqa: E402
                     qubo_energy, tci_minimize_bits, tci_sweep_bits)

HYB = os.environ.get("HYBRIDQ_PATH")
_spec = importlib.util.spec_from_file_location("hyb_examples", os.path.join(HYB, "examples.py"))
hyb_examples = importlib.util.module_from_spec(_spec)
sys.path.insert(0, HYB)
_spec.loader.exec_module(hyb_examples)

LAMS = 0.1 * 50 ** (np.arange(16) / 15)


def ls_runs(Q, starts):
    best, ev = (None, np.inf), 0
    for x0 in starts:
        x, e, k = local_search(np.asarray(x0), Q)
        ev += k / Q.shape[0] + 1
        if e < best[1]:
            best = (x, e)
    return best, ev


def run(n, seed):
    mu, sigma = portfolio_data(n, seed=seed)
    Qs = [portfolio_qubo(mu, sigma, n // 4, lam) for lam in LAMS]
    rng = np.random.default_rng(seed + 7)
    out = {}

    t = time.perf_counter()
    rs = [tci_minimize_bits(Q, seed=seed) for Q in Qs]
    out["TCI-bits/λ"] = {"E": [r["energy"] for r in rs], "time": time.perf_counter() - t,
                         "evals": sum(r["n_energy_evals"] for r in rs)}
    t = time.perf_counter()
    r = tci_sweep_bits(Qs, seed=seed)
    out["TCI-sweep"] = {"E": list(r["energy"]), "time": time.perf_counter() - t,
                        "evals": r["n_energy_evals"], "rank": r["max_rank"]}
    t = time.perf_counter()
    sa = [hyb_examples.classical_sa(Q, seed=seed) for Q in Qs]
    out["SA/λ"] = {"E": [qubo_energy(s[0], Q) for s, Q in zip(sa, Qs)],
                   "time": time.perf_counter() - t, "evals": int(sum(s[2] for s in sa))}
    t = time.perf_counter()
    E, ev = [], 0
    for Q in Qs:
        (x, e), k = ls_runs(Q, rng.integers(0, 2, (100, n)))
        E.append(e); ev += k
    out["LS/λ"] = {"E": E, "time": time.perf_counter() - t, "evals": ev}
    t = time.perf_counter()
    E, ev, prev = [], 0, None
    for Q in Qs:
        starts = list(rng.integers(0, 2, (10, n))) + ([prev] if prev is not None else [])
        (x, e), k = ls_runs(Q, starts)
        E.append(e); ev += k; prev = x
    out["LS-cont"] = {"E": E, "time": time.perf_counter() - t, "evals": ev}
    budget = out["TCI-sweep"]["time"] / len(Qs)
    t = time.perf_counter()
    E, ev = [], 0
    for Q in Qs:
        t0, best = time.perf_counter(), np.inf
        while time.perf_counter() - t0 < budget:
            (x, e), k = ls_runs(Q, [rng.integers(0, 2, n)])
            best = min(best, e); ev += k
        E.append(best)
    out["LS-matched"] = {"E": E, "time": time.perf_counter() - t, "evals": ev}

    if n <= 22:
        ref, exact = np.array([brute_force(Q)[1] for Q in Qs]), True
    else:
        ref, exact = np.min([v["E"] for v in out.values()], axis=0), False
    for v in out.values():
        Ev = np.array(v["E"])
        v["hits"] = int(np.sum(Ev <= ref + 1e-9))
        v["gap"] = float(np.mean((Ev - ref) / np.abs(ref)))
    return {"n": n, "seed": seed, "exact": exact, "ref": ref.tolist(), "methods": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="results/frontier.json")
    a = ap.parse_args()
    rows = []
    for n in a.sizes:
        for s in range(a.seeds):
            r = run(n, s)
            rows.append(r)
            print(f"n={n} seed={s} " + " | ".join(
                f"{k}: {v['hits']}/16 {v['evals']:.0f}ev {v['time']:.1f}s"
                for k, v in r["methods"].items()), flush=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print("\nn   method        hits      mean gap     evals/frontier   time/frontier (s)")
    for n in a.sizes:
        rr = [r for r in rows if r["n"] == n]
        for k in rr[0]["methods"]:
            h = sum(r["methods"][k]["hits"] for r in rr)
            g = np.mean([r["methods"][k]["gap"] for r in rr])
            ev = np.mean([r["methods"][k]["evals"] for r in rr])
            tm = np.mean([r["methods"][k]["time"] for r in rr])
            print(f"{n:<3} {k:12s} {h:3d}/{16 * len(rr)}  {100 * g:8.3f} %  {ev:14,.0f}  {tm:10.1f}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
