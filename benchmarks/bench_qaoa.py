"""
Benchmark 1: real (statevector) QAOA, and three ways to choose its 2p angles.

For each instance and depth p:
  restarts : 10 random starts + COBYLA
  interp   : INTERP warm start (Zhou et al. 2020)
  tci      : quantics TCI over the angle box + COBYLA polish (xfacpy)

Reported: approximation ratio (1 = ground state), ground-state probability, whether
1000 shots contain the optimum, circuit evaluations used, wall time, TT rank and
final pivot error of the learned landscape.

Usage:  python benchmarks/bench_qaoa.py --n 10 14 --p 1 2 3 --seeds 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tciqaoa.problems import maxcut_qubo, random_graph, random_qubo, all_energies
from tciqaoa.qaoa import QAOA
from tciqaoa.angles import optimize_restarts, optimize_interp, optimize_tci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[10, 14])
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results/qaoa.json")
    a = ap.parse_args()
    rows = []
    for kind in ["maxcut", "qubo"]:
        for n in a.n:
            for s in range(a.seeds):
                Q = maxcut_qubo(random_graph(n, 0.4, seed=s)) if kind == "maxcut" \
                    else random_qubo(n, 0.5, seed=s)
                E = all_energies(Q)
                for p in a.p:
                    for name in ["restarts", "interp", "tci"]:
                        q = QAOA(Q, p, energies=E)
                        t = time.perf_counter()
                        if name == "restarts":
                            r = optimize_restarts(q, n_starts=10, seed=s)
                        elif name == "interp":
                            r = optimize_interp(lambda k: QAOA(Q, k, energies=E), p, seed=s)
                        else:
                            r = optimize_tci(q, seed=s)
                        dt = time.perf_counter() - t
                        rep = q.report(r["angles"], shots=1000, rng=s)
                        row = {"kind": kind, "n": n, "seed": s, "p": p, "method": name,
                               "approx_ratio": rep["approx_ratio"], "p_ground": rep["p_ground"],
                               "shots_hit": bool(rep["best_sampled_energy"] <= q.E_min + 1e-9),
                               "n_evals": r["n_evals"], "time": dt,
                               "angles": [float(v) for v in r["angles"]]}
                        for k in ("tt_max_rank", "pivot_error", "n_tci_evals"):
                            if k in r:
                                row[k] = r[k]
                        rows.append(row)
                        print(f"{kind:6s} n={n} s={s} p={p} {name:8s} r={row['approx_ratio']:.4f} "
                              f"Pgs={row['p_ground']:.3f} evals={row['n_evals']} {dt:.1f}s", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
