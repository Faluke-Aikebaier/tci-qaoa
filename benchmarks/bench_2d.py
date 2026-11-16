"""
Benchmark 11: two-parameter sweeps.  One tensor train over (angles, t1, t2).

Families (n = 10), (t1, t2) on a 32 x 32 grid:
  port2      portfolio selection: risk aversion 0.2 + 6 t1 and cardinality-penalty
             weight 0.05 + 3 t2 (benchmark 5's *risk* and *penalty* families combined)
  mixfield   MaxCut blended into a random QUBO (t1, as *mix*) plus a uniform field
             of strength 3 t2 (in units of the largest |L| entry, as *field*)
Each instance is solved on nt x nt evenly spaced grid points, nt = 4, 8, 16 (16, 64 and
256 points), visited in snake order (``SweepFamily2D.points``).  The depth-1 TT is
built once per instance and shared, but every run is charged its full cost.  Exact <C>.

Methods (p = 1):  tt (one 2-D sweep TT, its slice minima polished), interp, ref
Methods (p >= 2): chain, chain-noTCI (seed 1), the same seeded at 16 points only
                  (chain-a16, chain-noTCI-a16: ``n_anchors=16``), interp, ref
Scored by score_2d.py against the best exact cost found at each point by any method
at any nt.

Usage:  python benchmarks/bench_2d.py --p 1 --families port2 --seeds 0
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tciqaoa import optimize_interp, optimize_restarts  # noqa: E402
from tciqaoa.problems import random_graph, random_qubo  # noqa: E402
from tciqaoa.sweep import SweepFamily2D, sweep_chain, tci_sweep  # noqa: E402


def family2(name, seed, n=10):
    rng = np.random.default_rng(seed)
    A = random_graph(n, 0.4, seed=seed)
    L = np.diag(A.sum(1)) - A
    if name == "mixfield":
        Qr = random_qubo(n, seed=seed + 500)
        Ln = L / np.abs(L).max()
        return lambda t1, t2: (1 - t1) * (-Ln) + t1 * Qr / 2 + t2 * 3 * np.eye(n) / np.abs(L).max()
    if name == "port2":
        mu = rng.uniform(0.05, 0.2, n)
        M = rng.standard_normal((n, n))
        S = M @ M.T / n
        S = S / np.abs(S).max()
        R = np.diag(mu / np.abs(mu).max())
        k = round(0.4 * n)
        P = np.ones((n, n)) - 2 * k * np.eye(n)
        return lambda t1, t2: (0.2 + 6 * t1) * S - R + (0.05 + 3 * t2) * P
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--nts", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--families", nargs="+", default=["port2", "mixfield"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/sweep2d_n{a.n}_p{a.p}.json"
    rows = []
    for name in a.families:
        for seed in a.seeds:
            fam = SweepFamily2D(family2(name, seed, a.n), R=5)
            tt = tci_sweep(fam, 1)
            print(f"{name} seed={seed} TT rank {tt['tt_max_rank']}, {tt['n_evals']:,} runs", flush=True)
            for nt in a.nts:
                ts = fam.points(nt)
                res = {}
                if a.p == 1:
                    r = sweep_chain(fam, ts, 1, tt=tt)
                    res["tt"] = (r["costs"], r["n_evals"])
                else:
                    r = sweep_chain(fam, ts, a.p, tt=tt, extra_restarts=0)
                    res["chain"] = (r["costs"], r["n_evals"])
                    r = sweep_chain(fam, ts, a.p, extra_restarts=0, use_tci=False, seed=1)
                    res["chain-noTCI"] = (r["costs"], r["n_evals"])
                    if nt * nt > 16:
                        r = sweep_chain(fam, ts, a.p, tt=tt, extra_restarts=0, n_anchors=16)
                        res["chain-a16"] = (r["costs"], r["n_evals"])
                        r = sweep_chain(fam, ts, a.p, extra_restarts=0, use_tci=False, seed=1,
                                        n_anchors=16)
                        res["chain-noTCI-a16"] = (r["costs"], r["n_evals"])
                it = [optimize_interp(lambda k, t=t: fam.qaoa(t, k), a.p, seed=seed) for t in ts]
                res["interp"] = ([x["cost"] for x in it], sum(x["n_evals"] for x in it))
                qs = [fam.qaoa(t, a.p) for t in ts]
                rf = [optimize_restarts(q, n_starts=20, seed=99) for q in qs]
                res["ref"] = ([x["cost"] for x in rf], sum(q.n_evals for q in qs))
                rows.append({"family": name, "seed": seed, "nt": nt, "ts": list(ts),
                             "params": [fam.decode(t) for t in ts],
                             "tt_evals": tt["n_evals"], "tt_rank": tt["tt_max_rank"],
                             "methods": {m: {"costs": np.asarray(c, dtype=float).tolist(),
                                             "n_evals": int(e)} for m, (c, e) in res.items()}})
                print(f"  {nt}x{nt} " + " | ".join(
                    f"{m}: {np.mean(c):.4f} {e / len(ts):,.0f}/pt" for m, (c, e) in res.items()), flush=True)
    with open(out, "w") as f:
        json.dump(rows, f)
    print("saved", out)


if __name__ == "__main__":
    main()
