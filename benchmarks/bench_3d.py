"""
Benchmark 12: three-parameter sweeps.  One tensor train over (angles, t1, t2, t3).

Families (n = 10), (t1, t2, t3) on a 32 x 32 x 32 grid:
  port3      portfolio selection: risk aversion 0.2 + 6 t1, cardinality-penalty weight
             0.05 + 3 t2 and return weight 0.5 + 1.5 t3 (benchmark 11's *port2* plus a
             third knob)
  mixfield3  benchmark 11's *mixfield* (MaxCut blended into a random QUBO by t1, uniform
             field 3 t2) plus a random field h_i in [-1, 1] of strength 2 t3 (both
             fields in units of the largest |L| entry)
Each instance is solved on nt x nt x nt evenly spaced grid points, nt = 4, 8 (64 and
512 points), visited in snake order (``SweepFamilyND.points``).  The depth-1 TT is
built once per instance and shared, but every run is charged its full cost.  Exact <C>.

Methods (p = 1):  tt (one 3-D sweep TT, its slice minima polished), interp, ref
Methods (p >= 2): chain, chain-noTCI (seed 1), the same seeded at 16 points only
                  (chain-a16, chain-noTCI-a16: ``n_anchors=16``), interp, ref
Scored by ``score_2d.py --prefix sweep3d`` against the best exact cost found at each
point by any method at any nt.

Usage:  python benchmarks/bench_3d.py --p 1 --families port3 --seeds 0
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tciqaoa import optimize_interp, optimize_restarts  # noqa: E402
from tciqaoa.problems import random_graph, random_qubo  # noqa: E402
from tciqaoa.sweep import SweepFamilyND, sweep_chain, tci_sweep  # noqa: E402


def family3(name, seed, n=10):
    rng = np.random.default_rng(seed)
    A = random_graph(n, 0.4, seed=seed)
    L = np.diag(A.sum(1)) - A
    if name == "mixfield3":
        Qr = random_qubo(n, seed=seed + 500)
        Ln = L / np.abs(L).max()
        h = np.random.default_rng(seed + 900).uniform(-1, 1, n)
        s = np.abs(L).max()
        return lambda t1, t2, t3: ((1 - t1) * (-Ln) + t1 * Qr / 2 + t2 * 3 * np.eye(n) / s
                                   + t3 * 2 * np.diag(h) / s)
    if name == "port3":
        mu = rng.uniform(0.05, 0.2, n)
        M = rng.standard_normal((n, n))
        S = M @ M.T / n
        S = S / np.abs(S).max()
        R = np.diag(mu / np.abs(mu).max())
        k = round(0.4 * n)
        P = np.ones((n, n)) - 2 * k * np.eye(n)
        return lambda t1, t2, t3: (0.2 + 6 * t1) * S - (0.5 + 1.5 * t3) * R + (0.05 + 3 * t2) * P
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--nts", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--families", nargs="+", default=["port3", "mixfield3"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/sweep3d_n{a.n}_p{a.p}.json"
    rows = []
    for name in a.families:
        for seed in a.seeds:
            fam = SweepFamilyND(family3(name, seed, a.n), 3, R=5)
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
                    if len(ts) > 16:
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
                rows.append({"family": name, "seed": seed, "nt": nt, "n_params": 3, "ts": list(ts),
                             "params": [fam.decode(t) for t in ts],
                             "tt_evals": tt["n_evals"], "tt_rank": tt["tt_max_rank"],
                             "methods": {m: {"costs": np.asarray(c, dtype=float).tolist(),
                                             "n_evals": int(e)} for m, (c, e) in res.items()}})
                print(f"  {nt}^3 " + " | ".join(
                    f"{m}: {np.mean(c):.4f} {e / len(ts):,.0f}/pt" for m, (c, e) in res.items()), flush=True)
    with open(out, "w") as f:
        json.dump(rows, f)
    print("saved", out)


if __name__ == "__main__":
    main()
