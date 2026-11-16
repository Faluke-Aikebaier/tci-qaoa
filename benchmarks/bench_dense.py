"""
Benchmark 10: denser sweeps.  Does the TT's shared build cost pay off as the number of
sweep points grows?

The depth-1 sweep TT is built once over (angles x t) and serves every t, so its cost per
sweep point should fall as the sweep gets denser, while INTERP and restarts cost the
same per point.  Here the t grid has 64 points (``SweepFamily(..., R_t=6)``; the angle
grids use 6 bits as well) and each instance is solved on nt = 8, 16, 32 and 64 equally
spaced points of it, t = k / nt.  The TT is built once per instance and shared, but every
run is charged its full cost.  Exact <C> (no shot noise).

Methods (p = 1):  tt (sweep_chain p = 1), interp (best of 5 restarts), ref (20 restarts)
Methods (p >= 2): chain, chain-noTCI (seed 1), the same two seeded at 8 fixed t only
                  (chain-a8, chain-noTCI-a8: ``n_anchors=8``), interp, ref
Scored by score_dense.py against the best exact cost found at each t by any method at
any nt.

Usage:  python benchmarks/bench_dense.py --p 2 --families mix
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from bench_sweep import family  # noqa: E402
from tciqaoa import optimize_interp, optimize_restarts  # noqa: E402
from tciqaoa.sweep import SweepFamily, sweep_chain, tci_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--nts", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--families", nargs="+", default=["mix", "field", "penalty", "risk"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/dense_n{a.n}_p{a.p}.json"
    rows = []
    for name in a.families:
        for seed in range(a.seeds):
            fam = SweepFamily(family(name, seed, a.n), R_t=6)
            tt = tci_sweep(fam, 1)
            print(f"{name} seed={seed} TT rank {tt['tt_max_rank']}, {tt['n_evals']:,} runs", flush=True)
            for nt in a.nts:
                ts = np.arange(nt) / nt
                res = {}
                if a.p == 1:
                    r = sweep_chain(fam, ts, 1, tt=tt)
                    res["tt"] = (r["costs"], r["n_evals"])
                else:
                    r = sweep_chain(fam, ts, a.p, tt=tt, extra_restarts=0)
                    res["chain"] = (r["costs"], r["n_evals"])
                    r = sweep_chain(fam, ts, a.p, extra_restarts=0, use_tci=False, seed=1)
                    res["chain-noTCI"] = (r["costs"], r["n_evals"])
                    if nt > 8:
                        r = sweep_chain(fam, ts, a.p, tt=tt, extra_restarts=0, n_anchors=8)
                        res["chain-a8"] = (r["costs"], r["n_evals"])
                        r = sweep_chain(fam, ts, a.p, extra_restarts=0, use_tci=False, seed=1, n_anchors=8)
                        res["chain-noTCI-a8"] = (r["costs"], r["n_evals"])
                it = [optimize_interp(lambda k, t=t: fam.qaoa(t, k), a.p, seed=seed) for t in ts]
                res["interp"] = ([x["cost"] for x in it], sum(x["n_evals"] for x in it))
                qs = [fam.qaoa(t, a.p) for t in ts]
                rf = [optimize_restarts(q, n_starts=20, seed=99) for q in qs]
                res["ref"] = ([x["cost"] for x in rf], sum(q.n_evals for q in qs))
                rows.append({"family": name, "seed": seed, "nt": nt, "ts": ts.tolist(),
                             "tt_evals": tt["n_evals"], "tt_rank": tt["tt_max_rank"],
                             "methods": {m: {"costs": np.asarray(c, dtype=float).tolist(),
                                             "n_evals": int(e)} for m, (c, e) in res.items()}})
                print(f"  nt={nt:3d} " + " | ".join(f"{m}: {e / nt:,.0f}/pt" for m, (c, e) in res.items()),
                      flush=True)
    with open(out, "w") as f:
        json.dump(rows, f)
    print("saved", out)


if __name__ == "__main__":
    main()
