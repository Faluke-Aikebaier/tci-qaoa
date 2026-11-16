"""
Benchmark 9: the sweep pipeline at larger n.

The benchmark-5 families (mix, field, penalty, risk; 3 instances each, 16 values of t)
at n qubits (budget k = round(0.4 n) in the portfolio families), depth p.  Methods:
  chain     sweep_chain(..., extra_restarts=0)   (deterministic; INTERP start at depth p)
  chain-noI the same without the INTERP start         (only with --ablate)
  chain+8   sweep_chain(...) default, library seeds F..F+S-1 (--lib-seed-from F,
            --lib-seeds S; S = 0 skips it)
  interp    INTERP from scratch at every t
  ref       20 random restarts at every t        (only with --ref)
Ablation and matched-budget controls (only when named in --methods):
  chain-noTCI     chain without TCI: no depth-1 TT, depth 2 seeded by random restarts
                  (seed --lib-seed-from)
  chain+8-noTCI   the same with the 8 wide restarts (library seed --lib-seed-from)
  chain-hyb       one run with both: the depth-1 TT, and depth 2 seeded by the TCI
                  sampler and the same random restarts as chain-noTCI (seed --lib-seed-from)
  chain+8-hyb     the same with the 8 wide restarts
  interp-x8       INTERP with seeds 0..7, best at each t (~chain+8's budget)
  ref40           40 random restarts at every t (~chain+8's budget; its first 20
                  starts are ref's)
OK = within 1e-3 of the best found by any method in this run, and in the runs given by
--prev (earlier result files on the same instances).  --methods runs a subset (e.g.
``--methods ref --prev results/large_n14_p4.json`` adds only the restart reference).

Usage:  python benchmarks/bench_large_n.py --n 12 --p 3 --ref
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
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--families", nargs="+", default=["mix", "field", "penalty", "risk"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lib-seeds", type=int, default=1)
    ap.add_argument("--lib-seed-from", type=int, default=1)
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--prev", nargs="*", default=[])
    ap.add_argument("--methods", nargs="+", default=["chain", "chain+8", "interp", "ref"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/large_n{a.n}_p{a.p}.json"
    ts = np.arange(0, 32, 2) / 32
    prev_best = {}
    for fn in a.prev:
        for r in json.load(open(fn)):
            k = (r["family"], r["seed"])
            prev_best[k] = np.minimum(prev_best.get(k, np.inf), r["best"])
    rows = []
    for name in a.families:
        for seed in range(a.seeds):
            fam = SweepFamily(family(name, seed, a.n))
            use = set(a.methods)
            tt = tci_sweep(fam, 1) if use & {"chain", "chain+8", "chain-hyb", "chain+8-hyb"} or a.ablate else {"tt_max_rank": None}
            cost, evals = {}, {}
            if "chain" in use:
                r = sweep_chain(fam, ts, a.p, tt=tt, extra_restarts=0)
                cost["chain"], evals["chain"] = [r["costs"]], [r["n_evals"]]
            if a.ablate:
                r = sweep_chain(fam, ts, a.p, tt=tt, extra_restarts=0, interp_start=False)
                cost["chain-noI"], evals["chain-noI"] = [r["costs"]], [r["n_evals"]]
            if a.lib_seeds and "chain+8" in use:
                rs = [sweep_chain(fam, ts, a.p, tt=tt, seed=s) for s in range(a.lib_seed_from, a.lib_seed_from + a.lib_seeds)]
                cost["chain+8"] = [r["costs"] for r in rs]
                evals["chain+8"] = [r["n_evals"] for r in rs]
            if "chain-noTCI" in use:
                r = sweep_chain(fam, ts, a.p, extra_restarts=0, use_tci=False, seed=a.lib_seed_from)
                cost["chain-noTCI"], evals["chain-noTCI"] = [r["costs"]], [r["n_evals"]]
            if "chain-hyb" in use:
                r = sweep_chain(fam, ts, a.p, tt=tt, extra_restarts=0, use_tci="hybrid", seed=a.lib_seed_from)
                cost["chain-hyb"], evals["chain-hyb"] = [r["costs"]], [r["n_evals"]]
            if "chain+8-hyb" in use:
                r = sweep_chain(fam, ts, a.p, tt=tt, use_tci="hybrid", seed=a.lib_seed_from)
                cost["chain+8-hyb"], evals["chain+8-hyb"] = [r["costs"]], [r["n_evals"]]
            if "chain+8-noTCI" in use:
                r = sweep_chain(fam, ts, a.p, seed=a.lib_seed_from, use_tci=False)
                cost["chain+8-noTCI"], evals["chain+8-noTCI"] = [r["costs"]], [r["n_evals"]]
            if "interp-x8" in use:
                its = [[optimize_interp(lambda k, t=t: fam.qaoa(t, k), a.p, seed=s) for t in ts]
                       for s in range(8)]
                cost["interp-x8"] = [np.min([[r["cost"] for r in it] for it in its], axis=0)]
                evals["interp-x8"] = [sum(r["n_evals"] for it in its for r in it)]
            if "ref40" in use:
                qs = [fam.qaoa(t, a.p) for t in ts]
                rf = [optimize_restarts(q, n_starts=40, seed=99) for q in qs]
                cost["ref40"] = [np.array([r["cost"] for r in rf])]
                evals["ref40"] = [sum(q.n_evals for q in qs)]
            if "interp" in use:
                it = [optimize_interp(lambda k, t=t: fam.qaoa(t, k), a.p, seed=seed) for t in ts]
                cost["interp"] = [np.array([r["cost"] for r in it])]
                evals["interp"] = [sum(r["n_evals"] for r in it)]
            if a.ref and "ref" in use:
                qs = [fam.qaoa(t, a.p) for t in ts]
                rf = [optimize_restarts(q, n_starts=20, seed=99) for q in qs]
                cost["ref"] = [np.array([r["cost"] for r in rf])]
                evals["ref"] = [sum(q.n_evals for q in qs)]
            best = np.min([c for v in cost.values() for c in v], axis=0)
            best = np.minimum(best, prev_best.get((name, seed), np.inf))
            row = {"family": name, "seed": seed, "n": a.n, "p": a.p, "tt_rank": tt["tt_max_rank"],
                   "best": best.tolist(), "methods": {
                       k: {"ok": [int(np.sum(c <= best + 1e-3)) for c in cost[k]],
                           "worst": [float(np.max(c - best)) for c in cost[k]],
                           "n_evals": [int(e) for e in evals[k]],
                           "costs": [np.asarray(c).tolist() for c in cost[k]]} for k in cost}}
            rows.append(row)
            print(f"{name:8s} seed={seed} " + " | ".join(
                f"{k}: {v['ok']}/16 {np.mean(v['n_evals']):,.0f}" for k, v in row["methods"].items()),
                flush=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nn = {a.n}, p = {a.p}")
    print(f"method     OK / {16 * len(rows)} per seed     mean    worst   runs/instance")
    for k in rows[0]["methods"]:
        m = len(rows[0]["methods"][k]["ok"])
        oks = [sum(r["methods"][k]["ok"][i] for r in rows) for i in range(m)]
        w = max(max(r["methods"][k]["worst"]) for r in rows)
        ev = np.mean([e for r in rows for e in r["methods"][k]["n_evals"]])
        print(f"{k:10s} {str(oks):18s} {np.mean(oks):6.1f}  {w:6.3f}  {ev:12,.0f}")
    for fn in a.families:
        rr = [r for r in rows if r["family"] == fn]
        print(fn, {k: sum(r["methods"][k]["ok"][0] for r in rr) for k in rr[0]["methods"]})
    print("saved", out)


if __name__ == "__main__":
    main()
