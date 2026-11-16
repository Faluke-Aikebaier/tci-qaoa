"""
Benchmark 8: the recommended sweep pipeline (``sweep_chain``) at depth p = 4.

Same 12 instances x 16 values of t as benchmarks 5 and 6 (n = 10).  Methods:
  chain          sweep_chain(..., extra_restarts=0): p = 1 TT -> p = 2 (TCI seeding)
                 -> p = 3 -> p = 4, each depth lifted from the one below (deterministic)
  chain+8        sweep_chain(...) default: plus 8 wide restarts per anchor with reserved
                 shortlist slots at p = 3 and p = 4 (library seeds 1..S)
  interp         INTERP from scratch at every t (p = 1 -> 2 -> 3 -> 4, per t)
  ref            20 random restarts at every t
The depth-1 TT and the p = 2 stage are shared by the chain runs (both are
deterministic), but each run is charged their full cost.  OK = within 1e-3 of the best
found by any method.  Also reports the depth gain: mean cost at p = 4 minus p = 3 (from
the chain's own p = 3 stage).

Usage:  python benchmarks/bench_p4.py
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
from tciqaoa.sweep import SweepFamily, sweep_library, tci_sweep  # noqa: E402


def chain_from(p2, fam, ts, p, extra, seed):
    """Depths 3..p of sweep_chain on a shared p = 2 result."""
    res, ev, by = p2, 0, {}
    for d in range(3, p + 1):
        kw = (dict(seeding="restarts", n_restarts=extra, n_seed_lib=2, gamma_other=4.5)
              if extra else dict(seeding=None))
        res = sweep_library(None, fam, ts, d, lower=res, seed=seed, **kw)
        ev += res["n_evals"]
        by[d] = res
    return by, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["mix", "field", "penalty", "risk"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lib-seeds", type=int, default=3)
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--out", default="results/p4.json")
    a = ap.parse_args()
    ts = np.arange(0, 32, 2) / 32
    rows = []
    for name in a.families:
        for seed in range(a.seeds):
            fam = SweepFamily(family(name, seed))
            tt = tci_sweep(fam, 1)
            p2 = sweep_library(tt, fam, ts, 2, seeding="tci")
            base = tt["n_evals"] + p2["n_evals"]
            runs = {}
            by, ev = chain_from(p2, fam, ts, a.p, 0, 1)
            runs["chain"] = [(by, base + ev)]
            runs["chain+8"] = [chain_from(p2, fam, ts, a.p, 8, s) for s in range(1, a.lib_seeds + 1)]
            runs["chain+8"] = [(b, base + e) for b, e in runs["chain+8"]]
            it = [optimize_interp(lambda k, t=t: fam.qaoa(t, k), a.p, seed=seed) for t in ts]
            qs = [fam.qaoa(t, a.p) for t in ts]
            rf = [optimize_restarts(q, n_starts=20, seed=99) for q in qs]
            cost = {k: [np.array(b[a.p]["costs"]) for b, _ in v] for k, v in runs.items()}
            cost["interp"] = [np.array([r["cost"] for r in it])]
            cost["ref"] = [np.array([r["cost"] for r in rf])]
            evals = {k: [e for _, e in v] for k, v in runs.items()}
            evals["interp"] = [sum(r["n_evals"] for r in it)]
            evals["ref"] = [sum(q.n_evals for q in qs)]
            best = np.min([c for v in cost.values() for c in v], axis=0)
            row = {"family": name, "seed": seed, "best": best.tolist(), "methods": {}}
            for k in cost:
                row["methods"][k] = {
                    "ok": [int(np.sum(c <= best + 1e-3)) for c in cost[k]],
                    "worst": [float(np.max(c - best)) for c in cost[k]],
                    "n_evals": evals[k], "costs": [c.tolist() for c in cost[k]]}
            row["depth_gain"] = {
                k: float(np.mean([np.mean(b[a.p]["costs"] - b[a.p - 1]["costs"]) for b, _ in v]))
                for k, v in runs.items()}
            rows.append(row)
            print(f"{name:8s} seed={seed} " + " | ".join(
                f"{k}: {v['ok']}/16 {np.mean(v['n_evals']):,.0f}" for k, v in row["methods"].items())
                + f" | gain p{a.p - 1}->p{a.p} {row['depth_gain']['chain']:.3f}", flush=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)

    n_pts = 16 * len(rows)
    print(f"\nmethod     OK / {n_pts} per library seed   mean    worst   runs/instance")
    for k in rows[0]["methods"]:
        n = len(rows[0]["methods"][k]["ok"])
        oks = [sum(r["methods"][k]["ok"][i] for r in rows) for i in range(n)]
        w = max(max(r["methods"][k]["worst"]) for r in rows)
        ev = np.mean([e for r in rows for e in r["methods"][k]["n_evals"]])
        print(f"{k:10s} {str(oks):24s} {np.mean(oks):6.1f}  {w:6.3f}  {ev:12,.0f}")
    for fam_name in a.families:
        rr = [r for r in rows if r["family"] == fam_name]
        print(fam_name, {k: [sum(r["methods"][k]["ok"][i] for r in rr)
                             for i in range(len(rr[0]["methods"][k]["ok"]))]
                         for k in rr[0]["methods"]},
              "gain", round(float(np.mean([r["depth_gain"]["chain"] for r in rr])), 3))
    print("saved", a.out)


if __name__ == "__main__":
    main()
