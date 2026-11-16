"""
Benchmark 6: where in the depth-p sweep pipeline does TCI pay?

The pipeline (``sweep_library``) has two stages that can use TCI or not; the polish is
COBYLA everywhere.  Every combination is run on the benchmark-5 families (n = 10,
4 families x 3 instances x 16 values of t) at depth p:

  depth-1 candidates   off          no lifted candidates
                       TT           TCI sweep TT over (gamma, beta, t) at depth 1, its
                                    slice minima lifted to depth p (INTERP rule)
  library seeding      none         no seeding (library = optima found along the sweep)
                       restarts     4 random restarts per anchor (every other t)
                       restarts8    8 per anchor (about the same budget as "tci")
                       tci          quantics TCI per anchor (bond dim 8) as a global
                                    sampler; its 4 best distinct points are polished
                       tci-small    the same with bond dim 5 (about the restarts' budget)
                       tci-sweep    one depth-p TCI over (angles, t) (bond dim 15, not
                                    converged); its 4 lowest slice minima per anchor

OK = within 1e-3 of the best found by any configuration or by a 20-restart-per-t
reference.  "source" counts which kind of start produced each final optimum.

Usage:  python benchmarks/bench_tci_placement.py --p 2
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from bench_sweep import family  # noqa: E402
from tciqaoa import optimize_restarts  # noqa: E402
from tciqaoa.sweep import SweepFamily, sweep_library, tci_sweep  # noqa: E402

SEEDING = {"none": dict(seeding=None), "restarts": dict(seeding="restarts"),
           "restarts8": dict(seeding="restarts", n_restarts=8),
           "tci": dict(seeding="tci"), "tci-small": dict(seeding="tci", tci_bond_dim=5),
           "tci-sweep": dict(seeding="tci-sweep")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["mix", "field", "penalty", "risk"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--out", default="results/tci_placement.json")
    a = ap.parse_args()
    ts = np.arange(0, 32, 2) / 32
    rows = []
    for name in a.families:
        for seed in range(a.seeds):
            fam = SweepFamily(family(name, seed))
            tt = tci_sweep(fam, 1)
            res = {}
            for use_tt in (False, True):
                for sname, kw in SEEDING.items():
                    lb = sweep_library(tt if use_tt else None, fam, ts, a.p, **kw)
                    lb["n_evals"] += tt["n_evals"] if use_tt else 0
                    res[("TT" if use_tt else "off", sname)] = lb
            rq = [fam.qaoa(t, a.p) for t in ts]
            ref = np.array([optimize_restarts(q, n_starts=20, seed=99)["cost"] for q in rq])
            best = np.minimum(ref, np.min([r["costs"] for r in res.values()], axis=0))
            row = {"family": name, "seed": seed, "p": a.p, "tt_build": tt["n_evals"],
                   "best": best.tolist(),
                   "ref": {"ok": int(np.sum(ref <= best + 1e-3)),
                           "worst": float(np.max(ref - best)),
                           "n_evals": int(sum(q.n_evals for q in rq))},
                   "configs": {}}
            for (tt_flag, sname), r in res.items():
                row["configs"][f"{tt_flag}|{sname}"] = {
                    "ok": int(np.sum(r["costs"] <= best + 1e-3)),
                    "worst": float(np.max(r["costs"] - best)),
                    "n_evals": int(r["n_evals"]), "n_seed_evals": int(r["n_seed_evals"]),
                    "source": dict(Counter(r["source"])), "costs": r["costs"].tolist()}
            rows.append(row)
            print(f"{name:8s} seed={seed} " + " | ".join(
                f"{k}: {v['ok']}/16 {v['n_evals']}" for k, v in row["configs"].items())
                + f" | ref {row['ref']['ok']}/16", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)

    print("\nconfig                 OK        worst   evals/instance  sources")
    for k in rows[0]["configs"]:
        ok = sum(r["configs"][k]["ok"] for r in rows)
        w = max(r["configs"][k]["worst"] for r in rows)
        ev = np.mean([r["configs"][k]["n_evals"] for r in rows])
        src = Counter()
        for r in rows:
            src.update(r["configs"][k]["source"])
        print(f"{k:22s} {ok:3d}/{16 * len(rows)}  {w:6.3f}  {ev:10,.0f}     {dict(src)}")
    ok = sum(r["ref"]["ok"] for r in rows)
    print(f"{'ref 20 restarts/t':22s} {ok:3d}/{16 * len(rows)}  "
          f"{max(r['ref']['worst'] for r in rows):6.3f}  "
          f"{np.mean([r['ref']['n_evals'] for r in rows]):10,.0f}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
