"""
Benchmark 6d: extra depth-3 seeding on top of the depth chain.

The depth chain (benchmark 6c: p = 1 TT -> p = 2 library with TCI seeding -> p = 3 by
lifting) gets 158/192.  Most remaining misses are depth-3-only basins at the far end of
the *field* sweep, found only by lucky p = 3 restarts.  Adding p = 3 restarts to the
chain naively made it worse (the seeds displaced lifted basins from the 3-member
shortlist), so here the seeds get their own shortlist slots (``n_seed_lib``).

Configurations at p = 3 (all on the same TCI-seeded p = 2 run per instance):
  chain                 lift only (benchmark 6c's best)
  +restarts4 mixed      4 p = 3 restarts per anchor, competing for the shortlist
  +restarts4 reserved   the same, 2 reserved slots for seeds
  +restarts8 reserved   8 restarts per anchor, 2 reserved slots
  +tci reserved         TCI sampler per anchor (chi = 8), 2 reserved slots
  ... wide              the same with later gammas sampled in [-4.5, 4.5] instead of
                        [-3, 3] (the missing field basins sit at |gamma_k| ~ 3.2-4.1)
Restart rows run library seeds 1..S.  Scored against the common best of benchmarks
6, 6b, 6c and these runs.  Costs include the TT build and the p = 2 run.

Usage:  python benchmarks/bench_seed_reserve.py --families field
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
from tciqaoa.sweep import SweepFamily, sweep_library, tci_sweep  # noqa: E402

CONFIGS = {  # name: (kwargs, randomised)
    "chain": (dict(seeding=None), False),
    "+restarts4 mixed": (dict(seeding="restarts"), True),
    "+restarts4 reserved": (dict(seeding="restarts", n_seed_lib=2), True),
    "+restarts8 reserved": (dict(seeding="restarts", n_restarts=8, n_seed_lib=2), True),
    "+tci reserved": (dict(seeding="tci", n_seed_lib=2), False),
    "+restarts4 wide": (dict(seeding="restarts", n_seed_lib=2, gamma_other=4.5), True),
    "+restarts8 wide": (dict(seeding="restarts", n_restarts=8, n_seed_lib=2, gamma_other=4.5), True),
    "+tci wide": (dict(seeding="tci", n_seed_lib=2, gamma_other=4.5), False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["mix", "field", "penalty", "risk"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lib-seeds", type=int, default=3)
    ap.add_argument("--chain", default="results/depth_chain_p3.json")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--out", default="results/seed_reserve_p3.json")
    a = ap.parse_args()
    prev_best = {k: np.array(v) for k, v in json.load(open(a.chain))["best"].items()}
    ts = np.arange(0, 32, 2) / 32
    runs = []
    for name in a.families:
        for seed in range(a.seeds):
            fam = SweepFamily(family(name, seed))
            tt = tci_sweep(fam, 1)
            p2 = sweep_library(tt, fam, ts, 2, seeding="tci")
            base = tt["n_evals"] + p2["n_evals"]
            for cname in a.configs:
                kw, rnd = CONFIGS[cname]
                for s in (range(1, a.lib_seeds + 1) if rnd else [1]):
                    r = sweep_library(None, fam, ts, 3, lower=p2, seed=s, **kw)
                    runs.append({"family": name, "seed": seed, "config": cname, "lib_seed": s,
                                 "costs": r["costs"].tolist(), "n_evals": int(base + r["n_evals"]),
                                 "source": dict(Counter(r["source"]))})
            print(name, seed, "done", flush=True)

    key = lambda r: f"{r['family']}|{r['seed']}"  # noqa: E731
    best = {}
    for r in runs:
        b = best.get(key(r), prev_best[key(r)])
        best[key(r)] = np.minimum(b, r["costs"])
    n_pts = 16 * len(best)
    summary = {}
    print(f"\nconfig                OK / {n_pts} per library seed   mean    worst   runs/instance  sources")
    for cname in a.configs:
        per, src = {}, Counter()
        for r in runs:
            if r["config"] == cname:
                d = per.setdefault(r["lib_seed"], [0, 0.0, []])
                c = np.array(r["costs"]) - best[key(r)]
                d[0] += int(np.sum(c <= 1e-3))
                d[1] = max(d[1], float(c.max()))
                d[2].append(r["n_evals"])
                src.update(r["source"])
        oks = [v[0] for v in per.values()]
        ev = float(np.mean([e for v in per.values() for e in v[2]]))
        summary[cname] = {"ok": oks, "worst": [v[1] for v in per.values()], "n_evals": ev}
        print(f"{cname:21s} {str(oks):24s} {np.mean(oks):6.1f}  "
              f"{max(v[1] for v in per.values()):6.3f}  {ev:12,.0f}  {dict(src)}")
    with open(a.out, "w") as f:
        json.dump({"runs": runs, "summary": summary,
                   "best": {k: v.tolist() for k, v in best.items()}}, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
