"""
Benchmark 6b: seed variance of the random-restart seedings in benchmark 6.

TCI seeding is deterministic; random-restart seeding is not, so one run cannot tell a
5/192 difference from luck.  This reruns the restart configurations with library seeds
2..5 on the same instances and scores them against benchmark 6's per-t best values
(lowered further if a rerun finds something better).

Usage:  python benchmarks/bench_tci_placement_seeds.py [--inp results/tci_placement.json]
                                                     [--out results/tci_placement_seeds.json]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from bench_sweep import family  # noqa: E402
from tciqaoa.sweep import SweepFamily, sweep_library, tci_sweep  # noqa: E402

CONFIGS = {"off|restarts": (False, 4), "off|restarts8": (False, 8), "TT|restarts": (True, 4)}
SEEDS = [2, 3, 4, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/tci_placement.json")
    ap.add_argument("--out", default="results/tci_placement_seeds.json")
    a = ap.parse_args()
    rows = json.load(open(a.inp))
    ts = np.arange(0, 32, 2) / 32
    runs = []
    for row in rows:
        fam = SweepFamily(family(row["family"], row["seed"]))
        tt = tci_sweep(fam, 1)
        for name, (use_tt, n) in CONFIGS.items():
            for s in SEEDS:
                lb = sweep_library(tt if use_tt else None, fam, ts, row["p"], n_restarts=n, seed=s,
                                   seeding="restarts")
                runs.append({"family": row["family"], "seed": row["seed"], "config": name,
                             "lib_seed": s, "costs": lb["costs"].tolist(),
                             "n_evals": lb["n_evals"] + (tt["n_evals"] if use_tt else 0)})
        print(row["family"], row["seed"], "done", flush=True)
    # common best per instance: benchmark 6 best and every rerun
    best = {}
    for row in rows:
        b = np.array(row["best"])
        for r in runs:
            if (r["family"], r["seed"]) == (row["family"], row["seed"]):
                b = np.minimum(b, r["costs"])
        best[(row["family"], row["seed"])] = b
    out = {}
    for name in CONFIGS:
        per_seed = {}
        cands = [(1, row, row["configs"][name]["costs"]) for row in rows] + \
                [(r["lib_seed"], r, r["costs"]) for r in runs if r["config"] == name]
        for s, row, c in cands:
            b = best[(row["family"], row["seed"])]
            d = per_seed.setdefault(s, [0, 0.0])
            d[0] += int(np.sum(np.array(c) <= b + 1e-3))
            d[1] = max(d[1], float(np.max(np.array(c) - b)))
        out[name] = per_seed
        oks = [v[0] for v in per_seed.values()]
        print(f"{name:15s} OK per library seed {oks}  mean {np.mean(oks):.1f} sd {np.std(oks):.1f}"
              f"  worst per seed {[round(v[1], 3) for v in per_seed.values()]}")
    for k in ("off|tci", "TT|tci"):
        ok = sum(int(np.sum(np.array(row["configs"][k]["costs"]) <= best[(row["family"], row["seed"])] + 1e-3))
                 for row in rows)
        print(f"{k:15s} OK {ok} (deterministic) against the same best")
    with open(a.out, "w") as f:
        json.dump({"runs": runs, "summary": out}, f, indent=1)


if __name__ == "__main__":
    main()
