"""
Benchmark 6c: chain the depths.  Lift the p - 1 basin library into the depth-p sweep.

At p = 3 the p = 2 recipe (benchmark 6) finds ~150 of 192 optima.  The p = 1 sweep TT
helped most there, through the INTERP lift.  This tests the same lift one level up:
solve the family at p = 2 first (``sweep_library``), then at p = 3 with
``sweep_library(..., lower=p2)``: the p = 2 optimum at each t, lifted, is a per-t
candidate, and the whole p = 2 library, lifted, joins the p = 3 library.

Configurations (cost includes every stage: TT build, p = 2 run, p = 3 run):
  TT|restarts            benchmark 6's best value at p = 3 (no chain), for reference
  chain|restarts         p = 2 (TT + 4 restarts) -> p = 3 (TT + 4 restarts + lift)
  chain|none             p = 2 (TT + 4 restarts) -> p = 3 (TT + lift, no p = 3 seeding)
  chain-noTT|none        p = 2 (TT + 4 restarts) -> p = 3 (lift only)
  chain|tci              p = 2 (TCI seeding)     -> p = 3 (TT + lift, no p = 3 seeding)
Library seeds 1..S for the restart-based ones.  Scored against the common best of
benchmark 6 (p = 3), its seed reruns (6b) and every run here.

Usage:  python benchmarks/bench_depth_chain.py
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


def run_configs(fam, tt, ts, s):
    out = {}
    r = sweep_library(tt, fam, ts, 3, seeding="restarts", seed=s)
    out["TT|restarts"] = (r, r["n_evals"] + tt["n_evals"])
    p2 = sweep_library(tt, fam, ts, 2, seeding="restarts", seed=s)
    base = p2["n_evals"] + tt["n_evals"]
    r = sweep_library(tt, fam, ts, 3, seeding="restarts", seed=s, lower=p2)
    out["chain|restarts"] = (r, base + r["n_evals"])
    r = sweep_library(tt, fam, ts, 3, seeding=None, seed=s, lower=p2)
    out["chain|none"] = (r, base + r["n_evals"])
    r = sweep_library(None, fam, ts, 3, seeding=None, seed=s, lower=p2)
    out["chain-noTT|none"] = (r, base + r["n_evals"])
    if s == 1:
        p2t = sweep_library(tt, fam, ts, 2, seeding="tci")
        r = sweep_library(tt, fam, ts, 3, seeding=None, lower=p2t)
        out["chain|tci"] = (r, p2t["n_evals"] + tt["n_evals"] + r["n_evals"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/tci_placement_p3.json")
    ap.add_argument("--inp-seeds", default="results/tci_placement_seeds_p3.json")
    ap.add_argument("--lib-seeds", type=int, default=3)
    ap.add_argument("--out", default="results/depth_chain_p3.json")
    a = ap.parse_args()
    rows = json.load(open(a.inp))
    old = json.load(open(a.inp_seeds))["runs"]
    ts = np.arange(0, 32, 2) / 32
    runs = []
    for row in rows:
        fam = SweepFamily(family(row["family"], row["seed"]))
        tt = tci_sweep(fam, 1)
        for s in range(1, a.lib_seeds + 1):
            for name, (r, ev) in run_configs(fam, tt, ts, s).items():
                runs.append({"family": row["family"], "seed": row["seed"], "config": name,
                             "lib_seed": s, "costs": r["costs"].tolist(), "n_evals": int(ev),
                             "source": dict(Counter(r["source"]))})
        print(row["family"], row["seed"], "done", flush=True)

    key = lambda r: (r["family"], r["seed"])  # noqa: E731
    best = {}
    for row in rows:
        b = np.array(row["best"])
        for r in old + runs:
            if key(r) == key(row):
                b = np.minimum(b, r["costs"])
        best[key(row)] = b
    summary = {}
    print("\nconfig            OK per library seed      mean    worst   runs/instance  sources")
    for name in dict.fromkeys(r["config"] for r in runs):
        per = {}
        src = Counter()
        for r in runs:
            if r["config"] == name:
                d = per.setdefault(r["lib_seed"], [0, 0.0, []])
                c = np.array(r["costs"]) - best[key(r)]
                d[0] += int(np.sum(c <= 1e-3))
                d[1] = max(d[1], float(c.max()))
                d[2].append(r["n_evals"])
                src.update(r["source"])
        oks = [v[0] for v in per.values()]
        summary[name] = {"ok": oks, "worst": [v[1] for v in per.values()],
                         "n_evals": float(np.mean([e for v in per.values() for e in v[2]]))}
        print(f"{name:17s} {str(oks):22s} {np.mean(oks):6.1f}  {max(v[1] for v in per.values()):6.3f}"
              f"  {summary[name]['n_evals']:12,.0f}  {dict(src)}")
    # earlier configurations, rescored against the new common best
    for name in ("off|restarts8", "off|tci", "TT|tci"):
        ok = sum(int(np.sum(np.array(row["configs"][name]["costs"]) <= best[key(row)] + 1e-3))
                 for row in rows)
        print(f"{name:17s} {ok} (benchmark 6, rescored)")
    with open(a.out, "w") as f:
        json.dump({"runs": runs, "summary": summary,
                   "best": {f"{k[0]}|{k[1]}": v.tolist() for k, v in best.items()}}, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
