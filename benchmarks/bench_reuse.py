"""
Benchmark 3: does a TCI landscape learned once pay off on new instances?

Train: learn the depth-p landscape <C>(gamma, beta) on ONE training instance with
       optimize_tci(return_tt=True).  This cost is paid once.
Test:  on each of N new instances, compare
  interp        INTERP from scratch (no reuse; at p = 1 this is 5 random restarts)
  point         reuse only the 2p numbers of the training optimum: one local polish
                (the control: "do we need the tensor train, or just the angles?")
  point+images  the same 2p numbers, plus their 2**p - 1 images under beta_k -> beta_k + pi/2
                (equivalent on MaxCut, not on QUBO): rank by one run each, polish the best
  TT-reuse      polish from the top-3 distinct minima of the stored tensor train
                (evaluated on the full grid at p = 1, no circuit runs)

Scenarios cover within-family transfer and harder shifts (other family, other size).
Success = within 1e-3 of the best approximation ratio any method found on that instance.

Usage:  python benchmarks/bench_reuse.py --tests 50
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tciqaoa import (QAOA, all_energies, landscape_minima, maxcut_qubo, optimize_from,
                     optimize_from_images,
                     optimize_interp, optimize_tci, random_graph, random_qubo)


def make(kind, n, seed):
    if kind == "maxcut":
        return maxcut_qubo(random_graph(n, 0.4, seed=seed))
    if kind == "maxcut-sparse":
        return maxcut_qubo(random_graph(n, 0.2, seed=seed))
    return random_qubo(n, 0.5, seed=seed)


SCENARIOS = [  # (train family, train n) -> (test family, test n)
    (("maxcut", 12), ("maxcut", 12)),
    (("maxcut", 12), ("maxcut", 16)),
    (("maxcut", 12), ("maxcut-sparse", 12)),
    (("maxcut", 12), ("qubo", 12)),
    (("qubo", 12), ("qubo", 12)),
    (("qubo", 12), ("maxcut", 12)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", type=int, default=50)
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", default="results/reuse.json")
    a = ap.parse_args()
    rows = []
    for p in a.p:
        for (tk, tn), (sk, sn) in SCENARIOS:
            Qt = make(tk, tn, 10_000)
            qt = QAOA(Qt, p)
            t0 = time.perf_counter()
            tr = optimize_tci(qt, return_tt=True)
            starts_tt = landscape_minima(tr, top_k=3)
            build = {"n_evals": tr["n_evals"], "time": time.perf_counter() - t0,
                     "tt_max_rank": tr["tt_max_rank"], "pivot_error": tr["pivot_error"]}
            label = f"p={p} {tk}{tn} -> {sk}{sn}"
            per = []
            for s in range(a.tests):
                Q = make(sk, sn, s)
                E = all_energies(Q)
                res = {}
                t = time.perf_counter()
                r = optimize_interp(lambda k: QAOA(Q, k, energies=E), p, seed=s)
                res["interp"] = (r, time.perf_counter() - t)
                for name, starts in [("point", [tr["angles"]]), ("TT-reuse", starts_tt)]:
                    q = QAOA(Q, p, energies=E)
                    t = time.perf_counter()
                    r = optimize_from(q, starts)
                    res[name] = (r, time.perf_counter() - t)
                q = QAOA(Q, p, energies=E)
                t = time.perf_counter()
                res["point+images"] = (optimize_from_images(q, tr["angles"]), time.perf_counter() - t)
                ref = QAOA(Q, p, energies=E)
                ratio = {k: ref.report(v[0]["angles"])["approx_ratio"] for k, v in res.items()}
                best = max(ratio.values())
                per.append({k: {"n_evals": v[0]["n_evals"], "time": v[1],
                                "approx_ratio": ratio[k], "ok": bool(ratio[k] >= best - 1e-3)}
                            for k, v in res.items()})
            row = {"p": p, "train": [tk, tn], "test": [sk, sn], "build": build, "tests": per}
            rows.append(row)
            summ = " | ".join(
                f"{m}: {sum(t[m]['ok'] for t in per)}/{len(per)} ok, "
                f"{np.mean([t[m]['n_evals'] for t in per]):.0f} evals"
                for m in ["interp", "point", "point+images", "TT-reuse"])
            print(f"{label:34s} build={build['n_evals']} (rank {build['tt_max_rank']})  {summ}",
                  flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
