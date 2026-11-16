"""
Score benchmark 10 (dense sweeps): results/dense_n{n}_p{p}_*.json.

The reference at each (family, instance, t) is the lowest exact cost found there by any
method at any sweep density nt (the t of a coarse sweep are points of the finer ones).
Per depth, method and nt it reports the share of points within 1e-3 of it, the worst
miss and the circuit runs per sweep point.

Usage:  python benchmarks/score_dense.py [--n 10]
"""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    for p in (1, 2, 3, 4):
        rows = [r for fn in sorted(glob.glob(f"results/dense_n{a.n}_p{p}_*.json"))
                for r in json.load(open(fn))]
        if not rows:
            continue
        best = defaultdict(lambda: np.inf)
        for r in rows:
            for v in r["methods"].values():
                for t, c in zip(r["ts"], v["costs"]):
                    k = (r["family"], r["seed"], round(t * 64))
                    best[k] = min(best[k], c)
        agg = defaultdict(lambda: [0, 0, 0.0, [], defaultdict(int)])
        for r in rows:
            for m, v in r["methods"].items():
                d = agg[(m, r["nt"])]
                g = np.array([c - best[(r["family"], r["seed"], round(t * 64))]
                              for t, c in zip(r["ts"], v["costs"])])
                d[0] += int(np.sum(g <= 1e-3))
                d[1] += len(g)
                d[2] = max(d[2], float(g.max()))
                d[3].append(v["n_evals"] / r["nt"])
                d[4][r["family"]] += int(np.sum(g <= 1e-3))
        tt = np.mean([r["tt_evals"] for r in rows if r["nt"] == rows[0]["nt"]])
        print(f"\n== p = {p}  (depth-1 TT: {tt:,.0f} runs per instance, rank "
              f"{min(r['tt_rank'] for r in rows)}-{max(r['tt_rank'] for r in rows)})")
        print(f"{'method':15s} {'nt':>3s} {'OK':>9s} {'%':>5s} {'worst':>6s} {'runs/pt':>8s}   OK per family")
        for (m, nt) in sorted(agg, key=lambda k: (k[0], k[1])):
            ok, tot, w, ev, fam = agg[(m, nt)]
            print(f"{m:15s} {nt:3d} {ok:4d}/{tot:4d} {100 * ok / tot:5.1f} {w:6.3f} "
                  f"{np.mean(ev):8,.0f}   {dict(fam)}")


if __name__ == "__main__":
    main()
