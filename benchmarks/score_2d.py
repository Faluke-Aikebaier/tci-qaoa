"""
Score benchmarks 11 and 12 (two- and three-parameter sweeps):
results/{prefix}_n{n}_p{p}_*.json, prefix sweep2d (default) or sweep3d.

The reference at each (family, instance, point) is the lowest exact cost found there by
any method at any grid size.  Per depth, method and grid size (nt x nt) it reports the
share of points within 1e-3 of it, the worst miss and the circuit runs per point.

Usage:  python benchmarks/score_2d.py [--n 10] [--prefix sweep3d]
"""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--prefix", default="sweep2d")
    a = ap.parse_args()
    for p in (1, 2, 3, 4):
        rows = [r for fn in sorted(glob.glob(f"results/{a.prefix}_n{a.n}_p{p}_*.json"))
                for r in json.load(open(fn))]
        if not rows:
            continue
        best = defaultdict(lambda: np.inf)
        for r in rows:
            for v in r["methods"].values():
                for t, c in zip(r["ts"], v["costs"]):
                    k = (r["family"], r["seed"], round(t * 2 ** 15))
                    best[k] = min(best[k], c)
        agg = defaultdict(lambda: [0, 0, 0.0, [], defaultdict(int)])
        tt = defaultdict(list)
        for r in rows:
            tt[r["family"]].append((r["tt_evals"], r["tt_rank"]))
            for m, v in r["methods"].items():
                d = agg[(m, r["nt"], r.get("n_params", 2))]
                g = np.array([c - best[(r["family"], r["seed"], round(t * 2 ** 15))]
                              for t, c in zip(r["ts"], v["costs"])])
                d[0] += int(np.sum(g <= 1e-3))
                d[1] += len(g)
                d[2] = max(d[2], float(g.max()))
                d[3].append(v["n_evals"] / len(r["ts"]))
                d[4][r["family"]] += int(np.sum(g <= 1e-3))
        print(f"\n== p = {p}  depth-1 TT: " + ", ".join(
            f"{f} {np.mean([e for e, _ in v]):,.0f} runs (rank {min(k for _, k in v)}-{max(k for _, k in v)})"
            for f, v in tt.items()))
        print(f"{'method':16s} {'grid':>5s} {'OK':>9s} {'%':>5s} {'worst':>6s} {'runs/pt':>8s}   OK per family")
        for (m, nt, k) in sorted(agg):
            ok, tot, w, ev, fam = agg[(m, nt, k)]
            print(f"{m:16s} {f'{nt}^{k}':>5s} {ok:4d}/{tot:4d} {100 * ok / tot:5.1f} {w:6.3f} "
                  f"{np.mean(ev):8,.0f}   {dict(fam)}")


if __name__ == "__main__":
    main()
