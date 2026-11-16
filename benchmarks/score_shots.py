"""
Score benchmark 10 (shot noise): results/shots_n{n}_p{p}_*.json.

For each depth, the reference at each (family, instance, t) is the lowest exact <C>
found by any method at any noise level.  Per method and shot count it reports
  OK      points within 1e-3 of it (of 16 x instances)
  <0.01   points within 0.01
  gap     mean exact excess over the reference
  P(gs)   mean ground-state probability of the returned angles
  runs    circuit runs per instance (each one costs ``shots`` shots on a device)

COBYLA and SPSA runs (results/shots_spsa_*, methods "+spsa") share the reference.

Usage:  python benchmarks/score_shots.py [--n 10]
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
        rows = [r for fn in sorted(glob.glob(f"results/shots_n{a.n}_p{p}_*.json"))
                + sorted(glob.glob(f"results/shots_spsa_n{a.n}_p{p}_*.json"))
                for r in json.load(open(fn))]
        for r in rows:              # SPSA runs: method "+spsa"
            if r.get("optimizer") == "spsa":
                r["methods"] = {m + "+spsa": v for m, v in r["methods"].items()}
        if not rows:
            continue
        best = {}
        for r in rows:
            k = (r["family"], r["seed"])
            for v in r["methods"].values():
                best[k] = np.minimum(best.get(k, np.inf), v["cost"])
        agg = defaultdict(lambda: defaultdict(list))
        for r in rows:
            b = best[(r["family"], r["seed"])]
            for m, v in r["methods"].items():
                d = agg[(m, r["shots"])]
                g = np.asarray(v["cost"]) - b
                d["ok"].append(int(np.sum(g <= 1e-3)))
                d["ok2"].append(int(np.sum(g <= 1e-2)))
                d["gap"].extend(g.tolist())
                d["pg"].extend(v["p_ground"])
                d["runs"].append(v["n_evals"])
                d["fam"].append((r["family"], int(np.sum(g <= 1e-2))))
        n_inst = len(best)
        print(f"\n== p = {p}  ({n_inst} instances x 16 t = {16 * n_inst} points)")
        print(f"{'method':12s} {'shots':>6s} {'OK':>5s} {'<0.01':>6s} {'gap':>7s} {'worst':>6s}"
              f" {'P(gs)':>6s} {'runs':>9s}   <0.01 per family")
        for (m, S) in sorted(agg, key=lambda k: (k[0], k[1])):
            d = agg[(m, S)]
            fam = defaultdict(int)
            for f, c in d["fam"]:
                fam[f] += c
            print(f"{m:12s} {S or 'exact':>6} {sum(d['ok']):5d} {sum(d['ok2']):6d} "
                  f"{np.mean(d['gap']):7.4f} {np.max(d['gap']):6.3f} {np.mean(d['pg']):6.3f} "
                  f"{np.mean(d['runs']):9,.0f}   {dict(fam)}")


if __name__ == "__main__":
    main()
