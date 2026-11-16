"""
Benchmark 10: shot noise.  Does the TCI stage still help when <C> is estimated from
a finite number of shots, as on a real device?

Every circuit evaluation returns the mean of C over ``shots`` sampled bitstrings
(``SweepFamily(..., shots=S)``), so every optimiser (the sweep TT, the TCI sampler,
COBYLA in all methods) sees the noise.  The returned angles are scored on the exact
<C>.  Same instances and 16 values of t as benchmarks 5-9 (n = 10 by default).

Methods at p = 1:
  tt          sweep_chain(..., p=1): the depth-1 sweep TT, its slice minima polished
  interp      INTERP's p = 1 stage: best of 5 random restarts at every t
  ref         20 random restarts at every t
At p >= 2:
  chain       sweep_chain(..., extra_restarts=0)           (TCI, deterministic)
  chain-noTCI sweep_chain(..., extra_restarts=0, use_tci=False, seed=1)
  interp      INTERP from scratch at every t
  ref         20 random restarts at every t
Shots 0 means exact (no noise).  ``--optimizer spsa`` swaps COBYLA for SPSA in every
polish of every method (``set_local_optimizer``; ``--spsa-iters`` iterations).  Scoring (in score_shots.py): against the best exact
cost found by any method at any noise level.

Usage:  python benchmarks/bench_shots.py --p 1 --shots 0 256 1024 4096 --families mix
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
from tciqaoa.angles import set_local_optimizer  # noqa: E402
from tciqaoa.sweep import SweepFamily, sweep_chain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 256, 1024, 4096])
    ap.add_argument("--families", nargs="+", default=["mix", "field", "penalty", "risk"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--optimizer", default="cobyla", choices=["cobyla", "spsa"])
    ap.add_argument("--spsa-iters", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/shots_n{a.n}_p{a.p}.json"
    set_local_optimizer(a.optimizer, a.spsa_iters)
    ts = np.arange(0, 32, 2) / 32
    rows = []
    for name in a.families:
        for seed in range(a.seeds):
            Qf = family(name, seed, a.n)
            ex = SweepFamily(Qf)
            for S in a.shots:
                fam = SweepFamily(Qf, shots=S or None, seed=a.noise_seed)
                ang, ev = {}, {}
                if a.p == 1:
                    r = sweep_chain(fam, ts, 1, warm_start=False)   # as committed (benchmark 9)
                    ang["tt"], ev["tt"] = r["angles"], r["n_evals"]
                else:
                    r = sweep_chain(fam, ts, a.p, extra_restarts=0)
                    ang["chain"], ev["chain"] = r["angles"], r["n_evals"]
                    r = sweep_chain(fam, ts, a.p, extra_restarts=0, use_tci=False, seed=1)
                    ang["chain-noTCI"], ev["chain-noTCI"] = r["angles"], r["n_evals"]
                it = [optimize_interp(lambda k, t=t: fam.qaoa(t, k), a.p, seed=seed) for t in ts]
                ang["interp"], ev["interp"] = [x["angles"] for x in it], sum(x["n_evals"] for x in it)
                rf = [optimize_restarts(fam.qaoa(t, a.p), n_starts=20, seed=99) for t in ts]
                ang["ref"], ev["ref"] = [x["angles"] for x in rf], sum(x["n_evals"] for x in rf)
                row = {"family": name, "seed": seed, "shots": S, "optimizer": a.optimizer, "methods": {}}
                for m in ang:
                    qs = [ex.qaoa(t, a.p) for t in ts]
                    row["methods"][m] = {
                        "cost": [q.exact_cost(x) for q, x in zip(qs, ang[m])],
                        "p_ground": [float(q.probs(x)[q.gs_mask].sum()) for q, x in zip(qs, ang[m])],
                        "n_evals": int(ev[m]),
                        "angles": [np.asarray(x).tolist() for x in ang[m]]}
                rows.append(row)
                print(f"{name:8s} seed={seed} shots={S:5d} " + " | ".join(
                    f"{m}: {np.mean(v['cost']):.4f} {v['n_evals']:,}" for m, v in row["methods"].items()),
                    flush=True)
    with open(out, "w") as f:
        json.dump(rows, f)
    print("saved", out)


if __name__ == "__main__":
    main()
