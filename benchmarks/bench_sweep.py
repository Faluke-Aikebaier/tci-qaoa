"""
Benchmark 5: a family Q(t), t in [0, 1], solved at 16 values of t.

Families (n = 10):
  mix        MaxCut (t=0) blended into a random QUBO (t=1). The MaxCut optimum is
             symmetric (degenerate copies), and the blend breaks the symmetry.
  field      MaxCut plus a uniform field t * 3 * sum(x).
  penalty    portfolio selection, cardinality-penalty weight swept from 0.05 to 3.05.
             Markowitz cost  lambda x^T S x - mu^T x + A (sum x - k)^2 ; for binary x the
             linear return -mu^T x is the diagonal term -diag(mu).
  risk       portfolio selection, risk aversion swept from 0.2 to 6.2 (control: smooth).

Methods, all counted in circuit evaluations:
  interp        INTERP from scratch at every t
  point         INTERP once at t = 0, then polish from that point at every t
  continuation  INTERP at t = 0, then each t starts from the previous t's optimum
  TT-sweep      ONE quantics TCI over (angles, t) at depth 1; at each t, polish the 3
                lowest minima of the slice, read off the tensor train.  At depth p >= 2
                the depth-p landscape is not low rank, so the depth-1 TT is used and its
                minima are lifted to depth p with the INTERP rule ("TT-lifted").
  TT+cont       the TT-sweep candidates plus the previous t's optimum (continuation)
  TT+library    sweep_library: TT-sweep candidates, the previous t's optimum, and a shared
                basin library seeded by 4 random restarts at every other t (symmetry-
                reduced box), grown along the sweep, with a backward pass (closes the
                p >= 2 gap; the anchor restarts are counted)
  library       ablation: TT+library without the TT candidates (no TCI at all)

OK = within 1e-3 (normalised cost) of the best found by any method or by a separate
20-restart reference (the reference is not counted).

Usage:  python benchmarks/bench_sweep.py --seeds 3 --p 1 2
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tciqaoa import optimize_from, optimize_interp, optimize_restarts, random_graph, random_qubo
from tciqaoa.sweep import SweepFamily, lift, sweep_library, sweep_minima, tci_sweep

N = 10


def scheduling_family(n, seed):
    """Job scheduling on identical machines: x_{ij} = 1 if job i runs on machine j.

    M = 3 machines when 3 divides n, else 2; J = n // M jobs, times U(2, 10) as in
    hybridq's example.  Objective: sum_j load_j^2 (normalised so a perfectly balanced
    schedule scores 1), a QUBO proxy for the makespan; hybridq's own objective
    sum_i t_i x_{ij} is the same for every feasible assignment.  Constraint (as in
    hybridq): lam(t) sum_i (sum_j x_{ij} - 1)^2, with lam(t) = (0.2 + 1.0 t) lam_feas,
    where lam_feas is the largest marginal objective cost of placing one job: at small t
    leaving jobs unassigned pays, at large t only full assignments are optimal.
    """
    M = 3 if n % 3 == 0 else 2
    J = n // M
    tm = np.random.default_rng(seed + 900).uniform(2.0, 10.0, J)
    T = tm.sum()
    scale = M / T ** 2
    O = np.zeros((n, n))
    P = np.zeros((n, n))
    for i in range(J):
        for j in range(M):
            for i2 in range(J):
                O[i * M + j, i2 * M + j] = tm[i] * tm[i2] * scale
            for j2 in range(M):
                P[i * M + j, i * M + j2] = -1.0 if j == j2 else 1.0
    lam_feas = float(np.max(tm ** 2 + 2 * tm * (T - tm)) * scale)
    return lambda t: O + (0.2 + 1.0 * t) * lam_feas * P


def family(name, seed, n=None):
    N = globals()["N"] if n is None else n      # noqa: N806
    rng = np.random.default_rng(seed)
    A = random_graph(N, 0.4, seed=seed)
    L = np.diag(A.sum(1)) - A
    if name == "mix":
        Qr = random_qubo(N, seed=seed + 500)
        return lambda t: (1 - t) * (-L) / np.abs(L).max() + t * Qr / 2
    if name == "field":
        return lambda t: -L + t * 3 * np.eye(N)
    if name == "sched":
        return scheduling_family(N, seed)
    mu = rng.uniform(0.05, 0.2, N)
    M = rng.standard_normal((N, N))
    S = M @ M.T / N
    mun = mu / np.abs(mu).max()
    S = S / np.abs(S).max()
    k = round(0.4 * N)                           # 4 at the default n = 10
    P = np.ones((N, N)) - 2 * k * np.eye(N)      # (sum x - k)^2 up to a constant
    R = np.diag(mun)                             # expected return mu^T x (x_i^2 = x_i)
    if name == "penalty":
        return lambda t: S - R + (0.05 + 3 * t) * P
    if name == "risk":
        return lambda t: (0.2 + 6 * t) * S - R + 1.0 * P
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["mix", "field", "penalty", "risk"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", default="results/sweep.json")
    a = ap.parse_args()
    rows = []
    ts = np.arange(0, 32, 2) / 32
    for p in a.p:
        for name in a.families:
            for seed in range(a.seeds):
                fam = SweepFamily(family(name, seed))
                cost = {"interp": 0, "point": 0, "continuation": 0, "TT-sweep": 0, "TT+cont": 0,
                        "TT+library": 0, "library": 0}
                found = {m: [] for m in cost}
                # build phases (paid once)
                t0 = time.perf_counter()
                tt = tci_sweep(fam, 1)            # depth-1 sweep TT, lifted for p >= 2
                cost["TT-sweep"] += tt["n_evals"]
                cost["TT+cont"] += tt["n_evals"]
                t_tt = time.perf_counter() - t0
                r0 = optimize_interp(lambda k: fam.qaoa(0.0, k), p, seed=seed)
                cost["point"] += r0["n_evals"]
                cost["continuation"] += r0["n_evals"]
                cost["TT+cont"] += r0["n_evals"]
                prev = r0["angles"]
                prev_tc = r0["angles"]
                for t in ts:
                    r = optimize_interp(lambda k: fam.qaoa(t, k), p, seed=seed)
                    cost["interp"] += r["n_evals"]; found["interp"].append(r["cost"])
                    r = optimize_from(fam.qaoa(t, p), [r0["angles"]])
                    cost["point"] += r["n_evals"]; found["point"].append(r["cost"])
                    r = optimize_from(fam.qaoa(t, p), [prev])
                    cost["continuation"] += r["n_evals"]; found["continuation"].append(r["cost"])
                    prev = r["angles"]
                    cands = [lift(a, 1, p) for a in sweep_minima(tt, t, 3)]
                    r = optimize_from(fam.qaoa(t, p), cands)
                    cost["TT-sweep"] += r["n_evals"]; found["TT-sweep"].append(r["cost"])
                    r = optimize_from(fam.qaoa(t, p), cands + [prev_tc])
                    cost["TT+cont"] += r["n_evals"]; found["TT+cont"].append(r["cost"])
                    prev_tc = r["angles"]
                lb = sweep_library(tt, fam, ts, p, seeding="restarts")
                cost["TT+library"] += tt["n_evals"] + lb["n_evals"]
                found["TT+library"] = list(lb["costs"])
                lb = sweep_library(None, fam, ts, p, seeding="restarts")
                cost["library"] += lb["n_evals"]
                found["library"] = list(lb["costs"])
                rq = [fam.qaoa(t, p) for t in ts]
                ref = [optimize_restarts(q, n_starts=20, seed=99)["cost"] for q in rq]
                ref_evals = sum(q.n_evals for q in rq)
                best = np.minimum(np.min([found[m] for m in found], axis=0), ref)
                ok = {m: int(np.sum(np.array(found[m]) <= best + 1e-3)) for m in found}
                gap = {m: float(np.max(np.array(found[m]) - best)) for m in found}
                row = {"p": p, "family": name, "seed": seed, "n_t": len(ts), "cost": cost,
                       "ok": ok, "worst_gap": gap, "tt_build": tt["n_evals"],
                       "tt_rank": tt["tt_max_rank"], "tt_pivot_error": tt["pivot_error"],
                       "tt_build_time": t_tt, "ref_evals": ref_evals,
                       "ref_ok": int(np.sum(np.array(ref) <= best + 1e-3))}
                rows.append(row)
                print(f"p={p} {name:8s} seed={seed} " + " | ".join(
                    f"{m}: {ok[m]}/{len(ts)} ok, {cost[m]} evals" for m in cost)
                    + f" | TT rank {tt['tt_max_rank']} piv {tt['pivot_error']:.0e}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
