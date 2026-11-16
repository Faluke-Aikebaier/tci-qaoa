"""
Main entry points: solve a QUBO with QAOA (or directly), with TCI switched on or off.

Each function takes ``tci=True`` / ``tci=False``.  ``tci=False`` runs the same pipeline
with the TCI stage replaced by the baseline it was benchmarked against, so the two can
be compared on equal terms.

  solve_qubo(Q, p, tci)          one QUBO, QAOA at depth p
      tci=True   p = 1: optimize_tci (TT over the angle box, best points polished)
                 p >= 2: the depth chain (p = 1 TT -> p = 2 library seeded by the TCI
                 sampler -> INTERP lift to p), on a one-point family
      tci=False  p = 1: INTERP (best of 5 random restarts)
                 p >= 2: the same chain, depth 2 seeded by random restarts

  solve_family(make_Q, p, tci)   a family Q(t_1, .., t_k), k = 1, 2, 3 .., at many points
      tci=True   p = 1: one TT over (angles, t_1 .. t_k), its slice minima polished at
                 each point (plus the previous point's optimum as a warm start)
                 p >= 2: the depth chain built on that TT (``sweep_chain``)
      tci=False  p = 1: INTERP at each point
                 p >= 2: the same chain, depth 2 seeded by random restarts

  solve_bits(Q, tci)             minimise x^T Q x over bitstrings directly (no QAOA)
      tci=True   tci_minimize_bits (TCI on exp(-beta E(x)), staged beta, 1-flip polish);
                 for a list of QUBOs, tci_sweep_bits (one TT for the whole list)
      tci=False  random-restart steepest-descent local search

When TCI helps (README, "Conclusions"): on families with many points (the p = 1 TT is
~1.8x cheaper than INTERP from ~64 points on) and, inside the chain, when it is seeded
at few points (``n_anchors``).  On one instance, or with full seeding, ``tci=False``
is as good and cheaper.  TCI needs xfacpy; ``tci=False`` does not.

Command line:  python -m tciqaoa --help
"""

import argparse
import importlib.util
import sys

import numpy as np

from .angles import optimize_interp, optimize_tci
from .problems import brute_force, local_search, symmetrize
from .qaoa import QAOA
from .sweep import SweepFamily, SweepFamilyND, sweep_chain, tci_sweep

__all__ = ["solve_qubo", "solve_family", "solve_bits", "main"]


def _on(tci):
    if isinstance(tci, str):
        if tci.lower() not in ("on", "off", "true", "false", "1", "0"):
            raise ValueError(f"tci must be on/off, got {tci!r}")
        return tci.lower() in ("on", "true", "1")
    return bool(tci)


def _with_report(q, out, report):
    if report:
        out["report"] = q.report(out["angles"], rng=0)
    return out


def solve_qubo(Q, p=1, tci=True, seed=0, extra_restarts=8, angle_pivots=4, report=True):
    """QAOA at depth p for one QUBO matrix Q (n <= ~22), TCI on or off.

    Returns dict(angles, cost, n_evals, tci, report): ``cost`` is the optimised <C>
    (normalised cost, lower is better), ``n_evals`` the circuit runs used, ``report``
    the physical figures of merit (approximation ratio, P(ground), best sampled
    bitstring; see ``QAOA.report``).  ``angle_pivots``: see ``solve_family``.
    """
    on = _on(tci)
    Q = symmetrize(np.asarray(Q, dtype=float))
    q = QAOA(Q, p)
    if p == 1:
        r = optimize_tci(q, seed=seed) if on else optimize_interp(lambda k: QAOA(Q, k, energies=q.E), 1, seed=seed)
        out = {"angles": np.asarray(r["angles"]), "cost": float(r["cost"]), "n_evals": int(r["n_evals"])}
    else:
        fam = SweepFamily(lambda t: Q)
        tt = tci_sweep(fam, 1, angle_pivots=angle_pivots) if on else None
        r = sweep_chain(fam, [0.0], p, tt=tt, extra_restarts=extra_restarts, use_tci=on, seed=seed + 1)
        out = {"angles": np.asarray(r["angles"][0]), "cost": float(r["costs"][0]),
               "n_evals": int(r["n_evals"])}
    out["tci"] = on
    return _with_report(q, out, report)


def solve_family(make_Q, p=1, tci=True, n_params=1, nt=8, points=None, R=5, n_anchors=None,
                 extra_restarts=0, seed=1, tt=None, angle_pivots=4, report=False):
    """QAOA at depth p for a whole family Q(t_1, .., t_k) at many parameter points.

    ``make_Q(t_1, .., t_k)`` returns the QUBO matrix; every t_j lies in [0, 1) (rescale
    your physical parameter inside ``make_Q``) and is snapped to a 2**R grid.  The
    points are either ``points`` (a list of floats for k = 1, or of k-tuples), visited
    in the given order, or by default an ``nt``-per-axis grid (nt**k points, nt
    dividing 2**R) in snake order, so consecutive points are neighbours; the chain and
    the warm start rely on that, so pass your own points in a neighbour-to-neighbour
    order too.

    ``n_anchors`` seeds the p >= 2 chain at that many points only (much cheaper on
    large grids; with ``tci=True`` it loses least accuracy).  ``extra_restarts`` adds
    random restarts per depth from depth 3 on (0 = deterministic, as benchmarked).
    ``tt``: a prebuilt depth-1 sweep TT (``tci_sweep(family, 1)``) to reuse.
    ``angle_pivots`` (default 4): the TT build also starts from 4 x 4 points spread over
    the angle box, so it cannot converge early around one basin (it did on a portfolio
    family with the default budget penalty: rank 4 after 224 runs, every point missed).
    It costs more runs per build (there 1.9k, on benchmark 12's hardest instance 10.7k
    instead of 6.0k).  The benchmarks were run with 0, ``tci_sweep``'s default.

    Returns dict(params, angles, costs, n_evals, tci, tt, reports): one entry per point
    (``params`` as k-tuples), total circuit runs including the TT build, and the TT
    (None with ``tci=False``) so another call can share it.
    """
    on = _on(tci)
    if n_params == 1:
        fam = SweepFamily(lambda t: make_Q(t), R_t=R)
    else:
        fam = SweepFamilyND(make_Q, n_params, R=R)
    if points is None:
        if n_params == 1:
            ts = list(np.arange(nt) / nt)
        else:
            ts = fam.points(nt)
    else:
        ts = [fam.encode(np.atleast_1d(x)) for x in points]
    if on:
        tt = tt if tt is not None else tci_sweep(fam, 1, angle_pivots=angle_pivots)
        r = sweep_chain(fam, ts, p, tt=tt, extra_restarts=extra_restarts, n_anchors=n_anchors, seed=seed)
        angles, costs, n_evals = r["angles"], list(r["costs"]), r["n_evals"]
    elif p == 1:
        rs = [optimize_interp(lambda k, t=t: fam.qaoa(t, k), 1, seed=seed) for t in ts]
        angles = [x["angles"] for x in rs]
        costs = [x["cost"] for x in rs]
        n_evals = sum(x["n_evals"] for x in rs)
        tt = None
    else:
        r = sweep_chain(fam, ts, p, extra_restarts=extra_restarts, n_anchors=n_anchors,
                        use_tci=False, seed=seed)
        angles, costs, n_evals = r["angles"], list(r["costs"]), r["n_evals"]
        tt = None
    out = {"params": [tuple(fam.decode(t)) for t in ts], "angles": [np.asarray(a) for a in angles],
           "costs": np.asarray(costs, dtype=float), "n_evals": int(n_evals), "tci": on, "tt": tt}
    if report:
        out["reports"] = [fam.qaoa(t, p).report(a, rng=0) for t, a in zip(ts, out["angles"])]
    return out


def solve_bits(Q, tci=True, seed=0, n_restarts=None, **tci_kw):
    """Minimise x^T Q x over x in {0,1}^n directly (no QAOA circuit), TCI on or off.

    ``Q`` is one matrix, or a list of matrices (e.g. one QUBO per point of an efficient
    frontier; its length must be a power of 2).  tci=True: ``tci_minimize_bits`` (one
    matrix) or ``tci_sweep_bits`` (list: one TT over (list index, bits)); extra keywords go to them.  tci=False:
    ``n_restarts`` random starts of steepest-descent local search (default 20 n);
    benchmarks 2 and 7 found this baseline better than TCI when time-matched.

    Returns dict(x, energy, n_energy_evals, tci) with x and energy per matrix for a list.
    """
    on = _on(tci)
    many = isinstance(Q, (list, tuple))
    Qs = [symmetrize(np.asarray(M, dtype=float)) for M in (Q if many else [Q])]
    if on:
        from .tci_bits import tci_minimize_bits, tci_sweep_bits
        if many:
            r = tci_sweep_bits(Qs, seed=seed, **tci_kw)
            return {"x": list(r["x"]), "energy": np.asarray(r["energy"]),
                    "n_energy_evals": int(r["n_energy_evals"]), "tci": True}
        r = tci_minimize_bits(Qs[0], seed=seed, **tci_kw)
        return {"x": r["x"], "energy": float(r["energy"]), "n_energy_evals": int(r["n_energy_evals"]),
                "tci": True}
    rng = np.random.default_rng(seed)
    xs, es, ev = [], [], 0
    for M in Qs:
        n = M.shape[0]
        best = (None, np.inf)
        for _ in range(n_restarts or 20 * n):
            x, E, k = local_search(rng.integers(0, 2, n).astype(float), M)
            ev += k
            if E < best[1]:
                best = (x, E)
        xs.append(best[0])
        es.append(best[1])
    if many:
        return {"x": xs, "energy": np.asarray(es), "n_energy_evals": ev, "tci": False}
    return {"x": xs[0], "energy": float(es[0]), "n_energy_evals": ev, "tci": False}


def _load_family(spec):
    path, _, name = spec.partition(":")
    mod_spec = importlib.util.spec_from_file_location("_tciqaoa_family", path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return getattr(mod, name or "make_Q")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m tciqaoa",
        description="Solve a QUBO with QAOA or directly, with TCI on or off.")
    ap.add_argument("qubo", nargs="?", help="QUBO matrix: .npy file, or text (np.loadtxt)")
    ap.add_argument("--mode", choices=["qaoa", "bits"], default="qaoa",
                    help="qaoa: find QAOA angles (n <= ~22); bits: minimise x^T Q x directly")
    ap.add_argument("--p", type=int, default=1, help="QAOA depth")
    ap.add_argument("--tci", choices=["on", "off"], default="on")
    ap.add_argument("--family", metavar="FILE.py:FUNC",
                    help="solve a family: FUNC(t_1, .., t_k) -> Q, each t_j in [0, 1)")
    ap.add_argument("--n-params", type=int, default=1, help="family: number of parameters k")
    ap.add_argument("--nt", type=int, default=8, help="family: points per parameter axis")
    ap.add_argument("--anchors", type=int, default=None, help="family, p >= 2: seed at this many points only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--brute-force", action="store_true", help="also print the exact optimum (n <= ~22)")
    a = ap.parse_args(argv)

    if a.family:
        r = solve_family(_load_family(a.family), a.p, a.tci, n_params=a.n_params, nt=a.nt,
                         n_anchors=a.anchors, seed=a.seed + 1, report=True)
        print(f"family, p = {a.p}, TCI {a.tci}: {len(r['params'])} points, {r['n_evals']:,} circuit runs")
        for par, c, rep in zip(r["params"], r["costs"], r["reports"]):
            x = "".join(str(int(v)) for v in rep["best_sampled_x"])
            print(f"  t = {', '.join(f'{v:.4f}' for v in par)}   <C> = {c:+.5f}   "
                  f"ratio {rep['approx_ratio']:.4f}   P(ground) {rep['p_ground']:.4f}   best x {x}")
        return 0
    if not a.qubo:
        ap.error("give a QUBO file, or --family")
    Q = np.load(a.qubo) if a.qubo.endswith(".npy") else np.loadtxt(a.qubo)
    if a.mode == "bits":
        r = solve_bits(Q, a.tci, seed=a.seed)
        x = "".join(str(int(v)) for v in r["x"])
        print(f"bits, TCI {a.tci}: energy {r['energy']:.6g}, x = {x}, {r['n_energy_evals']:,} energy evaluations")
    else:
        r = solve_qubo(Q, a.p, a.tci, seed=a.seed)
        rep = r["report"]
        x = "".join(str(int(v)) for v in rep["best_sampled_x"])
        print(f"QAOA p = {a.p}, TCI {a.tci}: {r['n_evals']:,} circuit runs")
        print(f"  angles (gamma.., beta..) = {np.round(r['angles'], 4).tolist()}")
        print(f"  approx ratio {rep['approx_ratio']:.4f}, P(ground) {rep['p_ground']:.4f}, "
              f"best of {rep['shots']} shots: energy {rep['best_sampled_energy']:.6g}, x = {x}")
    if a.brute_force:
        x, E = brute_force(symmetrize(Q))[:2]
        print(f"  exact optimum: energy {E:.6g}, x = {''.join(str(int(v)) for v in x)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
