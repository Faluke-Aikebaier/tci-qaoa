"""
Optimisers for the 2p QAOA angles (gamma_1..gamma_p, beta_1..beta_p).

    restarts : random initial angles + COBYLA, best of N   (standard practice)
    interp   : INTERP layer-by-layer warm start (Zhou et al., PRX 10, 021067 (2020))
    tci      : learn the whole landscape <C>(gamma, beta) as a quantics tensor train
               with xfacpy, then COBYLA-refine the best TCI-sampled points

All three count every circuit evaluation through QAOA.n_evals, so they can be
compared at equal cost.
"""

import numpy as np
from scipy.optimize import minimize

__all__ = ["GAMMA_MAX", "BETA_MAX", "optimize_restarts", "optimize_interp", "optimize_tci"]

GAMMA_MAX = 2.0          # normalised cost (std = 1); optimal gammas observed <= ~1.9
BETA_MAX = np.pi         # exp(-i beta sum X) is pi-periodic up to a global phase


def _bounds(p):
    return [(0.0, GAMMA_MAX)] * p + [(0.0, BETA_MAX)] * p


def _local(qaoa, x0, maxiter):
    r = minimize(qaoa.cost, x0, method="COBYLA",
                 options={"maxiter": maxiter, "rhobeg": 0.2})
    return r.x, float(r.fun)


def _result(qaoa, x, f, n0, extra=None):
    out = {"angles": np.asarray(x), "cost": f, "n_evals": qaoa.n_evals - n0}
    if extra:
        out.update(extra)
    return out


def optimize_restarts(qaoa, n_starts=10, maxiter=300, seed=0):
    rng = np.random.default_rng(seed)
    n0 = qaoa.n_evals
    best = (None, np.inf)
    lo, hi = np.array(_bounds(qaoa.p)).T
    for _ in range(n_starts):
        x, f = _local(qaoa, rng.uniform(lo, hi), maxiter)
        if f < best[1]:
            best = (x, f)
    return _result(qaoa, *best, n0)


def optimize_interp(qaoa_factory, p, n_starts_p1=5, maxiter=300, seed=0):
    """INTERP: optimise p=1, then warm-start each deeper level by linear interpolation.

    ``qaoa_factory(p)`` must return a QAOA object of depth p (sharing one energy table).
    Returns the result for depth p, with n_evals summed over all depths.
    """
    total = 0
    q1 = qaoa_factory(1)
    r = optimize_restarts(q1, n_starts=n_starts_p1, maxiter=maxiter, seed=seed)
    total += r["n_evals"]
    g, b = r["angles"][:1], r["angles"][1:]
    for k in range(2, p + 1):
        qk = qaoa_factory(k)
        # INTERP rule: new_i = (i-1)/(k-1) old_{i-1} + (k-i)/(k-1) old_i, i = 1..k
        def interp(v):
            old = np.concatenate([[0.0], v, [0.0]])
            i = np.arange(1, k + 1)
            return (i - 1) / (k - 1) * old[i - 1] + (k - i) / (k - 1) * old[i]
        x0 = np.concatenate([interp(g), interp(b)])
        n0 = qk.n_evals
        x, f = _local(qk, x0, maxiter)
        total += qk.n_evals - n0
        g, b = x[:k], x[k:]
        r = {"angles": x, "cost": f}
    r["n_evals"] = total
    return r


def optimize_tci(qaoa, R=5, bond_dim=20, reltol=1e-2, top_k=5, maxiter=300,
                 max_sweeps=20, seed=0, return_tt=False):
    """Quantics-TCI global search over the angle box, then local refinement.

    The landscape is sampled on a 2**R grid per angle; TCI adaptively picks which
    of the 2**(2pR) grid points to evaluate.  Returns the optimum plus the TT rank,
    pivot error and evaluation counts.
    """
    import xfacpy

    p = qaoa.p
    d = 2 * p
    scale = np.array([GAMMA_MAX] * p + [BETA_MAX] * p)
    n0 = qaoa.n_evals
    cache = {}

    def f(u):
        key = tuple(u)
        if key not in cache:
            cache[key] = qaoa.cost(np.asarray(u) * scale)
        return cache[key]

    grid = xfacpy.QuanticsGrid(a=0.0, b=1.0, nBit=R, dim=d)
    param = xfacpy.TensorCI2Param()
    param.reltol = reltol
    param.bondDim = bond_dim
    # start the pivot at a typical good region (small gamma, beta ~ pi/8), not the corner
    u0 = [0.1] * p + [0.125] * p
    param.pivot1 = list(grid.coord_to_id(u0))
    ci = xfacpy.QTensorCI(f, grid, param)
    sweeps = 0
    while not ci.isDone() and sweeps < max_sweeps:
        ci.iterate(1, 2)
        sweeps += 1
    n_tci = qaoa.n_evals - n0

    # candidate basins = best distinct cached points
    items = sorted(cache.items(), key=lambda kv: kv[1])
    seen, cands = set(), []
    for u, v in items:
        tag = tuple(np.round(u, 2))
        if tag in seen:
            continue
        seen.add(tag)
        cands.append(np.asarray(u) * scale)
        if len(cands) >= top_k:
            break
    best = (cands[0], items[0][1])
    for x0 in cands:
        x, fx = _local(qaoa, x0, maxiter)
        if fx < best[1]:
            best = (x, fx)

    ranks = [c.shape[2] for c in ci.tt.core[:-1]]
    extra = {
        "tt_max_rank": int(max(ranks)),
        "pivot_error": float(ci.pivotError[-1]),
        "tci_sweeps": sweeps,
        "n_tci_evals": n_tci,
        "grid_points": float(2.0 ** (d * R)),
    }
    if return_tt:
        extra.update(tt=ci.tt, grid=grid, scale=scale)
    return _result(qaoa, *best, n0, extra)
