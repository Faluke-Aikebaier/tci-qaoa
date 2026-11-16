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

__all__ = ["set_local_optimizer", "spsa", "GAMMA_MAX", "BETA_MAX", "optimize_restarts", "optimize_interp", "optimize_tci",
           "landscape_minima", "optimize_from", "beta_images", "optimize_from_images"]

GAMMA_MAX = 2.0          # normalised cost (std = 1); optimal gammas observed <= ~1.9
BETA_MAX = np.pi         # exp(-i beta sum X) is pi-periodic up to a global phase


def _bounds(p):
    return [(0.0, GAMMA_MAX)] * p + [(0.0, BETA_MAX)] * p


_LOCAL = {"method": "cobyla", "spsa_iters": 100, "spsa_c": 0.1,
          "rng": np.random.default_rng(0)}


def set_local_optimizer(method="cobyla", spsa_iters=100, spsa_c=0.1, seed=0):
    """Choose the local optimiser behind every polish in this package.

    "cobyla" (default): scipy COBYLA, ``maxiter`` and ``rhobeg`` as passed.
    "spsa": simultaneous-perturbation stochastic approximation (Spall 1998), the
    standard choice for noisy (shot-based) objectives: ``spsa_iters`` iterations of
    2 circuits each whatever ``maxiter`` is, perturbation ``spsa_c``, first step
    min(``rhobeg``, 0.1) per coordinate (0.2, COBYLA's default radius, made SPSA diverge
    at p = 3 even without noise).  Every caller (restarts, INTERP, the sweep and the chain) uses it.
    """
    assert method in ("cobyla", "spsa")
    _LOCAL.update(method=method, spsa_iters=int(spsa_iters), spsa_c=float(spsa_c),
                  rng=np.random.default_rng(seed))


def spsa(f, x0, n_iter=100, step=0.2, c=0.1, rng=None, n_cal=2, g_floor=1.0):
    """Minimise a noisy f with SPSA.  Returns (x, f(x)).

    Standard gains a_k = a / (k + 1 + A)^0.602, c_k = c / (k + 1)^0.101, A = n_iter / 10;
    a is calibrated from ``n_cal`` gradient estimates at x0 so that the first step moves
    each coordinate by about ``step``, with the gradient scale floored at ``g_floor``
    (the normalised cost has gradients of order 1): near a minimum the calibration
    gradient is mostly shot noise, and the step size it implied made the iterates
    random-walk far from warm starts.  Every update is also clipped to 2 ``step`` per
    coordinate.  The returned x is the mean of the iterates over
    the second half (iterate averaging, which damps the noise in the final angles); one
    more evaluation gives f(x).  Cost: 2 n_cal + 2 n_iter + 1 evaluations.
    """
    rng = np.random.default_rng(rng)
    x = np.asarray(x0, dtype=float).copy()
    d = len(x)
    A = 0.1 * n_iter
    g = []
    for _ in range(n_cal):
        delta = rng.choice([-1.0, 1.0], d)
        g.append(abs(f(x + c * delta) - f(x - c * delta)) / (2 * c))
    a = step * (A + 1) ** 0.602 / max(float(np.mean(g)), g_floor)
    tail = []
    for k in range(n_iter):
        ak, ck = a / (k + 1 + A) ** 0.602, c / (k + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], d)
        gk = (f(x + ck * delta) - f(x - ck * delta)) / (2 * ck) * delta
        x = x - np.clip(ak * gk, -2 * step, 2 * step)
        if k >= n_iter // 2:
            tail.append(x.copy())
    x = np.mean(tail, axis=0)
    return x, float(f(x))


def _local(qaoa, x0, maxiter, rhobeg=0.2):
    if _LOCAL["method"] == "spsa":
        return spsa(qaoa.cost, x0, _LOCAL["spsa_iters"], min(rhobeg, 0.1), _LOCAL["spsa_c"],
                    _LOCAL["rng"])
    r = minimize(qaoa.cost, x0, method="COBYLA",
                 options={"maxiter": maxiter, "rhobeg": rhobeg})
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
                 max_sweeps=20, seed=0, return_tt=False, full_pivot=True):
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
    param.fullPiv = full_pivot      # deterministic; xfacpy's rook search is history-dependent
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
        extra.update(tt=ci.tt, grid=grid, scale=scale, cache=cache)
    return _result(qaoa, *best, n0, extra)


# ---------------------------------------------------------------- reuse of a learned TT
def landscape_minima(tci_result, top_k=3, min_sep=0.1, max_enumerate=2 ** 16):
    """Distinct low points of a stored TCI landscape, as angle vectors.

    ``tci_result`` must come from ``optimize_tci(..., return_tt=True)``.  If the grid is
    small enough the tensor train is evaluated at every grid point (no circuit runs);
    otherwise the best points TCI sampled are used.  Points closer than ``min_sep``
    (in the unit box) to an already chosen one are skipped.
    """
    tt, grid, scale = tci_result["tt"], tci_result["grid"], tci_result["scale"]
    d = len(scale)
    R = grid.tensorLen // d
    n_pts = 2 ** (d * R)
    if n_pts <= max_enumerate:
        axis = np.arange(2 ** R) / 2 ** R
        mesh = np.stack(np.meshgrid(*[axis] * d, indexing="ij"), -1).reshape(-1, d)
        vals = np.array([tt.eval(grid.coord_to_id(list(u))) for u in mesh])
        order = np.argsort(vals)
        pts = mesh[order]
    else:
        items = sorted(tci_result["cache"].items(), key=lambda kv: kv[1])
        pts = np.array([u for u, _ in items])
    chosen = []
    for u in pts:
        if all(np.max(np.abs(u - c)) >= min_sep for c in chosen):
            chosen.append(u)
            if len(chosen) >= top_k:
                break
    return [c * scale for c in chosen]


def optimize_from(qaoa, starts, maxiter=300, rhobeg=0.05):
    """COBYLA from each given start angle vector on this instance; best result wins."""
    n0 = qaoa.n_evals
    best = (None, np.inf)
    for x0 in starts:
        x, f = _local(qaoa, np.asarray(x0, dtype=float), maxiter, rhobeg)
        if f < best[1]:
            best = (x, f)
    return _result(qaoa, *best, n0)


def beta_images(angles):
    """The 2**p images of an angle vector under beta_k -> beta_k + pi/2 (mod pi).

    If the cost is Z2 symmetric (C(x) = C(1 - x), as for MaxCut), X^n commutes with C
    and with the initial |+> state, and exp(-i (beta + pi/2) sum X) is exp(-i beta sum X)
    times X^n up to a phase.  So shifting any subset of the betas by pi/2 leaves <C>
    unchanged: all 2**p images are the same optimum.  On a cost without that symmetry
    (a QUBO with linear terms) they are different points, and in general only one of
    them is in the best basin.  The first image is the input itself.
    """
    x = np.asarray(angles, dtype=float)
    p = len(x) // 2
    out = []
    for mask in range(2 ** p):
        y = x.copy()
        for k in range(p):
            if mask >> k & 1:
                y[p + k] = (y[p + k] + np.pi / 2) % np.pi
        out.append(y)
    return out


def optimize_from_images(qaoa, angles, top_k=1, maxiter=300, rhobeg=0.05):
    """Reuse a stored optimum across problem families: rank its beta images by one
    circuit run each on this instance, then polish the top_k (see ``beta_images``).
    Costs 2**p runs more than ``optimize_from(qaoa, [angles])`` when top_k = 1."""
    n0 = qaoa.n_evals
    imgs = beta_images(angles)
    f = [qaoa.cost(y) for y in imgs]
    order = np.argsort(f, kind="stable")[:top_k]
    best = (None, np.inf)
    for i in order:
        x, fx = _local(qaoa, imgs[i], maxiter, rhobeg)
        if fx < best[1]:
            best = (x, fx)
    return _result(qaoa, *best, n0)
