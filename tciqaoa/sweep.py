"""
TCI over (QAOA angles x a problem parameter): one tensor train for a whole family.

A parameterised family Q(t), t in [0, 1] (a risk-aversion sweep, a penalty weight, a
field strength, an interpolation between two problems...) has a landscape
<C>(gamma, beta; t).  Treating t as one more quantics dimension, a single TCI learns
every member of the family at once.  For any t on the grid, the minima of that slice
are read off the tensor train with no circuit runs, then polished on the real circuit.

This is the use a stored point cannot cover: when the optimum moves or switches basin
as t changes, the tensor train still holds every basin for every t.
"""

import itertools

import numpy as np

from .problems import all_energies
from .qaoa import QAOA
from .angles import BETA_MAX, _bounds, _local, optimize_from, optimize_interp

__all__ = ["SweepFamily", "SweepFamily2D", "SweepFamilyND", "tci_sweep", "sweep_minima", "solve_from_sweep", "lift",
           "solve_lifted", "sweep_library", "sweep_chain", "SEEDINGS"]


class SweepFamily:
    """Wraps t -> Q(t) and caches the 2**n energy table per t (t is snapped to the grid).

    ``shots`` (optional): every circuit from ``qaoa`` estimates <C> from that many
    sampled bitstrings (shot noise), drawing from one generator seeded by ``seed``.
    """

    def __init__(self, make_Q, R_t=5, shots=None, seed=0):
        self.make_Q = make_Q
        self.R_t = R_t
        self.shots = shots
        self.rng = np.random.default_rng(seed)
        self._E = {}

    def t_grid(self):
        return np.arange(2 ** self.R_t) / 2 ** self.R_t

    def energies(self, t):
        key = round(float(t) * 2 ** self.R_t)
        if key not in self._E:
            self._E[key] = all_energies(self.make_Q(key / 2 ** self.R_t))
        return self._E[key]

    def qaoa(self, t, p):
        tq = round(float(t) * 2 ** self.R_t) / 2 ** self.R_t
        return QAOA(self.make_Q(tq), p, energies=self.energies(tq), shots=self.shots,
                    rng=self.rng if self.shots else None)

    # one sweep parameter: the point t is the parameter itself
    n_params = 1

    @property
    def R_param(self):
        return self.R_t

    def encode(self, params):
        return float(params[0])

    def decode(self, t):
        return (round(float(t) * 2 ** self.R_t) / 2 ** self.R_t,)


class SweepFamilyND(SweepFamily):
    """A family Q(t_1, ..., t_k), each t_j in [0, 1) on 2**R points (a 2**R-per-side grid).

    Each grid point is encoded as one scalar t (the k grid indices as base-2**R digits,
    t_1 most significant), so everything that takes a sweep point t (``qaoa``,
    ``sweep_library``, ``sweep_chain``) works unchanged; ``points(nt)`` gives an nt**k
    subgrid in snake (boustrophedon) order, so consecutive points are grid neighbours and
    the chain carries basins along the path.  ``tci_sweep`` learns one TT over
    (angles, t_1, ..., t_k).
    """

    def __init__(self, make_Qk, n_params, R=5, shots=None, seed=0):
        self.make_Qk = make_Qk
        self.n_params = n_params
        self.R = R
        super().__init__(lambda t: make_Qk(*self.decode(t)), R_t=n_params * R, shots=shots, seed=seed)

    @property
    def R_param(self):
        return self.R

    def encode(self, params):
        k = 0
        for x in params:
            k = (k << self.R) + round(float(x) * 2 ** self.R)
        return k / 2 ** (self.n_params * self.R)

    def decode(self, t):
        k = round(float(t) * 2 ** (self.n_params * self.R))
        m = 2 ** self.R - 1
        return tuple(((k >> (self.R * (self.n_params - 1 - j))) & m) / 2 ** self.R
                     for j in range(self.n_params))

    def points(self, nt):
        """nt**k evenly spaced grid points in snake order, as encoded scalars.

        A digit is reversed when the (already reflected) digits before it sum to an odd
        number, so exactly one coordinate moves by one step between neighbours.
        """
        out = []
        for idx in np.ndindex(*[nt] * self.n_params):
            e, s = [], 0
            for d in idx:
                e.append(nt - 1 - d if s % 2 else d)
                s += e[-1]
            out.append(self.encode([x / nt for x in e]))
        return out


class SweepFamily2D(SweepFamilyND):
    """A two-parameter family Q(t1, t2) on a 2**R x 2**R grid (see ``SweepFamilyND``)."""

    def __init__(self, make_Q2, R=5, shots=None, seed=0):
        self.make_Q2 = make_Q2
        super().__init__(make_Q2, 2, R=R, shots=shots, seed=seed)


def tci_sweep(family, p, gamma_max=2.5, bond_dim=30, reltol=1e-2, max_sweeps=20, box=None,
              full_pivot=True, angle_pivots=0):
    """Learn <C>(gamma, beta; t) for the whole family with one quantics TCI.

    Angles and t share R = family.R_t bits each.  For a ``SweepFamilyND`` the TT has
    k parameter dimensions (t_1 .. t_k), R = family.R bits each, and the result's
    ``decode`` maps a sweep point back to (t_1, .., t_k).  Returns dict(tt, grid, scale, p,
    n_evals, tt_max_rank, pivot_error).  n_evals counts circuit evaluations.
    ``box=(lo, hi)`` replaces the default angle box [0, gamma_max]^p x [0, pi)^p.
    ``full_pivot`` (default) uses full pivoting: xfacpy's rook search draws from an
    internal random generator that is not reset between calls, so with rook pivoting
    the same build gives a different TT (rank 4 .. 17 on one instance) depending on what
    ran before in the process, and sometimes stops after one sweep.  Full pivoting is
    deterministic and costs ~2k circuit runs at p = 1.
    ``angle_pivots=m`` (default 0, as in the benchmarks) also starts the build from an
    m**(2p) grid of points over the angle box, at the middle of the parameter range.  A
    family whose landscape looks flat near the default start can otherwise converge
    early around one basin and miss the optimum (a portfolio family with the default
    budget penalty: rank 4 after 224 runs; with m = 4, rank 11 after 1.9k runs and no
    misses).  The entry points in ``tciqaoa.solve`` use m = 4.
    """
    import xfacpy

    npar = getattr(family, "n_params", 1)
    d = 2 * p + npar
    R = family.R_param
    scale = np.array([gamma_max] * p + [BETA_MAX] * p)
    lo, hi = (np.zeros(2 * p), scale) if box is None else map(np.asarray, box)
    width = hi - lo
    qs = {}
    cache = {}

    def f(u):
        key = tuple(u)
        if key not in cache:
            t = family.encode(u[2 * p:])
            k = round(t * 2 ** family.R_t)
            if k not in qs:
                qs[k] = family.qaoa(t, p)
            cache[key] = qs[k].cost(lo + np.asarray(u[:2 * p]) * width)
        return cache[key]

    grid = xfacpy.QuanticsGrid(a=0.0, b=1.0, nBit=R, dim=d)
    param = xfacpy.TensorCI2Param()
    param.reltol = reltol
    param.bondDim = bond_dim
    param.fullPiv = full_pivot
    a0 = [0.4] * p + [0.85] * p
    param.pivot1 = list(grid.coord_to_id(a0 + [0.5] * npar))
    ci = xfacpy.QTensorCI(f, grid, param)
    # several global pivots along t, so the t direction is explored from the start
    if npar == 1:
        ci.addPivotPoints([a0 + [t] for t in (0.0, 0.25, 0.75, 31 / 32)])
    else:
        ci.addPivotPoints([a0 + list(c) for c in itertools.product((0.0, 0.5, 0.75), repeat=npar)
                           if c != (0.5,) * npar])
    if angle_pivots:
        # global pivots spread over the angle box (m points per angle, cell centres) at
        # the middle of the parameter range, so that a basin away from a0 is in the TT
        # from the first sweep (without them a family whose landscape looks flat near
        # a0 can converge early, at rank 4, and miss the optimum entirely)
        ax = (np.arange(angle_pivots) + 0.5) / angle_pivots
        ci.addPivotPoints([list(c) + [0.5] * npar for c in itertools.product(ax, repeat=2 * p)])
    sweeps = 0
    while not ci.isDone() and sweeps < max_sweeps:
        ci.iterate(1, 2)
        sweeps += 1
    return {"tt": ci.tt, "grid": grid, "scale": scale, "p": p, "R": R, "n_params": npar,
            "decode": family.decode,
            "n_evals": len(cache), "tt_max_rank": int(max(c.shape[2] for c in ci.tt.core[:-1])),
            "pivot_error": float(ci.pivotError[-1]), "cache": cache,
            "to_angles": lambda u, t: lo + np.asarray(u) * width}


def sweep_minima(res, t, top_k=3, min_sep=0.1, coarse_bits=None):
    """Lowest distinct minima of the slice at parameter t, read from the tensor train only.

    The slice is evaluated on a grid with ``coarse_bits`` bits per angle (default: the
    full grid when it has at most 2**16 points), and the points TCI itself sampled at
    this t are added as candidates.  Returns absolute angle vectors.
    """
    tt, grid, p, R = res["tt"], res["grid"], res["p"], res["R"]
    d = 2 * p
    cb = coarse_bits or min(R, 16 // d)
    axis = np.arange(2 ** cb) / 2 ** cb
    mesh = np.stack(np.meshgrid(*[axis] * d, indexing="ij"), -1).reshape(-1, d)
    if "decode" in res:
        par = list(res["decode"](t))
    else:
        par = [round(float(t) * 2 ** R) / 2 ** R]
    tq = par[-1] if len(par) == 1 else t
    sampled = np.array([u[:d] for u in res["cache"]
                        if np.max(np.abs(np.asarray(u[d:]) - par)) < 1e-12]).reshape(-1, d)
    mesh = np.vstack([mesh, sampled])
    vals = np.array([tt.eval(grid.coord_to_id(list(u) + par)) for u in mesh])
    chosen = []
    for i in np.argsort(vals):
        u = mesh[i]
        if all(np.max(np.abs(u - c)) >= min_sep for c in chosen):
            chosen.append(u)
            if len(chosen) >= top_k:
                break
    to_angles = res.get("to_angles") or (lambda u, t: np.asarray(u) * res["scale"])
    return [to_angles(c, tq) for c in chosen]


def solve_from_sweep(res, family, t, top_k=3):
    """Polish the slice minima at t on the real circuit. Returns optimize_from's dict."""
    return optimize_from(family.qaoa(t, res["p"]), sweep_minima(res, t, top_k))


def lift(angles, p_from, p_to):
    """INTERP rule (Zhou et al. 2020): depth-p angles -> depth-(p+1) starting angles, repeated."""
    a = np.asarray(angles, dtype=float)
    g, b = a[:p_from], a[p_from:]
    for k in range(p_from + 1, p_to + 1):
        def interp(v):
            old = np.concatenate([[0.0], v, [0.0]])
            i = np.arange(1, k + 1)
            return (i - 1) / (k - 1) * old[i - 1] + (k - i) / (k - 1) * old[i]
        g, b = interp(g), interp(b)
    return np.concatenate([g, b])


def solve_lifted(p1res, family, t, p, top_k=3):
    """Depth-p solve at parameter t from a converged depth-1 sweep TT.

    The depth-p landscape is not low rank (benchmark 5), but the depth-1 one is, and it
    knows which basin is best at every t.  Its ``top_k`` slice minima are lifted to depth
    p with the INTERP rule and polished on the real depth-p circuit.
    """
    assert p1res["p"] == 1
    starts = [lift(a, 1, p) for a in sweep_minima(p1res, t, top_k)]
    return optimize_from(family.qaoa(t, p), starts)


def _restart_box(p, gamma_max, gamma_other):
    """Symmetry-reduced sampling box for depth-p restarts.

    Time reversal (gamma, beta) -> (-gamma, -beta) fixes the sign of one gamma only, so
    for p >= 2 the later gammas must be sampled with both signs (some optimal depth-2
    basins have gamma_2 < 0 or gamma_2 ~ 3, outside [0, GAMMA_MAX]).
    """
    lo = np.r_[0.0, [-gamma_other] * (p - 1), [0.0] * p]
    hi = np.r_[gamma_max, [gamma_other] * (p - 1), [BETA_MAX] * p]
    return lo, hi


def _tci_seed(qaoa, lo, hi, R=4, bond_dim=8, max_sweeps=4, top_k=4, reltol=1e-2,
              full_pivot=True):
    """Quantics TCI over the angle box used as a global *sampler*, not a surrogate.

    The depth-p landscape is too high rank for an accurate TT, but TCI's pivot search
    still visits the low regions of the whole box.  Returns the ``top_k`` best distinct
    points it evaluated (as angle vectors).
    """
    import xfacpy

    d = len(lo)
    cache = {}

    def f(u):
        key = tuple(u)
        if key not in cache:
            cache[key] = qaoa.cost(lo + np.asarray(u) * (hi - lo))
        return cache[key]

    grid = xfacpy.QuanticsGrid(a=0.0, b=1.0, nBit=R, dim=d)
    param = xfacpy.TensorCI2Param()
    param.reltol = reltol
    param.bondDim = bond_dim
    param.fullPiv = full_pivot           # deterministic (rook search is history-dependent)
    param.pivot1 = list(grid.coord_to_id([0.4] + [0.5] * (d // 2 - 1) + [0.85] * (d // 2)))
    ci = xfacpy.QTensorCI(f, grid, param)
    for _ in range(max_sweeps):
        if ci.isDone():
            break
        ci.iterate(1, 2)
    out = []
    for u, _ in sorted(cache.items(), key=lambda kv: kv[1]):
        if all(np.max(np.abs(np.asarray(u) - o)) >= 0.1 for o in out):
            out.append(np.asarray(u))
            if len(out) >= top_k:
                break
    return [lo + o * (hi - lo) for o in out]


SEEDINGS = ("restarts", "tci", "tci+restarts", "tci-sweep", None)


def sweep_library(p1res, family, ts, p, seeding="tci", anchors=None, n_restarts=4,
                  top_k=3, n_lib=3, bidirectional=True, back_margin=0.3, gamma_max=2.5,
                  gamma_other=3.0, maxiter=300, seed=1, tci_bond_dim=8, sweep_bond_dim=15,
                  lower=None, n_seed_lib=0, extra_starts=None):
    """Solve a whole family at depth p with a shared basin library (closes the p >= 2 gap).

    Some depth-p basins are not reached from any depth-1 minimum, so lifting alone misses
    them.  They are, however, family-wide (angle concentration along t), so they only have
    to be found once:

      1. seeding at each anchor t (default every other t: some basins are optimal only
         over a short range of t, so spreading the search beats concentrating it), in the
         symmetry-reduced box (``_restart_box``); every distinct polished optimum goes
         into the library.  ``seeding`` chooses how:
           "restarts"   ``n_restarts`` random starts per anchor (classical);
           "tci"        (default) a small quantics TCI per anchor used as a global
                        sampler; its ``n_restarts`` best distinct points are polished.
                        Best placement of TCI found in benchmark 6: 185/192 with worst
                        miss 0.024, vs 178.6 +- 3.6 for random restarts at equal cost;
           "tci+restarts"  both: the ``n_restarts`` random starts (drawn as for
                        "restarts" with the same seed) and the TCI sampler's points;
           "tci-sweep"  one depth-p TCI over (angles, t) (not converged: a sampler);
                        the ``n_restarts`` lowest slice minima at each anchor are polished;
           None         no seeding;
      2. forward pass over ``ts``: polish the lifted depth-1 TT slice minima (skipped when
         ``p1res`` is None, the no-TCI ablation), the previous t's optimum, and the
         ``n_lib`` library members with the lowest cost at this t (one circuit run each);
         the optimum found joins the library;
      3. backward pass: re-polish t from its neighbour's optimum when that is a different
         basin whose start cost is within ``back_margin`` of the stored result.

    ``lower`` (optional) is a ``sweep_library`` result at depth p - 1 on the same ``ts``.
    Its optimum at each t is lifted to depth p (INTERP rule) and polished as a per-t
    candidate ("lift"), and every member of its library is lifted into this library
    (unpolished; they compete in the one-run ranking).  This chains depths: p = 1 TT ->
    p = 2 library -> p = 3.  Its own cost is not included in ``n_evals``.

    ``n_seed_lib`` > 0 reserves that many shortlist slots for seeding entries: at each t
    the ``n_seed_lib`` best seeded members are polished in addition to the ``n_lib``
    best of the others, so fresh depth-p seeds cannot push lifted basins out of the
    shortlist (with 0, all members compete for the ``n_lib`` slots).  Optima reached
    from a seeding entry along the sweep inherit its seed lineage and also compete only
    for the reserved slots (benchmark 9: otherwise they displaced the lifted basins), and
    the best result from a non-seed start is carried along t as well (library entry and
    "prev" start), so the non-seed line evolves as it would without seeding.

    ``extra_starts`` (optional): one more start angle vector per t (e.g. INTERP's
    optimum at that t), polished at every t with source tag "interp".

    Returns dict(costs, angles, n_evals, n_seed_evals, source, library, p) with one entry per t in
    ``ts``; ``source[i]`` says which kind of start produced the optimum at ts[i]: "tt"
    (lifted depth-1 TT minimum), "seed" (library entry from seeding), "sweep" (library
    entry found at another t), "prev" (previous t's optimum) or "back" (backward pass).
    """
    assert p1res is None or p1res["p"] == 1
    assert seeding in SEEDINGS
    ts = list(ts)
    anchors = ts[::2] if anchors is None else anchors
    n_evals, lib, origin, sline = 0, [], [], []
    lo, hi = _restart_box(p, gamma_max, gamma_other)
    rng = np.random.default_rng(seed)

    def add(x, tag, seed_line=False):
        x = np.asarray(x, dtype=float)
        if all(np.max(np.abs(x - o)) > 0.1 for o in lib):
            lib.append(x)
            origin.append(tag)
            sline.append(seed_line or tag == "seed")

    if seeding == "tci-sweep":
        sw = tci_sweep(family, p, bond_dim=sweep_bond_dim, box=(lo, hi))
        n_evals += sw["n_evals"]
    for ta in (anchors if seeding else []):
        q = family.qaoa(ta, p)
        if seeding == "restarts":
            starts = [rng.uniform(lo, hi) for _ in range(n_restarts)]
        elif seeding == "tci":
            starts = _tci_seed(q, lo, hi, bond_dim=tci_bond_dim, top_k=n_restarts)
        elif seeding == "tci+restarts":
            starts = [rng.uniform(lo, hi) for _ in range(n_restarts)]
            starts += _tci_seed(q, lo, hi, bond_dim=tci_bond_dim, top_k=n_restarts)
        else:
            starts = sweep_minima(sw, ta, n_restarts)
        for x0 in starts:
            add(_local(q, x0, maxiter)[0], "seed")
        n_evals += q.n_evals
    n_seed = n_evals
    if lower is not None:
        assert lower["p"] == p - 1 and len(lower["angles"]) == len(ts)
        for x in lower["library"]:
            add(lift(x, p - 1, p), "lift")

    found, ang, src = {}, {}, {}
    prev, prev_sl, prev_main = None, False, None
    for i_t, t in enumerate(ts):
        q = family.qaoa(t, p)
        v = [q.cost(c) for c in lib]
        starts = [(lift(a, 1, p), "tt", False) for a in sweep_minima(p1res, t, top_k)] \
            if p1res else []
        if lower is not None:
            starts.append((lift(lower["angles"][i_t], p - 1, p), "lift", False))
        if extra_starts is not None:
            starts.append((np.asarray(extra_starts[i_t], dtype=float), "interp", False))
        if n_seed_lib:
            order = np.argsort(v)
            seeded = [i for i in order if sline[i]][:n_seed_lib]
            others = [i for i in order if not sline[i]][:n_lib]
            starts += [(lib[i], origin[i], sline[i]) for i in others + seeded]
        else:
            starts += [(lib[i], origin[i], False) for i in np.argsort(v)[:n_lib]]
        if prev is not None:
            starts.append((prev, "prev", prev_sl))
        if prev_main is not None and prev_sl:
            starts.append((prev_main, "prev", False))
        if not starts:          # nothing to start from (no TT, no seeding, first t)
            starts = [(rng.uniform(lo, hi), "seed", False) for _ in range(n_restarts)]
        best, best_main = (None, np.inf, None, False), (None, np.inf)
        for x0, tag, sl in starts:
            x, fx = _local(q, x0, maxiter, 0.05)
            if fx < best[1]:
                best = (x, fx, tag, sl)
            if not sl and fx < best_main[1]:
                best_main = (x, fx)
        n_evals += q.n_evals
        ang[t], found[t], src[t], prev_sl = best
        prev_sl = prev_sl and bool(n_seed_lib)
        if prev_sl and best_main[0] is not None:
            add(best_main[0], "sweep")          # keep the non-seed line going
        add(ang[t], "sweep", prev_sl)
        prev, prev_main = ang[t], (best_main[0] if prev_sl else ang[t])
    if bidirectional:
        prev = None
        for t in ts[::-1]:
            q = family.qaoa(t, p)
            if prev is not None and np.max(np.abs(prev - ang[t])) > 0.1 \
                    and q.cost(prev) < found[t] + back_margin:
                x, fx = _local(q, prev, maxiter, 0.05)
                if fx < found[t]:
                    found[t], ang[t], src[t] = fx, x, "back"
            n_evals += q.n_evals
            prev = ang[t]
    return {"costs": np.array([found[t] for t in ts]), "angles": [ang[t] for t in ts],
            "n_evals": n_evals, "n_seed_evals": n_seed, "source": [src[t] for t in ts],
            "library": [np.array(x) for x in lib], "p": p}


def sweep_chain(family, ts, p, extra_restarts=8, gamma_wide=4.5, n_seed_lib=2, seed=1,
                tt=None, interp_start=True, use_tci=True, n_anchors=None, warm_start=True):
    """Recommended pipeline for a whole family at depth p (benchmarks 5, 6, 6c, 6d).

    Chains the depths: one depth-1 TCI sweep TT, then depth 2 with ``sweep_library``
    seeded by the TCI sampler, then each depth d = 3 .. p from the depth-(d-1) result
    lifted with the INTERP rule (``lower=``).  From depth 3 on, ``extra_restarts`` random
    restarts per anchor are added in a wider box (later gammas in +-``gamma_wide``), with
    ``n_seed_lib`` reserved shortlist slots so they cannot displace the lifted basins.
    ``extra_restarts=0`` gives the deterministic chain (benchmark 9, n = 10, p = 3:
    171/192 vs 180.0 on average with 8 restarts, for ~33% more circuit runs).  With ``interp_start`` (default)
    INTERP is also run from scratch at every t at the final depth and its optimum is one
    more start there: the chain can carry a basin past the point where it stops being
    best (benchmark 9), and INTERP costs ~25% of the chain.

    ``use_tci=False`` is the no-TCI ablation: no depth-1 TT, and depth 2 seeded by 4
    random restarts per anchor instead of the TCI sampler (needs p >= 2).
    ``use_tci="hybrid"`` merges the two into one run: the depth-1 TT starts as with TCI,
    and depth 2 seeded by both the 4 random restarts (the same ones as ``use_tci=False``
    with this ``seed``) and the TCI sampler's points, in one shared library.

    ``n_anchors`` (optional) seeds only that many evenly spaced t values at each depth
    instead of every other t (``sweep_library``'s default), so the seeding cost stops
    growing with the number of sweep points (benchmark 10, dense sweeps).

    p = 1 is the sweep TT alone (``solve_from_sweep`` at every t); with ``warm_start``
    (default) the previous sweep point's optimum is polished as one more start, so a
    slice minimum that the TT gets wrong cannot cost a point the neighbour already
    solved (benchmark 12: 8 misses of 3,456 without it).  Pass a prebuilt
    depth-1 ``tt`` to share it.  Returns dict(costs, angles, n_evals, by_depth): costs
    and angles at depth p per t, total circuit runs over every stage, and the per-depth
    ``sweep_library`` results (depth >= 2).
    """
    ts = list(ts)
    if not use_tci:
        if p < 2:
            raise ValueError("use_tci=False needs p >= 2")
        tt = None
    elif tt is None:
        tt = tci_sweep(family, 1)
    n_evals = tt["n_evals"] if tt is not None else 0
    anchors = None
    if n_anchors and n_anchors < len(ts):
        anchors = [ts[i] for i in np.unique(np.round(np.linspace(0, len(ts) - 1, n_anchors)).astype(int))]
    if p == 1:
        rs = []
        for t in ts:
            starts = sweep_minima(tt, t, 3)
            if warm_start and rs:
                starts = list(starts) + [rs[-1]["angles"]]
            rs.append(optimize_from(family.qaoa(t, 1), starts))
        return {"costs": np.array([r["cost"] for r in rs]), "angles": [r["angles"] for r in rs],
                "n_evals": n_evals + sum(r["n_evals"] for r in rs), "by_depth": {}}
    if use_tci == "hybrid":
        res = sweep_library(tt, family, ts, 2, seeding="tci+restarts", seed=seed, anchors=anchors)
    elif use_tci:
        res = sweep_library(tt, family, ts, 2, seeding="tci", anchors=anchors)
    else:
        res = sweep_library(None, family, ts, 2, seeding="restarts", seed=seed, anchors=anchors)
    n_evals += res["n_evals"]
    by_depth = {2: res}
    for d in range(3, p + 1):
        if extra_restarts:
            kw = dict(seeding="restarts", n_restarts=extra_restarts, n_seed_lib=n_seed_lib,
                      gamma_other=gamma_wide)
        else:
            kw = dict(seeding=None)
        if interp_start and d == p:
            its = [optimize_interp(lambda k, t=t: family.qaoa(t, k), p, seed=seed) for t in ts]
            n_evals += sum(r["n_evals"] for r in its)
            kw["extra_starts"] = [r["angles"] for r in its]
        res = sweep_library(None, family, ts, d, lower=res, seed=seed, anchors=anchors, **kw)
        n_evals += res["n_evals"]
        by_depth[d] = res
    return {"costs": res["costs"], "angles": res["angles"], "n_evals": n_evals,
            "by_depth": by_depth}
