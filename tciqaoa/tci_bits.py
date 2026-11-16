"""
TCI directly on the binary landscape: the "tensor-network analogue" of QAOA.

QAOA prepares a state concentrated on low-energy bitstrings.  The classical
tensor-network counterpart is the Boltzmann tensor

    F(x_1, ..., x_n) = exp(-beta * (E(x) - E_ref)),   x_i in {0, 1},

an n-leg tensor with local dimension 2 (exactly the QAOA register, one leg per qubit).
TCI (xfacpy.TensorCI2) learns a tensor train for F from a few evaluations, and its
pivot search (rook / full pivoting on the largest interpolation error) is pulled
towards the largest entries of F, i.e. the lowest energies.  Raising beta in stages
plays the role of deeper QAOA layers / a slower anneal: it sharpens F around the
minimum.  The best cached configurations are finished with a 1-flip local search.

Nothing here needs the 2**n energy table, so it runs at sizes where statevector
QAOA cannot (n = 30..100+).

Variable order matters for a tensor train (the user's own QTCI finding): with
order='spectral' the variables are sorted by the Fiedler vector of |Q| so strongly
coupled variables sit next to each other on the chain, which lowers the TT rank.
"""

import numpy as np

from .problems import symmetrize, local_search

__all__ = ["variable_order", "tci_minimize_bits", "tci_sweep_bits"]


def variable_order(Q, order="spectral"):
    n = Q.shape[0]
    if order in (None, "identity"):
        return np.arange(n)
    W = np.abs(symmetrize(Q))
    np.fill_diagonal(W, 0.0)
    if order == "spectral":
        L = np.diag(W.sum(1)) - W
        _, V = np.linalg.eigh(L)
        return np.argsort(V[:, 1])
    if order == "rcm":
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        return np.asarray(reverse_cuthill_mckee(csr_matrix(W > 1e-12), symmetric_mode=True))
    raise ValueError(order)


def tci_minimize_bits(Q, betas=(0.5, 1.0, 2.0, 4.0), bond_dim=20, sweeps_per_beta=4,
                      order="spectral", n_polish=20, n_seeds=4, seed=0, verbose=False,
                      full_pivot=True):
    """Minimise x^T Q x over {0,1}^n with staged-beta TCI on exp(-beta*(E - E_ref)).

    beta is given in units of 1/(typical single-flip energy change), so the same
    schedule works for any scale of Q.

    Returns dict(x, energy, n_energy_evals, max_rank, history).
    """
    import xfacpy

    Qs = symmetrize(Q)
    n = Qs.shape[0]
    rng = np.random.default_rng(seed)
    perm = variable_order(Qs, order)            # chain position k  <->  variable perm[k]
    Qp = Qs[np.ix_(perm, perm)]
    # energy unit: rms single-flip energy change at a random state
    unit = float(np.sqrt(np.mean((2 * Qp @ rng.integers(0, 2, n) + np.diag(Qp)) ** 2))) or 1.0

    cache = {}

    def energy(bits):
        key = tuple(bits)
        e = cache.get(key)
        if e is None:
            x = np.asarray(bits, dtype=float)
            e = float(x @ Qp @ x)
            cache[key] = e
        return e

    # starting points: a few local-search minima from random states
    starts = []
    extra_evals = 0
    for _ in range(n_seeds):
        x, e, ev = local_search(rng.integers(0, 2, n), Qp)
        extra_evals += ev // n + 1          # in units of full energy evaluations
        starts.append((e, x.astype(int)))
    starts.sort(key=lambda t: t[0])
    best_e, best_x = starts[0]
    history, max_rank = [], 0

    for beta in betas:
        E_ref = best_e
        b = beta / unit
        f = lambda bits: float(np.exp(-b * (energy(bits) - E_ref)))
        param = xfacpy.TensorCI2Param()
        param.bondDim = bond_dim
        param.reltol = 1e-12
        param.fullPiv = full_pivot      # deterministic (rook search is history-dependent)
        param.nRookIter = 5
        param.pivot1 = [int(v) for v in best_x]
        ci = xfacpy.TensorCI2(f, [2] * n, param)
        # seed the other local minima as global pivots so TCI sees several basins
        others = [[int(v) for v in xs] for _, xs in starts if not np.array_equal(xs, best_x)]
        if others:
            ci.addPivotsAllBonds(others)
        for _ in range(sweeps_per_beta):
            ci.iterate(1, 2)
        max_rank = max(max_rank, max(c.shape[2] for c in ci.tt.core[:-1]))
        # polish the lowest-energy configurations TCI touched
        cand = sorted(cache.items(), key=lambda kv: kv[1])[:n_polish]
        for bits, _ in cand:
            x, e, ev = local_search(np.array(bits), Qp)
            extra_evals += ev // n + 1
            if e < best_e - 1e-12:
                best_e, best_x = e, x.astype(int)
        history.append((beta, best_e, len(cache)))
        if verbose:
            print(f"  beta={beta:<5} best E={best_e:.6f}  evals={len(cache)}  rank={max_rank}")

    x = np.empty(n)
    x[perm] = best_x                            # undo the variable permutation
    return {
        "x": x,
        "energy": float(x @ Qs @ x),
        "n_energy_evals": len(cache) + extra_evals,
        "n_tci_evals": len(cache),
        "max_rank": int(max_rank),
        "history": history,
    }


def tci_sweep_bits(Q_list, betas=(0.5, 1.0, 2.0, 4.0), bond_dim=20, sweeps_per_beta=4,
                   order="spectral", n_polish=10, seed=0, full_pivot=True):
    """One TCI for a whole family of QUBOs Q(t) (e.g. an efficient frontier over lambda).

    The tensor has R_t = log2(len(Q_list)) legs for t, followed by the n variable legs:

        F(t, x) = exp(-beta * (E_t(x) - E_ref(t)) / unit(t))

    Each slice is normalised by its own best-known energy and energy unit, so every t
    gets the same weight.  After each beta stage, every t polishes (1-flip local
    search) its ``n_polish`` lowest cached configurations plus the current best of its
    two neighbours in t; finally every t is tried from the best of every other t.
    Returns dict(x: list per t, energy: array per t, n_energy_evals, max_rank).
    """
    import xfacpy

    T = len(Q_list)
    R = int(round(np.log2(T)))
    assert 2 ** R == T, "len(Q_list) must be a power of 2"
    Qs = [symmetrize(Q) for Q in Q_list]
    n = Qs[0].shape[0]
    rng = np.random.default_rng(seed)
    perm = variable_order(sum(np.abs(Q) for Q in Qs), order)
    Qp = [Q[np.ix_(perm, perm)] for Q in Qs]
    units = []
    for Q in Qp:
        u = float(np.sqrt(np.mean((2 * Q @ rng.integers(0, 2, n) + np.diag(Q)) ** 2)))
        units.append(u or 1.0)

    cache = {}
    extra = [0]

    def energy(bits):
        key = tuple(bits)
        e = cache.get(key)
        if e is None:
            t = int("".join(str(int(b)) for b in bits[:R]), 2)
            x = np.asarray(bits[R:], dtype=float)
            e = float(x @ Qp[t] @ x)
            cache[key] = e
        return e

    def tbits(t):
        return [int(c) for c in format(t, f"0{R}b")]

    def polish(t, x):
        y, e, ev = local_search(np.asarray(x), Qp[t])
        extra[0] += ev // n + 1
        return y.astype(int), e

    best = [polish(t, rng.integers(0, 2, n)) for t in range(T)]     # (x, E) per t
    max_rank = 0
    for beta in betas:
        ref = [b[1] for b in best]

        def f(bits, beta=beta, ref=ref):
            t = int("".join(str(int(b)) for b in bits[:R]), 2)
            return float(np.exp(-beta * (energy(bits) - ref[t]) / units[t]))

        param = xfacpy.TensorCI2Param()
        param.bondDim = bond_dim
        param.reltol = 1e-12
        param.fullPiv = full_pivot      # deterministic (rook search is history-dependent)
        param.nRookIter = 5
        param.pivot1 = tbits(0) + [int(v) for v in best[0][0]]
        ci = xfacpy.TensorCI2(f, [2] * (R + n), param)
        ci.addPivotsAllBonds([tbits(t) + [int(v) for v in best[t][0]] for t in range(1, T)])
        for _ in range(sweeps_per_beta):
            ci.iterate(1, 2)
        max_rank = max(max_rank, max(c.shape[2] for c in ci.tt.core[:-1]))
        by_t = [[] for _ in range(T)]
        for bits, e in cache.items():
            by_t[int("".join(str(b) for b in bits[:R]), 2)].append((e, bits[R:]))
        new = []
        for t in range(T):
            cands = [np.array(b) for _, b in sorted(by_t[t])[:n_polish]]
            cands += [best[s][0] for s in (t - 1, t + 1) if 0 <= s < T]
            bt = best[t]
            for x0 in cands:
                y, e = polish(t, x0)
                if e < bt[1] - 1e-12:
                    bt = (y, e)
            new.append(bt)
        best = new
    # reuse across the family: start every t from every other t's best
    for t in range(T):
        for s in range(T):
            if s != t and not np.array_equal(best[s][0], best[t][0]):
                y, e = polish(t, best[s][0])
                if e < best[t][1] - 1e-12:
                    best[t] = (y, e)
    xs = []
    for y, _ in best:
        x = np.empty(n)
        x[perm] = y
        xs.append(x)
    return {"x": xs, "energy": np.array([float(x @ Q @ x) for x, Q in zip(xs, Qs)]),
            "n_energy_evals": len(cache) + extra[0], "n_tci_evals": len(cache),
            "max_rank": int(max_rank)}
