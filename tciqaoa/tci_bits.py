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

__all__ = ["variable_order", "tci_minimize_bits"]


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
                      order="spectral", n_polish=20, n_seeds=4, seed=0, verbose=False):
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
        param.fullPiv = False
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
