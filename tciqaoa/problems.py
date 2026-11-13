"""
QUBO utilities shared by every solver.

Convention (identical to hybridq): minimise  E(x) = x^T Q_sym x,  x in {0,1}^n,
with Q_sym = (Q + Q^T)/2.  Bit i of a basis-state index z is variable x_i,
i.e. x_i = (z >> i) & 1.
"""

import numpy as np

__all__ = [
    "symmetrize", "qubo_energy", "all_energies", "brute_force",
    "maxcut_qubo", "cut_value", "random_qubo", "random_graph",
    "local_search",
]


def symmetrize(Q):
    Q = np.asarray(Q, dtype=float)
    return 0.5 * (Q + Q.T)


def qubo_energy(x, Q):
    Qs = symmetrize(Q)
    x = np.asarray(x, dtype=float)
    return float(x @ Qs @ x)


def all_energies(Q, chunk_bits=16):
    """Energy of every one of the 2**n bitstrings, as a float64 array indexed by z.

    Exact and vectorised; fine up to n ~ 24 (2**24 doubles = 128 MB).
    """
    Qs = symmetrize(Q)
    n = Qs.shape[0]
    N = 1 << n
    out = np.empty(N)
    step = 1 << min(n, chunk_bits)
    bits = np.arange(n)
    for start in range(0, N, step):
        z = np.arange(start, min(start + step, N), dtype=np.int64)
        X = ((z[:, None] >> bits) & 1).astype(float)          # (chunk, n)
        out[start:start + len(z)] = np.einsum("ki,ij,kj->k", X, Qs, X)
    return out


def brute_force(Q):
    """Exact ground state. Returns (x_opt, E_opt, energies)."""
    E = all_energies(Q)
    z = int(np.argmin(E))
    n = symmetrize(Q).shape[0]
    x = np.array([(z >> i) & 1 for i in range(n)], dtype=float)
    return x, float(E[z]), E


def maxcut_qubo(A):
    """MaxCut as a QUBO, same formulation as hybridq: Q = -(D - A)."""
    A = symmetrize(A)
    return -(np.diag(A.sum(axis=1)) - A)


def cut_value(x, A):
    A = symmetrize(A)
    x = np.asarray(x)
    diff = x[:, None] != x[None, :]
    return float((A * diff).sum() / 2)


# Generators copied from hybridq (bit-for-bit identical instances for a fair comparison)
def random_qubo(n, density=0.5, seed=42):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((n, n))
    mask = rng.random((n, n)) > density
    Q[mask] = 0.0
    return Q


def random_graph(n, edge_prob=0.3, seed=42):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < edge_prob).astype(float)
    A = np.triu(A, 1)
    return A + A.T


def local_search(x, Q, max_passes=100):
    """Steepest-descent single-bit-flip polish. Returns (x, E, n_flip_evals)."""
    Qs = symmetrize(Q)
    x = np.asarray(x, dtype=float).copy()
    d = np.diag(Qs)
    Qx = Qs @ x
    E = float(x @ Qx)
    evals = 0
    for _ in range(max_passes):
        # dE for flipping bit i: delta = 1 - 2x_i ; dE = delta*(2 Qx_i) + delta^2 * Q_ii
        delta = 1.0 - 2.0 * x
        dE = 2.0 * delta * Qx + d
        evals += len(x)
        i = int(np.argmin(dE))
        if dE[i] >= -1e-12:
            break
        x[i] += delta[i]
        Qx += Qs[:, i] * delta[i]
        E += float(dE[i])
    return x, E, evals
