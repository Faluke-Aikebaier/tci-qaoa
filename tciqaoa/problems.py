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
    "local_search", "portfolio_data", "portfolio_qubo",
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


def portfolio_data(n, seed=42):
    """hybridq's ``examples.make_portfolio_data``: n assets with sector structure.

    Covariance = F F^T (n // 7 sector factors) + idiosyncratic diagonal; returns
    (mu, sigma) with expected returns mu ~ U(0.05, 0.20).
    """
    rng = np.random.default_rng(seed)
    n_sec = max(2, n // 7)
    F = rng.standard_normal((n, n_sec))
    sigma = F @ F.T + np.diag(np.abs(rng.standard_normal(n)) * 0.05)
    sigma += np.eye(n) * 0.02
    mu = rng.uniform(0.05, 0.20, n)
    return mu, sigma


def portfolio_qubo(mu, sigma, k, risk_aversion=1.0, penalty=None):
    """Markowitz selection of exactly k assets as a QUBO.

    E(x) = lambda x^T Sigma x - mu_n^T x + A (sum x - k)^2 - A k^2,  mu_n = mu / max|mu|.
    For binary x the linear return is the diagonal term -diag(mu_n) (x_i^2 = x_i).
    (hybridq's ``solve_portfolio`` uses -mu_n mu_n^T, i.e. the *squared* return.)

    The default penalty A = 2 (lambda max_i sum_j |Sigma_ij| + 1) exceeds the largest
    objective change of a single flip, so every 1-flip local minimum, and hence the
    optimum, holds exactly k assets.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)
    mu_n = mu / max(np.abs(mu).max(), 1e-12)
    if penalty is None:
        penalty = 2.0 * (risk_aversion * np.abs(sigma).sum(1).max() + 1.0)
    P = np.ones((n, n)) - 2 * k * np.eye(n)          # (sum x - k)^2 - k^2
    return risk_aversion * sigma - np.diag(mu_n) + penalty * P
