import numpy as np
import pytest
from scipy.linalg import expm

from tciqaoa import (QAOA, all_energies, brute_force, cut_value, local_search, maxcut_qubo,
                     optimize_interp, optimize_restarts, qubo_energy, random_graph,
                     random_qubo, variable_order)

try:
    import xfacpy  # noqa: F401
    HAVE_XFAC = True
except ImportError:
    HAVE_XFAC = False
needs_xfac = pytest.mark.skipif(not HAVE_XFAC, reason="xfacpy not on PYTHONPATH")


def test_all_energies_matches_direct():
    Q = random_qubo(9, seed=3)
    E = all_energies(Q)
    for z in [0, 1, 77, 511]:
        x = np.array([(z >> i) & 1 for i in range(9)])
        assert np.isclose(E[z], qubo_energy(x, Q))


def test_maxcut_qubo_is_minus_cut():
    A = random_graph(8, 0.5, seed=1)
    Q = maxcut_qubo(A)
    x = np.random.default_rng(0).integers(0, 2, 8)
    assert np.isclose(qubo_energy(x, Q), -cut_value(x, A))


def test_qaoa_matches_dense_matrices():
    n = 4
    q = QAOA(random_qubo(n, seed=1), p=2)
    X = np.array([[0, 1], [1, 0]])
    B = sum(np.kron(np.kron(np.eye(2 ** (n - 1 - i)), X), np.eye(2 ** i)) for i in range(n))
    g, b = [0.3, 0.7], [0.4, 0.2]
    psi = np.ones(2 ** n) / 2 ** (n / 2)
    for gg, bb in zip(g, b):
        psi = expm(-1j * bb * B) @ (np.exp(-1j * gg * q.C) * psi)
    assert np.allclose(psi, q.state(g, b), atol=1e-12)


def test_qaoa_zero_angles_is_uniform():
    q = QAOA(random_qubo(6, seed=2), p=1)
    assert np.allclose(q.probs([0.0, 0.0]), 1 / 64)


def test_angle_optimisers_improve_on_uniform():
    Q = maxcut_qubo(random_graph(8, 0.5, seed=2))
    E = all_energies(Q)
    r_uniform = QAOA(Q, 1, energies=E).report([0.0, 0.0])["approx_ratio"]
    q = QAOA(Q, 1, energies=E)
    r1 = q.report(optimize_restarts(q, n_starts=3)["angles"])["approx_ratio"]
    r2 = optimize_interp(lambda k: QAOA(Q, k, energies=E), 2)
    r2 = QAOA(Q, 2, energies=E).report(r2["angles"])["approx_ratio"]
    assert r1 > r_uniform + 0.05
    assert r2 >= r1 - 1e-6


def test_local_search_reaches_local_minimum():
    Q = random_qubo(12, seed=4)
    x, e, _ = local_search(np.zeros(12), Q)
    assert np.isclose(e, qubo_energy(x, Q))
    for i in range(12):
        y = x.copy(); y[i] = 1 - y[i]
        assert qubo_energy(y, Q) >= e - 1e-9


def test_variable_order_is_permutation():
    Q = random_qubo(10, seed=5)
    for o in ["identity", "spectral", "rcm"]:
        assert sorted(variable_order(Q, o)) == list(range(10))


@needs_xfac
def test_tci_angles_p1_converges():
    from tciqaoa import optimize_tci
    Q = maxcut_qubo(random_graph(8, 0.5, seed=2))
    E = all_energies(Q)
    q = QAOA(Q, 1, energies=E)
    r = optimize_tci(q)
    ref = QAOA(Q, 1, energies=E)
    r_ref = optimize_restarts(ref, n_starts=10)
    assert r["cost"] <= r_ref["cost"] + 1e-4


@needs_xfac
def test_tci_bits_finds_exact_optimum_small():
    from tciqaoa import tci_minimize_bits
    Q = random_qubo(14, seed=7)
    _, e_opt, _ = brute_force(Q)
    r = tci_minimize_bits(Q, seed=0)
    assert np.isclose(r["energy"], e_opt)
    assert np.isclose(qubo_energy(r["x"], Q), r["energy"])
