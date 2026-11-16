import numpy as np
import pytest
from scipy.linalg import expm

from tciqaoa import (QAOA, all_energies, brute_force, cut_value, local_search, maxcut_qubo,
                     optimize_interp, optimize_restarts, qubo_energy, random_graph,
                     random_qubo, variable_order, beta_images, optimize_from_images)

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


def test_beta_images_are_maxcut_symmetries():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(0, 2, 2), rng.uniform(0, np.pi, 2)])
    imgs = beta_images(x)
    assert len(imgs) == 4 and np.allclose(imgs[0], x)
    qm = QAOA(maxcut_qubo(random_graph(8, 0.5, seed=1)), p=2)
    assert np.allclose([qm.cost(y) for y in imgs], qm.cost(x))
    qq = QAOA(random_qubo(8, seed=1), p=2)          # linear terms: not a symmetry
    assert np.ptp([qq.cost(y) for y in imgs]) > 1e-3
    r = optimize_from_images(qq, x)
    assert r["cost"] <= min(qq.cost(y) for y in imgs) + 1e-12
    assert r["n_evals"] > 4


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


@needs_xfac
def test_tt_reuse_transfers_to_new_instance():
    from tciqaoa import landscape_minima, optimize_from, optimize_tci
    Qt = maxcut_qubo(random_graph(8, 0.5, seed=100))
    tr = optimize_tci(QAOA(Qt, 1), return_tt=True)
    starts = landscape_minima(tr, top_k=3)
    assert len(starts) == 3
    Q = maxcut_qubo(random_graph(8, 0.5, seed=5))
    E = all_energies(Q)
    reuse = optimize_from(QAOA(Q, 1, energies=E), starts)
    scratch = optimize_restarts(QAOA(Q, 1, energies=E), n_starts=10)
    assert reuse["cost"] <= scratch["cost"] + 1e-4
    assert reuse["n_evals"] < scratch["n_evals"]


@needs_xfac
def test_sweep_tt_finds_optimum_across_family():
    from tciqaoa.sweep import SweepFamily, solve_from_sweep, tci_sweep
    A = random_graph(8, 0.5, seed=1)
    L = np.diag(A.sum(1)) - A
    fam = SweepFamily(lambda t: -L + t * 3 * np.eye(8), R_t=4)
    tt = tci_sweep(fam, 1)
    for t in [0.0, 0.5, 0.875]:
        r = solve_from_sweep(tt, fam, t)
        ref = optimize_restarts(fam.qaoa(t, 1), n_starts=10)
        assert r["cost"] <= ref["cost"] + 1e-3


def test_lift_matches_interp_rule():
    from tciqaoa.sweep import lift
    assert np.allclose(lift([1.0, 2.0], 1, 2), [1.0, 1.0, 2.0, 2.0])
    assert np.allclose(lift([1.0, 3.0, 2.0, 4.0], 2, 3), [1.0, 2.0, 3.0, 2.0, 3.0, 4.0])


@needs_xfac
def test_lifted_p2_from_p1_sweep():
    from tciqaoa.sweep import SweepFamily, solve_lifted, tci_sweep
    A = random_graph(8, 0.5, seed=1)
    L = np.diag(A.sum(1)) - A
    fam = SweepFamily(lambda t: -L + t * 3 * np.eye(8), R_t=4)
    tt = tci_sweep(fam, 1)
    r = solve_lifted(tt, fam, 0.5, 2)
    p1 = optimize_restarts(fam.qaoa(0.5, 1), n_starts=5)
    assert r["cost"] < p1["cost"]          # depth 2 beats the best depth-1 value


@needs_xfac
def test_sweep_library_no_worse_than_lifted():
    from tciqaoa.sweep import SweepFamily, solve_lifted, sweep_library, tci_sweep
    A = random_graph(8, 0.5, seed=1)
    L = np.diag(A.sum(1)) - A
    fam = SweepFamily(lambda t: -L + t * 3 * np.eye(8), R_t=4)
    tt = tci_sweep(fam, 1)
    ts = [0.0, 0.5, 0.875]
    lb = sweep_library(tt, fam, ts, 2, n_restarts=3)
    assert len(lb["costs"]) == 3 and lb["n_evals"] > 0
    for t, c in zip(ts, lb["costs"]):
        assert c <= solve_lifted(tt, fam, t, 2)["cost"] + 1e-6


@needs_xfac
def test_sweep_library_lower_depth_lift():
    from tciqaoa.sweep import SweepFamily, lift, sweep_library, tci_sweep
    A = random_graph(8, 0.5, seed=1)
    L = np.diag(A.sum(1)) - A
    fam = SweepFamily(lambda t: -L + t * 3 * np.eye(8), R_t=4)
    tt = tci_sweep(fam, 1)
    ts = [0.0, 0.5]
    p2 = sweep_library(tt, fam, ts, 2, seeding="restarts", n_restarts=2)
    assert p2["p"] == 2 and len(p2["library"]) >= 1
    p3 = sweep_library(None, fam, ts, 3, seeding=None, lower=p2)
    assert set(p3["source"]) <= {"lift", "sweep", "prev", "back"}
    for i, t in enumerate(ts):    # never worse than polishing the lifted p = 2 optimum alone
        q = fam.qaoa(t, 3)
        assert p3["costs"][i] <= q.cost(lift(p2["angles"][i], 2, 3)) + 1e-9
    p3s = sweep_library(None, fam, ts, 3, seeding="restarts", n_restarts=2, n_seed_lib=1,
                        lower=p2)
    assert np.all(p3s["costs"] <= p3["costs"] + 0.05)


@needs_xfac
def test_tci_sweep_is_deterministic():
    from tciqaoa.sweep import SweepFamily, tci_sweep
    A = random_graph(8, 0.5, seed=1)
    L = np.diag(A.sum(1)) - A
    builds = [tci_sweep(SweepFamily(lambda t: -L + t * 3 * np.eye(8), R_t=4), 1) for _ in range(3)]
    assert len({(b["n_evals"], b["tt_max_rank"]) for b in builds}) == 1


@needs_xfac
def test_sweep_chain_depths():
    from tciqaoa.sweep import SweepFamily, sweep_chain, tci_sweep
    A = random_graph(8, 0.5, seed=1)
    L = np.diag(A.sum(1)) - A
    fam = SweepFamily(lambda t: -L + t * 3 * np.eye(8), R_t=4)
    tt = tci_sweep(fam, 1)
    ts = [0.0, 0.5]
    r1 = sweep_chain(fam, ts, 1, tt=tt)
    r3 = sweep_chain(fam, ts, 3, tt=tt, extra_restarts=2, interp_start=False)
    assert sorted(r3["by_depth"]) == [2, 3] and r1["by_depth"] == {}
    assert r3["n_evals"] == tt["n_evals"] + sum(r["n_evals"] for r in r3["by_depth"].values())
    assert np.all(r3["costs"] <= r1["costs"] + 1e-6)      # deeper is never worse here
    r3i = sweep_chain(fam, ts, 3, tt=tt, extra_restarts=0)  # INTERP start at the last depth
    assert r3i["n_evals"] > tt["n_evals"] + sum(r["n_evals"] for r in r3i["by_depth"].values())
    assert set(r3i["by_depth"][3]["source"]) <= {"lift", "interp", "sweep", "prev", "back"}
    r3n = sweep_chain(fam, ts, 3, extra_restarts=0, interp_start=False, use_tci=False)
    assert r3n["n_evals"] == sum(r["n_evals"] for r in r3n["by_depth"].values())  # no TT
    assert "sweep" not in r3n["by_depth"][2]["source"]
    assert np.all(r3n["costs"] <= r1["costs"] + 1e-6)
    r3h = sweep_chain(fam, ts, 3, extra_restarts=0, interp_start=False, use_tci="hybrid")
    assert r3h["by_depth"][2]["n_seed_evals"] > r3n["by_depth"][2]["n_seed_evals"]
    assert np.all(r3h["costs"] <= r1["costs"] + 1e-6)


@needs_xfac
def test_sweep_library_seedings_run_and_attribute():
    from tciqaoa.sweep import SweepFamily, sweep_library, tci_sweep
    A = random_graph(8, 0.5, seed=1)
    L = np.diag(A.sum(1)) - A
    fam = SweepFamily(lambda t: -L + t * 3 * np.eye(8), R_t=4)
    tt = tci_sweep(fam, 1)
    ts = [0.0, 0.5]
    for p1, seeding in [(None, None), (None, "tci"), (tt, "restarts")]:
        lb = sweep_library(p1, fam, ts, 2, seeding=seeding, n_restarts=2)
        assert len(lb["costs"]) == 2 and np.all(np.isfinite(lb["costs"]))
        assert set(lb["source"]) <= {"tt", "seed", "sweep", "prev", "back"}
        if p1 is None:
            assert "tt" not in lb["source"]


def test_portfolio_qubo_linear_return_and_budget():
    from tciqaoa import portfolio_data, portfolio_qubo
    mu, S = portfolio_data(12, seed=0)
    k, lam = 3, 1.5
    Q = portfolio_qubo(mu, S, k, lam)
    x = np.zeros(12); x[[0, 4, 7]] = 1
    mu_n = mu / mu.max()
    # budget: the exact optimum holds exactly k assets
    xb, _, _ = brute_force(Q)
    assert xb.sum() == k
    # energy = Markowitz cost - A k^2 on feasible portfolios
    A = 2.0 * (lam * np.abs(S).sum(1).max() + 1.0)
    assert np.isclose(qubo_energy(x, Q), lam * x @ S @ x - mu_n @ x - A * k ** 2)


@needs_xfac
def test_tci_sweep_bits_finds_frontier_small():
    from tciqaoa import portfolio_data, portfolio_qubo, tci_sweep_bits
    mu, S = portfolio_data(12, seed=1)
    Qs = [portfolio_qubo(mu, S, 3, lam) for lam in (0.1, 0.5, 2.0, 5.0)]
    r = tci_sweep_bits(Qs, seed=0)
    exact = np.array([brute_force(Q)[1] for Q in Qs])
    assert np.allclose(r["energy"], exact)
    for x, Q, e in zip(r["x"], Qs, r["energy"]):
        assert np.isclose(qubo_energy(x, Q), e)


def test_shot_noise():
    from tciqaoa.qaoa import QAOA
    from tciqaoa.problems import random_qubo
    q = QAOA(random_qubo(8, seed=1), 1, shots=1024, rng=0)
    x = [0.4, 0.8]
    v = [q.cost(x) for _ in range(200)]
    assert q.n_evals == 200
    assert abs(np.mean(v) - q.exact_cost(x)) < 0.01
    assert 0.01 < np.std(v) < 0.06
    assert QAOA(random_qubo(8, seed=1), 1).cost(x) == q.exact_cost(x)


def test_spsa():
    from tciqaoa.angles import spsa, set_local_optimizer, optimize_restarts
    from tciqaoa.qaoa import QAOA
    from tciqaoa.problems import random_qubo
    f = lambda x: float(np.sum((np.asarray(x) - 0.3) ** 2))  # noqa: E731
    x, fx = spsa(f, [1.0, -0.5, 0.8], n_iter=200, step=0.1, rng=0)
    assert fx < 1e-3
    q = QAOA(random_qubo(8, seed=1), 1, shots=1024, rng=0)
    set_local_optimizer("spsa", spsa_iters=50)
    try:
        r = optimize_restarts(q, n_starts=3, seed=0)
    finally:
        set_local_optimizer("cobyla")
    assert r["n_evals"] == 3 * (2 * 2 + 2 * 50 + 1)
    exact = optimize_restarts(QAOA(random_qubo(8, seed=1), 1), n_starts=5, seed=0)
    assert q.exact_cost(r["angles"]) < exact["cost"] + 0.05


def test_sweep_chain_anchors():
    from tciqaoa.sweep import SweepFamily, sweep_chain
    A = random_graph(8, 0.5, seed=1)
    L = np.diag(A.sum(1)) - A
    fam = SweepFamily(lambda t: -L + t * 3 * np.eye(8), R_t=4)
    ts = list(np.arange(8) / 8)
    full = sweep_chain(fam, ts, 2, extra_restarts=0, use_tci=False)
    few = sweep_chain(fam, ts, 2, extra_restarts=0, use_tci=False, n_anchors=2)
    assert few["by_depth"][2]["n_seed_evals"] < full["by_depth"][2]["n_seed_evals"]
    assert np.all(np.isfinite(few["costs"])) and len(few["costs"]) == 8


@needs_xfac
def test_sweep_family_2d():
    from tciqaoa.sweep import SweepFamily2D, sweep_chain, tci_sweep, sweep_minima
    A = random_graph(6, 0.5, seed=1)
    L = np.diag(A.sum(1)) - A
    fam = SweepFamily2D(lambda t1, t2: -L + t1 * np.eye(6) + t2 * 2 * np.diag(np.arange(6) % 2), R=3)
    for t1, t2 in [(0.0, 0.0), (0.375, 0.75), (0.875, 0.125)]:
        assert fam.decode(fam.encode((t1, t2))) == (t1, t2)
    pts = fam.points(4)
    assert len(pts) == 16 and len(set(pts)) == 16
    d = [np.abs(np.subtract(fam.decode(a), fam.decode(b))).sum() for a, b in zip(pts, pts[1:])]
    assert np.allclose(d, 0.25)                       # snake order: grid neighbours
    tt = tci_sweep(fam, 1)
    assert tt["n_params"] == 2 and len(sweep_minima(tt, pts[5], 2)) >= 1
    r1 = sweep_chain(fam, pts, 1, tt=tt)
    r2 = sweep_chain(fam, pts, 2, tt=tt, extra_restarts=0)
    assert np.all(r2["costs"] <= r1["costs"] + 1e-6)


@needs_xfac
def test_sweep_family_3d():
    from tciqaoa.sweep import SweepFamilyND, sweep_chain, tci_sweep
    A = random_graph(6, 0.5, seed=1)
    L = np.diag(A.sum(1)) - A
    h = np.diag(np.arange(6) % 3 - 1.0)
    fam = SweepFamilyND(lambda t1, t2, t3: -L + t1 * np.eye(6) + t2 * 2 * np.diag(np.arange(6) % 2)
                        + t3 * h, 3, R=2)
    assert fam.decode(fam.encode((0.25, 0.75, 0.5))) == (0.25, 0.75, 0.5)
    pts = fam.points(2)
    assert len(set(pts)) == 8
    d = [np.abs(np.subtract(fam.decode(a), fam.decode(b))).sum() for a, b in zip(pts, pts[1:])]
    assert np.allclose(d, 0.5)                        # snake order: one step at a time
    tt = tci_sweep(fam, 1)
    assert tt["n_params"] == 3
    r = sweep_chain(fam, pts, 1, tt=tt)
    assert np.all(np.isfinite(r["costs"])) and len(r["costs"]) == 8


@needs_xfac
def test_solve_entry_points():
    from tciqaoa import solve_qubo, solve_family, solve_bits
    from tciqaoa.solve import main
    Q = random_qubo(6, seed=4)
    E0 = brute_force(Q)[1]
    for tci in ("on", "off"):
        r = solve_qubo(Q, p=1, tci=tci)
        assert r["tci"] == (tci == "on") and abs(r["report"]["best_sampled_energy"] - E0) < 1e-9
        b = solve_bits(Q, tci=tci)
        assert abs(b["energy"] - E0) < 1e-9
    r = solve_qubo(Q, p=2, tci=False, extra_restarts=0)
    assert len(r["angles"]) == 4 and r["n_evals"] > 0
    A = random_graph(6, 0.5, seed=1)
    L = np.diag(A.sum(1)) - A
    make_Q = lambda t: -L + 2 * t * np.eye(6)
    on = solve_family(make_Q, p=1, tci=True, nt=4)
    off = solve_family(make_Q, p=1, tci=False, nt=4)
    assert on["params"] == [(0.0,), (0.25,), (0.5,), (0.75,)] and on["tt"] is not None
    assert np.allclose(on["costs"], off["costs"], atol=1e-3)
    r2 = solve_family(lambda a, b: -L + a * np.eye(6) + b * np.diag(np.arange(6) % 2), p=2,
                      tci=False, n_params=2, nt=2, R=2)
    assert len(r2["costs"]) == 4
    assert main(["--family", __file__ + ":_family_for_cli", "--nt", "2", "--tci", "off"]) == 0


def _family_for_cli(t):
    return -np.diag(np.arange(4) % 2) + t * np.eye(4) + 0.3 * (np.ones((4, 4)) - np.eye(4))
