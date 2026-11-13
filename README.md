# tciqaoa

**Tensor Cross Interpolation (TCI, via [`xfacpy`](https://github.com/tensor4all/xfac)) applied to QAOA,
benchmarked against [hybridq](https://github.com/Faluke-Aikebaier/QAOA-inspired-optimisation)
and classical QUBO solvers.**

> **Status:** research code with honest benchmarks. Short version: TCI learns the
> p = 1 QAOA landscape cheaply and reliably, but at p ≥ 2 the landscape stops being
> low rank and INTERP is 6–20× cheaper. On the bitstring landscape, TCI beats hybridq
> by a wide margin, but random-restart local search given the same time beats TCI.
> All numbers below are reproducible with the scripts in `benchmarks/`.

Companion repos: [tci-optimizer](https://github.com/Faluke-Aikebaier/tci-optimizer)
(TCI for continuous global optimisation) and
[QAOA-inspired-optimisation](https://github.com/Faluke-Aikebaier/QAOA-inspired-optimisation)
(`hybridq`: simulated bifurcation and simulated quantum annealing).

---

## What is in here

| module | what it does |
|---|---|
| `tciqaoa.QAOA` | **Real** gate-model QAOA, `prod_k exp(-iβ_k ΣX) exp(-iγ_k C) \|+⟩ⁿ`, simulated exactly as a 2ⁿ statevector (checked against dense matrix exponentials to 1e-16). Up to n ≈ 22. |
| `optimize_restarts` | Baseline angle search: random starts + COBYLA. |
| `optimize_interp` | Baseline angle search: INTERP warm start (Zhou et al., PRX 10, 021067 (2020)). |
| `optimize_tci` | **TCI route 1**: learn the landscape ⟨C⟩(γ, β) as a quantics tensor train (`xfacpy.QTensorCI`), then COBYLA-polish the best TCI-sampled points. Returns the TT rank and pivot error as built-in diagnostics. |
| `tci_minimize_bits` | **TCI route 2**: TCI (`xfacpy.TensorCI2`) on the n-leg Boltzmann tensor `F(x) = exp(-β(E(x) − E_ref))`, x ∈ {0,1}ⁿ (one leg per qubit, as in the QAOA register), with staged β (deeper layers / slower anneal) and a 1-flip polish. Never builds the 2ⁿ table, so it runs at n = 80+. Variables are ordered along the chain by the Fiedler vector of \|Q\|, since flattening order matters for tensor trains. |

### How the two TCI routes map onto QAOA

```
QAOA                          TCI route 1 (angles)            TCI route 2 (bitstrings)
-----                         --------------------            ------------------------
2p variational angles   --->  tensor train over a 2^R grid
                              per angle; pivots = circuits
                              to run; surrogate = whole map
n-qubit register        ------------------------------->     n-leg tensor, local dim 2
|ψ(γ,β)|² peaks at low E ----------------------------->      F(x)=exp(-βE(x)) peaks at low E
deeper p                ------------------------------->     larger β (staged)
measurement shots       ------------------------------->     TCI pivots (rook search)
```

---

## Installation

```bash
git clone https://github.com/Faluke-Aikebaier/tci-qaoa.git && cd tci-qaoa
pip install -e .            # numpy, scipy
# xfacpy is not on PyPI; build it:
git clone https://github.com/tensor4all/xfac.git
cmake -S xfac -B xfac/build -D CMAKE_BUILD_TYPE=Release -D XFAC_BUILD_PYTHON=ON
cmake --build xfac/build -j 4
export PYTHONPATH="$PWD/xfac/build/python:$PYTHONPATH"
```

The QAOA simulator and the two baseline angle optimisers work without xfacpy.

## Quick start

```python
from tciqaoa import QAOA, maxcut_qubo, random_graph, optimize_tci, tci_minimize_bits

Q = maxcut_qubo(random_graph(12, 0.4, seed=3))

q = QAOA(Q, p=1)
r = optimize_tci(q)                          # TCI over (gamma, beta)
print(r["tt_max_rank"], r["pivot_error"], r["n_evals"])
print(q.report(r["angles"]))                 # approx ratio, P(ground), best of 1000 shots

r = tci_minimize_bits(Q)                     # TCI directly on exp(-beta E(x))
print(r["energy"], r["n_energy_evals"], r["max_rank"])
```

---

## Results

Hardware: Apple laptop (10 cores), single process. Instances come from hybridq's own
generators (`random_graph(n, 0.4, seed)` → MaxCut, `random_qubo(n, 0.5, seed)`),
so every method sees bit-for-bit identical problems.

### 1. QAOA angles: restarts vs INTERP vs TCI

`python benchmarks/bench_qaoa.py --n 10 14 --p 1 2 3 --seeds 5` (5 instances per row; means).

| problem | n | p | method | approx. ratio | P(ground) | 1000 shots hit opt. | circuit evals | time (s) | TT rank | pivot err |
|---|---|---|---|---|---|---|---|---|---|---|
| maxcut | 10 | 1 | interp | 0.8034 | 0.131 | 5/5 | 395 | 0.0 |  |  |
| maxcut | 10 | 1 | restarts | 0.8034 | 0.131 | 5/5 | 766 | 0.1 |  |  |
| maxcut | 10 | 1 | tci | 0.8034 | 0.131 | 5/5 | 558 | 0.1 | 7 | 3e-03 |
| maxcut | 10 | 2 | interp | 0.8766 | 0.296 | 5/5 | 564 | 0.1 |  |  |
| maxcut | 10 | 2 | restarts | 0.8766 | 0.296 | 5/5 | 2086 | 0.4 |  |  |
| maxcut | 10 | 2 | tci | 0.8766 | 0.296 | 5/5 | 5650 | 1.1 | 20 | 2e-01 |
| maxcut | 10 | 3 | interp | 0.9172 | 0.459 | 5/5 | 789 | 0.1 |  |  |
| maxcut | 10 | 3 | restarts | 0.9172 | 0.459 | 5/5 | 2932 | 0.9 |  |  |
| maxcut | 10 | 3 | tci | 0.9110 | 0.426 | 5/5 | 15098 | 4.6 | 20 | 1e+00 |
| maxcut | 14 | 1 | interp | 0.7909 | 0.016 | 5/5 | 528 | 0.7 |  |  |
| maxcut | 14 | 1 | restarts | 0.7909 | 0.016 | 5/5 | 1062 | 1.3 |  |  |
| maxcut | 14 | 1 | tci | 0.7909 | 0.016 | 5/5 | 690 | 0.8 | 7 | 1e-03 |
| maxcut | 14 | 2 | interp | 0.8568 | 0.067 | 5/5 | 711 | 1.1 |  |  |
| maxcut | 14 | 2 | restarts | 0.8568 | 0.067 | 5/5 | 2201 | 5.3 |  |  |
| maxcut | 14 | 2 | tci | 0.8568 | 0.067 | 5/5 | 4642 | 10.8 | 20 | 2e-01 |
| maxcut | 14 | 3 | interp | 0.8937 | 0.141 | 5/5 | 975 | 1.9 |  |  |
| maxcut | 14 | 3 | restarts | 0.8937 | 0.141 | 5/5 | 2950 | 10.0 |  |  |
| maxcut | 14 | 3 | tci | 0.8894 | 0.133 | 5/5 | 12541 | 41.6 | 20 | 1e+00 |
| qubo | 10 | 1 | interp | 0.7340 | 0.028 | 5/5 | 302 | 0.0 |  |  |
| qubo | 10 | 1 | restarts | 0.7340 | 0.028 | 5/5 | 661 | 0.1 |  |  |
| qubo | 10 | 1 | tci | 0.7340 | 0.028 | 5/5 | 433 | 0.0 | 6 | 6e-03 |
| qubo | 10 | 2 | interp | 0.8608 | 0.089 | 5/5 | 387 | 0.0 |  |  |
| qubo | 10 | 2 | restarts | 0.8608 | 0.089 | 5/5 | 1998 | 0.4 |  |  |
| qubo | 10 | 2 | tci | 0.8608 | 0.089 | 5/5 | 5734 | 1.0 | 20 | 6e-01 |
| qubo | 10 | 3 | interp | 0.9102 | 0.172 | 5/5 | 623 | 0.1 |  |  |
| qubo | 10 | 3 | restarts | 0.9098 | 0.170 | 5/5 | 2909 | 0.8 |  |  |
| qubo | 10 | 3 | tci | 0.9102 | 0.171 | 5/5 | 12394 | 3.2 | 20 | 1e+00 |
| qubo | 14 | 1 | interp | 0.7433 | 0.009 | 5/5 | 479 | 0.5 |  |  |
| qubo | 14 | 1 | restarts | 0.7433 | 0.009 | 5/5 | 907 | 1.0 |  |  |
| qubo | 14 | 1 | tci | 0.7433 | 0.009 | 5/5 | 512 | 0.6 | 6 | 1e-02 |
| qubo | 14 | 2 | interp | 0.8405 | 0.031 | 5/5 | 667 | 1.0 |  |  |
| qubo | 14 | 2 | restarts | 0.8489 | 0.035 | 5/5 | 2409 | 5.4 |  |  |
| qubo | 14 | 2 | tci | 0.8405 | 0.031 | 5/5 | 7156 | 15.3 | 20 | 4e-01 |
| qubo | 14 | 3 | interp | 0.8884 | 0.068 | 5/5 | 955 | 1.9 |  |  |
| qubo | 14 | 3 | restarts | 0.8900 | 0.070 | 5/5 | 2985 | 9.6 |  |  |
| qubo | 14 | 3 | tci | 0.8881 | 0.067 | 5/5 | 16746 | 53.0 | 20 | 1e+00 |

Takeaways

- **p = 1: TCI works.** The 2-angle landscape is a converged, low-rank quantics tensor
  train, and TCI finds the same optimum as 10 COBYLA restarts with ~30 % fewer circuit
  evaluations (INTERP, which at p = 1 is just 5 restarts, is cheaper still). As a bonus you get the whole landscape as a TT you can store and reuse.
- **p ≥ 2: TCI loses.** The rank hits the bond-dimension cap and the pivot error stops
  falling, so the surrogate never converges. The COBYLA polish still recovers the
  optimum in most cases, but at 6–20× the cost of INTERP.
- **The angles concentrate.** Optimal angles are nearly identical across instances of
  the same family (for example p = 2 MaxCut seeds 3 and 4: γ ≈ (0.89, 1.74) vs
  (0.92, 1.80)). This is the regime where a TCI landscape learned once and reused
  could pay off; see *Ideas* below.

### 2. Solving the QUBO: TCI-bits vs hybridq and classical baselines

`HYBRIDQ_PATH=... python benchmarks/bench_solvers.py --seeds 10` (10 instances per row).

| problem | n | ref | hybridq-SB | hybridq-QA | SA | Tabu | LS-multistart | TCI-bits | LS-time-matched |
|---|---|---|---|---|---|---|---|---|---|
| maxcut | 16 | exact | 0/10 (32.7%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) |
| maxcut | 20 | exact | 0/10 (27.6%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) |
| maxcut | 30 | best known | 0/10 (26.9%) | 10/10 (0.0%) | 10/10 (0.0%) | 9/10 (0.1%) | 9/10 (0.1%) | 10/10 (0.0%) | 10/10 (0.0%) |
| maxcut | 50 | best known | 0/10 (48.7%) | 0/10 (22.9%) | 7/10 (0.1%) | 5/10 (0.5%) | 6/10 (0.1%) | 7/10 (0.1%) | 10/10 (0.0%) |
| maxcut | 80 | best known | 0/10 (64.9%) | 0/10 (28.1%) | 3/10 (0.2%) | 1/10 (0.7%) | 0/10 (0.4%) | 1/10 (0.7%) | 10/10 (0.0%) |
| qubo | 16 | exact | 7/10 (1.1%) | 10/10 (0.0%) | 9/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) |
| qubo | 20 | exact | 9/10 (0.6%) | 10/10 (-0.0%) | 10/10 (-0.0%) | 10/10 (-0.0%) | 10/10 (-0.0%) | 10/10 (-0.0%) | 10/10 (-0.0%) |
| qubo | 30 | best known | 3/10 (3.2%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) |
| qubo | 50 | best known | 0/10 (17.1%) | 0/10 (13.9%) | 9/10 (0.0%) | 8/10 (0.1%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) |
| qubo | 80 | best known | 0/10 (23.3%) | 0/10 (21.8%) | 9/10 (0.1%) | 7/10 (0.2%) | 10/10 (0.0%) | 8/10 (0.1%) | 10/10 (0.0%) |

Cell = instances where the method reached the reference (mean relative energy gap).

| problem | n | hybridq-SB time (s) | hybridq-QA time (s) | SA time (s) | Tabu time (s) | LS-multistart time (s) | TCI-bits time (s) | LS-time-matched time (s) |
|---|---|---|---|---|---|---|---|---|
| maxcut | 16 | 0.09 | 0.05 | 0.06 | 0.02 | 0.00 | 0.12 | 0.12 |
| maxcut | 20 | 0.09 | 0.06 | 0.06 | 0.02 | 0.00 | 0.26 | 0.26 |
| maxcut | 30 | 0.09 | 0.05 | 0.06 | 0.02 | 0.00 | 0.75 | 0.75 |
| maxcut | 50 | 0.22 | 0.12 | 0.06 | 0.02 | 0.01 | 2.60 | 2.60 |
| maxcut | 80 | 0.32 | 0.11 | 0.06 | 0.02 | 0.01 | 5.62 | 5.62 |
| qubo | 16 | 0.09 | 0.05 | 0.06 | 0.02 | 0.00 | 0.07 | 0.07 |
| qubo | 20 | 0.09 | 0.05 | 0.06 | 0.02 | 0.00 | 0.16 | 0.16 |
| qubo | 30 | 0.09 | 0.05 | 0.06 | 0.02 | 0.01 | 0.39 | 0.39 |
| qubo | 50 | 0.19 | 0.10 | 0.06 | 0.02 | 0.01 | 1.50 | 1.50 |
| qubo | 80 | 0.29 | 0.10 | 0.06 | 0.02 | 0.02 | 4.07 | 4.07 |

TCI-bits mean energy evaluations / max TT rank: maxcut n=16: 15259 / 20, maxcut n=20: 36309 / 20, maxcut n=30: 95064 / 20, maxcut n=50: 213662 / 20, maxcut n=80: 381820 / 20, qubo n=16: 8864 / 20, qubo n=20: 19592 / 20, qubo n=30: 46992 / 20, qubo n=50: 123096 / 20, qubo n=80: 262694 / 20

The controls: `LS-multistart` is 100 random-restart steepest descents. `LS-time-matched`
spends exactly TCI-bits' wall time on random-restart steepest descent, which isolates
what the TCI search adds on top of its own local-search polish.

Takeaways

- **TCI-bits ≫ hybridq.** TCI-bits reaches the optimum on every exact instance
  (n ≤ 20) and stays within 0.1–0.7 % of best known at n = 50–80. hybridq-SB misses
  most instances outright, and hybridq-QA fails at n ≥ 50 (13–28 % gap). Part of this
  is a bug in hybridq; see below.
- **But TCI-bits < time-matched local search.** Given the same seconds, random-restart
  local search matches or beats TCI-bits everywhere and is the only method that hits
  best-known on all 100 instances. On these dense random instances, TCI's pivot search
  does not add anything a restart does not. The TT rank saturates at the cap (20):
  the Boltzmann tensor of a dense random QUBO is not low rank, which is the same
  "entry cost" law seen in the earlier xfacpy experiments on LLM weights.
- **SA (from hybridq's own `examples.py`) is the best fixed-budget baseline**: 0.06 s
  and within 0.2 % everywhere.

### 3. A bug found in hybridq's simulated bifurcation

`SimulatedBifurcation._ising_from_qubo` computes the local field as

```python
h = -0.25 * (Q.sum(axis=1) + np.diag(Q))
```

but substituting x = (s + 1)/2 into xᵀQx gives a linear term (Q·1/2)·s, i.e.

```python
h = -0.5 * Q.sum(axis=1)
```

For MaxCut (Q = −L, Q·1 = 0) the correct field is **zero** (the problem is Z₂ symmetric),
but hybridq applies h = deg/4 > 0, which pushes every spin to +1, the trivial
"everything on one side" cut. `benchmarks/check_hybridq_sb_mapping.py` checks the
mapping: the QUBO energy minus the Ising energy must be the same constant for every
state.

```
maxcut n=20        spread(hybridq h) =   45.500   spread(fixed h) = 0.0e+00
random QUBO n=20   spread(hybridq h) =    9.377   spread(fixed h) = 6.2e-15
```

With the fix, hybridq-SB goes from 0/3 to 3/3 exact on n = 20 MaxCut, but it still
trails at n ≥ 50, which points to further issues: the momentum is not reset at the
|x| = 1 walls (ballistic SB), and the decomposition/stitching runs at n > 40. The
benchmark above uses hybridq **as published**.

---

## Ideas worth testing next

1. **Transfer the p = 1 landscape TT.** It converges and the angles concentrate, so
   learn it once per problem family and reuse it as a zero-cost warm start.
2. **TCI-bits on structured, sparse instances** (chains, lattices, low-treewidth graphs)
   where the Boltzmann tensor really is low rank. Dense random QUBOs are the worst
   case for any tensor train.
3. **TCI on the QAOA output distribution** |⟨x|ψ(γ,β)⟩|² with an MPS simulator, to read
   off high-probability bitstrings at n beyond statevector reach.

## Tests

```bash
pytest -q          # 9 tests; the 2 TCI tests are skipped if xfacpy is not importable
```

## References

- E. Farhi, J. Goldstone, S. Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028.
- L. Zhou, S.-T. Wang, S. Choi, H. Pichler, M. D. Lukin, *Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices*, PRX 10, 021067 (2020).
- Y. Núñez Fernández et al., *Learning low-rank tensor train representations: new algorithms and libraries* (2024); `xfac`: https://github.com/tensor4all/xfac.
- H. Goto, K. Tatsumura, A. R. Dixon, *Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems*, Sci. Adv. 5, eaav2372 (2019).

## License

MIT — see [LICENSE](LICENSE).
