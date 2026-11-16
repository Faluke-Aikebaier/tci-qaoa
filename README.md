# tciqaoa

**Tensor Cross Interpolation (TCI, via [`xfacpy`](https://github.com/tensor4all/xfac)) applied to QAOA,
benchmarked against [hybridq](https://github.com/Faluke-Aikebaier/QAOA-inspired-optimisation)
and classical QUBO solvers.**

> **Bottom line:** TCI does not beat QAOA; it is a deterministic, reusable first stage
> for it. See [Conclusions](#conclusions), and [Four problem interfaces](#four-problem-interfaces)
> for the entry points with TCI on or off.
>
> **Status:** research code with honest benchmarks. Short version: TCI learns the
> p = 1 QAOA landscape cheaply and reliably, but at p ≥ 2 the landscape stops being
> low rank and INTERP is 6–20× cheaper. On the bitstring landscape, TCI beats hybridq
> by a wide margin, but random-restart local search given the same time beats TCI.
> All numbers below are reproducible with the scripts in `benchmarks/`.
>
> **Reuse, measured.** TCI's selling point is that it returns a reusable tensor train,
> not a single point. Benchmark 4 tests this on 600 new instances. A p = 1 landscape
> learned on one instance transfers to new instances, other sizes and even other
> problem families, cutting the circuit runs per instance about 2× (1.6–2.5×) versus INTERP from scratch.
> But storing just the 2p optimal angles transfers almost as well and is 3–4× cheaper
> still, because QAOA angles concentrate so strongly. The one catch is p = 2 MaxCut →
> QUBO: the stored point sits in a MaxCut symmetry image (β vs β + π/2) that is the
> wrong basin for QUBO (0/50). Storing its 2^p images too (`optimize_from_images`)
> fixes that: 600/600, for 2–5 extra runs per instance. See
> [Why TCI: reusability](#why-tci-reusability) and [benchmark 4](#4-reuse-does-a-learned-landscape-pay-off-on-new-instances).
>
> **Where the tensor train does win: parameter sweeps (benchmark 5).** One TCI over
> (angles × sweep parameter) learns a whole problem family. At p = 1 it was the only
> reuse method that found the optimum at every one of 192 parameter values. Stored-point
> and continuation warm starts fell into the wrong basin on one instance, by up to 1.1
> in normalised cost. It costs 1.2–1.9× fewer circuit runs than INTERP at every
> point, but 3–8× more than the warm starts when those don't fail.
>
> **At p = 2, the landscape cannot be a sweep TT** (accurate TTs need rank ≈ 75 and
> evaluate most of the grid). What works is a shared library of p = 2 basins, seeded
> at every other parameter value and reused along the sweep: 192/192 optima, against
> 118/192 for 20 random restarts at every point at 4× the cost. **TCI's best role
> there is as the global sampler that seeds the library**: 184/192 (worst miss 0.011),
> deterministic, while random restarts at the same cost range 168–190 (181.8 ± 7.3).
> So TCI gives an average run every time, not a better one. See [benchmark 5](#5-sweeps-one-tensor-train-for-a-whole-problem-family)
> and [benchmark 6](#6-where-should-tci-go-tci-placement), which tests every placement
> of TCI in the pipeline.
>
> **At p = 3 the gap reopens.** No pipeline finds every optimum: the best get 146–155
> of 192 (worst miss 0.02–0.05), 20 restarts at every point get 48/192. TCI is level
> with random restarts as a seeder (146 vs 148.6 ± 10.4 for 8 restarts). What TCI
> does do at p = 3 is make the result stable: with the p = 1 sweep TT's lifted
> candidates, 4 restarts score 150.0 ± 3.1 across library seeds instead of
> 116.6 ± 48.8 (one seed drops to 22). **Chaining the depths helps more:** solving the
> family at p = 2 first and lifting its whole basin library to p = 3
> (`sweep_library(..., lower=p2)`) raises the score from 139.3 to 150.7 on average and
> up to 159, against a stricter best that it found itself. See [p = 3](#p--3).
>
> **At p = 4 the chain holds up** (`sweep_chain`, [benchmark 8](#p--4)): 169/192
> deterministic, 178.0 on average with extra restarts, against 104 for INTERP and 4 for
> 20 random restarts at every point. On the portfolio families it finds 95 of 96 optima,
> INTERP 32.
>
> **At larger n** ([benchmark 9](#larger-n), p = 3) the chain stays ahead: 171, 146 and
> 189/192 at n = 10, 12, 14 (180, 152 with extra restarts) against 99, 87 and 103 for
> INTERP. **Reproducibility:** xfacpy's rook pivoting is history-dependent, so every
> TCI build now uses full pivoting by default, and all benchmarks were rerun with it;
> see the note after benchmark 9 for what moved.
> **p = 4 at larger n:** at n = 12, 140/192 deterministic and 149 with extra restarts
> (INTERP 86, 20 restarts 6); at n = 14, 152/192 deterministic and 162, 152,
> 166 (160.0) with extra restarts over 3 library seeds, 174 for the best of the three
> (INTERP 102, 20 restarts 2). The misses are late-t *field* and mid/late-t *penalty*
> points, by ≤ 0.046 (one *mix* instance at n = 12: 0.159). These scores are against
> the best of every run at that n, including the no-TCI seed check below, so they
> are lower than the first reported ones.
>
> **Job scheduling** ([benchmark 9](#sched-p4), n = 12, p = 4): the chain with extra
> restarts gets 30/48 with TCI, and 30, 7, 36 over three seeds without (46 for the best
> of the three); INTERP 0, 20 restarts 0. Without the extra restarts the chain locks
> into a worse basin (3/48 with TCI, 10, 3, 3 without).
>
> **The advantage is mostly the chaining; TCI is worth a few points per run**
> ([ablation](#ablation-p4)). The same chain *without any TCI* (no depth-1 TT; depth 2
> seeded by random restarts), over three restart seeds at n = 10 (p = 3, 4), 12 and 14
> (p = 4), scores on average 1–14 points *below* the TCI chain at n = 10 and 14 and
> 7–13 above it at n = 12, where one lucky seed found a better *penalty* basin
> (175 vs 140–143 for the other seeds). It is 12–26% cheaper (level at n = 10, p = 4) and, because its
> seeds differ already at depth 2, three no-TCI seeds together beat three TCI seeds
> (188 vs 174 at n = 14), and swapping one of them for the TCI run lowers the score
> ([mixing](#mix-p4)); merging TCI and restarts into one run is no better than the
> TCI run alone ([hybrid](#hybrid-p4)). Given the chain's budget, INTERP with 8 seeds gets 91/192 and
> 40 random restarts get 26 (n = 12).
>
> **Under shot noise the chain is the most robust method** ([benchmark 9](#9-shot-noise)).
> With ⟨C⟩ estimated from 1024 shots per circuit, at p = 2 the TCI chain lands 0.022
> above the best exact cost on average, the no-TCI chain 0.056, 20 random restarts
> 0.087 and INTERP 0.145; at p = 3 0.036, 0.051, 0.20 and 0.13. With SPSA (a
> noise-aware optimiser) instead of COBYLA every method improves, but the chain's lead
> survives: 0.003–0.014 vs 0.037–0.14 for INTERP and restarts at p = 2, 3. With SPSA
> the no-TCI chain is as good as or better than the TCI chain, for fewer runs, so
> under noise too TCI's own part is small (with COBYLA it had a small edge at p = 2).
> The p = 1 sweep TT, the worst p = 1 method under noise with COBYLA, is level with
> INTERP with SPSA.
>
> **Denser sweeps** ([benchmark 10](#10-denser-sweeps), 8 → 64 points of t): the p = 1
> sweep TT's cost per point halves (563 → 291 runs) while INTERP's stays at ~530, for the
> same answers, so here TCI is 1.8× cheaper. The chain gets *more accurate* with density
> (p = 2: 87.5% → 99.2%), while INTERP and restarts stay flat (48%, 61%). Seeding the
> no-TCI chain at 8 fixed points keeps 98.7% at 64 points for about INTERP's cost per
> point.
>
> **Two-parameter sweeps** ([benchmark 11](#11-two-parameter-sweeps), one TT over angles
> × risk aversion × penalty weight, or blend × field; up to 16 × 16 points): the p = 1 TT
> stays low rank (≤ 29) and is 1.8× cheaper per point than INTERP at 256 points. At
> p = 2 the chain reaches 99.9% (INTERP 50.5%, restarts 62.8%), and here TCI seeding
> helps when the grid or the seeding is sparse (97.9 vs 91.9% at 8 × 8; 99.0 vs 95.4%
> seeded at 16 points).
>
> **Three-parameter sweeps** ([benchmark 12](#12-three-parameter-sweeps), up to
> 8 × 8 × 8 points): the p = 1 TT is 1.9× cheaper per point than INTERP at 512 points
> and finds every optimum once each point is also warm-started from its neighbour
> (without that, 8 of 3,456 points missed); it costs up to 23k runs where the family
> is hard to compress. At p = 2, seeded at 16 points only, the TCI
> chain keeps 98.9% against 88.2% without TCI; with full seeding they are level, and
> at p = 3 the no-TCI chain is ahead (97.0 vs 93.3%).
>
> **TCI directly on the objective** (MaxCut, random QUBO, portfolio, and a whole
> efficient frontier with one TT) beats hybridq but loses to random-restart local
> search given the same time: these dense objectives are not low rank. See
> [benchmark 2](#2-solving-the-qubo-tci-bits-vs-hybridq-and-classical-baselines) and
> [benchmark 7](#7-tci-on-the-objective-for-a-whole-efficient-frontier).

Companion repos: [tci-optimizer](https://github.com/Faluke-Aikebaier/tci-optimizer)
(TCI for continuous global optimisation) and
[QAOA-inspired-optimisation](https://github.com/Faluke-Aikebaier/QAOA-inspired-optimisation)
(`hybridq`: simulated bifurcation and simulated quantum annealing).

---

## Installation

```bash
git clone https://github.com/Faluke-Aikebaier/tci-qaoa.git && cd tci-qaoa
pip install -e .            # numpy, scipy
# xfacpy is not on PyPI; build it (needed for tci=True only):
git clone https://github.com/tensor4all/xfac.git
cmake -S xfac -B xfac/build -D CMAKE_BUILD_TYPE=Release -D XFAC_BUILD_PYTHON=ON
cmake --build xfac/build -j 4
export PYTHONPATH="$PWD/xfac/build/python:$PYTHONPATH"
```

Everything with `tci=False` (the QAOA simulator and the baseline optimisers) works
without xfacpy.

---

## Quick start

Every solver takes `tci=True` or `tci=False`. With `tci=False` it runs the same
pipeline with the TCI stage replaced by the baseline TCI was benchmarked against, so
the two can be compared directly.

```python
import numpy as np
from tciqaoa import (solve_qubo, solve_maxcut, solve_portfolio,
                     random_qubo, random_graph, portfolio_data)

# ── Raw QUBO: minimise x^T Q x, QAOA at depth 2 ──────────────────────
Q   = random_qubo(12, seed=3)
res = solve_qubo(Q, p=2, tci=False)          # tci=True: same pipeline, TCI first stage
print(res["report"]["best_sampled_x"])       # [1. 1. 0. 1. 1. 1. 0. 1. 1. 0. 1. 1.]
print(res["report"]["best_sampled_energy"])  # -12.4285  (the exact optimum)
print(res["n_evals"])                        # 1,550 circuit runs

# ── MaxCut on a random graph ─────────────────────────────────────────
A   = random_graph(12, edge_prob=0.4, seed=3)
res = solve_maxcut(A, p=2, tci=False)
print(f"Cut: {res['cut_value']:.0f} / {res['n_edges']} edges")   # Cut: 18 / 23 edges

# ── Portfolio: select 4 of 12 assets ─────────────────────────────────
mu, sigma = portfolio_data(12, seed=2)
res = solve_portfolio(mu, sigma, k=4, p=2, tci=False, risk_aversion=0.5)
print(res["selected_assets"], f"Sharpe {res['sharpe']:.3f}")     # [ 3  4  6 10] Sharpe 0.737

# ── Efficient frontier: 32 risk aversions, one tensor train (TCI pays here) ──
fr = solve_portfolio(mu, sigma, k=4, p=1, tci=True, risk_aversion=(0.0, 1.0), nt=32)
for pt in fr["frontier"][::8]:
    print(f"lambda {pt['risk_aversion']:.2f}: {pt['selected_assets']}  "
          f"return {pt['expected_return']:.3f}  variance {pt['variance']:.3f}")
print(fr["n_evals"])                         # 15,903 circuit runs; tci=False: 27,714
# lambda 0.00: [2 4 6 8]  return 0.691  variance 6.298
# lambda 0.26: [ 3  4  6 10]  return 0.602  variance 0.668
# lambda 0.52: [ 3  4  6 10]  return 0.602  variance 0.668
# lambda 0.77: [ 3  4  6 11]  return 0.556  variance 0.333
```

QAOA returns a quantum state, not a bitstring: every solver reads its answer off as
the best of `shots` (default 4,096) bitstrings sampled from the optimised circuit, and
reports P(ground) so you can see how likely that is to be the optimum. At low depth
P(ground) can be small (0.1% for the portfolio above at p = 2), so raise `shots` or `p`
if the answer must be the exact optimum.

---

## Four problem interfaces

| | problem | TCI on: what TCI does | TCI off: the baseline | when TCI pays |
|---|---|---|---|---|
| `solve_qubo` | any QUBO, one instance | p = 1: a tensor train over the angle box (`optimize_tci`). p ≥ 2: the depth chain whose first stage is a TT and a TCI sampler | p = 1: INTERP (5 restarts). p ≥ 2: the same chain seeded by random restarts | rarely: on one instance `tci=False` is as good and cheaper |
| `solve_maxcut` | MaxCut on a graph | as `solve_qubo` | as `solve_qubo` | as `solve_qubo` |
| `solve_portfolio` | pick k of n assets; or a whole efficient frontier | one portfolio: as `solve_qubo`. Frontier: one TT over (angles × risk aversion) for every λ | INTERP at each λ (p = 1), or the chain seeded by random restarts (p ≥ 2) | for frontiers of ~32+ points: 1.7–1.8× fewer circuit runs at p = 1 |
| `solve_family` | any family of QUBOs Q(t₁, .., t_k) at many parameter points | one TT over (angles × t₁ .. t_k), and the depth chain built on it | as for the frontier | many points (~64+), or p ≥ 2 with few seeding points (`n_anchors`) |

QAOA is simulated exactly as a 2ⁿ statevector, so all four need n ≲ 22. Every result
carries `n_evals`, the circuit runs used, so TCI on and off can be compared on cost.
See [Conclusions](#conclusions) for the evidence behind the last column.

The tensor train is built with `angle_pivots=4`: the build also starts from 4 × 4
points spread over the angle box. Without them it can converge early around one basin;
on a portfolio family with the default budget penalty it stopped at rank 4 after 224
runs and missed the optimum at every point. With them it reaches rank 11 after 1.9k
runs and matches INTERP at all 32 points, for 2.1× fewer circuit runs in total. The
benchmarks below were run without this option (`tci_sweep`'s default,
`angle_pivots=0`).

### `solve_qubo` — Raw QUBO

Minimises E(x) = xᵀQx over x ∈ {0, 1}ⁿ. Linear terms go on the diagonal (xᵢ² = xᵢ);
Q need not be symmetric.

```python
from tciqaoa import solve_qubo

res = solve_qubo(
    Q              = Q,      # (n, n) QUBO matrix, n <= ~22
    p              = 2,      # QAOA depth
    tci            = True,   # True / False (or "on" / "off")
    extra_restarts = 8,      # p >= 3: random restarts per depth in the chain
    angle_pivots   = 4,      # TT build: also start from 4 x 4 points over the angle box
    shots          = 4096,   # bitstrings sampled for the read-out
    seed           = 0,
)

res["angles"]    # optimal (gamma_1 .. gamma_p, beta_1 .. beta_p)
res["cost"]      # optimised <C> (normalised: mean 0, std 1 over all bitstrings)
res["n_evals"]   # circuit runs used (including the TT build)
res["tci"]       # whether TCI was used
res["report"]    # dict: best_sampled_x, best_sampled_energy, approx_ratio,
                 #       p_ground, expected_energy, shots
```

### `solve_maxcut` — Graph partitioning

Splits the nodes into two sets to maximise the number of cut edges, with the QUBO
Q = −(D − A) (as in hybridq).

```python
from tciqaoa import solve_maxcut

res = solve_maxcut(
    A     = A,       # (n, n) symmetric adjacency matrix (weights allowed)
    p     = 2,
    tci   = False,
    shots = 4096,
    seed  = 0,
)

res["partition"]     # 0/1 per node, shape (n,)
res["cut_value"]     # (weighted) number of cut edges
res["n_edges"]       # edges in the graph
res["approx_ratio"]  # of the expected cut: 1 = every shot is a maximum cut
res["p_ground"]      # probability of sampling a maximum cut
res["angles"], res["n_evals"], res["tci"]
```

### `solve_portfolio` — Asset selection and efficient frontiers

Selects exactly k of n assets with the Markowitz QUBO
λ·Σ − diag(μ / max μ) + A·(Σᵢxᵢ − k)² (`portfolio_qubo`; the budget penalty A defaults
to a value that keeps every local minimum at exactly k assets).

```python
from tciqaoa import solve_portfolio

res = solve_portfolio(
    mu            = mu,      # (n,) expected returns
    sigma         = sigma,   # (n, n) covariance matrix
    k             = 4,       # number of assets to select
    p             = 2,
    tci           = False,
    risk_aversion = 0.5,     # lambda; or (lo, hi) for a frontier, see below
    penalty       = None,    # budget penalty A (None: safe default)
    shots         = 4096,
    seed          = 0,
)

res["selected_assets"]   # indices of the chosen assets
res["expected_return"]   # mu . x
res["variance"]          # x^T Sigma x
res["sharpe"]            # return / sqrt(variance)
res["feasible"]          # exactly k assets?
res["approx_ratio"], res["p_ground"], res["angles"], res["n_evals"], res["tci"]
```

**Efficient frontier:** pass `risk_aversion=(lo, hi)` and `nt` (a divisor of 32: 4, 8,
16 or 32 evenly spaced values from lo to hi). With `tci=True` one tensor train over
(angles × risk aversion) serves every λ; this is where TCI saves circuit runs.

```python
for tci in (True, False):
    fr = solve_portfolio(mu, sigma, k=4, p=1, tci=tci, risk_aversion=(0.0, 1.0), nt=32)
    print(tci, fr["n_evals"])            # True 15,903 / False 27,714 circuit runs

fr["frontier"]   # one dict per lambda: risk_aversion, selected_assets, expected_return,
                 #                      variance, sharpe, feasible, approx_ratio, p_ground
fr["tt"]         # the tensor train (tci=True), reusable: see solve_family(tt=...)
```

At p ≥ 2, `n_anchors=8` seeds the chain at 8 of the λ values only, which cuts the cost
on long frontiers.

### `solve_family` — Any parameterised family of QUBOs

The general form of the frontier: a family Q(t₁, .., t_k) with k = 1, 2, 3, … parameters
(a risk aversion, a penalty weight, a field strength, a blend of two problems…), solved
at every point of a grid. Write `make_Q` with each tⱼ in [0, 1) and rescale inside it;
points are snapped to a 2⁵ grid per parameter and visited neighbour to neighbour (a
snake path), which the chain and the warm start rely on.

```python
from tciqaoa import solve_family, portfolio_qubo

# two parameters: risk aversion in [0, 1) and budget penalty in [2, 10)
make_Q2 = lambda t1, t2: portfolio_qubo(mu, sigma, k=4, risk_aversion=t1, penalty=2 + 8 * t2)

res = solve_family(
    make_Q       = make_Q2,  # (t_1, .., t_k) -> (n, n) QUBO matrix
    p            = 1,
    tci          = True,
    n_params     = 2,        # k
    nt           = 8,        # points per parameter axis (divides 32): 8 x 8 = 64 points
    points       = None,     # or your own list of k-tuples, in neighbour order
    n_anchors    = None,     # p >= 2: seed the chain at this many points only
    tt           = None,     # a TT from an earlier call, to reuse
    report       = False,    # True: add the read-out per point
)

res["params"]    # the points, as k-tuples
res["angles"]    # optimal angles per point
res["costs"]     # optimised <C> per point
res["n_evals"]   # circuit runs, including the TT build: 31,728 (tci=False: 55,392)
res["tt"]        # the depth-1 TT (tci=True), reusable at another depth or other points
res["reports"]   # with report=True: best_sampled_x, approx_ratio, p_ground, ... per point
```

At p = 2, seeded at 8 of these 64 points, the same family costs 136,193 circuit runs
with TCI and 60,213 without, and the results without TCI are slightly better (mean
⟨C⟩ −0.8272 vs −0.8267): with 8 seeding points on a smooth 8 × 8 grid, coverage is
good enough that TCI does not pay (see [Conclusions](#conclusions)).

### Without QAOA: `solve_bits`

Minimises xᵀQx directly over bitstrings, without a circuit, so n can be 80 or more.

```python
from tciqaoa import solve_bits

res = solve_bits(Q, tci=True)    # TCI on exp(-beta E(x)) with staged beta (tci_minimize_bits)
res = solve_bits(Q, tci=False)   # random-restart steepest-descent local search
res = solve_bits([Q1, Q2, Q3, Q4], tci=True)   # a list (length 2^m): one TT for all

res["x"], res["energy"], res["n_energy_evals"], res["tci"]
```

TCI here beats hybridq but loses to local search given the same time (benchmarks 2
and 7), so `tci=False` is the better solver; `tci=True` is there to compare.

### Command line

```bash
python -m tciqaoa Q.npy --p 2 --tci on --brute-force     # QUBO from .npy or a text file
python -m tciqaoa Q.npy --mode bits --tci off            # bitstrings directly
python -m tciqaoa --family my_family.py:make_Q --p 2 --nt 16 --anchors 8 --tci on
python -m tciqaoa --family my_family.py:make_Q2 --n-params 2 --p 1 --nt 8 --tci off
```

`--family FILE.py:FUNC` names a function `FUNC(t_1, .., t_k) -> Q`.

### Building blocks

The interfaces wrap the lower-level functions listed under
[What is in here](#what-is-in-here), for finer control: `optimize_tci` (TCI over the
angles of one instance), `tci_sweep` and `sweep_chain` (with `use_tci`, `n_anchors`,
`warm_start`), `sweep_library`, `tci_minimize_bits` and `tci_sweep_bits`. For example:

```python
from tciqaoa import QAOA, maxcut_qubo, random_graph, optimize_tci

q = QAOA(maxcut_qubo(random_graph(12, 0.4, seed=3)), p=1)
r = optimize_tci(q)                          # TCI over (gamma, beta)
print(r["tt_max_rank"], r["pivot_error"], r["n_evals"])
print(q.report(r["angles"]))                 # approx ratio, P(ground), best sampled x
```

---

## Which problems this solves

All problems are **QUBOs**: minimise E(x) = xᵀQx over binary vectors x ∈ {0,1}ⁿ.
This is the same problem class as hybridq, and the same class QAOA targets on quantum
hardware. Four problem types are benchmarked:

| problem | QUBO | example use |
|---|---|---|
| **MaxCut** | Q = −(D − A), from the graph Laplacian | network partitioning, clustering, circuit layout |
| **Random dense QUBO** | Q with Gaussian entries, 50 % density | a worst-case stress test with no structure |
| **Portfolio selection** | Markowitz λxᵀΣx − μᵀx + A(Σx − k)² (mean-variance with a budget penalty) | choosing k assets; sweeping λ traces the efficient frontier |
| **Job scheduling** | Σⱼ loadⱼ² + λ Σᵢ(Σⱼ xᵢⱼ − 1)², xᵢⱼ = job i on machine j | balancing jobs over machines (a makespan proxy) |

hybridq's `solve_portfolio` maximises a Sharpe ratio; that is a quotient, not a QUBO,
so the portfolio benchmarks use mean-variance instead. hybridq's `solve_scheduling`
QUBO uses Σᵢ tᵢ xᵢⱼ as its objective, which equals the total work for *every* feasible
assignment, so it does not see the makespan at all; the scheduling family here uses the
squared machine loads instead and keeps hybridq's one-machine-per-job penalty. Any
other QUBO can be passed straight to the same functions.

Solving a QUBO with QAOA involves two separate problems, and TCI is applied to each:

| | sub-problem | type | what TCI learns |
|---|---|---|---|
| **A** | choose the 2p circuit angles (γ, β) that minimise ⟨C⟩ | continuous, expensive black box (each evaluation is a circuit run) | the whole energy landscape ⟨C⟩(γ, β) |
| **B** | find the bitstring x with the lowest energy | discrete, 2ⁿ candidates | the Boltzmann weight exp(−βE(x)) over all bitstrings |

## How TCI is applied

TCI (tensor cross interpolation) builds a **tensor train** (a matrix product state)
approximating a function f. It evaluates f only at a small, adaptively chosen set of
points ("pivots") and never forms the full tensor. If f has low rank in that format,
a few hundred to a few thousand evaluations describe a function on 2⁴⁰ or more points.

### A. TCI on the QAOA angle landscape (`optimize_tci`)

1. **Discretise.** Each angle gets a grid of 2ᴿ points (R = 5 by default): γ ∈ [0, 2] in
   units of the normalised cost, β ∈ [0, π). Each grid index is written in binary, so
   the landscape becomes a tensor with 2p·R legs of dimension 2. This is the
   *quantics* representation (`xfacpy.QuanticsGrid`), the same construction as in the
   earlier QTCI work on two-particle correlation matrices.
2. **Learn.** `xfacpy.QTensorCI` asks for ⟨C⟩ at the points it chooses. Each request
   is one QAOA circuit evaluation. Two-site sweeps let the rank grow until the pivot
   error drops below `reltol`, or the rank hits `bond_dim`.
3. **Optimise.** The best distinct points TCI sampled are polished with COBYLA on the
   true circuit, so the reported optimum never depends on surrogate error.
4. **Keep.** The learned tensor train is returned (`return_tt=True`). It is the full
   landscape: it can be evaluated anywhere, minimised again, or reused (next section).

Diagnostics: `tt_max_rank` and `pivot_error`. At p = 1 the rank is 6–7 and the pivot
error is about 1e-3: a faithful, reusable surrogate. At p ≥ 2 the rank hits the cap
and the pivot error stays near 1, so the surrogate should not be trusted.

### B. TCI on the bitstring landscape (`tci_minimize_bits`)

1. **Build the tensor.** F(x₁, …, xₙ) = exp(−β (E(x) − E_ref)) is a tensor with n legs
   of dimension 2, one per qubit, the same shape as the QAOA register. Its largest
   entries are the lowest energies.
2. **Order the legs.** A tensor train is a chain, so strongly coupled variables should
   be neighbours. Variables are sorted by the Fiedler vector of |Q| (the flattening
   order matters, as it did for the correlation matrices).
3. **Learn and search.** `xfacpy.TensorCI2` starts from a local-search minimum and adds
   other local minima as global pivots. Its pivot search goes where the interpolation
   error is largest, i.e. towards the largest entries of F: low energies.
4. **Anneal.** β is raised in stages (0.5 → 4, in units of the typical single-flip
   energy). This sharpens F around the minimum, the tensor-network counterpart of
   deeper QAOA layers or a slower anneal.
5. **Polish.** The lowest-energy configurations TCI touched get a 1-flip local search.

This never builds the 2ⁿ table, so it runs at n = 80 and beyond, where statevector
QAOA cannot.

### When TCI should and should not win

TCI succeeds when the function is **low rank** in tensor-train form: smooth, nearly
separable, or with short-range coupling along the chain. The number of local minima
does not matter; the rank does. Always check the returned rank and pivot error:

| object | rank / pivot error here | verdict |
|---|---|---|
| p = 1 angle landscape | 6–7 / 1e-3 | low rank: TCI is reliable and reusable |
| p ≥ 2 angle landscape | cap (20) / ~1 | not low rank on this grid: use INTERP |
| Boltzmann tensor of a dense random QUBO | cap (20) | not low rank: TCI adds nothing over local search |
| Boltzmann tensor of a sparse / chain-like QUBO | untested | expected to be low rank: the case to test next |

## Why TCI: reusability

A classical optimiser (COBYLA, INTERP, SA, SB) returns **one point** and forgets
everything it evaluated. TCI returns **the function itself**, as a compact tensor train.
That changes the cost accounting whenever the same landscape is needed more than once:

| use | one-shot optimiser | TCI tensor train |
|---|---|---|
| optimum of this instance | full search | minimise the stored TT, then a short polish |
| new instance of the same family | full search again | reuse the TT as the starting landscape |
| landscape map, sensitivity, robustness | thousands more circuit runs | free (evaluate the TT) |
| gradients, averages, integrals over the angle box | more circuit runs | free (`tt.sum`, finite differences on the TT) |
| warm start for depth p + 1 | interpolation heuristic (INTERP) | the converged p landscape as a starting point |

**Why reuse is plausible here: the optimal angles concentrate.** Within a problem
family, the optimal angles hardly change from one instance to the next. At p = 1, two
different random MaxCut graphs (seeds 3 and 4) gave γ = 1.202 and 1.204, β = 2.802 and
2.800; at p = 2, γ ≈ (0.89, 1.74) vs (0.92, 1.80). A p = 1 landscape learned once
therefore points straight at the optimum of the next instance.

**What the measurement says ([benchmark 4](#4-reuse-does-a-learned-landscape-pay-off-on-new-instances)).**

- **Reuse works.** At p = 1 the stored TT pays for itself after 2–6 new instances and
  uses about 2× (1.6–2.5×) fewer circuit runs per instance than INTERP from scratch, with the same
  answer on 300/300 test instances, including other sizes and other problem families.
- **But the tensor train is not what makes it work.** Keeping only the training
  optimum (2p numbers) and polishing from there reaches the same answer on all 300
  p = 1 test instances, with 3–4× fewer runs than TT-reuse. The angles concentrate so
  strongly that the extra information in the TT (the full map and the other basins)
  is rarely needed. The one exception is at p = 2, MaxCut → QUBO: the stored point is a
  MaxCut symmetry image (β vs β + π/2) that lies in the wrong basin for QUBO, and it
  misses on all 50 instances. TT-reuse, which keeps several minima, gets 50/50, and so
  does the point plus its 2^p β images (`optimize_from_images`) at a quarter of
  TT-reuse's cost.
- **At p = 2 the TT barely pays back.** Building it costs 5,100–9,400 runs, and
  since it never converged (rank 20 at the cap), TT-reuse is only slightly cheaper
  per instance than INTERP, and in one scenario (MaxCut → QUBO) more expensive.

So for *finding angles for new instances*, store the angles and their symmetry images,
not the landscape. The
TT's distinct value lies in what a single point cannot give: the full map, gradients
and integrals for free, and robustness when the optimum jumps between basins from one
instance to the next. The families in benchmark 4 have no such jumps, apart from the
symmetry-image case above.
[Benchmark 5](#5-sweeps-one-tensor-train-for-a-whole-problem-family) tests a family that
does: there, the sweep TT is the only reuse method that never misses (at p = 1).

Reuse only pays when the tensor train **converged**. The p = 1 landscape qualifies; the
p ≥ 2 landscapes and the dense-QUBO Boltzmann tensors in this repo do not.

---

## What is in here

| module | what it does |
|---|---|
| `tciqaoa.QAOA` | **Real** gate-model QAOA, `prod_k exp(-iβ_k ΣX) exp(-iγ_k C) \|+⟩ⁿ`, simulated exactly as a 2ⁿ statevector (checked against dense matrix exponentials to 1e-16). Up to n ≈ 22. |
| `optimize_restarts` | Baseline angle search: random starts + COBYLA. |
| `optimize_interp` | Baseline angle search: INTERP warm start (Zhou et al., PRX 10, 021067 (2020)). |
| `optimize_tci` | **TCI route 1**: learn the landscape ⟨C⟩(γ, β) as a quantics tensor train (`xfacpy.QTensorCI`), then COBYLA-polish the best TCI-sampled points. Returns the TT rank and pivot error as built-in diagnostics. |
| `landscape_minima`, `optimize_from`, `beta_images`, `optimize_from_images` | **Reuse**: read the lowest distinct minima off a stored TT (no circuit runs), and polish from any start angles on a new instance. `optimize_from_images` reuses a stored optimum across families: it ranks its 2^p images under β_k → β_k + π/2 (symmetries of Z2-symmetric costs such as MaxCut) by one run each and polishes the best. |
| `SweepFamily`, `tci_sweep`, `solve_from_sweep`, `solve_lifted`, `sweep_library`, `sweep_chain` | **Sweeps** (`sweep_chain(family, ts, p)` runs the recommended pipeline at any depth): one TCI over (angles × a problem parameter t) for a whole family Q(t); read any slice's minima off the TT and polish them. `solve_lifted` uses the depth-1 sweep TT for a depth-p solve (INTERP lift); `sweep_library` solves a whole family at depth p with a shared basin library, seeded by a TCI sampler (default), random restarts or a p = 2 TCI sweep, and reports which start produced each optimum; `lower=` chains depths by lifting a p − 1 run's optima and library. |
| `tci_minimize_bits` | **TCI route 2**: TCI (`xfacpy.TensorCI2`) on the n-leg Boltzmann tensor `F(x) = exp(-β(E(x) − E_ref))`, x ∈ {0,1}ⁿ (one leg per qubit, as in the QAOA register), with staged β (deeper layers / slower anneal) and a 1-flip polish. Never builds the 2ⁿ table, so it runs at n = 80+. Variables are ordered along the chain by the Fiedler vector of \|Q\|, since flattening order matters for tensor trains. |
| `tci_sweep_bits` | Route 2 for a whole family: **one** TCI over (parameter bits × variable bits), e.g. a full efficient frontier over risk aversion, with cross-parameter reuse of the best configurations. |
| `solve_qubo`, `solve_maxcut`, `solve_portfolio`, `solve_family`, `solve_bits` (`tciqaoa.solve`, `python -m tciqaoa`) | **Main entry points**, each with `tci=True/False`: one QUBO, MaxCut, a portfolio or efficient frontier, or any family of QUBOs at many parameter points, with QAOA; or the QUBO's bitstrings directly. See [Four problem interfaces](#four-problem-interfaces). |
| `portfolio_data`, `portfolio_qubo` | hybridq's sector-structured portfolio generator, and Markowitz selection of k assets as a QUBO with the **linear** return −μᵀx and a budget penalty that keeps every local minimum feasible. |

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
measurement shots       ------------------------------->     TCI pivots (full search)
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
| maxcut | 10 | 1 | tci | 0.8034 | 0.131 | 5/5 | 573 | 0.1 | 7 | 3e-03 |
| maxcut | 10 | 2 | interp | 0.8766 | 0.296 | 5/5 | 564 | 0.1 |  |  |
| maxcut | 10 | 2 | restarts | 0.8766 | 0.296 | 5/5 | 2086 | 0.4 |  |  |
| maxcut | 10 | 2 | tci | 0.8766 | 0.296 | 5/5 | 6359 | 1.6 | 20 | 2e-01 |
| maxcut | 10 | 3 | interp | 0.9172 | 0.459 | 5/5 | 789 | 0.1 |  |  |
| maxcut | 10 | 3 | restarts | 0.9172 | 0.459 | 5/5 | 2932 | 0.8 |  |  |
| maxcut | 10 | 3 | tci | 0.9172 | 0.457 | 5/5 | 15853 | 5.1 | 20 | 1e+00 |
| maxcut | 14 | 1 | interp | 0.7909 | 0.016 | 5/5 | 528 | 0.5 |  |  |
| maxcut | 14 | 1 | restarts | 0.7909 | 0.016 | 5/5 | 1062 | 1.1 |  |  |
| maxcut | 14 | 1 | tci | 0.7909 | 0.016 | 5/5 | 718 | 0.7 | 7 | 1e-02 |
| maxcut | 14 | 2 | interp | 0.8568 | 0.067 | 5/5 | 711 | 1.1 |  |  |
| maxcut | 14 | 2 | restarts | 0.8568 | 0.067 | 5/5 | 2201 | 4.8 |  |  |
| maxcut | 14 | 2 | tci | 0.8568 | 0.067 | 5/5 | 4870 | 10.7 | 20 | 1e-01 |
| maxcut | 14 | 3 | interp | 0.8937 | 0.141 | 5/5 | 975 | 1.7 |  |  |
| maxcut | 14 | 3 | restarts | 0.8937 | 0.141 | 5/5 | 2950 | 9.1 |  |  |
| maxcut | 14 | 3 | tci | 0.8925 | 0.138 | 5/5 | 15750 | 46.9 | 20 | 1e+00 |
| qubo | 10 | 1 | interp | 0.7340 | 0.028 | 5/5 | 302 | 0.0 |  |  |
| qubo | 10 | 1 | restarts | 0.7340 | 0.028 | 5/5 | 661 | 0.1 |  |  |
| qubo | 10 | 1 | tci | 0.7340 | 0.028 | 5/5 | 441 | 0.0 | 6 | 4e-03 |
| qubo | 10 | 2 | interp | 0.8608 | 0.089 | 5/5 | 387 | 0.0 |  |  |
| qubo | 10 | 2 | restarts | 0.8608 | 0.089 | 5/5 | 1998 | 0.4 |  |  |
| qubo | 10 | 2 | tci | 0.8608 | 0.089 | 5/5 | 6936 | 1.2 | 20 | 4e-01 |
| qubo | 10 | 3 | interp | 0.9102 | 0.172 | 5/5 | 623 | 0.1 |  |  |
| qubo | 10 | 3 | restarts | 0.9098 | 0.170 | 5/5 | 2909 | 0.8 |  |  |
| qubo | 10 | 3 | tci | 0.9093 | 0.169 | 5/5 | 13176 | 3.4 | 20 | 1e+00 |
| qubo | 14 | 1 | interp | 0.7433 | 0.009 | 5/5 | 479 | 0.5 |  |  |
| qubo | 14 | 1 | restarts | 0.7433 | 0.009 | 5/5 | 907 | 1.0 |  |  |
| qubo | 14 | 1 | tci | 0.7433 | 0.009 | 5/5 | 533 | 0.6 | 6 | 1e-02 |
| qubo | 14 | 2 | interp | 0.8405 | 0.031 | 5/5 | 667 | 0.9 |  |  |
| qubo | 14 | 2 | restarts | 0.8489 | 0.035 | 5/5 | 2409 | 5.1 |  |  |
| qubo | 14 | 2 | tci | 0.8405 | 0.031 | 5/5 | 5270 | 11.0 | 20 | 5e-01 |
| qubo | 14 | 3 | interp | 0.8884 | 0.068 | 5/5 | 955 | 1.8 |  |  |
| qubo | 14 | 3 | restarts | 0.8900 | 0.070 | 5/5 | 2985 | 9.3 |  |  |
| qubo | 14 | 3 | tci | 0.8883 | 0.067 | 5/5 | 13645 | 41.6 | 20 | 1e+00 |

Takeaways

- **p = 1: TCI works.** The 2-angle landscape is a converged, low-rank quantics tensor
  train, and TCI finds the same optimum as 10 COBYLA restarts with ~30 % fewer circuit
  evaluations (INTERP, which at p = 1 is just 5 restarts, is cheaper still). Unlike
  either baseline, it also returns the whole landscape as a converged TT that can be
  stored and reused; this table scores single solves only, so that value is not counted.
- **p ≥ 2: TCI loses.** The rank hits the bond-dimension cap and the pivot error stops
  falling, so the surrogate never converges. The COBYLA polish still recovers the
  optimum in most cases, but at 6–20× the cost of INTERP.
- **The angles concentrate.** Optimal angles are nearly identical across instances of
  the same family (for example p = 2 MaxCut seeds 3 and 4: γ ≈ (0.89, 1.74) vs
  (0.92, 1.80)). This is the regime where a TCI landscape learned once and reused
  could pay off; see [Why TCI: reusability](#why-tci-reusability).

### 2. Solving the QUBO: TCI-bits vs hybridq and classical baselines

`HYBRIDQ_PATH=... python benchmarks/bench_solvers.py --seeds 10` (10 instances per row).

| problem | n | ref | hybridq-SB | hybridq-QA | SA | Tabu | LS-multistart | TCI-bits | LS-time-matched |
|---|---|---|---|---|---|---|---|---|---|
| maxcut | 16 | exact | 0/10 (32.7%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) |
| maxcut | 20 | exact | 0/10 (27.6%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) |
| maxcut | 30 | best known | 0/10 (26.9%) | 10/10 (0.0%) | 10/10 (0.0%) | 9/10 (0.1%) | 9/10 (0.1%) | 10/10 (0.0%) | 10/10 (0.0%) |
| maxcut | 50 | best known | 0/10 (48.7%) | 0/10 (22.9%) | 7/10 (0.1%) | 5/10 (0.5%) | 6/10 (0.1%) | 9/10 (0.0%) | 10/10 (0.0%) |
| maxcut | 80 | best known | 0/10 (64.9%) | 0/10 (28.1%) | 3/10 (0.2%) | 1/10 (0.7%) | 0/10 (0.4%) | 2/10 (0.4%) | 10/10 (0.0%) |
| qubo | 16 | exact | 7/10 (1.1%) | 10/10 (0.0%) | 9/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) |
| qubo | 20 | exact | 9/10 (0.6%) | 10/10 (-0.0%) | 10/10 (-0.0%) | 10/10 (-0.0%) | 10/10 (-0.0%) | 10/10 (-0.0%) | 10/10 (-0.0%) |
| qubo | 30 | best known | 3/10 (3.2%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) |
| qubo | 50 | best known | 0/10 (17.1%) | 0/10 (13.9%) | 9/10 (0.0%) | 8/10 (0.1%) | 10/10 (0.0%) | 10/10 (0.0%) | 10/10 (0.0%) |
| qubo | 80 | best known | 0/10 (23.3%) | 0/10 (21.8%) | 9/10 (0.1%) | 7/10 (0.2%) | 10/10 (0.0%) | 8/10 (0.2%) | 10/10 (0.0%) |

Cell = instances where the method reached the reference (mean relative energy gap).

| problem | n | hybridq-SB time (s) | hybridq-QA time (s) | SA time (s) | Tabu time (s) | LS-multistart time (s) | TCI-bits time (s) | LS-time-matched time (s) |
|---|---|---|---|---|---|---|---|---|
| maxcut | 16 | 0.08 | 0.04 | 0.05 | 0.01 | 0.00 | 0.09 | 0.09 |
| maxcut | 20 | 0.08 | 0.05 | 0.05 | 0.01 | 0.00 | 0.18 | 0.18 |
| maxcut | 30 | 0.08 | 0.05 | 0.05 | 0.01 | 0.00 | 0.50 | 0.51 |
| maxcut | 50 | 0.18 | 0.09 | 0.05 | 0.02 | 0.01 | 1.82 | 1.82 |
| maxcut | 80 | 0.26 | 0.09 | 0.05 | 0.02 | 0.01 | 4.14 | 4.14 |
| qubo | 16 | 0.08 | 0.04 | 0.05 | 0.01 | 0.00 | 0.05 | 0.05 |
| qubo | 20 | 0.08 | 0.04 | 0.05 | 0.01 | 0.00 | 0.11 | 0.11 |
| qubo | 30 | 0.08 | 0.04 | 0.05 | 0.01 | 0.01 | 0.28 | 0.28 |
| qubo | 50 | 0.17 | 0.09 | 0.05 | 0.02 | 0.01 | 1.09 | 1.09 |
| qubo | 80 | 0.25 | 0.09 | 0.05 | 0.02 | 0.01 | 2.94 | 2.94 |

TCI-bits mean energy evaluations / max TT rank: maxcut n=16: 16004 / 20, maxcut n=20: 38771 / 20, maxcut n=30: 101748 / 20, maxcut n=50: 229956 / 20, maxcut n=80: 416762 / 20, qubo n=16: 8880 / 20, qubo n=20: 19941 / 20, qubo n=30: 48102 / 20, qubo n=50: 127213 / 20, qubo n=80: 274778 / 20

The controls: `LS-multistart` is 100 random-restart steepest descents. `LS-time-matched`
spends exactly TCI-bits' wall time on random-restart steepest descent, which isolates
what the TCI search adds on top of its own local-search polish.

Takeaways

- **TCI-bits ≫ hybridq.** TCI-bits reaches the optimum on every exact instance
  (n ≤ 20) and stays within 0.4 % of best known at n = 50–80. hybridq-SB misses
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

**Portfolio instances: TCI on the portfolio objective itself.**
`python benchmarks/bench_solvers.py --kinds portfolio --out results/solvers_portfolio.json`.
Markowitz selection of k = n/4 assets, λ = 1, on hybridq's own `make_portfolio_data`
(sector-structured covariance), with the linear return and a budget penalty large
enough that every local minimum holds exactly k assets (`portfolio_qubo`).

| n | ref | hybridq-SB | hybridq-QA | SA | Tabu | LS-multistart | TCI-bits | LS-time-matched |
|---|---|---|---|---|---|---|---|---|
| 16 | exact | 0/10 (674) | 0/10 (2.39) | 0/10 (1.82) | 5/10 (0.131) | 9/10 (0.00725) | 7/10 (0.0364) | 10/10 (2.27e-14) |
| 20 | exact | 0/10 (1.35e+03) | 0/10 (3.16) | 0/10 (2.61) | 4/10 (0.163) | 4/10 (0.0828) | 1/10 (0.176) | 10/10 (1.14e-13) |
| 30 | best known | 0/10 (5.66e+03) | 0/10 (12) | 0/10 (14.3) | 0/10 (0.804) | 0/10 (0.608) | 0/10 (0.7) | 10/10 (0) |
| 50 | best known | 0/10 (9.13e+04) | 0/10 (4.36e+04) | 0/10 (49.5) | 0/10 (1.79) | 0/10 (1.52) | 0/10 (2.38) | 10/10 (0) |
| 80 | best known | 0/10 (3.42e+05) | 0/10 (2.4e+05) | 0/10 (110) | 0/10 (2.98) | 0/10 (2.97) | 0/10 (4.74) | 10/10 (0) |

Cell = instances where the method reached the reference (mean absolute gap in cost
units; relative gaps are meaningless here because the penalty adds a large constant).
TCI-bits time / evaluations: n=16: 0.10 s / 15,858 evals; n=20: 0.20 s / 36,550 evals; n=30: 0.57 s / 112,969 evals; n=50: 2.03 s / 258,085 evals; n=80: 4.77 s / 482,848 evals. Its TT rank sits at the cap (20).

- **On portfolios TCI-bits does worse than on MaxCut and random QUBO:** 7/10 exact at
  n = 16, 1/10 at n = 20, 0/10 above. Time-matched local search hits 10/10 everywhere.
  The budget penalty couples every pair of assets, so the Boltzmann tensor is far from
  low rank.
- **hybridq-SB's portfolios break the budget** (gaps in the thousands are penalty
  violations), and hybridq-QA does too at n ≥ 50. SA (hybridq's `examples.py`) keeps
  the budget but never reaches the optimum.

### 4. Reuse: does a learned landscape pay off on new instances?

`python benchmarks/bench_reuse.py --tests 50` (12 scenarios × 50 new instances).

Train once: `optimize_tci(return_tt=True)` on one training instance (seed 10000).
Then on each new instance compare:

- **interp**: INTERP from scratch, no reuse.
- **point**: polish from the training optimum only, 2p stored numbers. This is the control.
- **point+images** (`optimize_from_images`): the same 2p numbers and their 2^p − 1
  images under β_k → β_k + π/2 (`beta_images`). Each is evaluated once on the new
  instance and the best one is polished: 2^p runs more than *point*.
- **TT-reuse**: polish from the 3 lowest distinct minima of the stored tensor train.
  At p = 1 these come from evaluating the TT on the full grid, with no circuit runs.

"OK" means within 1e-3 of the best approximation ratio any method found on that instance.
"Total for 50" includes the one-off build cost for every reuse method.

| p | train → test | build (rank) | interp OK / runs | point OK / runs | point+images OK / runs | TT-reuse OK / runs | TT break-even vs interp | total for 50: interp / point / point+images / TT |
|---|---|---|---|---|---|---|---|---|
| 1 | MaxCut12 → MaxCut12 | 688 (7) | 50/50 / 477 | 50/50 / 50 | 50/50 / 52 | 50/50 / 190 | 2.4 | 23,849 / **3,165** / 3,265 / 10,211 |
| 1 | MaxCut12 → MaxCut16 | 688 (7) | 50/50 / 641 | 50/50 / 110 | 50/50 / 112 | 50/50 / 298 | 2 | 32,055 / **6,200** / 6,300 / 15,565 |
| 1 | MaxCut12 → sparse MaxCut12 | 688 (7) | 50/50 / 380 | 50/50 / 47 | 50/50 / 49 | 50/50 / 162 | 3.1 | 19,024 / **3,019** / 3,119 / 8,771 |
| 1 | MaxCut12 → QUBO12 | 688 (7) | 50/50 / 333 | 50/50 / 52 | 50/50 / 54 | 50/50 / 212 | 5.7 | 16,657 / **3,288** / 3,388 / 11,289 |
| 1 | QUBO12 → QUBO12 | 637 (7) | 50/50 / 333 | 50/50 / 41 | 50/50 / 43 | 50/50 / 136 | 3.2 | 16,657 / **2,710** / 2,810 / 7,441 |
| 1 | QUBO12 → MaxCut12 | 637 (7) | 50/50 / 477 | 50/50 / 100 | 50/50 / 104 | 50/50 / 283 | 3.3 | 23,849 / **5,645** / 5,823 / 14,768 |
| 2 | MaxCut12 → MaxCut12 | 5,140 (20) | 50/50 / 663 | 50/50 / 150 | 50/50 / 153 | 50/50 / 480 | 28 | 33,144 / **12,633** / 12,797 / 29,116 |
| 2 | MaxCut12 → MaxCut16 | 5,140 (20) | 50/50 / 860 | 50/50 / 164 | 50/50 / 169 | 50/50 / 624 | 22 | 43,009 / **13,316** / 13,576 / 36,331 |
| 2 | MaxCut12 → sparse MaxCut12 | 5,140 (20) | 50/50 / 562 | 50/50 / 150 | 50/50 / 154 | 50/50 / 518 | 117 | 28,091 / **12,655** / 12,857 / 31,043 |
| 2 | MaxCut12 → QUBO12 | 5,140 (20) | 50/50 / 489 | 0/50 / 273 · miss 0.04–0.18 | 50/50 / 162 | 50/50 / 707 | never | 24,452 / 18,787 (fails) / **13,234** / 40,504 |
| 2 | QUBO12 → QUBO12 | 9,388 (20) | 50/50 / 489 | 50/50 / 149 | 50/50 / 153 | 50/50 / 443 | 204 | 24,452 / **16,820** / 17,020 / 31,536 |
| 2 | QUBO12 → MaxCut12 | 9,388 (20) | 50/50 / 663 | 50/50 / 213 | 50/50 / 216 | 50/50 / 479 | 51 | 33,144 / **20,015** / 20,213 / 33,319 |

Runs are mean circuit evaluations per new instance.

Takeaways

- **Reuse is real.** INTERP, point+images and TT-reuse reached the best answer on all
  600 test instances, the plain point method on 550 of them, and every reuse method cuts
  the cost of a new instance well below INTERP from scratch. The p = 1 landscape
  transfers across sizes (12 → 16 qubits), densities and even problem families
  (MaxCut ↔ random QUBO).
- **The angles carry the reuse, not the landscape.** Storing 2p numbers beats storing the
  tensor train by 3–4× per instance at p = 1, and is the cheapest in every scenario. With
  the build cost included, the point method is 4–7.5× cheaper than INTERP over 50
  instances at p = 1 and 1.5–3× cheaper at p = 2. The point method's own build cost
  is overstated here: it used the TCI run, while an ordinary INTERP solve (300–500
  runs) would be enough to find the training angles.
- **But a single stored point can sit in the wrong symmetry image.** At p = 2,
  MaxCut12 → QUBO12, the point method misses on all 50 instances (by 0.04–0.18 in
  approximation ratio). MaxCut has a bit-flip symmetry that makes β and β + π/2
  equivalent; a random QUBO does not, and only one of the two images of the MaxCut
  optimum lies in the QUBO basin. The earlier run (rook pivoting, see the note in
  benchmark 9) built a different TT for this scenario that happened to return the
  right image, and there the point method got 50/50.
- **Storing the images fixes it for 2^p runs.** point+images ranks the 4 images by one
  circuit run each and polishes the best: **50/50 at 162 runs** per instance, against
  273 for the failing point method, 707 for TT-reuse and 489 for INTERP. Everywhere
  else it costs 2–5 runs more than *point* and gets the same 50/50. It is the best
  reuse method overall: cheapest in the scenario where *point* fails and within 3% of
  *point* elsewhere.
- **Why TT-reuse costs more:** it polishes 3 candidate basins where 1 is usually
  enough, because the best basin hardly moves between instances. Its insurance did
  pay off once (MaxCut → QUBO at p = 2), but the symmetry images give the same
  insurance for a few runs.

### 5. Sweeps: one tensor train for a whole problem family

`python benchmarks/bench_sweep.py --seeds 3 --p 1 2` (n = 10; 4 families × 3 instances,
each solved at 16 values of a parameter t ∈ [0, 1]).

A family Q(t) changes smoothly with t, like the risk-aversion sweep hybridq uses to
trace a portfolio efficient frontier. `tci_sweep` treats t as one more quantics
dimension, so **one** TCI learns the depth-1 landscape ⟨C⟩(γ, β; t) for every member at
once. At each t, `sweep_minima` reads the 3 lowest minima of that slice off the
tensor train, with no circuit runs, and they are polished on the real circuit.

| family | what t sweeps |
|---|---|
| mix | MaxCut (t = 0) blended into a random QUBO (t = 1) |
| field | MaxCut plus a uniform field t·3·Σx |
| penalty | portfolio selection λxᵀΣx − μᵀx + A(Σx − 4)², penalty weight A = 0.05 → 3.05 |
| risk | the same portfolio, risk aversion λ = 0.2 → 6.2 (A = 1) |
| sched (benchmark 9 only) | job scheduling (load balance), penalty weight 0.2 → 1.2 × the largest single-job cost: from "leave jobs unassigned" to "assign every job" |

The methods, all counted in circuit runs, including every one-off cost:

- **interp**: INTERP from scratch at every t.
- **point**: INTERP once at t = 0, then polish from that point at every t.
- **continuation**: INTERP at t = 0, then each t starts from the previous t's optimum.
  This is the standard trick for sweeps.
- **TT-sweep**: one depth-1 TCI build, then polish its 3 slice minima at each t. At
  p = 2 the minima are lifted to depth 2 with the INTERP rule before polishing
  (`solve_lifted`); see *Why not a p = 2 sweep TT* below.
- **TT+cont**: the TT-sweep candidates plus the previous t's optimum.
- **TT+library** (`sweep_library(tt, ..., seeding="restarts")`): the TT-sweep
  candidates, the previous t's optimum, and a shared *basin library*. The library is
  seeded by 4 random depth-p restarts at every other t, sampled in the
  symmetry-reduced box (γ₁ ∈ [0, 2.5], later γ's ∈ [−3, 3], β ∈ [0, π)). Every optimum
  found along the sweep joins it. At each t the 3 library members with the lowest cost
  there (one circuit run each to rank them) are polished too, and a backward pass
  re-polishes t from its neighbour when that is a different basin whose start is
  within 0.3 of the stored result. Every run, anchors included, is counted.
- **library (no TCI)**: the same, without the TT candidates. This is the ablation that
  measures what the TCI part contributes.
- **ref**: 20 random restarts at every t, run on its own. It is part of the "best"
  but is now also scored, with its cost.

OK = within 1e-3 of the best found by any method, including the reference. Each row covers 48 solves (3 instances × 16 values of t). "worst" is the
largest miss in normalised cost.

| p | family | interp | point | continuation | TT-sweep | TT+cont | TT+library | library (no TCI) | ref (20 restarts/t) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | mix | 48/48 · 5,139 | 48/48 · 1,143 | 48/48 · 891 | 48/48 · 4,450 | 48/48 · 5,331 | 48/48 · 9,475 | 48/48 · 5,131 | 48/48 · 21,782 |
| 1 | field | 48/48 · 6,366 | 40/48 · 1,510 · worst 1.10 | 39/48 · 1,184 · worst 1.10 | 48/48 · 5,456 | 48/48 · 6,438 | 48/48 · 11,385 | 48/48 · 5,935 | 48/48 · 28,305 |
| 1 | penalty | 48/48 · 12,315 | 48/48 · 2,338 | 48/48 · 878 | 48/48 · 7,271 | 48/48 · 8,153 | 48/48 · 17,802 | 48/48 · 10,482 | 48/48 · 66,642 |
| 1 | risk | 48/48 · 9,787 | 48/48 · 1,717 | 48/48 · 1,616 | 48/48 · 5,035 | 48/48 · 6,664 | 48/48 · 13,471 | 48/48 · 8,446 | 48/48 · 53,489 |
| **1** | **all** | 192/192 · 8,402 | 184/192 · 1,677 · worst 1.10 | 183/192 · 1,142 · worst 1.10 | 192/192 · 5,553 | 192/192 · 6,646 | 192/192 · 13,033 | 192/192 · 7,498 | 192/192 · 42,555 |
| 2 | mix | 46/48 · 7,084 · worst 0.03 | 46/48 · 2,159 · worst 0.03 | 46/48 · 1,660 · worst 0.03 | 46/48 · 8,307 · worst 0.03 | 46/48 · 10,014 · worst 0.03 | 48/48 · 21,690 | 48/48 · 13,681 | 48/48 · 62,552 |
| 2 | field | 30/48 · 10,405 · worst 0.05 | 22/48 · 4,290 · worst 0.41 | 21/48 · 3,198 · worst 0.41 | 30/48 · 14,745 · worst 0.05 | 30/48 · 17,830 · worst 0.05 | 48/48 · 35,884 | 48/48 · 21,228 | 38/48 · 84,854 |
| 2 | penalty | 11/48 · 16,874 · worst 0.03 | 14/48 · 4,754 · worst 0.04 | 14/48 · 4,436 · worst 0.04 | 12/48 · 15,833 · worst 0.03 | 31/48 · 20,016 · worst 0.02 | 48/48 · 40,441 | 48/48 · 24,624 | 14/48 · 92,286 |
| 2 | risk | 7/48 · 14,587 · worst 0.03 | 8/48 · 5,886 · worst 0.01 | 33/48 · 4,473 · worst 0.01 | 10/48 · 15,576 · worst 0.04 | 34/48 · 20,104 · worst 0.01 | 48/48 · 42,045 | 48/48 · 26,469 | 18/48 · 92,885 |
| **2** | **all** | 94/192 · 12,237 · worst 0.05 | 90/192 · 4,272 · worst 0.41 | 114/192 · 3,442 · worst 0.41 | 98/192 · 13,615 · worst 0.05 | 141/192 · 16,991 · worst 0.05 | 192/192 · 35,015 | 192/192 · 21,500 | 118/192 · 83,144 |
| 3 | mix | 46/48 · 10,530 · worst 0.01 | 46/48 · 4,153 · worst 0.01 | 46/48 · 3,345 · worst 0.01 | 46/48 · 14,727 · worst 0.01 | 46/48 · 18,031 · worst 0.01 | 46/48 · 38,580 · worst 0.01 | 30/48 · 24,748 · worst 0.17 | 36/48 · 94,358 |
| 3 | field | 34/48 · 14,685 · worst 0.05 | 16/48 · 5,010 · worst 0.54 | 25/48 · 4,870 · worst 0.54 | 20/48 · 16,423 · worst 0.05 | 34/48 · 21,448 · worst 0.05 | 39/48 · 47,055 · worst 0.05 | 39/48 · 31,549 · worst 0.05 | 16/48 · 94,504 |
| 3 | penalty | 11/48 · 21,655 · worst 0.04 | 13/48 · 5,390 · worst 0.08 | 41/48 · 4,783 · worst 0.009 | 6/48 · 16,560 · worst 0.03 | 41/48 · 21,343 · worst 0.009 | 41/48 · 45,511 · worst 0.009 | 39/48 · 30,621 · worst 0.07 | 1/48 · 95,743 |
| 3 | risk | 12/48 · 19,387 · worst 0.04 | 0/48 · 6,186 · worst 0.04 | 34/48 · 5,932 · worst 0.03 | 0/48 · 15,576 · worst 0.04 | 33/48 · 21,573 · worst 0.02 | 48/48 · 46,514 | 45/48 · 30,456 · worst 0.008 | 0/48 · 95,947 |
| **3** | **all** | 103/192 · 16,564 · worst 0.05 | 75/192 · 5,185 · worst 0.54 | 146/192 · 4,733 · worst 0.54 | 72/192 · 15,822 · worst 0.05 | 154/192 · 20,599 · worst 0.05 | **174/192 · 44,415 · worst 0.05** | 153/192 · 29,344 · worst 0.17 | 53/192 · 95,138 |

Cells: OK count · mean circuit runs per instance (16 solves).

Takeaways at p = 1

- **The sweep TT is the only reuse method that never misses.** On one *field* instance
  the optimum switches basin partway through the sweep. Point and continuation warm
  starts follow the old basin and miss 8–9 of 16 values, by up to 1.1 in normalised
  cost. The sweep TT holds every basin for every t, so it hits all 192 values.
- **Its cost sits in between.** It is 1.2–1.9× cheaper than INTERP at every t (also
  192/192), because the build (290–2,400 runs) is shared by all 16 values. When
  the warm starts don't fail, they are 3–8× cheaper again.

Takeaways at p = 2

- **The basin library closes the p = 2 gap: 192/192**, with or without the TCI TT
  candidates, against 118/192 for the 20-restart reference and 90–141/192 for every
  other method. The portfolio families are the hard ones at p = 2: INTERP finds the
  optimum at only 7/48 *risk* points, the reference at 18/48.
- **What fixed it.** The missing p = 2 basins are not the lift of any p = 1 minimum.
  They have one negative γ, or γ₂ ≈ 3, or β values unrelated to the p = 1 optimum.
  Lifting tricks don't reach them: Hessian escape from transition states, β + π/2
  images and random kicks got at best 82/96 on the hard instances, vs 75 without. But
  these basins are **family-wide**: by angle concentration, the same basin (e.g.
  γ ≈ (1.22, 3.0), β ≈ (0.96, 1.9) on *field*) is optimal over a whole range of t, so
  it only has to be found once and is then reused. Three details mattered:
  1. *Sample the right box.* Time reversal fixes the sign of only one γ, so the later
     γ's must be sampled with both signs.
  2. *Always polish from the previous t's optimum.* Ranking it with the library
     members dropped it at one point.
  3. *Spread the seeding over t.* 4 restarts at every other t beat 10 at each of 3
     anchors at the same cost: some basins are optimal only over a short range of t.
- **How much is TCI?** Nothing here: without the TT candidates the library also gets
  192/192, at 21.5k runs instead of 35.0k. Benchmark 6 asks where TCI *does* help.
- **Cost.** The library costs 21.5k runs per instance (16 values of t), 1.8× INTERP,
  and a quarter of the reference. If accuracy at p = 2 does not matter, continuation
  is 6× cheaper but misses up to 0.41.

<a id="p--3"></a>Takeaways at p = 3 (`--p 3`; `results/sweep_p3.json`)

- **No method closes the gap.** The best is TT+library, 174/192 (worst 0.05), and here
  the TT candidates do matter: without them the library gets 153/192, and on one *mix*
  instance it settles in a basin 0.17 above the optimum for all 16 values of t. The
  20-restart reference collapses to 53/192 (1/48 on *penalty*, 0/48 on *risk*): random
  restarts at depth 3 rarely land in the best basin.
- **Continuation is strong again on the portfolio families** (41/48 *penalty*, 34/48
  *risk*, 4.8k runs), because the depth-3 optimum moves smoothly with t once found.
  It still fails on *field* (25/48, worst 0.54), where the optimum switches basin.
- **Cost.** TT+library costs 44k runs per instance, 2.7× INTERP and under half the
  reference. These OK counts are against this benchmark's own best; benchmark 6 runs
  many more configurations and finds better optima at some points, so the same
  pipeline scores lower there (147–155/192).

**Why not a p = 2 sweep TT.** We tried, and it cannot work on these problems.
- *Quantics TT over (4 angles, t):* hits the rank cap (30) without converging, costs
  23k–32k runs, and finds only 16–42 of 48 optima per family.
- *Narrower box:* restricting each slice to a box around the lifted p = 1 optimum still
  hits the cap.
- *Ordinary grid, 16–32 points per angle:* reports convergence (pivot error ~1e-3,
  rank 8–13), but its true error on held-out points is 0.27–0.36 RMS on a landscape
  spanning ~4. TCI missed whole features.
- *Pushed until really accurate:* rank ≈ 75, after evaluating 60–70% of all grid points.

So the depth-2 landscape is not low rank in either format. The depth-1 one is, which is
why lifting works.

### 6. Where should TCI go? (TCI placement)

`python benchmarks/bench_tci_placement.py --p 2` and
`python benchmarks/bench_tci_placement_seeds.py` (same 12 instances × 16 values of t as
benchmark 5, at p = 2).

The p = 2 sweep pipeline (`sweep_library`) has two stages where TCI can be used or not;
the polish is COBYLA everywhere (a local TCI would only be a costlier local optimiser).
Every combination was run:

- **p = 1 candidates:** none, or the TCI sweep TT over (γ, β, t) at depth 1, its slice
  minima lifted to p = 2.
- **Library seeding at the anchors** (every other t): none; random restarts (4 or 8 per
  anchor); a small quantics TCI per anchor over the symmetry-reduced box, used as a
  **global sampler** (its best distinct evaluated points are polished, the TT itself is
  not trusted); or one p = 2 TCI over (angles, t), whose slice minima at the anchors are
  polished.

Random-restart rows marked ± were run with 5 library seeds (mean ± sd, range); TCI rows are deterministic. OK is against the best found by any run.

| p = 1 TT candidates | library seeding | OK / 192 | worst miss | runs / instance |
|---|---|---|---|---|
| — | none | 52 | 0.813 | 13,997 |
| — | 4 random restarts / anchor | 171.6 ± 18.7 (143–190) | 0.003–0.049 | 21,500 |
| — | 8 random restarts / anchor | 181.8 ± 7.3 (168–190) | 0.003–0.018 | 29,873 |
| — | TCI sampler / anchor (χ = 8) | 184 | 0.011 | 31,942 |
| — | TCI sampler / anchor (χ = 5) | 176 | 0.049 | 25,999 |
| — | one p = 2 TCI sweep (χ = 15) | 162 | 0.024 | 30,834 |
| TT | none | 169 | 0.049 | 26,482 |
| TT | 4 random restarts / anchor | 180.6 ± 9.2 (164–190) | 0.003–0.049 | 35,015 |
| TT | 8 random restarts / anchor | 184 (1 seed) | 0.011 | 43,460 |
| TT | TCI sampler / anchor (χ = 8) | 184 | 0.011 | 45,533 |
| TT | TCI sampler / anchor (χ = 5) | 176 | 0.049 | 39,640 |
| TT | one p = 2 TCI sweep (χ = 15) | 185 | 0.011 | 44,472 |

Each run also records which kind of start produced each final optimum
(`sweep_library(...)["source"]`), in `results/tci_placement.json`.

Takeaways

- **The TCI sampler is as good as an average run of 8 random restarts, and
  reproducible.** With χ = 8 it gets **184/192, worst miss 0.011**, on every run (full
  pivoting makes it deterministic). Random restarts at about the same cost (8 per
  anchor) average 181.8 ± 7.3 (168–190); with 4 per anchor they average 171.6 ± 18.7
  and one seed drops to 143. The luckiest restart seeds reach 190, so TCI does not find
  more than the best restarts. What it buys is no bad runs. 5 of its 8 misses are one
  *field* instance's late-t basin, which the first 8-restart seed misses too.
  (With rook pivoting the sampler scored 188–189, but rook results depend on the run
  order; see the reproducibility note in benchmark 9.)
- **Too small a TCI loses its edge.** At χ = 5, TCI seeding gets 176, below the
  8-restart mean. TCI needs enough rank to explore.
- **The p = 1 sweep TT matters most when little else seeds.** Without seeding it lifts
  the score from 52 to 169/192. With 4 restarts it raises the mean from 171.6 ± 18.7 to
  180.6 ± 9.2 and the worst seed from 143 to 164, for 13.5k extra runs. With 8
  restarts or TCI seeding it adds nothing.
- **A p = 2 TCI sweep is a weak seeder on its own** (162/192), but with the p = 1 TT
  it reaches 185, the best deterministic score here. A p = 2 landscape over (angles, t)
  is too high rank for TCI to approximate with χ = 15; only its sampled points help.
- **Recommended p = 2 pipeline:** `sweep_library(None, family, ts, p)` (TCI seeding is
  the default: 184/192, deterministic), or with the p = 1 TT plus 4 restarts for a
  better mean at more variance. At p = 1, keep the TCI sweep TT (`solve_from_sweep`):
  it is the only reuse method with no misses (benchmark 5).

**At p = 3** (`--p 3`; `results/tci_placement_p3.json`, `results/tci_placement_seeds_p3.json`).
Every row is scored against one common best: every configuration and every library seed.

| p = 1 TT candidates | library seeding | OK / 192 | worst miss | runs / instance |
|---|---|---|---|---|
| — | none | 15 | 0.433 | 20,084 |
| — | 4 random restarts / anchor | 116.6 ± 48.8 (22–154) | 0.030–0.268 | 29,344 |
| — | 8 random restarts / anchor | 148.6 ± 10.4 (133–161) | 0.017–0.139 | 38,674 |
| — | TCI sampler / anchor (χ = 8) | 146 | 0.050 | 46,214 |
| — | TCI sampler / anchor (χ = 5) | 146 | 0.050 | 36,991 |
| — | one p = 3 TCI sweep (χ = 15) | 45 | 0.268 | 42,842 |
| TT | none | 147 | 0.050 | 32,488 |
| TT | 4 random restarts / anchor | 150.0 ± 3.1 (147–155) | 0.024–0.050 | 44,415 |
| TT | 8 random restarts / anchor | 149 (1 seed) | 0.050 | 54,477 |
| TT | TCI sampler / anchor (χ = 8) | 147 | 0.050 | 62,041 |
| TT | TCI sampler / anchor (χ = 5) | 147 | 0.050 | 52,846 |
| TT | one p = 3 TCI sweep (χ = 15) | 151 | 0.050 | 57,882 |
| ref | 20 random restarts at every t | 48 (own best) | 0.251 | 95,138 |

Takeaways at p = 3

- **The p = 2 recipe no longer finds every optimum.** The best configurations reach
  146–155 of 192. Misses are spread over *field* and *penalty* (27–32 of 48 there),
  while *mix* and *risk* stay at 42–46.
- **TCI's role flips.** At p = 2 the TCI sampler was the best seeder and the p = 1 TT
  mattered mostly without seeding. At p = 3 the TCI sampler (146) is level with 8
  restarts (148.6 ± 10.4), not better, and costs 20% more. The **p = 1 sweep TT** is
  now the useful part: alone it lifts *none* from 15 to 147, and with 4 restarts it
  removes the variance (150.0 ± 3.1, range 147–155, vs 116.6 ± 48.8, range 22–154).
  The lifted depth-1 basins are a reliable floor; random restarts alone sometimes
  miss a family-wide basin and then miss it at every t.
- **Best value for money at p = 3:** `sweep_library(tt, family, ts, 3, seeding="restarts")`
  (TT + 4 restarts, 44k runs): as good as anything tested and the most stable. The
  luckiest restart seeds (154–161) show that more of the gap can be closed with more
  seeding, so the limit is budget, not a basin that cannot be reached.
- **A depth-p TCI sweep is again the worst seeder** (45 without the TT; 151 with it,
  where the TT candidates do the work).

**Chaining the depths** (`python benchmarks/bench_depth_chain.py`;
`results/depth_chain_p3.json`). The p = 1 TT helped at p = 3 through the INTERP lift,
so the same lift was applied one level up. Solve the family at p = 2 with
`sweep_library`, then at p = 3 with `sweep_library(..., lower=p2)`. At each t the p = 2
optimum, lifted, is one more start, and the whole p = 2 library, lifted, joins the p = 3
library. Costs include every stage (TT build, p = 2 run, p = 3 run). Library seeds 1–3,
scored against one common best over benchmark 6, 6b and these runs. That best is lower
than before, because the chain found better optima on *penalty* (by up to 0.016) than
any of the ~30 earlier runs. Rescored, the benchmark-6 configurations drop to 137–139.

| pipeline | OK / 192 per library seed | mean | runs / instance |
|---|---|---|---|
| no chain: p = 3 with TT + 4 restarts | 138, 138, 142 | 139.3 | 44,464 |
| chain: p = 2 (TT + 4 restarts) → p = 3 (TT + 4 restarts + lift) | 152, 145, 143 | 146.7 | 83,105 |
| chain: p = 2 (TT + 4 restarts) → p = 3 (TT + lift) | 159, 155, 138 | 150.7 | 72,942 |
| **chain: p = 2 (TT + 4 restarts) → p = 3 (lift only)** | **158, 155, 138** | **150.3** | **59,077** |
| chain: p = 2 (TT + TCI seeding) → p = 3 (TT + lift) | 158 (deterministic) | 158 | 83,050 |

Per family (library seed 1), no chain vs chain: *mix* 46 → 46, *field* 29 → 29,
*penalty* 20 → 42, *risk* 43 → 41–45 (of 48).

- **The lift from p = 2 is the most useful step at p = 3:** +11 on average, +22 on
  *penalty*, where the p = 3 optimum is the lifted p = 2 one and random p = 3 restarts
  rarely find it. 131 of the 576 final optima (3 seeds) came directly from a lifted
  start; most of the rest came from optima carried along t (sweep 218, prev 115,
  back 112).
- **Once the chain is there, the p = 3 stage needs nothing else.** The p = 1 TT
  candidates at p = 3 add 1 point in one of three runs for 14k more runs, and adding
  p = 3 restarts *lowers* the score (146.7 vs 150.7): the extra library entries push the
  lifted ones out of the 3-member shortlist.
- **The variance now comes from the p = 2 stage** (138–159 over library seeds). With
  TCI seeding at p = 2 the chain is deterministic and gets the best score, 158.
- **What is left: 34 misses**, 19 of them on *field* at large t (t ≥ 0.6), where the
  optimum is a depth-3 basin with no p = 2 counterpart. Only 1–3 of ~40 runs found it,
  all by lucky p = 3 restarts. Lifting cannot reach these; they need wider p = 3
  seeding at the far end of the sweep.
- **Recommended p = 3 pipeline:** `p2 = sweep_library(tt, family, ts, 2, seeding="tci")`,
  then `sweep_library(None, family, ts, 3, seeding=None, lower=p2)`.

**Extra p = 3 seeding on *field*** (`python benchmarks/bench_seed_reserve.py --families field`;
`results/seed_reserve_p3_field.json`). Same TCI-seeded p = 2 run per instance; p = 3
seeding added on top of the chain. With `n_seed_lib=2`, fresh seeds get 2 reserved
shortlist slots, so they cannot push lifted basins out (the failure of "+restarts" in
the chain table). "wide" samples the later γ's in [−4.5, 4.5] instead of [−3, 3].

| p = 3 seeding on top of the chain | OK / 48 per library seed | mean | runs / instance |
|---|---|---|---|
| none (chain) | 29 | 29.0 | 71,537 |
| 4 restarts / anchor, competing for the shortlist | 29, 29, 29 | 29.0 | 81,377 |
| 4 restarts / anchor, 2 reserved slots | 29, 29, 29 | 29.0 | 91,688 |
| 8 restarts / anchor, 2 reserved slots | 29, 36, 39 | 34.7 | 101,490 |
| TCI sampler / anchor, 2 reserved slots | 29 | 29.0 | 112,813 |
| 4 restarts / anchor, reserved, wide box | 33, 34, 29 | 32.0 | 91,179 |
| **8 restarts / anchor, reserved, wide box** | **31, 39, 40** | **36.7** | **101,445** |
| TCI sampler / anchor, reserved, wide box | 31 | 31.0 | 112,606 |

- **The missing *field* basin is a needle.** Its later γ's sit at −3.2 to −4.1, at or
  beyond the edge of the usual box, and it is optimal only for t ≥ 0.6. At one such t,
  at most 1 of 60 random polished starts reaches it. Once found at any t it carries
  to the rest of the sweep, which is why a run gets either all of that block or none.
- **More random starts in a wider box are what help:** 29 → 36.7 of 48 on average
  (best runs 39–40; two runs get all 16/16 on one instance), for 40% more circuit runs.
  Still luck: the per-run range is 31–40.
- **The TCI sampler barely finds it** at χ = 8 on a 16-point grid: 29 in the usual box,
  31 in the wide box (2 extra points on one instance). In a
  single-anchor probe, a finer grid (32 points per angle) found it once in 16
  candidates at ~9k runs: the basin is narrower than the grid spacing TCI samples.
  TCI is a good sampler for broad basins (p = 2) and not for needles.
- **Reserved slots alone change nothing** (29 → 29.0): the problem was finding the
  basin, not keeping it in the shortlist. The option is kept (`n_seed_lib`, default 0)
  because it is needed once seeding is added to the chain.

**The same seeding on the other three families** (`--families mix penalty risk`;
`results/seed_reserve_p3_others.json`). It does no harm, and it does little good: the
chain gets 128/144 there, and the seeded variants 128–132 (mean 128.7–130.0), for
28–43% more runs. No run improved the earlier best on these families by more than
0.0003. Over all four families (192 points, scored against the common best):

| p = 3 pipeline | OK / 192 per library seed | mean | runs / instance |
|---|---|---|---|
| chain (p = 2 with TCI seeding → p = 3 lift) | 157 | 157.0 | ~69k |
| + 4 wide restarts / anchor, reserved slots | 161, 164, 159 | 161.3 | ~88k |
| **+ 8 wide restarts / anchor, reserved slots** | **159, 169, 168** | **165.3** | **~98k** |

Almost all of the gain is on *field*. With 8 wide restarts, the remaining misses are
*field* (8–17 per run, the needle), *penalty* 10, *risk* 4 and *mix* 0–2. On those
last three, each missed optimum was found by only some of the ~40 earlier runs.

- **Recommended pipeline: `sweep_chain(family, ts, p)`.** It runs the depth-1 TCI sweep TT,
  then p = 2 with TCI seeding, then each deeper level lifted from the one below with
  8 wide restarts per anchor and 2 reserved slots (165.3/192 on average at p = 3).
  `extra_restarts=0` gives the deterministic chain (157/192) for ~40% fewer runs. Since
  benchmark 9 it also adds INTERP's optimum as a start at the final depth
  (`interp_start=True`, ~20k runs); without it, it reproduces the benchmark-6d runs.
  `p = 1` falls back to the sweep TT alone.

  ```python
  from tciqaoa.sweep import SweepFamily, sweep_chain
  fam = SweepFamily(lambda t: Q0 + t * Q1)        # any t -> QUBO family
  res = sweep_chain(fam, ts=np.arange(16) / 16, p=3)
  res["costs"], res["angles"], res["n_evals"]     # per t, plus total circuit runs
  ```

<a id="p--4"></a>**At p = 4: benchmark 8** (`python benchmarks/bench_p4.py`; `results/p4.json`).
`sweep_chain` on the same 12 instances × 16 values of t, against INTERP from scratch at
every t and 20 random restarts at every t. The chain charges every stage (TT, p = 2, 3
and 4). OK is against the best found in this benchmark only; at p = 3, adding more runs
lowered the best and the scores by about 10, so these are upper bounds. Rescored against
the no-TCI seed check ([ablation](#ablation-p4)) they are 166 and 168, 172, 186 (175.3),
INTERP 104, restarts 4.

| method | OK / 192 per library seed | mean | worst | runs / instance |
|---|---|---|---|---|
| `sweep_chain(..., extra_restarts=0)` | 169 | 169.0 | 0.048 | 94,357 |
| **`sweep_chain(...)` (8 wide restarts, reserved slots)** | **171, 175, 188** | **178.0** | 0.048 | 154,501 |
| INTERP at every t | 104 | 104.0 | 0.048 | 21,297 |
| 20 random restarts at every t | 4 | 4.0 | 0.271 | 95,934 |

| family | chain | chain + 8 restarts | INTERP | 20 restarts | mean gain p = 3 → 4 |
|---|---|---|---|---|---|
| mix | 40 | 40, 40, 48 | 40 | 2 | −0.18 |
| field | 34 | 36, 40, 44 | 32 | 2 | −0.15 |
| penalty | 47 | 47, 47, 48 | 13 | 0 | −0.05 |
| risk | 48 | 48, 48, 48 | 19 | 0 | −0.06 |

- **The chain scales to p = 4.** It finds 169–188 of 192 optima, and on the portfolio
  families 95–96 of 96, where INTERP finds 32. Depth still pays: the p = 4 optimum is
  0.05–0.26 below p = 3 in normalised cost.
- **Random restarts are useless at p = 4:** 4/192 at the same cost as the chain. The
  8-dimensional box is too big to sample. Every useful start comes from lifting.
- **On *mix* and *field*, INTERP is nearly as good at a fifth of the cost** (72 vs 74
  of 96). There the per-t lift from p = 1 already finds the basins. The chain's edge is
  on the portfolio families, where the optimum at one t is best reached from the
  library carried along the sweep.
- **What is left is again rare basins.** On one *mix* instance a basin found by a single
  restart run lifts 8 points; on *field* the extra restarts add 2–10 points per run.
- **Cost.** The deterministic chain costs 94k runs per instance (5.9k per t), 4.4×
  INTERP. With the extra restarts it is 155k.

<a id="larger-n"></a>**Larger n: benchmark 9** (`python benchmarks/bench_large_n.py --n 12 --p 3 --ref --ablate`;
`results/large_n{10,12,14}_p3_v2.json`). The four families at n = 10, 12 and 14 (budget
k = round(0.4 n) in the portfolio families), p = 3, 3 instances each. A circuit run costs
~3× more at n = 12 than at n = 10 and ~10× more at n = 14, so n = 14 compares only the
chain and INTERP, and n = 12 runs one library seed for "chain + 8". OK is against the
best found in this run and in the first version of this benchmark (`results/large_n{12,14}_p3.json`).

This is the second version. The first showed two weaknesses in `sweep_chain`, now fixed:
(1) the extra restarts could make it worse: optima reached *from* restart seeds joined
the library and pushed the lifted basins out of the shortlist (16/16 → 4/16 on one
instance); now they compete only for the reserved slots, and the non-seed line is
carried along t as well; (2) the chain could carry a basin past the point where it
stops being best (12 late-t points lost to INTERP at n = 14); now INTERP's optimum at
each t is one more start at the final depth.

| n | method | OK / 192 | worst | runs / instance | mix | field | penalty | risk |
|---|---|---|---|---|---|---|---|---|
| 10 | `sweep_chain(..., extra_restarts=0)` | 171 | 0.050 | 89,501 | 46 | 29 | 48 | 48 |
| 10 | the same without the INTERP start | 169 | 0.050 | 69,209 | 46 | 29 | 48 | 46 |
| 10 | **`sweep_chain(...)` (8 wide restarts), 3 seeds** | **173, 185, 182 (180.0)** | 0.050 | 118,950 | 46 | 31 | 48 | 48 |
| 10 | INTERP at every t | 99 | 0.050 | 16,564 | 46 | 29 | 12 | 12 |
| 10 | 20 random restarts at every t | 47 | 0.251 | 95,138 | 34 | 12 | 1 | 0 |
| 12 | `sweep_chain(..., extra_restarts=0)` | 146 | 0.358 | 93,352 | 45 | 31 | 22 | 48 |
| 12 | the same without the INTERP start | 145 | 0.358 | 71,002 | 45 | 31 | 21 | 48 |
| 12 | **`sweep_chain(...)` (8 wide restarts)** | **152** | 0.037 | 122,790 | 48 | 34 | 22 | 48 |
| 12 | INTERP at every t | 87 | 0.358 | 18,698 | 45 | 31 | 7 | 4 |
| 12 | 20 random restarts at every t | 48 | 0.169 | 95,408 | 29 | 18 | 1 | 0 |
| 14 | **`sweep_chain(..., extra_restarts=0)`** | **189** | 0.009 | 101,682 | 48 | 48 | 48 | 45 |
| 14 | INTERP at every t | 103 | 0.060 | 20,910 | 48 | 44 | 7 | 4 |

- **The fixes work.** With restarts the chain is now (almost) never worse than without
  them: at most 0.003 at any t over all 48 instance-runs, against a loss of 12 points
  on one instance before. At n = 14 the fixed chain gets 189/192 (180 in the first
  version), worst miss 0.009; the INTERP start recovered the 12 late-t *penalty*
  points.
- **The INTERP start adds little elsewhere** (+1 to +2 points) for ~20k more runs
  (+25%). It is cheap insurance against the late-t failure, so it stays on by default.
- **The chain's lead over INTERP holds at every size,** and comes from the portfolio
  families (*penalty* + *risk*: 96 vs 24 at n = 10, 70 vs 11 at n = 12, 93 vs 11 at
  n = 14). On *mix* and *field* INTERP matches it at a fifth of the cost.
- **n = 12 *penalty*: 22/48, but the misses are small.** The best there comes from the
  first version of this benchmark; the chain now misses those late-t optima by at most
  0.006 (worst over the three instances: 0.003, 0.006, 0.004). See the reproducibility
  note below.
- **Cost** ~90–120k runs per instance, flat in n; the depth-1 TT stays at rank 10–21.

**p = 4 at larger n** (`python benchmarks/bench_large_n.py --n 12 --p 4 --ref` and
`--n 14 --p 4 --lib-seeds 0`, then `--n 14 --p 4 --ref --methods ref --prev
results/large_n14_p4.json` and `--methods chain+8 --lib-seed-from S --lib-seeds 1`
for S = 1, 2, 3; `results/large_n{12,14}_p4*.json`; at n = 14 the extra runs went per
family and seed, in parallel). Same instances, depth p = 4. Each chain goes p = 1 TT →
2 → 3 → 4. "chain + 8" uses one library seed at n = 12 and three at n = 14. OK is
against the best of all methods at that n, including the ablation, its seed check
and the matched-budget runs [below](#ablation-p4), which found better optima on
several instances, so the rows here are scored against them.

| n | method | OK / 192 | worst | runs / instance | mix | field | penalty | risk |
|---|---|---|---|---|---|---|---|---|
| 12 | `sweep_chain(..., extra_restarts=0)` | 140 | 0.159 | 125,216 | 45 | 32 | 19 | 44 |
| 12 | `sweep_chain(...)` (8 wide restarts) | 149 | 0.039 | 186,343 | 46 | 40 | 19 | 44 |
| 12 | INTERP at every t | 86 | 0.159 | 23,470 | 45 | 32 | 6 | 3 |
| 12 | 20 random restarts at every t | 6 | 0.306 | 95,942 | 3 | 3 | 0 | 0 |
| 14 | `sweep_chain(..., extra_restarts=0)` | 152 | 0.046 | 133,014 | 48 | 35 | 28 | 41 |
| 14 | **`sweep_chain(...)` (8 wide restarts), 3 seeds** | **162, 152, 166 (160.0)** | 0.043 | 194,647 | 48 | 46, 37, 45 | 28, 26, 32 | 40, 41, 41 |
| 14 | best of those 3 seeds | 174 | 0.023 | 583,941 | 48 | 48 | 35 | 43 |
| 14 | INTERP at every t | 102 | 0.066 | 25,701 | 48 | 34 | 6 | 14 |
| 14 | 20 random restarts at every t | 2 | 0.249 | 96,000 | 1 | 1 | 0 | 0 |

- **With the 8 wide restarts the chain finds 149/192 at n = 12 (one library seed) and
  160.0 at n = 14 (162, 152, 166 over three seeds).** (These were 192 and 173.0 before
  the no-TCI runs found better *penalty*, *risk* and *mix* basins; see below.) At n = 14 the seeds differ by up
  to 14 points: which late-t *field* basin and which *penalty* basin a seed's restarts
  reach is luck. Together the three seeds find 174, so more seeds do help, at ~195k
  runs each. INTERP finds 9 and 20
  of the 96 portfolio optima; the chain's lead over it is larger than at p = 3.
- **At n = 12 the deterministic chain misses 52 points:** *mix* (3, one instance,
  worst 0.159) and *field* (16, t ≥ 11/16), exactly where INTERP misses with
  the same costs, plus 29 *penalty* points (t ≥ 6/16, ≤ 0.039) and 4 *risk* points
  that only no-TCI runs found. The 8 wide restarts recover 1 *mix* and 8 of
  the *field* points, not the *penalty* ones.
- **At n = 14 the deterministic chain misses 40:** late-t *field* on all three
  instances (13, ≤ 0.046), *penalty* (20) and *risk* (7). Against INTERP and restarts
  alone it had scored 190/192; each stronger reference (one, then three chain+8 seeds,
  then the no-TCI seeds) exposed more misses. At n = 12 the same happened (174 → 140).
- **Random restarts collapse at p = 4:** 6/192 at n = 12 and 2/192 at n = 14 for 96k
  runs, against 48 at p = 3 (n = 12). The 8-angle landscape has too many local optima for 20 starts.
- **Cost** 125–135k runs per instance for the deterministic chain (+35% over p = 3),
  still flat from n = 12 to 14. The depth gain over p = 3 (mean best cost) is 0.17–0.18 on
  *mix*, 0.14 on *field*, 0.05–0.08 on the portfolio families.

<a id="ablation-p4"></a>**Does TCI help? Ablation over three seeds and four settings**
(`python benchmarks/bench_large_n.py --n N --p P --families F --methods chain-noTCI
chain+8-noTCI --lib-seed-from S --prev ...` for S = 1, 2, 3, and `--n 12 --p 4
--methods interp-x8 ref40`; `results/large_n12_p4_{noTCI,matched}_*.json`,
`results/notci_*.json`; scored by `python benchmarks/score_notci.py`). Same 12
instances × 16 t per setting, OK against the best of every run at that setting.

- `chain-noTCI`: `sweep_chain(..., extra_restarts=0, use_tci=False, seed=S)`: no
  depth-1 TT, and depth 2 seeded by 4 random restarts per anchor (seed S) instead of
  the TCI sampler. Depths 3–p are unchanged (lifting, INTERP start). `chain+8-noTCI`
  adds the 8 wide restarts, also with seed S.
- The TCI chain is deterministic (one value); "chain + 8" with TCI has three library
  seeds at n = 10 and 14 and one at n = 12.

| setting | chain (TCI) | chain, no TCI: seeds 1, 2, 3 (mean) | chain + 8 (TCI) | chain + 8, no TCI | best of 3 seeds, TCI / no TCI |
|---|---|---|---|---|---|
| n = 10, p = 3 | **165** | 167, 146, 140 (151.0) | 167, 176, 175 (**172.7**) | 168, 156, 151 (158.3) | 183 / **188** |
| n = 10, p = 4 | **166** | 161, 157, 144 (154.0) | 168, 172, 186 (**175.3**) | 167, 163, 164 (164.7) | 190 / **192** |
| n = 12, p = 4 | 140 | 175, 140, 143 (**152.7**) | 149 | 178, 144, 147 (**156.3**) | – / 190 |
| n = 14, p = 4 | **152** | 149, 148, 154 (150.3) | 162, 152, 166 (**160.0**) | 156, 150, 172 (159.3) | 174 / **188** |
| runs / instance, n = 12 | 125k | 102k | 186k | 163k | |

Matched budgets at n = 12, p = 4: INTERP with seeds 0..7 (best at each t) gets 91/192
for 201k runs, 40 random restarts at every t 26 for 192k; INTERP alone gets 86 and 20
restarts 6.

- **The first ablation was one lucky seed.** With seed 1 at n = 12 the no-TCI chain
  reached a lower *penalty* basin on all three instances (48/48 vs 19); seeds 2 and 3
  do not (19, 20), and score level with the TCI chain. Seed 1 hits the same basin at
  n = 10, p = 3 (48 vs 20–23 for seeds 2, 3).
- **Per run, TCI is worth a few points:** the TCI chain is ahead of the no-TCI mean by
  14 and 12 at n = 10 (p = 3, 4) and level at n = 14 (152 vs 150.3; 160.0 vs 159.3
  with restarts); only at n = 12 is the no-TCI mean ahead, by the seed-1 run. The TCI
  chain also needs no seed and has no bad runs (no-TCI ranges 140–175 at n = 12).
- **The no-TCI chain is cheaper** (18–26% for the deterministic chain, 12–20% with the
  8 restarts; level at n = 10, p = 4) and its seeds differ from depth 2 on, while the
  TCI seeds share the TCI depths 1–2. So **for several seeds no TCI wins:** three no-TCI
  seeds together get 188–192 everywhere, against 174–190 for three TCI seeds, for up
  to 20% fewer runs.
- **Either way the chain's lead comes from chaining the depths**, i.e. carrying the
  basin library along t and lifting it from depth to depth: INTERP gets 86–104 and 20
  random restarts 2–47 in the same settings, and the matched budgets (INTERP × 8, 40
  restarts) add little. `use_tci=True` stays the default for a single deterministic
  run; use `use_tci=False` with several seeds.

<a id="mix-p4"></a>**Mixing one TCI run with no-TCI seeds** (`python benchmarks/score_mix.py`; no
new runs). Independent runs combine by taking the best cost at each point, so a
portfolio of k runs is scored from the stored results above. Each row averages over
every choice of k runs of that kind (the range is in brackets); "mixed" is 1 TCI run
plus k − 1 no-TCI seeds. OK / 192 (48 for *sched*), same common best as above.

| setting | k = 2: mixed | k = 2: no TCI | k = 3: mixed | k = 3: no TCI | k = 3: TCI |
|---|---|---|---|---|---|
| *deterministic chain* | | | | | |
| n = 10, p = 3 | **168.0** (165–171) | 163.3 (148–173) | 171.0 (168–174) | **173** | – |
| n = 10, p = 4 | **167.3** (166–169) | 165.0 (162–169) | 168.3 (167–169) | **169** | – |
| n = 12, p = 4 | 156.0 (144–180) | **168.3** (146–180) | 170.0 (147–182) | **182** | – |
| n = 14, p = 4 | 161.3 (157–165) | **162.3** (160–167) | 167.3 (165–171) | **169** | – |
| sched, n = 12 | 5.7 (3–11) | **8.3** (3–11) | 8.3 (3–11) | **11** | – |
| *chain + 8* | | | | | |
| n = 10, p = 3 | **179.6** (173–186) | 176.0 (161–185) | 184.8 (181–189) | **188** | 183 |
| n = 10, p = 4 | 181.9 (169–190) | **183.0** (176–190) | 187.4 (176–192) | **192** | 190 |
| n = 12, p = 4 | 164.7 (154–184) | **174.7** (153–186) | 178.7 (159–189) | **190** | – |
| n = 14, p = 4 | 174.6 (156–183) | **178.3** (169–186) | 184.4 (173–191) | **188** | 174 |
| sched, n = 12 | 35.3 (30–44) | **38.0** (32–44) | 40.7 (32–46) | **46** | 30 |

- **Mixing does not pay.** With three runs, swapping one no-TCI seed for the TCI run
  lowers the score everywhere: by 1–12 points (deterministic) and 3–11 (chain + 8).
  With two runs the mix is ahead only at n = 10 (by 2–5), and behind at n = 12, 14 and
  on *sched*.
- **The TCI run is the best single run at n = 10, but as a second or third run it
  adds less than another no-TCI seed:** its basins overlap those the no-TCI seeds
  already find, while a fresh seed reaches new ones (the *penalty* basin at n = 12,
  the lower *sched* basin). It is also the costlier run.
- **Caveat:** the common best is defined by these same runs, and there are only three
  no-TCI seeds, so the k = 3 no-TCI score has one choice and no spread.
- **So: one run, use TCI (deterministic, no bad seed); two or more runs, use only
  no-TCI seeds.** Merging the two inside one run is tested next.

<a id="hybrid-p4"></a>**One merged run: TCI and restarts in the same library**
(`sweep_chain(..., use_tci="hybrid", seed=S)`; `python benchmarks/bench_large_n.py ...
--methods chain-hyb chain+8-hyb --lib-seed-from S`; `results/hyb_*.json`). The depth-1
TT starts as in the TCI chain, and depth 2 is seeded by both the TCI sampler and the
same 4 random restarts per anchor as the no-TCI seed S (`seeding="tci+restarts"`), in
one shared library. So hybrid seed S is the no-TCI seed S plus everything TCI adds.
Seeds 1–3 at the same settings; OK / 192 (48 for *sched*), mean over seeds.

| setting | chain: TCI | chain: hybrid | chain: no TCI | chain + 8: TCI | chain + 8: hybrid | chain + 8: no TCI |
|---|---|---|---|---|---|---|
| n = 10, p = 3 | 164 | **164.7** (168, 167, 159) | 151.0 | **171.7** | 170.0 (170, 175, 165) | 158.3 |
| n = 10, p = 4 | **166** | 164.3 (162, 166, 165) | 154.0 | **175.3** | 172.0 (164, 172, 180) | 164.7 |
| n = 12, p = 4 | 140 | 142.0 (143, 142, 141) | **152.7** | 149 | 147.3 (148, 145, 149) | **156.3** |
| n = 14, p = 4 | **151** | 144.3 (139, 146, 148) | 150.3 | 159.0 | 155.7 (149, 148, 170) | **159.3** |
| sched, n = 12 | 3 | 4.0 (6, 3, 3) | **5.3** | **30** | 24.0 (29, 7, 36) | 24.3 |
| runs / instance, n = 12 | 125k | 135k | 102k | 186k | 196k | 163k |
| best of 3 seeds, n = 14 | – | 156 | **169** | 173 | 182 | **188** |

- **Merging does not combine the strengths.** The hybrid is about as good as the TCI
  run on average (−7 to +2 points per setting), not better, and is the costliest
  (+9–10k runs over TCI, +35k at n = 10, p = 4; +33k over no TCI at n = 12).
- **Adding the TCI candidates can lose a no-TCI basin.** Hybrid seed 1 at n = 12 has
  the same restarts that found the lower *penalty* basin (48/48) without TCI, but gets
  19/48: with the TT and TCI candidates in the library the sweep takes a different
  path through the shortlist (3 slots per t) and the lifted depths follow it.
- **As seeds, hybrid runs are less diverse than no-TCI ones**: three hybrid seeds
  together get 149–191 against 169–192 for three no-TCI seeds, for 20–50% more runs.
- The hybrid runs found a few lower optima (*risk*, n = 10, p = 3 and n = 14), which
  lowers the TCI rows there by 1 point; the tables above are scored before them.
- **So TCI and random restarts do not add up inside one chain.** The recommendation
  stands: one run, TCI; two or more, no-TCI seeds.

<a id="sched-p4"></a>**Job scheduling at p = 4** (`python benchmarks/bench_large_n.py --n 12 --p 4
--families sched --ref --methods ...`; `results/sched_n12_p4_{tci,noTCI,base}.json`).
The `sched` family: 4 jobs × 3 machines (n = 12), job times U(2, 10) as in hybridq's
example, objective Σⱼ loadⱼ² normalised to 1 for a perfectly balanced schedule, and
the one-machine-per-job penalty with weight (0.2 + t) × the largest marginal cost of
one job. The exact ground state leaves 1–2 jobs unassigned at t < 0.3 and is a full,
balanced assignment from there on. 3 instances × 16 t; OK is against the best of all
six methods.

| method | OK / 48 | per instance | worst | runs / instance |
|---|---|---|---|---|
| chain (with TCI) | 3 | 3, 0, 0 | 0.036 | 134,836 |
| chain without TCI, seeds 1, 2, 3 | 10, 3, 3 | 2/5/3, 3/0/0, 3/0/0 | 0.021–0.033 | 104,667 |
| chain + 8 (with TCI) | 30 | 2, 14, 14 | 0.021 | 201,677 |
| chain + 8 without TCI, seeds 1, 2, 3 | 30, 7, 36 | 2/14/14, 2/5/0, 16/9/11 | 0.007–0.025 | 171,626 |
| **best of the 3 no-TCI "chain + 8" seeds** | **46** | 16, 15, 15 | 0.006 | 514,051 |
| INTERP at every t | 0 | 0, 0, 0 | 0.167 | 21,470 |
| 20 random restarts at every t | 0 | 0, 0, 0 | 0.144 | 96,000 |

- **The extra restarts are essential here.** The deterministic chains lock the sweep
  into a worse basin on most instances (3–10/48, up to 0.036 worse). With the 8 wide
  restarts they reach 30/48 with TCI, and 7–36 without depending on the seed.
- **Seed 3 found a lower basin on instance 0** (t ≥ 3/16, up to 0.020 lower) that every
  other run missed, which is why the first reported scores (43/48) dropped to 30.
  The three no-TCI seeds together get 46/48.
- **TCI is not clearly ahead on top of the restarts** (30 vs a 24.3 no-TCI mean, but the
  seed spread is 7–36), and costs 30k more runs per instance.
- **INTERP and random restarts fail on every point,** by up to 0.17 and 0.14.

**Reproducibility note.** xfacpy's rook pivot search draws from an internal random
generator that is not reset between calls, so a TCI build with rook pivoting depends on
everything that ran before it in the process. On one instance the same depth-1 sweep TT
came out at rank 4 (196 runs, stopped after one sweep) or rank 11–17 (1.1–1.5k runs).
Every TCI build in this repo (`optimize_tci`, `tci_sweep`, the TCI sampler,
`tci_minimize_bits`, `tci_sweep_bits`) now uses full pivoting by default
(`full_pivot=True`): deterministic, at about the same cost (+0–15% runs). With it,
`sweep_chain(..., extra_restarts=0)` is fully deterministic, and **all benchmarks above
(1–9) were rerun with full pivoting**. Compared with the earlier rook runs, the
conclusions held, but some counts moved:

- p = 2 TCI sampler (benchmark 6): 188–189 → 184/192. It is now level with an average
  8-restart run instead of above the best ones.
- p = 1 TT + 4 restarts at p = 2: 171.8 ± 18.9 → 180.6 ± 9.2. The TT now helps there too.
- One p = 2 TCI sweep as seeder: 146/153 → 162/185; at p = 3 without the TT: 93 → 45.
- Frontier (benchmark 7), TCI-sweep at n = 20: 40 → 28/48 optima.
- Reuse (benchmark 4), p = 2 MaxCut → QUBO: the point method went from 50/50 to 0/50
  because the training TT returned the other symmetry image of the same optimum.
- Everything else (chain, p = 3 seeding, p = 4, QAOA and solver tables) moved by 0–3
  points.

Where TCI pays, across all benchmarks:

| use of TCI | verdict |
|---|---|
| p = 1 angle landscape, one instance (`optimize_tci`) | works, but INTERP is as good and cheaper |
| p ≥ 2 angle landscape (TT surrogate) | does not converge: rank ≈ 75 |
| TT reuse on new instances | works, but storing the angles and their β images is 3–4× cheaper and as reliable |
| **p = 1 sweep TT over (angles × parameter)** | **best p = 1 sweep method: no misses** |
| TCI as a global sampler seeding a p = 2 basin library | deterministic 184/192, level with an average 8-restart run (181.8 ± 7.3), no bad runs |
| **p = 1 sweep TT lifted into a p = 2 / p = 3 basin library** | **cuts the run-to-run variance: 180.6 ± 9.2 vs 171.6 ± 18.7 at p = 2, 150.0 ± 3.1 vs 116.6 ± 48.8 at p = 3 (4 restarts)** |
| TCI as a global sampler at p = 3 | level with 8 random restarts, 20% costlier |
| **depth chain p = 1 TT → p = 2 library (TCI seeding) → p = 3 → p = 4 (`sweep_chain`)** | **best p = 3 and p = 4 pipeline: 157–158/192 at p = 3, 169/192 at p = 4, deterministic; 95/96 on the portfolio families at p = 4 vs 32 for INTERP** |
| the chain under shot noise (256–16384 shots per circuit, p = 2, 3) | **most robust method, with COBYLA or SPSA: 0.003–0.014 mean gap with SPSA vs 0.037–0.14 for INTERP and restarts with SPSA; TCI's own part: a small gain at p = 2 with COBYLA, none with SPSA** |
| the p = 1 sweep TT on denser sweeps (8 → 64 points of t) | **pays off: same answers as INTERP, cost per point 563 → 291 vs ~530, 1.8× cheaper at 64 points** |
| the chain on denser sweeps | **more accurate with density (p = 2: 87.5% → 99.2%; p = 3: 76% → 88%); seeded at 8 points only, the no-TCI chain gets 98.7% at p = 2 for INTERP's cost per point (47.5%)** |
| one TT over two sweep parameters (p = 1, up to 16 × 16 points) | **low rank (≤ 29); 1.8× cheaper per point than INTERP at 256 points, same answers** |
| TCI seeding in a two-parameter chain (p = 2) | **helps on sparse grids or sparse seeding (97.9 vs 91.9% at 8 × 8; 99.0 vs 95.4% with 16 seeding points); level on the dense grid; none at p = 3** |
| one TT over three sweep parameters (p = 1, up to 8 × 8 × 8 points) | **1.9× cheaper per point than INTERP at 512 points, same answers with the neighbour warm start (8 of 3,456 points missed without it); up to 23k runs (rank 30) to build on the harder family** |
| TCI seeding in a three-parameter chain (p = 2) | **matters most with sparse seeding: 98.9 vs 88.2% seeded at 16 of 512 points; level with full seeding; behind at p = 3 (93.3 vs 97.0%)** |
| the p = 1 sweep TT under shot noise | worst p = 1 method with COBYLA (the TT costs 2× more and is less accurate); level with INTERP and restarts with SPSA |
| TCI inside that chain (depth-1 TT + TCI sampler at p = 2), ablated over 3 seeds at n = 10 (p = 3, 4), 12 and 14 (p = 4) | **a few points per run: the no-TCI mean is 12–14 lower at n = 10, level at n = 14, 7–13 higher at n = 12 (one lucky seed); no TCI is 12–26% cheaper and 3 no-TCI seeds together beat 3 TCI seeds (188–192 vs 174–190). The chain's lead over INTERP and restarts comes mostly from chaining** |
| p = 2 / p = 3 sweep TT over (angles × parameter) | not low rank; weak seeder on its own |
| TCI-bits on the QUBO itself (`tci_minimize_bits`): MaxCut, random QUBO, portfolio | beats hybridq; loses to time-matched local search |
| TCI-bits over (assets × risk aversion): one TT for a whole efficient frontier | 6× cheaper and better than per-λ TCI; still loses to time-matched local search |

### 7. TCI on the objective for a whole efficient frontier

`HYBRIDQ_PATH=... python benchmarks/bench_frontier.py` (3 portfolios per size, each
solved at 16 risk-aversion values λ = 0.1 … 5, log spaced; hybridq's frontier range).

No QAOA here: every method minimises the portfolio cost E_λ(x) directly. This is the
reuse idea of benchmark 5 applied to the objective itself. `tci_sweep_bits` builds
**one** TCI over (λ bits, asset bits): F(λ, x) = exp(−β (E_λ(x) − E_ref(λ)) / unit(λ)),
each λ slice normalised by its own best energy and energy scale, with staged β as in
`tci_minimize_bits`. After each stage every λ polishes its lowest sampled portfolios
plus its neighbours' best, and at the end every λ is tried from every other λ's best.

- **TCI-bits/λ**: `tci_minimize_bits` from scratch at every λ.
- **TCI-sweep**: one `tci_sweep_bits` for the whole frontier.
- **SA/λ**: hybridq's simulated annealing at every λ.
- **LS/λ**: 100 random-restart steepest descents at every λ.
- **LS-cont**: steepest descent from the previous λ's best plus 10 random starts.
- **LS-matched**: random-restart steepest descent with TCI-sweep's wall time.

| n | TCI-bits/λ | TCI-sweep | SA/λ | LS/λ | LS-cont | LS-matched |
|---|---|---|---|---|---|---|
| 20 (exact) | 20/48 · 595,814 · 3.2 s | 28/48 · 92,431 · 0.7 s | 0/48 · 401,600 · 0.8 s | 27/48 · 11,250 · 0.0 s | 26/48 · 1,155 · 0.0 s | 46/48 · 198,417 · 0.7 s |
| 40 | 0/48 · 3,057,772 · 22.7 s | 0/48 · 309,691 · 3.4 s | 0/48 · 403,200 · 0.9 s | 1/48 · 19,270 · 0.1 s | 0/48 · 1,940 · 0.0 s | 47/48 · 883,567 · 3.4 s |
| 60 | 0/48 · 5,344,592 · 46.0 s | 1/48 · 473,663 · 6.1 s | 0/48 · 404,800 · 0.8 s | 3/48 · 27,261 · 0.1 s | 1/48 · 2,731 · 0.0 s | 43/48 · 1,604,600 · 6.1 s |

Cells: optimum hits (of 48) · energy evaluations per frontier · seconds per frontier.
Reference: brute force at n = 20, best found by any method above.

Takeaways

- **Reuse works inside TCI.** One TCI for the whole frontier beats 16 separate ones on
  every count: at n = 20 it finds 28/48 optima against 20/48, with 6× fewer
  evaluations and 5× less time. Neighbouring λ share their good portfolios through the
  tensor train and the cross-λ polish. (With rook pivoting the sweep found 40/48; its
  result depends strongly on which pivots the search happens to visit.)
- **But TCI on the objective still loses to plain local search.** Given the same wall
  time, random-restart steepest descent finds 46, 47 and 43 of 48 optima at n = 20,
  40 and 60. TCI-sweep finds 28, 0 and 1. The TT rank is at the cap: a dense
  covariance plus an all-to-all budget penalty is not a low-rank Boltzmann tensor.
- **Where this leaves TCI on the objective:** useful only where E(x) has low-rank
  structure (chains, lattices, sparse constraints; next steps), not on dense
  portfolio QUBOs. On QAOA *angles* (benchmarks 5–6) the picture is different, because
  that landscape is smooth and low-dimensional.

### 8. Bugs found in hybridq

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

**Squared return in `solve_portfolio`.** hybridq builds the portfolio QUBO as
`risk_aversion * sigma - np.outer(mu_norm, mu_norm)`, i.e. it rewards (μᵀx)², the
*squared* expected return, not the Markowitz return μᵀx. For binary x the linear return
is the diagonal `-np.diag(mu_norm)` (xᵢ² = xᵢ). This repo's `portfolio_qubo` uses the
linear form, and benchmarks 5–7 were rerun with it.

### 9. Shot noise

`python benchmarks/bench_shots.py --p P --families F --shots 0 256 1024 4096 16384`
for P = 1, 2, 3; `results/shots_n10_p*.json`; scored by `python benchmarks/score_shots.py`.

On a device ⟨C⟩ is not known exactly: it is the mean of C over a finite number of
measured bitstrings. `QAOA(..., shots=S)` (and `SweepFamily(..., shots=S)`) returns
exactly that estimate, one multinomial draw from |ψ|² per circuit, so every optimiser
sees the noise: the sweep TT, the TCI sampler and COBYLA in every method. (At 1024 shots
the noise has standard deviation ≈ 0.03 in the normalised cost.) The angles each
method returns are then scored on the *exact* ⟨C⟩, against the best exact cost any
method found at any noise level, and by the ground-state probability P(gs) they give.
Same 12 instances × 16 t as benchmarks 5–8 (n = 10); one noise realisation.

Mean exact excess over the best ("gap"), runs per instance, and P(gs):

| p | method | exact | 256 shots | 1024 | 4096 | 16384 | runs (1024) | P(gs) at 1024 / exact |
|---|---|---|---|---|---|---|---|---|
| 1 | sweep TT | 0.000 | 0.041 | 0.017 | 0.008 | 0.005 | 12,236 | 0.016 / 0.017 |
| 1 | INTERP (5 restarts) | 0.000 | 0.027 | 0.015 | 0.009 | 0.005 | 2,232 | 0.016 / 0.017 |
| 1 | 20 restarts | 0.000 | **0.017** | **0.010** | **0.005** | **0.003** | 9,065 | 0.016 / 0.017 |
| 2 | **chain (TCI)** | 0.000 | **0.039** | **0.022** | **0.014** | **0.010** | 25,633 | **0.045** / 0.048 |
| 2 | chain, no TCI | 0.000 | 0.089 | 0.056 | 0.032 | 0.017 | 4,107 | 0.042 / 0.048 |
| 2 | INTERP | 0.009 | 0.180 | 0.145 | 0.113 | 0.090 | 2,864 | 0.035 / 0.046 |
| 2 | 20 restarts | 0.006 | 0.113 | 0.087 | 0.061 | 0.047 | 14,268 | 0.040 / 0.047 |
| 3 | **chain (TCI)** | 0.001 | **0.062** | **0.036** | **0.024** | **0.014** | 34,288 | **0.090** / 0.094 |
| 3 | chain, no TCI | 0.001 | 0.091 | 0.051 | 0.033 | 0.020 | 12,844 | 0.085 / 0.094 |
| 3 | INTERP | 0.009 | 0.188 | 0.134 | 0.098 | 0.070 | 3,724 | 0.076 / 0.093 |
| 3 | 20 restarts | 0.033 | 0.263 | 0.199 | 0.156 | 0.128 | 19,926 | 0.055 / 0.085 |

The methods spend very different numbers of circuits, so the fair comparison is at
equal *total shots* (runs × shots per run), per instance:

| p | budget | chain (TCI) | chain, no TCI | INTERP | 20 restarts |
|---|---|---|---|---|---|
| 2 | ~5M shots | 0.039 (6.4M) | 0.056 (4.2M) | 0.145 (2.9M) | 0.113 (3.5M) |
| 2 | ~20M | 0.022 (26M) | 0.032 (17M) | 0.113 (12M) | 0.087 (15M) |
| 2 | ~100M | 0.014 (109M) | 0.017 (72M) | 0.090 (52M) | 0.061 (63M) |
| 3 | ~10M | 0.062 (8.5M) | 0.051 (13M) | 0.098 (16M) | 0.199 (20M) |
| 3 | ~40M | 0.036 (35M) | 0.033 (54M) | 0.070 (67M) | 0.156 (89M) |

- **At p ≥ 2 the chain is by far the most noise-robust method.** At 1024 shots it lands
  0.022 (p = 2) and 0.036 (p = 3) above the best on average, against 0.087–0.20 for
  INTERP and random restarts. Random restarts at 16384 shots and 270M shots per
  instance (p = 2) are worse than the TCI chain at 256 shots and 6M. The chain also
  keeps 94–96% of the best ground-state probability at 1024 shots; INTERP and restarts
  keep 65–85%.
- **Why: noisy local search needs good starts.** COBYLA under noise stops early (the
  restart and INTERP runs use 4–6× fewer circuits than without noise) and wherever it
  happens to be. The chain starts every polish from a basin already found at a
  neighbouring t or the depth below, so a short noisy polish suffices. INTERP is hit
  hardest: its p = 1 optimum is noisy and each lift carries the error up.
- **TCI's part, at equal total shots: a small gain at p = 2, none at p = 3.** At p = 2
  the TCI chain lies below the no-TCI one at every budget (e.g. 0.022 at 26M vs 0.032
  at 17M; 0.014 at 109M vs 0.017 at 72M); at p = 3 they are level. The no-TCI chain is
  far cheaper under noise (4–13k runs vs 26–36k), because its polishes stop early too,
  while the depth-1 TT alone costs ~12k runs under noise.
- **At p = 1 the sweep TT is the worst method under noise.** Noise is not low rank: the
  TT's cost doubles (5.6k → 12k runs) and its slice minima are less accurate (0.017 at
  1024 shots vs 0.010 for 20 restarts, which spend fewer shots). Exact, it was the best
  p = 1 sweep method.
- **Caveats:** one noise realisation; n = 10. COBYLA is not designed for noisy
  objectives, so the same benchmark was rerun with SPSA (below).

<a id="spsa"></a>**With SPSA instead of COBYLA** (`--optimizer spsa`;
`results/shots_spsa_n10_p*.json`). `set_local_optimizer("spsa")` replaces COBYLA in
every polish of every method with SPSA (simultaneous perturbation stochastic
approximation, the standard optimiser for shot-based QAOA): 100 iterations of 2
circuits, perturbation 0.1, step calibrated at the start to ≤ 0.1 per angle, updates
clipped, and the final angles averaged over the second half of the iterates. (A first
version, with a step of 0.2 and no floor on the calibration gradient, diverged at
p = 3 even without noise; those runs were discarded.) Scored against the same
reference as the COBYLA runs (the SPSA runs found no lower optima). Mean gap:

| p | method | 256 shots | 1024 | 4096 | 16384 | runs | P(gs) at 1024 |
|---|---|---|---|---|---|---|---|
| 1 | sweep TT + SPSA | 0.006 | **0.002** | **0.002** | **0.001** | 21,064 | 0.017 |
| 1 | INTERP + SPSA | **0.005** | 0.003 | 0.002 | 0.001 | 16,400 | 0.016 |
| 1 | 20 restarts + SPSA | 0.007 | 0.003 | 0.002 | 0.001 | 65,600 | 0.016 |
| 2 | chain (TCI) + SPSA | 0.011 | 0.005 | 0.008 | 0.007 | 50,911 | 0.047 |
| 2 | **chain, no TCI + SPSA** | **0.008** | **0.003** | **0.004** | **0.004** | 21,541 | **0.047** |
| 2 | INTERP + SPSA | 0.052 | 0.047 | 0.047 | 0.049 | 19,680 | 0.042 |
| 2 | 20 restarts + SPSA | 0.049 | 0.037 | 0.042 | 0.038 | 65,600 | 0.043 |
| 3 | chain (TCI) + SPSA | 0.014 | **0.011** | 0.010 | **0.007** | 95,251 | 0.090 |
| 3 | **chain, no TCI + SPSA** | **0.013** | 0.012 | **0.008** | 0.008 | 66,343 | 0.090 |
| 3 | INTERP + SPSA | 0.080 | 0.069 | 0.066 | 0.070 | 22,960 | 0.078 |
| 3 | 20 restarts + SPSA | 0.138 | 0.118 | 0.127 | 0.115 | 65,600 | 0.064 |

At equal total shots per instance, best of COBYLA and SPSA for each method:

| p | budget | chain (TCI) | chain, no TCI | INTERP | 20 restarts |
|---|---|---|---|---|---|
| 2 | ~5M | 0.039 (COBYLA, 6.4M) | **0.008** (SPSA, 5.6M) | 0.052 (SPSA, 5.0M) | 0.113 (COBYLA, 3.5M) |
| 2 | ~20M | 0.011 (SPSA, 13M) | **0.003** (SPSA, 22M) | 0.047 (SPSA, 20M) | 0.049 (SPSA, 17M) |
| 2 | ~70M | 0.005 (SPSA, 52M) | **0.004** (SPSA, 87M) | 0.047 (SPSA, 81M) | 0.037 (SPSA, 67M) |
| 3 | ~20M | 0.014 (SPSA, 25M) | **0.013** (SPSA, 17M) | 0.069 (SPSA, 24M) | 0.138 (SPSA, 17M) |
| 3 | ~70–100M | **0.011** (SPSA, 98M) | 0.012 (SPSA, 68M) | 0.066 (SPSA, 94M) | 0.118 (SPSA, 67M) |

- **SPSA helps every method, but the chain's lead survives.** At p = 2 and 3 the chain
  with SPSA lands 0.003–0.014 above the best, INTERP and restarts with SPSA 0.037–0.14:
  5–10× worse at equal shots. Restarts and INTERP improve a lot with SPSA at p = 2
  (0.087 → 0.037 and 0.145 → 0.047 at 1024 shots) but not enough; at p = 3 random
  starts still mostly end in poor basins.
- **With SPSA, TCI's part in the chain disappears.** The no-TCI chain is as good as or
  better than the TCI chain at every budget (p = 2: 0.003 vs 0.005–0.011; p = 3:
  level), for less than half the runs at p = 2. With COBYLA the TCI chain had a small
  edge at p = 2: the TCI candidates partly made up for COBYLA's poor noisy polish.
- **At p = 1 the sweep TT recovers with SPSA:** level with INTERP and restarts
  (0.001–0.002 from 1024 shots on), so its COBYLA failure was mostly the polish, not
  the TT. It costs a little more than INTERP (21k vs 16k runs).
- **SPSA's gaps stop improving above ~1024 shots** (the fixed 100 iterations limit the
  precision). More iterations, or shots that grow along the run, are the obvious next
  step.
- **Ground-state probability:** the chains with SPSA keep 96–98% of the best P(gs)
  (0.047 of 0.048 at p = 2, 0.090 of 0.094 at p = 3); INTERP and restarts with SPSA
  68–90%.

### 10. Denser sweeps

`python benchmarks/bench_dense.py --p P --families F` for P = 1, 2, 3;
`results/dense_n10_p*.json`; scored by `python benchmarks/score_dense.py`.

The depth-1 sweep TT is built once and serves every t, so its cost per sweep point
should fall as the sweep gets denser, while INTERP and restarts cost the same per point.
Here t has a 64-point grid (`SweepFamily(..., R_t=6)`) and each of the 12 instances
(n = 10) is solved on nt = 8, 16, 32 and 64 evenly spaced points of it. The TT is built
once per instance (2.5k runs, rank 5–22) and shared, but every run is charged its full
cost. Exact ⟨C⟩. OK is against the best found at each t by any method at any nt.

At p ≥ 2 most of the chain's cost is not the TT but the seeding at every other t, so
the chain is also run with `sweep_chain(..., n_anchors=8)`: seeding at 8 fixed t only,
whatever nt is ("8 anchors").

| p | method | OK, nt = 8 | 16 | 32 | 64 | runs / point, nt = 8 → 64 |
|---|---|---|---|---|---|---|
| 1 | **sweep TT** | 100% | 100% | 100% | 100% | **563 → 291** |
| 1 | INTERP (5 restarts) | 100% | 100% | 100% | 100% | 523 → 531 |
| 1 | 20 restarts | 100% | 100% | 100% | 100% | 2,632 → 2,671 |
| 2 | chain (TCI) | 87.5% | 95.3% | 99.0% | **99.2%** | 3,115 → 2,654 |
| 2 | chain (TCI), 8 anchors | – | 94.8% | 92.4% | 93.1% | 2,853 → 1,781 |
| 2 | chain, no TCI | 95.8% | 98.4% | 98.4% | 98.7% | 1,441 → 1,255 |
| 2 | **chain, no TCI, 8 anchors** | – | 98.4% | 98.4% | **98.7%** | 1,349 → **825** |
| 2 | INTERP | 49.0% | 48.4% | 47.9% | 47.5% | 759 → 773 |
| 2 | 20 restarts | 61.5% | 60.9% | 61.2% | 61.2% | 5,172 → 5,220 |
| 3 | chain (TCI) | 76.0% | 77.1% | 83.6% | 88.4% | 5,916 → 5,271 |
| 3 | chain (TCI), 8 anchors | – | 77.1% | 83.9% | **89.3%** | 5,601 → 4,385 |
| 3 | chain, no TCI | 78.1% | 76.6% | 86.2% | 88.0% | 4,242 → 3,919 |
| 3 | chain, no TCI, 8 anchors | – | 76.6% | 83.9% | 87.4% | 4,117 → **3,446** |
| 3 | INTERP | 55.2% | 54.2% | 53.1% | 53.4% | 1,028 → 1,044 |
| 3 | 20 restarts | 31.2% | 27.6% | 26.8% | 26.6% | 5,940 → 5,951 |

(The 8-anchor rows at nt = 8 equal the plain chain. nt = 16 here is not benchmark 5's
setting: the angle grid has 6 bits instead of 5, and the reference is stronger.)

- **At p = 1 the TT's shared build pays off.** All three methods find every optimum,
  but the sweep TT's cost per point halves from 8 to 64 points (563 → 291 runs), while
  INTERP stays at ~530: level at 8 points, 1.8× cheaper at 64. This is the regime where
  TCI is cheaper than the per-t methods, for the same answers.
- **Denser sweeps make the chain more accurate, not just cheaper per point.** At p = 2
  the TCI chain goes from 87.5% at 8 points to 99.2% at 64; at p = 3 from 76% to 88%.
  Neighbouring t are closer, so a basin carried along t is still close to the optimum
  at the next point. INTERP and restarts solve each t alone and stay flat (48% and 61%
  at p = 2).
- **With dense sweeps, 8 seeding points are enough, without TCI.** The no-TCI chain
  seeded at 8 fixed t keeps 98.7% at 64 points (p = 2) for 825 runs per point: about
  INTERP's cost (773) at twice its accuracy (47.5%). With TCI, seeding at 8 points
  loses *field* basins at p = 2 (93%), but is the most accurate at p = 3 (89.3%).
- **TCI's part in the chain stays small at p ≥ 2:** the TCI and no-TCI chains are level
  in accuracy (99.2 vs 98.7% at p = 2, 88.4 vs 88.0% at p = 3, nt = 64), and the TCI
  chain costs 1.3–2× more per point. The cheap TT is not the expensive part; the TCI
  sampler at the seeding points is.

### 11. Two-parameter sweeps

`python benchmarks/bench_2d.py --p P --families F --seeds S` for P = 1, 2, 3;
`results/sweep2d_n10_p*.json`; scored by `python benchmarks/score_2d.py`.

`SweepFamily2D` holds a family Q(t₁, t₂) on a 32 × 32 grid, and `tci_sweep` then learns
one TT over (angles, t₁, t₂). Each grid point is encoded as one scalar, so the chain
runs unchanged along a snake path through the grid (consecutive points are grid
neighbours). Two families, 3 instances each (n = 10):

- *port2*: portfolio selection, risk aversion 0.2 + 6 t₁ × penalty weight 0.05 + 3 t₂
  (benchmark 5's *risk* and *penalty* combined);
- *mixfield*: MaxCut blended into a random QUBO (t₁) plus a uniform field (t₂).

Each instance is solved on 4 × 4, 8 × 8 and 16 × 16 points of the grid (16, 64, 256
points). The TT is built once per instance (*port2*: 5.0k runs, rank 8–25; *mixfield*:
7.6k runs, rank 25–29) and shared, but every run is charged its full cost. At p ≥ 2 the
chains are also run seeded at 16 points only (`n_anchors=16`). Exact ⟨C⟩; OK is against
the best found at each point by any method at any grid size.

| p | method | OK, 4 × 4 | 8 × 8 | 16 × 16 | runs / point, 4 × 4 → 16 × 16 |
|---|---|---|---|---|---|
| 1 | **2-D sweep TT** | 100% | 100% | 100% | 643 → **266** |
| 1 | INTERP (5 restarts) | 100% | 100% | 100% | 475 → 492 |
| 1 | 20 restarts | 100% | 100% | 100% | 2,399 → 2,463 |
| 2 | **chain (TCI)** | **94.8%** | **97.9%** | **99.9%** | 3,288 → 2,790 |
| 2 | chain (TCI), 16 anchors | – | 97.1% | 99.0% | 2,302 → 1,693 |
| 2 | chain, no TCI | 90.6% | 91.9% | 99.7% | 1,549 → 1,611 |
| 2 | chain, no TCI, 16 anchors | – | 89.1% | 95.4% | 1,210 → 914 |
| 2 | INTERP | 54.2% | 51.8% | 50.5% | 710 → 728 |
| 2 | 20 restarts | 65.6% | 64.1% | 62.8% | 5,172 → 5,195 |
| 3 | chain (TCI) | 84.4% | 88.3% | 92.1% | 6,071 → 5,543 |
| 3 | chain (TCI), 16 anchors | – | 87.5% | 94.9% | 5,095 → 4,393 |
| 3 | chain, no TCI | 84.4% | **90.1%** | **95.4%** | 4,341 → 4,603 |
| 3 | chain, no TCI, 16 anchors | – | 88.5% | 95.2% | 4,034 → **3,657** |
| 3 | INTERP | 54.2% | 54.2% | 53.1% | 979 → 999 |
| 3 | 20 restarts | 27.1% | 26.6% | 25.7% | 5,948 → 5,956 |

- **At p = 1 the 2-D TT is 1.8× cheaper than INTERP at 256 points** (266 vs 492 runs
  per point), for the same answers; at 16 points it costs 35% more. One TT over both
  parameters stays low rank (≤ 29) and costs 5–8k runs: about 10–15 INTERP solves.
- **The chain gets more accurate with grid density, again:** at p = 2 from 94.8% to
  99.9%, at p = 3 from 84% to 92–95%; INTERP (~51–54%) and restarts (26–66%) stay flat.
  The chain's lead over them is the same as in one dimension.
- **In two dimensions TCI helps the chain when the grid or the seeding is sparse.** At
  p = 2 the TCI chain gets 94.8% and 97.9% on the 4 × 4 and 8 × 8 grids against 90.6%
  and 91.9% without TCI (on *port2*, 8 × 8: 190 vs 163 of 192); seeded at 16 points on
  the 16 × 16 grid it keeps 99.0% against 95.4%. On the dense grid with full seeding
  they are level (99.9 vs 99.7%). At p = 3 the no-TCI chain is level or ahead (95.4 vs
  92.1% at 16 × 16).
- **Cost:** the TCI chain costs 1.2–2× more per point than the no-TCI one at p ≥ 2, as
  in one dimension. At p = 2 the TCI chain with 16 anchors is the best trade-off
  (99.0% at 1,693 runs per point); at p = 3 the no-TCI chain with 16 anchors (95.2% at
  3,657).

### 12. Three-parameter sweeps

`python benchmarks/bench_3d.py --p P --families F --seeds S` for P = 1, 2, 3;
`results/sweep3d_n10_p*.json`; scored by `python benchmarks/score_2d.py --prefix sweep3d`.

`SweepFamilyND` generalises `SweepFamily2D` to k parameters (`SweepFamily2D` is now the
k = 2 case and gives the same points as before); `tci_sweep` learns one TT over
(angles, t₁, t₂, t₃) on a 32 × 32 × 32 grid, and the chain follows a 3-D snake path
(one coordinate moves one step between consecutive points). Two families, 3 instances
each (n = 10):

- *port3*: *port2* plus the return weight 0.5 + 1.5 t₃;
- *mixfield3*: *mixfield* plus a random field hᵢ ∈ [−1, 1] of strength 2 t₃.

Each instance is solved on 4 × 4 × 4 and 8 × 8 × 8 points (64 and 512). The TT is built
once per instance and shared, but every run is charged its full cost. *port3*: 3.6k
runs, rank 13. *mixfield3*: rank 30 (the bond-dimension cap) at 21–23k runs on two
instances, and rank 18 at 6.0k runs on the third. Exact ⟨C⟩; OK is against the best
found at each point by any method at either grid size.

| p | method | OK, 4³ | 8³ | runs / point, 4³ → 8³ |
|---|---|---|---|---|
| 1 | **3-D sweep TT** | 100% | 100% | 409 → **260** |
| 1 | INTERP (5 restarts) | 100% | 100% | 473 → 482 |
| 1 | 20 restarts | 100% | 100% | 2,346 → 2,409 |
| 2 | **chain (TCI)** | **98.4%** | 99.4% | 2,931 → 2,779 |
| 2 | chain (TCI), 16 anchors | 97.7% | **98.9%** | 2,350 → **1,585** |
| 2 | chain, no TCI | 97.1% | **99.6%** | 1,483 → 1,825 |
| 2 | chain, no TCI, 16 anchors | 97.9% | 88.2% | 1,182 → 850 |
| 2 | INTERP | 58.9% | 55.1% | 704 → 715 |
| 2 | 20 restarts | 67.7% | 65.4% | 5,118 → 5,153 |
| 3 | chain (TCI) | 90.4% | 93.3% | 5,717 → 5,566 |
| 3 | chain (TCI), 16 anchors | **91.1%** | 94.0% | 5,116 → 4,276 |
| 3 | chain, no TCI | 90.1% | **97.0%** | 4,333 → 5,045 |
| 3 | chain, no TCI, 16 anchors | 89.8% | 96.3% | 3,995 → **3,599** |
| 3 | INTERP | 59.9% | 59.9% | 978 → 989 |
| 3 | 20 restarts | 25.5% | 27.4% | 5,958 → 5,966 |

- **At p = 1 the 3-D TT is 1.9× cheaper than INTERP at 512 points** (260 vs 482 runs
  per point), for the same answers. The first run, which polished only the TT's slice
  minima, missed for the first time in any sweep: 8 points of 3,456, worst gap 0.84
  (223 runs per point), all on the t₁ = 0 face (pure MaxCut plus fields on
  *mixfield3* seed 2, whose TT stopped early at rank 18; low risk aversion on *port3*
  seed 0), where the slice minima led into a poor basin. `sweep_chain` at p = 1 now
  also polishes the previous snake point's optimum (`warm_start=True`, the default),
  which removes every miss for 34–48 more runs per point; the p = 1 rows of benchmarks
  10 and 11 were rerun with it (benchmark 9 keeps the TT-only polish). With three parameters the TT
  also gets expensive on *mixfield3* (up to 23k runs, rank at the cap of 30), while
  *port3* stays cheap (3.6k, rank 13).
- **The chain still gets more accurate with density** (p = 2: 98.4% → 99.4%; p = 3:
  90% → 93–97%), and INTERP (55–60%) and restarts (25–68%) stay flat.
- **With sparse seeding TCI matters more in three dimensions than in two.** Seeded at
  16 of 512 points, the TCI chain keeps 98.9% at p = 2 against 88.2% without TCI (on
  *port3*: 1,522 vs 1,188 of 1,536). In 2-D the same gap was 99.0 vs 95.4%. With full
  seeding the two are level at p = 2 (99.4 vs 99.6%). At p = 3 the no-TCI chain is
  ahead (97.0 vs 93.3%), as in 2-D.
- **Best trade-offs:** at p = 2 the TCI chain with 16 anchors (98.9% at 1,585 runs per
  point); at p = 3 the no-TCI chain with 16 anchors (96.3% at 3,599). These are the same
  choices as in 2-D.

## Conclusions

**TCI does not beat QAOA. It is a deterministic, reusable first stage for QAOA.**
Everything below is at n = 10–14 with exact statevector simulation.

**Where TCI works**

1. **One p = 1 tensor train for a whole problem family.** It is learned over
   (angles × problem parameters) and stays low rank: at most 29 with one or two
   parameters, and up to the cap of 30 with three. Once a sweep has enough points it
   costs less than solving each point with INTERP, for the same answers:
   - one parameter, 64 points: 1.8× cheaper ([benchmark 10](#10-denser-sweeps));
   - two parameters, 16 × 16 points: 1.8× cheaper ([benchmark 11](#11-two-parameter-sweeps));
   - three parameters, 8 × 8 × 8 points: 1.9× cheaper, with each point also
     warm-started from its neighbour ([benchmark 12](#12-three-parameter-sweeps));
   - sparse sweeps (about 8–16 points): level with INTERP, or up to 35% dearer.
2. **The depth chain built on it** (p = 1 TT → p = 2 library → p = 3, 4;
   `sweep_chain`) is the best pipeline at p ≥ 2:
   - exact evaluations: at p = 2 it gets 94–99.9% of the best answers, against about
     50–60% for INTERP and 60–68% for 20 restarts, and it gets more accurate as the
     sweep gets denser while the others stay flat;
   - shot noise, COBYLA: it is still the most robust method;
   - shot noise, SPSA: mean gap 0.003–0.014, against 0.037–0.14 for INTERP and
     restarts ([benchmark 9](#9-shot-noise)).
3. **TCI seeding inside the chain helps only when coverage is sparse:** a single run, a
   coarse grid, or seeding at only a few points. The largest gain was with three
   parameters seeded at 16 of 512 points: 98.9% against 88.2% without TCI.

**Where it does not**

- **Higher-depth landscapes:** at p ≥ 2 they are not low rank (rank about 75), so TCI
  cannot learn them directly.
- **Seeding with good coverage:** the chain does as well without TCI and costs 12–26%
  less, and at p = 3 the no-TCI chain is often ahead. Most of the chain's lead comes
  from chaining the depths, not from TCI.
- **The QUBO itself:** TCI on the bitstrings beats hybridq but loses to time-matched
  local search.
- **One instance at p = 1:** it works, but INTERP is as good and cheaper.
- **The TT build can stop early.** If the landscape looks flat near its starting
  point, TCI can declare convergence while it still misses the optimum's basin. This
  caused the three-parameter misses (benchmark 12) and a total miss on one untested
  portfolio family. Starting pivots spread over the angle box (`angle_pivots=4`, used by
  the entry points) fix both cases tried, even without the neighbour warm start. The
  build costs more: 10.7k instead of 6.0k runs on the three-parameter instance, and
  1.9k instead of 224 on the portfolio family, where the early stop had missed every point.

**Practical recipe.** For a family of related problems, build one p = 1 TT and run the
chain from it (`solve_family(make_Q, p, tci=True)`). Use the TCI seeding when you can
afford only a few seeding points (`n_anchors`); otherwise random seeds
(`tci=False`) do as well for less. For a single QUBO, use `tci=False`.

---

## Next steps

0. **Shot noise, further** ([benchmark 9](#9-shot-noise)): the chain's lead survives
   SPSA. Next: SPSA with more iterations or shots that grow along the run (its gaps
   stop improving above ~1024 shots), several noise realisations, and p = 4 / n = 12.
1. **What is left at p = 3** (benchmark 9): the *field* needle at large t (needs
   lucky restarts), and on n = 12 *penalty* late-t optima that the deterministic chain
   misses by ≤ 0.006. At p = 4 the deterministic chain misses late-t *field* (and at
   n = 14 *penalty*) optima; the 8 wide restarts find them, but at n = 14 one seed
   gets 162–180 of 192 and three seeds are needed for all 192. Next: a cheaper chain (polish fewer library
   members per t).
2. **Seed portfolios.** For two or more runs, no-TCI seeds beat all-TCI, mixed
   ([mixing](#mix-p4)) and merged ([hybrid](#hybrid-p4)) runs. Open: how many no-TCI
   seeds are worth it at n = 14 and beyond, and a wider depth-2 shortlist so that
   extra candidates cannot push a basin out.
3. **Cheaper sweep polish.** Polish only the best slice minimum, and the next two only
   if it lands far from the TT prediction. This would cut the per-t cost (about 150
   runs at p = 1, 600–800 at p = 2) towards the warm starts' cost.
4. **Larger parameter grids.** Two-parameter sweeps ([benchmark 11](#11-two-parameter-sweeps))
   behave like denser one-parameter ones: the p = 1 TT is 1.8× cheaper than INTERP at
   256 points, and TCI seeding helps the chain when the grid or the seeding is sparse.
   Three parameters ([benchmark 12](#12-three-parameter-sweeps)) keep the saving (1.9×)
   once each point is also warm-started from its neighbour, but the build cost grows
   to 23k runs on the harder family; with sparse seeding, TCI's help to the chain grows
   (98.9 vs 88.2%). The early-stopping TT build that caused the misses is fixed by
   starting pivots over the angle box (`angle_pivots=4`, see [Conclusions](#conclusions)).
   Next: rerun benchmarks 5 and 10–12 with it to measure its cost across the board,
   and larger n.
5. **TCI-bits on structured, sparse instances** (chains, lattices, low-treewidth graphs)
   where the Boltzmann tensor really is low rank. Dense random QUBOs are the worst
   case for any tensor train.
6. **TCI on the QAOA output distribution** |⟨x|ψ(γ,β)⟩|² with an MPS simulator, to read
   off high-probability bitstrings at n beyond statevector reach.

## Tests

```bash
pytest -q          # 28 tests; the 15 TCI tests are skipped if xfacpy is not importable
```

## References

- E. Farhi, J. Goldstone, S. Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028.
- L. Zhou, S.-T. Wang, S. Choi, H. Pichler, M. D. Lukin, *Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices*, PRX 10, 021067 (2020).
- Y. Núñez Fernández et al., *Learning low-rank tensor train representations: new algorithms and libraries* (2024); `xfac`: https://github.com/tensor4all/xfac.
- H. Goto, K. Tatsumura, A. R. Dixon, *Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems*, Sci. Adv. 5, eaav2372 (2019).

## License

MIT — see [LICENSE](LICENSE).
