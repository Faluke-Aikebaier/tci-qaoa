"""
Check hybridq's QUBO -> Ising conversion used by SimulatedBifurcation.

With x = (s + 1)/2, x^T Q x = s^T (Q/4) s + (Q 1 / 2) . s + const, so the Ising
energy -s^T J s - h^T s must differ from the QUBO energy by the SAME constant for
every s.  This script evaluates that difference on random states: a spread > 0
means the mapping is wrong.  It also shows the corrected field h = -Q 1 / 2.

Usage:  HYBRIDQ_PATH=/path/to/QAOA-inspired-optimisation python benchmarks/check_hybridq_sb_mapping.py
"""

import os
import sys

import numpy as np

if os.environ.get("HYBRIDQ_PATH"):
    sys.path.insert(0, os.environ["HYBRIDQ_PATH"])
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hybridq  # noqa: E402
from tciqaoa.problems import maxcut_qubo, random_graph, random_qubo, symmetrize  # noqa: E402


def spread(Q, J, h, trials=200):
    Qs = symmetrize(Q)
    rng = np.random.default_rng(0)
    d = []
    for _ in range(trials):
        x = rng.integers(0, 2, Q.shape[0])
        s = 2 * x - 1
        d.append(x @ Qs @ x - (-(s @ J @ s) - h @ s))
    return float(np.ptp(d))


for name, Q in [("maxcut n=20", maxcut_qubo(random_graph(20, 0.4, seed=0))),
                ("random QUBO n=20", random_qubo(20, seed=0))]:
    J, h, _ = hybridq.SimulatedBifurcation()._ising_from_qubo(Q)
    h_fixed = -0.5 * symmetrize(Q).sum(axis=1)
    print(f"{name:18s} spread(hybridq h) = {spread(Q, J, h):8.3f}   "
          f"spread(fixed h) = {spread(Q, J, h_fixed):.1e}")
