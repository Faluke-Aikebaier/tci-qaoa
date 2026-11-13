"""
tciqaoa — Tensor Cross Interpolation meets QAOA.

  QAOA                exact statevector QAOA for QUBO problems
  optimize_restarts   random-restart COBYLA angle search (baseline)
  optimize_interp     INTERP layer-by-layer warm start (baseline)
  optimize_tci        quantics-TCI landscape search over the angles (xfacpy)
  tci_minimize_bits   TCI on the Boltzmann tensor exp(-beta E(x)) over x in {0,1}^n (xfacpy)

xfacpy is imported lazily, so the QAOA simulator and baselines work without it.
"""

from .problems import (symmetrize, qubo_energy, all_energies, brute_force, maxcut_qubo,
                       cut_value, random_qubo, random_graph, local_search)
from .qaoa import QAOA
from .angles import optimize_restarts, optimize_interp, optimize_tci
from .tci_bits import tci_minimize_bits, variable_order

__version__ = "0.1.0"
