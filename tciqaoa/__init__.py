"""
tciqaoa — Tensor Cross Interpolation meets QAOA.

  QAOA                exact statevector QAOA for QUBO problems
  optimize_restarts   random-restart COBYLA angle search (baseline)
  optimize_interp     INTERP layer-by-layer warm start (baseline)
  optimize_tci        quantics-TCI landscape search over the angles (xfacpy)
  tci_minimize_bits   TCI on the Boltzmann tensor exp(-beta E(x)) over x in {0,1}^n (xfacpy)

Main entry points, each with tci=True / tci=False (see tciqaoa.solve):
  solve_qubo          one QUBO, QAOA at depth p
  solve_family        a family Q(t_1, .., t_k) at many parameter points, QAOA at depth p
  solve_bits          minimise x^T Q x directly (one QUBO or a list)
Command line: python -m tciqaoa --help

xfacpy is imported lazily, so the QAOA simulator and baselines work without it.
"""

from .problems import (symmetrize, qubo_energy, all_energies, brute_force, maxcut_qubo,
                       cut_value, random_qubo, random_graph, local_search,
                       portfolio_data, portfolio_qubo)
from .qaoa import QAOA
from .angles import (optimize_restarts, optimize_interp, optimize_tci, landscape_minima,
                     optimize_from, beta_images, optimize_from_images)
from .tci_bits import tci_minimize_bits, tci_sweep_bits, variable_order
from .solve import solve_qubo, solve_family, solve_bits

__version__ = "0.1.0"
