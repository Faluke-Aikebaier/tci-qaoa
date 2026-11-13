"""Turn results/*.json into the markdown tables used in README (written to results/summary.md)."""

import json
import os
from collections import defaultdict

import numpy as np

root = os.path.join(os.path.dirname(__file__), "..", "results")
lines = []

p = os.path.join(root, "qaoa.json")
if os.path.exists(p):
    rows = json.load(open(p))
    g = defaultdict(list)
    for r in rows:
        g[(r["kind"], r["n"], r["p"], r["method"])].append(r)
    lines += ["## QAOA angle optimisation (statevector QAOA)", "",
              "| problem | n | p | method | approx. ratio | P(ground) | 1000 shots hit opt. | circuit evals | time (s) | TT rank | pivot err |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (k, n, pp, m), rs in sorted(g.items()):
        f = lambda key: np.mean([r[key] for r in rs])
        tt = f"{f('tt_max_rank'):.0f}" if "tt_max_rank" in rs[0] else ""
        pe = f"{np.median([r['pivot_error'] for r in rs]):.0e}" if "pivot_error" in rs[0] else ""
        lines.append(f"| {k} | {n} | {pp} | {m} | {f('approx_ratio'):.4f} | {f('p_ground'):.3f} | "
                     f"{sum(r['shots_hit'] for r in rs)}/{len(rs)} | {f('n_evals'):.0f} | {f('time'):.1f} | {tt} | {pe} |")
    lines.append("")

p = os.path.join(root, "solvers.json")
if os.path.exists(p):
    rows = json.load(open(p))
    methods = list(rows[0]["methods"])
    g = defaultdict(list)
    for r in rows:
        g[(r["kind"], r["n"])].append(r)
    lines += ["## Solver comparison: hits of the reference energy (exact for n <= 22, else best known)", "",
              "| problem | n | ref | " + " | ".join(methods) + " |",
              "|---|---|---|" + "---|" * len(methods)]
    for (k, n), rs in sorted(g.items()):
        cells = []
        for m in methods:
            hit = sum(r["methods"][m]["hit"] for r in rs)
            gap = 100 * np.mean([r["methods"][m]["gap"] for r in rs])
            cells.append(f"{hit}/{len(rs)} ({gap:.1f}%)")
        ref = "exact" if rs[0]["exact"] else "best known"
        lines.append(f"| {k} | {n} | {ref} | " + " | ".join(cells) + " |")
    lines += ["", "Cell = instances where the method reached the reference (mean relative energy gap).", "",
              "| problem | n | " + " | ".join(f"{m} time (s)" for m in methods) + " |",
              "|---|---|" + "---|" * len(methods)]
    for (k, n), rs in sorted(g.items()):
        lines.append(f"| {k} | {n} | " + " | ".join(
            f"{np.mean([r['methods'][m]['time'] for r in rs]):.2f}" for m in methods) + " |")
    lines.append("")
    lines.append("TCI-bits mean energy evaluations / max TT rank: " + ", ".join(
        f"{k} n={n}: {np.mean([r['methods']['TCI-bits']['n_energy_evals'] for r in rs]):.0f} / "
        f"{max(r['methods']['TCI-bits']['max_rank'] for r in rs)}" for (k, n), rs in sorted(g.items())))

open(os.path.join(root, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
