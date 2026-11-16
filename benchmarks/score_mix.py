"""
Mixed seed portfolios: one TCI run plus no-TCI seeds, against all-TCI and all-no-TCI.

Independent runs combine by taking the best cost at each (instance, t), so a portfolio
of runs can be scored from the stored results of benchmark 9 (no new runs).  For a
portfolio size k, every choice of k runs of the given kinds is scored and the scores
are averaged, so no single lucky seed decides the comparison.  Runs per instance are
the sum over the portfolio.  Scored against the same common best as score_notci.py.

  mixed      1 TCI run + (k - 1) no-TCI seeds
  no TCI     k no-TCI seeds
  TCI        k TCI runs (chain + 8 only: the deterministic TCI chain has one run)

Usage:  python benchmarks/score_mix.py
"""

import os
import sys
from itertools import combinations, product

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from score_notci import score  # noqa: E402

SETTINGS = [(10, 3), (10, 4), (12, 4), (14, 4), ("sched", 4)]


def labels(ms, kind, restarts):
    base = "chain+8" if restarts else "chain"
    if kind == "tci":
        return sorted(l for l in ms if l == base or l.startswith(base + " s"))
    return sorted(l for l in ms if l.startswith(base + "-noTCI s"))


def portfolio(runs, picks):
    """OK and mean runs per instance of the portfolio picks[key] (list of labels)."""
    ok, ev = 0, []
    for key, ms in runs.items():
        best = np.min([c for c, _ in ms.values()], axis=0)
        b = np.min([ms[l][0] for l in picks], axis=0)
        ok += int(np.sum(b <= best + 1e-3))
        ev.append(sum(ms[l][1] for l in picks))
    return ok, float(np.mean(ev))


def choices(ms, k, mix, restarts):
    tci, no = labels(ms, "tci", restarts), labels(ms, "no", restarts)
    if mix == "mixed":
        return [[t] + list(c) for t, c in product(tci, combinations(no, k - 1))]
    pool = tci if mix == "TCI" else no
    return [list(c) for c in combinations(pool, k)]


def main():
    for restarts in (False, True):
        print(f"\n######## {'chain + 8 (8 wide restarts)' if restarts else 'deterministic chain'}")
        for setting in SETTINGS:
            runs, _ = score(setting)
            ms0 = next(iter(runs.values()))
            n_pts = 16 * len(runs)
            print(f"\n== n = {setting[0]}, p = {setting[1]}  (OK / {n_pts})")
            for k in (1, 2, 3):
                for mix in ("TCI", "mixed", "no TCI"):
                    if mix == "mixed" and k == 1:
                        continue
                    cs = choices(ms0, k, mix, restarts)
                    if not cs:
                        continue
                    res = [portfolio(runs, c) for c in cs]
                    oks = [r[0] for r in res]
                    print(f"k={k} {mix:7s} {np.mean(oks):6.1f}  ({min(oks)}-{max(oks)}, "
                          f"{len(cs)} choices)  {np.mean([r[1] for r in res]):10,.0f}")


if __name__ == "__main__":
    main()
