"""
Score the TCI ablation (benchmark 9): the chain with and without TCI, at every setting.

Reads every result file of a setting (the original TCI runs, the restart / chain+8 /
matched-budget runs and the no-TCI library seeds ``results/notci_*``), takes the common
best per (family, instance, t) over all of them (and the best stored in them), and reports OK / (16 x instances) per
method and library seed, worst miss and runs per instance.

Usage:  python benchmarks/score_notci.py
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

FAMS = ["mix", "field", "penalty", "risk"]


def tci_files(setting):
    n, p = setting
    if (n, p) == (10, 4):
        return ["results/p4.json"]
    if (n, p) == (10, 3):
        return ["results/large_n10_p3_v2.json"]
    if (n, p) == ("sched", 4):
        return [f"results/sched_n12_p4_{g}.json" for g in ("tci", "noTCI", "base")]
    return sorted(glob.glob(f"results/large_n{n}_p{p}*.json"))


def notci_files(setting):
    n, p = setting
    pre = "results/notci_sched_n12_p4_s*" if n == "sched" else f"results/notci_n{n}_p{p}_s*"
    hyb = "results/hyb_sched_n12_p4_s*" if n == "sched" else f"results/hyb_n{n}_p{p}_s*"
    return sorted(glob.glob(pre + ".json")) + sorted(glob.glob(hyb + ".json"))


def label(fn, method):
    """Method name, with the library seed for seed-dependent runs."""
    m = re.search(r"_s(\d)", os.path.basename(fn))
    if "notci" in fn or "hyb_" in os.path.basename(fn):
        return f"{method} s{m.group(1)}"
    if "large_n12_p4_noTCI" in fn or "sched_n12_p4_noTCI" in fn:
        return f"{method} s1"
    m2 = re.search(r"_c8(s\d)?_", fn)
    if m2 and method == "chain+8":
        return f"chain+8 s{(m2.group(1) or 's1')[1]}"
    return method


def score(setting):
    runs = defaultdict(dict)        # (family, seed) -> label -> (costs, evals)
    stored = {}                     # (family, seed) -> best stored in the files (incl. --prev)
    for fn in tci_files(setting) + notci_files(setting):
        for r in json.load(open(fn)):
            k = (r["family"], r["seed"])
            stored[k] = np.minimum(stored.get(k, np.inf), r["best"])
            for m, v in r["methods"].items():
                for i, c in enumerate(v["costs"]):
                    lab = label(fn, m)
                    if m in ("chain+8",) and len(v["costs"]) > 1:
                        lab = f"chain+8 s{i + 1}"
                    runs[(r["family"], r["seed"])][lab] = (np.asarray(c), v["n_evals"][i])
    out = defaultdict(lambda: [0, 0.0, [], defaultdict(int)])
    for key, ms in runs.items():
        fam = key[0]
        best = np.minimum(np.min([c for c, _ in ms.values()], axis=0), stored[key])
        for lab, (c, e) in ms.items():
            o = out[lab]
            ok = int(np.sum(c <= best + 1e-3))
            o[0] += ok
            o[1] = max(o[1], float(np.max(c - best)))
            o[2].append(e)
            o[3][fam] += ok
    return runs, out


def main():
    for setting in [(10, 3), (10, 4), (12, 4), (14, 4), ("sched", 4)]:
        runs, out = score(setting)
        if not runs:
            continue
        n_pts = 16 * len(runs)
        print(f"\n== n = {setting[0]}, p = {setting[1]}  ({len(runs)} instances, OK / {n_pts})")
        for lab in sorted(out):
            ok, w, ev, fam = out[lab]
            print(f"{lab:20s} {ok:4d}  worst {w:6.3f}  {np.mean(ev):10,.0f}  {dict(fam)}")


if __name__ == "__main__":
    main()
