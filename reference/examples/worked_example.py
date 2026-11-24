"""Worked example: judge an observed worst-group coverage against the exact reference.

1. Simulate one held-out population with six sex x age cells, survey weights and a label x cell split conformal
   calibration of the positive class (alpha = 0.10).
2. Compute the observed worst-group positive-class coverage over the 11 calibrated groups (2 sexes, 3 age bands,
   6 cells), unweighted and design-weighted.
3. Compare each with its reference: Proposition 3 (unweighted), Proposition 5 (weighted) and the simulation-free
   bounds and 5% cut-off of Proposition 4.
Run:  python examples/worked_example.py
"""
import math
import numpy as np
from worst_group_reference import worst_group_reference, worst_group_bounds

ALPHA = 0.10
rng = np.random.default_rng(1)
n_cal = {"F|18-44": 60, "F|45-64": 160, "F|65+": 260, "M|18-44": 80, "M|45-64": 200, "M|65+": 300}   # calibration positives
n_pos = {"F|18-44": 45, "F|45-64": 120, "F|65+": 200, "M|18-44": 60, "M|45-64": 150, "M|65+": 230}   # evaluation positives

# --- step 1: calibrate each cell on its own positives and cover the evaluation positives
cells, covered = [], []
for c in n_cal:
    cal = rng.random(n_cal[c])                                   # nonconformity scores of calibration positives
    k = math.ceil((1 - ALPHA) * (n_cal[c] + 1))
    q = np.inf if k > n_cal[c] else np.sort(cal)[k - 1]           # split conformal threshold of the cell
    ev = rng.random(n_pos[c])                                    # scores of evaluation positives (same distribution)
    cells += [c] * n_pos[c]; covered += list(ev <= q)
cells, covered = np.array(cells), np.array(covered, float)
weights = rng.lognormal(0.0, 1.0, len(cells))                    # survey weights of the evaluation positives

# --- step 2: observed worst-group coverage
sex = np.array([c[0] for c in cells]); age = np.array([c[2:] for c in cells])
groups = [sex == "F", sex == "M"] + [age == a for a in ["18-44", "45-64", "65+"]] + [cells == c for c in n_cal]
obs_u = min(covered[g].mean() for g in groups)
obs_w = min(np.average(covered[g], weights=weights[g]) for g in groups)

# --- step 3: references
ref_u = worst_group_reference(n_cal, cells, groups, alpha=ALPHA, sims=20000, seed=0)
ref_w = worst_group_reference(n_cal, cells, groups, alpha=ALPHA, sims=20000, seed=0, weights=weights)
bnd = worst_group_bounds(n_cal, cells, groups, alpha=ALPHA)

print(f"unweighted worst group {obs_u:.3f}; reference mean {ref_u['mean']:.3f}, 5th percentile {ref_u['p05']:.3f}; "
      f"simulation-free cut-off t05 {bnd['t05']:.3f}")
print(f"weighted   worst group {obs_w:.3f}; reference mean {ref_w['mean']:.3f}, 5th percentile {ref_w['p05']:.3f}")
for name, o, r in (("unweighted", obs_u, ref_u), ("weighted", obs_w, ref_w)):
    verdict = "below the 5th percentile: more shortfall than finite samples explain" if o < r["p05"] else \
              "within the range expected under exact calibration"
    print(f"{name}: {verdict}")
