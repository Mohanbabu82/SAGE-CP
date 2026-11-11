# Worst-group reference for label × cell split conformal prediction

`worst_group_reference.py` implements Proposition 3 and Theorem 1 of the paper.

```python
import numpy as np
from worst_group_reference import worst_group_reference, worst_group_bounds

n_cal = {"F|18-44": 60, "F|45-64": 160, "F|65+": 260, "M|18-44": 80, "M|45-64": 200, "M|65+": 300}  # calibration positives per cell
cells = np.repeat(list(n_cal), [45, 120, 200, 60, 150, 230])                                             # cell of each evaluation positive
sex = np.array([c[0] for c in cells]); age = np.array([c[2:] for c in cells])
groups = [sex == "F", sex == "M"] + [age == a for a in ["18-44", "45-64", "65+"]] + [cells == c for c in n_cal]

exact = worst_group_reference(n_cal, cells, groups, alpha=0.10, sims=20000)   # Monte Carlo of the exact law
bounds = worst_group_bounds(n_cal, cells, groups, alpha=0.10)                 # no simulation
print(exact["mean"], exact["p05"], bounds)   # bounds: upper, lower, t05 (valid 5% cut-off), clark (point approximation)
```

An observed worst-group coverage below `t05` is below the 5th percentile of the exact law under exact calibration.
`bounds_validation.py` reproduces Supplementary Table S22 (`bounds_validation.csv`).
