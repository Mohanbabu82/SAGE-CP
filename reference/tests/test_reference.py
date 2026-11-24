import numpy as np
from worst_group_reference import worst_group_reference, worst_group_bounds

def _setup():
    n_cal = {"a": 50, "b": 200}
    cells = np.repeat(["a", "b"], [40, 160])
    groups = [cells == "a", cells == "b", np.ones(200, bool)]
    return n_cal, cells, groups

def test_bounds_bracket_reference():
    n_cal, cells, groups = _setup()
    r = worst_group_reference(n_cal, cells, groups, sims=20000, seed=1)
    b = worst_group_bounds(n_cal, cells, groups)
    assert b["lower"] - 0.003 <= r["mean"] <= b["upper"] + 0.003
    assert b["t05"] <= r["p05"] + 0.003

def test_equal_weights_match_unweighted():
    n_cal, cells, groups = _setup()
    a = worst_group_reference(n_cal, cells, groups, sims=5000, seed=2)
    b = worst_group_reference(n_cal, cells, groups, sims=5000, seed=2, weights=np.ones(200))
    assert abs(a["mean"] - b["mean"]) < 1e-12
