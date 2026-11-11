"""Validation of the simulation-free results for the worst-group reference (worst_group_reference.py).
Configurations: (i) the 6 configurations of reference_validation.py (3 size regimes x 2 race structures), with the
calibrated-group family (sex, age, sex x age) and the race family; (ii) 300 random configurations; (iii) 12
configurations with a borrowed cell, where the truth is actual split conformal prediction with shared thresholds."""
import math, numpy as np, pandas as pd
from worst_group_reference import worst_group_reference, worst_group_bounds
ALPHA, MINPOS, SIMS = 0.10, 30, 40000
CELLS = ["S1|A1", "S1|A2", "S1|A3", "S2|A1", "S2|A2", "S2|A3"]
SIZES = {"small": ([25, 40, 60, 30, 50, 70], [35, 60, 90, 40, 70, 100]),
         "study": ([60, 160, 260, 80, 200, 300], [45, 120, 200, 60, 150, 230]),
         "large": ([300, 700, 1100, 400, 900, 1300], [150, 350, 550, 200, 450, 650])}
RACE_IND = [0.74, 0.09, 0.10, 0.05, 0.02]
RACE_DEP = {"A1": [0.60, 0.12, 0.18, 0.07, 0.03], "A2": [0.74, 0.10, 0.10, 0.05, 0.01], "A3": [0.84, 0.07, 0.05, 0.03, 0.01]}
rng = np.random.default_rng(20261111)

def groups_for(cell_pos, race):
    sex = np.array([c.split("|")[0] for c in cell_pos]); age = np.array([c.split("|")[1] for c in cell_pos])
    rc = rng.choice(5, size=len(cell_pos), p=RACE_IND) if race == "independent" else np.array([rng.choice(5, p=RACE_DEP[a]) for a in age])
    cal = [v == g for v in (sex, age, cell_pos) for g in np.unique(v) if (v == g).sum() >= MINPOS]
    rac = [rc == g for g in range(5) if (rc == g).sum() >= MINPOS]
    return {"cal": cal, "race": rac}

rows = []
def run(tag, ncal, m, race, pool_of=None, truth_fn=None):
    cell_pos = np.repeat(np.array(CELLS), m)
    fam = groups_for(cell_pos, race)
    n_cal = dict(zip(CELLS, ncal)) if pool_of is None else ncal
    for name, G in fam.items():
        if not G: continue
        ex = worst_group_reference(n_cal, cell_pos, G, ALPHA, SIMS, int(rng.integers(1e9)), pool_of)
        b = worst_group_bounds(n_cal, cell_pos, G, ALPHA, pool_of)
        r = dict(config=tag, race=race, family=name, n_groups=len(G), min_group=int(min(g.sum() for g in G)),
                 exact_mean=ex["mean"], exact_p05=ex["p05"], mc_se=ex["draws"].std() / math.sqrt(SIMS),
                 upper=b["upper"], lower=b["lower"], clark=b["clark"], t05=b["t05"],
                 size_t05=float((ex["draws"] < b["t05"]).mean()), one_minus_alpha=1 - ALPHA)
        if truth_fn is not None:
            tm = truth_fn(cell_pos, G); r.update(truth_mean=tm)
        rows.append(r)

for size, (ncal, m) in SIZES.items():
    for race in ["independent", "dependent"]:
        run(size, ncal, m, race)
for i in range(300):
    ncal = np.round(np.exp(rng.uniform(np.log(8), np.log(1500), 6))).astype(int).tolist()
    m = np.round(np.exp(rng.uniform(np.log(10), np.log(800), 6))).astype(int).tolist()
    run(f"random{i}", ncal, m, ["independent", "dependent"][i % 2])

# borrowed cells: cell S1|A1 has no calibration positives and borrows its age parent's pool (S1|A1 + S2|A1 pooled).
def borrowed_truth(ncal_parent, R=4000):
    def f(cell_pos, G):
        M = np.array(G, np.float32).T; cov = np.zeros((R, len(cell_pos)), np.float32)
        for c in CELLS:
            idx = np.where(cell_pos == c)[0]
            if c in ("S1|A1", "S2|A1"): continue
            n = ncal_parent[c]; k = math.ceil((1 - ALPHA) * (n + 1))
            q = np.sort(rng.random((R, n)), 1)[:, k - 1] if k <= n else np.full(R, np.inf)
            cov[:, idx] = rng.random((R, len(idx))) <= q[:, None]
        n = ncal_parent["A1pool"]; k = math.ceil((1 - ALPHA) * (n + 1))
        q = np.sort(rng.random((R, n)), 1)[:, k - 1]               # one threshold shared by both A1 cells
        for c in ("S1|A1", "S2|A1"):
            idx = np.where(cell_pos == c)[0]; cov[:, idx] = rng.random((R, len(idx))) <= q[:, None]
        return float(((cov @ M) / M.sum(0)).min(1).mean())
    return f
for j in range(12):
    nc = {c: int(np.exp(rng.uniform(np.log(30), np.log(800)))) for c in CELLS[1:3] + CELLS[4:]}
    nc["A1pool"] = int(np.exp(rng.uniform(np.log(10), np.log(300))))
    pool_of = {"S1|A1": "A1pool", "S2|A1": "A1pool", **{c: c for c in CELLS[1:3] + CELLS[4:]}}
    m = np.round(np.exp(rng.uniform(np.log(30), np.log(600), 6))).astype(int).tolist()
    run(f"borrowed{j}", nc, m, ["independent", "dependent"][j % 2], pool_of, borrowed_truth(nc))

df = pd.DataFrame(rows)
df.to_csv("bounds_validation.csv", index=False)
d = df
print("configs x families:", len(d))
print("bound order violations: lower>exact", int((d.lower > d.exact_mean + 3 * d.mc_se).sum()), " exact>upper", int((d.exact_mean > d.upper + 3 * d.mc_se).sum()))
print("max size of t05 cut-off:", round(d.size_t05.max(), 4), " mean", round(d.size_t05.mean(), 4))
for c in ["upper", "lower", "clark", "t05"]:
    e = (d[c] - (d.exact_p05 if c == "t05" else d.exact_mean)); print(c, "err: mean", round(e.mean(), 4), "max|.|", round(e.abs().max(), 4), "median|.|", round(e.abs().median(), 4))
print("1-alpha minus exact mean: mean", round((0.9 - d.exact_mean).mean(), 4), "max", round((0.9 - d.exact_mean).max(), 4))
b = d.dropna(subset=["truth_mean"]) if "truth_mean" in d else d.iloc[:0]
if len(b): print("borrowed: |exact-truth| max", round((b.exact_mean - b.truth_mean).abs().max(), 4), " mc_se", round(b.mc_se.max(), 4))
