"""Simulation-based calibration of the finite-sample worst-group references used in SAGE-CP v2.1.

Uses verbatim copies of the notebook's `weighted_q`, `_beta_pi` and `worst_group_references` (cell 06_conformal.py).
Truth: the empirical distribution of the minimum positive-class coverage over eligible groups when label x cell split
conformal prediction is actually run (unweighted, or weighted with the exact density ratio under covariate shift).
Reference: the notebook's reference computed from the same group memberships and calibration counts / Kish ESS.
"""
import math, json, sys
import numpy as np, pandas as pd

ALPHA, R, SIMS, MINPOS = 0.10, 2000, 2000, 30

# ---------------- verbatim from 06_conformal.py ----------------
def weighted_q(cal_scores, cal_w, test_w, alpha):
    o = np.argsort(cal_scores); s = cal_scores[o]; cw = np.cumsum(cal_w[o]); W = cw[-1]
    k = np.searchsorted(cw, (1 - alpha) * (W + test_w), side="left")
    return np.where(k < len(s), s[np.minimum(k, len(s) - 1)], np.inf)

def _beta_pi(ncell, alpha, sims, rng):
    n = int(ncell)
    if n <= 0: return np.ones(sims)
    k = math.ceil((1 - alpha) * (n + 1))
    return np.ones(sims) if k > n else rng.beta(k, n + 1 - k, size=sims)

def worst_group_references(gm, cell_pos, cal_n, cal_ess, alpha, sims, rng):
    out = {}
    npos = gm["cal"][0].shape[0]
    cells = sorted(set(cell_pos)); ci = np.searchsorted(np.array(cells), cell_pos) if npos else np.zeros(0, int)
    probs = {"overlap": None}
    def _n_eff(src, c):
        v = src.get(c, 0)
        return v if v > 0 else sum(x for k, x in src.items() if str(k).split("|")[-1] == str(c).split("|")[-1])
    for ref, src in [("overlap_beta", cal_n), ("overlap_beta_ess", cal_ess)]:
        pi = np.column_stack([_beta_pi(math.floor(_n_eff(src, c)), alpha, sims, rng) for c in cells]) if cells else np.ones((sims, 0))
        probs[ref] = pi[:, ci] if npos else None
    for ref, p in probs.items():
        Z = (rng.random((sims, npos)) < (1 - alpha if p is None else p)).astype(np.float32)
        for name in ["cal", "race"]:
            M = gm[name][0]
            if M.shape[1] == 0:
                out[f"ref_{ref}_{name}_mean"] = out[f"ref_{ref}_{name}_p05"] = np.nan; continue
            mn = ((Z @ M) / M.sum(0)).min(1)
            out[f"ref_{ref}_{name}_mean"], out[f"ref_{ref}_{name}_p05"] = float(mn.mean()), float(np.quantile(mn, .05))
            out[f"ref_{ref}_{name}_sd"] = float(mn.std())
    return out
# -----------------------------------------------------------------

CELLS = ["S1|A1", "S1|A2", "S1|A3", "S2|A1", "S2|A2", "S2|A3"]
SIZES = {  # calibration positives per cell, target positives per cell
    "small":  ([25, 40, 60, 30, 50, 70], [35, 60, 90, 40, 70, 100]),
    "study":  ([60, 160, 260, 80, 200, 300], [45, 120, 200, 60, 150, 230]),
    "large":  ([300, 700, 1100, 400, 900, 1300], [150, 350, 550, 200, 450, 650]),
}
RACE_IND = [0.74, 0.09, 0.10, 0.05, 0.02]
RACE_DEP = {"A1": [0.60, 0.12, 0.18, 0.07, 0.03], "A2": [0.74, 0.10, 0.10, 0.05, 0.01], "A3": [0.84, 0.07, 0.05, 0.03, 0.01]}

def memberships(m, race, rng):
    cell_pos = np.repeat(np.array(CELLS), m)
    sex = np.array([c.split("|")[0] for c in cell_pos]); age = np.array([c.split("|")[1] for c in cell_pos])
    if race == "independent":
        rc = rng.choice(5, size=len(cell_pos), p=RACE_IND)
    else:
        rc = np.array([rng.choice(5, p=RACE_DEP[a]) for a in age])
    cols, lab = [], []
    for ax, v in [("Sex", sex), ("Age", age), ("Cell", cell_pos)]:
        for g in np.unique(v):
            mm = v == g
            if mm.sum() >= MINPOS: cols.append(mm); lab.append(f"{ax}={g}")
    rcols = [rc == g for g in range(5) if (rc == g).sum() >= MINPOS]
    gm = {"cal": (np.array(cols, np.float32).T, lab), "race": (np.array(rcols, np.float32).T if rcols else np.zeros((len(cell_pos), 0), np.float32), [])}
    return gm, cell_pos

def truth(gm, cell_pos, ncal, mu, rng):
    """Empirical min coverage over eligible groups, R replications of actual (weighted) label x cell split CP."""
    npos = len(cell_pos); cov = np.zeros((R, npos), np.float32); ess = {c: [] for c in CELLS}
    for c, n in zip(CELLS, ncal):
        idx = np.where(cell_pos == c)[0]; m = len(idx)
        if mu == 0:   # exchangeable, continuous scores
            cs = np.sort(rng.random((R, n)), axis=1); k = math.ceil((1 - ALPHA) * (n + 1))
            q = cs[:, k - 1] if k <= n else np.full(R, np.inf)
            cov[:, idx] = rng.random((R, m)) <= q[:, None]; ess[c] = [n]
        else:         # covariate shift: x ~ N(0,1) source, N(mu,1) target; w = exp(mu x - mu^2/2); score depends on x
            xs = rng.normal(size=(R, n)); xt = rng.normal(mu, 1, size=(R, m))
            ws = np.exp(mu * xs - mu ** 2 / 2); wt = np.exp(mu * xt - mu ** 2 / 2)
            ss = 1 / (1 + np.exp(-(xs + rng.normal(size=(R, n))))); st = 1 / (1 + np.exp(-(xt + rng.normal(size=(R, m)))))
            for r in range(R):
                q = weighted_q(ss[r], ws[r], wt[r], ALPHA)
                cov[r, idx] = st[r] <= q
            ess[c] = list(ws.sum(1) ** 2 / (ws ** 2).sum(1))
    out = {}
    for name in ["cal", "race"]:
        M = gm[name][0]
        if M.shape[1] == 0: out[name] = (np.nan, np.nan, np.nan); continue
        mn = ((cov @ M) / M.sum(0)).min(1)
        out[name] = (float(mn.mean()), float(np.quantile(mn, .05)), float(mn.std()))
    return out, {c: float(np.mean(v)) for c, v in ess.items()}

rows = []
rng = np.random.default_rng(20261109)
for size, (ncal, m) in SIZES.items():
    for race in ["independent", "dependent"]:
        gm, cell_pos = memberships(m, race, rng)
        for mu in [0, 0.5, 1.0, 1.5]:
            tr, ess = truth(gm, cell_pos, ncal, mu, rng)
            cal_n = dict(zip(CELLS, ncal))
            ref = worst_group_references(gm, cell_pos, cal_n, ess, ALPHA, SIMS, rng)
            # independent-binomial construction (v2): groups independent, Binomial(n_g, 1-alpha)/n_g
            for name in ["cal", "race"]:
                M = gm[name][0]
                if M.shape[1] == 0: continue
                ng = M.sum(0).astype(int)
                ind = (rng.binomial(ng, 1 - ALPHA, size=(SIMS, len(ng))) / ng).min(1)
                refname = "overlap_beta" if mu == 0 else "overlap_beta_ess"
                tm, tp, tsd = tr[name]
                essfrac = float(np.mean([ess[c] / n for c, n in zip(CELLS, ncal)]))
                rows.append(dict(size=size, race_structure=race, shift_mu=mu, ess_over_n=round(essfrac, 3), family=name,
                                 n_groups=M.shape[1], true_mean=tm, true_p05=tp, true_sd=tsd,
                                 mc_se_true_mean=tsd / math.sqrt(R),
                                 ref_used=refname, ref_mean=ref[f"ref_{refname}_{name}_mean"], ref_p05=ref[f"ref_{refname}_{name}_p05"],
                                 ref_sd=ref[f"ref_{refname}_{name}_sd"], mc_se_ref_mean=ref[f"ref_{refname}_{name}_sd"] / math.sqrt(SIMS),
                                 err_mean=ref[f"ref_{refname}_{name}_mean"] - tm, err_p05=ref[f"ref_{refname}_{name}_p05"] - tp,
                                 overlap_mean_err=ref[f"ref_overlap_{name}_mean"] - tm, indep_mean_err=float(ind.mean()) - tm))
            print(size, race, mu, 'done', flush=True)
df = pd.DataFrame(rows)
df["z_mean"] = df.err_mean / np.sqrt(df.mc_se_true_mean ** 2 + df.mc_se_ref_mean ** 2)
df.to_csv("reference_validation_results.csv", index=False)
print(df[["size", "race_structure", "shift_mu", "ess_over_n", "family", "true_mean", "ref_mean", "err_mean", "z_mean", "err_p05", "overlap_mean_err", "indep_mean_err"]].round(4).to_string())
