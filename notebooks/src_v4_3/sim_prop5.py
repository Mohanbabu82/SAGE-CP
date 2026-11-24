"""Offline simulation for Additional file 1 (v4.3 revision).
Checks
 (1) Proposition 5: the design-weighted reference for weighted worst-group positive-class coverage under label x cell
     split conformal prediction with UNWEIGHTED calibration (claimed exact conditional on evaluation weights);
 (2) the Kish-ESS approximation of Proposition 5 for survey-weighted (SW) calibration;
 (3) that the eligibility rule (group size / positive count thresholds) does not by itself change the share of worst
     groups below the reference 5th percentile;
 (4) the effect of tied scores (coarse score grid) and randomised tie-breaking.
Lognormal weights mimic survey weights (sigma 1.0 gives CV about 1.3; sigma 1.3 about 2.1).
Writes results to final/derived/v4/sim_prop5.csv.  Seeded; runs on CPU in a few minutes."""
import numpy as np, pandas as pd, math, itertools, sys, time

ALPHA = 0.10
SEXES, AGES = ["F", "M"], ["18-34", "35-44", "45-54", "55-64", "65-74", "75+"]
CELLS = [f"{s}|{a}" for s in SEXES for a in AGES]

def beta_pi(n, sims, rng):
    n = int(n)
    if n <= 0: return np.ones(sims)
    k = math.ceil((1 - ALPHA) * (n + 1))
    return np.ones(sims) if k > n else rng.beta(k, n + 1 - k, size=sims)

def kish(w):
    w = np.asarray(w, float); return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0

def weighted_q(s, w, wt):
    o = np.argsort(s); s = s[o]; cw = np.cumsum(w[o]); W = cw[-1]
    k = np.searchsorted(cw, (1 - ALPHA) * (W + wt), side="left")
    return s[k] if k < len(s) else np.inf

def group_masks(cell_ev, y_ev, min_group, min_pos):
    sex = np.array([c.split("|")[0] for c in cell_ev]); age = np.array([c.split("|")[1] for c in cell_ev])
    M = []
    for lab in (sex, age, cell_ev):
        for g in np.unique(lab):
            ix = lab == g
            if ix.sum() >= min_group and y_ev[ix].sum() >= min_pos: M.append(ix)
    return np.array(M).T if M else None

def reference(cell_pos, wpos, Mpos, n_eff, sims, rng):
    cells = sorted(set(cell_pos)); ci = np.searchsorted(np.array(cells), cell_pos)
    pi = np.column_stack([beta_pi(n_eff[c], sims, rng) for c in cells])[:, ci]
    Z = (rng.random((sims, len(cell_pos))) < pi).astype(np.float64)
    Mw = Mpos * wpos[:, None]
    mn = ((Z @ Mw) / Mw.sum(0)).min(1)
    return float(mn.mean()), float(np.quantile(mn, 0.05))

def one_rep(rng, sig, cal_mode, grid, rt, min_group, min_pos, sims):
    # cell-level sizes: unequal, some small, as in held-out jurisdictions
    n_cal = {c: int(rng.integers(12, 260)) for c in CELLS}                    # calibration positives per cell
    m_ev = {c: int(rng.integers(10, 160)) for c in CELLS}                     # evaluation positives per cell
    neg_ev = {c: int(rng.integers(60, 900)) for c in CELLS}                   # evaluation negatives (for eligibility only)
    cell_ev = np.array(sum(([c] * (m_ev[c] + neg_ev[c]) for c in CELLS), []))
    y_ev = np.concatenate([np.r_[np.ones(m_ev[c], int), np.zeros(neg_ev[c], int)] for c in CELLS])
    w_ev = rng.lognormal(0, sig, len(y_ev)) if sig > 0 else np.ones(len(y_ev))
    M = group_masks(cell_ev, y_ev, min_group, min_pos)
    if M is None: return None
    pos = y_ev == 1; Mpos = M[pos].astype(np.float64); cell_pos = cell_ev[pos]; wpos = w_ev[pos]
    cover = np.zeros(pos.sum(), bool); n_eff = {}
    for c in CELLS:
        s_cal = rng.random(n_cal[c]); s_ev = rng.random(m_ev[c])            # exchangeable continuous nonconformity scores
        if grid:                                                              # coarse score grid -> ties
            s_cal = np.ceil(s_cal / grid) * grid; s_ev = np.ceil(s_ev / grid) * grid
            if rt: s_cal = s_cal + rng.random(n_cal[c]) * 1e-9; s_ev = s_ev + rng.random(m_ev[c]) * 1e-9
        if cal_mode == "unweighted":
            k = math.ceil((1 - ALPHA) * (n_cal[c] + 1)); q = np.inf if k > n_cal[c] else np.sort(s_cal)[k - 1]
            n_eff[c] = n_cal[c]
        else:                                                                 # SW: calibration weight = survey weight / mean, test weight 1
            wc = rng.lognormal(0, sig, n_cal[c]) if sig > 0 else np.ones(n_cal[c]); wc = wc / wc.mean()
            q = weighted_q(s_cal, wc, 1.0); n_eff[c] = int(round(kish(wc)))
        cover[cell_pos == c] = s_ev <= q
    Mw = Mpos * wpos[:, None]
    obs_w = float(((cover[:, None] * Mw).sum(0) / Mw.sum(0)).min())
    obs_u = float(((cover[:, None] * Mpos).sum(0) / Mpos.sum(0)).min())
    rw = reference(cell_pos, wpos, Mpos, n_eff, sims, rng)
    ru = reference(cell_pos, np.ones_like(wpos), Mpos, n_eff, sims, rng)
    return dict(obs_w=obs_w, obs_u=obs_u, refw_mean=rw[0], refw_p05=rw[1], refu_mean=ru[0], refu_p05=ru[1])

SCEN = []
for sig in (0.0, 1.0, 1.3):
    SCEN.append(dict(check="Prop5_unweighted_cal", sig=sig, cal_mode="unweighted", grid=0, rt=False, mg=100, mp=30))
for sig in (1.0, 1.3):
    SCEN.append(dict(check="Prop5_SW_cal_KishESS", sig=sig, cal_mode="SW", grid=0, rt=False, mg=100, mp=30))
for mg, mp in ((30, 8), (100, 30), (300, 60)):
    SCEN.append(dict(check="eligibility_rule", sig=1.0, cal_mode="unweighted", grid=0, rt=False, mg=mg, mp=mp))
for rt in (False, True):
    SCEN.append(dict(check="ties_grid_0.02", sig=1.0, cal_mode="unweighted", grid=0.02, rt=rt, mg=100, mp=30))

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000; SIMS = 2000
    out = []
    for i, sc in enumerate(SCEN):
        rng = np.random.default_rng(20261124 + i); t = time.time()
        rows = [r for r in (one_rep(rng, sc["sig"], sc["cal_mode"], sc["grid"], sc["rt"], sc["mg"], sc["mp"], SIMS) for _ in range(R)) if r]
        d = pd.DataFrame(rows)
        rec = dict(check=sc["check"], weight_sigma=sc["sig"], calibration=sc["cal_mode"], score_grid=sc["grid"], random_ties=sc["rt"],
                   min_group=sc["mg"], min_pos=sc["mp"], reps=len(d),
                   obs_weighted_mean=d.obs_w.mean(), refw_mean=d.refw_mean.mean(),
                   share_below_refw_p05=(d.obs_w < d.refw_p05).mean(),
                   share_weighted_below_unweighted_ref_p05=(d.obs_w < d.refu_p05).mean(),
                   obs_unweighted_mean=d.obs_u.mean(), refu_mean=d.refu_mean.mean(),
                   share_unweighted_below_refu_p05=(d.obs_u < d.refu_p05).mean())
        se = math.sqrt(0.05 * 0.95 / len(d)); rec["mc_se_share"] = se
        out.append(rec); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, f"{time.time()-t:.0f}s", flush=True)
    pd.DataFrame(out).to_csv("/home/claude/work/final/derived/v4/sim_prop5.csv", index=False)
