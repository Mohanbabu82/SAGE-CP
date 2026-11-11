"""worst_group_reference: finite-sample reference for the worst-group coverage of label x cell split conformal prediction.

Setting (Proposition 3 of the paper). Positives are calibrated within K disjoint cells. Cell k has n_k calibration
positives (a "pool"); a borrowed cell uses its parent's pool, so cells in the same pool share one threshold.
Evaluation positive i lies in cell k(i). Groups are arbitrary (possibly overlapping) sets of evaluation positives.
Under exchangeability within each pool and continuous scores, the coverage probability of pool p is
    pi_p ~ Beta(kappa_p, n_p + 1 - kappa_p),  kappa_p = ceil((1 - alpha)(n_p + 1))   (pi_p = 1 if kappa_p > n_p),
independent across pools, and Z_i | pi ~ Bernoulli(pi_{p(k(i))}) independently.

Functions
---------
worst_group_reference(n_cal, cell_of_pos, groups, alpha, sims, seed, pool_of=None)
    Exact law of min_h Cov_h by Monte Carlo of the two-stage model (mean, 5th percentile, draws).
worst_group_bounds(n_cal, cell_of_pos, groups, alpha, pool_of=None)
    Simulation-free results:
      upper  = min_h E[Cov_h]                      (E[min] <= upper; Jensen / min of means)
      lower  = E[min of independent copies with the exact marginals]   (E[min] >= lower; association, Theorem 1)
      t05    = largest t with prod_h P(Cov_h >= t) >= 0.95   (P(min < t05) <= 0.05; a valid level-0.05 cut-off)
      clark  = normal approximation to E[min] (Clark, 1961) using the exact means and covariances
    Marginals P(Cov_h = j/|h|) are exact: Cov_h * |h| is a sum over pools of independent beta-binomial counts.
"""
import math
import numpy as np
from scipy.stats import betabinom, norm


def _kappa(n, alpha):
    return math.ceil((1 - alpha) * (n + 1))


def _pools(n_cal, cells, pool_of):
    pool_of = pool_of or {c: c for c in cells}
    pools = sorted(set(pool_of[c] for c in cells))
    return pool_of, pools


def worst_group_reference(n_cal, cell_of_pos, groups, alpha=0.10, sims=2000, seed=0, pool_of=None):
    """n_cal: dict pool -> number of calibration positives; cell_of_pos: array of cell labels (one per evaluation
    positive); groups: list of boolean arrays over evaluation positives; pool_of: dict cell -> pool (default identity)."""
    rng = np.random.default_rng(seed)
    cell_of_pos = np.asarray(cell_of_pos)
    cells = sorted(set(cell_of_pos))
    pool_of, pools = _pools(n_cal, cells, pool_of)
    pi = {}
    for p in pools:
        n = int(n_cal[p]); k = _kappa(n, alpha)
        pi[p] = np.ones(sims) if (n <= 0 or k > n) else rng.beta(k, n + 1 - k, size=sims)
    P = np.column_stack([pi[pool_of[c]] for c in cell_of_pos])
    Z = (rng.random(P.shape) < P).astype(np.float32)
    M = np.array(groups, np.float32).T
    mn = ((Z @ M) / M.sum(0)).min(1)
    return dict(mean=float(mn.mean()), p05=float(np.quantile(mn, 0.05)), draws=mn)


def _group_pmf(counts_by_pool, n_cal, alpha):
    """Exact pmf of the number of covered positives in a group: convolution of independent beta-binomials."""
    pmf = np.array([1.0])
    for p, m in counts_by_pool.items():
        if m == 0: continue
        n = int(n_cal[p]); k = _kappa(n, alpha)
        if n <= 0 or k > n:
            f = np.zeros(m + 1); f[m] = 1.0
        else:
            f = betabinom.pmf(np.arange(m + 1), m, k, n + 1 - k)
        pmf = np.convolve(pmf, f)
    return pmf


def _moments(n, alpha):
    if n <= 0 or _kappa(n, alpha) > n: return 1.0, 0.0
    a = _kappa(n, alpha); b = n + 1 - a
    return a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1))


def worst_group_bounds(n_cal, cell_of_pos, groups, alpha=0.10, pool_of=None, grid=2001):
    cell_of_pos = np.asarray(cell_of_pos)
    cells = sorted(set(cell_of_pos))
    pool_of, pools = _pools(n_cal, cells, pool_of)
    pool_of_pos = np.array([pool_of[c] for c in cell_of_pos])
    G = [np.asarray(g, bool) for g in groups]
    t = np.linspace(0, 1, grid)
    surv = []   # P(Cov_h >= t) on the grid
    means = []
    for g in G:
        m = int(g.sum())
        cnt = {p: int((pool_of_pos[g] == p).sum()) for p in pools}
        pmf = _group_pmf(cnt, n_cal, alpha)
        vals = np.arange(len(pmf)) / m
        sf = 1 - np.concatenate([[0.0], np.cumsum(pmf)])        # sf[j] = P(X >= j)
        idx = np.ceil(t * m - 1e-12).astype(int)                 # P(Cov >= t) = P(X >= ceil(t m))
        surv.append(np.where(idx <= m, sf[np.minimum(idx, m)], 0.0))
        means.append(float((vals * pmf).sum()))
    S = np.prod(np.array(surv), axis=0)                          # P(min of independent copies >= t)
    lower = float(np.trapezoid(S, t)) if hasattr(np, "trapezoid") else float(np.trapz(S, t))
    ok = np.where(S >= 0.95)[0]
    t05 = float(t[ok[-1]]) if len(ok) else 0.0
    upper = float(min(means))
    # exact covariance of group coverages under the two-stage model
    mom = {p: _moments(int(n_cal[p]), alpha) for p in pools}
    H = len(G); C = np.zeros((H, H))
    for a in range(H):
        for b in range(a, H):
            ga, gb = G[a], G[b]; ma, mb = ga.sum(), gb.sum(); v = 0.0
            for p in pools:
                mu, var = mom[p]
                ia = ga & (pool_of_pos == p); ib = gb & (pool_of_pos == p)
                v += ia.sum() * ib.sum() / (ma * mb) * var                            # shared pi_p
                v += (ia & ib).sum() / (ma * mb) * (mu * (1 - mu) - var)              # shared Bernoulli draws
            C[a, b] = C[b, a] = v
    clark = _clark_min(np.array(means), C)
    return dict(upper=upper, lower=lower, t05=t05, clark=clark, means=means)


def _clark_min(mu, C):
    """Clark (1961) moment-matching approximation to E[min] of a Gaussian vector (applied to -X for the max)."""
    m = -mu.copy(); order = np.argsort(m)[::-1]
    m1, v1 = m[order[0]], C[order[0], order[0]]
    cov_run = C[order[0]].copy()          # covariance of running max with each component (sign-flipped vector: same C)
    for j in order[1:]:
        m2, v2, c12 = m[j], C[j, j], cov_run[j]
        a2 = max(v1 + v2 - 2 * c12, 1e-14); a = math.sqrt(a2); al = (m1 - m2) / a
        P, p = norm.cdf(al), norm.pdf(al)
        nm1 = m1 * P + m2 * (1 - P) + a * p
        nm2 = (m1 ** 2 + v1) * P + (m2 ** 2 + v2) * (1 - P) + (m1 + m2) * a * p
        cov_run = cov_run * P + C[j] * (1 - P)
        m1, v1 = nm1, max(nm2 - nm1 ** 2, 0.0)
    return float(-m1)
