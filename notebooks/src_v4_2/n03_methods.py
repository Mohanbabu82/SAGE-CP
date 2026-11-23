# ======================= 3. Models, conformal methods, metrics and the exact worst-group reference =======================
# weighted_q, conformal_sets, sage_sets, kish, _qr_lp, fit_domain, weights_from and the reference are VERBATIM from v2.1.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack as sp_hstack, identity as sp_identity
from scipy.stats import wilcoxon, binomtest

MODEL_BACKEND = {}
def make_model(name, seed):
    """Same settings as v2.1 (Table S4)."""
    if name == "LR":
        MODEL_BACKEND[name] = "sklearn LogisticRegression"
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0))
    try:
        import lightgbm as lgb; MODEL_BACKEND[name] = f"lightgbm {lgb.__version__}"
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, random_state=seed, verbose=-1)
    except ImportError:
        MODEL_BACKEND[name] = "FALLBACK sklearn HistGradientBoosting"
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)

def make_encoder(num, cat):
    return ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), cat)],
                             remainder="drop", verbose_feature_names_out=False)

def weighted_q(cal_scores, cal_w, test_w, alpha):
    o = np.argsort(cal_scores); s = cal_scores[o]; cw = np.cumsum(cal_w[o]); W = cw[-1]
    k = np.searchsorted(cw, (1 - alpha) * (W + test_w), side="left")
    return np.where(k < len(s), s[np.minimum(k, len(s) - 1)], np.inf)

def conformal_sets(p_cal, y_cal, p_te, alpha, w_cal=None, w_te=None, class_cond=False):
    n_te = len(p_te)
    w_cal = np.ones(len(p_cal)) if w_cal is None else w_cal
    w_te = np.ones(n_te) if w_te is None else w_te
    P_te, P_cal = np.c_[1 - p_te, p_te], np.c_[1 - p_cal, p_cal]
    S = np.zeros((n_te, 2), bool)
    if class_cond:
        for c in (0, 1):
            m = y_cal == c
            S[:, c] = (1 - P_te[:, c]) <= (weighted_q(1 - P_cal[m, c], w_cal[m], w_te, alpha) if m.any() else np.inf)
    else:
        q = weighted_q(1 - P_cal[np.arange(len(y_cal)), y_cal], w_cal, w_te, alpha)
        S = (1 - P_te) <= q[:, None]
    return S

def sage_sets(p_cal, y_cal, keys_cal, p_te, keys_te, alpha, w_cal=None, w_te=None, min_ess=100):
    n_te = len(p_te)
    w_cal = np.ones(len(p_cal)) if w_cal is None else w_cal
    w_te = np.ones(n_te) if w_te is None else w_te
    P_te, P_cal = np.c_[1 - p_te, p_te], np.c_[1 - p_cal, p_cal]
    S = np.zeros((n_te, 2), bool)
    for c in (0, 1):
        mc = y_cal == c; done = np.zeros(n_te, bool)
        for L, (kc, kt) in enumerate(zip(keys_cal, keys_te)):
            last = L == len(keys_cal) - 1
            for k in np.unique(kt[~done]):
                ti = np.where(~done & (kt == k))[0]
                cm = mc & (kc == k); wc = w_cal[cm]
                ess = wc.sum() ** 2 / (wc ** 2).sum() if cm.any() else 0.0
                if ess < min_ess and not last: continue
                q = weighted_q(1 - P_cal[cm, c], wc, w_te[ti], alpha) if cm.any() else np.full(len(ti), np.inf)
                S[ti, c] = (1 - P_te[ti, c]) <= q
                done[ti] = True
    return S

def kish(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) and (w ** 2).sum() > 0 else 0.0

def _qr_lp(Xd, s, w, tau):
    n, d = Xd.shape
    c = np.r_[np.zeros(d), tau * w, (1 - tau) * w]
    A = sp_hstack([csr_matrix(Xd), sp_identity(n), -sp_identity(n)]).tocsr()
    res = linprog(c, A_eq=A, b_eq=s, bounds=[(None, None)] * d + [(0, None)] * (2 * n), method="highs")
    if not res.success: raise RuntimeError("quantile-regression LP failed: " + res.message)
    return res.x[:d]

def fit_domain(Xs, Xt):
    Xd = np.r_[Xs, Xt]; yd = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=CFG3["DOMAIN_C"])).fit(Xd, yd)
    m.log_prior_ = np.log(len(Xs) / len(Xt))
    return m

def weights_from(m, Xq):
    return np.exp(np.clip(m.decision_function(Xq) + m.log_prior_, -30, 30))

# ---------------- tuned conditional calibration (Gibbs, Cherian & Candes, JRSSB 2025) ----------------
# Function class F = span of indicator features: intercept, Sex x Age cells, optionally race, optionally B bins of the
# estimated density ratio (bins = quantiles on s_dom, label-free). For a finite indicator class the augmented
# quantile-regression threshold g_S(x) depends on x only through its pattern (cell, race, bin). We use the
# deterministic conservative form: the test score is imputed at its maximum (S = 1 for the LAC score), so the set
# {y : S(x, y) <= g_1(x)} contains the exact non-randomised Gibbs set and inherits its coverage guarantee
# (P(Y in C(X) | X in group) >= 1 - alpha for every group in F; weighted versions for shifts in the span of F).
# Fitted per label (label-conditional), one LP per (label, pattern present in the evaluation data).
def _patterns(G, cols):
    return pd.Series(list(map(tuple, G[cols].to_numpy())))

def gibbs_thresholds(s_cal, D_cal, D_pat, tau, grid):
    """s_cal: calibration scores of one label; D_cal: their indicator rows; D_pat: one indicator row per pattern.
    Returns threshold per pattern (augmented fit with the pattern's test point at score 1)."""
    s = np.ceil(s_cal / grid) * grid
    key = pd.DataFrame(D_cal).astype(np.int8).astype(str).agg("".join, axis=1) + "|" + pd.Series(s).round(6).astype(str)
    comp = pd.DataFrame({"k": key, "i": np.arange(len(s))}).groupby("k")["i"].agg(["first", "size"])
    Dc, sc, wc = D_cal[comp["first"].to_numpy()], s[comp["first"].to_numpy()], comp["size"].to_numpy(float)
    act = D_cal.sum(0) > 0
    th = np.empty(len(D_pat))
    for j, dp in enumerate(D_pat):
        a = act | (dp > 0)
        beta = _qr_lp(np.r_[Dc, dp[None, :]][:, a], np.r_[sc, 1.0], np.r_[wc, 1.0], tau)
        th[j] = float(dp[a] @ beta)
    return th

def gibbs_sets(p_cal, y_cal, Gc, p_ev_list, G_ev_list, cols, alpha, grid):
    """cols: indicator columns of the function class (Gc / G_ev: DataFrames of 0/1 indicators incl. 'one')."""
    outs = [np.zeros((len(p), 2), bool) for p in p_ev_list]
    pats = pd.concat([G[cols] for G in G_ev_list]).drop_duplicates().to_numpy(float)
    pidx = {tuple(r): j for j, r in enumerate(pats)}
    for c in (0, 1):
        m = y_cal == c; n = int(m.sum())
        if n == 0:
            for S in outs: S[:, c] = True
            continue
        tau = 1 - alpha                                    # augmented fit: no finite-sample correction needed
        sc = 1 - (p_cal[m] if c == 1 else 1 - p_cal[m])
        th = gibbs_thresholds(sc, Gc[cols].to_numpy(float)[m], pats, tau, grid)
        for S, p, G in zip(outs, p_ev_list, G_ev_list):
            j = np.array([pidx[tuple(r)] for r in G[cols].to_numpy(float)])
            sc_ev = 1 - (p if c == 1 else 1 - p)
            S[:, c] = sc_ev <= th[j] + 1e-9
    return outs

# ---------------- metrics (optionally weighted by the BRFSS design weight) ----------------
def _wmean(x, w):
    return float(np.sum(w * x) / np.sum(w)) if len(x) and np.sum(w) > 0 else np.nan

def metrics(S, y, G, w=None, eligible=None):
    """Coverage, positive-class coverage, set size, share of {0,1}, empty and singleton-{1} rates, and the minimum
    positive-class coverage over ELIGIBLE groups (Sex, Age, SexxAge: 'cal'; Race: 'race'). Eligibility (MIN_GROUP rows and
    MIN_POS positives in the evaluation set) is decided on counts, so weighted and unweighted metrics use the same groups."""
    w = np.ones(len(y)) if w is None else w
    cov = S[np.arange(len(y)), y].astype(float); size = S.sum(1); pos = y == 1
    out = dict(n=len(y), n_pos=int(pos.sum()), coverage=_wmean(cov, w), cov_pos=_wmean(cov[pos], w[pos]),
               cov_neg=_wmean(cov[~pos], w[~pos]), set_size=_wmean(size, w), both=_wmean((size == 2).astype(float), w),
               empty=_wmean((size == 0).astype(float), w), single1=_wmean((S[:, 1] & ~S[:, 0]).astype(float), w),
               ppv_single1=_wmean(y[S[:, 1] & ~S[:, 0]].astype(float), w[S[:, 1] & ~S[:, 0]]))
    out["absdev_cov_pos"] = abs(out["cov_pos"] - (1 - ALPHA)) if not np.isnan(out["cov_pos"]) else np.nan
    for name, axes in [("cal", ["Sex", "Age", "SexxAge"]), ("race", ["Race"])]:
        vals = []
        for ax in axes:
            for g, ix in G.groupby(ax).indices.items():
                if len(ix) >= CFG3["MIN_GROUP"] and int(y[ix].sum()) >= CFG3["MIN_POS"]:
                    pi = ix[y[ix] == 1]; vals.append(_wmean(cov[pi], w[pi]))
        out[f"wg_{name}_pos"] = float(np.min(vals)) if vals else np.nan
        out[f"n_elig_{name}"] = len(vals)
    return out

def _beta_pi(ncell, alpha, sims, rng):
    n = int(ncell)
    if n <= 0: return np.ones(sims)
    k = math.ceil((1 - alpha) * (n + 1))
    return np.ones(sims) if k > n else rng.beta(k, n + 1 - k, size=sims)

def exact_reference(y_ev, G_ev, cal_n, alpha, sims, rng):
    """Proposition 3: law of the minimum positive-class coverage over eligible calibrated groups under label x cell split
    CP with cal_n[cell] calibration positives (exact for continuous scores; cells without positives use the Age parent).
    Returns mean and 5th percentile."""
    pos = np.where(y_ev == 1)[0]
    if not len(pos): return np.nan, np.nan
    cell = G_ev["SexxAge"].to_numpy(str)[pos]
    def n_eff(c):
        v = cal_n.get(c, 0)
        return v if v > 0 else sum(x for k, x in cal_n.items() if k.split("|")[-1] == c.split("|")[-1])
    cells = sorted(set(cell)); ci = np.searchsorted(np.array(cells), cell)
    pi = np.column_stack([_beta_pi(n_eff(c), alpha, sims, rng) for c in cells])[:, ci]
    M = []
    for ax in ["Sex", "Age", "SexxAge"]:
        for g, ix in G_ev.groupby(ax).indices.items():
            if len(ix) >= CFG3["MIN_GROUP"] and int(y_ev[ix].sum()) >= CFG3["MIN_POS"]:
                v = np.zeros(len(y_ev), bool); v[ix] = True; M.append(v[pos])
    if not M: return np.nan, np.nan
    M = np.array(M, np.float32).T
    Z = (rng.random((sims, len(pos))) < pi).astype(np.float32)
    mn = ((Z @ M) / M.sum(0)).min(1)
    return float(mn.mean()), float(np.quantile(mn, 0.05))

def gibbs_unaug_sets(p_cal, y_cal, Gc, p_ev, G_ev, cols, alpha, grid):
    """Unaugmented (split) fit used ONLY to select the function class on source data: one LP per label at level
    (1-alpha)(n+1)/n; threshold = D(x) beta."""
    S = np.zeros((len(p_ev), 2), bool)
    for c in (0, 1):
        m = y_cal == c; n = int(m.sum())
        tau = 1.0 if n == 0 else min(1.0, (1 - alpha) * (n + 1) / n)
        if tau >= 1.0: S[:, c] = True; continue
        sc = np.ceil((1 - (p_cal[m] if c == 1 else 1 - p_cal[m])) / grid) * grid
        D = Gc[cols].to_numpy(float)[m]
        key = pd.DataFrame(D).astype(np.int8).astype(str).agg("".join, axis=1) + "|" + pd.Series(sc).round(6).astype(str)
        comp = pd.DataFrame({"k": key, "i": np.arange(len(sc))}).groupby("k")["i"].agg(["first", "size"])
        act = D.sum(0) > 0; beta = np.zeros(D.shape[1])
        beta[act] = _qr_lp(D[comp["first"].to_numpy()][:, act], sc[comp["first"].to_numpy()], comp["size"].to_numpy(float), tau)
        S[:, c] = (1 - (p_ev if c == 1 else 1 - p_ev)) <= G_ev[cols].to_numpy(float) @ beta + 1e-9
    return S

# ======================= v4 additions =======================
from sklearn.calibration import CalibratedClassifierCV
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
_make_model_v3 = make_model
def make_model(name, seed):
    """v3 models plus CBCAL: CatBoost (500 trees, depth 6) with isotonic calibration (3-fold, on s_train only)."""
    if name != "CBCAL": return _make_model_v3(name, seed)
    try:
        from catboost import CatBoostClassifier; import catboost
        base = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, random_seed=seed, verbose=0, thread_count=-1)
        MODEL_BACKEND[name] = f"catboost {catboost.__version__} + isotonic"
    except ImportError:
        base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
        MODEL_BACKEND[name] = "FALLBACK sklearn HistGradientBoosting + isotonic"
    return CalibratedClassifierCV(base, method="isotonic", cv=3)

def make_encoder_v(num, cat, variant="median"):
    """median: v3 encoder. misind: median imputation plus a missing indicator for each numeric feature with NaNs."""
    if variant == "misind":
        return ColumnTransformer([("num", SimpleImputer(strategy="median", add_indicator=True), num),
                                  ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), cat)],
                                 remainder="drop", verbose_feature_names_out=False)
    return make_encoder(num, cat)

class MIEncoder:
    """Multiple imputation by chained equations for the numeric features (Bayesian ridge, posterior draws), with the
    one-hot categorical features (missing = its own level) as predictors. The outcome is NOT used, because it is unknown
    when a prediction set is formed. Fitted on (a random subset of) s_train only; draw j uses random_state = j."""
    def __init__(self, num, cat, j, fit_rows, max_iter):
        self.num, self.cat, self.j, self.fit_rows, self.max_iter = num, cat, j, fit_rows, max_iter
    def fit(self, X):
        self.oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32).fit(X[self.cat])
        Z = self._z(X); rs = np.random.default_rng(self.j)
        sub = Z if len(Z) <= self.fit_rows else Z[rs.choice(len(Z), self.fit_rows, replace=False)]
        self.imp = IterativeImputer(sample_posterior=True, max_iter=self.max_iter, random_state=self.j, n_nearest_features=15).fit(sub)
        return self
    def _z(self, X): return np.c_[X[self.num].to_numpy(float), self.oh.transform(X[self.cat])]
    def transform(self, X): return self.imp.transform(self._z(X)).astype(np.float32)

def metrics4(S, y, G, w=None):
    """v3 metrics plus the triage quantities: share ruled out ({0}), missed positives inside {0} and NPV of {0}."""
    out = metrics(S, y, G, w)
    w = np.ones(len(y)) if w is None else w
    zero = S[:, 0] & ~S[:, 1]
    out.update(zero=_wmean(zero.astype(float), w), miss0=_wmean((zero & (y == 1)).astype(float), w),
               npv0=_wmean((y[zero] == 0).astype(float), w[zero]))
    return out

def cutoff_rules(p_cal, y_cal, p_ev, y_ev, w):
    """Single risk cut-offs chosen on the calibration data: sensitivity 0.90 (10th percentile of the positives' risk),
    and Youden's J. Per 1,000 evaluated adults (design-weighted): flagged, ruled out, missed positives, PPV, NPV."""
    from sklearn.metrics import roc_curve
    t90 = float(np.quantile(p_cal[y_cal == 1], ALPHA)) if (y_cal == 1).any() else 0.5
    fpr, tpr, th = roc_curve(y_cal, p_cal); tY = float(th[np.argmax(tpr - fpr)])
    res = {}
    for nm, t in [("cut_sens90", t90), ("cut_youden", tY)]:
        f = p_ev >= t
        res[nm] = dict(threshold=t, flagged=_wmean(f.astype(float), w), ruled_out=_wmean((~f).astype(float), w),
                       missed=_wmean((~f & (y_ev == 1)).astype(float), w), ppv=_wmean(y_ev[f].astype(float), w[f]),
                       npv=_wmean((y_ev[~f] == 0).astype(float), w[~f]), sens=_wmean(f[y_ev == 1].astype(float), w[y_ev == 1]))
    return res

def rw_multipliers(strata, psu, psu_by_stratum, B, seed):
    """Rao-Wu rescaled bootstrap: in each stratum with n_h PSUs draw n_h - 1 PSUs with replacement; multiplier
    n_h/(n_h - 1) x count. Draws are seeded by stratum, so every analysis of the same respondents uses the same
    replicates (paired contrasts) and replicates are independent across strata (and so across states)."""
    M = np.ones((B, len(psu)), np.float32)
    for h in np.unique(strata):
        ix = np.where(strata == h)[0]; P = psu_by_stratum[h]; nh = len(P)
        if nh < 2: continue
        cnt = np.random.default_rng([seed, int(h)]).multinomial(nh - 1, np.full(nh, 1 / nh), size=B) * (nh / (nh - 1))
        col = np.searchsorted(P, psu[ix]); M[:, ix] = cnt[:, col]
    return M

def rep_metrics(S, y, G, W):
    """Design-weighted metrics under each replicate weight vector (rows of W); eligibility fixed by the full sample."""
    cov = S[np.arange(len(y)), y].astype(float); size = S.sum(1).astype(float); pos = y == 1
    r = dict(cov_pos=(W[:, pos] @ cov[pos]) / W[:, pos].sum(1), set_size=(W @ size) / W.sum(1),
             both=(W @ (size == 2).astype(float)) / W.sum(1))
    r["absdev_cov_pos"] = np.abs(r["cov_pos"] - (1 - ALPHA))
    vals = []
    for ax in ["Sex", "Age", "SexxAge"]:
        for g, ix in G.groupby(ax).indices.items():
            if len(ix) >= CFG3["MIN_GROUP"] and int(y[ix].sum()) >= CFG3["MIN_POS"]:
                pi = ix[y[ix] == 1]; vals.append((W[:, pi] @ cov[pi]) / W[:, pi].sum(1))
    r["wg_cal_pos"] = np.min(np.array(vals), 0) if vals else np.full(len(W), np.nan)
    return r
