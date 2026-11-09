"""Sensitivity of weighted split CP to a misspecified domain classifier (synthetic; not BRFSS evidence).
Source X ~ N(0,1); target X ~ N(mu, s^2) so the true log density ratio is quadratic in x.
Domain classifier: logistic regression on x only (misspecified) vs on (x, x^2) (correctly specified),
fitted on independent source / unlabelled-target folds with the fold-size offset, as in SAGE-CP. Clip at the
0.99 quantile of domain-fold weights. Score s = sigmoid(x + e)."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
ALPHA = 0.1
def weighted_q(cal_scores, cal_w, test_w, alpha):
    o = np.argsort(cal_scores); s = cal_scores[o]; cw = np.cumsum(cal_w[o]); W = cw[-1]
    k = np.searchsorted(cw, (1 - alpha) * (W + test_w), side="left")
    return np.where(k < len(s), s[np.minimum(k, len(s) - 1)], np.inf)
rng = np.random.default_rng(7)
rows = []
for mu, sd in [(0.5, 1.0), (0.5, 1.5), (0.0, 1.8)]:
    for rep in range(200):
        n = 2000
        xd = rng.normal(size=n); xu = rng.normal(mu, sd, size=n)            # domain fold, unlabelled target
        xc = rng.normal(size=n); sc = 1 / (1 + np.exp(-(xc + rng.normal(size=n))))
        xt = rng.normal(mu, sd, size=n); st = 1 / (1 + np.exp(-(xt + rng.normal(size=n))))
        true_w = lambda x: norm.pdf(x, mu, sd) / norm.pdf(x)
        res = {}
        for name, feat in [("linear (misspecified)", lambda x: x[:, None]), ("quadratic (correct)", lambda x: np.c_[x, x ** 2])]:
            clf = LogisticRegression(C=0.1, max_iter=1000).fit(np.r_[feat(xd), feat(xu)], np.r_[np.zeros(n), np.ones(n)])
            ew = lambda x: np.exp(np.clip(clf.decision_function(feat(x)), -30, 30))
            c = np.quantile(ew(xd), 0.99)
            res[name] = (np.minimum(ew(xc), c), np.minimum(ew(xt), c))
        res["true ratio"] = (true_w(xc), true_w(xt)); res["unweighted"] = (np.ones(n), np.ones(n))
        for name, (wc, wt) in res.items():
            q = weighted_q(sc, wc, wt, ALPHA)
            rows.append(dict(target=f"N({mu}, {sd}²)", weights=name, coverage=float((st <= q).mean()),
                             ess_frac=float(wc.sum() ** 2 / (wc ** 2).sum() / n), inf_share=float(np.isinf(q).mean())))
df = pd.DataFrame(rows)
s = df.groupby(["target", "weights"]).agg(coverage=("coverage", "mean"), mc_se=("coverage", lambda v: v.std() / np.sqrt(len(v))),
                                          ess_over_n=("ess_frac", "mean"), share_infinite_threshold=("inf_share", "mean")).round(4)
s.to_csv("misspecification_results.csv"); print(s.to_string())
