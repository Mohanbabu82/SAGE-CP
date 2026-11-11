# ======================= 6. Analysis (pre-specified contrasts; design-based and unit-bootstrap uncertainty) =======================
def _cat(pattern):
    fs = sorted(glob.glob(f"{UNIT_DIR}/{pattern}"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
_all = sorted(f for f in glob.glob(f"{UNIT_DIR}/*.csv") if os.path.basename(f).count(".") == 1)
U = pd.concat([pd.read_csv(f) for f in _all], ignore_index=True) if _all else pd.DataFrame()
BT, CT, GL = _cat("*.boot.csv"), _cat("*.cutoff.csv"), _cat("*.gibbs.csv")
for nm, d in [("units", U), ("boot_design_replicates", BT), ("cutoff_rules", CT), ("gibbs_selection", GL)]:
    if len(d): d.to_csv(f"{OUT}/{nm}.csv.gz" if nm == "boot_design_replicates" else f"{OUT}/{nm}.csv", index=False)
check("v4 results present", len(U) > 0)
CONTRASTS = [
    ("GroupCond - ClassCond", "GroupCond", "ClassCond", "wg_cal_pos", "+"), ("GroupCond - ClassCond", "GroupCond", "ClassCond", "set_size", "-"),
    ("GroupCond - ClassCond", "GroupCond", "ClassCond", "wg_race_pos", "+"), ("SAGE-CP - GroupCond", "SAGE-CP", "GroupCond", "absdev_cov_pos", "-"),
    ("SAGE-CP - GroupCond", "SAGE-CP", "GroupCond", "wg_cal_pos", "+"), ("SAGE-CP - GroupCond", "SAGE-CP", "GroupCond", "set_size", "-"),
    ("Weighted+ClassCond - ClassCond", "Weighted+ClassCond", "ClassCond", "absdev_cov_pos", "-"),
    ("Gibbs-tuned - GroupCond", "Gibbs-tuned", "GroupCond", "wg_cal_pos", "+"), ("Gibbs-tuned - GroupCond", "Gibbs-tuned", "GroupCond", "absdev_cov_pos", "-"),
    ("Gibbs-tuned - GroupCond", "Gibbs-tuned", "GroupCond", "set_size", "-"),
]
METRICS = ["coverage", "cov_pos", "absdev_cov_pos", "set_size", "both", "zero", "single1", "miss0", "npv0", "ppv_single1", "wg_cal_pos",
           "wg_race_pos", "n_elig_cal", "n_elig_race", "auc_target", "ref_exact_mean", "ref_exact_p05", "ref_ess_mean", "ref_ess_p05", "n", "n_pos"]
def holm(p):
    p = np.asarray(p, float); o = np.argsort(np.where(np.isnan(p), np.inf, p)); m = np.sum(~np.isnan(p)); adj = np.full(len(p), np.nan); run = 0
    for r, i in enumerate(o):
        if np.isnan(p[i]): continue
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj
if len(U):
    L = U[U.study.isin(["RAWP", "RAWP_MISIND", "RAWP_MI", "RAWP_FULL", "RAWP_CB", "LOSO_G"])].copy()
    L["unit"] = L.heldout.astype(str)
    # unit level: average over seeds, models and imputations (Rubin's point estimate for RAWP_MI)
    UL = L.groupby(["study", "weighting", "outcome", "unit", "method"], as_index=False)[[c for c in METRICS if c in L]].mean(numeric_only=True)
    UP = UL.groupby(["study", "weighting", "unit", "method"], as_index=False).mean(numeric_only=True).assign(outcome="pooled")
    UL = pd.concat([UL, UP], ignore_index=True); UL.to_csv(f"{OUT}/v4_unit_level.csv", index=False)
    rb = np.random.default_rng(CFG3["BOOT_SEED"]); res = []
    for (st, wt, oc), d in UL.groupby(["study", "weighting", "outcome"]):
        for name, A, B_, met, better in CONTRASTS:
            diff = (d[d.method == A].set_index("unit")[met] - d[d.method == B_].set_index("unit")[met]).dropna()
            if len(diff) < 2: continue
            n = len(diff); bs = rb.choice(diff.to_numpy(), size=(CFG3["BOOT_STATES"], n)).mean(1); nz = diff[diff != 0]
            res.append(dict(study=st, weighting=wt, outcome=oc, contrast=name, metric=met, better=better, n_units=n, mean_diff=diff.mean(),
                            median_diff=diff.median(), ci95_lo=np.quantile(bs, .025), ci95_hi=np.quantile(bs, .975),
                            n_A_better=int(((diff > 0) if better == "+" else (diff < 0)).sum()), n_ties=int((diff == 0).sum()),
                            p_wilcoxon=wilcoxon(diff).pvalue if len(nz) else np.nan,
                            p_sign=binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else np.nan))
    C = pd.DataFrame(res, columns=["study", "weighting", "outcome", "contrast", "metric", "better", "n_units", "mean_diff", "median_diff", "ci95_lo", "ci95_hi",
                                   "n_A_better", "n_ties", "p_wilcoxon", "p_sign"]); C["p_wilcoxon_holm"] = np.nan; C["holm_family_size"] = 0
    for _, ix in C.groupby(["study", "weighting", C.outcome == "pooled"]).groups.items():
        C.loc[ix, "p_wilcoxon_holm"] = holm(C.loc[ix, "p_wilcoxon"]); C.loc[ix, "holm_family_size"] = int(C.loc[ix, "p_wilcoxon"].notna().sum())
    # design-based SE of every pooled RAWP contrast (Rao-Wu replicates, seed 0; units and outcomes averaged within replicate)
    if len(BT):
        bc = [c for c in BT.columns if c.startswith("b") and c[1:].isdigit()]
        BTN = BT[BT.study.isin(["NHANES1", "NHANES2"])]
        if len(BTN):   # NHANES: replicate means over seeds (and models), then contrasts between methods
            g = BTN.groupby(["study", "eval_outcome", "model", "method", "metric"])[bc].mean()
            g.to_csv(f"{OUT}/v4_nhanes_replicates.csv")
        BT = BT[BT.study == "RAWP"]
        dz = []
        for name, A, B_, met, better in CONTRASTS:
            a = BT[(BT.method == A) & (BT.metric == met)].set_index(["heldout", "outcome"])[bc]
            b = BT[(BT.method == B_) & (BT.metric == met)].set_index(["heldout", "outcome"])[bc]
            if not len(a) or not len(b): continue
            dd = (a - b).dropna()
            for oc, sub in [("pooled", dd)] + [(o, dd.xs(o, level="outcome")) for o in dd.index.get_level_values("outcome").unique()]:
                rep = sub.groupby(level=0).mean().mean(0) if oc == "pooled" else sub.mean(0)
                dz.append(dict(study="RAWP", outcome=oc, contrast=name, metric=met, design_se=float(rep.std(ddof=1)),
                               design_lo=float(np.quantile(rep, .025)), design_hi=float(np.quantile(rep, .975)), n_replicates=len(bc)))
        DZ = pd.DataFrame(dz); DZ.to_csv(f"{OUT}/v4_design_se.csv", index=False)
        if len(DZ): C = C.merge(DZ.assign(weighting="design"), on=["study", "weighting", "outcome", "contrast", "metric"], how="left")
    C.to_csv(f"{OUT}/v4_contrasts.csv", index=False)
    SM = UL.groupby(["study", "weighting", "outcome", "method"]).mean(numeric_only=True); SM.to_csv(f"{OUT}/v4_summary_by_method.csv")
    # 17-state style check for RAWP: does the GroupCond gain follow state size and the number of eligible groups?
    d = UL[(UL.study == "RAWP") & (UL.weighting == "design") & (UL.outcome == "pooled")]
    if len(d):
        a = d[d.method == "GroupCond"].set_index("unit"); b = d[d.method == "ClassCond"].set_index("unit")
        from scipy.stats import spearmanr
        dg = a.wg_cal_pos - b.wg_cal_pos
        pd.DataFrame([dict(n_units=len(dg), n_helped=int((dg > 0).sum()), rho_n=spearmanr(dg, a.n).statistic,
                           rho_n_elig=spearmanr(dg, a.n_elig_cal).statistic, rho_classcond_wg=spearmanr(dg, b.wg_cal_pos).statistic)]).to_csv(f"{OUT}/v4_gc_gain_vs_size.csv", index=False)
    # NHANES summaries (seeds and base models averaged)
    N = U[U.study.isin(["NHANES1", "NHANES2"])]
    if len(N):
        cols = [c for c in ["coverage", "cov_pos", "set_size", "both", "zero", "single1", "miss0", "npv0", "ppv_single1", "wg_cal_pos",
                            "cov1_undiag", "only1_undiag", "zero_undiag", "cov1_diag", "auc_target", "ref_exact_mean", "n", "n_pos"] if c in N]
        N.groupby(["study", "eval_outcome", "weighting", "method"])[cols].mean().to_csv(f"{OUT}/v4_nhanes_summary.csv")
        N.groupby(["study", "eval_outcome", "weighting", "model", "method"])[cols].mean().to_csv(f"{OUT}/v4_nhanes_by_model.csv")
    if len(CT):
        CT.groupby(["study", "eval_outcome", "rule"]).mean(numeric_only=True).to_csv(f"{OUT}/v4_cutoff_summary.csv")
    W = U[U.study == "WEST"]
    if len(W):
        W.groupby(["region", "outcome"]).agg(prev_target=("prev_target", "mean"), prev_source=("prev_source", "mean"),
              prev_wsource_clipped=("prev_wsource_clipped", "mean"), oe=("oe", "mean"), auc_source=("auc_source", "mean"),
              auc_target=("auc_target", "mean"), lr_stat=("lr_stat", "median"), share_lr_p_below_001=("lr_p", lambda x: float((x < 0.001).mean())),
              region_intercept=("region_intercept", "mean"), region_slope=("region_slope", "mean"),
              max_decile_gap=("max_decile_gap", "mean")).to_csv(f"{OUT}/v4_west_concept_shift.csv")
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(C[C.outcome == "pooled"].round(4).to_string())
pd.DataFrame(CHECKS).to_csv(f"{OUT}/checks_v4.csv", index=False)
zp = f"{OUT}/../SAGE_CP_v4_results.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(f"{OUT}/**/*", recursive=True):
        if os.path.isfile(f) and not f.endswith(".zip"): z.write(f, os.path.relpath(f, OUT))
print("results zipped to", os.path.abspath(zp), "| elapsed min", round((time.time() - T0) / 60, 1))
