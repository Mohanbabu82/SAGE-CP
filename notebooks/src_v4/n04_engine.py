# ======================= 4. Engine v4 (same disjoint partitions and train-only preprocessing as v3) =======================
UNIT_DIR, DONE_JSON = f"{OUT}/units", f"{OUT}/done_units.json"; os.makedirs(UNIT_DIR, exist_ok=True)
DONE = set(map(tuple, json.load(open(DONE_JSON)))) if os.path.exists(DONE_JSON) else set()
STUDY_IX = {s: i for i, s in enumerate(["RAWP", "RAWP_MISIND", "RAWP_MI", "RAWP_FULL", "RAWP_CB", "LOSO_G", "WEST", "NHANES1", "NHANES2"])}
PSU_BY_STRATUM = {}   # filled in Section 5 from the full file (all respondents of each stratum)

def _save(rows, path):
    if rows: pd.DataFrame(rows).to_csv(path, index=False)

def _cap(idx, n, rng): return idx if len(idx) <= n else rng.choice(idx, n, replace=False)

def gframe(F, rows):
    G = F.iloc[rows][["Sex", "Age", "Race"]].reset_index(drop=True).astype(str)
    G["SexxAge"] = G["Sex"] + "|" + G["Age"]; return G

SA_CATS = [f"{s}|{a}" for s in ["Female", "Male"] for a in ["18-44", "45-64", "65+"]]
def indicators(G, wraw, race_cats, edges):
    D = pd.DataFrame({f"SA={c}": (G["SexxAge"] == c).astype(float).to_numpy() for c in SA_CATS})
    for r in race_cats[1:]: D[f"R={r}"] = (G["Race"] == r).astype(float).to_numpy()
    if edges is not None:
        b = np.digitize(wraw, edges)
        for k in range(1, len(edges) + 1): D[f"W={k}"] = (b == k).astype(float)
    return D

def _mark_done(key, tag, rows, extra=None):
    _save(rows, f"{UNIT_DIR}/{tag}.csv")
    for suffix, r in (extra or {}).items(): _save(r, f"{UNIT_DIR}/{tag}.{suffix}.csv")
    DONE.add(key); json.dump(sorted(map(list, DONE)), open(DONE_JSON, "w"))

def build_sets(pc, yc, kc, pe, ke, wc, we, pu=None, yu=None, ku=None):
    SETS = {"ClassCond": conformal_sets(pc, yc, pe, ALPHA, class_cond=True),
            "GroupCond": sage_sets(pc, yc, kc, pe, ke, ALPHA, min_ess=CFG3["MIN_ESS"])}
    if wc is not None:
        SETS["Weighted+ClassCond"] = conformal_sets(pc, yc, pe, ALPHA, w_cal=wc, w_te=we, class_cond=True)
        SETS["SAGE-CP"] = sage_sets(pc, yc, kc, pe, ke, ALPHA, w_cal=wc, w_te=we, min_ess=CFG3["MIN_ESS"])
    if pu is not None:
        SETS["TargetOracle"] = sage_sets(pu, yu, ku, pe, ke, ALPHA, min_ess=CFG3["MIN_ESS"])
    for a2 in CFG3["EXTRA_ALPHAS"]:
        SETS[f"ClassCond@{a2:.2f}"] = conformal_sets(pc, yc, pe, a2, class_cond=True)
        SETS[f"GroupCond@{a2:.2f}"] = sage_sets(pc, yc, kc, pe, ke, a2, min_ess=CFG3["MIN_ESS"])
        if wc is not None:
            SETS[f"SAGE-CP@{a2:.2f}"] = sage_sets(pc, yc, kc, pe, ke, a2, w_cal=wc, w_te=we, min_ess=CFG3["MIN_ESS"])
    return SETS

def gibbs_tuned(pc, yc, g, w, pr, y, race_cats, WR_dom, rng, glog, tagd):
    GB = CFG3["GIBBS"]; tg = time.time()
    sub = np.r_[_cap(np.where(yc == 0)[0], GB["n_cal_max"], rng), _cap(np.where(yc == 1)[0], GB["n_cal_max"], rng)]
    cand = []
    for B in GB["wbins"]:
        edges = None if B == 0 else np.quantile(WR_dom, np.linspace(0, 1, B + 1)[1:-1])
        for use_race in GB["race"]:
            rc = race_cats if use_race else race_cats[:1]
            Dc = indicators(g["s_cal"].iloc[sub].reset_index(drop=True), w["s_cal"][1][sub], rc, edges)
            Ds = indicators(g["s_test"], w["s_test"][1], rc, edges); cols = list(Dc.columns)
            S_st = gibbs_unaug_sets(pc[sub], yc[sub], Dc, pr["s_test"], Ds, cols, ALPHA, GB["score_grid"])
            est = metrics(S_st, y["s_test"], g["s_test"], w=w["s_test"][1])
            short = max(0.0, (1 - ALPHA) - est["wg_cal_pos"]) if not np.isnan(est["wg_cal_pos"]) else np.inf
            cand.append(dict(B=B, race=use_race, edges=edges, cols=cols, short=short, size=est["set_size"]))
            glog.append(dict(**tagd, wbins=B, race=use_race, n_cols=len(cols), est_wg_cal_pos=est["wg_cal_pos"],
                             est_shortfall=short, est_set_size=est["set_size"]))
    best = min(c["short"] for c in cand)
    sel = min([c for c in cand if c["short"] <= best + GB["tie_tol"]], key=lambda c: c["size"])
    rc = race_cats if sel["race"] else race_cats[:1]
    Dc = indicators(g["s_cal"].iloc[sub].reset_index(drop=True), w["s_cal"][1][sub], rc, sel["edges"])
    De = indicators(g["t_eval"], w["t_eval"][1], rc, sel["edges"])
    S = gibbs_sets(pc[sub], yc[sub], Dc, [pr["t_eval"]], [De], sel["cols"], ALPHA, GB["score_grid"])[0]
    for r in glog[-len(cand):]: r["selected"] = (r["wbins"] == sel["B"]) and (r["race"] == sel["race"])
    return S, dict(gibbs_wbins=sel["B"], gibbs_race=sel["race"], gibbs_seconds=time.time() - tg)

def run_unit4(study, unit, seed, Fs, Xs, src, Ft, Xt, tgt, num, cat, unit_index=0, variant="median", gibbs=False, boot=False,
              src_outcomes=None, eval_outcomes=None, cutoffs=False, eval_by=None):
    """Leave-one-unit-out run (v3 partitions). variant: median | misind | mi (m = MI_M imputations, results per imputation).
    src_outcomes/eval_outcomes: outcome used to train and calibrate, and the outcome(s) it is evaluated against
    (NHANES1: train on BRFSS self-report, evaluate on NHANES diagnosed and total diabetes)."""
    key = (study, str(unit), int(seed))
    if key in DONE: return "done"
    t0 = time.time(); rng = np.random.default_rng(seed * 1_000_003 + unit_index * 7919 + 17 * STUDY_IX[study])
    src = rng.permutation(src); tgt = rng.permutation(tgt)
    n_cal, n_st, n_dom = (min(CFG3[k], len(src) // 5) for k in ["N_CAL", "N_SRC_TEST", "N_DOM"])
    a = 0; P = {}
    P["s_cal"] = src[a:a + n_cal]; a += n_cal; P["s_test"] = src[a:a + n_st]; a += n_st; P["s_dom"] = src[a:a + n_dom]; a += n_dom
    P["s_train"] = _cap(src[a:], CFG3["MAX_TRAIN"], rng)
    nu = min(len(tgt) // 2, 20000) if eval_by is not None else len(tgt) // 2
    P["t_unlab"], P["t_eval"] = tgt[:nu], tgt[nu:]
    if Ft is Fs: assert not np.isin(P["t_eval"], np.r_[P["s_cal"], P["s_test"], P["s_dom"], P["s_train"], P["t_unlab"]]).any()
    SRC_PARTS = ["s_cal", "s_test", "s_dom", "s_train"]
    Xof = lambda k: (Xs if k in SRC_PARTS else Xt).iloc[P[k]]
    Fof = lambda k: (Fs if k in SRC_PARTS else Ft)
    enc_d = make_encoder(num, cat).fit(pd.concat([Xof("s_dom"), Xof("t_unlab")]))
    XD = {k: enc_d.transform(Xof(k)).astype(np.float32) for k in ["s_cal", "s_test", "s_dom", "t_unlab", "t_eval"]}
    dm = fit_domain(XD["s_dom"], XD["t_unlab"]); WR = {k: weights_from(dm, XD[k]) for k in XD}
    clipv = float(np.quantile(WR["s_dom"], CFG3["CLIP_Q"])); WC = {k: np.minimum(v, clipv) for k, v in WR.items()}
    G = {k: gframe(Fof(k), P[k]) for k in ["s_cal", "s_test", "t_unlab", "t_eval"]}
    race_cats = sorted(pd.unique(pd.concat([G["s_cal"]["Race"], G["t_eval"]["Race"]])))
    imps = range(CFG3["MI_M"]) if variant == "mi" else [None]
    rows, glog, brows, crow = [], [], [], []
    src_outcomes = src_outcomes or CFG3["OUTCOMES"]
    for j in imps:
        if variant == "mi":
            enc_b = MIEncoder(num, cat, j, CFG3["MI_FIT_ROWS"], CFG3["MI_MAX_ITER"]).fit(Xof("s_train"))
        else:
            enc_b = make_encoder_v(num, cat, variant).fit(Xof("s_train"))
        XB = {k: enc_b.transform(Xof(k)).astype(np.float32) for k in ["s_train", "s_cal", "s_test", "t_unlab", "t_eval"]}
        for oi, o in enumerate(src_outcomes):
            ev_list = (eval_outcomes or {}).get(o, [o])
            yv = {k: Fof(k)[o if k in SRC_PARTS else ev_list[0]].to_numpy()[P[k]] for k in P}
            ok = {k: ~np.isnan(yv[k]) for k in P}
            y = {k: yv[k][ok[k]].astype(int) for k in P}
            if y["t_eval"].sum() < 1 or y["s_cal"].sum() < 1: continue
            for mi, mname in enumerate(CFG3["MODELS"][study.replace("NHANES1", "NHANES")]):
                model = make_model(mname, seed).fit(XB["s_train"][ok["s_train"]], y["s_train"])
                pr = {k: model.predict_proba(XB[k][ok[k]])[:, 1] for k in ["s_cal", "s_test", "t_unlab", "t_eval"]}
                g = {k: G[k][ok[k]].reset_index(drop=True) for k in G}
                w = {k: (WC[k][ok[k]], WR[k][ok[k]]) for k in ["s_cal", "s_test", "t_unlab", "t_eval"]}
                keys = lambda k: [g[k]["SexxAge"].to_numpy(str), g[k]["Age"].to_numpy(str), np.array(["all"] * len(g[k]))]
                pc, yc = pr["s_cal"], y["s_cal"]
                same_label = ev_list[0] == o
                SETS = build_sets(pc, yc, keys("s_cal"), pr["t_eval"], keys("t_eval"), w["s_cal"][0], w["t_eval"][0],
                                  *( (pr["t_unlab"], y["t_unlab"], keys("t_unlab")) if same_label else (None, None, None)))
                gib_info = {}
                if gibbs and mname == "LR":
                    SETS["Gibbs-tuned"], gib_info = gibbs_tuned(pc, yc, g, w, pr, y, race_cats, WR["s_dom"], rng, glog,
                                                                dict(study=study, unit=unit, seed=seed, outcome=o, model=mname))
                cal_n = pd.Series(g["s_cal"]["SexxAge"].to_numpy()[yc == 1]).value_counts().to_dict()
                cal_ess = {c: kish(w["s_cal"][0][(yc == 1) & (g["s_cal"]["SexxAge"].to_numpy() == c)]) for c in cal_n}
                common = dict(study=study, seed=seed, imputation=j, outcome=o, model=mname, model_backend=MODEL_BACKEND.get(mname),
                              variant=variant, heldout=unit, n_s_cal=len(yc), n_cal_pos=int(yc.sum()),
                              ess_frac_cal=kish(w["s_cal"][0]) / len(yc), clip_value=clipv, **gib_info)
                tev = P["t_eval"][ok["t_eval"]]
                wdes = Ft["wdesign"].to_numpy()[tev]
                rr = np.random.default_rng(seed * 10007 + oi + unit_index * 31 + 101 * mi)
                for eo in ev_list:
                    ye = Ft[eo].to_numpy()[tev]
                    okk = ~np.isnan(ye); ye = ye[okk].astype(int); ge = g["t_eval"][okk].reset_index(drop=True); we = wdes[okk]
                    auc_t = roc_auc_score(ye, pr["t_eval"][okk]) if 0 < ye.mean() < 1 else np.nan
                    ref_m, ref_p = exact_reference(ye, ge, cal_n, ALPHA, CFG3["REF_SIMS"], rr)
                    refe_m, refe_p = exact_reference(ye, ge, {c: math.floor(v) for c, v in cal_ess.items()}, ALPHA, CFG3["REF_SIMS"], rr)
                    extra_masks = {}
                    if "undiagnosed" in Ft: extra_masks = {"undiag": Ft["undiagnosed"].to_numpy()[tev][okk] == 1,
                                                           "diag": Ft["diagnosed"].to_numpy()[tev][okk] == 1}
                    for meth, S in SETS.items():
                        Se = S[okk]
                        for wt in (["unweighted"] + (["design"] if not np.allclose(we, 1.0) else [])):
                            ww = None if wt == "unweighted" else we
                            m = metrics4(Se, ye, ge, w=ww)
                            for nm, mk in extra_masks.items():   # NHANES: share of the undiagnosed / diagnosed whose set contains 1
                                wv = np.ones(len(ye)) if ww is None else ww
                                m[f"cov1_{nm}"] = _wmean(Se[mk, 1].astype(float), wv[mk]); m[f"only1_{nm}"] = _wmean((Se[mk, 1] & ~Se[mk, 0]).astype(float), wv[mk])
                                m[f"zero_{nm}"] = _wmean((Se[mk, 0] & ~Se[mk, 1]).astype(float), wv[mk])
                            rows.append(dict(**common, eval_outcome=eo, eval_unit="ALL", method=meth, weighting=wt, auc_target=auc_t, **m,
                                             ref_exact_mean=ref_m, ref_exact_p05=ref_p, ref_ess_mean=refe_m, ref_ess_p05=refe_p))
                    if cutoffs:
                        cr = cutoff_rules(pc, yc, pr["t_eval"][okk], ye, we)
                        for nm, v in cr.items(): crow.append(dict(**common, eval_outcome=eo, rule=nm, **v))
                    if boot and j is None and (mname == "LR" or extra_masks):
                        stv, psv = Ft["strata"].to_numpy()[tev][okk], Ft["psu"].to_numpy()[tev][okk]
                        Wm = rw_multipliers(stv, psv, PSU_BY_STRATUM, CFG3["BOOT_DESIGN"], CFG3["BOOT_DESIGN_SEED"]) * we[None, :]
                        for meth in [m_ for m_ in ["ClassCond", "GroupCond", "SAGE-CP", "Weighted+ClassCond", "Gibbs-tuned", "ClassCond@0.20", "GroupCond@0.20"] if m_ in SETS]:
                            rm = rep_metrics(SETS[meth][okk], ye, ge, Wm)
                            for nm, mk in extra_masks.items():   # NHANES: share of the undiagnosed / diagnosed whose set contains 1, or {0}
                                Sm = SETS[meth][okk]
                                rm[f"cov1_{nm}"] = (Wm[:, mk] @ Sm[mk, 1].astype(float)) / Wm[:, mk].sum(1)
                                rm[f"zero_{nm}"] = (Wm[:, mk] @ (Sm[mk, 0] & ~Sm[mk, 1]).astype(float)) / Wm[:, mk].sum(1)
                            for met, arr in rm.items():
                                                brows.append(dict(study=study, seed=seed, heldout=unit, outcome=o, eval_outcome=eo, method=meth, metric=met, model=mname,
                                                  **{f"b{b}": float(v) for b, v in enumerate(arr)}))
    tag = f"{study}__{str(unit).replace(' ', '_').replace('>', '')}__s{seed}"
    _mark_done(key, tag, rows, {"gibbs": glog, "boot": brows, "cutoff": crow})
    print(f"{study} {unit} seed {seed}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    return "ok"

def run_within_nhanes(seed, F, X):
    """N2: the whole pipeline inside NHANES with MEASURED total diabetes (50% train / 25% calibration / 25% evaluation)."""
    key = ("NHANES2", "within", int(seed))
    if key in DONE: return "done"
    rng = np.random.default_rng(90001 + seed); idx = rng.permutation(len(F)); n = len(F)
    tr, ca, ev = idx[: n // 2], idx[n // 2: 3 * n // 4], idx[3 * n // 4:]
    enc = make_encoder(HARM_NUM, HARM_CAT).fit(X.iloc[tr])
    rows, crow, brows = [], [], []
    for o in ["total", "total_fpg"]:
        y_all = F[o].to_numpy(); okt, okc, oke = (~np.isnan(y_all[s]) for s in (tr, ca, ev))
        if okc.sum() < 50: continue
        for mname in CFG3["MODELS"]["NHANES"]:
            model = make_model(mname, seed).fit(enc.transform(X.iloc[tr[okt]]), y_all[tr[okt]].astype(int))
            pc = model.predict_proba(enc.transform(X.iloc[ca[okc]]))[:, 1]; yc = y_all[ca[okc]].astype(int)
            pe = model.predict_proba(enc.transform(X.iloc[ev[oke]]))[:, 1]; ye = y_all[ev[oke]].astype(int)
            gc, ge = gframe(F, ca[okc]), gframe(F, ev[oke])
            kc = [gc["SexxAge"].to_numpy(str), gc["Age"].to_numpy(str), np.array(["all"] * len(gc))]
            ke = [ge["SexxAge"].to_numpy(str), ge["Age"].to_numpy(str), np.array(["all"] * len(ge))]
            SETS = build_sets(pc, yc, kc, pe, ke, None, None)
            wv = F["wfast" if o == "total_fpg" else "wdesign"].to_numpy()[ev[oke]]
            und = F["undiag_fpg" if o == "total_fpg" else "undiagnosed"].to_numpy()[ev[oke]] == 1
            cal_n = pd.Series(gc["SexxAge"].to_numpy()[yc == 1]).value_counts().to_dict()
            ref_m, ref_p = exact_reference(ye, ge, cal_n, ALPHA, CFG3["REF_SIMS"], rng)
            for meth, S in SETS.items():
                for wt, ww in [("unweighted", None), ("design", wv)]:
                    m = metrics4(S, ye, ge, w=ww); w1 = np.ones(len(ye)) if ww is None else ww
                    m.update(cov1_undiag=_wmean(S[und, 1].astype(float), w1[und]), only1_undiag=_wmean((S[und, 1] & ~S[und, 0]).astype(float), w1[und]),
                             zero_undiag=_wmean((S[und, 0] & ~S[und, 1]).astype(float), w1[und]))
                    rows.append(dict(study="NHANES2", seed=seed, outcome=o, eval_outcome=o, model=mname, method=meth, weighting=wt,
                                     auc_target=roc_auc_score(ye, pe), ref_exact_mean=ref_m, ref_exact_p05=ref_p, **m))
            if CFG3.get("NHANES_BOOT"):
                Wm = rw_multipliers(F["strata"].to_numpy()[ev[oke]], F["psu"].to_numpy()[ev[oke]], PSU_BY_STRATUM, CFG3["BOOT_DESIGN"], CFG3["BOOT_DESIGN_SEED"]) * wv[None, :]
                for meth in ["ClassCond", "GroupCond", "GroupCond@0.20"]:
                    S = SETS[meth]; rm = rep_metrics(S, ye, ge, Wm)
                    rm["cov1_undiag"] = (Wm[:, und] @ S[und, 1].astype(float)) / Wm[:, und].sum(1)
                    rm["zero_undiag"] = (Wm[:, und] @ (S[und, 0] & ~S[und, 1]).astype(float)) / Wm[:, und].sum(1)
                    for met, arr in rm.items():
                        brows.append(dict(study="NHANES2", seed=seed, heldout="within", outcome=o, eval_outcome=o, method=meth, metric=met, model=mname,
                                          **{f"b{b}": float(v) for b, v in enumerate(arr)}))
            for nm, v in cutoff_rules(pc, yc, pe, ye, wv).items():
                crow.append(dict(study="NHANES2", seed=seed, outcome=o, eval_outcome=o, model=mname, rule=nm, **v))
    _mark_done(key, f"NHANES2__within__s{seed}", rows, {"cutoff": crow, "boot": brows})
    print(f"NHANES2 seed {seed}: {len(rows)} rows", flush=True)
    return "ok"

def _loglik(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

def run_west(seed, F, X, num, cat):
    """Concept-shift diagnostics per held-out region and outcome (Kaggle extract, v2.1 full feature set).
    prev_target: observed prevalence in t_eval; prev_wsource: importance-weighted source prevalence on s_test (what a
    pure covariate shift with correct weights would predict for the target); oe: observed / mean predicted risk in the
    target; lr_p: likelihood-ratio test (2 df) of a region intercept and region x logit(p) slope in a logistic model of
    Y on logit(p) fitted to s_test + t_eval, i.e. whether P(Y | model risk) differs between source and target."""
    key = ("WEST", "all", int(seed))
    if key in DONE: return "done"
    rows = []
    reg = F["state"].map(REGION).to_numpy()
    for ui, r in enumerate(REGIONS_ALL):
        src, tgt = np.where(reg != r)[0], np.where(reg == r)[0]
        rng = np.random.default_rng(seed * 1_000_003 + ui * 7919 + 17 * STUDY_IX["WEST"])
        src = rng.permutation(src); tgt = rng.permutation(tgt)
        n_cal, n_st, n_dom = (min(CFG3[k], len(src) // 5) for k in ["N_CAL", "N_SRC_TEST", "N_DOM"])
        s_test = src[n_cal:n_cal + n_st]; s_dom = src[n_cal + n_st:n_cal + n_st + n_dom]
        s_train = _cap(src[n_cal + n_st + n_dom:], CFG3["MAX_TRAIN"], rng)
        t_unlab, t_eval = tgt[: len(tgt) // 2], tgt[len(tgt) // 2:]
        enc_b = make_encoder(num, cat).fit(X.iloc[s_train]); enc_d = make_encoder(num, cat).fit(pd.concat([X.iloc[s_dom], X.iloc[t_unlab]]))
        dm = fit_domain(enc_d.transform(X.iloc[s_dom]), enc_d.transform(X.iloc[t_unlab]))
        wr = weights_from(dm, enc_d.transform(X.iloc[s_test])); wcl = np.minimum(wr, np.quantile(weights_from(dm, enc_d.transform(X.iloc[s_dom])), CFG3["CLIP_Q"]))
        for o in CFG3["OUTCOMES"]:
            ys, yt, ytr = F[o].to_numpy()[s_test], F[o].to_numpy()[t_eval], F[o].to_numpy()[s_train]
            for mname in CFG3["MODELS"]["WEST"]:
                model = make_model(mname, seed).fit(enc_b.transform(X.iloc[s_train]), ytr.astype(int))
                ps = model.predict_proba(enc_b.transform(X.iloc[s_test]))[:, 1]; pt = model.predict_proba(enc_b.transform(X.iloc[t_eval]))[:, 1]
                z = np.r_[np.log(ps / (1 - ps)), np.log(pt / (1 - pt))]; d = np.r_[np.zeros(len(ps)), np.ones(len(pt))]; yy = np.r_[ys, yt].astype(int)
                from sklearn.linear_model import LogisticRegression as _LR
                m0 = _LR(penalty=None, max_iter=1000).fit(z[:, None], yy); m1 = _LR(penalty=None, max_iter=1000).fit(np.c_[z, d, z * d], yy)
                lr = 2 * (_loglik(yy, m1.predict_proba(np.c_[z, d, z * d])[:, 1]) - _loglik(yy, m0.predict_proba(z[:, None])[:, 1]))
                from scipy.stats import chi2
                dec = np.quantile(ps, np.linspace(0, 1, 11)); bs, bt = np.digitize(ps, dec[1:-1]), np.digitize(pt, dec[1:-1])
                gaps = [abs(yt[bt == k].mean() - ys[bs == k].mean()) for k in range(10) if (bt == k).sum() >= 50]
                rows.append(dict(study="WEST", seed=seed, region=r, outcome=o, model=mname, n_target=len(yt),
                                 prev_target=float(yt.mean()), prev_source=float(ys.mean()),
                                 prev_wsource_clipped=float(np.average(ys, weights=wcl)), prev_wsource_unclipped=float(np.average(ys, weights=wr)),
                                 mean_pred_target=float(pt.mean()), oe=float(yt.mean() / pt.mean()),
                                 auc_source=float(roc_auc_score(ys, ps)), auc_target=float(roc_auc_score(yt, pt)),
                                 lr_stat=float(lr), lr_p=float(chi2.sf(lr, 2)), region_intercept=float(m1.coef_[0][1]), region_slope=float(m1.coef_[0][2]),
                                 max_decile_gap=float(max(gaps)) if gaps else np.nan, mean_decile_gap=float(np.mean(gaps)) if gaps else np.nan))
    _mark_done(key, f"WEST__all__s{seed}", rows)
    print(f"WEST seed {seed}: {len(rows)} rows", flush=True)
    return "ok"
