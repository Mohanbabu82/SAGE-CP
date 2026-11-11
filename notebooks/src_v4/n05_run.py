# ======================= 5. Run the v4 studies (resumable; each unit saved as soon as it finishes) =======================
def time_left(): return (time.time() - T0) / 60 < CFG3["TIME_BUDGET_MIN"]
DATA_INFO = {}
_prev_info = json.load(open(f"{OUT}/data_info.json")) if os.path.exists(f"{OUT}/data_info.json") else {}
def subset(X, num, cat, full=False):
    keep = list(X.columns) if full else [c for c in X.columns if c in CFG3["FEATURES_NO_PROXY"]]
    return X[keep], [c for c in num if c in keep], [c for c in cat if c in keep]
def units_of(F):
    vc = F["state"].value_counts(); return sorted(vc[vc >= CFG3["MIN_STATE_ROWS"]].index)
R22 = KF = None
def get_raw():
    global R22
    if R22 is None:
        R22 = load_raw(2022); DATA_INFO["raw2022"] = R22[4]
        F = R22[0]
        for h, d in F.groupby("strata")["psu"]: PSU_BY_STRATUM[h] = np.sort(d.unique())
    return R22
def get_kaggle():
    global KF
    if KF is None: KF = load_kaggle(); DATA_INFO["kaggle"] = KF[4]
    return KF
RAWP_VARIANT = {"RAWP": ("median", False), "RAWP_MISIND": ("misind", False), "RAWP_MI": ("mi", False), "RAWP_FULL": ("median", True), "RAWP_CB": ("median", False)}

for study in CFG3["STUDIES"]:
    if not time_left(): print("time budget reached: re-run the notebook to continue"); break
    try:
        if study == "NHANES":
            NF, NX, ninfo = load_nhanes(); DATA_INFO["nhanes"] = ninfo
            for h, d in NF.groupby("strata")["psu"]: PSU_BY_STRATUM[h] = np.sort(d.unique())
            F, X, num, cat, _ = get_raw()
            HX = brfss_harmonised(F, X); okb = (HX["Race4"] != "__MISSING__").to_numpy()
            Fb = F[okb].reset_index(drop=True); HXb = HX[okb].reset_index(drop=True)
            for seed in CFG3["SEEDS"]["NHANES"]:
                if not time_left(): break
                run_unit4("NHANES1", "BRFSS->NHANES", seed, Fb, HXb, np.arange(len(Fb)), NF, NX, np.arange(len(NF)), HARM_NUM, HARM_CAT,
                          src_outcomes=["HadDiabetes"], eval_outcomes={"HadDiabetes": ["diagnosed", "total"]}, cutoffs=True,
                          boot=CFG3.get("NHANES_BOOT", False))
                run_within_nhanes(seed, NF, NX)
        elif study == "WEST":
            F, X, num, cat, _ = get_kaggle()
            for seed in CFG3["SEEDS"]["WEST"]:
                if time_left(): run_west(seed, F, X, num, cat)
        elif study == "LOSO_G":
            F, X, num, cat, _ = get_kaggle(); Xn, numn, catn = subset(X, num, cat)
            for seed in CFG3["SEEDS"][study]:
                for ui, s in enumerate(units_of(F)):
                    if not time_left(): break
                    run_unit4(study, s, seed, F, Xn, np.where(F["state"] != s)[0], F, Xn, np.where(F["state"] == s)[0], numn, catn,
                              unit_index=ui, gibbs=True)
        elif study.startswith("RAWP"):
            F, X, num, cat, _ = get_raw()
            variant, full = RAWP_VARIANT[study]
            Xs_, nums_, cats_ = subset(X, num, cat, full=full)
            for seed in CFG3["SEEDS"][study]:
                for ui, s in enumerate(units_of(F)):
                    if not time_left(): break
                    run_unit4(study, s, seed, F, Xs_, np.where(F["state"] != s)[0], F, Xs_, np.where(F["state"] == s)[0], nums_, cats_,
                              unit_index=ui, variant=variant, gibbs=(study == "RAWP"), boot=(study == "RAWP" and seed == 0),
                              cutoffs=(study == "RAWP"))
    except Exception as e:
        traceback.print_exc(); check(f"study {study} completed without error", False, repr(e))
if RAW_MAP_LOG: pd.DataFrame(RAW_MAP_LOG).drop_duplicates().to_csv(f"{OUT}/raw_variable_map.csv", index=False)
json.dump({**_prev_info, **DATA_INFO}, open(f"{OUT}/data_info.json", "w"), indent=2, default=str)
print("elapsed min", round((time.time() - T0) / 60, 1))
