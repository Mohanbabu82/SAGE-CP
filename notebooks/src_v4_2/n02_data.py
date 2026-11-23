# ======================= 2. Data: Kaggle extract and raw CDC BRFSS files, mapped to one standard frame =======================
# Standard frame: state (name), Sex, Age (18-44 / 45-64 / 65+), Race, the three outcomes coded 0/1/NaN, wdesign (BRFSS
# design weight; 1 for the Kaggle extract) and the model features. Outcomes and other diagnoses never enter a feature set.
FIPS = {1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California", 8: "Colorado", 9: "Connecticut", 10: "Delaware",
        11: "District of Columbia", 12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois", 18: "Indiana", 19: "Iowa",
        20: "Kansas", 21: "Kentucky", 22: "Louisiana", 23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan", 27: "Minnesota",
        28: "Mississippi", 29: "Missouri", 30: "Montana", 31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
        35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota", 39: "Ohio", 40: "Oklahoma", 41: "Oregon",
        42: "Pennsylvania", 44: "Rhode Island", 45: "South Carolina", 46: "South Dakota", 47: "Tennessee", 48: "Texas", 49: "Utah",
        50: "Vermont", 51: "Virginia", 53: "Washington", 54: "West Virginia", 55: "Wisconsin", 56: "Wyoming", 66: "Guam",
        72: "Puerto Rico", 78: "Virgin Islands"}
REGION = {}
for s in ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont", "New Jersey", "New York", "Pennsylvania"]: REGION[s] = "Northeast"
for s in ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin", "Iowa", "Kansas", "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota"]: REGION[s] = "Midwest"
for s in ["Delaware", "District of Columbia", "Florida", "Georgia", "Maryland", "North Carolina", "South Carolina", "Virginia", "West Virginia",
          "Alabama", "Kentucky", "Mississippi", "Tennessee", "Arkansas", "Louisiana", "Oklahoma", "Texas"]: REGION[s] = "South"
for s in ["Arizona", "Colorado", "Idaho", "Montana", "Nevada", "New Mexico", "Utah", "Wyoming", "Alaska", "California", "Hawaii", "Oregon", "Washington"]: REGION[s] = "West"
for s in ["Guam", "Puerto Rico", "Virgin Islands"]: REGION[s] = "Territories"
REGIONS_ALL = ["Northeast", "Midwest", "South", "West", "Territories"]

def age3(lower):
    return np.where(lower < 45, "18-44", np.where(lower < 65, "45-64", "65+"))

def _find(name):
    hits = []
    for p in [f"/kaggle/input/**/{name}", f"./{name}", f"./data/{name}"]:
        hits += [h for h in sorted(glob.glob(p, recursive=True)) if h not in hits]
    return hits

def load_kaggle():
    """heart_2022_no_nans.csv exactly as used by v2.1 (complete cases, 246,022 rows)."""
    hits = _find("heart_2022_no_nans.csv")
    if not hits: raise FileNotFoundError("Add the Kaggle dataset 'kamilpytlak/personal-key-indicators-of-heart-disease' as notebook input.")
    d = pd.read_csv(hits[0]); check("Kaggle file has 246,022 rows x 40 columns", d.shape == (246022, 40) or CFG3["QUICK"], d.shape)
    F = pd.DataFrame({"state": d["State"].astype(str), "Sex": d["Sex"].astype(str)})
    F["Age"] = age3(d["AgeCategory"].astype(str).str.extract(r"(\d+)")[0].astype(float).to_numpy())
    F["Race"] = d["RaceEthnicityCategory"].astype(str)
    for o in CFG3["OUTCOMES"]:
        v = d[o].astype(str).str.strip().str.lower()
        F[o] = np.where(v.str.startswith("yes"), 1.0, np.where(v.str.startswith("no"), 0.0, np.nan))
        if o == "HadDiabetes":   # yes_only coding (v2.1 primary): 'Yes' = 1; 'No', borderline and pregnancy-only = 0
            F[o] = np.where(v == "yes", 1.0, np.where(v.isin(["no", "no, pre-diabetes or borderline diabetes",
                                                               "yes, but only during pregnancy (female)"]), 0.0, np.nan))
    F["wdesign"] = 1.0
    other = [c for c in d.columns if c.startswith("Had")]
    feats = [c for c in d.columns if c not in other + ["State"]]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(d[c])]; cat = [c for c in feats if c not in num]
    X = d[feats].copy()
    for c in cat: X[c] = X[c].astype(str)
    return F.reset_index(drop=True), X.reset_index(drop=True), num, cat, dict(source="Kaggle heart_2022_no_nans.csv", file=hits[0], n=len(d))

# ---- raw BRFSS: Kaggle-equivalent features from the CDC codebook variables ----
# Each feature lists candidate variable names (first present is used, so 2022 and 2023 names can differ); categorical
# features use the raw response code, with the listed codes treated as missing. Features that are missing in a year
# are dropped and logged. ChestScan is not used (not in the core questionnaire; a post-diagnostic marker per [28]).
RAW_FEATURES = {
    # name: (candidates, kind, missing codes or mapper)
    "Sex": (["SEXVAR", "_SEX"], "cat", []),
    "AgeCategory": (["_AGEG5YR"], "cat", [14]),
    "RaceEthnicityCategory": (["_IMPRACE", "_RACEPR1"], "cat", []),
    "GeneralHealth": (["GENHLTH"], "cat", [7, 9]),
    "PhysicalHealthDays": (["PHYSHLTH"], "days", None),
    "MentalHealthDays": (["MENTHLTH"], "days", None),
    "LastCheckupTime": (["CHECKUP1"], "cat", [7, 9]),
    "PhysicalActivities": (["EXERANY2"], "cat", [7, 9]),
    "SleepHours": (["SLEPTIM1"], "sleep", None),
    "RemovedTeeth": (["RMVTETH4"], "cat", [7, 9]),
    "DeafOrHardOfHearing": (["DEAF"], "cat", [7, 9]),
    "BlindOrVisionDifficulty": (["BLIND"], "cat", [7, 9]),
    "DifficultyConcentrating": (["DECIDE"], "cat", [7, 9]),
    "DifficultyWalking": (["DIFFWALK"], "cat", [7, 9]),
    "DifficultyDressingBathing": (["DIFFDRES"], "cat", [7, 9]),
    "DifficultyErrands": (["DIFFALON"], "cat", [7, 9]),
    "SmokerStatus": (["_SMOKER3"], "cat", [9]),
    "ECigaretteUsage": (["ECIGNOW2", "ECIGNOW3", "ECIGNOW1"], "cat", [7, 9]),
    "HeightInMeters": (["HTM4"], "scale100", None),
    "WeightInKilograms": (["WTKG3"], "scale100", None),
    "BMI": (["_BMI5"], "scale100", None),
    "AlcoholDrinkers": (["DRNKANY6", "DRNKANY5"], "cat", [7, 9]),
    "HIVTesting": (["HIVTST7"], "cat", [7, 9]),
    "FluVaxLast12": (["FLUSHOT7"], "cat", [7, 9]),
    "PneumoVaxEver": (["PNEUVAC4"], "cat", [7, 9]),
    "TetanusLast10Tdap": (["TETANUS1"], "cat", [7, 9]),
    "HighRiskLastYear": (["HIVRISK5"], "cat", [7, 9]),
    "CovidPos": (["COVIDPOS", "COVIDPO1"], "cat", [7, 9]),
}
RAW_OUTCOMES = {"HadHeartAttack": ["CVDINFR4"], "HadStroke": ["CVDSTRK3"], "HadDiabetes": ["DIABETE4"]}
RAW_DESIGN = {"wdesign": ["_LLCPWT"], "strata": ["_STSTR"], "psu": ["_PSU"]}

def _raw_file(year):
    """Local copy if attached as a Kaggle input (LLCP{year}.XPT or the zip); otherwise download from CDC (Internet ON)."""
    for nm in [f"LLCP_{year}.XPT", f"LLCP{year}.XPT", f"LLCP{year}.XPT ", f"LLCP{year}XPT.zip", f"llcp{year}.xpt", f"LLCP_{year}.xpt"]:
        h = _find(nm)
        if h: return h[0]
    dest = f"/kaggle/working/LLCP{year}XPT.zip" if os.path.isdir("/kaggle/working") else f"./LLCP{year}XPT.zip"
    if not os.path.exists(dest):
        print(f"downloading BRFSS {year} from CDC ...", flush=True)
        try:
            urllib.request.urlretrieve(CFG3["RAW_URLS"][year], dest)
        except Exception as e:
            raise RuntimeError(f"Could not download {CFG3['RAW_URLS'][year]} ({e}). In Kaggle: Settings -> Internet ON, or attach "
                               f"the file LLCP{year}XPT.zip as an input dataset.")
    return dest

def _read_xpt(path, keep):
    if path.lower().endswith(".zip"):
        z = zipfile.ZipFile(path); inner = [n for n in z.namelist() if n.strip().lower().endswith(".xpt")][0]
        path = z.extract(inner, os.path.dirname(path) or "."); path_clean = path.strip()
        if path_clean != path: os.replace(path, path_clean); path = path_clean
    parts = []
    for ch in pd.read_sas(path, format="xport", chunksize=50000):
        ch.columns = [c.strip().upper() for c in ch.columns]
        parts.append(ch[[c for c in keep if c in ch.columns]])
    return pd.concat(parts, ignore_index=True)

RAW_MAP_LOG = []
def load_raw(year, frame=None):
    """frame: optional pre-read DataFrame with raw CDC variable names (used by the offline test)."""
    keep = sorted({c for v in RAW_FEATURES.values() for c in v[0]} | {c for v in RAW_OUTCOMES.values() for c in v} |
                  {c for v in RAW_DESIGN.values() for c in v} | {"_STATE"})
    d = frame if frame is not None else _read_xpt(_raw_file(year), keep)
    n0 = len(d)
    def pick(cands):
        for c in cands:
            if c in d.columns: return c
        return None
    F = pd.DataFrame({"state": d["_STATE"].round().astype(int).map(FIPS)})
    agec = d["_AGEG5YR"].to_numpy(float); lower = np.where((agec >= 1) & (agec <= 13), 18 + np.maximum(0, agec - 1) * 5 + (agec > 1) * 2, np.nan)
    lower = np.where(agec == 1, 18, lower)                     # 1: 18-24, 2: 25-29, ..., 13: 80+
    F["Sex"] = np.where(d[pick(["SEXVAR", "_SEX"])] == 1, "Male", np.where(d[pick(["SEXVAR", "_SEX"])] == 2, "Female", "__MISSING__"))
    F["Age"] = np.where(np.isnan(lower), "__MISSING__", age3(np.nan_to_num(lower, nan=0)))
    rc = d[pick(["_IMPRACE", "_RACEPR1"])].to_numpy(float)
    F["Race"] = np.select([rc == 1, rc == 2, rc == 5, np.isin(rc, [3, 4, 6, 7])],
                          ["White, Non-Hispanic", "Black, Non-Hispanic", "Hispanic", "Other or multiracial, Non-Hispanic"], "__MISSING__")
    for o, cands in RAW_OUTCOMES.items():
        v = d[pick(cands)].to_numpy(float)
        F[o] = np.where(v == 1, 1.0, np.where(v == 2, 0.0, np.nan)) if o != "HadDiabetes" else \
               np.where(v == 1, 1.0, np.where(np.isin(v, [2, 3, 4]), 0.0, np.nan))   # yes_only, as v2.1
        RAW_MAP_LOG.append(dict(year=year, item=o, variable=pick(cands), kind="outcome"))
    for k, cands in RAW_DESIGN.items():
        c = pick(cands); F[k] = d[c].to_numpy(float) if c else np.nan
        RAW_MAP_LOG.append(dict(year=year, item=k, variable=c, kind="design"))
    X = pd.DataFrame(index=d.index); num, cat = [], []
    for name, (cands, kind, miss) in RAW_FEATURES.items():
        c = pick(cands)
        RAW_MAP_LOG.append(dict(year=year, item=name, variable=c, kind=kind))
        if c is None: continue
        v = d[c].to_numpy(float)
        if kind == "cat":
            X[name] = np.where(np.isnan(v) | np.isin(v, miss), "__MISSING__", pd.Series(v).round().astype("Int64").astype(str).to_numpy()); cat.append(name)
        elif kind == "days":
            X[name] = np.where(v == 88, 0.0, np.where((v >= 1) & (v <= 30), v, np.nan)); num.append(name)
        elif kind == "sleep":
            X[name] = np.where((v >= 1) & (v <= 24), v, np.nan); num.append(name)
        elif kind == "scale100":
            X[name] = np.where(v > 0, v / 100.0, np.nan); num.append(name)
    ok = F["state"].notna() & (F["Sex"] != "__MISSING__") & (F["Age"] != "__MISSING__") & (F["Race"] != "__MISSING__") & \
         F[CFG3["OUTCOMES"]].notna().any(axis=1) & (F["wdesign"] > 0)
    info = dict(source=f"CDC BRFSS LLCP{year}", n_raw=n0, n_kept=int(ok.sum()),
                n_dropped_missing_group_or_state=int((~ok).sum()),
                features_used=num + cat, features_missing=[n for n, (cands, _, _) in RAW_FEATURES.items() if pick(cands) is None])
    print(f"BRFSS {year}: {n0} raw rows, {int(ok.sum())} kept; features missing this year: {info['features_missing']}")
    return F[ok].reset_index(drop=True), X[ok].reset_index(drop=True), num, cat, info

# ---- NHANES August 2021-August 2023 (external validation with measured diabetes) ----
# Harmonised feature set available in BOTH surveys and free of post-diagnosis proxies: sex, 5-year age band (BRFSS
# _AGEG5YR codes 1-13), race/ethnicity (4 levels), measured (NHANES) or self-reported (BRFSS) height, weight and BMI, and
# smoking status (_SMOKER3 codes: 1 every day, 2 some days, 3 former, 4 never). Outcomes in NHANES (adults >= 18 with a
# valid HbA1c, phlebotomy weight WTPH2YR > 0): diagnosed = DIQ010 == 1 (told by a doctor; borderline = 0, as the BRFSS
# yes-only coding); measured = HbA1c >= 6.5%; total = diagnosed or measured; undiagnosed = measured and not diagnosed.
# Sensitivity: total_fpg also counts fasting plasma glucose >= 126 mg/dL in the fasting subsample (weight WTSAF2YR).
RACE4 = {"White": "White, Non-Hispanic", "Black": "Black, Non-Hispanic", "Hispanic": "Hispanic", "Other": "Other or multiracial, Non-Hispanic"}
HARM = ["Sex", "AgeCategory", "Race4", "SmokerStatus", "HeightInMeters", "WeightInKilograms", "BMI"]
HARM_NUM, HARM_CAT = ["HeightInMeters", "WeightInKilograms", "BMI"], ["Sex", "AgeCategory", "Race4", "SmokerStatus"]

def _nh_file(name):
    for nm in [f"{name}.xpt", f"{name}.XPT"]:
        h = _find(nm)
        if h: return h[0]
    dest = f"/kaggle/working/{name}.xpt" if os.path.isdir("/kaggle/working") else f"./{name}.xpt"
    if not os.path.exists(dest):
        try: urllib.request.urlretrieve(CFG3["NHANES_URL"].format(name), dest)
        except Exception as e:
            raise RuntimeError(f"NHANES file {name}.xpt not found. Attach the six NHANES 2021-2023 files ({', '.join(CFG3['NHANES_FILES'])}) "
                               f"as a Kaggle input dataset, or switch Internet ON ({e}).")
    return dest

def load_nhanes(frames=None):
    """frames: optional dict name -> DataFrame (offline test)."""
    T = {}
    for nm in CFG3["NHANES_FILES"]:
        d = frames[nm] if frames is not None else pd.read_sas(_nh_file(nm), format="xport")
        d.columns = [c.strip().upper() for c in d.columns]; T[nm] = d.set_index("SEQN")
    D = T["DEMO_L"].join([T[k] for k in CFG3["NHANES_FILES"] if k != "DEMO_L"], how="left")
    D = D[(D["RIDAGEYR"] >= 18) & (D.get("WTPH2YR", pd.Series(0, index=D.index)).fillna(0) > 0) & D["LBXGH"].notna()]
    age = D["RIDAGEYR"].to_numpy(float)
    code = np.where(age >= 80, 13, np.where(age < 25, 1, np.floor((age - 25) / 5) + 2)).astype(int)
    X = pd.DataFrame(index=D.index)
    X["Sex"] = D["RIAGENDR"].round().astype("Int64").astype(str).to_numpy()
    X["AgeCategory"] = code.astype(str)
    r = D["RIDRETH3"].to_numpy(float)
    X["Race4"] = np.select([r == 3, r == 4, np.isin(r, [1, 2]), np.isin(r, [6, 7])], ["White", "Black", "Hispanic", "Other"], "__MISSING__")
    s20, s40 = D["SMQ020"].to_numpy(float), D["SMQ040"].to_numpy(float)
    X["SmokerStatus"] = np.select([s20 == 2, (s20 == 1) & (s40 == 1), (s20 == 1) & (s40 == 2), (s20 == 1) & (s40 == 3)], ["4", "1", "2", "3"], "__MISSING__")
    X["HeightInMeters"] = D["BMXHT"].to_numpy(float) / 100; X["WeightInKilograms"] = D["BMXWT"].to_numpy(float); X["BMI"] = D["BMXBMI"].to_numpy(float)
    F = pd.DataFrame(index=D.index)
    F["Sex"] = np.where(D["RIAGENDR"] == 1, "Male", "Female"); F["Age"] = age3(age); F["Race"] = X["Race4"].map(RACE4).fillna("__MISSING__").to_numpy()
    dq = D["DIQ010"].to_numpy(float)
    F["diagnosed"] = np.where(dq == 1, 1.0, np.where(np.isin(dq, [2, 3]), 0.0, np.nan))
    F["measured"] = (D["LBXGH"].to_numpy(float) >= CFG3["HBA1C_CUT"]).astype(float)
    F["total"] = np.where(np.isnan(F["diagnosed"]), np.nan, np.maximum(F["diagnosed"], F["measured"]))
    F["undiagnosed"] = np.where(np.isnan(F["diagnosed"]), np.nan, ((F["measured"] == 1) & (F["diagnosed"] == 0)).astype(float))
    glu = D["LBXGLU"].to_numpy(float) if "LBXGLU" in D else np.full(len(D), np.nan)
    F["fasting"] = (D.get("WTSAF2YR", pd.Series(0, index=D.index)).fillna(0).to_numpy() > 0) & ~np.isnan(glu)
    F["total_fpg"] = np.where(F["fasting"], np.maximum(F["total"], (glu >= CFG3["FPG_CUT"]).astype(float)), np.nan)
    F["undiag_fpg"] = np.where(F["fasting"], ((F["total_fpg"] == 1) & (F["diagnosed"] == 0)).astype(float), np.nan)
    F["wdesign"] = D["WTPH2YR"].to_numpy(float); F["wfast"] = D.get("WTSAF2YR", pd.Series(np.nan, index=D.index)).to_numpy(float)
    F["strata"] = D["SDMVSTRA"].to_numpy(float); F["psu"] = D["SDMVSTRA"].to_numpy(float) * 10 + D["SDMVPSU"].to_numpy(float)
    ok = (F["Race"] != "__MISSING__") & F["diagnosed"].notna()
    info = dict(source="NHANES August 2021-August 2023", n_adults_with_hba1c=int(len(D)), n_kept=int(ok.sum()),
                prevalence_diagnosed=float(np.average(F.loc[ok, "diagnosed"], weights=F.loc[ok, "wdesign"])),
                prevalence_total=float(np.average(F.loc[ok, "total"], weights=F.loc[ok, "wdesign"])),
                prevalence_undiagnosed=float(np.average(F.loc[ok, "undiagnosed"], weights=F.loc[ok, "wdesign"])))
    print("NHANES:", info)
    return F[ok].reset_index(drop=True), X[ok].reset_index(drop=True), info

def brfss_harmonised(Fr, Xr):
    """BRFSS raw frame -> the harmonised NHANES feature set (Race4 from the standard Race column)."""
    X = pd.DataFrame({"Sex": Xr["Sex"].to_numpy(), "AgeCategory": Xr["AgeCategory"].to_numpy(),
                      "Race4": Fr["Race"].map({v: k for k, v in RACE4.items()}).fillna("__MISSING__").to_numpy(),
                      "SmokerStatus": Xr["SmokerStatus"].to_numpy(), "HeightInMeters": Xr["HeightInMeters"].to_numpy(),
                      "WeightInKilograms": Xr["WeightInKilograms"].to_numpy(), "BMI": Xr["BMI"].to_numpy()})
    return X
