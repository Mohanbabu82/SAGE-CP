# ======================= 1. Configuration (SAGE-CP v4 revision studies) =======================
# v4 adds, on top of v3 (all v3 functions are reused verbatim):
#   RAWP       raw BRFSS 2022 as the PRIMARY analysis: leave-one-state-out, logistic regression, 3 seeds, design-weighted
#              metrics, tuned Gibbs baseline (LR, 2,000 calibration points per label), risk-cutoff comparison, and a
#              design-based (Rao-Wu rescaled bootstrap over PSUs within strata) standard error for every pooled contrast
#   RAWP_MISIND  same, numeric missing values encoded with missing indicators (sensitivity)
#   RAWP_MI      same, multiple imputation (m = 5, chained equations, outcome not used) (sensitivity)
#   RAWP_CB      same, CatBoost with isotonic calibration (deferral sensitivity)
#   RAWP_FULL    same, logistic regression on the full raw feature set (deferral sensitivity)
#   LOSO_G     Kaggle leave-one-state-out with the tuned Gibbs baseline (LR, 2,000 calibration points per label)
#   WEST       concept-shift diagnostics by held-out region (Kaggle extract, v2.1 full feature set, LR and LightGBM)
#   NHANES     external validation with MEASURED diabetes (HbA1c >= 6.5%) in NHANES August 2021-August 2023:
#              N1 = calibrate on BRFSS 2022, evaluate in NHANES; N2 = the same pipeline within NHANES
# Four studies, each run unit by unit and saved after every unit, so the notebook can be stopped and RESUMED
# (re-running skips units already in results_v3/units.csv):
#   R5   five held-out regions on the Kaggle extract (the paper's design) + the tuned Gibbs-Cherian-Candes baseline
#   LOSO leave-one-state-out on the Kaggle extract (about 54 held-out units instead of 5)
#   RAW  leave-one-state-out on the raw CDC BRFSS 2022 file: all respondents (no complete-case filter), metrics also
#        weighted by the BRFSS design weight _LLCPWT
#   TEMP temporal + geographic: models and calibration on raw 2022, evaluation on raw 2023 (weighted), per state
# Every decision rule and setting is fixed here, before any result is seen.
import os, glob, json, time, math, hashlib, platform, sys, io, zipfile, urllib.request, warnings, traceback
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

CFG3 = dict(
    VERSION="SAGE-CP v4 revision studies (RAWP, LOSO_G, WEST, NHANES)",
    STUDIES=["NHANES"],                             # v4.1: NHANES only, with design-based replicates (full v4 list: NHANES, WEST, LOSO_G, RAWP, RAWP_MISIND, RAWP_MI, RAWP_FULL, RAWP_CB)
    NHANES_BOOT=True,
    ALPHA=0.10,
    OUTCOMES=["HadHeartAttack", "HadStroke", "HadDiabetes"],
    MODELS={"RAWP": ["LR"], "RAWP_MISIND": ["LR"], "RAWP_MI": ["LR"], "RAWP_FULL": ["LR"], "RAWP_CB": ["CBCAL"], "LOSO_G": ["LR"],
            "WEST": ["LR", "LGBM"], "NHANES": ["LR", "LGBM"]},
    SEEDS={"RAWP": [0, 1, 2], "RAWP_MISIND": [0], "RAWP_MI": [0], "RAWP_FULL": [0], "RAWP_CB": [0], "LOSO_G": [0],
           "WEST": [0, 1, 2, 3, 4], "NHANES": list(range(10))},
    MI_M=5, MI_FIT_ROWS=20000, MI_MAX_ITER=5,           # multiple imputation: imputations, rows used to fit the imputer, iterations
    BOOT_DESIGN=1000, BOOT_DESIGN_SEED=4242,           # Rao-Wu replicates (RAWP seed 0 and NHANES)
    NHANES_FILES=["DEMO_L", "DIQ_L", "GHB_L", "GLU_L", "BMX_L", "SMQ_L"],
    NHANES_URL="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/{}.xpt",
    HBA1C_CUT=6.5, FPG_CUT=126.0,
    # PRIMARY feature set = no_proxy (the 13 possible post-diagnosis proxies removed; v2.1 features_no_proxy.csv).
    # LOSO_FULL repeats leave-one-state-out with the full v2.1 set (Kaggle only) as a sensitivity analysis.
    FEATURES_NO_PROXY=["Sex", "MentalHealthDays", "PhysicalActivities", "SleepHours", "DeafOrHardOfHearing",
                       "BlindOrVisionDifficulty", "DifficultyConcentrating", "SmokerStatus", "ECigaretteUsage",
                       "RaceEthnicityCategory", "AgeCategory", "HeightInMeters", "WeightInKilograms", "BMI",
                       "AlcoholDrinkers", "CovidPos"],
    EXTRA_ALPHAS=[0.20],                              # informativeness at alpha = 0.20 (ClassCond, GroupCond, SAGE-CP)
    N_CAL=20000, N_SRC_TEST=20000, N_DOM=20000, MAX_TRAIN=60000,   # as in v2.1
    CLIP_Q=0.99, DOMAIN_C=0.1, MIN_ESS=1,
    MIN_GROUP=100, MIN_POS=30,                        # eligibility of a group for worst-group metrics (as in v2.1)
    MIN_STATE_ROWS=500,                               # a state with fewer rows is not a held-out unit (rows stay in the source)
    REF_SIMS=2000,                                    # simulations of the exact worst-group reference (Proposition 3)
    # ---- tuned Gibbs, Cherian & Candes (2025) conditional-calibration baseline ----
    GIBBS=dict(studies=["RAWP", "LOSO_G"],          # LR only (scorecard item 7)
               score_grid=1e-3,                       # scores rounded UP to this grid (conservative) to compress the LP
               n_cal_max=2000,                        # calibration points per label used by the LP (random subsample)
               wbins=[0, 4],                          # candidate numbers of density-ratio bins in the function class
               race=[False, True],                    # candidate: include race indicators
               # selection rule (label-free for the target): for each candidate class, fit the unaugmented quantile
               # regression (level (1-alpha)(n+1)/n), build sets on s_test and estimate every eligible calibrated-group
               # positive-class coverage with the unclipped importance weights; pick the class with the smallest estimated
               # worst-group shortfall below 1-alpha, ties (within tie_tol) broken by the smaller estimated set size.
               # The selected class is then refitted with the exact augmented procedure for the evaluation data.
               tie_tol=0.005),
    BOOT_STATES=5000, BOOT_SEED=2026,                 # bootstrap over held-out units (LOSO, RAW, TEMP)
    TIME_BUDGET_MIN=660,                              # stop starting new units after this; re-run the notebook to resume
    RAW_URLS={2022: "https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip",
              2023: "https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip"},
    QUICK=os.environ.get("QUICK", "0") == "1",
)
if CFG3["QUICK"]:   # offline synthetic test only
    CFG3.update(SEEDS={k: [0] for k in CFG3["SEEDS"]}, N_CAL=1500, N_SRC_TEST=1500, N_DOM=1500, MAX_TRAIN=3000,
                MIN_GROUP=30, MIN_POS=8, MIN_STATE_ROWS=200, REF_SIMS=200, BOOT_STATES=300)
    CFG3["GIBBS"].update(n_cal_max=800, wbins=[0, 3])
    CFG3.update(SEEDS={k: [0] for k in CFG3["SEEDS"]}, MI_M=2, MI_FIT_ROWS=2000, BOOT_DESIGN=20)
ALPHA = CFG3["ALPHA"]
OUT = "/kaggle/working/results_v4" if os.path.isdir("/kaggle/working") else os.environ.get("SAGE_OUT", "./results_v4")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
# resume across sessions: copy finished units from a previous run attached as an input (any folder named results_v3)
import shutil
for f in glob.glob("/kaggle/input/**/results_v4/units/*.csv", recursive=True) + glob.glob("/kaggle/input/**/results_v4/done_units.json", recursive=True):
    dst = f"{OUT}/units/{os.path.basename(f)}" if "/units/" in f else f"{OUT}/{os.path.basename(f)}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst): shutil.copy(f, dst)
CFG3_HASH = hashlib.sha256(json.dumps(CFG3, sort_keys=True, default=str).encode()).hexdigest()[:12]
HW = dict(cpu_count=os.cpu_count(), ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1) if hasattr(os, "sysconf") else None,
          processor=platform.processor() or platform.machine())
print("hardware:", HW)
json.dump(dict(CFG3=CFG3, cfg_hash=CFG3_HASH, python=sys.version, platform=platform.platform(), hardware=HW),
          open(f"{OUT}/config_v4.json", "w"), indent=2, default=str)
CHECKS = []
def check(name, ok, detail=""):
    CHECKS.append(dict(check=name, passed=bool(ok), detail=str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" | {detail}"))
    return bool(ok)
print(CFG3["VERSION"], "| config hash", CFG3_HASH, "| output", OUT)
