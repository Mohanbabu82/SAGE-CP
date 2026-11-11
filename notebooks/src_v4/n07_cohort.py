# ======================= Cohort description of the primary data (raw BRFSS 2022, 435,969 respondents) =======================
F, X, num, cat, info = load_raw(2022)
w = F["wdesign"].to_numpy()
rows = []
def add(var, level, mask):
    rows.append(dict(variable=var, level=level, n=int(mask.sum()), share_unweighted=float(mask.mean()),
                     share_weighted=float(np.average(mask.astype(float), weights=w))))
for o in CFG3["OUTCOMES"]:
    y = F[o].to_numpy(); ok = ~np.isnan(y)
    rows.append(dict(variable="outcome", level=o, n=int(np.nansum(y)), n_valid=int(ok.sum()), share_unweighted=float(np.nanmean(y)),
                     share_weighted=float(np.average(y[ok], weights=w[ok]))))
for v in ["Sex", "Age", "Race"]:
    for lv in sorted(F[v].unique()): add(v, lv, (F[v] == lv).to_numpy())
vc = F["state"].value_counts()
rows.append(dict(variable="jurisdictions", level="count", n=int(len(vc))))
for q, lab in [(0, "min"), (0.5, "median"), (1, "max")]:
    rows.append(dict(variable="jurisdiction_n", level=lab, n=int(vc.quantile(q, interpolation="nearest"))))
rows.append(dict(variable="jurisdiction_n", level="smallest", n=int(vc.min()), share_unweighted=None, share_weighted=None, note=vc.idxmin()))
rows.append(dict(variable="jurisdiction_n", level="largest", n=int(vc.max()), note=vc.idxmax()))
rows.append(dict(variable="design", level="strata", n=int(F["strata"].nunique())))
rows.append(dict(variable="design", level="psu", n=int(F[["strata", "psu"]].drop_duplicates().shape[0])))
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/cohort_primary.csv", index=False)
vc.rename("n").to_csv(f"{OUT}/jurisdiction_sizes.csv")
pd.set_option("display.width", 200); print(C.to_string())
with zipfile.ZipFile("/kaggle/working/SAGE_CP_cohort.zip" if os.path.isdir("/kaggle/working") else "SAGE_CP_cohort.zip", "w") as z:
    z.write(f"{OUT}/cohort_primary.csv", "cohort_primary.csv"); z.write(f"{OUT}/jurisdiction_sizes.csv", "jurisdiction_sizes.csv")
