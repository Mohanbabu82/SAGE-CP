"""Build the v4 tables (primary raw-BRFSS analysis, robustness, triage, NHANES, concept-shift) from the v4 results."""
import os, shutil, glob, numpy as np, pandas as pd
R = '../v4res'; D = 'derived/v4'; os.makedirs(D, exist_ok=True)
for f in glob.glob(f'{R}/*.csv'): shutil.copy(f, D)
shutil.copy('../v41res/v4_nhanes_replicates.csv', D)
S = pd.read_csv(f'{R}/v4_summary_by_method.csv'); C = pd.read_csv(f'{R}/v4_contrasts.csv'); U = pd.read_csv(f'{R}/units.csv', low_memory=False)
mn = lambda v: f'{v:.3f}'.replace('-', '−'); m4 = lambda v: (f'{v:+.4f}' if abs(v) < 0.1 else f'{v:+.3f}').replace('-', '−')
def pfmt(p):
    return '< 0.001' if p < 0.001 else (f'{p:.3f}' if p < 0.1 else f'{p:.2f}')
out = {}

# ---- primary: method-level (RAWP, design-weighted, pooled over outcomes) ----
x = S[(S.study == 'RAWP') & (S.outcome == 'pooled')].set_index(['weighting', 'method'])
rows = ['| Method | Coverage | Positive-class coverage | \\|Pos. cov. − 0.90\\| | Worst calibrated group (pos.) | Worst race group (pos.) | Mean set size | Share {0, 1} | Unweighted worst calibrated group |', '|' + '---|' * 9]
for m, lab in [('ClassCond', 'ClassCond'), ('Weighted+ClassCond', 'W+CC'), ('GroupCond', 'GroupCond'), ('SAGE-CP', 'SAGE-CP'), ('Gibbs-tuned', 'Gibbs-tuned'), ('TargetOracle', 'TargetOracle‡'),
               ('ClassCond@0.20', 'ClassCond, α = 0.20'), ('GroupCond@0.20', 'GroupCond, α = 0.20'), ('SAGE-CP@0.20', 'SAGE-CP, α = 0.20')]:
    d, u = x.loc[('design', m)], x.loc[('unweighted', m)]
    rows.append(f'| {lab} | {mn(d.coverage)} | {mn(d.cov_pos)} | {d.absdev_cov_pos:.4f} | {mn(d.wg_cal_pos)} | {mn(d.wg_race_pos)} | {d.set_size:.3f} | {d.both:.3f} | {mn(u.wg_cal_pos)} |')
ref = x.loc[('unweighted', 'GroupCond')]
out['prim'] = '\n'.join(rows)
out['prim_note'] = (f'Means over 54 jurisdictions (three seeds, logistic regression, no-proxy features, three outcomes); metrics weighted by \\_LLCPWT except the last column. '
                    f'Exact reference for the unweighted worst calibrated group (α = 0.10): mean {ref.ref_exact_mean:.3f}, 5th percentile {ref.ref_exact_p05:.3f}. ‡ Calibrated with target labels; a reference, not a method.')

# ---- robustness / contrasts ----
P = C[C.outcome == 'pooled']
def cell(st, wt, con, met, design=False):
    r = P[(P.study == st) & (P.weighting == wt) & (P.contrast == con) & (P.metric == met)]
    if not len(r): return '—'
    r = r.iloc[0]
    star = '†' if r.p_wilcoxon_holm < 0.05 else ''
    if not design: return f'{m4(r.mean_diff)} ({int(r.n_A_better)}){star}'
    s = f'{m4(r.mean_diff)} [{m4(r.ci95_lo)}, {m4(r.ci95_hi)}], {int(r.n_A_better)}/54, p {pfmt(r.p_wilcoxon_holm)}'
    if not np.isnan(r.get('design_se', np.nan)): s += f' (design SE {r.design_se:.4f})'
    return s
CON = [('GroupCond − ClassCond', 'GroupCond - ClassCond', 'wg_cal_pos', 'worst calibrated group'), ('', 'GroupCond - ClassCond', 'wg_race_pos', 'worst race group'),
       ('', 'GroupCond - ClassCond', 'set_size', 'set size'), ('SAGE-CP − GroupCond', 'SAGE-CP - GroupCond', 'absdev_cov_pos', '\\|pos. cov. − 0.90\\|'),
       ('', 'SAGE-CP - GroupCond', 'wg_cal_pos', 'worst calibrated group'), ('', 'SAGE-CP - GroupCond', 'set_size', 'set size'),
       ('W+CC − ClassCond', 'Weighted+ClassCond - ClassCond', 'absdev_cov_pos', '\\|pos. cov. − 0.90\\|'),
       ('Gibbs-tuned − GroupCond', 'Gibbs-tuned - GroupCond', 'wg_cal_pos', 'worst calibrated group'), ('', 'Gibbs-tuned - GroupCond', 'set_size', 'set size')]
rows = ['| Contrast | Metric | Primary: raw 2022, design-weighted | Missing indicators | Multiple imputation (m = 5) | Full features | CatBoost, calibrated | Kaggle extract, 54 states |', '|' + '---|' * 8]
for lab, con, met, mlab in CON:
    rows.append(f'| {lab} | {mlab} | {cell("RAWP","design",con,met,True)} | {cell("RAWP_MISIND","design",con,met)} | {cell("RAWP_MI","design",con,met)} | '
                f'{cell("RAWP_FULL","design",con,met)} | {cell("RAWP_CB","design",con,met)} | {cell("LOSO_G","unweighted",con,met)} |')
out['rob'] = '\n'.join(rows)

# ---- triage per 1,000 ----
CT = pd.read_csv(f'{R}/cutoff_rules.csv')
def tri(st, eo, w='design', meths=('ClassCond', 'GroupCond', 'SAGE-CP', 'ClassCond@0.20', 'GroupCond@0.20')):
    rr = []
    if st == 'RAWP':
        z = S[(S.study == 'RAWP') & (S.outcome == 'pooled') & (S.weighting == w)].set_index('method')
    else:
        N = pd.read_csv(f'{R}/v4_nhanes_summary.csv'); z = N[(N.study == st) & (N.eval_outcome == eo) & (N.weighting == w)].set_index('method')
    for m in meths:
        if m not in z.index: continue
        r = z.loc[m]
        rr.append((m.replace('@0.20', ', α = 0.20'), r.zero * 1000, r.miss0 * 1000, r.npv0, r.single1 * 1000, r.ppv_single1, r.both * 1000))
    c = CT[(CT.study == st) & ((CT.eval_outcome == eo) if st != 'RAWP' else True)]
    if st == 'RAWP':
        c = c.groupby(['eval_outcome', 'rule']).mean(numeric_only=True).groupby('rule').mean()
    else:
        c = c.groupby('rule').mean(numeric_only=True)
    for rule, lab in [('cut_sens90', 'Risk cut-off, sensitivity 0.90'), ('cut_youden', "Risk cut-off, Youden's J")]:
        r = c.loc[rule]; rr.append((lab, r.ruled_out * 1000, r.missed * 1000, r.npv, r.flagged * 1000, r.ppv, 0.0))
    return rr
rows = ['| Setting | Rule | Ruled out ({0}) | Positives among ruled out | NPV of {0} | Flagged ({1}) | PPV of {1} | Deferred ({0, 1}) |', '|' + '---|' * 8]
for st, eo, lab in [('RAWP', None, 'BRFSS 2022, 54 jurisdictions (self-report)'), ('NHANES1', 'total', 'NHANES 2021–2023 (measured, BRFSS-calibrated)')]:
    for i, (m, z0, mi, npv, f1, ppv, b) in enumerate(tri(st, eo)):
        rows.append(f'| {lab if i == 0 else ""} | {m} | {z0:.0f} | {mi:.1f} | {npv:.3f} | {f1:.0f} | {ppv:.3f} | {b:.0f} |')
out['triage'] = '\n'.join(rows)

# ---- NHANES ----
N = pd.read_csv(f'{R}/v4_nhanes_summary.csv'); CI = pd.read_csv('derived/v4/nhanes_design_ci.csv')
rows = ['| Design | Method | Coverage | Positive-class coverage | Undiagnosed covered (set contains 1) [95% design-based interval] | Undiagnosed ruled out ({0}) | Worst calibrated group (pos.) | Mean set size | Share {0, 1} |', '|' + '---|' * 9]
for st, eo, lab, meths in [('NHANES1', 'total', 'Calibrated on BRFSS 2022 self-report; evaluated on measured diabetes', ['ClassCond', 'Weighted+ClassCond', 'GroupCond', 'SAGE-CP', 'GroupCond@0.20']),
                           ('NHANES2', 'total', 'Within NHANES, measured diabetes', ['ClassCond', 'GroupCond', 'GroupCond@0.20'])]:
    z = N[(N.study == st) & (N.eval_outcome == eo) & (N.weighting == 'design')].set_index('method')
    for i, m in enumerate(meths):
        r = z.loc[m]
        ci = CI[(CI.study == st) & (CI.eval_outcome == eo) & (CI.method == m) & (CI.metric == 'cov1_undiag')]
        u = f'{r.cov1_undiag:.3f}' + (f' [{ci.lo.iloc[0]:.3f}, {ci.hi.iloc[0]:.3f}]' if len(ci) else '')
        rows.append(f'| {lab if i == 0 else ""} | {m.replace("Weighted+ClassCond", "W+CC").replace("@0.20", ", α = 0.20")} | {r.coverage:.3f} | {r.cov_pos:.3f} | {u} | {r.zero_undiag:.3f} | {r.wg_cal_pos:.3f} | {r.set_size:.3f} | {r.both:.3f} |')
out['nhanes'] = '\n'.join(rows)

# ---- concept-shift diagnostics ----
W = pd.read_csv(f'{R}/v4_west_concept_shift.csv')
rows = ['| Region | Outcome | Target prevalence | Source prevalence | Weighted source prevalence | Observed / expected | Median LR statistic (2 df) | Runs with p < 0.001 |', '|' + '---|' * 8]
for _, r in W.iterrows():
    rows.append(f'| {r.region} | {r.outcome.replace("Had", "")} | {r.prev_target:.4f} | {r.prev_source:.4f} | {r.prev_wsource_clipped:.4f} | {r.oe:.3f} | {r.lr_stat:.2f} | {int(round(r.share_lr_p_below_001 * 10))}/10 |')
out['west'] = '\n'.join(rows)
for k, v in out.items(): open(f'derived/v4_{k}.md', 'w').write(v + '\n')
print({k: len(v) for k, v in out.items()})
