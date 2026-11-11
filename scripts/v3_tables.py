"""Tables for the v3 extension studies (R5, LOSO, LOSO_FULL, RAW, TEMP) from derived/v3/*.csv."""
import pandas as pd, numpy as np
C = pd.read_csv('derived/v3/v3_contrasts.csv'); S = pd.read_csv('derived/v3/v3_summary_by_method.csv')
U = pd.read_csv('derived/v3/v3_unit_level.csv')
STUD = [('R5', 'unweighted', '5 regions'), ('LOSO', 'unweighted', '54 states'), ('RAW', 'design', 'Raw 2022, 54 states (weighted)'), ('TEMP', 'design', '2022→2023, 52 states (weighted)')]
ROWS = [('GroupCond - ClassCond', 'wg_cal_pos', 'GroupCond − ClassCond: worst calibrated group'),
        ('GroupCond - ClassCond', 'set_size', 'GroupCond − ClassCond: set size'),
        ('GroupCond - ClassCond', 'wg_race_pos', 'GroupCond − ClassCond: worst race group'),
        ('SAGE-CP - GroupCond', 'absdev_cov_pos', 'SAGE-CP − GroupCond: \\|Cov⁺ − 0.90\\|'),
        ('SAGE-CP - GroupCond', 'wg_cal_pos', 'SAGE-CP − GroupCond: worst calibrated group'),
        ('SAGE-CP - GroupCond', 'set_size', 'SAGE-CP − GroupCond: set size'),
        ('Gibbs-tuned - GroupCond', 'wg_cal_pos', 'Gibbs-tuned − GroupCond: worst calibrated group'),
        ('Gibbs-tuned - GroupCond', 'set_size', 'Gibbs-tuned − GroupCond: set size')]
def cell(st, wt, con, met):
    r = C[(C.study == st) & (C.weighting == wt) & (C.outcome == 'pooled') & (C.contrast == con) & (C.metric == met)]
    if not len(r): return '—'
    r = r.iloc[0]; p = r.p_wilcoxon_holm
    ps = '' if st == 'R5' else (f'; p<0.001' if p < 0.001 else f'; p={p:.3f}')
    return f"{r.mean_diff:+.4f} [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}]; {int(r.n_A_better)}/{int(r.n_units)}{ps}"
hdr = '| Contrast (A − B) | ' + ' | '.join(s[2] for s in STUD) + ' |\n|' + '---|' * (len(STUD) + 1) + '\n'
main = hdr + ''.join('| ' + lab + ' | ' + ' | '.join(cell(s, w, con, met) for s, w, _ in STUD) + ' |\n' for con, met, lab in ROWS)
open('derived/v3_main_table.md', 'w').write(main)
# method summary (supplement)
sm = S[(S.outcome.isin(['HadHeartAttack', 'HadStroke', 'HadDiabetes']))]
sm = sm.groupby(['study', 'weighting', 'method'])[['coverage', 'cov_pos', 'set_size', 'both', 'wg_cal_pos', 'wg_race_pos']].mean().reset_index()
keep = ['ClassCond', 'GroupCond', 'Weighted+ClassCond', 'SAGE-CP', 'Gibbs-tuned', 'TargetOracle', 'ClassCond@0.20', 'GroupCond@0.20', 'SAGE-CP@0.20']
sm = sm[sm.method.isin(keep)]
sm.to_csv('derived/v3_method_summary.csv', index=False)
lines = ['| Study | Weighting | Method | Coverage | Cov⁺ | Set size | {0,1} | Worst calibrated group | Worst race group |', '|---|---|---|---|---|---|---|---|---|']
for st in ['R5', 'LOSO', 'LOSO_FULL', 'RAW', 'TEMP']:
    for wt in ['unweighted', 'design']:
        d = sm[(sm.study == st) & (sm.weighting == wt)]
        for m in keep:
            x = d[d.method == m]
            if not len(x): continue
            x = x.iloc[0]
            lines.append(f"| {st} | {wt} | {m} | {x.coverage:.3f} | {x.cov_pos:.3f} | {x.set_size:.3f} | {x.both:.3f} | {x.wg_cal_pos:.3f} | {x.wg_race_pos:.3f} |")
open('derived/v3_method_table.md', 'w').write('\n'.join(lines) + '\n')
# full contrasts per outcome (supplement)
cc = C[C.study.isin(['R5', 'LOSO', 'LOSO_FULL', 'RAW', 'TEMP'])].copy()
cl = ['| Study | Weighting | Outcome | Contrast | Metric | Units | Mean diff. [95% CI] | A better | Holm p |', '|---|---|---|---|---|---|---|---|---|']
for _, r in cc.iterrows():
    hp = '—' if pd.isna(r.p_wilcoxon_holm) else ('<0.001' if r.p_wilcoxon_holm < 0.001 else f'{r.p_wilcoxon_holm:.3f}')
    cl.append(f"| {r.study} | {r.weighting} | {r.outcome} | {r.contrast} | {r.metric} | {int(r.n_units)} | {r.mean_diff:+.4f} [{r.ci95_lo:+.4f}, {r.ci95_hi:+.4f}] | {int(r.n_A_better)} | {hp} |")
open('derived/v3_contrast_table.md', 'w').write('\n'.join(cl) + '\n')
R = pd.read_csv('derived/v3/v3_reference_gaps.csv')
rl = ['| Study | Outcome | Method | Reference | Units | Mean gap | Share below 5th percentile |', '|---|---|---|---|---|---|---|']
for _, r in R.iterrows(): rl.append(f"| {r.study} | {r.outcome} | {r.method} | {r.reference.replace('ref_', '')} | {int(r.n_units)} | {r.mean_gap:+.4f} | {r.share_below_p05:.3f} |")
open('derived/v3_refgap_table.md', 'w').write('\n'.join(rl) + '\n')
print(main); print(len(cl), 'contrast rows')
