"""Post-hoc analyses requested in review (11 Nov 2026), computed from the archived v2.1 outputs only.
(1) Monitor with the SAME group set for estimate and actual: H = deployable AND evaluable calibrated groups.
(2) Rules re-specified against the finite-sample reference: failure / alarm when the minimum falls below the
    5th percentile of the exact reference (GroupCond, SCP, W+CC: overlap_beta; SAGE-CP: overlap_beta_ess).
(3) Equity rule against the reference: worst calibrated-group coverage >= reference 5th percentile."""
import pandas as pd, numpy as np, sys
R = '../real/results/'
M = pd.read_csv(R + 'monitor_results.csv', low_memory=False)
M = M[(M.exp == 'main') & (M.variant == 'primary') & (M.axis_type == 'calibrated') & (M.monitor_weights == 'B:unclipped (primary)')]
M['evaluable'] = (M.n_pos_act >= 30) & (M.n_act >= 100)
H = M[M.deployable & M.evaluable]
key = ['seed', 'region', 'outcome', 'model', 'method']
cell = H.groupby(key).agg(m_hat=('cov_pos_est', 'min'), m=('cov_pos_act', 'min'), n_groups=('group', 'size')).reset_index()
W = pd.read_csv(R + 'worst_group_reference_cells.csv')
W = W[W.exp == 'main'][['seed', 'region', 'outcome', 'ref_overlap_beta_cal_p05', 'ref_overlap_beta_ess_cal_p05', 'ref_overlap_beta_cal_mean']]
cell = cell.merge(W, on=['seed', 'region', 'outcome'], how='left')
cell['ref_p05'] = np.where(cell.method == 'SAGE-CP', cell.ref_overlap_beta_ess_cal_p05, cell.ref_overlap_beta_cal_p05)
def table(c, thr_col=None, thr=None, label=''):
    out = []
    for meth, d in c.groupby('method'):
        t = d[thr_col] if thr_col else thr
        a = d.m_hat < t; f = d.m < t
        tp, fn, fp, tn = (a & f).sum(), (~a & f).sum(), (a & ~f).sum(), (~a & ~f).sum()
        rec = tp / (tp + fn) if tp + fn else np.nan; spe = tn / (tn + fp) if tn + fp else np.nan
        out.append(dict(rule=label, method=meth, n_cells=len(d), n_failures=int(f.sum()), alarm_rate=a.mean(), recall=rec,
                        specificity=spe, balanced_accuracy=np.nanmean([rec, spe]), mae_min=(d.m_hat - d.m).abs().mean(),
                        accuracy=(a == f).mean(), accuracy_always_alarm=f.mean()))
    return pd.DataFrame(out)
T = pd.concat([table(cell, thr=0.88, label='same group set; threshold 0.88 (pre-specified threshold)'),
               table(cell, thr_col='ref_p05', label='same group set; threshold = reference 5th percentile (post hoc)')])
T.to_csv('derived/posthoc_monitor_rules.csv', index=False)
print(T.round(3).to_string())
# equity rule against the reference
Rr = pd.read_csv(R + 'raw_results.csv', low_memory=False)
Rr = Rr[(Rr.exp == 'main') & (Rr.split == 'target') & (Rr.variant.isin(['primary', 'baseline']))]
wcol = [c for c in Rr.columns if c.startswith('wg_') and 'cal' in c and 'pos' in c]
print('worst-group columns:', wcol)
W2 = pd.read_csv(R + 'worst_group_reference_cells.csv'); W2 = W2[W2.exp == 'main']
E = Rr[['seed', 'region', 'outcome', 'model', 'method', 'wg_cal_pos']].merge(
    W2[['seed', 'region', 'outcome', 'ref_overlap_beta_cal_p05', 'ref_overlap_beta_ess_cal_p05', 'ref_overlap_beta_cal_mean']], on=['seed', 'region', 'outcome'])
E['ref_p05'] = np.where(E.method == 'SAGE-CP', E.ref_overlap_beta_ess_cal_p05, E.ref_overlap_beta_cal_p05)
E = E[E.method.isin(['ClassCond', 'GroupCond', 'SAGE-CP', 'TargetOracle', 'Weighted+ClassCond', 'SCP', 'GroupQR+Race'])]
cellshare = E.assign(below=E.wg_cal_pos < E.ref_p05).groupby('method').below.mean().rename('share_cells_below_ref_p05')
ro = E.groupby(['method', 'region', 'outcome'])[['wg_cal_pos', 'ref_p05']].mean().reset_index()
rule = ro.assign(ok=ro.wg_cal_pos >= ro.ref_p05).groupby('method').agg(n_region_outcome=('ok', 'size'), n_meeting=('ok', 'sum'))
rule['rule_met_all'] = rule.n_meeting == rule.n_region_outcome
pre = ro.assign(ok=ro.wg_cal_pos >= 0.88).groupby('method').ok.sum().rename('n_meeting_0.88')
EQ = pd.concat([cellshare, rule, pre], axis=1).reset_index()
EQ.to_csv('derived/posthoc_equity_rule.csv', index=False)
print(EQ.round(3).to_string())
