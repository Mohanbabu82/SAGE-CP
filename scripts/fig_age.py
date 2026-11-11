"""Fig.: positive-class coverage by age band (replaces Table VI). Region-level means (mean over seeds, outcomes and
models within a region, groups with >= 30 target positives), then the mean over the five regions; bars show the range
of the five region means. Also writes derived/age_gradient_by_region.csv."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
G = pd.read_csv('../real/results/group_results.csv', low_memory=False)
G = G[(G.exp == 'main') & (G.variant == 'primary') & (G.axis == 'Age') & (G.n_pos >= 30) & G.method.isin(['ClassCond', 'GroupCond', 'SAGE-CP'])]
R = G.groupby(['method', 'group', 'region']).cov_pos.mean().reset_index()
S = R.groupby(['method', 'group']).cov_pos.agg(['mean', 'min', 'max']).reset_index()
R.to_csv('derived/age_gradient_by_region.csv', index=False); S.to_csv('derived/age_gradient_summary.csv', index=False)
print(S.round(3).to_string())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "savefig.dpi": 600, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ages = ['18-44', '45-64', '65+']; x = np.arange(3)
sty = {'ClassCond': ('#B5562F', 'o', -0.18), 'GroupCond': ('#2F6DB5', 's', 0.0), 'SAGE-CP': ('#3E8E5E', '^', 0.18)}
for m, (c, mk, dx) in sty.items():
    d = S[S.method == m].set_index('group').loc[ages]
    ax.errorbar(x + dx, d['mean'], yerr=[d['mean'] - d['min'], d['max'] - d['mean']], fmt=mk, color=c, ms=4, capsize=2, lw=0.8, label=m)
    if m == 'ClassCond':
        for xi, v in zip(x + dx, d['mean']): ax.text(xi - 0.06, v, f"{v:.2f}", fontsize=5.8, va='center', ha='right', color='#333333')
ax.axhline(0.9, color='#555555', lw=0.6, ls='--'); ax.text(1.5, 0.912, '0.90 target', fontsize=5.5, color='#555555', ha='center', va='bottom')
ax.set_xticks(x); ax.set_xticklabels(['18–44', '45–64', '65+']); ax.set_xlabel('Age band')
ax.set_ylabel('Positive-class coverage'); ax.set_ylim(0.3, 1.0); ax.set_xlim(-0.5, 2.5)
ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False, loc='lower right', fontsize=6)
fig.tight_layout(); fig.savefig('../figs/out/figures/fig8_age_gradient.png'); fig.savefig('../figs/out/figures/fig8_age_gradient.pdf')
