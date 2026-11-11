"""Fig.: population-weighted worst calibrated-group coverage per jurisdiction (primary analysis), ClassCond vs
GroupCond, with the exact reference for unweighted coverage shaded. Writes derived/v4/fig_primary_data.csv.
Input: v4_unit_level.csv from results/v4/SAGE_CP_v4_results.zip (unzipped to ../v4res relative to the working directory)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
U = pd.read_csv('../v4res/v4_unit_level.csv')
d = U[(U.study == 'RAWP') & (U.weighting == 'design') & (U.outcome == 'pooled') & U.method.isin(['ClassCond', 'GroupCond'])]
P = d.pivot(index='unit', columns='method', values='wg_cal_pos')
ref = U[(U.study == 'RAWP') & (U.weighting == 'unweighted') & (U.outcome == 'pooled') & (U.method == 'GroupCond')][['ref_exact_mean', 'ref_exact_p05']].mean()
AB = {'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA','Guam':'GU','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Puerto Rico':'PR','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virgin Islands':'VI','Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'}
P = P.sort_values('GroupCond'); P['abbr'] = [AB.get(s, s[:2].upper()) for s in P.index]
P.assign(ref_mean=ref.ref_exact_mean, ref_p05=ref.ref_exact_p05).to_csv('derived/v4/fig_primary_data.csv')
n_up = int((P.GroupCond > P.ClassCond).sum())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "savefig.dpi": 600, "pdf.fonttype": 42,
                     "xtick.labelsize": 5.6, "ytick.labelsize": 6.5})
fig, ax = plt.subplots(figsize=(7.16, 2.5))
x = np.arange(len(P))
ax.axhspan(ref.ref_exact_p05, ref.ref_exact_mean, color='#f6c9a8', alpha=0.55, lw=0, zorder=0)
ax.axhline(ref.ref_exact_mean, color='#eb6834', lw=0.8, zorder=1)
ax.axhline(0.9, color='#52514e', lw=0.6, ls='--', zorder=1)
ax.vlines(x, P.ClassCond, P.GroupCond, color='#b8b6b0', lw=0.7, zorder=2)
ax.scatter(x, P.ClassCond, s=14, facecolors='white', edgecolors='#52514e', linewidths=0.8, zorder=3, label='ClassCond')
ax.scatter(x, P.GroupCond, s=14, color='#2a78d6', zorder=4, label='GroupCond')
ax.set_xticks(x); ax.set_xticklabels(P.abbr, rotation=90)
ax.set_xlim(-0.8, len(P) - 0.2); ax.set_ylim(0.2, 1.0)
ax.set_ylabel('Worst calibrated-group\npositive-class coverage')
ax.text(-0.4, 0.896, 'nominal 0.90', fontsize=5.8, color='#52514e', va='top', ha='left')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
h, l = ax.get_legend_handles_labels()
h += [Patch(color='#f6c9a8'), Line2D([], [], color='#eb6834', lw=0.8)]
l += [f'Exact reference (unweighted): 5th percentile {ref.ref_exact_p05:.3f} to mean', f'reference mean {ref.ref_exact_mean:.3f}']
ax.legend(h, l, frameon=False, loc='lower right', ncol=2, fontsize=6.0, bbox_to_anchor=(1.0, 0.0))
ax.text(0.0, 0.33, f'GroupCond higher in {n_up} of {len(P)} jurisdictions', fontsize=6.3, color='#0b0b0b')
ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e4e3df', lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('../figs/out/figures/fig9_primary_jurisdictions.png', dpi=300); fig.savefig('../figs/out/figures/fig9_primary_jurisdictions.pdf')
print(n_up, ref.round(4).to_dict(), P[['ClassCond', 'GroupCond']].describe().round(3).loc[['min', 'max']].to_dict())
