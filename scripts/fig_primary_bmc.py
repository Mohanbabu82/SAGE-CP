"""Fig. 2 (BMC version): worst calibrated-group positive-class coverage per jurisdiction, primary analysis.
(a) population-weighted (the headline metric), with the design-weighted reference (Proposition 5, v4.3 run) shaded;
(b) unweighted, with the exact finite-sample reference (Proposition 3) shaded. Same jurisdiction order in both panels.
Writes ../bmc/pkg/figures/Fig3.{png,pdf} and derived/v4/fig2_bmc_data.csv."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
U = pd.read_csv('../v4res/v4_unit_level.csv')
base = U[(U.study == 'RAWP') & (U.outcome == 'pooled') & U.method.isin(['ClassCond', 'GroupCond'])]
W = base[base.weighting == 'design'].pivot(index='unit', columns='method', values='wg_cal_pos')
UW = base[base.weighting == 'unweighted'].pivot(index='unit', columns='method', values='wg_cal_pos')
ref = U[(U.study == 'RAWP') & (U.weighting == 'unweighted') & (U.outcome == 'pooled') & (U.method == 'GroupCond')][['ref_exact_mean', 'ref_exact_p05']].mean()
AB = {'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA','Guam':'GU','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Puerto Rico':'PR','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virgin Islands':'VI','Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'}
order = W.sort_values('GroupCond').index
W, UW = W.loc[order], UW.loc[order]
pd.concat({'weighted': W, 'unweighted': UW}, axis=1).assign(ref_mean=ref.ref_exact_mean, ref_p05=ref.ref_exact_p05).to_csv('derived/v4/fig2_bmc_data.csv')
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "xtick.labelsize": 5.6, "ytick.labelsize": 6.5})
REFW = pd.read_csv('derived/v43/summary_pooled.csv').query("weighting=='design' and method=='GroupCond'").iloc[0]
fig, axes = plt.subplots(2, 1, figsize=(6.69, 4.6), sharex=True)
x = np.arange(len(order))
for ax, P, lab, (rp, rm, rname) in ((axes[0], W, 'a  Population-weighted (primary metric), with the design-weighted reference', (REFW.ref_w_p05, REFW.ref_w_mean, 'Proposition 5')), (axes[1], UW, 'b  Unweighted, with the exact finite-sample reference', (ref.ref_exact_p05, ref.ref_exact_mean, 'Proposition 3'))):
    band = True
    ax.axhspan(rp, rm, color='#f6c9a8', alpha=0.55, lw=0, zorder=0)
    ax.axhline(rm, color='#eb6834', lw=0.8, zorder=1)
    ax.axhline(0.9, color='#52514e', lw=0.6, ls='--', zorder=1)
    ax.vlines(x, P.ClassCond, P.GroupCond, color='#b8b6b0', lw=0.7, zorder=2)
    ax.scatter(x, P.ClassCond, s=12, facecolors='white', edgecolors='#52514e', linewidths=0.8, zorder=3, label='ClassCond')
    ax.scatter(x, P.GroupCond, s=12, color='#2a78d6', zorder=4, label='GroupCond')
    ax.set_ylim(0.2, 1.0); ax.set_ylabel('Worst calibrated-group\npositive-class coverage')
    ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=7.2, fontweight='bold', va='bottom')
    n_up = int((P.GroupCond > P.ClassCond).sum())
    ax.text(0.01, 0.06, f'GroupCond higher in {n_up} of {len(P)} jurisdictions', transform=ax.transAxes, fontsize=6.3)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e4e3df', lw=0.5); ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels(); h.append(Line2D([], [], color='#52514e', lw=0.6, ls='--')); l.append('nominal 0.90')
    if band:
        h += [Patch(color='#f6c9a8'), Line2D([], [], color='#eb6834', lw=0.8)]
        l += [f'{rname} pooled over jurisdictions: 5th percentile {rp:.3f} to mean', f'pooled reference mean {rm:.3f}']
    ax.legend(h, l, frameon=False, loc='lower right', ncol=2 if band else 3, fontsize=5.8)
axes[1].set_xticks(x); axes[1].set_xticklabels([AB.get(s, s[:2].upper()) for s in order], rotation=90)
axes[1].set_xlim(-0.8, len(order) - 0.2)
fig.tight_layout()
fig.savefig('../bmc/pkg/figures/Fig3.png', dpi=300, facecolor='white'); fig.savefig('../bmc/pkg/figures/Fig3.pdf')
print({'weighted_up': int((W.GroupCond > W.ClassCond).sum()), 'unweighted_up': int((UW.GroupCond > UW.ClassCond).sum()),
       'unw_gc_mean': round(UW.GroupCond.mean(), 3), 'unw_cc_mean': round(UW.ClassCond.mean(), 3)})
