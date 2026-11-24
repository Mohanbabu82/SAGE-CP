"""Participant flow figure (new Fig. 2 of the BMC manuscript): data sources, exclusions and analysis partitions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

MM = 1 / 25.4
fig = plt.figure(figsize=(170 * MM, 150 * MM))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 170); ax.set_ylim(0, 150); ax.axis("off")
INK, MUTED, EDGE = "#1f1f1f", "#555555", "#444444"

def box(x, y, w, h, text, bold_first=True, fs=6.6, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fill, ec=EDGE, lw=0.7))
    lines = text.split("\n")
    ax.text(x + w / 2, y + h - 2.0, lines[0], ha="center", va="top", fontsize=fs, color=INK, fontweight="bold" if bold_first else "normal")
    if len(lines) > 1:
        ax.text(x + w / 2, y + h - 2.0 - fs * 0.62, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs - 0.6, color=INK, linespacing=1.25)

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=EDGE, mutation_scale=7))

# ---- primary data column
box(5, 132, 70, 14, "BRFSS 2022 raw file\n445,132 respondents in 54 jurisdictions")
box(85, 133, 80, 12, "Excluded: 9,163\nmissing jurisdiction, sex, age band, race/ethnicity or final weight", fill="#f4f4f4")
arrow(75, 139, 85, 139)
box(5, 106, 70, 20, "Primary analysis: 435,969 respondents\nheart attack 433,260; stroke 434,714; diabetes 435,192\n(valid outcome); 2,260 strata\nweights: final survey weight _LLCPWT")
arrow(40, 132, 40, 126)

box(5, 62, 70, 38, "Each held-out jurisdiction u (54), seeds 0–2\n"
    "Source: the other 53 jurisdictions\n"
    "  training ≤ 60,000 · calibration ≤ 20,000\n"
    "  source test ≤ 20,000 · domain ≤ 20,000\n"
    "Target: jurisdiction u (1,479 to 25,612 respondents)\n"
    "  half unlabelled (density ratio; TargetOracle)\n"
    "  half evaluation (all metrics, weighted to the\n"
    "  jurisdiction's adult population)\n"
    "Design-based SE: 200 Rao–Wu replicates (seed 0)")
arrow(40, 106, 40, 100)

# ---- sensitivity and temporal
box(85, 104, 80, 22, "Sensitivity analyses\n"
    "Missing indicators; multiple imputation; full\n"
    "feature set; calibrated CatBoost (435,969)\n"
    "Complete-case extract (Kaggle): 246,022 respondents\n"
    "  five regions; 54 jurisdictions")
arrow(75, 116, 85, 116)
box(85, 74, 80, 20, "Temporal check: BRFSS 2023\n433,323 raw; 425,468 kept\n52 jurisdictions (Kentucky and Pennsylvania absent)\nmodels and calibration from 2022")
arrow(75, 84, 85, 84)

# ---- external validation
box(5, 6, 160, 42, "External validation: NHANES August 2021–August 2023 (examination weights, strata, PSUs)", fs=6.8)
box(9, 9, 48, 31, "Diabetes\n6,002 adults with HbA1c; 6,001 kept\nmeasured: HbA1c ≥ 6.5%\ncalibrated on BRFSS 2022\nself-report", bold_first=True, fill="#fafafa")
box(61, 9, 48, 31, "Hypertension\n6,123 adults with oscillometric BP;\n6,119 kept\nmeasured: mean SBP ≥ 140 or\nDBP ≥ 90 mmHg; total adds\ndiagnosis or medication\ncalibrated on BRFSS 2023 self-report", fill="#fafafa")
box(113, 9, 48, 31, "High cholesterol\n5,717 adults with total cholesterol;\n5,684 kept\nmeasured: ≥ 240 mg/dL\ncalibrated on BRFSS 2023\nself-report", fill="#fafafa")
for fmt in ("pdf", "png"):
    fig.savefig(f"/home/claude/work/bmc/pkg/figures/Fig2.{fmt}", dpi=600 if fmt == "png" else None)
print("ok")
