# SAGE-CP v2.1

[![DOI](https://zenodo.org/badge/1451303698.svg)](https://doi.org/10.5281/zenodo.24082660)

Code for the study *Group-Conditional and Density-Ratio-Weighted Conformal Prediction Across U.S. Regions: A Component-Wise Evaluation on Cross-Sectional Health Survey Data* (Mohan Babu A, Vasumadhi V, E. S. Madhan; School of Computer Science and Engineering, Vellore Institute of Technology). Manuscript under preparation for IEEE Access.

SAGE-CP stands for shift-aware, group-conditional conformal prediction with effective-sample-size diagnostics. The name labels the configuration being evaluated; it does not claim equal coverage across groups.

## Contents

| Path | What it is |
|---|---|
| `notebooks/SAGE_CP_v2_1.ipynb` | The analysis notebook (clean, no outputs). |
| `notebooks/SAGE_CP_v2_1_executed_kaggle.ipynb` | The same notebook as executed on Kaggle for the reported run, with outputs, plus two cells that print the summaries and zip the results folder. |
| `notebooks/SAGE_CP_v3_extension_studies.ipynb` | Extension studies for the revision: leave-one-state-out, raw BRFSS 2022 with design weights, 2022→2023 temporal validation, and a tuned Gibbs–Cherian–Candès conditional-calibration baseline. Resumable; downloads the CDC files (Internet ON). |
| `notebooks/SAGE_CP_v3_run2_RAW_TEMP.ipynb` | Second session of the extension studies (RAW and TEMP only), reading the CDC LLCP2022/LLCP2023 XPT files from an attached Kaggle dataset. |
| `notebooks/SAGE_CP_v4_revision_studies.ipynb` | Second-revision studies: raw BRFSS 2022 as primary analysis with design-based (PSU/strata) standard errors, missing-data sensitivity (missing indicators, multiple imputation), calibrated CatBoost and full features, tuned Gibbs baseline in the 54-state studies, concept-shift diagnostics, and external validation with measured diabetes in NHANES 2021–2023. |
| `reference/` | `worst_group_reference(...)` and `worst_group_bounds(...)`: the finite-sample worst-group reference (Proposition 3) and its simulation-free bounds (Theorem 1), with their validation. |
| `results/v3/` | Combined outputs of both extension-study runs: per-unit results (`units.csv`, `v3_unit_level.csv`), method summaries, all pre-specified contrasts, finite-sample reference gaps, the Gibbs function-class selections, configurations and checks, and the Markdown tables used in the manuscript. |
| `scripts/v3_tables.py` | Builds Table IX and Supplementary Tables S21–S23 from `results/v3/`. |
| `results/v2.1/SAGE-CP_results_archive.zip` | The 93-file results archive of the v2.1 run (main analysis). |
| `scripts/verify_numbers.py` | Checks every decimal number in the manuscript and supplement (.docx or .md) against the result files: `python scripts/verify_numbers.py paper.docx supplement.docx --results results/ --raw <v2.1 raw_results.csv>`. The only expected unmatched value is the template's DOI placeholder. |
| `scripts/make_figures_tables.py` | Regenerates the manuscript's tables and figures from the run's CSV outputs. |
| `scripts/posthoc_rules.py` | Post hoc analyses from the archived v2.1 outputs: monitor with the same group set, and equity/alarm rules re-specified against the finite-sample reference. |
| `scripts/fig_age.py` | Age-gradient figure from `group_results.csv`. |
| `simulations/reference_validation.py` | Simulation-based calibration of the finite-sample worst-group references (Supplementary Section S3). |
| `simulations/misspec_sim.py` | Domain-classifier misspecification check (Supplementary Section S4). |

The two simulations are synthetic and were run after the main analysis. They are not evidence about the BRFSS.

## Data

The notebook uses the Kaggle dataset *Personal Key Indicators of Heart Disease* (`kamilpytlak/personal-key-indicators-of-heart-disease`), file `heart_2022_no_nans.csv` (246,022 rows × 40 columns). It is a processed, complete-case extract of the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2022. The data are not redistributed here; download them from Kaggle under the dataset's terms.

## Reproducing the run

1. On Kaggle, create a notebook, add the dataset above as input and import `notebooks/SAGE_CP_v2_1.ipynb`.
2. Run all cells. The reported run took about 78 minutes on Kaggle. Results are written to `/kaggle/working/results/`.
3. To rebuild tables and figures: `python scripts/make_figures_tables.py <results_dir> <out_dir>`.

Recorded environment of the reported run: Python 3.13.18, numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, scipy 1.16.3, matplotlib 3.10.0, xgboost 3.4.1, lightgbm 4.6.0, catboost 1.2.10. Seeds 0–4 (sensitivity runs 0–2), bootstrap seed 2026.

To run the simulations: `cd simulations && python reference_validation.py` and `python misspec_sim.py` (numpy, pandas, scipy, scikit-learn).

## Results archive

The 93 output files of the reported run are not in this repository. [To be deposited with an archived DOI before publication.]

## License and citation

License: MIT (see `LICENSE`). Archived release: v2.1, DOI [10.5281/zenodo.24082661](https://doi.org/10.5281/zenodo.24082661) (all versions: [10.5281/zenodo.24082660](https://doi.org/10.5281/zenodo.24082660)). Paper citation: [to be added after publication].
