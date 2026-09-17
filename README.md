# Integrating Inflammatory and Endothelial Biomarkers in Machine Learning Models to Predict Sepsis Development in Critically Ill Patients:
# A Combined Retrospective and Prospective Multicenter Study

Self-contained code to reproduce the results, tables, and figures of the manuscript
*"Integrating Inflammatory and Endothelial Biomarkers in Machine Learning Models to Predict Sepsis
Development in Critically Ill Patients."*

All analyses use a LightGBM model on non-temporal, first-24-hour ICU data. 
Evaluation is over 50 repeated `StratifiedShuffleSplit` (test_size=0.2, random_state=42).

## Folder structure
```
figures/                      Scripts that generate the manuscript figures
  gini_importance.py       Figure 3A (Gini importance, top-20)
  calibration_dca.py       Figure 1A/1B (calibration + DCA); prints 50-resample calibration stats
  shap_beeswarm_and_dependence.py   Figure 3B (SHAP dot beeswarm) + Figure 4 (SHAP dependence)
  ablation_roc_pr.py       Figure 1C/1D (ablation ROC & PR vs APACHE II / SOFA)
  figure2_threshold.py     Figure 2 (biomarker threshold association, FDR q-value curves)
models/
  all_table2_models.py        Table 2 (LR/RF/XGBoost/AdaBoost/LGBM, AUROC & AUPRC, 95% CI)
  table3_ablation.py          Table 3 (LGBM full vs biomarkers-removed vs vitals-removed)
analysis/
  calibration_dca_stats.py    Section 3.2 numbers (calibration slope/intercept/Brier; DCA crossings)
  threshold_test_counts.py    Section 3.4 numbers (n significant / n tested; strongest cutoff; span)
  shap_monotonicity.py        Section 3.3 (binned SHAP-vs-value)
```

## Installation
Python 3.12.10 and pip install -r requirements.txt
