# Figure: Ablation ROC + PR curves
#         LGBM (full) vs LGBM (biomarkers removed) vs APACHE II vs SOFA
# Model: LGBM only.
import os, re, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy import stats
warnings.filterwarnings("ignore")

import sys
DATA   = sys.argv[1] if len(sys.argv) > 1 else r""
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else r""
df = pd.read_csv(DATA)

def sanitize_columns(cols):
    out, seen = [], {}
    for c in cols:
        s = str(c).strip().replace("µ", "u")
        s = re.sub(r"\s+", "_", s); s = re.sub(r"[^0-9a-zA-Z_]+", "_", s); s = re.sub(r"_+", "_", s).strip("_")
        if s == "": s = "col"
        if s in seen: seen[s] += 1; s = f"{s}_{seen[s]}"
        else: seen[s] = 0
        out.append(s)
    return out
def _norm(s):
    s = str(s).strip().lower().replace("µ", "u"); s = re.sub(r"[^a-z0-9]+", "_", s); return re.sub(r"_+", "_", s).strip("_")
def make_lgbm_pipe_for_X(Xmat, lgb_params):
    bin_cols = [c for c in Xmat.columns if Xmat[c].dropna().nunique() == 2]
    cont_cols = [c for c in Xmat.columns if c not in bin_cols]
    pre = ColumnTransformer([("bin", SimpleImputer(strategy="most_frequent"), bin_cols),
                             ("cont", SimpleImputer(strategy="median"), cont_cols)], remainder="drop")
    return Pipeline([("prep", pre), ("clf", lgb.LGBMClassifier(**lgb_params))])
def summarize(scores, name):
    lo, hi = np.percentile(scores, [2.5, 97.5]); print(f"{name}: mean={scores.mean():.4f}, 95% CI=({lo:.4f}, {hi:.4f})")
def shade_std(x, mean, std, color, alpha=0.12, zorder=1):
    plt.fill_between(x, np.clip(mean-std,0,1), np.clip(mean+std,0,1), color=color, alpha=alpha, linewidth=0, zorder=zorder)

target_column = "SEPSIS"; APACHE_COL, SOFA_COL = "APACHE_II", "SOFA"
features_to_drop_base = [target_column, 'APACHE_II', 'SOFA']
drop_biomarkers_raw = ["IL-6", "sEPCR", "CRP", "PCT"]
n_splits, test_size, random_state = 50, 0.2, 42
lgb_params = dict(objective="binary", n_estimators=296, learning_rate=0.021758255723790025, num_leaves=8, max_depth=4,
    min_child_samples=17, min_split_gain=0.34081398506638294, lambda_l1=8.226779549337834e-05,
    lambda_l2=3.0581582963155785e-10, feature_fraction=0.5437381928004323, bagging_fraction=0.7389041657117149,
    bagging_freq=0, scale_pos_weight=1.1564125380200863, verbose=-1, random_state=42, n_jobs=-1)

apache = pd.to_numeric(df[APACHE_COL], errors="coerce").fillna(pd.to_numeric(df[APACHE_COL], errors="coerce").median())
sofa   = pd.to_numeric(df[SOFA_COL],   errors="coerce").fillna(pd.to_numeric(df[SOFA_COL],   errors="coerce").median())

X_full = df.drop(columns=features_to_drop_base, errors="ignore").select_dtypes(include=["number"]).copy()
y_full = df[target_column].copy()
X_full.columns = sanitize_columns(X_full.columns)
norm_to_actual = {_norm(c): c for c in X_full.columns}
drop_in_X = [norm_to_actual[_norm(r)] for r in drop_biomarkers_raw if _norm(r) in norm_to_actual]
print("Ablation dropping (found in X_full):", drop_in_X)
X_abl = X_full.drop(columns=drop_in_X, errors="ignore").copy()

lgb_full = make_lgbm_pipe_for_X(X_full, lgb_params); lgb_abl = make_lgbm_pipe_for_X(X_abl, lgb_params)
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
auroc_full, auprc_full, auroc_abl, auprc_abl, auroc_ap, auprc_ap, auroc_sf, auprc_sf = ([] for _ in range(8))
mean_fpr = np.linspace(0, 1, 50); tprs_full, tprs_abl, tprs_ap, tprs_sf = [], [], [], []
mean_recall = np.linspace(0, 1, 13); precs_full, precs_abl, precs_ap, precs_sf = [], [], [], []

for sid, (tr, te) in enumerate(sss.split(X_full, y_full), start=1):
    ytr, yte = y_full.iloc[tr], y_full.iloc[te]
    lgb_full.fit(X_full.iloc[tr], ytr); p_full = lgb_full.predict_proba(X_full.iloc[te])[:, 1]
    lgb_abl.fit(X_abl.iloc[tr], ytr);   p_abl  = lgb_abl.predict_proba(X_abl.iloc[te])[:, 1]
    p_ap = apache.iloc[te].values; p_sf = sofa.iloc[te].values
    auroc_full.append(roc_auc_score(yte, p_full)); auprc_full.append(average_precision_score(yte, p_full))
    auroc_abl.append(roc_auc_score(yte, p_abl));   auprc_abl.append(average_precision_score(yte, p_abl))
    auroc_ap.append(roc_auc_score(yte, p_ap));     auprc_ap.append(average_precision_score(yte, p_ap))
    auroc_sf.append(roc_auc_score(yte, p_sf));     auprc_sf.append(average_precision_score(yte, p_sf))
    for pv, store in [(p_full,tprs_full),(p_abl,tprs_abl),(p_ap,tprs_ap),(p_sf,tprs_sf)]:
        fpr, tpr, _ = roc_curve(yte, pv); t = np.interp(mean_fpr, fpr, tpr); t[0] = 0.0; store.append(t)
    for pv, store in [(p_full,precs_full),(p_abl,precs_abl),(p_ap,precs_ap),(p_sf,precs_sf)]:
        prec, rec, _ = precision_recall_curve(yte, pv); store.append(np.interp(mean_recall, rec[::-1], prec[::-1]))
    if sid % 10 == 0: print(f"[{sid}/{n_splits}] AUROC full={auroc_full[-1]:.3f} | abl={auroc_abl[-1]:.3f}")

auroc_full,auprc_full,auroc_abl,auprc_abl,auroc_ap,auprc_ap,auroc_sf,auprc_sf = map(np.array,
    (auroc_full,auprc_full,auroc_abl,auprc_abl,auroc_ap,auprc_ap,auroc_sf,auprc_sf))
print("\nPaired tests (LGBM Full vs Ablation):")
print(f"AUROC p={stats.ttest_rel(auroc_full, auroc_abl)[1]:.4e}")
print(f"AUPRC p={stats.ttest_rel(auprc_full, auprc_abl)[1]:.4e}")
print("\nSummary:")
for s,n in [(auroc_full,"LGBM full AUROC"),(auprc_full,"LGBM full AUPRC"),(auroc_abl,"LGBM ablation AUROC"),
            (auprc_abl,"LGBM ablation AUPRC"),(auroc_ap,"APACHE AUROC"),(auprc_ap,"APACHE AUPRC"),
            (auroc_sf,"SOFA AUROC"),(auprc_sf,"SOFA AUPRC")]: summarize(s,n)

tprs_full,tprs_abl,tprs_ap,tprs_sf = map(np.array,(tprs_full,tprs_abl,tprs_ap,tprs_sf))
precs_full,precs_abl,precs_ap,precs_sf = map(np.array,(precs_full,precs_abl,precs_ap,precs_sf))
mtf=tprs_full.mean(0); mtf[-1]=1.0; mta=tprs_abl.mean(0); mta[-1]=1.0; mtp=tprs_ap.mean(0); mtp[-1]=1.0; mts=tprs_sf.mean(0); mts[-1]=1.0
stf,sta,stp,sts = tprs_full.std(0),tprs_abl.std(0),tprs_ap.std(0),tprs_sf.std(0)
mpf,mpa,mpp,mps = precs_full.mean(0),precs_abl.mean(0),precs_ap.mean(0),precs_sf.mean(0)
spf,spa,spp,sps = precs_full.std(0),precs_abl.std(0),precs_ap.std(0),precs_sf.std(0)

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":16,"font.weight":"bold","axes.labelweight":"bold",
                     "axes.titlesize":18,"axes.titleweight":"bold","legend.fontsize":14})
#   LGBM full=#C0392B (red), LGBM ablation=#000000 (black),
#   APACHE II=#2E86C1 (blue), SOFA=#27AE60 (green).
C_FULL, C_ABL, C_AP, C_SF = "#C0392B", "#000000", "#2E86C1", "#27AE60"

# ROC
plt.figure(figsize=(8, 6))
shade_std(mean_fpr, mtf, stf, C_FULL); plt.plot(mean_fpr, mtf, label=f"LGBM (AUC: {auroc_full.mean():.2f})", linewidth=2, color=C_FULL, zorder=3)
shade_std(mean_fpr, mta, sta, C_ABL);  plt.plot(mean_fpr, mta, label=f"LGBM ablation (AUC: {auroc_abl.mean():.2f})", linewidth=2, color=C_ABL, zorder=3)
shade_std(mean_fpr, mtp, stp, C_AP);   plt.plot(mean_fpr, mtp, label=f"APACHE II (AUC: {auroc_ap.mean():.2f})", linewidth=2, color=C_AP, zorder=3)
shade_std(mean_fpr, mts, sts, C_SF, alpha=0.12); plt.plot(mean_fpr, mts, label=f"SOFA (AUC: {auroc_sf.mean():.2f})", linewidth=2, color=C_SF, zorder=3)
plt.xlim([0,1]); plt.ylim([0,1.05]); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Mean ROC Curves: LGBM vs APACHE II vs SOFA")
plt.legend(loc="best", frameon=True, facecolor="white", framealpha=0.85, edgecolor="none", fontsize=13, handlelength=2.0, labelspacing=0.3, borderpad=0.3)
plt.grid(False); plt.tight_layout()
out = os.path.join(OUTDIR, "Fig_Ablation_ROC.png"); plt.savefig(out, dpi=400, bbox_inches="tight"); plt.close(); print("Saved:", out)

# PR 
plt.figure(figsize=(8, 6))
shade_std(mean_recall, mpf, spf, C_FULL); plt.plot(mean_recall, mpf, label=f"LGBM (AUPRC: {auprc_full.mean():.2f})", linewidth=2, color=C_FULL, zorder=3)
shade_std(mean_recall, mpa, spa, C_ABL);  plt.plot(mean_recall, mpa, label=f"LGBM ablation (AUPRC: {auprc_abl.mean():.2f})", linewidth=2, color=C_ABL, zorder=3)
shade_std(mean_recall, mpp, spp, C_AP);   plt.plot(mean_recall, mpp, label=f"APACHE II (AUPRC: {auprc_ap.mean():.2f})", linewidth=2, color=C_AP, zorder=3)
shade_std(mean_recall, mps, sps, C_SF, alpha=0.12); plt.plot(mean_recall, mps, label=f"SOFA (AUPRC: {auprc_sf.mean():.2f})", linewidth=2, color=C_SF, zorder=3)
plt.xlim([0,1]); plt.ylim([0,1.05]); plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Mean P-R Curves: LGBM vs APACHE II vs SOFA")
plt.legend(loc="best", frameon=True, facecolor="white", framealpha=0.85, edgecolor="none", fontsize=13, handlelength=2.0, labelspacing=0.3, borderpad=0.3)
plt.grid(False); plt.tight_layout()
out = os.path.join(OUTDIR, "Fig_Ablation_PR.png"); plt.savefig(out, dpi=400, bbox_inches="tight"); plt.close(); print("Saved:", out)
