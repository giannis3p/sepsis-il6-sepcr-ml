# Table 3 - feature ablation for LGBM: full vs (biomarkers removed) vs (vital signs removed).
# Reports AUROC/AUPRC (mean, 95% CI over 50 splits) + paired significance vs full model.
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DATA = sys.argv[1] if len(sys.argv) > 1 else r"../data/combined_1257.csv"
DROP = ["SEPSIS", "APACHE_II", "SOFA"]
BIOM = ["sEPCR", "IL-6", "CRP", "PCT"]
VITALS = ["HEART RATE", "TEMPERATURE", "RR", "MAP"]
LGB = dict(objective="binary", n_estimators=296, learning_rate=0.021758255723790025, num_leaves=8, max_depth=4,
    min_child_samples=17, min_split_gain=0.34081398506638294, lambda_l1=8.226779549337834e-05,
    lambda_l2=3.0581582963155785e-10, feature_fraction=0.5437381928004323, bagging_fraction=0.7389041657117149,
    bagging_freq=0, scale_pos_weight=1.1564125380200863, verbose=-1, random_state=42, n_jobs=-1)

df = pd.read_csv(DATA); y = df["SEPSIS"].astype(int)
Xfull = df.drop(columns=DROP, errors="ignore").select_dtypes("number").copy()
def pipe(X):
    b = [c for c in X.columns if X[c].dropna().nunique() == 2]; c = [x for x in X.columns if x not in b]
    return Pipeline([("prep", ColumnTransformer([("bin", SimpleImputer(strategy="most_frequent"), b),
                     ("cont", SimpleImputer(strategy="median"), c)], remainder="drop")), ("lgb", lgb.LGBMClassifier(**LGB))])
sss = StratifiedShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
def evalX(X):
    m = pipe(X); au, ap = [], []
    for tr, te in sss.split(X, y):
        m.fit(X.iloc[tr], y.iloc[tr]); p = m.predict_proba(X.iloc[te])[:, 1]
        au.append(roc_auc_score(y.iloc[te], p)); ap.append(average_precision_score(y.iloc[te], p))
    return np.array(au), np.array(ap)
def fmt(a): return f"{a.mean():.3f} ({np.percentile(a,2.5):.3f}-{np.percentile(a,97.5):.3f})"
def rbr(d):
    d = d[d != 0]; r = stats.rankdata(np.abs(d)); return (r[d>0].sum()-r[d<0].sum())/(r[d>0].sum()+r[d<0].sum())

variants = [("LGBM", Xfull),
            ("without inflammatory and endothelial biomarkers", Xfull.drop(columns=BIOM, errors="ignore")),
            ("without vital signs", Xfull.drop(columns=VITALS, errors="ignore"))]
print(f"TABLE 3 ({DATA}, n={len(df)}, sepsis={int(y.sum())})")
res = {}
print(f"{'Model':52}{'AUC (95% CI)':>24}{'AUPRC (95% CI)':>24}")
for name, X in variants:
    au, ap = evalX(X); res[name] = (au, ap); print(f"{name:52}{fmt(au):>24}{fmt(ap):>24}")
fa, fp = res["LGBM"]
for name in [v[0] for v in variants[1:]]:
    au, ap = res[name]; dA, dP = fa-au, fp-ap
    print(f"\n  vs full: {name}")
    print(f"    dAUC ={dA.mean():+.3f} (t p={stats.ttest_rel(fa,au).pvalue:.1e}, Wilcoxon p={stats.wilcoxon(fa,au).pvalue:.1e}, r={rbr(dA):+.2f})")
    print(f"    dAUPRC={dP.mean():+.3f} (t p={stats.ttest_rel(fp,ap).pvalue:.1e}, Wilcoxon p={stats.wilcoxon(fp,ap).pvalue:.1e}, r={rbr(dP):+.2f})")
