# Section 3.3 - SHAP dependence shape check: binned mean SHAP by biomarker value.
# Confirms IL-6 and PCT are monotonic (threshold-like) while CRP and sEPCR are non-monotonic.
# Usage: python shap_monotonicity.py [data.csv]   (default = combined_1257.csv)
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb, shap
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DATA = sys.argv[1] if len(sys.argv) > 1 else r"../data/combined_1257.csv"
LGB = dict(objective="binary", n_estimators=296, learning_rate=0.021758255723790025, num_leaves=8, max_depth=4,
    min_child_samples=17, min_split_gain=0.34081398506638294, lambda_l1=8.226779549337834e-05,
    lambda_l2=3.0581582963155785e-10, feature_fraction=0.5437381928004323, bagging_fraction=0.7389041657117149,
    bagging_freq=0, scale_pos_weight=1.1564125380200863, verbose=-1, random_state=42, n_jobs=-1)
df = pd.read_csv(DATA)
X = df.drop(columns=["SEPSIS","APACHE_II","SOFA"], errors="ignore").select_dtypes("number").copy(); y = df["SEPSIS"].astype(int)
b = [c for c in X.columns if X[c].dropna().nunique() == 2]; c = [x for x in X.columns if x not in b]
m = Pipeline([("prep", ColumnTransformer([("bin", SimpleImputer(strategy="most_frequent"), b),
              ("cont", SimpleImputer(strategy="median"), c)], remainder="drop")), ("lgb", lgb.LGBMClassifier(**LGB))]).fit(X, y)
Xt = m.named_steps["prep"].transform(X); fo = b + c
sv = shap.TreeExplainer(m.named_steps["lgb"]).shap_values(Xt)
if isinstance(sv, list): sv = sv[1]
if sv.ndim == 3: sv = sv[..., 1]

def profile(name, edges):
    j = fo.index(name); xv = Xt[:, j]; s = sv[:, j]; vals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (xv >= lo) & (xv < hi); vals.append(s[mask].mean() if mask.sum() else np.nan)
    v = np.array([x for x in vals if not np.isnan(x)]); inc = np.all(np.diff(v) >= -0.03)
    print(f"{name:6}: " + " ".join(f"{lo:g}-{hi:g}:{mv:+.2f}" for lo,hi,mv in zip(edges[:-1],edges[1:],vals))
          + f"   -> {'MONOTONIC' if inc else 'NON-MONOTONIC'}")
print(f"{DATA}: binned mean SHAP by biomarker value\n")
profile("IL-6",  [0,100,150,200,300,500,4000])
profile("PCT",   [0,1,1.5,2,3,5,70])
profile("CRP",   [0,2,10,35,60,120,300])
profile("sEPCR", [0,30,70,110,180,250,450])
