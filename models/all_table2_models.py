# Table 2: 5-model discrimination (AUROC + AUPRC, mean & 95% CI)
# Models: LR, RF, XGBoost, AdaBoost, LGBM
# 50-split StratifiedShuffleSplit, test_size=0.2, random_state=42.
# Feature set: drop SEPSIS, APACHE_II, SOFA (Sepsis-3 leakage); numeric only.
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DATA  = sys.argv[1] if len(sys.argv) > 1 else r""
LABEL = sys.argv[2] if len(sys.argv) > 2 else DATA.split("/")[-1]
DROP_BASE = ["SEPSIS", "APACHE_II", "SOFA"]

LGB = dict(objective="binary", n_estimators=296, learning_rate=0.021758255723790025, num_leaves=8, max_depth=4,
    min_child_samples=17, min_split_gain=0.34081398506638294, lambda_l1=8.226779549337834e-05,
    lambda_l2=3.0581582963155785e-10, feature_fraction=0.5437381928004323, bagging_fraction=0.7389041657117149,
    bagging_freq=0, scale_pos_weight=1.1564125380200863, verbose=-1, random_state=42, n_jobs=-1)

def bc(X):
    b = [c for c in X.columns if X[c].dropna().nunique() == 2]
    return b, [c for c in X.columns if c not in b]
def make(name, X):
    b, c = bc(X)
    if name == "LR":
        cont = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
        pre = ColumnTransformer([("bin", SimpleImputer(strategy="most_frequent"), b), ("cont", cont, c)], remainder="drop")
        clf = LogisticRegression(solver="liblinear", penalty="l2", C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    else:
        pre = ColumnTransformer([("bin", SimpleImputer(strategy="most_frequent"), b),
                                 ("cont", SimpleImputer(strategy="median"), c)], remainder="drop")
        clf = dict(
            RF=RandomForestClassifier(criterion="log_loss", n_estimators=357, max_depth=17, min_samples_split=18,
                min_samples_leaf=1, max_features=None, bootstrap=True, class_weight="balanced", random_state=42, n_jobs=-1),
            XGB=xgb.XGBClassifier(objective="binary:logistic", n_estimators=1141, max_depth=5, learning_rate=0.028118802486807332,
                subsample=0.7384932380965475, colsample_bytree=0.5244446476630595, colsample_bynode=0.6862919889129073,
                scale_pos_weight=6.086088916811487, gamma=3.1593524010733565, min_child_weight=2.351524934064766,
                reg_alpha=4.256201793605995e-07, reg_lambda=0.7990226961765767, max_delta_step=8, eval_metric="logloss",
                random_state=42, n_jobs=-1),
            AdaBoost=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=28, min_samples_split=31,
                ccp_alpha=0.01540754359337574, random_state=42), n_estimators=672, learning_rate=0.5767179725608667, random_state=42),
            LGBM=lgb.LGBMClassifier(**LGB))[name]
    return Pipeline([("prep", pre), ("clf", clf)])

df = pd.read_csv(DATA)
X = df.drop(columns=DROP_BASE, errors="ignore").select_dtypes(include=["number"]).copy()
y = df["SEPSIS"].astype(int)
print(f"########## TABLE 2 : {LABEL}  (n={len(df)}, sepsis={int(y.sum())}, prev={y.mean()*100:.1f}%) ##########", flush=True)
sss = StratifiedShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
print(f"{'Model':9}{'AUROC (95% CI)':>26}{'AUPRC (95% CI)':>26}")
for name in ["LR", "RF", "XGBoost", "AdaBoost", "LGBM"]:
    m = make("XGB" if name == "XGBoost" else name, X)
    au, ap = [], []
    for tr, te in sss.split(X, y):
        m.fit(X.iloc[tr], y.iloc[tr]); p = m.predict_proba(X.iloc[te])[:, 1]
        au.append(roc_auc_score(y.iloc[te], p)); ap.append(average_precision_score(y.iloc[te], p))
    au, ap = np.array(au), np.array(ap)
    print(f"{name:9}{f'{au.mean():.3f} ({np.percentile(au,2.5):.3f}-{np.percentile(au,97.5):.3f})':>26}"
          f"{f'{ap.mean():.3f} ({np.percentile(ap,2.5):.3f}-{np.percentile(ap,97.5):.3f})':>26}", flush=True)
print("DONE", flush=True)
