# Section 3.2 - LGBM calibration (50-resample) before/after recalibration-in-the-large,
# and DCA crossing thresholds. Produces the numbers reported in the text and Figure 1A/1B box.
# Usage: python calibration_dca_stats.py [data.csv]   (default = combined_1257.csv)
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import statsmodels.api as sm, lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import brier_score_loss
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DATA = sys.argv[1] if len(sys.argv) > 1 else r"../data/combined_1257.csv"
LGB = dict(objective="binary", n_estimators=296, learning_rate=0.021758255723790025, num_leaves=8, max_depth=4,
    min_child_samples=17, min_split_gain=0.34081398506638294, lambda_l1=8.226779549337834e-05,
    lambda_l2=3.0581582963155785e-10, feature_fraction=0.5437381928004323, bagging_fraction=0.7389041657117149,
    bagging_freq=0, scale_pos_weight=1.1564125380200863, verbose=-1, random_state=42, n_jobs=-1)
df = pd.read_csv(DATA)
X = df.drop(columns=["SEPSIS","APACHE_II","SOFA"], errors="ignore").select_dtypes("number").copy(); y = df["SEPSIS"].astype(int); yv = y.values
b = [c for c in X.columns if X[c].dropna().nunique() == 2]; c = [x for x in X.columns if x not in b]
model = Pipeline([("prep", ColumnTransformer([("bin", SimpleImputer(strategy="most_frequent"), b),
                  ("cont", SimpleImputer(strategy="median"), c)], remainder="drop")), ("lgb", lgb.LGBMClassifier(**LGB))])
print(f"{DATA}: n={len(df)}, prevalence={yv.mean()*100:.1f}%")

def calib_row(yt, p):
    yt = np.asarray(yt).astype(int); p = np.clip(np.asarray(p), 1e-15, 1-1e-15); L = np.log(p/(1-p))
    out = dict(Brier=brier_score_loss(yt, p), Slope=np.nan, Intercept=np.nan)
    try: out["Slope"] = sm.Logit(yt, sm.add_constant(L)).fit(disp=0).params[1]
    except Exception: pass
    try: out["Intercept"] = sm.GLM(yt, np.ones_like(L), family=sm.families.Binomial(), offset=L).fit().params[0]
    except Exception: pass
    return out
def fit_int(Xtr, ytr, cv=5):
    p = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]; p = np.clip(p, 1e-15, 1-1e-15); L = np.log(p/(1-p))
    return sm.GLM(np.asarray(ytr).astype(int), np.ones_like(L), family=sm.families.Binomial(), offset=L).fit().params[0]
def app(p, a): p = np.clip(p, 1e-15, 1-1e-15); return 1/(1+np.exp(-(a+np.log(p/(1-p)))))

sss = StratifiedShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
raw, rec = [], []
for tr, te in sss.split(X, y):
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    p = model.fit(Xtr, ytr).predict_proba(Xte)[:, 1]; pr = app(p, fit_int(Xtr, ytr))
    raw.append(calib_row(yte, p)); rec.append(calib_row(yte, pr))
def s(rows, k):
    a = np.array([r[k] for r in rows]); a = a[~np.isnan(a)]; return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)
print("\n=== CALIBRATION (50 resamples) ===")
for lbl, rows in [("BEFORE recalibration", raw), ("AFTER recalibration-in-the-large", rec)]:
    print(f"  {lbl}: Slope {s(rows,'Slope')[0]:.2f} ({s(rows,'Slope')[1]:.2f}-{s(rows,'Slope')[2]:.2f}) | "
          f"Intercept {s(rows,'Intercept')[0]:+.2f} ({s(rows,'Intercept')[1]:+.2f} to {s(rows,'Intercept')[2]:+.2f}) | "
          f"Brier {s(rows,'Brier')[0]:.3f} ({s(rows,'Brier')[1]:.3f}-{s(rows,'Brier')[2]:.3f})")

# DCA crossings
def make_lr(): return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
ben = {"APACHE II": df[["APACHE_II"]].apply(pd.to_numeric, errors="coerce"), "SOFA": df[["SOFA"]].apply(pd.to_numeric, errors="coerce")}
def nb(yt, p, th):
    yt = np.asarray(yt).astype(int); n = len(yt)
    return np.array([((p>=t)&(yt==1)).sum()/n - ((p>=t)&(yt==0)).sum()/n*(t/(1-t)) for t in th])
th = np.linspace(0.01, 0.6, 120); NBb = {k: [] for k in ben}; NBl = []
for tr, te in sss.split(X, y):
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    model.fit(Xtr, ytr); a = fit_int(Xtr, ytr); NBl.append(nb(yte, app(model.predict_proba(Xte)[:, 1], a), th))
    for k, Xb in ben.items():
        mm = make_lr().fit(Xb.iloc[tr], ytr); NBb[k].append(nb(yte, mm.predict_proba(Xb.iloc[te])[:, 1], th))
print("\n=== DCA crossings ===")
print(f"  Treat-all crossing = prevalence = {yv.mean():.3f}")
for k in ben:
    mnb = np.vstack(NBb[k]).mean(0); below = np.where(mnb <= 0)[0]
    print(f"  {k}: mean net benefit reaches <=0 at threshold ~ {th[below[0]]:.3f}" if len(below) else f"  {k}: stays >0")
lnb = np.vstack(NBl).mean(0); print(f"  LGBM min mean net benefit over 0.01-0.60 = {lnb.min():+.4f} (stays>0: {lnb.min()>0})")
