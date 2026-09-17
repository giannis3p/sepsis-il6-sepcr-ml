# Figure: Calibration (LOESS + 95% band) + Decision Curve Analysis
# Model: LGBM (recal-in-the-large) vs APACHE II / SOFA benchmarks.
import os, re, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

import sys
DATA   = sys.argv[1] if len(sys.argv) > 1 else r""
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else r""
df = pd.read_csv(DATA)

target_column = "SEPSIS"
X = df.drop(columns=[target_column, 'APACHE_II', 'SOFA'])
y = df[target_column]
X = X.rename(columns={"INOTROPES/VASOPRESSORS (1=YES, 0=NO)": "INO_VASO", 'ADMISSION DIAGNOSIS': 'ADM_DIAGNOSIS'})
X = X.select_dtypes(include=["number"]).copy()
def sanitize_columns(cols):
    cleaned, seen = [], {}
    for c in cols:
        new = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)); new = re.sub(r"_+", "_", new).strip("_")
        if new == "": new = "col"
        if new in seen: seen[new] += 1; new = f"{new}_{seen[new]}"
        else: seen[new] = 0
        cleaned.append(new)
    return cleaned
X.columns = sanitize_columns(X.columns)
binary_cols     = [c for c in X.columns if X[c].dropna().nunique() == 2]
continuous_cols = [c for c in X.columns if c not in binary_cols]
preprocess = ColumnTransformer([("bin", SimpleImputer(strategy="most_frequent"), binary_cols),
                                ("cont", SimpleImputer(strategy="median"), continuous_cols)], remainder="drop")
lgb_best = lgb.LGBMClassifier(objective="binary", n_estimators=296, learning_rate=0.021758255723790025,
    num_leaves=8, max_depth=4, min_child_samples=17, min_split_gain=0.34081398506638294,
    lambda_l1=8.226779549337834e-05, lambda_l2=3.0581582963155785e-10, feature_fraction=0.5437381928004323,
    bagging_fraction=0.7389041657117149, bagging_freq=0, scale_pos_weight=1.1564125380200863,
    verbose=-1, random_state=42, n_jobs=-1)
model = Pipeline([("prep", preprocess), ("lgb", lgb_best)])

yv = np.asarray(y).astype(int)

# benchmarks
def make_lr():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000))])
benchmarks   = {"APACHE II": df[["APACHE_II"]].apply(pd.to_numeric, errors="coerce"),
                "SOFA":      df[["SOFA"]].apply(pd.to_numeric, errors="coerce")}
bench_colors = {"APACHE II": "#2E86C1", "SOFA": "#27AE60"}

def recal_intercept(est, Xtr, ytr, cv=5):
    p = cross_val_predict(est, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    p = np.clip(p, 1e-15, 1-1e-15); L = np.log(p/(1-p))
    return sm.GLM(np.asarray(ytr).astype(int), np.ones_like(L), family=sm.families.Binomial(), offset=L).fit().params[0]
def apply_int(p, a):
    p = np.clip(p, 1e-15, 1-1e-15); return 1/(1+np.exp(-(a+np.log(p/(1-p)))))
def nb(yt, p, th):
    yt = np.asarray(yt).astype(int); n = len(yt)
    return np.array([((p>=t)&(yt==1)).sum()/n - ((p>=t)&(yt==0)).sum()/n*(t/(1-t)) for t in th])
def nb_all(yt, th):
    prev = np.asarray(yt).astype(int).mean(); return prev - (1-prev)*(th/(1-th))

# ---------- calibration: pooled OOF + LOESS band ----------
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
p_oof = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
L = np.log(np.clip(p_oof,1e-15,1-1e-15)/(1-np.clip(p_oof,1e-15,1-1e-15)))
a_disp = sm.GLM(yv, np.ones_like(L), family=sm.families.Binomial(), offset=L).fit().params[0]
p_cal = apply_int(p_oof, a_disp)

# ---- calibration stats via the 50-RESAMPLE method
# slope & Brier are taken BEFORE recalibration; intercept is taken AFTER recalibration-in-the-large.
# Collected per split inside the DCA loop below (reuses those fits); cal_text is built afterwards.
def _calib(yt, p):
    yt = np.asarray(yt).astype(int); p = np.clip(np.asarray(p), 1e-15, 1-1e-15); Lz = np.log(p/(1-p))
    slope = sm.Logit(yt, sm.add_constant(Lz)).fit(disp=0).params[1]
    inter = sm.GLM(yt, np.ones_like(Lz), family=sm.families.Binomial(), offset=Lz).fit().params[0]
    return slope, inter, float(np.mean((p-yt)**2))
cal_slope, cal_brier, cal_int_after = [], [], []

ub = np.percentile(p_cal, 97.5); grid = np.linspace(0, ub, 100)
def loess(yy, pp, g, frac=0.66):
    s = sm.nonparametric.lowess(yy, pp, frac=frac, it=0, return_sorted=True)
    return np.clip(np.interp(g, s[:,0], s[:,1]), 0, 1)
mid = loess(yv, p_cal, grid)
rng2 = np.random.default_rng(42)
cb = np.vstack([loess(yv[i], p_cal[i], grid) for i in (rng2.integers(0,len(yv),len(yv)) for _ in range(500))])
lo_b, hi_b = np.percentile(cb, [2.5, 97.5], axis=0)

bench_cal = {}
for name, Xb in benchmarks.items():
    p_b = cross_val_predict(make_lr(), Xb, y, cv=skf, method="predict_proba")[:, 1]
    Lb = np.log(np.clip(p_b,1e-15,1-1e-15)/(1-np.clip(p_b,1e-15,1-1e-15)))
    ab = sm.GLM(yv, np.ones_like(Lb), family=sm.families.Binomial(), offset=Lb).fit().params[0]
    pcb = apply_int(p_b, ab); m = loess(yv, pcb, grid)
    lo_r, hi_r = np.percentile(pcb, [1, 99])
    bench_cal[name] = np.where((grid >= lo_r) & (grid <= hi_r), m, np.nan)

# DCA: 50-split net benefit
th = np.linspace(0.01, 0.6, 120)
sss = StratifiedShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
NB_lgb = []; NB_bench = {name: [] for name in benchmarks}
for tr, te in sss.split(X, y):
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    model.fit(Xtr, ytr); a = recal_intercept(model, Xtr, ytr, cv=5)
    p_raw = model.predict_proba(Xte)[:, 1]; p_rec = apply_int(p_raw, a)
    sl, _, br = _calib(yte, p_raw); _, ia, _ = _calib(yte, p_rec)   # slope/Brier before, intercept after recal
    cal_slope.append(sl); cal_brier.append(br); cal_int_after.append(ia)
    NB_lgb.append(nb(yte, p_rec, th))
    for name, Xb in benchmarks.items():
        b = make_lr().fit(Xb.iloc[tr], ytr); NB_bench[name].append(nb(yte, b.predict_proba(Xb.iloc[te])[:,1], th))
NB_lgb = np.vstack(NB_lgb); NB_bench = {k: np.vstack(v) for k, v in NB_bench.items()}

def _mci(a): a = np.array(a); a = a[~np.isnan(a)]; return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)
_sm, _sl, _sh = _mci(cal_slope); _bm, _bl, _bh = _mci(cal_brier); _im, _il, _ih = _mci(cal_int_after)
cal_text = (f"LightGBM:\nSlope {_sm:.2f} ({_sl:.2f}-{_sh:.2f})\n"
            f"Intercept {_im:+.2f} ({_il:+.2f} to {_ih:+.2f})\n"
            f"Brier {_bm:.3f} ({_bl:.3f}-{_bh:.3f})")
print("CAL(50-resample):", cal_text.replace(chr(10), " | "))

plt.rcParams.update({"font.size":14,"font.weight":"bold","axes.titlesize":18,"axes.titleweight":"bold",
                     "axes.labelsize":16,"axes.labelweight":"bold","xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":12})
fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
# Panel A: calibration
ax[0].plot([0, ub], [0, ub], "k--", lw=1, label="Ideal calibration")
ax[0].plot(grid, mid, color="#C0392B", lw=2, label="LightGBM (LOESS)")
ax[0].fill_between(grid, lo_b, hi_b, color="#C0392B", alpha=0.18, label="LightGBM 95% CI")
for name, m in bench_cal.items():
    ax[0].plot(grid, m, color=bench_colors[name], lw=2, label=f"{name} (LOESS)")
ax[0].set_xlim(0, ub); ax[0].set_ylim(0, ub)
ax[0].set_xlabel("Predicted probability"); ax[0].set_ylabel("Observed proportion")
ax[0].set_title("Calibration", loc="left")
ax[0].text(0.04, 0.96, cal_text, transform=ax[0].transAxes, va="top", fontsize=12,
           bbox=dict(boxstyle="round", fc="white", ec="0.8"))
axh = ax[0].twinx(); axh.hist(p_cal, bins=30, range=(0, ub), color="0.85"); axh.set_yticks([])
ax[0].set_zorder(axh.get_zorder()+1); ax[0].patch.set_visible(False)
ax[0].legend(loc="lower right", frameon=False, fontsize=11)
# Panel B: DCA
def band(a, A, color, label):
    m = A.mean(0); lo, hi = np.percentile(A, [2.5, 97.5], axis=0)
    a.plot(th, m, color=color, lw=2, label=label); a.fill_between(th, lo, hi, color=color, alpha=0.18)
band(ax[1], NB_lgb, "#C0392B", "LightGBM")
for name, A in NB_bench.items(): band(ax[1], A, bench_colors[name], name)
ax[1].plot(th, nb_all(y, th), color="0.5", ls="--", lw=1.5, label="Treat all")
ax[1].axhline(0, color="0.3", ls=":", lw=1.5, label="Treat none")
ax[1].set_ylim(-0.05, None); ax[1].set_xlim(0, 0.6)
ax[1].set_xlabel("Threshold probability"); ax[1].set_ylabel("Net benefit")
ax[1].set_title("Decision curve analysis", loc="left"); ax[1].legend(frameon=False, fontsize=12)
plt.tight_layout()
out = os.path.join(OUTDIR, "Fig_Calibration_DecisionCurve.png")
fig.savefig(out, dpi=400, bbox_inches="tight"); print("Saved:", out)
