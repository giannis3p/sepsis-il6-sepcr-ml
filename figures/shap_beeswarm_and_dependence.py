# Figure: SHAP beeswarm (layered_violin / violin / bar) over 50 shuffles
#         + SHAP dependence panels (Figure 4 a-d: IL-6, PCT, CRP, sEPCR)
# Model: LGBM only.
import os, re, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
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

shap_features_20 = ["IL_6","CRP","INO_VASO","PCT","RR","GLUCOSE","LDH","FIB","APTT","PH",
                    "PCO2","BUN","MG","NEUT","PT","ADM_DIAGNOSIS","sEPCR","AGE","HCT","LYMF"]

def shap_to_2d_class1(shap_out, n_features_expected=None):
    if isinstance(shap_out, list):
        arr = np.asarray(shap_out[1]) if len(shap_out) >= 2 else np.asarray(shap_out[-1])
    else:
        arr = np.asarray(shap_out)
    if arr.ndim == 2: out = arr
    elif arr.ndim == 3: out = arr[:, :, 1] if arr.shape[2] >= 2 else arr[:, :, 0]
    else: raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")
    if n_features_expected is not None and out.shape[1] == n_features_expected + 1: out = out[:, :-1]
    return out

sss = StratifiedShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
all_shap_rows, all_x_rows, selected_idx, selected_names = [], [], None, None
for i, (tr, te) in enumerate(sss.split(X, y), start=1):
    X_tr, X_te, y_tr = X.iloc[tr].copy(), X.iloc[te].copy(), y.iloc[tr].copy()
    m = clone(model); m.fit(X_tr, y_tr)
    prep_fitted, lgb_fitted = m.named_steps["prep"], m.named_steps["lgb"]
    X_te_tr = prep_fitted.transform(X_te)
    fn_clean = [n.split("__", 1)[-1] for n in prep_fitted.get_feature_names_out()]
    if selected_idx is None:
        missing = [f for f in shap_features_20 if f not in fn_clean]
        if missing: raise ValueError(f"Requested SHAP features not present: {missing}\nAvailable: {fn_clean}")
        selected_idx = [fn_clean.index(f) for f in shap_features_20]; selected_names = shap_features_20
    X_te_tr_df = pd.DataFrame(X_te_tr, columns=fn_clean, index=X_te.index)
    sv = shap.TreeExplainer(lgb_fitted).shap_values(X_te_tr_df)
    sv2d = shap_to_2d_class1(sv, n_features_expected=X_te_tr_df.shape[1])
    all_shap_rows.append(sv2d[:, selected_idx]); all_x_rows.append(X_te_tr_df.iloc[:, selected_idx].values)
    if i % 10 == 0 or i == 1: print(f"Split {i:02d}/50 | test n={len(X_te)}")
all_shap_20 = np.vstack(all_shap_rows); all_x_20 = np.vstack(all_x_rows)
X_plot_20 = pd.DataFrame(all_x_20, columns=selected_names)
print("Stacked:", all_shap_20.shape, X_plot_20.shape)

for ptype, tag, sz in [("dot","dot",(8,6)), ("layered_violin","layeredviolin",(8,6)), ("violin","violin",(8,5)), ("bar","bar",(8,6))]:
    plt.figure(dpi=500 if ptype=="layered_violin" else 300)
    shap.summary_plot(all_shap_20, X_plot_20, plot_type=ptype, max_display=20, plot_size=sz, show=False)
    out = os.path.join(OUTDIR, f"Fig_SHAP_beeswarm_{tag}.png")
    plt.savefig(out, dpi=400, bbox_inches="tight"); plt.close(); print("Saved:", out)

# dependence panels
dep_features = ["IL_6", "PCT", "CRP", "sEPCR"]; panel_letters = ["a", "b", "c", "d"]
plt.rcParams["figure.dpi"] = 300; dot_size, alpha = 12, 0.6
cols = list(X_plot_20.columns)
for letter, f in zip(panel_letters, dep_features):
    j = cols.index(f); x = X_plot_20[f].values; s = all_shap_20[:, j]
    lim = np.nanpercentile(np.abs(s), 99)
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(x, s, c=s, cmap="coolwarm", vmin=-lim, vmax=lim, s=dot_size, alpha=alpha, linewidths=0)
    cb = plt.colorbar(sc); cb.set_label("SHAP value", rotation=270, labelpad=15)
    plt.axhline(y=0, color="black", linewidth=1)
    plt.xlabel(f); plt.ylabel(f"SHAP value for {f}"); plt.title(f"SHAP Main Effect: {f}")
    plt.tight_layout()
    out = os.path.join(OUTDIR, f"Fig4{letter}_SHAP_dependence_{f}.png")
    plt.savefig(out, dpi=400, bbox_inches="tight"); plt.close(); print("Saved:", out)
