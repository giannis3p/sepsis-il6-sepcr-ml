# Figure: LightGBM Gini (split) Feature Importance  (Top 20)
# Model: LGBM only.
import os, re, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

import sys
DATA   = sys.argv[1] if len(sys.argv) > 1 else r""
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else r""
df = pd.read_csv(DATA)

target_column = "SEPSIS"
features_to_drop = [target_column, 'APACHE_II', 'SOFA']   # severity scores dropped
X = df.drop(columns=features_to_drop)
y = df[target_column]
X = X.rename(columns={"INOTROPES/VASOPRESSORS (1=YES, 0=NO)": "INO_VASO",
                      'ADMISSION DIAGNOSIS': 'ADM_DIAGNOSIS'})
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
print("Number of features used:", X.shape[1])
binary_cols     = [c for c in X.columns if X[c].dropna().nunique() == 2]
continuous_cols = [c for c in X.columns if c not in binary_cols]

preprocess = ColumnTransformer(
    transformers=[("bin",  SimpleImputer(strategy="most_frequent"), binary_cols),
                  ("cont", SimpleImputer(strategy="median"),        continuous_cols)],
    remainder="drop")

lgb_best = lgb.LGBMClassifier(
    objective="binary", n_estimators=296, learning_rate=0.021758255723790025,
    num_leaves=8, max_depth=4, min_child_samples=17, min_split_gain=0.34081398506638294,
    lambda_l1=8.226779549337834e-05, lambda_l2=3.0581582963155785e-10,
    feature_fraction=0.5437381928004323, bagging_fraction=0.7389041657117149,
    bagging_freq=0, scale_pos_weight=1.1564125380200863, verbose=-1, random_state=42, n_jobs=-1)
model = Pipeline([("prep", preprocess), ("lgb", lgb_best)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="roc_auc")
print("CV AUROC mean+/-sd:", cv_scores.mean(), cv_scores.std(ddof=1))
y_oof = cross_val_predict(model, X_train, y_train, cv=10, method="predict_proba")[:, 1]
print("OOF AUROC:", roc_auc_score(y_train, y_oof), "OOF AUPRC:", average_precision_score(y_train, y_oof))
model.fit(X_train, y_train); p_test = model.predict_proba(X_test)[:, 1]
print("Test AUROC:", roc_auc_score(y_test, p_test), "Test AUPRC:", average_precision_score(y_test, p_test))

# Repeated StratifiedShuffleSplit + feature importances
sss = StratifiedShuffleSplit(n_splits=50, test_size=0.2, random_state=42)
auroc_scores, auprc_scores, all_importances = [], [], []
for tr, te in sss.split(X, y):
    model.fit(X.iloc[tr], y.iloc[tr]); p = model.predict_proba(X.iloc[te])[:, 1]
    auroc_scores.append(roc_auc_score(y.iloc[te], p)); auprc_scores.append(average_precision_score(y.iloc[te], p))
    all_importances.append(model.named_steps["lgb"].feature_importances_)
auroc_scores, auprc_scores = np.array(auroc_scores), np.array(auprc_scores)
all_importances = np.vstack(all_importances)
print(f"Shuffle AUROC mean: {auroc_scores.mean():.4f}, 95% CI: ({np.percentile(auroc_scores,2.5):.4f}, {np.percentile(auroc_scores,97.5):.4f})")
print(f"Shuffle AUPRC mean: {auprc_scores.mean():.4f}, 95% CI: ({np.percentile(auprc_scores,2.5):.4f}, {np.percentile(auprc_scores,97.5):.4f})")

feature_names = [name.split("__", 1)[-1] for name in model.named_steps["prep"].get_feature_names_out()]
fi = pd.DataFrame({"feature": feature_names,
                   "mean_importance": all_importances.mean(axis=0),
                   "std_importance":  all_importances.std(axis=0, ddof=1)}).sort_values("mean_importance", ascending=False)
print("\nTop 20 Mean Feature Importances:\n", fi.head(20).to_string(index=False))

# Plot
top_n = min(20, len(fi)); fi_top = fi.head(top_n).iloc[::-1]
plt.figure(figsize=(8, max(5, 0.3 * top_n)), dpi=500)
plt.barh(fi_top["feature"], fi_top["mean_importance"], xerr=fi_top["std_importance"],
         capsize=4, color="#5DADE2", edgecolor="black")
plt.xlabel("Mean Feature Importance (+/- SD across shuffles)", fontsize=18, color="black")
plt.ylabel("Features", fontsize=18, color="black")
plt.title("LightGBM Feature Importance (Top 20)", fontsize=18, color="black", pad=15)
plt.xticks(fontsize=14, color="black"); plt.yticks(fontsize=13, color="black")
plt.tight_layout()
out = os.path.join(OUTDIR, "Fig_Gini_LGBM_feature_importance.png")
plt.savefig(out, dpi=500, bbox_inches="tight"); print("Saved:", out)
