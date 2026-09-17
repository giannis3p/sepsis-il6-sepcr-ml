# Figure 2: Biomarker threshold association with SEPSIS (FDR-corrected)
# 2x2 grid: a=IL-6, b=PCT, c=CRP, d=sEPCR  (manuscript order/colors)
# Statistical test: per-cutpoint chi-square + Benjamini-Hochberg FDR
import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")

import sys
DATA   = sys.argv[1] if len(sys.argv) > 1 else r""
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else r""
df = pd.read_csv(DATA)
y = df["SEPSIS"].astype(int)
print(f"Data: {os.path.basename(DATA)} | n={len(df)}, sepsis={int(y.sum())} ({y.mean()*100:.1f}%)")

def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n); pr = 1.0
    for i in range(n-1, -1, -1):
        v = p[o[i]] * n / (i+1); pr = min(pr, v); q[o[i]] = pr
    return np.clip(q, 0, 1)

panels = [("A", "IL-6",  "darkred"),
          ("B", "PCT",   "navy"),
          ("C", "CRP",   "black"),
          ("D", "sEPCR", "green")]

plt.rcParams.update({"font.size": 13})
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
for (letter, bm, color), ax in zip(panels, axs.ravel()):
    v = pd.to_numeric(df[bm], errors="coerce"); m = v.notna()
    vv = v[m].values; yy = df["SEPSIS"][m].astype(int).values
    cand = np.unique(np.nanpercentile(vv, np.arange(5, 96, 3)))
    ts, ps = [], []
    for t in cand:
        hi = vv > t
        if hi.sum() < 5 or (~hi).sum() < 5: continue
        ct = np.array([[np.sum(hi & (yy==1)), np.sum(hi & (yy==0))],
                       [np.sum(~hi & (yy==1)), np.sum(~hi & (yy==0))]])
        if (ct.sum(0)==0).any() or (ct.sum(1)==0).any(): continue
        ts.append(t); ps.append(stats.chi2_contingency(ct)[1])
    q = bh(ps) if ps else np.array([])
    nsig = int((q < 0.05).sum()) if len(q) else 0
    ax.axhspan(0.0, 0.05, color="0.85", zorder=0)          # grey significance band (q<0.05)
    ax.plot(ts, q, "o-", color=color, ms=6, lw=1.5)
    ax.axhline(0.05, ls="--", color="0.5", lw=1.2)          # grey dashed line at 0.05
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(f"{bm} threshold t ({bm} > t)")
    ax.set_ylabel("FDR q-value (BH)")
    ax.set_title(f"{bm} association with Sepsis\nacross thresholds", fontsize=13)
    ax.text(-0.18, 1.12, letter, transform=ax.transAxes, fontsize=22, fontweight="bold", va="top", ha="left")
    print(f"  {bm}: {nsig}/{len(ts)} thresholds significant (q<0.05)")

plt.tight_layout()
out = os.path.join(OUTDIR, "Figure2_threshold_association.png")
fig.savefig(out, dpi=400, bbox_inches="tight"); print("Saved:", out)
