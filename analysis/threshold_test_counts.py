# Section 3.4 / Figure 2 - biomarker threshold-association counts.
# 2x2 chi-square per cutoff (>=5 per group); Benjamini-Hochberg FDR; count q<0.05.
# Usage: python threshold_test_counts.py [data.csv]   (default = combined_1257.csv)
import sys, numpy as np, pandas as pd
from scipy import stats
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DATA = sys.argv[1] if len(sys.argv) > 1 else r"../data/combined_1257.csv"

def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n); pr = 1.0
    for i in range(n-1, -1, -1):
        v = p[o[i]]*n/(i+1); pr = min(pr, v); q[o[i]] = pr
    return np.clip(q, 0, 1)

df = pd.read_csv(DATA); y = df["SEPSIS"].astype(int)
print(f"{DATA}: n={len(df)}, sepsis={int(y.sum())} ({y.mean()*100:.1f}%)\n")
for bm, unit in [("IL-6","pg/mL"), ("PCT","ng/mL"), ("CRP","mg/L"), ("sEPCR","ng/mL")]:
    v = pd.to_numeric(df[bm], errors="coerce"); m = v.notna(); vv = v[m].values; yy = df["SEPSIS"][m].astype(int).values
    cand = np.unique(np.nanpercentile(vv, np.arange(5, 96, 3))); ts, ps = [], []
    for t in cand:
        hi = vv > t
        if hi.sum() < 5 or (~hi).sum() < 5: continue
        ct = np.array([[np.sum(hi&(yy==1)), np.sum(hi&(yy==0))], [np.sum(~hi&(yy==1)), np.sum(~hi&(yy==0))]])
        if (ct.sum(0)==0).any() or (ct.sum(1)==0).any(): continue
        ts.append(t); ps.append(stats.chi2_contingency(ct)[1])
    ts = np.array(ts); ps = np.array(ps); q = bh(ps); nsig = int((q < 0.05).sum())
    imin = int(np.argmin(q)); sigt = ts[q < 0.05]
    span = f"{sigt.min():.3g}-{sigt.max():.3g}" if len(sigt) else "none"
    print(f"{bm}: {nsig}/{len(ts)} significant ({100*nsig/len(ts):.0f}%)")
    print(f"   strongest cutoff={ts[imin]:.4g} {unit} (raw p={ps[imin]:.1e}, q={q[imin]:.1e}); min raw p={ps.min():.1e}; significant span {span}\n")
