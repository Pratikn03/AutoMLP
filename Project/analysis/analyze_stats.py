import os, numpy as np, pandas as pd
from scipy import stats

LB = "reports/leaderboard.csv"
MET_DIR = "reports/metrics"
os.makedirs("reports", exist_ok=True)

def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x)==0: return (float("nan"), float("nan"))
    m = x.mean()
    se = x.std(ddof=1) / (len(x)**0.5) if len(x) > 1 else float("nan")
    lo, hi = m-1.96*se if se==se else float("nan"), m+1.96*se if se==se else float("nan")
    return (lo, hi)

rows = []
for name in ["XGBoost","LightGBM"]:
    path = f"{MET_DIR}/{name}_folds.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        lo, hi = ci95(df["f1_macro"])
        rows.append({"framework":name,"metric":"f1_macro","ci95_lo":lo,"ci95_hi":hi})
        lo, hi = ci95(df["accuracy"])
        rows.append({"framework":name,"metric":"accuracy","ci95_lo":lo,"ci95_hi":hi})

out = pd.DataFrame(rows)
out.to_csv("reports/summary_ci.csv", index=False)

# paired t-test between XGB and LGBM if both exist
pair_out = []
try:
    xgb = pd.read_csv(f"{MET_DIR}/XGBoost_folds.csv")["f1_macro"]
    lgb = pd.read_csv(f"{MET_DIR}/LightGBM_folds.csv")["f1_macro"]
    t,p = stats.ttest_rel(xgb, lgb)
    pair_out.append({"pair":"XGBoost_vs_LightGBM","metric":"f1_macro","t":float(t),"p":float(p)})
except Exception:
    pass

pd.DataFrame(pair_out).to_csv("reports/paired_tests.csv", index=False)
print("Stats saved → reports/summary_ci.csv, reports/paired_tests.csv")
