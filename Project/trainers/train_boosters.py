import json
import os

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from Project.utils.io import guess_target_column, load_dataset
from Project.utils.sanitize import sanitize_columns

SEED = int(os.getenv("SEED", "42"))
N_SPLITS = int(os.getenv("N_SPLITS", "5"))
ART_DIR = "artifacts"
REP_DIR = "reports/metrics"
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

df = load_dataset()
df = sanitize_columns(df)
target_col = guess_target_column(df, os.getenv("TARGET"))
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found after sanitization.")
y = df[target_col]
X = df.drop(columns=[target_col])

# Map Yes/No to 0/1 if needed
if y.dtype == "object":
    if set(map(str, y.unique())) <= {"Yes", "No", "yes", "no"}:
        y = y.map({"No":0, "Yes":1, "no":0, "yes":1}).astype(int)

# Robustly detect categorical vs numeric columns
cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
num_cols = [c for c in X.columns if c not in cat_cols]
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

from Project.utils.standardize import save_metrics

def run_model(name, model):
    frows = []
    for i,(tr,va) in enumerate(cv.split(X,y),1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        pipe = Pipeline([("pre", pre), ("clf", model)])
        pipe.fit(Xtr, ytr)
        yp = pipe.predict(Xva)
        acc = accuracy_score(yva, yp)
        f1 = f1_score(yva, yp, average="macro")
        try:
            proba = pipe.predict_proba(Xva)[:,1]
            roc = roc_auc_score(yva, proba)
            ap = average_precision_score(yva, proba)
        except Exception:
            roc, ap = float("nan"), float("nan")
        frows.append({
            "fold": i,
            "framework": name,
            "accuracy": acc,
            "f1_macro": f1,
            "roc_auc_ovr": roc,
            "avg_precision_ovr": ap
        })
    
    # Save metrics using standardization utility
    dfm = pd.DataFrame(frows)
    save_metrics(dfm, name)
    # Leaderboard update is handled by save_metrics

    # Persist a final model trained on the full dataset for downstream use
    final_pipe = Pipeline([("pre", clone(pre)), ("clf", clone(model))])
    final_pipe.fit(X, y)
    out_dir = os.path.join(ART_DIR, name.lower())
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(final_pipe, os.path.join(out_dir, "model.pkl"))
    summary = dfm.mean(numeric_only=True).to_dict()
    summary.update({"folds": len(dfm), "target": target_col})
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

# XGBoost
xgb = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    random_state=SEED, tree_method="hist"
)
run_model("XGBoost", xgb)

# LightGBM
lgbm = LGBMClassifier(
    n_estimators=600, max_depth=-1, num_leaves=63,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=SEED
)
run_model("LightGBM", lgbm)
print(f"Booster baselines done for target '{target_col}' → reports/leaderboard.csv")
