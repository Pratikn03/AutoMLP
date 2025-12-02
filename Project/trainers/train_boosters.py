import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.utils.multiclass import type_of_target
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from Project.utils.io import guess_target_column, load_dataset
from Project.utils.sanitize import sanitize_columns
from Project.utils.memory import reduce_memory_usage, clear_memory

SEED = int(os.getenv("SEED", "42"))
N_SPLITS = int(os.getenv("N_SPLITS", "5"))
USE_LOW_MEMORY = os.getenv("LOW_MEMORY_MODE", "1") == "1"
ART_DIR = "artifacts"
REP_DIR = "reports/metrics"
os.makedirs(ART_DIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

df = load_dataset(low_memory=USE_LOW_MEMORY)
if USE_LOW_MEMORY:
    df = reduce_memory_usage(df, verbose=True)
df = sanitize_columns(df)
target_col = guess_target_column(df, os.getenv("TARGET"))
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found after sanitization.")
y = df[target_col]
X = df.drop(columns=[target_col])

target_kind = type_of_target(y)
if target_kind in {"continuous", "continuous-multioutput"}:
    problem_type = "regression"
else:
    problem_type = "classification"

if problem_type == "classification":
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        y = y.astype(str).str.strip()
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(y.to_numpy())
    encoded_array = np.asarray(encoded, dtype=np.int64)
    y = pd.Series(encoded_array, name=target_col)
else:
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna()
    if not mask.all():
        X = X.loc[mask]
        y = y.loc[mask]
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Robustly detect categorical vs numeric columns
cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
num_cols = [c for c in X.columns if c not in cat_cols]
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False))])
try:
    _dense_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:  # pragma: no cover - older sklearn fallback
    _dense_ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", _dense_ohe)])
pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

needs_strat = problem_type == "classification" and y.nunique() > 1

def build_splits():
    if N_SPLITS >= 2:
        cv_obj = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED) if needs_strat else KFold(
            n_splits=N_SPLITS, shuffle=True, random_state=SEED
        )
        try:
            splits = list(cv_obj.split(X, y))
            if splits:
                return splits
        except ValueError as exc:
            print(f"[Booster] Falling back to holdout split: {exc}")
    stratify = y if needs_strat else None
    idx = np.arange(len(y))
    try:
        tr, va = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=stratify)
    except ValueError:
        tr, va = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=None)
    return [(tr, va)]

SPLITS = build_splits()

from Project.utils.standardize import save_metrics


def run_model(name, model, task):
    frows = []
    if not SPLITS:
        print(f"[Booster] No valid splits → skipping {name}")
        return
    for i, (tr, va) in enumerate(SPLITS, 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        pipe = Pipeline([("pre", pre), ("clf", model)])
        pipe.fit(Xtr, ytr)
        yp = pipe.predict(Xva)
        if task == "classification":
            acc = accuracy_score(yva, yp)
            f1 = f1_score(yva, yp, average="macro")
            try:
                proba = pipe.predict_proba(Xva)[:, 1]
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
                "avg_precision_ovr": ap,
            })
        else:
            rmse = mean_squared_error(yva, yp, squared=False)
            mae = mean_absolute_error(yva, yp)
            r2 = r2_score(yva, yp)
            frows.append({
                "fold": i,
                "framework": name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            })
    
    # Save metrics using standardization utility
    dfm = pd.DataFrame(frows)
    if not dfm.empty:
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

# XGBoost - memory-optimized settings
xgb_params = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "tree_method": "hist",  # Memory-efficient histogram-based
}

if USE_LOW_MEMORY:
    # Additional memory optimizations
    xgb_params.update({
        "max_bin": 128,  # Reduce bins to save memory (default is 256)
        "grow_policy": "depthwise",  # More memory-efficient than lossguide
    })

if problem_type == "classification":
    xgb = XGBClassifier(eval_metric="logloss", **xgb_params)
else:
    xgb = XGBRegressor(eval_metric="rmse", **xgb_params)
run_model("XGBoost", xgb, problem_type)

# LightGBM - memory-optimized settings
lgbm_params = {
    "n_estimators": 600,
    "max_depth": -1,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
}

if USE_LOW_MEMORY:
    # Memory optimizations for LightGBM
    lgbm_params.update({
        "max_bin": 128,  # Reduce histogram bins (default 255)
        "num_leaves": 31,  # Fewer leaves = less memory
        "min_data_in_leaf": 50,  # Prevent overly granular splits
    })

if problem_type == "classification":
    lgbm = LGBMClassifier(**lgbm_params)
else:
    lgbm = LGBMRegressor(**lgbm_params)

run_model("LightGBM", lgbm, problem_type)
clear_memory()  # Free memory after training
print(f"Booster baselines done for target '{target_col}' → reports/leaderboard.csv")
