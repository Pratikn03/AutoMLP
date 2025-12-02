import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from Project.utils.io import guess_target_column, load_dataset
from Project.utils.sanitize import sanitize_columns
from Project.utils.standardize import save_metrics

# Allow overriding the time budget via CLI flag so automation can tune runtime easily.
parser = argparse.ArgumentParser(description="Train FLAML baseline.")
parser.add_argument("--time-budget", type=int, default=None, help="Seconds FLAML may spend searching (overrides FLAML_TIME_BUDGET).")
known_args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]

if known_args.time_budget is not None:
    os.environ["FLAML_TIME_BUDGET"] = str(max(1, known_args.time_budget))

SEED = int(os.getenv("SEED", "42"))
TIME_BUDGET = int(os.getenv("FLAML_TIME_BUDGET", "60"))
os.makedirs("reports/metrics", exist_ok=True)

df = load_dataset()
df = sanitize_columns(df)
target_col = guess_target_column(df, os.getenv("TARGET"))
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found after sanitization.")
y = df[target_col]
X = df.drop(columns=[target_col])
mask = y.notna()
if not mask.all():
    X = X.loc[mask]
    y = y.loc[mask]

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
    y = pd.Series(np.asarray(encoded, dtype=np.int64), name=target_col)
else:
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna()
    if not mask.all():
        X = X.loc[mask]
        y = y.loc[mask]
    y = y.astype(float)

# Ensure X is a DataFrame and y is a Series with proper dtypes
X = pd.DataFrame(X).reset_index(drop=True)
y = pd.Series(y).reset_index(drop=True)
n_rows, n_cols = X.shape
min_rows = int(os.getenv("FLAML_MIN_ROWS", "40"))
if n_cols == 0 or n_rows < min_rows:
    print(f"[FLAML] Skipping dataset (rows={n_rows}, cols={n_cols}) — insufficient data for FLAML.")
    sys.exit(0)
n_classes = int(y.nunique()) if problem_type == "classification" else None
is_binary = problem_type == "classification" and n_classes == 2

stratify = y if (problem_type == "classification" and y.nunique() > 1) else None
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=stratify, random_state=SEED)

automl = AutoML()
metric_name = "rmse"
if problem_type == "classification":
    metric_name = "f1" if is_binary else "macro_f1"
if problem_type == "classification":
    estimator_list = ["rf", "xgboost", "extra_tree", "xgb_limitdepth", "sgd", "lrl1", "catboost"]
else:
    estimator_list = ["rf", "xgboost", "extra_tree", "xgb_limitdepth", "sgd", "catboost"]
settings = {
    "time_budget": TIME_BUDGET,
    "metric": metric_name,
    "task": problem_type,
    "log_file_name": "flaml.log",
    "seed": SEED,
    "n_concurrent_trials": 1,
    "use_ray": False,
    "estimator_list": estimator_list,
}
automl.fit(X_train=Xtr, y_train=ytr, **settings)
yp = np.asarray(automl.predict(Xte))

if problem_type == "classification":
    f1 = f1_score(yte, yp, average="macro")
    acc = accuracy_score(yte, yp)
    if is_binary:
        try:
            proba_raw = automl.predict_proba(Xte)
            proba = np.asarray(proba_raw)[:, 1] if proba_raw is not None else None
            roc = roc_auc_score(yte, proba) if proba is not None else float("nan")
            ap = average_precision_score(yte, proba) if proba is not None else float("nan")
        except Exception:
            roc, ap = float("nan"), float("nan")
    else:
        roc, ap = float("nan"), float("nan")

    metrics_payload = {
        "framework": "FLAML",
        "f1_macro": f1,
        "accuracy": acc,
        "roc_auc_ovr": roc,
        "avg_precision_ovr": ap,
    }
else:
    rmse = mean_squared_error(yte, yp, squared=False)
    mae = mean_absolute_error(yte, yp)
    r2 = r2_score(yte, yp)
    metrics_payload = {
        "framework": "FLAML",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

metrics_row = pd.DataFrame([metrics_payload])
save_metrics(metrics_row, "FLAML")

art_dir = os.path.join("artifacts", "flaml")
os.makedirs(art_dir, exist_ok=True)
automl_path = os.path.join(art_dir, "best_automl.pkl")
best_config_path = os.path.join(art_dir, "best_config.json")
joblib.dump(automl, automl_path)
with open(best_config_path, "w", encoding="utf-8") as fh:
    json.dump(automl.best_config, fh, indent=2, default=str)

print(f"FLAML done for target '{target_col}' → reports/leaderboard.csv")
