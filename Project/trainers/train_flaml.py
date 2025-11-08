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
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

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

if y.dtype == "object":
    lower = set(str(v).strip().lower() for v in y.unique())
    if lower <= {"yes", "no", "1", "0", "true", "false"}:
        mapping = {"yes":1, "true":1, "1":1, "no":0, "false":0, "0":0}
        y = y.astype(str).str.strip().str.lower().map(mapping).astype(int)

# Ensure X is a DataFrame and y is a Series with proper dtypes
X = pd.DataFrame(X)
y = pd.Series(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

automl = AutoML()
settings = {
    "time_budget": TIME_BUDGET,
    "metric": "f1",
    "task": "classification",
    "log_file_name": "flaml.log",
    "seed": SEED,
}
automl.fit(X_train=Xtr, y_train=ytr, **settings)
yp = np.asarray(automl.predict(Xte))
f1 = f1_score(yte, yp, average="macro")
acc = accuracy_score(yte, yp)
try:
    proba_raw = automl.predict_proba(Xte)
    proba = np.asarray(proba_raw)[:, 1] if proba_raw is not None else None
    roc = roc_auc_score(yte, proba) if proba is not None else float("nan")
    ap = average_precision_score(yte, proba) if proba is not None else float("nan")
except Exception:
    roc, ap = float("nan"), float("nan")

metrics_row = pd.DataFrame([
    {
        "framework": "FLAML",
        "f1_macro": f1,
        "accuracy": acc,
        "roc_auc_ovr": roc,
        "avg_precision_ovr": ap,
    }
])
save_metrics(metrics_row, "FLAML")

art_dir = os.path.join("artifacts", "flaml")
os.makedirs(art_dir, exist_ok=True)
automl_path = os.path.join(art_dir, "best_automl.pkl")
best_config_path = os.path.join(art_dir, "best_config.json")
joblib.dump(automl, automl_path)
with open(best_config_path, "w", encoding="utf-8") as fh:
    json.dump(automl.best_config, fh, indent=2, default=str)

print(f"FLAML done for target '{target_col}' → reports/leaderboard.csv")
