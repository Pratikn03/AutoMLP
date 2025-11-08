"""CatBoost trainer with cross-validation metrics and artifact persistence."""

import json
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from Project.utils.io import guess_target_column, load_dataset
from Project.utils.sanitize import sanitize_columns
from Project.utils.standardize import save_metrics

SEED = int(os.getenv("SEED", "42"))
N_SPLITS = int(os.getenv("CATBOOST_N_SPLITS", "5"))


def _resolve_cat_features(frame: pd.DataFrame) -> List[int]:
    return [i for i, col in enumerate(frame.columns) if frame[col].dtype == "object" or str(frame[col].dtype).startswith("category")]


def _prepare_target(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        lowered = series.astype(str).str.strip().str.lower()
        yes = {"yes", "true", "1", "y"}
        no = {"no", "false", "0", "n"}
        if lowered.isin(yes | no).all():
            mapping = {"yes":1, "true":1, "1":1, "y":1, "no":0, "false":0, "0":0, "n":0}
            return lowered.map(mapping).astype(int)
    return series


def main() -> None:
    try:
        from catboost import CatBoostClassifier, Pool
    except Exception as exc:  # pragma: no cover - optional dependency
        print("[CatBoost] Not available → skipping. Reason:", exc)
        return

    df = load_dataset()
    df = sanitize_columns(df)
    target_col = guess_target_column(df, os.getenv("TARGET"))
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found after sanitization.")

    y = _prepare_target(df[target_col])
    X = df.drop(columns=[target_col])

    cat_indices = _resolve_cat_features(X)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    rows = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        model = CatBoostClassifier(
            depth=6,
            learning_rate=0.05,
            iterations=400,
            loss_function="Logloss",
            eval_metric="F1",
            random_seed=SEED,
            auto_class_weights="Balanced",
            verbose=False,
        )
        train_pool = Pool(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_indices or None)
        valid_pool = Pool(X.iloc[valid_idx], y.iloc[valid_idx], cat_features=cat_indices or None)
        model.fit(train_pool, eval_set=valid_pool, verbose=False)

        pred_labels = np.asarray(model.predict(valid_pool)).ravel()
        y_valid = np.asarray(y.iloc[valid_idx])
        acc = float(accuracy_score(y_valid, pred_labels))
        f1 = float(f1_score(y_valid, pred_labels, average="macro"))
        try:
            pred_prob = np.asarray(model.predict_proba(valid_pool))[:, 1]
            roc = float(roc_auc_score(y_valid, pred_prob))
            ap = float(average_precision_score(y_valid, pred_prob))
        except Exception:
            roc, ap = float("nan"), float("nan")

        rows.append({
            "fold": fold,
            "framework": "CatBoost",
            "accuracy": acc,
            "f1_macro": f1,
            "roc_auc_ovr": roc,
            "avg_precision_ovr": ap,
        })

    metrics_df = pd.DataFrame(rows)
    save_metrics(metrics_df, "CatBoost")

    # Persist a final model fitted on the full dataset
    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.05,
        iterations=400,
        loss_function="Logloss",
        eval_metric="F1",
        random_seed=SEED,
        auto_class_weights="Balanced",
        verbose=False,
    )
    full_pool = Pool(X, y, cat_features=cat_indices or None)
    model.fit(full_pool, verbose=False)

    art_dir = os.path.join("artifacts", "catboost")
    os.makedirs(art_dir, exist_ok=True)
    model_path = os.path.join(art_dir, "model.cbm")
    model.save_model(model_path)
    summary = metrics_df.mean(numeric_only=True).to_dict()
    summary.update({"folds": len(metrics_df), "target": target_col})
    with open(os.path.join(art_dir, "metrics_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"CatBoost done for target '{target_col}' → reports/leaderboard.csv")


if __name__ == "__main__":
    main()
