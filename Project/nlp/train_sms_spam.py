"""Lightweight SMS spam NLP runner for the benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from Project.utils.system import capture_resource_snapshot, merge_runtime_sections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SMS spam text classification.")
    parser.add_argument("--data-root", default="src/data/datasets/text", help="Base dataset directory.")
    parser.add_argument("--dataset", default="sms_spam_collection", help="Folder name containing sms_spam.csv.")
    parser.add_argument("--splits", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling folds.")
    parser.add_argument("--max-features", type=int, default=10000, help="Maximum TF-IDF features.")
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("Dataset must contain 'label' and 'text' columns.")
    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
    df["label_binary"] = (df["label"].str.lower() == "spam").astype(int)
    return df


def build_pipeline(max_features: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(stop_words="english", max_features=max_features),
            ),
            (
                "clf",
                LogisticRegression(solver="liblinear", random_state=42, max_iter=300),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.data_root) / args.dataset / "sms_spam.csv"
    df = load_data(dataset_root)
    X = df["text"].values
    y = df["label_binary"].values

    pipeline = build_pipeline(args.max_features)
    cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
        "avg_precision": "average_precision",
    }

    resource_before = capture_resource_snapshot()
    start = time.time()
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
    )
    duration = time.time() - start
    resource_after = capture_resource_snapshot()

    metrics = {f"{key}_mean": float(np.mean(values)) for key, values in results.items() if key.startswith("test_")}
    fit_time = float(np.mean(results["fit_time"]))
    predict_time = float(np.mean(results["score_time"]))

    leaderboard = Path("reports") / "leaderboard_nlp.csv"
    row = {
        "dataset": args.dataset,
        "framework": "Logistic_TFIDF",
        "accuracy": metrics.get("test_accuracy_mean", np.nan),
        "f1_macro": metrics.get("test_f1_macro_mean", np.nan),
        "roc_auc_ovr": metrics.get("test_roc_auc_mean", np.nan),
        "avg_precision_ovr": metrics.get("test_avg_precision_mean", np.nan),
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
        "duration_sec": duration,
    }

    leaderboard.parent.mkdir(parents=True, exist_ok=True)
    if leaderboard.exists():
        existing = pd.read_csv(leaderboard)
        existing = existing[~(existing["dataset"] == args.dataset)]
    else:
        existing = pd.DataFrame()
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    combined.to_csv(leaderboard, index=False)

    runtime_entry: Dict[str, object] = {
        "dataset": args.dataset,
        "script": "Project/nlp/train_sms_spam.py",
        "duration_sec": duration,
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
        "splits": args.splits,
        "seed": args.seed,
        "resource_before": resource_before,
        "resource_after": resource_after,
        "timestamp": time.time(),
        "model_size_bytes": None,
    }
    merge_runtime_sections({"nlp": [runtime_entry]})

    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
