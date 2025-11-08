"""Train a simple Keras MLP baseline for tabular data."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Project.utils.io import load_dataset
from Project.utils.sanitize import sanitize_columns

SEED = int(os.getenv("SEED", "42"))
TARGET_DEFAULT = "IsInsurable"
TARGET = os.getenv("TARGET", TARGET_DEFAULT)
REPORTS_DIR = REPO_ROOT / "reports"
ART_DIR = REPO_ROOT / "artifacts" / "keras_dense"
LEADERBOARD_PATH = REPORTS_DIR / "leaderboard.csv"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df = sanitize_columns(df)
    if TARGET not in df.columns:
        print(f"[Keras] Skipping because target '{TARGET}' is missing.")
        return

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    if y.dtype == "object":
        lowered = y.astype(str).str.strip().str.lower()
        lookup = {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}
        if set(lowered.dropna().unique()).issubset(lookup.keys()):
            y = lowered.map(lookup).astype(int)

    cat_cols = [col for col in X.columns if X[col].dtype == "object"]
    num_cols = [col for col in X.columns if X[col].dtype != "object"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler(with_mean=False)),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y if y.nunique() > 1 else None,
        random_state=SEED,
    )

    try:
        import tensorflow as tf  # type: ignore
        from tensorflow import keras  # type: ignore

        tf.keras.utils.set_random_seed(SEED)
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[Keras] Skipping (TensorFlow unavailable): {exc}")
        return

    X_train_tf = preprocessor.fit_transform(X_train)
    X_test_tf = preprocessor.transform(X_test)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(X_train_tf.shape[1],)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train_tf,
        y_train,
        validation_data=(X_test_tf, y_test),
        epochs=10,
        batch_size=64,
        verbose=0,
    )

    proba = model.predict(X_test_tf, verbose=0).ravel()
    y_pred = (proba >= 0.5).astype(int)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, proba)
        avg_precision = average_precision_score(y_test, proba)
    except Exception:
        roc_auc = float("nan")
        avg_precision = float("nan")

    model.save(ART_DIR / "model.keras", overwrite=True)
    joblib.dump(preprocessor, ART_DIR / "preprocessor.pkl")
    metrics_payload = {
        "f1_macro": float(f1_macro),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "avg_precision": float(avg_precision),
    }
    (ART_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    leaderboard_cols = ["framework", "f1_macro", "accuracy", "roc_auc_ovr", "avg_precision_ovr"]
    if LEADERBOARD_PATH.exists():
        leaderboard = pd.read_csv(LEADERBOARD_PATH)
    else:
        leaderboard = pd.DataFrame(columns=leaderboard_cols)
    row = pd.DataFrame(
        [
            {
                "framework": "Keras_MLP",
                "f1_macro": f1_macro,
                "accuracy": accuracy,
                "roc_auc_ovr": roc_auc,
                "avg_precision_ovr": avg_precision,
            }
        ]
    )
    leaderboard = pd.concat([leaderboard, row], ignore_index=True)
    leaderboard = leaderboard.drop_duplicates(subset=["framework"], keep="last")
    leaderboard.to_csv(LEADERBOARD_PATH, index=False)

    print("[Keras] Completed training and leaderboard update.")


if __name__ == "__main__":
    main()
