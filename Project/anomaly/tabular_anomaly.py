"""Run a simple IsolationForest-based anomaly detector."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Project.utils.io import load_dataset
from Project.utils.sanitize import sanitize_columns

REPORTS_DIR = REPO_ROOT / "reports"
TARGET_DEFAULT = "IsInsurable"
TARGET = os.getenv("TARGET", TARGET_DEFAULT)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df = sanitize_columns(df)
    if df.empty:
        print("[Anomaly] Dataset empty; skipping detection.")
        return

    features = df.drop(columns=[col for col in [TARGET] if col in df.columns])
    cat_cols = [col for col in features.columns if features[col].dtype == "object"]
    num_cols = [col for col in features.columns if features[col].dtype != "object"]

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
        ],
        remainder="drop",
    )

    transformed = preprocessor.fit_transform(features)
    if hasattr(transformed, "toarray"):
        transformed_dense = transformed.toarray()  # type: ignore[attr-defined]
    else:
        transformed_dense = transformed
    detector = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    detector.fit(transformed_dense)

    scores = -detector.decision_function(transformed_dense)
    anomalies = pd.DataFrame({
        "row_index": np.arange(len(features)),
        "anomaly_score": scores,
    }).sort_values("anomaly_score", ascending=False)

    anomalies.head(50).to_csv(REPO_ROOT / "reports" / "top_anomalies.csv", index=False)
    print("[Anomaly] Saved top anomalies → reports/top_anomalies.csv")


if __name__ == "__main__":
    main()
