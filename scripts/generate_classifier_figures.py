#!/usr/bin/env python3
"""Generate classifier metric figures for the paper."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create classifier metric plots used in the manuscript."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("src/data/datasets/tabular/heart_statlog.csv"),
        help="Path to the tabular dataset CSV.",
    )
    parser.add_argument(
        "--target",
        default="class",
        help="Name of the target column inside the dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where figure assets will be written.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("reports/metrics/classifier_metric_summary.csv"),
        help="CSV path for storing the aggregated classifier metrics.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by deterministic estimators.",
    )
    parser.add_argument(
        "--accuracy-filename",
        default="classifier_accuracy_placeholder.png",
        help="Filename for the accuracy figure (written inside --out-dir).",
    )
    parser.add_argument(
        "--precision-filename",
        default="classifier_precision_placeholder.png",
        help="Filename for the precision figure (written inside --out-dir).",
    )
    parser.add_argument(
        "--histogram-filename",
        default="classifier_histogram_placeholder.png",
        help="Filename for the multi-metric figure (written inside --out-dir).",
    )
    return parser.parse_args()


def load_dataset(path: Path, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {path}")
    y = df[target_column]
    df = df.drop(columns=[target_column])
    X = df.apply(pd.to_numeric, errors="coerce")
    if y.dtype == object or str(y.iloc[0]).isalpha():
        y_lower = y.astype(str).str.lower()
        if {"present", "absent"} <= set(y_lower.unique()):
            y = (y_lower == "present").astype(int)
        else:
            y = pd.Categorical(y_lower).codes
    else:
        y = y.astype(int)
    return X, y


def build_classifiers(random_state: int) -> Dict[str, object]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=None, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="liblinear"
        ),
        "SVM (RBF)": SVC(kernel="rbf", gamma="scale", probability=True, random_state=random_state),
        "GaussianNB": GaussianNB(),
    }


def evaluate_classifiers(
    classifiers: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int,
) -> pd.DataFrame:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    rows = []
    for name, estimator in classifiers.items():
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", estimator),
            ]
        )
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            error_score="raise",
            n_jobs=1,
        )
        rows.append(
            {
                "classifier": name,
                **{metric: scores[f"test_{metric}"].mean() for metric in scoring},
            }
        )
    return pd.DataFrame(rows)


def _write_barplot(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    ylabel: str,
) -> None:
    sns.set_theme(style="whitegrid")
    order = df.sort_values(metric, ascending=False)["classifier"]
    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=df,
        x="classifier",
        y=metric,
        order=order,
        hue="classifier",
        hue_order=order,
        dodge=False,
        legend=False,
        palette="colorblind",
    )
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0.7, 1.01)
    for idx, (_, row) in enumerate(df.set_index("classifier").loc[order].iterrows()):
        plt.text(
            idx,
            row[metric] + 0.005,
            f"{row[metric]:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def _write_histogram(df: pd.DataFrame, out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    long_df = df.melt(
        id_vars="classifier",
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    order = df.sort_values("accuracy", ascending=False)["classifier"]
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=long_df,
        x="classifier",
        y="score",
        hue="metric",
        order=order,
    )
    plt.xlabel("")
    plt.ylabel("Score")
    plt.ylim(0.7, 1.01)
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="", loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    X, y = load_dataset(args.data, args.target)
    classifiers = build_classifiers(args.random_state)
    metrics_df = evaluate_classifiers(classifiers, X, y, args.cv_splits)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.summary_path, index=False)

    accuracy_path = args.out_dir / args.accuracy_filename
    precision_path = args.out_dir / args.precision_filename
    histogram_path = args.out_dir / args.histogram_filename

    _write_barplot(metrics_df, "accuracy", accuracy_path, "Accuracy")
    _write_barplot(metrics_df, "precision", precision_path, "Precision")
    _write_histogram(metrics_df, histogram_path)


if __name__ == "__main__":
    main()
