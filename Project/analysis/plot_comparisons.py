from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Project.utils.standardize import (
    STANDARD_METRICS,
    ensure_directories,
    load_metrics,
)

SUMMARY_PATH = Path("reports/framework_summary.csv")
COMPARISONS_PATH = Path("reports/framework_comparisons.csv")
LEADERBOARD_PATH = Path("reports/leaderboard.csv")


def _style() -> None:
    ensure_directories()
    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams.update({"figure.dpi": 320, "savefig.dpi": 320})


def load_data() -> tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = load_metrics()
    summary = pd.read_csv(SUMMARY_PATH) if SUMMARY_PATH.exists() else pd.DataFrame()
    comparisons = (
        pd.read_csv(COMPARISONS_PATH)
        if COMPARISONS_PATH.exists()
        else pd.DataFrame()
    )
    leaderboard = (
        pd.read_csv(LEADERBOARD_PATH)
        if LEADERBOARD_PATH.exists()
        else pd.DataFrame()
    )
    return metrics, summary, comparisons, leaderboard


def plot_mean_ci(summary: pd.DataFrame) -> List[str]:
    paths: List[str] = []
    if summary.empty:
        print("Warning: framework_summary.csv missing – skipping mean/CI charts")
        return paths

    for metric in sorted(summary["metric"].unique()):
        data = summary[summary["metric"] == metric].copy()
        if data.empty:
            continue
        data = data.sort_values("mean", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        errs = np.vstack(
            [data["mean"] - data["ci95_low"], data["ci95_high"] - data["mean"]]
        )
        ax.bar(data["framework"], data["mean"], yerr=errs, capsize=6)
        ax.set_ylabel(metric)
        ax.set_title(f"Mean ± 95% CI — {metric}")
        ax.set_ylim(bottom=0)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = f"figures/metric_mean_ci_{metric}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _melt_metrics(metrics: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for framework, df in metrics.items():
        if framework == "leaderboard":
            continue
        for metric in STANDARD_METRICS:
            if metric in df.columns:
                for value in pd.to_numeric(df[metric], errors="coerce").dropna():
                    rows.append({"framework": framework, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def plot_violin(metrics: Dict[str, pd.DataFrame]) -> List[str]:
    paths: List[str] = []
    df = _melt_metrics(metrics)
    if df.empty:
        print("Warning: no detailed metrics for violin plots")
        return paths

    for metric in sorted(df["metric"].unique()):
        subset = df[df["metric"] == metric]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=subset, x="framework", y="value", ax=ax, cut=0, inner="box")
        ax.set_title(f"Distribution per Framework — {metric}")
        ax.set_ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = f"figures/metric_violin_{metric}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_box(metrics: Dict[str, pd.DataFrame]) -> str | None:
    df = _melt_metrics(metrics)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="framework", y="value", hue="metric", ax=ax)
    ax.set_title("Metric Distributions across Frameworks")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = "figures/performance_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    frameworks = sorted(
        set(df.get("framework_a", [])) | set(df.get("framework_b", []))
    )
    matrix = pd.DataFrame(0.0, index=frameworks, columns=frameworks)
    for _, row in df.iterrows():
        a, b, val = row["framework_a"], row["framework_b"], row[value_col]
        matrix.loc[a, b] = val
        matrix.loc[b, a] = -val if value_col == "diff" else val
    return matrix


def plot_heatmaps(comparisons: pd.DataFrame) -> List[str]:
    paths: List[str] = []
    if comparisons.empty:
        print("Warning: framework_comparisons.csv missing – skipping heatmaps")
        return paths

    for metric in sorted(comparisons["metric"].unique()):
        sub = comparisons[comparisons["metric"] == metric]
        if sub.empty:
            continue
        diff_matrix = _build_matrix(sub, "diff")
        p_matrix = _build_matrix(sub, "p_value").abs()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(diff_matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title(f"Mean Difference Heatmap — {metric}")
        plt.tight_layout()
        diff_path = f"figures/heatmap_diff_{metric}.png"
        fig.savefig(diff_path, bbox_inches="tight")
        plt.close(fig)
        paths.append(diff_path)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(p_matrix, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title(f"p-value Heatmap — {metric}")
        plt.tight_layout()
        p_path = f"figures/heatmap_pvalue_{metric}.png"
        fig.savefig(p_path, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_path)
    return paths


def plot_leaderboard(leaderboard: pd.DataFrame) -> List[str]:
    paths: List[str] = []
    if leaderboard.empty or "framework" not in leaderboard.columns:
        return paths

    leaderboard = leaderboard.copy()
    leaderboard = leaderboard.sort_values("f1_macro", ascending=False)

    if "f1_macro" in leaderboard.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(leaderboard["framework"], leaderboard["f1_macro"], color="teal")
        ax.invert_yaxis()
        ax.set_xlabel("f1_macro")
        ax.set_title("Leaderboard — F1 Macro")
        plt.tight_layout()
        path = "figures/leaderboard_f1_macro.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    if "accuracy" in leaderboard.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(leaderboard["framework"], leaderboard["accuracy"], color="steelblue")
        ax.invert_yaxis()
        ax.set_xlabel("accuracy")
        ax.set_title("Leaderboard — Accuracy")
        plt.tight_layout()
        path = "figures/leaderboard_accuracy.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    _style()
    metrics, summary, comparisons, leaderboard = load_data()

    produced: List[str] = []
    produced.extend(plot_mean_ci(summary))
    produced.extend(plot_violin(metrics))
    box_path = plot_box(metrics)
    if box_path:
        produced.append(box_path)
    produced.extend(plot_heatmaps(comparisons))
    produced.extend(plot_leaderboard(leaderboard))

    if produced:
        print("Generated visualizations:")
        for path in produced:
            print(f"- {path}")
    else:
        print("No visualizations generated (check inputs)")


if __name__ == "__main__":
    main()

