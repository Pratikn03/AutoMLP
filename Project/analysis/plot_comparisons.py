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
OPS_PATH = Path("reports/leaderboard_ops.csv")


def _style() -> None:
    ensure_directories()
    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams.update({"figure.dpi": 320, "savefig.dpi": 320})


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[plot_comparisons] Skipping {path}: {exc}")
        return pd.DataFrame()


def load_data() -> tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = load_metrics()
    summary = _safe_read_csv(SUMMARY_PATH)
    comparisons = _safe_read_csv(COMPARISONS_PATH)
    leaderboard = _safe_read_csv(LEADERBOARD_PATH)
    ops = _safe_read_csv(OPS_PATH)
    return metrics, summary, comparisons, leaderboard, ops


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


def plot_classifier_histograms(leaderboard: pd.DataFrame) -> str | None:
    if leaderboard.empty:
        return None
    required = {"framework", "accuracy", "f1_macro"}
    if not required.issubset(leaderboard.columns):
        return None
    grouped = (
        leaderboard.groupby("framework")[["accuracy", "f1_macro"]]
        .mean(numeric_only=True)
        .dropna()
        .sort_values("accuracy", ascending=True)
    )
    if grouped.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].barh(grouped.index, grouped["accuracy"], color="slateblue")
    axes[0].set_title("Classifier Accuracy")
    axes[0].set_xlabel("Accuracy")
    axes[1].barh(grouped.index, grouped["f1_macro"], color="darkorange")
    axes[1].set_title("Classifier F1 Macro")
    axes[1].set_xlabel("F1 Macro")
    plt.tight_layout()
    path = "figures/classifier_accuracy_precision.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accuracy_runtime_pareto(ops_table: pd.DataFrame) -> str | None:
    """Scatter plot accuracy vs predict_time_p95 with Pareto frontier highlighted."""
    path = "figures/pareto_accuracy_runtime.png"
    if ops_table.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No runtime data available yet.\nRun the full pipeline to populate metrics.", ha="center", va="center")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    required = {"framework", "accuracy", "predict_time_p95"}
    if not required.issubset(ops_table.columns):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "leaderboard_ops.csv missing accuracy/runtime columns.", ha="center", va="center")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    df = ops_table[list(required)].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["predict_time_p95"] > 0]
    if df.empty:
        return None
    df = df.sort_values("predict_time_p95").reset_index(drop=True)
    pareto_flags: List[bool] = []
    best_acc = -np.inf
    for _, row in df.iterrows():
        acc = float(row["accuracy"])
        if acc >= best_acc - 1e-10:
            pareto_flags.append(True)
            best_acc = max(best_acc, acc)
        else:
            pareto_flags.append(False)
    df["on_pareto"] = pareto_flags

    fig, ax = plt.subplots(figsize=(8, 5))
    off = df[~df["on_pareto"]]
    on = df[df["on_pareto"]]
    ax.scatter(
        off["predict_time_p95"],
        off["accuracy"],
        color="tab:gray",
        alpha=0.6,
        label="Frameworks",
    )
    ax.scatter(
        on["predict_time_p95"],
        on["accuracy"],
        color="tab:orange",
        label="Pareto front",
    )
    for _, row in on.iterrows():
        ax.annotate(
            row["framework"],
            xy=(row["predict_time_p95"], row["accuracy"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Predict time p95 (sec)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Runtime Pareto Frontier")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    _style()
    metrics, summary, comparisons, leaderboard, ops = load_data()

    produced: List[str] = []
    produced.extend(plot_mean_ci(summary))
    produced.extend(plot_violin(metrics))
    box_path = plot_box(metrics)
    if box_path:
        produced.append(box_path)
    produced.extend(plot_heatmaps(comparisons))
    produced.extend(plot_leaderboard(leaderboard))
    pareto_path = plot_accuracy_runtime_pareto(ops)
    if pareto_path:
        produced.append(pareto_path)
    hist_path = plot_classifier_histograms(leaderboard)
    if hist_path:
        produced.append(hist_path)

    if produced:
        print("Generated visualizations:")
        for path in produced:
            print(f"- {path}")
    else:
        print("No visualizations generated (check inputs)")


if __name__ == "__main__":
    main()
