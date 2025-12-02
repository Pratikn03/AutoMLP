#!/usr/bin/env python3
"""Plot score-versus-time curves for top frameworks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score vs time plot generator.")
    parser.add_argument(
        "--framework-summary",
        type=Path,
        default=Path("reports/framework_summary.csv"),
        help="CSV containing framework metrics (mean, metric).",
    )
    parser.add_argument(
        "--metric",
        default="accuracy",
        help="Metric column to plot on the Y axis (e.g., accuracy, f1_macro).",
    )
    parser.add_argument(
        "--time-metric",
        default="fit_time_sec",
        help="Metric name representing fit time.",
    )
    parser.add_argument(
        "--frameworks",
        nargs="*",
        default=[
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "FLAML",
            "H2O_AutoML",
        ],
        help="Ordered list of frameworks to include in the plot.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/score_vs_time_placeholder.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def load_metric(summary: pd.DataFrame, metric_name: str) -> pd.Series:
    subset = summary[summary["metric"] == metric_name]
    return subset.set_index("framework")["mean"]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.framework_summary)
    score = load_metric(df, args.metric)
    time = load_metric(df, args.time_metric)
    to_plot = (
        pd.DataFrame({"score": score, "fit_time": time})
        .loc[args.frameworks]
        .reset_index()
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=to_plot,
        x="fit_time",
        y="score",
        marker="o",
        sort=False,
        hue="framework",
        palette="colorblind",
    )
    for _, row in to_plot.iterrows():
        plt.text(
            row["fit_time"],
            row["score"] + 0.001,
            row["framework"],
            fontsize=8,
            ha="center",
        )

    plt.xlabel("Mean fit time (s)")
    plt.ylabel(args.metric.title())
    plt.title(f"{args.metric.title()} vs Fit Time")
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    plt.close()
    print(f"Wrote {args.out} for frameworks: {', '.join(args.frameworks)}")


if __name__ == "__main__":
    main()
