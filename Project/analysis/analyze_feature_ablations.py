"""Analyze feature-engineering ablation results and generate summary visuals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Project.utils.standardize import ensure_directories, load_metrics

ABLATION_REPORT = Path("reports/feature_ablation_summary.csv")
ABLATION_JSON = Path("reports/feature_ablation_summary.json")
ABLATION_DIR = Path("reports/ablations")
FIG_DIR = Path("figures/ablations")

METRICS_OF_INTEREST = ["f1_macro", "accuracy", "fit_time_sec", "predict_time_sec"]


def _prepare_dirs() -> None:
    ensure_directories()
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _collect_variant_data(metrics: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for framework, df in metrics.items():
        if df.empty:
            continue
        if "feature_variant_key" not in df.columns:
            continue
        numeric_cols = [col for col in df.columns if col in METRICS_OF_INTEREST]
        if not numeric_cols:
            continue
        grouped = df.groupby([
            "feature_variant_key",
            "feature_variant",
            "feature_description",
            "estimator_name",
        ], dropna=False)
        for (variant_key, variant_name, variant_desc, estimator_name), group in grouped:
            for metric in numeric_cols:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                if values.empty:
                    continue
                records.append({
                    "framework": framework,
                    "estimator": estimator_name if pd.notna(estimator_name) else framework,
                    "variant_key": variant_key,
                    "variant_name": variant_name,
                    "variant_description": variant_desc,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": int(len(values)),
                })
    return pd.DataFrame.from_records(records)


def _attach_baseline_delta(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    baseline = df[df.variant_key == "baseline"].set_index(["framework", "estimator", "metric"])["mean"]
    def compute_delta(row):
        key = (row["framework"], row["estimator"], row["metric"])
        if key in baseline.index:
            return float(row["mean"] - baseline.loc[key])
        return np.nan
    df["delta_vs_baseline"] = df.apply(compute_delta, axis=1)
    return df


def _save_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No ablation results found.")
        return
    df_sorted = df.sort_values(["metric", "framework", "variant_key"])
    df_sorted.to_csv(ABLATION_REPORT, index=False)
    df_sorted.to_json(ABLATION_JSON, orient="records", indent=2)
    df_sorted.to_csv(ABLATION_DIR / "feature_ablation_summary.csv", index=False)


def _plot_metric(df: pd.DataFrame, metric: str) -> None:
    subset = df[df.metric == metric]
    if subset.empty:
        return
    estimators = sorted(subset.estimator.unique())
    variants = [v for v in subset.variant_name.unique()]
    bar_width = 0.8 / max(1, len(variants))
    x = np.arange(len(estimators))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=320)
    for idx, variant in enumerate(variants):
        slice_df = subset[subset.variant_name == variant]
        means = [slice_df[slice_df.estimator == est]["mean"].mean() for est in estimators]
        offsets = x + idx * bar_width - (len(variants) - 1) * bar_width / 2
        ax.bar(offsets, means, width=bar_width, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(estimators, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Feature Ablation Impact — {metric}")
    ax.legend()
    plt.tight_layout()
    out_path = FIG_DIR / f"feature_ablation_{metric}.png"
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _plot_delta(df: pd.DataFrame, metric: str) -> None:
    subset = df[(df.metric == metric) & (df.variant_key != "baseline")]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6), dpi=320)
    sns = __import__("seaborn")
    sns.barplot(data=subset, x="variant_name", y="delta_vs_baseline", hue="estimator", ax=ax)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel(f"Δ {metric} vs baseline")
    ax.set_xlabel("Variant")
    ax.set_title(f"Performance Delta vs Baseline — {metric}")
    plt.tight_layout()
    out_path = FIG_DIR / f"feature_ablation_delta_{metric}.png"
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _prepare_dirs()
    metrics = load_metrics()
    df = _collect_variant_data(metrics)
    df = _attach_baseline_delta(df)
    _save_summary(df)

    for metric in METRICS_OF_INTEREST:
        _plot_metric(df, metric)
        _plot_delta(df, metric)

    if df.empty:
        return

    preview = df.head(10).to_dict(orient="records")
    print(json.dumps(preview, indent=2))


if __name__ == "__main__":
    main()
