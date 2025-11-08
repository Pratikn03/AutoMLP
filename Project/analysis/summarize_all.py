import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from numpy.random import default_rng

# Define paths
METRICS_DIR = Path("reports/metrics")
LEADERBOARD_PATH = Path("reports/leaderboard.csv")
SUMMARY_PATH = Path("reports/summary_ci.csv")
PAIRED_TESTS_PATH = Path("reports/paired_tests.csv")

from Project.utils.standardize import (
    load_metrics,
    standardize_metrics,
    ascii_table,
    STANDARD_METRICS,
)

# We will focus summaries on commonly available metrics only
METRIC_LIST = [
    "f1_macro",
    "accuracy",
    "roc_auc_ovr",
    "avg_precision_ovr",
    "fit_time_sec",
    "predict_time_sec",
]


def bootstrap_ci(values: np.ndarray, *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    if values.size == 0:
        return (np.nan, np.nan)
    rng = default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        resample = rng.choice(values, size=values.size, replace=True)
        boot_means[i] = float(np.mean(resample))
    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1 - alpha / 2))
    return lower, upper

def load_all_metrics():
    """Load all available metrics and summaries using standardization utility"""
    try:
        # Load metrics using standardization utility
        metrics = load_metrics()
        
        # Additional handling for H2O results if needed
        if LEADERBOARD_PATH.exists():
            try:
                h2o_results = pd.read_csv(LEADERBOARD_PATH)
                if "AutoML" in str(h2o_results.get("model", "")) or "AutoML" in str(h2o_results.get("framework", "")):
                    h2o_metrics = standardize_metrics(h2o_results, "H2O_AutoML")
                    metrics["H2O_AutoML"] = h2o_metrics
            except Exception as e:
                print(f"Warning: Error processing H2O results: {e}")
        
        return metrics
        
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return {}

def compute_framework_stats(metrics):
    """Compute summary statistics for each framework and metric present.

    Only include metrics from METRIC_LIST and skip if column missing or all NaN.
    """
    rows = []
    for framework, df in metrics.items():
        if framework == "leaderboard":
            continue
        # skip aggregated sources like 'leaderboard' for per-fold stats
        for metric in METRIC_LIST:
            if metric in df.columns:
                values = pd.to_numeric(df[metric], errors="coerce").dropna()
                if values.empty:
                    continue
                arr = values.to_numpy()
                mean = float(arr.mean())
                std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                ci_low = mean - 1.96 * std if std > 0 else mean
                ci_high = mean + 1.96 * std if std > 0 else mean
                boot_low, boot_high = bootstrap_ci(arr)
                rows.append(
                    {
                        "framework": framework,
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "boot_ci_low": boot_low,
                        "boot_ci_high": boot_high,
                        "n": int(arr.size),
                    }
                )
    return pd.DataFrame(rows)

def compare_frameworks(metrics):
    """Paired t-tests between frameworks on overlapping folds.

    We only compare frameworks that have per-fold results (same fold count).
    """
    comps = []
    frameworks = list(metrics.keys())
    comparison_metrics = ["f1_macro", "accuracy", "fit_time_sec", "predict_time_sec"]
    for i, a in enumerate(frameworks):
        for b in frameworks[i + 1 :]:
            for metric in comparison_metrics:
                if metric in metrics[a].columns and metric in metrics[b].columns:
                    try:
                        v1 = pd.to_numeric(metrics[a][metric], errors="coerce").dropna()
                        v2 = pd.to_numeric(metrics[b][metric], errors="coerce").dropna()
                        if len(v1) == len(v2) and len(v1) > 1:
                            t, p = stats.ttest_rel(v1, v2)
                            diff = np.asarray(v1) - np.asarray(v2)
                            boot_low, boot_high = bootstrap_ci(diff)
                            comps.append(
                                {
                                    "framework_a": a,
                                    "framework_b": b,
                                    "metric": metric,
                                    "diff": float(v1.mean() - v2.mean()),
                                    "t_stat": float(t),
                                    "p_value": float(p),
                                    "boot_ci_low": boot_low,
                                    "boot_ci_high": boot_high,
                                }
                            )
                    except Exception as e:  # pragma: no cover - defensive
                        print(f"Could not compare {a} vs {b} for {metric}: {e}")
    return pd.DataFrame(comps)

def print_summary(metrics, stats, comparisons):
    """Print comprehensive summary"""
    print("\n=== Framework Performance Summary ===")
    for framework in metrics.keys():
        frame_stats = stats[stats['framework'] == framework]
        if frame_stats.empty:
            continue
        print(f"\n{framework}:")
        rows = []
        for _, row in frame_stats.iterrows():
            try:
                rows.append(
                    {
                        "metric": row["metric"],
                        "mean": f"{float(row['mean']):.4f}",
                        "std": f"{float(row['std']):.4f}",
                        "ci95_low": f"{float(row['ci95_low']):.4f}",
                        "ci95_high": f"{float(row['ci95_high']):.4f}",
                        "boot_ci_low": f"{float(row['boot_ci_low']):.4f}",
                        "boot_ci_high": f"{float(row['boot_ci_high']):.4f}",
                        "n": int(row.get("n", 0)),
                    }
                )
            except Exception as e:
                rows.append({"metric": row.get("metric", "?"), "mean": f"ERR: {e}"})
        if rows:
            print(
                ascii_table(
                    rows,
                    headers=["metric", "mean", "std", "ci95_low", "ci95_high", "boot_ci_low", "boot_ci_high", "n"],
                )
            )
    
    print("\n=== Statistical Comparisons ===")
    if comparisons.empty:
        print("(no paired comparisons available)")
    else:
        rows = []
        for _, row in comparisons.iterrows():
            try:
                rows.append(
                    {
                        "A": row["framework_a"],
                        "B": row["framework_b"],
                        "metric": row["metric"],
                        "diff": f"{float(row['diff']):.4f}",
                        "p_value": f"{float(row['p_value']):.4g}",
                        "boot_ci_low": f"{float(row['boot_ci_low']):.4f}",
                        "boot_ci_high": f"{float(row['boot_ci_high']):.4f}",
                    }
                )
            except Exception as e:
                rows.append({"A": "?", "B": "?", "metric": "?", "diff": f"ERR {e}"})
        print(ascii_table(rows, headers=["A", "B", "metric", "diff", "p_value", "boot_ci_low", "boot_ci_high"]))

def main():
    # Create output directory
    os.makedirs("reports", exist_ok=True)
    
    try:
        # Load and analyze all metrics
        metrics = load_all_metrics()
        if not metrics:
            print("No metrics found!")
            return
        
        # Generate statistics and comparisons
        stats = compute_framework_stats(metrics)
        comparisons = compare_frameworks(metrics)
        
        # Save results (CSV + JSON)
        stats.to_csv("reports/framework_summary.csv", index=False)
        comparisons.to_csv("reports/framework_comparisons.csv", index=False)
        stats.to_json("reports/framework_summary.json", orient="records")
        comparisons.to_json("reports/framework_comparisons.json", orient="records")
        
        # Print summary
        print_summary(metrics, stats, comparisons)
        print("\nResults saved to:")
        print("- reports/framework_summary.csv (+ .json)")
        print("- reports/framework_comparisons.csv (+ .json)")
        
    except Exception as e:
        print(f"Error analyzing results: {e}")

if __name__ == "__main__":
    main()