"""CLI helper to execute AutoML frameworks with shared CV splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from Project.experiments.automl import run_automl_suite
from Project.experiments.preprocessing import PreprocessingConfig
from Project.experiments.runner import ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoML frameworks (AutoGluon, LightAutoML, FLAML, H2O) under shared CV.")
    parser.add_argument("--experiment-name", default="automl_suite", help="Base name used for persisted metrics and artifacts.")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["autogluon", "lightautoml", "flaml", "h2o"],
        choices=["autogluon", "lightautoml", "flaml", "h2o"],
        help="Subset of AutoML frameworks to evaluate.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds for outer CV repeats.")
    parser.add_argument("--splits", type=int, default=5, help="Number of stratified folds.")
    parser.add_argument("--target", type=str, default=None, help="Optional explicit target column name.")
    parser.add_argument("--data-path", type=str, default=None, help="Optional CSV path to override default dataset loader.")
    parser.add_argument("--time-limit", type=int, default=600, help="Per-fold training time budget (seconds).")
    parser.add_argument("--output-dir", type=str, default="reports/metrics", help="Directory for fold and summary CSVs.")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/experiments", help="Directory for persisted models.")
    parser.add_argument("--figure-dir", type=str, default="figures/feature_importance", help="Directory for importance plots (if available).")
    parser.add_argument("--top-k", type=int, default=20, help="Top-N features to include in importance plots when supported.")
    parser.add_argument("--skip-model-save", action="store_true", help="Disable persisting fitted estimators.")
    parser.add_argument("--skip-importance", action="store_true", help="Disable feature-importance CSV/figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preprocessing = PreprocessingConfig(scale_numeric=False, poly_degree=None, binning_strategy=None)
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        seeds=args.seeds,
        n_splits=args.splits,
        preprocessing=preprocessing,
        output_dir=Path(args.output_dir),
        artifact_dir=Path(args.artifact_dir),
        figure_dir=Path(args.figure_dir),
        save_pipeline=not args.skip_model_save,
        save_feature_importance=not args.skip_importance,
        feature_importance_top_k=args.top_k,
    )

    df: Optional[pd.DataFrame] = None
    if args.data_path:
        df = pd.read_csv(Path(args.data_path))

    results = run_automl_suite(
        config,
        frameworks=args.frameworks,
        time_limit=args.time_limit,
        df=df,
        target_override=args.target,
    )

    for framework, payload in results.items():
        if "error" in payload:
            print(f"[{framework}] skipped: {payload['error']}")
            continue
        summary = payload.get("summary")
        if summary is not None and not summary.empty:
            metric_cols = [col for col in summary.columns if col not in {"seed", "experiment", "target", "framework"}]
            metrics = {col: round(float(summary.iloc[0][col]), 4) for col in metric_cols}
            print(f"[{framework}] summary metrics: {metrics}")
        else:
            print(f"[{framework}] completed; summary not available (empty frame).")


if __name__ == "__main__":
    main()
