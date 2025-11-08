"""CLI helper to run the boosting experiment suite end-to-end."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from Project.experiments.boosting import run_boosting_suite
from Project.experiments.preprocessing import PreprocessingConfig
from Project.experiments.runner import ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XGBoost, LightGBM, and CatBoost experiments with shared CV.")
    parser.add_argument("--experiment-name", default="boosting_suite", help="Base name for saved artifacts and metrics.")
    parser.add_argument("--models", nargs="+", default=["xgboost", "lightgbm", "catboost"], choices=["xgboost", "lightgbm", "catboost"], help="Subset of boosting models to evaluate.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 77], help="Random seeds for outer CV repeats.")
    parser.add_argument("--splits", type=int, default=5, help="Number of stratified folds.")
    parser.add_argument("--target", type=str, default=None, help="Optional explicit target column name.")
    parser.add_argument("--data-path", type=str, default=None, help="Optional CSV path to override default dataset loader.")
    parser.add_argument("--output-dir", type=str, default="reports/metrics", help="Directory for fold and summary CSVs.")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/experiments", help="Directory for persisted pipelines.")
    parser.add_argument("--figure-dir", type=str, default="figures/feature_importance", help="Directory for feature-importance plots.")
    parser.add_argument("--tuning", choices=["none", "random", "optuna"], default="none", help="Enable optional inner-loop tuning.")
    parser.add_argument("--n-iter", type=int, default=25, help="Random search iterations (if enabled).")
    parser.add_argument("--n-trials", type=int, default=25, help="Optuna trials (if enabled).")
    parser.add_argument("--inner-cv", type=int, default=3, help="Inner CV splits for tuning.")
    parser.add_argument("--scoring", type=str, default="f1", help="Metric optimised during tuning.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for tuning evaluations.")
    parser.add_argument("--verbosity", type=int, default=0, help="Framework-specific verbosity level.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top features to plot per fold.")
    parser.add_argument("--skip-model-save", action="store_true", help="Disable persisting fitted pipelines.")
    parser.add_argument("--skip-importance", action="store_true", help="Disable saving feature importance outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preprocessing = PreprocessingConfig()
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

    tuning_strategy = None if args.tuning in (None, "none") else args.tuning
    results = run_boosting_suite(
        config,
        models=args.models,
        df=df,
        target_override=args.target,
        tuning_strategy=tuning_strategy,
        n_iter=args.n_iter,
        n_trials=args.n_trials,
        inner_cv=args.inner_cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        verbosity=args.verbosity,
    )

    # Provide lightweight CLI feedback summarising saved outputs.
    for model, payload in results.items():
        if "error" in payload:
            print(f"[{model}] skipped: {payload['error']}")
            continue
        summary = payload.get("summary")
        if summary is not None and not summary.empty:
            metric_cols = [col for col in summary.columns if col not in {"seed", "experiment", "target"}]
            metrics = {col: round(float(summary.iloc[0][col]), 4) for col in metric_cols}
            print(f"[{model}] summary metrics: {metrics}")
        else:
            print(f"[{model}] experiment completed; summary not available (empty frame).")


if __name__ == "__main__":
    main()
