"""Run feature-engineering ablations across selected estimators."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

import pandas as pd

from Project.experiments.ablations import DEFAULT_VARIANTS, FeatureVariant, run_feature_ablation_suite
from Project.experiments.preprocessing import PreprocessingConfig
from Project.experiments.runner import ExperimentConfig
from Project.experiments.boosting import BoostingName, make_boosting_factory
from Project.experiments.automl import AutoMLName, make_automl_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature-engineering ablations across estimators.")
    parser.add_argument("--experiment-name", default="feature_ablation", help="Base name for experiments.")
    parser.add_argument("--estimators", nargs="+", default=["xgboost", "lightgbm", "catboost", "flaml"],
                        choices=["xgboost", "lightgbm", "catboost", "flaml", "autogluon", "lightautoml", "h2o"],
                        help="Estimators or AutoML frameworks to include.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds for CV repeats.")
    parser.add_argument("--splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--time-limit", type=int, default=600, help="Per-fold time budget for AutoML frameworks.")
    parser.add_argument("--target", type=str, default=None, help="Optional explicit target column.")
    parser.add_argument("--data-path", type=str, default=None, help="Optional CSV dataset path.")
    parser.add_argument("--output-dir", type=str, default="reports/metrics", help="Directory for metrics outputs.")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/experiments", help="Directory for model artifacts.")
    parser.add_argument("--figure-dir", type=str, default="figures/feature_importance", help="Directory for importance plots.")
    parser.add_argument("--skip-model-save", action="store_true", help="Disable persistence of fitted models.")
    parser.add_argument("--skip-importance", action="store_true", help="Disable feature-importance exports.")
    return parser.parse_args()


def build_estimators(selected: list[str], time_limit: int) -> Dict[str, Callable[[], Any]]:
    factories: Dict[str, Callable[[], Any]] = {}
    for name in selected:
        if name in {"xgboost", "lightgbm", "catboost"}:
            factories[name] = make_boosting_factory(cast(BoostingName, name), tuning_strategy=None)
        else:
            factories[name] = make_automl_factory(cast(AutoMLName, name), time_limit=time_limit)
    return factories


def main() -> None:
    args = parse_args()

    preprocessing = PreprocessingConfig()
    base_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        seeds=args.seeds,
        n_splits=args.splits,
        preprocessing=preprocessing,
        output_dir=Path(args.output_dir),
        artifact_dir=Path(args.artifact_dir),
        figure_dir=Path(args.figure_dir),
        save_pipeline=not args.skip_model_save,
        save_feature_importance=not args.skip_importance,
        metadata={"ablation_suite": True},
    )

    df: Optional[pd.DataFrame] = None
    if args.data_path:
        df = pd.read_csv(Path(args.data_path))

    estimators = build_estimators(args.estimators, args.time_limit)

    results = run_feature_ablation_suite(
        base_config,
        estimators=estimators,
        variants=DEFAULT_VARIANTS,
        df=df,
        target_override=args.target,
    )

    for estimator, variant_payload in results.items():
        for variant_name, payload in variant_payload.items():
            if "error" in payload:
                print(f"[{estimator} | {variant_name}] skipped: {payload['error']}")
                continue
            summary = payload.get("summary")
            if summary is not None and not summary.empty:
                metrics = {col: round(float(summary.iloc[0][col]), 4) for col in summary.columns if col.endswith("_mean")}
                print(f"[{estimator} | {variant_name}] summary: {metrics}")
            else:
                print(f"[{estimator} | {variant_name}] completed; summary unavailable.")


if __name__ == "__main__":
    main()
