"""Reusable experiment runner enforcing cross-validation and reproducibility."""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, cast

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from Project.experiments.preprocessing import PreprocessingConfig, build_preprocessor
from Project.utils.io import guess_target_column, load_dataset
from Project.utils.sanitize import sanitize_columns
from Project.utils.standardize import ensure_directories
from Project.utils.system import capture_resource_snapshot, merge_runtime_sections


MetricCallable = Callable[[np.ndarray, np.ndarray], float]


def _default_metrics() -> Dict[str, MetricCallable]:
    return {
        "accuracy": lambda y_true, y_pred: float(accuracy_score(y_true, y_pred)),
        "precision": lambda y_true, y_pred: float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": lambda y_true, y_pred: float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": lambda y_true, y_pred: float(f1_score(y_true, y_pred, zero_division=0)),
    }


@dataclass
class ExperimentConfig:
    experiment_name: str
    seeds: Iterable[int] = field(default_factory=lambda: (42, 77))
    n_splits: int = 5
    metrics: Dict[str, MetricCallable] = field(default_factory=_default_metrics)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    output_dir: Path = Path("reports/metrics")
    artifact_dir: Path = Path("artifacts/experiments")
    figure_dir: Path = Path("figures/feature_importance")
    save_pipeline: bool = True
    save_feature_importance: bool = True
    feature_importance_top_k: int = 20
    metadata: Dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> Dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "seeds": list(self.seeds),
            "n_splits": self.n_splits,
            "metrics": list(self.metrics.keys()),
            "preprocessing": self.preprocessing.as_dict(),
            "output_dir": str(self.output_dir),
            "artifact_dir": str(self.artifact_dir),
            "figure_dir": str(self.figure_dir),
            "save_pipeline": self.save_pipeline,
            "save_feature_importance": self.save_feature_importance,
            "feature_importance_top_k": self.feature_importance_top_k,
            "metadata": {str(k): v for k, v in self.metadata.items()},
        }


class ExperimentRunner:
    """Coordinates cross-validated experiments for a single dataset/estimator."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        ensure_directories()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.config.figure_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_data(self, df: Optional[pd.DataFrame] = None, target_override: Optional[str] = None):
        data = df.copy() if df is not None else load_dataset()
        data = sanitize_columns(data)
        target_col = guess_target_column(data, target_override or os.getenv("TARGET"))
        if target_col not in data.columns:
            raise KeyError(f"Target column '{target_col}' not found after sanitization.")
        y = data[target_col]
        X = data.drop(columns=[target_col])
        return X, y, target_col

    def run(self, estimator_factory: Callable[[], BaseEstimator], df: Optional[pd.DataFrame] = None, target_override: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Execute stratified CV for each seed and persist fold-level metrics."""
        X, y, target_col = self._prepare_data(df, target_override)
        cfg = self.config
        outputs: Dict[str, pd.DataFrame] = {}
        summary_rows = []
        metadata = cfg.metadata or {}
        runtime_entries: List[Dict[str, Any]] = []

        for seed in cfg.seeds:
            cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=int(seed))
            fold_rows: List[Dict[str, object]] = []

            for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                estimator = clone(estimator_factory())
                estimator = self._set_random_state(estimator, seed)
                skip_preprocessing = bool(getattr(estimator, "skip_preprocessing", False))
                if skip_preprocessing:
                    pipeline: BaseEstimator = estimator
                else:
                    preprocessor = build_preprocessor(X_train, config=cfg.preprocessing)
                    pipeline = Pipeline([
                        ("pre", preprocessor),
                        ("clf", estimator),
                    ])

                resource_before = capture_resource_snapshot()
                wall_start = time.time()
                start_time = time.perf_counter()
                cast(Any, pipeline).fit(X_train, y_train)
                fit_time = time.perf_counter() - start_time

                infer_start = time.perf_counter()
                raw_pred = cast(Any, pipeline).predict(X_valid)
                infer_time = time.perf_counter() - infer_start
                resource_after = capture_resource_snapshot()
                total_wall = time.time() - wall_start

                if isinstance(raw_pred, tuple):
                    raw_pred = raw_pred[0]
                y_pred = np.asarray(raw_pred)
                y_true_np = np.asarray(y_valid)

                metrics = {name: func(y_true_np, y_pred) for name, func in cfg.metrics.items()}

                model_path = self._persist_artifacts(pipeline, seed=int(seed), fold_idx=fold_idx)
                model_size = model_path.stat().st_size if model_path and model_path.exists() else None

                row_payload = {
                    "framework": cfg.experiment_name,
                    "seed": int(seed),
                    "fold": fold_idx,
                    "fit_time_sec": fit_time,
                    "predict_time_sec": infer_time,
                    **metrics,
                }
                row_payload.update({str(k): metadata[k] for k in metadata})
                fold_rows.append(row_payload)

                summary_payload = {
                    "framework": cfg.experiment_name,
                    "seed": int(seed),
                    "fold": fold_idx,
                    "fit_time_sec": fit_time,
                    "predict_time_sec": infer_time,
                    **metrics,
                }
                summary_payload.update({str(k): metadata[k] for k in metadata})
                summary_rows.append(summary_payload)

                delta_rss = None
                if resource_before and resource_after:
                    before_rss = resource_before.get("rss_mb")
                    after_rss = resource_after.get("rss_mb")
                    if before_rss is not None and after_rss is not None:
                        delta_rss = after_rss - before_rss

                runtime_payload = {
                    "experiment": cfg.experiment_name,
                    "target": target_col,
                    "seed": int(seed),
                    "fold": fold_idx,
                    "fit_time_sec": float(fit_time),
                    "predict_time_sec": float(infer_time),
                    "wall_time_sec": float(total_wall),
                    "resource_before": resource_before,
                    "resource_after": resource_after,
                    "rss_delta_mb": float(delta_rss) if delta_rss is not None else None,
                    "model_path": str(model_path) if model_path else None,
                    "model_size_bytes": int(model_size) if model_size is not None else None,
                    "timestamp": float(time.time()),
                }
                for key, value in metadata.items():
                    runtime_payload[f"meta_{key}"] = value if isinstance(value, (int, float, str, bool)) or value is None else str(value)
                runtime_entries.append(runtime_payload)

            fold_df = pd.DataFrame(fold_rows)
            file_name = f"{cfg.experiment_name}_seed{seed}_folds.csv"
            fold_path = cfg.output_dir / file_name
            fold_df.to_csv(fold_path, index=False)
            outputs[f"seed_{seed}"] = fold_df

        if not summary_rows:
            outputs["summary"] = pd.DataFrame()
            return outputs

        all_df = pd.DataFrame(summary_rows)
        exclude_cols = {"seed", "fold", "framework"} | set(str(k) for k in metadata)
        metric_columns = [col for col in all_df.columns if col not in exclude_cols]
        summary = all_df.groupby("seed")[metric_columns].agg(["mean", "std"])
        summary.columns = ["_".join(col).strip("_") for col in summary.columns]
        summary = summary.reset_index()
        summary["target"] = target_col
        summary["experiment"] = cfg.experiment_name
        summary["framework"] = cfg.experiment_name
        for key, value in metadata.items():
            summary[str(key)] = value
        summary_path = cfg.output_dir / f"{cfg.experiment_name}_summary.csv"
        summary.to_csv(summary_path, index=False)
        outputs["summary"] = summary

        config_path = cfg.output_dir / f"{cfg.experiment_name}_config.json"
        with open(config_path, "w", encoding="utf-8") as fh:
            payload = cfg.serialize()
            estimator_sample = estimator_factory()
            payload.update({"target": target_col, "estimator": estimator_sample.__class__.__name__})
            try:
                params = estimator_sample.get_params(deep=True) if hasattr(estimator_sample, "get_params") else {}
                payload["estimator_params"] = params
            except Exception:
                payload["estimator_params"] = {}
            json.dump(payload, fh, indent=2)

        if runtime_entries:
            merge_runtime_sections({"training": runtime_entries})

        return outputs

    @staticmethod
    def _set_random_state(estimator: BaseEstimator, seed: int) -> BaseEstimator:
        try:
            return estimator.set_params(random_state=int(seed))  # type: ignore[arg-type]
        except (ValueError, AttributeError):
            return estimator

    def _persist_artifacts(self, pipeline: BaseEstimator, seed: int, fold_idx: int) -> Optional[Path]:
        cfg = self.config
        model_dir = cfg.artifact_dir / cfg.experiment_name / f"seed_{seed}"
        importance_dir = cfg.output_dir / f"{cfg.experiment_name}_importances"
        figure_dir = cfg.figure_dir / cfg.experiment_name / f"seed_{seed}"

        saved_model: Optional[Path] = None

        if cfg.save_pipeline:
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"fold_{fold_idx}_pipeline.joblib"
            try:
                joblib.dump(pipeline, model_path)
                saved_model = model_path
            except Exception as exc:
                warnings.warn(f"Failed to persist pipeline for seed {seed} fold {fold_idx}: {exc}")

        if not cfg.save_feature_importance:
            return saved_model

        importance_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

        estimator = None
        preprocessor = None
        pipe_any = cast(Any, pipeline)
        if hasattr(pipe_any, "named_steps"):
            estimator = pipe_any.named_steps.get("clf")
            preprocessor = pipe_any.named_steps.get("pre")
        else:
            estimator = pipe_any

        if estimator is None:
            return saved_model

        feature_names = self._safe_feature_names(preprocessor)
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            return saved_model

        try:
            importance_array = np.asarray(importances).ravel()
        except Exception:
            return saved_model

        if feature_names.size != importance_array.size:
            feature_names = np.array([f"feature_{idx}" for idx in range(importance_array.size)])

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_array,
        }).sort_values("importance", ascending=False)

        csv_path = importance_dir / f"seed_{seed}_fold_{fold_idx}.csv"
        importance_df.to_csv(csv_path, index=False)

        top_k = max(1, int(cfg.feature_importance_top_k))
        plot_df = importance_df.head(top_k)

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return saved_model

        plt.figure(figsize=(8, max(4, int(top_k * 0.35))))
        plt.barh(plot_df["feature"], plot_df["importance"], color="#2874A6")
        plt.gca().invert_yaxis()
        plt.title(f"{cfg.experiment_name} | Seed {seed} Fold {fold_idx}")
        plt.xlabel("Importance")
        plt.tight_layout()
        figure_path = figure_dir / f"fold_{fold_idx}_importance.png"
        plt.savefig(figure_path, dpi=150)
        plt.close()

        return saved_model

    @staticmethod
    def _safe_feature_names(preprocessor: Optional[Any]) -> np.ndarray:
        if preprocessor is None:
            return np.array([])
        try:
            names = preprocessor.get_feature_names_out()
            if isinstance(names, np.ndarray):
                return names
            return np.asarray(names)
        except Exception:
            pass
        if hasattr(preprocessor, "named_steps") and "column_transform" in preprocessor.named_steps:
            column_transform = preprocessor.named_steps["column_transform"]
            try:
                names = column_transform.get_feature_names_out()
                if isinstance(names, np.ndarray):
                    return names
                return np.asarray(names)
            except Exception:
                return np.array([])
        return np.array([])


__all__ = ["ExperimentConfig", "ExperimentRunner"]
