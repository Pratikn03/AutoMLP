"""AutoML framework integrations compatible with the experiment runner."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Optional, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import type_of_target

from Project.experiments.runner import ExperimentConfig, ExperimentRunner

try:  # AutoGluon
    from autogluon.tabular import TabularPredictor  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    TabularPredictor = None  # type: ignore

try:  # LightAutoML
    from lightautoml.automl.presets.tabular_presets import TabularAutoML  # type: ignore[import]
    from lightautoml.tasks import Task  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    TabularAutoML = None  # type: ignore
    Task = None  # type: ignore

try:  # FLAML
    from flaml import AutoML as FLAMLAutoML  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    FLAMLAutoML = None  # type: ignore

try:  # H2O AutoML
    import h2o  # type: ignore[import]
    from h2o.automl import H2OAutoML  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    h2o = None  # type: ignore
    H2OAutoML = None  # type: ignore


AutoMLName = Literal["autogluon", "lightautoml", "flaml", "h2o"]


def _ensure_dataframe(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True).copy()
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("Expected 2D data for AutoML estimators.")
    cols = [f"feature_{i}" for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=cols)


def _ensure_series(y: Any, name: str = "target") -> pd.Series:
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True).copy()
    return pd.Series(np.asarray(y), name=name)


@dataclass
class AutoGluonClassifier(ClassifierMixin, RegressorMixin, BaseEstimator):
    """Thin wrapper around AutoGluon Tabular supporting classification or regression."""

    time_limit: int = 600
    presets: str = "medium_quality"
    eval_metric: Optional[str] = None
    problem_type: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    verbosity: int = 0
    label: str = field(default="__target__", init=False)
    skip_preprocessing: bool = True
    task: Literal["auto", "classification", "regression"] = "auto"
    _resolved_task: Literal["classification", "regression"] = field(init=False, default="classification", repr=False)

    def fit(self, X: Any, y: Any):  # type: ignore[override]
        if TabularPredictor is None:
            raise ImportError("AutoGluon is not installed; install autogluon.tabular to enable this estimator.")
        X_df = _ensure_dataframe(X)
        y_series = _ensure_series(y, name=self.label)
        if self.task == "auto":
            resolved_task: Literal["classification", "regression"] = (
                "regression" if type_of_target(y_series) in {"continuous", "continuous-multioutput"} else "classification"
            )
        else:
            resolved_task = cast(Literal["classification", "regression"], self.task)

        problem_type = self.problem_type
        eval_metric = self.eval_metric
        if resolved_task == "regression":
            problem_type = problem_type or "regression"
            eval_metric = eval_metric or "rmse"
        else:
            num_classes = y_series.nunique(dropna=True)
            if problem_type is None:
                problem_type = "multiclass" if num_classes > 2 else "binary"
            eval_metric = eval_metric or "f1"
        train_df = X_df.copy()
        train_df[self.label] = y_series
        self._tmpdir = tempfile.mkdtemp(prefix="autogluon_", dir=None)
        self._predictor = TabularPredictor(
            label=self.label,
            path=self._tmpdir,
            problem_type=problem_type,
            eval_metric=eval_metric,
        )
        self._predictor.fit(
            train_data=train_df,
            presets=self.presets,
            time_limit=self.time_limit,
            hyperparameters=self.hyperparameters,
            verbosity=self.verbosity,
        )
        self._resolved_task = resolved_task
        if resolved_task == "classification":
            self.classes_ = np.array(self._predictor.class_labels)
        else:
            self.classes_ = None
        self.n_features_in_ = X_df.shape[1]
        return self

    def predict(self, X: Any):  # type: ignore[override]
        if not hasattr(self, "_predictor"):
            raise RuntimeError("AutoGluonClassifier is not fitted yet.")
        preds = self._predictor.predict(_ensure_dataframe(X))
        return preds.to_numpy()

    def predict_proba(self, X: Any):  # type: ignore[override]
        if not hasattr(self, "_predictor"):
            raise RuntimeError("AutoGluonClassifier is not fitted yet.")
        if self._resolved_task == "regression":
            raise AttributeError("Predict probabilities unavailable for regression tasks")
        proba = self._predictor.predict_proba(_ensure_dataframe(X))
        return proba.to_numpy()

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "time_limit": self.time_limit,
            "presets": self.presets,
            "eval_metric": self.eval_metric,
            "problem_type": self.problem_type,
            "hyperparameters": self.hyperparameters,
            "verbosity": self.verbosity,
            "task": self.task,
        }

    def set_params(self, **params):  # type: ignore[override]
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        tmpdir = getattr(self, "_tmpdir", None)
        if tmpdir and Path(tmpdir).exists():
            shutil.rmtree(tmpdir, ignore_errors=True)


@dataclass
class LightAutoMLClassifier(ClassifierMixin, RegressorMixin, BaseEstimator):
    """Wrapper around LightAutoML TabularAutoML preset."""

    time_limit: int = 600
    cpu_limit: int = 0
    task_name: Optional[str] = None
    random_state: int = 42
    skip_preprocessing: bool = True
    task: Literal["auto", "classification", "regression"] = "auto"
    _resolved_task: Literal["classification", "regression"] = field(init=False, default="classification", repr=False)

    def fit(self, X: Any, y: Any):  # type: ignore[override]
        if TabularAutoML is None or Task is None:
            raise ImportError("LightAutoML is not installed; install lightautoml to enable this estimator.")
        X_df = _ensure_dataframe(X)
        y_series = _ensure_series(y, name="__target__")
        train_df = X_df.copy()
        train_df[y_series.name] = y_series
        if self.task == "auto":
            resolved_task: Literal["classification", "regression"] = (
                "regression" if type_of_target(y_series) in {"continuous", "continuous-multioutput"} else "classification"
            )
        else:
            resolved_task = cast(Literal["classification", "regression"], self.task)

        task_name = self.task_name
        if resolved_task == "regression":
            task_name = task_name or "reg"
        else:
            classes = y_series.nunique(dropna=True)
            if task_name is None:
                task_name = "multiclass" if classes > 2 else "binary"

        task = Task(task_name)
        automl = TabularAutoML(task=task, timeout=self.time_limit, cpu_limit=self.cpu_limit, random_state=self.random_state)
        automl.fit_predict(train_df, roles={"target": y_series.name})
        self._automl = automl
        self._task = task
        self._resolved_task = resolved_task
        if resolved_task == "classification":
            self.classes_ = np.unique(y_series)
        else:
            self.classes_ = None
        self.n_features_in_ = X_df.shape[1]
        return self

    def predict(self, X: Any):  # type: ignore[override]
        if not hasattr(self, "_automl"):
            raise RuntimeError("LightAutoMLClassifier is not fitted yet.")
        if self._resolved_task == "regression":
            preds = self._automl.predict(_ensure_dataframe(X))
            return np.asarray(preds.data).reshape(-1)
        proba = self.predict_proba(X)
        if proba.shape[1] == 1:
            return (proba[:, 0] >= 0.5).astype(int)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: Any):  # type: ignore[override]
        if not hasattr(self, "_automl"):
            raise RuntimeError("LightAutoMLClassifier is not fitted yet.")
        if self._resolved_task == "regression":
            raise AttributeError("Predict probabilities unavailable for regression tasks")
        preds = self._automl.predict(_ensure_dataframe(X))
        data = np.asarray(preds.data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        # LightAutoML returns probability for positive class in binary tasks
        if data.shape[1] == 1:
            data = np.column_stack([1 - data, data])
        return data

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "time_limit": self.time_limit,
            "cpu_limit": self.cpu_limit,
            "task_name": self.task_name,
            "random_state": self.random_state,
            "task": self.task,
        }

    def set_params(self, **params):  # type: ignore[override]
        for key, value in params.items():
            setattr(self, key, value)
        return self


@dataclass
class FLAMLClassifier(ClassifierMixin, RegressorMixin, BaseEstimator):
    """FLAML AutoML estimator with configurable time budget."""

    time_limit: int = 600
    metric: Optional[str] = None
    task: str = "auto"
    log_file_name: Optional[str] = None
    estimator_list: Optional[Iterable[str]] = None
    n_jobs: int = -1
    random_state: Optional[int] = 42
    skip_preprocessing: bool = True
    _resolved_task: Literal["classification", "regression"] = field(init=False, default="classification", repr=False)

    def fit(self, X: Any, y: Any):  # type: ignore[override]
        if FLAMLAutoML is None:
            raise ImportError("FLAML is not installed; install flaml to enable this estimator.")
        X_df = _ensure_dataframe(X)
        y_series = _ensure_series(y)
        if self.task == "auto":
            resolved_task: Literal["classification", "regression"] = (
                "regression" if type_of_target(y_series) in {"continuous", "continuous-multioutput"} else "classification"
            )
        else:
            resolved_task = "regression" if self.task.lower().startswith("reg") else "classification"

        metric = self.metric
        if metric is None or metric == "auto":
            metric = "rmse" if resolved_task == "regression" else "f1"
        automl = FLAMLAutoML()
        automl.fit(
            X_train=X_df,
            y_train=y_series,
            task=resolved_task,
            metric=metric,
            time_budget=self.time_limit,
            log_file_name=self.log_file_name,
            estimator_list=list(self.estimator_list) if self.estimator_list is not None else None,
            n_jobs=self.n_jobs,
            seed=self.random_state,
        )
        self._automl = automl
        self._resolved_task = resolved_task
        if resolved_task == "classification":
            self.classes_ = np.unique(y_series)
        else:
            self.classes_ = None
        self.n_features_in_ = X_df.shape[1]
        return self

    def predict(self, X: Any):  # type: ignore[override]
        if not hasattr(self, "_automl"):
            raise RuntimeError("FLAMLClassifier is not fitted yet.")
        return self._automl.predict(_ensure_dataframe(X))

    def predict_proba(self, X: Any):  # type: ignore[override]
        if not hasattr(self, "_automl"):
            raise RuntimeError("FLAMLClassifier is not fitted yet.")
        if self._resolved_task == "regression" or not hasattr(self._automl, "predict_proba"):
            raise AttributeError("FLAML estimator does not expose predict_proba")
        return self._automl.predict_proba(_ensure_dataframe(X))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "time_limit": self.time_limit,
            "metric": self.metric,
            "task": self.task,
            "log_file_name": self.log_file_name,
            "estimator_list": self.estimator_list,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }

    def set_params(self, **params):  # type: ignore[override]
        for key, value in params.items():
            setattr(self, key, value)
        return self


@dataclass
class H2OAutoMLClassifier(ClassifierMixin, RegressorMixin, BaseEstimator):
    """Wrapper for H2O AutoML with runtime controls."""

    time_limit: int = 600
    max_models: Optional[int] = None
    project_name: Optional[str] = None
    max_mem_size: Optional[str] = None
    random_seed: int = 42
    skip_preprocessing: bool = True
    label: str = field(default="__target__", init=False)
    task: Literal["auto", "classification", "regression"] = "auto"
    _resolved_task: Literal["classification", "regression"] = field(init=False, default="classification", repr=False)

    def fit(self, X: Any, y: Any):  # type: ignore[override]
        if H2OAutoML is None or h2o is None:
            raise ImportError("h2o is not installed; install h2o to enable this estimator.")
        h2o_local = cast(Any, h2o)
        if h2o_local.connection() is None:
            h2o_local.init(max_mem_size=self.max_mem_size)
        X_df = _ensure_dataframe(X)
        y_series = _ensure_series(y, name=self.label)
        if self.task == "auto":
            resolved_task: Literal["classification", "regression"] = (
                "regression" if type_of_target(y_series) in {"continuous", "continuous-multioutput"} else "classification"
            )
        else:
            resolved_task = cast(Literal["classification", "regression"], self.task)
        train_df = X_df.copy()
        train_df[self.label] = y_series
        train_frame = h2o_local.H2OFrame(train_df)
        if resolved_task == "classification":
            train_col = train_frame[self.label]
            train_frame[self.label] = cast(Any, train_col).asfactor()
        automl_cls = cast(Any, H2OAutoML)
        automl = automl_cls(
            max_runtime_secs=self.time_limit,
            max_models=self.max_models,
            project_name=self.project_name,
            seed=self.random_seed,
        )
        automl.train(y=self.label, training_frame=train_frame)
        self._automl = automl
        self._leader = cast(Any, automl.leader)
        self._resolved_task = resolved_task
        if resolved_task == "classification":
            self.classes_ = np.unique(y_series)
        else:
            self.classes_ = None
        self.n_features_in_ = X_df.shape[1]
        return self

    def predict(self, X: Any):  # type: ignore[override]
        if not hasattr(self, "_leader"):
            raise RuntimeError("H2OAutoMLClassifier is not fitted yet.")
        h2o_local = cast(Any, h2o)
        frame = h2o_local.H2OFrame(_ensure_dataframe(X))
        preds = cast(Any, self._leader).predict(frame).as_data_frame(use_pandas=True)
        return preds["predict"].to_numpy()

    def predict_proba(self, X: Any):  # type: ignore[override]
        if not hasattr(self, "_leader"):
            raise RuntimeError("H2OAutoMLClassifier is not fitted yet.")
        h2o_local = cast(Any, h2o)
        frame = h2o_local.H2OFrame(_ensure_dataframe(X))
        preds = cast(Any, self._leader).predict(frame).as_data_frame(use_pandas=True)
        if self._resolved_task == "regression":
            raise AttributeError("Predict probabilities unavailable for regression tasks")
        prob_cols = [col for col in preds.columns if col.startswith("p")]
        if not prob_cols:
            raise AttributeError("H2OAutoML predictions do not include probability columns")
        return preds[prob_cols].to_numpy()

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "time_limit": self.time_limit,
            "max_models": self.max_models,
            "project_name": self.project_name,
            "max_mem_size": self.max_mem_size,
            "random_seed": self.random_seed,
            "task": self.task,
        }

    def set_params(self, **params):  # type: ignore[override]
        for key, value in params.items():
            setattr(self, key, value)
        return self


def make_automl_factory(
    framework: AutoMLName,
    *,
    time_limit: int = 600,
    **kwargs: Any,
) -> Callable[[], BaseEstimator]:
    """Create a factory callable producing a configured AutoML estimator."""

    def _factory() -> BaseEstimator:
        if framework == "autogluon":
            return AutoGluonClassifier(time_limit=time_limit, **kwargs)
        if framework == "lightautoml":
            return LightAutoMLClassifier(time_limit=time_limit, **kwargs)
        if framework == "flaml":
            return FLAMLClassifier(time_limit=time_limit, **kwargs)
        if framework == "h2o":
            return H2OAutoMLClassifier(time_limit=time_limit, **kwargs)
        raise ValueError(f"Unsupported AutoML framework '{framework}'")

    return _factory


def run_automl_suite(
    base_config: ExperimentConfig,
    *,
    frameworks: Iterable[AutoMLName] = ("autogluon", "lightautoml", "flaml", "h2o"),
    time_limit: int = 600,
    df: Optional[pd.DataFrame] = None,
    target_override: Optional[str] = None,
    per_framework_kwargs: Optional[Dict[AutoMLName, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Execute selected AutoML frameworks under identical CV splits."""

    results: Dict[str, Dict[str, Any]] = {}
    for name in frameworks:
        experiment_name = f"{base_config.experiment_name}_{name}"
        config = replace(base_config, experiment_name=experiment_name)
        runner = ExperimentRunner(config)
        kwargs = (per_framework_kwargs or {}).get(name, {})
        factory = make_automl_factory(name, time_limit=time_limit, **kwargs)
        try:
            results[name] = runner.run(factory, df=df, target_override=target_override)
        except ImportError as exc:
            results[name] = {"error": str(exc)}
    return results


__all__ = [
    "AutoGluonClassifier",
    "LightAutoMLClassifier",
    "FLAMLClassifier",
    "H2OAutoMLClassifier",
    "make_automl_factory",
    "run_automl_suite",
]
