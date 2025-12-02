"""Boosting suite utilities integrating with the experiment runner."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold, ParameterSampler, StratifiedKFold, cross_val_score
from sklearn.utils.multiclass import type_of_target

from Project.experiments.runner import ExperimentConfig, ExperimentRunner

try:  # Optional dependency for advanced search
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None  # type: ignore

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None  # type: ignore

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None  # type: ignore

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None  # type: ignore


BoostingName = Literal["xgboost", "lightgbm", "catboost"]


DEFAULT_SEARCH_SPACE: Dict[BoostingName, Dict[str, Dict[str, Any]]] = {
    "xgboost": {
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 8},
        "min_child_weight": {"type": "int", "low": 1, "high": 10},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "gamma": {"type": "float", "low": 0.0, "high": 5.0},
        "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    },
    "lightgbm": {
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "num_leaves": {"type": "int", "low": 15, "high": 127},
        "max_depth": {"type": "int", "low": 3, "high": 16},
        "min_child_samples": {"type": "int", "low": 10, "high": 80},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 1e-3, "high": 1.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-3, "high": 1.0, "log": True},
    },
    "catboost": {
        "depth": {"type": "int", "low": 4, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "l2_leaf_reg": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        "bagging_temperature": {"type": "float", "low": 0.0, "high": 5.0},
        "border_count": {"type": "int", "low": 32, "high": 255},
    },
}


CLASSIFICATION_SCORING = {"accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc", "roc_auc_ovr"}


@dataclass
class TunableBoostingClassifier(ClassifierMixin, RegressorMixin, BaseEstimator):
    """Gradient boosting wrapper supporting optional tuning for classification or regression."""

    model_name: BoostingName
    base_params: Optional[Dict[str, Any]] = None
    tuning_strategy: Optional[Literal["random", "optuna"]] = None
    tuning_param_space: Optional[Dict[str, Dict[str, Any]]] = None
    n_iter: int = 20
    n_trials: int = 25
    inner_cv: int = 3
    scoring: str = "auto"
    n_jobs: int = -1
    random_state: Optional[int] = None
    verbosity: int = 0
    best_estimator_: Any = field(init=False, default=None, repr=False)
    best_params_: Dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    best_score_: float = field(init=False, default=float("nan"), repr=False)
    _task_type: Literal["classification", "regression"] = field(init=False, default="classification", repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series):  # type: ignore[override]
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

        rng_seed = int(self.random_state) if self.random_state is not None else 42
        self._rng = np.random.default_rng(rng_seed)

        target_kind = type_of_target(y_arr)
        self._task_type = "regression" if target_kind in {"continuous", "continuous-multioutput"} else "classification"

        search_space = self.tuning_param_space or DEFAULT_SEARCH_SPACE[self.model_name]
        best_params: Dict[str, Any] = {}
        best_score = -math.inf

        scoring_name = self._resolve_scoring()

        if self.tuning_strategy == "random":
            best_params, best_score = self._random_search(X_arr, y_arr, search_space, scoring_name)
        elif self.tuning_strategy == "optuna":
            if optuna is None:
                raise ImportError("Optuna is not installed; cannot run optuna tuning.")
            best_params, best_score = self._optuna_search(X_arr, y_arr, search_space, scoring_name)
        else:
            best_score = float("nan")

        estimator = self._create_estimator()
        if best_params:
            estimator.set_params(**best_params)
        estimator.fit(X_arr, y_arr)
        self.best_estimator_ = estimator
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.classes_ = getattr(estimator, "classes_", None)
        self.n_features_in_ = getattr(estimator, "n_features_in_", X_arr.shape[1])
        return self

    def predict(self, X):  # type: ignore[override]
        if self.best_estimator_ is None:
            raise RuntimeError("Estimator not fitted. Call fit before predict.")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):  # type: ignore[override]
        if self.best_estimator_ is None:
            raise RuntimeError("Estimator not fitted. Call fit before predict.")
        if self._task_type == "regression" or not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError("Predict probabilities unavailable for regression or this estimator")
        return self.best_estimator_.predict_proba(X)

    def decision_function(self, X):  # type: ignore[override]
        if self.best_estimator_ is None:
            raise RuntimeError("Estimator not fitted. Call fit before decision_function.")
        if not hasattr(self.best_estimator_, "decision_function"):
            raise AttributeError("Underlying estimator does not support decision_function")
        return self.best_estimator_.decision_function(X)

    @property
    def feature_importances_(self):  # type: ignore[override]
        if self.best_estimator_ is None:
            return None
        return getattr(self.best_estimator_, "feature_importances_", None)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model_name": self.model_name,
            "base_params": self.base_params,
            "tuning_strategy": self.tuning_strategy,
            "tuning_param_space": self.tuning_param_space,
            "n_iter": self.n_iter,
            "n_trials": self.n_trials,
            "inner_cv": self.inner_cv,
            "scoring": self.scoring,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
        }
        return params

    def set_params(self, **params):  # type: ignore[override]
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _random_search(self, X: np.ndarray, y: np.ndarray, space: Dict[str, Dict[str, Any]], scoring: str):
        distributions = self._param_distributions_for_random(space)
        sampler_seed = int(self._rng.integers(1, 10_000))
        sampler = ParameterSampler(distributions, n_iter=self.n_iter, random_state=sampler_seed)
        cv_seed = int(self._rng.integers(1, 10_000))
        cv = self._make_inner_cv(cv_seed)
        best_score = -math.inf
        best_params: Dict[str, Any] = {}

        for params in sampler:
            candidate = self._create_estimator()
            candidate.set_params(**params)
            scores = cross_val_score(candidate, X, y, scoring=scoring, cv=cv, n_jobs=self.n_jobs, error_score=np.nan)
            score = float(np.nanmean(scores))
            if score > best_score:
                best_score = score
                best_params = params
        return best_params, best_score

    def _optuna_search(self, X: np.ndarray, y: np.ndarray, space: Dict[str, Dict[str, Any]], scoring: str):
        assert optuna is not None  # for type checkers
        cv_seed = int(self._rng.integers(1, 10_000))
        cv = self._make_inner_cv(cv_seed)
        direction = "maximize" if self._is_higher_better(scoring) else "minimize"

        def objective(trial) -> float:
            params = self._sample_with_optuna(trial, space)
            candidate = self._create_estimator()
            candidate.set_params(**params)
            scores = cross_val_score(candidate, X, y, scoring=scoring, cv=cv, n_jobs=self.n_jobs, error_score=np.nan)
            return float(np.nanmean(scores))

        sampler_seed = int(self._rng.integers(1, 10_000))
        sampler = optuna.samplers.TPESampler(seed=sampler_seed)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        return study.best_params, float(study.best_value)

    def _create_estimator(self) -> Any:
        base_params = dict(self.base_params or {})
        seed = int(self.random_state) if self.random_state is not None else 42
        if self.model_name == "xgboost":
            if self._task_type == "regression":
                if XGBRegressor is None:
                    raise ImportError("xgboost is not installed; install xgboost to use this estimator.")
                params = {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 1,
                    "gamma": 0.0,
                    "reg_lambda": 1.0,
                    "eval_metric": "rmse",
                    "tree_method": "hist",
                    "random_state": seed,
                    "n_jobs": self.n_jobs,
                    "verbosity": self.verbosity,
                }
                params.update(base_params)
                return XGBRegressor(**params)
            if XGBClassifier is None:
                raise ImportError("xgboost is not installed; install xgboost to use this estimator.")
            params = {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0.0,
                "reg_lambda": 1.0,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": seed,
                "n_jobs": self.n_jobs,
                "verbosity": self.verbosity,
            }
            params.update(base_params)
            return XGBClassifier(**params)

        if self.model_name == "lightgbm":
            if self._task_type == "regression":
                if LGBMRegressor is None:
                    raise ImportError("lightgbm is not installed; install lightgbm to use this estimator.")
                params = {
                    "n_estimators": 500,
                    "objective": "regression",
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": seed,
                    "n_jobs": self.n_jobs,
                    "verbosity": self.verbosity,
                }
                params.update(base_params)
                return LGBMRegressor(**params)
            if LGBMClassifier is None:
                raise ImportError("lightgbm is not installed; install lightgbm to use this estimator.")
            params = {
                "n_estimators": 500,
                "objective": "binary",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": seed,
                "n_jobs": self.n_jobs,
                "verbosity": self.verbosity,
            }
            params.update(base_params)
            return LGBMClassifier(**params)

        if self.model_name == "catboost":
            if self._task_type == "regression":
                if CatBoostRegressor is None:
                    raise ImportError("catboost is not installed; install catboost to use this estimator.")
                params = {
                    "iterations": 1000,
                    "learning_rate": 0.05,
                    "depth": 6,
                    "l2_leaf_reg": 3.0,
                    "loss_function": "RMSE",
                    "random_seed": seed,
                    "allow_writing_files": False,
                    "verbose": False,
                }
                params.update(base_params)
                return CatBoostRegressor(**params)
            if CatBoostClassifier is None:
                raise ImportError("catboost is not installed; install catboost to use this estimator.")
            params = {
                "iterations": 1000,
                "learning_rate": 0.05,
                "depth": 6,
                "l2_leaf_reg": 3.0,
                "loss_function": "Logloss",
                "random_seed": seed,
                "allow_writing_files": False,
                "verbose": False,
            }
            params.update(base_params)
            return CatBoostClassifier(**params)

        raise ValueError(f"Unsupported model_name '{self.model_name}'")

    @staticmethod
    def _is_higher_better(scoring: str) -> bool:
        scoring_lower = scoring.lower()
        if scoring_lower.startswith("neg_"):
            return True
        return scoring_lower in {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f1_macro",
            "roc_auc",
            "roc_auc_ovr",
            "r2",
        }

    def _param_distributions_for_random(self, space: Dict[str, Dict[str, Any]]) -> Dict[str, Iterable[Any]]:
        distributions: Dict[str, Iterable[Any]] = {}
        for key, spec in space.items():
            kind = spec.get("type")
            if kind == "int":
                distributions[key] = list(range(int(spec["low"]), int(spec["high"]) + 1))
            elif kind == "float":
                low = float(spec["low"])
                high = float(spec["high"])
                num = int(spec.get("num", 20))
                if spec.get("log"):
                    start = max(low, 1e-6)
                    stop = max(high, start) if math.isclose(low, high) else max(high, 1e-6)
                    if math.isclose(start, stop):
                        values = np.repeat(start, num)
                    else:
                        values = np.geomspace(start, stop, num=num)
                else:
                    values = np.linspace(low, high, num=num)
                distributions[key] = values.tolist()
            elif kind == "categorical":
                distributions[key] = list(spec["choices"])
            else:
                raise ValueError(f"Unsupported search space type '{kind}' for parameter '{key}'")
        return distributions

    @staticmethod
    def _sample_with_optuna(trial, space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for key, spec in space.items():
            kind = spec.get("type")
            if kind == "int":
                params[key] = trial.suggest_int(key, int(spec["low"]), int(spec["high"]))
            elif kind == "float":
                params[key] = trial.suggest_float(key, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
            elif kind == "categorical":
                params[key] = trial.suggest_categorical(key, list(spec["choices"]))
            else:
                raise ValueError(f"Unsupported search space type '{kind}' for parameter '{key}'")
        return params

    def _make_inner_cv(self, seed: int):
        if self._task_type == "regression":
            return KFold(n_splits=self.inner_cv, shuffle=True, random_state=seed)
        return StratifiedKFold(n_splits=self.inner_cv, shuffle=True, random_state=seed)

    def _resolve_scoring(self) -> str:
        if not isinstance(self.scoring, str):
            return "neg_root_mean_squared_error" if self._task_type == "regression" else "f1"
        scoring = self.scoring.lower()
        if scoring in (None, "auto"):
            return "neg_root_mean_squared_error" if self._task_type == "regression" else "f1"
        if self._task_type == "regression" and scoring in CLASSIFICATION_SCORING:
            warnings.warn(
                f"Scoring '{self.scoring}' is classification-only; switching to neg_root_mean_squared_error for regression.",
                RuntimeWarning,
            )
            return "neg_root_mean_squared_error"
        return self.scoring


def make_boosting_factory(
    model_name: BoostingName,
    *,
    tuning_strategy: Optional[Literal["random", "optuna"]] = None,
    tuning_param_space: Optional[Dict[str, Dict[str, Any]]] = None,
    base_params: Optional[Dict[str, Any]] = None,
    n_iter: int = 20,
    n_trials: int = 25,
    inner_cv: int = 3,
    scoring: str = "auto",
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    verbosity: int = 0,
) -> Callable[[], TunableBoostingClassifier]:
    """Return a factory callable suitable for ExperimentRunner."""

    def _factory() -> TunableBoostingClassifier:
        return TunableBoostingClassifier(
            model_name=model_name,
            base_params=base_params,
            tuning_strategy=tuning_strategy,
            tuning_param_space=tuning_param_space,
            n_iter=n_iter,
            n_trials=n_trials,
            inner_cv=inner_cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=verbosity,
        )

    return _factory


def run_boosting_suite(
    base_config: ExperimentConfig,
    *,
    models: Iterable[BoostingName] = ("xgboost", "lightgbm", "catboost"),
    df: Optional[pd.DataFrame] = None,
    target_override: Optional[str] = None,
    tuning_strategy: Optional[Literal["random", "optuna"]] = None,
    tuning_param_spaces: Optional[Dict[BoostingName, Dict[str, Dict[str, Any]]]] = None,
    base_params: Optional[Dict[BoostingName, Dict[str, Any]]] = None,
    n_iter: int = 20,
    n_trials: int = 25,
    inner_cv: int = 3,
    scoring: str = "auto",
    n_jobs: int = -1,
    verbosity: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Run the boosting suite under shared configuration and return outputs."""

    results: Dict[str, Dict[str, Any]] = {}
    suffix = "baseline" if tuning_strategy in (None, "none") else tuning_strategy

    for model in models:
        experiment_name = f"{base_config.experiment_name}_{model}_{suffix}"
        config = replace(base_config, experiment_name=experiment_name)
        runner = ExperimentRunner(config)
        params_for_model = (base_params or {}).get(model)
        space_for_model = (tuning_param_spaces or {}).get(model)
        factory = make_boosting_factory(
            model,
            tuning_strategy=tuning_strategy if tuning_strategy not in (None, "none") else None,
            tuning_param_space=space_for_model,
            base_params=params_for_model,
            n_iter=n_iter,
            n_trials=n_trials,
            inner_cv=inner_cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=None,
            verbosity=verbosity,
        )
        try:
            results[model] = runner.run(factory, df=df, target_override=target_override)
        except ImportError as exc:
            warnings.warn(f"Skipping {model} - dependency missing: {exc}")
            results[model] = {"error": str(exc)}

    return results


__all__ = [
    "TunableBoostingClassifier",
    "make_boosting_factory",
    "run_boosting_suite",
    "DEFAULT_SEARCH_SPACE",
]
