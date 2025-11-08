"""Configurable preprocessing pipelines for leakage-free experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, PolynomialFeatures, StandardScaler

from Project.utils.sanitize import sanitize_columns

try:  # prefer statsmodels for VIF computation
    from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    variance_inflation_factor = None  # type: ignore


@dataclass
class PreprocessingConfig:
    """Runtime configuration flags controlling preprocessing behaviour."""

    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    scale_numeric: bool = True
    poly_degree: Optional[int] = None
    binning_strategy: Optional[Literal["quantile", "uniform", "kmeans"]] = None
    n_bins: int = 5
    bin_encode: Literal["onehot", "onehot-dense", "ordinal"] = "onehot-dense"
    vif_threshold: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "numeric_imputer_strategy": self.numeric_imputer_strategy,
            "categorical_imputer_strategy": self.categorical_imputer_strategy,
            "scale_numeric": self.scale_numeric,
            "poly_degree": self.poly_degree,
            "binning_strategy": self.binning_strategy,
            "n_bins": self.n_bins,
            "bin_encode": self.bin_encode,
            "vif_threshold": self.vif_threshold,
        }


class VIFSelector(BaseEstimator, TransformerMixin):
    """Drop numeric features whose variance inflation factor exceeds a threshold."""

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):  # type: ignore[override]
        frame = self._to_frame(X)
        self.feature_names_in_ = list(frame.columns)
        if self.threshold is None or variance_inflation_factor is None or frame.shape[1] <= 1:
            self.selected_features_ = self.feature_names_in_
            if self.threshold is not None and variance_inflation_factor is None:
                import warnings

                warnings.warn("statsmodels not available; skipping VIF filtering.")
            return self

        keep = self.feature_names_in_.copy()
        updated = True
        while updated and len(keep) > 1:
            updated = False
            design = frame[keep].to_numpy(dtype=float)
            vif_values = [variance_inflation_factor(design, i) for i in range(len(keep))]
            max_vif = float(np.max(vif_values))
            if max_vif > float(self.threshold):
                drop_idx = int(np.argmax(vif_values))
                keep.pop(drop_idx)
                updated = True
        self.selected_features_ = keep
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        frame = self._to_frame(X)
        if not hasattr(self, "selected_features_"):
            return frame
        return frame.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None) -> np.ndarray:
        if hasattr(self, "selected_features_"):
            return np.array(self.selected_features_)
        if input_features is None:
            return np.array([])
        return np.array(list(input_features))

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


def infer_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    categorical = [col for col in df.columns if col not in numeric]
    return {"numeric": numeric, "categorical": categorical}


def build_preprocessor(df: pd.DataFrame, config: Optional[PreprocessingConfig] = None) -> Pipeline:
    df = sanitize_columns(df)
    cfg = config or PreprocessingConfig()
    types = infer_feature_types(df)

    numeric_steps: List[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=cfg.numeric_imputer_strategy)),
    ]
    if cfg.poly_degree and cfg.poly_degree > 1:
        numeric_steps.append(("poly", PolynomialFeatures(degree=cfg.poly_degree, include_bias=False)))
    if cfg.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    if cfg.binning_strategy:
        numeric_steps.append((
            "binning",
            KBinsDiscretizer(
                n_bins=cfg.n_bins,
                encode=cfg.bin_encode,
                strategy=cfg.binning_strategy,
            ),
        ))

    categorical_steps: List[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=cfg.categorical_imputer_strategy, fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]

    transformers: List[tuple[str, Any, List[str]]] = []
    if types["numeric"]:
        transformers.append(("numeric", Pipeline(numeric_steps), types["numeric"]))
    if types["categorical"]:
        transformers.append(("categorical", Pipeline(categorical_steps), types["categorical"]))

    if not transformers:
        raise ValueError("No features available for preprocessing after sanitization.")

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="pandas")

    steps: List[tuple[str, Any]] = [("column_transform", transformer)]
    if cfg.vif_threshold is not None:
        steps.append(("vif", VIFSelector(threshold=cfg.vif_threshold)))
    return Pipeline(steps=steps)


__all__ = [
    "PreprocessingConfig",
    "VIFSelector",
    "build_preprocessor",
    "infer_feature_types",
]
