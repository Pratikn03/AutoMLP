"""Feature-engineering ablation utilities built atop the experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import pandas as pd

from Project.experiments.runner import ExperimentConfig, ExperimentRunner
from Project.experiments.preprocessing import PreprocessingConfig


@dataclass(frozen=True)
class FeatureVariant:
    """Describes a single feature-engineering configuration variant."""

    name: str
    display_name: str
    description: str
    preprocessing_overrides: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    is_baseline: bool = False


DEFAULT_VARIANTS: tuple[FeatureVariant, ...] = (
    FeatureVariant(
        name="baseline",
        display_name="Baseline",
        description="Median/most-frequent imputation, scaling enabled, no extra transforms.",
        preprocessing_overrides={},
        metadata={"preprocessing_variant": "baseline"},
        is_baseline=True,
    ),
    FeatureVariant(
        name="poly2",
        display_name="Polynomial (deg=2)",
        description="Augment numeric features with 2nd-degree polynomial terms.",
        preprocessing_overrides={"poly_degree": 2},
        metadata={"preprocessing_variant": "poly2"},
    ),
    FeatureVariant(
        name="quantile_bins",
        display_name="Quantile Binning",
        description="Discretize numeric features into quantile bins and one-hot encode.",
        preprocessing_overrides={"binning_strategy": "quantile", "n_bins": 10, "bin_encode": "onehot"},
        metadata={"preprocessing_variant": "quantile_bins"},
    ),
    FeatureVariant(
        name="vif_filter",
        display_name="VIF Filter",
        description="Apply variance inflation factor filtering (threshold=8).",
        preprocessing_overrides={"vif_threshold": 8.0},
        metadata={"preprocessing_variant": "vif_filter"},
    ),
    FeatureVariant(
        name="no_scaling",
        display_name="No Scaling",
        description="Disable numeric scaling to test robustness to raw feature magnitudes.",
        preprocessing_overrides={"scale_numeric": False},
        metadata={"preprocessing_variant": "no_scaling"},
    ),
)


def build_variant_config(base: ExperimentConfig, variant: FeatureVariant) -> ExperimentConfig:
    """Return a copy of the base ExperimentConfig tailored for a variant."""

    preprocessing = base.preprocessing
    overrides = variant.preprocessing_overrides
    updated_preprocessing = PreprocessingConfig(**{
        **preprocessing.as_dict(),
        **overrides,
    })
    metadata = dict(base.metadata)
    if variant.metadata:
        metadata.update(variant.metadata)
    metadata.setdefault("feature_variant", variant.display_name)
    metadata.setdefault("feature_description", variant.description)
    metadata.setdefault("feature_variant_key", variant.name)

    config = ExperimentConfig(
        experiment_name=base.experiment_name,
        seeds=tuple(base.seeds),
        n_splits=base.n_splits,
        metrics=base.metrics,
        preprocessing=updated_preprocessing,
        output_dir=base.output_dir,
        artifact_dir=base.artifact_dir,
        figure_dir=base.figure_dir,
        save_pipeline=base.save_pipeline,
        save_feature_importance=base.save_feature_importance,
        feature_importance_top_k=base.feature_importance_top_k,
        metadata=metadata,
    )
    return config


def run_feature_ablation_suite(
    base_config: ExperimentConfig,
    estimators: Mapping[str, Callable[[], Any]],
    variants: Iterable[FeatureVariant] = DEFAULT_VARIANTS,
    df: Optional[pd.DataFrame] = None,
    target_override: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run a grid of estimators × feature variants under shared CV splits."""

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    variant_list = list(variants)
    for estimator_name, factory in estimators.items():
        estimator_results: Dict[str, Dict[str, Any]] = {}
        for variant in variant_list:
            variant_config = build_variant_config(base_config, variant)
            experiment_name = f"{base_config.experiment_name}_{estimator_name}_{variant.name}"
            variant_config.experiment_name = experiment_name
            variant_metadata = dict(variant_config.metadata)
            variant_metadata.setdefault("estimator_name", estimator_name)
            variant_metadata.setdefault("experiment_family", base_config.experiment_name)
            variant_config.metadata = variant_metadata
            runner = ExperimentRunner(variant_config)
            try:
                payload = runner.run(factory, df=df, target_override=target_override)
            except ImportError as exc:
                payload = {"error": str(exc)}
            estimator_results[variant.name] = payload
        results[estimator_name] = estimator_results
    return results


__all__ = [
    "FeatureVariant",
    "DEFAULT_VARIANTS",
    "build_variant_config",
    "run_feature_ablation_suite",
]
