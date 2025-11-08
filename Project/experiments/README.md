# Experiment Engine Overview

The `Project/experiments` package bundles reusable building blocks for our
AutoML benchmarking pipeline. Milestone 1 introduces a leakage-free
cross-validation runner and flexible preprocessing configuration so every
subsequent experiment can reuse the same scaffolding.

## Modules

- `preprocessing.py`
  - `PreprocessingConfig`: dataclass capturing configuration flags for
    imputation, scaling, polynomial features, binning, and VIF filtering.
  - `VIFSelector`: optional transformer that drops highly collinear numeric
    features using variance inflation factors (requires `statsmodels`).
  - `build_preprocessor`: constructs a `Pipeline` wrapping a
    `ColumnTransformer` with the chosen options and returns a ready-to-fit
    transformer that can be inserted ahead of any estimator.

- `runner.py`
  - `ExperimentConfig`: stores experiment metadata (name, seeds, metrics,
    preprocessing config, output directory, artifact/figure locations, and can
    be serialized to JSON for provenance.
  - `ExperimentRunner`: orchestrates stratified K-fold runs for one estimator
    across multiple random seeds. It builds the preprocessing pipeline, trains
    the estimator, records per-fold metrics and timings, persists fitted
    pipelines (optional), captures feature-importance CSVs/plots, and writes
    fold-level/summary CSVs plus a JSON config snapshot under `reports/metrics/`.

- `boosting.py`
  - `TunableBoostingClassifier`: thin wrapper around XGBoost, LightGBM, and
    CatBoost that plugs into the runner as a scikit-learn estimator with
    optional random-search or Optuna tuning.
  - `run_boosting_suite`: helper that executes the boosting trio with shared
    preprocessing and ExperimentConfig, emitting consistent metrics, artifacts,
    and plots.

- `automl.py`
  - `AutoGluonClassifier`, `LightAutoMLClassifier`, `FLAMLClassifier`, and
    `H2OAutoMLClassifier` expose battle-tested AutoML frameworks through the
    runner interface while honouring shared time budgets.
  - `run_automl_suite`: coordinates AutoML experiments to keep data splits and
    reporting aligned with the rest of the stack.

- `ablations.py`
  - `FeatureVariant` + `DEFAULT_VARIANTS`: predefined feature-engineering
    configurations (baseline, polynomial, quantile binning, VIF filtering,
    no-scaling) for ablation studies.
  - `run_feature_ablation_suite`: sweeps estimators across variants/datasets to
    quantify feature preprocessing impact using the shared runner.

- `automl.py`
  - `AutoGluonClassifier`, `LightAutoMLClassifier`, `FLAMLClassifier`,
    `H2OAutoMLClassifier`: estimators that expose the same scikit-learn
    interface while delegating training to the respective AutoML frameworks.
    They advertise `skip_preprocessing` so the runner feeds raw folds and still
    captures runtime metrics.
  - `run_automl_suite`: orchestrates the AutoML frameworks under a shared time
    budget, persisting fold-level metrics, summaries, configs, and artifacts.

## Usage Sketch

```python
from Project.experiments.preprocessing import PreprocessingConfig
from Project.experiments.runner import ExperimentConfig, ExperimentRunner
from sklearn.linear_model import LogisticRegression

config = ExperimentConfig(
    experiment_name="logreg_baseline",
    seeds=[42, 77],
    n_splits=5,
    preprocessing=PreprocessingConfig(scale_numeric=True, vif_threshold=10.0),
)

runner = ExperimentRunner(config)
outputs = runner.run(lambda: LogisticRegression(max_iter=1000))
print(outputs["summary"])
```

This snippet trains a logistic-regression baseline using the new leakage-free
pipeline, writes fold-level metrics to `reports/metrics/logreg_baseline_*`, and
returns the in-memory DataFrames for any downstream analysis.

## Next Steps

With both the boosting and AutoML suites wired into the shared runner, upcoming
milestones can dive into statistical comparison tables, feature-engineering
ablations, and comprehensive runtime tracking/visualisation across frameworks.
