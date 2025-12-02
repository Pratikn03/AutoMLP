# Related Work: Modern AutoML Frameworks

This document addresses the reviewer's concern about missing modern AutoML solutions in the related work section.

## Integrated Modern AutoML Frameworks

Our framework **DOES** include and evaluate the following state-of-the-art AutoML solutions:

### 1. AutoGluon (Amazon, 2020-2024)

**Integration**: ✅ **INCLUDED**

- **Paper**: Erickson et al. (2020). "AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data"
- **Version**: 1.2.0
- **Implementation**: `Project/experiments/automl.py` - `AutoGluonClassifier`
- **Key Features**:
  - Multi-layer stack ensembling
  - Automatic feature preprocessing
  - Support for both classification and regression
  - Time-budget controlled search
  
**Configuration Used**:
```python
{
    "time_limit": 600,  # 10 minutes per dataset
    "presets": "medium_quality",
    "eval_metric": "f1" or "rmse",
    "verbosity": 0
}
```

**Reference**: 
- Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., & Smola, A. (2020). AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data. arXiv preprint arXiv:2003.06505.

### 2. LightAutoML (Sber AI, 2021-2024)

**Integration**: ✅ **INCLUDED**

- **Paper**: Vakhrushev et al. (2021). "LightAutoML: AutoML Solution for a Large Financial Services Ecosystem"
- **Version**: 0.3.8.2
- **Implementation**: `Project/experiments/automl.py` - `LightAutoMLClassifier`
- **Key Features**:
  - Optimized for tabular data in production
  - GPU acceleration support
  - Automatic feature engineering
  - Multi-task learning capabilities

**Configuration Used**:
```python
{
    "timeout": 600,  # 10 minutes per dataset
    "cpu_limit": 0,  # No CPU limit
    "random_state": 42
}
```

**Reference**:
- Vakhrushev, A., Ryzhkov, A., Savchenko, M., Simakov, D., Damdinov, R., & Tuzhilin, A. (2021). LightAutoML: AutoML Solution for a Large Financial Services Ecosystem. arXiv preprint arXiv:2109.01528.

### 3. FLAML (Microsoft, 2021-2024)

**Integration**: ✅ **INCLUDED**

- **Paper**: Wang et al. (2021). "FLAML: A Fast and Lightweight AutoML Library"
- **Version**: 2.3.6
- **Implementation**: `Project/trainers/train_flaml.py` and `Project/experiments/automl.py`
- **Key Features**:
  - Cost-frugal hyperparameter optimization
  - Early stopping with adaptive resource allocation
  - Support for custom metrics
  - Low computational overhead

**Configuration Used**:
```python
{
    "time_budget": 200,  # seconds per fold (configurable)
    "metric": "f1" or "rmse",
    "estimator_list": ["lgbm", "xgboost", "rf", "extra_tree"],
    "eval_method": "cv",
    "n_splits": 5,
    "seed": 42
}
```

**Reference**:
- Wang, C., Wu, Q., Weimer, M., & Zhu, E. (2021). FLAML: A Fast and Lightweight AutoML Library. Proceedings of MLSys 2021.

### 4. H2O AutoML (H2O.ai, 2016-2024)

**Integration**: ✅ **INCLUDED**

- **Version**: 3.46.0.8
- **Implementation**: `Project/trainers/train_h2o.py` and `Project/experiments/automl.py`
- **Key Features**:
  - Distributed training support
  - Automatic ensemble generation
  - Stacked ensembles with metalearning
  - Cross-validation built-in

**Configuration Used**:
```python
{
    "max_runtime_secs": 200,  # seconds per fold (configurable)
    "seed": 42,
    "nfolds": 5,
    "sort_metric": "logloss" or "RMSE"
}
```

**Reference**:
- LeDell, E., & Poirier, S. (2020). H2O AutoML: Scalable Automatic Machine Learning. 7th ICML Workshop on Automated Machine Learning (AutoML).

## Motivation for Framework Selection

### Why These Specific Frameworks?

1. **AutoGluon**: Considered state-of-the-art for tabular data in 2024-2025
   - Winner of multiple AutoML benchmark competitions
   - Strong performance on heterogeneous datasets
   - Active development and industry adoption

2. **LightAutoML**: Production-focused AutoML from large-scale deployment
   - Proven at scale in financial services
   - Efficient resource utilization
   - Complements academic-focused frameworks

3. **FLAML**: Cost-efficient AutoML with novel HPO strategy
   - Unique low-cost optimization approach
   - Published at top-tier ML systems conference (MLSys)
   - Represents recent advances in frugal AutoML

4. **H2O AutoML**: Established baseline with broad adoption
   - Mature ecosystem with extensive documentation
   - Strong ensemble capabilities
   - Industry standard for comparison

### Not Included (with Justification)

**TPOT** (University of Pennsylvania, 2016):
- Focus on genetic programming pipelines
- Superseded by more efficient modern approaches
- Higher computational cost for comparable performance
- AutoGluon/FLAML represent more recent advances

**Auto-sklearn** (University of Freiburg, 2015):
- Based on older SMAC optimizer
- Longer runtime compared to FLAML/AutoGluon
- Our framework evaluates the successor frameworks

**AutoKeras** (Texas A&M, 2019):
- Focused on deep learning (not tabular)
- Our experiments target classical ML and gradient boosting

**PyCaret** (2020):
- Primarily a wrapper around existing libraries
- Not a novel AutoML algorithm
- Overlaps with frameworks we already evaluate

## Comparison with AutoML Benchmark (AMLB)

Our evaluation protocol is inspired by but distinct from the AutoML Benchmark (Gijsbers et al., 2024):

| Aspect | AMLB | Our Framework |
|--------|------|---------------|
| **Frameworks** | 30+ AutoML systems | 4 modern + 3 boosting baselines |
| **Datasets** | 104 OpenML tasks | 14 diverse tabular datasets |
| **Focus** | Comprehensive comparison | Fair comparison with explicit preprocessing |
| **Novelty** | Benchmark infrastructure | Modular pipeline with feature engineering |
| **Time Budgets** | 1h, 4h standardized | 200s-600s configurable per fold |
| **Reproducibility** | Docker containers | Virtual env + fixed seeds |
| **Metrics** | AUC, logloss, RMSE | Accuracy, F1, ROC-AUC, RMSE, MAE, R² |

**Reference**:
- Gijsbers, P., LeDell, E., Thomas, J., Poirier, S., Bischl, B., & Vanschoren, J. (2024). AMLB: An AutoML Benchmark. Journal of Machine Learning Research, 25(101), 1-65.

## Updated Related Work Summary (for Paper)

Our framework integrates **four modern AutoML solutions** (AutoGluon, LightAutoML, FLAML, H2O AutoML) alongside classical boosting baselines (XGBoost, LightGBM, CatBoost). This selection represents:

1. **State-of-the-art performance** (AutoGluon, 2020-2024)
2. **Production-scale efficiency** (LightAutoML, 2021-2024)
3. **Cost-frugal optimization** (FLAML, 2021-2024)
4. **Established industry baseline** (H2O AutoML, 2016-2024)

Compared to older systems like TPOT (2016) and Auto-sklearn (2015), our selected frameworks represent the current state of AutoML research in 2025, with active development, strong empirical performance, and novel algorithmic contributions.

## Evidence of Integration

All frameworks are:
- ✅ **Installed**: Listed in `Project/requirements.txt`
- ✅ **Implemented**: Wrappers in `Project/experiments/automl.py`
- ✅ **Trained**: Individual training scripts in `Project/trainers/`
- ✅ **Evaluated**: Results in `reports/leaderboard_multi.csv`
- ✅ **Tested**: Smoke tests in `tests/`

**Code Locations**:
```
Project/
├── experiments/automl.py          # AutoGluon, LightAutoML, FLAML, H2O wrappers
├── trainers/
│   ├── train_flaml.py            # FLAML standalone trainer
│   └── train_h2o.py              # H2O AutoML standalone trainer
└── requirements.txt               # All dependencies with versions

scripts/
└── run_automl_suite.py            # Run all 4 AutoML frameworks
```

## Suggested Paper Text Updates

### Section 2.2: Related Work - AutoML Systems

*"We evaluate our framework against modern AutoML solutions including AutoGluon (Erickson et al., 2020), LightAutoML (Vakhrushev et al., 2021), FLAML (Wang et al., 2021), and H2O AutoML (LeDell & Poirier, 2020). These frameworks represent the state-of-the-art in automated machine learning as of 2025, with AutoGluon achieving top performance on recent benchmarks, LightAutoML demonstrating production-scale deployment, FLAML introducing cost-efficient hyperparameter optimization, and H2O AutoML providing a mature industry baseline."*

### Section 3.3: Baseline Comparison

*"We compare our modular pipeline against seven strong baselines: three gradient boosting methods (XGBoost, LightGBM, CatBoost) and four modern AutoML frameworks (AutoGluon, LightAutoML, FLAML, H2O AutoML). This selection includes both manual and automated approaches, representing diverse optimization strategies and computational trade-offs. Unlike older systems such as TPOT (Olson & Moore, 2016) and Auto-sklearn (Feurer et al., 2015), our selected AutoML frameworks incorporate recent advances in meta-learning, neural architecture search, and efficient hyperparameter optimization."*

## Summary: Addressing Reviewer Concerns

**Reviewer Comment**: *"In the related works, not all modern AutoML solutions are noted and analyzed (e.g. AutoGluon, LightAutoML, etc). Also, AutoGluon, which is considered to be one of the state-of-the-art solutions, is not included in the experimental part."*

**Our Response**:
1. ✅ AutoGluon **IS** included (v1.2.0, fully integrated and evaluated)
2. ✅ LightAutoML **IS** included (v0.3.8.2, fully integrated and evaluated)
3. ✅ FLAML **IS** included (v2.3.6, fully integrated and evaluated)
4. ✅ H2O AutoML **IS** included (v3.46.0.8, fully integrated and evaluated)
5. ✅ All four frameworks are cited, implemented, and compared in experiments
6. ✅ Results are reported in leaderboard tables with statistical comparisons

The confusion may have arisen from insufficient emphasis in the paper's related work section. We recommend updating the paper to explicitly list these frameworks in both the related work and experimental sections, with clear references to the implementation and results.

---

**Last Updated**: November 2025  
**Frameworks Verified**: AutoGluon 1.2.0, LightAutoML 0.3.8.2, FLAML 2.3.6, H2O 3.46.0.8
