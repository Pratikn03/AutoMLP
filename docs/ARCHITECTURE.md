# AutoML Pro – Architecture and Code Walkthrough

This note documents how the repository fits together so you can explain it line‑by‑line in a notebook or presentation. It is organised by lifecycle stage: data, modelling, analysis, deployment, automation, and packaging.

---

## 1. Data procurement & normalisation

| Component | Key Files | What it does |
|-----------|-----------|--------------|
| Dataset registry | `configs/registry.yaml`, `reports/dataset_registry.*` | Canonical metadata for every dataset: modality, target column, split strategy, governance flags. |
| Discovery utilities | `Project/utils/io.py` | `find_csv()` locates the working dataset, `load_dataset()` reads it safely, and `guess_target_column()` figures out which column to predict (with fallbacks such as remapping `SLA_Breached` → `IsInsurable`). |
| Sanitisation | `Project/utils/sanitize.py` | Cleans column names, coerces booleans/strings, ensures the dataframe is usable downstream. |
| Split governance | `Project/utils/splits.py` | Converts registry settings into actual `KFold`, `StratifiedKFold`, `GroupKFold`, or `TimeSeriesSplit` iterators so every trainer stays consistent. |
| Staging scripts | `scripts/stage_datasets.py`, `scripts/download_datasets.py` | Optional helpers to fetch and cache public datasets (Kaggle/OpenML) before running the pipeline. |

**Notebook call‑outs:** import the helpers directly (`from Project.utils.io import load_dataset`) and show a few `head()` / `.info()` calls to document “data cleaning”.

---

## 2. Modelling / AutoML engines

| Module | Responsibilities |
|--------|------------------|
| `Project/trainers/train_boosters.py` | Baseline gradient boosters (XGBoost, LightGBM). Handles preprocessing with `ColumnTransformer`, `Pipeline`, records per-fold metrics + calibration bins, and persists fitted models in `artifacts/`. |
| `Project/trainers/train_catboost.py` | CatBoost CV runner with similar reporting contract. |
| `Project/trainers/train_flaml.py` | FLAML baseline with metric guards, LightGBM fallback, calibration saving, and final model persistence. |
| `Project/trainers/train_h2o.py` | H2O AutoML wrapper: sanitises data, runs AutoML within runtime budget, computes metrics, saves both the binary model and MOJO plus `model_metadata.json` for serving. |
| `Project/experiments/runner.py` | Shared experiment harness (seeded CV loops, timing logging, artifact persistence). Used by experiment definitions inside `Project/experiments/*.py`. |
| `Project/experiments/automl.py` | Integrations with AutoGluon, LightAutoML, etc. (supports classification+regression). |
| `Project/deeplearning/*` | Torch/Keras baselines for tabular, vision, audio modalities. |

**Entry points:** `scripts/run_boosting_suite.py` orchestrates booster runs, `scripts/run_automl_suite.py` triggers the AutoML experiments, and the “everything” runner is `scripts/run_all.py` (metadata below).

---

## 3. Orchestrator (`scripts/run_all.py`)

High‑level steps executed per dataset:

1. Discover dataset (via registry or `DATASET_PATHS`), capture reproducibility tags (git SHA, registry hash).
2. Wipe `reports/`, `figures/`, `artifacts/` to avoid cross-run leakage.
3. For each dataset: run trainers (`Project/trainers/*`), AutoML suites, deep learning baselines, anomaly detectors, explainability, and reporting scripts.
4. After each dataset: snapshot metrics/figures/artifacts to `runs/<dataset>/`.
5. After all datasets: run “global steps” (vision/audio pipelines, corruption tests), merge leaderboards, generate registries, and write `reports/runtime.json`.

The orchestrator captures per-step exit codes, durations, and seeds so you can trace failures via `reports/runtime.json` and the dataset archives.

---

## 4. Analysis & dashboards

| Component | Description |
|-----------|-------------|
| `Project/analysis/summarize_all.py` | Loads every `reports/metrics/*_folds.csv`, computes per-framework stats (mean/std/bootstrapped CI), and paired differences. Exports `reports/framework_summary.*`, `reports/framework_comparisons.*`, `reports/paired_tests.csv`, `reports/summary_ci.csv`. |
| `Project/analysis/plot_comparisons.py` | Generates visual diagnostics: mean ± CI bar charts, violin distributions, Box plots, heatmaps, leaderboards, Pareto plot (accuracy vs predict time), **and the new classifier accuracy vs F1 dual histogram** (`figures/classifier_accuracy_precision.png`). |
| `Project/analysis/explain_shap.py`, `Project/analysis/analyze_feature_ablations.py`, `Project/analysis/plot_comparisons.py` | Additional explainability + feature importance outputs (SHAP, LIME, ablation summary). |
| README snapshot automation | `scripts/generate_readme_assets.py` compiles leaderboard tables + dataset winners and injects them into README using HTML comment markers. It also writes `reports/dashboard_summary.json` for programmatic dashboards. |

**Notebook hook:** import these modules and call `summarize_all.main()` + `plot_comparisons.main()` to regenerate figures inside your notebook.

---

## 5. Deployment & Ops

| Piece | Files | Notes |
|-------|-------|------|
| FastAPI service | `Deploy/api/serve/app.py` | Exposes `/predict`, `/health`, `/readyz`, `/metrics`, `/probe/latency`. Automatically loads the latest H2O model using the `model_metadata.json` pointer and logs Prometheus metrics. |
| API dependencies | `Deploy/api/requirements-serve.txt` | Trimmed requirements for serving only. |
| Docker/K8s | `Deploy/api/Dockerfile`, `Deploy/docker-compose.yml`, `Deploy/k8s/*.yaml` | Container build + optional monitoring stack. `.dockerignore` now excludes heavy directories so CI builds stay fast. |
| Metrics/Monitoring | Prometheus & Grafana stubs (compose + k8s), `/metrics` endpoint, `reports/runtime.json` capturing timings. |

You can demo with `make demo` (builds & runs the Docker container) and `make serve` (local uvicorn).

---

## 6. Automation & Reproducibility

| Item | Description |
|------|-------------|
| `Makefile` targets | `make reproduce` (summary + plots + README update), `make serve` (run API locally), `make demo` (build/run Docker). |
| Packaging | `pyproject.toml` exposes the CLI `automl-run`, so `pip install .` installs the whole toolkit and lets you launch the orchestrator with `automl-run --max-datasets 1`. |
| CI considerations | `.dockerignore` keeps builds lean, `pytest.ini` ignores `runs/` to avoid flaky dataset-dependent tests. |

---

## 7. Suggested Narrative for Notebooks

1. **Data Cleaning Notebook**  
   - Import `load_dataset`, `sanitize_columns`. Show missing value handling, type coercion, target inference.
   - Visualise baseline distributions (e.g., `df.describe()` and histograms).

2. **Modelling Notebook**  
   - Call `Project/trainers/train_boosters.py` (or use `subprocess` to run it) on `modeldata_demo`.  
   - Load resulting `reports/metrics/*.csv`, plot accuracy vs F1 using `matplotlib` (or reuse `figures/classifier_accuracy_precision.png`).
   - Describe cross-validation splits and calibration metrics.

3. **Analysis & Deployment Notebook**  
   - Re-run `summarize_all` + `plot_comparisons` so readers see live figure generation.  
   - Document the FastAPI service by importing `Deploy.api.serve.app` and showing `app.routes` or sample `/predict` call.  
   - Summarise monitoring (Prometheus metrics, `reports/runtime.json`) and packaging (`pip install .`, `automl-run`).

With this architecture note + inline comments in the scripts, you can translate every segment into a clean `.ipynb` explanation for your professor.
