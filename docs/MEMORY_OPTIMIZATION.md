# Memory Optimization Guide

This guide explains how to train models with limited memory resources.

## Quick Start - Low Memory Mode

Enable low memory mode by setting environment variable before training:

```bash
export LOW_MEMORY_MODE=1
export MEMORY_LIMIT_MB=4096  # Optional: set memory limit

# Run training
PYTHONPATH=. python scripts/run_all.py --max-datasets 5
```

## Limit Tabular Runs to Safe Chunks

Large registries can contain a dozen+ CSVs. To stay within memory limits, process only six lightweight tabular datasets per invocation and let the orchestrator skip heavy files automatically:

```bash
# Pick the 6 smallest CSVs, reject anything above 150 MB, and keep cached vision/audio runs.
PYTHONPATH=. python scripts/run_all.py \
  --max-datasets 6 \
  --prefer-small-datasets \
  --dataset-max-mb 150
```

Flags to know:

- `--max-datasets 6` – hard cap on tabular datasets per run (defaults to `RUN_ALL_MAX_DATASETS` if set).
- `--prefer-small-datasets` – sorts candidates by file size so the smallest, low-memory CSVs win the slots.
- `--dataset-max-mb <size>` – skips anything larger than the provided threshold instead of risking an OOM.

The script prints which datasets were skipped and ends with a concise `[Summary] dataset (rows, columns)` line so you can report the coverage.

## What's Optimized

### 1. Data Loading
- **Automatic dtype optimization**: Downcasts integers and floats to smaller types
- **Category conversion**: Object columns with low cardinality become categorical
- **Memory reduction**: Typically 40-80% memory savings on initial load

### 2. Model Parameters
When `LOW_MEMORY_MODE=1`:
- **XGBoost**: `max_bin=128` (vs 256), `grow_policy=depthwise`
- **LightGBM**: `max_bin=128` (vs 255), `num_leaves=31` (vs 63)
- **Garbage collection**: Automatic cleanup between training runs

### 3. Additional Strategies

#### A. Process Fewer Datasets
```bash
# Train on 1-3 datasets instead of all 14
python scripts/run_all.py --max-datasets 3
```

#### B. Reduce Cross-Validation Folds
```bash
# Use 3 folds instead of 5
export N_SPLITS=3
export CATBOOST_N_SPLITS=3
```

#### C. Sample Large Datasets
For very large CSVs, sample before training:

```python
import pandas as pd
df = pd.read_csv('large_dataset.csv')
df_sample = df.sample(frac=0.5, random_state=42)  # Use 50%
df_sample.to_csv('sampled_dataset.csv', index=False)
```

#### D. Use External Memory (Advanced)
For datasets that don't fit in RAM, use chunked processing:

```python
from Project.utils.memory import load_dataset_chunked

# Load in 10k row chunks
df = load_dataset_chunked('large.csv', chunksize=10000)
```

## Memory Monitoring

### Check Current Usage
```bash
# Monitor memory during training
watch -n 1 'ps aux | grep python | head -5'
```

### Install psutil for automatic checks
```bash
pip install psutil
```

The training scripts will automatically warn if memory is low.

## Framework-Specific Tips

### AutoGluon
```bash
# Reduce preset quality
export AUTOGLUON_PRESET=medium  # instead of best_quality
```

### H2O AutoML
```bash
# Limit max runtime and models
export H2O_MAX_RUNTIME=180  # 3 minutes instead of 5
export H2O_MAX_MODELS=10    # Fewer models
```

### FLAML
```bash
# Reduce search time budget
export FLAML_TIME_BUDGET=120  # 2 minutes instead of 5
```

## Troubleshooting

### "MemoryError" or "Killed"
1. Enable `LOW_MEMORY_MODE=1`
2. Reduce `N_SPLITS` to 3
3. Train fewer datasets at once
4. Sample your data to 50-75% size
5. Use `--prefer-small-datasets --dataset-max-mb <size>` so the orchestrator auto-skips huge CSVs

### Model takes too long
1. Reduce `n_estimators` for boosters
2. Lower `H2O_MAX_RUNTIME` and `FLAML_TIME_BUDGET`
3. Skip deep learning with `--skip-deep-learning` flag
4. Skip cached global (vision/audio/timeseries) steps with `--skip-global` or rely on the default cache-aware behaviour (rerun them only when artifacts are missing; override with `--rerun-global`)

### Out of disk space
Clean up old artifacts:
```bash
rm -rf runs/*/artifacts
rm -rf artifacts/experiments
```

## Performance vs Memory Trade-offs

| Setting | Memory Impact | Accuracy Impact |
|---------|--------------|-----------------|
| `max_bin=128` | -20% | <1% drop |
| `num_leaves=31` | -15% | <2% drop |
| `N_SPLITS=3` | -33% | Similar CV reliability |
| Data sampling 50% | -50% | 2-5% drop (depends on dataset) |

## Example: Minimal Memory Configuration

For systems with 4-8GB RAM:

```bash
#!/bin/bash
export LOW_MEMORY_MODE=1
export N_SPLITS=3
export CATBOOST_N_SPLITS=3
export FLAML_TIME_BUDGET=120
export H2O_MAX_RUNTIME=180
export RUN_ALL_STEP_TIMEOUT=300

# Train on 3 small datasets
DATASET_PATHS="src/data/datasets/tabular/iris.csv:src/data/datasets/tabular/wine.csv:src/data/datasets/tabular/breast_cancer.csv" \
PYTHONPATH=. python scripts/run_all.py --max-datasets 3
```

## Monitoring Commands

```bash
# Real-time memory usage
htop

# Python-specific
pip install memory_profiler
python -m memory_profiler your_script.py

# macOS
vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages\s+([^:]+)[^\d]+(\d+)/ and printf("%-16s % 16.2f Mi\n", "$1:", $2 * $size / 1048576);'
```

## Summary

**Quick fixes for memory issues:**
1. ✅ Set `LOW_MEMORY_MODE=1`
2. ✅ Reduce `N_SPLITS=3`
3. ✅ Train fewer datasets `--max-datasets 3`
4. ✅ Lower time budgets for AutoML
5. ✅ Sample large datasets to 50-75%
