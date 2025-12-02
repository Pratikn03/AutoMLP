#!/usr/bin/env bash
set -euo pipefail

# === configuration knobs ===
RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RUN_ROOT}/.venv311/bin/python"
PIP_BIN="${RUN_ROOT}/.venv311/bin/pip"
DATASET_PATHS_ENV="src/data/datasets/tabular/adult_income.csv:src/data/datasets/tabular/bank_marketing.csv:src/data/datasets/tabular/breast_cancer.csv:src/data/datasets/tabular/diabetes_regression.csv"
FOLDS=${FOLDS:-3}
STEP_TIMEOUT=${STEP_TIMEOUT:-0}
FLAML_BUDGET=${FLAML_BUDGET:-250}
H2O_MAX_RUNTIME=${H2O_MAX_RUNTIME:-300}

# === helper logging ===
log(){
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

# === sanity checks ===
if [[ ! -x "${PYTHON_BIN}" ]]; then
  log "Python venv not found; create it via 'python3 -m venv .venv311 && . .venv311/bin/activate'"
  exit 1
fi

# shellcheck disable=SC1091
source "${RUN_ROOT}/.venv311/bin/activate"

# === dependency bootstrap ===
log "Installing core requirements"
"${PIP_BIN}" install -r "${RUN_ROOT}/Project/requirements_min.txt" -r "${RUN_ROOT}/Deploy/api/requirements.txt"

log "Installing optional extras"
"${PIP_BIN}" install shap lime h2o lightgbm xgboost catboost flaml autogluon lightautoml polars pyarrow --upgrade

# === dataset audit ===
audit_dataset(){
  local path="$1"
  if [[ ! -f "${RUN_ROOT}/${path}" ]]; then
    log "Missing dataset: ${path}"
    return 1
  fi
  head -n 1 "${RUN_ROOT}/${path}" >/dev/null || {
    log "Failed to read dataset header: ${path}"
    return 1
  }
  return 0
}

log "Auditing tabular datasets"
IFS=':' read -r -a DATASET_LIST <<< "${DATASET_PATHS_ENV}"
for ds in "${DATASET_LIST[@]}"; do
  audit_dataset "${ds}" || exit 1
  log "  ok -> ${ds}"
done

log "Checking image datasets"
for image_dir in beans fashion_mnist; do
  if [[ ! -d "${RUN_ROOT}/src/data/datasets/image/${image_dir}" ]]; then
    log "  warning: missing image dataset directory: ${image_dir}"
    continue
  fi
  log "  ok -> image/${image_dir}"
done

log "Checking audio datasets (optional)"
if compgen -G "${RUN_ROOT}/src/data/datasets/audio/*" > /dev/null; then
  find "${RUN_ROOT}/src/data/datasets/audio" -maxdepth 1 -type d -not -path '*/audio' -print | sed 's/^/  found -> /'
else
  log "  warning: no audio datasets located; audio trainers will be skipped"
fi

# === dry run ===
log "Dry-run sanity check (timeout 30 sec)"
RUN_ALL_TIMEOUT=30 TARGET="" FLAML_TIME_BUDGET="${FLAML_BUDGET}" H2O_MAX_RUNTIME="${H2O_MAX_RUNTIME}" N_SPLITS="${FOLDS}" DATASET_PATHS="${DATASET_PATHS_ENV}" \
  python "${RUN_ROOT}/scripts/run_all.py" --max-datasets 1 --step-timeout 30 || {
    log "Dry run failed; inspect logs before continuing"
    exit 1
  }

# === full launch ===
log "Starting full pipeline"
DATASET_PATHS="${DATASET_PATHS_ENV}" \
N_SPLITS="${FOLDS}" \
FLAML_TIME_BUDGET="${FLAML_BUDGET}" \
H2O_MAX_RUNTIME="${H2O_MAX_RUNTIME}" \
RUN_ALL_STEP_TIMEOUT="${STEP_TIMEOUT}" \
python "${RUN_ROOT}/scripts/run_all.py" --max-datasets "${#DATASET_LIST[@]}"

log "Pipeline finished. Artifacts in ${RUN_ROOT}/runs"