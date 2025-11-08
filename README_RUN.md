# Pro v8 — All-In-One (AutoML + Explainability + API + Monitoring)

## Quick start (training on demo data)
```bash
# Python 3.9+ recommended (3.10/3.11 OK). Virtual env suggested.
python -m pip install --upgrade pip
pip install -r Project/requirements_min.txt
python scripts/run_all.py
# optionally point to specific CSVs (up to 3) using PATH-separated values
# DATASET_PATHS="src/data/modeldata.csv:src/data/salary.csv:Project/src/data/modeldata_demo.csv" python scripts/run_all.py
```

Artifacts are written to `reports/` and `figures/`.

Key outputs after the orchestrator completes:
- `reports/framework_summary.csv` / `.json` and `reports/framework_comparisons.csv` / `.json`
- `reports/shap_global_summary.csv`, `reports/shap_sample_details.csv`, and `reports/lime_sample_details.json`
- `reports/runtime.json` (training, explainability, and orchestration timings with resource snapshots)
- `reports/leaderboard_multi.csv` aggregating per-dataset results, plus dataset-level registries in `reports/dataset_registry.{json,csv}` and framework metadata in `reports/framework_registry.{json,csv}`
- `runs/<dataset>/...` archives (metrics, figures, artifacts) for each dataset analysed; global summaries remain in `reports/`
- `figures/shap/*.png` alongside feature-importance plots under `figures/feature_importance/`

Re-running `python scripts/run_all.py` from the repository root will regenerate every artifact above using the persisted configuration and seeds; the `runs/` directory is refreshed with one archive per dataset (up to three discovered via `DATASET_PATHS` or automatic CSV discovery).

Note: the project's default target column name was renamed from `SLA_Breached` to `IsInsurable`.
The data loader (`Project/utils/io.py`) contains a backwards-compatibility shim that will automatically map `SLA_Breached` → `IsInsurable` if the old column name is present.

## Streamlit leaderboard (optional)
```bash
pip install -r Project/requirements_streamlit.txt
streamlit run Project/streamlit_leaderboard.py
# open http://localhost:8501
```

## FastAPI service (optional)
```bash
pip install -r Deploy/api/requirements.txt
uvicorn Deploy.api.serve.app:app --reload --port 8000
# health:   http://127.0.0.1:8000/healthz
# readiness:http://127.0.0.1:8000/readyz
# version:  http://127.0.0.1:8000/version
# metrics:  http://127.0.0.1:8000/metrics
# latency probe: http://127.0.0.1:8000/probe/latency
```

## Docker (API only)
```bash
# build
docker build -t pro_v8_api -f Deploy/api/Dockerfile .
# run
docker run -p 8000:8000 pro_v8_api
```

## Docker Compose (API + Prometheus + Grafana stubs)
```bash
docker compose -f Deploy/docker-compose.yml up
# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin/admin)
```

## GitHub Actions (CI)
See `.github/workflows/ci-cd.yaml` – runs smoke tests, builds Docker image.
Enable your repo's actions and set registry secrets if you want to push.
