## Quick orientation for AI coding assistants

This repository is a small AutoML + Explainability + API demo. Use the pointers below to be productive quickly — focus on concrete files and commands rather than generic advice.

- Big picture
  - Training and analysis live under `Project/` (trainers, analysis, data). Training scripts are procedural Python scripts (not a packaged library). Key paths:
    - `Project/trainers/*` — training drivers (e.g. `train_boosters.py`, `train_flaml.py`, `train_h2o.py`).
    - `Project/analysis/*` — post-train analysis and SHAP explanation (`analyze_stats.py`, `explain_shap.py`).
    - Demo data: `Project/src/data/modeldata_demo.csv` (tests expect a demo dataset under `src/data`).
  - The API service is in `Deploy/api/` implemented with FastAPI (`Deploy/api/serve/app.py`) and exposes `/healthz`, `/readyz`, `/version`, `/metrics`, `/predict`.
  - Monitoring and compose stubs are in `Deploy/` (see `Deploy/docker-compose.yml` and `Deploy/monitoring/prometheus.yml`).

- How the project is intended to be executed (important examples)
  - From the repository root run the training pipeline:
    - `pip install -r Project/requirements_min.txt`
    - `python scripts/run_all.py` — this creates `reports/` and `figures/` and invokes `../Project/trainers/*` using relative paths.
  - Streamlit leaderboard (optional):
    - `pip install -r Project/requirements_streamlit.txt`
    - `streamlit run Project/streamlit_leaderboard.py` (expects `reports/leaderboard.csv`).
  - FastAPI locally:
    - `pip install -r Deploy/api/requirements.txt`
    - `uvicorn Deploy.api.serve.app:app --reload --port 8000`
    - Health endpoints: `/healthz`, `/readyz`, `/version`, `/metrics`.
  - Docker / Compose (API + Prometheus + Grafana):
    - `docker build -t pro_v8_api -f Deploy/api/Dockerfile .`
    - `docker run -p 8000:8000 pro_v8_api`
    - `docker compose -f Deploy/docker-compose.yml up`

- Project-specific conventions and gotchas (use these often)
  - Scripts use relative paths assuming you run commands from the repository root (e.g., `python scripts/run_all.py`); keep cwd = repo root.
  - Optional dependencies: `shap` and `h2o` are optional — training/analysis scripts will skip or degrade gracefully if they are missing. Prefer checking for guarded imports before adding changes.
  - Artifacts and UX: training writes outputs to `reports/` (leaderboard, CSVs) and `figures/` (plots). Streamlit reads `reports/leaderboard.csv` directly.
  - Tests: `tests/test_smoke.py` asserts the demo CSV exists at `../src/data/modeldata_demo.csv` (when tests are executed from `tests/` or via `pytest` from repo root ensure paths align). If you add or move demo data, update tests accordingly.

- Integration points and what to edit for common tasks
  - To change model behavior exposed by the API: edit `Deploy/api/serve/app.py` — it currently simulates model scoring (replace stub with artifact loading and real scorer).
  - To add a new trainer/framework: add a script under `Project/trainers/` and wire it into `scripts/run_all.py` (the runner calls trainers via subprocess with `../Project/trainers/<script>.py`).
  - To add monitoring metrics, extend the Prometheus setup under `Deploy/monitoring/prometheus.yml` and expose metrics in `serve/app.py` (there is already `prometheus_client` usage).

- Helpful checks and quick commands for PRs
  - Run smoke tests locally: `pytest -q tests/test_smoke.py`
  - Recreate demo run quickly: `pip install -r Project/requirements_min.txt && python scripts/run_all.py` (watch for optional heavy installs like H2O).

- Style and safety cues for code edits
  - Keep changes minimal to existing training scripts; they are simple procedural scripts — prefer adding a small wrapper script rather than refactoring many files in one PR.
  - Prefer non-blocking behavior: training scripts often continue on non-zero exit (see `scripts/run_all.py`); preserve this pattern or explicitly document change in PR description.

If any part of the environment, demo data path, or CI commands are out-of-date, tell me what changed and I'll update this file. Ready for review — would you like minor edits (more examples, CI notes) or a more detailed architecture diagram next?
