PYTHON ?= python

.PHONY: reproduce serve demo

reproduce:
	@echo "Rebuilding reports, plots, and README snippets..."
	cd $(CURDIR) && PYTHONPATH=. $(PYTHON) Project/analysis/summarize_all.py
	cd $(CURDIR) && PYTHONPATH=. $(PYTHON) Project/analysis/plot_comparisons.py
	cd $(CURDIR) && PYTHONPATH=. $(PYTHON) scripts/generate_readme_assets.py

serve:
	@echo "Starting API locally on http://127.0.0.1:8000"
	pip install -r Deploy/api/requirements-serve.txt
	uvicorn Deploy.api.serve.app:app --host 0.0.0.0 --port 8000

demo:
	@echo "Building and running demo container..."
	docker build -t automl-demo -f Deploy/api/Dockerfile .
	docker rm -f automl-demo >/dev/null 2>&1 || true
	docker run -d --name automl-demo -p 8000:8000 automl-demo
	@echo "Open http://localhost:8000/docs then submit POST /predict"
