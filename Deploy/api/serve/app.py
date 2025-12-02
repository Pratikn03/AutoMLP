from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os, json, time, glob
from pathlib import Path
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

import pandas as pd

# Optional H2O (loaded at startup if available)
API_ENABLE_H2O = os.getenv("API_ENABLE_H2O", "").strip().lower() in {"1", "true", "yes", "on"}
H2O_AVAILABLE = False
H2O_MODEL = None
H2O = None
H2O_METADATA_PATH = Path("Project") / "artifacts" / "h2o" / "model_metadata.json"

# Simple in-memory "model" simulation; in real use, load from artifacts/
APP_VERSION = "pro_v8"
READY = False

app = FastAPI(title="AutoML Pro v8 API")
REQS = Counter("api_requests_total", "Total API requests", ["path","status"])
LAT = Histogram("api_latency_seconds", "Request latency", buckets=(0.05,0.1,0.2,0.5,1,2,5))

class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dictionary matching training schema")

@app.on_event("startup")
async def startup():
    global READY
    global H2O_AVAILABLE, H2O_MODEL, H2O
    # Simulate warmup: load artifacts if exist
    # For this template, just flip READY after short delay
    time.sleep(0.2)
    if API_ENABLE_H2O:
        # Try to load an H2O model if h2o is installed and an artifact exists.
        try:
            import h2o as _h2o

            H2O = _h2o
            H2O_AVAILABLE = True
            try:
                # initialize h2o JVM (no-ui)
                H2O.init()
            except Exception:
                # ignore init failures; loading may still work in some setups
                pass

            def _load_from_metadata() -> Optional[Any]:
                if not H2O_METADATA_PATH.exists():
                    return None
                payload = json.loads(H2O_METADATA_PATH.read_text(encoding="utf-8"))
                model_path = payload.get("model_path")
                mojo_path = payload.get("mojo_path")
                if model_path and Path(model_path).exists():
                    return H2O.load_model(str(Path(model_path)))
                if mojo_path and Path(mojo_path).exists():
                    try:
                        return H2O.import_mojo(str(Path(mojo_path)))
                    except Exception:
                        return None
                return None

            H2O_MODEL = _load_from_metadata()
            if H2O_MODEL is None:
                # fallback to best-effort scan of artifact directory
                mojo_dir = os.path.join("Project", "artifacts", "h2o")
                if os.path.isdir(mojo_dir):
                    candidates = glob.glob(os.path.join(mojo_dir, "*"))
                    for c in candidates:
                        try:
                            model = H2O.load_model(c)
                            H2O_MODEL = model
                            break
                        except Exception:
                            continue
        except Exception:
            H2O_AVAILABLE = False
    READY = True

def _status_payload() -> Dict[str, Any]:
    return {"status": "ok", "ready": READY, "h2o_model_loaded": bool(H2O_MODEL)}

@app.get("/health")
def health():
    payload = _status_payload()
    REQS.labels(path="/health", status="200").inc()
    return payload

@app.get("/healthz")
def healthz():
    REQS.labels(path="/healthz", status="200").inc()
    return _status_payload()

@app.get("/readyz")
def readyz():
    REQS.labels(path="/readyz", status="200").inc()
    return {"ready": READY}

@app.get("/version")
def version():
    REQS.labels(path="/version", status="200").inc()
    return {"version": APP_VERSION}

@app.get("/metrics")
def metrics():
    REQS.labels(path="/metrics", status="200").inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/probe/latency")
def probe_latency(iterations: int = 1000):
    started = time.perf_counter()
    upper = max(1, min(int(iterations), 50_000))
    _ = sum(i * i for i in range(upper))
    elapsed = time.perf_counter() - started
    LAT.observe(elapsed)
    REQS.labels(path="/probe/latency", status="200").inc()
    return {"probe_latency_sec": elapsed, "iterations": upper}

@app.post("/predict")
def predict(req: PredictRequest):
    if not READY:
        REQS.labels(path="/predict", status="503").inc()
        raise HTTPException(status_code=503, detail="Not ready")
    t0 = time.time()
    # If an H2O model was loaded at startup, use it
    if H2O_AVAILABLE and H2O_MODEL is not None and H2O is not None:
        try:
            # convert features to a single-row DataFrame
            df = pd.DataFrame([req.features])
            hf = H2O.H2OFrame(df)
            preds = H2O_MODEL.predict(hf)
            # preds -> H2OFrame; convert to pandas
            p = preds.as_data_frame(use_pandas=True)
            # try to extract binary probability column 'p1' else fallback to first numeric column
            if "p1" in p.columns:
                prob = float(p.loc[0, "p1"])
            else:
                # if predict returns label and probabilities, try last column
                try:
                    prob = float(p.iloc[0].dropna().astype(float).iloc[-1])
                except Exception:
                    prob = float("nan")
            # predicted class
            label = None
            if "predict" in p.columns:
                label = p.loc[0, "predict"]

            LAT.observe(time.time()-t0)
            REQS.labels(path="/predict", status="200").inc()
            out = {"probability": prob}
            if label is not None:
                out["label"] = label
            return out
        except Exception as e:
            # fall back to stub and continue
            print("[API] H2O predict failed:", e)

    # Fallback stubbed scorer (sum of numeric fields)
    score = 0.0
    for k, v in req.features.items():
        try:
            score += float(v)
        except Exception:
            pass
    prob = 1.0 / (1.0 + pow(2.71828, -0.001 * score))
    LAT.observe(time.time() - t0)
    REQS.labels(path="/predict", status="200").inc()
    return {"probability": prob}
