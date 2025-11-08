"""Generate a simple exponential smoothing time-series baseline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPO_ROOT / "figures"
DEFAULT_TS_PATH = REPO_ROOT / "Project" / "src" / "data" / "timeseries_demo.csv"


def build_demo_series(target_path: Path) -> None:
    idx = pd.date_range("2023-01-01", periods=180, freq="D")
    signal = 20 + 0.05 * np.arange(180) + 2 * np.sin(np.arange(180) / 7) + np.random.normal(0, 1, 180)
    pd.DataFrame({"date": idx, "y": signal}).to_csv(target_path, index=False)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    ts_path = Path(os.getenv("TS_PATH", str(DEFAULT_TS_PATH)))
    if not ts_path.exists():
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        build_demo_series(ts_path)

    df = pd.read_csv(ts_path, parse_dates=["date"])
    df = df.sort_values("date")

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[TS] Skipping baseline (statsmodels unavailable): {exc}")
        return

    if len(df) < 21:
        print("[TS] Not enough observations for baseline (need at least 21 rows).")
        return

    train = df.iloc[:-14]
    test = df.iloc[-14:]
    model = ExponentialSmoothing(train["y"], trend="add", seasonal="add", seasonal_periods=7).fit()
    forecast = model.forecast(14)
    mae = float(np.mean(np.abs(forecast.values - test["y"].values)))

    payload = {"model": "ETS(add,add,7)", "horizon": 14, "MAE": mae, "source": str(ts_path.relative_to(REPO_ROOT))}
    (REPORTS_DIR / "timeseries_metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"[TS] Forecast MAE(14d) = {mae:.3f}")


if __name__ == "__main__":
    main()
