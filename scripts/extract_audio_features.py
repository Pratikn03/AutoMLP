#!/usr/bin/env python3
"""Simple feature extractor for the synthetic audio datasets."""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
AUDIO_ROOT = REPO_ROOT / "src" / "data" / "datasets" / "audio"


def _load_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if data.size == 0:
        return np.zeros(1, dtype=np.float32), sr
    return data / 32768.0, sr


def _spectral_centroid(signal: np.ndarray, sr: int) -> float:
    if signal.size == 0:
        return 0.0
    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sr)
    denom = magnitudes.sum()
    if denom <= 1e-12:
        return 0.0
    return float((freqs * magnitudes).sum() / denom)


def extract_features(dataset: str) -> pd.DataFrame:
    dataset_root = AUDIO_ROOT / dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"Audio dataset not found: {dataset_root}")
    rows: List[Dict[str, float]] = []
    for wav_path in sorted(dataset_root.glob("*/*.wav")):
        waveform, sr = _load_wave(wav_path)
        duration = waveform.size / sr
        rms = float(np.sqrt(np.mean(np.square(waveform))))
        peak = float(np.max(np.abs(waveform)))
        mean_val = float(np.mean(waveform))
        std_val = float(np.std(waveform))
        zcr = float(np.mean(np.abs(np.diff(np.sign(waveform)))) / 2.0)
        centroid = _spectral_centroid(waveform, sr)
        rows.append(
            {
                "file": str(wav_path.relative_to(REPO_ROOT)),
                "class": wav_path.parent.name,
                "sample_rate": sr,
                "duration_sec": duration,
                "rms": rms,
                "peak": peak,
                "mean": mean_val,
                "std": std_val,
                "zero_crossing_rate": zcr,
                "spectral_centroid": centroid,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract simple audio features.")
    parser.add_argument("--dataset", required=True, help="Dataset folder name (e.g., fsdd).")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV path (defaults to reports/audio_features_<dataset>.csv).")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = extract_features(args.dataset)
    if df.empty:
        print("[audio] No WAV files located; nothing to extract.")
        return
    out_path = Path(args.out) if args.out else REPORTS_DIR / f"audio_features_{args.dataset}.csv"
    if out_path.exists() and not args.force:
        print(f"[audio] {out_path} already exists. Use --force to overwrite.")
    else:
        df.to_csv(out_path, index=False)
        print(f"[audio] Wrote feature table → {out_path}")
    summary = (
        df.groupby("class")[["duration_sec", "rms", "peak", "zero_crossing_rate", "spectral_centroid"]]
        .agg(["mean", "std"])
        .round(6)
    )
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(summary.to_json(orient="split"), encoding="utf-8")


if __name__ == "__main__":
    main()
