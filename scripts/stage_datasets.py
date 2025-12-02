#!/usr/bin/env python3
"""Utility to stage lightweight demo datasets for the AutoML project.

This script intentionally keeps dependencies simple (numpy + pillow) and
generates synthetic samples that resemble the folder layouts expected by
the vision/audio pipelines.  That way `PYTHONPATH=. python stage_datasets.py`
can be run on any machine (or in CI) without relying on large public downloads.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import wave
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
VISION_ROOT = REPO_ROOT / "src" / "data" / "datasets" / "image" / "cifar10"
AUDIO_ROOT = REPO_ROOT / "src" / "data" / "datasets" / "audio" / "fsdd"


def _log(msg: str) -> None:
    print(f"[stage] {msg}")


def _write_image(path: Path, seed: int, size: int = 64) -> None:
    rng = np.random.default_rng(seed)
    base = rng.random((size, size, 3), dtype=np.float32)
    # emphasise horizontal vs vertical structure to make classes distinct
    gradient = np.linspace(0, 1, size, dtype=np.float32)
    grad_map = np.tile(gradient[:, None], (1, size))
    base[:, :, 0] += grad_map
    base[:, :, 1] += grad_map.T
    base[:, :, 2] = 1.0 - base[:, :, 0]
    arr = np.clip(base * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, format="PNG")


def stage_vision(samples_per_class: int = 12, force: bool = False) -> List[str]:
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    VISION_ROOT.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for cls in classes:
        cls_dir = VISION_ROOT / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        existing = list(cls_dir.glob("*.png"))
        if existing and not force:
            continue
        for img_idx in range(samples_per_class):
            out_path = cls_dir / f"{cls}_{img_idx:03d}.png"
            _write_image(out_path, seed=hash((cls, img_idx)) & 0xFFFF)
            written.append(str(out_path.relative_to(REPO_ROOT)))
    meta = {
        "root": str(VISION_ROOT),
        "classes": classes,
        "total_images": sum(len(list((VISION_ROOT / c).glob("*.png"))) for c in classes),
    }
    (VISION_ROOT / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return written


def _write_wave(path: Path, freq: float, sr: int, duration: float, phase: float) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    signal = 0.5 * np.sin(2.0 * math.pi * freq * t + phase)
    signal += 0.05 * np.random.standard_normal(signal.shape)
    signal = np.clip(signal, -1.0, 1.0)
    pcm = (signal * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def stage_audio(samples_per_digit: int = 6, force: bool = False) -> List[str]:
    digits = {
        "zero": 440.0,
        "one": 494.0,
        "two": 523.0,
        "three": 587.0,
        "four": 659.0,
        "five": 698.0,
        "six": 784.0,
        "seven": 880.0,
        "eight": 988.0,
        "nine": 1047.0,
    }
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    sr = 16000
    duration = 1.0
    written: List[str] = []
    for digit, freq in digits.items():
        digit_dir = AUDIO_ROOT / digit
        digit_dir.mkdir(parents=True, exist_ok=True)
        existing = list(digit_dir.glob("*.wav"))
        if existing and not force:
            continue
        for idx in range(samples_per_digit):
            out_path = digit_dir / f"{digit}_{idx:02d}.wav"
            phase = random.random() * 2 * math.pi
            _write_wave(out_path, freq, sr, duration, phase)
            written.append(str(out_path.relative_to(REPO_ROOT)))
    meta = {
        "root": str(AUDIO_ROOT),
        "sample_rate": sr,
        "duration_seconds": duration,
        "digits": list(digits.keys()),
        "total_files": sum(len(list((AUDIO_ROOT / d).glob("*.wav"))) for d in digits),
    }
    (AUDIO_ROOT / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage lightweight demo datasets (vision/audio).")
    parser.add_argument("--vision", action="store_true", help="Generate the synthetic vision dataset (ImageFolder style).")
    parser.add_argument("--audio", action="store_true", help="Generate the synthetic audio dataset (digit WAVs).")
    parser.add_argument("--force", action="store_true", help="Regenerate files even if they already exist.")
    parser.add_argument("--vision-samples", type=int, default=12, help="Images per class to generate (default: 12).")
    parser.add_argument("--audio-samples", type=int, default=6, help="Audio clips per digit to generate (default: 6).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets: List[str] = []
    if args.vision:
        targets.append("vision")
    if args.audio:
        targets.append("audio")
    if not targets:
        targets = ["vision", "audio"]

    if "vision" in targets:
        created = stage_vision(samples_per_class=args.vision_samples, force=args.force)
        _log(f"Vision dataset ready at {VISION_ROOT} ({'regenerated' if created else 'already present'})")
    if "audio" in targets:
        created = stage_audio(samples_per_digit=args.audio_samples, force=args.force)
        _log(f"Audio dataset ready at {AUDIO_ROOT} ({'regenerated' if created else 'already present'})")


if __name__ == "__main__":
    main()
