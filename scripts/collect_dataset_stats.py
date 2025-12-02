#!/usr/bin/env python3
"""Collect dataset metadata for Table S1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pandas.api import types as ptypes


TABULAR_TARGET_CANDIDATES = [
    "target",
    "class",
    "label",
    "species",
    "variety",
    "diagnosis",
    "survived",
    "salary",
]

# Manual overrides for modality/task/source/license when we know them.
DATASET_METADATA: Dict[str, Dict[str, str]] = {
    "_salary_skipped": {
        "source": "Synthetic HR comp sample",
        "license": "CC0 (user provided)",
        "task": "regression",
    },
    "breast_cancer": {
        "source": "UCI / sklearn breast cancer",
        "license": "CC BY 4.0",
        "task": "classification",
    },
    "diabetes_regression": {
        "source": "UCI / sklearn diabetes",
        "license": "CC BY 4.0",
        "task": "regression",
    },
    "heart_statlog": {
        "source": "UCI Statlog (Heart)",
        "license": "CC BY 4.0",
        "task": "classification",
    },
    "iris": {
        "source": "UCI Iris",
        "license": "Public Domain",
        "task": "classification",
    },
    "modeldata": {
        "source": "Tidymodels modeldata insurance sample",
        "license": "CC BY 4.0",
        "task": "classification",
    },
    "titanic": {
        "source": "Kaggle Titanic (preprocessed)",
        "license": "CC0",
        "task": "classification",
    },
    "train": {
        "source": "Iris CSV (demo)",
        "license": "Public Domain",
        "task": "classification",
    },
    "wine": {
        "source": "UCI Wine",
        "license": "CC BY 4.0",
        "task": "classification",
    },
    "cifar10": {
        "source": "CIFAR-10 (MIT/Toronto)",
        "license": "MIT license",
        "task": "image classification",
    },
    "fsdd": {
        "source": "Free Spoken Digit Dataset",
        "license": "MIT license",
        "task": "audio classification",
    },
}


def infer_target_column(columns: List[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in columns}
    for candidate in TABULAR_TARGET_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    # Fallback: last column if nothing obvious
    return columns[-1] if columns else None


def infer_task(series: pd.Series, fallback: str = "classification") -> str:
    if series is None:
        return fallback
    if ptypes.is_numeric_dtype(series):
        unique_vals = series.nunique(dropna=True)
        return "regression" if unique_vals > 20 else "classification"
    return "classification"


def summarize_tabular(tabular_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for csv_path in sorted(tabular_dir.glob("*.csv")):
        name = csv_path.stem
        df = pd.read_csv(csv_path)
        target_col = infer_target_column(df.columns.tolist())
        target_series = df[target_col] if target_col in df.columns else None
        n_samples, n_columns = df.shape
        n_features = n_columns - (1 if target_series is not None else 0)
        metadata = DATASET_METADATA.get(name, {})
        task = metadata.get("task") or infer_task(target_series)

        rows.append(
            {
                "name": name,
                "modality": "tabular",
                "task": task,
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "n_classes": int(target_series.nunique()) if target_series is not None and task == "classification" else "",
                "source": metadata.get("source", ""),
                "license": metadata.get("license", ""),
            }
        )
    return rows


def summarize_image(image_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset_dir in sorted(image_dir.iterdir()):
        if dataset_dir.is_dir():
            metadata_file = dataset_dir / "metadata.json"
            info = {}
            if metadata_file.exists():
                info = json.loads(metadata_file.read_text())
            name = dataset_dir.name
            metadata = DATASET_METADATA.get(name, {})
            classes = info.get("classes") or sorted(
                [p.name for p in dataset_dir.iterdir() if p.is_dir() and p.name != "metadata.json"]
            )
            rows.append(
                {
                    "name": name,
                    "modality": "vision",
                    "task": metadata.get("task", "image classification"),
                    "n_samples": info.get("total_images", ""),
                    "n_features": "",
                    "n_classes": len(classes) if classes else "",
                    "source": metadata.get("source", ""),
                    "license": metadata.get("license", ""),
                }
            )
    return rows


def summarize_audio(audio_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset_dir in sorted(audio_dir.iterdir()):
        if dataset_dir.is_dir():
            metadata_file = dataset_dir / "metadata.json"
            info = {}
            if metadata_file.exists():
                info = json.loads(metadata_file.read_text())
            name = dataset_dir.name
            metadata = DATASET_METADATA.get(name, {})
            digits = info.get("digits") or []
            rows.append(
                {
                    "name": name,
                    "modality": "audio",
                    "task": metadata.get("task", "audio classification"),
                    "n_samples": info.get("total_files", ""),
                    "n_features": "",
                    "n_classes": len(digits) if digits else "",
                    "source": metadata.get("source", ""),
                    "license": metadata.get("license", ""),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect dataset stats for Table S1.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("src/data/datasets"),
        help="Root directory that houses tabular/image/audio folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/tables/datasets_summary.csv"),
        help="CSV path to write the dataset summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    rows: List[Dict[str, object]] = []

    tabular_dir = data_root / "tabular"
    if tabular_dir.exists():
        rows.extend(summarize_tabular(tabular_dir))

    image_dir = data_root / "image"
    if image_dir.exists():
        rows.extend(summarize_image(image_dir))

    audio_dir = data_root / "audio"
    if audio_dir.exists():
        rows.extend(summarize_audio(audio_dir))

    df = pd.DataFrame(rows)
    df = df.sort_values(["modality", "name"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote dataset summary for {len(df)} datasets to {args.out}")


if __name__ == "__main__":
    main()
