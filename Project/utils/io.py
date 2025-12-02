"""Utility helpers for dataset discovery and target selection."""

from __future__ import annotations

import os
import pathlib
from typing import Optional

import pandas as pd

from Project.utils.sanitize import safe_col

# Resolve repository root
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def find_csv() -> str:
    """Return path to dataset, preferring explicit CSV_PATH."""
    env = os.getenv("CSV_PATH")
    if env and pathlib.Path(env).exists():
        return env
    # Local project defaults
    candidates = [
        REPO_ROOT / "src/data/modeldata.csv",
        REPO_ROOT / "src/data/modeldata_demo.csv",
        REPO_ROOT / "Project/src/data/modeldata.csv",
        REPO_ROOT / "Project/src/data/modeldata_demo.csv",
        REPO_ROOT / "src/data/datasets/tabular/modeldata.csv",
        REPO_ROOT / "src/data/datasets/tabular/modeldata_demo.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Could not locate dataset. Set CSV_PATH or place modeldata.csv under src/data/."
    )


def load_dataset(low_memory: bool = True) -> pd.DataFrame:
    """Load dataset with light backward compatibility adjustments.
    
    Args:
        low_memory: If True, use memory-efficient dtypes and optimize memory usage
    """
    path = find_csv()
    
    if low_memory:
        # Read with low_memory mode and optimize dtypes
        df = pd.read_csv(path, low_memory=True)
        
        # Downcast numeric columns to save memory
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Convert object columns to category where appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
    else:
        df = pd.read_csv(path)
    
    if "IsInsurable" not in df.columns and "SLA_Breached" in df.columns:
        df = df.rename(columns={"SLA_Breached": "IsInsurable"})
    return df


def _is_binary(series: pd.Series) -> bool:
    """Return True if a column behaves like a binary/boolean feature."""
    if series.dtype == bool:
        return True
    # Drop NA and normalise string representations for common yes/no variants
    values = series.dropna().unique()
    if len(values) == 0:
        return False
    if len(values) <= 2:
        return True
    lowered = {str(v).strip().lower() for v in values if str(v).strip() != ""}
    return lowered <= {"yes", "no", "true", "false", "0", "1"}


def guess_target_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    """Heuristically determine the target column.

    Preference order:
      1. Explicit `preferred` name (or current TARGET env var) if present.
      2. Binary columns whose sanitised name starts with ``I`` (e.g. IsFraud).
      3. Any other binary/boolean column.
      4. Last column in the dataframe as a final fallback.
    """

    candidates = list(df.columns)
    if not candidates:
        raise ValueError("Dataframe has no columns to select a target from.")

    def _resolve(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        name_clean = name.strip()
        variants = [name_clean, safe_col(name_clean)]
        for variant in variants:
            if variant in df.columns:
                return variant
        return None

    env_target = os.getenv("TARGET")
    explicit = _resolve(preferred) or _resolve(env_target) or _resolve("IsInsurable")
    if explicit:
        return explicit

    binary_cols = [col for col in candidates if _is_binary(df[col])]
    prioritised = [col for col in binary_cols if col.lower().startswith("i")]
    if prioritised:
        return prioritised[0]
    if binary_cols:
        return binary_cols[0]
    return candidates[-1]


__all__ = ["find_csv", "load_dataset", "guess_target_column"]
