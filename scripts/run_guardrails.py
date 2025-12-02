#!/usr/bin/env python3
"""Lightweight guardrail checker for benchmark datasets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TARGET_CANDIDATES = [
    "target",
    "class",
    "label",
    "variety",
    "species",
    "diagnosis",
    "survived",
    "salary",
]

TIME_COLUMN_PATTERNS = re.compile(r"(date|time|timestamp)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run guardrail diagnostics for potential leakage issues."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("src/data/datasets/tabular"),
        help="Directory containing tabular CSV datasets to audit.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/guardrails"),
        help="Directory where guardrail JSON reports will be stored.",
    )
    parser.add_argument(
        "--min-rows-for-duplicate-check",
        type=int,
        default=50,
        help="Skip duplicate scanning for tiny datasets.",
    )
    parser.add_argument(
        "--high-cardinality-threshold",
        type=float,
        default=0.98,
        help="Threshold for flagging columns that almost uniquely determine the target.",
    )
    return parser.parse_args()


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    lowercase = {col.lower(): col for col in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lowercase:
            return lowercase[candidate]
    # fallback: last column if binary/categorical
    tail = df.columns[-1]
    if df[tail].nunique(dropna=True) <= 50:
        return tail
    return None


def check_duplicate_rows(
    df: pd.DataFrame, min_rows: int
) -> Optional[Dict[str, object]]:
    if len(df) < min_rows:
        return None
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        return None
    return {
        "id": "duplicate_rows",
        "severity": "warning",
        "summary": {
            "duplicates": dup_count,
            "fraction": dup_count / len(df),
        },
        "description": f"Found {dup_count} duplicated records. "
        "Duplicates across folds can leak validation data.",
        "suggested_fix": "Deduplicate before splitting or ensure GroupKFold/GroupShuffleSplit.",
    }


def check_high_cardinality_proxy(
    df: pd.DataFrame, target_col: str, threshold: float
) -> List[Dict[str, object]]:
    issues: List[Dict[str, object]] = []
    target = df[target_col]
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        if series.isnull().all():
            continue
        grouped = (
            df.groupby(series, dropna=False)[target_col]
            .nunique(dropna=True)
            .max()
        )
        if grouped == 1 and series.nunique(dropna=True) > 3:
            issues.append(
                {
                    "id": "deterministic_proxy",
                    "column": col,
                    "severity": "high",
                    "description": f"Column '{col}' deterministically maps to target '{target_col}'.",
                    "suggested_fix": "Drop the proxy feature or treat it as leakage.",
                }
            )
            continue
        if target.nunique() <= 1:
            continue
        ratio = df[[col, target_col]].dropna().groupby(col)[target_col].nunique().mean()
        # ratio near 1 indicates near-deterministic mapping
        if ratio <= 1.05 and series.nunique(dropna=True) / len(df) >= threshold:
            issues.append(
                {
                    "id": "high_cardinality_proxy",
                    "column": col,
                    "severity": "warning",
                    "description": f"Column '{col}' almost uniquely identifies rows, "
                    "risking leakage if fold splits are random.",
                    "suggested_fix": "Use GroupKFold on that column or remove it.",
                }
            )
    return issues


def check_path_token_leakage(df: pd.DataFrame, target_col: str) -> Optional[Dict[str, object]]:
    object_cols = df.select_dtypes(include=["object"]).columns
    if not len(object_cols):
        return None
    target_values = df[target_col].astype(str).unique()
    suspicious_cols: List[str] = []
    for col in object_cols:
        series = df[col].astype(str)
        if series.str.contains("/").any() or series.str.contains("\\\\").any():
            hits = series.str.contains("|".join(map(re.escape, target_values)), regex=True)
            if hits.any():
                suspicious_cols.append(col)
    if not suspicious_cols:
        return None
    return {
        "id": "path_token_leakage",
        "columns": suspicious_cols,
        "severity": "high",
        "description": "File-path like columns include target tokens; "
        "image/audio pipelines should sanitize paths before splitting.",
        "suggested_fix": "Strip target strings from paths or hash filenames pre-split.",
    }


def check_temporal_order(df: pd.DataFrame) -> Optional[Dict[str, object]]:
    candidate_cols = [
        col for col in df.columns if TIME_COLUMN_PATTERNS.search(col)
    ]
    if not candidate_cols:
        return None
    unsorted = []
    for col in candidate_cols:
        series = pd.to_datetime(df[col], errors="coerce")
        if series.isnull().all():
            continue
        if not series.is_monotonic_increasing:
            unsorted.append(col)
    if not unsorted:
        return None
    return {
        "id": "temporal_split_warning",
        "columns": unsorted,
        "severity": "info",
        "description": "Timestamp columns are not strictly sorted; ensure TimeSeriesSplit "
        "or blocking is used to avoid look-ahead leakage.",
        "suggested_fix": "Generate chronological folds via TimeSeriesSplit before training.",
    }


def audit_dataset(
    csv_path: Path,
    min_rows: int,
    threshold: float,
) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    report: Dict[str, object] = {
        "dataset": csv_path.stem,
        "path": str(csv_path),
        "rows": len(df),
        "columns": df.shape[1],
        "issues": [],
    }
    target_col = infer_target_column(df)
    if target_col is None:
        report["issues"].append(
            {
                "id": "missing_target",
                "severity": "info",
                "description": "No obvious target column found; skipping leakage-specific checks.",
                "suggested_fix": "Specify the target explicitly when running experiments.",
            }
        )
        return report

    duplicate_issue = check_duplicate_rows(df, min_rows)
    if duplicate_issue:
        report["issues"].append(duplicate_issue)

    report["issues"].extend(
        check_high_cardinality_proxy(df, target_col, threshold)
    )

    path_issue = check_path_token_leakage(df, target_col)
    if path_issue:
        report["issues"].append(path_issue)

    temporal_issue = check_temporal_order(df)
    if temporal_issue:
        report["issues"].append(temporal_issue)

    if not report["issues"]:
        report["status"] = "pass"
    else:
        report["status"] = "issues_found"
    return report


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []

    for csv_path in sorted(args.data_root.glob("*.csv")):
        report = audit_dataset(
            csv_path,
            args.min_rows_for_duplicate_check,
            args.high_cardinality_threshold,
        )
        out_file = args.out / f"{csv_path.stem}.json"
        out_file.write_text(json.dumps(report, indent=2))
        summaries.append(
            {
                "dataset": report["dataset"],
                "status": report["status"],
                "issue_count": len(report["issues"]),
                "path": str(out_file),
            }
        )
        print(f"[guardrails] {csv_path.stem}: {len(report['issues'])} issue(s)")

    index_path = args.out / "guardrails_index.json"
    index_path.write_text(json.dumps(summaries, indent=2))
    print(f"Wrote guardrail reports for {len(summaries)} datasets to {args.out}")


if __name__ == "__main__":
    main()
