"""Generate README leaderboard table and ensure Pareto plot references are fresh."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from Project.analysis.plot_comparisons import plot_accuracy_runtime_pareto

REPO_ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_MULTI = REPO_ROOT / "reports" / "leaderboard_multi.csv"
OPS_PATH = REPO_ROOT / "reports" / "leaderboard_ops.csv"
README_PATH = REPO_ROOT / "README_RUN.md"
TABLE_OUTPUT = REPO_ROOT / "reports" / "readme_leaderboard.md"
SUMMARY_OUTPUT = REPO_ROOT / "reports" / "dashboard_summary.json"
MARKER_START = "<!-- BEGIN_LEADERBOARD -->"
MARKER_END = "<!-- END_LEADERBOARD -->"
WINNER_START = "<!-- BEGIN_DATASET_WINNERS -->"
WINNER_END = "<!-- END_DATASET_WINNERS -->"


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[generate_readme_assets] Failed to read {path}: {exc}")
        return pd.DataFrame()


def build_table() -> tuple[str, pd.DataFrame, pd.DataFrame]:
    leaderboard = _load_dataframe(LEADERBOARD_MULTI)
    if leaderboard.empty:
        return "_Leaderboard not available. Run `python scripts/run_all.py` first._", leaderboard, pd.DataFrame()

    ops = _load_dataframe(OPS_PATH)[["framework", "predict_time_p95"]] if OPS_PATH.exists() else pd.DataFrame()
    merged = leaderboard.copy()
    if not ops.empty:
        merged = merged.merge(ops, on="framework", how="left")

    keep_cols = ["framework", "accuracy", "f1_macro", "predict_time_p95"]
    missing = [c for c in keep_cols if c not in merged.columns]
    if missing:
        for col in missing:
            merged[col] = float("nan")
    merged = merged[keep_cols].dropna(subset=["accuracy"]).sort_values("accuracy", ascending=False)
    top = merged.head(5).reset_index(drop=True)
    if top.empty:
        return "_Leaderboard not available. Run `python scripts/run_all.py` first._", leaderboard, top

    lines = [
        "| Rank | Framework | Accuracy | F1 Macro | Predict p95 (s) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rank, row in top.iterrows():
        lines.append(
            "| {rank} | {fw} | {acc:.3f} | {f1:.3f} | {p95} |".format(
                rank=rank + 1,
                fw=row["framework"],
                acc=float(row["accuracy"]),
                f1=float(row.get("f1_macro", float("nan"))),
                p95=("n/a" if pd.isna(row.get("predict_time_p95")) else f"{float(row['predict_time_p95']):.3f}"),
            )
        )
    lines.append("")
    lines.append(f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
    table = "\n".join(lines)
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUTPUT.write_text(table, encoding="utf-8")
    return table, leaderboard, top


def build_dataset_table(leaderboard: pd.DataFrame) -> str:
    if leaderboard.empty or "dataset" not in leaderboard.columns:
        return "_Dataset breakdown unavailable._"
    winners = (
        leaderboard.sort_values(["dataset", "accuracy"], ascending=[True, False])
        .groupby("dataset")
        .head(1)
        .reset_index(drop=True)
    )
    lines = [
        "| Dataset | Top Framework | Accuracy | F1 Macro |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in winners.iterrows():
        lines.append(
            "| {dataset} | {framework} | {acc:.3f} | {f1:.3f} |".format(
                dataset=row["dataset"],
                framework=row["framework"],
                acc=float(row["accuracy"]),
                f1=float(row.get("f1_macro", float("nan"))),
            )
        )
    return "\n".join(lines)


def update_readme(table_md: str, dataset_md: str) -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    content = f"\n{table_md}\n"
    if MARKER_START not in readme or MARKER_END not in readme:
        readme += f"\n{MARKER_START}{content}{MARKER_END}\n"
    else:
        before = readme.split(MARKER_START)[0]
        after = readme.split(MARKER_END)[-1]
        readme = f"{before}{MARKER_START}{content}{MARKER_END}{after}"
    dataset_block = f"\n{dataset_md}\n"
    if WINNER_START not in readme or WINNER_END not in readme:
        readme += f"\n{WINNER_START}{dataset_block}{WINNER_END}\n"
    else:
        before = readme.split(WINNER_START)[0]
        after = readme.split(WINNER_END)[-1]
        readme = f"{before}{WINNER_START}{dataset_block}{WINNER_END}{after}"
    README_PATH.write_text(readme, encoding="utf-8")


def main() -> None:
    # Ensure Pareto figure exists (will no-op if prerequisites missing)
    ops = _load_dataframe(OPS_PATH)
    if not ops.empty:
        plot_accuracy_runtime_pareto(ops)

    table_md, leaderboard, top_rows = build_table()
    dataset_md = build_dataset_table(leaderboard)
    update_readme(table_md, dataset_md)

    summary_payload = {
        "top_frameworks": top_rows[["framework", "accuracy", "f1_macro"]].to_dict(orient="records")
        if not top_rows.empty
        else [],
        "datasets_covered": int(leaderboard["dataset"].nunique()) if "dataset" in leaderboard.columns else 0,
        "dataset_winners": leaderboard[
            ["dataset", "framework", "accuracy", "f1_macro"]
        ]
        .sort_values(["dataset", "accuracy"], ascending=[True, False])
        .groupby("dataset")
        .head(1)
        .to_dict(orient="records")
        if not leaderboard.empty
        else [],
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print("Updated README leaderboard section and Pareto assets.")


if __name__ == "__main__":
    main()
