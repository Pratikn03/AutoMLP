import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

# Resolve paths relative to the repository root (parent of this scripts/ folder).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Add repository root to sys.path for module imports
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Project.utils.system import merge_runtime_sections


PIPELINE_STEPS = [
    "Project/trainers/train_boosters.py",
    "Project/trainers/train_catboost.py",
    "Project/trainers/train_flaml.py",
    "Project/trainers/train_h2o.py",
    "scripts/run_boosting_suite.py",
    "scripts/run_automl_suite.py",
    "scripts/run_feature_ablation.py",
    "Project/deeplearning/tabular_keras.py",
    "Project/anomaly/tabular_anomaly.py",
    "Project/analysis/analyze_stats.py",
    "Project/analysis/analyze_feature_ablations.py",
    "Project/analysis/plot_comparisons.py",
    "Project/analysis/summarize_all.py",
    "Project/analysis/explain_shap.py",
]

GLOBAL_STEPS = [
    "Project/deeplearning/image_cnn_torch.py",
    "Project/timeseries/forecast_baseline.py",
]

FRAMEWORK_META: Dict[str, Dict[str, str]] = {
    "XGBoost": {"category": "booster"},
    "LightGBM": {"category": "booster"},
    "CatBoost": {"category": "booster"},
    "FLAML": {"category": "automl"},
    "H2O_AutoML": {"category": "automl"},
    "AutoGluon": {"category": "automl"},
    "Keras_MLP": {"category": "deep_learning"},
    "ResNet18_CIFAR10": {"category": "vision"},
}


def abs_path(*parts: str) -> str:
    """Build an absolute path under the repository root."""
    return os.path.join(REPO_ROOT, *parts)


def discover_datasets(max_datasets: int = 3) -> List[Path]:
    candidates: List[Path] = []
    env_sources = os.getenv("DATASET_PATHS")
    if env_sources:
        for raw in env_sources.split(os.pathsep):
            trimmed = raw.strip()
            if not trimmed:
                continue
            path = Path(trimmed).expanduser()
            if path.exists():
                candidates.append(path.resolve())
    search_roots = [Path(REPO_ROOT) / "Project" / "src" / "data", Path(REPO_ROOT) / "src" / "data"]
    for root in search_roots:
        if not root.exists():
            continue
        for csv_path in sorted(root.glob("*.csv")):
            candidates.append(csv_path.resolve())
    unique: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        from Project.utils.io import find_csv  # lazy import to avoid circulars

        unique.append(Path(find_csv()).resolve())
    return unique[:max_datasets]


def dataset_slug(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def clean_base_dirs() -> None:
    reports_dir = Path(abs_path("reports"))
    figures_dir = Path(abs_path("figures"))
    artifacts_dir = Path(abs_path("artifacts"))
    experiments_dir = artifacts_dir / "experiments"
    for directory in [reports_dir, figures_dir]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    if experiments_dir.exists():
        shutil.rmtree(experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    for optional_dir in [artifacts_dir / "keras_dense", artifacts_dir / "vision"]:
        if optional_dir.exists():
            shutil.rmtree(optional_dir)


def run_step(rel_path: str, env: Dict[str, str], dataset: str, timeout: Optional[float]) -> Dict[str, object]:
    pyfile = abs_path(*rel_path.split("/"))
    print(f"[Run] ({dataset}) {pyfile}")
    started = time.time()
    status = "ok"
    process = subprocess.Popen([sys.executable, pyfile], cwd=REPO_ROOT, env=env)
    try:
        exit_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        status = "timeout"
        process.kill()
        exit_code = -9
        print(f"[WARN] {pyfile} exceeded timeout of {timeout} seconds; process terminated.")
    duration = time.time() - started
    entry = {
        "dataset": dataset,
        "script": rel_path,
        "path": pyfile,
        "exit_code": int(exit_code),
        "status": status if exit_code == 0 else (status if status != "ok" else "error"),
        "duration_sec": round(duration, 3),
        "timestamp": time.time(),
    }
    if exit_code != 0 and status != "timeout":
        print(f"[WARN] {pyfile} exited with {exit_code} (continuing).")
    return entry


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def collect_leaderboard(dataset: str) -> Tuple[str, Path, Path]:
    reports_root = Path(abs_path("reports"))
    figures_root = Path(abs_path("figures"))
    artifacts_root = Path(abs_path("artifacts"))
    dst_root = Path(abs_path("runs")) / dataset
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    copy_tree(reports_root, dst_root / "reports")
    copy_tree(figures_root, dst_root / "figures")
    copy_tree(artifacts_root, dst_root / "artifacts")
    return dataset, dst_root / "reports" / "leaderboard.csv", dst_root / "reports" / "runtime.json"


def load_csv_stats(path: Path) -> Dict[str, object]:
    import pandas as pd

    df = pd.read_csv(path)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_candidates": [col for col in df.columns if df[col].nunique() <= 20],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AutoML training + analysis pipeline across datasets.")
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=int(os.getenv("RUN_ALL_MAX_DATASETS", "3")),
        help="Maximum number of datasets to process (default: 3 or RUN_ALL_MAX_DATASETS).",
    )
    parser.add_argument(
        "--flaml-time-budget",
        type=int,
        default=int(os.getenv("FLAML_TIME_BUDGET", "45")),
        help="Fallback FLAML time budget in seconds if the environment variable is not set (default: 45).",
    )
    parser.add_argument(
        "--step-timeout",
        type=int,
        default=int(os.getenv("RUN_ALL_STEP_TIMEOUT", "0")),
        help="Optional timeout in seconds for each pipeline step. 0 disables the timeout.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    runs_root = Path(abs_path("runs"))
    runs_root.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(max_datasets=args.max_datasets)
    dataset_infos: List[Dict[str, object]] = []
    orchestrator_runtime: List[Dict[str, object]] = []
    leaderboard_frames: List[Any] = []
    timeout_val: Optional[float] = None
    if args.step_timeout > 0:
        timeout_val = float(args.step_timeout)

    clean_base_dirs()

    for idx, dataset_path in enumerate(datasets):
        slug = dataset_slug(dataset_path)
        stats = cast(Dict[str, Any], load_csv_stats(dataset_path))
        try:
            rows = int(stats.get("rows", 0))
        except (TypeError, ValueError):
            rows = 0
        try:
            columns = int(stats.get("columns", 0))
        except (TypeError, ValueError):
            columns = 0
        target_candidates = stats.get("target_candidates", [])
        if not isinstance(target_candidates, list):
            target_candidates = []
        info = {
            "dataset": slug,
            "path": str(dataset_path),
            "rows": rows,
            "columns": columns,
            "target_candidates": target_candidates[:5],
        }
        dataset_infos.append(info)

        print(f"\n=== Dataset {idx + 1}/{len(datasets)} :: {dataset_path} (slug={slug}) ===")

        env = os.environ.copy()
        env["PYTHONPATH"] = REPO_ROOT
        env.setdefault("PYTHONHASHSEED", "42")
        env.setdefault("TARGET", env.get("TARGET", "IsInsurable"))
        env.setdefault("FLAML_TIME_BUDGET", str(args.flaml_time_budget))
        env["CSV_PATH"] = str(dataset_path)

        for step in PIPELINE_STEPS:
            entry = run_step(step, env, slug, timeout_val)
            orchestrator_runtime.append(entry)

        dataset, leaderboard_path, _runtime_path = collect_leaderboard(slug)

        if leaderboard_path.exists():
            import pandas as pd

            df = pd.read_csv(leaderboard_path)
            df["dataset"] = dataset
            leaderboard_frames.append(df)

        if idx < len(datasets) - 1:
            clean_base_dirs()

    if GLOBAL_STEPS:
        global_env = os.environ.copy()
        global_env["PYTHONPATH"] = REPO_ROOT
        global_env.setdefault("PYTHONHASHSEED", "42")
        global_env.setdefault("TARGET", global_env.get("TARGET", "IsInsurable"))
        global_env.setdefault("FLAML_TIME_BUDGET", str(args.flaml_time_budget))
        for step in GLOBAL_STEPS:
            entry = run_step(step, global_env, "global", timeout_val)
            orchestrator_runtime.append(entry)

    if leaderboard_frames:
        import pandas as pd

        reports_base = Path(abs_path("reports"))
        reports_base.mkdir(parents=True, exist_ok=True)

        combined = pd.concat(leaderboard_frames, ignore_index=True)
        combined_path = reports_base / "leaderboard_multi.csv"
        combined.to_csv(combined_path, index=False)
        combined.to_json(combined_path.with_suffix(".json"), orient="records", indent=2)

        registry_rows = []
        for framework in sorted(combined["framework"].unique()):
            meta = FRAMEWORK_META.get(framework, {})
            registry_rows.append({
                "framework": framework,
                "category": meta.get("category", "unknown"),
                "is_booster": meta.get("category", "").lower() == "booster",
                "datasets_covered": int(combined.loc[combined["framework"].eq(framework), "dataset"].nunique()),
            })
        registry_path = Path(abs_path("reports")) / "framework_registry.json"
        registry_path.write_text(json.dumps(registry_rows, indent=2))
        pd.DataFrame(registry_rows).to_csv(registry_path.with_suffix(".csv"), index=False)

    reports_dir = Path(abs_path("reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    registry_dataset_path = reports_dir / "dataset_registry.json"
    registry_dataset_path.write_text(json.dumps(dataset_infos, indent=2))
    try:
        import pandas as pd

        pd.DataFrame(dataset_infos).to_csv(registry_dataset_path.with_suffix(".csv"), index=False)
    except Exception:
        pass

    if orchestrator_runtime:
        merge_runtime_sections({"orchestration": orchestrator_runtime})

    # Aggregate runtime logs across datasets (including training/explainability)
    aggregated_runtime: Dict[str, List[Dict[str, object]]] = {}
    for info in dataset_infos:
        dataset_slug_value = str(info.get("dataset"))
        runtime_path = runs_root / dataset_slug_value / "reports" / "runtime.json"
        if not runtime_path.exists():
            continue
        try:
            data = json.loads(runtime_path.read_text())
        except Exception:
            continue
        for key, entries in data.items():
            bucket = aggregated_runtime.setdefault(key, [])
            if isinstance(entries, list):
                bucket.extend(entries)
    aggregated_runtime.setdefault("orchestration", []).extend(orchestrator_runtime)
    runtime_root = Path(abs_path("reports")) / "runtime.json"
    runtime_root.write_text(json.dumps(aggregated_runtime, indent=2))

    print(f"All tasks finished. Datasets captured: {[info['dataset'] for info in dataset_infos]}")
    print(f"Reports root → {abs_path('reports')} | Runs archive → {abs_path('runs')}")


if __name__ == "__main__":
    main()
