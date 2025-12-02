
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


def _env_flag(name: str, default: bool = False) -> bool:
    """Interpret standard truthy environment variables."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
    "Project/nlp/train_sms_spam.py",
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
    "Logistic_TFIDF": {"category": "nlp"},
}


def abs_path(*parts: str) -> str:
    """Build an absolute path under the repository root."""
    return os.path.join(REPO_ROOT, *parts)


GLOBAL_STEP_OUTPUTS: Dict[str, List[Path]] = {
    "Project/deeplearning/image_cnn_torch.py": [
        Path(abs_path("reports")) / "leaderboard_vision.csv",
        Path(abs_path("reports")) / "vision_metrics.json",
    ],
    "Project/nlp/train_sms_spam.py": [
        Path(abs_path("reports")) / "leaderboard_nlp.csv",
    ],
    "Project/timeseries/forecast_baseline.py": [
        Path(abs_path("reports")) / "timeseries_metrics.json",
    ],
}


def discover_datasets(
    max_datasets: int = 3,
    prefer_small: bool = False,
    max_size_mb: Optional[float] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
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
    search_roots = [
        Path(REPO_ROOT) / "Project" / "src" / "data",
        Path(REPO_ROOT) / "src" / "data",
        Path(REPO_ROOT) / "Project" / "src" / "data" / "datasets",
        Path(REPO_ROOT) / "src" / "data" / "datasets",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for csv_path in sorted(root.rglob("*.csv")):
            candidates.append(csv_path.resolve())
    dataset_meta: List[Dict[str, object]] = []
    skipped_large: List[Dict[str, object]] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        key = str(path)
        if key not in seen:
            seen.add(key)
            size_bytes = path.stat().st_size
            size_mb = round(size_bytes / (1024**2), 3)
            meta = {"path": path, "size_mb": size_mb}
            if max_size_mb is not None and size_mb > max_size_mb:
                skipped_large.append(meta)
                continue
            dataset_meta.append(meta)
    if not dataset_meta:
        from Project.utils.io import find_csv  # lazy import to avoid circulars

        fallback = Path(find_csv()).resolve()
        dataset_meta.append({"path": fallback, "size_mb": round(fallback.stat().st_size / (1024**2), 3)})

    if prefer_small:
        dataset_meta.sort(key=lambda meta: float(meta["size_mb"]))

    selected = [meta["path"] for meta in dataset_meta[:max_datasets]]
    diagnostics = {
        "skipped_large": skipped_large,
        "candidates_considered": dataset_meta,
    }
    return selected, diagnostics


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
    parser.add_argument(
        "--prefer-small-datasets",
        action="store_true",
        default=_env_flag("RUN_ALL_PREFER_SMALL"),
        help="Prefer smaller CSVs first by sorting candidates by file size before selection.",
    )
    parser.add_argument(
        "--dataset-max-mb",
        type=float,
        default=float(os.getenv("RUN_ALL_DATASET_MAX_MB", "0") or 0),
        help="Skip datasets larger than this threshold (MB). 0 disables the filter.",
    )
    parser.add_argument(
        "--skip-global",
        action="store_true",
        default=_env_flag("RUN_ALL_SKIP_GLOBAL"),
        help="Skip global (vision/audio/timeseries) steps entirely.",
    )
    parser.add_argument(
        "--rerun-global",
        action="store_true",
        default=_env_flag("RUN_ALL_RERUN_GLOBAL"),
        help="Force running global steps even if their artifacts already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    size_cap = args.dataset_max_mb if args.dataset_max_mb > 0 else None
    runs_root = Path(abs_path("runs"))
    runs_root.mkdir(parents=True, exist_ok=True)

    datasets, dataset_diag = discover_datasets(
        max_datasets=args.max_datasets,
        prefer_small=args.prefer_small_datasets,
        max_size_mb=size_cap,
    )
    if dataset_diag.get("skipped_large"):
        print("[Info] Skipping datasets above --dataset-max-mb threshold:")
        for meta in dataset_diag["skipped_large"]:
            print(f"  - {meta['path']} ({meta['size_mb']} MB)")
    if not datasets:
        raise RuntimeError("No datasets available after applying filters.")
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

    if GLOBAL_STEPS and not args.skip_global:
        global_env = os.environ.copy()
        global_env["PYTHONPATH"] = REPO_ROOT
        global_env.setdefault("PYTHONHASHSEED", "42")
        global_env.setdefault("FLAML_TIME_BUDGET", str(args.flaml_time_budget))
        for step in GLOBAL_STEPS:
            if not args.rerun_global:
                outputs = GLOBAL_STEP_OUTPUTS.get(step, [])
                if outputs and all(path.exists() for path in outputs):
                    joined = ", ".join(str(path) for path in outputs)
                    print(f"[Skip] ({step}) Cached artifacts detected → {joined}")
                    continue
            entry = run_step(step, global_env, "global", timeout_val)
            orchestrator_runtime.append(entry)
    elif args.skip_global:
        print("[Info] Global steps skipped via --skip-global.")

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
    if dataset_infos:
        summary_line = ", ".join(
            f"{info['dataset']} (rows={info.get('rows')}, columns={info.get('columns')})" for info in dataset_infos
        )
        print(f"[Summary] {summary_line}")
    print(f"Reports root → {abs_path('reports')} | Runs archive → {abs_path('runs')}")


if __name__ == "__main__":
    main()
