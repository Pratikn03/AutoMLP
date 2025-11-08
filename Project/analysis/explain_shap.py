"""Compute SHAP & LIME explanations for persisted experiment pipelines."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from Project.utils.io import load_dataset
from Project.utils.sanitize import sanitize_columns
from Project.utils.standardize import ensure_directories
from Project.utils.system import capture_resource_snapshot, merge_runtime_sections

try:  # optional dependency guard (SHAP)
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional
    shap = None  # type: ignore

try:  # optional dependency guard (LIME)
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except Exception:  # pragma: no cover - optional
    LimeTabularExplainer = None  # type: ignore

try:
    import joblib
except Exception:  # pragma: no cover - defensive
    joblib = None


EXPERIMENT_ARTIFACTS = Path("artifacts/experiments")
METRICS_DIR = Path("reports/metrics")
FIG_DIR = Path("figures/shap")
GLOBAL_SUMMARY_PATH = Path("reports/shap_global_summary.csv")
SAMPLE_SUMMARY_PATH = Path("reports/shap_sample_details.csv")
LIME_DETAILS_PATH = Path("reports/lime_sample_details.json")

TARGET_ENV = os.getenv("TARGET")
MAX_GLOBAL_SAMPLES = int(os.getenv("SHAP_GLOBAL_SAMPLES", "400"))
MAX_SAMPLE_POINTS = int(os.getenv("SHAP_SAMPLE_POINTS", "5"))
MAX_BACKGROUND = int(os.getenv("SHAP_BACKGROUND_SIZE", "200"))
LIME_SAMPLES = int(os.getenv("LIME_SAMPLE_POINTS", "5"))


@dataclass
class PipelineDescriptor:
    experiment: str
    seed: int
    fold: int
    path: Path
    config: Dict[str, Any]


def iter_pipelines(base: Path = EXPERIMENT_ARTIFACTS) -> Iterable[PipelineDescriptor]:
    if not base.exists() or joblib is None:
        return
    for experiment_dir in sorted(base.glob("*")):
        if not experiment_dir.is_dir():
            continue
        config_path = METRICS_DIR / f"{experiment_dir.name}_config.json"
        config = {}
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text())
            except Exception:
                config = {}
        for seed_dir in sorted(experiment_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            for pipeline_path in sorted(seed_dir.glob("fold_*_pipeline.joblib")):
                try:
                    fold_str = pipeline_path.stem.split("_")[-1]
                    fold_idx = int(fold_str)
                except Exception:
                    fold_idx = -1
                try:
                    seed_idx = int(seed_dir.name.split("_")[-1])
                except Exception:
                    seed_idx = -1
                yield PipelineDescriptor(
                    experiment=experiment_dir.name,
                    seed=seed_idx,
                    fold=fold_idx,
                    path=pipeline_path,
                    config=config,
                )


def safe_feature_names(preprocessor: Optional[Any]) -> np.ndarray:
    if preprocessor is None:
        return np.array([])
    try:
        names = preprocessor.get_feature_names_out()
        if isinstance(names, np.ndarray):
            return names
        return np.asarray(names)
    except Exception:
        pass
    if hasattr(preprocessor, "named_steps") and "column_transform" in preprocessor.named_steps:
        column_transform = preprocessor.named_steps["column_transform"]
        try:
            names = column_transform.get_feature_names_out()
            if isinstance(names, np.ndarray):
                return names
            return np.asarray(names)
        except Exception:
            return np.array([])
    return np.array([])


def to_dataframe(data: Any, feature_names: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(data)
    if feature_names.size == frame.shape[1]:
        frame.columns = feature_names
    return frame


def compute_tree_shap(estimator: Any, X_data: np.ndarray):
    if shap is None:
        raise RuntimeError("SHAP not installed")
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_data)
    return shap_values, explainer.expected_value


def compute_kernel_shap(predict_fn, background: np.ndarray, samples: np.ndarray):
    if shap is None:
        raise RuntimeError("SHAP not installed")
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(samples, nsamples=min(100, background.shape[0]))
    return shap_values, explainer.expected_value


def compute_lime(predict_fn, training_data: np.ndarray, sample: np.ndarray, feature_names: List[str], class_names: List[str]) -> List[Tuple[str, float]]:
    if LimeTabularExplainer is None:
        raise RuntimeError("LIME not installed")
    explainer = LimeTabularExplainer(
        training_data,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )
    explanation = explainer.explain_instance(sample, predict_fn, num_features=min(10, len(feature_names)))
    return explanation.as_list()


def load_data(target: Optional[str]) -> Tuple[pd.DataFrame, pd.Series, str]:
    data = load_dataset()
    data = sanitize_columns(data)
    if target is None:
        from Project.utils.io import guess_target_column

        target = guess_target_column(data)
    if target not in data.columns:
        raise KeyError(f"Target column '{target}' not found in dataset")
    y = data[target]
    X = data.drop(columns=[target])
    return X, y, target


def ensure_numpy(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def main() -> None:
    ensure_directories()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_raw, y_raw, target = load_data(TARGET_ENV)
    dataset = pd.concat([X_raw, y_raw], axis=1)
    dataset = dataset.reset_index(drop=True)

    baselines = {}
    descriptors = list(iter_pipelines())
    if joblib is None:
        print("[Explain] joblib not available; cannot load persisted pipelines.")
        return
    if not descriptors:
        empty_global = pd.DataFrame(columns=["feature", "mean_abs_shap", "experiment", "seed", "fold"])
        empty_samples = pd.DataFrame(columns=["experiment", "seed", "fold", "sample_index", "prediction", "positive_class", "top_contributions", "expected_value"])
        GLOBAL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SAMPLE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        empty_global.to_csv(GLOBAL_SUMMARY_PATH, index=False)
        empty_samples.to_csv(SAMPLE_SUMMARY_PATH, index=False)
        print("[Explain] No stored pipelines found; run training before explainability.")
        return

    for desc in descriptors:
        if not desc.path.exists():
            print(f"[Explain] Skipping missing pipeline artifact: {desc.path}")
            continue
        try:
            pipeline = joblib.load(desc.path)
        except Exception as exc:
            print(f"[Explain] Failed to load pipeline {desc.path}: {exc}")
            continue

        if hasattr(pipeline, "named_steps"):
            estimator = pipeline.named_steps.get("clf")
            preprocessor = pipeline.named_steps.get("pre")
        else:
            estimator = pipeline
            preprocessor = None

        if estimator is None:
            print(f"[Explain] Missing estimator for {desc.path}")
            continue

        start_resources = capture_resource_snapshot()
        start_time = time.perf_counter()

        sample = dataset.sample(
            n=min(MAX_GLOBAL_SAMPLES, len(dataset)),
            random_state=desc.seed if desc.seed >= 0 else 42,
        )
        X_sample = sample.drop(columns=[target])
        y_sample = sample[target]
        raw_columns = X_sample.columns.tolist()
        raw_training = X_sample.to_numpy()

        # Keep an aligned copy for per-row predictions
        X_sample_ordered = X_sample.reset_index(drop=False)

        try:
            if preprocessor is not None:
                transformed = preprocessor.transform(X_sample)
            else:
                transformed = X_sample.to_numpy()
            transformed = ensure_numpy(transformed)
        except Exception as exc:
            print(f"[Explain] Failed to transform data for {desc.experiment}: {exc}")
            continue

        feature_names = safe_feature_names(preprocessor)
        if feature_names.size == 0 and transformed.shape[1] == X_sample.shape[1]:
            feature_names = X_sample.columns.to_numpy()
        elif feature_names.size == 0:
            feature_names = np.array([f"feature_{i}" for i in range(transformed.shape[1])])

        shap_values = None
        shap_expected = None
        shap_duration = None
        shap_runtime_error: Optional[str] = None
        shap_data_matrix = transformed
        shap_feature_names = feature_names

        if shap is not None:
            try:
                if hasattr(estimator, "predict_proba") and hasattr(estimator, "tree_structure"):
                    shap_values, shap_expected = compute_tree_shap(estimator, transformed)
                elif estimator.__class__.__name__.lower().startswith("catboost") or hasattr(estimator, "get_booster") or hasattr(estimator, "estimators_"):
                    shap_values, shap_expected = compute_tree_shap(estimator, transformed)
                else:
                    background_raw = X_sample_ordered[raw_columns].to_numpy()
                    samples_raw = background_raw

                    def predict_fn(raw_data: np.ndarray) -> np.ndarray:
                        frame = pd.DataFrame(raw_data, columns=raw_columns)
                        return pipeline.predict_proba(frame)[:, 1]

                    shap_values, shap_expected = compute_kernel_shap(predict_fn, background_raw, samples_raw)
                    shap_data_matrix = samples_raw
                    shap_feature_names = np.array(raw_columns)
                shap_duration = time.perf_counter() - start_time
            except Exception as exc:
                shap_runtime_error = str(exc)
                shap_values = None
                shap_expected = None
        else:
            shap_runtime_error = "shap_not_installed"

        lime_duration = None
        lime_payload: List[Dict[str, Any]] = []
        if shap_duration is not None:
            # Continue timer for LIME only after SHAP finished to avoid double counting
            start_lime = time.perf_counter()
        else:
            start_lime = time.perf_counter()

        if LimeTabularExplainer is not None:
            try:
                background = raw_training[: min(1000, raw_training.shape[0])]
                class_names = [str(cls) for cls in np.unique(y_sample)]

                def lime_predict_fn(data: np.ndarray) -> np.ndarray:
                    frame = pd.DataFrame(data, columns=raw_columns)
                    return pipeline.predict_proba(frame)

                for idx in range(min(LIME_SAMPLES, background.shape[0])):
                    explanation = compute_lime(
                        lime_predict_fn,
                        background,
                        background[idx],
                        raw_columns,
                        class_names,
                    )
                    lime_payload.append({
                        "experiment": desc.experiment,
                        "seed": desc.seed,
                        "fold": desc.fold,
                        "sample_index": int(sample.index[idx]),
                        "attribution": explanation,
                    })
                lime_duration = time.perf_counter() - start_lime
            except Exception as exc:
                lime_duration = None
                if shap_runtime_error is None:
                    shap_runtime_error = f"LIME failed: {exc}"
        else:
            if shap_runtime_error is None:
                shap_runtime_error = "lime_not_installed"

        runtime_entry: Dict[str, Any] = {
            "experiment": desc.experiment,
            "seed": desc.seed,
            "fold": desc.fold,
            "pipeline_path": str(desc.path),
            "target": target,
            "model_size_bytes": desc.path.stat().st_size if desc.path.exists() else None,
            "shap_time_sec": shap_duration,
            "lime_time_sec": lime_duration,
            "resource_before": start_resources,
            "resource_after": capture_resource_snapshot(),
            "shap_error": shap_runtime_error,
        }

        baselines.setdefault("runtime", []).append(runtime_entry)
        if lime_payload:
            baselines.setdefault("lime", []).extend(lime_payload)

        if shap_values is None:
            continue

        if isinstance(shap_values, list):
            shap_values_array = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        else:
            shap_values_array = np.asarray(shap_values)

        shap_df = pd.DataFrame({
            "feature": shap_feature_names,
            "mean_abs_shap": np.mean(np.abs(shap_values_array), axis=0),
            "experiment": desc.experiment,
            "seed": desc.seed,
            "fold": desc.fold,
        })

        baselines.setdefault("global", []).append(shap_df)

        top_samples = min(MAX_SAMPLE_POINTS, shap_values_array.shape[0])
        for idx in range(top_samples):
            row = X_sample_ordered.iloc[idx]
            raw_row = pd.DataFrame([row[raw_columns].values], columns=raw_columns)
            sample_idx = int(row["index"]) if "index" in row else int(idx)
            instance_values = shap_values_array[idx]
            contributions = sorted(
                zip(shap_feature_names.tolist(), instance_values.tolist()),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:10]
            baselines.setdefault("samples", []).append({
                "experiment": desc.experiment,
                "seed": desc.seed,
                "fold": desc.fold,
                "sample_index": sample_idx,
                "prediction": float(pipeline.predict_proba(raw_row)[0, 1]) if hasattr(pipeline, "predict_proba") else float(pipeline.predict(raw_row)[0]),
                "positive_class": str(np.unique(y_sample)[-1]) if y_sample.nunique() else "1",
                "top_contributions": contributions,
                "expected_value": shap_expected if np.isscalar(shap_expected) else None,
            })

        # Save SHAP summary plot per experiment (aggregate by fold only once)
        plot_path = FIG_DIR / f"{desc.experiment}_seed{desc.seed}_fold{desc.fold}_shap_summary.png"
        if shap is not None:
            try:
                shap.summary_plot(
                    shap_values_array,
                    shap_data_matrix,
                    feature_names=shap_feature_names,
                    show=False,
                )
                import matplotlib.pyplot as plt

                plt.tight_layout()
                plt.savefig(plot_path, dpi=320, bbox_inches="tight")
                plt.close()
            except Exception as exc:
                print(f"[Explain] Failed to render SHAP plot for {desc.experiment}: {exc}")

    global_frames = baselines.get("global", [])
    if global_frames:
        global_df = pd.concat(global_frames, ignore_index=True)
        global_df.to_csv(GLOBAL_SUMMARY_PATH, index=False)
    else:
        empty_global = pd.DataFrame(columns=["feature", "mean_abs_shap", "experiment", "seed", "fold"])
        empty_global.to_csv(GLOBAL_SUMMARY_PATH, index=False)

    sample_entries = baselines.get("samples", [])
    if sample_entries:
        sample_df = pd.DataFrame(sample_entries)
        sample_df.to_csv(SAMPLE_SUMMARY_PATH, index=False)
    else:
        empty_samples = pd.DataFrame(columns=["experiment", "seed", "fold", "sample_index", "prediction", "positive_class", "top_contributions", "expected_value"])
        empty_samples.to_csv(SAMPLE_SUMMARY_PATH, index=False)

    lime_entries = baselines.get("lime", [])
    if lime_entries:
        LIME_DETAILS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LIME_DETAILS_PATH.write_text(json.dumps(lime_entries, indent=2))
    elif LIME_DETAILS_PATH.exists():
        LIME_DETAILS_PATH.unlink()

    runtime_entries = baselines.get("runtime", [])
    if runtime_entries:
        merge_runtime_sections({"explainability": runtime_entries})

    print("Explainability analysis complete.")


if __name__ == "__main__":
    main()
