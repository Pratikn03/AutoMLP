"""Lightweight H2O AutoML trainer that works with the repo's utils.

This file intentionally keeps dependencies small and uses `Project.utils.io`
and `Project.utils.sanitize` so it runs correctly when invoked from
`scripts/run_all.py` with PYTHONPATH set to the repo root.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from Project.utils.io import guess_target_column, load_dataset
from Project.utils.sanitize import sanitize_columns
from Project.utils.standardize import save_metrics

SEED = int(os.getenv("SEED", "42"))
MAX_RUNTIME = int(os.getenv("H2O_MAX_RUNTIME", "60"))


def main():
    try:
        import h2o
        from h2o.automl import H2OAutoML
    except Exception as e:
        print("[H2O] Not available → skipping. Reason:", e)
        return

    # Optional speedups
    try:
        import polars  # type: ignore  # noqa: F401
        import pyarrow  # type: ignore  # noqa: F401
    except Exception:
        warnings.warn("Optional packages 'polars' and 'pyarrow' not installed. Install them for faster H2O->pandas conversion.")

    df = load_dataset()
    df = sanitize_columns(df)
    target_col = guess_target_column(df, os.getenv("TARGET"))
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found after sanitization.")

    # Normalize binary target to 'Yes'/'No' so H2O treats it as categorical
    target_str = df[target_col].astype(str).str.strip()
    lowered = target_str.str.lower()
    yes_vals = {"yes", "1", "true", "y"}
    no_vals = {"no", "0", "false", "n"}
    if lowered.isin(yes_vals | no_vals).all():
        df[target_col] = np.where(lowered.isin(yes_vals), "Yes", "No")
    else:
        df[target_col] = target_str

    # Run H2O AutoML
    h2o.init(max_mem_size="2G")
    hf = h2o.H2OFrame(df)
    try:
        hf[target_col] = hf[target_col].asfactor()  # type: ignore[attr-defined]
    except Exception:
        # fallback: coerce to string/category then factorize
        hf[target_col] = hf[target_col].ascharacter()  # type: ignore[attr-defined]
        try:
            hf[target_col] = hf[target_col].asfactor()  # type: ignore[attr-defined]
        except Exception:
            pass

    aml = H2OAutoML(max_runtime_secs=MAX_RUNTIME, seed=SEED, exclude_algos=["DeepLearning"])
    train_frame, test_frame = hf.split_frame(ratios=[0.8], seed=SEED)
    aml.train(y=target_col, training_frame=train_frame, leaderboard_frame=test_frame)

    metrics = {
        "f1_macro": float("nan"),
        "accuracy": float("nan"),
        "roc_auc_ovr": float("nan"),
        "avg_precision_ovr": float("nan"),
    }

    try:
        preds = aml.leader.predict(test_frame)  # type: ignore[attr-defined]
        preds_df = preds.as_data_frame(use_pandas=True)
        truth_df = test_frame[target_col].as_data_frame(use_pandas=True)

        y_true = truth_df[target_col].astype(str).str.strip().str.lower()
        y_pred = preds_df["predict"].astype(str).str.strip().str.lower()

        if y_true.nunique() > 1:
            y_true_np = y_true.to_numpy()
            y_pred_np = y_pred.to_numpy()
            metrics["accuracy"] = float(accuracy_score(y_true_np, y_pred_np))
            metrics["f1_macro"] = float(f1_score(y_true_np, y_pred_np, average="macro"))

            prob_cols = [c for c in preds_df.columns if c not in {"predict", "decision", "probabilities"}]
            positive_label = "yes" if "yes" in set(y_true.unique()) else sorted(y_true.unique())[-1]
            prob_col = None
            for col in prob_cols:
                normalized = col.lower().lstrip("p")
                if normalized == positive_label:
                    prob_col = col
                    break
            if prob_col is None and prob_cols:
                prob_col = prob_cols[-1]

            if prob_col is not None:
                y_prob = preds_df[prob_col].astype(float).to_numpy()
                y_true_bin = (y_true == positive_label).astype(int).to_numpy()
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_true_bin, y_prob))
                metrics["avg_precision_ovr"] = float(average_precision_score(y_true_bin, y_prob))
    except Exception as metric_err:
        warnings.warn(f"[H2O] Metric computation failed: {metric_err}")

    lb_raw = aml.leaderboard.as_data_frame(use_pandas=True)  # type: ignore[attr-defined]
    lb = pd.DataFrame(lb_raw)
    if not lb.empty:
        if "auc" in lb.columns and np.isnan(metrics["roc_auc_ovr"]):
            metrics["roc_auc_ovr"] = float(pd.to_numeric(lb.loc[0, "auc"]))
        if "pr_auc" in lb.columns and np.isnan(metrics["avg_precision_ovr"]):
            metrics["avg_precision_ovr"] = float(pd.to_numeric(lb.loc[0, "pr_auc"]))

    row = pd.DataFrame([
        {
            "framework": "H2O_AutoML",
            "f1_macro": metrics["f1_macro"],
            "accuracy": metrics["accuracy"],
            "roc_auc_ovr": metrics["roc_auc_ovr"],
            "avg_precision_ovr": metrics["avg_precision_ovr"],
        }
    ])
    save_metrics(row, "H2O_AutoML")
    print(f"H2O AutoML done for target '{target_col}' → reports/leaderboard.csv")

    # Persist the leader model so the API can load it later.
    art_dir = os.path.join("Project", "artifacts", "h2o")
    os.makedirs(art_dir, exist_ok=True)
    try:
        # Save a binary H2O model (loadable by h2o.load_model)
        model_path = h2o.save_model(aml.leader, path=art_dir, force=True)
        print(f"Saved H2O model: {model_path}")
    except Exception as e:
        print("[H2O] Could not save binary model:", e)
    try:
        # Also save a MOJO for production scoring (leader.download_mojo may exist)
        mojo_path = aml.leader.download_mojo(path=art_dir)  # type: ignore[attr-defined]
        print(f"Saved MOJO: {mojo_path}")
    except Exception as e:
        # fallback: some H2O versions expose download_mojo differently
        try:
            mojo_path = aml.leader.get_mojo_path()  # type: ignore[attr-defined]
            print(f"MOJO available at: {mojo_path}")
        except Exception:
            print("[H2O] Could not save MOJO:", e)

    # Shutdown cleanly
    try:
        cl = getattr(h2o, "cluster", None)
        if callable(cl):
            c = h2o.cluster()
            if c is not None and hasattr(c, "shutdown"):
                c.shutdown(prompt=False)
            elif hasattr(h2o, "shutdown"):
                h2o.shutdown(prompt=False)
        else:
            if hasattr(h2o, "shutdown"):
                h2o.shutdown(prompt=False)
    except Exception as e:
        print("[H2O] Warning: failed to shutdown cleanly:", e)


if __name__ == "__main__":
    main()
