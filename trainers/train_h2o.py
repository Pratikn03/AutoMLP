# Project/trainers/train_h2o.py
import os, sys
from pathlib import Path
from Project.common.utils import (
    resolve_csv_path, load_dataframe, detect_target, prepare_xy,
    compute_metrics, save_metrics_row, ensure_dirs
)

SEED = int(os.getenv("SEED", "42"))
TIME = int(os.getenv("H2O_TIME", "240"))

def main():
    ensure_dirs()
    try:
        import h2o
        from h2o.automl import H2OAutoML
    except Exception as e:
        print(f"[SKIP] H2O not available ({e}).")
        return

    df = load_dataframe(resolve_csv_path())
    target = os.getenv("TARGET") or detect_target(df)
    X, y = prepare_xy(df, target)

    # H2O expects a single frame with target
    import pandas as pd
    dfall = pd.concat([X, y.rename(target)], axis=1)
    h2o.init(nthreads=-1, max_mem_size="4G")
    hf = h2o.H2OFrame(dfall)
    # classification: make target factor
    if hf[target].isnumeric()[0]:
        # keep numeric for regression; here assume classification
        pass
    else:
        hf[target] = hf[target].asfactor()

    train, test = hf.split_frame(ratios=[0.8], seed=SEED)
    x = [c for c in hf.columns if c != target]
    aml = H2OAutoML(max_runtime_secs=TIME, seed=SEED, sort_metric="AUC", balance_classes=True)
    aml.train(x=x, y=target, training_frame=train)
    lb = aml.leaderboard.as_data_frame()

    # Evaluate (attempt AUC/AP for binary, else F1/accuracy using H2O)
    preds = aml.leader.predict(test)
    # Convert to pandas for our metrics (fallbacks included)
    import numpy as np
    y_true = test[target].as_data_frame().iloc[:,0]
    if "p1" in preds.columns:
        proba = preds["p1"].as_data_frame().iloc[:,0].values
        y_pred = (proba >= 0.5).astype(int)
    else:
        # multiclass: take predicted label column "predict"
        y_pred = preds["predict"].as_data_frame().iloc[:,0].values
        proba = None

    # Align types
    try:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
    except Exception:
        pass

    from Project.common.utils import compute_metrics
    metrics = compute_metrics(y_true, y_pred, proba)
    save_metrics_row("H2O_AutoML", metrics)

    # Save model
    model_path = h2o.save_model(aml.leader, path=str(Path("Project/artifacts").resolve()), force=True)
    (Path("Project/reports")/"h2o_leaderboard_top5.csv").write_text(lb.head(5).to_csv(index=False))

    print(f"H2O done. Saved leader model: {model_path}")

if __name__ == "__main__":
    main()
