"""Train, compare and save models, then write submission.csv.

Usage (from project root):
    python -m src.train                       # compare all models, train GradientBoosting, predict
    python -m src.train --final-model best    # use whichever model scores best in CV
    python -m src.train --skip-cv             # skip the model comparison (faster)
"""
import argparse
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.pipeline import (MODEL_DIR, OUT_DIR, RANDOM_STATE, get_models,
                          load_data, make_pipeline, split_xy)

FINAL_KEY = {"gbr": "GradientBoosting", "xgb": "XGBoost", "rf": "RandomForest"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-model", default="gbr",
                    help="gbr (default), xgb, rf, or best (highest CV R2)")
    ap.add_argument("--skip-cv", action="store_true", help="skip comparing all models")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-log-target", action="store_true",
                    help="train on raw SalePrice instead of log1p(SalePrice)")
    args = ap.parse_args()
    log_target = not args.no_log_target

    MODEL_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    train, test = load_data()
    X, y, X_test, test_ids, dropped = split_xy(train, test)
    print(f"Train: {X.shape}, Test: {X_test.shape}, dropped mostly-empty columns: {dropped}")

    final_name = FINAL_KEY.get(args.final_model.lower())

    # ---- 1. Cross-validated comparison ------------------------------------
    cv = KFold(n_splits=args.folds, shuffle=True, random_state=RANDOM_STATE)
    scores = {}
    if not args.skip_cv or args.final_model == "best":
        for name, model in get_models().items():
            pipe = make_pipeline(model, X, log_target)
            s = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
            scores[name] = s.mean()
            print(f"{name:<18} CV R2 = {s.mean():.4f} (+/- {s.std():.4f})")
        res = pd.Series(scores).sort_values()
        res.to_csv(OUT_DIR / "cv_scores.csv", header=["cv_r2"])
        ax = res.plot.barh(figsize=(8, 5), title="Cross-validated R2 by model")
        ax.set_xlim(max(0, res.min() - 0.05), 1)
        plt.tight_layout(); plt.savefig(OUT_DIR / "model_comparison.png", dpi=120); plt.close()

    if args.final_model == "best":
        final_name = max(scores, key=scores.get)
    if final_name is None or final_name not in get_models():
        raise SystemExit(f"Unknown/unavailable --final-model {args.final_model!r}. "
                         f"Choose from {list(FINAL_KEY)} or best.")
    print(f"\nFinal model: {final_name}")

    # ---- 2. Hold-out check -------------------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pipe = make_pipeline(get_models()[final_name], X_tr, log_target).fit(X_tr, y_tr)
    pred = pipe.predict(X_val)
    r2 = r2_score(y_val, pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
    rmse_log = float(np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(pred))))
    print(f"Hold-out R2 = {r2:.4f} | RMSE = {rmse:,.0f}")
    (OUT_DIR / "metrics.json").write_text(json.dumps(
        {"model": final_name, "r2": round(float(r2), 4), "rmse": round(rmse), "rmse_log": round(rmse_log, 4)}))
    plt.figure(figsize=(5, 5))
    plt.scatter(y_val, pred, s=10, alpha=0.6)
    lim = [y_val.min(), y_val.max()]
    plt.plot(lim, lim, "r--"); plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title(f"{final_name}: actual vs predicted (hold-out)")
    plt.tight_layout(); plt.savefig(OUT_DIR / "actual_vs_predicted.png", dpi=120); plt.close()

    # ---- 3. Fit on all training data, save model, predict ------------------
    final = make_pipeline(get_models()[final_name], X, log_target).fit(X, y)
    model_file = MODEL_DIR / ("gbr.pkl" if final_name == "GradientBoosting"
                              else f"{final_name.lower()}.pkl")
    joblib.dump(final, model_file)
    print(f"Saved model -> {model_file}")

    preds = final.predict(X_test)
    sub = pd.DataFrame({"Id": test_ids, "SalePrice": preds})
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"Saved submission -> {OUT_DIR / 'submission.csv'} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
