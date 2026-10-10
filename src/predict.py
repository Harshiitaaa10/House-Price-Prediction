"""Predict with a saved model.

Usage:
    python -m src.predict
    python -m src.predict --model models/gbr.pkl --input data/test.csv --output outputs/new_preds.csv
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.pipeline import DATA_DIR, MODEL_DIR, OUT_DIR, clean_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_DIR / "gbr.pkl"))
    ap.add_argument("--input", default=str(DATA_DIR / "test.csv"))
    ap.add_argument("--output", default=str(OUT_DIR / "predictions.csv"))
    a = ap.parse_args()

    if not Path(a.model).exists():
        raise SystemExit(f"Model file {a.model} not found. Train first:  python -m src.train")
    pipe = joblib.load(a.model)
    df = pd.read_csv(a.input)
    ids = df["Id"] if "Id" in df else pd.Series(range(len(df)), name="Id")
    X = df.drop(columns=[c for c in ("Id", "SalePrice") if c in df])
    X = clean_features(X)
    expected = pipe.named_steps["prep"].feature_names_in_
    X = X.reindex(columns=expected)  # columns dropped in training are ignored
    out = pd.DataFrame({"Id": ids, "SalePrice": pipe.predict(X)})
    out.to_csv(a.output, index=False)
    print(f"Wrote {len(out)} predictions -> {a.output}")


if __name__ == "__main__":
    main()
