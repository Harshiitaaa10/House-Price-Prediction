"""House Price Prediction website.

Run:  python app.py        (or: bash run_web.sh)
Then open the address printed in the terminal (it opens automatically).
"""
import json
import socket
import threading
import webbrowser

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.pipeline import MODEL_DIR, OUT_DIR, clean_features, load_data, split_xy

app = Flask(__name__)

# ---------------------------------------------------------------- model + defaults
MODEL_PATH = MODEL_DIR / "gbr.pkl"
if not MODEL_PATH.exists():
    raise SystemExit("Model not found. Train it first:  python -m src.train --skip-cv")
MODEL = joblib.load(MODEL_PATH)

_train, _test = load_data()
X_TRAIN, _y, _xt, _ids, _dropped = split_xy(_train, _test)
COLUMNS = list(MODEL.named_steps["prep"].feature_names_in_)

# a "typical house": median for numbers, most common value for categories
DEFAULTS = {}
for c in COLUMNS:
    col = X_TRAIN[c]
    if pd.api.types.is_numeric_dtype(col):
        DEFAULTS[c] = float(col.median())
    else:
        DEFAULTS[c] = col.mode(dropna=True).iloc[0]

NEIGHBORHOODS = sorted(X_TRAIN["Neighborhood"].dropna().unique())
GARAGE_AREA_PER_CAR = (_train.groupby("GarageCars")["GarageArea"].median().to_dict())
METRICS = {"model": "GradientBoosting", "r2": None, "rmse": None, "rmse_log": 0.12}
if (OUT_DIR / "metrics.json").exists():
    METRICS.update(json.loads((OUT_DIR / "metrics.json").read_text()))

NEIGHBORHOOD_NAMES = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale",
    "BrkSide": "Brookside", "ClearCr": "Clear Creek", "CollgCr": "College Creek",
    "Crawfor": "Crawford", "Edwards": "Edwards", "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT & Rail Road", "MeadowV": "Meadow Village", "Mitchel": "Mitchell",
    "NAmes": "North Ames", "NoRidge": "Northridge", "NPkVill": "Northpark Villa",
    "NridgHt": "Northridge Heights", "NWAmes": "Northwest Ames", "OldTown": "Old Town",
    "SWISU": "South & West of ISU", "Sawyer": "Sawyer", "SawyerW": "Sawyer West",
    "Somerst": "Somerset", "StoneBr": "Stone Brook", "Timber": "Timberland", "Veenker": "Veenker",
}

QUALITY = ["Ex", "Gd", "TA", "Fa", "Po"]
BLDG = {"1Fam": "Single-family", "2fmCon": "Two-family conversion", "Duplex": "Duplex",
        "TwnhsE": "Townhouse (end unit)", "Twnhs": "Townhouse (inner unit)"}


class InputError(ValueError):
    pass


def num(data, key, lo, hi, label, integer=True):
    raw = data.get(key, "")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise InputError(f"{label}: please enter a number")
    if not (lo <= v <= hi):
        raise InputError(f"{label} must be between {lo:g} and {hi:g}")
    return int(round(v)) if integer else v


def choice(data, key, allowed, label):
    v = data.get(key)
    if v not in allowed:
        raise InputError(f"{label}: invalid choice")
    return v


def build_row(d: dict) -> pd.DataFrame:
    """Turn the form answers into a full 75-column row (rest = typical house)."""
    row = dict(DEFAULTS)

    first = num(d, "FirstFlr", 300, 5000, "1st floor area")
    second = num(d, "SecondFlr", 0, 3000, "2nd floor area")
    bsmt = num(d, "TotalBsmtSF", 0, 4000, "Basement area")
    lot = num(d, "LotArea", 1000, 100000, "Lot size")
    built = num(d, "YearBuilt", 1872, 2010, "Year built")
    remod = num(d, "YearRemodAdd", 1872, 2010, "Year remodelled")
    # dataset convention: "remodel year" = build year if never remodelled, and 1950 for older houses
    remod = max(remod, built, 1950)
    beds = num(d, "BedroomAbvGr", 0, 8, "Bedrooms")
    baths = num(d, "FullBath", 0, 4, "Full bathrooms")
    half = num(d, "HalfBath", 0, 2, "Half bathrooms")
    cars = num(d, "GarageCars", 0, 4, "Garage capacity")
    fire = num(d, "Fireplaces", 0, 4, "Fireplaces")
    qual = num(d, "OverallQual", 1, 10, "Overall quality")
    cond = num(d, "OverallCond", 1, 10, "Overall condition")

    row.update({
        "Neighborhood": choice(d, "Neighborhood", NEIGHBORHOODS, "Neighborhood"),
        "BldgType": choice(d, "BldgType", list(BLDG), "Building type"),
        "KitchenQual": choice(d, "KitchenQual", QUALITY[:4], "Kitchen quality"),
        "ExterQual": choice(d, "ExterQual", QUALITY[:4], "Exterior quality"),
        "CentralAir": choice(d, "CentralAir", ["Y", "N"], "Central air"),
        "1stFlrSF": first, "2ndFlrSF": second, "GrLivArea": first + second, "LowQualFinSF": 0,
        "LotArea": lot, "YearBuilt": built, "YearRemodAdd": remod,
        "BedroomAbvGr": beds, "FullBath": baths, "HalfBath": half,
        "TotRmsAbvGrd": beds + baths + 2, "Fireplaces": fire,
        "OverallQual": qual, "OverallCond": cond, "GarageCars": cars,
        "HouseStyle": "2Story" if second > 0 else "1Story",
        "MSSubClass": "60" if second > 0 else "20",
    })

    # basement: keep sizes consistent, or mark "no basement"
    if bsmt == 0:
        row.update({"TotalBsmtSF": 0, "BsmtFinSF1": 0, "BsmtFinSF2": 0, "BsmtUnfSF": 0,
                    "BsmtFullBath": 0, "BsmtHalfBath": 0})
        for c in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
            row[c] = np.nan
    else:
        fin = round(bsmt * 0.5)
        row.update({"TotalBsmtSF": bsmt, "BsmtFinSF1": fin, "BsmtFinSF2": 0, "BsmtUnfSF": bsmt - fin})

    # garage
    if cars == 0:
        row.update({"GarageArea": 0, "GarageYrBlt": np.nan})
        for c in ("GarageType", "GarageFinish", "GarageQual", "GarageCond"):
            row[c] = np.nan
    else:
        row["GarageArea"] = float(GARAGE_AREA_PER_CAR.get(cars, 250 * cars))
        row["GarageYrBlt"] = built

    if fire == 0:
        row["FireplaceQu"] = np.nan

    df = pd.DataFrame([row])[COLUMNS]
    return clean_features(df)


# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    hood_options = [(h, NEIGHBORHOOD_NAMES.get(h, h)) for h in NEIGHBORHOODS]
    hood_options.sort(key=lambda t: t[1])
    return render_template("index.html", hoods=hood_options, bldg=BLDG, metrics=METRICS,
                           default_hood="NAmes" if "NAmes" in NEIGHBORHOODS else NEIGHBORHOODS[0])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    try:
        row = build_row(data)
    except InputError as e:
        return jsonify(error=str(e)), 400
    price = float(MODEL.predict(row)[0])
    spread = float(np.exp(METRICS.get("rmse_log") or 0.12))
    return jsonify(price=round(price), low=round(price / spread), high=round(price * spread))


def _free_port(start=8000):
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


if __name__ == "__main__":
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"\n  House Price website running at  {url}\n  (press Ctrl+C to stop)\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False)
