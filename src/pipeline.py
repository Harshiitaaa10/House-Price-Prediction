"""Shared preprocessing + model definitions for the House Prices project."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:  # XGBoost is optional so the project still runs without it
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover  (e.g. missing libomp on Mac)
    XGBRegressor = None

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
TARGET = "SalePrice"
RANDOM_STATE = 42
MISSING_THRESHOLD = 0.80  # drop columns with more than 80% missing values


def _find_file(name: str, roots):
    for root in roots:
        root = Path(root).expanduser()
        if root.is_dir():
            hits = sorted(root.rglob(name))
            if hits:
                return hits[0]
    return None


def ensure_data(data_dir: Path = DATA_DIR) -> None:
    """Make sure data/train.csv and data/test.csv exist.

    If they are missing, look for the Kaggle zip (in data/, the project folder or
    ~/Downloads), or loose CSVs in ~/Downloads, and set them up automatically.
    """
    data_dir.mkdir(exist_ok=True)
    if (data_dir / "train.csv").exists() and (data_dir / "test.csv").exists():
        return
    import shutil
    import zipfile

    # 1) a Kaggle zip in data/, project root or Downloads
    for folder in (data_dir, ROOT, Path.home() / "Downloads"):
        if not folder.is_dir():
            continue
        for z in sorted(folder.glob("house-prices*.zip")):
            try:
                with zipfile.ZipFile(z) as zf:
                    names = [n for n in zf.namelist() if n.endswith(("train.csv", "test.csv"))]
                    if not names:
                        continue
                    for n in names:
                        with zf.open(n) as src, open(data_dir / Path(n).name, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                print(f"Extracted train.csv/test.csv from {z}")
            except zipfile.BadZipFile:
                continue
            if (data_dir / "train.csv").exists() and (data_dir / "test.csv").exists():
                return
    # 2) loose CSVs already extracted somewhere (data/ subfolders or Downloads)
    for name in ("train.csv", "test.csv"):
        if not (data_dir / name).exists():
            hit = _find_file(name, [data_dir, Path.home() / "Downloads" / "house-prices-advanced-regression-techniques"])
            if hit:
                shutil.copy(hit, data_dir / name)
                print(f"Copied {hit} -> {data_dir / name}")


def load_data(data_dir: Path = DATA_DIR):
    ensure_data(data_dir)
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(
            "\nDATA MISSING: train.csv and test.csv were not found.\n"
            "1. Open https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data\n"
            "2. Log in, click 'Join Competition', then 'Download All'.\n"
            "3. Leave the downloaded zip in your Downloads folder (or put it in the data/ folder)\n"
            "   and run this command again. It will be unzipped automatically.\n"
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning applied identically to train, test and new data."""
    X = X.copy()
    # MSSubClass is a building-type code (20, 30, 60...), not a quantity -> treat as category
    if "MSSubClass" in X.columns:
        X["MSSubClass"] = X["MSSubClass"].astype(str)
    return X


def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Drop the 2 well-known outliers in this dataset (very large houses sold very cheaply).
    They are documented by the dataset author and badly distort every model."""
    mask = (train["GrLivArea"] > 4000) & (train[TARGET] < 300000)
    return train[~mask]


def split_xy(train: pd.DataFrame, test: pd.DataFrame):
    """Return X, y, X_test (Id removed), the test Ids and the dropped columns."""
    test_ids = test["Id"].copy()
    train = remove_outliers(train)
    X = train.drop(columns=[TARGET, "Id"])
    y = train[TARGET]
    X_test = test.drop(columns=["Id"])
    # Drop columns that are mostly empty (PoolQC, MiscFeature, Alley, Fence...)
    mostly_missing = X.columns[X.isna().mean() > MISSING_THRESHOLD].tolist()
    X = clean_features(X.drop(columns=mostly_missing))
    X_test = clean_features(X_test.drop(columns=mostly_missing, errors="ignore"))
    return X, y, X_test, test_ids, mostly_missing


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute + standardise numeric columns, impute + one-hot encode categoricals."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01)),
    ])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])


def get_models() -> dict:
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=10),
        "SVR": SVR(kernel="rbf", C=3, epsilon=0.05),
        "SGDRegressor": SGDRegressor(random_state=RANDOM_STATE, max_iter=2000, tol=1e-4),
        "KNeighbors": KNeighborsRegressor(n_neighbors=5),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=RANDOM_STATE),
        # neural nets need a standardised target to train well
        "MLP": TransformedTargetRegressor(
            regressor=MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=3000,
                                   random_state=RANDOM_STATE),
            transformer=StandardScaler()),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)
    return models


def make_pipeline(model, X: pd.DataFrame, log_target: bool = True) -> Pipeline:
    """Full pipeline: preprocessing -> (optionally log-target) model."""
    reg = (TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
           if log_target else model)
    return Pipeline([("prep", build_preprocessor(X)), ("model", reg)])
