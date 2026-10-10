# House Price Prediction (AI/ML) - with website

## Run the website (one command)
```bash
cd ~/Downloads/house-price-website
bash run_web.sh          # Windows: run_web.bat
```
The first run installs packages and trains the model (a few minutes). After that the website opens in your browser automatically (address like http://127.0.0.1:8000). Press `Ctrl+C` in the terminal to stop it.

Fill in the house details (neighbourhood, size, rooms, year, quality...) and the estimated price updates instantly. Prices are USD for Ames, Iowa, as in the Kaggle dataset. Anything you don't enter is set to a typical house.

---

Predict house prices with the Kaggle competition dataset
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
Final model: **GradientBoostingRegressor** (the original project reported R² ≈ 0.87; your exact score is printed when you run it).

## Project structure
```
house-price-prediction/
├── data/                      <- put train.csv and test.csv here
├── src/
│   ├── pipeline.py            preprocessing + model definitions
│   ├── train.py               compare models, train, save model + submission
│   └── predict.py             predict with a saved model
├── models/                    gbr.pkl is written here
├── outputs/                   submission.csv + plots are written here
├── app.py                     the website (Flask)
├── templates/index.html       the website page
├── run_web.sh / run_web.bat   one-command website launcher
├── house_price_prediction.ipynb   EDA + full workflow notebook
├── run.sh / run.bat           one-command setup + training
└── requirements.txt
```

> `data/` already contains the Kaggle files, and `outputs/` has a ready `submission.csv`. `models/gbr.pkl` is created the first time you train (it is not shipped because pickles depend on your scikit-learn version).

## Quick start (easiest)
1. Open this folder in VS Code, open a terminal (`` Ctrl+` ``) and run:
   ```bash
   bash run.sh          # Mac / Linux
   run.bat              # Windows
   ```
2. This creates a virtual environment, installs the packages, trains the models and writes the submission.

## Manual setup
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
```
Then in VS Code: `Cmd/Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`.

## Run it
**Notebook:** open `house_price_prediction.ipynb`, choose the `.venv` kernel, click **Run All**.

**Command line:**
```bash
python -m src.train                      # compare all models, train GBR, write submission
python -m src.train --final-model best   # use the best model from cross-validation
python -m src.train --skip-cv            # quicker: skip the comparison
python -m src.predict                    # re-predict test.csv from models/gbr.pkl
```
Outputs: `models/gbr.pkl`, `outputs/submission.csv`, `outputs/cv_scores.csv`, and two PNG plots.
Upload `outputs/submission.csv` to Kaggle to get a leaderboard score.

## Optional: XGBoost
`pip install -r requirements-optional.txt` (on Mac first run `brew install libomp`).
If XGBoost is missing or broken it is skipped automatically.

## What it does
- **Preprocessing:** removes the 2 documented outliers (huge houses sold very cheaply); drops columns with >80 % missing values; treats `MSSubClass` as a category; median-imputes and standardises numeric features; fills and one-hot-encodes categoricals (rare categories grouped). Everything sits inside one scikit-learn `Pipeline` (no train/test leakage).
- **Target:** trained on `log1p(SalePrice)` and converted back, which helps because prices are right-skewed (use `--no-log-target` to turn this off).
- **Models compared (5-fold CV R²):** Linear Regression, Ridge, SVR, SGDRegressor, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, MLP.

## Notes
- Kaggle's leaderboard metric is RMSE on log prices (lower is better), not R².
