#!/usr/bin/env bash
# One-command website launcher for Mac/Linux:   bash run_web.sh
set -e
cd "$(dirname "$0")"

PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
  echo "Python is not installed. Install it from https://www.python.org/downloads/ and run again."; exit 1
fi

if [ ! -d .venv ]; then
  echo ">> Creating virtual environment..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate

echo ">> Checking packages (first time takes a few minutes)..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

if [ ! -f models/gbr.pkl ] || [ ! -f outputs/metrics.json ]; then
  echo ">> Training the model (about 30 seconds, only the first time)..."
  python -m src.train --skip-cv
fi

echo ">> Starting the website..."
python app.py
