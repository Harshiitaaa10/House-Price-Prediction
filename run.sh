#!/usr/bin/env bash
# One-command setup + run for Mac/Linux:   bash run.sh
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

echo ">> Installing packages (first time takes a few minutes)..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo ">> Training..."
python -m src.train "$@"
