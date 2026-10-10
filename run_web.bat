@echo off
rem One-command website launcher for Windows:  run_web.bat
cd /d "%~dp0"
if not exist .venv (python -m venv .venv)
call .venv\Scripts\activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if not exist models\gbr.pkl (python -m src.train --skip-cv)
if not exist outputs\metrics.json (python -m src.train --skip-cv)
python app.py
