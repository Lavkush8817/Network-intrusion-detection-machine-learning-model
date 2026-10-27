@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)
%PYTHON% -c "import flask, joblib, numpy, pandas, sklearn" >nul 2>nul
if errorlevel 1 (
  echo Python dependencies are missing. Set up the app first with:
  echo   py -m venv .venv
  echo   .venv\Scripts\python -m pip install -r requirements.txt
  exit /b 1
)
%PYTHON% app.py
