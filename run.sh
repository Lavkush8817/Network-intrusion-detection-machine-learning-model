#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PYTHON="python3"
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
fi

if ! "$PYTHON" -c 'import flask, joblib, numpy, pandas, sklearn' >/dev/null 2>&1; then
  echo "Python dependencies are missing. Set up the app first with:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

exec "$PYTHON" app.py
