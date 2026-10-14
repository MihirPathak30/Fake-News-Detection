#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Fake News Detection Setup ==="

PYTHON=python3.11

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python 3.11 not found."
    exit 1
fi

echo "Using: $($PYTHON --version)"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing requirements..."
python -m pip install -r requirements.txt

echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" || true

echo "=== Starting project ==="
python app.py
