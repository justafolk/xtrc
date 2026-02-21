#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_DEV="${INSTALL_DEV:-0}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
python -m pip install -e .

if [[ "$INSTALL_DEV" == "1" ]]; then
  python -m pip install -r requirements-dev.txt
fi

if command -v npm >/dev/null 2>&1; then
  (
    cd vscode-extension
    npm install
  )
else
  echo "npm not found; skipping VS Code extension dependencies"
fi

echo "Installation complete. Activate with: source $VENV_DIR/bin/activate"
