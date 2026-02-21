#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/xtrc" ]]; then
  AINAV_BIN=".venv/bin/xtrc"
elif command -v xtrc >/dev/null 2>&1; then
  AINAV_BIN="$(command -v xtrc)"
else
  echo "xtrc not found. Run scripts/install.sh first."
  exit 1
fi

HOST="127.0.0.1"
PORT="8765"

"$AINAV_BIN" serve --host "$HOST" --port "$PORT" >/tmp/xtrc-demo.log 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for _ in {1..40}; do
  if curl -sS "http://$HOST:$PORT/status?repo_path=$ROOT_DIR/examples/demo_app" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

echo "Indexing demo repository..."
"$AINAV_BIN" index examples/demo_app

echo "Running demo query..."
"$AINAV_BIN" query "get user score" --repo examples/demo_app --top-k 5

echo "Demo complete. Server log: /tmp/xtrc-demo.log"
