#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${1:-.}"
HOST="${AINAV_HOST:-127.0.0.1}"
PORT="${AINAV_PORT:-8765}"

echo "[reindex] repo=${REPO_PATH} host=${HOST} port=${PORT}"
xtrc index "${REPO_PATH}" --rebuild --host "${HOST}" --port "${PORT}"
