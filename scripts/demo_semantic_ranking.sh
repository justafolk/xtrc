#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${1:-examples/demo_app}"
HOST="${AINAV_HOST:-127.0.0.1}"
PORT="${AINAV_PORT:-8765}"

export GEMINI_SUMMARIZE_ON_INDEX="${GEMINI_SUMMARIZE_ON_INDEX:-true}"
export QUERY_REWRITE_ENABLED="${QUERY_REWRITE_ENABLED:-true}"
export LOCAL_RERANKER_ENABLED="${LOCAL_RERANKER_ENABLED:-true}"

cat <<EOF
[demo] semantic ranking demo
[demo] repo=${REPO_PATH}
[demo] query=function to create new posts
EOF

xtrc index "${REPO_PATH}" --rebuild --host "${HOST}" --port "${PORT}"
xtrc query "function to create new posts" --repo "${REPO_PATH}" --host "${HOST}" --port "${PORT}" --json
