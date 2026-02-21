# xtrc

`xtrc` is a local, editor-agnostic AI code navigation system.
It indexes your repository with tree-sitter + embeddings, stores vectors in local Qdrant, and returns file+line jump targets from natural-language queries.

![Screenshare-2026-02-211_52_31PM-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/47df201a-8fc2-42d3-b14e-66a1dc99088d)
## Key Capabilities
- Local daemon with FastAPI APIs: `/index`, `/query`, `/status`
- Tree-sitter parsing for Python, JavaScript, TypeScript, TSX
- Semantic chunking with token-aware chunk size (200-800 tokens)
- Chunk metadata includes path, line range, symbol name, description
- Local embeddings via `BAAI/bge-base-en-v1.5`
- Embedding cache keyed by content hash
- Local Qdrant vector search (embedded mode)
- Incremental indexing (changed files only) with full rebuild option
- Honors `.gitignore` patterns during indexing
- Hybrid ranking: `0.50 vector + 0.18 keyword + 0.12 symbol + 0.12 intent + 0.08 structural`
- Optional Gemini reranking for low-confidence results (top 10 candidates)
- Optional Gemini-based query rewriting with response caching
- Optional Gemini-based chunk summaries generated once during indexing (cached)
- Intent-aware route encoding (`POST -> create`, `PUT/PATCH -> update`, `DELETE -> delete`)
- Intent + structural query boosting for API/route searches
- Semantic embedding input built from enriched metadata and summaries (no raw-code body embedding)
- Optional pre-embedding query rewriting to precise technical intent
- Local cross-encoder reranking on top candidates before final LLM rerank
- CLI, Neovim, and VS Code clients using the same HTTP protocol

## Architecture

```text
Clients (CLI, Neovim, VS Code)
          |
          | HTTP+JSON
          v
+-----------------------------+
| FastAPI daemon (xtrc)      |
| endpoints: /index /query    |
|            /status          |
+-----------------------------+
          |
          v
+-----------------------------+
| Core services               |
| - Repo walker + ignore      |
| - Tree-sitter parser        |
| - Chunk builder             |
| - Embedding service + cache |
| - Qdrant vector store       |
| - Hybrid query scorer       |
| - Optional Gemini reranker  |
| - Optional chunk summarizer |
| - Query rewriter            |
| - Local cross-encoder rank  |
+-----------------------------+
          |
          v
+-----------------------------+
| Local persistent state      |
| - .xtrc/metadata.db        |
| - .xtrc/qdrant/            |
| - .xtrc embeddings cache   |
+-----------------------------+
```

## Data Flow

1. `POST /index` walks repo recursively and filters ignored paths (`.git`, `node_modules`, `dist`, `build`, and `.gitignore` matches).
2. File content hash is compared with stored hash to detect changes.
3. Changed files are parsed by tree-sitter into symbols and blocks.
4. Chunk builder creates semantic chunks in token budget.
5. Embeddings are generated or loaded from hash-based cache.
6. Vectors are upserted into Qdrant with metadata payload.
7. During indexing, route chunks are enriched with explicit intent metadata:
   - `Intent: create resource`
   - `HTTP method: POST`
   - `Resource: post`
8. Optional index-time summary is generated once per chunk (cached by chunk content hash + model).
9. Embedding input is semantic text only:
   - `File`, `Symbol`, `Type`, `Intent`, `Summary`, HTTP metadata.
10. Query may be rewritten to technical intent before embedding.
11. `POST /query` embeds rewritten query, runs ANN search, and computes hybrid scores with structural heuristics.
12. Top candidates are locally reranked by cross-encoder before optional Gemini reranking.
13. Response includes ranked results with explanations (`matched intents`, `matched keywords`, `why`) plus final `selection`.

## Repository Layout

```text
xtrc/
  api/
  core/
  indexer/
  llm/
  query/
  ranking/
  cli.py
  client.py
  config.py
  schemas.py
  server.py
plugins/
  nvim/xtrc.lua
scripts/
  install.sh
  run_demo.sh
  bench_query.py
  reindex_with_semantics.sh
  demo_semantic_ranking.sh
examples/
  demo_app/
tests/
pyproject.toml
requirements.txt
requirements-dev.txt
README.md
```

## Install

### Python backend

```bash
scripts/install.sh
source .venv/bin/activate
```

Manual install alternative:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## LLM Configuration (Optional)

Set these env vars before running `xtrc serve`:

```bash
export USE_GEMINI=true
export GEMINI_API_KEY="<your_api_key>"
export GEMINI_MODEL=gemini-2.5-flash   # or another supported Gemini model
export GEMINI_THRESHOLD=0.85
export GEMINI_TIMEOUT_SECONDS=2
export GEMINI_ENABLE_REWRITE=false
export GEMINI_SUMMARIZE_ON_INDEX=true
export GEMINI_SUMMARY_MODEL=gemini-2.5-flash
export GEMINI_SUMMARY_MAX_CHARS=320
export LLM_PROVIDER=gemini            # gemini or openai
export LLM_TIMEOUT_SECONDS=2
export QUERY_REWRITE_ENABLED=false  # set true to enable LLM query rewrite (higher latency)
export QUERY_REWRITE_MODEL=gemini-2.5-flash
export LOCAL_RERANKER_ENABLED=false # set true to enable local cross-encoder rerank (higher latency)
export LOCAL_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
export LOCAL_RERANKER_TOP_K=10
export HEURISTIC_ROUTE_BOOST=1.3
export HEURISTIC_NOISE_PENALTY=0.7
export HEURISTIC_INTENT_BOOST=1.2
# If using OpenAI provider:
# export OPENAI_API_KEY="<your_openai_key>"
```

Behavior:

- Best vector similarity `>= GEMINI_THRESHOLD`: Gemini is skipped.
- Best vector similarity `< GEMINI_THRESHOLD`: top 10 candidates are sent to Gemini.
- If Gemini fails or times out, `xtrc` gracefully falls back to top vector result.
- If `GEMINI_SUMMARIZE_ON_INDEX=true`, chunk summaries are generated once and cached by chunk content key.
- Query rewrite and index-time summaries use a hard LLM timeout (`LLM_TIMEOUT_SECONDS`, default `2s`).
- Local cross-encoder reranking runs on top candidates before Gemini fallback when `LOCAL_RERANKER_ENABLED=true`.
- By default, `QUERY_REWRITE_ENABLED=false` and `LOCAL_RERANKER_ENABLED=false` for lower `/query` latency.

## Quick Start

1. Start daemon:

```bash
xtrc serve
```

2. Index repository:

```bash
xtrc index .
```

3. Query for jump target:

```bash
xtrc query "get user score" --repo . --top-k 8
```

4. Check status:

```bash
xtrc status .
```

## CLI Reference

- `xtrc serve [--host 127.0.0.1] [--port 8765]`
- `xtrc index <repo_path> [--rebuild] [--json]`
- `xtrc query "<natural language>" [--repo <path>] [--top-k N] [--json]`
- `xtrc status <repo_path> [--json]`

## HTTP API

### `POST /index`

Request:

```json
{
  "repo_path": ".",
  "rebuild": false
}
```

Response:

```json
{
  "status": "ok",
  "repo_path": "/absolute/path/to/repo",
  "files_scanned": 120,
  "files_indexed": 8,
  "files_deleted": 1,
  "chunks_indexed": 233,
  "duration_ms": 842
}
```

### `POST /query`

Request:

```json
{
  "repo_path": ".",
  "query": "get user score",
  "top_k": 8
}
```

Response:

```json
{
  "status": "ok",
  "repo_path": "/absolute/path/to/repo",
  "query": "get user score",
  "results": [
    {
      "file_path": "src/user.py",
      "start_line": 42,
      "end_line": 88,
      "symbol": "get_user_score",
      "description": "Function get_user_score in src/user.py...",
      "score": 0.83,
      "vector_score": 0.78,
      "keyword_score": 1.0,
      "symbol_score": 0.5,
      "intent_score": 1.0,
      "structural_score": 0.75,
      "matched_intents": ["create_resource"],
      "matched_keywords": ["create", "post", "endpoint"],
      "explanation": "semantic=0.780; keyword=1.000; symbol=0.500; intent=1.000; structural=0.750; heuristics=intent match: create_resource, route handler boost"
    }
  ],
  "duration_ms": 12,
  "selection": {
    "file": "src/user.py",
    "line": 42,
    "reason": "Chunk contains the user scoring path and return value."
  },
  "selection_source": "gemini",
  "used_gemini": true,
  "gemini_model": "gemini-2.5-flash",
  "gemini_latency_ms": 241,
  "rewritten_query": "Find where user score is computed and returned."
}
```

### `GET /status?repo_path=.`

Response:

```json
{
  "status": "ok",
  "repo_path": "/absolute/path/to/repo",
  "indexed_files": 120,
  "indexed_chunks": 2330,
  "model": "BAAI/bge-base-en-v1.5",
  "healthy": true,
  "last_indexed_at": "2026-02-20T22:08:17.439000+00:00"
}
```

### Error Envelope

```json
{
  "status": "error",
  "error": {
    "code": "INVALID_REPO",
    "message": "Repository path does not exist or is not a directory",
    "details": {}
  }
}
```

## Neovim Plugin

File: `plugins/nvim/xtrc.lua`

Minimal setup in Neovim config:

```lua
local xtrc = dofile("/absolute/path/to/repo/plugins/nvim/xtrc.lua")
xtrc.setup({
  server_url = "http://127.0.0.1:8765",
  top_k = 8,
})
```

Usage:

```vim
:xtrc get user score
```

Behavior:

- One result: opens file directly at line
- Multiple results: shows `vim.ui.select` picker

## Performance Notes

- Query path is optimized with cached query embeddings and ANN search.
- Gemini calls are gated by a similarity threshold to avoid unnecessary LLM latency.
- Gemini rerank + rewrite responses are cached with LRU strategy.
- Index-time chunk summaries are cached in SQLite and reused across subsequent indexes.
- Intent + structural boosts reduce route query ambiguity (`create post`, `update post`, `delete post`).
- Local cross-encoder reranking improves ordering among top semantic candidates.
- For BGE models, query embeddings use model-recommended retrieval instruction prefixing.
- Embedding cache uses content hash keys in local SQLite.
- Incremental index avoids re-embedding unchanged files.
- Qdrant collection is per repository for isolated search space.

Actual latency depends on hardware and model warm-up.

Benchmark helper:

```bash
scripts/bench_query.py "get user score" --repo . --runs 100
```

## Demo

Run complete demo flow:

```bash
scripts/run_demo.sh
```

Demo indexes `examples/demo_app` and runs query `get user score`.

Gemini rerank demo:

```bash
export USE_GEMINI=true
export GEMINI_API_KEY="<your_api_key>"
export GEMINI_MODEL=gemini-2.5-flash
xtrc serve
xtrc index examples/demo_app
xtrc query "where is score calculation used" --repo examples/demo_app --json
```

Check `selection`, `selection_source`, and `used_gemini` in the JSON response.

Semantic reindex migration:

```bash
scripts/reindex_with_semantics.sh .
```

Semantic ranking demo:

```bash
scripts/demo_semantic_ranking.sh examples/demo_app
```

## Testing

Run unit tests:

```bash
source .venv/bin/activate
.venv/bin/pytest
```

Quick quality checks:

```bash
python -m compileall xtrc
ruff check .
mypy xtrc
```

## Troubleshooting

- Server unreachable:
  - Start daemon with `xtrc serve`.
  - Verify host/port match client settings.
- First query is slow:
  - Embedding model warm-up can take seconds.
  - Subsequent queries use in-memory and SQLite cache.
- Indexing is slower after enabling summaries:
  - `GEMINI_SUMMARIZE_ON_INDEX=true` adds one LLM summary pass per new/changed chunk.
  - Summaries are cached and reused on future indexes.
- No results:
  - Ensure index completed successfully.
  - Use `xtrc status .` to verify chunk count.
  - Rebuild index with `xtrc index . --rebuild`.
- Gemini not used when expected:
  - Confirm `USE_GEMINI=true`.
  - Confirm `GEMINI_API_KEY` is set in the server process environment.
  - Lower `GEMINI_THRESHOLD` if vector confidence is usually high.
- Gemini failures/timeouts:
  - Default timeout is 2 seconds (`GEMINI_TIMEOUT_SECONDS`).
  - On timeout/failure, response falls back to vector top match.
- Crash after changing embedding model:
  - Existing vector collections may use a different dimension.
  - `xtrc` now auto-resets incompatible collections, but `xtrc index . --rebuild` is still recommended immediately after model changes.
  - If query returns `INDEX_DIMENSION_MISMATCH`, run `xtrc index <repo> --rebuild`.
- VS Code command missing:
  - Compile extension: `npm run compile`.
  - Relaunch extension host.
- Neovim plugin fails:
  - Ensure `curl` is installed.
  - Confirm plugin points to running `server_url`.

## License

MIT

## Counter

![Repo Views](https://visitor-badge.laobi.icu/badge?page_id=justafolk.xtrc)
