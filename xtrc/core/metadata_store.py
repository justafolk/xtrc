from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from xtrc.core.models import CodeChunk, StatusStats


class MetadataStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    repo_path TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    last_indexed_at TEXT NOT NULL,
                    PRIMARY KEY (repo_path, file_path)
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    repo_path TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    language TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    symbol TEXT,
                    symbol_kind TEXT,
                    description TEXT NOT NULL,
                    text TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    tokens INTEGER NOT NULL,
                    keywords TEXT NOT NULL,
                    symbol_terms TEXT NOT NULL,
                    route_method TEXT,
                    route_path TEXT,
                    route_intent TEXT,
                    route_resource TEXT,
                    intent_tags TEXT NOT NULL DEFAULT '[]',
                    structural_terms TEXT NOT NULL DEFAULT '[]',
                    llm_summary TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_repo_file
                    ON chunks (repo_path, file_path);

                CREATE INDEX IF NOT EXISTS idx_chunks_repo
                    ON chunks (repo_path);

                CREATE TABLE IF NOT EXISTS embeddings (
                    content_hash TEXT PRIMARY KEY,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS repo_meta (
                    repo_path TEXT PRIMARY KEY,
                    last_indexed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS llm_summaries (
                    summary_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
        self._ensure_column("chunks", "route_method", "TEXT")
        self._ensure_column("chunks", "route_path", "TEXT")
        self._ensure_column("chunks", "route_intent", "TEXT")
        self._ensure_column("chunks", "route_resource", "TEXT")
        self._ensure_column("chunks", "intent_tags", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("chunks", "structural_terms", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("chunks", "llm_summary", "TEXT")

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        with self._connect() as conn:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            if any(str(row["name"]) == column for row in rows):
                return
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def clear_repo(self, repo_path: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM files WHERE repo_path = ?", (repo_path,))
            conn.execute("DELETE FROM chunks WHERE repo_path = ?", (repo_path,))
            conn.execute("DELETE FROM repo_meta WHERE repo_path = ?", (repo_path,))

    def get_file_hashes(self, repo_path: str) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT file_path, content_hash FROM files WHERE repo_path = ?", (repo_path,)
            ).fetchall()
        return {str(row["file_path"]): str(row["content_hash"]) for row in rows}

    def upsert_file_hash(self, repo_path: str, file_path: str, content_hash: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (repo_path, file_path, content_hash, last_indexed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(repo_path, file_path)
                DO UPDATE SET content_hash = excluded.content_hash,
                              last_indexed_at = excluded.last_indexed_at
                """,
                (repo_path, file_path, content_hash, now),
            )

    def delete_files(self, repo_path: str, file_paths: Iterable[str]) -> None:
        file_paths = list(file_paths)
        if not file_paths:
            return
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM files WHERE repo_path = ? AND file_path = ?",
                ((repo_path, path) for path in file_paths),
            )

    def get_chunk_ids_for_file(self, repo_path: str, file_path: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks WHERE repo_path = ? AND file_path = ?",
                (repo_path, file_path),
            ).fetchall()
        return [str(row["chunk_id"]) for row in rows]

    def delete_chunks_by_file(self, repo_path: str, file_path: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE repo_path = ? AND file_path = ?",
                (repo_path, file_path),
            )

    def delete_chunks_by_ids(self, chunk_ids: Iterable[str]) -> None:
        chunk_ids = list(chunk_ids)
        if not chunk_ids:
            return
        with self._connect() as conn:
            conn.executemany("DELETE FROM chunks WHERE chunk_id = ?", ((chunk_id,) for chunk_id in chunk_ids))

    def upsert_chunks(self, chunks: list[CodeChunk]) -> None:
        if not chunks:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id,
                    repo_path,
                    file_path,
                    language,
                    start_line,
                    end_line,
                    symbol,
                    symbol_kind,
                    description,
                    text,
                    content_hash,
                    tokens,
                    keywords,
                    symbol_terms,
                    route_method,
                    route_path,
                    route_intent,
                    route_resource,
                    intent_tags,
                    structural_terms,
                    llm_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id)
                DO UPDATE SET
                    repo_path = excluded.repo_path,
                    file_path = excluded.file_path,
                    language = excluded.language,
                    start_line = excluded.start_line,
                    end_line = excluded.end_line,
                    symbol = excluded.symbol,
                    symbol_kind = excluded.symbol_kind,
                    description = excluded.description,
                    text = excluded.text,
                    content_hash = excluded.content_hash,
                    tokens = excluded.tokens,
                    keywords = excluded.keywords,
                    symbol_terms = excluded.symbol_terms,
                    route_method = excluded.route_method,
                    route_path = excluded.route_path,
                    route_intent = excluded.route_intent,
                    route_resource = excluded.route_resource,
                    intent_tags = excluded.intent_tags,
                    structural_terms = excluded.structural_terms,
                    llm_summary = excluded.llm_summary
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.repo_path,
                        chunk.file_path,
                        chunk.language,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.symbol,
                        chunk.symbol_kind,
                        chunk.description,
                        chunk.text,
                        chunk.content_hash,
                        chunk.tokens,
                        json.dumps(chunk.keywords),
                        json.dumps(chunk.symbol_terms),
                        chunk.route_method,
                        chunk.route_path,
                        chunk.route_intent,
                        chunk.route_resource,
                        json.dumps(chunk.intent_tags),
                        json.dumps(chunk.structural_terms),
                        chunk.llm_summary,
                    )
                    for chunk in chunks
                ],
            )

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, CodeChunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        query = f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})"
        with self._connect() as conn:
            rows = conn.execute(query, chunk_ids).fetchall()

        out: dict[str, CodeChunk] = {}
        for row in rows:
            keywords = json.loads(str(row["keywords"]))
            symbol_terms = json.loads(str(row["symbol_terms"]))
            raw_intents = row["intent_tags"] if "intent_tags" in row.keys() else "[]"
            try:
                intent_tags = json.loads(str(raw_intents)) if raw_intents is not None else []
            except json.JSONDecodeError:
                intent_tags = []
            raw_structural = row["structural_terms"] if "structural_terms" in row.keys() else "[]"
            try:
                structural_terms = json.loads(str(raw_structural)) if raw_structural is not None else []
            except json.JSONDecodeError:
                structural_terms = []
            chunk = CodeChunk(
                chunk_id=str(row["chunk_id"]),
                repo_path=str(row["repo_path"]),
                file_path=str(row["file_path"]),
                language=str(row["language"]),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                symbol=str(row["symbol"]) if row["symbol"] is not None else None,
                symbol_kind=str(row["symbol_kind"]) if row["symbol_kind"] is not None else None,
                description=str(row["description"]),
                text=str(row["text"]),
                content_hash=str(row["content_hash"]),
                tokens=int(row["tokens"]),
                keywords=[str(item) for item in keywords],
                symbol_terms=[str(item) for item in symbol_terms],
                route_method=str(row["route_method"]) if row["route_method"] is not None else None,
                route_path=str(row["route_path"]) if row["route_path"] is not None else None,
                route_intent=str(row["route_intent"]) if row["route_intent"] is not None else None,
                route_resource=str(row["route_resource"]) if row["route_resource"] is not None else None,
                intent_tags=[str(item) for item in intent_tags],
                structural_terms=[str(item) for item in structural_terms],
                llm_summary=str(row["llm_summary"]) if row["llm_summary"] is not None else None,
            )
            out[chunk.chunk_id] = chunk

        return out

    def set_repo_last_indexed(self, repo_path: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO repo_meta (repo_path, last_indexed_at)
                VALUES (?, ?)
                ON CONFLICT(repo_path) DO UPDATE SET
                    last_indexed_at = excluded.last_indexed_at
                """,
                (repo_path, now),
            )

    def get_status(self, repo_path: str) -> StatusStats:
        with self._connect() as conn:
            file_count_row = conn.execute(
                "SELECT COUNT(*) AS n FROM files WHERE repo_path = ?", (repo_path,)
            ).fetchone()
            chunk_count_row = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE repo_path = ?", (repo_path,)
            ).fetchone()
            meta_row = conn.execute(
                "SELECT last_indexed_at FROM repo_meta WHERE repo_path = ?", (repo_path,)
            ).fetchone()

        last_indexed = None
        if meta_row and meta_row["last_indexed_at"]:
            last_indexed = datetime.fromisoformat(str(meta_row["last_indexed_at"]))

        return StatusStats(
            repo_path=repo_path,
            indexed_files=int(file_count_row["n"] if file_count_row else 0),
            indexed_chunks=int(chunk_count_row["n"] if chunk_count_row else 0),
            last_indexed_at=last_indexed,
        )

    def get_cached_embeddings(self, content_hashes: list[str]) -> dict[str, np.ndarray]:
        if not content_hashes:
            return {}
        placeholders = ",".join("?" for _ in content_hashes)
        query = f"SELECT content_hash, dimension, vector FROM embeddings WHERE content_hash IN ({placeholders})"
        with self._connect() as conn:
            rows = conn.execute(query, content_hashes).fetchall()

        out: dict[str, np.ndarray] = {}
        for row in rows:
            dim = int(row["dimension"])
            vec = np.frombuffer(bytes(row["vector"]), dtype=np.float32, count=dim).copy()
            out[str(row["content_hash"])] = vec
        return out

    def upsert_cached_embeddings(self, vectors: dict[str, np.ndarray]) -> None:
        if not vectors:
            return
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO embeddings (content_hash, dimension, vector, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    dimension = excluded.dimension,
                    vector = excluded.vector,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        key,
                        int(vector.shape[0]),
                        sqlite3.Binary(vector.astype(np.float32).tobytes()),
                        now,
                    )
                    for key, vector in vectors.items()
                ],
            )

    def get_cached_chunk_summaries(self, keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        query = f"SELECT summary_key, summary FROM llm_summaries WHERE summary_key IN ({placeholders})"
        with self._connect() as conn:
            rows = conn.execute(query, keys).fetchall()
        return {str(row["summary_key"]): str(row["summary"]) for row in rows}

    def upsert_cached_chunk_summaries(self, model: str, summaries: dict[str, str]) -> None:
        if not summaries:
            return
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO llm_summaries (summary_key, model, summary, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(summary_key) DO UPDATE SET
                    model = excluded.model,
                    summary = excluded.summary,
                    updated_at = excluded.updated_at
                """,
                [(key, model, summary, now) for key, summary in summaries.items()],
            )
