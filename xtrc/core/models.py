from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class SymbolBlock:
    kind: str
    name: str | None
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    repo_path: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol: str | None
    symbol_kind: str | None
    description: str
    text: str
    content_hash: str
    tokens: int
    keywords: list[str]
    symbol_terms: list[str]
    route_method: str | None = None
    route_path: str | None = None
    route_intent: str | None = None
    route_resource: str | None = None
    intent_tags: list[str] = field(default_factory=list)
    structural_terms: list[str] = field(default_factory=list)
    llm_summary: str | None = None


@dataclass(frozen=True)
class IndexStats:
    repo_path: str
    files_scanned: int
    files_indexed: int
    files_deleted: int
    chunks_indexed: int
    duration_ms: int


@dataclass(frozen=True)
class StatusStats:
    repo_path: str
    indexed_files: int
    indexed_chunks: int
    last_indexed_at: datetime | None


@dataclass(frozen=True)
class QueryMatch:
    chunk: CodeChunk
    vector_score: float
    keyword_score: float
    symbol_score: float
    score: float
    intent_score: float = 0.0
    structural_score: float = 0.0
    matched_intents: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    explanation: str = ""


SelectionSource = Literal["vector", "gemini"]


@dataclass(frozen=True)
class QuerySelection:
    file: str
    line: int
    reason: str
    source: SelectionSource


@dataclass(frozen=True)
class QueryOutcome:
    matches: list[QueryMatch]
    duration_ms: int
    selection: QuerySelection | None
    used_gemini: bool
    gemini_model: str | None
    gemini_latency_ms: int | None
    rewritten_query: str | None
