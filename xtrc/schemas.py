from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorPayload(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    status: str = "error"
    error: ErrorPayload


class IndexRequest(BaseModel):
    repo_path: str = Field(default=".", description="Path to repository root")
    rebuild: bool = Field(default=False, description="Perform full rebuild")


class IndexResponse(BaseModel):
    status: str = "ok"
    repo_path: str
    files_scanned: int
    files_indexed: int
    files_deleted: int
    chunks_indexed: int
    duration_ms: int


class QueryRequest(BaseModel):
    repo_path: str = Field(default=".")
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)


class QueryResult(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    symbol: str | None
    description: str
    score: float
    vector_score: float
    keyword_score: float
    symbol_score: float
    intent_score: float = 0.0
    structural_score: float = 0.0
    matched_intents: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    explanation: str = ""


class QuerySelection(BaseModel):
    file: str
    line: int
    reason: str


class QueryResponse(BaseModel):
    status: str = "ok"
    repo_path: str
    query: str
    results: list[QueryResult]
    duration_ms: int
    selection: QuerySelection | None = None
    selection_source: Literal["vector", "gemini"] | None = None
    used_gemini: bool = False
    gemini_model: str | None = None
    gemini_latency_ms: int | None = None
    rewritten_query: str | None = None


class StatusResponse(BaseModel):
    status: str = "ok"
    repo_path: str
    indexed_files: int
    indexed_chunks: int
    model: str
    healthy: bool
    last_indexed_at: datetime | None = None
