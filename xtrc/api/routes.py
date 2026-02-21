from __future__ import annotations

import asyncio
from pathlib import Path
from fastapi import APIRouter

from xtrc.core.daemon import AinavDaemon
from xtrc.schemas import (
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    QuerySelection,
    StatusResponse,
)


def build_router(daemon: AinavDaemon) -> APIRouter:
    router = APIRouter()

    @router.post("/index", response_model=IndexResponse)
    async def index_repo(request: IndexRequest) -> IndexResponse:
        stats = await asyncio.to_thread(daemon.index, request.repo_path, request.rebuild)
        return IndexResponse(
            repo_path=stats.repo_path,
            files_scanned=stats.files_scanned,
            files_indexed=stats.files_indexed,
            files_deleted=stats.files_deleted,
            chunks_indexed=stats.chunks_indexed,
            duration_ms=stats.duration_ms,
        )

    @router.post("/query", response_model=QueryResponse)
    async def query_repo(request: QueryRequest) -> QueryResponse:
        outcome = await asyncio.to_thread(
            daemon.query,
            request.repo_path,
            request.query,
            request.top_k,
        )
        repo_path = str(Path(request.repo_path).expanduser().resolve())
        return QueryResponse(
            repo_path=repo_path,
            query=request.query,
            results=[
                QueryResult(
                    file_path=match.chunk.file_path,
                    start_line=match.chunk.start_line,
                    end_line=match.chunk.end_line,
                    symbol=match.chunk.symbol,
                    description=match.chunk.description,
                    score=round(match.score, 6),
                    vector_score=round(match.vector_score, 6),
                    keyword_score=round(match.keyword_score, 6),
                    symbol_score=round(match.symbol_score, 6),
                    intent_score=round(match.intent_score, 6),
                    structural_score=round(match.structural_score, 6),
                    matched_intents=match.matched_intents,
                    matched_keywords=match.matched_keywords,
                    explanation=match.explanation,
                )
                for match in outcome.matches
            ],
            duration_ms=outcome.duration_ms,
            selection=(
                QuerySelection(
                    file=outcome.selection.file,
                    line=outcome.selection.line,
                    reason=outcome.selection.reason,
                )
                if outcome.selection is not None
                else None
            ),
            selection_source=outcome.selection.source if outcome.selection is not None else None,
            used_gemini=outcome.used_gemini,
            gemini_model=outcome.gemini_model,
            gemini_latency_ms=outcome.gemini_latency_ms,
            rewritten_query=outcome.rewritten_query,
        )

    @router.get("/status", response_model=StatusResponse)
    async def status(repo_path: str = ".") -> StatusResponse:
        stats = await asyncio.to_thread(daemon.status, repo_path)
        return StatusResponse(
            repo_path=stats.repo_path,
            indexed_files=stats.indexed_files,
            indexed_chunks=stats.indexed_chunks,
            model=daemon.model_name(),
            healthy=True,
            last_indexed_at=stats.last_indexed_at,
        )

    return router
