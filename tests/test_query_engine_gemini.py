from pathlib import Path

import numpy as np

from xtrc.core.models import CodeChunk, QuerySelection
from xtrc.core.query_engine import QueryEngine
from xtrc.core.vector_store import SearchHit
from xtrc.llm.reranker import RerankDecision


class FakeEmbeddingService:
    def embed_query(self, query: str) -> np.ndarray:
        _ = query
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FakeMetadataStore:
    def __init__(self, chunks: dict[str, CodeChunk]) -> None:
        self._chunks = chunks

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, CodeChunk]:
        return {chunk_id: self._chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self._chunks}


class FakeVectorStore:
    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits

    def search(self, repo_path: str, query_vector: np.ndarray, limit: int) -> list[SearchHit]:
        _ = repo_path
        _ = query_vector
        return self._hits[:limit]


class FakeScorer:
    def score(
        self,
        query: str,
        vector_score: float,
        keywords: list[str],
        symbol_terms: list[str],
        route_intent: str | None = None,
        route_method: str | None = None,
        route_resource: str | None = None,
        structural_terms: list[str] | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        _ = query
        _ = keywords
        _ = symbol_terms
        _ = route_intent
        _ = route_method
        _ = route_resource
        _ = structural_terms
        return vector_score, vector_score, 0.0, 0.0, 0.0, 0.0


class FakeReranker:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, query: str, matches: list[object]) -> RerankDecision:
        _ = query
        _ = matches
        self.calls += 1
        return RerankDecision(
            selection=QuerySelection(
                file="src/b.py",
                line=25,
                reason="Gemini selected better behavioral match",
                source="gemini",
            ),
            used_gemini=True,
            gemini_model="gemini-1.5-flash",
            gemini_latency_ms=18,
            rewritten_query=None,
        )


def _chunk(chunk_id: str, file_path: str, start: int, end: int) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        repo_path="/tmp/repo",
        file_path=file_path,
        language="python",
        start_line=start,
        end_line=end,
        symbol="fn",
        symbol_kind="function",
        description=f"chunk in {file_path}",
        text="def fn():\n    pass",
        content_hash=f"hash-{chunk_id}",
        tokens=20,
        keywords=["score"],
        symbol_terms=["fn"],
    )


def test_query_engine_applies_reranker_selection() -> None:
    chunks = {
        "c1": _chunk("c1", "src/a.py", 10, 20),
        "c2": _chunk("c2", "src/b.py", 21, 40),
    }
    hits = [
        SearchHit(chunk_id="c1", score=0.61, payload={}),
        SearchHit(chunk_id="c2", score=0.6, payload={}),
    ]

    reranker = FakeReranker()
    engine = QueryEngine(
        metadata_store=FakeMetadataStore(chunks),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(hits),
        scorer=FakeScorer(),
        reranker=reranker,
    )

    outcome = engine.query(Path("/tmp/repo"), "where is scoring done", top_k=1)

    assert len(outcome.matches) == 1
    assert outcome.selection is not None
    assert outcome.selection.source == "gemini"
    assert outcome.selection.file == "src/b.py"
    assert outcome.used_gemini is True
    assert reranker.calls == 1


def test_query_engine_falls_back_to_vector_selection_without_reranker() -> None:
    chunks = {"c1": _chunk("c1", "src/a.py", 10, 20)}
    hits = [SearchHit(chunk_id="c1", score=0.91, payload={})]

    engine = QueryEngine(
        metadata_store=FakeMetadataStore(chunks),
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(hits),
        scorer=FakeScorer(),
        reranker=None,
    )

    outcome = engine.query(Path("/tmp/repo"), "where is scoring done", top_k=1)

    assert outcome.selection is not None
    assert outcome.selection.source == "vector"
    assert outcome.selection.file == "src/a.py"
    assert outcome.used_gemini is False
