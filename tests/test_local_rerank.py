from xtrc.core.models import CodeChunk, QueryMatch
from xtrc.query.rerank import LocalReranker


def _match(chunk_id: str, file_path: str, score: float) -> QueryMatch:
    chunk = CodeChunk(
        chunk_id=chunk_id,
        repo_path="/tmp/repo",
        file_path=file_path,
        language="python",
        start_line=1,
        end_line=5,
        symbol="fn",
        symbol_kind="function",
        description="handler",
        text="def fn(): pass",
        content_hash=f"h-{chunk_id}",
        tokens=10,
        keywords=["create", "post"],
        symbol_terms=["fn"],
        intent_tags=["create_resource"],
        llm_summary="Creates post",
    )
    return QueryMatch(
        chunk=chunk,
        score=score,
        vector_score=score,
        keyword_score=0.0,
        symbol_score=0.0,
    )


def test_local_reranker_reorders_top_candidates(monkeypatch) -> None:
    reranker = LocalReranker(enabled=True, timeout_seconds=2.0)
    matches = [
        _match("a", "a.py", 0.80),
        _match("b", "b.py", 0.79),
    ]

    monkeypatch.setattr(reranker, "_predict_scores", lambda query, items: [0.1, 3.0])

    out, used, latency = reranker.rerank("create post", matches)

    assert used is True
    assert latency is not None
    assert out[0].chunk.file_path == "b.py"


def test_local_reranker_disabled() -> None:
    reranker = LocalReranker(enabled=False)
    matches = [_match("a", "a.py", 0.80)]

    out, used, latency = reranker.rerank("create post", matches)

    assert out == matches
    assert used is False
    assert latency is None
