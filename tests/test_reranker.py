import json

from xtrc.core.models import CodeChunk, QueryMatch
from xtrc.llm.reranker import GeminiReranker


class FakeGeminiClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.complete_calls = 0
        self.rewrite_calls = 0
        self.last_prompt = ""

    def complete_json(self, prompt: str, *, model_name: str | None = None) -> tuple[dict[str, object], int]:
        self.complete_calls += 1
        self.last_prompt = prompt
        return self.payload, 42

    def rewrite_query(self, prompt: str, *, model_name: str | None = None) -> tuple[str, int]:
        self.rewrite_calls += 1
        return "find score calculation path for user", 11


def _make_match(file_path: str, start_line: int, end_line: int, vector_score: float) -> QueryMatch:
    chunk = CodeChunk(
        chunk_id=f"{file_path}:{start_line}:{end_line}",
        repo_path="/tmp/repo",
        file_path=file_path,
        language="python",
        start_line=start_line,
        end_line=end_line,
        symbol="fn",
        symbol_kind="function",
        description=f"code in {file_path}",
        text=f"def fn_{start_line}():\n    return {start_line}",
        content_hash=f"hash-{file_path}-{start_line}",
        tokens=10,
        keywords=["score", "user"],
        symbol_terms=["fn"],
    )
    return QueryMatch(
        chunk=chunk,
        score=vector_score,
        vector_score=vector_score,
        keyword_score=0.0,
        symbol_score=0.0,
    )


def test_reranker_skips_gemini_when_vector_confidence_is_high() -> None:
    client = FakeGeminiClient(payload={"file": "src/a.py", "line": 10, "reason": "unused"})
    reranker = GeminiReranker(client, model_name="gemini-1.5-flash", threshold=0.85)

    matches = [_make_match("src/a.py", 10, 20, vector_score=0.93)]
    decision = reranker.decide("where is score computed", matches)

    assert decision is not None
    assert decision.used_gemini is False
    assert decision.selection.source == "vector"
    assert client.complete_calls == 0


def test_reranker_uses_gemini_for_low_confidence_and_caps_candidates_to_ten() -> None:
    client = FakeGeminiClient(payload={"file": "src/candidate_2.py", "line": 12, "reason": "best match"})
    reranker = GeminiReranker(client, model_name="gemini-1.5-flash", threshold=0.85)

    matches = [_make_match(f"src/candidate_{i}.py", 10, 20, vector_score=0.4) for i in range(12)]
    decision = reranker.decide("where is score computed", matches)

    assert decision is not None
    assert decision.used_gemini is True
    assert decision.selection.source == "gemini"
    assert decision.selection.file == "src/candidate_2.py"
    assert decision.selection.line == 12
    assert decision.gemini_latency_ms == 42
    assert client.complete_calls == 1

    json_payload = client.last_prompt.split("Candidates (JSON):\n", maxsplit=1)[1]
    candidates = json.loads(json_payload)
    assert len(candidates) == 10


def test_reranker_falls_back_when_gemini_selects_unknown_file() -> None:
    client = FakeGeminiClient(payload={"file": "src/missing.py", "line": 99, "reason": "wrong"})
    reranker = GeminiReranker(client, model_name="gemini-1.5-flash", threshold=0.85)

    matches = [_make_match("src/top.py", 10, 20, vector_score=0.4)]
    decision = reranker.decide("where is score computed", matches)

    assert decision is not None
    assert decision.used_gemini is False
    assert decision.selection.source == "vector"
    assert decision.selection.file == "src/top.py"
    assert "Gemini rerank failed" in decision.selection.reason
    assert client.complete_calls == 1


def test_reranker_optional_query_rewrite() -> None:
    client = FakeGeminiClient(payload={"file": "src/a.py", "line": 10, "reason": "best match"})
    reranker = GeminiReranker(
        client,
        model_name="gemini-1.5-flash",
        threshold=0.85,
        enable_rewrite=True,
    )

    matches = [_make_match("src/a.py", 10, 20, vector_score=0.4)]
    decision = reranker.decide("user score", matches)

    assert decision is not None
    assert decision.used_gemini is True
    assert decision.rewritten_query == "find score calculation path for user"
    assert decision.gemini_latency_ms == 53
    assert client.rewrite_calls == 1
    assert client.complete_calls == 1
