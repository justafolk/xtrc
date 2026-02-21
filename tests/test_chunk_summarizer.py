from pathlib import Path

from xtrc.core.metadata_store import MetadataStore
from xtrc.core.models import CodeChunk
from xtrc.llm.chunk_summarizer import GeminiChunkSummarizer


class FakeGeminiClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete_text(self, prompt: str, *, model_name: str | None = None) -> tuple[str, int]:
        _ = prompt
        _ = model_name
        self.calls += 1
        return "Creates a post resource using a POST route handler.", 7


def _chunk() -> CodeChunk:
    return CodeChunk(
        chunk_id="chunk-1",
        repo_path="/tmp/repo",
        file_path="routes/posts.js",
        language="javascript",
        start_line=1,
        end_line=12,
        symbol="POST /posts",
        symbol_kind="route",
        description="Route handler POST /posts",
        text="router.post('/posts', createPostHandler)",
        content_hash="hash1",
        tokens=12,
        keywords=["post", "create"],
        symbol_terms=["post"],
        route_method="POST",
        route_intent="create",
        route_resource="post",
        structural_terms=["post", "create"],
    )


def test_chunk_summarizer_caches_by_summary_key(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.db")
    client = FakeGeminiClient()
    summarizer = GeminiChunkSummarizer(store, client, model_name="gemini-2.5-flash")

    chunk = _chunk()

    summaries1, latency1 = summarizer.summarize_chunks([chunk])
    summaries2, latency2 = summarizer.summarize_chunks([chunk])

    assert summaries1[chunk.chunk_id]
    assert summaries2[chunk.chunk_id] == summaries1[chunk.chunk_id]
    assert latency1 == 7
    assert latency2 == 0
    assert client.calls == 1


def test_apply_summaries_sets_chunk_llm_summary(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "metadata.db")
    client = FakeGeminiClient()
    summarizer = GeminiChunkSummarizer(store, client, model_name="gemini-2.5-flash")

    chunk = _chunk()
    summaries, _ = summarizer.summarize_chunks([chunk])
    updated = summarizer.apply_summaries([chunk], summaries)

    assert updated[0].llm_summary == summaries[chunk.chunk_id]
