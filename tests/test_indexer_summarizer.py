from xtrc.core.models import CodeChunk
from xtrc.indexer.summarizer import IndexChunkSummarizer


def _chunk() -> CodeChunk:
    return CodeChunk(
        chunk_id="c1",
        repo_path="/tmp/repo",
        file_path="routes/posts.js",
        language="javascript",
        start_line=1,
        end_line=6,
        symbol="POST /posts",
        symbol_kind="route",
        description="Route handler",
        text="router.post('/posts', createPostHandler)",
        content_hash="h",
        tokens=10,
        keywords=["create", "post"],
        symbol_terms=["post"],
        route_method="POST",
        route_path="/posts",
        route_intent="create",
        route_resource="post",
        intent_tags=["create_resource", "route_handler"],
        llm_summary="Creates a post record and returns created payload.",
    )


def test_build_embedding_text_uses_enriched_semantics_only() -> None:
    text = IndexChunkSummarizer.build_embedding_text(_chunk())

    assert "File: routes/posts.js" in text
    assert "Intent: create_resource, route_handler" in text
    assert "Summary:" in text
    assert "HTTP Metadata" in text
    assert "Method: POST" in text
    assert "Route: /posts" in text
    assert "router.post" not in text
