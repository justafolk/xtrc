from pathlib import Path

from xtrc.core.chunker import ChunkBuilder
from xtrc.core.models import SymbolBlock


def test_chunk_builder_emits_chunks_with_metadata() -> None:
    builder = ChunkBuilder(min_tokens=20, max_tokens=80, target_tokens=40)
    content = "\n".join([f"def f{i}():\n    return {i}" for i in range(12)])

    symbols = [
        SymbolBlock(kind="function", name=f"f{i}", start_line=2 * i + 1, end_line=2 * i + 2, text="")
        for i in range(12)
    ]

    chunks = builder.build_chunks(
        repo_path=Path("/tmp/repo"),
        file_path=Path("/tmp/repo/main.py"),
        language="python",
        file_hash="abc123",
        content=content,
        symbols=symbols,
    )

    assert chunks
    assert all(chunk.file_path == "main.py" for chunk in chunks)
    assert all(chunk.start_line <= chunk.end_line for chunk in chunks)
    assert all(chunk.description for chunk in chunks)


def test_chunk_builder_extracts_route_intent_metadata() -> None:
    builder = ChunkBuilder(min_tokens=5, max_tokens=120, target_tokens=60)
    content = "router.post('/posts', createPostHandler)"
    symbols = [
        SymbolBlock(
            kind="route",
            name="POST /posts",
            start_line=1,
            end_line=1,
            text=content,
        )
    ]

    chunks = builder.build_chunks(
        repo_path=Path("/tmp/repo"),
        file_path=Path("/tmp/repo/routes/posts.js"),
        language="javascript",
        file_hash="abc123",
        content=content,
        symbols=symbols,
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.route_method == "POST"
    assert chunk.route_intent == "create"
    assert chunk.route_resource == "post"
    assert "create" in chunk.structural_terms
    assert "post" in chunk.structural_terms
