from xtrc.core.models import CodeChunk
from xtrc.ranking.heuristics import RankingHeuristics


def _chunk(**overrides):
    base = dict(
        chunk_id="c1",
        repo_path="/tmp/repo",
        file_path="routes/posts.js",
        language="javascript",
        start_line=1,
        end_line=8,
        symbol="POST /posts",
        symbol_kind="route",
        description="Route handler",
        text="router.post('/posts', createPostHandler)",
        content_hash="h",
        tokens=10,
        keywords=["post", "create", "handler"],
        symbol_terms=["post", "create"],
        route_method="POST",
        route_path="/posts",
        route_intent="create",
        route_resource="post",
        intent_tags=["create_resource", "route_handler"],
        structural_terms=["post", "create", "endpoint"],
        llm_summary="Creates a post resource.",
    )
    base.update(overrides)
    return CodeChunk(**base)


def test_route_and_intent_boost_applied() -> None:
    heuristics = RankingHeuristics(route_boost=1.3, noise_penalty=0.7, intent_boost=1.2)
    decision = heuristics.evaluate("create post api endpoint", _chunk())

    assert decision.multiplier > 1.0
    assert "create_resource" in decision.matched_intents
    assert "route handler boost" in ", ".join(decision.reasons)


def test_noise_penalty_applied() -> None:
    heuristics = RankingHeuristics(route_boost=1.3, noise_penalty=0.7, intent_boost=1.2)
    decision = heuristics.evaluate(
        "create post api endpoint",
        _chunk(intent_tags=["create_resource", "test_script"], file_path="tests/test_posts.py"),
    )

    assert decision.multiplier < 1.3 * 1.2
    assert "noise/script penalty" in ", ".join(decision.reasons)
