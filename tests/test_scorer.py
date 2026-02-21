from xtrc.core.scorer import HybridScorer


def test_hybrid_scoring_weights_are_applied() -> None:
    scorer = HybridScorer()
    total, vector, keyword, symbol, intent, structural = scorer.score(
        query="get user score",
        vector_score=0.5,
        keywords=["get", "user", "score"],
        symbol_terms=["get", "user", "score"],
    )

    assert round(vector, 3) == 0.5
    assert keyword == 1.0
    assert symbol == 1.0
    assert intent == 0.0
    assert structural == 0.0
    assert round(total, 3) == round(0.5 * 0.5 + 0.18 + 0.12, 3)


def test_hybrid_scoring_boosts_route_intent_and_structure() -> None:
    scorer = HybridScorer()
    baseline, *_ = scorer.score(
        query="create post endpoint",
        vector_score=0.4,
        keywords=["handler"],
        symbol_terms=["handler"],
    )
    boosted, _, _, _, intent, structural = scorer.score(
        query="create post endpoint",
        vector_score=0.4,
        keywords=["handler", "post", "create"],
        symbol_terms=["create_post", "post"],
        route_intent="create",
        route_method="POST",
        route_resource="post",
        structural_terms=["create", "post", "posts"],
    )

    assert intent > 0.0
    assert structural > 0.0
    assert boosted > baseline
