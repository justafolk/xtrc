from xtrc.config import Settings


def test_query_quality_features_default_to_latency_first(monkeypatch) -> None:
    monkeypatch.delenv("QUERY_REWRITE_ENABLED", raising=False)
    monkeypatch.delenv("LOCAL_RERANKER_ENABLED", raising=False)

    settings = Settings.from_env()

    assert settings.query_rewrite_enabled is False
    assert settings.local_reranker_enabled is False


def test_query_quality_features_can_be_enabled_with_env(monkeypatch) -> None:
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "true")
    monkeypatch.setenv("LOCAL_RERANKER_ENABLED", "true")

    settings = Settings.from_env()

    assert settings.query_rewrite_enabled is True
    assert settings.local_reranker_enabled is True
