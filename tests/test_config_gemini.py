from xtrc.config import Settings


def test_settings_gemini_defaults(monkeypatch) -> None:
    monkeypatch.delenv("USE_GEMINI", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_THRESHOLD", raising=False)
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("GEMINI_ENABLE_REWRITE", raising=False)
    monkeypatch.delenv("GEMINI_CACHE_SIZE", raising=False)

    settings = Settings.from_env()

    assert settings.use_gemini is False
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.gemini_threshold == 0.85
    assert settings.gemini_timeout_seconds == 2.0
    assert settings.gemini_enable_rewrite is False
    assert settings.gemini_cache_size == 128


def test_settings_gemini_env_parsing_and_clamping(monkeypatch) -> None:
    monkeypatch.setenv("USE_GEMINI", "true")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")
    monkeypatch.setenv("GEMINI_THRESHOLD", "1.7")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("GEMINI_ENABLE_REWRITE", "1")
    monkeypatch.setenv("GEMINI_CACHE_SIZE", "-4")

    settings = Settings.from_env()

    assert settings.use_gemini is True
    assert settings.gemini_model == "gemini-1.5-pro"
    assert settings.gemini_threshold == 1.0
    assert settings.gemini_timeout_seconds == 0.1
    assert settings.gemini_enable_rewrite is True
    assert settings.gemini_cache_size == 1


def test_settings_invalid_numeric_values_fall_back_to_defaults(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_THRESHOLD", "not-a-number")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "bad")
    monkeypatch.setenv("GEMINI_CACHE_SIZE", "bad")

    settings = Settings.from_env()

    assert settings.gemini_threshold == 0.85
    assert settings.gemini_timeout_seconds == 2.0
    assert settings.gemini_cache_size == 128
