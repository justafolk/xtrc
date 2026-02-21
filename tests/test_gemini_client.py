import time

import pytest

import xtrc.llm.gemini_client as gemini_client
from xtrc.llm.gemini_client import GeminiClient, GeminiClientError, GeminiTimeoutError


def _install_fake_genai(monkeypatch: pytest.MonkeyPatch, responses: list[str], delay: float = 0.0) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def generate_content(self, prompt: str, generation_config: dict[str, object]) -> FakeResponse:
            _ = generation_config
            calls.append((self.model_name, prompt))
            if delay > 0:
                time.sleep(delay)
            if not responses:
                return FakeResponse('{"file": "src/default.py", "line": 1, "reason": "default"}')
            return FakeResponse(responses.pop(0))

    class FakeGenAI:
        configured_key: str | None = None

        @classmethod
        def configure(cls, api_key: str) -> None:
            cls.configured_key = api_key

        GenerativeModel = FakeModel

    monkeypatch.setattr(gemini_client, "genai", FakeGenAI)
    return calls


def test_gemini_client_caches_identical_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = ['{"file": "src/a.py", "line": 10, "reason": "match"}']
    calls = _install_fake_genai(monkeypatch, responses)

    client = GeminiClient(api_key="test", default_model="gemini-1.5-flash", timeout_seconds=1.0)
    first, _ = client.complete_json("pick best result")
    second, _ = client.complete_json("pick best result")

    assert first == second
    assert first["file"] == "src/a.py"
    assert len(calls) == 1


def test_gemini_client_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = ['{"file": "src/a.py", "line": 10, "reason": "match"}']
    _install_fake_genai(monkeypatch, responses, delay=0.05)

    client = GeminiClient(api_key="test", default_model="gemini-1.5-flash", timeout_seconds=0.01)
    with pytest.raises(GeminiTimeoutError):
        _ = client.complete_json("pick best result")


def test_gemini_client_parses_json_code_fence() -> None:
    payload = GeminiClient._parse_json_object(
        """```json
        {"file": "src/a.py", "line": 10, "reason": "match"}
        ```"""
    )
    assert payload["file"] == "src/a.py"
    assert payload["line"] == 10


def test_gemini_client_raises_on_invalid_json() -> None:
    with pytest.raises(GeminiClientError):
        _ = GeminiClient._parse_json_object("not json")
