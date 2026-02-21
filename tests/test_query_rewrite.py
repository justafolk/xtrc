from xtrc.query.rewrite import QueryRewriter


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete_text(self, prompt: str, *, model_name: str | None = None) -> tuple[str, int]:
        _ = prompt
        _ = model_name
        self.calls += 1
        return "backend POST route or function that creates user posts", 12


def test_query_rewriter_rewrites_and_caches() -> None:
    llm = FakeLLM()
    rewriter = QueryRewriter(llm_client=llm, model_name="m", enabled=True, cache_size=64)

    first, changed1, latency1 = rewriter.rewrite("function to create new posts")
    second, changed2, latency2 = rewriter.rewrite("function to create new posts")

    assert first == "backend POST route or function that creates user posts"
    assert second == first
    assert changed1 is True
    assert changed2 is True
    assert latency1 == 12
    assert latency2 == 12
    assert llm.calls == 1


def test_query_rewriter_fallback_when_disabled() -> None:
    rewriter = QueryRewriter(llm_client=None, model_name="m", enabled=False)
    query, changed, latency = rewriter.rewrite("find post creation")

    assert query == "find post creation"
    assert changed is False
    assert latency is None
