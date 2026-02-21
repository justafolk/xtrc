from __future__ import annotations

import logging
from functools import lru_cache

from xtrc.llm.text_client import LLMClientError, LLMTextClient

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """Rewrite this code search query into precise backend intent.

Rules:
- Keep original user intent.
- Mention endpoint/handler behavior when applicable.
- Include CRUD action and likely HTTP semantics if implied.
- Keep to one sentence.
- Return plain text only.

Query:
{query}
"""


class QueryRewriter:
    def __init__(
        self,
        *,
        llm_client: LLMTextClient | None,
        model_name: str,
        enabled: bool,
        cache_size: int = 256,
    ) -> None:
        self.llm_client = llm_client
        self.model_name = model_name
        self.enabled = enabled
        self._cached_rewrite = lru_cache(maxsize=max(1, cache_size))(self._rewrite_uncached)

    def rewrite(self, query: str) -> tuple[str, bool, int | None]:
        if not self.enabled or self.llm_client is None:
            return query, False, None
        normalized = query.strip()
        if not normalized:
            return query, False, None

        try:
            rewritten, latency = self._cached_rewrite(normalized)
        except LLMClientError as exc:
            logger.warning("Query rewrite failed: %s", exc)
            return query, False, None

        if not rewritten:
            return query, False, None
        return rewritten, rewritten != query, latency

    def _rewrite_uncached(self, query: str) -> tuple[str, int]:
        if self.llm_client is None:
            return query, 0
        prompt = _REWRITE_PROMPT.format(query=query)
        rewritten, latency = self.llm_client.complete_text(prompt, model_name=self.model_name)
        cleaned = self._clean_rewrite(rewritten)
        return cleaned or query, latency

    @staticmethod
    def _clean_rewrite(text: str) -> str:
        line = " ".join(text.strip().split())
        if not line:
            return ""
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].strip()
        if len(line) > 220:
            line = f"{line[:217].rstrip()}..."
        return line
