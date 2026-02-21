from __future__ import annotations

import hashlib
import logging
from dataclasses import replace

from xtrc.core.metadata_store import MetadataStore
from xtrc.core.models import CodeChunk
from xtrc.llm.text_client import LLMClientError, LLMTextClient

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """Summarize this backend code chunk for semantic retrieval.

Requirements:
- 1 to 2 short sentences.
- Explain purpose and business intent.
- Mention API behavior and data mutations when present.
- Avoid implementation details and low-level syntax.
- No markdown.

File: {file_path}
Symbol: {symbol}
Type: {chunk_type}
Route Method: {method}
Route Path: {path}
Intent Tags: {intent_tags}

Code:
{code}
"""


class IndexChunkSummarizer:
    def __init__(
        self,
        *,
        metadata_store: MetadataStore,
        llm_client: LLMTextClient | None,
        model_name: str,
        max_chars: int = 400,
    ) -> None:
        self.metadata_store = metadata_store
        self.llm_client = llm_client
        self.model_name = model_name
        self.max_chars = max(80, max_chars)

    def summarize_chunks(self, chunks: list[CodeChunk]) -> tuple[dict[str, str], int]:
        if not chunks:
            return {}, 0

        keys = {chunk.chunk_id: self._summary_key(chunk) for chunk in chunks}
        cached = self.metadata_store.get_cached_chunk_summaries(list(keys.values()))

        summaries: dict[str, str] = {}
        to_store: dict[str, str] = {}
        latency_ms = 0

        for chunk in chunks:
            key = keys[chunk.chunk_id]
            cached_summary = cached.get(key)
            if cached_summary:
                summaries[chunk.chunk_id] = cached_summary
                continue

            if self.llm_client is None:
                continue

            prompt = _SUMMARY_PROMPT.format(
                file_path=chunk.file_path,
                symbol=chunk.symbol or "-",
                chunk_type=chunk.symbol_kind or "major_block",
                method=chunk.route_method or "-",
                path=chunk.route_path or "-",
                intent_tags=", ".join(chunk.intent_tags) or "none",
                code=self._truncate_code(chunk.text),
            )
            try:
                text, elapsed = self.llm_client.complete_text(prompt, model_name=self.model_name)
            except LLMClientError as exc:
                logger.warning("Chunk summarization failed for %s: %s", chunk.chunk_id, exc)
                continue

            clean = self._clean_summary(text)
            if not clean:
                continue
            summaries[chunk.chunk_id] = clean
            to_store[key] = clean
            latency_ms += elapsed

        if to_store:
            self.metadata_store.upsert_cached_chunk_summaries(self.model_name, to_store)

        return summaries, latency_ms

    @staticmethod
    def apply_summaries(chunks: list[CodeChunk], summaries: dict[str, str]) -> list[CodeChunk]:
        if not summaries:
            return chunks
        return [replace(chunk, llm_summary=summaries.get(chunk.chunk_id, chunk.llm_summary)) for chunk in chunks]

    @staticmethod
    def build_embedding_text(chunk: CodeChunk) -> str:
        intent_line = ", ".join(chunk.intent_tags) or "unknown"
        parts = [
            f"File: {chunk.file_path}",
            f"Symbol: {chunk.symbol or '-'}",
            f"Type: {chunk.symbol_kind or 'major_block'}",
            f"Intent: {intent_line}",
            "",
            "Summary:",
            chunk.llm_summary or chunk.description,
        ]

        if chunk.route_method or chunk.route_path:
            parts.extend(
                [
                    "",
                    "HTTP Metadata (if applicable):",
                    f"Method: {chunk.route_method or '-'}",
                    f"Route: {chunk.route_path or '-'}",
                ]
            )

        return "\n".join(parts).strip()

    def _summary_key(self, chunk: CodeChunk) -> str:
        material = f"{self.model_name}|{chunk.chunk_id}|{chunk.content_hash}|{chunk.text}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _truncate_code(text: str, max_chars: int = 2600) -> str:
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}\n# ... truncated ..."

    def _clean_summary(self, text: str) -> str:
        normalized = " ".join(text.strip().split())
        if len(normalized) <= self.max_chars:
            return normalized
        return f"{normalized[: self.max_chars - 3].rstrip()}..."
