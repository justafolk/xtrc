from __future__ import annotations

import hashlib
import logging
from dataclasses import replace

from xtrc.core.metadata_store import MetadataStore
from xtrc.core.models import CodeChunk
from xtrc.llm.gemini_client import GeminiClient, GeminiClientError

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """Summarize this code chunk for retrieval indexing.

Rules:
- One concise sentence.
- Focus on behavior, side effects, and domain intent.
- Mention HTTP semantics if present (create/update/delete/read, method, resource).
- No markdown.
- Max 40 words.

Language: {language}
File: {file_path}
Symbol: {symbol}
Description: {description}

Code:
{code}
"""


class GeminiChunkSummarizer:
    def __init__(
        self,
        metadata_store: MetadataStore,
        client: GeminiClient,
        *,
        model_name: str,
        max_chars: int = 320,
    ) -> None:
        self.metadata_store = metadata_store
        self.client = client
        self.model_name = model_name
        self.max_chars = max(64, max_chars)

    def summarize_chunks(self, chunks: list[CodeChunk]) -> tuple[dict[str, str], int]:
        if not chunks:
            return {}, 0

        key_by_chunk_id = {chunk.chunk_id: self._summary_key(chunk) for chunk in chunks}
        cached = self.metadata_store.get_cached_chunk_summaries(list(key_by_chunk_id.values()))

        summaries_by_chunk_id: dict[str, str] = {}
        to_persist: dict[str, str] = {}
        total_latency_ms = 0

        for chunk in chunks:
            summary_key = key_by_chunk_id[chunk.chunk_id]
            cached_summary = cached.get(summary_key)
            if cached_summary:
                summaries_by_chunk_id[chunk.chunk_id] = cached_summary
                continue

            prompt = _SUMMARY_PROMPT.format(
                language=chunk.language,
                file_path=chunk.file_path,
                symbol=chunk.symbol or "-",
                description=chunk.description,
                code=self._truncate_code(chunk.text),
            )
            try:
                summary, latency_ms = self.client.complete_text(prompt, model_name=self.model_name)
            except GeminiClientError as exc:
                logger.warning("Gemini chunk summary failed for %s: %s", chunk.chunk_id, exc)
                continue

            cleaned = self._clean_summary(summary)
            if not cleaned:
                continue
            summaries_by_chunk_id[chunk.chunk_id] = cleaned
            to_persist[summary_key] = cleaned
            total_latency_ms += latency_ms

        if to_persist:
            self.metadata_store.upsert_cached_chunk_summaries(self.model_name, to_persist)

        return summaries_by_chunk_id, total_latency_ms

    @staticmethod
    def apply_summaries(chunks: list[CodeChunk], summaries: dict[str, str]) -> list[CodeChunk]:
        if not summaries:
            return chunks
        updated: list[CodeChunk] = []
        for chunk in chunks:
            summary = summaries.get(chunk.chunk_id)
            if summary:
                updated.append(replace(chunk, llm_summary=summary))
            else:
                updated.append(chunk)
        return updated

    @staticmethod
    def _summary_key(chunk: CodeChunk) -> str:
        material = (
            f"{chunk.language}|{chunk.route_method or ''}|{chunk.route_intent or ''}|"
            f"{chunk.route_resource or ''}|{chunk.text}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _truncate_code(text: str, limit: int = 2400) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}\n# ... truncated ..."

    def _clean_summary(self, text: str) -> str:
        one_line = " ".join(text.strip().split())
        if len(one_line) <= self.max_chars:
            return one_line
        return one_line[: self.max_chars - 3].rstrip() + "..."
