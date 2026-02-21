from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from xtrc.core.models import QueryMatch, QuerySelection
from xtrc.llm.gemini_client import GeminiClient, GeminiClientError

logger = logging.getLogger(__name__)

_RERANK_PROMPT = """You are reranking semantic code search candidates.

Task:
- Choose the single best code candidate that answers the user query.
- Only choose from the provided candidates.
- Prefer exact behavioral relevance over lexical overlap.

Return only a JSON object with this schema:
{{
  "file": "relative/path.py",
  "line": 42,
  "reason": "brief technical explanation"
}}

User Query:
{query}

Candidates (JSON):
{candidates_json}
"""

_REWRITE_PROMPT = """Rewrite this source-code search query to be more precise and technical.

Rules:
- Preserve intent.
- Mention likely symbols, behaviors, or API shape.
- Keep it as a single sentence.
- If already precise, return it unchanged.
- Return plain text only.

Query:
{query}
"""


@dataclass(frozen=True)
class RerankDecision:
    selection: QuerySelection
    used_gemini: bool
    gemini_model: str | None
    gemini_latency_ms: int | None
    rewritten_query: str | None


class GeminiReranker:
    def __init__(
        self,
        client: GeminiClient,
        *,
        model_name: str,
        threshold: float = 0.85,
        enable_rewrite: bool = False,
        max_candidates: int = 10,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.threshold = max(0.0, min(1.0, threshold))
        self.enable_rewrite = enable_rewrite
        self.max_candidates = max(1, max_candidates)

    def decide(self, query: str, matches: list[QueryMatch]) -> RerankDecision | None:
        if not matches:
            return None

        best = matches[0]
        if best.vector_score >= self.threshold:
            reason = (
                f"High vector confidence {best.vector_score:.3f} meets threshold "
                f"{self.threshold:.2f}; returning top semantic match."
            )
            return RerankDecision(
                selection=QuerySelection(
                    file=best.chunk.file_path,
                    line=best.chunk.start_line,
                    reason=reason,
                    source="vector",
                ),
                used_gemini=False,
                gemini_model=None,
                gemini_latency_ms=None,
                rewritten_query=None,
            )

        candidates = matches[: self.max_candidates]
        rewritten_query: str | None = None
        total_latency_ms = 0

        query_for_rerank = query
        if self.enable_rewrite:
            try:
                rewrite_prompt = _REWRITE_PROMPT.format(query=query)
                rewritten_query, rewrite_latency_ms = self.client.rewrite_query(
                    rewrite_prompt,
                    model_name=self.model_name,
                )
                query_for_rerank = rewritten_query
                total_latency_ms += rewrite_latency_ms
            except GeminiClientError as exc:
                logger.warning("Gemini query rewrite failed: %s", exc)

        logger.info(
            "Using Gemini reranker model=%s best_vector=%.3f threshold=%.2f",
            self.model_name,
            best.vector_score,
            self.threshold,
        )

        try:
            prompt = self._build_rerank_prompt(query_for_rerank, candidates)
            payload, rerank_latency_ms = self.client.complete_json(prompt, model_name=self.model_name)
            total_latency_ms += rerank_latency_ms
            selection = self._selection_from_payload(payload, candidates)
        except (GeminiClientError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Gemini rerank failed, falling back to top vector match: %s", exc)
            fallback_reason = (
                f"Gemini rerank failed ({exc}); falling back to top vector candidate "
                f"with score {best.vector_score:.3f}."
            )
            return RerankDecision(
                selection=QuerySelection(
                    file=best.chunk.file_path,
                    line=best.chunk.start_line,
                    reason=fallback_reason,
                    source="vector",
                ),
                used_gemini=False,
                gemini_model=self.model_name,
                gemini_latency_ms=total_latency_ms or None,
                rewritten_query=rewritten_query,
            )

        logger.info(
            "Gemini reranker selected %s:%s in %dms",
            selection.file,
            selection.line,
            total_latency_ms,
        )
        return RerankDecision(
            selection=selection,
            used_gemini=True,
            gemini_model=self.model_name,
            gemini_latency_ms=total_latency_ms,
            rewritten_query=rewritten_query,
        )

    def _build_rerank_prompt(self, query: str, candidates: list[QueryMatch]) -> str:
        serialized_candidates: list[dict[str, object]] = []
        for idx, match in enumerate(candidates, start=1):
            chunk = match.chunk
            serialized_candidates.append(
                {
                    "rank": idx,
                    "file_path": chunk.file_path,
                    "line_range": {"start": chunk.start_line, "end": chunk.end_line},
                    "code_snippet": self._truncate_snippet(chunk.text),
                    "metadata": {
                        "language": chunk.language,
                        "symbol": chunk.symbol,
                        "symbol_kind": chunk.symbol_kind,
                        "description": chunk.description,
                        "llm_summary": chunk.llm_summary,
                        "route_method": chunk.route_method,
                        "route_path": chunk.route_path,
                        "route_intent": chunk.route_intent,
                        "route_resource": chunk.route_resource,
                        "intent_tags": chunk.intent_tags,
                        "keywords": chunk.keywords,
                        "symbol_terms": chunk.symbol_terms,
                        "structural_terms": chunk.structural_terms,
                    },
                    "scores": {
                        "hybrid": round(match.score, 6),
                        "vector": round(match.vector_score, 6),
                        "keyword": round(match.keyword_score, 6),
                        "symbol": round(match.symbol_score, 6),
                    },
                }
            )

        return _RERANK_PROMPT.format(
            query=query,
            candidates_json=json.dumps(serialized_candidates, ensure_ascii=True, indent=2),
        )

    @staticmethod
    def _truncate_snippet(text: str, max_chars: int = 1800) -> str:
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}\n# ... truncated ..."

    @staticmethod
    def _selection_from_payload(payload: dict[str, object], candidates: list[QueryMatch]) -> QuerySelection:
        file_path = payload["file"]
        line = payload["line"]
        reason = payload["reason"]

        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("Gemini output must include non-empty string 'file'")
        if not isinstance(line, int) or line <= 0:
            raise ValueError("Gemini output must include positive integer 'line'")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Gemini output must include non-empty string 'reason'")

        candidate = next(
            (
                item
                for item in candidates
                if item.chunk.file_path == file_path
                and item.chunk.start_line <= line <= item.chunk.end_line
            ),
            None,
        )

        if candidate is None:
            same_file_candidates = [item for item in candidates if item.chunk.file_path == file_path]
            if same_file_candidates:
                candidate = max(same_file_candidates, key=lambda item: item.score)
                line = candidate.chunk.start_line
            else:
                raise ValueError("Gemini selected a file that is not part of the candidate list")

        return QuerySelection(
            file=candidate.chunk.file_path,
            line=line,
            reason=reason.strip(),
            source="gemini",
        )
