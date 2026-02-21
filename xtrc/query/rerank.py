from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import replace

from xtrc.core.models import QueryMatch

logger = logging.getLogger(__name__)


class LocalReranker:
    def __init__(
        self,
        *,
        enabled: bool,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_candidates: int = 10,
        timeout_seconds: float = 2.0,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.max_candidates = max(1, max_candidates)
        self.timeout_seconds = max(0.1, timeout_seconds)
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xtrc-local-rerank")

    def rerank(self, query: str, matches: list[QueryMatch]) -> tuple[list[QueryMatch], bool, int | None]:
        if not self.enabled or len(matches) <= 1:
            return matches, False, None

        target = matches[: self.max_candidates]
        remainder = matches[self.max_candidates :]

        try:
            started = time.perf_counter()
            scores = self._predict_scores(query, target)
            latency_ms = int((time.perf_counter() - started) * 1000)
        except Exception as exc:
            logger.warning("Local reranker skipped due to failure: %s", exc)
            return matches, False, None

        reranked: list[QueryMatch] = []
        for match, local_score in zip(target, scores, strict=True):
            combined = 0.7 * match.score + 0.3 * self._sigmoid(local_score)
            explanation = match.explanation
            if explanation:
                explanation = f"{explanation}; local reranker score={local_score:.3f}"
            else:
                explanation = f"local reranker score={local_score:.3f}"
            reranked.append(replace(match, score=combined, explanation=explanation))

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked + remainder, True, latency_ms

    def _predict_scores(self, query: str, matches: list[QueryMatch]) -> list[float]:
        if not matches:
            return []

        pairs = [(query, self._candidate_text(match)) for match in matches]
        future = self._executor.submit(self._predict_blocking, pairs)
        try:
            raw_scores = future.result(timeout=self.timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise RuntimeError(f"Local reranker timed out after {self.timeout_seconds:.1f}s") from exc

        if not isinstance(raw_scores, list):
            raise RuntimeError("Local reranker returned invalid score format")
        return [float(item) for item in raw_scores]

    def _predict_blocking(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._load_model()
        scores = model.predict(pairs)
        if hasattr(scores, "tolist"):
            values = scores.tolist()
            if isinstance(values, list):
                return [float(item) for item in values]
        if isinstance(scores, list):
            return [float(item) for item in scores]
        return [float(scores)]

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    @staticmethod
    def _candidate_text(match: QueryMatch) -> str:
        summary = match.chunk.llm_summary or match.chunk.description
        lines = [
            f"file: {match.chunk.file_path}",
            f"symbol: {match.chunk.symbol or '-'}",
            f"type: {match.chunk.symbol_kind or 'major_block'}",
            f"intent: {', '.join(match.chunk.intent_tags) or 'unknown'}",
            f"summary: {summary}",
        ]
        if match.chunk.route_method or match.chunk.route_path:
            lines.append(f"http: {match.chunk.route_method or '-'} {match.chunk.route_path or '-'}")
        return "\n".join(lines)

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))
