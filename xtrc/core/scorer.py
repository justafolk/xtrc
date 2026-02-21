from __future__ import annotations

from xtrc.core.route_signals import HTTP_INTENT_MAP, infer_query_signal
from xtrc.core.tokenizer import normalize_terms


class HybridScorer:
    VECTOR_WEIGHT = 0.50
    KEYWORD_WEIGHT = 0.18
    SYMBOL_WEIGHT = 0.12
    INTENT_WEIGHT = 0.12
    STRUCTURAL_WEIGHT = 0.08

    def score(
        self,
        query: str,
        vector_score: float,
        keywords: list[str],
        symbol_terms: list[str],
        route_intent: str | None = None,
        route_method: str | None = None,
        route_resource: str | None = None,
        structural_terms: list[str] | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        query_signal = infer_query_signal(query)
        query_terms = normalize_terms(query)
        keyword_score = self._overlap_score(query_terms, keywords)
        symbol_score = self._overlap_score(query_terms, symbol_terms)
        normalized_vector = self._normalize_vector_score(vector_score)
        intent_score = self._intent_score(query_signal.intents, route_intent, route_method)
        candidate_structural_terms = list(structural_terms or [])
        if route_method:
            candidate_structural_terms.append(route_method.lower())
        if route_intent:
            candidate_structural_terms.append(route_intent.lower())
        if route_resource:
            candidate_structural_terms.extend(normalize_terms(route_resource))
        structural_score = self._overlap_score(query_signal.structural_terms, candidate_structural_terms)

        total = (
            self.VECTOR_WEIGHT * normalized_vector
            + self.KEYWORD_WEIGHT * keyword_score
            + self.SYMBOL_WEIGHT * symbol_score
            + self.INTENT_WEIGHT * intent_score
            + self.STRUCTURAL_WEIGHT * structural_score
        )
        return (
            total,
            normalized_vector,
            keyword_score,
            symbol_score,
            intent_score,
            structural_score,
        )

    @staticmethod
    def _normalize_vector_score(score: float) -> float:
        if 0.0 <= score <= 1.0:
            return score
        # Cosine similarity can be in [-1, 1] depending on backend details.
        return max(0.0, min(1.0, (score + 1.0) / 2.0))

    @staticmethod
    def _overlap_score(query_terms: list[str], candidates: list[str]) -> float:
        if not query_terms or not candidates:
            return 0.0
        qset = set(query_terms)
        cset = set(candidates)
        overlap = len(qset.intersection(cset))
        return overlap / len(qset)

    @staticmethod
    def _intent_score(
        query_intents: list[str],
        route_intent: str | None,
        route_method: str | None,
    ) -> float:
        if not query_intents:
            return 0.0
        candidate: set[str] = set()
        if route_intent:
            candidate.add(route_intent.lower())
        if route_method:
            normalized_method = route_method.lower()
            candidate.add(normalized_method)
            mapped = HTTP_INTENT_MAP.get(normalized_method)
            if mapped:
                candidate.add(mapped)
        if not candidate:
            return 0.0
        overlap = len(set(query_intents).intersection(candidate))
        return overlap / len(set(query_intents))
