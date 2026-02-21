from __future__ import annotations

from dataclasses import dataclass

from xtrc.core.models import CodeChunk
from xtrc.core.route_signals import infer_query_signal
from xtrc.core.tokenizer import normalize_terms

_ROUTE_QUERY_HINTS = {"create", "post", "api", "endpoint", "route"}
_NEGATIVE_INTENTS = {"seed_data", "migration_script", "test_script", "script"}


@dataclass(frozen=True)
class HeuristicDecision:
    multiplier: float
    matched_intents: list[str]
    matched_keywords: list[str]
    reasons: list[str]


class RankingHeuristics:
    def __init__(
        self,
        *,
        route_boost: float = 1.3,
        noise_penalty: float = 0.7,
        intent_boost: float = 1.2,
    ) -> None:
        self.route_boost = max(0.1, route_boost)
        self.noise_penalty = max(0.1, noise_penalty)
        self.intent_boost = max(0.1, intent_boost)

    def evaluate(self, query: str, chunk: CodeChunk) -> HeuristicDecision:
        query_terms = set(normalize_terms(query))
        query_signal = infer_query_signal(query)

        multiplier = 1.0
        reasons: list[str] = []

        matched_intents = self._matched_intents(query_signal.intents, chunk.intent_tags)
        if matched_intents:
            multiplier *= self.intent_boost
            reasons.append(f"intent match: {', '.join(matched_intents)}")

        if query_terms.intersection(_ROUTE_QUERY_HINTS) and self._is_route_chunk(chunk):
            multiplier *= self.route_boost
            reasons.append("route handler boost")

        if _NEGATIVE_INTENTS.intersection(set(chunk.intent_tags)):
            multiplier *= self.noise_penalty
            reasons.append("noise/script penalty")

        matched_keywords = self._matched_keywords(query_terms, chunk)

        return HeuristicDecision(
            multiplier=multiplier,
            matched_intents=matched_intents,
            matched_keywords=matched_keywords,
            reasons=reasons,
        )

    @staticmethod
    def _is_route_chunk(chunk: CodeChunk) -> bool:
        return bool(chunk.route_method) or "route_handler" in chunk.intent_tags or chunk.symbol_kind == "route"

    @staticmethod
    def _matched_intents(query_intents: list[str], tags: list[str]) -> list[str]:
        if not query_intents:
            return []
        tag_set = set(tags)
        matched: list[str] = []
        for intent in query_intents:
            key = f"{intent}_resource"
            if key in tag_set:
                matched.append(key)
        return sorted(set(matched))

    @staticmethod
    def _matched_keywords(query_terms: set[str], chunk: CodeChunk) -> list[str]:
        candidate_terms = set(chunk.keywords)
        candidate_terms.update(chunk.symbol_terms)
        candidate_terms.update(chunk.structural_terms)
        if chunk.route_method:
            candidate_terms.add(chunk.route_method.lower())
        if chunk.route_resource:
            candidate_terms.update(normalize_terms(chunk.route_resource))
        overlap = sorted(query_terms.intersection(candidate_terms))
        return overlap[:8]
