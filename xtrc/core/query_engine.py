from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from xtrc.core.embeddings import EmbeddingService
from xtrc.core.metadata_store import MetadataStore
from xtrc.core.models import QueryMatch, QueryOutcome, QuerySelection
from xtrc.core.scorer import HybridScorer
from xtrc.core.vector_store import QdrantVectorStore

if TYPE_CHECKING:
    from xtrc.llm.reranker import GeminiReranker
    from xtrc.query.rerank import LocalReranker
    from xtrc.query.rewrite import QueryRewriter
    from xtrc.ranking.heuristics import RankingHeuristics


class QueryEngine:
    def __init__(
        self,
        metadata_store: MetadataStore,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        scorer: HybridScorer,
        query_rewriter: QueryRewriter | None = None,
        local_reranker: LocalReranker | None = None,
        ranking_heuristics: RankingHeuristics | None = None,
        reranker: GeminiReranker | None = None,
    ) -> None:
        self.metadata_store = metadata_store
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.scorer = scorer
        self.query_rewriter = query_rewriter
        self.local_reranker = local_reranker
        self.ranking_heuristics = ranking_heuristics
        self.reranker = reranker

    def query(self, repo_path: Path, query_text: str, top_k: int) -> QueryOutcome:
        started = time.perf_counter()
        repo_key = str(repo_path.resolve())
        rewritten_query: str | None = None
        query_for_search = query_text
        if self.query_rewriter is not None:
            candidate_query, changed, _ = self.query_rewriter.rewrite(query_text)
            query_for_search = candidate_query
            if changed:
                rewritten_query = candidate_query

        query_embedding = self.embedding_service.embed_query(query_for_search)

        candidate_limit = max(top_k * 12, top_k)
        hits = self.vector_store.search(repo_key, query_embedding, limit=candidate_limit)
        chunk_ids = [hit.chunk_id for hit in hits]
        chunks = self.metadata_store.get_chunks_by_ids(chunk_ids)

        matches: list[QueryMatch] = []
        for hit in hits:
            chunk = chunks.get(hit.chunk_id)
            if chunk is None:
                continue
            (
                total,
                normalized_vector,
                keyword_score,
                symbol_score,
                intent_score,
                structural_score,
            ) = self.scorer.score(
                query=query_for_search,
                vector_score=hit.score,
                keywords=chunk.keywords,
                symbol_terms=chunk.symbol_terms,
                route_intent=chunk.route_intent,
                route_method=chunk.route_method,
                route_resource=chunk.route_resource,
                structural_terms=chunk.structural_terms,
            )
            matched_intents: list[str] = []
            matched_keywords: list[str] = []
            explanation_bits = [
                f"semantic={normalized_vector:.3f}",
                f"keyword={keyword_score:.3f}",
                f"symbol={symbol_score:.3f}",
                f"intent={intent_score:.3f}",
                f"structural={structural_score:.3f}",
            ]
            adjusted_total = total
            if self.ranking_heuristics is not None:
                decision = self.ranking_heuristics.evaluate(query_for_search, chunk)
                adjusted_total = total * decision.multiplier
                matched_intents = decision.matched_intents
                matched_keywords = decision.matched_keywords
                if decision.reasons:
                    explanation_bits.append("heuristics=" + ", ".join(decision.reasons))

            matches.append(
                QueryMatch(
                    chunk=chunk,
                    score=adjusted_total,
                    vector_score=normalized_vector,
                    keyword_score=keyword_score,
                    symbol_score=symbol_score,
                    intent_score=intent_score,
                    structural_score=structural_score,
                    matched_intents=matched_intents,
                    matched_keywords=matched_keywords,
                    explanation="; ".join(explanation_bits),
                )
            )

        matches.sort(
            key=lambda item: (
                item.score,
                item.vector_score,
                1 if item.chunk.symbol else 0,
                -item.chunk.tokens,
            ),
            reverse=True,
        )
        if self.local_reranker is not None and matches:
            reranked, _, _ = self.local_reranker.rerank(query_for_search, matches[:10])
            if len(matches) > 10:
                reranked.extend(matches[10:])
            matches = reranked

        selection: QuerySelection | None = None
        used_gemini = False
        gemini_model: str | None = None
        gemini_latency_ms: int | None = None

        if matches:
            if self.reranker is None:
                top = matches[0]
                selection = QuerySelection(
                    file=top.chunk.file_path,
                    line=top.chunk.start_line,
                    reason=(
                        "Gemini reranker is disabled; returning highest scoring semantic result."
                    ),
                    source="vector",
                )
            else:
                decision = self.reranker.decide(query_for_search, matches)
                if decision is not None:
                    selection = decision.selection
                    used_gemini = decision.used_gemini
                    gemini_model = decision.gemini_model
                    gemini_latency_ms = decision.gemini_latency_ms
                    if decision.rewritten_query:
                        rewritten_query = decision.rewritten_query

        duration_ms = int((time.perf_counter() - started) * 1000)
        return QueryOutcome(
            matches=matches[:top_k],
            duration_ms=duration_ms,
            selection=selection,
            used_gemini=used_gemini,
            gemini_model=gemini_model,
            gemini_latency_ms=gemini_latency_ms,
            rewritten_query=rewritten_query,
        )
