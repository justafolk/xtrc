from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from xtrc.config import Settings, resolve_data_root
from xtrc.core.chunker import ChunkBuilder
from xtrc.core.embeddings import EmbeddingService
from xtrc.core.errors import AinavError
from xtrc.core.indexer import Indexer
from xtrc.core.metadata_store import MetadataStore
from xtrc.core.models import IndexStats, QueryOutcome, StatusStats
from xtrc.core.parser import TreeSitterCodeParser
from xtrc.core.query_engine import QueryEngine
from xtrc.core.scorer import HybridScorer
from xtrc.core.vector_store import QdrantVectorStore

if TYPE_CHECKING:
    from xtrc.indexer.summarizer import IndexChunkSummarizer
    from xtrc.llm.text_client import LLMTextClient
    from xtrc.llm.gemini_client import GeminiClient
    from xtrc.query.rerank import LocalReranker
    from xtrc.query.rewrite import QueryRewriter
    from xtrc.ranking.heuristics import RankingHeuristics
    from xtrc.llm.reranker import GeminiReranker

logger = logging.getLogger(__name__)


@dataclass
class RepoServices:
    metadata_store: MetadataStore
    parser: TreeSitterCodeParser
    chunk_builder: ChunkBuilder
    embedding_service: EmbeddingService
    vector_store: QdrantVectorStore
    indexer: Indexer
    query_engine: QueryEngine


class AinavDaemon:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._repos: dict[str, RepoServices] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._llm_text_client = self._build_llm_text_client(settings)
        self._query_rewriter = self._build_query_rewriter(settings, self._llm_text_client)
        self._local_reranker = self._build_local_reranker(settings)
        self._ranking_heuristics = self._build_ranking_heuristics(settings)
        self._gemini_client = self._build_gemini_client(settings)
        self._gemini_reranker = self._build_reranker(settings, self._gemini_client)

    def _resolve_repo_path(self, repo_path: str | Path) -> Path:
        path = Path(repo_path).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise AinavError(
                code="INVALID_REPO",
                message=f"Repository path does not exist or is not a directory: {path}",
                status_code=400,
            )
        return path

    def _get_or_create_repo_services(self, repo_path: Path) -> RepoServices:
        key = str(repo_path)
        with self._guard:
            services = self._repos.get(key)
            if services is not None:
                return services

            data_root = resolve_data_root(repo_path)
            metadata_store = MetadataStore(data_root / self.settings.sqlite_name)
            parser = TreeSitterCodeParser()
            chunk_builder = ChunkBuilder(min_tokens=200, max_tokens=800, target_tokens=500)
            embedding_service = EmbeddingService(self.settings, metadata_store)
            vector_store = QdrantVectorStore(data_root / self.settings.qdrant_dirname)
            scorer = HybridScorer()

            indexer = Indexer(
                metadata_store=metadata_store,
                parser=parser,
                chunk_builder=chunk_builder,
                embedding_service=embedding_service,
                vector_store=vector_store,
                chunk_summarizer=self._build_chunk_summarizer(metadata_store),
            )
            query_engine = QueryEngine(
                metadata_store=metadata_store,
                embedding_service=embedding_service,
                vector_store=vector_store,
                scorer=scorer,
                query_rewriter=self._query_rewriter,
                local_reranker=self._local_reranker,
                ranking_heuristics=self._ranking_heuristics,
                reranker=self._gemini_reranker,
            )

            services = RepoServices(
                metadata_store=metadata_store,
                parser=parser,
                chunk_builder=chunk_builder,
                embedding_service=embedding_service,
                vector_store=vector_store,
                indexer=indexer,
                query_engine=query_engine,
            )
            self._repos[key] = services
            self._locks[key] = threading.Lock()
            return services

    def _build_llm_text_client(self, settings: Settings) -> LLMTextClient | None:
        if not (settings.gemini_summarize_on_index or settings.query_rewrite_enabled):
            return None
        try:
            from xtrc.llm.text_client import LLMTextClient

            client = LLMTextClient(
                provider=settings.llm_provider,
                model=settings.gemini_model,
                timeout_seconds=settings.llm_timeout_seconds,
                cache_size=settings.llm_cache_size,
            )
            logger.info(
                "LLM text client enabled provider=%s timeout=%.1fs",
                settings.llm_provider,
                settings.llm_timeout_seconds,
            )
            return client
        except Exception as exc:
            logger.warning("Failed to initialize LLM text client: %s", exc)
            return None

    def _build_query_rewriter(
        self,
        settings: Settings,
        llm_text_client: LLMTextClient | None,
    ) -> QueryRewriter | None:
        if not settings.query_rewrite_enabled:
            return None
        try:
            from xtrc.query.rewrite import QueryRewriter

            model_name = settings.query_rewrite_model or settings.gemini_model
            rewriter = QueryRewriter(
                llm_client=llm_text_client,
                model_name=model_name,
                enabled=True,
                cache_size=settings.llm_cache_size,
            )
            logger.info("Query rewrite enabled model=%s", model_name)
            return rewriter
        except Exception as exc:
            logger.warning("Failed to initialize query rewriter: %s", exc)
            return None

    def _build_local_reranker(self, settings: Settings) -> LocalReranker | None:
        try:
            from xtrc.query.rerank import LocalReranker

            reranker = LocalReranker(
                enabled=settings.local_reranker_enabled,
                model_name=settings.local_reranker_model,
                max_candidates=settings.local_reranker_top_k,
                timeout_seconds=settings.llm_timeout_seconds,
            )
            if settings.local_reranker_enabled:
                logger.info(
                    "Local reranker enabled model=%s top_k=%d",
                    settings.local_reranker_model,
                    settings.local_reranker_top_k,
                )
            return reranker
        except Exception as exc:
            logger.warning("Failed to initialize local reranker: %s", exc)
            return None

    def _build_ranking_heuristics(self, settings: Settings) -> RankingHeuristics:
        from xtrc.ranking.heuristics import RankingHeuristics

        return RankingHeuristics(
            route_boost=settings.heuristic_route_boost,
            noise_penalty=settings.heuristic_noise_penalty,
            intent_boost=settings.heuristic_intent_boost,
        )

    def _build_gemini_client(self, settings: Settings) -> GeminiClient | None:
        if not (settings.use_gemini or settings.gemini_summarize_on_index):
            return None

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "Gemini features requested but GEMINI_API_KEY is not set; Gemini integrations disabled"
            )
            return None

        try:
            from xtrc.llm.gemini_client import GeminiClient

            return GeminiClient(
                api_key=api_key,
                default_model=settings.gemini_model,
                timeout_seconds=settings.gemini_timeout_seconds,
                cache_size=settings.gemini_cache_size,
            )
        except Exception as exc:
            logger.warning("Failed to initialize Gemini client: %s", exc)
            return None

    def _build_reranker(
        self,
        settings: Settings,
        client: GeminiClient | None,
    ) -> GeminiReranker | None:
        if not settings.use_gemini:
            return None
        if client is None:
            logger.warning("USE_GEMINI=true but Gemini client is unavailable; reranker disabled")
            return None

        from xtrc.llm.reranker import GeminiReranker

        logger.info(
            "Gemini reranker enabled model=%s threshold=%.2f rewrite=%s timeout=%.1fs",
            settings.gemini_model,
            settings.gemini_threshold,
            settings.gemini_enable_rewrite,
            settings.gemini_timeout_seconds,
        )
        return GeminiReranker(
            client=client,
            model_name=settings.gemini_model,
            threshold=settings.gemini_threshold,
            enable_rewrite=settings.gemini_enable_rewrite,
            max_candidates=10,
        )

    def _build_chunk_summarizer(
        self,
        metadata_store: MetadataStore,
    ) -> IndexChunkSummarizer | None:
        if not self.settings.gemini_summarize_on_index:
            return None
        if self._llm_text_client is None:
            logger.warning(
                "GEMINI_SUMMARIZE_ON_INDEX=true but LLM text client is unavailable; summarization disabled"
            )
            return None
        model_name = self.settings.gemini_summary_model or self.settings.gemini_model
        from xtrc.indexer.summarizer import IndexChunkSummarizer

        logger.info(
            "Index-time chunk summarization enabled provider=%s model=%s max_chars=%d",
            self.settings.llm_provider,
            model_name,
            self.settings.gemini_summary_max_chars,
        )
        return IndexChunkSummarizer(
            metadata_store=metadata_store,
            llm_client=self._llm_text_client,
            model_name=model_name,
            max_chars=self.settings.gemini_summary_max_chars,
        )

    def index(self, repo_path: str | Path, rebuild: bool) -> IndexStats:
        resolved = self._resolve_repo_path(repo_path)
        key = str(resolved)
        services = self._get_or_create_repo_services(resolved)
        with self._locks[key]:
            return services.indexer.index(resolved, rebuild=rebuild)

    def query(self, repo_path: str | Path, query_text: str, top_k: int) -> QueryOutcome:
        resolved = self._resolve_repo_path(repo_path)
        services = self._get_or_create_repo_services(resolved)
        return services.query_engine.query(resolved, query_text, top_k)

    def status(self, repo_path: str | Path) -> StatusStats:
        resolved = self._resolve_repo_path(repo_path)
        services = self._get_or_create_repo_services(resolved)
        return services.metadata_store.get_status(str(resolved))

    def model_name(self) -> str:
        return self.settings.model_name
