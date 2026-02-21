from __future__ import annotations

import logging
import time
from pathlib import Path

from xtrc.core.chunker import ChunkBuilder
from xtrc.core.embeddings import EmbeddingService
from xtrc.core.metadata_store import MetadataStore
from xtrc.core.models import IndexStats
from xtrc.core.parser import TreeSitterCodeParser
from xtrc.core.repo import detect_language, sha256_text, walk_source_files
from xtrc.core.vector_store import QdrantVectorStore
from xtrc.indexer.summarizer import IndexChunkSummarizer

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(
        self,
        metadata_store: MetadataStore,
        parser: TreeSitterCodeParser,
        chunk_builder: ChunkBuilder,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        chunk_summarizer: IndexChunkSummarizer | None = None,
    ) -> None:
        self.metadata_store = metadata_store
        self.parser = parser
        self.chunk_builder = chunk_builder
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chunk_summarizer = chunk_summarizer

    def index(self, repo_path: Path, rebuild: bool = False) -> IndexStats:
        started = time.perf_counter()
        repo_path = repo_path.resolve()
        repo_key = str(repo_path)

        files = walk_source_files(repo_path)
        files_scanned = len(files)
        logger.info("Index start repo=%s files=%d rebuild=%s", repo_key, files_scanned, rebuild)

        vector_size = self.embedding_service.dimension
        if rebuild:
            self.metadata_store.clear_repo(repo_key)
            self.vector_store.ensure_collection(repo_key, vector_size, recreate=True)
        else:
            recreated = self.vector_store.ensure_collection(repo_key, vector_size, recreate=False)
            if recreated:
                logger.warning(
                    "Vector collection was missing or incompatible for repo=%s; forcing full rebuild",
                    repo_key,
                )
                self.metadata_store.clear_repo(repo_key)
                rebuild = True

        known_hashes = self.metadata_store.get_file_hashes(repo_key)
        current_paths: set[str] = set()

        changed_files: list[tuple[Path, str, str, str, str]] = []
        for file_path in files:
            language = detect_language(file_path)
            if language is None:
                continue
            relative_path = str(file_path.relative_to(repo_path))
            current_paths.add(relative_path)

            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logger.warning("Skipping unreadable file %s: %s", file_path, exc)
                continue

            file_hash = sha256_text(text)
            if not rebuild and known_hashes.get(relative_path) == file_hash:
                continue
            changed_files.append((file_path, relative_path, language, text, file_hash))

        deleted_files = sorted(set(known_hashes.keys()) - current_paths)
        for relative_path in deleted_files:
            self.vector_store.delete_file_chunks(repo_key, relative_path)
            self.metadata_store.delete_chunks_by_file(repo_key, relative_path)
        self.metadata_store.delete_files(repo_key, deleted_files)

        files_indexed = 0
        chunks_indexed = 0

        for file_path, relative_path, language, text, file_hash in changed_files:
            old_chunk_ids = self.metadata_store.get_chunk_ids_for_file(repo_key, relative_path)
            if old_chunk_ids:
                self.vector_store.delete_chunk_ids(repo_key, old_chunk_ids)
                self.metadata_store.delete_chunks_by_ids(old_chunk_ids)

            symbols = self.parser.parse_symbols(file_path, language, text)
            chunks = self.chunk_builder.build_chunks(
                repo_path=repo_path,
                file_path=file_path,
                language=language,
                file_hash=file_hash,
                content=text,
                symbols=symbols,
            )

            if chunks:
                if self.chunk_summarizer is not None:
                    summaries, summary_latency_ms = self.chunk_summarizer.summarize_chunks(chunks)
                    if summaries:
                        chunks = self.chunk_summarizer.apply_summaries(chunks, summaries)
                        logger.info(
                            "Applied cached/generated summaries for %s chunks=%d latency_ms=%d",
                            relative_path,
                            len(summaries),
                            summary_latency_ms,
                        )

                embedding_input = [IndexChunkSummarizer.build_embedding_text(chunk) for chunk in chunks]
                vectors = self.embedding_service.embed_documents(embedding_input).vectors
                self.vector_store.upsert_chunks(repo_key, chunks, vectors)
                self.metadata_store.upsert_chunks(chunks)
                chunks_indexed += len(chunks)

            self.metadata_store.upsert_file_hash(repo_key, relative_path, file_hash)
            files_indexed += 1

        self.metadata_store.set_repo_last_indexed(repo_key)

        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "Index finished repo=%s scanned=%d indexed=%d deleted=%d chunks=%d duration_ms=%d",
            repo_key,
            files_scanned,
            files_indexed,
            len(deleted_files),
            chunks_indexed,
            duration_ms,
        )

        return IndexStats(
            repo_path=repo_key,
            files_scanned=files_scanned,
            files_indexed=files_indexed,
            files_deleted=len(deleted_files),
            chunks_indexed=chunks_indexed,
            duration_ms=duration_ms,
        )
