from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from xtrc.config import Settings
from xtrc.core.metadata_store import MetadataStore


@dataclass(frozen=True)
class EmbeddingResult:
    keys: list[str]
    vectors: np.ndarray


class EmbeddingService:
    def __init__(self, settings: Settings, metadata_store: MetadataStore) -> None:
        self.settings = settings
        self.metadata_store = metadata_store
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        self._memory_cache: dict[str, np.ndarray] = {}
        self._dimension: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = SentenceTransformer(self.settings.model_name)
                    self._dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            _ = self.model
        assert self._dimension is not None
        return self._dimension

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        prepared = [self._prepare_document_text(text) for text in texts]
        return self.embed_texts(prepared)

    def embed_query(self, query: str) -> np.ndarray:
        prepared = self._prepare_query_text(query)
        return self.embed_texts([prepared]).vectors[0]

    def _prepare_document_text(self, text: str) -> str:
        model_name = self.settings.model_name.lower()
        if "e5" in model_name:
            return text if text.startswith("passage: ") else f"passage: {text}"
        return text

    def _prepare_query_text(self, text: str) -> str:
        model_name = self.settings.model_name.lower()
        if "bge" in model_name:
            prefix = "Represent this sentence for searching relevant passages: "
            return text if text.startswith(prefix) else f"{prefix}{text}"
        if "e5" in model_name:
            return text if text.startswith("query: ") else f"query: {text}"
        return text

    def embed_texts(self, texts: list[str], keys: list[str] | None = None) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(keys=[], vectors=np.empty((0, self.dimension), dtype=np.float32))

        lookup_keys = keys or [self.hash_text(text) for text in texts]
        if len(lookup_keys) != len(texts):
            raise ValueError("keys and texts length mismatch")

        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing_items: list[tuple[int, str, str]] = []

        for idx, key in enumerate(lookup_keys):
            memory_vector = self._memory_cache.get(key)
            if memory_vector is not None:
                vectors[idx] = memory_vector
            else:
                missing_items.append((idx, key, texts[idx]))

        if missing_items:
            missing_keys = [item[1] for item in missing_items]
            persisted = self.metadata_store.get_cached_embeddings(missing_keys)
            unresolved: list[tuple[int, str, str]] = []
            for idx, key, text in missing_items:
                persisted_vec = persisted.get(key)
                if persisted_vec is not None:
                    vectors[idx] = persisted_vec
                    self._memory_cache[key] = persisted_vec
                else:
                    unresolved.append((idx, key, text))

            if unresolved:
                encoded = self.model.encode(
                    [item[2] for item in unresolved],
                    batch_size=min(self.settings.max_batch_size, len(unresolved)),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                if encoded.ndim == 1:
                    encoded = np.expand_dims(encoded, axis=0)

                to_persist: dict[str, np.ndarray] = {}
                for (idx, key, _), vector in zip(unresolved, encoded, strict=True):
                    vec = np.asarray(vector, dtype=np.float32)
                    vectors[idx] = vec
                    self._memory_cache[key] = vec
                    to_persist[key] = vec
                self.metadata_store.upsert_cached_embeddings(to_persist)

        if any(vec is None for vec in vectors):
            raise RuntimeError("Embedding resolution failed for one or more texts")
        output = np.stack([vec for vec in vectors if vec is not None]).astype(np.float32)
        return EmbeddingResult(keys=lookup_keys, vectors=output)
