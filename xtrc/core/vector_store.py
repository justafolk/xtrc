from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from xtrc.core.errors import AinavError
from xtrc.core.models import CodeChunk


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float
    payload: dict[str, object]


class QdrantVectorStore:
    def __init__(self, qdrant_path: Path) -> None:
        qdrant_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(qdrant_path))

    @staticmethod
    def collection_name(repo_path: str) -> str:
        digest = hashlib.sha1(repo_path.encode("utf-8")).hexdigest()[:20]
        return f"ainav_{digest}"

    @staticmethod
    def point_id(chunk_id: str) -> str:
        # Local Qdrant enforces UUID/int ids, so map stable chunk hashes to UUIDs.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def ensure_collection(self, repo_path: str, vector_size: int, recreate: bool = False) -> bool:
        collection = self.collection_name(repo_path)
        exists = self.client.collection_exists(collection_name=collection)
        recreated = False
        if recreate and exists:
            self.client.delete_collection(collection_name=collection)
            exists = False
            recreated = True

        if exists and not recreate:
            existing_size = self._collection_vector_size(collection)
            if existing_size is not None and existing_size != vector_size:
                self.client.delete_collection(collection_name=collection)
                exists = False
                recreated = True

        if not exists:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            recreated = True

        return recreated

    def _collection_vector_size(self, collection_name: str) -> int | None:
        info = self.client.get_collection(collection_name=collection_name)
        config = getattr(info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if vectors is None:
            return None

        # Single unnamed vector config
        size = getattr(vectors, "size", None)
        if isinstance(size, int):
            return size

        # Multi-vector config map
        if isinstance(vectors, dict):
            for value in vectors.values():
                v_size = getattr(value, "size", None)
                if isinstance(v_size, int):
                    return v_size
                if isinstance(value, dict) and isinstance(value.get("size"), int):
                    return int(value["size"])
        return None

    def upsert_chunks(self, repo_path: str, chunks: list[CodeChunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        collection = self.collection_name(repo_path)
        self.ensure_collection(repo_path, int(vectors.shape[1]))

        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload: dict[str, object] = {
                "chunk_id": chunk.chunk_id,
                "repo_path": chunk.repo_path,
                "file_path": chunk.file_path,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "symbol": chunk.symbol,
                "symbol_kind": chunk.symbol_kind,
                "description": chunk.description,
                "keywords": chunk.keywords,
                "symbol_terms": chunk.symbol_terms,
                "route_method": chunk.route_method,
                "route_path": chunk.route_path,
                "route_intent": chunk.route_intent,
                "route_resource": chunk.route_resource,
                "intent_tags": chunk.intent_tags,
                "structural_terms": chunk.structural_terms,
            }
            points.append(
                models.PointStruct(
                    id=self.point_id(chunk.chunk_id),
                    vector=vector.astype(np.float32).tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=collection, points=points, wait=True)

    def delete_chunk_ids(self, repo_path: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        collection = self.collection_name(repo_path)
        if not self.client.collection_exists(collection_name=collection):
            return
        point_ids = [self.point_id(chunk_id) for chunk_id in chunk_ids]
        self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=point_ids),
            wait=True,
        )

    def delete_file_chunks(self, repo_path: str, file_path: str) -> None:
        collection = self.collection_name(repo_path)
        if not self.client.collection_exists(collection_name=collection):
            return
        self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path", match=models.MatchValue(value=file_path)
                        )
                    ]
                )
            ),
            wait=True,
        )

    def search(self, repo_path: str, query_vector: np.ndarray, limit: int) -> list[SearchHit]:
        collection = self.collection_name(repo_path)
        if not self.client.collection_exists(collection_name=collection):
            return []

        expected_size = int(query_vector.shape[0])
        existing_size = self._collection_vector_size(collection)
        if existing_size is not None and existing_size != expected_size:
            raise AinavError(
                code="INDEX_DIMENSION_MISMATCH",
                message=(
                    "Indexed vectors are incompatible with current embedding model "
                    f"(index_dim={existing_size}, model_dim={expected_size}). "
                    "Run `xtrc index <repo> --rebuild`."
                ),
                status_code=409,
                details={"index_dim": existing_size, "model_dim": expected_size},
            )

        vector = query_vector.astype(np.float32).tolist()
        try:
            if hasattr(self.client, "search"):
                points = self.client.search(
                    collection_name=collection,
                    query_vector=vector,
                    limit=limit,
                    with_payload=True,
                )
            elif hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=collection,
                    query=vector,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
                points = getattr(response, "points", response)
            else:
                raise RuntimeError("Unsupported qdrant-client version: missing search/query_points")
        except ValueError as exc:
            # Older local qdrant versions may throw raw shape mismatch errors here.
            raise AinavError(
                code="INDEX_DIMENSION_MISMATCH",
                message=(
                    "Indexed vectors are incompatible with current embedding model. "
                    "Run `xtrc index <repo> --rebuild`."
                ),
                status_code=409,
            ) from exc

        hits: list[SearchHit] = []
        for point in points:
            payload = dict(point.payload or {})
            payload_chunk_id = payload.get("chunk_id")
            chunk_id = str(payload_chunk_id) if payload_chunk_id is not None else str(point.id)
            hits.append(SearchHit(chunk_id=chunk_id, score=float(point.score), payload=payload))
        return hits

    def count_chunks(self, repo_path: str) -> int:
        collection = self.collection_name(repo_path)
        if not self.client.collection_exists(collection_name=collection):
            return 0
        result = self.client.count(collection_name=collection)
        return int(result.count)
