"""Everything that talks to Qdrant lives here (the "repository" layer).

The rest of the app asks for things in its own terms ("search the dense
vectors", "which chunks are stored") and never builds Qdrant requests itself.
Swapping Qdrant for another vector DB would mean rewriting this file only.

The collection holds one POINT per chunk:
  id       the chunk's stable UUID (same chunk -> same id -> upsert overwrites)
  vectors  "dense": 768 numbers from Gemini (meaning)
           "bm25":  sparse {term_id: weight} (exact words)
  payload  the chunk's text and metadata (citation, paragraph, S/G, pages, tags)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from app.services.bm25_encoder import SparseVec

DENSE = "dense"
SPARSE = "bm25"
UPSERT_BATCH = 128
DEFAULT_EXCLUDED_TAGS = ("form_template",)  # blank forms: indexed, but hidden from search by default
KEYWORD_INDEXES = ("doc_code", "provision", "tags", "chunk_type", "scope")


@dataclass(frozen=True, slots=True)
class StoredPoint:
    id: str
    dense: list[float]
    sparse: SparseVec
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    score: float
    payload: dict[str, Any]


def _exclude_tags(tags: Sequence[str]) -> models.Filter | None:
    if not tags:
        return None
    return models.Filter(must_not=[models.FieldCondition(key="tags", match=models.MatchAny(any=list(tags)))])


class VectorRepository:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    # ------------------------------------------------------------ schema

    def ensure_collection(self, dense_size: int, recreate: bool = False) -> None:
        """Create the collection if needed. If it exists with a different
        vector size (e.g. you changed the embedding model), stop loudly
        instead of mixing incompatible vectors."""
        exists = self._client.collection_exists(self._collection)
        if exists and recreate:
            self._client.delete_collection(self._collection)
            exists = False
        if exists:
            current = self._client.get_collection(self._collection).config.params.vectors[DENSE].size
            if current != dense_size:
                raise RuntimeError(
                    f"Collection '{self._collection}' has {current}-dim vectors, not {dense_size}. "
                    "Re-run with --recreate to rebuild it."
                )
            return
        self._client.create_collection(
            self._collection,
            vectors_config={DENSE: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)},
            # Modifier.IDF: Qdrant computes BM25's "how rare is this word" part itself.
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        # Payload indexes make filters ("only S provisions", "only e-KYC") fast.
        for field in KEYWORD_INDEXES:
            self._client.create_payload_index(self._collection, field, models.PayloadSchemaType.KEYWORD)

    # ------------------------------------------------------------ writes

    def upsert(self, points: Iterable[StoredPoint]) -> int:
        batch: list[models.PointStruct] = []
        written = 0
        for point in points:
            batch.append(
                models.PointStruct(
                    id=point.id,
                    vector={
                        DENSE: point.dense,
                        SPARSE: models.SparseVector(indices=point.sparse.indices, values=point.sparse.values),
                    },
                    payload=point.payload,
                )
            )
            if len(batch) == UPSERT_BATCH:
                self._client.upsert(self._collection, points=batch, wait=True)
                written += len(batch)
                batch = []
        if batch:
            self._client.upsert(self._collection, points=batch, wait=True)
            written += len(batch)
        return written

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self._client.delete(self._collection, points_selector=models.PointIdsList(points=list(ids)), wait=True)

    # ------------------------------------------------------------ reads

    def stored_ids(self) -> set[str]:
        ids: set[str] = set()
        offset = None
        while True:
            records, offset = self._client.scroll(
                self._collection, limit=1000, offset=offset, with_payload=False, with_vectors=False
            )
            ids.update(str(record.id) for record in records)
            if offset is None:
                return ids

    def count(self) -> int:
        return self._client.count(self._collection, exact=True).count

    def search_dense(
        self, vector: list[float], limit: int = 20, exclude_tags: Sequence[str] = DEFAULT_EXCLUDED_TAGS
    ) -> list[SearchHit]:
        return self._query(vector, DENSE, limit, exclude_tags)

    def search_sparse(
        self, vector: SparseVec, limit: int = 20, exclude_tags: Sequence[str] = DEFAULT_EXCLUDED_TAGS
    ) -> list[SearchHit]:
        if not vector.indices:  # a query of only stopwords has no terms to match
            return []
        query = models.SparseVector(indices=vector.indices, values=vector.values)
        return self._query(query, SPARSE, limit, exclude_tags)

    def _query(self, query: Any, using: str, limit: int, exclude_tags: Sequence[str]) -> list[SearchHit]:
        response = self._client.query_points(
            self._collection,
            query=query,
            using=using,
            limit=limit,
            query_filter=_exclude_tags(exclude_tags),
            with_payload=True,
        )
        return [SearchHit(id=str(p.id), score=p.score, payload=p.payload or {}) for p in response.points]
