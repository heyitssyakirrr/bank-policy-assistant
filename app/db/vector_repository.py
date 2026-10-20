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
KEYWORD_INDEXES = ("doc_code", "provision", "tags", "chunk_type", "scope", "paragraph_id", "sector")
MODEL_FIELD = "embedding_model"  # stored on every point: which model made its dense vector


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


def _filter(exclude_tags: Sequence[str] = (), **must_any: Sequence[str]) -> models.Filter | None:
    """Build a Qdrant filter: hide chunks with any of `exclude_tags`, and keep
    only chunks whose field matches one of the given values, e.g.
    _filter(("form_template",), paragraph_id=["8.10"])."""
    must = [models.FieldCondition(key=key, match=models.MatchAny(any=list(values))) for key, values in must_any.items()]
    must_not = (
        [models.FieldCondition(key="tags", match=models.MatchAny(any=list(exclude_tags)))] if exclude_tags else []
    )
    return models.Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None


def _hits(points: Iterable[Any]) -> list[SearchHit]:
    return [SearchHit(id=str(p.id), score=getattr(p, "score", 0.0) or 0.0, payload=p.payload or {}) for p in points]


class VectorRepository:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    # ------------------------------------------------------------ schema

    def ensure_collection(self, dense_size: int, embedding_model: str = "", recreate: bool = False) -> None:
        """Create the collection if needed. If it already holds vectors from a
        different model or of a different size, stop loudly: vectors from two
        embedding models live in different "spaces" and mixing them silently
        ruins search (both Gemini models use 768 dims, so size alone can't tell)."""
        exists = self._client.collection_exists(self._collection)
        if exists and recreate:
            self._client.delete_collection(self._collection)
            exists = False
        if exists:
            current = self._client.get_collection(self._collection).config.params.vectors[DENSE].size
            if current != dense_size:
                raise RuntimeError(
                    f"Collection '{self._collection}' has {current}-dim vectors, not {dense_size}. "
                    "Re-run with --recreate, or set QDRANT_COLLECTION to a new name."
                )
            stored_model = self._stored_model()
            if embedding_model and stored_model and stored_model != embedding_model:
                raise RuntimeError(
                    f"Collection '{self._collection}' was built with {stored_model}, not {embedding_model}. "
                    "Use a separate collection per model: set QDRANT_COLLECTION (e.g. policy_chunks_v2)."
                )
            self._ensure_payload_indexes()  # adds indexes introduced after the collection was created
            return
        self._client.create_collection(
            self._collection,
            vectors_config={DENSE: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)},
            # Modifier.IDF: Qdrant computes BM25's "how rare is this word" part itself.
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Payload indexes make filters ("only S provisions", "paragraph 8.10")
        fast. Creating one that already exists is a no-op."""
        for field in KEYWORD_INDEXES:
            self._client.create_payload_index(self._collection, field, models.PayloadSchemaType.KEYWORD)

    def _stored_model(self) -> str | None:
        records, _ = self._client.scroll(self._collection, limit=1, with_payload=[MODEL_FIELD], with_vectors=False)
        return records[0].payload.get(MODEL_FIELD) if records and records[0].payload else None

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
        return self.search_hybrid(vector, None, limit, exclude_tags)[0]

    def search_sparse(
        self, vector: SparseVec, limit: int = 20, exclude_tags: Sequence[str] = DEFAULT_EXCLUDED_TAGS
    ) -> list[SearchHit]:
        return self.search_hybrid(None, vector, limit, exclude_tags)[1]

    def search_hybrid(
        self,
        dense: list[float] | None,
        sparse: SparseVec | None,
        limit: int = 20,
        exclude_tags: Sequence[str] = DEFAULT_EXCLUDED_TAGS,
    ) -> tuple[list[SearchHit], list[SearchHit]]:
        """Run the dense and BM25 searches in ONE round trip (Qdrant's batch
        query) and return both ranked lists. Fusing them is the caller's job:
        this layer stores and finds, it doesn't decide relevance."""
        query_filter = _filter(exclude_tags)
        requests: list[models.QueryRequest] = []
        if dense is not None:
            requests.append(
                models.QueryRequest(query=dense, using=DENSE, limit=limit, filter=query_filter, with_payload=True)
            )
        has_terms = sparse is not None and bool(sparse.indices)  # a query of only stopwords matches nothing
        if has_terms:
            sparse_query = models.SparseVector(indices=sparse.indices, values=sparse.values)
            requests.append(
                models.QueryRequest(
                    query=sparse_query, using=SPARSE, limit=limit, filter=query_filter, with_payload=True
                )
            )
        if not requests:
            return [], []
        responses = iter(self._client.query_batch_points(self._collection, requests=requests))
        dense_hits = _hits(next(responses).points) if dense is not None else []
        sparse_hits = _hits(next(responses).points) if has_terms else []
        return dense_hits, sparse_hits

    def get_by_paragraph_ids(
        self, paragraph_ids: Sequence[str], exclude_tags: Sequence[str] = DEFAULT_EXCLUDED_TAGS, limit: int = 20
    ) -> list[SearchHit]:
        """Exact lookup by paragraph number (a filter, not a similarity search)."""
        if not paragraph_ids:
            return []
        records, _ = self._client.scroll(
            self._collection,
            scroll_filter=_filter(exclude_tags, paragraph_id=paragraph_ids),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return _hits(records)
