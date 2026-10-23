"""Hybrid retrieval: dense + BM25 search, fused with RRF, plus exact paragraph lookup.

    question
      ├─ paragraph numbers in it? ("14A.9.6") ──► fetch those paragraphs directly
      ├─ dense search (meaning)  ─┐
      └─ BM25 search (words)     ─┴─► RRF fusion ──► ranked candidates (for the reranker)

Why RRF (Reciprocal Rank Fusion)
--------------------------------
Dense scores (cosine, ~0.6-0.8) and BM25 scores (~2-30) are on different
scales, so adding them is meaningless. RRF uses only each chunk's RANK:

    rrf_score(chunk) = sum over the lists it appears in of  1 / (k + rank)

A chunk ranked well by BOTH searches beats one ranked well by only one.
k (60) flattens the curve so rank 1 doesn't drown out ranks 2-10.

Why the exact paragraph lookup
------------------------------
Ranking is the wrong tool when the user names the thing they want. For
"14A.9.6", dense search doesn't find 14A.9.6 at all (a number has no
"meaning"), so RRF ranks 14A.9.7 (#1 dense, #2 BM25) above it. When the
question contains a paragraph number, we fetch that paragraph by its
metadata and put it first. Real systems call this query routing.

Paragraph numbers repeat across documents ("8.2" is in all three), so a
reference carries its document when it is known: the agent's planner
supplies `(document, paragraph_id)` pairs; the traditional mode finds bare
numbers in the question with a regex, limited to the documents searched.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.db.vector_repository import DEFAULT_EXCLUDED_TAGS, SearchHit, VectorRepository
from app.services.bm25_encoder import Bm25Encoder

logger = logging.getLogger("app.retrieval")

# "8.10", "14A.9.6", "para 29.1.1"; not "5%", "3.5%", "RM3.5" or a version like "1.2.3.4.5"
PARAGRAPH_REF_RE = re.compile(r"(?<![\w.])(\d{1,2}[A-Za-z]?(?:\.\d{1,2}){1,3})(?![\w.%])")


@dataclass(frozen=True, slots=True)
class ParagraphLookup:
    document: str | None  # doc_code; None = any document being searched
    paragraph_id: str


@dataclass(frozen=True, slots=True)
class RankedChunk:
    id: str
    payload: dict[str, Any]
    rrf_score: float
    dense_rank: int | None  # 1-based position in the dense list, None if absent
    sparse_rank: int | None  # 1-based position in the BM25 list, None if absent
    exact_match: bool = False  # fetched because the question named its paragraph number

    @property
    def citation(self) -> str:
        return self.payload.get("citation", self.id)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    question: str
    chunks: list[RankedChunk]  # final order: exact matches first, then RRF
    dense: list[SearchHit]  # raw lists, kept for debugging and evaluation
    sparse: list[SearchHit]
    paragraph_refs: list[ParagraphLookup]
    timings_ms: dict[str, float]
    dense_used: bool = True  # False = the query embedding failed: BM25 + exact lookup only


def find_paragraph_refs(question: str) -> list[str]:
    """Paragraph numbers mentioned in the question, normalised to how they're
    stored ("14a.9.6" -> "14A.9.6"), in order, without duplicates."""
    return list(dict.fromkeys(ref.upper() for ref in PARAGRAPH_REF_RE.findall(question)))


def rrf_fuse(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over lists of ids (best first).

    Returns [(id, score)] best first. Ties are broken by the best single rank,
    then by id, so the output is deterministic (same input -> same order)."""
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            best_rank[item_id] = min(rank, best_rank.get(item_id, rank))
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))


class RetrievalService:
    def __init__(
        self,
        repo: VectorRepository,
        embed_query: Callable[[str], list[float]],
        candidates: int = settings.retrieval_candidates,
        rrf_k: int = settings.rrf_k,
    ) -> None:
        self._repo = repo
        self._embed_query = embed_query  # injected: the real Gemini call in the app, a fake in tests
        self._candidates = candidates
        self._rrf_k = rrf_k

    def retrieve(
        self,
        question: str,
        limit: int | None = None,
        exclude_tags: Sequence[str] = DEFAULT_EXCLUDED_TAGS,
        documents: Sequence[str] = (),
        paragraph_refs: Sequence[ParagraphLookup] | None = None,
    ) -> RetrievalResult:
        """`documents` limits every search ([] = all). `paragraph_refs=None`
        finds paragraph numbers in the question; pass [] to skip the lookup."""
        limit = limit or self._candidates
        timings: dict[str, float] = {}

        # 1. Exact paragraph lookup (cheap metadata filter, no ranking involved).
        refs = (
            [ParagraphLookup(None, ref) for ref in find_paragraph_refs(question)]
            if paragraph_refs is None
            else list(paragraph_refs)
        )
        exact = self.lookup(refs, documents, exclude_tags)

        # 2. Encode the question both ways. Dense needs a Gemini call; BM25 is local.
        started = time.perf_counter()
        try:
            dense_vector: list[float] | None = self._embed_query(question)
        except Exception as exc:  # degrade to BM25 + exact lookup rather than fail the question
            logger.warning(f"Query embedding failed ({type(exc).__name__}: {exc}); BM25 only")
            dense_vector = None
        timings["embed"] = (time.perf_counter() - started) * 1000
        sparse_vector = Bm25Encoder.encode_query(question)

        # 3. Both searches in one Qdrant round trip.
        started = time.perf_counter()
        dense, sparse = self._repo.search_hybrid(dense_vector, sparse_vector, self._candidates, exclude_tags, documents)
        timings["search"] = (time.perf_counter() - started) * 1000

        # 4. Fuse, then put exact matches on top.
        chunks = self._combine(exact, dense, sparse, limit)
        logger.info(
            f"retrieve: {len(dense)} dense + {len(sparse)} bm25 -> {len(chunks)} fused"
            f"{f' ({len(exact)} exact for {[r.paragraph_id for r in refs]})' if refs else ''}"
            f" | embed {timings['embed']:.0f}ms, search {timings['search']:.0f}ms"
        )
        return RetrievalResult(question, chunks, dense, sparse, refs, timings, dense_used=dense_vector is not None)

    def lookup(
        self,
        refs: Sequence[ParagraphLookup],
        documents: Sequence[str] = (),
        exclude_tags: Sequence[str] = DEFAULT_EXCLUDED_TAGS,
    ) -> list[SearchHit]:
        """Fetch named paragraphs. A reference without a document is looked up
        in `documents` (all documents if empty). One Qdrant call per document."""
        by_document: dict[str | None, list[str]] = defaultdict(list)
        for ref in refs:
            by_document[ref.document].append(ref.paragraph_id.upper())
        hits: list[SearchHit] = []
        for document, ids in by_document.items():
            scope = [document] if document else list(documents)
            hits += self._repo.get_by_paragraph_ids(list(dict.fromkeys(ids)), exclude_tags, documents=scope)
        return hits

    def _combine(
        self, exact: list[SearchHit], dense: list[SearchHit], sparse: list[SearchHit], limit: int
    ) -> list[RankedChunk]:
        dense_rank = {hit.id: rank for rank, hit in enumerate(dense, start=1)}
        sparse_rank = {hit.id: rank for rank, hit in enumerate(sparse, start=1)}
        payloads = {hit.id: hit.payload for hit in (*exact, *dense, *sparse)}
        fused = rrf_fuse([[h.id for h in dense], [h.id for h in sparse]], k=self._rrf_k)
        fused_score = dict(fused)

        def ranked(chunk_id: str, exact_match: bool = False) -> RankedChunk:
            return RankedChunk(
                id=chunk_id,
                payload=payloads[chunk_id],
                rrf_score=fused_score.get(chunk_id, 0.0),
                dense_rank=dense_rank.get(chunk_id),
                sparse_rank=sparse_rank.get(chunk_id),
                exact_match=exact_match,
            )

        exact_ids = list(dict.fromkeys(hit.id for hit in exact))  # unique, order kept
        seen = set(exact_ids)
        results = [ranked(chunk_id, exact_match=True) for chunk_id in exact_ids]
        results += [ranked(chunk_id) for chunk_id, _ in fused if chunk_id not in seen]
        return results[:limit]
