"""The full search: retrieve -> rerank -> select. This is what the chat calls.

    query (+ documents filter, preferred sector)
      └─ RetrievalService   exact paragraph lookup + dense + BM25 + RRF  -> ~30 candidates
          └─ Reranker       scores every candidate against the question   -> best first
              └─ select_final  one copy per repeated sector rule,
                               named paragraphs guaranteed               -> final top 5

If the reranker fails (quota used up, Cohere down), the search still answers
using the fused order and says so (`reranked=False`). This is "graceful
degradation": a slightly worse answer beats an error page, and the flag
lets logs, monitoring and evaluation see it happened.

The preferred sector is the caller's decision (the planner's reading of the
question, else the user's sidebar choice); None means "show all sectors".
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.core.config import settings
from app.db.vector_repository import VectorRepository, connect_qdrant
from app.llm.cache import ResponseCache
from app.services.embedding_service import cached_embed_query, embed_query
from app.services.rerank_service import CachedReranker, CohereReranker, NoReranker, Reranker
from app.services.retrieval_service import ParagraphLookup, RankedChunk, RetrievalResult, RetrievalService
from app.services.selection import Candidate, Selection, select_final

logger = logging.getLogger("app.search")


@dataclass(frozen=True, slots=True)
class SearchResult:
    question: str
    final: list[Candidate]  # what the LLM will read, best first
    candidates: list[Candidate]  # every candidate after reranking (for debugging/evaluation)
    selection: Selection
    retrieval: RetrievalResult
    preferred_sector: str | None
    documents: list[str]  # the documents searched ([] = all)
    reranked: bool  # False = reranker failed, fused order used instead
    timings_ms: dict[str, float]


class SearchService:
    def __init__(
        self,
        retrieval: RetrievalService,
        reranker: Reranker,
        top_n: int = settings.rerank_top_n,
    ) -> None:
        self._retrieval = retrieval
        self._reranker = reranker
        self._top_n = top_n

    def search(
        self,
        question: str,
        sector: str | None,
        documents: Sequence[str] = (),
        paragraph_refs: Sequence[ParagraphLookup] | None = None,
    ) -> SearchResult:
        """`paragraph_refs=None` finds paragraph numbers in the question (the
        traditional mode); the agent passes [] and looks paragraphs up itself."""
        retrieval = self._retrieval.retrieve(question, documents=documents, paragraph_refs=paragraph_refs)
        candidates = [Candidate(chunk, fused_rank) for fused_rank, chunk in enumerate(retrieval.chunks, start=1)]

        started = time.perf_counter()
        candidates, reranked = self._rerank(question, candidates)
        rerank_ms = (time.perf_counter() - started) * 1000

        selection = select_final(candidates, self._top_n, sector)
        logger.info(
            f"search: {len(candidates)} candidates -> {len(selection.final)} final "
            f"({'reranked' if reranked else 'NOT reranked, fused order'}; "
            f"{len(selection.collapsed)} sector duplicates collapsed, prefer {sector or 'no sector'}) | rerank {rerank_ms:.0f}ms"
        )
        return SearchResult(
            question=question,
            final=selection.final,
            candidates=candidates,
            selection=selection,
            retrieval=retrieval,
            preferred_sector=sector,
            documents=list(documents),
            reranked=reranked,
            timings_ms={**retrieval.timings_ms, "rerank": rerank_ms},
        )

    def _rerank(self, question: str, candidates: list[Candidate]) -> tuple[list[Candidate], bool]:
        if not candidates:
            return candidates, True
        # embed_text = context header + text + footnotes: the reranker sees which
        # document/section a chunk is from, not just the bare paragraph.
        documents = [c.chunk.payload["embed_text"] for c in candidates]
        try:
            scores = self._reranker.rerank(question, documents)
        except Exception as exc:  # any failure: degrade, don't crash the user's question
            logger.warning(f"Reranker failed ({type(exc).__name__}: {exc}); using fused order")
            return candidates, False
        ordered = sorted(scores, key=lambda s: s.score, reverse=True)  # don't rely on the API's order
        return [Candidate(candidates[s.index].chunk, candidates[s.index].fused_rank, s.score) for s in ordered], True

    def lookup(self, refs: Sequence[ParagraphLookup]) -> list[Candidate]:
        """Named paragraphs, fetched directly (the agent's lookup tool)."""
        hits = self._retrieval.lookup(refs)
        return [
            Candidate(RankedChunk(hit.id, hit.payload, 0.0, None, None, exact_match=True), rank)
            for rank, hit in enumerate(hits, start=1)
        ]


@dataclass(frozen=True, slots=True)
class SearchParts:
    """The pieces of the search, also used one by one by the retrieval ablation."""

    repo: VectorRepository
    embed: Callable[[str], list[float]]
    reranker: Reranker


def build_search_parts(cache: ResponseCache | None = None) -> SearchParts:
    """Qdrant + Gemini query embeddings + Cohere. With a cache (development
    and evaluation), repeated queries cost nothing."""
    repo = VectorRepository(connect_qdrant(), settings.qdrant_collection)
    reranker: Reranker
    if settings.cohere_api_key:
        reranker = CohereReranker(settings.cohere_api_key, settings.rerank_model, settings.rerank_calls_per_minute)
        if cache is not None:
            reranker = CachedReranker(reranker, cache, settings.rerank_model)
    else:
        logger.warning("COHERE_API_KEY is not set: searches use the fused order, not reranked")
        reranker = NoReranker()
    embed = cached_embed_query(cache) if cache is not None else embed_query
    return SearchParts(repo, embed, reranker)


def build_search_service(cache: ResponseCache | None = None) -> SearchService:
    """The real search stack (see build_search_parts)."""
    parts = build_search_parts(cache)
    return SearchService(RetrievalService(parts.repo, parts.embed), parts.reranker)
