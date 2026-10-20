"""The full search: retrieve -> rerank -> select. This is what the chat calls.

    question
      └─ RetrievalService   exact paragraph lookup + dense + BM25 + RRF  -> ~30 candidates
          └─ Reranker       scores every candidate against the question   -> best first
              └─ select_final  one copy per repeated sector rule,
                               named paragraphs guaranteed               -> final top 5

If the reranker fails (quota used up, Cohere down), the search still answers
using the fused order and says so (`reranked=False`). This is "graceful
degradation": a slightly worse answer beats an error page, and the flag
lets logs, monitoring and evaluation see it happened.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.core.config import settings
from app.services.rerank_service import Reranker
from app.services.retrieval_service import RetrievalResult, RetrievalService
from app.services.selection import Candidate, Selection, detect_sector, select_final

logger = logging.getLogger("app.search")


@dataclass(frozen=True, slots=True)
class SearchResult:
    question: str
    final: list[Candidate]  # what the LLM will read, best first
    candidates: list[Candidate]  # every candidate after reranking (for debugging/evaluation)
    selection: Selection
    retrieval: RetrievalResult
    preferred_sector: str
    reranked: bool  # False = reranker failed, fused order used instead
    timings_ms: dict[str, float]


class SearchService:
    def __init__(
        self,
        retrieval: RetrievalService,
        reranker: Reranker,
        top_n: int = settings.rerank_top_n,
        default_sector: str = settings.default_sector,
    ) -> None:
        self._retrieval = retrieval
        self._reranker = reranker
        self._top_n = top_n
        self._default_sector = default_sector

    def search(self, question: str) -> SearchResult:
        retrieval = self._retrieval.retrieve(question)
        candidates = [Candidate(chunk, fused_rank) for fused_rank, chunk in enumerate(retrieval.chunks, start=1)]

        started = time.perf_counter()
        candidates, reranked = self._rerank(question, candidates)
        rerank_ms = (time.perf_counter() - started) * 1000

        preferred_sector = detect_sector(question) or self._default_sector
        selection = select_final(candidates, self._top_n, preferred_sector)
        logger.info(
            f"search: {len(candidates)} candidates -> {len(selection.final)} final "
            f"({'reranked' if reranked else 'NOT reranked, fused order'}; "
            f"{len(selection.collapsed)} sector duplicates collapsed, prefer {preferred_sector}) | rerank {rerank_ms:.0f}ms"
        )
        return SearchResult(
            question=question,
            final=selection.final,
            candidates=candidates,
            selection=selection,
            retrieval=retrieval,
            preferred_sector=preferred_sector,
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