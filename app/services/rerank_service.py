"""Reranking: re-score retrieval candidates by how well each one ANSWERS the question.

Why a reranker
--------------
Dense and BM25 search are fast because the question and each chunk are
encoded SEPARATELY (vectors computed in advance, compared with arithmetic).
They never read the two together. A reranker (a "cross-encoder") reads the
question and one chunk side by side and judges relevance, which is far more
accurate, but too slow to run on the whole corpus. So:

    retrieval: 1,219 chunks -> ~30 candidates   (cheap, high recall)
    reranker:  ~30 candidates -> scored, best first   (expensive, high precision)

The `Reranker` protocol is the contract the rest of the app depends on.
Cohere is one implementation; a local cross-encoder model could be another,
and tests use a fake. Nothing outside this file knows Cohere exists.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.rate_limit import MinuteRateLimiter
from app.core.retry import retry_transient

logger = logging.getLogger("app.rerank")

MAX_TOKENS_PER_DOC = 1024  # our chunks are <= ~500 tokens; caps cost if one ever isn't


@dataclass(frozen=True, slots=True)
class RerankScore:
    index: int  # position of the document in the list that was sent
    score: float  # relevance, 0..1 for Cohere; higher = better answer to the question


class Reranker(Protocol):
    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        """Score every document; return them best first."""
        ...


class CohereReranker:
    def __init__(self, api_key: str, model: str, calls_per_minute: int, client: Any | None = None) -> None:
        if client is None:
            if not api_key:
                raise ValueError("COHERE_API_KEY is not set (add it to .env)")
            import cohere  # imported here: only needed when this reranker is actually used

            client = cohere.ClientV2(api_key=api_key)
        self._client = client
        self._model = model
        # One rerank call = one request, whatever the number of documents.
        self._limiter = MinuteRateLimiter(max_items_per_minute=calls_per_minute, max_tokens_per_minute=10**9)

    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        if not documents:
            return []
        self._limiter.acquire(items=1, tokens=0)
        response = self._call(query, list(documents))
        return [RerankScore(index=r.index, score=r.relevance_score) for r in response.results]

    # 3 attempts, not 6: a user is waiting, and if Cohere stays down the search
    # falls back to the fused order (see search_service) instead of failing.
    @retry_transient(max_attempts=3)
    def _call(self, query: str, documents: list[str]) -> Any:
        return self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=len(documents),  # score ALL: selection (dedupe, exact matches) happens after
            max_tokens_per_doc=MAX_TOKENS_PER_DOC,
        )
