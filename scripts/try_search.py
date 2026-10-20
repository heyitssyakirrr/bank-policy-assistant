"""See hybrid retrieval at work on one question.

    python -m scripts.try_search "What must a bank do if FAR exceeds 5%?"
    python -m scripts.try_search "14A.9.6"

Prints three lists:
  DENSE  meaning-based search on its own
  BM25   exact-word search on its own
  FUSED  what the app actually uses: exact paragraph matches first, then RRF
         of the two lists. Each line shows where that chunk ranked in each
         search, e.g. [D3 B4] = #3 in dense, #4 in BM25; [D- B1] = BM25 only.
A learning/debugging tool, not the app. Costs one Gemini embedding call.
"""

import sys

from qdrant_client import QdrantClient

from app.core.config import settings
from app.db.vector_repository import SearchHit, VectorRepository
from app.services.bm25_encoder import tokenize
from app.services.embedding_service import embed_query
from app.services.retrieval_service import RankedChunk, RetrievalService

TOP_K = 5


def _snippet(payload: dict) -> str:
    return payload["text"].replace("\n", " ")[:120] + "…"


def show_hits(title: str, hits: list[SearchHit]) -> None:
    print(f"\n{title}")
    for rank, hit in enumerate(hits[:TOP_K], start=1):
        print(f"  {rank}. [{hit.score:.3f}] {hit.payload['citation']}\n       {_snippet(hit.payload)}")
    if not hits:
        print("  (no results)")


def show_fused(chunks: list[RankedChunk]) -> None:
    print("\nFUSED — exact paragraph matches first, then RRF (what the reranker will receive)")
    for rank, chunk in enumerate(chunks[:TOP_K], start=1):
        where = "EXACT" if chunk.exact_match else f"rrf {chunk.rrf_score:.4f}"
        ranks = f"D{chunk.dense_rank or '-'} B{chunk.sparse_rank or '-'}"
        print(f"  {rank}. [{where}] [{ranks}] {chunk.citation}\n       {_snippet(chunk.payload)}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    question = " ".join(sys.argv[1:])
    service = RetrievalService(
        VectorRepository(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection), embed_query
    )
    result = service.retrieve(question)

    print(f"Question: {question}")
    print(f"BM25 tokens: {tokenize(question)}   paragraph refs: {result.paragraph_refs or '-'}")
    show_hits("DENSE (meaning) — cosine similarity", result.dense)
    show_hits("BM25 (exact words) — BM25 score", result.sparse)
    show_fused(result.chunks)
    print(f"\nembed {result.timings_ms['embed']:.0f} ms, search {result.timings_ms['search']:.0f} ms")


if __name__ == "__main__":
    main()
