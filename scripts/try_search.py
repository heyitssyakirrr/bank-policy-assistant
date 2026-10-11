"""Compare dense (meaning) and BM25 (exact words) search on one question.

    python -m scripts.try_search "What must a bank do if FAR exceeds 5%?"
    python -m scripts.try_search "14A.9.6"

This is a learning/debugging tool, not the app. It shows the two searches
SEPARATELY so you can see why hybrid search exists: each one finds things
the other misses. Step 2 fuses them (RRF); step 3 reranks the result.
Costs one Gemini embedding call per question.
"""

import sys

from qdrant_client import QdrantClient

from app.core.config import settings
from app.db.vector_repository import SearchHit, VectorRepository
from app.services.bm25_encoder import Bm25Encoder, tokenize
from app.services.embedding_service import embed_query

TOP_K = 5


def show(title: str, hits: list[SearchHit]) -> None:
    print(f"\n{title}")
    if not hits:
        print("  (no results)")
    for rank, hit in enumerate(hits, start=1):
        text = hit.payload["text"].replace("\n", " ")
        print(f"  {rank}. [{hit.score:.3f}] {hit.payload['citation']}\n       {text[:140]}…")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    question = " ".join(sys.argv[1:])
    repo = VectorRepository(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection)

    print(f"Question: {question}")
    print(f"BM25 tokens: {tokenize(question)}")
    show("DENSE (meaning) — cosine similarity, higher = closer", repo.search_dense(embed_query(question), TOP_K))
    show(
        "BM25 (exact words) — BM25 score, higher = better match",
        repo.search_sparse(Bm25Encoder.encode_query(question), TOP_K),
    )


if __name__ == "__main__":
    main()
