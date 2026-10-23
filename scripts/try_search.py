"""Watch the whole search pipeline work on one question.

    python -m scripts.try_search "What must a bank do if FAR exceeds 5%?"
    python -m scripts.try_search "14A.9.6" --raw      # also show the raw dense/BM25 lists
    python -m scripts.try_search "remote onboarding" --sector insurance_takaful --documents amlcft

Shows:
  FUSED   the top candidates after RRF, with each one's dense/BM25 rank
          ([D3 B2] = #3 dense, #2 BM25; [D- B1] = BM25 only)
  FINAL   after reranking + selection: what the LLM will read. Each line
          shows the move, e.g. "fused #7 -> #1", and the reranker's score.
  COLLAPSED  repeated sector rules that were merged into one copy
A learning/debugging tool, not the app. Costs 1 Gemini + 1 Cohere call.
"""

import argparse

from app.agent.schemas import DOCUMENT_TITLES, SECTOR_LABELS
from app.core.config import settings
from app.db.vector_repository import SearchHit
from app.services.bm25_encoder import tokenize
from app.services.search_service import build_search_service
from app.services.selection import Candidate

FUSED_SHOWN = 10


def _snippet(payload: dict, width: int = 110) -> str:
    return payload["text"].replace("\n", " ")[:width] + "…"


def _ranks(candidate: Candidate) -> str:
    chunk = candidate.chunk
    return "EXACT" if chunk.exact_match else f"D{chunk.dense_rank or '-'} B{chunk.sparse_rank or '-'}"


def show_raw(title: str, hits: list[SearchHit]) -> None:
    print(f"\n{title}")
    for rank, hit in enumerate(hits[:5], start=1):
        print(f"  {rank}. [{hit.score:.3f}] {hit.payload['citation']}")


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("question", nargs="+")
    cli.add_argument("--raw", action="store_true", help="also print the raw dense and BM25 lists")
    cli.add_argument("--sector", choices=[*SECTOR_LABELS, "all"], default=settings.default_sector)
    cli.add_argument("--documents", nargs="*", choices=list(DOCUMENT_TITLES), default=[])
    args = cli.parse_args()
    question = " ".join(args.question)

    sector = None if args.sector == "all" else args.sector
    result = build_search_service().search(question, sector=sector, documents=args.documents)
    retrieval = result.retrieval

    print(f"Question: {question}")
    refs = [r.paragraph_id for r in retrieval.paragraph_refs]
    print(f"BM25 tokens: {tokenize(question)} | paragraph refs: {refs or '-'}")
    print(f"Preferred sector: {result.preferred_sector or 'none (all sectors)'} | documents: {args.documents or 'all'}")
    if args.raw:
        show_raw("DENSE (meaning)", retrieval.dense)
        show_raw("BM25 (exact words)", retrieval.sparse)

    print(f"\nFUSED — top {FUSED_SHOWN} of {len(result.candidates)} candidates sent to the reranker")
    for chunk_rank, chunk in enumerate(retrieval.chunks[:FUSED_SHOWN], start=1):
        tag = "EXACT" if chunk.exact_match else f"D{chunk.dense_rank or '-'} B{chunk.sparse_rank or '-'}"
        print(f"  {chunk_rank:>2}. [{tag:>8}] {chunk.citation}")

    status = f"reranked by {settings.rerank_model}" if result.reranked else "RERANKER FAILED: fused order used"
    print(f"\nFINAL — what the LLM will read ({status})")
    for rank, cand in enumerate(result.final, start=1):
        score = f"{cand.rerank_score:.3f}" if cand.rerank_score is not None else "  -  "
        print(f"  {rank}. [score {score}] [fused #{cand.fused_rank} -> #{rank}] [{_ranks(cand)}] {cand.chunk.citation}")
        print(f"       {_snippet(cand.chunk.payload)}")

    if result.selection.collapsed:
        print("\nCOLLAPSED — repeated sector rules, one copy kept")
        for dropped, kept in result.selection.collapsed:
            print(
                f"  dropped {dropped.chunk.payload.get('paragraph_id')} ({dropped.sector}) "
                f"-> kept {kept.chunk.payload.get('paragraph_id')} ({kept.sector})"
            )

    demoted_in_top = [c for c in result.selection.demoted[:5]]
    if demoted_in_top:
        print(f"\nDEMOTED — rules for other sectors, ranked below rules for {result.preferred_sector}")
        for cand in demoted_in_top:
            print(f"  {cand.chunk.payload.get('paragraph_id')} ({cand.sector})")

    t = result.timings_ms
    print(f"\nembed {t['embed']:.0f} ms | search {t['search']:.0f} ms | rerank {t['rerank']:.0f} ms")


if __name__ == "__main__":
    main()
