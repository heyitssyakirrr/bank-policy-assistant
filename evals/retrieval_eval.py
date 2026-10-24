"""Retrieval evaluation and the ablation that proves each optimisation helps.

For every single-turn answerable item, the same question goes through the
search one stage at a time. Every stage is scored the same way: its first 5
chunks (what the generator would read) against the gold paragraphs
(Hit@5, Recall@5, MRR@5):

    dense only        meaning
    BM25 only         exact words
    hybrid (RRF)      both, fused by rank
    + exact lookup    paragraph numbers in the question fetched directly
    + rerank          Cohere cross-encoder over the fused candidates
    + selection       sector copies collapsed, other sectors demoted (= the traditional mode's search)

The agentic mode's evidence is scored by the answer evaluation and added as
a last row. Cost per question: 1 query embedding + 1 rerank, both cached,
so every stage reuses them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from app.core.config import settings
from app.services.bm25_encoder import Bm25Encoder
from app.services.retrieval_service import RetrievalService, rrf_fuse
from app.services.search_service import SearchParts, SearchService
from evals.golden import GoldenItem, paragraph_key
from evals.metrics import hit_at_k, mean, recall_at_k, reciprocal_rank

TOP_K = 5
STAGES = ("dense only", "BM25 only", "hybrid (RRF)", "+ exact lookup", "+ rerank", "+ selection")


class StageScore(BaseModel):
    item_id: str
    category: str
    stage: str
    ranked: list[str]  # the first 5 chunks' paragraph keys, best first
    hit: float
    recall: float
    mrr: float


def score(item: GoldenItem, stage: str, ranked: Sequence[str]) -> StageScore:
    gold = item.expected_paragraphs
    top = list(ranked)[:TOP_K]  # chunks, not paragraphs: every stage gets the same 5 slots
    return StageScore(
        item_id=item.id,
        category=item.category,
        stage=stage,
        ranked=top,
        hit=hit_at_k(top, gold, TOP_K),
        recall=recall_at_k(top, gold, TOP_K),
        mrr=reciprocal_rank(top, gold),
    )


def _keys(payloads: Sequence[dict[str, Any]]) -> list[str]:
    return [paragraph_key(p) for p in payloads]


def stage_rankings(item: GoldenItem, parts: SearchParts, candidates: int) -> dict[str, list[str]]:
    question = item.question
    vector = parts.embed(question)
    dense = parts.repo.search_dense(vector, candidates)
    sparse = parts.repo.search_sparse(Bm25Encoder.encode_query(question), candidates)
    payloads = {hit.id: hit.payload for hit in (*dense, *sparse)}
    fused = [chunk_id for chunk_id, _ in rrf_fuse([[h.id for h in dense], [h.id for h in sparse]], k=settings.rrf_k)]

    retrieval = RetrievalService(parts.repo, parts.embed, candidates=candidates)
    with_exact = retrieval.retrieve(question).chunks
    scores = parts.reranker.rerank(question, [c.payload["embed_text"] for c in with_exact])
    reranked = [with_exact[s.index] for s in sorted(scores, key=lambda s: s.score, reverse=True)]
    selected = SearchService(retrieval, parts.reranker).search(question, sector=item.sector).final

    return {
        "dense only": _keys([h.payload for h in dense]),
        "BM25 only": _keys([h.payload for h in sparse]),
        "hybrid (RRF)": _keys([payloads[i] for i in fused]),
        "+ exact lookup": _keys([c.payload for c in with_exact]),
        "+ rerank": _keys([c.payload for c in reranked]),
        "+ selection": _keys([c.chunk.payload for c in selected]),
    }


def retrieval_items(items: Sequence[GoldenItem]) -> list[GoldenItem]:
    """Single-turn answerable items only: a follow-up's text alone isn't a query."""
    return [item for item in items if item.answerable and not item.previous_turns]


def evaluate_retrieval(
    items: Sequence[GoldenItem],
    parts: SearchParts,
    candidates: int = settings.retrieval_candidates,
    on_item: Callable[[GoldenItem], None] = lambda item: None,
) -> tuple[list[StageScore], list[str]]:
    """(scores, errors): an item whose search fails (quota, timeout) is
    skipped and reported, and the rest of the run goes on."""
    scores: list[StageScore] = []
    errors: list[str] = []
    for item in retrieval_items(items):
        on_item(item)
        try:
            rankings = stage_rankings(item, parts, candidates)
        except Exception as exc:
            errors.append(f"{item.id}: {type(exc).__name__}: {exc}")
            continue
        scores += [score(item, stage, ranked) for stage, ranked in rankings.items()]
    return scores, errors


CUMULATIVE_STAGES = STAGES[2:]  # dense and BM25 are alternatives; each later stage builds on the one before


def regressions(
    scores: Sequence[StageScore], stages: Sequence[str] = CUMULATIVE_STAGES
) -> list[tuple[StageScore, StageScore]]:
    """(before, after) wherever an item's Hit@5 or MRR@5 drops when a stage is added."""
    by_item: dict[str, dict[str, StageScore]] = {}
    for s in scores:
        by_item.setdefault(s.item_id, {})[s.stage] = s
    drops = []
    for stage_scores in by_item.values():
        present = [stage_scores[stage] for stage in stages if stage in stage_scores]
        for before, after in zip(present, present[1:], strict=False):
            if after.hit < before.hit or after.mrr < before.mrr:
                drops.append((before, after))
    return drops


def summarise(scores: Sequence[StageScore], stages: Sequence[str] = STAGES) -> list[dict[str, Any]]:
    rows = []
    for stage in stages:
        of_stage = [s for s in scores if s.stage == stage]
        if of_stage:
            rows.append(
                {
                    "stage": stage,
                    "n": len(of_stage),
                    "hit@5": mean([s.hit for s in of_stage]),
                    "recall@5": mean([s.recall for s in of_stage]),
                    "mrr@5": mean([s.mrr for s in of_stage]),
                }
            )
    return rows
