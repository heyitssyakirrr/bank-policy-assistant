"""Evaluation metrics: pure functions, no LLM (PROJECT_PLAN §8.3).

Retrieval, over a ranked list of paragraph keys and the gold keys:
  Hit@k     1 if any gold paragraph is in the top k, else 0
  Recall@k  the share of gold paragraphs found in the top k
  MRR       1 / rank of the first gold paragraph (0 if none)

Answers:
  outcome correct   answered when it should, refused or clarified when it should
                    (a crash or outage is an "error", never a refusal: it is reported apart)
  citation recall   the share of gold paragraphs the answer cites
  faithfulness      the judge's share of supported statements (partly = 0.5)
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _dedupe(ranked: Sequence[str]) -> list[str]:
    """A long paragraph can be several chunks: count each paragraph once, at its best rank."""
    return list(dict.fromkeys(ranked))


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 5) -> float:
    return float(any(key in gold for key in _dedupe(ranked)[:k]))


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int = 5) -> float:
    if not gold:
        return 0.0
    top = set(_dedupe(ranked)[:k])
    return sum(key in top for key in set(gold)) / len(set(gold))


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str]) -> float:
    for rank, key in enumerate(_dedupe(ranked), start=1):
        if key in gold:
            return 1.0 / rank
    return 0.0


def coverage(found: Sequence[str], gold: Sequence[str]) -> float:
    """The share of gold paragraphs in `found` (any order): citation and evidence recall."""
    if not gold:
        return 0.0
    return sum(key in set(found) for key in set(gold)) / len(set(gold))


def observed_outcome(outcome: str, status: str, degraded: bool = False) -> str:
    """What the system did, in the golden set's terms: answer, refuse, clarify,
    or error (an exception, or a model outage that forced a fallback answer)."""
    if degraded or status in ("error", "cancelled"):
        return "error"
    if outcome == "clarify":
        return "clarify"
    if outcome in ("out_of_scope", "small_talk") or status == "not_covered":
        return "refuse"
    return "answer"


def faithfulness(verdicts: Sequence[str]) -> float | None:
    """The judge's verdict per statement ("yes" | "partly" | "no") -> 0..1."""
    if not verdicts:
        return None
    score = {"yes": 1.0, "partly": 0.5, "no": 0.0}
    return sum(score[v] for v in verdicts) / len(verdicts)


def mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0..100): p50 = median, p95 = tail latency."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]
