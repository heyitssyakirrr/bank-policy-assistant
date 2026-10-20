"""Choosing the final chunks the LLM will read, after reranking.

The reranker orders candidates by relevance. Two domain rules then decide
the final top N:

1. ONE COPY PER REPEATED RULE. The AML policy repeats many rules per sector:
   14A.15.1 (banking), 14B.17.1 (insurance), 14C.16.1 (money services) and
   14D.16.1 (e-money) are word-for-word the same. All four would rank highly
   and fill the top 5 with one rule, crowding out other sources. We keep one
   copy: the user's sector (banking by default, or the sector the question
   names), else the best-ranked.

2. OTHER SECTORS' RULES RANK LOWER. A rule that exists only for money
   services (no banking twin to collapse into) is still the wrong answer
   for a bank employee. Chunks for another sector are moved below chunks
   that apply to the user (their sector, or no sector). Demoted, not
   deleted: they remain a fallback, and naming a sector in the question
   ("remittance companies") makes that sector the preferred one.

3. NAMED PARAGRAPHS ARE NEVER DROPPED. If the question names a paragraph
   ("14A.9.6"), that chunk is guaranteed a place, whatever its rerank score.

Everything here is a pure function (data in, data out, no API calls), so
each rule is tested exactly in tests/test_rerank.py.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.services.retrieval_service import RankedChunk

# Word overlap (Jaccard) above which two sector rules count as the same rule.
# Tuned on the real AML corpus: 0.85 catches copies with tiny wording changes
# ("a party"), while rules that genuinely differ (14A.8 vs 14B.9, ~0.78) stay
# apart. A wrong merge is low-risk by design: the copy for the user's own
# sector is the one kept. A missed duplicate wastes one of only 5 slots.
DUPLICATE_SIMILARITY = 0.85
_LEADING_NUMBER_RE = re.compile(r"^(?:[SG]\s+)?\d{1,2}[A-Z]?(?:\.\d{1,2})+\s*")
_WORD_RE = re.compile(r"\w+")

# Phrases that mean the user is asking about a specific sector. Checked in order.
SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "insurance_takaful": ("insurance", "insurer", "takaful", "policyholder"),
    "money_services": ("money services", "money-changing", "money changer", "remittance", "msb", "wholesale currency"),
    "non_bank_e_money": ("e-money", "non-bank", "designated payment instrument"),
    "banking": ("deposit-taking", "banking institution", "licensed bank"),
}


@dataclass(frozen=True, slots=True)
class Candidate:
    """A retrieval candidate plus what the reranker said about it."""

    chunk: RankedChunk
    fused_rank: int  # 1-based position before reranking (exact matches first, then RRF)
    rerank_score: float | None = None  # None if reranking was skipped (fallback)

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def sector(self) -> str | None:
        return self.chunk.payload.get("sector")

    @property
    def exact_match(self) -> bool:
        return self.chunk.exact_match


@dataclass(frozen=True, slots=True)
class Selection:
    final: list[Candidate]
    collapsed: list[tuple[Candidate, Candidate]] = field(default_factory=list)  # (dropped, kept instead)
    demoted: list[Candidate] = field(default_factory=list)  # other-sector chunks moved down the order


def detect_sector(question: str) -> str | None:
    text = question.lower()
    for sector, phrases in SECTOR_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in phrases):
            return sector
    return None


def _words(candidate: Candidate) -> frozenset[str]:
    """The rule's words, ignoring anything with a digit. Copies of a rule
    differ in exactly that: their own number ("14A.15.1" vs "14B.17.1") and
    references to their own sector's paragraphs ("paragraph 14A.5" vs
    "paragraph 14C.6"). If copies differ in an amount, the sector preference
    still keeps the copy that applies to the user."""
    body = _LEADING_NUMBER_RE.sub("", candidate.chunk.payload.get("text", ""))
    return frozenset(w for w in _WORD_RE.findall(body.lower()) if not any(ch.isdigit() for ch in w))


def _similar(a: frozenset[str], b: frozenset[str]) -> bool:
    return bool(a and b) and len(a & b) / len(a | b) >= DUPLICATE_SIMILARITY


def _wins(challenger: Candidate, holder: Candidate, preferred_sector: str | None) -> bool:
    """Should `challenger` replace `holder` (its copy in another sector)?"""
    if challenger.exact_match != holder.exact_match:
        return challenger.exact_match  # a paragraph the user named always wins
    return challenger.sector == preferred_sector and holder.sector != preferred_sector


def collapse_sector_duplicates(
    candidates: Sequence[Candidate], preferred_sector: str | None
) -> tuple[list[Candidate], list[tuple[Candidate, Candidate]]]:
    """Keep one copy of each rule repeated across sectors. Input order (best
    first) is preserved; a winning copy takes the position of the copy it
    replaces, i.e. the best rank any copy achieved."""
    kept: list[Candidate] = []
    words: dict[str, frozenset[str]] = {}
    collapsed: list[tuple[Candidate, Candidate]] = []

    for candidate in candidates:
        if candidate.sector is None:
            kept.append(candidate)
            continue
        words[candidate.id] = _words(candidate)
        twin_pos = next(
            (
                pos
                for pos, holder in enumerate(kept)
                if holder.sector not in (None, candidate.sector) and _similar(words[holder.id], words[candidate.id])
            ),
            None,
        )
        if twin_pos is None:
            kept.append(candidate)
        elif _wins(candidate, kept[twin_pos], preferred_sector):
            collapsed.append((kept[twin_pos], candidate))
            kept[twin_pos] = candidate
        else:
            collapsed.append((candidate, kept[twin_pos]))
    return kept, collapsed


def applies_to_user(candidate: Candidate, preferred_sector: str | None) -> bool:
    return candidate.exact_match or candidate.sector in (None, preferred_sector)


def select_final(candidates: Sequence[Candidate], top_n: int, preferred_sector: str | None) -> Selection:
    """Best-first candidates -> the final top_n, applying the three rules above."""
    collapsed_kept, collapsed = collapse_sector_duplicates(candidates, preferred_sector)
    # Stable sort: within each group the reranker's order is kept.
    kept = sorted(collapsed_kept, key=lambda c: not applies_to_user(c, preferred_sector))
    demoted = [c for c in kept if not applies_to_user(c, preferred_sector)]
    final = kept[:top_n]

    # Guarantee named paragraphs a place: each one missing replaces the
    # lowest-ranked chunk that isn't itself a named paragraph.
    for missing in (c for c in kept[top_n:] if c.exact_match):
        replaceable = [pos for pos, c in enumerate(final) if not c.exact_match]
        if not replaceable:
            break
        final[replaceable[-1]] = missing
    order = {c.id: pos for pos, c in enumerate(kept)}
    final.sort(key=lambda c: order[c.id])  # keep best-first order
    return Selection(final=final, collapsed=collapsed, demoted=demoted)
