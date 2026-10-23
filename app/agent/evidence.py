"""Evidence: the retrieved passages the LLM reads, each labelled E1, E2, ...

The LLM cites a label, never a paragraph number or a page, so it cannot
invent a citation. Code owns the mapping back to the stored citation
("e-KYC Policy Document, para 8.4 (S), p. 6") and to the PDF page.

Labels are stable within a question: when new searches add passages, the
existing ones keep their labels and the new ones get the next numbers.
Labels restart at E1 for every question.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from app.services.selection import Candidate, rule_words, same_rule

EvidenceSource = Literal["search", "lookup"]


class Evidence(BaseModel):
    id: str  # "E3"
    chunk_id: str
    document: str  # doc_code: "ekyc" | "ftfc" | "amlcft"
    paragraph_id: str | None
    provision: Literal["S", "G"] | None  # S = obligation, G = guidance, None = not marked (appendix, table, text)
    sector: str | None
    citation: str
    text: str  # paragraph text plus its footnotes: exactly what the LLM reads
    page: int  # 1-based PDF page (for "open PDF at page"), not the printed page number
    source: EvidenceSource
    query: str  # the search query or paragraph lookup that fetched it
    rerank_score: float | None = None


def passage_text(payload: dict[str, Any]) -> str:
    footnotes = payload.get("footnotes") or []
    return "\n".join([payload.get("text", ""), *footnotes]).strip()


def to_evidence(candidates: Iterable[Candidate], source: EvidenceSource, query: str) -> list[Evidence]:
    """Unlabelled evidence from a search or lookup; `merge_evidence` labels it."""
    evidence = []
    for candidate in candidates:
        payload = candidate.chunk.payload
        evidence.append(
            Evidence(
                id="",
                chunk_id=candidate.id,
                document=payload["doc_code"],
                paragraph_id=payload.get("paragraph_id"),
                provision=payload.get("provision"),
                sector=payload.get("sector"),
                citation=payload.get("citation", candidate.id),
                text=passage_text(payload),
                page=payload.get("page_start", 1),
                source=source,
                query=query,
                rerank_score=candidate.rerank_score,
            )
        )
    return evidence


def _sector_copy_of_existing(new: Evidence, existing: Sequence[Evidence]) -> bool:
    """Is `new` another sector's copy of a rule already in the evidence? Each
    search collapses its own copies; this catches copies found by two
    different searches. A paragraph the user named (a lookup) is always kept."""
    if new.source == "lookup" or new.sector is None:
        return False
    words = rule_words(new.text)
    return any(e.sector not in (None, new.sector) and same_rule(rule_words(e.text), words) for e in existing)


def merge_evidence(current: list[Evidence], new: list[Evidence] | None) -> list[Evidence]:
    """The graph's reducer for `evidence`. Parallel searches each return
    unlabelled evidence; this appends what is new (by chunk ID, and not a
    sector copy of a rule already present) with the next labels E1, E2, ...
    Existing labels never change. `None` starts a new turn: no evidence."""
    if new is None:
        return []
    merged = list(current)
    seen = {e.chunk_id for e in merged}
    for item in new:
        if item.chunk_id in seen or _sector_copy_of_existing(item, merged):
            continue
        seen.add(item.chunk_id)
        merged.append(item.model_copy(update={"id": f"E{len(merged) + 1}"}))
    return merged


def format_evidence(evidence: Sequence[Evidence]) -> str:
    """The numbered blocks shown to the LLM. Passages are data, fenced off
    so instructions inside a document can't pass as ours."""
    blocks = [f'[{e.id}] {e.citation}\n"""\n{e.text}\n"""' for e in evidence]
    return "\n\n".join(blocks) if blocks else "(no passages found)"
