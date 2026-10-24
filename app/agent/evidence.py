"""Evidence: the retrieved passages the LLM reads, each labelled E1, E2, ...

The LLM cites a label, never a paragraph number or a page, so it cannot
invent a citation. Code owns the mapping back to the stored citation
("e-KYC Policy Document, para 8.4 (S), p. 6") and to the PDF page.

Labels are stable within a question: when new searches add passages, the
existing ones keep their labels and the new ones get the next numbers.
Labels restart at E1 for every question.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from app.services.selection import Candidate, rule_words, same_rule

EvidenceSource = Literal["search", "lookup"]
# A dependency on another paragraph: "as required under paragraph 8.4", "in relation
# to paragraphs 8.10 and 8.13". A bare mention ("notwithstanding paragraphs 8.19 to
# 8.21") is not one, so it isn't followed.
_DEPENDENCY_RE = re.compile(
    r"(?:as required under|in relation to|in respect of|for the purposes? of|subject to|pursuant to"
    r"|in accordance with|in meeting (?:the )?requirements? under)\s+paragraphs?\s+",
    re.IGNORECASE,
)
_PARAGRAPH_LIST_RE = re.compile(r"(?:\d{1,2}[A-Za-z]?(?:\.\d{1,2})*(?:\s*\([a-z0-9]+\))*\s*(?:,|and|to|-)?\s*)+")
_PARAGRAPH_ID_RE = re.compile(r"\d{1,2}[A-Za-z]?(?:\.\d{1,2})+")
_OTHER_DOCUMENT_RE = re.compile(r"of the\b", re.IGNORECASE)  # "... of the AML/CFT policy document"


def cross_reference_ids(text: str) -> set[str]:
    """Paragraph numbers this text depends on, in its own document. Lists
    ("8.10 and 8.13") give every number; ranges ("8.19 to 8.21") both ends;
    "paragraph 6.2 of the AML/CFT policy document" is another document's."""
    ids: set[str] = set()
    for match in _DEPENDENCY_RE.finditer(text):
        listed = _PARAGRAPH_LIST_RE.match(text, match.end())
        if listed is None:
            continue
        if _OTHER_DOCUMENT_RE.match(text, listed.end()):
            continue
        ids.update(i.upper() for i in _PARAGRAPH_ID_RE.findall(listed.group(0)))
    return ids


class Evidence(BaseModel):
    id: str  # "E3"
    chunk_id: str
    document: str  # doc_code: "ekyc" | "ftfc" | "amlcft"
    scope: str = "main"  # "main" or the appendix ("Appendix 2"): appendices reuse paragraph numbers
    paragraph_id: str | None
    provision: Literal["S", "G"] | None  # S = obligation, G = guidance, None = not marked (appendix, table, text)
    sector: str | None
    citation: str
    text: str  # paragraph text plus its footnotes: exactly what the LLM reads
    page: int  # 1-based PDF page (for "open PDF at page"), not the printed page number
    source: EvidenceSource
    query: str  # the search query or paragraph lookup that fetched it
    rerank_score: float | None = None
    requested_sector: str | None = None  # the sector its search explicitly asked for (e.g. a comparison side)


def passage_text(payload: dict[str, Any]) -> str:
    footnotes = payload.get("footnotes") or []
    return "\n".join([payload.get("text", ""), *footnotes]).strip()


def to_evidence(
    candidates: Iterable[Candidate], source: EvidenceSource, query: str, requested_sector: str | None = None
) -> list[Evidence]:
    """Unlabelled evidence from a search or lookup; `merge_evidence` labels it."""
    evidence = []
    for candidate in candidates:
        payload = candidate.chunk.payload
        evidence.append(
            Evidence(
                id="",
                chunk_id=candidate.id,
                document=payload["doc_code"],
                scope=payload.get("scope", "main"),
                paragraph_id=payload.get("paragraph_id"),
                provision=payload.get("provision"),
                sector=payload.get("sector"),
                citation=payload.get("citation", candidate.id),
                text=passage_text(payload),
                page=payload.get("page_start", 1),
                source=source,
                query=query,
                rerank_score=candidate.rerank_score,
                requested_sector=requested_sector,
            )
        )
    return evidence


def _sector_copy_of_existing(new: Evidence, existing: Sequence[Evidence]) -> bool:
    """Is `new` another sector's copy of a rule already in the evidence? Each
    search collapses its own copies; this catches copies found by two
    different searches. Kept regardless: a paragraph the user named (a
    lookup), and a copy its search explicitly asked for (the takaful side of
    a banks-vs-takaful comparison)."""
    if new.source == "lookup" or new.sector is None or new.sector == new.requested_sector:
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


def cross_references(item: Evidence, evidence: Sequence[Evidence]) -> list[Evidence]:
    """The other given passages this one depends on ("as required under
    paragraph 8.4"), in the same document. A fact, so code finds it."""
    named = cross_reference_ids(item.text)
    return [
        e
        for e in evidence
        if e.chunk_id != item.chunk_id
        and e.document == item.document
        and e.scope == item.scope
        and e.paragraph_id in named
    ]


def format_evidence(evidence: Sequence[Evidence]) -> str:
    """The numbered blocks shown to the LLM. Passages are data, fenced off
    so instructions inside a document can't pass as ours. A passage that
    depends on another one given here says so: "Cross-reference: E2 (para 8.4)"."""
    blocks = []
    for e in evidence:
        linked = ", ".join(f"{r.id} (para {r.paragraph_id})" for r in cross_references(e, evidence))
        note = f"\nCross-reference: {linked}" if linked else ""
        blocks.append(f'[{e.id}] {e.citation}{note}\n"""\n{e.text}\n"""')
    return "\n\n".join(blocks) if blocks else "(no passages found)"
