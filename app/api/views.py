"""What the API returns for a turn: the answer with its citations resolved.

The generator cites E-labels; here code maps each label back to the stored
citation, the S/G marker, a text snippet and a link to the PDF page. The
same view serves the live answer and the conversation history.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from app.agent.prompts import applies_to
from app.agent.schemas import DISCLAIMER
from app.agent.steps import SHORT_TITLES, STEP_TITLES
from app.db.models import Turn, TurnEvent

SNIPPET_CHARS = 400


class SourceView(BaseModel):
    id: str  # "E3"
    label: str  # "e-KYC 8.4 (S)": short enough for a citation chip
    citation: str  # "e-KYC Policy Document, para 8.4 (S), p. 6"
    document: str
    provision: str | None  # "S" | "G" | None
    snippet: str
    pdf_url: str  # relative to the API: "/documents/ekyc.pdf#page=8"


class StepView(BaseModel):
    node: str  # "review_evidence"
    title: str  # "Review": the step timeline's label
    lines: list[str]  # what the step did, in plain English
    ms: int  # time since the turn started


class StatementView(BaseModel):
    text: str
    obligation: str  # "must" | "should_or_may" | "information"
    quote: str
    topic: str
    sources: list[str]  # E-labels, each present in TurnView.sources


class TurnView(BaseModel):
    turn_id: uuid.UUID
    conversation_id: uuid.UUID
    mode: str
    question: str
    standalone_question: str | None
    pii_masked: list[str]
    outcome: str
    status: str
    summary: str
    statements: list[StatementView]
    gaps: str | None
    sources: list[SourceView]  # only the passages the answer cites
    applies_to: str
    verified: bool
    notes: list[str]
    steps: list[StepView]  # the step timeline (the "finish" node and empty steps left out)
    llm_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    models: list[str]
    disclaimer: str = DISCLAIMER
    created_at: datetime | None = None


DOCUMENT_FILES = {
    "ekyc": "pd-ekyc-apr2024.pdf",
    "ftfc": "pd-ftfc-mar24.pdf",
    "amlcft": "PD_AMLCFTCPF_TFS_FI_Feb2024_v2.pdf",
}


def pdf_path(document: str) -> str:
    return f"/documents/{document}.pdf"


def pdf_url(document: str, page: int) -> str:
    """Opens the browser's PDF viewer at the cited page."""
    return f"{pdf_path(document)}#page={page}"


def source_view(evidence: dict[str, Any]) -> SourceView:
    text = evidence["text"]
    scope = evidence.get("scope", "main")
    paragraph = evidence.get("paragraph_id") or "table"
    if scope != "main":  # appendices reuse paragraph numbers: "e-KYC App. 2 para 13"
        paragraph = f"{scope.replace('Appendix', 'App.')} para {paragraph}"
    marker = f" ({evidence['provision']})" if evidence.get("provision") else ""
    return SourceView(
        id=evidence["id"],
        label=f"{SHORT_TITLES[evidence['document']]} {paragraph}{marker}",
        citation=evidence["citation"],
        document=evidence["document"],
        provision=evidence.get("provision"),
        snippet=text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " …",
        pdf_url=pdf_url(evidence["document"], evidence["page"]),
    )


def turn_view(turn: Turn, events: Sequence[TurnEvent]) -> TurnView:
    answer = turn.answer or {}
    statements = answer.get("statements", [])
    cited = dict.fromkeys(i for s in statements for i in s["evidence_ids"])
    evidence = {e["id"]: e for e in turn.evidence}
    verifications = turn.verification or []
    return TurnView(
        turn_id=turn.id,
        conversation_id=turn.conversation_id,
        mode=turn.mode,
        question=turn.question,
        standalone_question=turn.standalone_question,
        pii_masked=turn.pii_masked,
        outcome=turn.outcome,
        status=turn.status,
        summary=answer.get("summary", ""),
        statements=[
            StatementView(
                text=s["text"],
                obligation=s["obligation"],
                quote=s["quote"],
                topic=s["topic"],
                sources=s["evidence_ids"],
            )
            for s in statements
        ],
        gaps=answer.get("gaps"),
        sources=[source_view(evidence[i]) for i in cited if i in evidence],
        applies_to=applies_to(turn.sector),
        verified=bool(verifications) and not verifications[-1].get("serious"),
        notes=turn.notes,
        steps=[
            StepView(node=e.node, title=STEP_TITLES[e.node], lines=e.lines, ms=e.ms)
            for e in events
            if e.node in STEP_TITLES and e.lines
        ],
        llm_calls=turn.llm_calls,
        input_tokens=turn.input_tokens,
        output_tokens=turn.output_tokens,
        latency_ms=turn.latency_ms,
        models=turn.models,
        created_at=turn.created_at,
    )
