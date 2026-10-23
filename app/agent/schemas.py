"""The contracts the LLM must fill (structured output), PROJECT_PLAN §5.4.

Kept flat, with Literal enums and short field descriptions: Flash-Lite
models fill simple schemas far more reliably than nested or free-form ones.
Every LLM output is still sanitised by code after validation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocCode = Literal["ekyc", "ftfc", "amlcft"]
Sector = Literal["banking", "insurance_takaful", "money_services", "non_bank_e_money"]
Obligation = Literal["must", "should_or_may", "information"]
AnswerStatus = Literal["answered", "partially_answered", "not_covered"]
Route = Literal["search", "lookup", "search_and_lookup", "clarify", "out_of_scope", "small_talk"]
Mode = Literal["agentic", "traditional"]  # which graph answers: the LLM decides, or a fixed line

DISCLAIMER = "Based on BNM policy documents. Not legal advice."
DOCUMENT_TITLES: dict[str, str] = {
    "ekyc": "e-KYC Policy Document",
    "ftfc": "Fair Treatment of Financial Consumers Policy Document",
    "amlcft": "AML/CFT/CPF and TFS for FIs Policy Document",
}
SECTOR_LABELS: dict[str, str] = {
    "banking": "banking institutions",
    "insurance_takaful": "insurers and takaful operators",
    "money_services": "money services businesses",
    "non_bank_e_money": "non-bank e-money issuers",
}


class SearchRequest(BaseModel):
    query: str = Field(description="A self-contained search query in the documents' own wording")
    documents: list[DocCode] = Field(default_factory=list, description="[] searches all documents")
    sector: Sector | None = Field(default=None, description="null = the user's own sector")


class ParagraphRef(BaseModel):
    document: DocCode | None = Field(description="null only when the user did not say which document")
    paragraph_id: str = Field(description='e.g. "8.2", "14A.9.6"')


class QueryPlan(BaseModel):
    route: Route
    standalone_question: str = Field(description="The question rewritten to stand alone, using the chat history")
    searches: list[SearchRequest] = Field(default_factory=list, description="1-3 searches; one per part")
    paragraph_refs: list[ParagraphRef] = Field(default_factory=list)
    clarifying_question: str | None = None
    reasoning: str = Field(description="One sentence explaining the route, shown to the user")


class EvidenceReview(BaseModel):
    relevant_evidence: list[str] = Field(description='IDs of the passages that help answer, e.g. ["E1", "E4"]')
    is_sufficient: bool
    missing_information: str | None = None
    follow_up_searches: list[SearchRequest] = Field(default_factory=list)
    cross_references: list[ParagraphRef] = Field(default_factory=list)


class Statement(BaseModel):
    text: str
    obligation: Obligation
    evidence_ids: list[str] = Field(description='The passages supporting this statement, e.g. ["E3"]')
    quote: str = Field(description="Exact words copied from one cited passage that support the statement")
    topic: str = Field(description="Which part of the question this statement answers")


class GroundedAnswer(BaseModel):
    status: AnswerStatus
    summary: str = Field(description="The direct answer in 1-2 sentences")
    statements: list[Statement] = Field(default_factory=list)
    gaps: str | None = Field(default=None, description="What the documents do not say")
