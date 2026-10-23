"""The state that flows through a graph, and the dependencies its nodes use.

A conversation is one checkpointer thread, so the state survives between
questions. Only `messages` (the chat history) should carry over; every
other field belongs to one question (a "turn"). `turn_input` therefore
resets them all at the start of each turn:
  - plain fields are overwritten with their starting value;
  - fields with a reducer (summed counters, merged evidence, appended logs)
    accept `None` as "reset", because a reducer can't otherwise be cleared.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages

from app.agent.budget import Budget
from app.agent.evidence import Evidence, merge_evidence
from app.agent.schemas import EvidenceReview, GroundedAnswer, QueryPlan
from app.agent.verification import VerificationReport
from app.core.config import settings
from app.llm.models import StructuredLLM
from app.services.retrieval_service import ParagraphLookup
from app.services.search_service import SearchResult
from app.services.selection import Candidate

Outcome = Literal["answer", "out_of_scope", "small_talk", "clarify"]


class SearchTool(Protocol):
    """What the tools need from the search engine (SearchService, or a fake)."""

    def search(
        self,
        question: str,
        sector: str | None,
        documents: Sequence[str] = (),
        paragraph_refs: Sequence[ParagraphLookup] | None = None,
    ) -> SearchResult: ...

    def lookup(self, refs: Sequence[ParagraphLookup]) -> list[Candidate]: ...


@dataclass(frozen=True, slots=True)
class AgentContext:
    """What nodes need from outside the graph, passed at invoke time
    (`graph.invoke(state, context=AgentContext(...))`), so tests can pass fakes."""

    llm: StructuredLLM
    search: SearchTool
    budget: Budget = field(default_factory=Budget)
    clock: Callable[[], float] = time.time


def add_or_reset(current: int, update: int | None) -> int:
    return 0 if update is None else current + update


def extend_or_reset[T](current: list[T], update: list[T] | None) -> list[T]:
    return [] if update is None else current + update


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]  # the conversation: kept across turns
    question: str
    sector: str | None  # the preferred sector; None = all sectors
    started_at: float  # epoch seconds: the request budget's clock
    plan: QueryPlan | None
    review: EvidenceReview | None
    rounds: int  # review rounds done this turn
    evidence: Annotated[list[Evidence], merge_evidence]
    answer: GroundedAnswer | None
    verifications: Annotated[list[VerificationReport], extend_or_reset]  # one per generation
    feedback: str | None  # the verifier's notes for a regeneration
    generations: int
    outcome: Outcome
    notes: Annotated[list[str], extend_or_reset]  # degradations and limits hit, shown to the user
    models: Annotated[list[str], extend_or_reset]  # every model that answered, in order
    llm_calls: Annotated[int, add_or_reset]
    input_tokens: Annotated[int, add_or_reset]
    output_tokens: Annotated[int, add_or_reset]
    search_log: Annotated[list[dict[str, Any]], extend_or_reset]  # each search/lookup: query, filters, chunks


def turn_input(question: str, sector: str | None, started_at: float | None = None) -> dict[str, Any]:
    """The graph input for a new question: resets every per-turn field and
    adds the question to the conversation. Long input is cut (§5.5)."""
    question = question.strip()[: settings.max_question_chars]
    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "sector": sector,
        "started_at": time.time() if started_at is None else started_at,
        "plan": None,
        "review": None,
        "rounds": 0,
        "evidence": None,
        "answer": None,
        "verifications": None,
        "feedback": None,
        "generations": 0,
        "outcome": "answer",
        "notes": None,
        "models": None,
        "llm_calls": None,
        "input_tokens": None,
        "output_tokens": None,
        "search_log": None,
    }
