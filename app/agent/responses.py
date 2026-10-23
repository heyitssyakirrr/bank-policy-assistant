"""Answers that need no search: refusals, greetings and clarifying questions.

Pure code: the planner already decided the route, so writing these costs no
further LLM call. Refusing before any search is the agentic mode's first
safety net for out-of-scope questions (the reviewer is the second).
"""

from __future__ import annotations

from typing import Any

from app.agent.schemas import GroundedAnswer
from app.agent.state import AgentState

OUT_OF_SCOPE = (
    "I can only answer questions about three Bank Negara Malaysia policy documents: e-KYC, Fair Treatment "
    "of Financial Consumers, and AML/CFT/CPF & TFS for Financial Institutions. This question is outside them."
)
GREETING = (
    "Hello! I answer questions about BNM's e-KYC, Fair Treatment of Financial Consumers and AML/CFT/CPF & TFS "
    'policy documents, with paragraph citations. For example: "Can a customer be onboarded without meeting '
    'face to face?"'
)


def respond_directly(state: AgentState) -> dict[str, Any]:
    route = state["plan"].route
    summary = GREETING if route == "small_talk" else OUT_OF_SCOPE
    return {"outcome": route, "answer": GroundedAnswer(status="not_covered", summary=summary)}


def ask_clarification(state: AgentState) -> dict[str, Any]:
    question = state["plan"].clarifying_question
    return {"outcome": "clarify", "answer": GroundedAnswer(status="not_covered", summary=question)}
