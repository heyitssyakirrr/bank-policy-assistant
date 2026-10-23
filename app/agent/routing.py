"""Conditional edges: where the LLM's decisions become control flow.

Both functions are pure code reading the state. The planner's route picks
the branch; the reviewer's verdict (already trimmed to the budget by the
review node) decides whether to loop. `Send` starts one tool call per
search, in parallel.
"""

from __future__ import annotations

from collections.abc import Sequence

from langgraph.types import Send

from app.agent.schemas import ParagraphRef, SearchRequest
from app.agent.state import AgentState


def fan_out(
    searches: Sequence[SearchRequest], refs: Sequence[ParagraphRef], sector: str | None, question: str
) -> list[Send]:
    """One `search_policies` per search, plus one `lookup_paragraphs` for all references."""
    sends = [Send("search_policies", {"request": search, "sector": sector}) for search in searches]
    if refs:
        sends.append(Send("lookup_paragraphs", {"refs": list(refs), "sector": sector, "question": question}))
    return sends


def route_after_planning(state: AgentState) -> str | list[Send]:
    plan = state["plan"]
    if plan.route in ("out_of_scope", "small_talk"):
        return "respond_directly"
    if plan.route == "clarify":
        return "ask_clarification"
    return fan_out(plan.searches, plan.paragraph_refs, state.get("sector"), plan.standalone_question)


def route_after_review(state: AgentState) -> str | list[Send]:
    review = state["review"]
    if review.is_sufficient:
        return "generate_answer"
    question = state["plan"].standalone_question
    sends = fan_out(review.follow_up_searches, review.cross_references, state.get("sector"), question)
    return sends or "generate_answer"
