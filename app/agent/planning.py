"""The planner: `understand_question` turns the question (and the recent
conversation) into a QueryPlan. The LLM decides the route; code then
sanitises the plan, because a prompt is a request, not a guarantee:

- at most 3 searches, none empty;
- paragraph references must look like a paragraph number ("8.2", "14A.9.6");
  the index confirms they exist when they are looked up;
- the route must agree with the content (a lookup with no valid
  reference becomes a search, a clarification with no question becomes a
  search, and so on).

If the planner fails (both models down, or invalid output twice), the
question is searched as typed: the traditional path, never an error.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime

from app.agent.budget import MAX_SEARCHES_PER_PLAN
from app.agent.prompts import planner_messages
from app.agent.schemas import ParagraphRef, QueryPlan, SearchRequest
from app.agent.state import AgentContext, AgentState
from app.llm.models import LLMUnavailableError

HISTORY_MESSAGES = 6  # the planner sees only the last few messages: small, fast prompts
MAX_PARAGRAPH_REFS = 5
PARAGRAPH_ID_RE = re.compile(r"\d{1,2}[A-Z]?(?:\.\d{1,2}){0,3}")
_PARA_PREFIX_RE = re.compile(r"^(?:para(?:graph)?\.?\s*)", re.IGNORECASE)
SEARCH_ROUTES = ("search", "search_and_lookup")
LOOKUP_ROUTES = ("lookup", "search_and_lookup")


def clean_paragraph_id(raw: str) -> str | None:
    """'para 14a.9.6' -> '14A.9.6'; something that isn't a paragraph number -> None."""
    candidate = _PARA_PREFIX_RE.sub("", raw.strip()).upper()
    return candidate if PARAGRAPH_ID_RE.fullmatch(candidate) else None


def _clean_refs(refs: list[ParagraphRef]) -> list[ParagraphRef]:
    cleaned: dict[tuple[str | None, str], ParagraphRef] = {}
    for ref in refs:
        if (paragraph_id := clean_paragraph_id(ref.paragraph_id)) is not None:
            cleaned.setdefault(
                (ref.document, paragraph_id), ParagraphRef(document=ref.document, paragraph_id=paragraph_id)
            )
    return list(cleaned.values())[:MAX_PARAGRAPH_REFS]


def _clean_searches(searches: list[SearchRequest]) -> list[SearchRequest]:
    cleaned = [
        s.model_copy(update={"query": s.query.strip(), "documents": list(dict.fromkeys(s.documents))})
        for s in searches
        if s.query.strip()
    ]
    return cleaned[:MAX_SEARCHES_PER_PLAN]


def sanitise_plan(plan: QueryPlan, question: str) -> QueryPlan:
    standalone = plan.standalone_question.strip() or question
    route = plan.route
    searches = _clean_searches(plan.searches)
    refs = _clean_refs(plan.paragraph_refs)
    clarifying = (plan.clarifying_question or "").strip() or None

    if route in LOOKUP_ROUTES and not refs:
        route = "search"
    if route == "search_and_lookup" and not searches:
        searches = [SearchRequest(query=standalone)]
    if route == "clarify" and not clarifying:
        route = "search"
    if route == "search" and not searches:
        searches = [SearchRequest(query=standalone)]

    return QueryPlan(
        route=route,
        standalone_question=standalone,
        searches=searches if route in SEARCH_ROUTES else [],
        paragraph_refs=refs if route in LOOKUP_ROUTES else [],
        clarifying_question=clarifying if route == "clarify" else None,
        reasoning=plan.reasoning.strip(),
    )


def fallback_plan(question: str) -> QueryPlan:
    return QueryPlan(
        route="search",
        standalone_question=question,
        searches=[SearchRequest(query=question)],
        reasoning="The planner was unavailable, so the question is searched as asked.",
    )


def recent_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The last few messages before the current question."""
    return messages[:-1][-HISTORY_MESSAGES:]


def understand_question(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    question = state["question"]
    messages = planner_messages(question, recent_history(state.get("messages", [])), state.get("sector"))
    try:
        result = runtime.context.llm.invoke("planner", messages, QueryPlan)
    except LLMUnavailableError as exc:
        return {
            "plan": fallback_plan(question),
            "notes": ["Planner unavailable; searched the question as asked."],
            "llm_calls": exc.calls,
        }
    plan = sanitise_plan(result.output, question)
    update: dict[str, Any] = {}
    sectors = {s.sector for s in plan.searches}
    if len(sectors) == 1 and None not in sectors:  # every search names one sector ("takaful operators")
        update["sector"] = sectors.pop()
    return update | {
        "plan": plan,
        "models": [result.model],
        "llm_calls": result.calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
