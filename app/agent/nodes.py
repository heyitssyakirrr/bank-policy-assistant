"""Nodes shared by both graphs: the generator, the verifier and `finish`.

Both modes end with the same `generate_answer -> verify_answer -> finish`
steps, so the traditional-vs-agentic comparison measures only the
decision-making before them.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.agent.evidence import Evidence
from app.agent.prompts import generator_messages
from app.agent.schemas import GroundedAnswer
from app.agent.state import AgentContext, AgentState
from app.agent.verification import NOT_COVERED_SUMMARY, check_answer, finalize, regeneration_feedback
from app.llm.models import LLMUnavailableError
from app.services.search_service import SearchResult

logger = logging.getLogger("app.agent")

MAX_GENERATIONS = 2  # the first answer plus one regeneration after a failed verification
UNAVAILABLE_SUMMARY = "The answer service is temporarily unavailable. Please try again in a minute."


def search_record(result: SearchResult) -> dict[str, Any]:
    """One search, as recorded in the audit trail and shown in the agent steps."""
    return {
        "kind": "search",
        "query": result.question,
        "documents": result.documents,
        "sector": result.preferred_sector,
        "reranked": result.reranked,
        "dense_used": result.retrieval.dense_used,
        "chunks": [{"id": c.id, "citation": c.chunk.citation, "rerank_score": c.rerank_score} for c in result.final],
        "collapsed": len(result.selection.collapsed),
        "timings_ms": {k: round(v) for k, v in result.timings_ms.items()},
    }


def answer_evidence(state: AgentState) -> list[Evidence]:
    """The evidence the generator reads: what the reviewer judged relevant,
    or everything when there was no review (the traditional mode)."""
    evidence = state.get("evidence", [])
    review = state.get("review")
    if review is None:
        return evidence
    relevant = set(review.relevant_evidence)
    return [e for e in evidence if e.id in relevant]


def generate_answer(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    evidence = answer_evidence(state)
    generations = state.get("generations", 0) + 1
    if not evidence:  # nothing relevant: refuse without spending an LLM call
        answer = GroundedAnswer(status="not_covered", summary=NOT_COVERED_SUMMARY)
        return {"answer": answer, "generations": generations}

    plan = state.get("plan")  # agentic: the planner's self-contained rewrite of a follow-up
    question = plan.standalone_question if plan else state["question"]
    messages = generator_messages(question, evidence, state.get("sector"), state.get("feedback"))
    try:
        result = runtime.context.llm.invoke("generator", messages, GroundedAnswer)
    except LLMUnavailableError as exc:
        logger.error(f"generator failed: {exc}")
        verifications = state.get("verifications", [])
        if verifications:  # a regeneration failed: keep the verified part of the first answer
            fallback = finalize(verifications[-1])
        else:
            fallback = GroundedAnswer(status="not_covered", summary=UNAVAILABLE_SUMMARY)
        return {
            "answer": fallback,
            "generations": generations,
            "feedback": None,
            "notes": ["Generator unavailable."],
            "llm_calls": exc.calls,
        }
    return {
        "answer": result.output,
        "generations": generations,
        "feedback": None,
        "models": [result.model],
        "llm_calls": result.calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


def verify_answer(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    report = check_answer(state["answer"], answer_evidence(state))
    update: dict[str, Any] = {"verifications": [report]}
    if report.passed:
        return update | {"answer": report.answer}
    budget = runtime.context.budget
    can_regenerate = (
        state.get("generations", 0) < MAX_GENERATIONS
        and budget.llm_calls_left(state) >= 1
        and not budget.out_of_time(state, runtime.context.clock())
    )
    if can_regenerate:
        return update | {"feedback": regeneration_feedback(report)}
    notes = [] if state.get("generations", 0) >= MAX_GENERATIONS else ["Regeneration skipped: request budget reached."]
    return update | {"answer": finalize(report), "notes": notes}


def route_after_verification(state: AgentState) -> str:
    return "generate_answer" if state.get("feedback") else "finish"


def finish(state: AgentState) -> dict[str, Any]:
    """Add the reply to the conversation, so the next question can refer to it."""
    return {"messages": [AIMessage(content=state["answer"].summary)]}
