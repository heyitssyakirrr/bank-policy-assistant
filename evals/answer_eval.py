"""Answer evaluation: every item through both modes, scored without an LLM.

For each item and mode the full graph runs (earlier turns first, for a
follow-up) and the turn is scored on:
  - outcome: answered, refused or clarified, as expected?
  - citation recall: the share of gold paragraphs the answer cites
  - evidence recall: the share of gold paragraphs the mode retrieved at all
  - verified: the final answer passed every verifier check
  - cost: LLM calls, tokens, latency (latency only when no response was cached)
The judge (judge.py) then adds faithfulness and relevancy.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from app.agent.checkpoint import state_serializer
from app.agent.evidence import Evidence
from app.agent.graph import build_agentic_graph
from app.agent.schemas import GroundedAnswer
from app.agent.state import AgentContext, turn_input
from app.agent.traditional import build_traditional_graph
from evals.golden import GoldenItem, paragraph_key
from evals.metrics import coverage, observed_outcome

BUILDERS = {"traditional": build_traditional_graph, "agentic": build_agentic_graph}


class TurnResult(BaseModel):
    item_id: str
    category: str
    mode: str
    conversation: str  # what the user typed: earlier turns, then this question (what the judge sees)
    standalone_question: str  # the planner's rewrite (agentic), else the question as asked
    expected_outcome: str
    observed_outcome: str
    outcome_correct: bool
    status: str
    answer: GroundedAnswer | None
    evidence: list[Evidence]
    cited_keys: list[str]
    evidence_keys: list[str]  # every passage the mode retrieved, in label order
    citation_recall: float | None  # None for unanswerable items
    evidence_recall: float | None
    verified: bool
    notes: list[str]
    llm_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cached: bool  # True: no API request was sent, so latency_ms is not a real latency
    error: str | None = None
    # filled in by the judge
    verdicts: list[str] = []
    faithfulness: float | None = None
    relevancy: int | None = None
    judge_model: str | None = None
    self_judged: bool = False  # the judge fell back to the generator's own model
    cross_check_verdicts: list[str] = []


def _keys(evidence: Sequence[Evidence]) -> list[str]:
    return [paragraph_key({"document": e.document, "scope": e.scope, "paragraph_id": e.paragraph_id}) for e in evidence]


def conversation_text(item: GoldenItem) -> str:
    turns = [*item.previous_turns, item.question]
    return "\n".join(f"User: {turn}" for turn in turns)


def score_turn(item: GoldenItem, mode: str, state: dict[str, Any], latency_ms: int, cached: bool) -> TurnResult:
    answer: GroundedAnswer | None = state.get("answer")
    evidence: list[Evidence] = state.get("evidence", [])
    status = answer.status if answer else "error"
    notes = state.get("notes", [])
    degraded = any("unavailable" in note for note in notes)  # an outage forced a fallback answer
    observed = observed_outcome(state.get("outcome", "answer"), status, degraded)
    cited_ids = {i for s in (answer.statements if answer else []) for i in s.evidence_ids}
    cited = _keys([e for e in evidence if e.id in cited_ids])
    retrieved = _keys(evidence)
    verifications = state.get("verifications", [])
    gold = item.expected_paragraphs
    plan = state.get("plan")
    return TurnResult(
        item_id=item.id,
        category=item.category,
        mode=mode,
        conversation=conversation_text(item),
        standalone_question=plan.standalone_question if plan else item.question,
        expected_outcome=item.expected_outcome,
        observed_outcome=observed,
        outcome_correct=observed == item.expected_outcome,
        status=status,
        answer=answer,
        evidence=evidence,
        cited_keys=cited,
        evidence_keys=retrieved,
        citation_recall=coverage(cited, gold) if gold else None,
        evidence_recall=coverage(retrieved, gold) if gold else None,
        verified=bool(verifications) and verifications[-1].passed,
        notes=notes,
        llm_calls=state.get("llm_calls", 0),
        input_tokens=state.get("input_tokens", 0),
        output_tokens=state.get("output_tokens", 0),
        latency_ms=latency_ms,
        cached=cached,
    )


def run_item(
    item: GoldenItem, mode: str, graph: Any, context: AgentContext, requests_sent: Callable[[], int]
) -> TurnResult:
    """`requests_sent` counts real API requests so far: unchanged = the turn was a cached replay."""
    config = {"configurable": {"thread_id": f"{item.id}:{mode}"}}
    try:
        for previous in item.previous_turns:  # the conversation so far (its cost isn't counted for this turn)
            graph.invoke(turn_input(previous, item.sector), config, context=context)
        requests_before = requests_sent()
        started = time.perf_counter()
        state = graph.invoke(turn_input(item.question, item.sector), config, context=context)
    except Exception as exc:  # one broken item must not stop the whole run
        result = score_turn(item, mode, {}, 0, cached=False)
        return result.model_copy(update={"error": f"{type(exc).__name__}: {exc}"})
    latency_ms = int((time.perf_counter() - started) * 1000)
    return score_turn(item, mode, dict(state), latency_ms, cached=requests_sent() == requests_before)


def evaluate_answers(
    items: Sequence[GoldenItem],
    modes: Sequence[str],
    context: AgentContext,
    requests_sent: Callable[[], int],
    on_turn: Callable[[TurnResult], None] = lambda result: None,
) -> list[TurnResult]:
    graphs = {mode: BUILDERS[mode](checkpointer=InMemorySaver(serde=state_serializer())) for mode in modes}
    results = []
    for item in items:
        for mode in modes:
            result = run_item(item, mode, graphs[mode], context, requests_sent)
            on_turn(result)
            results.append(result)
    return results
