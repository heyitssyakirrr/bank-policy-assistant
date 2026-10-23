"""The reviewer: `review_evidence` judges whether the evidence answers the
question (Corrective RAG). The LLM decides; code sanitises:

- relevant IDs must be evidence that exists;
- follow-up searches are capped by the request budget (and at 2 per round);
- no further round after 2 reviews, or when the call/time budget can't pay for it;
- a cross-reference to a paragraph already in the evidence is dropped, and so
  is a follow-up search that only names paragraph numbers (the lookup does that);
- a "sufficient" verdict carries no follow-ups.

If the reviewer fails, all evidence counts as relevant and the graph goes on
to answer: a review is an improvement, never a reason to fail the question.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langgraph.runtime import Runtime

from app.agent.budget import MAX_REVIEW_ROUNDS
from app.agent.evidence import Evidence
from app.agent.planning import clean_paragraph_id
from app.agent.prompts import reviewer_messages
from app.agent.schemas import EvidenceReview, ParagraphRef
from app.agent.state import AgentContext, AgentState
from app.agent.verification import normalise_id
from app.llm.models import LLMUnavailableError

MAX_FOLLOW_UPS_PER_ROUND = 2
MAX_CROSS_REFERENCES = 3


def _new_cross_references(
    refs: list[ParagraphRef], evidence: Sequence[Evidence], relevant: Sequence[str]
) -> list[ParagraphRef]:
    """Cross-references not already in the evidence. A reference without a
    document gets the document of the relevant evidence when that is a
    single document: "paragraph 8.4" in an e-KYC passage means e-KYC 8.4."""
    present = {(e.document, e.paragraph_id) for e in evidence}
    relevant_documents = {e.document for e in evidence if e.id in relevant}
    default_document = next(iter(relevant_documents)) if len(relevant_documents) == 1 else None
    kept: dict[tuple[str | None, str], ParagraphRef] = {}
    for ref in refs:
        paragraph_id = clean_paragraph_id(ref.paragraph_id)
        document = ref.document or default_document
        if paragraph_id is None or (document, paragraph_id) in present:
            continue
        kept.setdefault((document, paragraph_id), ParagraphRef(document=document, paragraph_id=paragraph_id))
    return list(kept.values())


def _only_paragraph_numbers(query: str) -> bool:
    """'paragraph 8.4' as a search duplicates a cross-reference lookup."""
    words = [w for w in query.replace(",", " ").split() if w.lower() not in ("para", "paragraph", "paragraphs", "and")]
    return bool(words) and all(clean_paragraph_id(w) for w in words)


def sanitise_review(review: EvidenceReview, evidence: Sequence[Evidence], searches_left: int) -> EvidenceReview:
    known = {e.id for e in evidence}
    relevant = list(dict.fromkeys(i for i in map(normalise_id, review.relevant_evidence) if i in known))
    if review.is_sufficient:  # "sufficient" with nothing relevant is contradictory: keep all evidence
        return EvidenceReview(relevant_evidence=relevant or [e.id for e in evidence], is_sufficient=True)
    follow_ups = [s for s in review.follow_up_searches if s.query.strip() and not _only_paragraph_numbers(s.query)][
        : min(MAX_FOLLOW_UPS_PER_ROUND, searches_left)
    ]
    # All cross-references go in one lookup: one search, plus one fallback
    # search if a paragraph isn't found. Both must fit in the budget.
    room_for_lookup = searches_left - len(follow_ups) >= 2
    cross_references = _new_cross_references(review.cross_references, evidence, relevant) if room_for_lookup else []
    return EvidenceReview(
        relevant_evidence=relevant,
        is_sufficient=False,
        missing_information=review.missing_information,
        follow_up_searches=follow_ups,
        cross_references=cross_references[:MAX_CROSS_REFERENCES],
    )


def _stop_reason(state: AgentState, rounds: int, calls_after: int, context: AgentContext) -> str | None:
    """Why another round is not allowed, or None if it is. Another round needs
    one more review and the generator: 2 LLM calls."""
    if rounds >= MAX_REVIEW_ROUNDS:
        return "review round limit"
    if context.budget.max_llm_calls - calls_after < 2:
        return "LLM call budget"
    if context.budget.out_of_time(state, context.clock()):
        return "time budget"
    return None


def review_evidence(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    evidence = state.get("evidence", [])
    rounds = state.get("rounds", 0) + 1
    if not evidence:  # nothing to review: the generator will refuse without an LLM call
        return {"rounds": rounds, "review": EvidenceReview(relevant_evidence=[], is_sufficient=False)}

    context = runtime.context
    searched = [record["query"] for record in state.get("search_log", [])]
    messages = reviewer_messages(state["plan"].standalone_question, evidence, state.get("sector"), searched)
    try:
        result = context.llm.invoke("reviewer", messages, EvidenceReview)
    except LLMUnavailableError as exc:
        review = EvidenceReview(relevant_evidence=[e.id for e in evidence], is_sufficient=True)
        return {
            "rounds": rounds,
            "review": review,
            "notes": ["Reviewer unavailable; answered from all evidence."],
            "llm_calls": exc.calls,
        }

    review = sanitise_review(result.output, evidence, context.budget.searches_left(state))
    notes = []
    wants_more = review.follow_up_searches or review.cross_references
    reason = _stop_reason(state, rounds, state.get("llm_calls", 0) + result.calls, context)
    if wants_more and reason:
        review = review.model_copy(update={"follow_up_searches": [], "cross_references": []})
        notes.append(f"Stopped searching ({reason}); answered with the evidence found.")
    return {
        "rounds": rounds,
        "review": review,
        "notes": notes,
        "models": [result.model],
        "llm_calls": result.calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }
