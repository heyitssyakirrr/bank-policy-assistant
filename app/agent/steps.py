"""What each graph step did, in one line of plain English.

The graph streams one update per node (`stream_mode="updates"`). This turns
each update into the line the user sees in "Agent steps", e.g.
"Searched e-KYC for 'biometric verification': 5 passages". One function,
used by scripts/ask.py, the API's SSE stream and the UI.
"""

from __future__ import annotations

from typing import Any

SHORT_TITLES = {"ekyc": "e-KYC", "ftfc": "FTFC", "amlcft": "AML/CFT"}
# The step timeline's label for each node; nodes without one (finish) aren't shown.
STEP_TITLES = {
    "understand_question": "Plan",
    "retrieve": "Search",
    "search_policies": "Search",
    "lookup_paragraphs": "Look up",
    "review_evidence": "Review",
    "generate_answer": "Write",
    "verify_answer": "Verify",
    "respond_directly": "Respond",
    "ask_clarification": "Clarify",
}


def _documents(codes: list[str]) -> str:
    return " + ".join(SHORT_TITLES[c] for c in codes) if codes else "all documents"


def _paragraph(ref: Any) -> str:
    return f"{SHORT_TITLES[ref.document]} para {ref.paragraph_id}" if ref.document else f"para {ref.paragraph_id}"


def _search_line(record: dict[str, Any]) -> str:
    if "error" in record:
        return f"The {record['kind']} for '{record['query']}' failed ({record['error']})"
    if record["kind"] == "lookup":
        missing = f"; not found: {', '.join(record['missing'])}" if record["missing"] else ""
        return f"Looked up {record['query']}: {len(record['chunks'])} passages{missing}"
    fallbacks = "" if record["reranked"] else " (not reranked)"
    fallbacks += "" if record["dense_used"] else " (keyword search only)"
    return f"Searched {_documents(record['documents'])} for '{record['query']}': {len(record['chunks'])} passages{fallbacks}"


def describe_step(node: str, update: dict[str, Any]) -> list[str]:
    """The lines for one node's update (a tool branch may report several searches)."""
    lines: list[str]
    match node:
        case "understand_question":
            plan = update["plan"]
            lines = [f"Understanding question: {plan.reasoning or plan.route}"]
            if len(plan.searches) > 1:
                lines.append(f"Split into {len(plan.searches)} searches")
        case "search_policies" | "lookup_paragraphs" | "retrieve":
            lines = [_search_line(record) for record in update.get("search_log", [])]
        case "review_evidence":
            review = update["review"]
            if not review.relevant_evidence and not review.is_sufficient and not review.missing_information:
                lines = ["Reviewed evidence: no passages found"]
            elif review.is_sufficient:
                lines = [f"Reviewed evidence: {len(review.relevant_evidence)} relevant passages, enough to answer"]
            else:
                lines = [f"Reviewed evidence: missing {review.missing_information or 'information'}"]
                lines += [f"Following cross-reference {_paragraph(r)}" for r in review.cross_references]
                lines += [f"Searching again: '{s.query}'" for s in review.follow_up_searches]
        case "generate_answer":
            answer = update["answer"]
            lines = [f"Writing answer: {answer.status.replace('_', ' ')}, {len(answer.statements)} statements"]
        case "verify_answer":
            report = update["verifications"][-1]
            if update.get("feedback") and report.passed:
                lines = [f"Answer misses a cross-referenced passage ({report.incomplete[0].detail}): regenerating"]
            elif update.get("feedback"):
                lines = [f"Verification found {len(report.serious)} problem(s): regenerating"]
            elif report.passed:
                lines = ["Citations verified"]
            else:
                lines = [f"Verification found {len(report.serious)} problem(s): unverified statements removed"]
        case "respond_directly":
            lines = [
                "Outside the policy documents: no search needed" if update["outcome"] == "out_of_scope" else "Greeting"
            ]
        case "ask_clarification":
            lines = ["Asking a clarifying question"]
        case _:
            lines = []
    return lines + [f"Note: {note}" for note in update.get("notes", [])]
