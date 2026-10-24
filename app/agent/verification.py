"""The verifier: code checks the generator's citations (PROJECT_PLAN §5.4).

| # | Check                                              | If it fails                    |
|---|----------------------------------------------------|--------------------------------|
| 1 | every cited ID is evidence the model was given     | drop the ID (fix)              |
| 2 | every statement still has >= 1 valid ID            | drop the statement (fix)       |
| 3 | the quote appears in one of its cited passages     | serious                        |
| 4 | a "must" statement doesn't cite only G paragraphs  | serious                        |
| 5 | status agrees with content                         | correct the status (fix)       |
| 6 | something valid is left                            | becomes not_covered (refusal)  |
| 7 | no cross-reference left out: a cited passage that  | incomplete                     |
|   | depends on another given passage ("as required     |                                |
|   | under paragraph 8.4") is answered with it          |                                |

Fixes are applied silently and recorded. Serious problems get one
regeneration with feedback (the graph decides that); if they persist,
`finalize` removes the bad statements and downgrades the status. An
incomplete answer also gets the one regeneration, but if it stays
incomplete it is kept as it is: nothing in it is wrong, only missing.

Check 4 flags only G: appendix paragraphs and tables carry no S/G marker
(e.g. the FAR thresholds, e-KYC Appendix 2 para 13), and they can state
obligations. Presenting *guidance* as an obligation is the error to catch.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.evidence import Evidence, cross_references
from app.agent.schemas import GroundedAnswer, Statement

NOT_COVERED_SUMMARY = (
    "The BNM policy documents I can search (e-KYC, FTFC and AML/CFT/CPF & TFS) don't cover this question."
)
PARTIAL_SUMMARY_PREFIX = "Only part of the answer could be verified against the documents: "
MIN_QUOTE_WORDS = 3
_ELLIPSIS_RE = re.compile(r"\.{3,}|…")
_WORD_RE = re.compile(r"\w+")
_EVIDENCE_ID_RE = re.compile(r"^\[?\s*[Ee]\s*(\d+)\s*\]?$")

CheckName = Literal[
    "unknown_id", "uncited", "quote_not_found", "must_cites_only_guidance", "status", "missing_cross_reference"
]


class Issue(BaseModel):
    check: CheckName
    # Index of the statement: for fixes, in the answer as generated; for
    # serious issues, in the corrected answer (VerificationReport.answer).
    statement: int | None
    detail: str


class VerificationReport(BaseModel):
    answer: GroundedAnswer  # after the fixable corrections
    fixes: list[Issue] = Field(default_factory=list)
    serious: list[Issue] = Field(default_factory=list)
    incomplete: list[Issue] = Field(default_factory=list)  # check 7: worth one regeneration, never a removal

    @property
    def passed(self) -> bool:
        return not self.serious

    @property
    def complete(self) -> bool:
        return self.passed and not self.incomplete


def normalise_id(raw: str) -> str | None:
    """'E3', '[E3]', 'e 3' -> 'E3'; anything else -> None."""
    match = _EVIDENCE_ID_RE.match(raw.strip())
    return f"E{int(match.group(1))}" if match else None


def _words(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def quote_found(quote: str, passage: str) -> bool:
    """The quote's words appear in the passage, ignoring case, punctuation
    and spacing. "..." may join fragments; each fragment needs at least
    MIN_QUOTE_WORDS words and they must appear in order, so a quote like
    "Board ... the ... shall" can't match almost any passage."""
    fragments = [f for f in (_words(f) for f in _ELLIPSIS_RE.split(quote)) if f]
    if not fragments or any(len(f.split()) < MIN_QUOTE_WORDS for f in fragments):
        return False
    haystack = f" {_words(passage)} "
    position = 0
    for fragment in fragments:
        found = haystack.find(f" {fragment} ", position)
        if found < 0:
            return False
        position = found + len(fragment) + 1
    return True


def _status_for(answer: GroundedAnswer, statements: list[Statement]) -> GroundedAnswer:
    """Checks 5 and 6: make the status agree with what is left."""
    if not statements:
        if answer.status == "not_covered":
            return answer.model_copy(update={"statements": []})
        return GroundedAnswer(status="not_covered", summary=NOT_COVERED_SUMMARY, statements=[], gaps=answer.gaps)
    status = "partially_answered" if answer.status == "not_covered" else answer.status
    return answer.model_copy(update={"status": status, "statements": statements})


def check_answer(answer: GroundedAnswer, evidence: Sequence[Evidence]) -> VerificationReport:
    by_id = {e.id: e for e in evidence}
    fixes: list[Issue] = []
    kept: list[Statement] = []

    for index, statement in enumerate(answer.statements):
        valid = []
        for raw in statement.evidence_ids:
            evidence_id = normalise_id(raw)
            if evidence_id in valid:
                continue  # the same passage cited twice
            if evidence_id in by_id:
                valid.append(evidence_id)
            else:
                fixes.append(Issue(check="unknown_id", statement=index, detail=f"dropped citation {raw!r}"))
        if not valid:
            fixes.append(Issue(check="uncited", statement=index, detail=f"dropped uncited statement: {statement.text}"))
            continue
        kept.append(statement.model_copy(update={"evidence_ids": valid}))

    checked = _status_for(answer, kept)
    if checked.status != answer.status:
        fixes.append(Issue(check="status", statement=None, detail=f"status {answer.status} -> {checked.status}"))

    serious: list[Issue] = []
    for index, statement in enumerate(checked.statements):
        cited = [by_id[i] for i in statement.evidence_ids]
        if not any(quote_found(statement.quote, e.text) for e in cited):
            serious.append(
                Issue(
                    check="quote_not_found",
                    statement=index,
                    detail=f"quote {statement.quote!r} is not in {', '.join(statement.evidence_ids)}",
                )
            )
        if statement.obligation == "must" and all(e.provision == "G" for e in cited):
            serious.append(
                Issue(
                    check="must_cites_only_guidance",
                    statement=index,
                    detail=f"'must' statement cites only guidance (G) paragraphs {', '.join(statement.evidence_ids)}",
                )
            )
    return VerificationReport(
        answer=checked, fixes=fixes, serious=serious, incomplete=_missing_cross_references(checked, evidence)
    )


def _missing_cross_references(answer: GroundedAnswer, evidence: Sequence[Evidence]) -> list[Issue]:
    """Check 7: e-KYC 8.10 (G) says "as required under paragraph 8.4"; if
    8.4 was given to the generator, an answer citing 8.10 must use it too."""
    cited = {i for s in answer.statements for i in s.evidence_ids}
    missing: list[Issue] = []
    for item in (e for e in evidence if e.id in cited):
        for ref in cross_references(item, evidence):
            if ref.id not in cited:
                detail = f"{item.id} depends on {ref.id} (para {ref.paragraph_id}): add what {ref.id} requires"
                missing.append(Issue(check="missing_cross_reference", statement=None, detail=detail))
    return missing


def finalize(report: VerificationReport) -> GroundedAnswer:
    """Remove statements with serious problems and downgrade the status.
    The model's summary may repeat a removed claim (e.g. guidance stated as
    an obligation), so it is replaced by the statements that survived."""
    if report.passed:
        return report.answer
    bad = {issue.statement for issue in report.serious}
    statements = [s for i, s in enumerate(report.answer.statements) if i not in bad]
    answer = _status_for(report.answer, statements)
    if not statements:
        return answer
    summary = PARTIAL_SUMMARY_PREFIX + " ".join(s.text for s in statements)
    return answer.model_copy(update={"status": "partially_answered", "summary": summary})


def regeneration_feedback(report: VerificationReport) -> str:
    statements = report.answer.statements
    lines = [f"- {statements[issue.statement].text!r}: {issue.detail}" for issue in report.serious]
    lines += [f"- {issue.detail}" for issue in report.incomplete]
    opening = "failed the citation check" if report.serious else "is incomplete"
    return (
        f"Your previous answer {opening}:\n" + "\n".join(lines) + "\nFix these: copy each quote word "
        "for word from the passage you cite, mark a statement 'must' only if an S paragraph or an unmarked "
        "passage states it as a requirement, and include every cross-referenced passage."
    )
