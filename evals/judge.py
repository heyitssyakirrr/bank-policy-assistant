"""Our LLM judge: faithfulness and relevancy in one call per answer.

Our answers are already claims (statements, each with a quote and its cited
passages), so one call can check every statement. RAGAS would first extract
claims and then check each one: several calls per answer on a 500/day quota.

The judge sees the user's own conversation, never the system's rewrite of
it, so a wrong rewrite can't make an off-target answer look relevant. It is
gemini-3.1-flash-lite, a different model from the generator; a sample is
judged again by gemini-3.8-flash to measure how often two judges agree.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.evidence import Evidence, format_evidence
from app.agent.prompts import judge_messages
from app.agent.schemas import GroundedAnswer
from app.llm.models import LLMResult, StructuredLLM

Verdict = Literal["yes", "partly", "no"]


class StatementVerdict(BaseModel):
    statement: int = Field(description="The statement number, starting at 1")
    reason: str = Field(description="One sentence, written before the verdict")
    supported: Verdict


class AnswerJudgement(BaseModel):
    statements: list[StatementVerdict]
    relevancy_reason: str = Field(description="One sentence, written before the score")
    relevancy: int = Field(ge=1, le=5)


class JudgeMismatchError(ValueError):
    """The judge's statement numbers don't match the answer's: unusable, not 'unsupported'."""


def judge_answer(
    llm: StructuredLLM, conversation: str, answer: GroundedAnswer, evidence: Sequence[Evidence]
) -> LLMResult[AnswerJudgement]:
    by_id = {e.id: e for e in evidence}
    statements = [
        (s.obligation, s.text, s.quote, format_evidence([by_id[i] for i in s.evidence_ids if i in by_id]))
        for s in answer.statements
    ]
    return llm.invoke("judge", judge_messages(conversation, answer.summary, statements), AnswerJudgement)


def verdicts(judgement: AnswerJudgement, statement_count: int) -> list[Verdict]:
    """One verdict per statement, in order. The judge must number them 1..n;
    anything else (from 0, gaps, extras) is a judging failure, not a verdict."""
    by_number = {v.statement: v.supported for v in judgement.statements}
    if set(by_number) != set(range(1, statement_count + 1)):
        raise JudgeMismatchError(f"judged statements {sorted(by_number)}, expected 1..{statement_count}")
    return [by_number[number] for number in range(1, statement_count + 1)]


def agreement(first: Sequence[Verdict], second: Sequence[Verdict]) -> float | None:
    """The share of statements two judges gave the same verdict."""
    pairs = list(zip(first, second, strict=True))
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else None
